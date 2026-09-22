# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globbed all NWB files under `/app/data/sub-*/*.nwb`, but hard-excluded one session by filename before processing. Each remaining file is opened directly with `h5py` and read from HDF5 paths such as `intervals/trials`, `units`, and `acquisition/BehavioralEvents`.

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
    trial_start_times = np.asarray(h5file["intervals"]["trials"]["start_time"][()], dtype=np.float64)
```

iii. In `CONVERSION_NOTES.md` and the trajectory, the AI justified this as direct NWB access and said the named excluded session was the one session that lacked classifier-good units and explained the 174-file versus 173-session discrepancy.

## 1-b. How are the data split into subjects?

i. The AI reads `general/subject/subject_id` from each NWB file, with a fallback to the parent `sub-*` folder name if that field cannot be read. During assembly it builds `subjects` in first-seen order and maps each session to `subject_idx`.

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

iii. The justification is implicit in the code and metadata: the subject ID stored in the NWB file is treated as the canonical mouse identifier, and the folder-name fallback is used only as a defensive backup.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Session identity is taken from the filename stem rather than `nwb.identifier`, and session order is the sorted filesystem order after removing the hardcoded excluded session.

ii. 
```python
def session_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    for suffix in (".nwb",):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base
```

```python
all_paths = sorted(glob.glob("/app/data/sub-*/*.nwb"))
```

iii. In the trajectory and notes, the AI described the dataset as “one NWB file per session” and used the 173 kept files as the retained session list.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table and event arrays, but instead of asserting equal lengths it truncates all trial-aligned arrays to `ntrials = min(...)` across the trials table, behavioral labels, photostim columns, and go-cue timestamps. It then filters that shared prefix to get retained trials.

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
trial_start_times = trial_start_times[:ntrials]
...
go_times_abs = go_times_abs[:ntrials]
```

iii. The AI does not give a strong explicit justification in the notes; this looks like defensive handling for possible length mismatches rather than a reference-derived trial-definition rule.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials that are not `auto_water`, not `free_water`, have a recoverable tone onset, are marked as covered by `units/is_good_trials` for all retained good units, overlap the global spike-time span of the retained units, and still contain some nonzero neural activity after binning. Sessions with fewer than two retained trials are dropped.

ii. 
```python
is_good_trials = np.asarray(h5file["units"]["is_good_trials"][()], dtype=bool)
neural_coverage_mask = build_common_neural_trial_mask(
    is_good_trials=is_good_trials,
    good_unit_indices=good_unit_indices,
    ntrials=ntrials,
)
spike_support_mask = (
    np.isfinite(spike_span_min)
    & np.isfinite(spike_span_max)
    & ((go_times_abs + WINDOW_END_S) > spike_span_min)
    & ((go_times_abs + WINDOW_START_S) < spike_span_max)
)
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
if n_zero_neural:
    neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_neural_mask) if keep]
```

iii. The notes and trajectory explicitly justify these filters as needed to avoid trials outside neural coverage and to remove `auto_water`/`free_water` trials while retaining `early_lick`, `ignore`, and photostim trials because those are required decoder variables.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units/spike_times`, `units/spike_times_index`, and the subset of units with `units/classification == "good"`. Go-cue timestamps define the per-trial alignment windows.

ii. 
```python
classification = decode_string_array(h5file["units"]["classification"])
good_unit_indices = np.flatnonzero(classification == "good")
spike_times_all = np.asarray(h5file["units"]["spike_times"][()], dtype=np.float64)
spike_times_index = np.asarray(h5file["units"]["spike_times_index"][()], dtype=np.int64)
go_times_abs = np.asarray(
    h5file["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][()],
    dtype=np.float64,
)
```

iii. The AI’s notes say the NWB export already contains the classifier-QC field needed to recover the paper-style good-unit set, so spike times from those units are used directly.

## 2-b. How is the `neural` data processed?

i. The AI bins each retained good unit’s spikes into non-overlapping 50 ms bins from -2.5 s to +1.5 s around the go cue, computes spike counts with `np.searchsorted`, converts counts to firing rates by dividing by bin width, and stores the result as `float16`.

ii. 
```python
BIN_SIZE_S = 0.05
BIN_EDGES_REL = WINDOW_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
```

```python
bin_edges_abs = go_times_abs[:, None] + BIN_EDGES_REL[None, :]
for out_idx, unit_idx in enumerate(good_unit_indices):
    start = 0 if unit_idx == 0 else int(spike_times_index[unit_idx - 1])
    stop = int(spike_times_index[unit_idx])
    spikes = spike_times_all[start:stop]
    insertion_idx = np.searchsorted(spikes, bin_edges_abs, side="left")
    counts = np.diff(insertion_idx, axis=1)
    neural_stack[out_idx] = (counts.astype(np.float32) / BIN_SIZE_S).astype(np.float16)
```

iii. In the trajectory, the AI said it would “compute firing rates from spike counts in non-overlapping 50 ms bins” and use `float16` to reduce memory pressure for full-dataset decoder training.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == "good"` are retained. Sessions with no such units are skipped.

ii. 
```python
classification = decode_string_array(h5file["units"]["classification"])
good_unit_indices = np.flatnonzero(classification == "good")
if len(good_unit_indices) == 0:
    print(f"[session] {session_name}: skipping (0 classifier-good units)", flush=True)
    return None
```

iii. The notes explicitly state that `units/classification` was identified as the NWB version of the classifier-QC output used by the paper/reference pipeline.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go-cue onset by adding a fixed go-cue-relative edge vector to each trial’s absolute go-cue timestamp.

ii. 
```python
BIN_EDGES_REL = WINDOW_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
...
bin_edges_abs = go_times_abs[:, None] + BIN_EDGES_REL[None, :]
```

iii. The justification in the notes is that all NWB timestamps are already on a common session clock, so alignment is done by direct timestamp arithmetic around the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins, producing 80 time bins from -2.5 s to +1.5 s. The AI does not apply any further temporal smoothing or rebinning.

ii. 
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
N_BINS = int(round((WINDOW_END_S - WINDOW_START_S) / BIN_SIZE_S))
```

iii. The notes explain that the reference papers use different neural bins for a different analysis, but the decoder task here explicitly requires 50 ms bins in the narrower go-cue-aligned window.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from trial start times, go-cue times, and `sample_start_times`. It finds the latest sample-start event that falls after trial start and before the trial’s go cue.

ii. 
```python
def build_tone_onsets_abs(
    trial_start_times: np.ndarray,
    go_times_abs: np.ndarray,
    sample_start_times_abs: np.ndarray,
) -> np.ndarray:
    start_idx = np.searchsorted(sample_start_times_abs, trial_start_times, side="left")
    end_idx = np.searchsorted(sample_start_times_abs, go_times_abs, side="left")
    tone_onsets_abs = np.full(go_times_abs.shape, np.nan, dtype=np.float64)
    for trial_idx in range(len(go_times_abs)):
        if end_idx[trial_idx] > start_idx[trial_idx]:
            tone_onsets_abs[trial_idx] = sample_start_times_abs[end_idx[trial_idx] - 1]
```

iii. The AI justified this in the trajectory by noting that `sample_start_times` can contain replayed events after early licks, so the correct per-trial tone is the last one before the go cue, constrained to the trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After deriving the per-trial tone timestamp, the AI computes each bin’s value as the bin center relative to go cue minus the tone’s relative time to go cue, yielding seconds since tone onset at each neural bin.

ii. 
```python
go_keep = go_times_abs[kept_trial_indices]
tone_rel_s = tone_onsets_abs[kept_trial_indices] - go_keep
...
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel_s[:, None]
```

iii. The notes describe this as go-cue-centered bins re-expressed on a “time since tone” axis.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed directly on the same 80 bin centers used for neural firing rates, so it is aligned bin-for-bin with the neural data.

ii. 
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0
...
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel_s[:, None]
inputs_2d = np.stack([time_from_tone, photostim_binary], axis=1).astype(np.float32)
```

iii. The AI’s stated goal in the notes was to align all streams to go cue on the same 50 ms grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the trial-table columns `photostim_onset` and `photostim_duration`, together with trial start time and go-cue time.

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

iii. The trajectory says the AI validated that these NWB trial-table onsets are stored relative to trial start and occur at the expected delay-period timing relative to go cue.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts trial-relative onset/duration into go-cue-relative onset/offset, then marks a bin as `1` if any part of the 50 ms bin overlaps the stimulation interval, otherwise `0`.

ii. 
```python
def build_photostim_binary(
    stim_on_rel_s: np.ndarray,
    stim_off_rel_s: np.ndarray,
) -> np.ndarray:
    bin_starts = BIN_EDGES_REL[:-1][None, :]
    bin_ends = BIN_EDGES_REL[1:][None, :]
    photostim = np.zeros((len(stim_on_rel_s), N_BINS), dtype=np.float32)
    valid = np.isfinite(stim_on_rel_s) & np.isfinite(stim_off_rel_s)
    if np.any(valid):
        on = stim_on_rel_s[valid][:, None]
        off = stim_off_rel_s[valid][:, None]
        photostim[valid] = ((bin_starts < off) & (bin_ends > on)).astype(np.float32)
```

iii. The notes justify this as a time-varying binary stimulus input rather than a per-trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI expresses stimulation onset and offset relative to the go cue and compares them against the same bin edges used for neural binning.

ii. 
```python
stim_on_rel_s[valid_stim] = (
    trial_start_keep[valid_stim] + photostim_onset_keep[valid_stim] - go_keep[valid_stim]
)
stim_off_rel_s = stim_on_rel_s + photostim_duration_keep
photostim_binary = build_photostim_binary(stim_on_rel_s, stim_off_rel_s)
```

iii. The notes and trajectory say all streams are explicitly placed onto the go-cue-aligned neural time axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the event streams `left_lick_times` and `right_lick_times`, not from `trial_instruction` and `outcome`. It looks for the first lick in the 1.5 s response window after go cue.

ii. 
```python
left_lick_times_abs = np.asarray(
    h5file["acquisition"]["BehavioralEvents"]["left_lick_times"]["timestamps"][()],
    dtype=np.float64,
)
right_lick_times_abs = np.asarray(
    h5file["acquisition"]["BehavioralEvents"]["right_lick_times"]["timestamps"][()],
    dtype=np.float64,
)
```

```python
def reconstruct_choice_labels(
    go_times_abs: np.ndarray,
    left_lick_times_abs: np.ndarray,
    right_lick_times_abs: np.ndarray,
) -> np.ndarray:
    """Use the first lick in the 1.5 s response window after go cue."""
```

iii. The notes explicitly justify this as using “actual behavior instead of inferred correctness,” and report >99.6% agreement with the instruction-plus-outcome derivation.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each retained trial, the AI finds the first post-go left and right lick inside `[go, go+1.5)`, chooses whichever occurs earlier, assigns `no lick` if neither occurs, maps labels to integers `0/1/2`, and repeats that class across all 80 bins.

ii. 
```python
if left_time == np.inf and right_time == np.inf:
    labels[trial_idx] = "no lick"
elif left_time <= right_time:
    labels[trial_idx] = "left"
else:
    labels[trial_idx] = "right"
```

```python
trial_choice_int = np.asarray([CHOICE_TO_INT[x] for x in choice_labels], dtype=np.int8)
outputs_2d[:, 0, :] = trial_choice_int[:, None]
```

iii. The AI justified the first-lick rule as the most direct behavioral readout of the requested choice label.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the NWB trial-table `outcome` column.

ii. 
```python
outcome = decode_string_array(h5file["intervals"]["trials"]["outcome"])
...
outcome_keep = outcome[kept_trial_indices]
```

iii. The notes describe this as a direct mapping from the NWB trial outcome string.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore`, `miss`, and `hit` to integers `0`, `1`, and `2`, then repeats the per-trial value across all time bins.

ii. 
```python
OUTCOME_TO_INT = {"ignore": 0, "miss": 1, "hit": 2}
trial_outcome_int = np.asarray([OUTCOME_TO_INT[x] for x in outcome_keep], dtype=np.int8)
outputs_2d[:, 1, :] = trial_outcome_int[:, None]
```

iii. The justification is that outcome already exists in the required categorical form.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the NWB trial-table `early_lick` column.

ii. 
```python
early_lick = decode_string_array(h5file["intervals"]["trials"]["early_lick"])
...
early_lick_keep = early_lick[kept_trial_indices]
```

iii. The notes describe this as a direct NWB label that the decoder task explicitly requires preserving.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early` to `0` and `early` to `1`, then repeats the per-trial label across all 80 bins.

ii. 
```python
EARLY_TO_INT = {"no early": 0, "early": 1}
trial_early_int = np.asarray([EARLY_TO_INT[x] for x in early_lick_keep], dtype=np.int8)
outputs_2d[:, 2, :] = trial_early_int[:, None]
```

iii. The notes justify this as a direct categorical decoder output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using column 1 as `tongue_y` and column 2 as `tongue_likelihood`, together with its timestamps.

ii. 
```python
tongue_data = np.asarray(
    h5file["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]["data"][()],
    dtype=np.float32,
)
tongue_timestamps_abs = np.asarray(
    h5file["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]["timestamps"][()],
    dtype=np.float64,
)
tongue_y_raw = tongue_data[:, 1]
tongue_likelihood_raw = tongue_data[:, 2]
```

iii. The notes say the side-camera tongue tracker is the relevant raw tongue signal available in every session.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first defines globally visible frames as those with finite `tongue_y` and `tongue_likelihood >= 0.9`. It computes the 40th and 60th percentiles from the raw visible-frame `tongue_y` values over the full session. For each trial/bin it samples the last tongue frame before the bin end (if that frame still falls inside the bin), checks its likelihood against the same threshold, and maps it into one of four classes.

ii. 
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
...
visible_global = (
    np.isfinite(tongue_y_raw)
    & np.isfinite(tongue_likelihood_raw)
    & (tongue_likelihood_raw >= TONGUE_LIKELIHOOD_THRESHOLD)
)
...
tongue_q40, tongue_q60 = np.quantile(tongue_y_raw[visible_global], [0.4, 0.6])
```

```python
last_idx = np.searchsorted(tongue_timestamps_abs, bin_ends_abs, side="left") - 1
...
sampled_y.ravel()[flat_positions] = tongue_y[chosen]
sampled_likelihood.ravel()[flat_positions] = tongue_likelihood[chosen]
```

iii. The trajectory and notes justify this by saying tongue likelihood is strongly bimodal and that a high-confidence threshold plus sessionwise visible-frame percentiles is a defensible way to create the required categories.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Category `3` means not visible. Among visible bins, the AI assigns `0` if sampled `y < q40`, `1` if `q40 <= y <= q60`, and `2` if `y > q60`, where `q40` and `q60` are the 40th and 60th percentiles of visible raw frames over the session.

ii. 
```python
classes = np.full(bin_starts_abs.shape, 3, dtype=np.int8)
visible = (
    np.isfinite(sampled_y)
    & np.isfinite(sampled_likelihood)
    & (sampled_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
)
classes[visible & (sampled_y < visible_q40)] = 0
classes[visible & (sampled_y >= visible_q40) & (sampled_y <= visible_q60)] = 1
classes[visible & (sampled_y > visible_q60)] = 2
```

iii. The AI’s justification is the same as in 8-b: visible-frame percentiles define the within-session categories and low-confidence bins become `not visible`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue data to the neural grid by building the same go-cue-relative 50 ms bin boundaries in absolute time for each trial, then assigning each bin the last tongue frame observed before the bin end if that frame still lies within the bin.

ii. 
```python
bin_starts_abs = go_times_abs[:, None] + BIN_EDGES_REL[:-1][None, :]
bin_ends_abs = go_times_abs[:, None] + BIN_EDGES_REL[1:][None, :]
last_idx = np.searchsorted(tongue_timestamps_abs, bin_ends_abs, side="left") - 1
...
keep = last_times >= flat_starts
```

iii. The notes emphasize that all streams share the same absolute clock and are aligned to the same go-cue-centered 50 ms bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses several defensive fallbacks: non-byte/non-string entries are stringified by `decode_string_array`; subject ID falls back to the folder name if missing; trial-aligned arrays are silently truncated to a common minimum length; missing tone onsets cause trial exclusion; zero-good-unit sessions are dropped; and tongue bins with no usable high-confidence sample remain in the `not visible` class.

ii. 
```python
def decode_string_array(dataset) -> np.ndarray:
    values = dataset[()]
    out = []
    for item in values:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
```

```python
ntrials = min(
    len(trial_start_times),
    ...
    len(go_times_abs),
)
```

```python
if end_idx[trial_idx] > start_idx[trial_idx]:
    tone_onsets_abs[trial_idx] = sample_start_times_abs[end_idx[trial_idx] - 1]
...
classes = np.full(bin_starts_abs.shape, 3, dtype=np.int8)
```

iii. The trajectory shows the AI added these checks after encountering sessions with incomplete neural coverage and edge-case NWB layouts; the notes frame them as guardrails against fabricating invalid decoder examples.

## 10-a. What are the most time-consuming steps of the code?

i. The code is dominated by per-session NWB I/O, loading the large `spike_times` and tongue-tracking arrays, and the per-unit spike binning loop. If optional plotting is enabled, figure generation adds extra cost.

ii. 
```python
with h5py.File(path, "r") as h5file:
    ...
    spike_times_all = np.asarray(h5file["units"]["spike_times"][()], dtype=np.float64)
    ...
    tongue_data = np.asarray(
        h5file["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]["data"][()],
        dtype=np.float32,
    )
```

```python
for out_idx, unit_idx in enumerate(good_unit_indices):
    ...
    insertion_idx = np.searchsorted(spikes, bin_edges_abs, side="left")
```

iii. The AI’s notes say the full conversion finished in about 2.1 minutes and discuss memory/runtime pressure as the reason for using compact dtypes and adding progress logging.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops remain: per-trial tone reconstruction, per-trial first-lick choice reconstruction, per-good-unit spike-span reconstruction, per-good-unit neural binning, and per-file session processing. The tone and choice loops are especially vectorizable relative to the reference solution.

ii. 
```python
for trial_idx in range(len(go_times_abs)):
    if end_idx[trial_idx] > start_idx[trial_idx]:
        tone_onsets_abs[trial_idx] = sample_start_times_abs[end_idx[trial_idx] - 1]
```

```python
for trial_idx, go_time in enumerate(go_times_abs):
    ...
```

```python
for out_idx, unit_idx in enumerate(good_unit_indices):
    ...
```

iii. There is no explicit optimization justification in the notes beyond making the full run feasible; the retained loops reflect implementation convenience more than a deliberate “do not vectorize” decision.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats trial masking/copying after neural binning when it removes all-zero neural windows, and it keeps parallel diagnostic copies of many arrays for notes/plots/metadata. It also computes both pre-binning coverage filters and a post-binning zero-neural filter.

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
if n_zero_neural:
    neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_neural_mask) if keep]
    inputs_2d = inputs_2d[nonzero_neural_mask]
    outputs_2d = outputs_2d[nonzero_neural_mask]
    go_keep = go_keep[nonzero_neural_mask]
    ...
```

iii. The trajectory explains that the post-binning filter was added after validation exposed remaining all-zero trials that earlier coverage checks did not remove.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes and retains substantial diagnostic state that is not needed for the final exported dataset: `classification_counts`, exclusion counts, raw go/tone/stim arrays, raw tongue arrays, label distributions, and plotting helpers. Some loaded/derived values are not used by the final export at all, such as `trial_stop_times` and `trial_instruction_keep`.

ii. 
```python
trial_stop_times = np.asarray(h5file["intervals"]["trials"]["stop_time"][()], dtype=np.float64)
...
trial_instruction_keep = trial_instruction[kept_trial_indices]
```

```python
return {
    "classification_counts": classification_counts,
    "n_trials_total": int(ntrials),
    ...
    "go_times_abs": go_keep,
    "tone_rel_s": tone_rel_s,
    "stim_on_rel_s": stim_on_rel_s,
    "stim_off_rel_s": stim_off_rel_s,
    "tongue_timestamps_abs": tongue_timestamps_abs,
    "tongue_y_raw": tongue_y_raw,
    "tongue_likelihood_raw": tongue_likelihood_raw,
```

iii. The notes and README show that these extras were kept to support sanity checks, visualizations, and narrative documentation rather than the decoder-ready output itself.
