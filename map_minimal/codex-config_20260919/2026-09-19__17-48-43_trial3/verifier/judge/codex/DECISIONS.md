# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by globbing `sub-*/*.nwb` under the data directory, sorting the paths, and opening each file twice with `h5py`: once in `scan_vocab()` to build subject and region vocabularies and drop sessions with no good units, and once in `convert_session()` to do the actual conversion. It intentionally uses raw HDF5 access rather than `pynwb`.

ii. 
```python
def scan_vocab(paths: list[Path]) -> tuple[list[str], list[str], list[Path]]:
    """Find stable subject/region vocabularies and discard empty sessions."""
    subjects: set[str] = set()
    regions: set[str] = set()
    usable: list[Path] = []
    for path in paths:
        with h5py.File(path, "r") as nwb:
            classification = decode_strings(nwb["units/classification"][:])
            good = classification == "good"
            if np.count_nonzero(good) == 0:
                print(f"Skipping {path.name}: no classifier-approved units", flush=True)
                continue
            subjects.add(
                nwb["general/subject/subject_id"][()].decode("utf-8").strip()
            )
            annotations = decode_strings(nwb["units/anno_name"][:])[good]
            regions.update(annotations.tolist())
            usable.append(path)
    return sorted(subjects), sorted(regions), usable
```

```python
def convert(data_dir: Path, output_path: Path) -> dict:
    paths = sorted(data_dir.glob("sub-*/*.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found below {data_dir}")
    subjects, brain_regions, paths = scan_vocab(paths)
    ...
    for session, path in enumerate(paths, start=1):
        converted = convert_session(path, subject_to_idx, region_to_idx)
```

```python
with h5py.File(path, "r") as nwb:
    trials = nwb["intervals/trials"]
    events = nwb["acquisition/BehavioralEvents"]
    units = nwb["units"]
```

iii. In the script header, the AI says it "intentionally uses h5py rather than pynwb" because the archived files use an older NWB schema while the needed fields are still standard HDF5 datasets. In the trajectory it also notes that the dataset is one NWB file per session, so sorting the globbed file list gives a deterministic pass over the whole release.

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB field `general/subject/subject_id`. The AI gathers the unique subject IDs across all usable sessions, sorts them, and stores per-session `subject_idx` values into that sorted list.

ii. 
```python
subjects.add(
    nwb["general/subject/subject_id"][()].decode("utf-8").strip()
)
```

```python
subjects, brain_regions, paths = scan_vocab(paths)
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
...
subject = nwb["general/subject/subject_id"][()].decode("utf-8").strip()
...
return (
    neural_trials,
    input_trials,
    output_trials,
    subject_to_idx[subject],
    region_idx,
    session_info,
)
```

iii. The trajectory does not show a long separate argument here; the AI simply treats the NWB subject field as the canonical source of subject identity, consistent with how it inspected the files early in the run.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Session order follows the sorted path list. Session identity is read from `identifier` and written into `metadata["session_info"]`.

ii. 
```python
paths = sorted(data_dir.glob("sub-*/*.nwb"))
```

```python
identifier = nwb["identifier"][()].decode("utf-8").strip()
session_info = {
    "identifier": identifier,
    "source_file": str(path),
    ...
}
```

```python
for session, path in enumerate(paths, start=1):
    converted = convert_session(path, subject_to_idx, region_to_idx)
```

iii. In the trajectory the AI consistently talks about "174 session files" and later "173 sessions" after quality filtering, so it clearly used file boundaries as session boundaries.

## 1-d. How are the data split into trials?

i. The AI starts from the NWB trials table and the go-cue event series. It checks that `len(go_start_times) == len(trials["id"])`, then truncates to the first `n_ephys_trials = units["is_good_trials"].shape[1]` trials, assuming later behavioral trials are outside the electrophysiology recording. It does not match trial rows to `obs_intervals` start times.

ii. 
```python
n_behavior_trials = len(trials["id"])
all_go_times = events["go_start_times/timestamps"][:]
if len(all_go_times) != n_behavior_trials:
    raise ValueError(
        f"{path}: {n_behavior_trials} trials but {len(all_go_times)} go cues"
    )
```

```python
n_ephys_trials = units["is_good_trials"].shape[1]
if n_ephys_trials > n_behavior_trials:
    raise ValueError(f"{path}: more ephys trials than behavioral trials")
source_trial_idx = np.arange(n_ephys_trials, dtype=np.int64)
go_times = all_go_times[source_trial_idx]
```

iii. After validation exposed many zero-neural trials, the AI states in trajectory step 42 that "several trial tables extend beyond the electrophysiology recording" and that `obs_intervals`/`is_good_trials` encode the actual recorded trial span. Its fix was to retain only the ephys-covered prefix of the trial table.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials in two stages. First, it removes behavioral-only tail trials by keeping only the first `n_ephys_trials`. Second, after neural binning, it removes any remaining trials whose neural tensor is all zeros across all good units and time bins. It does not exclude `free_water` trials, and it keeps early-lick, ignore, and photostimulation trials if they have neural data. Sessions with fewer than two remaining trials are skipped.

ii. 
```python
n_ephys_trials = units["is_good_trials"].shape[1]
source_trial_idx = np.arange(n_ephys_trials, dtype=np.int64)
go_times = all_go_times[source_trial_idx]
```

```python
has_neural_data = np.any(rates != 0, axis=(0, 2))
n_zero_neural = int(np.count_nonzero(~has_neural_data))
source_trial_idx = source_trial_idx[has_neural_data]
go_times = go_times[has_neural_data]
rates = rates[:, has_neural_data, :]
n_trials = len(source_trial_idx)
```

```python
if len(n) < 2:
    print(f"Skipping {path.name}: fewer than two trials", flush=True)
    continue
```

iii. The script header says "Every trial with electrophysiology is retained, except acquisition gaps with no spikes from any unit" and explicitly argues that excluding early-lick, ignore, free-water, and photostimulation trials would remove requested labels or inputs. In trajectory step 42, the AI adds that behavioral-only tails must be removed because they create artificial all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units/spike_times` and `units/spike_times_index`, restricted to units whose `units/classification` equals `"good"`. Trial go-cue timestamps provide the alignment windows.

ii. 
```python
classification = decode_strings(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
...
rates = bin_good_units(units, good_indices, go_times)
```

```python
all_spikes = units["spike_times"]
ends = units["spike_times_index"][:].astype(np.int64, copy=False)
starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
window_starts = go_times + OFF_START_S
window_ends = go_times + OFF_END_S
```

iii. In the trajectory the AI says the repository confirmed that "classifier-QC 'good' units are the paper’s analysis population" and that spikes are aligned by subtracting each trial’s go-cue time.

## 2-b. How is the `neural` data processed?

i. For each good unit, the AI bins absolute spike times into non-overlapping 50 ms bins from -2.5 s to +1.5 s around the go cue, assigns spikes to trials using the fact that trial windows do not overlap, and adds `1 / BIN_SIZE_S` per spike so the final values are firing rates in Hz. It applies no smoothing, baseline subtraction, or normalization.

ii. 
```python
BIN_SIZE_S = 0.050
OFF_START_S = -2.5
OFF_END_S = 1.5
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
BIN_EDGES = OFF_START_S + np.arange(N_TIME + 1, dtype=np.float64) * BIN_SIZE_S
```

```python
for output_unit, source_unit in enumerate(good_indices):
    spikes = all_spikes[starts[source_unit]:ends[source_unit]]
    if len(spikes) == 0:
        continue
    trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
    ...
    relative = candidate_spikes - go_times[candidate_trials]
    bin_idx = np.searchsorted(BIN_EDGES, relative, side="right") - 1
    ...
    np.add.at(
        rates[output_unit],
        (candidate_trials[valid_bin], bin_idx[valid_bin]),
        np.float32(1.0 / BIN_SIZE_S),
    )
```

iii. The AI states in trajectory step 9 that the repository showed firing rate should be spike count divided by bin width, and the script header repeats that firing rate is a 50 ms spike histogram divided by 0.05 s.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == "good"` are kept. If a session has no such units, it is dropped before conversion.

ii. 
```python
classification = decode_strings(nwb["units/classification"][:])
good = classification == "good"
if np.count_nonzero(good) == 0:
    print(f"Skipping {path.name}: no classifier-approved units", flush=True)
    continue
```

```python
classification = decode_strings(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
```

iii. In trajectory step 23 the AI says the NWB export already carries the paper’s classifier result directly, so "no quality model needs to be recreated."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go-cue onset. The AI keeps all timestamps on the common NWB session clock, constructs trial windows as `go_times + [-2.5, 1.5]`, and converts each spike time to a go-cue-relative bin index by subtracting that trial’s go time.

ii. 
```python
window_starts = go_times + OFF_START_S
window_ends = go_times + OFF_END_S
```

```python
relative = candidate_spikes - go_times[candidate_trials]
bin_idx = np.searchsorted(BIN_EDGES, relative, side="right") - 1
```

iii. The script header says all streams are "aligned in their common NWB session clock, then expressed relative to each trial's go onset." The trajectory says the repository confirmed go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins, spanning 80 time bins over [-2.5 s, 1.5 s). No further temporal rebinning or overlap/stride is applied.

ii. 
```python
BIN_SIZE_S = 0.050
OFF_START_S = -2.5
OFF_END_S = 1.5
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
BIN_EDGES = OFF_START_S + np.arange(N_TIME + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
```

iii. In trajectory step 6 the AI says it will preserve the requested 50 ms bins and -2.5 to +1.5 s window exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `acquisition/BehavioralEvents/sample_start_times/timestamps`, `intervals/trials/start_time`, and go-cue timestamps. The AI picks the last sample/tone onset before the go cue, constrained to be after the trial start.

ii. 
```python
def final_tone_onsets(
    trial_starts: np.ndarray,
    go_times: np.ndarray,
    sample_starts: np.ndarray,
) -> np.ndarray:
    """Select the final tone onset before go, after any early-lick replay."""
    result = np.empty(len(go_times), dtype=np.float64)
    for trial, (start, go) in enumerate(zip(trial_starts, go_times)):
        stop_idx = np.searchsorted(sample_starts, go, side="right")
        start_idx = np.searchsorted(sample_starts, start, side="left")
        ...
        result[trial] = sample_starts[stop_idx - 1]
    return result
```

```python
trial_starts = trials["start_time"][:][source_trial_idx]
sample_starts = events["sample_start_times/timestamps"][:]
tone_onsets = final_tone_onsets(trial_starts, go_times, sample_starts)
```

iii. The helper docstring and `metadata["tone_onset_definition"]` say the AI chose the "final sample/tone onset preceding go, after any early-lick replay."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes time from tone onset at each neural bin center by adding the go-relative bin center to the per-trial delay between go and tone onset: `BIN_CENTERS + (go_times - tone_onsets)`.

ii. 
```python
time_from_tone = (
    BIN_CENTERS[None, :] + (go_times - tone_onsets)[:, None]
).astype(np.float32)
```

iii. The trajectory indicates the AI wanted this input on the same 80-bin grid as the neural data, so it reused the go-centered bin centers instead of building a separate time axis.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by construction to the same go-cue-relative 80-bin grid as the neural data, using the same `BIN_CENTERS`.

ii. 
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
...
time_from_tone = (
    BIN_CENTERS[None, :] + (go_times - tone_onsets)[:, None]
).astype(np.float32)
```

iii. The AI’s general alignment rationale, stated in the script header and trajectory, is that all streams share the NWB clock and are therefore placed on a single go-cue-relative grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial-table columns `photostim_onset`, `photostim_duration`, and `start_time`, together with the go-cue times used to locate bin centers in absolute time.

ii. 
```python
photo_onset = parse_optional_floats(
    trials["photostim_onset"][:][source_trial_idx]
)
photo_duration = parse_optional_floats(
    trials["photostim_duration"][:][source_trial_idx]
)
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
absolute_photo_start = trial_starts + photo_onset
absolute_photo_end = absolute_photo_start + photo_duration
```

iii. The trajectory shows the AI checked both the trial-table photostim fields and the event streams, then used the trial-table onset and duration because they directly encode per-trial stimulation relative to trial start.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI decodes the onset and duration strings, maps `'N/A'` to `NaN`, converts onset to absolute session time by adding `trial_starts`, and sets each bin to 1 if the absolute bin center falls within `[stim_start, stim_end)`. Non-stimulated trials remain all zero because their start time is `NaN`.

ii. 
```python
def parse_optional_floats(values: Iterable[object]) -> np.ndarray:
    """Convert NWB string numbers, mapping 'N/A' to NaN."""
    text = decode_strings(values)
    result = np.full(text.shape, np.nan, dtype=np.float64)
    valid = text != "N/A"
    result[valid] = text[valid].astype(np.float64)
    return result
```

```python
photo_on = (
    (absolute_centers >= absolute_photo_start[:, None])
    & (absolute_centers < absolute_photo_end[:, None])
    & np.isfinite(absolute_photo_start[:, None])
).astype(np.float32)
```

iii. The AI’s trajectory says it wanted photostimulation as a time-varying decoder input rather than a trial-level flag, so it explicitly converts trial metadata into a binary time series on the decoder grid.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation to the neural data by comparing absolute photostim intervals against `absolute_centers = go_times + BIN_CENTERS`, so the photostim vector is sampled at the same time points as the neural firing rates.

ii. 
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
absolute_photo_start = trial_starts + photo_onset
absolute_photo_end = absolute_photo_start + photo_duration
photo_on = (
    (absolute_centers >= absolute_photo_start[:, None])
    & (absolute_centers < absolute_photo_end[:, None])
    & np.isfinite(absolute_photo_start[:, None])
).astype(np.float32)
```

iii. The general justification, repeated in the script header, is that all streams live on one session clock and are expressed relative to go onset only after that shared-time comparison.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the behavioral event streams `left_lick_times/timestamps` and `right_lick_times/timestamps`, not from `trial_instruction` and `outcome`. It finds the first lick after the go cue within the 1.5 s response window.

ii. 
```python
def event_choice(
    go_times: np.ndarray,
    left_licks: np.ndarray,
    right_licks: np.ndarray,
) -> np.ndarray:
    """Return first response-epoch lick: left=0, right=1, no lick=2."""
    choice = np.full(len(go_times), 2, dtype=np.int8)
    for trial, go in enumerate(go_times):
        li = np.searchsorted(left_licks, go, side="left")
        ri = np.searchsorted(right_licks, go, side="left")
        left_time = left_licks[li] if li < len(left_licks) else np.inf
        right_time = right_licks[ri] if ri < len(right_licks) else np.inf
        response_end = go + 1.5
        if left_time < response_end and left_time <= right_time:
            choice[trial] = 0
        elif right_time < response_end:
            choice[trial] = 1
    return choice
```

```python
left_licks = events["left_lick_times/timestamps"][:]
right_licks = events["right_lick_times/timestamps"][:]
choice = event_choice(go_times, left_licks, right_licks)
```

iii. In trajectory step 23, before the final implementation diverged, the AI said lick direction "can be recovered exactly from instruction plus hit/miss, with ignores mapped to no lick." The code does not follow that claim; the implemented choice logic instead uses first post-go lick events.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as `0 = left`, `1 = right`, `2 = no lick`, then repeated across all 80 time bins for each trial.

ii. 
```python
choice = np.full(len(go_times), 2, dtype=np.int8)
...
if left_time < response_end and left_time <= right_time:
    choice[trial] = 0
elif right_time < response_end:
    choice[trial] = 1
```

```python
for trial in range(n_trials):
    output_trial = np.empty((4, N_TIME), dtype=np.int8)
    output_trial[0, :] = choice[trial]
    output_trial[1, :] = outcome[trial]
    output_trial[2, :] = early[trial]
    output_trial[3, :] = tongue_class[trial]
    output_trials.append(output_trial)
```

iii. The AI’s trajectory says it tiled per-trial outputs across time because the target format should not mix scalar and time-varying rows inside a single output array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from the `intervals/trials/outcome` column.

ii. 
```python
outcome_text = decode_strings(trials["outcome"][:][source_trial_idx])
```

iii. In trajectory step 23 the AI explicitly notes that the NWB trial table already uses the requested `ignore/miss/hit` labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped with `{"ignore": 0, "miss": 1, "hit": 2}` and repeated across all 80 bins for each trial.

ii. 
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
unknown_outcome = sorted(set(outcome_text) - set(outcome_map))
if unknown_outcome:
    raise ValueError(f"{path}: unknown outcomes {unknown_outcome}")
outcome = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.int8)
```

```python
output_trial[1, :] = outcome[trial]
```

iii. The trajectory justification is straightforward: the dataset already contains the required labels, so only integer encoding is needed for the decoder.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the `intervals/trials/early_lick` column.

ii. 
```python
early_text = decode_strings(trials["early_lick"][:][source_trial_idx])
```

iii. The AI treats the trials-table label as authoritative and keeps those trials because early lick itself is a requested decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `"no early"` to `0` and `"early"` to `1` via a boolean comparison, then repeats that per-trial value across all bins.

ii. 
```python
unknown_early = sorted(set(early_text) - {"no early", "early"})
if unknown_early:
    raise ValueError(f"{path}: unknown early-lick labels {unknown_early}")
early = (early_text == "early").astype(np.int8)
```

```python
output_trial[2, :] = early[trial]
```

iii. The AI’s trajectory argues that early-lick trials should be retained because otherwise one of the requested decoder outputs would be removed from the dataset.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` and `timestamps`. The AI uses column 1 as tongue y and column 2 as the tracking likelihood/visibility score.

ii. 
```python
tongue = nwb[
    "acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"
]
tongue_data = tongue["data"][:]
tongue_times = tongue["timestamps"][:]
tongue_y, visible, n_outliers = clean_tongue_y(
    tongue_data[:, 1], tongue_data[:, 2]
)
```

iii. The script header says tongue position uses the side camera "as in the method paper," and the trajectory shows the AI inspected the tracking arrays before settling on this source.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first marks frames as visible if likelihood is at least 0.9, then identifies five-sigma velocity outliers across consecutive visible frames and linearly interpolates those outliers from neighboring good samples. It computes session-wide 40th and 60th percentiles from all visible tongue-y samples, not from 50 ms bin means. To label trials, it samples the nearest camera frame to each neural bin center, checks whether that sampled frame is visible, and thresholds the sampled y value with the session percentiles.

ii. 
```python
TONGUE_LIKELIHOOD_CUTOFF = 0.9
```

```python
def clean_tongue_y(
    y: np.ndarray, likelihood: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Interpolate visible five-sigma velocity outliers."""
    clean = np.asarray(y, dtype=np.float64).copy()
    visible = (
        np.isfinite(clean)
        & np.isfinite(likelihood)
        & (likelihood >= TONGUE_LIKELIHOOD_CUTOFF)
    )
    consecutive = visible[:-1] & visible[1:]
    speeds = np.abs(np.diff(clean))
    speed_sample = speeds[consecutive & np.isfinite(speeds)]
    ...
    if np.any(outlier) and np.count_nonzero(good) >= 2:
        x = np.arange(len(clean), dtype=np.float64)
        clean[outlier] = np.interp(x[outlier], x[good], clean[good])
```

```python
q40, q60 = np.percentile(tongue_y[visible], [40.0, 60.0])
video_idx = nearest_indices(tongue_times, absolute_centers.ravel()).reshape(
    n_trials, N_TIME
)
sampled_y = tongue_y[video_idx]
sampled_visible = visible[video_idx]
```

iii. The script header says this follows the method paper by interpolating visible five-sigma velocity outliers, while low-confidence samples become the requested `"not visible"` class. The trajectory also notes that the AI deliberately did not impute occluded samples because the decoder task asks for a separate not-visible category.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI creates four categories: `0` below the session-wide visible-sample 40th percentile, `1` between the 40th and 60th percentiles inclusive, `2` above the 60th percentile, and `3` not visible.

ii. 
```python
q40, q60 = np.percentile(tongue_y[visible], [40.0, 60.0])
...
tongue_class = np.full((n_trials, N_TIME), 3, dtype=np.int8)
tongue_class[sampled_visible & (sampled_y < q40)] = 0
tongue_class[
    sampled_visible & (sampled_y >= q40) & (sampled_y <= q60)
] = 1
tongue_class[sampled_visible & (sampled_y > q60)] = 2
```

iii. The AI’s stated justification is that low-confidence samples should map to the task’s explicit `"not visible"` category, and that session-wide percentiles are the requested discretization rule.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue position by taking, for each neural bin center, the nearest camera frame in absolute time and assigning that frame’s discretized tongue class to the bin. It does not average all frames within each 50 ms bin.

ii. 
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
...
video_idx = nearest_indices(tongue_times, absolute_centers.ravel()).reshape(
    n_trials, N_TIME
)
sampled_y = tongue_y[video_idx]
sampled_visible = visible[video_idx]
```

iii. The trajectory and script header both emphasize that all streams share one session clock, and the AI chose nearest-sample alignment to put video on the same grid as the firing-rate bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missingness/error cases explicitly: `'N/A'` photostim fields are converted to `NaN`; sessions with no `"good"` units are skipped; behavioral-only trial tails and all-zero neural trials are excluded; low-likelihood tongue frames are marked not visible instead of imputed; five-sigma visible-frame tongue outliers are interpolated; and a session would error if it had fewer than two visible tongue samples. It does not use a special helper like the reference `_text()` to coerce missing text labels to empty strings.

ii. 
```python
def parse_optional_floats(values: Iterable[object]) -> np.ndarray:
    """Convert NWB string numbers, mapping 'N/A' to NaN."""
    text = decode_strings(values)
    result = np.full(text.shape, np.nan, dtype=np.float64)
    valid = text != "N/A"
    result[valid] = text[valid].astype(np.float64)
    return result
```

```python
if np.count_nonzero(good) == 0:
    print(f"Skipping {path.name}: no classifier-approved units", flush=True)
    continue
```

```python
has_neural_data = np.any(rates != 0, axis=(0, 2))
...
tongue_y, visible, n_outliers = clean_tongue_y(
    tongue_data[:, 1], tongue_data[:, 2]
)
if np.count_nonzero(visible) < 2:
    raise ValueError(f"{path}: fewer than two visible tongue samples")
```

iii. The script header explains the two tongue-related policies: low-confidence samples stay as `"not visible"` because the decoder requests that class, while visible outliers are interpolated to follow the method paper. Trajectory step 42 justifies removing behavioral-only tails because otherwise they appear as artificial zero-neural trials.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive parts are opening and scanning every NWB file twice, reading large spike-time and tongue-tracking arrays from disk, and the per-unit spike binning loop in `bin_good_units()`. The code structure suggests neural binning dominates conversion cost.

ii. 
```python
for path in paths:
    with h5py.File(path, "r") as nwb:
        ...
```

```python
with h5py.File(path, "r") as nwb:
    ...
    rates = bin_good_units(units, good_indices, go_times)
    ...
    tongue_data = tongue["data"][:]
    tongue_times = tongue["timestamps"][:]
```

```python
for output_unit, source_unit in enumerate(good_indices):
    spikes = all_spikes[starts[source_unit]:ends[source_unit]]
    ...
    np.add.at(
        rates[output_unit],
        (candidate_trials[valid_bin], bin_idx[valid_bin]),
        np.float32(1.0 / BIN_SIZE_S),
    )
```

iii. The trajectory focuses performance discussion on validation/training rather than conversion timing, but the implementation itself shows the likely bottlenecks: full-file HDF5 I/O plus per-unit spike processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops remain: the per-unit loop in `bin_good_units()`, the per-trial loops in `event_choice()` and `final_tone_onsets()`, and the per-trial loop that assembles `output_trials`. These could be reduced with more vectorized indexing or batch construction.

ii. 
```python
for output_unit, source_unit in enumerate(good_indices):
    ...
```

```python
for trial, go in enumerate(go_times):
    ...
```

```python
for trial, (start, go) in enumerate(zip(trial_starts, go_times)):
    ...
```

```python
for trial in range(n_trials):
    output_trial = np.empty((4, N_TIME), dtype=np.int8)
    ...
    output_trials.append(output_trial)
```

iii. The AI did not explicitly justify these remaining loops in the trajectory, but the code suggests it favored straightforward implementations over deeper vectorization once the dataset passed decoder validation.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some work. Every NWB file is opened once in `scan_vocab()` and again in `convert_session()`. It also repeatedly decodes string datasets column by column and repeatedly copies data into per-trial Python lists after building larger session-level arrays.

ii. 
```python
subjects, brain_regions, paths = scan_vocab(paths)
...
for session, path in enumerate(paths, start=1):
    converted = convert_session(path, subject_to_idx, region_to_idx)
```

```python
classification = decode_strings(nwb["units/classification"][:])
...
annotations = decode_strings(nwb["units/anno_name"][:])[good]
```

```python
neural_trials = [rates[:, trial, :].copy() for trial in range(n_trials)]
input_trials = [
    np.stack((time_from_tone[trial], photo_on[trial])).astype(
        np.float32, copy=False
    )
    for trial in range(n_trials)
]
```

iii. The trajectory does not present this as an intentional optimization tradeoff; it is simply how the final implementation is structured.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes some quantities that are only written into metadata and not used by the decoder, such as tongue percentiles, number of tongue outliers imputed, and per-session label-count summaries. It also makes an extra `scan_vocab()` pass over every file before real conversion, which is not needed for the final decoder arrays themselves.

ii. 
```python
session_info = {
    "identifier": identifier,
    "source_file": str(path),
    "subject": subject,
    "n_trials": int(n_trials),
    "n_behavior_trials": int(n_behavior_trials),
    "n_ephys_trials": int(n_ephys_trials),
    "n_zero_neural_trials_excluded": n_zero_neural,
    "n_good_units": int(len(good_indices)),
    "tongue_visible_y_percentile_40": float(q40),
    "tongue_visible_y_percentile_60": float(q60),
    "tongue_velocity_outliers_imputed": n_outliers,
    "choice_counts": np.bincount(choice, minlength=3).astype(int).tolist(),
    "outcome_counts": np.bincount(outcome, minlength=3).astype(int).tolist(),
    "early_lick_counts": np.bincount(early, minlength=2).astype(int).tolist(),
}
```

```python
subjects, brain_regions, paths = scan_vocab(paths)
```

iii. The trajectory emphasizes correctness and decoder performance, not minimal preprocessing. These extra metadata computations appear to be provenance conveniences rather than required decoder inputs.
