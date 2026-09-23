# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI 000363) is one NWB file per session under `data/sub-<subject_id>/`. The AI enumerates every session with a single sorted glob over that layout and reads the files with **raw `h5py`** rather than `pynwb`, justifying this in the module docstring by the archived files using an older NWB schema while every field needed is a plain HDF5 dataset. Each file is then opened **twice**: once in `scan_vocab()` (to build the global subject and brain-region vocabularies and to discard unusable sessions) and once in `convert_session()` (for the actual conversion). Within a session the AI reads `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, `units/*` and `general/subject/subject_id`. All 174 files are enumerated; 173 reach the output.

ii.
```python
def convert(data_dir: Path, output_path: Path) -> dict:
    paths = sorted(data_dir.glob("sub-*/*.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found below {data_dir}")
    subjects, brain_regions, paths = scan_vocab(paths)
```
```python
with h5py.File(path, "r") as nwb:
    trials = nwb["intervals/trials"]
    events = nwb["acquisition/BehavioralEvents"]
    units = nwb["units"]
```

iii. From the trajectory (step 42 patch header and the module docstring): "The script intentionally uses h5py rather than pynwb: the archived files use an older NWB schema, while all fields needed here are standard HDF5 datasets." The agent first inventoried the HDF5 tree of sample files (steps 10–13, 26) to confirm the exact dataset paths before committing to direct reads, and confirmed the file count/subject count against the dandiset layout (step 32).

## 1-b. How are the data split into subjects (mice)?

i. The animal identity is read per session from `general/subject/subject_id` (a numeric string such as `'440956'`). `scan_vocab()` collects the sorted unique set over all usable sessions into `subjects`, and each session stores its index into that list in `subject_idx`. Result: 28 subjects, 3–10 sessions each.

ii.
```python
subjects.add(nwb["general/subject/subject_id"][()].decode("utf-8").strip())
...
return sorted(subjects), sorted(regions), usable
```
```python
subject = nwb["general/subject/subject_id"][()].decode("utf-8").strip()
...
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int32),
```

iii. The agent verified (steps 26, 32) that `subject_id` is the canonical animal field in each file and that it matches the containing `sub-<id>` directory name, so no separate grouping is needed. It kept the numeric DANDI id rather than the mouse name (`SC015`, …) embedded in `identifier`, and preserved the full session identifier separately in `metadata['session_info']`.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no splitting or grouping is performed. Session order follows the sorted file list (which is chronological within a subject because the filename embeds the acquisition timestamp). Each session's `identifier` (e.g. `SC015_20190207_120657_s1`) and source path are recorded in `metadata['session_info']`. Sessions are dropped in two places: in `scan_vocab()` if they have no classifier-approved units, and in `convert()` if fewer than two trials survive. 173 of 174 sessions are kept.

ii.
```python
for path in paths:
    with h5py.File(path, "r") as nwb:
        classification = decode_strings(nwb["units/classification"][:])
        good = classification == "good"
        if np.count_nonzero(good) == 0:
            print(f"Skipping {path.name}: no classifier-approved units", flush=True)
            continue
```
```python
if len(n) < 2:
    print(f"Skipping {path.name}: fewer than two trials", flush=True)
    continue
```

iii. Step 29: "I found one NWB session with zero classifier-approved units. Excluding only that unusable session yields the paper's reported 173-session count." The agent explicitly checked (step 32) for duplicate identifiers, sessions failing the paper's behavioural selection criteria (>65% performance, ≥50 correct left/right trials) and sessions without tongue tracking, and chose not to apply the paper's behavioural session criteria because it found no session failing them beyond the unusable one.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioural trial, with the go cue for trial *i* taken as element *i* of `BehavioralEvents/go_start_times/timestamps`. The one-go-cue-per-row assumption is asserted explicitly. The per-trial window is then `[go - 2.5 s, go + 1.5 s)`, and `bin_good_units()` assigns each spike to exactly one trial, relying on the (verified) fact that these 4-s windows never overlap.

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
# Trial windows do not overlap in this dataset (the shortest go-to-go
# interval exceeds four seconds), so each in-window spike has one trial.
trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
```

iii. The agent inspected the trials table and event groups (step 26) and established that `go_start_times` has exactly one entry per trial row, unlike `sample_start_times`/`delay_start_times`, which get replayed after early licks. The non-overlap claim was checked before relying on it (I re-verified it: the minimum go-to-go interval over all 94,816 trials is 4.58 s > 4 s).

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both aimed only at absence of neural data; no behavioural quality filter is applied.
  1. **Behaviour-only tail**: the number of trials covered by ephys is taken as `units/is_good_trials.shape[1]`, and only the *first* `n_ephys_trials` rows of the trials table are converted.
  2. **Acquisition gaps**: after binning, any trial with no spikes at all from any good unit is dropped (`np.any(rates != 0)`). This implicitly removes the `free_water` trials, which carry no spikes.
A session is skipped if fewer than 2 trials survive. Early-lick, `ignore` and photostimulation trials are deliberately kept. Final: 90,734 trials over 173 sessions.

ii.
```python
# A few NWB files contain behavior continuing after the ephys recording.
# is_good_trials has one column per trial actually covered by ephys (as
# do each unit's ragged obs_intervals); later behavioral-only trials must
# not become artificial all-zero neural trials.
n_ephys_trials = units["is_good_trials"].shape[1]
if n_ephys_trials > n_behavior_trials:
    raise ValueError(f"{path}: more ephys trials than behavioral trials")
source_trial_idx = np.arange(n_ephys_trials, dtype=np.int64)
go_times = all_go_times[source_trial_idx]
```
```python
# Some archived trials have an observation interval but no spikes from
# any unit at all (including unclassified units).  These acquisition
# gaps cannot supply a neural decoder input and are excluded explicitly.
has_neural_data = np.any(rates != 0, axis=(0, 2))
n_zero_neural = int(np.count_nonzero(~has_neural_data))
source_trial_idx = source_trial_idx[has_neural_data]
```

iii. Docstring: "Every trial with electrophysiology is retained, except acquisition gaps with no spikes from any unit. The papers excluded early-lick, ignore, free-water, and photostimulation trials for particular analyses, but those exclusions would remove labels or inputs explicitly requested by this decoder task." Step 42: "The verifier exposed an important NWB-specific issue: several trial tables extend beyond the electrophysiology recording, producing zero-neural tails. The per-unit `obs_intervals`/`is_good_trials` fields encode the actual recorded trial span. I'm correcting trial curation to retain only go-cue windows covered by electrophysiology." The agent then audited the residual all-zero trials directly against spike times (steps 72–74) before adding the second filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (ragged, session-absolute seconds) with `units/spike_times_index` for the per-unit offsets, restricted to units whose `units/classification == 'good'`; `BehavioralEvents/go_start_times/timestamps` supplies the alignment times. `units/anno_name` supplies each retained unit's brain region.

ii.
```python
classification = decode_strings(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
annotations = decode_strings(units["anno_name"][:])[good_indices]
rates = bin_good_units(units, good_indices, go_times)
```
```python
all_spikes = units["spike_times"]
ends = units["spike_times_index"][:].astype(np.int64, copy=False)
starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
```

iii. Step 9: "The repository confirms the key source choices: spikes are aligned by subtracting each trial's go-cue time, firing rate is spike count divided by bin width, and classifier-QC 'good' units are the paper's analysis population." Spike times are the only neural representation in the files.

## 2-b. How is the `neural` data processed?

i. Per good unit, spikes are assigned to a trial, then to one of 80 non-overlapping 50-ms bins, and accumulated with a weight of `1/0.05`, so the stored value is a firing rate in Hz (multiples of 20 Hz). No smoothing, no normalisation, no baseline subtraction. Rates are stored `float32`, one `(n_neurons, 80)` array per trial.

ii.
```python
rates = np.zeros((n_units, n_trials, N_TIME), dtype=np.float32)
...
relative = candidate_spikes - go_times[candidate_trials]
bin_idx = np.searchsorted(BIN_EDGES, relative, side="right") - 1
valid_bin = (bin_idx >= 0) & (bin_idx < N_TIME)
np.add.at(
    rates[output_unit],
    (candidate_trials[valid_bin], bin_idx[valid_bin]),
    np.float32(1.0 / BIN_SIZE_S),
)
```
```python
neural_trials = [rates[:, trial, :].copy() for trial in range(n_trials)]
```

iii. Docstring: "Firing rate is a non-overlapping 50-ms spike histogram divided by 0.05 s. The half-open bins exactly tile [-2.5, 1.5), giving 80 samples." This follows `sliding_histogram(..., rate=True)` in the supplied repository (`code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py`, which returns `binSpikes/float(bin_width)`), which the agent read in step 8. Step 32 records the dry-run check that rates came out in 20-Hz increments with shape `(neurons, 80)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `units/classification == 'good'` units are used — the verdict of the paper's region-specific QC classifiers. No individual quality metric is thresholded, and the older `units/unit_quality` label is not used. A good unit with an empty `anno_name` raises an error. A session with zero good units is dropped. Result: 69,453 units over 173 sessions (mean 401/session, range 90–923).

ii.
```python
classification = decode_strings(nwb["units/classification"][:])
good = classification == "good"
if np.count_nonzero(good) == 0:
    print(f"Skipping {path.name}: no classifier-approved units", flush=True)
    continue
...
annotations = decode_strings(nwb["units/anno_name"][:])[good]
if np.any(annotations == ""):
    raise ValueError(f"Good unit without anatomical annotation in {path}")
```

iii. Step 23: "The NWB export carries the paper's classifier result directly (`units/classification == 'good'`), so no quality model needs to be recreated." Step 29: "I'll keep all classifier-approved units (the repository preprocessing does so) and record exact Allen CCF annotation names rather than inventing broader region groupings absent from the NWB files." The agent cross-read the QC white paper (steps 16–22) to confirm `classification` is the classifier output described there.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Everything in the file is on one session-absolute clock, so no resampling or offset correction is needed. For each trial the window `[go - 2.5, go + 1.5)` is taken in absolute time; each spike is mapped to its trial by `searchsorted` on the window starts, and its bin index is computed from `spike - go`. The alignment event is recorded as `'go cue onset'` in the metadata.

ii.
```python
window_starts = go_times + OFF_START_S
window_ends = go_times + OFF_END_S
...
trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
valid_trial = trial_idx >= 0
candidate_spikes = spikes[valid_trial]
candidate_trials = trial_idx[valid_trial]
in_window = candidate_spikes < window_ends[candidate_trials]
...
relative = candidate_spikes - go_times[candidate_trials]
```

iii. Docstring: "Spike times, video, tones, licks, and laser epochs are all aligned in their common NWB session clock, then expressed relative to each trial's go onset." Step 9 confirmed against the supplied repository that alignment is done by subtracting the trial's go-cue time. Step 91 was a final "provenance/alignment audit against event timestamps".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, fixed for every trial and session, with 80 half-open bins tiling `[-2.5, +1.5)` s relative to the go cue. The edges/centres are computed once at module scope and reused everywhere (neural, photostim, time-from-tone, tongue). Spike times are binned once at this resolution; no rebinning or re-sampling of an intermediate grid occurs. `metadata['time_bin_size'] = 50.0` (ms) and the 80 bin centres are also stored.

ii.
```python
BIN_SIZE_S = 0.050
OFF_START_S = -2.5
OFF_END_S = 1.5
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
BIN_EDGES = OFF_START_S + np.arange(N_TIME + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
```

iii. The window and bin width are prescribed by the instructions; the docstring notes the half-open convention was chosen so the bins "exactly tile [-2.5, 1.5), giving 80 samples". The verifier confirmed `T_min = T_max = 80`.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times/timestamps` (the sample-epoch tone onsets), bounded by `intervals/trials/start_time` and the trial's go cue. Because an early lick replays the sample epoch, a trial can contain several sample onsets; the **last one at or before the go cue and at or after the trial start** is taken.

ii.
```python
def final_tone_onsets(trial_starts, go_times, sample_starts):
    """Select the final tone onset before go, after any early-lick replay."""
    result = np.empty(len(go_times), dtype=np.float64)
    for trial, (start, go) in enumerate(zip(trial_starts, go_times)):
        stop_idx = np.searchsorted(sample_starts, go, side="right")
        start_idx = np.searchsorted(sample_starts, start, side="left")
        if stop_idx <= start_idx:
            raise ValueError(f"Trial {trial} has no sample/tone onset before go")
        result[trial] = sample_starts[stop_idx - 1]
    return result
```

iii. `metadata['tone_onset_definition']`: "final sample/tone onset preceding go, after any early-lick replay". The agent read in `methods.txt` that "Licking early during the sample/delay epoch triggered a replay of the epoch", and confirmed in step 26 that `sample_start_times` therefore has a variable number of entries per trial. Requiring the onset to lie inside the trial is a guard that turns a silent mis-assignment into a hard error.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying input: for each trial, value of bin *k* = (bin centre relative to go) + (go − tone). It is stored as `float32` in row 0 of the `(2, 80)` input array. No clipping, normalisation or binarisation. Observed range across the dataset: −1.525 s to 11.89 s (long values come from early-lick replays).

ii.
```python
tone_onsets = final_tone_onsets(trial_starts, go_times, sample_starts)
time_from_tone = (
    BIN_CENTERS[None, :] + (go_times - tone_onsets)[:, None]
).astype(np.float32)
```

iii. The decoder specification lists this input as "continuous, time-varying", so it is kept as seconds rather than converted to a binary onset indicator. Step 32 records the sanity check: "sensible tone-time limits (−0.625 to 3.325 s for a standard trial)", i.e. consistent with the 0.65 s sample + 1.2 s delay structure described in `methods.txt`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: it is built from the same `BIN_CENTERS` array that defines the firing-rate bins, shifted by the scalar tone-to-go interval of that trial. Bin *k* of the input therefore covers exactly the same interval as bin *k* of the neural data.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
...
time_from_tone = (
    BIN_CENTERS[None, :] + (go_times - tone_onsets)[:, None]
).astype(np.float32)
```

iii. No separate alignment step is needed: both streams are expressed on one go-cue-relative grid defined once at module scope, and all event times live on the same session clock.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration` (stored as strings, relative to trial start, `'N/A'` when the trial was not stimulated), together with `intervals/trials/start_time` to convert them to the session clock, and the trial's go cue to place them on the bin grid.

ii.
```python
photo_onset = parse_optional_floats(
    trials["photostim_onset"][:][source_trial_idx]
)
photo_duration = parse_optional_floats(
    trials["photostim_duration"][:][source_trial_idx]
)
```
```python
def parse_optional_floats(values):
    """Convert NWB string numbers, mapping 'N/A' to NaN."""
    text = decode_strings(values)
    result = np.full(text.shape, np.nan, dtype=np.float64)
    valid = text != "N/A"
    result[valid] = text[valid].astype(np.float64)
    return result
```

iii. The agent dumped the trials table in step 26 and found the onset/duration columns are string-typed with `'N/A'` sentinels measured from trial start, so an explicit parse and a re-referencing to the session clock are required. `methods.txt` describes photoinhibition as a late-delay epoch event, which is inside the −2.5 s window.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time series rather than a per-trial flag: a bin is 1 when its centre lies in `[photostim start, photostim start + duration)` in absolute time. Non-stimulated trials carry NaN bounds and are forced to 0 by an explicit `isfinite` guard. Stored `float32` as row 1 of the input array; verified range 0–1.

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

iii. The decoder specification asks for "whether photostimulation is on at every time point (discrete, time-varying)", so the laser epoch is expanded over bins rather than reduced to a trial-level flag. The `isfinite` term makes the not-stimulated case explicit instead of relying on NaN comparison semantics.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Both the laser window and the bin centres are converted to absolute session time (`go + BIN_CENTERS`) and compared there, so the photostim row uses exactly the same 80-bin, go-cue-aligned grid as the firing rates.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
absolute_photo_start = trial_starts + photo_onset
```

iii. Same rationale as 2-d: all streams share one clock, so aligning is a matter of expressing the laser epoch in that clock and evaluating it at the bin centres. No interpolation or offset correction is involved.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Not from the trials table. The AI derives choice from the raw lick events: `BehavioralEvents/left_lick_times/timestamps` and `BehavioralEvents/right_lick_times/timestamps`, evaluated against each trial's go cue. (`trial_instruction` and `outcome` — the fields the reference used — are read but only `outcome` is used, for its own output.)

ii.
```python
left_licks = events["left_lick_times/timestamps"][:]
right_licks = events["right_lick_times/timestamps"][:]
choice = event_choice(go_times, left_licks, right_licks)
```
```python
def event_choice(go_times, left_licks, right_licks):
    """Return first response-epoch lick: left=0, right=1, no lick=2."""
```

iii. `metadata['choice_definition']`: "first left/right lick in [go onset, go onset + 1.5 s); otherwise no lick". Step 23 notes that "lick direction can be recovered exactly from instruction plus hit/miss, with ignores mapped to no lick", i.e. the agent was aware of the label-based derivation and chose the directly measured lick event instead, using the 1.5 s answer period stated in `methods.txt`. (I checked 12 sessions / 6,351 trials: the two derivations agree on 100% of trials.)

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the first left lick and the first right lick at or after the go cue are found by `searchsorted`; whichever is earlier and falls before `go + 1.5 s` gives the class (`0` left, `1` right), with ties going to left; if neither occurs in the window the class is `2` (no lick). The per-trial value is tiled across all 80 bins into row 0 of the `(4, 80)` `int8` output array. `output_values[0] = ['left', 'right', 'no lick']`. Resulting distribution: 43.0% left / 42.0% right / 15.0% no lick.

ii.
```python
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
```
```python
output_trial = np.empty((4, N_TIME), dtype=np.int8)
output_trial[0, :] = choice[trial]
```

iii. Step 23: "Because the decoder format cannot mix scalar and temporal rows in one output array, I'll tile the three per-trial labels across time and keep tongue position time-varying." The `left=0 / right=1` coding follows the instruction ordering, and the third class is required because `ignore` trials have no lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `intervals/trials/outcome` column, which already contains exactly the three requested strings `'ignore'`, `'miss'`, `'hit'`. Any other value is a hard error.

ii.
```python
outcome_text = decode_strings(trials["outcome"][:][source_trial_idx])
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
unknown_outcome = sorted(set(outcome_text) - set(outcome_map))
if unknown_outcome:
    raise ValueError(f"{path}: unknown outcomes {unknown_outcome}")
```

iii. Step 23: "Trial-table `outcome` uses the requested `ignore/miss/hit` labels", confirmed by enumerating the unique values across all sessions in step 32. No derivation is needed; the explicit unknown-value check guards the assumption.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String → integer map `ignore=0, miss=1, hit=2` (the instruction's ordering), tiled across all 80 bins into row 1 of the output array, `int8`. `output_values[1] = ['ignore', 'miss', 'hit']`. Distribution: 14.9% ignore / 16.6% miss / 68.4% hit — consistent with the ~84% correct rate reported in `methods.txt` once ignore trials are included.

ii.
```python
outcome = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.int8)
...
output_trial[1, :] = outcome[trial]
```

iii. One value per trial, so it is repeated over bins for the same structural reason given in 5-b. The code ordering follows the instruction list exactly so that `output_values` indices line up.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The `intervals/trials/early_lick` column, which holds `'no early'` / `'early'`. Unexpected values raise.

ii.
```python
early_text = decode_strings(trials["early_lick"][:][source_trial_idx])
unknown_early = sorted(set(early_text) - {"no early", "early"})
if unknown_early:
    raise ValueError(f"{path}: unknown early-lick labels {unknown_early}")
```

iii. The flag is stored explicitly in the trials table (verified in steps 26 and 32), so no derivation from lick times is needed. The agent kept early-lick trials rather than excluding them as the data paper does, because early lick is a requested decoder output (docstring and step 29).

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` → 1, `'no early'` → 0, tiled across all 80 bins into row 2 of the output array, `int8`. `output_values[2] = ['no', 'yes']`. Distribution: 88.4% no / 11.6% yes.

ii.
```python
early = (early_text == "early").astype(np.int8)
...
output_trial[2, :] = early[trial]
```

iii. Binary coding follows the instruction ("no, yes"). The triggering lick occurs during the sample/delay epoch, i.e. inside the −2.5 s window, so the label is decodable from the extracted window.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps` (~294 Hz). Column 1 provides the position and column 2 the DeepLabCut likelihood used for the visibility test. Column 0 is read but unused.

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

iii. `metadata['tongue_tracking_view'] = 'Camera0 side view'`; the docstring says "Tongue position uses the side camera, as in the method paper." The agent listed the tracking groups and the per-column quantiles in step 26 before fixing the column order, and confirmed from `methods.txt`/the method paper that DeepLabCut tracked tongue/jaw/nose from the side and bottom views.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps: (1) a frame counts as *visible* only if its likelihood ≥ 0.9 and the values are finite; (2) following the method paper, transitions between consecutive visible frames whose speed exceeds mean + 5·SD are flagged as outliers and linearly interpolated from neighbouring good samples (occluded frames are deliberately **not** imputed, since "not visible" is a requested class); (3) the 40th and 60th percentiles are taken over **all visible frames of the session** (not over bin means) and stored per session in `session_info`.

ii.
```python
visible = (
    np.isfinite(clean) & np.isfinite(likelihood)
    & (likelihood >= TONGUE_LIKELIHOOD_CUTOFF)
)
consecutive = visible[:-1] & visible[1:]
speeds = np.abs(np.diff(clean))
speed_sample = speeds[consecutive & np.isfinite(speeds)]
outlier = np.zeros(clean.shape, dtype=bool)
if speed_sample.size >= 2:
    cutoff = speed_sample.mean() + 5.0 * speed_sample.std()
    outlier[1:] = consecutive & (speeds > cutoff)
good = visible & ~outlier
if np.any(outlier) and np.count_nonzero(good) >= 2:
    x = np.arange(len(clean), dtype=np.float64)
    clean[outlier] = np.interp(x[outlier], x[good], clean[good])
```
```python
q40, q60 = np.percentile(tongue_y[visible], [40.0, 60.0])
```

iii. The method paper states: "we found outliers that harmed the prediction of firing rates or animal behavior. We identified outliers by a five-sigma threshold on velocity across frames and imputed outliers from nearby frames" — the agent read this passage in steps 18–19 and implemented it. The paper then sets occluded tongue positions to their mean, which the agent deliberately replaced with the requested "not visible" class. The 0.9 likelihood cutoff is justified in the code comment as a "conservative standard p-cutoff" given the strongly bimodal DeepLabCut probabilities.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Exactly the four classes in the instruction: `0` if y < q40, `1` if q40 ≤ y ≤ q60, `2` if y > q60, `3` if the sampled frame is not visible. Percentiles are per session. `output_values[3] = ['below 40th percentile', '40th to 60th percentile', 'above 60th percentile', 'not visible']`. Resulting distribution: 6.2% / 3.2% / 6.5% / 84.1% — i.e. among visible bins the split is ≈39/20/41, as intended.

ii.
```python
tongue_class = np.full((n_trials, N_TIME), 3, dtype=np.int8)
tongue_class[sampled_visible & (sampled_y < q40)] = 0
tongue_class[
    sampled_visible & (sampled_y >= q40) & (sampled_y <= q60)
] = 1
tongue_class[sampled_visible & (sampled_y > q60)] = 2
```

iii. The class boundaries and the per-session scope are dictated by the instructions; the AI implements the stated inequalities literally (strict below/above, inclusive middle) and uses the default of class 3 for everything not visible. `metadata['tongue_discretization']` documents the scheme.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. This is the only genuinely time-varying output. For each of the 80 bins of each trial, the AI takes the video frame whose timestamp is **nearest** the bin centre (in absolute session time) and uses that single frame's class — a point sample rather than an average of the frames inside the bin.

ii.
```python
def nearest_indices(sorted_times, query_times):
    """Indices of samples nearest each query in a sorted timestamp vector."""
    right = np.searchsorted(sorted_times, query_times, side="left")
    right = np.clip(right, 0, len(sorted_times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(query_times - sorted_times[left]) <= np.abs(
        sorted_times[right] - query_times
    )
    return np.where(choose_left, left, right)
```
```python
video_idx = nearest_indices(tongue_times, absolute_centers.ravel()).reshape(
    n_trials, N_TIME
)
sampled_y = tongue_y[video_idx]
sampled_visible = visible[video_idx]
```

iii. Code comment: "Thresholds use all visible session frames, while labels are sampled nearest each firing-rate bin center." The camera timestamps live on the same session clock as spikes and go cues, so nearest-neighbour sampling needs no interpolation or offset correction. At ~294 Hz a frame is within 1.7 ms of almost every bin centre (I measured ≤0.3% of bins landing in inter-trial video gaps, where the nearest frame can be up to ~0.36 s away; essentially all of those are occluded anyway and end up in class 3).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. A mix of exclusion, explicit categories, and loud failure:
- **Session with no QC'd units** (`classification`/`anno_name` are NaN): `decode_strings` stringifies non-bytes, no unit matches `'good'`, and the session is skipped (1 session).
- **Behaviour trials outside the ephys recording** and **trials with no spikes at all**: excluded (1-e).
- **`'N/A'` photostim strings**: parsed to NaN and forced to 0 via `isfinite`.
- **Occluded tongue frames**: not imputed, but given the explicit class 3.
- **Tracking outliers**: interpolated; if interpolation is impossible the frame is marked not visible.
- **Unexpected values** (unknown outcome or early-lick labels, good unit without annotation, go-cue/trial count mismatch, more ephys than behaviour trials, fewer than 2 visible tongue samples, trial with no tone before go): raise `ValueError` rather than being silently coerced.
- **Sessions ending with <2 trials**: skipped, as the target format requires ≥2 trials.

ii.
```python
result = np.full(text.shape, np.nan, dtype=np.float64)
valid = text != "N/A"
result[valid] = text[valid].astype(np.float64)
```
```python
elif np.any(outlier):
    visible[outlier] = False
```
```python
if np.count_nonzero(visible) < 2:
    raise ValueError(f"{path}: fewer than two visible tongue samples")
```
```python
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
with temporary.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
os.replace(temporary, output_path)
```

iii. The agent's stated position (step 42 and code comments) is that data that was never recorded must not be fabricated as zeros — "later behavioral-only trials must not become artificial all-zero neural trials" — while data whose value is legitimately undefined (retracted tongue, no photostim) gets an explicit encoding. Everything else that would indicate a misunderstood file layout is made to fail fast. The output is written to a temporary file and atomically renamed so a crash cannot leave a truncated 11 GB pickle.

## 10-a. What are the most time-consuming steps of the code?

i. The run is I/O- and allocation-bound. In order: (1) the `scan_vocab()` pre-pass, which opens all 174 files and decodes the `classification`/`anno_name` string columns; (2) per session, reading the ragged `spike_times` buffer one unit-slice at a time from HDF5 and the `(n_frames, 3)` tongue array (~680 k × 3 doubles); (3) the per-unit spike histogram, where `np.add.at` is used for the accumulation (an unbuffered ufunc, substantially slower than a `bincount`/`searchsorted`-difference formulation); (4) writing the 11.47 GiB pickle. The full conversion ran in roughly 2.5–3 minutes of wall time plus the pickle write.

ii.
```python
subjects, brain_regions, paths = scan_vocab(paths)   # full extra pass over all files
```
```python
all_spikes = units["spike_times"]        # h5py dataset: one read per unit below
for output_unit, source_unit in enumerate(good_indices):
    spikes = all_spikes[starts[source_unit]:ends[source_unit]]
    ...
    np.add.at(rates[output_unit], (...), np.float32(1.0 / BIN_SIZE_S))
```

iii. The agent did not profile explicitly; it monitored the background run (steps 34–38, 54–59) for progress and memory stability and accepted the runtime, having decided in step 71 to run the full pipeline rather than a reduced test. Its stated priority was correctness of curation and a complete 200-epoch decoder validation rather than conversion speed.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python-level loops are avoidable, and one is inherent:
- `event_choice()` loops over trials doing two scalar `searchsorted` calls each; this is one vectorised `np.searchsorted(left_licks, go_times)` away from being loop-free.
- `final_tone_onsets()` loops over trials for the same reason; `np.searchsorted(sample_starts, go_times, 'right') - 1` computes the whole vector at once.
- The per-trial list comprehensions that slice `rates`/`time_from_tone`/`photo_on` and build each `(4, 80)` output array could be built as whole-array operations and then split.
- The per-unit loop in `bin_good_units()` is inherent to the ragged `spike_times` storage, but inside it `np.add.at` could be replaced by `np.bincount` on a flattened `(trial, bin)` index (typically 10–50× faster), and the whole loop could be replaced by a single pass over the flat spike buffer with a unit index derived from `spike_times_index`.

ii.
```python
for trial, go in enumerate(go_times):
    li = np.searchsorted(left_licks, go, side="left")
    ri = np.searchsorted(right_licks, go, side="left")
```
```python
for trial, (start, go) in enumerate(zip(trial_starts, go_times)):
    stop_idx = np.searchsorted(sample_starts, go, side="right")
    start_idx = np.searchsorted(sample_starts, start, side="left")
```
```python
np.add.at(
    rates[output_unit],
    (candidate_trials[valid_bin], bin_idx[valid_bin]),
    np.float32(1.0 / BIN_SIZE_S),
)
```

iii. No justification is given in the code or trajectory; the loops appear to have been written for clarity (each is a short, readable per-trial rule) and were not revisited because total runtime was a few minutes, well inside the budget. The tongue and photostim computations, which dominate array size, *are* fully vectorised.

## 10-c. What processing does the code repeat multiple times?

i. Three repetitions:
- **Every NWB file is opened and parsed twice** — once in `scan_vocab()` and once in `convert_session()` — and `units/classification` plus `units/anno_name` are read and `decode_strings`-decoded in both passes.
- **Firing rates are computed for all ephys-covered trials and then discarded** for the trials removed by the `has_neural_data` filter (including every `free_water` trial).
- **Whole trial-table columns are read then immediately subset** (`trials["start_time"][:][source_trial_idx]`, and the same pattern for `photostim_onset`, `photostim_duration`, `outcome`, `early_lick`), so the full column is materialised five times per session even when only a prefix is needed.

ii.
```python
def scan_vocab(paths):
    for path in paths:
        with h5py.File(path, "r") as nwb:
            classification = decode_strings(nwb["units/classification"][:])
            ...
            annotations = decode_strings(nwb["units/anno_name"][:])[good]
```
```python
trial_starts = trials["start_time"][:][source_trial_idx]
photo_onset = parse_optional_floats(
    trials["photostim_onset"][:][source_trial_idx]
)
```

iii. The pre-pass is a deliberate design choice, not an oversight: building the global `subjects` and `brain_regions` vocabularies up front means every session can emit final indices in one streaming pass and never has to be revisited or re-indexed, and it lets unusable sessions be reported before any heavy work. It only re-reads two small string columns per file. The rate-then-filter ordering follows from the AI's decision to define an empty trial by the *absence of binned spikes* rather than by a metadata field, which necessarily requires binning first.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little of consequence, but several small items:
- `tongue_data[:, 0]` (tongue *x*) is loaded as part of the full `(n_frames, 3)` array and never used.
- Per-session diagnostics accumulated in `session_info` — `choice_counts`, `outcome_counts`, `early_lick_counts`, `tongue_velocity_outliers_imputed`, the q40/q60 values, `n_zero_neural_trials_excluded` — are provenance only and never read by the decoder.
- `metadata['time_bin_centers_s']` (the 80 centres) duplicates information already implied by `time_bin_size`/`off_start`/`off_end`.
- `brain_regions` keeps 293 full Allen CCF strings (e.g. "Agranular insular area, dorsal part, layer 5"); `decoder.py` only validates `brain_region_idx` and never uses it for decoding, so the extra granularity (and the vocabulary pre-pass that produces it) buys nothing downstream.
- Firing rates computed for trials later dropped (see 10-c).

ii.
```python
"tongue_visible_y_percentile_40": float(q40),
"tongue_visible_y_percentile_60": float(q60),
"tongue_velocity_outliers_imputed": n_outliers,
"choice_counts": np.bincount(choice, minlength=3).astype(int).tolist(),
```
```python
"time_bin_centers_s": BIN_CENTERS.tolist(),
```

iii. Step 29 gives the explicit rationale for the region granularity: "record exact Allen CCF annotation names rather than inventing broader region groupings absent from the NWB files". The diagnostics are justified as an audit trail — the agent used them (steps 72–75, 88–92) to verify trial curation and class balance — and cost a negligible fraction of the 11.47 GiB output.
