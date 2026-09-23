# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script finds all session files with a sorted glob over `sub-*/*.nwb`, then iterates through them one by one. Each file is opened directly with `h5py.File`, and the code reads the `units`, `intervals/trials`, `acquisition/BehavioralEvents`, and `acquisition/BehavioralTimeSeries` groups from that NWB/HDF5 file.

ii. 
```python
paths = sorted(glob.glob(os.path.join(data_dir, "sub-*", "*.nwb")))
...
for path in paths:
    print(f"Converting {path}", flush=True)
    session = convert_session(path)
```

```python
with h5py.File(path, "r") as nwb:
    units = nwb["units"]
    trials = nwb["intervals/trials"]
    go_times = nwb[
        "acquisition/BehavioralEvents/go_start_times/timestamps"
    ][:]
```

iii. In the trajectory, the agent first enumerated the dataset and found 174 NWB files, then concluded that the dataset is organized as one file per session. Later trajectory steps show it treating the one file with missing classifier QC as a session to skip, not as a loading/layout problem.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is taken from the NWB field `general/subject/subject_id`. After all sessions are converted, the script builds a sorted unique subject list and an integer `subject_idx` per session.

ii. 
```python
subject = _scalar_string(nwb["general/subject/subject_id"])
```

```python
subjects = sorted({session["subject"] for session in converted_sessions})
subject_to_index = {subject: i for i, subject in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.array(
    [subject_to_index[session["subject"]] for session in converted_sessions],
    dtype=np.int16,
),
```

iii. The trajectory shows the agent verifying that the curated output contained 28 subjects, and it used the subject ID stored in the NWB metadata rather than inferring mouse identity from filenames.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as exactly one session. Session order follows the sorted file list, and per-session metadata are stored in `session_info`, including the NWB `identifier` and source filename.

ii. 
```python
paths = sorted(glob.glob(os.path.join(data_dir, "sub-*", "*.nwb")))
...
for path in paths:
    session = convert_session(path)
```

```python
identifier = _scalar_string(nwb["identifier"])
...
"session_info": {
    "identifier": identifier,
    "source_file": os.path.relpath(path, Path(__file__).parent),
    ...
},
```

iii. In the trajectory, the agent explicitly reasoned that 174 files were present, one session lacked QC labels, and the remaining 173 files corresponded to the curated session set.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table, `intervals/trials`. The script assumes one go cue per trial, checks that `len(go_times)` equals the number of trial rows, and then filters trials by boolean masks rather than re-deriving trial boundaries from event streams.

ii. 
```python
trials = nwb["intervals/trials"]
go_times = nwb[
    "acquisition/BehavioralEvents/go_start_times/timestamps"
][:]
if len(go_times) != len(trials["id"]):
    raise ValueError(f"{path}: go cue and trial counts differ")
```

iii. The trajectory shows the agent inspecting event streams and using the go-cue count as the consistency check for trial structure.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only if all retained units mark that trial as valid in `units/is_good_trials`, the trial is not an `auto_water` or `free_water` assistance trial, and the binned neural array for that trial is not entirely zero. Sessions with fewer than 2 retained trials are dropped.

ii. 
```python
unit_trial_validity = units["is_good_trials"][:][good_indices]
n_ephys_trials = unit_trial_validity.shape[1]
...
neural_valid = np.zeros(len(go_times), dtype=bool)
neural_valid[:n_ephys_trials] = np.all(unit_trial_validity, axis=0)
```

```python
assistance = (trials["auto_water"][:] != 0) | (trials["free_water"][:] != 0)
keep = neural_valid & ~assistance
...
firing_rates_all = _bin_good_units(units, good_indices, go_times)
neural_nonzero = np.any(firing_rates_all != 0, axis=(1, 2))
empty_neural_trials = neural_valid & ~neural_nonzero
keep &= neural_nonzero
...
if len(kept_indices) < 2:
    return None
```

iii. The trajectory shows that this decision changed over time. Early on, the agent said it would retain all behavioral trial types except the uncurated session. Later, after inspecting `is_good_trials` patterns and finding sessions whose behavioral tables extended past valid ephys coverage, it patched the script to require `is_good_trials`, then added an extra exclusion for trials with no population spikes at all.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index`, with go-cue timestamps from `BehavioralEvents/go_start_times/timestamps` used to define the per-trial windows. Only units passing the script’s QC filter (`classification == "good"` and non-empty `anno_name`) are included.

ii. 
```python
classification = units["classification"]
class_values = classification.asstr()[:]
annotations = units["anno_name"].asstr()[:]
annotation_present = np.array([bool(x.strip()) for x in annotations])
good = (class_values == "good") & annotation_present
good_indices = np.flatnonzero(good)
```

```python
all_spikes = units["spike_times"][:]
spike_ends = units["spike_times_index"][:].astype(np.int64)
spike_starts = np.r_[0, spike_ends[:-1]]
window_starts = go_times + OFF_START
window_ends = go_times + OFF_END
```

iii. In the trajectory, the agent repeatedly focused on classifier-approved units and verified the count of 69,453 retained units. It treated spike times as the canonical raw neural signal.

## 2-b. How is the `neural` data processed?

i. The script converts spike times into firing rates in 50 ms non-overlapping bins. For each retained unit, spikes are assigned to trials by comparing spike times to go-cue-aligned trial windows, then assigned to time bins within each trial, counted with `np.bincount`, and divided by bin width to produce spikes/s. No smoothing or normalization is applied.

ii. 
```python
for out_unit, unit_index in enumerate(good_indices):
    spikes = all_spikes[spike_starts[unit_index] : spike_ends[unit_index]]
    trial_index = np.searchsorted(window_starts, spikes, side="right") - 1
    ...
    time_index = np.floor(
        (spikes - window_starts[trial_index]) / BIN_S + 1e-12
    ).astype(np.int64)
    ...
    counts = np.bincount(flat_index, minlength=n_trials * N_TIME)
    rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_S
```

iii. The trajectory shows the agent inspecting the reference preprocessing code and then implementing a direct binned-rate conversion in the requested 50 ms bins. It did not leave a separate textual argument for smoothing or normalization, and none was applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `units/classification == "good"` and `units/anno_name` is non-empty. If the classification dataset is not a string type, the whole session is dropped. Sessions are also dropped if no such units remain.

ii. 
```python
classification = units["classification"]
# One released session has NaNs here instead of classifier predictions.
if classification.dtype.kind not in "OSU":
    return None
class_values = classification.asstr()[:]
annotations = units["anno_name"].asstr()[:]
annotation_present = np.array([bool(x.strip()) for x in annotations])
good = (class_values == "good") & annotation_present
good_indices = np.flatnonzero(good)
if len(good_indices) == 0:
    return None
```

iii. The trajectory shows the agent identifying one session whose `classification` values were all `NaN`, then concluding that the curated dataset should contain 173 sessions and 69,453 “classifier-approved, histologically localized” units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural windows are aligned to go-cue onset. The script computes each trial’s start and end as `go_times + OFF_START` and `go_times + OFF_END`, then bins spikes in that absolute session-time window.

ii. 
```python
OFF_START = -2.5
OFF_END = 1.5
...
window_starts = go_times + OFF_START
window_ends = go_times + OFF_END
```

iii. The trajectory consistently treated the go cue as the alignment event and used NWB timestamps as a shared absolute clock across modalities.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 50 ms bins from -2.5 s to +1.5 s around the go cue, giving 80 time bins per trial. No further temporal rebinning is applied.

ii. 
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_S = 0.050
N_TIME = int(round((OFF_END - OFF_START) / BIN_S))
BIN_EDGES = OFF_START + np.arange(N_TIME + 1) * BIN_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. This follows the decoder task specification; the trajectory does not show any alternate binning choice.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives tone onset from `acquisition/BehavioralEvents/presample_stop_times/timestamps`, then subtracts that per-trial timestamp from the go-cue-aligned bin centers. It does not use `sample_start_times`.

ii. 
```python
# presample_stop is the first tone/sample onset and, unlike sample_start,
# has exactly one entry per trial even when an early lick replays epochs.
tone_onsets = nwb[
    "acquisition/BehavioralEvents/presample_stop_times/timestamps"
][:][keep]
time_from_tone = absolute_bin_times - tone_onsets[:, None]
```

iii. The trajectory shows why the agent made this choice: it inspected early-lick trials and found repeated `sample_start_times` within a single behavioral trial, then chose `presample_stop_times` because it has exactly one event per trial and therefore avoids replay duplicates.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The script computes absolute neural bin-center times and subtracts the chosen tone-onset timestamp from each bin center, yielding a continuous time-varying value in seconds for each bin.

ii. 
```python
absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]
...
time_from_tone = absolute_bin_times - tone_onsets[:, None]
```

iii. The trajectory does not show extra processing beyond the event-choice decision. The computation itself is a direct subtraction on the same bin grid used for the neural data.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by evaluating time-from-tone at the same absolute bin centers used for the neural firing rates: `kept_go + BIN_CENTERS`.

ii. 
```python
absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]
time_from_tone = absolute_bin_times - tone_onsets[:, None]
```

iii. The trajectory’s handling of all modalities assumes a shared absolute timestamp basis in the NWB files, so the same bin centers are reused instead of interpolating onto a different grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial-table columns `photostim_onset`, `photostim_duration`, and `start_time`.

ii. 
```python
onset = _numeric_or_nan(trials["photostim_onset"].asstr()[:])[keep]
duration = _numeric_or_nan(trials["photostim_duration"].asstr()[:])[keep]
trial_starts = trials["start_time"][:][keep]
stimulation_start = trial_starts + onset
stimulation_end = stimulation_start + duration
```

iii. In the trajectory, the agent inspected the distributions of photostimulation onset and duration strings and saw that most trials contained `"N/A"` while stimulated trials had numeric onset/duration values.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The script parses string-valued onset/duration columns, converting `"N/A"` to `NaN`. It then converts onset/duration into absolute stimulation start and end times and marks each neural bin center as stimulated or not, producing a binary time series.

ii. 
```python
def _numeric_or_nan(values: np.ndarray) -> np.ndarray:
    return np.array(
        [np.nan if value == "N/A" else float(value) for value in values],
        dtype=np.float64,
    )
```

```python
photostim = (
    (absolute_bin_times >= stimulation_start[:, None])
    & (absolute_bin_times < stimulation_end[:, None])
)
```

iii. The trajectory shows the agent explicitly checking that non-stim trials are encoded as `"N/A"` and stimulated trials carry a fixed duration, which matches the NaN-based implementation.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI places photostimulation on the same absolute session-time axis as the neural bins and compares the absolute bin centers against the absolute stimulation interval.

ii. 
```python
absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]
stimulation_start = trial_starts + onset
stimulation_end = stimulation_start + duration
photostim = (
    (absolute_bin_times >= stimulation_start[:, None])
    & (absolute_bin_times < stimulation_end[:, None])
)
```

iii. The trajectory does not show a separate alignment argument here; the code directly expresses stimulation on the same timestamp basis as neural firing rates.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not taken from raw lick event streams. Instead, it is derived from `trial_instruction` and `outcome`: the instructed side plus whether the trial was a hit, miss, or ignore.

ii. 
```python
instruction = trials["trial_instruction"].asstr()[:][keep]
outcome_text = trials["outcome"].asstr()[:][keep]
...
choice = np.full(len(kept_indices), 2, dtype=np.int8)  # no lick
responded = outcome_text != "ignore"
reported_left = ((outcome_text == "hit") & (instruction == "left")) | (
    (outcome_text == "miss") & (instruction == "right")
)
choice[responded & reported_left] = 0
choice[responded & ~reported_left] = 1
```

iii. The trajectory shows the agent explicitly comparing raw lick-event-derived labels against `trial_instruction` plus `outcome`, finding mismatches, and concluding that the trial-table outcome/instruction combination was the more robust source for decoder choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived choice is encoded as `0 = left`, `1 = right`, `2 = no lick`, then repeated across all 80 bins in the output tensor for that trial.

ii. 
```python
choice = np.full(len(kept_indices), 2, dtype=np.int8)  # no lick
...
choice[responded & reported_left] = 0
choice[responded & ~reported_left] = 1
```

```python
session_outputs.append(
    np.vstack(
        [
            np.full(N_TIME, choice[trial_index], dtype=np.int8),
            np.full(N_TIME, outcome[trial_index], dtype=np.int8),
            np.full(N_TIME, early[trial_index], dtype=np.int8),
            tongue[trial_index],
        ]
    )
)
```

iii. The trajectory indicates that the agent wanted to avoid noisy raw lick streams and keep one categorical trial-level choice value per trial, then broadcast it across time so all outputs share one `(n_output, n_time)` shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table column `outcome`.

ii. 
```python
outcome_text = trials["outcome"].asstr()[:][keep]
```

iii. The trajectory does not show any extra derivation step here; the outcome is already explicitly stored in the NWB trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped to `0 = ignore`, `1 = miss`, `2 = hit`, then repeated across all time bins for each trial.

ii. 
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_text], dtype=np.int8)
```

```python
np.full(N_TIME, outcome[trial_index], dtype=np.int8)
```

iii. The AI followed the requested categorical decoder-output format and made the per-trial outcome time-constant across bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table column `early_lick`.

ii. 
```python
early_text = trials["early_lick"].asstr()[:][keep]
```

iii. The trajectory includes summary counts for early vs non-early trials, showing that the agent inspected this field as a direct trial annotation.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The script converts the trial-table strings into a binary category with `1` for `"early"` and `0` otherwise, then repeats that per-trial value across all 80 bins.

ii. 
```python
early = (early_text == "early").astype(np.int8)
```

```python
np.full(N_TIME, early[trial_index], dtype=np.int8)
```

iii. The AI treated early lick as a trial-level categorical output required by the decoder task rather than as an event stream to reconstruct.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. The code uses column 1 as tongue `y` and column 2 as DeepLabCut likelihood, together with the tracking timestamps.

ii. 
```python
tracking = behavior["Camera0_side_TongueTracking"]
tracking_data = tracking["data"][:, 1:3]  # y, DeepLabCut likelihood
tracking_times = tracking["timestamps"][:]
y_all = tracking_data[:, 0]
likelihood_all = tracking_data[:, 1]
```

iii. The trajectory shows that the agent checked that tracking/video data existed for all sessions and inspected likelihood values before choosing its visibility rule.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI keeps only frames with finite `y`, finite likelihood, and likelihood at least `0.90`. It computes the session’s 40th and 60th percentiles from all visible raw `y` samples, not from 50 ms bin means. For each neural bin, it samples the most recent camera frame at or before the bin center and classifies that single sampled `y` value.

ii. 
```python
TONGUE_LIKELIHOOD_CUTOFF = 0.90
...
visible_all = (
    np.isfinite(y_all)
    & np.isfinite(likelihood_all)
    & (likelihood_all >= TONGUE_LIKELIHOOD_CUTOFF)
)
...
percentile_40, percentile_60 = np.percentile(y_all[visible_all], [40, 60])
```

```python
frame_index = np.searchsorted(
    tracking_times, absolute_bin_times.ravel(), side="right"
) - 1
...
sampled = tracking_data[safe_index]
y = sampled[:, 0]
likelihood = sampled[:, 1]
visible = (
    in_range
    & np.isfinite(y)
    & np.isfinite(likelihood)
    & (likelihood >= TONGUE_LIKELIHOOD_CUTOFF)
)
```

iii. The trajectory shows that the agent inspected the likelihood distribution and visible-frame fraction, then encoded a stricter `0.90` visibility cutoff and raw-frame percentile thresholds. It did not leave a fuller written justification beyond those probes and code comments.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses four categories: `0` below the session 40th percentile, `1` between the 40th and 60th percentiles, `2` above the 60th percentile, and `3` for not visible. The percentiles are computed from visible raw frame values over the full session.

ii. 
```python
result = np.full(len(frame_index), 3, dtype=np.int8)
result[visible & (y < percentile_40)] = 0
result[visible & (y >= percentile_40) & (y <= percentile_60)] = 1
result[visible & (y > percentile_60)] = 2
```

iii. The trajectory justification is mostly implicit: the category layout matches the task spec, while the concrete thresholding rule comes from the code the agent wrote after inspecting tracking likelihood values.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Instead of bin-averaging all camera frames inside each 50 ms neural bin, the AI aligns tongue output by taking the latest video frame at or before each neural bin center and classifying that single sampled frame.

ii. 
```python
absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]
...
frame_index = np.searchsorted(
    tracking_times, absolute_bin_times.ravel(), side="right"
) - 1
...
return (
    result.reshape(absolute_bin_times.shape),
    float(percentile_40),
    float(percentile_60),
    float(np.mean(visible_all)),
)
```

iii. The trajectory does not include a separate prose defense of this alignment choice; the decision is visible in `_tongue_classes`, which uses neural bin centers as sample times on the camera timestamp axis.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several missing-data cases by exclusion or explicit categories. A session is dropped if `classification` is non-string/NaN. Trials are excluded if `is_good_trials` marks them invalid, if assistance-trial flags are set, or if the binned population activity is entirely zero. Tongue samples with low-confidence tracking are treated as not visible and mapped to class `3`. Sessions with fewer than 2 retained trials are dropped.

ii. 
```python
if classification.dtype.kind not in "OSU":
    return None
...
good_indices = np.flatnonzero(good)
if len(good_indices) == 0:
    return None
```

```python
neural_valid[:n_ephys_trials] = np.all(unit_trial_validity, axis=0)
...
keep = neural_valid & ~assistance
...
empty_neural_trials = neural_valid & ~neural_nonzero
keep &= neural_nonzero
...
if len(kept_indices) < 2:
    return None
```

```python
result = np.full(len(frame_index), 3, dtype=np.int8)
```

iii. The trajectory shows the agent finding one all-NaN QC session and later discovering that some nominally valid sessions still had retained all-zero trials, which prompted the extra empty-neural-trial exclusion. Low-confidence tongue frames were handled in-code as a separate “not visible” class.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are reading large NWB arrays (`spike_times`, tracking data), binning spikes in `_bin_good_units` with a per-unit loop, and writing the large output pickle.

ii. 
```python
all_spikes = units["spike_times"][:]
spike_ends = units["spike_times_index"][:].astype(np.int64)
...
for out_unit, unit_index in enumerate(good_indices):
    ...
    counts = np.bincount(flat_index, minlength=n_trials * N_TIME)
```

```python
tracking_data = tracking["data"][:, 1:3]
tracking_times = tracking["timestamps"][:]
```

```python
with open(output_path, "wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In the trajectory, the agent checked the 50 GB source dataset size, monitored system resources, and later reported a 10.8 GiB output pickle, which supports the conclusion that I/O and spike binning dominate runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main explicit loop is the per-unit neural binning loop in `_bin_good_units`. There is also a per-trial Python loop that assembles `session_neural`, `session_inputs`, and `session_outputs`. The tongue alignment itself is already vectorized over all bin centers.

ii. 
```python
for out_unit, unit_index in enumerate(good_indices):
    spikes = all_spikes[spike_starts[unit_index] : spike_ends[unit_index]]
    ...
```

```python
session_inputs = []
session_outputs = []
session_neural = []
for trial_index in range(len(kept_indices)):
    session_neural.append(firing_rates[trial_index])
    session_inputs.append(
        np.stack(
            [time_from_tone[trial_index], photostim[trial_index]], axis=0
        ).astype(np.float32)
    )
    session_outputs.append(
        np.vstack(
            [
                np.full(N_TIME, choice[trial_index], dtype=np.int8),
                np.full(N_TIME, outcome[trial_index], dtype=np.int8),
                np.full(N_TIME, early[trial_index], dtype=np.int8),
                tongue[trial_index],
            ]
        )
    )
```

iii. The trajectory does not show an explicit vectorization discussion. The script uses simple loops where the data are ragged or being converted into per-trial Python lists.

## 10-c. What processing does the code repeat multiple times?

i. The script does not re-open files or recompute whole sessions, but it does compute `firing_rates_all` for all behavior trials before applying the final empty-neural-trial exclusion. That means some trials are fully binned and then discarded.

ii. 
```python
firing_rates_all = _bin_good_units(units, good_indices, go_times)
...
neural_nonzero = np.any(firing_rates_all != 0, axis=(1, 2))
empty_neural_trials = neural_valid & ~neural_nonzero
keep &= neural_nonzero
...
firing_rates = firing_rates_all[keep]
```

iii. The trajectory shows that this was introduced as a bug fix: after finding retained all-zero trials, the agent added the `neural_nonzero` filter without restructuring earlier filtering to avoid binning those trials in the first place.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest extra work is computing full-session debug/metadata statistics that are not part of the decoder inputs/outputs, and binning trials into `firing_rates_all` before later throwing some of them away via the empty-neural-trial filter.

ii. 
```python
"session_info": {
    "identifier": identifier,
    "source_file": os.path.relpath(path, Path(__file__).parent),
    "source_trial_count": int(len(keep)),
    "ephys_trial_count": int(n_ephys_trials),
    "retained_trial_count": int(np.sum(keep)),
    "excluded_no_ephys_or_unit_invalid_trials": int(np.sum(~neural_valid)),
    "excluded_empty_neural_trials": int(np.sum(empty_neural_trials)),
    "excluded_auto_or_free_water_trials": int(np.sum(neural_valid & assistance)),
    "classifier_good_units": int(len(good_indices)),
    "tongue_y_percentile_40": p40,
    "tongue_y_percentile_60": p60,
    "tongue_visible_frame_fraction": visible_fraction,
},
```

```python
firing_rates_all = _bin_good_units(units, good_indices, go_times)
...
keep &= neural_nonzero
```

iii. The trajectory indicates these fields were added to support debugging and verification rather than to feed the decoder. The code retains them in metadata, but they are not used by downstream decoding.
