# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globbed all session files under `sub-*/*.nwb`, processed them one file at a time, and read the NWB contents directly with `h5py` rather than `pynwb`. Within each file it loads the `units` table, the `intervals/trials` table, `BehavioralEvents`, and `BehavioralTimeSeries/Camera0_side_TongueTracking`.

ii.
```python
def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))

...

with h5py.File(file_path, "r") as h5:
    units = h5["units"]
    trials = h5["intervals"]["trials"]
    go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials]
    tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
```

iii. In `CONVERSION_NOTES.md`, the AI says it switched from `PyNWB` to direct `h5py` reads to avoid object-construction overhead and warning spam while preserving the published NWB layout as the source of truth.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the folder name of each session file. The script strips the `sub-` prefix from the parent directory name, then later builds `subjects` as the sorted unique subject ids and `subject_idx` from those ids.

ii.
```python
def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id

...

subjects = sorted({r.subject_id for r in results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes justify this as a direct consequence of the DANDI directory layout: one `sub-<id>` folder per animal, with the folder id matching the subject id carried by the dataset.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order follows the sorted file list, and the session id stored in the output is the filename stem rather than `nwb.identifier`.

ii.
```python
def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))

def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id
```

iii. The notes explicitly state that the dataset is “one NWB file per session,” so the file boundary is used as the session boundary.

## 1-d. How are the data split into trials?

i. Trial structure comes from the NWB `intervals/trials` table. After choosing a subset of valid behavioral trial indices, the script slices all per-trial columns (`start_time`, `stop_time`, `trial_instruction`, `outcome`, `early_lick`, photostim fields, and go cues) with the same index array.

ii.
```python
trials = h5["intervals"]["trials"]
trial_start_all = trials["start_time"][:].astype(np.float64)
trial_stop_all = trials["stop_time"][:].astype(np.float64)

selected_trial_idx, session_obs_start, session_obs_stop = select_trial_indices(
    go_times_all=go_times_all,
    good_unit_obs_intervals=unit_obs_intervals,
)

trial_start = trial_start_all[selected_trial_idx]
trial_stop = trial_stop_all[selected_trial_idx]
trial_instruction = trial_instruction_all[selected_trial_idx]
outcome = outcome_all[selected_trial_idx]
go_times = go_times_all[selected_trial_idx]
```

iii. The notes describe this as using the behavioral trial table as the canonical trial definition, with subsequent restriction to trials that the AI considered neurally supported.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages. First, the script keeps only trials whose full `[-2.5, +1.5]` s go-aligned window lies inside the minimum start and maximum stop of the selected good units’ `obs_intervals`. Second, during neural binning it zeros unit-trial bins marked bad by `units/is_good_trials` when that matrix width matches the selected trials; afterward it drops any trial whose full neural tensor is all zeros. It does not explicitly exclude `free_water` trials.

ii.
```python
def select_trial_indices(
    go_times_all: np.ndarray,
    good_unit_obs_intervals: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    session_obs_start = float(np.min(good_unit_obs_intervals[:, 0]))
    session_obs_stop = float(np.max(good_unit_obs_intervals[:, 1]))
    full_window_mask = (
        (go_times_all + WINDOW_START_S >= session_obs_start)
        & (go_times_all + WINDOW_END_S <= session_obs_stop)
    )
    trial_idx = np.flatnonzero(full_window_mask)
    return trial_idx, session_obs_start, session_obs_stop

...

if uses_direct_is_good_trials:
    invalid_trials = ~is_good_trials[unit_pos]
else:
    valid_obs = (go_times + WINDOW_START_S >= obs_start) & (go_times + WINDOW_END_S <= obs_stop)
    invalid_trials = ~valid_obs
if np.any(invalid_trials):
    fr[invalid_trials] = 0.0

nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
```

iii. The notes justify this as retaining early-lick, miss, ignore, and photostimulation trials because those are decoder targets or inputs, while enforcing a stricter “full neural window must be supported” rule derived from `obs_intervals` and `is_good_trials`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units/spike_times` and `units/spike_times_index` for units whose `units/classification` is `"good"`. Trial go-cue timestamps define the bin edges used to convert those spike times into per-trial matrices.

ii.
```python
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")

spike_times = units["spike_times"][:].astype(np.float64)
spike_times_index = units["spike_times_index"][:]
spike_starts, spike_ends = get_ragged_row_bounds(spike_times_index)

abs_edges = go_times[:, None] + bin_edges_rel[None, :]
```

iii. The notes explicitly say the converter should use classifier-approved good units and bin raw spike times relative to go cue.

## 2-b. How is the `neural` data processed?

i. For each kept unit, the AI bins spike times into non-overlapping 50 ms bins over `[-2.5, 1.5)` s around go cue, computes spike counts by differencing `searchsorted` results at consecutive edges, divides by the bin width to obtain firing rates in Hz, and stores the result as `float16`.

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
    neural_session[:, unit_pos, :] = fr.astype(np.float16)
```

iii. The notes say the AI intentionally used vectorized `searchsorted` binning per unit for speed, and converted counts to Hz because the decoder task asked for 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Unit-level filtering keeps only units labeled `classification == "good"`. In addition, the AI applies per-unit or session-level trial validity masking from `units/is_good_trials` or `obs_intervals`, setting unsupported unit-trial firing-rate rows to zero. Sessions with no good units are skipped entirely.

ii.
```python
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")
if len(good_unit_idx) == 0:
    print(f"Skipping {session_id}: zero good units")
    return None

...

is_good_trials_raw = units["is_good_trials"][good_unit_idx, :n_recorded_trials].astype(bool)

...

if np.any(invalid_trials):
    fr[invalid_trials] = 0.0
```

iii. The notes justify `classification == good` as the intended classifier-based curation and describe the unit-trial masking as an NWB-specific validity rule needed to avoid using behavior trials that lacked neural support.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go cue. The script builds trial-specific absolute bin edges by adding the fixed relative edge vector to each trial’s go-cue timestamp, then bins spikes against those edges directly.

ii.
```python
def bin_edges_and_centers():
    edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S, dtype=np.float64)
    centers = edges[:-1] + BIN_WIDTH_S / 2.0
    return edges, centers

...

abs_edges = go_times[:, None] + bin_edges_rel[None, :]
flat_edges = abs_edges.reshape(-1)
```

iii. The notes repeatedly describe go cue as the common alignment anchor for neural, input, and output streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins from `-2.5` s to `+1.5` s around go cue. No additional temporal rebinning or smoothing is applied after that binning step.

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

iii. The notes say this was an intentional deviation from the paper’s own decoding binning because the task instructions explicitly required 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `BehavioralEvents/sample_start_times`, `trials/start_time`, `trials/stop_time`, and `go_start_times`. It finds sample-start events that fall within each trial window and uses the first such event as the tone onset; if none exist, it falls back to a fixed offset of `-1.85` s relative to go cue.

ii.
```python
EXPECTED_SAMPLE_ONSET_REL_GO = -1.85

sample_slice_starts, sample_slice_ends = event_slices_for_trials(sample_start_times, trial_start, trial_stop)

sample_onset_abs = np.empty(n_trials, dtype=np.float64)
for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    if len(events):
        sample_onset_abs[trial] = events[0]
    else:
        sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
```

iii. The notes justify the choice of the earliest in-trial sample event as preserving replay-induced timing shifts on early-lick trials, and describe the `-1.85` s fallback as a safeguard for missing events.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After selecting a per-trial tone onset, the script subtracts that onset from the fixed go-cue-centered bin centers. The resulting value is time since tone onset at each bin center.

ii.
```python
sample_onset_rel_go = sample_onset_abs - go_times

for trial in range(n_trials):
    inp = np.zeros((2, n_bins), dtype=np.float32)
    inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. The notes describe this as the simplest way to express tone-relative time on the same go-cue-centered time grid used everywhere else.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the exact same 50 ms bin centers used for neural activity. The signal differs across trials only by the per-trial tone onset shift.

ii.
```python
bin_edges_rel, bin_centers_rel = bin_edges_and_centers()

inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)

abs_edges = go_times[:, None] + bin_edges_rel[None, :]
```

iii. The notes justify this by using go cue as the common reference frame for all streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trials-table columns `photostim_onset` and `photostim_duration`, together with each trial’s `start_time` and go-cue timestamp.

ii.
```python
photostim_onset_all = decode_str_array(trials["photostim_onset"][:])
photostim_duration_all = decode_str_array(trials["photostim_duration"][:])

...

if str(photostim_onset[trial]) != "N/A":
    stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
    stim_rel_off = stim_rel_on + float(photostim_duration[trial])
```

iii. The notes say the trial table directly exposes the stimulation timing and that representing it as a time-varying input best matches the decoder specification.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI converts the onset and duration into a go-cue-relative interval and marks bins whose centers fall inside that interval as `1`; all others remain `0`.

ii.
```python
inp = np.zeros((2, n_bins), dtype=np.float32)

if str(photostim_onset[trial]) != "N/A":
    stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
    stim_rel_off = stim_rel_on + float(photostim_duration[trial])
    inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. The notes justify this as producing a binary on/off time series rather than only a per-trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is placed onto the same go-cue-relative axis as the neural bins by subtracting `go_time` from the absolute stimulation onset derived from `trial_start + photostim_onset`.

ii.
```python
stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
stim_rel_off = stim_rel_on + float(photostim_duration[trial])
inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. The notes describe go cue as the common alignment frame, so once the stimulation window is expressed relative to go cue it can be compared directly with the neural bin centers.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice primarily from `BehavioralEvents/left_lick_times` and `right_lick_times`, using trial boundaries and go time to determine which lick occurred first. If there is no post-go lick it falls back to the first lick anywhere in the trial, and if there is no lick at all it falls back to the instructed side from `trial_instruction`.

ii.
```python
def infer_choice_for_trial(
    trial_start: float,
    trial_stop: float,
    go_time: float,
    instruction: str,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> tuple[int, str]:
    ...
    if len(left_post) or len(right_post):
        return (0, "post_go_lick") if first_left < first_right else (1, "post_go_lick")
    ...
    if len(left_any) or len(right_any):
        return (0, "any_lick") if first_left < first_right else (1, "any_lick")

    return (0 if instruction == "left" else 1, "instruction_fallback")
```

iii. The notes justify this as trying to recover actual lick direction from behavior events, while still forcing every trial into a binary left/right choice for the decoder, including ignore trials with no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as a binary class, `0` for left and `1` for right, then repeated across all time bins for each trial. The output metadata therefore lists only two choice values.

ii.
```python
choice_trials = np.empty(n_trials, dtype=np.int8)
for trial in range(n_trials):
    choice_val, source = infer_choice_for_trial(...)
    choice_trials[trial] = choice_val

...

out = np.empty((4, n_bins), dtype=np.int8)
out[0] = choice_trials[trial]

...

"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_p40", "p40_to_p60", "gt_p60"],
],
```

iii. The notes say this fallback hierarchy was necessary because the decoder specification asked for choice on all trials, including trials with no response.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table column `outcome`.

ii.
```python
outcome_all = np.char.lower(decode_str_array(trials["outcome"][:]))
...
outcome = outcome_all[selected_trial_idx]
```

iii. The notes treat `outcome` as an already-curated categorical field exposed directly in the NWB trial table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped to integer labels `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeated across all time bins within each trial.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome_trials = np.array([outcome_map[str(x)] for x in outcome], dtype=np.int8)

...

out[1] = outcome_trials[trial]
```

iii. The notes say the mapping follows the decoder target categories directly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table column `early_lick`.

ii.
```python
early_lick_all = np.char.lower(decode_str_array(trials["early_lick"][:]))
...
early_lick = early_lick_all[selected_trial_idx]
```

iii. The notes justify retaining rather than excluding these trials because early lick itself is one of the requested decoder outputs.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The strings are mapped to integer labels `no early -> 0` and `early -> 1`, then repeated across all time bins within each trial.

ii.
```python
early_map = {"no early": 0, "early": 1}
early_trials = np.array([early_map[str(x)] for x in early_lick], dtype=np.int8)

...

out[2] = early_trials[trial]
```

iii. The notes say this preserves early-lick information instead of using the paper’s analysis-time exclusion mask.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI reads `BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically the `data[:, 1]` tongue-y coordinate and the matching `timestamps`. It also reads `data[:, 2]` likelihood but does not use it for the output construction.

ii.
```python
tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_values = tongue_group["data"][:].astype(np.float64)
tongue_y = tongue_values[:, 1]
tongue_likelihood = tongue_values[:, 2]
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. The notes say the converter should use side-camera tongue tracking and explicitly note a decision to use likelihood only for diagnostics, not for thresholding.

## 8-b. How is `output` *Tongue y-position* derived?

i. The AI bins tongue y by taking the last camera sample that falls in each 50 ms neural bin. It then pools all finite binned values from the kept trials in that session, computes the 40th and 60th percentiles, and discretizes each binned value relative to those thresholds. There is no averaging within bin and no visibility filtering by likelihood.

ii.
```python
def bin_tongue_y(...):
    abs_edges = go_times[:, None] + bin_edges_rel[None, :]
    start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
    end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1
    clipped_end = np.clip(end_idx, 0, len(tongue_y) - 1)
    valid = end_idx >= start_idx
    binned = np.full(end_idx.shape, np.nan, dtype=np.float32)
    binned[valid] = tongue_y[clipped_end[valid]].astype(np.float32)
    return binned, valid

...

valid_values = tongue_y_binned[np.isfinite(tongue_y_binned)]
tongue_p40 = float(np.percentile(valid_values, 40))
tongue_p60 = float(np.percentile(valid_values, 60))
```

iii. The notes justify the “last frame within bin” rule as matching the marker-alignment convention they believed the reference code used, and justify the percentile cutoffs as the required per-session discretization.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses three categories only: below the 40th percentile, between the 40th and 60th percentiles inclusive, and above the 60th percentile. Missing bins remain `0` because the array is initialized with zeros and there is no explicit “not visible” class.

ii.
```python
tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
tongue_discrete[tongue_y_binned > tongue_p60] = 2

...

"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_p40", "p40_to_p60", "gt_p60"],
],
```

iii. The notes frame this as following the requested 40th/60th percentile discretization, but they do not preserve the requested fourth “not visible” category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The alignment is done on the same go-cue-relative 50 ms bins as the neural data. For each trial and each bin, the script finds the last tongue sample before the bin’s right edge and assigns its y-value to that bin if at least one frame fell inside the bin.

ii.
```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1

valid = end_idx >= start_idx
binned[valid] = tongue_y[clipped_end[valid]].astype(np.float32)
```

iii. The notes justify this as using the same go-cue-centered time grid for both video and neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or problematic data are mostly handled by exclusion or fallback. Sessions with zero recorded trials or zero good units are skipped. Good units with empty `anno_name` cause an error. Missing per-trial sample events trigger a fixed `-1.85 s` fallback tone onset. Trials with all-zero neural activity after masking are dropped. If no finite tongue values remain after alignment, the session raises an error instead of emitting a special category.

ii.
```python
if n_recorded_trials == 0:
    print(f"Skipping {session_id}: zero recorded trials in units/is_good_trials")
    return None

if len(good_unit_idx) == 0:
    print(f"Skipping {session_id}: zero good units")
    return None

if np.any(brain_region_names == ""):
    raise ValueError(f"{session_id}: found kept good units with empty anno_name")

...

if len(events):
    sample_onset_abs[trial] = events[0]
else:
    sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO

...

if len(valid_values) == 0:
    raise ValueError(f"{session_id}: no valid tongue_y values after alignment")
```

iii. The notes argue that unsupported neural windows should be excluded rather than fabricated, while missing timing fields should fall back to a canonical task interval when possible.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified session I/O, spike-time loading and binning, and tongue-alignment work as the expensive parts. In code terms, the heaviest operations are reading large arrays from HDF5 and the per-unit `searchsorted` loop over all trial/bin edges.

ii.
```python
with h5py.File(file_path, "r") as h5:
    ...
    spike_times = units["spike_times"][:].astype(np.float64)
    tongue_values = tongue_group["data"][:].astype(np.float64)

...

for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
```

iii. The notes explicitly call out direct NWB/HDF5 reads, ragged spike extraction, and per-unit vectorized binning as the main runtime drivers.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops remain: per-trial loops for tone-onset selection, input assembly, choice inference, and output assembly, plus the per-unit neural loop. The tongue binning helper is partly vectorized already but still performs per-bin selection through `searchsorted`.

ii.
```python
for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    ...

for trial in range(n_trials):
    inp = np.zeros((2, n_bins), dtype=np.float32)
    ...

for trial in range(n_trials):
    choice_val, source = infer_choice_for_trial(...)
    ...

for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    ...
```

iii. The notes emphasize that the AI vectorized the heaviest inner dimension with `searchsorted`, but left several trial-wise loops in place for clarity.

## 10-c. What processing does the code repeat multiple times?

i. The code makes multiple separate per-trial passes over the same selected trial set: one to choose tone onsets, one to build inputs, one to infer choice, and one to package outputs. It also performs additional list and array filtering after dropping all-zero trials, repeating the same mask application across many arrays.

ii.
```python
for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    ...

for trial in range(n_trials):
    inp = np.zeros((2, n_bins), dtype=np.float32)
    ...

for trial in range(n_trials):
    choice_val, source = infer_choice_for_trial(...)
    ...

if dropped_zero_trials:
    neural_session = neural_session[nonzero_trial_mask]
    sample_onset_rel_go = sample_onset_rel_go[nonzero_trial_mask]
    ...
    input_trials = [trial for keep, trial in zip(nonzero_trial_mask, input_trials) if keep]
    output_trials = [trial for keep, trial in zip(nonzero_trial_mask, output_trials) if keep]
```

iii. The notes do not present this as a problem, but the implementation clearly revisits the same trial-aligned arrays multiple times.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script performs some extra work that is not needed for the saved decoder dataset: it reads `delay_start_times` and tongue likelihood even though those are only used for optional plots or diagnostics, builds `plot_payload` and figure outputs when requested, and computes extensive `stats` bookkeeping that is not stored in the final exported dataset.

ii.
```python
delay_start_times = h5["acquisition"]["BehavioralEvents"]["delay_start_times"]["timestamps"][:].astype(np.float64)
tongue_likelihood = tongue_values[:, 2]

...

if show_processing:
    ...
    plot_payload = {
        ...
        "delay_events_rel_go": delay_events,
        "tongue_likelihood_window": tongue_likelihood[tongue_window],
    }

stats = {
    "choice_source_counts": choice_source_counts,
    "sample_onset_fallbacks": sample_onset_fallbacks,
    "fraction_invalid_unit_trials": float(np.mean(~is_good_trials)),
    ...
}
```

iii. The notes describe some of this as deliberate validation and visualization support, but it is not necessary for the final converted arrays used by downstream decoding.
