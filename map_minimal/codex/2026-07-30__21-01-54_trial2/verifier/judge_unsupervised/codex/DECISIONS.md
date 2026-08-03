# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every NWB session file under `/app/data` by globbing `sub-*/*.nwb`, sorting the paths, and opening each file with `h5py`. Within each file it reads the trial table, behavior event timestamps, unit tables, and tongue-tracking acquisition data, then converts the session into per-trial tensors before concatenating sessions into a subject-indexed dataset.

ii. `convert_data.py`:
```python
session_files = sorted(data_dir.glob("sub-*/*.nwb"))

for sess_path in tqdm(session_files, desc="Converting sessions"):
    with h5py.File(sess_path, "r") as f:
        trials = f["intervals"]["trials"]
        ...
        go_times = np.asarray(
            f["processing"]["behavior"]["BehavioralEvents"]["go_start_times"]["timestamps"][()]
        ).reshape(-1)
        sample_times = np.asarray(
            f["processing"]["behavior"]["BehavioralEvents"]["sample_start_times"]["timestamps"][()]
        ).reshape(-1)
        ...
        tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
        tongue_t = np.asarray(tongue_group["timestamps"][()])
        tongue_y = np.asarray(tongue_group["data"][()])[:, 1].astype(np.float32)
```

iii. The trajectory shows the agent first inspecting the NWB layout and then deciding to convert directly from NWB instead of rebuilding the intermediate MATLAB-exported structures used by the reference code. `CONVERSION_NOTES.md` also describes a full-dataset pass over all NWB files and reports the resulting dataset-wide counts.

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB field `general/subject/subject_id`. The converter maintains a `subject_ids` list, assigns each unique ID a stable integer index, and stores each converted session under that subject index.

ii. `convert_data.py`:
```python
subject_id = _decode_string(f["general"]["subject"]["subject_id"][()])
if subject_id not in subject_ids:
    subject_ids.append(subject_id)
subject_idx = subject_ids.index(subject_id)
...
subject_index.append(subject_idx)
subject_session_keys.append((subject_idx, session_id))
```

iii. The trajectory contains a full-dataset scan reporting 28 unique subjects, and `CONVERSION_NOTES.md` repeats that count. There is no sign of any subject-level inference beyond using the NWB subject identifier directly.

## 1-c. How are the data split into sessions?

i. The agent treats each NWB file as one session. Session identity is taken from the filename stem. Sessions are skipped if they contain no classifier-good units, or if fewer than 2 trials remain after trial filtering and the final all-zero-neural sanity filter.

ii. `convert_data.py`:
```python
session_id = sess_path.stem
...
good_units_mask = unit_class == "good"
if good_units_mask.sum() == 0:
    skipped_no_good_units += 1
    continue
...
keep_idx = np.flatnonzero(trial_keep)
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
...
if len(keep_idx) < 2:
    skipped_too_few_trials += 1
    continue
```

iii. In `CONVERSION_NOTES.md`, the agent explicitly documents dropping one NWB file with zero good units and reports 173 retained sessions. The trajectory also shows the agent checking whether it needed the paper’s behavioral session-selection rules; it concluded the released NWB sessions were already close to the paper set and did not implement the performance-based session filter.

## 1-d. How are the data split into trials?

i. Trials are split by rows of the NWB `intervals/trials` table. The converter assumes one `go_start_times` event per behavioral trial, verifies the counts match, and then selects a subset of trial indices with a boolean mask.

ii. `convert_data.py`:
```python
trials = f["intervals"]["trials"]
trial_start = np.asarray(trials["start_time"][()]).reshape(-1)
trial_stop = np.asarray(trials["stop_time"][()]).reshape(-1)
...
go_times = np.asarray(
    f["processing"]["behavior"]["BehavioralEvents"]["go_start_times"]["timestamps"][()]
).reshape(-1)
...
if len(go_times) != len(trial_start):
    raise RuntimeError(f"{session_id}: go cue count {len(go_times)} != trial count {len(trial_start)}")
...
keep_idx = np.flatnonzero(trial_keep)
```

iii. The trajectory shows the agent inspecting trial counts and reconciling them against event streams, especially because `sample_start_times` can include replayed events after early licks while `go_start_times` still tracks trial count cleanly.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps trials only if they are not `auto_water` and not `free_water`, and if they are within the electrophysiologically recorded trial span judged from `units/is_good_trials` and `units/obs_intervals`. It retains early-lick, ignore/no-response, and photostimulation trials. After building neural tensors, it also removes any remaining trials whose neural array is all zeros.

ii. `convert_data.py`:
```python
trial_keep = (auto_water == 0) & (free_water == 0)
...
recorded_trial_mask = np.zeros(len(trial_start), dtype=bool)
recorded_trial_mask[first_start_idx:first_end_idx] = True
common_good = np.all(is_good_trials[good_units_mask], axis=0)
recorded_trial_mask[first_start_idx:first_start_idx + len(common_good)] &= common_good
trial_keep &= recorded_trial_mask
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
removed_allzero_trials += int((~nonzero_trial_mask).sum())
session_neural = session_neural[nonzero_trial_mask]
```

iii. `CONVERSION_NOTES.md` justifies this choice directly: early lick, ignore, and photostim trials were intentionally retained because the requested outputs include `Early lick`, `Outcome`, and `Photostimulation`. The trajectory also shows the agent comparing this against the paper code, which more often defined “regular” trials by excluding early-lick, no-response, and stimulation trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from NWB unit-level spike times and unit metadata. The main raw sources are `units/spike_times`, `units/spike_times_index`, `units/classification`, `units/is_good_trials`, `units/obs_intervals`, and `units/anno_name`.

ii. `convert_data.py`:
```python
spike_times_all = _extract_spike_times(f["units"])
obs_intervals = _extract_obs_intervals(f["units"])
unit_class = np.array([_decode_string(x) for x in f["units"]["classification"][()]])
...
is_good_trials = np.asarray(f["units"]["is_good_trials"][()])
...
areas = np.array([_decode_string(x) for x in f["units"]["anno_name"][()]])
```

iii. The trajectory shows the agent inspecting the NWB `units` group to map it onto the paper pipeline’s spike-sorted “good” units and per-trial recording-quality masks. `CONVERSION_NOTES.md` also reports the final total number of good units, confirming this was the intended source.

## 2-b. How is the `neural` data processed?

i. For each kept trial and each classifier-good unit, the agent bins spike times into 50 ms bins over a window from -2.5 s to +1.5 s around the go cue, then divides by bin width to convert counts to firing rates in Hz. No smoothing or additional normalization is applied.

ii. `convert_data.py`:
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_SIZE = 0.05
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE, dtype=np.float64)
...
session_neural = np.zeros((len(keep_idx), good_units_mask.sum(), N_BINS), dtype=np.float32)
for unit_pos, unit_idx in enumerate(good_unit_indices):
    spk = spike_times_all[unit_idx]
    for trial_pos, trial_idx in enumerate(keep_idx):
        edges = go_times[trial_idx] + BIN_EDGES
        lo = np.searchsorted(spk, edges[0], side="left")
        hi = np.searchsorted(spk, edges[-1], side="left")
        counts, _ = np.histogram(spk[lo:hi], bins=edges)
        session_neural[trial_pos, unit_pos] = counts.astype(np.float32) / BIN_SIZE
```

iii. The trajectory shows the agent checking the reference preprocessing code and concluding the target representation should be go-cue-aligned binned spike rates. It used direct NWB spike times rather than the intermediate `.mat` exports, but preserved the same alignment target and fixed-width binning required by the task instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered to those labeled `classification == "good"`. Trials are then filtered to the common subset marked as good across those good units and lying within the recorded observation interval range.

ii. `convert_data.py`:
```python
unit_class = np.array([_decode_string(x) for x in f["units"]["classification"][()]])
good_units_mask = unit_class == "good"
...
is_good_trials = np.asarray(f["units"]["is_good_trials"][()])
...
common_good = np.all(is_good_trials[good_units_mask], axis=0)
recorded_trial_mask[first_start_idx:first_start_idx + len(common_good)] &= common_good
trial_keep &= recorded_trial_mask
```

iii. `CONVERSION_NOTES.md` states that “good” follows the released spike-sorting QC labels, matching the Chen, Liu et al. QC paper. The trajectory also documents why `obs_intervals` were used: some sessions have a valid recording block that is not simply the first `N` trials, so the mask must be anchored to actual recording start and end times.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. For each trial the bin edges are defined by adding the fixed relative bin edges `[-2.5, 1.5]` to that trial’s `go_start_times` timestamp.

ii. `convert_data.py`:
```python
go_times = np.asarray(
    f["processing"]["behavior"]["BehavioralEvents"]["go_start_times"]["timestamps"][()]
).reshape(-1)
...
edges = go_times[trial_idx] + BIN_EDGES
counts, _ = np.histogram(spk[lo:hi], bins=edges)
```

iii. This follows the task instruction exactly. The trajectory also records the agent inspecting the paper code, which used `task_cue_time` as the go-cue alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms. The converter bins directly at that resolution and does not apply any second-stage rebinning or temporal smoothing.

ii. `convert_data.py`:
```python
BIN_SIZE = 0.05
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE, dtype=np.float64)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
N_BINS = len(BIN_CENTERS)
```

iii. The instructions explicitly require 50 ms bins, and the code implements that directly. The agent did not document any additional temporal aggregation in the notes or trajectory.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `sample_start_times` and `go_start_times`, with `start_time` used as a fallback constraint when resolving which sample onset belongs to which trial.

ii. `convert_data.py`:
```python
sample_times = np.asarray(
    f["processing"]["behavior"]["BehavioralEvents"]["sample_start_times"]["timestamps"][()]
).reshape(-1)
...
tone_onsets = _resolve_sample_onsets(sample_times, go_times, trial_start)
...
time_from_tone = (go_times[trial_idx] + BIN_CENTERS) - tone_onsets[trial_idx]
```

iii. The trajectory shows the agent investigating why there were more `sample_start_times` than trials and concluding that replayed sample epochs after aborted early-lick trials caused the mismatch. That motivated the explicit onset-resolution helper.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The agent first resolves one tone/sample onset per trial by taking the last `sample_start_times` event that occurs before each go cue, with a fallback that restricts the candidate onset to the interval between trial start and go cue. It then computes a continuous value at each neural bin center as `bin_time - tone_onset`.

ii. `convert_data.py`:
```python
def _resolve_sample_onsets(sample_times, go_times, trial_start):
    sample_times = np.asarray(sample_times, dtype=np.float64)
    resolved = np.empty_like(go_times, dtype=np.float64)
    for i, go_t in enumerate(go_times):
        prev = sample_times[sample_times < go_t]
        if prev.size == 0:
            raise RuntimeError(f"No sample onset preceding go cue for trial {i}")
        candidate = prev[-1]
        if candidate < trial_start[i]:
            in_trial = sample_times[(sample_times >= trial_start[i]) & (sample_times < go_t)]
            if in_trial.size == 0:
                raise RuntimeError(f"No in-trial sample onset for trial {i}")
            candidate = in_trial[-1]
        resolved[i] = candidate
    return resolved
...
time_from_tone = (go_times[trial_idx] + BIN_CENTERS) - tone_onsets[trial_idx]
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as handling replayed sample epochs. The trajectory records the same reasoning after the agent inspected event counts and sample/go relationships in several sessions.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same 50 ms bin centers as the neural tensor, on the same go-cue-aligned window. Each trial therefore has one scalar time-from-tone value per neural bin.

ii. `convert_data.py`:
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
...
time_from_tone = (go_times[trial_idx] + BIN_CENTERS) - tone_onsets[trial_idx]
session_inputs[trial_pos, :, 0] = time_from_tone.astype(np.float32)
```

iii. The instructions require all fields to share the same trial-by-time structure. The agent’s notes and final dataset schema both reflect that shared alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial table columns `photostim_onset` and `photostim_duration`. The code also parses `photostim_power`, but that value is not used in the final input feature.

ii. `convert_data.py`:
```python
photostim_onset = np.array([_maybe_float(x) for x in trials["photostim_onset"][()]], dtype=np.float64)
photostim_duration = np.array([_maybe_float(x) for x in trials["photostim_duration"][()]], dtype=np.float64)
photostim_power = np.array([_maybe_float(x) for x in trials["photostim_power"][()]], dtype=np.float64)
```

iii. The trajectory shows the agent verifying from the NWB trial table that these values are present and that `photostim_onset` is measured relative to trial start. The notes then state that this interpretation was used in the converter.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The agent creates a binary per-bin indicator. If onset and duration are finite, it converts the trial-relative stimulation interval to absolute time using `trial_start`, then marks bins whose centers fall within `[stim_start, stim_stop)`. If onset or duration is missing, the feature stays all zeros.

ii. `convert_data.py`:
```python
stim_on = np.zeros(N_BINS, dtype=np.float32)
onset = photostim_onset[trial_idx]
duration = photostim_duration[trial_idx]
if np.isfinite(onset) and np.isfinite(duration):
    stim_start = trial_start[trial_idx] + onset
    stim_stop = stim_start + duration
    abs_centers = go_times[trial_idx] + BIN_CENTERS
    stim_on[(abs_centers >= stim_start) & (abs_centers < stim_stop)] = 1.0
session_inputs[trial_pos, :, 1] = stim_on
```

iii. `CONVERSION_NOTES.md` says the agent interpreted `photostim_onset` as trial-relative. The trajectory supports that by showing a direct inspection of trial values around 2.0 s plus 0.5 s duration, which only make sense in trial time rather than go-cue-relative time.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned by evaluating the stimulation interval against the same absolute bin centers used for the neural tensor, namely `go_time + BIN_CENTERS`. This yields one binary stimulation value per neural bin.

ii. `convert_data.py`:
```python
abs_centers = go_times[trial_idx] + BIN_CENTERS
stim_on[(abs_centers >= stim_start) & (abs_centers < stim_stop)] = 1.0
session_inputs[trial_pos, :, 1] = stim_on
```

iii. The notes present this as part of the general rule that all inputs and outputs are written on the same go-cue-aligned time grid as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent derives this output from the trial table field `trial_instruction`, not from an actual lick-response variable. It effectively uses left/right trial type as a proxy for lick direction choice.

ii. `convert_data.py`:
```python
trial_instruction = np.array([_decode_string(x) for x in trials["trial_instruction"][()]])
...
choice_val = 0 if trial_instruction[trial_idx] == "left" else 1
```

iii. `CONVERSION_NOTES.md` explicitly states that this was intentional: the agent says it encoded “choice” from `trial_instruction` to match the paper code’s left/right trial label. The trajectory likewise shows the agent finding reference code that worked with `trial_type` labels rather than actual lick-side behavior.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code maps `left -> 0` and every non-left trial to `1` (effectively right), then repeats that scalar over all 80 time bins for the trial.

ii. `convert_data.py`:
```python
choice_val = 0 if trial_instruction[trial_idx] == "left" else 1
...
session_outputs[trial_pos, :, 0] = choice_val
```

iii. The notes justify the use of `trial_instruction` as matching the reference decoder labels. The trajectory does not show any attempt to find a separate actual lick-direction variable in NWB.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from the trial table field `outcome`.

ii. `convert_data.py`:
```python
outcome = np.array([_decode_string(x) for x in trials["outcome"][()]])
```

iii. The notes and trajectory both treat outcome as an explicit NWB trial variable that can be carried through directly, without reconstruction from other fields.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent maps string labels to integers with `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats that category across all time bins within the trial.

ii. `convert_data.py`:
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
...
session_outputs[trial_pos, :, 1] = outcome_map[outcome[trial_idx]]
```

iii. This mapping is documented implicitly by the code and reflected in the final metadata. There is no contradictory justification elsewhere in the notes or trajectory.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the trial table field `early_lick`.

ii. `convert_data.py`:
```python
early_lick = np.array([_decode_string(x) for x in trials["early_lick"][()]])
```

iii. The notes explain that early-lick trials were retained precisely because this output variable was requested. The trajectory also shows the agent contrasting that decision with the reference code’s usual regular-trial filtering.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code binarizes the string field with `no early -> 0` and anything else to `1`, then repeats that scalar across all time bins of the trial.

ii. `convert_data.py`:
```python
early_val = 0 if early_lick[trial_idx] == "no early" else 1
...
session_outputs[trial_pos, :, 2] = early_val
```

iii. The notes justify retention of the variable, but they do not describe any more elaborate processing. The code therefore embodies the full decision.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the acquisition time series `BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically the second column of the `data` array together with its timestamps.

ii. `convert_data.py`:
```python
tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_t = np.asarray(tongue_group["timestamps"][()])
tongue_y = np.asarray(tongue_group["data"][()])[:, 1].astype(np.float32)
```

iii. The trajectory shows the agent inspecting the marker-alignment code from the reference repository and identifying `tongue_y` as the relevant marker channel. The converter then maps that onto the NWB camera time series.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent aligns tongue y to each trial by scanning each 50 ms bin and taking the last tongue camera sample whose timestamp falls inside that bin. If a bin has no sample, it leaves the value at 0. These aligned continuous values are then discretized into 3 categories.

ii. `convert_data.py`:
```python
def _bin_tongue_y(tongue_t, tongue_y, edges):
    out = np.zeros(len(edges) - 1, dtype=np.float32)
    idx = np.searchsorted(tongue_t, edges[0], side="left")
    for b in range(len(out)):
        left, right = edges[b], edges[b + 1]
        j = np.searchsorted(tongue_t, right, side="left")
        if j > idx:
            out[b] = tongue_y[j - 1]
        idx = j
    return out
...
tongue_vals = _bin_tongue_y(tongue_t, tongue_y, edges)
```

iii. The trajectory includes the reference `align_markers.py`, which also used the last marker sample in each time interval rather than interpolation. The agent appears to have adapted that logic to the required 50 ms bins, though it did not separately justify the choice to leave missing bins at zero.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The agent computes the 40th and 60th percentiles of the session-wide tongue y trace, then thresholds each aligned bin value into category 0, 1, or 2. Values below `q40` stay 0, values at or above `q40` become 1, and values strictly above `q60` become 2.

ii. `convert_data.py`:
```python
q40, q60 = np.nanpercentile(tongue_y, [40, 60])
...
tongue_cat = np.zeros(N_BINS, dtype=np.int16)
tongue_cat[tongue_vals >= q40] = 1
tongue_cat[tongue_vals > q60] = 2
session_outputs[trial_pos, :, 3] = tongue_cat
```

iii. This follows the explicit task instruction to use session-specific 40th and 60th percentile thresholds. The notes restate that this was the chosen discretization rule.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y is aligned on the same go-cue-centered 50 ms bins as the neural data. For each kept trial, the code forms `edges = go_time + BIN_EDGES`, extracts one tongue y value per bin, and writes the discretized categories into the output tensor for that same trial-by-time grid.

ii. `convert_data.py`:
```python
edges = go_times[trial_idx] + BIN_EDGES
...
tongue_vals = _bin_tongue_y(tongue_t, tongue_y, edges)
...
session_outputs[trial_pos, :, 3] = tongue_cat
```

iii. The instructions require all converted modalities to be aligned to go cue onset on the same time base. The trajectory shows the agent matching this against the paper’s go-cue-centered marker alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several irregularities defensively. It converts string placeholders such as `N/A` to `NaN` for optional photostimulation fields, resolves ambiguous tone/sample onsets with a fallback restricted to the current trial, skips sessions with zero good units, removes any trial with all-zero neural data after conversion, and skips sessions left with fewer than 2 trials. Missing tongue samples inside a bin are implicitly left as 0.

ii. `convert_data.py`:
```python
def _maybe_float(x):
    s = _decode_string(x)
    if s in {"", "N/A", "nan", "NaN"}:
        return np.nan
    return float(s)
...
if candidate < trial_start[i]:
    in_trial = sample_times[(sample_times >= trial_start[i]) & (sample_times < go_t)]
    if in_trial.size == 0:
        raise RuntimeError(f"No in-trial sample onset for trial {i}")
...
if good_units_mask.sum() == 0:
    skipped_no_good_units += 1
    continue
...
out = np.zeros(len(edges) - 1, dtype=np.float32)
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
...
if len(keep_idx) < 2:
    skipped_too_few_trials += 1
    continue
```

iii. `CONVERSION_NOTES.md` documents the dropped zero-good-unit session and the final removal of two residual all-zero-neural trials. The trajectory explains why the tone-onset fallback was added, after the agent found replayed sample epochs and mismatched trial-support arrays in some sessions.

## 10-a. What are the most time-consuming steps of the code?

i. The slowest work is the nested unit-by-trial spike binning loop, because it bins every good unit separately for every kept trial. Secondary costs come from the per-trial loop that computes time-from-tone, photostimulation, tongue alignment, and repeated scalar output filling.

ii. `convert_data.py`:
```python
for unit_pos, unit_idx in enumerate(good_unit_indices):
    spk = spike_times_all[unit_idx]
    for trial_pos, trial_idx in enumerate(keep_idx):
        edges = go_times[trial_idx] + BIN_EDGES
        ...
        counts, _ = np.histogram(spk[lo:hi], bins=edges)
        session_neural[trial_pos, unit_pos] = counts.astype(np.float32) / BIN_SIZE
...
for trial_pos, trial_idx in enumerate(keep_idx):
    time_from_tone = (go_times[trial_idx] + BIN_CENTERS) - tone_onsets[trial_idx]
    ...
    tongue_vals = _bin_tongue_y(tongue_t, tongue_y, edges)
```

iii. The trajectory repeatedly refers to full-dataset conversion as the expensive step and logs complete conversion runs for validation. The code structure makes the main bottleneck clear even without profiler output.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike-binning inner loop over trials could be vectorized or moved to a batched sparse/bin-count implementation. The outer loop that constructs per-trial inputs and outputs could also be partly vectorized because time-from-tone, stimulation masks, and repeated scalar outputs are all computed trial by trial even though most operations are array-friendly. The loop that remaps `obs_intervals` to trial indices and the brain-region index assignment loop are smaller additional targets.

ii. `convert_data.py`:
```python
for unit_pos, unit_idx in enumerate(good_unit_indices):
    ...
    for trial_pos, trial_idx in enumerate(keep_idx):
        ...
for trial_pos, trial_idx in enumerate(keep_idx):
    ...
for gi, unit_idx in enumerate(good_unit_indices):
    brain_regions[gi, 0] = area_to_idx[areas[unit_idx]]
...
for i, start_t in enumerate(unit_starts):
    ...
for i, stop_t in enumerate(unit_stops):
    ...
```

iii. Neither the notes nor the trajectory claim these loops were optimized. The implementation is straightforward rather than performance-oriented, and the opportunities for vectorization are visible from the code path.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly recomputes `edges = go_time + BIN_EDGES` inside both the spike-binning loop and the per-trial feature loop, so each trial’s bin edges are formed multiple times. It also repeatedly constructs per-trial constant outputs by assigning the same scalar across all bins separately for each output variable.

ii. `convert_data.py`:
```python
for unit_pos, unit_idx in enumerate(good_unit_indices):
    ...
    for trial_pos, trial_idx in enumerate(keep_idx):
        edges = go_times[trial_idx] + BIN_EDGES
...
for trial_pos, trial_idx in enumerate(keep_idx):
    ...
    edges = go_times[trial_idx] + BIN_EDGES
    ...
    session_outputs[trial_pos, :, 0] = choice_val
    session_outputs[trial_pos, :, 1] = outcome_map[outcome[trial_idx]]
    session_outputs[trial_pos, :, 2] = early_val
```

iii. This repetition is not called out in the notes, but it follows directly from the implementation. The trajectory also suggests the agent prioritized a robust end-to-end conversion over micro-optimizing repeated calculations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code parses `photostim_power` but never uses it. It also computes and stores full time-resolved constant outputs for `choice`, `outcome`, and `early_lick` even though those variables are trial-level labels repeated at every time bin. Finally, it performs all input/output construction for some trials before later discarding trials with all-zero neural activity.

ii. `convert_data.py`:
```python
photostim_power = np.array([_maybe_float(x) for x in trials["photostim_power"][()]], dtype=np.float64)
...
session_outputs[trial_pos, :, 0] = choice_val
session_outputs[trial_pos, :, 1] = outcome_map[outcome[trial_idx]]
session_outputs[trial_pos, :, 2] = early_val
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
session_neural = session_neural[nonzero_trial_mask]
session_inputs = session_inputs[nonzero_trial_mask]
session_outputs = session_outputs[nonzero_trial_mask]
```

iii. `CONVERSION_NOTES.md` confirms that two all-zero-neural trials were only removed at the end as a sanity filter. The code therefore does some work on trials and fields that do not contribute meaningfully to downstream decoding.
