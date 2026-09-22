# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks `/app/data` recursively, collects every `.nwb` file, then prefilters to a curated session list by opening each file with `h5py` and checking whether `units/classification` contains at least one `"good"` unit. Each curated file is then reopened and read by direct HDF5 paths for the trials table, behavioral events, behavioral time series, unit table, and electrode metadata.

ii. 
```python
def list_session_files() -> list[str]:
    files = []
    for dirpath, _, filenames in os.walk(DATA_ROOT):
        for name in filenames:
            if name.endswith(".nwb"):
                files.append(os.path.join(dirpath, name))
    return sorted(files)

def session_has_good_units(path: str) -> bool:
    with h5py.File(path, "r") as h5:
        cls = decode_vector(h5["units"]["classification"][:])
    return "good" in cls

def discover_curated_sessions() -> list[str]:
    files = list_session_files()
    curated = [path for path in files if session_has_good_units(path)]
    return curated
```

```python
with h5py.File(path, "r") as h5:
    trials = h5["intervals"]["trials"]
    ev = h5["acquisition"]["BehavioralEvents"]
    tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
    units = h5["units"]
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as a consequence of the provided source being NWB rather than the paper code's original `.mat` exports, so it chose to map NWB fields directly onto the paper concepts. In the trajectory it also justified the 173-session curated set as the NWB sessions with at least one classifier-approved good unit.

## 1-b. How are the data split into subjects (mice)?

i. The AI uses the parent folder name of each NWB file, e.g. `sub-440956`, as the subject ID. After per-session processing, it builds `subjects` as the sorted unique folder names and `subject_idx` as the index of each kept session into that list.

ii.
```python
def get_subject_id(path: str) -> str:
    return os.path.basename(os.path.dirname(path))
```

```python
subject = get_subject_id(path)
...
subjects = sorted({sess["subject"] for sess in session_dicts})
subject_to_idx = {sub: i for i, sub in enumerate(subjects)}
...
"subject_idx": np.array([subject_to_idx[sess["subject"]] for sess in session_dicts], dtype=np.int64),
```

iii. The notes say the dataset is organized as 28 subject folders named `sub-<id>/`, and the AI chose to preserve those folder identifiers directly as the mouse IDs.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. It derives a session ID from the filename by stripping the NWB suffix, and the output session order follows the sorted file list after the curated-session filter.

ii.
```python
def get_session_id(path: str) -> str:
    return os.path.basename(path).replace("_behavior+ecephys+ogen.nwb", "").replace("_behavior+ecephys.nwb", "")
```

```python
curated_sessions = discover_curated_sessions()
...
for path in curated_sessions:
    session_id = get_session_id(path)
    session_data, session_summary = process_session(path, make_plot=session_id in plot_ids)
```

iii. In the notes, the AI states that `/app/data` contains one NWB file per session, so it used the file boundary as the session boundary.

## 1-d. How are the data split into trials?

i. Trials are taken from `intervals/trials`, with one trial-table row treated as one behavioral trial. Trial-aligned event times such as `go_start_times` are read separately, and later filters keep or drop specific trial-table rows by index.

ii.
```python
trials = h5["intervals"]["trials"]
trial_start = np.asarray(trials["start_time"][:], dtype=np.float64)
trial_stop = np.asarray(trials["stop_time"][:], dtype=np.float64)
...
go_times = np.asarray(ev["go_start_times"]["timestamps"][:], dtype=np.float64)
```

```python
for i in range(len(trial_start)):
    if i not in recorded_trial_set:
        continue
    ...
    candidate_trial_idx.append(i)
```

iii. The notes describe the NWB trials table and behavioral event streams as the native session structure. In the trajectory, the AI says it fixed a zero-neural bug by using `obs_intervals` to decide which trials-table rows are actually covered by ephys.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials whose `(start_time, stop_time)` interval appears in the good units' `obs_intervals`, excludes `auto_water == 1` and `free_water == 1`, requires the full `[-2.5, 1.5]` decoder window to lie inside the tongue-tracking timestamps, requires a sample-start event before go within the trial, drops trials whose binned neural activity is all zero, and drops sessions with fewer than 2 surviving trials.

ii.
```python
coverage_obs = ragged_rows(obs_intervals_flat, obs_intervals_index, good_units[0])
recorded_trial_idx = map_obs_intervals_to_trial_indices(trial_start, trial_stop, coverage_obs)
recorded_trial_set = set(recorded_trial_idx.tolist())
```

```python
for i in range(len(trial_start)):
    if i not in recorded_trial_set:
        continue
    if auto_water[i] == 1 or free_water[i] == 1:
        continue
    go = float(go_times[i])
    edges_abs = go + BIN_EDGES_REL
    if edges_abs[0] < tongue_timestamps[0] or edges_abs[-1] > tongue_timestamps[-1]:
        continue
    sample_start = last_event_before(sample_start_times, float(trial_start[i]), go)
    if sample_start is None:
        continue
```

```python
if not np.any(neural):
    dropped_all_zero_neural += 1
    continue
...
if len(trial_keep) < 2:
    raise ValueError(f"{session_id}: fewer than 2 trials remain after neural validation")
```

iii. The notes and trajectory justify the `obs_intervals` filter as a fix for sessions where behavioral trials extend beyond ephys coverage. The notes also say the AI excluded `auto_water` and `free_water` as non-regular trials, kept early/ignore/photostim because the decoder needs them, and required tongue-window coverage because the decoder output uses tongue tracking.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units/classification`, `units/spike_times`, and `units/spike_times_index`, with `BehavioralEvents/go_start_times` providing the alignment event. `obs_intervals` is also used to decide which trials have usable ephys coverage.

ii.
```python
classification = decode_vector(units["classification"][:])
spike_flat = np.asarray(units["spike_times"][:], dtype=np.float64)
spike_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
obs_intervals_flat = np.asarray(units["obs_intervals"][:], dtype=np.float64)
obs_intervals_index = np.asarray(units["obs_intervals_index"][:], dtype=np.int64)
go_times = np.asarray(ev["go_start_times"]["timestamps"][:], dtype=np.float64)
```

iii. In the notes, the AI identifies `classification == "good"` as the closest NWB equivalent of the paper's good-unit list, and it treats `spike_times` as the raw neural signal to bin around the go cue.

## 2-b. How is the `neural` data processed?

i. For each kept trial and each good unit, the AI bins absolute spike times into 50 ms bins spanning `[-2.5, 1.5]` around the go cue and divides by bin width to convert counts to firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
def bin_spike_rates(spike_times_abs: np.ndarray, trial_edges_abs: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(spike_times_abs, trial_edges_abs, side="left")
    counts = np.diff(idx)
    return counts.astype(np.float32) / BIN_SIZE
```

```python
neural = np.empty((len(good_units), NBINS), dtype=np.float32)
for unit_row, spikes_abs in enumerate(unit_spike_times):
    neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

iii. The notes explicitly map this to the paper/reference-code practice of turning spike times into go-cue-centered firing rates, adapted to the decoder's required 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == "good"` and drops sessions with no such units. It does not threshold additional QC metrics. Separately, it restricts trials to the `obs_intervals`-covered subset and drops residual all-zero neural trials after binning.

ii.
```python
good_units = [i for i, c in enumerate(classification) if c == "good"]
if not good_units:
    raise ValueError(f"{session_id}: no classifier-good units")
```

```python
coverage_obs = ragged_rows(obs_intervals_flat, obs_intervals_index, good_units[0])
recorded_trial_idx = map_obs_intervals_to_trial_indices(trial_start, trial_stop, coverage_obs)
```

iii. In `CONVERSION_NOTES.md`, the AI argues that `classification == "good"` is the closest NWB analog of the paper's classifier-approved unit set and reproduces the paper-scale good-unit count much better than `unit_quality`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI uses go-cue onset as time zero. For each kept trial it adds a fixed vector of relative bin edges to that trial's absolute go time and bins spikes directly against those absolute edges.

ii.
```python
BIN_EDGES_REL = np.linspace(WINDOW_START, WINDOW_END, NBINS + 1, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

```python
go = float(go_times[trial_idx])
edges_abs = go + BIN_EDGES_REL
...
neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

iii. The notes repeatedly describe the converted dataset as go-cue aligned, and the trajectory says the AI treated the spikes and events as sharing one session-absolute clock after restricting to `obs_intervals`-covered trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 80 non-overlapping 50 ms bins from `-2.5 s` to `+1.5 s` around go cue onset. There is no extra temporal rebinning beyond directly binning spikes into those bins.

ii.
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_SIZE = 0.05
NBINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES_REL = np.linspace(WINDOW_START, WINDOW_END, NBINS + 1, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

iii. The notes state that the paper's 40 ms/3.4 ms processing was used as a consistency check, but the final dataset was adapted to the user-required 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `BehavioralEvents/sample_start_times`, `intervals/trials/start_time`, and `BehavioralEvents/go_start_times`. For each kept trial it chooses the last sample-start event between that trial's start and its go cue.

ii.
```python
sample_start_times = np.asarray(ev["sample_start_times"]["timestamps"][:], dtype=np.float64)
...
sample_start = last_event_before(sample_start_times, float(trial_start[i]), go)
```

iii. In the notes, the AI explains that early licks can replay the sample/delay epochs, so the correct tone onset is the last `sample_start` before that trial's go cue rather than the first one in the trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes the sample-start time relative to go cue for each trial, then fills every bin with the bin-center time measured relative to that sample onset. The result is a continuous time-varying signal in seconds.

ii.
```python
candidate_sample_onsets_rel.append(sample_start - go)
```

```python
tone_on_rel = float(candidate_sample_onsets_rel[kept_idx])
input_arr = np.empty((2, NBINS), dtype=np.float32)
input_arr[0] = (BIN_CENTERS_REL - tone_on_rel).astype(np.float32)
```

iii. The notes say this is the decoder-specific adaptation of the trial timing: the bins are go-cue centered, so the time-from-tone variable is the same bin centers shifted by the tone-to-go offset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is defined on the exact same 80 go-cue-centered bin centers as the neural firing rates, using the same kept trials and trial order.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
...
input_arr[0] = (BIN_CENTERS_REL - tone_on_rel).astype(np.float32)
```

```python
go = float(go_times[trial_idx])
edges_abs = go + BIN_EDGES_REL
...
neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

iii. The AI's notes describe this input as living on the same go-cue-centered decoder grid as the neural data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the trial-table columns `photostim_onset` and `photostim_duration`, together with `start_time` to place those relative offsets on the absolute session timeline.

ii.
```python
photo_onset_text = decode_vector(trials["photostim_onset"][:])
photo_dur_text = decode_vector(trials["photostim_duration"][:])
```

```python
if photo_onset_text[i] != "N/A" and photo_dur_text[i] != "N/A":
    stim_on = float(trial_start[i]) + float(photo_onset_text[i])
    stim_off = stim_on + float(photo_dur_text[i])
else:
    stim_on = math.nan
    stim_off = math.nan
```

iii. The notes say the NWB trial table already carries the stimulation timing needed for the decoder input, so the AI used those fields directly.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI turns photostimulation into a binary per-bin input: 1 when the absolute center of a bin lies between stimulation onset and offset, and 0 otherwise. Trials with `N/A` onset/duration become all zeros.

ii.
```python
if math.isnan(stim_on):
    input_arr[1] = 0.0
else:
    abs_centers = go + BIN_CENTERS_REL
    input_arr[1] = ((abs_centers >= stim_on) & (abs_centers < stim_off)).astype(np.float32)
```

iii. In the notes, the AI justifies this as matching the decoder specification, which asked for whether photostimulation is on at every time point.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI compares stimulation times and bin centers on the same absolute session clock. Neural bins are defined by `go + BIN_EDGES_REL`, while photostimulation uses `go + BIN_CENTERS_REL`.

ii.
```python
go = float(go_times[trial_idx])
edges_abs = go + BIN_EDGES_REL
...
abs_centers = go + BIN_CENTERS_REL
input_arr[1] = ((abs_centers >= stim_on) & (abs_centers < stim_off)).astype(np.float32)
```

iii. The notes say all time series are aligned to go cue on the same session clock, so photostimulation can be expressed directly on the neural bin grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the absolute left-lick and right-lick event streams plus the go-cue times. It does not use `trial_instruction` and `outcome` for the final choice value; instead it finds the first post-go lick within the 1.5 s response window and labels the trial as left, right, or no lick.

ii.
```python
def derive_choice_per_trial(
    go_times: np.ndarray,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> np.ndarray:
    choice = np.full(len(go_times), 2, dtype=np.int64)  # default no lick
    for i, go in enumerate(go_times):
        end = go + RESPONSE_WINDOW
        left = left_lick_times[(left_lick_times >= go) & (left_lick_times < end)]
        right = right_lick_times[(right_lick_times >= go) & (right_lick_times < end)]
        if len(left) == 0 and len(right) == 0:
            continue
        if len(left) > 0 and (len(right) == 0 or left[0] < right[0]):
            choice[i] = 0
        elif len(right) > 0 and (len(left) == 0 or right[0] < left[0]):
            choice[i] = 1
    return choice
```

```python
choice = derive_choice_per_trial(go_times, left_lick_times, right_lick_times)
```

iii. The trajectory says the AI checked this reconstruction and found that the first post-go lick reproduced the trial labels "almost perfectly." In the notes it explicitly preferred this because the NWB files do not contain an explicit per-trial choice column.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI codes choice as `0 = left`, `1 = right`, `2 = no lick`, then broadcasts that per-trial label across all 80 bins in output row 0.

ii.
```python
OUTPUT_VALUES = [
    ["left", "right", "no lick"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["<40th percentile", "40th-60th percentile", ">60th percentile", "not visible"],
]
```

```python
output_arr = np.empty((4, NBINS), dtype=np.int64)
output_arr[0] = choice[trial_idx]
```

iii. The notes say the decoder format should keep all outputs on a common `(n_output, n_timepoints)` shape, so trialwise labels are repeated across the 80 bins.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is read directly from the trial-table `outcome` column.

ii.
```python
outcome_text = decode_vector(trials["outcome"][:])
```

iii. The notes treat the NWB trial table as the native source for per-trial behavioral labels, and `outcome` already has the required categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps the strings `ignore`, `miss`, and `hit` to `0`, `1`, and `2`, then broadcasts that per-trial label across all bins in output row 1.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_text], dtype=np.int64)
...
output_arr[1] = outcome[trial_idx]
```

iii. The notes say the decoder needs categorical outputs, so the existing string labels were converted to fixed integer classes and stored on the shared output tensor.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is read directly from the trial-table `early_lick` column.

ii.
```python
early_text = decode_vector(trials["early_lick"][:])
```

iii. The notes identify `early_lick` as one of the trial-level behavior fields already present in the NWB trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early` to `0` and `early` to `1`, then broadcasts that per-trial label across all bins in output row 2.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_text], dtype=np.int64)
...
output_arr[2] = early[trial_idx]
```

iii. As with the other per-trial outputs, the notes say the label is repeated across bins so all outputs fit the same tensor shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using its `timestamps` and the three-column `data` array. The AI uses columns 0-1 as `x,y` and column 2 as tracking likelihood.

ii.
```python
tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
```

```python
xy = np.asarray(data[:, :2], dtype=np.float64)
likelihood = np.asarray(data[:, 2], dtype=np.float64)
```

iii. In the notes, the AI says this is the relevant side-camera tongue stream present across the NWB sessions and the required source for the requested tongue output.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first marks visible tongue samples using a conservative likelihood threshold (`>= 0.9`) and a 5-sigma velocity-outlier rejection rule. It computes session-level 40th and 60th percentile thresholds from the visible raw `y` samples that survive this filtering. For each trial, it then assigns each 50 ms bin the class implied by the last frame in that bin if that last frame is visible; otherwise the bin is `not visible`.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
TONGUE_VELOCITY_SIGMA = 5.0
```

```python
def build_tongue_session_stats(
    timestamps: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    xy = np.asarray(data[:, :2], dtype=np.float64)
    likelihood = np.asarray(data[:, 2], dtype=np.float64)
    dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.0034
    outliers = velocity_outlier_mask(xy, dt)
    visible = np.isfinite(xy[:, 1]) & np.isfinite(likelihood) & (~outliers) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    visible_y = xy[visible, 1]
    q40, q60 = np.quantile(visible_y, [0.4, 0.6])
```

```python
def discretize_tongue_bins(
    frame_timestamps: np.ndarray,
    tongue_xy: np.ndarray,
    tongue_likelihood: np.ndarray,
    tongue_visible_mask: np.ndarray,
    q40: float,
    q60: float,
    trial_edges_abs: np.ndarray,
) -> np.ndarray:
    out = np.full(NBINS, 3, dtype=np.int64)
    frame_idx = np.searchsorted(frame_timestamps, trial_edges_abs)
    for b in range(NBINS):
        start = int(frame_idx[b])
        end = int(frame_idx[b + 1])
        if end <= start:
            continue
        last = end - 1
        if not tongue_visible_mask[last]:
            continue
        y = float(tongue_xy[last, 1])
        if y < q40:
            out[b] = 0
        elif y <= q60:
            out[b] = 1
        else:
            out[b] = 2
```

iii. The notes and trajectory justify this by referring to the method paper's 5-sigma marker-cleaning rule, the explicit decoder requirement for a `not visible` class, a "conservative" visibility threshold because the likelihood distribution looked bimodal, and a "reference-style sample-and-hold" binning rule using the last frame in each bin.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI computes session-wide `q40` and `q60` from visible raw tongue-y samples. Each bin is then labeled `0` if `y < q40`, `1` if `q40 <= y <= q60`, `2` if `y > q60`, and `3` if the bin has no visible last frame.

ii.
```python
visible_y = xy[visible, 1]
q40, q60 = np.quantile(visible_y, [0.4, 0.6])
```

```python
if y < q40:
    out[b] = 0
elif y <= q60:
    out[b] = 1
else:
    out[b] = 2
```

iii. The notes say the 40th/60th percentile split follows the task definition, while the percentiles are restricted to visible samples because the output also reserves an explicit `not visible` category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue data to the same go-cue-centered 50 ms grid as the neural data. For each trial it computes absolute trial edges as `go + BIN_EDGES_REL`, finds the camera frames falling in each bin by `searchsorted`, and assigns one class per neural bin.

ii.
```python
go = float(go_times[trial_idx])
edges_abs = go + BIN_EDGES_REL
```

```python
frame_idx = np.searchsorted(frame_timestamps, trial_edges_abs)
for b in range(NBINS):
    start = int(frame_idx[b])
    end = int(frame_idx[b + 1])
```

iii. The notes describe this as using the same go-cue-centered grid as the neural data, with the additional trial filter that the whole decoder window must be covered by tongue timestamps.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mainly handles problematic data by exclusion rather than imputation: sessions with no good units are skipped, trials outside ephys coverage are removed via `obs_intervals`, `auto_water`/`free_water` trials are removed, trials with incomplete tongue-window coverage are removed, trials with no sample-start event before go are removed, and trials with all-zero neural activity are removed after binning. Within kept trials, tongue bins whose last frame is not visible remain as class `3` (`not visible`).

ii.
```python
good_units = [i for i, c in enumerate(classification) if c == "good"]
if not good_units:
    raise ValueError(f"{session_id}: no classifier-good units")
```

```python
if i not in recorded_trial_set:
    continue
if auto_water[i] == 1 or free_water[i] == 1:
    continue
if edges_abs[0] < tongue_timestamps[0] or edges_abs[-1] > tongue_timestamps[-1]:
    continue
sample_start = last_event_before(sample_start_times, float(trial_start[i]), go)
if sample_start is None:
    continue
```

```python
if not np.any(neural):
    dropped_all_zero_neural += 1
    continue
```

iii. The trajectory shows the AI discovered an all-zero-neural bug and justified the `obs_intervals` filter as the raw coverage signal. The notes justify the conservative tongue handling by the explicit `not visible` category and the desire to keep only trials with full decoder-window support.

## 10-a. What are the most time-consuming steps of the code?

i. The code's expensive steps are opening many NWB files twice, reading large unit-spike and tongue-tracking arrays from HDF5, and performing per-trial/per-unit spike binning. Optional plotting also adds overhead when enabled.

ii.
```python
def discover_curated_sessions() -> list[str]:
    files = list_session_files()
    curated = [path for path in files if session_has_good_units(path)]
    return curated
```

```python
with h5py.File(path, "r") as h5:
    ...
    spike_flat = np.asarray(units["spike_times"][:], dtype=np.float64)
    tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
```

```python
for kept_idx, trial_idx in enumerate(candidate_trial_idx):
    ...
    for unit_row, spikes_abs in enumerate(unit_spike_times):
        neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

iii. The notes mention that curated-session discovery reopens every NWB file once just to inspect `classification`, and the trajectory repeatedly calls out the coverage-check and spike-binning stages as the main runtime work.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain fully in Python: trialwise choice reconstruction, mapping region labels over units, the candidate-trial filtering loop, the nested per-trial/per-unit neural-binning loop, and the per-bin tongue discretization loop. The neural loop is the largest missed vectorization opportunity because the reference solution bins all trials for one unit in a single `searchsorted`.

ii.
```python
for i, go in enumerate(go_times):
    ...
```

```python
for i in range(len(trial_start)):
    ...
```

```python
for kept_idx, trial_idx in enumerate(candidate_trial_idx):
    ...
    for unit_row, spikes_abs in enumerate(unit_spike_times):
        neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

```python
for b in range(NBINS):
    ...
```

iii. The AI did not explicitly defend these loops in the notes; the implicit rationale is readability plus runtime being acceptable after filtering, but no strong vectorization argument is documented.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of work. It scans every NWB once in `discover_curated_sessions()` and then reopens curated files again in `process_session()`. It also constructs trial edges and reruns `searchsorted` separately for every `(trial, unit)` pair, rather than flattening all trial edges per unit as in the reference solution.

ii.
```python
curated_sessions = discover_curated_sessions()
...
for path in curated_sessions:
    session_data, session_summary = process_session(path, make_plot=session_id in plot_ids)
```

```python
for kept_idx, trial_idx in enumerate(candidate_trial_idx):
    go = float(go_times[trial_idx])
    edges_abs = go + BIN_EDGES_REL
    ...
    for unit_row, spikes_abs in enumerate(unit_spike_times):
        neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

iii. The notes acknowledge the extra startup overhead from opening every NWB during curated-session discovery, but otherwise do not present this repeated work as a deliberate choice.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some extra diagnostic work that is not needed by the final converted dataset: it builds detailed `session_summary` objects containing raw tongue arrays and count summaries, computes optional processing plots, tracks `region_counts`, passes an unused `tongue_likelihood` argument into `discretize_tongue_bins`, and fully computes neural arrays for some trials only to discard them as all-zero.

ii.
```python
session_summary = {
    ...
    "region_counts": dict(region_counts),
    "tongue_timestamps": tongue_timestamps,
    "tongue_xy": tongue_xy,
    "tongue_likelihood": tongue_likelihood,
    "tongue_visible": tongue_visible_mask,
}
```

```python
if make_plot:
    example = {
        "go_time": float(go_times[trial_keep[example_trial_idx]]),
        "neural": neural_trials[example_trial_idx],
        "input": input_trials[example_trial_idx],
        "output": output_trials[example_trial_idx],
    }
    outpath = f"/app/processing_{session_id}.png"
    plot_processing_figure(session_id, session_summary, example, outpath)
```

```python
def discretize_tongue_bins(
    frame_timestamps: np.ndarray,
    tongue_xy: np.ndarray,
    tongue_likelihood: np.ndarray,
    tongue_visible_mask: np.ndarray,
    ...
) -> np.ndarray:
```

iii. These pieces are justified only indirectly by the notes' emphasis on validation plots and sanity checks. They do not contribute to the final `converted_data.pkl` contents.
