# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively walks `/app/data`, sorts every `.nwb` path, pre-reads each file with `h5py` to retain sessions having a `classification == "good"` unit, then opens every retained NWB once more for conversion.

ii.
```python
for dirpath, _, filenames in os.walk(DATA_ROOT):
    for name in filenames:
        if name.endswith(".nwb"):
            files.append(os.path.join(dirpath, name))
return sorted(files)
...
curated = [path for path in files if session_has_good_units(path)]
```

iii. The notes identify the NWBs as the provided source and explain that excluding the one file without classifier-approved units recovers the paper's 173-session set.

## 1-b. How are the data split into subjects?

i. A subject is the parent directory name of each session file (for example, `sub-440956`). Unique values are sorted and each session is mapped to their index.

ii.
```python
def get_subject_id(path: str) -> str:
    return os.path.basename(os.path.dirname(path))
...
subjects = sorted({sess["subject"] for sess in session_dicts})
subject_idx = np.array([subject_to_idx[sess["subject"]] for sess in session_dicts])
```

iii. The agent says the folder is a stable NWB subject identifier. It preserves the `sub-` prefix rather than reading `nwb.subject.subject_id`.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; file order is sorted, and the filename supplies the session ID. Files with no good units are removed before conversion.

ii.
```python
def get_session_id(path: str) -> str:
    return os.path.basename(path).replace("_behavior+ecephys+ogen.nwb", "").replace("_behavior+ecephys.nwb", "")
...
for path in curated_sessions:
    session_data, session_summary = process_session(path, ...)
```

iii. The notes state that one NWB is one session and that the 173 classifier-curated files match the paper-wide ephys set.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials. Same-index go cues and trial fields are used, and retained trial indices are stored as separate matrices in session lists.

ii.
```python
trials = h5["intervals"]["trials"]
trial_start = np.asarray(trials["start_time"][:], dtype=np.float64)
...
for i in range(len(trial_start)):
    go = float(go_times[i])
```

iii. The agent regards the NWB trials table as authoritative and uses `obs_intervals` only to decide which table rows have ephys coverage.

## 1-e. How are trials filtered based on quality controls?

i. Trials must map to a good unit's `obs_intervals`; `auto_water` and `free_water` trials, trials whose full window exceeds tongue timestamps, trials without a preceding sample event, and residual all-zero-neural trials are removed. Early-lick, ignore, and photostim trials remain. Sessions must retain at least two trials.

ii.
```python
if i not in recorded_trial_set: continue
if auto_water[i] == 1 or free_water[i] == 1: continue
if edges_abs[0] < tongue_timestamps[0] or edges_abs[-1] > tongue_timestamps[-1]: continue
if sample_start is None: continue
...
if not np.any(neural): continue
```

iii. The notes justify coverage/all-zero filtering as preventing missing ephys from becoming 0 Hz, exclude water trials as atypical contingencies, and retain early/ignore/photostim because they are required labels or inputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from ragged `units/spike_times`, restricted by `units/classification`, with go-cue timestamps defining absolute bin edges.

ii.
```python
spike_flat = np.asarray(units["spike_times"][:], dtype=np.float64)
classification = decode_vector(units["classification"][:])
good_units = [i for i, c in enumerate(classification) if c == "good"]
```

iii. The classifier field is documented as the NWB analogue of the paper's QC-approved unit list.

## 2-b. How is the `neural` data processed?

i. For every retained trial and good unit, sorted spikes are counted between 81 edges with `searchsorted`, differenced into 80 counts, and divided by 0.05 s to yield Hz. There is no smoothing or normalization.

ii.
```python
idx = np.searchsorted(spike_times_abs, trial_edges_abs, side="left")
counts = np.diff(idx)
return counts.astype(np.float32) / BIN_SIZE
```

iii. The notes say this preserves reference-style firing-rate construction while adapting the bin width to the explicit decoder requirement.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose classifier label is exactly `good` are retained; a session with none is excluded. `unit_quality` and per-unit metric thresholds are not used.

ii.
```python
good_units = [i for i, c in enumerate(classification) if c == "good"]
if not good_units:
    raise ValueError(f"{session_id}: no classifier-good units")
```

iii. This produces 69,453 units over 173 sessions, close to the reported 69,943; the notes explain that `unit_quality` is much too permissive.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Relative edges from -2.5 to +1.5 s are added to each absolute go-cue time; spikes and events already share the NWB session clock.

ii.
```python
go = float(go_times[trial_idx])
edges_abs = go + BIN_EDGES_REL
neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

iii. The agent states that the common NWB clock makes offset correction or interpolation unnecessary.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The result has 80 non-overlapping 50-ms bins across the four-second window. Raw spike timestamps are newly binned; no further temporal rebinning is applied.

ii.
```python
BIN_SIZE = 0.05
NBINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES_REL = np.linspace(WINDOW_START, WINDOW_END, NBINS + 1)
```

iii. Fifty milliseconds and the window are explicit task requirements, superseding the paper's 40-ms analysis bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times`, trial start, and the go cue; the last sample start in the trial at or before go is selected.

ii.
```python
sample_start = last_event_before(sample_start_times, float(trial_start[i]), go)
candidate_sample_onsets_rel.append(sample_start - go)
```

iii. The last event handles sample replay after early licking.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The sample onset relative to go is subtracted from each go-relative bin center.

ii.
```python
tone_on_rel = float(candidate_sample_onsets_rel[kept_idx])
input_arr[0] = (BIN_CENTERS_REL - tone_on_rel).astype(np.float32)
```

iii. This directly expresses elapsed seconds from the final tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the exact same 80 go-relative bin centers as the neural intervals.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
input_arr[0] = BIN_CENTERS_REL - tone_on_rel
```

iii. The shared go cue and bin grid guarantee one input value per neural bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It comes from trial `photostim_onset`, `photostim_duration`, and `start_time`, with `N/A` representing no stimulation.

ii.
```python
stim_on = float(trial_start[i]) + float(photo_onset_text[i])
stim_off = stim_on + float(photo_dur_text[i])
```

iii. The agent notes that onset is trial-start-relative and must be converted to the absolute NWB clock.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Each bin center is coded 1 inside the half-open stimulation interval and 0 otherwise; nonstimulated trials are all zero.

ii.
```python
abs_centers = go + BIN_CENTERS_REL
input_arr[1] = ((abs_centers >= stim_on) & (abs_centers < stim_off)).astype(np.float32)
```

iii. The binary time series satisfies the requested per-time-point indicator.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation bounds are compared with absolute centers constructed from the same go cue and relative grid as neural bins.

ii.
```python
abs_centers = go + BIN_CENTERS_REL
input_arr[1] = ((abs_centers >= stim_on) & (abs_centers < stim_off))
```

iii. The common session clock and grid align corresponding columns.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is reconstructed from session-level left- and right-lick event timestamps and each trial's go time, rather than from trial instruction and outcome.

ii.
```python
choice = derive_choice_per_trial(go_times, left_lick_times, right_lick_times)
```

iii. The notes say the first post-go lick matched hit/miss/ignore semantics in spot checks and most directly represents actual choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The earliest lick in `[go, go+1.5)` is coded left=0 or right=1; absence of either is 2. The scalar is broadcast across 80 bins.

ii.
```python
choice = np.full(len(go_times), 2, dtype=np.int64)
if len(left) > 0 and (len(right) == 0 or left[0] < right[0]): choice[i] = 0
elif len(right) > 0 and (len(left) == 0 or right[0] < left[0]): choice[i] = 1
...
output_arr[0] = choice[trial_idx]
```

iii. The response window comes from the task, and broadcasting allows trial-level and time-varying outputs to share one tensor.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is read directly from the trials table's `outcome` field.

ii.
```python
outcome_text = decode_vector(trials["outcome"][:])
```

iii. The raw categories already match the requested output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map as ignore=0, miss=1, hit=2, then the value is broadcast across bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_text])
output_arr[1] = outcome[trial_idx]
```

iii. The mapping follows requested category order; broadcasting gives a consistent output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is read directly from the trial `early_lick` field.

ii.
```python
early_text = decode_vector(trials["early_lick"][:])
```

iii. The trial table already supplies the required flag.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` to 1; the value is broadcast across bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_text])
output_arr[2] = early[trial_idx]
```

iii. The mapping matches the requested no/yes order.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns y (index 1) and tracking likelihood (index 2) from `Camera0_side_TongueTracking`; x is also used to compute 2-D velocity outliers.

ii.
```python
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
xy = np.asarray(data[:, :2], dtype=np.float64)
likelihood = np.asarray(data[:, 2], dtype=np.float64)
```

iii. The notes identify this side-camera stream as the relevant tongue marker source used by the paper.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames are visible only if finite, likelihood is at least 0.9, and neither endpoint is flagged by a session-wide speed greater than mean plus 5 SD. For each 50-ms trial bin the last frame is used; an invisible or absent last frame yields class 3.

ii.
```python
visible = np.isfinite(xy[:, 1]) & np.isfinite(likelihood) & (~outliers) & (likelihood >= 0.9)
...
last = end - 1
if not tongue_visible_mask[last]: continue
y = float(tongue_xy[last, 1])
```

iii. The agent cites method-paper 5-sigma marker cleaning, bimodal likelihoods supporting a conservative 0.9 cutoff, and reference-style last-frame/sample-and-hold alignment.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are computed per session over all visible, cleaned raw frame y values. A retained frame is class 0 below q40, class 1 from q40 through q60, class 2 above q60; class 3 means not visible.

ii.
```python
q40, q60 = np.quantile(visible_y, [0.4, 0.6])
if y < q40: out[b] = 0
elif y <= q60: out[b] = 1
else: out[b] = 2
```

iii. The notes argue that the explicit not-visible class means percentile estimation should use only visible samples.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames are selected using each trial's absolute neural-bin edges. The last camera frame in each interval supplies the corresponding output bin.

ii.
```python
frame_idx = np.searchsorted(frame_timestamps, trial_edges_abs)
start = int(frame_idx[b]); end = int(frame_idx[b + 1])
last = end - 1
```

iii. Camera, go cues, and spikes share the NWB absolute clock; the agent describes last-frame selection as sample-and-hold consistent with its reading of reference marker alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions without good units error out/pre-filter away; unmatched observation intervals raise; trials outside ephys or tongue coverage, without sample onset, or with all-zero neural windows are dropped; bins without a visible tongue frame become explicit class 3; missing anatomy falls back to electrode target or `Unknown`.

ii.
```python
if key not in trial_lookup: raise ValueError(...)
if not np.any(neural): continue
out = np.full(NBINS, 3, dtype=np.int64)
unit_region = target if target else "Unknown"
```

iii. The rationale distinguishes absent recordings (exclude rather than fabricate) from meaningful tongue invisibility (retain as a requested category), while preserving anatomical rows with fallback labels.

## 10-a. What are the most time-consuming steps of the code?

i. The nested trial-by-unit spike binning is the principal conversion cost, along with reading large spike/tracking arrays and serializing the full dataset.

ii.
```python
for kept_idx, trial_idx in enumerate(candidate_trial_idx):
    for unit_row, spikes_abs in enumerate(unit_spike_times):
        neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

iii. The notes explicitly identify repeated per-trial spike binning as the main scaling cost (`n_trials * n_units`) and report pre-slicing spike trains to reduce overhead.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial/unit neural loop could search all trial edges at once per unit, as the human implementation does. The per-bin tongue loop could also be replaced by indexed aggregation; trial filtering and region-label loops are smaller candidates.

ii.
```python
for kept_idx, trial_idx in enumerate(candidate_trial_idx):
    for unit_row, spikes_abs in enumerate(unit_spike_times): ...
...
for b in range(NBINS): ...
```

iii. The agent claimed `searchsorted` kept counting vectorized and emphasized pre-slicing, but it did not vectorize over trials; its own timing note recognizes the nested loop as the dominant cost.

## 10-c. What processing does the code repeat multiple times?

i. Every NWB is opened once during session discovery and again during conversion. Spike-edge searches repeat for each trial and unit; tongue bin lookup repeats for every retained trial.

ii.
```python
curated = [path for path in files if session_has_good_units(path)]
...
with h5py.File(path, "r") as h5:  # process_session opens it again
```

iii. Pre-discovery was chosen to establish the curated 173 sessions, while unit spike arrays are pre-sliced once; nevertheless, the second file open and per-trial computations are real repeated work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads `trial_instruction` but never uses it, computes detailed `session_summary` fields (large tongue arrays, counts, percentiles, region counts) mostly discarded after optional plotting, and reads/uses x solely for velocity rejection. Pre-scan classification data are also discarded and reread.

ii.
```python
trial_instruction = decode_vector(trials["trial_instruction"][:])
...
session_summary = {"tongue_timestamps": tongue_timestamps, "tongue_xy": tongue_xy, ...}
```

iii. These values support diagnostics, plots, and validation described in the notes, but most do not enter the saved decoder dataset.
