# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files with a glob over `/app/data/sub-*/*.nwb`, opens each file directly with `h5py`, and reads groups/datasets by HDF5 path rather than using `pynwb`. Within each file it reads subject metadata, unit tables, trial tables, behavioral event timestamps, and tongue-tracking arrays.

ii.
```python
DATA_GLOB = "/app/data/sub-*/*.nwb"

for path in sorted(glob.glob(DATA_GLOB)):
    session = process_session(path, region_to_index)
```

```python
with h5py.File(path, "r") as f:
    subject = decode_scalar(f["general/subject/subject_id"])
    classification = decode_array(f["units/classification"])
    spike_times_flat = f["units/spike_times"][()]
    trials = f["intervals/trials"]
    go_times = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
```

iii. In the trajectory, the agent repeatedly described the NWB files as the complete session set and chose to read them directly from HDF5 because the needed trial, spike, lick, and tongue fields were already exposed there.

## 1-b. How are the data split into subjects (mice)?

i. The AI uses `general/subject/subject_id` from each NWB file as the mouse identifier. Subjects are accumulated in first-seen order during the session loop, and `subject_idx` records that index per retained session.

ii.
```python
subject = decode_scalar(f["general/subject/subject_id"])
```

```python
if subject not in subject_to_index:
    subject_to_index[subject] = len(data["subjects"])
    data["subjects"].append(subject)

subject_idx.append(subject_to_index[subject])
```

iii. The trajectory shows the agent treating the NWB subject field as canonical and carrying it through directly rather than inferring mouse identity from filenames or paper names.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order is the sorted file order returned by the glob.

ii.
```python
for path in sorted(glob.glob(DATA_GLOB)):
    session = process_session(path, region_to_index)
```

iii. In the trajectory, the agent explicitly relied on the one-file-per-session NWB layout and did not introduce any extra regrouping step.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table, and trial-aligned arrays are indexed by the same trial index into `go_start_times`, `photostim` fields, and later the neural and output arrays. After neural binning, only trials with any nonzero activity in the kept units are retained.

ii.
```python
trials = f["intervals/trials"]
n_trials = len(trials["id"])
trial_start = trials["start_time"][()]
go_times = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
```

```python
keep_trials = np.any(rates != 0, axis=(0, 2))
kept_trial_indices = np.flatnonzero(keep_trials)
neural_trials = [rates[:, trial_idx, :].copy() for trial_idx in kept_trial_indices]
```

iii. The trajectory shows the agent checking the NWB trial/event structure directly and then using the trials table as the session’s base indexing structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not use `obs_intervals` or `free_water` directly. Instead it first drops entire sessions that fail a behavioral criterion: control trials only, excluding early licks, with `performance > 0.65` and at least `50` correct left and `50` correct right trials. Within retained sessions, it drops any trial whose binned activity is all zero across the retained units and bins.

ii.
```python
MIN_CONTROL_PERFORMANCE = 0.65
MIN_CORRECT_PER_SIDE = 50

behavior_metrics = session_behavior_metrics(
    outcome, trial_instruction, early, photostim_onset
)
if not behavior_metrics["passes_filter"]:
    return None
```

```python
keep_trials = np.any(rates != 0, axis=(0, 2))
if not np.any(keep_trials):
    return None
```

iii. The trajectory makes this justification explicit. The agent said the paper-style session filter was “more defensible” and would “materially reduce training cost,” and later described keeping all trial types within those sessions because `outcome` and `early_lick` were decoder targets.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index`, filtered by `units/classification` and by brain region recovered from electrode locations. Trial alignment uses `BehavioralEvents/go_start_times`.

ii.
```python
classification = decode_array(f["units/classification"])
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
spike_starts = np.r_[0, spike_times_index[:-1]]
```

```python
unit_regions = session_region_labels(f, np.arange(len(classification)))
keep_mask = (classification == "good") & np.isin(unit_regions, list(REGIONS_TO_KEEP))
go_times = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
```

iii. In the trajectory, the agent explicitly checked that the NWB `units` table exposed a `classification` label and used go-cue timestamps as the alignment anchor.

## 2-b. How is the `neural` data processed?

i. For each kept unit, the AI bins absolute spike times into trial windows around each go cue using `assign_events_to_windows`, then divides counts by bin width to get firing rates. Rates are stored as `float16`.

ii.
```python
window_starts = go_times + BIN_EDGES[0]
window_ends = go_times + BIN_EDGES[-1]

rates = np.empty((keep_units.size, n_trials, N_BINS), dtype=np.float16)
for out_idx, unit_idx in enumerate(keep_units):
    spikes = spike_times_flat[spike_starts[unit_idx] : spike_times_index[unit_idx]]
    counts = assign_events_to_windows(spikes, window_starts, window_ends, BIN_WIDTH, N_BINS)
    rates[out_idx] = (counts / BIN_WIDTH).astype(np.float16)
```

iii. The trajectory shows the agent focusing on direct binning from NWB spike times and on keeping the export small enough to train, which is consistent with the `float16` storage and area restriction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == "good"` and further restricts them to `left ALM` and `right ALM`. Sessions failing the behavioral session filter are dropped, and trials with all-zero activity in the kept units are dropped.

ii.
```python
REGIONS_TO_KEEP = {"left ALM", "right ALM"}
keep_mask = (classification == "good") & np.isin(unit_regions, list(REGIONS_TO_KEEP))
keep_units = np.flatnonzero(keep_mask)
if keep_units.size == 0:
    return None
```

```python
if not behavior_metrics["passes_filter"]:
    return None
...
keep_trials = np.any(rates != 0, axis=(0, 2))
```

iii. The trajectory justifies this choice mainly on scale and compatibility grounds. The agent said it was constraining the export to ALM so the dataset stayed “region-homogeneous and small enough for the provided decoder to train on,” then later tightened session selection to the paper-style behavioral filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural windows are aligned to go-cue onset by adding fixed relative bin edges to each trial’s `go_times` and binning spikes inside those absolute windows.

ii.
```python
ALIGN_EVENT = "Go cue onset"
BEGIN_TIME = -2.5
END_TIME = 1.5
```

```python
window_starts = go_times + BIN_EDGES[0]
window_ends = go_times + BIN_EDGES[-1]
counts = assign_events_to_windows(spikes, window_starts, window_ends, BIN_WIDTH, N_BINS)
```

iii. The trajectory repeatedly states that the task is aligned to go-cue onset and that the converter should follow NWB trial/go-cue timing directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses `BIN_WIDTH = 0.05` and `BIN_STRIDE = 0.05`, so it intends non-overlapping 50 ms bins. However, its `compute_bin_centers` function produces `81` bins from `-2.5` to `+1.5` inclusive, not the expected `80` bins.

ii.
```python
BIN_WIDTH = 0.05
BIN_STRIDE = 0.05

def compute_bin_centers(begin_time, end_time, stride):
    span = (end_time - begin_time) / stride
    if np.allclose(span, math.floor(span) + 1):
        n_bins = math.floor(span) + 2
    else:
        n_bins = math.floor(span) + 1
    return begin_time + np.arange(n_bins) * stride
```

```python
BIN_CENTERS = compute_bin_centers(BEGIN_TIME, END_TIME, BIN_STRIDE).astype(np.float32)
N_BINS = len(BIN_CENTERS)
```

iii. The trajectory states the intended choice clearly: 50 ms bins from `-2.5 s` to `+1.5 s` around the go cue. The extra bin appears to be an implementation mistake rather than a separately justified decision.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. In the final AI code, this input is not derived from any tone-onset variable. It is derived only from the fixed bin centers relative to the go cue. The trajectory says the AI deliberately interpreted this as time from the go-cue tone, not from the sample tone.

ii.
```python
BIN_CENTERS = compute_bin_centers(BEGIN_TIME, END_TIME, BIN_STRIDE).astype(np.float32)
```

```python
input_trials.append(
    np.vstack([BIN_CENTERS, stim_series]).astype(np.float32, copy=False)
)
```

iii. The trajectory justification is explicit in the final summary: the agent said “The ‘time from tone onset’ input is encoded as time from the Go-cue tone, which is the unambiguous tone aligned by this task.”

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. No tone-onset computation is performed. The AI simply uses the go-cue-relative bin centers as the first input channel for every trial.

ii.
```python
input_trials.append(
    np.vstack([BIN_CENTERS, stim_series]).astype(np.float32, copy=False)
)
```

iii. The trajectory indicates the agent chose the go-cue tone interpretation to avoid ambiguity about repeated sample tones.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time channel is perfectly co-indexed with the neural data because it is the same shared vector of bin centers used for every trial’s neural window.

ii.
```python
window_starts = go_times + BIN_EDGES[0]
window_ends = go_times + BIN_EDGES[-1]
...
input_trials.append(
    np.vstack([BIN_CENTERS, stim_series]).astype(np.float32, copy=False)
)
```

iii. The trajectory shows the agent cared about keeping the exported tensors directly compatible with the decoder’s time axis, so it used one shared per-bin time vector.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `trials/start_time`, `trials/photostim_onset`, `trials/photostim_duration`, and the per-trial `go_times`.

ii.
```python
trial_start = trials["start_time"][()]
photostim_onset = decode_array(trials["photostim_onset"])
photostim_duration = decode_array(trials["photostim_duration"])
go_times = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
```

```python
def parse_photostim_series(trial_start, photostim_onset, photostim_duration, go_time):
    onset_abs = trial_start + float(photostim_onset)
    offset_abs = onset_abs + float(photostim_duration)
    rel_on = onset_abs - go_time
    rel_off = offset_abs - go_time
```

iii. In the trajectory, the agent inspected both the trial-table photostim fields and the photostim event timestamps and concluded that the trial-table onset/duration relative to trial start were sufficient.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each kept trial, the AI converts the trial-relative stimulation onset/duration to a go-cue-relative interval and marks bins whose centers fall inside that interval with `1.0`; all other bins are `0.0`.

ii.
```python
stim = np.zeros(N_BINS, dtype=np.float32)
if photostim_onset == "N/A":
    return stim
...
stim[(BIN_CENTERS >= rel_on) & (BIN_CENTERS < rel_off)] = 1.0
```

iii. The trajectory shows the agent verifying that the table onset and duration matched the explicit photostim event stream before keeping this simpler implementation.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned by converting stimulation onset and offset into the same go-cue-relative coordinates as the neural bins and then evaluating those intervals against `BIN_CENTERS`.

ii.
```python
rel_on = onset_abs - go_time
rel_off = offset_abs - go_time
stim[(BIN_CENTERS >= rel_on) & (BIN_CENTERS < rel_off)] = 1.0
```

iii. The trajectory reflects the same logic: the agent aligned all trial-wise inputs to the go cue because the neural windows were defined on that axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the behavioral event streams `left_lick_times` and `right_lick_times`, not from `trial_instruction` plus `outcome`. It searches the response window after each go cue and picks the earliest lick side, or “no lick” if neither side occurs.

ii.
```python
left_licks = f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()]
right_licks = f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()]
```

```python
def first_choice_after_go(left_licks, right_licks, go_time, response_window=1.5):
    ...
    if np.isfinite(first_left) and first_left < first_right:
        return 0
    if np.isfinite(first_right):
        return 1
    return 2
```

iii. The trajectory explicitly says the “exact lick-choice definition” was an open point. The agent checked miss and ignore trials against lick timestamps and then chose “first choice after go” as the behavioral definition.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The first lick after go within a 1.5 s response window is coded as `0` for left and `1` for right; if neither side occurs, the code is `2` (“no lick”). That per-trial value is then repeated across all bins.

ii.
```python
choice = first_choice_after_go(left_licks, right_licks, go_times[trial_idx])
...
np.full(N_BINS, choice, dtype=np.int64)
```

iii. The trajectory shows the agent validating this with direct spot-checks of miss and ignore trials before implementing it.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is read directly from the trial-table `outcome` field.

ii.
```python
outcome = decode_array(trials["outcome"])
```

iii. The trajectory repeatedly identified the trial-table outcome field as one of the straightforward labels already present in NWB.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore`, `miss`, and `hit` to `0`, `1`, and `2`, then repeats the per-trial category across all bins.

ii.
```python
outcome_label = {"ignore": 0, "miss": 1, "hit": 2}[outcome[trial_idx]]
...
np.full(N_BINS, outcome_label, dtype=np.int64)
```

iii. The trajectory did not present this as a disputed point; the agent treated it as a direct categorical export from the trials table.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is read directly from the trial-table `early_lick` field.

ii.
```python
early = decode_array(trials["early_lick"])
```

iii. The trajectory identifies `early_lick` as one of the explicit trial fields available in NWB and preserves it because it is a decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early` to `0` and `early` to `1`, then repeats the per-trial value across all bins.

ii.
```python
early_label = {"no early": 0, "early": 1}[early[trial_idx]]
...
np.full(N_BINS, early_label, dtype=np.int64)
```

iii. The trajectory justification was that early-lick labels needed to stay in the export even though many paper analyses excluded such trials.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `Camera0_side_TongueTracking/data` and `.../timestamps`, specifically column 1 (`track[:, 1]`) for y-position and column 2 (`track[:, 2]`) for visibility/confidence.

ii.
```python
track = np.asarray(f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"])
track_times = np.asarray(
    f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"]
)
tongue_bins, p40, p60 = build_tongue_bins(track_times, track[:, 1], track[:, 2], go_times)
```

iii. The trajectory shows the agent inspecting the tracking array layout and the likelihood distribution before deciding how to threshold visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI keeps only frames with `track_prob > 0.9`, computes the 40th and 60th percentiles over all visible y-values in the session, assigns each visible frame to class 0/1/2 from those thresholds, and writes those classes directly into trial bins. Empty bins remain class `3` (“not visible”). If multiple visible frames land in the same bin, later frames overwrite earlier ones; there is no per-bin averaging.

ii.
```python
VISIBILITY_THRESHOLD = 0.9
...
visible = track_prob > VISIBILITY_THRESHOLD
y_visible = track_y[visible]
p40, p60 = np.percentile(y_visible, [40, 60])
```

```python
valid = probs > VISIBILITY_THRESHOLD
...
classes = np.full(len(ys), 1, dtype=np.int64)
classes[ys < p40] = 0
classes[ys > p60] = 2

flat = tongue_bins.reshape(-1)
flat[trial_idx * N_BINS + bin_idx] = classes
```

iii. The trajectory justification was mainly empirical: the agent measured that likelihood values were near-binary and concluded that `>0.9` was close to `>0.5`, then kept a simpler frame-level discretization instead of reproducing the reference’s bin-mean approach.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses three visible categories plus a hidden category: `0` for y-values below the session 40th percentile of visible frames, `1` for between the 40th and 60th percentiles, `2` for above the 60th percentile, and `3` for bins with no visible frame.

ii.
```python
classes = np.full(len(ys), 1, dtype=np.int64)
classes[ys < p40] = 0
classes[ys > p60] = 2
...
tongue_bins = np.full((len(go_times), N_BINS), 3, dtype=np.int64)
```

iii. The trajectory indicates the agent was trying to honor the requested 40/60 percentile discretization while simplifying visibility handling and using a stricter visibility threshold.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are assigned into the same go-cue-centered trial windows used for neural binning. Each visible frame is mapped to a bin index from its offset relative to `go_time + BIN_EDGES[0]`.

ii.
```python
window_starts = go_times + BIN_EDGES[0]
window_ends = go_times + BIN_EDGES[-1]
trial_idx = np.searchsorted(window_starts, track_times, side="right") - 1
...
bin_idx = np.floor((times - window_starts[trial_idx]) / BIN_WIDTH).astype(np.int64)
```

iii. The trajectory consistently framed all time-varying signals as being brought onto the same go-cue-relative axis as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several issues implicitly:
- Non-string HDF5 entries are converted by `decode_array`/`decode_scalar`.
- Sessions with no retained ALM `classification == good` units are dropped.
- Sessions failing the behavioral session filter are dropped.
- Trials with all-zero activity in retained units are dropped.
- Tongue bins default to class `3` when no visible frame passes the confidence threshold.

ii.
```python
def decode_array(dataset):
    arr = np.asarray(dataset)
    if arr.dtype.kind == "S":
        return arr.astype(str)
    if arr.dtype == object:
        ...
```

```python
if keep_units.size == 0:
    return None
...
if not behavior_metrics["passes_filter"]:
    return None
...
keep_trials = np.any(rates != 0, axis=(0, 2))
```

iii. In the trajectory, the agent’s stated emphasis was on defensible curation and keeping the dataset trainable, so its missing-data handling is largely implemented as dropping sessions/trials that do not satisfy those constraints.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are opening every NWB file, reading full spike and tracking arrays, per-unit spike binning, and constructing full trial-by-time neural matrices before later filtering trials. The trajectory also shows the agent thinking about total export size and decoder runtime as major constraints.

ii.
```python
with h5py.File(path, "r") as f:
    ...
    spike_times_flat = f["units/spike_times"][()]
    ...
    track = np.asarray(f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"])
```

```python
rates = np.empty((keep_units.size, n_trials, N_BINS), dtype=np.float16)
for out_idx, unit_idx in enumerate(keep_units):
    ...
    counts = assign_events_to_windows(spikes, window_starts, window_ends, BIN_WIDTH, N_BINS)
```

iii. The trajectory explicitly quantified expected dataset size and used that as a reason to restrict regions/sessions before running conversion and training.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit neural loop, the per-trial assembly loop, and the frame-to-bin tongue assignment all remain explicit Python-level loops and could be reduced or partially vectorized.

ii.
```python
for out_idx, unit_idx in enumerate(keep_units):
    ...
```

```python
for trial_idx in kept_trial_indices:
    stim_series = parse_photostim_series(...)
    ...
```

iii. The trajectory does not contain an explicit optimization discussion beyond size/runtime concerns, but these are the obvious remaining loops in the implemented code.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly rescans whole-trial arrays to compute session behavior metrics, then later rescans per-trial lick arrays again to derive choice, and it computes full neural/tongue arrays for all trials before trimming some trials afterward. It also re-evaluates photostim intervals trial by trial.

ii.
```python
behavior_metrics = session_behavior_metrics(
    outcome, trial_instruction, early, photostim_onset
)
```

```python
for trial_idx in kept_trial_indices:
    stim_series = parse_photostim_series(...)
    choice = first_choice_after_go(left_licks, right_licks, go_times[trial_idx])
```

iii. The trajectory emphasis on getting a smaller, trainable export led the agent to prefer direct, readable per-trial computations over minimizing repeated passes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes full neural arrays for all session trials and only afterward drops zero-activity trials. It also computes tongue classes for all trials before applying that trial filter, and it computes session behavior metrics mainly to decide whether to keep a session and to store metadata rather than to populate decoder inputs/outputs.

ii.
```python
rates = np.empty((keep_units.size, n_trials, N_BINS), dtype=np.float16)
...
keep_trials = np.any(rates != 0, axis=(0, 2))
kept_trial_indices = np.flatnonzero(keep_trials)
```

```python
tongue_bins, p40, p60 = build_tongue_bins(track_times, track[:, 1], track[:, 2], go_times)
...
session_metadata = {
    ...
    **behavior_metrics,
}
```

iii. The trajectory shows the agent optimizing for overall trainability and not for eliminating all intermediate work, so some computations are performed before later exclusion.
