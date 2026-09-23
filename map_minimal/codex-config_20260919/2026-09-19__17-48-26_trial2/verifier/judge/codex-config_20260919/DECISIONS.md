# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent finds every `data/sub-*/*.nwb` file with a sorted glob, opens each file once with `h5py`, converts it as one session, and accumulates surviving sessions.

ii.
```python
paths = sorted(glob.glob(os.path.join(data_dir, "sub-*", "*.nwb")))
for path in paths:
    session = convert_session(path)
```
```python
with h5py.File(path, "r") as nwb:
    units = nwb["units"]
    trials = nwb["intervals/trials"]
```

iii. The trajectory says it inspected the NWB release and established that 174 files exist, one per session; sorting provides deterministic processing. It ultimately retained 173 curated sessions after dropping the session without classifier QC.

## 1-b. How are the data split into subjects?

i. The subject ID is read from each NWB file. Unique IDs are sorted, and each session gets an integer index into that list.

ii.
```python
subject = _scalar_string(nwb["general/subject/subject_id"])
subjects = sorted({session["subject"] for session in converted_sessions})
subject_to_index = {subject: i for i, subject in enumerate(subjects)}
```

iii. The agent relied on the canonical NWB subject field and verified the final artifact contained 28 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; the output session order is the sorted path order. The NWB identifier and source path are retained as metadata.

ii.
```python
identifier = _scalar_string(nwb["identifier"])
return {"neural": session_neural, ..., "session_info": {"identifier": identifier, ...}}
```

iii. The trajectory identified 174 files and 173 curated behavioral sessions; the QC-less file was intentionally skipped.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials. The agent reads one go-cue timestamp per row and raises an error if the counts differ. Retained rows become separate list entries.

ii.
```python
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
if len(go_times) != len(trials["id"]):
    raise ValueError(f"{path}: go cue and trial counts differ")
```

iii. It explicitly checked the one-to-one trial/event relationship rather than inferring trials from event gaps.

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept only if every retained unit's `is_good_trials` entry is true, it is neither auto-water nor free-water, and the entire population window is not all zero. Sessions with fewer than two remaining trials are dropped. Early-lick, ignore, and photostimulation trials are retained.

ii.
```python
unit_trial_validity = units["is_good_trials"][:][good_indices]
neural_valid[:n_ephys_trials] = np.all(unit_trial_validity, axis=0)
assistance = (trials["auto_water"][:] != 0) | (trials["free_water"][:] != 0)
keep = neural_valid & ~assistance
...
keep &= np.any(firing_rates_all != 0, axis=(1, 2))
```

iii. The agent found behavioral coverage beyond ephys in eight files and unit-specific invalid blocks in four more. It said simultaneous validity avoids representing missing recording as silence, and excluded assistance trials while preserving trial types required as decoder variables. This stricter policy yielded 88,943 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` for classifier-good, annotated units, plus go-cue timestamps used to define trial windows.

ii.
```python
all_spikes = units["spike_times"][:]
spike_ends = units["spike_times_index"][:].astype(np.int64)
window_starts = go_times + OFF_START
```

iii. The agent selected raw spike times as the source needed to reproduce firing-rate binning.

## 2-b. How is the `neural` data processed?

i. For each good unit, spikes are assigned to a trial window and 50-ms bin. `bincount` produces counts, which are divided by 0.05 s to obtain Hz. There is no smoothing or normalization.

ii.
```python
time_index = np.floor((spikes - window_starts[trial_index]) / BIN_S + 1e-12).astype(np.int64)
counts = np.bincount(flat_index, minlength=n_trials * N_TIME)
rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_S
```

iii. The stated goal was 80 bins of 50-ms firing rates, consistent with the instructions and reference processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == "good"` and a nonblank `anno_name`. A session with no such units is skipped.

ii.
```python
annotation_present = np.array([bool(x.strip()) for x in annotations])
good = (class_values == "good") & annotation_present
good_indices = np.flatnonzero(good)
```

iii. The trajectory ties `classification` to the paper's classifier QC and says the annotation condition restricts units to histologically localized neurons. It retained 69,453 units and skipped the one unclassified session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial windows start 2.5 s before and end 1.5 s after each absolute go-cue timestamp. Spike offsets from the window start determine their bins.

ii.
```python
window_starts = go_times + OFF_START
window_ends = go_times + OFF_END
time_index = np.floor((spikes - window_starts[trial_index]) / BIN_S + 1e-12)
```

iii. The agent treated spikes and behavioral events as sharing an absolute NWB clock and verified the resulting 80-bin aligned arrays.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms: 80 nonoverlapping bins over the four-second window. Raw spike times are binned directly; no later rebinning is applied.

ii.
```python
BIN_S = 0.050
N_TIME = int(round((OFF_END - OFF_START) / BIN_S))
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. This directly follows the decoder specification; the trajectory reports validation of exactly 80 × 50-ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/presample_stop_times` as one initial tone/sample onset per trial, together with go-cue-aligned bin centers.

ii.
```python
tone_onsets = nwb["acquisition/BehavioralEvents/presample_stop_times/timestamps"][:][keep]
time_from_tone = absolute_bin_times - tone_onsets[:, None]
```

iii. The agent reasoned that `presample_stop` is one-to-one with trials and represents the initial sample onset even when early licking replays task epochs.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial and bin, the absolute tone onset is subtracted from the absolute neural-bin center. The result is cast to `float32` when stacked into inputs.

ii.
```python
absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]
time_from_tone = absolute_bin_times - tone_onsets[:, None]
```

iii. No further transformation was considered necessary; the intended value is elapsed seconds.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same absolute 50-ms bin centers used for each go-cue-aligned neural trial.

ii.
```python
absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]
session_inputs.append(np.stack([time_from_tone[trial_index], photostim[trial_index]], axis=0))
```

iii. The shared bin-center array guarantees equal length and index alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, and `start_time`, plus the absolute bin centers.

ii.
```python
onset = _numeric_or_nan(trials["photostim_onset"].asstr()[:])[keep]
duration = _numeric_or_nan(trials["photostim_duration"].asstr()[:])[keep]
trial_starts = trials["start_time"][:][keep]
```

iii. The agent recognized that onset is relative to trial start and that `N/A` must represent no stimulation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. `N/A` is converted to NaN; otherwise absolute start and end times are calculated. A bin is true when its center is at or after onset and before offset.

ii.
```python
stimulation_start = trial_starts + onset
stimulation_end = stimulation_start + duration
photostim = ((absolute_bin_times >= stimulation_start[:, None]) &
             (absolute_bin_times < stimulation_end[:, None]))
```

iii. NaN comparisons naturally leave all bins false on non-stimulation trials.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation bounds and neural bin centers are both placed on the absolute session clock and compared at the same centers.

ii.
```python
absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]
photostim = (absolute_bin_times >= stimulation_start[:, None]) & ...
```

iii. This avoids a separate resampling operation and produces one stimulation value per neural bin.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trial-table `trial_instruction` and `outcome`: ignore means no lick; hit means the instructed side; miss means the opposite side.

ii.
```python
instruction = trials["trial_instruction"].asstr()[:][keep]
outcome_text = trials["outcome"].asstr()[:][keep]
reported_left = ((outcome_text == "hit") & (instruction == "left")) | ((outcome_text == "miss") & (instruction == "right"))
```

iii. The agent said these fields encode the report more robustly than raw lick events, which may contain brief licks at the other port.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice codes are 0 left, 1 right, and 2 no lick, repeated across all 80 output timepoints.

ii.
```python
choice = np.full(len(kept_indices), 2, dtype=np.int8)
choice[responded & reported_left] = 0
choice[responded & ~reported_left] = 1
np.full(N_TIME, choice[trial_index], dtype=np.int8)
```

iii. Repetition allows all categorical outputs, including the time-varying tongue output, to share one rectangular array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trial-table `outcome` strings.

ii.
```python
outcome_text = trials["outcome"].asstr()[:][keep]
```

iii. The raw column already contains the requested three categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to 0 ignore, 1 miss, and 2 hit, then repeated across 80 bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_text], dtype=np.int8)
```

iii. The mapping follows the requested category order and makes the per-trial label compatible with the common time axis.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trial-table `early_lick` string.

ii.
```python
early_text = trials["early_lick"].asstr()[:][keep]
```

iii. The NWB table already provides the requested per-trial flag.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The value is 1 exactly when the string is `early`, otherwise 0, and it is repeated over all bins.

ii.
```python
early = (early_text == "early").astype(np.int8)
np.full(N_TIME, early[trial_index], dtype=np.int8)
```

iii. This implements the requested no/yes coding while preserving early-lick trials for decoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses columns 1 and 2 of `Camera0_side_TongueTracking/data` as y-position and likelihood, with the series timestamps.

ii.
```python
tracking = behavior["Camera0_side_TongueTracking"]
tracking_data = tracking["data"][:, 1:3]
tracking_times = tracking["timestamps"][:]
```

iii. The agent identified side-camera tracking as the session's tongue signal and used confidence to distinguish visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames are visible if y and likelihood are finite and likelihood is at least 0.90. Session percentiles are computed from all visible raw-frame y values. Each output bin samples the most recent preceding video frame; an unavailable or low-confidence sample becomes not visible.

ii.
```python
visible_all = np.isfinite(y_all) & np.isfinite(likelihood_all) & (likelihood_all >= 0.90)
percentile_40, percentile_60 = np.percentile(y_all[visible_all], [40, 60])
frame_index = np.searchsorted(tracking_times, absolute_bin_times.ravel(), side="right") - 1
```

iii. The code's docstring describes this as sampling preceding frames and discretizing visible y per session. The trajectory focused on matching tongue visibility and alignment but gives no further justification for choosing point samples or 0.90.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Visible samples below the session's raw-frame 40th percentile are 0; values from the 40th through 60th percentiles are 1; values above the 60th are 2; invalid/low-confidence samples are 3.

ii.
```python
result = np.full(len(frame_index), 3, dtype=np.int8)
result[visible & (y < percentile_40)] = 0
result[visible & (y >= percentile_40) & (y <= percentile_60)] = 1
result[visible & (y > percentile_60)] = 2
```

iii. The 40/60 split and per-session scope follow the task; the fourth category represents no visible tongue.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For every absolute neural-bin center, `searchsorted(..., side="right") - 1` selects the preceding camera frame. No averaging or interpolation is performed.

ii.
```python
frame_index = np.searchsorted(tracking_times, absolute_bin_times.ravel(), side="right") - 1
sampled = tracking_data[safe_index]
return result.reshape(absolute_bin_times.shape), ...
```

iii. The agent used the shared absolute timestamps and exact neural-bin-center array to force one tongue class per neural timepoint.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The non-string classifier session and sessions with no good units are dropped; invalid/unrecorded or all-zero neural trials are dropped; `N/A` photostimulation values become NaN/all false; missing or low-confidence tongue samples become category 3. Count inconsistencies raise errors.

ii.
```python
if classification.dtype.kind not in "OSU":
    return None
...
keep &= neural_nonzero
...
result = np.full(len(frame_index), 3, dtype=np.int8)
```

iii. After the verifier exposed all-zero trials, the agent traced validity fields and rebuilt the artifact so missing recording was not treated as neural silence. It then reported no verifier warnings.

## 10-a. What are the most time-consuming steps of the code?

i. The code does not instrument substeps. Likely dominant conversion work is loading large HDF5 spike/tracking arrays and binning every retained unit across every trial. The trajectory says full decoder validation was even more expensive, especially 173 independent SVD initializations, but that is outside `convert_data.py`.

ii.
```python
all_spikes = units["spike_times"][:]
for out_unit, unit_index in enumerate(good_indices):
    ...
    counts = np.bincount(...)
```

iii. The trajectory described conversion as progressing session by session and decoder SVD/training as the expensive final check; it did not provide conversion profiling.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike loop and final per-trial packaging loop remain. Spike assignment is vectorized over all spikes of one unit; packaging could be replaced by array construction followed by list conversion, though it is unlikely to dominate.

ii.
```python
for out_unit, unit_index in enumerate(good_indices):
    ...
for trial_index in range(len(kept_indices)):
    session_neural.append(...)
```

iii. The agent did not explicitly discuss loop-vectorization tradeoffs. Its unit loop handles ragged spike trains, while tongue lookup itself is vectorized across all requested centers.

## 10-c. What processing does the code repeat multiple times?

i. Each file is opened once and most derived arrays are computed once. It does, however, build `kept_indices` twice and computes population nonzero status after already constructing a preliminary keep mask. Per-trial constants are repeatedly allocated into 80-element arrays.

ii.
```python
kept_indices = np.flatnonzero(keep)
...
keep &= neural_nonzero
kept_indices = np.flatnonzero(keep)
```

iii. The repetition supports an early fewer-than-two check and then a second check after detecting empty neural windows; the trajectory says this second safeguard was added after full-artifact verification found gaps.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `_bin_good_units` bins all behavioral trials before applying `keep`, so rates for invalid, assistance, and later all-zero rejected trials are computed and discarded. It also computes/stores detailed metadata not used by the decoder, though that metadata aids auditing.

ii.
```python
firing_rates_all = _bin_good_units(units, good_indices, go_times)
...
firing_rates = firing_rates_all[keep]
```

iii. The agent chose full-session binning to detect nominally valid but empty final trials. The trajectory shows that this validation safeguard arose from a real alignment failure, but it still performs avoidable work for trials already known to be excluded.
