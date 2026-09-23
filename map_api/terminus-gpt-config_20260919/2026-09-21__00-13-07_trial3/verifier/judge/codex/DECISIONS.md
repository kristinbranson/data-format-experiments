# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by recursively finding every `.nwb` file under `/app/data`, sorting the paths, and processing one file at a time with `pynwb.NWBHDF5IO`. Inside each session it reads the NWB root object, then accesses `nwb.trials` and `nwb.acquisition['BehavioralEvents'].time_series`.

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
...
for p in files:
    res = process_session(p, make_plot)
```

```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
    trials = nwb.trials
    events = nwb.acquisition['BehavioralEvents'].time_series
```

iii. In `CONVERSION_NOTES.md`, the AI says all inspection and conversion must use `pynwb`, that `/app/data` contains one NWB file per session, and that the dataset has a uniform schema across 174 files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from `nwb.subject.subject_id` for each session. After all sessions are processed, unique subject IDs are sorted to form `subjects`, and each session gets an index into that list.

ii.
```python
subject = str(nwb.subject.subject_id)
```

```python
unique_subjects = sorted(set(subjects))
subject_lookup = {x: i for i, x in enumerate(unique_subjects)}
...
'subjects': unique_subjects,
'subject_idx': np.asarray([subject_lookup[r['subject']] for r in results], dtype=np.int64),
```

iii. The notes say the NWB `subject_id` field is the canonical subject identifier and that the release contains 28 subjects.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order follows the sorted file list, and each session carries `nwb.identifier` as its session identifier.

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
```

```python
identifier = str(nwb.identifier)
...
info = dict(identifier=identifier, source_file=str(path), subject=subject, ...)
```

iii. In the notes and trajectory, the AI repeatedly states that `/app/data` is organized as one NWB file per session and that one unlabeled session is skipped only because no units are selected.

## 1-d. How are the data split into trials?

i. Trials are taken directly from `nwb.trials`, with one trial-table row expected per go-cue event. The code does not re-derive trial boundaries from raw events; it checks only that the number of `go_start_times` matches the number of trial rows.

ii.
```python
trials = nwb.trials
events = nwb.acquisition['BehavioralEvents'].time_series
go_all = np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64)
assert len(go_all) == len(trials)
```

iii. The notes say every session has one go-start event per trial and that the trial table supplies the main per-trial metadata, so trial rows are used directly.

## 1-e. How are trials filtered based on quality controls?

i. In the final code, trials are **not** filtered by `obs_intervals`, `free_water`, or other trial-table QC fields. Instead, all trial rows are initially kept, firing rates are computed, and then any trial with zero total spikes across all selected units and all 80 bins is dropped. A session is dropped if it ends with fewer than two remaining trials.

ii.
```python
# obs_intervals encode a paper-analysis subset that strongly excludes error
# trials, not general acquisition validity. Preserve all released trial rows;
# true simultaneous acquisition gaps are detected from zero total raw spikes.
trial_keep = np.ones(len(go_all), dtype=bool)
original_trial_indices = np.arange(len(go_all), dtype=np.int64)
go = go_all
```

```python
rates = bin_spikes(nwb.units, unit_inds, go)
activity_keep = np.any(rates != 0, axis=(1, 2))
excluded_zero_activity_trials = int((~activity_keep).sum())
if activity_keep.sum() < 2:
    return None
rates = rates[activity_keep]
...
original_trial_indices = original_trial_indices[activity_keep]
```

iii. The justification in the notes and trajectory is that `obs_intervals` appeared to behave like an analysis subset that removed many miss/error trials, so the AI chose to preserve all released trial classes and exclude only what it considered objective acquisition gaps: windows with zero spikes across all selected units.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']` for selected units, together with `BehavioralEvents/go_start_times` to place the per-trial windows.

ii.
```python
classification = np.asarray(units['classification'][:]).astype(str)
...
spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
```

```python
go_all = np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64)
```

iii. The notes identify the modality as electrophysiology and say the neural representation should be firing rates computed from raw spike times on the shared NWB session clock.

## 2-b. How is the `neural` data processed?

i. For each selected unit, spike times are binned into 80 non-overlapping 50 ms bins covering `[-2.5, 1.5)` s relative to the go cue. Counts are divided by `BIN` to convert them to firing rates in Hz. The implementation assigns spikes to trial windows by `searchsorted` on trial starts and accumulates counts with `np.bincount`.

ii.
```python
BIN = 0.05
OFF_START, OFF_END = -2.5, 1.5
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = 80
```

```python
starts = go + OFF_START
rates = np.zeros((ntr, nneu, N_TIME), dtype=np.float32)
for k, unit_i in enumerate(unit_inds):
    spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
    trial_i = np.searchsorted(starts, spikes, side='right') - 1
    ...
    bi = np.floor(rel[valid2] / BIN).astype(np.int64)
    flat = ti * N_TIME + bi
    counts = np.bincount(flat, minlength=ntr * N_TIME).reshape(ntr, N_TIME)
    rates[:, k, :] = counts.astype(np.float32) / BIN
```

iii. The notes justify this as the task-mandated 50 ms go-aligned firing-rate representation, replacing the reference paper’s 40 ms / 17 ms sliding windows.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered in two stages. First, only units with `classification == 'good'` are candidates. Second, the AI keeps only classifier-good units whose per-unit `is_good_trials` array is `True` for **every** trial in the session. Sessions with no such units are dropped.

ii.
```python
def selected_unit_indices(units):
    """Paper-matched classifier-good units valid on every trial."""
    classification = np.asarray(units['classification'][:]).astype(str)
    candidates = np.flatnonzero(classification == 'good')
    keep = []
    for j in candidates:
        if np.asarray(units['is_good_trials'][int(j)], dtype=bool).all():
            keep.append(int(j))
    return np.asarray(keep, dtype=np.int64), len(candidates)
```

```python
unit_inds, classifier_good = selected_unit_indices(nwb.units)
if len(unit_inds) == 0:
    return None
```

iii. The notes say this was chosen to keep a fixed neuron set within each session while respecting `is_good_trials`. The AI’s Step 5 notes specifically say dropping 565 units with any invalid trial was cleaner than dropping trials for those units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go-cue onset. The code uses absolute `go_start_times` timestamps, adds the fixed window start offset to define each trial window, and bins spikes relative to that window.

ii.
```python
go_all = np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64)
...
starts = go + OFF_START
```

```python
rel = sp - starts[ti]
valid2 = (rel >= 0) & (rel < (OFF_END - OFF_START))
bi = np.floor(rel[valid2] / BIN).astype(np.int64)
```

iii. The notes say all relevant NWB timestamps already share the same session time base, so alignment only requires selecting the go cue and using go-relative windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 50 ms bins and 80 timepoints per trial. No post hoc temporal rebinning is applied beyond the initial 50 ms binning.

ii.
```python
BIN = 0.05
OFF_START, OFF_END = -2.5, 1.5
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = 80
```

iii. The notes explicitly say 50 ms non-overlapping bins are used because the decoder task overrides the reference paper’s 40 ms / 17 ms analysis bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` and `go_start_times`. For each trial, the code takes the most recent sample/tone onset at or before the go cue.

ii.
```python
sample = np.asarray(events['sample_start_times'].timestamps[:], dtype=np.float64)
sample_i = np.searchsorted(sample, go, side='right') - 1
assert np.all(sample_i >= 0)
tone = sample[sample_i]
```

iii. The AI justifies using the latest sample onset because early licks can replay the sample epoch, and the latest sample event is the final tone preceding that trial’s go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The code first computes the absolute time of each neural bin center (`go + center_offset`) and then subtracts the selected tone onset, producing a continuous time-varying ramp in seconds since tone onset.

ii.
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
time_from_tone = (absolute_centers - tone[:, None]).astype(np.float32)
```

iii. The notes say the decoder asked for “time from tone onset in seconds,” so the AI chose a continuous signal rather than a binary onset pulse.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned on the same 80 go-relative bin centers used for the neural data. The code builds `absolute_centers` from the go cue and uses those same centers both for neural binning and for `time_from_tone`.

ii.
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
time_from_tone = (absolute_centers - tone[:, None]).astype(np.float32)
```

```python
starts = go + OFF_START
...
bi = np.floor(rel[valid2] / BIN).astype(np.int64)
```

iii. The notes say all streams share the NWB session clock, so using the shared go-centered time grid is sufficient for alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the behavioral event streams `photostim_start_times` and `photostim_stop_times`, not from the trial-table `photostim_onset` / `photostim_duration` fields.

ii.
```python
ps = np.asarray(events['photostim_start_times'].timestamps[:], dtype=np.float64)
pe = np.asarray(events['photostim_stop_times'].timestamps[:], dtype=np.float64)
assert len(ps) == len(pe)
```

iii. In Step 5 of the notes, the AI says timestamped event intervals are preferred to string-valued trial fields once equivalence is checked.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Photostimulation is represented as a binary time series at neural bin centers. For each bin center, the code finds the latest stimulation-start event not after that time and marks the bin as 1 only if the bin center is still before the corresponding stop time.

ii.
```python
photostim = np.zeros((len(go), N_TIME), dtype=np.float32)
...
if len(ps):
    event_i = np.searchsorted(ps, absolute_centers, side='right') - 1
    valid_event = event_i >= 0
    safe_i = np.maximum(event_i, 0)
    photostim = (valid_event & (absolute_centers < pe[safe_i])).astype(np.float32)
```

iii. The notes justify this as directly encoding whether photostimulation is on at each decoder time point, which is what the task asked for.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation state is evaluated exactly at the same absolute bin-center timestamps used for the neural bins. The code does not convert stimulation to trial-relative times first; it compares stimulation intervals against `absolute_centers`.

ii.
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
...
photostim = (valid_event & (absolute_centers < pe[safe_i])).astype(np.float32)
```

iii. The notes say the streams are already on the same NWB clock, so direct comparison at shared bin-center timestamps is enough.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the trial-table columns `trial_instruction` and `outcome`. There is no separate stored choice field.

ii.
```python
instruction = np.asarray(trials['trial_instruction'][:]).astype(str)[trial_keep]
outcome_s = np.asarray(trials['outcome'][:]).astype(str)[trial_keep]
```

iii. The notes say a hit implies the instructed side, a miss implies the opposite side, and an ignore implies no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code maps choice to `0=left`, `1=right`, `2=no lick`, then repeats the per-trial choice value across all 80 time bins so that output tensors share shape `(4, 80)`.

ii.
```python
choice = np.full(len(go), 2, dtype=np.int64)  # no lick
choice[(outcome_s == 'hit') & (instruction == 'left')] = 0
choice[(outcome_s == 'hit') & (instruction == 'right')] = 1
choice[(outcome_s == 'miss') & (instruction == 'left')] = 1
choice[(outcome_s == 'miss') & (instruction == 'right')] = 0
```

```python
outputs = [np.stack((np.full(N_TIME, choice[j]), np.full(N_TIME, outcome[j]),
                     np.full(N_TIME, early[j]), tongue_class[j]), axis=0).astype(np.int8)
           for j in range(len(go))]
```

iii. The notes say outcome-derived choice avoids incidental lick-event ambiguity and yields a categorical per-trial variable required by the decoder.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from `trials['outcome']`.

ii.
```python
outcome_s = np.asarray(trials['outcome'][:]).astype(str)[trial_keep]
```

iii. The notes say the NWB trial table already contains the required categories `ignore`, `miss`, and `hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped to integers `ignore=0`, `miss=1`, `hit=2`, then repeated across all 80 time bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.asarray([outcome_map[x] for x in outcome_s], dtype=np.int64)
```

```python
outputs = [np.stack((np.full(N_TIME, choice[j]), np.full(N_TIME, outcome[j]),
                     np.full(N_TIME, early[j]), tongue_class[j]), axis=0).astype(np.int8)
           for j in range(len(go))]
```

iii. The notes say this is a direct categorical encoding of the trial-table outcome.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from `trials['early_lick']`.

ii.
```python
early_s = np.asarray(trials['early_lick'][:]).astype(str)[trial_keep]
```

iii. The notes say the trial table already labels early versus non-early trials, and those trials are retained because early lick is a decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `early` to 1 and `no early` to 0, then repeats that per-trial value across all 80 bins.

ii.
```python
early = (early_s == 'early').astype(np.int64)
```

```python
outputs = [np.stack((np.full(N_TIME, choice[j]), np.full(N_TIME, outcome[j]),
                     np.full(N_TIME, early[j]), tongue_class[j]), axis=0).astype(np.int8)
           for j in range(len(go))]
```

iii. The notes say this is a task-required categorical output, so the AI preserves the class instead of applying the paper’s early-lick exclusion.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`. The code uses the timestamps, the y-coordinate in column 1, and the tracking likelihood in column 2.

ii.
```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_t = np.asarray(tongue_ts.timestamps[:], dtype=np.float64)
tongue_data = np.asarray(tongue_ts.data[:], dtype=np.float64)
y_all, like_all = tongue_data[:, 1], tongue_data[:, 2]
```

iii. The notes say Camera0 tongue tracking is the one uniform tongue source across sessions and that truncated video should be handled by mapping missing coverage to “not visible.”

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI uses nearest-neighbor sampling rather than per-bin averaging. It marks “visible” camera samples as those with finite y and likelihood at least `0.9`, computes session-wide 40th and 60th percentiles from all such visible raw y samples, and then, for each neural bin center, assigns the nearest camera sample if it is within 10 ms.

ii.
```python
LIKELIHOOD_THRESHOLD = 0.9
MAX_VIDEO_GAP = 0.010
```

```python
visible_all = np.isfinite(y_all) & np.isfinite(like_all) & (like_all >= LIKELIHOOD_THRESHOLD)
if visible_all.any():
    p40, p60 = np.percentile(y_all[visible_all], [40, 60])
else:
    p40 = p60 = np.nan
query = absolute_centers
y, gaps, nearest_i = nearest_values(query, tongue_t, y_all)
nearest_like = like_all[nearest_i] if len(tongue_t) >= 2 else np.full(query.shape, np.nan)
visible = (gaps <= MAX_VIDEO_GAP) & np.isfinite(y) & np.isfinite(nearest_like) & (nearest_like >= LIKELIHOOD_THRESHOLD)
```

iii. The notes justify this with a conservative DeepLabCut-style confidence threshold (`0.9`), explicit temporal coverage handling for truncated videos, and the claim that the tongue is visible only intermittently so low-confidence or uncovered bins should become “not visible.”

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The code computes `p40` and `p60` from all visible raw y samples in the session. Then each visible bin center is mapped to class `0` if `y < p40`, `1` if `p40 <= y <= p60`, `2` if `y > p60`, and `3` if no acceptable visible sample is available.

ii.
```python
if visible_all.any():
    p40, p60 = np.percentile(y_all[visible_all], [40, 60])
else:
    p40 = p60 = np.nan
...
tongue_class = np.full(query.shape, 3, dtype=np.int64)
tongue_class[visible & (y < p40)] = 0
tongue_class[visible & (y >= p40) & (y <= p60)] = 1
tongue_class[visible & (y > p60)] = 2
```

iii. The notes say the percentiles are computed per session “over the session,” and class 3 is used for low-confidence or out-of-coverage bins instead of imputing values.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is aligned to the neural data by evaluating tongue state at the same go-centered neural bin centers. The code does this by querying the nearest camera sample to each `absolute_centers` time and accepting it only when the nearest sample is within 10 ms.

ii.
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
...
query = absolute_centers
y, gaps, nearest_i = nearest_values(query, tongue_t, y_all)
...
visible = (gaps <= MAX_VIDEO_GAP) & ...
```

iii. The notes justify this as a common-clock alignment strategy and say bins outside video coverage should be mapped to the explicit “not visible” class.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases explicitly: sessions with no selected units are dropped; truncated or missing tongue coverage is converted to the “not visible” class via the nearest-sample gap and likelihood checks; and trials with zero total spikes across all selected units are removed. The code also has a fallback for tongue series shorter than two timestamps.

ii.
```python
unit_inds, classifier_good = selected_unit_indices(nwb.units)
if len(unit_inds) == 0:
    return None
```

```python
if len(timestamps) < 2:
    shape = query.shape
    return np.full(shape, np.nan), np.full(shape, np.inf), np.zeros(shape, dtype=np.int64)
```

```python
activity_keep = np.any(rates != 0, axis=(1, 2))
excluded_zero_activity_trials = int((~activity_keep).sum())
if activity_keep.sum() < 2:
    return None
```

iii. The notes say missing tongue measurements should not cause session drops, while all-zero neural windows are treated as acquisition gaps and unlabeled sessions are dropped because they lack usable classifier-good units.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work in the final code is opening and reading each NWB file, loading large spike-time and tongue-tracking arrays, looping through selected units in `bin_spikes`, and writing the final pickle. Optional plotting also adds overhead when enabled.

ii.
```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
```

```python
for k, unit_i in enumerate(unit_inds):
    spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
    ...
    counts = np.bincount(flat, minlength=ntr * N_TIME).reshape(ntr, N_TIME)
```

```python
tongue_data = np.asarray(tongue_ts.data[:], dtype=np.float64)
...
with out.open('wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes say I/O dominates, followed by spike binning and tongue-array reads, with full conversion taking about 4 minutes and pickle writing around 12 seconds.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious loops are over units in `selected_unit_indices`, `bin_spikes`, and `coarse_regions`, plus per-trial list construction for `inputs`, `outputs`, and `neural`. The AI already vectorized the trial/time dimension of spike binning and used vectorized nearest-neighbor lookup for tongue alignment, so it did not leave a per-trial tongue loop.

ii.
```python
for j in candidates:
    if np.asarray(units['is_good_trials'][int(j)], dtype=bool).all():
        keep.append(int(j))
```

```python
for k, unit_i in enumerate(unit_inds):
    spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
    ...
```

```python
inputs = [np.stack((time_from_tone[j], photostim[j]), axis=0).astype(np.float32)
          for j in range(len(go))]
outputs = [np.stack((np.full(N_TIME, choice[j]), np.full(N_TIME, outcome[j]),
                     np.full(N_TIME, early[j]), tongue_class[j]), axis=0).astype(np.int8)
           for j in range(len(go))]
```

iii. The notes emphasize that a naive nested trial-by-unit histogram loop would have been too slow, so the AI kept only loops it considered hard to remove because of ragged spike trains or per-unit metadata access.

## 10-c. What processing does the code repeat multiple times?

i. Within one run of the final script, very little is intentionally recomputed. Each session is opened once, processed once, and appended once. The shared time grid is defined once at module scope and reused across all sessions.

ii.
```python
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```

```python
for p in files:
    res = process_session(p, make_plot)
    ...
    results.append(res)
```

iii. The notes explicitly frame the conversion as a one-pass per-session pipeline and contrast it with repeated exploratory reruns that happened during development, not inside the final script itself.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main extra processing is for debugging, plotting, and metadata rather than for the decoder inputs themselves. The code computes nearest raw tongue `y` and likelihood arrays for every bin so plots can be drawn; it collects `fine_regions`, `unit_ids`, and multiple session-level QC counters into `metadata['session_info']`; and it defines an unused `fully_observed_trial_mask` helper that is never called.

ii.
```python
def fully_observed_trial_mask(units, unit_inds, go):
    ...
    return keep
```

```python
y, gaps, nearest_i = nearest_values(query, tongue_t, y_all)
nearest_like = like_all[nearest_i] if len(tongue_t) >= 2 else np.full(query.shape, np.nan)
...
if make_plot:
    plot_info = dict(..., tongue_y=y, tongue_like=nearest_like, ...)
```

```python
fine_regions = np.asarray(nwb.units['anno_name'][:]).astype(str)[unit_inds]
unit_ids = np.asarray(nwb.units.id[:])[unit_inds].tolist()
...
'session_info': [r['info'] for r in results],
```

iii. The notes justify these additions as sanity-check and audit support. They are not needed by the downstream decoder itself, but were used for validation, figures, and richer metadata.
