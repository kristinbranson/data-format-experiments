# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all session files by recursively scanning `/app/data` for `*.nwb` files, then opens each file directly with `h5py` inside `session_from_file`. Within each file it reads HDF5 groups such as `units`, `intervals/trials`, and `acquisition`.

ii. 
```python
DATA_DIR = Path('/app/data')
...
files = sorted(DATA_DIR.rglob('*.nwb'))
```

```python
with h5py.File(path, 'r') as h:
    units = h['units']
    trials = h['intervals']['trials']
    acq = h['acquisition']
```

iii. In `CONVERSION_NOTES.md`, the agent documented that the dataset is "native NWB/HDF5 electrophysiology + behavior + optogenetics data" and chose to work from the HDF5 layout directly. The Step 5 mapping notes and trajectory show it treated recursive file discovery plus direct group access as sufficient to cover the full dataset.

## 1-b. How are the data split into subjects?

i. The agent uses the subject folder name, not the NWB subject metadata. It strips the `sub-` prefix from `path.parent.name`, appends unique subject ids to `subjects` in encounter order, and records `subject_idx` by position in that list.

ii.
```python
subject = path.parent.name.replace('sub-', '')
```

```python
if subj not in subjects:
    subjects.append(subj)
subject_idx.append(subjects.index(subj))
```

iii. The Step 2 notes say the files are organized as `/app/data/sub-<subject_id>/...`, and the agent treated that folder name as the subject identifier. No separate justification from the trajectory was given beyond relying on the dataset layout.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. The dataset is processed file-by-file, and each surviving file contributes one session entry to `neural`, `input`, `output`, and `brain_region_idx`.

ii.
```python
for i, f in enumerate(files):
    neural, inp, out, sess_regions, subj, info = session_from_file(f, make_plot=show_processing and i < 2)
    ...
    neural_all.append(neural)
    input_all.append(inp)
    output_all.append(out)
```

iii. The Step 2 notes describe the data as "NWB files organized by subject folders" with one session file per recording. The trajectory repeatedly refers to "session files" and full conversion over 174 files, so file boundaries were used as session boundaries.

## 1-d. How are the data split into trials?

i. Trials are indexed primarily by the `go_start_times` event series. The agent sets `n_trials = len(go_times)`, slices trial-table columns to that length, and iterates `for tr in range(n_trials)`. It does not assert that the trials table length matches the number of go cues.

ii.
```python
go_times = np.asarray(beh_events['go_start_times']['timestamps'][()]).astype(float)
...
n_trials = len(go_times)
trial_instruction = _decode_arr(trials['trial_instruction'][()])[:n_trials]
outcome = _decode_arr(trials['outcome'][()])[:n_trials]
early_lick = _decode_arr(trials['early_lick'][()])[:n_trials]
```

```python
for tr in range(n_trials):
    if not valid_trial_mask[tr]:
        continue
    gt = go_times[tr]
```

iii. The Step 5 mapping notes say the decoder should align to `BehavioralEvents/go_start_times`, and the trajectory shows the agent focused on go-cue indexing while debugging. It appears to have assumed one go cue per trial and used the go cue count as the authoritative trial count.

## 1-e. How are trials filtered based on quality controls?

i. The final code filters trials using `units['is_good_trials']` when present. It keeps only trial columns covered by that array and only if at least some good units are marked good on that trial. After constructing per-trial neural matrices, it also drops any trial whose neural data are all zeros. It does not use `obs_intervals` or explicitly exclude `free_water` trials.

ii.
```python
valid_trial_mask = np.ones(n_trials, dtype=bool)
if 'is_good_trials' in units:
    igt = np.asarray(units['is_good_trials'][()])
    n_valid_cols = min(igt.shape[1], n_trials)
    trial_good_frac = igt[good_inds, :n_valid_cols].mean(axis=0) if len(good_inds) else np.zeros(n_valid_cols)
    valid_trial_mask[:] = False
    valid_trial_mask[:n_valid_cols] = trial_good_frac > 0
```

```python
for tr in range(n_trials):
    if not valid_trial_mask[tr]:
        continue
    ...
    if np.all(neural == 0):
        continue
```

iii. The Step 10 notes and trajectory give the justification: verification found many all-zero neural trials; the agent traced that to sessions where `is_good_trials` had fewer columns than go-cue trials, then restricted trials to valid `is_good_trials` coverage and dropped any residual all-zero trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from `units['spike_times']` for units selected by `classification == 'good'`, aligned to `BehavioralEvents/go_start_times`.

ii.
```python
unit_class = _decode_arr(units['classification'][()]) if 'classification' in units else None
good_mask = unit_class == 'good' if unit_class is not None else np.ones(len(units['id']), dtype=bool)
good_inds = np.flatnonzero(good_mask)
good_spike_times = [get_unit_spike_times(units, int(ui)) for ui in good_inds]
...
go_times = np.asarray(beh_events['go_start_times']['timestamps'][()]).astype(float)
```

iii. The Step 4 and Step 5 notes explicitly justify `classification == 'good'` as the paper-consistent neuron curation rule, and the trajectory says the close match to the paper's good-unit count was the key reason to use that QC field.

## 2-b. How is the `neural` data processed?

i. For each retained trial and each retained unit, the agent subtracts the trial's go cue from spike times, bins the relative spike times with `np.histogram` into 50 ms bins over `[-2.5, 1.5]`, and converts counts to rates by dividing by `BIN_SIZE`. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
BIN_SIZE = 0.05
WIN_START = -2.5
WIN_END = 1.5
N_BINS = int(round((WIN_END - WIN_START) / BIN_SIZE))
BIN_EDGES = np.linspace(WIN_START, WIN_END, N_BINS + 1)
```

```python
neural = np.zeros((len(good_inds), N_BINS), dtype=np.float32)
for j, st in enumerate(good_spike_times):
    rel = st - gt
    counts, _ = np.histogram(rel, bins=BIN_EDGES)
    neural[j] = counts.astype(np.float32) / BIN_SIZE
```

iii. The Step 5 notes say the neural signal should be "Bin spike times into 50 ms spike counts or rates over [-2.5, +1.5] s relative to `go_start_times`," using paper-consistent QC but task-required binning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by `classification == 'good'`. If `classification` is absent, the code falls back to keeping all units. Sessions with no usable neural trials after later filtering are skipped in `build_dataset`.

ii.
```python
unit_class = _decode_arr(units['classification'][()]) if 'classification' in units else None
good_mask = unit_class == 'good' if unit_class is not None else np.ones(len(units['id']), dtype=bool)
good_inds = np.flatnonzero(good_mask)
```

```python
if len(neural) < 2 or neural[0].shape[0] == 0:
    print(f'skipping {f} due to insufficient trials or neurons', flush=True)
    continue
```

iii. The Step 4 and Step 5 notes justify the classifier-based filter because it reproduced the paper's reported good-unit totals much better than other QC fields. The fallback behavior for missing `classification` was not separately justified.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to its go cue. The code builds absolute bin edges and centers by adding the shared relative bin grid to the trial's `go_start_times` timestamp, then bins spikes relative to that go cue.

ii.
```python
gt = go_times[tr]
abs_edges = gt + BIN_EDGES
abs_centers = gt + BIN_CENTERS
```

```python
for j, st in enumerate(good_spike_times):
    rel = st - gt
    counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. The Step 5 notes list temporal alignment to `BehavioralEvents/go_start_times` as a key decision because the decoder task explicitly required go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins across a 4 s window from -2.5 s to +1.5 s around go cue, giving 80 bins per trial. The code bins raw spikes directly into that grid; there is no second rebinning stage.

ii.
```python
BIN_SIZE = 0.05
WIN_START = -2.5
WIN_END = 1.5
N_BINS = int(round((WIN_END - WIN_START) / BIN_SIZE))
BIN_EDGES = np.linspace(WIN_START, WIN_END, N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. The Step 5 notes explicitly say the reference code's finer 40 ms / 3.4 ms preprocessing was overridden because this decoder task specified 50 ms bins and go-cue alignment.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times` and `go_start_times`. If the sample event series is missing or its length does not match `n_trials`, the agent infers a per-trial sample time as the last sample event sufficiently before the go cue, with a hard-coded fallback of `go - 0.6`.

ii.
```python
sample_times = np.asarray(beh_events['sample_start_times']['timestamps'][()]).astype(float) if 'sample_start_times' in beh_events else None
...
if sample_times is None or len(sample_times) != n_trials:
    sample_times = infer_trial_event_times(sample_times, go_times)
else:
    sample_times = infer_trial_event_times(sample_times[:], go_times)
```

```python
def infer_trial_event_times(event_ts, go_times, default_offset=-0.6, min_pre=0.05, max_pre=5.0):
    ...
    if event_ts is None or len(event_ts) == 0:
        return go_times + default_offset
```

iii. The Step 5 notes mapped this input to "sample/tone onset" plus go cue. The Step 10 notes justify the fallback logic as a fix for NaN/Inf inputs found during verification.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the code computes the absolute center time of each 50 ms neural bin and subtracts the estimated sample/tone onset time for that trial. This yields a continuous time-since-tone trace per bin.

ii.
```python
stime = sample_times[tr] if np.isfinite(sample_times[tr]) else (gt - 0.6)
time_from_tone = abs_centers - stime
```

```python
inp = np.vstack([time_from_tone.astype(np.float32), photostim_on.astype(np.float32)])
```

iii. The Step 5 notes say this variable should be "Continuous time-from-tone-onset per bin." The later Step 10 notes show the agent added a finite fallback so verification would not fail on missing values.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on the same `abs_centers` array used to describe the aligned neural bins, so each input timepoint corresponds to the center of a neural 50 ms bin.

ii.
```python
abs_centers = gt + BIN_CENTERS
...
time_from_tone = abs_centers - stime
```

iii. The Step 5 notes explicitly planned all trial variables on the go-cue-aligned bin grid, so the tone-onset input shares the neural alignment by construction.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The code primarily uses the trial-table fields `photostim_onset` and `photostim_duration`. If those fields are absent, it falls back to the event series `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`.

ii.
```python
photostim_start = np.asarray(beh_events['photostim_start_times']['timestamps'][()]).astype(float) if 'photostim_start_times' in beh_events else np.array([])
photostim_stop = np.asarray(beh_events['photostim_stop_times']['timestamps'][()]).astype(float) if 'photostim_stop_times' in beh_events else np.array([])
```

```python
if 'photostim_duration' in trials:
    pdur = _decode_arr(trials['photostim_duration'][()])[:n_trials]
    pon = _decode_arr(trials['photostim_onset'][()])[:n_trials] if 'photostim_onset' in trials else np.array(['N/A'] * n_trials, dtype=object)
else:
    pdur = np.array(['N/A'] * n_trials, dtype=object)
    pon = np.array(['N/A'] * n_trials, dtype=object)
```

iii. The Step 5 notes said this input could come from `BehavioralEvents/photostim_start_times` and `photostim_stop_times` "or trial photostim fields," with the goal of building a time-varying on/off regressor.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the code initializes an all-zero binary vector. If the trial-table onset and duration are not `'N/A'`, it treats `photostim_onset` as a numeric time, converts it to float, and marks bins whose absolute centers fall inside `[pstart, pstart + duration)`. Otherwise it scans all session photostim start/stop intervals and marks any overlap with the trial window.

ii.
```python
photostim_on = np.zeros(N_BINS, dtype=np.float32)
if pon[tr] != 'N/A' and pdur[tr] != 'N/A':
    try:
        pstart = float(pon[tr])
        pd = float(pdur[tr])
        photostim_on = ((abs_centers >= pstart) & (abs_centers < pstart + pd)).astype(np.float32)
    except Exception:
        pass
else:
    for ps, pe in zip(photostim_start, photostim_stop):
        if pe >= abs_edges[0] and ps <= abs_edges[-1]:
            photostim_on |= ((abs_centers >= ps) & (abs_centers < pe))
    photostim_on = photostim_on.astype(np.float32)
```

iii. The Step 5 mapping says photostimulation should be a binary time-varying input. The sample notes also say some sessions had all-zero photostim channels, and the trajectory shows the agent wanted a fallback for sessions lacking explicit optogenetic fields.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The code compares photostim times against `abs_centers`, the absolute-time centers of the go-cue-aligned neural bins. Alignment is therefore done by putting photostim intervals and neural bin centers on the same absolute session clock.

ii.
```python
abs_centers = gt + BIN_CENTERS
...
photostim_on = ((abs_centers >= pstart) & (abs_centers < pstart + pd)).astype(np.float32)
```

iii. The Step 5 notes explicitly intended to "derive photostimulation on/off from photostim start/stop events," so the justification was to express the light as a time series on the same grid as the neural bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The implemented code derives choice from the event series `left_lick_times` and `right_lick_times`, using the first post-go lick in the response window. It does not use `trial_instruction` or `outcome` to derive choice, even though those trial-table variables are loaded.

ii.
```python
left_lick_times = np.asarray(beh_events['left_lick_times']['timestamps'][()]).astype(float) if 'left_lick_times' in beh_events else np.array([])
right_lick_times = np.asarray(beh_events['right_lick_times']['timestamps'][()]).astype(float) if 'right_lick_times' in beh_events else np.array([])
```

```python
lmask = (left_lick_times >= gt) & (left_lick_times < gt + WIN_END)
rmask = (right_lick_times >= gt) & (right_lick_times < gt + WIN_END)
lfirst = left_lick_times[lmask][0] if np.any(lmask) else np.inf
rfirst = right_lick_times[rmask][0] if np.any(rmask) else np.inf
```

iii. The Step 5 "Key Decisions" section says: "Define choice from actual lick behavior when possible (left/right lick times after go cue); assign `no lick` when no lick is observed in the response period or outcome is ignore." The trajectory presents this as a deliberate behavioral choice definition.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The first left or right lick after the go cue and before `go + 1.5 s` determines the class: `0` for left, `1` for right, and `2` for no lick if neither side licks first. That per-trial class is then repeated across all 80 bins.

ii.
```python
if np.isfinite(lfirst) and (lfirst < rfirst):
    choice = 0  # left
elif np.isfinite(rfirst) and (rfirst < lfirst):
    choice = 1  # right
else:
    choice = 2  # no lick
```

```python
out = np.vstack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, out_val, dtype=np.int64),
    np.full(N_BINS, early_val, dtype=np.int64),
    tongue_disc.astype(np.int64),
])
```

iii. The Step 5 notes justify this as using actual behavior rather than an inferred choice label. The code treats choice as a per-trial quantity, so it is repeated across bins to fit the common `(n_output, n_timepoints)` format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials-table `outcome` column.

ii.
```python
outcome = _decode_arr(trials['outcome'][()])[:n_trials]
```

iii. The Step 5 notes mapped `outcome` directly from the NWB trial table, with no extra derivation.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent maps string values to integers with `{'ignore': 0, 'miss': 1, 'hit': 2}` and repeats the per-trial code across all bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = outcome_map.get(outcome[tr], 0)
```

```python
np.full(N_BINS, out_val, dtype=np.int64)
```

iii. The Step 5 notes say the trial-table labels already match the desired categories, so the only processing needed is categorical encoding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick comes directly from the trials-table `early_lick` column.

ii.
```python
early_lick = _decode_arr(trials['early_lick'][()])[:n_trials]
```

iii. The Step 5 notes mapped `early_lick` directly from the NWB trial table as a per-trial categorical variable.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `'no early'` to `0` and `'early'` to `1`, then repeats the resulting trial label across all time bins.

ii.
```python
early_map = {'no early': 0, 'early': 1}
early_val = early_map.get(early_lick[tr], 0)
```

```python
np.full(N_BINS, early_val, dtype=np.int64)
```

iii. The Step 5 notes say no collapse of categories was needed because the trial table already provides the requested binary label.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking/data` and its timestamps. The code assumes the tracking data columns are `(x, y, likelihood)` and selects columns 1 and 2 as `tongue_y` and visibility/confidence.

ii.
```python
tongue = np.asarray(beh_ts['Camera0_side_TongueTracking']['data'][()]).astype(float)
tongue_t = np.asarray(beh_ts['Camera0_side_TongueTracking']['timestamps'][()]).astype(float)
```

```python
def choose_tongue_columns(tongue_data):
    if tongue_data.shape[1] < 3:
        raise ValueError('Tongue tracking data expected to have at least 3 columns')
    # assume x, y, likelihood
    return 1, 2
```

iii. The Step 5 notes said the channel layout was "likely x,y,likelihood with low likelihood => not visible." The Step 7 notes acknowledge heavy use of the not-visible class and describe the threshold as possibly conservative.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent marks frames as visible when `tongue_y` and confidence are finite and confidence exceeds `0.5`. It then computes session-wide 40th and 60th percentiles from all visible raw `tongue_y` frames. For each trial and bin, it picks the nearest camera sample to the neural bin center and classifies that single frame's `tongue_y`; bins with invisible samples are assigned class `3`.

ii.
```python
visible = np.isfinite(tongue_y) & np.isfinite(tongue_vis) & (tongue_vis > 0.5)
if np.any(visible):
    q40, q60 = np.quantile(tongue_y[visible], [0.4, 0.6])
else:
    q40, q60 = 0.0, 0.0
```

```python
idx = np.searchsorted(tongue_t, abs_centers, side='left')
idx = np.clip(idx, 0, len(tongue_t) - 1)
ty = tongue_y[idx]
tv = tongue_vis[idx]
tongue_disc = np.full(N_BINS, 3, dtype=np.int64)
vis_now = np.isfinite(ty) & np.isfinite(tv) & (tv > 0.5)
```

iii. The Step 5 notes justify using session-wide percentiles over visible frames only, with class `3` for not visible. The trajectory and sample notes suggest the agent considered the high not-visible fraction acceptable, though possibly conservative.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Using session-wide thresholds `q40` and `q60`, visible bins are discretized to `0` if `y < q40`, `1` if `q40 <= y <= q60`, and `2` if `y > q60`. Non-visible bins stay at `3`.

ii.
```python
tongue_disc = np.full(N_BINS, 3, dtype=np.int64)
vis_now = np.isfinite(ty) & np.isfinite(tv) & (tv > 0.5)
tongue_disc[vis_now & (ty < q40)] = 0
tongue_disc[vis_now & (ty >= q40) & (ty <= q60)] = 1
tongue_disc[vis_now & (ty > q60)] = 2
```

iii. The Step 5 notes explicitly planned the 40th/60th percentile discretization and a fourth "not visible" class to satisfy the decoder task requirements.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The code aligns tongue output by looking up the first camera frame at or after each neural bin center in absolute time. It does not average all camera frames within a 50 ms neural bin.

ii.
```python
abs_centers = gt + BIN_CENTERS
...
idx = np.searchsorted(tongue_t, abs_centers, side='left')
idx = np.clip(idx, 0, len(tongue_t) - 1)
ty = tongue_y[idx]
tv = tongue_vis[idx]
```

iii. The Step 5 notes say tongue tracking should be aligned to trial bins, but the implementation chose nearest-sample lookup rather than within-bin averaging. The trajectory does not provide a separate justification beyond wanting aligned trial-bin outputs.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses several defensive fallbacks. Missing or mismatched sample/tone events are replaced by an inferred last-preceding event or, failing that, `go - 0.6`. Missing `classification` causes all units to be kept. Missing photostim fields yield zeros or event-based fallback. Low-confidence tongue samples become class `3`. Trials outside `is_good_trials` coverage and residual all-zero neural trials are dropped.

ii.
```python
def infer_trial_event_times(event_ts, go_times, default_offset=-0.6, min_pre=0.05, max_pre=5.0):
    ...
    if event_ts is None or len(event_ts) == 0:
        return go_times + default_offset
```

```python
good_mask = unit_class == 'good' if unit_class is not None else np.ones(len(units['id']), dtype=bool)
```

```python
if not valid_trial_mask[tr]:
    continue
...
if np.all(neural == 0):
    continue
```

iii. The Step 10 notes justify these changes as fixes discovered during verification: a "robust finite tone-onset fallback" was added for NaN/Inf inputs, and `is_good_trials` plus all-zero-trial dropping were added after warnings about zero neural trials. The overall pattern is pragmatic fallback handling rather than strict exclusion based on NWB semantics.

## 10-a. What are the most time-consuming steps of the code?

i. The code is dominated by per-session loading of large spike and tongue arrays and, within each trial, a nested loop over all good units that calls `np.histogram` once per unit. The trajectory also notes long runtimes and later caching of spike times per good unit.

ii.
```python
good_spike_times = [get_unit_spike_times(units, int(ui)) for ui in good_inds]
```

```python
for tr in range(n_trials):
    ...
    neural = np.zeros((len(good_inds), N_BINS), dtype=np.float32)
    for j, st in enumerate(good_spike_times):
        rel = st - gt
        counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. The Step 10 notes mention "Performance regression from per-trial spike extraction: mitigated by caching spike times per good unit once per session," and the trajectory describes the full conversion as very long, confirming that spike extraction/binning was a major bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main missed vectorization opportunities are the outer trial loop combined with the inner per-unit histogram loop for neural data, the per-trial photostim interval scan in the fallback branch, and the per-trial nearest-sample tongue lookup/classification. These are all implemented serially.

ii.
```python
for tr in range(n_trials):
    ...
    for j, st in enumerate(good_spike_times):
        rel = st - gt
        counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

```python
for ps, pe in zip(photostim_start, photostim_stop):
    if pe >= abs_edges[0] and ps <= abs_edges[-1]:
        photostim_on |= ((abs_centers >= ps) & (abs_centers < pe))
```

iii. The Step 6 instructions explicitly asked for vectorization and timing work, but the final code keeps multiple Python-level loops. The trajectory shows some optimization effort, but mostly by caching spike times rather than restructuring the binning.

## 10-c. What processing does the code repeat multiple times?

i. It repeats several computations across trials: spike-time re-centering and histogramming for every unit on every trial, repeated fallback search for sample/tone times, and repeated nearest-frame tongue lookup per trial. It also decodes several trial columns up front even when some are only used in fallback branches.

ii.
```python
for j, st in enumerate(good_spike_times):
    rel = st - gt
    counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

```python
if sample_times is None or len(sample_times) != n_trials:
    sample_times = infer_trial_event_times(sample_times, go_times)
else:
    sample_times = infer_trial_event_times(sample_times[:], go_times)
```

```python
idx = np.searchsorted(tongue_t, abs_centers, side='left')
idx = np.clip(idx, 0, len(tongue_t) - 1)
```

iii. The trajectory shows that the agent later recognized repeated spike extraction as a bottleneck and added `good_spike_times` caching, but the main trial-by-trial recomputation still remains.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some extra work beyond what is needed for the final stored dataset: it loads both trial-table and event-stream photostim representations even though one would suffice, loads left and right lick event streams instead of deriving choice from trial labels, computes optional plotting outputs in `--show-processing` mode, and tracks `load_time_sec` metadata that is not used downstream by the decoder.

ii.
```python
photostim_start = np.asarray(beh_events['photostim_start_times']['timestamps'][()]).astype(float) if 'photostim_start_times' in beh_events else np.array([])
photostim_stop = np.asarray(beh_events['photostim_stop_times']['timestamps'][()]).astype(float) if 'photostim_stop_times' in beh_events else np.array([])
left_lick_times = np.asarray(beh_events['left_lick_times']['timestamps'][()]).astype(float) if 'left_lick_times' in beh_events else np.array([])
right_lick_times = np.asarray(beh_events['right_lick_times']['timestamps'][()]).astype(float) if 'right_lick_times' in beh_events else np.array([])
```

```python
if make_plot:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ...
    fig.savefig(f'/app/processing_{path.stem}.png', dpi=150)
```

iii. The notes do not explicitly call this out as unnecessary work, but the trajectory shows the agent spent effort on plotting and on alternative fallback paths that are not required by the final decoder format itself.
