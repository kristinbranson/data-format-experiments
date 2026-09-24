# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds every NWB file below `/app/data`, sorts the paths, and processes each file as one session with `h5py`. It reads units, trials, behavioral events, behavioral time series, and acquisition data from each file.

ii.
```python
files = sorted(DATA_DIR.rglob('*.nwb'))
with h5py.File(path, 'r') as h:
    units = h['units']
    trials = h['intervals']['trials']
    acq = h['acquisition']
```

iii. The notes identify the native layout as NWB files organized under subject directories and report 174 sessions and 28 subjects. Direct HDF5 access was chosen for the conversion.

## 1-b. How are the data split into subjects?

i. A subject ID is parsed from each file's parent directory (`sub-...`). Subjects are accumulated in first-seen sorted-file order, and each retained session receives the corresponding index.

ii.
```python
subject = path.parent.name.replace('sub-', '')
if subj not in subjects:
    subjects.append(subj)
subject_idx.append(subjects.index(subj))
```

iii. The notes state that the directory organization is `sub-<subject_id>` and contains 28 subjects, so the agent treated the directory name as the canonical grouping key.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Retained per-file results are appended as one element of each top-level session list.

ii.
```python
for i, f in enumerate(files):
    neural, inp, out, sess_regions, subj, info = session_from_file(f, ...)
    neural_all.append(neural)
```

iii. The agent's exploration concluded that the dataset contains one behavior/ecephys/optogenetics NWB file per session.

## 1-d. How are the data split into trials?

i. The number and ordering of trials are taken from `go_start_times`. Trial-table arrays are sliced to that count, and one converted item is constructed per go cue, subject to filtering.

ii.
```python
go_times = np.asarray(beh_events['go_start_times']['timestamps'][()]).astype(float)
n_trials = len(go_times)
trial_instruction = _decode_arr(trials['trial_instruction'][()])[:n_trials]
for tr in range(n_trials):
```

iii. The agent regarded go cues and trial-table rows as trial-aligned and used go cue onset as the required alignment event.

## 1-e. How are trials filtered based on quality controls?

i. A trial is retained if at least one classifier-good unit has a positive `is_good_trials` entry for that trial. Columns absent from that matrix are rejected. After conversion, any trial whose complete neural matrix is zero is also discarded; sessions with fewer than two retained trials are skipped. The code does not explicitly remove `free_water` trials or match trials to `obs_intervals`.

ii.
```python
trial_good_frac = igt[good_inds, :n_valid_cols].mean(axis=0) if len(good_inds) else np.zeros(n_valid_cols)
valid_trial_mask[:n_valid_cols] = trial_good_frac > 0
...
if np.all(neural == 0):
    continue
```

iii. The notes say this was added after full validation exposed late all-zero trials where `is_good_trials` had fewer columns than behavioral trials; residual all-zero trials were then dropped. The agent intended to retain non-regular trials because early lick, outcome, and stimulation are decoder targets.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from `units/spike_times`, restricted using `units/classification`; go-cue timestamps define trial windows.

ii.
```python
unit_class = _decode_arr(units['classification'][()])
good_inds = np.flatnonzero(unit_class == 'good')
good_spike_times = [get_unit_spike_times(units, int(ui)) for ui in good_inds]
```

iii. The notes identify raw spike times and the classifier QC verdict as the appropriate source, matching the paper's classifier-based curation.

## 2-b. How is the `neural` data processed?

i. For every retained trial and good unit, spike times are shifted by the go cue, histogrammed into fixed bins, and divided by 0.05 s to produce firing rates in Hz. There is no smoothing or normalization.

ii.
```python
rel = st - gt
counts, _ = np.histogram(rel, bins=BIN_EDGES)
neural[j] = counts.astype(np.float32) / BIN_SIZE
```

iii. The agent notes that the reference uses binned firing rates and that the decoder specification overrides the reference video's finer sliding-window parameters with 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units whose decoded `classification` equals `good` are retained. If the column were absent, the code would retain all units; sessions with no retained neurons are later skipped.

ii.
```python
good_mask = unit_class == 'good' if unit_class is not None else np.ones(len(units['id']), dtype=bool)
good_inds = np.flatnonzero(good_mask)
```

iii. The notes explicitly select classifier QC because the method-paper pipeline used `qc_mode='classifier'` and the expected total is about 69,943 good units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each unit's session-absolute spike times are shifted by that trial's go-cue timestamp, then binned from -2.5 to +1.5 s.

ii.
```python
gt = go_times[tr]
rel = st - gt
counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. Go cue onset is the required alignment event, and the NWB event and spike timestamps share a clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 50 ms: 80 non-overlapping bins across four seconds. Raw spikes are directly binned at this resolution; no later rebinning is applied.

ii.
```python
BIN_SIZE = 0.05
WIN_START = -2.5
WIN_END = 1.5
N_BINS = int(round((WIN_END - WIN_START) / BIN_SIZE))
```

iii. The notes say this follows the decoder task even though the method code used 40 ms windows and 3.4 ms strides for a different analysis.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, `go_start_times`, and the aligned bin centers. The most recent plausible sample event before each go cue is selected.

ii.
```python
sample_times = np.asarray(beh_events['sample_start_times']['timestamps'][()]).astype(float)
sample_times = infer_trial_event_times(sample_times, go_times)
```

iii. The notes recognize sample onset as tone onset and describe nearest-preceding matching to handle replayed epochs and mismatched event counts.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The helper searches 0.05–5 s before each go cue for the last sample event. If none exists it fabricates an onset 0.6 s before the cue. Absolute bin centers minus that onset give seconds from tone.

ii.
```python
cand = event_ts[(event_ts <= gt - min_pre) & (event_ts >= gt - max_pre)]
out[i] = cand[-1] if cand.size else gt + default_offset
...
time_from_tone = abs_centers - stime
```

iii. Full verification found NaN inputs, so the agent added robust preceding-event matching and a finite approximately 0.6 s fallback.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same absolute centers (`go + BIN_CENTERS`) as the neural bins.

ii.
```python
abs_centers = gt + BIN_CENTERS
time_from_tone = abs_centers - stime
```

iii. The agent intended all time-varying channels to share the go-cue-centered 80-bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The primary source is trial-table `photostim_onset` and `photostim_duration`. When those contain `N/A`, session-absolute `photostim_start_times` and `photostim_stop_times` behavioral events are used.

ii.
```python
pdur = _decode_arr(trials['photostim_duration'][()])[:n_trials]
pon = _decode_arr(trials['photostim_onset'][()])[:n_trials]
photostim_start = np.asarray(beh_events['photostim_start_times']['timestamps'][()])
```

iii. The notes explicitly planned this dual-source strategy to handle `N/A` trial fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary vector is set to one where a bin center lies in the half-open stimulation interval. For populated trial fields, the code converts their strings directly to floats and treats onset as an absolute time; otherwise it checks all event onset/offset pairs overlapping the trial.

ii.
```python
pstart = float(pon[tr])
pd = float(pdur[tr])
photostim_on = ((abs_centers >= pstart) & (abs_centers < pstart + pd)).astype(np.float32)
```

iii. The justification was to represent stimulation as the requested time-varying binary input. The notes did not justify treating the trial-table onset as session-absolute.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Event-series timestamps are compared to the same absolute bin centers as neural data. However, the trial-table path compares a trial-relative onset directly with absolute centers, without adding `trials/start_time`, so that path is misaligned.

ii.
```python
photostim_on = ((abs_centers >= pstart) & (abs_centers < pstart + pd))
```

iii. The agent stated that the go-cue bin and photostimulation start/stop times should be checked together, but its documented sanity-check box was not completed and the clock conversion is absent.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the first left or right lick event in `[go, go + 1.5 s)`. `trial_instruction` is loaded but not used for this output.

ii.
```python
lmask = (left_lick_times >= gt) & (left_lick_times < gt + WIN_END)
rmask = (right_lick_times >= gt) & (right_lick_times < gt + WIN_END)
```

iii. The notes deliberately prefer actual lick behavior when available and reserve no-lick for trials with no response-period lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The earlier first lick maps to 0 (left) or 1 (right); no lick or an exact tie maps to 2. The per-trial code is repeated across 80 bins.

ii.
```python
if np.isfinite(lfirst) and (lfirst < rfirst): choice = 0
elif np.isfinite(rfirst) and (rfirst < lfirst): choice = 1
else: choice = 2
np.full(N_BINS, choice, dtype=np.int64)
```

iii. This implements the notes' behavioral definition and the requested left/right/no-lick categories.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials table's `outcome` column.

ii.
```python
outcome = _decode_arr(trials['outcome'][()])[:n_trials]
```

iii. The NWB column already contains the requested outcome labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped as ignore=0, miss=1, hit=2, with unknown values defaulting to ignore; the value is repeated over time.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = outcome_map.get(outcome[tr], 0)
```

iii. The mapping follows the requested category order; no specific justification is given for silently mapping unknown values to ignore.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials table's `early_lick` column.

ii.
```python
early_lick = _decode_arr(trials['early_lick'][()])[:n_trials]
```

iii. The trial table explicitly supplies this behavioral label.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` to 1, with unknown values defaulting to 0; the per-trial value is repeated over all bins.

ii.
```python
early_map = {'no early': 0, 'early': 1}
early_val = early_map.get(early_lick[tr], 0)
```

iii. The mapping matches the requested no/yes categories.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking/data` and timestamps, assuming columns are x, y, likelihood. Column 1 is y and column 2 is visibility confidence.

ii.
```python
tongue = np.asarray(beh_ts['Camera0_side_TongueTracking']['data'][()]).astype(float)
tongue_t = np.asarray(beh_ts['Camera0_side_TongueTracking']['timestamps'][()]).astype(float)
y_col, vis_col = choose_tongue_columns(tongue)
```

iii. The notes identify this series as the tongue source and infer the x/y/likelihood layout.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Visible raw frames (`likelihood > 0.5`) define session-wide 40th and 60th percentiles. For each neural bin center, the code selects the first camera frame at or after that center and classifies its y value; it does not average frames within a 50 ms bin.

ii.
```python
visible = np.isfinite(tongue_y) & np.isfinite(tongue_vis) & (tongue_vis > 0.5)
q40, q60 = np.quantile(tongue_y[visible], [0.4, 0.6])
idx = np.searchsorted(tongue_t, abs_centers, side='left')
ty = tongue_y[idx]
```

iii. The agent justified per-session thresholds over visible frames and class 3 for low-confidence tracking, but noted that not-visible frames dominated and should be revisited.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Visible values below q40 are 0, values from q40 through q60 inclusive are 1, values above q60 are 2, and nonfinite/low-confidence samples are 3.

ii.
```python
tongue_disc[vis_now & (ty < q40)] = 0
tongue_disc[vis_now & (ty >= q40) & (ty <= q60)] = 1
tongue_disc[vis_now & (ty > q60)] = 2
```

iii. This implements the requested per-session 40th/60th percentile categories and an explicit not-visible category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Each neural bin center is mapped to the first camera timestamp at or after it, clipped to the camera array bounds. Thus alignment is center sampling rather than aggregation over the identical neural-bin interval.

ii.
```python
idx = np.searchsorted(tongue_t, abs_centers, side='left')
idx = np.clip(idx, 0, len(tongue_t) - 1)
```

iii. The notes invoke the reference's cross-modal timestamp alignment and intended the tracking output to share the neural grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing event streams return `None`; absent tone matches are imputed at go−0.6 s; absent trial photostim values fall back to event streams; invisible tongue samples become class 3; unknown categorical labels default to category 0; missing `classification` would retain every unit; all-zero neural trials are discarded.

ii.
```python
if event_ts is None or len(event_ts) == 0:
    return go_times + default_offset
...
out_val = outcome_map.get(outcome[tr], 0)
if np.all(neural == 0): continue
```

iii. The notes describe fixes prompted by validation: finite tone fallbacks and filtering trials not covered by neural validity metadata. Several defaults prioritize a decoder-compatible finite output over surfacing malformed data.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant computation is the nested trial-by-unit spike histogramming, which repeatedly scans each unit's spike array. Reading large NWB spike/tracking arrays and pickling the roughly 12 GB result are also costly. The agent measured about 8 s/session before optimization.

ii.
```python
for tr in range(n_trials):
    for j, st in enumerate(good_spike_times):
        counts, _ = np.histogram(st - gt, bins=BIN_EDGES)
```

iii. The notes explicitly identify performance regression from per-trial spike extraction and say caching each unit's spike times once mitigated it, but do not provide a completed detailed efficiency analysis.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-by-unit histogram loop could vectorize the trial dimension by searching all trial edges once per unit. Trial event matching, first-lick searches, tongue sampling, and repeated construction of constant per-trial output rows could also be vectorized.

ii.
```python
for tr in range(n_trials):
    ...
    for j, st in enumerate(good_spike_times):
        counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. The notes mention caching but leave the “code inefficiencies” and “speedups” fields essentially blank; they do not document the available across-trial vectorization.

## 10-c. What processing does the code repeat multiple times?

i. It subtracts each go cue from an entire unit spike train and histograms it once per trial, repeatedly scans full lick-event arrays per trial, repeatedly loops over all stimulation intervals on fallback trials, and rebuilds mapping dictionaries and constant output vectors inside the trial loop.

ii.
```python
rel = st - gt
counts, _ = np.histogram(rel, bins=BIN_EDGES)
lmask = (left_lick_times >= gt) & (left_lick_times < gt + WIN_END)
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
```

iii. The only documented repetition fix is caching spike arrays per unit rather than extracting them anew for every trial.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `trial_instruction` but never uses it, computes `abs_edges` only for the fallback stimulation overlap test, records timing/file metadata that the decoder does not use, and optionally creates diagnostic plots. Most importantly, the per-trial `rel = st - gt` materializes all spikes although only spikes near the four-second window affect the histogram.

ii.
```python
trial_instruction = _decode_arr(trials['trial_instruction'][()])[:n_trials]
rel = st - gt
session_info = {'file': str(path), ..., 'load_time_sec': ...}
```

iii. The agent justified plots and metadata as validation/documentation aids. It did not document discarded computation in its efficiency section.
