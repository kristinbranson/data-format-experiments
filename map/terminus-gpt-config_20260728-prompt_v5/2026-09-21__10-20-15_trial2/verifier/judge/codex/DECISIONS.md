# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by recursively globbing `/app/data` for `*.nwb`, sorts the paths, and processes each file with `pynwb.NWBHDF5IO`. Within each file it reads the NWB trials table, units table, behavioral events, and behavioral time series directly.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
```

```python
io = NWBHDF5IO(str(path), 'r', load_namespaces=True)
nwb = io.read()
trial_starts, trial_stops, td = infer_trial_intervals(nwb)
go_times = assign_events_to_trials(get_event_times(nwb, 'go_start_times'), trial_starts, trial_stops)
```

iii. The trajectory says the agent wanted a single script that "loads NWB files" and uses "robust field discovery" because exact names might vary. `CONVERSION_NOTES.md` also treats the NWB files under `/app/data/sub-<subject>/` as the authoritative raw dataset.

## 1-b. How are the data split into subjects?

i. The AI uses `nwb.subject.subject_id` as the subject id when present, otherwise it falls back to the parent directory name. Subjects are accumulated in first-seen order during the main loop, and `subject_idx` stores the index assigned at first encounter.

ii.
```python
def get_subject_name(nwb, path):
    sid = getattr(getattr(nwb, 'subject', None), 'subject_id', None)
    return sid if sid is not None else path.parent.name
```

```python
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
data['subject_idx'].append(subject_to_idx[subj])
```

iii. The explicit justification is minimal. The trajectory shows the agent wanted a tolerant loader that would keep working if some NWB metadata were missing, which explains the fallback to the folder name.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session metadata is recorded from the filename stem, not from `nwb.identifier`.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
for path in files:
    sess = process_session(path, show_processing=args.show_processing)
```

```python
info = {
    'session_id': path.stem,
    'subject': subj,
    'n_trials': len(neural_trials),
    'n_units_kept': len(kept_unit_idx),
    ...
}
```

iii. `CONVERSION_NOTES.md` states that each session is one `*_behavior+ecephys+ogen.nwb` file, so the file boundary is being used as the session boundary.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table for trial start and stop times, then assigns event streams to trials by asking which events fall inside each trial interval. It keeps trials whose assigned go cue is finite.

ii.
```python
def infer_trial_intervals(nwb):
    td = get_trial_table_dict(nwb)
    cols = list(td.keys())
    start_col = find_first(cols, ['start_time', 'start'])
    stop_col = find_first(cols, ['stop_time', 'stop'])
    return np.asarray(td[start_col], float), np.asarray(td[stop_col], float), td
```

```python
def assign_events_to_trials(event_times, trial_starts, trial_stops):
    out = np.full(len(trial_starts), np.nan, dtype=float)
    for i, (a, b) in enumerate(zip(trial_starts, trial_stops)):
        m = (event_times >= a) & (event_times <= b)
        if np.any(m):
            out[i] = event_times[m][0]
    return out
```

iii. The trajectory describes this as part of a "robust" scheme that infers mappings from the raw NWB structure rather than relying on any one exact convention.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not implement the reference trial QC. It only drops trials with no assigned go cue. At the session level it drops sessions with fewer than 2 remaining trials or zero kept units. It does not use `obs_intervals` or `free_water` to remove trials with no spike data.

ii.
```python
for i, go in enumerate(go_times):
    if not np.isfinite(go):
        continue
    neural_trials.append(bin_spikes_for_trial(spikes, go).astype(np.float32))
    ...
    kept_trial_idx.append(i)
```

```python
if len(neural_trials) < 2 or len(reg_kept) == 0:
    print('SKIP insufficient trials or units', path)
    continue
```

iii. `CONVERSION_NOTES.md` says the agent planned to "retain trial-level outputs even if papers excluded some trial types." There is no documented justification for omitting the `obs_intervals` and `free_water` filtering used by the reference solution.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']` for the units selected by the AI's quality mask. The go-cue event stream provides the alignment times used to define each trial window.

ii.
```python
kept_unit_idx, spikes = spike_times_list(nwb, unit_mask)
go_times = assign_events_to_trials(get_event_times(nwb, 'go_start_times'), trial_starts, trial_stops)
```

```python
for i in idx:
    spikes.append(np.asarray(nwb.units['spike_times'][i], dtype=float))
```

iii. `CONVERSION_NOTES.md` planned to use "NWB `units` spike times" as the neural source and to align them to the go cue in 50 ms bins.

## 2-b. How is the `neural` data processed?

i. The AI converts spike times to firing rates by histogramming each selected unit into 50 ms bins spanning -2.5 s to +1.5 s around the go cue, then dividing counts by the bin width.

ii.
```python
BIN = 0.05
T_START = -2.5
T_END = 1.5
BIN_EDGES = np.arange(T_START, T_END + BIN + 1e-9, BIN)
BIN_CENTERS = BIN_EDGES[:-1] + BIN / 2
```

```python
def bin_spikes_for_trial(spike_times, align_time):
    arr = np.zeros((len(spike_times), len(BIN_CENTERS)), dtype=np.float32)
    rel_edges = align_time + BIN_EDGES
    for i, st in enumerate(spike_times):
        arr[i] = np.histogram(st, bins=rel_edges)[0].astype(np.float32) / BIN
    return arr
```

iii. This follows the mapping plan written in `CONVERSION_NOTES.md`: "Bin spikes in 50 ms bins from -2.5 s to +1.5 s around Go cue; convert to firing rates or spike counts consistently across sessions."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI tries to infer a "good units" column heuristically, but in practice it picks the first units-table column whose name contains `good`, `quality`, `label`, or `unit_quality`. On the real files this resolves to `is_good_trials`, not `classification`. The result is used as the unit mask without checking dimensionality.

ii.
```python
def get_unit_mask_and_regions(nwb):
    cols = list(nwb.units.colnames)
    good_col = find_first(cols, ['good', 'quality', 'label', 'unit_quality'])
    ...
    if good_col is not None:
        vals = np.asarray(nwb.units[good_col][:])
        if vals.dtype.kind in 'OUS':
            sval = np.array([str(v).lower() for v in vals])
            mask = np.array([('good' in v) or (v == '1') or (v == 'true') for v in sval], dtype=bool)
        else:
            mask = vals.astype(bool)
```

```python
def spike_times_list(nwb, unit_mask):
    idx = np.where(unit_mask)[0]
    spikes = []
    for i in idx:
        spikes.append(np.asarray(nwb.units['spike_times'][i], dtype=float))
    return idx, spikes
```

iii. The trajectory explicitly says the agent would use "robust field discovery" because it had not pinned down the exact NWB QC field. `CONVERSION_NOTES.md` says "Use units labeled as `good` ... when that information is available," but it does not identify the correct field before coding.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to its assigned go cue. The bin edges are made by adding the common relative grid to the trial's go time, and spikes are histogrammed in that absolute window.

ii.
```python
rel_edges = align_time + BIN_EDGES
for i, st in enumerate(spike_times):
    arr[i] = np.histogram(st, bins=rel_edges)[0].astype(np.float32) / BIN
```

iii. `CONVERSION_NOTES.md` explicitly names go-cue onset as the required alignment event, consistent with the task instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins over a 4 s window, giving 80 timepoints. There is no second-stage rebinning.

ii.
```python
BIN = 0.05
T_START = -2.5
T_END = 1.5
BIN_EDGES = np.arange(T_START, T_END + BIN + 1e-9, BIN)
BIN_CENTERS = BIN_EDGES[:-1] + BIN / 2
```

```python
'time_bin_size': 50.0,
'n_timepoints': len(BIN_CENTERS),
```

iii. This comes directly from the task instructions and the Step 5 mapping plan in `CONVERSION_NOTES.md`.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` and the trial go cue. The AI assigns a sample-start event to each trial by taking the first sample event inside that trial interval.

ii.
```python
sample_times = assign_events_to_trials(get_event_times(nwb, 'sample_start_times'), trial_starts, trial_stops)
```

```python
def build_inputs(go_time, sample_start, photostim_intervals):
    tone_time = sample_start
    time_from_tone = (go_time + BIN_CENTERS) - tone_time
```

iii. `CONVERSION_NOTES.md` planned to use "trial event timestamps from NWB" and treat the tone onset as the sample start. No separate justification is given for taking the first sample event rather than the last one before go.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes a continuous value at each bin center by subtracting the inferred tone time from the absolute bin center time. If the sample start is missing, it falls back to `go_time - 1.5`.

ii.
```python
time_from_tone = (go_time + BIN_CENTERS) - tone_time
return np.vstack([time_from_tone.astype(np.float32), phot])
```

```python
input_trials.append(build_inputs(
    go,
    sample_times[i] if np.isfinite(sample_times[i]) else go - 1.5,
    phot_int
).astype(np.float32))
```

iii. The justification is mostly implicit: the mapping notes say this input should be a time-varying quantity aligned to the go cue. The fallback to `go - 1.5` is not justified in the notes.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 80 go-aligned bin centers used for neural activity.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN / 2
time_from_tone = (go_time + BIN_CENTERS) - tone_time
```

iii. The agent's Step 5 mapping notes explicitly planned to align this input to the go-cue-centered neural window.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the `BehavioralEvents` event streams `photostim_start_times` and `photostim_stop_times`, not from the per-trial `photostim_onset` and `photostim_duration` fields in the trials table.

ii.
```python
def get_photostim_intervals(nwb):
    keys = list(nwb.acquisition['BehavioralEvents'].time_series.keys())
    if 'photostim_start_times' not in keys or 'photostim_stop_times' not in keys:
        return np.empty((0, 2), dtype=float)
    s = get_event_times(nwb, 'photostim_start_times')
    e = get_event_times(nwb, 'photostim_stop_times')
    n = min(len(s), len(e))
    return np.c_[s[:n], e[:n]] if n else np.empty((0, 2), dtype=float)
```

iii. `CONVERSION_NOTES.md` explicitly planned to use `BehavioralEvents photostim_start/stop_times` for this variable.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI makes a binary time series. For each trial it compares the absolute bin centers to every global photostim interval in the session and marks bins as 1 when they fall within any interval.

ii.
```python
phot = np.zeros(len(BIN_CENTERS), dtype=np.float32)
abs_centers = go_time + BIN_CENTERS
for a, b in photostim_intervals:
    phot[(abs_centers >= a) & (abs_centers <= b)] = 1.0
```

iii. The mapping notes say this target should be "binary time-varying" and should use the behavioral event streams.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Alignment is done by putting both on the same absolute time axis: neural bins are centered at `go_time + BIN_CENTERS`, and photostim intervals are compared directly to those absolute bin centers.

ii.
```python
abs_centers = go_time + BIN_CENTERS
for a, b in photostim_intervals:
    phot[(abs_centers >= a) & (abs_centers <= b)] = 1.0
```

iii. The justification is implicit in the choice to use the go-centered bin centers for all time-varying streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI looks for a pre-existing trial column named like `choice`, `lick_direction`, or `response_side`. If none is found, it does not derive choice from `trial_instruction` and `outcome`; instead it falls back to the constant class `2` (`no lick`) for all trials.

ii.
```python
def infer_trial_labels(td):
    cols = list(td.keys())
    choice_col = find_first(cols, ['choice', 'lick_direction', 'response_side'])
    outcome_col = find_first(cols, ['outcome', 'trial_outcome', 'result', 'correctness'])
    early_col = find_first(cols, ['early', 'early_lick'])
    return choice_col, outcome_col, early_col
```

```python
choice = map_choice(td[choice_col][i]) if choice_col is not None else 2
```

iii. `CONVERSION_NOTES.md` says "Determine exact NWB trial column encoding," which shows the agent knew this mapping was unresolved. The trajectory says it would rely on "robust field discovery" pending validation.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. When a choice-like column exists, the AI maps strings containing `left` to 0, `right` to 1, and strings containing `no`, `ignore`, or `miss` to 2. The resulting per-trial label is then repeated across all time bins. If no choice-like column exists, it uses 2 everywhere.

ii.
```python
def map_choice(v):
    s = str(v).lower()
    if 'left' in s:
        return 0
    if 'right' in s:
        return 1
    if 'no' in s or 'ignore' in s or 'miss' in s or s in ('nan', ''):
        return 2
    ...
    return 2
```

```python
out = np.vstack([
    np.full(len(BIN_CENTERS), choice, dtype=np.uint8),
    np.full(len(BIN_CENTERS), outcome, dtype=np.uint8),
    np.full(len(BIN_CENTERS), early, dtype=np.uint8),
    tongue_disc[j].astype(np.uint8),
])
```

iii. No strong justification is documented beyond the generic "field mappings may require refinement after sample validation" note in the metadata.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from whichever trial column first matches `outcome`, `trial_outcome`, `result`, or `correctness`. On the real files this resolves to the `outcome` column.

ii.
```python
outcome_col = find_first(cols, ['outcome', 'trial_outcome', 'result', 'correctness'])
...
outcome = map_outcome(td[outcome_col][i]) if outcome_col is not None else 0
```

iii. `CONVERSION_NOTES.md` planned to read outcome from "trial metadata and/or correctness columns," so this heuristic is consistent with the agent's stated plan.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps strings containing `ignore` to 0, `miss` to 1, and `hit` or `correct` to 2, with some extra integer fallbacks. The final per-trial label is repeated across all bins.

ii.
```python
def map_outcome(v):
    s = str(v).lower()
    if 'ignore' in s:
        return 0
    if 'miss' in s:
        return 1
    if 'hit' in s or 'correct' in s or s == '1':
        return 2
    ...
```

```python
np.full(len(BIN_CENTERS), outcome, dtype=np.uint8)
```

iii. The notes say the target should be the categorical label `ignore / miss / hit`. No additional reasoning is given because the raw NWB table already has an outcome field.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from a trial column matched by the names `early` or `early_lick`. On the real files this resolves to `early_lick`.

ii.
```python
early_col = find_first(cols, ['early', 'early_lick'])
...
early = map_early(td[early_col][i]) if early_col is not None else 0
```

iii. `CONVERSION_NOTES.md` planned to take this directly from the NWB trial metadata.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI uses a substring heuristic: any string containing `true`, `yes`, `1`, or `early` maps to 1; otherwise it tries an integer cast and falls back to 0. This means the real label `no early` is incorrectly mapped to 1, because it still contains the substring `early`. The per-trial result is repeated across bins.

ii.
```python
def map_early(v):
    s = str(v).lower()
    if 'true' in s or 'yes' in s or s == '1' or 'early' in s:
        return 1
    try:
        return int(bool(int(v)))
    except Exception:
        return 0
```

iii. No explicit justification is documented. The trajectory only indicates that the agent planned to use heuristic field/value discovery before later validation.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`. The AI assumes the second column is tongue y and the third column is a visibility/likelihood score when present.

ii.
```python
def tongue_series(nwb):
    ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
    data = np.asarray(ts.data[:])
    t = np.asarray(ts.timestamps[:], dtype=float)
    return t, data
```

```python
y = data[:, 1] if data.ndim > 1 and data.shape[1] > 1 else data[:, 0]
vis = np.ones_like(y, dtype=bool)
if data.ndim > 1 and data.shape[1] > 2:
    vis = np.asarray(data[:, 2] > 0.5)
```

iii. `CONVERSION_NOTES.md` says tongue y should come from `Camera0_side_TongueTracking` and that the agent needed to identify the coordinate/confidence columns. The code hard-codes that assumption.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI interpolates tongue y and a visibility mask onto the 80 go-aligned bin centers for each trial. It then pools all finite interpolated values across all trials in a session, computes the 40th and 60th percentiles of those pooled values, and discretizes each trial's interpolated timepoints against those thresholds. Missing values become class 3.

ii.
```python
sample_t = go_time + BIN_CENTERS
y_interp = np.interp(sample_t, t, y, left=np.nan, right=np.nan)
vis_interp = np.interp(sample_t, t, vis.astype(float), left=0, right=0) > 0.5
y_interp[~vis_interp] = np.nan
```

```python
all_y = np.concatenate([y[np.isfinite(y)] for y in trial_y_list if np.any(np.isfinite(y))]) if trial_y_list else np.array([])
q40, q60 = np.percentile(all_y, [40, 60])
```

iii. `CONVERSION_NOTES.md` only states the high-level requirement: discretize tongue y by per-session percentiles and use class 3 when the tongue is not visible. It does not justify the interpolation-based implementation.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses session-specific 40th and 60th percentiles of pooled finite tongue-y samples. Values below `q40` map to 0, values from `q40` through `q60` map to 1, values above `q60` map to 2, and non-finite values map to 3.

ii.
```python
q40, q60 = np.percentile(all_y, [40, 60])
...
d = np.full(len(y), 3, dtype=np.int64)
m = np.isfinite(y)
d[m & (y < q40)] = 0
d[m & (y >= q40) & (y <= q60)] = 1
d[m & (y > q60)] = 2
```

iii. The 40/60 thresholds and the `not_visible` class come directly from the task instructions and Step 5 notes. There is no further written justification.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue y by interpolating it to the same absolute bin centers (`go_time + BIN_CENTERS`) used for neural activity.

ii.
```python
sample_t = go_time + BIN_CENTERS
y_interp = np.interp(sample_t, t, y, left=np.nan, right=np.nan)
vis_interp = np.interp(sample_t, t, vis.astype(float), left=0, right=0) > 0.5
```

iii. The justification is implicit: the code uses the same go-aligned bin centers for every time-varying stream.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly uses silent fallbacks. Missing subject ids fall back to the folder name. Missing go cues cause the trial to be dropped. Missing sample times fall back to `go_time - 1.5`. Missing photostim streams give an all-zero signal. Missing region fields become `unknown`. Missing tongue visibility becomes NaN and then class 3 after discretization.

ii.
```python
return sid if sid is not None else path.parent.name
```

```python
if not np.isfinite(go):
    continue
...
sample_times[i] if np.isfinite(sample_times[i]) else go - 1.5
```

```python
if 'photostim_start_times' not in keys or 'photostim_stop_times' not in keys:
    return np.empty((0, 2), dtype=float)
...
regions = np.array(['unknown'] * n_units, dtype=object)
```

iii. The trajectory repeatedly frames these choices as "robust field discovery" and tolerant handling of schema variation. There is little explicit discussion of whether these fallbacks are scientifically appropriate.

## 10-a. What are the most time-consuming steps of the code?

i. The likely dominant costs are per-file NWB I/O, reading every kept unit's spike times one unit at a time, per-trial/per-unit spike histogramming, and reloading/interpolating tongue tracking for every trial.

ii.
```python
for i in idx:
    spikes.append(np.asarray(nwb.units['spike_times'][i], dtype=float))
```

```python
for i, go in enumerate(go_times):
    ...
    neural_trials.append(bin_spikes_for_trial(spikes, go).astype(np.float32))
    ...
    tongue_y_trials.append(interpolate_tongue_y(nwb, go))
```

iii. The AI did not complete the runtime-analysis parts of `CONVERSION_NOTES.md`; the file still shows placeholders under Step 6 and Step 7. The assessment here is therefore inferred from the code structure rather than from a documented self-analysis.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code leaves several expensive Python loops in place: iterating over units to collect spike trains, iterating over trials, iterating over units again inside every trial for histogramming, iterating over all photostim intervals per trial, and iterating over trials to discretize tongue signals.

ii.
```python
for i in idx:
    spikes.append(np.asarray(nwb.units['spike_times'][i], dtype=float))
```

```python
for i, go in enumerate(go_times):
    ...
    neural_trials.append(bin_spikes_for_trial(spikes, go).astype(np.float32))
```

```python
for i, st in enumerate(spike_times):
    arr[i] = np.histogram(st, bins=rel_edges)[0].astype(np.float32) / BIN
```

iii. The AI did not document any considered vectorizations. The trajectory instead emphasizes getting a generic loader working before later refinement.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations: it scans global photostim intervals separately for every trial, rereads the full tongue time series inside `interpolate_tongue_y` for every trial, and performs unit-by-unit histogramming separately for each trial instead of binning all trials for a unit in one pass.

ii.
```python
for a, b in photostim_intervals:
    phot[(abs_centers >= a) & (abs_centers <= b)] = 1.0
```

```python
def interpolate_tongue_y(nwb, go_time):
    t, data = tongue_series(nwb)
    ...
```

```python
for i, go in enumerate(go_times):
    neural_trials.append(bin_spikes_for_trial(spikes, go).astype(np.float32))
```

iii. There is no explicit justification. The trajectory suggests the agent prioritized schema-tolerant implementation first and expected to refine after validation, but that validation step never happened.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes optional plotting for `--show-processing`, carries debugging metadata such as `good_col`, `region_col`, `choice_col`, `outcome_col`, `early_col`, and `tongue_quantiles` into `session_info`, and imports `Counter` without using it. The largest scientifically unnecessary work is the repeated interpolation and plotting/debugging scaffolding, not any reference-matching analysis step.

ii.
```python
from collections import Counter
import matplotlib.pyplot as plt
```

```python
if show_processing and len(neural_trials) > 0:
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    ...
    fig.savefig(f"/app/processing_{path.stem}.png", dpi=150)
```

```python
info = {
    ...
    'good_col': good_col,
    'region_col': region_col,
    'choice_col': choice_col,
    'outcome_col': outcome_col,
    'early_col': early_col,
    'tongue_quantiles': tongue_q,
}
```

iii. The trajectory shows the agent wanted verbose sanity checks and visual inspection (`--show-processing`) while debugging, which explains the extra plotting/debug metadata.
