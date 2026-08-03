# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by recursively scanning `data/` for `*.nwb` files with `Path('data').rglob('*.nwb')`, then opens each file directly with `h5py.File`. Within each file it reads trial, unit, and behavior groups manually from HDF5 paths such as `intervals/trials`, `units`, and `acquisition/BehavioralTimeSeries`.

ii.
```python
paths = sorted(Path('data').rglob('*.nwb'))
```

```python
with h5py.File(session_path, 'r') as f:
    trials = f['intervals/trials']
    units = f['units']
```

iii. The notes say the dataset is organized as subject-specific directories under `data/`, each containing NWB session files, and that the AI chose to use the NWB files as the source of truth. In the trajectory it explicitly states that it would "use NWB as source" and write a converter that loads those raw files directly.

## 1-b. How are the data split into subjects?

i. The AI assigns each session to a subject by reading `general/subject/subject_id` when available; if that read fails, it falls back to the parent directory name.

ii.
```python
def get_subject_id(f, path):
    try:
        sid = f['general/subject/subject_id'][()]
        return _decode(sid)
    except Exception:
        return path.parent.name
```

iii. The trajectory shows the AI identified subject-specific directories such as `data/sub-440958/...` and treated those NWB session files as belonging to one subject each. No stronger justification than the fallback logic appears in the notes.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. `extract_session()` converts one file, and `build_dataset()` appends one converted session per file.

ii.
```python
def extract_session(session_path, show_processing=False):
    ...
```

```python
for path in paths:
    ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
    ...
    neural.append(ntr)
    inputs.append(itr)
    outputs.append(otr)
```

iii. The notes state that data are organized as subject-specific directories, each containing NWB session files named like `sub-<id>_ses-<timestamp>_...nwb`, and the trajectory repeatedly describes the file boundary as the session boundary.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table under `intervals/trials` and iterates over `range(n_trials)`, where `n_trials` is the length of `trials['id']`. Each retained row becomes one trial in the output.

ii.
```python
trials = f['intervals/trials']
n_trials = len(trials['id'])
```

```python
for i in range(n_trials):
    if not valid_trials[i]:
        continue
    ...
    neural_trials.append(fr)
    input_trials.append(inp)
    output_trials.append(out)
```

iii. The notes describe trial metadata as stored under `intervals/trials`, and the trajectory says the converter will use the trial table directly for outputs and for derived event timing.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials with recognized categorical labels (`trial_instruction`, `outcome`, `early_lick`), whose derived go-cue window lies within the global spike-time min/max of the retained units, and whose final binned neural matrix is not all zero. Sessions with fewer than 2 retained trials or with zero good units are skipped.

ii.
```python
valid_trials = np.array([
    ti in ('left', 'right') and oc in ('ignore', 'miss', 'hit') and el in ('no early', 'early')
    for ti, oc, el in zip(trial_instruction, outcome, early_lick)
], dtype=bool)
```

```python
window_ok = (go_cue_times + WINDOW_START >= spike_t_min) & (go_cue_times + WINDOW_END <= spike_t_max)
valid_trials = valid_trials & window_ok
```

```python
if np.all(fr == 0):
    continue
```

```python
if len(ntr) < 2 or (len(ntr) > 0 and ntr[0].shape[0] == 0):
    print(f'Skipping {path.name}: insufficient kept trials or no good units')
    continue
```

iii. The notes and trajectory justify this as a response to "all-zero neural trials." The AI found that many trials had no concurrent spike recording when trial timestamps extended beyond the spike-time range, then patched the code to exclude windows outside spike coverage and to drop all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units/spike_times`, filtered by `unit_quality == 'good'`. Trial alignment uses a derived go-cue time computed from `intervals/trials/start_time` plus a fixed 1.85 s offset.

ii.
```python
trial_start = trials['start_time'][:].astype(float)
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
```

```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
spike_times_list = get_unit_spike_times(units)
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```

iii. The notes say the reference code operated on firing-rate arrays while the converter would load raw NWB data directly. The trajectory shows the AI decided to use `unit_quality == good` and derive go cue timing from the methods text because it believed explicit go-cue fields were absent from the trial table.

## 2-b. How is the `neural` data processed?

i. For each kept trial and kept unit, the AI bins spikes into 50 ms bins over `[-2.5, 1.5]` s relative to its derived go cue, then divides counts by `BIN_SIZE` to convert to firing rate in Hz. There is no smoothing or baseline normalization.

ii.
```python
edges_abs = align + BIN_EDGES
```

```python
fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. The notes explicitly say the script "bins spikes at 50 ms" and that the per-neuron histogram loop was a bottleneck. No additional processing beyond histogramming is justified in the notes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps units where `units['unit_quality'] == 'good'`. It does not use the NWB `classification` field from the QC classifier described in the reference materials. If no units pass this filter, the session is skipped.

ii.
```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
...
if len(ntr) < 2 or (len(ntr) > 0 and ntr[0].shape[0] == 0):
    print(f'Skipping {path.name}: insufficient kept trials or no good units')
    continue
```

iii. The notes say "Filter units using `units/unit_quality == good`," and later document a persistent mismatch versus the paper's good-unit count. The trajectory explicitly records that the AI knew this remained unresolved: "`unit_quality == good` yields 154,948 units, which does not match the 69,943 good units reported in the paper."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to a synthetic go cue defined as `trial_start + 1.85 s`, where 1.85 s came from the task timing described in `methods.txt`. It then constructs absolute bin edges around that derived time.

ii.
```python
GO_CUE_FROM_TRIAL_START = 1.85
...
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
```

```python
align = go_cue_times[i]
edges_abs = align + BIN_EDGES
```

iii. The notes say the trial table lacked explicit go-cue fields in the inspected sessions, so the AI chose to derive go cue timing from the documented task structure. The trajectory states this directly: "derive tone and go cue timing relative to `start_time` using methods-defined timing."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms bins, from -2.5 s to +1.5 s around the derived go cue, giving 80 bins per trial. Spikes are binned directly into that grid; no further temporal rebinning is applied.

ii.
```python
BIN_SIZE = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_BINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. The notes repeatedly say the decoder task required 50 ms bins aligned to the go cue from -2.5 s to +1.5 s, and that the converter followed that requirement.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives time from tone onset entirely from trial start time: it assumes tone onset is exactly at trial start (`TONE_ONSET_FROM_TRIAL_START = 0.0`) and never reads an explicit event stream for tone/sample onset.

ii.
```python
TONE_ONSET_FROM_TRIAL_START = 0.0
...
trial_start = trials['start_time'][:].astype(float)
tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START
```

iii. The notes say "derive tone onset from trial `start_time`" and the trajectory states this decision came from the methods text after the AI concluded that explicit tone fields were absent from the trial table and stimulus groups it inspected.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the AI computes the bin-center timestamps in absolute time and subtracts the assumed tone onset time, producing a continuous, time-varying signal in seconds.

ii.
```python
centers_rel = BIN_CENTERS.copy()
centers_abs = align + centers_rel
...
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. The notes justify this as using task timing relative to trial start. No additional transformations are described.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI places `time_from_tone` on the same 80-bin grid as neural data by using the same per-trial bin centers defined around the derived go cue.

ii.
```python
align = go_cue_times[i]
edges_abs = align + BIN_EDGES
centers_abs = align + centers_rel
...
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. The trajectory consistently describes all time-varying inputs and outputs as being rasterized onto the common 50 ms go-cue-centered grid used for neural activity.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from trial-table fields `photostim_onset` and `photostim_duration`, using `start_time` to convert those relative offsets into absolute session times.

ii.
```python
photostim_onset = [parse_optional_time(x) for x in trials['photostim_onset'][:]]
photostim_duration = [parse_optional_time(x) for x in trials['photostim_duration'][:]]
```

```python
p_start = trial_start[i] + p_on
p_stop = p_start + p_dur
```

iii. The notes and trajectory both identify `photostim_onset` and `photostim_duration` as the exact trial fields for this input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI parses optional trial-table values, treats `'N/A'` as no stimulation, and creates a binary time series whose bins are 1 when the bin center lies between the photostimulation start and stop times.

ii.
```python
def parse_optional_time(x):
    x = _decode(x)
    if isinstance(x, str) and x == 'N/A':
        return None
    try:
        return float(x)
    except Exception:
        return None
```

```python
photo = np.zeros(N_BINS, dtype=np.float32)
...
if p_on is not None and p_dur is not None:
    p_start = trial_start[i] + p_on
    p_stop = p_start + p_dur
    photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. The notes describe photostim as a time-varying binary input derived from the trial fields; no more elaborate justification is given.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation to neural data by comparing the same absolute bin centers used for neural alignment against absolute photostimulation start/stop times for each trial.

ii.
```python
centers_abs = align + centers_rel
...
photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. The trajectory repeatedly says all streams were placed on a common go-cue-centered 50 ms grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. In the final code, the AI derives choice only from `trial_instruction`, mapping instructed left/right directly to output classes. It does not combine `trial_instruction` with `outcome`, and it does not create a separate no-lick class for `ignore` trials.

ii.
```python
CHOICE_MAP = {b'left': 0, b'right': 1, 'left': 0, 'right': 1}
```

```python
trial_instruction = np.array([_decode(x) for x in trials['trial_instruction'][:]])
...
choice_val = CHOICE_MAP[trial_instruction[i]]
```

iii. The notes originally described "trial choice/outcome/early-lick fields" as outputs, and the trajectory focused more on structural validity than on deriving choice from outcome. No explicit justification for omitting the outcome-dependent derivation appears in the notes.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes `left` as 0 and `right` as 1, repeats that single per-trial value across all 80 bins, and exposes only two category names in `output_values`.

ii.
```python
choice_val = CHOICE_MAP[trial_instruction[i]]
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
...
out = np.stack([choice_ts, outcome_ts, early_ts, tongue_disc], axis=0)
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['lt_40pct', '40_to_60pct', 'gt_60pct'],
],
```

iii. No explicit justification was documented. The decision is only implicit in the code.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is read directly from the trial-table `outcome` column.

ii.
```python
outcome = np.array([_decode(x) for x in trials['outcome'][:]])
```

iii. The notes and mapping plan explicitly identify `outcome` as an exact trial field.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore`, `miss`, and `hit` to 0, 1, and 2, then repeats the per-trial category across all 80 bins.

ii.
```python
OUTCOME_MAP = {b'ignore': 0, b'miss': 1, b'hit': 2, 'ignore': 0, 'miss': 1, 'hit': 2}
```

```python
outcome_val = OUTCOME_MAP[outcome[i]]
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. The notes describe outcome as one of the exact trial-table variables to keep as a categorical decoder output.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is read directly from the trial-table `early_lick` column.

ii.
```python
early_lick = np.array([_decode(x) for x in trials['early_lick'][:]])
```

iii. The notes and mapping plan explicitly identify `early_lick` as the source variable.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early` to 0 and `early` to 1, then repeats the per-trial category across all 80 bins.

ii.
```python
EARLY_MAP = {b'no early': 0, b'early': 1, 'no early': 0, 'early': 1}
```

```python
early_val = EARLY_MAP[early_lick[i]]
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. The notes describe early lick as a per-trial categorical trial-table output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI searches `acquisition/BehavioralTimeSeries` for any group name containing `TongueTracking`, prefers a `Camera0...` series if present, then uses column 1 of its `data` array as tongue y and column 2 as confidence/likelihood when available.

ii.
```python
def choose_tongue_group(f):
    bts = f.get('acquisition/BehavioralTimeSeries')
    if bts is None:
        return None
    candidates = [k for k in bts.keys() if 'TongueTracking' in k]
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda x: (not x.startswith('Camera0'), x))
    return bts[candidates[0]]
```

```python
tongue_group = choose_tongue_group(f)
...
tongue_data = tongue_group['data'][:]
tongue_t = tongue_group['timestamps'][:].astype(float)
...
tongue_y = tongue_data[:, 1].astype(float)
if tongue_data.shape[1] >= 3:
    lik = tongue_data[:, 2].astype(float)
    tongue_y = tongue_y.copy()
    tongue_y[lik < 0.5] = np.nan
```

iii. The notes say the AI first had the wrong tongue path, then discovered `Camera0_side_TongueTracking` and used it. The trajectory also states the converter would use that path and column conventions.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI masks low-likelihood samples (`lik < 0.5`), computes 40th/60th percentiles over all valid raw tongue-y samples in the session, then for each neural bin finds the nearest valid tongue sample in time (within 100 ms) and discretizes that single sample. Missing/too-far bins default to class 1 rather than becoming an explicit missing class.

ii.
```python
if tongue_y is not None and np.isfinite(tongue_y).any():
    q40, q60 = np.nanpercentile(tongue_y, [40, 60])
else:
    q40, q60 = np.nan, np.nan
```

```python
valid_idx = np.flatnonzero(np.isfinite(tongue_y))
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
if len(valid_idx) > 0:
    vt = tongue_t[valid_idx]
    vy = tongue_y[valid_idx]
    idx = np.searchsorted(vt, centers_abs, side='left')
    idx = np.clip(idx, 0, len(vt) - 1)
    y = vy[idx]
    far = np.abs(vt[idx] - centers_abs) > 0.1
    y = y.copy()
    y[far] = np.nan
    finite = np.isfinite(y)
    tongue_disc[finite & (y < q40)] = 0
    tongue_disc[finite & (y > q60)] = 2
```

iii. The notes say tongue discretization was revised after sample validation because the output was initially too skewed, and that the AI switched to "nearest-valid sampling" with the likelihood threshold retained.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses session-wide thresholds `q40` and `q60` from raw valid tongue-y samples. Values below `q40` map to 0, above `q60` to 2, and everything else remains the default class 1. There is no explicit "not visible" class in the final output values.

ii.
```python
q40, q60 = np.nanpercentile(tongue_y, [40, 60])
```

```python
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
...
tongue_disc[finite & (y < q40)] = 0
tongue_disc[finite & (y > q60)] = 2
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['lt_40pct', '40_to_60pct', 'gt_60pct'],
],
```

iii. The notes only justify the 40th/60th percentile split by the decoder specification; they do not justify collapsing missing bins into the middle class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output to neural data by taking each neural bin center in absolute time and looking up the nearest valid tongue sample to that center, as long as it is within 100 ms.

ii.
```python
centers_abs = align + centers_rel
...
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
far = np.abs(vt[idx] - centers_abs) > 0.1
```

iii. The notes say this "nearest-valid sampling" was introduced after validation, because the earlier discretization produced an implausibly concentrated middle class.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses a mix of fallbacks and exclusions. `'N/A'` photostim entries become `None` and leave the photostim signal at 0. Missing `subject_id` falls back to the parent directory name. Missing/failed brain-region parsing falls back to `'unknown'`. Missing tongue samples are masked by likelihood, then effectively default to the middle tongue class if no valid nearby sample is found. Invalid categorical trial labels, no-good-unit sessions, windows outside spike coverage, and all-zero neural trials are dropped.

ii.
```python
if isinstance(x, str) and x == 'N/A':
    return None
```

```python
except Exception:
    return path.parent.name
```

```python
except Exception:
    brain_region_labels = ['unknown' for _ in good_spike_times]
```

```python
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
```

```python
valid_trials = valid_trials & window_ok
...
if np.all(fr == 0):
    continue
```

iii. The trajectory documents explicit reasoning for dropping no-neural-data trials and for adding the `'unknown'` region fallback when region metadata had not yet been recovered. No explicit defense was given for mapping missing tongue bins to the middle class.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified the per-neuron spike histogram loop as the main bottleneck and estimated the initial full conversion would take roughly 50+ minutes before optimizations. In code, the expensive work is repeatedly histogramming each good unit for each retained trial.

ii.
```python
fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. The notes explicitly say: "Per-neuron histogram loop is a bottleneck on full dataset; observed ~17.7 s/session over first 3 sessions, implying ~50+ min for full run if unoptimized."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorizable loop is the nested neural binning path: for every retained trial, the code loops over every kept unit and calls `np.histogram`. The per-trial tongue discretization loop also remains scalarized over trials.

ii.
```python
for i in range(n_trials):
    ...
    fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
    for n, st in enumerate(good_spike_times):
        ...
        counts, _ = np.histogram(st[mask], bins=edges_abs)
```

```python
for i in range(n_trials):
    ...
    if len(valid_idx) > 0:
        ...
```

iii. The notes explicitly mention the per-neuron histogram loop as the efficiency problem and say "Need further optimization before full conversion because projected runtime exceeds 15 minutes."

## 10-c. What processing does the code repeat multiple times?

i. The AI code repeats several computations per trial that could have been shared or vectorized: it rebuilds `centers_rel`/`centers_abs`, reruns unit-by-unit histogramming for each trial, and separately derives repeated per-trial constant output time series. It also computes `go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START` twice in `extract_session()`.

ii.
```python
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
...
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
```

```python
for i in range(n_trials):
    ...
    centers_rel = BIN_CENTERS.copy()
    centers_abs = align + centers_rel
```

```python
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. No explicit justification was documented for this repeated work. It is only implicit in the implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI code performs some work that is not used for downstream decoding decisions: it accepts a `show_processing` argument but never uses it, makes an unnecessary `BIN_CENTERS.copy()` per trial, and computes coarse session-level diagnostics such as `load_seconds`, `n_trials_raw`, `n_units_raw`, `q40`, and `q60` for `info` even though those are not used to build decoder features.

ii.
```python
def extract_session(session_path, show_processing=False):
```

```python
centers_rel = BIN_CENTERS.copy()
```

```python
info = {
    'subject': get_subject_id(f, session_path),
    'brain_region_labels': brain_region_labels,
    'n_trials_raw': n_trials,
    'n_trials_kept': len(neural_trials),
    'n_units_raw': len(unit_quality),
    'n_units_good': int(good_mask.sum()),
    'q40': float(q40) if np.isfinite(q40) else None,
    'q60': float(q60) if np.isfinite(q60) else None,
    'session_id': session_path.stem,
    'load_seconds': time.time() - t0,
}
```

iii. The notes say `--show-processing` was "reserved; plotting omitted in this implementation," which is the clearest explicit acknowledgement of discarded/unrealized processing. The rest is implicit from the code.
