# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every NWB file under `data/` using `Path('data').rglob('*.nwb')`, sorts the paths, and processes each file as one session with `h5py`. Within each file it reads trial data from `intervals/trials`, unit data from `units`, behavioral events from `acquisition/BehavioralEvents`, and tongue tracking from `acquisition/BehavioralTimeSeries`.

ii. 
```python
paths = sorted(Path('data').rglob('*.nwb'))

for path in paths:
    ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
```

```python
with h5py.File(session_path, 'r') as f:
    trials = f['intervals/trials']
    units = f['units']
```

iii. In `CONVERSION_NOTES.md`, the agent says it will "Use NWB as source of truth" and that the analysis code mostly operated on preprocessed arrays, so it chose to load raw NWB sessions directly.

## 1-b. How are the data split into subjects?

i. Subjects are defined per session by reading `general/subject/subject_id`; if that fails, the parent folder name is used. Unique subject IDs are collected into `subjects`, and each session gets an integer `subject_idx`.

ii. 
```python
def get_subject_id(f, path):
    try:
        sid = f['general/subject/subject_id'][()]
        return _decode(sid)
    except Exception:
        return path.parent.name
```

```python
sid = info['subject']
if sid not in subject_to_idx:
    subject_to_idx[sid] = len(subjects)
    subjects.append(sid)
subject_idx.append(subject_to_idx[sid])
```

iii. The trajectory shows the agent explicitly inspected subject IDs in NWB files and used them to build session-to-subject indexing, with the folder fallback as a robustness measure.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session order is simply the sorted file order from the recursive file search.

ii. 
```python
paths = sorted(Path('data').rglob('*.nwb'))
for path in paths:
    ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
```

```python
info = {
    ...
    'session_id': session_path.stem,
}
```

iii. The notes repeatedly describe the dataset as "subject-specific directories under `data/`, each containing NWB session files", so the agent made file = session its organizing assumption.

## 1-d. How are the data split into trials?

i. Trials come from rows of `intervals/trials`. The code reads trial-level arrays, loops over the row index `i`, and emits one converted trial per kept row.

ii. 
```python
trials = f['intervals/trials']
n_trials = len(trials['id'])
trial_start = trials['start_time'][:].astype(float)
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

iii. The trajectory shows the agent first identified `intervals/trials` as the central trial table and then based all later trial construction on those rows.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if required categorical labels are present, the inferred neural window falls within the min/max spike time range of good units, and the resulting neural matrix is not all zeros. The agent does not use an explicit reference-paper trial QC rule beyond these checks.

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
...
if np.all(fr == 0):
    continue
```

iii. The trajectory shows this filtering was added after the verifier found thousands of all-zero neural trials. The agent justified it as excluding trials outside concurrent neural coverage rather than as a paper-derived curation rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units/spike_times` and `units/spike_times_index`, after unit selection using `units/unit_quality`. Brain-region annotations are separately derived from `units/electrodes`, `units/electrodes_index`, and `general/extracellular_ephys/electrodes/location`.

ii. 
```python
def get_unit_spike_times(units_group):
    spikes = units_group['spike_times'][:]
    index = units_group['spike_times_index'][:]
```

```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
spike_times_list = get_unit_spike_times(units)
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```

iii. In the notes, the agent explicitly mapped "`units/spike_times` + unit metadata" to the `neural` field and later added electrode-derived region labels after first using `unknown`.

## 2-b. How is the `neural` data processed?

i. For each kept trial, the code builds 50 ms bins over a 4 s window around the inferred go cue, histograms each kept unit's spikes into those bins, and divides counts by `BIN_SIZE` to produce firing rates.

ii. 
```python
edges_abs = align + BIN_EDGES
fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. `CONVERSION_NOTES.md` says the reference analysis used trial-aligned firing-rate arrays (`fr`), and the agent adapted that idea to the decoder-required 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered with `unit_quality == 'good'`. The code ignores the separate `classification` field and ignores `is_good_trials` for unit/trial curation.

ii. 
```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```

iii. The trajectory and notes say the agent saw `unit_quality` values `good` and `multi` and treated `unit_quality == good` as the paper's good-unit QC. Even after noticing that this yielded 154,948 units instead of the paper's reported 69,943, it kept this choice.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns each trial to a derived go cue time computed as `trial_start + 1.85`, not to the explicit `go_start_times` event stream present in the NWB.

ii. 
```python
GO_CUE_FROM_TRIAL_START = 1.85
...
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
...
align = go_cue_times[i]
edges_abs = align + BIN_EDGES
```

iii. The notes say the agent believed go-cue times were absent from the trial table and therefore "derive go cue/tone from fixed task timing documented in methods". The trajectory shows this was a deliberate fallback, not an accidental omission.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins. No secondary rebinning is applied; spikes are directly histogrammed into the final bins.

ii. 
```python
BIN_SIZE = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_BINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE)
```

```python
'metadata': {
    'time_bin_size': 50.0,
}
```

iii. This came directly from the decoder-task instructions and is also stated in the README and notes.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `intervals/trials/start_time` plus a hard-coded constant `TONE_ONSET_FROM_TRIAL_START = 0.0`. No explicit tone event is used.

ii. 
```python
TONE_ONSET_FROM_TRIAL_START = 0.0
...
tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START
```

iii. The agent's notes say the trial table lacked explicit tone columns and that it would derive tone timing from the fixed task structure relative to `start_time`.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial/bin, the code subtracts the derived tone onset time from each bin center in absolute time, producing a continuous ramp in seconds.

ii. 
```python
centers_rel = BIN_CENTERS.copy()
centers_abs = align + centers_rel
...
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. The trajectory describes this as a simple derived time-varying decoder input once tone onset had been set from fixed trial timing.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled on the same 80 bin centers used for `neural`, using the derived absolute bin centers for each trial.

ii. 
```python
centers_abs = align + centers_rel
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
inp = np.stack([time_from_tone, photo], axis=0)
```

iii. The notes say all streams would be placed on a "common 50 ms grid" aligned to the go cue.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration`, interpreted relative to `trial_start`.

ii. 
```python
photostim_onset = [parse_optional_time(x) for x in trials['photostim_onset'][:]]
photostim_duration = [parse_optional_time(x) for x in trials['photostim_duration'][:]]
```

iii. The notes explicitly map "photostim timing from trial fields" to the `photostim_on` input. The trajectory shows the agent inspected these columns early and chose them instead of the explicit behavioral photostim event stream.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The code converts onset/duration to absolute start/stop times per trial and sets bins whose centers fall inside that interval to `1.0`, otherwise `0.0`.

ii. 
```python
photo = np.zeros(N_BINS, dtype=np.float32)
p_on = photostim_onset[i]
p_dur = photostim_duration[i]
if p_on is not None and p_dur is not None:
    p_start = trial_start[i] + p_on
    p_stop = p_start + p_dur
    photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. The trajectory describes this as rasterizing photostimulation onto the same 50 ms grid. The notes also mention the photostim protocol ending before go cue.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned by evaluating photostimulation on the same per-trial bin centers `centers_abs` used for `neural`.

ii. 
```python
centers_abs = align + centers_rel
photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. The justification is the same common-grid strategy used for all time-varying variables.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent derives `choice` from `intervals/trials/trial_instruction`, not from actual left/right lick events.

ii. 
```python
trial_instruction = np.array([_decode(x) for x in trials['trial_instruction'][:]])
...
choice_val = CHOICE_MAP[trial_instruction[i]]
```

iii. The trajectory shows the agent treated `trial_instruction` as the needed left/right label because it was readily available in the trial table and fit the decoder output shape requirement.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. `trial_instruction` is mapped to `0/1` via `CHOICE_MAP`, then repeated across all 80 time bins as a per-trial constant output row.

ii. 
```python
CHOICE_MAP = {b'left': 0, b'right': 1, 'left': 0, 'right': 1}
...
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
```

iii. The agent justified this in practice by treating per-trial categorical outputs as allowed to be repeated across time. The notes do not discuss the distinction between instructed side and actual lick choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `Outcome` is derived directly from `intervals/trials/outcome`.

ii. 
```python
outcome = np.array([_decode(x) for x in trials['outcome'][:]])
...
outcome_val = OUTCOME_MAP[outcome[i]]
```

iii. The field was identified in the early NWB inspection and directly used because it already matched the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings `ignore`, `miss`, and `hit` are mapped to `0/1/2` and repeated across all 80 time bins.

ii. 
```python
OUTCOME_MAP = {b'ignore': 0, b'miss': 1, b'hit': 2, 'ignore': 0, 'miss': 1, 'hit': 2}
...
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. The agent used the decoder spec's categorical ordering directly and treated outcome as a trial-level label.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. `Early lick` is derived directly from `intervals/trials/early_lick`.

ii. 
```python
early_lick = np.array([_decode(x) for x in trials['early_lick'][:]])
...
early_val = EARLY_MAP[early_lick[i]]
```

iii. The field was confirmed during NWB inspection and then used directly in the planned variable mapping.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Strings `no early` and `early` are mapped to `0/1` and repeated across all 80 bins.

ii. 
```python
EARLY_MAP = {b'no early': 0, b'early': 1, 'no early': 0, 'early': 1}
...
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. The agent relied on the trial table labels and on the decoder's allowance for per-trial categorical outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position comes from `acquisition/BehavioralTimeSeries/*TongueTracking*/data` and its `timestamps`, with the code preferring a `Camera0` group if present.

ii. 
```python
def choose_tongue_group(f):
    bts = f.get('acquisition/BehavioralTimeSeries')
    candidates = [k for k in bts.keys() if 'TongueTracking' in k]
    candidates = sorted(candidates, key=lambda x: (not x.startswith('Camera0'), x))
    return bts[candidates[0]]
```

```python
tongue_data = tongue_group['data'][:]
tongue_t = tongue_group['timestamps'][:].astype(float)
```

iii. The trajectory shows the agent first found the correct path by exploring the behavioral time-series groups, after initially using a wrong tongue path in the notes.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The code assumes tongue-tracking rows are `[x, y, likelihood]`, takes column 1 as `y`, masks rows with likelihood `< 0.5`, and for each neural bin picks the nearest valid tongue sample if it is within 0.1 s; otherwise that bin remains at the default middle category.

ii. 
```python
if tongue_data.ndim == 2 and tongue_data.shape[1] >= 2:
    tongue_y = tongue_data[:, 1].astype(float)
    if tongue_data.shape[1] >= 3:
        lik = tongue_data[:, 2].astype(float)
        tongue_y = tongue_y.copy()
        tongue_y[lik < 0.5] = np.nan
```

```python
valid_idx = np.flatnonzero(np.isfinite(tongue_y))
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
...
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
far = np.abs(vt[idx] - centers_abs) > 0.1
y = y.copy()
y[far] = np.nan
```

iii. The notes say the agent changed tongue processing after seeing extreme class imbalance, settling on a lower likelihood threshold and nearest-valid sampling.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide 40th and 60th percentiles are computed from all finite `tongue_y` values. Per bin, values below `q40` are category `0`, above `q60` are `2`, and everything else is `1`.

ii. 
```python
if tongue_y is not None and np.isfinite(tongue_y).any():
    q40, q60 = np.nanpercentile(tongue_y, [40, 60])
```

```python
tongue_disc[finite & (y < q40)] = 0
tongue_disc[finite & (y > q60)] = 2
```

iii. This directly mirrors the decoder-task discretization rule in the instructions, which the notes explicitly cite.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue samples are aligned to the same absolute bin centers `centers_abs` used for neural data by nearest-neighbor sampling in time.

ii. 
```python
centers_abs = align + centers_rel
...
idx = np.searchsorted(vt, centers_abs, side='left')
...
y = vy[idx]
```

iii. The agent's justification was to place all time-varying signals on the same 50 ms go-cue-centered grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code has several fallbacks: bytes are decoded; `'N/A'` times become `None`; missing subject IDs fall back to folder names; missing/failed brain-region parsing falls back to `'unknown'`; low-confidence or too-distant tongue samples become missing and then effectively default to the middle tongue category; invalid trial labels are excluded; all-zero neural trials are dropped.

ii. 
```python
def parse_optional_time(x):
    x = _decode(x)
    if isinstance(x, str) and x == 'N/A':
        return None
```

```python
except Exception:
    brain_region_labels = ['unknown' for _ in good_spike_times]
```

```python
else:
    tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
```

iii. The trajectory shows these were pragmatic robustness decisions made during debugging, especially after verification failures and after discovering missing or low-confidence tongue data.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are reading each large NWB file, unpacking ragged spike times, and the nested per-trial, per-neuron histogramming loop. The full conversion log shows many sessions taking tens of seconds, with large sessions taking over a minute.

ii. 
```python
spike_times_list = get_unit_spike_times(units)
...
for i in range(n_trials):
    ...
    for n, st in enumerate(good_spike_times):
        ...
        counts, _ = np.histogram(st[mask], bins=edges_abs)
```

iii. The notes explicitly identify the per-neuron histogram loop as the bottleneck and estimate that early versions would take 50+ minutes over the full dataset.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron histogram loop inside each trial is the clearest vectorization target. Additional targets are the spike-time ragged-array unpacking loop, the per-unit brain-region reconstruction loop, and repeated per-trial nearest-neighbor tongue indexing.

ii. 
```python
for stop in index:
    out.append(spikes[start:stop])
    start = stop
```

```python
for n, st in enumerate(good_spike_times):
    ...
    counts, _ = np.histogram(st[mask], bins=edges_abs)
```

iii. `CONVERSION_NOTES.md` explicitly calls out the per-neuron histogram loop as the main inefficiency; the other loops are visibly scalar in the implementation.

## 10-c. What processing does the code repeat multiple times?

i. The code recomputes go-cue times twice inside `extract_session`, repeatedly slices/masks spike trains for each trial-neuron pair, repeatedly constructs constant output vectors for per-trial labels, and repeatedly rebuilds nearest-neighbor tongue alignment from scratch per trial.

ii. 
```python
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
...
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
```

```python
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. These are direct properties of the code. The trajectory also mentions repeated processing during iterative re-validation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code materializes 80-bin repeated copies of trial-constant outputs (`choice`, `outcome`, `early_lick`) even though they carry no within-trial temporal information. It also computes full neural matrices for trials that are later discarded because they are all zero, and it stores session metadata such as `q40`/`q60` mainly for bookkeeping rather than decoder use.

ii. 
```python
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

```python
out = np.stack([choice_ts, outcome_ts, early_ts, tongue_disc], axis=0)

if np.all(fr == 0):
    continue
```

iii. These are not described as intentional optimizations anywhere in the notes; they are just consequences of the chosen representation and filtering order.
