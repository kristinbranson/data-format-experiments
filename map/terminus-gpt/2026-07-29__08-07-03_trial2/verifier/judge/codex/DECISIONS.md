# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all session files by recursively globbing `data/**/*.nwb`, then opens each NWB file with `h5py.File(...)` and processes it with `extract_session(...)`. It does not load any of the reference preprocessed pickle outputs; it treats raw NWB as the source of truth.

ii.
```python
def main():
    ...
    paths = sorted(Path('data').rglob('*.nwb'))
    if args.sample:
        paths = paths[:2]
    data = build_dataset(paths, show_processing=args.show_processing)

def build_dataset(paths, show_processing=False):
    ...
    for path in paths:
        ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
```

iii. `CONVERSION_NOTES.md` Step 5 says “Use NWB as source of truth,” and Step 2 describes the dataset as subject-specific directories containing NWB session files.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `general/subject/subject_id` in each NWB file, with fallback to the parent directory name. Unique subject IDs are accumulated into `subjects`, and each kept session gets a `subject_idx`.

ii.
```python
def get_subject_id(f, path):
    try:
        sid = f['general/subject/subject_id'][()]
        return _decode(sid)
    except Exception:
        return path.parent.name

...
sid = info['subject']
if sid not in subject_to_idx:
    subject_to_idx[sid] = len(subjects)
    subjects.append(sid)
subject_idx.append(subject_to_idx[sid])
```

iii. Step 2 of `CONVERSION_NOTES.md` says the data are organized as subject-specific directories under `data/`.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. `extract_session` processes one file at a time, and each accepted file contributes one top-level entry to `neural`, `input`, and `output`.

ii.
```python
for path in paths:
    ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
    if len(ntr) < 2 or (len(ntr) > 0 and ntr[0].shape[0] == 0):
        print(f'Skipping {path.name}: insufficient kept trials or no good units')
        continue
    neural.append(ntr)
    inputs.append(itr)
    outputs.append(otr)
```

iii. Step 2 notes that each subject directory contains NWB session files named like `sub-<id>_ses-<timestamp>_behavior+ecephys+ogen.nwb`.

## 1-d. How are the data split into trials?

i. Trials are the rows of `intervals/trials`. The code iterates `for i in range(n_trials)` and, for each kept trial, builds one neural matrix, one input matrix, and one output matrix.

ii.
```python
trials = f['intervals/trials']
n_trials = len(trials['id'])
...
for i in range(n_trials):
    if not valid_trials[i]:
        continue
    ...
    neural_trials.append(fr)
    input_trials.append(inp)
    output_trials.append(out)
```

iii. This follows the NWB trial table directly. The notes repeatedly describe trial metadata as living under `intervals/trials`.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only trials whose categorical labels are present (`trial_instruction`, `outcome`, `early_lick`), whose derived go-cue window lies inside the min/max spike-time support of the kept units, and whose binned neural activity is not all zeros. It does not apply the reference code’s regular-trial mask (`no early lick`, `no auto water`, `no free water`, `correctness != -1`, `no stimulation`) or the paper’s session-level behavioral inclusion criteria.

ii.
```python
valid_trials = np.array([
    ti in ('left', 'right') and oc in ('ignore', 'miss', 'hit') and el in ('no early', 'early')
    for ti, oc, el in zip(trial_instruction, outcome, early_lick)
], dtype=bool)

window_ok = (go_cue_times + WINDOW_START >= spike_t_min) & (go_cue_times + WINDOW_END <= spike_t_max)
valid_trials = valid_trials & window_ok

...
if np.all(fr == 0):
    continue
```

iii. Step 5 says “Restrict to trials with complete required variables.” Step 10 explains that later trials were dropped when the derived neural window fell outside spike coverage. The reference code instead defines a stricter regular-trial mask.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from `units/spike_times` and `units/spike_times_index`, after filtering units by `units/unit_quality == 'good'`. Brain-region metadata are derived separately from `units/electrodes`, `units/electrodes_index`, and `general/extracellular_ephys/electrodes/location`.

ii.
```python
def get_unit_spike_times(units_group):
    spikes = units_group['spike_times'][:]
    index = units_group['spike_times_index'][:]
    ...

units = f['units']
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
spike_times_list = get_unit_spike_times(units)
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```

iii. Step 5 maps `units/spike_times` plus unit metadata to the target `neural` field and explicitly says to filter with `unit_quality == good`.

## 2-b. How is the `neural` data processed?

i. The agent reconstructs each unit’s spike train, then for each kept trial histograms spikes into 50 ms bins over a 4 s window and converts counts to firing rates by dividing by `BIN_SIZE`.

ii.
```python
BIN_SIZE = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
...
fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. Step 5 says to “Bin spikes into 50 ms bins from -2.5 s to +1.5 s around go cue.” Step 6 says the initial implementation “bins spikes at 50 ms.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level QC is `unit_quality == 'good'`. The code does not reproduce the reference repository’s region-specific `Idx` QC lists, histology intersection, or the paper’s reported 69,943-unit curated dataset.

ii.
```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
...
'n_units_good': int(good_mask.sum()),
```

iii. Step 5 says “Filter units using `units/unit_quality == good`.” Step 10 acknowledges this remains unresolved because it yields 154,948 units rather than the paper’s 69,943 curated units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns neural data to a *derived* go cue computed as `trial_start + 1.85`, not to the explicit per-trial `go_start_times` in the NWB. Every bin edge is formed by adding the fixed relative window to that derived time.

ii.
```python
# go cue at 1.85 s after trial start
GO_CUE_FROM_TRIAL_START = 1.85

trial_start = trials['start_time'][:].astype(float)
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
...
align = go_cue_times[i]
edges_abs = align + BIN_EDGES
centers_abs = align + centers_rel
```

iii. Step 4 says “If explicit event timestamps are absent, derive tone/go cue times from documented task structure relative to trial start.” Step 6/9 repeats that go cue and tone onset were derived from methods-defined timing rather than explicit NWB timestamps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins. The code does not rebin precomputed firing rates; it computes new 50 ms firing-rate bins directly from spike times.

ii.
```python
BIN_SIZE = 0.05
N_BINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
...
fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. This follows the decoder task spec in the instructions. Step 3 of the notes explicitly contrasts the visible 40 ms analysis bins in reference scripts with the decoder’s required 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `intervals/trials/start_time` plus the hard-coded constant `TONE_ONSET_FROM_TRIAL_START = 0.0`. The code does not use explicit sample/tone event timestamps.

ii.
```python
TONE_ONSET_FROM_TRIAL_START = 0.0
...
trial_start = trials['start_time'][:].astype(float)
tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START
```

iii. Step 5 states “Derive tone onset from trial `start_time`.” The metadata note also says “Go cue and tone onset derived from methods-defined task timing relative to trial start.”

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each kept trial, the code subtracts the derived tone-onset time from every bin center and stores the resulting continuous time series.

ii.
```python
centers_abs = align + centers_rel
...
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
inp = np.stack([time_from_tone, photo], axis=0)
```

iii. The agent’s planned mapping in Step 5 was to construct “time-from-tone input ... on common 50 ms grid.”

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 50 ms bin centers used for the neural matrix. Because the neural alignment anchor is the derived go cue, the time-from-tone series inherits that same alignment choice.

ii.
```python
align = go_cue_times[i]
centers_rel = BIN_CENTERS.copy()
centers_abs = align + centers_rel
...
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. Step 5 says to rasterize behavior inputs onto the same 50 ms go-cue-centered grid used for neural data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from the per-trial `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration` fields, together with `intervals/trials/start_time`.

ii.
```python
photostim_onset = [parse_optional_time(x) for x in trials['photostim_onset'][:]]
photostim_duration = [parse_optional_time(x) for x in trials['photostim_duration'][:]]
...
p_on = photostim_onset[i]
p_dur = photostim_duration[i]
if p_on is not None and p_dur is not None:
    p_start = trial_start[i] + p_on
    p_stop = p_start + p_dur
```

iii. Step 5 explicitly maps “trial `start_time` plus ... `photostim_onset`, `photostim_duration`” to the decoder’s photostim input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The code parses `N/A` values to `None`, converts non-missing onset/duration strings to floats, computes absolute photostim start/stop times, and marks each 50 ms bin center as 1 if it falls within `[start, stop)`, else 0.

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

photo = np.zeros(N_BINS, dtype=np.float32)
...
photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. Step 5 says to “construct ... photostim-on binary time series on common 50 ms grid.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim series is placed on the same `centers_abs` grid as the neural bins. Since `centers_abs` are derived from the agent’s fixed-offset go cue, the photostim/neural alignment inherits that same trial-window placement.

ii.
```python
align = go_cue_times[i]
centers_abs = align + centers_rel
...
photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. Step 5 says all streams will be “align[ed] ... to go cue in 50 ms bins.”

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `intervals/trials/trial_instruction`, mapped as left = 0 and right = 1.

ii.
```python
CHOICE_MAP = {b'left': 0, b'right': 1, 'left': 0, 'right': 1}
...
trial_instruction = np.array([_decode(x) for x in trials['trial_instruction'][:]])
...
choice_val = CHOICE_MAP[trial_instruction[i]]
```

iii. Step 5 maps the trial field `trial_instruction` to choice, and the decoder spec asks for left/right categorical choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The mapped per-trial scalar is expanded to a length-`N_BINS` constant time series, so each bin in a trial carries the same choice label.

ii.
```python
choice_val = CHOICE_MAP[trial_instruction[i]]
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
```

iii. The agent chose to represent the per-trial target as a time-varying row repeated across bins, matching its comment “outputs: 3 per-trial categorical rows repeated across time.”

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `intervals/trials/outcome`.

ii.
```python
OUTCOME_MAP = {b'ignore': 0, b'miss': 1, b'hit': 2, 'ignore': 0, 'miss': 1, 'hit': 2}
...
outcome = np.array([_decode(x) for x in trials['outcome'][:]])
...
outcome_val = OUTCOME_MAP[outcome[i]]
```

iii. Step 5 lists the trial `outcome` field as one of the required decoder outputs.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps `ignore/miss/hit` to `0/1/2` and repeats that categorical label across all time bins in the trial.

ii.
```python
outcome_val = OUTCOME_MAP[outcome[i]]
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. This follows the decoder target spec and the same repeated-per-bin design used for choice and early lick.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. The prompt here appears to have a typo: the agent’s code has no `distance to reward zone` output. Interpreting this as the alignment of `output` *Outcome*, the code aligns it by repeating the per-trial outcome value over the same 50 ms bins as neural activity.

ii.
```python
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
...
out = np.stack([choice_ts, outcome_ts, early_ts, tongue_disc], axis=0)
```

iii. There is no separate justification for a reward-zone variable in the notes; this section is inferred from the actual output structure in `convert_data.py`.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from `intervals/trials/early_lick`.

ii.
```python
EARLY_MAP = {b'no early': 0, b'early': 1, 'no early': 0, 'early': 1}
...
early_lick = np.array([_decode(x) for x in trials['early_lick'][:]])
...
early_val = EARLY_MAP[early_lick[i]]
```

iii. Step 5 includes `early_lick` among the chosen trial fields.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `no early` to 0 and `early` to 1, then repeats that label across all time bins in the trial.

ii.
```python
early_val = EARLY_MAP[early_lick[i]]
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. This is the same repeated-per-bin representation used for the other per-trial categorical outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The agent derives tongue y-position from the first behavioral time-series group whose name contains `TongueTracking`, preferring `Camera0...` if present. It uses the second column of `data` as `y`.

ii.
```python
def choose_tongue_group(f):
    bts = f.get('acquisition/BehavioralTimeSeries')
    ...
    candidates = [k for k in bts.keys() if 'TongueTracking' in k]
    ...
    candidates = sorted(candidates, key=lambda x: (not x.startswith('Camera0'), x))
    return bts[candidates[0]]

...
tongue_data = tongue_group['data'][:]
tongue_t = tongue_group['timestamps'][:].astype(float)
...
tongue_y = tongue_data[:, 1].astype(float)
```

iii. Step 5 explicitly says to use `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` y-coordinate with its timestamps.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The code assumes the tracking array columns are `[x, y, likelihood]`. If a likelihood column exists, it masks any sample with `likelihood < 0.5` to `NaN`. Session thresholds are then computed from all remaining finite `y` samples.

ii.
```python
# Convention in these tracking arrays is assumed [x, y, likelihood]
if tongue_data.ndim == 2 and tongue_data.shape[1] >= 2:
    tongue_y = tongue_data[:, 1].astype(float)
    # mask low-confidence frames if likelihood available
    if tongue_data.shape[1] >= 3:
        lik = tongue_data[:, 2].astype(float)
        tongue_y = tongue_y.copy()
        tongue_y[lik < 0.5] = np.nan

if tongue_y is not None and np.isfinite(tongue_y).any():
    q40, q60 = np.nanpercentile(tongue_y, [40, 60])
```

iii. Step 5 justifies the `Camera0...` y-coordinate choice. Step 7 says tongue discretization “improved after lowering likelihood threshold,” but the final code still uses `0.5`.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. It uses per-session 40th and 60th percentiles. Values below `q40` become category 0, values above `q60` become category 2, and everything else remains category 1.

ii.
```python
q40, q60 = np.nanpercentile(tongue_y, [40, 60])
...
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
...
tongue_disc[finite & (y < q40)] = 0
tongue_disc[finite & (y > q60)] = 2
```

iii. This directly follows the decoder task specification in the instructions and Step 5’s planned mapping.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each neural bin center, the code finds the nearest finite tongue timestamp with `np.searchsorted(..., side='left')`, clips to the valid index range, drops matches more than 100 ms away, and defaults missing bins to the middle class (1). The aligned discrete tongue row is then stacked with the other outputs for the same trial.

ii.
```python
valid_idx = np.flatnonzero(np.isfinite(tongue_y))
...
vt = tongue_t[valid_idx]
vy = tongue_y[valid_idx]
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
far = np.abs(vt[idx] - centers_abs) > 0.1
y = y.copy()
y[far] = np.nan
```

iii. Step 5 says to “resample” behavioral/video streams onto the same 50 ms grid. The exact nearest-sample plus 100 ms cutoff rule is only visible in code.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing optional times encoded as `'N/A'` become `None`. Missing/undecodable subject IDs fall back to the parent folder name. Missing brain-region metadata fall back to `'unknown'`. Missing or unusable tongue data produce `q40/q60 = NaN` and a default tongue category of 1 for all bins. Trials with bad labels, missing neural coverage, or all-zero binned firing rates are dropped.

ii.
```python
if isinstance(x, str) and x == 'N/A':
    return None
...
except Exception:
    return path.parent.name
...
except Exception:
    brain_region_labels = ['unknown' for _ in good_spike_times]
...
if tongue_y is not None and np.isfinite(tongue_y).any():
    q40, q60 = np.nanpercentile(tongue_y, [40, 60])
else:
    q40, q60 = np.nan, np.nan
...
else:
    tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
```

iii. The notes call out several of these explicitly: Step 4/5 says to handle unavailable fields with documented fallbacks; Step 10 explains that out-of-coverage and all-zero trials were ultimately excluded.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is the nested per-trial/per-neuron histogramming of spike times, plus the initial full-materialization of `units/spike_times` for every session. Reading large tongue-tracking arrays also adds cost.

ii.
```python
spikes = units_group['spike_times'][:]
...
for i in range(n_trials):
    ...
    fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
    for n, st in enumerate(good_spike_times):
        mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
        if np.any(mask):
            counts, _ = np.histogram(st[mask], bins=edges_abs)
```

iii. Step 6 explicitly says “Per-neuron histogram loop is a bottleneck,” with a projected 50+ minute runtime on the full dataset.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are: reconstructing spike-time lists in `get_unit_spike_times`, the nested `for trial -> for neuron -> np.histogram` loop, the per-unit brain-region extraction loop over `electrodes_index`, and the per-trial nearest-tongue lookup.

ii.
```python
for stop in index:
    out.append(spikes[start:stop])

for i in range(n_trials):
    ...
    for n, st in enumerate(good_spike_times):
        ...

for stop_i in eidx:
    inds = elec[start_i:stop_i]
    ...
```

iii. Only the histogram bottleneck is stated in the notes; the rest is inferred from the code’s structure.

## 10-c. What processing does the code repeat multiple times?

i. It recomputes `go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START` twice in `extract_session`, rebuilds absolute bin edges and centers separately for every trial, and repeats constant per-trial output labels across all time bins even though they do not vary within trial.

ii.
```python
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
...
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
...
edges_abs = align + BIN_EDGES
centers_rel = BIN_CENTERS.copy()
centers_abs = align + centers_rel
...
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. There is no explicit note for these repetitions; they are visible directly in the final script.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes full `input`/`output` arrays even for trials later dropped because `fr` is all zeros; it derives and stores brain-region/session metadata that the provided decoder does not need for the core train/verify path; and it performs a likelihood-mask branch for tongue data that, with the final `0.5` threshold, can erase nearly all tongue samples before discretization.

ii.
```python
inp = np.stack([time_from_tone, photo], axis=0)
...
out = np.stack([choice_ts, outcome_ts, early_ts, tongue_disc], axis=0)

if np.all(fr == 0):
    continue

...
'brain_regions': brain_regions,
'brain_region_idx': brain_region_idx,
'metadata': {
    ...
    'session_info': session_info,
}
```

iii. Step 10 notes that many zero-neural trials were created and then excluded. The rest is inferred from how the final dataset is constructed versus what the decoder actually consumes.
