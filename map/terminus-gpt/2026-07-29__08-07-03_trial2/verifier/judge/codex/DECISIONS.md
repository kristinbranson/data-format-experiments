# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by recursively globbing `data/**/*.nwb` with `Path('data').rglob('*.nwb')`, then opens each NWB file with `h5py.File`. Inside each file it directly reads the `intervals/trials` and `units` groups and later behavioral groups as needed.

ii. 
```python
paths = sorted(Path('data').rglob('*.nwb'))
```

```python
with h5py.File(session_path, 'r') as f:
    trials = f['intervals/trials']
    ...
    units = f['units']
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI justified this by noting that the data are organized as subject-specific directories containing NWB session files, and in trajectory step 30 it explicitly decided to "use NWB as source" and write a converter directly on the raw NWB files.

## 1-b. How are the data split into subjects?

i. Subjects are taken per session from `general/subject/subject_id`; if that read fails, the AI falls back to the parent directory name. While assembling the dataset, it assigns a new subject index the first time each subject id is encountered.

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

iii. The notes say the dataset is organized by subject folder under `data/`, and the trajectory shows the AI inspected `general/subject/subject_id` early in dataset exploration. There is no further explicit justification for the fallback or the encounter-order subject list.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order is the sorted file path order, and each session id stored in metadata is the filename stem rather than the NWB `identifier`.

ii. 
```python
paths = sorted(Path('data').rglob('*.nwb'))
```

```python
'session_id': session_path.stem,
```

iii. `CONVERSION_NOTES.md` Step 2 says each subject directory contains NWB session files and describes the filename format as session-specific. The trajectory also shows the AI using "NWB session files" as the natural session boundary.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trial table `intervals/trials`. The AI reads the number of trials from `len(trials['id'])` and iterates trial-by-trial over row indices.

ii. 
```python
trials = f['intervals/trials']
n_trials = len(trials['id'])
...
for i in range(n_trials):
    if not valid_trials[i]:
        continue
```

iii. The trajectory shows the AI inspecting trial columns directly and concluding that trial metadata live in `intervals/trials`. It does not document a separate justification beyond using the NWB trial table as the source of behavioral trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials with recognized `trial_instruction`, `outcome`, and `early_lick` labels, then further filters to trials whose derived go-cue window lies between the minimum and maximum spike time seen among good units. After constructing trial neural activity, it also drops any trial whose firing-rate matrix is all zeros. At the session level it skips sessions with fewer than 2 kept trials.

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

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI wrote that it would "Restrict to trials with complete required variables." Later, in Step 10, it justified the spike-coverage filter as a fix for sessions where `trial_start` extended beyond available spikes, saying conversion "must exclude trials whose aligned window falls outside neural recording coverage."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units/spike_times` and `units/spike_times_index`, restricted to units with `unit_quality == 'good'`. Trial alignment uses `intervals/trials/start_time` plus a fixed `GO_CUE_FROM_TRIAL_START = 1.85` seconds to create putative go-cue times.

ii. 
```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
spike_times_list = get_unit_spike_times(units)
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```

```python
trial_start = trials['start_time'][:].astype(float)
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
```

iii. Step 5 of the notes says the variable mapping would use "`units/spike_times` + unit metadata in `units`" for neural data, and step 30 of the trajectory says the AI decided to "filter units by `unit_quality == good`" and "derive tone and go cue timing relative to `start_time` using methods-defined timing."

## 2-b. How is the `neural` data processed?

i. For each kept trial, the AI builds absolute bin edges from the trial's derived go cue, then loops over every good unit and uses `np.histogram` to count spikes in each 50 ms bin. Counts are divided by `BIN_SIZE` to convert to firing rate in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii. 
```python
edges_abs = align + BIN_EDGES
...
fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. In Step 6 the AI described the script as one that "bins spikes into 50 ms bins from -2.5 s to +1.5 s around go cue," and it flagged the per-neuron histogram loop as the main bottleneck. There is no separate methodological justification beyond matching the decoder binning requirement.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters units with `units/unit_quality == 'good'`. It does not use the NWB `classification` field. Sessions are later skipped only if no good units remain or if fewer than 2 trials survive.

ii. 
```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
```

iii. Step 5 of the notes explicitly says "Filter units using `units/unit_quality == good`." In the trajectory, the AI says `unit_quality` had values like `good` and `multi` and "likely supports filtering to good units"; later it documents that this yields 154,948 units and does not match the paper's 69,943 curated units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to a derived go cue defined as exactly 1.85 s after each trial's `start_time`. For each trial, it shifts the shared relative bin edges by that derived go-cue time and bins spikes in the resulting absolute interval.

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

iii. Step 4 and Step 5 of `CONVERSION_NOTES.md` say that when explicit go-cue fields were not found in the trial table, the AI would "derive tone/go cue times from documented task structure relative to trial start." Trajectory step 29 calls this "the most defensible approach" because the AI believed go/tone timestamps were absent from the relevant fields it inspected.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data are represented in 50 ms bins over a -2.5 s to +1.5 s window around the derived go cue, giving 80 bins per trial. Spike times are binned directly into that grid; there is no additional temporal rebinning after histogramming.

ii. 
```python
BIN_SIZE = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_BINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. The notes repeatedly justify this as following the decoder task requirement: Step 5 says "Use 50 ms bins for neural and derived time-varying variables," and Step 3 notes that reference analyses used other bins but the decoder task specifically requires 50 ms.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `intervals/trials/start_time` plus a fixed tone onset offset of 0.0 s from trial start. The AI does not use behavioral event timestamps such as `sample_start_times`.

ii. 
```python
TONE_ONSET_FROM_TRIAL_START = 0.0
...
trial_start = trials['start_time'][:].astype(float)
tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START
```

iii. In Step 5 notes, the AI states it will "derive tone onset from trial `start_time`" using task timing from methods. Trajectory step 29 says that because explicit tone fields were not found where it expected them, it would derive tone timing from the documented task structure.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each kept trial, the AI computes absolute bin centers for the neural grid and subtracts the trial's derived tone onset time, producing a continuous 80-sample trace of seconds since tone onset.

ii. 
```python
centers_rel = BIN_CENTERS.copy()
centers_abs = align + centers_rel
...
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. The notes justify this only indirectly: Step 5 says the input should be constructed on the common 50 ms grid after deriving tone timing from methods-defined trial structure. There is no additional justification for not using event timestamps.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by evaluating it on the same 80 bin centers used for neural firing rates. The AI computes the neural bin centers in absolute session time and subtracts tone onset, so the time-from-tone trace is defined sample-by-sample on the same grid as `fr`.

ii. 
```python
edges_abs = align + BIN_EDGES
centers_rel = BIN_CENTERS.copy()
centers_abs = align + centers_rel
...
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. Step 5 of the notes says "Align all streams to go cue" and "resample or rasterize onto this grid," so the intended justification was that all time-varying inputs and outputs should share the same go-cue-centered 50 ms bins as neural activity.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial table fields `photostim_onset`, `photostim_duration`, and `start_time`. `photostim_onset` and `photostim_duration` are parsed from strings, with `'N/A'` treated as missing.

ii. 
```python
photostim_onset = [parse_optional_time(x) for x in trials['photostim_onset'][:]]
photostim_duration = [parse_optional_time(x) for x in trials['photostim_duration'][:]]
```

```python
p_start = trial_start[i] + p_on
p_stop = p_start + p_dur
```

iii. Step 5 notes explicitly map "photostim timing from trial fields" and trajectory step 22 records that `photostim_onset`, `photostim_duration`, and `photostim_power` existed in the trial table, which the AI treated as exactly what it needed for this input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts onset and duration to absolute start/stop times within the session, then marks each 50 ms bin center as 1.0 if it falls within the stimulation interval and 0.0 otherwise.

ii. 
```python
photo = np.zeros(N_BINS, dtype=np.float32)
p_on = photostim_onset[i]
p_dur = photostim_duration[i]
if p_on is not None and p_dur is not None:
    p_start = trial_start[i] + p_on
    p_stop = p_start + p_dur
    photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
inp = np.stack([time_from_tone, photo], axis=0)
```

iii. The notes justify this by saying behavior/video streams would be "resampled or rasterized" onto the shared 50 ms grid. There is no more detailed justification beyond converting per-trial stimulation timing into a time-varying decoder input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned by comparing photostim absolute start/stop times against the same absolute bin centers used for neural data. So photostim is not expressed relative to go cue explicitly, but it is sampled on the same per-trial aligned grid.

ii. 
```python
centers_abs = align + centers_rel
...
photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. Step 5 notes say all streams should be aligned to go cue and represented on the common 50 ms grid. That is the only explicit justification given.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. In the final code, choice is derived only from the `trial_instruction` field. The AI does not combine `trial_instruction` with `outcome`, and it does not create a separate "no lick" class for ignore trials.

ii. 
```python
CHOICE_MAP = {b'left': 0, b'right': 1, 'left': 0, 'right': 1}
...
choice_val = CHOICE_MAP[trial_instruction[i]]
```

iii. The trajectory shows the AI confirming that `trial_instruction` exists and deciding it could be used for outputs. There is no explicit justification in the notes for ignoring `outcome`; the code simply implements choice as instructed side.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps `trial_instruction` to `0/1` for left/right and repeats that scalar across all 80 bins. The saved metadata only names two choice values, `['left', 'right']`.

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

iii. No explicit justification is documented beyond the apparent assumption that `trial_instruction` can stand in for choice. The verification output also confirms only classes 0 and 1 appear for choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table `outcome` field.

ii. 
```python
outcome = np.array([_decode(x) for x in trials['outcome'][:]])
```

iii. The AI identified `outcome` as an exact trial field during NWB inspection, as documented in trajectory steps 22 and 25. No further derivation was needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped with `OUTCOME_MAP` to 0 ignore, 1 miss, 2 hit, then repeated across all 80 bins as a per-trial categorical time series.

ii. 
```python
OUTCOME_MAP = {b'ignore': 0, b'miss': 1, b'hit': 2, 'ignore': 0, 'miss': 1, 'hit': 2}
...
outcome_val = OUTCOME_MAP[outcome[i]]
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. The notes say the converter would "use trial fields for outputs," and `outcome` was one of the explicitly confirmed fields. The mapping itself is straightforward and not further justified.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table `early_lick` field.

ii. 
```python
early_lick = np.array([_decode(x) for x in trials['early_lick'][:]])
```

iii. As with `outcome`, the trajectory shows the AI identified `early_lick` during raw NWB inspection and used it directly. There is no additional justification in the notes.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Strings are mapped with `EARLY_MAP` to 0 for `no early` and 1 for `early`, then repeated across all 80 bins.

ii. 
```python
EARLY_MAP = {b'no early': 0, b'early': 1, 'no early': 0, 'early': 1}
...
early_val = EARLY_MAP[early_lick[i]]
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. The notes justify using per-trial categorical outputs repeated across time because the target format expects all outputs under one array shape. No extra methodological rationale is given.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI searches `acquisition/BehavioralTimeSeries` for any group whose name contains `TongueTracking`, preferring `Camera0...` if present. It uses `timestamps` and, from `data`, assumes column 1 is tongue y-position and column 2 is likelihood.

ii. 
```python
def choose_tongue_group(f):
    bts = f.get('acquisition/BehavioralTimeSeries')
    ...
    candidates = [k for k in bts.keys() if 'TongueTracking' in k]
    ...
    return bts[candidates[0]]
```

```python
tongue_data = tongue_group['data'][:]
tongue_t = tongue_group['timestamps'][:].astype(float)
...
tongue_y = tongue_data[:, 1].astype(float)
...
lik = tongue_data[:, 2].astype(float)
```

iii. The trajectory shows the AI inspecting the available behavior time series and concluding that `Camera0_side_TongueTracking` provided `(x, y, likelihood)` style data. The code comment says this column convention is "assumed."

## 8-b. How is `output` *Tongue y-position* processed?

i. Frames with likelihood below 0.5 are set to `NaN`. Session thresholds `q40` and `q60` are computed from all finite tongue-y samples across the whole session, not from 50 ms bin means. For each trial/bin, the AI takes the nearest valid tongue frame to each neural bin center, discards it if it is more than 0.1 s away, and then assigns class 0 below `q40`, class 2 above `q60`, and otherwise leaves the default class 1.

ii. 
```python
if tongue_data.shape[1] >= 3:
    lik = tongue_data[:, 2].astype(float)
    tongue_y = tongue_y.copy()
    tongue_y[lik < 0.5] = np.nan
```

```python
if tongue_y is not None and np.isfinite(tongue_y).any():
    q40, q60 = np.nanpercentile(tongue_y, [40, 60])
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
finite = np.isfinite(y)
tongue_disc[finite & (y < q40)] = 0
tongue_disc[finite & (y > q60)] = 2
```

iii. The trajectory documents why this changed from an even worse initial version: the AI found extreme middle-class imbalance and concluded defaulting missing bins to the middle class was a problem, so it switched to "nearest valid tongue samples" and experimented with a lower confidence threshold, settling on 0.5. The notes still describe tongue discretization as under review and imbalanced.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The implemented thresholds are the 40th and 60th percentiles of all valid raw y-samples over the session. The code outputs only three categories, 0/1/2; missing or far-away bins are left at the default middle class 1. There is no explicit `not visible` class 3 in the final dataset.

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

iii. The notes justify the session-level percentile idea because the decoder spec asked for 40th/60th percentile session discretization. The trajectory explains the fallback to class 1 as part of the "nearest valid sample" workaround for sparse tongue visibility, not as a principled missing-data category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output to neural data by querying the nearest valid tongue sample to each neural bin center in absolute time, with a maximum tolerated gap of 0.1 s. It does not average camera frames within the same 50 ms bins used for the neural data.

ii. 
```python
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
far = np.abs(vt[idx] - centers_abs) > 0.1
y = y.copy()
y[far] = np.nan
```

iii. The trajectory explicitly says the AI moved to "nearest valid tongue samples only" after seeing that its first discretization made almost all bins class 1. This was a pragmatic fix for sparse tracking coverage rather than something justified from the reference processing.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing-data cases by exclusion or fallback. Unknown trial labels make a trial invalid and it is dropped. Missing photostim values (`'N/A'`) become `None`, leaving the photostim trace all zeros. Missing subject ids fall back to the directory name. Missing/low-confidence tongue frames become `NaN`, but bins without an acceptable nearby frame are effectively assigned tongue class 1 rather than a separate missing class. Trials with all-zero neural matrices are dropped.

ii. 
```python
if isinstance(x, str) and x == 'N/A':
    return None
```

```python
valid_trials = np.array([
    ti in ('left', 'right') and oc in ('ignore', 'miss', 'hit') and el in ('no early', 'early')
    for ti, oc, el in zip(trial_instruction, outcome, early_lick)
], dtype=bool)
```

```python
except Exception:
    return path.parent.name
```

```python
tongue_y[lik < 0.5] = np.nan
...
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
```

```python
if np.all(fr == 0):
    continue
```

iii. Step 10 of the notes justifies the spike-coverage and all-zero-trial exclusions as necessary because some trial windows had no concurrent neural recording. For tongue data, the trajectory justification was practical: sparse visibility led the AI to use nearest valid samples and a middle-class default instead of a dedicated missing-data class.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is the per-trial, per-unit spike histogramming, plus the cost of loading large NWB arrays such as spike times and tongue tracking data. The AI itself noted the per-neuron histogram loop as the main bottleneck and the conversion log shows some sessions taking tens of seconds to over 100 seconds.

ii. 
```python
for i in range(n_trials):
    ...
    fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
    for n, st in enumerate(good_spike_times):
        mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
        if np.any(mask):
            counts, _ = np.histogram(st[mask], bins=edges_abs)
            fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. Step 6 of `CONVERSION_NOTES.md` explicitly says "Per-neuron histogram loop is a bottleneck," estimating very long full-dataset runtime before later optimizations. The full conversion log corroborates that some sessions take 50 to 140 seconds.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are: the nested trial-by-trial, unit-by-unit neural histogram loop; the loop over `spike_times_index` inside `get_unit_spike_times`; the loop over electrode mappings used to build brain-region labels; and the per-trial nearest-sample tongue loop logic. The biggest avoidable cost is recomputing spike masks and histograms separately for every unit in every trial.

ii. 
```python
for stop in index:
    out.append(spikes[start:stop])
    start = stop
```

```python
for stop_i in eidx:
    inds = elec[start_i:stop_i]
    ...
    all_unit_regions.append(region)
```

```python
for i in range(n_trials):
    ...
    for n, st in enumerate(good_spike_times):
        mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
        ...
```

iii. The AI only explicitly called out the per-neuron histogram loop in the notes. It did not document broader vectorization opportunities, but they are evident from the final code structure.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations: `go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START` is assigned twice; the same bin-center arrays are copied on every trial; trial outputs and inputs are constructed before the code checks whether the neural matrix is all zeros and discards that trial; and every unit is rescanned with a fresh mask for every trial rather than binning all trials jointly.

ii. 
```python
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
...
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
```

```python
centers_rel = BIN_CENTERS.copy()
centers_abs = align + centers_rel
```

```python
inp = np.stack([time_from_tone, photo], axis=0)
...
out = np.stack([choice_ts, outcome_ts, early_ts, tongue_disc], axis=0)

if np.all(fr == 0):
    continue
```

iii. The notes focus on runtime bottlenecks rather than repeated processing, so there is no explicit AI-written justification for these repetitions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is that the converter fully computes `inp`, `out`, and the trial neural matrix before dropping any all-zero-neural trial. It also copies `BIN_CENTERS` per trial, computes and stores session `q40/q60` mainly for logging, and performs some metadata bookkeeping that is not used by downstream decoding. The biggest unnecessary cost is work spent on trials that are later thrown away.

ii. 
```python
inp = np.stack([time_from_tone, photo], axis=0)
...
out = np.stack([choice_ts, outcome_ts, early_ts, tongue_disc], axis=0)

if np.all(fr == 0):
    continue
```

```python
centers_rel = BIN_CENTERS.copy()
```

```python
'q40': float(q40) if np.isfinite(q40) else None,
'q60': float(q60) if np.isfinite(q60) else None,
```

iii. The AI did not explicitly justify this extra work. The notes instead acknowledge runtime issues and the need for further optimization, which implicitly recognizes that some of the implementation is inefficient.
