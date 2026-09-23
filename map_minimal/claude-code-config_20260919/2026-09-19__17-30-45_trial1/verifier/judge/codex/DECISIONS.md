# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data/sub-*/*.nwb`, sorts the paths, and processes each NWB file as one session. It loads the files with `h5py` rather than `pynwb`, reads HDF5 groups directly, and parallelizes session processing with a multiprocessing pool. Within each file it reads subject info, the trials table, the units table, and behavioral event/time-series groups.

ii. 
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
...
with Pool(args.workers) as pool:
    for i, s in enumerate(pool.imap(_worker, jobs)):
        summaries.append(s)
```

```python
with h5py.File(path, 'r') as nwb:
    subject = nwb['general/subject/subject_id'][()].decode()
    trials = nwb['intervals/trials']
    ...
    go_times = nwb['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. In the trajectory's final summary, the AI says it read NWB files directly with `h5py` and treated the dataset as one file per session. The code comments say it was trying to follow the papers and `/app/code` unless the decoder task forced a deviation.

## 1-b. How are the data split into subjects?

i. The AI uses each file's `general/subject/subject_id` field as the subject identifier. After processing sessions, it builds `subjects` as the sorted unique IDs and `subject_idx` as an index per retained session.

ii. 
```python
subject = nwb['general/subject/subject_id'][()].decode()
```

```python
subjects = sorted({s['subject'] for s in kept})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_index[s['subject']] for s in kept], dtype=np.int64),
```

iii. The trajectory summary reports 28 mice and describes subject handling as coming directly from the NWB files. No alternative subject grouping logic is present.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order follows the sorted file list, and the session identifier it stores is the filename stem rather than `nwb.identifier`.

ii. 
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
```

```python
name = os.path.basename(path).replace('.nwb', '')
...
summary = dict(session=name, subject=subject, path=path, ...)
```

iii. In the final trajectory summary, the AI describes the data as NWB session files and reports counts in terms of retained sessions out of all session files.

## 1-d. How are the data split into trials?

i. The AI uses rows of `intervals/trials` as trials, with one go cue per row from `acquisition/BehavioralEvents/go_start_times/timestamps`. It asserts the go-cue count matches the trial count, then uses the retained trial indices to slice all per-trial quantities.

ii. 
```python
trials = nwb['intervals/trials']
trial_start = trials['start_time'][:]
trial_stop = trials['stop_time'][:]
...
go_times = nwb['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(go_times) == len(trial_start), 'go cue count != trial count'
```

```python
trial_idx = np.flatnonzero(keep)
go = go_times[trial_idx]
```

iii. The code and final summary both frame go-cue-aligned windows as being built per behavioral trial, so the trial table is the source of trial structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out `auto_water` and `free_water` trials, then further restricts to trials covered by the `obs_intervals` of every good unit. It keeps photostim, early-lick, and ignore trials. Separately, session-level behavioral criteria are applied before any retained trials are emitted.

ii. 
```python
auto_water = trials['auto_water'][:] != 0
free_water = trials['free_water'][:] != 0
...
good = _str(nwb['units']['classification'][:]) == 'good'
keep = ~(auto_water | free_water)
if good.any():
    keep &= observed_trials(nwb, good, trial_start)
```

```python
def observed_trials(nwb, good, trial_start):
    ...
    covered = np.ones(len(trial_start), dtype=bool)
    for u in np.flatnonzero(good):
        covered &= np.isin(trial_start, obs[starts[u]:index[u], 0])
    return covered
```

iii. In the final trajectory summary, the AI explicitly justifies dropping auto-water/free-water trials because water was given regardless of choice, and dropping trials outside `obs_intervals` because they would otherwise look like brain-wide silence. It also explicitly says photostim, early-lick, and ignore trials were kept because the decoder task requires them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units/spike_times` and `units/spike_times_index`, using go-cue timestamps to define bin edges. Unit selection is based on `units/classification`, `units/is_good_trials`, unit electrode assignments, and CCF metadata for region labels.

ii. 
```python
spike_index = units['spike_times_index'][:]
starts = np.concatenate([[0], spike_index[:-1]])
...
spike_times = units['spike_times']
for i, u in enumerate(unit_ids):
    st = spike_times[starts[u]:spike_index[u]]
```

```python
go_times = nwb['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
```

iii. The trajectory summary says firing rates are spike counts around `go_start_times`, and neuron curation is based on `units/classification == 'good'` with an additional `is_good_trials` restriction.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into 80 non-overlapping 50 ms bins from -2.5 s to +1.5 s around the go cue. For each retained unit, it uses `np.searchsorted` over flattened absolute bin edges, differences adjacent counts, and divides by 0.05 s to convert counts to firing rates.

ii. 
```python
T_START = -2.5
T_STOP = 1.5
BIN_WIDTH = 0.05
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
```

```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
...
idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
rates[i] = np.diff(idx, axis=1) / BIN_WIDTH
```

iii. The final trajectory summary says this mirrors the reference pipeline's go-cue-aligned spike binning, except using the assignment's -2.5 to +1.5 s window and 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first keeps only units with `classification == 'good'`. It then further drops any of those units that are not flagged good on every analyzed trial via `units/is_good_trials`. A session is rejected if no units remain.

ii. 
```python
good = _str(nwb['units']['classification'][:]) == 'good'
...
good[good] = units_good_on_trials(nwb, good, trial_idx)
...
if good.sum() == 0:
    summary['rejected'] = True
    summary['reject_reason'] = 'no unit passes quality control on all trials'
    return summary
```

```python
def units_good_on_trials(nwb, good, trial_idx):
    flags = nwb['units']['is_good_trials'][:][good][:, trial_idx]
    return flags.all(axis=1)
```

iii. The final trajectory summary says `classification == 'good'` comes from the spike-sorting white paper and that the extra `is_good_trials` restriction was added because the AI wanted a fixed neuron-by-trial matrix.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go-cue onset by adding a fixed relative edge vector to each trial's go-cue timestamp. The code assumes spikes and go cues are already on the same absolute timebase.

ii. 
```python
go_times = nwb['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
```

iii. The trajectory summary explicitly says firing rates are computed in windows around `go_start_times`, and the code comments say the bins are laid out relative to the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms, with 80 non-overlapping bins per trial spanning 4 seconds. The AI does not apply any additional rebinning or smoothing beyond this binning step.

ii. 
```python
BIN_WIDTH = 0.05
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))       # 80
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
```

iii. The final trajectory summary says the code uses non-overlapping 50 ms bins tiling [-2.5, +1.5] s because that is what the decoder task asked for.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times/timestamps`, `intervals/trials/start_time`, and go-cue timestamps. It finds the last sample-epoch onset between trial start and go cue for each trial.

ii. 
```python
sample = nwb['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
onsets = np.full(len(go_times), np.nan)
lo = np.searchsorted(sample, trial_start, side='left')
hi = np.searchsorted(sample, go_times, side='right')
for i in range(len(go_times)):
    if hi[i] > lo[i]:
        onsets[i] = sample[hi[i] - 1]
```

iii. In the final trajectory summary, the AI says it uses the last sample-epoch onset before the go cue because early licks replay the epoch and that is the tone the animal answered.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After finding tone onset per trial, the AI computes each bin's value as absolute bin-center time minus tone time, i.e. `(go + bin_center) - tone`. If no tone onset is found, it falls back to `go - 1.85`, assuming a 0.65 s sample plus 1.2 s delay.

ii. 
```python
missing_tone = ~np.isfinite(tone)
if missing_tone.any():
    tone[missing_tone] = go[missing_tone] - 1.85
...
time_from_tone = (go[:, None] + BIN_CENTERS[None, :]
                  - tone[:, None]).astype(np.float32)
```

iii. The trajectory summary justifies the "last onset before go" rule from replayed sample epochs. The fallback is justified in a code comment as the nominal task timing when no onset is recorded.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 80 go-cue-relative bin centers used for neural firing rates, so each time value corresponds to the matching neural bin.

ii. 
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
...
time_from_tone = (go[:, None] + BIN_CENTERS[None, :]
                  - tone[:, None]).astype(np.float32)
```

iii. The code uses the same `BIN_CENTERS` and `go` values used in spike binning, and the trajectory summary describes both inputs and neural data as go-cue aligned.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from trial-table columns `photostim_onset` and `photostim_duration`, together with trial start times and go-cue times to express stimulation on the go-cue-relative axis.

ii. 
```python
onset = np.array([_parse_float(v) for v in _str(trials['photostim_onset'][:])])
duration = np.array([_parse_float(v) for v in _str(trials['photostim_duration'][:])])
...
on = trial_start + onset - go_times
off = on + duration
```

iii. The final trajectory summary says photostim is built from trial-relative `photostim_onset` plus `photostim_duration`.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI parses the string-valued onset and duration fields into floats, treating non-numeric values such as `N/A` as `NaN`. It then creates a binary `(n_trials, n_bins)` mask where a bin is marked on if the stimulation interval overlaps that bin at all.

ii. 
```python
def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
```

```python
stim = np.zeros((len(go_times), N_BINS), dtype=np.float32)
on = trial_start + onset - go_times
off = on + duration
valid = np.isfinite(on) & np.isfinite(off)
for i in np.flatnonzero(valid):
    stim[i] = ((BIN_EDGES[1:] > on[i]) & (BIN_EDGES[:-1] < off[i])).astype(np.float32)
```

iii. The trajectory summary justifies photostim as a binary mask. The code comment specifically says a bin is marked as stimulated when it overlaps the stimulation interval.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI converts photostim onset from trial-relative absolute time into time relative to the go cue, then compares that interval against the same go-cue-relative bin grid used for neural firing rates.

ii. 
```python
on = trial_start + onset - go_times          # relative to the go cue
off = on + duration
...
stim[i] = ((BIN_EDGES[1:] > on[i]) & (BIN_EDGES[:-1] < off[i])).astype(np.float32)
```

iii. The final trajectory summary says all inputs are expressed on the same go-cue-aligned time axis as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives lick direction from `trial_instruction` and `outcome`. A hit means the instructed side, a miss means the opposite side, and an ignore means no lick.

ii. 
```python
instruction = _str(trials['trial_instruction'][:])
outcome = _str(trials['outcome'][:])
...
other = np.where(instruction == 'left', 'right', 'left')
licked = np.where(outcome == 'hit', instruction,
                  np.where(outcome == 'miss', other, 'no lick'))
```

iii. The final trajectory summary says the AI checked this derivation against recorded lick times and found perfect agreement.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps `left`, `right`, and `no lick` to categorical codes using the order in `CHOICE_VALUES`, then repeats the chosen class across all 80 time bins in the output array.

ii. 
```python
CHOICE_VALUES = ['left', 'right', 'no lick']
...
choice = np.array([CHOICE_VALUES.index(v) for v in licked], dtype=np.int8)
```

```python
ones = np.ones(N_BINS, dtype=np.int8)
...
outputs.append(np.stack([choice[j] * ones, outcome_code[j] * ones,
                         early_code[j] * ones, tongue[j]]).astype(np.int8))
```

iii. The trajectory summary says lick direction is a per-trial output, so it is represented categorically rather than as an event stream.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI uses the `outcome` column from the trial table directly.

ii. 
```python
outcome = _str(trials['outcome'][:])
```

iii. No further derivation is claimed in the trajectory; the code treats the stored outcome labels as the source of truth.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore`, `miss`, and `hit` to integer category codes via `OUTCOME_CODE`, then repeats that category across all 80 bins.

ii. 
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
...
outcome_code = np.array([OUTCOME_CODE[v] for v in outcome], dtype=np.int8)
```

```python
outputs.append(np.stack([choice[j] * ones, outcome_code[j] * ones,
                         early_code[j] * ones, tongue[j]]).astype(np.int8))
```

iii. The trajectory summary lists outcome as one of the categorical decoder outputs and does not justify any transformation beyond coding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The AI derives early lick from the `early_lick` trial-table column.

ii. 
```python
early = _str(trials['early_lick'][:]) == 'early'
```

iii. The trajectory summary says early-lick trials were kept because early lick is a required decoder output class.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI converts the Boolean `early` array into integer codes `0/1` with `astype(np.int8)` and repeats the category across all 80 bins.

ii. 
```python
early_code = early.astype(np.int8)
```

```python
outputs.append(np.stack([choice[j] * ones, outcome_code[j] * ones,
                         early_code[j] * ones, tongue[j]]).astype(np.int8))
```

iii. The code and trajectory both treat early lick as a per-trial categorical label required by the decoder task.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically its `timestamps` and `data` array. It uses column 1 as `y` and column 2 as DeepLabCut likelihood.

ii. 
```python
track = nwb['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
ts = track['timestamps'][:]
data = track['data'][:]
...
y = chunk[:, 1]
visible = chunk[:, 2] >= TONGUE_P_CUTOFF
```

iii. The final trajectory summary says tongue y uses DeepLabCut likelihood and per-session percentile thresholds.

## 8-b. How is `output` *Tongue y-position* derived?

i. For each retained trial, the AI finds frames in the go-cue-aligned window, keeps only frames with likelihood at least 0.5, averages visible-frame `y` values within each 50 ms bin, then computes the 40th and 60th percentiles over all visible binned values across analyzed trials in that session.

ii. 
```python
for i, go in enumerate(go_times):
    lo, hi = np.searchsorted(ts, [go + T_START, go + T_STOP])
    ...
    visible = chunk[:, 2] >= TONGUE_P_CUTOFF
    ...
    counts = np.bincount(which[ok], minlength=N_BINS)
    sums = np.bincount(which[ok], weights=y[ok], minlength=N_BINS)
    seen = counts > 0
    y_binned[i, seen] = sums[seen] / counts[seen]
```

```python
seen = np.isfinite(y_binned)
...
if seen.any():
    p40, p60 = np.percentile(y_binned[seen], TONGUE_PCTILES)
```

iii. In the trajectory summary, the AI justifies the 0.5 cutoff by saying the likelihood is essentially binary and justifies per-session 40th/60th percentile discretization as required by the task.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses four categories: `0` if the bin mean is below the session 40th percentile, `1` if it is between the 40th and 60th percentiles, `2` if above the 60th percentile, and `3` if the tongue is not visible in that bin.

ii. 
```python
classes = np.full((n_trials, N_BINS), 3, dtype=np.int8)
if seen.any():
    p40, p60 = np.percentile(y_binned[seen], TONGUE_PCTILES)
    vals = y_binned[seen]
    classes[seen] = np.where(vals < p40, 0, np.where(vals <= p60, 1, 2)).astype(np.int8)
```

iii. The code comment says this implements the task's four-class discretization, with class `3` for bins lacking a confident tongue detection.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue data by selecting camera frames between `go + T_START` and `go + T_STOP` and assigning them to the same go-cue-relative 50 ms bins used for neural activity.

ii. 
```python
lo, hi = np.searchsorted(ts, [go + T_START, go + T_STOP])
...
which = np.searchsorted(BIN_EDGES, ts[lo:hi] - go, side='right') - 1
ok = visible & (which >= 0) & (which < N_BINS)
```

iii. The trajectory summary flags that video, like spikes, only covers trial time, but otherwise treats the camera timestamps as being on the same clock as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing-data cases explicitly. `photostim_onset` and `photostim_duration` strings that are not numeric become `NaN`. Trials with missing sample onset would fall back to a nominal 1.85 s tone-to-go interval. Bins with no visible tongue remain `not visible`. If a unit lacks a CCF annotation, the code raises an error; if a unit lacks an ML coordinate, it falls back to probe-target hemisphere. Sessions with no good units or with too few surviving trials are dropped.

ii. 
```python
def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
```

```python
if missing_tone.any():
    tone[missing_tone] = go[missing_tone] - 1.85
```

```python
side = np.where(np.isnan(ml_unit),
                np.array([t.split(' ')[0] if t else 'left' for t in target]),
                np.where(ml_unit >= ML_MIDLINE, 'left', 'right'))
...
if group is None:
    raise ValueError('good unit without a CCF annotation')
```

iii. The justifications are mostly in code comments: use nominal task timing if tone onset is missing, use probe target when ML coordinate is missing, and avoid inventing data by dropping sessions/trials that fail curation.

## 10-a. What are the most time-consuming steps of the code?

i. The AI's code is structured as if NWB I/O and per-session processing are the dominant costs. The heavy steps are opening each NWB file, loading large spike and tongue arrays, looping over units in spike binning, and then writing per-session cache pickles plus the final pickle.

ii. 
```python
with h5py.File(path, 'r') as nwb:
    ...
    rates = bin_spikes(nwb, good, go)
    ...
    tongue, p40, p60, frac_visible = tongue_classes(nwb, go)
```

```python
with open(out_path, 'wb') as fh:
    pickle.dump({'neural': neural, 'input': inputs, 'output': outputs,
                 'regions': regions, 'subject': subject, 'session': name},
                fh, protocol=4)
```

iii. In the trajectory summary, the AI emphasizes direct NWB reads, population spike binning, and full-dataset serialization. The use of multiprocessing and per-session caches also implies it saw I/O and session conversion as the expensive parts.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain non-vectorized: the loop over good units in `observed_trials`, the loop over good units in `bin_spikes`, the loop over valid trials in `photostim_series`, the loop over trials in `tongue_classes`, and the final loop that assembles per-trial arrays. These could potentially be vectorized or rewritten to reduce Python overhead.

ii. 
```python
for u in np.flatnonzero(good):
    covered &= np.isin(trial_start, obs[starts[u]:index[u], 0])
```

```python
for i, u in enumerate(unit_ids):
    st = spike_times[starts[u]:spike_index[u]]
    idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
    rates[i] = np.diff(idx, axis=1) / BIN_WIDTH
```

```python
for i, go in enumerate(go_times):
    ...
for j in range(len(trial_idx)):
    neural.append(np.ascontiguousarray(rates[:, j, :]))
```

iii. The trajectory does not explicitly discuss vectorization, but the final summary and code structure show the AI prioritized clarity and throughput via multiprocessing rather than removing all Python loops.

## 10-c. What processing does the code repeat multiple times?

i. The AI does not recompute the same derived data many times, but it does repeat some I/O and parsing. It writes each kept session to a cache pickle and later reopens those caches during final assembly. It also parses photostimulation strings once for trial curation (`stim_trial`) and again when building the photostim time series.

ii. 
```python
stim_trial = _str(trials['photostim_onset'][:]) != 'N/A'
...
stim = photostim_series(trials, trial_start, go_times)[trial_idx]
```

```python
with open(out_path, 'wb') as fh:
    pickle.dump({...}, fh, protocol=4)
...
for i, s in enumerate(kept):
    with open(s['cache'], 'rb') as fh:
        sess = pickle.load(fh)
```

iii. The trajectory summary does not call this out, but the code clearly uses per-session caching to manage assembly, which trades repeated I/O for modular processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores a large amount of summary-only metadata that is not needed by downstream decoding: per-session performance, rejection reasons, tongue percentile values, visible-frame fractions, class-count summaries, mean firing rates, and session cache files. The temporary per-session pickle caches are also discarded after assembly from the decoder's point of view.

ii. 
```python
summary = dict(session=name, subject=subject, path=path,
               n_trials_total=int(len(trial_start)),
               n_trials_kept=int(keep.sum()),
               performance=float(performance),
               n_correct_left=n_left, n_correct_right=n_right,
               rejected=bool(rejected))
```

```python
summary.update(
    n_neurons=len(regions),
    n_trials=len(trial_idx),
    regions=regions,
    n_photostim_trials=int(np.sum(stim.any(axis=1))),
    choice_counts=np.bincount(choice, minlength=3).tolist(),
    outcome_counts=np.bincount(outcome_code, minlength=3).tolist(),
    early_counts=np.bincount(early_code, minlength=2).tolist(),
    tongue_counts=np.bincount(tongue.ravel(), minlength=4).tolist(),
    mean_rate=float(rates.mean()),
)
```

iii. The trajectory summary focuses on correctness and decoder performance, not these summaries, so these extra calculations appear to be for inspection/debugging rather than for the converted dataset itself.
