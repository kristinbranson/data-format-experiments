# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script treats the raw NWB files as the source of truth. It glob-loads every `data/sub-*/*.nwb` file, opens each session with `h5py`, and inside each file loads the trial table, behavioral event timestamps, tongue tracking time series, unit classifications, spike times, and optional `is_good_trials`.

ii. 
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
for p in files:
    info = process_session(p, edges, bin_centers, show_processing=args.show_processing)
```

```python
with h5py.File(path, 'r') as f:
    trial = load_trial_table(f)
    go_times = load_event_times(f, 'go_start_times')
    sample_event_times = load_event_times(f, 'sample_start_times')
    left_licks = load_event_times(f, 'left_lick_times')
    right_licks = load_event_times(f, 'right_lick_times')
    tongue_ts, tongue_data = load_tongue(f)
    ...
    cls = decode_arr(f['units']['classification'][()])
    spike_times = f['units']['spike_times'][()]
    spike_index = f['units']['spike_times_index'][()]
```

iii. In `CONVERSION_NOTES.md`, the agent says the raw NWB files are the “source universe,” even though the reference code operates on curated per-session pickles. The trajectory shows it explicitly decided to reconstruct the preprocessing from raw NWB rather than reuse the paper’s preprocessed pickle format.

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB subject metadata, specifically `general/subject/subject_id`. After all valid sessions are processed, unique subject IDs are sorted into `subjects`, and each session gets a `subject_idx`.

ii. 
```python
subject = f['general']['subject']['subject_id'][()]
if isinstance(subject, bytes):
    subject = subject.decode()
```

```python
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The notes say subject IDs should come from NWB subject metadata and be represented as unique mice plus a per-session index.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. A session is processed independently by `process_session`, then appended if it returns non-`None`. One session is dropped because it has zero `good` units.

ii. 
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
sessions = []
for p in files:
    info = process_session(p, edges, bin_centers, show_processing=args.show_processing)
    if info is not None:
        sessions.append(info)
```

```python
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
```

iii. The notes emphasize that the dataset is “organized under `data/` by subject folders” with “each session a single NWB file.” The agent also documented the 174-vs-173 discrepancy and expected one session to be excluded by curation; in practice the code excludes the single zero-good-unit session.

## 1-d. How are the data split into trials?

i. Trials are split by rows of `intervals/trials`. The script uses `len(trial['start_time'])` as the trial count, asserts that `go_start_times` has the same length, and loops once per trial index.

ii. 
```python
trial = load_trial_table(f)
...
n_trials = len(trial['start_time'])
assert len(go_times) == n_trials, (len(go_times), n_trials)
...
for i in range(n_trials):
    go = float(go_times[i])
```

iii. The notes say trial metadata live in `intervals/trials` and that behavioral alignment should use event timestamps with go cue as the anchor.

## 1-e. How are trials filtered based on quality controls?

i. There is no paper-style “regular trial” mask. Trials are only skipped if the go-aligned window would start before time 0, if `is_good_trials` exists and all good units are invalid for that trial, or if the window slice would fall outside the pre-binned session array. Entire sessions are dropped if they end up with fewer than 2 kept trials or no finite tongue data.

ii. 
```python
if start < 0:
    continue

if is_good_trials is not None:
    valid_units = is_good_trials[good_idx, i]
    if not np.any(valid_units):
        continue

start_idx = int(go_bin_start[i])
end_idx = start_idx + len(bin_centers)
if start_idx < 0 or end_idx > global_rates.shape[1]:
    continue
```

```python
if len(session_trial_neural) < 2:
    return None
...
if len(all_y) == 0:
    return None
```

iii. In `CONVERSION_NOTES.md`, the agent explicitly notes that reference code uses `get_regular_trial_mask` to exclude early lick, auto-water, free-water, no-response, and stimulation trials, but decides to keep a broader trial set because photostimulation must remain available as an input variable for this decoder task.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB unit table fields `units/classification`, `units/spike_times`, and `units/spike_times_index`. Brain-region metadata are derived from `units/anno_name`, with fallback to `units/electrode_group`.

ii. 
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
spike_times = f['units']['spike_times'][()]
spike_index = f['units']['spike_times_index'][()]
region_names = infer_brain_region_names(f, good_mask)
```

iii. The notes say the paper’s curated dataset should use only units labeled `good` by the QC classifier, and the trajectory shows the agent checked that `classification` contains `good` and `unlabelled`.

## 2-b. How is the `neural` data processed?

i. The code bins all good-unit spike times into 50 ms bins over the whole session once, converts counts to firing rates by dividing by bin width, then slices each trial’s go-aligned `[-2.5, 1.5]` s window from the session-wide array.

ii. 
```python
bin_size = edges[1] - edges[0]
session_t0 = float(np.min(go_times) + edges[0])
session_t1 = float(np.max(go_times) + edges[-1])
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
```

```python
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The notes say reference session dicts already appear to contain pre-binned firing-rate-like neural arrays, and the trajectory shows the agent later optimized from trial-by-trial histograms to this session-wide pre-binning approach because it was much faster.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The main QC filter is `classification == 'good'`. In addition, if `units/is_good_trials` exists and its second dimension matches the number of behavioral trials, the code zeroes invalid units on a per-trial basis and can skip a trial if no good units remain valid.

ii. 
```python
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
...
is_good_trials = np.asarray(f['units']['is_good_trials'][()], dtype=bool) \
    if ('is_good_trials' in f['units'] and f['units']['is_good_trials'].shape[1] == len(trial['start_time'])) else None
```

```python
if is_good_trials is not None:
    valid_units = is_good_trials[good_idx, i]
    if not np.any(valid_units):
        continue
...
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0
```

iii. The notes justify using only `good` units from the paper’s QC classifier. The per-trial `is_good_trials` logic was added later during debugging after decoder warnings about all-zero trials; the trajectory shows this was an ad hoc attempt to respect raw validity flags.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `go_start_times`, with go cue treated as time 0. Each trial extracts a `[-2.5, +1.5]` second window around that event.

ii. 
```python
go_times = load_event_times(f, 'go_start_times')
...
pre = 2.5
post = 1.5
bin_size = 0.05
edges, bin_centers = build_edges(pre, post, bin_size)
```

```python
go = float(go_times[i])
start = go + edges[0]
stop = go + edges[-1]
...
go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The instructions required go-cue alignment, and the notes explicitly say behavioral alignment should use `go_start_times` with a `[-2.5, +1.5]` s window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins. The neural spikes are binned directly at 50 ms, and no second-stage temporal rebinning is applied.

ii. 
```python
pre = 2.5
post = 1.5
bin_size = 0.05
edges, bin_centers = build_edges(pre, post, bin_size)
```

```python
'metadata': {
    ...
    'time_bin_size': 50.0,
    ...
}
```

iii. The choice comes directly from the task instructions and is repeated in the notes as part of the planned mapping.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times/timestamps`, plus the trial `start_time` and `stop_time` used to assign one sample event to each trial, and `go_start_times` used to convert that sample time into a go-relative offset.

ii. 
```python
sample_event_times = load_event_times(f, 'sample_start_times')
...
sample_times = np.full(n_trials, np.nan, dtype=float)
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) &
                              (sample_event_times <= trial['stop_time'][i])]
    if len(hits):
        sample_times[i] = hits[0]
```

iii. In the notes, the agent says `sample_start_times` “likely” corresponds to the task’s tone/sample onset. The trajectory shows this was chosen after discovering `sample_start_times` does not have exactly one entry per trial and needed per-trial assignment.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the first `sample_start_times` timestamp inside that trial is selected, converted to a go-relative offset `tone_rel = sample_time - go`, and then each neural bin center is transformed into “time from tone onset” by `bin_centers - tone_rel`.

ii. 
```python
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
```

```python
def build_time_from_tone(bin_centers, tone_time_rel):
    return bin_centers - tone_time_rel
```

iii. The trajectory shows the agent originally assumed one `sample_start_times` entry per trial, then changed to “first hit within trial” after finding 5,534 trials with multiple sample events. That is the operative justification.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same 80 go-aligned bin centers as the neural matrix, so the input is a time-varying vector with one value per neural time bin.

ii. 
```python
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
inp = np.stack([inp0, inp1], axis=0)
```

iii. The notes say the reference code uses a common `bin_centers` axis for trial-aligned data, so the agent chose to express tone timing on the same per-bin axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from the per-trial fields `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration`.

ii. 
```python
ps_on = parse_optional_float(trial['photostim_onset'][i]) if 'photostim_onset' in trial else np.nan
ps_dur = parse_optional_float(trial['photostim_duration'][i]) if 'photostim_duration' in trial else np.nan
```

iii. The notes say raw NWB stores photostimulation in trial timing fields plus event times, and the agent chose to build the decoder input from the raw trial timing fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Strings such as `N/A` are converted to `NaN`; otherwise onset and duration are parsed as floats. A per-bin binary vector is then created by setting bins in `[onset, onset + duration)` to 1.

ii. 
```python
def parse_optional_float(x):
    if isinstance(x, str):
        if x in ('N/A', 'nan', ''):
            return np.nan
        return float(x)
    return float(x)
```

```python
def build_photostim_vector(bin_centers, onset_rel, duration):
    x = np.zeros(bin_centers.shape[0], dtype=np.float32)
    ...
    off = onset_rel + duration
    x[(bin_centers >= onset_rel) & (bin_centers < off)] = 1.0
    return x
```

iii. The notes say photostimulation should become a time-varying binary input. The trajectory does not show an additional derivation beyond using the trial timing fields directly.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The intended alignment is “same bin centers as neural,” but the actual code does **not** convert `photostim_onset` from its trial-relative frame into the go-relative frame. It passes the raw onset directly to `build_photostim_vector`, so the resulting photostim input is all zeros in the converted dataset.

ii. 
```python
ps_on = parse_optional_float(trial['photostim_onset'][i]) if 'photostim_onset' in trial else np.nan
ps_dur = parse_optional_float(trial['photostim_duration'][i]) if 'photostim_duration' in trial else np.nan
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
```

iii. The notes say photostimulation should be “aligned to go cue,” but the code never subtracts go time or trial start. The trajectory shows the agent knew from the papers that photostim ends before go cue, but did not implement the needed coordinate conversion.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. It is derived primarily from `left_lick_times/timestamps` and `right_lick_times/timestamps`, with fallback to `intervals/trials/trial_instruction` when there is no post-go lick.

ii. 
```python
choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1
```

iii. In the notes, the agent says choice should reflect “actual choice semantics” and planned to use lick events, with instruction as a fallback if needed.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The script finds the first left lick and first right lick between go cue and trial stop, then chooses whichever happened first (`left=0`, `right=1`). If neither side licks after go, it imputes the instructed side from `trial_instruction`.

ii. 
```python
def find_choice_from_licks(left_licks, right_licks, go_time, stop_time):
    l = left_licks[(left_licks >= go_time) & (left_licks <= stop_time)]
    r = right_licks[(right_licks >= go_time) & (right_licks <= stop_time)]
    tl = l[0] if len(l) else np.inf
    tr = r[0] if len(r) else np.inf
    if tl == np.inf and tr == np.inf:
        return None
    return 0 if tl < tr else 1
```

iii. The notes/trajectory show the agent wanted a categorical choice on every trial and used the instruction fallback to avoid missing labels, even though the reference code’s regular-trial mask excludes no-response trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is derived directly from `intervals/trials/outcome`.

ii. 
```python
outcome = out_outcome_map[str(trial['outcome'][i])]
```

iii. The notes explicitly say to use the NWB trial `outcome` strings `ignore`, `miss`, and `hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code performs a direct categorical mapping: `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeats that value across all 80 time bins for the trial.

ii. 
```python
out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = out_outcome_map[str(trial['outcome'][i])]
...
out[1, :] = outcome
```

iii. The notes say this should be a direct mapping from the NWB string labels to the decoder categories.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is derived directly from `intervals/trials/early_lick`.

ii. 
```python
early = out_early_map[str(trial['early_lick'][i])]
```

iii. The notes explicitly identify the NWB `early_lick` field as the source variable.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `no early -> 0` and `early -> 1`, then repeats that categorical value across all 80 bins of the trial.

ii. 
```python
out_early_map = {'no early': 0, 'early': 1}
early = out_early_map[str(trial['early_lick'][i])]
...
out[2, :] = early
```

iii. The notes justify this as a direct categorical conversion from the NWB strings.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` and its `timestamps`. The script assumes column 1 of the `data` matrix is the tongue y-coordinate.

ii. 
```python
def load_tongue(f):
    grp = f['acquisition/BehavioralTimeSeries']['Camera0_side_TongueTracking']
    data = np.asarray(grp['data'][()], dtype=float)
    ts = np.asarray(grp['timestamps'][()], dtype=float)
    return ts, data
```

```python
def choose_tongue_y_column(data):
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError('Unexpected tongue tracking shape')
    return 1
```

iii. The notes say tongue position should come from the side-view tongue tracking stream and that the y-position likely corresponds to the second coordinate column.

## 8-b. How is `output` *Tongue y-position* processed?

i. The tongue y samples within each go-aligned trial window are averaged inside the same 50 ms bins used for neural data. No interpolation or likelihood filtering is applied. After all trials in a session are binned, session-wide 40th and 60th percentiles are computed on finite values.

ii. 
```python
mask = (tongue_ts >= start) & (tongue_ts < stop)
yt = tongue_y[mask]
tt = tongue_ts[mask] - go
binned_y = np.full(len(bin_centers), np.nan, dtype=np.float32)
if len(tt):
    inds = np.digitize(tt, edges) - 1
    ok = (inds >= 0) & (inds < len(bin_centers)) & np.isfinite(yt)
    if np.any(ok):
        sums = np.zeros(len(bin_centers), dtype=np.float64)
        cnts = np.zeros(len(bin_centers), dtype=np.int64)
        np.add.at(sums, inds[ok], yt[ok])
        np.add.at(cnts, inds[ok], 1)
        nz = cnts > 0
        binned_y[nz] = (sums[nz] / cnts[nz]).astype(np.float32)
```

```python
all_y = np.concatenate([x[np.isfinite(x)] for x in all_binned_y if np.any(np.isfinite(x))]) \
    if any(np.any(np.isfinite(x)) for x in all_binned_y) else np.array([], dtype=float)
q40, q60 = np.percentile(all_y, [40, 60])
```

iii. The notes say tongue y should be discretized by session percentiles after binning to the decoder time base; the code’s simple within-bin averaging is the agent’s concrete implementation of that plan.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Finite binned y-values below the session 40th percentile are category 0, values above the 60th percentile are category 2, and values between the thresholds are category 1.

ii. 
```python
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
finite = np.isfinite(binned_y)
ycat[finite & (binned_y < q40)] = 0
ycat[finite & (binned_y > q60)] = 2
ycat[finite & (binned_y >= q40) & (binned_y <= q60)] = 1
```

iii. This matches the explicit percentile-thresholding rule the agent copied into `CONVERSION_NOTES.md` from the task instructions.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue trace is aligned to go cue by selecting tongue samples between `go-2.5` and `go+1.5`, subtracting `go` from the timestamps, and digitizing the result into the same `edges` used for the neural matrix.

ii. 
```python
start = go + edges[0]
stop = go + edges[-1]
...
mask = (tongue_ts >= start) & (tongue_ts < stop)
tt = tongue_ts[mask] - go
inds = np.digitize(tt, edges) - 1
```

iii. The notes say all streams should share go-cue alignment and the common decoder time axis.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missing or inconsistent data with several ad hoc rules:
   - `photostim_onset` / `photostim_duration` strings like `N/A` become `NaN`, which produces an all-zero photostim vector.
   - If `sample_start_times` has multiple hits within a trial, the first one is used.
   - If a trial has no sample event, `tone_rel` becomes `NaN`, so the time-from-tone vector becomes `NaN`.
   - If a trial has no post-go lick, choice falls back to the instructed side.
   - If a tongue bin has no samples, it stays `NaN` until categorization, then defaults to category 1 because `ycat` is initialized to ones.
   - If `is_good_trials` has a mismatched trial dimension, it is ignored entirely.
   - Sessions with zero good units, fewer than 2 kept trials, or no finite tongue values are dropped.

ii. 
```python
if x in ('N/A', 'nan', ''):
    return np.nan
...
if len(hits):
    sample_times[i] = hits[0]
...
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
```

```python
choice = find_choice_from_licks(...)
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1
...
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
finite = np.isfinite(binned_y)
```

iii. The trajectory shows these rules mostly emerged during debugging: the agent discovered extra sample events, mismatched `is_good_trials`, and decoder warnings, then added guards rather than re-deriving the paper pipeline more faithfully.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is the per-session loop over all good units that histograms spikes into `global_rates`. Secondary costs are the per-trial tongue binning loop and the large pickle write. During development, the agent identified the original per-trial/per-unit histogram loop as too slow and replaced it with session-wide pre-binning.

ii. 
```python
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
```

```python
for i in range(n_trials):
    ...
    if len(tt):
        inds = np.digitize(tt, edges) - 1
        ...
        np.add.at(sums, inds[ok], yt[ok])
```

iii. `CONVERSION_NOTES.md` says the “current implementation loops over good units within each trial and may be slow,” then later notes the session-wide optimization reduced runtime to about 1.5 s/session on sample data.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization candidates are:
   - the loop assigning one `sample_start_times` event to each trial;
   - the per-trial left/right lick filtering in `find_choice_from_licks`;
   - the per-trial tongue masking/digitization/accumulation;
   - the Python loop in `decode_arr`.
   The unit histogram loop is harder to eliminate completely but could be replaced with more specialized spike-binning code.

ii. 
```python
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) &
                              (sample_event_times <= trial['stop_time'][i])]
```

```python
for i in range(n_trials):
    ...
    choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
    ...
    if len(tt):
        inds = np.digitize(tt, edges) - 1
```

iii. The trajectory shows the agent was actively looking for speedups after the first implementation proved slow, but beyond session-wide pre-binning it did not pursue further vectorization.

## 10-c. What processing does the code repeat multiple times?

i. Repeated work includes:
   - repeatedly filtering event arrays against each trial window (`sample_start_times`, left licks, right licks, tongue timestamps);
   - repeatedly reconstructing fixed per-trial outputs across all 80 bins for choice/outcome/early lick;
   - repeated finite checks when collecting `all_y` for percentile computation.

ii. 
```python
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) &
                              (sample_event_times <= trial['stop_time'][i])]
```

```python
choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
...
out = np.zeros((4, len(bin_centers)), dtype=np.int64)
out[0, :] = choice
out[1, :] = outcome
out[2, :] = early
```

iii. There is no explicit written justification for these repetitions; they are a straightforward consequence of the trial-by-trial implementation style seen in the script and trajectory.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Some work is computed or stored without downstream benefit:
   - `valid_trial_ids` is accumulated but never used;
   - `global_centers`, `n_time`, `math`, and `os` are unused;
   - `maybe_plot` and `--show-processing` generate plots only for manual inspection;
   - constant trial labels (choice/outcome/early lick) are expanded to full 80-bin vectors even though they are trial-level variables;
   - the broken photostim construction does work to build vectors that end up all zero.

ii. 
```python
n_time = len(bin_centers)
...
valid_trial_ids = []
...
valid_trial_ids.append(i)
```

```python
out = np.zeros((4, len(bin_centers)), dtype=np.int64)
out[0, :] = choice
out[1, :] = outcome
out[2, :] = early
```

iii. The notes mention diagnostic plots and speed concerns, but do not explicitly justify these extra computations; they appear to be convenience or debugging artifacts of the implementation.
