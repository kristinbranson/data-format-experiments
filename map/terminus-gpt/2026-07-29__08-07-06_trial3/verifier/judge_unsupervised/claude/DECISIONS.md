# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB (HDF5) files organized under `data/sub-<subject_id>/` directories. Each session is a single `.nwb` file. The script uses `h5py` to open each file and reads the trial table from `intervals/trials`, spike data from `units/spike_times` and `units/spike_times_index`, behavioral events from `acquisition/BehavioralEvents`, and tongue tracking from `acquisition/BehavioralTimeSeries`. Files are discovered by globbing `data/sub-*/*.nwb` and processed sequentially.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
if args.sample:
    files = files[:2]
sessions = []
for p in files:
    info = process_session(p, edges, bin_centers, show_processing=args.show_processing)
    if info is not None:
        sessions.append(info)
```

```python
def process_session(path, edges, bin_centers, show_processing=False):
    t0 = time.time()
    with h5py.File(path, 'r') as f:
        trial = load_trial_table(f)
        go_times = load_event_times(f, 'go_start_times')
        sample_event_times = load_event_times(f, 'sample_start_times')
        left_licks = load_event_times(f, 'left_lick_times')
        right_licks = load_event_times(f, 'right_lick_times')
        tongue_ts, tongue_data = load_tongue(f)
```

iii. The AI's CONVERSION_NOTES.md (Step 2) documents that data are organized under `data/` by subject folders with NWB files. The trajectory shows the AI explored the NWB structure in steps 11-13, identifying trial metadata under `intervals/trials`, behavioral streams under `acquisition/BehavioralEvents` and `acquisition/BehavioralTimeSeries`, and spike-sorted data in the `units` table.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `general/subject/subject_id` field in each NWB file. After processing all sessions, unique subjects are collected and sorted, and a mapping from subject to index is created.

ii.
```python
subject = f['general']['subject']['subject_id'][()]
if isinstance(subject, bytes):
    subject = subject.decode()
```

```python
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

```python
'subjects': subjects,
'subject_idx': np.array([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The AI documented in CONVERSION_NOTES Step 5 that subject IDs come from NWB metadata. The final output has 28 subjects matching the papers.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed independently. Sessions are excluded if they have zero good units or fewer than 2 valid trials after filtering. The final dataset has 173 sessions (out of 174 NWB files).

ii.
```python
if len(good_idx) == 0:
    return None
```
```python
if len(session_trial_neural) < 2:
    return None
```

iii. CONVERSION_NOTES Step 4 documents the 174 vs 173 session discrepancy: the paper reports 173 behavioral sessions, and one session is excluded during conversion due to having zero good units (no units with `classification == 'good'`).

## 1-d. How are the data split into trials?

i. Trials are defined by the `intervals/trials` table in each NWB file. The code iterates over all trials (indexed by `start_time`, `stop_time` from the trial table) and pairs each with its corresponding Go cue event from `go_start_times`. The code asserts that the number of go cue events matches the number of trials.

ii.
```python
n_trials = len(trial['start_time'])
assert len(go_times) == n_trials, (len(go_times), n_trials)
```

```python
for i in range(n_trials):
    go = float(go_times[i])
    start = go + edges[0]
    stop = go + edges[-1]
    if start < 0:
        continue
```

iii. The AI identified trials through NWB trial table exploration and documented this in Steps 2 and 5 of CONVERSION_NOTES.

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT apply the reference code's regular-trial filtering (which excludes early lick, auto water, free water, no-response, and photostimulation trials). Instead, all trials are included, with only minimal filtering: (1) trials where `go_time + edges[0] < 0` are skipped (i.e., go cue too early in the recording), (2) trials where the spike-binning window falls outside the pre-computed global rate array are skipped, and (3) if `is_good_trials` exists and has the right dimensions, trials where no good units are valid are skipped.

ii.
```python
if start < 0:
    continue

if is_good_trials is not None:
    valid_units = is_good_trials[good_idx, i]
    if not np.any(valid_units):
        continue
```

```python
start_idx = int(go_bin_start[i])
end_idx = start_idx + len(bin_centers)
if start_idx < 0 or end_idx > global_rates.shape[1]:
    continue
```

iii. The AI's CONVERSION_NOTES Step 4 notes that the reference code's `get_regular_trial_mask` excludes early lick, auto water, free water, no-response, and stimulation trials, but in Step 5, the AI decided to keep a broader trial set since the decoder task includes photostimulation as an input variable. No explicit filtering for auto_water, free_water, or no-response trials is applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged array of spike times per unit), `units/spike_times_index` (index into the ragged array), and `units/classification` (used to select only "good" units).

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
spike_times = f['units']['spike_times'][()]
spike_index = f['units']['spike_times_index'][()]
```

iii. CONVERSION_NOTES Steps 3-5 document that the paper uses region-specific QC classifiers to identify "good" units (69,943 out of 272,227 total), and the `units/classification` field stores this label.

## 2-b. How is the `neural` data processed?

i. Spike times for each good unit are binned into 50 ms time bins across a global session timeline. Spike counts are divided by bin size (0.05 s) to produce firing rates in Hz. For each trial, a window of 80 bins (4.0 s total: -2.5 s to +1.5 s relative to Go cue) is extracted from the pre-computed global rate matrix.

ii.
```python
bin_size = edges[1] - edges[0]
session_t0 = float(np.min(go_times) + edges[0])
session_t1 = float(np.max(go_times) + edges[-1])
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
global_centers = (global_edges[:-1] + global_edges[1:]) / 2
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
```

```python
go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
# ...
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The AI decided to use firing rates (counts/bin_size) rather than raw counts, and used a global binning strategy for efficiency. CONVERSION_NOTES Step 6 documents the vectorized approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are included. Additionally, if `is_good_trials` is available and has matching dimensions, units marked as not valid for a particular trial have their firing rates zeroed out (set to 0.0) rather than being excluded.

ii.
```python
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
```

```python
is_good_trials = np.asarray(f['units']['is_good_trials'][()], dtype=bool) if ('is_good_trials' in f['units'] and f['units']['is_good_trials'].shape[1] == len(trial['start_time'])) else None
```

```python
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0
```

iii. CONVERSION_NOTES Steps 3-4 document the neuron curation decision to use `classification == 'good'` to match the paper's curated dataset of 69,943 good units. The `is_good_trials` handling was added after debugging all-zero neural warnings during Step 7/Step 10.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the Go cue onset (`go_start_times`). For each trial, the go cue time is used as t=0, and a window from -2.5 s to +1.5 s is extracted. The global binning grid is computed relative to the earliest and latest go cue times in the session, and each trial's starting bin is computed as `np.rint((go_time + edges[0] - session_t0) / bin_size)`.

ii.
```python
pre = 2.5
post = 1.5
bin_size = 0.05
edges, bin_centers = build_edges(pre, post, bin_size)
```

```python
go_times = load_event_times(f, 'go_start_times')
# ...
go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
```

iii. CONVERSION_NOTES Steps 4-5 document the decision to align to Go cue onset with the specified time window. This matches the instructions' requirement to align to "Go cue onset" with "2.5 s before to 1.5 s after".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50 ms (0.05 s), resulting in 80 time bins per trial over the 4.0 s window. Spike counts are computed by histogramming spike times into these bins directly from raw spike times -- no intermediate binning or rebinning is applied.

ii.
```python
pre = 2.5
post = 1.5
bin_size = 0.05
edges, bin_centers = build_edges(pre, post, bin_size)
```

```python
def build_edges(pre, post, bin_size):
    n_bins = int(round((pre + post) / bin_size))
    edges = np.linspace(-pre, post, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers
```

iii. The instructions specify 50 ms bins, and the AI implemented this directly. Metadata records `time_bin_size: 50.0`.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The tone onset is derived from `acquisition/BehavioralEvents/sample_start_times`. The AI matches each sample event to a trial by checking which sample events fall within a trial's `start_time` to `stop_time` interval. The time from tone onset is then computed as `bin_center - (sample_time - go_time)`.

ii.
```python
sample_event_times = load_event_times(f, 'sample_start_times')
```

```python
sample_times = np.full(n_trials, np.nan, dtype=float)
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
    if len(hits):
        sample_times[i] = hits[0]
```

iii. The AI's trajectory (step 27) documents discovering that `sample_start_times` is not one-per-trial (405 events vs 368 trials in one session), necessitating a per-trial matching approach. CONVERSION_NOTES Step 5 identifies `sample_start_times` as the tone/auditory cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the tone onset time relative to the Go cue is computed as `sample_time - go_time`. Then, for each time bin, the time from tone onset is computed as `bin_center - tone_relative_time`. This produces a continuous, linearly increasing time series.

ii.
```python
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
```

```python
def build_time_from_tone(bin_centers, tone_time_rel):
    return bin_centers - tone_time_rel
```

iii. The AI chose to represent time from tone onset as a continuous value in seconds, which matches the instructions' specification of "Time from tone onset in seconds (continuous, time-varying)".

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both share the same `bin_centers` time axis (80 bins from -2.5 to +1.5 s relative to Go cue). The tone onset time is subtracted from each bin center, so the time-from-tone-onset at each bin corresponds exactly to the neural data at that bin.

ii.
```python
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
# bin_centers are relative to go cue
# tone_rel = sample_time - go_time (relative to go cue)
# result = bin_center - tone_rel = time from tone onset
```

iii. By using the same bin_centers array for both neural binning and input construction, the AI ensures temporal alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration` from the trial table.

ii.
```python
ps_on = parse_optional_float(trial['photostim_onset'][i]) if 'photostim_onset' in trial else np.nan
ps_dur = parse_optional_float(trial['photostim_duration'][i]) if 'photostim_duration' in trial else np.nan
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
```

iii. CONVERSION_NOTES Step 4 documents that photostimulation timing comes from `photostim_onset`/`photostim_duration` per trial and from `photostim_start_times`/`photostim_stop_times` event streams.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary vector is constructed over the bin_centers time axis. If the trial has valid (non-NaN) photostim_onset and photostim_duration, the bins where `bin_center >= onset` and `bin_center < onset + duration` are set to 1.0; all others are 0.0. For non-stimulation trials (NaN onset/duration), the entire vector is zeros.

ii.
```python
def build_photostim_vector(bin_centers, onset_rel, duration):
    x = np.zeros(bin_centers.shape[0], dtype=np.float32)
    if onset_rel is None or duration is None:
        return x
    if not np.isfinite(onset_rel) or not np.isfinite(duration):
        return x
    off = onset_rel + duration
    x[(bin_centers >= onset_rel) & (bin_centers < off)] = 1.0
    return x
```

iii. **CRITICAL ISSUE**: The function treats `photostim_onset` as if it were already relative to the Go cue, but the NWB trial table likely stores photostim_onset as either an absolute timestamp or relative to trial start, not relative to the Go cue. The verification output shows `photostimulation_on: [0.0, 0.0]` for ALL 173 sessions, meaning photostimulation is never detected as active. This is inconsistent with the paper's statement that ~25% of trials have photostimulation. The onset value is likely an absolute time that doesn't fall within the [-2.5, 1.5] bin_centers range, so the comparison `bin_centers >= onset_rel` never triggers.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation vector uses the same `bin_centers` as the neural data, so they are aligned on the same time grid. However, as noted above, the photostimulation is always zero due to a likely bug in converting the onset time to Go-cue-relative coordinates.

ii. See 4-b code.

iii. The AI's CONVERSION_NOTES Step 7 notes that "photostimulation input all zeros, likely because these particular sessions/trials lacked stimulation; this should be checked on broader data." However, the full conversion also shows all zeros, and the AI did not resolve this issue.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `acquisition/BehavioralEvents/left_lick_times` and `right_lick_times` event streams. If no licks are found in the trial window, the code falls back to `intervals/trials/trial_instruction` (left/right).

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

```python
choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1
```

iii. The AI opted to determine choice from actual lick behavior (first lick direction after Go cue) rather than instructed direction, with a fallback to instructed direction for no-response trials. CONVERSION_NOTES Step 5 notes: "Need to determine whether choice should reflect instructed side or actual lick direction; prefer actual choice semantics."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The first left and right lick times after the Go cue (up to trial stop_time) are compared. Whichever comes first determines the choice (left=0, right=1). If neither lick occurs, the instructed direction from `trial_instruction` is used as fallback. The choice is per-trial (constant across all time bins).

ii. See 5-a code snippets.

iii. The AI documented this mapping in CONVERSION_NOTES Step 5.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `intervals/trials/outcome` field, which contains string values `'ignore'`, `'miss'`, or `'hit'`.

ii.
```python
out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = out_outcome_map[str(trial['outcome'][i])]
```

iii. CONVERSION_NOTES Step 5 documents the direct mapping from NWB string values to integers.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A direct string-to-integer mapping: `ignore` -> 0, `miss` -> 1, `hit` -> 2. The outcome is per-trial (constant across all time bins).

ii. See 6-a code.

iii. This matches the instructions' specification: "Outcome (ignore = 0, miss = 1, hit = 2, per-trial)".

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from `intervals/trials/early_lick`, which contains string values `'no early'` or `'early'`.

ii.
```python
out_early_map = {'no early': 0, 'early': 1}
early = out_early_map[str(trial['early_lick'][i])]
```

iii. CONVERSION_NOTES Step 5 documents this direct mapping.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A direct string-to-integer mapping: `'no early'` -> 0, `'early'` -> 1. The value is per-trial (constant across all time bins).

ii. See 7-a code.

iii. This matches the instructions: "Early lick (no = 0, yes = 1, per-trial)".

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically the `data` array (column index 1, assumed to be y-position) and corresponding `timestamps`.

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

```python
ycol = choose_tongue_y_column(tongue_data)
tongue_y = tongue_data[:, ycol]
```

iii. CONVERSION_NOTES Step 5 notes: "Need to identify which column is y-position; likely second column if data are [x, y, likelihood]."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Tongue y-position values are binned into the same 50 ms time bins as the neural data. For each trial, tongue tracking data within the trial window is assigned to bins using `np.digitize`. Multiple values within the same bin are averaged. NaN is assigned to bins with no tongue data.

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

iii. The AI implemented temporal binning of tongue tracking data using vectorized numpy operations. The approach is reasonable for mapping 300 Hz video data to 50 ms bins.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After computing binned tongue y-position for all valid trials in a session, the 40th and 60th percentiles are computed from all finite y-values across all trials in the session. Each time bin is then discretized: values < 40th percentile -> 0, values between 40th-60th percentile -> 1, values > 60th percentile -> 2. NaN values (no tongue data) default to category 1 (middle).

ii.
```python
all_y = np.concatenate([x[np.isfinite(x)] for x in all_binned_y if np.any(np.isfinite(x))]) if any(np.any(np.isfinite(x)) for x in all_binned_y) else np.array([], dtype=float)
if len(all_y) == 0:
    return None
q40, q60 = np.percentile(all_y, [40, 60])

final_outputs = []
for choice, outcome, early, binned_y in session_trial_output:
    ycat = np.full(len(bin_centers), 1, dtype=np.int64)
    finite = np.isfinite(binned_y)
    ycat[finite & (binned_y < q40)] = 0
    ycat[finite & (binned_y > q60)] = 2
    ycat[finite & (binned_y >= q40) & (binned_y <= q60)] = 1
```

iii. This matches the instructions: "0: < 40th percentile, 1: 40th to 60th percentile, 2: > 60th percentile" with per-session discretization. The AI handles the edge case of NaN values by defaulting to category 1.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is aligned using the same bin edges as the neural data (derived from Go cue onset). The tongue tracking timestamps are converted to Go-cue-relative time by subtracting the go cue time, then digitized into the same bins.

ii.
```python
tt = tongue_ts[mask] - go
# ...
inds = np.digitize(tt, edges) - 1
```

iii. By using the same `edges` array for both neural and tongue data, temporal alignment is maintained.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches:
- **Missing sample/tone times**: If no sample event falls within a trial's time window, `sample_times[i]` remains NaN, and `time_from_tone_onset` becomes NaN for all bins of that trial.
- **Missing photostim**: If `photostim_onset` or `photostim_duration` is 'N/A' or NaN, the photostim vector is all zeros.
- **Missing tongue data**: If no tongue tracking data exists within a trial window, tongue y is NaN for all bins, which then defaults to category 1 in discretization.
- **is_good_trials dimension mismatch**: If `is_good_trials` doesn't have the right number of columns (trials), it's set to None and all good units are considered valid.
- **Zero good units**: Sessions with no good units return None and are excluded.
- **No licks in trial**: If neither left nor right licks occur between go cue and trial stop, choice falls back to instructed direction.

ii.
```python
# Missing photostim
ps_on = parse_optional_float(trial['photostim_onset'][i]) if 'photostim_onset' in trial else np.nan
# parse_optional_float handles 'N/A', 'nan', '' -> np.nan

# Missing tongue -> default category 1
ycat = np.full(len(bin_centers), 1, dtype=np.int64)

# is_good_trials dimension guard
is_good_trials = np.asarray(f['units']['is_good_trials'][()], dtype=bool) if ('is_good_trials' in f['units'] and f['units']['is_good_trials'].shape[1] == len(trial['start_time'])) else None
```

iii. The trajectory shows iterative debugging: step 39 uncovered the `is_good_trials` dimension mismatch, step 27 the sample_start_times count mismatch. The AI handled these robustly with guards.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning: histogramming all good units' spike times across the global session timeline. This is done once per session by looping over all good units and computing `np.histogram` for each. Processing times range from ~1 to ~6 seconds per session.

ii.
```python
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
```

iii. The AI identified this in CONVERSION_NOTES Step 7 and optimized from per-trial per-unit histograms to per-unit global histograms with trial slicing, reducing total conversion time from ~40+ minutes to ~6 minutes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops remain unvectorized:
1. The per-unit spike histogram loop (iterating `good_idx` to histogram each unit separately).
2. The sample_times assignment loop (iterating over trials to match sample events by interval).

ii.
```python
# Per-unit histogram loop
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size

# Sample time assignment loop
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
    if len(hits):
        sample_times[i] = hits[0]
```

iii. The per-unit loop is difficult to fully vectorize because spike times are stored as a ragged array (variable length per unit). The sample time matching loop could potentially be vectorized with `np.searchsorted`. The AI focused on the bigger optimization (global binning) rather than these smaller loops.

## 10-c. What processing does the code repeat multiple times?

i. The main trial loop computes tongue binning and input/output construction per trial, which is inherently necessary (each trial is unique). The global spike binning is done once per session, which is efficient. The code does not appear to repeat any significant processing unnecessarily within a session. However, `all_binned_y` stores the raw binned tongue y values and then iterates over them again for discretization, which involves two passes over trial data.

ii.
```python
# First pass: collect binned_y for each trial
all_binned_y.append(binned_y)
session_trial_output.append([choice, outcome, early, binned_y])

# Second pass: discretize after computing session percentiles
for choice, outcome, early, binned_y in session_trial_output:
    # ... discretize binned_y
```

iii. The two-pass approach is necessary because session-wide percentiles must be computed before discretization can occur.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several potentially unnecessary computations:
1. **All trials included**: The code includes early lick, auto water, free water, and no-response trials, which the reference code's `get_regular_trial_mask` excludes. These extra trials are processed but may hurt decoder performance.
2. **is_good_trials zeroing**: For trials where some units are marked invalid by `is_good_trials`, their firing rates are set to 0 rather than being excluded. This produces misleading "all neural data is zero" entries that pollute the dataset.
3. **Fallback choice for no-response trials**: Computing a choice for trials with no licks (falling back to instructed direction) creates potentially misleading labels.

ii.
```python
# No trial filtering for auto_water, free_water, etc.
# All trials processed unless they fail basic validity

# Zeroing invalid units rather than excluding
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0

# Fallback choice
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1
```

iii. The AI acknowledged the trial filtering discrepancy in CONVERSION_NOTES Step 4 but chose to include all trials to preserve photostimulation trial information for the decoder input.
