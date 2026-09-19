# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files organized under `data/sub-*/` directories. It uses `h5py` (not `pynwb`) to open each file directly and reads trial tables, event times, unit spike data, and tongue tracking data from the HDF5 structure. All sessions are discovered via a sorted glob of `sub-*/*.nwb`.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
...
for p in files:
    info = process_session(p, edges, bin_centers, show_processing=args.show_processing)
```

```python
with h5py.File(path, 'r') as f:
    trial = load_trial_table(f)
    go_times = load_event_times(f, 'go_start_times')
    ...
```

iii. The AI noted in CONVERSION_NOTES.md that "Data are organized under `data/` by subject folders named `sub-<subject_id>`" and "Each session is a single NWB file." Using `h5py` directly rather than `pynwb` is a valid alternative for reading NWB files.

## 1-b. How are the data split into subjects?

i. The AI reads `subject_id` from `general/subject/subject_id` in each NWB file. Unique subjects are collected and sorted, and each session is assigned an index into this list.

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

iii. The AI documented 28 unique subjects in the dataset, consistent with the reference papers.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. The AI uses the file stem as the session identifier (e.g., `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`). Sessions are ordered by the sorted file list.

ii.
```python
session_id = path.stem
```

iii. The AI documented 174 NWB files with 173 retained after filtering (one session dropped for having no good units). This matches the reference papers' 173 sessions.

## 1-d. How are the data split into trials?

i. Trials are read from the `intervals/trials` table in each NWB file. The AI loads all trial columns and asserts that the number of go cue events matches the number of trials.

ii.
```python
trial = load_trial_table(f)
go_times = load_event_times(f, 'go_start_times')
n_trials = len(trial['start_time'])
assert len(go_times) == n_trials, (len(go_times), n_trials)
```

iii. The AI correctly identifies trials from the NWB trials table and validates consistency with go cue event counts.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials in two ways: (1) it skips trials where `go + edges[0] < 0` (i.e., the trial window starts before time 0), and (2) it uses `is_good_trials` if available to check per-unit validity, skipping trials where no units are valid. The AI does NOT filter based on `obs_intervals` or `free_water`. This results in 94,364 retained trials.

ii.
```python
if start < 0:
    continue

if is_good_trials is not None:
    valid_units = is_good_trials[good_idx, i]
    if not np.any(valid_units):
        continue
```

iii. The AI's CONVERSION_NOTES.md mentions "Reference decoding code defines regular trials by excluding early lick, auto water, free water, no-response trials, and stimulation trials" but states they chose a "broader trial set." The AI does not explicitly justify why `obs_intervals` and `free_water` filtering were omitted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index`. Only units with `classification == 'good'` are included.

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
spike_times = f['units']['spike_times'][()]
spike_index = f['units']['spike_times_index'][()]
```

iii. The AI documented using only `classification == 'good'` units, consistent with the reference papers' 69,943 good units.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms bins using `np.histogram` on a global session-wide grid, then converted to firing rates by dividing by bin size. The per-trial neural data is extracted by slicing the global rate matrix. For trials with `is_good_trials` information, units marked invalid have their rates zeroed out.

ii.
```python
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size

go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
...
trial_mat = global_rates[:, start_idx:end_idx].copy()
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0
```

iii. The AI noted using histogram binning and conversion to firing rates. The global grid approach risks floating-point alignment issues at trial boundaries. Setting invalid units to 0.0 rather than excluding them is also a deviation from the reference approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are retained. Additionally, the AI attempts to use `is_good_trials` to zero out invalid units per trial rather than exclude them.

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
```

```python
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0
```

iii. The AI correctly uses the QC classifier (`classification == 'good'`). However, zeroing invalid units instead of excluding them means the neural matrix still includes those units with artificial zero values.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns to the go cue by building a global session-wide binning grid and computing the starting bin index for each trial relative to the go cue time. The trial window is -2.5s to +1.5s relative to the go cue.

ii.
```python
session_t0 = float(np.min(go_times) + edges[0])
session_t1 = float(np.max(go_times) + edges[-1])
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
...
start_idx = int(go_bin_start[i])
end_idx = start_idx + len(bin_centers)
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The go cue alignment is correct in intent. The global grid approach with `np.rint` rounding could introduce small temporal misalignment compared to the reference's per-trial edge approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins spanning -2.5s to +1.5s relative to the go cue, giving 80 time bins per trial. No rebinning is applied.

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

iii. This matches the instructions' specification of 50 ms bins and the -2.5s to +1.5s window.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (behavioral events) and `go_start_times`. The AI selects the FIRST `sample_start_time` that falls within the trial's `[start_time, stop_time]` window.

ii.
```python
sample_event_times = load_event_times(f, 'sample_start_times')
...
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
    if len(hits):
        sample_times[i] = hits[0]
```

iii. The AI selects the first sample start time within the trial window. The reference selects the last sample_start_time before the go cue, reasoning that early lick replays cause multiple sample onsets per trial and the last one is the relevant one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone time relative to the go cue is computed, then subtracted from the bin centers to get time from tone onset at each bin.

ii.
```python
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
```

```python
def build_time_from_tone(bin_centers, tone_time_rel):
    return bin_centers - tone_time_rel
```

iii. The formula `bin_centers - (tone - go)` is mathematically equivalent to `bin_centers + (go - tone)`, matching the reference formula. The issue is which tone onset is selected (first vs last).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both share the same bin centers derived from `build_edges(-2.5, 1.5, 0.05)`, so they are inherently aligned.

ii.
```python
edges, bin_centers = build_edges(pre, post, bin_size)
```

iii. Same bin grid is used for neural and input data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table.

ii.
```python
ps_on = parse_optional_float(trial['photostim_onset'][i]) if 'photostim_onset' in trial else np.nan
ps_dur = parse_optional_float(trial['photostim_duration'][i]) if 'photostim_duration' in trial else np.nan
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
```

iii. The AI identified the correct source variables.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI treats `photostim_onset` as already relative to the go cue and builds a binary vector where bin centers fall between onset and onset+duration. However, `photostim_onset` is actually stored relative to trial start, not the go cue. The AI does not convert from trial-start-relative to go-cue-relative coordinates. This results in all photostimulation values being zero in the output.

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

iii. The verification output confirms photostimulation input is [0.0, 0.0] for all 173 sessions, indicating the photostim computation is broken. The onset values (relative to trial start, typically positive ~1-2s) don't overlap with bin_centers (relative to go cue, range -2.475 to +1.475), so no bins are marked as stimulated.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The intent is to use the same bin centers, but due to the incorrect coordinate frame for the onset, the alignment is broken and all values are zero.

ii. See 4-b code.

iii. The reference correctly converts `photostim_onset` from trial-start-relative to go-cue-relative by computing `trials['start_time'] + photostim_onset - go_time`.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the actual lick event timestamps (`left_lick_times` and `right_lick_times`), finding the first lick after the go cue. If no lick is found, it falls back to the instructed side (`trial_instruction`).

ii.
```python
left_licks = load_event_times(f, 'left_lick_times')
right_licks = load_event_times(f, 'right_lick_times')
...
choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1
```

iii. The AI chose to use actual lick events rather than deriving choice from `trial_instruction` x `outcome`. The AI's CONVERSION_NOTES.md mentions wanting to "reconstruct decoder outputs consistent with reference semantics."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps choice to 0 (left) or 1 (right) only. There is NO "no lick" category. When no lick is found (ignore trials), the AI incorrectly falls back to the instructed side, assigning a lick direction even when the animal didn't lick. The output_values has only `['left', 'right']`.

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
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1
```

```python
'output_values': [
    ['left', 'right'],
    ...
]
```

iii. The instructions explicitly specify three choice values: "left, right, no lick". The AI omits the "no lick" category and fabricates a choice for ignore trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column in the trials table, which contains `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = out_outcome_map[str(trial['outcome'][i])]
```

iii. Direct mapping from the NWB trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three outcome strings are mapped to integers: ignore=0, miss=1, hit=2. The value is repeated across all 80 time bins.

ii.
```python
out[1, :] = outcome
```

iii. This matches the reference approach.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column in the trials table, containing `'no early'` and `'early'`.

ii.
```python
out_early_map = {'no early': 0, 'early': 1}
early = out_early_map[str(trial['early_lick'][i])]
```

iii. Direct mapping from the NWB trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no early) and 1 (early), repeated across all 80 time bins.

ii.
```python
out[2, :] = early
```

iii. Matches the reference approach.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically column 1 (y-position) of the data array, along with timestamps.

ii.
```python
grp = f['acquisition/BehavioralTimeSeries']['Camera0_side_TongueTracking']
data = np.asarray(grp['data'][()], dtype=float)
ts = np.asarray(grp['timestamps'][()], dtype=float)
...
tongue_y = tongue_data[:, ycol]  # ycol = 1
```

iii. Correct source variable identified.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI bins tongue y-values into 50ms bins per trial using `np.digitize` and computes mean values per bin. It does NOT filter by tongue tracking likelihood/confidence. Then session-level 40th/60th percentiles are computed from all valid (finite) binned values across trials. Bins are discretized into 3 categories (0, 1, 2) with no "not visible" category.

ii.
```python
# No likelihood filtering - all tongue_y values used
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
...
q40, q60 = np.percentile(all_y, [40, 60])
...
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
finite = np.isfinite(binned_y)
ycat[finite & (binned_y < q40)] = 0
ycat[finite & (binned_y > q60)] = 2
ycat[finite & (binned_y >= q40) & (binned_y <= q60)] = 1
```

iii. The AI does not filter by tongue likelihood, meaning tracker noise from frames where the tongue is not visible (retracted) contaminates the y-position values and the percentile computation. The reference filters frames with likelihood < 0.5.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses 3 categories only: 0 (< 40th percentile), 1 (40th-60th percentile), 2 (> 60th percentile). Non-finite bins default to category 1 (middle). There is NO category 3 for "not visible."

ii.
```python
ycat = np.full(len(bin_centers), 1, dtype=np.int64)  # default to middle category
finite = np.isfinite(binned_y)
ycat[finite & (binned_y < q40)] = 0
ycat[finite & (binned_y > q60)] = 2
```

```python
'output_values': [
    ...
    ['lt_40pct', '40to60pct', 'gt_60pct'],  # only 3 categories
]
```

iii. The instructions explicitly specify 4 categories including "3: not visible." The AI omits this category and assigns non-visible bins to category 1 by default.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses the same trial window (go-cue-relative) and bin edges for tongue data as for neural data, binning tongue timestamps into the same 80 bins.

ii.
```python
mask = (tongue_ts >= start) & (tongue_ts < stop)
tt = tongue_ts[mask] - go
inds = np.digitize(tt, edges) - 1
```

iii. Alignment is consistent with the neural data bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three approaches: (1) Sessions with no good units return None and are dropped. (2) The `is_good_trials` field (when present) is used to zero out invalid units rather than drop trials. (3) Missing tongue data bins get NaN which defaults to the middle tongue category. (4) Missing `photostim_onset` values ('N/A') are parsed to NaN, resulting in zero photostim vectors.

ii.
```python
if len(good_idx) == 0:
    return None

if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0

# NaN defaults to category 1 for tongue
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
```

iii. Setting invalid units to 0.0 rather than excluding them or the trial introduces artificial zero-firing signals. Defaulting missing tongue data to the middle category rather than a dedicated "not visible" category misrepresents the data.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file with h5py and building the global rate matrix via per-unit histogram calls dominate. The conversion output shows ~1.5-6s per session depending on unit count.

ii. N/A

iii. The AI noted "~1.5 s/session on sample" and estimated ~4.5 min total for 174 sessions.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit histogram loop could potentially be vectorized, but spike times are ragged. The per-trial loop for tongue binning could also be vectorized. The per-trial loop for sample time selection iterates over all trials individually.

ii.
```python
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
```

```python
for i in range(n_trials):
    hits = sample_event_times[...]
    ...
```

iii. The per-trial sample time loop is unnecessary and could use searchsorted. The main per-unit loop is inherent to ragged spike data.

## 10-c. What processing does the code repeat multiple times?

i. Nothing is obviously repeated. Each NWB file is opened once and all quantities are computed in a single pass.

ii. N/A

iii. No redundant processing identified.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `left_lick_times` and `right_lick_times` event streams to derive choice, which is more complex than necessary since choice can be derived from `trial_instruction` x `outcome`. The global rate matrix is computed for the entire session timeline, including inter-trial intervals that are never used.

ii.
```python
left_licks = load_event_times(f, 'left_lick_times')
right_licks = load_event_times(f, 'right_lick_times')
```

```python
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
```

iii. Computing firing rates for the full session timeline wastes memory and computation on inter-trial periods. Loading lick event data is unnecessary given the simpler instruction x outcome derivation.
