# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files using `h5py` (not `pynwb`). It globs for all NWB files under `data/sub-*/*.nwb`, sorts them, and processes each file sequentially. Each file is opened with `h5py.File`, and trial metadata, behavioral events, tongue tracking data, unit spike times, and quality labels are read from the HDF5 groups directly.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
if args.sample:
    files = files[:2]
# ...
for p in files:
    info = process_session(p, edges, bin_centers, show_processing=args.show_processing)
```

Within `process_session`:
```python
with h5py.File(path, 'r') as f:
    trial = load_trial_table(f)
    go_times = load_event_times(f, 'go_start_times')
    sample_event_times = load_event_times(f, 'sample_start_times')
    left_licks = load_event_times(f, 'left_lick_times')
    right_licks = load_event_times(f, 'right_lick_times')
    tongue_ts, tongue_data = load_tongue(f)
```

iii. The AI noted in CONVERSION_NOTES.md that the data are organized as one NWB file per session under `data/sub-<subject_id>/`. Using `h5py` directly instead of `pynwb` is a valid alternative approach to reading NWB files, since NWB is an HDF5-based format.

## 1-b. How are the data split into subjects?

i. Each NWB file's subject is read from `general/subject/subject_id`. Unique subjects are collected across all sessions, sorted, and mapped to integer indices.

ii.
```python
subject = f['general']['subject']['subject_id'][()]
if isinstance(subject, bytes):
    subject = subject.decode()
```

Assembly:
```python
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI identified that subject IDs are stored in the NWB general/subject group. This approach correctly extracts the numeric subject ID (e.g., '440956') from each file.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The session ID is taken from the file stem (e.g., `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`), not from `nwb.identifier`. Sessions are sorted by file path, giving chronological order within subjects.

ii.
```python
session_id = path.stem
```

iii. The AI's CONVERSION_NOTES.md documents that there are 174 NWB files and 28 subjects. The final output has 173 sessions (one dropped for having no good units), matching the paper's stated 173 sessions.

## 1-d. How are the data split into trials?

i. Trials come from the `intervals/trials` table in each NWB file, with one row per behavioral trial. The number of go-cue events is asserted to match the number of trials.

ii.
```python
trial = load_trial_table(f)
go_times = load_event_times(f, 'go_start_times')
n_trials = len(trial['start_time'])
assert len(go_times) == n_trials, (len(go_times), n_trials)
```

iii. The AI correctly identifies that trials are defined by the trials table and validates consistency with go-cue events.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials based on two criteria: (1) `start < 0` — trials whose go-cue-aligned window starts before time 0 are skipped; (2) `is_good_trials` — if this matrix exists in the NWB file, trials where no good unit has valid data are skipped. Importantly, the AI does NOT filter `free_water` trials and does NOT use `obs_intervals`. The AI also zeroes out neural data for units that are not valid on a given trial (rather than dropping the trial).

ii.
```python
if start < 0:
    continue

if is_good_trials is not None:
    valid_units = is_good_trials[good_idx, i]
    if not np.any(valid_units):
        continue
else:
    valid_units = np.ones(len(good_idx), dtype=bool)
```

And later:
```python
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0
```

iii. The AI's CONVERSION_NOTES.md acknowledges that the reference code's `get_regular_trial_mask` excludes early lick, auto water, free water, no-response, and stimulation trials, but the AI chose to keep a broader trial set. However, the AI did not filter `free_water` trials, which the reference specifically excludes. The verification output shows 94,364 total trials vs the reference's ~90,860 — the ~3,500 extra trials are largely free_water and out-of-obs-intervals trials that should have been excluded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index` for units where `units/classification == 'good'`. Go-cue times from `BehavioralEvents/go_start_times` are used for alignment.

ii.
```python
spike_times = f['units']['spike_times'][()]
spike_index = f['units']['spike_times_index'][()]
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
```

iii. The AI correctly identifies that spike times are the source for neural data and that only 'good' units should be used.

## 2-b. How is the `neural` data processed?

i. The AI pre-bins all good units across the entire session timeline into a global histogram grid, then slices trial windows from this global grid. Spike counts are divided by bin width to get firing rates in Hz. On trials where `is_good_trials` marks a unit as invalid, that unit's firing rates are zeroed out rather than the trial being dropped.

ii.
```python
session_t0 = float(np.min(go_times) + edges[0])
session_t1 = float(np.max(go_times) + edges[-1])
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
global_centers = (global_edges[:-1] + global_edges[1:]) / 2
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size

go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
# ...
trial_mat = global_rates[:, start_idx:end_idx].copy()
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0
```

iii. The global histogram approach is an optimization to avoid per-trial binning. However, using `np.rint` to align go-cue bins to the global grid can introduce small alignment errors (up to half a bin width). The reference uses per-trial exact bin edges computed directly from the go cue time, avoiding this issue.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept, same as the reference. However, the AI additionally uses `is_good_trials` to zero out invalid unit-trial combinations rather than filtering entire trials by `obs_intervals`.

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
```

iii. The AI correctly uses the QC classifier verdict (`classification == 'good'`). However, zeroing out units on invalid trials rather than dropping those trials produces different results from the reference: the verification output shows many trials with all-zero neural data in the final dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue. The bin edges span -2.5 s to +1.5 s relative to the go cue, with 50 ms bins, giving 80 time bins. The alignment is done via a global session timeline with `np.rint`-based index lookup.

ii.
```python
edges, bin_centers = build_edges(pre, post, bin_size)
# ...
go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
start_idx = int(go_bin_start[i])
end_idx = start_idx + len(bin_centers)
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The alignment event (go cue) and window (-2.5 to +1.5 s) match the instructions. The global-grid slicing approach is an approximation; the reference computes exact per-trial bin edges.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins are used, producing 80 time bins per trial. No additional rebinning is applied.

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

iii. The 50 ms bin width matches the instructions. The AI uses `np.linspace` to create edges, which is equivalent to the reference's `T_START + BIN * np.arange(N_BINS + 1)`.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `BehavioralEvents/sample_start_times` (the tone onset events) and `go_start_times` (for alignment). The AI finds the first `sample_start_times` event within each trial's `[start_time, stop_time]` window.

ii.
```python
sample_event_times = load_event_times(f, 'sample_start_times')
# ...
sample_times = np.full(n_trials, np.nan, dtype=float)
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
    if len(hits):
        sample_times[i] = hits[0]
```

iii. The AI identifies sample_start_times as the tone onset, which is correct. However, it takes the first event in the trial window rather than the last event before the go cue (as the reference does). For early-lick trials where the sample epoch replays, these differ.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as `bin_centers - tone_time_relative_to_go`, giving the elapsed time since tone onset at each bin center. This is a continuous, time-varying input.

ii.
```python
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
```

```python
def build_time_from_tone(bin_centers, tone_time_rel):
    return bin_centers - tone_time_rel
```

iii. The formula `bin_centers - (sample_time - go_time)` = `(t - go) - (sample - go)` = `t - sample` correctly computes time since tone onset. The result matches the reference formula `CENTERS + (go - tone)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers (go-cue-relative) are used for both neural data and the time-from-tone input, so they are aligned by construction.

ii. Both use the same `bin_centers` array from `build_edges`.

iii. Alignment is correct since the same time grid is shared.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from the trial table columns `photostim_onset` (seconds relative to trial start) and `photostim_duration`.

ii.
```python
ps_on = parse_optional_float(trial['photostim_onset'][i]) if 'photostim_onset' in trial else np.nan
ps_dur = parse_optional_float(trial['photostim_duration'][i]) if 'photostim_duration' in trial else np.nan
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
```

iii. The AI correctly identifies the source variables. However, `photostim_onset` is relative to trial start, not relative to the go cue.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary vector is constructed where bins with centers in `[onset, onset + duration)` are set to 1.0, and all others are 0.0. However, there is a critical bug: `photostim_onset` is relative to trial start (a positive number like 2-4 seconds), but it is compared directly against go-cue-relative bin centers (which range from -2.475 to 1.475). Since photostim ends before the go cue, the trial-start-relative onset will never overlap with the go-cue-relative bin centers.

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

iii. The verification output confirms `photostimulation_on: [0.0, 0.0]` — photostim is always zero across the entire dataset. The reference code correctly converts: `stim_on = trials['start_time'] + photostim_onset - go_time`, putting the onset into go-cue-relative coordinates before comparison.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The same bin centers are used, so the alignment framework is correct. However, because of the coordinate system bug described above, the photostim input is always zero.

ii. See 4-b.

iii. The alignment mechanism is correct but the actual values are wrong due to the missing coordinate transformation.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the actual lick events (`left_lick_times` and `right_lick_times`), finding which side was licked first after the go cue. If no lick is found, it falls back to the instructed side (`trial_instruction`).

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

# In process_session:
choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1
```

iii. The AI chose to determine choice from actual lick events rather than deriving it from instruction x outcome as the reference does.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as left=0, right=1 (two classes). When no lick is found (ignore trials), the AI falls back to the instructed side rather than creating a third "no lick" class. Choice is per-trial and repeated across all 80 bins.

ii.
```python
out = np.zeros((4, len(bin_centers)), dtype=np.int64)
out[0, :] = choice
```

Output values:
```python
'output_values': [
    ['left', 'right'],
    ...
]
```

iii. The instructions say "Lick direction choice (left = 0, right = 1, per-trial)" with only two values specified. The AI followed this literally but does not handle the no-lick case correctly — for ignore trials, it assigns the instructed side as the choice, which is misleading since the animal didn't actually lick. The reference creates a third class (2 = "no lick") for these trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which holds strings 'ignore', 'miss', 'hit'.

ii.
```python
out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = out_outcome_map[str(trial['outcome'][i])]
```

iii. Direct categorical mapping from the trials table, matching the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers (ignore=0, miss=1, hit=2) and repeated across all 80 bins as a per-trial value.

ii.
```python
out[1, :] = outcome
```

iii. The mapping matches the instructions and reference exactly.

## 6-c. How is `output` *Outcome* aligned with the neural data?

i. Per-trial value repeated across all bins. Same trial ordering as neural data, so alignment is by construction.

ii. N/A — single value per trial.

iii. Correct alignment.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'no early' and 'early'.

ii.
```python
out_early_map = {'no early': 0, 'early': 1}
early = out_early_map[str(trial['early_lick'][i])]
```

iii. Direct categorical mapping from the trials table, matching the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Two strings mapped to integers (no early=0, early=1). Per-trial value repeated across all 80 bins.

ii.
```python
out[2, :] = early
```

iii. Matches the instructions and reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically column 1 (y-position) of the `(n_frames, 3)` data array.

ii.
```python
def load_tongue(f):
    grp = f['acquisition/BehavioralTimeSeries']['Camera0_side_TongueTracking']
    data = np.asarray(grp['data'][()], dtype=float)
    ts = np.asarray(grp['timestamps'][()], dtype=float)
    return ts, data

ycol = choose_tongue_y_column(tongue_data)  # returns 1
tongue_y = tongue_data[:, ycol]
```

iii. Correctly identifies the tongue tracking data source and the y-position column.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI bins tongue y-values into the 50 ms bins using `np.digitize` and averages within each bin. It does NOT filter frames by tracking likelihood/confidence — all frames are used regardless of whether the tongue is actually visible. It then computes 40th and 60th percentiles over all finite binned values in the session.

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

Percentile computation (after all trials processed):
```python
all_y = np.concatenate([x[np.isfinite(x)] for x in all_binned_y ...])
q40, q60 = np.percentile(all_y, [40, 60])
```

iii. The AI does not apply a likelihood threshold to filter out frames where the tongue is not visible. The reference filters with `likelihood < 0.5` and sets those to NaN. Since ~90% of frames have the tongue retracted (likelihood < 0.01), including them shifts the percentile thresholds and mixes noise positions with real tongue protrusions.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories: 0 (below 40th percentile), 1 (40th-60th percentile), 2 (above 60th percentile). Non-finite values default to class 1. There is no "not visible" class.

ii.
```python
ycat = np.full(len(bin_centers), 1, dtype=np.int64)  # default to middle class
finite = np.isfinite(binned_y)
ycat[finite & (binned_y < q40)] = 0
ycat[finite & (binned_y > q60)] = 2
ycat[finite & (binned_y >= q40) & (binned_y <= q60)] = 1
```

Output values:
```python
['lt_40pct', '40to60pct', 'gt_60pct']
```

iii. The reference uses 4 categories with a "not visible" class (3) for bins where no tongue frame has sufficient tracking confidence. The AI assigns non-finite bins to class 1 (middle), which conflates missing data with genuine middle-range tongue positions. Given that ~75% of bins have no visible tongue, this significantly distorts the class distribution.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue data is binned into the same 50 ms go-cue-relative bins as the neural data, using the same edges. Alignment is by construction.

ii.
```python
mask = (tongue_ts >= start) & (tongue_ts < stop)
tt = tongue_ts[mask] - go
inds = np.digitize(tt, edges) - 1
```

iii. The alignment approach is correct — same time grid as neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with no good units are dropped (returns None). (2) Sessions with fewer than 2 valid trials are dropped. (3) For tongue data, NaN binned values are handled by defaulting to class 1 (middle class). The AI does NOT handle the case of sessions that were never quality-controlled (where `classification` might be NaN/non-string) — the `decode_arr` function converts non-string values to their string representation, which may not match 'good'.

ii.
```python
def decode_arr(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode())
        else:
            out.append(str(x))
    return np.array(out, dtype=object)
```

```python
if len(good_idx) == 0:
    return None
# ...
if len(session_trial_neural) < 2:
    return None
```

iii. The AI handles the main missing-data cases (no good units, insufficient trials) but does not explicitly handle NaN classifications — `str(nan)` becomes the string `'nan'`, which doesn't match 'good', so the session would correctly be dropped. For tongue data, defaulting NaN to class 1 is problematic since it hides missing data.

## 10-a. What are the most time-consuming steps of the code?

i. Per-unit histogram computation across the global session timeline, and reading large spike time arrays from the HDF5 files. The full conversion took approximately 4-5 minutes for 174 sessions (~1.5-6 s per session depending on neuron count).

ii. N/A — from conversion output timing.

iii. The AI's CONVERSION_NOTES.md estimates ~1.5 s/session and ~4.5 min total. The actual output confirms similar timings.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that processes each trial sequentially (lines 167-226) iterates over all trials doing per-trial tongue binning, input construction, and output assignment. The sample_times computation also loops over trials individually. These could be vectorized.

ii.
```python
for i in range(n_trials):
    # ... per-trial tongue binning, input/output construction
```

```python
sample_times = np.full(n_trials, np.nan, dtype=float)
for i in range(n_trials):
    hits = sample_event_times[...]
    if len(hits):
        sample_times[i] = hits[0]
```

iii. The reference vectorizes the sample time lookup using `np.searchsorted` and constructs all trial inputs/outputs in vectorized array operations.

## 10-c. What processing does the code repeat multiple times?

i. The code processes tongue tracking data twice per trial — once during the per-trial loop to compute binned y values, and then again in a second pass to discretize into categories after percentiles are computed. This two-pass approach is necessary because percentiles require seeing all data first.

ii.
```python
# First pass: compute binned_y
all_binned_y.append(binned_y)
# ...
# Second pass: discretize
for choice, outcome, early, binned_y in session_trial_output:
    ycat = np.full(len(bin_centers), 1, dtype=np.int64)
    ...
```

iii. The two-pass approach for tongue discretization is inherent to the per-session percentile computation. The reference avoids this by computing the session-wide percentiles in one pass over the global bin means, then discretizing trial data in a single loop.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `left_lick_times` and `right_lick_times` for choice derivation. While these are used, the approach is more complex than needed — the reference derives choice from `trial_instruction` x `outcome` without needing the raw lick events. The AI also stores `bin_centers_s` in metadata, which is not required by the target format.

ii.
```python
left_licks = load_event_times(f, 'left_lick_times')
right_licks = load_event_times(f, 'right_lick_times')
```

```python
'bin_centers_s': bin_centers.astype(np.float32),
```

iii. Loading lick events is unnecessary overhead since choice can be derived more simply. The extra metadata field is harmless but unnecessary.
