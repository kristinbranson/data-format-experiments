# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All NWB files are found with a glob pattern, and each file is opened with `h5py` (not `pynwb`). Subjects, trials, and units are then read from within each file. The AI uses multiprocessing (`Pool`) to scan and convert sessions in parallel.

ii. Finding all files:
```python
files = sorted(glob.glob(os.path.join(args.data_dir, '*', '*.nwb')))
```

Opening one session:
```python
with h5py.File(path, 'r') as f:
    tt = _trial_table(f)
    classification = f['units/classification'][:].astype(str)
    good = classification == 'good'
    region_names, has_region = unit_regions(f, good, region_map)
```

iii. The AI stated: "NWB (DANDI:000363) read directly with h5py; spot-checked rate bins against an independent histogram of raw spike times." The choice of h5py over pynwb was likely for performance, since the AI uses multiprocessing and h5py is lighter weight. The glob pattern `'*/*.nwb'` differs slightly from the reference's `'sub-*/*.nwb'` but captures the same files.

## 1-b. How are the data split into subjects?

i. Each NWB file records its animal in `general/subject/subject_id`. That value is read for every session and carried through to assembly, where `subjects` is the sorted set of unique ids and `subject_idx` gives each session's index into that list.

ii. Per session:
```python
subject = str(f['general/subject/subject_id'][()].decode())
```

At assembly:
```python
subjects = sorted({s['subject'] for s in sessions})
subject_idx = np.array([subjects.index(s['subject']) for s in sessions], dtype=np.int64)
```

iii. The AI uses the numeric `subject_id` from the NWB file, same as the reference.

## 1-c. How are the data split into sessions?

i. One NWB file is one session. However, the AI applies **session-selection criteria** from the data paper: behavioral performance > 65% and at least 50 correct lick-left and 50 correct lick-right trials. This results in 143 sessions (out of 174), compared to the reference's 173 sessions. Sessions with no good units are also rejected.

ii. Session scanning:
```python
def scan_session(path):
    with h5py.File(path, 'r') as f:
        tt = _trial_table(f)
        performance, n_left, n_right = session_performance(tt)
        ...
```

Session selection:
```python
if s['performance'] <= MIN_PERFORMANCE:
    reasons.append('performance %.3f' % s['performance'])
if min(s['n_correct_left'], s['n_correct_right']) < MIN_CORRECT_PER_DIRECTION:
    reasons.append('correct trials %d/%d' % (s['n_correct_left'], s['n_correct_right']))
if s['n_good_units'] == 0:
    reasons.append('no good units')
```

iii. The AI stated: "data-paper criteria applied — performance > 65% and >= 50 correct left and right trials (143/174 kept, 31 rejected and listed in metadata). Performance is hits/(hits+misses) on control, non-early-lick trials, which reproduces the paper's ~84%/65-99% range."

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioral trial. The go-cue times are read from `BehavioralEvents/go_start_times`. The trial table columns are read directly via h5py.

ii.
```python
def _trial_table(f):
    tr = f['intervals/trials']
    out = {
        'start_time': tr['start_time'][:],
        'stop_time': tr['stop_time'][:],
        'outcome': tr['outcome'][:].astype(str),
        ...
    }
    events = f['acquisition/BehavioralEvents']
    out['go_time'] = events['go_start_times']['timestamps'][:]
```

iii. Trials are defined by the trials table, consistent with the reference approach.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in several ways: (1) `auto_water` trials are dropped, (2) `free_water` trials are dropped, (3) trials outside `obs_intervals` (where electrophysiology was not recorded) are dropped, and (4) trials whose tone onset cannot be located are dropped. The `auto_water` filter is an addition compared to the reference. The `obs_intervals` check is more thorough: it intersects observed intervals across ALL good units, rather than using a single unit.

ii.
```python
def trial_mask(tt):
    return ((tt['auto_water'] == 0) & (tt['free_water'] == 0)
            & tt['observed'] & tt['tone_valid'])
```

The `observed_trials` function:
```python
def observed_trials(f, trial_start):
    ...
    observed = np.ones(len(trial_start), dtype=bool)
    for u in good_idx:
        interval_starts = obs[start[u]:stop[u], 0]
        ...
        observed &= unit_observed
    return observed
```

iii. The AI stated: "auto-water/free-water trials dropped (reward not choice-contingent), as in `get_regular_trial_mask`. Trials outside `units/obs_intervals` are dropped: in 9 sessions the ephys covers only part of the behavior file."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times`, the sorted spike times of each unit. Only units with `classification == 'good'` AND a valid CCF annotation (`anno_name`) contribute. The go-cue times are used to place bin edges.

ii.
```python
spike_times = units['spike_times'][:]
stop = units['spike_times_index'][:]
start = np.concatenate([[0], stop[:-1]])
```

iii. The AI uses the same underlying spike times as the reference. The additional CCF annotation requirement is a notable difference.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each good unit, bin edges for every trial are built as absolute times, `np.searchsorted` gives running spike counts, and differencing gives counts per bin. Counts are divided by bin width (0.05 s) to give Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
edges = go_time[:, None] + BIN_EDGES[None, :]
flat_edges = edges.ravel()
rates = np.zeros((n_trials, N_BINS, len(unit_ids)), dtype=np.float32)
for i, u in enumerate(unit_ids):
    st = spike_times[start[u]:stop[u]]
    counts = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
    rates[:, :, i] = np.diff(counts, axis=1) / BIN_WIDTH
```

iii. This is functionally identical to the reference approach. The output array shape is `(n_trials, N_BINS, n_units)` which is then transposed when assembling per-trial arrays as `(n_neurons, n_timepoints)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` AND a valid CCF annotation (`anno_name` mapped through the Allen CCFv3 ontology) are kept. Units without an annotation are excluded. A session with no such units (after the session selection) is dropped.

ii.
```python
classification = f['units/classification'][:].astype(str)
good = classification == 'good'
region_names, has_region = unit_regions(f, good, region_map)
good_idx = np.where(good)[0][has_region]
good = np.zeros(len(classification), dtype=bool)
good[good_idx] = True
```

In `unit_regions`:
```python
regions = np.array([region_map.get(a.strip(), None) for a in anno], dtype=object)
keep = np.array([r is not None for r in regions])
```

iii. The AI stated: "only `classification == 'good'` with a CCF annotation -- matching the reference, which keeps only units with both ephys and histology." This results in 57,560 neurons compared to the reference's 69,453.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and event times are on the same session-absolute clock. The bin edges relative to the go cue are added to each trial's go-cue time to give absolute time windows, and spikes are binned against those edges directly.

ii.
```python
edges = go_time[:, None] + BIN_EDGES[None, :]
flat_edges = edges.ravel()
...
counts = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
```

iii. Identical approach to the reference. No resampling or interpolation needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue. The bin grid is defined once and reused for every trial and session.

ii.
```python
T_START = -2.5
T_STOP = 1.5
BIN_WIDTH = 0.05
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))  # 80
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
```

iii. This matches the instructions exactly and is identical to the reference.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (the tone onsets) and the go-cue time. The tone taken for a trial is the **last** `sample_start_times` entry before its go cue.

ii.
```python
sample_starts = events['sample_start_times']['timestamps'][:]
idx = np.searchsorted(sample_starts, out['go_time']) - 1
out['tone_time'] = sample_starts[np.clip(idx, 0, len(sample_starts) - 1)]
out['tone_valid'] = (idx >= 0) & (out['tone_time'] >= out['start_time'])
```

iii. The AI noted: "The tone is played during the sample epoch. On early-lick trials the sample/delay epoch is replayed, so the tone the animal finally responded to is the last sample onset preceding the go cue." This matches the reference reasoning.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as the bin center time (relative to go cue) minus the tone-to-go-cue offset (which is negative, making the subtraction add).

ii.
```python
tone_rel = tt['tone_time'][trials] - go_time  # negative
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]
```

iii. This is algebraically equivalent to the reference's `CENTERS + (go - tone)`. The sign convention differs but the result is the same: positive values after tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both share the same bin grid defined by `BIN_CENTERS` relative to the go cue. Each bin center's value is the time since tone onset at that moment.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]
```

iii. Same alignment as the reference -- both use the same go-cue-relative bin centers.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` and the go cue used to place them on the trial's time axis.

ii.
```python
onset = tt['photostim_onset'][trials]
duration = tt['photostim_duration'][trials]
start_time = tt['start_time'][trials]
for i in range(n_trials):
    if onset[i] == 'N/A':
        continue
    on = start_time[i] + float(onset[i]) - go_time[i]
    off = on + float(duration[i])
```

iii. Same source variables as the reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 if the bin interval overlaps with the photostimulation interval, using bin edges rather than bin centers. This differs from the reference, which uses bin centers.

ii.
```python
photostim[i] = ((BIN_EDGES[:-1] < off) & (BIN_EDGES[1:] > on))
```

iii. The AI uses bin-edge overlap (any overlap between `[edge_start, edge_end]` and `[on, off]` marks the bin as 1), while the reference uses bin-center containment (`center >= on and center < off`). This is a minor difference that may affect 1-2 edge bins per stimulated trial.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are expressed relative to the go cue, the same event the neural bins are aligned on, so they can be compared against the bin edges/centers directly.

ii.
```python
on = start_time[i] + float(onset[i]) - go_time[i]
off = on + float(duration[i])
photostim[i] = ((BIN_EDGES[:-1] < off) & (BIN_EDGES[1:] > on))
```

iii. Same alignment principle as the reference.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from two trials-table columns: `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore'). A hit means the animal licked the instructed side, a miss means it licked the other side, and an ignore means no lick.

ii.
```python
outcome_str = tt['outcome'][trials]
instruction = tt['instruction'][trials]
choice = np.full(n_trials, 2, dtype=np.int64)  # no lick
licked_left = (((outcome_str == 'hit') & (instruction == 'left'))
               | ((outcome_str == 'miss') & (instruction == 'right')))
licked_right = (((outcome_str == 'hit') & (instruction == 'right'))
                | ((outcome_str == 'miss') & (instruction == 'left')))
choice[licked_left] = 0
choice[licked_right] = 1
```

iii. The AI stated: "Checked against the first lick after the go cue: they agree on > 99.8% of trials." This matches the reference approach.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0 (left), 1 (right), 2 (no lick), and repeated across all 80 bins per trial.

ii.
```python
outputs = [np.stack([np.full(N_BINS, choice[i]),
                     np.full(N_BINS, outcome[i]),
                     np.full(N_BINS, early_lick[i]),
                     tongue[i]]).astype(np.int64)
           for i in range(n_trials)]
```

iii. Same coding as the reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, which holds 'ignore', 'miss', and 'hit'.

ii.
```python
outcome_str = tt['outcome'][trials]
outcome = np.full(n_trials, -1, dtype=np.int64)
outcome[outcome_str == 'ignore'] = 0
outcome[outcome_str == 'miss'] = 1
outcome[outcome_str == 'hit'] = 2
```

iii. Same as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to 0 (ignore), 1 (miss), 2 (hit) and repeated across all 80 bins.

ii.
```python
outcome[outcome_str == 'ignore'] = 0
outcome[outcome_str == 'miss'] = 1
outcome[outcome_str == 'hit'] = 2
```

iii. Same coding as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'no early' and 'early'.

ii.
```python
early = tt['early_lick'][trials]
early_lick = (early == 'early').astype(np.int64)
```

iii. Same as the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no) and 1 (yes), repeated across all 80 bins.

ii.
```python
early_lick = (early == 'early').astype(np.int64)
```

iii. Same coding as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = tongue_x, tongue_y, tongue_likelihood, with matching `timestamps`. Column 1 (tongue_y) is the value; column 2 (likelihood) decides whether the tongue is visible.

ii.
```python
tracking = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
ts = tracking['timestamps'][:]
data = tracking['data'][:]
y = data[:, 1]
visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESHOLD
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood <= 0.5 are excluded. Visible frames are averaged per 50 ms bin using cumulative sums (vectorized). The 40th and 60th percentiles are taken over the **raw visible values** across the session (not bin means as in the reference). Each trial's bins are then discretized: 0 (< 40th pct), 1 (40th-60th pct), 2 (> 60th pct), 3 (not visible).

ii.
```python
visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESHOLD
csum_y = np.concatenate([[0.0], np.cumsum(np.where(visible, y, 0.0))])
csum_n = np.concatenate([[0], np.cumsum(visible.astype(np.int64))])
...
mean_y = np.where(n_visible > 0, sum_y / np.maximum(n_visible, 1), np.nan)
```

Percentile computation:
```python
finite = tongue_y[np.isfinite(tongue_y)]
if finite.size >= 10:
    p40, p60 = np.percentile(finite, [40, 60])
```

Discretization:
```python
tongue[visible & (tongue_y <= p60)] = 1
tongue[visible & (tongue_y < p40)] = 0
tongue[visible & (tongue_y > p60)] = 2
```

iii. The AI uses cumulative sums for efficient vectorized bin-mean computation. However, the percentiles are taken over the raw per-bin mean values of the trial windows (the `tongue_y` here is already per-bin means from `tongue_y_per_bin`), NOT over the session-wide bin means as in the reference. The reference bins the entire session into 50ms bins first, takes bin means, then computes percentiles over those session-wide bin means. The AI computes per-trial bin means and takes percentiles over those. This is a subtle difference.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The discretization uses `<=` and `>` comparisons rather than `np.digitize`. Values <= p60 get class 1, values < p40 get class 0 (overwriting class 1), values > p60 get class 2. Bins with no visible frames stay at class 3.

ii.
```python
tongue[visible & (tongue_y <= p60)] = 1
tongue[visible & (tongue_y < p40)] = 0
tongue[visible & (tongue_y > p60)] = 2
```

iii. This gives: 0 for y < p40, 1 for p40 <= y <= p60, 2 for y > p60. The reference uses `np.digitize(m, edges)` which gives: 0 for y < p40, 1 for p40 <= y < p60, 2 for y >= p60. The boundary handling at exactly p60 differs: the AI assigns class 1, the reference assigns class 2. This is a minor difference that affects very few bins.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and go cues. Frame-to-bin assignment uses `searchsorted` on the camera timestamps at each trial's bin edges, and cumulative sums compute per-bin means efficiently.

ii.
```python
edges = go_time[:, None] + BIN_EDGES[None, :]
idx = np.searchsorted(ts, edges)
n_visible = np.diff(csum_n[idx], axis=1)
sum_y = np.diff(csum_y[idx], axis=1)
```

iii. The vectorized approach using cumulative sums is more efficient than the reference's per-trial loop but achieves the same alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Sessions without good units**: rejected during session selection.
- **Sessions failing behavioral criteria**: rejected (31 sessions).
- **Units without CCF annotation**: excluded from the dataset.
- **Trials outside obs_intervals**: excluded via `observed_trials`.
- **Trials without locatable tone onset**: excluded via `tone_valid`.
- **Frames with low tongue tracking likelihood**: excluded from bin means; bins with no visible frames get class 3 ("not visible").
- **Sessions with < 10 visible tongue bins**: entire session tongue output set to class 3.
- **NaN classification values**: handled by `.astype(str)` which converts NaN to string 'nan', which doesn't match 'good'.

ii.
```python
classification = f['units/classification'][:].astype(str)
good = classification == 'good'
```

```python
out['tone_valid'] = (idx >= 0) & (out['tone_time'] >= out['start_time'])
```

```python
if finite.size >= 10:
    p40, p60 = np.percentile(finite, [40, 60])
```

iii. The AI's approach is more aggressive in filtering -- it removes more sessions and units than the reference but provides explicit documentation of what was removed and why.

## 10-a. What are the most time-consuming steps of the code?

i. The AI uses multiprocessing (`Pool` with 16 workers) for both session scanning and conversion, which parallelizes I/O and computation. The most time-consuming steps per session are reading the NWB file (especially `spike_times` and tongue tracking arrays), computing the spike binning (per-unit `searchsorted` loop), and the tongue y-position computation.

ii.
```python
with Pool(args.nproc) as pool:
    scans = pool.map(scan_session, files)
...
with Pool(args.nproc) as pool:
    for i, result in enumerate(pool.imap(convert_session, jobs)):
```

iii. The multiprocessing approach is an efficiency improvement over the reference's sequential processing, though it requires reading each file twice (once for scanning, once for conversion).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops: (1) the per-unit spike binning loop iterates over each good unit, and (2) the photostim computation loops over trials. The tongue y computation is already vectorized using cumulative sums.

ii.
Per-unit loop:
```python
for i, u in enumerate(unit_ids):
    st = spike_times[start[u]:stop[u]]
    counts = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
    rates[:, :, i] = np.diff(counts, axis=1) / BIN_WIDTH
```

Per-trial photostim loop:
```python
for i in range(n_trials):
    if onset[i] == 'N/A':
        continue
    on = start_time[i] + float(onset[i]) - go_time[i]
    off = on + float(duration[i])
    photostim[i] = ((BIN_EDGES[:-1] < off) & (BIN_EDGES[1:] > on))
```

iii. The per-unit loop is inherent to ragged spike time storage and cannot be easily vectorized. The photostim loop could be vectorized (as the reference does). The tongue computation is already vectorized.

## 10-c. What processing does the code repeat multiple times?

i. The code reads each NWB file **twice**: once in `scan_session` (to decide whether to include it) and once in `convert_session` (to do the actual conversion). The trial table, classification, and some other fields are read both times. The AI mitigates this with a caching mechanism.

ii.
```python
# First pass:
def scan_session(path):
    with h5py.File(path, 'r') as f:
        tt = _trial_table(f)
        ...

# Second pass:
def convert_session(args):
    ...
    with h5py.File(path, 'r') as f:
        tt = _trial_table(f)
        ...
```

iii. The double-read is a consequence of the two-phase architecture (scan then convert). The caching mechanism avoids recomputation on subsequent runs.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI downloads and processes the full Allen CCFv3 structure graph CSV to build a region mapping, which adds complexity. The `session_performance` computation is done to filter sessions, but the performance values are only stored in metadata and not used downstream. The AI also computes and stores extensive metadata that isn't used by the decoder.

ii.
```python
def build_region_map():
    if not os.path.exists(STRUCTURE_CSV):
        _download_structures(STRUCTURE_CSV)
    rows = list(csv.DictReader(open(STRUCTURE_CSV)))
    ...
```

iii. The Allen CCF structure graph processing is substantial additional work that the reference avoids entirely by using the fine-grained `anno_name` labels directly.
