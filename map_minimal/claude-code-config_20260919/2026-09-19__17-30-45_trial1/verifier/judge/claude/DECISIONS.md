# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All sessions are found with a glob pattern `sub-*/*.nwb`, and each file is opened with **h5py** (not pynwb) and processed individually. Subjects, trials, units, and behavioral events are read from the HDF5 group structure directly. The agent uses multiprocessing (`Pool`) to process sessions in parallel with up to 16 workers.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
```

```python
with h5py.File(path, 'r') as nwb:
    subject = nwb['general/subject/subject_id'][()].decode()
    trials = nwb['intervals/trials']
    trial_start = trials['start_time'][:]
    ...
    go_times = nwb['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. The agent explored both pynwb and h5py (both available) and chose h5py for direct HDF5 access. It found 174 NWB files across 28 subjects totaling ~50 GB. The agent verified file structure by walking the HDF5 tree before writing the conversion code.

## 1-b. How are the data split into subjects?

i. Each NWB file records its animal in `general/subject/subject_id`. That value is read for every session and used during assembly, where `subjects` is the sorted set of unique IDs and `subject_idx` gives each session's index into that list.

ii.
```python
subject = nwb['general/subject/subject_id'][()].decode()
```

At assembly:
```python
subjects = sorted({s['subject'] for s in kept})
subject_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_index[s['subject']] for s in kept], dtype=np.int64),
```

iii. `subject_id` is the canonical animal identifier in the NWB file. The agent verified 28 subjects across 174 sessions.

## 1-c. How are the data split into sessions?

i. One NWB file is one session. Each file is processed independently. After session-level curation (performance thresholds), 142 of 174 sessions are retained. Session order follows the sorted file list.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
```

iii. The dandiset stores one session per file, so the file boundary is the session boundary. The agent applies additional session-level curation criteria from the data paper (see 1-e).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioral trial. The go-cue count is verified to match the trial count.

ii.
```python
trials = nwb['intervals/trials']
trial_start = trials['start_time'][:]
trial_stop = trials['stop_time'][:]
go_times = nwb['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(go_times) == len(trial_start), 'go cue count != trial count'
```

iii. Trials are clearly defined by the trials table. The agent verified the go-cue count matches the trial count.

## 1-e. How are trials filtered based on quality controls?

i. Three levels of filtering are applied:

1. **Trial-level**: auto_water and free_water trials are excluded (outcomes don't reflect animal decisions). Trials outside the ephys recording period (`obs_intervals`) are also dropped.
2. **Session-level**: Sessions are rejected if control-trial performance is <= 65% or if there are fewer than 50 correct lick-left or lick-right trials (data paper criteria).
3. **Minimum trial count**: Sessions with fewer than 2 surviving trials or 0 good units are rejected.

Photostim, early-lick, and ignore trials are deliberately kept because the decoder task requires them.

ii.
```python
keep = ~(auto_water | free_water)
if good.any():
    keep &= observed_trials(nwb, good, trial_start)
```

Session curation:
```python
control = keep & ~stim_trial & ~early
hit = outcome == 'hit'
miss = outcome == 'miss'
responded = (hit | miss) & control
performance = hit[responded].mean() if responded.any() else 0.0
n_left = int(np.sum(hit & control & (instruction == 'left')))
n_right = int(np.sum(hit & control & (instruction == 'right')))
rejected = (performance <= MIN_PERFORMANCE
            or n_left < MIN_CORRECT_PER_DIRECTION
            or n_right < MIN_CORRECT_PER_DIRECTION)
```

iii. The agent followed the data paper's session selection criteria ("overall behavioral performance > 65%, and at least 50 correct lick left and lick right trials each") and the method paper's trial exclusion of free-water/auto-water trials. The agent explicitly noted that photostim, early-lick, and ignore trials are kept despite the method paper excluding them, because the decoder task requires them as inputs/outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (sorted spike times per unit in session-absolute seconds). Only units with `classification == 'good'` contribute. The go-cue times (`BehavioralEvents/go_start_times`) provide alignment.

ii.
```python
spike_index = units['spike_times_index'][:]
starts = np.concatenate([[0], spike_index[:-1]])
...
spike_times = units['spike_times']
for i, u in enumerate(unit_ids):
    st = spike_times[starts[u]:spike_index[u]]
```

iii. `spike_times` is the only neural representation in the NWB file; firing rates are computed from it directly.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue. For each good unit, bin edges for all trials are flattened, `np.searchsorted` gives running spike counts, differencing gives per-bin counts, divided by bin width (0.05 s) for Hz. No smoothing, normalization, or baseline subtraction.

ii.
```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
rates = np.empty((len(unit_ids), n_trials, N_BINS), dtype=np.float32)
spike_times = units['spike_times']
for i, u in enumerate(unit_ids):
    st = spike_times[starts[u]:spike_index[u]]
    idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
    rates[i] = np.diff(idx, axis=1) / BIN_WIDTH
```

iii. The agent noted this mirrors the reference pipeline (`preprocessing_DJ_2022Aug.process_one_area` which bins a go-cue-aligned spike train and divides by bin width), with the window/width specified by the task instead of the pipeline's -3..+3 s / 40 ms.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two levels of unit filtering:
1. Only units with `units/classification == 'good'` are kept (the QC classifiers from the spike-sorting white paper).
2. Among good units, only those flagged as good on **every analysed trial** via `units/is_good_trials` survive (costs ~0.8% of units).

A session with no surviving units is dropped.

ii.
```python
good = _str(nwb['units']['classification'][:]) == 'good'
```

```python
def units_good_on_trials(nwb, good, trial_idx):
    flags = nwb['units']['is_good_trials'][:][good][:, trial_idx]
    return flags.all(axis=1)
```

```python
good[good] = units_good_on_trials(nwb, good, trial_idx)
```

iii. The agent verified that `classification == 'good'` matches the data paper's analyzed units. The additional `is_good_trials` filtering was added after the agent discovered some sessions had all-zero neural trials, and verified that only 0.8% of units are lost by this additional filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All times in the NWB file share a session-absolute clock. Bin edges relative to the go cue are added to each trial's go-cue time to produce absolute time edges, and spikes are binned against those edges directly.

ii.
```python
go_times = nwb['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
```

iii. Everything in the NWB file is timestamped on one global clock, so aligning to the go cue only requires looking up each trial's go-cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial, spanning -2.5 s to +1.5 s relative to the go cue. The bin grid is defined once and reused for every trial and session.

ii.
```python
T_START = -2.5
T_STOP = 1.5
BIN_WIDTH = 0.05
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))       # 80
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)   # relative to go cue
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
```

iii. The window and 50 ms bin width are specified by the instructions. No rebinning is applied since the data starts as raw spike times.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` timestamps (the tone onsets) and the go-cue times. For each trial, the last sample_start_time before the go cue is taken as the tone onset, since early licks replay the sample epoch.

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

iii. The agent noted that early licks replay the sample epoch, so a trial can have more than one tone; the last one before the go cue is the one the animal actually answered.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset at each bin is computed as `bin_center_relative_to_go + go_time - tone_onset`. For trials where no sample_start_time is found (missing tone), the agent falls back to `go - 1.85` (the nominal 0.65 s sample + 1.2 s delay structure).

ii.
```python
missing_tone = ~np.isfinite(tone)
if missing_tone.any():
    tone[missing_tone] = go[missing_tone] - 1.85
time_from_tone = (go[:, None] + BIN_CENTERS[None, :]
                  - tone[:, None]).astype(np.float32)
```

iii. The fallback for missing tones uses the nominal task timing structure. The agent tracks `n_trials_missing_tone` in the summary.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It shares the same bin grid as the neural data. The bin centers are defined relative to the go cue, and the tone-onset offset is applied per trial, so each bin's value represents the time from tone onset at the center of that neural bin.

ii.
```python
time_from_tone = (go[:, None] + BIN_CENTERS[None, :]
                  - tone[:, None]).astype(np.float32)
```

iii. The same go-cue-relative grid underlies both neural and input arrays.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, along with `start_time` and go-cue times to compute absolute onset/offset.

ii.
```python
onset = np.array([_parse_float(v) for v in _str(trials['photostim_onset'][:])])
duration = np.array([_parse_float(v) for v in _str(trials['photostim_duration'][:])])
```

iii. The onsets are stored as strings relative to trial start, with 'N/A' on control trials. Both are parsed to float (NaN for N/A).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: a bin is marked as stimulated (1.0) when the bin **overlaps** the stimulation interval (i.e., `bin_right_edge > stim_on AND bin_left_edge < stim_off`), and 0 otherwise.

ii.
```python
on = trial_start + onset - go_times          # relative to the go cue
off = on + duration
valid = np.isfinite(on) & np.isfinite(off)
for i in np.flatnonzero(valid):
    stim[i] = ((BIN_EDGES[1:] > on[i]) & (BIN_EDGES[:-1] < off[i])).astype(np.float32)
```

iii. The agent uses an overlap criterion (any overlap between the bin and the stim interval) rather than testing whether the bin center falls within the interval.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are expressed relative to the go cue (same reference as the neural bins), so the bin edges can be compared directly.

ii.
```python
on = trial_start + onset - go_times          # relative to the go cue
```

iii. Same go-cue-relative coordinates as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore'). A hit means the animal licked the instructed side, a miss means the opposite side, and ignore means no lick.

ii.
```python
other = np.where(instruction == 'left', 'right', 'left')
licked = np.where(outcome == 'hit', instruction,
                  np.where(outcome == 'miss', other, 'no lick'))
choice = np.array([CHOICE_VALUES.index(v) for v in licked], dtype=np.int8)
```

iii. The agent verified this derivation against the recorded lick times and found perfect agreement.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0=left, 1=right, 2=no lick. The per-trial value is repeated across all 80 bins to form a time-varying output array.

ii.
```python
CHOICE_VALUES = ['left', 'right', 'no lick']
...
outputs.append(np.stack([choice[j] * ones, outcome_code[j] * ones,
                         early_code[j] * ones, tongue[j]]).astype(np.int8))
```

iii. Choice is one value per trial, repeated across bins for a consistent (n_output, n_timepoints) shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which holds 'ignore', 'miss', and 'hit'.

ii.
```python
outcome = _str(trials['outcome'][:])
outcome_code = np.array([OUTCOME_CODE[v] for v in outcome], dtype=np.int8)
```

iii. The trials table stores the outcome explicitly with the three required categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to 0=ignore, 1=miss, 2=hit and repeated across all 80 bins.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_code = np.array([OUTCOME_CODE[v] for v in outcome], dtype=np.int8)
```

iii. Direct mapping, one value per trial repeated across bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'no early' and 'early'.

ii.
```python
early = _str(trials['early_lick'][:]) == 'early'
early_code = early.astype(np.int8)
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Coded as 0=no, 1=yes (boolean cast to int8) and repeated across all 80 bins.

ii.
```python
EARLY_LICK_VALUES = ['no', 'yes']
early_code = early.astype(np.int8)
```

iii. Direct boolean conversion, one value per trial repeated across bins.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, whose data is (n_frames, 3) = (tongue_x, tongue_y, tongue_likelihood) with matching timestamps. Column 1 (tongue_y) is the value; column 2 (likelihood) determines visibility.

ii.
```python
track = nwb['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
ts = track['timestamps'][:]
data = track['data'][:]
y = chunk[:, 1]
visible = chunk[:, 2] >= TONGUE_P_CUTOFF
```

iii. This is the only tongue measurement in the file. The agent verified the likelihood is essentially binary (~5e-5 when tongue is in, >0.999 when out).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Steps:
1. Frames with DeepLabCut likelihood < 0.5 are excluded.
2. For each trial, video frames are assigned to 50ms bins and averaged per bin (mean y over visible frames).
3. Per-session 40th and 60th percentiles are computed over **all bins of all analysed trials** where the tongue was visible.
4. Each bin is classified: 0 (below 40th), 1 (40th-60th), 2 (above 60th), 3 (not visible).

ii.
```python
visible = chunk[:, 2] >= TONGUE_P_CUTOFF
...
counts = np.bincount(which[ok], minlength=N_BINS)
sums = np.bincount(which[ok], weights=y[ok], minlength=N_BINS)
seen = counts > 0
y_binned[i, seen] = sums[seen] / counts[seen]
...
seen = np.isfinite(y_binned)
if seen.any():
    p40, p60 = np.percentile(y_binned[seen], TONGUE_PCTILES)
    vals = y_binned[seen]
    classes[seen] = np.where(vals < p40, 0, np.where(vals <= p60, 1, 2)).astype(np.int8)
```

iii. The percentiles are taken over the binned y-values (bin means) of analysed trials only, not over raw frames or over the whole session including non-analysed trials.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Using `np.where` with strict less-than for class 0 and less-than-or-equal for class 1 boundary: values < p40 -> 0, p40 <= values <= p60 -> 1, values > p60 -> 2, not visible -> 3.

ii.
```python
classes[seen] = np.where(vals < p40, 0, np.where(vals <= p60, 1, 2)).astype(np.int8)
```

iii. This means the boundary value exactly at p40 falls into class 1, and the value exactly at p60 also falls into class 1.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the same absolute clock. For each trial, frames in the window [go+T_START, go+T_STOP] are found via searchsorted, then assigned to bins by their offset from the go cue using the same bin edges as neural data.

ii.
```python
lo, hi = np.searchsorted(ts, [go + T_START, go + T_STOP])
...
which = np.searchsorted(BIN_EDGES, ts[lo:hi] - go, side='right') - 1
```

iii. The same bin grid is used for tongue and neural data, guaranteeing temporal alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several categories:
- **Missing tone onset**: Falls back to `go - 1.85` (nominal task timing).
- **Trials outside ephys recording**: Dropped via `observed_trials` (checks obs_intervals for all good units).
- **Tongue not visible**: Bins with no visible frames get class 3 ("not visible").
- **Miss trials cut short**: Trailing bins have zero spikes, reported as zero rate (not imputed).
- **photostim_onset = 'N/A'**: Parsed as NaN, resulting in no stimulation marking.
- **Units failing QC on some trials**: Dropped entirely via `is_good_trials`.
- **auto_water/free_water trials**: Excluded.

ii.
```python
# Missing tone fallback
tone[missing_tone] = go[missing_tone] - 1.85
```

```python
# obs_intervals check
covered = np.ones(len(trial_start), dtype=bool)
for u in np.flatnonzero(good):
    covered &= np.isin(trial_start, obs[starts[u]:index[u], 0])
```

```python
# Tongue not visible
classes = np.full((n_trials, N_BINS), 3, dtype=np.int8)
```

iii. The agent documents known limitations in the metadata and handles each type of missing data appropriately.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files (especially spike_times and tongue tracking arrays) dominates. The agent uses multiprocessing with 16 workers to parallelize session processing. Assembly involves loading per-session pickle caches.

ii.
```python
with Pool(args.workers) as pool:
    for i, s in enumerate(pool.imap(_worker, jobs)):
```

iii. The multiprocessing approach speeds up I/O-bound NWB reading across sessions.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops remain:
1. Per-unit loop in `bin_spikes`: one `searchsorted` per unit over all trials at once (trial dimension is vectorized via flattened edges).
2. Per-trial loop in `tongue_classes`: bins one trial's frames at a time.
3. Per-trial loop in `photostim_series` for valid (stimulated) trials only.

ii.
```python
for i, u in enumerate(unit_ids):
    st = spike_times[starts[u]:spike_index[u]]
    idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
```

```python
for i, go in enumerate(go_times):
    ...  # tongue binning per trial
```

iii. The per-unit loop cannot be collapsed because each unit has different spike counts (ragged storage). The tongue and photostim loops could potentially be vectorized but are not the bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. The `tone_onsets` function computes tone onsets for all trials (including those later filtered), and the result is then indexed by `trial_idx`. Similarly, `photostim_series` processes all trials before indexing. The `_str` conversion is called multiple times on different columns.

ii.
```python
tone = tone_onsets(nwb, trial_start, trial_stop, go_times)[trial_idx]
stim = photostim_series(trials, trial_start, go_times)[trial_idx]
```

iii. These compute values for all trials then select the kept subset, doing some unnecessary work on excluded trials.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes tone onsets and photostim for **all** trials (including auto_water, free_water, and unobserved trials), then discards the excluded ones via `[trial_idx]`. The elaborate brain region mapping with 14 groups and hemisphere prefixes produces ~28 region labels, which may be more granular than needed. Per-session pickle caches are written and re-read during assembly.

ii.
```python
tone = tone_onsets(nwb, trial_start, trial_stop, go_times)[trial_idx]
stim = photostim_series(trials, trial_start, go_times)[trial_idx]
```

iii. The extra computation on excluded trials is minor. The session caching adds I/O overhead but enables parallel processing.
