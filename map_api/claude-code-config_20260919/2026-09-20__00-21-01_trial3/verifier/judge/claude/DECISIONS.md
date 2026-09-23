# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all NWB files under `data/sub-*/` directories, sorts them, and processes each with `pynwb.NWBHDF5IO`. It uses `ProcessPoolExecutor` with up to 24 workers for parallel processing. Each session reads trials, behavioral events, tracking data, and units via the pynwb API.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for res in pool.map(_worker, jobs):
        results.append(res)
```

```python
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
    raw = read_session(nwb)
```

`read_session` extracts trials, events, tracking, units, electrodes:
```python
trials = nwb.intervals['trials']
events = nwb.acquisition['BehavioralEvents'].time_series
tracking = nwb.acquisition['BehavioralTimeSeries'].time_series
# ...
units = nwb.units
classification = np.asarray(units['classification'][:])
```

iii. The AI documents in CONVERSION_NOTES.md that NWB is the published format and pynwb is required by the instructions. It found 174 NWB files across 28 subjects matching `dandiset.yaml`. Parallel processing was chosen for efficiency (26s total vs ~2.5 min serial).

## 1-b. How are the data split into subjects?

i. The AI identifies subjects using `nwb.subject.description` (the mouse name, e.g. `SC015`), not `nwb.subject.subject_id` (the numeric identifier, e.g. `440956`). Unique mouse names are collected and sorted to form the `subjects` list, and each session gets an index into that list.

ii.
```python
out = {
    'mouse': nwb.subject.description,
    'subject_id': str(nwb.subject.subject_id),
    # ...
}
```

```python
mice = sorted({r['mouse'] for r in sessions})
data = {
    'subjects': mice,
    'subject_idx': np.array([mice.index(r['mouse']) for r in sessions], dtype=np.int64),
    # ...
}
```

iii. CONVERSION_NOTES.md Step 2 documents that `nwb.subject.description` is the mouse name (e.g. `SC015`) and `subject_id` is a numeric identifier. The AI chose to use the human-readable mouse name for the `subjects` list. Both are unique 1:1 mappings (28 of each).

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. Session identity is `nwb.identifier` (e.g. `SC015_20190207_120657_s1`). Sessions are sorted by their file path (which embeds subject and date), giving deterministic chronological ordering within each subject.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
sessions = sorted(sessions, key=lambda r: r['identifier'])
```

iii. CONVERSION_NOTES.md Step 2 documents one NWB file per session, with 174 files total. The AI found this consistent with the dandiset metadata.

## 1-d. How are the data split into trials?

i. Trials come from `nwb.intervals['trials']`, one row per behavioral trial. The AI asserts that the number of go-cue events matches the number of trials.

ii.
```python
n_trials_all = len(raw['start_time'])
assert len(go) == n_trials_all, (
    f"{raw['identifier']}: {len(go)} go cues for {n_trials_all} trials")
```

iii. CONVERSION_NOTES.md Step 2 documents the trials table columns. Step 4 verifies exactly one go cue per trial across all 174 sessions.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple layers of filtering:

**Trial-level curation:**
1. Trials outside the electrophysiological recording (`obs_intervals`) are removed.
2. `auto_water` and `free_water` trials are removed (reference `get_regular_trial_mask`).
3. Trials with no video frame in the analysis window are removed.

**Session-level curation** (from the data paper):
4. Sessions with 0 good units are dropped.
5. Sessions with < 2 usable trials are dropped.
6. Sessions with behavioral performance ≤ 65% are dropped.
7. Sessions with < 50 correct lick-left OR < 50 correct lick-right control trials are dropped.

This results in 142 sessions with 73,845 trials (from 174 sessions / 94,990 trials).

ii.
```python
# Trial curation
keep = raw['recorded'] & (raw['auto_water'] == 0) & (raw['free_water'] == 0)
keep &= n_frames_all > 0
trial_idx = np.flatnonzero(keep)

# Session curation
performance, n_left, n_right = session_performance(raw, keep)
reject = None
if n_good == 0:
    reject = 'no good units'
elif len(trial_idx) < 2:
    reject = f'only {len(trial_idx)} usable trials'
elif performance <= MIN_PERFORMANCE:
    reject = f'performance {performance:.3f} <= {MIN_PERFORMANCE}'
elif n_left < MIN_CORRECT_PER_DIRECTION or n_right < MIN_CORRECT_PER_DIRECTION:
    reject = f'correct trials L={n_left} R={n_right} < {MIN_CORRECT_PER_DIRECTION}'
```

```python
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50
```

iii. CONVERSION_NOTES.md Step 5 justifies each decision:
- `auto_water`/`free_water`: reference `get_regular_trial_mask` excludes both; outcomes aren't genuine behavioral reports.
- No-video trials: every bin would be "tongue not visible" when truth is "not measured" (0.8% of trials).
- Session curation: explicitly from the data paper ("overall behavioral performance > 65%, and at least 50 correct lick left and lick right trials each"). Performance computed on control, non-early-lick trials, excluding auto/free-water. The AI validates that selected sessions match paper's statistics (83.8% mean, range 65.8–98.9% vs paper's 84%, 65–99%).
- Early-lick, no-response, and photostim trials are deliberately kept because they are decoder outputs/inputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units['spike_times']` for units with `classification == 'good'`. Go-cue times from `BehavioralEvents/go_start_times` define the alignment.

ii.
```python
spike_index = units['spike_times']
counts = np.empty((n_good, n_trials, N_BINS), dtype=np.int32)
for i, unit in enumerate(raw['good_units']):
    counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
```

iii. CONVERSION_NOTES.md Step 1 identifies `spike_times` as the only neural representation in the NWB files. The AI notes this is electrophysiology (not imaging), so no dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning [-2.5, +1.5) s relative to the go cue. Counts are divided by the bin width (0.05 s) to get firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
def bin_spike_times(spike_times, edges_abs):
    pos = np.searchsorted(spike_times, edges_abs.ravel()).reshape(edges_abs.shape)
    return np.diff(pos, axis=1).astype(np.int32)
```

```python
neural = np.ascontiguousarray(
    counts.transpose(1, 0, 2).astype(np.float32) / BIN_WIDTH)
```

iii. CONVERSION_NOTES.md Step 3 notes the reference code uses `sliding_histogram(..., rate=True)` which returns `count / bin_width`. The AI follows the same rate convention but with 50 ms bins (as required by the instructions) instead of the reference's 40 ms / 3.4 ms stride.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units['classification'] == 'good'` are kept. No individual quality metric thresholds are applied. Sessions with 0 good units are dropped.

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.flatnonzero(classification == 'good')
```

iii. CONVERSION_NOTES.md Step 1 identifies this as the reference pipeline's `qc_mode='classifier'`. Step 3 notes 69,453 good units across 173 sessions with good units (25.5% of clusters), close to the paper's 69,943 / 25.9%. The difference is attributed to the dandiset version (0.230822 vs 0.231012).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All times in the NWB file share a session-absolute clock. For each trial, the go-cue time is used to compute absolute bin edges, and spikes are binned against those edges via `searchsorted`.

ii.
```python
def window_edges(go_times):
    return go_times[:, None] + BIN_EDGES[None, :]

edges = window_edges(go)
# ...
counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
```

iii. CONVERSION_NOTES.md Step 4 confirms go cue alignment: "everything is aligned to the go cue." No resampling or per-stream offset correction needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins spanning [-2.5, +1.5) s relative to go cue. Bin edges defined once as 81 offsets from the go cue. No temporal rebinning is applied (spikes are binned directly from raw spike times).

ii.
```python
OFF_START = -2.5          # s
OFF_END = 1.5             # s
BIN_WIDTH = 0.05          # s
N_BINS = int(round((OFF_END - OFF_START) / BIN_WIDTH))          # 80
BIN_EDGES = OFF_START + BIN_WIDTH * np.arange(N_BINS + 1)       # (81,)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])            # (80,)
```

iii. CONVERSION_NOTES.md Step 3 notes the reference uses 40 ms / 3.4 ms stride, but the task prescribes 50 ms bins. The AI correctly follows the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (tone onset events) and `go_start_times` (go cue). The tone for a trial is the last sample onset at or before the go cue.

ii.
```python
'sample_start': np.asarray(events['sample_start_times'].timestamps[:], dtype=np.float64),
'go': np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64),
```

```python
tone_idx = np.searchsorted(raw['sample_start'], go, side='right') - 1
assert np.all(tone_idx >= 0), f"{raw['identifier']}: go cue before any tone"
tone = raw['sample_start'][tone_idx]
```

iii. CONVERSION_NOTES.md Step 4 explains that early licking replays the sample epoch, producing extra sample_start_times per trial. The last onset before the go cue is the tone the animal acted on. This is verified to always lie within the trial (0 violations dataset-wide).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the value at each bin is the bin center (relative to go cue) plus the gap between go cue and tone onset, giving seconds elapsed since tone onset.

ii.
```python
time_from_tone = (BIN_CENTERS[None, :] + (go - tone)[:, None]).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 maps this to the reference's `task_sample_time`. The value crosses zero exactly at the true tone onset time.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin grid (80 centers at [-2.475, -2.425, ..., +1.475] s relative to the go cue) is used for both neural activity and the time-from-tone input, ensuring bin k of the input corresponds to the same time window as bin k of the neural data.

ii.
```python
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
time_from_tone = (BIN_CENTERS[None, :] + (go - tone)[:, None]).astype(np.float32)
```

iii. No additional alignment is needed since both use the same go-cue-relative grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `BehavioralEvents/photostim_start_times` and `photostim_stop_times` (absolute timestamps of stimulation onset and offset). The AI uses these event streams rather than the trials table columns (`photostim_onset`, `photostim_duration`).

ii.
```python
'photostim_start': np.asarray(events['photostim_start_times'].timestamps[:], dtype=np.float64),
'photostim_stop': np.asarray(events['photostim_stop_times'].timestamps[:], dtype=np.float64),
```

iii. CONVERSION_NOTES.md Step 4 verifies that photostim events map 1:1 to trials with `photostim_onset != 'N/A'` in all 174 sessions. Step 1 notes the NWB equivalent of the reference's `task_stimulation` laser on/off times.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Each photostim event is assigned to a trial via `searchsorted` on trial start times. A bin is marked as 1 if the bin interval `[edge_start, edge_end)` overlaps with `[stim_start, stim_stop)`, and 0 otherwise. This is a binary time series, not a per-trial flag.

ii.
```python
photostim = np.zeros((n_trials, N_BINS), dtype=np.float32)
if len(raw['photostim_start']):
    stim_trial = np.searchsorted(raw['start_time'], raw['photostim_start'],
                                 side='right') - 1
    position = np.searchsorted(trial_idx, stim_trial)
    in_kept = (position < n_trials) & (trial_idx[np.clip(position, 0, n_trials - 1)]
                                       == stim_trial)
    for row, on, off in zip(position[in_kept], raw['photostim_start'][in_kept],
                            raw['photostim_stop'][in_kept]):
        overlap = (edges[row, :-1] < off) & (edges[row, 1:] > on)
        photostim[row, overlap] = 1.0
```

iii. CONVERSION_NOTES.md Step 4 verifies stimulation always ends before (or at) the go cue. The overlap approach (`bin_start < stim_stop AND bin_end > stim_start`) detects any intersection of the bin with the stimulation period.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim onset and offset are absolute session times. The bin edges for each trial are also absolute times (go cue + relative offset). The overlap is computed in absolute time, ensuring the same bin grid is used for both neural and photostim data.

ii.
```python
edges = window_edges(go)  # absolute times
overlap = (edges[row, :-1] < off) & (edges[row, 1:] > on)
```

iii. Both use the same go-cue-aligned bin edges, so alignment is guaranteed.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From `trials['trial_instruction']` (left/right) and `trials['outcome']` (hit/miss/ignore). Choice is not stored directly and must be derived from the combination.

ii.
```python
instruction = raw['instruction'][trial_idx]
outcome = raw['outcome'][trial_idx]

choice = np.where(instruction == 'left', CHOICE_LEFT, CHOICE_RIGHT)
choice = np.where(outcome == 'miss', 1 - choice, choice)
choice = np.where(outcome == 'ignore', CHOICE_NOLICK, choice)
```

iii. CONVERSION_NOTES.md Step 5 explains: hit means the animal licked the instructed side, miss means it licked the opposite, ignore means no lick. This is the standard reading matching the reference's `correctness` semantics.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived choice is coded as 0=left, 1=right, 2=no lick, and broadcast across all 80 bins (constant within a trial).

ii.
```python
CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
```

```python
outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int8)
outputs[:, 0, :] = choice[:, None]
```

iii. CONVERSION_NOTES.md Step 5 maps this to the reference's `trial_type × correctness` from `create_4fold_trial_type_mask`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `trials['outcome']`, which contains 'ignore', 'miss', or 'hit'.

ii.
```python
'outcome': np.asarray(trials['outcome'][:]),
```

iii. The trials table stores outcome with exactly the three categories the instructions require.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to integers: ignore=0, miss=1, hit=2. The value is constant within a trial, broadcast across all 80 bins.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_code = np.array([OUTCOME_CODE[o] for o in outcome], dtype=np.int8)
outputs[:, 1, :] = outcome_code[:, None]
```

iii. Coding follows the instructions' ordering.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `trials['early_lick']`, which holds 'no early' or 'early'.

ii.
```python
'early_lick': np.asarray(trials['early_lick'][:]),
```

iii. The trials table stores early lick status directly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Coded as 0=no, 1=yes. Constant within a trial, broadcast across all 80 bins.

ii.
```python
early_code = (early == 'early').astype(np.int8)
outputs[:, 2, :] = early_code[:, None]
```

iii. Binary coding follows the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains (n_frames, 3) = (tongue_x, tongue_y, tongue_likelihood) with per-frame timestamps. Column 1 (y) is the position; column 2 (likelihood) determines visibility.

ii.
```python
tongue = tracking['Camera0_side_TongueTracking']
out['video_t'] = np.asarray(tongue.timestamps[:], dtype=np.float64)
tongue_data = np.asarray(tongue.data[:], dtype=np.float64)
out['tongue_y'] = tongue_data[:, 1]
out['tongue_likelihood'] = tongue_data[:, 2]
```

iii. CONVERSION_NOTES.md Step 2 confirms this is the only tongue measurement, present in all 174 sessions at ~294 Hz. The column layout is from the series' description attribute.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Processing steps:
1. Frames with `likelihood <= 0.9` are marked as invisible (the AI uses a threshold of 0.9, noting the bimodal distribution makes the exact threshold immaterial).
2. For each trial, visible frames within each 50 ms bin are averaged to get a per-bin mean y position, using a vectorized prefix-sum approach (`segment_sums`).
3. Per-session percentiles (40th and 60th) of all visible per-bin mean y values across all retained trials are computed.
4. Bins with visible frames are classified: 0 (below 40th pct), 1 (between), 2 (above 60th pct). Bins with no visible frame get class 3 ("not visible").

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
TONGUE_LOW_PCT, TONGUE_HIGH_PCT = 40.0, 60.0
```

```python
visible = raw['tongue_likelihood'] > TONGUE_LIKELIHOOD_THRESHOLD
_, n_visible_bin, y_sum_bin = segment_sums(
    raw['tongue_y'], visible, edges, raw['video_t'])
bin_visible = n_visible_bin > 0
y_mean = np.where(bin_visible, y_sum_bin / np.maximum(n_visible_bin, 1), np.nan)

tongue_class = np.full((n_trials, N_BINS), TONGUE_INVISIBLE, dtype=np.int8)
if bin_visible.any():
    visible_values = y_mean[bin_visible]
    p_low, p_high = np.percentile(visible_values, [TONGUE_LOW_PCT, TONGUE_HIGH_PCT])
    cls = np.where(y_mean < p_low, 0, np.where(y_mean > p_high, 2, 1))
    tongue_class[bin_visible] = cls[bin_visible].astype(np.int8)
```

iii. CONVERSION_NOTES.md Step 5 decision 7: the likelihood is extremely bimodal (87% < 1e-4, 11% > 0.999), so threshold choice is immaterial. Decision 8: percentiles are computed per-session over the visible per-bin y-values of all retained trials, so classes 0/1/2 hold ~40%/20%/40% of visible bins by construction. The AI chose not to apply the reference code's 5-sigma velocity outlier removal, arguing the coarse 3-class percentile bands are insensitive to the residual outliers.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles of visible per-bin mean y values (computed per-session, over all trial-window bins) define two thresholds. Values below the 40th percentile → class 0, between 40th and 60th → class 1, above 60th → class 2, not visible → class 3.

ii.
```python
p_low, p_high = np.percentile(visible_values, [TONGUE_LOW_PCT, TONGUE_HIGH_PCT])
cls = np.where(y_mean < p_low, 0, np.where(y_mean > p_high, 2, 1))
tongue_class[bin_visible] = cls[bin_visible].astype(np.int8)
```

Note: values exactly at `p_low` get class 1 (not 0), and values exactly at `p_high` also get class 1 (not 2).

iii. CONVERSION_NOTES.md Step 5 maps this to the instructions' discretization scheme: 0 < 40th pct, 1 40th-60th pct, 2 > 60th pct, 3 not visible.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The same go-cue-aligned bin edges are used for both neural activity and tongue tracking. Camera timestamps share the session-absolute clock. The prefix-sum approach (`segment_sums`) uses `searchsorted` on camera timestamps at the absolute bin edges to assign frames to bins.

ii.
```python
_, n_visible_bin, y_sum_bin = segment_sums(
    raw['tongue_y'], visible, edges, raw['video_t'])
```

```python
def segment_sums(values, mask, edges_abs, sample_times):
    pos = np.searchsorted(sample_times, edges_abs.ravel()).reshape(edges_abs.shape)
    # ...
```

iii. Both streams use the same bin edges, guaranteeing bin k of tongue output covers the same time interval as bin k of neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Session with no good units** (1 session): dropped entirely when `good` array is empty.
- **Session never quality-controlled** (same session): `classification` is not a string, detected when it doesn't equal `'good'`.
- **Trials outside ephys recording** (8 sessions affected): detected via `obs_intervals`, cross-checked between first and last good units; 736 trials removed.
- **Auto-water and free-water trials**: removed because their outcomes aren't genuine behavioral reports.
- **Trials with no video in the analysis window** (3 sessions, 239 trials): removed to avoid false "not visible" labels.
- **Tongue not visible**: bins get class 3 instead of imputation.
- **Bins outside trial `[start_time, stop_time]`**: firing rate is 0; documented in metadata as a known limitation.

ii.
```python
not_recorded = ~raw['recorded']
keep = raw['recorded'] & (raw['auto_water'] == 0) & (raw['free_water'] == 0)
keep &= n_frames_all > 0
```

```python
'known_limitation': 'The NWB release stores spikes only within each trial\'s '
                    '[start_time, stop_time] interval ...'
```

iii. CONVERSION_NOTES.md Step 10 documents each edge case and its handling. Issue 1 was discovered during Step 7 when a processing plot showed an empty spike raster. The `recorded_trials()` function was added to fix it.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies NWB file I/O (reading) as the dominant cost, with the full conversion completing in 25 seconds using 24 parallel workers. Writing the 9.66 GB pickle takes ~11 seconds. Per-session conversion takes 0.5–2.1 seconds depending on unit/trial count.

ii.
```python
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for res in pool.map(_worker, jobs):
        results.append(res)
```

iii. CONVERSION_NOTES.md Step 7 estimates: 0.5–1.3 s per session (24 in parallel), ~11 s for pickle writing, 26 s total. Well under the 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. One loop remains: the per-unit spike binning loop, which runs one `searchsorted` per unit across all trials. The AI already vectorized the per-trial dimension by stacking all trial edges into one array. The tongue processing is fully vectorized via prefix sums.

ii.
```python
for i, unit in enumerate(raw['good_units']):
    counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
```

The photostim loop also iterates per-event:
```python
for row, on, off in zip(position[in_kept], raw['photostim_start'][in_kept],
                        raw['photostim_stop'][in_kept]):
    overlap = (edges[row, :-1] < off) & (edges[row, 1:] > on)
    photostim[row, overlap] = 1.0
```

iii. CONVERSION_NOTES.md Step 6 notes the per-unit loop cannot be collapsed because each unit has a different number of spikes (ragged storage). The tongue processing was vectorized from the start using prefix sums.

## 10-c. What processing does the code repeat multiple times?

i. Nothing is recomputed. Each NWB file is opened once. The bin grid is defined once at module level. Session-level quantities (e.g. tongue percentiles) are computed inside the single-pass conversion.

ii. N/A

iii. CONVERSION_NOTES.md Step 6 describes a single-pass architecture.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes substantial extra metadata not needed by the decoder:
- Per-neuron CCF coordinates and hemisphere (`neuron_ccf`, `neuron_hemisphere`)
- Per-neuron annotation strings (`neuron_annotation`)
- Per-session performance statistics, trial breakdown counts
- Trial indices within the original file
- Electrode indices

These are all stored in `metadata` and not used by `train_decoder.py`.

ii.
```python
hemisphere = np.where(raw['ccf'][:, 0] >= 5700.0, 'left', 'right')
# ...
'neuron_ccf_coordinates': [r['neuron_ccf'] for r in sessions],
'neuron_hemisphere': [r['neuron_hemisphere'] for r in sessions],
'neuron_annotation': [r['neuron_annotation'] for r in sessions],
```

Also, the AI created `ccf_regions.py` (~200 lines) for a complex CCF-to-region mapping that produces `brain_region_idx`, which the decoder uses, but the extra hemisphere/coordinate data is purely documentation.

iii. CONVERSION_NOTES.md does not explicitly discuss this as unnecessary processing. The extra metadata serves documentation/provenance purposes but is not needed for decoding.
