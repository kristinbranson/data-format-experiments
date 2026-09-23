# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed all NWB files under `/app/data/sub-*/*.nwb`, then opened each file with `pynwb.NWBHDF5IO`. Within each file it pulled the trials table, behavioral event timestamps, behavioral time series, unit table, and electrode metadata into a raw-session dictionary. Session conversion was parallelized with `ProcessPoolExecutor`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for res in pool.map(_worker, jobs):
        results.append(res)
```

```python
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
    raw = read_session(nwb)
```

```python
trials = nwb.intervals['trials']
events = nwb.acquisition['BehavioralEvents'].time_series
tracking = nwb.acquisition['BehavioralTimeSeries'].time_series
...
units = nwb.units
```

iii. In `CONVERSION_NOTES.md`, the agent justified this as the required `pynwb`-only loading path for the published NWB release, with one file per recording session.

## 1-b. How are the data split into subjects (mice)?

i. The agent read both `nwb.subject.subject_id` and `nwb.subject.description`, but it used `subject.description` (mouse names such as `SC015`) as the actual subject key in the output `subjects` list and `subject_idx`. The numeric `subject_id` was kept only in metadata.

ii.
```python
out = {
    'identifier': nwb.identifier,
    'mouse': nwb.subject.description,
    'subject_id': str(nwb.subject.subject_id),
    ...
}
```

```python
mice = sorted({r['mouse'] for r in sessions})
...
'subjects': mice,
'subject_idx': np.array([mice.index(r['mouse']) for r in sessions], dtype=np.int64),
```

iii. The notes say mouse names were used because they are the paper-facing animal identifiers, while numeric `subject_id` was still preserved in session metadata.

## 1-c. How are the data split into sessions?

i. The agent treated one NWB file as one session. After conversion, it kept only sessions that passed its curation rules, then sorted surviving sessions by `identifier` before assembly.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
kept = [r for r in results if r.get('rejected') is None]
...
sessions = sorted(sessions, key=lambda r: r['identifier'])
```

iii. The notes describe the dataset as one NWB file per recording session, so the file boundary defines the session boundary. The agent additionally applied paper-style session selection criteria before keeping a session.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. The agent read per-trial `start_time`, `stop_time`, and other trial columns from `nwb.intervals['trials']`, and asserted that the number of go cues matched the number of trial rows.

ii.
```python
trials = nwb.intervals['trials']
...
'start_time': np.asarray(trials['start_time'][:], dtype=np.float64),
'stop_time': np.asarray(trials['stop_time'][:], dtype=np.float64),
...
'go': np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64),
```

```python
n_trials_all = len(raw['start_time'])
assert len(go) == n_trials_all, (
    f"{raw['identifier']}: {len(go)} go cues for {n_trials_all} trials")
```

iii. The notes say the trials table is the native behavioral trial structure and that `go_start_times` is one-per-trial, which makes the trial split unambiguous.

## 1-e. How are trials filtered based on quality controls?

i. The agent kept early-lick, ignore, and photostimulation trials, but dropped trials that were outside electrophysiology recording coverage (`obs_intervals`), trials with `auto_water` or `free_water`, and trials with no video frame in the analysis window. It then rejected entire sessions if too few trials remained or if paper-style behavioral-performance thresholds were not met.

ii.
```python
not_recorded = ~raw['recorded']
keep = raw['recorded'] & (raw['auto_water'] == 0) & (raw['free_water'] == 0)
```

```python
edges_all = window_edges(go)
n_frames_all = (np.searchsorted(raw['video_t'], edges_all[:, -1])
                - np.searchsorted(raw['video_t'], edges_all[:, 0]))
no_video = keep & (n_frames_all == 0)
keep &= n_frames_all > 0
trial_idx = np.flatnonzero(keep)
```

```python
elif len(trial_idx) < 2:
    reject = f'only {len(trial_idx)} usable trials'
elif performance <= MIN_PERFORMANCE:
    reject = f'performance {performance:.3f} <= {MIN_PERFORMANCE}'
elif n_left < MIN_CORRECT_PER_DIRECTION or n_right < MIN_CORRECT_PER_DIRECTION:
    reject = f'correct trials L={n_left} R={n_right} < {MIN_CORRECT_PER_DIRECTION}'
```

iii. The notes justify this as a mix of reference-paper curation and task-driven exceptions: keep early-lick/ignore/photostim because they are decoder variables, remove `obs_intervals` gaps because they yield all-zero spikes, remove auto/free-water as non-standard behavior, and remove no-video trials because “not visible” would otherwise mean “not measured.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `nwb.units['spike_times']` for units whose `classification` is `'good'`, with trial alignment determined by `BehavioralEvents/go_start_times`.

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.flatnonzero(classification == 'good')
```

```python
'go': np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64),
```

```python
spike_index = units['spike_times']
for i, unit in enumerate(raw['good_units']):
    counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
```

iii. The agent’s notes say this is electrophysiology, not imaging, so the only neural signal to use is spike times from QC-passing units, aligned to the go cue.

## 2-b. How is the `neural` data processed?

i. The agent converted spike times to 50 ms binned firing rates in Hz. For each good unit, it counted spikes in each go-cue-relative bin via `np.searchsorted`, then divided counts by 0.05 s.

ii.
```python
def bin_spike_times(spike_times, edges_abs):
    pos = np.searchsorted(spike_times, edges_abs.ravel()).reshape(edges_abs.shape)
    return np.diff(pos, axis=1).astype(np.int32)
```

```python
counts = np.empty((n_good, n_trials, N_BINS), dtype=np.int32)
for i, unit in enumerate(raw['good_units']):
    counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
neural = np.ascontiguousarray(
    counts.transpose(1, 0, 2).astype(np.float32) / BIN_WIDTH)
```

iii. The notes explicitly cite the reference firing-rate convention: count per bin divided by bin width, with no smoothing or baseline subtraction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent kept only units with `units['classification'] == 'good'`. Sessions with zero such units were rejected.

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.flatnonzero(classification == 'good')
```

```python
if n_good == 0:
    reject = 'no good units'
```

iii. The notes justify this as the NWB materialization of the reference QC classifier output, and explicitly say not to re-threshold the individual quality metrics.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligned every trial to go-cue onset by constructing absolute bin edges as `go_time + BIN_EDGES` for each trial, then binning spikes directly against those absolute edges.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_EDGES = OFF_START + BIN_WIDTH * np.arange(N_BINS + 1)
```

```python
def window_edges(go_times):
    return go_times[:, None] + BIN_EDGES[None, :]
```

```python
go = go[trial_idx]
edges = window_edges(go)
```

iii. The notes say all NWB timestamps already share a session-absolute clock, so alignment only requires subtracting the go cue implicitly via go-cue-relative bin edges.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 80 non-overlapping 50 ms bins spanning -2.5 s to +1.5 s around the go cue. Raw spikes are rebinned onto that grid; no further resampling is applied.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_WIDTH = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_WIDTH))
BIN_EDGES = OFF_START + BIN_WIDTH * np.arange(N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
```

iii. The notes say the 50 ms non-overlapping binning is a deliberate task-driven deviation from the method paper’s 40 ms / 3.4 ms sliding bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times` and the trial go cue. For each trial, the agent took the last sample-epoch onset at or before the go cue as the relevant tone onset.

ii.
```python
'sample_start': np.asarray(events['sample_start_times'].timestamps[:], dtype=np.float64),
```

```python
tone_idx = np.searchsorted(raw['sample_start'], go, side='right') - 1
tone = raw['sample_start'][tone_idx]
```

iii. The notes justify “last sample onset before go” by the task structure: early licks replay the sample/delay epoch, so there can be multiple sample onsets in a trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The value is continuous and time-varying. For each trial/bin, the agent computed seconds since tone onset as the bin center relative to the go cue plus the trial-specific go-minus-tone offset.

ii.
```python
time_from_tone = (BIN_CENTERS[None, :] + (go - tone)[:, None]).astype(np.float32)
```

iii. The notes say this is the direct way to express the go-cue-centered neural grid in tone-centered units without resampling anything else.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 80 go-cue-centered bin centers used for neural binning, so input bin `k` and neural bin `k` refer to the same time interval.

ii.
```python
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
...
time_from_tone = (BIN_CENTERS[None, :] + (go - tone)[:, None]).astype(np.float32)
```

iii. The notes explicitly say the bin center is used for time-valued inputs so that the input array is on the same grid as the firing rates.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The agent derived photostimulation from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, then mapped those absolute stimulation intervals back to retained trials using `start_time`.

ii.
```python
'photostim_start': np.asarray(events['photostim_start_times'].timestamps[:], dtype=np.float64),
'photostim_stop': np.asarray(events['photostim_stop_times'].timestamps[:], dtype=np.float64),
```

```python
stim_trial = np.searchsorted(raw['start_time'], raw['photostim_start'],
                             side='right') - 1
position = np.searchsorted(trial_idx, stim_trial)
```

iii. The notes justify this as the NWB equivalent of the reference photostimulation timing, using actual event timestamps instead of parsing relative-onset strings for the decoder input itself.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The agent made `photostim` a binary time series. A bin was set to 1 if the 50 ms bin interval overlapped the stimulation interval `[photostim_start, photostim_stop)`, and 0 otherwise.

ii.
```python
photostim = np.zeros((n_trials, N_BINS), dtype=np.float32)
...
for row, on, off in zip(position[in_kept], raw['photostim_start'][in_kept],
                        raw['photostim_stop'][in_kept]):
    overlap = (edges[row, :-1] < off) & (edges[row, 1:] > on)
    photostim[row, overlap] = 1.0
```

iii. The notes say the task asks whether photostimulation is on over time, so interval overlap is the natural binarization rule.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is compared against the same absolute bin edges used to bin spikes, so it is aligned to the same go-cue-relative trial window.

ii.
```python
edges = window_edges(go)
...
overlap = (edges[row, :-1] < off) & (edges[row, 1:] > on)
```

iii. The notes treat spikes, events, and video as living on the same session clock, so no separate offset correction is applied.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read directly from one NWB column. The agent derived it from `trial_instruction` and `outcome`.

ii.
```python
'instruction': np.asarray(trials['trial_instruction'][:]),
'outcome': np.asarray(trials['outcome'][:]),
```

```python
instruction = raw['instruction'][trial_idx]
outcome = raw['outcome'][trial_idx]
```

iii. The notes justify this by the task semantics: hit means lick the instructed side, miss means lick the opposite side, and ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The agent encoded choice as `0=left`, `1=right`, `2=no lick`, derived miss trials as the opposite of the instruction, and repeated the per-trial value across all 80 bins.

ii.
```python
CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
```

```python
choice = np.where(instruction == 'left', CHOICE_LEFT, CHOICE_RIGHT)
choice = np.where(outcome == 'miss', 1 - choice, choice)
choice = np.where(outcome == 'ignore', CHOICE_NOLICK, choice)
...
outputs[:, 0, :] = choice[:, None]
```

iii. The notes explicitly state that “choice on error trials is the opposite of the instruction,” matching the behavioral interpretation of `miss`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trials-table `outcome` column.

ii.
```python
'outcome': np.asarray(trials['outcome'][:]),
```

```python
outcome = raw['outcome'][trial_idx]
```

iii. The notes say the NWB trials table already stores the exact three requested outcome classes.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent mapped `ignore`, `miss`, and `hit` to `0`, `1`, and `2`, then repeated the per-trial code across all time bins.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
```

```python
outcome_code = np.array([OUTCOME_CODE[o] for o in outcome], dtype=np.int8)
...
outputs[:, 1, :] = outcome_code[:, None]
```

iii. The notes justify this as a direct categorical encoding of the raw trial label.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trials-table `early_lick` column.

ii.
```python
'early_lick': np.asarray(trials['early_lick'][:]),
```

```python
early = raw['early_lick'][trial_idx]
```

iii. The notes describe this as an explicit per-trial behavioral flag already present in the NWB file.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The agent converted `'early'` to `1` and everything else (`'no early'`) to `0`, then repeated that per-trial label across all 80 bins.

ii.
```python
early_code = (early == 'early').astype(np.int8)
...
outputs[:, 2, :] = early_code[:, None]
```

iii. The notes justify keeping early-lick trials because `early_lick` is itself a decoder output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`: timestamps plus column 1 of the data array for tongue y-position and column 2 for DeepLabCut likelihood.

ii.
```python
tongue = tracking['Camera0_side_TongueTracking']
out['video_t'] = np.asarray(tongue.timestamps[:], dtype=np.float64)
tongue_data = np.asarray(tongue.data[:], dtype=np.float64)
out['tongue_y'] = tongue_data[:, 1]
out['tongue_likelihood'] = tongue_data[:, 2]
```

iii. The notes identify the side-view tongue tracker as the relevant video stream present across the dataset.

## 8-b. How is `output` *Tongue y-position* processed?

i. The agent treated frames with likelihood `> 0.9` as visible, reduced visible tongue y-values into per-trial/per-bin means on the 50 ms grid with `segment_sums`, then computed session-specific 40th and 60th percentiles from all visible retained-trial bin means. Visible bins were classified relative to those two percentile cutoffs.

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
```

```python
visible_values = y_mean[bin_visible]
p_low, p_high = np.percentile(visible_values, [TONGUE_LOW_PCT, TONGUE_HIGH_PCT])
cls = np.where(y_mean < p_low, 0, np.where(y_mean > p_high, 2, 1))
```

iii. The notes justify the 0.9 likelihood threshold by saying the confidence distribution is extremely bimodal, so the exact cutoff is almost immaterial, and justify taking percentiles over retained-trial bin means because those are the values actually being labeled.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The agent used four classes: `0` for below the session 40th percentile, `1` for the 40th-60th percentile band, `2` for above the 60th percentile, and `3` for bins with no visible tongue frame.

ii.
```python
TONGUE_INVISIBLE = 3
```

```python
tongue_class = np.full((n_trials, N_BINS), TONGUE_INVISIBLE, dtype=np.int8)
...
cls = np.where(y_mean < p_low, 0, np.where(y_mean > p_high, 2, 1))
tongue_class[bin_visible] = cls[bin_visible].astype(np.int8)
```

iii. The notes say this matches the requested percentile discretization while reserving a separate class for “not visible.”

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The agent aligned tongue data using the same absolute go-cue-centered bin edges as the neural data. Frame timestamps were reduced directly into those 50 ms bins with `segment_sums`.

ii.
```python
edges = window_edges(go)
...
_, n_visible_bin, y_sum_bin = segment_sums(
    raw['tongue_y'], visible, edges, raw['video_t'])
```

iii. The notes say the camera timestamps live on the same NWB clock as spikes and events, so applying the same bin edges is sufficient for alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handled several missing/partial-data cases by exclusion rather than imputation: sessions with no good units were rejected; trials not covered by `obs_intervals` were dropped; trials with no video in the analysis window were dropped; bins with no visible tongue frame were assigned the explicit “not visible” class; bins outside trial start/stop were left as zero firing rate because the NWB release contains no spikes there.

ii.
```python
if len(units) == 0:
    out['good_units'] = np.zeros(0, dtype=np.int64)
    ...
    out['recorded'] = np.zeros(len(out['start_time']), dtype=bool)
    return out
```

```python
keep = raw['recorded'] & (raw['auto_water'] == 0) & (raw['free_water'] == 0)
...
keep &= n_frames_all > 0
```

```python
tongue_class = np.full((n_trials, N_BINS), TONGUE_INVISIBLE, dtype=np.int8)
```

```python
'known_limitation': 'The NWB release stores spikes only within each trial\'s '
                    '[start_time, stop_time] interval ... so those bins contain no spikes '
                    'and have a firing rate of 0.',
```

iii. The notes argue that missing recordings should be excluded so the conversion does not fabricate data, while absent tongue visibility is a legitimate output state and should therefore remain explicit.

## 10-a. What are the most time-consuming steps of the code?

i. The agent designed the code assuming the expensive parts were per-session NWB reads, spike-time binning across many units/trials, and writing the large pickle. It also treated video reduction as a potential bottleneck and vectorized it.

ii.
```python
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for res in pool.map(_worker, jobs):
        results.append(res)
```

```python
counts = np.empty((n_good, n_trials, N_BINS), dtype=np.int32)
for i, unit in enumerate(raw['good_units']):
    counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
```

```python
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly call out vectorized spike binning, prefix-sum video reduction, and parallel per-session processing as the main performance choices.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorized the main trial-by-bin work, but it still left a Python loop over good units for spike binning and a Python loop over photostimulation episodes when filling the photostim input. Tongue binning itself was vectorized through cumulative sums.

ii.
```python
for i, unit in enumerate(raw['good_units']):
    counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
```

```python
for row, on, off in zip(position[in_kept], raw['photostim_start'][in_kept],
                        raw['photostim_stop'][in_kept]):
    overlap = (edges[row, :-1] < off) & (edges[row, 1:] > on)
    photostim[row, overlap] = 1.0
```

iii. The notes justify the implementation by saying the expensive triple loops were removed and the remaining work was already fast enough under parallel execution.

## 10-c. What processing does the code repeat multiple times?

i. The code does not repeatedly re-open or reprocess sessions, but it does compute trial-window edges twice per session: once for all trials to detect missing video coverage, and again for the retained trials to build the actual aligned arrays. It also computes extra per-session diagnostics and summary statistics beyond the final neural/input/output tensors.

ii.
```python
edges_all = window_edges(go)
...
go = go[trial_idx]
edges = window_edges(go)
```

```python
'n_silent_trials': int(np.sum(counts.sum(axis=(0, 2)) == 0)),
'frac_bins_no_spikes_in_trial': float(np.mean(
    (edges[:, 1:] > raw['stop_time'][trial_idx][:, None])
    | (edges[:, :-1] < raw['start_time'][trial_idx][:, None]))),
```

iii. The notes mostly present the conversion as a single pass, with extra repeated work accepted because it supported curation and validation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Beyond the arrays needed by the decoder, the agent computes substantial extra curation, diagnostics, and metadata: session performance thresholds, video-coverage checks, per-session tongue percentiles/visibility fractions, silent-trial counts, neuron hemisphere/CCF metadata, optional debug traces, plots, and printed summaries. Some of that ends up only in logs or metadata rather than in the main decoder tensors.

ii.
```python
performance, n_left, n_right = session_performance(raw, keep)
```

```python
hemisphere = np.where(raw['ccf'][:, 0] >= 5700.0, 'left', 'right')
```

```python
if collect_debug:
    result['debug'] = {
        'go': go, 'tone': tone, 'edges': edges,
        ...
        'spike_times': [np.asarray(spike_index[int(u)])
                        for u in raw['good_units'][:60]],
    }
```

iii. The notes frame these computations as validation and provenance work rather than decoder necessities, and the script’s `--show-processing` and rich `metadata` fields make that explicit.
