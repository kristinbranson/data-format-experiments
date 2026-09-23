# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI opens the AllenSDK cache at `/app/data`, enumerates the locally cached experiment NWBs, filters the experiment table to those local experiment IDs with `passive == False`, sorts by `ophys_session_id`, and then loads each experiment in each selected session with `get_behavior_ophys_experiment()`.

ii.
```python
def get_cache():
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=CACHE_DIR)

def local_experiment_ids():
    d = os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0', 'behavior_ophys_experiments')
    return sorted(int(re.findall(r'(\d+)', f)[0]) for f in os.listdir(d))

def select_sessions(cache):
    et = cache.get_ophys_experiment_table()
    ids = local_experiment_ids()
    sel = et[et.index.isin(ids) & (~et.passive)].copy()
    sel = sel.sort_values(['ophys_session_id', 'ophys_experiment_id'])
    session_ids = sorted(sel.ophys_session_id.unique().tolist())
    return sel, session_ids

datasets = [cache.get_behavior_ophys_experiment(int(o)) for o in oeids]
```

iii. In `CONVERSION_NOTES.md`, the AI says it must use the AllenSDK cache, that only locally present NWBs can actually be converted, and that passive sessions are excluded because they are not active behavior sessions and would make `trial_outcome` degenerate.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id`. During assembly, the AI builds a globally sorted unique subject list from the converted session results and stores `subject_idx` per session.

ii.
```python
subjects = sorted({r['mouse_id'] for r in results})
subject_to_idx = {s: i for i, s in enumerate(subjects)}

'subject_idx': np.array([subject_to_idx[r['mouse_id']] for r in results], dtype=np.int64),
```

iii. The notes describe `mouse_id` as the canonical SDK animal identifier and explicitly map `ds.metadata['mouse_id']` to `subjects` / `subject_idx`.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique `ophys_session_id`. If a session has multiple simultaneously recorded experiments/planes, the AI groups them into one session and concatenates them along the neuron axis.

ii.
```python
session_ids = sorted(sel.ophys_session_id.unique().tolist())
...
oeids = exps[exps.ophys_session_id == sid].index.tolist()
...
neural = np.concatenate(neural_blocks, axis=0)
```

iii. The notes justify this by saying `ophys_session_id` is the continuous recording session, while per-plane experiments are simultaneous slices of that same session.

## 1-d. How are the data split into trials?

i. Trials come from `ds0.trials`, but instead of using each trial’s full `start_time` to `stop_time`, the AI creates a fixed 21-bin window from `-2.25 s` to `+3.0 s` around `change_time` for each selected trial.

ii.
```python
trials = ds0.trials
sel = select_trials(trials)
change_times = sel.change_time.values.astype(np.float64)
edges = bin_edges_for_trials(change_times)
...
OFF_START = -2.25
OFF_END = 3.00
BIN_SIZE = 0.25
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

iii. The notes say `change_time` is the common event for go and catch trials, and that this window captures 3 flash cycles before change, the change flash, and 3 flash cycles after change without leaking into neighboring trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only go or catch trials, excludes aborted and auto-rewarded trials, requires non-null `change_time`, drops sessions with fewer than 2 usable trials, drops trials with incomplete neural/behavior bins, drops sessions with no eye tracking, and drops trials with long behavioral sensor gaps.

ii.
```python
def select_trials(trials):
    m = (trials.go | trials.catch) & (~trials.aborted) & (~trials.auto_rewarded)
    m &= trials.change_time.notna()
    return trials[m]

if len(sel) < 2:
    return {'session_id': session_id, 'skip': f'only {len(sel)} usable trials'}

if len(et) == 0:
    return {'session_id': session_id, 'skip': 'no eye tracking'}

trial_ok = np.isfinite(neural).all(axis=(0, 2))
...
trial_ok &= ok
...
if trial_ok.sum() < 2:
    return {'session_id': session_id,
            'skip': f'only {int(trial_ok.sum())} trials with complete data'}
```

iii. The notes say the go/catch filter follows the decoder task, and justify the extra missing-data rules by arguing pupil is a required output and that long sensor dropouts should be dropped while short gaps should be interpolated.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. By default the neural data comes from `ds.dff_traces.dff`. The script also allows `events` or `filtered_events`, but the default and final choice is `dff`.

ii.
```python
NEURAL_SIGNAL = 'dff'
...
if signal_name == 'dff':
    traces = np.vstack(ds.dff_traces.dff.values).astype(np.float64)
    index = ds.dff_traces.index.values
else:
    traces = np.vstack(ds.events[signal_name].values).astype(np.float64)
    index = ds.events.index.values
```

iii. The notes state that the AI initially considered `events` because the paper used them, but switched to `dff` after empirical decoder comparisons showed much better decoding at 250 ms resolution.

## 2-b. How is the `neural` data processed?

i. For each plane/experiment, the AI bins the neural traces by averaging all frames whose `ophys_timestamps` fall into each 250 ms trial bin, reshapes those bins into `(neurons, trials, 21)`, and concatenates all planes for the session along the neuron axis.

ii.
```python
means, counts = bin_mean(traces, ts - neural_lag, edges_flat)
block = means[:, keep].reshape(traces.shape[0], len(sel), NBINS)
...
neural_blocks.append(block)
...
neural = np.concatenate(neural_blocks, axis=0)
```

iii. The notes justify this by saying one uniform bin size is needed across 31 Hz and 11 Hz rigs, and that 250 ms preserves flash structure while avoiding empty bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply additional neuron-level QC beyond AllenSDK release filtering. It keeps all released cells, but drops trials whose binned neural window contains missing values.

ii.
```python
trial_ok = np.isfinite(neural).all(axis=(0, 2))
...
'neuron_selection': (
    'all cells in the released NWB files (cell_specimen_table.valid_roi is True for every cell; '
    'ROI filtering, demixing and neuropil correction were applied by the Allen pipeline)'),
```

iii. The notes say all released cells already have Allen pipeline QC applied and that no further ROI filtering is justified.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `trials.change_time`, not `trial_start`. The AI builds absolute-time bins around each trial’s change time, then assigns frames to those bins using `ophys_timestamps`.

ii.
```python
change_times = sel.change_time.values.astype(np.float64)
edges = bin_edges_for_trials(change_times)
...
means, counts = bin_mean(traces, ts - neural_lag, edges_flat)
```

iii. The notes say `change_time` is exactly the onset of the change/sham-change flash and is the event shared by all included trials, so the trial grid is phase-locked to that event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 250 ms bins, and the neural signal is explicitly rebinned from the native ophys frame times into those bins.

ii.
```python
BIN_SIZE = 0.25
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
...
'time_bin_size': BIN_SIZE * 1000.0,
```

iii. The notes justify 250 ms as a common cross-rig bin size that divides the 750 ms flash cycle into three bins and still contains multiple frames even for 11 Hz Multiscope recordings.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The AI derives image identity from `stimulus_presentations.start_time` and `stimulus_presentations.image_name` within the `change_detection_behavior` stimulus block, not from `initial_image_name` / `change_image_name` in the trials table.

ii.
```python
sp = ds0.stimulus_presentations
spa = sp[sp.stimulus_block_name == STIM_BLOCK]
flash_start = spa.start_time.values.astype(np.float64)
flash_name = spa.image_name.values.astype(object)
```

iii. The notes say this follows the paper’s “image presentation interval” convention and also allows omitted flashes to remain a distinct category.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each bin center is assigned the identity of the most recent flash whose 750 ms interval contains that bin center. Bins outside a valid interval are marked `'none'` and then dropped via `trial_ok`; omitted flashes are kept as the `'omitted'` category. Final labels are mapped to global integer codes during assembly.

ii.
```python
j = np.searchsorted(flash_start, centers_flat, side='right') - 1
valid = (j >= 0) & (j < len(flash_start))
j_clipped = np.clip(j, 0, len(flash_start) - 1)
within = valid & (centers_flat - flash_start[j_clipped] < FLASH_CYCLE + 0.05)
image_name_flat = np.where(within, flash_name[j_clipped], 'none')
image_names = image_name_flat[keep].reshape(len(sel), NBINS)
trial_ok &= ~(image_names == 'none').any(axis=1)
...
image_values = sorted({n for r in results for n in np.unique(r['image_names']) if n != 'omitted'})
image_values = image_values + ['omitted']
image_to_idx = {n: i for i, n in enumerate(image_values)}
```

iii. The notes justify this by citing the paper’s 750 ms “image presentation interval,” keeping gray periods attached to the flash identity, and preserving omissions as their own stimulus condition.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the exact same change-centered 250 ms bin centers used for the neural data, so every trial has one image-identity label per neural bin.

ii.
```python
centers_flat = (edges_flat[:-1] + edges_flat[1:]) / 2.0
...
image_name_flat = np.where(within, flash_name[j_clipped], 'none')
image_names = image_name_flat[keep].reshape(len(sel), NBINS)
...
out = np.stack([
    img[t],
    r['image_change'][t],
    r['run_bin'][t],
    r['pupil_bin'][t],
    np.full(NBINS, r['outcome'][t], dtype=np.int64),
], axis=0)
```

iii. The notes say all streams are put onto the same absolute-time bin grid anchored to `change_time`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the trial table’s change timing and change flag, specifically `sel.change_time` and `sel.is_change`.

ii.
```python
change_times = sel.change_time.values.astype(np.float64)
...
is_change = sel.is_change.values.astype(bool)
```

iii. The notes say `change_time` exactly matches the change/sham-change flash onset and that catch trials have sham changes rather than real identity changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI computes image change from bin centers relative to `change_time`; it is `1` only for bins whose centers fall in the first 500 ms after the change, and only on true change trials.

ii.
```python
CHANGE_WINDOW = 0.5
...
rel_centers = centers - change_times[:, None]
image_change = ((rel_centers >= 0) & (rel_centers < CHANGE_WINDOW)
                & is_change[:, None]).astype(np.int64)
```

iii. The notes justify the 500 ms window as the representable width closest to the paper’s 400 ms post-stimulus decoding window, and mention it was chosen after empirical decoder comparisons.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is thresholded into a binary categorical variable with labels `0 = no_change` and `1 = change`.

ii.
```python
image_change = ((rel_centers >= 0) & (rel_centers < CHANGE_WINDOW)
                & is_change[:, None]).astype(np.int64)
...
'output_values': [image_values, ['no_change', 'change'],
                  QUANTILE_NAMES, QUANTILE_NAMES, OUTCOMES],
```

iii. The notes describe `image_change` as a binary indicator and explicitly define `1` as the short post-change window on go trials and `0` elsewhere.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned to the same change-centered 250 ms bins as the neural data by evaluating each neural/bin center relative to the trial’s `change_time`.

ii.
```python
rel_centers = centers - change_times[:, None]
image_change = ((rel_centers >= 0) & (rel_centers < CHANGE_WINDOW)
                & is_change[:, None]).astype(np.int64)
```

iii. The notes say the full bin grid for all outputs is shared with the neural data and is anchored to `change_time`.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds0.running_speed.timestamps` and `ds0.running_speed.speed`.

ii.
```python
rs = ds0.running_speed
run_ts = rs.timestamps.values.astype(np.float64)
run_v = interpolate_nans(rs.speed.values)
```

iii. The notes identify `running_speed` as the standard AllenSDK locomotion stream and cite the whitepaper’s description of it as filtered wheel speed in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI first fills NaNs in the raw speed trace by linear interpolation if needed, then averages running speed into the same 250 ms trial bins used for neural data, interpolates across short empty binned gaps, drops trials with long gaps, and only later discretizes the binned values.

ii.
```python
run_v = interpolate_nans(rs.speed.values)
...
run_binned, _ = bin_mean(run_v, run_ts, edges_flat)
run_binned = run_binned[keep].reshape(len(sel), NBINS)
run_binned, ok = fill_short_gaps(run_binned, centers, MAX_SENSOR_GAP)
trial_ok &= ok
```

iii. The notes justify this by saying all outputs must share the same trial bin grid, and that short camera/sensor dropouts should be interpolated rather than forcing whole-session exclusion.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five equal-percentile bins computed per session over exactly the exported binned timepoints.

ii.
```python
def quantile_bins(values, nq=NQUANTILES):
    edges = np.percentile(values, np.linspace(0, 100, nq + 1)[1:-1])
    labels = np.searchsorted(edges, values, side='right').astype(np.int64)
    return np.clip(labels, 0, nq - 1), edges

run_bin, run_edges = quantile_bins(run_binned.ravel())
```

iii. The notes justify per-session quintiles by arguing that running distributions differ strongly across mice and sessions, so per-session percentiles make the five labels carry comparable within-session meaning and guarantee balanced occupancy.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by binning it into the exact same change-centered 250 ms bins used for neural data.

ii.
```python
run_binned, _ = bin_mean(run_v, run_ts, edges_flat)
run_binned = run_binned[keep].reshape(len(sel), NBINS)
```

iii. The notes say every stream is resampled onto the same absolute-time bins anchored to `change_time`.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI derives pupil diameter from `ds0.eye_tracking.pupil_area` and `timestamps`, not from `pupil_width`. It converts area to diameter by `2*sqrt(area/pi)`.

ii.
```python
et = ds0.eye_tracking
eye_ts = et.timestamps.values.astype(np.float64)
pupil_area = et.pupil_area.values.astype(np.float64)
pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The notes cite the whitepaper’s definition that pupil area is the area of a circle whose diameter is the ellipse major axis, and explicitly justify `2*sqrt(area/pi)` from that description.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. After converting `pupil_area` to diameter, the AI linearly interpolates over blink-related NaNs, bins the resulting trace into the same 250 ms trial bins as neural data, fills short missing binned gaps, drops trials with long gaps, and then discretizes the binned values.

ii.
```python
pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diam = interpolate_nans(pupil_diam_raw)
...
pupil_binned, _ = bin_mean(pupil_diam, eye_ts, edges_flat)
pupil_binned = pupil_binned[keep].reshape(len(sel), NBINS)
pupil_binned, ok = fill_short_gaps(pupil_binned, centers, MAX_SENSOR_GAP)
trial_ok &= ok
```

iii. The notes say blink frames appear as NaNs and are short enough to interpolate over, and that output variables must all live on the common trial bin grid.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five equal-percentile bins computed per session over the exported binned pupil values.

ii.
```python
pupil_bin, pupil_edges = quantile_bins(pupil_binned.ravel())
pupil_bin = pupil_bin.reshape(len(sel), NBINS)
```

iii. The notes justify this by saying absolute pupil scale varies across sessions/mice because of eye size and camera geometry, so per-session quintiles are more meaningful.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by binning it onto the same change-centered 250 ms trial bins used for the neural data.

ii.
```python
pupil_binned, _ = bin_mean(pupil_diam, eye_ts, edges_flat)
pupil_binned = pupil_binned[keep].reshape(len(sel), NBINS)
```

iii. The notes state that all data streams are synchronized to the same session clock and are resampled to the same exported bin grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']

def outcome_labels(sel):
    lab = np.full(len(sel), -1, dtype=np.int64)
    for i, name in enumerate(OUTCOMES):
        lab[sel[name].values.astype(bool)] = i
    return lab
```

iii. The notes explicitly map `ds.trials.hit/miss/false_alarm/correct_reject` to `trial_outcome`.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps each trial to an integer in the fixed order `['hit', 'miss', 'false_alarm', 'correct_reject']`, then broadcasts that single trial-level label across all 21 bins of the trial in the final output array.

ii.
```python
outcome = outcome_labels(sel)
...
out = np.stack([
    img[t],
    r['image_change'][t],
    r['run_bin'][t],
    r['pupil_bin'][t],
    np.full(NBINS, r['outcome'][t], dtype=np.int64),
], axis=0)
```

iii. The notes say the format expects one `(n_output, n_timepoints)` array per trial, so static trial outcome is broadcast over time.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data by interpolating NaNs in 1-D behavioral traces, filling short empty binned behavioral gaps, dropping trials with longer gaps, dropping sessions with no eye tracking or all-NaN pupil traces, rejecting sessions with overlapping fixed windows or too few complete trials, and checking that binned values stay within the raw-signal range.

ii.
```python
def interpolate_nans(x):
    ...
    return np.interp(idx, idx[good], x[good])

def fill_short_gaps(binned, centers, max_gap):
    ...
    trial_ok = ok.reshape(binned.shape).all(axis=1)
    return filled.reshape(binned.shape), trial_ok

def within_range(binned, raw, tol=1e-6):
    return (np.nanmin(binned) >= np.nanmin(raw) - tol) and (np.nanmax(binned) <= np.nanmax(raw) + tol)

if len(et) == 0:
    return {'session_id': session_id, 'skip': 'no eye tracking'}
if pupil_diam is None:
    return {'session_id': session_id, 'skip': 'pupil all NaN'}
if np.any(np.diff(edges_flat) <= 0):
    return {'session_id': session_id, 'skip': 'overlapping trial windows'}
```

iii. The notes say these rules were added after finding real edge cases during validation, especially a `reduceat` binning bug and short pupil-camera dropouts that should be interpolated rather than forcing session loss.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identifies experiment/session loading through AllenSDK as the dominant cost; the actual binning and output construction are much cheaper.

ii.
```python
datasets = [cache.get_behavior_ophys_experiment(int(o)) for o in oeids]
timing['load'] = time.time() - t0
...
with Pool(args.nproc) as pool:
```

iii. The notes say NWB loads take about `3.5 s / experiment` and dominate wall clock, while binning is only about `0.05-0.1 s / session`.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the main trial/bin binning work with `searchsorted` + `reduceat`. Remaining non-vectorized loops are mostly the per-dataset plane loop and the final per-trial assembly loop.

ii.
```python
idx = np.searchsorted(timestamps, edges_flat, side='left')
counts = np.diff(idx)
sums = np.add.reduceat(v[:, :stop], starts, axis=1)
...
for ds in datasets:
    ...
for t in range(n_trials):
    neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :]))
    ...
```

iii. The notes explicitly say one reason for the implementation is “no Python loop over trials or bins” and describe the vectorized binning as a major speedup.

## 9-c. What processing does the code repeat multiple times?

i. The AI repeats some work across sessions and during assembly: each worker/session reopens the cache and reloads experiments, image names are first stored as strings and later remapped to integers, and trial-level outputs are reconstructed again in `assemble()`.

ii.
```python
def convert_session(args):
    ...
    cache = get_cache()
    ...
    datasets = [cache.get_behavior_ophys_experiment(int(o)) for o in oeids]
...
'image_names': image_names,
...
img = np.vectorize(image_to_idx.get)(r['image_names']).astype(np.int64)
...
for t in range(n_trials):
    ...
    out = np.stack([
        img[t],
        r['image_change'][t],
        r['run_bin'][t],
        r['pupil_bin'][t],
        np.full(NBINS, r['outcome'][t], dtype=np.int64),
    ], axis=0)
```

iii. The notes emphasize speedups in binning and multiprocessing, but these repeated steps are still present in the code path.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI keeps several intermediate items that are not used by the final decoder dataset, including `cell_ids`, `change_times`, `trial_ids`, raw binned continuous running/pupil arrays, and detailed timing information. Optional `--show-processing` plotting also computes extra diagnostics that are not part of the final dataset.

ii.
```python
result = {
    ...
    'cell_ids': cell_ids,
    'change_times': change_times,
    'trial_ids': sel.index.values,
    'run_edges': run_edges,
    'pupil_edges': pupil_edges,
    'run_binned': run_binned,
    'pupil_binned': pupil_binned,
    'timing': timing,
    'total_time': time.time() - t_start,
}
...
if show_processing:
    plot_processing(...)
```

iii. The notes present these as validation/debugging aids rather than final-task requirements, and most of them are not preserved in the exported `neural` / `input` / `output` arrays used downstream.
