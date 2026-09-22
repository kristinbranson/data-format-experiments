# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent opens `/app/data` with `VisualBehaviorOphysProjectCache.from_s3_cache`, intersects the SDK experiment table with locally present experiment files, excludes passive experiments, groups experiments into sessions, and loads every plane through `get_behavior_ophys_experiment`. Full mode parallelizes sessions with a process pool.

ii.
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=CACHE_DIR)
sel = et[et.index.isin(ids) & (~et.passive)].copy()
datasets = [cache.get_behavior_ophys_experiment(int(o)) for o in oeids]
```

iii. The notes justify the SDK as required, local-file discovery as protection against downloading absent release files, passive-session exclusion because trial outcome is degenerate there, and multiprocessing because NWB loading dominates runtime.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `mouse_id` values among successfully converted sessions; each session receives an index into that list.

ii.
```python
subjects = sorted({r['mouse_id'] for r in results})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_to_idx[r['mouse_id']] for r in results])
```

iii. The agent treats the SDK mouse identifier as the animal identifier and constructs subjects after failed sessions are removed.

## 1-c. How are the data split into sessions?

i. A session is a unique `ophys_session_id`; all simultaneous experiments/planes with that ID are loaded together and concatenated along the neuron axis.

ii.
```python
session_ids = sorted(sel.ophys_session_id.unique().tolist())
oeids = selected[selected.ophys_session_id == sid].index.tolist()
neural = np.concatenate(neural_blocks, axis=0)
```

iii. The notes correctly distinguish an experiment (one plane) from an ophys session and preserve simultaneous planes as one decoder session.

## 1-d. How are the data split into trials?

i. Trials come from `ds0.trials`, but each exported trial is a fixed 5.25 s window from -2.25 to +3.0 s around `change_time`, represented by 21 bins rather than the SDK `start_time`–`stop_time` interval.

ii.
```python
sel = select_trials(trials)
change_times = sel.change_time.values.astype(np.float64)
edges = change_times[:, None] + (OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
```

iii. The agent chose a change-aligned fixed window to give every trial identical shape and include three flash cycles before and four after the change.

## 1-e. How are trials filtered based on quality controls?

i. It retains explicit go/catch trials with valid change times, excludes aborted and auto-rewarded trials, rejects overlapping windows and trials with incomplete neural/stimulus data or long sensor gaps, and drops sessions with fewer than two complete trials.

ii.
```python
m = (trials.go | trials.catch) & (~trials.aborted) & (~trials.auto_rewarded)
m &= trials.change_time.notna()
trial_ok = np.isfinite(neural).all(axis=(0, 2))
trial_ok &= ~(image_names == 'none').any(axis=1)
```

iii. The required trial exclusions are explicit; extra completeness checks are justified as preventing invented or missing decoder values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The delivered default uses `BehaviorOphysExperiment.dff_traces.dff`; command-line alternatives allow `events` or `filtered_events`.

ii.
```python
if signal_name == 'dff':
    traces = np.vstack(ds.dff_traces.dff.values).astype(np.float64)
else:
    traces = np.vstack(ds.events[signal_name].values).astype(np.float64)
```

iii. Although the paper uses detected events, the agent ultimately selected dF/F to match the human/reference conversion and documented the alternatives.

## 2-b. How is the `neural` data processed?

i. Each cell's trace is averaged within 250 ms bins, optionally after a fitted neural time lag (zero in the full run), then planes are concatenated. No normalization is added.

ii.
```python
means, counts = bin_mean(traces, ts - neural_lag, edges_flat)
block = means[:, keep].reshape(traces.shape[0], len(sel), NBINS)
neural = np.concatenate(neural_blocks, axis=0)
```

iii. The notes say bin means preserve dF/F units and provide multiple ophys samples per bin; upstream SDK processing already performs demixing, neuropil correction, and ROI curation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. All released cells are retained. The code asserts timestamp/trace length agreement, range-checks bin means, and drops trials containing nonfinite binned neural values.

ii.
```python
assert traces.shape[1] == len(ts)
if not within_range(block, traces): ...
trial_ok = np.isfinite(neural).all(axis=(0, 2))
```

iii. The agent found released `valid_roi` values already true and therefore relied on Allen pipeline cell QC rather than imposing another cell filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural bins are aligned to `trials.change_time`, from -2.25 to +3.0 s. A lag-estimation facility exists, but the exported run used zero lag.

ii.
```python
edges = bin_edges_for_trials(change_times)
means, counts = bin_mean(traces, ts - neural_lag, edges_flat)
```

iii. The agent considered change onset the behaviorally meaningful common event and tested rather than applying an unsupported calcium lag correction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 250 ms, with explicit mean rebinning of native ~11 Hz ophys samples into 21 bins.

ii.
```python
BIN_SIZE = 0.25
'time_bin_size': BIN_SIZE * 1000.0
```

iii. The agent chose one third of the 750 ms flash cycle, citing stable aggregation and at least two ophys frames per bin.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `stimulus_presentations.start_time`, `image_name`, and `stimulus_block_name`, not from trial initial/change image columns.

ii.
```python
spa = sp[sp.stimulus_block_name == STIM_BLOCK]
flash_start = spa.start_time.values.astype(np.float64)
flash_name = spa.image_name.values.astype(object)
```

iii. The agent used the flash table to capture every actual flash, including omissions, within the longer peri-change window.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each bin center is assigned the most recent flash name if it lies within approximately one 750 ms flash cycle; names are then globally mapped to categorical integers, with omitted as a category.

ii.
```python
j = np.searchsorted(flash_start, centers_flat, side='right') - 1
within = valid & (centers_flat - flash_start[j_clipped] < FLASH_CYCLE + 0.05)
image_name_flat = np.where(within, flash_name[j_clipped], 'none')
img = np.vectorize(image_to_idx.get)(r['image_names']).astype(np.int64)
```

iii. The notes invoke the paper's “image presentation interval” convention and preserve omitted flashes rather than fabricating an image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated at the centers of the exact absolute-time bins used to average neural activity.

ii.
```python
centers = centers_flat[keep].reshape(len(sel), NBINS)
image_names = image_name_flat[keep].reshape(len(sel), NBINS)
```

iii. Shared bin centers and synchronized session clocks are the alignment basis.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses trial `change_time` and `is_change` (go versus sham/catch status).

ii.
```python
is_change = sel.is_change.values.astype(bool)
rel_centers = centers - change_times[:, None]
```

iii. This distinguishes real changes from catch-trial sham changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code creates a binary indicator for bin centers in the first 0.5 s after a real change.

ii.
```python
image_change = ((rel_centers >= 0) & (rel_centers < CHANGE_WINDOW)
                & is_change[:, None]).astype(np.int64)
```

iii. The agent interpreted “right after” as a short transient covering the new-image presentation and beginning of gray, rather than the entire post-change trial.

## 4-c. How is `output` *Image change* thresholded into categories?

i. The Boolean temporal and go-trial conditions directly yield category 0 (`no_change`) or 1 (`change`); no learned threshold is used.

ii.
```python
'output_values': [image_values, ['no_change', 'change'], ...]
```

iii. The source event is intrinsically binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is computed from the same bin-center timestamps as neural binning, relative to the same trial change time.

ii.
```python
rel_centers = centers - change_times[:, None]
```

iii. Shared centers make the label and neural matrix timepoint-for-timepoint aligned.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `ds0.running_speed.timestamps` and the SDK-processed `speed` values.

ii.
```python
rs = ds0.running_speed
run_ts = rs.timestamps.values.astype(np.float64)
run_v = interpolate_nans(rs.speed.values)
```

iii. The notes identify this as the Allen pipeline's filtered wheel speed in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Raw NaNs are interpolated, samples are averaged within the common 250 ms bins, short empty-bin gaps are interpolated, and values are converted to quintiles separately within each session.

ii.
```python
run_binned, _ = bin_mean(run_v, run_ts, edges_flat)
run_binned, ok = fill_short_gaps(run_binned, centers, MAX_SENSOR_GAP)
run_bin, run_edges = quantile_bins(run_binned.ravel())
```

iii. Per-session quintiles were chosen to balance classes within each decoder session; bin averaging avoids point-sampling a 60 Hz signal.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four session-specific 20th/40th/60th/80th percentile edges create labels 0–4.

ii.
```python
edges = np.percentile(values, np.linspace(0, 100, nq + 1)[1:-1])
labels = np.searchsorted(edges, values, side='right').astype(np.int64)
```

iii. This implements five equal-percentile bins over exported timepoints.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running samples are independently averaged over the identical absolute bin edges used for neural traces.

ii.
```python
run_binned, _ = bin_mean(run_v, run_ts, edges_flat)
```

iii. Hardware-synchronized timestamps allow direct common-clock binning.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking `timestamps` and `pupil_area`; diameter is reconstructed as the diameter of the corresponding circle.

ii.
```python
pupil_area = et.pupil_area.values.astype(np.float64)
pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The notes cite the whitepaper definition: pupil area is based on a circle whose diameter is the ellipse major axis, and blink frames are NaN.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area is converted to diameter, blink/missing NaNs are linearly interpolated, data are averaged into 250 ms bins, short gaps are filled, and values are session-wise quintile binned.

ii.
```python
pupil_diam = interpolate_nans(pupil_diam_raw)
pupil_binned, _ = bin_mean(pupil_diam, eye_ts, edges_flat)
pupil_binned, ok = fill_short_gaps(pupil_binned, centers, MAX_SENSOR_GAP)
pupil_bin, pupil_edges = quantile_bins(pupil_binned.ravel())
```

iii. Interpolation treats blink NaNs as artifacts while the two-second maximum prevents inventing long missing stretches.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four session-specific percentile edges divide binned diameters into five equal-frequency labels 0–4.

ii.
```python
pupil_bin, pupil_edges = quantile_bins(pupil_binned.ravel())
```

iii. The agent used the same balanced-class rationale as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye-tracking samples are averaged over the same absolute 250 ms trial bins as neural activity.

ii.
```python
pupil_binned, _ = bin_mean(pupil_diam, eye_ts, edges_flat)
```

iii. This uses the SDK-synchronized shared session clock.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It derives outcome from the trials-table Boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
for i, name in enumerate(OUTCOMES):
    lab[sel[name].values.astype(bool)] = i
```

iii. These are the SDK's canonical mutually exclusive outcome fields for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are mapped in fixed order to integers 0–3, invalid outcomes cause the session to be skipped, and the valid scalar label is repeated across all 21 timepoints.

ii.
```python
if (outcome < 0).any():
    return {'session_id': session_id, 'skip': 'trial without an outcome label'}
np.full(NBINS, r['outcome'][t], dtype=np.int64)
```

iii. Repetition fits the decoder's common time-varying output matrix while representing a static trial property.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN sensors or missing eye tracking skip a session; short NaN/sample gaps are interpolated, gaps over two seconds drop affected trials, incomplete neural/stimulus windows are dropped, invalid outcomes and overlapping trial windows skip sessions, and fewer than two complete trials skips a session.

ii.
```python
if pupil_diam is None: return {'skip': 'pupil all NaN', ...}
pupil_binned, ok = fill_short_gaps(..., MAX_SENSOR_GAP)
trial_ok &= ok
if trial_ok.sum() < 2: return {'skip': ...}
```

iii. The agent explicitly preferred dropping unverifiable data to encoding missingness as a real category, while allowing bounded interpolation for ordinary sensor artifacts.

## 9-a. What are the most time-consuming steps of the code?

i. Loading large experiments through the SDK is dominant; neural binning is next. The code records load, neural, behavior, and stimulus timings and uses multiprocessing in full mode.

ii.
```python
timing['load'] = time.time() - t0
timing['neural'] = time.time() - t0
with Pool(processes=n_workers) as pool:
    results = pool.map(convert_session, jobs)
```

iii. The notes and timing output identify NWB parsing/I/O as the main cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Most time-series work is already vectorized. Remaining loops over planes, outcome fields, sessions, and final trials could partly be removed, especially `np.vectorize(image_to_idx.get)` and per-trial assembly, but are small relative to loading and neural aggregation.

ii.
```python
for ds in datasets: ...
for t in range(n_trials):
    neural_trials.append(...)
```

iii. The agent deliberately vectorized bin searches/reductions across all trials; residual loops primarily construct ragged target-format lists.

## 9-c. What processing does the code repeat multiple times?

i. Each worker independently opens the cache; every stream is passed through similar binning/gap/range checks; assembly traverses results again for image, region, subject, metadata, and list construction. Optional lag estimation can reconvert sample sessions for each lag candidate.

ii.
```python
cache = get_cache()
run_binned, _ = bin_mean(...)
pupil_binned, _ = bin_mean(...)
```

iii. Repetition is largely required by distinct timestamp streams and multiprocessing isolation; lag-search repetition is diagnostic and optional.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `cell_ids`, raw binned running/pupil arrays, change times, trial IDs, detailed timings, and some session fields are retained in intermediate results but not exported. Optional processing plots and lag estimation are diagnostic only. `counts` from neural binning is computed but unused.

ii.
```python
means, counts = bin_mean(traces, ts - neural_lag, edges_flat)
'cell_ids': cell_ids,
'change_times': change_times,
'timing': timing,
```

iii. The agent kept these values for validation, summaries, and diagnostics; the final decoder structure discards them.
