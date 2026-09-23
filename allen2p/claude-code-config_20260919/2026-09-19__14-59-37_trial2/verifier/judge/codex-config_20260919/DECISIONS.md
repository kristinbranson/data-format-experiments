# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local Allen release metadata, intersects it with NWB files actually present, removes passive experiments, groups the remainder by `ophys_session_id`, and loads every NWB plane in each selected session directly with `BehaviorOphysExperiment.from_nwb_path`. Session jobs are processed in a process pool.

ii.
```python
exp = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
have = {int(f.split('_')[-1].split('.')[0]) for f in os.listdir(EXP_DIR)
        if f.endswith('.nwb')}
exp = exp[exp.ophys_experiment_id.isin(have)]
exp = exp[~exp.passive]
experiments = [load_experiment(e) for e in eids]
```

iii. The notes say local loading avoids network fetches, active sessions are required because passive sessions have no meaningful trial outcome, and all locally available active experiments should be used. Parallelism is justified because NWB loading is the main bottleneck.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s from successfully converted sessions; each session receives an index into the sorted subject list.

ii.
```python
subjects = sorted({r['mouse_id'] for r in ok})
'subject_idx': np.array([subjects.index(r['mouse_id']) for r in ok], dtype=np.int64)
```

iii. The agent treats the SDK `mouse_id` as the unique animal identifier and uses a deterministic sorted mapping.

## 1-c. How are the data split into sessions?

i. One session is one `ophys_session_id`. All simultaneously acquired experiments/planes sharing that ID are grouped, and their neurons are concatenated. Final sessions are sorted by session ID.

ii.
```python
groups = exp.groupby('ophys_session_id')
experiment_ids=g.ophys_experiment_id.tolist()
neural = np.concatenate(mats, axis=0) if len(mats) > 1 else mats[0]
ok.sort(key=lambda r: r['ophys_session_id'])
```

iii. The notes cite the whitepaper definition of a session as a continuous recording and explain that multiscope planes share behavior and trial structure.

## 1-d. How are the data split into trials?

i. Trials are SDK `trials` rows for which `go | catch` is true, sorted by start time. Each variable-length trial covers `[start_time, stop_time)` on a new uniform grid.

ii.
```python
sel = trials[(trials.go | trials.catch)].sort_values('start_time')
t_beg, t_stop = float(tr.start_time), float(tr.stop_time)
nbins = int(np.floor((t_stop - t_beg) / BIN_SIZE))
tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
```

iii. The agent says `go | catch` exactly implements the requested inclusion of Go and Catch and exclusion of Aborted and Auto-rewarded trials, while the full behavioral-program trial preserves pre- and post-change context.

## 1-e. How are trials filtered based on quality controls?

i. The code asserts selected trials are neither aborted nor auto-rewarded and have change times; it skips trials shorter than two bins, outside any plane's ophys coverage, or lacking one of the four outcome flags. Sessions with fewer than two usable trials are dropped. It also drops passive sessions and sessions lacking pupil/running data.

ii.
```python
assert not sel.aborted.any() and not sel.auto_rewarded.any()
assert sel.change_time.notna().all()
if nbins < 2: continue
if any(tc[0] < ts[0] or tc[-1] > ts[-1] for ts in plane_ts): continue
else: continue  # go/catch trial with no outcome flag
if len(neural_trials) < 2:
    return {'ophys_session_id': sid, 'error': 'fewer than 2 usable trials'}
```

iii. The notes justify outcome/coverage checks as defensive integrity checks and exclusion of no-eye sessions because pupil diameter is required rather than something to fabricate.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. By default neural data come from each experiment's `dff_traces['dff']`. The command line alternatively permits `events` or `filtered_events` from `ds.events`.

ii.
```python
if trace_name == 'dff':
    tbl = ds.dff_traces
    mat = np.stack(tbl['dff'].values).astype(np.float32)
else:
    tbl = ds.events
    mat = np.stack(tbl[trace_name].values).astype(np.float32)
```

iii. Although the paper used detected events, the notes report a six-session decoder comparison in which dF/F performed best and argue that it is the whitepaper's standard processed calcium signal and is more usable for per-time-bin decoding.

## 2-b. How is the `neural` data processed?

i. Traces are cast to float32, sampled at the nearest native ophys frame for every 1/31-second target-bin center, and concatenated across planes along the neuron axis. No normalization or smoothing is added for the default dF/F trace.

ii.
```python
tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
mats = [mat[:, nearest_index(ts, tc)] for mat, ts in zip(plane_traces, plane_ts)]
neural = np.concatenate(mats, axis=0) if len(mats) > 1 else mats[0]
neural_trials.append(np.ascontiguousarray(neural, dtype=np.float32))
```

iii. The notes say this is effectively one-to-one sampling for 31 Hz rigs and zero-order-hold upsampling for 11 Hz multiscope planes, chosen to satisfy the common-bin-size requirement while merging planes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional cell filtering is performed; every ROI exposed in the selected trace table is kept. Sessions/trials can be excluded for behavioral-data, coverage, or integrity failures.

ii.
```python
mat = np.stack(tbl['dff'].values).astype(np.float32)
plane_cellid.append(np.asarray(tbl.index.values))
```

iii. The agent states that the Allen pipeline already removes invalid ROIs (motion-border, duplicate, dendritic, low-quality, and related cases), so further cell curation would duplicate upstream QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `trials.start_time`. Bin centers are offsets from that event, and each plane contributes its nearest ophys frame at each center.

ii.
```python
tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
mats = [mat[:, nearest_index(ts, tc)] for mat, ts in zip(plane_traces, plane_ts)]
```

iii. The agent argues all NWB streams share the synchronized clock, so sampling everything at common target times with neural values selected by ophys timestamps ensures alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is fixed at `1/31` second, or about 32.258 ms, for every session. Native 31 Hz data are nearest-frame sampled and native 11 Hz multiscope data are repeated across target bins (zero-order hold), so temporal resampling is applied.

ii.
```python
BIN_SIZE = 1.0 / 31.0
nbins = int(np.floor((t_stop - t_beg) / BIN_SIZE))
'time_bin_size': 1000.0 * BIN_SIZE
```

iii. The notes justify a fixed grid because the target format requires a common time-bin size and because it permits concatenating planes with differing native rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from behavior-block `stimulus_presentations`: `start_time`, `end_time`, `image_name`, and `omitted`.

ii.
```python
beh = stim[stim.stimulus_block_name == BEHAVIOR_BLOCK].sort_values('start_time')
flash_start = beh.start_time.values.astype(float)
flash_end = beh.end_time.values.astype(float)
flash_img = beh.image_name.values.astype(str)
flash_omitted = beh.omitted.values.astype(bool)
```

iii. The agent chose the presentation table because the requested label is the image actually present during non-gray screen periods, including blanks and omissions correctly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each bin is labeled gray (0) unless its center falls within a non-omitted flash; then it receives the session-local sorted image code. Local codes are later remapped to a global 16-image mapping, yielding 17 categories including gray.

ii.
```python
flash_end = np.where(flash_omitted | ~np.isfinite(flash_end), -np.inf, flash_end)
on_screen = (tc >= flash_start[j]) & (tc < flash_end[j])
img = np.zeros(nbins, dtype=np.int64)
img[on_screen] = [img_lookup[n] for n in flash_img[j[on_screen]]]
out[0] = remap[out[0]]
```

iii. The notes explain that natural images occupy 250 ms and gray occupies the 500 ms inter-flash interval; omissions are also gray. Global remapping keeps category meanings consistent across image sets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image label is evaluated at exactly the same `tc` bin centers used to select neural frames.

ii.
```python
j = np.searchsorted(flash_start, tc, side='right') - 1
on_screen = (tc >= flash_start[j]) & (tc < flash_end[j])
mats = [mat[:, nearest_index(ts, tc)] for mat, ts in zip(plane_traces, plane_ts)]
```

iii. The common synchronized time coordinates provide frame-level alignment without relying only on trial-level initial/change labels.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change uses trial `go` and `change_time`, plus behavior-block presentation `start_time`, `end_time`, and `is_change`.

ii.
```python
flash_is_change = beh.is_change.values.astype(bool)
if bool(tr.go):
    k = int(np.searchsorted(flash_start, float(tr.change_time), side='right') - 1)
    assert flash_is_change[k]
```

iii. The agent distinguishes real changes on Go trials from sham changes on Catch trials.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and set to one at bin centers inside the changed-image flash, approximately 250 ms, only on Go trials.

ii.
```python
chg = np.zeros(nbins, dtype=np.int64)
if bool(tr.go):
    chg[(tc >= flash_start[k]) & (tc < flash_end[k])] = 1
```

iii. The notes say a flash-length label represents the change stimulus itself and avoids an extremely sparse single-bin event; Catch trials remain zero because identity does not change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is intrinsically binary: 0 for no changed-image flash and 1 inside a Go changed-image flash; there is no numeric threshold.

ii.
```python
chg = np.zeros(nbins, dtype=np.int64)
chg[mask] = 1
['no_change', 'change']
```

iii. This directly implements a binary decoder target.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The flash mask and neural sampling both use the same trial bin centers `tc`.

ii.
```python
chg[(tc >= flash_start[k]) & (tc < flash_end[k])] = 1
mats = [mat[:, nearest_index(ts, tc)] for mat, ts in zip(plane_traces, plane_ts)]
```

iii. The agent validates that trial `change_time` corresponds to an `is_change` presentation, then applies the shared clock/grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ds.running_speed['timestamps']` and `['speed']`, using the first plane with nonempty session-level behavior.

ii.
```python
running = ds0.running_speed
run_t = running['timestamps'].values.astype(float)
run_v = running['speed'].values.astype(float)
```

iii. The notes identify this as the SDK's filtered running-speed stream used by the tutorials.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are linearly interpolated to trial bin centers, concatenated across included trials within a session, and discretized using that session's 20/40/60/80 percentiles.

ii.
```python
run_trials.append(interp_nonan(run_t, run_v, tc))
run_all = np.concatenate(run_trials)
run_lab, run_edges = quantile_bin(run_all)
```

iii. Per-session quintiles were chosen to give balanced classes within each recording and accommodate strong cross-session/mouse differences.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four per-session quantile edges define five integer classes 0–4 using right-sided `searchsorted`.

ii.
```python
edges = np.quantile(values, np.arange(1, nbins) / nbins)
labels = np.searchsorted(edges, values, side='right').astype(np.int64)
return np.clip(labels, 0, nbins - 1), edges
```

iii. The agent interprets “five equal percentile bins” as equal-frequency bins in every session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is linearly interpolated at the same `tc` centers used for nearest-frame neural sampling.

ii.
```python
run_trials.append(interp_nonan(run_t, run_v, tc))
mats = [mat[:, nearest_index(ts, tc)] for mat, ts in zip(plane_traces, plane_ts)]
```

iii. The notes rely on hardware-synchronized NWB timestamps and a common sampling grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is `eye_tracking['pupil_width']` paired with eye-tracking `timestamps`.

ii.
```python
eye_t = eye['timestamps'].values.astype(float)
eye_v = eye['pupil_width'].values.astype(float)
```

iii. The agent cites tutorial usage of pupil width as pupil diameter and notes that it is in camera pixels.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Non-finite pupil samples (including blink/outlier gaps) are ignored during linear interpolation to trial bin centers. Values are then concatenated and quantile-binned within session; a session is rejected if interpolation leaves non-finite values.

ii.
```python
ok = np.isfinite(y) & np.isfinite(x)
return np.interp(xq, x[ok], y[ok])
pup_all = np.concatenate(pup_trials)
pup_lab, pup_edges = quantile_bin(pup_all)
```

iii. The notes say interpolation prevents blink artifacts from becoming labels, while within-session binning is appropriate because pixel scale varies with rig, zoom, and eye position.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four within-session percentile edges create five integer classes 0–4, identically to running speed.

ii.
```python
pup_lab, pup_edges = quantile_bin(pup_all)
out[3] = pup_lab[pos:pos + n]
```

iii. This is the agent's per-session interpretation of five equal percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil width is interpolated at the same trial bin centers used for neural nearest-frame sampling.

ii.
```python
pup_trials.append(interp_nonan(eye_t, eye_v, tc))
mats = [mat[:, nearest_index(ts, tc)] for mat, ts in zip(plane_traces, plane_ts)]
```

iii. Shared synchronized timestamps and bin centers are the stated alignment mechanism.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It comes from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
if bool(tr.hit): outcome = 0
elif bool(tr.miss): outcome = 1
elif bool(tr.false_alarm): outcome = 2
elif bool(tr.correct_reject): outcome = 3
```

iii. The agent uses the SDK's mutually exclusive canonical outcome flags for retained Go/Catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes map to fixed codes 0–3 and are broadcast across all time bins of the trial. Trials with no recognized flag are skipped.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
out[4] = outcome_trials[i]
```

iii. Broadcasting lets the static target share the required `(n_output, T)` output matrix while preserving its per-trial meaning.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Empty pupil/running streams, entirely invalid pupil, non-finite interpolated behavior, inconsistent plane trial tables, uncovered/too-short trials, missing outcomes, and too-few-trial sessions are rejected or skipped. Pupil NaNs are interpolated over; per-session exceptions are caught and logged without stopping the conversion. Plotting failures never discard data.

ii.
```python
if len(eye) == 0 or not np.any(np.isfinite(eye['pupil_width'].values)):
    return {'ophys_session_id': sid, 'error': 'no eye tracking data'}
if not np.all(np.isfinite(pup_all)):
    return {'ophys_session_id': sid, 'error': 'non-finite pupil after interpolation'}
except Exception as exc:
    return {'ophys_session_id': sid, 'error': repr(exc), 'traceback': traceback.format_exc()}
```

iii. The notes emphasize not fabricating a required output, maintaining shape/alignment integrity, and allowing one bad session or optional visualization to leave the rest of a long conversion usable.

## 9-a. What are the most time-consuming steps of the code?

i. Reading large NWB files and materializing full neural trace tables dominate. Optional diagnostic plotting and full decoder training are also expensive but are outside the normal core conversion path.

ii.
```python
experiments = [load_experiment(e) for e in eids]
timing['load_nwb'] = time.time() - t0
mat = np.stack(tbl['dff'].values).astype(np.float32)
```

iii. The notes report roughly 250 MB per experiment and explicitly identify NWB I/O as the bottleneck; this motivated session-level multiprocessing.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loop is over trials, whose variable lengths make full vectorization awkward. Plane loading within multiscope sessions is serial. Several small mapping/metadata loops and optional plotting `iterrows()` loops could also be vectorized, but timepoint lookup, interpolation, and binning already use NumPy.

ii.
```python
for tid, tr in sel.iterrows():
    mats = [mat[:, nearest_index(ts, tc)] for mat, ts in zip(plane_traces, plane_ts)]
for ds, eid in zip(experiments, eids):
    ...
```

iii. The agent says serial plane loading affects only six multiscope sessions and is not worth nested process management; vectorized `searchsorted` is already used within each trial.

## 9-c. What processing does the code repeat multiple times?

i. Each multiscope NWB contains duplicated session-level behavior tables; the code loads every plane and checks trial timing across planes, though it uses behavior from one plane. It also computes local image mappings in each worker and later remaps them globally. No session is reloaded during final assembly.

ii.
```python
experiments = [load_experiment(e) for e in eids]
for ds in experiments[1:]:
    assert len(ds.trials) == len(trials)
img_lookup = {name: i + 1 for i, name in enumerate(image_names)}
out[0] = remap[out[0]]
```

iii. The repeated behavior copies are inherent to the plane NWBs and are used for a consistency check; local-to-global image remapping enables independent parallel session conversion.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `cell_specimen_ids`, detailed `trial_info`, and timing dictionaries are constructed and returned by workers but not stored in the final pickle. With `--show-processing`, extensive diagnostic arrays/plots are computed solely for validation. Cross-plane behavior checks and rich session metadata are not decoder features.

ii.
```python
cell_specimen_ids=np.concatenate(plane_cellid),
trial_info=trial_info,
timing=timing,
if job.get('show_processing'):
    _plot_processing(...)
```

iii. The agent presents these as provenance, sanity-check, and profiling aids; plotting is opt-in and deliberately isolated from conversion success.
