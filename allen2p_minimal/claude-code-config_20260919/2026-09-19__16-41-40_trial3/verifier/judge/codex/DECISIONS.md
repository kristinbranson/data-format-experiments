# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache the way the reference solution does. It reads the local metadata table `ophys_experiment_table.csv`, scans the local NWB directory for available experiment files, filters to the single-plane `VisualBehavior` project and to active sessions (`passive == False`), and then loads each experiment directly from its NWB file with `BehaviorOphysExperiment.from_nwb_path`. Each selected experiment is converted independently and cached as a pickle.

ii. 
```python
exp = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
available = set()
for fname in os.listdir(NWB_DIR):
    m = re.match(r'behavior_ophys_experiment_(\d+)\.nwb$', fname)
    if m:
        available.add(int(m.group(1)))
exp = exp[exp.ophys_experiment_id.isin(available)]
exp = exp[exp.project_code == 'VisualBehavior']
exp = exp[~exp.passive.astype(bool)]
```

```python
nwb_path = os.path.join(
    NWB_DIR, f'behavior_ophys_experiment_{ophys_experiment_id}.nwb')
ds = BehaviorOphysExperiment.from_nwb_path(nwb_path)
```

iii. In the trajectory, the AI justified this as a session-selection decision: single-plane `VisualBehavior` experiments share one common frame rate, whereas Multiscope experiments do not; passive sessions were excluded because the lick spout is retracted and trial outcome would be undefined; and three experiments were dropped because they had no eye-tracking data. This rationale is stated in the script docstring and repeated in the final trajectory summary.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values collected from the converted sessions. After loading cached session pickles, the AI builds a sorted unique subject list and then maps each session's `mouse_id` to an integer subject index.

ii.
```python
subjects = sorted({s['mouse_id'] for s in sessions})
subject_to_idx = {m: i for i, m in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[s['mouse_id']])
```

iii. The trajectory does not contain a separate discussion of subject splitting beyond using mouse IDs throughout. The justification is implicit: `mouse_id` is treated as the canonical animal identifier.

## 1-c. How are the data split into sessions?

i. The AI treats each selected `ophys_experiment_id` as one final session. Because it restricts the dataset to single-plane `VisualBehavior`, it assumes "session == experiment" and then sorts sessions by experiment ID when assembling the final dataset.

ii.
```python
# single-plane "VisualBehavior" variant -> one common ophys frame rate and
# one imaging plane per session
exp = exp[exp.project_code == 'VisualBehavior']
...
sessions.sort(key=lambda s: s['ophys_experiment_id'])
```

iii. The trajectory explicitly says the single-plane rigs have exactly one imaging plane per ophys session, so treating each experiment as a session is, in the AI's view, unambiguous and guarantees a common time bin across the dataset.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`. The AI keeps go or catch trials, drops aborted and auto-rewarded trials, converts `start_time` and `stop_time` to ophys-frame indices with `np.searchsorted`, and slices each stream over that variable-length window.

ii.
```python
trials = ds.trials
keep = ((trials['go'].astype(bool) | trials['catch'].astype(bool))
        & ~trials['aborted'].astype(bool)
        & ~trials['auto_rewarded'].astype(bool))
trials = trials[keep]
```

```python
starts = np.searchsorted(ts, trials['start_time'].values, side='left')
stops = np.searchsorted(ts, trials['stop_time'].values, side='left')
...
i0, i1 = int(starts[k]), int(stops[k])
neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. The trajectory summary says the AI used the experiment-defined `trials` table, `start_time -> stop_time`, with variable trial length because `change_time` is drawn from a truncated exponential.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to go/catch, non-aborted, non-auto-rewarded trials. The AI then drops trials shorter than 2 ophys frames and drops trials whose sliced image labels still contain `None` because the trial began before the first valid flash label. It also drops sessions with fewer than 2 remaining trials, and drops entire experiments that cannot produce pupil data.

ii.
```python
keep = ((trials['go'].astype(bool) | trials['catch'].astype(bool))
        & ~trials['aborted'].astype(bool)
        & ~trials['auto_rewarded'].astype(bool))
...
if i1 - i0 < 2:
    continue
if np.any(image_name[i0:i1] == None):
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. The trajectory explicitly justifies excluding aborted and auto-rewarded trials as matching the instructions. It also states that sessions without pupil data were dropped because the required pupil output could not be produced. The short-trial and `None`-label filters are implemented in code but were not separately discussed in the trajectory.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. By default, neural data is derived from `ds.dff_traces['dff']`. The script also supports `events` or `filtered_events` through the `VB_TRACE` environment variable, but the default conversion uses dF/F.

ii.
```python
if TRACE == 'dff':
    src = ds.dff_traces
    col = 'dff'
else:
    src = ds.events
    col = TRACE
...
traces = np.vstack([np.asarray(v, dtype=np.float32)
                    for v in src[col].values])
```

iii. The trajectory first explored event-based traces, then switched to dF/F. In step 127 the AI states that dF/F decodes substantially better than events, and in the final summary it says this was a deliberate departure from the reference paper because per-frame decoding at 32 ms was too sparse with event traces.

## 2-b. How is the `neural` data processed?

i. The AI stacks all ROI traces for an experiment into a `neurons x time` matrix, converts them to `float32`, and replaces NaNs or infinities with zero. It does not perform additional smoothing, deconvolution, or plane-merging beyond what the Allen pipeline already provided.

ii.
```python
traces = np.vstack([np.asarray(v, dtype=np.float32)
                    for v in src[col].values])
assert traces.shape[1] == ts.shape[0]
np.nan_to_num(traces, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The trajectory justifies this by relying on Allen pipeline preprocessing and by arguing that dF/F already carries the per-frame graded signal needed for the decoder. The code docstring also says all released ROIs are kept and no further neuron-level curation is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply neuron-level QC beyond what is already in the released Allen data. It keeps all ROIs found in the selected trace table and only sanitizes numeric gaps with `np.nan_to_num`.

ii.
```python
cell_ids = src.index.values
traces = np.vstack([np.asarray(v, dtype=np.float32)
                    for v in src[col].values])
np.nan_to_num(traces, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The script docstring states that every ROI in the release has already passed Allen pipeline QC (`valid_roi == True`) and that the reference paper applied no further neuron-level curation. The trajectory repeats that all released ROIs were kept.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys frames, and each trial is sliced from the first ophys frame at or after `trials.start_time` through the first ophys frame at or after `trials.stop_time`. The metadata declares the temporal alignment event as `trial start_time`.

ii.
```python
starts = np.searchsorted(ts, trials['start_time'].values, side='left')
stops = np.searchsorted(ts, trials['stop_time'].values, side='left')
...
neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
```

```python
'temporal_alignment_event': (
    'trial start_time (onset of the change-detection trial, i.e. '
    'the first ophys frame at or after trials.start_time); all data '
    'streams are sampled on the ophys frame clock'),
```

iii. The trajectory summary says ophys timestamps were treated as the master clock and that trials run from `start_time` to `stop_time`. That is the explicit justification for the alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native ophys frame rate of the selected single-plane experiments, about 31 Hz or 32.32 ms per bin. No temporal rebinning of neural data is applied.

ii.
```python
# The ophys timestamps are the master clock...
# Time bins are the ophys frames themselves (~32.32 ms)
```

```python
dts = np.array([s['dt'] for s in sessions])
time_bin_size = float(np.median(dts) * 1000.0)
```

iii. The trajectory explicitly says the AI kept only single-plane experiments because they share one frame rate, then used ophys frames directly with no neural resampling.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is not derived from the trials table. It is derived from `ds.stimulus_presentations`, specifically the `image_name` values from the `change_detection` stimulus block and their `start_time`s. The resulting label can also be `'omitted'`.

ii.
```python
stim = ds.stimulus_presentations
if 'stimulus_block_name' in stim.columns:
    stim = stim[stim['stimulus_block_name'].astype(str)
                .str.contains('change_detection')]
...
names = stim['image_name'].values.astype(object)
```

iii. The trajectory says the AI wanted image identity to label the 750 ms image-presentation interval defined in the paper, including omitted flashes as their own class, instead of using only `initial_image_name` and `change_image_name` from the trials table.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI first labels every ophys frame in a session with the image from the most recent flash onset, so the label extends across the 250 ms image display and the following 500 ms gray period. After trial slicing, it builds one global vocabulary over all sessions, keeps `'omitted'` as its own category, and maps image names to integer codes.

ii.
```python
idx = np.searchsorted(starts, ts, side='right') - 1
valid = idx >= 0
idx_clipped = np.where(valid, idx, 0)
image_name = np.where(valid, names[idx_clipped], None)
```

```python
images = sorted({name for s in sessions for tr in s['image_name']
                 for name in np.unique(tr)} - {'omitted'})
image_values = images + ['omitted']
image_to_idx = {name: i for i, name in enumerate(image_values)}
```

iii. The trajectory explicitly justifies this with the paper's 750 ms "image presentation interval" convention and says omitted flashes should get their own class.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is assigned on the full ophys timebase first, then sliced with the same per-trial frame boundaries used for neural data, so the image label array and neural matrix share the same time bins.

ii.
```python
image_name, is_change = _flash_labels(stim, ts)
...
image_trials.append(image_name[i0:i1].copy())
neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. The trajectory summary says all outputs were aligned to the ophys frame clock. The specific justification for image identity is that each ophys frame inherits the label of the current flash interval.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `ds.stimulus_presentations['is_change']` after restricting to the change-detection stimulus block. It is therefore tied to the stimulus stream, not to `trials.go` plus `change_time`.

ii.
```python
changes = stim['is_change'].values.astype(bool)
...
is_change = np.where(valid, changes[idx_clipped], False)
```

iii. The trajectory says the AI wanted change labels to correspond to the flash at which the image identity changed, frame by frame, and later reports that it verified this against the SDK to within one frame.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI labels every ophys frame with the `is_change` value of the most recent flash onset, so a whole 750 ms flash cycle is marked as change if it is the changed flash, and otherwise 0.

ii.
```python
idx = np.searchsorted(starts, ts, side='right') - 1
...
is_change = np.where(valid, changes[idx_clipped], False)
...
change_trials.append(is_change[i0:i1].astype(np.int64))
```

iii. The trajectory explicitly says "change = 1 through the changed flash's interval" and that the label spans exactly one 750 ms flash cycle.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is made binary with categories `0 = no_change` and `1 = change`. No extra thresholding beyond the boolean `is_change` value is applied.

ii.
```python
'output_values': [
    image_values,
    ['no_change', 'change'],
    _edge_labels(run_edges, 'cm/s'),
    _edge_labels(pup_edges, 'pix'),
    OUTCOME_NAMES,
],
```

iii. The trajectory final summary describes image change as a binary output and does not mention any further thresholding because the raw stimulus flag is already categorical.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is computed on the ophys timebase first and then sliced with the same trial frame indices used for neural data.

ii.
```python
image_name, is_change = _flash_labels(stim, ts)
...
change_trials.append(is_change[i0:i1].astype(np.int64))
neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. The trajectory explicitly says all data streams were sampled on the ophys frame clock and that the change label was checked against `change_time - start_time` to within one frame.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `ds.running_speed`, specifically its `timestamps` and `speed` columns.

ii.
```python
run = ds.running_speed
run_t = np.asarray(run['timestamps'].values, dtype=np.float64)
run_v = np.asarray(run['speed'].values, dtype=np.float64)
```

iii. The trajectory does not separately debate the raw source; it treats `running_speed` as the canonical Allen locomotion stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI filters out non-finite timestamp or speed values, linearly interpolates running speed onto the ophys timestamps with `np.interp`, pools all trial time bins across sessions, computes global quintile cut points with `np.percentile`, and then discretizes each trial using `np.searchsorted`.

ii.
```python
good = np.isfinite(run_t) & np.isfinite(run_v)
running = np.interp(ts, run_t[good], run_v[good])
```

```python
run_all = np.concatenate([np.concatenate(s['running_speed']) for s in sessions])
qs = np.linspace(0, 100, NBINS + 1)[1:-1]
run_edges = np.percentile(run_all, qs)
...
run = np.searchsorted(run_edges, s['running_speed'][k],
                      side='right').astype(np.int64)
```

iii. The trajectory final summary says running was interpolated onto ophys frames and binned into quintiles from the pooled dataset distribution because "five equal percentile bins" was read as dataset-wide.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five global percentile bins using the 20th, 40th, 60th, and 80th percentiles of all running samples pooled across all kept sessions and trials.

ii.
```python
qs = np.linspace(0, 100, NBINS + 1)[1:-1]
run_edges = np.percentile(run_all, qs)
...
run = np.searchsorted(run_edges, s['running_speed'][k],
                      side='right').astype(np.int64)
```

iii. The trajectory explicitly says the AI chose pooled dataset-wide quintiles rather than per-session quintiles so that one bin index would correspond to the same absolute running-speed range everywhere.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is resampled to the ophys timestamps before trial slicing, and then trial windows are cut with the same `[i0:i1]` indices used for neural activity.

ii.
```python
running = np.interp(ts, run_t[good], run_v[good])
...
running_trials.append(running[i0:i1].astype(np.float64))
neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. The trajectory summary says ophys timestamps are the master clock and that running is interpolated onto those frame times.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking['pupil_area']` and `ds.eye_tracking['timestamps']`, not from `pupil_width`. The AI computes diameter as `2 * sqrt(area / pi)`.

ii.
```python
t = eye_tracking['timestamps'].values
area = eye_tracking['pupil_area'].values.astype(float)
good = np.isfinite(area) & np.isfinite(t) & (area > 0)
...
return t[good], 2.0 * np.sqrt(area[good] / np.pi)
```

iii. The trajectory justification is in the code docstring: the AI considered area-derived diameter more robust than a single ellipse axis. It also notes that the release can lack eye tracking entirely, in which case the experiment is dropped.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI removes non-finite or non-positive pupil-area samples, requires at least 100 remaining eye-tracking samples, converts area to diameter, linearly interpolates the resulting signal to the ophys timestamps with `np.interp`, then computes global quintile thresholds across all pooled pupil samples and discretizes each trial with `np.searchsorted`.

ii.
```python
good = np.isfinite(area) & np.isfinite(t) & (area > 0)
if good.sum() < 100:
    return None, None
return t[good], 2.0 * np.sqrt(area[good] / np.pi)
...
pupil = np.interp(ts, pt, pv)
```

```python
pup_all = np.concatenate([np.concatenate(s['pupil_diameter']) for s in sessions])
pup_edges = np.percentile(pup_all, qs)
...
pup = np.searchsorted(pup_edges, s['pupil_diameter'][k],
                      side='right').astype(np.int64)
```

iii. The trajectory says sessions without usable eye tracking were dropped because the required pupil output could not be produced. The global-quintile rationale matches the one used for running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five pooled global percentile bins using the 20th, 40th, 60th, and 80th percentiles over all kept pupil-diameter samples.

ii.
```python
pup_edges = np.percentile(pup_all, qs)
...
pup = np.searchsorted(pup_edges, s['pupil_diameter'][k],
                      side='right').astype(np.int64)
```

iii. The trajectory final summary says pupil, like running, was binned into dataset-wide quintiles.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is converted to the ophys timebase with linear interpolation and then sliced with the same trial frame indices as the neural traces.

ii.
```python
pupil = np.interp(ts, pt, pv)
...
pupil_trials.append(pupil[i0:i1].astype(np.float64))
neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. The trajectory summary treats pupil the same way as running: interpolate to ophys timestamps, then slice by ophys-frame trial windows.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `ds.trials`.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome = np.full(len(trials), -1, dtype=np.int64)
for i, name in enumerate(OUTCOME_NAMES):
    outcome[trials[name].astype(bool).values] = i
```

iii. The trajectory does not separately justify this choice; it uses the standard Allen trial outcome fields and treats them as exhaustive for valid go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four outcome booleans to integer codes `0..3`, filters out any trial that fails to map, stores one code per trial, and then broadcasts that code across all time bins of the trial in the final `output` tensor.

ii.
```python
for i, name in enumerate(OUTCOME_NAMES):
    outcome[trials[name].astype(bool).values] = i
if np.any(outcome < 0):
    ok = outcome >= 0
    trials = trials[ok]
    outcome = outcome[ok]
```

```python
out = np.full(T, s['trial_outcome'][k], dtype=np.int64)
sess_output.append(np.stack([img, chg, run, pup, out], axis=0))
```

iii. The trajectory summary says trial outcome is constant within each trial and is held across the time bins because the target format prefers time-varying outputs where possible.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several failure cases by dropping or repairing data. Non-finite neural values are replaced with zero. Non-finite running-speed samples are removed before interpolation. Missing or unusable eye tracking causes the whole experiment to be dropped. Trials shorter than two frames or beginning before the first valid flash label are skipped. Trials with unmapped outcomes are removed. Exceptions in worker processes are caught and reported, and the corresponding experiment is dropped.

ii.
```python
np.nan_to_num(traces, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
...
good = np.isfinite(run_t) & np.isfinite(run_v)
running = np.interp(ts, run_t[good], run_v[good])
...
if pt is None:
    return None
...
if i1 - i0 < 2:
    continue
if np.any(image_name[i0:i1] == None):
    continue
```

```python
def _worker(eid):
    try:
        return eid, convert_experiment(eid), None
    except Exception:
        import traceback
        return eid, None, traceback.format_exc()
```

iii. The trajectory explicitly mentions dropping three experiments with no eye tracking and argues that without pupil data the required outputs cannot be produced. Other safeguards are visible in code but were not individually justified in the trajectory.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is per-experiment NWB loading and conversion inside `convert_experiment`, especially reading all session-long neural traces and then generating cached pickles. The AI also parallelizes this step with a `ProcessPoolExecutor`, which implies it expected per-experiment conversion to dominate runtime.

ii.
```python
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futs = [pool.submit(_worker, e) for e in eids]
```

```python
ds = BehaviorOphysExperiment.from_nwb_path(nwb_path)
...
traces = np.vstack([np.asarray(v, dtype=np.float32)
                    for v in src[col].values])
```

iii. The trajectory does not explicitly rank bottlenecks, but it repeatedly focuses on parallel experiment conversion, caching, and full decoder validation on multi-gigabyte outputs, which indicates that experiment loading and conversion were treated as the dominant costs.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code keeps several Python loops that could be vectorized further: the per-trial slicing loop in `convert_experiment`, the per-session / per-trial assembly loop in `assemble`, the list comprehension that stacks every ROI trace, and the per-frame list comprehension that maps image names to integer codes.

ii.
```python
traces = np.vstack([np.asarray(v, dtype=np.float32)
                    for v in src[col].values])
...
for k in range(len(trials)):
    i0, i1 = int(starts[k]), int(stops[k])
    ...
```

```python
for s in sessions:
    ...
    for k in range(ntrials):
        img = np.array([image_to_idx[n] for n in s['image_name'][k]],
                       dtype=np.int64)
```

iii. The trajectory does not discuss vectorization explicitly. This is an inference from the code structure rather than a stated design choice.

## 9-c. What processing does the code repeat multiple times?

i. The AI repeats some work across passes. It first loops over all trials to build per-trial arrays in `convert_experiment`, then loops over all sessions and all trials again in `assemble` to discretize outputs and build the final tensors. It also recomputes pooled running and pupil arrays by concatenating every trial from every session after those trial arrays have already been stored.

ii.
```python
for k in range(len(trials)):
    ...
    running_trials.append(running[i0:i1].astype(np.float64))
    pupil_trials.append(pupil[i0:i1].astype(np.float64))
```

```python
run_all = np.concatenate([np.concatenate(s['running_speed']) for s in sessions])
pup_all = np.concatenate([np.concatenate(s['pupil_diameter']) for s in sessions])
...
for s in sessions:
    ...
    for k in range(ntrials):
        run = np.searchsorted(run_edges, s['running_speed'][k],
                              side='right').astype(np.int64)
```

iii. The trajectory does not explicitly call this out. This section is based on the structure of the final code.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code retains and serializes some information that is not used by the downstream decoder inputs or outputs, such as `cell_specimen_ids` in the cached per-experiment pickles and detailed `session_info` metadata fields like `cre_line`, `equipment_name`, and `imaging_depth`. The code also creates human-readable bin-edge labels solely for metadata / output naming, not for the model-facing arrays.

ii.
```python
out = {
    ...
    'cell_specimen_ids': np.asarray(cell_ids),
    ...
}
```

```python
session_info.append({
    'ophys_experiment_id': s['ophys_experiment_id'],
    'ophys_session_id': s['ophys_session_id'],
    'behavior_session_id': s['behavior_session_id'],
    'mouse_id': s['mouse_id'],
    'cre_line': s['cre_line'],
    'session_type': s['session_type'],
    'equipment_name': s['equipment_name'],
    'targeted_structure': s['targeted_structure'],
    'imaging_depth': s['imaging_depth'],
    'ophys_frame_rate': s['ophys_frame_rate'],
    'n_neurons': int(nneurons),
    'n_trials': int(ntrials),
})
```

iii. The trajectory does not explicitly justify these extra retained fields. They appear to have been kept for interpretability and bookkeeping rather than for the decoder itself.
