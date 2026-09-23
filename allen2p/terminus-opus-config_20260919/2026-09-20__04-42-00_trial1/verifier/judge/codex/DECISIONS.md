# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK project cache the way the reference does. Instead, it reads `ophys_experiment_table.csv` from the local metadata directory, intersects that table with the NWB files physically present on disk, drops passive experiments, groups experiments by `ophys_session_id`, and then loads each experiment directly from its NWB file with `BehaviorOphysExperiment.from_nwb_path(...)`. It therefore loads all available active local experiments, including Multiscope sessions.

ii. 
```python
et = pd.read_csv(META_DIR + 'ophys_experiment_table.csv')
have = set(int(f.split('_')[-1].split('.')[0])
           for f in os.listdir(DATA_DIR + 'behavior_ophys_experiments'))
et = et[et.ophys_experiment_id.isin(have)]
et = et[~et.passive]
...
for eid in eids:
    planes.append((eid, BehaviorOphysExperiment.from_nwb_path(NWB_FMT % eid)))
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as replicating the SDK's underlying loader while avoiding S3/cache access because the NWB files are already local. It also states that passive sessions are excluded because they have no licks or rewards, making trial outcome degenerate.

## 1-b. How are the data split into subjects?

i. Subjects are unique mouse IDs taken from session metadata after session conversion. The final `subjects` list is the sorted set of `mouse` values from all retained sessions.

ii.
```python
result = dict(
    session_id=sid, experiment_ids=eids, mouse=str(md['mouse_id']),
    ...
)
...
subjects = sorted(set(r['mouse'] for r in results))
```

iii. The justification in the notes is simply that `mouse_id` is the canonical subject identifier in the Allen metadata.

## 1-c. How are the data split into sessions?

i. Sessions are grouped by `ophys_session_id`. If a session contains multiple experiments or planes, they are merged into one decoder session and their neurons are concatenated.

ii.
```python
sessions = [(int(sid), list(map(int, g.ophys_experiment_id.values)))
            for sid, g in et.groupby('ophys_session_id')]
...
for eid in eids:
    planes.append((eid, BehaviorOphysExperiment.from_nwb_path(NWB_FMT % eid)))
```

iii. The notes explicitly justify this by saying that a decoder session should correspond to one continuous recording, and that Multiscope planes share the same behavior stream and clock, so they should be merged.

## 1-d. How are the data split into trials?

i. The AI does not use the full SDK trial window from `start_time` to `stop_time`. It keeps only go or catch rows from `ref.trials`, finds the flash whose onset matches `change_time`, and defines each trial as a fixed 8-bin window: 3 flashes before the change, the change flash itself, and 4 flashes after.

ii.
```python
trials = ref.trials
keep = (trials.go.values.astype(bool) | trials['catch'].values.astype(bool))
trials = trials[keep]
...
change_times = trials.change_time.values.astype(float)
j = np.searchsorted(flash_start, change_times - 1e-6, side='left')
...
bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]
bt0 = flash_start[bin_flash]
bt1 = bt0 + BIN_SIZE
```

iii. The notes justify this as matching the paper's 750 ms image-presentation interval and yielding a fixed trial length across both 31 Hz and 11 Hz rigs.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in several stages. First, only `go` or `catch` rows are kept. Then the code requires finite `change_time`, requires `change_time` to match an actual flash onset, requires enough preceding and following flashes to form the 8-bin window, and requires the outcome to map to one of `hit`, `miss`, `false_alarm`, or `correct_reject`. Sessions with fewer than 2 remaining trials are dropped.

ii.
```python
keep = (trials.go.values.astype(bool) | trials['catch'].values.astype(bool))
trials = trials[keep]
...
good &= np.isfinite(change_times)
good &= np.abs(flash_start[j_clipped] - change_times) < 1e-4
good &= (j - N_PRE >= 0) & (j + N_POST < len(flash_start))
...
for k, name in enumerate(OUTCOMES):
    oc[trials[name].values.astype(bool)] = k
good &= oc >= 0
...
if len(idx) < 2:
    return None
```

iii. The notes justify these filters as ensuring that every kept trial has a well-defined aligned flash-centered window and a valid behavioral outcome category.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural activity from `ds.events.events`, not from `dff_traces`.

ii.
```python
for eid, ds in planes:
    ev = np.vstack(ds.events.events.values).astype(np.float64)
    ts = ds.ophys_timestamps.astype(float)
```

iii. The notes justify this by citing the paper's statement that analyses used detected calcium events rather than dF/F.

## 2-b. How is the `neural` data processed?

i. For each plane, detected event magnitudes are summed within each 750 ms trial bin. The per-plane binned arrays are reshaped to `(n_cells, n_trials, 8)` and concatenated across planes along the neuron axis. The final per-trial matrices saved to output are `(n_neurons, 8)` float32 arrays.

ii.
```python
s, _ = binned_sum(ev, ts, flat_t0, flat_t1)
neural_planes.append(s.reshape(ev.shape[0], n_trials, N_BINS))
...
neural = np.concatenate(neural_planes, axis=0).astype(np.float32)
...
data['neural'].append([np.ascontiguousarray(r['neural'][:, t, :]) for t in range(n_tr)])
```

iii. The notes justify summing events per 750 ms bin as the natural aggregation for the paper's image-presentation interval and as a way to make 11 Hz and 31 Hz sessions comparable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no explicit custom neuron filter in `convert_data.py`, but it relies on the AllenSDK loader's default valid-ROI filtering by calling `BehaviorOphysExperiment.from_nwb_path(...)`.

ii.
```python
planes.append((eid, BehaviorOphysExperiment.from_nwb_path(NWB_FMT % eid)))
```

iii. The notes state that `from_nwb_path` uses `exclude_invalid_rois=True` by default, so invalid ROIs are already excluded and no additional neuron curation is applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `trials.change_time`, interpreted as the real change on go trials and the sham change on catch trials. The aligned trial window is the fixed 8-bin flash sequence centered on that event.

ii.
```python
change_times = trials.change_time.values.astype(float)
j = np.searchsorted(flash_start, change_times - 1e-6, side='left')
...
bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]
bt0 = flash_start[bin_flash]
bt1 = bt0 + BIN_SIZE
...
s, _ = binned_sum(ev, ts, flat_t0, flat_t1)
```

iii. The notes justify this by saying `change_time` is the event the animal reports and that it coincides exactly with a flash onset for both go and catch trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 750 ms per bin, with 8 bins per trial. Substantial temporal rebinning is applied: framewise neural events are summed into those 750 ms bins.

ii.
```python
BIN_SIZE = 0.75
N_PRE = 3
N_POST = 4
N_BINS = N_PRE + 1 + N_POST
...
bt1 = bt0 + BIN_SIZE
...
neural = np.concatenate(neural_planes, axis=0).astype(np.float32)
```

iii. The notes justify this as matching the "image presentation interval" used in the paper and as the simplest common time base across rig types.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from the flash-level stimulus table, specifically `stimulus_presentations.image_name` after filtering to the change-detection block. Omitted flashes are forced to the label `'omitted'`.

ii.
```python
sp = ref.stimulus_presentations
sp = sp[sp.stimulus_block_name.str.contains('change_detection', na=False)]
...
flash_image = sp.image_name.values.astype(object)
flash_omitted = sp.omitted.values.astype(bool)
flash_image = np.where(flash_omitted, 'omitted', flash_image)
...
img = flash_image[bin_flash]
```

iii. The notes justify this by saying image identity should reflect the actual flash sequence around the aligned change, including omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code indexes the flash sequence at the 8 aligned flash positions around each trial's change, producing an `(n_trials, 8)` string array. Later it builds a global image vocabulary across sessions, moves `'omitted'` to the end, and maps images to integer class IDs.

ii.
```python
img = flash_image[bin_flash]
...
images = sorted(set(v for r in results for v in np.unique(r['image'])))
images = [v for v in images if v != 'omitted'] + ['omitted']
img_map = {v: i for i, v in enumerate(images)}
...
img_idx = np.vectorize(img_map.get)(r['image']).astype(np.int64)
```

iii. The notes justify the shared vocabulary as making class labels consistent across sessions and preserving omitted flashes as their own category instead of fabricating an image label there.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is sampled at the exact same 8 flash bins used to bin neural activity. The image row for each trial therefore has one category per neural time bin.

ii.
```python
bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]
...
img = flash_image[bin_flash]
...
data['output'].append([
    np.stack([img_idx[t], r['change'][t], r['run_cls'][t], r['pupil_cls'][t],
              np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64)
    for t in range(n_tr)])
```

iii. The notes justify this by treating the flash-centered 750 ms bins as the common alignment grid for every stream.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change`, evaluated at the aligned flash bins.

ii.
```python
flash_ischange = sp.is_change.values.astype(bool)
...
chg = flash_ischange[bin_flash].astype(np.int64)
```

iii. The notes justify this as giving the semantics required by the task: `is_change` is 1 only on real image changes, so catch trials remain all-zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. No extra thresholding or interpolation is applied. The code directly indexes the flash-level `is_change` boolean at each of the 8 aligned bins and casts it to integer labels.

ii.
```python
chg = flash_ischange[bin_flash].astype(np.int64)
...
assert np.all(chg[is_go, N_PRE] == 1)
assert np.all(chg[~is_go, N_PRE] == 0)
assert np.all(chg[:, :N_PRE] == 0)
```

iii. The notes say an earlier mistaken assumption that catch trials should also have a change flag was corrected after verifying that `is_change` marks only real changes, not sham changes.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is not thresholded from a continuous value. The output is already binary and is encoded directly as integer categories `0` and `1`.

ii.
```python
chg = flash_ischange[bin_flash].astype(np.int64)
...
output_values=[images, ['no_change', 'change'],
               [f'speed_q{k+1}' for k in range(N_QUANTILES)],
               [f'pupil_q{k+1}' for k in range(N_QUANTILES)],
               OUTCOMES]
```

iii. The justification is that `is_change` is already the categorical event the task wants, so no additional thresholding is needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned by reading `is_change` at the same 8 flash bins used for neural binning.

ii.
```python
bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]
...
chg = flash_ischange[bin_flash].astype(np.int64)
...
np.stack([img_idx[t], r['change'][t], r['run_cls'][t], r['pupil_cls'][t],
          np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64)
```

iii. The notes justify the alignment by treating the flash grid around `change_time` as the shared clock for all outputs and neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ref.running_speed`, using its `timestamps` and `speed` columns.

ii.
```python
run = ref.running_speed
run_t = run.timestamps.values.astype(float)
run_v = run.speed.values.astype(float)
```

iii. The notes identify this as the standard running stream in the Allen behavior session object.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The running signal is sorted by timestamp, NaNs are interpolated if present, then a mean speed is computed within each of the 8 flash bins. The resulting binned continuous values are discretized into 5 quantile classes within the same session.

ii.
```python
order = np.argsort(run_t)
run_t, run_v = run_t[order], run_v[order]
run_v = interpolate_nans(run_v)
...
run_binned = binned_mean_1d(run_v, run_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
run_cls, run_edges = quantile_bin(run_binned.ravel())
run_cls = run_cls.reshape(n_trials, N_BINS)
```

iii. The notes justify session-wise percentile binning by arguing that absolute running ranges differ strongly across mice and days, so within-session bins better encode relative state.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins using quantiles computed from all running bins within the session, not globally across the dataset.

ii.
```python
def quantile_bin(x, n=N_QUANTILES):
    edges = np.quantile(x, np.linspace(0, 1, n + 1)[1:-1])
    return np.searchsorted(edges, x, side='right').astype(np.int64), edges
...
run_cls, run_edges = quantile_bin(run_binned.ravel())
```

iii. The notes explicitly justify this as making the class labels represent relative running level within each session rather than absolute speed across all animals.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by averaging samples whose timestamps fall into the same 8 flash bins used for neural binning.

ii.
```python
bt0 = flash_start[bin_flash]
bt1 = bt0 + BIN_SIZE
...
run_binned = binned_mean_1d(run_v, run_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
...
np.stack([img_idx[t], r['change'][t], r['run_cls'][t], r['pupil_cls'][t],
          np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64)
```

iii. The notes justify this by saying all data streams are already on the session clock, so they only need to be summarized onto the chosen flash-centered bin grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ref.eye_tracking.pupil_width`, together with `eye.timestamps`. Blink-related missing samples are handled through NaN interpolation; the code does not explicitly use `likely_blink`.

ii.
```python
eye = ref.eye_tracking
...
pupil = interpolate_nans(eye.pupil_width.values)
...
eye_t = eye.timestamps.values.astype(float)
```

iii. The notes justify `pupil_width` as the closest available diameter-like measure and state that blink corruption appears as NaNs in this stream.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code interpolates NaNs in the pupil-width trace, sorts timestamps, computes a mean pupil value within each 750 ms flash bin, and then discretizes those bin means into 5 within-session quantile classes.

ii.
```python
pupil = interpolate_nans(eye.pupil_width.values)
...
order = np.argsort(eye_t)
eye_t, pupil = eye_t[order], pupil[order]
...
pup_binned = binned_mean_1d(pupil, eye_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
...
pup_cls, pup_edges = quantile_bin(pup_binned.ravel())
pup_cls = pup_cls.reshape(n_trials, N_BINS)
```

iii. The notes justify within-session percentile binning by arguing that pupil is measured in camera pixels, so absolute scale varies across sessions.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into 5 quantile bins computed within each session from the binned continuous pupil values.

ii.
```python
pup_cls, pup_edges = quantile_bin(pup_binned.ravel())
```

iii. The notes explicitly defend within-session quintiles as making categories comparable in relative meaning across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by averaging eye-tracking samples within the same 8 flash bins used for neural data.

ii.
```python
bt0 = flash_start[bin_flash]
bt1 = bt0 + BIN_SIZE
...
pup_binned = binned_mean_1d(pupil, eye_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
```

iii. The notes justify this by using the same flash-centered time base for all streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
oc = np.full(len(trials), -1, dtype=np.int64)
for k, name in enumerate(OUTCOMES):
    oc[trials[name].values.astype(bool)] = k
```

iii. The notes justify these as the canonical mutually exclusive Allen outcome labels for non-aborted, non-auto-rewarded change-detection trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcome booleans are mapped to fixed integer codes in the order `['hit', 'miss', 'false_alarm', 'correct_reject']`. The per-trial code is then broadcast across all 8 bins for that trial in the final output tensor.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome = oc[idx]
...
np.full(N_BINS, r['outcome'][t], dtype=np.int64)
```

iii. The notes justify broadcasting because trial outcome is static per trial but the target format prefers time-varying outputs when possible.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or problematic data are handled by dropping some sessions and interpolating some signals. Sessions with no eye-tracking rows or all-NaN pupil traces are dropped entirely. NaNs inside 1-D running or pupil traces are linearly interpolated, and empty running/pupil bins fall back to interpolation at the bin center. Trials with invalid `change_time`, non-matching flash alignment, insufficient surrounding flashes, or invalid outcome categories are dropped. Sessions with fewer than 2 retained trials are also dropped.

ii.
```python
if len(eye) == 0:
    print(f'  session {sid}: DROPPED (no eye tracking data)', flush=True)
    return None
...
if pupil is None:
    print(f'  session {sid}: DROPPED (pupil all NaN)', flush=True)
    return None
...
if bad.any():
    idx = np.arange(len(x))
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
...
if np.any(~ok):
    centres = 0.5 * (t0[~ok] + t1[~ok])
    out[~ok] = np.interp(centres, timestamps, values)
...
if len(idx) < 2:
    return None
```

iii. The notes justify these choices as preserving a usable categorical output for every retained trial while avoiding NaNs that would break decoder validation.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading NWB experiments from disk and binning full-session neural event matrices. The code treats session loading as the main bottleneck and parallelizes conversion across sessions.

ii.
```python
for eid in eids:
    planes.append((eid, BehaviorOphysExperiment.from_nwb_path(NWB_FMT % eid)))
...
with Pool(min(args.workers, len(jobs))) as pool:
    results = pool.map(process_session, jobs)
```

iii. The notes explicitly say load time per plane dominates and that multiprocessing was added to reduce wall-clock time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest per-bin computations are already vectorized using cumulative sums plus `searchsorted`. Remaining Python loops that could be reduced further are the per-plane loop over experiments, the per-trial assembly loop when building output lists, and repeated list-index lookups for subjects and regions.

ii.
```python
for eid, ds in planes:
    ev = np.vstack(ds.events.events.values).astype(np.float64)
    ...
for r in results:
    ...
    data['output'].append([
        np.stack([img_idx[t], r['change'][t], r['run_cls'][t], r['pupil_cls'][t],
                  np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64)
        for t in range(n_tr)])
    data['subject_idx'].append(subjects.index(r['mouse']))
    data['brain_region_idx'].append(np.array([regions.index(x) for x in r['regions']],
                                             dtype=np.int64))
```

iii. The notes emphasize that the main intended optimization was vectorizing binning itself; these remaining loops are smaller-scale assembly work rather than the dominant cost.

## 9-c. What processing does the code repeat multiple times?

i. There is no major repeated full-data pass like reloading sessions, but the code does repeat some small computations during final assembly: it maps image strings to codes per session with `np.vectorize`, linearly searches `subjects.index(...)` per session, and linearly searches `regions.index(...)` for every neuron in each session.

ii.
```python
img_idx = np.vectorize(img_map.get)(r['image']).astype(np.int64)
...
data['subject_idx'].append(subjects.index(r['mouse']))
data['brain_region_idx'].append(np.array([regions.index(x) for x in r['regions']],
                                         dtype=np.int64))
```

iii. The notes mostly argue that expensive repeated processing was avoided by loading each session once and by vectorizing the binning step.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `process_session()` computes and returns several diagnostic values that are not written into the final pickle and are only used for checks or optional plotting: continuous running and pupil traces per bin, quantile edges, timings, change times, `is_go`, and per-plane cell counts.

ii.
```python
result = dict(
    ...
    run_cont=run_binned, pupil_cont=pup_binned,
    run_edges=run_edges, pupil_edges=pup_edges,
    n_trials=n_trials, n_dropped=n_dropped, n_cells_plane=n_cells_plane,
    ...
    trial_change_times=change_times[idx], is_go=is_go,
)
```

iii. The notes justify these extras as sanity-check and plotting support rather than part of the downstream decoder dataset itself.
