# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the full VisualBehavior dataset via the SDK cache table. It reads the local `ophys_experiment_table.csv`, filters that table down to experiment IDs that have local NWB files, then further restricts to active, familiar sessions and loads each selected experiment through `VisualBehaviorOphysProjectCache.from_local_cache(...)`. Trials are then pulled from each loaded dataset's `trials` table.

ii.
```python
def get_experiment_table():
    tbl = pd.read_csv(os.path.join(
        CACHE_DIR, 'visual-behavior-ophys-1.1.0', 'project_metadata',
        'ophys_experiment_table.csv'))
    have = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in glob.glob(
        os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0',
                     'behavior_ophys_experiments', '*.nwb'))]
    return tbl[tbl.ophys_experiment_id.isin(have)]

def select_sessions():
    tbl = get_experiment_table()
    sel = tbl[(tbl.behavior_type == 'active_behavior') &
              (tbl.experience_level == 'Familiar')]
    return sel.groupby('ophys_session_id').ophys_experiment_id.apply(list).to_dict()

for oeid in oeids:
    ds = cache.get_behavior_ophys_experiment(int(oeid))
```

iii. In the trajectory, the AI explicitly decided to use only active familiar sessions because it believed passive sessions lacked decodable behavior and the reference paper restricted neural analysis to familiar stimuli. It also chose local-cache loading after inspecting the local NWB layout and said it wanted sessions with shared image labels and locally available files.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values taken from the selected sessions' metadata. The final `subjects` list is sorted unique mouse IDs, and each session's `subject_idx` is the index of that session's mouse ID into that list.

ii.
```python
subjects = sorted({r['info']['mouse_id'] for r in results})
'subject_idx': np.array([subjects.index(r['info']['mouse_id'])
                         for r in results], dtype=np.int64),
```

iii. The trajectory repeatedly describes the dataset in terms of mice and sessions and uses `mouse_id` as the animal identifier when surveying the metadata. There is no alternative subject definition in the run.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `ophys_session_id`. For Multiscope sessions, all experiments with the same `ophys_session_id` are merged into a single converted session; each plane contributes neurons to that combined session.

ii.
```python
def select_sessions():
    ...
    return sel.groupby('ophys_session_id').ophys_experiment_id.apply(list).to_dict()

def convert_session(osid, oeids, neural_key='events'):
    ...
    for oeid in oeids:
        ds = cache.get_behavior_ophys_experiment(int(oeid))
```

iii. Early in the trajectory the AI first considered single-plane experiments only, then explicitly revised that choice after noticing the reference paper used the multi-plane rig. It justified the final session definition by saying simultaneously imaged Mesoscope planes should be merged into one session so VISp and VISl neurons are analyzed together.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`, but the AI does not keep the original ophys-frame trial window. Instead, it keeps go or catch trials with a valid `change_time`, then converts each trial into a variable-length set of fixed 250 ms bins fully contained within `[start_time, stop_time)`, anchored so one bin edge lies exactly at `change_time`.

ii.
```python
tr = ds.trials
tr = tr[(tr.go | tr.catch) & tr.change_time.notna()]

for _, trial in tr.iterrows():
    k0 = int(np.ceil((trial.start_time - trial.change_time) / BIN_SIZE))
    k1 = int(np.floor((trial.stop_time - trial.change_time) / BIN_SIZE))
    edges = trial.change_time + BIN_SIZE * np.arange(k0, k1 + 1)
    if len(edges) < 3 or edges[0] < tmin or edges[-1] > tmax:
        continue
```

iii. The trajectory shows the AI first tried native-frame trials, then switched after pilot decoder tests. It explicitly said it wanted a fixed 250 ms grid, aligned to change time, because sparse calcium events decoded better in larger bins and because that grid matches the stimulus flash cadence.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by requiring `go` or `catch`, non-null `change_time`, exactly one recognized outcome label, full temporal coverage by all required streams, at least one sample per neural/running/pupil bin, and at least two surviving trials per session. Sessions without eye tracking or without enough valid trials are dropped entirely.

ii.
```python
tr = tr[(tr.go | tr.catch) & tr.change_time.notna()]
...
oc = [k for k in OUTCOME_NAMES if bool(trial[k])]
if len(oc) != 1:
    continue
...
if len(edges) < 3 or edges[0] < tmin or edges[-1] > tmax:
    continue
...
if np.any(nfr < 1):
    ok = False
    break
...
if np.any(rn < 1) or np.any(pn < 1):
    continue
...
if len(trials) < 2:
    return None
```

iii. In the trajectory, the AI justified these extra filters as necessary to guarantee every bin had support from every stream on the shared 250 ms grid and to drop sessions where pupil output could not be produced. It explicitly called out one session being dropped for missing eye tracking.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `ds.events`, specifically the `events` or `filtered_events` column chosen by the `--neural` argument. The final full run defaults to `events`.

ii.
```python
def convert_session(osid, oeids, neural_key='events'):
    ...
    ev = ds.events
    act = np.vstack(ev[neural_key].values).astype(np.float64)
```

iii. The trajectory states several times that the paper analyzed detected calcium events rather than raw dF/F, and the AI adopted that as a deliberate choice. It also ran pilot comparisons between `events` and `filtered_events` before settling back on raw `events` with coarser temporal bins.

## 2-b. How is the `neural` data processed?

i. Neural activity from each plane is converted to cumulative sums, summed within each 250 ms trial bin, and vertically stacked across planes within a session. No per-neuron normalization is applied after this; the binned event sums are written directly as `float32`.

ii.
```python
act = np.vstack(ev[neural_key].values).astype(np.float64)
...
'cum': np.concatenate([np.zeros((act.shape[0], 1)),
                       np.cumsum(act, axis=1)], axis=1),
...
vals, nfr = bin_sum(p['cum'], p['ts'], edges)
...
neural_trial = np.vstack(neural_parts).astype(np.float32)
```

iii. The AI's explicit justification in the trajectory was performance- and paper-driven: events were sparse at native frame rate, pilot decoding improved substantially after summing into larger bins, and the paper referenced discrete event representations rather than dF/F traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code applies no explicit cell-level QC filter beyond trusting the released Allen data. However, it skips any plane with zero events rows and skips trials where any neural bin would have zero ophys samples.

ii.
```python
ev = ds.events
if len(ev) == 0:
    continue
...
vals, nfr = bin_sum(p['cum'], p['ts'], edges)
if np.any(nfr < 1):
    ok = False
    break
```

iii. The metadata and trajectory both say the released ROI set had already passed Allen QC. The AI therefore treated Allen's upstream filtering as sufficient and only added coverage checks needed by its own binning scheme.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `change_time` rather than trial start. Each trial uses 250 ms bins whose edges are defined relative to `change_time`, and only bins fully inside the trial window are kept.

ii.
```python
k0 = int(np.ceil((trial.start_time - trial.change_time) / BIN_SIZE))
k1 = int(np.floor((trial.stop_time - trial.change_time) / BIN_SIZE))
edges = trial.change_time + BIN_SIZE * np.arange(k0, k1 + 1)
...
vals, nfr = bin_sum(p['cum'], p['ts'], edges)
```

iii. The trajectory is explicit here: the AI intentionally moved to change-time alignment so bin edges line up with the stimulus flash cycle and so sparse event signals would decode better on a common fixed grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 250 ms bin size. Yes, substantial rebinning is applied: ophys events are summed per 250 ms bin and the behavioral streams are averaged per 250 ms bin.

ii.
```python
BIN_SIZE = 0.25
...
'time_bin_size': BIN_SIZE * 1000.0,
...
vals, nfr = bin_sum(p['cum'], p['ts'], edges)
...
run_trial = rsum / rn
pupil_trial = psum / pn
```

iii. The trajectory states that the AI chose 250 ms because it is one image-flash duration, keeps bin edges locked to the 750 ms image cycle, and improved pilot decoder performance relative to native-frame bins.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `ds.stimulus_presentations`, using `image_name`, `start_time`, `omitted`, and `stimulus_block_name == 'change_detection_behavior'`. It is not derived from the trials table's `initial_image_name`/`change_image_name`.

ii.
```python
sp = ds.stimulus_presentations
sp = sp[sp.stimulus_block_name == 'change_detection_behavior']
image_names = sorted(set(sp.image_name.unique()) - {'omitted'})
shown = sp[(~sp.omitted) & (sp.image_name != 'omitted')].sort_values('start_time')
pres_start = shown.start_time.values
pres_img = np.array([image_names.index(n) for n in shown.image_name.values])
```

iii. The trajectory says the AI wanted stimulus labels tied to the actual image-flash sequence rather than only pre/post labels from the trials table. It also relied on its observation that familiar active sessions all used image set A with the same eight images.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a sorted list of image names for the session, converts each non-omitted flash into an integer code, and for each trial bin assigns the identity of the most recent shown image. That identity is held through the gray interval and across omitted flashes.

ii.
```python
image_names = sorted(set(sp.image_name.unique()) - {'omitted'})
pres_img = np.array([image_names.index(n) for n in shown.image_name.values])
...
j = np.searchsorted(pres_start, centers, side='right') - 1
if np.any(j < 0):
    continue
image_trial = pres_img[j].astype(np.int64)
```

iii. The AI explicitly justified this in the script header and trajectory by citing the paper's behavioral convention of assigning labels over the full 750 ms image-presentation interval, including the gray screen and omissions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned to the same 250 ms change-centered bins as the neural data. The bin centers are used to look up the most recent image flash onset.

ii.
```python
centers = 0.5 * (edges[:-1] + edges[1:])
...
j = np.searchsorted(pres_start, centers, side='right') - 1
image_trial = pres_img[j].astype(np.int64)
```

iii. The trajectory justification is the same as for the neural alignment: one shared change-centered 250 ms grid for all streams, with stimulus values evaluated at those common bins.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image_change` is derived from `ds.stimulus_presentations`, specifically the start times of flashes marked `is_change` within the change-detection block. It is not derived directly from `trial.go` plus `trial.change_time`.

ii.
```python
shown = sp[(~sp.omitted) & (sp.image_name != 'omitted')].sort_values('start_time')
pres_start = shown.start_time.values
change_start = pres_start[shown.is_change.values]
```

iii. The trajectory indicates the AI wanted image-change labels tied to actual changed-image presentations on the stimulus timeline, consistent with its choice to represent stimulus variables from `stimulus_presentations` rather than from the trial table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each trial bin center, the AI finds the most recent changed-image flash and marks the bin as 1 if it falls within 750 ms of that flash; otherwise 0. This produces a 750 ms-wide "change" interval rather than a single-bin or post-change-until-end label.

ii.
```python
jc = np.searchsorted(change_start, centers, side='right') - 1
change_trial = np.zeros(len(centers), dtype=np.int64)
good = jc >= 0
change_trial[good] = (
    (centers[good] - change_start[jc[good]]) < FLASH_INTERVAL).astype(np.int64)
```

iii. The AI justified this as matching the paper's 750 ms image-presentation interval convention: one 250 ms flash plus the following 500 ms gray screen.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is not thresholded from a continuous value; the code directly creates a binary category with values 0 (`no_change`) and 1 (`change`).

ii.
```python
change_trial = np.zeros(len(centers), dtype=np.int64)
...
'output_values': [
    list(image_names),
    ['no_change', 'change'],
```

iii. The trajectory treats image change as a binary event variable throughout. No alternative discretization was considered.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned on the same change-centered 250 ms trial grid as neural activity, using trial-bin centers and changed-flash onset times.

ii.
```python
edges = trial.change_time + BIN_SIZE * np.arange(k0, k1 + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
...
jc = np.searchsorted(change_start, centers, side='right') - 1
```

iii. The trajectory explicitly says all data streams were to share one common 250 ms grid anchored at the change time.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, using both `timestamps` and `speed`.

ii.
```python
run = ds.running_speed
run_t = run.timestamps.values
run_v = interp_nans(run.speed.values)
```

iii. The trajectory treats `running_speed` as the standard locomotion stream from the Allen dataset and never considers another source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates/fills NaNs in the native running speed series, converts it to a cumulative sum, averages it within each 250 ms trial bin, and finally discretizes the concatenated within-session binned values into five equal-percentile bins.

ii.
```python
run_v = interp_nans(run.speed.values)
run_cum = np.concatenate([[0.0], np.cumsum(run_v)])
...
rsum, rn = bin_sum(run_cum, run_t, edges)
run_trial = rsum / rn
...
run_code = quantile_bins(np.concatenate([t['run'] for t in trials]))
```

iii. The trajectory justifies the binning by saying all streams should live on the same change-centered 250 ms grid, and the metadata justifies session-wise quantization by arguing mice differ substantially in running statistics.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five within-session equal-percentile bins. The code computes the quantile edges from all trial bins in that session and encodes categories `0..4`.

ii.
```python
def quantile_bins(values, nbins=NQUANTILES):
    edges = np.quantile(values, np.linspace(0, 1, nbins + 1)[1:-1])
    return np.searchsorted(edges, values, side='right').astype(np.int64)

run_code = quantile_bins(np.concatenate([t['run'] for t in trials]))
```

iii. The metadata states that per-session quantization is meaningful because running distributions vary strongly across mice and sessions. That rationale appears in the final code comments rather than a separate trajectory note.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by averaging native running-speed samples inside the exact same 250 ms trial bins used for the neural data.

ii.
```python
edges = trial.change_time + BIN_SIZE * np.arange(k0, k1 + 1)
...
rsum, rn = bin_sum(run_cum, run_t, edges)
run_trial = rsum / rn
```

iii. The trajectory repeatedly describes a shared fixed grid for all streams, with behavior binned from its own native timestamps rather than first interpolated to ophys frames.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking`, specifically `pupil_area` and `timestamps`. The code converts area to diameter via `2 * sqrt(area / pi)`.

ii.
```python
eye = ds.eye_tracking
...
pupil_area = interp_nans(eye.pupil_area.values)
pupil_v = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
eye_t = eye.timestamps.values
```

iii. The trajectory and metadata describe this as producing a diameter-valued signal from the eye-tracking stream. The AI did not use `pupil_width`; it preferred an explicit area-to-diameter conversion.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI interpolates NaNs in `pupil_area`, converts area to diameter, computes cumulative sums, averages the diameter within each 250 ms trial bin, and discretizes the within-session binned values into five equal-percentile categories.

ii.
```python
pupil_area = interp_nans(eye.pupil_area.values)
...
pupil_v = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
pupil_cum = np.concatenate([[0.0], np.cumsum(pupil_v)])
...
psum, pn = bin_sum(pupil_cum, eye_t, edges)
pupil_trial = psum / pn
...
pupil_code = quantile_bins(np.concatenate([t['pupil'] for t in trials]))
```

iii. The trajectory justification mirrors the running-speed case: keep all streams on the same trial grid and discretize per session because the signal scale is session-specific. The metadata adds that blinks are linearly interpolated.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five within-session equal-percentile bins using the concatenated binned pupil values from that session's retained trials.

ii.
```python
pupil_code = quantile_bins(np.concatenate([t['pupil'] for t in trials]))
...
['pupil_quintile_%d' % (i + 1) for i in range(NQUANTILES)]
```

iii. The explicit rationale in the code comments is that pupil size is measured in camera pixels and is therefore session-specific, so per-session quantization is more meaningful than a global scale.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by averaging native eye-tracking samples inside the same change-centered 250 ms bins as the neural data.

ii.
```python
edges = trial.change_time + BIN_SIZE * np.arange(k0, k1 + 1)
...
psum, pn = bin_sum(pupil_cum, eye_t, edges)
pupil_trial = psum / pn
```

iii. The trajectory explicitly says pupil should be binned from its own native timestamps onto the shared trial grid rather than first forced onto the ophys frame times.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean outcome columns in `ds.trials`: `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
oc = [k for k in OUTCOME_NAMES if bool(trial[k])]
if len(oc) != 1:
    continue
```

iii. The trajectory consistently describes trial outcome using those four canonical labels for go and catch trials. No other outcome source is considered.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps the single true outcome label to its index in `OUTCOME_NAMES` and tiles that integer across every time bin of the trial.

ii.
```python
trials.append({...,
               'outcome': OUTCOME_NAMES.index(oc[0])})
...
output.append(np.stack([
    t['image'],
    t['change'],
    run_code[pos:pos + T],
    pupil_code[pos:pos + T],
    np.full(T, t['outcome'], dtype=np.int64),
]).astype(np.int64))
```

iii. The trajectory states that trial outcome is static per trial, so it is stored as a constant categorical row across time bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or invalid data are mostly handled by dropping the affected session or trial. NaNs in running speed and pupil area are linearly interpolated; sessions with no usable eye tracking or no usable running values are dropped; trials lacking full stream coverage or enough per-bin samples are dropped; worker exceptions are caught and the session is skipped.

ii.
```python
def interp_nans(x):
    ...
    if bad.all():
        return None
    if bad.any():
        ...
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x

if run_v is None:
    return None
...
if eye is None or len(eye) == 0:
    return None
...
if len(edges) < 3 or edges[0] < tmin or edges[-1] > tmax:
    continue
...
except Exception as exc:
    print('FAILED session %s: %s: %s' % (osid, type(exc).__name__, exc), flush=True)
    return None
```

iii. The trajectory explicitly mentions dropping one session for missing eye tracking and justifies the coverage checks as necessary to keep all streams aligned on the shared 250 ms grid. It treats skipping failed sessions as a pragmatic way to let the full conversion finish.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is loading every experiment in `convert_session`, especially for multi-plane sessions, and then binning the full-session neural and behavioral streams into trials. The code uses multiprocessing to parallelize this across sessions.

ii.
```python
for oeid in oeids:
    ds = cache.get_behavior_ophys_experiment(int(oeid))
    ...

with Pool(args.workers) as pool:
    results = pool.map(_worker, items)
```

iii. The trajectory repeatedly uses pilot runs, full runs, and multiprocessing and treats conversion as expensive enough to warrant `Pool(..., workers=12)`. It does not single out another heavier step.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main non-vectorized loops are the per-trial loop over `tr.iterrows()`, the per-plane loop inside each trial when binning neural data, and the Python list lookup used to map image names to integer IDs. Some of this is numerically vectorized inside helper functions, but trial construction is still largely Python-loop driven.

ii.
```python
for _, trial in tr.iterrows():
    ...
    for p in planes:
        vals, nfr = bin_sum(p['cum'], p['ts'], edges)
        ...

pres_img = np.array([image_names.index(n) for n in shown.image_name.values])
```

iii. The trajectory never discusses vectorization directly. This assessment is inferred from the final code structure.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats cache construction and experiment loading work in every worker process, recomputes session-local image-name indexing from strings for each session, and separately recomputes per-session quantile binning for running and pupil after trial extraction. It also went through multiple pilot/full conversions during development, though that repetition is in the trajectory rather than the final script.

ii.
```python
def convert_session(osid, oeids, neural_key='events'):
    ...
    cache = VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)
    for oeid in oeids:
        ds = cache.get_behavior_ophys_experiment(int(oeid))

image_names = sorted(set(sp.image_name.unique()) - {'omitted'})
pres_img = np.array([image_names.index(n) for n in shown.image_name.values])
...
run_code = quantile_bins(np.concatenate([t['run'] for t in trials]))
pupil_code = quantile_bins(np.concatenate([t['pupil'] for t in trials]))
```

iii. The trajectory does not justify this as an intentional optimization tradeoff. It is mostly a consequence of structuring the conversion around independent per-session worker calls.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores substantial metadata that the decoder does not use, including `cell_specimen_ids`, imaging depths, frame rates, equipment name, and per-session bookkeeping in `metadata['session_info']`. It also carries per-session `image_names` through `results` only to assert consistency before discarding the per-session copies.

ii.
```python
info = {
    'ophys_session_id': int(osid),
    'ophys_experiment_ids': [p['oeid'] for p in planes],
    ...
    'imaging_depths': [p['depth'] for p in planes],
    'ophys_frame_rates': [p['frame_rate'] for p in planes],
    ...
    'cell_specimen_ids': sum([p['cell_ids'] for p in planes], []),
}
...
image_names = results[0]['image_names']
assert all(r['image_names'] == image_names for r in results), 'image sets differ'
...
'session_info': [r['info'] for r in results],
```

iii. The trajectory does not present these as necessary for decoding. They appear to be documentation-heavy metadata and validation aids rather than downstream-required computations.
