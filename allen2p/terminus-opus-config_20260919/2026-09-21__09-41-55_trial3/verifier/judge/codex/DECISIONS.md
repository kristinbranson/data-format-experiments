# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI restricts loading to NWB files that are already present under `/app/data`, intersects those file IDs with the AllenSDK experiment table, then further filters to active familiar sessions (`OPHYS_1_images_A`, `OPHYS_3_images_A`). It groups imaging planes by `ophys_session_id` and loads each plane with `get_behavior_ophys_experiment()` from a local AllenSDK cache.

ii.
```python
def local_experiment_table():
    ids = sorted(int(re.search(r'(\d+)\.nwb', f).group(1))
                 for f in glob.glob(os.path.join(NWB_DIR, '*.nwb')))
    et = get_cache().get_ophys_experiment_table()
    et = et.loc[et.index.intersection(ids)]
    return et

def select_experiments():
    et = local_experiment_table()
    et = et[~et.passive]
    et = et[et.session_type.isin(FAMILIAR_SESSION_TYPES)]
    return et
```

```python
for eid in exp_ids:
    ds = bc.get_behavior_ophys_experiment(int(eid))
```

iii. In `CONVERSION_NOTES.md`, the AI says the local file set defines the available dataset, passive sessions lack meaningful trial outcomes, and familiar image-set A sessions match the paper's neural-analysis restriction and give a consistent 8-image output space.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the converted sessions. They are converted to strings, sorted, and referenced by `subject_idx`.

ii.
```python
meta['mouse_id'] = str(meta['mouse_id'])
...
subjects = sorted({r['mouse_id'] for r in good})
subj_to_idx = {s: i for i, s in enumerate(subjects)}
...
subject_idx=np.array([subj_to_idx[r['mouse_id']] for r in good], dtype=np.int64),
```

iii. The notes explicitly map `experiment_table.mouse_id` to `subjects`/`subject_idx` and treat mice as the subject boundary.

## 1-c. How are the data split into sessions?

i. Sessions are grouped by `ophys_session_id`. If several experiments belong to the same session, their imaging planes are merged into one converted session.

ii.
```python
for sid, grp in et.groupby('ophys_session_id'):
    meta = grp.iloc[0][['mouse_id', 'cre_line', 'session_type', 'experience_level',
                        'equipment_name', 'project_code', 'image_set']].to_dict()
    sessions.append((int(sid), list(grp.index.values), meta))
sessions.sort(key=lambda s: s[0])
```

iii. The notes justify this by saying one NWB file is one imaging plane, while one `ophys_session_id` is one simultaneous behavioral/ophys session; mesoscope planes should therefore be merged.

## 1-d. How are the data split into trials?

i. Trials are defined from the AllenSDK `trials` table, but not by using each row's full `start_time` to `stop_time`. Instead, the AI keeps go/catch trials and cuts a fixed window around `change_time`: `[-2.25, +3.75]` seconds, yielding 8 bins of 750 ms each.

ii.
```python
tr = beh['trials']
keep = (tr['go'].astype(bool) | tr['catch'].astype(bool)) & ~tr['auto_rewarded'].astype(bool)
tr = tr[keep]
...
offs = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
edges = change_times[:, None] + offs[None, :]
centers = edges[:, :-1] + BIN_SIZE / 2.0
```

iii. The notes say the first implementation used the full trial differently, but the final decision was revised because the paper analyzes 750 ms image-presentation intervals and because a fixed change-centered window avoids overlapping other changes while still fitting inside every kept trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only go or catch trials, excludes auto-rewarded trials, asserts every kept trial has a `change_time`, drops sessions without eye tracking, and later drops any trial with non-finite binned running or pupil values. Sessions with fewer than 2 usable trials are skipped.

ii.
```python
keep = (tr['go'].astype(bool) | tr['catch'].astype(bool)) & ~tr['auto_rewarded'].astype(bool)
tr = tr[keep]
assert tr['change_time'].notna().all(), 'missing change_time on a go/catch trial'
...
if beh['eye'] is None or len(beh['eye']) == 0:
    return dict(session_id=session_id, skip='no eye tracking', timings=timings)
...
bad = (~np.isfinite(run_binned).all(axis=1)) | (~np.isfinite(pupil_binned).all(axis=1))
...
if good.sum() < 2:
    return dict(session_id=session_id, skip=f'<2 usable trials ({int(good.sum())})',
                timings=timings)
```

iii. The notes justify excluding auto-rewarded trials from the task, dropping no-eye-tracking sessions because pupil is a required output, and discarding trials with unresolved behavioral NaNs rather than imputing them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data come from the AllenSDK `events` table, using the selected column in `ds.events` and defaulting to `events` rather than `filtered_events` or `dff_traces`.

ii.
```python
ap.add_argument('--neural-signal', type=str, default='events',
                choices=['events', 'filtered_events'])
...
ev = np.vstack(ds.events[signal].values).astype(np.float64)
```

iii. The notes and trajectory say this was chosen because the paper states that its neural analyses use detected calcium events, while `filtered_events` is described by the SDK as a visualization-oriented smoothing.

## 2-b. How is the `neural` data processed?

i. The AI stacks neurons across all planes in a session, bins event magnitudes into the fixed trial bins by mean within each bin, and then applies per-neuron normalization. By default it uses per-neuron z-scoring within session across all selected go/catch trial bins.

ii.
```python
sums, counts = bin_sum_count(p['events'], p['ts'], edges)
neural_blocks.append((sums / counts[None, :, :]).astype(np.float32))
...
neural = np.concatenate(neural_blocks, axis=0)
...
if normalize in ('std', 'zscore'):
    flat = neural.reshape(neural.shape[0], -1)
    mu = flat.mean(axis=1, keepdims=True)
    sd = flat.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    if normalize == 'zscore':
        flat = (flat - mu) / sd
```

iii. The revision note says 750 ms bins match the paper's unit of analysis and improve SNR for sparse events, while z-scoring was added because the provided decoder starts from a raw SVD and otherwise overweights high-amplitude cells.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional per-neuron filtering is applied beyond what the AllenSDK already exposes. The code includes all cells returned in `ds.events`.

ii.
```python
planes.append(dict(eid=int(eid), events=ev, ts=ts,
                   cell_ids=list(ds.events.index.values),
                   structure=ds.metadata['targeted_structure'],
                   depth=ds.metadata['imaging_depth'],
                   frame_rate=ds.metadata['ophys_frame_rate']))
```

iii. The notes cite AllenSDK behavior-ophys objects and state that valid ROI filtering has already been applied upstream, so no extra cell QC was added.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to `trials.change_time`, not to trial start. Each trial uses the same set of offsets from that change/sham-change event.

ii.
```python
change_times = tr['change_time'].values.astype(np.float64)
offs = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
edges = change_times[:, None] + offs[None, :]
...
neural_trials = [np.ascontiguousarray(neural[:, i, :]) for i in range(n_good)]
```

iii. The notes justify this as matching the reference trial-response code and the paper's change-aligned analyses; they also state that every `change_time` coincides with a flash onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The final converted data use 750 ms bins. Yes: temporal rebinning is applied, with neural values averaged within each 750 ms image-presentation interval.

ii.
```python
BIN_SIZE = 0.750
OFF_START = -2.25
OFF_END = 3.75
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
...
ap.add_argument('--bin-size', type=float, default=BIN_SIZE)
```

iii. The revision note says the AI changed from 250 ms to 750 ms after testing, because the paper's unit of analysis is the 750 ms image-presentation interval and sparse event data were too noisy in 250 ms bins.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations` in the `change_detection` block, specifically `image_name` plus `omitted` status, rather than directly from `trials.initial_image_name` and `trials.change_image_name`.

ii.
```python
stim = beh['stim']
stim = stim[stim.stimulus_block_name.astype(str).str.contains('change_detection')]
...
omitted = stim['omitted'].fillna(False).astype(bool).values
names = stim['image_name'].astype(str).values
```

iii. The notes say this follows the paper's image-presentation-interval convention and allows omitted flashes to be handled explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code forward-fills image identity across omitted flashes, enumerates the resulting image names into a sorted global list for the session set, and labels each time bin by the flash interval containing that bin's center.

ii.
```python
names_ff = pd.Series(np.where(omitted, None, names)).ffill().bfill().values
image_names = sorted(set(names_ff[~pd.isna(names_ff)]) - {'omitted'})
name_to_idx = {n: i for i, n in enumerate(image_names)}
img_idx = np.array([name_to_idx[n] for n in names_ff], dtype=np.int64)
...
j = np.searchsorted(fstart, centers, side='right') - 1
image_identity = img_idx[j]
```

iii. The notes justify forward-filling omissions by saying omissions replace repeat flashes, so the latent image identity stays the same; the sorted mapping gives consistent codes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned by evaluating the stimulus interval containing each neural bin center. Therefore image identity and neural activity share the same 8 change-centered bins.

ii.
```python
centers = edges[:, :-1] + BIN_SIZE / 2.0
...
j = np.searchsorted(fstart, centers, side='right') - 1
image_identity = img_idx[j]
```

iii. The notes say all streams are on the Allen sync clock and binning every stream on the same edges guarantees alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change`, after restricting to the `change_detection` stimulus block. Catch trials are implicitly all-zero because sham changes have `is_change == False`.

ii.
```python
stim = stim[stim.stimulus_block_name.astype(str).str.contains('change_detection')]
...
is_change = stim['is_change'].fillna(False).astype(bool).values
...
image_change = is_change[j].astype(np.int64)
```

iii. The notes say the bin that begins at `change_time` is the actual image-change interval on go trials, while catch trials are sham changes and should not be labeled as real image changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. No smoothing or extra transform is applied beyond taking `is_change` from the flash interval selected for each bin center and casting it to integers.

ii.
```python
j = np.searchsorted(fstart, centers, side='right') - 1
...
image_change = is_change[j].astype(np.int64)
```

iii. The notes frame this as matching the image-presentation-interval analysis unit used in the paper.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary categorical output with values `0 = no_change` and `1 = change`.

ii.
```python
output_names=['image_identity', 'image_change', 'running_speed_quintile',
              'pupil_diameter_quintile', 'trial_outcome'],
output_values=[list(image_names),
               ['no_change', 'change'],
               [f'speed_q{i+1}' for i in range(NQUANTILES)],
               [f'pupil_q{i+1}' for i in range(NQUANTILES)],
               list(OUTCOME_NAMES)],
```

iii. The choice is justified in the notes as a direct real-change versus not-change label, with sham changes left at zero.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is assigned using the same change-centered bin centers that are used for neural activity, so both share the same 8-bin trial structure.

ii.
```python
centers = edges[:, :-1] + BIN_SIZE / 2.0
...
image_change = is_change[j].astype(np.int64)
...
neural_trials = [np.ascontiguousarray(neural[:, i, :]) for i in range(n_good)]
```

iii. The notes say all streams are binned on the same change-aligned edges, and the sanity checks verify the expected change-bin pattern.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, specifically its `speed` and `timestamps` columns.

ii.
```python
run = beh['run']
run_t = run['timestamps'].values.astype(np.float64)
run_v = run['speed'].values.astype(np.float64)
```

iii. The notes identify AllenSDK `running_speed` as the filtered locomotion signal in cm/s and choose it over the raw wheel signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code computes a NaN-aware mean running speed within each trial bin, then discretizes those binned values into quintiles later in `main()`.

ii.
```python
run_binned = bin_mean_1d(run_v, run_t, edges, nan_aware=True)
...
if args.quantile_scope == 'global':
    run_all = np.concatenate([r['run_binned'].ravel() for r in good])
    _, run_edges_g = quantile_bin(run_all)
...
run_cls = np.digitize(r['run_binned'], run_edges)
```

iii. The final justification in the notes is that percentile bins should be global so the same class index means the same physical speed across sessions; the AI explicitly revised an earlier per-session-bin decision after testing.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 percentile bins. By default the thresholds are global 20/40/60/80th percentiles over all kept session bins, though the script leaves a `--quantile-scope session` option.

ii.
```python
ap.add_argument('--quantile-scope', type=str, default='global',
                choices=['global', 'session'])
...
_, run_edges_g = quantile_bin(run_all)
...
run_cls = np.digitize(r['run_binned'], run_edges)
```

iii. The trajectory says the AI switched from session-level to global bins because the decoder uses a shared readout across sessions and the task wording literally says five equal percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is not first resampled to ophys frames. Instead, it is averaged directly on the same change-centered bin edges used for neural activity.

ii.
```python
edges = change_times[:, None] + offs[None, :]
...
run_binned = bin_mean_1d(run_v, run_t, edges, nan_aware=True)
...
neural_blocks.append((sums / counts[None, :, :]).astype(np.float32))
```

iii. The notes justify this by saying all streams already share sync-clock timestamps, so common bin edges are sufficient for alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `eye_tracking.pupil_area`, converted to diameter as `2 * sqrt(area / pi)`, plus `eye_tracking.timestamps`.

ii.
```python
eye = beh['eye']
eye_t = eye['timestamps'].values.astype(np.float64)
pupil_diam = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
```

iii. The notes say the AI checked the SDK code and concluded `pupil_area` is based on a circular-area convention, so area-derived diameter is the intended measure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code converts pupil area to diameter, linearly interpolates only short NaN gaps (`<= 1 s`), averages within each trial bin, and later discretizes the binned values into quintiles. Trials with remaining non-finite pupil bins are dropped.

ii.
```python
PUPIL_INTERP_MAX_GAP = 1.0
...
pupil_diam = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_INTERP_MAX_GAP)
pupil_binned = bin_mean_1d(pupil_diam, eye_t, edges, nan_aware=True)
...
bad = (~np.isfinite(run_binned).all(axis=1)) | (~np.isfinite(pupil_binned).all(axis=1))
```

iii. The notes justify short-gap interpolation as blink handling while preserving longer tracking failures as missing data; the missing trials are then removed rather than imputed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 percentile bins, globally by default and per-session only if the optional CLI switch is used.

ii.
```python
_, pupil_edges_g = quantile_bin(pupil_all)
...
pupil_cls = np.digitize(r['pupil_binned'], pupil_edges)
...
output_values=[list(image_names),
               ['no_change', 'change'],
               [f'speed_q{i+1}' for i in range(NQUANTILES)],
               [f'pupil_q{i+1}' for i in range(NQUANTILES)],
               list(OUTCOME_NAMES)],
```

iii. The same revision note that motivated global running-speed bins also motivated global pupil bins so categories have the same physical meaning across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by binning the eye-tracking stream on the same change-centered trial edges as the neural signal.

ii.
```python
edges = change_times[:, None] + offs[None, :]
...
pupil_binned = bin_mean_1d(pupil_diam, eye_t, edges, nan_aware=True)
...
neural_trials = [np.ascontiguousarray(neural[:, i, :]) for i in range(n_good)]
```

iii. The notes state that all streams are timestamped on the same sync clock, so identical bin edges guarantee alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome_mat = np.stack([tr[c].astype(bool).values for c in OUTCOME_NAMES], axis=1)
outcome = np.argmax(outcome_mat, axis=1).astype(np.int64)
```

iii. The notes say these four flags exactly partition the selected go/catch non-auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome booleans are converted to class indices via `argmax` over the ordered outcome matrix, and the resulting class is broadcast across all bins in that trial.

ii.
```python
outcome_mat = np.stack([tr[c].astype(bool).values for c in OUTCOME_NAMES], axis=1)
outcome = np.argmax(outcome_mat, axis=1).astype(np.int64)
...
output_trials.append(np.stack([
    image_identity[i],
    image_change[i],
    np.zeros(NBINS, dtype=np.int64),
    np.zeros(NBINS, dtype=np.int64),
    np.full(NBINS, outcome[i], dtype=np.int64),
]).astype(np.int64))
```

iii. The notes justify keeping the natural four-way task outcome categories and broadcasting them because the file format expects per-time-bin outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data conservatively: sessions with no eye tracking are skipped; short pupil gaps are interpolated; trials with any remaining missing binned running or pupil values are dropped; sessions with too few remaining trials are dropped; conversion exceptions skip the whole session.

ii.
```python
if beh['eye'] is None or len(beh['eye']) == 0:
    return dict(session_id=session_id, skip='no eye tracking', timings=timings)
...
if t[e - 1] - t[s] <= max_gap:
    x[s:e] = xi[s:e]
...
bad = (~np.isfinite(run_binned).all(axis=1)) | (~np.isfinite(pupil_binned).all(axis=1))
...
except Exception as ex:
    return dict(session_id=int(session_id), skip='ERROR: ' + repr(ex),
                traceback=traceback.format_exc(), timings=timings)
```

iii. The notes explicitly say long eye-tracking failures are left as NaN so affected trials can be dropped, and that dropping the one no-eye-tracking session is preferable to inventing a pupil output.

## 9-a. What are the most time-consuming steps of the code?

i. The code structure implies the main costs are loading AllenSDK experiments/NWB files and per-session conversion over all planes. The AI also records per-session `load`, `neural`, `stim`, and `behavior` timings.

ii.
```python
t0 = time.time()
for eid in exp_ids:
    ds = bc.get_behavior_ophys_experiment(int(eid))
...
timings['load'] = time.time() - t0
...
timings['neural'] = time.time() - t0
...
timings['behavior'] = time.time() - t0
```

iii. The notes say loading each NWB once per session is the main bottleneck and describe later vectorization/multiprocessing as attempts to reduce processing overhead around that I/O.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the heavy binning step. `bin_sum_count()` bins all neurons and all trials at once via cumulative sums and `searchsorted`, replacing more obvious per-trial/per-bin loops. Remaining Python loops are mostly over planes, sessions, and final per-trial assembly.

ii.
```python
def bin_sum_count(values, timestamps, edges_flat):
    v = np.atleast_2d(values).astype(np.float64)
    idx = np.searchsorted(timestamps, edges_flat)
    cs = np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(v, axis=1)], axis=1)
    sums = cs[:, idx[:, 1:]] - cs[:, idx[:, :-1]]
    counts = (idx[:, 1:] - idx[:, :-1]).astype(np.float64)
    return sums, counts
```

iii. The notes explicitly call this vectorized binning a speedup added during development and contrast it with slower trial-by-trial masking.

## 9-c. What processing does the code repeat multiple times?

i. The final script does a second pass over all converted sessions to compute global running/pupil quantile edges and then revisit every session to assign classes. It also recreates the AllenSDK cache inside each worker/session conversion, and it computes some per-session checks/plot data in addition to the main conversion.

ii.
```python
def get_cache():
    return bpc.VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)
...
bc = get_cache()
```

```python
if args.quantile_scope == 'global':
    run_all = np.concatenate([r['run_binned'].ravel() for r in good])
    pupil_all = np.concatenate([r['pupil_binned'].ravel() for r in good])
...
for r in good:
    ...
    run_cls = np.digitize(r['run_binned'], run_edges)
```

iii. This is not singled out in the notes as a problem; it is an artifact of the decision to defer discretization until global edges are known and of running each session conversion independently for multiprocessing.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores intermediate values that are not part of the final saved dataset: `depths`, `cell_ids`, `timings`, raw `run_binned`/`pupil_binned`, per-session plotting arguments, and detailed sanity-check metadata. Some of these are used only for diagnostics and are then deleted or omitted from the final pickle.

ii.
```python
result = dict(
    ...
    run_binned=run_binned.astype(np.float32),
    pupil_binned=pupil_binned.astype(np.float32),
    depths=np.array(depth_list),
    cell_ids=np.array(cellid_list),
    ...
    timings=timings,
)
...
if '_plotargs' in r:
    plot_processing(r, run_edges, pupil_edges)
    del r['_plotargs']
del r['run_binned'], r['pupil_binned']
```

iii. The notes justify these as validation aids and debugging support, especially the sanity checks and processing plots, rather than as quantities needed by downstream decoder training.
