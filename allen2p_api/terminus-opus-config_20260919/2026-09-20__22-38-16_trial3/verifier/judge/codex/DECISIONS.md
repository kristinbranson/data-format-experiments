# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data exclusively through the AllenSDK cache, but only from NWB experiment files already present in the local cache. It builds an experiment table restricted to downloaded NWBs, then filters to active (`passive == False`) experiments before grouping sessions.

ii.
```python
def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    return VisualBehaviorOphysProjectCache.from_local_cache(
        cache_dir=CACHE_DIR, use_static_cache=False)

def downloaded_experiment_table(cache):
    et = cache.get_ophys_experiment_table()
    ids = sorted(int(os.path.basename(f).split('_')[-1].split('.')[0])
                 for f in glob.glob(NWB_GLOB))
    return et.loc[et.index.isin(ids)].copy()

cache = get_cache()
et = downloaded_experiment_table(cache)
act = et[~et['passive'].astype(bool)].copy()
```

iii. In `CONVERSION_NOTES.md`, the AI says it used the AllenSDK local cache because only 284 experiment NWBs were present locally, and it restricted to active sessions because passive sessions have no usable trial-outcome behavior.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `mouse_id`. During assembly, each session’s `mouse_id` is added once to `subjects`, and `subject_idx` stores the corresponding index per session.

ii.
```python
mouse = info['mouse_id']
if mouse not in subjects:
    subjects.append(mouse)
data['subject_idx'].append(subjects.index(mouse))
```

iii. The notes explicitly state: “`subjects` = `mouse_id` strings; `subject_idx` = index per session.”

## 1-c. How are the data split into sessions?

i. A session is defined as one `ophys_session_id`. All experiments/imaging planes sharing that `ophys_session_id` are grouped together and processed as one simultaneous recording session.

ii.
```python
groups = act.groupby('ophys_session_id')
session_ids = sorted(groups.groups.keys())

for sid in session_ids:
    exp_ids = list(groups.get_group(sid).index.values)
```

iii. The AI justifies this in the notes as the “physically correct notion of a simultaneous population recording,” with all planes from one ophys session concatenated along the neuron axis.

## 1-d. How are the data split into trials?

i. Trials come from `ref.trials`, but instead of using each trial’s full `start_time` to `stop_time` window, the AI keeps go/catch trials with finite `change_time` and extracts a fixed window from `-3 s` to `+3 s` around `change_time`.

ii.
```python
trials = ref.trials
keep = (trials['go'].astype(bool) | trials['catch'].astype(bool))
tr = trials[keep].copy()
tr = tr[np.isfinite(tr['change_time'].values)]

change_times = tr['change_time'].values.astype(np.float64)
edges = change_times[:, None] + (OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
```

iii. The notes say the AI chose `change_time` as the alignment event and a symmetric `[-3.0, +3.0] s` window because it stays inside the trial and spans eight flash cycles.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only go/catch trials with finite `change_time`, then drops trials whose fixed `[-3, +3] s` window would fall outside the available ophys/running/eye/stimulus data. It also drops trials lacking one of the four standard outcomes and drops sessions with fewer than two remaining trials.

ii.
```python
keep = (trials['go'].astype(bool) | trials['catch'].astype(bool))
tr = trials[keep].copy()
tr = tr[np.isfinite(tr['change_time'].values)]

valid = ((change_times + OFF_START) >= t_lo) & ((change_times + OFF_END) <= t_hi)
tr = tr[valid]

outcome = np.full(ntrials, -1, dtype=np.int64)
outcome[tr['hit'].astype(bool).values] = 0
outcome[tr['miss'].astype(bool).values] = 1
outcome[tr['false_alarm'].astype(bool).values] = 2
outcome[tr['correct_reject'].astype(bool).values] = 3
ok = outcome >= 0
```

iii. The notes justify go/catch selection as matching `trial_masks.contingent_trials`, and justify dropping out-of-window trials because the fixed alignment window must be valid for every output stream.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. In the final code path, neural data are derived from `dff_traces.dff` by default. The script also supports `events` and `filtered_events` as optional alternatives via `--neural`, but `--neural dff` is the default.

ii.
```python
ap.add_argument('--neural', default='dff',
                choices=['events', 'filtered_events', 'dff'])

tbl = e.events if neural_source in ('events', 'filtered_events') else e.dff_traces
col = {'events': 'events', 'filtered_events': 'filtered_events',
       'dff': 'dff'}[neural_source]
traces = np.vstack(tbl[col].values)
```

iii. In the notes, the AI says it initially intended to use calcium `events` to match the paper, but switched the default to dF/F after comparing decoder performance and finding dF/F worked better with the provided instantaneous decoder.

## 2-b. How is the `neural` data processed?

i. Neural traces from all planes in a session are vertically concatenated across neurons, optionally z-scored per neuron, then averaged within fixed 250 ms bins in the `[-3, +3] s` trial window.

ii.
```python
for eid, e in exps:
    tbl = e.events if neural_source in ('events', 'filtered_events') else e.dff_traces
    traces = np.vstack(tbl[col].values)
    if zscore:
        mu = traces.mean(axis=1, keepdims=True)
        sd = traces.std(axis=1, keepdims=True)
        traces = (traces - mu) / np.maximum(sd, 1e-9)
    binned, counts = bin_means(traces, ts, edges)
    neural_planes.append(binned.astype(np.float32))
neural = np.concatenate(neural_planes, axis=0)
```

iii. The notes justify 250 ms binning as matching the image-on period and as a way to make one common time base work for both 31 Hz and 11 Hz rigs. The switch to dF/F is justified by sample decoder performance.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply additional per-neuron quality filtering beyond what the AllenSDK already exposes. It relies on the SDK’s default valid-ROI filtering.

ii.
```python
if len(tbl) == 0:
    continue
...
neuron_selection=('all cells released by the AllenSDK, i.e. valid ROIs only '
                  '(exclude_invalid_rois=True is the SDK default)')
```

iii. The notes explicitly state that no extra filtering is needed because the Allen pipeline already filters invalid ROIs before exposing `dff_traces` and `events`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to `change_time`, not trial start. The code constructs bin edges relative to `change_time` and bins ophys frames into those trial-centered windows.

ii.
```python
change_times = tr['change_time'].values.astype(np.float64)
edges = change_times[:, None] + (OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
...
binned, counts = bin_means(traces, ts, edges)
```

iii. The notes call `change_time` the “canonical alignment” for go/catch trials and justify the symmetric window by the flash-cycle structure of the task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins the data to a fixed 250 ms resolution. Each trial has 24 bins covering 6 s total.

ii.
```python
BIN_SIZE = 0.25
OFF_START = -3.0
OFF_END = 3.0
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 24
```

iii. The notes justify 250 ms as the image-on duration and one-third of the 750 ms flash cycle, and as large enough to work across both single-plane and multiscope sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `image_identity` is derived from `stimulus_presentations.image_name` within the `change_detection_behavior` block, not from the trial table’s `initial_image_name` / `change_image_name`.

ii.
```python
sp = ref.stimulus_presentations
beh = sp[sp['stimulus_block_name'] == BEHAVIOR_BLOCK].copy()
pres_start = beh['start_time'].values.astype(np.float64)
pres_image = beh['image_name'].values.astype(object)
```

iii. The notes justify this as using the paper’s “image-presentation interval” convention and allowing omitted flashes to be represented explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each time bin, the code finds the most recent stimulus presentation onset preceding that bin’s center and assigns that presentation’s `image_name`. It then maps names to integer codes using a fixed global list of 16 natural images plus `omitted`.

ii.
```python
IMAGE_VALUES = IMAGE_NAMES + ['omitted']
IMAGE_TO_IDX = {n: i for i, n in enumerate(IMAGE_VALUES)}

pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
pidx = np.clip(pidx, 0, len(pres_start) - 1)
img_names = pres_image[pidx]
img_identity = np.array([IMAGE_TO_IDX[n] for n in img_names],
                        dtype=np.int64).reshape(ntrials, NBINS)
```

iii. The notes say this labels each bin with the image of the 750 ms presentation interval containing the bin center, and uses a separate `omitted` class for omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned at the same 250 ms trial bins as the neural data. The label for each bin is chosen using that bin’s center time.

ii.
```python
edges = change_times[:, None] + (OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
...
img_identity = np.array([IMAGE_TO_IDX[n] for n in img_names],
                        dtype=np.int64).reshape(ntrials, NBINS)
```

iii. The notes justify this as bin-level alignment on the common sync clock, with stimulus identity defined over image-presentation intervals rather than native ophys frames.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image_change` is derived from `stimulus_presentations.is_change` in the behavior block. The trial’s `change_time` is used only indirectly through the change-centered binning.

ii.
```python
pres_change = beh['is_change'].astype(bool).values
...
img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)
```

iii. The notes justify this as representing whether the current 750 ms image-presentation interval is a changed image interval.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code maps each bin center to the most recent stimulus presentation onset and copies that presentation’s `is_change` flag into the bin.

ii.
```python
pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
pidx = np.clip(pidx, 0, len(pres_start) - 1)
img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)
```

iii. The notes describe this as the paper’s image-interval convention: bins inside the changed presentation interval get value 1, others 0.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary in the source stimulus table, so the code simply stores it as integer labels `0`/`1`, with output values `['no_change', 'change']`.

ii.
```python
img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)
...
output_values=[IMAGE_VALUES,
               ['no_change', 'change'],
               [f'quintile_{i}' for i in range(NQUANTILES)],
               [f'quintile_{i}' for i in range(NQUANTILES)],
               OUTCOME_VALUES]
```

iii. No separate thresholding justification is given beyond using the binary change indicator already provided by the stimulus table.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. `image_change` is aligned to the same 250 ms bins as the neural data, again using bin centers.

ii.
```python
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
...
img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)
output_list = [np.stack([img_identity[i], img_change[i], run_q[i], pupil_q[i],
                         outcome_bins[i]], axis=0) for i in range(ntrials)]
```

iii. The notes justify this with the same common-bin alignment used for the other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ref.running_speed`, using the `speed` column and its timestamps.

ii.
```python
run = ref.running_speed
run_t = run['timestamps'].values.astype(np.float64)
run_v = interpolate_nans(run['speed'].values.astype(np.float64), run_t)
```

iii. The notes cite the SDK `running_speed` stream as the correct filtered locomotion signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code linearly interpolates NaNs within the running-speed trace, averages running speed within each 250 ms trial bin, then discretizes the resulting binned values.

ii.
```python
run_v = interpolate_nans(run['speed'].values.astype(np.float64), run_t)
...
run_binned = bin_means(run_v, run_t, edges)[0][0]
run_q = quantile_bin(run_binned)
```

iii. The notes justify averaging within the same bins as neural activity and then discretizing into quintiles for decoding.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five equal-percentile bins computed within each session by flattening all binned running values from that session.

ii.
```python
def quantile_bin(values, nq=NQUANTILES):
    flat = values.ravel()
    edges = np.percentile(flat, np.linspace(0, 100, nq + 1)[1:-1])
    return np.digitize(values, edges, right=False).astype(np.int64)

run_q = quantile_bin(run_binned)
```

iii. The notes explicitly justify per-session quintiles as a way to avoid session-identity leakage and to compensate for large between-session differences in running propensity.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by averaging it into the same change-centered 250 ms bins used for the neural data.

ii.
```python
edges = change_times[:, None] + (OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
...
run_binned = bin_means(run_v, run_t, edges)[0][0]
```

iii. The notes say all streams share the AllenSDK sync clock, so binning every stream on the same edges gives alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ref.eye_tracking['pupil_area']`, converted to an equivalent circular diameter. It is not taken directly from `pupil_width`.

ii.
```python
eye = ref.eye_tracking
eye_t = eye['timestamps'].values.astype(np.float64)
pupil_d = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
```

iii. The notes justify this as converting area to a diameter-like quantity and mention that the AI checked consistency with width/height.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code computes equivalent diameter from area, linearly interpolates NaNs/blink gaps over time, averages within each 250 ms trial bin, and then discretizes the binned values.

ii.
```python
frac_blink = float(np.mean(~np.isfinite(pupil_d)))
pupil_d = interpolate_nans(pupil_d, eye_t)
...
pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
pupil_q = quantile_bin(pupil_binned)
```

iii. The notes justify interpolation because blink frames are missing and a pupil-diameter output cannot otherwise be defined continuously across bins.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five within-session equal-percentile bins using the same `quantile_bin` function as running speed.

ii.
```python
pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
pupil_q = quantile_bin(pupil_binned)
```

iii. The notes give the same per-session-quintile justification as for running speed, emphasizing session-specific camera geometry and scale.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by averaging it into the same change-centered 250 ms bins used for neural activity.

ii.
```python
edges = change_times[:, None] + (OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
...
pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
```

iii. The notes justify this with the common AllenSDK sync clock and common-bin resampling.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the four boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
outcome = np.full(ntrials, -1, dtype=np.int64)
outcome[tr['hit'].astype(bool).values] = 0
outcome[tr['miss'].astype(bool).values] = 1
outcome[tr['false_alarm'].astype(bool).values] = 2
outcome[tr['correct_reject'].astype(bool).values] = 3
```

iii. The notes cite `Trial._get_trial_data` and standard go/catch outcome definitions as the source for these labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are mapped to fixed integer codes `[hit, miss, false_alarm, correct_reject] = [0,1,2,3]`, trials without one of those labels are dropped, and the code is broadcast across all 24 bins of that trial.

ii.
```python
ok = outcome >= 0
...
outcome_bins = np.repeat(outcome[:, None], NBINS, axis=1)
output_list = [np.stack([img_identity[i], img_change[i], run_q[i], pupil_q[i],
                         outcome_bins[i]], axis=0) for i in range(ntrials)]
```

iii. The notes justify the outcome as a static per-trial variable that should be repeated over the time axis to fit the decoder format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI drops sessions if experiments fail to load, if running or eye streams are unusable, or if fewer than two valid trials remain. Within retained sessions, it interpolates NaNs in running speed and pupil diameter, drops trials whose fixed window falls outside available data, and drops trials without one of the four standard outcomes.

ii.
```python
try:
    exps.append((eid, cache.get_behavior_ophys_experiment(eid)))
except Exception as e:
    print(f'  [session {session_id}] could not load experiment {eid}: {e}')

run_v = interpolate_nans(run['speed'].values.astype(np.float64), run_t)
if run_v is None:
    return None

if eye is None or len(eye) == 0:
    return None
...
valid = ((change_times + OFF_START) >= t_lo) & ((change_times + OFF_END) <= t_hi)
...
if ntrials < 2:
    return None
```

iii. The notes explicitly justify dropping sessions with empty eye-tracking tables because `pupil_diameter` is a required output, and justify interpolation over short blink gaps.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identifies loading experiments from the AllenSDK cache and processing large full-session traces into binned trial matrices as the most time-consuming steps.

ii.
```python
for eid in experiment_ids:
    exps.append((eid, cache.get_behavior_ophys_experiment(eid)))
...
binned, counts = bin_means(traces, ts, edges)
...
with ProcessPoolExecutor(max_workers=args.workers) as ex:
```

iii. In the notes, the AI says the expensive parts were loading each experiment, naïve per-trial slicing of long traces, and repeated NWB access, and that it added vectorized binning plus multiprocessing to reduce wall-clock time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI explicitly identifies the naïve per-trial slicing/binning of long traces as the loop that should be vectorized. It replaces that pattern with `bin_means`, which bins all trials at once using `searchsorted` and cumulative sums.

ii.
```python
def bin_means(values, timestamps, edges):
    idx = np.searchsorted(timestamps, edges.ravel(), side='left').reshape(ntrials, nedge)
    counts = np.diff(idx, axis=1)
    csum = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)],
                          axis=1)
    sums = csum[:, idx[:, 1:]] - csum[:, idx[:, :-1]]
```

iii. The notes state this vectorization was added specifically to avoid Python loops over trials and bins.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats some session-local setup work, notably constructing a fresh AllenSDK cache object inside every worker/session and calling the same binning routine separately for neural, running, and pupil streams. It does not re-walk trials multiple times in separate passes after session processing.

ii.
```python
def process_session(session_id, experiment_ids, neural_source='events',
                    show_processing=False, plot_path=None, zscore=False):
    cache = get_cache()
...
    binned, counts = bin_means(traces, ts, edges)
...
    run_binned = bin_means(run_v, run_t, edges)[0][0]
    pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
```

iii. The notes mention repeated NWB/cache access as an efficiency concern, but do not give a separate explicit justification for the remaining repeated per-session setup.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and carries some values that are not used by the final decoder dataset: `cell_ids` are returned but never saved, continuous `run_binned` and `pupil_binned` are kept mostly for diagnostics/metadata after being discretized, and extensive `session_info` diagnostics are stored only in metadata.

ii.
```python
neural_planes, region_names, cell_ids, zero_bins = [], [], [], 0
...
run_binned = bin_means(run_v, run_t, edges)[0][0]
pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
run_q = quantile_bin(run_binned)
pupil_q = quantile_bin(pupil_binned)
...
return dict(neural=neural_list, output=output_list, region_names=region_names,
            cell_ids=cell_ids, info=info)
```

iii. The notes justify much of this as supporting sanity checks, provenance, and optional processing plots, but they do not claim these extra continuous or identifier fields are needed by downstream decoder training itself.
