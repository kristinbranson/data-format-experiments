# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading `ophys_experiment_table.csv` from the project metadata directory, intersecting it with the NWB files actually present on disk, filtering out passive sessions, and then loading each experiment via `BehaviorOphysExperiment.from_nwb_path()` directly from local NWB files. Sessions are processed in parallel using `multiprocessing.Pool`.

ii.
```python
DATA_DIR = '/app/data/visual-behavior-ophys-1.1.0/'
NWB_FMT = DATA_DIR + 'behavior_ophys_experiments/behavior_ophys_experiment_%d.nwb'
META_DIR = DATA_DIR + 'project_metadata/'

def get_session_table(sample=False):
    et = pd.read_csv(META_DIR + 'ophys_experiment_table.csv')
    have = set(int(f.split('_')[-1].split('.')[0])
               for f in os.listdir(DATA_DIR + 'behavior_ophys_experiments'))
    et = et[et.ophys_experiment_id.isin(have)]
    et = et[~et.passive]
    ...

def process_session(args):
    ...
    for eid in eids:
        planes.append((eid, BehaviorOphysExperiment.from_nwb_path(NWB_FMT % eid)))
```

iii. The AI chose to load NWB files directly using `BehaviorOphysExperiment.from_nwb_path()` rather than through the SDK's `VisualBehaviorOphysProjectCache`, avoiding S3 access. This approach is functionally equivalent but bypasses the cache layer.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `mouse_id` field in each session's metadata. Unique mouse IDs are collected from all processed sessions and sorted.

ii.
```python
subjects = sorted(set(r['mouse'] for r in results))
# In process_session:
result = dict(..., mouse=str(md['mouse_id']), ...)
```

iii. The `mouse_id` field is the standard identifier for each animal in the Allen SDK metadata.

## 1-c. How are the data split into sessions?

i. Sessions correspond to unique `ophys_session_id` values. Multiple experiments (imaging planes) sharing the same `ophys_session_id` are grouped together and their neurons are concatenated. The experiment table is grouped by `ophys_session_id`.

ii.
```python
sessions = [(int(sid), list(map(int, g.ophys_experiment_id.values)))
            for sid, g in et.groupby('ophys_session_id')]
```

iii. Multiscope sessions contain multiple simultaneously-recorded imaging planes that share one behavior stream and clock. Merging them into one decoder session is appropriate since they form one population recording.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's built-in `trials` table. Only `go` and `catch` trials are kept. Each trial is aligned to `change_time`, and a fixed window of 8 bins (each 750ms) is extracted: 3 bins before the change through 4 bins after (plus the change bin itself). This gives fixed-length trials of 8 time bins.

ii.
```python
BIN_SIZE = 0.75
N_PRE = 3
N_POST = 4
N_BINS = N_PRE + 1 + N_POST   # = 8

trials = ref.trials
keep = (trials.go.values.astype(bool) | trials['catch'].values.astype(bool))
trials = trials[keep]

change_times = trials.change_time.values.astype(float)
j = np.searchsorted(flash_start, change_times - 1e-6, side='left')
...
bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]
bt0 = flash_start[bin_flash]
bt1 = bt0 + BIN_SIZE
```

iii. The AI justified the 750ms bin and 8-bin trial window by referencing the paper's use of "image presentation intervals" of 750ms (250ms image + 500ms grey). The fixed-length window ensures consistent trial dimensions across sessions and rigs with different frame rates.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials by: (1) keeping only go and catch trials (filtering out aborted and auto-rewarded), (2) requiring a valid (finite) `change_time`, (3) requiring the change flash to coincide exactly with a flash onset (within 1e-4s), (4) requiring the 8-bin window to fit within the stimulus block, (5) requiring the trial outcome to be one of the four valid categories, and (6) requiring at least 2 valid trials per session. Additionally, passive sessions are excluded entirely, and sessions without eye tracking data are dropped.

ii.
```python
keep = (trials.go.values.astype(bool) | trials['catch'].values.astype(bool))
trials = trials[keep]

good = np.ones(len(trials), dtype=bool)
good &= np.isfinite(change_times)
j_clipped = np.clip(j, 0, len(flash_start) - 1)
good &= np.abs(flash_start[j_clipped] - change_times) < 1e-4
good &= (j - N_PRE >= 0) & (j + N_POST < len(flash_start))

oc = np.full(len(trials), -1, dtype=np.int64)
for k, name in enumerate(OUTCOMES):
    oc[trials[name].values.astype(bool)] = k
good &= oc >= 0

# passive sessions excluded:
et = et[~et.passive]

# eye tracking required:
if len(eye) == 0:
    return None
```

iii. The filtering is more aggressive than the reference. The passive session exclusion is justified because passive sessions lack licks/rewards so trial outcomes would be degenerate. The eye tracking requirement is justified because pupil diameter is a required output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `events.events` — the detected calcium event magnitudes — from each experiment in the session.

ii.
```python
for eid, ds in planes:
    ev = np.vstack(ds.events.events.values).astype(np.float64)
    ts = ds.ophys_timestamps.astype(float)
    s, _ = binned_sum(ev, ts, flat_t0, flat_t1)
    neural_planes.append(s.reshape(ev.shape[0], n_trials, N_BINS))
```

iii. The AI justified using `events` (detected calcium events) by citing the paper: "For all analysis of neural data we used the detected calcium events." The CONVERSION_NOTES.md step 4 explicitly discusses this decision.

## 2-b. How is the `neural` data processed?

i. For each neuron, detected calcium event magnitudes are summed within each 750ms bin using a cumulative sum + searchsorted approach. Neurons from multiple imaging planes within a session are concatenated along the neuron axis. The resulting neural data has shape (n_neurons, n_trials, 8).

ii.
```python
def binned_sum(values, timestamps, t0, t1):
    csum = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)], axis=1)
    i0 = np.searchsorted(timestamps, t0, side='left')
    i1 = np.searchsorted(timestamps, t1, side='left')
    return csum[:, i1] - csum[:, i0], (i1 - i0)

neural_planes = []
for eid, ds in planes:
    ev = np.vstack(ds.events.events.values).astype(np.float64)
    ts = ds.ophys_timestamps.astype(float)
    s, _ = binned_sum(ev, ts, flat_t0, flat_t1)
    neural_planes.append(s.reshape(ev.shape[0], n_trials, N_BINS))
neural = np.concatenate(neural_planes, axis=0).astype(np.float32)
```

iii. The AI justified summing events within bins as "the natural per-bin aggregate" that keeps rigs with different frame rates comparable. The cumsum+searchsorted approach is efficient.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI relies on the SDK's default `exclude_invalid_rois=True` parameter in `BehaviorOphysExperiment.from_nwb_path()`, which filters out invalid ROIs. No additional neuron-level filtering is applied.

ii.
```python
planes.append((eid, BehaviorOphysExperiment.from_nwb_path(NWB_FMT % eid)))
```

iii. The AI verified that `valid_roi` is True for 100% of returned cells, confirming the SDK's default filtering. No electrophysiology-style quality metrics are applicable for 2-photon imaging.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to `change_time` — the time of the image change (go trials) or sham change (catch trials). The alignment is done by finding the flash onset that coincides with `change_time`, then taking N_PRE=3 bins before through N_POST=4 bins after.

ii.
```python
change_times = trials.change_time.values.astype(float)
j = np.searchsorted(flash_start, change_times - 1e-6, side='left')
...
bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]
bt0 = flash_start[bin_flash]
bt1 = bt0 + BIN_SIZE
```

iii. The AI justified aligning to `change_time` as it "coincides exactly with an image flash onset" and is "the natural per-trial alignment event." The instructions say to "temporally align based on ophys timestamp," which the AI interpreted as meaning the ophys time coordinate system rather than trial_start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 750ms, corresponding to one image presentation interval (250ms image + 500ms grey). Data is rebinned from the native ophys frame rate into these 750ms bins. Each trial has exactly 8 bins.

ii.
```python
BIN_SIZE = 0.75          # s, one image presentation interval
N_BINS = N_PRE + 1 + N_POST   # = 8

data['metadata'] = dict(
    ...
    time_bin_size=BIN_SIZE * 1000.0,
    ...
)
```

iii. The AI justified this choice by referencing the paper: "By image presentation interval we refer to the 750 ms interval beginning with each image presentation." This makes the bin size match the natural temporal structure of the task.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations.image_name` — the image shown in each flash — indexed at the flash positions corresponding to each of the 8 bins.

ii.
```python
sp = ref.stimulus_presentations
sp = sp[sp.stimulus_block_name.str.contains('change_detection', na=False)]
flash_image = sp.image_name.values.astype(object)
...
img = flash_image[bin_flash]  # (n_trials, 8) of str
```

iii. Using `stimulus_presentations` directly provides the actual image shown at each flash, including handling of omitted flashes (where `image_name` is 'omitted'). This is more precise than deriving it from the trials table.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names at each of the 8 flash positions are looked up from the stimulus_presentations table. A global vocabulary of all unique image names (16 natural images + 'omitted') is built, with 'omitted' placed last. Each image name is mapped to an integer index.

ii.
```python
images = sorted(set(v for r in results for v in np.unique(r['image'])))
images = [v for v in images if v != 'omitted'] + ['omitted']
img_map = {v: i for i, v in enumerate(images)}
img_idx = np.vectorize(img_map.get)(r['image']).astype(np.int64)
```

iii. The vocabulary includes 'omitted' as a separate category rather than filling it with the surrounding image. The AI justified this: "Give omissions their own image-identity category rather than dropping or filling them, so no timepoint is fabricated."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is inherently aligned because it is indexed by the same flash positions used to define the 750ms neural bins. Each bin corresponds to one flash, so `flash_image[bin_flash]` gives the image shown during that bin.

ii.
```python
bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]
img = flash_image[bin_flash]
```

iii. Since neural data is binned by flash intervals and image identity is looked up at the same flash positions, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change`, which is True only for flashes where a real image change occurs (go trials only, not catch trials).

ii.
```python
flash_ischange = sp.is_change.values.astype(bool)
chg = flash_ischange[bin_flash].astype(np.int64)
```

iii. The AI noted that `is_change` marks only real image changes. Catch trials are sham changes where the image doesn't actually change, so they are correctly coded as 0. This matches the instruction: "Have value of 1 right after a change in image identity."

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean from stimulus_presentations is directly indexed at the 8 flash positions per trial and cast to int64. No additional processing is needed.

ii.
```python
chg = flash_ischange[bin_flash].astype(np.int64)
```

iii. The AI validated this with assertions: change is 1 at bin N_PRE (the change bin) for go trials, 0 for catch trials, and 0 for all pre-change bins.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1) from `is_change`, with no thresholding needed. It is 1 at the change flash on go trials, 0 everywhere else.

ii.
```python
chg = flash_ischange[bin_flash].astype(np.int64)
```

iii. N/A — the variable is already binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — indexed by the same flash positions that define the neural bins, so alignment is by construction.

ii.
```python
chg = flash_ischange[bin_flash].astype(np.int64)
```

iii. Same flash-indexed approach as all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `running_speed.speed` and `running_speed.timestamps` from the SDK.

ii.
```python
run = ref.running_speed
run_t = run.timestamps.values.astype(float)
run_v = run.speed.values.astype(float)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is first NaN-interpolated, then the mean speed within each 750ms bin is computed using `binned_mean_1d()`. The continuous binned values are then discretized into 5 equal-percentile bins computed **within each session**.

ii.
```python
run_v = interpolate_nans(run_v)
run_binned = binned_mean_1d(run_v, run_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
run_cls, run_edges = quantile_bin(run_binned.ravel())
```

iii. The AI justified within-session quantile binning: "running propensity varies enormously between animals, so a within-session percentile makes the class labels mean the same thing (relative level) in every session." The `interpolate_nans` function fills NaN values via linear interpolation.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins using quantile edges computed within each session. The `quantile_bin` function computes edges from `np.quantile` at linspace(0,1,6)[1:-1] and uses `np.searchsorted` to assign class labels 0-4.

ii.
```python
def quantile_bin(x, n=N_QUANTILES):
    edges = np.quantile(x, np.linspace(0, 1, n + 1)[1:-1])
    return np.searchsorted(edges, x, side='right').astype(np.int64), edges
```

iii. Within-session computation ensures each class holds ~20% of samples within each session. The AI verified this: "0.199-0.202" per class.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is binned by computing the mean within each 750ms flash window. The same bin start/end times (`bt0`, `bt1`) are used for neural and behavioral data, so alignment is by construction.

ii.
```python
run_binned = binned_mean_1d(run_v, run_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
```

iii. The `binned_mean_1d` function computes the mean of running speed samples within each [t0, t1) window. Empty windows fall back to linear interpolation at the window center.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `eye_tracking.pupil_width` and `eye_tracking.timestamps`. Blinks are handled via NaN interpolation.

ii.
```python
eye = ref.eye_tracking
pupil = interpolate_nans(eye.pupil_width.values)
eye_t = eye.timestamps.values.astype(float)
```

iii. The AI chose `pupil_width` as the pupil diameter measure, noting it "correlates 0.986 with area."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaN values (from blinks) are linearly interpolated in the raw pupil_width signal. Then the mean pupil width within each 750ms bin is computed using `binned_mean_1d()`. The continuous binned values are discretized into 5 equal-percentile bins computed within each session.

ii.
```python
pupil = interpolate_nans(eye.pupil_width.values)
if pupil is None:
    return None
...
pup_binned = binned_mean_1d(pupil, eye_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
pup_cls, pup_edges = quantile_bin(pup_binned.ravel())
```

iii. The AI interpolates blinks (NaN values) before binning rather than excluding them. Sessions where all pupil values are NaN are dropped entirely.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal-percentile bins computed within each session via `quantile_bin()`.

ii.
```python
pup_cls, pup_edges = quantile_bin(pup_binned.ravel())
```

iii. Same justification as running speed — within-session quantile normalization accounts for camera alignment differences between sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed — mean within each 750ms bin window, aligned by construction with the neural bins.

ii.
```python
pup_binned = binned_mean_1d(pupil, eye_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
```

iii. Same flash-interval alignment as all other signals.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
oc = np.full(len(trials), -1, dtype=np.int64)
for k, name in enumerate(OUTCOMES):
    oc[trials[name].values.astype(bool)] = k
good &= oc >= 0
```

iii. These four outcome categories are mutually exclusive for go and catch trials. Trials that don't match any category (oc == -1) are excluded.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are mapped to integer codes 0-3 (hit=0, miss=1, false_alarm=2, correct_reject=3). The outcome code is static per trial but broadcast across all 8 bins.

ii.
```python
outcome = oc[idx]
...
np.full(N_BINS, r['outcome'][t], dtype=np.int64)
```

iii. The per-trial outcome is constant but replicated across time bins to maintain consistent output dimensions, as the instruction says to make outputs time-varying "if at all possible."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing eye tracking**: 3 sessions with empty eye tracking tables are dropped entirely (`return None`).
- **All-NaN pupil**: Sessions where `interpolate_nans` returns None are dropped.
- **NaN in pupil/running**: Linearly interpolated before binning.
- **Empty bins in binned_mean_1d**: Fall back to linear interpolation at the bin center.
- **Trials not matching outcome categories**: Filtered out (`good &= oc >= 0`).
- **change_time not matching a flash onset**: Filtered out.
- **Window outside stimulus block**: Filtered out.
- **Sessions with <2 trials**: Dropped.

ii.
```python
pupil = interpolate_nans(eye.pupil_width.values)
if pupil is None:
    return None

def binned_mean_1d(values, timestamps, t0, t1):
    ...
    if np.any(~ok):
        centres = 0.5 * (t0[~ok] + t1[~ok])
        out[~ok] = np.interp(centres, timestamps, values)
    return out
```

iii. The AI's approach is to drop sessions/trials where essential data is missing rather than fabricating values. NaN interpolation for pupil blinks is justified as these are transient measurement artifacts, not missing data.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each experiment from NWB files via `BehaviorOphysExperiment.from_nwb_path()` is the main bottleneck (~2.5-3s per experiment). The AI uses `multiprocessing.Pool` with up to 24 workers to parallelize across sessions.

ii.
```python
if args.sample or args.workers <= 1:
    results = [process_session(j) for j in jobs]
else:
    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(process_session, jobs)
```

iii. The full conversion took 51 seconds with 24 workers. NWB I/O dominates; the binning and output computation are fast.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for assembling outputs (converting image strings to indices, stacking output arrays) could potentially be vectorized. However, the main computational work (binned_sum, binned_mean_1d) is already vectorized using cumulative sums and searchsorted.

ii. N/A — the binning is already vectorized.

iii. The trial assembly loop is not a bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is loaded and processed once. The shared vocabularies (image names, regions, subjects) are built in a single pass over results.

ii. N/A

iii. The code is structured to avoid redundant computation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores `run_cont` (continuous binned running speed) and `pupil_cont` (continuous binned pupil diameter) alongside the discretized versions. These continuous values are used for plotting but not included in the final pickle output.

ii.
```python
result = dict(
    ...
    run_cont=run_binned, pupil_cont=pup_binned,
    run_edges=run_edges, pupil_edges=pup_edges,
    ...
)
```

iii. These intermediate values serve the `--show-processing` plotting mode but are not written to the final output file.
