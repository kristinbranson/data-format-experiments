# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent did not use the ONE API in the final converter. It read `/app/code/code_zhang2025/data/bwm_release.csv`, used each row's `lab`, `subject`, `date`, `eid`, and `probe_name` to build filesystem paths under the cache, grouped rows by `eid`, and then loaded session contents by direct `rglob`, `pd.read_parquet`, and `np.load` calls.

ii.
```python
def find_session_paths(cache_dir):
    bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
    ...
    session_path = Path(cache_dir) / lab / 'Subjects' / subject / date / '001'
    ...
    sessions[eid]['probes'].append(probe_name)
```

```python
trials_files = list(session_path.rglob('_ibl_trials.table.pqt'))
trials = pd.read_parquet(trials_files[0])
...
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
```

iii. In trajectory step 55, the agent said the ONE API "lists only certain datasets" and decided to "work directly with the file system instead." That is the explicit justification for bypassing the API and loading files by path.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. Each `eid` entry stores one subject string, and during final assembly the code builds `subject_set` in first-seen order while iterating sorted session ids; `subject_idx` stores the corresponding index for each kept session.

ii.
```python
sessions[eid] = {
    'path': session_path,
    'probes': [],
    'subject': subject,
    'lab': lab,
    'date': date,
    'eid': eid,
}
```

```python
subject_set = []
...
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subj_idx = subject_set.index(subject)
```

iii. The trajectory does not contain a separate argument for subject handling beyond the decision to use the release CSV and filesystem layout. In step 54 the agent explicitly inspected `subject` in the release CSV and then reused that metadata in the final script.

## 1-c. How are the data split into sessions?

i. The agent treated each unique `eid` in `bwm_release.csv` as one session. Multiple rows with the same `eid` are merged by adding probe names to that session record.

ii.
```python
for _, row in bwm_df.iterrows():
    eid = row['eid']
    ...
    if eid not in sessions:
        sessions[eid] = {...}
    sessions[eid]['probes'].append(probe_name)
```

```python
for i, (eid, session_info) in enumerate(sorted(sessions.items())):
    result = process_session(session_info, cache_dir, br)
```

iii. In step 54 the agent used the BWM release CSV to get an `eid` and its probes for a test session. After step 55 it kept that representation instead of switching back to ONE-managed session discovery.

## 1-d. How are the data split into trials?

i. Trials come from rows of `_ibl_trials.table.pqt`. After building a boolean mask, the code subsets the table to `valid_trials = trials[mask].copy()` and then iterates row-by-row; each surviving row becomes one trial in the converted dataset.

ii.
```python
trials = pd.read_parquet(trials_files[0])
...
valid_trials = trials[mask].copy()
...
for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
    stim_on = trial[ALIGN_TIME]
```

iii. In step 57 the agent inspected a trials parquet file and saw that it already contained one row per trial. The final code follows that structure directly.

## 1-e. How are trials filtered based on quality controls?

i. The code first masks out trials with NaNs in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`, then removes trials with reaction time outside `[0.08, 2.0]` and `choice == 0`. After that, trials are dropped if wheel or whisker interpolation fails or yields NaNs, and entire sessions are dropped if fewer than two trials remain. The code does not explicitly require `probabilityLeft` to be one of `{0.2, 0.5, 0.8}`.

ii.
```python
NAN_EXCLUDE = [
    'stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType'
]
...
for event in NAN_EXCLUDE:
    if event in trials.columns:
        mask &= ~trials[event].isna()
...
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
...
mask &= (trials['choice'] != 0)
```

```python
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. In steps 67 and 107 the agent said it was matching the reference `load_trials_and_mask()` logic. The code reflects that intent, but it also adds extra NaN filters and trial dropping via failed behavior interpolation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices are derived from spike times and spike cluster assignments loaded from `spikes.times.npy` and `spikes.clusters.npy`. The code also loads `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` to assign brain regions, and `clusters.metrics.pqt` to read labels, but those labels are not used to filter the neural data.

ii.
```python
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
clusters_channels = np.load(revision_path / 'clusters.channels.npy')
chan_brain_ids = np.load(revision_path / 'channels.brainLocationIds_ccf_2017.npy')
...
cluster_labels = metrics['label'].values
```

iii. In steps 57 and 58 the agent manually inspected the spike arrays, cluster metrics, and channel brain IDs for one session, then used those same files in the converter.

## 2-b. How is the `neural` data processed?

i. Probe spike streams are merged within a session by offsetting cluster ids and concatenating/sorting spikes by time. The merged cluster ids are remapped to dense indices, and for each trial spikes are counted into 20 ms bins over the `[stimOn-0.5, stimOn+1.5]` window. The saved neural arrays are spike counts in `float32`; the code does not divide by bin width to convert them to firing rates.

ii.
```python
merged_clusters.append(clusters + cluster_offset)
...
sort_idx = np.argsort(all_times, kind='stable')
all_times = all_times[sort_idx]
all_clusters = all_clusters[sort_idx]
```

```python
bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)
bin_idx = np.clip(bin_idx, 0, n_bins - 1)
valid = clusters_sel < n_clusters
np.add.at(binned, (clusters_sel[valid], bin_idx[valid]), 1)
```

iii. In step 67 the agent explicitly chose stimulus-onset alignment, a `(-0.5, 1.5)` window, 20 ms bins, and probe merging. In step 85 it said `bin_spikes_trial` was slow, and step 86 changed the implementation to `searchsorted` plus `np.add.at`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The final converter does not apply neural quality-control filtering. It keeps all clusters that appear in the spike arrays, regardless of `cluster_labels`, and does not drop `void` regions. Only missing probe data is skipped.

ii.
```python
metrics_files = list(revision_path.glob('clusters.metrics.pqt'))
cluster_labels = None
if metrics_files:
    metrics = pd.read_parquet(metrics_files[0])
    cluster_labels = metrics['label'].values
...
for probe_name in session_info['probes']:
    spike_times, spike_clusters, clusters_channels, chan_brain_ids, cluster_labels = \
        load_spikes(session_path, probe_name)
    if spike_times is None:
        continue
```

iii. In step 83 the agent explicitly reasoned that the reference code used `qc=None`, so it should "keep all clusters" and not apply quality filtering in the converter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to stimulus onset. For each row in the filtered trials table, the code defines a trial window from `stimOn_times - 0.5` to `stimOn_times + 1.5` and bins spikes inside that window.

ii.
```python
stim_on = trial[ALIGN_TIME]
t_start = stim_on + TIME_WINDOW[0]
t_end = stim_on + TIME_WINDOW[1]
...
neural = bin_spikes_trial(
    spike_times, spike_clusters_remapped, n_clusters,
    t_start, t_end, BINSIZE, N_BINS
)
```

iii. In step 67 the agent highlighted `align_time: 'stimOn_times'` and the `(-0.5, 1.5)` stimulus-centered window from the reference code, and used those constants directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data use 20 ms bins, giving 100 bins per 2 s trial window. No additional temporal rebinning or smoothing is applied after that.

ii.
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. In step 67 the agent said the reference caching code used `binsize: 0.02` and therefore chose 20 ms bins for all targets.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time input is not loaded from its own raw array. It is synthesized as a fixed time grid relative to the alignment event `stimOn_times`, using `TIME_WINDOW` and `BINSIZE`.

ii.
```python
stim_on = trial[ALIGN_TIME]
...
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. In step 67 the agent said the decoder input should be continuous time relative to stimulus onset, with the same stimulus-aligned window as the reference code. The trajectory does not show a separate raw-data source beyond `stimOn_times`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code creates the time input by calling `np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)` for every trial and casts it to `float32`. It does not compute bin centers the same way as the reference.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
...
input_data = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_num, dtype=np.float32)
], axis=0)
```

iii. The trajectory only states the intent: in step 67 the agent chose a 20 ms stimulus-aligned grid. It did not separately justify the `linspace` endpoint convention used in the final code.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The agent intended the time input to share the same per-trial binning grid as neural and behavioral signals, but the implementation uses `np.linspace(..., t_end, N_BINS)`, which yields values from `-0.48` to `1.5` relative to stimulus onset instead of the neural bin centers `-0.49` to `1.49`.

ii.
```python
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
...
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. In steps 67 and 107 the agent described behavior signals as interpolated to the neural bin centers and the time input as using that same grid. The trajectory does not recognize that the implemented grid is shifted by half a bin and includes the trial endpoint.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft`. The code infers block boundaries by finding changes in that column.

ii.
```python
trials_masked = trials_df[mask].copy()
prob_left = trials_masked['probabilityLeft'].values
block_changes = np.concatenate([[0], np.where(np.diff(prob_left) != 0)[0] + 1])
```

iii. In step 107 the agent described the task as having blocks with alternating prior probabilities, and the implementation uses `probabilityLeft` as the block label.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code first filters the trials table with `mask`, then counts trials from zero within each contiguous run of identical `probabilityLeft` among the surviving trials only. That means filtered-out trials do not advance the block counter.

ii.
```python
trials_masked = trials_df[mask].copy()
...
for i in range(len(block_changes)):
    start = block_changes[i]
    end = block_changes[i + 1] if i + 1 < len(block_changes) else len(prob_left)
    trial_in_block[start:end] = np.arange(end - start)
```

iii. The trajectory does not contain a separate argument for computing trial-within-block beyond using block priors as context. The implementation choice is implicit in `compute_trial_number_in_block`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table.

ii.
```python
if 'choice' in trials.columns:
    mask &= (trials['choice'] != 0)
...
choice_val = 0 if trial['choice'] == 1 else 1
```

iii. In steps 67 and 107 the agent stated that IBL uses `1=left` and `-1=right`, and the decoder task required `left=0, right=1`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. No-response trials (`choice == 0`) are removed. Surviving values are recoded as `1 -> 0` for left and `-1 -> 1` for right, then broadcast across all time bins of the trial.

ii.
```python
if 'choice' in trials.columns:
    mask &= (trials['choice'] != 0)
...
choice_val = 0 if trial['choice'] == 1 else 1
...
np.full(N_BINS, choice_val, dtype=np.int64)
```

iii. In step 107 the agent justified the left/right recoding directly from the task instructions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior is derived from the `probabilityLeft` column of the trials table.

ii.
```python
prob_left = trial['probabilityLeft']
if prob_left == 0.2:
    prior_val = 0
elif prob_left == 0.5:
    prior_val = 1
elif prob_left == 0.8:
    prior_val = 2
```

iii. In step 107 the agent said the decoder task required the mapping `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps `0.2`, `0.5`, and `0.8` to `0`, `1`, and `2`, broadcasts the result across time bins, and silently falls back to `1` for any unexpected value.

ii.
```python
if prob_left == 0.2:
    prior_val = 0
elif prob_left == 0.5:
    prior_val = 1
elif prob_left == 0.8:
    prior_val = 2
else:
    prior_val = 1  # fallback
```

iii. The explicit justification in step 107 was the task-specified three-value mapping. The fallback behavior is not justified in the trajectory.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
wheel_pos_files = list(session_path.rglob('_ibl_wheel.position.npy'))
wheel_ts_files = list(session_path.rglob('_ibl_wheel.timestamps.npy'))
...
wheel_pos = np.load(wheel_pos_files[0])
wheel_ts = np.load(wheel_ts_files[0])
```

iii. In step 62 the agent examined the wheel position and timestamp arrays and decided to compute speed from them.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code takes finite differences of wheel position over timestamps, converts that to absolute velocity, assigns midpoint timestamps to the velocity samples, and linearly interpolates the speed trace onto the trial grid.

ii.
```python
dt = np.diff(wheel_ts)
dp = np.diff(wheel_pos)
vel = dp / dt
speed = np.abs(vel)
vel_ts = wheel_ts[:-1] + dt / 2
```

```python
ws = interpolate_behavior_to_bins(wheel_ts, wheel_speed, t_start, t_end, BINSIZE, N_BINS)
```

iii. In step 62 the agent explicitly noted that the reference `SessionLoader.load_wheel()` applies interpolation and smoothing, but concluded that "a simple diff/dt should work." Step 107 repeats that as a known limitation.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After collecting interpolated wheel-speed traces from all kept trials in a session, the code concatenates all non-NaN values, computes quantile edges for three equal-count bins, sets the outer edges to `-inf` and `inf`, and uses `np.digitize` to assign categories `0`, `1`, or `2`.

ii.
```python
quantiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(all_vals, quantiles)
edges[0] = -np.inf
edges[-1] = np.inf
...
d = np.digitize(v, edges[1:-1])
```

```python
wheel_disc, wheel_edges = discretize_to_bins(wheel_speed_raw, N_WHEEL_BINS)
```

iii. In step 107 the agent justified three quantile bins as a way to satisfy the decoder requirement while keeping class frequencies approximately balanced within each session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. For each trial, wheel speed is interpolated onto the same grid that the code uses for behavioral traces and the time input: `np.linspace(t_start + binsize, t_end, n_bins)`. The agent intended this to match the neural bins, but the implemented grid is shifted relative to the true neural bin centers.

ii.
```python
def interpolate_behavior_to_bins(beh_times, beh_values, t_start, t_end, binsize, n_bins):
    bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
    ...
    f = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')
    return f(bin_centers)
```

iii. In step 107 the agent said wheel speed was interpolated to neural bin centers. The code shows that intended alignment rule, even though the actual centers are offset.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy` when available, otherwise `rightCamera.ROIMotionEnergy.npy` plus `_ibl_rightCamera.times.npy`.

ii.
```python
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
left_times_files = list(session_path.rglob('_ibl_leftCamera.times.npy'))
...
right_me_files = list(session_path.rglob('rightCamera.ROIMotionEnergy.npy'))
right_times_files = list(session_path.rglob('_ibl_rightCamera.times.npy'))
```

iii. In steps 59 and 60 the agent inspected both left and right camera files. In step 107 it justified preferring the left camera and falling back to the right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code uses the released motion-energy trace directly, adjusts timestamps if the signal has one fewer sample than the camera times, and linearly interpolates the trace onto the trial grid. It does not apply additional filtering or normalization.

ii.
```python
if len(me) == len(times):
    return times, me
elif len(me) == len(times) - 1:
    return times[:-1], me
```

```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
```

iii. In step 60 the agent concluded that the motion-energy trace was already aligned to camera timestamps. In step 107 it justified linear interpolation from the reference utilities.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The code uses the same three-bin quantile discretization as wheel speed, computed per session over all concatenated kept-trial values.

ii.
```python
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)
...
d = np.digitize(v, edges[1:-1])
```

iii. In step 107 the agent justified quantile discretization for whisker motion energy in the same way as for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Like wheel speed, whisker motion energy is interpolated per trial onto `np.linspace(t_start + binsize, t_end, n_bins)`. The intended rule is shared alignment with the neural bins, but the implemented grid is shifted relative to the true bin centers.

ii.
```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
...
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. The trajectory describes this as interpolation to the neural bin centers, but the code does not exactly implement the reference center locations.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing session components usually cause the code to skip data rather than repair it. Missing trials parquet returns `None`; missing probe revisions or load errors skip that probe; missing wheel data skips the session; wheel or whisker interpolation failures or NaNs skip individual trials; sessions with fewer than two surviving trials are dropped. Unexpected prior values are silently mapped to the middle category.

ii.
```python
if not trials_files:
    return None, None
...
if not revisions:
    return None, None, None, None, None
...
except Exception as e:
    print(f"  Error loading spikes for {probe_name}: {e}")
    return None, None, None, None, None
```

```python
if wheel_speed is None:
    return None
...
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
...
else:
    prior_val = 1  # fallback
```

iii. The trajectory does not contain a comprehensive missing-data policy. The clearest explicit justification is step 55, where the agent chose direct file loading because ONE was not exposing all expected datasets; the rest of the handling is implicit in the final code.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive parts of this implementation are repeated filesystem scans and file loads for each session/probe, plus the per-trial spike binning and behavioral interpolation loops. The agent explicitly identified `bin_spikes_trial` as a performance problem during debugging.

ii.
```python
trials_files = list(session_path.rglob('_ibl_trials.table.pqt'))
wheel_pos_files = list(session_path.rglob('_ibl_wheel.position.npy'))
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
```

```python
for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
    neural = bin_spikes_trial(...)
    ws = interpolate_behavior_to_bins(...)
    wm = interpolate_behavior_to_bins(...)
```

iii. In step 85 the agent said the conversion was slow because of the per-spike loop in `bin_spikes_trial`, then optimized that function in step 86.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain vectorizable: the block-counting loop in `compute_trial_number_in_block`, the loop assigning cluster brain ids, the outer per-trial session loop, and the loops that build global brain-region indices by repeated Python membership and `.index` lookups. The agent only vectorized the inner spike accumulation step.

ii.
```python
for i in range(len(block_changes)):
    start = block_changes[i]
    end = block_changes[i + 1] if i + 1 < len(block_changes) else len(prob_left)
    trial_in_block[start:end] = np.arange(end - start)
```

```python
for cid in cluster_ids:
    if cid < len(all_channels):
        ch = int(all_channels[cid])
        ...
```

iii. Step 85 identifies the spike-binning loop as the part worth optimizing, and step 86 replaces its inner Python loop with `np.add.at`. The trajectory does not show further vectorization of the remaining loops.

## 10-c. What processing does the code repeat multiple times?

i. The code recomputes the same time grid for every trial, rescans directories with `rglob` for every session, runs nearly identical interpolation logic separately for wheel and whisker traces on every trial, and repeatedly uses linear-time membership/index lookups while assembling subjects and brain-region indices.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

```python
wheel_pos_files = list(session_path.rglob('_ibl_wheel.position.npy'))
...
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
...
idx = np.array([all_regions_flat.index(r) for r in regions])
```

iii. The trajectory does not call out these repetitions explicitly. The only explicit performance discussion is the spike-binning optimization in steps 85-86.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `clusters.depths.npy`, `clusters.metrics.pqt`, and concatenates `all_labels`, but never uses them for filtering or outputs. It also creates `valid_indices`, `good_trial_indices`, `all_subjects`, and `all_brain_region_idx` without using them downstream, and returns `wheel_edges` and `whisker_edges` from `process_session` even though the final dataset drops them.

ii.
```python
clusters_depths = np.load(revision_path / 'clusters.depths.npy')
...
cluster_labels = metrics['label'].values
...
all_labels = np.concatenate(merged_labels) if merged_labels else None
```

```python
valid_indices = valid_trials.index.tolist()
...
good_trial_indices = []
...
all_subjects = []
all_brain_region_idx = []
...
'wheel_edges': wheel_edges,
'whisker_edges': whisker_edges,
```

iii. There is no explicit trajectory justification for these extra computations; they appear to be leftovers from development and debugging rather than deliberate downstream requirements.
