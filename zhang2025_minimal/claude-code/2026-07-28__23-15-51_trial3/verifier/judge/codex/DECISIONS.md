# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent did not use the ONE API loaders used by the reference. It read `/app/code/code_zhang2025/data/bwm_release.csv`, constructed cache paths manually, and then walked the filesystem with `Path`, `rglob`, `np.load`, and `pd.read_parquet` to find trials, wheel, motion-energy, and spike files.

ii. ```python
def find_session_paths(cache_dir):
    bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
    ...
    session_path = Path(cache_dir) / lab / 'Subjects' / subject / date / '001'
```

```python
trials_files = list(session_path.rglob('_ibl_trials.table.pqt'))
trials = pd.read_parquet(trials_files[0])
...
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
```

iii. The trajectory explicitly says the agent abandoned the ONE API because "The ONE API lists only certain datasets... Let me work directly with the file system instead." `CONVERSION_NOTES.md` presents this as loading the BWM release, but does not justify the departure from the reference loading path.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of `bwm_release.csv` and attached to each session record. In the final dataset, subjects are accumulated in first-seen order while iterating through processed sessions, and `subject_idx` is the index into that ordered list.

ii. ```python
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

iii. No separate reasoning is in the code comments. The implied justification is that `bwm_release.csv` already provides subject identity, so no derivation from paths or files was needed.

## 1-c. How are the data split into sessions?

i. Each unique `eid` from `bwm_release.csv` is treated as one session. Multiple probes for the same `eid` are gathered under the same session record and later merged.

ii. ```python
eid = row['eid']
...
if eid not in sessions:
    sessions[eid] = {...}
sessions[eid]['probes'].append(probe_name)
```

iii. The agent treated the release CSV as the session index. This mirrors the assumption in the reference that one `eid` corresponds to one session, but it was justified through the release table rather than `one.search`.

## 1-d. How are the data split into trials?

i. Trials are taken as rows of `_ibl_trials.table.pqt`. After loading the full trials table, the code filters rows with a boolean mask and then iterates row-by-row over `valid_trials`.

ii. ```python
trials = pd.read_parquet(trials_files[0])
...
valid_trials = trials[mask].copy()
...
for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
```

iii. No explicit justification beyond following the trials table structure. The code assumes, as in the reference, that each row is a trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are first filtered for non-NaN required events, reaction time between 0.08 s and 2.0 s, and `choice != 0`. After that, trials are filtered again implicitly: any trial whose interpolated wheel or whisker trace is missing or contains NaNs is skipped inside the per-trial loop.

ii. ```python
mask = pd.Series(True, index=trials.index)
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
ws = interpolate_behavior_to_bins(...)
wm = interpolate_behavior_to_bins(...)
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
```

iii. `CONVERSION_NOTES.md` says this "matches `load_trials_and_mask()`" and the trajectory summary repeats "NaN events, RT range [0.08, 2.0]s, exclude no-choice." It does not mention the later behavior-coverage skip as a separate filtering stage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrix is built from `spikes.times.npy` and `spikes.clusters.npy` for each probe. Region labels come from `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy`, but the spike matrix itself is derived from spike times and cluster assignments.

ii. ```python
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
clusters_channels = np.load(revision_path / 'clusters.channels.npy')
chan_brain_ids = np.load(revision_path / 'channels.brainLocationIds_ccf_2017.npy')
```

iii. `CONVERSION_NOTES.md` says the agent followed the reference idea of loading spike times and cluster assignments and then using Beryl region mapping. The trajectory shows the agent believed this matched `load_spiking_data`.

## 2-b. How is the `neural` data processed?

i. Spikes from all probes in a session are merged, cluster ids are remapped to a contiguous index, and spikes are binned per trial into 20 ms bins using counts. The code does not divide by bin width, so the stored neural signal is spike counts per bin rather than firing rate in Hz.

ii. ```python
spike_times, spike_clusters, all_channels, all_brain_ids, all_labels = \
    merge_probes(spikes_list, channels_list, brain_ids_list, labels_list)
...
cluster_ids = np.unique(spike_clusters)
cluster_id_to_idx = {cid: idx for idx, cid in enumerate(cluster_ids)}
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])
```

```python
bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)
bin_idx = np.clip(bin_idx, 0, n_bins - 1)
np.add.at(binned, (clusters_sel[valid], bin_idx[valid]), 1)
```

iii. The justification in the trajectory leans on the method paper's "temporally binned spike counts" language. `CONVERSION_NOTES.md` focuses on 20 ms binning and probe merging, but it does not discuss the reference code's later conversion from counts to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is effectively not filtered by neural quality control. The code loads `clusters.metrics.pqt` and cluster labels, but never applies them to remove low-quality clusters or spikes. All encountered clusters are kept.

ii. ```python
metrics_files = list(revision_path.glob('clusters.metrics.pqt'))
cluster_labels = None
if metrics_files:
    metrics = pd.read_parquet(metrics_files[0])
    cluster_labels = metrics['label'].values
```

```python
cluster_ids = np.unique(spike_clusters)
n_clusters = len(cluster_ids)
```

iii. `CONVERSION_NOTES.md` explicitly states: "Load ALL clusters (not filtered by quality metric)." The trajectory summary also says "Clusters: All loaded (no quality filtering, matching reference `qc=None`)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. For a trial with onset `stim_on`, the code sets `t_start = stim_on - 0.5` and `t_end = stim_on + 1.5`, then bins spikes falling in that window.

ii. ```python
stim_on = trial[ALIGN_TIME]
t_start = stim_on + TIME_WINDOW[0]
t_end = stim_on + TIME_WINDOW[1]
```

```python
neural = bin_spikes_trial(
    spike_times, spike_clusters_remapped, n_clusters,
    t_start, t_end, BINSIZE, N_BINS
)
```

iii. `CONVERSION_NOTES.md` says alignment is to stimulus onset because both the decoder task and reference code use `stimOn_times`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins, with 100 bins over the 2 s window from -0.5 to 1.5 s. No additional temporal rebinning is applied after binning or interpolation.

ii. ```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. `CONVERSION_NOTES.md` justifies this by citing the reference parameter `'binsize': 0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial alignment event `stimOn_times`, but the values themselves are generated synthetically from the configured time window and bin size rather than copied from a raw time-series variable.

ii. ```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
...
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. `CONVERSION_NOTES.md` says all trials are aligned to `stimOn_times`; the time input is treated as a constructed decoder input on that aligned grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code creates a length-100 vector with `np.linspace(-0.48, 1.5, 100)` and stores that same vector for every trial. It is not derived from raw timestamps after alignment; it is generated directly from the configured window and bin size.

ii. ```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
...
input_data = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_num, dtype=np.float32)
], axis=0)
```

iii. The notes justify this only at a high level: use stimulus-aligned 20 ms bins. No separate explanation is given for the exact use of `np.linspace` endpoints.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The agent intended it to share the same 100-step grid used for aligned trial data. In code, it uses the same `np.linspace(t_start + binsize, t_end, n_bins)` style grid used for behavioral interpolation, not an explicit neural-bin-center array.

ii. ```python
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
...
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says wheel speed and whisker motion energy are "interpolate[d] ... to neural bin centers," so the justification was that all aligned streams should share one common grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the trials table. Block boundaries are inferred by changes in `probabilityLeft`.

ii. ```python
prob_left = trials_masked['probabilityLeft'].values
block_changes = np.concatenate([[0], np.where(np.diff(prob_left) != 0)[0] + 1])
```

iii. The implied justification is the same as the reference logic: `probabilityLeft` is the available block indicator.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code first applies the main trial mask, then computes block changes only within the surviving trials, and assigns `0, 1, 2, ...` within each filtered block. The resulting scalar is broadcast across all 100 time bins for each kept trial.

ii. ```python
trials_masked = trials_df[mask].copy()
prob_left = trials_masked['probabilityLeft'].values
...
trial_in_block[start:end] = np.arange(end - start)
```

```python
trial_num = np.float32(trial_in_block[trial_idx])
...
np.full(N_BINS, trial_num, dtype=np.float32)
```

iii. No separate justification appears in the notes. The code comment says "Compute trial number within each block (0-indexed)."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column in the trials table.

ii. ```python
choice_val = 0 if trial['choice'] == 1 else 1
```

iii. `CONVERSION_NOTES.md` justifies this from the decoder specification: IBL `1=left`, `-1=right`, remapped to `left=0`, `right=1`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Trials with `choice == 0` are removed. For kept trials, `choice == 1` is encoded as `0` and everything else that survives the mask is encoded as `1`. That scalar is broadcast across all time bins.

ii. ```python
if 'choice' in trials.columns:
    mask &= (trials['choice'] != 0)
...
choice_val = 0 if trial['choice'] == 1 else 1
...
np.full(N_BINS, choice_val, dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` says this matches the decoder convention `left = 0, right = 1`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii. ```python
prob_left = trial['probabilityLeft']
```

iii. `CONVERSION_NOTES.md` justifies this directly from the decoder task and the reference block-probability interpretation.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then broadcasts the result across time bins. If a kept trial somehow has another value, it silently falls back to `1`.

ii. ```python
if prob_left == 0.2:
    prior_val = 0
elif prob_left == 0.5:
    prior_val = 1
elif prob_left == 0.8:
    prior_val = 2
else:
    prior_val = 1
```

iii. `CONVERSION_NOTES.md` states the 0.2/0.5/0.8 mapping came from the decoder task. No note justifies the fallback branch.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. ```python
wheel_pos_files = list(session_path.rglob('_ibl_wheel.position.npy'))
wheel_ts_files = list(session_path.rglob('_ibl_wheel.timestamps.npy'))
...
wheel_pos = np.load(wheel_pos_files[0])
wheel_ts = np.load(wheel_ts_files[0])
```

iii. `CONVERSION_NOTES.md` says this follows the reference idea of using wheel velocity and then absolute value for speed.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code computes a simple finite-difference velocity `dp / dt`, takes its absolute value, assigns midpoint timestamps to that derived velocity trace, linearly interpolates the trace to the trial grid, and later discretizes it into three quantile bins per session.

ii. ```python
dt = np.diff(wheel_ts)
dp = np.diff(wheel_pos)
vel = dp / dt
speed = np.abs(vel)
vel_ts = wheel_ts[:-1] + dt / 2
```

```python
ws = interpolate_behavior_to_bins(wheel_ts, wheel_speed, t_start, t_end, BINSIZE, N_BINS)
...
wheel_disc, wheel_edges = discretize_to_bins(wheel_speed_raw, N_WHEEL_BINS)
```

iii. `CONVERSION_NOTES.md` explicitly justifies the finite-difference version as an approximation to the reference wheel velocity, and lists as a limitation that the reference uses `SessionLoader.load_wheel()` smoothing while this code does not.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After all usable wheel traces for a session are collected, the code pools all wheel values across those trials, computes session-level quantile edges for three bins, and discretizes each time point with `np.digitize` into categories `0, 1, 2`.

ii. ```python
all_vals = np.concatenate(all_vals)
quantiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(all_vals, quantiles)
edges[0] = -np.inf
edges[-1] = np.inf
...
d = np.digitize(v, edges[1:-1])
```

iii. `CONVERSION_NOTES.md` says the bins are "3 equal-count bins using quantiles" and that the quantiles are computed per session "to maintain local context."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated separately for each trial onto a 100-point grid spanning that trial window. The grid is generated as `np.linspace(t_start + binsize, t_end, n_bins)`, which the agent intended to be the common trial-aligned time axis for outputs and inputs.

ii. ```python
def interpolate_behavior_to_bins(beh_times, beh_values, t_start, t_end, binsize, n_bins):
    bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
    ...
    f = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')
    return f(bin_centers)
```

iii. `CONVERSION_NOTES.md` says behavioral signals are "interpolate[d] ... to neural bin centers using linear interpolation."

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from either `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, or, if unavailable, from the corresponding right-camera files.

ii. ```python
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
left_times_files = list(session_path.rglob('_ibl_leftCamera.times.npy'))
...
right_me_files = list(session_path.rglob('rightCamera.ROIMotionEnergy.npy'))
right_times_files = list(session_path.rglob('_ibl_rightCamera.times.npy'))
```

iii. `CONVERSION_NOTES.md` says the agent preferred left camera and fell back to right camera because that matched the reference camera-choice logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code loads the released motion-energy vector, adjusts for the case where it is one sample shorter than the camera times, chooses left camera if available, linearly interpolates the trace to the trial grid, and later discretizes it by session-level quantiles. It does not apply extra filtering or normalization.

ii. ```python
if len(me) == len(times):
    return times, me
elif len(me) == len(times) - 1:
    return times[:-1], me
```

```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
...
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)
```

iii. `CONVERSION_NOTES.md` justifies using the released motion-energy trace and the left-then-right camera preference as matching the reference behavior loader.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded exactly like wheel speed: pool all session values, compute two quantile cut points for three equal-count bins, and apply `np.digitize` to every time point.

ii. ```python
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)
...
d = np.digitize(v, edges[1:-1])
```

iii. `CONVERSION_NOTES.md` says whisker motion energy is discretized into 3 equal-count quantile bins, computed per session.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker trace is linearly interpolated trial-by-trial onto the same generated 100-point grid used for wheel speed and the synthetic time input.

ii. ```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
```

```python
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. `CONVERSION_NOTES.md` justifies this as interpolation to the neural bin centers.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable data are mostly handled by skipping. Sessions are skipped if trials or spikes are missing, if wheel data are missing, or if fewer than two valid trials remain. Individual trials are skipped if wheel/whisker interpolation fails or returns NaNs. Two additional silent fallbacks are used: unknown brain-region lookups become region id `0`/"root", and unexpected `probabilityLeft` values are mapped to prior class `1`.

ii. ```python
if not trials_files:
    return None, None
...
if not spikes_list:
    print(f"    Skipping: no spike data")
    return None
...
if wheel_speed is None:
    print(f"    Skipping: no wheel data")
    return None
```

```python
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
...
cluster_brain_ids.append(0)
...
prior_val = 1  # fallback
```

iii. The trajectory summary emphasizes skipping 20 sessions with missing behavioral data. `CONVERSION_NOTES.md` frames missing wheel or whisker data as the main reason for skipped sessions, and notes the wheel-smoothing and discretization differences as limitations.

## 10-a. What are the most time-consuming steps of the code?

i. The likely dominant costs are repeated disk reads of large spike files, per-session filesystem scans with `rglob`, per-trial spike binning, and per-trial interpolation of wheel and whisker traces. The code runs all sessions serially, so none of this work is parallelized.

ii. ```python
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
```

```python
for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
    ...
    neural = bin_spikes_trial(...)
    ws = interpolate_behavior_to_bins(...)
    wm = interpolate_behavior_to_bins(...)
```

iii. The trajectory shows the agent explicitly optimizing `bin_spikes_trial` with `searchsorted` and `np.add.at`, which implies it recognized per-trial spike binning as expensive. The notes do not contain a separate runtime analysis.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain scalar or Python-level: the per-trial session loop, the block-count loop in `compute_trial_number_in_block`, the cluster-id remapping list comprehension, and the repeated linear searches used to build `brain_region_idx` and `subject_idx`.

ii. ```python
for i in range(len(block_changes)):
    start = block_changes[i]
    end = block_changes[i + 1] if i + 1 < len(block_changes) else len(prob_left)
    trial_in_block[start:end] = np.arange(end - start)
```

```python
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])
...
for i, (eid, session_info) in enumerate(sorted(sessions.items())):
    result = process_session(session_info, cache_dir, br)
```

iii. The trajectory contains one explicit micro-optimization, replacing a per-spike Python loop in `bin_spikes_trial` with `np.add.at`. No broader justification was given for keeping the other Python loops.

## 10-c. What processing does the code repeat multiple times?

i. It rebuilds the same time-since-stimulus vector for every kept trial, repeatedly scans the filesystem with `rglob` for each session and data stream, repeatedly performs list-membership lookups when building subject and region indices, and computes interpolation independently for wheel and whisker on every trial.

ii. ```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

```python
wheel_pos_files = list(session_path.rglob('_ibl_wheel.position.npy'))
...
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
...
idx = np.array([all_regions_flat.index(r) for r in regions])
```

iii. No explicit justification is given. The code appears to prioritize straightforward implementation over avoiding repeated work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `clusters.depths.npy` and cluster labels but never uses them, merges and returns `all_labels` without applying QC, stores `wheel_edges` and `whisker_edges` in each session result but never puts them in the final output, and keeps unused local variables such as `valid_indices` and `good_trial_indices`.

ii. ```python
clusters_depths = np.load(revision_path / 'clusters.depths.npy')
...
cluster_labels = metrics['label'].values
...
return spike_times, spike_clusters, clusters_channels, chan_brain_ids, cluster_labels
```

```python
valid_indices = valid_trials.index.tolist()
...
good_trial_indices = []
...
'wheel_edges': wheel_edges,
'whisker_edges': whisker_edges,
```

iii. No justification was documented. These look like by-products of development rather than deliberate outputs needed by downstream analysis.
