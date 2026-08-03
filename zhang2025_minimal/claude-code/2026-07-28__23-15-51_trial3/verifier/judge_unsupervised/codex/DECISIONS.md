# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads a freeze file (`bwm_release.csv`) to enumerate sessions and probes, reconstructs each local ONE-cache session path from `lab/subject/date/session`, groups probes by `eid`, then processes each grouped session by loading a trials parquet, spike `.npy` files, wheel files, and whisker motion-energy files from disk.

ii. 
```python
def find_session_paths(cache_dir):
    bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
    sessions = {}
    for _, row in bwm_df.iterrows():
        eid = row['eid']
        ...
        session_path = Path(cache_dir) / lab / 'Subjects' / subject / date / '001'
        ...
        if session_path.exists():
            if eid not in sessions:
                sessions[eid] = {'path': session_path, 'probes': [], 'subject': subject, ...}
            sessions[eid]['probes'].append(probe_name)

def convert_data(cache_dir, max_sessions=None, sample=False):
    sessions = find_session_paths(cache_dir)
    for i, (eid, session_info) in enumerate(sorted(sessions.items())):
        result = process_session(session_info, cache_dir, br)
```

iii. In `CONVERSION_NOTES.md`, the agent says it is following `0_data_caching.py` and the BWM release freeze file. In trajectory step 16, it noted that the reference script reads `data/bwm_release.csv`, then calls `prepare_data`, `bin_spiking_data`, `bin_behaviors`, and `align_spike_behavior`.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the `subject` column of `bwm_release.csv`. The final dataset stores an ordered unique `subjects` list and a `subject_idx` entry per processed session.

ii.
```python
sessions[eid] = {
    'path': session_path,
    'probes': [],
    'subject': subject,
    ...
}

subject_set = []
...
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subj_idx = subject_set.index(subject)
...
'subjects': subject_set,
'subject_idx': np.array(all_subject_idx),
```

iii. The trajectory shows the agent read the reference caching script, which groups BWM metadata by subject when choosing sample sessions. Its notes also report subject counts from the BWM release as a sanity check.

## 1-c. How are the data split into sessions?

i. Sessions are keyed by `eid`. All probe insertions with the same `eid` are grouped into a single session, then processed once and later stored as one session entry in `neural`, `input`, `output`, and `subject_idx`.

ii.
```python
for _, row in bwm_df.iterrows():
    eid = row['eid']
    ...
    if eid not in sessions:
        sessions[eid] = {...}
    sessions[eid]['probes'].append(probe_name)

for i, (eid, session_info) in enumerate(sorted(sessions.items())):
    result = process_session(session_info, cache_dir, br)
```

iii. In `CONVERSION_NOTES.md`, the agent states that probes are merged within session because the reference code merges probes for a session `eid`. In trajectory step 17, `prepare_data` is shown calling `one.eid2pid(eid)` and then `merge_probes(...)`.

## 1-d. How are the data split into trials?

i. Trials are read from `_ibl_trials.table.pqt`. After masking, each remaining row is treated as one trial. For every valid row, the code makes one neural matrix, one input array, and one output array using a fixed trial window around `stimOn_times`.

ii.
```python
trials = pd.read_parquet(trials_files[0])
...
valid_trials = trials[mask].copy()
...
for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
    stim_on = trial[ALIGN_TIME]
    t_start = stim_on + TIME_WINDOW[0]
    t_end = stim_on + TIME_WINDOW[1]
    neural = bin_spikes_trial(...)
```

iii. The agent cites the reference `load_trials_and_mask()` function and the `bin_spiking_data(..., trials_df=...)` path in `0_data_caching.py`. Its notes frame the trial unit as a 2 s window aligned to stimulus onset.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies a boolean mask that removes trials with NaNs in six event columns, reaction times outside `[0.08, 2.0]` s, and `choice == 0`. After that, it applies additional trial filtering by dropping any trial whose wheel or whisker signal cannot be interpolated over the whole window or contains NaNs.

ii.
```python
for event in NAN_EXCLUDE:
    if event in trials.columns:
        mask &= ~trials[event].isna()
...
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
...
mask &= (trials['choice'] != 0)
...
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
```

iii. `CONVERSION_NOTES.md` says this matches `load_trials_and_mask()` defaults. Trajectory step 17 shows the reference mask defaults (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, plus RT filtering), and step 6 quotes the data paper’s trial exclusions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from per-probe spike times and spike cluster ids, with cluster-to-channel and channel-to-brain-id mappings used to attach brain regions. The code also reads cluster labels but does not use them to build `neural`.

ii.
```python
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
clusters_channels = np.load(revision_path / 'clusters.channels.npy')
chan_brain_ids = np.load(revision_path / 'channels.brainLocationIds_ccf_2017.npy')
...
metrics = pd.read_parquet(metrics_files[0])
cluster_labels = metrics['label'].values
```

iii. The notes say the neural stream follows the reference spiking-data loader and Beryl mapping. In trajectory step 17, the agent read `load_spiking_data()`, which returns spikes and merged cluster metadata from `SpikeSortingLoader`.

## 2-b. How is the `neural` data processed?

i. The code merges probes within a session, reindexes merged cluster ids, maps each cluster to a Beryl brain region, then bins spikes into per-trial count matrices of shape `(n_clusters, 100)` using 20 ms bins over a 2 s window.

ii.
```python
spike_times, spike_clusters, all_channels, all_brain_ids, all_labels = \
    merge_probes(spikes_list, channels_list, brain_ids_list, labels_list)
...
cluster_id_to_idx = {cid: idx for idx, cid in enumerate(cluster_ids)}
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])
...
beryl_regions = br.acronym2acronym(cluster_acronyms, mapping='Beryl')
...
neural = bin_spikes_trial(
    spike_times, spike_clusters_remapped, n_clusters,
    t_start, t_end, BINSIZE, N_BINS
)
```

iii. `CONVERSION_NOTES.md` explicitly says: merge probes, use Beryl mapping, and bin 20 ms spike counts. Trajectory steps 16 and 17 show the reference functions `merge_probes`, `list_brain_regions`, and `bin_spiking_data`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality-control filter is applied to the actual neural matrices. The code loads `cluster_labels` but never filters on them, and it also never enforces its top-of-file comment about “Minimum 5 good neurons per session.” The only hard exclusions are session-level skips when trials or spikes are missing or fewer than two trials survive.

ii.
```python
# Load cluster metrics for quality labels
metrics = pd.read_parquet(metrics_files[0])
cluster_labels = metrics['label'].values
...
return spike_times, spike_clusters, clusters_channels, chan_brain_ids, cluster_labels
...
if n_valid < 2:
    return None
...
if not spikes_list:
    return None
```

iii. The notes justify this by saying the reference caching code calls `load_spiking_data(..., qc=None)` and therefore “loads ALL clusters.” Trajectory step 17 shows `load_spiking_data` returning all clusters when `qc is None`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial is aligned to `stimOn_times`. The per-trial neural window runs from `-0.5` s to `+1.5` s relative to stimulus onset, and spikes are binned inside that interval.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_on = trial[ALIGN_TIME]
t_start = stim_on + TIME_WINDOW[0]
t_end = stim_on + TIME_WINDOW[1]
neural = bin_spikes_trial(..., t_start, t_end, BINSIZE, N_BINS)
```

iii. The notes cite both the task instruction “Temporally align based on stimulus onset” and the reference caching parameters `align_time='stimOn_times'`, `time_window=(-.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted dataset uses fixed 20 ms bins, yielding 100 bins per 2 s trial. No additional temporal rebinning is applied after spike counting or behavior interpolation.

ii.
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
...
bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)
...
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The notes justify 20 ms bins from `0_data_caching.py`. Trajectory step 67 also notes that the methods text mentions 50 ms bins for choice/prior, but the reference caching code uses 20 ms uniformly.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This input is derived from the alignment event `stimOn_times` together with the fixed global window and bin size; the stored values are relative times, not raw timestamps.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
...
stim_on = trial[ALIGN_TIME]
...
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The notes say the input is “time since stimulus onset” because the decoder task asked for that exact variable after stimulus-onset alignment.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code creates the same length-100 floating-point vector for every kept trial using `np.linspace(-0.48, 1.5, 100)`. It uses the right edge of each 20 ms bin rather than an absolute timestamp series.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. In its notes, the agent explains this as a direct consequence of using stimulus-onset alignment and 20 ms bins. In trajectory step 17, it also noted that the reference behavior interpolation uses `linspace(interval_beg + binsize, interval_end, n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned by construction: the vector has the same number of bins as the neural matrix for the same `stimOn_times` trial window, and it is stacked into the per-trial `input` array alongside the block-trial number.

ii.
```python
neural = bin_spikes_trial(..., t_start, t_end, BINSIZE, N_BINS)
...
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
input_data = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_num, dtype=np.float32)
], axis=0)
```

iii. The notes repeatedly justify all streams as sharing the same 20 ms stimulus-aligned bins. The agent cites the reference interpolation helper for this alignment choice.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived entirely from the per-trial `probabilityLeft` column after trial masking.

ii.
```python
def compute_trial_number_in_block(trials_df, mask):
    trials_masked = trials_df[mask].copy()
    prob_left = trials_masked['probabilityLeft'].values
```

iii. The notes explain the block variable using the task’s prior-probability structure from the papers: an initial 0.5 block followed by alternating 0.2 and 0.8 blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code finds changes in `probabilityLeft` across masked trials, treats each contiguous constant run as a block, numbers trials within each run from 0, then broadcasts the resulting scalar across all time bins for that trial.

ii.
```python
block_changes = np.concatenate([[0], np.where(np.diff(prob_left) != 0)[0] + 1])
trial_in_block = np.zeros(len(prob_left), dtype=int)
for i in range(len(block_changes)):
    start = block_changes[i]
    end = block_changes[i + 1] if i + 1 < len(block_changes) else len(prob_left)
    trial_in_block[start:end] = np.arange(end - start)
...
trial_num = np.float32(trial_in_block[trial_idx])
np.full(N_BINS, trial_num, dtype=np.float32)
```

iii. The justification is implicit rather than deeply documented. The notes use the papers’ description of alternating probability blocks and the initial 90-trial unbiased block as the rationale for deriving a block-position covariate.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the raw `choice` column in the trials table.

ii.
```python
choice_val = 0 if trial['choice'] == 1 else 1
```

iii. `CONVERSION_NOTES.md` says the agent is using the IBL choice convention and remapping it to the decoder convention requested by the task.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code first filters out `choice == 0` trials, then maps IBL’s `1 = left` and `-1 = right` onto `left = 0`, `right = 1`, and finally broadcasts the categorical value across all 100 time bins.

ii.
```python
mask &= (trials['choice'] != 0)
...
choice_val = 0 if trial['choice'] == 1 else 1
...
np.full(N_BINS, choice_val, dtype=np.int64)
```

iii. The notes explicitly cite the decoder-task requirement “left = 0, right = 1” as the reason for the remapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The prior output is derived from the `probabilityLeft` column in the trials table.

ii.
```python
prob_left = trial['probabilityLeft']
```

iii. The notes say this follows the decoder-task specification for prior probability categories and the papers’ biased-block design.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, with an undocumented fallback to `1` for any other value, then broadcasts the resulting category across time bins.

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
...
np.full(N_BINS, prior_val, dtype=np.int64)
```

iii. The notes justify the mapping directly from the decoder task: “0.2 -> 0, 0.5 -> 1, 0.8 -> 2.”

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from raw wheel position samples and wheel timestamps read from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
wheel_pos = np.load(wheel_pos_files[0])
wheel_ts = np.load(wheel_ts_files[0])
dt = np.diff(wheel_ts)
dp = np.diff(wheel_pos)
vel = dp / dt
speed = np.abs(vel)
```

iii. In the notes, the agent says this corresponds to the reference `wheel-speed` target, which in the reference loader is `abs(velocity)`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code computes a finite-difference velocity from wheel position and timestamps, takes the absolute value to get speed, places velocity timestamps at sample midpoints, interpolates the signal onto the trial bins, and later discretizes it into categories.

ii.
```python
dt = np.diff(wheel_ts)
dp = np.diff(wheel_pos)
vel = dp / dt
speed = np.abs(vel)
vel_ts = wheel_ts[:-1] + dt / 2
...
ws = interpolate_behavior_to_bins(wheel_ts, wheel_speed, t_start, t_end, BINSIZE, N_BINS)
```

iii. `CONVERSION_NOTES.md` says the intent was to match the reference `wheel-speed` behavior and admits a known limitation: the reference uses `SessionLoader.load_wheel()` with Gaussian smoothing, whereas this script uses a simple finite difference.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The agent pools all interpolated wheel-speed values from a session, computes session-specific quantile cut points for three equal-count bins, and assigns each time point to `low`, `medium`, or `high`.

ii.
```python
quantiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(all_vals, quantiles)
edges[0] = -np.inf
edges[-1] = np.inf
...
d = np.digitize(v, edges[1:-1])
...
wheel_disc, wheel_edges = discretize_to_bins(wheel_speed_raw, N_WHEEL_BINS)
```

iii. The notes justify quantile bins as a way to satisfy the task’s required 3-bin discretization while keeping class frequencies approximately balanced. They also state that the binning is done per session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned trial-by-trial using the same stimulus-onset window as the neural data. The code linearly interpolates wheel speed onto 100 points from `t_start + binsize` through `t_end` and drops trials without adequate coverage.

ii.
```python
stim_on = trial[ALIGN_TIME]
t_start = stim_on + TIME_WINDOW[0]
t_end = stim_on + TIME_WINDOW[1]
...
ws = interpolate_behavior_to_bins(wheel_ts, wheel_speed, t_start, t_end, BINSIZE, N_BINS)
...
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. The notes justify this with the stimulus-onset alignment instruction and cite the reference interpolation helper `get_behavior_per_interval()` from the trajectory.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, or from the corresponding right-camera files if the left-camera signal is unavailable.

ii.
```python
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
left_times_files = list(session_path.rglob('_ibl_leftCamera.times.npy'))
...
right_me_files = list(session_path.rglob('rightCamera.ROIMotionEnergy.npy'))
right_times_files = list(session_path.rglob('_ibl_rightCamera.times.npy'))
```

iii. The notes say this follows the reference code’s preference for left whisker motion energy with fallback to right, and they cite the data paper’s description of left and right cameras.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code prefers the left camera, falls back to the right camera, fixes a possible one-sample length mismatch by truncating timestamps, then linearly interpolates the motion-energy trace to the neural trial bins and later discretizes it.

ii.
```python
if left_me_files and left_times_files:
    me = np.load(left_me_files[0])
    times = np.load(left_times_files[0])
    if len(me) == len(times):
        return times, me
    elif len(me) == len(times) - 1:
        return times[:-1], me
...
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
```

iii. `CONVERSION_NOTES.md` states that the left camera is preferred because the reference code tries left first and then right. It also cites the trajectory’s `load_target_behavior()` logic for whisker motion energy.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is pooled within a session and discretized into three quantile-based bins labeled `low`, `medium`, and `high`.

ii.
```python
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, N_WHISKER_BINS)
...
d = np.digitize(v, edges[1:-1])
```

iii. The notes give the same justification as wheel speed: the task demands 3 bins, and quantiles provide roughly balanced class counts. The notes also explicitly say this is done per session.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The code aligns whisker motion energy to each stimulus-onset trial window, interpolates it onto the same 100-bin timeline as `neural`, and drops trials with missing coverage or NaNs.

ii.
```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
...
if ws is None or wm is None:
    continue
if np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
```

iii. The notes say that all behavior streams are interpolated to the neural bin grid using the same linear-interpolation strategy as the reference helper.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses a mixture of skipping and fallback behavior. Missing trials or spikes cause a whole session to be skipped. Missing wheel data also skips the session. Missing whisker or wheel coverage on a specific trial causes that trial to be dropped. Missing or out-of-range brain/channel mappings fall back to region id `0` (“root”), unknown `probabilityLeft` values fall back to prior category `1`, and missing `001` session folders fall back to the first date subdirectory found.

ii.
```python
if not trials_files:
    return None, None
...
if not spikes_list:
    return None
...
if wheel_speed is None:
    return None
...
if ws is None or wm is None:
    continue
...
cluster_brain_ids.append(0)  # root
...
prior_val = 1  # fallback
...
subdirs = sorted(parent.iterdir())
if subdirs:
    session_path = subdirs[0]
```

iii. The notes frame these choices as practical handling for missing modalities and local-cache irregularities. The trajectory also shows the agent checking cache contents and deciding to skip sessions with missing behavioral data.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive work is the full session loop, especially the per-trial spike binning and per-trial behavior interpolation inside `process_session`, plus repeated file-system scans with `rglob`.

ii.
```python
for i, (eid, session_info) in enumerate(sorted(sessions.items())):
    result = process_session(session_info, cache_dir, br)
...
for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
    neural = bin_spikes_trial(...)
    ws = interpolate_behavior_to_bins(...)
    wm = interpolate_behavior_to_bins(...)
```

iii. The trajectory shows the agent first writing a slower spike-binning loop, then replacing it with `searchsorted` and `np.add.at` after realizing conversion time would be large on the full dataset.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main candidates are the session-level Python loop over trials, the loop over block changes in `compute_trial_number_in_block`, the list-comprehension remap of every spike cluster id, the loop assigning each cluster’s brain id, and the repeated list-based global-region indexing.

ii.
```python
for i in range(len(block_changes)):
    ...
    trial_in_block[start:end] = np.arange(end - start)
...
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])
...
for cid in cluster_ids:
    ...
...
idx = np.array([all_regions_flat.index(r) for r in regions])
```

iii. The trajectory shows the agent explicitly optimizing `bin_spikes_trial`, but it did not revisit these remaining loops. No note claims these are already optimal.

## 10-c. What processing does the code repeat multiple times?

i. It rebuilds the same `time_since_stim` vector for every trial, repeatedly performs linear interpolation trial-by-trial for both wheel and whisker signals, repeatedly scans the file tree with `rglob`, and uses repeated list membership/index lookups when constructing global brain-region tables.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
...
ws = interpolate_behavior_to_bins(...)
wm = interpolate_behavior_to_bins(...)
...
trials_files = list(session_path.rglob('_ibl_trials.table.pqt'))
...
wheel_pos_files = list(session_path.rglob('_ibl_wheel.position.npy'))
...
if r not in all_regions_flat:
    all_regions_flat.append(r)
```

iii. The agent’s notes do not emphasize these repeats; they mostly focus on correctness and decoder validation. The repetition is visible in the final code.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and propagates several values that are never used downstream: `clusters_depths`, `cluster_labels`, `all_labels`, `valid_indices`, `good_trial_indices`, `all_subjects`, and `all_brain_region_idx`. It also computes and returns `wheel_edges` and `whisker_edges` from `process_session`, but those values are never written into the final dataset.

ii.
```python
clusters_depths = np.load(revision_path / 'clusters.depths.npy')
...
cluster_labels = metrics['label'].values
...
spike_times, spike_clusters, all_channels, all_brain_ids, all_labels = \
    merge_probes(...)
...
valid_indices = valid_trials.index.tolist()
...
good_trial_indices = []
...
'wheel_edges': wheel_edges,
'whisker_edges': whisker_edges,
```

iii. These are not justified in the notes. They appear to be leftovers from exploratory development and from carrying more metadata through intermediate steps than the final target format actually uses.
