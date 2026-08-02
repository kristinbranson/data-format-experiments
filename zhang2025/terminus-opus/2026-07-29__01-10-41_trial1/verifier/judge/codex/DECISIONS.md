# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script starts from `code/code_zhang2025/data/bwm_release.csv`, groups rows by `eid` to define sessions, and then iterates session-by-session. For each session it resolves a ONE-cache directory from `lab/subject/date`, loads the trials parquet table, loads spikes from every listed probe, loads wheel files from `alf/`, and loads whisker motion energy from left camera with fallback to right camera. Sessions with missing core files or with fewer than two valid trials are skipped rather than imputed.

ii.
```python
bwm_df = pd.read_csv('code/code_zhang2025/data/bwm_release.csv', index_col=0)
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first',
    'subject': 'first',
    'date': 'first',
    'probe_name': list,
    'pid': list
}).reset_index()

session_dir = find_session_dir(lab, subject, date)
trials_df = load_trials(session_dir)
spike_times, spike_clusters, cluster_regions = load_spikes(session_dir, probe_names)
wheel_times, wheel_speed = load_wheel_speed(session_dir)
me_times, me_values = load_whisker_motion_energy(session_dir)
```

iii. In `CONVERSION_NOTES.md`, the agent says it wanted to mirror the Zhang preprocessing pipeline while working directly from the cached filesystem rather than via ONE/brainbox loaders. In the trajectory it also notes that missing wheel or motion-energy files caused some sessions to be skipped, and it accepted that as part of its loading/validation logic.

## 1-b. How are the data split into subjects?

i. Subjects are not inferred from raw files; they are taken from the `subject` column in `bwm_release.csv`. During aggregation, each processed session gets a subject string and a numeric `subject_idx` produced by `subject_map`.

ii.
```python
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first',
    'subject': 'first',
    'date': 'first',
    'probe_name': list,
    'pid': list
}).reset_index()

subject_map = {}
if subj not in subject_map:
    subject_map[subj] = len(all_subjects)
    all_subjects.append(subj)
subject_idx_list.append(subject_map[subj])
```

iii. The notes describe the ONE-cache layout as `data/one_cache/{lab}/Subjects/{subject}/{date}/001/alf/`, so the agent treated the `subject` metadata in the release table as the subject split key.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique `eid` values from `bwm_release.csv`. Probe rows sharing an `eid` are collapsed into one session record, and all probes for that session are merged before trialization.

ii.
```python
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first',
    'subject': 'first',
    'date': 'first',
    'probe_name': list,
    'pid': list
}).reset_index()
session_groups.rename(columns={'probe_name': 'probe_names'}, inplace=True)

for sess_idx, (_, row) in enumerate(session_groups.iterrows()):
    eid = row['eid']
    probe_names = row['probe_names']
```

iii. `CONVERSION_NOTES.md` explicitly says the reference pipeline merges probes within each session because probes in the same session are not treated as independent. The trajectory repeatedly describes the session unit as `eid`.

## 1-d. How are the data split into trials?

i. The script treats each row of the trials table as a trial, defines a 2 s interval around `stimOn_times` (`-0.5` to `+1.5` s), bins spikes within that interval, interpolates behaviors onto the same interval, and then stores each remaining trial as one element in the session-level `neural`, `input`, and `output` lists.

ii.
```python
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]

binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)

valid_idx = np.where(combined_mask)[0]
neural_data = binned_spikes[valid_idx]
valid_trials = trials_df.iloc[valid_idx]
```

iii. The agent’s notes say the reference code uses `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`, and `interval_len=2`, so it used one trial row plus one aligned 2 s window as the trial definition.

## 1-e. How are trials filtered based on quality controls?

i. Trials are first filtered by reaction time, maximum trial length, required non-NaN task events, and exclusion of `choice == 0`. They are then filtered again by requiring both wheel and whisker traces to be loadable and interpolable for the chosen interval. Sessions with fewer than two surviving trials are discarded.

ii.
```python
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)
trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN)
for event in nan_exclude:
    if event in trials_df.columns:
        mask &= ~trials_df[event].isna()
mask &= (trials_df['choice'] != 0)

combined_mask = mask.values & wheel_valid & me_valid
if n_valid < 2:
    return None
```

iii. The notes say this mask was meant to match `load_trials_and_mask()` from the Zhang utility code, and the trajectory shows the agent added behavior-availability filtering after observing sessions with zero valid trials when wheel or camera data were missing.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw spike times and cluster IDs (`spikes.times.npy`, `spikes.clusters.npy`) for each probe, plus `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` to assign each cluster a brain region.

ii.
```python
spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
clusters_channels = np.load(os.path.join(ks_dir, 'clusters.channels.npy')).flatten()
brain_ids = np.load(brain_id_files[0]).flatten()
cluster_brain_ids = brain_ids[clusters_channels]
```

iii. In the notes, the agent summarizes the reference neural path as “load spikes, merge probes, map to Beryl, bin spike counts,” and the trajectory shows it inspected exactly these `.npy` files while reconstructing the pipeline offline.

## 2-b. How is the `neural` data processed?

i. The script merges spikes across probes by offsetting cluster IDs, sorts merged spikes by time, maps channels to Beryl regions, and bins spike counts into 20 ms bins for each trial window. No firing-rate normalization or smoothing is applied after binning.

ii.
```python
spike_clusters = spike_clusters + cluster_offset
cluster_offset += n_clusters
merged_times = np.concatenate(all_spike_times)
merged_clusters = np.concatenate(all_spike_clusters)
sort_idx = np.argsort(merged_times, kind='stable')

time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
time_bin_idx = np.clip(time_bin_idx, 0, N_BINS - 1)
np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
```

iii. `CONVERSION_NOTES.md` says the intended match was the Zhang chain `prepare_data -> merge_probes -> bin_spiking_data`, with 20 ms spike counts in a 2 s window around `stimOn_times`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script does not apply neuron-level QC. It keeps every cluster present in the spike-sorting output for processed probes and only filters at the trial/session level. Brain-region metadata are still attached to every kept cluster.

ii.
```python
def load_spikes(session_dir, probe_names):
    ...
    spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
    spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
    ...
    n_clusters = len(clusters_channels)
```

iii. The notes explicitly justify this with “reference code uses ALL clusters (no QC filtering, qc=None).” The trajectory repeats that choice after inspecting cluster labels, even though the BWM paper separately discusses well-isolated-neuron filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All neural trials are aligned to `stimOn_times`, with a fixed window from `-0.5` s to `+1.5` s. The 100 output bins therefore represent the same stimulus-onset-centered time axis for every trial.

ii.
```python
BINSIZE = 0.02
TIME_WINDOW = (-0.5, 1.5)
ALIGN_TIME = 'stimOn_times'

stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
```

iii. The notes frame this as an explicit choice: the task instructions said “Temporally align based on stimulus onset,” and the agent decided to follow that plus `0_data_caching.py`, even though the methods text uses different alignments for some decoders.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins throughout, producing `N_BINS = 100` over a 2 s interval. There is no second-stage rebinning after spike counts are assigned to bins.

ii.
```python
BINSIZE = 0.02  # 20 ms
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. The notes cite the Zhang code constants `binsize=0.02`, `interval_len=2`, and the trajectory repeatedly treats 20 ms as the global dataset resolution required by the decoder task.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This input is only anchored to raw `stimOn_times`; the actual values are generated from fixed constants (`BINSIZE`, `TIME_WINDOW`) rather than read from a trial column. It is therefore a synthetic time axis with `stimOn_times` defining zero.

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
time_since_stim = time_since_stim.astype(np.float32)
```

iii. The trajectory’s mapping plan says this input should be “time since stimulus onset - continuous, time-varying,” so the agent made a standard relative-time vector instead of copying any raw timestamp array into each trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code computes the center of each 20 ms bin from `-0.49` s to `+1.49` s relative to the alignment event. The same 100-value vector is reused for every valid trial.

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
...
inp = np.vstack([
    time_since_stim[np.newaxis, :],
    np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)
])
```

iii. The justification in the trajectory is pragmatic: the decoder input was requested as a continuous, time-varying representation aligned to stimulus onset, so bin centers were used to match the neural bins exactly.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned one-to-one with the neural bins: same trial window, same number of bins, same 20 ms spacing, and the values represent the centers of the neural count bins.

ii.
```python
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
...
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
```

iii. The notes say the agent chose to keep all decoder variables on the same `stimOn_times` axis, so time-since-stimulus was intentionally constructed to share the neural bin grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table’s `probabilityLeft` column. The code interprets any change in `probabilityLeft` as a block boundary.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[valid_idx]
```

iii. The trajectory mapping plan explicitly identifies `probabilityLeft` as the source for block structure because that is the only trial-wise block variable exposed in the table and in the papers’ task description.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code walks through all trials in order, resets a counter to `1` when `probabilityLeft` changes from the previous trial, increments within each run, and only after that subsamples to valid trials. It does not separately special-case the initial unbiased block.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    counter = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            counter = 1
        trial_nums[i] = counter
        counter += 1
    return trial_nums
```

iii. The agent’s trajectory says “Need to compute on ALL trials first, then select valid ones,” which explains why it preserved original trial order before applying the trial mask. There is no richer justification beyond using `probabilityLeft` transitions as block boundaries.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived directly from `trials_df['choice']`.

ii.
```python
choice = valid_trials['choice'].values.copy()
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

iii. The notes and trajectory both mention that IBL trial tables encode choice as `-1`, `0`, or `1`, so the agent used the trial-table choice variable and relied on earlier filtering to remove `0` no-choice trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Trials with `choice == 0` are removed by the trial mask. Remaining choices are remapped from the IBL convention `-1/1` to decoder labels `0/1`, and then repeated across all 100 bins so the output tensor keeps a uniform `(4, 100)` shape.

ii.
```python
mask &= (trials_df['choice'] != 0)
...
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
...
np.full((1, N_BINS), choice_mapped[i], dtype=np.int64)
```

iii. The trajectory explicitly states “Need to map: -1 -> left (0), 1 -> right (1),” and the notes describe choice as a binary per-trial decoder target.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior is derived directly from `trials_df['probabilityLeft']`.

ii.
```python
prob_left = valid_trials['probabilityLeft'].values
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

iii. The task specification itself gave the target mapping `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, and the notes list the same mapping under expected processing details.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps each `probabilityLeft` value into one of three integer classes using a hard-coded dictionary and silently falls back to class `1` if an unexpected probability appears. Like choice, it then repeats the class value across time bins for each trial.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
...
np.full((1, N_BINS), prior_mapped[i], dtype=np.int64)
```

iii. The notes justify the main mapping from the decoder task. The trajectory does not justify the fallback behavior; that appears to be an implementation shortcut rather than a documented requirement.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from raw wheel position and timestamps: `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
pos_file = os.path.join(session_dir, 'alf', '_ibl_wheel.position.npy')
ts_file = os.path.join(session_dir, 'alf', '_ibl_wheel.timestamps.npy')
wheel_pos = np.load(pos_file).flatten()
wheel_ts = np.load(ts_file).flatten()
```

iii. The notes say the intent was to match the `brainbox` wheel preprocessing path used by `SessionLoader.load_wheel()`, which starts from position plus timestamps.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The script linearly interpolates wheel position to 1000 Hz, applies an 8th-order 20 Hz Butterworth low-pass filter, differentiates to obtain velocity, takes absolute value to obtain speed, and then linearly interpolates that continuous speed onto the trial bins.

ii.
```python
t_interp = np.arange(wheel_ts[0], wheel_ts[-1], 1.0 / WHEEL_FS)
pos_interp = interpolate.interp1d(wheel_ts, wheel_pos, kind='linear')(t_interp)
sos = signal.butter(N=WHEEL_FILTER_ORDER, Wn=WHEEL_CORNER_FREQ / WHEEL_FS * 2,
                    btype='lowpass', output='sos')
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * WHEEL_FS
speed = np.abs(vel)
...
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, trial_starts, trial_ends)
```

iii. `CONVERSION_NOTES.md` says this was taken from “brainbox wheel.py,” and that justification is consistent with the bundled `ibllib` implementation the agent inspected during the trajectory.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. After interpolation to the trial grid, wheel speed is discretized into three ordinal classes by computing global within-session quantile thresholds over all valid wheel-speed bin values, then applying `np.digitize`.

ii.
```python
flat = values[~np.isnan(values)].flatten()
quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
boundaries = np.quantile(flat, quantiles)
result = np.digitize(values, boundaries).astype(np.int64)
```

iii. The trajectory only says wheel speed must be “discretized into 3 bins”; it does not cite a reference thresholding rule. This quantile strategy appears to be the agent’s own choice to satisfy the categorical-output requirement.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same `stimOn_times` windows as the neural data, using the identical `trial_starts`/`trial_ends` and the same 100 bin centers. It is not aligned to first movement onset.

ii.
```python
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
...
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, trial_starts, trial_ends)
```

iii. The notes explicitly acknowledge the mismatch with the methods text and justify the choice by the task statement “Temporally align based on stimulus onset” plus the parameters in `0_data_caching.py`.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from precomputed ROI motion-energy arrays and corresponding camera timestamps, preferring the left camera and falling back to the right camera.

ii.
```python
me_file = find_file(session_dir, 'leftCamera.ROIMotionEnergy.npy')
cam_file = find_file(session_dir, '_ibl_leftCamera.times.npy')
...
me_file = find_file(session_dir, 'rightCamera.ROIMotionEnergy.npy')
cam_file = find_file(session_dir, '_ibl_rightCamera.times.npy')
```

iii. The notes say this matches the reference code’s left-camera-first behavior. The methods text supplied the scientific origin of the metric, but the script itself only loads the precomputed motion-energy traces.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script does not recompute motion energy from video. It loads the precomputed 1D trace, checks that the trace length matches camera timestamps, selects left or right view, and then linearly interpolates the trace onto the trial bins.

ii.
```python
if me_file and cam_file:
    me = np.load(me_file).flatten()
    cam_times = np.load(cam_file).flatten()
    if len(me) == len(cam_times) and len(me) > 0:
        return cam_times, me
...
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```

iii. The notes justify this by saying the reference utility `load_target_behavior()` loads whisker motion energy directly from IBL motion-energy objects, so the agent preserved that level of preprocessing rather than rebuilding the computer-vision pipeline.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The trace is discretized with the same generic routine used for wheel speed: within-session quantile thresholds over all valid interpolated values, followed by `np.digitize` into classes `0,1,2`.

ii.
```python
me_data = me_binned[valid_idx]
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
```

iii. The task required a 3-bin categorical output but did not prescribe thresholds. The trajectory does not provide a paper-backed rationale here; the quantile scheme is another agent choice.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned exactly like wheel speed: `stimOn_times`, `-0.5` to `+1.5` s, 20 ms bins, same bin centers as the neural matrix.

ii.
```python
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
...
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```

iii. The notes explicitly record this as a deliberate choice to follow the decoder-task alignment requirement instead of the methods paper’s first-movement alignment for dynamic behaviors.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missingness conservatively. Missing trial files, spike files, wheel files, or motion-energy files cause a session to be skipped. Length mismatches, too-short traces, NaNs in required trial columns, NaNs or too-few behavior samples inside a trial interval, and sessions with fewer than two valid trials are all handled by discarding data rather than filling it in. For whisker motion energy only, the code falls back from left camera to right camera.

ii.
```python
if not trial_files:
    return None
...
if not os.path.exists(pos_file) or not os.path.exists(ts_file):
    return None, None
if len(wheel_pos) != len(wheel_ts):
    return None, None
...
if len(beh_v) < 2:
    valid[trial_idx] = False
    continue
...
combined_mask = mask.values & wheel_valid & me_valid
if n_valid < 2:
    return None
```

iii. The trajectory shows the agent encountered skipped sessions and treated that as acceptable validation behavior. The notes emphasize consistency checks and make no attempt to justify any imputation strategy, because none was implemented.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is per-trial spike binning, especially for sessions with many clusters and trials. Behavior interpolation is also repeated trial-by-trial, but the spike-count loop dominates runtime.

ii.
```python
print(f"  Binning spikes ({n_clusters} clusters, {len(trials_df)} trials)...", end=' ')
t_bin = time.time()
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
print(f"{time.time()-t_bin:.1f}s")
```

iii. The trajectory explicitly records timings such as `104.1s` and `43.4s` for spike binning and says “Spike binning is very slow,” which is the clearest justification for identifying it as the main bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop in `bin_spikes_per_trial`, the trial loop in `interpolate_behavior_to_bins`, the loop that fills `input_list`, the loop that fills `output_list`, and the per-neuron region-index loop could all be vectorized or at least batched more aggressively.

ii.
```python
for trial_idx in valid_indices:
    ...

for trial_idx in range(n_trials):
    ...

for i in range(n_valid_trials):
    input_list.append(inp)

for i in range(n_valid_trials):
    output_list.append(out)

for i, r in enumerate(regions):
    ...
```

iii. The trajectory shows the agent recognized the spike loop as slow and described a later revision as “optimize spike binning,” but the final code still contains several Python-level loops over trials.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly computes identical time-since-stimulus vectors for every trial output container, repeatedly materializes constant per-trial choice/prior arrays across 100 bins, separately interpolates wheel and whisker traces with nearly identical logic, and reruns the same left/right-camera search pattern per session.

ii.
```python
inp = np.vstack([
    time_since_stim[np.newaxis, :],
    np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)
])

out = np.vstack([
    np.full((1, N_BINS), choice_mapped[i], dtype=np.int64),
    np.full((1, N_BINS), prior_mapped[i], dtype=np.int64),
    wheel_disc[i:i+1, :],
    me_disc[i:i+1, :],
])
```

iii. The trajectory does not present this as a deliberate design goal; it mainly focused on getting a valid dataset. The repeated materialization is therefore best understood as convenience-driven implementation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins spikes for every trial before applying the trial mask, computes continuous wheel speed and continuous whisker motion energy and then discards them after discretization, and expands scalar per-trial targets into full-length 100-bin constant arrays even though the values are static within a trial.

ii.
```python
# Bin spikes per trial (for ALL trials first, then filter)
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
...
wheel_data = wheel_binned[valid_idx]
me_data = me_binned[valid_idx]
wheel_disc = discretize_time_varying(wheel_data, N_DISC_BINS)
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
...
np.full((1, N_BINS), choice_mapped[i], dtype=np.int64)
np.full((1, N_BINS), prior_mapped[i], dtype=np.int64)
```

iii. The trajectory acknowledges at least one of these inefficiencies directly: it notes that spike binning was done “for ALL trials first, then filter” and that runtime was a concern. The rest are consequences of the chosen output format rather than paper-driven requirements.
