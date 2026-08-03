# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads session metadata from cached parquet tables and a BWM freeze CSV, intersects the freeze-file `eid`s with the locally cached session IDs, and then iterates session-by-session over that cached subset. For each `eid`, it resolves a local session path from `datasets.pqt` or `sessions.pqt` and reads local ALF files directly. It does not download the missing sessions; the notes say only 354 of 459 sessions were available locally.

ii. 
```python
sessions_df = pd.read_parquet(os.path.join(TABLES_DIR, 'sessions.pqt'))
datasets_df = pd.read_parquet(os.path.join(TABLES_DIR, 'datasets.pqt'))
bwm_df = pd.read_csv(BWM_CSV, index_col=0)

bwm_eids = bwm_df['eid'].unique()
cache_eids = set(sessions_df.index.astype(str))
available_eids = [eid for eid in bwm_eids if eid in cache_eids]

for eid_idx, eid in enumerate(available_eids):
    session_path = find_session_path(eid, sessions_df, datasets_df)
```

iii. `CONVERSION_NOTES.md` says “354 of 459 sessions available in local cache.” In the trajectory, the agent first used `/app/code/code_zhang2025/data/bwm_release.csv`, then patched `find_session_path()` to use `datasets_df['session_path']` because the local cache layout differed from its first assumption.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from `sessions_df.loc[eid]['subject']`. Unique subject names are appended once to `all_subjects`, and each session gets an integer `subject_idx`.

ii. 
```python
row = sessions_df.loc[eid]
subject = row['subject']

if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subject_to_idx)
    all_subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. The trajectory shows the agent using the BWM session tables as the authoritative session/subject index. The notes describe the dataset as multi-mouse and report 107 processed subjects.

## 1-c. How are the data split into sessions?

i. Each `eid` in `available_eids` becomes one output session. The outer lists `all_neural`, `all_input`, `all_output`, and `all_brain_region_idx` each receive one entry per processed `eid`.

ii. 
```python
for eid_idx, eid in enumerate(available_eids):
    ...
    session_neural = []
    session_input = []
    session_output = []
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
    all_brain_region_idx.append(beryl_acronyms)
```

iii. The reference code read by the agent (`0_data_caching.py`) is also session-oriented around `eid`. The agent followed that structure but processed all locally available sessions instead of sampling.

## 1-d. How are the data split into trials?

i. Trials start from the `_ibl_trials.table.pqt` rows. The code first applies the trial-quality mask from `load_trials()`, then keeps only the masked rows in `valid_trials`, and finally keeps only the trial indices that also have both wheel and whisker behavioral coverage (`combined_mask`).

ii. 
```python
trials = pd.read_parquet(trials_file)
...
valid_trials = trials[mask].copy()
align_times = valid_trials[ALIGN_TIME].values
...
combined_mask = wheel_mask & me_mask
combined_indices = np.where(combined_mask)[0]

for trial_local_idx, trial_global_idx in enumerate(combined_indices):
    neural_trial = binned_spikes[trial_global_idx]
    ...
    session_neural.append(neural_trial.astype(np.float64))
```

iii. The notes say “Trials with all data” are retained and sessions are skipped if too few remain. This matches the behavior-completeness logic the agent inferred from the reference code’s later alignment stage.

## 1-e. How are trials filtered based on quality controls?

i. The explicit trial mask removes trials with NaNs in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`; removes trials with reaction time outside 0.08 to 2.0 s; and removes no-choice trials (`choice == 0`). After that, trials are further dropped if wheel or whisker data are missing, contain NaNs, or fail interval-coverage checks during interpolation.

ii. 
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

for col in NAN_EXCLUDE:
    if col in trials.columns:
        mask &= ~trials[col].isna()

rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)
mask &= (trials['choice'] != 0)
...
if np.any(np.isnan(local_vals)):
    good_mask[trial_idx] = False
...
combined_mask = wheel_mask & me_mask
```

iii. The notes justify this as “matching reference code `load_trials_and_mask()` defaults.” In the trajectory, the agent explicitly cited that helper as the source for the NaN, RT, and no-choice exclusions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is built from per-probe spike times and spike-cluster assignments plus cluster-to-channel and channel-to-brain-location mappings. The relevant raw files are `spikes.times.npy`, `spikes.clusters.npy`, `clusters.channels.npy`, `channels.brainLocationIds_ccf_2017.npy`, and `clusters.metrics.pqt` when the quality filter is applied.

ii. 
```python
spike_times_file = find_file_with_revision(probe_dir, 'spikes.times.npy')
spike_clusters_file = find_file_with_revision(probe_dir, 'spikes.clusters.npy')
clusters_channels_file = find_file_with_revision(probe_dir, 'clusters.channels.npy')
clusters_metrics_file = find_file_with_revision(probe_dir, 'clusters.metrics.pqt')
channels_brain_ids_file = find_file_with_revision(probe_dir, 'channels.brainLocationIds_ccf_2017.npy')

spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
clusters_channels = np.load(clusters_channels_file).flatten()
channels_brain_ids = np.load(channels_brain_ids_file).flatten()
cluster_brain_ids = channels_brain_ids[clusters_channels]
```

iii. The trajectory shows the agent reading `ibl_data_utils.load_spiking_data()` and then reproducing its use of spikes, clusters, and channel metadata with a local-file implementation.

## 2-b. How is the `neural` data processed?

i. The code merges probes within a session, reindexes cluster IDs across probes, sorts all spikes by time, and bins spike counts per trial into 20 ms bins over a 2 s window around the alignment event. The stored value is spike count, not firing rate or smoothed activity.

ii. 
```python
for spike_times, spike_clusters, brain_ids, n_clusters in probes_data:
    all_spike_times.append(spike_times)
    all_spike_clusters.append(spike_clusters + cluster_offset)
    ...
sort_idx = np.argsort(merged_times, kind='stable')
merged_times = merged_times[sort_idx]
merged_clusters = merged_clusters[sort_idx]

for trial_idx in range(n_trials):
    t_beg = align_times[trial_idx] + time_window[0]
    t_end = align_times[trial_idx] + time_window[1]
    ...
    bin_indices = np.minimum(
        ((trial_times - t_beg) / binsize).astype(np.int64),
        n_bins - 1
    )
    np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

iii. The notes cite `merge_probes()` and `bin_spiking_data() -> get_spike_data_per_interval()` from the reference code. The methods excerpt in the trajectory also says the decoder uses “temporally binned spike counts.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The operative code filters to clusters with `metrics['label'] >= 1.0` on each probe, remaps the kept cluster IDs, and then skips any session with fewer than 5 remaining clusters. This conflicts with the script header and metadata string, which still claim “all clusters” and “cluster_quality_filter: none.”

ii. 
```python
def load_spike_data(session_path, probe_name, qc_threshold=1.0):
    ...
    if qc_threshold is not None and clusters_metrics_file is not None:
        metrics = pd.read_parquet(clusters_metrics_file)
        if 'label' in metrics.columns:
            good_mask = metrics['label'].values >= qc_threshold
            good_cluster_ids = np.where(good_mask)[0]
            ...
            spike_mask, ib = ismember(spike_clusters, good_cluster_ids)
            spike_times = spike_times[spike_mask]
            spike_clusters = ib.astype(np.int32)
            cluster_brain_ids = cluster_brain_ids[good_cluster_ids]
            n_clusters = len(good_cluster_ids)
...
if n_clusters < MIN_NEURONS:
    print(f"  SKIP: too few clusters ({n_clusters} < {MIN_NEURONS})")
```

iii. The justification in `CONVERSION_NOTES.md` explicitly says the agent chose “well-isolated neurons (cluster quality label >= 1.0)” even though “the Zhang reference code uses qc=None.” The trajectory shows the agent noticing that mismatch and deliberately choosing the paper-style filter anyway.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to stimulus onset, specifically `stimOn_times`, with a fixed window from -0.5 s to +1.5 s around each valid trial’s stimulus onset.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
valid_trials = trials[mask].copy()
align_times = valid_trials[ALIGN_TIME].values
...
t_beg = align_times[trial_idx] + time_window[0]
t_end = align_times[trial_idx] + time_window[1]
```

iii. Both `CONVERSION_NOTES.md` and the trajectory cite `0_data_caching.py` line 53 and the decoder-task instruction to align everything to stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins, yielding 100 bins over the 2 s window. No later temporal rebinning is applied; the binned counts are stored directly.

ii. 
```python
BINSIZE = 0.02  # 20 ms bins
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_TIMEBINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
...
binned = np.zeros((n_trials, n_clusters_total, n_bins), dtype=np.float32)
```

iii. The notes cite both the reference code and the copied methods text: “20-ms bins, producing T = 100 time steps.”

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is only indirectly derived from the raw trial timing through the choice of alignment event `stimOn_times`; the actual values are generated from the global constants `TIME_WINDOW`, `BINSIZE`, and `N_TIMEBINS`, not from per-trial timestamps beyond that alignment choice.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_TIMEBINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
...
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. The notes describe this input as “continuous, time-varying” and tied to the stimulus-onset alignment.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code constructs one fixed 100-sample vector for every trial using `np.linspace(-0.48, 1.5, 100)`. It does not compute a binary onset indicator; it stores continuous elapsed time relative to stimulus onset.

ii. 
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
...
input_trial_full = np.vstack([
    time_since_stim.reshape(1, -1),
    np.full((1, N_TIMEBINS), trial_num, dtype=np.float64)
])
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as “continuous, time-varying,” which follows the decoder-task specification even though the generic format guidance suggested binary time-series for event-onset variables.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same 100-bin trial grid as the neural and interpolated behavioral data. The time axis runs over the same stimulus-aligned window and is stacked trial-by-trial alongside the neural matrix.

ii. 
```python
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters,
    align_times, TIME_WINDOW, BINSIZE
)
...
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
)
...
session_input.append(input_trial_full)
```

iii. The notes say behavior interpolation also uses bin centers `linspace(t_beg + binsize, t_end, n_bins)`, so the agent intended the time input to live on that same grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column in the trials table.

ii. 
```python
def compute_trial_number_in_block(trials_df, mask):
    ...
    prob_left = trials_df['probabilityLeft'].values
```

iii. The notes justify this as a block-context input for the decoder and describe it as a count from the start of each block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code walks through `probabilityLeft` sequentially, resets the block start index whenever the probability changes (or becomes NaN), computes a 0-based within-block counter on the full trial table, and only then applies the trial mask.

ii. 
```python
trial_nums = np.zeros(len(trials_df), dtype=np.float64)
block_start = 0
for i in range(1, len(prob_left)):
    if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
        block_start = i
    trial_nums[i] = i - block_start

return trial_nums[mask]
```

iii. `CONVERSION_NOTES.md` says this variable is a “0-indexed count from block start.”

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from the `choice` column of the trials table.

ii. 
```python
choice_vals = valid_trials['choice'].values  # -1 or 1
```

iii. The notes and the methods text both treat choice as a per-trial behavioral variable from the IBL task.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw IBL coding is converted from `-1/1` to `0/1`, where left becomes 0 and right becomes 1. The chosen label is then repeated across all 100 time bins for the trial.

ii. 
```python
choice = 0 if choice_vals[trial_global_idx] == -1 else 1
...
output_trial = np.vstack([
    np.full((1, N_TIMEBINS), choice, dtype=np.int64),
    ...
])
```

iii. `CONVERSION_NOTES.md` states this mapping explicitly: “Original data: -1=left, 1=right.”

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior is derived from the `probabilityLeft` column in the trials table.

ii. 
```python
prob_left_vals = valid_trials['probabilityLeft'].values  # 0.2, 0.5, 0.8
```

iii. The notes and methods text both describe this task variable as the block-dependent prior probability of the stimulus appearing on the left.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps `probabilityLeft` from the raw values 0.2, 0.5, and 0.8 into categorical labels 0, 1, and 2, then repeats that category across all 100 time bins of the trial.

ii. 
```python
prob = prob_left_vals[trial_global_idx]
if prob == 0.2:
    prior = 0
elif prob == 0.5:
    prior = 1
else:  # 0.8
    prior = 2
...
np.full((1, N_TIMEBINS), prior, dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` states the same 0.2/0.5/0.8 to 0/1/2 conversion.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
pos_file = alf_dir / '_ibl_wheel.position.npy'
ts_file = alf_dir / '_ibl_wheel.timestamps.npy'
...
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The notes say the source is “raw wheel position and timestamps.”

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code interpolates wheel position onto a uniform 1 kHz grid with `interpolate_position`, computes filtered velocity with `velocity_filtered`, takes the absolute value to get speed, and then linearly interpolates that speed onto the trial bin centers in the stimulus-aligned window.

ii. 
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
pos_interp, ts_interp = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(pos_interp, 1000)
speed = np.abs(velocity)
...
wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, TIME_WINDOW, BINSIZE
)
```

iii. The trajectory shows the agent patching `load_wheel_speed()` to better mimic `SessionLoader.load_wheel()`, and the notes say this matches ibllib’s wheel processing.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After interpolation, the code concatenates wheel-speed samples from all retained trials in a session, computes the 33.33rd and 66.67th percentiles, and uses `np.digitize()` to assign low/medium/high classes 0/1/2 per time bin.

ii. 
```python
all_wheel_concat = np.concatenate(all_wheel_vals)
wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
...
wheel_trial = wheel_binned_list[trial_global_idx]
wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` justifies this as “3 equal-frequency bins (33rd and 67th percentiles per session).”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned with the same per-trial `align_times`, `TIME_WINDOW`, and `BINSIZE` used for neural binning, and interpolated onto the same 100 sample points.

ii. 
```python
wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, TIME_WINDOW, BINSIZE
)
...
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The notes say behavior interpolation is used to “match neural bin times.” The instruction to align everything to stimulus onset is the reason the dynamic behavior is not first-movement aligned here.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy comes from `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`, with fallback to the analogous right-camera files when the left-camera files are unavailable.

ii. 
```python
me_file = find_file_with_revision(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
ts_file = find_file_with_revision(alf_dir, '_ibl_leftCamera.times.npy')

if me_file is None or ts_file is None:
    me_file = find_file_with_revision(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    ts_file = find_file_with_revision(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. The notes explicitly say “left camera preferred, right camera fallback.”

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion-energy vector and its timestamps are trimmed to equal length, NaN samples are removed, sessions with too few valid samples are rejected for this signal, and the remaining signal is linearly interpolated onto the 100 trial bin centers.

ii. 
```python
min_len = min(len(me), len(times))
me = me[:min_len]
times = times[:min_len]

valid = ~np.isnan(me) & ~np.isnan(times)
if np.sum(valid) < 10:
    return None, None

return times[valid], me[valid]
...
me_binned_list, me_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, TIME_WINDOW, BINSIZE
)
```

iii. The methods text in the trajectory describes whisker motion energy as frame-difference energy from the whisker pad. The notes add the left/right camera choice and interpolation strategy.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The code concatenates all interpolated whisker-motion-energy values from retained trials in a session, computes the 33.33rd and 66.67th percentiles, and digitizes each time bin into low/medium/high classes 0/1/2.

ii. 
```python
all_me_concat = np.concatenate(all_me_vals)
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
...
me_trial = me_binned_list[trial_global_idx]
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` gives the same justification as wheel speed: “3 equal-frequency bins (33rd and 67th percentiles per session).”

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned exactly like wheel speed: the same stimulus-aligned trial window and the same 100 bin-center time points are used.

ii. 
```python
me_binned_list, me_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, TIME_WINDOW, BINSIZE
)
...
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The notes say behavior interpolation is done to “match neural bin times,” and the overall alignment decision is stimulus onset.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code mainly handles missing or malformed data by dropping it. It searches revision subdirectories for missing files, skips sessions with missing trials/probe data or too few neurons/trials, removes trial rows with NaN task events, drops behavioral trials with empty data, NaNs, or insufficient coverage, truncates whisker arrays to the shorter of values and timestamps, and keeps zero-spike trials as all-zero neural matrices. It does not impute missing task values or attempt to recover missing sessions by download.

ii. 
```python
def find_file_with_revision(base_dir, filename):
    ...
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith('#') and d.name.endswith('#'):
            candidate = d / filename
            if candidate.exists():
                return candidate

if trials_file is None:
    return None, None
...
if len(local_vals) == 0:
    good_mask[trial_idx] = False
...
min_len = min(len(me), len(times))
me = me[:min_len]
times = times[:min_len]
...
if n_final < 2:
    print(f"  SKIP: too few trials with complete data")
```

iii. The notes describe missing local cache coverage as a known limitation. The full verification log summarized in the notes also acknowledges three all-zero neural trials that were left in the dataset.

## 10-a. What are the most time-consuming steps of the code?

i. The heavy steps are loading large parquet/NumPy files for every session and probe, per-trial spike binning across all spikes, per-trial behavioral interpolation, and concatenating all retained wheel/whisker samples to compute session-wise bin thresholds. The implementation is single-process despite exposing `--n-workers`.

ii. 
```python
for eid_idx, eid in enumerate(available_eids):
    ...
    for probe_name in probe_dirs:
        result = load_spike_data(session_path, probe_name)
    ...
    binned_spikes = bin_spikes_per_trial(...)
    ...
    wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(...)
    me_binned_list, me_mask = interpolate_behavior_to_bins(...)
    ...
    all_wheel_concat = np.concatenate(all_wheel_vals)
    all_me_concat = np.concatenate(all_me_vals)
```

iii. The trajectory shows long-running full-conversion and validation jobs. The reference code the agent read used multiprocessing for both spike and behavior binning, which highlights these steps as the intended bottlenecks.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest candidates are the outer per-trial loop in `bin_spikes_per_trial()`, the per-trial interpolation loop in `interpolate_behavior_to_bins()`, and the two passes over retained trials used first to gather all wheel/ME values and then to build per-trial outputs.

ii. 
```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)

for trial_idx in range(n_trials):
    ...
    y_interp = interp1d(local_times, local_vals, kind='linear',
                       fill_value='extrapolate')(x_interp)

for idx in combined_indices:
    ...

for trial_local_idx, trial_global_idx in enumerate(combined_indices):
    ...
```

iii. The agent justified these helpers as simplified local reimplementations of the reference code. The reference code it read instead used multiprocessing around comparable loops.

## 10-c. What processing does the code repeat multiple times?

i. It repeats file-revision directory scans for each requested file, iterates over retained trials once to collect continuous wheel/ME values and again to discretize/store them, recomputes interpolation bin centers separately for every trial, and scans session metadata structures repeatedly while building output bookkeeping.

ii. 
```python
for d in sorted(base_dir.iterdir()):
    ...

for idx in combined_indices:
    if wheel_binned_list[idx] is not None:
        all_wheel_vals.append(wheel_binned_list[idx])
    if me_binned_list[idx] is not None:
        all_me_vals.append(me_binned_list[idx])

for trial_local_idx, trial_global_idx in enumerate(combined_indices):
    ...

x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The notes emphasize reproducibility and matching the reference code, but the actual implementation favors straightforward repeated passes over more careful reuse of already computed results.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several intermediates are computed but never used: `input_trial`, the placeholder variables `wheel_binned` and `me_binned`, and the helper `discretize_to_bins()`. The script also expands trial-level variables (`trial_number_in_block`, `choice`, `prior`) into full 100-bin constant time series even though they are static within a trial. Finally, the metadata still records `cluster_quality_filter` as “none,” which is discarded by the actual filtering path used earlier.

ii. 
```python
wheel_binned = None
...
me_binned = None
...
def discretize_to_bins(values, n_bins=3):
    ...

input_trial = np.array([trial_num], dtype=np.float64)  # (1,) per-trial
...
np.full((1, N_TIMEBINS), trial_num, dtype=np.float64)
np.full((1, N_TIMEBINS), choice, dtype=np.int64)
np.full((1, N_TIMEBINS), prior, dtype=np.int64)
...
'cluster_quality_filter': 'none (all clusters, matching reference code)',
```

iii. The trajectory shows the agent originally intending an all-clusters pipeline, then switching to `label >= 1.0` filtering. The stale metadata and unused helpers/intermediates are leftovers from that iteration process.
