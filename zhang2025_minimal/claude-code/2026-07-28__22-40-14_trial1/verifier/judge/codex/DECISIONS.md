# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the ONE API or the IBL loader classes for the main conversion path. Instead, it read release-index parquet/CSV files directly from disk, selected session `eid`s by intersecting `bwm_release.csv` with `sessions.pqt`, then opened raw parquet and `.npy` files from the cache by constructing filesystem paths and searching revision directories. Trials were then loaded from `_ibl_trials.table.pqt`, spike data from per-probe `pykilosort` folders, wheel data from `_ibl_wheel.*.npy`, and whisker motion energy from camera `.npy` files.

ii. 
```python
sessions_df = pd.read_parquet(os.path.join(TABLES_DIR, 'sessions.pqt'))
datasets_df = pd.read_parquet(os.path.join(TABLES_DIR, 'datasets.pqt'))
bwm_df = pd.read_csv(BWM_CSV, index_col=0)

bwm_eids = bwm_df['eid'].unique()
cache_eids = set(sessions_df.index.astype(str))
available_eids = [eid for eid in bwm_eids if eid in cache_eids]
```

```python
trials_file = find_file_with_revision(alf_dir, '_ibl_trials.table.pqt')
spike_times_file = find_file_with_revision(probe_dir, 'spikes.times.npy')
pos_file = alf_dir / '_ibl_wheel.position.npy'
me_file = find_file_with_revision(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
```

iii. In the trajectory, the AI first tried `SessionLoader`, found that it was not resolving the versioned trials table correctly, inspected the session directory, and then explicitly decided to “write a direct data loading approach” instead (steps 43-48). The stated justification was robustness to revision directories while still matching the reference processing.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` field in `sessions.pqt` for each surviving `eid`. During assembly, the AI builds `subjects` incrementally in first-seen session order and records `subject_idx` from that insertion-order mapping.

ii. 
```python
row = sessions_df.loc[eid]
subject = row['subject']
```

```python
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subject_to_idx)
    all_subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. The trajectory does not contain a separate justification for subject handling. The choice is implicit in the direct-loading design: once the AI moved away from ONE, it relied on the session metadata table as the source of subject IDs.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid`s from `bwm_release.csv`, restricted to those present in the local cache. The conversion then iterates one `eid` at a time, so the session is the unit of processing throughout the script.

ii. 
```python
bwm_eids = bwm_df['eid'].unique()
cache_eids = set(sessions_df.index.astype(str))
available_eids = [eid for eid in bwm_eids if eid in cache_eids]

for eid_idx, eid in enumerate(available_eids):
    ...
```

iii. The trajectory shows the AI counting unique `eid`s in the release CSV and confirming how many were present in cache (steps 59-61). It treated those `eid`s as the natural session boundaries.

## 1-d. How are the data split into trials?

i. Trials are taken directly from rows of the trials parquet table. After loading the full table for a session, the AI applies a boolean mask and uses the surviving rows as the trial list.

ii. 
```python
trials = pd.read_parquet(trials_file)
mask = pd.Series(True, index=trials.index)
...
valid_trials = trials[mask].copy()
align_times = valid_trials[ALIGN_TIME].values
```

iii. The trajectory does not show a separate trial-splitting argument; the AI treated the trials table as already trial-structured after it verified the columns in the parquet file (step 47).

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering happens in two stages. First, `load_trials` keeps trials with non-NaN values in several required columns, reaction time between 0.08 and 2.0 s, and nonzero choice. Second, after behavioral interpolation, the AI keeps only trials whose wheel and whisker traces cover the requested window; sessions with fewer than two fully covered trials are dropped.

ii. 
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
...
for col in NAN_EXCLUDE:
    if col in trials.columns:
        mask &= ~trials[col].isna()
...
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)
mask &= (trials['choice'] != 0)
```

```python
wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(...)
me_binned_list, me_mask = interpolate_behavior_to_bins(...)
combined_mask = wheel_mask & me_mask
```

iii. In step 63, the AI explicitly said it was matching the reference reaction-time window and no-choice exclusion. The extra NaN exclusions and coverage-based removal were justified earlier by inspecting the raw tables and revisioned files and then making direct behavior-window checks in its own code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural arrays are derived primarily from `spikes.times.npy` and `spikes.clusters.npy`. The AI also uses `clusters.channels.npy`, `channels.brainLocationIds_ccf_2017.npy`, and optionally `clusters.metrics.pqt` to map clusters to brain regions and apply cluster filtering.

ii. 
```python
spike_times_file = find_file_with_revision(probe_dir, 'spikes.times.npy')
spike_clusters_file = find_file_with_revision(probe_dir, 'spikes.clusters.npy')
clusters_channels_file = find_file_with_revision(probe_dir, 'clusters.channels.npy')
clusters_metrics_file = find_file_with_revision(probe_dir, 'clusters.metrics.pqt')
channels_brain_ids_file = find_file_with_revision(probe_dir, 'channels.brainLocationIds_ccf_2017.npy')
```

iii. The trajectory shows the AI manually inspecting those files and their shapes to understand what was available per probe (steps 49-50). The justification was to reproduce the reference decoding inputs from raw spike times and cluster assignments.

## 2-b. How is the `neural` data processed?

i. The AI merges probe data within a session, bins spikes into 20 ms bins in a `(-0.5, 1.5)` s window around stimulus onset, and stores the resulting per-trial neuron-by-time arrays. Unlike the human reference, it does not divide counts by the bin width, so the stored neural values are spike counts cast to float rather than firing rates in Hz.

ii. 
```python
spike_times, spike_clusters, cluster_brain_ids = merge_probes_data(probes_data)
...
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters,
    align_times, TIME_WINDOW, BINSIZE
)
...
neural_trial = binned_spikes[trial_global_idx]
session_neural.append(neural_trial.astype(np.float64))
```

```python
np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

iii. In step 63, the AI said it wanted to follow the reference binning setup: align to `stimOn_times`, use a 2 s window, and use 20 ms bins. There is no explicit trajectory justification for leaving the values as counts instead of converting to Hz; that omission appears to be an implementation decision rather than a stated rationale.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI’s documented and implemented decisions conflict. The file header and metadata say it includes “all spike-sorted clusters” and applies “none” as the cluster-quality filter, but `load_spike_data` defaults to `qc_threshold=1.0` and therefore keeps only clusters whose metric label is at least 1.0. It does not additionally exclude clusters mapped to Beryl `void`.

ii. 
```python
# Key processing decisions (matching reference code):
# - All spike-sorted clusters included (no quality filter, matching reference code's qc=None)
```

```python
def load_spike_data(session_path, probe_name, qc_threshold=1.0):
    ...
    if qc_threshold is not None and clusters_metrics_file is not None:
        metrics = pd.read_parquet(clusters_metrics_file)
        if 'label' in metrics.columns:
            good_mask = metrics['label'].values >= qc_threshold
            ...
            spike_times = spike_times[spike_mask]
            spike_clusters = ib.astype(np.int32)
```

```python
'cluster_quality_filter': 'none (all clusters, matching reference code)',
```

iii. Step 63 shows the AI explicitly debating whether to use all clusters (`qc=None`, matching Zhang’s code path) or only “well-isolated neurons” (`label >= 1`, matching the data paper). The justification in the trajectory is inconsistent: it first argued for all clusters from the reference code, but the final implementation kept only `label >= 1` clusters while leaving the comments and metadata claiming the opposite.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to stimulus onset, `stimOn_times`. For each trial, the AI defines `t_beg = stimOn + (-0.5)` and `t_end = stimOn + 1.5`, then bins spikes falling in that interval.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
t_beg = align_times[trial_idx] + time_window[0]
t_end = align_times[trial_idx] + time_window[1]
i_start = np.searchsorted(spike_times, t_beg, side='left')
i_end = np.searchsorted(spike_times, t_end, side='left')
```

iii. The trajectory repeatedly cites stimulus-onset alignment as a key decision copied from the reference code and methods text (step 63 and the file header).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data use 20 ms bins across a 2 s window, giving 100 time bins per trial. No additional temporal rebinning or smoothing is applied after spike counting.

ii. 
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_TIMEBINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

```python
bin_indices = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int64),
    n_bins - 1
)
```

iii. Step 63 explicitly lists 20 ms binning and 100 time bins as a reference-matching decision.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Conceptually it is derived from the choice to align trials to `stimOn_times`, but the actual values are generated from `TIME_WINDOW` and `BINSIZE` constants rather than read from a raw time-series variable.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
```

```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. The trajectory treats this input as a constructed variable implied by the stimulus-onset alignment and decoder binning scheme, not as a separate measured stream.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI constructs one constant time vector per session using `np.linspace` from `-0.48` to `1.5` seconds, then copies that same vector into every trial. It does not compute the true 20 ms bin centers used by the reference solution.

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

iii. The trajectory does not give a dedicated justification for the exact time-vector formula. The surrounding rationale was simply to make the decoder inputs match the stimulus-aligned 100-bin trial representation.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The AI intends it to share the same 100-bin trial grid as the neural data, but it uses a shifted grid running from `t_beg + 0.02` to `t_end`, while the neural counts are binned on intervals starting at `t_beg`. So the intended alignment is “same trial grid,” but the implementation is offset from the neural-bin centers.

ii. 
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

```python
bin_indices = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int64),
    n_bins - 1
)
```

iii. The trajectory justification is implicit: the AI wanted all streams aligned to stimulus onset with the same number of bins. It did not discuss the bin-center vs. bin-edge issue.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the trials table. A block boundary is inferred whenever `probabilityLeft` changes value.

ii. 
```python
prob_left = trials_df['probabilityLeft'].values
...
if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
    block_start = i
```

iii. The trajectory does not discuss this separately, but the implementation follows the common inference that `probabilityLeft` defines block identity because there is no explicit block-number column.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI scans the full session’s `probabilityLeft` sequence, resets the current block start whenever the prior changes, and assigns each trial the zero-based offset from that block start. It computes the counts on the unfiltered trial table and only applies the trial mask at the end.

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

iii. The trajectory does not state this explicitly, but the code structure shows the AI was trying to preserve the original within-block trial count before trial filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table.

ii. 
```python
choice_vals = valid_trials['choice'].values
```

iii. The trajectory shows the AI inspecting the raw choice values in the trials table to confirm the available coding (step 49).

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI drops no-choice trials earlier, then maps `choice == -1` to decoder class `0` and everything else to class `1`. That means it interprets `-1` as left and `+1` as right, which is the reverse of the human reference.

ii. 
```python
# Choice: binary, left=0, right=1
choice = 0 if choice_vals[trial_global_idx] == -1 else 1
```

iii. The trajectory does not contain a clear justification for this sign convention. The AI verified the raw values were `[-1, 1]` (step 49) but did not record a separate argument for the final remapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii. 
```python
prob_left_vals = valid_trials['probabilityLeft'].values
```

iii. The trajectory shows the AI checking the raw `probabilityLeft` values and their distribution before coding the mapping (step 49).

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and any other surviving value to `2` (implicitly `0.8`). The result is then repeated across all 100 bins of the trial.

ii. 
```python
if prob == 0.2:
    prior = 0
elif prob == 0.5:
    prior = 1
else:  # 0.8
    prior = 2
```

iii. The trajectory justification is implicit in the task specification and the AI’s inspection of the raw prior values in step 49.

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

iii. The trajectory shows the AI explicitly inspecting those wheel arrays while reverse-engineering the session contents (step 49).

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates wheel position to a uniform 1 kHz grid with `interpolate_position`, computes velocity with `velocity_filtered`, takes the absolute value to obtain speed, and then linearly interpolates that speed onto the 100 decoder time points for each trial.

ii. 
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
pos_interp, ts_interp = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(pos_interp, 1000)
speed = np.abs(velocity)
```

```python
y_interp = interp1d(local_times, local_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. The trajectory shows the AI iterating on this function after checking the ibllib helper signatures and trying to match what `SessionLoader.load_wheel` does internally (steps 76, 80, and 86).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI concatenates all interpolated wheel-speed samples from trials that pass the combined wheel/whisker coverage mask, computes the 33.33rd and 66.67th percentiles for that session, and uses `np.digitize` to assign each time bin to class 0, 1, or 2.

ii. 
```python
all_wheel_vals = []
for idx in combined_indices:
    if wheel_binned_list[idx] is not None:
        all_wheel_vals.append(wheel_binned_list[idx])
all_wheel_concat = np.concatenate(all_wheel_vals)
wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
...
wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)
```

iii. The trajectory justification is indirect: the AI described this as reference-matching percentile binning for three equal-frequency classes in its script comments and notes.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The AI aligns wheel speed to the same stimulus-locked trial window as the neural data, but it resamples onto `np.linspace(t_beg + 0.02, t_end, 100)` rather than the neural-bin centers. So the alignment is intended to be bin-for-bin, but the actual temporal grid is shifted relative to the reference.

ii. 
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. The trajectory rationale is that all decoder streams should share the same stimulus-aligned 100-bin representation. It does not mention the resulting 10 ms offset.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, with fallback to the right camera equivalents when the left camera files are absent.

ii. 
```python
me_file = find_file_with_revision(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
ts_file = find_file_with_revision(alf_dir, '_ibl_leftCamera.times.npy')

if me_file is None or ts_file is None:
    me_file = find_file_with_revision(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    ts_file = find_file_with_revision(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. The trajectory shows the AI inspecting which camera files existed in the session cache and then deciding to prefer the left camera with right-camera fallback to mirror the reference behavior.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI loads the released motion-energy trace, truncates the signal and timestamp arrays to the same length, removes NaN samples globally, and linearly interpolates the remaining values onto the 100 decoder time points for each trial. It does not apply additional filtering or normalization.

ii. 
```python
me = np.load(me_file).flatten()
times = np.load(ts_file).flatten()
min_len = min(len(me), len(times))
me = me[:min_len]
times = times[:min_len]

valid = ~np.isnan(me) & ~np.isnan(times)
return times[valid], me[valid]
```

```python
me_binned_list, me_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, TIME_WINDOW, BINSIZE
)
```

iii. The trajectory justification is mostly practical: after inspecting the raw files, the AI chose a direct load-and-interpolate approach analogous to the wheel handling, while keeping the left-camera preference from the reference pipeline.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is discretized exactly like wheel speed: all surviving per-bin values in a session are concatenated, 33.33rd and 66.67th percentile cut points are computed, and `np.digitize` maps each time point to one of three equal-frequency classes.

ii. 
```python
all_me_vals = []
for idx in combined_indices:
    if me_binned_list[idx] is not None:
        all_me_vals.append(me_binned_list[idx])
all_me_concat = np.concatenate(all_me_vals)
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
...
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)
```

iii. The AI’s written comments describe this as matching the reference percentile-based discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. As with wheel speed, the AI aligns whisker motion energy to the stimulus-locked neural window by interpolation, but uses the shifted grid `t_beg + 0.02` through `t_end` instead of the neural-bin centers.

ii. 
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. The trajectory rationale is the same as for wheel speed: every trial should be represented on a shared 100-step stimulus-aligned time axis.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or malformed data by skipping. It searches revision directories for missing canonical paths, drops sessions whose path, trials, probes, wheel data, or whisker data cannot be loaded, removes NaN whisker samples, drops trials with missing required trial-table fields or incomplete wheel/whisker windows, and skips sessions with fewer than two usable trials or fewer than five surviving clusters.

ii. 
```python
def find_file_with_revision(base_dir, filename):
    ...
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith('#') and d.name.endswith('#'):
            candidate = d / filename
            if candidate.exists():
                return candidate
```

```python
if trials is None:
    ...
if len(probes_data) == 0:
    ...
if wheel_times is None:
    wheel_mask = np.zeros(n_valid_trials, dtype=bool)
if me_times is None:
    me_mask = np.zeros(n_valid_trials, dtype=bool)
if n_final < 2:
    ...
if n_clusters < MIN_NEURONS:
    ...
```

iii. The trajectory justification was pragmatic. After discovering that some required files only existed under revision directories (steps 45-48), the AI added revision-aware loading and then chose to skip sessions/trials when required data were still missing.

## 10-a. What are the most time-consuming steps of the code?

i. The code’s dominant expensive steps are loading large spike arrays from disk, binning spikes trial-by-trial, and interpolating behavior trial-by-trial. The script itself flags spike binning as a slow step.

ii. 
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
```

```python
print(f"  Binning spikes (this may take a moment)...")
binned_spikes = bin_spikes_per_trial(...)
```

iii. The trajectory supports this indirectly: the AI inspected 22 million spikes in one example session (step 49) and later emitted progress messages around per-session spike binning during full conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain vectorizable: the per-trial spike-binning loop, the per-trial behavior interpolation loop, the per-trial block-number loop, the loop collecting all wheel/whisker values for percentile estimation, and the final loop assembling per-trial input/output arrays. The AI did partially vectorize spike counting inside each trial with `np.add.at`, but not across trials.

ii. 
```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

```python
for trial_idx in range(n_trials):
    ...
    y_interp = interp1d(local_times, local_vals, kind='linear',
                       fill_value='extrapolate')(x_interp)
```

```python
for i in range(1, len(prob_left)):
    ...
for trial_local_idx, trial_global_idx in enumerate(combined_indices):
    ...
```

iii. The trajectory shows at least one explicit micro-optimization attempt: the AI replaced an inner per-spike loop with `np.add.at` in step 95. It did not record a broader justification for leaving the remaining loops as-is.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several scans that could have been consolidated: it searches revision directories separately for every file, scans full behavior arrays trial-by-trial with a fresh boolean mask in `interpolate_behavior_to_bins`, and loops over the same filtered trial set once to collect percentile values and again to build outputs.

ii. 
```python
trials_file = find_file_with_revision(alf_dir, '_ibl_trials.table.pqt')
spike_times_file = find_file_with_revision(probe_dir, 'spikes.times.npy')
spike_clusters_file = find_file_with_revision(probe_dir, 'spikes.clusters.npy')
...
```

```python
beh_mask = (beh_times >= t_beg - binsize) & (beh_times <= t_end + binsize)
local_times = beh_times[beh_mask]
local_vals = beh_values[beh_mask]
```

```python
for idx in combined_indices:
    ...
for trial_local_idx, trial_global_idx in enumerate(combined_indices):
    ...
```

iii. The trajectory does not discuss repeated processing directly. These repetitions are visible in the final script structure.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script contains a few dead or effectively discarded computations. It constructs `input_trial = np.array([trial_num])` and never uses it, initializes `wheel_binned` and `me_binned` but never uses them, defines `discretize_to_bins` but never calls it, and parses `--n-workers` without using it. None of these affect the saved dataset.

ii. 
```python
wheel_binned = None
...
me_binned = None
```

```python
def discretize_to_bins(values, n_bins=3):
    ...
```

```python
trial_num = trial_nums_in_block[trial_global_idx]
input_trial = np.array([trial_num], dtype=np.float64)  # (1,) per-trial
input_trial_full = np.vstack([...])
```

```python
ap.add_argument('--n-workers', type=int, default=1)
```

iii. The trajectory does not justify these extra computations. They appear to be leftovers from intermediate implementation revisions.
