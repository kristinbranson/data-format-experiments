# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans the local cache under `data/one_cache/*/Subjects/*/*/001`, keeps only session directories that already have spikes, a trials parquet, wheel timestamps, and whisker motion-energy files, and then processes sessions one at a time. It does not use the BWM freeze table or ONE API session IDs from the reference code.

ii.
```python
def find_session_dirs():
    session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
    valid = []
    for sdir in session_dirs:
        has_spikes = len(glob.glob(os.path.join(sdir, 'alf/probe*/pykilosort/*/spikes.times.npy'))) > 0
        has_trials = len(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt'))) > 0
        has_wheel = os.path.exists(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy'))
        has_me = ...
        if has_spikes and has_trials and has_wheel and has_me:
            valid.append(sdir)
    return valid

session_dirs = find_session_dirs()
for i, sdir in enumerate(session_dirs):
    result = process_session(sdir, br, ...)
```

iii. In `CONVERSION_NOTES.md`, the agent justified this as using “all 393 sessions with complete data.” It explicitly resolved the paper/code/data mismatch by using all locally complete sessions rather than the paper’s 433-session cohort or the release table used by the reference code.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is parsed from the session path segment immediately after `Subjects`. The final dataset stores unique subject names in first-seen order and a `subject_idx` array mapping each processed session to that subject list.

ii.
```python
def parse_session_info(sdir):
    parts = Path(sdir).parts
    sub_idx = parts.index('Subjects')
    lab = parts[sub_idx - 1]
    subject = parts[sub_idx + 1]
    date = parts[sub_idx + 2]
    return lab, subject, date

if subject not in all_subjects:
    all_subjects.append(subject)
subject_per_session.append(subject)
subject_idx = np.array([all_subjects.index(s) for s in subject_per_session], dtype=np.int32)
```

iii. The notes describe the data layout as `data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf/`, so the agent treated the path itself as the authoritative subject split.

## 1-c. How are the data split into sessions?

i. Each directory matching `<lab>/Subjects/<subject>/<date>/001` is treated as one session. The date and subject are combined into a session identifier such as `NYU-11_2020-02-18`.

ii.
```python
session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))

def parse_session_info(sdir):
    ...
    date = parts[sub_idx + 2]
    return lab, subject, date

session_id = f"{subject}_{date}"
```

iii. The notes repeatedly describe the local cache as session-organized by `subject/date/001`, and the agent used that hierarchy instead of the reference code’s `eid`-based session selection.

## 1-d. How are the data split into trials?

i. Trials come from rows of the `_ibl_trials.table.pqt` parquet file. For each row, the agent creates a trial interval from `stimOn_times + (-0.5, 1.5)` and bins spikes and behavior in that window, then subsets to trials that pass masking.

ii.
```python
trials = load_trials(sdir)
stim_on = trials[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]

binned_spikes = bin_spikes_vectorized(
    spike_times, spike_clusters, n_clusters, interval_begs, interval_ends
)

good_indices = np.where(combined_mask)[0]
trials_good = trials.iloc[good_indices]
```

iii. The notes say the script follows the reference trial-aligned code path: “split into 2-s trials” aligned to `stimOn_times`.

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes trials with reaction time outside 0.08-2.0 s, trial length over 10 s, no choice (`choice == 0`), or NaNs in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`. It then further excludes trials lacking adequate wheel or whisker coverage after interpolation.

ii.
```python
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)

trial_len = trials['feedback_times'] - trials['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()

mask &= (trials['choice'] != 0)

for col in NAN_EXCLUDE:
    if col in trials.columns:
        mask &= ~trials[col].isna()

combined_mask = mask.values & wheel_mask & whisker_mask
```

iii. This is directly justified in the notes as matching `load_trials_and_mask`: RT bounds, no-choice exclusion, NaN exclusion, and `max_trial_len=10.0`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from spike times and cluster assignments in each probe directory, plus `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` to assign each cluster to a Beryl brain region.

ii.
```python
st_file = os.path.join(pdir, 'spikes.times.npy')
sc_file = os.path.join(pdir, 'spikes.clusters.npy')
cc_file = os.path.join(pdir, 'clusters.channels.npy')
cb_file = os.path.join(pdir, 'channels.brainLocationIds_ccf_2017.npy')

spike_times = np.load(st_file).flatten()
spike_clusters = np.load(sc_file).flatten()
cluster_channels = np.load(cc_file).flatten()
channel_brain_ids = np.load(cb_file).flatten()
```

iii. The notes state that the reference path loads spike sorting data, merges probes, and maps clusters to Beryl atlas regions. The agent recreated that from raw local files.

## 2-b. How is the `neural` data processed?

i. The agent merges all probes within a session by offsetting cluster IDs, sorts merged spikes by time, bins spike counts into 20 ms bins over 100 bins per trial, and finally stores each trial’s neural matrix as `uint8` after clipping counts to 255.

ii.
```python
spike_clusters_offset = spike_clusters + cluster_offset
cluster_offset += n_clusters
...
sort_idx = np.argsort(merged_times, kind='stable')
merged_times = merged_times[sort_idx]
merged_clusters = merged_clusters[sort_idx].astype(np.int32)
...
bin_idx = np.minimum(((times_trial - t_beg) / BINSIZE).astype(np.int32), N_BINS - 1)
flat_idx = clusters_trial * N_BINS + bin_idx
counts = np.bincount(flat_idx, minlength=n_clusters * N_BINS)
binned[trial_idx] = counts[:n_clusters * N_BINS].reshape(n_clusters, N_BINS)
...
neural_list = [np.clip(neural_trials[i], 0, 255).astype(np.uint8) for i in range(n_trials)]
```

iii. The notes justify the main processing as “bin into 20 ms bins per trial, aligned to stimOn_times” and “use all neurons.” The trajectory also mentions memory reduction as the reason for storing neural data as `uint8`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by unit quality. All clusters from all probes are kept as long as the probe has the required spike and channel files.

ii.
```python
def load_spikes(sdir):
    """Load and merge spikes from all probes in a session.

    Following reference code: no QC filtering (qc=None).
    """
    probe_dirs = sorted(glob.glob(os.path.join(sdir, 'alf/probe*/pykilosort/*')))
    ...
    return merged_times, merged_clusters, np.array(all_cluster_regions)
```

iii. The notes explicitly say, “Use ALL neurons (qc=None), matching reference code,” after noting the paper’s separate well-isolated-neuron criteria.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every neural trial is aligned to `stimOn_times` with a fixed window from -0.5 s to +1.5 s.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_on = trials[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. The notes acknowledge that the methods paper uses different alignments for different targets, but the agent chose to “follow code + decoder task” and therefore align everything to stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use a uniform 20 ms bin size, giving 100 bins over each 2 s trial. No later rebinning is applied.

ii.
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes say the agent chose “20 ms bins, matching code,” despite recording that the methods paper used 50 ms bins for choice/prior.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not computed from a per-trial raw timeseries. It is derived from the chosen alignment event (`stimOn_times`) plus the fixed global constants `TIME_WINDOW`, `BINSIZE`, and `N_BINS`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes describe this input as “Time since stimulus onset: center of each 20 ms bin,” so the agent treated it as a constructed coordinate rather than a raw recorded channel.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent creates a 100-element vector of bin centers from -0.49 s to 1.49 s and copies that same vector into every trial.

ii.
```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. The notes say this input should be “the center of each 20 ms bin,” consistent with the chosen 2 s stimulus-aligned window.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same trial window and bin count as the neural data, so each timepoint in `time_since_stim_onset` is aligned one-to-one with the corresponding neural bin.

ii.
```python
inp = np.stack([
    time_input,
    np.full(N_BINS, trial_num_in_block[i], dtype=np.float32)
], axis=0)  # (2, 100)
```

iii. The notes explicitly planned `time_since_stim_onset` as shape `(1, 100)` with the same 100 time bins used for spikes.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the per-trial `probabilityLeft` values in the trials table.

ii.
```python
prob_left = trials_good['probabilityLeft'].values
trial_num_in_block = compute_trial_num_in_block(prob_left)
```

iii. The notes define this variable as “count trials since last block change” and explicitly say it is derived “from probabilityLeft.”

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans the filtered `probabilityLeft` sequence and increments a counter while consecutive filtered trials have the same value; the counter resets to 1 whenever `probabilityLeft` changes. It then repeats that scalar across all 100 time bins for the trial.

ii.
```python
def compute_trial_num_in_block(prob_left):
    trial_nums = np.ones(len(prob_left), dtype=np.int32)
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            trial_nums[i] = trial_nums[i - 1] + 1
        else:
            trial_nums[i] = 1
    return trial_nums

inp = np.stack([
    time_input,
    np.full(N_BINS, trial_num_in_block[i], dtype=np.float32)
], axis=0)
```

iii. The notes justify this as “Block trial number: computed as position within contiguous block of same probabilityLeft.” There is no more sophisticated treatment of dropped trials or the initial unbiased block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column in the trials table.

ii.
```python
choice = trials_good['choice'].values.copy()
```

iii. The notes map `choice` directly from the trials table as a per-trial variable.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent converts `choice` into a binary label with `np.where(choice == 1, 1, 0)` and then repeats that label across all 100 time bins in the output matrix. In the notes, it describes this as mapping `-1` to left and `1` to right.

ii.
```python
# Choice: -1 (left) -> 0, 1 (right) -> 1
choice = trials_good['choice'].values.copy()
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)

out = np.stack([
    np.full(N_BINS, choice_binary[i], dtype=np.int64),
    ...
], axis=0)
```

iii. The notes say, “choice: -1(left)->0, 1(right)->1.” The trajectory shows the agent committed to that sign convention rather than checking the actual trial-table encoding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column in the trials table.

ii.
```python
prob_left = trials_good['probabilityLeft'].values
```

iii. The notes explicitly identify “Prior = probabilityLeft.”

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent maps `probabilityLeft` to a 3-class categorical variable using approximate matching: values near 0.2 map to 0, 0.5 to 1, and 0.8 to 2. The resulting class is repeated across all time bins for the trial.

ii.
```python
prior = np.full(len(prob_left), 1, dtype=np.int32)
prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
```

iii. The notes justify this directly from the decoder task: “Prior = probabilityLeft mapped to categories: 0.2->0, 0.5->1, 0.8->2.”

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
wh_pos = np.load(os.path.join(sdir, 'alf/_ibl_wheel.position.npy')).flatten()
wh_times = np.load(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy')).flatten()
```

iii. The notes say wheel speed should follow the reference behavior loader, which in the reference code comes from session wheel times and velocity.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent interpolates wheel position to a uniform 1 kHz grid with `np.interp`, computes velocity with `np.gradient`, takes its absolute value, then linearly interpolates that continuous speed trace into each 20 ms trial bin.

ii.
```python
dt = 0.001
t_uniform = np.arange(wh_times[0], wh_times[-1], dt)
pos_interp = np.interp(t_uniform, wh_times, wh_pos)
velocity = np.gradient(pos_interp, dt)
speed = np.abs(velocity)
...
wheel_vals, wheel_mask = interpolate_behavior_to_bins(
    wh_times, wh_speed, interval_begs, interval_ends
)
```

iii. The notes claim this is intended to replicate `SessionLoader.load_wheel()` behavior. The code comments specifically say the reference uses interpolated wheel traces and Gaussian-smoothed velocity, and the agent attempted a local approximation from raw files.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. The continuous wheel-speed values are discretized into 3 bins using quantile boundaries computed from all non-NaN wheel values in that session.

ii.
```python
def discretize_to_bins(values, n_bins=3):
    flat = values[~np.isnan(values)].flatten()
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    boundaries = np.percentile(flat, quantiles)
    result = np.digitize(values, boundaries).astype(np.int32)
    return result

wheel_discrete = discretize_to_bins(wheel_trials, n_bins=3)
```

iii. The notes explicitly state, “Discretization: Use session-wide terciles for wheel speed and whisker ME,” because the decoder requires categorical outputs while the reference code treats these as continuous.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset trial windows used for neural data: `stimOn_times + (-0.5, 1.5)` with 100 interpolated bins.

ii.
```python
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
wheel_vals, wheel_mask = interpolate_behavior_to_bins(
    wh_times, wh_speed, interval_begs, interval_ends
)
```

iii. The notes acknowledge that the paper’s dynamic-behavior alignment differs, but say the agent chose to “follow code + decoder task” and use stimulus onset for all outputs.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from either `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, or, if the left view is unavailable, the corresponding right-camera files.

ii.
```python
left_me_files, left_time_files = _find_files(
    sdir, 'leftCamera.ROIMotionEnergy.npy', '_ibl_leftCamera.times.npy')
...
right_me_files, right_time_files = _find_files(
    sdir, 'rightCamera.ROIMotionEnergy.npy', '_ibl_rightCamera.times.npy')
```

iii. The notes justify this as matching the reference code’s “Try left camera first, fall back to right.”

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent loads the most recent left-camera motion-energy trace if available, otherwise the right-camera trace, truncates times and values to the shorter length if they disagree, and then linearly interpolates the signal into the per-trial bins.

ii.
```python
if left_me_files and left_time_files:
    me = np.load(left_me_files[-1]).flatten()
    times = np.load(left_time_files[-1]).flatten()
    min_len = min(len(me), len(times))
    return times[:min_len], me[:min_len]
...
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
```

iii. The notes describe this as matching the reference behavior loader for whisker motion energy, with a left-camera preference and interpolation to the trial bins.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is discretized into 3 categories using session-wide quantile boundaries over all interpolated whisker-motion-energy values.

ii.
```python
whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
```

iii. The notes use the same justification as wheel speed: the reference variable is continuous, so the agent chose session-wide terciles to satisfy the decoder’s categorical-output requirement.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is interpolated into the same stimulus-onset-aligned 2 s windows and 20 ms bins as the neural data.

ii.
```python
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
```

iii. The notes again say the agent followed the reference code path plus the decoder instruction to align everything to stimulus onset.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or incomplete sessions are skipped up front by `find_session_dirs()`. Within a session, trials with missing key events are excluded by the trial mask; wheel or whisker intervals with no samples or inadequate coverage are masked out; mismatched whisker time/value lengths are truncated to the shorter array; and any session left with fewer than 2 valid trials is skipped entirely.

ii.
```python
if has_spikes and has_trials and has_wheel and has_me:
    valid.append(sdir)
...
if len(beh_v) == 0:
    mask[trial_idx] = False
...
if np.abs(t_beg - beh_t[0]) > BINSIZE:
    mask[trial_idx] = False
...
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len]
...
if len(good_indices) < 2:
    return None
```

iii. The notes and comments justify this as “graceful handling of missing data” and session skipping when wheel or whisker data are unavailable.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant costs are session-by-session spike binning over all trials and neurons, plus per-trial behavioral interpolation for wheel speed and whisker motion energy. The code’s own progress messages and comments point to these as the expensive parts.

ii.
```python
print(f"  Binning spikes ({n_clusters} neurons, {len(trials)} trials)...", flush=True)
binned_spikes = bin_spikes_vectorized(...)
print(f"  Spike binning: {time.time() - t_bin:.1f}s", flush=True)

wheel_vals, wheel_mask = interpolate_behavior_to_bins(...)
whisker_vals, whisker_mask = interpolate_behavior_to_bins(...)
```

iii. The trajectory and comments focus optimization effort on spike binning, memory monitoring, and behavior interpolation, which indicates the agent saw those as the main runtime bottlenecks.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit Python loops over trials in `bin_spikes_vectorized()`, `interpolate_behavior_to_bins()`, `compute_trial_num_in_block()`, and the per-trial loops that build `neural_list`, `input_list`, and `output_list` could all be pushed further toward vectorized array operations.

ii.
```python
for trial_idx in range(n_trials):
    ...

for trial_idx in range(n_trials):
    ...

for i in range(1, len(prob_left)):
    ...

for i in range(n_trials):
    input_list.append(inp)

for i in range(n_trials):
    output_list.append(out)
```

iii. The agent justified some of this as a pragmatic compromise: comments mention “searchsorted for fast trial assignment” and “incremental building to save memory,” suggesting it prioritized a simple memory-safe implementation over deeper vectorization.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly searches the filesystem with `glob`, repeatedly loops over every trial for each session when binning spikes and interpolating behaviors, and repeats per-trial broadcasting when constructing inputs and outputs. It also computes similar discretization logic separately for wheel and whisker signals.

ii.
```python
session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
probe_dirs = sorted(glob.glob(os.path.join(sdir, 'alf/probe*/pykilosort/*')))
trial_files = sorted(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt')))
...
wheel_vals, wheel_mask = interpolate_behavior_to_bins(...)
whisker_vals, whisker_mask = interpolate_behavior_to_bins(...)
...
wheel_discrete = discretize_to_bins(wheel_trials, n_bins=3)
whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
```

iii. There is no explicit separate justification in the notes beyond keeping the implementation straightforward and local-file-based.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins spikes and behavior for all trials before masking out bad trials, computes and stores session bookkeeping fields like `session_id`, `n_trials`, and `n_neurons` that are discarded when the final dataset dict is built, and optionally generates processing plots that are unrelated to downstream decoder training. It also maps `lab` in `parse_session_info()` but never uses it downstream.

ii.
```python
# 4. Bin spikes for ALL trials first (before masking)
binned_spikes = bin_spikes_vectorized(...)
...
good_indices = np.where(combined_mask)[0]
neural_trials = binned_spikes[good_indices]
...
return {
    'neural': neural_list,
    'input': input_list,
    'output': output_list,
    'subject': subject,
    'cluster_regions': cluster_regions,
    'n_trials': n_trials,
    'n_neurons': n_clusters,
    'session_id': session_id,
}
```

iii. The agent comments justify some of this as a consequence of “following reference code align_spike_behavior” and adding diagnostics like plots and memory summaries during development, even though those extra computations are not needed by the final decoder input.
