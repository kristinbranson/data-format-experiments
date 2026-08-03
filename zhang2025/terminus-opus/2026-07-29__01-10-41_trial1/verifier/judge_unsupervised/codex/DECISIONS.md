# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent starts from `code/code_zhang2025/data/bwm_release.csv`, groups rows by `eid`, and then tries to find each session on disk under `data/one_cache/{lab}/Subjects/{subject}/{date}/001`. For each found session it loads one trials parquet, all listed probes' spike files, wheel files, and whisker motion-energy files. Sessions are only kept if they can be found locally and end with at least 2 valid trials.

ii. 
```python
bwm_df = pd.read_csv('code/code_zhang2025/data/bwm_release.csv', index_col=0)
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first', 'subject': 'first', 'date': 'first',
    'probe_name': list, 'pid': list
}).reset_index()

def find_session_dir(lab, subject, date, base_dir='data/one_cache'):
    session_dir = os.path.join(base_dir, lab, 'Subjects', subject, date, '001')

trials_df = load_trials(session_dir)
spike_times, spike_clusters, cluster_regions = load_spikes(session_dir, probe_names)
wheel_times, wheel_speed = load_wheel_speed(session_dir)
me_times, me_values = load_whisker_motion_energy(session_dir)
```

iii. In `CONVERSION_NOTES.md`, the agent says it would process “all sessions in `bwm_release`” and handle missing behavior by excluding trials. In the trajectory, it justified this as following `0_data_caching.py` at the session level, but adapted to the local cache instead of using `ONE`/`SessionLoader`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the `subject` column in `bwm_release.csv`. After each session is processed, the code builds `subjects` and `subject_idx` by assigning each unique subject string an integer ID in first-seen order.

ii. 
```python
subject_map = {}
all_subjects = []

subj = result['subject']
if subj not in subject_map:
    subject_map[subj] = len(all_subjects)
    all_subjects.append(subj)
subject_idx_list.append(subject_map[subj])
```

iii. The notes describe “subjects: 139” in the reference material and treat the mouse identity as the `subject` field from the release table. No alternative subject-splitting logic appears in the trajectory.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique `eid` values from `bwm_release.csv`, with per-session probe lists aggregated before processing. The on-disk lookup then assumes that each session lives at `{lab}/Subjects/{subject}/{date}/001`, so a session is effectively “the unique `eid` whose data are found in that hard-coded local path.”

ii. 
```python
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first',
    'subject': 'first',
    'date': 'first',
    'probe_name': list,
    'pid': list
}).reset_index()

session_dir = find_session_dir(lab, subject, date)
if session_dir is None:
    print(f"  Skipping: session directory not found")
```

iii. In the notes, the agent explicitly says it will “process all sessions with spikes+trials” from `bwm_release`. The trajectory shows it switched from the reference code’s `ONE`-based `eid` loading to direct cache-path loading for practicality.

## 1-d. How are the data split into trials?

i. Trials are the rows of the loaded trials table. The code first computes a stimulus-aligned interval for every row, bins spikes and interpolates behavior for every row, then selects the subset indexed by `valid_idx`.

ii. 
```python
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]

binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)

valid_idx = np.where(combined_mask)[0]
valid_trials = trials_df.iloc[valid_idx]
```

iii. The notes say neural and behavior data are trial-aligned to `stimOn_times` with a 2 s window, matching the `0_data_caching.py` parameters. The trajectory excerpts from `bin_spiking_data` show the reference code also treats trials as rows of `trials_df`.

## 1-e. How are trials filtered based on quality controls?

i. The code applies a trial mask for reaction time, max trial length, NaNs in key trial columns, and no-choice trials. It then adds a second filter requiring wheel and whisker data to be successfully interpolated for that trial.

ii. 
```python
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)
trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN)
for event in nan_exclude:
    mask &= ~trials_df[event].isna()
mask &= (trials_df['choice'] != 0)

combined_mask = mask.values & wheel_valid & me_valid
```

iii. `CONVERSION_NOTES.md` Step 5 lists `min_rt=0.08`, `max_rt=2.0`, `max_trial_len=10.0`, NaN exclusion, and no-choice exclusion as deliberate copies of `load_trials_and_mask`. The same notes also say missing wheel or motion-energy data would cause trial exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw spike times and cluster IDs from each probe’s Kilosort output. Region labels are derived separately from `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy`.

ii. 
```python
spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
clusters_channels = np.load(os.path.join(ks_dir, 'clusters.channels.npy')).flatten()
brain_ids = np.load(brain_id_files[0]).flatten()
cluster_brain_ids = brain_ids[clusters_channels]
```

iii. The trajectory shows the reference helper `prepare_data()` builds `neural_dict` from `spikes['times']`, `spikes['clusters']`, and `clusters['acronym']`. The notes summarize this as “Neuropixels recordings, all clusters, Beryl mapping.”

## 2-b. How is the `neural` data processed?

i. Probe spike streams are merged by offsetting cluster IDs, concatenating spikes, and sorting by time. Then spikes are counted into 20 ms bins for each trial’s stimulus-aligned 2 s window, yielding `(n_trials, n_clusters, 100)` before conversion to per-trial `(n_neurons, n_timepoints)` arrays.

ii. 
```python
spike_clusters = spike_clusters + cluster_offset
merged_times = np.concatenate(all_spike_times)
merged_clusters = np.concatenate(all_spike_clusters)
sort_idx = np.argsort(merged_times, kind='stable')

time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
```

iii. The notes say this is intended to match `merge_probes`, `bin_spiking_data`, and `get_spike_data_per_interval`. The trajectory confirms the reference code merges probe clusters and bins spike counts at `binsize=0.02` over `time_window=(-.5, 1.5)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not apply neuron/unit QC filtering. It keeps all clusters present in the spike-sorting outputs and only drops whole sessions if spike files are missing.

ii. 
```python
spike_times, spike_clusters, cluster_regions = load_spikes(session_dir, probe_names)
if spike_times is None:
    print(f"  Skipping {eid}: no spike data")
    return None
```

iii. The notes repeatedly justify this with “No QC filtering on clusters (qc=None in `load_spiking_data`)”. The trajectory excerpt from `load_spiking_data` shows the reference code returns all clusters when `qc is None`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial is aligned to `stimOn_times`, with a window from `-0.5` s to `+1.5` s relative to that event.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
```

iii. The notes explicitly call out a “Key Decision: Alignment” and say the task description and `0_data_caching.py` both point to `stimOn_times`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 20 ms bins. No secondary rebinning is applied after spike counting and behavior interpolation.

ii. 
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. The notes say “20ms bins, 100 time steps,” citing both the reference code parameters and the methods text.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a dedicated raw variable. It is derived from the alignment event choice (`stimOn_times`) plus the fixed `TIME_WINDOW`/`BINSIZE` constants.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
```

iii. In Step 5 of the notes, the agent records this mapping as “time since stimOn -> `np.arange(-0.5, 1.5, 0.02) + 0.01`.”

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code creates a single vector of bin-center times from `-0.49` to `1.49` s and copies it into every trial.

ii. 
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
time_since_stim = time_since_stim.astype(np.float32)
```

iii. The notes justify this as the natural time-varying decoder input implied by the stimulus-aligned 20 ms bins.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same 100 stimulus-aligned bin centers as the neural matrices, and is stacked into each trial’s `input` array with identical length.

ii. 
```python
inp = np.vstack([
    time_since_stim[np.newaxis, :],
    np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)
])
```

iii. The notes describe this input as “time-varying, same for all trials,” specifically to match the neural binning.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table’s `probabilityLeft` column.

ii. 
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
```

iii. In Step 5 of the notes, the agent states that block boundaries are inferred from `probabilityLeft` changes.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code walks through all trials in order, increments a counter within a block, and resets the counter to 1 whenever `probabilityLeft` changes. It computes this on all trials before subselecting valid trials.

ii. 
```python
def compute_trial_number_in_block(prob_left):
    counter = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            counter = 1
        trial_nums[i] = counter
        counter += 1
```

iii. The notes justify this as “Count trials within each block, reset when `probabilityLeft` changes.”

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from the trials table’s `choice` column.

ii. 
```python
choice = valid_trials['choice'].values.copy()
```

iii. The notes and trajectory both identify `trials_df['choice']` as the source variable used in the reference code for trial-level behavior.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code maps IBL’s `choice==-1` to decoder class `0` (left) and everything else remaining after no-choice filtering to `1` (right). It then repeats that scalar across all 100 bins for each trial.

ii. 
```python
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
np.full((1, N_BINS), choice_mapped[i], dtype=np.int64)
```

iii. Step 5 of the notes explicitly says “IBL choice -1 = left = 0, IBL choice 1 = right = 1.”

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trials table’s `probabilityLeft` column.

ii. 
```python
prob_left = valid_trials['probabilityLeft'].values
```

iii. The notes cite `probabilityLeft` directly and tie it to the task’s required 3-class prior output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, defaulting any unexpected value to `1`. It then repeats the class across all bins of the trial.

ii. 
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
np.full((1, N_BINS), prior_mapped[i], dtype=np.int64)
```

iii. The notes justify the `0.2/0.5/0.8` mapping from the task specification and the papers’ block structure.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
pos_file = os.path.join(session_dir, 'alf', '_ibl_wheel.position.npy')
ts_file = os.path.join(session_dir, 'alf', '_ibl_wheel.timestamps.npy')
wheel_pos = np.load(pos_file).flatten()
wheel_ts = np.load(ts_file).flatten()
```

iii. The notes say wheel speed is loaded from raw wheel position and timestamps to match the wheel-processing utilities in `brainbox`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code linearly interpolates wheel position to 1000 Hz, low-pass filters it with an 8th-order Butterworth filter at 20 Hz, differentiates to velocity, takes the absolute value to get speed, then linearly interpolates that speed trace onto each trial’s 20 ms bin centers.

ii. 
```python
t_interp = np.arange(wheel_ts[0], wheel_ts[-1], 1.0 / WHEEL_FS)
pos_interp = interpolate.interp1d(wheel_ts, wheel_pos, kind='linear')(t_interp)
sos = signal.butter(N=WHEEL_FILTER_ORDER, Wn=WHEEL_CORNER_FREQ / WHEEL_FS * 2,
                    btype='lowpass', output='sos')
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * WHEEL_FS
speed = np.abs(vel)
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, trial_starts, trial_ends)
```

iii. The notes explicitly cite `brainbox wheel.py`, and the trajectory shows the agent read `interpolate_position()` and `velocity_filtered()` before implementing this.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins by computing session-wide quantile boundaries across all non-NaN valid trial/bin values, then using `np.digitize`.

ii. 
```python
flat = values[~np.isnan(values)].flatten()
quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
boundaries = np.quantile(flat, quantiles)
result = np.digitize(values, boundaries).astype(np.int64)
```

iii. The notes justify this as a decoder-task adaptation: the reference code keeps wheel speed continuous, so the agent chose equal-frequency discretization because the task required 3 categories.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is resampled onto the same stimulus-aligned trial windows and the same 100 bin centers used for the neural data.

ii. 
```python
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, trial_starts, trial_ends)
```

iii. The notes say wheel/whisker signals are “interpolated to 20ms bins within trial windows,” following `get_behavior_per_interval`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` or `rightCamera.ROIMotionEnergy.npy`, together with the matching `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii. 
```python
me_file = find_file(session_dir, 'leftCamera.ROIMotionEnergy.npy')
cam_file = find_file(session_dir, '_ibl_leftCamera.times.npy')
...
me_file = find_file(session_dir, 'rightCamera.ROIMotionEnergy.npy')
cam_file = find_file(session_dir, '_ibl_rightCamera.times.npy')
```

iii. The notes state “Try left camera first, then right camera,” matching the trajectory excerpt from `bin_behaviors`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The chosen camera’s raw motion-energy trace is loaded and then linearly interpolated onto each trial’s 20 ms bin centers within the stimulus-aligned interval.

ii. 
```python
if me_file and cam_file:
    me = np.load(me_file).flatten()
    cam_times = np.load(cam_file).flatten()

me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```

iii. The notes justify this as matching the reference helper `load_target_behavior()` for whisker motion energy plus `get_behavior_per_interval()`.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same per-session quantile discretization as wheel speed: 3 equal-frequency bins over all valid trial/bin values in that session.

ii. 
```python
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
```

iii. The notes state this was the same discretization choice used for wheel speed because the task required categorical outputs.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is interpolated to the same 100 stimulus-aligned bin centers used for the neural matrices.

ii. 
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
```

iii. The notes explicitly say wheel and whisker signals are aligned to the common stimulus-onset trial window.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing trials or spike files cause an entire session to be skipped. Missing wheel/whisker files or interpolation failures mark all affected trials invalid. Length mismatches, too-short wheel traces, and intervals with fewer than 2 behavior samples also invalidate behavior for that trial/session. Missing region IDs fall back to `'unknown'`, and unexpected prior values silently map to class `1`.

ii. 
```python
if not trial_files:
    return None
...
if not all_spike_times:
    return None, None, None
...
if len(wheel_pos) != len(wheel_ts):
    return None, None
if len(wheel_pos) < 10:
    return None, None
...
if len(beh_v) < 2:
    valid[trial_idx] = False
...
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

iii. Step 5 of the notes says “Skip sessions missing spikes or trials; for missing wheel/ME, exclude those trials.” The trajectory shows the agent consciously chose this instead of reproducing the reference code’s `allow_nans=True` behavior.

## 10-a. What are the most time-consuming steps of the code?

i. The main bottleneck is spike binning across all trials and all clusters in each session. The next heaviest repeated work is behavior interpolation trial-by-trial after wheel-speed preprocessing.

ii. 
```python
print(f"  Binning spikes ({n_clusters} clusters, {len(trials_df)} trials)...", end=' ')
binned_spikes = bin_spikes_per_trial(...)

wheel_binned, wheel_valid = interpolate_behavior_to_bins(...)
me_binned, me_valid = interpolate_behavior_to_bins(...)
```

iii. The notes say the code was instrumented with timing information to identify bottlenecks, and `conversion_full_out.txt` shows per-session times dominated by the spike-binning stage.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest remaining candidates are the loop over valid trials in `bin_spikes_per_trial`, the loop over all trials in `interpolate_behavior_to_bins`, the per-trial construction of `input_list` and `output_list`, and the per-neuron loop that builds `brain_region_idx`.

ii. 
```python
for trial_idx in valid_indices:
    ...

for trial_idx in range(n_trials):
    ...

for i in range(n_valid_trials):
    input_list.append(inp)
...
for i, r in enumerate(regions):
    ...
```

iii. The trajectory and notes describe the spike binning as “optimized,” but the implementation still leaves these Python loops in place.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly searches the filesystem with `glob`/`find_file`, recomputes per-trial interpolations separately for wheel and whisker, and copies identical trial-invariant values (`time_since_stim`, repeated choice, repeated prior) into every trial array.

ii. 
```python
trial_files = glob.glob(...)
spike_files = glob.glob(...)
matches = glob.glob(os.path.join(session_dir, 'alf', '*', filename))

wheel_binned, wheel_valid = interpolate_behavior_to_bins(...)
me_binned, me_valid = interpolate_behavior_to_bins(...)

np.full((1, N_BINS), choice_mapped[i], dtype=np.int64)
np.full((1, N_BINS), prior_mapped[i], dtype=np.int64)
```

iii. The notes do not flag these repetitions; they mainly justify them as straightforward implementation choices.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins spikes and interpolates behavior for every trial before filtering, then throws away invalid trials. It also computes and stores fully repeated 100-bin representations of per-trial scalar targets/inputs, and optionally generates processing plots that are not used by downstream decoder training.

ii. 
```python
binned_spikes = bin_spikes_per_trial(...)
wheel_binned, wheel_valid = interpolate_behavior_to_bins(...)
me_binned, me_valid = interpolate_behavior_to_bins(...)
combined_mask = mask.values & wheel_valid & me_valid
valid_idx = np.where(combined_mask)[0]

np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)
np.full((1, N_BINS), choice_mapped[i], dtype=np.int64)
np.full((1, N_BINS), prior_mapped[i], dtype=np.int64)
```

iii. The notes frame this as acceptable because the target format allows time-varying arrays, but they do not argue that this extra work is necessary for the downstream decoder.
