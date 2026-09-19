# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the ONE API, `SessionLoader`, or `SpikeSortingLoader` to enumerate and load the release. Instead, it reads `code/code_zhang2025/data/bwm_release.csv`, groups rows by `eid`, finds matching cache directories under `data/one_cache`, and opens parquet and `.npy` files directly from disk. Trials, wheel, whisker motion energy, and probe spike files are all loaded by custom path logic.

ii. 
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)

sessions = {}
for _, row in bwm_df.iterrows():
    eid = row.eid
    if eid not in sessions:
        sessions[eid] = []
```

```python
def find_session_path(lab, subject, date):
    for sess_num in ['001', '002', '003']:
        p = BASE_PATH / lab / 'Subjects' / subject / date / sess_num / 'alf'
        if p.exists():
            return p
```

iii. The explicit justification is in `CONVERSION_NOTES.md`: Step 6 says the script uses "Direct file loading (no SpikeSortingLoader dependency)", and Step 10 claims that this "matches SpikeSortingLoader output."

## 1-b. How are the data split into subjects?

i. Sessions are assigned to subjects using the `subject` column from `bwm_release.csv`. In the final dataset, unique subject names are sorted, and each session stores an integer index into that sorted subject list.

ii. 
```python
sessions[eid].append({
    'pid': row.pid,
    'probe_name': row.probe_name,
    'subject': row.subject,
    'lab': row.lab,
    'date': row.date,
})
```

```python
all_subjects = sorted(list(set(sess['subject'] for sess in session_results)))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
subject_idx_list.append(subject_to_idx[sess['subject']])
```

iii. The justification is implicit: the notes treat the release CSV as the authoritative session index and subject source.

## 1-c. How are the data split into sessions?

i. The agent treats each unique `eid` in `bwm_release.csv` as one session, collects all probe rows for that `eid`, and processes each session once.

ii. 
```python
for _, row in bwm_df.iterrows():
    eid = row.eid
    if eid not in sessions:
        sessions[eid] = []
    sessions[eid].append({
        'pid': row.pid,
        'probe_name': row.probe_name,
        'subject': row.subject,
        'lab': row.lab,
        'date': row.date,
    })
```

iii. The notes describe the release CSV as the session index and report session counts from that file.

## 1-d. How are the data split into trials?

i. Trials are taken as rows of `_ibl_trials.table.pqt`. After loading the table, the code filters rows with a boolean mask and then uses the surviving rows as the session's trials.

ii. 
```python
trials_df = pd.read_parquet(trials_file)
...
valid_trials = trials_df[trials_mask].copy()
```

iii. The notes repeatedly describe the trials table as the per-trial source and the trial mask as the curation step.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies several filters in `load_trials_and_mask`: reaction time between 0.08 s and 2.0 s, optional trial-length filter `feedback_times - goCue_times <= 10 s`, NaN exclusion for `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`, and exclusion of `choice == 0`. It then drops additional trials if wheel or whisker traces do not cover the full window.

ii. 
```python
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)
...
trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN)
...
for event in NAN_EXCLUDE:
    if event in trials_df.columns:
        mask &= ~trials_df[event].isna()
...
mask &= (trials_df['choice'] != 0)
```

```python
combined_valid = wheel_valid & me_valid
...
valid_trials_final = valid_trials[combined_valid].copy()
```

iii. The justification is explicit in the notes: Step 1 and Step 6 say the code uses trial filtering "matching reference code defaults", and Step 10 repeats that claim.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural arrays are built from per-probe `spikes.times.npy` and `spikes.clusters.npy`. Additional files, `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`, are loaded to assign each cluster a brain region.

ii. 
```python
spike_times_file = spike_dir / 'spikes.times.npy'
spike_clusters_file = spike_dir / 'spikes.clusters.npy'
...
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
```

```python
cluster_channels = np.load(spike_dir / 'clusters.channels.npy').flatten()
chan_brain_ids = np.load(spike_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
cluster_brain_ids = chan_brain_ids[cluster_channels]
```

iii. The notes say the agent wanted "Direct file loading" while still using Beryl region mapping via `iblatlas`.

## 2-b. How is the `neural` data processed?

i. Spikes from all probes in a session are merged by offsetting cluster IDs probe-by-probe, concatenating, and sorting by spike time. The merged spikes are then binned into 20 ms bins over the `[-0.5, 1.5]` window around stimulus onset. The output stored in `neural` is spike counts per bin, not firing rate.

ii. 
```python
for st, sc, nc, cbi in probe_data:
    all_times.append(st)
    all_clusters.append(sc + cluster_offset)
    ...
spike_times = np.concatenate(all_times)
spike_clusters = np.concatenate(all_clusters)
...
sort_idx = np.argsort(spike_times, kind='stable')
```

```python
bin_idx = np.minimum(np.floor((trial_times - t_start) / binsize).astype(int), n_bins - 1)
np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
```

iii. The notes justify this as "Spike times + clusters | neural | Bin at 20ms" and the README describes the resulting arrays as "spike counts in 20ms bins."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not apply neural QC filtering. All clusters present in the probe files are kept, and there is no `label >= 1` filter and no exclusion of Beryl `void` clusters.

ii. 
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
...
n_clusters = len(cluster_channels)
cluster_brain_ids = chan_brain_ids[cluster_channels]
return spike_times, spike_clusters, n_clusters, cluster_brain_ids
```

iii. The justification is explicit and repeated: the notes say "No QC filtering (all clusters used)" because the reference code calls `load_spiking_data(..., qc=None)`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial uses a fixed window from 0.5 s before to 1.5 s after `stimOn_times`. Spike binning uses absolute spike times inside `[stim_on - 0.5, stim_on + 1.5)`, so the bin grid is aligned to stimulus onset.

ii. 
```python
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

```python
idx_start = np.searchsorted(spike_times, t_start, side='left')
idx_end = np.searchsorted(spike_times, t_end, side='left')
trial_times = spike_times[idx_start:idx_end]
```

iii. The notes explicitly say alignment is `stimOn_times + (-0.5, 1.5)` and that this matches the reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins over a 2 s trial window, for 100 bins per trial. No additional temporal rebinning is applied after this spike binning.

ii. 
```python
BINSIZE = 0.02
TIME_WINDOW = (-0.5, 1.5)
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes cite the Zhang reference parameters and repeatedly state "20ms bins" and "100 bins."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from `stimOn_times` as the alignment event plus the fixed decoding window. The actual per-trial input values are not read from raw data; they are generated from the chosen bin grid.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
...
stim_on = valid_trials[ALIGN_TIME].values
```

```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes justify this as the requested decoder input and describe it as the time vector associated with the aligned bins.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent creates a single 100-sample time vector with `np.linspace(-0.48, 1.5, 100)` and reuses it for every trial. This is intended to represent the trial's aligned time axis.

ii. 
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes explicitly describe this as `linspace(-0.48, 1.5, 100)` and later say the observed range `[-0.48, 1.5]` was "as expected."

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The same time vector is paired with the same 100 bins used for spike counts, and it is broadcast unchanged to every trial. In the agent's implementation, this grid is also the target grid for behavioral interpolation.

ii. 
```python
inp = np.stack([
    sess['time_since_stim'],
    np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32),
], axis=0)
```

```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
result[trial_idx] = f_interp(x_interp)
```

iii. The justification is implicit in comments such as "Matches the bin centers used in reference code" and in the shared reuse of the same 100-point grid across inputs and outputs.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft`. A block is defined as a consecutive run of trials with the same `probabilityLeft` value.

ii. 
```python
def compute_trial_number_in_block(prob_left):
    """Compute trial number within each block.
    
    A block is a sequence of consecutive trials with the same probabilityLeft.
```

iii. The notes explicitly say "Count trials since last block change."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans through `probabilityLeft` values after trial filtering, increments a counter within each consecutive block, resets when `probabilityLeft` changes, and stores the count starting at 1 for the first surviving trial of a block. The value is then broadcast across all 100 time bins of that trial.

ii. 
```python
current_block = prob_left.iloc[0] if hasattr(prob_left, 'iloc') else prob_left[0]
count = 0
for i in range(len(prob_left)):
    val = prob_left.iloc[i] if hasattr(prob_left, 'iloc') else prob_left[i]
    if val == current_block:
        count += 1
    else:
        current_block = val
        count = 1
    trial_nums[i] = count
```

```python
np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32)
```

iii. The notes justify this only at a high level: "trial number within block" and "Count within block, broadcast to time." There is no explicit justification for starting at 1 or for computing it after filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table.

ii. 
```python
choice = valid_trials_final['choice'].values.copy()
```

iii. The notes and README both describe `choice` as a trial-level variable taken from the trials table.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After removing `choice == 0` trials, the agent maps `choice == -1` to class 0 and every remaining value to class 1, then broadcasts that class across all 100 bins of the trial.

ii. 
```python
choice = valid_trials_final['choice'].values.copy()
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
```

```python
np.full(N_BINS, int(sess["choice_binary"][trial_idx]), dtype=np.int64)
```

iii. The justification is explicit in the notes: Step 5 says "Choice | left(-1)->0, right(1)->1," showing the agent's assumed sign convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii. 
```python
prob_left = valid_trials_final['probabilityLeft'].values.copy()
```

iii. The notes describe this as the block prior / prior probability of left.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent maps `probabilityLeft` values `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then broadcasts the resulting category across all 100 bins of the trial.

ii. 
```python
prior_cat = np.zeros(len(prob_left), dtype=np.float32)
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
```

iii. The justification is explicit in the notes and follows the decoder task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
pos_file = sess_path / '_ibl_wheel.position.npy'
ts_file = sess_path / '_ibl_wheel.timestamps.npy'
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The notes explicitly say wheel processing starts from wheel position and timestamps and is intended to match `SessionLoader.load_wheel()`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent linearly interpolates wheel position onto a 1000 Hz uniform time grid, applies an 8th-order 20 Hz low-pass Butterworth filter, differentiates the filtered position to get velocity, takes absolute value to get speed, and then linearly interpolates that speed trace onto the 100 trial bins.

ii. 
```python
t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
pos_interp = interp1d(timestamps, position, kind='linear')(t_uniform)
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
```

```python
wheel_binned, wheel_valid = interpolate_behavior(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. The justification is explicit: the notes repeatedly say this is meant to match `interpolate_position + velocity_filtered + abs()`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After all sessions are processed, the agent pools every wheel-speed sample from every session, computes the 1/3 and 2/3 global quantiles, and discretizes each wheel-speed time point into `low`, `medium`, or `high` using those global thresholds.

ii. 
```python
all_wheel = []
for sess in session_results:
    all_wheel.append(sess['wheel_binned'].flatten())
...
wheel_quantiles = np.array([-np.inf,
                             np.quantile(all_wheel, 1/3),
                             np.quantile(all_wheel, 2/3),
                             np.inf])
```

```python
wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1]).astype(np.float32)
wheel_disc = np.clip(wheel_disc, 0, 2)
```

iii. The justification is explicit in the notes and README: they say wheel speed is discretized with "global quantile-based discretization" or "global quantiles."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel-speed trace is interpolated separately for each trial onto the same 100-point aligned grid used elsewhere in the session, and trials whose wheel trace does not cover the full window are dropped.

ii. 
```python
idx_start = np.searchsorted(beh_times, t_start, side='right')
idx_end = np.searchsorted(beh_times, t_end, side='left')
...
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
result[trial_idx] = f_interp(x_interp)
```

iii. The justification is again "matching reference code" behavior interpolation and full-window coverage checks.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`, falling back to `rightCamera.ROIMotionEnergy.npy` with `_ibl_rightCamera.times.npy` if left-camera data are missing.

ii. 
```python
me_file = find_versioned_file(sess_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_versioned_file(sess_path, '_ibl_leftCamera.times.npy')
if me_file is None or times_file is None:
    me_file = find_versioned_file(sess_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_versioned_file(sess_path, '_ibl_rightCamera.times.npy')
```

iii. The notes explicitly say whisker motion energy is loaded from the left camera with right-camera fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is loaded directly, truncated so values and timestamps have equal length, stripped of NaNs, and then linearly interpolated trial-by-trial onto the same 100 aligned time points used for wheel speed.

ii. 
```python
me_values = np.load(me_file).flatten()
me_times = np.load(times_file).flatten()
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]
valid = ~np.isnan(me_values) & ~np.isnan(me_times)
me_values = me_values[valid]
me_times = me_times[valid]
```

```python
me_binned, me_valid = interpolate_behavior(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. The justification is limited to comments saying this matches the reference's left-first camera logic and behavior interpolation.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The agent pools every whisker-motion-energy sample across all sessions, computes global 1/3 and 2/3 quantiles, and discretizes each time point into `low`, `medium`, or `high` using those global thresholds.

ii. 
```python
all_me = []
for sess in session_results:
    all_me.append(sess['me_binned'].flatten())
...
me_quantiles = np.array([-np.inf,
                          np.quantile(all_me, 1/3),
                          np.quantile(all_me, 2/3),
                          np.inf])
```

```python
me_disc = np.digitize(sess['me_binned'], me_quantiles[1:-1]).astype(np.float32)
me_disc = np.clip(me_disc, 0, 2)
```

iii. The justification is explicit in the notes and README: "Global quantile-based discretization for wheel speed and whisker ME."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker trace is interpolated trial-by-trial onto the same 100-point aligned grid used for wheel speed and inputs, and trials without full whisker coverage are dropped.

ii. 
```python
me_binned, me_valid = interpolate_behavior(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
combined_valid = wheel_valid & me_valid
```

iii. The justification is the same as for wheel speed: behavior interpolation "matching reference code" plus full-window coverage checks.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files usually cause the session to be skipped: no session path, no trials table, no probe data, no wheel data, no whisker motion energy, or fewer than two usable trials all return `None`. Within a usable session, NaNs in whisker traces are removed, and trials failing wheel/whisker coverage checks are dropped.

ii. 
```python
if sess_path is None:
    return None
...
if not probe_data:
    return None
...
if wheel_times is None:
    return None
...
if me_times is None:
    return None
...
if np.sum(combined_valid) < 2:
    return None
```

```python
valid = ~np.isnan(me_values) & ~np.isnan(me_times)
me_values = me_values[valid]
me_times = me_times[valid]
```

iii. The notes justify this as sensible edge-case handling and explicitly say sessions with missing wheel/ME data are skipped and behavior coverage is enforced.

## 10-a. What are the most time-consuming steps of the code?

i. The agent appears to treat spike binning, wheel processing, and whisker-motion-energy interpolation as the main expensive per-session steps, because it times and prints those three separately for every session. Reading and scanning session files is also repeated frequently but is not called out explicitly.

ii. 
```python
t1 = time.time()
binned_spikes = bin_spikes_fast(...)
t_spike = time.time() - t1
...
t1 = time.time()
wheel_binned, wheel_valid = interpolate_behavior(...)
t_wheel = time.time() - t1
...
t1 = time.time()
me_binned, me_valid = interpolate_behavior(...)
t_me = time.time() - t1
```

iii. The notes say sample runs take about 4 s per session and emphasize optimization of the conversion loop, but they do not explicitly analyze file-I/O separately.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several per-trial Python loops remain: spike binning loops over trials, behavior interpolation loops over trials, and final dataset assembly loops over trials again to build per-trial `neural`, `input`, and `output` lists. The presence of an unused `bin_spikes_vectorized` function shows the agent considered but did not fully remove these loops.

ii. 
```python
for trial_idx in range(n_trials):
    t_start = interval_starts[trial_idx]
    t_end = interval_ends[trial_idx]
    ...
    np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
```

```python
for trial_idx in range(n_trials):
    ...
    result[trial_idx] = f_interp(x_interp)
```

iii. There is no explicit written justification beyond the attempt to add a faster spike-binning path.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly scans versioned directories to find files, repeatedly performs per-trial interpolation for both wheel and whisker traces, and then iterates over every trial again to rebuild `neural`, `input`, and `output` one trial at a time for the final pickle.

ii. 
```python
versioned_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('#')], reverse=True)
for vd in versioned_dirs:
    f = vd / filename
    if f.exists():
        return f
```

```python
for trial_idx in range(n_trials):
    session_neural.append(sess['binned_spikes'][trial_idx].astype(np.float32))
...
for trial_idx in range(n_trials):
    inp = np.stack([...], axis=0)
...
for trial_idx in range(n_trials):
    out = np.stack([...], axis=0)
```

iii. No explicit justification is given for these repeated passes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script contains an unused `merge_probes` helper with dead code, computes and stores `lab` in intermediate session results even though it is never written to the final dataset, and performs plotting-only work when `--show-processing` is used. It also keeps continuous wheel and whisker traces for all sessions until a later global discretization pass, even though only the discretized categories are saved.

ii. 
```python
def merge_probes(spikes_list, clusters_info_list):
    ...
    for spike_times, spike_clusters, n_clusters, cluster_brain_ids in zip(*[iter(x) for x in [spikes_list]]):
        pass  # This won't work, let me fix
```

```python
result = {
    'eid': eid,
    'subject': first_probe['subject'],
    'lab': first_probe['lab'],
    'wheel_binned': wheel_binned,
    'me_binned': me_binned,
```

iii. There is no explicit justification for these extra steps beyond the notes' general emphasis on debugging plots and global quantile discretization.
