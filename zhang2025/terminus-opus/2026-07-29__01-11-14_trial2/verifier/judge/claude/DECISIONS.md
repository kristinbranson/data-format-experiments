# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the session list from `bwm_release.csv` (699 PIDs, 459 sessions, 139 subjects). It iterates over each unique EID, finds the corresponding session path in the ONE cache (`data/one_cache/<lab>/Subjects/<subject>/<date>/<session>/alf/`), and loads the trials table from `_ibl_trials.table.pqt`, spike data from probe subdirectories, wheel data from `_ibl_wheel.position.npy`/`timestamps.npy`, and whisker motion energy from camera files. Sessions that lack any required data (wheel, whisker ME, or spikes) are skipped.

ii.
```python
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'
BASE_PATH = Path('data/one_cache')

bwm_df = pd.read_csv(BWM_CSV, index_col=0)
# Group by session (eid)
sessions = {}
for _, row in bwm_df.iterrows():
    eid = row.eid
    if eid not in sessions:
        sessions[eid] = []
    sessions[eid].append({...})

# Filter to available sessions
available_sessions = {}
for eid, probes in sessions.items():
    sess_path = find_session_path(probes[0]['lab'], probes[0]['subject'], probes[0]['date'])
    if sess_path is not None:
        available_sessions[eid] = probes
```

iii. The AI followed the reference code's approach of using `bwm_release.csv` as the session index (same file used in `0_data_caching.py`). It noted that 454 of 459 sessions were locally available, with 378 ultimately succeeding after filtering for data completeness.

---

## 1-b. How are the data split into subjects (mice)?

i. The AI extracts the subject name from `bwm_release.csv` per session. After processing all sessions, unique subjects are collected, sorted, and assigned indices. Each session is mapped to its subject via `subject_idx`.

ii.
```python
all_subjects = sorted(list(set(sess['subject'] for sess in session_results)))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
# ...
subject_idx_list.append(subject_to_idx[sess['subject']])
```

iii. The AI documented 125 subjects in the final dataset (14 subjects had all sessions fail due to missing data). This is consistent with the 139 subjects in the release minus those with no viable sessions.

---

## 1-c. How are the data split into sessions?

i. Sessions are identified by their EID (experiment ID) from `bwm_release.csv`. Each unique EID is processed independently. Multiple probes per session are merged. Sessions missing required data are excluded.

ii.
```python
for i, (eid, probes) in enumerate(available_sessions.items()):
    result = process_session(eid, probes, show_processing=args.show_processing)
    if result is not None:
        session_results.append(result)
```

iii. The AI noted 454 available sessions (5 missing from cache), with 378 succeeding. Failed sessions were due to missing wheel data (63), missing whisker ME (12), or too few valid trials (1).

---

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file per session. Each row is one trial. After applying quality filters, the surviving trials are used. For each trial, a 2-second window aligned to stimulus onset is extracted.

ii.
```python
trials_df = pd.read_parquet(trials_file)
# ... apply mask ...
valid_trials = trials_df[trials_mask].copy()
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]  # stimOn - 0.5s
interval_ends = stim_on + TIME_WINDOW[1]    # stimOn + 1.5s
```

iii. The AI computed trial intervals relative to stimulus onset matching the reference code's approach. Each trial spans (-0.5, 1.5) seconds relative to stimOn_times.

---

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using multiple criteria: (1) reaction time between 0.08s and 2.0s, (2) trial length (feedback_times - goCue_times) <= 10s, (3) NaN exclusion for key events (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (4) no-choice trials excluded (choice != 0). Additionally, trials where wheel or whisker ME data is unavailable or doesn't cover the interval are excluded.

ii.
```python
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
EXCLUDE_NOCHOICE = True
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)
# ...
combined_valid = wheel_valid & me_valid
```

iii. The AI matched all trial filtering parameters from the reference code (`load_trials_and_mask` defaults plus `max_trial_len=10.0` from the `prepare_data` call). This is consistent with the data paper's inclusion criteria.

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times (`spikes.times.npy`) and spike cluster assignments (`spikes.clusters.npy`) from each probe's pykilosort output directory.

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
cluster_channels = np.load(spike_dir / 'clusters.channels.npy').flatten()
```

iii. The AI loads raw spike sorting output files directly rather than using the SpikeSortingLoader API (which was not available in the local environment). This is functionally equivalent.

---

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms time bins within each trial's 2-second window (-0.5s to 1.5s relative to stimulus onset), producing 100 time bins per trial. Multiple probes within a session are merged by offsetting cluster IDs and concatenating. The result is spike count matrices of shape (n_clusters, 100) per trial.

ii.
```python
BINSIZE = 0.02  # 20ms bins
TIME_WINDOW = (-0.5, 1.5)
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins

binned_spikes = bin_spikes_fast(
    spike_times, spike_clusters, interval_starts, interval_ends,
    n_clusters, BINSIZE, N_BINS
)
```

iii. The AI documented that binning parameters match the reference code: 20ms bins, stimOn alignment, (-0.5, 1.5) window, producing 100 bins. The `bin_spikes_fast` function uses searchsorted for efficiency.

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI uses NO quality control filtering on neurons. All clusters from the spike sorting output are included (equivalent to `qc=None` in the reference code).

ii.
```python
# In load_spike_data:
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
# No QC filtering applied
```

iii. The AI noted that the reference code calls `load_spiking_data(one, pid)` with default `qc=None`, meaning all clusters are used regardless of quality. The data paper's stringent QC metrics (amplitude > 50uV, noise cutoff < 20uV, refractory period violation) are NOT applied, matching the reference method paper's approach of "using all neurons, sorted by Kilosort."

However, there is a difference: the reference code's `get_spike_data_per_interval` uses `cluster_ids = np.unique(clusters)` which only includes clusters that fired at least one spike. The AI uses `n_clusters = len(cluster_channels)` which includes ALL clusters from the sorting output, including potentially silent clusters.

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). Each trial's spike window starts at `stimOn_times - 0.5s` and ends at `stimOn_times + 1.5s`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. The AI followed the instructions ("Temporally align based on stimulus onset") and reference code (`align_time='stimOn_times'`, `time_window=(-.5, 1.5)`).

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s) per bin, producing 100 bins per 2-second trial. No temporal rebinning is applied after the initial binning.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
```

iii. The AI matched the reference code's 20ms bin size. The methods paper states "The neural activity within each trial is binned into non-overlapping 20 ms bins" for the trial-aligned dataset.

---

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is not derived from any raw data variable. It is a computed time vector based on the bin structure and the alignment event (stimulus onset).

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The AI constructs the time vector to match the interpolation time points used in the reference code's `get_behavior_per_interval`: `x_interp = np.linspace(interval_beg + binsize, interval_end, n_bins)`. The values range from -0.48s to 1.5s in 100 bins.

---

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A linearly spaced vector is created from `TIME_WINDOW[0] + BINSIZE` (-0.48) to `TIME_WINDOW[1]` (1.5) with `N_BINS` (100) points. This represents the right edge / interpolation point of each time bin relative to stimulus onset.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The AI uses the same bin timing formula as the reference code's behavior interpolation. The time values represent bin right edges (shifted by one binsize from the interval start).

---

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same time vector is used for all trials (it is identical for every trial since all trials use the same window relative to stimulus onset). It is broadcast as a time-varying input of shape (n_bins,) for each trial.

ii.
```python
inp = np.stack([
    sess['time_since_stim'],  # (n_bins,)
    np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32),
], axis=0)  # (2, n_bins)
```

iii. The time vector is consistent with the neural data binning since both use the same window and bin structure.

---

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column of the trials table. Block boundaries are detected as changes in probabilityLeft.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
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
    return trial_nums
```

iii. The reference code stores `block = trials_df['probabilityLeft'].to_numpy()` as the block identity. The AI interprets "trial number in block" as the ordinal position within each block, which is a reasonable interpretation of the decoder task instructions.

---

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI iterates through the (filtered) trials in order, tracking the current block (identified by probabilityLeft value). When probabilityLeft changes, a new block starts and the counter resets to 1. Each trial's count within its block is recorded.

ii.
```python
for i in range(len(prob_left)):
    val = prob_left.iloc[i] if hasattr(prob_left, 'iloc') else prob_left[i]
    if val == current_block:
        count += 1
    else:
        current_block = val
        count = 1
    trial_nums[i] = count
```

Note: This is computed on the filtered trial set (`valid_trials_final['probabilityLeft']`), meaning filtered-out trials are not counted, which could shift block boundaries and trial counts compared to the original trial sequence. The per-trial value is broadcast to all time bins within a trial.

iii. The AI computes this as a running count within contiguous blocks of same probabilityLeft. It is then broadcast to all time bins as a constant-per-trial, time-varying input.

---

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table (`_ibl_trials.table.pqt`).

ii.
```python
choice = valid_trials_final['choice'].values.copy()
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
```

iii. In the IBL convention, choice=-1 is left and choice=1 is right.

---

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice=-1 (left) to 0 and choice=1 (right) to 1, as specified in the decoder task instructions. No-choice trials (choice=0) are already excluded by the trial mask. The per-trial value is broadcast to all time bins.

ii.
```python
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
# In build_final_dataset:
np.full(N_BINS, int(sess["choice_binary"][trial_idx]), dtype=np.int64)
```

iii. The mapping matches the instructions: "left = 0, right = 1". The output is stored as int64 broadcast across all time bins.

---

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column of the trials table.

ii.
```python
prob_left = valid_trials_final['probabilityLeft'].values.copy()
```

iii. probabilityLeft takes values 0.2, 0.5, or 0.8 in the IBL task.

---

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps continuous probabilityLeft values to categories: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2, matching the decoder task instructions. The per-trial value is broadcast to all time bins.

ii.
```python
prior_cat = np.zeros(len(prob_left), dtype=np.float32)
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
# In build_final_dataset:
np.full(N_BINS, int(sess["prior_cat"][trial_idx]), dtype=np.int64)
```

iii. The encoding follows the instructions exactly: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

---

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy` files.

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. These are the raw wheel encoder position and timestamps.

---

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes wheel speed in three steps matching `SessionLoader.load_wheel()`: (1) interpolate wheel position to 1000Hz uniform sampling, (2) apply 8th-order Butterworth low-pass filter at 20Hz corner frequency and compute velocity via differentiation, (3) take absolute value to get speed. The speed is then interpolated to match neural time bins using linear interpolation.

ii.
```python
def compute_wheel_speed(sess_path, fs=1000, corner_frequency=20, order=8):
    # Step 1: Interpolate position to uniform sampling
    t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    pos_interp = interp1d(timestamps, position, kind='linear')(t_uniform)
    # Step 2: Butterworth filter + velocity
    sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2, btype='lowpass', output='sos')
    vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
    # Step 3: Speed = abs(velocity)
    speed = np.abs(vel).astype(np.float32)
    return t_uniform, speed
```

iii. The AI documented this matches `interpolate_position` + `velocity_filtered` from the reference wheel.py, which is exactly what `SessionLoader.load_wheel()` calls.

---

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using global quantile boundaries (terciles) computed across all sessions. Values below the 33rd percentile are bin 0, between 33rd and 67th percentile are bin 1, and above 67th percentile are bin 2.

ii.
```python
# In build_final_dataset:
wheel_quantiles = np.array([-np.inf,
                             np.quantile(all_wheel, 1/3),
                             np.quantile(all_wheel, 2/3),
                             np.inf])
wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1]).astype(np.float32)
wheel_disc = np.clip(wheel_disc, 0, 2)
```

iii. The instructions say "Wheel speed discretized into 3 bins, time-varying." The AI chose quantile-based discretization across all sessions to ensure roughly equal class frequencies. The reference code does not perform discretization (it stores continuous values), so this is the AI's design choice.

---

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed (at 1000Hz) is interpolated to match the neural time bins using the same formula as the reference code's `get_behavior_per_interval`: `x_interp = np.linspace(interval_beg + binsize, interval_end, n_bins)`.

ii.
```python
def interpolate_behavior(beh_times, beh_values, interval_starts, interval_ends, binsize, n_bins):
    x_interp = np.linspace(t_start + binsize, t_end, n_bins)
    f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
    result[trial_idx] = f_interp(x_interp)
```

iii. The interpolation points match the reference code exactly. Trials where behavior data doesn't cover the interval are excluded.

---

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from camera ROI motion energy files: `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), with corresponding camera timestamps.

ii.
```python
def load_whisker_motion_energy(sess_path):
    me_file = find_versioned_file(sess_path, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_versioned_file(sess_path, '_ibl_leftCamera.times.npy')
    if me_file is None or times_file is None:
        me_file = find_versioned_file(sess_path, 'rightCamera.ROIMotionEnergy.npy')
        times_file = find_versioned_file(sess_path, '_ibl_rightCamera.times.npy')
```

iii. The AI matches the reference code's `bin_behaviors` which loads 'left-whisker-motion-energy' first and falls back to 'right-whisker-motion-energy'.

---

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded, NaN values are removed, and the signal is interpolated to match neural time bins using the same method as wheel speed (linear interpolation to bin centers).

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

iii. The AI handles potential length mismatches between timestamps and values, which is a practical consideration for camera data.

---

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global quantile-based discretization into 3 bins using terciles across all sessions.

ii.
```python
me_quantiles = np.array([-np.inf,
                          np.quantile(all_me, 1/3),
                          np.quantile(all_me, 2/3),
                          np.inf])
me_disc = np.digitize(sess['me_binned'], me_quantiles[1:-1]).astype(np.float32)
me_disc = np.clip(me_disc, 0, 2)
```

iii. Same rationale as wheel speed discretization.

---

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same interpolation approach as wheel speed, using `interpolate_behavior` with matching time bins.

ii.
```python
me_binned, me_valid = interpolate_behavior(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. The interpolation points and validity checks match the reference code's `get_behavior_per_interval`.

---

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several categories of missing/problematic data:
- **Missing sessions**: Sessions not found in the ONE cache are skipped (5 of 459).
- **Missing probes**: Individual probes that fail to load are skipped; sessions continue with remaining probes.
- **Missing behavioral data**: Sessions without wheel data (63) or whisker ME (12) are entirely excluded.
- **Invalid trials**: Trials with NaN events, out-of-range reaction times, or missing behavior coverage are excluded.
- **Camera data length mismatch**: Timestamps and values are truncated to the shorter length.
- **NaN in behavior data**: NaN values in whisker ME are removed before interpolation.
- **Zero neural data**: Sessions with all-zero neural data for some trials generate warnings but are kept.

ii.
```python
# Missing probes
if result[0] is not None:
    probe_data.append(result)
else:
    print(f'  Session {eid}: failed to load probe {probe_name}')

# Behavior validity
combined_valid = wheel_valid & me_valid
if np.sum(combined_valid) < 2:
    return None

# Camera length mismatch
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
```

iii. The AI documented failure counts in CONVERSION_NOTES.md. The approach of skipping sessions with missing behavioral data is consistent with the reference code's behavior.

---

## 12-a. What are the most time-consuming steps of the code?

i. Based on the AI's timing output, the most time-consuming steps are: (1) spike binning (`bin_spikes_fast`), which loops over all trials per session, and (2) wheel speed computation, which involves 1000Hz interpolation and Butterworth filtering of the entire session's wheel data. The full conversion took approximately 35.5 minutes for 378 sessions.

ii.
```python
# Timing output per session:
print(f'  Session {eid}: ... (spike:{t_spike:.1f}s, wheel:{t_wheel:.1f}s, ME:{t_me:.1f}s, total:{t_total:.1f}s)')
```

iii. The AI noted approximately 4s per session average.

---

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
- `bin_spikes_fast`: Loops over trials sequentially. Could potentially be parallelized across trials.
- `interpolate_behavior`: Loops over trials. Each trial's interpolation is independent and could be vectorized.
- `compute_trial_number_in_block`: Simple sequential loop that could use `np.diff` + `np.cumsum` for block detection.
- The main session processing loop is sequential (`for i, (eid, probes) in enumerate(available_sessions.items())`).

ii.
```python
# bin_spikes_fast trial loop
for trial_idx in range(n_trials):
    # ...
    np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)

# interpolate_behavior trial loop
for trial_idx in range(n_trials):
    f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
    result[trial_idx] = f_interp(x_interp)
```

iii. The reference code uses `multiprocessing.Pool` for both spike binning and behavior interpolation. The AI chose sequential loops, which is simpler but slower.

---

## 12-c. What processing does the code repeat multiple times?

i. The AI does not substantially repeat processing. Each step (trial filtering, spike binning, wheel computation, ME loading, behavior interpolation) is performed once per session. The `BrainRegions()` object is created anew for each session (`br = BrainRegions()` inside `process_session`), which involves loading atlas data repeatedly.

ii.
```python
# Created fresh each session:
br = BrainRegions()
cluster_acronyms_raw = br.id2acronym(cluster_brain_ids)
cluster_acronyms_beryl = br.acronym2acronym(cluster_acronyms_raw, mapping='Beryl')
```

iii. The `BrainRegions()` initialization could be done once and passed to each session for efficiency.

---

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are performed but not fully used:
- Wheel speed is computed for the **entire session** (all timepoints at 1000Hz), but only the portions within trial windows are used.
- The full `trials_df` is loaded including many columns (contrastLeft, contrastRight, rewardVolume, etc.) that are never used in the conversion.
- The `merge_probes` function defined on lines 154-180 is never called (inline merging is used instead in `process_session`).
- Additional trial timing files (`stimOnTrigger_times`, `goCueTrigger_times`) are loaded but not used in any processing step.

ii.
```python
# Full session wheel computation (only trial windows used)
t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
pos_interp = interp1d(timestamps, position, kind='linear')(t_uniform)

# Unused loaded data
stim_trig_file = find_versioned_file(sess_path, '_ibl_trials.stimOnTrigger_times.npy')
go_trig_file = find_versioned_file(sess_path, '_ibl_trials.goCueTrigger_times.npy')
```

iii. Computing wheel speed for the full session is consistent with the reference code's approach (SessionLoader.load_wheel processes the whole session), but it's computationally wasteful.
