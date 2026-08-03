# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script reads the session/probe manifest from `bwm_release.csv`, groups rows by `eid`, keeps only sessions whose ONE-cache folder exists locally, and then processes each surviving session by loading trials, spikes, wheel data, and whisker motion energy directly from the cache instead of using `ONE`/`SpikeSortingLoader`.

ii. ```python
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'

bwm_df = pd.read_csv(BWM_CSV, index_col=0)

sessions = {}
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

available_sessions = {}
for eid, probes in sessions.items():
    sess_path = find_session_path(probes[0]['lab'], probes[0]['subject'], probes[0]['date'])
    if sess_path is not None:
        available_sessions[eid] = probes
```

iii. In `CONVERSION_NOTES.md`, the agent says it replaced `SpikeSortingLoader` with direct file loading because the needed package path was unavailable locally, and its trajectory says this was “fine” because the same `.npy`/`.pqt` files could be read directly while preserving the reference processing.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column in `bwm_release.csv`. After session processing, the script builds a sorted list of unique subject IDs and records one `subject_idx` per session from the first probe’s subject label.

ii. ```python
all_subjects = sorted(list(set(sess['subject'] for sess in session_results)))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx_list.append(subject_to_idx[sess['subject']])
```

iii. The notes describe the cache as organized under `Subjects/<mouse>/<date>/<session>/alf/`, and the trajectory shows the agent treating `row.subject` from `bwm_release.csv` as the authoritative mouse ID.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from `bwm_release.csv`. All probes sharing an `eid` are grouped together and processed as one merged session.

ii. ```python
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

iii. The trajectory records that the agent identified `0_data_caching.py` as grouping by session and then merging probes for that session through `prepare_data(...)`.

## 1-d. How are the data split into trials?

i. Trials come from rows of `_ibl_trials.table.pqt`. The script first applies the trial-quality mask, then constructs one aligned interval per surviving row using `stimOn_times + (-0.5, 1.5)`, and finally drops any trials whose wheel or whisker signals do not adequately cover that interval.

ii. ```python
valid_trials = trials_df[trials_mask].copy()
...
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
...
combined_valid = wheel_valid & me_valid
...
valid_trials_final = valid_trials[combined_valid].copy()
valid_trials_final.reset_index(drop=True, inplace=True)
```

iii. The notes say trial curation follows `load_trials_and_mask`, and the trajectory identifies `align_spike_behavior` / `get_behavior_per_interval` as the reference mechanism for deleting trials whose behavioral streams are unusable after alignment.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by reaction time (`0.08 <= firstMovement_times - stimOn_times <= 2.0`), maximum trial length (`feedback_times - goCue_times <= 10.0` when available), required non-NaN trial fields, and exclusion of `choice == 0`. After that, trials are also removed if wheel or whisker data fail interval-coverage checks, and sessions with fewer than two remaining trials are discarded.

ii. ```python
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
if MIN_RT is not None:
    mask &= (rt >= MIN_RT)
if MAX_RT is not None:
    mask &= (rt <= MAX_RT)
...
for event in NAN_EXCLUDE:
    if event in trials_df.columns:
        mask &= ~trials_df[event].isna()
...
if EXCLUDE_NOCHOICE:
    mask &= (trials_df['choice'] != 0)
...
combined_valid = wheel_valid & me_valid
if np.sum(combined_valid) < 2:
    return None
```

iii. `CONVERSION_NOTES.md` explicitly lists these defaults as taken from `load_trials_and_mask` and `prepare_data`, and the trajectory logs the same min/max RT, NaN exclusions, and no-choice rule while reading `ibl_data_utils.py`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from spike times and spike cluster IDs for each probe, plus cluster-to-channel and channel-to-brain-region mappings.

ii. ```python
spike_times = np.load(spike_dir / 'spikes.times.npy').flatten()
spike_clusters = np.load(spike_dir / 'spikes.clusters.npy').flatten()
cluster_channels = np.load(spike_dir / 'clusters.channels.npy').flatten()
chan_brain_ids = np.load(spike_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
cluster_brain_ids = chan_brain_ids[cluster_channels]
```

iii. The notes summarize the neural source as “Spike times + clusters,” and the trajectory shows the agent inspecting the same ALF/pykilosort files in the cache.

## 2-b. How is the `neural` data processed?

i. For each session, spikes from all probes are merged by offsetting cluster IDs, concatenating, and time-sorting. The merged spike train is then binned into 20 ms bins over a 2 s window around `stimOn_times`, producing per-trial arrays of shape `(n_neurons, 100)`. Brain-region labels are converted to Beryl acronyms but not used to subset neurons.

ii. ```python
for st, sc, nc, cbi in probe_data:
    all_times.append(st)
    all_clusters.append(sc + cluster_offset)
    all_brain_ids.append(cbi)
    cluster_offset += nc
...
sort_idx = np.argsort(spike_times, kind='stable')
spike_times = spike_times[sort_idx]
spike_clusters = spike_clusters[sort_idx]
...
binned_spikes = bin_spikes_fast(
    spike_times, spike_clusters, interval_starts, interval_ends,
    n_clusters, BINSIZE, N_BINS
)
```

iii. The trajectory says the agent identified the reference path as `prepare_data -> list_brain_regions -> select_brain_regions -> bin_spiking_data`, with `single_region=False`, so it intentionally kept all merged neurons together rather than splitting by region.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is effectively not neuron-filtered: all clusters on available probes are kept, regardless of QC labels. Missing probes cause warnings or session failure only if no probe data remain.

ii. ```python
# Load cluster info
cluster_channels = np.load(spike_dir / 'clusters.channels.npy').flatten()
...
if not probe_data:
    print(f'  Session {eid}: no probe data loaded')
    return None
```

iii. The notes repeatedly say “No QC filtering” and tie that to `load_spiking_data(..., qc=None)` from the reference code. The trajectory highlights this as a deliberate match to `prepare_data`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural trial is aligned to `stimOn_times` with a fixed window from `-0.5 s` to `+1.5 s`.

ii. ```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. The notes call this out as a direct carryover from `0_data_caching.py`, and the trajectory records those exact parameter values while reading the reference script.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data use 20 ms bins, yielding 100 bins over the 2 s alignment window. No coarser rebinning is applied after spike counting.

ii. ```python
BINSIZE = 0.02  # 20ms bins
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
```

iii. `CONVERSION_NOTES.md` says 20 ms came from both the methods text and `0_data_caching.py`, and the trajectory identifies `binsize=0.02` as one of the core reference parameters.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a dedicated raw array. It is derived from the alignment choice `stimOn_times` together with the fixed window and bin size, producing a relative-time vector shared by all trials.

ii. ```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes say this input is “Time since stimOn,” and the trajectory says the agent wanted this to “match the bin centers used in reference code.”

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script computes the bin-center times from `-0.48` to `1.5` seconds using `np.linspace`, which corresponds to the interpolation grid used for behavior alignment.

ii. ```python
# Matches the bin centers used in reference code
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The agent’s notes and trajectory both cite the reference interpolation rule `x_interp = np.linspace(interval_beg + binsize, interval_end, n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned by construction: the same window and number of bins as the neural trial matrices are used, and the same vector is broadcast into every trial.

ii. ```python
inp = np.stack([
    sess['time_since_stim'],
    np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32),
], axis=0)
```

iii. The justification in the notes is that the time vector should match the reference bin centers, so it can be paired one-for-one with the neural bins.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the per-trial `probabilityLeft` field in the filtered trials table.

ii. ```python
trial_num_in_block = compute_trial_number_in_block(
    valid_trials_final['probabilityLeft']
)
```

iii. In the notes, the agent maps “Trial number in block” to a count within blocks defined by `probabilityLeft`, because that is the available block-identity variable in the task data.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script scans the filtered `probabilityLeft` sequence, increments a counter while the value stays the same, resets the counter when the probability changes, starts each block at `1`, and then broadcasts the per-trial scalar across all 100 time bins.

ii. ```python
for i in range(len(prob_left)):
    val = prob_left.iloc[i] if hasattr(prob_left, 'iloc') else prob_left[i]
    if val == current_block:
        count += 1
    else:
        current_block = val
        count = 1
    trial_nums[i] = count
...
np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32)
```

iii. The trajectory does not show a reference implementation for this variable; the notes present it as the agent’s chosen mapping from `probabilityLeft` to the requested decoder input.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the trial-table `choice` column.

ii. ```python
choice = valid_trials_final['choice'].values.copy()
```

iii. The notes and trajectory both identify `choice` as a directly available per-trial variable in the trials table and in the reference `bin_behaviors` helper.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The script converts IBL’s left/right coding from `-1/+1` to the requested `0/1`, after already excluding `choice == 0` no-response trials. It then repeats the scalar category over all time bins in the final output tensor.

ii. ```python
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
...
np.full(N_BINS, int(sess["choice_binary"][trial_idx]), dtype=np.int64)
```

iii. The notes explicitly say “left(-1)->0, right(1)->1,” and the trajectory includes a spot check where the agent verified this mapping on a raw trial.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial-table `probabilityLeft` column.

ii. ```python
prob_left = valid_trials_final['probabilityLeft'].values.copy()
```

iii. The notes map “Prior prob left” directly from `probabilityLeft`, matching the block variable used in the reference behavior table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The script maps the three expected probabilities to integer class IDs: `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, then repeats the class over time bins.

ii. ```python
prior_cat = np.zeros(len(prob_left), dtype=np.float32)
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
...
np.full(N_BINS, int(sess["prior_cat"][trial_idx]), dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` lists this exact remapping in Step 5, and the trajectory shows the agent verifying it against a raw trial value.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. ```python
pos_file = sess_path / '_ibl_wheel.position.npy'
ts_file = sess_path / '_ibl_wheel.timestamps.npy'
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The notes call this out as matching `SessionLoader.load_wheel()`, and the trajectory shows the agent inspecting both the raw wheel arrays and the IBL wheel-processing code before finalizing the implementation.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The script interpolates wheel position onto a 1000 Hz uniform time grid, low-pass filters the interpolated position with an 8th-order Butterworth filter at 20 Hz, differentiates to velocity, then takes the absolute value to obtain speed.

ii. ```python
t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
pos_interp = interp1d(timestamps, position, kind='linear')(t_uniform)
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
```

iii. The trajectory shows the agent first trying a simpler finite-difference method, then replacing it after reading IBL wheel code and noting that the reference path used interpolation plus filtered velocity.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is discretized into three global equal-frequency bins using the 1/3 and 2/3 quantiles computed across all wheel-speed samples from all retained sessions and trials.

ii. ```python
all_wheel = np.concatenate(all_wheel)
wheel_quantiles = np.array([-np.inf,
                             np.quantile(all_wheel, 1/3),
                             np.quantile(all_wheel, 2/3),
                             np.inf])
...
wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1]).astype(np.float32)
wheel_disc = np.clip(wheel_disc, 0, 2)
```

iii. The notes say the task required categorical outputs, so the agent chose “global quantile-based discretization” as a way to produce balanced 3-class targets.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned by slicing the continuous wheel signal over the same per-trial `stimOn_times + (-0.5, 1.5)` intervals, requiring interval coverage within one bin, and linearly interpolating onto the 100 neural bin centers.

ii. ```python
wheel_binned, wheel_valid = interpolate_behavior(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
)
...
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
result[trial_idx] = f_interp(x_interp)
```

iii. The notes tie this to `get_behavior_per_interval`, and the trajectory explicitly records the agent copying that interpolation grid and interval-validity logic from the reference helper.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` and `_ibl_leftCamera.times.npy`, with fallback to the corresponding right-camera files if the left-camera data are unavailable.

ii. ```python
me_file = find_versioned_file(sess_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_versioned_file(sess_path, '_ibl_leftCamera.times.npy')
if me_file is None or times_file is None:
    me_file = find_versioned_file(sess_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_versioned_file(sess_path, '_ibl_rightCamera.times.npy')
```

iii. The notes and trajectory both say this was chosen to match the reference `bin_behaviors` logic, which tries left whisker motion energy first and falls back to right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads the motion-energy values and timestamps, truncates them to a common minimum length, removes NaN samples, and then later interpolates the signal onto per-trial aligned bins.

ii. ```python
me_values = np.load(me_file).flatten()
me_times = np.load(times_file).flatten()
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]
valid = ~np.isnan(me_values) & ~np.isnan(me_times)
me_values = me_values[valid]
me_times = me_times[valid]
```

iii. The notes describe this simply as “loaded from left camera (fallback to right), interpolated to match neural bins.” The NaN and length cleanup appear to be the agent’s own robustness choice.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is discretized into three global quantile bins using thresholds computed over all retained continuous samples from all sessions.

ii. ```python
all_me = np.concatenate(all_me)
me_quantiles = np.array([-np.inf,
                          np.quantile(all_me, 1/3),
                          np.quantile(all_me, 2/3),
                          np.inf])
...
me_disc = np.digitize(sess['me_binned'], me_quantiles[1:-1]).astype(np.float32)
me_disc = np.clip(me_disc, 0, 2)
```

iii. The notes justify this the same way as wheel speed: the task demanded categorical outputs, so the agent used global quantile bins to create three classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned with exactly the same interval extraction and interpolation scheme as wheel speed: per-trial `stimOn_times + (-0.5, 1.5)` intervals, coverage checks, then linear interpolation onto the 100 neural bin centers.

ii. ```python
me_binned, me_valid = interpolate_behavior(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
...
combined_valid = wheel_valid & me_valid
```

iii. The notes say behavioral signals were “interpolated to same time bins,” and the trajectory associates this with the reference `get_behavior_per_interval` helper.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing trials files cause session failure; missing probes are tolerated unless all probes fail; missing wheel data, missing whisker data, or fewer than two surviving trials cause the whole session to be skipped. Versioned ALF files are resolved by searching dated subdirectories. Whisker arrays are clipped to matching lengths and NaNs are removed. During interpolation, trials with inadequate signal coverage are dropped.

ii. ```python
if trials_file is None:
    raise FileNotFoundError(...)
...
if result[0] is not None:
    probe_data.append(result)
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
min_len = min(len(me_values), len(me_times))
...
valid = ~np.isnan(me_values) & ~np.isnan(me_times)
...
if len(seg_vals) == 0:
    valid_mask[trial_idx] = False
```

iii. The notes explicitly list skipped sessions with missing wheel/ME data as an edge-case policy, and the trajectory shows the agent discovering these gaps during full conversion and accepting session skipping as the chosen fallback.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are session-wise spike binning, wheel resampling/filtering, behavior interpolation, and the full-data global discretization pass over wheel and whisker arrays. The progress printouts also show that wheel processing can dominate some long sessions.

ii. ```python
t1 = time.time()
binned_spikes = bin_spikes_fast(...)
t_spike = time.time() - t1
...
t1 = time.time()
wheel_times, wheel_speed = compute_wheel_speed(sess_path)
...
t_wheel = time.time() - t1
...
print(f'  Session {eid}: ... (spike:{t_spike:.1f}s, wheel:{t_wheel:.1f}s, ME:{t_me:.1f}s, total:{t_total:.1f}s)')
...
all_wheel = np.concatenate(all_wheel)
all_me = np.concatenate(all_me)
```

iii. The trajectory repeatedly comments on spike and wheel timing during the full run, including sessions where wheel interpolation took tens of seconds.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops could be reduced: the per-trial loop in `bin_spikes_vectorized`, the per-trial loop in `bin_spikes_fast`, the per-trial interpolation loop in `interpolate_behavior`, the block-count loop in `compute_trial_number_in_block`, and the repeated per-trial construction loops in `build_final_dataset`.

ii. ```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)

for trial_idx in range(n_trials):
    ...
    result[trial_idx] = f_interp(x_interp)

for i in range(len(prob_left)):
    ...

for trial_idx in range(n_trials):
    session_neural.append(...)
...
for trial_idx in range(n_trials):
    session_input.append(inp)
...
for trial_idx in range(n_trials):
    session_output.append(out)
```

iii. The notes frame the script as a pragmatic direct-file implementation, not a highly optimized one, and the presence of an unused slower helper (`bin_spikes_vectorized`) makes the vectorization opportunities explicit.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly scans versioned directories for files, re-instantiates `BrainRegions()` per session, loops over trials multiple times for binning and then again for dataset assembly, and computes trial-aligned wheel/whisker arrays only to flatten them later for a second-pass global discretization.

ii. ```python
versioned_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('#')], reverse=True)
...
br = BrainRegions()
...
for trial_idx in range(n_trials):
    ...
for trial_idx in range(n_trials):
    ...
for trial_idx in range(n_trials):
    ...
all_wheel.append(sess['wheel_binned'].flatten())
all_me.append(sess['me_binned'].flatten())
```

iii. The trajectory shows the agent first optimizing wheel processing, then accepting the remaining repeated passes as good enough for the full conversion runtime.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script contains an unused helper (`merge_probes`) with a dead placeholder block and an unused `bin_spikes_vectorized`. It also performs optional plotting/debug storage, prints large brain-region summaries, and expands inherently per-trial outputs (`choice`, `prior`, and block index) into full 100-bin time series even though those values are constant within each trial.

ii. ```python
def merge_probes(spikes_list, clusters_info_list):
    ...
    for spike_times, spike_clusters, n_clusters, cluster_brain_ids in zip(*[iter(x) for x in [spikes_list]]):
        pass  # This won't work, let me fix

def bin_spikes_vectorized(...):
    ...

np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32)
...
np.full(N_BINS, int(sess["choice_binary"][trial_idx]), dtype=np.int64)
np.full(N_BINS, int(sess["prior_cat"][trial_idx]), dtype=np.int64)
```

iii. These choices are visible directly in the code; the notes do not justify them as required downstream behavior, and the trajectory treats plotting and diagnostics as convenience features rather than part of the reference processing.
