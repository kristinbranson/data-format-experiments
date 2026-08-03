# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the ONE API. It loads a release index from `code/code_zhang2025/data/bwm_release.csv`, groups rows by `eid`, resolves each session by constructing a cache path under `data/one_cache`, and then loads trials, spikes, wheel, and motion-energy files directly from the filesystem.

ii. 
```python
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'
BASE_PATH = Path('data/one_cache')
```

```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
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

iii. The notes explicitly say `Direct file loading (no SpikeSortingLoader dependency)` and the trajectory says the script should “Load bwm_release.csv to get session list” and then “load spike data, trial data, wheel data, whisker ME data.”

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. The final subject list is the sorted unique set of session subjects, and each session gets an integer `subject_idx`.

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

iii. The notes say the BWM release CSV contains `459 sessions, 139 subjects`, and the trajectory describes using that CSV as the source for the session list and metadata.

## 1-c. How are the data split into sessions?

i. Sessions are identified by `eid` in `bwm_release.csv`. The agent groups all probe rows by `eid`, so one grouped entry becomes one session in the output.

ii. 
```python
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
```

iii. The notes describe `bwm_release.csv` as the session index and report available-session counts from that table.

## 1-d. How are the data split into trials?

i. Trials come from rows of the parquet trials table for each session. After loading, the agent subsets rows with boolean masks and then treats each remaining row as one trial.

ii. 
```python
trials_df = pd.read_parquet(trials_file)
```

```python
valid_trials = trials_df[trials_mask].copy()
```

```python
for trial_idx in range(n_trials):
    session_neural.append(sess['binned_spikes'][trial_idx].astype(np.float32))
```

iii. The notes repeatedly refer to “trial filtering” on the loaded trials table and use post-mask trial counts as the session trial counts.

## 1-e. How are trials filtered based on quality controls?

i. The agent first filters trials by reaction time, maximum trial length, missing values in several trial columns, and no-choice exclusion. It then requires wheel and whisker traces to interpolate successfully across the whole decoding window and drops any trials failing either behavioral validity mask.

ii. 
```python
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
if MIN_RT is not None:
    mask &= (rt >= MIN_RT)
if MAX_RT is not None:
    mask &= (rt <= MAX_RT)
```

```python
if MAX_TRIAL_LEN is not None:
    if 'goCue_times' in trials_df.columns:
        trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN)
```

```python
for event in NAN_EXCLUDE:
    if event in trials_df.columns:
        mask &= ~trials_df[event].isna()
if EXCLUDE_NOCHOICE:
    mask &= (trials_df['choice'] != 0)
```

```python
combined_valid = wheel_valid & me_valid
binned_spikes = binned_spikes[combined_valid]
wheel_binned = wheel_binned[combined_valid]
me_binned = me_binned[combined_valid]
valid_trials_final = valid_trials[combined_valid].copy()
```

iii. The notes claim this “matches reference code” and list `min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude no-choice`, while the trajectory shows the agent investigated unexpectedly low trial counts and kept the extra wheel/ME validity filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural tensor is built from raw spike times and spike cluster assignments for each probe. Additional raw arrays are loaded to infer cluster count and brain-region labels.

ii. 
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
cluster_channels = np.load(spike_dir / 'clusters.channels.npy').flatten()
chan_brain_ids = np.load(spike_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
cluster_brain_ids = chan_brain_ids[cluster_channels]
```

iii. The notes map `Spike times + clusters -> neural` and separately mention Beryl brain-region mapping via `iblatlas`.

## 2-b. How is the `neural` data processed?

i. The agent merges probes within a session, sorts spikes by time, and bins spikes into 20 ms bins for each trial window. The stored neural arrays are spike counts per bin, not rates in Hz.

ii. 
```python
all_clusters.append(sc + cluster_offset)
cluster_offset += nc
sort_idx = np.argsort(spike_times, kind='stable')
spike_times = spike_times[sort_idx]
spike_clusters = spike_clusters[sort_idx]
```

```python
bin_idx = np.minimum(np.floor((trial_times - t_start) / binsize).astype(int), n_bins - 1)
np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
```

```python
session_neural.append(sess['binned_spikes'][trial_idx].astype(np.float32))
```

iii. The notes say `Bin at 20ms, shape (n_neurons, 100)` and the trajectory emphasizes matching `binsize=0.02`, but there is no note or code step converting counts to firing rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by spike-sorting QC label. The agent keeps all clusters present in the probe output.

ii. 
```python
n_clusters = len(cluster_channels)
cluster_brain_ids = chan_brain_ids[cluster_channels]
return spike_times, spike_clusters, n_clusters, cluster_brain_ids
```

iii. The notes explicitly say `No QC filtering` and say this matches `qc=None` in the Zhang utility code; the trajectory repeats that conclusion several times.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial window is aligned to `stimOn_times` with a fixed window of `(-0.5, 1.5)` seconds. Spikes are selected from the absolute interval `[stimOn-0.5, stimOn+1.5)` and then binned relative to the window start.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

```python
idx_start = np.searchsorted(spike_times, t_start, side='left')
idx_end = np.searchsorted(spike_times, t_end, side='left')
bin_idx = np.minimum(np.floor((trial_times - t_start) / binsize).astype(int), n_bins - 1)
```

iii. The notes repeatedly say `align to stimOn_times` and the trajectory highlights `time_window=(-0.5, 1.5)` as a key reference parameter.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 20 ms bins and 100 bins per 2 s trial window. No additional temporal rebinning is applied after initial spike binning.

ii. 
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes say `20ms bins` and `100 bins`, and the trajectory repeatedly cites the same reference parameters.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a raw signal array. The agent derives it from the chosen alignment event (`stimOn_times`) and the fixed window/binning constants.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
```

```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes describe this input as `Time since stimOn` and the trajectory treats it as one of the constructed decoder inputs rather than a directly loaded variable.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent constructs a fixed 100-point vector from `-0.48` to `1.5` s using `np.linspace`, corresponding to one value per bin.

ii. 
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes say this variable is a direct mapping of the decoder timing grid and do not describe any additional processing.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The same 100-bin trial grid used for behavioral interpolation and spike binning is reused as the time input, so the input and neural arrays share a common per-trial time axis.

ii. 
```python
wheel_binned, wheel_valid = interpolate_behavior(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

```python
inp = np.stack([
    sess['time_since_stim'],
    np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32),
], axis=0)
```

iii. The notes say behavior is “interpolated to same time bins,” and the trajectory frames the time input as the bin centers used throughout the conversion.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the filtered trial table. The agent infers block boundaries from changes in that value.

ii. 
```python
def compute_trial_number_in_block(prob_left):
    current_block = prob_left.iloc[0] if hasattr(prob_left, 'iloc') else prob_left[0]
```

```python
trial_num_in_block = compute_trial_number_in_block(
    valid_trials_final['probabilityLeft']
)
```

iii. The notes list `Trial number in block` as derived by `Count within block`, and the trajectory calls it out as one of the constructed decoder inputs.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent scans the filtered trials in order, increments a counter while `probabilityLeft` stays constant, resets the counter when it changes, and assigns the first trial in each block the value `1`. It computes this after filtering, not on the original full trial table.

ii. 
```python
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
```

iii. The notes say `Count within block`, and the verification output in the trajectory shows the resulting range starts at `1.0`, confirming one-based numbering after filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the trials table `choice` column after excluding `choice == 0` trials.

ii. 
```python
if EXCLUDE_NOCHOICE:
    mask &= (trials_df['choice'] != 0)
```

```python
choice = valid_trials_final['choice'].values.copy()
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
```

iii. The notes claim `Choice | left(-1)->0, right(1)->1`, and the trajectory repeats that mapping while discussing output construction.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent converts the per-trial choice value to a binary category with `-1 -> 0` and every other surviving value to `1`, then broadcasts that constant label across all 100 time bins of the trial.

ii. 
```python
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
```

```python
np.full(N_BINS, int(sess["choice_binary"][trial_idx]), dtype=np.int64)
```

iii. The notes and trajectory both justify this as the requested left/right binary output, although the concrete mapping they wrote is the opposite of the reference convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trials table `probabilityLeft` column.

ii. 
```python
prob_left = valid_trials_final['probabilityLeft'].values.copy()
prior_cat = np.zeros(len(prob_left), dtype=np.float32)
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
```

iii. The notes list the exact mapping `0.2->0, 0.5->1, 0.8->2`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent performs a direct categorical remapping from the three probability values to integer categories, then broadcasts that constant value across all time bins for the trial.

ii. 
```python
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
```

```python
np.full(N_BINS, int(sess["prior_cat"][trial_idx]), dtype=np.int64)
```

iii. The notes treat this as a direct task-driven remapping with no additional signal processing.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
pos_file = sess_path / '_ibl_wheel.position.npy'
ts_file = sess_path / '_ibl_wheel.timestamps.npy'
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The notes explicitly say `Wheel: interpolate_position(1000Hz) -> velocity_filtered(corner=20Hz, order=8) -> abs()`, and the trajectory documents a debugging pass to make the implementation follow `SessionLoader.load_wheel()`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent linearly interpolates wheel position to a uniform 1000 Hz grid, applies an 8th-order 20 Hz low-pass Butterworth filter, differentiates to velocity, takes absolute value to get speed, and then linearly interpolates that speed trace onto each trial’s 100-bin decoding grid.

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

iii. The notes say this matches `SessionLoader.load_wheel()`, and the trajectory shows the agent changed its initial approach after discovering wheel-coverage problems.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The agent thresholds wheel speed using global quantile boundaries computed across all sessions and all time bins in the converted dataset, then maps values into three bins with `np.digitize`.

ii. 
```python
all_wheel.append(sess['wheel_binned'].flatten())
all_wheel = np.concatenate(all_wheel)
wheel_quantiles = np.array([-np.inf,
                             np.quantile(all_wheel, 1/3),
                             np.quantile(all_wheel, 2/3),
                             np.inf])
```

```python
wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1]).astype(np.float32)
wheel_disc = np.clip(wheel_disc, 0, 2)
```

iii. The notes explicitly say `Global quantile-based discretization for wheel speed and whisker ME`, and the sample/full conversion logs in the trajectory print the computed global boundaries.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel-speed trace is resampled onto the same 100-bin trial grid defined by stimulus-aligned trial intervals, so it is aligned to the neural data bin by bin.

ii. 
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
result[trial_idx] = f_interp(x_interp)
```

iii. The notes say behavioral signals are “interpolated to same time bins,” and the trajectory frames this as matching the reference alignment logic.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from either `leftCamera.ROIMotionEnergy.npy` and `_ibl_leftCamera.times.npy` or, if those are missing, the corresponding right-camera files.

ii. 
```python
me_file = find_versioned_file(sess_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_versioned_file(sess_path, '_ibl_leftCamera.times.npy')

if me_file is None or times_file is None:
    me_file = find_versioned_file(sess_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_versioned_file(sess_path, '_ibl_rightCamera.times.npy')
```

iii. The notes say the script uses `left camera first, then right`, matching the reference intent.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent loads the released motion-energy trace, trims the values and timestamps to equal length, removes NaNs, and interpolates the trace onto each trial’s 100-bin decoding grid.

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

iii. The notes describe whisker motion energy as a continuous time-varying signal that is later discretized, and the trajectory treats the interpolation as parallel to the wheel processing.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The agent thresholds whisker motion energy using global quantile boundaries computed across all sessions and all time bins, then assigns three bins with `np.digitize`.

ii. 
```python
all_me.append(sess['me_binned'].flatten())
all_me = np.concatenate(all_me)
me_quantiles = np.array([-np.inf,
                          np.quantile(all_me, 1/3),
                          np.quantile(all_me, 2/3),
                          np.inf])
```

```python
me_disc = np.digitize(sess['me_binned'], me_quantiles[1:-1]).astype(np.float32)
me_disc = np.clip(me_disc, 0, 2)
```

iii. The notes explicitly say `Global quantile-based discretization` for whisker motion energy, and the trajectory prints the resulting global quantiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker trace is interpolated onto the same stimulus-aligned 100-bin grid used for the neural data, so the two are aligned bin by bin.

ii. 
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
result[trial_idx] = f_interp(x_interp)
```

iii. The notes say behavioral variables are aligned to the same time bins as neural activity.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles missing or unusable data by skipping missing files, dropping trials that fail interpolation coverage checks, removing NaNs from whisker traces, and dropping sessions with no usable probes or fewer than two valid trials.

ii. 
```python
if trials_file is None:
    raise FileNotFoundError(f'No trials table found in {sess_path}')
```

```python
if result[0] is not None:
    probe_data.append(result)
else:
    print(f'  Session {eid}: failed to load probe {probe_name}')
```

```python
if np.sum(combined_valid) < 2:
    print(f'  Session {eid}: too few valid trials after behavior filtering ({np.sum(combined_valid)})')
    return None
```

```python
valid = ~np.isnan(me_values) & ~np.isnan(me_times)
me_values = me_values[valid]
me_times = me_times[valid]
```

iii. The notes devote a full edge-cases section to skipped wheel/ME sessions, too-few-trial sessions, and zero-neural-data warnings; the trajectory shows the agent deliberately kept dropping sessions lacking required modalities.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps in this implementation are per-session spike loading and per-trial spike binning, with wheel interpolation/filtering also becoming expensive for long sessions. Optional plotting adds extra overhead in sample runs.

ii. 
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
```

```python
for trial_idx in range(n_trials):
    idx_start = np.searchsorted(spike_times, t_start, side='left')
    idx_end = np.searchsorted(spike_times, t_end, side='left')
    np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
```

```python
t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
```

iii. The notes estimate `~4s per session average` and the trajectory repeatedly tracks spike, wheel, and total per-session timings, which indicates the agent was treating these as the main bottlenecks.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several explicit Python loops could have been vectorized further: the per-trial spike-binning loop, the per-trial behavior interpolation loop, the loop that computes trial number in block, and the loops that assemble per-trial input/output arrays. There is also an unused slower spike-binning implementation with an inner per-spike loop.

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

```python
for i in range(len(prob_left)):
    ...
    trial_nums[i] = count
```

iii. The trajectory explicitly discusses trying to make spike binning “fast” and keeps both `bin_spikes_vectorized` and `bin_spikes_fast`, which shows efficiency was a concern but only partially addressed.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of work: it rescans cache directories for each session and file, recomputes the time input vector per session, manually reimplements probe merging inside `process_session` instead of reusing the helper, and makes a full extra pass over every session to compute global discretization boundaries before building outputs.

ii. 
```python
def find_versioned_file(base_dir, filename):
    direct = base_dir / filename
    ...
    versioned_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('#')], reverse=True)
```

```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

```python
# First pass: collect all wheel speed and whisker ME values for global discretization
for sess in session_results:
    all_wheel.append(sess['wheel_binned'].flatten())
    all_me.append(sess['me_binned'].flatten())
```

iii. The notes justify the extra global discretization pass directly, and the trajectory shows the agent intentionally added it as a design choice.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main unnecessary work is optional plotting/visualization, which does not contribute to the saved dataset; computing and storing global quantile metadata is also only ancillary. The file also retains unused helper implementations such as `merge_probes`, `bin_spikes_vectorized`, and `discretize_to_bins`.

ii. 
```python
def plot_processing(session_data, session_idx, save_prefix='processing'):
    ...
    fig.savefig(f'{save_prefix}_{eid}.png', dpi=100)
```

```python
if args.show_processing and len(session_results) < 2:
    plot_processing(result, len(session_results))
```

```python
'wheel_speed_quantiles': wheel_quantiles.tolist(),
'whisker_me_quantiles': me_quantiles.tolist(),
```

iii. The notes call these “processing visualization plots,” and the trajectory shows they were added for validation rather than for the downstream decoder format itself.
