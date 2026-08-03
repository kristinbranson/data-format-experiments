# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script does not use the reference release table or `ONE` API. It scans `data/one_cache` for directories matching `*/Subjects/*/*/001`, then keeps only session directories that already contain spikes, a trials parquet, wheel timestamps, and whisker motion-energy files plus camera times. It then loads trials, spikes, wheel, and whisker data session by session from those local files.

ii. 
```python
DATA_ROOT = Path('data/one_cache')

def find_session_dirs():
    session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
    valid = []
    for sdir in session_dirs:
        has_spikes = len(glob.glob(os.path.join(sdir, 'alf/probe*/pykilosort/*/spikes.times.npy'))) > 0
        has_trials = len(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt'))) > 0
        has_wheel = os.path.exists(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy'))
        ...
        if has_spikes and has_trials and has_wheel and has_me:
            valid.append(sdir)
```

iii. The notes say the agent chose to “use all 393 sessions with complete data” on disk rather than the release-table session list, and the trajectory summary says it deliberately processed sessions from the local cache layout under `data/one_cache`.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the path segment immediately after `Subjects/`. The script keeps a unique ordered `subjects` list and then builds `subject_idx` per session from the session’s parsed subject name.

ii. 
```python
def parse_session_info(sdir):
    parts = Path(sdir).parts
    sub_idx = parts.index('Subjects')
    lab = parts[sub_idx - 1]
    subject = parts[sub_idx + 1]
    date = parts[sub_idx + 2]
    return lab, subject, date

...
if subject not in all_subjects:
    all_subjects.append(subject)
subject_per_session.append(subject)
...
subject_idx = np.array([all_subjects.index(s) for s in subject_per_session], dtype=np.int32)
```

iii. The notes describe the cache structure as `data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf/`, so the agent justified path-based subject extraction from the on-disk organization.

## 1-c. How are the data split into sessions?

i. Each `.../Subjects/<subject>/<date>/001` directory is treated as one session. After `find_session_dirs()` filters for required files, the main loop processes each session directory independently and appends one session entry to `neural`, `input`, and `output`.

ii. 
```python
session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
...
for i, sdir in enumerate(session_dirs):
    lab, subject, date = parse_session_info(sdir)
    result = process_session(sdir, br, ...)
    if result is not None:
        neural.append(result['neural'])
        input_data.append(result['input'])
        output_data.append(result['output'])
```

iii. The notes explicitly say the data structure is organized by session directory and that the converter “processes sessions from `data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf/`.”

## 1-d. How are the data split into trials?

i. Trials are the rows of the session’s `_ibl_trials.table.pqt` file. The script reads the most recent parquet revision, computes one time interval per row, bins spikes and behavior per row, then keeps only the rows whose combined trial mask passes.

ii. 
```python
def load_trials(sdir):
    trial_files = sorted(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt')))
    trials = pd.read_parquet(trial_files[-1])
    return trials

...
stim_on = trials[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
...
good_indices = np.where(combined_mask)[0]
trials_good = trials.iloc[good_indices]
```

iii. The notes describe the trials parquet as the canonical trial table and say trial curation follows `load_trials_and_mask` from the reference code.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered with a boolean mask that excludes reaction times outside 0.08 to 2.0 s, trials longer than 10 s from `goCue_times` to `feedback_times`, no-choice trials (`choice == 0`), and trials with NaNs in key columns. This mask is then intersected with wheel and whisker availability masks.

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

iii. The notes say this was copied from reference `load_trials_and_mask`, and the trajectory excerpt from `ibl_data_utils.py` shows the same defaults: RT 0.08 to 2.0, `exclude_nochoice=True`, default NaN exclusions, and `max_trial_len=10.0`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from spike times and spike cluster ids from each probe, with cluster-to-channel and channel-to-brain-location arrays used only to annotate neurons by region.

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

iii. The notes say the agent followed `load_spiking_data` and `merge_probes`, but reimplemented them directly from local `.npy` files instead of using `SpikeSortingLoader`.

## 2-b. How is the `neural` data processed?

i. For each session, probe data are merged by offsetting cluster ids across probes, concatenating spikes, sorting by spike time, mapping cluster regions into the Beryl atlas, and then binning spike counts into 100 bins of 20 ms for each trial interval.

ii. 
```python
valid_channels = np.clip(cluster_channels, 0, len(channel_brain_ids) - 1)
cluster_brain_ids = channel_brain_ids[valid_channels]
cluster_acronyms = br.id2acronym(cluster_brain_ids)
beryl_regions = br.acronym2acronym(cluster_acronyms, mapping='Beryl')

spike_clusters_offset = spike_clusters + cluster_offset
cluster_offset += n_clusters
...
sort_idx = np.argsort(merged_times, kind='stable')
...
bin_idx = np.minimum(
    ((times_trial - t_beg) / BINSIZE).astype(np.int32),
    N_BINS - 1
)
flat_idx = clusters_trial * N_BINS + bin_idx
counts = np.bincount(flat_idx, minlength=n_clusters * N_BINS)
```

iii. The notes justify this as matching `merge_probes`, `list_brain_regions`, and `bin_spiking_data` from the reference code while using a faster local implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not neuron-QC filtered. The script keeps all clusters it can load from the probes and does not apply a “good units only” threshold.

ii. 
```python
def load_spikes(sdir):
    """Load and merge spikes from all probes in a session.

    Following reference code: no QC filtering (qc=None).
    """
```

iii. The notes repeatedly justify this with the reference code’s `load_spiking_data(..., qc=None)` behavior, which the trajectory excerpt from `ibl_data_utils.py` confirms.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial is aligned to `stimOn_times`. The script constructs a fixed window from 0.5 s before to 1.5 s after stimulus onset, then bins spikes within that interval.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

stim_on = trials[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. The notes say the agent noticed the methods paper aligned wheel and whisker to first movement, but chose stimulus onset because `0_data_caching.py` used `stimOn_times` and the decoder task explicitly said “Temporally align based on stimulus onset.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use uniform 20 ms bins for all signals, giving 100 bins over the 2 s window. No later temporal rebinning is applied.

ii. 
```python
BINSIZE = 0.02  # 20ms
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0s
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
...
'time_bin_size': BINSIZE * 1000,  # 20ms
'n_time_bins': N_BINS,
```

iii. The notes say this choice was made to match `0_data_caching.py`, even though the methods text quoted in the notes says choice and prior were described with 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read as a raw recorded signal. It is derived from the chosen alignment event (`stimOn_times`) together with the fixed window and bin size constants.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
...
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. The notes describe this as “Time since stimulus onset: center of each 20 ms bin,” which is an invented decoder input required by the target format rather than a field from the source code.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script computes one vector of bin centers from -0.49 s to 1.49 s using `np.linspace`, then copies that same vector into every trial’s input array.

ii. 
```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
...
inp = np.stack([
    time_input,
    np.full(N_BINS, trial_num_in_block[i], dtype=np.float32)
], axis=0)
```

iii. The notes justify the bin-center convention by saying the input should correspond to the 20 ms bins used for neural and behavioral time series.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned exactly to the neural data because the vector length, bin size, and start/end offsets are built from the same `TIME_WINDOW`, `BINSIZE`, and `N_BINS` used for spike binning.

ii. 
```python
bin_idx = np.minimum(
    ((times_trial - t_beg) / BINSIZE).astype(np.int32),
    N_BINS - 1
)
...
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
)
```

iii. The notes say the time input was intended to be “the center of each 20 ms bin,” so it tracks the trial-aligned neural bins directly.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial-table column `probabilityLeft`.

ii. 
```python
prob_left = trials_good['probabilityLeft'].values
trial_num_in_block = compute_trial_num_in_block(prob_left)
```

iii. The notes explicitly map “Trial number in block” to “From `probabilityLeft`,” using block probability changes to identify block boundaries.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script scans the filtered trials in order and increments the counter while consecutive `probabilityLeft` values stay equal; the counter resets to 1 whenever the probability changes.

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
```

iii. The notes justify this with “Block trial number: computed as position within contiguous block of same `probabilityLeft`.” There is no indication that the agent compared this derived feature to a reference implementation.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the trial-table `choice` column.

ii. 
```python
choice = trials_good['choice'].values.copy()
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
```

iii. The notes map `choice` directly from the trials table and describe the intended recoding as left `-1 -> 0`, right `1 -> 1`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After filtering, the script converts the IBL coding to binary by assigning rightward choices (`1`) to category `1` and everything else that survived filtering to category `0`, which effectively makes leftward choice (`-1`) equal to `0` because `choice == 0` trials were excluded earlier.

ii. 
```python
mask &= (trials['choice'] != 0)
...
choice = trials_good['choice'].values.copy()
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
...
np.full(N_BINS, choice_binary[i], dtype=np.int64)
```

iii. The notes explicitly justify this as “Choice: `-1 (left) -> 0`, `1 (right) -> 1`,” matching the decoder task.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial-table `probabilityLeft` column.

ii. 
```python
prob_left = trials_good['probabilityLeft'].values
prior = np.full(len(prob_left), 1, dtype=np.int32)
prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
```

iii. The notes say “Prior/Block = `probabilityLeft`” and map it to the required categories from the decoder task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The script keeps a default class of `1` for 0.5 blocks, maps values near 0.2 to `0`, and values near 0.8 to `2` using `np.isclose(..., atol=0.05)`. It then repeats that class across all time bins in the trial.

ii. 
```python
prior = np.full(len(prob_left), 1, dtype=np.int32)  # default 0.5->1
prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
...
np.full(N_BINS, prior[i], dtype=np.int64)
```

iii. The notes justify this as the required decoder mapping: `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
wh_pos = np.load(os.path.join(sdir, 'alf/_ibl_wheel.position.npy')).flatten()
wh_times = np.load(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy')).flatten()
```

iii. The notes say the reference behavior loader uses `SessionLoader.load_wheel()`, but the agent re-derived the behavior from the local raw wheel position and timestamps.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The script interpolates wheel position to a 1 kHz uniform grid, differentiates it with `np.gradient`, and takes the absolute value to get speed. It then linearly interpolates that continuous speed trace onto each trial’s 20 ms bins.

ii. 
```python
dt = 0.001  # 1kHz
t_uniform = np.arange(wh_times[0], wh_times[-1], dt)
pos_interp = np.interp(t_uniform, wh_times, wh_pos)
velocity = np.gradient(pos_interp, dt)
speed = np.abs(velocity)
...
wheel_vals, wheel_mask = interpolate_behavior_to_bins(
    wh_times, wh_speed, interval_begs, interval_ends
)
```

iii. The notes justify this as an attempted replication of `SessionLoader.load_wheel()`, and the trajectory excerpt shows the agent knew the reference loader computed wheel velocity from an interpolated trace with Gaussian smoothing.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Continuous wheel speed is discretized into three categories using session-wide quantile boundaries, effectively terciles across all valid wheel-speed samples in the session.

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

iii. The notes say “Use session-wide terciles for wheel speed and whisker ME,” which was the agent’s own discretization choice because the reference decoder treated these as continuous targets.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset trial intervals as the neural data. For each trial, the script interpolates wheel speed at `linspace(t_beg + binsize, t_end, N_BINS)` over the stim-on window.

ii. 
```python
wheel_vals, wheel_mask = interpolate_behavior_to_bins(
    wh_times, wh_speed, interval_begs, interval_ends
)
...
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
interp_func = interp1d(beh_t, beh_v, kind='linear', fill_value='extrapolate')
values[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The notes explicitly acknowledge that the methods paper described first-movement alignment for wheel speed, but say the agent chose stimulus-onset alignment because the decoder task and `0_data_caching.py` used that alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from either `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, or, if left-camera data are unavailable, the corresponding right-camera files.

ii. 
```python
left_me_files, left_time_files = _find_files(
    sdir, 'leftCamera.ROIMotionEnergy.npy', '_ibl_leftCamera.times.npy')
...
right_me_files, right_time_files = _find_files(
    sdir, 'rightCamera.ROIMotionEnergy.npy', '_ibl_rightCamera.times.npy')
```

iii. The notes say the reference code tries left camera first and falls back to right, and the trajectory around steps 128 to 129 shows the agent debugging path-layout differences for those files.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads the selected camera’s motion-energy trace and timestamps, truncates both to the shorter length if needed, and linearly interpolates the continuous trace into the per-trial 20 ms bins.

ii. 
```python
me = np.load(left_me_files[-1]).flatten()
times = np.load(left_time_files[-1]).flatten()
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len]
...
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
```

iii. The notes justify this as matching `load_target_behavior` and `get_behavior_per_interval`, while also handling inconsistent on-disk layouts by checking both dated subdirectories and the `alf/` root.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is discretized into three bins using the same session-wide quantile procedure used for wheel speed.

ii. 
```python
whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
...
boundaries = np.percentile(flat, quantiles)
result = np.digitize(values, boundaries).astype(np.int32)
```

iii. The notes explicitly say “Use session-wide terciles for wheel speed and whisker ME.”

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to the same stimulus-onset trial window as the neural data and wheel speed, using the same per-trial interpolation routine and the same 100 20 ms bins.

ii. 
```python
me_times, me_vals_raw = load_whisker_me(sdir)
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
...
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
```

iii. The notes say the agent chose stim-on alignment for all behaviors despite the methods-paper mismatch because that was the common alignment in the task instructions and in `0_data_caching.py`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles minor data problems mostly by skipping. Entire sessions are excluded up front if required files are missing, and `process_session()` returns `None` on any exception or if fewer than two good trials remain. At the trial level, wheel and whisker intervals are masked out if timestamps do not cover the full interval, and whisker traces are silently truncated to `min(len(me), len(times))` when lengths disagree.

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
...
except Exception as e:
    ...
    return None
```

iii. The trajectory shows the agent explicitly debugging missing whisker files and deciding to make session selection stricter. The notes also say the converter “gracefully handle[s] missing data (skips sessions without wheel/whisker data).”

## 10-a. What are the most time-consuming steps of the code?

i. The main bottlenecks are per-session spike binning across all neurons and trials, and the two passes of per-trial behavioral interpolation for wheel speed and whisker motion energy.

ii. 
```python
print(f"  Binning spikes ({n_clusters} neurons, {len(trials)} trials)...", flush=True)
binned_spikes = bin_spikes_vectorized(...)
...
wheel_vals, wheel_mask = interpolate_behavior_to_bins(...)
...
whisker_vals, whisker_mask = interpolate_behavior_to_bins(...)
```

iii. The notes and trajectory repeatedly discuss spike binning as the expensive step, including the searchsorted-based optimization and later memory reductions to finish the full run.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious remaining loops are the per-trial loop in `bin_spikes_vectorized()`, the per-trial loop in `interpolate_behavior_to_bins()`, and the loops that build `input_list` and `output_list` one trial at a time.

ii. 
```python
for trial_idx in range(n_trials):
    ...

for trial_idx in range(n_trials):
    ...

for i in range(n_trials):
    inp = np.stack([...], axis=0)
    input_list.append(inp)

for i in range(n_trials):
    out = np.stack([...], axis=0)
    output_list.append(out)
```

iii. The trajectory shows the agent focused on speeding up spike binning with `searchsorted` and `bincount`, but it left several trial loops intact for behavior interpolation and output assembly.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly searches and slices timestamps separately for every trial and for every behavior stream, rebuilds constant choice/prior/time arrays for every trial, and processes all trials before discarding masked ones.

ii. 
```python
for trial_idx in range(n_trials):
    i_start = np.searchsorted(spike_times, t_beg, side='left')
    i_end = np.searchsorted(spike_times, t_end, side='left')
...
for trial_idx in range(n_trials):
    idx_beg = np.searchsorted(beh_times, t_beg, side='right')
    idx_end = np.searchsorted(beh_times, t_end, side='left')
...
np.full(N_BINS, choice_binary[i], dtype=np.int64)
np.full(N_BINS, prior[i], dtype=np.int64)
```

iii. The notes describe the implementation as an incremental, session-by-session converter, but do not claim that these repeated per-trial operations were minimized.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins spikes for all trials before applying the trial mask, computes continuous wheel and whisker traces only to discard them after discretization, and keeps optional plotting code that is not needed for downstream decoder training.

ii. 
```python
# 4. Bin spikes for ALL trials first (before masking)
binned_spikes = bin_spikes_vectorized(...)
...
wheel_trials = wheel_vals[good_indices]
whisker_trials = whisker_vals[good_indices]
...
wheel_discrete = discretize_to_bins(wheel_trials, n_bins=3)
whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
...
if show_processing:
    plot_processing(...)
```

iii. The notes justify the “bin all trials first” choice as the simplest way to mirror the reference flow before applying the combined mask, but that work is partly thrown away once invalid trials are removed.
