# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the `bwm_release.csv` file from the reference code directory to enumerate all sessions (PIDs), then constructs file paths directly by combining lab, subject, date, and session number fields. It does NOT use the ONE API for session discovery or data loading. Instead, it manually finds session directories with `find_session_path()` and loads `.npy` and `.pqt` files directly, handling versioned subdirectories with `find_versioned_file()`.

ii.
```python
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'
bwm_df = pd.read_csv(BWM_CSV, index_col=0)

def find_session_path(lab, subject, date):
    for sess_num in ['001', '002', '003']:
        p = BASE_PATH / lab / 'Subjects' / subject / date / sess_num / 'alf'
        if p.exists():
            return p
    return None
```

iii. The AI chose to bypass the ONE API and load files directly from the file system to avoid API dependency issues. The CONVERSION_NOTES document "Direct file loading (no SpikeSortingLoader dependency)" as a deliberate design choice.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of `bwm_release.csv`. Sessions are grouped by `eid`, and each session's subject comes from the CSV. Unique subjects are sorted alphabetically and indexed.

ii.
```python
for _, row in bwm_df.iterrows():
    eid = row.eid
    sessions[eid].append({
        'subject': row.subject,
        ...
    })
all_subjects = sorted(list(set(sess['subject'] for sess in session_results)))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. The CSV provides subject names directly.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in `bwm_release.csv`. The AI groups probes by eid, then processes each eid as one session.

ii.
```python
sessions = {}
for _, row in bwm_df.iterrows():
    eid = row.eid
    if eid not in sessions:
        sessions[eid] = []
    sessions[eid].append({...})
```

iii. The CSV already lists sessions by eid.

## 1-d. How are the data split into trials?

i. Each session's trials are loaded from the `_ibl_trials.table.pqt` parquet file, where each row is one trial.

ii.
```python
trials_file = find_versioned_file(sess_path, '_ibl_trials.table.pqt')
trials_df = pd.read_parquet(trials_file)
```

iii. The trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: (1) reaction time between 0.08s and 2.0s, (2) trial length (feedback_times - goCue_times) <= 10s, (3) NaN exclusion on stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType, (4) exclusion of no-choice trials (choice != 0), (5) behavioral data coverage check via interpolate_behavior validity mask.

ii.
```python
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)

if MAX_TRIAL_LEN is not None:
    if 'goCue_times' in trials_df.columns:
        trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN)

for event in NAN_EXCLUDE:
    if event in trials_df.columns:
        mask &= ~trials_df[event].isna()

mask &= (trials_df['choice'] != 0)
```

iii. The AI's CONVERSION_NOTES state this matches the reference code's `load_trials_and_mask` defaults and `prepare_data` call parameters. The AI adds NaN exclusion on additional fields beyond what the reference solution checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike times (`spikes.times.npy`) and spike cluster assignments (`spikes.clusters.npy`) from each probe, plus `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` for brain region mapping.

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
cluster_channels = np.load(spike_dir / 'clusters.channels.npy').flatten()
chan_brain_ids = np.load(spike_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
```

iii. The spike data is loaded directly from files rather than through the SpikeSortingLoader API.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20ms bins over the trial window (-0.5 to 1.5s relative to stimOn_times). The spike counts are stored directly as float32 WITHOUT converting to firing rates (no division by bin width). Multi-probe sessions have their probes merged with cluster IDs offset to create a continuous neuron index, then sorted by time.

ii.
```python
def bin_spikes_fast(spike_times, spike_clusters, interval_starts, interval_ends,
                    n_clusters_total, binsize, n_bins):
    binned = np.zeros((n_trials, n_clusters_total, n_bins), dtype=np.float32)
    for trial_idx in range(n_trials):
        bin_idx = np.minimum(np.floor((trial_times - t_start) / binsize).astype(int), n_bins - 1)
        np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
    return binned

# In build_final_dataset:
session_neural.append(sess['binned_spikes'][trial_idx].astype(np.float32))
```

iii. The CONVERSION_NOTES state "Spikes binned at 20ms, aligned to stimOn_times." There is no mention of converting to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies NO quality control filtering on clusters. All clusters from the spike sorting are included regardless of their quality label. Void brain regions are also not excluded.

ii.
```python
def load_spike_data(sess_path, probe_name):
    spike_times = np.load(spike_times_file).flatten()
    spike_clusters = np.load(spike_clusters_file).flatten()
    # No filtering step - all clusters are kept
    return spike_times, spike_clusters, n_clusters, cluster_brain_ids
```

iii. The CONVERSION_NOTES explicitly state "No QC filtering (all clusters used)" and "No QC filtering: load_spiking_data called with qc=None in prepare_data" and "QC filtering: Use qc=None as in reference code". The AI followed the reference code's `prepare_data` function which calls `load_spiking_data` with `qc=None`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike window is defined by stimOn_times + TIME_WINDOW[0] and stimOn_times + TIME_WINDOW[1], i.e. -0.5 to 1.5s relative to stimulus onset. Spikes within this window are binned, with bin 0 starting at -0.5s.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]

bin_idx = np.minimum(np.floor((trial_times - t_start) / binsize).astype(int), n_bins - 1)
```

iii. The alignment matches the reference code parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20ms bins, 100 bins per trial spanning the 2s window. No rebinning is applied.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
```

iii. Matches the reference code parameter `binsize=0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the trial window parameters (TIME_WINDOW and BINSIZE), not from any raw data variable. It is a synthetic time axis.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The AI's CONVERSION_NOTES say the time values match "the bin centers used in reference code."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are computed as `np.linspace(-0.48, 1.5, 100)`, producing values from -0.48 to 1.50 in steps of 0.02. This matches the reference code's `get_behavior_per_interval` interpolation grid, but differs from the bin centers which would be -0.49, -0.47, ..., 1.49.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
# Produces: [-0.48, -0.46, -0.44, ..., 1.48, 1.50]
```

iii. The AI followed the `get_behavior_per_interval` convention from the reference code rather than computing true bin centers (edge + BIN/2).

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time values are offset from the neural bin centers by half a bin width (0.01s). Neural bins are centered at -0.49, -0.47, ..., 1.49, while the input time values are at -0.48, -0.46, ..., 1.50.

ii.
```python
# Neural bins: floor((spike_time - t_start) / binsize) puts spikes into bins
# starting at t_start = stim_on - 0.5, bin edges at [-0.50, -0.48, ..., 1.48, 1.50]
# True bin centers: [-0.49, -0.47, ..., 1.49]

# Input time: linspace(-0.48, 1.5, 100) = [-0.48, -0.46, ..., 1.50]
# These are the RIGHT edges of bins, not centers
```

iii. The AI does not discuss this offset.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. A block is defined as consecutive trials with the same probabilityLeft value.

ii.
```python
def compute_trial_number_in_block(prob_left):
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

iii. The approach of detecting block boundaries from changes in probabilityLeft is the same concept as the reference.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Two key differences from the reference: (1) The AI counts starting from 1, not 0. (2) The AI computes trial_number_in_block on the FILTERED trials (`valid_trials_final`), not the full trials table. This means block boundaries may be incorrectly detected if a trial at a block transition was filtered out, and trial counts within blocks will be wrong since filtered trials don't advance the count.

ii.
```python
# Computed on filtered trials (after mask application):
trial_num_in_block = compute_trial_number_in_block(
    valid_trials_final['probabilityLeft']    # <-- filtered trials, not all trials
)

# Counts from 1:
count = 0
for i in range(len(prob_left)):
    if val == current_block:
        count += 1  # starts at 1 for first trial
    else:
        current_block = val
        count = 1
```

iii. The CONVERSION_NOTES do not discuss whether block counting should be computed before or after trial filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, where IBL convention is +1 for left, -1 for right, 0 for no-go.

ii.
```python
choice = valid_trials_final['choice'].values.copy()
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
```

iii. The AI references the IBL choice encoding.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice=-1 (right) to 0 and choice=1 (left) to 1. This is INVERTED from the instructions which specify "left = 0, right = 1". The AI's mapping produces left=1, right=0.

ii.
```python
# IBL: choice=1 means left, choice=-1 means right
# Instructions: left=0, right=1
# AI mapping (INVERTED):
choice_binary = np.where(choice == -1, 0, 1)
# Result: left(1) -> 1, right(-1) -> 0
```

iii. The CONVERSION_NOTES state "left(-1)->0, right(1)->1" in the variable mapping table, but this describes the IBL convention direction wrongly. In IBL, -1 is right (not left).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prob_left = valid_trials_final['probabilityLeft'].values.copy()
prior_cat = np.zeros(len(prob_left), dtype=np.float32)
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
```

iii. Matches the instruction mapping: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct mapping of the three values to categories 0, 1, 2. No additional processing.

ii. Same as 6-a.

iii. N/A

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. Same raw data as the reference.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI re-implements the wheel processing matching `SessionLoader.load_wheel()`: (1) interpolate position to 1000Hz uniform sampling, (2) apply order-8 Butterworth low-pass filter at 20Hz, (3) differentiate to get velocity, (4) take absolute value for speed. The wheel speed is then interpolated to trial time bins using `interp1d` with `linspace(t_start + binsize, t_end, n_bins)`.

ii.
```python
t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
pos_interp = interp1d(timestamps, position, kind='linear')(t_uniform)
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
```

iii. The CONVERSION_NOTES confirm matching SessionLoader.load_wheel().

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI discretizes wheel speed using GLOBAL quantiles (33rd and 67th percentiles across ALL sessions), not per-session percentiles as the reference does.

ii.
```python
# Global discretization across all sessions:
all_wheel = np.concatenate(all_wheel)
wheel_quantiles = np.array([-np.inf,
                             np.quantile(all_wheel, 1/3),
                             np.quantile(all_wheel, 2/3),
                             np.inf])
wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1])
```

iii. The CONVERSION_NOTES mention "Global quantile-based discretization for wheel speed and whisker ME" but don't justify why global was chosen over per-session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated at `np.linspace(t_start + binsize, t_end, n_bins)` which corresponds to the same shifted time grid as the input time values (-0.48, -0.46, ..., 1.50), offset by half a bin from the neural bin centers.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
result[trial_idx] = f_interp(x_interp)
```

iii. The AI follows the reference code's `get_behavior_per_interval` interpolation convention.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy`, with corresponding `_ibl_<side>Camera.times.npy`.

ii.
```python
me_file = find_versioned_file(sess_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_versioned_file(sess_path, '_ibl_leftCamera.times.npy')
if me_file is None or times_file is None:
    me_file = find_versioned_file(sess_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_versioned_file(sess_path, '_ibl_rightCamera.times.npy')
```

iii. Left camera preferred, falling back to right camera. Matches reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded and NaN values are removed. The trace is then interpolated to trial time bins using the same `interpolate_behavior` function as wheel speed, using `interp1d` with `linspace(t_start + binsize, t_end, n_bins)`.

ii.
```python
me_values = np.load(me_file).flatten()
me_times = np.load(times_file).flatten()
valid = ~np.isnan(me_values) & ~np.isnan(me_times)
me_values = me_values[valid]
me_times = me_times[valid]

me_binned, me_valid = interpolate_behavior(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. No additional filtering or normalization beyond NaN removal.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: GLOBAL quantiles (33rd and 67th percentiles across all sessions) rather than per-session percentiles.

ii.
```python
all_me = np.concatenate(all_me)
me_quantiles = np.array([-np.inf,
                          np.quantile(all_me, 1/3),
                          np.quantile(all_me, 2/3),
                          np.inf])
me_disc = np.digitize(sess['me_binned'], me_quantiles[1:-1])
```

iii. Same justification as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated at the shifted time grid (linspace), offset by half a bin from neural bin centers.

ii.
```python
me_binned, me_valid = interpolate_behavior(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. Same approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of handling: (1) Sessions where probe data cannot be loaded are skipped. (2) Sessions without wheel or whisker ME data are skipped. (3) Trials where behavioral interpolation fails (insufficient coverage) are marked invalid. (4) Sessions with fewer than 2 valid trials are skipped. (5) NaN values in motion energy are pre-filtered.

ii.
```python
if result[0] is not None:
    probe_data.append(result)
else:
    print(f'  Session {eid}: failed to load probe {probe_name}')

combined_valid = wheel_valid & me_valid
if np.sum(combined_valid) < 2:
    return None

valid = ~np.isnan(me_values) & ~np.isnan(me_times)
```

iii. The approach drops sessions/trials with missing data rather than imputing.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies spike binning and wheel processing as time-consuming, reporting timing for each in the output. No parallelism is used - sessions are processed sequentially in a for loop.

ii.
```python
for i, (eid, probes) in enumerate(available_sessions.items()):
    result = process_session(eid, probes, show_processing=args.show_processing)
```

iii. The CONVERSION_NOTES report ~4s per session average, with a full conversion taking ~35.5 minutes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning function `bin_spikes_fast` loops over trials and uses `np.add.at` for accumulation, which is reasonably efficient. The `interpolate_behavior` function loops over trials with `interp1d` calls. The `compute_trial_number_in_block` function has a Python loop over every trial.

ii.
```python
# Trial loop in bin_spikes_fast
for trial_idx in range(n_trials):
    np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)

# Trial loop in interpolate_behavior
for trial_idx in range(n_trials):
    f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
```

iii. The CONVERSION_NOTES don't discuss vectorization opportunities in detail.

## 10-c. What processing does the code repeat multiple times?

i. The AI instantiates a new `BrainRegions()` object inside `process_session` for every session, repeating the atlas loading. The `find_versioned_file` function is called multiple times per session.

ii.
```python
def process_session(eid, probes_info, show_processing=False):
    ...
    br = BrainRegions()  # Created fresh for each session
```

iii. Not discussed in CONVERSION_NOTES.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads additional trial timing files (stimOnTrigger_times, goCueTrigger_times) that are not used for any of the decoder inputs or outputs. It also applies a max_trial_len filter based on feedback_times and goCue_times that is not relevant to the decoder task.

ii.
```python
stim_trig_file = find_versioned_file(sess_path, '_ibl_trials.stimOnTrigger_times.npy')
if stim_trig_file is not None:
    trials_df['stimOnTrigger_times'] = np.load(stim_trig_file)

go_trig_file = find_versioned_file(sess_path, '_ibl_trials.goCueTrigger_times.npy')
```

iii. Not discussed in CONVERSION_NOTES.
