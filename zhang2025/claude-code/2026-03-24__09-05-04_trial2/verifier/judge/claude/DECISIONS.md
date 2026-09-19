# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by directly scanning the filesystem using glob patterns to find session directories under `data/one_cache/*/Subjects/*/*/001`. It checks for the existence of required files (spikes, trials, wheel, motion energy) and collects valid session directories. It does NOT use the ONE API or the IBL release index.

ii.
```python
def find_session_dirs():
    """Find all session directories with required data."""
    session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
    valid = []
    for sdir in session_dirs:
        has_spikes = len(glob.glob(os.path.join(sdir, 'alf/probe*/pykilosort/*/spikes.times.npy'))) > 0
        has_trials = len(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt'))) > 0
        has_wheel = os.path.exists(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy'))
        has_me = (len(glob.glob(os.path.join(sdir, 'alf/*/leftCamera.ROIMotionEnergy.npy'))) > 0 or
                  len(glob.glob(os.path.join(sdir, 'alf/*/rightCamera.ROIMotionEnergy.npy'))) > 0)
        ...
```

iii. The AI chose to directly scan the filesystem rather than use the ONE API, noting it found 393 sessions with complete data. It did not use `DATALIMIT_SUBSET.csv` to restrict sessions. The CONVERSION_NOTES state the data is organized under `data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf/`.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the directory path structure, parsing the path component after "Subjects". Subjects are collected into a list as sessions are processed, and `subject_idx` maps each session to its position in the subject list.

ii.
```python
def parse_session_info(sdir):
    parts = Path(sdir).parts
    sub_idx = parts.index('Subjects')
    subject = parts[sub_idx + 1]
    ...
```

```python
if subject not in all_subjects:
    all_subjects.append(subject)
subject_per_session.append(subject)
```

iii. The AI parses subject names from directory paths since it loads data via filesystem rather than ONE API. The ordering depends on glob sort order rather than alphabetical sorting of subject names.

## 1-c. How are the data split into sessions?

i. Each directory at the `*/Subjects/<subject>/<date>/001` level is treated as a session. Sessions are discovered by globbing and processed sequentially.

ii.
```python
session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
```

iii. Sessions correspond to directories, same as the natural structure of the IBL data cache.

## 1-d. How are the data split into trials?

i. Trials are loaded from the parquet trials table, one row per trial. The trial table is loaded from `_ibl_trials.table.pqt`.

ii.
```python
def load_trials(sdir):
    trial_files = sorted(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt')))
    trials = pd.read_parquet(trial_files[-1])
    return trials
```

iii. The trial structure is given by the data; each row in the trials table is one trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: (1) reaction time between 0.08-2.0s, (2) exclude no-choice trials (choice==0), (3) exclude trials with NaN in key columns (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (4) trial length filter (goCue to feedback <= 10s), and (5) behavioral coverage (wheel and whisker must have data in the trial window).

ii.
```python
NAN_EXCLUDE = [
    'stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType'
]

def create_trial_mask(trials):
    mask = pd.Series(True, index=trials.index)
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
    if 'goCue_times' in trials.columns and 'feedback_times' in trials.columns:
        trial_len = trials['feedback_times'] - trials['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
    mask &= (trials['choice'] != 0)
    for col in NAN_EXCLUDE:
        if col in trials.columns:
            mask &= ~trials[col].isna()
    return mask
```

iii. The AI notes that these filters follow the reference code's `load_trials_and_mask` function. The CONVERSION_NOTES document: "RT filter", "no-choice exclusion", "NaN exclusion" and "max_trial_len=10.0".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike times (`spikes.times.npy`) and spike cluster assignments (`spikes.clusters.npy`) from each probe directory, plus `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` for brain region mapping.

ii.
```python
spike_times = np.load(st_file).flatten()
spike_clusters = np.load(sc_file).flatten()
cluster_channels = np.load(cc_file).flatten()
channel_brain_ids = np.load(cb_file).flatten()
```

iii. The AI loads spike data directly from numpy files rather than through SpikeSortingLoader.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms bins over a 2s trial window (-0.5 to 1.5s around stimulus onset), producing 100 time bins per trial. Multiple probes are merged by offsetting cluster IDs. The result is stored as raw spike counts (uint8), NOT converted to firing rates.

ii.
```python
bin_idx = np.minimum(
    ((times_trial - t_beg) / BINSIZE).astype(np.int32),
    N_BINS - 1
)
flat_idx = clusters_trial * N_BINS + bin_idx
counts = np.bincount(flat_idx, minlength=n_clusters * N_BINS)
binned[trial_idx] = counts[:n_clusters * N_BINS].reshape(n_clusters, N_BINS)
```

```python
neural_list = [np.clip(neural_trials[i], 0, 255).astype(np.uint8) for i in range(n_trials)]
```

iii. The AI stores spike counts as uint8 to save memory, noting "spike counts rarely exceed 255." The data is NOT divided by bin width to produce firing rates, unlike the reference which divides by BIN (0.02) to get Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies NO quality control filtering on neurons. All clusters are included regardless of their quality label.

ii.
```python
def load_spikes(sdir):
    """Load and merge spikes from all probes in a session.
    Following reference code: no QC filtering (qc=None).
    """
    # No label/QC check here - all clusters are kept
    ...
```

iii. The AI explicitly justifies this in CONVERSION_NOTES: "No QC filter applied in prepare_data (qc=None by default). ALL neurons used, not just good ones (label >= 1 not required)." The AI follows the reference code's `prepare_data` default parameter (qc=None) rather than the data paper's quality control.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `stimOn_times`. The interval for each trial runs from `stimOn_times - 0.5` to `stimOn_times + 1.5`.

ii.
```python
stim_on = trials[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. Alignment to stimulus onset matches both the reference code and the decoder task instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20ms bins (BINSIZE=0.02), producing 100 time steps per 2s trial. No rebinning is applied.

ii.
```python
BINSIZE = 0.02  # 20ms
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. Matches the reference code's binsize of 0.02.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the bin grid itself: centers of 100 evenly spaced 20ms bins in the window (-0.5, 1.5)s.

ii.
```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. The time input is a deterministic array defined by the binning parameters, identical for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing; the array is computed as `linspace(-0.49, 1.49, 100)`, which are the centers of the 20ms bins.

ii.
```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. The AI uses `linspace` to generate evenly spaced bin centers. The values are the same as `EDGES[:-1] + BIN/2` in the reference.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same bin centers as the neural data, so they are aligned by construction.

ii.
```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. Both share the same 100-bin time grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. A block boundary is detected when `probabilityLeft` changes.

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

iii. The AI detects block boundaries from changes in probabilityLeft and counts the position within each block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number counts from 1 (first trial in a block is 1). It is computed on the filtered trials only (post-mask), not on the full trial table. This means that filtered-out trials do not advance the count.

ii.
```python
# In process_session:
trials_good = trials.iloc[good_indices]
prob_left = trials_good['probabilityLeft'].values
trial_num_in_block = compute_trial_num_in_block(prob_left)
```

iii. The AI computes block trial number AFTER filtering, meaning the count reflects only the surviving trials. The reference computes this BEFORE filtering, so the count reflects the animal's actual position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table.

ii.
```python
choice = trials_good['choice'].values.copy()
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
```

iii. The AI uses the `choice` column directly.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice values: where choice==1 (IBL convention: LEFT), set to 1; otherwise (including -1, which is RIGHT), set to 0. This produces left=1, right=0.

ii.
```python
# Choice: -1 (left) -> 0, 1 (right) -> 1
choice = trials_good['choice'].values.copy()
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
```

iii. The AI's comment says "-1 (left) -> 0, 1 (right) -> 1" but this is WRONG about IBL conventions. In IBL, +1 = left and -1 = right. The actual mapping implemented is +1 (left) -> 1, -1 (right) -> 0, which is the INVERSE of the instructions ("left = 0, right = 1").

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table.

ii.
```python
prob_left = trials_good['probabilityLeft'].values
prior = np.full(len(prob_left), 1, dtype=np.int32)  # default 0.5->1
prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
```

iii. Maps probabilityLeft to categorical values using approximate matching.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Uses `np.isclose` with atol=0.05 to match the three probability values (0.2, 0.5, 0.8) to categories (0, 1, 2). The default is 1 (0.5).

ii.
```python
prior = np.full(len(prob_left), 1, dtype=np.int32)
prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
```

iii. The reference uses exact mapping via `.map({0.2: 0, 0.5: 1, 0.8: 2})`. The AI uses approximate matching, which is more tolerant of floating point imprecision but could misclassify edge cases.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
wh_pos = np.load(os.path.join(sdir, 'alf/_ibl_wheel.position.npy')).flatten()
wh_times = np.load(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy')).flatten()
```

iii. The AI loads wheel position and timestamps directly rather than through SessionLoader.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates position to 1kHz, computes velocity via `np.gradient` (finite differences), then takes the absolute value. This differs from the reference which uses SessionLoader's Butterworth low-pass filter. The continuous speed is then interpolated to trial time bins using `interp1d` with a different time grid, and discretized into 3 bins using session-wide tercile percentiles.

ii.
```python
t_uniform = np.arange(wh_times[0], wh_times[-1], dt)
pos_interp = np.interp(t_uniform, wh_times, wh_pos)
velocity = np.gradient(pos_interp, dt)
speed = np.abs(velocity)
```

```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
interp_func = interp1d(beh_t, beh_v, kind='linear', fill_value='extrapolate')
values[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The AI acknowledges the reference uses SessionLoader's Butterworth filter but reimplements wheel processing from scratch using finite differences. The CONVERSION_NOTES state: "Wheel speed: absolute value of wheel velocity from SessionLoader" but the code does NOT use SessionLoader.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 bins using tercile percentiles (33.3% and 66.7%) computed across the entire session (all trials and time bins of the session pooled together).

ii.
```python
def discretize_to_bins(values, n_bins=3):
    flat = values[~np.isnan(values)].flatten()
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [33.3, 66.7]
    boundaries = np.percentile(flat, quantiles)
    result = np.digitize(values, boundaries).astype(np.int32)
    return result
```

iii. This uses session-wide percentiles for equal-sized bins, matching the reference approach.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to the trial time grid using `np.linspace(t_beg + BINSIZE, t_end, N_BINS)` where `t_beg = stimOn + T_START` and `t_end = stimOn + T_STOP`. This gives time points from `t_beg + 0.02` to `t_end` (i.e., -0.48 to 1.5s relative to stimulus onset).

ii.
```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
```

iii. This time grid does NOT match the neural data's time grid. The neural bins start at T_START (-0.5) while the behavior interpolation starts at T_START + BINSIZE (-0.48). The reference uses `onset + TIME` (bin centers at -0.49 to 1.49) for behavior interpolation, matching the neural binning exactly.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` with corresponding `_ibl_<side>Camera.times.npy`.

ii.
```python
left_me_files, left_time_files = _find_files(
    sdir, 'leftCamera.ROIMotionEnergy.npy', '_ibl_leftCamera.times.npy')
if left_me_files and left_time_files:
    me = np.load(left_me_files[-1]).flatten()
    times = np.load(left_time_files[-1]).flatten()
```

iii. Left camera is preferred, falling back to right camera, matching the reference logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are used as-is (no filtering or normalization). They are interpolated to trial time bins using the same `interp1d` approach as wheel speed, then discretized into 3 bins using session-wide tercile percentiles.

ii.
```python
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
```

iii. Same processing pipeline as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: 3 bins via session-wide tercile percentiles (33.3%, 66.7%).

ii.
```python
whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
```

iii. Matches the reference approach of equal-sized bins via percentiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated to `np.linspace(t_beg + BINSIZE, t_end, N_BINS)`, which is offset from the neural time grid by half a bin.

ii.
```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
```

iii. Same misalignment issue as wheel speed (7-d).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions without required files (spikes, trials, wheel, motion energy) are excluded during directory scanning. Trials with NaN in key columns are filtered out. Sessions with fewer than 2 valid trials are skipped. Behavioral coverage is checked (if wheel/whisker data doesn't cover the trial window, the trial is excluded). Exceptions during session processing are caught and the session is skipped.

ii.
```python
if len(good_indices) < 2:
    print(f"  WARNING: Only {len(good_indices)} valid trials, skipping session", flush=True)
    return None
```

```python
except Exception as e:
    print(f"  ERROR processing {session_id}: {e}", flush=True)
    return None
```

iii. The AI handles missing data by skipping problematic sessions/trials. 57 sessions were skipped due to missing wheel data (PL050/hausserlab).

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike data from disk and spike binning. The CONVERSION_NOTES report spike binning takes 0.2-0.3s per session and total per-session processing is ~2.5s. Total conversion of 393 sessions took approximately 16 minutes.

ii.
```python
spike_times = np.load(st_file).flatten()
spike_clusters = np.load(sc_file).flatten()
```

iii. The AI measured timing: spike binning 0.2-0.3s, wheel/whisker loading 0.5-1s, total ~2.5s per session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates per-trial, and the behavior interpolation loop also iterates per-trial. Both could potentially be vectorized. The `compute_trial_num_in_block` function also uses a Python loop.

ii.
```python
for trial_idx in range(n_trials):
    # ... spike binning per trial
```

```python
for trial_idx in range(n_trials):
    # ... behavior interpolation per trial
```

iii. The AI acknowledged this by documenting timing per step but did not vectorize these loops.

## 10-c. What processing does the code repeat multiple times?

i. Spike binning is done for ALL trials (before filtering), then the mask is applied afterward. This means spikes are binned for trials that will be discarded. The AI bins spikes first, then applies the trial quality mask, wasting computation on excluded trials.

ii.
```python
# Bin spikes for ALL trials first (before masking)
binned_spikes = bin_spikes_vectorized(
    spike_times, spike_clusters, n_clusters, interval_begs, interval_ends
)
# ... then later
combined_mask = mask.values & wheel_mask & whisker_mask
neural_trials = binned_spikes[good_indices]
```

iii. The reference processes only the filtered trials.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI bins spikes for all trials before filtering, so trials that are filtered out still have their spikes binned. Additionally, `void` and `root` brain region neurons are included in neural data but may not be meaningful. The AI also stores neural data as raw spike counts rather than firing rates, requiring conversion at training time.

ii.
```python
# Bins all trials, but only good_indices are kept
binned_spikes = bin_spikes_vectorized(
    spike_times, spike_clusters, n_clusters, interval_begs, interval_ends
)
```

iii. Processing all trials before filtering wastes computation proportional to the number of excluded trials.
