# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by scanning the local `data/one_cache` filesystem directly rather than using the ONE API. It finds session directories with required files by globbing paths under `data/one_cache/<lab>/Subjects/<subject>/<date>/001`, then reads trial parquet files and `.npy` arrays from those directories. It does not load the Brainwidemap release index or use `eid` identifiers.

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
    return valid
```

```python
def load_trials(sdir):
    trial_files = sorted(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt')))
    trials = pd.read_parquet(trial_files[-1])
    return trials
```

iii. The justification in `CONVERSION_NOTES.md` is that the available data are laid out under `data/one_cache/<lab>/Subjects/<mouse>/<date>/001/alf/`, so the agent chose to work from that on-disk structure. The trajectory summary also states that the task was treated as loading “from local files”.

## 1-b. How are the data split into subjects?

i. Subjects are split by parsing the session directory path and taking the path component after `Subjects`. The final `subjects` list preserves first-seen order, and `subject_idx` is built by indexing into that list.

ii. 
```python
def parse_session_info(sdir):
    parts = Path(sdir).parts
    sub_idx = parts.index('Subjects')
    lab = parts[sub_idx - 1]
    subject = parts[sub_idx + 1]
    date = parts[sub_idx + 2]
    return lab, subject, date
```

```python
if subject not in all_subjects:
    all_subjects.append(subject)
subject_per_session.append(subject)
...
subject_idx = np.array([all_subjects.index(s) for s in subject_per_session], dtype=np.int32)
```

iii. The notes describe the cache layout explicitly, so the subject split is justified by directory organization rather than by metadata from the ONE API.

## 1-c. How are the data split into sessions?

i. Sessions are split by directory, with each `.../Subjects/<subject>/<date>/001` directory treated as one session. The agent constructs a session id as `<subject>_<date>`.

ii. 
```python
session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
```

```python
lab, subject, date = parse_session_info(sdir)
session_id = f"{subject}_{date}"
```

iii. The justification in the notes is again the on-disk cache layout, where each date folder with run `001` is treated as a session container.

## 1-d. How are the data split into trials?

i. Trials are split by rows of the trial table loaded from `_ibl_trials.table.pqt`. The code keeps the table row structure intact and later indexes trials by row number.

ii. 
```python
trials = load_trials(sdir)
...
stim_on = trials[ALIGN_TIME].values
...
trials_good = trials.iloc[good_indices]
```

iii. The notes describe `_ibl_trials.table.pqt` as the session trial table, so the AI relied on the table already being one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering happens in two stages. First, `create_trial_mask()` filters by reaction time, no-choice trials, NaNs in a fixed list of columns, and trial length greater than 10 s. Second, `process_session()` intersects that mask with wheel and whisker availability masks produced during interpolation, and sessions with fewer than two surviving trials are skipped.

ii. 
```python
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
...
trial_len = trials['feedback_times'] - trials['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
...
mask &= (trials['choice'] != 0)
...
for col in NAN_EXCLUDE:
    if col in trials.columns:
        mask &= ~trials[col].isna()
```

```python
wheel_vals, wheel_mask = interpolate_behavior_to_bins(...)
whisker_vals, whisker_mask = interpolate_behavior_to_bins(...)
combined_mask = mask.values & wheel_mask & whisker_mask
good_indices = np.where(combined_mask)[0]
if len(good_indices) < 2:
    return None
```

iii. `CONVERSION_NOTES.md` says the agent followed the reference `load_trials_and_mask` logic from Zhang et al. and explicitly lists RT filtering, no-choice exclusion, NaN exclusion, and `max_trial_len=10.0`. It also says behavior-availability masking follows `align_spike_behavior`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` array is derived from `spikes.times.npy` and `spikes.clusters.npy` across all probes in a session. Additional files, `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`, are used to assign brain-region labels per cluster, but not to compute spike counts themselves.

ii. 
```python
st_file = os.path.join(pdir, 'spikes.times.npy')
sc_file = os.path.join(pdir, 'spikes.clusters.npy')
cc_file = os.path.join(pdir, 'clusters.channels.npy')
cb_file = os.path.join(pdir, 'channels.brainLocationIds_ccf_2017.npy')
...
spike_times = np.load(st_file).flatten()
spike_clusters = np.load(sc_file).flatten()
cluster_channels = np.load(cc_file).flatten()
channel_brain_ids = np.load(cb_file).flatten()
```

iii. The notes summarize the same decision: “spikes.times + spikes.clusters” are mapped to `neural`, while channel and cluster metadata are used for region mapping.

## 2-b. How is the `neural` data processed?

i. The AI merges probes by offsetting cluster ids, sorts all spikes by time, bins spikes into 20 ms bins over a 2 s stimulus-aligned window, and stores per-trial neuron-by-time spike counts. It does not convert counts to Hz. It then clips counts to `[0, 255]` and stores each trial as `uint8`.

ii. 
```python
spike_clusters_offset = spike_clusters + cluster_offset
cluster_offset += n_clusters
...
sort_idx = np.argsort(merged_times, kind='stable')
merged_times = merged_times[sort_idx]
merged_clusters = merged_clusters[sort_idx].astype(np.int32)
```

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

iii. The notes say the code should “Bin into 20ms bins per trial, aligned to stimOn_times” and that “ALL neurons” should be used. The memory section also justifies `uint8` storage as a size-saving choice.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by spike-sorting quality. The agent explicitly keeps all clusters from all probes, maps them to Beryl regions, and retains units labeled `root` or `void` instead of removing them up front.

ii. 
```python
def load_spikes(sdir):
    """Load and merge spikes from all probes in a session.

    Following reference code: no QC filtering (qc=None).
    """
```

```python
cluster_acronyms = br.id2acronym(cluster_brain_ids)
beryl_regions = br.acronym2acronym(cluster_acronyms, mapping='Beryl')
...
all_cluster_regions.extend(beryl_regions)
```

```python
brain_regions = sorted([r for r in all_brain_regions if r not in ('root', 'void')])
if 'void' in all_brain_regions:
    brain_regions.append('void')
if 'root' in all_brain_regions:
    brain_regions.append('root')
```

iii. `CONVERSION_NOTES.md` repeatedly states “Use ALL neurons (qc=None), matching reference code” and lists “No QC filter applied in `prepare_data`”. The trajectory summary also says the agent intentionally chose “ALL neurons”.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `stimOn_times`. For each trial, the neural window is `[stimOn - 0.5, stimOn + 1.5]`, and bin indices are computed relative to `t_beg = stimOn - 0.5`.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
stim_on = trials[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

```python
bin_idx = np.minimum(
    ((times_trial - t_beg) / BINSIZE).astype(np.int32),
    N_BINS - 1
)
```

iii. The notes explicitly record a discrepancy between the paper text and the Zhang code, then resolve it by following the code and decoder instructions: align everything to stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 20 ms bins for a 2 s window, producing 100 time bins. No temporal rebinning beyond this binning step is applied.

ii. 
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes say the agent followed the reference code choice of uniform 20 ms bins and a 2 s window.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the stimulus onset event `stimOn_times`, combined with the fixed decoder window and bin size. The final values are synthetic bin-center times measured relative to each trial’s stimulus onset.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
stim_on = trials[ALIGN_TIME].values
```

```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. The notes describe this field as “Time since stimOn” with values equal to the centers of the 20 ms bins in the stimulus-aligned window.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code computes a 100-element vector of equally spaced bin centers from `-0.49` to `1.49` seconds and reuses that same vector for every trial in a session.

ii. 
```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. The justification in the notes is simply that the decoder input should represent time relative to stimulus onset on the same 20 ms grid used elsewhere.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is intended to match the neural binning grid: the input uses the same window and number of bins as the neural counts, and each trial’s input matrix repeats the same `time_input` vector.

ii. 
```python
inp = np.stack([
    time_input,
    np.full(N_BINS, trial_num_in_block[i], dtype=np.float32)
], axis=0)
```

iii. The notes explicitly say the time input is the center of each bin in the stimulus-aligned neural window, so it is meant to be bin-for-bin aligned with neural activity.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table column `probabilityLeft`, treating each contiguous run of the same value as a block.

ii. 
```python
prob_left = trials_good['probabilityLeft'].values
trial_num_in_block = compute_trial_num_in_block(prob_left)
```

```python
def compute_trial_num_in_block(prob_left):
    trial_nums = np.ones(len(prob_left), dtype=np.int32)
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            trial_nums[i] = trial_nums[i - 1] + 1
        else:
            trial_nums[i] = 1
```

iii. The notes say “Block trial number: Computed as position within contiguous block of same probabilityLeft”.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. After filtering trials, the AI counts how many consecutive surviving trials have the same `probabilityLeft`. The count is 1-based, resetting to `1` at each block boundary, and it ignores filtered-out trials when counting.

ii. 
```python
trials_good = trials.iloc[good_indices]
...
prob_left = trials_good['probabilityLeft'].values
trial_num_in_block = compute_trial_num_in_block(prob_left)
```

```python
trial_nums = np.ones(len(prob_left), dtype=np.int32)
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        trial_nums[i] = trial_nums[i - 1] + 1
    else:
        trial_nums[i] = 1
```

iii. The notes justify the feature as “count trials since last block change”, but do not explain why it is computed after filtering or why indexing starts at 1 instead of 0.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column of the trial table.

ii. 
```python
choice = trials_good['choice'].values.copy()
```

iii. The notes and mapping table identify `choice` from the trials table as the source variable for this decoder output.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code turns `choice` into a binary value using `np.where(choice == 1, 1, 0)`. The agent’s comments and notes interpret this as `-1 (left) -> 0` and `1 (right) -> 1`, and no additional processing is applied.

ii. 
```python
# Choice: -1 (left) -> 0, 1 (right) -> 1
choice = trials_good['choice'].values.copy()
choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
```

iii. The only explicit justification is the agent’s own comment and the mapping table in `CONVERSION_NOTES.md`, which state that `-1` means left and `1` means right.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial table column `probabilityLeft`.

ii. 
```python
prob_left = trials_good['probabilityLeft'].values
```

iii. The notes identify “Prior/Block = probabilityLeft” as the source variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, with `0.5` as the default class and `np.isclose(..., atol=0.05)` used for the other two levels.

ii. 
```python
prior = np.full(len(prob_left), 1, dtype=np.int32)
prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
```

iii. The notes say the decoder task requires the mapping `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, and the code follows that directly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
wh_pos = np.load(os.path.join(sdir, 'alf/_ibl_wheel.position.npy')).flatten()
wh_times = np.load(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy')).flatten()
```

iii. The notes describe wheel speed as coming from the wheel position and timestamps, matching the reference target variable description.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates wheel position onto a uniform 1 kHz grid, differentiates with `np.gradient`, takes absolute velocity as speed, then linearly interpolates that trace into the 100 trial bins and discretizes the result into three quantile bins across the session.

ii. 
```python
dt = 0.001
t_uniform = np.arange(wh_times[0], wh_times[-1], dt)
pos_interp = np.interp(t_uniform, wh_times, wh_pos)
velocity = np.gradient(pos_interp, dt)
speed = np.abs(velocity)
```

```python
wheel_vals, wheel_mask = interpolate_behavior_to_bins(
    wh_times, wh_speed, interval_begs, interval_ends
)
...
wheel_discrete = discretize_to_bins(wheel_trials, n_bins=3)
```

iii. The notes say the reference loader returns interpolated wheel velocity and that the AI tried to “replicate this from raw position + timestamps”. They also note that wheel speed is the absolute value of velocity and should be discretized into terciles.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is thresholded into three categories using session-level quantile boundaries computed from all non-NaN binned wheel-speed samples that survive trial filtering.

ii. 
```python
def discretize_to_bins(values, n_bins=3):
    flat = values[~np.isnan(values)].flatten()
    ...
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    boundaries = np.percentile(flat, quantiles)
    result = np.digitize(values, boundaries).astype(np.int32)
    return result
```

iii. `CONVERSION_NOTES.md` explicitly says “Use session-wide terciles for wheel speed and whisker ME”.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The code aligns wheel data to the same stimulus-centered trial windows as neural data, but it resamples wheel speed at `np.linspace(t_beg + BINSIZE, t_end, N_BINS)`, which corresponds to bin right edges rather than neural bin centers.

ii. 
```python
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
...
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
interp_func = interp1d(beh_t, beh_v, kind='linear', fill_value='extrapolate')
values[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The notes justify the overall alignment choice by saying everything should follow the reference code and the decoder task, which both use stimulus-onset alignment for all outputs.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` or `rightCamera.ROIMotionEnergy.npy` together with the matching `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`, with left camera preferred.

ii. 
```python
left_me_files, left_time_files = _find_files(
    sdir, 'leftCamera.ROIMotionEnergy.npy', '_ibl_leftCamera.times.npy')
...
right_me_files, right_time_files = _find_files(
    sdir, 'rightCamera.ROIMotionEnergy.npy', '_ibl_rightCamera.times.npy')
```

iii. The notes explicitly say “Whisker ME source: Try left camera first, fall back to right, matching code”.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is used directly, trimmed to the shorter of the trace and timestamp arrays if they differ in length, then interpolated into trial bins and discretized into session-level terciles. No filtering or normalization is applied.

ii. 
```python
me = np.load(left_me_files[-1]).flatten()
times = np.load(left_time_files[-1]).flatten()
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len]
```

```python
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
...
whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
```

iii. The notes say whisker motion energy should be loaded from the camera trace and discretized into three bins, with no additional processing described beyond interpolation.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded exactly like wheel speed: three categories from session-level quantiles computed across all valid binned whisker motion-energy samples.

ii. 
```python
whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
```

```python
boundaries = np.percentile(flat, quantiles)
result = np.digitize(values, boundaries).astype(np.int32)
```

iii. The notes state the same “session-wide terciles” rule for both wheel speed and whisker motion energy.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. As with wheel speed, whisker motion energy is aligned to stimulus-centered trial windows, but the interpolation points are the bin right edges rather than neural bin centers.

ii. 
```python
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
...
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
```

iii. The notes justify the stimulus-onset alignment choice by appealing to the Zhang code and decoder instructions, even though the methods text described a different alignment for movement-related variables.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing or problematic data by skipping. Missing trial values are filtered by `create_trial_mask()`. Missing or insufficient behavior coverage marks individual trials invalid. Sessions with fewer than two valid trials return `None`. Missing files or other runtime errors in session processing are caught and the entire session is skipped. For whisker files, if time and value arrays have different lengths, they are truncated to the shorter length.

ii. 
```python
for col in NAN_EXCLUDE:
    if col in trials.columns:
        mask &= ~trials[col].isna()
```

```python
if len(beh_v) == 0:
    mask[trial_idx] = False
    continue
if np.abs(t_beg - beh_t[0]) > BINSIZE:
    mask[trial_idx] = False
    continue
if np.abs(t_end - beh_t[-1]) > BINSIZE:
    mask[trial_idx] = False
    continue
```

```python
if len(good_indices) < 2:
    return None
...
except Exception as e:
    ...
    return None
```

iii. The notes frame this as “graceful handling of missing data” and report that sessions without usable wheel or whisker data were skipped. The trajectory also shows the agent debugging missing whisker files by widening its path search logic.

## 10-a. What are the most time-consuming steps of the code?

i. The AI’s notes identify spike binning and wheel/whisker interpolation as the main time costs, with rough per-session timings of about 0.2 to 0.3 s for spike binning and 0.5 to 1 s for wheel/whisker processing.

ii. 
```python
print(f"  Binning spikes ({n_clusters} neurons, {len(trials)} trials)...", flush=True)
t_bin = time.time()
binned_spikes = bin_spikes_vectorized(...)
print(f"  Spike binning: {time.time() - t_bin:.1f}s", flush=True)
```

```python
print(f"  Loading wheel speed...", flush=True)
wh_times, wh_speed = load_wheel_speed(sdir)
...
print(f"  Loading whisker ME...", flush=True)
me_times, me_vals_raw = load_whisker_me(sdir)
```

iii. The justification is from the runtime table in `CONVERSION_NOTES.md`, which reports these stages as the measured dominant costs during conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Despite the function name `bin_spikes_vectorized`, several per-trial loops remain. The spike binning loop over trials, the behavior interpolation loop over trials, and the final loops that build `input_list` and `output_list` could all have been vectorized further.

ii. 
```python
for trial_idx in range(n_trials):
    t_beg = interval_begs[trial_idx]
    ...
    i_start = np.searchsorted(spike_times, t_beg, side='left')
```

```python
for trial_idx in range(n_trials):
    t_beg = interval_begs[trial_idx]
    ...
    values[trial_idx] = interp_func(x_interp).astype(np.float32)
```

```python
for i in range(n_trials):
    inp = np.stack([...], axis=0)
    input_list.append(inp)

for i in range(n_trials):
    out = np.stack([...], axis=0)
    output_list.append(out)
```

iii. The notes emphasize speedups from `searchsorted` and flat indexing, but the code still keeps several explicit Python loops for per-trial work.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several kinds of processing: it runs `np.searchsorted` separately for every trial in both spike binning and behavior interpolation, rebuilds the same `time_input` vector for every session, and constructs per-trial input/output arrays one trial at a time in Python loops.

ii. 
```python
for trial_idx in range(n_trials):
    i_start = np.searchsorted(spike_times, t_beg, side='left')
    i_end = np.searchsorted(spike_times, t_end, side='left')
```

```python
for trial_idx in range(n_trials):
    idx_beg = np.searchsorted(beh_times, t_beg, side='right')
    idx_end = np.searchsorted(beh_times, t_end, side='left')
```

```python
time_input = np.linspace(
    TIME_WINDOW[0] + BINSIZE / 2,
    TIME_WINDOW[1] - BINSIZE / 2,
    N_BINS
).astype(np.float32)
```

iii. There is no explicit justification in the notes beyond prioritizing a simple session-by-session implementation and manual memory management.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is that it bins spikes and interpolates behavior for all trials before applying the combined trial mask, so masked-out trials are fully processed and then discarded. It also clips neural counts to `uint8`, which is a storage optimization rather than part of the target scientific processing.

ii. 
```python
# 4. Bin spikes for ALL trials first (before masking)
binned_spikes = bin_spikes_vectorized(
    spike_times, spike_clusters, n_clusters, interval_begs, interval_ends
)
...
combined_mask = mask.values & wheel_mask & whisker_mask
good_indices = np.where(combined_mask)[0]
neural_trials = binned_spikes[good_indices]
```

```python
wheel_vals, wheel_mask = interpolate_behavior_to_bins(
    wh_times, wh_speed, interval_begs, interval_ends
)
whisker_vals, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_vals_raw, interval_begs, interval_ends
)
```

```python
neural_list = [np.clip(neural_trials[i], 0, 255).astype(np.uint8) for i in range(n_trials)]
```

iii. The notes justify the `uint8` conversion as a memory-saving measure, but they do not justify processing invalid trials before discarding them.
