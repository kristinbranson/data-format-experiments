# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the release manifest from `code/code_zhang2025/data/bwm_release.csv`, scans `data/one_cache` to build a `(subject, date) -> session_path` map, groups the manifest by `eid`, and then for each session loads trials, spikes, wheel, and whisker files directly from the ALF cache instead of using the reference ONE/SessionLoader path.

ii. 
```python
DATA_DIR = 'data/one_cache'
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'

def build_session_map(data_dir):
    session_map = {}
    ...
        key = (subject, date)
        session_map[key] = sess_path

def load_trials(alf_dir):
    trials_file = find_file(alf_dir, '_ibl_trials.table.pqt')
    trials_df = pd.read_parquet(trials_file)
    return trials_df

def load_spikes(alf_dir, probe_name):
    ...
    spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
    spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()

bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_map = build_session_map(DATA_DIR)
for eid, group in bwm_df.groupby('eid'):
    ...
    sessions_info.append({... 'probe_names': probe_names, 'path': session_map[key]})
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 5, the agent says it is using the local ONE cache layout and `bwm_release.csv` to recover the same 459 released sessions. In trajectory Step 56 and Step 58 it explicitly justifies direct cache loading as sufficient once it had matched all release sessions.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. The script keeps a unique `all_subjects` list and assigns each kept session an integer subject index through `subject_to_idx`.

ii. 
```python
for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
    ...

all_subjects = []
all_subject_idx = []
subject_to_idx = {}

if subject not in subject_to_idx:
    subject_to_idx[subject] = len(all_subjects)
    all_subjects.append(subject)

all_subject_idx.append(subject_to_idx[subject])
```

iii. The notes repeatedly describe subjects as coming from the release table, and Step 7/9 summary statistics are reported in terms of unique mice recovered from processed sessions.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique `eid` groups from `bwm_release.csv`. Each grouped `eid` contributes one session entry, with all of its listed probes merged into that session.

ii. 
```python
sessions_info = []
for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
    date = group['date'].iloc[0]
    probe_names = list(group['probe_name'].unique())
    ...
    sessions_info.append({
        'eid': eid,
        'subject': subject,
        'date': date,
        'probe_names': probe_names,
        'path': session_map[key]
    })
```

iii. In `CONVERSION_NOTES.md` Step 4 and trajectory Step 56, the agent states that the release file defines 459 sessions and that sessions should be grouped by `eid`, not by individual probes.

## 1-d. How are the data split into trials?

i. For each session, the full trial table is loaded, filtered by `create_trials_mask`, then each remaining trial is represented by one stimulus-aligned interval from `stimOn_times - 0.5 s` to `stimOn_times + 1.5 s`. Neural and behavioral arrays are stored as per-trial list elements.

ii. 
```python
trials_df = load_trials(alf_dir)
mask = create_trials_mask(trials_df)
valid_trials_df = trials_df[mask].reset_index(drop=True)

stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]

for t in range(n_final_trials):
    neural_list.append(final_spikes[t].astype(np.float32))
    ...
    input_list.append(inp)
```

iii. The notes Step 1 and Step 5 say the agent is following the reference `stimOn_times`, `(-0.5, 1.5)`, `20 ms` trialization scheme from the provided code.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if reaction time is outside `0.08` to `2.0 s`, if `feedback_times - goCue_times > 10 s`, if any of `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType` is null, or if `choice == 0`. Unbiased `probabilityLeft == 0.5` trials are kept. After that, trials are further dropped if wheel or whisker interpolation failed.

ii. 
```python
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
EXCLUDE_UNBIASED = False
EXCLUDE_NOCHOICE = True
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

if MAX_TRIAL_LEN is not None:
    query_parts.append(f'(feedback_times - goCue_times > {MAX_TRIAL_LEN})')
...
if EXCLUDE_NOCHOICE:
    query_parts.append('(choice == 0)')

combined_valid = wheel_valid & me_valid
final_trials = valid_trials_df[combined_valid].reset_index(drop=True)
```

iii. In trajectory Step 58, the agent explicitly corrected itself after rereading the reference defaults: `min_rt=0.08`, `max_rt=2.0`, `exclude_unbiased=False`, `exclude_nochoice=True`. In later notes/trajectory it justified the extra wheel/whisker validity filtering as a way to avoid missing dynamic outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from per-probe spike times and spike cluster assignments, with brain-region labels derived from cluster-channel mappings and channel brain-location IDs.

ii. 
```python
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
cluster_channels = np.load(os.path.join(version_dir, 'clusters.channels.npy')).flatten()
channel_brain_ids = np.load(os.path.join(version_dir, 'channels.brainLocationIds_ccf_2017.npy')).flatten()

cluster_brain_ids = channel_brain_ids[cluster_channels]
```

iii. `CONVERSION_NOTES.md` Step 1 and Step 2 identify the reference spike-sorting loaders and the local ALF files the agent planned to mirror.

## 2-b. How is the `neural` data processed?

i. The script merges all probes within a session by offsetting cluster IDs, maps cluster acronyms into the Beryl atlas, and bins spikes into 20 ms counts over the 2 s stimulus-aligned window for every trial.

ii. 
```python
for spike_times, spike_clusters, cluster_channels, channel_brain_ids in probes_data:
    n_clusters = len(cluster_channels)
    all_spike_clusters.append(spike_clusters + cluster_offset)
    ...
    cluster_offset += n_clusters

acronyms = br.id2acronym(cluster_brain_ids)
beryl_regions = br.acronym2acronym(acronyms, mapping='Beryl')

binned_spikes = bin_spikes_fast(
    spike_times, spike_clusters, n_clusters,
    interval_begs, interval_ends, BINSIZE, N_BINS
)
```

iii. The notes Step 1 and Step 5 say the agent intended to match `merge_probes`, `list_brain_regions`, and `bin_spiking_data`, including Beryl mapping and all-region decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is effectively not QC-filtered at the neuron level: all clusters from all listed probes are included.

ii. 
```python
for probe_name in probe_names:
    result = load_spikes(alf_dir, probe_name)
    if result[0] is not None:
        probes_data.append(result)

spike_times, spike_clusters, cluster_brain_ids = merge_probes_data(probes_data)
n_clusters = len(cluster_brain_ids)
```

iii. `CONVERSION_NOTES.md` Step 1 says “No QC filtering: `load_spiking_data` called without qc parameter (defaults to None = all clusters),” and trajectory Steps 10 and 56 repeat that justification.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to stimulus onset by taking a window from `stimOn_times - 0.5 s` to `stimOn_times + 1.5 s`.

ii. 
```python
TIME_WINDOW = (-0.5, 1.5)
ALIGN_TIME = 'stimOn_times'

stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. The notes Step 1 and Step 3 explicitly record `align_time='stimOn_times'` and `time_window=(-0.5, 1.5)` from the reference code and the decoder task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins for a total of 100 bins per trial. No additional rebinning is applied after spike counting.

ii. 
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes Step 1 and Step 3 identify 20 ms and 100 bins as a direct carryover from the reference code and methods text.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the choice of alignment event `stimOn_times` plus the fixed `TIME_WINDOW` and `BINSIZE`; it is not loaded from a separate recorded trace.

ii. 
```python
TIME_WINDOW = (-0.5, 1.5)
ALIGN_TIME = 'stimOn_times'
BINSIZE = 0.02

time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent planned this as a computed variable rather than a loaded dataset: “Time since stim onset | Computed.”

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script creates a fixed length-100 vector of bin-center times from `-0.49 s` to `1.49 s` and reuses it for every trial.

ii. 
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. The justification in the notes Step 5 is that the decoder input is meant to be a continuous, time-varying description of the stimulus-aligned trial axis.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is stacked into each trial’s `input` array with the same `N_BINS` used for the neural spike matrix, so bin index matches neural time bin index.

ii. 
```python
inp = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_nums_final[t], dtype=np.float32)
], axis=0)
input_list.append(inp)
```

iii. The notes Step 5 frame both decoder inputs as being built on the same per-trial, 100-bin structure as the neural data.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table’s `probabilityLeft` column.

ii. 
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
```

iii. `CONVERSION_NOTES.md` Step 5 identifies “Trial number in block | Computed from probabilityLeft.”

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code increments a counter within each run of constant `probabilityLeft`, resets the count when `probabilityLeft` changes, computes this on all raw trials, and then subselects to the valid/final kept trials.

ii. 
```python
def compute_trial_number_in_block(prob_left):
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    current_block_start = 0
    current_prob = prob_left[0]
    for i in range(len(prob_left)):
        if prob_left[i] != current_prob:
            current_block_start = i
            current_prob = prob_left[i]
        trial_nums[i] = i - current_block_start + 1
    return trial_nums

valid_indices = np.where(mask.values)[0]
trial_nums_valid = all_trial_nums[valid_indices]
trial_nums_final = trial_nums_valid[combined_valid]
```

iii. In notes Step 5, the agent justified this as the simplest way to expose block context for the decoder while preserving the original session block structure.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column in the trial table.

ii. 
```python
choice = final_trials['choice'].values.copy()
```

iii. The notes Step 5 map `choice` directly from `trials_df.choice`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. No-choice trials are removed earlier, then the remaining IBL coding is remapped from `-1/1` to `0/1` by sending left (`-1`) to `0` and keeping right (`1`) as `1`. The value is then broadcast across all time bins in each trial.

ii. 
```python
choice = final_trials['choice'].values.copy()
choice[choice == -1] = 0
choice = choice.astype(np.float32)

np.full(N_BINS, int(choice[t]), dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Step 4 and Step 5 explicitly justify this remapping because the decoder task requested `left = 0, right = 1`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column in the trial table.

ii. 
```python
prob_left = final_trials['probabilityLeft'].values.copy()
```

iii. The notes Step 4 and Step 5 identify “prior” with `probabilityLeft`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps the three allowed values into decoder classes: `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then broadcasts the per-trial class across all bins.

ii. 
```python
prior = np.zeros(len(prob_left), dtype=np.float32)
prior[np.isclose(prob_left, 0.2)] = 0
prior[np.isclose(prob_left, 0.5)] = 1
prior[np.isclose(prob_left, 0.8)] = 2

np.full(N_BINS, int(prior[t]), dtype=np.int64)
```

iii. In trajectory Step 58 the agent specifically says unbiased `0.5` trials must be kept because the task defines them as a valid prior class.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
pos_file = find_file(alf_dir, '_ibl_wheel.position.npy')
ts_file = find_file(alf_dir, '_ibl_wheel.timestamps.npy')
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The notes Step 2 identify these wheel files, and trajectory Steps 192-198 show the agent revising the script to better match the reference wheel-processing path.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The code estimates sampling rate from wheel timestamps, computes Butterworth-filtered wheel velocity with `velocity_filtered`, takes the absolute value to get speed, interpolates the continuous speed trace onto the trial bins, and later discretizes it.

ii. 
```python
dt_median = np.median(np.diff(timestamps))
fs = 1.0 / dt_median
velocity, _ = velocity_filtered(position, fs)
speed = np.abs(velocity)

wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_begs, interval_ends, BINSIZE, N_BINS
)
```

iii. Trajectory Steps 189-198 show the agent explicitly comparing simple finite differences against `velocity_filtered`, concluding the mismatch was too large, and updating the code to match the reference wheel loader more closely.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. After all sessions are processed, all wheel-speed samples are pooled globally and thresholded into 3 equal-frequency bins using the 33.33rd and 66.67th percentiles.

ii. 
```python
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
wheel_percentiles = np.percentile(all_wheel_vals[~np.isnan(all_wheel_vals)], [33.33, 66.67])
wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent states the planned discretization as “3 equal-frequency bins across all data.”

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated onto the same 100 stimulus-aligned bins used for neural activity, then each trial’s discretized wheel-speed vector becomes output row 2.

ii. 
```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_begs, interval_ends, BINSIZE, N_BINS
)

out = np.stack([
    ...,
    wheel_disc[t],
    me_disc[t],
], axis=0).astype(np.int64)
```

iii. The notes Step 5 say all outputs in this conversion were aligned to stimulus onset with the shared `(-0.5, 1.5)` window and `20 ms` bins to satisfy the decoder task.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`, and if those are unavailable it falls back to `rightCamera.ROIMotionEnergy.npy` with `_ibl_rightCamera.times.npy`.

ii. 
```python
me_file = find_file(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_dir, '_ibl_leftCamera.times.npy')
if me_file is None or times_file is None:
    me_file = find_file(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. `CONVERSION_NOTES.md` Step 1 and Step 5 explicitly note “tries left camera first, falls back to right.”

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code loads the precomputed motion-energy trace, truncates data/time arrays to a common minimum length, removes NaNs, interpolates the trace to the stimulus-aligned trial bins, and later discretizes it globally.

ii. 
```python
me_values = np.load(me_file).flatten()
me_times = np.load(times_file).flatten()
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]
valid = ~(np.isnan(me_values) | np.isnan(me_times))
me_values = me_values[valid]
me_times = me_times[valid]

me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, interval_begs, interval_ends, BINSIZE, N_BINS
)
```

iii. The notes Step 2 document these source files, and Step 5 describes whisker motion energy as a continuous signal later discretized into 3 bins.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, it is thresholded into 3 equal-frequency global bins using pooled percentiles across all sessions and trial bins.

ii. 
```python
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
me_percentiles = np.percentile(all_me_vals[~np.isnan(all_me_vals)], [33.33, 66.67])
me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 states the planned whisker-motion-energy discretization as “3 equal-frequency bins across all data.”

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is interpolated onto the same stimulus-aligned `100 x 20 ms` trial axis as the neural data, then stored as output row 3.

ii. 
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, interval_begs, interval_ends, BINSIZE, N_BINS
)

out = np.stack([
    ...,
    wheel_disc[t],
    me_disc[t],
], axis=0).astype(np.int64)
```

iii. The notes Step 5 say the conversion intentionally puts all outputs onto the shared stimulus-aligned decoder axis.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files generally cause the whole session to be skipped. Within kept sessions, wheel filter failures fall back to simple finite differences, whisker traces are truncated to the shorter of times/values and NaNs are removed, and trials with failed wheel/whisker interpolation are dropped.

ii. 
```python
if pos_file is None or ts_file is None:
    return None, None
...
except Exception:
    dt = np.diff(timestamps)
    ...
    velocity[1:] = np.diff(position) / dt

min_len = min(len(me_values), len(me_times))
...
valid = ~(np.isnan(me_values) | np.isnan(me_times))

if wheel_times is None:
    return None
if me_times is None:
    return None

combined_valid = wheel_valid & me_valid
```

iii. The notes Step 9 and trajectory Steps 74, 97, and 105 justify skipping sessions with missing whisker data rather than imputing them; the agent treated this as acceptable curation to ensure valid outputs.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant work is per-session spike binning across all trials and clusters, plus per-trial interpolation of wheel and whisker signals, followed by the global pass used for discretization and output assembly.

ii. 
```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx].ravel(), linear_idx, 1)

for trial_idx in range(n_trials):
    ...
    result[trial_idx] = interp_func(x_interp).astype(np.float32)

for sess_i, sess_info in enumerate(sessions_info):
    result = process_session(...)
```

iii. `CONVERSION_NOTES.md` Step 7 includes runtime estimates per session, and trajectory discussions around optimization focus on spike binning and behavioral interpolation.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit trial loop in `bin_spikes_fast`, the trial loop in `interpolate_behavior_to_bins`, the trial loop that builds `neural_list`/`input_list`, and the per-session trial loop that builds `all_output` could all be reduced further with more array-level assembly.

ii. 
```python
for trial_idx in range(n_trials):
    ...

for trial_idx in range(n_trials):
    ...

for t in range(n_final_trials):
    neural_list.append(final_spikes[t].astype(np.float32))
    ...

for t in range(n_trials):
    out = np.stack([...], axis=0).astype(np.int64)
    output_list.append(out)
```

iii. The agent’s notes mention trying to speed up the pipeline, but its implementation still leaves several Python loops on the hot path.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly searches trial-local index ranges for wheel and whisker interpolation, repeatedly allocates broadcasted per-trial input/output arrays, and repeats similar interpolation logic separately for wheel and whisker traces.

ii. 
```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(...)
me_binned, me_valid = interpolate_behavior_to_bins(...)

inp = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_nums_final[t], dtype=np.float32)
], axis=0)

out = np.stack([
    np.full(N_BINS, int(choice[t]), dtype=np.int64),
    np.full(N_BINS, int(prior[t]), dtype=np.int64),
    wheel_disc[t],
    me_disc[t],
], axis=0).astype(np.int64)
```

iii. The runtime notes and trajectory optimization comments show the agent was aware of repeated per-trial work, especially for behavioral interpolation and output assembly.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The helper `discretize_continuous` is defined but never used; `output_list_wheel` and `output_list_me` are allocated but never populated or consumed; and the optional plotting path builds extra visualization artifacts that are not part of the converted dataset.

ii. 
```python
def discretize_continuous(values, n_bins=3):
    ...

output_list_wheel = []
output_list_me = []

if args.show_processing and len(all_neural) > 0:
    ...
    fig.savefig(f'processing_session_{sess_i}.png', dpi=150)
```

iii. These are not discussed as deliberate design choices in the notes; they appear to be leftover utilities or debugging aids added during iteration.
