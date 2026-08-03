# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads session metadata from `code/code_zhang2025/data/bwm_release.csv`, scans the local `data/one_cache` tree, builds a `(subject, date) -> session_path` map, groups the CSV by `eid`, and then processes each matched session path one by one. Within each session it loads the trials parquet, probe spike files, wheel files, and camera motion-energy files directly from the cache instead of using `ONE`/`SessionLoader`.

ii.
```python
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'
DATA_DIR = 'data/one_cache'

def build_session_map(data_dir):
    session_map = {}
    ...
                for sess_num in os.listdir(date_path):
                    sess_path = os.path.join(date_path, sess_num)
                    if os.path.isdir(sess_path):
                        key = (subject, date)
                        session_map[key] = sess_path

...
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_map = build_session_map(DATA_DIR)

for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
    date = group['date'].iloc[0]
    probe_names = list(group['probe_name'].unique())
    key = (subject, date)
    if key in session_map:
        sessions_info.append({... 'path': session_map[key]})
```

iii. In `CONVERSION_NOTES.md` Step 1 the agent says it is matching `prepare_data` from the reference code but simplifying it to direct local-file access. The trajectory shows the agent explicitly deciding to use `bwm_release.csv` plus the cache tree as the global session index.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. As sessions are processed, each new subject name is added to `all_subjects`, and each kept session stores the integer index of its subject in `all_subject_idx`.

ii.
```python
all_subjects = []
all_subject_idx = []
subject_to_idx = {}

for sess_i, sess_info in enumerate(sessions_info):
    subject = sess_info['subject']
    ...
    if subject not in subject_to_idx:
        subject_to_idx[subject] = len(all_subjects)
        all_subjects.append(subject)
    ...
    all_subject_idx.append(subject_to_idx[subject])
```

iii. `CONVERSION_NOTES.md` Step 5 lists `subjects` and `subject_idx` as direct target-format fields and treats subject identity as coming from the session metadata in `bwm_release.csv`.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` groups in `bwm_release.csv`. For each grouped `eid`, the agent gathers all probe names, resolves a single local session directory from `(subject, date)`, and appends one session entry to the output lists if processing succeeds.

ii.
```python
sessions_info = []
for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
    date = group['date'].iloc[0]
    probe_names = list(group['probe_name'].unique())
    key = (subject, date)
    if key in session_map:
        sessions_info.append({
            'eid': eid,
            'subject': subject,
            'date': date,
            'probe_names': probe_names,
            'path': session_map[key]
        })

for sess_i, sess_info in enumerate(sessions_info):
    result = process_session(...)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. The notes say the reference dataset has 459 sessions and that the conversion should process sessions from `bwm_release.csv`. The trajectory shows the agent reading `0_data_caching.py`, observing that the reference pipeline iterates over `eid`, then recreating that logic locally.

## 1-d. How are the data split into trials?

i. Trials are taken from `_ibl_trials.table.pqt`. After masking invalid trials, each remaining row is treated as one trial. Trial-aligned spike counts, inputs, and outputs are built as per-trial arrays and appended to Python lists, one element per trial.

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
    inp = np.stack([
        time_since_stim,
        np.full(N_BINS, trial_nums_final[t], dtype=np.float32)
    ], axis=0)
    input_list.append(inp)
```

iii. In Step 1 the agent identified `bin_spiking_data` and `bin_behaviors` as the reference trial-segmentation stage, and Step 5 says each output session should be a list of trials aligned to stimulus onset.

## 1-e. How are trials filtered based on quality controls?

i. The agent first filters trials by reaction time, trial duration, missing key fields, and no-choice responses. It does **not** exclude unbiased (`probabilityLeft == 0.5`) trials. It then applies an additional behavior-availability mask requiring both wheel and whisker interpolation to succeed, and drops sessions with fewer than two surviving trials.

ii.
```python
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
EXCLUDE_UNBIASED = False
EXCLUDE_NOCHOICE = True
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

def create_trials_mask(trials_df):
    ...
    if MIN_RT is not None:
        query_parts.append(f'(firstMovement_times - stimOn_times < {MIN_RT})')
    if MAX_RT is not None:
        query_parts.append(f'(firstMovement_times - stimOn_times > {MAX_RT})')
    if MAX_TRIAL_LEN is not None:
        query_parts.append(f'(feedback_times - goCue_times > {MAX_TRIAL_LEN})')
    ...
    if EXCLUDE_NOCHOICE:
        query_parts.append('(choice == 0)')
    return ~trials_df.eval(' | '.join(query_parts))

combined_valid = wheel_valid & me_valid
if n_final < 2:
    return None
```

iii. Step 4 and Step 5 of `CONVERSION_NOTES.md` justify the main deviation: the agent explicitly chose `EXCLUDE_UNBIASED = False` because the decoder task requires the `0.5` prior class. The notes also describe skipping sessions/trials with missing whisker motion energy as an edge-case handling choice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from spike timestamps and spike cluster IDs from each probe, plus cluster-to-channel and channel-to-brain-location arrays so neurons can be labeled by region.

ii.
```python
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
cluster_channels = np.load(os.path.join(version_dir, 'clusters.channels.npy')).flatten()
channel_brain_ids = np.load(
    os.path.join(version_dir, 'channels.brainLocationIds_ccf_2017.npy')
).flatten()
```

iii. The notes state that the reference neural path is `load_spiking_data -> merge_probes -> bin_spiking_data`. The agent’s local reimplementation keeps the same basic spike-times-and-clusters representation, just loading the `.npy` files directly.

## 2-b. How is the `neural` data processed?

i. For each session, all probes are merged by offsetting cluster IDs and concatenating spike trains, then spikes are sorted by time and binned into 20 ms counts over a `[-0.5, 1.5]` second window around stimulus onset for every valid trial. The per-trial matrices are stored as `(n_neurons, 100)` arrays.

ii.
```python
for spike_times, spike_clusters, cluster_channels, channel_brain_ids in probes_data:
    all_spike_times.append(spike_times)
    all_spike_clusters.append(spike_clusters + cluster_offset)
    ...
    cluster_offset += n_clusters

sort_idx = np.argsort(merged_times)
merged_times = merged_times[sort_idx]
merged_clusters = merged_clusters[sort_idx]

binned_spikes = bin_spikes_fast(
    spike_times, spike_clusters, n_clusters,
    interval_begs, interval_ends, BINSIZE, N_BINS
)
```

iii. `CONVERSION_NOTES.md` Step 1 lists `merge_probes` and `bin_spiking_data` as the key reference functions, and Step 3 records the intended 20 ms, 2 s, stimulus-aligned neural representation from the papers/code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is effectively not filtered by neural QC. All clusters found in the chosen pykilosort output are kept; there is no `label >= 1` or similar cluster-quality threshold in the conversion script.

ii.
```python
def load_spikes(alf_dir, probe_name):
    ...
    spike_times = np.load(...)
    spike_clusters = np.load(...)
    cluster_channels = np.load(...)
    channel_brain_ids = np.load(...)
    return spike_times, spike_clusters, cluster_channels, channel_brain_ids
```

iii. The notes explicitly say “No QC filtering” and cite the reference `load_spiking_data(..., qc=None)` call path in `prepare_data`. The trajectory repeats this as a deliberate match to the reference code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`, with trial windows from `stimOn_times - 0.5` to `stimOn_times + 1.5`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. This is justified everywhere in the notes: Step 1 copies the parameters from `0_data_caching.py`, and Step 3 restates the method-paper description of stimulus-onset alignment over a 2 s window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins (`0.02` s), giving `100` bins per trial. No extra temporal rebinning is applied after this binning step.

ii.
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. The notes cite `binsize = 0.02` from `0_data_caching.py` and the method paper’s “20-ms bins, T = 100 time steps” description as the justification.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not loaded from a dedicated raw variable; it is synthesized from the chosen alignment event `stimOn_times`, the fixed time window, and the fixed bin size. The raw dependency is therefore the trial-wise stimulus-onset timestamps plus the conversion configuration.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02

time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. In Step 5 the agent planned this variable as a constructed decoder input rather than a directly loaded data field.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent creates a single 100-element vector of bin-center times from `-0.49` s to `1.49` s and reuses that same vector for every trial.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
...
inp = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_nums_final[t], dtype=np.float32)
], axis=0)
```

iii. The notes justify this as the natural continuous representation of elapsed time after stimulus onset for the decoder input.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is intended to align one value per neural time bin and is broadcast identically across trials. The agent uses bin centers for this input, while the behavioral interpolation code uses right-edge samples (`t_beg + binsize` through `t_end`).

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
)

x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The trajectory and notes do not show a deeper justification beyond “time-varying input aligned to stimulus onset.” The implementation choice appears to be the agent’s own convention rather than something copied from the reference code.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table’s `probabilityLeft` sequence.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
```

iii. Step 5 of the notes maps “Trial number in block” directly from `probabilityLeft` and describes it as computed from block transitions.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent scans all trials in order, resets the counter whenever `probabilityLeft` changes, starts each block at `1`, then applies the trial-quality mask and the behavior-validity mask. The resulting scalar is broadcast across all 100 time bins of that trial.

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

iii. The notes justify this as preserving block structure from the raw experiment rather than recomputing after dropping bad trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the trial table column `choice`.

ii.
```python
choice = final_trials['choice'].values.copy()
choice[choice == -1] = 0
choice = choice.astype(np.float32)
```

iii. Step 5 of the notes states the source variable is `trials_df.choice` and the intended mapping is left/right for the decoder output.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent excludes `choice == 0` trials in the trial mask, remaps `-1` to `0` for left, keeps `1` as right, and then broadcasts the per-trial label across all 100 time bins in the final `output` tensor.

ii.
```python
if EXCLUDE_NOCHOICE:
    query_parts.append('(choice == 0)')

choice = final_trials['choice'].values.copy()
choice[choice == -1] = 0

out = np.stack([
    np.full(N_BINS, int(choice[t]), dtype=np.int64),
    ...
], axis=0)
```

iii. The notes explicitly justify excluding no-choice trials because the decoder task requires a binary left/right target.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial table column `probabilityLeft`.

ii.
```python
prob_left = final_trials['probabilityLeft'].values.copy()
```

iii. Step 5 of the notes maps prior directly from `probabilityLeft`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent keeps unbiased trials, then maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`. Like choice, the resulting class is broadcast across time bins in the final output array.

ii.
```python
EXCLUDE_UNBIASED = False

prior = np.zeros(len(prob_left), dtype=np.float32)
prior[np.isclose(prob_left, 0.2)] = 0
prior[np.isclose(prob_left, 0.5)] = 1
prior[np.isclose(prob_left, 0.8)] = 2

out = np.stack([
    ...,
    np.full(N_BINS, int(prior[t]), dtype=np.int64),
    ...
], axis=0)
```

iii. The notes discuss this as the key deliberate deviation from the reference trial mask: the agent says unbiased trials must be retained because the decoder task explicitly includes `0.5` as a prior class.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
pos_file = find_file(alf_dir, '_ibl_wheel.position.npy')
ts_file = find_file(alf_dir, '_ibl_wheel.timestamps.npy')
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The notes and trajectory show the agent first implemented a simple finite-difference velocity, then revised the script after checking the reference `load_target_behavior('wheel-speed')` path and concluding it should use filtered velocity instead.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The final script estimates a sampling frequency from the median timestamp spacing, computes Butterworth-filtered wheel velocity with `velocity_filtered`, takes the absolute value to get speed, then linearly interpolates the continuous speed trace into the 100 stimulus-aligned trial bins.

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

iii. The trajectory contains an explicit wheel-speed investigation: the agent compared simple-difference speed to `velocity_filtered`, found a low correlation, and updated the script to “match the reference code” more closely.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After all sessions are processed, the agent pools all wheel-speed samples across all sessions/trials/time bins, computes the 33.33rd and 66.67th percentiles, and uses those as global thresholds for three equal-frequency bins.

ii.
```python
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
wheel_percentiles = np.percentile(
    all_wheel_vals[~np.isnan(all_wheel_vals)], [33.33, 66.67]
)
wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
```

iii. Step 5 of the notes says “3 equal-frequency bins across all data.” This was the agent’s chosen discretization policy for the decoder task; it is not taken from the reference code, which uses continuous wheel-speed traces.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset trial windows as neural data and interpolated onto 100 trial bins at `np.linspace(t_beg + binsize, t_end, n_bins)`.

ii.
```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_begs, interval_ends, BINSIZE, N_BINS
)

def interpolate_behavior_to_bins(...):
    x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
    interp_func = interp1d(local_times, local_vals, kind='linear',
                           fill_value='extrapolate')
```

iii. The notes say this was chosen to match the reference `get_behavior_per_interval` interpolation rule from `ibl_data_utils.py`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, with fallback to `rightCamera.ROIMotionEnergy.npy` and `_ibl_rightCamera.times.npy`.

ii.
```python
me_file = find_file(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_dir, '_ibl_leftCamera.times.npy')
if me_file is None or times_file is None:
    me_file = find_file(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. The notes justify “left camera first, falls back to right” by pointing to the reference `bin_behaviors` logic for `'whisker-motion-energy'`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent loads the motion-energy values and timestamps, truncates them to a shared minimum length, removes `NaN`s, then interpolates the continuous trace onto the same 100 stimulus-aligned trial bins used for wheel speed.

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

iii. The notes justify the left/right camera policy and interpolation by reference-code matching; the length-trimming and explicit `NaN` removal are the agent’s own robustness additions.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is discretized exactly like wheel speed: the agent pools all whisker-motion-energy samples, computes global 33.33% and 66.67% quantiles, and uses them as low/medium/high thresholds.

ii.
```python
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
me_percentiles = np.percentile(
    all_me_vals[~np.isnan(all_me_vals)], [33.33, 66.67]
)
me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
```

iii. Step 5 of the notes states the agent’s planned rule was “3 equal-frequency bins across all data.”

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to stimulus onset with the same `[-0.5, 1.5]` window and the same interpolation rule used for wheel speed.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, interval_begs, interval_ends, BINSIZE, N_BINS
)

def interpolate_behavior_to_bins(...):
    x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The notes say this interpolation rule was copied from the reference `get_behavior_per_interval` code.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles missing or malformed data by skipping missing files/sessions, masking bad trials, truncating length mismatches, dropping `NaN`s from whisker traces, invalidating trials whose behavior traces cannot be interpolated, and falling back to simple wheel velocity if filtered velocity fails. Sessions with fewer than two surviving trials are discarded.

ii.
```python
if trials_file is None:
    return None
...
if me_file is None or times_file is None:
    return None, None

min_len = min(len(me_values), len(me_times))
...
valid = ~(np.isnan(me_values) | np.isnan(me_times))
...
if i_end - i_start < 2:
    valid_mask[trial_idx] = False
...
except Exception:
    valid_mask[trial_idx] = False

if n_final < 2:
    return None
```

iii. `CONVERSION_NOTES.md` Step 10 explicitly lists the edge cases the agent considered resolved: `NaN` trial values, missing whisker ME data, length mismatches, and sessions/trials with insufficient valid data.

## 10-a. What are the most time-consuming steps of the code?

i. The main costs are session-wide spike loading, per-trial spike binning, full-dataset serialization to the giant pickle, and the full-dataset percentile pass for wheel/whisker discretization. Behavior interpolation also loops over every trial, but the code’s own timing printout focuses on spike binning and per-session runtime.

ii.
```python
t0 = time.time()
binned_spikes = bin_spikes_fast(...)
print(f'  Spike binning: {time.time()-t0:.1f}s, shape={binned_spikes.shape}')

for sess_i, sess_info in enumerate(sessions_info):
    sess_start = time.time()
    result = process_session(...)
    ...
    print(f'  Done in {elapsed:.1f}s')

all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])

with open(args.output, 'wb') as f:
    pickle.dump(data, f)
```

iii. The notes estimate full runtime from sample timings and describe the full run as ~34 minutes plus save time, which is consistent with spike binning and I/O dominating.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several trial loops remain: `bin_spikes_fast` still loops over trials; `interpolate_behavior_to_bins` loops over trials; `compute_trial_number_in_block` loops over trials; the final construction of `input_list` and `output_list` loops over trials; and `build_session_map` walks the filesystem imperatively. The inner-spike loop in `bin_spikes_vectorized` is especially inefficient, although that helper is unused.

ii.
```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx].ravel(), linear_idx, 1)

for trial_idx in range(n_trials):
    ...
    result[trial_idx] = interp_func(x_interp).astype(np.float32)

for i in range(len(prob_left)):
    ...

for t in range(n_final_trials):
    neural_list.append(...)
    input_list.append(...)

for t in range(n_trials):
    output_list.append(out)
```

iii. The notes mention speed-ups around spike binning, but they do not claim the whole script is fully vectorized. This answer is mostly an inspection of the final code structure.

## 10-c. What processing does the code repeat multiple times?

i. The script recomputes the same `time_since_stim` vector for every session, repeatedly interpolates behaviors trial by trial for each session, repeatedly builds Python lists of per-trial arrays, and performs multiple full passes over wheel/whisker data: once to store raw traces and again to flatten them for global quantiles before a third pass to build discretized outputs.

ii.
```python
time_since_stim = np.linspace(...)

wheel_binned, wheel_valid = interpolate_behavior_to_bins(...)
me_binned, me_valid = interpolate_behavior_to_bins(...)

all_wheel_raw.append(result['wheel_speed_raw'])
all_me_raw.append(result['me_raw'])
...
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
...
for sess_i in range(len(all_neural)):
    wheel_disc = np.digitize(wheel_raw, wheel_percentiles)
    me_disc = np.digitize(me_raw, me_percentiles)
```

iii. This is not explicitly justified in the notes; it follows from the staged design the agent chose to allow global discretization after all sessions had been loaded.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script defines but never uses `bin_spikes_vectorized` and `discretize_continuous`; it stores `eid`, `n_clusters`, and `n_trials` in intermediate per-session dicts only to use them for control flow rather than the final dataset; it computes/keeps full raw wheel and whisker matrices solely to derive global quantile bins and then discards those continuous traces from the final output; and it allocates `output_list_wheel` / `output_list_me` variables that are never used.

ii.
```python
def bin_spikes_vectorized(...):
    ...

def discretize_continuous(values, n_bins=3):
    ...

output_list_wheel = []
output_list_me = []

return {
    ...
    'wheel_speed_raw': final_wheel,
    'me_raw': final_me,
    'n_clusters': n_clusters,
    'n_trials': n_final_trials,
    'eid': eid,
}
```

iii. There is no explicit note justifying these as necessary. They are artifacts of the agent’s incremental implementation and the choice to defer discretization until all sessions were processed.
