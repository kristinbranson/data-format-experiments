# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the BWM release table from `bwm_release.csv`, takes unique `eid` values as sessions, then for each session loads probe spike-sorting data, trial tables, wheel data, and whisker motion energy. It uses the `ONE` API plus `SpikeSortingLoader` and `SessionLoader`.

ii. ```python
bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
eids = bwm_df['eid'].unique()

rows = bwm_df[bwm_df['eid'] == eid]
pids = rows['pid'].values
probe_names = rows['probe_name'].values

spks, clust = load_spiking_data(one, pid, eid=eid, pname=pname)
trials, mask, sl = load_trials_and_mask(one, eid)
sl.load_wheel()
sl.load_motion_energy(views=['left'])
```

iii. `CONVERSION_NOTES.md` says the data source is the IBL BWM release and the script follows Zhang et al. and IBL. The trajectory shows the agent explicitly relied on the reference caching script, which also starts from `data/bwm_release.csv`, iterates over `eid`, and calls `prepare_data(...)`.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of the release table. The agent builds a `subjects_list` in first-seen order and assigns each session a `subject_idx`.

ii. ```python
subject = rows.iloc[0]['subject']

if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects_list)
    subjects_list.append(subject)
subj_idx = subject_to_idx[subject]
```

iii. The notes state the release contains 139 subjects and the agent reports matching that count. There is no deeper justification beyond mirroring the release metadata.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from the release table. All probes listed for the same `eid` are merged into one session-level dataset.

ii. ```python
eids = bwm_df['eid'].unique()

for eid_idx, eid in enumerate(eids):
    rows = bwm_df[bwm_df['eid'] == eid]
    pids = rows['pid'].values
    probe_names = rows['probe_name'].values
    ...
    if len(spikes_list) > 1:
        spikes, clusters = merge_probes(spikes_list, clusters_list)
```

iii. `CONVERSION_NOTES.md` says probe insertions are merged within a session. The trajectory-preserved reference code also works session-by-session over `eid` and merges all probes in `prepare_data`.

## 1-d. How are the data split into trials?

i. Trials come from `SessionLoader(...).load_trials()`. The agent first applies a task-trial mask, then applies a second behavior-validity mask; each retained row becomes one trial in the output.

ii. ```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials = sl.trials
...
masked_trials = trials[mask].reset_index(drop=True)
align_times = masked_trials['stimOn_times'].values
...
combined_mask = wheel_mask & whisker_mask
...
binned_spikes = binned_spikes[combined_mask]
```

iii. The notes justify trial filtering from the reference `load_trials_and_mask` logic. The second mask is justified implicitly by the agent's need to keep neural and behavioral arrays aligned after interpolation.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by reaction time, missing key trial fields, no-choice trials, and later by whether wheel and whisker signals can be interpolated with adequate coverage. Entire sessions are skipped if fewer than 10 masked trials remain before or after behavior masking.

ii. ```python
rt = trials['firstMovement_times'] - trials['stimOn_times']
if min_rt is not None:
    mask &= rt >= min_rt
if max_rt is not None:
    mask &= rt <= max_rt
for event in nan_exclude:
    mask &= ~trials[event].isnull()
mask &= trials['choice'] != 0

if mask.sum() < 10:
    ...

combined_mask = wheel_mask & whisker_mask
if combined_mask.sum() < 10:
    ...
```

iii. `CONVERSION_NOTES.md` explicitly justifies the RT, NaN, and no-choice filters from the reference code and papers. The trajectory-preserved reference code also shows `load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)`, but the agent did not mention that extra `max_trial_len=10.0` filter in its notes or code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from spike-sorting outputs: spike times and spike cluster IDs loaded per probe, plus cluster metadata for region labels.

ii. ```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
binned_spikes = bin_spikes_in_window(
    spikes['times'], spikes['clusters'], cluster_ids,
    align_times, WINDOW, BINSIZE
)
```

iii. The notes justify using all spike-sorted clusters because the reference `load_spiking_data(..., qc=None)` path returns all clusters. The trajectory preserves that exact reference function signature.

## 2-b. How is the `neural` data processed?

i. The agent merges probes within a session, sorts spikes by time, then bins spike counts into 20 ms bins over a 2 s window around stimulus onset. It outputs per-trial matrices of shape `(n_neurons, 100)`.

ii. ```python
if len(spikes_list) > 1:
    spikes, clusters = merge_probes(spikes_list, clusters_list)

sort_idx = np.argsort(spike_times)
sorted_times = spike_times[sort_idx]
sorted_clusters = spike_clusters[sort_idx]
...
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
```

iii. The notes say this matches the reference code and the method text that uses temporally binned spike counts. No extra smoothing, normalization, or firing-rate conversion is described by the agent.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not apply an explicit neuron QC threshold. It keeps all spike-sorted clusters that load successfully, then only filters neurons indirectly through session inclusion and the requirement that behavioral signals be alignable for the retained trials.

ii. ```python
def load_spiking_data(one, pid, eid='', pname=''):
    """Load spike sorting data for a probe insertion. Uses all clusters (no QC filter)
    matching the reference code's default behavior in prepare_data."""
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    return spikes, clusters_labeled
```

iii. `CONVERSION_NOTES.md` explicitly says the agent chose all clusters because the reference code calls `load_spiking_data` without a `qc` argument, leaving `qc=None`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to `stimOn_times`, with a per-trial window from `-0.5` s to `+1.5` s relative to stimulus onset.

ii. ```python
WINDOW = (-0.5, 1.5)
...
align_times = masked_trials['stimOn_times'].values
...
t_start = align_times[trial_idx] + window[0]
t_end = align_times[trial_idx] + window[1]
```

iii. The notes justify this from both the task instruction "Temporally align based on stimulus onset" and the preserved reference parameter block `{'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins. Spikes are rebinned into 100 non-overlapping bins across the 2 s trial window. No further temporal aggregation is applied after that.

ii. ```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))  # = 100
...
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` justifies 20 ms from the reference code and from the methods excerpt stating that 2 s trials are divided into 20 ms bins. The same notes also acknowledge that the methods text mentions 50 ms bins for choice/prior-specific decoding, but the agent resolved that ambiguity by following the reference code's universal `binsize=0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the chosen alignment event `stimOn_times` plus the fixed analysis window and bin size. The actual stored value is a relative-time template, not a per-trial raw timestamp vector.

ii. ```python
align_times = masked_trials['stimOn_times'].values
...
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes describe this as a continuous time variable "same for all trials" and tie it to the same bin centers used for behavioral interpolation.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent does not compute it from measured per-trial timestamps beyond choosing stimulus onset as the origin. It simply constructs a fixed length-100 vector of bin centers from `-0.48` s to `1.5` s.

ii. ```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly justifies the `linspace` choice as matching the bin centers used elsewhere in the pipeline.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned by construction: the same `WINDOW`, `BINSIZE`, and `N_BINS` are used for spike binning and for the time vector, and the vector is copied into every trial.

ii. ```python
inp = np.stack([time_input, trial_num], axis=0)  # (2, T)
session_input.append(inp.astype(np.float32))
...
session_neural.append(sm['binned_spikes'][trial].astype(np.float32))
```

iii. The notes say this input matches the neural bin centers. There is no additional interpolation step.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column in the session trial table.

ii. ```python
pLeft = trials_df['probabilityLeft'].values
...
trial_num_in_block = compute_trial_number_in_block(trials, mask.values)
```

iii. The notes describe it as the count within each block of constant `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent scans the full session's `probabilityLeft` sequence, resets the counter whenever `probabilityLeft` changes, uses 1-based counting within each block, then masks to retained trials and replicates the value across time bins for each trial.

ii. ```python
block_counter = 1
for i in range(len(pLeft)):
    if i == 0 or pLeft[i] != pLeft[i-1]:
        block_counter = 1
    trial_num[i] = block_counter
    block_counter += 1
...
trial_num = np.full(N_BINS, sm['trial_num_in_block'][trial], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly says it counts trials within blocks of constant `probabilityLeft`, starting from 1.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `Choice` is derived from the `choice` column in the masked trial table.

ii. ```python
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
```

iii. The notes cite the decoder specification `left = 0, right = 1` and note that the IBL raw encoding is `-1` for left and `1` for right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent first removes no-choice trials, then maps raw `choice == 1` to class `1` and every remaining nonzero value to class `0`. It later repeats that per-trial class across all 100 time bins.

ii. ```python
mask &= trials['choice'] != 0
...
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
...
choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` gives the left/right mapping as the justification. The replication across time is justified later in the notes under "Output Format".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column in the masked trial table.

ii. ```python
pLeft = masked_trials.iloc[masked_idx]['probabilityLeft'].values
```

iii. The notes explicitly state the mapping from `probabilityLeft` to the required 3-class output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent maps raw probabilities `0.2`, `0.5`, and `0.8` to classes `0`, `1`, and `2`, respectively, then repeats the class across all 100 time bins for each trial.

ii. ```python
prior = np.zeros(len(pLeft), dtype=np.int64)
prior[pLeft == 0.2] = 0
prior[pLeft == 0.5] = 1
prior[pLeft == 0.8] = 2
...
prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
```

iii. The notes justify the mapping directly from the task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from the wheel time series loaded by `SessionLoader`, specifically the `velocity` column of `sl.wheel`, together with the corresponding wheel timestamps.

ii. ```python
sl.load_wheel()
wheel_times = sl.wheel['times'].values
wheel_speed = np.abs(sl.wheel['velocity'].values)
```

iii. `CONVERSION_NOTES.md` explicitly cites the reference `load_target_behavior(..., 'wheel-speed')` logic, which uses the absolute wheel velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent converts velocity to speed with an absolute value, then linearly interpolates the session-wide wheel trace into the stimulus-aligned 20 ms trial bins. After a first pass across all sessions it computes global tercile thresholds and digitizes each binned value.

ii. ```python
wheel_speed = np.abs(sl.wheel['velocity'].values)
wheel_binned, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
)
...
wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])
...
wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
```

iii. The notes justify the absolute-velocity choice from the reference behavior loader and justify the interpolation from the preserved `get_behavior_per_interval` bin-center calculation. The tercile discretization is the agent's own choice to satisfy the task's categorical-output requirement.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is thresholded into three categories using global terciles computed over all wheel-speed values from all retained trials and time bins.

ii. ```python
all_wheel_flat = np.concatenate([w.flatten() for w in all_wheel_speed_raw])
wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])
...
wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
```

iii. The notes say "3 bins using global tercile thresholds." No further reference-based justification is given because the original reference problem appears to decode wheel speed as a continuous variable rather than as a discretized class label.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is aligned to the same stimulus-onset-centered 2 s window and resampled to the same 100 bin centers used for neural data.

ii. ```python
wheel_binned, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
)
...
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. The notes justify this using the reference interpolation helper's bin-center rule and the explicit task instruction to align the converted dataset to stimulus onset.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `sl.motion_energy['leftCamera']['whiskerMotionEnergy']` and its timestamps when available, with fallback to the right camera equivalents.

ii. ```python
sl.load_motion_energy(views=['left'])
me_times = sl.motion_energy['leftCamera']['times'].values
me_values = sl.motion_energy['leftCamera']['whiskerMotionEnergy'].values
...
sl.load_motion_energy(views=['right'])
me_times = sl.motion_energy['rightCamera']['times'].values
me_values = sl.motion_energy['rightCamera']['whiskerMotionEnergy'].values
```

iii. `CONVERSION_NOTES.md` explicitly says the agent follows the reference pattern of trying left whisker motion energy first and falling back to right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent loads the motion-energy trace from the left camera if possible, otherwise the right camera, linearly interpolates it into the stimulus-aligned 20 ms bins, then discretizes the binned values using global terciles computed over all retained sessions.

ii. ```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
...
whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])
...
whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)
```

iii. The notes justify the left-then-right fallback from the reference behavior loader and justify the interpolation from the reference binning helper.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded into three categories using global terciles over all interpolated whisker-motion-energy values.

ii. ```python
all_whisker_flat = np.concatenate([w.flatten() for w in all_whisker_me_raw])
whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])
...
whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)
```

iii. The notes say "3 bins using global tercile thresholds." As with wheel speed, this is a task-driven discretization choice rather than a clearly cited reference-paper step.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to the same stimulus-onset-centered trial window and sampled at the same 100 bin centers as the neural data.

ii. ```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
...
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. The notes justify this with the same interpolation rule used for wheel speed and by pointing to the task instruction to align the converted dataset to stimulus onset.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles missing or bad data by exclusion rather than imputation. Trials are excluded for NaN task fields or unusable behavior interpolation. Sessions are skipped when probes fail to load, whisker motion energy cannot be loaded, or too few valid trials remain. Behavior traces must cover the whole window within one bin and contain no NaNs.

ii. ```python
for event in nan_exclude:
    mask &= ~trials[event].isnull()
...
if len(bt) < 2:
    good_mask[trial_idx] = False
...
if np.any(np.isnan(bv)):
    good_mask[trial_idx] = False
...
if whisker_binned is None:
    print(f'  Skipping: no whisker motion energy')
    skipped_sessions += 1
    continue
```

iii. The notes justify NaN exclusion and behavior interpolation from the reference code. There is no explicit agent justification for the stronger session-level skip rules beyond practicality and keeping decoder inputs and outputs aligned.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are session-wise data loading from ONE, per-trial spike binning, per-trial behavioral interpolation for wheel and whisker traces, and storing the very large full dataset to disk.

ii. ```python
for eid_idx, eid in enumerate(eids):
    ...
    for pid, pname in zip(pids, probe_names):
        spks, clust = load_spiking_data(one, pid, eid=eid, pname=pname)
    ...
    binned_spikes = bin_spikes_in_window(...)
    wheel_binned, wheel_mask = interpolate_behavior_to_bins(...)
    whisker_binned, whisker_mask = interpolate_behavior_to_bins(...)
```

iii. The agent did not explicitly justify performance characteristics in its notes; this follows directly from the nested session/probe/trial loops and from the large output size reported in the trajectory.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the loop over trials in `bin_spikes_in_window`, the loop over trials in `interpolate_behavior_to_bins`, the loop over trials in `compute_trial_number_in_block`, and the final nested loops that assemble per-trial `input` and `output` arrays.

ii. ```python
for trial_idx in range(n_trials):
    ...

for trial_idx in range(n_trials):
    ...

for i in range(len(pLeft)):
    ...

for sm in session_meta:
    ...
    for trial in range(n_trials):
        ...
```

iii. No explicit justification is given by the agent. This is an evaluator inference from the implementation.

## 10-c. What processing does the code repeat multiple times?

i. It repeats linear interpolation logic separately for wheel and whisker data, repeatedly constructs per-trial constant arrays for choice, prior, and trial number, and does a two-pass workflow where raw behavioral values are first collected for global thresholds and then revisited for discretization.

ii. ```python
wheel_binned, wheel_mask = interpolate_behavior_to_bins(...)
...
whisker_binned, whisker_mask = interpolate_behavior_to_bins(...)
...
all_wheel_speed_raw.append(wheel_binned)
all_whisker_me_raw.append(whisker_binned)
...
choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
trial_num = np.full(N_BINS, sm['trial_num_in_block'][trial], dtype=np.float32)
```

iii. The agent did not discuss this in its notes. This is visible directly in `convert_data.py`.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script defines and partially maintains unused containers (`all_neural`, `all_input`, `all_output`, `all_subject_names`, `all_subject_idx`, `all_brain_region_idx`, `all_brain_regions_set`), defines an unused `discretize_to_bins` helper, stores session metadata fields like `lab` and `subject` that are not emitted in the final tensors, and temporarily keeps continuous wheel/whisker arrays only to discard them after discretization.

ii. ```python
all_neural = []
all_input = []
all_output = []
all_subject_names = []
all_subject_idx = []
all_brain_region_idx = []
all_brain_regions_set = set()
...
def discretize_to_bins(values, n_bins=3):
    ...
...
session_meta.append({
    'eid': eid,
    'subject': subject,
    'lab': lab,
    ...
    'wheel_binned': wheel_binned,
    'whisker_binned': whisker_binned,
})
```

iii. There is no explicit justification from the agent for these extra structures. This is evident from the implementation and from what the final output dictionary actually retains.
