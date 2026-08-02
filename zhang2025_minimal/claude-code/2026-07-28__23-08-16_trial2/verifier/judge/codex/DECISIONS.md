# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the IBL Brain Wide Map release table from `bwm_release.csv`, takes every unique session `eid`, and then loads per-session data through the ONE API. For each session it loads spike sorting for every probe listed in the release table, trial metadata through `SessionLoader.load_trials()`, wheel data through `SessionLoader.load_wheel()`, and whisker motion energy through `SessionLoader.load_motion_energy()`.

ii. 
```python
bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
eids = bwm_df['eid'].unique()

for eid_idx, eid in enumerate(eids):
    rows = bwm_df[bwm_df['eid'] == eid]
    pids = rows['pid'].values
    probe_names = rows['probe_name'].values

    for pid, pname in zip(pids, probe_names):
        spks, clust = load_spiking_data(one, pid, eid=eid, pname=pname)

    trials, mask, sl = load_trials_and_mask(one, eid)
    sl.load_wheel()
    sl.load_motion_energy(views=['left'])
```

iii. The justification in `CONVERSION_NOTES.md` is that the BWM release CSV is the reference session/probe inventory and that the agent wanted to follow the Zhang et al. code path using ONE, `SpikeSortingLoader`, and `SessionLoader`.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of the BWM release table. The agent builds a `subjects_list` of unique mouse IDs and a `subject_to_idx` mapping, then records one subject index per retained session.

ii. 
```python
rows = bwm_df[bwm_df['eid'] == eid]
subject = rows.iloc[0]['subject']

if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects_list)
    subjects_list.append(subject)
subj_idx = subject_to_idx[subject]

subject_idx_list.append(sm['subject_idx'])
```

iii. The notes justify this by treating the BWM release metadata as the authoritative source for subject IDs and by matching the target format requirement of `subjects` plus `subject_idx`.

## 1-c. How are the data split into sessions?

i. The agent defines one session per unique experiment ID `eid`. All probes that share that `eid` are treated as belonging to the same session and are merged before trialization.

ii. 
```python
eids = bwm_df['eid'].unique()

for eid_idx, eid in enumerate(eids):
    rows = bwm_df[bwm_df['eid'] == eid]
    pids = rows['pid'].values

    if len(spikes_list) > 1:
        spikes, clusters = merge_probes(spikes_list, clusters_list)
```

iii. `CONVERSION_NOTES.md` explicitly says the agent followed the reference decision to merge probes within a session because probes from the same behavioral session are not independent.

## 1-d. How are the data split into trials?

i. Trials come from the session `trials` table loaded by `SessionLoader`. After trial-level masking, each remaining row is treated as one trial. Neural and behavioral data are then segmented per trial using the chosen alignment event and time window.

ii. 
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials = sl.trials

masked_trials = trials[mask].reset_index(drop=True)
align_times = masked_trials['stimOn_times'].values

binned_spikes = bin_spikes_in_window(
    spikes['times'], spikes['clusters'], cluster_ids,
    align_times, WINDOW, BINSIZE
)
```

iii. The notes justify this as following the reference trial-aligned decoder code, where trial rows define the units of analysis and per-trial intervals are built from an alignment time plus a fixed window.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies the reference-style trial mask: remove trials with reaction time outside `[0.08, 2.0]` s, missing `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`, and no-choice trials (`choice == 0`). It then additionally removes trials whose wheel or whisker traces cannot support interpolation across the whole alignment window, and it skips sessions with fewer than 10 valid trials after filtering.

ii. 
```python
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= rt >= min_rt
mask &= rt <= max_rt

for event in nan_exclude:
    mask &= ~trials[event].isnull()

mask &= trials['choice'] != 0

wheel_binned, wheel_mask = interpolate_behavior_to_bins(...)
whisker_binned, whisker_mask = interpolate_behavior_to_bins(...)
combined_mask = wheel_mask & whisker_mask
```

iii. `CONVERSION_NOTES.md` cites the reference `load_trials_and_mask` behavior and the BWM paper’s trial exclusions. The later wheel/whisker masking is justified indirectly by the reference interpolation utility, which also marks intervals as invalid when target data are missing or do not span the full window.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` arrays are derived from the raw spike-sorting outputs: spike times and spike cluster assignments from `SpikeSortingLoader.load_spike_sorting()`, plus cluster metadata merged with channels via `SpikeSortingLoader.merge_clusters(...)`.

ii. 
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()

binned_spikes = bin_spikes_in_window(
    spikes['times'], spikes['clusters'], cluster_ids,
    align_times, WINDOW, BINSIZE
)
```

iii. The notes justify this by saying the agent followed the Zhang utility `load_spiking_data`, which uses all spike-sorted clusters by default and bins spike counts from the session.

## 2-b. How is the `neural` data processed?

i. The agent merges probes within a session, maps cluster acronyms to Beryl regions, then bins spike counts into non-overlapping 20 ms bins over a 2 s window from `-0.5` s to `+1.5` s around `stimOn_times`. Counts are stored per trial as `(n_neurons, 100)`.

ii. 
```python
WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))

beryl_reg = brainreg.acronym2acronym(clusters['acronym'].values, mapping='Beryl')

binned_spikes = bin_spikes_in_window(
    spikes['times'], spikes['clusters'], cluster_ids,
    align_times, WINDOW, BINSIZE
)
```

iii. `CONVERSION_NOTES.md` says this choice was based on the reference code parameters `align_time='stimOn_times'`, `time_window=(-.5, 1.5)`, and `binsize=0.02`, even though the paper text describes target-specific windows for some tasks.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not apply neuron-level QC filtering. It keeps all spike-sorted clusters that load successfully for each retained session.

ii. 
```python
def load_spiking_data(one, pid, eid='', pname=''):
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    return spikes, clusters_labeled
```

iii. The notes explicitly justify this by citing the reference code path where `prepare_data` calls `load_spiking_data` without a `qc` argument, so `qc=None` returns all clusters. The agent acknowledges that this differs from the BWM paper’s well-isolated-neuron criteria.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to stimulus onset, using `stimOn_times` as the event and extracting neural activity from `-0.5` s to `+1.5` s around that event.

ii. 
```python
WINDOW = (-0.5, 1.5)
align_times = masked_trials['stimOn_times'].values

t_start = align_times[trial_idx] + window[0]
t_end = align_times[trial_idx] + window[1]
```

iii. `CONVERSION_NOTES.md` states this was chosen because the task instructions explicitly said to align based on stimulus onset, and because the reference aligned-cache script uses the same `stimOn_times` window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms non-overlapping bins, giving 100 time bins across the 2 s window. No further temporal rebinning is applied after spike counting.

ii. 
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))  # = 100

bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
```

iii. The notes justify 20 ms bins by citing the reference code and the generic methods text about 2 s trials divided into 20 ms bins, while also acknowledging that the paper text mentions 50 ms bins for choice and prior decoding.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This input is not loaded directly from a raw recorded variable. It is constructed from the chosen alignment event `stimOn_times` together with the fixed decoding window and bin size.

ii. 
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes justify this as a task-required decoder input: a continuous time variable anchored to stimulus onset and matched to the neural time grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent computes a single relative-time vector from `-0.48` s to `1.5` s using `np.linspace`, then reuses that same vector for every trial in the session.

ii. 
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)

time_input = sm['time_since_stim'].copy()
inp = np.stack([time_input, trial_num], axis=0)
```

iii. `CONVERSION_NOTES.md` explicitly says this was intended to match the interpolation time grid used in the reference code and to satisfy the decoder-task requirement.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned exactly to the neural bins by using the same window and the same 20 ms time grid as the spike-count matrices.

ii. 
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)

session_neural.append(sm['binned_spikes'][trial].astype(np.float32))
time_input = sm['time_since_stim'].copy()
```

iii. The notes justify this by saying the time input should reflect the same temporal grid used for neural and behavioral interpolation.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial-wise `probabilityLeft` values in the raw trials table.

ii. 
```python
pLeft = trials_df['probabilityLeft'].values
```

iii. The notes say this input is meant to encode trial position within blocks of constant left prior, so `probabilityLeft` is the raw variable that defines the blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent scans the full session trial table in order, resets a counter whenever `probabilityLeft` changes, assigns a 1-indexed within-block trial number, masks out excluded trials, and then repeats the retained scalar value across all 100 time bins within each output trial.

ii. 
```python
block_counter = 1
for i in range(len(pLeft)):
    if i == 0 or pLeft[i] != pLeft[i-1]:
        block_counter = 1
    trial_num[i] = block_counter
    block_counter += 1

return trial_num[mask]
```

iii. The notes justify this as a direct implementation of “trial number in block” from the task specification rather than something taken from the reference codebase.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials table.

ii. 
```python
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
```

iii. The notes say the decoder task requires left/right choice and that the IBL `choice` field already contains the signed choice code after excluding no-choice trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent converts the signed IBL choice code into the requested binary label, mapping `choice == 1` to `1` and everything else that remains after filtering (that is, `-1`) to `0`. It then replicates that scalar across all 100 time bins of the trial.

ii. 
```python
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)

choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` justifies the binary mapping from the decoder-task specification `left = 0, right = 1`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from the trial-wise `probabilityLeft` field.

ii. 
```python
pLeft = masked_trials.iloc[masked_idx]['probabilityLeft'].values
```

iii. The notes explicitly say the agent used `probabilityLeft` as the raw prior variable because the decoder task asked for prior probability of left with categories `0.2`, `0.5`, and `0.8`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent maps the continuous probabilities to class labels `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then repeats the resulting class across all time bins in each trial.

ii. 
```python
prior = np.zeros(len(pLeft), dtype=np.int64)
prior[pLeft == 0.2] = 0
prior[pLeft == 0.5] = 1
prior[pLeft == 0.8] = 2

prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
```

iii. The notes justify this entirely from the decoder-task specification, which asked for those three discrete output values.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the session wheel time series loaded by `SessionLoader.load_wheel()`, specifically from `wheel['times']` and the absolute value of `wheel['velocity']`.

ii. 
```python
sl.load_wheel()
wheel_times = sl.wheel['times'].values
wheel_speed = np.abs(sl.wheel['velocity'].values)
```

iii. The notes justify this by citing the reference helper `load_target_behavior(..., 'wheel-speed')`, which also uses absolute wheel velocity.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent interpolates absolute wheel velocity into the same 20 ms bins used for neural data, over the stimulus-onset-centered 2 s trial window. It later discretizes the interpolated values with global tercile thresholds computed across all sessions and time bins.

ii. 
```python
wheel_binned, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
)

all_wheel_flat = np.concatenate([w.flatten() for w in all_wheel_speed_raw])
wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])
wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
```

iii. The notes justify the source signal and interpolation pattern by reference to `load_target_behavior` and `get_behavior_per_interval`, but the tercile discretization is justified only as a way to satisfy the task’s required 3-bin categorical output.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. The agent computes two global percentile thresholds at the 33.3rd and 66.7th percentiles across all retained wheel-speed samples from all sessions, then applies `np.digitize` to produce labels `0`, `1`, and `2`.

ii. 
```python
wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])
wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says the wheel-speed output was discretized into 3 bins using global tercile thresholds because the decoder task required a categorical output but the reference wheel signal is continuous.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the neural data by interpolating the wheel trace onto the same stimulus-onset-aligned 20 ms time grid as the neural matrices.

ii. 
```python
wheel_binned, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
)

x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. The notes justify the interpolation rule from the reference behavior-alignment helper, but the choice of stimulus-onset alignment comes from the task instructions and the aligned-cache script, not from the dynamic-behavior section of the paper.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from the motion-energy time series loaded from the left camera when available, otherwise the right camera, using the `whiskerMotionEnergy` column and its timestamps.

ii. 
```python
sl.load_motion_energy(views=['left'])
me_times = sl.motion_energy['leftCamera']['times'].values
me_values = sl.motion_energy['leftCamera']['whiskerMotionEnergy'].values

sl.load_motion_energy(views=['right'])
me_times = sl.motion_energy['rightCamera']['times'].values
me_values = sl.motion_energy['rightCamera']['whiskerMotionEnergy'].values
```

iii. The notes justify this with the reference loader, which also tries the left whisker-motion-energy trace first and falls back to the right trace if needed.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent linearly interpolates whisker motion energy to the same stimulus-onset-aligned 20 ms time grid as the neural data, then discretizes the interpolated values using global tercile thresholds across all retained samples.

ii. 
```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)

all_whisker_flat = np.concatenate([w.flatten() for w in all_whisker_me_raw])
whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])
whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)
```

iii. The notes justify the source signal and interpolation from the reference code, while the discretization is an added decision made to satisfy the target categorical format.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The agent computes global whisker-motion-energy terciles over all sessions and time bins, then digitizes each interpolated sample into categories `0`, `1`, or `2`.

ii. 
```python
whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])
whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says this was chosen because the task required a 3-bin categorical whisker-motion-energy output, while the raw/reference signal is continuous.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned by interpolating the motion-energy trace to the same 20 ms stimulus-onset-centered time grid used for neural activity.

ii. 
```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)

x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. The notes justify the interpolation time grid from the reference behavior-alignment utility, while the use of stimulus onset comes from the global alignment choice for this converted dataset.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or bad data are mostly handled by exclusion rather than imputation. Failed probes are skipped; sessions with no usable probes, no whisker motion energy, or too few valid trials are skipped; trials with bad wheel/whisker coverage or NaNs are removed via masks. The script does not repair all-zero neural trials; they remain in the final dataset if they survive filtering.

ii. 
```python
except Exception as e:
    print(f'  Warning: Failed to load probe {pid}: {e}')

if len(spikes_list) == 0:
    skipped_sessions += 1
    continue

if whisker_binned is None:
    skipped_sessions += 1
    continue

if len(bt) < 2 or np.any(np.isnan(bv)):
    good_mask[trial_idx] = False
```

iii. The notes justify these choices as consistency-preserving: the agent preferred skipping unusable probes/trials/sessions rather than fabricating missing neural or behavioral values.

## 12-a. What are the most time-consuming steps of the code?

i. The main expensive steps are session-by-session I/O through ONE, per-trial spike binning in `bin_spikes_in_window`, per-trial behavioral interpolation in `interpolate_behavior_to_bins`, and the second pass that reconstructs nested trial lists for every session.

ii. 
```python
for eid_idx, eid in enumerate(eids):
    ...
    for pid, pname in zip(pids, probe_names):
        spks, clust = load_spiking_data(...)
    ...
    binned_spikes = bin_spikes_in_window(...)
    wheel_binned, wheel_mask = interpolate_behavior_to_bins(...)
    whisker_binned, whisker_mask = interpolate_behavior_to_bins(...)

for sm in session_meta:
    for trial in range(n_trials):
        session_neural.append(...)
        session_input.append(...)
        session_output.append(...)
```

iii. There is no explicit efficiency justification in the notes beyond “uses vectorized numpy operations for speed” inside the custom spike-binning helper. The trajectory and notes focus more on correctness and passing decoder validation than on runtime.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `bin_spikes_in_window`, the per-trial loop in `interpolate_behavior_to_bins`, the Python loop in `compute_trial_number_in_block`, and the final nested session/trial assembly loop could all have been reduced or parallelized further. The code is partly vectorized within each trial, but not across trials.

ii. 
```python
for trial_idx in range(n_trials):
    ...

for trial_idx in range(n_trials):
    ...

for i in range(len(pLeft)):
    ...

for sm in session_meta:
    for trial in range(n_trials):
        ...
```

iii. The agent did not explicitly justify these loops. Relative to the reference utilities, it replaced multiprocessing with simpler serial Python loops, likely for implementation simplicity.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly scans all sessions once to collect raw/interpolated data and again to build the nested output structure; it computes wheel and whisker interpolation separately even though the interpolation logic is the same; and it repeatedly expands static trial variables (`choice`, `prior`, `trial number`) into full-length per-time-bin vectors.

ii. 
```python
# First pass: collect all data and raw behavior values
for eid_idx, eid in enumerate(eids):
    ...
    wheel_binned, wheel_mask = interpolate_behavior_to_bins(...)
    whisker_binned, whisker_mask = interpolate_behavior_to_bins(...)
    session_meta.append({...})

# Now build the final data structure
for sm in session_meta:
    ...
    choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
    prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
```

iii. The notes do not explicitly discuss this repetition. It appears to be a convenience-oriented design: first gather everything needed for global thresholds, then assemble the final nested format in a second pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script defines `discretize_to_bins()` but never uses it, creates several unused accumulator variables (`all_neural`, `all_input`, `all_output`, `all_subject_names`, `all_subject_idx`, `all_brain_region_idx`, `all_brain_regions_set`), imports `ismember` without using it, and duplicates static per-trial labels across 100 bins even though their values do not vary within a trial.

ii. 
```python
all_neural = []
all_input = []
all_output = []
all_subject_names = []
all_subject_idx = []
all_brain_region_idx = []
all_brain_regions_set = set()

def discretize_to_bins(values, n_bins=3):
    ...

choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
```

iii. There is no explicit justification for these extra pieces in the notes. The strongest implicit justification is format uniformity: the agent chose to make every output time-varying, even when that duplicates static labels across bins.
