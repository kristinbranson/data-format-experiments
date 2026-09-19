# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI starts from `/app/code/code_zhang2025/data/bwm_release.csv`, takes the unique `eid` values as sessions, then iterates session-by-session. For each session it uses the CSV rows to find probe IDs and subject/lab metadata, loads spike sorting per probe with `SpikeSortingLoader`, and loads the trial table and behavior streams with `SessionLoader`. It does not use `one.search(...)` to select locally available sessions and does not handle the subset CSV logic from the reference.

ii.
```python
bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
eids = bwm_df['eid'].unique()
```

```python
rows = bwm_df[bwm_df['eid'] == eid]
spks, clust = load_spiking_data(one, pid, eid=eid, pname=pname)
trials, mask, sl = load_trials_and_mask(one, eid)
```

iii. In the trajectory, the AI justified this by saying the BWM release CSV listed 699 probes across 459 sessions and that it would process sessions from that table, then use ONE/ibllib loaders to resolve the actual files. In its notes it described the pipeline as “ONE API with cache_dir=/app/data/one_cache” plus the BWM release CSV as the session inventory.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken directly from the `subject` column of the BWM release CSV. The AI assigns subject indices in first-seen order while iterating sessions.

ii.
```python
subject = rows.iloc[0]['subject']
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects_list)
    subjects_list.append(subject)
subj_idx = subject_to_idx[subject]
```

iii. In the trajectory, the AI said the BWM release CSV already contains the subject identifiers, so it reused that metadata instead of deriving subjects from paths or separate API queries.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique `eid` values from the BWM release CSV. Each unique `eid` is processed once.

ii.
```python
eids = bwm_df['eid'].unique()
for eid_idx, eid in enumerate(eids):
```

iii. The trajectory shows the AI treating the release CSV as the session list and describing the dataset as 459 unique sessions identified by `eid`.

## 1-d. How are the data split into trials?

i. Trials come from `SessionLoader(...).load_trials()`. After loading the trials table, the AI uses the rows that survive the trial mask, then applies a second behavior-validity mask, so the final trial split is the filtered rows of the trials table.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials = sl.trials
```

```python
masked_trials = trials[mask].reset_index(drop=True)
align_times = masked_trials['stimOn_times'].values
masked_idx = np.where(combined_mask)[0]
```

iii. In the trajectory, the AI treated the trials table as the per-trial unit and described sample/full outputs in terms of surviving trials per session after filtering.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out trials with reaction time outside `[0.08, 2.0]` s, NaNs in several key trial fields, and no-choice trials (`choice == 0`). It then drops trials whose wheel or whisker traces cannot be interpolated across the full trial window, by requiring behavior coverage masks from the interpolation helper. It also skips whole sessions with fewer than 10 valid trials before or after the behavior mask.

ii.
```python
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= rt >= min_rt
mask &= rt <= max_rt
for event in nan_exclude:
    mask &= ~trials[event].isnull()
mask &= trials['choice'] != 0
```

```python
wheel_binned, wheel_mask = interpolate_behavior_to_bins(...)
whisker_binned, whisker_mask = interpolate_behavior_to_bins(...)
combined_mask = wheel_mask & whisker_mask
```

```python
if mask.sum() < 10:
    ...
if combined_mask.sum() < 10:
    ...
```

iii. In the trajectory and `CONVERSION_NOTES.md`, the AI justified this as matching the reference RT bounds and excluding NaN key events and no-choice trials. It also said sessions missing whisker motion energy or with too few valid trials should be skipped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is built from the spike sorting output: spike times and spike cluster assignments for each probe insertion. Cluster metadata is also loaded for acronyms/labels, but the actual neural tensor is built from `spikes['times']` and `spikes['clusters']`.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

```python
binned_spikes = bin_spikes_in_window(
    spikes['times'], spikes['clusters'], cluster_ids,
    align_times, WINDOW, BINSIZE
)
```

iii. In the trajectory, the AI justified this as following the reference spike-sorted data path through `SpikeSortingLoader`, with probe metadata coming from the release CSV and ONE cache.

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session, bins spikes into 20 ms bins over a `[-0.5, 1.5]` s window around stimulus onset, and stores the result as per-trial `(n_neurons, 100)` arrays. It keeps raw spike counts; it does not convert them to firing rates by dividing by the bin width.

ii.
```python
if len(spikes_list) > 1:
    spikes, clusters = merge_probes(spikes_list, clusters_list)
```

```python
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
```

```python
session_neural.append(sm['binned_spikes'][trial].astype(np.float32))
```

iii. In the trajectory, the AI described this as “probe merging” and “spike binning” with 20 ms bins and 100 time steps. It also later optimized `bin_spikes_in_window` for speed, which shows it considered this explicit per-trial binning step the core neural processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply any neuron QC filter. It loads all spike-sorted clusters, keeps them all, and maps all cluster acronyms to Beryl regions without excluding low-QC units or `void` regions.

ii.
```python
def load_spiking_data(one, pid, eid='', pname=''):
    """Load spike sorting data for a probe insertion. Uses all clusters (no QC filter)
    matching the reference code's default behavior in prepare_data."""
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
```

```python
beryl_reg = brainreg.acronym2acronym(clusters['acronym'].values, mapping='Beryl')
```

iii. In the trajectory and `CONVERSION_NOTES.md`, the AI explicitly justified this as matching the reference code’s default `qc=None`, arguing that the decoder task should use all clusters rather than only “good” units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`, with a window from `-0.5` s to `+1.5` s around stimulus onset.

ii.
```python
WINDOW = (-0.5, 1.5)
align_times = masked_trials['stimOn_times'].values
```

```python
t_start = align_times[trial_idx] + window[0]
t_end = align_times[trial_idx] + window[1]
```

iii. In the trajectory, the AI repeatedly justified this as matching both the task instruction (“Temporally align based on stimulus onset”) and the method/reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 20 ms bins for a 2 s window, giving 100 bins per trial. No secondary temporal rebinning is applied.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))
```

```python
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
```

iii. In the trajectory and notes, the AI justified 20 ms bins as matching the methods paper and reference code.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The actual values are synthesized from the chosen decoding window and bin size, but they are tied to each trial’s `stimOn_times` because the whole dataset is aligned to that event.

ii.
```python
align_times = masked_trials['stimOn_times'].values
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

iii. In the trajectory, the AI justified this as a bin-center time axis for trials aligned to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes one fixed vector of bin centers from `-0.48` to `1.5` s using `np.linspace(...)`, then reuses that same vector for every trial.

ii.
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

```python
time_input = sm['time_since_stim'].copy()
inp = np.stack([time_input, trial_num], axis=0)
```

iii. In the trajectory notes, the AI said this was “same for all trials” and meant to match the bin centers used for behavior interpolation.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It shares the same 100-bin trial grid as the neural data. The time vector is stacked into the per-trial input arrays using the same `N_BINS` used for spike binning.

ii.
```python
N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))
```

```python
time_input = sm['time_since_stim'].copy()
trial_num = np.full(N_BINS, sm['trial_num_in_block'][trial], dtype=np.float32)
inp = np.stack([time_input, trial_num], axis=0)
```

iii. In the trajectory, the AI justified this as a common time basis for neural and behavioral signals.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials['probabilityLeft']`. A block boundary is defined when `probabilityLeft` changes from one trial to the next.

ii.
```python
pLeft = trials_df['probabilityLeft'].values
for i in range(len(pLeft)):
    if i == 0 or pLeft[i] != pLeft[i-1]:
        block_counter = 1
```

iii. In the trajectory notes, the AI justified this by saying block identity is implicit in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counts trials within each block of constant `probabilityLeft`, resets the count at block changes, and keeps the numbering 1-indexed. It computes the count on the full trial table and only then applies the trial mask.

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

iii. In `CONVERSION_NOTES.md`, the AI explicitly justified this as “Count of trial within its block of constant probabilityLeft, starting from 1.”

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from the `choice` column of the filtered trials table.

ii.
```python
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
```

iii. In the trajectory notes, the AI justified this as direct use of the IBL trial `choice` variable, then recoding it to the task’s requested binary labels.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI recodes trial `choice` values into a binary variable and then replicates the result across all time bins of the trial. Its chosen mapping is effectively `choice == 1 -> 1`, all other kept choices -> `0`, and its notes state that IBL `choice=-1` means left and `choice=1` means right.

ii.
```python
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
```

```python
choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by saying the decoder task wanted left `= 0`, right `= 1`, and that “In IBL data, choice=-1 is left, choice=1 is right.”

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `probabilityLeft` in the filtered trials table.

ii.
```python
pLeft = masked_trials.iloc[masked_idx]['probabilityLeft'].values
```

iii. In the trajectory notes, the AI justified this as the task’s prior/block variable from the IBL trials table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then replicates the per-trial class across all time bins.

ii.
```python
prior = np.zeros(len(pLeft), dtype=np.int64)
prior[pLeft == 0.2] = 0
prior[pLeft == 0.5] = 1
prior[pLeft == 0.8] = 2
```

```python
prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
```

iii. In the trajectory notes, the AI justified this as directly required by the decoder task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The AI uses the wheel stream loaded by `SessionLoader.load_wheel()`, specifically `sl.wheel['times']` and `sl.wheel['velocity']`, then takes the absolute value of velocity as speed.

ii.
```python
sl.load_wheel()
wheel_times = sl.wheel['times'].values
wheel_speed = np.abs(sl.wheel['velocity'].values)
```

iii. In the trajectory notes, the AI justified this as matching the reference behavior loader, which computes wheel speed as absolute wheel velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI takes absolute wheel velocity, interpolates it into the 100 trial bins, stores the continuous values, and later discretizes them.

ii.
```python
wheel_speed = np.abs(sl.wheel['velocity'].values)
wheel_binned, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
)
```

```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

iii. In the trajectory, the AI justified this as following the reference interpolation scheme and using wheel speed rather than signed velocity.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes wheel-speed tercile thresholds globally across all retained sessions and all time bins, then digitizes each session’s wheel trace against those global edges.

ii.
```python
all_wheel_speed_raw.append(wheel_binned)
...
all_wheel_flat = np.concatenate([w.flatten() for w in all_wheel_speed_raw])
wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])
```

```python
wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly justified this as “3 bins using global tercile thresholds.”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated onto the same 100-bin grid used for the aligned neural trials.

ii.
```python
wheel_binned, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
)
```

```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. In the trajectory notes, the AI justified this as matching the neural trial bins and the reference behavior interpolation function.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from camera motion-energy time series: the AI tries left camera first, then right camera as a fallback, using the camera time vector and whisker motion energy column from `sl.motion_energy`.

ii.
```python
sl.load_motion_energy(views=['left'])
me_times = sl.motion_energy['leftCamera']['times'].values
me_values = sl.motion_energy['leftCamera']['whiskerMotionEnergy'].values
```

```python
sl.load_motion_energy(views=['right'])
me_times = sl.motion_energy['rightCamera']['times'].values
me_values = sl.motion_energy['rightCamera']['whiskerMotionEnergy'].values
```

iii. In the trajectory notes, the AI justified this as matching the reference behavior loader’s left-first, right-fallback strategy.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released whisker motion energy trace directly, interpolates it onto the trial bin grid, stores the continuous values, and later discretizes them.

ii.
```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
```

```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
```

iii. In the trajectory, the AI justified this as a direct interpolation step with no extra filtering or normalization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The AI computes whisker-motion-energy tercile thresholds globally across all retained sessions and all time bins, then digitizes each session against those global edges.

ii.
```python
all_whisker_me_raw.append(whisker_binned)
...
all_whisker_flat = np.concatenate([w.flatten() for w in all_whisker_me_raw])
whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])
```

```python
whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly justified this as “3 bins using global tercile thresholds.”

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is interpolated onto the same 100-bin grid used for the aligned neural trials.

ii.
```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
```

```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. In the trajectory notes, the AI justified this as matching the reference bin centers used for neural-aligned behavior.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data mostly by skipping it. Failed probe loads are warned and ignored; sessions with no successfully loaded probes are skipped; sessions with missing whisker motion energy are skipped; trials with bad wheel/whisker interpolation coverage or NaNs are dropped; sessions with too few valid trials are skipped.

ii.
```python
for pid, pname in zip(pids, probe_names):
    try:
        spks, clust = load_spiking_data(one, pid, eid=eid, pname=pname)
    except Exception as e:
        print(f'  Warning: Failed to load probe {pid}: {e}')
```

```python
if len(spikes_list) == 0:
    ...
if whisker_binned is None:
    ...
if combined_mask.sum() < 10:
    ...
```

```python
if len(bt) < 2:
    good_mask[trial_idx] = False
...
if np.any(np.isnan(bv)):
    good_mask[trial_idx] = False
```

iii. In the trajectory, the AI justified this as expected handling for sessions missing whisker motion energy and as part of its stricter validity filtering before decoder training.

## 10-a. What are the most time-consuming steps of the code?

i. The code is dominated by session-by-session probe loading and per-trial spike binning/interpolation. The trajectory shows the AI focusing optimization effort on `bin_spikes_in_window`, which implies it identified that section as a major cost in addition to loading large spike-sorting files.

ii.
```python
spks, clust = load_spiking_data(one, pid, eid=eid, pname=pname)
```

```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
```

iii. In the trajectory, the AI explicitly rewrote the original spike-binning code “for speed” using `searchsorted` and `np.add.at`, which indicates it treated spike binning as a hot path.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining candidates are the per-trial loop in `bin_spikes_in_window`, the per-trial loop in `interpolate_behavior_to_bins`, the trial-number loop in `compute_trial_number_in_block`, and repeated per-session filtering of the release DataFrame.

ii.
```python
for trial_idx in range(n_trials):
    ...
```

```python
for trial_idx in range(n_trials):
    ...
    binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

```python
for i in range(len(pLeft)):
    ...
```

iii. The trajectory shows the AI already vectorizing away an inner per-spike loop, but it left the larger per-trial loops in place.

## 10-c. What processing does the code repeat multiple times?

i. The AI does a two-pass treatment of wheel speed and whisker motion energy: first it stores every session’s continuous traces to compute global thresholds, then it loops over sessions again to discretize them and build final outputs. It also repeatedly filters `bwm_df` by `eid` inside the session loop.

ii.
```python
rows = bwm_df[bwm_df['eid'] == eid]
```

```python
all_wheel_speed_raw.append(wheel_binned)
all_whisker_me_raw.append(whisker_binned)
...
wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)
```

iii. In the trajectory notes, the AI justified the repeated pass by its choice to use global tercile thresholds for the movement variables.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps full continuous wheel and whisker traces in `session_meta` and in the `all_*_raw` lists only to compute global thresholds and build the discretized outputs later; those continuous traces are not saved in the final dataset. It also defines several top-level accumulator lists that are never used.

ii.
```python
all_neural = []
all_input = []
all_output = []
all_subject_names = []
all_subject_idx = []
all_brain_region_idx = []
all_brain_regions_set = set()
```

```python
all_wheel_speed_raw.append(wheel_binned)
all_whisker_me_raw.append(whisker_binned)
...
'wheel_binned': wheel_binned,
'whisker_binned': whisker_binned,
```

iii. The trajectory does not present this as a deliberate optimization tradeoff; it is mainly a consequence of the AI’s choice to discretize wheel speed and whisker motion energy globally in a second pass.
