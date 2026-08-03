# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates sessions from `/app/code/code_zhang2025/data/bwm_release.csv`, takes unique `eid` values from that CSV, then uses the ONE API plus `SpikeSortingLoader` and `SessionLoader` to load per-probe spike sorting and per-session trials/behavior. It does not use `one.search(...)` to define the released/locally available session set the way the reference solution does.

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

iii. The script header and `CONVERSION_NOTES.md` say it is loading IBL Brain Wide Map data through ONE/ibllib, but the actual session inventory comes from the release CSV rather than from `one.search(...)`. The trajectory memory also records the release CSV path as the session source.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of the release CSV. A subject index is assigned in first-seen order as sessions are processed, and each session stores that integer in `subject_idx`.

ii. 
```python
subject = rows.iloc[0]['subject']
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects_list)
    subjects_list.append(subject)
subj_idx = subject_to_idx[subject]
```

iii. The README and notes say subject identities come from the BWM release metadata. No separate justification beyond using release metadata appears in the trajectory.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values from the release CSV. The conversion loops once per unique `eid`; sessions can later be skipped if probes or required behavioral streams cannot be loaded, or if too few valid trials remain.

ii. 
```python
eids = bwm_df['eid'].unique()
for eid_idx, eid in enumerate(eids):
    rows = bwm_df[bwm_df['eid'] == eid]
```

```python
if len(spikes_list) == 0:
    print(f'  Skipping: no probes loaded')
    skipped_sessions += 1
    continue
```

iii. The AI treats the release CSV as the authoritative session list. This is consistent with its trajectory notes, which cite the CSV as containing 699 probes across 459 sessions.

## 1-d. How are the data split into trials?

i. Trials come from `SessionLoader.load_trials()`. The trial table rows are filtered by a boolean mask, then further filtered by wheel/whisker coverage masks, leaving the surviving rows as the per-session trial list.

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

iii. The AI did not justify trial splitting separately; it follows the row-per-trial structure provided by `SessionLoader`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages. First, `load_trials_and_mask` keeps trials with reaction time in `[0.08, 2.0]`, non-null `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`, and `choice != 0`. Second, behavior interpolation computes `wheel_mask` and `whisker_mask`; only trials with valid coverage in both streams are retained. Sessions with fewer than 10 surviving trials are skipped.

ii. 
```python
rt = trials['firstMovement_times'] - trials['stimOn_times']
if min_rt is not None:
    mask &= rt >= min_rt
if max_rt is not None:
    mask &= rt <= max_rt

for event in nan_exclude:
    mask &= ~trials[event].isnull()

mask &= trials['choice'] != 0
```

```python
combined_mask = wheel_mask & whisker_mask
if combined_mask.sum() < 10:
    print(f'  Skipping: only {combined_mask.sum()} trials with valid behavior')
    skipped_sessions += 1
    continue
```

iii. `CONVERSION_NOTES.md` explicitly justifies the RT, NaN-event, and no-choice filters as matching the paper/reference code. It does not call out the later behavior-coverage filter or the 10-trial minimum, which are only visible in the code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural tensor is derived from spike times and cluster assignments loaded by `SpikeSortingLoader`. The cluster table is also merged in to obtain cluster labels and acronyms, but the actual trialwise neural tensor is built from `spikes['times']` and `spikes['clusters']`.

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

iii. The header comments and notes say neural data come from spike-sorted clusters loaded through `SpikeSortingLoader`.

## 2-b. How is the `neural` data processed?

i. Probe insertions within a session are merged, cluster ids are renumbered across probes, then spikes are counted into 20 ms bins in a `[-0.5, 1.5]` window around stimulus onset. The resulting arrays are stored as `float32` counts. Unlike the human reference, the code does not divide by bin width to convert counts to firing rates in Hz.

ii. 
```python
if len(spikes_list) > 1:
    spikes, clusters = merge_probes(spikes_list, clusters_list)
```

```python
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
np.clip(bin_indices, 0, n_bins - 1, out=bin_indices)
np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
```

```python
session_neural.append(sm['binned_spikes'][trial].astype(np.float32))
```

iii. The notes justify 20 ms temporal binning and probe merging as matching the papers/reference, but they do not mention that the stored neural values are raw counts rather than Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps all spike-sorted clusters and applies no neuron QC filter. It still maps every cluster to a Beryl region, but does not drop clusters based on `label`.

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

iii. `CONVERSION_NOTES.md` explicitly defends this as using the reference code's "default qc=None" and therefore including all clusters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. For each retained trial, spike times are restricted to `[stimOn - 0.5, stimOn + 1.5]` and binned relative to the start of that trial window.

ii. 
```python
WINDOW = (-0.5, 1.5)  # seconds relative to stimOn_times
align_times = masked_trials['stimOn_times'].values
```

```python
t_start = align_times[trial_idx] + window[0]
t_end = align_times[trial_idx] + window[1]
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
```

iii. The notes justify stimulus-onset alignment from both the task instructions and the methods paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins across a 2 s window, for 100 bins per trial. No additional temporal rebinning is applied after that binning/interpolation step.

ii. 
```python
WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))
```

iii. `CONVERSION_NOTES.md` explicitly justifies 20 ms bins from the methods paper and reference code.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not loaded as a raw data column. The AI constructs it from the chosen alignment convention around `stimOn_times`: a fixed 100-point relative-time vector reused for every trial in a session.

ii. 
```python
align_times = masked_trials['stimOn_times'].values
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes justify this as "time since stimulus onset" and say it is the same for all trials.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes a fixed linearly spaced vector from `-0.48` s to `1.5` s with 100 points and broadcasts it to every trial. It describes this as matching bin centers, although the code uses right-shifted points relative to true 20 ms bin centers.

ii. 
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says this is "linspace from -0.48 to 1.5, matching bin centers."

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The code uses the same 100-step relative-time vector for every trial as the intended neural/behavior time axis, and wheel/whisker interpolation is evaluated on the same grid. In practice that grid is shifted 10 ms later than the neural bin centers implied by the spike-count window.

ii. 
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. The notes justify this as matching the behavioral interpolation grid and bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the trials table. A block boundary is defined wherever `probabilityLeft` changes from one trial to the next.

ii. 
```python
pLeft = trials_df['probabilityLeft'].values
for i in range(len(pLeft)):
    if i == 0 or pLeft[i] != pLeft[i-1]:
        block_counter = 1
```

iii. The notes state that trial number in block is computed from blocks of constant `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI loops through the full session trial table, resets a counter whenever `probabilityLeft` changes, stores a 1-indexed trial count within each block, then applies the trial masks. The value is then repeated across all 100 bins of each surviving trial.

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

```python
trial_num = np.full(N_BINS, sm['trial_num_in_block'][trial], dtype=np.float32)
inp = np.stack([time_input, trial_num], axis=0)
```

iii. `CONVERSION_NOTES.md` explicitly says this count starts from 1.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials table after the no-choice (`0`) trials have been removed.

ii. 
```python
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
```

iii. The notes describe choice as coming from IBL trial choice values and being recoded to a binary left/right output.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After no-choice trials are excluded, the code maps `choice == 1` to output class `1` and everything else to `0`. In effect, it treats `+1` as right and `-1` as left.

ii. 
```python
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` justifies this by asserting that in IBL data `choice=-1` is left and `choice=1` is right.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior is derived from the `probabilityLeft` column of the trials table.

ii. 
```python
pLeft = masked_trials.iloc[masked_idx]['probabilityLeft'].values
```

iii. The notes say this output is the per-trial prior probability of left from the task blocks.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `probabilityLeft` values `0.2`, `0.5`, and `0.8` to categorical codes `0`, `1`, and `2`.

ii. 
```python
prior = np.zeros(len(pLeft), dtype=np.int64)
prior[pLeft == 0.2] = 0
prior[pLeft == 0.5] = 1
prior[pLeft == 0.8] = 2
```

iii. `CONVERSION_NOTES.md` cites the decoder-task specification for this mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the wheel stream loaded by `SessionLoader.load_wheel()`, specifically the loader's `times` and `velocity` outputs. Those are upstream products of the raw wheel position/timestamp files.

ii. 
```python
sl.load_wheel()
wheel_times = sl.wheel['times'].values
wheel_speed = np.abs(sl.wheel['velocity'].values)
```

iii. The notes justify this as matching the reference code's use of absolute wheel velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code takes absolute wheel velocity and linearly interpolates it onto a 100-point trial grid for each retained trial. No extra smoothing is implemented in this script beyond whatever `SessionLoader.load_wheel()` has already done internally.

ii. 
```python
wheel_speed = np.abs(sl.wheel['velocity'].values)
wheel_binned, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
)
```

iii. `CONVERSION_NOTES.md` says this follows the reference code's wheel-speed handling.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI collects wheel-speed values from all processed sessions, computes global 33rd and 67th percentile thresholds over the concatenated values, and digitizes each session with those global edges.

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

iii. `CONVERSION_NOTES.md` explicitly says wheel speed is discretized with global tercile thresholds.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. For each trial, the wheel trace is linearly interpolated onto `np.linspace(t_start + binsize, t_end, n_bins)`, where `t_start = stimOn - 0.5` and `t_end = stimOn + 1.5`. Coverage checks require samples near both trial-window edges before interpolation.

ii. 
```python
t_start = align_times[trial_idx] + window[0]
t_end = align_times[trial_idx] + window[1]
```

```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

iii. The notes claim this interpolation grid matches the reference code.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera` or `rightCamera` motion-energy traces and their timestamps. The code prefers the left camera and falls back to the right camera if left is unavailable.

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

iii. `CONVERSION_NOTES.md` explicitly justifies "left first, fallback to right" as matching the reference code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released whisker motion-energy values directly and linearly interpolates them to a 100-point grid for each trial. It does not add extra filtering or normalization in this script.

ii. 
```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
```

iii. The notes describe this as direct use of the camera motion-energy trace with linear interpolation.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As with wheel speed, the AI concatenates whisker motion-energy values across all processed sessions, computes global tercile thresholds, and digitizes each session with those global thresholds.

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

iii. `CONVERSION_NOTES.md` explicitly says whisker motion energy uses global tercile thresholds.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned the same way as wheel speed: the code interpolates it onto `np.linspace(t_start + binsize, t_end, n_bins)` inside the stimulus-centered trial window, after requiring coverage near the window edges.

ii. 
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

iii. The notes justify this as matching the intended behavioral interpolation grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing trial-event values are filtered out. Missing or insufficient wheel/whisker samples cause a trial to fail the behavior mask. Probe-load failures are caught and ignored if at least one probe remains. Sessions are skipped if no probes load, if whisker motion energy cannot be loaded, or if fewer than 10 valid trials survive.

ii. 
```python
for event in nan_exclude:
    mask &= ~trials[event].isnull()
```

```python
if len(bt) < 2:
    good_mask[trial_idx] = False
    continue
if np.any(np.isnan(bv)):
    good_mask[trial_idx] = False
    continue
```

```python
except Exception as e:
    print(f'  Warning: Failed to load probe {pid}: {e}')
...
if whisker_binned is None:
    print(f'  Skipping: no whisker motion energy')
```

iii. The notes justify dropping missing trial-event values, but the rest of the missing-data handling is only evident in the code and runtime logs.

## 10-a. What are the most time-consuming steps of the code?

i. The code structure suggests the most expensive steps are loading spike sorting for each probe, trialwise spike binning across all clusters, and trialwise behavioral interpolation. The trajectory also shows the full conversion spending most of its time inside session loading/conversion rather than final assembly.

ii. 
```python
spikes, clusters, channels = ssl.load_spike_sorting()
```

```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
```

```python
for trial_idx in range(n_trials):
    ...
    binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

iii. The trajectory messages focus on long-running full conversion and repeated probe/session loading, which is the closest thing to an explicit justification.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already uses some vectorized pieces, but it still loops over trials for spike binning, over trials for behavior interpolation, over clusters to build `cluster_to_idx`, and over trials again when assembling the final per-trial lists.

ii. 
```python
for idx, cid in enumerate(cluster_ids):
    cluster_to_idx[cid] = idx
```

```python
for trial_idx in range(n_trials):
    ...
```

```python
for trial in range(n_trials):
    session_neural.append(sm['binned_spikes'][trial].astype(np.float32))
    ...
```

iii. The function docstring for `bin_spikes_in_window` says it uses vectorized numpy operations for speed, so the AI was explicitly trying to optimize this path while still leaving several Python loops in place.

## 10-c. What processing does the code repeat multiple times?

i. The code makes a two-pass behavioral pipeline: first it stores raw `wheel_binned` and `whisker_binned` arrays for every session, then later it concatenates them to compute global thresholds and loops through all sessions again to digitize and assemble outputs. It also sorts spike times again inside `bin_spikes_in_window` even though merged probe spikes were already sorted.

ii. 
```python
all_wheel_speed_raw.append(wheel_binned)
all_whisker_me_raw.append(whisker_binned)
session_meta.append({...})
```

```python
all_wheel_flat = np.concatenate([w.flatten() for w in all_wheel_speed_raw])
all_whisker_flat = np.concatenate([w.flatten() for w in all_whisker_me_raw])
```

```python
sort_idx = np.argsort(spike_times)
sorted_times = spike_times[sort_idx]
sorted_clusters = spike_clusters[sort_idx]
```

iii. No explicit justification for these repeated passes appears in the notes; they are inferred from the code.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code accumulates `all_wheel_speed_raw`, `all_whisker_me_raw`, and large `session_meta` entries solely to compute global discretization thresholds and then rebuild the final data structure, after which those intermediates are discarded. It also defines unused accumulators/imports/helpers (`all_neural`, `all_input`, `all_output`, `all_subject_names`, `all_brain_region_idx`, `all_brain_regions_set`, `discretize_to_bins`, `ismember`, `Path`, `sys`).

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
def discretize_to_bins(values, n_bins=3):
    ...
```

```python
session_meta.append({
    ...
    'binned_spikes': binned_spikes,
    ...
    'wheel_binned': wheel_binned,
    'whisker_binned': whisker_binned,
})
```

iii. There is no explicit justification for these discarded intermediates in the notes or trajectory.
