# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the Zhang BWM release CSV, constructs local ONE-cache paths from lab/subject/date, groups probe names by `eid`, and directly discovers trials, spikes, wheel, and camera files with `rglob`. It does not use ONE loaders or restrict the release using `DATALIMIT_SUBSET.csv`.

ii. ```python
bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
session_path = Path(cache_dir) / lab / 'Subjects' / subject / date / '001'
trials_files = list(session_path.rglob('_ibl_trials.table.pqt'))
```

iii. The trajectory says the local ONE registry exposed only some datasets, so the agent deliberately switched to direct filesystem loading. It believed the release CSV represented all applicable sessions.

## 1-b. How are the data split into subjects?

i. The `subject` column in the release CSV supplies the mouse ID. An insertion-ordered unique subject list is built while sessions are processed, and each session receives the corresponding index.

ii. ```python
subject = row['subject']
if subject not in subject_set:
    subject_set.append(subject)
subj_idx = subject_set.index(subject)
```

iii. The agent treated the release metadata as the authoritative subject identifier; the trajectory reports 135 subjects in the converted result.

## 1-c. How are the data split into sessions?

i. Rows in the release CSV are grouped by `eid`; repeated rows add probes to the same session. Each grouped session is processed once.

ii. ```python
if eid not in sessions:
    sessions[eid] = {'path': session_path, 'probes': [], 'subject': subject, ...}
sessions[eid]['probes'].append(probe_name)
```

iii. The trajectory identified `eid` as the unique session identifier and found 459 release sessions.

## 1-d. How are the data split into trials?

i. The trials parquet already has one row per trial. After masking, the code iterates over rows and creates one neural, input, and output array per surviving row.

ii. ```python
valid_trials = trials[mask].copy()
for trial_idx, (df_idx, trial) in enumerate(valid_trials.iterrows()):
```

iii. The agent relied on the IBL trials table's row-per-trial organization.

## 1-e. How are trials filtered based on quality controls?

i. Trials are removed for missing values in six named columns, reaction times outside 0.08–2 s, or `choice == 0`. Trials are subsequently dropped when interpolated wheel or whisker data lack coverage or contain NaNs; sessions with fewer than two retained trials are dropped.

ii. ```python
for event in NAN_EXCLUDE:
    mask &= ~trials[event].isna()
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trials['choice'] != 0)
if ws is None or wm is None or np.any(np.isnan(ws)) or np.any(np.isnan(wm)):
    continue
```

iii. The trajectory says these exclusions match `load_trials_and_mask`: required events present, valid RT, and no no-choice trials. Behavioral coverage was added so every saved output is complete.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices derive from `spikes.times.npy` and `spikes.clusters.npy`. Cluster-channel and channel-atlas arrays determine region labels; cluster metrics are loaded but not used to filter the activity.

ii. ```python
spike_times = np.load(revision_path / 'spikes.times.npy')
spike_clusters = np.load(revision_path / 'spikes.clusters.npy')
clusters_channels = np.load(revision_path / 'clusters.channels.npy')
chan_brain_ids = np.load(revision_path / 'channels.brainLocationIds_ccf_2017.npy')
```

iii. The agent identified spike time and assigned cluster as the necessary activity variables, with atlas arrays needed only for metadata.

## 2-b. How is the `neural` data processed?

i. Probes are merged with cluster/channel offsets, spikes are time-sorted, cluster IDs are remapped contiguously, and spikes are counted per cluster in 100 20-ms bins. Counts are not divided by bin width, so the saved values are spike counts rather than Hz.

ii. ```python
merged_clusters.append(clusters + cluster_offset)
bin_idx = np.floor((times_sel - t_start) / binsize).astype(int)
np.add.at(binned, (clusters_sel[valid], bin_idx[valid]), 1)
```

iii. The trajectory says merging probes and 20-ms binning follow the reference caching code. It does not discuss the omitted conversion from counts to firing rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered. Although metric labels are read, every cluster appearing in the spikes is retained, including root/void locations.

ii. ```python
cluster_labels = metrics['label'].values
# later: no use of all_labels
cluster_ids = np.unique(spike_clusters)
```

iii. The agent explicitly reasoned that reference caching calls `load_spiking_data(..., qc=None)`, so all clusters should be retained, despite noting the data paper's well-isolated-neuron criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial uses absolute bounds `stimOn_times - 0.5` through `stimOn_times + 1.5`; sorted spikes in that interval are assigned bins relative to the lower bound.

ii. ```python
stim_on = trial[ALIGN_TIME]
t_start = stim_on + TIME_WINDOW[0]
t_end = stim_on + TIME_WINDOW[1]
i_start = np.searchsorted(spike_times, t_start, side='left')
```

iii. The trajectory repeatedly identifies stimulus onset and the −0.5 to +1.5 s window as the reference alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, with 100 non-overlapping bins over two seconds. Spikes are binned once; there is no later rebinning or smoothing.

ii. ```python
BINSIZE = 0.02
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The agent chose the 20-ms setting from `0_data_caching.py`, preferring it over a conflicting 50-ms prose description.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is based on the configured window relative to each trial's `stimOn_times`, though the stored vector is the same generated grid for every trial.

ii. ```python
stim_on = trial[ALIGN_TIME]
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The agent viewed `stimOn_times` as the required common alignment event.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A 100-value float32 linear grid from −0.48 to +1.50 s is generated. These are right bin edges, not bin centers.

ii. ```python
time_since_stim = np.linspace(-0.5 + 0.02, 1.5, 100).astype(np.float32)
```

iii. The code comment claims this matches reference bin centers; the trajectory does not justify choosing right edges rather than centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It has 100 entries and is stacked with each 100-bin neural trial, but its values denote right edges while neural counts represent intervals beginning at −0.5 s.

ii. ```python
input_data = np.stack([time_since_stim, np.full(N_BINS, trial_num)], axis=0)
```

iii. The agent considered equal length and use of the same configured window sufficient alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` after the trial quality mask has been applied.

ii. ```python
trials_masked = trials_df[mask].copy()
prob_left = trials_masked['probabilityLeft'].values
```

iii. The agent used block-prior changes to identify blocks, consistent with the task structure.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Changes in the filtered prior sequence start blocks, and trials are numbered from zero within each block. The scalar is broadcast across all time bins. Filtering first compresses numbering when an original trial was removed.

ii. ```python
block_changes = np.concatenate([[0], np.where(np.diff(prob_left) != 0)[0] + 1])
trial_in_block[start:end] = np.arange(end - start)
np.full(N_BINS, trial_num, dtype=np.float32)
```

iii. The trajectory does not discuss the consequence of computing this after filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from the trials table's `choice` column.

ii. ```python
choice_val = 0 if trial['choice'] == 1 else 1
```

iii. The agent recognized the IBL convention `+1=left`, `-1=right` and the requested output convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL `+1` is mapped to 0 and `-1` to 1, then the per-trial value is broadcast over 100 time bins.

ii. ```python
np.full(N_BINS, choice_val, dtype=np.int64)
```

iii. The trajectory says integer broadcasting was necessary for decoder validation and was corrected after an initial float implementation.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `probabilityLeft` in the trials table.

ii. ```python
prob_left = trial['probabilityLeft']
```

iii. The agent followed the explicitly requested prior variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 map to categories 0, 1, and 2; an unexpected value falls back to 1. The result is broadcast over time.

ii. ```python
if prob_left == 0.2: prior_val = 0
elif prob_left == 0.5: prior_val = 1
elif prob_left == 0.8: prior_val = 2
else: prior_val = 1
```

iii. The three-way mapping is specified directly by the task. The fallback was not separately justified.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. ```python
wheel_pos = np.load(wheel_pos_files[0])
wheel_ts = np.load(wheel_ts_files[0])
```

iii. The agent identified these as the locally available raw wheel streams.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is first-differenced and divided by timestamp differences; absolute velocity is assigned midpoint timestamps. It is linearly interpolated to the trial grid and discretized by session-wide terciles. It does not perform the reference 1-kHz position interpolation and filtered differentiation.

ii. ```python
vel = np.diff(wheel_pos) / np.diff(wheel_ts)
speed = np.abs(vel)
vel_ts = wheel_ts[:-1] + dt / 2
```

iii. The trajectory calls this a simple finite difference while claiming it matches brainbox; no justification is given for omitting the reference filter.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All retained interpolated values in a session are pooled; the 33.3rd and 66.7th percentiles form three approximately equal-count categories.

ii. ```python
edges = np.percentile(all_vals, np.linspace(0, 100, n_bins + 1))
d = np.digitize(v, edges[1:-1])
```

iii. The agent chose per-session quantiles to preserve local context and produce balanced categories.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated at `np.linspace(t_start + 0.02, t_end, 100)`, yielding the same length as neural data but sampling right edges rather than neural bin centers.

ii. ```python
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
return f(bin_centers)
```

iii. The agent believed these were the reference bin centers and used shared absolute session timestamps.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `leftCamera.ROIMotionEnergy.npy` and left-camera timestamps when available, otherwise the corresponding right-camera files.

ii. ```python
left_me_files = list(session_path.rglob('leftCamera.ROIMotionEnergy.npy'))
right_me_files = list(session_path.rglob('rightCamera.ROIMotionEnergy.npy'))
```

iii. The agent preferred the 60-Hz left camera and added right-camera fallback for session coverage.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is used without filtering, linearly interpolated to 100 trial points, and discretized using session-wide terciles.

ii. ```python
wm = interpolate_behavior_to_bins(me_ts, me_values, t_start, t_end, BINSIZE, N_BINS)
whisker_disc, whisker_edges = discretize_to_bins(whisker_me_raw, 3)
```

iii. The trajectory describes the released trace as already processed and uses quantiles for three categorical outputs.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. All retained values in each session are pooled and split at their 33.3rd and 66.7th percentiles.

ii. ```python
edges = np.percentile(all_vals, quantiles)
np.digitize(v, edges[1:-1])
```

iii. The same per-session equal-count rationale as wheel speed was used.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera values are interpolated using common session timestamps at points from `t_start + 0.02` to `t_end`. This matches neural length but is shifted 10 ms from neural bin centers.

ii. ```python
bin_centers = np.linspace(t_start + binsize, t_end, n_bins)
return f(bin_centers)
```

iii. The agent believed the generated points were matching bin centers.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing trials tables, spike data, wheel data, or insufficient retained trials cause a session to be skipped. Missing individual probes are skipped. Missing/short behavioral coverage and NaNs cause trials to be skipped. Camera choice falls back from left to right.

ii. ```python
if spike_times is None:
    continue
if not spikes_list or wheel_speed is None:
    return None
if ws is None or wm is None:
    continue
```

iii. The trajectory reports 20 sessions skipped for missing behavioral data and treats dropping incomplete records as preferable to fabricating values.

## 10-a. What are the most time-consuming steps of the code?

i. Loading and processing large spike arrays, per-trial spike binning, and retaining the full converted dataset dominate runtime and memory. The full run took long enough to be launched in the background and produced a 38.6-GB pickle.

ii. ```python
spike_times = np.load(...)
for trial_idx, ... in enumerate(valid_trials.iterrows()):
    neural = bin_spikes_trial(...)
```

iii. The trajectory specifically noticed slow spike binning, optimized its inner fill, and monitored a long, memory-growing full conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Session and trial loops are structurally natural, but spike-cluster remapping, region lookup, behavioral discretization, output assembly, and potentially multi-trial binning could be vectorized. The inner spike fill already uses `np.add.at`.

ii. ```python
spike_clusters_remapped = np.array([cluster_id_to_idx[c] for c in spike_clusters])
for cid in cluster_ids:
for trial_idx, ... in enumerate(valid_trials.iterrows()):
```

iii. The trajectory says it optimized a prior per-spike loop after observing poor speed; it did not discuss the remaining Python loops.

## 10-c. What processing does the code repeat multiple times?

i. It rebuilds the identical time vector and trial-number broadcast inside every trial, performs two separate behavioral interpolation calls with identical control flow, repeatedly performs linear subject membership/index searches, and later loops again to assemble outputs and global region indices.

ii. ```python
time_since_stim = np.linspace(...).astype(np.float32)
ws = interpolate_behavior_to_bins(...)
wm = interpolate_behavior_to_bins(...)
subj_idx = subject_set.index(subject)
```

iii. No explicit justification for these repetitions appears in the trajectory; they favor straightforward code.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `clusters.depths.npy` and cluster quality labels but never uses them, creates unused `valid_indices` and `good_trial_indices`, retains discretization edges only in temporary session results, and stores duplicate region arrays during assembly.

ii. ```python
clusters_depths = np.load(revision_path / 'clusters.depths.npy')
valid_indices = valid_trials.index.tolist()
good_trial_indices.append(trial_idx)
all_brain_region_idx.append(result['beryl_regions'])
```

iii. The trajectory does not justify these discarded values; several are remnants of exploratory or planned QC logic.
