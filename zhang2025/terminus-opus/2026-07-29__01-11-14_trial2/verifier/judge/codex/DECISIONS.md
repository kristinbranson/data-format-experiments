# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the release index from `code/code_zhang2025/data/bwm_release.csv`, groups rows by `eid`, and treats each row as one probe insertion belonging to a session. For each available session, it locates the local ONE-cache path from `lab/subject/date/session`, then directly reads trial parquet files, spike-sorting `.npy` files, wheel `.npy` files, and whisker motion-energy `.npy` files from disk instead of using `ONE`, `SessionLoader`, or `SpikeSortingLoader`.

ii. 
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)

sessions = {}
for _, row in bwm_df.iterrows():
    eid = row.eid
    if eid not in sessions:
        sessions[eid] = []
    sessions[eid].append({
        'pid': row.pid,
        'probe_name': row.probe_name,
        'subject': row.subject,
        'lab': row.lab,
        'date': row.date,
    })
```

```python
trials_df = pd.read_parquet(trials_file)
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
me_values = np.load(me_file).flatten()
me_times = np.load(times_file).flatten()
```

iii. The notes and trajectory say the agent first tried to use the reference loaders, then switched to direct file reads because `neuropixel` / `SpikeSortingLoader` was unavailable locally. It justified the switch as a transparent way to reproduce the same underlying cache contents while avoiding loader dependencies.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. After session processing, the agent builds a sorted unique subject list from the surviving sessions and stores one subject index per session.

ii.
```python
all_subjects = sorted(list(set(sess['subject'] for sess in session_results)))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx_list.append(subject_to_idx[sess['subject']])
```

iii. The notes describe the dataset as organized in the release table by subject and session, and the agent followed that organization directly rather than inferring subject IDs from filesystem names alone.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from the release CSV. Each `eid` collects all its probes, and only sessions with a discoverable local cache path are considered; later, any session missing required trial, spike, wheel, or whisker data is dropped.

ii.
```python
for _, row in bwm_df.iterrows():
    eid = row.eid
    if eid not in sessions:
        sessions[eid] = []
    sessions[eid].append({...})

available_sessions = {}
for eid, probes in sessions.items():
    sess_path = find_session_path(probes[0]['lab'], probes[0]['subject'], probes[0]['date'])
    if sess_path is not None:
        available_sessions[eid] = probes
```

iii. In `CONVERSION_NOTES.md`, the agent explicitly notes that the release has 459 sessions but only 454 were locally available in cache, so it chose to process only locally available sessions.

## 1-d. How are the data split into trials?

i. Trials come from rows of `_ibl_trials.table.pqt`. The agent first applies the reference-style trial mask, then defines one per-trial interval from `stimOn_times + (-0.5, 1.5)`, and finally removes any trials that fail wheel or whisker coverage checks. Each remaining row becomes one converted trial.

ii.
```python
valid_trials = trials_df[trials_mask].copy()
...
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
...
combined_valid = wheel_valid & me_valid
...
valid_trials_final = valid_trials[combined_valid].copy()
```

iii. The notes say the agent wanted trial structure to match the reference pipeline, then added an extra behavior-validity requirement so neural and behavior arrays stayed aligned trial-by-trial.

## 1-e. How are trials filtered based on quality controls?

i. The code reproduces the reference trial mask: exclude trials with reaction times outside 0.08-2.0 s, trial length above 10 s, missing `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`, and exclude no-choice trials (`choice == 0`). It then further removes trials where wheel or whisker time series do not cover the requested interval closely enough for interpolation.

ii.
```python
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
if MIN_RT is not None:
    mask &= (rt >= MIN_RT)
if MAX_RT is not None:
    mask &= (rt <= MAX_RT)
...
for event in NAN_EXCLUDE:
    if event in trials_df.columns:
        mask &= ~trials_df[event].isna()
...
if EXCLUDE_NOCHOICE:
    mask &= (trials_df['choice'] != 0)
```

```python
if np.abs(t_start - seg_times[0]) > binsize:
    valid_mask[trial_idx] = False
...
if np.abs(t_end - seg_times[-1]) > binsize:
    valid_mask[trial_idx] = False
```

iii. The notes cite `load_trials_and_mask` from the reference code for the first-stage mask. The trajectory shows the agent initially got poor wheel coverage, then changed wheel processing so more trials passed the behavior-alignment checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times and cluster identities in each probe's `spikes.times.npy` and `spikes.clusters.npy`. The cluster-to-region metadata comes from `clusters.channels.npy` together with `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
cluster_channels = np.load(spike_dir / 'clusters.channels.npy').flatten()
chan_brain_ids = np.load(spike_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
cluster_brain_ids = chan_brain_ids[cluster_channels]
```

iii. The notes summarize this as direct loading of spike-sorting outputs from the ONE cache, replacing `SpikeSortingLoader` while keeping the same underlying raw inputs.

## 2-b. How is the `neural` data processed?

i. Probe data from the same session are merged by offsetting cluster IDs and concatenating spikes across probes, then sorting all spikes by time. The merged spike trains are binned into 20 ms counts over 100 bins in a 2 s window around stimulus onset. Brain locations are additionally mapped from Allen IDs to Beryl acronyms for metadata.

ii.
```python
for st, sc, nc, cbi in probe_data:
    all_times.append(st)
    all_clusters.append(sc + cluster_offset)
    all_brain_ids.append(cbi)
    cluster_offset += nc
...
sort_idx = np.argsort(spike_times, kind='stable')
spike_times = spike_times[sort_idx]
spike_clusters = spike_clusters[sort_idx]
```

```python
binned_spikes = bin_spikes_fast(
    spike_times, spike_clusters, interval_starts, interval_ends,
    n_clusters, BINSIZE, N_BINS
)
...
cluster_acronyms_raw = br.id2acronym(cluster_brain_ids)
cluster_acronyms_beryl = br.acronym2acronym(cluster_acronyms_raw, mapping='Beryl')
```

iii. The notes repeatedly say the intended behavior was to match `merge_probes`, `bin_spiking_data`, and Beryl mapping in the reference utilities. The trajectory also mentions a direct spike-count spot check against raw data as validation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is effectively not filtered at the neuron/unit level. The agent keeps every cluster present in the spike-sorting outputs for each loaded probe and does not apply the paper's single-unit QC thresholds.

ii.
```python
def load_spike_data(sess_path, probe_name):
    ...
    return spike_times, spike_clusters, n_clusters, cluster_brain_ids
```

```python
if not probe_data:
    print(f'  Session {eid}: no probe data loaded')
    return None
```

iii. The notes justify this by pointing to the reference code path where `load_spiking_data(..., qc=None)` is used. The trajectory explicitly calls out "No QC filtering" as an intentional choice copied from the code, even though the data paper describes stricter neuron QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial is aligned to stimulus onset, using `stimOn_times` as the anchor. The neural window starts 0.5 s before stimulus onset and ends 1.5 s after stimulus onset.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. Both the notes and the trajectory say the agent copied the `0_data_caching.py` parameter block (`align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`) because the task instructions also said to align based on stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 20 ms bins, giving 100 bins across the 2 s window. There is no second-stage temporal rebinning after spike counts are assigned to these bins.

ii.
```python
BINSIZE = 0.02  # 20ms bins
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
```

```python
bin_idx = np.minimum(np.floor((trial_times - t_start) / binsize).astype(int), n_bins - 1)
np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
```

iii. The notes say 20 ms bins were chosen to match the reference code and the decoder task. The agent also cites the methods excerpt for 20 ms dynamic-behavior binning.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not computed from a trial-specific raw array beyond the fact that trials are aligned to `stimOn_times`. The actual input values are derived from the fixed alignment choice (`stimOn_times`) plus the fixed window and bin size constants.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
...
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes frame this variable as "time since stimOn" and treat it as a deterministic companion to the chosen alignment/binning scheme rather than something read directly from the raw trial table.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code constructs a 100-element vector of bin centers from -0.48 s to 1.5 s relative to stimulus onset, then repeats that same vector for every trial in a session.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

```python
inp = np.stack([
    sess['time_since_stim'],
    np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32),
], axis=0)
```

iii. The code comment says this "matches the bin centers used in reference code." The notes also report a sanity check that the input time range matched expectations.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned exactly to the neural bins: same 2 s window, same 20 ms spacing, and the same 100 bin centers used for spike counts.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
...
session_neural.append(sess['binned_spikes'][trial_idx].astype(np.float32))
...
session_input.append(inp)
```

iii. The agent's justification is implicit in the notes and comments: the time input is meant to be the decoder-side representation of the exact binning used to form each trial's neural array.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived entirely from the `probabilityLeft` trial variable. The code treats a block as a consecutive run of equal `probabilityLeft` values.

ii.
```python
trial_num_in_block = compute_trial_number_in_block(
    valid_trials_final['probabilityLeft']
)
```

```python
if val == current_block:
    count += 1
else:
    current_block = val
    count = 1
trial_nums[i] = count
```

iii. The notes describe this mapping explicitly: "Trial number in block: Count within block." There is no reference-code helper for this variable, so this was the agent's own construction.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. After trial filtering, the agent scans the retained trials in order, increments a counter while `probabilityLeft` stays unchanged, resets the counter when `probabilityLeft` changes, and then broadcasts that scalar across all 100 time bins for each trial.

ii.
```python
for i in range(len(prob_left)):
    val = prob_left.iloc[i] if hasattr(prob_left, 'iloc') else prob_left[i]
    if val == current_block:
        count += 1
    else:
        current_block = val
        count = 1
    trial_nums[i] = count
```

```python
np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32)
```

iii. The notes justify this only at a high level: blocks are identified from `probabilityLeft`, and the resulting count is used as a contextual decoder input. The trajectory does not show any deeper validation of whether counting after filtering preserves the original block index.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from the trial-table `choice` column after trial filtering.

ii.
```python
choice = valid_trials_final['choice'].values.copy()
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
```

iii. The notes map raw `choice` directly to decoder output choice and state the intended coding as left `(-1) -> 0`, right `(1) -> 1`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent excludes `choice == 0` trials earlier in trial filtering, then remaps `-1` to `0` and `1` to `1`. In the final dataset it repeats that categorical label across all 100 time bins of the trial.

ii.
```python
if EXCLUDE_NOCHOICE:
    mask &= (trials_df['choice'] != 0)
...
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
```

```python
np.full(N_BINS, int(sess["choice_binary"][trial_idx]), dtype=np.int64)
```

iii. The notes and README both document the binary remapping explicitly. The trajectory also records a later dtype fix so these outputs would verify cleanly with the decoder checker.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial-table `probabilityLeft` column after filtering.

ii.
```python
prob_left = valid_trials_final['probabilityLeft'].values.copy()
```

iii. The notes state the intended mapping directly from `probabilityLeft` to the categorical prior output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code converts raw probabilities into three integer classes: `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`. Like choice, it then broadcasts that per-trial label across all 100 bins.

ii.
```python
prior_cat = np.zeros(len(prob_left), dtype=np.float32)
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
```

```python
np.full(N_BINS, int(sess["prior_cat"][trial_idx]), dtype=np.int64)
```

iii. The notes and README both give the same three-way categorical mapping. The agent also used the converted distribution as a sanity check during review.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
pos_file = sess_path / '_ibl_wheel.position.npy'
ts_file = sess_path / '_ibl_wheel.timestamps.npy'
...
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The trajectory shows the agent initially used a simpler derivative-based approach, then switched to the reference wheel pipeline after discovering that irregular timestamps caused poor trial coverage.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The code linearly interpolates wheel position onto a uniform 1000 Hz grid, applies an 8th-order 20 Hz low-pass Butterworth filter, differentiates the filtered position to get velocity, and takes the absolute value to obtain speed. For each trial, it then linearly interpolates that continuous speed trace onto the neural bin centers within the stimulus-aligned interval.

ii.
```python
t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
pos_interp = interp1d(timestamps, position, kind='linear')(t_uniform)
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
```

```python
wheel_binned, wheel_valid = interpolate_behavior(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. The notes justify this as matching `SessionLoader.load_wheel()` plus `get_behavior_per_interval`. The trajectory explicitly documents a debugging pass where the agent compared its original implementation against the IBL wheel helpers and rewrote the function to match them.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. After all sessions are processed, the agent pools every wheel-speed bin from every retained trial, computes the 1/3 and 2/3 global quantiles, and uses those thresholds to digitize wheel speed into three categories labeled low, medium, and high.

ii.
```python
all_wheel = np.concatenate(all_wheel)
all_wheel = all_wheel[~np.isnan(all_wheel)]
wheel_quantiles = np.array([
    -np.inf,
    np.quantile(all_wheel, 1/3),
    np.quantile(all_wheel, 2/3),
    np.inf
])
...
wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1]).astype(np.float32)
wheel_disc = np.clip(wheel_disc, 0, 2)
```

iii. The notes justify this as satisfying the task requirement that outputs be categorical, and explicitly call the choice "global quantile-based discretization." This binning rule comes from the agent, not from the reference pipeline.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset-centered trial windows as the neural data. Within each trial window, wheel samples must cover the interval endpoints within one bin, then the signal is linearly interpolated to the same 100 bin centers used by the neural arrays.

ii.
```python
wheel_binned, wheel_valid = interpolate_behavior(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
)
...
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
result[trial_idx] = f_interp(x_interp)
```

iii. The notes say the agent aligned behavior using the reference `get_behavior_per_interval` logic and the same `stimOn_times` window that it used for spikes, because the task instructions asked for stimulus-onset alignment.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`, and falls back to the corresponding right-camera files if the left-camera pair is absent.

ii.
```python
me_file = find_versioned_file(sess_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_versioned_file(sess_path, '_ibl_leftCamera.times.npy')

if me_file is None or times_file is None:
    me_file = find_versioned_file(sess_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_versioned_file(sess_path, '_ibl_rightCamera.times.npy')
```

iii. The notes describe this as matching the reference behavior loader, which tries the left camera first and falls back to the right camera for whisker motion energy.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code loads the precomputed motion-energy trace, truncates the value and timestamp arrays to the same minimum length, removes any NaNs, and then interpolates each valid trial segment to the neural bin centers.

ii.
```python
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]

valid = ~np.isnan(me_values) & ~np.isnan(me_times)
me_values = me_values[valid]
me_times = me_times[valid]
```

```python
me_binned, me_valid = interpolate_behavior(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. The notes justify this as reproducing the reference interpolation logic while being robust to small file inconsistencies such as length mismatch or NaNs.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded exactly like wheel speed: all retained whisker motion-energy bins are pooled globally, the 1/3 and 2/3 quantiles are computed, and each value is digitized into low, medium, or high.

ii.
```python
all_me = np.concatenate(all_me)
all_me = all_me[~np.isnan(all_me)]
me_quantiles = np.array([
    -np.inf,
    np.quantile(all_me, 1/3),
    np.quantile(all_me, 2/3),
    np.inf
])
...
me_disc = np.digitize(sess['me_binned'], me_quantiles[1:-1]).astype(np.float32)
me_disc = np.clip(me_disc, 0, 2)
```

iii. The notes explicitly call this "global quantile-based discretization." As with wheel speed, that rule was chosen by the agent to satisfy the categorical-output requirement.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned with exactly the same per-trial `stimOn_times` windows as spikes. A trial is only kept if the whisker trace covers the window closely enough at both ends, and successful trials are linearly interpolated to the same 100 bin centers as the neural data.

ii.
```python
me_binned, me_valid = interpolate_behavior(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
combined_valid = wheel_valid & me_valid
```

```python
if np.abs(t_start - seg_times[0]) > binsize:
    valid_mask[trial_idx] = False
...
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. The notes and trajectory justify this the same way as wheel speed: the agent followed the reference interpolation helper and the task's stimulus-onset alignment instruction.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses version-aware file lookup, optional loading for auxiliary timing files, right-camera fallback for whisker motion energy, array truncation to a common length, NaN removal for whisker traces, and trial-level rejection when behavior coverage is inadequate. At a larger scale, it skips entire sessions if core modalities are missing (`trials`, spike data, wheel, whisker) or if fewer than two valid trials remain.

ii.
```python
def find_versioned_file(base_dir, filename):
    direct = base_dir / filename
    if direct.exists():
        return direct
    versioned_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('#')], reverse=True)
    for vd in versioned_dirs:
        f = vd / filename
        if f.exists():
            return f
```

```python
if me_file is None or times_file is None:
    me_file = find_versioned_file(sess_path, 'rightCamera.ROIMotionEnergy.npy')
...
min_len = min(len(me_values), len(me_times))
...
valid = ~np.isnan(me_values) & ~np.isnan(me_times)
```

```python
if wheel_times is None:
    print(f'  Session {eid}: no wheel data')
    return None
...
if np.sum(combined_valid) < 2:
    print(f'  Session {eid}: too few valid trials after behavior filtering ({np.sum(combined_valid)})')
    return None
```

iii. The notes explicitly list missing wheel data, missing whisker data, and too-few-valid-trial sessions as handled edge cases. The trajectory also shows the agent debugging wheel coverage rather than silently accepting large data loss.

## 12-a. What are the most time-consuming steps of the code?

i. The main runtime cost comes from per-session spike binning, wheel interpolation/filtering at 1000 Hz, and per-trial behavior interpolation for wheel and whisker signals. The full conversion logs in the notes also show dataset-wide global discretization and pickle writing as additional heavy steps.

ii.
```python
binned_spikes = bin_spikes_fast(
    spike_times, spike_clusters, interval_starts, interval_ends,
    n_clusters, BINSIZE, N_BINS
)
...
wheel_times, wheel_speed = compute_wheel_speed(sess_path)
...
wheel_binned, wheel_valid = interpolate_behavior(...)
me_binned, me_valid = interpolate_behavior(...)
```

iii. The notes report per-session timings separately for spike binning, wheel processing, and whisker motion energy, and the trajectory comments on slow sessions with long wheel traces as the dominant runtime spikes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several central loops remain scalar or trial-by-trial: the loop over trials in `bin_spikes_fast`, the loop over trials in `interpolate_behavior`, the sequential scan in `compute_trial_number_in_block`, and the loops that rebuild lists of per-trial `neural`, `input`, and `output` arrays. These could have been partially vectorized or delegated to the reference utilities.

ii.
```python
for trial_idx in range(n_trials):
    t_start = interval_starts[trial_idx]
    t_end = interval_ends[trial_idx]
    ...
    np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
```

```python
for trial_idx in range(n_trials):
    ...
    result[trial_idx] = f_interp(x_interp)
```

```python
for trial_idx in range(n_trials):
    session_neural.append(sess['binned_spikes'][trial_idx].astype(np.float32))
...
for trial_idx in range(n_trials):
    session_output.append(out)
```

iii. The notes mention runtime estimates and some optimization thinking, but the final implementation still favors straightforward Python loops over more vectorized or parallel versions.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly searches time intervals trial-by-trial for spikes, wheel, and whisker traces; repeatedly allocates full-length constant vectors for static per-trial variables; and repeats some merge logic both in the unused `merge_probes` helper and inline inside `process_session`. It also computes global discretization thresholds in one pass, then digitizes each session again in a second pass.

ii.
```python
idx_start = np.searchsorted(spike_times, t_start, side='left')
idx_end = np.searchsorted(spike_times, t_end, side='left')
```

```python
inp = np.stack([
    sess['time_since_stim'],
    np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32),
], axis=0)
```

```python
all_wheel.append(sess['wheel_binned'].flatten())
...
wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1]).astype(np.float32)
```

iii. The trajectory reflects some awareness of runtime, but there is no evidence that the agent revisited these repeated patterns after the pipeline passed verification.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script includes optional plotting and stores extra `show_processing` fields (`trials_df`, `stim_on`) that are not part of the final dataset. It also computes and retains continuous wheel and whisker traces only to discard them after quantile digitization, and it defines unused helpers/imports such as `merge_probes`, `bin_spikes_vectorized`, `discretize_to_bins`, and `ProcessPoolExecutor`.

ii.
```python
if show_processing:
    result['trials_df'] = valid_trials_final
    result['stim_on'] = stim_on[combined_valid]
```

```python
wheel_binned = wheel_binned[combined_valid]
me_binned = me_binned[combined_valid]
...
wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1]).astype(np.float32)
me_disc = np.digitize(sess['me_binned'], me_quantiles[1:-1]).astype(np.float32)
```

iii. The notes say processing plots were added for verification, not for the final decoder dataset. The trajectory also shows that some helper code was written during iteration and then left unused once the pipeline stabilized.
