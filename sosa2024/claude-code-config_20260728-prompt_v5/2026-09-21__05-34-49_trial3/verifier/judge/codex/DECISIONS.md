# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for `sub-*` directories, then for `.nwb` files inside each subject directory. Each file is treated as one session and is loaded with `h5py`, not `pynwb`. Within each session it loads imaging arrays, behavioral time series, reward timestamps, and metadata, then later splits them into trials.

ii.
```python
def collect_nwb_files(data_dir, sample=False):
    files = []
    for subj_dir in sorted(os.listdir(data_dir)):
        if not subj_dir.startswith('sub-'):
            continue
        subj_path = os.path.join(data_dir, subj_dir)
        for nwb_file in sorted(os.listdir(subj_path)):
            if nwb_file.endswith('.nwb'):
                files.append(os.path.join(subj_path, nwb_file))
    return files

def load_nwb_session(filepath):
    data = {}
    with h5py.File(filepath, 'r') as f:
        ophys = f['processing']['ophys']
        bts = f['processing']['behavior']['BehavioralTimeSeries']
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by noting the dataset layout is one subject directory per mouse and 152 NWB files total. The trajectory shows it deliberately chose direct NWB/HDF5 access after surveying the file structure and contents.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the NWB metadata field `general/subject/subject_id`, with one subject index assigned per processed session when building the final dataset.

ii.
```python
data['subject_id'] = f['general']['subject']['subject_id'][()].decode()

subjects = sorted(set(r['subject_id'] for r in results))
subject_to_idx = {s: i for i, s in enumerate(subjects)}

for r in results:
    subject_idx.append(subject_to_idx[r['subject_id']])
```

iii. The notes say the data contain 11 switch-task mice and that subject IDs in the NWB files agree with the directory naming. Using NWB metadata appears to have been the AI’s preferred source of truth.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The final output keeps one top-level session entry per processed file.

ii.
```python
for nwb_file in sorted(os.listdir(subj_path)):
    if nwb_file.endswith('.nwb'):
        files.append(os.path.join(subj_path, nwb_file))

for i, filepath in enumerate(nwb_files):
    result = process_session(filepath, show_processing=args.show_processing,
                            session_label=label)
```

iii. The notes describe the dataset as 152 NWB files across 11 mice and repeatedly refer to one recording day per NWB file. The AI treated that file/session mapping as straightforward and consistent with the dataset organization.

## 1-d. How are the data split into trials?

i. Trials are defined from `trial_start` to the first subsequent `teleport` sample. The start index is inclusive and the teleport frame is excluded.

ii.
```python
def get_trial_boundaries(trial_start_signal, teleport_signal, trial_number_signal):
    starts = np.where(trial_start_signal == 1)[0]
    teleports = np.where(teleport_signal == 1)[0]

    trial_starts = []
    trial_ends = []

    for s in starts:
        next_teleports = teleports[teleports > s]
        if len(next_teleports) > 0:
            trial_starts.append(s)
            trial_ends.append(next_teleports[0])
```

iii. In the notes, the AI explicitly recorded the decision “trial_start==1 to teleport==1.” In the trajectory it says this matched the trial structure it observed in the NWB behavior streams and the paper’s lap-based task.

## 1-e. How are trials filtered based on quality controls?

i. The AI removes trials flagged as lick-sensor errors, skips trials that become shorter than 2 bins after dual-plane downsampling, skips trials whose neural data are all NaN, and skips entire sessions with fewer than 2 usable trials. It does not implement the reference solution’s `<50`-timepoint trial filter.

ii.
```python
lick_error_trials = detect_lick_errors(raw['lick'], trial_starts, trial_ends)

for i in range(n_trials):
    if i in lick_error_trials:
        continue
    ...
    if np.all(np.isnan(trial_neural)):
        continue
    ...
    if downsample > 1:
        T_new = T // downsample
        if T_new < 2:
            continue
```

iii. `CONVERSION_NOTES.md` says the AI chose the paper’s lick-error criterion because it reproduced the paper’s reported 81 removed trials. The notes also emphasize the decoder requirement that each session retain at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the raw `Fluorescence` and `Neuropil` NWB groups, after filtering ROIs by `iscell`. It does not use the NWB `Deconvolved` signal.

ii.
```python
seg = ophys['ImageSegmentation']['PlaneSegmentation']
iscell = seg['iscell'][:]
plane_idx = seg['planeIdx'][:]
...
fl = fluor_grp[pk]['data'][:].T
neu = ophys['Neuropil'][pk]['data'][:].T
...
fluor_list.append(fl[plane_iscell])
neuropil_list.append(neu[plane_iscell])
```

iii. The notes explicitly say “Compute dF/F from raw Fluorescence + Neuropil” and describe that as matching the reference preprocessing better than using Suite2P’s stored deconvolution.

## 2-b. How is the `neural` data processed?

i. The AI computes dF/F by keeping only in-trial samples, subtracting `0.7 * neuropil`, adding back each trial’s mean neuropil, computing a maximin baseline with Gaussian smoothing plus min/max filters, and smoothing the resulting dF/F with sigma 2. It does not deconvolve the dF/F into events and does not implement the reference `keep_teleports` logic.

ii.
```python
f_[:, start:end] = f_raw[:, start:end]
f_neu_[:, start:end] = f_neu[:, start:end]
f_ = f_ - NEUROPIL_COEF * f_neu_
...
neuropil_mean = np.nanmean(f_neu_[:, start:end], axis=1, keepdims=True)
trial_data = trial_data + NEUROPIL_COEF * neuropil_mean
smoothed = nansmooth(trial_data, BASELINE_SMOOTH_SIGMA, axis=1)
bl = ndi.minimum_filter1d(smoothed, BASELINE_MINMAX_WINDOW, axis=-1)
bl = ndi.maximum_filter1d(bl, BASELINE_MINMAX_WINDOW, axis=-1)
dff[:, nanmask] = (f_[:, nanmask] - baseline[:, nanmask]) / np.abs(baseline[:, nanmask])
dff[:, start:end] = nansmooth(dff[:, start:end], DFF_SMOOTH_SIGMA, axis=1)
```

iii. The notes and trajectory cite the paper’s dF/F recipe: neuropil subtraction, maximin baseline over about 20 s, then smoothing. The AI explicitly justified not using deconvolution by saying dF/F would be “more informative” for the decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are first restricted to those marked `iscell`, then putative interneurons are removed if their dF/F correlates with speed above `r > 0.5`.

ii.
```python
plane_iscell = iscell[plane_mask, 0].astype(bool)
fluor_list.append(fl[plane_iscell])
neuropil_list.append(neu[plane_iscell])
...
is_interneuron = detect_interneurons(dff, speed_for_corr, nanmask)
keep_mask = ~is_interneuron
dff = dff[keep_mask]
```

iii. The notes say this follows the paper’s manual ROI curation plus interneuron exclusion criterion. The trajectory also shows the AI explicitly chose `r > 0.5` because it matched the paper rather than the code default.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start of trial simply by cutting each trial from `trial_start` to `teleport`; no additional temporal shift is applied.

ii.
```python
trial_starts, trial_ends = get_trial_boundaries(
    raw['trial_start'], raw['teleport'], raw['trial_number'])
...
trial_neural = dff[:, start:end]
```

iii. The notes say the temporal alignment event is “Start of trial (entry onto linear track)” and treat the trial split itself as the alignment operation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI aims for a canonical 15.5078125 Hz sampling rate (`~64.48 ms` bins). Single-plane sessions are kept as-is; dual-plane sessions are downsampled by 2 so all sessions have the same effective rate.

ii.
```python
TARGET_RATE = 15.5078125
...
downsample = 2 if raw['is_dual_plane'] else 1
effective_rate = raw['rate'] / downsample
...
trial_neural = trial_neural[:, :T_new * downsample].reshape(n_cells, T_new, downsample).mean(axis=2)
...
'time_bin_size': 1000.0 / TARGET_RATE,
```

iii. The notes justify this as making all sessions share one time bin size and specifically call out m17/m18 as dual-plane recordings that should be downsampled to match the others.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The AI derives this input from trial length in frames plus the assumed sampling rate, not from the stored behavior timestamps.

ii.
```python
downsample = 2 if raw['is_dual_plane'] else 1
effective_rate = raw['rate'] / downsample
...
time_from_start = np.arange(T, dtype=np.float32) / effective_rate
```

iii. The notes say the intended time bin is globally fixed at about 64.48 ms and frame counts can therefore be converted directly to seconds. The trajectory reflects the same rate-based reasoning.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI constructs a synthetic evenly spaced time vector starting at zero and incrementing by `1/effective_rate`.

ii.
```python
T = trial_neural.shape[1]
time_from_start = np.arange(T, dtype=np.float32) / effective_rate
inputs = np.stack([time_from_start, env, trial_num, prev_outcome], axis=0)
```

iii. The notes justify this as a direct consequence of using a fixed imaging rate and, for dual-plane sessions, the post-downsampling effective rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the time vector is created with the same per-trial length `T` as the neural trial matrix, after any downsampling.

ii.
```python
trial_neural = dff[:, start:end]
...
T = trial_neural.shape[1]
time_from_start = np.arange(T, dtype=np.float32) / effective_rate
```

iii. The AI’s justification is implicit in the code and explicit in the notes’ fixed-rate design: after any rebinning, all trial-wise inputs and outputs are rebuilt at the same length as `trial_neural`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the raw behavior `environment` time series.

ii.
```python
data['environment'] = bts['environment']['data'][:].astype(np.float64)
...
env_vals = raw['environment'][start:end]
```

iii. The notes map `environment` directly to decoder input `environment` and describe it as the paper’s binary ENV1/ENV2 variable.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the AI takes the first nonnegative environment value within the trial, converts it to an integer, and broadcasts it across all trial time bins.

ii.
```python
environments = []
for start, end in zip(trial_starts, trial_ends):
    env_vals = raw['environment'][start:end]
    active_env = env_vals[env_vals >= 0]
    if len(active_env) > 0:
        environments.append(int(round(active_env[0])))
    else:
        environments.append(0)
...
env = np.full(T, environments[i], dtype=np.float32)
```

iii. In the notes, the AI says environment is a per-trial contextual variable and effectively constant on each trial, so broadcasting one trial label is sufficient.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index over the extracted trials, not from the raw NWB `trial number` stream.

ii.
```python
for i in range(n_trials):
    ...
    trial_num = np.full(T, i, dtype=np.float32)
```

iii. The notes say “Trial number: Integer trial index within session,” and the trajectory shows the AI preferred its own trial extraction over trusting the raw `trial number` signal.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing is applied beyond taking the sequential within-session trial index and broadcasting it across time.

ii.
```python
trial_num = np.full(T, i, dtype=np.float32)
inputs = np.stack([time_from_start, env, trial_num, prev_outcome], axis=0)
```

iii. The notes justify this as the simplest session-local trial counter for decoder input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward event timestamps, behavior timestamps, and the raw `trial_number` signal. The code first determines a per-trial rewarded/not-rewarded vector, then uses the previous trial’s entry.

ii.
```python
data['reward_timestamps'] = reward_ts_data['timestamps'][:]
data['behavior_timestamps'] = bts['position']['timestamps'][:]
data['trial_number'] = bts['trial number']['data'][:].astype(np.float64)
...
rewarded = determine_reward_per_trial(
    raw['reward_timestamps'], raw['behavior_timestamps'],
    raw['trial_number'], trial_starts, trial_ends)
...
prev_outcome = np.full(T, rewarded[i - 1] if i > 0 else 0, dtype=np.float32)
```

iii. The trajectory shows the AI recognized reward was stored as timestamped events, not a regularly sampled series. The notes justify the per-trial reward vector as the basis for both `previous_trial_outcome` and `reward_outcome`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each reward timestamp is matched to the nearest behavior frame using `argmin`; the reward is then assigned to the trial index stored in the raw `trial_number` array. For trial `i`, the previous outcome is `rewarded[i-1]`, with trial 0 forced to 0.

ii.
```python
def determine_reward_per_trial(reward_timestamps, behavior_timestamps, trial_number_signal,
                                trial_starts, trial_ends):
    rewarded = np.zeros(n_trials, dtype=int)
    for rt in reward_timestamps:
        frame_idx = np.argmin(np.abs(behavior_timestamps - rt))
        trial_num = int(trial_number_signal[frame_idx])
        if trial_num >= 0 and trial_num < n_trials:
            rewarded[trial_num] = 1
...
prev_outcome = np.full(T, rewarded[i - 1] if i > 0 else 0, dtype=np.float32)
```

iii. The notes say the first trial has no previous outcome, so it should be 0. The trajectory indicates the AI viewed nearest-frame reward alignment as adequate for this binary per-trial variable.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the raw `position` time series plus an inferred per-trial reward-zone label obtained from the raw `reward_zone` signal and position values.

ii.
```python
zone_labels = determine_reward_zone(raw['position'], raw['reward_zone'],
                                     trial_starts, trial_ends)
...
trial_pos = raw['position'][start:end]
zone_label = zone_labels[i]
zone_start, zone_end = REWARD_ZONES[zone_label] if zone_label else (80, 130)
distance = compute_distance_to_reward_zone(trial_pos, zone_start, zone_end)
```

iii. The notes say the NWB `reward_zone` values are noisy state codes where `>0` indicates being in or near the reward zone. The AI justified inferring zone identity from positions where `reward_zone > 0`.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI first infers a zone label per trial by taking the mean position over samples where `reward_zone > 0`, assigning the closest canonical zone center, and filling missing trials from nearby trials. It then computes signed distance to the nearest edge of that zone: negative before, zero inside, positive after.

ii.
```python
mask = rz > 0
if mask.sum() > 0:
    mean_pos = pos[mask].mean()
    for label, (zs, ze) in REWARD_ZONES.items():
        center = (zs + ze) / 2
        dist = abs(mean_pos - center)
...
def compute_distance_to_reward_zone(position, zone_start, zone_end):
    distance = np.zeros_like(position)
    before = position < zone_start
    inside = (position >= zone_start) & (position <= zone_end)
    after = position > zone_end
    distance[before] = position[before] - zone_start
    distance[inside] = 0
    distance[after] = position[after] - zone_end
```

iii. The notes justify this by saying actual reward-zone positions cluster around the paper’s A/B/C locations and nearest-zone assignment produced roughly equal class fractions. The trajectory shows the AI chose a simple positional heuristic after inspecting reward-zone positions.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is discretized into 7 bins corresponding to the decoder specification.

ii.
```python
def discretize_distance(distance):
    bins = np.zeros(len(distance), dtype=int)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
```

iii. The notes explicitly list the required bins and say this output should match the decoder task definition.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial slice as the neural data and, for dual-plane sessions, is downsampled to the same rebinned length before distance binning.

ii.
```python
trial_neural = dff[:, start:end]
trial_pos = raw['position'][start:end]
...
if downsample > 1:
    trial_pos = trial_pos[:T_new * downsample].reshape(T_new, downsample).mean(axis=1)
...
distance = compute_distance_to_reward_zone(trial_pos, zone_start, zone_end)
```

iii. The notes’ global design choice is that all trialwise signals should share the same time base as `trial_neural`, including after the dual-plane downsampling step.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the raw `position` behavior time series.

ii.
```python
data['position'] = bts['position']['data'][:].astype(np.float64)
...
trial_pos = raw['position'][start:end]
```

iii. The notes directly map `position` to decoder output `absolute_position`.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices the per-trial position signal and, for dual-plane sessions, downsamples it by averaging pairs of samples before discretization.

ii.
```python
trial_pos = raw['position'][start:end]
...
if downsample > 1:
    trial_pos = trial_pos[:T_new * downsample].reshape(T_new, downsample).mean(axis=1)
...
pos_bins = discretize_position(trial_pos)
```

iii. The notes justify the averaging step as part of forcing all sessions onto a common ~15.5 Hz time base.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is binned into five 90 cm bins spanning the 450 cm track.

ii.
```python
def discretize_position(position):
    bins = np.zeros(len(position), dtype=int)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
```

iii. The notes cite the 450 cm track length and the decoder task’s “5 equal-sized bins” requirement.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned by taking the same trial slice as the neural data and applying the same dual-plane downsampling where relevant.

ii.
```python
trial_neural = dff[:, start:end]
trial_pos = raw['position'][start:end]
...
if downsample > 1:
    trial_pos = trial_pos[:T_new * downsample].reshape(T_new, downsample).mean(axis=1)
```

iii. The notes emphasize that all decoder variables should be rebuilt to the same per-trial time base as the neural matrix.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw `lick` behavior time series.

ii.
```python
data['lick'] = bts['lick']['data'][:].astype(np.float64)
...
trial_lick = raw['lick'][start:end]
```

iii. The notes map the raw lick signal directly to the binary lick decoder output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick is binarized as `lick > 0`. For dual-plane sessions, the raw lick signal is first downsampled with max-pooling over pairs of frames.

ii.
```python
if downsample > 1:
    trial_lick = trial_lick[:T_new * downsample].reshape(T_new, downsample).max(axis=1)
...
lick_binary = (trial_lick > 0).astype(int)
```

iii. The notes justify binarization from the decoder specification and justify max-pooling as preserving any lick event within a rebinned time window.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is taken from the same trial slice as the neural data and, when dual-plane sessions are rebinned, is reduced to the same number of time bins.

ii.
```python
trial_neural = dff[:, start:end]
trial_lick = raw['lick'][start:end]
...
if downsample > 1:
    trial_lick = trial_lick[:T_new * downsample].reshape(T_new, downsample).max(axis=1)
```

iii. The notes’ general alignment rule is that all trialwise variables must share the same time axis as `trial_neural`.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the raw `reward_zone` behavior signal plus the raw `position` signal, through the same per-trial zone-label inference used for distance-to-zone.

ii.
```python
zone_labels = determine_reward_zone(raw['position'], raw['reward_zone'],
                                     trial_starts, trial_ends)
...
zone_label = zone_labels[i]
zone_idx = ZONE_LABELS.index(zone_label) if zone_label in ZONE_LABELS else 0
```

iii. The notes explicitly say reward-zone location should come from “position where `reward_zone > 0`,” because the stored `reward_zone` values are state codes rather than already being A/B/C.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the AI finds the mean position of samples where `reward_zone > 0`, assigns the nearest canonical zone center A/B/C, fills missing labels from nearby trials, and then broadcasts the resulting zone index across the trial.

ii.
```python
if mask.sum() > 0:
    mean_pos = pos[mask].mean()
    ...
    zone_labels[i] = best_zone
...
for i in range(n_trials):
    if zone_labels[i] is None:
        for d in range(1, n_trials):
            if i - d >= 0 and zone_labels[i - d] is not None:
                zone_labels[i] = zone_labels[i - d]
                break
...
rz_loc = np.full(T, zone_idx, dtype=int)
```

iii. The trajectory shows the AI inspected reward-zone position clusters and decided nearest-center assignment was adequate. The notes cite approximately equal A/B/C fractions as a sanity check.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from reward event timestamps, behavior timestamps, and the raw `trial_number` signal, via the per-trial `rewarded` vector.

ii.
```python
rewarded = determine_reward_per_trial(
    raw['reward_timestamps'], raw['behavior_timestamps'],
    raw['trial_number'], trial_starts, trial_ends)
...
reward_out = np.full(T, rewarded[i], dtype=int)
```

iii. The notes say reward outcome and previous reward outcome should share the same underlying per-trial reward labeling.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are matched to the nearest behavior timestamp using `argmin`, then mapped to the integer in the raw `trial_number` array at that frame. A trial’s reward-outcome output is 1 if that trial was marked rewarded, otherwise 0, and the label is broadcast across time.

ii.
```python
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behavior_timestamps - rt))
    trial_num = int(trial_number_signal[frame_idx])
    if trial_num >= 0 and trial_num < n_trials:
        rewarded[trial_num] = 1
...
reward_out = np.full(T, rewarded[i], dtype=int)
```

iii. The trajectory shows the AI understood reward was event-based and needed temporal matching to the behavior stream. The notes justify the final per-trial binary as the decoder target requested by the instructions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI crops neural and behavior arrays to a common minimum length if they disagree, inherits missing reward-zone labels from nearby trials, skips sessions with too few trials or zero surviving cells, skips trials with all-NaN neural data or too few rebinned samples, and wraps per-session processing in `try/except` so failed sessions do not stop the full run.

ii.
```python
if n_neural != n_behav:
    data['fluorescence'] = data['fluorescence'][:, :n_min]
    ...
    data['behavior_timestamps'] = data['behavior_timestamps'][:n_min]
...
if zone_labels[i] is None:
    for d in range(1, n_trials):
        if i - d >= 0 and zone_labels[i - d] is not None:
            zone_labels[i] = zone_labels[i - d]
            break
...
if np.all(np.isnan(trial_neural)):
    continue
...
except Exception as e:
    print(f"  ERROR processing {label}: {e}")
```

iii. The notes say the AI added defensive handling after exploring mismatched lengths, missing reward-zone activity, and dual-plane edge cases. They present the neighbor-fill rule and truncation as pragmatic fixes to keep sessions usable.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading large NWB files, computing full-session dF/F with repeated Gaussian/min/max filtering, processing every trial within every session, and writing the large output pickle.

ii.
```python
raw = load_nwb_session(filepath)
...
dff = compute_dff(raw['fluorescence'], raw['neuropil'], trial_starts, trial_ends)
...
for i in range(n_trials):
    ...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f)
```

iii. The notes estimate full conversion time at roughly 25 minutes and explicitly highlight per-session processing cost. The code itself times loading and dF/F separately, indicating the AI expected those steps to dominate runtime.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI code uses explicit Python loops over trials in `compute_dff`, `detect_lick_errors`, `determine_reward_zone`, `determine_reward_per_trial`, environment extraction, and final trial assembly; it also loops over cells in interneuron detection. Several of those could be vectorized or at least batched.

ii.
```python
for start, end in zip(trial_starts, trial_ends):
    ...
for c in range(n_cells):
    valid = nanmask & ~np.isnan(dff[c, :]) & ~np.isnan(speed)
    r = np.corrcoef(dff[c, valid], speed[valid])[0, 1]
...
for i in range(n_trials):
    ...
```

iii. The AI did not document a vectorization discussion in depth, but the code structure shows it prioritized a straightforward session/trial pipeline over heavier optimization.

## 13-c. What processing does the code repeat multiple times?

i. The code makes several separate passes over the same trial boundaries: once for dF/F masking and baseline estimation, again for lick-error detection, again for reward-zone inference, again for environment extraction, and again when constructing the final neural/input/output trial arrays.

ii.
```python
for start, end in zip(trial_starts, trial_ends):
    f_[:, start:end] = f_raw[:, start:end]
...
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    trial_licks = lick[start:end]
...
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    rz = reward_zone_signal[start:end]
...
for i in range(n_trials):
    start = trial_starts[i]
    end = trial_ends[i]
```

iii. This repetition follows the AI’s choice to keep each processing step simple and legible. Unlike the human reference, it avoids a separate survey pass over the whole dataset, but still repeats several within-session passes.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main discarded work is computing full dF/F traces for cells that are later removed as interneurons, plus temporary full-session arrays used only to support later slicing. There are also minor unused bookkeeping variables such as `cell_mask_list`, `offset`, and the unused `trial_number_signal` parameter to `get_trial_boundaries`.

ii.
```python
cell_mask_list = []
offset = 0
...
dff = compute_dff(raw['fluorescence'], raw['neuropil'], trial_starts, trial_ends)
is_interneuron = detect_interneurons(dff, speed_for_corr, nanmask)
keep_mask = ~is_interneuron
dff = dff[keep_mask]

def get_trial_boundaries(trial_start_signal, teleport_signal, trial_number_signal):
```

iii. The AI did not explicitly call these out in its notes, but they follow directly from its processing order: interneuron filtering happens only after full dF/F computation, and some helper bookkeeping remains even though it is not used in the final exported dataset.
