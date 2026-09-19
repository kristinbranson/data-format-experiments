# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI gathers every `.nwb` file under `data/sub-*`, treats each file as one candidate session, and opens each file with `h5py`. Within each file it loads the behavioral arrays from `processing/behavior/BehavioralTimeSeries` and the optical physiology arrays from `processing/ophys`. Trials are not loaded as explicit NWB trial objects; they are reconstructed later from behavioral flags.

ii.
```python
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))

for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
```

```python
with h5py.File(nwb_path, 'r') as f:
    bts = f['processing/behavior/BehavioralTimeSeries']
    ophys = f['processing/ophys']
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset is organized as `data/sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb`, totals 152 sessions, and that it implemented NWB loading with `h5py`.

## 1-b. How are the data split into subjects?

i. Subjects are determined from the NWB metadata field `general/subject/subject_id`. As sessions are processed, unique subject IDs are collected into a set and later sorted into `data['subjects']`.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
...
subjects_set = set()
...
subjects_set.add(result['subject_id'])
...
subjects = sorted(subjects_set)
```

iii. The notes say there are 11 mice and list their IDs, so the AI trusted the NWB metadata to define subject identity.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The script iterates over the file list, processes each file independently, and stores one session entry in `neural`, `input`, and `output` for each successful file.

ii.
```python
for nwb_path in nwb_files:
    fname = os.path.basename(nwb_path)
    session_label = fname.replace('_behavior+ophys.nwb', '')
    result = process_session(nwb_path, ...)
```

```python
session_id = f['general/session_id'][()].decode()
```

iii. The notes explicitly describe the directory pattern as one file per session and report 152 total sessions.

## 1-d. How are the data split into trials?

i. Trial starts are all indices where `trial_start_flag > 0`. Trial ends are all indices where `teleport_flag > 0`. The script truncates the two index arrays to the same length and defines trial `i` as the half-open slice `[s:e)`.

ii.
```python
trial_start_inds = np.where(trial_start_flag > 0)[0]
teleport_inds = np.where(teleport_flag > 0)[0]

n_trials = min(len(trial_start_inds), len(teleport_inds))
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
```

```python
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    ...
    trial_neural = neural_data[:, s:e]
```

iii. In the notes the AI repeatedly describes trials as running from `trial_start` to `teleport` and says this matched its sanity checks on position range and trial counts.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not implement the reference solution's `< 50` timepoint filter. Instead, it skips only obviously invalid trials where `e <= s` or the slice length is less than 2 samples, and it rejects sessions with fewer than 2 surviving trials.

ii.
```python
if n_trials < 2:
    print(f"  Skipping {session_label}: only {n_trials} trials")
    return None
...
if e <= s or (e - s) < 2:
    ...
    continue
...
if valid_trial_count < 2:
    print(f"  Skipping {session_label}: only {valid_trial_count} valid trials")
    return None
```

iii. The notes emphasize the decoder-format requirement that each session have at least two trials, but they do not mention the reference solution's 50-sample minimum trial length.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final saved neural matrices come from the NWB `Deconvolved` datasets. The script also loads `Fluorescence` and `Neuropil`, but only to compute trialwise dF/F for interneuron detection.

ii.
```python
deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
```

```python
deconv_cells = deconv_all[:, cell_mask].T
...
neural_data = deconv_cells[final_cell_mask]
```

iii. The notes explicitly justify this by saying the paper's decoder used deconvolved events and the NWB files already contain precomputed deconvolved traces.

## 2-b. How is the `neural` data processed?

i. Processing happens in two branches. For the saved neural signal, the AI concatenates the already deconvolved NWB arrays across planes and keeps them at native resolution. Separately, it computes per-trial dF/F from `F` and `Fneu` using neuropil subtraction, a maximin baseline, and Gaussian smoothing, but it does not deconvolve that dF/F; it uses it only for interneuron detection.

ii.
```python
F_corr = F_trial - NEUROPIL_COEF * Fneu_trial
smoothed = gaussian_filter1d(F_corr, sigma=15, axis=1)
baseline = minimum_filter1d(smoothed, size=min(baseline_window, n_t), axis=1)
baseline = maximum_filter1d(baseline, size=min(baseline_window, n_t), axis=1)
dff = (F_corr - baseline) / abs_baseline
dff = gaussian_filter1d(dff, sigma=DFF_SMOOTH_SIGMA, axis=1)
```

```python
dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
...
neural_data = deconv_cells[final_cell_mask]
```

iii. `CONVERSION_NOTES.md` says the AI chose "Deconvolved events as neural data" because the NWB already contains them, and only computed dF/F "for interneuron detection."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first keeps only ROIs with `iscell[:, 0] == 1`. It then computes dF/F-based speed correlations and removes cells with correlation greater than `0.5` as putative interneurons. Entire sessions are dropped if fewer than two neurons remain.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
...
is_interneuron = detect_interneurons(dff_full, speed)
...
final_cell_mask = ~is_interneuron
...
if n_final_cells < 2:
    return None
```

iii. The notes cite the paper's suite2p/manual curation and the interneuron exclusion rule `speed correlation > 0.5`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start implicitly by trial slicing. Each per-trial neural array starts at the `trial_start_flag` index and ends at the chosen teleport index.

ii.
```python
s = trial_start_inds[i]
e = teleport_inds[i]
trial_neural = neural_data[:, s:e]
```

iii. The notes say "Trial alignment = trial start" because that is what the decoder task requested.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native imaging frame as the time bin. For multi-plane sessions it divides the scanner rate by `n_planes` to get an effective per-plane rate. No rebinning or interpolation is applied.

ii.
```python
n_planes = len(fluor_planes)
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
time_bin_ms = 1000.0 / effective_rate
```

iii. The notes say the time bin should be the imaging frame, about `64.5 ms`, and that this matches the native sampling rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from trial length and the inferred effective imaging rate, not from any recorded timestamp array. The script does load `position/timestamps`, but it does not use them to build this input.

ii.
```python
trial_timestamps = pos_timestamps[s:e]
...
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. In the notes the AI planned this as `(t - trial_start) / imaging_rate` and justified it by claiming the behavioral and neural streams were already synchronized at a constant rate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For a trial with `n_t` samples, it creates the sequence `0, 1/effective_rate, 2/effective_rate, ...` and broadcasts nothing else. There is no subtraction of the recorded trial-start timestamp.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
input_arr[0, :] = time_from_start
```

iii. The notes say the input is "continuous seconds" from trial start and assume a fixed frame interval.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: both `time_from_start` and `trial_neural` use the same `n_t = e - s` frames from the same trial slice. No interpolation or timestamp matching is performed.

ii.
```python
trial_neural = neural_data[:, s:e]
...
n_t = e - s
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. The notes state that all streams are already synchronized at about `15.5 Hz`, so the AI treated shared frame count as sufficient alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral time series `environment/data`.

ii.
```python
environment = bts['environment/data'][:]
...
trial_env = environment[s:e]
```

iii. The notes map this directly to the requested `0=ENV1, 1=ENV2` input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI collapses the within-trial `environment` samples to a single scalar by taking the median of all nonnegative samples in the trial, then repeats that scalar across all time bins of the converted trial.

ii.
```python
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
...
input_arr[1, :] = env_type
```

iii. The notes describe environment as a per-trial scalar and imply the raw value is constant within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is not taken from the NWB `trial number` stream. It is derived from the loop index `i` after trials are defined from `trial_start` and `teleport`.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = np.float32(i)
```

iii. The notes say the AI wanted "trial number within session" and elsewhere treats `trial_start` and `teleport` as the authoritative trial definition.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The within-session trial index is cast to `float32` and broadcast across all time bins of that trial.

ii.
```python
trial_number = np.float32(i)
...
input_arr[2, :] = trial_number
```

iii. The notes describe trial number as a per-trial scalar in the decoder input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward/timestamps` stream together with trial start and end times from `position/timestamps` and the trial boundary indices.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
trial_start_time = trial_timestamps[0]
trial_end_time = trial_timestamps[-1]
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
```

iii. The notes say the AI would determine reward by checking whether any reward timestamp falls inside a trial window.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first trial gets `0`. For every later trial, the code carries forward whether the immediately preceding trial had any reward event in its time interval and broadcasts that binary value across the current trial.

ii.
```python
prev_trial_rewarded = 0
...
prev_outcome = np.float32(prev_trial_rewarded)
...
prev_trial_rewarded = int(was_rewarded)
input_arr[3, :] = prev_outcome
```

iii. The notes justify this as matching the requested per-trial binary input `omitted = 0, rewarded = 1`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The distance output is derived from `position/data` and from reward-zone coordinates inferred from the NWB `identifier` scene string plus the assumed switch trial `30`. Although `reward_zone/data` is loaded, it is not actually used.

ii.
```python
position = bts['position/data'][:]
rzone_cumul = bts['reward_zone/data'][:]
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
```

```python
rz_start, rz_end = rz_coords[i]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes explicitly say reward-zone location should come from "scene name + trial number" and mention fixing cross-environment scene parsing.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes signed distance to the nearest point in the trial's reward zone: negative before the zone, zero inside it, positive after it. It then discretizes that continuous distance.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    dist[before] = position[before] - rz_start
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end
    return dist
```

iii. The notes describe this output as "distance to any point in the reward zone" and define the same signed interpretation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded into the requested 7 bins with explicit comparisons rather than `np.digitize`.

ii.
```python
out[dist < -50] = 0
out[(dist >= -50) & (dist < -10)] = 1
out[(dist >= -10) & (dist < 0)] = 2
out[dist == 0] = 3
out[(dist > 0) & (dist <= 10)] = 4
out[(dist > 10) & (dist <= 50)] = 5
out[dist > 50] = 6
```

iii. The notes list the same seven bins as part of the planned output discretization.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity are aligned by the same per-trial frame slice `[s:e)`. There is no extra resampling or interpolation.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
...
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes say neural and behavior are synchronized at the imaging frame rate, so shared indexing was treated as sufficient.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position/data` stream.

ii.
```python
position = bts['position/data'][:]
...
trial_pos = position[s:e]
```

iii. The notes describe position as the animal's VR location on the 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices the position trace trial by trial and discretizes it into 5 equal-width bins spanning `0` to `450` cm. Values outside the range are clipped into the end bins.

ii.
```python
def discretize_position(position, n_bins=POSITION_BINS):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    binned = np.digitize(position, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned
```

iii. The notes justify this from the 450 cm track length and the task instruction to use 5 equal bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Thresholds are the implicit `0, 90, 180, 270, 360, 450` cm edges from `np.linspace(0, 450, 6)`, with clipping into bins `0` through `4`.

ii.
```python
POSITION_BINS = 5
...
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
binned = np.digitize(position, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The notes define the same 90 cm binning scheme.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same trial slice boundaries as the neural data, so each neural frame is paired with the position sample at the same index.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
```

iii. The notes say all behavior streams and neural frames are treated as already synchronized.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick/data` stream.

ii.
```python
lick_raw = bts['lick/data'][:]
...
trial_lick = lick[s:e].copy()
```

iii. The notes map `lick` directly from the behavioral time series to the decoder output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI first applies a trial-level artifact rule: if more than 35% of samples in a trial exceed 2, the entire lick trace for that trial is replaced with zeros. It then caps values above 1 to 1 and binarizes `>0`.

ii.
```python
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
trial_lick[trial_lick > 1] = 1
trial_lick = (trial_lick > 0).astype(np.float32)
```

iii. The notes cite the paper's lick sensor error correction threshold and say the AI changed bad trials to zeros rather than NaNs for decoder compatibility.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by using the same per-trial sample indices `[s:e)` as the neural activity.

ii.
```python
trial_lick = lick[s:e].copy()
trial_neural = neural_data[:, s:e]
```

iii. The notes again rely on the assumption that neural and behavioral streams are frame-synchronous.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB `identifier` scene string and from trial order within the session, not from the `reward_zone` behavioral signal.

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
```

iii. The notes explicitly list reward-zone location as coming from "scene name + trial number."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed into a fixed reward-zone schedule using `get_reward_zones(...)`, with a hard-coded switch at trial `30` for switching sessions. The resulting `A/B/C` label for a trial is converted to integer `0/1/2` and broadcast across the trial.

ii.
```python
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
rz_loc = rz_label_to_idx(rz_labels[i])
...
output_arr[4, :] = rz_loc
```

iii. The notes call out "Change trial = 30" as a key decision and say the parsing logic was made to match the reference code's scene-pattern handling.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward/timestamps` stream, together with the trial start and end times obtained from trial slicing and `position/timestamps`.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
trial_start_time = trial_timestamps[0]
trial_end_time = trial_timestamps[-1]
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
```

iii. The notes say reward outcome should come from checking whether any reward timestamp falls within the trial window.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward outcome is `1` if any reward timestamp falls between the trial start time and trial end time, otherwise `0`. That scalar is then broadcast across all time bins of the trial in `output_arr[5, :]`.

ii.
```python
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    if len(reward_timestamps) == 0:
        return False
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
```

```python
reward_outcome = int(was_rewarded)
output_arr[5, :] = reward_outcome
```

iii. The notes justify this as matching the requested per-trial binary output and say reward detection should be based on timestamps within the trial window.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases defensively:
- If behavioral and neural streams differ in length, both are cropped to the minimum length.
- Sessions with fewer than 2 post-filter neurons or fewer than 2 trials are skipped.
- Trials with invalid ordering or fewer than 2 samples are skipped.
- If a trial has no nonnegative `environment` value, environment defaults to `0.0`.
- If a scene string is unrecognized, reward zone defaults to A with a warning.
- If a dF/F trial has fewer than 3 samples, `compute_dff_trial` returns zeros.
- If a lick trace looks corrupted by the `>35% >2` rule, that trial's licks are zeroed.

ii.
```python
if n_timepoints_neural != n_timepoints_behav:
    deconv_all = deconv_all[:n_timepoints_total]
    F_all = F_all[:n_timepoints_total]
    ...
    pos_timestamps = pos_timestamps[:n_timepoints_total]
```

```python
if n_t < 3:
    return np.zeros_like(F_corr)
...
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
...
print(f"  WARNING: Unrecognized scene '{scene}', defaulting to zone A")
```

iii. The notes mention the multi-plane off-by-one mismatch as a resolved issue, emphasize the first-trial previous-outcome edge case, and generally frame these branches as practical safeguards.

## 13-a. What are the most time-consuming steps of the code?

i. The dominant costs in the AI code are loading large NWB arrays from disk, computing trialwise dF/F for all kept cells in order to detect interneurons, and writing the final multi-gigabyte pickle. Unlike the human reference solution, this code does not do a separate full survey pass over the dataset.

ii.
```python
for pi, plane_name in enumerate(fluor_planes):
    deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
    F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
    Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
```

```python
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    ...
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
```

iii. The notes report a runtime win from vectorizing interneuron detection and note that the full output file is about `9.4 GB`, which makes both I/O and pickling expensive.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious candidates are the per-trial dF/F loop and the per-trial extraction loop that slices behavior, neural data, and outputs one trial at a time. Interneuron detection itself was already vectorized.

ii.
```python
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    ...
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
```

```python
for i in range(n_trials):
    ...
    trial_pos = position[s:e]
    trial_speed = speed[s:e]
    trial_neural = neural_data[:, s:e]
    ...
    neural_trials.append(trial_neural.astype(np.float32))
```

iii. The notes explicitly advertise "Vectorized interneuron detection," which implies the other trial-level loops remained unvectorized.

## 13-c. What processing does the code repeat multiple times?

i. The code does repeated work in a few places. It loads `Deconvolved`, `Fluorescence`, and `Neuropil` for every session even though only the deconvolved signal is saved. It makes one pass over trials to build `dff_full` and a second pass over the same trials to build the final per-trial outputs. It also performs reward detection separately for skipped trials and again for valid trials.

ii.
```python
deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
```

```python
for i in range(n_trials):
    ...
    dff_full[:, s:e] = compute_dff_trial(...)
...
for i in range(n_trials):
    ...
    was_rewarded = detect_reward_in_trial(...)
```

iii. The notes explicitly chose precomputed deconvolved events for the final neural output, so the extra fluorescence-processing pass exists only to support filtering.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several loaded or created values are never used downstream: `rzone_cumul`, `trial_num`, `scanning`, `reward_data`, `plane_idx`, `plane_assignment`, `n_rois_total`, `time_bin_ms` inside `process_session`, and the temporary `input_data` and `trial_scalars` arrays. The large `dff_full` array is also discarded after interneuron detection and never saved.

ii.
```python
rzone_cumul = bts['reward_zone/data'][:]
trial_num = bts['trial number/data'][:]
scanning = bts['scanning/data'][:]
reward_data = bts['Reward/data'][:]
...
plane_idx = ophys['ImageSegmentation/PlaneSegmentation/planeIdx'][:]
plane_assignment = []
```

```python
input_data = np.array([
    time_from_start,
], dtype=np.float32)
trial_scalars = np.array([env_type, trial_number, prev_outcome], dtype=np.float32)
```

iii. The notes acknowledge that the saved neural output is the NWB deconvolved signal, so all dF/F intermediates are temporary. The code itself also shows several variables that are loaded or created but never referenced again.
