# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files by globbing `data/sub-*/*.nwb`, then processes each file as one session. Within each file it directly reads behavioral and ophys arrays with `h5py` rather than `pynwb`.

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

iii. In `CONVERSION_NOTES.md`, the AI documented that the dataset is organized as one NWB file per session under subject directories and that there are 11 subjects and about 152 sessions. Its rationale was that globbing all NWB files from those subject folders would include the full dataset, and that direct HDF5 access was sufficient because the NWB file structure had been inspected already.

## 1-b. How are the data split into subjects?

i. Subjects are recovered from the NWB metadata field `general/subject/subject_id`. The code accumulates unique subject IDs across processed sessions, sorts them, and later maps each session to an index in that sorted list.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
...
subjects_set.add(result['subject_id'])
...
subjects = sorted(subjects_set)
subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The notes say the NWB files expose `general/subject/subject_id` and that these IDs match the subject directory structure. The AI therefore treated the metadata field as the authoritative subject split.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as exactly one session. The code loops over the list of NWB files, calls `process_session` once per file, and uses the file/session metadata as session identity.

ii.
```python
for nwb_path in nwb_files:
    fname = os.path.basename(nwb_path)
    session_label = fname.replace('_behavior+ophys.nwb', '')

    result = process_session(
        nwb_path,
        show_processing=args.show_processing and session_count < 2,
        session_label=session_label
    )
```

```python
session_id = f['general/session_id'][()].decode()
```

iii. In `CONVERSION_NOTES.md`, the AI wrote that the NWB file layout is `data/sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb` and described each NWB file as one session. That is the justification it used.

## 1-d. How are the data split into trials?

i. Trial starts are all indices where `trial_start_flag > 0`. Trial ends are all indices where `teleport_flag > 0`. The code pairs the first `min(n_starts, n_teleports)` of those indices and slices each trial as `s:e`.

ii.
```python
trial_start_inds = np.where(trial_start_flag > 0)[0]
teleport_inds = np.where(teleport_flag > 0)[0]

n_trials = min(len(trial_start_inds), len(teleport_inds))
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
...
trial_neural = neural_data[:, s:e]
```

iii. The notes and trajectory repeatedly state that trials should run from `trial_start` to `teleport`, and the AI adopted that literally. There is no separate recorded justification for using every positive `teleport` sample rather than teleport onsets.

## 1-e. How are trials filtered based on quality controls?

i. The code does not apply the human reference's `<50` timepoint trial filter. Instead, it skips trials only when `e <= s` or when the extracted trial has fewer than 2 samples, and it drops entire sessions if fewer than 2 trials remain.

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

iii. The AI's notes emphasize the decoder requirement that each session must contain at least two usable trials. No explicit note justifies the lack of a stricter per-trial minimum length filter beyond that.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` output is based on the NWB `Deconvolved` activity, but neuron inclusion is additionally determined using `iscell`, raw fluorescence `Fluorescence`, neuropil `Neuropil`, and behavioral `speed` for interneuron detection.

ii.
```python
iscell = ophys['ImageSegmentation/PlaneSegmentation/iscell'][:]
...
deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
...
speed = bts['speed/data'][:]
```

iii. The notes explicitly say the paper's decoder uses deconvolved events and that dF/F should be computed only to reproduce the interneuron-exclusion step from the reference pipeline. That is the AI's justification for using both deconvolved activity and fluorescence-derived curation signals.

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, adjusts the effective frame rate for multi-plane recordings, computes trialwise dF/F from `F` and `Fneu` using neuropil subtraction plus a maximin baseline and Gaussian smoothing, uses that dF/F for speed-correlation-based interneuron detection, and then keeps the filtered deconvolved events as the final neural data.

ii.
```python
deconv_all = np.concatenate(deconv_list, axis=1)
F_all = np.concatenate(F_list, axis=1)
Fneu_all = np.concatenate(Fneu_list, axis=1)
...
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
```

```python
F_corr = F_trial - NEUROPIL_COEF * Fneu_trial
smoothed = gaussian_filter1d(F_corr, sigma=15, axis=1)
baseline = minimum_filter1d(smoothed, size=min(baseline_window, n_t), axis=1)
baseline = maximum_filter1d(baseline, size=min(baseline_window, n_t), axis=1)
dff = (F_corr - baseline) / abs_baseline
dff = gaussian_filter1d(dff, sigma=DFF_SMOOTH_SIGMA, axis=1)
```

iii. `CONVERSION_NOTES.md` records this almost verbatim: use deconvolved events for the decoder, compute dF/F with the same maximin/neuropil-subtraction recipe as the paper, and use that dF/F only to exclude putative interneurons.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two filters: first `iscell[:, 0]`, then exclusion of cells whose dF/F is too strongly correlated with speed (`corr > 0.5`), labeling them as putative interneurons.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
...
is_interneuron = detect_interneurons(dff_full, speed)
...
final_cell_mask = ~is_interneuron
neural_data = deconv_cells[final_cell_mask]
```

```python
corr = dff_centered @ speed_centered / (dff_std * speed_std + 1e-10)
is_interneuron = corr > threshold
```

iii. The notes cite the paper's suite2p/manual curation and the additional interneuron exclusion criterion `Pearson corr > 0.5 with speed`. That is the justification the AI recorded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to trial start simply by slicing each trial from its `trial_start` index forward; there is no additional temporal shift or interpolation. The alignment event is "start of trial."

ii.
```python
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    ...
    trial_neural = neural_data[:, s:e]
```

iii. The notes say that all behavioral and neural streams are already synchronized at the imaging frame rate and that the decoder alignment event should be trial start. The AI therefore treated indexing by trial boundaries as sufficient alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native imaging-frame resolution. For single-plane sessions the bin size is `1000 / imaging_rate` ms; for multi-plane sessions it uses `imaging_rate / n_planes` as the effective per-plane rate. No rebinning or resampling is applied.

ii.
```python
n_planes = len(fluor_planes)
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
time_bin_ms = 1000.0 / effective_rate
```

```python
'time_bin_size': time_bin_ms,
'imaging_rate_hz': all_sessions[0]['imaging_rate'],
```

iii. The notes explicitly say "Time bin = imaging frame" and cite ~15.5 Hz single-plane imaging and ~31 Hz two-plane scanning yielding an effective ~15.5 Hz per plane.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is not taken from the stored behavior timestamps. Instead, it is derived from trial length in frames and the effective imaging rate computed from imaging metadata.

ii.
```python
imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
...
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
...
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. The notes justify this by saying each imaging frame is one time bin and the streams are already synchronized at that frame rate, so an evenly spaced frame-count clock was considered adequate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI creates a fresh zero-based time axis with `np.arange(n_t) / effective_rate`. This resets to 0 at trial onset and increases in constant frame-sized increments.

ii.
```python
n_t = e - s
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
...
input_arr[0, :] = time_from_start
```

iii. The recorded rationale is the same as in 3-a: native frame times were treated as uniform, so subtracting timestamps was replaced with reconstructing time from frame count.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the time vector length is `n_t`, where `n_t` is the same per-trial slice length used for the neural matrix.

ii.
```python
trial_neural = neural_data[:, s:e]
...
n_t = e - s
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
input_arr = np.zeros((4, n_t), dtype=np.float32)
```

iii. The notes say there is no extra alignment step because all signals are already sampled at the imaging frame rate and trialized with the same indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii.
```python
environment = bts['environment/data'][:]
...
trial_env = environment[s:e]
```

iii. `CONVERSION_NOTES.md` lists `environment` as the source variable for the decoder input `Environment type`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code collapses the per-frame `environment` values within a trial to one scalar by taking the median of nonnegative samples, casting that to `float32`, and broadcasting it across all timepoints in the trial.

ii.
```python
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
...
input_arr[1, :] = env_type
```

iii. The notes frame environment as a per-trial scalar input. The extra median/nonnegative handling appears to be the AI's robustness choice for dealing with invalid values such as `-1`.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the trial loop index after trialization, not from the raw NWB `trial number` series.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = np.float32(i)
```

iii. In its mapping notes, the AI explicitly decided that trial number should be the within-session trial index and not the raw `trial number` stream.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No additional processing is applied beyond converting the loop index to `float32` and repeating it across the trial.

ii.
```python
trial_number = np.float32(i)
...
input_arr[2, :] = trial_number
```

iii. The justification in the notes is simply that this should be a per-trial scalar "trial number within session."

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward/timestamps` series. The code checks whether any reward timestamp falls inside each trial and then carries the previous trial's result forward.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
```

```python
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    if len(reward_timestamps) == 0:
        return False
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
```

iii. The notes state that previous trial outcome should come from reward detection in the previous trial window and be encoded as omitted=0, rewarded=1.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first trial is assigned 0. For every later trial, the code uses a running variable `prev_trial_rewarded` that was set from the preceding trial's reward detection result, then broadcasts that scalar across the whole trial.

ii.
```python
prev_trial_rewarded = 0
...
prev_outcome = np.float32(prev_trial_rewarded)
...
input_arr[3, :] = prev_outcome
...
prev_trial_rewarded = int(was_rewarded)
```

iii. The notes justify this as a direct encoding of the requested binary previous-outcome decoder input.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the behavioral `position` time series plus per-trial reward-zone coordinates inferred from the NWB `identifier` scene string, with a hard-coded switch at trial 30 for switch sessions.

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
...
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
trial_pos = position[s:e]
```

```python
def get_reward_zones(scene, n_trials, change_trial=DEFAULT_CHANGE_TRIAL):
    ...
    elif 'B_to' in scene and scene[-1] == 'A':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['B']
        rz_labels[:change_trial] = 'B'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['A']
        rz_labels[change_trial:] = 'A'
```

iii. The notes say the scene name is embedded in the NWB identifier and that the AI wanted to mirror `behavior.get_reward_zones()` from the reference code instead of inferring reward zones from the cumulative `reward_zone` signal.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes signed distance to the nearest point in the current trial's reward zone: negative before the zone, zero inside it, positive after it.

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

```python
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes describe this exactly as "Distance = position - nearest_reward_zone_edge (signed, negative=before, 0=in zone)."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The code uses seven manual threshold regions: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

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

iii. The notes explicitly list these seven bins and say they were taken from the decoder task instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by slicing `position` with the same per-trial frame indices used for `trial_neural`, then computing distance on that slice.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
...
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes say no extra alignment is needed because behavior and neural activity are already synchronized at the frame level.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` time series.

ii.
```python
position = bts['position/data'][:]
...
trial_pos = position[s:e]
```

iii. The notes map `position` directly to the decoder output `Absolute position`.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI discretizes raw position into 5 equal-width bins spanning 0 to 450 cm and clips values outside that range into the nearest edge bin.

ii.
```python
def discretize_position(position, n_bins=POSITION_BINS):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    binned = np.digitize(position, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned
```

```python
pos_binned = discretize_position(trial_pos)
```

iii. In the planning notes, the AI explicitly says "Absolute position (5 bins): [0,90), [90,180), [180,270), [270,360), [360,450] cm" because the instructions asked for 5 equal-sized bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into the five equal-width bins implied by `np.linspace(0, 450, 6)`, with clipping at the ends.

ii.
```python
POSITION_BINS = 5
TRACK_LENGTH = 450.0
...
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
binned = np.digitize(position, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The justification is the same as in 8-b: the AI was trying to follow the explicit decoder instruction for equal-sized corridor bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned using the same trial slice `s:e` used for neural data.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
```

iii. The notes say all behavioral and neural streams are already synchronized frame-by-frame, so identical indexing is sufficient.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii.
```python
lick_raw = bts['lick/data'][:]
...
trial_lick = lick[s:e].copy()
```

iii. The notes directly map the NWB lick series to the decoder output `Lick`.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI copies the raw lick trace, zeros an entire trial if more than 35% of its samples exceed 2, caps remaining values above 1 down to 1, and then binarizes to 0/1.

ii.
```python
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
trial_lick[trial_lick > 1] = 1
trial_lick = (trial_lick > 0).astype(np.float32)
...
lick_binned = trial_lick.astype(np.int64)
```

iii. The notes cite the reference code's lick sensor error heuristic `>35% of samples having cumulative lick >2`, plus capping licks at 1. The AI adapted the erroneous-trial handling to zero-fill rather than NaN because the decoder output must be categorical.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing the lick trace with the same trial indices used for the neural data.

ii.
```python
trial_lick = lick[s:e].copy()
trial_neural = neural_data[:, s:e]
```

iii. The notes again justify this by saying the NWB behavioral and neural streams are already synchronized frame-by-frame.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the NWB `identifier` scene string, which is parsed into a scene label and then converted into per-trial reward-zone labels and coordinates.

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
```

iii. The notes say the scene name is embedded in `identifier` and that this allowed the AI to reuse the reference code's scene-to-reward-zone logic directly.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene string, assigns zone labels `A/B/C` across trials with a switch at trial 30 for switch sessions, then converts each trial's label to an integer `A=0, B=1, C=2` and broadcasts it across the trial.

ii.
```python
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
def rz_label_to_idx(label):
    mapping = {'A': 0, 'B': 1, 'C': 2}
    return mapping.get(label, -1)
...
rz_loc = rz_label_to_idx(rz_labels[i])
output_arr[4, :] = rz_loc
```

iii. The notes justify this by saying it follows `behavior.get_reward_zones()` from the paper code and preserves the trialwise reward-zone switch structure.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward/timestamps` series, compared against each trial's start and end times.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
```

iii. The notes explicitly say reward detection should be based on whether any reward timestamp falls inside the trial window.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI checks whether any reward timestamp lies between that trial's first and last behavioral timestamps. The boolean result is converted to `0/1` and repeated across all timepoints in the trial.

ii.
```python
trial_start_time = trial_timestamps[0]
trial_end_time = trial_timestamps[-1]
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
...
reward_outcome = int(was_rewarded)
output_arr[5, :] = reward_outcome
```

iii. The notes justify this as a direct implementation of the requested per-trial binary reward outcome variable.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles a few edge cases directly in code: neural/behavior length mismatches are cropped to the shorter length; sessions are skipped if too few cells or trials remain; invalid/too-short trials are skipped; unknown scenes default to reward zone A; very short dF/F segments return zeros; missing/invalid environment values are ignored by taking the median of nonnegative samples; the first trial's previous outcome defaults to 0; and lick-error trials are zeroed out.

ii.
```python
if n_timepoints_neural != n_timepoints_behav:
    deconv_all = deconv_all[:n_timepoints_total]
    ...
    pos_timestamps = pos_timestamps[:n_timepoints_total]
```

```python
if n_cells_iscell < 2:
    return None
...
if e <= s or (e - s) < 2:
    ...
    continue
...
elif 'Training' in scene:
    rz_coords[:] = [275, 325]
    rz_labels[:] = 'T'
else:
    print(f"  WARNING: Unrecognized scene '{scene}', defaulting to zone A")
    rz_coords[:] = REWARD_ZONE_DICT['A']
```

iii. The notes discuss several of these choices as robustness measures while trying to preserve the paper's processing, especially cropping small neural/behavior mismatches and keeping the decoder format categorical. There is no single consolidated justification beyond those notes and the defensive code paths.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are reading large HDF5 arrays from every NWB file, computing trialwise dF/F for every iscell ROI, running session-wide interneuron detection, iterating trial-by-trial to build output arrays, and optionally producing plots and writing the pickle.

ii.
```python
for pi, plane_name in enumerate(fluor_planes):
    deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
    F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
    Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
```

```python
dff_full = np.full_like(F_cells, np.nan)
for i in range(n_trials):
    ...
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
...
for i in range(n_trials):
    ...
    neural_trials.append(...)
    input_trials.append(...)
    output_trials.append(...)
```

iii. The AI did not write an explicit bottleneck analysis in the notes, but its notes do emphasize that dF/F is computed only for interneuron detection and that all sessions are processed end-to-end from NWB, which implies the main cost centers above.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main candidates are the per-trial dF/F loop and the per-trial conversion loop. The code already vectorizes the correlation calculation and the within-trial discretizations, but it still loops over trials to slice data, compute dF/F, and build per-trial outputs.

ii.
```python
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    if e <= s:
        continue
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
```

```python
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    ...
    dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
    ...
    neural_trials.append(trial_neural.astype(np.float32))
```

iii. The code itself suggests this tradeoff: the AI already vectorized the Pearson-correlation step in `detect_interneurons`, but kept trial loops because the session is organized as variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats trial-by-trial slicing work for both dF/F computation and final dataset construction. It also reads `F`, `Fneu`, and `Deconvolved` for every session even though only the filtered deconvolved events are ultimately saved.

ii.
```python
for i in range(n_trials):
    ...
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
...
for i in range(n_trials):
    ...
    trial_neural = neural_data[:, s:e]
    trial_pos = position[s:e]
    trial_speed = speed[s:e]
    trial_lick = lick[s:e].copy()
```

iii. The notes explicitly justify one repeated piece of work: dF/F is recomputed from fluorescence even though the saved neural output is deconvolved, because the AI wanted to reproduce the paper's interneuron exclusion criterion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several intermediate computations are not retained in the output: trialwise dF/F is computed only for interneuron detection and then discarded; `reward_data`, `trial_num`, `rzone_cumul`, `scanning`, `plane_idx`, and `plane_assignment` are loaded but unused downstream; and temporary arrays like `input_data` and `trial_scalars` are created but never used.

ii.
```python
reward_data = bts['Reward/data'][:]
rzone_cumul = bts['reward_zone/data'][:]
trial_num = bts['trial number/data'][:]
scanning = bts['scanning/data'][:]
plane_idx = ophys['ImageSegmentation/PlaneSegmentation/planeIdx'][:]
plane_assignment = []
```

```python
dff_full = np.full_like(F_cells, np.nan)
...
is_interneuron = detect_interneurons(dff_full, speed)
...
input_data = np.array([
    time_from_start,
], dtype=np.float32)
trial_scalars = np.array([env_type, trial_number, prev_outcome], dtype=np.float32)
```

iii. The notes explicitly defend the dF/F computation as necessary for interneuron exclusion, but the rest of the unused loads and temporary allocations do not appear to have a recorded justification beyond implementation convenience.
