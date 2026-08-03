# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans the `data/` directory for `sub-*` folders, then scans each subject folder for `*.nwb` files, treating each NWB file as one session. Within each NWB it reads behavioral timeseries, timestamps, fluorescence, neuropil, deconvolved traces, curated-cell labels, and plane indices.

ii. 
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])

for subj_dir in subjects:
    subj_path = os.path.join(data_dir, subj_dir)
    nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
```

```python
data = {
    'position': np.array(bts.time_series['position'].data[:]),
    'speed': np.array(bts.time_series['speed'].data[:]),
    'lick': np.array(bts.time_series['lick'].data[:]),
    'reward_zone': np.array(bts.time_series['reward_zone'].data[:]),
    'teleport': np.array(bts.time_series['teleport'].data[:]),
    'trial_start': np.array(bts.time_series['trial_start'].data[:]),
    'trial_number': np.array(bts.time_series['trial number'].data[:]),
    'environment': np.array(bts.time_series['environment'].data[:]),
    'scanning': np.array(bts.time_series['scanning'].data[:]),
    'autoreward': np.array(bts.time_series['autoreward'].data[:]),
    'timestamps': np.array(bts.time_series['position'].timestamps[:]),
    'reward_timestamps': np.array(bts.time_series['Reward'].timestamps[:]),
    'reward_data': np.array(bts.time_series['Reward'].data[:]),
    'fluorescence': fluorescence,
    'neuropil': neuropil_data,
    'deconvolved': deconvolved,
    'iscell': np.array(seg['iscell'].data[:]),
    'plane_idx': np.array(seg['planeIdx'].data[:]),
}
```

iii. In `CONVERSION_NOTES.md`, the agent says it used NWB files from 11 switch-condition mice and processed each session directly from NWB. In the trajectory it justified this by noting the NWB files already contain behavior-aligned timeseries such as `position`, `speed`, `lick`, `reward_zone`, `trial_start`, `teleport`, and ophys arrays.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directory name. The agent strips the `sub-` prefix and stores a per-session `subject_idx`.

ii. 
```python
subject_name = subj_dir.replace('sub-', '')
if subject_name not in all_subjects:
    all_subjects.append(subject_name)
subj_idx = all_subjects.index(subject_name)
...
subject_idx.append(subj_idx)
```

iii. `CONVERSION_NOTES.md` states the dataset contains 11 mice (`m3`, `m4`, `m7`, `m11`-`m15`, `m17`-`m19`). The trajectory also records the agent listing those subject directories before conversion.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session metadata are taken from `nwb.session_id`, and each file contributes one entry to the top-level `neural`, `input`, and `output` lists.

ii. 
```python
for nwb_file in nwb_files:
    result = process_session(nwb_file, compute_own_dff=True)
    if result is None:
        continue
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
```

```python
session_id = nwb_data['session_id']
```

iii. In the trajectory, the agent explored the data layout and concluded there was one `behavior+ophys.nwb` file per session. `CONVERSION_NOTES.md` summarizes the result as 152 total sessions.

## 1-d. How are the data split into trials?

i. Trials are split by pairing each `trial_start` event with the next `teleport` event, so each trial spans track entry to teleport-zone entry. Trials are only kept if `scanning == 1` for at least part of that interval.

ii. 
```python
start_indices = np.where(trial_start_signal > 0)[0]
end_indices = np.where(teleport_signal > 0)[0]
...
for s in start_indices:
    next_ends = end_indices[end_indices > s]
    if len(next_ends) > 0:
        e = next_ends[0]
        if np.any(scanning[s:e] == 1):
            trial_starts.append(s)
            trial_ends.append(e)
```

iii. The reference `multi_anim_sess_README.md` recovered in the trajectory describes `trial_start_inds` as track start and `teleport_inds` as teleport start, and `behavior.py:get_trial_types()` also uses `firstI, lastI = sess.trial_start_inds[trial], sess.teleport_inds[trial]`. `CONVERSION_NOTES.md` incorrectly says “from one trial_start to the next”; the implemented code does not do that.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level filtering is minimal and ad hoc. The agent keeps only intervals that have a matching start/end pair, contain some scanning, and are at least 5 frames long. It also skips whole sessions with fewer than 5 curated cells, fewer than 3 detected trials, or fewer than 2 valid retained trials.

ii. 
```python
if n_total_cells < 5:
    return None
...
if len(trial_starts) < 3:
    return None
...
if np.any(scanning[s:e] == 1):
    trial_starts.append(s)
    trial_ends.append(e)
...
if n_timepoints < 5:
    continue
...
if valid_trial_count < 2:
    return None
```

iii. The justification in the trajectory is mostly decoder-driven rather than paper-driven: the agent wanted valid sessions for training and wanted to exclude non-imaged periods. There is no corresponding reference-paper statement for the `>=5 cells`, `>=3 trials`, or `>=5 frames` thresholds.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data are derived from raw fluorescence (`Fluorescence`) and neuropil (`Neuropil`) arrays, with curated-cell labels from `iscell`; precomputed deconvolved data are loaded but not used in the main path. Running speed is also used for interneuron filtering.

ii. 
```python
fluorescence = np.array(ophys['Fluorescence']['plane0'].data[:])
neuropil_data = np.array(ophys['Neuropil']['plane0'].data[:])
deconvolved = np.array(ophys['Deconvolved']['plane0'].data[:])
...
iscell = nwb_data['iscell'][:, 0].astype(bool)
...
F = nwb_data['fluorescence'].T
Fneu = nwb_data['neuropil'].T
deconv_nwb = nwb_data['deconvolved'].T
speed = nwb_data['speed']
```

iii. `CONVERSION_NOTES.md` says the agent recomputed dF/F from fluorescence and neuropil to match `preprocessing.py`, then deconvolved and filtered interneurons. The trajectory says it explicitly chose to recompute rather than trust the NWB deconvolved field.

## 2-b. How is the `neural` data processed?

i. The agent recomputes dF/F per trial by masking to within-trial samples, subtracting `0.7 * neuropil`, adding back the per-trial neuropil mean, computing a per-trial maximin baseline with a 300-sample window, dividing by `abs(baseline)`, smoothing with a Gaussian of sigma 2 samples, then deconvolving with OASIS.

ii. 
```python
f_[:, nanmask] = f_[:, nanmask] - NEUROPIL_COEF * f_neu_[:, nanmask]
...
f_[:, start:end] = f_[:, start:end] + NEUROPIL_COEF * np.nanmean(
    f_neu_[:, start:end], axis=1, keepdims=True)
f_smooth = nansmooth(f_[:, start:end], 15, axis=1)
baseline = ndimage.minimum_filter1d(f_smooth, window_size, axis=-1)
baseline = ndimage.maximum_filter1d(baseline, window_size, axis=-1)
dff[:, start:end] = (f_[:, start:end] - baseline) / np.abs(baseline)
...
dff[:, start:end] = nansmooth(dff[:, start:end], 2, axis=1)
...
events_trial = deconvolve_oasis(trial_dff_clean,
                                 frame_rate=effective_frame_rate)
```

iii. This is the clearest place where the agent cites the paper and `preprocessing.py`: both the paper excerpt and `CONVERSION_NOTES.md` describe per-trial maximin dF/F, 2-sample smoothing, and OASIS deconvolution.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered in two stages: first by the Suite2p curation flag `iscell[:, 0] == 1`, then by excluding putative interneurons whose dF/F has Pearson correlation greater than `0.5` with running speed.

ii. 
```python
iscell = nwb_data['iscell'][:, 0].astype(bool)
...
cell_mask = filter_interneurons(dff, speed, iscell)
```

```python
for idx in cell_indices:
    cell_data = dff[idx, valid]
    speed_data = speed[valid]
    if np.std(cell_data) > 0 and np.std(speed_data) > 0:
        corr = np.corrcoef(cell_data, speed_data)[0, 1]
        if corr > INTERNEURON_SPEED_CORR_THR:
            mask[idx] = False
```

iii. The paper excerpt in the trajectory states that curated ROIs were used and “additional putative interneurons” were excluded if the dF/F-speed Pearson correlation exceeded 0.5. `CONVERSION_NOTES.md` matches that criterion, though one sentence there mistakenly describes the correlation as being on deconvolved activity.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing each per-session event matrix from `trial_start` index `s` to paired `teleport` index `e`. Within each trial, time zero is implicitly the first column of `events[:, s:e]`.

ii. 
```python
for t in range(len(trial_starts)):
    s, e = trial_starts[t], trial_ends[t]
    ...
    neural_trial = events[:, s:e].copy()
```

iii. The task instructions required alignment to start of trial. The agent’s metadata also says `temporal_alignment_event: 'start of trial (entry to linear track at 0 cm)'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the imaging-frame resolution of `15.5078125 Hz`, i.e. `1000 / 15.5078125 = 64.48 ms` per sample. No temporal rebinning is applied.

ii. 
```python
FRAME_RATE = 15.5078125  # Hz, from NWB files
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~64.5 ms
```

```python
'metadata': {
    'time_bin_size': TIME_BIN_MS,
    'frame_rate_hz': FRAME_RATE,
}
```

iii. `CONVERSION_NOTES.md` explicitly reports `time_bin_size: 64.48 ms` and says the imaging was at `15.5078125 Hz`.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavioral timestamps associated with the `position` timeseries.

ii. 
```python
'timestamps': np.array(bts.time_series['position'].timestamps[:]),
...
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The trajectory shows the agent chose `position.timestamps` as the common frame clock for all aligned behavioral streams, consistent with the NWB layout it inspected.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the agent subtracts the first timestamp of that trial from every timestamp in the trial slice, producing a continuous per-frame elapsed-time vector.

ii. 
```python
time_from_start = (timestamps[s:e] - timestamps[s])
input_trial[0, :] = time_from_start
```

iii. The justification is implicit in the task requirement to align to trial start; the agent’s notes describe this variable as “Seconds since trial start.”

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is cut with the same `[s:e]` indices used for `neural_trial`, so it has the same number of columns as the neural matrix for that trial.

ii. 
```python
neural_trial = events[:, s:e].copy()
time_from_start = (timestamps[s:e] - timestamps[s])
input_trial = np.zeros((4, n_timepoints))
input_trial[0, :] = time_from_start
```

iii. The code constructs `n_timepoints = e - s` once and uses it for neural, input, and output arrays, showing that the alignment is by shared sample index.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` timeseries in the NWB file.

ii. 
```python
'environment': np.array(bts.time_series['environment'].data[:]),
...
env_type = float(environment[s])
```

iii. `CONVERSION_NOTES.md` says environment came from the NWB `environment` timeseries and corresponds to `ENV1`/`ENV2`. In the reference code recovered from the trajectory, the analogous variable is `morph`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent takes the environment value at the first frame of the trial and broadcasts it across the whole trial. If that value is negative, it defaults it to `0.0` (`ENV1`).

ii. 
```python
env_type = float(environment[s])  # per trial
if env_type < 0:
    env_type = 0.0  # default to ENV1 if unknown
...
input_trial[1, :] = env_type
```

iii. The trajectory shows the agent knew pre-synchronization samples could have `-1` in environment-like variables. Its handling of those values is ad hoc; the notes simply say environment is `0 or 1`.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. In the implemented code it is not taken from the raw `trial number` timeseries; instead it is derived from the loop index `t` over retained trials in the session.

ii. 
```python
'trial_number': np.array(bts.time_series['trial number'].data[:]),
...
for t in range(len(trial_starts)):
    ...
    trial_num = float(t)
```

iii. `CONVERSION_NOTES.md` describes this as “0-indexed trial number within session,” which matches the loop-index implementation rather than the raw field.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The agent renumbers trials from `0` to `N-1` in the order they were retained after segmentation and filtering, then broadcasts that scalar over the whole trial.

ii. 
```python
trial_num = float(t)
...
input_trial[2, :] = trial_num
```

iii. The trajectory contains no deeper reference-based argument here; this appears to have been a convenience choice by the agent.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the internally computed `is_rewarded` list, which itself is based on reward-delivery timestamps (`Reward.timestamps`) and the per-frame `reward_zone` signal within each trial.

ii. 
```python
'reward_timestamps': np.array(bts.time_series['Reward'].timestamps[:]),
'reward_zone': np.array(bts.time_series['reward_zone'].data[:]),
...
rew = determine_trial_rewarded(
    position, rz_signal, reward_timestamps, timestamps,
    trial_starts[i], trial_ends[i])
is_rewarded.append(rew)
```

iii. The reference `behavior.py:get_trial_types()` recovered in the trajectory similarly defines per-trial reward from reward delivery together with reward-zone occupancy.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trials after the first, the agent copies the previous entry of `is_rewarded`. For the first trial of a session, it hard-codes the previous outcome to `1.0` and broadcasts the scalar across the trial.

ii. 
```python
if t == 0:
    prev_outcome = 1.0  # assume rewarded before session
else:
    prev_outcome = float(is_rewarded[t - 1])
...
input_trial[3, :] = prev_outcome
```

iii. `CONVERSION_NOTES.md` does not explain the `t == 0` convention; it only says the field means “reward on previous trial.” The assumption that the pre-session trial was rewarded appears to be an unsupported agent choice.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from per-frame `position` plus a per-trial reward-zone label inferred from the `reward_zone` signal and position samples where `reward_zone > 0`.

ii. 
```python
rz_labels = determine_reward_zone_for_all_trials(
    position, rz_signal, trial_starts, trial_ends)
...
rz_positions = pos_trial[rz_trial > 0]
...
rz_start, rz_end = REWARD_ZONES[rz_label]
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end)
                       for p in pos_trial])
```

iii. `CONVERSION_NOTES.md` explicitly says reward zone was determined from the `reward_zone` behavioral signal and that distance was computed from current position to the active zone edge. In the reference code, by contrast, reward-zone coordinates are normally derived from session scene metadata.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Once a trial’s active zone is chosen, the agent computes signed distance to the nearest boundary of that zone: negative before the zone, zero inside, positive after the zone.

ii. 
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    if position < rz_start:
        return position - rz_start
    elif position > rz_end:
        return position - rz_end
    else:
        return 0.0
```

iii. This matches the agent’s notes, which define negative as before zone, zero as inside, and positive as past zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is binned into the seven requested categories: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50` cm.

ii. 
```python
if d < -50:
    result[i] = 0
elif d < -10:
    result[i] = 1
elif d < 0:
    result[i] = 2
elif d == 0:
    result[i] = 3
elif d <= 10:
    result[i] = 4
elif d <= 50:
    result[i] = 5
else:
    result[i] = 6
```

iii. These bins were taken directly from the decoder task and restated in `CONVERSION_NOTES.md`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `pos_trial = position[s:e]`, so each distance category is one sample per neural frame within the same trial slice.

ii. 
```python
pos_trial = position[s:e]
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end)
                       for p in pos_trial])
output_trial[0, :] = dist_binned
```

iii. The code uses the same `s:e` indices for neural, inputs, and outputs, which is the only explicit alignment mechanism.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the per-frame `position` timeseries.

ii. 
```python
'position': np.array(bts.time_series['position'].data[:]),
...
pos_trial = position[s:e]
```

iii. `CONVERSION_NOTES.md` describes position as extracted directly from the `position` timeseries on the 0-450 cm linear track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to the valid corridor range `[0, 450)` and then converted to equal-width bins spanning the track.

ii. 
```python
pos_clipped = np.clip(pos_trial, 0, TRACK_LENGTH - 0.001)
pos_binned = discretize_position(pos_clipped, n_bins=5)
```

```python
bin_size = TRACK_LENGTH / n_bins  # 90 cm bins
result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
```

iii. The trajectory shows the agent recognized that pre-synchronization position values can be `-500`; clipping prevents those values from entering the output after segmentation.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into five equal 90 cm bins: `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm.

ii. 
```python
'output_values': [
    ...,
    ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
    ...
]
```

iii. `CONVERSION_NOTES.md` gives the same five bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same trial slice `[s:e]` as the neural data, giving one position-bin label per neural frame.

ii. 
```python
pos_trial = position[s:e]
...
output_trial[1, :] = pos_binned
```

iii. The alignment is purely index-based and shared across all streams.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the NWB `lick` timeseries.

ii. 
```python
'lick': np.array(bts.time_series['lick'].data[:]),
...
lick_cumul = nwb_data['lick']
lick_trial = lick_cumul[s:e].copy()
```

iii. `CONVERSION_NOTES.md` says the lick signal came from the NWB lick timeseries. The trajectory also records the agent examining that field while inspecting the NWB schema.

## 9-b. What processing is involved in computing `output` *Lick*?

i. In the implemented code, lick counts are binarized framewise as `lick > 0`. No temporal differencing is actually performed.

ii. 
```python
# Lick: convert cumulative to binary
lick_trial = lick_cumul[s:e].copy()
lick_binary = (lick_trial > 0).astype(float)
...
output_trial[3, :] = lick_binary.astype(int)
```

iii. `CONVERSION_NOTES.md` claims the agent computed a diff of cumulative counts, but that is not what `convert_data.py` does. The reference `multi_anim_sess_README.md` recovered in the trajectory describes licks as per-frame counts that are typically clipped to binary, which is closer to the code than to the notes.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is taken from the same `[s:e]` trial slice as the neural data and converted to one binary label per neural frame.

ii. 
```python
lick_trial = lick_cumul[s:e].copy()
...
output_trial[3, :] = lick_binary.astype(int)
```

iii. The alignment is the same shared trial-index slicing used everywhere else in the script.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the `reward_zone` signal together with `position`, by finding where `reward_zone > 0` within each trial and matching the mean active position to one of the hard-coded zone definitions `A/B/C`.

ii. 
```python
rz_positions = pos_trial[rz_trial > 0]
...
for zone, (start, end) in REWARD_ZONES.items():
    zone_center = (start + end) / 2
    dist = abs(mean_rz_pos - zone_center)
```

iii. `CONVERSION_NOTES.md` explicitly justifies reward-zone identification this way. The reference `behavior.py:get_reward_zones()` instead derives trialwise zone identity from session scene metadata and known switch timing.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code first infers a text label per trial, fills missing labels from neighboring trials, falls back to `'A'` if the label is still missing, and then maps `A/B/C` to integers `0/1/2`.

ii. 
```python
labels[i] = determine_reward_zone_label(
    position, rz_signal, trial_starts[i], trial_ends[i])
...
if labels[i] is None:
    for j in range(i + 1, n_trials):
        if labels[j] is not None:
            labels[i] = labels[j]
            break
...
if rz_label is None:
    rz_label = 'A'  # fallback
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The justification in the notes is that omission trials may lack direct reward-zone evidence, so neighboring trials are used to infer the zone. The final hard-coded fallback to `'A'` is not justified in the notes.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward.timestamps` and the trial’s `reward_zone` signal. A trial is considered rewarded only if a reward delivery occurred during the trial and the animal entered the reward zone.

ii. 
```python
'reward_timestamps': np.array(bts.time_series['Reward'].timestamps[:]),
'reward_zone': np.array(bts.time_series['reward_zone'].data[:]),
...
reward_in_trial = np.any(
    (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
)
entered_rz = np.any(rz_trial > 0)
return int(reward_in_trial and entered_rz)
```

iii. This mirrors the reference `behavior.py:get_trial_types()` logic shown in the trajectory, where `isreward` depends on both reward delivery and reward-zone occupancy.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The agent computes a per-trial binary `is_rewarded` list first, then broadcasts the per-trial reward outcome across all frames of that trial in the output tensor.

ii. 
```python
is_rewarded = []
for i in range(len(trial_starts)):
    rew = determine_trial_rewarded(...)
    is_rewarded.append(rew)
...
rew_outcome = is_rewarded[t]
...
output_trial[5, :] = rew_outcome
```

iii. `CONVERSION_NOTES.md` says reward outcome was “detected from `Reward` timestamps relative to trial boundaries,” which is accurate but omits the additional reward-zone check found in the code.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses several ad hoc repairs: it truncates behavioral and neural arrays to the minimum common length when they differ by one sample; it fills missing reward-zone labels from neighboring trials and otherwise defaults to zone `A`; it converts negative environment values to `0`; it replaces NaNs in neural data with `0` before deconvolution and before saving trials; and it skips sessions or trials that are too small.

ii. 
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]
    ...
    nwb_data['timestamps'] = nwb_data['timestamps'][:min_len]
```

```python
if labels[i] is None:
    ...
if rz_label is None:
    rz_label = 'A'  # fallback
```

```python
if env_type < 0:
    env_type = 0.0
...
trial_dff_clean = np.nan_to_num(trial_dff, nan=0.0)
...
neural_trial = np.nan_to_num(neural_trial, nan=0.0)
```

iii. `CONVERSION_NOTES.md` explicitly justifies the length truncation and neighbor-based reward-zone filling. The other fallbacks are implicit implementation choices rather than documented reference decisions.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading each NWB session, recomputing dF/F over all cells and samples, deconvolving every trial separately, and running the per-cell speed-correlation filter. The full conversion log shows this was run over 152 sessions and 138,276 neuron-sessions.

ii. 
```python
nwb_data = load_nwb(filepath)
...
dff = compute_dff_per_trial(F, Fneu, trial_starts, trial_ends,
                             frame_rate=effective_frame_rate)
cell_mask = filter_interneurons(dff, speed, iscell)
...
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    ...
    events_trial = deconvolve_oasis(trial_dff_clean,
                                     frame_rate=effective_frame_rate)
```

iii. The trajectory and conversion logs show the agent expected full-dataset conversion and decoder training to be heavy; these loops dominate the runtime by inspection.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell interneuron-correlation loop, the per-sample loops in `discretize_distance` and `discretize_speed`, the per-trial reward-zone inference loop, and the Python list-comprehension for signed distance are all obvious vectorization candidates.

ii. 
```python
for idx in cell_indices:
    ...
    corr = np.corrcoef(cell_data, speed_data)[0, 1]
```

```python
for i, d in enumerate(distances):
    ...
for i, s in enumerate(speeds):
    ...
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end)
                       for p in pos_trial])
```

iii. The agent did not document these efficiency issues directly, but they are clear from `convert_data.py` and matter because the dataset is large.

## 13-c. What processing does the code repeat multiple times?

i. The code loops over the same trial boundaries multiple times for masking, baseline computation, smoothing, deconvolution, reward-zone inference, reward-outcome inference, and final packaging. It also loads the NWB deconvolved traces even though it recomputes its own events in the main path.

ii. 
```python
for start, end in zip(trial_starts, trial_ends):
    ...
for start, end in zip(trial_starts, trial_ends):
    ...
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    ...
for i in range(len(trial_starts)):
    ...
for t in range(len(trial_starts)):
    ...
```

iii. This repetition is mostly a direct consequence of implementing each processing stage separately rather than building a more fused pipeline.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main unnecessary work is reading `Deconvolved` from NWB and storing it in `deconv_nwb` even though `compute_own_dff=True` is always used in `convert_all_data`; similarly, `reward_data`, `n_planes`, and the imported `interpolate` module are not used downstream in the generated dataset. The code also computes trialwise dF/F and OASIS events from scratch even though the NWB already contains deconvolved traces.

ii. 
```python
deconvolved = np.array(ophys['Deconvolved']['plane0'].data[:])
...
deconv_nwb = nwb_data['deconvolved'].T  # Pre-computed deconvolved
...
result = process_session(nwb_file, compute_own_dff=True)
```

```python
'reward_data': np.array(bts.time_series['Reward'].data[:]),
...
from scipy import ndimage, interpolate
```

iii. The agent justified recomputation as being closer to the paper’s preprocessing, but from the perspective of runtime and final saved outputs, several loaded or computed objects are discarded.
