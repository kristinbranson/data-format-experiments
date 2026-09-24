# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered with a single glob over `<script_dir>/data/sub-*/*.nwb` (152 files) and each is opened directly with `h5py` (not `pynwb`). For each file the script reads, in one pass: the metadata (`general/subject/subject_id`, `general/session_id`, `identifier`, `general/optophysiology/ImagingPlane/imaging_rate`), every behavioural time series it needs from `processing/behavior/BehavioralTimeSeries` (position, speed, lick, reward_zone, trial number, trial_start, teleport, environment, scanning, position timestamps, Reward data + Reward timestamps), and the ophys arrays from `processing/ophys` (`iscell`, `planeIdx`, and `Fluorescence`, `Neuropil`, `Deconvolved` for every plane). Multi-plane sessions are concatenated along the ROI axis. There is only one pass over the data — no separate survey pass.

ii.
```python
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
...
for nwb_path in nwb_files:
    fname = os.path.basename(nwb_path)
    session_label = fname.replace('_behavior+ophys.nwb', '')
    result = process_session(nwb_path, ...)
```
```python
with h5py.File(nwb_path, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    identifier = f['identifier'][()].decode()
    imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
    bts = f['processing/behavior/BehavioralTimeSeries']
    position = bts['position/data'][:]
    ...
    fluor_planes = sorted([k for k in ophys['Fluorescence'].keys() if k.startswith('plane')])
    for pi, plane_name in enumerate(fluor_planes):
        deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
        F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
        Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
    deconv_all = np.concatenate(deconv_list, axis=1)
```

iii. From CONVERSION_NOTES Step 2: the directory layout is `data/sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb`, one level deep, giving "11 subjects (m3, m4, m7, m11-m15, m17-m19), 12-14 sessions each, ~152 total sessions". The agent used raw `h5py` rather than `pynwb` because it only needs a fixed set of paths and wanted to avoid the NWB object-model overhead. It cross-checked the resulting counts against the paper (11 switch mice, 14 imaging days, m11 starting on day 3 → 12 sessions) and reports 152/152 sessions converted.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from inside each NWB file (`general/subject/subject_id`), not from the directory name. The unique ids are collected into a set, sorted, and each session gets an index into that list.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
...
subjects_set.add(result['subject_id'])
...
subjects = sorted(subjects_set)
for sess in all_sessions:
    subject_idx.append(subjects.index(sess['subject_id']))
data['subjects'] = subjects
data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
```

iii. CONVERSION_NOTES Step 2/Step 9: the agent verified that this yields 11 subjects with the expected per-subject session counts (m11: 12, all others: 14), matching "n = 11 mice" in the paper and the fact that imaging for m11 started on day 3.

## 1-c. How are the data split into sessions?

i. One session per NWB file. Sessions are kept in `sorted(glob(...))` order (subject directory order, then session number), and `session_id` from the file is used only to build the `session_info` metadata string. A session is dropped if it has fewer than 2 curated cells or fewer than 2 usable trials; in practice none were dropped.

ii.
```python
session_id = f['general/session_id'][()].decode()
...
'session_info': f"{subject_id}_ses-{session_id}_{scene}",
...
if valid_trial_count < 2:
    print(f"  Skipping {session_label}: only {valid_trial_count} valid trials")
    return None
```

iii. CONVERSION_NOTES Step 2: the file name encodes `ses-<number>`, and each file holds one continuous imaging day. The ≥2-trial / ≥2-cell guards exist because the target format requires "at least two trials within each session in order to evaluate the decoder performance".

## 1-d. How are the data split into trials?

i. A trial is the half-open sample range `[trial_start_index, teleport_index)`. Both boundaries come from the binary behavioural flags `trial_start` and `teleport`; the agent takes every nonzero sample of each flag and pairs them in order, truncating to `min(len(starts), len(teleports))`. The teleport/inter-trial-interval samples are therefore excluded from the converted trials.

ii.
```python
trial_start_inds = np.where(trial_start_flag > 0)[0]
teleport_inds = np.where(teleport_flag > 0)[0]

n_trials = min(len(trial_start_inds), len(teleport_inds))
if n_trials < 2:
    print(f"  Skipping {session_label}: only {n_trials} trials")
    return None

trial_start_inds = trial_start_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
...
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    ...
    trial_pos = position[s:e]
    trial_neural = neural_data[:, s:e]
```

iii. CONVERSION_NOTES Step 1 records the reference code's trial convention ("Trial alignment: trial_start_inds to teleport_inds") and Step 10 states "Trial alignment verified: our NWB flags correspond to reference's start-1/stop-1 indexing". Step 9 reports 12,216 trials, 80.4 per session, against the paper's "80.5 ± 7.4 trials".

## 1-e. How are trials filtered based on quality controls?

i. Three guards, all structural rather than behavioural: (1) a trial is skipped if `teleport_index <= trial_start_index` or the trial is shorter than 2 samples; (2) a session is skipped if it has fewer than 2 cells after curation; (3) a session is skipped if fewer than 2 trials survive. No trial is dropped for behavioural reasons. The lick-sensor-error rule (see 9-b) modifies a trial's lick output but does not drop the trial. When a trial is skipped, the running "previous trial rewarded" state is still updated so the next trial's input stays correct.

ii.
```python
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]

    if e <= s or (e - s) < 2:
        # Still need to track prev_trial_rewarded
        trial_start_time = pos_timestamps[s] if s < len(pos_timestamps) else 0
        trial_end_time = pos_timestamps[min(e, len(pos_timestamps) - 1)]
        was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
        prev_trial_rewarded = int(was_rewarded)
        continue
```
```python
if n_cells_iscell < 2:
    print(f"  Skipping {session_label}: only {n_cells_iscell} cells after iscell filter")
    return None
...
if n_final_cells < 2:
    print(f"  Skipping {session_label}: only {n_final_cells} cells after interneuron exclusion")
    return None
```

iii. The agent's stated trial curation rules (CONVERSION_NOTES Step 3) are the paper's: lick-sensor error handling and restriction to trial_start→teleport. It deliberately did **not** apply the paper's 2 cm/s speed threshold, because "Speed is an output to decode, so we keep all frames. The reference decoder applied speed thresholding, but our task is different" (Step 5, Key Decision 2). The 2-sample / 2-trial / 2-cell guards are defensive minimums for the decoder format. Nothing was actually dropped: 12,216 trials and 152 sessions were emitted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` array is the NWB **`processing/ophys/Deconvolved/plane*/data`** array, concatenated across planes and subset to the curated cells. `Fluorescence` and `Neuropil` are also loaded, but only to compute a dF/F trace used for interneuron detection — that dF/F never reaches the output. The agent did not recompute the paper's own dF/F→OASIS "events".

ii.
```python
for pi, plane_name in enumerate(fluor_planes):
    deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]  # (n_timepoints, n_rois_plane)
    F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
    Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
    ...
deconv_all = np.concatenate(deconv_list, axis=1)
...
deconv_cells = deconv_all[:, cell_mask].T  # (n_cells, n_timepoints)
...
neural_data = deconv_cells[final_cell_mask]
...
trial_neural = neural_data[:, s:e]
neural_trials.append(trial_neural.astype(np.float32))
```

iii. CONVERSION_NOTES Step 1: "The NWB files already have the deconvolved events computed. We need to: 1. Compute dF/F ourselves from F and Fneu (same as reference code) OR 2. use the pre-computed deconvolved events directly ... The reference decoder (Fig3) uses deconvolved events, NOT dF/F." Step 4 resolves this as "The NWB data already has deconvolved events pre-computed via the reference pipeline (suite2p OASIS). We should use these directly." Step 10 Check 3 asserts "deconvolved events from NWB match `sess.timeseries['events']`", and metadata records `'neural_data_type': 'deconvolved calcium events (OASIS)'`.

## 2-b. How is the `neural` data processed?

i. Essentially no processing: the stored `Deconvolved` traces are sliced per trial and cast to `float32`. No neuropil subtraction, no maximin baseline, no dF/F normalisation, no re-deconvolution, no smoothing, no per-cell scaling is applied to the emitted signal. The paper's dF/F pipeline (neuropil subtraction with coefficient 0.7, Gaussian σ=15 smoothing, 300-sample running-min then running-max maximin baseline, `(F-baseline)/|baseline|`, Gaussian σ=2 smoothing) **is** implemented in `compute_dff_trial()` and run per trial, but its only consumer is the speed-correlation interneuron test.

ii.
```python
def compute_dff_trial(F_trial, Fneu_trial, baseline_window=BASELINE_WINDOW):
    F_corr = F_trial - NEUROPIL_COEF * Fneu_trial
    ...
    smoothed = gaussian_filter1d(F_corr, sigma=15, axis=1)
    baseline = minimum_filter1d(smoothed, size=min(baseline_window, n_t), axis=1)
    baseline = maximum_filter1d(baseline, size=min(baseline_window, n_t), axis=1)
    abs_baseline = np.abs(baseline)
    abs_baseline[abs_baseline < 1e-10] = 1e-10
    dff = (F_corr - baseline) / abs_baseline
    dff = gaussian_filter1d(dff, sigma=DFF_SMOOTH_SIGMA, axis=1)
    return dff
```
```python
dff_full = np.full_like(F_cells, np.nan)
for i in range(n_trials):
    s = trial_start_inds[i]; e = teleport_inds[i]
    if e <= s: continue
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])

is_interneuron = detect_interneurons(dff_full, speed)   # only use of dff_full
...
neural_data = deconv_cells[final_cell_mask]             # emitted signal is the NWB Deconvolved
```

iii. CONVERSION_NOTES Step 5, Key Decision 1: "**Neural data = deconvolved events**: The paper's decoder uses deconvolved events, and NWB has them pre-computed", and Step 4: "Can use directly instead of recomputing dF/F then deconvolving". The dF/F function is documented as existing solely "for interneuron detection (maximin baseline, neuropil subtraction)" (Step 6).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, applied in series and matching the paper: (1) suite2p's curated `iscell` column 0 == 1; (2) putative interneurons removed, defined as cells whose per-trial dF/F correlates with running speed at Pearson r > 0.5 over all within-trial samples of the session. The correlation is computed with a vectorised formula rather than a per-cell loop, restricted to samples where speed and dF/F are both finite (dF/F is NaN outside trials by construction).

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
n_cells_iscell = cell_mask.sum()
...
F_cells = F_all[:, cell_mask].T
Fneu_cells = Fneu_all[:, cell_mask].T
deconv_cells = deconv_all[:, cell_mask].T
...
is_interneuron = detect_interneurons(dff_full, speed)
final_cell_mask = ~is_interneuron
neural_data = deconv_cells[final_cell_mask]
```
```python
def detect_interneurons(dff_all, speed_all, threshold=INTERNEURON_SPEED_CORR_THRESHOLD):
    valid = ~np.isnan(speed_all) & ~np.isnan(dff_all[0])
    ...
    corr = dff_centered @ speed_centered / (dff_std * speed_std + 1e-10)
    is_interneuron = corr > threshold
```

iii. CONVERSION_NOTES Step 3 "Neuron curation rules": "1. Suite2p automatic classification + manual curation (stored in iscell); 2. Interneuron exclusion: Pearson correlation of dF/F with speed > 0.5". Expected statistics table records the paper's "Interneurons excluded 0.42±0.85% (Pearson corr >0.5 with speed)". The conversion log shows 284 interneurons removed of 138,678 curated cells (0.20%), and 910.5 neurons/session (paper range 155–2172; converted 155–2327).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start is achieved purely by the index slice: the neural array and every behavioural array share one sample index, so slicing `[:, s:e]` with `s = trial_start_index` puts sample 0 of every trial at the trial-start event. `off_start` is recorded as 0.0 and `off_end` as `None` (variable trial length). Nothing is padded, shifted, or interpolated. The only alignment operation is a defensive crop of neural and behavioural streams to a common length when they differ (see 12).

ii.
```python
s = trial_start_inds[i]
e = teleport_inds[i]
...
trial_neural = neural_data[:, s:e]   # (n_cells, n_timepoints)
trial_pos = position[s:e]
trial_timestamps = pos_timestamps[s:e]
```
```python
'temporal_alignment_event': 'Start of trial (entry to linear track)',
'off_start': 0.0,
'off_end': None,  # variable trial length
```

iii. CONVERSION_NOTES Step 5, Key Decision 4: "Trial alignment = trial start: As specified in Decoder Task section", and Step 3: "Temporal alignment: all behavioral and neural data at ~15.5 Hz frame rate, already synchronized". Step 10 Check 2 spot-checks a converted sample against the NWB array at a named (session, trial, neuron, timepoint).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning at all: one converted time bin is one imaging frame. The bin size is `1000 / effective_rate` ms, where `effective_rate` is the stored `imaging_rate` for single-plane sessions and `imaging_rate / n_planes` for the two-plane animals (m17, m18), because the stored rate there is the scanner rate (31.015625 Hz) rather than the per-plane rate (15.5078125 Hz). This gives 64.48 ms for every session. The metadata `time_bin_size` is taken from the first session.

ii.
```python
n_planes = len(fluor_planes)
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
time_bin_ms = 1000.0 / effective_rate
```
```python
time_bin_ms = 1000.0 / all_sessions[0]['imaging_rate']
...
'time_bin_size': time_bin_ms,
'imaging_rate_hz': all_sessions[0]['imaging_rate'],
```

iii. CONVERSION_NOTES Step 5, Key Decision 3: "**Time bin = imaging frame**: ~64.5 ms per frame at ~15.5 Hz. This matches the native sampling rate." Step 4 records the multi-plane resolution: "Multi-plane animals (m17, m18): Have plane0 and plane1, imaging_rate=31.015625 Hz (15.5 Hz per plane)", consistent with the paper's "unidirectional scanning at ~15.5 Hz".

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is **not** read from the stored behavioural timestamps. It is synthesised from the sample index within the trial and the effective imaging rate: `arange(n_t) / effective_rate`. The stored `position/timestamps` array is loaded and used, but only to locate reward events relative to trial boundaries.

ii.
```python
pos_timestamps = bts['position/timestamps'][:]
...
n_t = e - s
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
...
input_arr[0, :] = time_from_start
```

iii. CONVERSION_NOTES Step 5 variable mapping: "Time from trial start | input[0] | (t - trial_start) / imaging_rate, continuous seconds | Time-varying", justified by Step 3/Step 4's finding that "all behavioral and neural data [are] at ~15.5 Hz frame rate, already synchronized" — i.e. the agent treats the sampling as exactly uniform, so an index divided by the rate is the elapsed time. Step 10 Check 2 verifies it: "Input time_from_start at timepoint 10: Expected=0.644836, Converted=0.644836 - MATCH".

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. None beyond the construction itself. Because the value is built as `arange(n_t)/rate`, it starts at exactly 0.0 at the trial-start sample and increases by one frame period per sample; no subtraction of an offset is needed. It is stored as `float32` in row 0 of the `(4, n_t)` input array.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
input_arr = np.zeros((4, n_t), dtype=np.float32)
input_arr[0, :] = time_from_start
```

iii. Common sense plus the uniform-rate assumption documented in Step 3/4 (see 3-a). The agent verified the resulting range in Step 9 (`time_from_trial_start_s: [0.0, 216.5]` s across the dataset, with per-session maxima mostly 15–70 s).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the input row is generated with exactly `n_t = e - s` samples from the same `[s, e)` slice used for the neural data, so the two are aligned element-for-element. The only alignment work in the script is the up-front crop of all neural and behavioural streams to a common length when the NWB arrays disagree by a frame.

ii.
```python
n_timepoints_total = min(n_timepoints_neural, n_timepoints_behav)
if n_timepoints_neural != n_timepoints_behav:
    deconv_all = deconv_all[:n_timepoints_total]
    ...
    pos_timestamps = pos_timestamps[:n_timepoints_total]
...
n_t = e - s
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
input_arr = np.zeros((4, n_t), dtype=np.float32)
```

iii. CONVERSION_NOTES Step 10, "Issues Found and Resolved": "**Multi-plane length mismatch**: m17/m18 sessions had neural data 1 frame longer than behavioral. Fixed by truncating to min length." Step 3: behavioural and neural streams are sampled on the same clock.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioural time series (`processing/behavior/BehavioralTimeSeries/environment/data`).

ii.
```python
environment = bts['environment/data'][:]
...
trial_env = environment[s:e]
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
```

iii. CONVERSION_NOTES Step 5 mapping: "Environment (morph) | input[1] | 0=ENV1, 1=ENV2, per-trial scalar | From `environment` behavioral TS". Step 9 confirms the converted range is exactly `[0.0, 1.0]`, matching "Environment types 0 (Env1) and 1 (Env2)".

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The environment samples inside the trial are reduced to a single per-trial scalar by taking the median of the samples with value ≥ 0 (the series is −1 during the inter-trial/teleport period), with a fallback of 0.0 if no valid sample exists. That scalar is then broadcast across all timepoints of the trial so the input stays a `(4, n_t)` time-varying array.

ii.
```python
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
...
input_arr[1, :] = env_type  # broadcast
```

iii. The masking of negative values is the agent's handling of the sentinel `-1` that the `environment` series carries outside laps; the median makes the per-trial value robust to any residual sentinel samples. The target format requires `(n_input, n_timepoints)` arrays, hence the broadcast. Step 9 verifies that per-session environment ranges are `[0,0]`, `[1,1]`, or `[0,1]` for the cross-environment switch sessions.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. From the loop counter over the `trial_start`/`teleport` index pairs — i.e. the 0-based ordinal position of the trial within the session. The stored `trial number` behavioural time series is read into `trial_num` but never used. (CONVERSION_NOTES Step 5 states the source as "From `trial number` behavioral TS", which does not describe what the code does; the two agree numerically, since the stored series equals the loop index on every trial I checked.)

ii.
```python
trial_num = bts['trial number/data'][:]   # loaded, never used
...
for i in range(n_trials):
    ...
    # Trial number (per trial scalar)
    trial_number = np.float32(i)
    ...
    input_arr[2, :] = trial_number  # broadcast
```

iii. CONVERSION_NOTES Step 5 lists it as "Trial number within session | input[2] | 0-indexed, per-trial scalar". Step 9 checks the range: `trial_number: [0.0, 99.0]`, with per-session maxima of 79 (the usual 80-trial session), and 39/49/59/74/89/99 for the short and long sessions.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None: the loop index is cast to `float32` and broadcast constant across all timepoints of the trial. Note that the index counts *all* `trial_start` events, including any that were skipped by the length guard, so numbering stays tied to the session's true trial ordinal rather than to the position in the emitted list.

ii.
```python
trial_number = np.float32(i)
input_arr[2, :] = trial_number  # broadcast
```

iii. As above — a sequential within-session index, kept continuous so that the decoder sees a monotone within-session time-on-task covariate, which is what the reward-zone switch at trial 30 is defined against.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the `Reward` behavioural time series' **timestamps** (`BehavioralTimeSeries/Reward/timestamps`), compared against the trial's own start/end times taken from `position/timestamps`. The `Reward/data` amounts are read but not used.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
reward_data = bts['Reward/data'][:]      # loaded, never used
...
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    if len(reward_timestamps) == 0:
        return False
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
```

iii. CONVERSION_NOTES Step 5 mapping: "Previous trial outcome | input[3] | 0=omitted, 1=rewarded, per-trial scalar | From reward detection in previous trial", and Key Decision 7: "**Reward detection**: Check if any reward timestamp falls within trial window". The `Reward` series carries its own timestamps rather than being sampled on the behaviour clock, hence the time-window test rather than an index lookup.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A running scalar `prev_trial_rewarded` is carried through the trial loop. It is initialised to 0, so the first trial of every session gets "previous = omitted". At the end of each iteration it is set to this trial's reward outcome, and it is also updated inside the skip branch so a dropped trial does not break the chain. The value is broadcast constant across the trial's timepoints.

ii.
```python
prev_trial_rewarded = 0  # For the first trial, assume no previous reward

for i in range(n_trials):
    ...
    was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
    ...
    prev_outcome = np.float32(prev_trial_rewarded)
    ...
    input_arr[3, :] = prev_outcome  # broadcast
    ...
    prev_trial_rewarded = int(was_rewarded)
```

iii. CONVERSION_NOTES Step 10, Check 5 (edge cases): "First trial prev_outcome = 0 (correct)". The binary coding (omitted = 0, rewarded = 1) is taken straight from the Decoder Task specification; Step 9 verifies the converted range is `[0.0, 1.0]` in every session.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From two things: the `position` behavioural time series, and the reward-zone coordinates for that trial. The zone coordinates are **not** read from the `reward_zone` time series (which is loaded as `rzone_cumul` but never used). They are looked up from the scene name embedded in the NWB `identifier` field, combined with a fixed within-session switch trial of 30 and the paper's canonical zone coordinates A = [80,130], B = [200,250], C = [320,370]. This reimplements the reference repo's `behavior.get_reward_zones()`.

ii.
```python
REWARD_ZONE_DICT = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
DEFAULT_CHANGE_TRIAL = 30

def get_scene_from_identifier(identifier):
    return identifier.split('/')[-1]

def get_reward_zones(scene, n_trials, change_trial=DEFAULT_CHANGE_TRIAL):
    rz_coords = np.zeros((n_trials, 2)); rz_labels = np.empty(n_trials, dtype='U1')
    if 'Location' in scene and '_to' not in scene:
        loc = scene.split('Location')[-1]
        rz_coords[:] = REWARD_ZONE_DICT[loc]; rz_labels[:] = loc
    elif 'A_to' in scene and scene[-1] == 'B':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['A']; rz_labels[:change_trial] = 'A'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['B']; rz_labels[change_trial:] = 'B'
    ...
    return rz_coords, rz_labels
```
```python
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
rz_start, rz_end = rz_coords[i]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. CONVERSION_NOTES Step 1 lists `get_reward_zones()` (behavior.py) as the reference function that "Maps scene name to reward zone [start,stop] coords and labels per trial" and records "Switch trial: trial 30 (0-indexed), change_trial parameter in get_reward_zones" and "Reward zones: A=[80,130], B=[200,250], C=[320,370] (from reward_zone_dict using X,Y,Z keys)". Step 4 cross-checks the zone coordinates against the paper ("zone A, 80-130 cm" etc.) and flags the change-trial assumption: "Use change_trial=30 by default; need to detect from data for robustness". Step 10 notes a fix for cross-environment scenes: "Initial parser didn't handle scenes like 'Env1_B_to_Env2_C'. Fixed to use reference code's pattern matching logic".

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed distance to the nearest point of the zone: `position - rz_start` when before the zone (negative), exactly 0.0 anywhere inside `[rz_start, rz_end]`, and `position - rz_end` when past it (positive). The continuous value is then discretised (7-c). No smoothing, clipping, or wrapping around the end of the track is applied.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    dist[before] = position[before] - rz_start  # negative
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end      # positive
    return dist
```

iii. CONVERSION_NOTES Step 5: "Distance = position - nearest_reward_zone_edge (signed, negative=before, 0=in zone) ... Need to compute: min distance to any point in reward zone (0 if inside)", which is the Decoder Task's "Distance to any location in the reward zone". Step 9 reports the resulting class fractions (`0.253/0.102/0.074/0.237/0.021/0.072/0.242`).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Into the 7 categories of the Decoder Task, by explicit boolean masks rather than `np.digitize`: `< -50 → 0`, `[-50,-10) → 1`, `[-10,0) → 2`, `== 0 → 3`, `(0,10] → 4`, `(10,50] → 5`, `> 50 → 6`. The `== 0` test is exact and is what makes "inside the reward zone" its own class, which works because `compute_distance_to_reward_zone` writes a literal 0.0 for inside samples. (The module-level `DIST_RZ_BIN_EDGES` constant is defined but unused.)

ii.
```python
def discretize_distance_to_rz(dist):
    out = np.zeros(len(dist), dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
...
dist_to_rz_binned = discretize_distance_to_rz(dist_to_rz)
output_arr[0, :] = dist_to_rz_binned
```

iii. The bin edges are copied verbatim from the Decoder Task specification (CONVERSION_NOTES Step 5, "Output Discretization"), and the `output_values` names in the saved file spell them out: `['<-50cm', '-50to-10cm', '-10to0cm', '0cm_in_zone', '0to10cm', '10to50cm', '>50cm']`. Step 9 verifies the range is `[0, 6]` in every session and all 7 classes are populated.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `position[s:e]`, the same slice indices used for `neural_data[:, s:e]`, so the two are sample-for-sample aligned with no offset. The per-trial reward-zone coordinates are indexed by the same trial counter `i`.

ii.
```python
s = trial_start_inds[i]; e = teleport_inds[i]
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
...
rz_start, rz_end = rz_coords[i]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
output_arr = np.zeros((6, n_t), dtype=np.int64)
output_arr[0, :] = dist_to_rz_binned
```

iii. CONVERSION_NOTES Step 3: neural and behavioural series share one clock, so a shared index slice is sufficient; Step 10 Check 2 spot-checks a converted position bin against the raw NWB value at a named timepoint ("Raw=35.3cm, Expected bin=0, Converted bin=0 - MATCH").

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Directly from the `position` behavioural time series, in cm along the 450 cm virtual corridor.

ii.
```python
position = bts['position/data'][:]
...
trial_pos = position[s:e]
pos_binned = discretize_position(trial_pos)
output_arr[1, :] = pos_binned
```

iii. CONVERSION_NOTES Step 5 mapping: "Absolute position | output[1] | Discretized into 5 equal bins (90cm each), time-varying | From `position` behavioral TS". Step 9 checks "Track length 450 cm | pos [0, 450]".

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond the per-trial slice and the discretisation. Raw positions are used as stored; the only extra step is a `np.clip` after `np.digitize`, which folds the handful of samples marginally outside `[0, 450]` into the first and last bins instead of creating spurious classes.

ii.
```python
def discretize_position(position, n_bins=POSITION_BINS):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    binned = np.digitize(position, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned
```

iii. CONVERSION_NOTES Step 3 records the paper's "450 cm virtual linear track", and Step 5 specifies "[0,90), [90,180), [180,270), [270,360), [360,450] cm". The resulting distribution (`0.211/0.178/0.231/0.227/0.154`) is reported in Step 9.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Into 5 equal 90 cm bins spanning the 450 cm track, via `np.digitize` against `np.linspace(0, 450, 6)` minus 1, then clipped to `[0, 4]`.

ii.
```python
TRACK_LENGTH = 450.0
POSITION_BINS = 5

bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)   # [0, 90, 180, 270, 360, 450]
binned = np.digitize(position, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```
```python
'output_values': [... ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'], ...]
```

iii. Straight from the Decoder Task's "Discretized into 5 equal-sized bins spanning the 450 cm track" with the listed cut points (0/90/180/270/360). The clip is the agent's handling of samples that fall a fraction of a cm outside the nominal track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same mechanism as every other time-varying stream: `position[s:e]` uses the identical slice as `neural_data[:, s:e]`, after both have been cropped to a common session length.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
...
output_arr[1, :] = discretize_position(trial_pos)
```

iii. Same justification as 7-d: the NWB behavioural and ophys series are sampled on the same imaging clock (CONVERSION_NOTES Step 3/Step 4), verified by the Step 10 spot-check.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. From the `lick` behavioural time series (a per-sample cumulative lick count taking values 0–6 in this dataset).

ii.
```python
lick_raw = bts['lick/data'][:]
...
lick = lick_raw.copy()
...
trial_lick = lick[s:e].copy()
```

iii. CONVERSION_NOTES Step 5 mapping: "Lick | output[3] | Binary 0/1, time-varying | From `lick` behavioral TS, cap at 1".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps. First, the reference repo's lick-sensor-error rule: if more than 35% of a trial's samples have a cumulative lick count > 2, the whole trial's lick trace is **set to 0** (the reference code sets it to NaN to exclude it; the agent substitutes 0 because the decoder output must be a valid class label). Second, the remaining values are binarised: anything > 0 becomes 1.

ii.
```python
LICK_ERROR_FRACTION_THRESHOLD = 0.35  # from reference code (>35% samples with lick>2)
...
# Lick sensor error correction (from reference code):
# if >35% of samples have cumulative lick count > 2, set licks to NaN for this trial
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0  # set to 0 instead of NaN for decoder output
# Cap licks at 1 (binary)
trial_lick[trial_lick > 1] = 1
trial_lick = (trial_lick > 0).astype(np.float32)
...
output_arr[3, :] = trial_lick.astype(np.int64)
```

iii. CONVERSION_NOTES Step 1: "Lick sensor error correction: trials with >35% of samples having cumulative lick >2 get licks set to NaN; Licks capped at 1 (binary)". Step 4 resolves a paper/code discrepancy in favour of the code: "Code uses 35% (>0.35), paper says 30%. Use code value (0.35)". Step 3 expects this to affect "~0.65% (~81/12376)" of trials. The 0-substitution is justified in the code comment as being required because the decoder needs a categorical label rather than a missing value.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same index slice as the neural data (`lick[s:e]` versus `neural_data[:, s:e]`); no shift or resampling. Licks are sampled on the behaviour clock, which is the imaging clock.

ii.
```python
trial_lick = lick[s:e].copy()
trial_neural = neural_data[:, s:e]
...
output_arr[3, :] = lick_binned
```

iii. Same justification as 7-d/8-d — one shared sample index across all streams after the common-length crop.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the scene name in the NWB `identifier` field plus the trial ordinal, exactly as in 7-a: `get_reward_zones()` returns a per-trial label array of 'A'/'B'/'C' (switching at trial 30 for the `X_to_Y` sessions), which is then mapped to 0/1/2. The `reward_zone` behavioural time series is not used.

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
def rz_label_to_idx(label):
    mapping = {'A': 0, 'B': 1, 'C': 2}
    return mapping.get(label, -1)
...
rz_loc = rz_label_to_idx(rz_labels[i])
output_arr[4, :] = rz_loc  # broadcast
```

iii. See 7-a. CONVERSION_NOTES Step 5 mapping: "Reward zone location | output[4] | 0=A, 1=B, 2=C, per-trial | From scene name + trial number". Step 9 reports the resulting distribution as near-uniform ("RZ distribution: Equal (A, B, C) | A=32.9%, B=33.7%, C=33.5%"), which is the agent's sanity check that the scene parsing and switch trial are right.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The per-trial letter label is mapped to an integer class (A→0, B→1, C→2) and broadcast constant across the trial's timepoints so it is stored as a time-varying row of the `(6, n_t)` output array. Two fallback branches exist that never fire on this dataset: a `'Training'` scene would be labelled `'T'` and map to `-1`, and an unrecognised scene prints a warning and defaults to zone A.

ii.
```python
elif 'Training' in scene:
    rz_coords[:] = [275, 325]
    rz_labels[:] = 'T'
else:
    print(f"  WARNING: Unrecognized scene '{scene}', defaulting to zone A")
    rz_coords[:] = REWARD_ZONE_DICT['A']
    rz_labels[:] = 'A'
...
rz_loc = rz_label_to_idx(rz_labels[i])   # {'A':0,'B':1,'C':2}, else -1
output_arr[4, :] = rz_loc
```

iii. CONVERSION_NOTES Step 5, "Output Discretization": "Reward zone location (3 bins): 0=A, 1=B, 2=C". The target format wants time-varying outputs "if at all possible", hence the broadcast; the saved `output_values` entry is `['zone_A','zone_B','zone_C']`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From the `Reward` time series' own timestamps, tested against the trial's first and last behaviour timestamps. The stored reward amounts (`Reward/data`) are read but unused, so an event of any amount counts as a reward.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
trial_timestamps = pos_timestamps[s:e]
trial_start_time = trial_timestamps[0]
trial_end_time = trial_timestamps[-1]
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
```

iii. CONVERSION_NOTES Step 5 mapping: "Reward outcome | output[5] | 0=no, 1=yes, per-trial | From Reward timestamps within trial"; Key Decision 7: "Check if any reward timestamp falls within trial window". The `Reward` series is stored with its own timestamps rather than on the behaviour sample grid, so a time-window test is used rather than an index lookup.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A boolean "any reward timestamp inside the closed interval `[t_start, t_end]`" per trial, cast to int and broadcast across the trial's timepoints. The same call supplies the previous-trial-outcome input for the following trial, and it is also evaluated for skipped trials so the chain is unbroken.

ii.
```python
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    if len(reward_timestamps) == 0:
        return False
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
...
reward_outcome = int(was_rewarded)
output_arr[5, :] = reward_outcome  # broadcast
```

iii. CONVERSION_NOTES Step 3 expects "Reward omission rate ~15% ('randomly omitted on ~15% of trials')"; Step 9 reports the converted values as "Reward rate ~85% | 84.3%" and "Omission rate ~15% | 15.7%", which is the agent's consistency check on this output.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled defensively:
- **Neural/behaviour length mismatch**: every neural and behavioural array is cropped to `min(n_neural, n_behaviour)`. This fires on 10 of the 152 sessions (m17/m18 two-plane sessions where the ophys arrays are one frame longer than the behaviour).
- **Degenerate trials**: trials with `teleport <= trial_start` or fewer than 2 samples are skipped, while still advancing the previous-reward state.
- **Degenerate sessions**: sessions with fewer than 2 curated cells, fewer than 2 detected trials, or fewer than 2 surviving trials return `None` and are dropped.
- **Unequal trial-start/teleport counts**: the two index arrays are truncated to the shorter of the two.
- **Sentinel values in `environment`**: samples with `-1` are excluded from the per-trial median, with a 0.0 fallback if none remain.
- **Lick-sensor errors**: trials over the 35% threshold have their lick trace zeroed (9-b).
- **Division by zero in dF/F**: near-zero baselines are floored at `1e-10`; degenerate short segments return zeros; `detect_interneurons` returns "no interneurons" if fewer than 10 valid samples or zero speed variance.
- **Unrecognised scene names**: a warning is printed and zone A is assumed (never triggered).

ii.
```python
n_timepoints_total = min(n_timepoints_neural, n_timepoints_behav)
if n_timepoints_neural != n_timepoints_behav:
    deconv_all = deconv_all[:n_timepoints_total]
    ...
    pos_timestamps = pos_timestamps[:n_timepoints_total]
```
```python
n_trials = min(len(trial_start_inds), len(teleport_inds))
...
if e <= s or (e - s) < 2:
    ...
    prev_trial_rewarded = int(was_rewarded)
    continue
```
```python
abs_baseline = np.abs(baseline)
abs_baseline[abs_baseline < 1e-10] = 1e-10  # avoid division by zero
...
valid = ~np.isnan(speed_all) & ~np.isnan(dff_all[0])
if valid.sum() < 10:
    return is_interneuron
```

iii. CONVERSION_NOTES Step 10, "Issues Found and Resolved": "**Multi-plane length mismatch**: m17/m18 sessions had neural data 1 frame longer than behavioral. Fixed by truncating to min length", and "**Cross-env scene parsing**: Initial parser didn't handle scenes like 'Env1_B_to_Env2_C'. Fixed to use reference code's pattern matching logic". Step 10 Check 5 also records "No NaN in neural data" and the first-trial edge case.

## 13-a. What are the most time-consuming steps of the code?

i. The script prints per-session timing and a total. Full conversion took 681.4 s wall-clock for 152 sessions. Per-session times scale with cell count (0.8–1.1 s for the ~200-cell m11 sessions, 3.4–5.8 s for the ~1000-cell m7 sessions, up to ~10 s for the ~2300-cell m18 sessions); summing the printed per-session times accounts for roughly 500 s, leaving ~180 s in the final `pickle.dump` of the 9.4 GB output. Within a session the dominant costs are (1) reading three full `(n_timepoints, n_rois)` float arrays per plane (`Fluorescence`, `Neuropil`, `Deconvolved`) off disk, and (2) the per-trial dF/F, which runs a σ=15 Gaussian filter plus 300-sample minimum and maximum filters over every curated cell — all of it solely to detect ~0.2% interneurons.

ii.
```python
t0 = time.time()
...
elapsed = time.time() - t0
print(f"  {session_label}: {n_final_cells} neurons, {valid_trial_count} trials, "
      f"{n_interneurons} interneurons removed, {elapsed:.1f}s")
...
print(f"Saved {args.output} ({file_size:.1f} MB)")
print(f"Total time: {elapsed:.1f}s")
```

iii. CONVERSION_NOTES Step 7 records the one optimisation the agent found and applied ("Vectorized interneuron detection: 18.2s/2sess → 7.1s/2sess") and its extrapolation: "Estimated full conversion: ~9 minutes (well under 15 min limit)". The actual run was 11.4 minutes. The notes do not separately identify file I/O or pickling as costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. One loop was vectorised — the per-cell Pearson correlation in `detect_interneurons`, rewritten as a single centred matrix–vector product instead of 150–2300 calls to `scipy.stats.pearsonr` (the import is left behind, unused). Three loops remain:
- the per-trial dF/F loop, which calls three `scipy.ndimage` filters per trial (80–100 calls per session) where one segment-aware pass over the session would do;
- the main per-trial conversion loop, where `compute_distance_to_reward_zone`, `discretize_position`, `discretize_speed` and `discretize_distance_to_rz` are each applied to short trial slices although they are pure elementwise functions that could be applied once to the whole session array before splitting;
- the per-plane loop, which is inherently small (1–2 iterations).

ii.
```python
# vectorized:
dff_mean = dff_valid.mean(axis=1, keepdims=True)
dff_centered = dff_valid - dff_mean
dff_std = np.sqrt(np.sum(dff_centered ** 2, axis=1))
corr = dff_centered @ speed_centered / (dff_std * speed_std + 1e-10)
```
```python
# not vectorized:
for i in range(n_trials):
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
...
for i in range(n_trials):
    dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
    pos_binned = discretize_position(trial_pos)
    speed_binned = discretize_speed(trial_speed)
```

iii. CONVERSION_NOTES Step 7 documents only the interneuron speed-up. The per-trial dF/F loop is structurally required by the paper's method (the maximin baseline is defined within a trial), and the per-trial output loop follows the trial-list structure of the target format, so neither was treated as a bottleneck once the run came in under the 15-minute budget.

## 13-c. What processing does the code repeat multiple times?

i. Very little. Each NWB file is opened exactly once and each array read once; there is no separate survey/statistics pass, so nothing is loaded twice. The one genuine duplication is inside `process_session`: the `[s:e]` trial slices are recomputed in the dF/F loop and again in the main conversion loop, and `detect_reward_in_trial` is called once per trial to set that trial's outcome and its result is then reused (not recomputed) for the next trial's previous-outcome input — so even that is not repeated. `lick = lick_raw.copy()` makes a whole-session copy that is then copied again per trial (`lick[s:e].copy()`).

ii.
```python
for i in range(n_trials):
    s = trial_start_inds[i]; e = teleport_inds[i]
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])   # first pass over trials
...
for i in range(n_trials):
    s = trial_start_inds[i]; e = teleport_inds[i]                               # second pass over trials
    trial_pos = position[s:e]
    ...
lick = lick_raw.copy()
...
trial_lick = lick[s:e].copy()
```

iii. Not discussed in CONVERSION_NOTES. The two-pass structure over trials is forced by ordering: interneuron detection needs the whole session's dF/F before any cell can be dropped, and cells must be dropped before the neural trials are emitted.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small pieces of work are done and thrown away:
- **Arrays read but never used**: `rzone_cumul` (`reward_zone/data`), `trial_num` (`trial number/data`), `scanning`, `reward_data` (`Reward/data`), and `plane_idx` (`planeIdx`) are all read into memory and never referenced again.
- **Dead code in the input block**: an `input_data` array and a `trial_scalars` array are constructed, commented on at length, and then discarded — only `input_arr` is kept.
- **Unused module state**: `POSITION_BIN_EDGES`, `DIST_RZ_BIN_EDGES` and `SPEED_BIN_EDGES` are defined but the discretisation functions use inline literals instead; `pearsonr` and `matplotlib` are imported for code paths that either no longer use them or only run under `--show-processing`.
- **dF/F for all cells**: a full-session dF/F is computed for every curated cell (the most expensive computation in the script) and used only to compute one correlation coefficient per cell, discarding 284 of 138,678 cells. It is never emitted.
- **Loading `Fluorescence`/`Neuropil` at all** is a consequence of that: the emitted neural signal comes from `Deconvolved`, so two of the three large arrays per plane exist only for the interneuron test.

ii.
```python
rzone_cumul = bts['reward_zone/data'][:]      # never used
trial_num = bts['trial number/data'][:]       # never used
scanning = bts['scanning/data'][:]            # never used
reward_data = bts['Reward/data'][:]           # never used
plane_idx = ophys['ImageSegmentation/PlaneSegmentation/planeIdx'][:]  # never used
```
```python
# Build input array: time_from_start is (1, n_t), others are (1,) scalars
input_data = np.array([
    time_from_start,                     # (n_t,) time-varying
], dtype=np.float32)  # shape (1, n_t)
# Add per-trial scalars
trial_scalars = np.array([env_type, trial_number, prev_outcome], dtype=np.float32)
...
input_arr = np.zeros((4, n_t), dtype=np.float32)   # only this is used
```

iii. Not discussed in CONVERSION_NOTES. The unused reads are leftovers from the exploration phase (the `reward_zone` and `trial number` series were candidate sources that the final design replaced with the scene-name lookup and the loop index respectively); the dead `input_data`/`trial_scalars` block is a visible remnant of the agent working out how to represent mixed per-trial and time-varying inputs, resolved in favour of broadcasting everything to `(n_input, n_timepoints)`.
