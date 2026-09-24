# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are discovered as all `sub-*` sub-directories of `/app/data`; sessions are all `*.nwb` files inside each subject directory (session number parsed out of the `ses-XX` token of the filename). Files are opened with `h5py` directly (not `pynwb`) and the needed HDF5 datasets are read by path: behaviour under `processing/behavior/BehavioralTimeSeries/*`, neural data under `processing/ophys/{Deconvolved,Fluorescence,Neuropil}/plane*/data`, ROI curation under `processing/ophys/ImageSegmentation/PlaneSegmentation/{iscell,planeIdx}`, and the frame rate from `acquisition/TwoPhotonSeries/imaging_plane/imaging_rate`. Every file is read exactly once, in a single pass, and all 152 sessions / 12,216 trials end up in the output (no session or trial is dropped in practice).

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
subjects = [d.replace('sub-', '') for d in subjects_dirs]
...
for subj_i, (subj_dir, subj_id) in enumerate(zip(subjects_dirs, subjects)):
    subj_path = os.path.join(data_dir, subj_dir)
    nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
    ...
    for nwb_file in nwb_files:
        fname = os.path.basename(nwb_file)
        session_num = fname.split('ses-')[1].split('_')[0]
        result = convert_session(nwb_file, subj_id, session_num)
```
```python
with h5py.File(nwb_path, 'r') as f:
    imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
    pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
    speed = f['processing/behavior/BehavioralTimeSeries/speed/data'][:]
    lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
    rzone = f['processing/behavior/BehavioralTimeSeries/reward_zone/data'][:]
    env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
    trial_start = f['processing/behavior/BehavioralTimeSeries/trial_start/data'][:]
    teleport = f['processing/behavior/BehavioralTimeSeries/teleport/data'][:]
    timestamps = f['processing/behavior/BehavioralTimeSeries/position/timestamps'][:]
    reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
    iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:]
    deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
```

iii. From the trajectory: the agent first listed `/app/data`, counted 152 `.nwb` files across 11 `sub-*` directories, then dumped the full HDF5 tree of one file and dispatched an `Explore` sub-agent to characterise every field (reward encoding, `iscell`, multi-plane layout, sparse `Reward` series). It concluded (CONVERSION_NOTES.md) that the 11 NWB subjects are the paper's 11 "switch" mice (GCAMP3/4/7/11-15/17-19) and that the 3 "fixed-condition" mice are simply not in the DANDI release, so reading every file is reading all of the data. It used `h5py` rather than `pynwb` without commenting on the choice (it simply read the file by HDF5 path from the first exploration step onward).

## 1-b. How are the data split into subjects?

i. One subject per `sub-<id>` directory; the directory name minus the `sub-` prefix becomes the subject id (`m3`, `m4`, `m7`, `m11`...`m19`). `subject_idx` is the index of the subject directory in that sorted list, appended once per converted session.

ii.
```python
subjects = [d.replace('sub-', '') for d in subjects_dirs]
...
subject_idx_list.append(subj_i)
...
data['subjects'] = subjects
data['subject_idx'] = np.array(subject_idx_list, dtype=int)
```

iii. The agent verified the subject list against the paper (11 switch mice) and built an explicit NWB-subject → paper-animal table in CONVERSION_NOTES.md (`sub-m3` = GCAMP3, ..., `sub-m17`/`sub-m18` = the two multi-plane animals), cross-checking session counts (14 each, 12 for m11 which starts at `ses-03` because imaging began on day 3) and the starting environment (m17/m18 start in ENV2) against `sessions_dict.py`.

## 1-c. How are the data split into sessions?

i. One session per `.nwb` file. Sessions are processed in sorted filename order (= experiment day order) and appended as one entry to `neural`/`input`/`output`/`subject_idx`/`brain_region_idx`. A session would be skipped if it yielded fewer than 2 usable trials or fewer than 2 surviving cells; neither condition occurs, so all 152 sessions are kept. Cells are not tracked across sessions (the paper's cross-day ROI matching is ignored).

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
for nwb_file in nwb_files:
    session_num = fname.split('ses-')[1].split('_')[0]
    result = convert_session(nwb_file, subj_id, session_num)
    if result is None:
        continue
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
    subject_idx_list.append(subj_i)
    brain_region_idx_list.append(np.zeros(result['n_kept'], dtype=int))
```
```python
if n_trials < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {n_trials} valid trials")
    return None
```

iii. The agent read the filename convention `sub-<id>_ses-<NN>_behavior+ophys.nwb` and confirmed against `sessions_dict.py` that `ses-NN` is the experiment day (m11 starting at day 3 was used as the check). It reported "152 sessions (11 subjects, 12-14 sessions each)" as a match to the paper.

## 1-d. How are the data split into trials?

i. A trial is the on-track lap from the `trial_start` event index to the following `teleport` event index, i.e. the half-open sample range `[trial_start, teleport)`; the teleport/inter-trial period is excluded. Start and end indices are taken as all samples where each signal is > 0, truncated to the same count, and paired element-wise; pairs where the teleport index is not after the start index are dropped.

ii.
```python
tstart_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport > 0)[0]

# Ensure matching number of starts and teleports
n_trials = min(len(tstart_inds), len(teleport_inds))
tstart_inds = tstart_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]

# Remove trials where teleport comes before or at trial start
valid = teleport_inds > tstart_inds
tstart_inds = tstart_inds[valid]
teleport_inds = teleport_inds[valid]
n_trials = len(tstart_inds)
```
```python
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    n_timepoints = e - s
    neural = deconv_filtered[s:e, :].T.astype(np.float32)
    trial_pos = pos[s:e]
```

iii. CONVERSION_NOTES.md: "Start: `trial_start` signal (re-entry into virtual environment at position 0); End: `teleport` signal (end of trial, entering teleport zone)" and "Trial data includes only the on-track period (trial_start to teleport), excluding teleport zone". The agent validated the resulting counts against the paper: 12,216 trials, 80.4 ± 6.1 per session vs. the paper's 80.5 ± 7.4. It did not use the NWB `trial number` series. When it noticed a 216 s maximum trial it checked the distribution (median 12.3 s, 211/12,216 trials > 500 samples) and decided the outliers were acceptable rather than a boundary-detection failure.

## 1-e. How are trials filtered based on quality controls?

i. Essentially no quality filtering of trials. The only trial-level rejections are structural: a start/teleport pair in the wrong order, or a trial shorter than 2 samples. There is no minimum-duration criterion, no speed criterion, and no rejection of trials with bad licking, bad reward-zone information or missing data. Trials flagged as lick-sensor errors are kept, with their lick channel zeroed (see 9-b). All 12,216 raw trials survive into the output.

ii.
```python
valid = teleport_inds > tstart_inds
...
if n_timepoints < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {len(neural_trials)} valid trials after filtering")
    return None
```

iii. Not explicitly justified in the trajectory. The related decision that is justified is CONVERSION_NOTES.md item 4: "No speed threshold applied to neural data for the decoder (the paper applies speed < 2 cm/s threshold for spatial analyses, but the decoder benefits from all timepoints)". The agent also explicitly considered and rejected dropping the very long (up to 216 s) trials after inspecting the trial-length distribution.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrices are taken straight from the NWB `processing/ophys/Deconvolved/plane*/data` arrays (suite2p's own deconvolution of raw fluorescence), column-subset to the kept cells and transposed to (n_neurons, n_timepoints). `Fluorescence` and `Neuropil` are also loaded, but only to build a proxy dF/F used for interneuron detection — that dF/F is discarded and never becomes the neural signal. For the two-plane animals (m17, m18) the two planes' deconvolved arrays are concatenated along the cell axis.

ii.
```python
deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
fluor_p0  = f['processing/ophys/Fluorescence/plane0/data'][:]
neuro_p0  = f['processing/ophys/Neuropil/plane0/data'][:]
if has_multi_plane:
    deconv_p1 = f['processing/ophys/Deconvolved/plane1/data'][:]
    ...
    deconv_all = np.concatenate([deconv_p0, deconv_p1], axis=1)
...
deconv_filtered = deconv_all[:, cell_mask]
...
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. Module docstring: "Neural data: deconvolved calcium activity (OASIS algorithm, already in NWB)". In the step-35 reasoning the agent asserted that "the NWB files already contain processed Fluorescence data (which is dF/F computed per trial with a 20-second maximin baseline, smoothed with a Gaussian kernel, then deconvolved using OASIS)", i.e. it believed the stored `Deconvolved` array *is* the paper's activity-rate signal. The `Explore` sub-agent it dispatched reported the pipeline as "Raw fluorescence → neuropil subtraction → deconvolution → spike inference ... Already processed spike inference (likely OASIS algorithm)", which reinforced that belief. The agent later weighed computing dF/F itself and decided "Rather than getting bogged down in the exact dF/F computation, I'll focus on implementing the conversion script correctly and use a straightforward approach for the speed correlation check."

## 2-b. How is the `neural` data processed?

i. No processing at all beyond plane concatenation, cell selection, length truncation, per-trial slicing and a cast to float32. There is no neuropil subtraction, no per-trial maximin baseline, no dF/F, no 2-sample Gaussian smoothing and no OASIS deconvolution applied to the emitted signal; the stored suite2p values (raw-fluorescence units, range 0 to ~24,000, ~69% zeros) are passed through unchanged. The separate `compute_dff_simple` routine (neuropil subtraction with coef 0.7, per-trial mean neuropil added back, 15-sample boxcar smoothing, 300-sample minimum then maximum filter, `(F-baseline)/|baseline|`) is computed only to feed the interneuron detector, and it is not smoothed by 2 samples or deconvolved.

ii.
```python
if has_multi_plane:
    deconv_all = np.concatenate([deconv_p0, deconv_p1], axis=1)
else:
    deconv_all = deconv_p0
...
deconv_filtered = deconv_all[:, cell_mask]
...
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```
```python
def compute_dff_simple(fluorescence, neuropil, tstart_inds, teleport_inds,
                       neu_coef=0.7, baseline_window=300):
    f_corrected = fluorescence - neu_coef * neuropil
    for s, e in zip(tstart_inds, teleport_inds):
        trial_f = f_corrected[s:e, :]
        trial_f = trial_f + neu_coef * np.nanmean(neuropil[s:e, :], axis=0, keepdims=True)
        trial_smooth = ndimage.uniform_filter1d(trial_f, size=15, axis=0)
        win = min(baseline_window, trial_f.shape[0])
        baseline = ndimage.minimum_filter1d(trial_smooth, size=win, axis=0)
        baseline = ndimage.maximum_filter1d(baseline, size=win, axis=0)
        dff[s:e, :] = (trial_f - baseline) / abs_baseline
    return dff
```

iii. CONVERSION_NOTES.md, "Key Decisions" 1: "Used deconvolved activity (not dF/F) as neural data, matching the paper's use of deconvolved activity for most analyses", and 2: "Multi-plane data (m17, m18) concatenated across planes, following the paper's approach of pooling planes for all analyses except Extended Data Fig. 7". The justification for not recomputing the pipeline is the (mistaken) premise that the NWB array is already the paper's output, plus the explicit time-saving choice quoted in 2-a.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) Suite2p manual curation: only ROIs with `iscell[:,0] == 1` (138,678 of 312,110 ROIs, 44%). (2) Putative interneurons: cells whose proxy dF/F has Pearson r > 0.5 with running speed over the within-trial samples are dropped (119 cells total, 0.086% of curated cells — the paper reports 0.42 ± 0.85% and the expert reference removed 380). A session with fewer than 2 surviving cells would be dropped; this never happens. 138,559 neurons are kept in total.

ii.
```python
curated_mask = iscell[:, 0] == 1
...
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
cell_mask = curated_mask & ~is_interneuron
```
```python
def detect_interneurons(dff, speed, tstart_inds, teleport_inds, threshold=0.5):
    valid_mask = np.zeros(dff.shape[0], dtype=bool)
    for s, e in zip(tstart_inds, teleport_inds):
        valid_mask[s:e] = True
    valid_mask &= ~np.isnan(dff[:, 0])
    speed_valid = speed[valid_mask]
    for cell in range(n_cells):
        dff_valid = dff[valid_mask, cell]
        r, _ = stats.pearsonr(dff_valid, speed_valid)
        if r > threshold:
            is_interneuron[cell] = True
    return is_interneuron
```

iii. CONVERSION_NOTES.md: "1. Manual curation: Only cells with `iscell[:,0] == 1` included (Suite2P manual curation). 2. Interneuron exclusion: Cells with Pearson correlation > 0.5 between simplified dF/F and running speed are excluded — dF/F computed from raw fluorescence with neuropil subtraction (coef=0.7) and maximin baseline. Paper reports 0.42 +/- 0.85% excluded; our conversion removed 119 total interneurons". After the pilot run the agent checked the plausibility of the small counts: "the paper says only 0.42% of cells are excluded as interneurons, so finding very few is expected. m3 had 21/1052 = 2%, which is a bit high, but in the ballpark." It also validated the resulting cell counts against the paper's 155-2,172 range (obtaining 155-2,339, "slight excess likely due to multi-plane pooling").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start is implicit in the slicing: each trial's neural matrix begins exactly at the `trial_start` sample index and runs to the teleport sample, so sample 0 of every trial is the alignment event. There is no pre-event window (`off_start = 0.0`) and no post-event padding (`off_end = None`, variable trial length). Behaviour and neural series are indexed with the identical `[s:e]` slice, so the streams stay aligned sample-for-sample.

ii.
```python
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    neural = deconv_filtered[s:e, :].T.astype(np.float32)
    trial_pos = pos[s:e]
    trial_speed = speed[s:e]
    trial_lick = lick_binary[s:e]
```
```python
'temporal_alignment_event': 'start of trial (re-entry into virtual environment at position 0)',
'off_start': 0.0,
'off_end': None,  # Variable trial length
```

iii. CONVERSION_NOTES.md: "Temporal alignment: start of each trial", as required by the instructions. The agent established from the sub-agent report that "All behavioral data (except Reward) shares identical timestamps" and that the ophys arrays share the same sample count as the behaviour arrays (up to a rare off-by-one), so a common index is a valid alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: the data are kept at the native per-plane imaging/behaviour sample rate of 15.5078125 Hz, i.e. 64.4836 ms bins, identical for all sessions (the two-plane sessions store one sample per plane-frame, so their arrays are also at 15.5 Hz). `metadata['time_bin_size']` is hard-coded from that rate. The only length manipulation is truncating neural and behaviour arrays to the shorter of the two when they differ by one sample.

ii.
```python
'time_bin_size': 1000.0 / 15.5078125,  # ms (1/imaging_rate * 1000)
'imaging_rate_hz': 15.5078125,
```
```python
min_len = min(len(pos), deconv_all.shape[0])
pos = pos[:min_len]
...
deconv_all = deconv_all[:min_len]
```

iii. CONVERSION_NOTES.md: "Time Bin Size — 64.48 ms (1/15.5078125 Hz); Matches the 2-photon imaging frame rate." The agent verified the rate from the NWB metadata and the paper ("Imaging rate 15.5 Hz" vs "paper: ~15.5 Hz" in its sanity-check table) and used the same figure to convert sample counts to seconds when checking trial durations.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Not from the stored behaviour timestamps. It is synthesised from the sample index within the trial multiplied by a nominal frame period, where the frame period is `1 / imaging_rate` and `imaging_rate` is read per session from `acquisition/TwoPhotonSeries/imaging_plane/imaging_rate`. That attribute is the *scanner* rate: 15.5078125 Hz for the single-plane animals, but 31.015625 Hz for the two-plane animals m17 and m18, whose true per-plane sample period is 64.48 ms.

ii.
```python
imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
...
frame_time = 1.0 / imaging_rate  # seconds per frame
...
time_from_start = np.arange(n_timepoints) * frame_time
input_data[0, :] = time_from_start
```

iii. CONVERSION_NOTES.md: "time_from_trial_start: Time in seconds from trial start (0 at trial onset)", with the global claim "Imaging rate 15.5 Hz (64.48 ms time bins)". The trajectory shows the agent reading `imaging_rate` from this path in its very first exploration of an m11 (single-plane) file, where it equals 15.5078125, and re-using it without re-checking it on the two-plane animals — even though the methods text it had read states "frames bidirectionally imaged at ~31 Hz interleaved in the scan for a sampling rate of ~15.5 Hz per plane". No justification is given for preferring the nominal rate over the stored `timestamps` array that it loads anyway.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. `np.arange(n_timepoints) * frame_time` — a linear ramp starting at exactly 0 in every trial, written into row 0 of the (4, n_timepoints) input array. Because `frame_time` comes from the scanner rate, the ramp advances at 32.24 ms/sample instead of 64.48 ms/sample for the 28 sessions of m17 and m18, so their trial clocks run twice as fast as real time (verified by re-running `convert_session` on `sub-m17_ses-01`: `input[0]` steps by 0.0322 s, vs 0.0645 s for `sub-m3_ses-01`). The values are also inconsistent with the `time_bin_size` the same script writes into metadata for those sessions.

ii.
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
...
time_from_start = np.arange(n_timepoints) * frame_time
input_data = np.zeros((4, n_timepoints), dtype=np.float32)
input_data[0, :] = time_from_start
```

iii. No explicit justification. The agent did sanity-check the maximum value ("max trial time 216.5 s seems high") and converted trial lengths to seconds using the 64.48 ms figure, but that check was run on the pooled data and is dominated by single-plane sessions, so the two-plane error was not surfaced.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the ramp has exactly `n_timepoints = e - s` entries, the same length as the trial's neural matrix, and starts at 0 on the same sample as the alignment event, so element *t* of the time input corresponds to column *t* of the neural matrix. No interpolation or resampling of behaviour onto neural time is needed, because both streams are already on the same sample grid (both are truncated to `min_len` first).

ii.
```python
min_len = min(len(pos), deconv_all.shape[0])
...
n_timepoints = e - s
neural = deconv_filtered[s:e, :].T.astype(np.float32)
time_from_start = np.arange(n_timepoints) * frame_time
input_data = np.zeros((4, n_timepoints), dtype=np.float32)
input_data[0, :] = time_from_start
```

iii. The sub-agent report the agent relied on states that all behavioural series share identical timestamps and the ophys arrays match them in length; the agent confirmed this empirically when it hit the off-by-one in m18 ("The neural data has one more timepoint than behavioral data. Let me fix this by truncating to the minimum length") and moved that truncation ahead of all downstream slicing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioural time series (`processing/behavior/BehavioralTimeSeries/environment/data`), which takes the values -1 (inter-trial), 0 (ENV1) and 1 (ENV2).

ii.
```python
env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
...
env_vals = env[s:e]
env_valid = env_vals[env_vals >= 0]
```

iii. CONVERSION_NOTES.md: "Environment Type — Directly from NWB `environment` field: 0 = ENV1, 1 = ENV2. Verified for m17/m18 which start in ENV2." The trajectory shows the verification: the agent checked `m17 ses-01` (env = 1, and `sessions_dict` says m17 starts in ENV2), `m17 ses-08` (the switch day, env = {0,1}), `m17 ses-09` (env = 1 only) and `m3 ses-01` (env = 0), concluding "env field directly encodes: 0=ENV1, 1=ENV2 (not first/second)".

## 4-b. What processing is involved in computing `input` *Environment type*?

i. One value per trial: the median of the in-trial `environment` samples after discarding the -1 placeholder, cast to int, defaulting to 0 if a trial has no valid samples. That scalar is broadcast across all timepoints of the trial into row 1 of the input array. (Empirically the raw series is already constant within every trial — no trial contains a -1 sample and no trial mixes 0 and 1 — so the median is a no-op relative to taking the raw per-sample values.)

ii.
```python
env_per_trial = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    env_vals = env[s:e]
    env_valid = env_vals[env_vals >= 0]
    if len(env_valid) > 0:
        env_per_trial[i] = int(np.median(env_valid))
    else:
        env_per_trial[i] = 0
...
env_type = float(env_per_trial[i])
input_data[1, :] = env_type
```

iii. From the step-35 reasoning: "The environment field in the NWB data takes values of -1 for inter-trial periods, 0 for ENV1, and 1 for ENV2, and for each trial I'm taking the majority value excluding the inter-trial marker." The resulting balance (ENV1 51.0%, ENV2 49.0%) was reported as a sanity check.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. No raw variable: it is the loop counter over the trial list within a session (which itself derives from the `trial_start`/`teleport` boundaries), converted to a 1-based index. The NWB `trial number` series is loaded nowhere in the conversion.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = float(i + 1)
    input_data[2, :] = trial_number
```

iii. CONVERSION_NOTES.md: "trial_number: 1-indexed trial number within session, constant within trial". The agent had inspected the NWB `trial number` field during exploration but built trial structure from `trial_start`/`teleport` instead, so the sequential counter is the natural index. The resulting range in the converted data is 1-100.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond `i + 1` and broadcasting the constant across all timepoints of the trial. No normalisation, no session-length scaling, no continuation of numbering across sessions.

ii.
```python
trial_number = float(i + 1)
input_data = np.zeros((4, n_timepoints), dtype=np.float32)
input_data[2, :] = trial_number
```

iii. As above — the instruction asks for "Trial number (continuous, per trial)" and the agent supplied the within-session ordinal, broadcast to a time series so that all four inputs share the (4, n_timepoints) shape.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The sparse `Reward` behavioural series — specifically its `timestamps` (the `data` values are all the same constant reward amount and are not used for the outcome). Reward times are compared against the trial's start/end wall-clock times taken from the behaviour `timestamps` array, giving a per-trial binary outcome; input 3 of trial *i* is the outcome of trial *i-1*.

ii.
```python
reward_data = f['processing/behavior/BehavioralTimeSeries/Reward/data'][:]
reward_ts   = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
def determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds):
    for i in range(n_trials):
        t_start = timestamps[tstart_inds[i]]
        t_end = timestamps[teleport_inds[i]]
        n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
        outcomes[i] = 1 if n_rewards > 0 else 0
    return outcomes
```

iii. The agent's exploration (step 40) printed the `Reward` series and confirmed "Reward data: all equal? True", that there are fewer reward events than trials, and that matching `reward_ts` to `[t_start, t_end]` windows gives 0 or 1 reward per trial. The sub-agent report described the series as "sparse event-based ... actual reward delivery times". CONVERSION_NOTES.md: "Reward Outcome — Determined by matching sparse reward event timestamps to trial time windows. Omission rate: 15.3% (paper: ~15%)".

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `previous_trial_outcome[i] = reward_outcomes[i-1]`, with 0 for the first trial of each session; broadcast constant across the trial's timepoints. Outcomes never carry over between sessions.

ii.
```python
if i == 0:
    prev_outcome = 0.0  # No previous trial
else:
    prev_outcome = float(reward_outcomes[i - 1])
input_data[3, :] = prev_outcome
```

iii. CONVERSION_NOTES.md: "previous_trial_outcome: Binary (0=omission, 1=rewarded), constant within trial; first trial of session = 0" — a direct reading of the decoder-input specification in the instructions.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` series plus a per-trial reward-zone identity inferred from the `reward_zone` series. For each trial the samples with `reward_zone > 0` are found and the *median* position over those samples is matched to one of the three known zones A = [80,130], B = [200,250], C = [320,370] with a ±20 cm tolerance. Trials in which the animal never triggers `reward_zone` (1,822 of 12,216, i.e. 15%, essentially the omission trials) get no label in this pass and inherit one by forward fill, then backward fill, from neighbouring trials. The distance is then computed against the fixed boundaries of the assigned zone, not against anything measured in that trial.

ii.
```python
REWARD_ZONES = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}

def get_reward_zone_label(position_when_in_rzone):
    median_pos = np.median(position_when_in_rzone)
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 <= median_pos <= end + 20:
            return label
    return None

def determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds):
    for i in range(n_trials):
        s, e = tstart_inds[i], teleport_inds[i]
        rz_mask = rzone[s:e] > 0
        if np.any(rz_mask):
            zone_labels[i] = get_reward_zone_label(pos[s:e][rz_mask])
    # forward fill
    last_label = None
    for i in range(n_trials):
        if zone_labels[i] is not None: last_label = zone_labels[i]
        elif last_label is not None:   zone_labels[i] = last_label
    # backward fill
    ...
    return zone_labels
```

iii. CONVERSION_NOTES.md: "Reward zone location determined from position data when `reward_zone` field > 0. Zone A: 80-130 cm, Zone B: 200-250 cm, Zone C: 320-370 cm. For omission trials (no rzone activation), zone inherited from neighboring trials via forward/backward fill." The zone coordinates come from the reference repo's `behavior.py` (the script comments them as "Zone X/Y/Z in code"). The agent verified the scheme empirically at step 39: "the reward zone detection works. Trial 30+ switches from C (320-370) to A (80-130) as expected for the C_to_A switch on day 3", and validated the final marginal distribution (A 32.8%, B 33.7%, C 33.5%).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance in cm to the nearest edge of the trial's reward zone: negative before the zone (`position - zone_start`), exactly 0 anywhere inside the zone, positive after it (`position - zone_end`). Computed per sample on the raw (unclipped) position, then discretised (7-c). Row 0 of the output array.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position, dtype=float)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    distance[before] = position[before] - rz_start
    distance[inside] = 0
    distance[after] = position[after] - rz_end
    return distance
...
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
dist_disc = discretize_distance(dist)
output_data[0, :] = dist_disc
```

iii. CONVERSION_NOTES.md: "distance_to_reward_zone (7 classes, time-varying): Signed distance to nearest edge of reward zone". This is the paper's "distance relative to reward" coordinate, and the sign convention plus the "0 = in zone" class are read directly from the instruction's bin table (the step-35 reasoning spells out "if the position is before the zone starts, the distance is negative; if it's past the zone, the distance is positive; and if it's within the zone, the distance is zero").

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned with explicit boolean masks: `< -50` → 0, `[-50, -10)` → 1, `[-10, 0)` → 2, `== 0` → 3, `(0, 10]` → 4, `(10, 50]` → 5, `> 50` → 6. Ties at exactly ±10 and ±50 fall in the inner bin on the positive side and in the outer bin on the negative side, a negligible difference from the expert's `np.digitize` edges. The realised class fractions (0.253 / 0.102 / 0.074 / 0.237 / 0.021 / 0.072 / 0.242) match the expert's to within 0.001.

ii.
```python
def discretize_distance(distance):
    bins = np.zeros_like(distance, dtype=int)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```
```python
'output_values': [
    ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '0 to +10cm', '+10 to +50cm', '> +50cm'],
    ...
```

iii. The docstring reproduces the instruction's bin table verbatim, and the agent transcribed it into `output_values`. Its reasoning at step 35 notes "The spec then bins these distances into categories ranging from less than -50 cm up through greater than +50 cm, with a specific bin for exactly 0 cm."

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same-index slicing: the position samples used are `pos[s:e]`, exactly the samples whose neural columns form the trial, so output column *t* corresponds to neural column *t*. Both arrays were truncated to the common `min_len` before slicing, and the reward-zone identity is a per-trial constant so it needs no alignment.

ii.
```python
min_len = min(len(pos), deconv_all.shape[0])
...
s, e = tstart_inds[i], teleport_inds[i]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
trial_pos = pos[s:e]
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
output_data = np.zeros((6, n_timepoints), dtype=np.int64)
output_data[0, :] = dist_disc
```

iii. Same reasoning as 2-d/3-c: the behaviour and ophys streams are sampled on the same clock (confirmed by the sub-agent's report that all behavioural series share timestamps, and by the agent's own handling of the one-sample ophys overhang).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural series (cm along the 450 cm virtual corridor), same array as 7-a.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = pos[s:e]
```

iii. Implicit — `position` is the only positional variable in the file, and the agent used it throughout its exploration (printing per-trial position ranges, checking the correspondence between `reward_zone > 0` and position).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The raw per-sample position is first clipped to [0, 450] cm and then binned into 5 equal 90 cm bins by integer division, with the bin index clipped to [0, 4]. Clipping means the handful of samples marginally outside the track (and the exact 450 cm endpoint) land in the first/last bin rather than out of range — the same outcome as the expert's open-ended end bins.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped, n_bins=5)
...
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return bins
```

iii. CONVERSION_NOTES.md: "absolute_position (5 classes, time-varying): Position discretized into 5 equal 90cm bins (0-90, 90-180, 180-270, 270-360, 360-450)", with `TRACK_LENGTH = 450` taken from the paper's task description. The code comment "Clip position to valid range" is the only justification given for clipping.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five classes of 90 cm each: 0: <90, 1: 90-180, 2: 180-270, 3: 270-360, 4: ≥360, produced by `floor(position/90)` with clipping. Realised fractions 0.211 / 0.178 / 0.231 / 0.227 / 0.154, matching the expert's to within 0.001.

ii.
```python
bin_size = TRACK_LENGTH / n_bins          # 450/5 = 90
bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
...
['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
```

iii. Directly from the instruction's specification of "5 equal-sized bins spanning the 450 cm track"; the agent parameterised it as `TRACK_LENGTH / n_bins` rather than hard-coding the edges.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same-index slicing (`pos[s:e]`), identical to 7-d; no resampling.

ii.
```python
trial_pos = pos[s:e]
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
output_data[1, :] = discretize_position(pos_clipped, n_bins=5)
```

iii. Same as 7-d.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural series (a per-frame lick count, not a 0/1 flag), plus the trial boundaries used by the sensor-error correction.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
...
lick_corrected, error_trials = correct_lick_sensor_error(
    lick, tstart_inds, teleport_inds, correction_thr=LICK_CORRECTION_THR)
lick_binary = (lick_corrected > 0).astype(float)
```

iii. The agent read the reference repo's `behavior.py` (trajectory step 21), which is where both the lick field's semantics (counts, can exceed 1) and the `correct_lick_sensor_error` routine come from.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps. (1) The paper's lick-sensor-error correction, ported from `behavior.py`: for each trial, if more than 35% of samples have a lick count > 2, the sensor is deemed stuck and the whole trial's lick trace is overwritten — with 0 rather than the reference code's NaN, so the trial is kept in the dataset but relabelled "no lick" throughout. This fires on 63 of 12,216 trials (0.52%; the paper reports 81/12,376). (2) Binarisation: any remaining positive count becomes 1. The overall lick rate is 0.220 of samples (expert reference: 0.230).

ii.
```python
def correct_lick_sensor_error(licks, tstart_inds, teleport_inds, correction_thr=0.35):
    licks_corrected = np.copy(licks)
    error_trials = []
    for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):
        trial_licks = licks_corrected[s:e]
        if len(trial_licks) > 0:
            frac_high = np.sum(trial_licks > 2) / len(trial_licks)
            if frac_high > correction_thr:
                licks_corrected[s:e] = 0  # Set to 0 instead of NaN for binary
                error_trials.append(i)
    return licks_corrected, error_trials
...
lick_binary = (lick_corrected > 0).astype(float)
...
output_data[3, :] = trial_lick.astype(int)
```

iii. CONVERSION_NOTES.md: "Following the reference code (`behavior.py:correct_lick_sensor_error`): Trials where >35% of samples have cumulative lick count >2 are flagged as sensor errors; Lick data on these trials set to 0. Paper reports ~0.65% of all imaged trials affected (81/12,376)", and "Key Decisions" 5: "Lick data binarized (0/1) after sensor error correction". The threshold 0.35 is the value the reference repo passes in `lick_pos_std`. The inline comment gives the reason for 0 instead of NaN: the output channel must be a categorical 0/1.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same-index slicing of the (already corrected and binarised) session-length lick vector; no shifting or smoothing, so a lick appears in the same time bin as the neural sample in which it was recorded.

ii.
```python
lick_binary = (lick_corrected > 0).astype(float)
...
trial_lick = lick_binary[s:e]
...
lick_disc = trial_lick.astype(int)
output_data[3, :] = lick_disc
```

iii. Same as 7-d/8-d: the lick series is on the behaviour sample grid, which is the neural sample grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Identical to 7-a: `reward_zone` (to find in-zone samples) and `position` (to identify which of A/B/C those samples correspond to), with forward/backward fill across trials that never triggered the zone.

ii.
```python
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
...
rz_label = zone_labels[i]
if rz_label is None:
    rz_label = 'A'  # fallback
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. See 7-a. The agent additionally used the marginal distribution as its check ("Reward zone distribution: A:34.3%, B:32.8%, C:32.9%" in its notes; 0.328/0.337/0.335 by timepoint in the final verification).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The per-trial letter label is mapped to the integer code A→0, B→1, C→2 and broadcast across all timepoints of the trial (row 4). A trial whose label is still `None` after both fills would default to 'A'; this never happens in practice, since the median rule labels every one of the 10,394 trials that trigger `reward_zone` and every session contains such trials.

ii.
```python
rz_label = zone_labels[i]
if rz_label is None:
    rz_label = 'A'  # fallback
rz_start, rz_end = REWARD_ZONES[rz_label]
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
output_data[4, :] = rz_loc
...
'output_values': [..., ['zone_A', 'zone_B', 'zone_C'], ...]
```

iii. CONVERSION_NOTES.md: "reward_zone_location (3 classes, per-trial): 0=Zone A, 1=Zone B, 2=Zone C", matching the instruction's "0 = A, 1 = B, 2 = C". Broadcasting per-trial values over time follows the instruction's preference for time-varying outputs where possible.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` series' timestamps, exactly as in 6-a (the same `reward_outcomes` array feeds both input 3 and output 5).

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. See 6-a; the agent verified event-to-trial matching by printing the reward count per trial for the first ten trials of a session.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, count reward events whose timestamp lies in the closed interval between the behaviour timestamps of the trial's start and teleport samples; the output is 1 if that count is non-zero, else 0, broadcast across the trial's timepoints. No tolerance check or assertion is made on how well reward times line up with the sample grid. Resulting omission rate 15.7% by timepoint (the agent reports 15.3% by trial), against the paper's ~15%.

ii.
```python
def determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds):
    outcomes = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        t_start = timestamps[tstart_inds[i]]
        t_end = timestamps[teleport_inds[i]]
        n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
        outcomes[i] = 1 if n_rewards > 0 else 0
    return outcomes
...
rew_out = int(reward_outcomes[i])
output_data[5, :] = rew_out
```

iii. CONVERSION_NOTES.md: "reward_outcome (2 classes, per-trial): 0=no reward (omission), 1=reward delivered ... Omission rate: 15.3% (paper: ~15%)". The paper states reward was "randomly omitted on ~15% of trials", and the agent used that number as its validation of the matching procedure.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive behaviours, all silent-fallback rather than assert-and-fail:
- **Neural/behaviour length mismatch** (ophys one sample longer than behaviour in 10 of the m17/m18 sessions): every array is truncated to `min(len(pos), n_neural_samples)` before any other processing.
- **Unpaired trial boundaries**: start/teleport lists truncated to a common length and pairs with `teleport <= start` dropped.
- **Degenerate trials/sessions**: trials with fewer than 2 samples are skipped; a session with fewer than 2 trials or fewer than 2 surviving cells is dropped entirely. (None of these fire on this dataset.)
- **Missing reward-zone activation** (1,822 trials): forward/backward fill from neighbouring trials, with a hard-coded 'A' fallback if a session never activates the zone.
- **Missing environment samples**: `-1` samples excluded from the per-trial median, defaulting to 0 if none remain.
- **Lick-sensor errors**: whole trial's lick channel zeroed (see 9-b).
- **Cell-count consistency**: an `assert` that plane0 + plane1 cell counts equal the `iscell` length is the only hard check in the script; there is no check that behaviour series share timestamps, and none that reward event times fall within half a bin of the sample grid.

ii.
```python
min_len = min(len(pos), deconv_all.shape[0])
pos = pos[:min_len]; speed = speed[:min_len]; ...; deconv_all = deconv_all[:min_len]
```
```python
assert n_p0 + n_p1 == n_total_cells, \
    f"Cell count mismatch: {n_p0}+{n_p1} != {n_total_cells}"
```
```python
if n_trials < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {n_trials} valid trials")
    return None
if n_kept < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {n_kept} cells after filtering")
    return None
if n_timepoints < 2:
    continue
```
```python
if rz_label is None:
    rz_label = 'A'  # fallback
```

iii. The length-mismatch handling was developed reactively: the pilot run crashed on m18, the agent diagnosed it ("There's a shape mismatch for multi-plane data ... Off by one. The neural data has one more timepoint than behavioral data") and then moved the truncation to after the plane concatenation so that all streams are cut consistently. The other guards are not discussed in the trajectory; CONVERSION_NOTES.md only documents the neighbour-fill rule for omission trials.

## 13-a. What are the most time-consuming steps of the code?

i. The agent never analysed this; from the code the dominant costs are:
1. **HDF5 reads** — for every session the script reads `Deconvolved`, `Fluorescence` *and* `Neuropil` in full (three ~T×n_ROI float32 arrays, up to ~5,000 ROIs × 30,000 samples for the two-plane sessions), plus all behaviour series.
2. **`compute_dff_simple`** — per trial, a 15-sample boxcar plus a 300-sample minimum and a 300-sample maximum filter over *all* 312,110 ROIs in the dataset, of which only 138,678 are ever used.
3. **`detect_interneurons`** — a Python loop calling `scipy.stats.pearsonr` once per ROI (312,110 calls overall).
4. **Serialisation** — pickling a 9.2 GB dictionary, plus `copy.deepcopy` of that entire dictionary inside `create_sample`.
Unlike the expert solution, the script does *not* pay for a second full pass over the files (no survey stage), and it does not run OASIS deconvolution at all.

ii.
```python
deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
fluor_p0  = f['processing/ophys/Fluorescence/plane0/data'][:]
neuro_p0  = f['processing/ophys/Neuropil/plane0/data'][:]
```
```python
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)   # all ROIs
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds, ...)
```
```python
sample = copy.deepcopy(data)        # deep-copies the whole ~9 GB dataset
...
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. Not documented. The only performance-related remarks in the trajectory concern wall-clock of the *decoder* run (the agent backgrounded `train_decoder.py` and polled it) and the size of the output file (9.2 GB), not the conversion itself.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three candidates:
- `detect_interneurons`'s per-cell `pearsonr` loop — a single centred matrix product (or `np.corrcoef` on the masked matrix) computes all ROI-speed correlations at once; this is the clearest win, and the p-value `pearsonr` returns is discarded.
- The per-trial loop in `compute_dff_simple` — unavoidable in principle (baselines are per trial), but the boxcar/min/max filters could be applied once to the whole session with the ITI samples masked, or at least restricted to curated ROIs.
- The per-trial conversion loop in `convert_session` — `compute_distance_to_reward_zone`, `discretize_position`, `discretize_speed` and the lick binarisation are all elementwise and could be computed once on the full session vectors and then sliced, as could the `env_per_trial` and `determine_reward_zone_per_trial` passes (which loop over trials three separate times). Variable trial length makes the final packing into per-trial arrays inherently loop-shaped, so the gain is modest.

ii.
```python
for cell in range(n_cells):
    dff_valid = dff[valid_mask, cell]
    if np.std(dff_valid) < 1e-10 or np.std(speed_valid) < 1e-10:
        continue
    r, _ = stats.pearsonr(dff_valid, speed_valid)
```
```python
for i in range(n_trials):
    ...
    dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
    dist_disc = discretize_distance(dist)
    pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
    pos_disc = discretize_position(pos_clipped, n_bins=5)
    speed_disc = discretize_speed(np.abs(trial_speed))
```

iii. Not documented; the agent made no efficiency claims about its loops.

## 13-c. What processing does the code repeat multiple times?

i. Little repeated *I/O*: each NWB file is opened exactly once and each array read once — the expert solution reads every file twice (a survey pass and a conversion pass). What is repeated within a session is iteration: the trial list is walked four separate times (dF/F baseline, interneuron valid-mask, lick correction, reward-zone labelling, environment median, and finally the main conversion loop), and `zip(tstart_inds, teleport_inds)` masks are rebuilt in each of them. `speed` is also slice-indexed twice (once for the interneuron correlation, once per trial for the output). The whole converted dataset is additionally traversed twice after conversion, once by `create_sample`'s deep copy and once by `print_sanity_checks`.

ii.
```python
for s, e in zip(tstart_inds, teleport_inds):      # in compute_dff_simple
for s, e in zip(tstart_inds, teleport_inds):      # in detect_interneurons (valid_mask)
for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):   # in correct_lick_sensor_error
for i in range(n_trials):                          # in determine_reward_zone_per_trial
for i in range(n_trials):                          # env_per_trial
for i in range(n_trials):                          # main conversion loop
```

iii. Not documented.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three things are computed and thrown away:
- **dF/F for every ROI** — `compute_dff_simple` runs on all 312,110 ROIs although only the 138,678 curated ones can ever matter, and the resulting dF/F traces are used solely for one scalar correlation per cell and then discarded (the neural output uses the stored `Deconvolved` array instead). Restricting the computation to `curated_mask` first would halve it with identical results.
- **`Fluorescence`/`Neuropil` loading** — those two full arrays (the larger part of the I/O) exist in the pipeline only to serve that discarded dF/F.
- **Side products** — `reward_data` and `planeIdx` are read and never used; `pearsonr`'s p-value is discarded; `convert_session` returns `zone_labels`, `env_per_trial` and `lick_error_trials` that are used only for a progress message; `create_sample` deep-copies the complete ~9 GB dictionary before throwing away all but 5 sessions × 20 trials; and `sample_data.pkl`, `README.md`, `predictions.png` and `sample_trials.png` are extra artefacts beyond the requested deliverables.
None of this changes the converted data.

ii.
```python
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)  # all ROIs, then discarded
...
reward_data = f['processing/behavior/BehavioralTimeSeries/Reward/data'][:]  # never used
planeIdx = f['processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx'][:]  # never used
...
def create_sample(data, max_sessions=5, max_trials=20):
    import copy
    sample = copy.deepcopy(data)
```

iii. Not documented. The dF/F-on-all-ROIs case follows from the agent's decision (2-a) to take the neural signal from the NWB `Deconvolved` array while still needing a dF/F proxy for the paper's interneuron filter; the sample dataset was created deliberately, as the agent used it to iterate on the decoder quickly before the full run.
