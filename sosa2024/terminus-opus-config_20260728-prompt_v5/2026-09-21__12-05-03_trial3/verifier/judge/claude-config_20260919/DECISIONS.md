# Decisions

All code snippets below are from the AI's `/app/convert_data.py`. Justifications are drawn from
`/app/CONVERSION_NOTES.md`, `/app/README.md`, and the agent trajectory (`/logs/agent/trajectory.json`).

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks `/app/data`, treats every `sub-*` sub-directory as a subject, globs every `*.nwb`
file inside each subject directory as a session, and parses the session number out of the file
name (`sub-<id>_ses-<NN>_behavior+ophys.nwb`). Every one of the 152 files is opened and processed;
no file is skipped. Files are read with **`h5py` directly** (not `pynwb`), reading the raw HDF5
paths `processing/ophys/{Fluorescence,Neuropil}/plane*/data`,
`processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`, and
`processing/behavior/BehavioralTimeSeries/*`. Multi-plane sessions (m17, m18) are handled by
sorting the `plane*` keys and concatenating along the ROI axis. Each file is read exactly once
(single pass — there is no separate survey pass).

ii.
```python
data_dir = '/app/data'
subjects = sorted([d.replace('sub-', '') for d in os.listdir(data_dir) if d.startswith('sub-')])

all_sessions = []
for subj in subjects:
    subj_dir = os.path.join(data_dir, f'sub-{subj}')
    nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
    for nwb_file in nwb_files:
        ses_str = os.path.basename(nwb_file).split('ses-')[1].split('_')[0]
        ses_num = int(ses_str)
        all_sessions.append((subj, ses_num, nwb_file))
```
```python
with h5py.File(nwb_path, 'r') as f:
    fluor_group = f['processing']['ophys']['Fluorescence']
    plane_keys = sorted([k for k in fluor_group.keys() if k.startswith('plane')])
    F_planes = []
    Fneu_planes = []
    for pk in plane_keys:
        F_planes.append(f['processing']['ophys']['Fluorescence'][pk]['data'][()].T)
        Fneu_planes.append(f['processing']['ophys']['Neuropil'][pk]['data'][()].T)
    F_raw = np.concatenate(F_planes, axis=0)   # (n_rois, n_timepoints)
    Fneu_raw = np.concatenate(Fneu_planes, axis=0)
    iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][()]

    beh = f['processing']['behavior']['BehavioralTimeSeries']
    position = beh['position']['data'][()]
    speed = beh['speed']['data'][()]
    lick = beh['lick']['data'][()]
    environment = beh['environment']['data'][()]
    ...
    timestamps = beh['position']['timestamps'][()]
    reward_ts = beh['Reward']['timestamps'][()]
```

iii. CONVERSION_NOTES Step 2 documents the directory layout ("11 subjects: m3, m4, m7, m11-m15,
m17-m19"; "m11: 12 sessions (ses-03 to ses-14, missing ses-01/02); All others: 14 sessions";
"Multi-plane: m17 (2 planes), m18 (2 planes)") and Step 9 records that all 152 sessions were
converted, which matches the paper's 11 switch-task mice × 14 days − 2 missing m11 days. The AI
chose h5py over pynwb because it reads only the datasets it needs from the HDF5 file.

## 1-b. How are the data split into subjects?

i. One subject per `sub-*` directory; the id is the directory name with the `sub-` prefix stripped
(`m3`, `m4`, `m7`, `m11`…`m19`). `data['subjects']` is the list of subject ids in first-appearance
(= sorted) order, and `data['subject_idx']` records, for each converted session, the index of the
subject that session belongs to.

ii.
```python
subjects = sorted([d.replace('sub-', '') for d in os.listdir(data_dir) if d.startswith('sub-')])
...
if subj not in subjects_seen:
    subjects_seen.append(subj)
subject_idx_list.append(subjects_seen.index(subj))
...
'subjects': subjects_seen,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 3 cites the paper's "n = 11 mice" for the switch task and Step 9 records
11 subjects in the converted file, matching. The trajectory (step 20) additionally records the
mapping between the released ids and the paper's internal animal names
("GCAMP3->m3, GCAMP4->m4, GCAMP7->m7, GCAMP11->m11, etc.").

## 1-c. How are the data split into sessions?

i. One session per `.nwb` file. The session number is parsed from the file name and is used as the
experiment-day index (`ses-NN` → `exp_day NN`) to look up the session's scene in the hard-coded
`SESSIONS_SCENES` table. Sessions are kept separate; no across-session neuron registration is
attempted.

ii.
```python
ses_str = os.path.basename(nwb_file).split('ses-')[1].split('_')[0]
ses_num = int(ses_str)
...
scene = SESSIONS_SCENES[subject_id][ses_num]
```
```python
session_info_list.append({
    'subject': subj, 'session': ses_num, 'scene': result['scene'],
    'n_cells': result['n_cells'], 'n_trials': len(result['neural']),
})
```

iii. Trajectory step 24: "The NWB session numbers map to exp_day in sessions_dict — For m11,
ses-03 = exp_day 3 (scene: Env1_LocationB_to_A), ses-04 = exp_day 4 (Env1_LocationA)"; step 33:
"m11 is missing ses-01 and ses-02 (GCAMP11 sessions_dict shows exp_day 1 and 2 had no scanning)".
The AI verified this identity by cross-checking the scene names against the reference
`sessions_dict.py` (trajectory steps 51–52, after finding its first hand-copied table was wrong).

## 1-d. How are the data split into trials?

i. A trial runs from a `trial_start` marker to the following `teleport` marker, half-open
`[trial_start, teleport)`. Start indices are the frames where `trial_start == 1`, end indices the
frames where `teleport == 1`. The two lists are truncated to their common length (`min`) rather
than asserted equal.

ii.
```python
tstart_idx = np.where(trial_start_arr == 1)[0]
teleport_idx = np.where(teleport_arr == 1)[0]
n_trials = min(len(tstart_idx), len(teleport_idx))

if n_trials < 2:
    print(f"    Skipping: only {n_trials} trials")
    return None

tstart_idx = tstart_idx[:n_trials]
teleport_idx = teleport_idx[:n_trials]
```
```python
for t in range(n_trials):
    start = tstart_idx[t]
    stop = teleport_idx[t]
    n_tp = stop - start
    ...
    neural = dff_valid[:, start:stop].astype(np.float32)
```

iii. CONVERSION_NOTES Step 3/5: "Trial boundaries: trial_start and teleport markers"; Step 10
Check 3 lists "Trial boundaries | trial_start -> teleport | trial_start_inds -> teleport_inds |
YES" as matching the reference code. The stored `trial number` behavioral variable is loaded but
deliberately not used to cut trials. Step 9 reports 12,216 trials and 80.4 ± 6.1 trials/session
against the paper's "80.5 ± 7.4".

## 1-e. How are trials filtered based on quality controls?

i. Essentially no trial-level quality filtering. The only trial-level rule is that a trial with
fewer than 2 timepoints is dropped. Two session-level guards exist: a session is dropped if it has
fewer than 2 trial-start/teleport pairs, fewer than 2 surviving cells, or fewer than 2 surviving
trials (this guarantees the format requirement of ≥2 trials per session). No speed/stationarity
filter and no exclusion of the paper's ~0.65% "erroneous lick" trials is applied. In practice
nothing is removed: all 152 sessions and all 12,216 trials survive.

ii.
```python
if n_trials < 2:
    print(f"    Skipping: only {n_trials} trials")
    return None
...
if n_cells < 2:
    print(f"    Skipping: only {n_cells} valid cells")
    return None
...
for t in range(n_trials):
    start = tstart_idx[t]
    stop = teleport_idx[t]
    n_tp = stop - start
    if n_tp < 2:
        continue
...
if len(neural_trials) < 2:
    print(f"    Skipping: only {len(neural_trials)} valid trials after filtering")
    return None
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "**No speed filtering**: Unlike the paper's spatial
analyses which exclude <2 cm/s, we include all data within trials for the decoder." Step 3
"Trial curation: Use trial_start to teleport boundaries". Step 10 Check 5 notes the edge cases it
inspected instead of filtering: "Sessions with few trials (m4 ses-04: 41 trials): included, valid
data; Very long trials (>100s): real data from slow-running mouse, included."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces `processing/ophys/Fluorescence/plane*/data` (F) and
`processing/ophys/Neuropil/plane*/data` (Fneu). The NWB `Deconvolved` array is explicitly **not**
used.

ii.
```python
F_planes.append(f['processing']['ophys']['Fluorescence'][pk]['data'][()].T)
Fneu_planes.append(f['processing']['ophys']['Neuropil'][pk]['data'][()].T)
F_raw = np.concatenate(F_planes, axis=0)
Fneu_raw = np.concatenate(Fneu_planes, axis=0)
```

iii. CONVERSION_NOTES Step 1: "The Deconvolved data in NWB is suite2p's default deconvolution of
raw F, NOT the paper's custom dF/F pipeline." Trajectory step 39: "**Deconvolved**: Deconvolved
events (non-negative, 72.4% zeros) — these are suite2p's deconvolved spikes from raw F, NOT from
dF/F. The paper says they compute dF/F from raw F using maximin baseline, then deconvolve."
Step 4 discrepancy table resolution: "Compute dF/F from raw F+Fneu following paper method."

## 2-b. How is the `neural` data processed?

i. The AI reimplements the paper's `preprocessing.dff` up to, but **not including**, the OASIS
deconvolution. Per session (over all ROIs, both planes pooled): subtract `0.7 * Fneu`; then for
each trial window `[trial_start, teleport)` add back `0.7 * mean(Fneu)` of that trial, take a
maximin baseline (Gaussian smoothing with sigma 15 samples along time, then a 300-sample running
minimum followed by a 300-sample running maximum ≈ the Methods' 20 s window), and finally
`dff = (F - baseline)/|baseline|`; then smooth dF/F per trial with a 2-sample Gaussian. Samples
outside trial windows stay NaN. The **final `neural` array is this dF/F**, cast to float32, with
any residual NaN replaced by 0. There is no deconvolution into "events", and the reference's
per-mouse/per-day `keep_teleports` distinction (whether the baseline window may span the teleport)
is not implemented.

ii.
```python
def compute_dff(F, Fneu, trial_start_idx, teleport_idx, neu_coef=0.7):
    f_ = F.astype(np.float64).copy()
    f_neu_ = Fneu.astype(np.float64).copy()
    f_ -= neu_coef * f_neu_
    flow = np.full_like(f_, np.nan)
    dff  = np.full_like(f_, np.nan)

    for i in range(len(trial_start_idx)):
        start = trial_start_idx[i]; stop = teleport_idx[i]
        if stop <= start: continue
        f_[:, start:stop] = f_[:, start:stop] + neu_coef * np.nanmean(
            f_neu_[:, start:stop], axis=1, keepdims=True)
        trial_data = f_[:, start:stop]
        flow[:, start:stop] = nansmooth(trial_data, 15, axis=1)
        flow[:, start:stop] = minimum_filter1d(flow[:, start:stop], 300, axis=-1)
        flow[:, start:stop] = maximum_filter1d(flow[:, start:stop], 300, axis=-1)

    valid = ~np.isnan(flow) & (np.abs(flow) > 0)
    dff[valid] = (f_[valid] - flow[valid]) / np.abs(flow[valid])

    for i in range(len(trial_start_idx)):
        start = trial_start_idx[i]; stop = teleport_idx[i]
        if stop <= start: continue
        dff[:, start:stop] = nansmooth(dff[:, start:stop], 2, axis=1)
    return dff
```
```python
dff = compute_dff(F_raw, Fneu_raw, tstart_idx, teleport_idx, neu_coef=0.7)
...
neural = dff_valid[:, start:stop].astype(np.float32)
neural = np.nan_to_num(neural, nan=0.0)
```

iii. CONVERSION_NOTES Step 5 Key Decision 1: "**Neural signal: dF/F (not deconvolved)**: Computed
dF/F from raw F following paper's method. Did not use suite2p deconvolved data because it doesn't
match the paper's processing pipeline." The metadata field records
`'neural_signal': 'dF/F (neuropil-subtracted, maximin baseline)'`. For the omission of OASIS the
only rationale in the record is trajectory step 42: "The paper specifically says they deconvolve
dF/F, not raw F. I should compute dF/F from scratch and then use that. For the decoder, dF/F
(without deconvolution) might actually work well." Step 59 adds that skipping the paper's pipeline
entirely "would deviate from the paper's method" but does not revisit the missing deconvolution.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both taken from the paper. (1) ROIs are restricted to suite2p's manually curated
cells, `iscell[:,0] == 1`. (2) Putative interneurons are then dropped: any cell whose dF/F has a
Pearson correlation with running speed greater than 0.5 (computed over the non-NaN, i.e. in-trial,
samples). Cells are pooled across planes after filtering. Reported exclusion rate ≈ 0.4%.

ii.
```python
cell_mask = iscell[:, 0] == 1
...
dff_cells = dff[cell_mask]
is_interneuron = detect_interneurons(dff_cells, speed, threshold=0.5)
cell_indices = np.where(cell_mask)[0]
valid_cell_indices = cell_indices[~is_interneuron]
dff_valid = dff[valid_cell_indices]
n_cells = len(valid_cell_indices)
```
```python
def detect_interneurons(dff, speed, threshold=0.5):
    ...
    for i in range(n_cells):
        valid = ~np.isnan(dff[i]) & speed_valid
        if valid.sum() > 10:
            d = dff[i, valid]; s = speed_clean[valid]
            d_centered = d - d.mean(); s_centered = s - s.mean()
            num = np.dot(d_centered, s_centered)
            denom = np.sqrt(np.dot(d_centered, d_centered) * np.dot(s_centered, s_centered))
            if denom > 0 and num / denom > threshold:
                is_interneuron[i] = True
    return is_interneuron
```

iii. CONVERSION_NOTES Step 3: "Cell curation: iscell (manual) + interneuron exclusion
(corr(dF/F, speed) > 0.5)", sourced to the Methods quote "Pearson correlation of >0.5". Step 10
Check 4 compares the resulting exclusion rate to the paper: "Interneuron excl. | 0.42±0.85% |
~0.4% | YES", and neuron counts "155-2172 (paper) | 154-2323 (converted) | CLOSE".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instructions ask for alignment to the start of the trial, which requires no work beyond
cutting the continuous session arrays at the `trial_start` frames: each trial's neural matrix is
`dff_valid[:, trial_start : teleport]`, so column 0 is the alignment event. No pre-event window is
included (`off_start = 0.0`) and trials run to their natural end (`off_end = None`). Behavioral and
neural streams are sampled on the same frame clock, so the same index range is used for both.

ii.
```python
'temporal_alignment_event': 'start of trial (entry to linear track)',
'off_start': 0.0,   # trial starts at alignment event
'off_end': None,    # variable trial duration
```
```python
start = tstart_idx[t]
stop  = teleport_idx[t]
neural = dff_valid[:, start:stop].astype(np.float32)
time_from_start = timestamps[start:stop] - timestamps[start]
trial_pos   = position[start:stop]
trial_speed = speed[start:stop]
trial_lick  = lick[start:stop]
```

iii. CONVERSION_NOTES Step 2 documents that behavior and ophys share the frame clock ("all 19818
timepoints" for both behavior and ophys in m11 ses-03), so a common index range aligns them.
Any residual length difference is removed up front by truncating every stream to the common length
(see 12).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling of any kind. Data are kept at the native two-photon frame clock,
~15.5 Hz (64.5 ms per bin), the same rate for all sessions. For the two two-plane animals (m17,
m18) the NWB `rate` attribute is the 31 Hz scanner rate, but each plane's series and the behavior
timestamps are still at 15.5 Hz, so no adjustment was needed. The bin size is written into the
metadata as a hard-coded constant `64.5` ms rather than derived from the file (the true value is
64.4836 ms).

ii.
```python
dt = np.mean(np.diff(timestamps))  # ~0.0645s
```
```python
'time_bin_size': 64.5,  # ms, approximate
...
'frame_rate_hz': 15.5,
```

iii. CONVERSION_NOTES Step 2: "Frame rate | ~15.51 Hz (~64.5 ms/frame)"; Step 3 cites the Methods'
"~15.5 Hz" and "All behavioral and neural time series sampled at ~15.5 Hz imaging frame rate"
(trajectory step 30). Step 9 lists "Frame rate | ~15.5 Hz | ~15.5 Hz | YES". Keeping the native
rate preserves maximum temporal information for the decoder and keeps neural/behavior alignment
trivial.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attached to the `position` behavioral time series (all behavior series in the
file carry the same timestamp vector).

ii.
```python
timestamps = beh['position']['timestamps'][()]
...
time_from_start = timestamps[start:stop] - timestamps[start]
inp[0, :] = time_from_start
```

iii. CONVERSION_NOTES Step 5 mapping table: "timestamps | input[0] | time_from_trial_start =
ts - ts[trial_start] | Continuous, time-varying". The behavior timestamps are the imaging frame
times, so they are the natural clock for the trial.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the timestamp of the trial's first frame from the trial's timestamp vector, so the
value is 0 at the alignment event and increases in seconds. No smoothing, rescaling, or clipping.
Stored as float32.

ii.
```python
inp = np.zeros((4, n_tp), dtype=np.float32)
inp[0, :] = timestamps[start:stop] - timestamps[start]
```

iii. Direct implementation of the requested input, "Time from start of trial in seconds
(continuous, time-varying)". Sanity check in CONVERSION_NOTES Step 10 Check 2: "**Time input**:
Time from trial start at t=10 = 0.6448s matches raw timestamps. PASS." (10 × 64.48 ms).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses exactly the same frame index range `[start, stop)` as the neural slice, so alignment is
automatic. Before slicing, every neural and behavioral stream is truncated to the common length
`min(n_timepoints_neural, n_timepoints_beh)` so that index `i` means the same frame in all streams.
Unlike the reference, the AI does not assert that the per-series timestamp vectors agree.

ii.
```python
n_timepoints = min(n_timepoints_neural, n_timepoints_beh)
if n_timepoints_neural != n_timepoints_beh:
    F_raw = F_raw[:, :n_timepoints]; Fneu_raw = Fneu_raw[:, :n_timepoints]
    position = position[:n_timepoints]; speed = speed[:n_timepoints]
    ...
    timestamps = timestamps[:n_timepoints]
```
```python
neural = dff_valid[:, start:stop]
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. CONVERSION_NOTES Step 4: "Neural/beh mismatch | Some sessions off-by-1 | Truncate to minimum
length"; Step 10 Check 5: "Off-by-one between neural and behavioral data: handled by truncation".
Step 10 Check 2 sanity check 1 verifies a trial's timepoint count against the raw trial boundaries.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioral time series (0 = ENV1, 1 = ENV2; a −1 code appears between trials).

ii.
```python
environment = beh['environment']['data'][()]
```

iii. CONVERSION_NOTES Step 5 mapping table: "environment | input[1] | 0=ENV1, 1=ENV2 | Binary,
per-trial". Trajectory step 22 notes the reference code's `env_morph_dict`: "Env1=0, Env2=1,
Env3=0.5", confirming the encoding.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, the in-trial `environment` samples are taken, values < 0 (the inter-trial "not in VR"
code) are discarded, and the **median** of the remainder is cast to int and broadcast as a constant
across all timepoints of the trial. Trials with no valid sample default to 0.

ii.
```python
env_per_trial = np.zeros(n_trials, dtype=int)
for t in range(n_trials):
    start = tstart_idx[t]; stop = teleport_idx[t]
    env_vals = environment[start:stop]
    env_vals = env_vals[env_vals >= 0]  # exclude -1
    if len(env_vals) > 0:
        env_per_trial[t] = int(np.median(env_vals))
...
inp[1, :] = env_per_trial[t]
```

iii. The instructions specify environment as "binary, ENV1 vs ENV2, per trial", so a single value
per trial is required; the median over valid in-trial samples is used as a robust per-trial
summary, and the explicit `>= 0` mask guards against the −1 sentinel the AI saw in the raw arrays.
CONVERSION_NOTES Step 3 records that the environment identity changes only on the switch day
(day 8), which the per-trial reduction reproduces. Verification output confirms
`environment_type: [0.0, 1.0]`.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Not from any raw variable — it is the position of the trial in the session's ordered list of
`trial_start`/`teleport` pairs, i.e. the loop counter. The NWB `trial number` time series is loaded
but never used for this.

ii.
```python
for t in range(n_trials):
    ...
    inp[2, :] = t  # trial number (0-indexed)
```

iii. CONVERSION_NOTES Step 5 mapping: "trial_number | input[2] | 0-indexed trial number |
Continuous, per-trial". The trial index derived from the `trial_start`/`teleport` markers is the
same quantity used everywhere else in the conversion (reward-zone switch at trial 30, previous
outcome), so using the loop index keeps all per-trial variables on one consistent indexing.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond assigning the 0-based within-session index and broadcasting it constant across the
trial's timepoints. It is not normalized and it restarts at 0 in every session.

ii.
```python
inp[2, :] = t
```

iii. The instruction asks for "Trial number (continuous, per trial)". Verification output confirms
the range `trial_number: [0.0, 99.0]`, consistent with the longest sessions having 100 trials.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` behavioral time series' own `timestamps` (the times of reward delivery events),
compared against the trial's start and end times taken from the behavior `timestamps` vector.

ii.
```python
reward_ts = beh['Reward']['timestamps'][()]
...
rewarded_trials = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    t_start_time = timestamps[tstart_idx[t]]
    t_end_time   = timestamps[teleport_idx[t]]
    rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
```

iii. CONVERSION_NOTES Step 5 mapping: "reward (prev trial) | input[3] | Previous trial rewarded?
0/1 | Binary, per-trial". The `Reward` series has its own event timestamps rather than a
frame-aligned data vector, so the AI matched reward times to the trial time window rather than to a
frame index. Trajectory step 44 sanity check: "69 reward deliveries out of 80 trials (~86%,
consistent with ~15% omission rate)".

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial `t > 0` the input takes the boolean `rewarded_trials[t-1]` as a float; for the first
trial of a session it is set to 0 (there is no previous trial). The value is constant across all
timepoints of the trial. Note that "previous" means the previous trial *within the same session* —
it is not carried across session boundaries.

ii.
```python
prev_outcome = float(rewarded_trials[t-1]) if t > 0 else 0.0  # first trial: no previous
inp[3, :] = prev_outcome
```

iii. The instruction specifies "Previous trial outcome (binary, omitted = 0, rewarded = 1, per
trial)"; README documents `previous_trial_outcome: 0=omission, 1=rewarded`. Verification output
confirms the range `[0.0, 1.0]`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Two things: the `position` behavioral time series, and a per-trial reward-zone interval. The
reward-zone interval is **not** read from the data — it is looked up from the experimental design:
a hard-coded `SESSIONS_SCENES` table (copied programmatically out of the reference
`sessions_dict.py`) maps (subject, session/exp-day) to a scene name such as `Env1_LocationB_to_A`,
and `get_reward_zone_for_trial()` (a reimplementation of the reference `get_reward_zones()`) turns
that scene name plus the trial index into a zone label and its coordinates, using
`REWARD_ZONE_DICT = {A:[80,130], B:[200,250], C:[320,370]}` and a switch at trial 30 on
"`_to_`" sessions. The raw `reward_zone` time series is loaded but not used for this.

ii.
```python
REWARD_ZONE_DICT = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}

SESSIONS_SCENES = {
    'm11': {1: 'Env1_LocationB', ..., 3: 'Env1_LocationB_to_A', ...},
    ...
}

def get_reward_zone_for_trial(scene, trial_idx, n_trials, change_trial=30):
    if 'LocationA' in scene and '_to_' not in scene:
        return REWARD_ZONE_DICT['A'], 'A'
    ...
    elif '_to_' in scene or '_to_Env' in scene:
        parts = scene.split('_to_')
        pre_part, post_part = parts[0], parts[1]
        ...
        if trial_idx < change_trial:
            return REWARD_ZONE_DICT[pre_zone], pre_zone
        else:
            return REWARD_ZONE_DICT[post_zone], post_zone
```
```python
scene = SESSIONS_SCENES[subject_id][ses_num]
...
for t in range(n_trials):
    coords, label = get_reward_zone_for_trial(scene, t, n_trials, change_trial=30)
    rz_labels.append(label); rz_coords.append(coords)
```

iii. CONVERSION_NOTES Step 5 Key Decision 2: "**Reward zone from scene names**: Used
sessions_dict.py scene names to determine reward zone per trial, with change_trial=30 for switch
sessions." Step 3 sources the numbers to the paper: "Zone A 80–130 cm / Zone B 200–250 cm / Zone C
320–370 cm" and "Switch trial | 30 | 'Each switch occurred after 30 trials'". The AI first
hand-copied the table, found mismatches, and re-extracted it programmatically from the reference
`sessions_dict.py` (trajectory steps 51–52). Trajectory step 44 cross-checks the result against the
data: "m11 ses-05 (exp_day 5): Switch from A to C at trial 30, matching sessions_dict
(Env1_LocationA_to_C); Trials 0-29: zone A (80-130 cm); Trials 30+: zone C (320-370 cm)". The AI
rejected inferring the zone from the raw `reward_zone` counter because "The reward_zone variable in
the NWB is cumulative (like lick), not a zone label" (trajectory step 23).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For every timepoint, the signed distance from the animal's position to the nearest edge of that
trial's reward zone: negative before the zone (`pos - rz_start`), positive past it
(`pos - rz_end`), and exactly 0 while inside the zone. Computed vectorized over the trial with
nested `np.where`. No smoothing or clipping; the continuous value is then discretized (7-c).

ii.
```python
rz_start_cm, rz_end_cm = rz_coords[t]
dist_to_rz = np.where(trial_pos < rz_start_cm, trial_pos - rz_start_cm,
                      np.where(trial_pos > rz_end_cm, trial_pos - rz_end_cm, 0.0))
```

iii. CONVERSION_NOTES Step 5 mapping: "position, rz_coords | output[0] |
distance_to_reward_zone, 7 bins | Time-varying". Trajectory step 46: "Distance to reward zone works
correctly: negative means before zone, positive means after zone, 0 means in zone." This matches
the instruction's "Distance to any location in the reward zone", which is 0 anywhere inside the
50 cm zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned by explicit boolean masks that transcribe the instruction's bin edges
literally, with exactly 0 cm given its own class (bin 3). The array is initialized to −1 so any
unassigned (e.g. NaN) value would be visible; in the converted data no −1 remains.

ii.
```python
def discretize_distance_to_rz(distance):
    bins = np.full_like(distance, -1, dtype=np.int64)
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
    ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', '0 cm', '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
    ...
```

iii. The bin edges are copied directly from the Decoder Task specification. The separate `== 0`
class exists because the specification asks for "3: 0 cm" as its own category, which is exactly the
in-zone condition produced by 7-b. Resulting distribution (verification output):
`{< -50 (0.253), -50 to -10 (0.102), -10 to 0 (0.074), 0 cm (0.237), >0 to +10 (0.021),
+10 to +50 (0.072), > +50 (0.242)}`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the same `[start, stop)` frame index range as the neural data, after the
session-level truncation to the common length, so no further alignment is required.

ii.
```python
trial_pos = position[start:stop]
...
neural = dff_valid[:, start:stop].astype(np.float32)
```

iii. Behavior and ophys share the imaging frame clock (CONVERSION_NOTES Step 2), so identical
indexing guarantees alignment. CONVERSION_NOTES Step 10 Check 2 sanity check 2 verifies a specific
frame: "Raw position 35.3cm at t=10 correctly maps to bin 0 (<90cm). PASS."

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioral time series (VR corridor position in cm), used raw.

ii.
```python
position = beh['position']['data'][()]
...
trial_pos = position[start:stop]
```

iii. CONVERSION_NOTES Step 5 mapping: "position | output[1] | absolute_position, 5 bins of 90cm |
Time-varying". Step 9 checks the range: "Track length | 450 cm | 0-451 cm | YES".

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing the trial and discretizing. No re-referencing, unwrapping, or clipping.
(Trajectory step 46 notes that position runs down to −50 cm during the inter-trial teleport period,
but those samples fall outside `[trial_start, teleport)` and so never enter the converted trials.)

ii.
```python
trial_pos = position[start:stop]
pos_bins = discretize_position(trial_pos)
out_tv[1, :] = pos_bins
```

iii. Position is already in centimetres along the 450 cm track, which is exactly what the
instruction's bin edges assume.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five categories with edges 90/180/270/360 cm — equal 90 cm bins spanning the 450 cm track — with
the first and last bins left open so that the handful of samples marginally outside `[0, 450]` fall
into the end classes rather than being dropped.

ii.
```python
def discretize_position(position):
    bins = np.full_like(position, -1, dtype=np.int64)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins
```

iii. Direct transcription of the Decoder Task specification, "Discretized into 5 equal-sized bins
spanning the 450 cm track". Resulting distribution (verification output):
`{0-90 (0.211), 90-180 (0.178), 180-270 (0.231), 270-360 (0.227), 360+ (0.154)}` — roughly uniform,
as expected for a run down a linear track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[start, stop)` index range as the neural data; no separate alignment step.

ii.
```python
trial_pos = position[start:stop]
neural    = dff_valid[:, start:stop]
```

iii. Same reasoning as 7-d: a single shared frame clock after the common-length truncation.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioral time series, which holds a per-frame lick count (values 0…6 observed).

ii.
```python
lick = beh['lick']['data'][()]
...
trial_lick = lick[start:stop]
```

iii. CONVERSION_NOTES Step 5 mapping: "lick | output[3] | lick binary (>0 = lick) | Time-varying".
Trajectory step 46: "Lick data: cumulative count per frame... 15.5% of scanning frames have licks."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized by thresholding at > 0: any frame with at least one lick becomes 1, otherwise 0.
No smoothing or dilation.

ii.
```python
lick_binary = (trial_lick > 0).astype(np.int64)
out_tv[3, :] = lick_binary
```

iii. CONVERSION_NOTES Step 5 Key Decision 3: "**Lick binarization**: lick > 0 = lick present
(cumulative count per frame)". Trajectory step 46 justifies the threshold choice: "Lick diff shows
values from -5 to 4, meaning the cumulative count can decrease between frames (likely resets
between trials). So lick > 0 is the right approach for binary." The instruction asks for
"Lick, time-varying. 0 = no, 1 = yes". Converted distribution: `{no lick 0.770, lick 0.230}`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start, stop)` index range as the neural data.

ii.
```python
trial_lick = lick[start:stop]
neural     = dff_valid[:, start:stop]
```

iii. Same shared frame clock as all other behavioral streams.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Same source as 7-a: it does not come from the raw arrays at all but from the `SESSIONS_SCENES`
scene-name table (extracted from the reference `sessions_dict.py`) combined with the trial index
and the 30-trial switch rule. The label returned by `get_reward_zone_for_trial()` is mapped
A→0, B→1, C→2.

ii.
```python
coords, label = get_reward_zone_for_trial(scene, t, n_trials, change_trial=30)
...
rz_label_int = {'A': 0, 'B': 1, 'C': 2}[rz_labels[t]]
out[4, :] = rz_label_int
```
```python
'output_values': [..., ['A (80-130)', 'B (200-250)', 'C (320-370)'], ...]
```

iii. See 7-a. Additional cross-check in trajectory step 58 on the sample: "reward_zone_location:
session 2 (m17) shows only A (100%), which is correct for Env1_LocationA". The full-dataset
distribution is `{A 0.329, B 0.337, C 0.335}`, i.e. balanced across the three zones as the
counterbalanced design implies.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene-name parsing (pre-switch zone vs post-switch zone for `*_to_*` scenes), a trial-index
comparison against `change_trial = 30`, and a label→integer mapping. The per-trial integer is
broadcast constant across all timepoints of the trial so that the output is time-varying in shape,
as the format prefers.

ii.
```python
if trial_idx < change_trial:
    return REWARD_ZONE_DICT[pre_zone], pre_zone
else:
    return REWARD_ZONE_DICT[post_zone], post_zone
```
```python
out = np.zeros((6, n_tp), dtype=np.int64)
out[:4, :] = out_tv
out[4, :] = rz_label_int
out[5, :] = reward_outcome
```

iii. CONVERSION_NOTES Step 3: "Switch trial | 30 | 'Each switch occurred after 30 trials'". The
instruction specifies "Reward zone location, per-trial. 0 = A, 1 = B, 2 = C"; the format guidance
says "If at all possible, make it time-varying", hence the broadcast.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` behavioral time series' event `timestamps` (the same `rewarded_trials` array used
for the previous-trial input, question 6-a).

ii.
```python
reward_ts = beh['Reward']['timestamps'][()]
...
rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
```

iii. CONVERSION_NOTES Step 5 mapping: "reward_ts | output[5] | reward_outcome (0/1) | Per-trial".
The paper's task randomly omits reward on ~15% of trials, so the presence/absence of a delivery
event within the trial is the outcome.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, test whether any reward event timestamp falls inside the trial's time window
`[t_start_time, t_end_time]` (times taken from the behavior timestamps at the trial-start and
teleport frames, inclusive at both ends). The resulting boolean is cast to int and broadcast
constant across all timepoints of the trial. The reward *amount* array is not used.

ii.
```python
for t in range(n_trials):
    t_start_time = timestamps[tstart_idx[t]]
    t_end_time   = timestamps[teleport_idx[t]]
    rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
...
reward_outcome = int(rewarded_trials[t])
out[5, :] = reward_outcome
```

iii. CONVERSION_NOTES Step 9 validates the result against the paper: "Reward omission rate | ~15% |
15.3% | YES" (verification output: `{no reward 0.157, reward 0.843}`). Working in the time domain
rather than the frame domain is required because the `Reward` series carries its own timestamps
rather than a frame-aligned data vector.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive behaviours, mostly silent:
- **Neural/behaviour length mismatch**: every neural and behavioural stream is truncated to
  `min(n_timepoints_neural, n_timepoints_beh)` with a printed note. This fires on the 10 two-plane
  sessions (m17 ses-04/06, m18 ses-01/05/07/10-14) where the neural array is one frame longer.
- **Unequal trial-start/teleport counts**: paired up to the common length via `min()` rather than
  asserted equal.
- **Degenerate trials/sessions**: trials with < 2 timepoints skipped; sessions with < 2 trials,
  < 2 cells, or < 2 surviving trials skipped entirely (returns `None`).
- **NaN neural values**: silently replaced by 0 with `np.nan_to_num` (rather than checked for).
- **Degenerate dF/F baseline**: frames where the baseline is NaN or exactly 0 are excluded from the
  dF/F division and remain NaN (then become 0 by the previous rule).
- **Empty/inverted trial windows** in `compute_dff` skipped via `if stop <= start: continue`.
- **Interneuron detection** skips cells with ≤ 10 valid samples or zero variance.
- **Environment −1 sentinel**: excluded before taking the per-trial median (see 4-b).
- **Discretizers** initialize to −1 so an unassigned value would be detectable.

ii.
```python
n_timepoints = min(n_timepoints_neural, n_timepoints_beh)
if n_timepoints_neural != n_timepoints_beh:
    print(f"    Note: neural ({n_timepoints_neural}) and behavioral ({n_timepoints_beh}) timepoints differ, truncating to {n_timepoints}")
    F_raw = F_raw[:, :n_timepoints]
    ...
```
```python
n_trials = min(len(tstart_idx), len(teleport_idx))
if n_trials < 2:
    return None
...
neural = np.nan_to_num(neural, nan=0.0)
...
valid = ~np.isnan(flow) & (np.abs(flow) > 0)
dff[valid] = (f_[valid] - flow[valid]) / np.abs(flow[valid])
...
if stop <= start:
    continue
...
if n_valid > 10:
    ...
    if denom > 0:
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "Neural/beh mismatch | Some sessions off-by-1 |
Truncate to minimum length"; Step 10 Check 5: "Off-by-one between neural and behavioral data:
handled by truncation; Multi-plane sessions: handled by concatenating planes". The session-level
`< 2 trials` / `< 2 cells` guards exist to satisfy the format requirement that "There needs to be
at least two trials within each session in order to evaluate the decoder performance." In practice
none of the guards fires: all 152 sessions and all 12,216 trials survive.

## 13-a. What are the most time-consuming steps of the code?

i. Identified and measured by the AI:
1. **dF/F computation** (`compute_dff`) — the dominant cost, 0.8–8.4 s per session. It runs three
   Python loops over trials, each doing a Gaussian smooth plus a 300-sample minimum and maximum
   filter over a `(n_rois, n_trial_frames)` float64 array, **for all ROIs**, not just the curated
   cells.
2. **NWB file I/O** — reading the full `(T, n_rois)` Fluorescence and Neuropil arrays for all 152
   files (each file read exactly once).
3. **Interneuron detection** — a Python loop over up to ~5000 ROIs per session.
4. **Pickling the result** — the output is a 9.8 GB float32 dense dF/F file.

Measured total: ~800 s (13.3 min) for the full conversion, under the 15-minute budget.

ii.
```python
t0 = time.time()
print(f"  Processing {subject_id} ses-{ses_num:02d}...")
...
print(f"    Computing dF/F for {cell_mask.sum()} cells, {n_timepoints} timepoints...")
dff = compute_dff(F_raw, Fneu_raw, tstart_idx, teleport_idx, neu_coef=0.7)
...
elapsed = time.time() - t0
print(f"    Done: {n_cells} cells, {len(neural_trials)} trials, {elapsed:.1f}s")
...
total_elapsed = time.time() - total_t0
print(f"\n\nTotal processing time: {total_elapsed:.1f}s")
```

iii. CONVERSION_NOTES Step 7: "| dF/F computation | 0.8-8.4s | ~13 min for 152 sessions |" and
"Actual total time: 800s (~13.3 minutes)". Trajectory step 59: "The bottleneck is the dF/F
computation which involves per-trial smoothing and filtering."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Loops that remain in the code and could be reduced or removed:
- `detect_interneurons`: a per-cell `for i in range(n_cells)` loop computing a Pearson correlation.
  This is fully vectorizable as one centered matrix–vector product over all cells at once. Note the
  docstring already claims "Vectorized implementation for speed" — it is not vectorized, only
  hand-rolled per cell.
- `compute_dff`: **three** separate `for i in range(len(trial_start_idx))` loops (neuropil add-back
  + baseline; and dF/F smoothing). The first two could be merged, and the per-trial slices could be
  handled with a single masked/segmented filter pass.
- The per-trial bookkeeping loops in `process_session` — `rewarded_trials`, `rz_labels`/`rz_coords`,
  and `env_per_trial` — are three separate Python loops over trials that could be computed with
  `np.searchsorted`/`np.add.reduceat` over the whole session at once, and merged into the main
  trial loop.
- Inside the main trial loop, `discretize_position`, `discretize_speed`, and the lick
  binarization are per-trial calls on data that could be discretized once over the whole session
  array before slicing.

ii.
```python
for i in range(n_cells):                 # detect_interneurons: not vectorized despite docstring
    valid = ~np.isnan(dff[i]) & speed_valid
    ...
```
```python
for i in range(len(trial_start_idx)):    # compute_dff loop 1: neuropil add-back + baseline
    ...
for i in range(len(trial_start_idx)):    # compute_dff loop 2: dF/F smoothing
    ...
```
```python
for t in range(n_trials):                # rewarded_trials
    ...
for t in range(n_trials):                # rz_labels / rz_coords
    ...
for t in range(n_trials):                # env_per_trial
    ...
for t in range(n_trials):                # main conversion loop
    ...
```

iii. The AI recognized the opportunity — trajectory step 59: "The main optimization opportunities:
1. The interneuron detection uses a loop over cells with pearsonr - this can be vectorized" — and
replaced `scipy.stats.pearsonr` with inline dot products, but left the loop itself in place while
labelling the function "Vectorized implementation for speed". Because the measured runtime
(13.3 min) came in under the 15-minute budget, no further optimization was pursued
(CONVERSION_NOTES Step 7).

## 13-c. What processing does the code repeat multiple times?

i. Repeats present in the code:
- The trial window `[tstart_idx[t], teleport_idx[t])` is sliced five separate times per session
  (twice inside `compute_dff`, then in `rewarded_trials`, `env_per_trial`, and the main loop).
- `nansmooth` rebuilds and Gaussian-filters the `one` weighting array from scratch on every call,
  i.e. `2 × n_trials` times per session, even though the trial windows contain no NaNs at all (all
  NaNs are outside the windows) so the whole NaN-weighting machinery is redundant there.
- `compute_dff` walks the trial list twice in full.
- The out-of-trial NaN mask is recomputed per cell inside `detect_interneurons`
  (`~np.isnan(dff[i])`) although it is identical for all cells.
- The per-trial outputs are written into `out_tv` and then copied again into `out`.

What the code notably does **not** repeat: unlike the reference, there is no separate survey pass,
so each NWB file is opened and read exactly once.

ii.
```python
def nansmooth(a, sig, axis=-1):
    nan_inds = np.isnan(a)
    ...
    one = np.ones(a.shape)          # rebuilt and filtered on every per-trial call
    one[nan_inds] = 0.001
    a_nanless = gaussian_filter1d(a_nanless, sig, axis=axis)
    one = gaussian_filter1d(one, sig, axis=axis)
    return a_nanless / one
```
```python
out_tv = np.zeros((4, n_tp), dtype=np.int64)
out_tv[0, :] = dist_bins
...
out = np.zeros((6, n_tp), dtype=np.int64)
out[:4, :] = out_tv          # copied a second time
```

iii. Not discussed in CONVERSION_NOTES; the single-pass design follows from the AI's decision to
resolve reward zones from the `SESSIONS_SCENES` metadata table rather than from a data-driven
survey, which removed the need to read every file twice.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things:
- **dF/F is computed for every ROI, then ~58% of the result is thrown away.** `compute_dff` is
  called with `F_raw`/`Fneu_raw`, the full ROI set, and only afterwards is `cell_mask` applied.
  Across the dataset there are ~2.4 ROIs per curated cell, so the dominant cost of the script does
  roughly 2.4× more work than needed. Because every step of `compute_dff` is per-cell (the Gaussian
  smoothing is 1-D along time only), subsetting first would give numerically identical results.
- **Behavioral arrays that are never used** are read out of the HDF5 file and truncated:
  `trial_number`, `scanning`, `autoreward`, and `reward_zone` (the last is loaded even though the
  AI decided to take reward zones from the scene table instead).
- **`out_trial` is built and never used** — it is constructed and immediately superseded by the
  broadcast assignment into `out`.
- **Unused imports**: `scipy.stats.pearsonr` and `scipy.ndimage.gaussian_filter` are imported but
  never called (leftovers from the pre-optimization version).
- In `plot_processing`, `pos = np.argmax(...)` is computed and never used.
- The output is stored as a dense float32 dF/F array, producing a 9.8 GB pickle.

ii.
```python
dff = compute_dff(F_raw, Fneu_raw, tstart_idx, teleport_idx, neu_coef=0.7)  # all ROIs
dff_cells = dff[cell_mask]                                                  # ~42% kept
```
```python
trial_number = beh['trial number']['data'][()]   # never used
scanning     = beh['scanning']['data'][()]       # never used
autoreward   = beh['autoreward']['data'][()]     # never used
reward_zone  = beh['reward_zone']['data'][()]    # never used
```
```python
out_trial = np.array([rz_label_int, reward_outcome], dtype=np.int64)   # never used
```
```python
from scipy.stats import pearsonr                                  # never used
from scipy.ndimage import ..., gaussian_filter                    # never used
```

iii. Not discussed in CONVERSION_NOTES. The unused behavioral arrays appear to be leftovers from
the exploration phase, when the AI was still evaluating `reward_zone`, `trial_number`, `scanning`,
and `autoreward` as candidate sources (trajectory steps 23–24, 44). The ROI-wide dF/F computation
was never flagged, even though CONVERSION_NOTES Step 7 identifies dF/F as the runtime bottleneck.
