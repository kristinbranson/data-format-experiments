# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every NWB file with a single recursive glob over `data/sub-*/*.nwb`, sorted, and processes each file independently in a serial loop with `pynwb.NWBHDF5IO`. All 152 files (11 subject directories) are found. `--sample` truncates to the first 2 files. Within each file it reads exactly two groups: `processing['behavior']['BehavioralTimeSeries']` (11 named series) and `processing['ophys']['Deconvolved'].roi_response_series['plane0']`. It does **not** read `Fluorescence`, `Neuropil`, `ImageSegmentation`/`iscell`, or `plane1` (28 of the 152 sessions, all of subjects m17 and m18, have a second imaging plane holding 52,019 additional ROIs, 17% of all ROIs, which are silently discarded). No neural/behavior length check is made (10 sessions have one more neural sample than behavior samples).

ii.
```python
def get_files(sample=False):
    files = sorted(Path('data').glob('sub-*/*.nwb'))
    return files[:2] if sample else files
```
```python
def process_file(fpath):
    with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
        nwb = io.read()
        subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)
        sess_id = nwb.session_id
        bts = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
        deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']

        neural = np.asarray(deconv.data[:], dtype=np.float32)
        ...
        behavior = {}
        for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
            behavior[k] = _ts_data(bts[k])
```
```python
    for i, f in enumerate(files, 1):
        sess = process_file(f)
        if len(sess['neural']) >= 2:
            sessions.append(sess)
```

iii. CONVERSION_NOTES Step 2: "Data are organized as NWB files under per-subject directories `data/sub-*`. There are 152 session files total across 11 subjects. Each session is a single `*_behavior+ophys.nwb` file containing both behavioral time series and 2-photon ophys data." Step 10 Check 4 claims "converted dataset matches full-release counts of 11 subjects, 152 sessions, and 260091 neurons". No justification is given anywhere in the notes or trajectory for reading only `plane0`; the trajectory records only the observation that "`plane0` [is] shaped `(time, neurons)` at rate 15.5078125 Hz" from a single representative (single-plane) file, and `plane1` is never mentioned again.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the NWB metadata field `nwb.subject.subject_id`, with the parent directory name as a fallback. The unique subject strings across all included sessions are sorted to build `data['subjects']`, and `subject_idx` is the index of each session's subject in that list. This yields the same 11 subjects and the same sessions-per-subject counts as the reference (m11: 12; all others: 14).

ii.
```python
subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)
```
```python
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. Not explicitly justified beyond CONVERSION_NOTES Step 2 ("Data are organized as NWB files under per-subject directories `data/sub-*` ... across 11 subjects") and Step 10 Check 4 ("converted dataset matches full-release counts of 11 subjects"). The subject id is read from the file's own metadata rather than parsed from the path, which is the more authoritative source.

## 1-c. How are the data split into sessions?

i. One session per NWB file. Each `process_file` call returns one session dict, and sessions are appended to the output lists in sorted-filename order. The session is retained only if it yields at least 2 trials. No cross-session neuron alignment/registration is attempted. 152 sessions result.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
...
for i, f in enumerate(files, 1):
    sess = process_file(f)
    if len(sess['neural']) >= 2:
        sessions.append(sess)
```
```python
sess_id = nwb.session_id
```

iii. CONVERSION_NOTES Step 5, Key Decision 8: "**Session inclusion**: Keep sessions with at least two trials after segmentation; later filtering may be needed if environment/reward-zone metadata are invalid." The 2-trial floor comes from the instruction "There needs to be at least two trials within each session in order to evaluate the decoder performance." No session was actually dropped.

## 1-d. How are the data split into trials?

i. Trial onsets are the indices where the `trial_start` behavior series is positive. **Each trial ends at the next `trial_start`**, and the final trial ends at the end of the recording. The `teleport` series is loaded but never used. Consequently each "trial" is the lap **plus** the whole following inter-trial/teleport interval. Measured on `sub-m11_ses-03`, 26.9% of all samples inside the AI's trial windows are inter-trial samples (position sentinel `-50`), versus 0.01% for the reference `trial_start`→`teleport` windows; the longest emitted "trial" is 10,132 samples (653 s) of which 95% is inter-trial time. Mean trial length is 297.8 samples vs 216.8 for the reference, and max is 10,132 vs 3,359.

ii.
```python
def trial_bounds_from_trial_start(trial_start):
    starts = np.flatnonzero(trial_start > 0)
    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_start)
        if e > s:
            bounds.append((s, e))
    return bounds
...
bounds = trial_bounds_from_trial_start(trial_start)
...
for ti, (s, e) in enumerate(bounds):
```

iii. CONVERSION_NOTES Step 5, Key Decision 3: "**Trial boundaries**: Use each `trial_start` as onset and the next `trial_start` (or `teleport`/end-of-recording for the final trial) as trial end." Step 4 records that `teleport` exists with the same count as `trial_start` ("`teleport` is also a binary impulse with 80 events later in each trial") but no reason is given for preferring the next trial start. Trajectory step ~379: "trials can be derived from `trial_start` or `teleport`".

## 1-e. How are trials filtered based on quality controls?

i. Three per-trial guards, applied inside the trial loop: (1) drop trials shorter than 2 samples; (2) drop trials in which the `trial number` series never becomes non-negative (pre-task baseline); (3) drop trials in which position never rises above -100 cm (pure teleport/blank periods). There is no minimum-duration quality criterion (the reference drops trials with < 50 timepoints). Because the AI's windows include the inter-trial interval, the shortest emitted trial is 104 samples, so no trial would have been caught by a 50-sample rule either; the resulting total is 12,216 trials, the same total as the reference.

ii.
```python
for ti, (s, e) in enumerate(bounds):
    if e - s < 2:
        continue
    if np.nanmax(trial_num[s:e]) < 0:
        continue
    if np.all(position[s:e] <= -100):
        continue
```

iii. CONVERSION_NOTES Step 5, Key Decision 9: "**Baseline exclusion**: Exclude pre-task baseline samples where trial number/environment are -1 or position is sentinel-valued (for example -500 cm) from trial segmentation." Step 10 Check 5: "observed nonuniform trial counts across sessions (for example 41, 50, 60, 75, 90, 100) are preserved rather than forced to 80, consistent with raw data." Note the stated decision is to exclude sentinel *samples*; the code only excludes whole trials that are entirely sentinel.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is the NWB `processing['ophys']['Deconvolved'].roi_response_series['plane0']` array, cast to float32 and transposed to (n_neurons, n_timepoints) per trial. `Fluorescence` (F) and `Neuropil` (Fneu) are never read. `plane1` is never read, so in the 28 two-plane sessions (all of m17 and m18) roughly half of each session's ROIs — 52,019 in total — are dropped.

ii.
```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
neural = np.asarray(deconv.data[:], dtype=np.float32)
...
trial_neural = neural[s:e, :].T.astype(np.float32)
sess_neural.append(trial_neural)
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "Reference methods use deconvolved activity for remapping and GLM analyses | NWB contains `Deconvolved`, `Fluorescence`, and `Neuropil` interfaces | Methods explicitly state deconvolved activity matrices were used | **Use deconvolved activity as primary neural signal**." Step 5 Key Decision 1: "**Neural signal**: Use deconvolved activity because both methods and NWB organization indicate this is the primary processed neural variable for remapping/GLM analyses." No justification is offered for ignoring `plane1`.

## 2-b. How is the `neural` data processed?

i. No processing is applied. The stored deconvolved trace is sliced by trial, transposed and cast to float32. There is no neuropil subtraction, no maximin baseline, no dF/F computation, no Gaussian smoothing, and no OASIS deconvolution — i.e. none of the pipeline described in the paper's Methods and implemented in the reference `dff()`.

ii.
```python
neural = np.asarray(deconv.data[:], dtype=np.float32)
...
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Implicit in Step 5 Key Decision 1 ("Use deconvolved activity because both methods and NWB organization indicate this is the primary processed neural variable"). Step 3 "Processing Details" records only "Neural signal used in reference text is deconvolved activity." The instruction "For neural imaging data, does delta F over F need to be computed?" is never answered in the notes, and the trajectory shows the agent never opened `Fluorescence` or `Neuropil`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality control at all. Every ROI in `Deconvolved/plane0` is kept — 260,091 ROIs in total. The `iscell` column of `ImageSegmentation/PlaneSegmentation` (suite2p's manual curation) is present in every file but never read; in `sub-m11_ses-03` only 155 of 349 ROIs are flagged `iscell`. The reference's second filter (dropping putative interneurons whose dF/F correlates with running speed at r > 0.5) is also absent. Reference total after both filters: 138,298 neurons. As a side effect, several sessions have so many ROIs (up to 3,934) that the decoder falls back to random projection, as noted in the training log.

ii. There is no filtering code. The only ROI-related code is:
```python
n_neurons = neural.shape[1]
region_idx = np.zeros(n_neurons, dtype=np.int64)
```
```python
'brain_region_idx': [np.full(len(s['brain_region_idx']), region_to_idx[s['region']], dtype=np.int64) for s in sessions],
```

iii. No justification is given. CONVERSION_NOTES Step 3 "Neuron curation rules" lists only analysis-specific criteria from the paper that were not applied ("included place cells required significant spatial information (SI)…", "cells with fraction deviance explained (FDE) > 0.15"), and the notes present 260,091 as the correct neuron count ("converted dataset matches full-release counts of … 260091 neurons"). Step 2 lists `ImageSegmentation` as available ("ROI/cell metadata are available through the ophys interfaces including `ImageSegmentation`") but it is never inspected.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, achieved by slicing: the neural array and every behavior array share the same sample index, and each trial is the slice `[s:e]` beginning exactly at the `trial_start` sample. No offset, padding, or interpolation is applied; `off_start` is recorded as 0.0 and `off_end` as None.

ii.
```python
for ti, (s, e) in enumerate(bounds):
    ...
    trial_neural = neural[s:e, :].T.astype(np.float32)
```
```python
'temporal_alignment_event': 'trial_start',
'off_start': 0.0,
'off_end': None,
```

iii. CONVERSION_NOTES Step 5, Key Decision 2: "**Temporal alignment**: Align trials to `trial_start`, matching the decoder task requirement and consistent with trial-based analyses in the reference materials." Key Decision 4: "**Time base**: Use the native shared sampling grid (~15.5 Hz) because behavior and deconvolved traces appear synchronized on the same timestamps." Step 10 sanity check: "converted neural trial 0 matches the raw deconvolved slice" via `np.allclose`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. The native per-sample resolution of the stored traces is kept (64.4836 ms per sample, 15.5078 Hz, uniform across all 152 sessions). `metadata['time_bin_size']` is computed as `1000 / rate` of the **first** session only, giving 64.48 ms. A caveat: `rate` is read verbatim from each series, and for the 28 two-plane sessions the stored `rate` is the scanner rate 31.0156 Hz rather than the per-plane rate 15.5078 Hz; the neural samples themselves are unaffected (no rebinning happens), but the derived time axis for those sessions is 2× too fast (see 3-b and 11-b).

ii.
```python
rate = float(deconv.rate) if getattr(deconv, 'rate', None) is not None else float(1.0 / np.median(np.diff(timestamps)))
```
```python
'time_bin_size': float(1000.0 / sessions[0]['rate']) if sessions else None,
```

iii. CONVERSION_NOTES Step 5, Key Decision 4: "**Time base**: Use the native shared sampling grid (~15.5 Hz) because behavior and deconvolved traces appear synchronized on the same timestamps." Step 3: "~64.5 ms sample interval (15.5078125 Hz) in NWB deconvolved traces". Trajectory step ~370: "All behavior streams share timestamps from 0 to ~1277.87 s with median dt ~0.06448 s, matching the ophys sampling interval (~15.5 Hz)" — verified on one single-plane session only.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From the ophys series' own time base, not from the behavior timestamps. `Deconvolved/plane0` stores no `timestamps`, so the code falls through to the synthetic grid `np.arange(n_samples) / deconv.rate`. The stored behavior `timestamps` (available on every behavior series, starting at 0.0 with dt = 0.0644836 s) are loaded by `_ts_data` for all 11 series but never used for this input.

ii.
```python
neural = np.asarray(deconv.data[:], dtype=np.float32)
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
rate = float(deconv.rate) if getattr(deconv, 'rate', None) is not None else float(1.0 / np.median(np.diff(timestamps)))
```

iii. CONVERSION_NOTES Step 5 mapping table: "behavior timestamps + `trial_start` → input[0] time_from_trial_start | per-trial continuous time in seconds from 0 at trial start | Align all trials to trial start." Key Decision 4 justifies the shared time base: "behavior and deconvolved traces appear synchronized on the same timestamps." (The notes say behavior timestamps; the code actually reconstructs the axis from the ophys `rate`.)

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial's first timestamp is subtracted from the trial's timestamp slice, giving 0 at trial start, and the result is cast to float32 and stored as row 0 of the per-trial `input` matrix (time-varying).

ii.
```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
inp = np.vstack([
    rel_time,
    np.full(e - s, env_label, dtype=np.float32),
    np.full(e - s, tr_label, dtype=np.float32),
    np.full(e - s, prev_rew, dtype=np.float32),
])
```

iii. CONVERSION_NOTES Step 5 mapping table: "per-trial continuous time in seconds from 0 at trial start". Not otherwise justified — treated as self-evident.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the timestamp vector is derived from the neural array's own length and rate, and is sliced with the same `[s:e]` indices as the neural matrix, so the two are index-identical. No resampling or interpolation is performed and no assertion checks behavior-vs-neural length agreement.

ii.
```python
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
...
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. CONVERSION_NOTES Step 5, Key Decision 4: "Use the native shared sampling grid (~15.5 Hz) because behavior and deconvolved traces appear synchronized on the same timestamps." Step 10 sanity check 3 was planned as "For one trial, verify neural and behavior arrays have matching timepoint counts after segmentation."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. From the `environment` behavior time series in `BehavioralTimeSeries`. Values are -1 during the pre-task baseline and 0 or 1 during the task.

ii.
```python
for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
    behavior[k] = _ts_data(bts[k])
env = np.asarray(behavior['environment'][0]).ravel()
```

iii. CONVERSION_NOTES Step 5 mapping table: "`environment` → input[1] environment_type | per-trial binary label, broadcast across timepoints | Use valid task codes 0/1 directly; exclude baseline -1." Key Decision 10: "**Environment mapping**: Valid task samples use environment codes 0 and 1 directly; baseline samples use -1 and should be excluded." Trajectory step ~404: "Environment codes appear to be directly 0/1 during valid task periods (with session 08 mixed), so environment can likely be used as-is after excluding baseline -1."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial window, samples with `environment < 0` (baseline sentinel) are discarded, the median of the remaining values is taken and rounded to an integer, and that single label is broadcast as a constant across all timepoints of the trial. If no valid sample exists, the label defaults to 0. The verification output confirms the range is [0, 1] and that environment is constant within a session except at the switch session boundary.

ii.
```python
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
...
inp = np.vstack([
    rel_time,
    np.full(e - s, env_label, dtype=np.float32),
    ...
])
```

iii. Key Decision 5: "**Inputs as time-varying matrices**: Broadcast per-trial scalar inputs (environment, trial number, previous outcome) across trial timepoints so each trial input has shape `(n_input, n_timepoints)`." Key Decision 10 (above) justifies dropping the -1 baseline code. The median is used for robustness against the mixed/sentinel samples the agent observed ("session 08 mixed").

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. From the stored `trial number` behavior time series (not the loop index). The loop index `ti` is only used as a fallback if the trial window contains no non-negative `trial number` sample.

ii.
```python
trial_num = np.asarray(behavior['trial number'][0]).ravel()
...
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
```

iii. CONVERSION_NOTES Step 5 mapping table: "`trial number` → input[2] trial_number | per-trial continuous label, broadcast across timepoints | Use within-session trial index." Step 4: "`trial number` runs from -1 baseline to 80, indicating trial indexing over time." The reference instead deliberately used the sequential loop index because the stored `trial number` "did not agree with the `trial_start` variable"; measured here, the AI's median-of-`trial number` differs from the sequential index in 246 of 12,216 trials (2.0%).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Baseline sentinel samples (`trial number < 0`) are dropped, the median of the remaining values within the trial window is taken as a float, and it is broadcast as a constant across all timepoints of the trial. Ranges in the converted data are [0, 79], [0, 99], etc., matching the number of trials per session.

ii.
```python
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
...
np.full(e - s, tr_label, dtype=np.float32),
```

iii. Key Decision 5 (broadcast per-trial scalars across timepoints) and Key Decision 9 (exclude baseline sentinel values). The median gives robustness when a window spans the -1 baseline or a boundary.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Indirectly from the `Reward` behavior time series' **timestamps**. Per trial, `rew` is 1 if any reward timestamp falls inside the trial's time window; these per-trial outcomes are accumulated in a list, and the previous trial's entry is read back as the previous-trial-outcome input.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
...
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)
```

iii. CONVERSION_NOTES Step 5 mapping table: "previous trial reward outcome from `Reward` events → input[3] previous_trial_outcome | per-trial binary label, broadcast across timepoints | First trial may use 0 or NaN-safe default; likely 0." Step 4: "`Reward` is an event series of length 74, suggesting 74 rewarded trials out of 80 … Define per-trial reward outcome by whether a `Reward` event occurs within the trial."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `reward_outcomes[-2]` is read (the current trial's outcome has already been appended, so `[-2]` is the previous *retained* trial), defaulting to 0 when fewer than two trials have been accumulated (i.e. the session's first trial). The scalar is broadcast across all timepoints of the trial as row 3 of `input`.

ii.
```python
prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
...
inp = np.vstack([
    rel_time,
    np.full(e - s, env_label, dtype=np.float32),
    np.full(e - s, tr_label, dtype=np.float32),
    np.full(e - s, prev_rew, dtype=np.float32),
])
```

iii. Key Decision 5 (broadcast per-trial scalars) and the Step 5 mapping note "First trial may use 0 … likely 0", consistent with the instruction "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behavior series plus a per-trial reward-zone label derived from `reward_zone` and `position`. The label is obtained in three steps: (1) `reward_zone_centers` computes, **once per session**, the median position of all samples carrying each non-zero `reward_zone` code 1–6; (2) per trial, the modal non-zero `reward_zone` code is looked up in that session-wide table; (3) the resulting position is collapsed to A/B/C by the hard thresholds 150 cm and 260 cm. If the trial has no non-zero `reward_zone` but was rewarded, the position at the reward timestamp is used; if neither, the trial's **median position** is used as a stand-in for the reward-zone location. Finally the label is converted back to a single point via `{0: 90.0, 1: 205.0, 2: 325.0}` and distance is `position − that point`.

ii.
```python
def reward_zone_centers(position, reward_zone):
    centers = {}
    for code in sorted(c for c in np.unique(reward_zone) if c > 0):
        pos = position[reward_zone == code]
        pos = pos[np.isfinite(pos)]
        if len(pos):
            centers[int(code)] = float(np.median(pos))
    return centers

def collapse_zone_position_to_abc(pos):
    if pos < 150:
        return 0
    if pos < 260:
        return 1
    return 2
```
```python
rz_vals = reward_zone[s:e]
rz_nz = rz_vals[rz_vals > 0]
if len(rz_nz):
    code = int(np.bincount(rz_nz.astype(int)).argmax())
    center = zone_centers.get(code, np.nan)
elif rew:
    ridx = np.searchsorted(timestamps, reward_ts[(reward_ts >= t0) & (reward_ts <= t1 + 1e-9)][0])
    ridx = min(ridx, len(position) - 1)
    center = float(position[ridx])
else:
    center = np.nanmedian(position[s:e])
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
```
```python
zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
zc = zone_center_lookup[zone_label]
d = position[s:e] - zc
```

iii. CONVERSION_NOTES Step 5, Key Decision 11: "**Reward-zone mapping**: NWB reward_zone codes 1-6 appear to collapse onto three physical reward locations along the corridor (~80-100 cm, ~200 cm, ~320-330 cm), so map trial reward-zone identity A/B/C by the physical position of active reward-zone samples or reward delivery, not by raw code alone." Trajectory step ~404: "Reward-zone codes are 1–6, but the reward positions cluster into three broad physical locations across sessions … implying the six codes likely correspond to two environment-specific encodings for the same three physical reward-zone identities A/B/C … reward zone identity can be inferred by position of the active reward-zone code or reward event, mapping to A/B/C by ascending corridor position."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. `d = position − zone_center_point`, where the zone is represented by a **single point** (90, 205 or 325 cm) rather than an interval. There is no notion of being "inside" the zone, so the signed distance is essentially never exactly 0. Every sample of the trial window, including inter-trial samples where position is the sentinel −50, is converted. The resulting continuous distance is then bucketed sample-by-sample by the Python function `distance_bin`.

ii.
```python
zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
zc = zone_center_lookup[zone_label]
d = position[s:e] - zc
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5 mapping table: "`position` + `reward_zone` → output[0] distance_to_reward_zone | time-varying discretized categorical 0-6 per task spec | Need mapping from reward-zone identity to zone location and signed distance." Step 7 "Processing Plots Review" records the consequence without fixing it: "Distance-to-reward-zone distribution appears not to include the exact 0 bin in the sample, suggesting the current center-based discretization may need refinement." Trajectory step ~442 repeats: "`distance_to_reward_zone` output values do not include the exact `0` bin … because we used approximate zone centers rather than exact zone occupancy. This may be acceptable but should be investigated."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. A scalar Python function applied per sample in a list comprehension, with edges matching the instruction spec exactly: `< -50 → 0`, `[-50, -10) → 1`, `[-10, 0) → 2`, `== 0 → 3`, `(0, 10] → 4`, `(10, 50] → 5`, `> 50 → 6`.

ii.
```python
def distance_bin(d):
    if d < -50:
        return 0
    if d < -10:
        return 1
    if d < 0:
        return 2
    if d == 0:
        return 3
    if d <= 10:
        return 4
    if d <= 50:
        return 5
    return 6
```
```python
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
```
```python
'output_values': [
    ['lt_neg50', 'neg50_to_neg10', 'neg10_to_0', '0', '0_to_10', '10_to_50', 'gt_50'],
    ...
]
```

iii. Directly transcribed from the Decoder Task spec in the instructions ("0: < -50 cm; 1: -50 to -10 cm; 2: -10 cm to < 0 cm; 3: 0 cm; 4: >0 cm to +10 cm; 5: +10 to +50 cm; 6: > +50 cm"). The class-3 emptiness caused by 7-b was noted in Step 7 but not corrected.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. `position` is sliced with the same `[s:e]` window as the neural matrix, so the distance series is index-aligned to the neural series sample-for-sample. No shifting or interpolation.

ii.
```python
d = position[s:e] - zc
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
...
output = np.vstack([out_dist, out_pos, out_speed, out_lick, out_rz, out_rew])
...
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Key Decision 4: "Use the native shared sampling grid (~15.5 Hz) because behavior and deconvolved traces appear synchronized on the same timestamps." Step 10 sanity checks verified via `np.allclose` that "converted neural trial 0 matches the raw deconvolved slice" and "converted lick output matches binarized raw lick for a chosen trial", i.e. that neural and behavior slices use the same indices.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. From the `position` behavior time series (cm along the VR corridor), sliced by trial window. No other variable is used.

ii.
```python
position = np.asarray(behavior['position'][0]).ravel()
...
out_pos = pos_bins(position[s:e])
```

iii. CONVERSION_NOTES Step 5 mapping table: "`position` → output[1] absolute_position_bin | time-varying discretized into 5 equal bins across corridor | Likely use valid corridor span excluding pre-trial sentinel values." Step 4/trajectory: "`position` is in cm and includes values like -500 and around -50 up to ~447".

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The raw position slice is **clipped** to `[0, 450)` and then digitized. Because the AI's trial windows include the inter-trial interval, all the sentinel `-50` samples (26.9% of all samples in the converted trials, measured on `sub-m11_ses-03`) are clipped to 0 and land in bin 0. The resulting bin-0 fraction is 0.421, against 0.211 for the reference. Nothing else is done — no smoothing, no lap-wrapping.

ii.
```python
def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
    return out.astype(np.int64)
...
out_pos = pos_bins(position[s:e])
```

iii. CONVERSION_NOTES Step 5 mapping note: "Likely use valid corridor span excluding pre-trial sentinel values", and Key Decision 9: "**Baseline exclusion**: Exclude pre-task baseline samples where trial number/environment are -1 or position is sentinel-valued (for example -500 cm) from trial segmentation." Step 3 records the 450 cm track ("RR/remapping analyses use a 450 cm track represented with 45 bins"). The clipping choice itself is not discussed anywhere.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins over the 450 cm track, via `np.linspace(0, 450, 6)` with the two outer edges dropped and `np.digitize`: `< 90 → 0`, `90–180 → 1`, `180–270 → 2`, `270–360 → 3`, `≥ 360 → 4`. Because of the prior clip, out-of-range samples on either end are folded into bins 0 and 4.

ii.
```python
def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
    return out.astype(np.int64)
```
```python
['bin0', 'bin1', 'bin2', 'bin3', 'bin4'],
```

iii. Directly from the Decoder Task spec: "Absolute position in corridor, time-varying. Discretized into 5 equal-sized bins spanning the 450 cm track: 0: < 90 cm; 1: 90 to 180 cm; …; 4: > 360 cm." Step 3 confirms the 450 cm track length from the Methods.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same-index slicing: `position[s:e]` uses the identical window as `neural[s:e]`, so the binned position series is sample-aligned to the neural matrix. No resampling.

ii.
```python
out_pos = pos_bins(position[s:e])
...
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Key Decision 4 ("behavior and deconvolved traces appear synchronized on the same timestamps"), plus the Step 10 spot-checks against direct NWB reads.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. From the `lick` behavior time series, an integer-valued (0–6) per-sample lick count.

ii.
```python
lick = np.asarray(behavior['lick'][0]).ravel()
...
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 mapping table: "`lick` → output[3] lick | time-varying binary; convert nonzero lick counts to 1 | Observed lick values 0-6." Trajectory step ~370: "`lick` is an AU-valued discrete time series with values 0–6".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarization only: any value > 0 becomes 1, everything else 0. No smoothing or event detection. Resulting fraction of lick=1 is 0.183 (reference 0.230; the difference is attributable to dilution by the inter-trial samples included in the AI's trial windows, during which the animal does not lick).

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
```
```python
['no', 'yes'],
```

iii. Key Decision 6: "**Lick binarization**: Convert lick values >0 to 1 for decoder output", matching the instruction "Lick, time-varying. 0 = no, 1 = yes."

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same-index slicing `lick[s:e]`, identical window to `neural[s:e]`.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
...
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Key Decision 4 (shared native sampling grid). Step 10 sanity check: "converted lick output matches binarized raw lick for a chosen trial" (trial 5, exact match).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Same derivation as 7-a: the `reward_zone` behavior series (codes 1–6), the `position` series, and — in the fallback branches — the `Reward` timestamps. The per-trial A/B/C label is emitted as a constant time series (row 4 of `output`).

ii.
```python
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
trial_zone_labels.append(zone_label)
...
out_rz = np.full(e - s, zone_label, dtype=np.int64)
```
```python
['A', 'B', 'C'],
```

iii. See 7-a. Key Decision 11: map "trial reward-zone identity A/B/C by the physical position of active reward-zone samples or reward delivery, not by raw code alone."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Three-branch assignment per trial (modal `reward_zone` code → session-wide median position for that code; else position at the reward event; else the trial's median position), then the fixed thresholds 150/260 cm map the position to A=0, B=1, C=2, and 225.0 cm is the default if everything is NaN. Two consequences are measurable: (1) because the code→position table is built from the **whole session**, it averages across the mid-session reward-zone switch that is the paper's central manipulation, so a switch session is collapsed to a single label — e.g. in `sub-m11_ses-03` trials 0–29 have the reward zone at ~200 cm (B) and trials 30–79 at ~82 cm (A), yet every trial is labelled A; agreement with a per-trial assignment is 84.1% over a 9-session sample and as low as 60% in switch sessions. (2) 15.8% of trials (the omission trials, which carry no `reward_zone` samples and no reward) fall to the `np.nanmedian(position)` branch, which is not a reward-zone measurement at all and is biased toward A by the included inter-trial samples. Resulting class fractions are A 0.409 / B 0.306 / C 0.284 vs the reference's 0.329 / 0.337 / 0.334.

ii.
```python
zone_centers = reward_zone_centers(position, reward_zone)   # computed once per session
...
rz_vals = reward_zone[s:e]
rz_nz = rz_vals[rz_vals > 0]
if len(rz_nz):
    code = int(np.bincount(rz_nz.astype(int)).argmax())
    center = zone_centers.get(code, np.nan)
elif rew:
    ridx = np.searchsorted(timestamps, reward_ts[(reward_ts >= t0) & (reward_ts <= t1 + 1e-9)][0])
    ridx = min(ridx, len(position) - 1)
    center = float(position[ridx])
else:
    center = np.nanmedian(position[s:e])
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
```

iii. Key Decision 11 (above) and the trajectory's cross-session survey: "session 03 reward events occur near ~200 cm, session 04 near ~80–90 cm, session 06 near ~320–330 cm. Thus, reward zone identity can be inferred by position of the active reward-zone code or reward event, mapping to A/B/C by ascending corridor position." The 150/260 cm split points and the 90/205/325 cm centers are not justified in the notes. Step 7 flags only that "Reward-zone labels in the sample only cover A/B, with C absent in the first two sessions."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From the `Reward` behavior time series' **timestamps** (its `data` values are read but unused). A trial is rewarded if any reward timestamp falls in the closed interval `[timestamps[s], timestamps[e-1]]` of the trial.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
...
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
```

iii. CONVERSION_NOTES Step 4: "Representative NWB file has 74 `Reward` events for 80 trials; reward events occur at specific timestamps rather than dense per-frame labels … Define per-trial reward outcome by whether a `Reward` event occurs within the trial." Key Decision 7: "**Reward outcome**: Mark a trial rewarded if any `Reward` event timestamp falls within that trial."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The binary flag is broadcast as a constant across all timepoints of the trial (row 5 of `output`) and additionally retained in `reward_outcomes` for the previous-trial input. The comparison window is built from the neural-derived `timestamps` array. Because that array is `np.arange(T)/rate` and `rate` for the 28 two-plane sessions (all of m17 and m18) is the scanner rate 31.0156 Hz rather than the true per-plane 15.5078 Hz, the comparison windows in those sessions are at half the true elapsed time: on `sub-m17_ses-01` this yields 32 "rewarded" trials where 66 are truly rewarded, and only 36/80 trials (45%) get the right label. Across the whole dataset the reward=1 fraction is 0.740 (timepoint-weighted) vs 0.843 for the reference. The same defect propagates into *Previous trial outcome* (6-b) and into the reward-zone fallback branch (10-b).

ii.
```python
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
...
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)
...
out_rew = np.full(e - s, rew, dtype=np.int64)
```

iii. Key Decision 7 (above). The notes present this as validated: Step 10, "reward outcome for the first 10 trials matches perfectly" — but the spot check was run only on `sub-m11_ses-03`, a single-plane session, so the two-plane failure mode was never exercised. No reward-timestamp alignment assertion is present (the reference asserts the alignment error is within half a time bin).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles a handful of cases defensively and leaves several unhandled.

Handled:
- Degenerate/empty trial windows (`e > s`, and `e - s < 2` skipped).
- Pre-task baseline windows: skipped when `trial number` never reaches ≥ 0, or when position never exceeds −100 cm.
- Sentinel values in the `environment` and `trial number` inputs: negative samples are filtered before taking the median, with 0 / the loop index as fallbacks.
- Missing ROI-region metadata: `infer_region` is wrapped in a bare `try/except` returning `'unknown'`.
- Missing `Reward` timestamps: an empty array is substituted.
- Out-of-range positions: clipped into `[0, 450)`.
- Negative speeds: clamped to 0.
- Missing `reward_zone` and unknown reward-zone centers: cascading fallbacks ending in 225.0 cm.

Not handled:
- Neural/behavior length mismatch. 10 sessions have one more neural sample than behavior samples; the code slices the neural array with behavior-derived indices and so silently drops the extra sample without any check or warning (the reference explicitly crops to the minimum and warns).
- No minimum-trial-duration criterion.
- No assertion that reward timestamps land within half a bin of a behavior sample.
- No check that the ophys `rate` attribute is the per-plane rate (the root cause of the m17/m18 errors above).
- Missing imaging planes: `plane1` is not looked for, so its absence/presence is never handled.

ii.
```python
if e - s < 2:
    continue
if np.nanmax(trial_num[s:e]) < 0:
    continue
if np.all(position[s:e] <= -100):
    continue
```
```python
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
```
```python
def infer_region(nwb):
    try:
        ...
        return region if region else 'unknown'
    except Exception:
        return 'unknown'
```
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
...
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
```

iii. CONVERSION_NOTES Key Decision 9 ("Baseline exclusion: Exclude pre-task baseline samples where trial number/environment are -1 or position is sentinel-valued (for example -500 cm)") and Key Decision 8 ("Keep sessions with at least two trials after segmentation"). Step 10 Check 5 claims the edge-case review found only that "observed nonuniform trial counts across sessions … are preserved rather than forced to 80, consistent with raw data", and Step 10/12 both conclude "No format errors or verification warnings in the full dataset."

## 13-a. What are the most time-consuming steps of the code?

i. From `conversion_full_out.txt`: the whole full conversion takes 201.3 s. Per-file processing sums to 158.1 s (mean 1.04 s/session, max 1.6 s), essentially all of it NWB I/O — reading the full `Deconvolved/plane0` array (up to 50,033 × 3,934 float values) and the 11 behavior series with their timestamp vectors into memory. The remaining ~43 s is the single `pickle.dump` of the 25.2 GB output file. The per-trial Python loops are negligible in comparison. The dominant *downstream* cost is the file size itself: verification and decoder training each spend minutes just loading the 25 GB pickle (the trajectory shows many turns spent waiting on this), and the decoder is forced into random projection on high-ROI sessions.

ii.
```python
    t0 = time.time()
    sessions = []
    for i, f in enumerate(files, 1):
        st = time.time()
        sess = process_file(f)
        ...
        print(f'processed {i}/{len(files)} {f.name}: trials={len(sess["neural"])} neurons={(sess["brain_region_idx"].shape[0])} time={time.time()-st:.2f}s', flush=True)
```
```python
    with open(args.outpickle, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpickle} with {len(sessions)} sessions in {time.time()-t0:.2f}s', flush=True)
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Repeated full NWB reads may be slow for all 152 sessions; optimize later if full conversion is too slow. Code speedups added: Uses vectorized slicing on synchronized time series and single-pass per-session processing." Step 7 estimated "~0.7 s/session, ~2-3 min for 152 sessions", which matched the actual 3.4 min, so no further optimization was pursued.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-sample Python list comprehensions in the hot path call a scalar function on every timepoint of every trial (~3.6 M samples each across the dataset): `distance_bin` and `speed_bin`. Both are pure threshold lookups and could be a single `np.digitize` call on the whole session array — exactly what the AI already does for position via `pos_bins`, and what the reference does for all three. `trial_bounds_from_trial_start` also builds its list in a Python loop over trial starts, which is a two-line vectorized `np.stack([starts, np.append(starts[1:], T)])`. More broadly, the whole per-trial loop's element-wise work (distance, position bins, speed bins, lick binarization) could be computed once per session on the full arrays and then sliced.

ii.
```python
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
```
```python
def trial_bounds_from_trial_start(trial_start):
    starts = np.flatnonzero(trial_start > 0)
    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_start)
        if e > s:
            bounds.append((s, e))
    return bounds
```

iii. Not identified by the AI. CONVERSION_NOTES Step 6 asserts the opposite: "Code speedups added: Uses vectorized slicing on synchronized time series and single-pass per-session processing." Since the measured runtime (3.4 min) was under the 15-minute threshold in the instructions, no vectorization pass was undertaken.

## 13-c. What processing does the code repeat multiple times?

i. Within the trial loop several slices and objects are recomputed rather than hoisted: `env[s:e]` is sliced twice on consecutive lines, `trial_num[s:e]` twice, `position[s:e]` three times (for distance, for position bins, and potentially for the median fallback), and the constant dict `zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}` is rebuilt on every trial iteration. `_ts_data` reads the full `timestamps` vector for all 11 behavior series on every file even though only `Reward`'s timestamps are ever used — 10 redundant full-length array reads per session. The per-sample discretizations are also recomputed per trial instead of once per session (see 13-b). Unlike the reference, however, the script makes only **one** pass over each NWB file (the reference reads every file twice: once in `survey()` and once in the conversion).

ii.
```python
def _ts_data(ts):
    data = np.asarray(ts.data[:])
    t = np.asarray(ts.timestamps[:]) if ts.timestamps is not None else None
    return data, t
...
for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
    behavior[k] = _ts_data(bts[k])
```
```python
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
```
```python
zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
zc = zone_center_lookup[zone_label]
```

iii. Not identified. CONVERSION_NOTES Step 6 claims "single-pass per-session processing" as a speedup, which is true at the file level.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of work produce nothing used downstream:
- Three behavior series — `teleport`, `autoreward`, `scanning` — are read in full (data *and* timestamps) and never referenced afterwards. `Reward`'s `data` values are read and only its timestamps used.
- `_ts_data` reads the timestamps vector of all 11 series; only one is used.
- `contiguous_segments()` is defined but never called anywhere — dead code.
- `trial_zone_labels` is accumulated per session and then discarded (never returned or stored).
- `rate` has a `1/np.median(np.diff(timestamps))` fallback branch that can never be reached (it is only evaluated when `deconv.rate` is None, in which case `timestamps` was already built from that same missing rate).
- `sess['session_id']` is computed and returned but never written into the output dictionary, so session identity is lost.
- `--show-processing` is accepted as an argument and silently ignored; no plots are produced (acknowledged in Step 7: "`--show-processing` flag currently does not generate plots yet").
- By far the largest waste is carrying the ~56% of ROIs that `iscell` marks as not cells and the ~27% of samples that are inter-trial interval: these inflate `converted_data.pkl` to 25.2 GB, add minutes of load time to every verification and training run, and push several sessions past the decoder's neuron limit so that it random-projects them.

ii.
```python
def contiguous_segments(mask):   # never called
    idx = np.flatnonzero(mask)
    ...
```
```python
for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
    behavior[k] = _ts_data(bts[k])
```
```python
        sess_neural, sess_input, sess_output = [], [], []
        reward_outcomes = []
        trial_zone_labels = []        # accumulated, never used
        ...
            trial_zone_labels.append(zone_label)
```
```python
ap.add_argument('--show-processing', action='store_true', help='Placeholder; save processing plots for up to 2 sessions')
```
```python
        return {
            'subject': subj,
            'session_id': sess_id,   # never placed in the output dict
            ...
        }
```

iii. Not identified by the AI as waste. The `--show-processing` gap is the only item acknowledged: CONVERSION_NOTES Step 6, "`--show-processing` is currently a placeholder flag and will need visualization support if required by later checks", and Step 7, "`--show-processing` flag currently does not generate plots yet; this remains to be implemented if required." The oversized pickle is noted only as a fact ("`converted_data.pkl`: 25211315179 bytes"), not as a problem.
