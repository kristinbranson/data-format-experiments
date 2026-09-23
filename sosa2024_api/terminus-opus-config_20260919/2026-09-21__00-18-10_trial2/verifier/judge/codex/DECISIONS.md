# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI gathers every NWB file under `/app/data/sub-*/*.nwb`, processes each file as one session, and loads both behavior and ophys arrays from each file with `pynwb.NWBHDF5IO`. Trials are then derived from the loaded session arrays.

ii. 
```python
files = sorted(glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
...
with NWBHDF5IO(fn, 'r', load_namespaces=True) as io:
    nwb = io.read()
    ...
    beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
    ...
    oph = nwb.processing['ophys'].data_interfaces
```

iii. In `CONVERSION_NOTES.md`, the AI says the release contains 11 subject directories and 152 NWB files, one per session, and that the NWB files already contain the behavior series aligned to imaging frames, so reading every NWB file is the full dataset.

## 1-b. How are the data split into subjects?

i. Subjects are identified from each NWB file's `subject.subject_id`, then deduplicated and sorted into the top-level `subjects` list. Each session gets a `subject_idx` pointing into that list.

ii. 
```python
subject = nwb.subject.subject_id
...
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. The AI's notes say the dataset has 11 mice (`m3 ... m19`) and that the NWB metadata are consistent with the subject-directory layout, so using the embedded subject id is sufficient.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The script processes files independently and appends one session entry per file to `data['neural']`, `data['input']`, and `data['output']`.

ii. 
```python
files = sorted(glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
...
for i, res in enumerate(pool.imap(_worker, jobs)):
    results.append(res)
...
'neural': [r['neural'] for r in results],
'input': [r['input'] for r in results],
'output': [r['output'] for r in results],
```

iii. The notes say the filename pattern is `sub-<mouse>_ses-<day>_behavior+ophys.nwb`, with one file per session, so the AI kept that one-file/one-session organization.

## 1-d. How are the data split into trials?

i. Trials are defined as frames from `trial_start` up to but not including `teleport`. The AI finds all trial-start indices and teleport indices, then slices `[start:stop)` for every trial.

ii. 
```python
trial_start = np.asarray(beh['trial_start'].data[:], dtype=np.float64)
teleport = np.asarray(beh['teleport'].data[:], dtype=np.float64)
...
starts = np.where(trial_start > 0)[0]
teles = np.where(teleport > 0)[0]
assert len(starts) == len(teles) and np.all(teles > starts)
...
for i, (a, b) in enumerate(zip(starts, teles)):
    sl = slice(a, b)
```

iii. The AI's notes say the paper defines a trial as track entry to track exit, with teleport/ITI excluded, and explicitly state that it chose `[start, stop)` to keep behavior and neural slices aligned.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials flagged as lick-sensor failures. A trial is removed if more than 30% of its frames have cumulative lick count greater than 2. It does not apply a minimum-length filter.

ii. 
```python
lick_err = np.array([(lick[a:b] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC
                     for a, b in zip(starts, teles)])
...
for i, (a, b) in enumerate(zip(starts, teles)):
    if lick_err[i]:
        continue
```

iii. In the notes, the AI cites the paper's rule and says this reproduces the 81 lick-error trials removed in the methods, so it treated that as the main trial-quality control.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` arrays are derived from raw `Fluorescence` (`F`) and `Neuropil` (`Fneu`) ROI traces in the NWB ophys processing group, after restricting to curated `iscell` ROIs. They are not taken from the NWB `Deconvolved` series.

ii. 
```python
Fseries = oph['Fluorescence'].roi_response_series
Nseries = oph['Neuropil'].roi_response_series
...
F = np.concatenate([np.asarray(Fseries[k].data[:nframes, :], dtype=np.float32).T
                    for k in plane_keys], axis=0)
Fneu = np.concatenate([np.asarray(Nseries[k].data[:nframes, :], dtype=np.float32).T
                       for k in plane_keys], axis=0)
...
keep = np.where(S['iscell'])[0]
F = S['F'][keep]
Fneu = S['Fneu'][keep]
```

iii. The notes explicitly say the stored NWB `Deconvolved` traces are suite2p spikes from raw fluorescence and do not match the paper's analysis signal, so the AI chose to recompute the paper-style event signal from `F` and `Fneu`.

## 2-b. How is the `neural` data processed?

i. The AI recomputes per-trial dF/F and then deconvolves it into events. It keeps only on-trial frames, subtracts `0.7 * Fneu`, adds back each trial's mean neuropil, computes a maximin baseline (Gaussian sigma 15 frames, then 300-frame min and 300-frame max filters), forms `(F - F0) / |F0|`, smooths dF/F with Gaussian sigma 2 frames, and runs OASIS with `tau = 0.7`.

ii. 
```python
for s, e in zip(starts, teles):
    f_[:, s:e] = F[:, s:e]
    fneu_[:, s:e] = Fneu[:, s:e]
...
f_ -= NEU_COEF * fneu_
...
for s, e in zip(starts, teles):
    f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
    seg = nansmooth_nd(f_[:, s:e], [0, BASELINE_SMOOTH])
    seg = ndi.minimum_filter1d(seg, BASELINE_WIN, axis=-1)
    seg = ndi.maximum_filter1d(seg, BASELINE_WIN, axis=-1)
    flow[:, s:e] = seg
...
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
...
for s, e in zip(starts, teles):
    dff[:, s:e] = nansmooth1d(dff[:, s:e], DFF_SMOOTH, axis=1)
    events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, fs)
```

iii. The notes say this was intended as a direct port of `reward_relative/preprocessing.py::dff`, except that the AI chose a simpler `[start, stop)` trial window for all sessions rather than reproducing the reference's special teleport-handling cases.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first keeps only ROIs with `iscell == 1`, then removes putative interneurons defined by correlation between dF/F and running speed greater than 0.5.

ii. 
```python
keep = np.where(S['iscell'])[0]
F = S['F'][keep]
Fneu = S['Fneu'][keep]
...
corr_speed = np.where(denom > 0, (Dz @ spz) / np.maximum(denom, 1e-12), 0.0)
is_interneuron = corr_speed > INTERNEURON_R
cells = np.where(~is_interneuron)[0]
...
neu = events[np.ix_(cells, np.arange(a, b))].astype(np.float32)
```

iii. The notes cite the paper's neuron curation: suite2p manual `iscell` labels plus exclusion of putative interneurons with Pearson `r(dF/F, speed) > 0.5`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to trial start by emitting one neural matrix per `[trial_start, teleport)` slice. No extra offset is added.

ii. 
```python
for i, (a, b) in enumerate(zip(starts, teles)):
    ...
    neu = events[np.ix_(cells, np.arange(a, b))].astype(np.float32)
    neural.append(neu)
```

iii. The notes say the alignment event is "trial start (entry to the linear track at position 0 cm)" and that simply splitting the already frame-aligned session arrays is enough to align the trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native imaging frame rate, about 15.5078 Hz per plane, or 64.484 ms per sample. The AI does not rebin or resample the traces.

ii. 
```python
fs = S['rate'] / S['n_planes']
...
'time_bin_size': 1000.0 / (15.5078125),
'imaging_rate_hz': 15.5078125,
```

iii. The notes say behavior is already aligned 1:1 to imaging frames in the NWB release, so no temporal rebinning was needed and keeping the native frame period best matches the reference workflow.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps loaded from `behavior/position.timestamps`, which the AI treats as the session frame-time axis.

ii. 
```python
t = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
...
tt = (t[sl] - t[a]).astype(np.float32)
inp[0] = tt
```

iii. The notes say all behavior time series share the same timestamps and are already aligned to imaging frames, so using `position.timestamps` was a convenient representative choice.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The AI subtracts the timestamp at the first frame of the trial from every timestamp in the trial slice.

ii. 
```python
tt = (t[sl] - t[a]).astype(np.float32)
inp[0] = tt
```

iii. The notes describe this input simply as "t - t[trial_start]".

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the exact same frame indices as the neural slice for each trial, so alignment is inherited from the NWB frame grid.

ii. 
```python
sl = slice(a, b)
tt = (t[sl] - t[a]).astype(np.float32)
...
neu = events[np.ix_(cells, np.arange(a, b))].astype(np.float32)
```

iii. The AI's notes state that the NWB behavior series are already interpolated onto imaging frames 1:1, so no extra alignment code was necessary.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `behavior/environment` time series.

ii. 
```python
env = np.asarray(beh['environment'].data[:], dtype=np.float64)
...
morph = np.array([np.unique(env[a:b])[0] for a, b in zip(starts, teles)])
```

iii. The notes identify NWB `environment` as the binary ENV1/ENV2 variable and say it is constant within each trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI takes the unique environment value within each trial, asserts that it is 0 or 1, and then broadcasts that constant value across all timepoints in the trial input matrix.

ii. 
```python
morph = np.array([np.unique(env[a:b])[0] for a, b in zip(starts, teles)])
assert np.all(np.isin(morph, [0.0, 1.0]))
...
inp[1] = morph[i]
```

iii. The notes say the variable is constant within every trial, so collapsing to one per-trial value and broadcasting it over time preserves the information while fitting the required `(4, T)` input shape.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The AI derives trial number from the within-session trial loop index, not from the NWB `trial number` time series.

ii. 
```python
for i, (a, b) in enumerate(zip(starts, teles)):
    ...
    inp[2] = i
```

iii. The notes say the output should be a 0-based within-session index and that the trial segmentation itself comes from `trial_start` and `teleport`.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. There is no extra processing beyond assigning the loop index and broadcasting it across the trial's timepoints.

ii. 
```python
inp[2] = i
```

iii. The notes describe this as a per-trial variable broadcast across time to fit the common `(4, T)` format.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the per-trial `isreward` labels, and those labels are computed from two raw variables: the `Reward` event timestamps (converted into a framewise reward series) and the `reward_zone` behavior signal.

ii. 
```python
reward_t = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
...
reward = np.zeros(nframes, dtype=np.float64)
if len(reward_t):
    ridx = np.searchsorted(t, reward_t)
    ridx = np.clip(ridx, 0, nframes - 1)
    reward[ridx] = 1.0
...
isreward = np.array([bool((reward[a:b] > 0).any() and (rzone[a:b] > 0).any())
                     for a, b in zip(starts, teles)])
```

iii. The notes say the AI followed the reference `get_trial_types` logic, where a trial is rewarded only if reward was delivered and the reward zone was entered.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial `i > 0`, the AI uses the binary reward outcome of trial `i-1` and broadcasts it across the whole trial. For the first trial, it sets the previous-trial outcome to `1.0` rather than `0`, based on the experiment's warm-up trials.

ii. 
```python
inp[3] = 1.0 if i == 0 else float(isreward[i - 1])
```

iii. The notes justify the first-trial special case by saying each imaging session was preceded by 30 rewarded warm-up trials, so the unobserved immediately previous trial was more plausibly rewarded than omitted.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The continuous distance is derived from `behavior/position` and the active reward-zone identity for that trial. The active zone is inferred primarily from the scene name in `nwb.identifier`, with `behavior/reward_zone` and position used as an empirical cross-check and fallback.

ii. 
```python
scene = nwb.identifier.split('/')[-1]
...
zone_labels = scene_reward_zones(S['scene'], ntrials_raw)
emp = empirical_reward_zones(pos, rzone, starts, teles)
...
zl = zone_labels[i]
z0, z1 = REWARD_ZONES[zl]
d = signed_distance_to_zone(pos[sl], z0, z1)
```

iii. The notes say this mirrors the reference code's scene-based reward-zone logic, but adds a verification pass against observed reward-zone entries and falls back to the observed zone if the two disagree.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the AI computes signed distance to the nearest edge of the active reward zone: negative before the zone, zero inside it, and positive after it.

ii. 
```python
def signed_distance_to_zone(pos, zone_start, zone_end):
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d
```

iii. The notes map this directly to the decoder task's "distance to any location in the reward zone" variable and to the linear relative-position construction in the reference analysis code.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI bins signed distance into 7 categories: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50` cm.

ii. 
```python
def discretize_distance(d):
    out = np.full(d.shape, 3, dtype=np.int8)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```

iii. The notes say these thresholds were taken directly from the decoder task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `position[trial_slice]` using the same `[start:stop)` trial frame indices used for the neural data.

ii. 
```python
sl = slice(a, b)
d = signed_distance_to_zone(pos[sl], z0, z1)
...
neu = events[np.ix_(cells, np.arange(a, b))].astype(np.float32)
```

iii. The notes say both behavior and neural arrays are already on the same imaging-frame grid, so shared slicing is the alignment mechanism.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the `behavior/position` time series.

ii. 
```python
pos = np.asarray(beh['position'].data[:], dtype=np.float64)
...
out[1] = discretize_position(pos[sl])
```

iii. The notes treat `position` as the direct VR track-position signal in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI clips position to the 450 cm track and discretizes it into five 90 cm bins using floor division.

ii. 
```python
def discretize_position(pos):
    p = np.clip(pos, 0.0, TRACK_LENGTH - 1e-9)
    return np.floor(p / 90.0).astype(np.int8)
```

iii. The notes say the track is 450 cm long, so 5 equal bins are 90 cm wide; clipping keeps boundary samples inside the intended range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The categories are `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm, represented as integer bins 0 through 4.

ii. 
```python
def discretize_position(pos):
    p = np.clip(pos, 0.0, TRACK_LENGTH - 1e-9)
    return np.floor(p / 90.0).astype(np.int8)
```

iii. The notes say these are the task-specified five equal-sized bins spanning the 450 cm corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same `[start:stop)` frame slice as the neural data for each trial.

ii. 
```python
sl = slice(a, b)
out[1] = discretize_position(pos[sl])
...
neu = events[np.ix_(cells, np.arange(a, b))].astype(np.float32)
```

iii. The notes say behavior was already aligned to imaging frames in the NWB data, so no extra synchronization step was required.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `behavior/lick` time series.

ii. 
```python
lick = np.asarray(beh['lick'].data[:], dtype=np.float64)
...
out[3] = (lick[sl] > 0).astype(np.int8)
```

iii. The notes describe the NWB `lick` variable as cumulative lick count per frame and say it is converted to a binary lick/no-lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes the lick signal at `> 0`.

ii. 
```python
out[3] = (lick[sl] > 0).astype(np.int8)
```

iii. The notes justify this as matching the decoder task's binary lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is sliced with the same per-trial frame indices as the neural data.

ii. 
```python
sl = slice(a, b)
out[3] = (lick[sl] > 0).astype(np.int8)
...
neu = events[np.ix_(cells, np.arange(a, b))].astype(np.float32)
```

iii. The notes say the lick series is already sampled on the imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The reward-zone location is derived primarily from the scene name embedded in `nwb.identifier`; `behavior/reward_zone` plus position are used to verify or override that label when they disagree.

ii. 
```python
scene = nwb.identifier.split('/')[-1]
...
zone_labels = scene_reward_zones(S['scene'], ntrials_raw)
emp = empirical_reward_zones(pos, rzone, starts, teles)
...
out[4] = ZONE_TO_IDX[zl]
```

iii. The notes say the reference code defines reward zones from the scene and switch trial, and that the empirical reward-zone entry positions agreed everywhere in the release, so the check was mainly a safeguard.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses fixed scenes like `Env1_LocationB` and switch scenes like `...A_to_...B`, assigns zone A/B/C labels per trial with the change at trial index 30, optionally replaces scene-derived labels with empirically inferred labels where available, and maps A/B/C to 0/1/2.

ii. 
```python
def scene_reward_zones(scene, ntrials, change_trial=CHANGE_TRIAL):
    m = re.search(r'^Env\\d_Location([ABC])$', scene)
    if m:
        return np.array([m.group(1)] * ntrials)
    m = re.search(r'([ABC])_to_(?:Env\\d_)?(?:Location)?([ABC])$', scene)
    if m:
        n0 = min(change_trial, ntrials)
        return np.array([m.group(1)] * n0 + [m.group(2)] * (ntrials - n0))
...
out[4] = ZONE_TO_IDX[zl]
```

iii. The notes say this was chosen to mirror `behavior.get_reward_zones` in the reference repository and to preserve the paper's fixed-vs-switch reward-zone logic.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward` event timestamps and the `reward_zone` behavior series. The AI first builds a framewise reward indicator from the timestamps, then defines a trial as rewarded only if reward delivery and reward-zone entry both occurred during that trial.

ii. 
```python
reward_t = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
...
reward = np.zeros(nframes, dtype=np.float64)
if len(reward_t):
    ridx = np.searchsorted(t, reward_t)
    ridx = np.clip(ridx, 0, nframes - 1)
    reward[ridx] = 1.0
...
isreward = np.array([bool((reward[a:b] > 0).any() and (rzone[a:b] > 0).any())
                     for a, b in zip(starts, teles)])
```

iii. The notes say this matches the paper/reference definition of per-trial `isreward`, not just raw reward delivery.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The AI maps reward timestamps onto frame indices with `searchsorted`, creates a framewise reward indicator, computes one Boolean reward label per trial using reward delivery AND reward-zone entry, and broadcasts that binary label over all timepoints of the trial.

ii. 
```python
if len(reward_t):
    ridx = np.searchsorted(t, reward_t)
    ridx = np.clip(ridx, 0, nframes - 1)
    reward[ridx] = 1.0
...
isreward = np.array([bool((reward[a:b] > 0).any() and (rzone[a:b] > 0).any())
                     for a, b in zip(starts, teles)])
...
out[5] = int(isreward[i])
```

iii. The notes say this reproduces the reference `get_trial_types` logic and reflects the task's per-trial rewarded-vs-omitted label.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles irregularities by targeted fixes plus assertions. It truncates ophys arrays to the behavior length, clips reward timestamps into valid frame bounds, uses scene-derived reward-zone labels for omission trials where empirical zone entry is missing, and falls back to empirically observed zone labels if they disagree with scene-derived labels. Unexpected structure (bad trial indices, unexpected environment values, NaNs in emitted neural data) raises assertions instead of being silently repaired.

ii. 
```python
nframes = len(t)
F = np.concatenate([np.asarray(Fseries[k].data[:nframes, :], dtype=np.float32).T
                    for k in plane_keys], axis=0)
...
if len(reward_t):
    ridx = np.searchsorted(t, reward_t)
    ridx = np.clip(ridx, 0, nframes - 1)
    reward[ridx] = 1.0
...
emp = empirical_reward_zones(pos, rzone, starts, teles)
valid = emp != '?'
...
if zone_agree < 1.0:
    fixed = zone_labels.copy()
    fixed[valid] = emp[valid]
    zone_labels = fixed
...
assert len(starts) == len(teles) and np.all(teles > starts)
assert np.all(np.isin(morph, [0.0, 1.0]))
assert not np.isnan(neu).any()
```

iii. The notes mention a real dataset issue with 10 sessions whose ophys arrays are one frame longer than behavior, omission trials with no reward-zone trigger, and the zone-agreement cross-check; the AI chose to correct those concrete issues but leave harder unexpected cases as hard failures.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading large NWB arrays and the dF/F-to-events pipeline, especially the per-trial baseline filtering and OASIS deconvolution in `compute_dff_events`. Multiprocessing is used to reduce wall-clock time across sessions.

ii. 
```python
with NWBHDF5IO(fn, 'r', load_namespaces=True) as io:
    nwb = io.read()
    ...
for s, e in zip(starts, teles):
    seg = nansmooth_nd(f_[:, s:e], [0, BASELINE_SMOOTH])
    seg = ndi.minimum_filter1d(seg, BASELINE_WIN, axis=-1)
    seg = ndi.maximum_filter1d(seg, BASELINE_WIN, axis=-1)
...
for s, e in zip(starts, teles):
    events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, fs)
```

iii. The notes explicitly estimate NWB load at roughly 0.5-1.2 s/session and dF/F + OASIS as the dominant compute stage, which is why the AI added a 12-process worker pool.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining non-vectorized loops are the per-trial loops in `compute_dff_events`, the per-trial construction loop in `process_session`, and the per-trial empirical reward-zone scan. The AI already vectorized across neurons within a session, but these loops are still sequential over trials.

ii. 
```python
for s, e in zip(starts, teles):
    f_[:, s:e] = F[:, s:e]
    fneu_[:, s:e] = Fneu[:, s:e]
...
for s, e in zip(starts, teles):
    ...
    flow[:, s:e] = seg
...
for i, (a, b) in enumerate(zip(starts, teles)):
    ...
    neural.append(neu)
    inputs.append(inp)
    outputs.append(out)
...
for a, b in zip(starts, teles):
    idx = np.where(rzone[a:b] > 0)[0]
```

iii. The notes say the cell-level math was already vectorized and only the natural per-trial structure remains; further vectorization would mostly target repeated trial loops rather than per-cell operations.

## 13-c. What processing does the code repeat multiple times?

i. The code loops over trials multiple times inside `compute_dff_events` (copying trial frames, then computing trial baselines, then smoothing/deconvolving trial segments). It also derives reward zones twice per session: once from the scene name and again empirically from `reward_zone` entries for validation/fallback. If plotting is enabled, some derived quantities are recomputed again for visualization.

ii. 
```python
for s, e in zip(starts, teles):
    f_[:, s:e] = F[:, s:e]
    fneu_[:, s:e] = Fneu[:, s:e]
...
for s, e in zip(starts, teles):
    ...
    flow[:, s:e] = seg
...
for s, e in zip(starts, teles):
    dff[:, s:e] = nansmooth1d(dff[:, s:e], DFF_SMOOTH, axis=1)
    events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, fs)
...
zone_labels = scene_reward_zones(S['scene'], ntrials_raw)
emp = empirical_reward_zones(pos, rzone, starts, teles)
```

iii. The notes describe the empirical reward-zone pass as a validation safeguard rather than the primary label source, and the plotting recomputation exists only for `--show-processing`.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes full dF/F traces even though only deconvolved events are saved; dF/F is kept mainly to support interneuron filtering and optional plots. It also computes empirical zone-agreement statistics and extensive per-session metadata that are not used by the downstream decoder, and optional plotting produces additional derived arrays only for inspection.

ii. 
```python
dff, events = compute_dff_events(F, Fneu, starts, teles, fs)
...
Dz = D - D.mean(axis=1, keepdims=True)
...
emp = empirical_reward_zones(pos, rzone, starts, teles)
valid = emp != '?'
zone_agree = float(np.mean(emp[valid] == zone_labels[valid])) if valid.any() else np.nan
...
info = dict(... zone_agreement=zone_agree, ... t_load=t_load, t_dff=t_dff)
...
if show_processing:
    plot_processing(...)
```

iii. The notes frame these as validation and documentation costs: they help prove the conversion is correct, but they are not required by the final decoder input format itself.
