# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by globbing `DATA_ROOT/sub-*/\*.nwb` (sorted). Each NWB file is loaded with `pynwb.NWBHDF5IO`. From each file, the AI extracts behavior time series (position, speed, lick, reward_zone, environment, trial_start, teleport, Reward timestamps), ophys data (Fluorescence F and Neuropil Fneu per plane, iscell from ImageSegmentation), and metadata (subject_id, scene name from identifier, rate, n_planes). All 152 NWB files across 11 subjects are processed.

ii.
```python
files = sorted(glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
...
def load_session(fn):
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(fn, 'r', load_namespaces=True) as io:
        nwb = io.read()
        subject = nwb.subject.subject_id
        scene = nwb.identifier.split('/')[-1]
        beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
        t = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
        pos = np.asarray(beh['position'].data[:], dtype=np.float64)
        ...
        oph = nwb.processing['ophys'].data_interfaces
        Fseries = oph['Fluorescence'].roi_response_series
        Nseries = oph['Neuropil'].roi_response_series
        ...
        F = np.concatenate([np.asarray(Fseries[k].data[:nframes, :], dtype=np.float32).T
                            for k in plane_keys], axis=0)
        ...
```

iii. The AI documented that the data directory contains 11 subject directories with 152 NWB files total. The glob pattern ensures all files are found. Loading uses `pynwb` as required by the instructions.

## 1-b. How are the data split into subjects?

i. Subjects are identified from `nwb.subject.subject_id` within each NWB file. After processing all sessions, unique subjects are collected as a sorted set.

ii.
```python
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. Each NWB file contains a subject identifier (e.g., m3, m11). The AI extracts this directly from the NWB metadata rather than parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI processes each file independently via `process_session()` and collects results into the session-indexed data structure.

ii.
```python
files = sorted(glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
...
for i, job in enumerate(jobs):
    res = _worker(job)
    results.append(res)
```

iii. The one-NWB-file-per-session structure is standard for this dataset format. The AI noted 152 total sessions across 11 subjects.

## 1-d. How are the data split into trials?

i. Trials are defined by `trial_start` (positive values mark trial onset) and `teleport` (positive values mark trial end). The AI finds the indices where `trial_start > 0` for starts and `teleport > 0` for ends. Each trial spans `[start, teleport)`.

ii.
```python
starts = np.where(trial_start > 0)[0]
teles = np.where(teleport > 0)[0]
assert len(starts) == len(teles) and np.all(teles > starts), f'bad trial indices in {fn}'
...
for i, (a, b) in enumerate(zip(starts, teles)):
    sl = slice(a, b)
```

iii. The CONVERSION_NOTES document that `trial_start` and `teleport` binary series define trial boundaries, consistent with the reference code's `sess.trial_start_inds` and `sess.teleport_inds`.

## 1-e. How are trials filtered based on quality controls?

i. Trials with lick-sensor errors are dropped. The criterion is: if more than 30% of frames in a trial have cumulative lick count > 2, the trial is flagged as a lick-sensor error and excluded. This matches the paper's rule exactly (81 out of 12,376 trials removed).

ii.
```python
LICK_ERROR_FRAC = 0.30
LICK_ERROR_COUNT = 2
...
lick_err = np.array([(lick[a:b] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC
                     for a, b in zip(starts, teles)])
...
for i, (a, b) in enumerate(zip(starts, teles)):
    if lick_err[i]:
        continue
```

iii. The AI documented in CONVERSION_NOTES that this rule reproduced exactly the 81 trials the paper removed, confirming correct implementation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p `Fluorescence` (F) and `Neuropil` (Fneu) traces stored in the NWB ophys processing module. The AI explicitly does NOT use the NWB `Deconvolved` series, noting that it is suite2p's own deconvolution from raw F, not the paper's dF/F-based events.

ii.
```python
oph = nwb.processing['ophys'].data_interfaces
Fseries = oph['Fluorescence'].roi_response_series
Nseries = oph['Neuropil'].roi_response_series
...
F = np.concatenate([np.asarray(Fseries[k].data[:nframes, :], dtype=np.float32).T
                    for k in plane_keys], axis=0)
Fneu = np.concatenate([np.asarray(Nseries[k].data[:nframes, :], dtype=np.float32).T
                       for k in plane_keys], axis=0)
```

iii. The AI's CONVERSION_NOTES extensively document the distinction between the NWB Deconvolved series and the paper's own events, and justify recomputing dF/F + OASIS from F and Fneu.

## 2-b. How is the `neural` data processed?

i. The AI recomputes dF/F and deconvolved events following the reference `preprocessing.dff`:
1. Restrict to on-trial frames (start to teleport); inter-trial frames are NaN.
2. Subtract 0.7 * Fneu from F (neuropil correction).
3. Add back per-trial mean neuropil so dF/F denominators are not tiny.
4. Maximin baseline: Gaussian smooth with sigma [0, 15 frames], then 300-frame minimum filter, then 300-frame maximum filter.
5. dF/F = (F - baseline) / |baseline|.
6. Gaussian smooth dF/F with sigma 2 frames.
7. OASIS deconvolution (suite2p dcnv.oasis, tau=0.7, fs=rate/n_planes).

The AI uses `keep_teleports=False` for all sessions (does not account for per-session teleport imaging metadata).

ii.
```python
def compute_dff_events(F, Fneu, starts, teles, fs):
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, teles):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]
    f_ -= NEU_COEF * fneu_
    ...
    for s, e in zip(starts, teles):
        f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        seg = nansmooth_nd(f_[:, s:e], [0, BASELINE_SMOOTH])
        seg = ndi.minimum_filter1d(seg, BASELINE_WIN, axis=-1)
        seg = ndi.maximum_filter1d(seg, BASELINE_WIN, axis=-1)
        flow[:, s:e] = seg
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    for s, e in zip(starts, teles):
        dff[:, s:e] = nansmooth1d(dff[:, s:e], DFF_SMOOTH, axis=1)
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, fs)
```

iii. The AI documented in CONVERSION_NOTES that this follows `reward_relative/preprocessing.py::dff` exactly, including the same parameters (neu_coef=0.7, tau=0.7, baseline_method='maximin'). The AI also noted that `keep_teleports=False` is used, but did not implement the per-session teleport metadata logic.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied:
1. Suite2p `iscell == 1` from ImageSegmentation (manual curation).
2. Putative interneurons excluded: cells with Pearson r(dF/F, speed) > 0.5 are removed.

ii.
```python
ps = oph['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
iscell = np.asarray(ps['iscell'].data[:])[:, 0] == 1
...
keep = np.where(S['iscell'])[0]
F = S['F'][keep]
Fneu = S['Fneu'][keep]
...
corr_speed = np.where(denom > 0, (Dz @ spz) / np.maximum(denom, 1e-12), 0.0)
is_interneuron = corr_speed > INTERNEURON_R
cells = np.where(~is_interneuron)[0]
```

iii. The AI noted that suite2p iscell provides the manual curation, and the interneuron exclusion matches the paper's rule (Pearson r > 0.5). The AI's vectorized correlation implementation uses a matrix-vector product rather than a per-cell loop.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instructions say to align to trial start. Since the neural data and behavior data share the same frame indices, and trials are defined by `trial_start` indices, the neural data for each trial is simply the slice `events[cells, start:teleport]`. No additional alignment processing is needed.

ii.
```python
neu = events[np.ix_(cells, np.arange(a, b))].astype(np.float32)
```

iii. The AI documented that behavior is already interpolated onto imaging frame times in the NWB files (1:1 correspondence), so alignment is trivial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame period: ~64.484 ms (1000 / 15.5078125 Hz). No temporal rebinning is applied. This is consistent across all sessions.

ii.
```python
'time_bin_size': 1000.0 / (15.5078125),  # ms per imaging frame (per plane)
```

iii. The AI noted that behavior in the NWB is already 1:1 with imaging frames, so no rebinning is needed.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` behavior time series timestamps (which are the same for all behavior variables).

ii.
```python
t = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
...
tt = (t[sl] - t[a]).astype(np.float32)
inp[0] = tt
```

iii. All behavior time series share the same timestamps; the AI used `position` timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial start timestamp is subtracted from each frame's timestamp within the trial, giving seconds from trial start.

ii.
```python
tt = (t[sl] - t[a]).astype(np.float32)
inp[0] = tt
```

iii. Straightforward subtraction of the first timestamp in the trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The behavior timestamps and neural data share the same frame indices (1:1 correspondence in the NWB), so the same slice `[a:b]` is used for both. No additional alignment is needed.

ii. Same indexing: `sl = slice(a, b)` used for both neural and behavior data.

iii. Verified by the AI through timestamp assertions and processing plots.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env = np.asarray(beh['environment'].data[:], dtype=np.float64)
...
morph = np.array([np.unique(env[a:b])[0] for a, b in zip(starts, teles)])
...
inp[1] = morph[i]
```

iii. The AI noted that `environment` is constant within each trial (0 = ENV1, 1 = ENV2), confirmed by assertion.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial environment value is extracted as the unique value of `environment` within the trial. It is broadcast as a constant across all timepoints in the trial.

ii.
```python
morph = np.array([np.unique(env[a:b])[0] for a, b in zip(starts, teles)])
assert np.all(np.isin(morph, [0.0, 1.0]))
inp[1] = morph[i]
```

iii. The AI verified that the environment is always 0 or 1 and constant within trials.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index over trials within a session (0-based). It is NOT derived from the NWB `trial number` behavior time series.

ii.
```python
for i, (a, b) in enumerate(zip(starts, teles)):
    ...
    inp[2] = i
```

iii. The AI uses the sequential trial index within the session. Note that filtered (lick-error) trials are skipped but the index `i` still reflects the original trial number in the session, not a re-indexed count.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant (broadcast) across all timepoints within a trial.

ii.
```python
inp[2] = i
```

iii. Simple assignment of the 0-based trial index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and `reward_zone` behavior time series. `isreward` for each trial is True if both a reward was delivered AND the reward zone was entered during that trial.

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

iii. The AI follows the reference code's `get_trial_types` definition of `isreward`: both reward delivery AND reward zone entry must occur in the trial. This correctly handles omission trials where the mouse enters the zone but no reward is delivered.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's `isreward` value is used. For the first trial, the value is set to 1 (rewarded), based on the reasoning that imaging sessions are preceded by 30 rewarded warm-up trials.

ii.
```python
inp[3] = 1.0 if i == 0 else float(isreward[i - 1])
```

iii. The AI documented the decision to set the first trial's previous outcome to 1 in CONVERSION_NOTES Key Decision 8.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` behavior time series and the per-trial reward zone location. The reward zone is determined from the VR scene name parsed from the NWB identifier, with switch at trial index 30 for switch sessions. An empirical check against `reward_zone` entry positions is also performed.

ii.
```python
def scene_reward_zones(scene, ntrials, change_trial=CHANGE_TRIAL):
    m = re.search(r'^Env\d_Location([ABC])$', scene)
    if m:
        return np.array([m.group(1)] * ntrials)
    m = re.search(r'([ABC])_to_(?:Env\d_)?(?:Location)?([ABC])$', scene)
    if m:
        n0 = min(change_trial, ntrials)
        return np.array([m.group(1)] * n0 + [m.group(2)] * (ntrials - n0))
...
zone_labels = scene_reward_zones(S['scene'], ntrials_raw)
emp = empirical_reward_zones(pos, rzone, starts, teles)
```

iii. The AI ported the reference code's `behavior.get_reward_zones` logic, including the switch at trial 30 and the zone dictionary (A=[80,130], B=[200,250], C=[320,370]).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, compute the signed distance from position to the nearest edge of the active reward zone. Distance is 0 inside the zone, negative before, positive after.

ii.
```python
def signed_distance_to_zone(pos, zone_start, zone_end):
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d
...
d = signed_distance_to_zone(pos[sl], z0, z1)
```

iii. This matches the reference code's definition of reward-relative position.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using explicit conditional assignment: bin 3 is the default (d==0, inside zone), bins 0-2 for negative distances, bins 4-6 for positive distances. Boundaries: -50, -10, 0, +10, +50.

ii.
```python
def discretize_distance(d):
    out = np.full(d.shape, 3, dtype=np.int8)  # 3 == inside the zone (d == 0)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```

iii. The bin edges match the instructions' specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices as neural data (same slice `[a:b]`). No additional alignment needed.

ii.
```python
d = signed_distance_to_zone(pos[sl], z0, z1)
out[0] = discretize_distance(d)
```

iii. 1:1 correspondence between behavior and neural frames.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = np.asarray(beh['position'].data[:], dtype=np.float64)
...
out[1] = discretize_position(pos[sl])
```

iii. The `position` variable records the animal's position in cm along the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450) and then divided by 90 to create 5 equal bins. The clipping handles edge cases where position slightly exceeds track bounds.

ii.
```python
def discretize_position(pos):
    p = np.clip(pos, 0.0, TRACK_LENGTH - 1e-9)
    return np.floor(p / 90.0).astype(np.int8)
```

iii. The track is 450 cm, so 5 bins of 90 cm each. Clipping ensures no out-of-range bin indices.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 equal-sized bins of 90 cm each: [0,90), [90,180), [180,270), [270,360), [360,450). Implemented via `floor(clip(pos, 0, 449.999) / 90)`.

ii.
```python
p = np.clip(pos, 0.0, TRACK_LENGTH - 1e-9)
return np.floor(p / 90.0).astype(np.int8)
```

iii. Matches the instructions' specification of 5 equal-sized bins spanning 450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data. No additional alignment.

ii. Same slice `pos[sl]` where `sl = slice(a, b)`.

iii. 1:1 correspondence.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(beh['lick'].data[:], dtype=np.float64)
...
out[3] = (lick[sl] > 0).astype(np.int8)
```

iii. The `lick` variable records cumulative lick count per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0. Trials with lick-sensor errors are dropped entirely (not included in the output).

ii.
```python
out[3] = (lick[sl] > 0).astype(np.int8)
```

iii. The instructions specify binary output. The reference code also binarizes licks (`licks > 0 -> 1`).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data. No additional alignment.

ii. Same slice `lick[sl]`.

iii. 1:1 correspondence.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the VR scene name in the NWB identifier (parsed via regex), with an empirical cross-check against the `reward_zone` and `position` behavior time series. The scene name encodes which zone is active, and for switch sessions, the zone changes at trial index 30.

ii.
```python
scene = nwb.identifier.split('/')[-1]
...
zone_labels = scene_reward_zones(S['scene'], ntrials_raw)
emp = empirical_reward_zones(pos, rzone, starts, teles)
...
out[4] = ZONE_TO_IDX[zl]  # A=0, B=1, C=2
```

iii. The AI ported `behavior.get_reward_zones` from the reference code and verified 100% agreement with empirically observed zone entries across all 152 sessions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to extract zone labels (A, B, or C). For switch sessions (`X_to_Y` pattern), zone X is used for the first 30 trials and zone Y for the rest. The zone label is mapped to an integer (A=0, B=1, C=2) and broadcast across all timepoints in the trial.

ii.
```python
ZONE_TO_IDX = {'A': 0, 'B': 1, 'C': 2}
CHANGE_TRIAL = 30
...
def scene_reward_zones(scene, ntrials, change_trial=CHANGE_TRIAL):
    m = re.search(r'^Env\d_Location([ABC])$', scene)
    if m:
        return np.array([m.group(1)] * ntrials)
    m = re.search(r'([ABC])_to_(?:Env\d_)?(?:Location)?([ABC])$', scene)
    if m:
        n0 = min(change_trial, ntrials)
        return np.array([m.group(1)] * n0 + [m.group(2)] * (ntrials - n0))
```

iii. Matches the reference code's `get_reward_zones` with `change_trial=30`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps and `reward_zone` behavior time series. A trial is rewarded if both a reward event occurred AND the reward zone was entered.

ii.
```python
isreward = np.array([bool((reward[a:b] > 0).any() and (rzone[a:b] > 0).any())
                     for a, b in zip(starts, teles)])
...
out[5] = int(isreward[i])
```

iii. Follows the reference `behavior.get_trial_types` definition of `isreward`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame indices via `searchsorted`, creating a binary reward array. For each trial, `isreward` is True if any reward frame is present AND the reward zone was entered. The value is constant (broadcast) across all timepoints in the trial.

ii.
```python
ridx = np.searchsorted(t, reward_t)
ridx = np.clip(ridx, 0, nframes - 1)
reward[ridx] = 1.0
...
isreward = np.array([bool((reward[a:b] > 0).any() and (rzone[a:b] > 0).any())
                     for a, b in zip(starts, teles)])
out[5] = int(isreward[i])
```

iii. The dual condition (reward AND zone entry) matches the paper's definition and distinguishes true rewards from potential artifacts.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Ophys/behavior length mismatch**: 10 sessions (m17/m18) have ophys arrays 1 frame longer than behavior; ophys is truncated to behavior length (`F = Fseries[k].data[:nframes, :]`).
- **Lick-sensor errors**: Trials with >30% of frames having cumulative lick > 2 are dropped entirely (81 trials).
- **Reward zone disagreement**: If the scene-derived zone labels disagree with empirically observed zone entries, the empirical values are used with a warning.
- **Reward timestamp alignment**: Reward timestamps are clipped to valid frame indices.
- **Trial start/teleport consistency**: An assertion verifies equal counts and proper ordering.

ii.
```python
F = np.concatenate([np.asarray(Fseries[k].data[:nframes, :], dtype=np.float32).T ...])  # truncate to nframes
...
lick_err = np.array([(lick[a:b] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC ...])
...
ridx = np.clip(ridx, 0, nframes - 1)
...
assert len(starts) == len(teles) and np.all(teles > starts)
```

iii. The AI documented these edge cases in CONVERSION_NOTES Steps 4 and 10.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **dF/F + OASIS deconvolution** per session (3-6 seconds per session depending on neuron count).
2. **NWB file loading** (0.5-1.2 seconds per session, I/O bound).
3. **Pickle saving** of the final 9.5 GB file.

ii. N/A (timing info from CONVERSION_NOTES)

iii. The AI timed each step and estimated total processing at ~3-5 minutes with 12 parallel workers.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for building input/output arrays iterates over ~80 trials per session. Operations like `discretize_distance`, `discretize_position`, and `discretize_speed` could theoretically be applied to the full session before splitting, but variable trial lengths make this awkward. The interneuron correlation is already vectorized (matrix-vector product). The `isreward` and `morph` computations use list comprehensions over trials.

ii. N/A

iii. The per-trial loop is the natural structure given variable-length trials. The AI vectorized the most expensive operation (interneuron detection).

## 13-c. What processing does the code repeat multiple times?

i. Unlike the reference solution which has a separate survey step, the AI processes each session only once (no survey step). However, the `empirical_reward_zones` function re-iterates over all trials to check zone agreement after `scene_reward_zones` already computed the labels.

ii. N/A

iii. The AI's multiprocessing approach processes each session independently in a single pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes `dff` (the full dF/F matrix) in addition to `events` (deconvolved), but only `events` is used in the final output. The `dff` is used for interneuron detection (correlation with speed) and optionally for processing plots, so it is not entirely unnecessary but the full matrix is kept in memory longer than needed.

ii.
```python
dff, events = compute_dff_events(F, Fneu, starts, teles, fs)
...
# dff used for interneuron detection:
D = dff[:, onmask]
```

iii. The dF/F is needed for the interneuron correlation check, which is part of the reference pipeline.
