# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data`, finds every `.nwb` file under every subject directory, sorts them deterministically by subject and session number, and processes each file as one session. It reads NWB contents with `h5py` rather than `pynwb`, then loads behavior and fluorescence streams from HDF5 paths.

ii.
```python
def list_sessions():
    files = []
    for sub in sorted(os.listdir(DATA_ROOT)):
        d = os.path.join(DATA_ROOT, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.nwb'):
                files.append(os.path.join(d, fn))
```
```python
with h5py.File(path, 'r') as f:
    beh = load_behavior(f)
    F, Fneu, plane_of_cell = load_fluorescence(f)
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI says the dataset is `/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, one file per mouse per experiment day, and that all 152 files should be included.

## 1-b. How are the data split into subjects?

i. Subjects are identified per session from the NWB `general/subject/subject_id` field, then uniqued and sorted to build `data['subjects']`. The directory layout also encodes subjects via `sub-m*`.

ii.
```python
with h5py.File(path, 'r') as f:
    subject = f['general/subject/subject_id'][()].decode()
```
```python
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
sub_index = {s: i for i, s in enumerate(subjects)}
```

iii. The notes say the DANDI set contains 11 subject folders (`sub-m3 … sub-m19`) and that NWB `subject_id` matches those mouse IDs.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identity comes from `general/session_id` and from the filename `ses-<NN>`.

ii.
```python
with h5py.File(path, 'r') as f:
    session_id = f['general/session_id'][()].decode()
```
```python
data = {
    'neural': [r['neural'] for r in results],
    'input': [r['input'] for r in results],
    'output': [r['output'] for r in results],
```

iii. In Step 2 of the notes, the AI states that there is one file per mouse per experiment day and that `ses-NN == experiment day`.

## 1-d. How are the data split into trials?

i. Trials are defined from positive `trial_start` frames to positive `teleport` frames, with each exported trial using the half-open slice `[trial_start, teleport)`. The code also trims unpaired leading teleports or trailing trial starts for robustness.

ii.
```python
trial_starts = np.where(beh['trial_start'] > 0)[0]
teleports = np.where(beh['teleport'] > 0)[0]
if len(teleports) and len(trial_starts) and teleports[0] < trial_starts[0]:
    teleports = teleports[1:]
if len(trial_starts) > len(teleports):
    trial_starts = trial_starts[:len(teleports)]
if len(teleports) > len(trial_starts):
    teleports = teleports[:len(trial_starts)]
```
```python
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    pos = beh['position'][st:sp_]
    act = events[:, st:sp_]
```

iii. The notes explicitly justify `[trial_start:teleport)` in Step 4: it excludes the corrupted teleport frame and keeps the on-track lap aligned to the trial start.

## 1-e. How are trials filtered based on quality controls?

i. The AI removes lick-sensor-error trials, defined as trials where more than 30% of frames have cumulative lick count greater than 2. It does not apply the human reference solution’s short-trial `<50` filter.

ii.
```python
LICK_ERROR_FRAC_THRESH = 0.3
LICK_ERROR_COUNT_THRESH = 2
...
seg = beh['lick'][st:sp_]
lick_error[i] = (np.sum(seg > LICK_ERROR_COUNT_THRESH) / len(seg)) > LICK_ERROR_FRAC_THRESH
```
```python
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    if lick_error[i]:
        continue
```

iii. In Step 4 and Step 5, the AI says this reproduces the paper’s reported removal exactly: 81 trials at the 30% threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from raw suite2p fluorescence `Fluorescence/.../data` and neuropil `Neuropil/.../data`, after restricting to manually curated `iscell` ROIs. The AI explicitly avoids using the NWB `Deconvolved` stream.

ii.
```python
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = seg['iscell'][:, 0].astype(bool)
...
Fp = f[f'processing/ophys/Fluorescence/{plane}/data']
Np = f[f'processing/ophys/Neuropil/{plane}/data']
```

iii. The notes say the paper analyzes custom `events` derived from raw `F` and `Fneu`, and that the NWB `Deconvolved` array is suite2p’s own deconvolution on raw fluorescence and should not be used.

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, computes dF/F with neuropil subtraction (`0.7 * Fneu`), adds back the per-segment neuropil mean, estimates a maximin baseline with Gaussian smoothing and min/max filters, converts to dF/F, smooths with sigma 2 frames, and deconvolves with OASIS to obtain events.

ii.
```python
f_ -= neu_coef * f_neu_
...
base = nansmooth(seg, BASELINE_SMOOTH_SIGMA, axis=-1)
base = sp.ndimage.minimum_filter1d(base, BASELINE_FILTER_WIN, axis=-1)
base = sp.ndimage.maximum_filter1d(base, BASELINE_FILTER_WIN, axis=-1)
flow[:, start:stop] = base
```
```python
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
...
smoothed = nansmooth(dff[:, start:stop], DFF_SMOOTH_SIGMA, axis=-1)
events[:, start:stop] = dcnv.oasis(np.ascontiguousarray(smoothed), 2000, tau, fs)
```

iii. Step 1 and Step 5 of the notes say this is a faithful port of `reward_relative.preprocessing.dff`, with the one intentional change that the kept samples use `[start:stop]` rather than the paper code’s legacy `[start-1:stop-1]`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. After `iscell` filtering, the AI removes putative interneurons whose dF/F is too correlated with running speed (`r > 0.5`). It also drops cells with non-finite event values or non-finite speed correlation estimates.

ii.
```python
speed_corr = (dv @ sv) / denom
is_interneuron = speed_corr > INTERNEURON_R_THRESH
```
```python
finite_cells = np.all(np.isfinite(events[:, valid]), axis=1) & np.isfinite(speed_corr)
keep_cells = (~is_interneuron) & finite_cells
events = events[keep_cells]
```

iii. The notes cite the paper’s interneuron exclusion rule and add a defensive note that all-NaN or otherwise degenerate cells should be dropped rather than passed downstream.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned by trial start simply by exporting per-trial windows that begin at `trial_start` and run until `teleport`, with no extra temporal shifting.

ii.
```python
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    act = events[:, st:sp_]
    neural.append(np.ascontiguousarray(act, dtype=np.float32))
```

iii. In Step 5, the AI explicitly states `trial window = [trial_start, teleport)` and `alignment event = trial start`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native imaging-frame resolution and does not rebin. It estimates the time bin size from the median spacing of the behavior timestamps, which are already on the imaging-frame grid.

ii.
```python
dt = float(np.median(np.diff(beh['time'])))
fs = 1.0 / dt
```
```python
'time_bin_size': float(np.median(dts) * 1000.0),
'time_bin_size_range_ms': [float(dts.min() * 1000), float(dts.max() * 1000)],
```

iii. The notes say all behavioral and neural time series are already sampled at the imaging frame rate, about 15.5 Hz (~64.5 ms bins), so no further temporal rebinning is needed.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps loaded from `position/timestamps`, which the AI treats as the shared per-frame timebase.

ii.
```python
beh = {
    'time': b['position/timestamps'][:],
```
```python
tt = beh['time'][st:sp_] - beh['time'][st]
inp[0] = tt
```

iii. The notes say all behavioral streams are already on the imaging-frame grid, so any of those timestamps are equivalent for this purpose.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the timestamp of the first frame in that trial so time starts at 0 seconds.

ii.
```python
tt = beh['time'][st:sp_] - beh['time'][st]
...
inp[0] = tt
```

iii. The notes describe this variable as “time since trial start (s)” and use trial start as the alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the same per-trial frame slice for both neural activity and timestamps; no interpolation is done inside the converter because the NWB behavior has already been aligned to imaging frames.

ii.
```python
act = events[:, st:sp_]
tt = beh['time'][st:sp_] - beh['time'][st]
```

iii. Step 1 and Step 3 of the notes say the NWB `BehavioralTimeSeries` is already interpolated onto the imaging-frame grid by the original preprocessing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavioral time series.

ii.
```python
'environment': b['environment/data'][:],
```
```python
env_vals = np.unique(beh['environment'][st:sp_])
environment[i] = int(env_vals[0])
```

iii. The notes map this directly to the paper’s two environments, ENV1 and ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the AI takes the unique nonnegative environment value, asserts there is exactly one, stores it as a per-trial value, and broadcasts it across time within the trial.

ii.
```python
env_vals = np.unique(beh['environment'][st:sp_])
env_vals = env_vals[env_vals >= 0]
assert len(env_vals) == 1
environment[i] = int(env_vals[0])
...
inp[1] = environment[i]
```

iii. The AI’s notes say environment is constant within a trial and should therefore be treated as a per-trial input.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The AI derives trial number from the loop index over the detected trial boundaries, not from the stored NWB `trial number` series.

ii.
```python
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    ...
    inp[2] = i
```

iii. In Step 5, the notes say “trial number within the session (0-indexed, constant within a trial)”.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing is applied beyond using the 0-indexed trial loop counter and broadcasting it across all frames in the trial.

ii.
```python
inp[2] = i
```

iii. The AI treats trial number as a simple within-session ordinal variable.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The AI derives previous trial outcome from reward timestamps plus the `reward_zone` stream: a trial is considered rewarded only if a reward occurs during the trial and the reward zone was active in that trial.

ii.
```python
reward_frames = np.searchsorted(beh['time'], beh['reward_times'])
...
any_reward = np.any((reward_frames >= st) & (reward_frames < sp_))
any_rzone = np.any(beh['reward_zone'][st:sp_] > 0)
is_reward[i] = int(bool(any_reward and any_rzone))
```

iii. The notes cite `behavior.get_trial_types` from the paper code and say the reference defines rewarded trials as reward delivered while the reward zone was active.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI constructs a per-trial `is_reward` vector, then shifts it by one trial so each trial gets the previous trial’s outcome. For the first trial of a session, it imputes `1` rather than `0`.

ii.
```python
prev_outcome = np.empty(n_trials, dtype=np.int64)
prev_outcome[1:] = is_reward[:-1]
prev_outcome[0] = 1
...
inp[3] = prev_outcome[i]
```

iii. In Step 5, the AI explicitly justifies the first-trial imputation as “modal outcome (84.7%)” and mentions warm-up trials on the same reward zone before imaging.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The AI derives this output from trial position plus reward-zone identity inferred from the session’s scene name and trial index. It does not infer zones from the `reward_zone` samples directly.

ii.
```python
scene = identifier.split('/')[-1]
zone_labels = zones_from_scene(scene, n_trials)
zone_starts = np.array([REWARD_ZONE_CM[z][0] for z in zone_labels])
zone_stops = np.array([REWARD_ZONE_CM[z][1] for z in zone_labels])
```
```python
pos = beh['position'][st:sp_]
before = pos < zone_starts[i]
after = pos > zone_stops[i]
```

iii. The notes say this matches `behavior.get_reward_zones` in the paper code and was cross-validated against the empirical `reward_zone` activations in every trial.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the AI computes signed distance to the nearest reward-zone boundary: negative before the zone, zero in the zone, positive after the zone. It then digitizes the result into 7 categories.

ii.
```python
d = np.zeros(T)
before = pos < zone_starts[i]
after = pos > zone_stops[i]
d[before] = pos[before] - zone_starts[i]
d[after] = pos[after] - zone_stops[i]
```
```python
out[0] = digitize_distance_to_reward(d)
```

iii. The notes describe this as the “signed distance to the nearest point of the active reward zone, 0 inside it”.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses explicit threshold logic implementing the 7 bins from the task specification.

ii.
```python
out[d < -50] = 0
out[(d >= -50) & (d < -10)] = 1
out[(d >= -10) & (d < 0)] = 2
out[d == 0] = 3
out[(d > 0) & (d <= 10)] = 4
out[(d > 10) & (d <= 50)] = 5
out[d > 50] = 6
```

iii. The notes list the same discretization edges and say they are driven by the decoder task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The output is aligned by using the same `[trial_start:teleport)` per-trial slice as the neural data.

ii.
```python
pos = beh['position'][st:sp_]
act = events[:, st:sp_]
...
out[0] = digitize_distance_to_reward(d)
```

iii. The notes say both behavior and neural streams are already on the imaging-frame grid, so shared trial indexing is sufficient.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavioral time series.

ii.
```python
'position': b['position/data'][:],
...
pos = beh['position'][st:sp_]
```

iii. The notes identify `position` as the animal’s corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI takes per-trial position samples and discretizes them into five 90 cm bins spanning the 450 cm track, implemented via floor division and clipping.

ii.
```python
def digitize_position(pos):
    return np.clip((pos // 90.0), 0, 4).astype(np.int64)
```
```python
out[1] = digitize_position(pos)
```

iii. In Step 5, the notes define the bins as `<90`, `[90,180)`, `[180,270)`, `[270,360)`, and `>=360`.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is thresholded into 5 equal-width bins over track position: 0 to 4 corresponding to 90 cm chunks.

ii.
```python
def digitize_position(pos):
    return np.clip((pos // 90.0), 0, 4).astype(np.int64)
```

iii. The notes say the 450 cm track should be divided into 5 equal bins, so 90 cm per bin.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned by taking the same frame indices used for each neural trial segment.

ii.
```python
pos = beh['position'][st:sp_]
act = events[:, st:sp_]
```

iii. The AI assumes the NWB behavior has already been aligned to the imaging-frame grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived directly from the `lick` behavioral time series.

ii.
```python
'lick': b['lick/data'][:],
...
lick = beh['lick'][st:sp_]
```

iii. The notes identify the lick stream as a per-frame cumulative lick count that must be converted to a binary decoder target.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes licks at each frame by thresholding `lick > 0`.

ii.
```python
out[3] = (lick > 0).astype(np.int64)
```

iii. The task specification requires a binary no/yes lick output, so the AI collapses cumulative counts to presence/absence.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by taking the same trial slice as the neural activity and all other framewise variables.

ii.
```python
lick = beh['lick'][st:sp_]
act = events[:, st:sp_]
```

iii. The AI’s notes state that the NWB behavior streams are already on the imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the session scene string (`identifier`) and trial index through `zones_from_scene`, not from direct clustering of the `reward_zone` time series.

ii.
```python
identifier = f['identifier'][()].decode()
scene = identifier.split('/')[-1]
zone_labels = zones_from_scene(scene, n_trials)
```

iii. The notes say this follows the paper’s `behavior.get_reward_zones` logic and was cross-checked against observed reward-zone-active positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses scene names: fixed scenes map to one constant zone, and `*_to_*` scenes switch to a second zone after trial 30. It then maps labels A/B/C to 0/1/2 and broadcasts the result within each trial.

ii.
```python
for first in REWARD_ZONE_LABELS:
    if f'{first}_to' in scene:
        second = scene[-1]
        ...
        labels = np.array([first] * min(change_trial, n_trials)
                          + [second] * max(0, n_trials - change_trial))
```
```python
out[4] = REWARD_ZONE_LABELS.index(zone_labels[i])
```

iii. The notes cite the paper statement that each switch occurred after 30 trials and the code `behavior.get_reward_zones`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward timestamps plus the `reward_zone` activity signal, using the same per-trial `is_reward` definition that feeds previous-trial outcome.

ii.
```python
reward_frames = np.searchsorted(beh['time'], beh['reward_times'])
...
any_reward = np.any((reward_frames >= st) & (reward_frames < sp_))
any_rzone = np.any(beh['reward_zone'][st:sp_] > 0)
is_reward[i] = int(bool(any_reward and any_rzone))
```

iii. The notes say this matches the paper code’s trial-type logic rather than simply asking whether any reward timestamp fell inside the trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The AI maps sparse reward timestamps onto frame indices with `searchsorted`, computes a per-trial rewarded/omitted label using reward delivery and reward-zone activation, and writes that label as a constant over all frames in the trial.

ii.
```python
reward_frames = np.searchsorted(beh['time'], beh['reward_times'])
reward_frames = np.clip(reward_frames, 0, n_frames - 1)
...
is_reward[i] = int(bool(any_reward and any_rzone))
...
out[5] = is_reward[i]
```

iii. The notes justify this by pointing to `behavior.get_trial_types`, which defines rewarded trials using both reward delivery and active reward zone.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI includes several defensive checks: it trims mismatched leading/trailing trial markers, asserts teleports follow starts, drops non-finite cells, and zero-fills any unexpected non-finite within-trial events. However, it does not handle one-frame neural/behavior length mismatches; it asserts exact equality and crashes on those sessions.

ii.
```python
if len(teleports) and len(trial_starts) and teleports[0] < trial_starts[0]:
    teleports = teleports[1:]
if len(trial_starts) > len(teleports):
    trial_starts = trial_starts[:len(teleports)]
if len(teleports) > len(trial_starts):
    teleports = teleports[:len(trial_starts)]
assert np.all(teleports > trial_starts)
```
```python
assert F.shape[1] == n_frames, f'{path}: F has {F.shape[1]} frames, behavior has {n_frames}'
...
finite_cells = np.all(np.isfinite(events[:, valid]), axis=1) & np.isfinite(speed_corr)
...
if not np.all(np.isfinite(act)):
    act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The notes justify the trimming and non-finite filtering as robustness measures. There is no note justifying the hard assertion on frame-count mismatches, and the full conversion log shows that this assertion fails on 10 sessions.

## 13-a. What are the most time-consuming steps of the code?

i. The code structure indicates that the most expensive steps are reading fluorescence arrays, computing dF/F plus OASIS deconvolution, multiprocessing over sessions, and writing the final pickle. The script even records per-session timing for behavior read, fluorescence read, and `dff_events`.

ii.
```python
timing = {}
...
t1 = time.time(); timing['read_behavior'] = t1 - t0
F, Fneu, plane_of_cell = load_fluorescence(f)
t2 = time.time(); timing['read_fluorescence'] = t2 - t1
...
dff, events = dff_and_events(F, Fneu, trial_starts, teleports, keep_teleports, fs)
timing['dff_events'] = time.time() - t3
```

iii. No explicit prose justification is in the notes. The inference comes from the timing fields and from the full-run log, where per-session runtime scales with cell count and dF/F processing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loops are over dF/F trial segments, per-trial behavioral summarization, and per-trial export of neural/input/output arrays. These could be reduced only by working on padded or session-wide arrays, which the AI chose not to do.

ii.
```python
for start, stop in zip(start_inds, stop_inds):
    ...
```
```python
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    ...
    neural.append(np.ascontiguousarray(act, dtype=np.float32))
    inputs.append(inp)
    outputs.append(out)
```

iii. There is no explicit justification in the notes. The implementation suggests the AI accepted trial loops because the output format is inherently ragged by trial.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats trial-wise iteration at least twice per session: once to compute rewarded/environment/lick-error summaries, and again to slice/export the trial data. Inside `dff_and_events`, it also loops over the same segments once for baseline computation and again for smoothing/deconvolution.

ii.
```python
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    any_reward = ...
    env_vals = ...
    seg = beh['lick'][st:sp_]
```
```python
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    pos = beh['position'][st:sp_]
    speed = beh['speed'][st:sp_]
    lick = beh['lick'][st:sp_]
```

iii. No explicit note discusses this repetition. It is visible directly in the code structure.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI performs optional diagnostic plotting and computes several diagnostic or bookkeeping quantities that are not needed by the downstream decoder itself, such as `dff_kept`, detailed per-session timing, plotting-only recomputations, and rich metadata fields.

ii.
```python
dff_kept = dff[keep_cells]
...
if show_processing:
    plot_processing(result, beh, F, Fneu, dff_kept, events,
                    trial_starts, teleports, zone_starts, zone_stops, lick_error)
```
```python
'timing': timing,
'total_time': time.time() - t0,
'identifier': identifier,
'scene': scene,
```

iii. The notes present these as validation and sanity-check machinery rather than as essential decoder inputs. No explicit defense is given beyond documentation and diagnostics.
