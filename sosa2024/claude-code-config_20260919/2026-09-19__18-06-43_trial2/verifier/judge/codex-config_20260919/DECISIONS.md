# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI recursively enumerates every subject directory under `/app/data`, collects every `.nwb` file, sorts them numerically by mouse and session, and processes all 152 files by default. It reads NWB/HDF5 datasets directly with `h5py`, optionally in a multiprocessing pool.

ii.
```python
for sub in sorted(os.listdir(DATA_ROOT)):
    d = os.path.join(DATA_ROOT, sub)
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.nwb'):
            files.append(os.path.join(d, fn))
...
with h5py.File(path, 'r') as f:
    beh = load_behavior(f)
    F, Fneu, plane_of_cell = load_fluorescence(f)
```

iii. The notes report 152 NWB files in 11 subject folders and explain that direct HDF5 access was chosen to read the NWB streams efficiently. Full conversion is the default; `--sample` is only an explicit test mode.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB file's `general/subject/subject_id`. Results are grouped into a numerically sorted unique subject list, and each session gets an index into it.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
sub_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([sub_index[r['subject']] for r in results], dtype=np.int64),
```

iii. The AI verified that the 11 IDs (`m3` through `m19`, non-contiguous) agree with the switch-task cohort in the paper.

## 1-c. How are the data split into sessions?

i. Every NWB file is one session. Session/day is read from `general/session_id`; converted sessions are sorted by numeric subject and experiment day.

ii.
```python
session_id = f['general/session_id'][()].decode()
exp_day = int(session_id)
...
results.sort(key=lambda r: (int(r['subject'][1:]), r['exp_day']))
```

iii. The notes cross-check `ses-NN` against the repository's session metadata and report 14 sessions per mouse except 12 for m11.

## 1-d. How are the data split into trials?

i. Trial starts are positive samples of `trial_start`, and trial ends are positive `teleport` samples. Each exported lap is the half-open interval `[trial_start, teleport)`. Unpaired boundary events are trimmed defensively.

ii.
```python
trial_starts = np.where(beh['trial_start'] > 0)[0]
teleports = np.where(beh['teleport'] > 0)[0]
...
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    act = events[:, st:sp_]
```

iii. The AI found the teleport-index position to be interpolated/corrupt and selected `[start:teleport)` as the complete on-track lap. It notes that this also reproduces the paper's lick-error count.

## 1-e. How are trials filtered based on quality controls?

i. It drops a whole trial when more than 30% of its frames have cumulative lick count greater than 2. It retains all other trials, including slow-running frames.

ii.
```python
seg = beh['lick'][st:sp_]
lick_error[i] = (np.sum(seg > LICK_ERROR_COUNT_THRESH) / len(seg)) > LICK_ERROR_FRAC_THRESH
...
if lick_error[i]:
    continue
```

iii. This implements the Methods' lick-sensor-error rule and flags exactly 81 trials, matching the paper. The speed filter is deliberately omitted because retaining the `<2 cm/s` decoder class is required.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from raw suite2p fluorescence `Fluorescence`, neuropil fluorescence `Neuropil`, ROI `iscell`, and plane membership. The stored NWB `Deconvolved` stream is not used.

ii.
```python
Fp = f[f'processing/ophys/Fluorescence/{plane}/data']
Np = f[f'processing/ophys/Neuropil/{plane}/data']
...
sel = iscell[offset:offset + n_roi]
Fs.append(np.asarray(Fp[:, :], dtype=np.float32)[:, sel].T)
Fneus.append(np.asarray(Np[:, :], dtype=np.float32)[:, sel].T)
```

iii. The AI determined that NWB `Deconvolved` is suite2p output on raw F, whereas the paper analyzes OASIS events computed from its custom neuropil-corrected dF/F.

## 2-b. How is the `neural` data processed?

i. Planes are pooled after `iscell` selection. For each appropriate lap/ITI segment, the code subtracts `0.7*Fneu`, adds back the segment mean neuropil, applies the paper's maximin baseline (Gaussian sigma 15, 300-frame minimum then maximum), calculates `(F-baseline)/abs(baseline)`, smooths dF/F with sigma 2, and OASIS-deconvolves with tau 0.7. Teleports enter baseline windows only on the paper-listed animal-days.

ii.
```python
f_ -= neu_coef * f_neu_
...
base = nansmooth(seg, BASELINE_SMOOTH_SIGMA, axis=-1)
base = sp.ndimage.minimum_filter1d(base, BASELINE_FILTER_WIN, axis=-1)
base = sp.ndimage.maximum_filter1d(base, BASELINE_FILTER_WIN, axis=-1)
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
...
events[:, start:stop] = dcnv.oasis(np.ascontiguousarray(smoothed), 2000, tau, fs)
```

iii. The constants and processing order are traced to `reward_relative.preprocessing.dff`, the Methods, suite2p settings, and `teleport_metadata.py`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It first keeps manually curated `iscell==1` ROIs, then removes cells whose dF/F–speed Pearson correlation exceeds 0.5, plus cells with non-finite event values or correlation.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)
...
is_interneuron = speed_corr > INTERNEURON_R_THRESH
finite_cells = np.all(np.isfinite(events[:, valid]), axis=1) & np.isfinite(speed_corr)
keep_cells = (~is_interneuron) & finite_cells
events = events[keep_cells]
```

iii. Manual `iscell` curation and the `r>0.5` putative-interneuron exclusion are explicitly prescribed by the paper; the finite-value filter protects the decoder from invalid arrays.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural and behavior already share the imaging-frame grid. Slicing each event matrix at `[trial_start:teleport)` makes column zero the trial-start event.

ii.
```python
act = events[:, st:sp_]
tt = beh['time'][st:sp_] - beh['time'][st]
```

iii. The notes identify trial start as the requested alignment event and set `off_start=0`; laps remain variable length.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging-frame resolution is retained, approximately 64.48 ms (15.51 Hz). No temporal rebinning or resampling is applied.

ii.
```python
dt = float(np.median(np.diff(beh['time'])))
fs = 1.0 / dt
...
'time_bin_size': float(np.median(dts) * 1000.0),
```

iii. The paper states that behavioral and neural series are sampled at the imaging rate; the AI therefore preserves that grid.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the NWB `position/timestamps` array, stored as `beh['time']`.

ii.
```python
'time': b['position/timestamps'][:],
...
tt = beh['time'][st:sp_] - beh['time'][st]
```

iii. The notes state that all behavior fields have already been aligned to this imaging-frame time grid.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial start is subtracted from every timestamp in that trial.

ii.
```python
tt = beh['time'][st:sp_] - beh['time'][st]
inp[0] = tt
```

iii. This directly expresses elapsed seconds and makes the first retained sample zero.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses exactly the same `[st:sp_]` indices and therefore has one timestamp for every neural column.

ii.
```python
act = events[:, st:sp_]
tt = beh['time'][st:sp_] - beh['time'][st]
```

iii. The AI asserts globally that fluorescence frame count equals behavior timestamp count before conversion.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the NWB `environment/data` behavior stream.

ii.
```python
'environment': b['environment/data'][:],
env_vals = np.unique(beh['environment'][st:sp_])
```

iii. The data exploration established that 0 and 1 denote ENV1 and ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative/pre-TTL values are excluded, the code asserts one unique environment per trial, and broadcasts that integer over all trial frames.

ii.
```python
env_vals = env_vals[env_vals >= 0]
assert len(env_vals) == 1
environment[i] = int(env_vals[0])
...
inp[1] = environment[i]
```

iii. This follows the paper helper's per-trial morph/environment extraction and validates the expected constancy.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Although raw `trial number` is loaded, the exported value is the zero-based index `i` of paired trial-start/teleport boundaries.

ii.
```python
'trial_number': b['trial number/data'][:],
...
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    inp[2] = i
```

iii. The notes describe trial number as a zero-indexed within-session variable; the loop index is consistent with the observed raw numbering.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transform beyond zero-based enumeration and broadcasting across the trial.

ii.
```python
inp[2] = i
```

iii. It is intended as a per-trial contextual input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Current outcomes use sparse `Reward/timestamps` mapped to imaging frames plus the dense `reward_zone` stream; previous outcome is the one-trial lag of that result.

ii.
```python
reward_frames = np.searchsorted(beh['time'], beh['reward_times'])
...
is_reward[i] = int(bool(any_reward and any_rzone))
prev_outcome[1:] = is_reward[:-1]
```

iii. This mirrors `behavior.get_trial_types`, in which a rewarded trial has both a reward delivery and an active reward zone.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Outcomes are shifted by one trial and broadcast. The undefined first trial is imputed as 1 (rewarded), not 0.

ii.
```python
prev_outcome[1:] = is_reward[:-1]
prev_outcome[0] = 1
...
inp[3] = prev_outcome[i]
```

iii. The AI chose the modal value (84.7% rewarded) and argued that mice had warm-up trials before imaging and one imputed sample per session would have little effect.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position`, the session scene parsed from the NWB identifier, trial index, and fixed paper zone boundaries A=80–130, B=200–250, C=320–370 cm.

ii.
```python
scene = identifier.split('/')[-1]
zone_labels = zones_from_scene(scene, n_trials)
zone_starts = np.array([REWARD_ZONE_CM[z][0] for z in zone_labels])
...
pos = beh['position'][st:sp_]
```

iii. This ports `behavior.get_reward_zones`: transition scenes switch zones after 30 trials. The AI cross-validated labels against active-zone positions for all trials.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is zero inside the active interval, position minus the start before it, and position minus the end after it; that distance is then categorized.

ii.
```python
d = np.zeros(T)
before = pos < zone_starts[i]
after = pos > zone_stops[i]
d[before] = pos[before] - zone_starts[i]
d[after] = pos[after] - zone_stops[i]
out[0] = digitize_distance_to_reward(d)
```

iii. This is the instructed distance to any point in the reward zone and preserves direction relative to the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks implement seven categories: `<-50`, `[-50,-10)`, `[-10,0)`, exactly zero, `(0,10]`, `(10,50]`, and `>50`.

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

iii. The explicit comparisons were chosen to reproduce the instruction's asymmetric inclusivity exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and events are sliced using the identical frame interval.

ii.
```python
pos = beh['position'][st:sp_]
act = events[:, st:sp_]
```

iii. Both streams are already on the imaging-frame grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
'position': b['position/data'][:],
pos = beh['position'][st:sp_]
```

iii. The stream is position in centimeters along the 450 cm virtual track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is divided by 90 cm with floor division, clipped to classes 0–4, and cast to integer.

ii.
```python
def digitize_position(pos):
    return np.clip((pos // 90.0), 0, 4).astype(np.int64)
```

iii. Five equal bins over 450 cm are 90 cm wide; clipping absorbs minor values outside nominal track bounds.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Classes are `<90`, 90–<180, 180–<270, 270–<360, and `>=360`, with out-of-range samples clipped to end classes.

ii.
```python
out[1] = digitize_position(pos)
```

iii. This directly implements the requested five equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same trial frame slice as neural events.

ii.
```python
pos = beh['position'][st:sp_]
act = events[:, st:sp_]
```

iii. No interpolation is needed because NWB behavior was already aligned to imaging frames.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the NWB `lick/data` behavior stream.

ii.
```python
'lick': b['lick/data'][:],
lick = beh['lick'][st:sp_]
```

iii. Exploration showed this is a per-imaging-frame cumulative lick count.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive count is 1 and zero is 0; trials judged to have lick-sensor errors are omitted earlier.

ii.
```python
out[3] = (lick > 0).astype(np.int64)
```

iii. This supplies the required binary no/yes decoder target while excluding known sensor artifacts.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays use the same `[st:sp_)` imaging-frame indices.

ii.
```python
lick = beh['lick'][st:sp_]
act = events[:, st:sp_]
```

iii. The NWB lick stream was pre-aligned to the imaging grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the scene suffix in the NWB identifier and the trial index (including the trial-30 switch rule), rather than inferred from reward delivery samples.

ii.
```python
scene = identifier.split('/')[-1]
zone_labels = zones_from_scene(scene, n_trials)
```

iii. This is a port of the repository's `behavior.get_reward_zones` and remains defined on omission trials, when the active-zone stream is absent.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene text maps to A/B/C; `*_to_*` sessions use the first label for 30 trials and the second thereafter. A/B/C are encoded 0/1/2 and broadcast over time.

ii.
```python
labels = np.array([first] * min(change_trial, n_trials)
                  + [second] * max(0, n_trials - change_trial))
...
out[4] = REWARD_ZONE_LABELS.index(zone_labels[i])
```

iii. The rule matches the paper's “switch after 30 trials” design and was empirically cross-checked.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses sparse `Reward/timestamps`, behavior timestamps, trial boundaries, and dense `reward_zone/data`.

ii.
```python
'reward_times': b['Reward/timestamps'][:],
reward_frames = np.searchsorted(beh['time'], beh['reward_times'])
any_reward = np.any((reward_frames >= st) & (reward_frames < sp_))
any_rzone = np.any(beh['reward_zone'][st:sp_] > 0)
```

iii. The AI follows the reference behavior helper rather than treating any timestamp alone as proof of a valid rewarded trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame insertion indices. A trial is 1 only when it contains both a reward event and active reward-zone samples; the scalar is broadcast across time.

ii.
```python
is_reward[i] = int(bool(any_reward and any_rzone))
...
out[5] = is_reward[i]
```

iii. This implements `behavior.get_trial_types` and yields the expected approximately 85% rewarded rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Boundary mismatches are trimmed, reward indices clipped, constant-environment and equal neural/behavior lengths asserted, lick-error trials removed, non-finite cells dropped, and any unexpected remaining non-finite within-trial activity zero-filled. Worker errors abort the full conversion rather than silently losing sessions.

ii.
```python
if len(trial_starts) > len(teleports):
    trial_starts = trial_starts[:len(teleports)]
...
assert F.shape[1] == n_frames
...
if not np.all(np.isfinite(act)):
    act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The notes distinguish expected NaNs outside processed laps from invalid within-lap values and emphasize that the decoder must never receive NaNs.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large F/Fneu arrays and recomputing dF/F plus OASIS for every cell/session are dominant; serialization of the large pickle also costs time. The code records read and processing timings and parallelizes sessions.

ii.
```python
timing['read_fluorescence'] = t2 - t1
...
timing['dff_events'] = time.time() - t3
...
with ctx.Pool(workers, maxtasksperchild=2) as pool:
```

iii. The AI's documentation identifies the 92 GB source dataset and uses direct HDF5 plus up to ten workers to control runtime and memory recycling.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial reward/environment/lick-error calculation and per-trial export loops could partly be vectorized over full-session frame arrays, although variable-length trials still require splitting. Segment baseline/OASIS loops are intrinsically organized by trial/ITI segment. Session work is parallelized rather than vectorized.

ii.
```python
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    any_reward = np.any((reward_frames >= st) & (reward_frames < sp_))
...
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    pos = beh['position'][st:sp_]
```

iii. The agent supplied no explicit efficiency justification for these loops; its design favors readable variable-length trial handling and coarse session-level multiprocessing.

## 13-c. What processing does the code repeat multiple times?

i. It traverses trial boundaries once to derive outcomes/environment/lick errors and again to build arrays. Diagnostic plotting, when requested, traverses retained trials again and recomputes reward distance and concatenated raw vectors. Baseline smoothing and deconvolution necessarily repeat for each segment.

ii.
```python
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    ... # task variables
...
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    ... # exported arrays
```

iii. The notes do not call this out as a concern; diagnostics are optional and intended for validation rather than the normal full run.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `scanning`, computes/keeps some provenance and timing fields, and retains `dff_kept` only for optional plots; these do not enter the final decoder dictionary. In plot mode it also creates extensive diagnostic figures that downstream training discards.

ii.
```python
'scanning': b['scanning/data'][:],
...
dff_kept = dff[keep_cells]
...
if show_processing:
    plot_processing(..., dff_kept, events, ...)
```

iii. The extra work supports sanity checks and provenance. Plotting is opt-in, but `scanning` is loaded even in ordinary conversion without being used.
