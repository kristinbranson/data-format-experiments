# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI gathers every `.nwb` file under `/app/data` with a recursive `glob`, sorts them naturally, and processes them one session at a time with `h5py`. Within each session it opens the NWB/HDF5 file directly, reads behavior streams from `processing/behavior/BehavioralTimeSeries`, and reads neural data from `processing/ophys`.

ii.
```python
files=sorted(glob.glob(DATA_ROOT+'/**/*.nwb',recursive=True),key=natural_key)
...
with h5py.File(path, 'r') as h:
    b = h['processing/behavior/BehavioralTimeSeries']
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 10, the AI says the released NWBs are already aligned and already contain the neural representation it wants, so it chose direct HDF5 reads and one-session-at-a-time processing for lower memory use.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the NWB field `general/subject/subject_id`. As sessions are processed, the AI appends unseen subject IDs to `subjects` and stores the per-session index in `subject_idx`.

ii.
```python
subject = h['general/subject/subject_id'][()].decode()
...
if info['subject'] not in subjects: subjects.append(info['subject'])
subject_idx.append(subjects.index(info['subject']))
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI states that subject ID should come from the NWB subject table and that sessions are indexed into a unique subject list.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The main loop iterates over the sorted list of files and each call to `process_session` returns one session’s trial list.

ii.
```python
files=sorted(glob.glob(DATA_ROOT+'/**/*.nwb',recursive=True),key=natural_key)
...
for i,f in enumerate(files):
    ns,xs,ys,info=process_session(f, make_plot=args.show_processing and i<2)
```

iii. The AI’s notes repeatedly describe the released dataset as “152 NWB files” and treat those files as the session unit.

## 1-d. How are the data split into trials?

i. Trials are defined from the behavior streams `trial_start` and `teleport`. For every start pulse, the AI finds the first teleport after that start and before the next start; each trial is the half-open interval `[start, teleport)`.

ii.
```python
def find_complete_trials(b):
    start_flags = np.asarray(b['trial_start/data'][:])
    teleport_flags = np.asarray(b['teleport/data'][:])
    starts = np.flatnonzero(start_flags > 0)
    teleports = np.flatnonzero(teleport_flags > 0)
    ...
    candidates = teleports[(teleports > start) & (teleports < next_start)]
    ...
    trials.append((int(start), int(candidates[0])))  # stop is exclusive
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 10, the AI says trial IDs cross teleport boundaries in the released data, so it intentionally ignored raw trial IDs for slicing and instead used explicit start-to-teleport pairing.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes trials with corrupted lick-sensor readings: if more than 30% of samples in a trial have `lick > 2`, that trial is skipped entirely. It also rejects sessions that end up with fewer than two kept trials.

ii.
```python
bad_lick = 0
for j, (start, stop) in enumerate(trials):
    ...
    lick_raw = streams['lick'][q]
    if np.mean(lick_raw > 2) > 0.30:
        bad_lick += 1
        continue
...
if len(neural_trials) < 2:
    raise ValueError(f'Only {len(neural_trials)} valid trials')
```

iii. In `CONVERSION_NOTES.md` Steps 3, 5, 9, and 10, the AI cites the paper’s lick-corruption criterion and argues that, because the requested lick target is categorical, it cannot represent NaN lick trials and therefore those trials should be removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the stored NWB `processing/ophys/Deconvolved/<plane>/data` arrays plus `ImageSegmentation/PlaneSegmentation/iscell` and `planeIdx`. It does not use `Fluorescence` or `Neuropil`.

ii.
```python
dec = h['processing/ophys/Deconvolved']
plane_names = sorted((k for k, v in dec.items() if isinstance(v, h5py.Group)), key=natural_key)
arrays = [np.asarray(dec[p]['data'][:], dtype=np.float32) for p in plane_names]
...
seg = h['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = np.asarray(seg['iscell'][:, 0]) > 0
plane_idx = np.asarray(seg['planeIdx'][:], dtype=np.int16)
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, 6, and 10, the AI explicitly argues that the released NWBs already contain the needed deconvolved neural signal, so recomputing dF/F and deconvolution is unnecessary.

## 2-b. How is the `neural` data processed?

i. The AI concatenates the per-plane deconvolved matrices across neurons, filters them by `iscell`, optionally trims one trailing neural frame if behavior is shorter by exactly one sample, and then slices the per-trial neural matrices directly from that session-level array.

ii.
```python
full = np.concatenate(arrays, axis=1)
...
return full[:, iscell], plane_idx[iscell], plane_names
...
neural_excess_frames = int(neural_all.shape[0] - ntime)
if neural_excess_frames:
    neural_all = neural_all[:ntime]
...
neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
```

iii. The AI’s notes say its main processing choice was to “read only released Deconvolved arrays,” concatenate planes once, and avoid the paper’s fluorescence-to-dF/F-to-events recomputation to save time and memory.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered only by Suite2p’s `iscell` flag. The AI keeps all `iscell` ROIs and does not remove putative interneurons or other analysis-specific cell subsets.

ii.
```python
seg = h['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = np.asarray(seg['iscell'][:, 0]) > 0
...
return full[:, iscell], plane_idx[iscell], plane_names
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 10, the AI says `iscell` is the recording-quality filter but deliberately declines to apply place-cell, reward-relative, or interneuron exclusions because it views them as analysis-specific and potentially biasing for a general decoder.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start simply by slicing each trial from the shared session timeline using the `[start, stop)` interval defined from `trial_start` and `teleport`. No additional shift or interpolation is applied.

ii.
```python
for j, (start, stop) in enumerate(trials):
    q = slice(start, stop)
    ...
    neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
```

iii. The AI’s notes say the released behavior and ophys rows are already on the same frame grid, so start alignment only requires slicing by the trial boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native imaging/behavior sample grid, with median spacing checked against `1 / 15.5078125` s. No temporal rebinning or resampling is performed.

ii.
```python
DT_REFERENCE = 1.0 / 15.5078125
...
timestamps = np.asarray(b['position/timestamps'][:], dtype=np.float64)
dt = float(np.median(np.diff(timestamps)))
if not np.isclose(dt, DT_REFERENCE, rtol=0, atol=2e-5):
    raise ValueError(f'Unexpected behavior dt {dt}')
...
time_bin_size=float(np.median(dts)*1000)
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 10, the AI says the data are already synchronized at about 15.5 Hz per plane and should remain on that native time grid.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The AI derives time-from-start from the behavior timestamps attached to the `position` stream.

ii.
```python
timestamps = np.asarray(b['position/timestamps'][:], dtype=np.float64)
...
inp = np.vstack([
    (timestamps[q]-timestamps[start]).astype(np.float32),
    ...
])
```

iii. The AI’s notes say the frame-aligned behavior streams share the same timestamp grid, so choosing `position/timestamps` was sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the timestamp at the trial’s first sample from every timestamp in that trial.

ii.
```python
(timestamps[q]-timestamps[start]).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI describes this as direct start-aligned frame time on the native sample grid.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time input is aligned to neural data by using the exact same trial slice `q` on the shared session timeline. The AI also trims the neural array to the behavior length if the session has one extra trailing neural frame.

ii.
```python
if neural_excess_frames:
    neural_all = neural_all[:ntime]
...
q = slice(start, stop)
...
neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
...
(timestamps[q]-timestamps[start]).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 10, the AI says the released ophys and behavior rows are already aligned, and only the known +1 trailing-frame case needs correction.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI derives environment type from the behavior stream `environment`.

ii.
```python
names = ['environment', 'position', 'speed', 'lick', 'trial number']
streams = {n: np.asarray(b[n+'/data'][:]) for n in names}
...
env_values = np.unique(streams['environment'][q])
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 5, the AI says the framewise `environment` stream is authoritative and already coded as 0/1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the AI checks that `environment` is constant and binary, then repeats that single value across all timepoints in the trial.

ii.
```python
env_values = np.unique(streams['environment'][q])
env_values = env_values[env_values >= 0]
if len(env_values) != 1 or env_values[0] not in (0, 1):
    raise ValueError(f'Non-binary/nonconstant environment trial {j}: {env_values}')
env = int(env_values[0])
...
np.full(n, env, dtype=np.float32)
```

iii. The AI’s notes justify this as a per-trial decoder input and treat any within-trial change as a data error.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The AI derives trial number from the stored behavior stream `trial number`, specifically the value at the trial’s start frame.

ii.
```python
names = ['environment', 'position', 'speed', 'lick', 'trial number']
...
trial_id = int(round(float(streams['trial number'][start])))
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says it chose to preserve the source trial ID rather than replace it with a purely local loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The AI rounds the `trial number` value at the start frame to an integer and repeats that constant value across the whole trial.

ii.
```python
trial_id = int(round(float(streams['trial number'][start])))
...
np.full(n, trial_id, dtype=np.float32)
```

iii. The notes describe this as preserving the original trial identity from the aligned behavior stream.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from `Reward/timestamps`, together with the trial start/stop timestamps used to determine whether each trial contained a reward.

ii.
```python
def reward_outcomes(b, timestamps, trials):
    reward_t = np.asarray(b['Reward/timestamps'][:], dtype=float)
    return np.asarray([np.any((reward_t >= timestamps[a]) & (reward_t < timestamps[z]))
                       for a, z in trials], dtype=np.int64)
```

iii. In `CONVERSION_NOTES.md` Steps 5 and 10, the AI says sparse reward timestamps are the direct evidence for trial outcome and should drive previous-outcome labels.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a Boolean outcome for every complete trial, then for trial `j` uses outcome `j-1`; the first trial gets 0. That value is repeated across all frames in the current trial.

ii.
```python
outcomes = reward_outcomes(b, timestamps, trials)
...
prev_outcome = int(outcomes[j-1]) if j > 0 else 0
...
np.full(n, prev_outcome, dtype=np.float32)
```

iii. The notes explicitly say previous outcome should follow the immediately preceding source trial, even if that previous trial is later excluded for bad licking.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The AI derives distance to reward zone from the `position` stream plus a per-trial reward-zone identity inferred from the NWB `identifier` string. It parses one or two A/B/C labels from the scene name and applies a switch after trial 30 when two labels are present.

ii.
```python
identifier = h['identifier'][()].decode()
scene, labels = parse_scene(identifier)
...
zone = zone_for_trial(labels, j)
zstart, zstop = ZONE_COORDS[zone]
pos = streams['position'][q].astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 10, the AI argues that the framewise `reward_zone` codes are event-state values rather than zone identity, and that parsing the scene schedule better matches the paper’s `get_reward_zones` logic.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the AI computes signed distance to the nearest point in the current trial’s reward-zone interval: negative before the zone, zero inside the zone, positive after the zone. It returns both the discrete class and the continuous distance.

ii.
```python
def distance_classes(position, start, stop):
    distance = np.where(position < start, position-start,
                        np.where(position > stop, position-stop, 0.0))
    ...
    return out, distance
```

iii. The AI’s notes describe this as “distance to any location in the zone,” so inside-zone samples must map to exactly zero distance.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Thresholding is hard-coded into seven classes: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50` cm.

ii.
```python
out[distance < -50] = 0
out[(distance >= -50) & (distance < -10)] = 1
out[(distance >= -10) & (distance < 0)] = 2
out[distance == 0] = 3
out[(distance > 0) & (distance <= 10)] = 4
out[(distance > 10) & (distance <= 50)] = 5
out[distance > 50] = 6
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says these boundaries were chosen to match the decoder specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance-to-zone classes are computed from `position[q]` using the exact same trial slice `q` that is used for neural data.

ii.
```python
q = slice(start, stop)
...
pos = streams['position'][q].astype(np.float32)
...
dclass, distance = distance_classes(pos, zstart, zstop)
...
neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
```

iii. The AI’s notes say all framewise streams are already on the same aligned grid, so common slicing is enough.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the behavior stream `position`.

ii.
```python
names = ['environment', 'position', 'speed', 'lick', 'trial number']
...
pos = streams['position'][q].astype(np.float32)
```

iii. The notes describe `position` as the authoritative framewise track coordinate.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices `position` per trial and discretizes it into 90 cm bins over the 450 cm track using `np.digitize`.

ii.
```python
posclass = np.digitize(pos, [90, 180, 270, 360], right=False).astype(np.int64)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says the task asks for five equal-width bins, so direct thresholding of raw position is sufficient.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into five bins with edges at 90, 180, 270, and 360 cm.

ii.
```python
posclass = np.digitize(pos, [90, 180, 270, 360], right=False).astype(np.int64)
```

iii. The AI’s notes describe these as the requested equal-size track bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position classes and neural activity are aligned by applying the same per-trial slice `q` to both streams.

ii.
```python
q = slice(start, stop)
pos = streams['position'][q].astype(np.float32)
neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
```

iii. The AI justifies this with the already aligned behavior/ophys frame grid discussed in `CONVERSION_NOTES.md`.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick output is derived from the behavior stream `lick`.

ii.
```python
names = ['environment', 'position', 'speed', 'lick', 'trial number']
...
lick_raw = streams['lick'][q]
```

iii. The AI’s notes describe the NWB lick stream as cumulative lick-count samples that must be binarized after corruption filtering.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes lick samples as `lick > 0`, after first excluding the entire trial if lick corruption exceeds the 30% `lick > 2` criterion.

ii.
```python
if np.mean(lick_raw > 2) > 0.30:
    bad_lick += 1
    continue
...
lickclass = (lick_raw > 0).astype(np.int64)
```

iii. In `CONVERSION_NOTES.md` Steps 3, 5, 9, and 10, the AI says this follows the paper’s sensor-corruption rule and requested binary lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned to neural activity by taking the same trial slice `q` on the shared session timeline.

ii.
```python
q = slice(start, stop)
lick_raw = streams['lick'][q]
...
neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
```

iii. The AI relies on the released frame alignment rather than any extra interpolation.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the NWB `identifier` string, parsed into one or two A/B/C labels, together with the chronological trial index used to decide whether a switch-session trial is before or after trial 30.

ii.
```python
def parse_scene(identifier):
    scene = identifier.rstrip('/').split('/')[-1]
    labels = re.findall(r'(?:Location)?([ABC])', scene)
    ...
def zone_for_trial(labels, chronological_index):
    return labels[0] if len(labels) == 1 or chronological_index < 30 else labels[1]
...
identifier = h['identifier'][()].decode()
scene, labels = parse_scene(identifier)
zone = zone_for_trial(labels, j)
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 10, the AI says the schedule encoded in the scene name is the correct source of per-trial reward-zone identity.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene labels, applies the `trial < 30` switch rule when there are two labels, converts A/B/C to 0/1/2, and repeats that zone label across the whole trial.

ii.
```python
zone = zone_for_trial(labels, j)
...
np.full(n, ZONE_INDEX[zone], dtype=np.int64)
```

iii. The AI’s notes explicitly say this was meant to mirror the paper’s `get_reward_zones(..., change_trial=30)` behavior.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps`, compared against each trial’s start and stop timestamps.

ii.
```python
def reward_outcomes(b, timestamps, trials):
    reward_t = np.asarray(b['Reward/timestamps'][:], dtype=float)
    return np.asarray([np.any((reward_t >= timestamps[a]) & (reward_t < timestamps[z]))
                       for a, z in trials], dtype=np.int64)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says delivered reward events are the direct ground truth for outcome and should not be inferred indirectly.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The AI precomputes a Boolean outcome for each complete trial by checking whether any reward timestamp falls inside that trial’s `[start, stop)` window, then repeats that 0/1 label across every frame of the trial.

ii.
```python
outcomes = reward_outcomes(b, timestamps, trials)
...
out = np.vstack([
    dclass, posclass, speedclass, lickclass,
    np.full(n, ZONE_INDEX[zone], dtype=np.int64),
    np.full(n, outcomes[j], dtype=np.int64),
])
```

iii. The AI’s notes say the output is intentionally trial-level and repeated framewise because that is the requested format.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly uses strict validation, but it has one explicit repair path: if neural data are exactly one frame longer than behavior, it trims the extra trailing neural row. It otherwise raises errors on behavior-length mismatches, unexpected sample intervals, non-binary/nonconstant environment trials, plane-length mismatches, and nonfinite values. It also drops lick-corrupt trials.

ii.
```python
if any(len(x) != ntime for x in streams.values()):
    raise ValueError('Behavior stream length mismatch')
...
if not np.isclose(dt, DT_REFERENCE, rtol=0, atol=2e-5):
    raise ValueError(f'Unexpected behavior dt {dt}')
...
if neural_excess_frames not in (0, 1):
    raise ValueError(f'Unexpected neural/behavior row difference: {neural_all.shape[0]} vs {ntime}')
if neural_excess_frames:
    neural_all = neural_all[:ntime]
...
if not (np.isfinite(neu).all() and np.isfinite(inp).all()):
    raise ValueError('Nonfinite neural/input values')
```

iii. In `CONVERSION_NOTES.md` Steps 6, 9, and 10, the AI says it found exactly ten two-plane sessions with a trailing extra neural row and added a strict `0 or +1 only` trim for that known release artifact; it preferred hard failures for anything else.

## 13-a. What are the most time-consuming steps of the code?

i. The AI’s notes identify loading large neural matrices and writing the very large pickle as the dominant costs. The code also spends substantial time looping over all sessions and slicing variable-length trials, but it intentionally avoids the much slower fluorescence-to-dF/F-to-events recomputation.

ii.
```python
arrays = [np.asarray(dec[p]['data'][:], dtype=np.float32) for p in plane_names]
...
for i,f in enumerate(files):
    ns,xs,ys,info=process_session(f, make_plot=args.show_processing and i<2)
...
with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
```

iii. `CONVERSION_NOTES.md` Step 6 says “Large neural matrices dominate I/O and pickle size” and presents “read only released Deconvolved arrays” as the main speed optimization.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious remaining non-vectorized loop is the per-trial loop inside `process_session`. Class thresholding is already vectorized within a trial, but trial-level slicing, bad-lick exclusion, `np.unique` environment checks, and repeated `np.full` allocations still happen one trial at a time.

ii.
```python
for j, (start, stop) in enumerate(trials):
    q = slice(start, stop)
    ...
    env_values = np.unique(streams['environment'][q])
    ...
    inp = np.vstack([...])
    out = np.vstack([...])
```

iii. The AI did not emphasize this in detail, but its Step 6 notes do say it already vectorized class construction and kept one-session-at-a-time processing because I/O and memory, not scalar math, were the main bottlenecks.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly slices session arrays per trial and repeatedly allocates constant per-trial arrays for environment, trial number, previous outcome, reward-zone location, and reward outcome. It also recomputes `np.unique` on the environment stream for every trial.

ii.
```python
for j, (start, stop) in enumerate(trials):
    q = slice(start, stop)
    ...
    env_values = np.unique(streams['environment'][q])
    ...
    np.full(n, env, dtype=np.float32)
    np.full(n, trial_id, dtype=np.float32)
    np.full(n, prev_outcome, dtype=np.float32)
    ...
    np.full(n, ZONE_INDEX[zone], dtype=np.int64)
    np.full(n, outcomes[j], dtype=np.int64)
```

iii. The AI’s notes do not call out all of these individually, but they do state that the code prioritizes simple one-pass session processing over more elaborate reuse or precomputation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores some data that are not needed by the decoder itself: continuous `distance` values are kept only for metadata summaries, `curated_plane_idx` and detailed `kept_info` are saved into `session_info`, and optional plotting code builds figures not used downstream. It also returns `plane_names` only to store them in metadata.

ii.
```python
dclass, distance = distance_classes(pos, zstart, zstop)
...
kept_info.append(dict(source_trial_index=j, trial_id=trial_id, start=start, stop=stop,
                      zone=zone, outcome=int(outcomes[j]), environment=env,
                      distance_min=float(distance.min()), distance_max=float(distance.max())))
...
info = dict(..., planes=plane_names, curated_plane_idx=curated_planes, ..., trials=kept_info, ...)
...
if make_plot:
    plot_processing(path, neural_trials, input_trials, output_trials, info)
```

iii. The AI’s notes frame these as diagnostics and provenance rather than core decoder inputs; they are retained for validation and documentation rather than because training requires them.
