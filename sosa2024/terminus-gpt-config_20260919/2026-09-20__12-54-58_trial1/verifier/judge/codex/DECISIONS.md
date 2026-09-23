# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every `.nwb` file under `/app/data` with `Path.rglob('*.nwb')`, treats each file as one session, and reads the file directly with `h5py` rather than `pynwb`.

ii.
```python
DATA_ROOT = Path('/app/data')
...
def main():
    ...
    files=sorted(DATA_ROOT.rglob('*.nwb'))
    ...

def load_session(path):
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        ...
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 6, the AI says the dataset is a directory tree of NWB/HDF5 files and emphasizes direct HDF5 reads for efficiency.

## 1-b. How are the data split into subjects?

i. Subjects are split by the parent directory name of each NWB file (`sub-<mouse>`), and those names are normalized by removing the `sub-` prefix.

ii.
```python
subjects = sorted({p.parent.name.replace('sub-','') for p in files},
                  key=lambda x: int(re.sub(r'\D','',x)))
subject_lookup = {s:i for i,s in enumerate(subjects)}
...
info_clean['subject_idx']=subject_lookup[info['subject']]
```

iii. The notes state that `/app/data` has one directory per subject and list the 11 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session, and the session identifier saved in metadata is the file stem.

ii.
```python
for i,p in enumerate(files):
    q0=time.perf_counter(); n,x,y,info=load_session(p)
    ...

info = dict(path=str(path), session=path.stem, identifier=identifier,
            subject=subject, ...)
```

iii. The notes explicitly say there is one `*_behavior+ophys.nwb` file per session.

## 1-d. How are the data split into trials?

i. Trials are split by pairing each positive `trial_start` sample with the first later positive `teleport` sample, then slicing from `trial_start` inclusive to `teleport` exclusive.

ii.
```python
def pair_trials(start_signal, teleport_signal, n_samples):
    starts = np.flatnonzero(start_signal > 0)
    teleports = np.flatnonzero(teleport_signal > 0)
    pairs = []
    for s in starts:
        k = np.searchsorted(teleports, s, side='left')
        if k < len(teleports) and teleports[k] > s:
            pairs.append((int(s), int(teleports[k])))  # end is exclusive
    return pairs
...
sl = slice(s, e)  # start included; teleport excluded
```

iii. In Step 5, the AI justifies the trial window as “trial_start sample inclusive to teleport sample exclusive.”

## 1-e. How are trials filtered based on quality controls?

i. The AI does almost no trial-quality filtering. It skips degenerate pairs where `e <= s`, but otherwise keeps all trials and does not apply a minimum-length threshold.

ii.
```python
for j, (s, e) in enumerate(pairs):
    if e <= s:
        continue
    sl = slice(s, e)
    ...
    if not (neu.shape[1] == inp.shape[1] == out.shape[1] > 0):
        raise ValueError(...)
```

iii. The notes say every session already has at least 41 complete trials and therefore all sessions/trials were retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the NWB `processing/ophys/Deconvolved/plane*/data` arrays plus ROI metadata from `ImageSegmentation/PlaneSegmentation` (`iscell`, `planeIdx`).

ii.
```python
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell0 = np.asarray(seg['iscell'])
iscell = (iscell0[:, 0] if iscell0.ndim > 1 else iscell0) > 0
plane_idx = np.asarray(seg['planeIdx']).astype(int)
...
for plane_name in sorted(f['processing/ophys/Deconvolved'].keys()):
    plane = int(plane_name.replace('plane',''))
    global_ids = np.flatnonzero(plane_idx == plane)
    ds = f['processing/ophys/Deconvolved'][plane_name]['data']
```

iii. In Step 4 and Step 5, the AI explicitly decided to use the supplied deconvolved activity rather than recompute it.

## 2-b. How is the `neural` data processed?

i. The AI does not recompute dF/F or deconvolution. It selects `iscell` ROIs, reads deconvolved traces plane by plane, trims them to the behavior length if needed, concatenates planes across cells, and transposes per trial to `(neurons, time)`.

ii.
```python
local_keep = np.flatnonzero(iscell[global_ids])
arr = np.asarray(ds[:, local_keep], dtype=np.float32)
pieces.append(arr[:n_beh])
...
neural_tn = np.concatenate(pieces, axis=1)
...
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
```

iii. The notes justify this as using “supplied `Deconvolved` activity, avoiding an irreproducible second deconvolution.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neural data only by Suite2p `iscell` and does not remove putative interneurons or any additional cell classes.

ii.
```python
iscell0 = np.asarray(seg['iscell'])
iscell = (iscell0[:, 0] if iscell0.ndim > 1 else iscell0) > 0
...
local_keep = np.flatnonzero(iscell[global_ids])
```

iii. Step 5 says “Retain `iscell==1` only” and explicitly says not to impose place/RR/TR labels or other filters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned by using the same trial-start-based slices used for behavior, so each trial begins at the `trial_start` sample.

ii.
```python
pairs = pair_trials(beh['trial_start'], beh['teleport'], n_beh)
...
sl = slice(s, e)  # start included; teleport excluded
...
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
```

iii. The notes describe the alignment event as the `trial_start` pulse / entry onto the track.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses a fixed native sample period `DT = 1/15.5078125` seconds (about 64.48 ms) and does no temporal rebinning.

ii.
```python
DT = 1.0 / 15.5078125
...
'time_bin_size':DT*1000.0,
```

iii. The notes say the synchronized NWB rows are already at the native volume rate and should be preserved without rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps, specifically `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii.
```python
timestamps = np.asarray(b['position']['timestamps'], dtype=np.float64)
...
reltime = (timestamps[sl] - timestamps[s]).astype(np.float32)
```

iii. The AI treated the behavior timestamp stream as the session clock for all aligned variables.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the first timestamp of the slice so the trial starts at 0 seconds.

ii.
```python
reltime = (timestamps[sl] - timestamps[s]).astype(np.float32)
...
inp = np.vstack([
    reltime,
    ...
])
```

iii. The code also checks that the first time bin is effectively zero:

```python
if abs(float(inp[0,0])) > 1e-7 ...
```

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by slicing timestamps and neural rows with the same `slice(s, e)` indices for each trial.

ii.
```python
sl = slice(s, e)
reltime = (timestamps[sl] - timestamps[s]).astype(np.float32)
...
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
if not (neu.shape[1] == inp.shape[1] == out.shape[1] > 0):
    raise ValueError(...)
```

iii. The notes repeatedly state that behavior and neural streams are already synchronized row-wise in the NWB exports.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the synchronized behavior variable `environment`.

ii.
```python
aligned_names = ['autoreward','environment','lick','position','reward_zone',
                 'scanning','speed','teleport','trial number','trial_start']
beh = {k: np.asarray(b[k]['data']) for k in aligned_names}
...
env_values = np.asarray(beh['environment'][sl])
```

iii. The notes identify `environment` as the native stream for ENV1 vs ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI takes the median/mode-like value across the trial and repeats that single code across all trial time bins.

ii.
```python
env_values = np.asarray(beh['environment'][sl])
env = int(np.rint(np.median(env_values)))
...
np.full(len(pos), env, np.float32),
```

iii. In the notes, the AI says environment is effectively per-trial and constant within complete trials.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the raw `trial number` behavior stream, using the value at the trial start sample.

ii.
```python
source_trial = float(beh['trial number'][s])
...
np.full(len(pos), source_trial, np.float32),
```

iii. Step 5 says to “Preserve source 0-based number,” rather than recomputing a separate loop index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing beyond reading the trial-start value and repeating it across the trial.

ii.
```python
source_trial = float(beh['trial number'][s])
...
np.full(len(pos), source_trial, np.float32),
```

iii. The AI treated trial number as a per-trial scalar made time-aligned by repetition.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward-event timestamps (`Reward/timestamps`) after the AI converts them into per-trial reward outcomes.

ii.
```python
reward_times = np.asarray(b['Reward']['timestamps'], dtype=np.float64)
...
outcomes = np.asarray([
    int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))
    for s, e in pairs
], dtype=np.int64)
...
prev = int(outcomes[j-1]) if j > 0 else 0
```

iii. The notes describe outcomes as being assigned from sparse reward timestamps to trial intervals.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a binary outcome for each trial (`1` if any reward event falls in that trial), then sets the current trial’s previous outcome to the prior trial’s outcome; the first trial gets `0`.

ii.
```python
outcomes = np.asarray([
    int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))
    for s, e in pairs
], dtype=np.int64)
...
prev = int(outcomes[j-1]) if j > 0 else 0
...
np.full(len(pos), prev, np.float32),
```

iii. Step 5 says “Previous outcome is within-session only; first retained trial is 0.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from trial-sliced `position` plus a per-trial reward-zone interval inferred from the session `identifier`/scene string, not from the frame-wise `reward_zone` stream.

ii.
```python
identifier = scalar_text(f['identifier'])
...
labels, zone_coords, zone_codes = scene_zone_sequence(scene_from_identifier(identifier), len(pairs))
...
pos = np.asarray(beh['position'][sl], dtype=np.float32)
...
zstart, zstop = zone_coords[j]
signed_dist = distance_to_interval(pos, zstart, zstop)
```

iii. In Step 4 and Step 5, the AI says the exported `reward_zone` stream is uninformative and that zone identity should instead come from scene labels and switch structure.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes signed distance to the reward-zone interval: negative before the zone, zero inside it, positive after it.

ii.
```python
def distance_to_interval(pos, start, stop):
    return np.where(pos < start, pos - start, np.where(pos > stop, pos - stop, 0.0))
...
signed_dist = distance_to_interval(pos, zstart, zstop)
```

iii. The notes say the transform should be reward-relative position using exact A/B/C intervals.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI discretizes the signed distance into 7 categories with the specified boundaries: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

ii.
```python
def bin_distance(x):
    y = np.empty(x.shape, np.int64)
    y[x < -50] = 0
    y[(x >= -50) & (x < -10)] = 1
    y[(x >= -10) & (x < 0)] = 2
    y[x == 0] = 3
    y[(x > 0) & (x <= 10)] = 4
    y[(x > 10) & (x <= 50)] = 5
    y[x > 50] = 6
    return y
```

iii. Step 5 lists these exact category boundaries.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same per-trial time slice as neural and other behavior variables.

ii.
```python
sl = slice(s, e)
pos = np.asarray(beh['position'][sl], dtype=np.float32)
...
signed_dist = distance_to_interval(pos, zstart, zstop)
...
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
```

iii. The code enforces equal time lengths for `neu`, `inp`, and `out` on every trial.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the synchronized behavior variable `position`.

ii.
```python
pos = np.asarray(beh['position'][sl], dtype=np.float32)
...
bin_position(pos)
```

iii. The notes identify native track position as the source for absolute position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI simply slices trial position and discretizes it; it does not smooth or resample.

ii.
```python
pos = np.asarray(beh['position'][sl], dtype=np.float32)
...
out = np.vstack([
    bin_distance(signed_dist), bin_position(pos), ...
])
```

iii. The notes say temporal samples are kept natively rather than position-binned as in some paper analyses.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into 5 bins: `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360`.

ii.
```python
def bin_position(x):
    y = np.empty(x.shape, np.int64)
    y[x < 90] = 0
    y[(x >= 90) & (x < 180)] = 1
    y[(x >= 180) & (x < 270)] = 2
    y[(x >= 270) & (x <= 360)] = 3
    y[x > 360] = 4
    return y
```

iii. Step 5 lists these exact bin inequalities.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by slicing `position` and neural rows with the same per-trial indices.

ii.
```python
sl = slice(s, e)
pos = np.asarray(beh['position'][sl], dtype=np.float32)
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
```

iii. The AI assumes and validates row-wise synchronization of behavior and neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the synchronized behavior variable `lick`.

ii.
```python
lick = (np.asarray(beh['lick'][sl]) > 0).astype(np.int64)
```

iii. The notes identify lick as a native frame-aligned behavior stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick is binarized by thresholding the raw values at `> 0`.

ii.
```python
lick = (np.asarray(beh['lick'][sl]) > 0).astype(np.int64)
...
out = np.vstack([
    ..., lick,
    ...
])
```

iii. Step 5 explicitly maps lick to a binary output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by slicing the lick stream with the same per-trial indices as the neural matrix.

ii.
```python
sl = slice(s, e)
lick = (np.asarray(beh['lick'][sl]) > 0).astype(np.int64)
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
```

iii. The per-trial shape check ensures the alignment is preserved.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the session `identifier` string parsed into a scene label and converted into a trial-by-trial zone sequence.

ii.
```python
identifier = scalar_text(f['identifier'])
...
labels, zone_coords, zone_codes = scene_zone_sequence(scene_from_identifier(identifier), len(pairs))
```

iii. The notes say the exported `reward_zone` frame stream should not be trusted and zone identity should be reconstructed from session metadata plus switch structure.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses either a fixed zone (`LocationA/B/C`) or a switch scene (`A_to_B`, etc.), assumes the switch occurs at trial 30, converts zones A/B/C to codes 0/1/2, and repeats the code across each trial.

ii.
```python
def scene_zone_sequence(scene, n_trials, change_trial=30):
    fixed = re.search(r'Location([ABC])$', scene)
    if fixed:
        labels = [fixed.group(1)] * n_trials
    else:
        m = re.search(r'(?:Location)?([ABC])_to_(?:Env[123]_)?(?:Location)?([ABC])$', scene)
        ...
        labels = [before] * c + [after] * (n_trials - c)
    coords = np.asarray([ZONE_INTERVALS[x] for x in labels], dtype=np.float32)
    codes = np.asarray([ZONE_CODES[x] for x in labels], dtype=np.int64)
    return labels, coords, codes
...
np.full(len(pos), zone_codes[j], np.int64),
```

iii. Step 5 calls this “reference scene-to-zone mapping” and says omission trials inherit the scene-defined zone.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, compared against each trial’s time window.

ii.
```python
reward_times = np.asarray(b['Reward']['timestamps'], dtype=np.float64)
...
outcomes = np.asarray([
    int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))
    for s, e in pairs
], dtype=np.int64)
```

iii. The notes describe reward as a sparse timestamped event series rather than a frame-aligned variable.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Each trial gets `1` if any reward timestamp falls within that trial’s interval and `0` otherwise, and the result is repeated across time bins in the output matrix.

ii.
```python
outcomes = np.asarray([
    int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))
    for s, e in pairs
], dtype=np.int64)
...
np.full(len(pos), outcomes[j], np.int64),
```

iii. Step 5 says “1 iff any reward event timestamp falls from trial start through paired teleport, else 0.”

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles only a few edge cases: it raises an error if frame-aligned behavior streams disagree in length, trims neural arrays to the behavior length when neural has an extra terminal row, skips degenerate trial pairs, and raises errors on nonfinite or misaligned per-trial data. It does not implement a missing-data imputation strategy.

ii.
```python
if any(len(v) != n_beh for v in beh.values()):
    raise ValueError(f'Frame-aligned behavior length mismatch in {path.name}')
...
pieces.append(arr[:n_beh])
...
for j, (s, e) in enumerate(pairs):
    if e <= s:
        continue
...
if abs(float(inp[0,0])) > 1e-7 or np.any(~np.isfinite(neu)) or np.any(~np.isfinite(inp)):
    raise ValueError(f'Nonfinite data or bad alignment {path.name} trial {j}')
```

iii. The notes emphasize that the main observed edge case was ten files with one extra neural row, which should be truncated rather than interpolated.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are reading each NWB session from disk, reading large deconvolved matrices from HDF5, iterating over all sessions/trials to materialize trial arrays, and serializing the large pickle output.

ii.
```python
for i,p in enumerate(files):
    q0=time.perf_counter(); n,x,y,info=load_session(p)
    ...
with out.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 6 and Step 9 discuss HDF5 column reads, total runtime, and the 9.8 GB output file as the dominant cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious loops are the `pair_trials` loop over start indices, the per-trial loop in `load_session`, and the per-session loop in `convert`. Within `load_session`, several constant-per-trial quantities are allocated trial by trial instead of being precomputed in session form.

ii.
```python
for s in starts:
    k = np.searchsorted(teleports, s, side='left')
    ...
for j, (s, e) in enumerate(pairs):
    ...
for i,p in enumerate(files):
    q0=time.perf_counter(); n,x,y,info=load_session(p)
```

iii. The AI’s notes say it already vectorized discretization and HDF5 column selection, so the loops left are mainly structural trial/session loops.

## 13-c. What processing does the code repeat multiple times?

i. Relative to the human reference, the AI intentionally avoids a separate survey pass and repeated neural preprocessing. The main repeated work left is small per-trial allocation/repetition inside `load_session`, not a second read of every file.

ii.
```python
for j, (s, e) in enumerate(pairs):
    ...
    inp = np.vstack([
        reltime,
        np.full(len(pos), env, np.float32),
        np.full(len(pos), source_trial, np.float32),
        np.full(len(pos), prev, np.float32),
    ])
```

iii. Step 6 explicitly says the script avoids “redundant deconvolution” and processes one session at a time.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores `plot_info` for every session even when plots are not requested, reads some aligned behavior variables that are not used for outputs (`autoreward`, `reward_zone`, `scanning`), and builds `selected_global`, `pairs`, `timestamps`, `outcomes`, and `zone_labels` in `info` even though most of them are removed before saving.

ii.
```python
aligned_names = ['autoreward','environment','lick','position','reward_zone',
                 'scanning','speed','teleport','trial number','trial_start']
...
selected_global = []
...
plot_info = []
...
plot_info.append((pos, speed, lick, signed_dist, out, labels[j]))
...
info_clean={k:v for k,v in info.items() if k not in ('plot_info','timestamps','pairs','selected_global','outcomes','zone_labels')}
```

iii. There is no strong explicit justification for this in the notes; the extra structures appear to support diagnostics and development convenience rather than the final converted dataset.
