# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `h5py` to directly read the NWB/HDF5 files. All `.nwb` files are found by recursively globbing `DATA_ROOT.rglob('*.nwb')` under `/app/data`. Each file is loaded with `h5py.File(path, 'r')` and behavioral + neural data are extracted via HDF5 paths.

ii.
```python
files=sorted(DATA_ROOT.rglob('*.nwb'))
...
def load_session(path):
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        aligned_names = ['autoreward','environment','lick','position','reward_zone',
                         'scanning','speed','teleport','trial number','trial_start']
        beh = {k: np.asarray(b[k]['data']) for k in aligned_names}
        timestamps = np.asarray(b['position']['timestamps'], dtype=np.float64)
        reward_times = np.asarray(b['Reward']['timestamps'], dtype=np.float64)
```

iii. The AI chose `h5py` for direct HDF5 access rather than `pynwb`, reading all behavior and neural arrays from known HDF5 paths. The CONVERSION_NOTES document that 152 NWB files were found across 11 subjects.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from NWB file internal metadata (`general/subject/subject_id`), then deduplicated and sorted by numeric component.

ii.
```python
subjects = sorted({p.parent.name.replace('sub-','') for p in files}, key=lambda x: int(re.sub(r'\D','',x)))
...
subject = scalar_text(f['general/subject/subject_id'])
```

iii. The AI reads the subject ID from each NWB file's metadata. The parent directory name (`sub-*`) is used for sorting, but the internal metadata provides the actual subject ID. 11 subjects are identified.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All files are processed in sorted order.

ii.
```python
files=sorted(DATA_ROOT.rglob('*.nwb'))
...
for i,p in enumerate(files):
    q0=time.perf_counter(); n,x,y,info=load_session(p)
```

iii. The NWB files have one session each, identified by the filename pattern `sub-*_ses-*_behavior+ophys.nwb`.

## 1-d. How are the data split into trials?

i. Trials are identified using the `trial_start` and `teleport` behavior signals. The `pair_trials` function finds each `trial_start > 0` sample and pairs it with the next `teleport > 0` sample. The trial spans from start (inclusive) to teleport (exclusive).

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
```

iii. The AI uses `trial_start` and `teleport` signals to define trial boundaries, consistent with the paper's definition of trials as track traversals between entry and teleport.

## 1-e. How are trials filtered based on quality controls?

i. Trials are only skipped if `e <= s` (empty trial). There is no minimum trial length filter.

ii.
```python
for j, (s, e) in enumerate(pairs):
    if e <= s:
        continue
```

iii. The AI does not apply a minimum trial length filter. No justification is given for this omission.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the pre-computed `Deconvolved` data from the NWB file (`processing/ophys/Deconvolved/plane*/data`). It does NOT use the raw Fluorescence or Neuropil traces.

ii.
```python
for plane_name in sorted(f['processing/ophys/Deconvolved'].keys()):
    plane = int(plane_name.replace('plane',''))
    global_ids = np.flatnonzero(plane_idx == plane)
    ds = f['processing/ophys/Deconvolved'][plane_name]['data']
    ...
    arr = np.asarray(ds[:, local_keep], dtype=np.float32)
    pieces.append(arr[:n_beh])
```

iii. The CONVERSION_NOTES state: "Use supplied `Deconvolved` activity, avoiding an irreproducible second deconvolution; select `iscell`." The AI reasoned that the NWB already contains processed Suite2p output and that recomputing dF/F would be redundant. However, the NWB's `Deconvolved` is Suite2p's default deconvolution, not the paper's custom processing pipeline.

## 2-b. How is the `neural` data processed?

i. The AI performs no neural signal processing. The pre-computed Deconvolved data is read directly, filtered by `iscell`, and used as-is.

ii.
```python
arr = np.asarray(ds[:, local_keep], dtype=np.float32)
pieces.append(arr[:n_beh])
...
neural_tn = np.concatenate(pieces, axis=1)
...
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
```

iii. The AI explicitly decided not to recompute dF/F or run deconvolution, stating "Avoid recomputing dF/F/OASIS because processed deconvolved activity is supplied." The CONVERSION_NOTES acknowledge the paper's processing pipeline (neuropil subtraction, maximin baseline, dF/F, smoothing, OASIS) but argue that the supplied Deconvolved data is equivalent.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `iscell` filtering is applied. Cells not marked as `iscell==1` in Suite2p's ROI table are excluded. No putative interneuron filtering is performed.

ii.
```python
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell0 = np.asarray(seg['iscell'])
iscell = (iscell0[:, 0] if iscell0.ndim > 1 else iscell0) > 0
...
local_keep = np.flatnonzero(iscell[global_ids])
arr = np.asarray(ds[:, local_keep], dtype=np.float32)
```

iii. The CONVERSION_NOTES state: "Apply `iscell` only; do not restrict to functional subsets." The AI considered place/RR/TR labels as "analysis-specific" and not required for a general decoder. However, the paper's Methods describe filtering putative interneurons (dF/F speed correlation > 0.5) as a general preprocessing step, not an analysis-specific filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by slicing neural data at the trial_start index. Since neural and behavioral data share the same time indices, no additional alignment is needed beyond trial slicing.

ii.
```python
sl = slice(s, e)  # start included; teleport excluded
...
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
```

iii. The instructions specify alignment to trial start. The AI correctly uses the trial_start sample as the beginning of each trial slice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native temporal resolution is preserved at ~15.5 Hz (DT = 1/15.5078125 s ≈ 64.48 ms). No temporal rebinning is applied.

ii.
```python
DT = 1.0 / 15.5078125
...
'time_bin_size':DT*1000.0,
```

iii. The AI uses the known frame rate from the data. The paper states "All behavioral and neural time series were sampled at ~15.5 Hz."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the behavior `timestamps` array (specifically from `b['position']['timestamps']`).

ii.
```python
timestamps = np.asarray(b['position']['timestamps'], dtype=np.float64)
...
reltime = (timestamps[sl] - timestamps[s]).astype(np.float32)
```

iii. All behavior time series share the same timestamps. The position timestamps are used.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial start timestamp is subtracted from all timestamps within the trial.

ii.
```python
reltime = (timestamps[sl] - timestamps[s]).astype(np.float32)
```

iii. Standard relative time computation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data use the same row indices (synchronized in the NWB), so no additional alignment is needed.

ii. Same slice `sl = slice(s, e)` is used for both neural and behavioral data.

iii. The AI verified that neural and behavior streams are already synchronized in NWB row order.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env_values = np.asarray(beh['environment'][sl])
env = int(np.rint(np.median(env_values)))
```

iii. The environment variable records ENV1 (0) vs ENV2 (1) per timepoint.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median of the environment values within the trial is taken and rounded to the nearest integer. This per-trial scalar is then broadcast across all timepoints.

ii.
```python
env = int(np.rint(np.median(env_values)))
inp = np.vstack([
    ...
    np.full(len(pos), env, np.float32),
    ...
])
```

iii. The AI takes the median to get a single per-trial value, then broadcasts. Since environment is constant within a trial, the median equals the constant value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series stored in the NWB file. Specifically, the value at the start of each trial is used.

ii.
```python
source_trial = float(beh['trial number'][s])
inp = np.vstack([
    ...
    np.full(len(pos), source_trial, np.float32),
    ...
])
```

iii. The AI uses the NWB's stored trial number rather than a sequential loop index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number at the first sample of the trial is read and broadcast as a constant across all timepoints in the trial.

ii.
```python
source_trial = float(beh['trial number'][s])
np.full(len(pos), source_trial, np.float32)
```

iii. No additional processing beyond reading and broadcasting.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps. For each trial, reward outcome is determined by checking if any reward event timestamp falls within the trial's time window.

ii.
```python
reward_times = np.asarray(b['Reward']['timestamps'], dtype=np.float64)
...
outcomes = np.asarray([
    int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))
    for s, e in pairs
], dtype=np.int64)
```

iii. The Reward time series has sparse timestamps. Each trial's outcome is computed first, then the previous trial's outcome is used as the input.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial j, the previous trial outcome is `outcomes[j-1]`. For the first trial (j=0), it defaults to 0.

ii.
```python
prev = int(outcomes[j-1]) if j > 0 else 0
inp = np.vstack([
    ...
    np.full(len(pos), prev, np.float32),
])
```

iii. Binary per-trial value: 0=omitted, 1=rewarded.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone intervals. The reward zone for each trial is determined from the session identifier (scene name) parsed via regex, using the `scene_zone_sequence` function.

ii.
```python
def scene_zone_sequence(scene, n_trials, change_trial=30):
    fixed = re.search(r'Location([ABC])$', scene)
    if fixed:
        labels = [fixed.group(1)] * n_trials
    else:
        m = re.search(r'(?:Location)?([ABC])_to_(?:Env[123]_)?(?:Location)?([ABC])$', scene)
        ...
        c = min(int(change_trial), n_trials)
        labels = [before] * c + [after] * (n_trials - c)
    coords = np.asarray([ZONE_INTERVALS[x] for x in labels], dtype=np.float32)
    ...
```

iii. The AI uses the session identifier to determine zone sequences, following the reference code's `get_reward_zones` logic. The `reward_zone` NWB field was found to be uninformative (constant 0), so the AI used scene parsing instead.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed from the animal's position to the nearest edge of the reward zone interval. Distance is 0 inside the zone, negative before, positive after. The continuous distance is then discretized.

ii.
```python
def distance_to_interval(pos, start, stop):
    return np.where(pos < start, pos - start, np.where(pos > stop, pos - stop, 0.0))
```

iii. Standard signed-distance computation matching the paper's reward-relative position concept.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The distance is binned into 7 categories using explicit conditional logic.

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

iii. The bin edges match the instructions: 0: <-50, 1: -50 to -10, 2: -10 to <0, 3: 0, 4: >0 to +10, 5: +10 to +50, 6: >+50.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices (same slice `sl`) used for both position and neural data within each trial.

ii.
```python
sl = slice(s, e)
pos = np.asarray(beh['position'][sl], dtype=np.float32)
...
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
```

iii. Neural and behavioral data are synchronized in the NWB row order.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = np.asarray(beh['position'][sl], dtype=np.float32)
```

iii. Direct position measurement in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
pos = np.asarray(beh['position'][sl], dtype=np.float32)
...
bin_position(pos)
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using explicit conditional logic.

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

iii. Five equal-sized 90 cm bins spanning the 450 cm track, matching the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same slice used for both position and neural data.

ii. Same `sl = slice(s, e)` applied to both.

iii. Synchronized NWB row order.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = (np.asarray(beh['lick'][sl]) > 0).astype(np.int64)
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick = (np.asarray(beh['lick'][sl]) > 0).astype(np.int64)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices within each trial slice.

ii. Same `sl = slice(s, e)`.

iii. Synchronized NWB row order.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the session identifier string parsed from the NWB file's `identifier` field. The `scene_zone_sequence` function determines the reward zone for each trial based on the scene name (LocationA, LocationB, LocationC, or switch patterns like LocationA_to_B).

ii.
```python
identifier = scalar_text(f['identifier'])
...
labels, zone_coords, zone_codes = scene_zone_sequence(scene_from_identifier(identifier), len(pairs))
```

iii. The AI found the NWB `reward_zone` field to be constant 0 and uninformative, so it used the session identifier to derive zone sequences, following the reference code's `get_reward_zones` logic.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed with regex to identify the reward zone pattern. For fixed-zone sessions, all trials get the same zone. For switch sessions, trials before `change_trial=30` get the first zone and trials after get the second zone. Zone labels are mapped to codes: A=0, B=1, C=2.

ii.
```python
def scene_zone_sequence(scene, n_trials, change_trial=30):
    fixed = re.search(r'Location([ABC])$', scene)
    if fixed:
        labels = [fixed.group(1)] * n_trials
    else:
        m = re.search(r'(?:Location)?([ABC])_to_(?:Env[123]_)?(?:Location)?([ABC])$', scene)
        before, after = m.groups()
        c = min(int(change_trial), n_trials)
        labels = [before] * c + [after] * (n_trials - c)
    ...
    codes = np.asarray([ZONE_CODES[x] for x in labels], dtype=np.int64)
```

iii. The `change_trial=30` default matches the reference code's `get_reward_zones` default.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the sparse `Reward` timestamps in the NWB file.

ii.
```python
reward_times = np.asarray(b['Reward']['timestamps'], dtype=np.float64)
...
outcomes = np.asarray([
    int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))
    for s, e in pairs
], dtype=np.int64)
```

iii. The Reward time series has event timestamps at a different rate than the behavior sampling.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check whether any reward event timestamp falls within the trial's time window (from trial start to teleport). Binary output: 0=omitted, 1=rewarded.

ii.
```python
outcomes = np.asarray([
    int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))
    for s, e in pairs
], dtype=np.int64)
...
out = np.vstack([
    ...
    np.full(len(pos), outcomes[j], np.int64),
])
```

iii. The reward events are assigned by comparing timestamps to trial boundaries.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: Neural arrays longer than behavior are trimmed to behavior length (`pieces.append(arr[:n_beh])`).
- **Empty trials**: Trials where `e <= s` are skipped.
- **Nonfinite data**: An assertion checks for NaN/nonfinite values in neural and input data.
- **Alignment validation**: Checks that `inp[0,0]` is close to 0 (time starts at trial start).

ii.
```python
pieces.append(arr[:n_beh])
...
if e <= s:
    continue
...
if abs(float(inp[0,0])) > 1e-7 or np.any(~np.isfinite(neu)) or np.any(~np.isfinite(inp)):
    raise ValueError(f'Nonfinite data or bad alignment {path.name} trial {j}')
```

iii. The AI identified that ten sessions have neural arrays one row longer than behavior and trims them.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB/HDF5 files and reading large arrays (I/O bound)
2. The single-pass conversion loop processes all sessions sequentially

ii. N/A

iii. The AI designed a single-pass architecture to avoid redundant file loading.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `load_session` iterates over each trial sequentially. The reward outcome computation uses a list comprehension over trials. The binning functions are already vectorized (numpy operations on arrays).

ii. N/A

iii. Most operations are already vectorized at the array level within each trial.

## 13-c. What processing does the code repeat multiple times?

i. The code does not repeat processing. It uses a single pass through all NWB files. Unlike the reference solution which has a separate survey step, this code loads each file once.

ii. N/A

iii. Single-pass design avoids redundant file loading.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores `plot_info` for every trial (position, speed, lick, signed distance, output, zone label) even when `show_processing` is False. This data is discarded in the `info_clean` dictionary but still computed and held in memory during conversion.

ii.
```python
plot_info.append((pos, speed, lick, signed_dist, out, labels[j]))
...
info_clean={k:v for k,v in info.items() if k not in ('plot_info','timestamps','pairs','selected_global','outcomes','zone_labels')}
```

iii. The plot_info is only used when `show_processing` is True, but is computed for all sessions.
