# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every `*.nwb` file under `/app/data` with a recursive glob (`DATA_ROOT.rglob('*.nwb')`), sorted, giving 152 files across 11 `sub-*` directories. Each file is one session and is opened **directly with `h5py`** rather than with `pynwb`. From each file it reads: ten frame-aligned `processing/behavior/BehavioralTimeSeries` streams (`autoreward`, `environment`, `lick`, `position`, `reward_zone`, `scanning`, `speed`, `teleport`, `trial number`, `trial_start`), the `position` timestamps, the sparse `Reward` event timestamps, `processing/ophys/ImageSegmentation/PlaneSegmentation` (`iscell`, `planeIdx`), the per-plane `processing/ophys/Deconvolved` matrices, `identifier`, and `general/subject/subject_id`. `--sample` truncates the file list to the first two files; `--full` (default) processes all 152. All data are converted in a single pass, one session at a time.

ii.
```python
DATA_ROOT = Path('/app/data')
...
files=sorted(DATA_ROOT.rglob('*.nwb'))
if args.sample: files=files[:2]
if not files: raise RuntimeError('No NWB files found')
data=convert(files,args.show_processing)
```
```python
def load_session(path):
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        aligned_names = ['autoreward','environment','lick','position','reward_zone',
                         'scanning','speed','teleport','trial number','trial_start']
        beh = {k: np.asarray(b[k]['data']) for k in aligned_names}
        timestamps = np.asarray(b['position']['timestamps'], dtype=np.float64)
        reward_times = np.asarray(b['Reward']['timestamps'], dtype=np.float64)
        n_beh = len(timestamps)
        if any(len(v) != n_beh for v in beh.values()):
            raise ValueError(f'Frame-aligned behavior length mismatch in {path.name}')
```

iii. From CONVERSION_NOTES Step 2/Step 4: "`/app/data` contains 152 NWB 2.8/HDF5 files … arranged as one directory per subject and one `*_behavior+ophys.nwb` file per session. There are 11 subjects". The AI cross-checked the count of 11 mice against the paper's "switch task (n = 11 mice)" statement and decided to "Retain all 152 sessions and 12,216 complete pulse-defined trials". It used raw `h5py` because the NWB files are plain HDF5 and it wanted to read only the selected ROI columns of the neural matrices ("Read monotonic accepted-cell columns directly from HDF5 and cast to float32"), avoiding loading the full fluorescence matrices.

## 1-b. How are the data split into subjects?

i. The subject list is built from the parent directory names of the NWB files (`sub-mNN` → `mNN`), sorted numerically. Each session's `subject_idx` is obtained by looking up the subject id **read from inside the NWB file** (`general/subject/subject_id`) in that list, so the directory name and the in-file metadata must agree (they do; a mismatch would raise a `KeyError`). Result: 11 subjects, m11 with 12 sessions, all others with 14.

ii.
```python
subjects = sorted({p.parent.name.replace('sub-','') for p in files},
                  key=lambda x: int(re.sub(r'\D','',x)))
subject_lookup = {s:i for i,s in enumerate(subjects)}
...
subject = scalar_text(f['general/subject/subject_id'])
...
info_clean['subject_idx']=subject_lookup[info['subject']]
...
'subjects':subjects,
'subject_idx':np.asarray([z['subject_idx'] for z in session_info],dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2: "There are 11 subjects: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, and m19. m11 has 12 supplied sessions (03-14); every other subject has 14." Step 3 ties this to the paper's statement of 11 switch-task mice, and Step 4 concludes "All 11 switch-task subjects, 152 sessions, complete trials, and accepted cells are retained."

## 1-c. How are the data split into sessions?

i. One session = one NWB file. The file list is the flat sorted glob, so sessions appear grouped by subject and ordered by session number. No merging or alignment of cells across sessions/days is attempted; each session contributes its own independent neuron set. No sessions are excluded.

ii.
```python
files=sorted(DATA_ROOT.rglob('*.nwb'))
...
for i,p in enumerate(files):
    q0=time.perf_counter(); n,x,y,info=load_session(p)
    neural.append(n); inputs.append(x); outputs.append(y)
    region_idx.append(np.zeros(info['n_neurons'], dtype=np.int64))
```

iii. Step 2: "one `*_behavior+ophys.nwb` file per session". Step 4 explicitly rejects the paper's analysis-specific session filter: "Some remapping analyses retain 50/77 significant k=2 sessions … Filter is analysis-specific → Retain all 152 sessions for requested decoder." All sessions were confirmed to have ≥41 trials, satisfying the ≥2-trial requirement.

## 1-d. How are the data split into trials?

i. Trials are defined by pairing each `trial_start` pulse with the **first subsequent** `teleport` pulse. The trial window is `[trial_start_sample, teleport_sample)` — the start sample is included, the teleport sample is excluded. `trial_start` pulses with no following teleport (an incomplete final lap) are dropped. This yields 12,216 trials (mean T = 216.8, min 96, max 3359 samples), identical to the reference. The stored `trial number` stream (which carries a `-1` sentinel before the first lap) is not used to define boundaries, only to read off the trial index.

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
pairs = pair_trials(beh['trial_start'], beh['teleport'], n_beh)
...
for j, (s, e) in enumerate(pairs):
    if e <= s:
        continue
    sl = slice(s, e)  # start included; teleport excluded
```

iii. Step 4: "Trials are laps/traversals; teleport excluded → Slice from each `trial_start` pulse through sample before paired teleport. Exclude sentinel/pre-track and teleport samples." Step 5 Key Decision 2: "Include the `trial_start` sample and stop before the paired teleport sample. This yields track traversal only and excludes teleport/intertrial laser artifacts as in reference dF/F processing." This mirrors `trial_start_inds`/`teleport_inds` in the reference code.

## 1-e. How are trials filtered based on quality controls?

i. Essentially no quality filter is applied. The only rejections are structural: a `trial_start` with no following `teleport` is never added to `pairs`, and a defensive `if e <= s: continue` (never triggers). There is **no minimum-trial-length filter**, no speed/stationarity filter, and no removal of omission trials. Every complete lap in every session is kept (12,216 trials). Per-trial assertions exist but they *raise* rather than filter: a shape mismatch, a non-finite value, or a first timestamp not equal to zero aborts the whole conversion.

ii.
```python
for j, (s, e) in enumerate(pairs):
    if e <= s:
        continue
    ...
    if not (neu.shape[1] == inp.shape[1] == out.shape[1] > 0):
        raise ValueError(f'Trial shape mismatch {path.name} trial {j}')
    if abs(float(inp[0,0])) > 1e-7 or np.any(~np.isfinite(neu)) or np.any(~np.isfinite(inp)):
        raise ValueError(f'Nonfinite data or bad alignment {path.name} trial {j}')
```

iii. Step 3 "Trial curation rules": "Use complete track traversals bounded by entry (`trial_start`) and teleport, excluding pre-track sentinel and teleport/intertrial samples. Preserve rewarded and omission trials; both are scientifically central and explicitly stratified in the reference GLM." Step 10 Check 5 reports a global audit finding "zero empty trials, shape mismatches, nonzero trial starts, nonconstant per-trial fields, invalid first-trial previous outcomes, or out-of-range categories", i.e. the AI's position is that no trial needed removal.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is taken **directly from the stored `processing/ophys/Deconvolved/plane*/data` matrices** (Suite2p's own `spks` output, time × ROI, in raw fluorescence units). The raw `Fluorescence` (F) and `Neuropil` (Fneu) arrays present in the same NWB files are deliberately *not* read. ROIs are restricted to `iscell` using `PlaneSegmentation/iscell`, with `planeIdx` mapping global ROI ids to per-plane columns; accepted cells from both planes of two-plane sessions are concatenated on the cell axis. The matrix is transposed to (n_neurons, n_timepoints) and stored as float32.

ii.
```python
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell0 = np.asarray(seg['iscell'])
iscell = (iscell0[:, 0] if iscell0.ndim > 1 else iscell0) > 0
plane_idx = np.asarray(seg['planeIdx']).astype(int)
pieces = []
for plane_name in sorted(f['processing/ophys/Deconvolved'].keys()):
    plane = int(plane_name.replace('plane',''))
    global_ids = np.flatnonzero(plane_idx == plane)
    ds = f['processing/ophys/Deconvolved'][plane_name]['data']
    local_keep = np.flatnonzero(iscell[global_ids])
    arr = np.asarray(ds[:, local_keep], dtype=np.float32)
    pieces.append(arr[:n_beh])
neural_tn = np.concatenate(pieces, axis=1)
...
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
```

iii. Step 4 discrepancy table: "Neural signal | Workflow computes dF/F and OASIS events; analyses commonly use events | NWB supplies Fluorescence, Neuropil, and `Deconvolved` | Encoding/clustering response is deconvolved activity | **Use supplied `Deconvolved` activity, avoiding an irreproducible second deconvolution; select `iscell`.**" Step 5 Key Decision 3: "Use supplied deconvolved events, not raw F or a second dF/F/deconvolution pass. The NWB is explicitly processed Suite2p output and paper models use deconvolved activity." Step 6: "Avoid recomputing dF/F/OASIS because processed deconvolved activity is supplied."

## 2-b. How is the `neural` data processed?

i. Beyond `iscell` selection, plane concatenation, trimming to the behavior length, transposition and a float32 cast, **no processing is performed**. In particular the AI does *not* perform any of the steps the paper's Methods and `preprocessing.dff()` describe: no 0.7 × neuropil subtraction, no per-trial maximin (20 s sliding window) baseline, no `(F − baseline)/|baseline|` normalization, no 2-sample Gaussian smoothing, and no OASIS deconvolution at `tau = 0.7` with the per-plane frame rate. The `keep_teleports`/`teleport_metadata.py` per-mouse-per-day distinction is not used at all. The values written out are Suite2p `spks` in raw-fluorescence units (session mean ≈ 37 for the spot-checked session, versus ≈ 0.004 for the paper's events).

ii.
```python
arr = np.asarray(ds[:, local_keep], dtype=np.float32)
pieces.append(arr[:n_beh])
...
neural_tn = np.concatenate(pieces, axis=1)
...
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
```
(There is no other neural-processing code anywhere in `convert_data.py`.)

iii. Step 5 Key Decision 3 and Step 10 Check 3: "Neural processing: paper analyses use OASIS deconvolved calcium activity; converter uses supplied `Deconvolved` without duplicate processing." The AI's stated reasons are (a) the NWB export is already a processed Suite2p product, (b) recomputing would be an "irreproducible second deconvolution", and (c) the paper's models take deconvolved activity as their response variable. Notably, Step 3 of its own notes correctly transcribes the paper's per-trial maximin dF/F + OASIS recipe, so the AI was aware of the recipe and chose not to apply it.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single filter: Suite2p's `iscell` column (`iscell[:,0] > 0`), applied per plane via `planeIdx`. This retains 138,678 cells across 152 sessions (155–2,341 per session). The Methods' **second** curation step — excluding putative interneurons whose dF/F correlates with running speed at r > 0.5 — is **not** applied (the reference code function `is_putative_interneuron` is not used); it is not mentioned anywhere in the notes. Functional place-cell / reward-relative / track-relative classifications are also not applied, which the AI argues explicitly. A consistency check asserts that the number of retained columns equals `iscell.sum()`.

ii.
```python
iscell0 = np.asarray(seg['iscell'])
iscell = (iscell0[:, 0] if iscell0.ndim > 1 else iscell0) > 0
...
local_keep = np.flatnonzero(iscell[global_ids])
arr = np.asarray(ds[:, local_keep], dtype=np.float32)
...
if neural_tn.shape[1] != int(iscell.sum()):
    raise ValueError(f'Accepted-cell count mismatch in {path.name}')
```

iii. Step 3 "Neuron curation rules": "Suite2p `iscell` defines accepted cellular ROIs in the supplied NWB files. Place-cell, track-relative, and reward-relative labels are downstream functional classifications based partly on shuffles; they are not general imaging-quality filters and should not restrict a decoder intended to use population activity. FDE > 0.15 was only used for the paper's model-ablation interpretation, not initial neural data inclusion." Step 5 Key Decision 4: "Retain `iscell==1` only; concatenate accepted ROIs across planes." No justification is given for omitting the speed-correlation interneuron filter, because the AI never flags it as a curation step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the trial start, which is achieved purely by slicing: the neural matrix rows and the behavior rows share the same index space in the NWB export, so the same `slice(s, e)` (with `s` = the `trial_start` sample) is applied to both. No resampling, interpolation or offset is used. The only length reconciliation is a terminal trim of the neural matrix to the behavior length (`arr[:n_beh]`), which affects the ten files whose neural array has one extra final row; a shorter-than-behavior neural array raises. `off_start` is recorded as `0.0` (trial begins at the alignment event). The AI verifies alignment per trial by asserting the first element of the time input is exactly 0.

ii.
```python
pieces.append(arr[:n_beh])
...
if neural_tn.shape[0] < n_beh:
    raise ValueError(f'Neural shorter than behavior in {path.name}')
# In ten source files neural has one extra terminal row; slicing above trims it.
...
sl = slice(s, e)
reltime = (timestamps[sl] - timestamps[s]).astype(np.float32)
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
if abs(float(inp[0,0])) > 1e-7 ...: raise ValueError(...)
...
'temporal_alignment_event':'trial_start pulse: entry onto the 450 cm linear track',
'off_start':0.0, 'off_end':None,
```

iii. Step 2/Step 4: "Neural and behavior streams are already synchronized in exported NWB row order"; "Treat each synchronized row as one 64.4836 ms volume bin; concatenate accepted ROIs across planes by row." Step 10 Check 2 reports independent raw-NWB `np.allclose` comparisons of converted neural trials against re-selected/transposed raw rows for six trials (including a two-plane session and one of the one-extra-row sessions), which passed with zero tolerance.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native acquisition resolution is kept: one sample per imaging volume, **64.4836 ms** (`DT = 1/15.5078125` s ≈ 15.5078 Hz). No rebinning, averaging, downsampling, interpolation or smoothing is applied, so trials are variable-length (96–3,359 bins). The bin size is a **hard-coded module constant**; the AI does not read the per-series `rate` attribute or divide it by the plane count. For the 28 two-plane sessions, the stored per-plane series rate is 31.015625 Hz but the rows are volume-synchronized, so 15.5078 Hz is still the correct effective rate; the AI resolves this by row alignment rather than by an arithmetic rate/n_planes check. The value written to `metadata['time_bin_size']` is 64.4836 ms, matching the reference.

ii.
```python
DT = 1.0 / 15.5078125
...
'time_bin_size':DT*1000.0,
```

iii. Step 3: "Neural data time bin | ~15.5 Hz (~64.5 ms) | 'All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate.'" Step 4: "Per-plane metadata reports 31.015625 Hz while rows and behavior are volume-synchronized at ~15.5 Hz. Resolved by native row alignment, consistent with the paper's statement." Step 5 Key Decision 1: "Preserve every native synchronized ~15.5 Hz row (64.4836 ms) rather than position-bin averaging. This matches the paper's temporal GLM streams and requested trial-start temporal alignment." The AI explicitly rejects the paper's 10 cm position binning as "analysis-specific and not appropriate for the requested time-aligned decoder."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From the `timestamps` attribute of the `position` behavioral time series (seconds). The AI checks that all ten frame-aligned behavior streams have the same length as these timestamps and raises otherwise; it does not separately assert that each stream's own timestamps vector is identical.

ii.
```python
timestamps = np.asarray(b['position']['timestamps'], dtype=np.float64)
n_beh = len(timestamps)
if any(len(v) != n_beh for v in beh.values()):
    raise ValueError(f'Frame-aligned behavior length mismatch in {path.name}')
```

iii. Step 5 variable mapping: "behavior timestamps relative to `trial_start` pulse → `input[0]`, `(timestamp - timestamp[start])` in seconds, time-varying continuous." Step 2 notes that all frame-aligned streams share the same sampling and row order, so the choice of which stream's timestamps to use is immaterial.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial's first timestamp is subtracted from the trial's timestamps, so every trial starts at exactly 0 s; the result is cast to float32 and placed in row 0 of the input matrix. Observed range across the dataset is 0–216.5 s.

ii.
```python
reltime = (timestamps[sl] - timestamps[s]).astype(np.float32)
inp = np.vstack([
    reltime,
    np.full(len(pos), env, np.float32),
    np.full(len(pos), source_trial, np.float32),
    np.full(len(pos), prev, np.float32),
]).astype(np.float32, copy=False)
...
if abs(float(inp[0,0])) > 1e-7 ...: raise ValueError('Nonfinite data or bad alignment ...')
```

iii. Step 5 mapping table; the per-trial assertion that `inp[0,0] == 0` is listed in the Step 5 planned sanity checks ("Confirm every trial starts at time zero") and reported as passing globally in Step 10 Check 5.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. No alignment operation is needed: the same `slice(s, e)` indexes the timestamp vector and the neural matrix, and the neural matrix has already been trimmed to the behavior length. A per-trial assertion requires `neu.shape[1] == inp.shape[1] == out.shape[1] > 0`.

ii.
```python
sl = slice(s, e)
reltime = (timestamps[sl] - timestamps[s]).astype(np.float32)
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
if not (neu.shape[1] == inp.shape[1] == out.shape[1] > 0):
    raise ValueError(f'Trial shape mismatch {path.name} trial {j}')
```

iii. Step 4: "Neural and behavior streams are already synchronized in exported NWB row order"; Step 10 Check 2's independent raw-file reconstruction of the time input passed at `atol=1e-6`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The frame-aligned `environment` behavioral time series (values are `-1` outside laps and `0` or `1` within laps, corresponding to ENV1/ENV2).

ii.
```python
aligned_names = ['autoreward','environment','lick','position','reward_zone',
                 'scanning','speed','teleport','trial number','trial_start']
beh = {k: np.asarray(b[k]['data']) for k in aligned_names}
...
env_values = np.asarray(beh['environment'][sl])
```

iii. Step 5 mapping: "behavior `environment` → `input[1]`, Trial mode, map native 0/1 to ENV1/ENV2, repeat across trial." Step 2 reports the per-trial ENV1/ENV2 split as 6,226 / 5,990, used as a consistency check.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. One scalar per trial is computed as the rounded **median** of the `environment` samples inside the trial window, then broadcast to a constant row across all timepoints of that trial (float32). The median makes the value robust to any stray `-1` sentinel samples. Verified range of the converted input is [0, 1].

ii.
```python
env_values = np.asarray(beh['environment'][sl])
env = int(np.rint(np.median(env_values)))
...
inp = np.vstack([
    reltime,
    np.full(len(pos), env, np.float32),
    ...
])
```

iii. Step 5: "Time-varying matrix row but constant per trial." Step 10 Check 5 reports a global audit that found no "nonconstant per-trial fields", i.e. the environment row is constant within every trial as intended.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. From the stored `trial number` behavioral time series, read at the trial's first sample (`beh['trial number'][s]`). The `-1` sentinel that precedes the first lap is avoided because `s` is always a `trial_start` sample. Observed range 0–99.

ii.
```python
source_trial = float(beh['trial number'][s])
```

iii. Step 2: "the trial-number stream has an initial `-1` sentinel and valid trial labels thereafter." Step 5 mapping: "native trial label → `input[2]`, Valid trial number as continuous float, repeat across trial. Preserve source 0-based number, excluding -1 sentinel." The AI preferred the recorded label over a recomputed loop index so that the value is the experiment's own trial identity.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. A float cast and broadcast to a constant row across the trial's timepoints. No renumbering, no offsetting, no normalization, no cross-session continuation. (In this dataset the stored label at every `trial_start` sample equals the 0-based sequential index of the complete lap, so this is numerically the same as a loop counter.)

ii.
```python
source_trial = float(beh['trial number'][s])
inp = np.vstack([
    reltime,
    np.full(len(pos), env, np.float32),
    np.full(len(pos), source_trial, np.float32),
    np.full(len(pos), prev, np.float32),
]).astype(np.float32, copy=False)
```

iii. Step 5 mapping table ("Preserve source 0-based number"). Verified in Step 9/10 as an input range of [0, 99] consistent with the maximum of 100 trials per session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the sparse `Reward` TimeSeries **timestamps** (not a frame-aligned stream). Per-trial reward outcomes are computed once for the whole session into the `outcomes` array, and the previous-trial input simply reads `outcomes[j-1]`.

ii.
```python
reward_times = np.asarray(b['Reward']['timestamps'], dtype=np.float64)
...
outcomes = np.asarray([
    int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))
    for s, e in pairs
], dtype=np.int64)
```

iii. Step 2: "`Reward` differs: it is a sparse event TimeSeries with event timestamps, not a frame-length vector." Step 4: "Assign sparse reward timestamps to trial time intervals; any event means rewarded", which the AI matched to the reference `get_trial_types` logic ("rewarded outcome is any reward event within trial boundaries").

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial index `j > 0` the value is `outcomes[j-1]`; for the first trial of a session it is set to 0. The value is broadcast to a constant float32 row across the trial. "Previous" is defined over the list of complete laps within the same session — there is no carry-over across sessions and no separate "unknown" category for the first trial.

ii.
```python
prev = int(outcomes[j-1]) if j > 0 else 0
inp = np.vstack([
    reltime,
    np.full(len(pos), env, np.float32),
    np.full(len(pos), source_trial, np.float32),
    np.full(len(pos), prev, np.float32),
]).astype(np.float32, copy=False)
```

iii. Step 5 Key Decision 6: "Previous outcome is within-session only; first retained trial is 0 (omitted/no known prior), as required binary without an unknown category." Step 10 Check 5 reports the audit found no "invalid first-trial previous outcomes".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the frame-aligned `position` stream plus a per-trial reward-zone interval. The interval is **not** taken from the `reward_zone` stream; it is derived from the session's `identifier` string (the scene name, e.g. `Env1_LocationB_to_A`, `Env1_A_to_Env2_B`, `Env2_LocationA`) using a regex re-implementation of the reference `reward_relative.behavior.get_reward_zones`, with the module's `X/Y/Z` intervals renamed A/B/C: A = [80, 130], B = [200, 250], C = [320, 370] cm, and a switch at trial 30 for "X_to_Y" scenes.

ii.
```python
ZONE_INTERVALS = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_CODES = {'A': 0, 'B': 1, 'C': 2}

def scene_zone_sequence(scene, n_trials, change_trial=30):
    """Match reward_relative.behavior.get_reward_zones for X/Y/Z=A/B/C."""
    fixed = re.search(r'Location([ABC])$', scene)
    if fixed:
        labels = [fixed.group(1)] * n_trials
    else:
        # Handles Env1_LocationA_to_B and Env1_A_to_Env2_B forms.
        m = re.search(r'(?:Location)?([ABC])_to_(?:Env[123]_)?(?:Location)?([ABC])$', scene)
        if not m:
            raise ValueError(f'Cannot parse reward-zone sequence from scene {scene!r}')
        before, after = m.groups()
        c = min(int(change_trial), n_trials)
        labels = [before] * c + [after] * (n_trials - c)
    coords = np.asarray([ZONE_INTERVALS[x] for x in labels], dtype=np.float32)
    codes = np.asarray([ZONE_CODES[x] for x in labels], dtype=np.int64)
    return labels, coords, codes
...
labels, zone_coords, zone_codes = scene_zone_sequence(scene_from_identifier(identifier), len(pairs))
```

iii. Step 4: "Reward-zone identity | `get_reward_zones` derives zones from reward-location values/positions and labels | Exported `reward_zone` frame field is constant 0; identifier encodes A/B/C and switches | … | **Infer per-trial zone from session identifier plus reward-position transitions**; never treat constant field as all-A." Step 5 Key Decision 5: "Follow reference `get_reward_zones` scene logic and switch point. Cross-check labels against sparse reward-event positions nearest A/B/C clusters. For omission trials, inherit the scene-defined pre/post-switch zone rather than infer from absent events." Step 10 reports the cross-check: "Reward positions validated the assigned class on 10,337/10,342 rewarded trials (99.9517%)".

Note on the premise: the claim that the exported `reward_zone` stream is "constant 0" is factually wrong — it takes values 0–6 and is non-zero on ~1% of samples. The AI reached this conclusion because its survey script took the *per-trial mode* of the stream, which is 0 for every trial. The resulting labels are nevertheless correct (see 10-b).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each sample, a signed distance to the trial's reward-zone interval: `pos − start` when before the zone, `pos − stop` when past the zone, and exactly `0.0` while inside the (inclusive) interval. This is computed vectorized over the whole trial with `np.where`, then discretized (see 7-c).

ii.
```python
def distance_to_interval(pos, start, stop):
    return np.where(pos < start, pos - start, np.where(pos > stop, pos - stop, 0.0))
...
zstart, zstop = zone_coords[j]
signed_dist = distance_to_interval(pos, zstart, zstop)
```

iii. Step 5 mapping: "position + per-trial reward-zone interval → `output[0]` distance to reward zone; Signed distance: `pos-start` before zone, 0 inside inclusive interval, `pos-stop` after; discretize with task thresholds." This matches the instruction's "Distance to **any location in** the reward zone", i.e. distance 0 anywhere inside the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned by explicit boolean masks: 0 for `< −50`; 1 for `[−50, −10)`; 2 for `[−10, 0)`; 3 for exactly `0` (inside the zone); 4 for `(0, 10]`; 5 for `(10, 50]`; 6 for `> 50` cm. Resulting distribution: [0.253, 0.102, 0.074, 0.237, 0.021, 0.072, 0.242].

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
...
out = np.vstack([bin_distance(signed_dist), ...])
```

iii. Step 5 "Exact Output Categories": "Distance: 0 `< -50`; 1 `[-50,-10)`; 2 `[-10,0)`; 3 exactly `0`; 4 `(0,10]`; 5 `(10,50]`; 6 `>50` cm." Step 10 Check 5: "Independent threshold tests passed at every distance/position/speed boundary." The AI notes the class-3 fraction (0.237) is large because "0" covers the entire 50 cm-wide zone, as the instructions specify.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No explicit alignment step — `position` is sliced with the same `slice(s, e)` used for the neural matrix, so the distance series is sample-for-sample aligned by construction. Equal lengths are asserted per trial.

ii.
```python
sl = slice(s, e)
pos = np.asarray(beh['position'][sl], dtype=np.float32)
...
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
if not (neu.shape[1] == inp.shape[1] == out.shape[1] > 0):
    raise ValueError(f'Trial shape mismatch {path.name} trial {j}')
```

iii. Step 4: streams are "already synchronized in exported NWB row order." The `--show-processing` plots (Step 7) were used to confirm visually that "signed distance is exactly zero over zone intervals and category transitions follow the requested thresholds. No temporal discontinuity beyond expected trial concatenation was detected."

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Directly from the frame-aligned `position` behavioral time series (cm along the 450 cm virtual corridor), cast to float32 and sliced per trial.

ii.
```python
beh = {k: np.asarray(b[k]['data']) for k in aligned_names}   # includes 'position'
...
pos = np.asarray(beh['position'][sl], dtype=np.float32)
```

iii. Step 5 mapping: "behavior `position` → `output[1]` absolute position … native synchronized position." Step 3 records the paper's "position is from 0 to 450 cm" as the reference for the binning span.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None other than the per-trial slice and discretization — no clipping, no unwrapping, no smoothing, no offsetting. The raw −500 / −50 cm teleport-period values never enter because they lie outside the `[trial_start, teleport)` window.

ii.
```python
pos = np.asarray(beh['position'][sl], dtype=np.float32)
...
out = np.vstack([
    bin_distance(signed_dist), bin_position(pos), bin_speed(speed), lick, ...
])
```

iii. Step 5 mapping note: "Clip no values; apply exact inequalities." Step 5 Key Decision 2 explains that restricting trials to `[trial_start, teleport)` is what keeps the intertrial teleport positions out of the converted data.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm categories by explicit boolean masks: 0 `< 90`; 1 `[90, 180)`; 2 `[180, 270)`; 3 `[270, 360]`; 4 `> 360` cm. The first and last bins are open-ended so that the handful of samples slightly below 0 or above 450 cm fall into the end classes rather than becoming invalid. Resulting distribution: [0.211, 0.178, 0.231, 0.227, 0.154].

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

iii. Step 5 "Exact Output Categories": "Position: 0 `<90`; 1 `[90,180)`; 2 `[180,270)`; 3 `[270,360]`; 4 `>360` cm", taken verbatim from the Decoder Task specification (5 equal 90 cm bins spanning the 450 cm track). Step 10 Check 5 reports that independent boundary tests passed.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same-index slicing, as for all other frame-aligned streams; nothing further is done. Length equality with the neural trial is asserted.

ii.
```python
sl = slice(s, e)
pos = np.asarray(beh['position'][sl], dtype=np.float32)
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
if not (neu.shape[1] == inp.shape[1] == out.shape[1] > 0):
    raise ValueError(f'Trial shape mismatch {path.name} trial {j}')
```

iii. Step 4/Step 10: rows are natively synchronized; Step 10 Check 2 independently reconstructed position categories from the raw NWB for six trials and matched with zero tolerance.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. From the frame-aligned `lick` behavioral time series, which holds per-frame lick counts (integer values 0–6 in the inspected sessions).

ii.
```python
lick = (np.asarray(beh['lick'][sl]) > 0).astype(np.int64)
```

iii. Step 5 mapping: "behavior `lick` → `output[3]` lick; `lick > 0` -> 1, else 0; native lick series; Binary per frame."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarization by thresholding at `> 0`, cast to int64. No smoothing, no dilation, no exclusion of consummatory licks. Resulting distribution: 0.770 no-lick / 0.230 lick.

ii.
```python
lick = (np.asarray(beh['lick'][sl]) > 0).astype(np.int64)
...
out = np.vstack([
    bin_distance(signed_dist), bin_position(pos), bin_speed(speed), lick,
    np.full(len(pos), zone_codes[j], np.int64),
    np.full(len(pos), outcomes[j], np.int64),
]).astype(np.int64, copy=False)
```

iii. Step 5 "Exact Output Categories": "Lick: 0 no lick; 1 lick." Step 3 notes the paper's GLM used "smoothed binary lick counts", but the AI deliberately keeps the unsmoothed binary form because the Decoder Task asks for a binary per-timepoint lick output ("Keep native … no paper GLM smoothing because requested categorical instantaneous speed/lick").

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same-index slicing with the neural data; no offset or resampling. Length equality asserted per trial.

ii.
```python
sl = slice(s, e)
lick = (np.asarray(beh['lick'][sl]) > 0).astype(np.int64)
neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
```

iii. Step 4 (native row synchronization) and Step 12, where the AI re-loaded raw NWBs for three trials and confirmed "np.allclose passed for complete lick vectors and outcome labels" after the lick decoding accuracy came out modest (0.651 validation).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the session `identifier` (scene name) alone, via `scene_zone_sequence` — the same function described in 7-a. The exported `reward_zone` frame stream is read into `beh` but never used. The per-trial label is encoded A=0, B=1, C=2 and broadcast across the trial's timepoints.

ii.
```python
identifier = scalar_text(f['identifier'])
...
def scene_from_identifier(identifier):
    return identifier.rstrip('/').split('/')[-1]
...
labels, zone_coords, zone_codes = scene_zone_sequence(scene_from_identifier(identifier), len(pairs))
...
out = np.vstack([..., np.full(len(pos), zone_codes[j], np.int64), ...])
```

iii. See 7-a. Step 5 Key Decision 5 and Step 10: the AI validated the scene-derived labels against the positions at which reward events occurred, reporting 99.95% agreement, and treated the five disagreements as "delayed delivery/frame sampling, not label errors".

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene string → regex → an ordered list of per-trial labels. Fixed-location scenes (`...LocationA`/`B`/`C`) give one label for all trials; switch scenes (`...LocationB_to_A`, `...A_to_Env2_B`) give the first label for trials 0–29 and the second for trials 30 onward, with `change_trial` clamped to the trial count. Labels map to codes 0/1/2 and to the intervals used for the distance output. `change_trial = 30` is hard-coded (the reference function's default; the reference allows a per-session `sess.change_reward_trial` override, which is not available in the NWB export). Resulting distribution: A 0.329 / B 0.337 / C 0.335, identical to the reference.

ii.
```python
c = min(int(change_trial), n_trials)
labels = [before] * c + [after] * (n_trials - c)
coords = np.asarray([ZONE_INTERVALS[x] for x in labels], dtype=np.float32)
codes = np.asarray([ZONE_CODES[x] for x in labels], dtype=np.int64)
```

iii. Step 10 Check 3: "Reward zones exactly reproduce `get_reward_zones`: A=[80,130], B=[200,250], C=[320,370], switch at trial 30." (Independently verified here: for all 10,394 trials in which the `reward_zone` stream is active, the zone nearest the first in-zone position equals the AI's scene-derived label — 0 mismatches across all 152 sessions.)

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From the sparse `Reward` TimeSeries `timestamps` (74 events in the inspected session), compared against the behavior timestamps at the trial boundaries. The `Reward` `data` values (a constant 0.004 mL) are not used.

ii.
```python
reward_times = np.asarray(b['Reward']['timestamps'], dtype=np.float64)
...
outcomes = np.asarray([
    int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))
    for s, e in pairs
], dtype=np.int64)
```

iii. Step 2: "`Reward` … is a sparse event TimeSeries with event timestamps, not a frame-length vector." Step 4: "Assign sparse reward timestamps to trial time intervals; any event means rewarded", matching the reference `get_trial_types`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is rewarded (1) if at least one reward timestamp falls in the closed interval `[timestamps[trial_start], timestamps[teleport]]`, otherwise omitted (0). Note that the right endpoint is the teleport sample's timestamp, one sample beyond the trial slice used for all the time-varying data. The per-trial scalar is broadcast across the trial's timepoints as output row 5, and reused for the previous-trial-outcome input. Resulting distribution: 0.157 omitted / 0.843 rewarded (time-weighted), i.e. 1,874 omitted / 10,342 rewarded trials.

ii.
```python
outcomes = np.asarray([
    int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))
    for s, e in pairs
], dtype=np.int64)
...
out = np.vstack([..., np.full(len(pos), outcomes[j], np.int64)]).astype(np.int64, copy=False)
```

iii. Step 5 mapping: "sparse `Reward` timestamps → `output[5]` reward outcome; 1 iff any reward event timestamp falls from trial start through paired teleport, else 0." Step 9 lists the outcome distribution as a consistency check against the raw event count, and Step 12 re-verified outcome labels against raw NWB for three trials after outcome decoding came out near chance.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI's policy is "fail loudly" rather than "repair", with one explicit exception:
- **Neural array one row longer than behavior (10 files)**: the neural matrix is trimmed at the end (`arr[:n_beh]`) — the only tolerated mismatch. A neural array *shorter* than behavior raises.
- **Behavior stream length disagreement**: raises `ValueError`.
- **ROI/column mapping mismatch** between `planeIdx` and the `Deconvolved` column count, or accepted-cell count mismatch: raises.
- **Unparseable scene name**: raises.
- **Non-finite neural/input values, trial shape mismatch, or a trial whose first timestamp is not 0**: raises.
- **`-1` environment sentinel**: absorbed by taking the per-trial median.
- **`-1` trial-number sentinel**: avoided by reading the value at the `trial_start` sample.
- **Teleport-period positions (−500, −50 cm)**: excluded by the trial window rather than clipped.
- **Missing/ambiguous reward-zone identity on omission trials**: handled by inheriting the scene-defined zone (no imputation from data needed).
- **Incomplete final lap (a `trial_start` with no following `teleport`)**: silently dropped.
No NaN filling, interpolation or time-shifting is used anywhere.

ii.
```python
if any(len(v) != n_beh for v in beh.values()):
    raise ValueError(f'Frame-aligned behavior length mismatch in {path.name}')
...
if ds.shape[1] != len(global_ids):
    raise ValueError(f'ROI mapping mismatch {path.name} {plane_name}')
...
pieces.append(arr[:n_beh])
if neural_tn.shape[0] < n_beh:
    raise ValueError(f'Neural shorter than behavior in {path.name}')
# In ten source files neural has one extra terminal row; slicing above trims it.
if neural_tn.shape[1] != int(iscell.sum()):
    raise ValueError(f'Accepted-cell count mismatch in {path.name}')
...
env = int(np.rint(np.median(env_values)))
...
if abs(float(inp[0,0])) > 1e-7 or np.any(~np.isfinite(neu)) or np.any(~np.isfinite(inp)):
    raise ValueError(f'Nonfinite data or bad alignment {path.name} trial {j}')
```

iii. Step 4: "Terminal lengths | Assumes synchronized streams | Ten sessions have neural arrays one sample longer | No conflicting statement | **Trim only terminal excess neural row to common behavior length; trial windows are unaffected.**" Step 5 Key Decision 7: "Truncate the ten one-row-long neural streams at the terminal end to behavior length before trial slicing; never shift/interpolate." Step 10 Check 2 reports an exact raw-data match for one of the ten affected sessions.

## 13-a. What are the most time-consuming steps of the code?

i. The AI identifies (and the timing printout confirms) that the cost is dominated by HDF5 I/O:
1. Reading the selected columns of the per-plane `Deconvolved` matrices with fancy indexing (`ds[:, local_keep]`) — the dominant per-session cost, scaling with samples × accepted cells.
2. Reading the ten frame-aligned behavior arrays per file.
3. Serializing the 9.84 GB pickle.
Total: 67.90 s of conversion + 7.15 s of pickling for 152 sessions. Per-trial numpy work (binning, slicing, `np.vstack`) is negligible. By deciding not to recompute dF/F and OASIS, the AI removed what would otherwise be the single largest cost.

ii.
```python
q0=time.perf_counter(); n,x,y,info=load_session(p)
...
print(f"[{i+1}/{len(files)}] {p.name}: {info['n_trials']} trials, {info['n_neurons']} cells, {time.perf_counter()-q0:.2f}s",flush=True)
...
print(f'Converted {len(files)} sessions, {total_trials} trials, {total_cells} session-cells in {time.perf_counter()-t0:.2f}s',flush=True)
...
with out.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
print(f'Saved {out} ({out.stat().st_size/1e6:.1f} MB) in {time.perf_counter()-t:.2f}s',flush=True)
```

iii. Step 6 "Code inefficiencies identified": "Loading full raw fluorescence and neuropil would waste memory; only selected columns of Deconvolved are read. Retaining session-scale time×cell matrices after trial slicing would duplicate memory unnecessarily." Step 9: "Conversion runtime: 67.90 s computation + 7.15 s serialization, well below the 15-minute threshold."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain:
1. `pair_trials` loops in Python over every `trial_start` pulse calling `np.searchsorted` one index at a time; the whole pairing is a single vectorized `np.searchsorted(teleports, starts)` call.
2. The `outcomes` list comprehension performs a full boolean scan of `reward_times` per trial (O(n_trials × n_reward_events)); `np.searchsorted` on the sorted reward times would make it O(n log n) in one shot.
3. The main `for j, (s, e) in enumerate(pairs)` loop re-invokes `bin_distance`, `bin_position`, `bin_speed` and the `np.vstack`/`np.full` construction per trial; the binning of `position` and `speed` could be done once per session on the full arrays and then sliced (only `bin_distance` genuinely depends on the per-trial zone, and even that could be done session-wide from a per-sample zone-interval array).
Each masked assignment inside the `bin_*` helpers also builds several temporary boolean arrays where a single `np.digitize`/`np.searchsorted` call would suffice.
The per-plane loop is not a real target (≤2 iterations).

ii.
```python
for s in starts:
    k = np.searchsorted(teleports, s, side='left')
    if k < len(teleports) and teleports[k] > s:
        pairs.append((int(s), int(teleports[k])))
```
```python
outcomes = np.asarray([
    int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))
    for s, e in pairs
], dtype=np.int64)
```
```python
for j, (s, e) in enumerate(pairs):
    ...
    out = np.vstack([
        bin_distance(signed_dist), bin_position(pos), bin_speed(speed), lick, ...
    ])
```

iii. The AI's notes do not identify any of these; Step 6 claims only "Process one session at a time and use vectorized discretization and trial slicing." Its justification for leaving them is implicit: the whole conversion runs in 68 s, far under the 15-minute budget set by the instructions, so no further optimization was pursued.

## 13-c. What processing does the code repeat multiple times?

i. Very little — unlike the reference, the AI makes a **single pass** over each NWB file (no separate survey pass), which is its main efficiency advantage. What is repeated:
- `np.asarray(beh[...][sl])` re-materializes a fresh array per trial per stream, and `np.full(len(pos), ...)` rebuilds constant rows per trial for the four per-trial scalars.
- The `bin_position`/`bin_speed` mapping is recomputed per trial on data that could have been binned once per session.
- `timestamps[s]`/`timestamps[e]` lookups are repeated in both the `outcomes` comprehension and the main loop.
- `region_idx` (all-zeros CA1 index) is rebuilt per session, and `session_info` is assembled twice (once as `info`, once as the filtered `info_clean`).

ii.
```python
info_clean={k:v for k,v in info.items() if k not in ('plot_info','timestamps','pairs','selected_global','outcomes','zone_labels')}
...
region_idx.append(np.zeros(info['n_neurons'], dtype=np.int64))
```
```python
pos = np.asarray(beh['position'][sl], dtype=np.float32)
speed = np.asarray(beh['speed'][sl], dtype=np.float32)
lick = (np.asarray(beh['lick'][sl]) > 0).astype(np.int64)
env_values = np.asarray(beh['environment'][sl])
```

iii. Not discussed in the notes. Step 6's stated design goal — "Process one session at a time … Avoid recomputing dF/F/OASIS" — shows the AI targeted the big repeats (a second file pass, a second deconvolution) and considered the remainder immaterial at 68 s total.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
1. **`plot_info` is always built.** Every trial appends a 6-tuple of full-length arrays (`pos`, `speed`, `lick`, `signed_dist`, `out`, `label`) to `plot_info`, even when `--show-processing` is not given and even for sessions beyond the first two that are ever plotted. This roughly doubles the peak per-session behavioral memory and is then thrown away.
2. **Unused behavior streams are read in full.** `autoreward`, `scanning`, `teleport` (after pairing) and especially `reward_zone` are loaded into `beh` for every session; `autoreward`, `scanning` and `reward_zone` are never used at all.
3. **`info` fields computed then stripped.** `timestamps`, `pairs`, `outcomes`, `zone_labels`, `selected_global` (built by an `extend` over a Python list of ROI ids) and `plot_info` are all assembled and then removed by `info_clean` before saving; `selected_global` in particular is pure bookkeeping that is never consulted.
4. **Per-trial-constant rows broadcast to full time series.** `environment`, `trial number`, `previous outcome`, `reward zone location` and `reward outcome` are stored as full-length constant rows, which is what inflates the pickle to 9.84 GB. (This one is arguably required by the target format's uniform `(d, n_timepoints)` convention, so it is a defensible cost rather than a mistake.)

ii.
```python
plot_info.append((pos, speed, lick, signed_dist, out, labels[j]))
...
info = dict(path=str(path), session=path.stem, identifier=identifier, ...,
            zone_labels=labels, pairs=pairs, timestamps=timestamps,
            selected_global=np.asarray(selected_global), plot_info=plot_info)
```
```python
aligned_names = ['autoreward','environment','lick','position','reward_zone',
                 'scanning','speed','teleport','trial number','trial_start']
beh = {k: np.asarray(b[k]['data']) for k in aligned_names}
```
```python
info_clean={k:v for k,v in info.items() if k not in ('plot_info','timestamps','pairs','selected_global','outcomes','zone_labels')}
```

iii. Not discussed. The closest statement is Step 6's "Retaining session-scale time×cell matrices after trial slicing would duplicate memory unnecessarily" — the AI applied that principle to the neural data but not to the behavioral `plot_info` copies. Reading all ten aligned streams is used as a uniform length-consistency check (`if any(len(v) != n_beh ...)`), which is a partial justification for loading the unused ones.
