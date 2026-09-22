# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All 152 NWB files are discovered with a single recursive glob over `/app/data` (`sub-*/*.nwb`), sorted, and each file is treated as one session. Files are read directly with `h5py` (not `pynwb`), reaching into the HDF5 paths `processing/behavior/BehavioralTimeSeries/*`, `processing/ophys/Deconvolved/plane*/data`, `processing/ophys/ImageSegmentation/PlaneSegmentation`, and `general/subject/subject_id`. Each session is opened twice: once in `inspect_session()` to read the small behavior arrays and ROI tables, and once in `convert_session()` to read the behavior arrays again plus the neural slices. `--sample` takes the first 2 files; `--full` (default) takes all 152.

ii.
```python
DATA_ROOT = Path('/app/data')
...
    files=sorted(DATA_ROOT.glob('sub-*/*.nwb'))
    if args.sample: files=files[:2]
    print(f'Converting {len(files)} of {len(list(DATA_ROOT.glob("sub-*/*.nwb")))} sessions',flush=True)
```
```python
def inspect_session(path):
    """Read small behavior arrays and derive trial metadata."""
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        trial = b['trial number/data'][:]
        ts = b['trial number/timestamps'][:]
        position = b['position/data'][:]
        zone_event = b['reward_zone/data'][:]
        environment = b['environment/data'][:]
        reward_ts = b['Reward/timestamps'][:]
```

iii. From CONVERSION_NOTES.md Step 2: "`/app/data` is approximately 87 GB and contains 152 NWB 2.5/HDF5 files organized as `sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`; no separate README is present." The agent verified 11 subjects × 14 sessions (m11 has 12) = 152 and that "All 152 sessions expose identical behavior variable names and all have at least two trials." `h5py` was chosen over `pynwb` for direct, lazy slicing of the large ophys arrays ("read only selected neural columns and trial rows from HDF5").

## 1-b. How are the data split into subjects?

i. The subject label for each session is read from the NWB metadata field `general/subject/subject_id` inside each file (not parsed from the directory name). The unique labels are sorted numerically by the integer after `m`, and `subject_idx` indexes that list per session. 11 subjects result: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19.

ii.
```python
        subject = f['general/subject/subject_id'][()].decode()
```
```python
    subjects=sorted(set(session_subjects), key=lambda x:(int(x[1:]) if x[1:].isdigit() else x))
    subject_idx=np.array([subjects.index(x) for x in session_subjects],dtype=np.int64)
```

iii. CONVERSION_NOTES Step 2: "Subjects | 11: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19" and "Sessions / subject | 14 each, except m11 has 12; 152 total". The agent used the in-file subject id so the label comes from the authoritative NWB metadata rather than from the path.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Sessions are ordered by the sorted file paths; no merging or cross-session ROI matching is done (the paper's across-day ROI registration is ignored).

ii.
```python
    files=sorted(DATA_ROOT.glob('sub-*/*.nwb'))
    ...
    for si,path in enumerate(files):
        print(f'[{si+1}/{len(files)}] {path}',flush=True)
        n,i,o,subject,summary=convert_session(path, args.show_processing and si<2)
```

iii. CONVERSION_NOTES Step 2/Step 10: "Reference `create_sess` loads scan/VR/Suite2p/behavior into synchronized sessions. Conversion reads the equivalent synchronized NWB processing modules." Each file is one synchronized recording day; the session number in the filename is the experiment day.

## 1-d. How are the data split into trials?

i. Trials are segmented by contiguous runs of the framewise `trial number` behavior stream (values 0..N-1; -1 treated as invalid/intertrial). For each trial id the block is `[first index, last index + 1)`. The code asserts trial ids are contiguous from 0 and that each id's samples are contiguous. `trial_start` is used **only** as a keep/drop flag (1-e), not as the trial boundary; `teleport` is not used at all.

Measured consequence (verified independently here against the raw NWB): a `trial number` block begins during the inter-trial/teleport period, a median of ~54–74 samples (3.5–4.8 s, range 11–182 samples) **before** the `trial_start` marker, with position pinned at the −50 cm sentinel and wheel speed often 50–70 cm/s. So each converted "trial" is ITI + lap, and 19,682 of the 19,818 frames (99.3%) of session `sub-m11_ses-03` end up inside trials. The mean trial length is 298 frames vs 217 in the reference, and maximum trial length is 10,968 frames (707 s).

ii.
```python
        trial_ids = np.unique(trial[trial >= 0]).astype(int)
        if not np.array_equal(trial_ids, np.arange(len(trial_ids))):
            raise ValueError(f'{path.name}: non-contiguous trial numbers')
        ...
        for tid in trial_ids:
            idx = np.flatnonzero(trial == tid)
            if not len(idx) or np.any(np.diff(idx) != 1):
                raise ValueError(f'{path.name}: trial {tid} is empty/noncontiguous')
            lo, hi = int(idx[0]), int(idx[-1])
            bounds.append((lo, hi + 1))
```
```python
        for trial_idx, (lo, hi) in enumerate(info['bounds']):
            tid = int(info['trial_ids'][trial_idx])
            hi = min(hi, n_common)
```

iii. CONVERSION_NOTES Step 4: "`trial number` is contiguous 0..N-1; -1 is intertrial; one boundary marker is missing globally … Segment by nonnegative trial number, which is more robust than event markers." Step 2 states "Trial numbers are contiguous 0..N-1; -1 denotes intertrial/invalid periods", and the saved metadata says "intertrial/teleport samples are excluded." The agent's own Step 1/2 notes had earlier observed that "Trial start and teleport explicitly mark the beginning and end of each lap, matching the requested alignment," but it preferred `trial number` because `trial_start`/`teleport` markers numbered 12,216 vs 12,217 numbered ids. The premise that `trial number == -1` marks the ITI is false in this dataset: −1 occurs only at the head of the recording (133 samples in m11 ses-03), and the ITI is assigned to the *upcoming* trial number.

## 1-e. How are trials filtered based on quality controls?

i. A single filter: numbered trials that contain no `trial_start` pulse are dropped as truncated recording-edge fragments. Exactly one trial in the whole dataset is removed (m11 ses-03 native trial 80, 3 frames long), leaving 12,216 trials. Two hard errors also act as curation guards (non-contiguous trial ids, or a trial with no samples in common between behavior and neural). There is no minimum-length filter, no speed filter, no rewarded/omission stratification, and no session-level exclusion.

ii.
```python
            has_start[tid] = bool(np.any(b['trial_start/data'][idx] > 0))
        ...
        # Exclude incomplete recording-edge fragments lacking a trial-start marker.
        # Preserve native trial number and previous native-trial outcome.
        previous = np.r_[0, outcomes[:-1]].astype(np.int8)
        keep = has_start
        if np.sum(~keep):
            print(f'  {path.name}: excluding {int(np.sum(~keep))} numbered fragment(s) without trial_start', flush=True)
        trial_ids = trial_ids[keep]
        bounds = [x for x, k in zip(bounds, keep) if k]
        zones = zones[keep]; outcomes = outcomes[keep]; envs = envs[keep]; previous = previous[keep]
```

iii. CONVERSION_NOTES Step 10: "Raw m11 session 03 native trial 80 was uniquely only 3 frames, had no `trial_start` or `teleport`, moved backward over a partial corridor, and was the only numbered ID lacking a start. It was excluded. All other 12,216 trials have a start marker; missing teleport markers are common and therefore not a filter." Step 3: paper-specific selections ("30 trials pre-switch, rewarded/omission stratification, k-means significance, reward-zone subsets, or speed >2 cm/s) apply only to individual analyses and should not remove trials here. The requested speed output requires retaining low-speed frames."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Exclusively from the NWB `processing/ophys/Deconvolved/plane*/data` arrays (Suite2p `spks`), restricted to Suite2p-accepted ROIs. `Fluorescence` (F) and `Neuropil` (Fneu) are read by no part of the conversion; no dF/F is computed.

ii.
```python
        neural_series = [f['processing/ophys/Deconvolved/'+name+'/data'] for name in info['plane_names']]
        cell_indices = [np.flatnonzero(mask) for mask in info['plane_masks']]
        ...
            plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
                          for ds, idx in zip(neural_series, cell_indices)]
            neural = np.concatenate(plane_data, axis=0)
```

iii. CONVERSION_NOTES Step 4: "Neural signal | Paper decoder uses deconvolved events; dF/F used for spatial-field analyses | NWB provides Deconvolved, raw Fluorescence, Neuropil | Decoder methods use deconvolved calcium events | **Use provided Deconvolved activity, avoiding redundant dF/F recomputation.**" Step 1 acknowledged that "`multi_anim_sess` computes maximin-baseline dF/F" and that "if provided files already contain processed neural streams, those streams should not be recomputed blindly."

## 2-b. How is the `neural` data processed?

i. Essentially no processing: the stored Deconvolved values are sliced per trial, the accepted-cell columns selected, transposed to (n_neurons, n_timepoints), cast to float32, and any non-finite entries replaced by 0. Planes are concatenated along the neuron axis. There is no neuropil subtraction (`neu_coef = 0.7`), no per-trial maximin baseline over a 20 s window, no `(F − baseline)/|baseline|`, no 2-sample Gaussian smoothing, no OASIS deconvolution at `tau = 0.7`, and no per-cell normalization. The values retain raw-fluorescence units (order 10–1000).

ii.
```python
            plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
                          for ds, idx in zip(neural_series, cell_indices)]
            neural = np.concatenate(plane_data, axis=0)
            if not np.all(np.isfinite(neural)):
                neural = np.nan_to_num(neural, copy=False)
```

iii. CONVERSION_NOTES Step 5 mapping row: "`processing/ophys/Deconvolved/plane0/data[:, iscell]` → `neural` | Select `iscell[:,0]==1`, transpose each trial to neurons x time, float32 | `glmUtils.get_timeseries_data` | Paper decoder uses deconvolved events." Step 4 justified skipping dF/F as "avoiding redundant dF/F recomputation," i.e. the agent assumed the NWB `Deconvolved` stream is the paper's event series. Step 10 Check 5: "Reference `create_sess` loads scan/VR/Suite2p/behavior into synchronized sessions. Conversion reads the equivalent synchronized NWB processing modules. Difference: NWB is the supplied finalized format, so no raw Scanbox/VR reconstruction is needed."

## 2-c. How is the `neural` data filtered based on quality controls?

i. One filter: Suite2p manual curation, `iscell[:,0] > 0`, applied per imaging plane using the `planeIdx` column of the combined `PlaneSegmentation` table (needed for the 28 two-plane m17/m18 sessions). 138,678 of 260,091 ROIs are kept (155–2,341 per session). The paper's additional exclusion of putative interneurons (Pearson r > 0.5 between a cell's dF/F and running speed) is not applied, and no place-cell/RR/TR subselection is applied.

ii.
```python
        ps = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = ps['iscell'][:]
        accepted = iscell[:, 0] > 0 if iscell.ndim == 2 else iscell > 0
        plane_idx = ps['planeIdx'][:].astype(int) if 'planeIdx' in ps else np.zeros(len(accepted), dtype=int)
        plane_names = sorted(f['processing/ophys/Deconvolved'].keys(), key=lambda x: int(x.replace('plane','')))
        for plane_name in plane_names:
            plane_num = int(plane_name.replace('plane',''))
            shape = f['processing/ophys/Deconvolved/'+plane_name+'/data'].shape
            mask = accepted[plane_idx == plane_num]
            if shape[1] != len(mask):
                raise ValueError(f'{path.name}: {plane_name} ROI mask/data mismatch {len(mask)} != {shape[1]}')
```

iii. CONVERSION_NOTES Step 3: "Use Suite2p-accepted cells (`iscell[:,0] == 1`) as the recording-quality criterion. Place-cell, reward-relative-cell, and track-relative-cell masks are scientific subpopulation definitions and are not general quality filters." Step 10: "Conversion uses Suite2p `iscell[:,0]` only, retaining all quality-accepted cells because scientific subpopulation selection would bias this general decoder." The interneuron-correlation filter in the Methods is never mentioned anywhere in the notes or trajectory.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No realignment is performed: the neural rows are sliced with the same `[lo, hi)` indices as the behavior, and `t = 0` is defined as the first sample of the `trial number` block. The saved metadata declares `temporal_alignment_event = 'start of numbered corridor trial (entry to linear track)'` with `off_start = 0.0`. Because the `trial number` block starts during the preceding ITI (see 1-d), the actual alignment event is ITI onset, which leads track entry by a *variable* 11–182 samples (0.7–11.7 s).

ii.
```python
            t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
```
```python
       time_bin_size=float(BIN_MS), temporal_alignment_event='start of numbered corridor trial (entry to linear track)',
       off_start=0.0, off_end=None, neural_signal='Suite2p deconvolved calcium events from accepted cells',
```

iii. CONVERSION_NOTES Step 10 Check 7: "Reference `get_timeseries_data` aligns deconvolved events and behavior at imaging frames. Conversion preserves one-to-one NWB rows and aligns trials at their first numbered sample/start." Step 5: "Trials: include every complete numbered trial (12,216 total); exclude intertrial samples … Variable trial lengths are preserved."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 64.4836 ms (1000/15.5078125 Hz), one converted sample per stored imaging row. No rebinning, resampling, smoothing, or interpolation. The 28 sessions whose RoiResponseSeries advertise 31.015625 Hz are two-plane recordings; the agent kept them at one row per stored sample rather than downsampling, reasoning that the advertised rate is the scanner rate over two planes while each plane's rows are already 1:1 with the 15.5 Hz behavior timestamps.

ii.
```python
BIN_MS = 1000.0 / 15.5078125
...
       time_bin_size=float(BIN_MS), ...
```

iii. CONVERSION_NOTES Step 4: "Preserve rowwise alignment and use common 64.4836 ms bins; do **not** factor-2 downsample nominal-31-Hz files because that would misalign and discard half the behavior-aligned neural samples." Step 10: "Two-plane series report aggregate scanner rate 31.015625 Hz, but each plane row is aligned one-to-one with behavior timestamps at 15.5078125 Hz." Methods quote used: "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attached to the `trial number` behavior time series (`processing/behavior/BehavioralTimeSeries/trial number/timestamps`), read once per session in `inspect_session`.

ii.
```python
        ts = b['trial number/timestamps'][:]
...
        timestamps = info['timestamps']
        ...
            t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
```

iii. CONVERSION_NOTES Step 2: "Behavior has explicit synchronized timestamps and no NaNs in core continuous/categorical fields." Step 5 mapping: "behavior timestamps + trial sample indices → `input[0]` | Seconds since first frame of numbered trial | synchronized `sess` timeseries | Continuous time-varying."

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtraction of the trial window's first timestamp, cast to float32. No re-zeroing on `trial_start`, so the value at the moment the animal actually enters the corridor is not 0 but the (variable) ITI duration; the resulting range is [0, 707.2] s across the dataset versus [0, 216.5] s for the reference.

ii.
```python
            t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
```

iii. Step 5 key decision 2: "include every complete numbered trial … Variable trial lengths are preserved." Step 10 Check 12: "all trial-relative times start at zero and are monotonic."

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Identical sample indices are used for both streams. Before slicing, a common length `n_common = min(len(position), all neural row counts)` is computed and every trial's upper bound is clipped to it, which absorbs the ten sessions with one extra neural sample. A shape assertion then requires `neural.shape[1] == inp.shape[1] == out.shape[1]` for every trial.

ii.
```python
        n_common = min([len(position)] + [ds.shape[0] for ds in neural_series])
        for trial_idx, (lo, hi) in enumerate(info['bounds']):
            tid = int(info['trial_ids'][trial_idx])
            hi = min(hi, n_common)
            if hi <= lo:
                raise ValueError(f'{path.name}: no common samples in trial {tid}')
            ...
            if neural.shape[1] != inp.shape[1] or inp.shape[1] != out.shape[1]:
                raise ValueError(f'{path.name}: trial {tid} temporal shape mismatch')
```

iii. CONVERSION_NOTES Step 2: "Ten sessions have one extra neural sample relative to every framewise behavior stream; conversion must use the common stream length." Step 4: behavior and neural rows are already synchronized one-to-one in the NWB export, so no interpolation is required.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The framewise `environment` behavior time series (values −1 outside trials, 0 = ENV1, 1 = ENV2).

ii.
```python
        environment = b['environment/data'][:]
        ...
            ev = environment[idx]
            ev = ev[(ev == 0) | (ev == 1)]
```

iii. CONVERSION_NOTES Step 2: "Environment uses 0/1 in trials and -1 outside trials." Step 2 statistics: "73 sessions contain ENV1 trials only, 68 ENV2 only, and 11 contain both environment codes."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial's index range, samples with value −1 are discarded, the modal remaining value (0 or 1) is taken with `np.bincount(...).argmax()`, and that single value is broadcast as a constant over all timepoints of the trial. A trial with no valid environment sample raises an error.

ii.
```python
            ev = environment[idx]
            ev = ev[(ev == 0) | (ev == 1)]
            if not len(ev):
                raise ValueError(f'{path.name}: no valid environment in trial {tid}')
            envs[tid] = int(np.bincount(ev.astype(int), minlength=2).argmax())
```
```python
            inp = np.vstack((t_rel,
                             np.full(n, info['environments'][trial_idx], np.float32),
                             ...
```

iii. CONVERSION_NOTES Step 5 mapping: "`environment` → `input[1]` | Modal in-trial 0/1 value, broadcast over trial | ENV1=0, ENV2=1." The instructions specify environment as a per-trial binary, and the −1 sentinel occurs inside the agent's trial windows (the ITI portion), so a mode over valid codes is used instead of the raw framewise value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The native `trial number` behavior stream value for the block (`tid`), not a loop counter. Values run 0..N−1 within a session (global range [0, 99]).

ii.
```python
        trial_ids = np.unique(trial[trial >= 0]).astype(int)
        ...
            tid = int(info['trial_ids'][trial_idx])
```

iii. CONVERSION_NOTES Step 5 mapping: "`trial number` → `input[2]` | Native zero-based continuous trial index, broadcast | Preserves source numbering." Step 2: "Trial numbering is contiguous and has no detected gaps in any session."

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the native integer to a float32 constant across the trial's timepoints. Because only one fragment was dropped dataset-wide (and it was the last trial of its session), the retained numbering remains 0..N−1 contiguous.

ii.
```python
            inp = np.vstack((t_rel,
                             np.full(n, info['environments'][trial_idx], np.float32),
                             np.full(n, tid, np.float32),
                             np.full(n, previous, np.float32))).astype(np.float32)
```

iii. Step 10 Check 3: "For the same three trials, independently reconstructed relative time, environment, native trial number, and prior native-trial reward outcome all match exactly."

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The sparse `Reward` event series' `timestamps` (reward amounts are constant 0.004 mL and unused), tested against the behavior timestamp interval of the *preceding* native trial.

ii.
```python
        reward_ts = b['Reward/timestamps'][:]
        ...
            outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
```

iii. CONVERSION_NOTES Step 2: "`Reward` is instead a sparse timestamped event series whose values are 0.004 mL." Step 4: "A trial is rewarded if any sparse reward timestamp falls within its frame timestamp interval. Multiple deliveries remain binary 1."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Outcomes are computed for **all** native trials first, then shifted by one with 0 prepended (first trial = omitted = 0); the shift is applied *before* the `trial_start` keep-mask, so the previous-trial value is the true preceding native trial even when a neighbour is dropped. The per-trial value is broadcast over all timepoints.

ii.
```python
        previous = np.r_[0, outcomes[:-1]].astype(np.int8)
        keep = has_start
        ...
        zones = zones[keep]; outcomes = outcomes[keep]; envs = envs[keep]; previous = previous[keep]
```
```python
            previous = info['previous_outcomes'][trial_idx]
            inp = np.vstack((..., np.full(n, previous, np.float32))).astype(np.float32)
```

iii. CONVERSION_NOTES Step 10 Check 9: "Previous outcome is shifted by one native trial, with first trial set to omission=0." Step 10 fix note: the keep-mask was applied so as to "preserve native trial number and previous native-trial outcome."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The framewise `position` stream plus a per-trial reward-zone identity inferred from the `reward_zone` counter stream and `position`. For each trial the position at the *first* frame with `reward_zone > 0` is snapped to the nearest canonical zone start (80, 200 or 320 cm). Trials with no zone entry (omissions that never triggered the counter) inherit the label of the nearest trial that has one. A guard rejects any session inferred to have more than one zone switch.

ii.
```python
ZONE_STARTS = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_WIDTH = 50.0
...
            event_idx = idx[zone_event[idx] > 0]
            if len(event_idx):
                entry_pos = position[event_idx[0]]
                zones[tid] = int(np.argmin(np.abs(ZONE_STARTS-entry_pos)))
        zones = nearest_fill(zones)
        ...
        changes = int(np.sum(np.diff(zones) != 0))
        if changes > 1:
            raise ValueError(f'{path.name}: inferred {changes} reward-zone switches')
```
```python
def nearest_fill(labels):
    """Fill missing trial labels from nearest trial with observed zone entry."""
    ...
        nearest = np.argmin(np.abs(missing[:, None] - known[None, :]), axis=1)
        labels[missing] = labels[known[nearest]]
```

iii. CONVERSION_NOTES Step 4: "The `reward_zone` stream is not reward-zone identity: values 0–8 show it is likely a cumulative zone-entry event counter, so reward-zone location must be … inferred from reward metadata/position." And: "First positive `reward_zone` frame occurs near 80, 200, or 320 cm; omission trials may lack an event … Label observed trials by nearest start; nearest-known trial propagation gives one block in 75 sessions and exactly two blocks in 77, with no extra switches. Counts A/B/C = 4,185/4,008/4,024."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed Euclidean distance to the closed interval [start, start+50]: negative before the zone, exactly 0 anywhere inside it, positive past it. Applied to every sample in the trial window, including the ITI samples where `position` is the −50 cm sentinel (those yield distances of −130/−250/−370 cm and land in class 0). Class-0 frequency is 0.452 in the converted data versus 0.253 in the human reference.

ii.
```python
def discretize_distance(position, zone_start):
    """Signed distance to closed reward-zone interval, then requested 7 bins."""
    p = np.asarray(position)
    distance = np.where(p < zone_start, p-zone_start,
                        np.where(p > zone_start+ZONE_WIDTH,
                                 p-(zone_start+ZONE_WIDTH), 0.0))
```

iii. CONVERSION_NOTES Step 4: "For requested 'distance to any location in reward zone,' use signed Euclidean distance to interval [start,start+50]: negative before, exactly 0 inside, positive after. This differs intentionally from circular distance-to-start because the requested zero class denotes being anywhere in-zone." Step 5 decision 6 asserts "intertrial sentinel positions (-50/-500) are excluded naturally."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned by explicit boolean masks: 0: d < −50; 1: −50 ≤ d < −10; 2: −10 ≤ d < 0; 3: d == 0; 4: 0 < d ≤ 10; 5: 10 < d ≤ 50; 6: d > 50. (The positive-side bins are right-closed, where the reference's `np.digitize` bins are left-closed; only samples at exactly +10/+50 cm differ.) Stored as int8.

ii.
```python
    out = np.empty(p.shape, dtype=np.int8)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
```

iii. CONVERSION_NOTES Step 6: "Boundary tests cover every specified discretization edge." Step 5 mapping row states the class edges verbatim from the Decoder Task specification, with class 3 defined as "Exactly zero throughout zone."

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same `[lo, hi)` index range as the neural slice, so it is sample-for-sample aligned with the neural matrix; no lag or shift is introduced. The per-trial shape assertion enforces equal lengths.

ii.
```python
            dist_cls, signed_dist = discretize_distance(position[lo:hi], zone_start)
            out = np.vstack((dist_cls, ...)).astype(np.int8)
            if neural.shape[1] != inp.shape[1] or inp.shape[1] != out.shape[1]:
                raise ValueError(f'{path.name}: trial {tid} temporal shape mismatch')
```

iii. Step 10 Check 4: "Independently reconstructed signed zone distance classes, position classes, speed classes, binary lick, inferred A/B/C zone, and sparse-timestamp reward outcome all match exactly" against the raw files.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The framewise `position` behavior time series (cm), read once per session and sliced per trial.

ii.
```python
        position = b['position/data'][:]
        ...
            out = np.vstack((dist_cls,
                             discretize_position(position[lo:hi]),
```

iii. CONVERSION_NOTES Step 2: "Position is in cm and speed in cm/s." Step 5 mapping: "`position` → `output[1]` | Classes `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360` | Five equal 90 cm bins."

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None other than discretization: the raw cm value is binned directly. Because the trial window starts in the ITI, the −50 cm teleport sentinel samples are included and fall in class 0, which raises class-0 frequency to 0.421 (reference: 0.211) and makes the five classes markedly non-uniform (0.421/0.130/0.169/0.166/0.114 vs the reference's 0.211/0.178/0.231/0.227/0.154).

ii.
```python
def discretize_position(position):
    # <90, [90,180), [180,270), [270,360], >360
    p = np.asarray(position)
    return np.select([p < 90, p < 180, p < 270, p <= 360],
                     [0, 1, 2, 3], default=4).astype(np.int8)
```

iii. CONVERSION_NOTES Step 5 key decision 6: "**Position outliers**: only numbered-trial samples are used; intertrial sentinel positions (-50/-500) are excluded naturally." Step 4: track is 0–450 cm so five 90 cm bins are equal-sized.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.select` with 90 cm edges: 0: p < 90; 1: 90 ≤ p < 180; 2: 180 ≤ p < 270; 3: 270 ≤ p ≤ 360; 4: p > 360. The first and last bins are open, so off-track values fall into the end classes. (Only p exactly 360 is classified differently from the reference.)

ii.
```python
    return np.select([p < 90, p < 180, p < 270, p <= 360],
                     [0, 1, 2, 3], default=4).astype(np.int8)
```

iii. Step 5 mapping: "Five equal 90 cm bins," matching the Decoder Task specification for a 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[lo, hi)` indices as the neural slice; no shifting or resampling. Behavior and ophys rows are 1:1 in the NWB export, and the trial-level shape assertion enforces equality.

ii.
```python
            out = np.vstack((dist_cls,
                             discretize_position(position[lo:hi]),
                             discretize_speed(speed[lo:hi]),
                             (lick[lo:hi] > 0).astype(np.int8), ...
```

iii. Step 10 Check 7: "Conversion preserves one-to-one NWB rows and aligns trials at their first numbered sample/start. Explicit timestamps yield 64.4836 ms spacing."

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The framewise `lick` behavior time series (per-frame lick counts, integer values 0–6).

ii.
```python
        lick = b['lick/data'][:]
        ...
                             (lick[lo:hi] > 0).astype(np.int8),
```

iii. CONVERSION_NOTES Step 4: "Values 0–6 fluctuate within trials rather than being globally cumulative … Binary lick output is `lick > 0` per frame, not a difference of adjacent frames."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Thresholding at > 0 to a binary int8 series; no smoothing, no dilation, no rate conversion. Converted fraction of licking frames is 0.182 (reference 0.230; the difference comes from the extra non-licking ITI frames included in each trial).

ii.
```python
                             (lick[lo:hi] > 0).astype(np.int8),
```

iii. Step 5 mapping: "`lick` → `output[3]` | `(lick > 0).astype(int)` per frame | Frame-bin lick presence." The instructions require a binary 0/1 output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[lo, hi)` index range as the neural data; framewise, no shift.

ii.
```python
            out = np.vstack((..., (lick[lo:hi] > 0).astype(np.int8), ...))
```

iii. Step 10 Check 4 verified the binary lick series against an independent raw reconstruction for three trials in one-plane and two-plane sessions.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Same source as 7-a: the `reward_zone` counter stream (first positive frame per trial) combined with `position` at that frame, snapped to the nearest of 80/200/320 cm, with nearest-trial propagation for trials lacking an entry event.

ii.
```python
            event_idx = idx[zone_event[idx] > 0]
            if len(event_idx):
                entry_pos = position[event_idx[0]]
                zones[tid] = int(np.argmin(np.abs(ZONE_STARTS-entry_pos)))
        zones = nearest_fill(zones)
```

iii. See 7-a. CONVERSION_NOTES Step 4 documents the validation: at most one switch per session, and near-uniform A/B/C counts of 4,185/4,008/4,024.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The inferred zone index (0 = A, 1 = B, 2 = C) is broadcast as a constant int8 over all timepoints of the trial, so the per-trial label is stored as a time-varying row. Converted class fractions are 0.331/0.331/0.339 (reference 0.329/0.337/0.334).

ii.
```python
            out = np.vstack((..., np.full(n, info['zones'][trial_idx], np.int8),
                             np.full(n, info['outcomes'][trial_idx], np.int8))).astype(np.int8)
```
```python
       output_values=[..., ['A','B','C'], ['omitted','rewarded']],
```

iii. Step 5 mapping: "first reward-zone entry position, block-filled → `output[4]` | nearest 80/200/320 cm → A/B/C = 0/1/2, broadcast | Nearest-known trial fill handles omissions." The instructions ask for per-trial outputs to be time-varying where possible.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward/timestamps` event series, compared against the trial's behavior timestamp interval `[ts[lo], ts[hi]]`.

ii.
```python
        reward_ts = b['Reward/timestamps'][:]
        ...
            outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
```

iii. Step 4: "Reward is event-based … `Reward` has sparse timestamps and 0.004 mL values … A trial is rewarded if any sparse reward timestamp falls within its frame timestamp interval."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary per trial (any reward event in the interval → 1, multiple deliveries still 1), broadcast over all timepoints as int8. No nearest-bin `searchsorted` mapping and no assertion that the event time lies within half a time bin of a behavior sample — interval containment is used instead. Converted fractions: 0.156 omitted / 0.844 rewarded (reference 0.157/0.843).

ii.
```python
            outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
...
            out = np.vstack((..., np.full(n, info['outcomes'][trial_idx], np.int8))).astype(np.int8)
```

iii. Step 12: "Reward outcome and previous outcome were checked on three specific raw trials (rewarded, omitted, and a transition) by direct sparse timestamp alignment. Converted labels and previous-outcome inputs matched exactly." Step 10 Check 11 reports the trial-level reward rate of ~84.65% in the raw data.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases:
- **Neural/behavior length mismatch** (10 sessions with one extra neural sample): a common length is computed and every trial's end index is clipped to it.
- **Non-finite neural values**: replaced with 0 via `np.nan_to_num`, guarded by a cheap `np.all(np.isfinite)` test.
- **Missing reward-zone entry on omission trials**: filled from the nearest trial with an observed entry.
- **Truncated recording-edge trial** (no `trial_start`): dropped.
- **Multi-plane ROI table** (28 m17/m18 sessions where `PlaneSegmentation` pools both planes): the accepted-cell mask is split by `planeIdx` and each plane's Deconvolved matrix is masked separately, with a hard shape check.
Structural violations (non-contiguous trial numbers, non-contiguous trial samples, empty trials, no valid environment code, more than one inferred zone switch, per-trial shape mismatch, ROI/mask mismatch) raise exceptions rather than being silently patched. A final `validate_data()` re-checks all shapes and finiteness before pickling.

ii.
```python
        n_common = min([len(position)] + [ds.shape[0] for ds in neural_series])
        ...
            hi = min(hi, n_common)
            if hi <= lo:
                raise ValueError(f'{path.name}: no common samples in trial {tid}')
            ...
            if not np.all(np.isfinite(neural)):
                neural = np.nan_to_num(neural, copy=False)
```
```python
def validate_data(data):
    ns=len(data['neural'])
    assert ns==len(data['input'])==len(data['output'])==len(data['subject_idx'])==len(data['brain_region_idx'])
    for s in range(ns):
        assert len(data['neural'][s])==len(data['input'][s])==len(data['output'][s])>=2
        ...
            assert np.isfinite(n).all() and np.isfinite(i).all()
```

iii. CONVERSION_NOTES Step 9/10: "Ten one-sample neural/behavior length discrepancies are safely handled by behavior trial indexing"; "Initial full run exposed 28 m17/m18 sessions where PlaneSegmentation combines both planes. Fixed by selecting accepted cells separately using `planeIdx` and concatenating every `Deconvolved/plane*`"; "Raw m11 session 03 native trial 80 was uniquely only 3 frames … It was excluded."

## 13-a. What are the most time-consuming steps of the code?

i. Per-session timing is printed (0.10–0.37 s per session) and the total run is reported: 152 sessions in 60.46 s, of which the dominant costs are (1) the per-trial HDF5 reads of the Deconvolved arrays — 12,216 slice reads, each pulling *all* ROI columns before masking — and (2) writing the 13.33 GB pickle at the end. The behavior-array reads in `inspect_session` (which also re-reads `trial_start/data` once per trial) are the next largest cost. Everything downstream (discretization, broadcasting) is vectorized and negligible.

ii.
```python
def convert_session(path, make_plot=False):
    t0 = time.perf_counter()
    ...
    elapsed=time.perf_counter()-t0
    print(f'  {path.name}: {len(trials_neural)} trials, {info["n_rois"]} ROIs -> '
          f'{info["n_cells"]} cells, {sum(x.shape[1] for x in trials_neural)} samples, {elapsed:.2f}s', flush=True)
```
```python
    dt=time.perf_counter()-t0
    print(f'Done: {len(neural)} sessions, {sum(map(len,neural))} trials, {sum(x[0].shape[0] for x in neural)} session-cells; {out.stat().st_size/1e9:.3f} GB; {dt:.2f}s',flush=True)
```

iii. CONVERSION_NOTES Step 6: "Loading all 260,091 segmented ROI streams would waste memory because only 138,678 accepted cells are used. Reopening files or reading each neural timepoint separately would cause excess HDF5 I/O." Step 7 estimated ~97 s for the full set and Step 9 reported 52.6 s/60.5 s actual, far under the 15-minute budget.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain:
- `inspect_session`'s per-trial loop calls `np.flatnonzero(trial == tid)` once per trial, an O(n_trials × T) scan that a single `np.searchsorted` / `np.diff` on the trial-number array would replace, and it issues one fancy-indexed HDF5 read of `trial_start/data` per trial instead of reading the array once.
- The reward test `np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi]))` is re-evaluated over the whole reward vector for every trial; a single `np.searchsorted` would do all trials at once.
- `convert_session`'s per-trial loop performs one HDF5 read per trial per plane; the discretizations inside it could be computed once for the whole session and then sliced.
None of these were flagged in the notes; the agent considered the loop structure adequate given the 60 s runtime.

ii.
```python
        for tid in trial_ids:
            idx = np.flatnonzero(trial == tid)
            ...
            has_start[tid] = bool(np.any(b['trial_start/data'][idx] > 0))
            ...
            outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
```

iii. CONVERSION_NOTES Step 6 claims the speedups that were made: "Each session is opened once for conversion; each trial is read as one contiguous HDF5 slice and then cell-selected. Behavior arrays are read once per session and vectorized discretization is used."

## 13-c. What processing does the code repeat multiple times?

i. Every session's file is opened twice — once by `inspect_session` and again by `convert_session` — and the behavior arrays `position` and the trial-number timestamps are read in both passes. `trial_start/data` is re-read once per trial rather than once per session. Within the trial loop, the full-width neural block `ds[lo:hi, :]` is read and then discarded down to the accepted columns, so roughly twice the required bytes are transferred (138,678 of 260,091 ROIs are kept). Verification/statistics passes (`train_decoder.py --verify-only`, the independent sanity script) re-read the same raw files, as in the reference.

ii.
```python
def convert_session(path, make_plot=False):
    t0 = time.perf_counter()
    info = inspect_session(path)          # opens the file once
    ...
    with h5py.File(path, 'r') as f:       # opens it again
        b = f['processing/behavior/BehavioralTimeSeries']
        ...
        position = b['position/data'][:]
```

iii. CONVERSION_NOTES Step 6 states the intent that "Each session is opened once for conversion" — i.e. the agent counted the inspect pass and the conversion pass as one logical unit, and judged the repeated small-array reads acceptable because behavior arrays are tiny relative to the ophys arrays.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Minor items:
- `discretize_distance` returns a continuous `signed_dist` array for every trial; it is only consumed by the `--show-processing` plot and is otherwise discarded.
- `np.all(np.isfinite(neural))` is evaluated on every trial of every session although no non-finite values are ever found.
- Per-session `summary` dicts (zone counts, ROI counts, timings) are assembled for all 152 sessions and embedded in `metadata['session_info']`, unused by the decoder.
- `inspect_session` computes `zones`, `outcomes`, `envs` and `previous` for trials that are subsequently dropped by the keep-mask (only one trial dataset-wide, so negligible).
- The trial slices read all ROI columns from HDF5 and then discard the ~47% that are not accepted cells, contrary to the stated plan to "read only selected neural columns."

ii.
```python
            dist_cls, signed_dist = discretize_distance(position[lo:hi], zone_start)
            ...
            if make_plot and plot_data is None and trial_idx == min(4, len(info['bounds'])-1):
                plot_data=(tid, neural, t_rel, position[lo:hi], speed[lo:hi],
                           lick[lo:hi], signed_dist, out)
```
```python
            plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
                          for ds, idx in zip(neural_series, cell_indices)]
```

iii. CONVERSION_NOTES Step 5 key decision 7: "**Memory**: float32 neural/input and compact integer outputs; read only selected neural columns and trial rows from HDF5." Step 6: "Neural arrays use float32 and outputs use int8 to reduce pickle size and memory."
