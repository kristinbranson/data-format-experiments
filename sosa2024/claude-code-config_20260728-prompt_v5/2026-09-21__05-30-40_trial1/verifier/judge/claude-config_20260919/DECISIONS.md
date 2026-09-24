# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects as every directory in `/app/data` matching `sub-*`, then globs every `*.nwb` file inside each subject directory (152 files total). Files are opened directly with `h5py` (not `pynwb`) and the needed HDF5 datasets are read eagerly into numpy arrays: behaviour (`processing/behavior/BehavioralTimeSeries/*`), ophys (`processing/ophys/Fluorescence/plane*/data`, `Neuropil/plane*/data`, `ImageSegmentation/PlaneSegmentation/iscell` and `planeIdx`), and the imaging rate from `acquisition/TwoPhotonSeries/imaging_plane/imaging_rate`. One pass over the dataset; no survey/caching pass. `--sample` picks 2 hard-coded files (index 0 and 14); `--full` (default) processes all.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])

all_nwb_files = []
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
    all_nwb_files.extend(nwb_files)
```
```python
    with h5py.File(nwb_path, 'r') as f:
        imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
        bts = f['processing/behavior/BehavioralTimeSeries']
        position = bts['position/data'][:]
        ...
        ophys = f['processing/ophys']
        iscell_full = ophys['ImageSegmentation/PlaneSegmentation/iscell'][:]
        plane_idx  = ophys['ImageSegmentation/PlaneSegmentation/planeIdx'][:]
        planes = sorted([k for k in ophys['Fluorescence'].keys() if k.startswith('plane')])
```

iii. CONVERSION_NOTES Step 2 documents the NWB layout the AI reverse-engineered (`dandiset.yaml` confirmed DANDI:001361 with 152 files / 11 subjects), and Step 10 records the sanity check "Sessions 152 / Subjects 11 / m11 = 12 sessions, all others 14" as PASS. `h5py` was used because the AI only needed a fixed list of arrays and wanted to avoid the `pynwb` object-construction overhead.

## 1-b. How are the data split into subjects?

i. One subject per `sub-*` directory. The subject id is taken from the parent directory name of each NWB file and stripped of the `sub-` prefix. The subject list in the output is `sorted(set(...))` of those ids (alphabetical: m11…m19, m3, m4, m7) and `subject_idx` indexes into it per session.

ii.
```python
subject = os.path.basename(os.path.dirname(nwb_path))
...
session_info = {'subject': subject.replace('sub-', ''), ...}
```
```python
unique_subjects = sorted(set(si['subject'] for si in session_infos))
subject_idx = np.array([unique_subjects.index(si['subject']) for si in session_infos])
```

iii. Step 2 of the notes: "11 subjects: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19", cross-checked in Step 4 against the paper's "n = 11 mice" for the switch task. No cross-animal cell alignment is attempted.

## 1-c. How are the data split into sessions?

i. One session per NWB file. The session id string (`ses-03`, …) is parsed out of the filename and kept in `metadata['session_info']`; sessions are not renumbered. Sessions are appended in sorted (subject, filename) order, and `subject_idx` is built in the same order. A session is dropped only if it yields fewer than 2 trials (never happens).

ii.
```python
ses_id = os.path.basename(nwb_path).split('_')[1]   # 'ses-03'
```
```python
        if len(neural_trials) < 2:
            print(f"    WARNING: Skipping session with {len(neural_trials)} trials")
            continue
```

iii. Step 2: "Each subject has 12-14 NWB files (sessions), one per day. m11 starts at ses-03 (imaging began day 3)". Step 10 verifies 152 sessions with the expected per-subject counts. The requirement of ≥2 trials per session comes from the target-format spec.

## 1-d. How are the data split into trials?

i. A trial is the half-open window from a `trial_start` event to the next `teleport` event, i.e. the lap, excluding the inter-trial teleport period. The AI takes *every* sample where `trial_start > 0` and every sample where `teleport > 0`, converts them to 1-based indices, and truncates both lists to a common length if they differ. Per trial it then slices `[s, e)` with `s = trial_start-1` (the trial-start sample) and `e = teleport-1` (the teleport sample, exclusive).

ii.
```python
    trial_start_inds = np.where(trial_start_signal > 0)[0] + 1   # convert to 1-indexed
    teleport_inds    = np.where(teleport_signal   > 0)[0] + 1

    if len(trial_start_inds) != len(teleport_inds):
        min_len = min(len(trial_start_inds), len(teleport_inds))
        trial_start_inds = trial_start_inds[:min_len]
        teleport_inds    = teleport_inds[:min_len]
```
```python
    for i in range(n_trials):
        s = trial_start_inds[i] - 1   # 0-indexed start
        e = teleport_inds[i] - 1      # 0-indexed end
        if e <= s:
            continue
```

iii. Step 5, Key Decision 5: "Trial definition: Each trial runs from trial_start to teleport. Teleport/ITI periods excluded (set to NaN in reference)." The 1-based convention was adopted so the same index arrays could be handed to the reference `dff()` function, which internally slices `start-1 : stop-1`. Step 10 checks trials/session range 41–100 and mean 80.4 against the paper's 80.5 ± 7.4.

## 1-e. How are trials filtered based on quality controls?

i. Essentially no trial-level quality filter is applied. Only degenerate trials (`teleport_index <= trial_start_index`) are skipped, and sessions with <2 usable trials are dropped. There is no minimum-duration criterion. The one quality correction that *is* applied is per-trial and affects only the lick output, not trial inclusion: trials where >35 % of frames have cumulative lick > 2 are treated as a stuck lick sensor and have their lick trace zeroed (see 9-b). All 12,216 trials in the dataset are retained.

ii.
```python
        if e <= s:
            continue
```
```python
    for i in range(n_trials):
        s = trial_start_inds[i] - 1
        e = teleport_inds[i] - 1
        trial_lick = lick_corrected[s:e]
        n_frames = len(trial_lick)
        if n_frames > 0 and np.sum(trial_lick > 2) / n_frames > LICK_ERROR_THR:
            lick_corrected[s:e] = 0
```

iii. Step 3, "Trial curation rules": "1. Lick sensor error: trials with >35% of frames having cumulative lick > 2 -> lick data set to NaN for that trial. 2. All valid trials (trial_number >= 0) included." The AI found no other trial-exclusion rule in the paper, and Step 10 notes that the long trials it saw (up to 216 s) are "genuine behavioral variability" rather than artefacts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From the raw suite2p traces `processing/ophys/Fluorescence/plane<N>/data` (F) and `processing/ophys/Neuropil/plane<N>/data` (Fneu), pooled over planes. The NWB `Deconvolved` array is explicitly **not** used. Cell identity comes from `ImageSegmentation/PlaneSegmentation/iscell`, with `planeIdx` used to map ROI rows to planes.

ii.
```python
        for plane in planes:
            F_plane    = ophys[f'Fluorescence/{plane}/data'][:]
            Fneu_plane = ophys[f'Neuropil/{plane}/data'][:]
            pnum = int(plane.replace('plane', ''))
            plane_roi_mask = plane_idx == pnum
            iscell_plane = iscell_full[plane_roi_mask, 0]
            assert F_plane.shape[1] == np.sum(plane_roi_mask), ...
            F_all.append(F_plane); Fneu_all.append(Fneu_plane)
            cell_mask_all.append(iscell_plane > 0)
        F_concat    = np.concatenate(F_all, axis=1)
        Fneu_concat = np.concatenate(Fneu_all, axis=1)
        cell_mask   = np.concatenate(cell_mask_all)
```

iii. Step 4, Discrepancies table: "NWB Deconvolved = Raw suite2p deconv (no NaN in ITI) … Paper says Custom dF/F + OASIS → **Must recompute dF/F from F and Fneu**". Step 5, Key Decision 1 repeats this. Key Decision 6: "Multi-plane animals (m17, m18): Pool neurons from both planes (matching reference)."

## 2-b. How is the `neural` data processed?

i. The AI re-implements the paper's `preprocessing.dff` in `compute_dff_and_deconvolve`: (1) blank everything outside laps to NaN, (2) subtract `0.7 * Fneu`, (3) per trial add back `0.7 *` the trial's mean neuropil, (4) per trial compute a maximin baseline (Gaussian σ = 15 samples along time, then a 300-sample running minimum, then a 300-sample running maximum ≈ the Methods' 20 s window), (5) `dF/F = (F − baseline) / |baseline|`, (6) per trial smooth dF/F with a 2-sample Gaussian, (7) per trial deconvolve with suite2p's OASIS at `tau = 0.7` and `frame_rate / n_planes`. dF/F is computed for *all* ROIs and the `iscell` subset is taken afterwards. Baselines are always restricted to the lap — the paper's `keep_teleports` branch (per-animal, per-day, from `teleport_metadata.py`) is never used.

ii.
```python
    f_ -= NEU_COEF * f_neu_
    ...
        f_[:, s:e] = f_[:, s:e] + NEU_COEF * np.nanmean(f_neu_[:, s:e], axis=1, keepdims=True)
        flow[:, s:e] = nansmooth(f_[:, s:e], 15, axis=1)
        flow[:, s:e] = minimum_filter1d(flow[:, s:e], BASELINE_WINDOW, axis=-1)
        flow[:, s:e] = maximum_filter1d(flow[:, s:e], BASELINE_WINDOW, axis=-1)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    ...
        dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)
        trial_dff = dff[:, s:e].copy()
        trial_dff[np.isnan(trial_dff)] = 0
        spks[:, s:e] = oasis(trial_dff, 2000, TAU, frame_rate / n_planes)
```
```python
NEU_COEF = 0.7; TAU = 0.7; BASELINE_WINDOW = 300; DFF_SMOOTH_SIGMA = 2
```

iii. Step 1 of the notes transcribes the eight-step pipeline out of `preprocessing.py` and Step 5 Key Decision 1 justifies recomputing it rather than using the stored `Deconvolved` array. The trajectory (step 45) records the AI explicitly rejecting the shortcut: "the task explicitly says to match the reference processing, so I need to take that seriously rather than assume equivalence."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only one filter: ROIs with `iscell[:,0] > 0` (suite2p's curated cells) are kept. The paper's additional exclusion of putative interneurons (dF/F–speed Pearson r > 0.5) is deliberately **not** applied. Result: 138,678 neurons, 155–2,341 per session.

ii.
```python
            iscell_plane = iscell_full[plane_roi_mask, 0]
            cell_mask_all.append(iscell_plane > 0)
    ...
    cell_indices = np.where(cell_mask)[0]
    events_cells = events[cell_indices, :]   # (n_cells, n_timepoints)
    n_neurons = len(cell_indices)
```

iii. Step 5, Key Decision 3: "We do NOT apply the interneuron correlation filter because: (a) it would require computing the full dF/F timeseries first and correlating with speed, (b) it only excludes ~0.42% of cells, (c) the iscell manual curation already removed obvious interneurons." Step 10 accounts for the resulting discrepancy: "Max neuron count (2341, m18) slightly exceeds paper's 2172 because we skip interneuron correlation filter". (Rationale (a) is not actually true of the AI's own script, which does compute the full dF/F trace.)

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires no extra work: the first column of every trial's neural matrix is the frame at the `trial_start` index, because neural and behavioural samples share one frame index. Trial neural data is the slice `events_cells[:, s:e]`, NaNs replaced by 0 (there are none inside a lap by construction). `metadata['temporal_alignment_event'] = 'start of trial (trial_start signal)'`, `off_start = 0.0`, `off_end = None`.

ii.
```python
        trial_neural = events_cells[:, s:e].copy()
        trial_neural[np.isnan(trial_neural)] = 0
        neural_trials.append(trial_neural.astype(np.float32))
```
```python
            'temporal_alignment_event': 'start of trial (trial_start signal)',
            'off_start': 0.0,
            'off_end': None,
```

iii. Step 5, Key Decision 4/5 and the mapping table: neural and behaviour are sampled on the same imaging frames, so the trial window index is shared. The `--show-processing` figure plots the trial-0 neural trace against the behavioural traces on a common time axis as the visual check.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One time bin = one imaging frame per plane ≈ 64.484 ms (15.5078125 Hz). No rebinning, resampling or interpolation is performed anywhere. For the two-plane animals (m17, m18) the stored `imaging_rate` is the 31.0156 Hz scanner rate, and the AI divides it by `n_planes` to recover the 15.5 Hz per-plane rate — used both for the OASIS kernel and for the time axis. `metadata['time_bin_size']` is hard-coded as `1000/15.5078125`.

ii.
```python
    effective_rate = imaging_rate
    if n_planes > 1:
        effective_rate = imaging_rate / n_planes
    ...
        spks[:, s:e] = oasis(trial_dff, 2000, TAU, frame_rate / n_planes)
```
```python
            'time_bin_size': 1000.0 / 15.5078125,  # ms per frame
            'imaging_rate_hz': 15.5078125,
```

iii. Step 5, Key Decision 4: "Time bin = imaging frame. Each time bin is one imaging frame (~64.5 ms at 15.5 Hz). This matches the reference temporal resolution." Step 2: "Multi-plane animals (m17, m18) … Imaging rate ~31 Hz interleaved -> ~15.5 Hz per plane", so bins are the same size for every session.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Not from the stored behavioural `timestamps` at all. It is synthesised from the sample index within the trial and the per-plane imaging rate (`imaging_rate`, divided by `n_planes` where applicable). Behavioural `position/timestamps` is read, but only for the reward-event comparison.

ii.
```python
        time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate
```

iii. The notes' Step 5 mapping table lists the transform as "(frame_idx - trial_start_idx) / imaging_rate". The implicit justification is that the acquisition clock is fixed: the AI treats frames as uniformly spaced at the imaging rate. (In this dataset that is exactly true — `np.diff(timestamps)` is constant to 1e-12 s and equals `n_planes/imaging_rate` — so the two routes give identical numbers.)

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. None beyond `index / rate`, which starts each trial at exactly 0.0 s and increases by one bin per sample. Stored as row 0 of the `(4, n_timepoints)` input array in float32. Full-data range: [0.0, 216.5] s.

ii.
```python
        time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate
        input_tv = time_from_start.reshape(1, -1)
        inputs = np.vstack([input_tv, ...]).astype(np.float32)
```

iii. No separate justification is given; it follows directly from the decoder-input spec "Time from start of trial in seconds (continuous, time-varying)". Step 10 flags the 216.5 s maximum as worth checking and concludes it is a genuine long trial (m4/ses-04), not an alignment bug.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the same `s:e` index window is used to slice the neural matrix and to generate the time vector, and both have length `n_tp = e - s`. The AI treats behavioural and ophys streams as sharing one frame index and performs no resampling, no offset correction, and no assertion that the two streams have equal length.

ii.
```python
        n_tp = e - s
        trial_neural = events_cells[:, s:e].copy()
        ...
        time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate
```

iii. Step 3, Processing Details: "Temporal alignment: All data aligned to 2P imaging frames (~15.5 Hz)". The `--show-processing` plots overlay neural traces and behavioural variables on a shared time axis to make misalignment visible; Step 7 records "Processing plots generated and reviewed - all look correct."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioural time series (`-1` before scanning starts, `0` = ENV1, `1` = ENV2).

ii.
```python
        environment = bts['environment/data'][:]
```

iii. Step 2: "`environment`: -1 before scanning, 0=ENV1, 1=ENV2", reconciled in Step 4 with the paper's ENV1/ENV2 nomenclature.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, the AI takes the median of the environment samples inside the trial after discarding any negative (pre-scanning) value, casts to int, and broadcasts that single value across all timepoints of the trial (row 1 of the input array). If a trial had no valid sample, the previous trial's value is carried forward.

ii.
```python
    env_per_trial = np.zeros(n_trials, dtype=np.int64)
    for i in range(n_trials):
        s = trial_start_inds[i] - 1
        e = teleport_inds[i] - 1
        env_vals = environment[s:e]
        valid_env = env_vals[env_vals >= 0]
        if len(valid_env) > 0:
            env_per_trial[i] = int(np.median(valid_env))
        elif i > 0:
            env_per_trial[i] = env_per_trial[i - 1]
```
```python
            np.full((1, n_tp), env_type, dtype=np.float32),   # (1, n_tp) - env type
```

iii. Step 4 resolution: "Environment encoding … -1 before scanning … Use env value per trial (ignoring -1)." The median over the trial is the AI's robustness device against stray samples. Per-trial constants are broadcast to a full time series because the format spec prefers `(d_input, n_timepoints)` arrays.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. From the position of the trial in the `trial_start`/`teleport` sequence — i.e. the Python loop counter `i` — not from the NWB `trial number` time series. (`trial number` is read into a local variable but never used.)

ii.
```python
        trial_number = np.float32(i)
```
```python
        trial_num = bts['trial number/data'][:]   # loaded, never used
```

iii. The Step 5 mapping table lists "NWB trial number → `input[2]`: trial_number, 0-indexed trial number", but the implementation uses the loop index. The two coincide because trials are enumerated in temporal order from the same `trial_start` events; no explicit justification for preferring the counter is recorded.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None. The 0-based within-session index is cast to float32 and broadcast constant across the trial's timepoints (row 2). It restarts at 0 in every session and reaches 99 in the 100-trial sessions.

ii.
```python
            np.full((1, n_tp), trial_number, dtype=np.float32),  # (1, n_tp) - trial number
```

iii. Follows the decoder spec "Trial number (continuous, per trial)". Verified indirectly by the reported input range `trial_number: [0.0, 99.0]`.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the same per-trial `rewarded` vector used for the reward-outcome output. That vector is built from the sparse `Reward` event time series (its own `timestamps`, compared against `position/timestamps`), and is additionally OR-ed with the `autoreward` channel inside the trial.

ii.
```python
    rewarded = determine_reward_per_trial(
        reward_timestamps, behav_timestamps, trial_start_inds, teleport_inds)

    for i in range(n_trials):
        s = trial_start_inds[i] - 1
        e = teleport_inds[i] - 1
        if np.any(autoreward[s:e] > 0):
            rewarded[i] = 1
```

iii. Step 2: "`Reward`: separate array with reward event timestamps (sparse)"; "`autoreward`: binary, marks auto-reward delivery". Step 3 notes "Auto-reward on first 10 trials of new condition", which is why the AI folds autoreward into the outcome. (In this release `autoreward` is identically 0 in all 152 sessions, so the OR is a no-op.)

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial *i* > 0 the value is `rewarded[i-1]` (1 if any reward event fell inside trial *i*−1, else 0). Trial 0 is set to 0. The value is broadcast constant over the trial (row 3).

ii.
```python
        if i == 0:
            prev_outcome = np.float32(0)  # no previous trial
        else:
            prev_outcome = np.float32(rewarded[i - 1])
        ...
            np.full((1, n_tp), prev_outcome, dtype=np.float32),  # (1, n_tp) - prev outcome
```

iii. Directly from the decoder spec "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)". The 0 for the first trial is the AI's convention for "no previous trial". Step 10 cross-checks the underlying reward rate at 84.7 % against the paper's ~15 % omission rate.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From `position` together with a per-trial reward-zone assignment. The zone is identified from the `reward_zone` behavioural channel: within a trial the AI takes the mean of `position` over the samples where `reward_zone > 0` and assigns whichever of the fixed zones A = [80,130], B = [200,250], C = [320,370] has the nearest centre (105 / 225 / 345 cm). If a trial has no `reward_zone > 0` sample at all (≈16 % of trials), the previous trial's zone is carried forward; if it is the first trial of the session, zone A is assumed.

ii.
```python
REWARD_ZONES = {'A': (80, 130), 'B': (200, 250), 'C': (320, 370)}
ZONE_CENTERS = {k: (v[0] + v[1]) / 2 for k, v in REWARD_ZONES.items()}
```
```python
        rz_active = trial_rz > 0
        if np.any(rz_active):
            rz_positions = trial_pos[rz_active]
            rz_center = np.mean(rz_positions)
        else:
            if len(zone_labels) > 0:
                zone_labels.append(zone_labels[-1]); zone_coords.append(zone_coords[-1]); continue
            else:
                zone_labels.append('A'); zone_coords.append(REWARD_ZONES['A']); continue
        dists = {k: abs(rz_center - c) for k, c in ZONE_CENTERS.items()}
        zone = min(dists, key=dists.get)
```

iii. Step 1: "Reward zone locations: A=[80,130], B=[200,250], C=[320,370] (from `reward_zone_dict` using X, Y, Z keys)". Step 5 Key Decision 8: "Reward zone determination per trial: Determine from position where reward_zone signal > 0. Zone A: center ~105 cm, Zone B: center ~225 cm, Zone C: center ~345 cm." The trajectory (step 38) shows the AI inspecting the raw `reward_zone` values (0–6) and concluding they are the VR zone signal rather than a distance measure. Step 10 validates the result: zone distribution 33.0 / 33.6 / 33.4 %.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance in cm from the animal's position to the nearest edge of the assigned zone: negative before the zone, exactly 0 anywhere inside `[zone_start, zone_end]`, positive after.

ii.
```python
def compute_distance_to_reward_zone(position, zone_start, zone_end):
    dist = np.zeros_like(position)
    before = position < zone_start
    inside = (position >= zone_start) & (position <= zone_end)
    after  = position > zone_end
    dist[before] = position[before] - zone_start   # negative
    dist[inside] = 0.0
    dist[after]  = position[after] - zone_end      # positive
    return dist
```

iii. Step 5, Key Decision 7: "Compute as signed distance from position to nearest point in the reward zone [start, end]. Negative = before zone, 0 = in zone, positive = past zone. Discretize per decoder spec." Step 10 sanity-checks the resulting distribution and explains the small bin-4 fraction (2.1 %) as the animals slowing/stopping in the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Into the 7 categories of the decoder spec by explicit boolean masks rather than `np.digitize`: `<-50 → 0`, `[-50,-10) → 1`, `[-10,0) → 2`, `==0 → 3`, `(0,10] → 4`, `(10,50] → 5`, `>50 → 6`. Note the upper boundaries 10 and 50 are inclusive here (the spec's "+10 to +50" wording is ambiguous; this only matters for exactly-equal values).

ii.
```python
def discretize_distance_to_rz(distance):
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3  # in zone
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. Transcribed straight from the Decoder Outputs specification. `output_values` documents the labels; Step 7/10 confirm all 7 bins are populated and Step 10 discusses the asymmetry of the distribution.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No alignment step. `position` is sliced with the same `s:e` window as the neural matrix, giving the same length and the same per-bin correspondence.

ii.
```python
        trial_pos = position[s:e]
        zs, ze = zone_coords[i]
        dist = compute_distance_to_reward_zone(trial_pos, zs, ze)
        dist_binned = discretize_distance_to_rz(dist)
```

iii. Same reasoning as 3-c: behaviour and ophys share the imaging frame index, so an identical slice is an identical alignment. The `--show-processing` figure plots the discretised distance for a sample trial as the visual check.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural time series (cm along the VR corridor).

ii.
```python
        position = bts['position/data'][:]
        ...
        trial_pos = position[s:e]
```

iii. Step 2: "`position`: 0-450 cm on track, -500 in teleport zone, negative before track start". Since trials stop at the teleport sample, the −500 teleport values never enter a trial.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing and discretisation — the raw cm value is used as-is.

ii.
```python
        pos_binned = discretize_position(trial_pos)
```

iii. The paper's 450 cm track and the spec's "5 equal-sized bins spanning the 450 cm track" mean the raw value needs no transformation.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm bins by boolean masks, with the first and last left open so that samples marginally outside [0, 450] fall into bins 0 and 4: `<90 → 0`, `[90,180) → 1`, `[180,270) → 2`, `[270,360) → 3`, `>=360 → 4`.

ii.
```python
def discretize_position(position):
    bins = np.zeros(len(position), dtype=np.int64)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins
```

iii. Directly from the Decoder Outputs spec. Step 10 records "Position bins: 5 populated / All populated — PASS"; the converted fractions are 0.211 / 0.178 / 0.231 / 0.227 / 0.154.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. No alignment step; the same `s:e` trial slice is used as for the neural matrix.

ii.
```python
        trial_pos = position[s:e]
        pos_binned = discretize_position(trial_pos)
```

iii. As in 3-c/7-d: one shared frame index for behaviour and ophys.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural time series (cumulative lick count per frame, values 0–6).

ii.
```python
        lick = bts['lick/data'][:]
```

iii. Step 2: "`lick`: cumulative lick count per frame (0-6)".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps. First the paper's stuck-sensor correction: for each trial, if more than 35 % of frames have cumulative lick > 2, the whole trial's lick trace is set to 0 (the paper's `correct_lick_sensor_error` sets it to NaN; the AI uses 0 because the output must be a valid category). Then binarisation: any remaining value > 0 becomes 1. This affects 69 of 12,216 trials (0.6 %).

ii.
```python
LICK_ERROR_THR = 0.35  # fraction of frames with cumsum lick > 2
...
    lick_corrected = lick.copy()
    for i in range(n_trials):
        s = trial_start_inds[i] - 1; e = teleport_inds[i] - 1
        trial_lick = lick_corrected[s:e]
        n_frames = len(trial_lick)
        if n_frames > 0 and np.sum(trial_lick > 2) / n_frames > LICK_ERROR_THR:
            lick_corrected[s:e] = 0
    lick_binary = (lick_corrected > 0).astype(np.int64)
```

iii. Step 1 identifies `correct_lick_sensor_error()` in `behavior.py` as a CURATION function ("trials where >35% of frames have cumulative lick > 2 are set to NaN"), and Step 5 Key Decision 10 states the intent: "Following reference, if >35% of frames in a trial have cumulative lick count > 2, set lick to 0 for that trial (matching reference behavior.py)." Binarisation follows the spec "Lick, time-varying. 0 = no, 1 = yes".

## 9-c. How is `output` *Lick* aligned with the neural data?

i. No alignment step; the binarised lick vector is sliced with the same `s:e` window.

ii.
```python
        trial_lick = lick_binary[s:e]
        ...
            trial_lick.reshape(1, -1),      # (1, n_tp) - lick
```

iii. As in 3-c: shared imaging-frame index.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The same per-trial zone assignment described in 7-a — `reward_zone > 0` samples plus `position`, mapped to the nearest of the three fixed zones, with carry-forward of the previous trial's label when the zone signal is absent.

ii.
```python
    zone_labels, zone_coords = determine_reward_zone_per_trial(
        position, rz_signal, trial_start_inds, teleport_inds)
```

iii. See 7-a: Step 5 Key Decision 8.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The letter label is mapped to an integer (`A→0, B→1, C→2`) and broadcast constant over the trial's timepoints as row 4 of the output array.

ii.
```python
        zone_label = zone_labels[i]
        rz_loc = {'A': 0, 'B': 1, 'C': 2}[zone_label]
        ...
            np.full((1, n_tp), rz_loc, dtype=np.int64),      # (1, n_tp) - RZ location
```
```python
            ['Zone A', 'Zone B', 'Zone C'],
```

iii. Spec: "Reward zone location, per-trial. 0 = A, 1 = B, 2 = C". Broadcast to a time series because the format spec says "If at all possible, make it time-varying". Step 10 checks the marginal distribution (33.0 / 33.6 / 33.4 %) against the expectation of roughly equal use of the three zones.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward` event series — specifically its `timestamps` — compared against the behavioural `position/timestamps`; OR-ed with any `autoreward > 0` sample inside the trial (a no-op in this release, where `autoreward` is identically zero). The `Reward/data` amounts are read but unused.

ii.
```python
        reward_data = bts['Reward/data'][:]              # unused
        reward_timestamps = bts['Reward/timestamps'][:]
        autoreward = bts['autoreward/data'][:]
        behav_timestamps = bts['position/timestamps'][:]
```

iii. Step 5, Key Decision 9: "Reward outcome per trial: Determined from Reward timestamps array - check if any reward event falls within trial boundaries." Step 2 documents that `Reward` is a sparse event series with its own timestamps rather than a per-frame channel.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial the behavioural timestamps of the first and last sample of the trial define a time window `[t_start, t_end]`; the trial is scored 1 if any reward timestamp falls inside it, else 0. The binary value is broadcast constant over the trial (row 5). Resulting rate: 84.3 % rewarded / 15.7 % omitted.

ii.
```python
def determine_reward_per_trial(reward_timestamps, behav_timestamps, trial_starts, teleports):
    rewarded = np.zeros(n_trials, dtype=np.int64)
    for i in range(n_trials):
        s = trial_starts[i] - 1
        e = teleports[i] - 1
        t_start = behav_timestamps[s]
        t_end   = behav_timestamps[min(e, len(behav_timestamps) - 1)]
        in_trial = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
        if np.any(in_trial):
            rewarded[i] = 1
    return rewarded
```
```python
            np.full((1, n_tp), reward_out, dtype=np.int64),   # (1, n_tp) - reward outcome
```

iii. Step 3 expects "~15%" omission from the Methods ("randomly omitted on ~15% of trials"); Step 10 records the achieved 15.3 % as a PASS. Working in timestamp space rather than index space avoids having to map the sparse reward events onto frames.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The handled cases are: (a) a mismatch between the number of `trial_start` and `teleport` events — both lists are silently truncated to the shorter one; (b) degenerate trials with `teleport <= trial_start` — skipped; (c) sessions yielding fewer than 2 trials — skipped with a warning; (d) trials with no `reward_zone > 0` sample — previous trial's zone carried forward, zone A if it is the session's first trial; (e) `environment == -1` samples — excluded from the per-trial median, previous trial carried forward if no valid sample; (f) stuck lick sensor — trial's lick zeroed; (g) NaNs in the neural trace — replaced by 0 in the emitted trial matrices and by 0 before OASIS. A `min(e, len-1)` guard protects the reward-window lookup. There are *no* checks that the ophys and behaviour streams have the same number of samples (10 sessions, all m17/m18, have one extra imaging frame; harmless here because all indices come from the behaviour stream) and no assertion that reward timestamps land within one bin of a behavioural sample.

ii.
```python
    if len(trial_start_inds) != len(teleport_inds):
        min_len = min(len(trial_start_inds), len(teleport_inds))
        trial_start_inds = trial_start_inds[:min_len]
        teleport_inds = teleport_inds[:min_len]
```
```python
        if e <= s:
            continue
        ...
        trial_neural[np.isnan(trial_neural)] = 0
```
```python
        if len(neural_trials) < 2:
            print(f"    WARNING: Skipping session with {len(neural_trials)} trials")
            continue
```
```python
        valid_env = env_vals[env_vals >= 0]
        if len(valid_env) > 0:
            env_per_trial[i] = int(np.median(valid_env))
        elif i > 0:
            env_per_trial[i] = env_per_trial[i - 1]
```

iii. Step 5 lists "Handle missing data appropriately" among the requirements and Key Decisions 8/10 cover the zone and lick fallbacks. Step 10 notes a suspected single dropped trial ("Trial total off by 1 (12,216 vs 12,217) - likely one trial with e <= s was skipped") — in fact the dataset contains exactly 12,216 `trial_start` events, so nothing was dropped and the Step-2 count of 12,217 was the error.

## 13-a. What are the most time-consuming steps of the code?

i. The AI instrumented the script with per-session and per-stage timers but never wrote an explicit bottleneck analysis into CONVERSION_NOTES (the Step 6 "Code inefficiencies identified / speedups added" fields were left out). From its own log, the dominant cost is the dF/F + OASIS deconvolution stage: 850 s of the 984 s total (86 %), with the remaining ~130 s split between HDF5 reads of the F/Fneu arrays and per-trial assembly. Writing the 9.84 GB pickle is the other large step (not timed).

ii.
```python
    t_dff = time.time()
    events, dff = compute_dff_and_deconvolve(...)
    print(f"    dF/F + deconv: {time.time() - t_dff:.1f}s")
```
```python
        elapsed_total = time.time() - total_start
        est_remaining = elapsed_total / sessions_done * (len(all_nwb_files) - sessions_done)
        print(f"    [{sessions_done}/{len(all_nwb_files)}] "
              f"Elapsed: {elapsed_total:.0f}s, Est remaining: {est_remaining:.0f}s")
```

iii. The instructions required printing timing information and keeping the full run under ~15 minutes; the running estimate satisfies the "update your estimate as it runs" instruction. Step 9 records "All 152 sessions converted in 984s (~16 min)" with no further optimisation attempted.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Undocumented by the AI. The discretisation helpers are already fully vectorised over a trial (boolean masks rather than per-sample loops), and there is no Viterbi/HMM pass. What remains are Python loops over trials: four separate trial loops inside `compute_dff_and_deconvolve` (masking, baseline, smoothing+deconvolution), and four more at session level (`determine_reward_zone_per_trial`, `determine_reward_per_trial`, the environment loop, the lick-correction loop) plus the main assembly loop — eight or nine passes over the same trial boundaries that could be merged into one, or expressed with `np.add.reduceat` / segment-wise operations. `determine_reward_per_trial` could be replaced by a single `np.searchsorted` of reward times against trial-start times.

ii.
```python
    for i in range(n_trials):   # reward zone
    ...
    for i in range(n_trials):   # autoreward
    ...
    for i in range(n_trials):   # environment
    ...
    for i in range(n_trials):   # lick correction
    ...
    for i in range(n_trials):   # main assembly
```

iii. No rationale is given; the trial loops are inherent to the reference `dff()` implementation the AI copied (baseline and deconvolution are genuinely per-trial), and the session-level loops were left as written because the conversion already met the runtime budget.

## 13-c. What processing does the code repeat multiple times?

i. Undocumented by the AI. The script makes a single pass over each NWB file (no survey pass), so no file is read twice. Within a session, the trial-boundary index arithmetic `s = trial_start_inds[i] - 1; e = teleport_inds[i] - 1` is recomputed in eight different loops, and the trial windows are traversed three times inside `compute_dff_and_deconvolve` alone. `behav_timestamps` is read but the timing information it carries is recomputed independently as `arange(n_tp)/effective_rate`.

ii.
```python
    for start, stop in zip(start_inds, stop_inds):   # mask F/Fneu
    ...
    for start, stop in zip(start_inds, stop_inds):   # neuropil add-back + baseline
    ...
    for start, stop in zip(start_inds, stop_inds):   # smooth + deconvolve
```

iii. No rationale recorded. The three-pass structure inside the dF/F function is inherited verbatim from the paper's `preprocessing.dff`, which the AI chose to reproduce faithfully (Step 5, Key Decision 1).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Undocumented by the AI. Several things are computed and then thrown away:
- **dF/F and deconvolution are run on every ROI, then 58 % of the results are discarded.** Only ~42 % of ROIs pass `iscell`, and every operation in the pipeline is per-cell, so subsetting before the dF/F would have cut the dominant 850 s stage by more than half at identical output.
- The full `dff` array is returned and retained alongside `events` even though nothing but the optional plotting function uses it (the AI does not use it for interneuron detection).
- `trial_num = bts['trial number/data'][:]` and `reward_data = bts['Reward/data'][:]` are loaded and never used.
- The `autoreward` loop over all trials is a no-op — `autoreward` is identically 0 in all 152 sessions.
- The four per-trial-constant variables (environment, trial number, previous outcome, zone location, reward outcome) are materialised as full-length rows, which is what inflates the pickle to 9.84 GB.

ii.
```python
    events, dff = compute_dff_and_deconvolve(
        F_concat, Fneu_concat, trial_start_inds, teleport_inds, imaging_rate, n_planes)
    ...
    cell_indices = np.where(cell_mask)[0]
    events_cells = events[cell_indices, :]   # iscell applied only AFTER the expensive stage
```
```python
        trial_num = bts['trial number/data'][:]     # never used
        reward_data = bts['Reward/data'][:]         # never used
```
```python
    for i in range(n_trials):
        if np.any(autoreward[s:e] > 0):   # autoreward is all zeros in this dataset
            rewarded[i] = 1
```

iii. No rationale is recorded; CONVERSION_NOTES Step 6 does not contain the "Code inefficiencies identified" analysis the template asks for, and Step 12's "Potential Improvements" list is about decoding accuracy rather than wasted computation.
