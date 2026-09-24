# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** discover data by scanning the filesystem. It hard-codes a `SESSIONS_META`
dictionary that it transcribed from the paper repo's `src/reward_relative/sessions_dict.py`
(`GCAMP<N>` → `sub-m<N>`), holding `(session_number, scene, exp_day)` for each of the 11 switch-task
mice. `main()` iterates over the 11 subjects and, for each entry in the table, constructs the NWB
filename `sub-{id}_ses-{exp_day:02d}_behavior+ophys.nwb`, warns and skips if the file is absent, and
otherwise calls `process_session`. Files are read with raw `h5py` (not `pynwb`), pulling
`processing/behavior/BehavioralTimeSeries/*`, `processing/ophys/Fluorescence|Neuropil/plane*`,
`processing/ophys/ImageSegmentation/PlaneSegmentation/{iscell,planeIdx}` and
`acquisition/TwoPhotonSeries/imaging_plane/imaging_rate`. The result covers all 152 NWB files
(11 subjects; 14 days each except m11 which starts at day 3), 12,216 trials — identical coverage to
the reference.

ii.
```python
SESSIONS_META = {
    'm3': [
        (1, 'Env1_LocationC', 1), (2, 'Env1_LocationC', 2),
        (3, 'Env1_LocationC_to_A', 3), (4, 'Env1_LocationA', 4),
        ...
    ],
    ...
}
```
```python
    subjects = sorted(SESSIONS_META.keys())

    for sub_idx, subject_id in enumerate(subjects):
        sub_dir = os.path.join(DATA_DIR, f'sub-{subject_id}')
        if not os.path.exists(sub_dir):
            print(f"WARNING: {sub_dir} not found, skipping")
            continue

        sessions = SESSIONS_META[subject_id]
        for ses_num, scene, exp_day in sessions:
            # NWB session number matches exp_day directly
            nwb_filename = f'sub-{subject_id}_ses-{exp_day:02d}_behavior+ophys.nwb'
            nwb_path = os.path.join(sub_dir, nwb_filename)
            if not os.path.exists(nwb_path):
                print(f"  WARNING: {nwb_path} not found, skipping")
                continue
            result = process_session(nwb_path, subject_id, scene, exp_day)
```
```python
    with h5py.File(nwb_path, 'r') as f:
        bts = f['processing']['behavior']['BehavioralTimeSeries']
        position = bts['position']['data'][:]
        speed = bts['speed']['data'][:]
        lick = bts['lick']['data'][:]
        trial_start_signal = bts['trial_start']['data'][:]
        teleport_signal = bts['teleport']['data'][:]
        timestamps = bts['position']['timestamps'][:]
        reward_data = bts['Reward']['data'][:]
        reward_timestamps = bts['Reward']['timestamps'][:]
```

iii. The AI's stated reason for the table (trajectory steps 20, 27) is that it needs the *scene* name
per session in order to know the reward-zone location and the environment, and `sessions_dict.py` is
the paper's own authoritative record of that: "mapping each subject (sub-m3, sub-m4, sub-m7, etc.) to
its GCAMP entry in sessions_dict, then for each session determining the scene/reward zone locations".
It cross-checked the GCAMP↔sub-m mapping by noting that GCAMP2/6/10 are the fixed-condition mice and
the remaining 11 match "the switch task's n = 11 mice" from the paper. The `exp_day` → `ses-NN`
mapping was initially wrong (it assumed sequential numbering from 01, which silently dropped m11 days
3 and 4 and produced 150 sessions); at step 77 the AI checked the stored `general/session_id` of every
m11 file, found that "the NWB session number actually equals the exp_day itself", and fixed it,
recovering all 152 sessions.

## 1-b. How are the data split into subjects?

i. One subject per mouse ID. The subject list is `sorted(SESSIONS_META.keys())`
(`['m11','m12','m13','m14','m15','m17','m18','m19','m3','m4','m7']`, lexicographic), and
`subject_idx` records the index of the owning subject for each emitted session. This is the same
set and the same order as the reference.

ii.
```python
    subjects = sorted(SESSIONS_META.keys())
    for sub_idx, subject_id in enumerate(subjects):
        ...
            all_subject_idx.append(sub_idx)
...
        'subjects': subjects,
        'subject_idx': np.array(all_subject_idx),
```

iii. The AI verified (trajectory step 30) that every `sub-m<N>` directory's NWB files carry a matching
`general/subject/subject_id`, and that there are exactly 11 such directories, which it matched to the
11 switch-task mice described in the paper.

## 1-c. How are the data split into sessions?

i. One session per NWB file, i.e. one experiment day per mouse. No pooling or alignment of neurons
across days is attempted; each session contributes its own neuron set and its own
`brain_region_idx` entry. 152 sessions are emitted (12 for m11, 14 for each of the other 10).

ii.
```python
        for ses_num, scene, exp_day in sessions:
            nwb_filename = f'sub-{subject_id}_ses-{exp_day:02d}_behavior+ophys.nwb'
            ...
            all_neural.append(result['neural'])
            all_input.append(result['input'])
            all_output.append(result['output'])
            all_subject_idx.append(sub_idx)
            all_brain_region_idx.append(np.zeros(result['n_neurons'], dtype=int))  # all CA1
            session_info.append({'subject': subject_id, 'exp_day': exp_day,
                                 'scene': scene, 'imaging_rate': result['imaging_rate']})
```

iii. The AI confirmed from `general/session_id` that each file's session id equals the experiment day
(step 76–77) and that m11's imaging started on day 3, matching the paper ("sub-m11's imaging starts
from day 3, so it only has 12 sessions"). It never considered cross-day cell registration.

## 1-d. How are the data split into trials?

i. A trial runs from a `trial_start` pulse to the matching `teleport` pulse, `[t0, t1)`. Both signals
are read as all frames where the value is `> 0` (not onset-detected), truncated to the same count,
and paired index-by-index; pairs where the start does not precede the teleport are dropped. The
inter-trial teleport period is excluded from the emitted trials entirely.

ii.
```python
    # Find trial boundaries
    tstart_inds = np.where(trial_start_signal > 0)[0]
    teleport_inds = np.where(teleport_signal > 0)[0]

    n_trials = min(len(tstart_inds), len(teleport_inds))
    tstart_inds = tstart_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]

    # Ensure each trial_start comes before its teleport
    valid = []
    for i in range(n_trials):
        if tstart_inds[i] < teleport_inds[i]:
            valid.append(i)
    tstart_inds = tstart_inds[valid]
    teleport_inds = teleport_inds[valid]
    n_trials = len(tstart_inds)
```
```python
    for i in range(n_trials):
        t0, t1 = tstart_inds[i], teleport_inds[i]
        n_timepoints = t1 - t0
```

iii. The AI inspected the signals directly (step 23–24): "trial_start unique: [0. 1.] / teleport
unique: [0. 1.] / n trials (trial_start): 80 / n teleports: 80", and printed the first ten
start/teleport index pairs to confirm they interleave. It rejected the stored `trial number` series
(range −1 to 79, i.e. −1 during the ITI) in favour of the pulse pair, and states in the module
docstring: "Trials: Defined by trial_start and teleport markers. Teleport periods excluded."

## 1-e. How are trials filtered based on quality controls?

i. Almost no filtering. Two structural guards only: a trial is dropped if it has `< 2` timepoints,
and a whole session is dropped if it yields `< 2` valid trials (both before and after the per-trial
loop). Neither guard fires on this dataset — all 12,216 trials survive (the shortest trial in the
data is 96 frames). There is no behavioural quality criterion (no speed, completion, or lick-based
trial exclusion).

ii.
```python
    if n_trials < 2:
        print(f"    Skipping: only {n_trials} valid trials")
        return None
...
        if n_timepoints < 2:
            continue
...
    if len(neural_trials) < 2:
        print(f"    Skipping: fewer than 2 valid trials after processing")
        return None
```

iii. The `< 2 trials` guard is explicitly aimed at the format requirement ("There needs to be at
least two trials within each session in order to evaluate the decoder performance"). The AI gives no
justification for omitting behavioural trial QC; its plan (step 20) records the related judgement
that the paper's speed threshold is only for place-cell analysis: "I should not apply a speed
threshold for the decoder itself, only for place cell analysis."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces: `processing/ophys/Fluorescence/plane<k>/data` (F) and
`processing/ophys/Neuropil/plane<k>/data` (Fneu), transposed to (n_rois, n_frames) and concatenated
across planes in plane order. The NWB `Deconvolved` field is deliberately **not** used.

ii.
```python
        fluor_group = f['processing']['ophys']['Fluorescence']
        neuro_group = f['processing']['ophys']['Neuropil']
        F_list = []
        Fneu_list = []
        for p in sorted(unique_planes if n_planes > 1 else [0]):
            plane_key = f'plane{int(p)}'
            F_list.append(fluor_group[plane_key]['data'][:].T)  # (n_rois_plane, n_frames)
            Fneu_list.append(neuro_group[plane_key]['data'][:].T)
        F = np.concatenate(F_list, axis=0)
        Fneu = np.concatenate(Fneu_list, axis=0)
```

iii. From the trajectory (step 20): "Now I realize the paper's custom dF/F pipeline with maximin
baseline may not match what's actually in the NWB file — the deconvolved trace there comes straight
from suite2p, not the paper's own neuropil-subtraction-plus-OASIS pipeline with tau=0.7", and
(step 27) "I'm weighing whether to use the suite2p-deconvolved data already in the NWB file or
recompute deconvolution myself, since the paper's custom pipeline differs from suite2p's standard
processing — to match the paper accurately, I think I need to recompute it from scratch." It then
checked that `suite2p.extraction.dcnv` was installed before committing to this.

## 2-b. How is the `neural` data processed?

i. A reimplementation of the paper's `preprocessing.dff`. Per session: build F and Fneu arrays that
are NaN everywhere outside `[trial_start, teleport)` (so the teleport/ITI never enters any baseline
window); subtract `0.7 * Fneu`; per trial, add that trial's mean neuropil back
(`+ 0.7 * nanmean(Fneu_trial)`) so the ratio is a true dF/F; per trial compute a maximin baseline —
NaN-aware Gaussian smoothing with sigma 15 frames, then a 300-frame `minimum_filter1d` followed by a
300-frame `maximum_filter1d` (the Methods' ~20 s window at 15.5 Hz); form
`dF/F = (F_corr − baseline)/|baseline|`; per trial smooth dF/F with a 2-frame Gaussian and deconvolve
with suite2p's OASIS at `tau = 0.7` and `frame_rate / n_planes`. Cells from multiple planes are
pooled by concatenation before this. The emitted `neural` array is the deconvolved events
(`spks`), float64, shape (n_neurons, n_timepoints).

ii.
```python
NEU_COEF = 0.7
BASELINE_WINDOW = 300  # frames (~20 s at 15.5 Hz)
BASELINE_SMOOTH_SIGMA = 15
DFF_SMOOTH_SIGMA = 2
TAU = 0.7  # calcium decay time constant for OASIS
```
```python
    f_ = np.full((n_cells, n_frames), np.nan)
    for start, stop in zip(trial_starts, trial_ends):
        f_[:, start:stop] = F[:, start:stop]
    f_neu_ = np.full((n_cells, n_frames), np.nan)
    for start, stop in zip(trial_starts, trial_ends):
        f_neu_[:, start:stop] = Fneu[:, start:stop]

    # Neuropil subtraction
    nanmask = ~np.isnan(f_[0, :])
    f_[:, nanmask] = f_[:, nanmask] - NEU_COEF * f_neu_[:, nanmask]

    flow = np.full((n_cells, n_frames), np.nan)
    for start, stop in zip(trial_starts, trial_ends):
        f_[:, start:stop] = f_[:, start:stop] + NEU_COEF * np.nanmean(
            f_neu_[:, start:stop], axis=1, keepdims=True)
        flow[:, start:stop] = nansmooth(f_[:, start:stop], BASELINE_SMOOTH_SIGMA, axis=1)
        flow[:, start:stop] = minimum_filter1d(flow[:, start:stop], BASELINE_WINDOW, axis=-1)
        flow[:, start:stop] = maximum_filter1d(flow[:, start:stop], BASELINE_WINDOW, axis=-1)

    dff = np.full((n_cells, n_frames), np.nan)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])

    spks = np.full((n_cells, n_frames), np.nan)
    for start, stop in zip(trial_starts, trial_ends):
        dff[:, start:stop] = nansmooth(dff[:, start:stop], DFF_SMOOTH_SIGMA, axis=1)
        spks[:, start:stop] = dcnv.oasis(dff[:, start:stop], 2000, TAU, frame_rate / n_planes)
```

iii. The module docstring states the intent: "Neural data: Neuropil subtraction (coef=0.7), maximin
baseline per trial, dF/F, 2-sample Gaussian smoothing, OASIS deconvolution (tau=0.7)." The AI read
`src/reward_relative/preprocessing.py` directly (step 18) and its plan (step 27) lists exactly these
steps: "neuropil subtraction, maximin baseline computation, dF/F normalization, Gaussian smoothing,
OASIS deconvolution, and filtering out putative interneurons based on speed correlation." The
`frame_rate / n_planes` argument is justified by the two-plane sessions (m17, m18) whose stored
scanner rate is 31 Hz for a 15.5 Hz per-plane rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, matching the paper. (1) ROIs are restricted to suite2p's curated cells,
`iscell[:,0] == 1`, applied to the concatenated across-plane ROI array before dF/F. (2) Putative
interneurons are dropped: any cell whose within-trial dF/F has Pearson `r > 0.5` with the animal's
running speed. 138,276 neurons survive in total (154 to 2,323 per session, mean 910).

ii.
```python
INTERNEURON_SPEED_CORR_THR = 0.5
...
    # Filter cells by iscell
    cell_mask = iscell[:, 0] == 1
    cell_indices = np.where(cell_mask)[0]
    F_cells = F[cell_indices, :]
    Fneu_cells = Fneu[cell_indices, :]

    dff, spks = compute_dff_and_deconvolve(
        F_cells, Fneu_cells, tstart_inds, teleport_inds, imaging_rate, n_planes)

    # Filter putative interneurons (speed correlation > 0.5 with dF/F)
    valid_mask = ~np.isnan(dff[0, :])
    speed_valid = speed[valid_mask]
    keep_neuron = np.ones(dff.shape[0], dtype=bool)
    for n_idx in range(dff.shape[0]):
        dff_valid = dff[n_idx, valid_mask]
        if np.std(dff_valid) > 0 and np.std(speed_valid) > 0:
            r, _ = pearsonr(dff_valid, speed_valid)
            if r > INTERNEURON_SPEED_CORR_THR:
                keep_neuron[n_idx] = False
    spks = spks[keep_neuron, :]
```

iii. Docstring: "Neuron filtering: Suite2p iscell, then exclude putative interneurons (Pearson
r > 0.5 between dF/F and running speed)." The AI verified the ROI ordering needed to apply a single
`iscell` array to plane-concatenated traces (step 52): it checked that `planeIdx` is sorted with a
single transition, "plane transitions at: [998] ... So iscell indices 0..N0−1 are plane0, N0..N0+N1−1
are plane1. This matches the concatenation order in F."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Nothing beyond trial segmentation. The neural array is sampled on the same frame grid as the
behaviour, so the requested alignment (trial start) is achieved by slicing `spks[:, t0:t1]` where
`t0` is the `trial_start` frame; `time 0` of every trial is its first imaging frame. Metadata records
`temporal_alignment_event = 'start of trial (first imaging frame after teleport)'`, `off_start = 0.0`,
`off_end = None`. No pre-trial window is kept.

ii.
```python
        t0, t1 = tstart_inds[i], teleport_inds[i]
        trial_neural = spks[:, t0:t1]
        trial_neural = np.nan_to_num(trial_neural, nan=0.0)
...
            'temporal_alignment_event': 'start of trial (first imaging frame after teleport)',
            'off_start': 0.0,
            'off_end': None,
```

iii. Docstring: "Temporal alignment: Start of each trial (time 0 = first frame of trial)." The AI
treated the NWB behaviour and ophys streams as already sample-aligned (it only equalises their
lengths, see 12), so no resampling or offset was needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: the data are kept at the native per-plane imaging frame rate,
~15.51 Hz → 64.48 ms per bin, which is the same for single-plane sessions and (after the per-plane
correction) for the two-plane sessions. The stored `metadata['time_bin_size']` is the **median over
sessions** of `1000 / imaging_rate`, which evaluates to 64.48362720403023 ms — identical to the
reference. Note the per-session quantity being medianed is not divided by `n_planes`, so for the 28
two-plane sessions it is 32.24 ms; only the fact that 124 of 152 sessions are single-plane makes the
median come out right. The `n_planes` division *is* applied correctly where it matters numerically,
in the OASIS call.

ii.
```python
        imaging_rate = f['acquisition']['TwoPhotonSeries']['imaging_plane']['imaging_rate'][()]
...
        spks[:, start:stop] = dcnv.oasis(dff[:, start:stop], 2000, TAU, frame_rate / n_planes)
...
    time_bin_sizes = []
    for info in session_info:
        time_bin_sizes.append(1000.0 / info['imaging_rate'])
    median_time_bin = np.median(time_bin_sizes) if time_bin_sizes else 64.5
...
            'time_bin_size': median_time_bin,
```

iii. The AI's plan (step 20) states the choice: "Temporal alignment: Align to start of trial, using
the native imaging frame rate of roughly 15.5 Hz (about 64.5 ms per bin)". Keeping the native rate
avoids any interpolation between the neural and behavioural streams, which already share a grid.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `position` behavioural time series' `timestamps` array (all behaviour series in these files
share one timestamp vector).

ii.
```python
        timestamps = bts['position']['timestamps'][:]
...
        time_from_start = (timestamps[t0:t1] - timestamps[t0])
```

iii. The AI read the timestamps alongside the behaviour in its first inspection (step 23), observing
`timestamps range: 0.0 2186.768765743073` for a session, and used them as the common clock. It did
not assert that the other series' timestamps are identical (the reference does).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the trial's first timestamp, giving seconds since trial start; stored as input row 0,
the only genuinely time-varying input.

ii.
```python
        time_from_start = (timestamps[t0:t1] - timestamps[t0])
        trial_input = np.array([
            time_from_start,
            np.full(n_timepoints, env_type, dtype=float),
            np.full(n_timepoints, trial_number, dtype=float),
            np.full(n_timepoints, prev_outcome, dtype=float),
        ])
```

iii. Direct reading of the instruction "Time from start of trial in seconds (continuous,
time-varying)"; no further rationale given.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the timestamps are indexed with the same `[t0:t1]` slice as the neural data.
Before any slicing, all behaviour arrays and both fluorescence arrays are truncated to a common
`n_frames = min(len(position), F.shape[1])`, so the two streams are guaranteed to be the same length
and index-aligned.

ii.
```python
    # Ensure behavioral and neural data have same length
    n_frames = min(len(position), F.shape[1])
    position = position[:n_frames]
    speed = speed[:n_frames]
    lick = lick[:n_frames]
    trial_start_signal = trial_start_signal[:n_frames]
    teleport_signal = teleport_signal[:n_frames]
    timestamps = timestamps[:n_frames]
    F = F[:, :n_frames]
    Fneu = Fneu[:, :n_frames]
```

iii. The AI hit the length mismatch as a crash during the full run and diagnosed it at step 47: "The
behavioral data and neural data have slightly different lengths for some sessions. This is likely
because the multi-plane imaging interleaves frames, leading to potential off-by-one issues. Let me
fix this by trimming to the minimum length."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Not from the data at all: from the hard-coded `scene` string of `SESSIONS_META`. `Env1 → 0`,
`Env2 → 1`; on the cross-environment day (scene contains `_to_Env`) the environment switches at
trial index 30. The NWB `environment` behavioural time series is loaded by neither name nor path.

ii.
```python
def get_environment_per_trial(scene, trial_idx, change_trial=30):
    """Return environment for a specific trial (handles cross-env switches)."""
    if '_to_Env' in scene:
        parts = scene.split('_to_Env')
        if trial_idx < change_trial:
            return 0 if 'Env1' in parts[0] else 1
        else:
            return 0 if parts[1].startswith('1') else 1
    elif scene.startswith('Env2'):
        return 1
    else:
        return 0
...
        env_type = get_environment_per_trial(scene, i)
```

iii. The AI's plan (step 27) treats the scene string as the source of truth for both environment and
reward-zone identity: "for each session determining the scene/reward zone locations". It had seen in
its own data inspection (step 23) that `environment unique: [-1. 0.]` for a single-environment
session — i.e. the raw variable is −1 during the ITI — and in the switch session that "the active
reward zone shifts from zone C to zone A around trial 30". It generalised the trial-30 switch point
from that one session to all sessions.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to 0/1 as above and broadcast as a constant across all timepoints of the
trial (input row 1). No smoothing, no use of the raw `environment` variable.

ii.
```python
        trial_input = np.array([
            time_from_start,
            np.full(n_timepoints, env_type, dtype=float),
            ...
        ])
```

iii. As 4-a; the instruction asks for a binary ENV1/ENV2 indicator, which the scene string supplies
directly.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Nothing in the raw data — it is the loop counter, i.e. the 0-based index of the trial within the
session, where the trial boundaries themselves come from `trial_start`/`teleport`. The stored
`trial number` behavioural series is not used.

ii.
```python
    for i in range(n_trials):
        ...
        trial_number = float(i)
```

iii. The AI observed (step 23) that the stored series runs `trial_num range: -1.0 79.0`, i.e. it is
−1 between trials, and consistently derived everything trial-indexed from the pulse-pair segmentation
instead.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond a cast to float and broadcasting the constant across the trial's timepoints
(input row 2). Values run 0 to 99 (some sessions have 90 or 100 trials).

ii.
```python
            np.full(n_timepoints, trial_number, dtype=float),
```

iii. Direct reading of "Trial number (continuous, per trial)".

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` behavioural time series, specifically its own `timestamps` array (its `data` array,
all values 0.004 mL, is loaded but unused). A per-trial boolean `trial_rewarded` is built by testing
whether any reward timestamp falls in the closed time interval spanned by the trial.

ii.
```python
        reward_data = bts['Reward']['data'][:]
        reward_timestamps = bts['Reward']['timestamps'][:]
...
    trial_rewarded = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        t_start_time = timestamps[tstart_inds[i]]
        t_end_time = timestamps[teleport_inds[i]]
        reward_in_trial = (reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)
        if np.any(reward_in_trial):
            trial_rewarded[i] = 1
```

iii. The AI saw at step 23 that `Reward` is a sparse event series with its own clock
(`reward_data shape: (71,) unique: [0.004]` for an 80-trial session) rather than a per-frame signal,
so it matched reward events to trials by time interval rather than by frame index.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Input row 3 is `trial_rewarded[i-1]`, broadcast constant across the trial. For the first trial of
each session, where there is no previous trial, the AI sets the value to **1 (rewarded)** rather
than 0. This affects 152 of 12,216 trials.

ii.
```python
        # Previous trial outcome (0=omission, 1=rewarded)
        if i == 0:
            prev_outcome = 1.0  # first trial: no previous, default to rewarded
        else:
            prev_outcome = float(trial_rewarded[i - 1])
```

iii. The only justification is the inline comment: "first trial: no previous, default to rewarded".
No further reasoning appears in the trajectory.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` behavioural time series, combined with the reward-zone boundaries for the trial.
The boundaries come from a hard-coded `REWARD_ZONES` dict transcribed from the paper's
`behavior.py::reward_zone_dict` (`X/Y/Z → A/B/C`), and the *which zone* question is answered from the
`scene` string plus the trial-30 switch rule (see 10-a), not from the NWB `reward_zone` series.

ii.
```python
# Reward zone positions from behavior.py: X=[80,130], Y=[200,250], Z=[320,370]
# Labels: A->X, B->Y, C->Z
REWARD_ZONES = {
    'A': (80, 130),
    'B': (200, 250),
    'C': (320, 370),
}
...
        rz_label = get_reward_zone_for_trial(scene, i, n_trials)
        rz_start, rz_end = REWARD_ZONES[rz_label]
        pos_trial = position[t0:t1]
        dist_to_rz = discretize_distance_to_reward(pos_trial, rz_start, rz_end)
```

iii. The AI read `behavior.py` (step 19) and copied the `X/Y/Z` ranges. It explicitly investigated
and then rejected the NWB `reward_zone` series as a source: at step 25 it hypothesised that
`reward_zone` (values 0–6) might already be the requested discretisation, tested it against position
("a position of 319.5 cm ... that should fall in bin 2 but instead shows bin 1, which doesn't fit"),
printed the series frame by frame for a trial (step 26), and concluded at step 27: "The reward_zone
data is sparse — it's only non-zero at certain frames ... So reward_zone in the NWB is NOT the
distance discretization I need. It's the original rzone signal from the VR system. I need to compute
distance to reward zone myself."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest edge of the zone: negative before the zone (`position − rz_start`),
exactly 0 anywhere inside it, positive after (`position − rz_end`). Computed vectorised over the
trial's positions, then discretised (7-c).

ii.
```python
    dist = np.where(
        position < rz_start, position - rz_start,
        np.where(position > rz_end, position - rz_end, 0.0)
    )
```

iii. Follows the instruction "Distance to any location in the reward zone" — the nearest point of the
zone, hence 0 throughout the zone — with the zone edges taken from the paper's own dictionary.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned by explicit boolean masks rather than `np.digitize`:
`<-50 → 0`, `[-50,-10) → 1`, `[-10,0) → 2`, `==0 → 3`, `(0,10] → 4`, `(10,50] → 5`, `>50 → 6`.
The resulting class fractions (0.253/0.102/0.074/0.237/0.021/0.072/0.242) match the reference to
within 0.0007.

ii.
```python
    bins = np.zeros(len(dist), dtype=int)
    bins[dist < -50] = 0
    bins[(dist >= -50) & (dist < -10)] = 1
    bins[(dist >= -10) & (dist < 0)] = 2
    bins[dist == 0] = 3  # inside zone
    bins[(dist > 0) & (dist <= 10)] = 4
    bins[(dist > 10) & (dist <= 50)] = 5
    bins[dist > 50] = 6
```

iii. The docstring reproduces the instruction's bin table verbatim; the `dist == 0` test is used to
isolate "inside the zone" because the distance function emits exact 0.0 there.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No alignment step: `position` was truncated to the same `n_frames` as the neural data and is
sliced with the same `[t0:t1]` indices, so output row 0 is sample-for-sample matched to the neural
matrix.

ii.
```python
        t0, t1 = tstart_inds[i], teleport_inds[i]
        trial_neural = spks[:, t0:t1]
        ...
        pos_trial = position[t0:t1]
        dist_to_rz = discretize_distance_to_reward(pos_trial, rz_start, rz_end)
```

iii. Same reasoning as 3-c: behaviour and ophys share the NWB frame grid once truncated to a common
length.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural time series (VR corridor position in cm).

ii.
```python
        position = bts['position']['data'][:]
...
        pos_trial = position[t0:t1]
        abs_position = discretize_position(pos_trial)
```

iii. The AI checked the raw range at step 23 (`position range: -500.0 450.3810162629735`, the −500
belonging to the teleport period which is excluded from trials) and used the variable directly.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing and discretising — the raw cm value is used as-is.

ii.
```python
def discretize_position(position):
    """Discretize position into 5 equal bins spanning 450 cm."""
    bins = np.clip((position / 90).astype(int), 0, 4)
    return bins
```

iii. Track length is 450 cm per the Methods (`TRACK_LENGTH = 450` is defined as a constant, though
the literal 90 is used in the function).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins via integer division, clipped to [0,4]. The clip makes the first and last
bins open-ended, so the handful of samples marginally outside 0–450 cm land in bins 0 and 4 rather
than creating extra classes. The resulting class fractions
(0.2107/0.1777/0.2310/0.2265/0.1540) are **identical** to the reference's to 16 significant figures.

ii.
```python
    bins = np.clip((position / 90).astype(int), 0, 4)
```

iii. Directly implements "Discretized into 5 equal-sized bins spanning the 450 cm track"; 450/5 = 90.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[t0:t1]` slice as the neural data, on arrays already truncated to a common length. No
separate alignment.

ii.
```python
        pos_trial = position[t0:t1]
        abs_position = discretize_position(pos_trial)
```

iii. As 3-c / 7-d.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural time series (per-frame lick counts, integer values 0–6 as the AI observed).

ii.
```python
        lick = bts['lick']['data'][:]
...
        lick_trial = lick_corrected[t0:t1]
        lick_binary = np.where(np.isnan(lick_trial), 0, (lick_trial > 0).astype(int))
```

iii. Step 23 inspection: `lick unique: [0. 1. 2. 3. 4. 5. 6.]`, i.e. a count per imaging frame rather
than a binary flag.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps. First a lick-sensor-error correction copied in spirit from the paper's
`behavior.py::correct_lick_sensor_error`: for each trial, if more than 30% of frames have a lick
count `> 2`, the whole trial's lick trace is set to NaN. Then binarisation: `lick > 0 → 1`, with NaN
mapped to **0**. So the 81 affected trials (0.66% of trials) are emitted as containing no licks at
all. Overall lick rate is 0.2188 versus the reference's 0.2304.

ii.
```python
LICK_SENSOR_ERROR_THR = 0.5  # paper uses 0.3 (>30% of frames with cumulative lick > 2)
...
    # Correct lick sensor errors per trial
    lick_corrected = np.copy(lick)
    for i in range(n_trials):
        t0, t1 = tstart_inds[i], teleport_inds[i]
        trial_licks = lick_corrected[t0:t1]
        n_frames_trial = len(trial_licks)
        if n_frames_trial > 0:
            # Paper: >30% of frames with cumulative lick count > 2
            frac_high = np.sum(trial_licks > 2) / n_frames_trial
            if frac_high > 0.3:
                lick_corrected[t0:t1] = np.nan
...
        lick_binary = np.where(np.isnan(lick_trial), 0, (lick_trial > 0).astype(int))
```

iii. The correction is attributed to the paper: the AI listed `correct_lick_sensor_error()` among the
`behavior.py` functions it read (step 19 / session summary) and the inline comment cites the paper's
30% / >2-licks criterion. Binarisation is justified by the instruction "Lick, time-varying. 0 = no,
1 = yes". No justification is given for mapping the flagged trials to 0 rather than excluding them.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[t0:t1]` slice on a length-matched array; no separate alignment.

ii.
```python
        lick_trial = lick_corrected[t0:t1]
```

iii. As 3-c.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. No raw-data variable. The zone label (A/B/C) is parsed out of the hard-coded `scene` string for
the session, with the switch sessions (`..._to_...`) changing label at trial index 30.
`get_reward_zone_for_trial` handles both within-environment switches (`Env1_LocationA_to_B`) and the
cross-environment day (`Env1_C_to_Env2_B`).

ii.
```python
def get_reward_zone_for_trial(scene, trial_idx, n_trials, change_trial=30):
    if 'LocationA' in scene and '_to_' not in scene and ...:
        return 'A'
    ...
    if '_to_Env' in scene:
        parts = scene.split('_to_')
        before_zone = parts[0].split('_')[-1]
        after_zone = parts[1].split('_')[-1]
    elif '_to_' in scene:
        parts = scene.split('_to_')
        before_zone = parts[0].split('Location')[-1]
        after_zone = parts[1]
    ...
    if trial_idx < change_trial:
        return before_zone
    else:
        return after_zone
```

iii. As 7-a: having established that the NWB `reward_zone` series is the sparse VR in-zone signal and
not a zone identity, the AI fell back on the paper's own session metadata. Its evidence for
`change_trial = 30` is the single observation at step 25 that in one switch session "the active
reward zone shifts from zone C to zone A around trial 30, matching the position ranges where nonzero
values appear."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The label is mapped `A→0, B→1, C→2` and broadcast as a constant across the trial's timepoints
(output row 4). Class fractions come out 0.329/0.337/0.335.

ii.
```python
def rz_label_to_idx(label):
    """Map reward zone label to index: A=0, B=1, C=2."""
    return {'A': 0, 'B': 1, 'C': 2}[label]
...
        rz_location = rz_label_to_idx(rz_label)
...
            np.full(n_timepoints, rz_location, dtype=int),
```

iii. The instruction specifies "Reward zone location, per-trial. 0 = A, 1 = B, 2 = C"; the A/B/C ↔
X/Y/Z mapping is taken from `behavior.py` (comment at `REWARD_ZONES`).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` time series' timestamps, via the same `trial_rewarded` array used for the
previous-trial input (6-a).

ii.
```python
        reward_timestamps = bts['Reward']['timestamps'][:]
...
        reward_in_trial = (reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)
        if np.any(reward_in_trial):
            trial_rewarded[i] = 1
```

iii. As 6-a: reward is a sparse, separately-clocked event series, so trial attribution is done by
time interval.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary per trial (1 if any reward event fell inside the trial's time window), broadcast across all
timepoints as output row 5. Resulting fractions 0.157 / 0.843 — identical to the reference.

ii.
```python
        reward_outcome = trial_rewarded[i]
...
            np.full(n_timepoints, reward_outcome, dtype=int),
```

iii. The instruction specifies "Reward outcome, per-trial. 0 = no, 1 = yes"; the format allows a
per-trial constant, and the AI broadcast it to keep every output row time-varying and the same shape.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, mostly by silent repair:
- **Neural/behaviour length mismatch**: every behaviour array and both fluorescence arrays are
  truncated to `min(len(position), F.shape[1])`. No warning is printed (the reference warns).
- **Unequal numbers of trial_start and teleport pulses**: both lists are truncated to the shorter,
  and any pair whose start does not precede its teleport is dropped.
- **Missing NWB file**: a warning is printed and the session is skipped, leaving the dataset short.
  This path actually fired during development (m11 days 3–4 were silently dropped, 150 instead of
  152 sessions) and was only caught because the AI went back and checked m11 by hand.
- **NaN in the deconvolved events inside a trial**: replaced by 0 with `np.nan_to_num`.
- **Lick sensor errors**: whole-trial NaN, then mapped to 0 (see 9-b).
- **Degenerate cells/sessions**: cells with zero dF/F variance are skipped by the interneuron
  correlation test; sessions with fewer than 2 usable trials or 0 neurons are dropped.

ii.
```python
    n_frames = min(len(position), F.shape[1])
    ...
    n_trials = min(len(tstart_inds), len(teleport_inds))
    ...
    if not os.path.exists(nwb_path):
        print(f"  WARNING: {nwb_path} not found, skipping")
        continue
    ...
        trial_neural = np.nan_to_num(trial_neural, nan=0.0)
    ...
        if np.std(dff_valid) > 0 and np.std(speed_valid) > 0:
            r, _ = pearsonr(dff_valid, speed_valid)
    ...
    if n_neurons < 1:
        print(f"    Skipping: no neurons after filtering")
        return None
```

iii. These are all defensive fixes added reactively when the full run crashed: the multi-plane
loading fix at step 44, the length truncation at step 50 ("The behavioral data and neural data have
slightly different lengths. Let me fix by trimming to the minimum length"), the int-dtype output fix
at step 66, and the session-numbering fix at step 81.

## 13-a. What are the most time-consuming steps of the code?

i. In rough order:
1. **Reading F and Fneu out of each NWB file** — two full (n_frames × n_rois) float64 arrays per
   plane, up to ~34,000 × 2,900, read with `[:]` for *all* ROIs before `iscell` subsetting. Pure I/O
   plus peak memory.
2. **The maximin baseline** — per trial, a sigma-15 NaN-aware Gaussian (which itself runs three
   `gaussian_filter1d` passes inside `nansmooth`) plus a 300-sample min and a 300-sample max filter
   over all curated cells.
3. **OASIS deconvolution**, called once per trial (~80–100 calls per session × 152 sessions).
4. **The per-cell interneuron loop** — a Python-level `pearsonr` per cell, ~900 cells × 152 sessions
   ≈ 138,000 scipy calls.
5. **Pickling the result** — the output file is 19.5 GB.

ii.
```python
            F_list.append(fluor_group[plane_key]['data'][:].T)
...
        flow[:, start:stop] = nansmooth(f_[:, start:stop], BASELINE_SMOOTH_SIGMA, axis=1)
        flow[:, start:stop] = minimum_filter1d(flow[:, start:stop], BASELINE_WINDOW, axis=-1)
        flow[:, start:stop] = maximum_filter1d(flow[:, start:stop], BASELINE_WINDOW, axis=-1)
...
    for n_idx in range(dff.shape[0]):
        r, _ = pearsonr(dff[n_idx, valid_mask], speed_valid)
...
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
```

iii. Not discussed by the AI. It did run the conversion in the background with a 600 s timeout each
time, implying it expected it to be slow, but never profiled or optimised.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i.
- The **interneuron correlation loop** is the clearest case: a single centred-and-normalised matrix
  product against speed computes all per-cell correlations at once (this is exactly what the
  reference does not do either, but the reference at least uses `np.corrcoef`).
- The **four separate `for start, stop in zip(...)` loops** in `compute_dff_and_deconvolve` walk the
  same trial list four times (mask F, mask Fneu, baseline, smooth+deconvolve). They could be one
  loop, and the two masking loops could be replaced by a single boolean index built from the trial
  boundaries.
- The **`valid` trial-ordering loop** is a one-line comparison of two arrays
  (`tstart_inds < teleport_inds`).
- The **`trial_rewarded` loop** is a `np.searchsorted` of reward times into trial-start times.
- The **lick-correction loop** and the discretisation calls inside the main trial loop could operate
  on whole-session arrays before splitting.

ii.
```python
    for n_idx in range(dff.shape[0]):
        dff_valid = dff[n_idx, valid_mask]
        if np.std(dff_valid) > 0 and np.std(speed_valid) > 0:
            r, _ = pearsonr(dff_valid, speed_valid)
```
```python
    valid = []
    for i in range(n_trials):
        if tstart_inds[i] < teleport_inds[i]:
            valid.append(i)
```
```python
    for i in range(n_trials):
        t_start_time = timestamps[tstart_inds[i]]
        t_end_time = timestamps[teleport_inds[i]]
        reward_in_trial = (reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)
```

iii. Not discussed. The per-trial structure of the dF/F loops is inherited from the paper's own
`preprocessing.dff`, so keeping it aids comparability even where it costs speed.

## 13-c. What processing does the code repeat multiple times?

i.
- Each NWB file is opened and read exactly **once** — there is no separate survey pass (the reference
  makes two passes over every file).
- Within a session, the trial boundary list is walked six separate times (two masking loops, the
  baseline loop, the smooth/deconvolve loop, the `trial_rewarded` loop, the lick-correction loop, and
  the main emit loop), each re-slicing the same `[t0:t1]` windows.
- `nansmooth` recomputes a Gaussian smoothing of the validity mask on every call even though that
  mask is constant within a trial slice.
- F and Fneu are read in full (all ROIs) and then immediately subset to `iscell`, so roughly
  two-thirds of the data read from disk is discarded.
- `np.nanmean(f_neu_[:, start:stop])` is recomputed inside the baseline loop rather than reused.

ii.
```python
    for start, stop in zip(trial_starts, trial_ends):
        f_[:, start:stop] = F[:, start:stop]
    f_neu_ = np.full((n_cells, n_frames), np.nan)
    for start, stop in zip(trial_starts, trial_ends):
        f_neu_[:, start:stop] = Fneu[:, start:stop]
    ...
    for start, stop in zip(trial_starts, trial_ends):
        f_[:, start:stop] = f_[:, start:stop] + NEU_COEF * np.nanmean(...)
        flow[:, start:stop] = nansmooth(...)
    ...
    for start, stop in zip(trial_starts, trial_ends):
        dff[:, start:stop] = nansmooth(...)
        spks[:, start:stop] = dcnv.oasis(...)
```
```python
        F_list.append(fluor_group[plane_key]['data'][:].T)   # all ROIs
        ...
        F_cells = F[cell_indices, :]                          # then discard ~2/3
```

iii. Not discussed by the AI.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **`dff` is filtered and then never used**: `dff = dff[keep_neuron, :]` at line 443 produces a large
  array that is dead from that point on.
- **Dead constants and functions**: `TRACK_LENGTH` is defined but never referenced (the literal 90 is
  used instead); `LICK_SENSOR_ERROR_THR = 0.5` is defined, its own comment says "paper uses 0.3", and
  the code hard-codes 0.3 — the constant is never read; `get_environment(scene)` is defined and never
  called (superseded by `get_environment_per_trial`); `ses_num` is unpacked from `SESSIONS_META` and
  never used.
- **`reward_data`** (the reward volumes) is read from disk and never used; only the timestamps are.
- **`frame_time = np.median(np.diff(timestamps))`** is computed and returned in the session result
  dict but never consulted — `time_bin_size` is derived from `imaging_rate` instead.
- **Full-ROI reads**: all non-cell ROIs are loaded, transposed and concatenated before being dropped.
- **Storage**: `neural` is stored as float64, doubling the 19.5 GB pickle for a signal that carries no
  meaningful precision beyond float32; and the four per-trial-constant input rows and two
  per-trial-constant output rows are materialised at full time resolution, which the target format
  explicitly does not require (`shape (n_output,)` is allowed).

ii.
```python
TRACK_LENGTH = 450  # cm   <- never used
LICK_SENSOR_ERROR_THR = 0.5  # paper uses 0.3 ...   <- never used; 0.3 hard-coded below
def get_environment(scene):   # <- never called
...
        reward_data = bts['Reward']['data'][:]   # <- never used
    frame_time = np.median(np.diff(timestamps))  # <- returned, never used
...
    spks = spks[keep_neuron, :]
    dff = dff[keep_neuron, :]    # <- dff is dead after this line
```

iii. Not discussed by the AI; these are leftovers from the iterative development visible in the
trajectory (the `get_environment`/`get_environment_per_trial` pair, and the threshold constant whose
comment contradicts it, both date from the first `Write` of the file at step 33 and were never
cleaned up).
