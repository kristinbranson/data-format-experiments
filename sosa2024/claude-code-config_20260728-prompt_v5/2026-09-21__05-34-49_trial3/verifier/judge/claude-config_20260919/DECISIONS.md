# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs `/app/data` for every subdirectory whose name begins with `sub-` and then every
`*.nwb` file inside each of those directories, sorted by subject then filename. It does **not** use
`pynwb`; it opens each NWB file directly as HDF5 with `h5py` and reads the specific datasets it
needs (`processing/ophys/Fluorescence/plane*`, `processing/ophys/Neuropil/plane*`,
`processing/ophys/ImageSegmentation/PlaneSegmentation/{iscell,planeIdx}`, and the
`processing/behavior/BehavioralTimeSeries/*` time series). All 152 files (11 subjects) are found
and processed; each file is loaded exactly once, in a single pass. Each session is wrapped in a
`try/except` so a failing file is reported and skipped rather than aborting the run.

ii.
```python
def collect_nwb_files(data_dir, sample=False):
    """Collect all NWB files, sorted by subject then session."""
    files = []
    for subj_dir in sorted(os.listdir(data_dir)):
        if not subj_dir.startswith('sub-'):
            continue
        subj_path = os.path.join(data_dir, subj_dir)
        for nwb_file in sorted(os.listdir(subj_path)):
            if nwb_file.endswith('.nwb'):
                files.append(os.path.join(subj_path, nwb_file))
    ...
    return files
```

```python
def load_nwb_session(filepath):
    """Load all needed data from an NWB file."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        ophys = f['processing']['ophys']
        bts = f['processing']['behavior']['BehavioralTimeSeries']
        seg = ophys['ImageSegmentation']['PlaneSegmentation']
        iscell = seg['iscell'][:]
        plane_idx = seg['planeIdx'][:]
        fluor_grp = ophys['Fluorescence']
        plane_keys = sorted(fluor_grp.keys())
        ...
```

```python
    for i, filepath in enumerate(nwb_files):
        ...
        try:
            result = process_session(filepath, show_processing=args.show_processing,
                                    session_label=label)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR processing {label}: {e}")
```

iii. From CONVERSION_NOTES.md Step 2: "`/app/data/sub-{id}/sub-{id}_ses-{nn}_behavior+ophys.nwb`,
NWB format v2.8.0", and Step 9 verifies "Sessions | 152 (14/mouse, 12 for m11) | 152 | YES". The AI
chose raw `h5py` over `pynwb` for speed (only the needed arrays are read, one pass per file, 546 s
total for the full conversion). No justification is given for not using `pynwb`, but the notes show
the resulting session/subject/trial counts were cross-checked against the paper.

## 1-b. How are the data split into subjects?

i. Subjects are read out of each NWB file's `general/subject/subject_id` field rather than from the
directory name. The unique set is sorted to form `data['subjects']`, and each session's
`subject_idx` is the index of its own `subject_id` in that sorted list. This yields 11 subjects
(m11, m12, m13, m14, m15, m17, m18, m19, m3, m4, m7).

ii.
```python
        data['subject_id'] = f['general']['subject']['subject_id'][()].decode()
        data['session_id'] = f['general']['session_id'][()].decode()
```

```python
def build_data_dict(results):
    subjects = sorted(set(r['subject_id'] for r in results))
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    ...
        subject_idx.append(subject_to_idx[r['subject_id']])
```

iii. CONVERSION_NOTES.md Step 2/3: the data contain "11 (m3, m4, m7, m11-m15, m17-m19)", and Step 4
resolves the apparent discrepancy with the paper: "Paper says 14 mice total (11 switch + 3 fixed) …
Data contains only 11 switch-task mice. Fixed-condition mice (n=3) not included. Consistent."
Reading the ID from the file metadata rather than the path was not explicitly justified, but it is
self-consistent with the directory names.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Files are enumerated in sorted order within each subject directory,
so sessions appear in chronological (`ses-01 … ses-14`) order. Session identity is also stored in
metadata via `general/session_id`. No cross-session neuron alignment is attempted; each session's
neurons are treated as an independent population.

ii.
```python
        for nwb_file in sorted(os.listdir(subj_path)):
            if nwb_file.endswith('.nwb'):
                files.append(os.path.join(subj_path, nwb_file))
```
```python
            'session_info': [
                {'subject': r['subject_id'], 'session': r['session_id'],
                 'n_cells': r['n_cells'], 'n_trials': len(r['neural']), ...}
                for r in results
            ],
```
A session is dropped only if it yields fewer than 2 usable trials:
```python
    if n_trials < 2:
        print(f"  [{session_label}] SKIP: fewer than 2 trials")
        return None
```

iii. Step 2 of the notes documents the one-file-per-session naming convention, and Step 9 confirms
"Sessions/subject: m11 = 12 (ses-03 to ses-14), all others = 14", cross-checked against the paper's
statement that "imaging started on day 3 for m11 owing to lower viral expression". No session was
actually dropped (152/152 retained).

## 1-d. How are the data split into trials?

i. A trial runs from a frame where the `trial_start` time series equals 1 up to (but not including)
the first subsequent frame where `teleport` equals 1. Both signals are strictly binary in this
dataset. Trials with no following teleport are discarded (there are none in practice). This gives
12,216 trials over 152 sessions before filtering.

ii.
```python
def get_trial_boundaries(trial_start_signal, teleport_signal, trial_number_signal):
    """Extract trial start and end frame indices."""
    starts = np.where(trial_start_signal == 1)[0]
    teleports = np.where(teleport_signal == 1)[0]

    trial_starts = []
    trial_ends = []

    for s in starts:
        # Find next teleport after this start
        next_teleports = teleports[teleports > s]
        if len(next_teleports) > 0:
            trial_starts.append(s)
            trial_ends.append(next_teleports[0])  # teleport frame (exclusive end)

    return trial_starts, trial_ends
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 6: "Trial boundaries: trial_start==1 to teleport==1
(inclusive of start, exclusive of teleport frame). Extract only active trial frames." The ITI is
excluded because `position` is set to −500 and `environment` to −1 outside trials (Step 2 notes),
and the reference `dff()` likewise masks out the ITI.

## 1-e. How are trials filtered based on quality controls?

i. Four filters:
1. **Lick-sensor error trials** — the paper's own criterion: a trial is dropped if more than 30% of
   its frames have a lick count > 2. This removes exactly **81** trials, the number the paper
   reports.
2. Trials whose neural slice is entirely NaN.
3. Trials with fewer than 2 time bins (after any dual-plane downsampling).
4. Sessions left with fewer than 2 valid trials are dropped entirely (none were).

12,216 → 12,135 trials.

ii.
```python
def detect_lick_errors(lick, trial_starts, trial_ends, threshold=LICK_ERROR_THRESH):
    """Detect trials with stuck lick sensor (paper: >30% frames with lick>2)."""
    error_trials = []
    for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
        trial_licks = lick[start:end]
        n_frames = len(trial_licks)
        if n_frames == 0:
            continue
        frac_high = np.sum(trial_licks > 2) / n_frames
        if frac_high > threshold:
            error_trials.append(i)
    return error_trials
```
```python
    for i in range(n_trials):
        if i in lick_error_trials:
            continue
        ...
        # Skip trials with all-NaN neural data
        if np.all(np.isnan(trial_neural)):
            continue
        ...
        T = trial_neural.shape[1]
        if T < 2:
            continue
```

iii. CONVERSION_NOTES.md Step 4 records the discrepancy it resolved: "Lick error threshold:
`correction_thr=0.5` (code default) vs paper's '>30% frames with lick>2' → Paper is authoritative:
use 0.3 to match paper's reported 81 removed trials." Step 10, Check 3 confirms: "Lick error
threshold: paper's 0.3 gives exactly 81 (verified), code default 0.5 gives only 44." The ≥2-timepoint
rule exists because the decoder requires at least two trials/timepoints per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces: `processing/ophys/Fluorescence/plane*/data` (F) and
`processing/ophys/Neuropil/plane*/data` (Fneu), restricted to the ROIs flagged as cells by
`ImageSegmentation/PlaneSegmentation/iscell[:,0] == 1`. The NWB `Deconvolved` field is **not** used.
For the two dual-plane mice (m17, m18) the per-plane ROI sets are selected with `planeIdx` and
concatenated along the cell axis.

ii.
```python
        seg = ophys['ImageSegmentation']['PlaneSegmentation']
        iscell = seg['iscell'][:]
        plane_idx = seg['planeIdx'][:]
        fluor_grp = ophys['Fluorescence']
        plane_keys = sorted(fluor_grp.keys())
        ...
        for pk in plane_keys:
            plane_num = int(pk.replace('plane', ''))
            plane_mask = plane_idx == plane_num
            plane_iscell = iscell[plane_mask, 0].astype(bool)
            fl = fluor_grp[pk]['data'][:].T  # (n_rois, T)
            neu = ophys['Neuropil'][pk]['data'][:].T  # (n_rois, T)
            fluor_list.append(fl[plane_iscell])
            neuropil_list.append(neu[plane_iscell])
        data['fluorescence'] = np.concatenate(fluor_list, axis=0).astype(np.float64)
        data['neuropil'] = np.concatenate(neuropil_list, axis=0).astype(np.float64)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 1: "Neural signal: Compute dF/F from raw Fluorescence
+ Neuropil … Matches reference processing exactly." Step 1 identifies `preprocessing.dff()` as the
reference function that computes the paper's neural signal from F and Fneu rather than reading a
stored trace.

## 2-b. How is the `neural` data processed?

i. A direct re-implementation of the reference `preprocessing.dff()`, up to but **excluding**
deconvolution. Per session: mask everything outside trials to NaN; subtract `0.7 × Fneu`; add each
trial's mean neuropil back so the ratio is a true dF/F; compute a maximin baseline within each trial
(NaN-aware Gaussian smoothing with σ = 15 frames, then a 300-frame running minimum followed by a
300-frame running maximum — the Methods' 20 s window); form `(F − baseline)/|baseline|`; smooth with
a 2-frame Gaussian. Remaining NaNs are replaced by 0 in the per-trial output, and arrays are cast to
`float32`.

**The OASIS deconvolution step is not performed** — the stored `neural` signal is smoothed dF/F, not
the paper's "activity rate"/events. The reference solution runs `dcnv.oasis(..., tau=0.7,
frame_rate/n_planes)` and stores the events. The per-session `keep_teleports` exception from the
paper's `teleport_metadata.py` is also not implemented.

ii.
```python
def compute_dff(f_raw, f_neu, trial_starts, trial_ends):
    n_cells, n_time = f_raw.shape
    f_ = np.full_like(f_raw, np.nan)
    f_neu_ = np.full_like(f_neu, np.nan)
    for start, end in zip(trial_starts, trial_ends):
        f_[:, start:end] = f_raw[:, start:end]
        f_neu_[:, start:end] = f_neu[:, start:end]
    f_ = f_ - NEUROPIL_COEF * f_neu_
    baseline = np.full_like(f_, np.nan)
    dff = np.full_like(f_, np.nan)
    for start, end in zip(trial_starts, trial_ends):
        trial_data = f_[:, start:end]
        neuropil_mean = np.nanmean(f_neu_[:, start:end], axis=1, keepdims=True)
        trial_data = trial_data + NEUROPIL_COEF * neuropil_mean
        f_[:, start:end] = trial_data
        smoothed = nansmooth(trial_data, BASELINE_SMOOTH_SIGMA, axis=1)
        bl = ndi.minimum_filter1d(smoothed, BASELINE_MINMAX_WINDOW, axis=-1)
        bl = ndi.maximum_filter1d(bl, BASELINE_MINMAX_WINDOW, axis=-1)
        baseline[:, start:end] = bl
    nanmask = ~np.isnan(f_[0, :])
    dff[:, nanmask] = (f_[:, nanmask] - baseline[:, nanmask]) / np.abs(baseline[:, nanmask])
    for start, end in zip(trial_starts, trial_ends):
        dff[:, start:end] = nansmooth(dff[:, start:end], DFF_SMOOTH_SIGMA, axis=1)
    return dff
```
```python
NEUROPIL_COEF = 0.7
BASELINE_SMOOTH_SIGMA = 15    # frames
BASELINE_MINMAX_WINDOW = 300  # frames (~20 s at 15.5 Hz)
DFF_SMOOTH_SIGMA = 2          # frames
```
```python
        trial_neural = np.nan_to_num(trial_neural, nan=0.0).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 1: "Matches reference processing exactly. **More
informative than deconvolved events for neural network decoder.** Reference method: neuropil
subtract (0.7), add back neuropil mean per trial, maximin baseline (smooth sigma=15, min filter 300,
max filter 300), dF/F = (F−baseline)/|baseline|, smooth sigma=2." The trajectory (steps 49, 57)
shows the AI read the reference `dff()` carefully, including the `deconvolve` branch, but the notes
never state a reason for omitting deconvolution beyond the "more informative" clause, and the claim
that the pipeline "matches the reference exactly" is not accurate on this point.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's:
1. **suite2p manual curation** — only ROIs with `iscell[:,0] == 1` are loaded (applied before dF/F,
   per plane).
2. **Putative interneurons** — after dF/F, any cell whose dF/F correlates with running speed at
   Pearson r > 0.5 is dropped. Cells with fewer than 10 valid frames are kept.

No place-cell selection is applied. Result: 138,276 neurons, 909.7 ± 448.0 per session, interneuron
removal rate 0.29%.

ii.
```python
INTERNEURON_R_THRESH = 0.5    # Pearson r threshold (paper)

def detect_interneurons(dff, speed, nanmask=None):
    """Detect putative interneurons by speed-dFF correlation (paper: r > 0.5)."""
    n_cells = dff.shape[0]
    if nanmask is None:
        nanmask = ~np.isnan(dff[0, :])
    is_interneuron = np.zeros(n_cells, dtype=bool)
    for c in range(n_cells):
        valid = nanmask & ~np.isnan(dff[c, :]) & ~np.isnan(speed)
        if valid.sum() < 10:
            continue
        r = np.corrcoef(dff[c, valid], speed[valid])[0, 1]
        if r > INTERNEURON_R_THRESH:
            is_interneuron[c] = True
    return is_interneuron
```
```python
    is_interneuron = detect_interneurons(dff, speed_for_corr, nanmask)
    keep_mask = ~is_interneuron
    dff = dff[keep_mask]
    n_cells = dff.shape[0]
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 2: "Cell filtering: iscell + interneuron exclusion
(r>0.5). Use Suite2P manual curation (`iscell[:,0]==1`) then exclude putative interneurons with
speed-dFF correlation > 0.5 (paper threshold). **Do NOT filter to place cells only** — use all
curated pyramidal cells for the decoder." Step 4 notes the reference code's default `r_thresh=0.3`
was overridden by the paper's stated 0.5. Key Decision 10: "No speed filtering for neural data …
speed < 2 cm/s filtering is only for place cell identification, not for general neural activity."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires no extra work: the neural and behavioral arrays share one frame
index, so each trial is simply the slice `[trial_start_frame : teleport_frame)` of both streams. The
first neural column of every trial is therefore the trial-start frame, and `off_start` is recorded
as 0.0.

ii.
```python
        start = trial_starts[i]
        end = trial_ends[i]
        trial_neural = dff[:, start:end]
        trial_pos = raw['position'][start:end]
        trial_speed = raw['speed'][start:end]
        trial_lick = raw['lick'][start:end]
```
```python
            'temporal_alignment_event': 'Start of trial (entry onto linear track)',
            'off_start': 0.0,
            'off_end': None,  # Variable trial length
```

iii. Trajectory step 40: "Neural and behavior data are on the SAME timebase (same number of frames,
both at ~15.5 Hz)". The notes' Step 2 records identical array lengths for neural and behavior
streams, and `load_nwb_session` crops both to the shorter when they differ by a frame.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The declared bin is **64.48 ms** (1/15.5078125 Hz) for all sessions, and for the 124 single-plane
sessions no rebinning is done — frames are kept at their stored rate.

For the 28 dual-plane sessions (m17, m18) the AI **does** rebin: it reads the NWB `rate` attribute
(31.015625 Hz), concludes the streams are sampled at 31 Hz, and averages every 2 consecutive frames
of the neural, position and speed arrays (max over the pair for lick), setting
`effective_rate = rate / 2 = 15.5 Hz`.

This premise is wrong. The stored `rate` is the *scanner* rate; the arrays are already stored once
per plane cycle. `behavior_timestamps` in those very files have a median spacing of 64.48 ms
(22,634 samples spanning 1,459 s), identical to the single-plane sessions. Averaging pairs therefore
produces **128.97 ms** bins for m17/m18 while the metadata reports 64.48 ms for the whole dataset,
so the bin size is not in fact constant across sessions. It also halves those sessions' trial
lengths (verification reports T_min = 62 and T_mean = 195.1 versus the reference's 96 and 216.8).

ii.
```python
TARGET_RATE = 15.5078125       # Hz, canonical frame rate
...
        rate = fluor_grp[plane_keys[0]]['starting_time'].attrs['rate']
        data['rate'] = float(rate)
        data['is_dual_plane'] = len(plane_keys) > 1
```
```python
    # Downsample dual-plane data
    downsample = 2 if raw['is_dual_plane'] else 1
    effective_rate = raw['rate'] / downsample
    ...
        if downsample > 1:
            T = trial_neural.shape[1]
            T_new = T // downsample
            if T_new < 2:
                continue
            trial_neural = trial_neural[:, :T_new * downsample].reshape(n_cells, T_new, downsample).mean(axis=2)
            trial_pos = trial_pos[:T_new * downsample].reshape(T_new, downsample).mean(axis=1)
            trial_speed = trial_speed[:T_new * downsample].reshape(T_new, downsample).mean(axis=1)
            trial_lick = trial_lick[:T_new * downsample].reshape(T_new, downsample).max(axis=1)
```
```python
            'time_bin_size': 1000.0 / TARGET_RATE,  # ~64.48 ms
```

iii. CONVERSION_NOTES.md Step 4: "Dual-plane mice … m17,m18: rate=31.02 Hz, 2 planes … Need to
downsample by 2x to match 15.5 Hz", and Step 5, Key Decision 4: "Dual-plane mice (m17, m18):
Downsample by 2x. Average every 2 consecutive frames for neural and behavioral data to get
consistent ~15.5 Hz across all sessions. Combine neurons from both planes." Key Decision 5: "Time
bin: 1/15.5078125 Hz ≈ 64.48 ms for all sessions." Trajectory step 44 states the assumption
explicitly — "The behavior data has 22634/22795 frames at 31 Hz" — and it was never checked against
the behavior timestamps, which the loader reads but never uses for this purpose.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Not from any raw time variable. It is synthesized from the within-trial frame index and the
`rate` attribute of the fluorescence series (`effective_rate`). The file's
`BehavioralTimeSeries/position/timestamps` array *is* loaded (as `behavior_timestamps`) but is used
only for matching reward events, never for this input.

ii.
```python
        rate = fluor_grp[plane_keys[0]]['starting_time'].attrs['rate']
        data['rate'] = float(rate)
...
    effective_rate = raw['rate'] / downsample
...
        time_from_start = np.arange(T, dtype=np.float32) / effective_rate
```

iii. The notes' Step 5 mapping table gives: "Frame timestamps within trial → input[0]:
time_from_trial_start → `(frame_idx - trial_start_idx) / rate`". The AI had verified (trajectory
step 40) that behavior timestamps for m11 advance at the fluorescence `rate`, and generalized from
that one check.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. `np.arange(T) / effective_rate`, so the first sample of every trial is exactly 0 s and the step
is one bin. For the 124 single-plane sessions this is numerically identical to
`timestamps − timestamps[0]` (1/15.5078125 = 0.0644836 s, matching the stored timestamp spacing to
machine precision; the global maximum, 216.5 s, equals the reference's 216.536 s).

For the 28 dual-plane sessions the values are **compressed by a factor of 2**: because the frames
were averaged in pairs but `effective_rate` is still 15.5 Hz, a m17 trial that truly lasts 22.18 s
is emitted as 172 samples spanning 11.03 s.

ii.
```python
        time_from_start = np.arange(T, dtype=np.float32) / effective_rate
        ...
        inputs = np.stack([time_from_start, env, trial_num, prev_outcome], axis=0)
```

iii. Notes Step 7 report the sanity check "Time from trial start [0, 31.5] seconds" for the sample
sessions (both single-plane), and Step 9 checks the global range. No per-mouse check of elapsed time
against the stored timestamps was performed, so the dual-plane compression was not caught.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: `T` is taken from `trial_neural.shape[1]` after any downsampling, so the time
vector always has exactly one entry per neural bin and starts at 0 on the trial-start frame. No
interpolation or offset is applied. The behavioral and neural arrays are cropped to a common length
at load time whenever they differ (10 sessions differ by one frame).

ii.
```python
        T = trial_neural.shape[1]
        if T < 2:
            continue
        ...
        time_from_start = np.arange(T, dtype=np.float32) / effective_rate
```
```python
        n_neural = data['fluorescence'].shape[1]
        n_behav = len(data['position'])
        n_min = min(n_neural, n_behav)
        if n_neural != n_behav:
            data['fluorescence'] = data['fluorescence'][:, :n_min]
            ...
```

iii. Step 5, Key Decision 9: "All inputs/outputs time-varying: per-trial values broadcast to (d, T)
for consistency." Step 10, Check 5: "Dual-plane frame mismatch: fixed by truncation."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `BehavioralTimeSeries/environment` time series, which takes the values −1 (ITI), 0 (ENV1) and
1 (ENV2).

ii.
```python
        data['environment'] = bts['environment']['data'][:].astype(np.float64)
```
```python
    environments = []
    for start, end in zip(trial_starts, trial_ends):
        env_vals = raw['environment'][start:end]
        active_env = env_vals[env_vals >= 0]
```

iii. Notes Step 2 record "environment: {−1, 0} (−1 = ITI)" for m11, and trajectory step 42 confirms
"m17 and m18 have env=1 (ENV2), matching the paper's statement that these 2 mice started in ENV2."
Step 4 records that only session 08 contains a within-session environment switch, consistent with
the paper's day-8 cross-environment day.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the −1 (ITI) samples are discarded and the **first** remaining value is taken
as the trial's environment, rounded to an integer and broadcast across all T bins. If a trial
contains no non-negative sample, the environment silently defaults to 0.

ii.
```python
    environments = []
    for start, end in zip(trial_starts, trial_ends):
        env_vals = raw['environment'][start:end]
        active_env = env_vals[env_vals >= 0]
        if len(active_env) > 0:
            environments.append(int(round(active_env[0])))
        else:
            environments.append(0)
...
        env = np.full(T, environments[i], dtype=np.float32)
```

iii. Notes Step 5 mapping: "environment variable → input[1]: environment → 0=ENV1, 1=ENV2 →
per-trial, broadcast to T". Verification confirms the input range is exactly [0, 1] and that whole
sessions are [0,0] or [1,1] except the day-8 switch sessions, which read [0,1].

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Not from the stored `trial number` time series. It is the position of the trial in the
`trial_starts`/`trial_ends` lists derived from the `trial_start` and `teleport` signals, i.e. the
loop counter `i`, which restarts at 0 in each session. (The stored `trial number` series is loaded
and used only to attribute reward events to trials — see 6-a.)

ii.
```python
    for i in range(n_trials):
        ...
        trial_num = np.full(T, i, dtype=np.float32)
```

iii. The notes' Step 5 mapping row reads "trial number variable → input[2]: trial_number → Integer
trial index within session". Verification shows the resulting range is [0, 79] for the standard
80-trial sessions and up to [0, 99] for the 100-trial ones, matching the raw trial counts.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the loop index over the trial's T bins. The index is the **original**
trial index, so removing a lick-error trial leaves a gap in the sequence rather than renumbering the
survivors — the number still reflects how far into the session the animal is.

ii.
```python
        trial_num = np.full(T, i, dtype=np.float32)
        inputs = np.stack([time_from_start, env, trial_num, prev_outcome], axis=0)
```

iii. Not separately justified in the notes beyond the mapping table. It is consistent with the
instruction that trial number is a continuous per-trial input, and keeping the original index is
what makes "previous trial outcome" (6-b) refer to the true preceding trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `BehavioralTimeSeries/Reward` time series' **timestamps** (reward delivery events, on their
own irregular clock), mapped onto the behavior frame grid via `behavior_timestamps`, and then
attributed to a trial using the stored `trial number` time series at that frame.

ii.
```python
        data['behavior_timestamps'] = bts['position']['timestamps'][:]
        reward_ts_data = bts['Reward']
        data['reward_timestamps'] = reward_ts_data['timestamps'][:]
```
```python
def determine_reward_per_trial(reward_timestamps, behavior_timestamps, trial_number_signal,
                                trial_starts, trial_ends):
    """Determine which trials received reward from reward event timestamps."""
    n_trials = len(trial_starts)
    rewarded = np.zeros(n_trials, dtype=int)
    if len(reward_timestamps) == 0:
        return rewarded
    for rt in reward_timestamps:
        frame_idx = np.argmin(np.abs(behavior_timestamps - rt))
        trial_num = int(trial_number_signal[frame_idx])
        if trial_num >= 0 and trial_num < n_trials:
            rewarded[trial_num] = 1
    return rewarded
```

iii. Notes Step 5 mapping: "Reward events of previous trial → input[3]: previous_trial_outcome →
0 = omission/no-reward, 1 = rewarded → per-trial, broadcast to T. First trial = 0", citing
`behavior.get_trial_types()` as the reference analogue. Step 2 notes "Reward events: separate
timestamps array (one entry per reward event)", which is why nearest-frame matching is used. Rewards
falling in the ITI (`trial number` = −1) are ignored by the `trial_num >= 0` guard.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial `i > 0` the input is `rewarded[i-1]`, broadcast across the trial's T bins; for `i == 0`
it is 0. Because `i` is the original trial index, "previous" always means the physically preceding
trial, even if that trial was itself dropped as a lick-sensor error.

ii.
```python
        prev_outcome = np.full(T, rewarded[i - 1] if i > 0 else 0, dtype=np.float32)
```

iii. Notes Step 10, Check 2, sanity check 2: "Input prev_outcome: Matches actual reward of preceding
trial for 5 trials checked" against the raw NWB data. The 0-for-first-trial convention follows the
instruction's binary (omitted = 0, rewarded = 1) specification.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` time series, together with the trial's reward zone identity, which is itself
derived from the `reward_zone` time series (integer 0–6, non-zero only in/near the active zone) and
`position`. The zone coordinates are the paper's fixed A [80, 130], B [200, 250], C [320, 370] cm.

ii.
```python
REWARD_ZONES = {
    'A': (80, 130),
    'B': (200, 250),
    'C': (320, 370),
}
ZONE_LABELS = ['A', 'B', 'C']
```
```python
        data['position'] = bts['position']['data'][:].astype(np.float64)
        data['reward_zone'] = bts['reward_zone']['data'][:].astype(np.float64)
```
```python
        zone_label = zone_labels[i]
        zone_idx = ZONE_LABELS.index(zone_label) if zone_label in ZONE_LABELS else 0
        zone_start, zone_end = REWARD_ZONES[zone_label] if zone_label else (80, 130)
        distance = compute_distance_to_reward_zone(trial_pos, zone_start, zone_end)
```

iii. Notes Step 4 resolve the labelling conflict: the reference code's `get_reward_zones()` keys
A/B/C denote an older task ([175,225], [390,440], [325,375]), while the paper's A/B/C correspond to
the code's X [80,130], Y [200,250], Z [320,370]; the AI checked the `reward_zone > 0` positions in
the data and confirmed they cluster at the paper's ranges. Step 4 also records "reward_zone in NWB:
0 = outside zone, >0 = inside/near zone (different interaction states). Confirmed by position
analysis."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Per timepoint, the signed distance to the nearest edge of the trial's reward zone: negative
before the zone, exactly 0 anywhere inside it, positive past it. For dual-plane sessions the
position used is the 2-frame average.

ii.
```python
def compute_distance_to_reward_zone(position, zone_start, zone_end):
    """Compute signed distance from position to nearest point in reward zone."""
    distance = np.zeros_like(position)
    before = position < zone_start
    inside = (position >= zone_start) & (position <= zone_end)
    after = position > zone_end
    distance[before] = position[before] - zone_start
    distance[inside] = 0
    distance[after] = position[after] - zone_end
    return distance
```

iii. Notes Step 5, Key Decision 7: "Distance to reward zone: signed distance = pos − rz_start if
pos < rz_start, 0 if inside zone, pos − rz_end if pos > rz_end." Step 10, Check 2, sanity check 4:
"Output distance_to_RZ: Verified against raw position and zone B [200,250]." Trajectory step 84 also
sanity-checks the resulting class fractions: "Distance to RZ shows 25% of timepoints inside the
zone, which initially seems high since the zone only spans 50 cm of the 450 cm track (~11%). But
that's plausible if the mouse slows down substantially once inside the reward zone."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit boolean masks implementing the instruction's bin edges, with the "in zone" class
defined as exactly 0: `< −50 → 0`, `[−50, −10) → 1`, `[−10, 0) → 2`, `== 0 → 3`, `(0, 10] → 4`,
`(10, 50] → 5`, `> 50 → 6`. (Right-closed on the positive side, left-closed on the negative side —
the reference is left-closed on both, so the two differ only for positions falling exactly on +10 or
+50 cm.) Resulting fractions: 0.251 / 0.103 / 0.075 / 0.239 / 0.021 / 0.071 / 0.240, within 0.005 of
the reference solution's on every class.

ii.
```python
def discretize_distance(distance):
    """Discretize distance to reward zone into 7 bins."""
    bins = np.zeros(len(distance), dtype=int)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3  # inside reward zone or exactly at boundary
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```
```python
        'output_values': [
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '>0 to +10cm', '+10 to +50cm', '> +50cm'],
```

iii. The bin edges are taken verbatim from the Decoder Task specification. The `--show-processing`
plots overlay the discretized trace on the continuous one to confirm the thresholding visually
(`plot_processing`, row 3).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial position slice `[start:end]` used for the neural slice, and
(for dual-plane sessions) downsampled with the same 2-frame reshape, so index *t* of the output
always corresponds to index *t* of the neural matrix. No lag or shift is introduced.

ii.
```python
        trial_neural = dff[:, start:end]
        trial_pos = raw['position'][start:end]
        ...
            trial_neural = trial_neural[:, :T_new * downsample].reshape(n_cells, T_new, downsample).mean(axis=2)
            trial_pos = trial_pos[:T_new * downsample].reshape(T_new, downsample).mean(axis=1)
        ...
        distance = compute_distance_to_reward_zone(trial_pos, zone_start, zone_end)
        dist_bins = discretize_distance(distance)
        outputs = np.stack([dist_bins, pos_bins, speed_bins, lick_binary, rz_loc, reward_out], axis=0).astype(int)
```

iii. Notes Step 6 list the `--show-processing` plots as the alignment check: "Plots for
--show-processing mode should visually convince the user that … there are no temporal
misalignments"; `plot_processing` plots raw position and the discretized outputs on the same
per-trial time axis.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `BehavioralTimeSeries/position` time series (cm along the 450 cm virtual corridor; −500
during the ITI, which is excluded because trials stop at the teleport frame).

ii.
```python
        data['position'] = bts['position']['data'][:].astype(np.float64)
...
        trial_pos = raw['position'][start:end]
```

iii. Notes Step 2: "position: (19818,) − range −500 to ~451 cm (−500 = ITI)"; Step 3 records the
paper's 450 cm track length.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond the per-trial slice (and the 2-frame average for dual-plane sessions) followed by
discretization. Raw centimetre values are used directly, with no wrapping or renormalization.

ii.
```python
        trial_pos = raw['position'][start:end]
        ...
        pos_bins = discretize_position(trial_pos)
```

iii. Notes Step 5 mapping: "position → output[1]: absolute_position → Discretize into 5 bins (90 cm
each)". Step 10, Check 2, sanity check 3: "Output position: 9 spot-checks all match raw position
data."

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm bins spanning the 450 cm track, with the first and last bins left open so the handful
of samples marginally outside [0, 450] are absorbed rather than forming extra classes:
`< 90 → 0`, `[90,180) → 1`, `[180,270) → 2`, `[270,360) → 3`, `≥ 360 → 4`. Resulting fractions
0.212 / 0.176 / 0.231 / 0.227 / 0.154, within 0.002 of the reference on every class.

ii.
```python
def discretize_position(position):
    """Discretize position into 5 equal bins (90 cm each)."""
    bins = np.zeros(len(position), dtype=int)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins
```

iii. Bin edges are taken directly from the Decoder Task specification ("5 equal-sized bins spanning
the 450 cm track"); the track length is confirmed at 450 cm in Step 3 of the notes.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identically to 7-d: the same `[start:end]` slice and the same 2-frame downsampling as the neural
matrix, so the two are sample-for-sample aligned.

ii.
```python
        trial_neural = dff[:, start:end]
        trial_pos = raw['position'][start:end]
        ...
        pos_bins = discretize_position(trial_pos)
```

iii. Same as 7-d — no realignment is needed because all NWB behavioral series share the frame grid
of the imaging data; the `--show-processing` plots were used to confirm this visually.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `BehavioralTimeSeries/lick` time series (an integer lick count per imaging frame, observed
range 0–6).

ii.
```python
        data['lick'] = bts['lick']['data'][:].astype(np.float64)
...
        trial_lick = raw['lick'][start:end]
```

iii. Notes Step 2: "lick: (19818,) − lick count per frame (0−6)". The same series drives the
lick-sensor-error trial rejection (1-e).

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized as `lick > 0`. For dual-plane sessions the 2-frame downsampling uses `max` rather than
`mean`, so a lick anywhere in the pair survives as a 1. Resulting distribution 76.7% no-lick /
23.3% lick, essentially identical to the reference's 77.0% / 23.0%.

ii.
```python
        lick_binary = (trial_lick > 0).astype(int)
```
```python
            trial_lick = trial_lick[:T_new * downsample].reshape(T_new, downsample).max(axis=1)
```

iii. Notes Step 5 mapping: "lick → output[3]: lick → Binary (>0 → 1)". Step 10, Check 2, sanity
check 6: "Output lick: 266/266 frames match raw lick>0." Choosing `max` for downsampling preserves
the binary "did a lick occur" semantics rather than diluting it.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start:end]` slice and same downsampling factor as the neural matrix, so one lick value per
neural bin, starting at the trial-start frame.

ii.
```python
        trial_lick = raw['lick'][start:end]
        ...
            trial_lick = trial_lick[:T_new * downsample].reshape(T_new, downsample).max(axis=1)
        lick_binary = (trial_lick > 0).astype(int)
        outputs = np.stack([dist_bins, pos_bins, speed_bins, lick_binary, rz_loc, reward_out], axis=0).astype(int)
```

iii. Same reasoning as 7-d/8-d: all behavior series in the NWB file share the imaging frame grid, so
common indexing guarantees alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The `reward_zone` time series (non-zero only when the animal is in/near the active zone) combined
with `position`, matched against the paper's three fixed zone ranges A [80,130], B [200,250],
C [320,370].

ii.
```python
        data['reward_zone'] = bts['reward_zone']['data'][:].astype(np.float64)
...
    zone_labels = determine_reward_zone(raw['position'], raw['reward_zone'],
                                         trial_starts, trial_ends)
```

iii. See 7-a: Step 4 of the notes documents the A/B/C ↔ X/Y/Z relabelling between the paper and the
reference code and the empirical confirmation from the `reward_zone > 0` positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Per trial: take the **mean** position over the frames where `reward_zone > 0`, and assign the zone
whose **centre** (105, 225, 345 cm) is closest. Trials in which `reward_zone` is never non-zero get
no label in the first pass and then inherit the label of the nearest labelled trial (searching
backwards first, then forwards). The zone index (A = 0, B = 1, C = 2) is broadcast across all T bins.
If a label is still missing at use time, zone A / [80,130] is used as a silent fallback. Resulting
distribution A 32.8% / B 33.7% / C 33.5%, matching the reference's 32.9 / 33.7 / 33.4 to within 0.1%.

ii.
```python
def determine_reward_zone(position, reward_zone_signal, trial_starts, trial_ends):
    n_trials = len(trial_starts)
    zone_labels = [None] * n_trials
    for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
        rz = reward_zone_signal[start:end]
        pos = position[start:end]
        mask = rz > 0
        if mask.sum() > 0:
            mean_pos = pos[mask].mean()
            best_zone = None
            best_dist = float('inf')
            for label, (zs, ze) in REWARD_ZONES.items():
                center = (zs + ze) / 2
                dist = abs(mean_pos - center)
                if dist < best_dist:
                    best_dist = dist
                    best_zone = label
            zone_labels[i] = best_zone
    # Fill missing labels by inheriting from nearest neighbor
    for i in range(n_trials):
        if zone_labels[i] is None:
            for d in range(1, n_trials):
                if i - d >= 0 and zone_labels[i - d] is not None:
                    zone_labels[i] = zone_labels[i - d]
                    break
                if i + d < n_trials and zone_labels[i + d] is not None:
                    zone_labels[i] = zone_labels[i + d]
                    break
    return zone_labels
```
```python
        rz_loc = np.full(T, zone_idx, dtype=int)
```

iii. Notes Step 5, Key Decision 8: "Reward zone determination: For each trial, use mean position
where reward_zone > 0 to determine zone. If no rz entry, inherit from nearest trial with rz entry."
The planned sanity check "Reward zone positions: A near 105, B near 225, C near 345 (center of
zones)" is listed in Step 5 and the ~1/3-each distribution is confirmed in Step 9. Trajectory step 84
additionally checks individual sessions: "m19 ses-01 uses zone C throughout, while m11 ses-03
involves zones A and B, consistent with it being a switch session (B→A at trial 30)."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` time series' timestamps, exactly as for the previous-trial-outcome input (6-a): each
reward event timestamp is matched to the nearest behavior frame and attributed to the trial that the
stored `trial number` signal reports at that frame.

ii.
```python
        reward_ts_data = bts['Reward']
        data['reward_timestamps'] = reward_ts_data['timestamps'][:]
...
    rewarded = determine_reward_per_trial(
        raw['reward_timestamps'], raw['behavior_timestamps'],
        raw['trial_number'], trial_starts, trial_ends)
```

iii. Notes Step 2: "Reward events: separate timestamps array (one entry per reward event)"; Step 5
mapping: "Reward events → output[5]: reward_outcome → 0 = no, 1 = yes → Match reward timestamps to
trials".

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if at least one reward event maps into it, else 0 (the `rewarded` array is
initialised to zeros and set to 1 on a hit, so multiple rewards in one trial collapse to 1). The
per-trial scalar is broadcast across all T bins. Reward events landing in the ITI (`trial number`
= −1) or beyond the last trial are ignored. Overall reward rate: 84.1%, against the reference's
84.3% and the paper's ~15% omission rate.

ii.
```python
    for rt in reward_timestamps:
        frame_idx = np.argmin(np.abs(behavior_timestamps - rt))
        trial_num = int(trial_number_signal[frame_idx])
        if trial_num >= 0 and trial_num < n_trials:
            rewarded[trial_num] = 1
```
```python
        reward_out = np.full(T, rewarded[i], dtype=int)
```

iii. Notes Step 5 planned sanity check "Reward rate ~85% (paper: ~15% omission)", confirmed in Step 9
as "Reward rate | ~85% | 84.1% | CLOSE", and in Step 7 for the sample ("Reward rate 87.2%").

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Six cases:
- **Neural/behavior length mismatch** (10 sessions differ by one frame): every stream is cropped to
  the shorter length at load time.
- **Trial starts with no following teleport**: silently dropped by `get_trial_boundaries`.
- **All-NaN neural trials** and **trials shorter than 2 bins**: skipped.
- **Sessions with fewer than 2 (valid) trials**: the whole session is skipped and returns `None`.
- **Trials with no `reward_zone > 0` samples**: the zone label is inherited from the nearest
  labelled trial; if none exists anywhere, zone A is used as a fallback.
- **Trials with no non-ITI `environment` sample**: environment defaults to 0.
- Residual NaNs in the neural matrix are converted to 0.
- Any unexpected exception in a session is caught, printed with a traceback, and that session is
  skipped rather than aborting the run.

There is **no** assertion that the different behavioral series share timestamps, and no assertion
bounding the reward-event-to-frame matching error (the reference has both).

ii.
```python
        n_neural = data['fluorescence'].shape[1]
        n_behav = len(data['position'])
        n_min = min(n_neural, n_behav)
        if n_neural != n_behav:
            data['fluorescence'] = data['fluorescence'][:, :n_min]
            data['neuropil'] = data['neuropil'][:, :n_min]
            data['position'] = data['position'][:n_min]
            ...
```
```python
    if n_trials < 2:
        print(f"  [{session_label}] SKIP: fewer than 2 trials")
        return None
    ...
    if n_cells == 0:
        print(f"  [{session_label}] SKIP: no cells after filtering")
        return None
    ...
        if np.all(np.isnan(trial_neural)):
            continue
    ...
        trial_neural = np.nan_to_num(trial_neural, nan=0.0).astype(np.float32)
    ...
    if len(neural_trials) < 2:
        print(f"  [{session_label}] SKIP: fewer than 2 valid trials")
        return None
```
```python
        except Exception as e:
            print(f"  ERROR processing {label}: {e}")
            import traceback
            traceback.print_exc()
```

iii. Notes Step 10, Check 5: "Edge cases verified — Low-trial sessions: early termination, valid;
Long trials: mouse stopped running, valid; Dual-plane frame mismatch: fixed by truncation."
Trajectory steps 102–103 show the length mismatch was discovered as a broadcast error during the
first full run and fixed by cropping. All 152 sessions completed with zero `ERROR` lines in
`conversion_full_out.txt`, so no session was silently lost.

## 13-a. What are the most time-consuming steps of the code?

i. The script times and prints each stage. From `conversion_full_out.txt` the whole conversion is
546 s for 152 sessions (~3.6 s/session), split roughly as:
1. **NWB/HDF5 reading** of the F and Fneu arrays — 0.2 s (155 cells) to ~1.1 s (2,300 cells).
2. **dF/F computation** — 0.3 s to 2.5 s per session; the largest single cost, dominated by the
   per-trial `nansmooth` and the 300-frame min/max filters over (n_cells × T).
3. The remainder of `process_session` (interneuron correlation loop + per-trial extraction) — ~1 s
   for large sessions.
4. **Pickling the 8.4 GB output**, which is not separately timed but is a substantial share of the
   wall clock and dominates memory.

ii.
```python
    t0 = time.time()
    raw = load_nwb_session(filepath)
    t_load = time.time() - t0
    print(f"  [{session_label}] Loaded: {raw['n_cells']} cells, rate={raw['rate']:.2f} Hz, "
          f"dual_plane={raw['is_dual_plane']}, time={t_load:.1f}s")
    ...
    t1 = time.time()
    dff = compute_dff(raw['fluorescence'], raw['neuropil'], trial_starts, trial_ends)
    t_dff = time.time() - t1
    print(f"  [{session_label}] dF/F computed: time={t_dff:.1f}s")
    ...
    t_total = time.time() - t0
    print(f"  [{session_label}] Done: {len(neural_trials)} valid trials, "
          f"{n_cells} cells, time={t_total:.1f}s")
```

iii. Notes Step 7: "Processing time: 2.7 s for 2 sessions … Estimated full conversion: ~25 minutes
for 152 sessions"; the actual run took 9 minutes, comfortably inside the instruction's 15-minute
target, so no further optimization was pursued.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python-level loops remain:
- `detect_interneurons` loops over cells, recomputing a validity mask and calling `np.corrcoef`
  once per cell; the whole thing is a single masked matrix–vector correlation.
- `determine_reward_per_trial` loops over reward events doing an O(T) `np.argmin` per event; a
  single `np.searchsorted` on the sorted timestamps would be O(n log T).
- `get_trial_boundaries` loops over trial starts and re-filters the full teleport array each
  iteration; one `np.searchsorted` call replaces it.
- `compute_dff` walks the trial list three separate times (mask, baseline, smooth).
- `detect_lick_errors`, `determine_reward_zone` (plus its inner 3-zone loop, replaceable by
  `np.argmin` over zone centres), and the `environments` loop each make an extra full pass over the
  trials.
- The main per-trial loop uses `if i in lick_error_trials` — a linear scan of a list inside a loop
  over trials.

None of these are bottlenecks at this dataset size; the two dominant costs (HDF5 reads and the
`scipy.ndimage` filters) are already vectorized.

ii.
```python
    for c in range(n_cells):
        valid = nanmask & ~np.isnan(dff[c, :]) & ~np.isnan(speed)
        if valid.sum() < 10:
            continue
        r = np.corrcoef(dff[c, valid], speed[valid])[0, 1]
```
```python
    for rt in reward_timestamps:
        frame_idx = np.argmin(np.abs(behavior_timestamps - rt))
```
```python
    for s in starts:
        next_teleports = teleports[teleports > s]
```

iii. The notes do not discuss these loops; the AI stopped optimizing once the projected runtime met
the instruction's threshold ("Estimated full conversion: ~25 minutes", actual 9 minutes).

## 13-c. What processing does the code repeat multiple times?

i. Relatively little. Each NWB file is opened and read exactly **once** — there is no separate
survey/statistics pass, so the full neural arrays are not re-read. Within a session, however:
- the trial list is traversed six times (dff masking, dff baseline, dff smoothing, lick-error
  detection, zone determination, environment extraction, then the main extraction loop);
- the per-trial `[start:end]` slices are recomputed in each of those passes;
- `nanmask`/`np.isnan(dff)` is recomputed inside `compute_dff` and again in `process_session` and
  once per cell inside `detect_interneurons`;
- `raw['fluorescence']` and `raw['neuropil']` are held in memory for the whole session even though
  only `dff` is needed after `compute_dff` (they are used again only in `--show-processing` mode).

ii.
```python
    dff = compute_dff(raw['fluorescence'], raw['neuropil'], trial_starts, trial_ends)
    ...
    nanmask = ~np.isnan(dff[0, :])
    is_interneuron = detect_interneurons(dff, speed_for_corr, nanmask)
    ...
    lick_error_trials = detect_lick_errors(raw['lick'], trial_starts, trial_ends)
    zone_labels = determine_reward_zone(raw['position'], raw['reward_zone'], trial_starts, trial_ends)
    rewarded = determine_reward_per_trial(...)
    environments = []
    for start, end in zip(trial_starts, trial_ends):
        ...
    for i in range(n_trials):
        ...
```

iii. Not discussed in the notes. The single-pass design is a consequence of the AI deriving reward
zones per session on the fly (10-b) rather than needing a global survey first.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Minor waste only:
- dF/F is computed for **all** curated cells, including the putative interneurons that are discarded
  immediately afterwards (~0.3% of cells — negligible, and unavoidable since the interneuron test
  needs dF/F).
- The four inputs and the three per-trial outputs (reward zone location, reward outcome, and the
  constant environment/trial-number/previous-outcome inputs) are broadcast to full length-T vectors
  even though the target format permits per-trial scalars; this multiplies six constant channels by
  T and contributes to the 8.4 GB pickle.
- `load_nwb_session` reads `trial_number` and `speed` in full, plus `n_rois`/`offset`/`cell_mask_list`
  bookkeeping in the plane loop that is computed and then never used.
- The raw fluorescence and neuropil arrays are retained on the `raw` dict after `compute_dff`
  returns, although only `--show-processing` needs them.
- dF/F is computed for every trial including the 81 lick-error trials that are then dropped.

ii.
```python
            cell_mask_list.append(plane_iscell)
            offset += n_rois
```
```python
        env = np.full(T, environments[i], dtype=np.float32)
        trial_num = np.full(T, i, dtype=np.float32)
        prev_outcome = np.full(T, rewarded[i - 1] if i > 0 else 0, dtype=np.float32)
        ...
        rz_loc = np.full(T, zone_idx, dtype=int)
        reward_out = np.full(T, rewarded[i], dtype=int)
```

iii. Notes Step 5, Key Decision 9 justifies the broadcasting: "All inputs/outputs time-varying:
Per-trial values broadcast to (d, T) for consistency. Decoder handles both shapes." This follows the
instruction's preference to "make it time-varying … if at all possible".
