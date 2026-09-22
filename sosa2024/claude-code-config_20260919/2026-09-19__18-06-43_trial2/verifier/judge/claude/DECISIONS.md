# Decisions

> **Scope note.** All code snippets below are from `/app/convert_data.py` (the AI's conversion
> script). Justifications are drawn from `/app/CONVERSION_NOTES.md` (Steps 0–5, the only steps the
> agent filled in) and from the recorded trajectory `/logs/agent/trajectory.json`.
>
> **Important context for the whole document:** the agent was cut off during Step 9. The last full
> conversion run (`/app/conversion_full_out.txt`) **failed**: `convert_data.py` raises an
> `AssertionError` on the 10 sessions where the fluorescence array has one more frame than the
> behaviour arrays (m17 ses-04/06, m18 ses-01/05/07/10/11/12/13/14), and `main()` then calls
> `SystemExit('conversion failed')`. Consequently `/app/converted_data.pkl`, `/app/README.md`,
> `/app/verification_full_out.txt` and `/app/train_decoder_full_out.txt` **do not exist**, and
> CONVERSION_NOTES.md Steps 6–13 are still marked `NOT STARTED`. Only the 2-session sample
> (`/app/sample_data.pkl`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`) completed.
> The decisions documented below are therefore read off the script and the sample run.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script walks `/app/data`, treats every sub-directory as a subject folder and every `*.nwb`
file inside it as one session, giving 152 files over 11 subjects (`m3, m4, m7, m11–m15, m17–m19`).
Files are sorted deterministically by (subject number, session number). Each file is opened
**directly with `h5py`** rather than `pynwb` (a deliberate speed choice), and each session is
processed in a separate worker process via `multiprocessing` (`spawn`, default ≤10 workers,
`maxtasksperchild=2`). From each file the agent reads: subject id, session id (= experiment day),
`identifier` (which carries the VR scene name), imaging rate, imaging-plane location, the full
`BehavioralTimeSeries` group, and the raw suite2p `Fluorescence` and `Neuropil` arrays for every
plane. Results from all sessions are gathered, re-sorted, and assembled into the target dict.

Per CONVERSION_NOTES Step 5 decision 10, "**Sessions kept**: all 152."

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
    def key(p):
        base = os.path.basename(p)
        sub = base.split('_')[0].replace('sub-m', '')
        ses = base.split('_')[1].replace('ses-', '')
        return (int(sub), int(ses))
    return sorted(files, key=key)
```

```python
    with h5py.File(path, 'r') as f:
        subject = f['general/subject/subject_id'][()].decode()
        session_id = f['general/session_id'][()].decode()
        exp_day = int(session_id)
        identifier = f['identifier'][()].decode()
        scene = identifier.split('/')[-1]
        imaging_rate = float(f['general/optophysiology/ImagingPlane/imaging_rate'][()])
        region = f['general/optophysiology/ImagingPlane/location'][()].decode()
        beh = load_behavior(f)
        F, Fneu, plane_of_cell = load_fluorescence(f)
```

```python
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, maxtasksperchild=2) as pool:
            for r in pool.imap(_worker, jobs):
```

iii. Step 2 of CONVERSION_NOTES documents the directory layout and verifies it against the paper:
152 NWB files, 11 subjects, 14 sessions each except m11 (12, "imaging started on day 3"), which
matches the paper's "n = 11 mice" switch cohort and the `dd.define_anim_list` cohort in the
reference repo. `h5py` was chosen over `pynwb` for speed (full conversion of 152 sessions ran at
~2.3 min wall clock with 12 workers). The agent also cross-checked `ses-NN` == experiment day using
the `identifier` date and `sessions_dict.py`.

**However**, `process_session` contains a hard assertion that the fluorescence and behaviour arrays
have exactly the same number of frames; 10 sessions violate it by one frame, every worker for those
sessions raises, and the driver aborts the whole run. So although the *intent* is "all 152
sessions", the script as it stands loads none of them to completion.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the NWB metadata field `general/subject/subject_id` (e.g. `m11`),
not from the directory name. The unique set is sorted numerically to build `subjects`, and
`subject_idx` indexes into it per session.

ii.
```python
        subject = f['general/subject/subject_id'][()].decode()
...
    subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
    sub_index = {s: i for i, s in enumerate(subjects)}
...
        'subjects': subjects,
        'subject_idx': np.array([sub_index[r['subject']] for r in results], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2: the 11 subject folders `sub-m3 … sub-m19` match the file-level
`subject_id`, and 11 matches the paper's switch cohort ("Mice were randomly selected to experience
the switch task (n = 11 mice)"). The subject id is also used to look up the per-animal
`keep_teleports` day list, for which the agent established the mapping m\<N\> ≡ the paper's
internal `GCAMP<N>` (Step 4, verified via `identifier` and `sessions_dict.py`).

## 1-c. How are the data split into sessions?

i. One NWB file = one session. `general/session_id` is parsed as an integer experiment day
(`exp_day`), which drives the per-animal `keep_teleports` lookup. A session key
`m<N>_ses-<NN>` is stored, and sessions are ordered by (subject, experiment day). No
cross-session neuron alignment/registration is attempted — planes and sessions are kept separate.
All sessions are retained (all have ≥40 trials, well above the required 2).

ii.
```python
        session_id = f['general/session_id'][()].decode()
        exp_day = int(session_id)
...
    results.sort(key=lambda r: (int(r['subject'][1:]), r['exp_day']))
...
        'session_key': f'{subject}_ses-{session_id}',
```

iii. Step 2: "`/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, one file per mouse per
experiment day. `ses-NN` == experiment day (1-indexed), verified against `sessions_dict.py` (e.g.
`sub-m11_ses-03` has identifier `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`, which is
exactly GCAMP11 exp_day 3)." Treating the session id as experiment day is required because
`teleport_metadata.teleport_sessions` is keyed by experiment day.

## 1-d. How are the data split into trials?

i. A trial is the on-track lap `[trial_start, teleport)`: start indices are the frames where the
binary `trial_start` stream is > 0, end indices (exclusive) are the frames where the binary
`teleport` stream is > 0. The frame carrying the teleport event is deliberately **excluded**
because its interpolated `position` is corrupted. Two robustness fixes handle unpaired events (a
leading teleport before the first trial start, or a trailing trial start with no teleport), and an
assertion checks every teleport follows its trial start.

ii.
```python
    trial_starts = np.where(beh['trial_start'] > 0)[0]
    teleports = np.where(beh['teleport'] > 0)[0]
    # robustness: drop an unpaired leading teleport / trailing trial start
    if len(teleports) and len(trial_starts) and teleports[0] < trial_starts[0]:
        teleports = teleports[1:]
    if len(trial_starts) > len(teleports):
        trial_starts = trial_starts[:len(teleports)]
    if len(teleports) > len(trial_starts):
        teleports = teleports[:len(trial_starts)]
    assert np.all(teleports > trial_starts), f'{path}: teleport before trial start'
    n_trials = len(trial_starts)
...
        pos = beh['position'][st:sp_]
        speed = beh['speed'][st:sp_]
        lick = beh['lick'][st:sp_]
        tt = beh['time'][st:sp_] - beh['time'][st]
        T = sp_ - st
        act = events[:, st:sp_]
```

iii. Step 4 discrepancy table: "`pp.dff`/`glmUtils` use `[start-1 : stop-1]`; `behav.get_trial_types`
uses `[start:stop]` … At the `teleport` index the interpolated `position` is a nonsense value
between 450 and −50 (e.g. 200.8 between 448.9 and −50). … **Resolution: use `[trial_start :
teleport)`.** This excludes the corrupted teleport frame (as the reference's `-1` offset also does)
and, unlike the reference's legacy offset, does not pull in a teleport-tunnel frame at the start nor
drop the last on-track frame." The agent independently validated the boundary definition by showing
that the paper's lick-sensor-error rule applied to these windows reproduces the paper's count of 81
removed trials exactly. Resulting statistics: 12,216 trials, mean 80.4 ± 6.2 per session (paper:
80.5 ± 7.4).

## 1-e. How are trials filtered based on quality controls?

i. One trial-level quality filter: **lick-sensor-error trials are dropped**. A trial is flagged when
more than 30% of its frames carry a cumulative lick count > 2 (the paper's criterion). Flagged
trials are excluded entirely from `neural`/`input`/`output` (but still count for trial numbering and
for the previous-trial-outcome lookup). Across the dataset this removes 81 of 12,216 trials. No
minimum-trial-length filter is applied, and no session-level filter is applied.

ii.
```python
LICK_ERROR_FRAC_THRESH = 0.3  # Methods: ">30% of the ... samples in the trial containing a cumulative lick count >2"
LICK_ERROR_COUNT_THRESH = 2
...
        # behavior.correct_lick_sensor_error, threshold from Methods (>30%)
        seg = beh['lick'][st:sp_]
        lick_error[i] = (np.sum(seg > LICK_ERROR_COUNT_THRESH) / len(seg)) > LICK_ERROR_FRAC_THRESH
...
    for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
        if lick_error[i]:
            continue
```

iii. Step 4: the reference's `behav.correct_lick_sensor_error` uses a threshold of 0.5 by default and
0.35 in `glmUtils`, but the Methods say ">30% of the 0.0645 s imaging frame samples in the trial
containing a cumulative lick count >2 … n = 81 out of 12,376 trials removed across 11 switch mice".
The agent scanned all 152 files (`cache/check_lick_thresh.py`) and found 0.30 → 81 trials, 0.35 →
69, 0.50 → 44, so 0.30 reproduces the paper exactly; this simultaneously validated the trial
boundaries. Step 5 decision 5: "Drop lick-sensor-error trials … Matches the paper's removal exactly.
These trials' lick output would be pure artefact." (The reference code NaNs the licks, which in
`glmUtils` removes those samples from the dataset anyway; the agent drops the whole trial because the
decoder needs contiguous per-trial time series.)

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From the **raw suite2p traces** `processing/ophys/Fluorescence/plane*/data` (F) and
`processing/ophys/Neuropil/plane*/data` (Fneu), restricted to `iscell[:,0]==1`. The NWB
`processing/ophys/Deconvolved` array is explicitly **not** used.

ii.
```python
def load_fluorescence(f):
    seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
    iscell = seg['iscell'][:, 0].astype(bool)
    plane_idx = seg['planeIdx'][:]
    Fs, Fneus, keep = [], [], []
    offset = 0
    for plane in sorted(f['processing/ophys/Fluorescence'].keys()):
        Fp = f[f'processing/ophys/Fluorescence/{plane}/data']
        Np = f[f'processing/ophys/Neuropil/{plane}/data']
        n_roi = Fp.shape[1]
        sel = iscell[offset:offset + n_roi]
        Fs.append(np.asarray(Fp[:, :], dtype=np.float32)[:, sel].T)
        Fneus.append(np.asarray(Np[:, :], dtype=np.float32)[:, sel].T)
        keep.append(plane_idx[offset:offset + n_roi][sel])
        offset += n_roi
    assert offset == len(iscell), 'plane ROI counts do not cover PlaneSegmentation'
    return np.concatenate(Fs, axis=0), np.concatenate(Fneus, axis=0), np.concatenate(keep)
```

iii. Step 2/Step 4: "`processing/ophys/Deconvolved/plane*/data` — suite2p `spks` computed from **raw
F** (no NaNs, raw-F scale up to ~1.1e4) → *not* the paper's `events`, which are deconvolved from the
custom ΔF/F. We therefore recompute ΔF/F + OASIS ourselves." Step 5 decision 1: "the paper's
decoder/GLM (Fig. 3, Fig. 7) operate on 'the deconvolved calcium event timeseries'; those events come
from `pp.dff(..., deconvolve=True)` which applies neuropil subtraction and a per-trial maximin ΔF/F
baseline before OASIS." Planes are pooled ("the paper pools imaging planes for all analyses except
Extended Data Fig. 7").

## 2-b. How is the `neural` data processed?

i. A faithful port of `reward_relative.preprocessing.dff(neuropil_method='subtract',
baseline_method='maximin', subtract_baseline=True, deconvolve=True)`:
restrict samples to the lap windows (or lap+preceding ITI when `keep_teleports` is true for that
animal/day), subtract `0.7 × Fneu`, add each segment's mean neuropil back so the ratio is a true
ΔF/F, compute a maximin baseline (NaN-safe Gaussian σ = 15 frames → `minimum_filter1d(300)` →
`maximum_filter1d(300)`, i.e. the Methods' 20 s window), form `(F − baseline)/|baseline|`, smooth
with a σ = 2-frame Gaussian, and deconvolve with OASIS (`suite2p.extraction.dcnv.oasis`, `tau = 0.7`,
`fs` = per-plane frame rate ≈ 15.5078 Hz). The stored `neural` value per trial is the deconvolved
"events" trace, float32.

The one deliberate deviation from the reference is the slicing convention: the reference slices
`[start-1:stop-1]`, the script slices `[start:stop]` so the ΔF/F segments coincide exactly with the
exported laps. `keep_teleports` is taken per animal/day from `teleport_metadata.teleport_sessions`.

ii.
```python
NEU_COEF = 0.7               # pp.dff / make_multi_anim_sess dff_method['neu_coef']
TAU = 0.7                    # pp.dff default (suite2p ops['tau'])
BASELINE_SMOOTH_SIGMA = 15   # pp.dff maximin: nansmooth(f, [0, 15])
BASELINE_FILTER_WIN = 300    # pp.dff maximin: min/max filter over 300 samples (~20 s)
DFF_SMOOTH_SIGMA = 2         # paper: "smoothed with a two-sample (~0.129 s) s.d. Gaussian kernel"
```

```python
    if keep_teleports:
        start_inds = [int(trial_starts[0])] + (np.asarray(teleports[:-1]) + 1).tolist()
        stop_inds = np.asarray(teleports).tolist()
    else:
        start_inds = np.asarray(trial_starts).tolist()
        stop_inds = np.asarray(teleports).tolist()

    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    f_neu_ = np.full(Fneu.shape, np.nan, dtype=np.float32)
    for start, stop in zip(start_inds, stop_inds):
        f_[:, start:stop] = F[:, start:stop]
        f_neu_[:, start:stop] = Fneu[:, start:stop]
    nanmask = ~np.isnan(f_[0, :])
    f_ -= neu_coef * f_neu_

    flow = np.full(f_.shape, np.nan, dtype=np.float32)
    for start, stop in zip(start_inds, stop_inds):
        seg = f_[:, start:stop]
        seg += neu_coef * np.nanmean(f_neu_[:, start:stop], axis=1, keepdims=True)
        f_[:, start:stop] = seg
        base = nansmooth(seg, BASELINE_SMOOTH_SIGMA, axis=-1)
        base = sp.ndimage.minimum_filter1d(base, BASELINE_FILTER_WIN, axis=-1)
        base = sp.ndimage.maximum_filter1d(base, BASELINE_FILTER_WIN, axis=-1)
        flow[:, start:stop] = base

    dff = np.full(f_.shape, np.nan, dtype=np.float32)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])

    events = np.full(f_.shape, np.nan, dtype=np.float32)
    for start, stop in zip(start_inds, stop_inds):
        smoothed = nansmooth(dff[:, start:stop], DFF_SMOOTH_SIGMA, axis=-1)
        dff[:, start:stop] = smoothed
        events[:, start:stop] = dcnv.oasis(np.ascontiguousarray(smoothed), 2000, tau, fs)
```

```python
KEEP_TELEPORT_DAYS = {
    'm11': [1, 7, 8, 14, 15], 'm12': [1, 7, 8, 14, 15], 'm13': [1, 7, 8, 14, 15],
    'm14': [1, 7, 8, 14, 15], 'm15': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm17': [...], 'm18': [...], 'm19': [...],
}
...
    keep_teleports = exp_day in KEEP_TELEPORT_DAYS.get(subject, [])
    dff, events = dff_and_events(F, Fneu, trial_starts, teleports, keep_teleports, fs)
```

iii. Step 3 quotes the Methods for every parameter ("baseline fluorescence was calculated within each
trial independently using a maximin procedure with a 20 s sliding window", "smoothed with a
two-sample (~0.129 s) s.d. Gaussian kernel", "deconvolving dF/F with a canonical calcium kernel using
the OASIS algorithm as used in Suite2p"); `neu_coef = 0.7` and `baseline_method='maximin'` come from
`notebooks/make_multi_anim_sess.md`, and `tau = 0.7` from the suite2p ops. Step 5 decision 2 records
the per-animal/day `keep_teleports` table copied from `teleport_metadata.py`, noting "this only
changes which samples enter the ΔF/F baseline windows; the exported trials are always
`[trial_start:teleport)`". The `[start:stop]` slicing deviation is argued in the Step 4 table
(the reference itself is internally inconsistent between `pp.dff` and `behav.get_trial_types`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. (1) ROIs are restricted to suite2p's manual curation `iscell[:,0]==1` at load time.
(2) Putative interneurons are dropped: cells whose (smoothed) ΔF/F correlates with running speed at
Pearson r > 0.5 over the valid (within-window) samples — the correlation is computed vectorised
rather than in a per-cell loop. (3) A defensive filter drops any cell whose events are not finite
everywhere inside the valid samples.

ii.
```python
INTERNEURON_R_THRESH = 0.5   # Methods: "Pearson correlation of >0.5 between dF/F and running speed"
...
    valid = ~np.isnan(dff[0, :])
    speed_valid = beh['speed'][valid]
    dv = dff[:, valid]
    dv = dv - dv.mean(axis=1, keepdims=True)
    sv = speed_valid - speed_valid.mean()
    denom = np.sqrt((dv ** 2).sum(axis=1) * (sv ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        speed_corr = (dv @ sv) / denom
    is_interneuron = speed_corr > INTERNEURON_R_THRESH

    finite_cells = np.all(np.isfinite(events[:, valid]), axis=1) & np.isfinite(speed_corr)
    keep_cells = (~is_interneuron) & finite_cells
    events = events[keep_cells]
```

iii. Step 3/Step 5: `iscell` is the manual suite2p curation the Methods describe, and the agent
validated it by noting that the minimum `iscell` count per session is exactly 155, matching the
paper's "155–2172 putative pyramidal neurons per session". The interneuron rule is
`spatial.is_putative_interneuron` with `r_thresh = 0.5`, "the value the Methods quote"; the paper
says this excludes 0.42 ± 0.85% of cells, and the sample run reported 0.41 ± 0.34% (2 of 746 cells).
Step 5 decision 7 documents the non-finite-cell guard: "Cells with all-NaN ΔF/F (e.g. a cell whose
baseline is degenerate) are dropped; events are NaN outside trials by construction, and within-trial
NaNs (there should be none) are zero-filled."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial start is the alignment event, and it is simply the left edge of the exported slice, so no
additional alignment is needed. Every trial's arrays begin at the `trial_start` frame. Metadata
records `temporal_alignment_event = 'trial start …'`, `off_start = 0.0`, `off_end = None` (laps have
variable duration).

ii.
```python
        act = events[:, st:sp_]
...
        'temporal_alignment_event':
            'trial start (entry to the linear track at position 0 cm, NWB "trial_start" event)',
        'off_start': 0.0,
        'off_end': None,
        'trial_window':
            'each trial spans [trial_start, teleport), i.e. the whole on-track lap; laps have '
            'variable duration (median ~12.2 s) so off_end is not a fixed value',
```

iii. Step 5 decision 3: "Trial window = `[trial_start, teleport)`; alignment event = trial start,
`off_start = 0`, `off_end = None` (variable-length laps)." All behaviour streams in the NWB are
already interpolated onto the imaging-frame grid (Step 1, `TwoPUtils.preprocessing.vr_align_to_2P`),
so neural and behavioural samples share one index; slicing both with the same `[st:sp_]` guarantees
alignment. The `--show-processing` figures overlay raw F, ΔF/F, events, position, reward zone and
every discretised output on a common time axis to check for misalignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning at all: the data stay at the native imaging frame rate. The bin size is derived
per-session as the median inter-sample interval of the behaviour timestamps (≈ 0.064484 s → 64.48 ms,
i.e. ~15.5078 Hz), and that same `fs = 1/dt` is passed to OASIS. This automatically yields the
**per-plane** rate on the two two-plane mice (m17, m18), whose NWB `imaging_rate` field stores the
31.0156 Hz volume rate. The exported `metadata['time_bin_size']` is the median over sessions, with
the min/max range also stored.

ii.
```python
    dt = float(np.median(np.diff(beh['time'])))
    fs = 1.0 / dt
...
    dts = np.array([r['dt'] for r in results])
...
        'time_bin_size': float(np.median(dts) * 1000.0),
        'time_bin_size_range_ms': [float(dts.min() * 1000), float(dts.max() * 1000)],
```

iii. Step 3: "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame
rate", and Step 3 Processing Details: "Temporal binning: native imaging frame (~64.5 ms). The paper
explicitly uses the frame rate for all decoder/GLM timeseries." Step 2 notes the two-plane subtlety
("2-plane mice m17, m18 store 31.0156 Hz volume rate") — taking `fs` from the timestamps rather than
from `imaging_rate` sidesteps it. The sample run reported `time bin: 64.484 ms`.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From `processing/behavior/BehavioralTimeSeries/position/timestamps`, loaded once as `beh['time']`
and reused for every stream (all behaviour streams in these files share one timestamp vector on the
imaging-frame grid).

ii.
```python
def load_behavior(f):
    b = f['processing/behavior/BehavioralTimeSeries']
    beh = {
        'time': b['position/timestamps'][:],
        'position': b['position/data'][:],
        ...
    }
```

iii. Step 1/Step 2: all `BehavioralTimeSeries` members are "already sampled on the imaging-frame grid
(`timestamps`, ~15.51 Hz, Δt ≈ 0.0645 s), length = n frames", because `vr_align_to_2P` interpolated
them there. Any stream's timestamps would do; `position` was used as the canonical one.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the trial's first timestamp from the per-trial slice, yielding seconds since trial start
(starting at exactly 0). Stored as a time-varying float32 row of `input`.

ii.
```python
        tt = beh['time'][st:sp_] - beh['time'][st]
...
        inp = np.empty((len(INPUT_NAMES), T), dtype=np.float32)
        inp[0] = tt
```

iii. Directly implements the decoder spec ("Time from start of trial in seconds (continuous,
time-varying)") combined with the alignment decision (Step 5 decision 3) that the trial starts at the
`trial_start` frame. The sample verification reports the range `[0.0, 30.3] s`, consistent with the
median ~12.2 s lap and a long tail.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. No resampling is performed: behaviour and neural samples are already on the same frame grid, and
both are sliced with the identical `[st:sp_]` index. The script *verifies* this by asserting that the
fluorescence array and the behaviour timestamp vector have exactly the same length.

ii.
```python
    n_frames = len(beh['time'])
    assert F.shape[1] == n_frames, f'{path}: F has {F.shape[1]} frames, behavior has {n_frames}'
```

iii. Step 1: the NWB `BehavioralTimeSeries` is exactly the output of
`TwoPUtils.preprocessing.vr_align_to_2P`, which interpolates every VR variable onto the imaging-frame
grid, so no further alignment is possible or needed. The agent's diagnostic plots
(`processing_<session>.png`) overlay neural and behavioural traces on a common time axis as a visual
alignment check.

**Caveat:** the assertion is a hard failure rather than a repair. 10 of 152 sessions have exactly one
more imaging frame than behaviour samples, and the agent discovered this only at the very end (step
119 of the trajectory enumerates the 10 offending files) without getting a chance to fix it. See 12.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. From the `environment` behaviour stream (`morph`: 0 = ENV 1, 1 = ENV 2; −1 before the imaging
TTLs start).

ii.
```python
        'environment': b['environment/data'][:],
```

iii. Step 2 documents the encoding: "`environment` (morph: 0 = ENV 1, 1 = ENV 2; −1 pre-TTL)", and
Step 1 identifies `behav.get_trial_types` as the reference function that reads `morph` per trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, the unique non-negative value of `environment` within the lap is taken (with an
assertion that there is exactly one), and that single integer is broadcast across all timepoints of
the trial.

ii.
```python
        env_vals = np.unique(beh['environment'][st:sp_])
        env_vals = env_vals[env_vals >= 0]
        assert len(env_vals) == 1, f'{path}: trial {i} has environment values {env_vals}'
        environment[i] = int(env_vals[0])
...
        inp[1] = environment[i]
```

iii. This is exactly `behav.get_trial_types`'s `morph` computation (Step 1 table:
"`morph[i]` = unique env value in trial"). The decoder spec asks for a per-trial binary, and the
mapping table in Step 5 records "unique value in trial (0 = ENV1, 1 = ENV2) … per-trial, broadcast
over T". Filtering `>= 0` guards against the pre-TTL sentinel value of −1.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Not from any raw variable: it is the 0-indexed position of the trial in the session's ordered list
of `trial_start`/`teleport` pairs. The stored `trial number` behaviour stream is read into `beh` but
is not used for this input.

ii.
```python
    trial_starts = np.where(beh['trial_start'] > 0)[0]
    teleports = np.where(beh['teleport'] > 0)[0]
...
    for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
```

iii. Step 4/Step 5: the trial index derived from `trial_start`/`teleport` is the same index used by
`behav.get_reward_zones` to decide which trials fall before/after the reward-zone switch at trial 30,
so using the loop index keeps the trial number, the zone label and the switch point mutually
consistent.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the integer index across the trial's timepoints. The index counts **all**
detected trials, so removing a lick-error trial leaves a gap in the sequence rather than renumbering
(this is what keeps the trial number aligned to the trial-30 switch).

ii.
```python
        inp[2] = i
```

iii. Step 5 mapping table: "`trial number` … 0-indexed trial index in session … per-trial,
broadcast". The sample verification reports the range `[0.0, 79.0]` per session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the sparse `Reward` TimeSeries (its own `timestamps`, one sample per delivered reward) **and**
the `reward_zone` stream. Reward timestamps are mapped to frame indices with `np.searchsorted`; a
trial counts as rewarded when a reward event falls inside it *and* the reward zone was active during
it. The previous trial's value is then carried forward.

ii.
```python
        'reward_zone': b['reward_zone/data'][:],
        'reward_times': b['Reward/timestamps'][:],
...
    reward_frames = np.searchsorted(beh['time'], beh['reward_times'])
    reward_frames = np.clip(reward_frames, 0, n_frames - 1)
...
        # behavior.get_trial_types: reward delivered AND the reward zone was active
        any_reward = np.any((reward_frames >= st) & (reward_frames < sp_))
        any_rzone = np.any(beh['reward_zone'][st:sp_] > 0)
        is_reward[i] = int(bool(any_reward and any_rzone))
```

iii. Step 1 table: "`behav.get_trial_types` … `isreward[i] = any(reward>0) and any(rzone>0)` within
trial". Step 2 notes `Reward` is "a *sparse* TimeSeries: one 0.004 (mL) sample per reward with its own
timestamps", hence the `searchsorted` mapping onto the frame grid, and that `reward_zone` is
"identically 0 on omission trials", which is what makes the conjunction meaningful (and which
`ra.get_omission_trials` also relies on).

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `prev_outcome[i] = is_reward[i-1]`, broadcast across the trial's timepoints. The first trial of
each session, where the value is undefined, is **imputed as 1 (rewarded)**.

ii.
```python
    prev_outcome = np.empty(n_trials, dtype=np.int64)
    prev_outcome[1:] = is_reward[:-1]
    prev_outcome[0] = 1  # see CONVERSION_NOTES Step 5, decision 8
...
        inp[3] = prev_outcome[i]
```

iii. Step 5 decision 8: "First trial of a session: `previous trial outcome` is undefined. Set to 1
(rewarded) — the modal outcome (84.7%), and mice ran ~30 warm-up trials on the same reward zone
immediately before each imaging session. It is an *input*, so a single imputed value per session
cannot leak or distort the decoded targets." Because `prev_outcome` is indexed over the raw trial
list, a dropped lick-error trial still supplies the correct previous outcome to the following trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behaviour stream plus the per-trial reward-zone interval. The zone interval is
derived **from the VR scene name** in the NWB `identifier` (plus the trial index and the 30-trial
switch rule), not from the `reward_zone` stream: `A = [80, 130]`, `B = [200, 250]`,
`C = [320, 370]` cm.

ii.
```python
REWARD_ZONE_CM = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
CHANGE_TRIAL = 30            # behavior.get_reward_zones default; "Each switch occurred after 30 trials"

def zones_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    """Port of `behavior.get_reward_zones` ..."""
    for first in REWARD_ZONE_LABELS:
        if f'{first}_to' in scene:
            second = scene[-1]
            if second in REWARD_ZONE_LABELS and second != first:
                labels = np.array([first] * min(change_trial, n_trials)
                                  + [second] * max(0, n_trials - change_trial))
                return labels
    for zone in REWARD_ZONE_LABELS:
        if scene.endswith('Location' + zone):
            return np.array([zone] * n_trials)
    raise NotImplementedError(f'Reward zone not defined for scene {scene!r}')
...
        scene = identifier.split('/')[-1]
...
    zone_labels = zones_from_scene(scene, n_trials)
    zone_starts = np.array([REWARD_ZONE_CM[z][0] for z in zone_labels])
    zone_stops = np.array([REWARD_ZONE_CM[z][1] for z in zone_labels])
```

iii. Step 1 identifies `behav.get_reward_zones` as the reference function that assigns the zone
"from the **scene name**; `X=[80,130]` for 'LocationA', `Y=[200,250]` for B, `Z=[320,370]` for C; on
`*_to_*` scenes the first `change_trial=30` trials use the first zone, the rest the second", matching
the paper's "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm" and "Each switch occurred
after 30 trials." The agent then **cross-validated** it empirically (`cache/check_zones.py`, Step 4):
"for all 12,216 trials with an active zone, the empirical first in-zone position is within
[start−1, start+50) of the scene-derived zone start for every trial (5 trials land 1–4 cm early, i.e.
interpolation jitter). The scene+30-trial rule is correct."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Per timepoint, the signed distance to the **nearest point** of the zone interval: negative before
the zone (`pos − zone_start`), positive after it (`pos − zone_stop`), and exactly 0 while inside.
Computed vectorised over the trial with two boolean masks.

ii.
```python
        d = np.zeros(T)
        before = pos < zone_starts[i]
        after = pos > zone_stops[i]
        d[before] = pos[before] - zone_starts[i]
        d[after] = pos[after] - zone_stops[i]
```

iii. Step 5 mapping table: "signed distance to nearest point of [zstart, zstart+50], discretised into
7 bins", referencing `behav.get_reward_zones` and the `glmUtils` reward-relative-position idea in its
linear (non-circular) variant, because the decoder spec asks for a signed linear distance with a
distinct "0 cm" (in-zone) class.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned with explicit boolean masks:
`d < −50 → 0`; `−50 ≤ d < −10 → 1`; `−10 ≤ d < 0 → 2`; `d == 0 → 3`; `0 < d ≤ 10 → 4`;
`10 < d ≤ 50 → 5`; `d > 50 → 6`.

ii.
```python
def digitize_distance_to_reward(d):
    out = np.empty(d.shape, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```

iii. Step 5 lists these edges explicitly and they reproduce the decoder spec's bin table one-for-one.
The dedicated `d == 0` class is exact because the distance is set to exactly 0.0 by construction
inside the zone. The `--show-processing` "discretisation check" figure scatters raw distance against
the assigned bin to confirm the mapping visually. Sample-run class fractions:
`{0.217, 0.096, 0.040, 0.225, 0.021, 0.087, 0.315}`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No separate alignment — `position` is sliced with the same `[st:sp_]` index as the neural events,
on the same imaging-frame grid.

ii.
```python
        pos = beh['position'][st:sp_]
        act = events[:, st:sp_]
```

iii. As in 2-d/3-c: all behaviour streams were interpolated onto the imaging-frame grid before being
written to NWB, and the frame-count assertion plus the diagnostic plots are the agent's checks that
this holds.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. From the `position` behaviour stream (cm along the 450 cm virtual track).

ii.
```python
        'position': b['position/data'][:],
...
        pos = beh['position'][st:sp_]
```

iii. Step 2: "`position` (cm; −500 before imaging TTLs start, −50…0 in the teleport tunnel, 0–450 on
track)". Within a lap the value is the on-track position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing the trial and discretising — the raw centimetre values are used directly.

ii.
```python
        out[1] = digitize_position(pos)
```

iii. Step 5 mapping table: "`position` → `output[1]` absolute position → 5 equal 90 cm bins over
0–450". The track is 450 cm long (Methods) and each lap starts at 0 cm, so no normalisation is
required.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm bins obtained by integer division and clipping, so that any sample marginally outside
`[0, 450)` falls into the first or last bin rather than creating an extra class:
`<90 → 0`, `[90,180) → 1`, `[180,270) → 2`, `[270,360) → 3`, `≥360 → 4`.

ii.
```python
def digitize_position(pos):
    """5 equal 90 cm bins over the 450 cm track."""
    return np.clip((pos // 90.0), 0, 4).astype(np.int64)
```

iii. Step 5 lists the edges explicitly; they implement the spec's "5 equal-sized bins spanning the
450 cm track". The clip handles the small number of samples slightly below 0 or at exactly 450. The
discretisation-check figure plots the assigned bin against raw position. Sample-run class fractions:
`{0.213, 0.196, 0.268, 0.176, 0.149}`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[st:sp_]` slice as the neural data; no resampling.

ii.
```python
        pos = beh['position'][st:sp_]
        act = events[:, st:sp_]
```

iii. Same justification as 2-d / 7-d — the NWB behaviour streams are already on the imaging-frame
grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. From the `lick` behaviour stream (cumulative lick count per imaging frame, 0–8).

ii.
```python
        'lick': b['lick/data'][:],
...
        lick = beh['lick'][st:sp_]
```

iii. Step 2: "`lick` (cumulative licks per frame, 0–8)", produced by the reference's count-style
cumsum-interp-diff alignment onto the imaging frames.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarised: any frame with a lick count > 0 becomes 1, otherwise 0. Additionally, trials whose lick
trace is judged to be a sensor error are removed entirely (see 1-e) rather than being NaN'd.

ii.
```python
        out[3] = (lick > 0).astype(np.int64)
```

iii. Step 5 mapping table: "`lick` → `output[3]` lick → `lick > 0` → 1", referencing `glmUtils`
(which binarises licks for its decoder dataset). The decoder spec requires a binary 0/1 output. The
lick-sensor-error removal is Step 5 decision 5: "These trials' lick output would be pure artefact."
Sample-run class fractions: `{no lick 0.812, lick 0.188}`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[st:sp_]` slice as the neural data; no resampling.

ii.
```python
        lick = beh['lick'][st:sp_]
        act = events[:, st:sp_]
```

iii. Same as 2-d — the lick counts were already binned onto imaging frames by `vr_align_to_2P` before
NWB export.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the VR scene name in the NWB `identifier` field, combined with the trial index and the
30-trial switch rule (see 7-a). The `reward_zone` behaviour stream is used only to validate this
assignment and to gate the reward-outcome variable, not to assign the label.

ii.
```python
        identifier = f['identifier'][()].decode()
        scene = identifier.split('/')[-1]
...
    zone_labels = zones_from_scene(scene, n_trials)
```

iii. See 7-a: this is a direct port of `behav.get_reward_zones`, and the agent validated it against
the empirically observed in-zone entry positions on every one of the 12,216 trials
(`cache/check_zones.py`; Step 4 table row "Reward zone per trial … **Cross-validated**").

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The letter label is mapped to an integer `A → 0, B → 1, C → 2` and broadcast across the trial's
timepoints (the spec's per-trial variable, stored time-varying).

ii.
```python
REWARD_ZONE_LABELS = ['A', 'B', 'C']
...
        out[4] = REWARD_ZONE_LABELS.index(zone_labels[i])
```

iii. Step 5 mapping table: "scene + trial index → `output[4]` reward zone location → A→0, B→1, C→2 →
`behav.get_reward_zones` → per-trial, broadcast", matching the decoder spec ("Reward zone location,
per-trial. 0 = A, 1 = B, 2 = C").

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From the sparse `Reward` TimeSeries timestamps and the `reward_zone` stream — the same `is_reward`
vector used for the previous-trial-outcome input (6-a).

ii.
```python
    reward_frames = np.searchsorted(beh['time'], beh['reward_times'])
    reward_frames = np.clip(reward_frames, 0, n_frames - 1)
...
        any_reward = np.any((reward_frames >= st) & (reward_frames < sp_))
        any_rzone = np.any(beh['reward_zone'][st:sp_] > 0)
        is_reward[i] = int(bool(any_reward and any_rzone))
```

iii. As 6-a: this is `behav.get_trial_types` verbatim in logic. Step 2 records that the `Reward`
series is sparse with its own timestamps, and that `reward_zone` is "identically 0 on omission
trials".

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A per-trial binary (reward delivered *and* zone active) broadcast over the trial's timepoints.

ii.
```python
        out[5] = is_reward[i]
```

iii. Decoder spec: "Reward outcome, per-trial. 0 = no, 1 = yes". The agent validated the resulting
rate against the paper: Step 2 reports a rewarded fraction of 0.8466 (omission 0.1534) over the full
dataset, matching the Methods' "the reward was randomly omitted on ~15% of trials"; the conversion
script prints this comparison at the end of every run (`rewarded fraction: … (paper: ~85%, i.e. ~15%
omission)`).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled, and one important one is **not**:

- **Unpaired trial boundaries**: a leading `teleport` with no preceding `trial_start`, or a trailing
  `trial_start` with no `teleport`, is dropped; an assertion then checks every teleport follows its
  start.
- **Corrupted teleport-frame position**: the teleport frame is excluded from every trial by using a
  half-open `[trial_start, teleport)` window.
- **Pre-TTL sentinel values**: `environment == −1` samples are filtered out before taking the unique
  per-trial value.
- **Reward timestamps off the frame grid**: mapped with `searchsorted` and clipped into range.
- **Degenerate cells**: cells whose ΔF/F or events are non-finite are dropped; any residual
  within-trial non-finite value is zero-filled so the decoder never sees NaN.
- **Lick-sensor errors**: whole trial dropped (1-e).
- **Worker failures**: each session runs in a try/except that prints a traceback and returns an error
  record; the driver then reports every error and aborts.
- **NOT handled — fluorescence/behaviour frame-count mismatch.** 10 of 152 sessions store exactly one
  more imaging frame than behaviour samples. The script asserts equality and therefore raises on
  those sessions, and because the driver treats any per-session error as fatal, the **entire full
  conversion aborts** and no `converted_data.pkl` is written.

ii.
```python
    if len(teleports) and len(trial_starts) and teleports[0] < trial_starts[0]:
        teleports = teleports[1:]
    if len(trial_starts) > len(teleports):
        trial_starts = trial_starts[:len(teleports)]
    if len(teleports) > len(trial_starts):
        teleports = teleports[:len(trial_starts)]
    assert np.all(teleports > trial_starts), f'{path}: teleport before trial start'
```

```python
    n_frames = len(beh['time'])
    assert F.shape[1] == n_frames, f'{path}: F has {F.shape[1]} frames, behavior has {n_frames}'
```

```python
        act = events[:, st:sp_]
        if not np.all(np.isfinite(act)):
            # should never happen; guard so the decoder never sees NaN
            act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
```

```python
    errors = [r for r in results if 'error' in r]
    if errors:
        for e in errors:
            print('ERROR:', e['error'])
        raise SystemExit('conversion failed')
```

iii. The handled cases are documented in Step 4 (teleport-frame corruption, pre-TTL sentinels) and
Step 5 decisions 5 and 7. The frame-count mismatch was **discovered but never resolved**: the
trajectory shows the agent reading the failure (step 115), enumerating the 10 offending files (step
119: m17 ses-04/06 and m18 ses-01/05/07/10/11/12/13/14, each with F one frame longer than behaviour),
and then being cut off while killing the run. No fix (e.g. cropping both streams to the shorter
length, as the human reference does) was applied, and CONVERSION_NOTES Steps 6–13 were never written.

## 13-a. What are the most time-consuming steps of the code?

i. As instrumented by the script itself (a per-session `timing` dict with `read_behavior`,
`read_fluorescence`, `dff_events`, plus `total_time`), the two dominant costs are:

1. **Reading the raw fluorescence from HDF5** — `Fluorescence` and `Neuropil` are read in full
   (all ROIs, every frame) for every plane before the `iscell` subset is applied.
2. **`dff_and_events`** — the per-segment NaN-safe Gaussian smoothing, the 300-sample
   `minimum_filter1d`/`maximum_filter1d` pair, and the OASIS deconvolution, all run once per trial
   segment per session (≈80 segments × ~1000 cells).

Everything else (trial parsing, discretisation, assembling the dict) is negligible. Session totals
in the full run were ~4–15 s; the whole 152-session dataset processed in ~2.3 min wall clock with 12
worker processes. Pickling the ~5 GB result is the final non-trivial cost.

ii.
```python
        beh = load_behavior(f)
        t1 = time.time(); timing['read_behavior'] = t1 - t0
        F, Fneu, plane_of_cell = load_fluorescence(f)
        t2 = time.time(); timing['read_fluorescence'] = t2 - t1
...
    t3 = time.time()
    dff, events = dff_and_events(F, Fneu, trial_starts, teleports, keep_teleports, fs)
    timing['dff_events'] = time.time() - t3
```

```python
                print(f"[{len(results)}/{len(files)}] {r.get('session_key', r.get('error'))} "
                      f"cells={r.get('n_neurons')} trials={r.get('n_trials_kept')} "
                      f"t={r.get('total_time', 0):.1f}s "
                      f"| elapsed {elapsed/60:.1f} min, ETA {(len(files)-len(results))*rate/60:.1f} min",
```

iii. The instructions required timing information and a <15 min full run. The agent addressed the I/O
bottleneck by bypassing `pynwb` in favour of raw `h5py` reads, by making a single pass over each file
(no separate survey pass), and by distributing sessions across processes with a live ETA printout.
Note the per-step `timing` dict is stored in each result but never printed, so only `total_time` is
actually visible in the logs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Most of the obvious loops are already vectorised: the interneuron speed correlation is a single
matrix product rather than the reference's per-cell `np.corrcoef` loop; all discretisations are
array operations; the distance-to-zone computation uses boolean masks. The loops that remain are:

- The three `for start, stop in zip(start_inds, stop_inds)` loops in `dff_and_events` (copy,
  baseline, smooth+deconvolve). These are inherently per-segment because the maximin baseline must be
  computed independently within each trial; they could in principle be merged into one loop (saving
  two passes) or replaced with a padded/blocked representation.
- The per-trial task-variable loop (`is_reward`, `environment`, `lick_error`), which does three
  reductions per trial and could be done with `np.add.reduceat`-style segmented reductions.
- The per-trial export loop, which allocates and fills small arrays; this is unavoidable given
  variable-length trials and the required list-of-arrays output format.
- `load_fluorescence` reads and then boolean-subsets each plane in memory, deliberately, because
  fancy-indexing columns through HDF5 is much slower.

ii.
```python
    for start, stop in zip(start_inds, stop_inds):
        f_[:, start:stop] = F[:, start:stop]
        f_neu_[:, start:stop] = Fneu[:, start:stop]
...
    for start, stop in zip(start_inds, stop_inds):
        seg = f_[:, start:stop]
        ...
    for start, stop in zip(start_inds, stop_inds):
        smoothed = nansmooth(dff[:, start:stop], DFF_SMOOTH_SIGMA, axis=-1)
```

```python
    dv = dff[:, valid]
    dv = dv - dv.mean(axis=1, keepdims=True)
    sv = speed_valid - speed_valid.mean()
    denom = np.sqrt((dv ** 2).sum(axis=1) * (sv ** 2).sum())
    speed_corr = (dv @ sv) / denom
```

iii. The reference `pp.dff` uses exactly the same per-segment loop structure, so keeping it preserves
numerical equivalence with the paper's pipeline; the agent's efficiency gains were instead sought at
the process level (multiprocessing) and in the I/O layer. The vectorised correlation is an explicit
improvement over `spatial.is_putative_interneuron`'s per-cell loop and is numerically identical.

## 13-c. What processing does the code repeat multiple times?

i. Very little. The script makes a **single pass** over each NWB file — unlike the reference
approach, there is no separate "survey" pass, because the reward-zone labels come from the scene name
rather than from a dataset-wide inference that would need a first pass. Within a session the
repetitions are:

- `dff_and_events` walks the trial segments three times (copy, baseline, smooth+deconvolve) and
  computes `np.nanmean` of the neuropil per segment in the second pass.
- The trial segments are iterated twice more at session level: once for the task variables
  (`is_reward`/`environment`/`lick_error`) and once to build the per-trial arrays.
- `plot_processing` (only under `--show-processing`) recomputes the distance-to-zone series from raw
  position in order to check it against the exported bins — a deliberate, independent recomputation.

The one-off exploratory scans (`cache/scan_data.py`, `check_zones.py`, `check_lick_thresh.py`) each
read all 152 files, but they are separate analysis scripts, not part of the conversion.

ii.
```python
    for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
        any_reward = np.any((reward_frames >= st) & (reward_frames < sp_))
        ...
    for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
        if lick_error[i]:
            continue
        pos = beh['position'][st:sp_]
```

iii. The three-pass structure of `dff_and_events` is inherited from the reference `pp.dff` and kept
for fidelity. The split between the task-variable loop and the export loop exists because
`prev_outcome[i] = is_reward[i-1]` needs the complete `is_reward` vector before any trial can be
written out.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A few small items:

- `load_behavior` reads `scanning` and `trial number` from every file; neither is used anywhere in
  the conversion (the agent established in Step 2 that `scanning` is constant and that trial identity
  comes from `trial_start`/`teleport`). `autoreward` was checked during exploration and deliberately
  not read, since it is all zeros in all 152 files.
- `load_fluorescence` reads the **full** `F`/`Fneu` arrays including non-`iscell` ROIs before
  subsetting — a deliberate trade (documented in a comment) of extra I/O for much faster HDF5 access.
- The ΔF/F array `dff` is computed and retained for the whole session, but downstream only the
  deconvolved `events` are exported; `dff` is needed for the interneuron correlation and is otherwise
  only used by the optional plotting.
- `imaging_rate` is read but unused (the sampling rate is taken from the behaviour timestamps
  instead); `region` is read per session but collapsed to the single constant `'CA1'`.
- Interneuron cells are fully processed through ΔF/F and OASIS before being discarded — unavoidable,
  since the exclusion criterion is defined on their ΔF/F.

None of these is a material cost relative to the HDF5 reads and the deconvolution.

ii.
```python
        'trial_number': b['trial number/data'][:],
        'scanning': b['scanning/data'][:],
```

```python
        # read then subset: column fancy-indexing through hdf5 is far slower
        Fs.append(np.asarray(Fp[:, :], dtype=np.float32)[:, sel].T)
```

```python
        imaging_rate = float(f['general/optophysiology/ImagingPlane/imaging_rate'][()])
        region = f['general/optophysiology/ImagingPlane/location'][()].decode()
```

iii. The unused behaviour streams are cheap (one vector of ~20k floats each) and were kept from the
exploration phase; the full-ROI read is an explicitly justified speed trade-off; the `dff` retention
is required by the interneuron filter (Step 5 decision 6) and by the `--show-processing`
verification plots the instructions asked for.
