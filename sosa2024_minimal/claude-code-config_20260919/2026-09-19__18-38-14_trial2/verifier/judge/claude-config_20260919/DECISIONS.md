# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every `*.nwb` file under every sub-directory of `/app/data` is converted: 152 files, 11 subjects. Files are enumerated with `os.listdir` (one level deep) and sorted by subject number then session (experiment-day) number. Files are read **directly with `h5py`** rather than with `pynwb`, reading only the datasets the conversion needs (`processing/behavior/BehavioralTimeSeries/*`, `processing/ophys/Fluorescence`, `.../Neuropil`, `.../ImageSegmentation/PlaneSegmentation/iscell`, plus `identifier`, `general/session_id`, `general/subject/subject_id`). Sessions are converted in parallel (`multiprocessing`, 6–8 spawn workers), each worker caching its per-session result to `/app/.session_cache/<file>.pkl`; the parent then re-reads the caches **in the canonical file order** so session order is deterministic and independent of completion order. A session that raises or yields <2 trials is skipped with a printed message (none did: all 152 sessions are in the output).

ii.
```python
    files = []
    for sub in sorted(os.listdir(DATA_ROOT)):
        d = os.path.join(DATA_ROOT, sub)
        if os.path.isdir(d):
            files += [os.path.join(d, fn) for fn in sorted(os.listdir(d)) if fn.endswith('.nwb')]
    # order sessions by subject number then experiment day
    files.sort(key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                              int(re.search(r'ses-(\d+)', p).group(1))))
```
```python
    with h5py.File(path, 'r') as f:
        subject = f['general/subject/subject_id'][()].decode()
        exp_day = int(f['general/session_id'][()].decode())
        scene = f['identifier'][()].decode().rstrip('/').split('/')[-1]

        beh = f['processing/behavior/BehavioralTimeSeries']
        ts = beh['position/timestamps'][:]
        nframes = len(ts)
        ...
        plane_names = sorted(f['processing/ophys/Fluorescence'].keys())
        F = np.concatenate([f['processing/ophys/Fluorescence'][p]['data'][:nframes, :]
                            for p in plane_names], axis=1).T
```
```python
    ctx = get_context('spawn')
    with ctx.Pool(args.workers) as pool:
        for n, (path, cache, err) in enumerate(pool.imap_unordered(_worker, files), 1):
```

iii. From the trajectory: the agent first listed `/app/data` and dumped the full HDF5 tree of one file with `f.visititems`, then ran a survey over all 152 files (step 42) printing subject, session id, scene, #planes, #ROIs, #`iscell`, and #frames for every file, confirming the directory is one level deep, that all files parse, and that the released cohort is the 11 "switch" mice of the paper. It chose `h5py` over `pynwb` for speed/memory (it only ever materialises the arrays it needs), and used a disk cache plus multiprocessing after measuring (step 44) available RAM/CPU and estimating the output at ~9.7 GB of cell×sample float32 (step 64).

## 1-b. How are the data split into subjects (mice)?

i. One subject per `sub-<id>` directory. The subject list is built from the file paths (regex `sub-(m\d+)`), de-duplicated and sorted numerically, giving `['m3','m4','m7','m11',...,'m19']` (11 subjects). Each converted session also carries the `general/subject/subject_id` string read out of its own NWB file, and `subject_idx` is looked up from that string, so the index cannot drift from the file contents.

ii.
```python
    subjects = sorted({re.search(r'sub-(m\d+)', p).group(1) for p in files},
                      key=lambda s: int(s[1:]))
    ...
        subject_idx.append(subjects.index(res['subject']))   # res['subject'] read from the NWB file
    data['subjects'] = subjects
    data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
```

iii. The agent verified (step 42) that the directory name and the in-file `general/subject/subject_id` agree for all 152 files, and that there are 11 mice — the paper's n = 11 switch mice. The subject id `m<N>` is also the key it used to map onto the paper's internal `GCAMP<N>` animal names for the teleport-imaging table (see 2-b).

## 1-c. How are the data split into sessions?

i. One session = one NWB file = one experiment day. The experiment day is read from `general/session_id` (used for the `keep_teleports` lookup) and is also the `ses-<nn>` field of the filename (used for ordering). Sessions are emitted in (subject number, experiment day) order; `metadata['session_info']` records subject, experiment day, scene, environments, reward zones, trial and neuron counts for each one.

ii.
```python
        exp_day = int(f['general/session_id'][()].decode())
    ...
    files.sort(key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                              int(re.search(r'ses-(\d+)', p).group(1))))
    ...
    info = {'subject': subject, 'experiment_day': exp_day, 'scene': scene, ...}
```

iii. The survey in step 42 showed one scene/one day per file and a contiguous run of session numbers per mouse (m11 starting at day 3, consistent with the paper's statement that imaging for that mouse began later); the agent used the session number as the paper's "experiment day" because the `teleport_metadata.py` table is indexed by experiment day, and it cross-checked that this indexing selects 47 sessions, i.e. exactly the day lists in that table. Cross-day cell alignment (the paper's `multiDayROIAlign`) was deliberately not attempted — each session is treated as an independent population.

## 1-d. How are the data split into trials?

i. A trial is one lap: from the `trial_start` event frame to the `teleport` event frame. Frames are obtained as `flatnonzero(trial_start > 0)` and `flatnonzero(teleport > 0)`, and the code asserts the two have equal length. The emitted slice is `[trial_start - 1, teleport - 1)` — i.e. shifted one sample earlier than the raw event indices — because that is exactly the window the paper's `preprocessing.dff` carves out (`f_[:, start-1:stop-1]`); the neural, input and output streams are all cut with the same shifted indices, so all streams stay mutually aligned. Trial length is left variable (median 194 bins ≈ 12.5 s); nothing is padded or truncated.

ii.
```python
        trial_starts = np.flatnonzero(beh['trial_start/data'][:] > 0)
        teleports = np.flatnonzero(beh['teleport/data'][:] > 0)
    ...
    assert F.shape[1] == nframes and len(trial_starts) == len(teleports)
    ...
        # slice exactly as the reference dff() does: [trial_start-1, teleport-1)
        a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
        if b - a < 2:
            continue
        assert np.all(scanning[a:b] > 0)
```

iii. The agent inspected the behaviour streams around a trial boundary (step 28): position ramps from −5 cm up through 0 at `trial_start` and jumps to −50 cm one sample after `teleport`, so `[trial_start, teleport)` is the lap and the teleport sample itself is the interpolated jump the paper's code explicitly excludes. It checked globally (step 58) that starts and teleports are equal in number and strictly interleaved in all 152 sessions, that there are no NaNs in position/speed, and that `trial number` is an unreliable alternative (it is −1 during ITIs and increments at the teleport, not at the lap start). The one-sample shift was chosen to reproduce the paper's own `dff` window; the cost is that the first sample of each emitted trial sits at a position of about −2 cm (range −9.1 to +0.7 cm) instead of ≈0 cm.

## 1-e. How are trials filtered based on quality controls?

i. Three filters:
1. **Lick-sensor artefact laps are dropped** — the paper's `behavior.correct_lick_sensor_error` criterion: a lap in which >30% of imaging frames carry a cumulative lick count >2. This flags exactly 81 of 12,216 laps, the number the Methods report. The paper NaNs the lick trace on those laps; because lick is a required decoder output here, the agent drops the whole lap instead.
2. **The first lap of every session is dropped** (152 laps) because its `previous_trial_outcome` is unknowable: the Methods state mice ran 30 un-imaged warm-up trials immediately before the imaging session.
3. Degenerate guards: laps with <2 samples, laps whose deconvolved events are not all finite, and sessions left with <2 trials or 0 neurons are skipped. None of these actually triggered (11,983 = 12,216 − 152 − 81).

No minimum trial-length filter, no speed filter, no session-level or mouse-level exclusion.

ii.
```python
LICK_ERROR_THRESH = 0.30
...
        # behavior.correct_lick_sensor_error: capacitive sensor stuck on
        lick_error[i] = np.mean(lick[a:b] > 2) > LICK_ERROR_THRESH
...
    for i in range(ntrials):
        # The first imaged lap has no known preceding outcome (the ~30 un-imaged warm-up
        # laps are not in the file), so it cannot carry the "previous trial outcome" input.
        if i == 0:
            continue
        if lick_error[i]:
            continue
        a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
        if b - a < 2:
            continue
        ...
        neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
        if not np.all(np.isfinite(neu)):
            continue
...
    if len(neural_trials) < 2:
        return None
```

iii. The agent read `behavior.correct_lick_sensor_error` (default threshold 0.5, called with 0.3/0.35 elsewhere in the repo) and the Methods sentence "n = 81 out of 12,376 trials ... detected by >30% of the ... frame samples in the trial containing a cumulative lick count >2". It initially coded 0.35, then swept thresholds over the whole dataset (step 83: >0.30 → 81 trials, >0.35 → 69, >0.5 → 44), found 0.30 reproduces the paper's 81 exactly, and edited the constant (step 85). It flagged that one session (m4 day 14) loses 33 of 80 laps this way. For the first-lap rule it cited the Methods' 30 warm-up trials, saying it "preferred dropping 152 laps (1.2%) over inventing a label", and kept `trial_number` as the true ordinal so the drop is visible downstream. It explicitly rejected the paper's <2 cm/s speed filter because speed is itself a decoded output with a "<2 cm/s" class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces `processing/ophys/Fluorescence/plane<i>/data` (F) and `processing/ophys/Neuropil/plane<i>/data` (Fneu), pooled across imaging planes. The NWB `Deconvolved` field is deliberately **not** used.

ii.
```python
        plane_names = sorted(f['processing/ophys/Fluorescence'].keys())
        F = np.concatenate([f['processing/ophys/Fluorescence'][p]['data'][:nframes, :]
                            for p in plane_names], axis=1).T
        Fneu = np.concatenate([f['processing/ophys/Neuropil'][p]['data'][:nframes, :]
                               for p in plane_names], axis=1).T
```

iii. From the header of `convert_data.py` and the final report: "The NWB `Deconvolved` field is suite2p's own `spks` from raw F. The paper instead computes its own dF/F and deconvolves that." The agent had read `preprocessing.dff` and `utilities.multi_anim_sess` / `make_multi_anim_sess.md` (steps 13, 20, 54) and saw that the paper's `sess.timeseries['events']` — the signal used for place-cell detection, decoding and the GLM — is produced by `dff(..., deconvolve=True)` from F and Fneu.

## 2-b. How is the `neural` data processed?

i. A faithful re-implementation of `reward_relative.preprocessing.dff`: restrict samples to the per-lap segments (or lap+preceding teleport on sessions where the laser was not blanked), subtract `0.7 × Fneu`, add each segment's mean neuropil back, take a maximin baseline (Gaussian σ = 15 samples, then a 300-sample minimum filter followed by a 300-sample maximum filter ≈ the Methods' 20 s window), form `(F − F0)/|F0|`, smooth with a 2-sample Gaussian, and deconvolve each segment with `suite2p.extraction.dcnv.oasis(..., 2000, tau=0.7, 15.5078125 Hz)`. Samples outside the segments stay NaN. Planes are pooled before this (every step is per-cell, so pooling is numerically identical to per-plane processing). `keep_teleports` is taken per (mouse, experiment day) from the paper's `teleport_metadata.teleport_sessions` table (47 of 152 sessions).

ii.
```python
    if keep_teleports:
        start_inds = [int(trial_starts[0])] + (np.asarray(teleports[:-1]) + 2).tolist()
    else:
        start_inds = [int(s) for s in trial_starts]
    stop_inds = [int(s) for s in teleports]
    segments = [(a - 1, b - 1) for a, b in zip(start_inds, stop_inds)]
    ...
    f_ -= NEU_COEF * fneu_
    for a, b in segments:
        f_[:, a:b] += NEU_COEF * np.nanmean(fneu_[:, a:b], axis=1, keepdims=True)
        seg = ndi.gaussian_filter1d(f_[:, a:b], BASELINE_SMOOTH_SIG, axis=1)
        seg = ndi.minimum_filter1d(seg, BASELINE_WIN, axis=-1)
        seg = ndi.maximum_filter1d(seg, BASELINE_WIN, axis=-1)
        flow[:, a:b] = seg
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    for a, b in segments:
        dff[:, a:b] = ndi.gaussian_filter1d(dff[:, a:b], DFF_SMOOTH_SIG, axis=1)
        events[:, a:b] = dcnv.oasis(np.ascontiguousarray(dff[:, a:b], dtype=np.float32),
                                    2000, TAU, FRAME_RATE)
```
with
```python
FRAME_RATE = 15.5078125       # Hz, per imaging plane
NEU_COEF = 0.7 ; TAU = 0.7 ; BASELINE_WIN = 300
BASELINE_SMOOTH_SIG = 15 ; DFF_SMOOTH_SIG = 2
TELEPORT_SESSIONS = {'m10': [1, 7, 8, 14, 15], ..., 'm19': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17]}
```

iii. The agent read `preprocessing.dff` line by line (steps 13–14), `utilities.nansmooth` (step 68), the `dcnv.oasis` signature (step 70), `make_multi_anim_sess.md` for the call-site parameters `neu_coef = 0.7`, `baseline_method = 'maximin'` (step 54), `tau = 0.7` from the suite2p ops, and `teleport_metadata.py` (step 48) for the per-animal/day blanking table. It substituted plain `gaussian_filter1d` for the repo's `nansmooth` after checking the segments contain no NaN (so the NaN-weighting reduces to an identity). It checked the per-plane rate explicitly (step 56): the stored `rate` is 31.015625 Hz on the two-plane mice (m17/m18) and 15.5078125 Hz on the single-plane mice, i.e. one common per-plane rate of 15.5078125 Hz, and the behaviour `dt` is 0.06448363 s everywhere (step 58). It sanity-checked the output (step 79): events ≥ 0, ~75–78% exact zeros, max ≈ 0.85.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) suite2p manual curation: only ROIs with `iscell[:,0] > 0` are kept. (2) Putative interneurons are dropped: any surviving cell whose dF/F correlates with running speed at Pearson r > 0.5 (`int_thresh` in the paper's `dayData`). The correlation is computed over exactly the samples that are inside a processed segment (the non-NaN mask), vectorised rather than looped. Result: 138,288 neurons over 152 sessions (mean 910, range 154–2,320); interneurons removed are 0.33 ± 0.61% of curated cells.

ii.
```python
        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] > 0
    ...
    F = np.ascontiguousarray(F[iscell], dtype=np.float64)
    Fneu = np.ascontiguousarray(Fneu[iscell], dtype=np.float64)
    ...
    valid = ~np.isnan(dff[0, :])
    sp_v = speed[valid]
    dv = dff[:, valid]; dv = dv - dv.mean(axis=1, keepdims=True)
    spc = sp_v - sp_v.mean()
    denom = np.sqrt((dv ** 2).sum(axis=1) * (spc ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        speed_corr = (dv @ spc) / denom
    is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
    events = events[~is_int]
```

iii. The agent read `spatial.is_putative_interneuron` and grepped the repo for `int_thresh` (steps 48–51), finding that the analyses always pass 0.5 even though the function defaults to 0.3, and that the Methods quote "a Pearson correlation of >0.5". It then checked its own result against the paper: "This removed 0.33 ± 0.61% of cells — the paper reports 0.42 ± 0.85%" (step 99). It verified the `iscell` table is a single PlaneSegmentation covering both planes with `planeIdx` 0 then 1 (step 56), which is why concatenating planes in sorted order aligns with the `iscell` rows.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to lap start, which the trial slicing already provides: sample 0 of every trial is the frame `trial_start − 1`. No re-referencing, interpolation or windowing is applied; neural, input and output are cut with the same index pair, and `metadata` records `temporal_alignment_event` = lap start, `off_start = 0.0`, `off_end = None` (laps are self-paced, so there is no fixed end offset).

ii.
```python
        a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
        neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
        t_rel = (ts[a:b] - ts[a]).astype(np.float32)
```
```python
        'temporal_alignment_event': (
            'trial (lap) start: the VR frame at which the mouse enters the linear track at '
            'position 0 cm, after the inter-trial teleport period'),
        'off_start': 0.0,
        'off_end': None,
```

iii. Imaging frames and behaviour samples are the same clock (one behaviour row per imaging frame, `dt` constant at 0.06448363 s, verified across all sessions in step 58), so alignment to trial start needs nothing beyond slicing. The agent set `off_end = None` deliberately and added a `trial_duration_note` rather than truncating laps to a common length.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 64.4836 ms per bin (1000/15.5078125 Hz), the native per-plane imaging rate. **No** rebinning, resampling or interpolation is performed anywhere: neural and behavioural samples are used at the acquisition rate, and the rate is identical for every session, including the two-plane mice, once the plane interleave is accounted for.

ii.
```python
FRAME_RATE = 15.5078125       # Hz, per imaging plane
DT = 1.0 / FRAME_RATE         # s
...
        'time_bin_size': 1000.0 / FRAME_RATE,
```

iii. Step 56 confirmed the two-plane sessions store `rate = 31.015625` with two planes (→ 15.5078125 Hz/plane) and step 58 confirmed behaviour `dt` spans only 0.06448362720357181–0.06448362720493606 s over all 152 sessions, so a single constant bin size is exact for the whole dataset and no resampling is needed. The constant is also what is fed to OASIS as the deconvolution sampling rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. `processing/behavior/BehavioralTimeSeries/position/timestamps` (seconds). All behaviour series in these files share one timestamp vector, and it is also the imaging-frame clock.

ii.
```python
        ts = beh['position/timestamps'][:]
        nframes = len(ts)
    ...
        t_rel = (ts[a:b] - ts[a]).astype(np.float32)
        inp[0] = t_rel
```

iii. The agent checked (step 58) that `np.diff(ts)` is constant to 1e-12 across every session and equals the imaging period, and that the number of behaviour rows equals the number of imaging frames in 142 of 152 sessions (off by exactly one in the other 10, see 12), so the timestamp vector can be used directly as the common clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the trial's first timestamp, so every trial starts at 0 s and increases at 64.48 ms per bin. Stored as float32. Observed range 0–216.5 s.

ii.
```python
        t_rel = (ts[a:b] - ts[a]).astype(np.float32)
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = t_rel
```

iii. Direct reading of the requirement "Time from start of trial in seconds (continuous, time-varying)". No smoothing or rescaling was applied.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the same `[a, b)` index range slices `ts`, the events matrix and every behavioural stream, so element *k* of the time input is the same frame as column *k* of the neural matrix. The only alignment work is truncating the ophys arrays to the behaviour length when the NWB stores one extra imaging frame (10 sessions), plus an assertion that the two lengths then agree.

ii.
```python
        F = np.concatenate([f['processing/ophys/Fluorescence'][p]['data'][:nframes, :]
                            for p in plane_names], axis=1).T
    ...
    assert F.shape[1] == nframes and len(trial_starts) == len(teleports)
```

iii. The agent verified that the behaviour sampling period equals the per-plane imaging period and that behaviour and ophys share frame indexing (steps 56, 58, 60); the ten off-by-one sessions are all "ophys has exactly one more frame than behaviour", with all trials well inside the shorter length, so truncating is lossless.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioural time series (values −1 during the inter-trial interval, 0 = ENV 1, 1 = ENV 2 inside a lap).

ii.
```python
        env_ts = beh['environment/data'][:]
    ...
        env_vals = np.unique(env_ts[a:b])
        assert len(env_vals) == 1, f'{path}: trial {i} has environments {env_vals}'
        environment[i] = int(env_vals[0])
```

iii. Step 26 showed `environment` takes only {−1, 0} or {−1, 1} (and both 0 and 1 in the 11 day-8 environment-switch sessions), matching the paper's two visually distinct environments; step 72 verified that within every lap of every session the value is constant, so a per-trial scalar loses nothing.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Take the unique value over the raw lap window `[trial_start, teleport)` (which excludes the ITI, where the code is −1), assert it is unique, cast to int, and broadcast that constant across all timepoints of the trial as input row 1. Range in the converted data is 0–1.

ii.
```python
    environment = np.zeros(ntrials, dtype=np.int64)
    for i, (a, b) in enumerate(zip(trial_starts, teleports)):
        env_vals = np.unique(env_ts[a:b])
        assert len(env_vals) == 1, ...
        environment[i] = int(env_vals[0])
    ...
        inp[1] = environment[i]
```

iii. The environment code is read on the un-shifted lap window specifically so the ITI value of −1 (present at the sample just before `trial_start`) can never leak into the label; the assertion documents that the variable really is a per-trial constant. It is broadcast rather than stored as a scalar so that every input has the same `(d_input, n_timepoints)` shape.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The ordinal position of the lap within the session, i.e. the index into the `trial_start`/`teleport` event arrays. The NWB `trial number` series is not used.

ii.
```python
    ntrials = len(trial_starts)
    ...
    for i in range(ntrials):
        ...
        inp[2] = i
```

iii. Step 26 showed the stored `trial number` series is −1 during ITIs and increments at the teleport rather than at the lap start, so it does not index the laps the conversion emits; the loop index over `trial_start` events is the unambiguous lap ordinal.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the 0-based lap index across the trial. Crucially the **raw** index `i` is used, not a counter over surviving trials, so the ordinal still reflects the animal's true position in the session after the first lap and any lick-artefact laps have been dropped. Observed range in the converted data is 1–99 (1 rather than 0 because lap 0 is always dropped).

ii.
```python
        inp[2] = i
```

iii. Stated in the final report: "`trial_number` still carries the true ordinal (starts at 1)." Since experience within a session is the variable of interest (the paper's reward-zone switch happens at lap 30), renumbering after exclusions would have misrepresented the position of the switch.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` time series, using its **timestamps** only: `Reward/data` is identically zero in every file, so the reward amount carries no information. Reward timestamps are mapped to frame indices with `np.searchsorted` against the behaviour clock, and a lap counts as rewarded if any reward frame falls in `[trial_start, teleport)`.

ii.
```python
        reward_ts = beh['Reward/timestamps'][:]
    ...
    rew_frames = np.searchsorted(ts, reward_ts)
    rewarded = np.zeros(ntrials, dtype=np.int64)
    for i, (a, b) in enumerate(zip(trial_starts, teleports)):
        rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
```

iii. Step 26/28 showed `Reward/data` is all zeros with its own sparse timestamp vector; step 62 cross-checked the derived per-lap `isrew` against the `reward_zone` entries for three sessions and against the omission rate. The agent reported a mean reward rate of 84.7%, "consistent with the ~15% programmed omission rate" in the Methods. It also noted `autoreward` is identically zero in all 152 files, so the paper's automatic reward on the first ten post-switch laps cannot be recovered from this release.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial *i*, input row 3 is the binary `rewarded[i-1]` of the immediately preceding lap **as recorded**, broadcast across the trial. The preceding lap is the raw neighbour, so a dropped lick-artefact lap still supplies the previous outcome of the lap that follows it. Trial 0 is not given a fabricated value — it is dropped from the dataset entirely (see 1-e), because the preceding lap is one of the 30 un-imaged warm-up laps.

ii.
```python
        if i == 0:
            continue
        ...
        inp[3] = rewarded[i - 1]
```

iii. Final report: "The first imaged lap of each session, because the preceding lap is one of the un-imaged warm-up laps, so `previous_trial_outcome` is genuinely unknown — I preferred dropping 152 laps (1.2%) over inventing a label." Supported by the Methods: "Before the imaging session, mice were provided 30 'warm-up' trials using the task and reward zone from the previous day."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` behavioural series plus the active reward zone of that lap. The zone is **not** inferred from the `reward_zone` series; it is taken from the VR scene name stored in the NWB `identifier` (e.g. `Env1_LocationB_to_A`) together with the paper's fixed 30-lap switch point, exactly as `reward_relative.behavior.get_reward_zones` does, and mapped to the paper's coordinates (X/Y/Z → A/B/C = 80–130 / 200–250 / 320–370 cm).

ii.
```python
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
SWITCH_TRIAL = 30             # reward zone is moved after 30 trials (behavior.get_reward_zones)
...
def scene_reward_zones(scene, ntrials):
    m = re.fullmatch(r'Env\d_Location([ABC])', scene)
    if m:
        return [m.group(1)] * ntrials
    m = (re.fullmatch(r'Env\d_Location([ABC])_to_([ABC])', scene)
         or re.fullmatch(r'Env\d_([ABC])_to_Env\d_([ABC])', scene))
    if m:
        return [m.group(1)] * SWITCH_TRIAL + [m.group(2)] * (ntrials - SWITCH_TRIAL)
    raise ValueError(f'unrecognised scene name: {scene}')
...
        zstart, zstop = REWARD_ZONES[zone_labels[i]]
        p = pos[a:b]
```

iii. The agent read `behavior.get_reward_zones` (step 23) and `reward_zone_dict` (step 24), noting the label mapping A/B/C → X/Y/Z. It then validated the scene rule empirically over the entire dataset (step 72): for every lap in which the `reward_zone` sensor fires, it took the animal's position at zone entry and checked which of the three zones it is nearest — **zero mismatches in 152 sessions**, and every lap had a single environment. It also observed that `reward_zone` only fires on laps the animal entered/was rewarded, so it cannot label omission laps on its own, which is why the scene name (a session-level ground truth) was preferred. The 30-lap switch is the Methods' "Each switch occurred after 30 trials".

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from the animal's position to the nearest point of the active zone: negative before the zone (`pos − zone_start`), zero inside, positive after (`pos − zone_stop`). Computed per timepoint from the raw position trace, then discretised (7-c). Nothing is smoothed or clipped.

ii.
```python
def reward_zone_distance_bin(pos, zstart, zstop):
    d = np.zeros_like(pos)
    before = pos < zstart
    after = pos > zstop
    d[before] = pos[before] - zstart
    d[after] = pos[after] - zstop
```

iii. This is the "distance to any location in the reward zone" the instructions ask for, and it matches the paper's reward-relative distance convention (0 throughout the 50 cm zone). Zone edges come from the paper's `reward_zone_dict`.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned by explicit boolean masks rather than `digitize`, with the "inside the zone" class as the default: 0 for d < −50; 1 for −50 ≤ d < −10; 2 for −10 ≤ d < 0; 3 for d = 0 (inside); 4 for 0 < d ≤ 10; 5 for 10 < d ≤ 50; 6 for d > 50. The resulting class fractions are 0.255 / 0.102 / 0.074 / 0.239 / 0.021 / 0.072 / 0.238.

ii.
```python
    out = np.full(pos.shape, 3, dtype=np.int64)          # 3: inside the zone (d == 0)
    out[d < -50.0] = 0
    out[(d >= -50.0) & (d < -10.0)] = 1
    out[(d >= -10.0) & (d < 0.0)] = 2
    out[(d > 0.0) & (d <= 10.0)] = 4
    out[(d > 10.0) & (d <= 50.0)] = 5
    out[d > 50.0] = 6
```
```python
OUTPUT_VALUES[0] = ['< -50 cm', '-50 to -10 cm', '-10 to <0 cm', '0 cm (inside reward zone)',
                    '>0 to +10 cm', '+10 to +50 cm', '> +50 cm']
```

iii. The edges are the instructions' edges. Writing the masks out (instead of `np.digitize` with a 1e-6 epsilon edge) makes the "exactly 0 = inside the zone" class exact rather than dependent on a floating-point epsilon; the positive bins are closed on the right, which reads the instruction "4: >0 cm to +10 cm" literally.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same `pos[a:b]` slice used for every other stream, so it is sample-for-sample aligned with the neural matrix; no shifting, no interpolation.

ii.
```python
        p = pos[a:b]
        out = np.empty((6, T), dtype=np.int64)
        out[0] = reward_zone_distance_bin(p, zstart, zstop)
```

iii. Behaviour and imaging share one clock and one frame index (see 3-c), so identical indexing is sufficient alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural series (cm along the 450 cm virtual track), read directly.

ii.
```python
        pos = beh['position/data'][:]
    ...
        p = pos[a:b]
        out[1] = np.digitize(p, POSITION_EDGES)
```

iii. Step 26 showed `position` spans −500 to 450.8 cm over the whole recording, where the large negative values belong to the teleport/ITI period; inside a lap it runs from ≈0 to ≈450 cm (step 28 traced the ramp from −5 cm through 0 at lap start and the jump to −50 cm after the teleport), so the raw trace needs no correction.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing and discretisation — no unwrapping, smoothing or clipping. Because the trial slice begins one sample before `trial_start`, the first sample of each trial can be slightly negative (−9.1 to +0.7 cm); those samples fall in class 0, which is open on the left.

ii.
```python
POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]
...
        out[1] = np.digitize(p, POSITION_EDGES)
```

iii. The Methods' 450 cm track divided into the five equal bins the instructions specify; the raw values are already in centimetres on that track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with interior edges [90, 180, 270, 360], giving 5 classes of 90 cm each: 0 = <90, 1 = 90–180, 2 = 180–270, 3 = 270–360, 4 = >360. Edges are left-closed, and the first/last classes are open so that samples marginally outside [0, 450] are absorbed rather than forming extra classes. Observed fractions 0.216 / 0.176 / 0.232 / 0.227 / 0.149.

ii.
```python
POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]
...
        out[1] = np.digitize(p, POSITION_EDGES)
...
OUTPUT_VALUES[1] = ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm']
```

iii. Exactly the discretisation the instructions give ("5 equal-sized bins spanning the 450 cm track"); `np.digitize` on interior edges already returns 0-based class indices, so no offset correction is needed.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same slice, same indices as the neural matrix — nothing further.

ii.
```python
        a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
        neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
        p = pos[a:b]
```

iii. One behaviour row per imaging frame (verified in steps 56–60), so index equality is alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural series, a per-frame cumulative lick count (integer 0–8 in this dataset).

ii.
```python
        lick = beh['lick/data'][:]
    ...
        out[3] = (lick[a:b] > 0).astype(np.int64)
```

iii. Step 26 found `lick` takes integer values 0–6 in the first session inspected and up to 8 across the dataset (step 83), i.e. it is a count per frame, not a binary flag — which is also what the paper's artefact criterion ("cumulative lick count >2") assumes.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarised at >0 ("at least one lick detected in this imaging frame"). Laps flagged as lick-sensor artefacts are removed rather than binarised (see 1-e). Resulting class balance 0.778 / 0.222.

ii.
```python
        out[3] = (lick[a:b] > 0).astype(np.int64)
```
```python
        lick_error[i] = np.mean(lick[a:b] > 2) > LICK_ERROR_THRESH
```

iii. The instructions require a binary lick output; the artefact laps are exactly the ones on which the raw counts are untrustworthy (the paper NaNs them), so they are excluded instead of being turned into a spurious run of 1s.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[a, b)` slice as the neural data; no lead/lag correction.

ii.
```python
        out[3] = (lick[a:b] > 0).astype(np.int64)
```

iii. The lick sensor is sampled on the same VR frame clock as the imaging (constant `dt` = 64.48 ms verified across all sessions), so no resampling is required.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The scene name in the NWB `identifier` plus the 30-lap switch point — the same per-lap zone labels used for the distance output (see 7-a). Labels A/B/C are emitted as 0/1/2.

ii.
```python
        scene = f['identifier'][()].decode().rstrip('/').split('/')[-1]
    ...
    zone_labels = scene_reward_zones(scene, ntrials)
    ...
        out[4] = ZONE_LABELS.index(zone_labels[i])
```

iii. See 7-a: the rule comes from the paper's `behavior.get_reward_zones` and was validated against the `reward_zone` sensor on every lap of every session with zero mismatches. Class fractions come out at 0.331 / 0.336 / 0.332, i.e. the counterbalanced design the paper describes.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Parse the scene with a regex into one of three forms — constant location, within-environment switch (`Env<n>_Location<Z1>_to_<Z2>`), or environment+location switch (`Env<n>_<Z1>_to_Env<m>_<Z2>`) — then emit label 1 for the first 30 laps and label 2 thereafter; unrecognised scene names raise. The label is mapped A→0, B→1, C→2 and broadcast across all timepoints of the trial (time-varying storage of a per-trial constant, as the format prefers). `metadata['session_info']` records the zones and switch trial per session.

ii.
```python
    m = (re.fullmatch(r'Env\d_Location([ABC])_to_([ABC])', scene)
         or re.fullmatch(r'Env\d_([ABC])_to_Env\d_([ABC])', scene))
    if m:
        return [m.group(1)] * SWITCH_TRIAL + [m.group(2)] * (ntrials - SWITCH_TRIAL)
    raise ValueError(f'unrecognised scene name: {scene}')
...
ZONE_LABELS = ['A', 'B', 'C']
        out[4] = ZONE_LABELS.index(zone_labels[i])
        ...
        'reward_switch_trial': SWITCH_TRIAL if len(set(zone_labels)) > 1 else None,
```

iii. The three scene grammars are the ones the paper's `get_reward_zones` branches on; the `raise` guarantees an unexpected scene cannot be silently mislabelled. The agent enumerated the scenes over the dataset and the rule covered all 152 files.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` time series timestamps (the same per-lap `rewarded` vector used for the previous-trial input, see 6-a); `Reward/data` is unusable because it is identically zero.

ii.
```python
        reward_ts = beh['Reward/timestamps'][:]
    ...
    rew_frames = np.searchsorted(ts, reward_ts)
    rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
```

iii. Verified in steps 26/28 (`Reward/data` unique value = 0.0, separate sparse timestamps) and cross-checked per lap against the `reward_zone` entries and the overall ~15% omission rate in step 62.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward event times are converted to frame indices by `searchsorted` on the behaviour clock; a lap is rewarded (1) if at least one reward frame lies in `[trial_start, teleport)`, otherwise 0. The per-lap value is broadcast over all timepoints of the trial. Mean over the converted dataset: 0.843 rewarded.

ii.
```python
    rew_frames = np.searchsorted(ts, reward_ts)
    for i, (a, b) in enumerate(zip(trial_starts, teleports)):
        rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
    ...
        out[5] = rewarded[i]
```

iii. Reward is a per-lap event, so "any reward in the lap" is the natural binary outcome; the reward window uses the un-shifted lap boundaries so a reward cannot be attributed to the neighbouring lap. The resulting 84.7% rewarded matches the Methods' ~15% programmed omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Ophys/behaviour length mismatch** (10 sessions where the ophys arrays have exactly one frame more than the behaviour clock): the ophys arrays are truncated to the behaviour length at read time, then `assert F.shape[1] == nframes`.
- **`Reward/data` identically zero**: reward is taken from timestamps instead of amounts.
- **`autoreward` identically zero**: reported as not recoverable and left out of the conversion rather than encoded as a fake all-zero input.
- **ITI contamination**: environment and reward are read on the un-shifted lap window so the ITI sentinel (−1) cannot leak in; `assert np.all(scanning[a:b] > 0)` guarantees every emitted trial was actually being imaged (the first ~130 frames of each session, where `scanning` is negative, always precede the first lap).
- **Degenerate trials/sessions**: trials shorter than 2 samples or containing a non-finite event value are skipped; sessions yielding 0 neurons or <2 trials return `None`; any session that raises is caught in the worker, reported with a traceback and excluded rather than aborting the run.
- **Missing per-lap reward-zone sensor data** is a non-issue by construction, because the zone label comes from the scene name, not the sensor.

ii.
```python
        F = np.concatenate([f['processing/ophys/Fluorescence'][p]['data'][:nframes, :]
                            for p in plane_names], axis=1).T
    assert F.shape[1] == nframes and len(trial_starts) == len(teleports)
    ...
        assert np.all(scanning[a:b] > 0)
        neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
        if not np.all(np.isfinite(neu)):
            continue
    ...
    if len(neural_trials) < 2:
        return None
    ...
    try:
        res = convert_session(path)
    except Exception:
        return path, None, traceback.format_exc()
```

iii. The agent enumerated the anomalies before writing the code: step 58 checked, for all 152 sessions, the timestamp regularity, start/teleport counts and ordering, NaNs in position/speed (none), and the `scanning < 0` prefix; step 60 printed the ten length-mismatched files and confirmed the extra frame is trailing and that all laps lie inside the shorter length; step 64 confirmed `autoreward` is zero in every session. The assertions are deliberately loud: the agent preferred a session to fail visibly (and be reported) over silently emitting misaligned data.

## 13-a. What are the most time-consuming steps of the code?

i. Per session, the dominant cost is the neural pipeline in `compute_dff_and_events` — the 300-sample min/max filters and the per-segment OASIS deconvolution over ~1,000–2,300 cells × ~20,000 frames — followed by reading the raw `Fluorescence`/`Neuropil` arrays out of HDF5 (~5–8 s per session single-threaded, ~5.4 GB peak RSS). Globally, the other large cost is serialisation: ~9.5 GB is pickled to the per-session cache and then re-read and re-pickled into `converted_data.pkl`. The agent mitigated the compute cost with a 6–8-way process pool, bringing the whole conversion to 2 m 21 s wall clock (241 min CPU).

ii.
```python
    ctx = get_context('spawn')
    with ctx.Pool(args.workers) as pool:
        for n, (path, cache, err) in enumerate(pool.imap_unordered(_worker, files), 1):
```
```python
def _worker(path):
    cache = os.path.join(CACHE_DIR, os.path.basename(path).replace('.nwb', '.pkl'))
    if os.path.exists(cache):
        return path, cache, None
```

iii. The agent timed single sessions before the full run (step 79: 7.7 s and 5.6 s, 5.4 GB peak RSS) and checked machine limits first (step 44: free RAM, CPUs, disk), then sized the worker pool so that peak memory stayed within budget. The cache also makes the run restartable, which is why sessions are re-read from disk instead of being kept in memory.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain, all cheap or unavoidable:
- the three per-segment loops in `compute_dff_and_events` (~80 laps/session). They cannot be collapsed into one array operation because each lap gets its own baseline; the filters themselves are already vectorised across cells.
- the per-trial bookkeeping loop computing `rewarded`, `environment` and `lick_error` — could be done with `np.add.reduceat`/`searchsorted` over the whole session, but it is ~80 iterations of small slices.
- the trial-assembly loop. Variable trial length makes a fully vectorised version awkward, and its per-trial work (`digitize`, boolean masks) is already vectorised over time.
The inner numerics — the speed/dF/F correlation over all cells, the distance binning, the position/speed digitisation — are already written as array operations (the interneuron correlation in particular is a single matrix-vector product rather than the paper's per-cell `np.corrcoef` loop).

ii.
```python
    dv = dv - dv.mean(axis=1, keepdims=True)
    spc = sp_v - sp_v.mean()
    denom = np.sqrt((dv ** 2).sum(axis=1) * (spc ** 2).sum())
    speed_corr = (dv @ spc) / denom
```
```python
    for i, (a, b) in enumerate(zip(trial_starts, teleports)):
        rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
        env_vals = np.unique(env_ts[a:b])
        ...
```

iii. Not discussed explicitly in the trajectory; the vectorised correlation and the choice to keep per-lap loops follow from the paper's per-lap baseline definition, which is intrinsically sequential over laps.

## 13-c. What processing does the code repeat multiple times?

i. Little: each NWB file is opened exactly once and each session's neural pipeline runs once. What is repeated is (a) iterating the lap segments three times inside `compute_dff_and_events` (neuropil add-back + baseline, then dF/F smoothing + deconvolution, plus the initial masking loop), which could be one pass; (b) the per-trial loop over `trial_starts`/`teleports` runs twice — once to compute the per-trial scalars and once to assemble trials; and (c) the pickle round-trip, where every session's arrays are written to the cache and read back to build the final dictionary. There is no second "survey" pass over the dataset: the only session-level fact needed up front (the reward-zone label) is derived from the scene name inside the same pass.

ii.
```python
    for a, b in segments:
        f_[:, a:b] = F[:, a:b]
        fneu_[:, a:b] = Fneu[:, a:b]
    ...
    for a, b in segments:
        f_[:, a:b] += NEU_COEF * np.nanmean(fneu_[:, a:b], axis=1, keepdims=True)
        ...
    for a, b in segments:
        dff[:, a:b] = ndi.gaussian_filter1d(dff[:, a:b], DFF_SMOOTH_SIG, axis=1)
        events[:, a:b] = dcnv.oasis(...)
```

iii. The segment loops are kept separate because that is the order of operations in the paper's `dff` (masking → neuropil correction → baseline → smoothing → deconvolution), and the agent's stated priority was to reproduce that function faithfully. The cache round-trip is the deliberate price of parallelism plus restartability.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Some, all minor:
- **dF/F for every curated cell** is computed in full although the dF/F itself is never exported — it is needed only to produce the events and to run the speed-correlation interneuron test. Likewise, dF/F and events are computed for the cells that are then dropped as interneurons (~0.33%).
- **The cache round-trip**: ~9.5 GB written to `/app/.session_cache` and read back, then written again to `converted_data.pkl`.
- **Outputs are stored as `int64`** (six small-cardinality classes per timepoint) where `int8` would do; ~0.12 GB of the file, negligible against the float32 neural arrays but free to save.
- **Per-trial constants are broadcast to full length** (environment, trial number, previous outcome, reward zone, reward outcome are five of the ten input/output rows), which the format explicitly prefers but which the decoder could consume as scalars.
- **Diagnostic metadata** (`n_trials_lick_artifact`, `n_neurons_putative_interneuron`, `frac_rewarded`, and other `session_info` fields, plus the `scanning` assertion read) is computed for every session and never used by the decoder.

ii.
```python
    dff, events = compute_dff_and_events(F, Fneu, trial_starts, teleports, keep_teleports)
    ...
    is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
    events = events[~is_int]
    del dff, dv
```
```python
        out = np.empty((6, T), dtype=np.int64)
```
```python
    tmp = cache + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache)
```

iii. The dF/F must be computed regardless (it is the input to the deconvolution and the interneuron test), and the code frees it as soon as the filter has run (`del dff, dv`) to hold peak RSS near 5.4 GB. The cache exists so the 152-session run can be parallelised and resumed; the agent deleted it after the run (step 111). The extra metadata was kept deliberately — it is what let the agent check its own curation against the paper's reported numbers (81 artefact laps, 0.42 ± 0.85% interneurons, ~15% omissions).
