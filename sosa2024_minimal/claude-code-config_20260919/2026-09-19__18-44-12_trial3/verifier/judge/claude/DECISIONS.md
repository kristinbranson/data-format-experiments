# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every NWB file under `/app/data/sub-*/` is converted: the AI globs `sub-*/*.nwb`, sorts the list numerically by subject number and then by session (`ses-NN`) number, and hands each file to a worker process. All 152 files (11 mice × 14 days, minus the 2 days m11 was not imaged) are loaded. Files are opened directly with `h5py` rather than `pynwb`, reading the HDF5 paths (`processing/behavior/BehavioralTimeSeries/...`, `processing/ophys/Fluorescence/planeN/data`, `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`, `general/subject/subject_id`, `general/session_id`, `identifier`, `general/optophysiology/ImagingPlane/location`). Conversion is parallelised over 12 processes with a `spawn` context. The final run reported `152 sessions, 11 subjects, 12,135 trials, 138,276 neurons`.

ii.
```python
def convert_session(path):
    with h5py.File(path, 'r') as f:
        scene = f['identifier'][()].decode().split('/')[-1]
        subject = f['general/subject/subject_id'][()].decode()
        day = int(f['general/session_id'][()].decode())
        region = f['general/optophysiology/ImagingPlane/location'][()].decode()

        beh = f['processing/behavior/BehavioralTimeSeries']
        pos = beh['position/data'][:]
        speed = beh['speed/data'][:]
        lick = beh['lick/data'][:]
        rzone = beh['reward_zone/data'][:]
        tstart = beh['trial_start/data'][:]
        teleport = beh['teleport/data'][:]
        stamps = beh['position/timestamps'][:]
        reward_times = beh['Reward/timestamps'][:]

        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = seg['iscell'][:, 0].astype(bool)

        planes = sorted(f['processing/ophys/Fluorescence'].keys())
        Fs, Fneus, roi_ids = [], [], []
        for p in planes:
            Fs.append(f[f'processing/ophys/Fluorescence/{p}/data'][:].T)
            Fneus.append(f[f'processing/ophys/Neuropil/{p}/data'][:].T)
            roi_ids.append(f[f'processing/ophys/Fluorescence/{p}/rois'][:])
```
```python
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')),
                   key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                                  int(re.search(r'ses-(\d+)', p).group(1))))
    ...
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as ex:
        for i, res in enumerate(ex.map(convert_session, files)):
```

iii. From the trajectory, the AI first ran an exploratory survey over all `/app/data/*/*.nwb` files (step 44) to establish that every file has the same group layout, the same behaviour time series, one `PlaneSegmentation` shared across planes, and a per-file scene identifier. It chose raw `h5py` over `pynwb` because it only needs a handful of datasets per file and the whole conversion is I/O-bound (the full run took 2 min 26 s wall on 12 workers). It also verified (step 91) that `PlaneSegmentation/id` and the concatenated `rois` indices are exactly `arange(n_rois)` in every file, which is what makes the `h5py`-level indexing of `iscell` safe.

## 1-b. How are the data split into subjects (mice)?

i. One subject per `sub-m<N>` directory. The subject id is not parsed from the path but read from each file's `general/subject/subject_id`; the unique set is sorted numerically to give `subjects`, and `subject_idx` indexes into it per session. 11 subjects result.

ii.
```python
        subject = f['general/subject/subject_id'][()].decode()
```
```python
    subjects = sorted({r['subject'] for r in results},
                      key=lambda s: int(re.sub(r'\D', '', s)))
    ...
        'subject_idx': np.array([subjects.index(r['subject']) for r in results],
                                dtype=np.int64),
```

iii. The AI's exploration confirmed that the 11 `sub-*` directories correspond to the 11 "switch task" mice of the paper, and that the in-file `subject_id` agrees with the directory name. It preferred the in-file field as the authoritative identifier. In its summary it notes the mapping `m<N>` ↔ the paper's internal `GCAMP<N>` and that m11 lacks days 1–2 "which were never imaged".

## 1-c. How are the data split into sessions?

i. One session per NWB file. Sessions are ordered by the `ses-NN` number in the filename, which the AI treats as the experiment day (it also stores `exp_day` from `general/session_id` and the VR `scene` name in per-session metadata). No cross-session neuron alignment is attempted; each session has its own neuron set.

ii.
```python
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')),
                   key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                                  int(re.search(r'ses-(\d+)', p).group(1))))
```
```python
        day = int(f['general/session_id'][()].decode())
    ...
    info = {'subject': subject, 'exp_day': day, 'scene': scene, ...}
```

iii. The survey in the trajectory printed `sid` (`general/session_id`) alongside the filename for every file and showed that they agree, and that each file holds one continuous imaging session with ~80 laps. The scene identifier (e.g. `Env1_LocationB_to_A`) is per file, which the AI then uses for the per-session task condition, so "one file = one session" is the natural unit.

## 1-d. How are the data split into trials?

i. A trial is one lap: from the frame where `trial_start == 1` (inclusive) to the frame where `teleport == 1` (exclusive). The inter-trial teleport/grey period is not included in any trial. The code asserts that the numbers of starts and teleports match and that every teleport follows its start, and clips the teleport index to the number of usable frames.

ii.
```python
    starts = np.where(tstart == 1)[0]
    stops = np.where(teleport == 1)[0]
    assert len(starts) == len(stops) and np.all(stops > starts)
    stops = np.minimum(stops, nframes)
    ntrials = len(starts)
```
```python
    for t, (s, e) in enumerate(zip(starts, stops)):
        ...
        neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. This is the trial window the paper's own code uses (`behavior.get_trial_types` slices `sess.trial_start_inds[trial] : sess.teleport_inds[trial]`), which the AI read at step 50/steps 25–50. The module docstring records the choice: "Trials: one lap of the virtual track, from the `trial_start` frame (inclusive) to the `teleport` frame (exclusive), i.e. the ITI/teleport period is not included." The exploratory survey confirmed 80 `trial_start` and 80 `teleport` events per session and that `trial number` is constant within each such window.

## 1-e. How are trials filtered based on quality controls?

i. One trial-level filter: trials with a stuck lick sensor are dropped entirely. The criterion is the paper's — more than 30% of the frames in the trial have a cumulative lick count > 2. 81 of 12,216 trials are removed (12,135 kept). No minimum-trial-length filter is applied (the shortest surviving trial is 96 frames ≈ 6 s). No sessions or mice are excluded.

ii.
```python
LICK_ERROR_FRAC = 0.3     # >30% of samples in a trial with cumulative lick count > 2
LICK_ERROR_COUNT = 2      # => stuck lick sensor, trial discarded
```
```python
    for t, (s, e) in enumerate(zip(starts, stops)):
        licks = lick[s:e]
        # trials with a stuck lick sensor (Methods, "Quantification of licking
        # behavior") -- licking is a decoder output, so the trial is discarded
        if np.mean(licks > LICK_ERROR_COUNT) > LICK_ERROR_FRAC:
            n_lick_error += 1
            continue
```

iii. The AI located `behavior.correct_lick_sensor_error` in the paper's repo (step 50) and the matching Methods sentence: "A very small number of trials with erroneous lick detection … (~0.65% of all imaged trials, n = 81 out of 12,376 trials removed across 11 switch mice). These trials were detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2". Its count of 81 reproduces the paper's number exactly. It reports the reason for dropping the whole trial rather than NaN-ing the lick trace as the paper does: "dropping rather than NaN-ing them is required because lick is a decoder output" — the target format has no representation for a missing categorical output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces `processing/ophys/Fluorescence/plane*/data` (F) and `processing/ophys/Neuropil/plane*/data` (Fneu), with ROI curation from `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`. The NWB's stored `Deconvolved` array is deliberately **not** used. For the two-plane mice the per-plane F/Fneu arrays are concatenated along the cell axis.

ii.
```python
        planes = sorted(f['processing/ophys/Fluorescence'].keys())
        Fs, Fneus, roi_ids = [], [], []
        for p in planes:
            Fs.append(f[f'processing/ophys/Fluorescence/{p}/data'][:].T)
            Fneus.append(f[f'processing/ophys/Neuropil/{p}/data'][:].T)
            roi_ids.append(f[f'processing/ophys/Fluorescence/{p}/rois'][:])
    F = np.concatenate(Fs, axis=0)
    Fneu = np.concatenate(Fneus, axis=0)
```

iii. From the module docstring and final summary: "Deconvolved activity is what the paper uses for its own decoder, GLM and place-cell analyses. The NWB's stored `Deconvolved` is suite2p's spks off raw F, not this pipeline, so it wasn't used." The AI checked this empirically at step 58: its own re-computed events correlate only r = 0.41 with the stored `Deconvolved` trace, confirming they are different signals.

## 2-b. How is the `neural` data processed?

i. dF/F is recomputed from F and Fneu following `reward_relative.preprocessing.dff` and then deconvolved with OASIS. Per trial window `[trial_start, teleport)`: subtract `0.7 × Fneu`; add the trial-mean neuropil back so the ratio is a true dF/F; take a maximin baseline (Gaussian smoothing σ = 15 samples, then a 300-sample running minimum followed by a 300-sample running maximum ≈ the Methods' 20 s window); form `(F − baseline)/|baseline|`; smooth with a 2-sample s.d. Gaussian; deconvolve with `suite2p.extraction.dcnv.oasis` at `tau = 0.7` and the per-plane frame rate `rate/n_planes`. Everything outside a lap stays NaN/0 and is never emitted. Planes are pooled after processing. One deviation from the paper's code: the baseline window is *always* restricted to the lap; the repo's per-mouse/per-day `keep_teleports` table (sessions where the laser was not blanked, so the baseline may span the teleport) is not used.

ii.
```python
def compute_dff_events(F, Fneu, starts, stops, fs):
    f = np.full(F.shape, np.nan, dtype=np.float64)
    fneu = np.full(F.shape, np.nan, dtype=np.float64)
    for s, e in zip(starts, stops):
        f[:, s:e] = F[:, s:e]
        fneu[:, s:e] = Fneu[:, s:e]

    # neuropil subtraction
    f -= NEU_COEF * fneu

    dff = np.zeros(F.shape, dtype=np.float64)
    events = np.zeros(F.shape, dtype=np.float32)
    for s, e in zip(starts, stops):
        trial = f[:, s:e] + NEU_COEF * np.nanmean(fneu[:, s:e], axis=1, keepdims=True)
        base = gaussian_filter1d(trial, BASELINE_SMOOTH, axis=1)
        base = minimum_filter1d(base, BASELINE_WIN, axis=-1)
        base = maximum_filter1d(base, BASELINE_WIN, axis=-1)
        d = (trial - base) / np.abs(base)
        d = gaussian_filter1d(d, DFF_SMOOTH, axis=1)
        dff[:, s:e] = d
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(d, dtype=np.float32),
                                    2000, TAU, fs)
    return dff, events
```
```python
NEU_COEF = 0.7            # neuropil coefficient
BASELINE_WIN = 300        # samples (~20 s at 15.5 Hz) for the maximin baseline
BASELINE_SMOOTH = 15      # samples, pre-baseline smoothing
DFF_SMOOTH = 2            # samples s.d. Gaussian on dF/F (~0.129 s)
TAU = 0.7                 # calcium kernel decay for OASIS deconvolution
```
```python
        fs = float(f[f'processing/ophys/Fluorescence/{planes[0]}/starting_time']
                   .attrs['rate']) / len(planes)
```

iii. The AI read `preprocessing.py` (steps 16–18) and reproduced it line-for-line, then validated the re-implementation on one session at step 58 (dF/F range −0.34 to 3.25, events sparse and non-negative). It justifies the parameters from the paper: neuropil coefficient 0.7 and `baseline_method='maximin'` from the repo, `tau = 0.7` from the suite2p ops, and the 2-sample Gaussian and 20 s window from the Methods. For the teleport question it states the Methods reading directly: "Baselines are computed within each trial only (the ITI/teleport period is excluded), as in the paper" and "Baselines are per-trial so the blanked-laser teleport period never enters them" — i.e. it took the Methods sentence "baseline fluorescence was calculated within each trial independently" at face value rather than following the repo's per-session `teleport_metadata` exception. It also noted the per-plane rate issue explicitly: "the NWB `rate` attribute is the scan rate over all planes, so divide by the number of simultaneously imaged planes".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both from the paper. (1) Only ROIs with `iscell == 1` (the authors' manual suite2p curation) are kept. (2) Putative interneurons are then removed: any cell whose within-lap dF/F has Pearson r > 0.5 with the animal's running speed. 402 cells (0.29%) are dropped by the second filter, leaving 138,276 neurons (154–2,323 per session). No session or neuron-count-based session exclusion.

ii.
```python
    keep = iscell[roi_ids]
    F, Fneu = F[keep], Fneu[keep]
```
```python
    # drop putative interneurons: dF/F correlated with running speed
    inside = np.zeros(nframes, dtype=bool)
    for s, e in zip(starts, stops):
        inside[s:e] = True
    d = dff[:, inside]
    sp = speed[:nframes][inside]
    d = d - d.mean(axis=1, keepdims=True)
    spc = sp - sp.mean()
    denom = np.sqrt((d ** 2).sum(axis=1) * (spc ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (d @ spc) / denom
    good = ~(r > INTERNEURON_R)
    events = events[good]
```

iii. Both filters are quoted in the docstring against the Methods ("Calcium data processing"). The AI found `spatial.is_putative_interneuron` (steps 70–71), noticed its signature default is `r_thresh=0.3`, and chose 0.5 because that is the value the Methods quote and the value `dayData` passes. It sanity-checked the outcome against the paper's reported figure: "402 cells, 0.29%, vs the paper's 0.42 ± 0.85%", and the per-session cell counts against the paper's stated 155–2172 range.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instruction is to align to trial start, which is exactly the left edge of the trial slice, so no extra alignment step is needed: the trial's neural matrix is `events[:, trial_start:teleport]` and its first column is the trial-start frame. Metadata records `temporal_alignment_event = 'trial start (entry into the virtual linear track)'`, `off_start = 0.0`, `off_end = None` (variable-length trials).

ii.
```python
            neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```
```python
            'temporal_alignment_event': 'trial start (entry into the virtual linear track)',
            'off_start': 0.0,
            'off_end': None,  # trials run until teleport, so their length varies
```

iii. Nothing is said beyond the docstring's trial definition; the alignment is implicit in cutting trials at `trial_start`. The AI's sample-trial plot check (step 89) confirmed that position and distance-to-zone start at the beginning of the track in every plotted trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. Data stay at the native per-plane imaging rate, 15.5078125 Hz → 64.4836 ms bins. For the two-plane mice the stored `rate` of 31.015625 Hz is divided by the number of planes so the effective bin is the same for every session; the deconvolution also receives this per-plane rate. `metadata['time_bin_size']` is written as `1000/15.5078125`.

ii.
```python
        fs = float(f[f'processing/ophys/Fluorescence/{planes[0]}/starting_time']
                   .attrs['rate']) / len(planes)
```
```python
            'time_bin_size': 1000.0 / 15.5078125,
            ...
            'sampling_rate_hz': 15.5078125,
```

iii. The exploratory survey (step 44) recorded `rate` and `nplanes` for all 152 files and found only two values, 15.5078125 (1 plane) and 31.015625 (2 planes), i.e. a single common per-plane rate; step 46 confirmed the behaviour timestamps step by exactly 0.06448363 s = 1/15.5078125 in a two-plane session too. Since all sessions already share a bin size, no resampling is required and none is done.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is not read from a stored time series: it is constructed from the frame index within the trial and the per-plane frame rate `fs`, i.e. `0, 1/fs, 2/fs, …`. The raw quantities involved are the trial boundaries (`trial_start`, `teleport`) and the imaging `rate`/`n_planes` attribute.

ii.
```python
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = np.arange(T, dtype=np.float32) / fs
```

iii. The AI verified at step 46 that the behaviour `timestamps` are exactly uniform with `diff = 0.06448363 s = 1/15.5078125` and that the behaviour clock is the imaging frame clock ("the VR streams already interpolated to the imaging frame clock in the NWB files (~15.5 Hz)", module docstring). Given that, the frame index divided by the rate is identical to the stored timestamps re-zeroed at trial start, and avoids carrying the absolute session clock around.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. None beyond the construction above: the vector starts at 0.0 on the trial-start frame and increments by one frame period. The verification run reports the range as [0.0, 216.5] s across all trials, matching the longest trial (3,359 frames).

ii.
```python
        inp[0] = np.arange(T, dtype=np.float32) / fs
```

iii. The value is by construction "time from start of trial", so no re-zeroing step is needed. Stored as float32 like the rest of the input block.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the vector has exactly `T = e - s` entries, the same `T` as the trial's neural matrix, and entry 0 corresponds to the same frame as neural column 0. Behaviour and imaging are on the same frame clock in the NWB files; where they differ in length (10 sessions, by one frame) the code takes the overlap.

ii.
```python
    nframes = min(F.shape[1], len(pos))
    F, Fneu = F[:, :nframes], Fneu[:, :nframes]
    ...
    stops = np.minimum(stops, nframes)
```
```python
        T = e - s
        ...
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = np.arange(T, dtype=np.float32) / fs
```

iii. Step 52 of the trajectory enumerated every file where the behaviour and ophys frame counts disagree — 10 sessions, always by exactly one frame — and the AI handled them by cropping to the shorter of the two ("behaviour and imaging occasionally differ by one frame; use the overlap"). Because every stream is indexed with the same `[s:e]` slice, alignment needs nothing further.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. From the session's VR scene name in `identifier` (e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`), parsed to `Env1` → 0 / `Env2` → 1, with the environment changing after trial 30 on the day-8 "environment switch" scenes (`Env1_C_to_Env2_A`). The `environment` behaviour time series is *not* used as the source, but was used to validate the parse.

ii.
```python
def parse_scene(scene, ntrials, change_trial=CHANGE_TRIAL):
    m = re.fullmatch(r'Env(\d)_Location([ABC])', scene)
    if m:
        env = int(m.group(1)) - 1
        return [m.group(2)] * ntrials, [env] * ntrials
    m = re.fullmatch(r'Env(\d)_Location([ABC])_to_([ABC])', scene)
    if m:
        env = int(m.group(1)) - 1
        zones = [m.group(2)] * change_trial + [m.group(3)] * (ntrials - change_trial)
        return zones[:ntrials], [env] * ntrials
    m = re.fullmatch(r'Env(\d)_([ABC])_to_Env(\d)_([ABC])', scene)
    if m:
        zones = [m.group(2)] * change_trial + [m.group(4)] * (ntrials - change_trial)
        envs = ([int(m.group(1)) - 1] * change_trial
                + [int(m.group(3)) - 1] * (ntrials - change_trial))
        return zones[:ntrials], envs[:ntrials]
    raise ValueError(f'unrecognized scene name: {scene}')
```
```python
    zones, envs = parse_scene(scene, ntrials)
    ...
        inp[1] = envs[t]
```

iii. This mirrors `reward_relative.behavior.get_reward_zones`/`get_trial_types`, which derive the session condition from `sess.scene`. The AI cross-checked the parse against the raw `environment` time series over the whole dataset in step 52 and reported "total envmis 0" across all 12,216 trials, so the scene-derived label is identical to the recorded one. The raw series carries −1 during the ITI, which is another reason to prefer the per-trial scene label.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Only the regex parse plus the "switch after trial 30" rule; the resulting 0/1 scalar is broadcast across all timepoints of the trial, so the input row is a constant time series.

ii.
```python
        inp[1] = envs[t]
```
```python
CHANGE_TRIAL = 30         # reward zone switches after 30 trials (Methods)
```

iii. The instructions call for a binary ENV1/ENV2 per-trial input; the AI emits it as a time-varying row of the constant value so all four input rows share the `(4, T)` shape. The change trial of 30 is the paper's default in `get_reward_zones(change_trial=30)` and was validated by the zero-mismatch check above.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The within-session lap index `t`, i.e. the enumeration of the `trial_start`/`teleport` pairs. The stored `trial number` behaviour series is not used.

ii.
```python
    for t, (s, e) in enumerate(zip(starts, stops)):
        ...
        inp[2] = t
```

iii. The AI's exploration (step 48) printed `np.unique(tn[s:e])` for the first laps and found `trial number` to be constant and equal to the lap index within a trial, so the loop index carries the same information without depending on a separate stream. Note that the index is the *original* lap number: a trial dropped for a stuck lick sensor leaves a gap in the sequence rather than renumbering the later trials, which keeps trial number a faithful measure of how far into the session the animal is.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None; the raw integer index (0-based) is broadcast across all timepoints of the trial as a float32 row. The verification run reports the range as [0, 99].

ii.
```python
        inp = np.empty((4, T), dtype=np.float32)
        ...
        inp[2] = t
```

iii. No normalisation is applied, matching the instruction that trial number is a continuous per-trial input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the per-trial `rewarded` vector, which is computed from the `Reward` time series' own timestamps together with the `reward_zone` behaviour series: a trial counts as rewarded if a reward event falls inside the trial's timestamp window **and** the reward zone signal was active at some point in the trial. `input[3]` for trial *t* is `rewarded[t-1]`.

ii.
```python
    rewarded = np.zeros(ntrials, dtype=np.int64)
    for t, (s, e) in enumerate(zip(starts, stops)):
        in_zone = np.any(rzone[s:e] > 0)
        got = np.any((reward_times >= stamps[s]) & (reward_times < stamps[e]))
        rewarded[t] = int(in_zone and got)
```
```python
        prev = 1 if t == 0 else int(rewarded[t - 1])
        ...
        inp[3] = prev
```

iii. The AI copied the rule from the paper's `behavior.get_trial_types`, which computes `isreward = (np.any(tmp_reward > 0) and np.any(tmp_rzone > 0)) * 1` over `[trial_start, teleport)`. Because the `Reward` series has its own sparse timestamp vector rather than a per-frame value, the AI compares reward times against the trial's first and last behaviour timestamps instead of indexing by frame.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The previous lap's outcome is looked up and broadcast across the current trial's timepoints. For the first trial of a session, where no predecessor exists in the file, the value is set to **1 (rewarded)** rather than 0.

ii.
```python
        # previous trial outcome; on the first imaged trial of a session the
        # preceding (warm-up) trial is not in the file, so it is set to rewarded,
        # the outcome of ~85% of trials
        prev = 1 if t == 0 else int(rewarded[t - 1])
```

iii. From the final summary: "One judgment call: the first trial of each session has no recorded predecessor, so 'previous trial rewarded' is set to 1 — the imaging session is preceded by ~30 warm-up trials in the same condition, and ~85% of trials are rewarded. It affects one trial per session." The lookup uses the raw lap index `t-1`, so a lap dropped for a lick-sensor error is still used as the predecessor of the following lap, which is the behaviourally correct thing to do.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behaviour series and the trial's reward-zone identity. The zone identity comes from the scene name (see 4-a / 10-a) and its coordinates from the paper's `reward_zone_dict`: A = 80–130 cm, B = 200–250 cm, C = 320–370 cm. The `reward_zone` occupancy series is used only for validation and for the reward-outcome rule, not to define the zone boundaries.

ii.
```python
REWARD_ZONES = {'A': (80., 130.), 'B': (200., 250.), 'C': (320., 370.)}
ZONE_ORDER = ['A', 'B', 'C']
```
```python
        pos = beh['position/data'][:]
    ...
        p = pos[s:e]
        out[0] = bin_reward_distance(p, zones[t])
```

iii. The AI read `reward_relative/behavior.py` (step 25) and found `reward_zone_dict` with entries `'X': [80, 130], 'Y': [200, 250], 'Z': [320, 370]`, and `get_reward_zones` mapping scene `LocationA/B/C` onto `X/Y/Z`; its constant table records this ("'X', 'Y', 'Z' there are the A, B, C zones of the switch task"). It then validated the scene-derived zone against the recorded `reward_zone` occupancy on every trial that has one: "**0 mismatches** in all 10,394 trials that have one" (step 52: `total mismatch 0 noz 1822`).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest edge of the zone: negative before the zone (`position − zone_start`), zero anywhere inside the zone, positive after it (`position − zone_stop`). Computed per timepoint from the trial's raw position trace, then discretised (7-c).

ii.
```python
def bin_reward_distance(pos, zone):
    """Signed distance to the nearest point of the reward zone, discretized."""
    start, stop = REWARD_ZONES[zone]
    d = np.zeros_like(pos)
    d[pos < start] = pos[pos < start] - start
    d[pos > stop] = pos[pos > stop] - stop
```

iii. This is the "distance to any location in the reward zone" the instructions ask for, and matches the paper's reward-relative distance convention (0 throughout the 50 cm zone). No smoothing or spatial binning of position is applied first — the paper's 10 cm spatial bins are for its own place-field analyses, not for a frame-resolved decoder target.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Into the 7 categories given in the instructions, using explicit boolean masks with the "inside the zone" class as the default: `< -50 → 0`, `[-50, -10) → 1`, `[-10, 0) → 2`, `d == 0 → 3`, `(0, 10] → 4`, `(10, 50] → 5`, `> 50 → 6`.

ii.
```python
    out = np.full(pos.shape, 3, dtype=np.int64)     # 3: inside the zone (d == 0)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```
```python
            ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in reward zone (0 cm)',
             '0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
```

iii. The bin edges are read straight off the Decoder Task specification. Setting class 3 as the array default is the clean way to express "exactly 0 cm = inside the zone", since the distance is exactly zero for every sample within the 50 cm zone. The resulting class fractions (0.251 / 0.102 / 0.073 / 0.238 / 0.021 / 0.072 / 0.243) were checked in the verification run.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the same `[s:e]` frame indices as the neural matrix, so the output row has the same `T` and the same per-frame correspondence. No resampling or lag is applied.

ii.
```python
    for t, (s, e) in enumerate(zip(starts, stops)):
        ...
        T = e - s
        p = pos[s:e]
        ...
        out = np.empty((6, T), dtype=np.int64)
        out[0] = bin_reward_distance(p, zones[t])
        ...
        neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. The VR streams in these NWB files are already interpolated onto the imaging frame clock (module docstring), verified by the uniform 0.06448363 s timestamps and the matching frame counts; the 10 one-frame mismatches are handled by the `nframes` crop. The AI additionally eyeballed the alignment in `sample_trials.png` (step 89): "position and distance-to-zone rise monotonically within a trial, and licking coincides with the speed dip at the reward zone."

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behaviour time series (cm along the VR corridor), used raw.

ii.
```python
        pos = beh['position/data'][:]
    ...
        p = pos[s:e]
        out[1] = bin_position(p)
```

iii. The survey (step 44) recorded `posmax` per session, ~450–452 cm, consistent with the Methods' 450 cm track, so the series is already in centimetres and needs no rescaling.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond the per-trial slice and discretisation: integer division of position by 90 cm (= 450/5), clipped into [0, 4].

ii.
```python
TRACK_LENGTH = 450.       # cm

def bin_position(pos):
    """Track position into 5 equal bins of 90 cm spanning the 450 cm track."""
    return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```

iii. The instructions ask for "5 equal-sized bins spanning the 450 cm track"; 450/5 = 90 cm. The clip absorbs the handful of samples that fall marginally outside [0, 450] (the survey found positions up to ~451.9 cm) rather than creating spurious extra classes.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five classes with edges at 90, 180, 270 and 360 cm, implemented as the clipped integer division above; the first and last classes are open-ended (`< 90` and `> 360`) so off-track samples land in the end bins.

ii.
```python
    return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```
```python
            ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
```

iii. Directly from the Decoder Task specification. The verification run gives class fractions 0.212 / 0.177 / 0.231 / 0.226 / 0.154, i.e. roughly uniform with the expected under-representation of the last bin (animals run fastest at the end of the lap).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[s:e]` slice as the neural data; no further alignment.

ii.
```python
        p = pos[s:e]
        out[1] = bin_position(p)
```

iii. Same justification as 7-d — behaviour and imaging share the frame clock, and the one-frame-short sessions are cropped to the overlap before trials are cut.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behaviour time series (per-frame cumulative lick counts).

ii.
```python
        lick = beh['lick/data'][:]
    ...
        licks = lick[s:e]
        ...
        out[3] = (licks > 0).astype(np.int64)
```

iii. The exploratory dump at step 48 showed `lick` to be a non-negative per-frame count (tens of counts per trial), which the instructions require as a binary no/yes output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation: any frame with a count > 0 becomes 1. In addition, whole trials whose lick trace shows the paper's stuck-sensor signature (>30% of frames with count > 2) are discarded before this point, so no corrupted lick trace reaches the output (81 trials, see 1-e).

ii.
```python
        if np.mean(licks > LICK_ERROR_COUNT) > LICK_ERROR_FRAC:
            n_lick_error += 1
            continue
        ...
        out[3] = (licks > 0).astype(np.int64)
```
```python
            ['no lick', 'lick'],
```

iii. The Methods describe exactly this two-step treatment: erroneous trials are removed, and "Remaining lick counts were converted to a binary vector". The AI deviates only in that the paper NaNs the trial's licks whereas the AI drops the trial entirely, because the target format has no missing-value representation for a categorical decoder output. Resulting fractions: 0.777 no-lick / 0.223 lick.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[s:e]` frame slice as the neural matrix; no shift or resampling.

ii.
```python
        licks = lick[s:e]
        ...
        out[3] = (licks > 0).astype(np.int64)
```

iii. Same justification as 7-d/8-d. The sample-trial plot check confirmed licking coincides with the reward-zone slow-down.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the session's scene name in `identifier`, parsed by `parse_scene` (the same call that yields the environment), with the switch after trial 30 on switch sessions. The zone letter is mapped to the class index via `ZONE_ORDER = ['A', 'B', 'C']`. The `reward_zone` occupancy series was used to validate, not to derive.

ii.
```python
    zones, envs = parse_scene(scene, ntrials)
    ...
        out[4] = ZONE_ORDER.index(zones[t])
```
```python
            ['A (80-130 cm)', 'B (200-250 cm)', 'C (320-370 cm)'],
```

iii. This is `reward_relative.behavior.get_reward_zones` reproduced: the scene id fully determines the zone for constant sessions and gives the pre/post-switch pair for `X_to_Y` sessions, with `change_trial = 30`. The AI validated the labels against the recorded reward-zone occupancy for every trial in which the animal actually entered a zone: 0 mismatches over 10,394 trials, the remaining 1,822 trials having no occupancy sample to check against. The resulting class balance is 0.332 / 0.336 / 0.333.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regex parse of the scene name → per-trial letter list (first 30 trials the old zone, the rest the new zone on switch sessions) → index into `['A','B','C']` → broadcast across all timepoints of the trial.

ii.
```python
    m = re.fullmatch(r'Env(\d)_Location([ABC])_to_([ABC])', scene)
    if m:
        env = int(m.group(1)) - 1
        zones = [m.group(2)] * change_trial + [m.group(3)] * (ntrials - change_trial)
        return zones[:ntrials], [env] * ntrials
```
```python
        out[4] = ZONE_ORDER.index(zones[t])
```

iii. As above: this is the paper's own labelling function, made deterministic and validated against the data. Emitting it as a constant time series rather than a scalar keeps the output block a single `(6, T)` array, which the instructions encourage ("If at all possible, make it time-varying").

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` time series' timestamps together with the `reward_zone` occupancy series and the behaviour `timestamps` that define the trial window.

ii.
```python
        stamps = beh['position/timestamps'][:]
        reward_times = beh['Reward/timestamps'][:]
    ...
    rewarded = np.zeros(ntrials, dtype=np.int64)
    for t, (s, e) in enumerate(zip(starts, stops)):
        in_zone = np.any(rzone[s:e] > 0)
        got = np.any((reward_times >= stamps[s]) & (reward_times < stamps[e]))
        rewarded[t] = int(in_zone and got)
```

iii. The `Reward` series is event-based (74 events for 80 trials in the file the AI inspected at step 48) with its own timestamps, so it must be matched to the trial window by time rather than by frame index. The conjunction with `reward_zone` is the paper's own definition in `behavior.get_trial_types`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per trial: 1 if a reward event fell inside `[stamps[trial_start], stamps[teleport])` **and** the reward zone was active somewhere in the trial, else 0. The scalar is broadcast across all timepoints. Resulting fractions: 0.842 rewarded / 0.158 omitted.

ii.
```python
    # per-trial reward: a reward was delivered inside the (active) reward zone,
    # following reward_relative.behavior.get_trial_types
        in_zone = np.any(rzone[s:e] > 0)
        got = np.any((reward_times >= stamps[s]) & (reward_times < stamps[e]))
        rewarded[t] = int(in_zone and got)
    ...
        out[5] = rewarded[t]
```

iii. The AI states the check it used: "Reward outcome uses the paper's rule (reward delivered *and* zone active) → 84.2% rewarded, matching the ~15% omission rate" described in the Methods for this operant task.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases:
- **Behaviour/imaging length mismatch** (10 of 152 sessions, always one frame): everything is cropped to `nframes = min(n_imaging, n_behaviour)` and teleport indices are clipped to it.
- **Corrupted lick trials** (81): dropped outright (see 1-e).
- **No previous trial for the first lap of a session**: set to 1 (see 6-b).
- **Trials in which the animal never triggered the reward-zone signal** (1,822): no problem, because the zone identity is taken from the scene name rather than from the occupancy signal; only the reward-outcome flag depends on `rzone`, and there `in_zone == False` correctly means "no reward collected".
Structural sanity is enforced with an assertion on the trial boundaries. There is no minimum-trial-length filter, and no assertion that the reward timestamps land within half a bin of a behaviour timestamp.

ii.
```python
    # behaviour and imaging occasionally differ by one frame; use the overlap
    nframes = min(F.shape[1], len(pos))
    F, Fneu = F[:, :nframes], Fneu[:, :nframes]
    ...
    assert len(starts) == len(stops) and np.all(stops > starts)
    stops = np.minimum(stops, nframes)
```
```python
        if np.mean(licks > LICK_ERROR_COUNT) > LICK_ERROR_FRAC:
            n_lick_error += 1
            continue
```
```python
        prev = 1 if t == 0 else int(rewarded[t - 1])
```

iii. Each case was found empirically first: step 52 enumerated the ten one-frame-mismatch sessions and counted the 81 lick-error trials and the 1,822 zone-less trials; step 91 confirmed the ROI id/`rois` indexing assumption across all files; step 58 confirmed `scanning != 1` never occurs inside a trial, so no frames need to be dropped for the scanner being off. The AI also kept a per-session `info` record (`n_trials_total`, `n_trials_lick_error`, `n_interneurons_excluded`, `trial_indices`, …) in `metadata['session_info']` so that any dropped trial can be traced afterwards.

## 13-a. What are the most time-consuming steps of the code?

i. In order: (1) reading the F, Fneu (and per-plane ROI) arrays out of each NWB file — every ROI is read, including the ~50% that fail `iscell` in the two-plane sessions; (2) the OASIS deconvolution, called once per trial per session (≈ 80 calls × 152 sessions) inside `compute_dff_events`, plus the maximin baseline filters; (3) pickling the 9.6 GB result in one `pickle.dump`. The whole conversion took 2 min 26 s wall clock with 12 worker processes (248 min of CPU), so the per-session work is heavily parallelised and the serial tail is the accumulation of results in the parent and the single write.

ii.
```python
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as ex:
        for i, res in enumerate(ex.map(convert_session, files)):
```
```python
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(d, dtype=np.float32),
                                    2000, TAU, fs)
```
```python
    with open(args.out, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
```

iii. The AI measured this: step 58 timed the dF/F step at 0.23 s for a 155-cell session, step 68 timed four sessions at 9.5 s on four workers, and step 54 computed the output size in advance (`total elements 2418071718, GB float32 9.67`) before committing to the format. It chose process-level parallelism over vectorisation, and switched from `fork` to `spawn` after finding that suite2p's numba-threaded OASIS is not fork-safe (steps 62–74).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several, all over trials:
- `compute_dff_events` iterates over trials three times (copy into the NaN-masked array, then the baseline/dF/F/deconvolution pass). The copy loop and the `inside` mask loop could both be replaced by a single boolean mask built with `np.add.reduceat`/index arithmetic.
- The `inside` mask construction in `convert_session` is a third pass over the same `(starts, stops)` pairs.
- The `rewarded` loop is a fourth; `in_zone` could be computed for all trials at once with `np.maximum.reduceat(rzone > 0, starts)` and `got` with a single `np.searchsorted(stamps, reward_times)`.
- The main emission loop is a fifth; `bin_reward_distance`, `bin_position` and `bin_speed` could be applied once to the whole session array and then sliced, since only the zone identity varies per trial.
The maximin baseline could also be run on the whole session at once with a trial-id-aware mask instead of per-trial `gaussian_filter1d`/`minimum_filter1d`/`maximum_filter1d` calls. The OASIS call genuinely has to stay per trial, since the baseline and kernel state are per trial by design.

ii.
```python
    for s, e in zip(starts, stops):
        f[:, s:e] = F[:, s:e]
        fneu[:, s:e] = Fneu[:, s:e]
```
```python
    inside = np.zeros(nframes, dtype=bool)
    for s, e in zip(starts, stops):
        inside[s:e] = True
```
```python
    for t, (s, e) in enumerate(zip(starts, stops)):
        in_zone = np.any(rzone[s:e] > 0)
        got = np.any((reward_times >= stamps[s]) & (reward_times < stamps[e]))
```

iii. Not discussed in the trajectory. The AI's efficiency effort went into multiprocessing instead, which already reduced the total runtime to under three minutes, so the per-trial Python loops (~80 iterations per session) were never a bottleneck worth removing. The interneuron correlation, by contrast, *was* deliberately vectorised into a single matrix–vector product rather than the paper's per-cell `np.corrcoef` loop.

## 13-c. What processing does the code repeat multiple times?

i. Within a session: the trial-boundary iteration is repeated five times as listed in 13-b, and `np.nanmean(fneu[:, s:e])` is recomputed even though `fneu` was just written from `Fneu`. Inside `bin_reward_distance` the boolean masks `pos < start` and `pos > stop` are each evaluated twice (once to index the left-hand side, once the right). `parse_scene` is called once per session, which is fine. Across sessions there is *no* repetition: each NWB file is opened exactly once and all derived quantities come from that single read — there is no separate survey pass. The one genuinely duplicated effort is in the trajectory rather than in the script: the AI ran standalone survey scripts (`/tmp/survey.py`, `/tmp/beh.py`) over all 152 files before writing `convert_data.py`, but those results are not needed at conversion time.

ii.
```python
        trial = f[:, s:e] + NEU_COEF * np.nanmean(fneu[:, s:e], axis=1, keepdims=True)
```
```python
    d[pos < start] = pos[pos < start] - start
    d[pos > stop] = pos[pos > stop] - stop
```

iii. Not discussed. The single-pass design is a deliberate consequence of deriving the per-trial condition from the scene name rather than from a dataset-wide inference over the reward-zone occupancy: nothing in the conversion depends on statistics pooled across sessions, so no pre-pass is required.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four items:
- **F and Fneu are read for every ROI, then subset to `iscell`.** In the two-plane sessions this reads and transposes roughly twice the data that is kept (e.g. 3,007 ROIs read, 1,382 kept). Subsetting at read time would halve the I/O and peak memory.
- **dF/F and OASIS events are computed for putative interneurons that are then dropped.** `events = events[good]` discards 402 cells' worth of deconvolution. (The dF/F itself is needed for the speed correlation, but the deconvolution of those cells is pure waste.)
- **The full-session `dff` array is kept alive** after the interneuron correlation, even though only `events` is emitted; it is the largest temporary in the worker.
- **Neural events, `rewarded` and the dF/F are computed for the 81 lick-error trials** that are then skipped in the emission loop.
Additionally, `metadata['session_info']` stores a full per-session record including `trial_indices` (a list of every kept lap index) which the decoder never reads — harmless, but it is carried in the 9.6 GB pickle.

ii.
```python
    # curated cells only (suite2p manual curation by the authors)
    keep = iscell[roi_ids]
    F, Fneu = F[keep], Fneu[keep]
```
```python
    dff, events = compute_dff_events(F, Fneu, starts, stops, fs)
    ...
    good = ~(r > INTERNEURON_R)
    events = events[good]
    n_interneurons = int((~good).sum())
```
```python
            'session_info': [r['info'] for r in results],
```

iii. Not discussed in the trajectory. The ordering is forced in part by the science — the interneuron filter is defined on dF/F, so dF/F must exist for cells that are later dropped — but the deconvolution of those cells and the reading of non-curated ROIs are avoidable. Because the whole conversion runs in under three minutes, none of this was worth optimising in practice.
