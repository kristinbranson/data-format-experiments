# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every `.nwb` file under `/app/data/sub-*/` is globbed (152 files, 11 mice) and each file is converted independently by `load_session()`. Files are sorted by numeric subject id then filename, so sessions appear in subject/day order. The NWB files are opened **directly with `h5py`** rather than `pynwb`; the agent reads the HDF5 paths it needs (`processing/behavior/BehavioralTimeSeries/*`, `processing/ophys/Fluorescence|Neuropil|ImageSegmentation`, `general/subject/subject_id`, `general/session_id`, `general/optophysiology/ImagingPlane/imaging_rate`, `identifier`). Sessions are converted in parallel with a 12-worker `ProcessPoolExecutor` using the `spawn` start method. Nothing is subsampled: all 152 sessions, all curated ROIs and all laps enter the pipeline (`--nsessions` exists only for testing).

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', 'sub-*.nwb')),
               key=lambda p: (int(os.path.basename(p).split('_')[0][5:]),
                              os.path.basename(p)))
...
ctx = multiprocessing.get_context('spawn')
with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
    for i, res in enumerate(pool.map(load_session, files)):
        results.append(res)
```
```python
def load_session(path):
    """Read one NWB session and return the per-trial neural/input/output arrays."""
    with h5py.File(path, 'r') as f:
        ident = f['identifier'][()].decode()
        subject = f['general/subject/subject_id'][()].decode()
        day = int(f['general/session_id'][()].decode())
        beh = f['processing/behavior/BehavioralTimeSeries']
        ...
        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
```

iii. From the trajectory: the agent first dumped the whole HDF5 tree (`f.visititems`) of one file and enumerated every behavioural time series, ophys group and attribute, then ran a survey over all 152 files (steps 44–66) confirming 152 sessions / 12,216 trials / 138,678 curated ROIs / one common `dt` of 0.064484 s / 1–2 imaging planes. It used `h5py` rather than `pynwb` because an NWB file is an HDF5 file and only a handful of datasets are needed — this lets it slice `Fluorescence`/`Neuropil` lazily and keeps memory low enough to run 12 sessions concurrently. The agent checked resources (`free -g`, `nproc`) before choosing the parallel design.

## 1-b. How are the data split into subjects?

i. The subject of a session is read from inside the file (`general/subject/subject_id`), not from the directory name. The unique subject ids are collected after conversion and sorted numerically; `subject_idx` indexes that list for each session.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
data['subject_idx'] = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. The agent verified in its survey that the 11 `sub-*` directories correspond to the 11 mice of the paper, and that the internal `subject_id`, the filename and the `identifier` string (`/data/InVivoDA/GCAMP3/...`) all agree. Reading the id from the file rather than the path is the more authoritative source. The resulting list is `['m3','m4','m7','m11','m12','m13','m14','m15','m17','m18','m19']` with 14 sessions each except m11 (12).

## 1-c. How are the data split into sessions?

i. One `.nwb` file = one session. The experiment day is read from `general/session_id` and stored in `metadata['session_info']` together with the scene string and date; no cross-session cell registration is attempted, so each session's neuron set is independent.

ii.
```python
day = int(f['general/session_id'][()].decode())
ident = f['identifier'][()].decode()
scene = ident.split('/')[-1]      # e.g. 'Env1_LocationC_to_A'
date  = ident.split('/')[-2]
...
'session_info': [r['info'] for r in results],
```

iii. The agent confirmed each file's `session_id` matches the `ses-NN` in the filename and that the `identifier` encodes animal/date/scene, so a file is unambiguously one recording day. It deliberately kept sessions separate (the decoder is fit per session), which also sidesteps the paper's ROI-matching-across-days machinery.

## 1-d. How are the data split into trials?

i. A trial is one lap of the 450 cm virtual track. Trial boundaries come from the `trial_start` and `teleport` behaviour channels, and the extracted window is **`[trial_start - 1, teleport - 1)`** — i.e. exactly the window `reward_relative.preprocessing.dff` slices when it is called with the session's `trial_start_inds`/`teleport_inds`. The same index window is used for the neural data and for every behavioural stream, so all streams stay mutually aligned. The inter-trial teleport period is excluded entirely.

ii.
```python
trial_starts = np.nonzero(beh['trial_start/data'][:])[0]
teleports    = np.nonzero(beh['teleport/data'][:])[0]
# trial windows, matching preprocessing.dff: [start-1, stop-1)
starts = trial_starts - 1
stops  = teleports - 1
assert np.all(starts[1:] > stops[:-1]) and starts[0] >= 0
...
s, t = starts[i], stops[i]
T = t - s
p  = pos[s:t]
v  = speed[s:t]
neural.append(np.ascontiguousarray(events[:, s:t], dtype=np.float32))
```

iii. From the docstring/trajectory: "samples `[trial_start - 1, teleport - 1)`, exactly the window used by `preprocessing.dff` (it drops the teleport sample, whose position is interpolated across the jump from the end of the track to the teleport zone)". The agent printed positions around the boundaries (step 39) and saw that the teleport sample carries an interpolated position (e.g. 84.2 cm on a lap that ended at 449.6 cm), so it must be dropped. The ITI is excluded because "the decoded variables (track position, distance to the reward zone) are undefined there and because the laser was blanked during the ITI on most sessions". It also checked `len(trial_start) == len(teleport)` implicitly through the interleaving assertion, which held for all 152 sessions (12,216 laps total).

## 1-e. How are trials filtered based on quality controls?

i. Two filters, no others:
- **Stuck lick sensor**: a trial is dropped if >30% of its imaging samples have a cumulative lick count > 2 — the paper's `behavior.correct_lick_sensor_error` criterion. This flags exactly 81 trials, the number the Methods report. The paper NaNs the licks of these trials; because lick is a decoder *output* here and NaN is not a valid category, the whole trial is dropped.
- **First trial of each session** is dropped because "previous trial outcome", a required decoder input, is undefined for it (152 trials).

No speed threshold, no minimum trial length, no session or mouse exclusion: 12,216 − 81 − 152 = **11,983 trials** over all 152 sessions and 11 mice.

ii.
```python
LICK_ERR_FRAC = 0.3     # >30% of samples with cumulative lick count > 2 => sensor error
...
lick_error[i] = np.mean(lick[s:t] > 2) > LICK_ERR_FRAC
...
for i in range(ntrials):
    if i == 0:                      # previous trial outcome undefined
        continue
    if lick_error[i]:               # stuck lick sensor
        continue
```

iii. The agent read `behavior.correct_lick_sensor_error` (`correction_thr=0.5` default, but the Methods state 30%) and the Methods sentence "n = 81 out of 12,376 trials removed ... detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2". Its whole-dataset scan (step 63) printed `lick err totals [81 69 43]` for thresholds 0.3/0.4/0.5, so it adopted 0.3 and reported that this "flags 81 trials, matching the 81 trials reported in the paper". It explicitly declined the paper's ≥2 cm/s running filter "because speed is itself a decoded output".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces `processing/ophys/Fluorescence/plane*/data` (F) and `processing/ophys/Neuropil/plane*/data` (Fneu), restricted to `iscell` ROIs. The NWB `Deconvolved` series is deliberately **not** used.

ii.
```python
planes = sorted(f['processing/ophys/Fluorescence'].keys())
for pi, plane in enumerate(planes):
    rois = f['processing/ophys/Fluorescence'][plane]['rois'][:]
    keep = iscell[rois]
    assert np.all(plane_idx[rois] == pi)
    F_list.append(f['processing/ophys/Fluorescence'][plane]['data'][:, :].T[keep])
    Fneu_list.append(f['processing/ophys/Neuropil'][plane]['data'][:, :].T[keep])
```

iii. Docstring: "The NWB file's 'Deconvolved' series is raw suite2p `spks` (non-zero during the inter-trial interval, i.e. not from the paper's within-trial dF/F), so it is NOT used; F and Fneu are re-processed instead." The agent verified this empirically (step 28): the `Deconvolved` array has mean 132 during the ITI with only 68% zeros, so it cannot be the paper's trial-restricted signal. The paper instead computes its own signal with `preprocessing.dff(..., deconvolve=True)`.

## 2-b. How is the `neural` data processed?

i. A re-implementation of the paper's `reward_relative.preprocessing.dff` followed by OASIS deconvolution, applied per trial window:
1. `F − 0.7·Fneu` (neuropil subtraction, `neu_coef = 0.7`),
2. add the trial's mean neuropil back (`+ 0.7·mean(Fneu)`) so the ratio is a true dF/F,
3. maximin baseline: Gaussian smoothing with σ = 15 samples, then a 300-sample (≈20 s) running minimum followed by a 300-sample running maximum,
4. `dF/F = (F − baseline)/|baseline|`,
5. Gaussian smoothing with σ = 2 samples (~0.129 s),
6. `suite2p.extraction.dcnv.oasis(dff, 2000, tau=0.7, frame_rate/n_planes)`.

Planes are processed in one pooled array and the per-plane frame rate (`imaging_rate/len(planes)`, 15.5078 Hz) is used for the deconvolution kernel. The deconvolved "events" are what is written to `neural` (float32). One deviation from the paper's code: the baseline window is **always** restricted to the lap; the `keep_teleports` per-mouse/per-day table in `teleport_metadata.py` (days on which the laser was not blanked, where the paper lets the baseline span the teleport) is not applied.

ii.
```python
NEU_COEF = 0.7; TAU = 0.7; BASELINE_WIN = 300; BASELINE_SIG = 15; DFF_SIG = 2

def compute_dff_and_events(F, Fneu, starts, stops, frame_rate):
    dff = np.full(F.shape, np.nan, dtype=np.float32)
    events = np.full(F.shape, np.nan, dtype=np.float32)
    for start, stop in zip(starts, stops):
        f = F[:, start:stop] - NEU_COEF * Fneu[:, start:stop]
        # add the trial's mean neuropil back so dF/F is not divided by a tiny baseline
        f = f + NEU_COEF * np.nanmean(Fneu[:, start:stop], axis=1, keepdims=True)
        flow = gaussian_filter1d(f, BASELINE_SIG, axis=-1)
        flow = minimum_filter1d(flow, BASELINE_WIN, axis=-1)
        flow = maximum_filter1d(flow, BASELINE_WIN, axis=-1)
        d = (f - flow) / np.abs(flow)
        d = gaussian_filter1d(d, DFF_SIG, axis=-1)
        dff[:, start:stop] = d
        events[:, start:stop] = dcnv.oasis(d, 2000, TAU, frame_rate)
    return dff, events
```
```python
imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
frame_rate = float(imaging_rate) / len(planes)
```

iii. The agent read `preprocessing.py` in full, plus `utilities.multi_anim_sess` for the call-site defaults (`neuropil_method='subtract'`, `baseline_method='maximin'`, `neu_coef=0.7`) and the suite2p ops notebook for `tau=0.7`. It matched the Methods text ("maximin procedure with a 20 s sliding window ... divided by the absolute value of the baseline, then smoothed with a two-sample (~0.129 s) s.d. Gaussian kernel ... deconvolving dF/F ... using the OASIS algorithm"). It checked the two-plane files (step 67) and found the stored `imaging_rate` is the scanner rate (31.0156 Hz for m17/m18), so it divides by the number of planes. It also ran a pilot comparison of dF/F vs events as decoder input on 4 sessions (steps 83–86): dF/F scored ~0.05–0.08 higher, but it chose events because "events are the paper-consistent choice". The `keep_teleports` table was read (step 34) but not used; the docstring nevertheless claims the dF/F is recomputed "exactly as in `reward_relative.preprocessing.dff`".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) Only `iscell == 1` ROIs from the NWB `PlaneSegmentation` (suite2p manual curation); planes are pooled for the two multi-plane mice. (2) Putative interneurons are removed: cells whose dF/F has Pearson r > 0.5 with running speed over all within-trial samples. 409 of 138,678 curated ROIs (0.29%) were dropped, leaving **138,269 neurons**.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)
...
keep = iscell[rois]
```
```python
INT_R_THRESH = 0.5
valid = ~np.isnan(dff[0])
dff_v = dff[:, valid]; spd_v = speed[valid]
dff_c = dff_v - dff_v.mean(axis=1, keepdims=True)
spd_c = spd_v - spd_v.mean()
denom = np.sqrt((dff_c ** 2).sum(axis=1) * (spd_c ** 2).sum())
speed_corr = (dff_c @ spd_c) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
keep_cells = ~is_int
events = events[keep_cells]
```

iii. Methods: "Manual curation eliminated ROIs containing multiple somata or dendrites..." and "Additional putative interneurons were detected for exclusion ... by a Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed, excluding 0.42 ± 0.85% of cells". The agent located `spatial.is_putative_interneuron` and confirmed `dayData` passes `int_thresh = 0.5` (the function's own default is 0.3). It spot-checked three sessions (step 87) getting 0–3.6% flagged per session, and reported the dataset-wide 0.29% as "in line with the paper's 0.42 ± 0.85%". The correlation is computed only over samples inside laps (the non-NaN dF/F mask), and vectorised as a matrix product rather than a per-cell loop.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, which requires no extra work beyond the trial split: sample 0 of every trial is the first sample of the lap window, and `metadata['off_start'] = 0.0`, `off_end = None` (trials have variable length). The neural slice uses exactly the same index window as all behavioural streams.

ii.
```python
s, t = starts[i], stops[i]
neural.append(np.ascontiguousarray(events[:, s:t], dtype=np.float32))
inp[0] = np.arange(T, dtype=np.float32) * dt
...
'temporal_alignment_event': ('start of the trial (lap): the imaging frame at which the animal '
                             'enters the virtual track at position 0 cm'),
'off_start': 0.0,
'off_end': None,
```

iii. The instructions say "Temporally align based on start of the trial". Because the NWB behaviour series are already sampled on the imaging frame grid (the agent verified a single `dt = 0.064484 s` shared by all 152 sessions and by the imaging rate), slicing neural and behaviour with the same indices is sufficient; no resampling or shifting is needed. `off_end` is `None` because laps have different durations.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning. Data are kept at the native imaging-frame resolution: **64.484 ms** (15.5078 Hz per plane). `dt` is taken as the median of the behaviour timestamp differences per session, and the code asserts all sessions share the same value before writing `metadata['time_bin_size']` in ms.

ii.
```python
dt = float(np.median(np.diff(tstamps)))
...
dts = {round(r['dt'], 6) for r in results}
assert len(dts) == 1, f'inconsistent time bins: {dts}'
...
'time_bin_size': float(round(list(dts)[0] * 1000, 4)),
```

iii. The agent's survey found a single `dt` across all sessions (`dt uniq {0.06448}`) equal to `n_planes / imaging_rate`, so no resampling is needed to make bins uniform across trials and sessions. Keeping the native rate preserves all temporal information for the decoder; the two-plane sessions are already sampled at the same per-plane rate as the one-plane sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From the behaviour timestamps: `dt` is the median of `diff(position/timestamps)` for the session, and the per-trial time axis is the sample index times `dt`.

ii.
```python
tstamps = beh['position/timestamps'][:]
...
dt = float(np.median(np.diff(tstamps)))
...
inp[0] = np.arange(T, dtype=np.float32) * dt
```

iii. The agent checked that all behaviour series carry identical timestamps and that they are on a perfectly regular grid (`timestamps diff 0.06448362720402656`, step 28). Using `arange(T)*dt` is therefore numerically identical to `timestamps[idx] - timestamps[idx][0]` (I verified the maximum discrepancy over whole sessions is 3e-12 s) and guarantees the axis starts at exactly 0.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. None beyond multiplying the within-trial sample index by the bin width, so the value is 0 at the first sample of the lap and increases by 64.484 ms per bin. Stored as float32, range over the dataset 0 – 216.5 s.

ii.
```python
inp = np.empty((4, T), dtype=np.float32)
inp[0] = np.arange(T, dtype=np.float32) * dt
```

iii. The decoder input is specified as "Time from start of trial in seconds", and alignment is to trial start, so the first sample must be 0. No other processing is warranted.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: it is built from the length `T` of the same neural slice, so input, output and neural arrays always have identical `n_timepoints`. The imaging and VR streams are already on a common clock in the NWB file; the only mismatch the agent found was one *extra* imaging frame in 10 multi-plane sessions, which is truncated before anything is sliced.

ii.
```python
nframes = len(pos)
assert 0 <= F.shape[1] - nframes <= 1, (path, F.shape, nframes)
assert stops[-1] <= nframes
F = F[:, :nframes]
Fneu = Fneu[:, :nframes]
...
T = t - s
inp = np.empty((4, T), dtype=np.float32)
out = np.empty((6, T), dtype=np.int64)
```

iii. Step 91 enumerated every session and found the ophys arrays are the same length as the behaviour arrays except for 10 m17/m18 sessions with exactly one extra frame after the last teleport; the docstring attributes this to "the 'one frame correction' in TwoPUtils' `vr_align_to_2P`" and notes "that trailing frame has no behaviour and falls after the last teleport", so dropping it is lossless.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behaviour time series (the VR "morph" value), read per trial.

ii.
```python
env = beh['environment/data'][:]
...
env_vals = np.unique(env[s:t])
assert len(env_vals) == 1, f'{path}: trial {i} spans environments {env_vals}'
envs[i] = int(env_vals[0])
```

iii. The survey found `environment` takes values {−1, 0, 1}, with −1 only outside laps; within a lap it is constant. 0/1 map onto the paper's ENV1/ENV2 visually distinct environments. The agent also confirmed (step 45) that the 11 `Env1_*_to_Env2_*` sessions are exactly the ones where the value changes across trials within a session.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. It is reduced to one value per trial (asserting the trial does not span two environments) and then broadcast across all timepoints of the trial as a 0/1 input. `metadata['session_info']` additionally records `ENV1`, `ENV2` or `ENV1_to_ENV2` per session.

ii.
```python
inp[1] = envs[i]
...
'environment': ['ENV1', 'ENV2'][int(envs[0])] if len(np.unique(envs)) == 1
               else 'ENV%d_to_ENV%d' % (envs[0] + 1, envs[-1] + 1),
```

iii. The specification calls environment "binary, ENV1 vs ENV2, per trial", so a per-trial scalar broadcast over time is the right representation. The assertion is the agent's check that the raw values really are constant within a lap; it passed for all 12,216 laps.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. From the position of the lap in the session — the index `i` of the loop over the trial boundaries derived from `trial_start`/`teleport`. The NWB `trial number` time series is not used.

ii.
```python
ntrials = len(starts)
for i in range(ntrials):
    ...
    inp[2] = i
```

iii. The agent inspected the stored `trial number` channel (step 39) and found it disagrees with `trial_start`/`teleport` (`trialnum at starts [0,1,2,3,4]` but `at tel [1,1,2,4,4]`), i.e. it is unreliable at boundaries. Counting laps from the boundary channels is consistent with how trials are defined everywhere else in the script.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None: the raw within-session lap index is broadcast over all timepoints of the trial, stored as float32. Because the first lap of each session and the stuck-lick laps are dropped, the numbering keeps its original values and simply has gaps (range over the dataset 1 – 99).

ii.
```python
inp[2] = i
```

iii. Trial number is specified as "continuous, per trial". Keeping the original index (rather than renumbering the retained trials) preserves the true ordinal position of each lap within the session, which is what the variable is meant to convey (e.g. the reward-zone switch happens at lap 30).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the same per-trial `rewarded` vector used for the reward-outcome output: reward-event timestamps (`Reward/timestamps`) intersected with the previous lap's time window, conjoined with the requirement that the `reward_zone` flag was on at some point in that lap.

ii.
```python
reward_t = beh['Reward/timestamps'][:]
...
for i, (s, t) in enumerate(zip(starts, stops)):
    in_zone = rzone[s:t] > 0
    got_reward = np.any((reward_t >= tstamps[s]) & (reward_t < tstamps[t]))
    rewarded[i] = int(in_zone.any() and got_reward)
...
inp[3] = rewarded[i - 1]
```

iii. This is the paper's own trial-type definition (`behavior.get_trial_types`: `np.any(tmp_reward > 0) and np.any(tmp_rzone > 0)`), which the agent read at step 36. Comparing reward timestamps directly to the first/last behaviour timestamp of the lap avoids having to snap reward events onto the sample grid.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The previous lap's binary outcome is broadcast over all timepoints of the current trial. The first lap of a session is not given a fabricated value — it is dropped from the dataset entirely, so every emitted trial has a genuinely observed predecessor. Note the predecessor is the previous *lap*, including laps that were themselves dropped for a stuck lick sensor.

ii.
```python
for i in range(ntrials):
    if i == 0:                      # previous trial outcome undefined
        continue
    ...
    inp[3] = rewarded[i - 1]
```

iii. Docstring: "The first trial of each session is dropped because 'previous trial outcome' (a required decoder input) is undefined for it." The instruction defines the input as binary (omitted = 0, rewarded = 1), so assigning 0 to a trial with no predecessor would create a false "omitted" label; the agent preferred to lose 152 of 12,216 trials (1.2%).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behaviour time series and the trial's reward-zone identity. The zone boundaries are the paper's constants — A: 80–130, B: 200–250, C: 320–370 cm — and the per-trial zone is inferred from the `reward_zone` flag channel plus the protocol's switch trial (see 10-a/10-b).

ii.
```python
REWARD_ZONES = {0: (80.0, 130.0), 1: (200.0, 250.0), 2: (320.0, 370.0)}  # A, B, C
...
pos = beh['position/data'][:]
rzone = beh['reward_zone/data'][:]
...
p = pos[s:t]
z0, z1 = REWARD_ZONES[zone_of_trial[i]]
```

iii. The agent took the coordinates from `behavior.reward_zone_dict` (entries `'X'`, `'Y'`, `'Z'` in the repo, relabelled A/B/C) and confirmed against the data: the minimum position at which the `reward_zone` flag is on is 80.0 cm on 3,542 trials, 200.0 on 3,388 and 320.0 on 3,433 (step 63), i.e. the flag switches on exactly at the documented zone starts.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest point of the zone: negative before the zone (`p − z0`), exactly 0 anywhere inside it, positive after it (`p − z1`). Computed vectorised over the trial with a nested `np.where`.

ii.
```python
dist = np.where(p < z0, p - z0, np.where(p > z1, p - z1, 0.0))
out[0] = discretize_distance(dist)
```

iii. This implements "distance to **any** location in the reward zone" from the instructions — the whole 50 cm zone collapses to distance 0, which is also why the specification has an explicit "0 cm" category.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories, assigned by explicit boolean masks rather than `np.digitize`, so that the exact value 0 gets its own class: `< −50 → 0`, `[−50, −10) → 1`, `[−10, 0) → 2`, `== 0 → 3`, `(0, 10] → 4`, `(10, 50] → 5`, `> 50 → 6`.

ii.
```python
def discretize_distance(d):
    """Signed distance (cm) to the nearest point of the reward zone -> 7 bins."""
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
```python
OUTPUT_VALUES[0] = ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in reward zone (0 cm)',
                    '0 to 10 cm', '10 to 50 cm', '> 50 cm']
```

iii. The bin edges are taken verbatim from the Decoder Task specification. The mask form is used because class 3 is the single value 0 (the whole reward zone), which cannot be expressed with a monotone `digitize` edge list without an epsilon hack. The resulting class fractions are 0.255 / 0.102 / 0.074 / 0.239 / 0.021 / 0.072 / 0.238.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `pos[s:t]`, the same sample window as `events[:, s:t]`, so it is aligned by construction and has the same number of timepoints.

ii.
```python
s, t = starts[i], stops[i]
p = pos[s:t]
...
out = np.empty((6, T), dtype=np.int64)
out[0] = discretize_distance(dist)
neural.append(np.ascontiguousarray(events[:, s:t], dtype=np.float32))
```

iii. Imaging and VR data are stored on a common frame grid in the NWB file (one behaviour sample per imaging frame, verified by the shared `dt` and the length checks), so identical indexing is exact alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behaviour time series (cm along the 450 cm corridor), used raw.

ii.
```python
pos = beh['position/data'][:]
...
p = pos[s:t]
```

iii. Straightforward — this channel is the VR track position, and the agent confirmed its within-lap range runs from ~0 to ~450 cm, with −50 only during the teleport/ITI period that is excluded.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing and discretising. Within the emitted windows position spans −9.1 to 449.5 cm (the slightly negative values come from the one sample before `trial_start` that the `[start−1, stop−1)` window includes); the open outer bins absorb these.

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
...
out[1] = np.digitize(p, POS_EDGES)
```

iii. The Methods describe a 450 cm track on which each lap starts at 0 cm; no transformation is needed for the decoder, only discretisation.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with edges `[90, 180, 270, 360]`, giving the five 90 cm bins required by the instructions (`< 90`, `90–180`, `180–270`, `270–360`, `> 360`), with the first and last bins left open.

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
out[1] = np.digitize(p, POS_EDGES)
OUTPUT_VALUES[1] = ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm']
```

iii. "Discretized into 5 equal-sized bins spanning the 450 cm track" ⇒ 90 cm per bin. `np.digitize` with 4 interior edges returns 0–4 directly (no −1 correction needed). Resulting fractions 0.217 / 0.176 / 0.232 / 0.227 / 0.149.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same index window `[s, t)` as the neural data — no separate alignment step.

ii.
```python
p = pos[s:t]
out[1] = np.digitize(p, POS_EDGES)
neural.append(np.ascontiguousarray(events[:, s:t], dtype=np.float32))
```

iii. As in 7-d: behaviour and imaging share the frame grid, and the extra trailing imaging frame present in 10 sessions is truncated before slicing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behaviour time series (a per-frame lick count, integer 0–6 in this dataset).

ii.
```python
lick = beh['lick/data'][:]
...
lk = (lick[s:t] > 0).astype(np.int64)
```

iii. The agent checked the unique values of the channel (`lick uniq [0..6]`, step 39) and identified it as a per-frame count from the capacitive sensor described in the Methods.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarised at > 0 (any lick in the frame ⇒ 1). Separately, trials whose sensor was stuck are removed altogether rather than binarised (see 1-e), so the emitted lick labels exclude the 81 corrupted trials. Class fractions 0.778 / 0.222.

ii.
```python
lk = (lick[s:t] > 0).astype(np.int64)
...
out[3] = lk
OUTPUT_VALUES[3] = ['no lick', 'lick']
```

iii. The instruction specifies "Lick, time-varying. 0 = no, 1 = yes", and the raw channel is a count, so thresholding at > 0 is required. The paper likewise "converted [remaining lick counts] to a binary vector".

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[s, t)` window as the neural data; licks are already sampled on the imaging frame grid (the Methods refer to "the 0.0645 s imaging frame samples"), so no resampling is required.

ii.
```python
lk = (lick[s:t] > 0).astype(np.int64)
out[3] = lk
```

iii. As above — a single common clock for all behavioural and imaging streams.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the `reward_zone` flag channel together with `position`: for each lap, the minimum position at which the flag is on is mapped to A/B/C by thresholds at 150 and 280 cm. Laps on which the flag never turns on (1,822 of 12,216) get no observation and are filled in from the session's switch structure.

ii.
```python
def zone_label(zone_start):
    """Map an observed reward-zone start position to the zone label A/B/C -> 0/1/2."""
    if zone_start < 150:
        return 0
    if zone_start < 280:
        return 1
    return 2
...
in_zone = rzone[s:t] > 0
if in_zone.any():
    observed_zone[i] = zone_label(pos[s:t][in_zone].min())
```

iii. The agent tabulated the flag-onset positions across the whole dataset (step 63): they cluster tightly at 80, 200 and 320 cm (3,542 / 3,388 / 3,433 trials) with only a handful at 90/210/330, so a coarse three-way threshold is unambiguous. Zone coordinates come from `behavior.reward_zone_dict`.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Per session: collect the observed labels; if they are all identical the session is a fixed-zone session and every lap gets that label. Otherwise it is a switch session — the code asserts there is exactly one change, and places the switch at **lap 30**, the protocol constant `change_trial=30` used by `behavior.get_reward_zones`, provided the observed bracket contains it (it always does); otherwise it falls back to the first observed post-switch lap. A final assertion checks the assignment agrees with every observed lap. The label is then broadcast over all timepoints of the trial.

ii.
```python
SWITCH_TRIAL = 30       # reward zone moves after 30 trials on switch sessions
...
known = np.nonzero(~np.isnan(observed_zone))[0]
labels = observed_zone[known]
zone_of_trial = np.empty(ntrials, dtype=np.int64)
if np.all(labels == labels[0]):
    zone_of_trial[:] = int(labels[0]); switch_trial = None
else:
    changes = np.nonzero(np.diff(labels))[0]
    assert len(changes) == 1, f'{path}: more than one reward zone switch'
    last_before = known[changes[0]]; first_after = known[changes[0] + 1]
    switch_trial = (SWITCH_TRIAL if last_before < SWITCH_TRIAL <= first_after
                    else first_after)
    zone_of_trial[:switch_trial] = int(labels[0])
    zone_of_trial[switch_trial:] = int(labels[-1])
assert np.all(zone_of_trial[known] == observed_zone[known])
...
out[4] = zone_of_trial[i]
```

iii. Docstring: "on switch sessions, the switch is placed at trial 30, the protocol value used by `behavior.get_reward_zones` (verified against the flagged trials in every session)". The agent scanned all 152 sessions (step 63) printing scene name, observed labels and observed switch index, and confirmed that fixed-zone scenes (`Env1_LocationA`, …) show one label and `*_to_*` scenes show exactly one change bracketing lap 30 (observed at 30 or 31 depending on whether lap 30 had a zone entry). This lets unobserved laps inherit the protocol-correct label. Resulting class fractions 0.331 / 0.336 / 0.332.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From the `Reward` event timestamps together with the `reward_zone` flag: a lap counts as rewarded if a reward event falls inside the lap's time window **and** the animal was flagged as being in the reward zone during the lap.

ii.
```python
reward_t = beh['Reward/timestamps'][:]
rzone = beh['reward_zone/data'][:]
...
in_zone = rzone[s:t] > 0
got_reward = np.any((reward_t >= tstamps[s]) & (reward_t < tstamps[t]))
rewarded[i] = int(in_zone.any() and got_reward)
```

iii. This reproduces `behavior.get_trial_types` from the paper's repo, which the agent read at step 36:
`isreward = (np.any(tmp_reward > 0) and np.any(tmp_rzone > 0)) * 1`. The agent also confirmed from the survey that `autoreward` is identically 0 in this dataset and that reward amounts are constant, so the event timestamps alone are sufficient.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The per-lap binary is broadcast over all timepoints of the trial. The `Reward` series has its own timestamps, so instead of snapping them to the sample grid the code compares them against the first and last behaviour timestamps of the lap. The overall rewarded fraction is 0.843 / 0.157 omitted, matching the paper's ~15% random omission.

ii.
```python
out[5] = rewarded[i]
OUTPUT_VALUES[5] = ['omitted', 'rewarded']
...
'rewarded_fraction': float(np.mean(rewarded)),
```

iii. Docstring: "Rewarded: reward delivered inside the reward zone on that trial (`behavior.get_trial_types`: reward AND reward-zone flag), i.e. 0 on the ~15% randomly omitted trials." The agent's dataset scan gave a rewarded fraction of 0.847 before trial curation, which it cross-checked against the paper's stated ~15% omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles these cases:
- **Extra imaging frame**: 10 m17/m18 sessions have exactly one more imaging frame than behaviour samples; the script asserts the difference is 0 or 1, asserts the last teleport is within the behaviour array, and truncates the ophys arrays.
- **Missing reward-zone flag** (1,822 laps): the zone label is inferred from the session's fixed zone or from the protocol switch at lap 30, with an assertion that the inference agrees with every observed lap.
- **Interpolated teleport sample**: dropped by the `[start−1, stop−1)` window.
- **Corrupted lick trials**: dropped (see 1-e).
- **Degenerate speed correlation**: `nan_to_num(..., nan=0.0)` so a constant-dF/F cell is not classed as an interneuron.
- Consistency is enforced with hard assertions rather than warnings: trial windows must not overlap, a lap must not span two environments, a session must not have more than one zone switch.
- No NaN imputation is done for behaviour, and none is needed: I verified there are no NaNs in `position`, `speed` or `lick` inside any emitted window.

ii.
```python
nframes = len(pos)
assert 0 <= F.shape[1] - nframes <= 1, (path, F.shape, nframes)
assert stops[-1] <= nframes
F = F[:, :nframes]; Fneu = Fneu[:, :nframes]
...
assert np.all(starts[1:] > stops[:-1]) and starts[0] >= 0
assert len(env_vals) == 1, f'{path}: trial {i} spans environments {env_vals}'
assert len(changes) == 1, f'{path}: more than one reward zone switch'
assert np.all(zone_of_trial[known] == observed_zone[known])
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
```

iii. Each assertion follows from an explicit whole-dataset check in the trajectory: step 91 enumerated the length mismatches before the assertion was written (the edit at step 94 added it), step 63 checked for multi-environment trials (0 found) and NaNs (0 found), and step 63/65 verified the single-switch structure of every session. The design is "fail loudly on anything unexpected" — the conversion is cheap enough (~2.5 min) to re-run if an assertion fires.

## 13-a. What are the most time-consuming steps of the code?

i. In rough order: (1) reading the `Fluorescence` and `Neuropil` arrays out of HDF5 — tens of thousands of frames × up to ~2,500 ROIs per session, ~0.5 s per plane per session but the dominant I/O cost; (2) the per-trial dF/F + OASIS deconvolution (`gaussian_filter1d` / `minimum_filter1d` / `maximum_filter1d` / `dcnv.oasis`), ~1.3 s per session single-threaded; (3) pickling the 9.5 GB result. There is a single pass over the data, parallelised 12 ways with `ProcessPoolExecutor`; the full conversion of 152 sessions takes ~2.5 min wall clock.

ii.
```python
ctx = multiprocessing.get_context('spawn')
with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
    for i, res in enumerate(pool.map(load_session, files)):
```
```python
with open(args.out, 'wb') as fh:
    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent timed the pieces on one session before writing the script (step 60: `load 0.48 s`, `neuropil 0.09 s`, `dff+oasis 1.29 s`), checked available RAM/cores (`free -g`, `nproc`), and sized the worker pool accordingly; it first tried 8/6 workers, hit a `BrokenProcessPool` from forking the numba-jitted OASIS, and switched to `spawn`.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python loops remain, all over trials: the dF/F loop in `compute_dff_and_events`, the per-trial task-variable loop, and the emit loop. The dF/F loop is intrinsic (the baseline is defined per trial, and the filters need contiguous per-trial segments). The task-variable loop (`in_zone`, `got_reward`, `env_vals`, `lick_error`) could be vectorised with `np.add.reduceat`-style segment reductions over the session, and `got_reward` could be replaced by a single `np.searchsorted` of all reward times against all trial boundaries instead of an O(n_trials × n_rewards) scan. The emit loop's `discretize_distance` / `np.digitize` calls could be applied once to the whole session and then sliced. The interneuron correlation, by contrast, is already fully vectorised as a matrix product rather than the reference's per-cell `np.corrcoef` loop.

ii.
```python
for i, (s, t) in enumerate(zip(starts, stops)):
    in_zone = rzone[s:t] > 0
    got_reward = np.any((reward_t >= tstamps[s]) & (reward_t < tstamps[t]))
    ...
```
```python
dff_c = dff_v - dff_v.mean(axis=1, keepdims=True)
speed_corr = (dff_c @ spd_c) / denom      # vectorised over cells
```

iii. Not discussed explicitly by the agent; the loops are over ~80 trials per session, so they are negligible next to the per-session HDF5 read and deconvolution, and the parallel map already saturates the machine.

## 13-c. What processing does the code repeat multiple times?

i. Very little. Each NWB file is opened exactly **once** — there is no separate survey pass — and F/Fneu are read once per plane. Within a session, the trial window `[s, t)` is re-sliced in three separate loops (dF/F, task variables, emit), and `rzone[s:t]` in particular is computed twice (once for `rewarded`, once for `observed_zone`, in the same iteration). The exploratory survey the agent used to make its decisions was run as throwaway scripts in `/tmp`, not baked into `convert_data.py`.

ii.
```python
for i, (s, t) in enumerate(zip(starts, stops)):     # pass 1: task variables
    in_zone = rzone[s:t] > 0
...
for i in range(ntrials):                            # pass 2: emit
    s, t = starts[i], stops[i]
    p = pos[s:t]
```

iii. The two passes are needed because the reward-zone label of a trial depends on session-level structure (which laps observed which zone, and where the switch is), so all trials must be surveyed before any can be emitted. The agent kept this inside one file read rather than doing a second pass over the dataset.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Minor items:
- dF/F and OASIS events are computed for **all** curated ROIs, including the 409 cells later dropped as putative interneurons (unavoidable — the interneuron test needs the dF/F).
- Events are computed for all 12,216 laps, including the 233 later dropped (152 first laps + 81 stuck-lick laps).
- The full-session `dff` array is materialised only to compute one correlation vector, then deleted; only `events` is kept.
- `cellplane` / `n_neurons_per_plane`, `scene`, `date`, `trial_indices`, `rewarded_fraction` and the rest of `session_info` are provenance metadata that the decoder never reads.
- `np.ascontiguousarray` copies each trial's neural slice (needed, since a view of the session array would pickle the whole array).

ii.
```python
dff, events = compute_dff_and_events(F, Fneu, starts, stops, frame_rate)
...
del dff, dff_v, dff_c
...
'n_neurons_per_plane': np.bincount(cellplane, minlength=len(planes)).tolist(),
```

iii. The agent explicitly economised where it mattered — subsetting to `iscell` before computing dF/F, storing neural data as float32 (halving the pickle), deleting `F`, `Fneu` and `dff` as soon as they are consumed — which is what keeps peak memory low enough for 12 concurrent workers. The remaining discarded work is a few percent of runtime.
