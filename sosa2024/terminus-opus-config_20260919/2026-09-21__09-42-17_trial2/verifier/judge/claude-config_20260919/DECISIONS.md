# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs every NWB file under `/app/data/sub-*/` (152 files, 11 subject directories) and opens each one **directly with `h5py`** rather than `pynwb`. One file = one session; each session is converted independently in a `multiprocessing` (spawn) `Pool` of 8 workers, and the per-session results are collected, sorted by `(subject, session_id)`, and assembled into the target dictionary. From each file it reads: `identifier` (scene name), `general/subject/subject_id`, `general/session_id`, `general/optophysiology/ImagingPlane/{location, imaging_rate}`, all of `processing/behavior/BehavioralTimeSeries`, and `processing/ophys/{Fluorescence,Neuropil,ImageSegmentation}`.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
...
jobs = [(f, i < show_n, args.neural_signal) for i, f in enumerate(files)]
ctx = mp.get_context('spawn')
with ctx.Pool(args.nproc, maxtasksperchild=1) as pool:
    for i, res in enumerate(pool.imap(_worker, jobs)):
        if res is not None:
            results.append(res)
```
```python
with h5py.File(fname, 'r') as f:
    scene = f['identifier'][()].decode().split('/')[-1]
    subject = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    region = f['general/optophysiology/ImagingPlane/location'][()].decode()
    rate = float(f['general/optophysiology/ImagingPlane/imaging_rate'][()])
    beh = load_behavior(f)
    F, Fneu, plane_idx = load_fluorescence(f)
```
```python
def load_behavior(f):
    b = f['processing/behavior/BehavioralTimeSeries']
    beh = {k: b[k]['data'][:] for k in
           ['position', 'environment', 'lick', 'reward_zone', 'speed',
            'teleport', 'trial_start', 'trial number', 'autoreward', 'scanning']}
    beh['t'] = b['position']['timestamps'][:]
```

iii. From CONVERSION_NOTES Step 2: "`/app/data` is DANDI dandiset 001361 (v0.251124.0550): 11 subject folders ... one NWB file per session ... 152 files total." The AI cross-checked the file count against the paper (14 days/mouse, m11 starting on day 3 → 152 released sessions) and against the paper's trial count (12,376 vs 12,216 found, explained as two sessions not in the release). `h5py` was chosen over `pynwb` for speed and because the NWB layout was explicitly enumerated and verified first ("NWB contents (verified with h5py)").

## 1-b. How are the data split into subjects?

i. Subject identity is read from inside each file (`general/subject/subject_id`), not from the directory name. The unique subject strings are sorted numerically (`m3, m4, m7, m11, ...`) to form `subjects`, and `subject_idx` indexes into that list per session.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
subjects = sorted({r['subject'] for r in results},
                  key=lambda s: int(re.sub(r'\D', '', s)))
data['subject_idx'] = np.array([subjects.index(r['subject']) for r in results],
                               dtype=np.int64)
```

iii. Step 2 of the notes records "11 subject folders (`sub-m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19`)" and Step 3 matches that to the paper's "n = 11 mice" switch cohort (the separate 3-mouse fixed-condition group is not in the release). Reading the ID from the file rather than the path is the more authoritative source and was verified to agree with the directory names.

## 1-c. How are the data split into sessions?

i. One session = one NWB file = one experiment day. Sessions are not merged or split; no across-mouse or across-day cell alignment is attempted. Sessions are ordered by `(subject, session_id)` and each contributes one entry to `neural`/`input`/`output`/`subject_idx`/`brain_region_idx`, plus a `metadata['session_info']` record.

ii.
```python
results.sort(key=lambda r: (r['subject'], r['session_id']))
data = {
    'neural': [r['neural'] for r in results],
    'input':  [r['input']  for r in results],
    'output': [r['output'] for r in results],
    ...
    'session_info': [
        {'file': r['file'], 'subject': r['subject'], 'day': r['session_id'],
         'scene': r['scene'], 'n_neurons': r['n_neurons'], ...}
        for r in results],
```

iii. Step 2: "`sub-<mouse>_ses-<NN>_behavior+ophys.nwb` (NN = experiment day, 01-14)"; Step 9 verifies 14 sessions per mouse except m11 (12), matching the paper's statement that imaging for m11 started on day 3. Step 5 Key Decision 9: "All 152 sessions and all 11 mice are converted (no session-level exclusions; the paper uses all task days)."

## 1-d. How are the data split into trials?

i. A trial is the half-open window `[trial_start_index, teleport_index)` taken from the binary `trial_start` and `teleport` behavior time series. The teleport frame itself is excluded because its position has been interpolated between the end of the track and the ITI. The code asserts that the number of starts equals the number of teleports and that every teleport follows its start.

ii.
```python
# ---- trials: [trial_start, teleport); the teleport frame has an interpolated position
starts = np.where(beh['trial_start'] > 0)[0]
stops = np.where(beh['teleport'] > 0)[0]
assert len(starts) == len(stops) and np.all(stops > starts)
# a trial whose teleport was cut off by the truncation above is dropped
valid = stops < nframes
if not np.all(valid):
    print(f'  {os.path.basename(fname)}: dropping {int((~valid).sum())} trial(s) '
          f'truncated at the end of the recording')
    starts, stops = starts[valid], stops[valid]
ntrials_raw = len(starts)
```
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    T = e - s
    pos = beh['position'][s:e].astype(np.float64)
```

iii. Step 1 of the notes identifies `sess.trial_start_inds` / `sess.teleport_inds` as the reference repo's trial definition, and Step 2 documents the edge case: "The frame flagged by `teleport` already has an interpolated position between the end of the track and -50 (e.g. 448.8 -> 200.8 -> -50), so trials must be taken as `[trial_start_ind, teleport_ind)` (matches `f_[:, start-1:stop-1]` windows in the reference dff)." The NWB `trial number` stream was not used for segmentation. Step 10 Check 3 records "temporal alignment ... `[start, stop)`, same length, teleport sample excluded; behaviour uses the same window so all streams share one index — yes".

## 1-e. How are trials filtered based on quality controls?

i. Three filters:
1. **Lick-sensor error trials are dropped**: a trial is bad if more than 30% of its frames have a cumulative lick count > 2 — the paper's own criterion. This removes exactly 81 of 12,216 trials, reproducing the paper's published count.
2. Trials whose teleport index falls past the truncation point of a behavior/imaging length mismatch are dropped (never triggered in practice).
3. Sessions left with fewer than 2 usable trials are skipped entirely (never triggered).

There is **no minimum trial-length filter**; the shortest surviving trial is 96 frames.

ii.
```python
LICK_ERROR_FRAC = 0.30             # >30% of frames with cumulative lick count > 2
LICK_ERROR_COUNT = 2
...
# lick-sensor error trials (paper: >30% of frames with cumulative lick count > 2)
lick_bad = np.array([(beh['lick'][s:e] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC
                     for s, e in zip(starts, stops)])

for i, (s, e) in enumerate(zip(starts, stops)):
    if lick_bad[i]:
        continue
...
if len(neural) < 2:
    print(f'  !! {os.path.basename(fname)}: only {len(neural)} usable trials, skipping')
    return None
```

iii. Step 4 discrepancy table: the repo's `glmUtils.get_timeseries_data` uses 0.35 and `behavior.correct_lick_sensor_error` defaults to 0.5, but the Methods say ">30% of the ... frame samples in the trial containing a cumulative lick count >2" and report "n = 81 out of 12,376 trials"; the AI found "With threshold 0.30 I find **exactly 81** bad trials", so it adopted the paper's 30% rule. The deviation from the reference repo (dropping the trial instead of NaN-ing the lick trace) is explicitly justified in Step 10 Check 3: "error trials are *dropped* (a decoder output cannot be NaN)". No speed-based sample masking is applied (Key Decision 6): "keeping all within-trial samples is required to keep the neural, input and output series aligned and to retain the <2 cm/s speed class."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces: `processing/ophys/Fluorescence/plane<N>/data` (F) and `processing/ophys/Neuropil/plane<N>/data` (Fneu), restricted to ROIs with `ImageSegmentation/PlaneSegmentation/iscell[:,0] == 1`. Planes are pooled by concatenation along the cell axis (verified to be the PlaneSegmentation row order). The NWB `Deconvolved` series is deliberately **not** used.

ii.
```python
def load_fluorescence(f):
    """Pooled F / Fneu of manually curated cells (iscell), planes concatenated."""
    seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
    iscell = seg['iscell'][:, 0].astype(bool)
    plane_idx = seg['planeIdx'][:]
    Fs, Fns = [], []
    for p in np.unique(plane_idx):
        Fs.append(f[f'processing/ophys/Fluorescence/plane{p}/data'][:].T)
        Fns.append(f[f'processing/ophys/Neuropil/plane{p}/data'][:].T)
    F = np.concatenate(Fs, axis=0).astype(np.float32)
    Fneu = np.concatenate(Fns, axis=0).astype(np.float32)
    assert F.shape[0] == len(iscell), (F.shape, len(iscell))
    return F[iscell], Fneu[iscell], plane_idx[iscell]
```

iii. Step 1 notes: "**dF/F must be computed by me**: NWB contains raw `Fluorescence` (F) and `Neuropil` (Fneu) plus suite2p `Deconvolved` (computed from raw F, not from the paper's dF/F). The paper's pipeline is F -> neuropil subtraction -> maximin dF/F -> 2-sample smoothing -> OASIS deconvolution (`events`)." Step 4: "The NWB `Deconvolved` series is *not* the paper's `events` and is not used."

## 2-b. How is the `neural` data processed?

i. A line-by-line reimplementation of `reward_relative.preprocessing.dff(..., neuropil_method='subtract', baseline_method='maximin', neu_coef=0.7, subtract_baseline=True)`: restrict samples to within-trial windows (everything else NaN), subtract `0.7 * Fneu`, add each trial's mean neuropil back, take a per-trial maximin baseline (Gaussian smoothing sigma = 15 frames, then a 300-frame minimum filter followed by a 300-frame maximum filter ≈ the Methods' 20 s window), form `(F - baseline)/|baseline|`, and smooth per trial with a 2-frame (~0.129 s) Gaussian. OASIS deconvolution (`suite2p.extraction.dcnv.oasis`, `tau = 0.7`, per-plane rate 15.5078 Hz) is also implemented and is run on every session.

**The saved neural stream is the dF/F, not the deconvolved `events`.** `--neural-signal events` switches to the deconvolved trace; the default is `dff`. Two further simplifications relative to the reference: `keep_teleports` is always effectively `False` (the reference consults a per-mouse/per-day `teleport_metadata` table), and computation is in float32 rather than float64.

ii.
```python
def compute_dff(F, Fneu, starts, stops):
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]

    # neuropil subtraction
    f_ -= NEU_COEF * fneu_

    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        # add back the per-trial mean neuropil so dF/F is close to true dF/F
        f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        seg = nansmooth(f_[:, s:e], BASELINE_SMOOTH_SIGMA, axis=-1)
        seg = ndimage.minimum_filter1d(seg, BASELINE_FILTER_WIN, axis=-1)
        seg = ndimage.maximum_filter1d(seg, BASELINE_FILTER_WIN, axis=-1)
        flow[:, s:e] = seg

    mask = ~np.isnan(f_[0, :])
    dff = np.full(F.shape, np.nan, dtype=np.float32)
    dff[:, mask] = (f_[:, mask] - flow[:, mask]) / np.abs(flow[:, mask])

    for s, e in zip(starts, stops):
        dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)
    return dff


def deconvolve(dff, starts, stops):
    from suite2p.extraction import dcnv
    spks = np.zeros(dff.shape, dtype=np.float32)
    for s, e in zip(starts, stops):
        spks[:, s:e] = dcnv.oasis(
            np.ascontiguousarray(dff[:, s:e], dtype=np.float32),
            2000, TAU, FRAME_RATE)
    return spks
```
```python
dff_kept = dff[keep_cells]
events = deconvolve(dff_kept, starts, stops)
activity = events if neural_signal == 'events' else np.nan_to_num(dff_kept)
```

iii. Step 3 quotes the Methods verbatim for the maximin/20 s/2-sample-Gaussian/OASIS pipeline, and Step 10 Check 3 certifies the dF/F stage as a "line-by-line reimplementation (same constants, same `nansmooth`)". For the dF/F-vs-events choice, Step 12 Check 1 reports a controlled comparison in which only the neural array changed (6 sessions from 6 mice): dF/F won on 5/6 outputs (mean val. balanced accuracy 0.682 vs 0.640), and on the full dataset dF/F beat events on all six (e.g. position 0.770 vs 0.643, distance 0.632 vs 0.532). Justification given: "Both signals are part of the reference processing pipeline ...; the paper itself uses dF/F for time-resolved analyses ... Deconvolution discards the amplitude information in the decay of each transient and yields a signal that is zero on ~55% of samples, which costs a per-timepoint decoder accuracy." For `keep_teleports`, Step 4 states: "Irrelevant here: the decoder uses only within-trial samples, so I always compute the baseline per trial."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both from the paper: (1) suite2p manual curation, keeping only ROIs with `iscell[:,0] == 1`; (2) exclusion of putative interneurons — cells whose (smoothed) dF/F has Pearson r > 0.5 with running speed computed over all within-trial samples. The correlation is computed with a single vectorized matrix product rather than a per-cell loop. Result: 138,276 neurons kept, 402 (0.29%) excluded as interneurons, 154–2323 per session.

ii.
```python
INTERNEURON_R_THRESH = 0.5         # Pearson r(dF/F, speed) > 0.5 -> putative interneuron
...
iscell = seg['iscell'][:, 0].astype(bool)
...
return F[iscell], Fneu[iscell], plane_idx[iscell]
```
```python
# ---- neuron curation: exclude putative interneurons (r(dF/F, speed) > 0.5)
in_trial = np.zeros(nframes, dtype=bool)
for s, e in zip(starts, stops):
    in_trial[s:e] = True
d = dff[:, in_trial]
sp = beh['speed'][in_trial].astype(np.float32)
dz = d - d.mean(axis=1, keepdims=True)
sz = sp - sp.mean()
denom = (np.sqrt((dz ** 2).sum(axis=1)) * np.sqrt((sz ** 2).sum()))
with np.errstate(invalid='ignore', divide='ignore'):
    r_speed = (dz @ sz) / denom
r_speed = np.nan_to_num(r_speed, nan=0.0)
keep_cells = r_speed <= INTERNEURON_R_THRESH
```

iii. Step 3 Curation: "1. suite2p manual curation -> `iscell` ...; 2. Exclude putative interneurons: Pearson r(dF/F, speed) > 0.5 over the session", quoting the Methods "a Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed, excluding 0.42 ± 0.85% of cells". Step 9 compares 0.29% excluded here against that 0.42 ± 0.85%, and 154 minimum neurons/session against the paper's 155. Step 10 Check 2 records that an independent check initially disagreed by one cell because it correlated *unsmoothed* dF/F with speed; the converter's order (smooth, then correlate) was confirmed correct against the Methods wording and the check script was fixed. Place-cell selection was deliberately not applied (Step 3: "not appropriate for a population decoder").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No extra alignment is needed: the VR/behavior streams in the NWB release are already interpolated onto imaging frame times, so slicing every stream with the same `[s:e)` index range (where `s` is the `trial_start` frame) aligns the neural data to trial start by construction. `metadata['off_start'] = 0.0`, `off_end = None` (variable-length, self-paced laps).

ii.
```python
act = np.ascontiguousarray(activity[:, s:e], dtype=np.float32)
...
'temporal_alignment_event': (
    'start of trial = entry into the virtual linear track at position 0 cm '
    '(NWB behavior `trial_start` event); trials end at the `teleport` event '
    '(entry into the inter-trial teleport period), which is excluded'),
'off_start': 0.0,
'off_end': None,   # trials have variable length (self-paced laps)
```

iii. Step 5 Key Decision 2: "VR data in the NWB file are already interpolated onto imaging frames, which guarantees exact temporal alignment." Step 10 Check 5 confirms the frame period is 64.4836 ms in every session, so behavior and ophys indices are the same index. Step 10 Check 2 spot-checked converted neural values against dF/F recomputed independently from raw F/Fneu with `np.allclose` at specific (trial, neuron, timepoint) coordinates.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame: 15.5078125 Hz per plane → 64.4836 ms, stored in `metadata['time_bin_size']`. **No rebinning, no resampling, no smoothing beyond the reference dF/F pipeline.** The frame rate is a hard-coded module constant; the NWB `imaging_rate` attribute is read but only reported (it is the scanner rate, 31.0156 Hz on the two 2-plane mice, so the per-plane rate is the same 15.5078 Hz everywhere).

ii.
```python
FRAME_RATE = 15.5078125            # Hz, imaging frame rate per plane (NWB ImagingPlane)
DT = 1.0 / FRAME_RATE              # 64.48 ms
...
'time_bin_size': 1000.0 / FRAME_RATE,          # ms (64.4836 ms imaging frame)
'imaging_rate_hz': FRAME_RATE,
```

iii. Step 3: "**Temporal binning**: native imaging frame (~64.5 ms). No coarser binning in the paper for time-series analyses (only 10 cm spatial bins for spatial analyses)", quoting "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate." Step 5 Key Decision 2 makes it explicit: "Time bin = 1 imaging frame = 64.48 ms (15.5078 Hz), i.e. no re-binning." Step 10 Check 5 notes the two-plane case: "`imaging_rate` attribute is 31.0 Hz total but the per-plane series are 15.5 Hz — verified against the behaviour timestamps (dt = 64.4836 ms in every session)." The AI initially considered coarser binning to shrink the pickle (trajectory step 50) and then rejected it after checking that the decoder streams sessions to GPU (step 65).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` array of the `position` behavior time series (all behavior streams in the file share one timestamp vector, which also matches the imaging frames).

ii.
```python
beh['t'] = b['position']['timestamps'][:]
...
tt = beh['t'][s:e] - beh['t'][s]
inp[0] = tt
```

iii. Step 5 maps `input[0]` from the frame index within the trial, `(i - si) * 0.0644836` s, and the implementation uses the actual stored timestamps, which the AI verified are exactly `k * 64.4836 ms` ("Step 10 Check 2: `input[0]` == `k * 64.4836 ms` for every checked trial — PASS"). Step 2 records that all behavior streams are "sampled at imaging frames, 0.0644836 s".

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtraction of the trial's first timestamp, giving a vector starting at exactly 0 for every trial. No other processing; stored as float32. Range over the full dataset is [0, 216.5] s.

ii.
```python
tt = beh['t'][s:e] - beh['t'][s]
inp = np.empty((len(INPUT_NAMES), T), dtype=np.float32)
inp[0] = tt
```

iii. Follows directly from the instruction "Time from start of trial in seconds (continuous, time-varying)" with the trial-start alignment event. Step 9 reports the range and flags the maximum (216.5 s) as a genuine trial "where the mouse stopped" rather than an artefact (Step 10 Check 5: "very long trials (up to 216 s) | kept; they are genuine trials in which the mouse stopped running").

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is sliced with the same `[s:e)` index range as the neural array, so alignment is automatic. Before any slicing, if the behavior stream and the imaging stream differ in length (10 two-plane sessions have exactly one extra behavior sample), both are truncated to the common length, with an assertion that the difference is ≤ 5 samples.

ii.
```python
n_beh = len(beh['position'])
n_ophys = F.shape[1]
nframes = min(n_beh, n_ophys)
if n_beh != n_ophys:
    assert abs(n_beh - n_ophys) <= 5, (n_beh, n_ophys)
    print(f'  {os.path.basename(fname)}: behavior has {n_beh} samples, imaging '
          f'{n_ophys}; truncating both to {nframes} (reference one-frame correction)')
    for k in list(beh.keys()):
        beh[k] = beh[k][:nframes]
    F = F[:, :nframes]
    Fneu = Fneu[:, :nframes]
```
```python
assert a.shape[1] == ii.shape[1] == oo.shape[1]
```

iii. Step 9: "**Bug found and fixed during the first full run**: 10 sessions of the two two-plane mice ... have one *more* behaviour sample than imaging frames ... This is precisely the 'one frame correction ... scan stopping mid frame' case handled in the reference `TwoPUtils.preprocessing.vr_align_to_2P`." Step 10 Check 3 rates this "yes" against the reference behaviour.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavior time series (0 = ENV1, 1 = ENV2). It is cross-validated against the environment parsed from the scene name in `identifier` (e.g. `Env1_C_to_Env2_A`), but the value actually written is taken from the data stream.

ii.
```python
beh = {k: b[k]['data'][:] for k in
       ['position', 'environment', ...]}
...
env_stream = np.array([np.median(beh['environment'][s:e])
                       for s, e in zip(starts, stops)])
...
inp[1] = env_stream[i]
```

iii. Step 2: "`environment` (0=ENV1, 1=ENV2, -1 pre-sync)". Step 4: "`environment` stream matches scene-derived env on 12,216/12,216 trials", so the two independent sources agree exactly; the AI used the stream and kept `parse_scene`'s environment only as a check.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial median of the `environment` stream is taken and broadcast across all timepoints of the trial (the stream is constant within a trial, so the median is just that value; environment switches only happen at trial boundaries, on the `Env1_X_to_Env2_Y` scenes). No recoding is done — the raw 0/1 values are already the required binary encoding.

ii.
```python
env_stream = np.array([np.median(beh['environment'][s:e])
                       for s, e in zip(starts, stops)])
...
inp[1] = env_stream[i]
```

iii. Step 5 Key Decision 7: "Per-trial variables are broadcast over time (environment, trial number, previous outcome, reward zone location, reward outcome) so every entry is time-varying, as the spec prefers." Taking a per-trial summary rather than the raw samples guards against any single-sample glitch; the AI verified with an independent script that "per-trial variables (`input[1..3]`, `output[4..5]`) constant within each trial — PASS for all 12,135 trials" (Step 10 Check 2). Note that the Step 5 mapping table describes this as the "mode over trial" while the code takes the median; for a constant stream the two agree.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Not from any stored stream — it is the positional index of the trial within the session, i.e. the index into the `trial_start`/`teleport` arrays. The stored `trial number` behavior stream is loaded but never used.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    if lick_bad[i]:
        continue
    ...
    inp[2] = i
```

iii. Step 5 maps `input[2]` to "0-based index of the trial within the session (after dropping lick-error trials the *original* index is kept)". Keeping the original index (rather than renumbering the surviving trials) preserves the true ordinal position of each trial in the session, which is what "trial number" means for the reward-zone switch at trial 30 and for within-session learning effects. Verified in Step 10 Check 2 ("trial number = raw trial index — PASS").

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond assigning the loop index; the value is constant across all timepoints of the trial and stored as float32. Range across the dataset is [0, 99].

ii.
```python
inp[2] = i
```

iii. Same as 5-a. The per-session ranges in `verification_full_out.txt` (`[0, 79]` for most sessions, `[0, 99]` for the long ones, `[0, 40]`/`[0, 49]` for the short ones) confirm the numbering follows the raw trial count including dropped trials.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The sparse `Reward` TimeSeries (one entry per delivered reward, with its own timestamps) expanded onto the behavior frame grid via `searchsorted`, combined with the `reward_zone` stream. A trial counts as rewarded when a reward was delivered **and** the reward zone was entered — the rule in the repo's `behavior.get_trial_types`. The previous trial's value is then broadcast onto the current trial.

ii.
```python
rew_t = b['Reward']['timestamps'][:]
rew_bin = np.zeros_like(beh['position'])
if len(rew_t):
    idx = np.searchsorted(beh['t'], rew_t)
    idx = np.clip(idx, 0, len(rew_bin) - 1)
    rew_bin[idx] = 1
beh['reward'] = rew_bin
```
```python
# reward outcome, as in behavior.get_trial_types: reward delivered AND zone entered
isreward = np.array([(beh['reward'][s:e].sum() > 0) and
                     (beh['reward_zone'][s:e].sum() > 0)
                     for s, e in zip(starts, stops)], dtype=np.int64)
```

iii. Step 4: "`autoreward` is all zeros in every session ... Reward outcome is therefore taken as in `behavior.get_trial_types` (reward delivered AND reward zone entered), which does not need autoreward." Step 2 documents `Reward` as "a sparse TimeSeries: one entry per delivered reward with timestamps; data = 0.004 mL". The resulting reward rate, 84.64%, matches the paper's ~85% (~15% omission).

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `input[3] = isreward[i-1]` for every trial `i > 0`, broadcast over all timepoints. **For the first trial of a session the value is set to 1 (rewarded)**, not 0. "Previous" means the previous *raw* trial, so if trial `i-1` was dropped as a lick-error trial its outcome is still what trial `i` sees.

ii.
```python
inp[3] = isreward[i - 1] if i > 0 else 1   # see notes: warm-up trials precede
```

iii. Step 5 Key Decision 8: "**First trial of a session**: `prev_trial_outcome = 1`. Rationale: 30 warm-up trials with the same reward zone immediately precede each imaging session, and 84.7% of trials are rewarded." This is grounded in the Methods ("Before the imaging session, mice were provided 30 'warm-up' trials using the task and reward zone from the previous day"). Step 12 Check 3 records a leakage check: "I verified with `np.allclose` that input[3] of trial i equals output[5] of trial i-1 and never equals output[5] of trial i unless the outcomes genuinely coincide", and Step 10 Check 2 re-derived `input[3]` from the raw NWB files for all 152 sessions / 12,135 trials with 0 failures.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` behavior time series plus the reward zone for that trial. The zone is **not** derived from the `reward_zone` stream: it is parsed from the scene name stored in the NWB `identifier` (e.g. `Env1_LocationB_to_A`) and switched after trial 30 on switch sessions, exactly as the reference repo's `behavior.get_reward_zones` does. Zone coordinates are A = 80–130, B = 200–250, C = 320–370 cm.

ii.
```python
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
CHANGE_TRIAL = 30                  # reward zone / environment switch trial (behavior.py)

def parse_scene(scene):
    m = re.match(r'^Env(\d)_Location([ABC])$', scene)
    if m:
        return m.group(2), None, int(m.group(1)) - 1, None
    m = re.match(r'^Env(\d)_Location([ABC])_to_([ABC])$', scene)
    if m:
        e = int(m.group(1)) - 1
        return m.group(2), m.group(3), e, e
    m = re.match(r'^Env(\d)_([ABC])_to_Env(\d)_([ABC])$', scene)
    if m:
        return m.group(2), m.group(4), int(m.group(1)) - 1, int(m.group(3)) - 1
    raise ValueError(f'unrecognized scene name: {scene}')

def zone_per_trial(scene, ntrials):
    z1, z2, e1, e2 = parse_scene(scene)
    if z2 is None:
        return [z1] * ntrials, [e1] * ntrials
    n1 = min(CHANGE_TRIAL, ntrials)
    return ([z1] * n1 + [z2] * (ntrials - n1),
            [e1] * n1 + [e2] * (ntrials - n1))
```
```python
zone = REWARD_ZONES[zones[i]]
dist = signed_distance_to_zone(pos, zone)
```

iii. Step 1 identifies `get_reward_zones(sess, rz_dict, change_trial=30)` as the reference function and notes "`reward_zone_dict`: X/A=[80,130], Y/B=[200,250], Z/C=[320,370]; on switch scenes (`*_A_to_B` etc) first `change_trial=30` trials get the first zone, the rest the second". Step 2 identified that "`identifier` = original data path, whose last element is the **scene name** ... This is exactly the `sess.scene` used by `behavior.get_reward_zones`." Step 4 validates the derived zones empirically: "reward delivery positions fall inside the scene-derived zone on 10,334/10,342 rewarded trials" and "`environment` stream matches scene-derived env on 12,216/12,216 trials"; the 8 exceptions are "<15 cm past the zone end, i.e. frame-interpolation edge cases".

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the **nearest point** of the 50 cm zone: `pos - zone_start` when before the zone (negative), `pos - zone_end` when past it (positive), exactly 0 while inside. The continuous value is then discretized into the 7 specified bins.

ii.
```python
def signed_distance_to_zone(pos, zone):
    """Signed distance (cm) from position to the nearest point of the reward zone.

    0 inside the zone, negative before the zone, positive past the zone.
    (Reference GLM code uses pos - zone_start; here we use the distance to *any*
    location in the zone, as required by the decoder output specification.)
    """
    lo, hi = zone
    d = np.zeros_like(pos)
    before = pos < lo
    after = pos > hi
    d[before] = pos[before] - lo
    d[after] = pos[after] - hi
    return d
```

iii. Step 10 Check 3 flags this as a deliberate, spec-driven refinement of the reference: "distance measured to the *nearest point* of the zone (decoder spec asks for distance to 'any location in the reward zone')" — the repo's `glmUtils.get_timeseries_data` instead uses `rel_pos = pos - zone start`. The instruction text ("Distance to any location in the reward zone") supports the nearest-point definition.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories, assigned by explicit boolean masks rather than `np.digitize`, with an assertion that every sample was assigned: `< -50 → 0`, `[-50, -10) → 1`, `[-10, 0) → 2`, `== 0 → 3` (in zone), `(0, 10] → 4`, `(10, 50] → 5`, `> 50 → 6`. Resulting fractions on the full dataset: [0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243].

ii.
```python
def bin_distance(d):
    out = np.full(d.shape, -1, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    assert np.all(out >= 0)
    return out
```
```python
OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to <0 cm', 'in zone (0 cm)',
     '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'], ...
```

iii. From the trajectory (step 69): "Bins: <-50 →0; [-50,-10) →1; [-10,0) →2; ==0 →3; (0,10] →4; (10,50] →5; >50 →6. Edge definitions: spec says 1: -50 to -10, 2: -10 to <0, 4: >0 to +10, 5: +10 to +50, 6: >+50." Step 10 Check 2 verifies the boundary case independently: "`output[0] == 3` <=> `zone_start <= pos <= zone_end` — PASS", and the `--show-processing` plots overlay the raw distance with its bin assignment.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same `[s:e)` slice of the same frame-indexed `position` array that produced the neural slice; nothing further is needed. Shape equality between the neural, input and output arrays is asserted for every trial.

ii.
```python
pos = beh['position'][s:e].astype(np.float64)
...
out[0] = bin_distance(dist)
...
act = np.ascontiguousarray(activity[:, s:e], dtype=np.float32)
assert a.shape[1] == ii.shape[1] == oo.shape[1]
```

iii. Step 5 Key Decision 2: the VR data in the NWB release "are already interpolated onto imaging frames, which guarantees exact temporal alignment." The processing plots (`processing_*_trials.png`) overlay position in cm with `position_bin × 90` and mark the in-zone bin to visually confirm there is no shift.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior time series (cm along the 450 cm virtual corridor), used directly.

ii.
```python
beh = {k: b[k]['data'][:] for k in ['position', ...]}
...
pos = beh['position'][s:e].astype(np.float64)
```

iii. Step 2: "`position` (cm; -500 before VR/imaging TTL sync)", with the pre-sync samples verified to lie outside every trial ("133 pre-TTL frames (pos = -500, env = -1) | always before the first `trial_start`, never inside a trial (verified for all 152 sessions)", Step 10 Check 5).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing and discretization into 5 equal 90 cm bins.

ii.
```python
TRACK_LENGTH = 450.0

def bin_position(pos):
    return np.clip(np.floor(pos / (TRACK_LENGTH / 5.0)), 0, 4).astype(np.int64)
...
out[1] = bin_position(pos)
```

iii. Step 5 maps `behavior/position` → `output[1]` with "`clip(floor(pos/90),0,4)` (5 x 90 cm bins over 450 cm)", coarsening the paper's 10 cm spatial bins to the resolution the decoder spec requires.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `floor(pos / 90)` clipped to `[0, 4]`, i.e. bin edges at 90/180/270/360 cm with the first and last bins left open so that the handful of samples marginally outside `[0, 450]` land in the end bins. Full-dataset fractions: [0.212, 0.177, 0.231, 0.226, 0.154].

ii.
```python
def bin_position(pos):
    return np.clip(np.floor(pos / (TRACK_LENGTH / 5.0)), 0, 4).astype(np.int64)
```
```python
OUTPUT_VALUES[1] = ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm']
```

iii. Directly implements the instruction "Discretized into 5 equal-sized bins spanning the 450 cm track". Step 3 records the track length from the Methods ("450 cm linear track"), and Step 9 confirms all five bins are occupied and that max raw position is 450.8 cm. Independently re-derived as `pos // 90` in the Step 10 sanity-check script — PASS.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[s:e)` slice as the neural data; no separate alignment step.

ii.
```python
pos = beh['position'][s:e].astype(np.float64)
out[1] = bin_position(pos)
```

iii. As in 7-d. The `--show-processing` figure explicitly plots raw position against `position_bin × 90` for three trials ("they track exactly", Step 7) and marks trial-start/teleport boundaries against the position resets.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior time series, which holds a cumulative lick count per frame.

ii.
```python
beh = {k: b[k]['data'][:] for k in ['position', 'environment', 'lick', ...]}
...
licks = beh['lick'][s:e].astype(np.float64)
```

iii. Step 2: "`lick` (cumulative lick count per frame)". The same stream drives the lick-sensor-error trial filter (1-e).

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized at `> 0`. Trials flagged as lick-sensor errors are removed from the dataset altogether rather than having their lick trace NaN-ed, since the format forbids NaN outputs. Resulting distribution: 77.7% no-lick / 22.3% lick.

ii.
```python
out[3] = (licks > 0).astype(np.int64)
```
```python
lick_bad = np.array([(beh['lick'][s:e] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC
                     for s, e in zip(starts, stops)])
for i, (s, e) in enumerate(zip(starts, stops)):
    if lick_bad[i]:
        continue
```

iii. Step 5 maps this via the repo's `glmUtils` (`licks[licks>1]=1`) and `behavior.correct_lick_sensor_error`; Step 10 Check 3 documents the single deviation: "error trials are *dropped* (a decoder output cannot be NaN)". Independently re-derived as `lick > 0` in the sanity-check script — PASS.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[s:e)` slice; no additional alignment.

ii.
```python
licks = beh['lick'][s:e].astype(np.float64)
out[3] = (licks > 0).astype(np.int64)
```

iii. As in 7-d/8-d; the lick trace is one of the streams plotted against time in `processing_<session>.png` together with the reward markers, and shape equality is asserted per trial.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The scene name in the NWB `identifier` plus the trial index (switch after 30 trials) — see 7-a. The `reward_zone` behavior stream is used only as a consistency check and to gate the reward outcome, not to label the zone.

ii.
```python
zones, envs_scene = zone_per_trial(scene, ntrials_raw)
...
ZONE_TO_IDX = {'A': 0, 'B': 1, 'C': 2}
out[4] = ZONE_TO_IDX[zones[i]]
```

iii. See 7-a. Step 4: "Use scene name + 30-trial switch, exactly as reference code. Validated." Step 9 reports the resulting class balance, "A 33.2%, B 33.6%, C 33.3% of samples", consistent with the paper's counterbalanced design.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Zone letter → index (A→0, B→1, C→2), broadcast over all timepoints of the trial. On the 15 scenes of the form `*_X_to_Y`, the first 30 trials get the first zone and the remainder the second (`min(CHANGE_TRIAL, ntrials)` guards short sessions).

ii.
```python
def zone_per_trial(scene, ntrials):
    z1, z2, e1, e2 = parse_scene(scene)
    if z2 is None:
        return [z1] * ntrials, [e1] * ntrials
    n1 = min(CHANGE_TRIAL, ntrials)
    return ([z1] * n1 + [z2] * (ntrials - n1),
            [e1] * n1 + [e2] * (ntrials - n1))
...
out[4] = ZONE_TO_IDX[zones[i]]
```

iii. Mirrors `behavior.get_reward_zones(..., change_trial=30)`, which the AI documented in Step 1 and quoted the Methods for in Step 3 ("Each switch occurred after 30 trials"). Note the zone label is indexed by the *raw* trial index `i`, so the switch stays at raw trial 30 even when earlier trials were dropped.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward` TimeSeries (expanded onto the frame grid) combined with the `reward_zone` stream — the same `isreward` array used for the previous-trial-outcome input.

ii.
```python
rew_t = b['Reward']['timestamps'][:]
rew_bin = np.zeros_like(beh['position'])
if len(rew_t):
    idx = np.searchsorted(beh['t'], rew_t)
    idx = np.clip(idx, 0, len(rew_bin) - 1)
    rew_bin[idx] = 1
beh['reward'] = rew_bin
```

iii. See 6-a. Step 2: the `Reward` series has its own timestamps because rewards are events, not a per-frame stream, so `searchsorted` maps each reward onto the nearest frame.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per trial: 1 if at least one reward event fell inside `[s, e)` **and** the reward zone was entered during the trial, else 0; broadcast over all timepoints. Full-dataset distribution: 15.8% omitted / 84.2% rewarded (by sample); 84.64% of trials rewarded.

ii.
```python
# reward outcome, as in behavior.get_trial_types: reward delivered AND zone entered
isreward = np.array([(beh['reward'][s:e].sum() > 0) and
                     (beh['reward_zone'][s:e].sum() > 0)
                     for s, e in zip(starts, stops)], dtype=np.int64)
...
out[5] = isreward[i]
```

iii. Step 5 maps this to `behavior.get_trial_types` ("isreward"). Step 9 compares the result to the paper: "Reward rate | ~85% (15% omission) | isreward = reward & rzone entry | 84.66% | 84.64%". Step 12 explains that the low decoding accuracy for this output (0.604) is expected, since "before the animal reaches the reward zone, the rewarded and omission trials are physically identical".

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Behavior/imaging length mismatch** (10 two-plane sessions, always exactly 1 sample): both streams truncated to the common length, with `assert abs(n_beh - n_ophys) <= 5` so a larger mismatch would fail loudly rather than silently.
- **Trial whose teleport falls past the truncation**: dropped with a printed message.
- **Reward timestamps beyond the last behavior frame**: `searchsorted` index clipped into range.
- **Cells whose dF/F has zero variance** (undefined speed correlation): `np.nan_to_num(r_speed, nan=0.0)` so they are kept rather than silently dropped.
- **NaNs outside trials** (a by-product of the per-trial dF/F baseline): `np.nan_to_num` on the activity array, plus `assert np.all(np.isfinite(act)) and np.all(np.isfinite(inp))` on every emitted trial.
- **Sessions reduced to <2 usable trials**: skipped, since the decoder needs at least 2 trials per session.
- **Unrecognised scene name**: `parse_scene` raises rather than guessing.
- **Per-session failures**: caught in the worker, printed with a traceback, and the session omitted rather than aborting the whole run.
- Pre-TTL frames (`pos = -500`, `env = -1`) need no handling — they were verified to always precede the first `trial_start`.

ii.
```python
if n_beh != n_ophys:
    assert abs(n_beh - n_ophys) <= 5, (n_beh, n_ophys)
    print(f'  {os.path.basename(fname)}: behavior has {n_beh} samples, imaging '
          f'{n_ophys}; truncating both to {nframes} (reference one-frame correction)')
```
```python
valid = stops < nframes
if not np.all(valid):
    print(f'  ... dropping {int((~valid).sum())} trial(s) truncated at the end')
    starts, stops = starts[valid], stops[valid]
```
```python
idx = np.clip(idx, 0, len(rew_bin) - 1)
...
r_speed = np.nan_to_num(r_speed, nan=0.0)
...
assert np.all(np.isfinite(act)) and np.all(np.isfinite(inp))
...
if len(neural) < 2:
    print(f'  !! {os.path.basename(fname)}: only {len(neural)} usable trials, skipping')
    return None
```
```python
def _worker(args):
    try:
        return convert_session(fname, show_processing=show, neural_signal=neural_signal)
    except Exception as exc:  # keep going, but report loudly
        import traceback
        print(f'  !! FAILED {fname}: {exc}')
        traceback.print_exc()
        return None
```

iii. Step 10 Check 5 tabulates each edge case and its handling. The length-mismatch fix is tied to the reference: "This is precisely the 'one frame correction ... scan stopping mid frame' case handled in the reference `TwoPUtils.preprocessing.vr_align_to_2P`" (Step 9). `conversion_full_out.txt` shows all 152 sessions converting with no failures.

## 13-a. What are the most time-consuming steps of the code?

i. The script times each stage per session and prints them. Measured (Step 7 table):
1. **OASIS deconvolution** — ~3.5 s/session, the single largest cost (~9 min serial across the dataset).
2. **dF/F computation** — 0.2–1.1 s/session, up to 4.6 s on the 2,341-cell session (~5 min serial).
3. **h5py loading of F/Fneu** — 0.2–0.7 s/session (~1.5 min serial), I/O bound.
4. **Pickling the 9.63 GB result** at the end (single-threaded, unavoidably large).

Total wall clock with 8 workers: **3.4 min** for all 152 sessions.

ii.
```python
t0 = time.time(); timings = {}
...
timings['load'] = time.time() - t0
...
t1 = time.time(); dff = compute_dff(F, Fneu, starts, stops); timings['dff'] = time.time() - t1
...
t1 = time.time(); events = deconvolve(dff_kept, starts, stops); timings['deconv'] = time.time() - t1
...
print(f'  {os.path.basename(fname)}: ... {result["total_time"]:.1f} s '
      f'(load {timings["load"]:.1f}, dff {timings["dff"]:.1f}, '
      f'deconv {timings["deconv"]:.1f})', flush=True)
...
print(f'[{i+1}/{len(files)}] elapsed {el/60:.1f} min, '
      f'projected total {el/(i+1)*len(files)/60:.1f} min', flush=True)
```

iii. Step 6/7: timing instrumentation was added so bottlenecks could be measured rather than guessed, a running projection of total time is printed during the full run, and the 8-worker spawn pool was adopted to bring the estimated ~15 min serial conversion down to ~3 min.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining Python-level loops, all over trials (~80 per session):
- `compute_dff` iterates over trials **three separate times** (mask construction, baseline, dF/F smoothing); these three passes could be fused into one, and the maximin filters could be run once over the whole session with per-trial resets.
- `deconvolve` loops over trials; OASIS itself is the cost, so little to gain.
- Four separate list comprehensions over `zip(starts, stops)` build `env_stream`, `isreward`, `lick_bad`, and `in_trial`; these could be a single pass, or be computed with `np.add.reduceat` over the concatenated trial windows.
- The main per-trial emission loop performs `bin_distance`, `bin_position`, `bin_speed` and the lick binarization per trial; because the zone changes only at trial 30, these could all be computed once on the full session arrays and then sliced.
- `load_fluorescence` loops over planes (at most 2) — negligible.

The one loop the AI *did* vectorize is the interneuron correlation, replaced by a single matrix product.

ii.
```python
dz = d - d.mean(axis=1, keepdims=True)
sz = sp - sp.mean()
denom = (np.sqrt((dz ** 2).sum(axis=1)) * np.sqrt((sz ** 2).sum()))
with np.errstate(invalid='ignore', divide='ignore'):
    r_speed = (dz @ sz) / denom
```
```python
for s, e in zip(starts, stops):
    f_[:, s:e] = F[:, s:e]
    fneu_[:, s:e] = Fneu[:, s:e]
...
for s, e in zip(starts, stops):
    f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
    ...
for s, e in zip(starts, stops):
    dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)
```

iii. Step 6: "Code inefficiencies identified ... the Pearson r(dF/F, speed) loop over cells was replaced by a single matrix product"; "Code speedups added: 8 parallel worker processes (one session each); float32 throughout; per-trial slices are contiguous copies only once." The remaining per-trial loops were left in place because the per-trial windows are inherently variable-length and each already performs a vectorized operation on a (n_cells × T) block, and because parallelism over sessions brought the total to 3.4 min — well inside the 15-minute budget.

## 13-c. What processing does the code repeat multiple times?

i. The conversion is a **single pass** over the data — unlike a survey-then-convert design, each NWB file is opened exactly once and each array read once. What is repeated within a session:
- Three separate loops over trial windows inside `compute_dff` (see 13-b), each re-walking the same boundaries.
- Four further independent passes over `zip(starts, stops)` for `in_trial`, `env_stream`, `isreward`, `lick_bad`.
- `signed_distance_to_zone` is recomputed inside `plot_processing` for the plotted trials, duplicating work already done in the conversion loop.
- `beh['reward_zone']` is scanned once for `isreward` and the `lick` stream twice (once for `lick_bad`, once per trial for the output).

ii.
```python
in_trial = np.zeros(nframes, dtype=bool)
for s, e in zip(starts, stops):
    in_trial[s:e] = True
...
env_stream = np.array([np.median(beh['environment'][s:e]) for s, e in zip(starts, stops)])
isreward   = np.array([... for s, e in zip(starts, stops)], dtype=np.int64)
lick_bad   = np.array([... for s, e in zip(starts, stops)])
```
```python
# in plot_processing
dist_all = np.full(len(beh['position']), np.nan)
for i, (s, e) in enumerate(zip(starts, stops)):
    dist_all[s:e] = signed_distance_to_zone(beh['position'][s:e],
                                            REWARD_ZONES[res['zones'][i]])
```

iii. Not discussed explicitly in CONVERSION_NOTES. Implicitly, the single-pass design follows from the AI's decision to derive the reward zone from the scene name instead of from the data, which removes the need for any cross-session pre-pass; and from Step 6's stated aim to "avoid unnecessary file I/O". The residual repeated trial-boundary walks are each cheap relative to the deconvolution.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
- **The OASIS deconvolution is always run and then thrown away in the default configuration.** `events = deconvolve(dff_kept, starts, stops)` executes on every session, but with the default `--neural-signal dff` the saved `activity` is the dF/F; `events` survives only as `events.shape[0]`, a neuron count already available from `dff_kept`. This is the single most expensive step in the script (~3.5 s of the ~5 s per session, i.e. roughly 70% of the compute).
- Behavior streams `trial number`, `autoreward` and `scanning` are read from disk on every session and never used.
- dF/F is computed for **all** curated cells, including the putative interneurons that are then discarded (unavoidable: the interneuron criterion is defined on dF/F).
- `envs_scene` (the scene-derived environment) is computed per trial and kept in the per-session result but never written into the converted dataset — it served as a validation quantity.
- Per-session bookkeeping (`plane_idx`, `n_roi_iscell`, `kept_trials`, `timings`) is computed for all sessions; most of it ends up only in `metadata['session_info']` or is dropped.

ii.
```python
dff_kept = dff[keep_cells]
events = deconvolve(dff_kept, starts, stops)
timings['deconv'] = time.time() - t1
activity = events if neural_signal == 'events' else np.nan_to_num(dff_kept)
...
n_neurons=int(events.shape[0]), ...
```
```python
beh = {k: b[k]['data'][:] for k in
       ['position', 'environment', 'lick', 'reward_zone', 'speed',
        'teleport', 'trial_start', 'trial number', 'autoreward', 'scanning']}
```

iii. Not flagged in CONVERSION_NOTES as waste. The deconvolution remains in the default path because the script supports both neural streams through one code path and because the AI reported the deconvolved comparison numbers alongside the dF/F ones (Step 12 Check 1); the `autoreward`/`scanning`/`trial number` reads are leftovers of the exploration documented in Step 2 (where the AI established that `autoreward` is all zeros and `scanning` all ones in the release, so neither is usable). The overall runtime (3.4 min) stayed far inside the 15-minute budget, so none of this was revisited.
