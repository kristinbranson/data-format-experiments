# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every `.nwb` file under `/app/data/sub-*/` is globbed (152 files, 11 subjects) and each file is opened with `pynwb.NWBHDF5IO`. One file = one session. Sessions are processed independently and in parallel (`ProcessPoolExecutor`, spawn context, 8–12 workers). From each file the AI reads the behavior streams (`processing['behavior'] → BehavioralTimeSeries`: `position`, `speed`, `lick`, `environment`, `reward_zone`, `scanning`, `trial_start`, `teleport`, `Reward`) and the ophys streams (`processing['ophys']`: `Fluorescence/planeK`, `Neuropil/planeK`, `ImageSegmentation/PlaneSegmentation` with `iscell`/`planeIdx`), plus metadata (`nwb.subject.subject_id`, `nwb.session_id`, `nwb.identifier`, `nwb.imaging_planes['ImagingPlane'].location`). The stored `Deconvolved` series is deliberately *not* used. All 152 sessions survive to the final pickle.

ii.
```python
files = sorted(glob.glob(os.path.join(args.datadir, 'sub-*', '*.nwb')))
...
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
    subject = nwb.subject.subject_id
    session_id = nwb.session_id
    identifier = nwb.identifier
    scene = identifier.split('/')[-1]
    region = nwb.imaging_planes['ImagingPlane'].location

    beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
    g = lambda k: np.asarray(beh[k].data[:])
    t = np.asarray(beh['position'].timestamps[:])
    pos = g('position'); speed = g('speed'); lick = g('lick'); env = g('environment')
    rz_series = g('reward_zone'); scanning = g('scanning')
    starts = np.where(g('trial_start') > 0)[0]
    stops  = np.where(g('teleport') > 0)[0]
    reward_times = np.asarray(beh['Reward'].timestamps[:])

    oph = nwb.processing['ophys'].data_interfaces
    ps = oph['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
    iscell = np.asarray(ps['iscell'].data[:])[:, 0] == 1
    plane_idx = np.asarray(ps['planeIdx'].data[:]).astype(int)
    for plane in sorted(np.unique(plane_idx)):
        Fp = np.asarray(oph['Fluorescence'].roi_response_series[f'plane{plane}'].data[:]).T
        Np = np.asarray(oph['Neuropil'].roi_response_series[f'plane{plane}'].data[:]).T
```

iii. CONVERSION_NOTES Step 2: the dandiset is organised one directory deep as `sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`; the AI counted 152 files across 11 mice and cross-checked this against the paper ("n = 11 mice", 14 imaging days each, m11 imaged only from day 3 → 12), and against 12,216 `trial_start` events vs the paper's 12,376 trials (difference = m11's two unreleased days). It used `pynwb` as required and noted that no dF/F is stored in the files, so the neural signal must be recomputed from `Fluorescence`/`Neuropil`.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the NWB metadata field `nwb.subject.subject_id` (m3, m4, m7, m11 … m19), not from the directory name. The unique set is sorted numerically and `subject_idx` indexes it per session. 11 subjects, with 14 sessions each except m11 (12).

ii.
```python
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
...
'subjects': subjects,
'subject_idx': subject_idx,
```

iii. CONVERSION_NOTES Step 2/Step 9: the subject IDs in the NWB metadata (m3 … m19) map onto the original animal names GCAMP3 … GCAMP19 recovered from `nwb.identifier`; the resulting count (11) and the per-subject session counts (14, except m11 = 12) match the paper exactly ("n = 11 mice"; "imaging started on day 3" for m11).

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Session identity is `nwb.session_id` (experiment day 01–14); the scene, date and original animal name are additionally parsed from `nwb.identifier` and stored in `metadata['session_info']`. Sessions are only excluded if fewer than 2 trials survive curation (no session was excluded).

ii.
```python
session_id = nwb.session_id
animal_orig = identifier.split('/')[-3]
date = identifier.split('/')[-2]
scene = identifier.split('/')[-1]
...
results = [r for r in results if len(r['neural']) >= 2]
...
'session_info': [dict(subject=r['subject'], session_id=r['session_id'], date=r['date'],
                      scene=r['scene'], animal_orig=r['animal_orig'],
                      n_cells=int(r['n_cells']), n_trials=len(r['neural']), ...) for r in results]
```

iii. CONVERSION_NOTES Step 2 and Step 9: file naming (`ses-<NN>`) and `nwb.session_id` agree; 152 sessions = 14 × 10 mice + 12 for m11, matching the paper's design. The ≥2-trial rule is the format requirement from the instructions ("at least two trials within each session"); the AI verified that even the shortest session (39 kept trials) passes.

## 1-d. How are the data split into trials?

i. A trial is the on-track lap: from the frame flagged by the binary `trial_start` behavior series up to (but excluding) the frame flagged by `teleport`. The inter-trial interval / teleport period is excluded. The AI asserts `len(starts) == len(stops)`, that every teleport follows its trial start, and that the last teleport index lies inside the available frames.

ii.
```python
starts = np.where(g('trial_start') > 0)[0]
stops  = np.where(g('teleport') > 0)[0]
...
n_trials_raw = len(starts)
assert len(stops) == n_trials_raw, 'trial_start/teleport count mismatch'
assert np.all(stops > starts), 'teleport before trial_start'
assert stops[-1] <= F.shape[1], 'teleport index beyond available frames'
...
for i, (s, e) in enumerate(zip(starts, stops)):
    ev = events[:, s:e]; p = pos[s:e]; sp = speed[s:e]; lk = lick[s:e]; tt = t[s:e] - t[s]
```

iii. CONVERSION_NOTES Step 5 (decision 3) and Step 10 (Check 3c): the reference `preprocessing.dff` and `glmUtils.get_timeseries_data` slice trials `[trial_start-1 : teleport-1]` in the authors' 1-indexed VR bookkeeping; in the NWB the `trial_start` flag is already on the first on-track frame and the `teleport` frame already carries interpolated ITI position. The AI verified empirically that `pos[start] ≈ 0–5 cm`, `pos[teleport-1] ≈ 445–450 cm`, and `pos[teleport]` is an ITI value, so `[start, teleport)` is the equivalent window. It also verified `n(trial_start) == n(teleport)` in all 152 files (no truncated trials). It rejected the NWB `trial number` stream as the primary trial definition.

## 1-e. How are trials filtered based on quality controls?

i. Five filters, applied per trial:
1. the **first trial of every session is dropped** (152 trials) because `prev_trial_outcome` is undefined for it;
2. **lick-sensor-error trials** are dropped (81 trials): >30 % of frames in the trial have a cumulative lick count > 2 (the paper's criterion; the reference code uses 0.35);
3. trials shorter than `MIN_TRIAL_FRAMES = 10` frames (0 trials);
4. trials containing any non-finite neural/position/speed/lick sample (0 trials);
5. trials containing frames where `scanning != 1` (0 trials).
Result: 11,983 of 12,216 raw trials kept. Sessions with <2 remaining trials would be dropped (none were). No speed threshold is applied (deliberate deviation, see 2-b/8-b).

ii.
```python
LICK_ERROR_FRAC = 0.30           # >30% of frames with cumulative lick count > 2
MIN_TRIAL_FRAMES = 10
...
    lick_error[i] = (lick[s:e] > 2).mean() > LICK_ERROR_FRAC
...
for i, (s, e) in enumerate(zip(starts, stops)):
    if i == 0:
        n_drop_first += 1                      # previous-trial outcome undefined
        continue
    if lick_error[i]:
        n_drop_lick += 1                       # lick-sensor error (paper: NaN'd)
        continue
    if (e - s) < MIN_TRIAL_FRAMES:
        n_drop_short += 1
        continue
    if np.any(scanning[s:e] != 1):
        n_drop_scan += 1                       # no 2P scanning -> invalid fluorescence
        continue
    ...
    if (np.any(~np.isfinite(ev)) or np.any(~np.isfinite(p)) or
            np.any(~np.isfinite(sp)) or np.any(~np.isfinite(lk))):
        n_drop_nan += 1
        continue
```

iii. CONVERSION_NOTES Step 4/Step 5 (decision 5) and Step 10 (Check 3): the paper states "81 out of 12,376 trials removed across 11 switch mice", "detected by >30 % of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2". Using >0.30 (rather than the code's 0.35) reproduced **exactly 81** trials across the 152 sessions, which the AI treats as a strong independent confirmation that its trial segmentation matches the authors'. The trials are *dropped* rather than NaN-masked (as the reference does) because the target format forbids NaNs and lick is a decoder output. The first-trial drop is justified by the `previous trial outcome` input being undefined on trial 0. The remaining filters are defensive and removed nothing.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces: `processing['ophys']['Fluorescence'].roi_response_series['planeK']` (F) and `['Neuropil'].roi_response_series['planeK']` (Fneu), restricted to ROIs with `iscell == 1` from `ImageSegmentation/PlaneSegmentation`, with `planeIdx` used to map segmentation rows to per-plane series and pool planes. The NWB `Deconvolved` series is explicitly rejected.

ii.
```python
ps = oph['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
iscell = np.asarray(ps['iscell'].data[:])[:, 0] == 1
plane_idx = np.asarray(ps['planeIdx'].data[:]).astype(int)
F_list, Fneu_list, keep_list = [], [], []
for plane in sorted(np.unique(plane_idx)):
    name = f'plane{plane}'
    Fp = np.asarray(oph['Fluorescence'].roi_response_series[name].data[:]).T
    Np = np.asarray(oph['Neuropil'].roi_response_series[name].data[:]).T
    keep = iscell[plane_idx == plane]
    F_list.append(Fp[keep]); Fneu_list.append(Np[keep])
    keep_list.append(np.full(int(keep.sum()), plane))
F = np.concatenate(F_list, axis=0).astype(np.float32)
Fneu = np.concatenate(Fneu_list, axis=0).astype(np.float32)
```

iii. CONVERSION_NOTES Step 4 and Step 5 (decision 1): "The stored `Deconvolved` is *not* the paper's signal (it is suite2p's default from raw F, without neuropil correction or the per-trial maximin baseline). I recompute dF/F and OASIS events exactly as in `pp.dff(..., deconvolve=True)`." No dF/F is stored in the NWB, so it must be recomputed from F and Fneu.

## 2-b. How is the `neural` data processed?

i. A faithful re-implementation of `reward_relative.preprocessing.dff(..., neuropil_method='subtract', baseline_method='maximin', neu_coef=0.7, tau=0.7, deconvolve=True)`: mask everything outside trials to NaN → subtract `0.7 × Fneu` → per trial add back `0.7 × mean(Fneu)` → per-trial maximin baseline (Gaussian σ = 15 frames, then a 300-frame `minimum_filter1d` then a 300-frame `maximum_filter1d`, the Methods' ~20 s window) → `dF/F = (F − baseline)/|baseline|` → per-trial Gaussian smoothing σ = 2 frames → per-trial OASIS deconvolution (`suite2p.extraction.dcnv.oasis`, tau = 0.7, fs = 15.5078 Hz) producing "events". Planes are pooled before this step.

**The signal finally stored in `neural` is the dF/F, not the deconvolved events** (`--signal dff` is the default and was used for `converted_data.pkl`; `--signal events` remains available). Teleport samples are never included in the baseline window (`keep_teleports` behaviour is not implemented; the reference default is also `False`, but the authors' `teleport_metadata.py` marks some mouse/day combinations where imaging continued through the ITI).

ii.
```python
def compute_dff_and_events(F, Fneu, starts, stops):
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s:e] = F[:, s:e]; fneu_[:, s:e] = Fneu[:, s:e]
    f_ -= NEU_COEF * fneu_                                   # NEU_COEF = 0.7
    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        seg = gaussian_filter1d(f_[:, s:e], BASELINE_SIGMA, axis=-1)   # 15
        seg = minimum_filter1d(seg, BASELINE_WIN, axis=-1)             # 300
        seg = maximum_filter1d(seg, BASELINE_WIN, axis=-1)             # 300
        flow[:, s:e] = seg
    valid = ~np.isnan(f_[0])
    dff = np.full(F.shape, np.nan, dtype=np.float32)
    dff[:, valid] = (f_[:, valid] - flow[:, valid]) / np.abs(flow[:, valid])
    events = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        dff[:, s:e] = gaussian_filter1d(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=-1)   # 2
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, FRAME_RATE)
    return dff, events
...
dff, events = compute_dff_and_events(F, Fneu, starts, stops)
...
if signal == 'dff':
    events = dff          # optional alternative neural signal (for comparison tests)
```

iii. CONVERSION_NOTES Step 3/Step 5: the pipeline and its constants come from the Methods ("baseline fluorescence was calculated within each trial independently using a maximin procedure with a 20 s sliding window … smoothed with a two-sample (~0.129 s) s.d. Gaussian kernel … deconvolving dF/F with a canonical calcium kernel using the OASIS algorithm") and from the reference code (`neu_coef = 0.7`, `tau = 0.7`). The switch from events to dF/F is documented in Step 12: on a matched 8-session subset dF/F beat the deconvolved events on **every** output (e.g. distance 0.647 vs 0.554, position 0.754 vs 0.665), and the AI argues that "the paper applies OASIS on top of it only 'as a method to eliminate the asymmetric smoothing of the calcium signal', which discards amplitude information that a decoder can exploit", and that the paper itself uses dF/F directly for other quantifications. Full-dataset accuracies rose for every output after the change (distance 0.530 → 0.631, position 0.615 → 0.772).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters (plus one defensive one): (1) only suite2p `iscell == 1` ROIs, pooled across planes; (2) putative interneurons excluded — Pearson r between the cell's dF/F and the animal's running speed > 0.5, computed over all within-trial frames; (3) ROIs with zero variance (non-finite r) are also dropped. 138,678 iscell ROIs → 138,276 neurons (402 interneurons, 0.29 %).

ii.
```python
keep = iscell[plane_idx == plane]          # (1) iscell, per plane, then pooled
...
valid = ~np.isnan(dff[0])                  # (2) interneuron exclusion
sp_v = speed[valid]; dff_v = dff[:, valid]
sp_c = sp_v - sp_v.mean()
dff_c = dff_v - dff_v.mean(axis=1, keepdims=True)
denom = np.sqrt((dff_c ** 2).sum(axis=1) * (sp_c ** 2).sum())
with np.errstate(invalid='ignore', divide='ignore'):
    r_speed = (dff_c @ sp_c) / denom
keep_cells = ~(r_speed > SPEED_CORR_THR)   # SPEED_CORR_THR = 0.5
keep_cells &= np.isfinite(r_speed)         # (3) drop dead ROIs (zero variance)
events = events[keep_cells]
```

iii. CONVERSION_NOTES Step 3/Step 5 (decision 4): `iscell` is the suite2p + manual curation described in the Methods; the interneuron rule is the Methods' "Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed" (paper: 0.42 ± 0.85 % of cells excluded; AI obtained 0.29 %, same order). The AI cross-checked cells/session (154–2,323, mean 910) against the paper's "155–2172 putative pyramidal neurons per session" — the lower bound matches exactly, and the higher upper bound is discussed in Step 9 (the paper's range is over the cells that entered its analyses).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, which requires no extra work: all behavior streams are already interpolated onto the imaging frame clock in the NWB, so slicing the neural matrix with the same `[trial_start, teleport)` frame indices as the behavior gives trial-start alignment. `metadata['temporal_alignment_event'] = 'trial start (entry to the linear track at 0 cm)'`, `off_start = 0.0`, `off_end = None` (variable trial length). No pre-trial window is included.

ii.
```python
ev = events[:, s:e]
...
neural.append(np.ascontiguousarray(ev, dtype=np.float32))
...
'temporal_alignment_event': 'trial start (entry to the linear track at 0 cm)',
'off_start': 0.0,
'off_end': None,
```

iii. CONVERSION_NOTES Step 3/Step 10 (Check 3c): "VR behavior is interpolated onto the 2P frame clock (`vr_align_to_2P`), so all streams share a single time base in the NWB → temporal alignment is inherent". Verified by inspecting positions at the trial-start and teleport frames and by the `--show-processing` plots (position ramps 0 → 450 cm between each blue trial-start and red teleport marker).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging-frame resolution: 15.5078125 Hz → `time_bin_size = 64.4836 ms`. **No rebinning, resampling or down-sampling is applied.** The rate is used as a module constant (also for the OASIS kernel); for the two-plane mice (m17, m18) the stored series `rate` is the 31 Hz volume rate, and the AI uses the per-plane 15.5078 Hz, which it verified equals the behavior timestamp spacing (dt = 0.0644836 s) in every session.

ii.
```python
FRAME_RATE = 15.5078125            # Hz, per imaging plane (from the NWB)
TIME_BIN_MS = 1000.0 / FRAME_RATE  # 64.48 ms
...
'time_bin_size': TIME_BIN_MS,
'sampling_rate_hz': FRAME_RATE,
```

iii. CONVERSION_NOTES Step 5 (decision 2): "The reference analyses (GLM, decoder) operate at the imaging frame rate; no additional temporal binning is applied, maximizing temporal information for the decoder." Step 4 records that the multi-plane `RoiResponseSeries` carry `rate = 31.0` (volume rate) but have exactly as many rows as behavior frames, so the per-plane rate 15.5078 Hz is the correct common clock; the paper says "~15.5 Hz" and "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate".

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` vector of the `position` behavior TimeSeries (all behavior series in the file share one timestamps vector on the imaging frame clock).

ii.
```python
t = np.asarray(beh['position'].timestamps[:])
...
tt = t[s:e] - t[s]
inp[0] = tt
```

iii. CONVERSION_NOTES Step 2/Step 5: all `BehavioralTimeSeries` are sampled on the imaging frame clock with a shared timestamps vector (median dt = 0.06448 s), so any of them can supply the clock; `position` is used. The sanity-check script re-derived `time_from_trial_start` from the raw NWB independently and matched with `np.allclose`.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Only subtraction of the trial's first timestamp, giving a time-varying vector starting at exactly 0 s for every trial. Stored as float32. Range over the dataset: 0 – 216.5 s.

ii.
```python
tt = t[s:e] - t[s]
T = e - s
inp = np.empty((4, T), dtype=np.float32)
inp[0] = tt
```

iii. Straightforward; the AI documents it in the Step 5 mapping table ("t − t[trial_start], seconds, time-varying") and verified in Step 10 Check 2 that `time_from_trial_start == timestamps[s:e] - timestamps[s]` recomputed from the raw file.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. No realignment is needed: the neural matrix and all behavior streams are indexed with the identical `[s:e)` frame range. The only alignment work is a global truncation: in 10 two-plane sessions (m17/m18) the fluorescence has one frame more than the behavior streams, so everything is truncated to the common frame count before trials are cut.

ii.
```python
n_frames = min(F.shape[1], len(t))
n_frames_trunc = (F.shape[1] - n_frames, len(t) - n_frames)
if F.shape[1] != n_frames:
    F = F[:, :n_frames]; Fneu = Fneu[:, :n_frames]
if len(t) != n_frames:
    t = t[:n_frames]; pos = pos[:n_frames]; speed = speed[:n_frames]
    lick = lick[:n_frames]; env = env[:n_frames]
    rz_series = rz_series[:n_frames]; scanning = scanning[:n_frames]
assert stops[-1] <= F.shape[1], 'teleport index beyond available frames'
```

iii. CONVERSION_NOTES Step 9/Step 10 (Check 5): the NWB behavior was exported after `vr_align_to_2P`, so streams are already on the frame clock; the extra frame in the 2-plane sessions was found when the first full run crashed, and "the last teleport index always precedes this boundary, so no trial data is lost (verified by an assertion)".

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavior TimeSeries (−1 during the ITI, 0 = ENV1, 1 = ENV2).

ii.
```python
env = g('environment')
...
env_vals = np.unique(env[s:e])
env_trial[i] = int(env_vals[0]) if len(env_vals) == 1 else int(np.round(np.median(env[s:e])))
```

iii. CONVERSION_NOTES Step 2/Step 4: the `environment` stream corresponds to the reference code's `morph` variable used by `behavior.get_trial_types`; it is −1 only in the ITI (which is outside every trial) and is constant within a trial. 73 sessions are ENV1-only, 68 ENV2-only, and 11 contain both (the environment switch days), always switching at trial index 30, consistent with the paper.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. A single scalar per trial (the unique value inside the trial, or the rounded median as a fallback if more than one value occurs), broadcast across all timepoints of the trial as input row 1. No other transformation; ENV1 → 0, ENV2 → 1. Full-dataset split: 51.0 % / 49.0 %.

ii.
```python
env_vals = np.unique(env[s:e])
env_trial[i] = int(env_vals[0]) if len(env_vals) == 1 else int(np.round(np.median(env[s:e])))
...
inp[1] = env_trial[i]
```

iii. Step 5 mapping table: "unique value within trial: 0 = ENV1, 1 = ENV2 (per trial, broadcast)", mirroring `behavior.get_trial_types`, which takes `np.unique(morph[firstI:lastI])` per trial. The median fallback is a defensive guard against a mixed-value trial; Step 10 Check 2 verified the per-trial environment against an independent recomputation from the raw file.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The within-session index `i` of the loop over the `trial_start`/`teleport` pairs — i.e. the ordinal position of the lap within the session, counted over *all* raw trials (not renumbered after curation). The NWB `trial number` stream is not used.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    inp[2] = i
```

iii. Step 5 mapping table: trial number corresponds to the reference `glmUtils.get_timeseries_data` field `trials` (the trial index). Because the index is taken over raw trials, it stays consistent with the reward-zone switch at trial index 30 and with the dropped trials; the observed range is 1–99 (1 rather than 0 because trial 0 is always dropped).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the integer index across all timepoints of the trial (stored as float32 in the input matrix). No normalisation or rescaling.

ii.
```python
inp = np.empty((4, T), dtype=np.float32)
inp[2] = i
```

iii. Same as 5-a; verified in the Step 10 sanity checks ("`trial_number` == raw trial index", `np.allclose`).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The per-trial reward outcome of trial *i−1*, which is itself derived from the `Reward` TimeSeries (its own sparse timestamps, mapped to frame indices with `np.searchsorted` on the behavior timestamps) **and** the `reward_zone` behavior series — a trial counts as rewarded only if a reward was delivered *and* the animal was inside the reward zone during the trial.

ii.
```python
reward_times = np.asarray(beh['Reward'].timestamps[:])
reward_idx = np.searchsorted(t, reward_times)
rewarded = np.zeros(n_trials_raw, dtype=int)
for i, (s, e) in enumerate(zip(starts, stops)):
    got_reward = np.any((reward_idx >= s) & (reward_idx < e))
    in_zone = np.any(rz_series[s:e] > 0)
    rewarded[i] = int(got_reward and in_zone)     # behavior.get_trial_types
```

iii. Step 5 mapping table: this is exactly the reference `behavior.get_trial_types`, which computes `isreward = (np.any(tmp_reward > 0) and np.any(tmp_rzone > 0))` per trial. The `Reward` series has its own timestamps, hence `searchsorted` onto the frame clock.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Input row 3 of trial *i* is set to the constant `rewarded[i−1]` (0 = omitted, 1 = rewarded), broadcast over all timepoints. The first trial of each session has no predecessor and is **dropped** rather than assigned a default. Dataset mean = 0.844.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    if i == 0:
        n_drop_first += 1                      # previous-trial outcome undefined
        continue
    ...
    inp[3] = rewarded[i - 1]
```

iii. CONVERSION_NOTES Step 5 (decision 5a) and Step 10: "drop the first trial of each session because `previous trial outcome` is undefined for it". Note that `rewarded[i−1]` is used even when trial *i−1* was itself dropped for a lick-sensor error, which preserves the true behavioural history. Step 12 Check 3 explicitly checks for leakage: "`prev_trial_outcome` … is the outcome of trial *i−1*, never of trial *i* — verified in the sanity checks by recomputing it independently from the raw NWB".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` behavior series together with the per-trial reward-zone identity, which is derived from the **scene name** parsed out of `nwb.identifier` (e.g. `Env1_LocationC`, `Env1_LocationA_to_C`, `Env1_A_to_Env2_B`), with the zone switching after 30 trials on switch days — a re-implementation of the reference `behavior.get_reward_zones`. Zone bounds are the paper's: A 80–130, B 200–250, C 320–370 cm. The `reward_zone` behavior stream is used only as an independent validation of that label (and for the reward-outcome definition), because it marks frames only on rewarded trials.

ii.
```python
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
CHANGE_TRIAL = 30                # reference behavior.get_reward_zones(change_trial=30)

def scene_reward_zones(scene, n_trials, change_trial=CHANGE_TRIAL):
    m = re.fullmatch(r'Env\d_Location([ABC])', scene)
    if m:
        return np.array([m.group(1)] * n_trials, dtype='<U1')
    m = re.search(r'([ABC])_to_(?:Env\d_)?(?:Location)?([ABC])$', scene)
    if m:
        labels = np.array([m.group(1)] * n_trials, dtype='<U1')
        labels[change_trial:] = m.group(2)
        return labels
    raise ValueError(f'Unrecognized scene name: {scene}')
...
zone_labels = scene_reward_zones(scene, n_trials_raw)
# sanity check: scene-derived zone matches the recorded reward-zone occupancy
for i, (s, e) in enumerate(zip(starts, stops)):
    inz = np.where(rz_series[s:e] > 0)[0]
    if len(inz):
        zstart = pos[s + inz].min()
        data_lab = min(REWARD_ZONES, key=lambda k: abs(REWARD_ZONES[k][0] - zstart))
        zone_mismatch += int(data_lab != zone_labels[i])
```

iii. CONVERSION_NOTES Step 4/Step 5 (decision 7): the scene name is the authors' own source of zone identity (`get_reward_zones`), and unlike the `reward_zone` stream it is defined on omission trials too. The AI validated it in every session: "Scene-derived labels agree with the data-derived zone on **every** rewarded trial of all 152 sessions (0 mismatches)", and separately confirmed the switch index ("first post-switch *rewarded* trial at index 30 (64 sessions), 31 (11), 32 (2) — later indices occur when trial 30/31 was an omission"). A further behavioural check confirmed that every reward delivery falls inside the assigned zone (±15 cm).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the signed distance in cm from the animal's position to the **nearest point of the reward zone**: negative before the zone (`pos − zone_start`), exactly 0 while inside, positive after (`pos − zone_end`). Time-varying, then discretised (7-c).

ii.
```python
def signed_distance_to_zone(pos, zone_start, zone_end):
    """Signed distance (cm) to the nearest point of the reward zone; 0 inside."""
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d
...
z0, z1 = REWARD_ZONES[zlab]
d = signed_distance_to_zone(p, z0, z1)
out[0] = discretize_distance(d)
```

iii. Step 5 mapping table and Step 10 Check 3f: the paper's reward-relative coordinate is distance to the zone *start*, but "the task asks for distance to *any* location in the reward zone, hence the in-zone = 0 definition" — listed as a deliberate, specification-driven difference from the reference. Linear (not circular) distance is used, also per the task spec.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes via explicit boolean masks: 0 for `d < −50`, 1 for `−50 ≤ d < −10`, 2 for `−10 ≤ d < 0`, 3 for exactly 0 (inside the zone), 4 for `0 < d ≤ 10`, 5 for `10 < d ≤ 50`, 6 for `d > 50`. Resulting distribution: [0.250, 0.102, 0.074, 0.239, 0.021, 0.072, 0.242].

ii.
```python
def discretize_distance(d):
    """7-way discretization of the signed distance to the reward zone."""
    b = np.full(d.shape, 3, dtype=np.int64)       # 3: exactly 0 (inside the zone)
    b[(d < 0) & (d >= -10)] = 2
    b[(d < -10) & (d >= -50)] = 1
    b[d < -50] = 0
    b[(d > 0) & (d <= 10)] = 4
    b[(d > 10) & (d <= 50)] = 5
    b[d > 50] = 6
    return b
```

iii. The bin edges are copied verbatim from the Decoder Task specification; class 3 is defined as exactly 0 so that it means "inside the reward zone". CONVERSION_NOTES Step 7 notes the `--show-processing` plots were used to confirm the bins step exactly at the intended thresholds and that "the reward-zone lines bracket the interval where the distance bin equals 3", and Step 7 explains the over-representation of bin 3 (mice slow down/stop inside the 50 cm zone).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same `[s:e)` frame indices as the neural matrix — no separate alignment. Position and neural data share the imaging frame clock.

ii.
```python
ev = events[:, s:e]
p = pos[s:e]
...
out = np.empty((6, T), dtype=np.int64)
out[0] = discretize_distance(d)
```

iii. As in 2-d/3-c: streams are pre-aligned in the NWB; verified by the frame-count truncation assertion, the trial-boundary position checks, and the per-trial processing plots.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior TimeSeries (cm along the 450 cm virtual track), used raw.

ii.
```python
pos = g('position')
...
p = pos[s:e]
out[1] = np.digitize(p, POS_EDGES)
```

iii. Step 2 notes that `position` is in cm, with a −500 placeholder at the very start of the file and ~−50…0 values during the ITI/teleport jitter zone; since trials exclude the ITI, only on-track values (0–450 cm) enter the conversion.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing the trial and discretising; no smoothing, no speed masking, no recentring. Note the deliberate deviation from the paper's spatial analyses: samples with speed < 2 cm/s are **kept**, because the task requires a `<2 cm/s` speed class and contiguous time series.

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]     # 5 equal bins over the 450 cm track
...
out[1] = np.digitize(p, POS_EDGES)
```

iii. Step 5 (decision 6) and Step 10 Check 3: "No >2 cm/s speed masking: the task requires a `<2 cm/s` speed class and contiguous per-trial time series. (The reference applies it only to spatial/decoding analyses.)" Step 10 Check 2/7 verified the position bins against the raw NWB and investigated non-monotonic position steps (largest backwards step −1.71 cm, VR jitter, faithful to the raw data).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five classes from `np.digitize` with interior edges [90, 180, 270, 360] cm (90 cm bins over the 450 cm track); the first and last bins are open, so the rare samples slightly outside 0–450 cm fall into the end classes. Distribution: [0.212, 0.176, 0.232, 0.227, 0.154].

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
...
out[1] = np.digitize(p, POS_EDGES)
```

iii. Directly from the Decoder Task specification ("5 equal-sized bins spanning the 450 cm track"); `TRACK_LENGTH = 450.0` is recorded as a module constant and the plots in `--show-processing` overlay the bin index (×90 + 45) on the raw position trace to confirm the thresholds.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identical `[s:e)` indexing as the neural data; no extra alignment step.

ii.
```python
p = pos[s:e]
ev = events[:, s:e]
```

iii. Shared imaging frame clock (see 2-d/3-c), confirmed by the trial-boundary position checks and the processing plots.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior TimeSeries, which holds a cumulative lick count per imaging frame (values 0–6).

ii.
```python
lick = g('lick')
...
lk = lick[s:e]
out[3] = (lk > 0).astype(np.int64)
```

iii. Step 2 documents `lick` as "cumulative lick count per imaging frame, 0..6"; the reference `glmUtils.get_timeseries_data` uses the same `sess.timeseries['licks']` stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarised with `lk > 0` (any lick in the frame → 1). In addition, whole trials flagged as lick-sensor errors (>30 % of frames with cumulative count > 2) are removed from the dataset rather than NaN-masked. Distribution: 0.778 / 0.222.

ii.
```python
LICK_ERROR_FRAC = 0.30
...
lick_error[i] = (lick[s:e] > 2).mean() > LICK_ERROR_FRAC
...
if lick_error[i]:
    n_drop_lick += 1                       # lick-sensor error (paper: NaN'd)
    continue
...
out[3] = (lk > 0).astype(np.int64)
```

iii. Step 1/Step 5: the reference binarises cumulative counts (`licks[licks > 1] = 1` after its own handling) and NaNs sensor-error trials; the AI binarises at >0 and drops the error trials because "the target format forbids NaNs and lick is an output". Its >30 % threshold reproduces the paper's exact count of 81 error trials.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[s:e)` frame slice as the neural data; no realignment.

ii.
```python
lk = lick[s:e]
ev = events[:, s:e]
```

iii. Shared frame clock; the `--show-processing` panel overlays the raw cumulative-lick trace and the binary lick output on the same trial time axis to confirm alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The scene name parsed from `nwb.identifier`, with the switch after 30 trials on switch sessions (re-implementation of `behavior.get_reward_zones`), cross-validated against the `reward_zone` + `position` streams. See 7-a.

ii.
```python
scene = identifier.split('/')[-1]
zone_labels = scene_reward_zones(scene, n_trials_raw)
...
ZONE_TO_IDX = {'A': 0, 'B': 1, 'C': 2}
out[4] = ZONE_TO_IDX[zlab]
```

iii. See 7-a: the scene name is the authors' own source, it is defined on omission trials (where the `reward_zone` stream is silent), and the AI reported 0 label mismatches against the data across all 152 sessions, plus a reward-position check (all reward deliveries inside the assigned zone).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The per-trial letter label is mapped A → 0, B → 1, C → 2 and broadcast across all timepoints of the trial (time-varying row, constant within trial). Distribution: 0.331 / 0.336 / 0.332.

ii.
```python
zlab = zone_labels[i]
...
out[4] = ZONE_TO_IDX[zlab]
...
OUTPUT_VALUES[4] = ['A (80-130 cm)', 'B (200-250 cm)', 'C (320-370 cm)']
```

iii. The instruction specifies "Reward zone location, per-trial. 0 = A, 1 = B, 2 = C"; the AI makes it time-varying (constant within the trial) per the format guidance "If at all possible, make it time-varying", and the near-uniform 1/3 distribution is listed as a sanity check in Step 9.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` TimeSeries (sparse, with its own timestamps) combined with the `reward_zone` behavior series, exactly as in the reference `behavior.get_trial_types`.

ii.
```python
reward_times = np.asarray(beh['Reward'].timestamps[:])
reward_idx = np.searchsorted(t, reward_times)
...
got_reward = np.any((reward_idx >= s) & (reward_idx < e))
in_zone = np.any(rz_series[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)     # behavior.get_trial_types
```

iii. Step 4/Step 5: `autoreward` is all zeros in the NWB export and is not needed; reward outcome comes from the `Reward` deliveries restricted to in-zone trials, matching `get_trial_types`. The resulting omission rate (15.7 %) matches the paper's "reward was randomly omitted on ~15 % of trials".

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame indices with `np.searchsorted`; a trial is rewarded (1) if at least one reward index falls in `[s, e)` **and** the `reward_zone` stream is non-zero somewhere in the trial, otherwise omitted (0). The per-trial scalar is broadcast across all timepoints. Distribution: 0.157 omitted / 0.843 rewarded.

ii.
```python
reward_idx = np.searchsorted(t, reward_times)
rewarded = np.zeros(n_trials_raw, dtype=int)
for i, (s, e) in enumerate(zip(starts, stops)):
    got_reward = np.any((reward_idx >= s) & (reward_idx < e))
    in_zone = np.any(rz_series[s:e] > 0)
    rewarded[i] = int(got_reward and in_zone)
...
out[5] = rewarded[i]
```

iii. As above; additionally Step 12 discusses that this output is intrinsically hard to decode early in a trial ("43.0 % of all timepoints occur *before* the animal reaches the reward zone, where whether this trial will be rewarded is not yet determined"), computing a theoretical balanced-accuracy ceiling of 0.785.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Frame-count mismatch** (10 two-plane m17/m18 sessions have one extra imaging frame): all streams truncated to the common length, with an assertion that the final teleport still fits, so no trial data is lost.
- **Lick-sensor errors**: 81 trials dropped (paper's criterion).
- **Pathologically short trials**: <10 frames dropped (0 occurred).
- **Non-finite samples** in neural/position/speed/lick: whole trial dropped (0 occurred).
- **Frames without 2P scanning** (`scanning != 1`): whole trial dropped (0 occurred).
- **Dead/zero-variance ROIs**: excluded via `np.isfinite(r_speed)` so the interneuron correlation cannot yield NaN.
- **Ambiguous per-trial environment**: rounded median fallback.
- **Plot failures** are caught so they cannot break the conversion.
- Sessions ending with <2 usable trials would be dropped (none were).
- Structural assertions: equal numbers of `trial_start` and `teleport` flags, teleport after start, teleport within the frame range.

ii.
```python
n_frames = min(F.shape[1], len(t))
...
assert len(stops) == n_trials_raw, 'trial_start/teleport count mismatch'
assert np.all(stops > starts), 'teleport before trial_start'
assert stops[-1] <= F.shape[1], 'teleport index beyond available frames'
...
keep_cells &= np.isfinite(r_speed)          # drop dead ROIs (zero variance)
...
if np.any(scanning[s:e] != 1):
    n_drop_scan += 1
    continue
if (np.any(~np.isfinite(ev)) or np.any(~np.isfinite(p)) or
        np.any(~np.isfinite(sp)) or np.any(~np.isfinite(lk))):
    n_drop_nan += 1
    continue
...
results = [r for r in results if len(r['neural']) >= 2]
...
except Exception as exc:                    # plotting must never break conversion
    warnings.warn(f'plotting failed for {path}: {exc}')
```

iii. CONVERSION_NOTES Step 9/Step 10 (Check 5) and Step 12. The frame-count mismatch was found when the first full run crashed on `sub-m17_ses-04` and is documented with the fix and why it is lossless. The other guards are described as defensive checks; each drop counter is printed per session in `conversion_full_out.txt`, showing that only the first-trial and lick-error rules actually fire (152 and 81 trials). Edge cases explicitly examined and *kept* include very long trials (max T = 3,359 frames where the mouse paused), negative speeds and small backwards position jitter ("retained as recorded").

## 13-a. What are the most time-consuming steps of the code?

i. The AI instrumented the two dominant steps per session with a `timing` dict: (1) `load_ophys` — bulk HDF5 reads of `Fluorescence`/`Neuropil` for all ROIs (I/O bound, 0.4–1.0 s/session) and (2) `dff_oasis` — the per-trial maximin baseline + Gaussian filters + OASIS deconvolution (1.5–2.5 s/session, up to ~10 s for the largest 2-plane sessions). Pickling the 9.47 GB output is the remaining sizeable cost. Total: 127 s of session processing with 12 workers, 136 s wall clock including the write.

ii.
```python
t_load0 = time.time()
...
timing['load_ophys'] = time.time() - t_load0
...
t0 = time.time()
dff, events = compute_dff_and_events(F, Fneu, starts, stops)
timing['dff_oasis'] = time.time() - t0
...
result['timing']['total'] = time.time() - t_start
...
print(f'Session processing took {elapsed:.1f}s ({elapsed/len(files):.1f}s/session)', flush=True)
```

iii. CONVERSION_NOTES Step 6/Step 7: the AI benchmarked one session before the full run, estimated <5 min wall clock for 152 sessions with 8–12 workers, and listed its speed-ups and their gains (parallel sessions ≈ 8×; float32 + axis-restricted filters ≈ 2× on the dF/F step; a single bulk HDF5 read per plane instead of per-ROI fancy indexing ≈ 5× on loading). It also notes that `fork` cannot be used because suite2p/numba initialise OpenMP at import, hence the `spawn` context.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI states that the remaining per-trial Python loops are inherent to the method (per-trial baselines) and are already vectorized across cells. Loops that remain and could be merged or vectorized further:
- `compute_dff_and_events` iterates over trials **three separate times** (masking, baseline, smoothing+OASIS); these could be fused into a single pass, and the masking loop could be replaced with a boolean frame mask built with `np.add.reduceat`/`np.repeat`.
- `process_session` iterates over trials **three more times** (per-trial task variables, the zone-mismatch sanity check, the trial-building loop).
- The per-trial `got_reward` test is `O(n_trials × n_rewards)`; it could be a single `np.searchsorted` of `reward_idx` into `starts`/`stops`.
- The per-trial discretisations (`np.digitize` on position and speed, distance binning, lick binarisation) could be computed once for the whole session and then sliced.
The interneuron correlation, by contrast, was explicitly vectorized as a matrix product instead of a per-cell `np.corrcoef` loop.

ii.
```python
# three passes over trials inside compute_dff_and_events
for s, e in zip(starts, stops):      # 1. mask to within-trial
for s, e in zip(starts, stops):      # 2. neuropil mean + maximin baseline
for s, e in zip(starts, stops):      # 3. smooth + oasis
# three more passes inside process_session
for i, (s, e) in enumerate(zip(starts, stops)):   # per-trial task variables
for i, (s, e) in enumerate(zip(starts, stops)):   # zone-mismatch sanity check
for i, (s, e) in enumerate(zip(starts, stops)):   # build trials
# vectorized interneuron correlation (no per-cell loop)
r_speed = (dff_c @ sp_c) / denom
```

iii. CONVERSION_NOTES Step 6: "Per-trial python loops for the maximin baseline are unavoidable (per-trial baselines are part of the reference method) but operate on whole (cells × frames) blocks, so they are vectorized across cells." The AI did not document the duplicated per-trial passes, but the per-session cost is small (~2–3 s) and the run finished in 136 s, far inside the 15-minute budget.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and read exactly **once** (a single pass; there is no separate survey pass). What is repeated:
- the six per-trial loops described in 13-b (the trial slicing work is repeated three times inside the dF/F routine and three times in `process_session`);
- `Fluorescence`/`Neuropil` are read for *all* ROIs and then subset to `iscell` — a deliberate trade (bulk contiguous read vs HDF5 fancy indexing);
- the zone-mismatch sanity check re-derives the per-trial reward zone from the data even though the label used is the scene-derived one;
- the pickle is written once; the sample run repeats the whole conversion on 2 sessions.

ii.
```python
Fp = np.asarray(oph['Fluorescence'].roi_response_series[name].data[:]).T   # read all ROIs
keep = iscell[plane_idx == plane]
F_list.append(Fp[keep])                                                    # then subset
...
zone_mismatch = 0                       # second per-trial pass, purely a check
for i, (s, e) in enumerate(zip(starts, stops)):
    inz = np.where(rz_series[s:e] > 0)[0]
    ...
```

iii. CONVERSION_NOTES Step 6: "Reading `Fluorescence`/`Neuropil` for *all* ROIs and then subsetting: HDF5 fancy-indexing per ROI is much slower than a single contiguous read, so the full array is read then masked (0.4–1 s/session)"; "Single pass over the behavior series; per-trial task variables computed with vectorized searchsorted for reward times."

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Undocumented waste, all of it inside the delivered default path:
- **OASIS deconvolution is always computed and then thrown away.** `compute_dff_and_events` unconditionally runs `dcnv.oasis` per trial, but with the default `--signal dff` (which produced `converted_data.pkl`) the result is immediately overwritten by `events = dff`. This is a substantial fraction of the timed `dff_oasis` step for every one of the 152 sessions.
- `dff_keep = dff[keep_cells]` is computed for every session but used only by the optional plotting function.
- The `zone_mismatch` loop, `n_frames_trunc`, `plane_of_cell` and the drop counters are diagnostics that do not enter the saved dataset (cheap, and useful as QC).
- Full-array reads of non-`iscell` ROI traces (see 13-c), and `--show-processing` plotting for 2 sessions.
Nothing unnecessary is *stored*: the pickle contains only the specified fields plus `metadata['session_info']`.

ii.
```python
    events = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        dff[:, s:e] = gaussian_filter1d(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=-1)
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, FRAME_RATE)
    return dff, events
...
dff, events = compute_dff_and_events(F, Fneu, starts, stops)
...
if signal == 'dff':
    events = dff          # <- the deconvolution just computed is discarded
events = events[keep_cells]
dff_keep = dff[keep_cells]          # only used by make_processing_plot
```

iii. CONVERSION_NOTES does not flag any of this; Step 6 instead claims the speed-ups include "removing unnecessary or redundant computations", and Step 12 documents only the *choice* of dF/F over events (keeping `--signal events` available as an option, which is why the code path was left in place). The practical impact is limited — the whole conversion takes 136 s — but the deconvolution could be skipped with a one-line guard when `signal == 'dff'`.
