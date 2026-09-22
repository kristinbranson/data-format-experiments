# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers every NWB file with a single recursive glob over `/app/data` (`sub-*/sub-*.nwb`), sorts them, and processes each file independently as one session. It found and processed all **152** NWB files (11 subject directories; 87 GB). Every file is opened with `pynwb.NWBHDF5IO` (no `h5py`), and from each file it reads: `processing/behavior/BehavioralTimeSeries` (position, speed, lick, reward_zone, environment, trial number, scanning, trial_start, teleport, Reward event timestamps), `processing/ophys/Fluorescence` + `Neuropil` (all planes), `ophys/ImageSegmentation/PlaneSegmentation` (`iscell`, `planeIdx`), the NWB `identifier` (which encodes animal / date / **scene**), `subject.subject_id`, and `imaging_planes['ImagingPlane'].location`. Sessions are converted in parallel with a 12-worker `ProcessPoolExecutor` (spawn context). Trials are then reconstructed inside each session from the `trial_start` / `teleport` behavioral flags (there is no NWB `trials`/`intervals` table).

ii.
```python
DATA_DIR = '/app/data'
...
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))
print(f'found {len(files)} NWB files')
...
ctx = mp.get_context('spawn')
with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
    for res in ex.map(_worker, remaining):
        results.append(res)
```

```python
def read_nwb_session(path):
    """Read everything needed from one NWB file using pynwb."""
    with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
        nwb = io.read()
        beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
        oph = nwb.processing['ophys'].data_interfaces
        d = {}
        d['subject'] = nwb.subject.subject_id
        d['identifier'] = nwb.identifier
        d['scene'] = nwb.identifier.split('/')[-1]
        ...
        d['time'] = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
        d['pos'] = np.asarray(beh['position'].data[:], dtype=np.float64)
        ...
        planes = sorted(oph['Fluorescence'].roi_response_series.keys())
        for pl in planes:
            rrs = oph['Fluorescence'].roi_response_series[pl]
            Fs.append(np.asarray(rrs.data[:], dtype=np.float32))
            Fneus.append(np.asarray(oph['Neuropil'].roi_response_series[pl].data[:], dtype=np.float32))
            roi_idx.append(np.asarray(rrs.rois.data[:], dtype=np.int64))
```

iii. From CONVERSION_NOTES Step 2: the directory is DANDI dandiset 001361, "one directory per subject, one NWB file per session: `sub-<id>/sub-<id>_ses-<NN>_behavior+ophys.nwb` (87 GB total, 152 files)". The AI checked the total against the paper: 11 switch mice × 14 days = 154, minus the two m11 days that were never imaged ("imaging started on day 3 for m11") = **152**, which is exactly what the glob finds. The instructions mandate `pynwb`, so all I/O goes through `NWBHDF5IO`. The AI also notes "No `trials` table / `intervals` → trials must be reconstructed from `trial_start` / `teleport`".

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the NWB metadata field `nwb.subject.subject_id` of each file (not from the directory name). The unique set over all converted sessions is sorted numerically to produce `data['subjects']`, and `data['subject_idx']` indexes that list for each session. Result: `['m3','m4','m7','m11','m12','m13','m14','m15','m17','m18','m19']` — 11 subjects.

ii.
```python
d['subject'] = nwb.subject.subject_id
...
subjects = sorted({inf['subject'] for inf in infos}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(inf['subject']) for inf in infos], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5: "**subjects**: `nwb.subject.subject_id` -> ['m3','m4','m7','m11','m12','m13','m14','m15','m17','m18','m19']". The AI cross-checked this against the paper's "n = 11 mice" switch cohort and against the 11 `sub-*` directories, and notes that the paper's additional 3 "fixed-condition" mice are not in this DANDI set. Reading the id from the file metadata rather than the path keeps subject identity tied to the data itself.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; sessions are kept in sorted filename order, so per subject they run in experiment-day order. The experiment day is parsed out of the filename (`sub-m11_ses-03` → day 3) and is used for the `keep_teleports` lookup. All 152 sessions are kept (no restriction to the paper's 77 "switch" days). A session is dropped only if it ends up with fewer than 2 usable trials (this never happens; the minimum is 40).

ii.
```python
# experiment day = session number in the file name (e.g. sub-m11_ses-03 -> day 3)
exp_day = int(os.path.basename(path).split('_ses-')[1].split('_')[0])
keep_teleports = exp_day in TELEPORT_SESSIONS.get(raw['subject'], [])
```

```python
for (n, i, o, info) in results:
    if len(n) < 2:
        print(f"  SKIPPING {info['file']}: only {len(n)} usable trials")
        continue
    neural.append(n); inputs.append(i); outputs.append(o); infos.append(info)
```

iii. CONVERSION_NOTES Step 5, Key Decision 2: "**All 152 sessions included**, not just the 77 switch sessions. The decoder task's variables are defined on every session, and more sessions give the decoder more data; the paper's restriction to switch days was specific to its reward-relative remapping question." The AI verified 14 sessions for every mouse except m11 (12), and that 77 of the 152 scenes contain `_to_` (switch sessions) — matching the paper's "n = 77 sessions, 11 mice, seven switch days".

## 1-d. How are the data split into trials?

i. Trial boundaries come from the two behavioral flag series: `starts = np.where(trial_start > 0)[0]` and `stops = np.where(teleport > 0)[0]`. The trial window is the half-open frame range **`[start-1, stop-1)`**, i.e. the convention used by the reference repo itself (`preprocessing.dff` and `glmUtils.get_timeseries_data` both slice `[trial_start_inds-1 : teleport_inds-1]`). The ITI/teleport period and the teleport frame itself are excluded. Exactly the same `[s-1, e-1)` slice is used for the neural data, all inputs and all outputs, so every stream is sample-aligned. Assertions check that the number of trial starts equals the number of teleports and that every teleport follows its start; `starts` is clamped with `np.maximum(starts, 1)` so `start-1` can never be negative.

ii.
```python
d['trial_starts'] = np.where(np.asarray(beh['trial_start'].data[:]) > 0)[0]
d['teleports']    = np.where(np.asarray(beh['teleport'].data[:]) > 0)[0]
...
starts = raw['trial_starts']
stops  = raw['teleports']
assert len(starts) == len(stops), 'trial_start / teleport count mismatch'
assert np.all(stops > starts), 'teleport before trial start'
# the reference indexes [start-1:stop-1]; guard against start == 0
starts = np.maximum(starts, 1)
ntrials_all = len(starts)
...
for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s - 1, e - 1)
    T = (e - 1) - (s - 1)
```

iii. CONVERSION_NOTES Step 1: "**Trial windows**: trials run from `trial_start_inds` to `teleport_inds` (with the -1 index convention in glmUtils/dff). ITIs (post-teleport) are excluded from dF/F baseline and from the decoder data." Step 5: "`[trial_start-1, teleport-1)` frames (reference convention; excludes the teleport frame, whose position value is interpolated across the teleport, and excludes the ITI)." Step 10 Check 3 lists alignment as "identical" to the reference. Step 10 Check 5 verified `trial_start` count == `teleport` count in all 152 sessions, that no `trial_start` is at frame 0, and that the pre-scanning position sentinel (-500) never occurs inside a trial window. Total: 12,216 trial_start events across 152 sessions, mean 80.4 trials/session (paper: 80.5 ± 7.4).

## 1-e. How are trials filtered based on quality controls?

i. Four filters:
1. **Lick-sensor-error trials are dropped**: a trial is flagged if more than 30% of its frames have a cumulative lick count > 2 (the rule stated in the paper's Methods; the repo's `glmUtils` uses 35% and `behavior.correct_lick_sensor_error` uses 50%). This flags **exactly 81** trials out of 12,216, reproducing the paper's "n = 81 out of 12,376 trials removed" count.
2. Trials shorter than 2 frames are dropped (never triggered; the shortest surviving trial is 96 frames).
3. Trials whose deconvolved events contain any non-finite value are dropped (defensive guard; never triggered).
4. Sessions left with fewer than 2 usable trials are dropped (never triggered; minimum is 40).

No speed threshold is applied (the paper's <2 cm/s mask is deliberately not applied). 12,135 of 12,216 trials are kept.

ii.
```python
LICK_ERROR_FRAC = 0.30      # >30% of frames with cumulative lick > 2 -> lick sensor error (methods)
LICK_ERROR_COUNT = 2
...
for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s - 1, e - 1)
    ...
    lk = lick[sl]
    lick_error[i] = lk.size > 0 and (np.sum(lk > LICK_ERROR_COUNT) / lk.size) > LICK_ERROR_FRAC
...
for i, (s, e) in enumerate(zip(starts, stops)):
    if lick_error[i]:
        continue                      # lick sensor error (methods): licks unusable
    sl = slice(s - 1, e - 1)
    T = (e - 1) - (s - 1)
    if T < 2:
        continue
    ev_trial = events[:, sl]
    if not np.all(np.isfinite(ev_trial)):
        continue                      # should not happen; guard against bad frames
```

iii. CONVERSION_NOTES Step 5, Key Decision 4: "**Trials with lick-sensor errors are dropped** (the paper NaNs their licks; we cannot store NaN in a categorical output). Rule (from methods): > 30% of frames in the trial with cumulative lick count > 2. This rule reproduces the paper's count of 81 trials exactly." The trajectory (step 94) records the trade-off explicitly: "Dropping 35 trials from one session loses neural data... An alternative... keep the trials but set lick=0... would inject wrong labels. Dropping is cleaner... only 81/12,216 trials (0.66%)". Key Decision 3 justifies *not* applying the paper's 2 cm/s speed mask: "(a) a decoder needs contiguous, equally spaced time bins within a trial and (b) speed itself is a decoder output whose first bin is defined as < 2 cm/s, so removing those samples would delete an entire output class."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are recomputed from the **raw suite2p traces**: `processing/ophys/Fluorescence/plane*` (F) and `processing/ophys/Neuropil/plane*` (Fneu), pooled over planes along the ROI axis. The NWB `Deconvolved` series is explicitly **not** used. ROI-to-`iscell` mapping is done through the `rois` DynamicTableRegion index of each ROIResponseSeries into `ImageSegmentation/PlaneSegmentation`.

ii.
```python
planes = sorted(oph['Fluorescence'].roi_response_series.keys())
Fs, Fneus, roi_idx = [], [], []
for pl in planes:
    rrs = oph['Fluorescence'].roi_response_series[pl]
    Fs.append(np.asarray(rrs.data[:], dtype=np.float32))
    Fneus.append(np.asarray(oph['Neuropil'].roi_response_series[pl].data[:], dtype=np.float32))
    roi_idx.append(np.asarray(rrs.rois.data[:], dtype=np.int64))
d['F']    = np.concatenate(Fs,   axis=1).T      # (n_roi, n_frames)
d['Fneu'] = np.concatenate(Fneus, axis=1).T
roi_idx = np.concatenate(roi_idx)
ps = oph['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
iscell = np.asarray(ps['iscell'].data[:])[:, 0]
d['iscell'] = iscell[roi_idx] == 1
```

iii. CONVERSION_NOTES Step 4 documents a direct empirical test: "NWB `Deconvolved` has no NaNs, is non-zero during the ITI and has raw-F magnitudes (mean ~96); corr with my re-implementation of the paper pipeline = 0.38 ... NWB `Deconvolved` is suite2p's own `spks` on raw F, **not** the paper's `events`. I recompute dF/F + OASIS from the NWB `Fluorescence` and `Neuropil` exactly as in `preprocessing.dff()`." Step 1 notes that the paper's Fig. 3 decoder and its GLM both consume `sess.timeseries['events']`, which `utilities.multi_anim_sess` produces by calling `preprocessing.dff(..., deconvolve=True)` on F/Fneu. Methods "planes were pooled for all analyses" justifies concatenating planes.

## 2-b. How is the `neural` data processed?

i. A faithful port of `reward_relative.preprocessing.dff(..., deconvolve=True)`, run per plane at that plane's own sampling rate:
1. Mask everything outside the per-trial (or, on `keep_teleports` days, trial-plus-ITI) segments to NaN.
2. Neuropil subtraction: `F - 0.7 * Fneu`.
3. Per segment, add back `0.7 × mean(Fneu)` of that segment so the ratio is a true dF/F.
4. Maximin baseline per segment: NaN-aware Gaussian smoothing with sigma `[0, 15]` samples → `minimum_filter1d(300)` → `maximum_filter1d(300)` (the Methods' 20 s window at 15.5 Hz).
5. `dF/F = (F - F0) / |F0|`.
6. 2-sample Gaussian smoothing of dF/F per segment.
7. `suite2p.extraction.dcnv.oasis(dff, 2000, tau=0.7, fs)` per segment → "events".

`fs` is the **per-plane** rate (`scanner rate / n_planes`), so the two-plane m17/m18 sessions use 15.5078 Hz rather than the stored 31.0156 Hz. The `keep_teleports` flag (whether the baseline window may span the teleport) is looked up per animal and experiment day from the reference `teleport_metadata.teleport_sessions` table. Planes are pooled after processing. The final `neural[session][trial]` array is `(n_neurons, n_timepoints)` float32.

ii.
```python
def compute_events(F, Fneu, starts, stops, fs, keep_teleports=False):
    starts, stops = baseline_segments(starts, stops, keep_teleports)
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s - 1:e - 1] = F[:, s - 1:e - 1]
        fneu_[:, s - 1:e - 1] = Fneu[:, s - 1:e - 1]
    nanmask = ~np.isnan(f_[0, :])
    f_ -= NEU_COEF * fneu_                                     # neuropil subtraction
    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        sl = slice(s - 1, e - 1)
        f_[:, sl] = f_[:, sl] + NEU_COEF * np.nanmean(fneu_[:, sl], axis=1, keepdims=True)
        seg = nansmooth(f_[:, sl], [0, BASELINE_SMOOTH])       # sigma 15
        seg = minimum_filter1d(seg, BASELINE_WINDOW, axis=-1)  # 300 samples ~ 20 s
        seg = maximum_filter1d(seg, BASELINE_WINDOW, axis=-1)
        flow[:, sl] = seg
    dff = np.full(F.shape, np.nan, dtype=np.float32)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    events = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        sl = slice(s - 1, e - 1)
        dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH, axis=1)  # sigma 2
        events[:, sl] = dcnv.oasis(np.ascontiguousarray(dff[:, sl]), 2000, TAU, fs)
    return dff, events
```

```python
TELEPORT_SESSIONS = {'m10': [1, 7, 8, 14, 15], ..., 'm19': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17]}

def baseline_segments(starts, stops, keep_teleports):
    if keep_teleports:
        seg_starts = np.append(starts[0], np.asarray(stops[:-1]) + 2)
    else:
        seg_starts = np.asarray(starts)
    return np.maximum(seg_starts, 1), np.asarray(stops)

d['fs'] = d['scan_rate'] / len(planes)     # per-plane sampling rate
```

iii. CONVERSION_NOTES Step 3 quotes the Methods: per-trial maximin baseline with a 20 s sliding window, `dF/F = (F - F0)/|F0|`, 2-sample (~0.129 s) Gaussian smoothing, "activity rate ... extracted by deconvolving dF/F ... using the OASIS algorithm as used in Suite2p". Parameter provenance is documented: `neu_coef = 0.7` and `baseline_method='maximin'` from `preprocessing.dff`, `tau = 0.7` from "the suite2p ops in the repo's example notebook". Step 10 Check 3 tabulates the dF/F, deconvolution, `keep_teleports` and multi-plane steps as "identical" to the reference. The `keep_teleports` table was added as an explicit Step 10 fix: "the first full conversion used `keep_teleports=False` everywhere, but the reference sets it per animal/day from `reward_relative/teleport_metadata.py` ... and all data were re-converted, re-verified, re-sanity-checked and re-trained."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both from the paper:
1. **`iscell`**: only ROIs with `PlaneSegmentation['iscell'][:, 0] == 1` (suite2p detection + the authors' manual curation) are kept. 138,678 of 260,091 ROIs.
2. **Putative-interneuron exclusion**: any remaining cell whose dF/F correlates with running speed at Pearson r > 0.5 over the in-trial samples is dropped (`spatial.is_putative_interneuron`, `r_thresh = 0.5`). 390 cells (0.28%) removed.

The correlation is computed with a vectorised matrix–vector formulation rather than a per-cell loop, and cells with zero variance (NaN r) are treated as non-interneurons. Final total: 138,288 neurons, mean 910/session, range 154–2320.

ii.
```python
F = raw['F'][raw['iscell']]
Fneu = raw['Fneu'][raw['iscell']]
...
nanmask = ~np.isnan(dff[0, :])
sp = speed_all[nanmask]
dm = dff[:, nanmask]
dm_c = dm - dm.mean(axis=1, keepdims=True)
sp_c = sp - sp.mean()
denom = np.sqrt((dm_c ** 2).sum(axis=1) * (sp_c ** 2).sum())
with np.errstate(invalid='ignore', divide='ignore'):
    speed_corr = (dm_c @ sp_c) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INTERNEURON_R_THRESH   # 0.5
keep_cells = ~is_int
events = events[keep_cells]
```

iii. CONVERSION_NOTES Step 3 Curation Steps: "1. Keep only ROIs with `iscell[:,0] == 1` (suite2p + manual curation). 2. Exclude putative interneurons: Pearson r(dF/F, speed) > 0.5 computed over within-trial samples (`spatial.is_putative_interneuron`, `ts_key='dff'`, `r_thresh=0.5`)", quoting the Methods "Additional putative interneurons were detected for exclusion ... by a Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed". Step 9 consistency table: interneurons excluded 0.28–0.35% vs the paper's 0.42 ± 0.85%; neurons/session 154–2320 vs the paper's "155–2172 putative pyramidal neurons per session" (lower bound matches, the upper bound difference is attributed to the paper's per-day cell-tracking subsets).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is the start of the trial. No resampling or shifting is needed: the NWB behavioral series are already on the imaging-frame clock (the NWB file *is* the reference pipeline's `vr_align_to_2P` product), so neural and behavioral samples share indices. Alignment is therefore just the per-trial slice `[trial_start-1, teleport-1)`, applied identically to neural, inputs and outputs. `off_start = 0.0`, `off_end = None` (variable-length trials). Note the window begins one frame *before* the `trial_start` flag, following the reference code's convention — the same one-frame shift is applied to every stream, so there is no neural/behavior misalignment.

ii.
```python
'temporal_alignment_event': 'trial start (teleport into the track at position 0 cm)',
'off_start': 0.0,
'off_end': None,
'trial_window': 'frames [trial_start-1, teleport-1); inter-trial/teleport period excluded',
...
sl = slice(s - 1, e - 1)
ev_trial = events[:, sl]
p  = pos[sl]
sp_t = speed_all[sl]
lk = lick[sl]
t_rel = time_v[sl] - time_v[s - 1]
```

iii. CONVERSION_NOTES Step 5: "**Alignment event**: start of the trial (`trial_start` frame, i.e. entry to the linear track at position 0 cm)". Step 10 Check 3(a): "the NWB `BehavioralTimeSeries` **is** that aligned product (timestamps = imaging frames)". The AI also ran an independent alignment check (`/app/cache/check_alignment.py`): "cross-correlation of mean population activity with running speed peaks at lag -3 to +2 frames (r = 0.37-0.45), i.e. at zero lag within calcium-kinetics resolution -> no temporal offset between neural and behavioural streams."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame period is kept: **64.4836 ms** (15.5078 Hz), identical for every session. **No rebinning, resampling or additional smoothing** is applied beyond the reference's own 2-sample dF/F Gaussian. For the two-plane animals (m17, m18) the stored scanner rate is 31.0156 Hz; the AI divides by `n_planes` so the per-plane bin is also 64.4836 ms, making the bin size uniform across all 152 sessions as the target format requires. `metadata['time_bin_size']` is written as `1000 / fs`.

ii.
```python
d['scan_rate'] = float(oph['Fluorescence'].roi_response_series[planes[0]].rate)
d['n_planes']  = len(planes)
d['fs'] = d['scan_rate'] / len(planes)     # per-plane sampling rate
...
'time_bin_size': float(1000.0 / infos[0]['fs']),   # ms
'sampling_rate_hz': float(infos[0]['fs']),
```

iii. CONVERSION_NOTES Step 5: "**Time bin**: the native imaging frame period, 1/15.5078 Hz = **64.48 ms**, identical for every session (2-plane sessions are sampled at 15.5078 Hz per plane). No re-binning: neural and behavior are already on the same clock in the NWB file, which is exactly the sampling the paper used ('All behavioral and neural time series were sampled at ~15.5 Hz')." Step 9 verifies the NWB median dt = 0.06448 s against the Methods' "~15.5 Hz (0.0645 s)".

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From the NWB behavioral `timestamps`, read off the `position` time series (all `BehavioralTimeSeries` share the same imaging-frame timestamps).

ii.
```python
d['time'] = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
...
time_v = raw['time']
t_rel = time_v[sl] - time_v[s - 1]
inp[0] = t_rel
```

iii. CONVERSION_NOTES Step 2 establishes that all behavioral series are "sampled on the **imaging frame clock** (timestamps present, median dt = 0.06448 s = 1/15.5078 Hz), length = n imaging frames", so any one series' timestamps serve. Step 5 maps input 0 to "`timestamps - timestamps[trial_start-1]`". Step 10 Check 2 verified `time_from_trial_start` against an independent re-derivation from the raw NWB with `np.allclose` (PASS).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Only subtraction of the trial's first timestamp, giving seconds elapsed since the first frame of the trial window. Stored as a float32 time-varying row of the `(4, T)` input array. Observed range over the whole dataset: [0, 216.5] s.

ii.
```python
inp = np.empty((4, T), dtype=np.float32)
t_rel = time_v[sl] - time_v[s - 1]
inp[0] = t_rel
```

iii. Straightforward implementation of the Decoder Task's "Time from start of trial in seconds (continuous, time-varying)". The AI checked the resulting range against the NWB trial durations: "trial durations 6.2-216.6 s | [0, 216.5] s | YES" (Step 9 consistency table), and the processing plots show "time from trial start resets every trial".

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. No alignment operation is needed — the timestamps *are* the imaging frame times, and the same slice `[s-1, e-1)` indexes both neural and timestamps. The only correction is a length harmonisation: in 10 sessions (all two-plane m17/m18 recordings) the behavioral series are exactly one sample shorter than the ophys series, so every stream is truncated to the common length before any slicing.

ii.
```python
nB = len(d['time'])
nF = d['F'].shape[1]
n = min(nB, nF)
d['n_trunc'] = max(nB, nF) - n
if d['n_trunc'] > 0:
    for k in ('time', 'pos', 'speed', 'lick', 'rzone', 'env', 'trialnum', 'scanning'):
        d[k] = d[k][:n]
    d['F'] = d['F'][:, :n]
    d['Fneu'] = d['Fneu'][:, :n]
    d['trial_starts'] = d['trial_starts'][d['trial_starts'] < n]
    d['teleports'] = d['teleports'][d['teleports'] < n]
    d['trial_starts'] = d['trial_starts'][:len(d['teleports'])]
```

iii. CONVERSION_NOTES Step 9: "**Off-by-one between behaviour and ophys length in 10 sessions** (all 2-plane m17/m18 recordings: the behaviour series are exactly 1 sample shorter than the ophys series, the 'one frame correction ... scan stopping mid frame' case the reference alignment code warns about). Fixed by truncating every stream to the common length; no trial is affected because all `trial_start`/`teleport` indices are inside the shorter array."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioral time series (0 = ENV1, 1 = ENV2, −1 outside laps).

ii.
```python
d['env'] = np.asarray(beh['environment'].data[:], dtype=np.float64)
...
env = raw['env']
```

iii. CONVERSION_NOTES Step 2 documents the series as "`environment` | AU | 0 = ENV1, 1 = ENV2, -1 outside laps", which matches the reference code's `get_trial_types` `morph` variable (`env_morph_dict = {'Env1': 0, 'Env2': 1}`) and the Decoder Task's "Environment type (binary, ENV1 vs ENV2, per trial)".

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, the environment value is reduced to a single integer — the rounded **median of the valid (≥ 0) samples** in the trial window — and then broadcast across all timepoints of the trial. Samples with the −1 sentinel (which can occur at the `start-1` frame) are excluded from the median; if a trial had no valid sample the value defaults to 0.

ii.
```python
env_trial = np.zeros(ntrials_all, dtype=np.int64)
for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s - 1, e - 1)
    ...
    ev = env[sl]
    ev = ev[ev >= 0]
    env_trial[i] = int(np.round(np.median(ev))) if ev.size else 0
...
inp[1] = env_trial[i]
```

iii. CONVERSION_NOTES Step 5 lists input 1 as "`environment` series: 0 = ENV1, 1 = ENV2 (mode within the trial)", and Step 10 Check 5 verifies the assumption behind reducing to one value per trial: "Environment changes within a session only on day 8, always exactly at trial 30; per-trial environment is the median of valid (>= 0) samples, so these sessions are labelled correctly." The Decoder Task specifies environment as a per-trial binary variable, which the reduction enforces; the −1 filter makes it robust to the one pre-lap sample included by the `[s-1, ...)` window.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The within-session sequential index `i` of the trial in the `trial_start`-ordered list of all detected trials (0-based). The stored NWB `trial number` series is read but deliberately not used for this. Because the index runs over *all* detected trials, dropped (lick-error) trials leave gaps in the numbering, preserving the true ordinal position of each surviving trial. Observed range [0, 99].

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    inp[2] = i
```

iii. CONVERSION_NOTES Step 5 maps input 2 to "index of the trial within the session (0-based, from `trial_start` order)". This is the same indexing the reference repo uses for trial identity (`glmUtils.get_timeseries_data` writes `trial_ids[start-1:stop-1] = i`), and it matters here because the reward-zone switch and the paper's trial-set definitions are all expressed in terms of that index (switch at trial 30).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond assigning the loop index; the scalar is broadcast across all timepoints of the trial in the `(4, T)` float32 input array.

ii.
```python
inp = np.empty((4, T), dtype=np.float32)
inp[2] = i
```

iii. "All inputs are stored as a (4, T) float32 array so every input is time-varying-compatible" (CONVERSION_NOTES Step 5). Step 10 Check 2 verified `trial_number` against an independent re-derivation from the raw NWB (PASS).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the per-trial rewarded flag `isreward`, which is computed from **two** raw variables following `reward_relative.behavior.get_trial_types`: the `Reward` event series' own `timestamps` (mapped to frame indices with `np.searchsorted` on the behavioral timestamps), **and** the `reward_zone` behavioral series. A trial counts as rewarded only if a reward event fell inside the trial window *and* the reward zone was entered during that trial.

ii.
```python
d['rzone'] = np.asarray(beh['reward_zone'].data[:], dtype=np.float64)
d['reward_times'] = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
...
rew_frames = np.searchsorted(time_v, raw['reward_times'])
rew_frames = rew_frames[rew_frames < len(time_v)]
...
for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s - 1, e - 1)
    has_rew = np.any((rew_frames >= s - 1) & (rew_frames < e - 1))
    in_zone = np.any(rzone[sl] > 0)
    isreward[i] = int(bool(has_rew) and bool(in_zone))
```

iii. CONVERSION_NOTES Step 1 identifies `get_trial_types` as producing "Per-trial `isreward` (reward>0 AND rzone>0 within trial)"; Step 5 maps output 5 / input 3 to that same definition. Step 2 documents `Reward` as "an **event series**: one 0.004 mL sample per delivered reward, with timestamps", i.e. it is not on the frame clock, hence `searchsorted`. Step 4 checks the resulting rate against the paper: "`get_trial_types`: rewarded = reward>0 AND rzone>0 in trial | 84.66% of trials rewarded | '~15%' omission | Consistent (15.3%)."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial `i`, the input is `isreward[i-1]`; for the first trial of a session (`i == 0`) it is set to 0. The index runs over *all* detected trials, so the "previous trial" is the true preceding lap even when that lap was dropped for a lick-sensor error. The scalar is broadcast across the trial's timepoints.

ii.
```python
inp[3] = isreward[i - 1] if i > 0 else 0
```

iii. CONVERSION_NOTES Step 5, Key Decision 6: "**Previous-trial outcome for the first trial of a session = 0**. There is no preceding trial in the imaged session (mice did run ~30 un-imaged warm-up trials beforehand, but those data are not in the file), so the most conservative encoding is 'not rewarded'; this affects 152 of 12,135 trials (1.3%)." This matches the Decoder Task's "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behavioral series combined with the per-trial reward-zone boundaries. The zone identity for each trial is reconstructed the way the reference code does it — from the **scene name** embedded in the NWB `identifier` (e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`) plus the switch-at-trial-30 rule, a port of `reward_relative.behavior.get_reward_zones`. Zone coordinates are the paper's: A = 80–130, B = 200–250, C = 320–370 cm.

ii.
```python
ZONE_DICT = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
CHANGE_TRIAL = 30           # reward zone switches after 30 trials (methods)

def get_reward_zone_labels(scene, ntrials, change_trial=CHANGE_TRIAL):
    """Per-trial reward zone label, replicating reward_relative.behavior.get_reward_zones."""
    if scene.endswith('LocationA'):
        return ['A'] * ntrials
    if scene.endswith('LocationB'):
        return ['B'] * ntrials
    if scene.endswith('LocationC'):
        return ['C'] * ntrials
    for first in ('A', 'B', 'C'):
        if f'{first}_to' in scene:
            second = scene[-1]
            if second not in ZONE_DICT:
                raise ValueError(f'unrecognised switch scene {scene}')
            n0 = min(change_trial, ntrials)
            return [first] * n0 + [second] * (ntrials - n0)
    raise ValueError(f'unrecognised scene {scene}')
```

```python
d['scene'] = nwb.identifier.split('/')[-1]
...
labels = get_reward_zone_labels(raw['scene'], ntrials_all)
...
zlab = labels[i]
z0, z1 = ZONE_DICT[zlab]
rz_cat, _ = discretize_reward_distance(p, z0, z1)
```

iii. CONVERSION_NOTES Step 1 identifies `get_reward_zones` as the reference function: "Per-trial reward zone [start,stop] cm and label from scene name. Zones used in this experiment: A = [80,130], B = [200,250], C = [320,370]; switch days change zone at trial 30". Step 2 notes that the NWB `identifier` preserves the original scene string. Crucially, Step 4 reports an end-to-end validation against the data: "reconstructing zone labels from the scene name in `identifier` (reference logic) matched the zone position observed in the data on **every one of the 12,216 trials** (0 mismatches)." The trajectory (step 39) records the direct observation that confirmed the trial-30 rule: "trial 0-29 = reward zone b (~200-250), trial 30+ = zone a (~80-130), confirming change at trial index 30 (0-indexed), matching change_trial=30 in get_reward_zones."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. A signed linear distance to the **nearest point** of the reward zone: negative before the zone (`pos − zone_start`), exactly 0 anywhere inside `[zone_start, zone_end]`, positive after (`pos − zone_end`). This continuous value is then discretized (see 7-c). The continuous distance itself is not stored (only used for the `--show-processing` plots).

ii.
```python
def discretize_reward_distance(pos, zone_start, zone_end):
    """Signed distance to the nearest point of the reward zone -> 7 categories."""
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start          # negative: before the zone
    d[after] = pos[after] - zone_end              # positive: past the zone
    ...
    return cat, d
```

iii. CONVERSION_NOTES Step 5 outputs table: "signed distance to the **nearest point of the reward zone**: `d = pos - zone_start` if `pos < zone_start`; `d = 0` if inside `[zone_start, zone_end]`; `d = pos - zone_end` if `pos > zone_end`." Step 1 notes the deviation from the reference's own variable: "**Reward-relative position**: `rel_pos` = position minus reward-zone start, either linear (wrapped into [0,450)) or circular ... The Decoder Task here asks for *linear* distance in cm to the nearest point of the reward zone, so we adapt." Step 10 Check 3(f) records this as a "deviation required by task".

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned by explicit boolean masks, matching the Decoder Task's bins: `d < −50` → 0; `−50 ≤ d < −10` → 1; `−10 ≤ d < 0` → 2; `d == 0` → 3 (i.e. inside the zone); `0 < d ≤ 10` → 4; `10 < d ≤ 50` → 5; `d > 50` → 6.

ii.
```python
cat = np.zeros(pos.shape, dtype=np.int64)
cat[d < -50] = 0
cat[(d >= -50) & (d < -10)] = 1
cat[(d >= -10) & (d < 0)] = 2
cat[d == 0] = 3
cat[(d > 0) & (d <= 10)] = 4
cat[(d > 10) & (d <= 50)] = 5
cat[d > 50] = 6
```
```python
OUTPUT_VALUES[0] = ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in reward zone (0 cm)',
                    '0 to +10 cm', '+10 to +50 cm', '> +50 cm']
```

iii. The edges are taken verbatim from the Decoder Task specification; category 3 is defined as exactly 0, which by construction is the set of samples inside the reward zone. CONVERSION_NOTES Step 5 planned the check "`reward_zone_distance == 3` (distance 0) occurs for exactly the samples with position inside the zone", and Step 10 Check 2 reports an independent re-derivation with `np.select` passing. The resulting distribution [0.256, 0.102, 0.073, 0.238, 0.021, 0.072, 0.238] was checked for full class coverage.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No separate alignment: `position` is sliced with the same `[s-1, e-1)` index range as the neural events, so distance categories are sample-for-sample aligned with the neural matrix.

ii.
```python
sl = slice(s - 1, e - 1)
ev_trial = events[:, sl]
p = pos[sl]
...
rz_cat, _ = discretize_reward_distance(p, z0, z1)
out[0] = rz_cat
```

iii. All streams in the NWB share the imaging-frame clock (CONVERSION_NOTES Step 10 Check 3(a)), so identical indexing is sufficient. The `--show-processing` figure panel "position, reward zone (green) and reward deliveries (red)" was used as a visual alignment check: "every delivery falls inside the shaded zone -> alignment of behaviour, reward times and zone identity is correct" (Step 7).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioral time series (VR position in cm along the 450 cm corridor).

ii.
```python
d['pos'] = np.asarray(beh['position'].data[:], dtype=np.float64)
...
p = pos[sl]
```

iii. CONVERSION_NOTES Step 2: "`position` | cm | VR position 0-450; -500 during pre-scan samples". Step 10 Check 5 confirms the sentinel is never inside a trial window: "Position = -500 (pre-scanning) never occurs inside a trial window."

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing the trial window and discretizing; the raw cm values are used directly.

ii.
```python
p = pos[sl]
pos_cat = np.digitize(p, POS_EDGES).astype(np.int64)
out[1] = pos_cat
```

iii. CONVERSION_NOTES Step 5 outputs table: "`pos` into 5 equal 90 cm bins of the 450 cm track". Step 9 verified the position range against the reference data ("position max ~450.8") and the Methods' 450 cm track length.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with the four interior edges `[90, 180, 270, 360]`, giving 5 equal 90 cm bins over the 450 cm track. The first and last bins are open-ended, so the handful of samples marginally outside 0–450 cm (the window's `start-1` frame can sit at ~−5 cm, and positions can reach ~450.8 cm) fall into bins 0 and 4 rather than creating extra classes.

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
...
pos_cat = np.digitize(p, POS_EDGES).astype(np.int64)
```
```python
OUTPUT_VALUES[1] = ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm']
```

iii. The edges come directly from the Decoder Task ("Discretized into 5 equal-sized bins spanning the 450 cm track"); the output value names mirror the specification's open-ended first and last labels. Step 10 Check 2 verified the bins against an independent `np.digitize(pos, [90,180,270,360])` on the raw NWB (PASS). Resulting occupancy [0.217, 0.177, 0.231, 0.226, 0.149] was judged "plausible" for an occupancy-weighted distribution.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[s-1, e-1)` slice as the neural events; no further alignment.

ii.
```python
sl = slice(s - 1, e - 1)
ev_trial = events[:, sl]
p = pos[sl]
out[1] = np.digitize(p, POS_EDGES).astype(np.int64)
```

iii. Behavioral timestamps are the imaging frame times, so identical index ranges guarantee alignment; the independent alignment check (`check_alignment.py`, Step 10) found "no temporal offset between neural and behavioural streams", and 32–42% of cells were strongly position-modulated.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioral time series, which holds a cumulative lick count per imaging frame (observed values 0–6).

ii.
```python
d['lick'] = np.asarray(beh['lick'].data[:], dtype=np.float64)
...
lk = lick[sl]
```

iii. CONVERSION_NOTES Step 2: "`lick` | AU | cumulative lick count per imaging frame (0-6)". The same series drives the lick-sensor-error trial filter (see 1-e).

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation: any frame with a lick count > 0 becomes 1, otherwise 0. Separately, trials flagged as lick-sensor errors (>30% of frames with cumulative count > 2) are removed from the dataset entirely rather than having their licks set to NaN.

ii.
```python
lick_cat = (lk > 0).astype(np.int64)
out[3] = lick_cat
```
```python
LICK_ERROR_FRAC = 0.30
LICK_ERROR_COUNT = 2
lick_error[i] = lk.size > 0 and (np.sum(lk > LICK_ERROR_COUNT) / lk.size) > LICK_ERROR_FRAC
...
if lick_error[i]:
    continue
```

iii. CONVERSION_NOTES Step 5: "`lick` cumulative count per frame > 0 -> 1 (methods: 'Remaining lick counts were converted to a binary vector')" — this is also what the reference `glmUtils` does (`licks[licks > 1] = 1`). The trial-drop decision is Key Decision 4 (see 1-e): the paper NaNs those licks, but a categorical output cannot hold NaN and setting them to 0 "would inject wrong labels". The resulting distribution is 0.777 / 0.223.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[s-1, e-1)` slice as the neural events; no further alignment.

ii.
```python
sl = slice(s - 1, e - 1)
lk = lick[sl]
out[3] = (lk > 0).astype(np.int64)
```

iii. Licks are already resampled onto the imaging frame clock in the NWB file (the reference's `vr_align_to_2P` does "lick ... cumulative-sum-interp then diff", CONVERSION_NOTES Step 1), so identical indexing suffices. Step 10 Check 2 verified `lick > 0` against an independent derivation from the raw NWB (PASS).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the **scene string** in the NWB `identifier` (last path component, e.g. `Env1_LocationB_to_A`, `Env1_B_to_Env2_C`, `Env2_LocationC`), combined with the trial index and the fixed switch-at-trial-30 rule — a port of `reward_relative.behavior.get_reward_zones`. The `reward_zone` behavioral series is not used to *assign* the label, but was used to *validate* it.

ii. See the code in 7-a (`get_reward_zone_labels`, `d['scene'] = nwb.identifier.split('/')[-1]`), plus:
```python
zone_code = {'A': 0, 'B': 1, 'C': 2}
...
out[4] = zone_code[zlab]
```
```python
OUTPUT_VALUES[4] = ['zone A (80-130 cm)', 'zone B (200-250 cm)', 'zone C (320-370 cm)']
```

iii. CONVERSION_NOTES Step 2: the `identifier` "= original path, e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A` -> gives animal (GCAMPnn), date, and **scene** (needed for the reward-zone lookup used by the reference code)". Step 4 records the validation against the data on all 12,216 trials (0 mismatches), and Step 10 Check 2 re-validates "label vs position where `reward_zone > 0` in that trial" (PASS). Step 10 Check 5 adds "No switch session has <= 30 trials, so post-switch trial sets are never empty." The resulting class balance is [0.332, 0.336, 0.333], consistent with the paper's counterbalanced zone assignment.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The per-trial label string A/B/C is mapped to the integer code 0/1/2 and broadcast across all timepoints of the trial. On non-switch sessions all trials get the single scene zone; on switch sessions trials 0–29 get the first zone and trials ≥ 30 the second (`n0 = min(change_trial, ntrials)` guards against sessions shorter than 30 trials). Note the trial index used here is the index over *all* detected trials, so dropping a lick-error trial does not shift the switch point.

ii.
```python
n0 = min(change_trial, ntrials)
return [first] * n0 + [second] * (ntrials - n0)
...
labels = get_reward_zone_labels(raw['scene'], ntrials_all)
zlab = labels[i]
out[4] = zone_code[zlab]
```

iii. CONVERSION_NOTES Step 5: "per-trial, from the scene name in `identifier` + the trial-30 switch rule (`behavior.get_reward_zones`); broadcast in time". Step 3 quotes the Methods: "Each switch occurred after 30 trials." Step 9 consistency table: "Switch trial | after 30 trials | `change_trial=30` | zone change observed at trial 30 | same | YES".

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` event series' `timestamps` **and** the `reward_zone` behavioral series — the same `isreward` used for input 3 (see 6-a), following `reward_relative.behavior.get_trial_types`.

ii.
```python
d['reward_times'] = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
d['rzone'] = np.asarray(beh['reward_zone'].data[:], dtype=np.float64)
...
rew_frames = np.searchsorted(time_v, raw['reward_times'])
rew_frames = rew_frames[rew_frames < len(time_v)]
has_rew = np.any((rew_frames >= s - 1) & (rew_frames < e - 1))
in_zone = np.any(rzone[sl] > 0)
isreward[i] = int(bool(has_rew) and bool(in_zone))
```

iii. CONVERSION_NOTES Step 1 identifies `get_trial_types` as the reference function producing "Per-trial `isreward` (reward>0 AND rzone>0 within trial)". Step 2 documents `Reward` as an event series with its own timestamps rather than a frame-clock series, requiring `searchsorted` to locate its frames.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward event times are converted to frame indices with `np.searchsorted` on the behavioral timestamps (and indices past the end of the truncated timestamp array are discarded). For each trial, `isreward = 1` iff at least one reward frame falls inside `[s-1, e-1)` **and** the reward zone was active at some point in that window; otherwise 0. The value is broadcast across all timepoints of the trial. Resulting rate: 84.6% rewarded / 15.8% omitted (time-weighted).

ii.
```python
rew_frames = np.searchsorted(time_v, raw['reward_times'])
rew_frames = rew_frames[rew_frames < len(time_v)]
...
isreward[i] = int(bool(has_rew) and bool(in_zone))
...
out[5] = isreward[i]
```
```python
OUTPUT_VALUES[5] = ['omitted', 'rewarded']
```

iii. CONVERSION_NOTES Step 5: "per-trial `isreward` = (a reward was delivered inside the trial) AND (reward zone was entered), `behavior.get_trial_types`; broadcast in time". Step 4 validates the rate against the paper's "~15%" omission: "84.66% of trials rewarded ... Consistent (15.3%)". Step 10 Check 5 decomposes the unrewarded trials: "Unrewarded trials = 15.3% true omissions (zone never activated) + 0.8% lapses (zone entered, no reward); both are `reward_outcome = 0`, matching `behavior.get_trial_types`." Step 10 Check 2 verified this output against raw NWB reward timestamps (PASS).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Six cases are handled explicitly:
1. **Behavior/ophys length mismatch** (10 sessions, all two-plane m17/m18, behavior one sample shorter): every stream is truncated to the common length, and trial start/teleport indices beyond the common length are dropped, keeping starts and stops paired.
2. **Trial start at frame 0** would make `start-1` negative: `starts` is clamped with `np.maximum(starts, 1)` (and the same clamp is applied to the baseline segment starts).
3. **Reward events past the end of the timestamp array**: `rew_frames` is filtered to valid indices.
4. **The `environment = −1` sentinel** at out-of-lap samples: excluded before taking the per-trial median, with a fallback of 0 if no valid sample exists.
5. **Non-finite deconvolved events inside a trial**: the trial is skipped (defensive guard; never triggered).
6. **Degenerate trials/sessions**: trials with fewer than 2 frames and sessions with fewer than 2 usable trials are dropped.

Plus the quality-control drop of 81 lick-sensor-error trials (see 1-e). Structural assumptions are asserted rather than assumed: `len(trial_starts) == len(teleports)` and `all(teleports > trial_starts)`.

ii.
```python
nB = len(d['time']); nF = d['F'].shape[1]; n = min(nB, nF)
d['n_trunc'] = max(nB, nF) - n
if d['n_trunc'] > 0:
    for k in ('time', 'pos', 'speed', 'lick', 'rzone', 'env', 'trialnum', 'scanning'):
        d[k] = d[k][:n]
    d['F'] = d['F'][:, :n]; d['Fneu'] = d['Fneu'][:, :n]
    d['trial_starts'] = d['trial_starts'][d['trial_starts'] < n]
    d['teleports'] = d['teleports'][d['teleports'] < n]
    d['trial_starts'] = d['trial_starts'][:len(d['teleports'])]
```
```python
assert len(starts) == len(stops), 'trial_start / teleport count mismatch'
assert np.all(stops > starts), 'teleport before trial start'
starts = np.maximum(starts, 1)
...
rew_frames = rew_frames[rew_frames < len(time_v)]
...
ev = env[sl]; ev = ev[ev >= 0]
env_trial[i] = int(np.round(np.median(ev))) if ev.size else 0
...
if T < 2: continue
if not np.all(np.isfinite(ev_trial)): continue
...
if len(n) < 2:
    print(f"  SKIPPING {info['file']}: only {len(n)} usable trials")
    continue
```

iii. CONVERSION_NOTES Step 9 documents the length-mismatch discovery and fix ("the 'one frame correction ... scan stopping mid frame' case the reference alignment code warns about ... no trial is affected because all `trial_start`/`teleport` indices are inside the shorter array"). Step 10 Check 5 is a dedicated edge-case audit (`/app/cache/check_edges.py`) reporting: no `trial_start` at frame 0; start/teleport counts equal in all 152 sessions; the 10 truncated sessions; no switch session with ≤ 30 trials; environment changes only at trial boundaries; no session with < 2 usable trials (minimum 40); and "Position = -500 (pre-scanning) never occurs inside a trial window."

## 13-a. What are the most time-consuming steps of the code?

i. The script prints per-session timings (`t_load`, `t_dff`, `t_total`). Measured on the full run:
1. **dF/F + OASIS deconvolution** (`compute_events`) — dominant: 0.3 s on the smallest session up to ~20 s on the largest (4174 ROIs); roughly 3–4× the load time on typical sessions.
2. **NWB reading** of the full `Fluorescence`/`Neuropil` arrays — I/O bound, 0.3–3.0 s per session.
3. **Pickling** the 9.63 GB output — 9.7 s.
4. Per-trial assembly and the interneuron correlation are negligible.

With 12 parallel worker processes, all 152 sessions complete in **130 s** wall clock (plus ~10 s to write the pickle), far under the 15-minute target.

ii.
```python
t_start = time.time()
raw = read_nwb_session(path)
t_load = time.time() - t_start
...
t0 = time.time()
dff, events = compute_events(F, Fneu, starts, stops, raw['fs'], keep_teleports=keep_teleports)
t_dff = time.time() - t0
...
print(f"  {info['file']}: ... load {t_load:.1f}s dff {t_dff:.1f}s total {info['t_total']:.1f}s", flush=True)
```
```python
ctx = mp.get_context('spawn')
with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
    for res in ex.map(_worker, remaining):
        results.append(res)
print(f'all sessions processed in {time.time() - t0:.1f}s')
```

iii. CONVERSION_NOTES Step 6: "Loading the full `Fluorescence`/`Neuropil` arrays is the main I/O cost (0.3-1.5 s/session)." Step 7 Run Time Estimates table: "NWB load ... ~3 min serial; dF/F + OASIS ... ~12 min serial; assembly + pickling ... ~2 min; **Total (12 workers) ~2-4 min**". Step 9 confirms the realised time: "152 sessions, 12 workers, **131 s** wall clock, well under the 15 min target."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorised the one loop that mattered: the per-cell Pearson correlation in the interneuron detector was replaced by a single matrix–vector product. The remaining Python loops are all over variable-length trial segments and are inherently hard to vectorise:
- `compute_events` runs **three** separate `for s, e in zip(starts, stops)` passes (masking, baseline, smoothing + OASIS) over ~80 segments per session. These could at least be fused, and the masking pass could be replaced by a single boolean mask built with `np.add.reduceat`/index arithmetic instead of a per-trial slice assignment.
- The per-trial `isreward`/`lick_error`/`env_trial` loop computes three reductions per trial; these could be done for the whole session with `np.add.reduceat` over the trial boundaries.
- The per-trial assembly loop calls `np.digitize` on each trial's position and speed separately; both could be applied once to the whole session array and then sliced.
- `dcnv.oasis` is called per segment — this one genuinely cannot be merged, because the deconvolution must not run across trial boundaries.

ii.
```python
# vectorised (was a per-cell python loop):
dm_c = dm - dm.mean(axis=1, keepdims=True)
sp_c = sp - sp.mean()
denom = np.sqrt((dm_c ** 2).sum(axis=1) * (sp_c ** 2).sum())
speed_corr = (dm_c @ sp_c) / denom
```
```python
# remaining per-segment loops in compute_events:
for s, e in zip(starts, stops):          # pass 1: mask
    f_[:, s - 1:e - 1] = F[:, s - 1:e - 1]
    fneu_[:, s - 1:e - 1] = Fneu[:, s - 1:e - 1]
for s, e in zip(starts, stops):          # pass 2: neuropil mean + maximin baseline
    ...
for s, e in zip(starts, stops):          # pass 3: smoothing + OASIS
    ...
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: ... The interneuron speed correlation was originally a per-cell python loop -> replaced by a vectorised matrix-vector correlation. Code speedups added: Vectorised correlation for interneuron detection; float32 throughout; `dcnv.oasis` called once per trial on a contiguous array; Multiprocessing over sessions (12 workers, each session is independent)." The AI's position is that the remaining per-segment loops are required by the reference algorithm (per-trial baselines and per-trial deconvolution) and are cheap relative to the filtering/OASIS work inside them; Step 7 shows the resulting time was already well within budget.

## 13-c. What processing does the code repeat multiple times?

i. Within `compute_events`, the trial-segment loop is walked **three times** over the same `(starts, stops)` pairs, so the slice arithmetic and the traversal of the full `(n_roi, n_frames)` arrays happen three times instead of once (passes 2 and 3 could be fused). The trial loop is also walked twice in `convert_session` — once to compute `isreward` / `lick_error` / `env_trial` for all trials, and once to assemble the kept trials — because the "previous trial outcome" input needs the outcome of trials that may themselves be dropped. `np.searchsorted(time, reward_times)` is computed once in `convert_session` and again inside `make_processing_plots`, and `discretize_reward_distance` is recomputed in the plotting function for the plotted trials.

Notably, the code does **not** repeat the expensive part: unlike a survey-then-convert design, each NWB file is opened and read exactly once for the whole conversion.

ii.
```python
# three passes over the same segments
for s, e in zip(starts, stops):  ...   # mask
for s, e in zip(starts, stops):  ...   # baseline
for s, e in zip(starts, stops):  ...   # smooth + oasis

# two passes over trials
for i, (s, e) in enumerate(zip(starts, stops)):   # isreward / lick_error / env_trial
    ...
for i, (s, e) in enumerate(zip(starts, stops)):   # assembly
    ...
```
```python
# recomputed inside the plotting helper
rf = np.searchsorted(raw['time'], raw['reward_times'])
...
_, d = discretize_reward_distance(p, z0, z1)
```

iii. The two-pass trial structure is a deliberate consequence of the `previous_trial_outcome` input and of the lick-error filter: the outcome of trial `i-1` must be known even if trial `i-1` is not emitted (CONVERSION_NOTES Step 5, inputs table). The repeated work in `make_processing_plots` only runs under `--show-processing` for at most 2 sessions. The AI's documented efficiency effort (Step 6/7) focused on the dominant costs — the dF/F/OASIS pipeline and per-session parallelism — and the realised 130 s runtime made further deduplication unnecessary.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A small amount, none of it on the critical path:
- `read_nwb_session` loads **all** ROIs' `F` and `Fneu` (260,091 ROIs total) and only afterwards subsets to the 138,678 `iscell` ROIs — roughly half the fluorescence I/O is discarded. (The `iscell` mask is available from `PlaneSegmentation` before the traces are read, so it could be applied during reading.)
- Three behavioral series are read and truncated but never used: `scanning`, `trialnum` (the NWB `trial number`), and `planeIdx`. `dff` is computed for all kept cells and returned, but only its speed correlation is used (the traces themselves are discarded outside `--show-processing`).
- OASIS deconvolution is run on the putative interneurons too; they are dropped immediately afterwards (390 cells, ~0.28% — negligible).
- `discretize_reward_distance` returns the continuous distance `d` alongside the categories; the continuous value is discarded except in the plotting function.
- `info` carries per-session diagnostics (`t_load`, `t_dff`, `t_total`, `frac_rewarded`, `is_switch`, `kept_trials`, `n_trunc`, ...) into `metadata['session_info']`; these are not consumed by the decoder, though they are useful provenance.

ii.
```python
# all ROIs read, then subset
Fs.append(np.asarray(rrs.data[:], dtype=np.float32))
...
F = raw['F'][raw['iscell']]
Fneu = raw['Fneu'][raw['iscell']]
```
```python
# read but never used downstream
d['scanning'] = np.asarray(beh['scanning'].data[:], dtype=np.float64)
d['trialnum'] = np.asarray(beh['trial number'].data[:], dtype=np.float64)
d['planeIdx'] = np.asarray(ps['planeIdx'].data[:])[roi_idx]
```
```python
# deconvolution runs before interneuron exclusion
dff, events = compute_events(F, Fneu, starts, stops, raw['fs'], keep_teleports=keep_teleports)
...
events = events[keep_cells]
```

iii. CONVERSION_NOTES does not flag these explicitly; its efficiency discussion (Step 6) concentrates on the interneuron-correlation vectorisation, float32 arithmetic and multiprocessing, and Step 7/9 show the full conversion at 130 s. The sequencing that causes the residual waste is forced by the algorithm: the interneuron test needs the dF/F, which needs the full `preprocessing.dff` pipeline, so deconvolution of the ~0.28% interneurons cannot easily be avoided without splitting `dff()` in two. Reading all ROIs then subsetting mirrors the reference repo's own loading order. The unused `scanning`/`trialnum` series were retained from the exploration phase (Step 2 used them to verify "Trials overlapping non-scanning samples | 0" and that `trial number` disagrees with `trial_start`).
