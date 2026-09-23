# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every NWB file under `/app/data/sub-*/` is globbed and processed (152 files = 11 subjects x 14 days, m11 has 12). Files are read directly with `h5py` rather than `pynwb` (the NWB file *is* an HDF5 file, and reading it directly lets the loader slice only the `iscell` ROI columns out of the fluorescence matrices). From each file the agent reads: all 11 behavioural time series of `processing/behavior/BehavioralTimeSeries`, the shared frame timestamps, the sparse `Reward` event timestamps, the `trial_start`/`teleport` flag frames, the subject id, session id, imaging rate, imaging-plane location, and the *scene name* parsed out of the NWB `identifier` string (e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`). From `processing/ophys` it reads raw `Fluorescence` and `Neuropil` per plane, plus the `PlaneSegmentation/iscell` and `planeIdx` tables. Sessions are processed in parallel (`multiprocessing`, 8-12 workers, `maxtasksperchild=1`).

ii.
```python
files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
...
def load_session(fn):
    """Read everything needed from one NWB file (h5py, no pynwb needed)."""
    with h5py.File(fn, 'r') as f:
        beh = f['processing/behavior/BehavioralTimeSeries']
        g = lambda k: beh[k]['data'][()].astype(np.float64)
        d = dict(
            file=fn,
            subject=f['general/subject/subject_id'][()].decode(),
            session_id=f['general/session_id'][()].decode(),
            scene=f['identifier'][()].decode().split('/')[-1],
            imaging_rate=float(f['general/optophysiology/ImagingPlane/imaging_rate'][()]),
            location=f['general/optophysiology/ImagingPlane/location'][()].decode(),
            pos=g('position'), speed=g('speed'), lick=g('lick'),
            rzone_flag=g('reward_zone'), env=g('environment'),
            trialnum=g('trial number'), autoreward=g('autoreward'),
            scanning=g('scanning'),
            t=beh['position']['timestamps'][()].astype(np.float64),
            reward_t=beh['Reward']['timestamps'][()].astype(np.float64),
            starts=np.where(g('trial_start') > 0)[0],
            stops=np.where(g('teleport') > 0)[0],
        )
        ps = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = ps['iscell'][()][:, 0].astype(bool)
        planes = sorted(int(p) for p in np.unique(ps['planeIdx'][()]))
        Fl, Nl = [], []
        for p in planes:
            rois = f['processing/ophys/Fluorescence/plane%d/rois' % p][()]
            keep = iscell[rois]
            Fl.append(f['processing/ophys/Fluorescence/plane%d/data' % p][:, keep].T)
            Nl.append(f['processing/ophys/Neuropil/plane%d/data' % p][:, keep].T)
        d['F'] = np.concatenate(Fl, axis=0).astype(np.float64)
        d['Fneu'] = np.concatenate(Nl, axis=0).astype(np.float64)
```

iii. The agent verified that this enumerates all the data: 152 sessions, 11 subjects, 12,216 complete laps, 138,678 `iscell` ROIs, matching the paper's 11 switch mice, "14 days/mouse with imaging for m11 starting on day 3", 80.5 +/- 7.4 trials/session (measured 80.37 +/- 6.16), and the 155-2172 cells/session range (measured 155-2341, the upper end being the two pooled-plane mice). `h5py` was chosen over `pynwb` so that only the curated ROI columns are pulled off disk, and per-session parallelism was used to keep the full run under 4 minutes.

## 1-b. How are the data split into subjects (mice)?

i. One subject per NWB `general/subject/subject_id` field. The unique subject ids of the converted sessions are sorted numerically to form `data['subjects']`, and `subject_idx` indexes into that list per session.

ii.
```python
subject=f['general/subject/subject_id'][()].decode(),
...
subjects = sorted({s['subject'] for s in sessions}, key=lambda x: int(x[1:]))
...
subject_idx=np.array([subjects.index(s['subject']) for s in sessions], dtype=np.int64),
```

iii. The subject id is stored in the file itself and is redundant with the `sub-<id>` directory/file name. The agent recorded that this gives the 11 switch mice of the paper (m3, m4, m7, m11-m15, m17-m19); verification output confirms 11 subjects with 14 sessions each except m11 with 12.

## 1-c. How are the data split into sessions?

i. One session per NWB file. The session identifier is `general/session_id`, which the agent determined is the experiment day (1-14), consistent with the `ses-NN` in the file name. No cross-session neuron alignment is attempted; each session contributes its own neuron set.

ii.
```python
session_id=f['general/session_id'][()].decode(),
...
sessions = [r[0] for r in results if r[0] is not None and len(r[0]['neural']) >= 2]
...
session_info=[dict(subject=s['subject'], session=s['session_id'], scene=s['scene'],
                   date=s['date'], n_neurons=s['n_neurons'],
                   n_trials=len(s['neural']),
                   laps=s['kept_trials'].tolist()) for s in sessions],
```

iii. Notes: "`ses-NN` == experiment day (1-14) ... m11 has only days 3-14 (12 sessions; no imaging on days 1-2, as stated in Methods); all others have 14", which reproduces the paper's session count exactly. A session is only emitted if it has >= 2 usable trials (the decoder's requirement); no session was actually lost this way.

## 1-d. How are the data split into trials?

i. A trial ("lap") is the pair of a `trial_start` flag frame and the following `teleport` flag frame. Both flags are single-frame binary events, so the frame indices are taken with `np.where(... > 0)`. The sample window used for the trial is `[trial_start-1, teleport-1)`, which is the window the reference code uses in both `preprocessing.dff` and `glmUtils.get_timeseries_data`. The same index range is used for the neural, input and output arrays.

ii.
```python
starts=np.where(g('trial_start') > 0)[0],
stops=np.where(g('teleport') > 0)[0],
...
starts, stops = S['starts'], S['stops']
n_trials = len(starts)
assert n_trials == len(stops) and np.all(starts < stops) and starts[0] >= 1
...
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    sl = slice(s - 1, e - 1)
    T = (e - 1) - (s - 1)
    ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]
```

iii. From the notes: "the reference code (`preprocessing.dff`, `glmUtils.get_timeseries_data`) uses `[trial_start-1, teleport-1)`, i.e. from one frame before the trial-start flag up to one frame before teleport". The agent verified on all 152 sessions that the number of `trial_start` flags equals the number of `teleport` flags, that starts always precede their teleport, that no `trial_start` occurs at frame 0 (so `start-1` is always valid), and that no window contains pre-synchronisation samples (`trial number` = -1, position = -500), non-scanning samples, or teleport-period positions (< -20 cm). The resulting 12,216 complete laps are 1.3% fewer than the paper's 12,376 imaged trials, which the agent attributes to incomplete laps dropped by the DANDI conversion.

## 1-e. How are trials filtered based on quality controls?

i. Four filters:
1. **Lick-sensor error laps** — a lap is dropped if more than 30% of its frames have a cumulative lick count > 2 (the Methods' rule, implemented in the reference as `behavior.correct_lick_sensor_error`, which sets those lick values to NaN). Because lick is a required decoder output, the agent drops the lap entirely rather than NaN-ing it. This flags **exactly 81 laps**, the number quoted in the paper.
2. **First lap of every session** (152 laps) — dropped because "previous trial outcome" is a required decoder input and is undefined for the first imaged lap.
3. **Laps with any non-finite deconvolved event** are dropped (none occurred).
4. **Sessions with fewer than 2 kept trials** are dropped (none occurred).
No minimum trial-length filter is applied.

ii.
```python
LICK_ERROR_THRESH = 0.30    # Methods: >30% of frames with cumulative lick count > 2
...
for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s - 1, e - 1)
    ...
    L = S['lick'][sl]
    lick_error[i] = (np.sum(L > 2) / len(L)) > LICK_ERROR_THRESH
...
for i, (s, e) in enumerate(zip(starts, stops)):
    if i == 0:
        continue                      # previous-trial outcome undefined
    if lick_error[i]:
        continue                      # invalid lick data (reference sets to NaN)
    ...
    if not np.all(np.isfinite(ev)):
        print('  WARNING %s trial %d: non-finite events, trial dropped' % ...)
        continue
...
sessions = [r[0] for r in results if r[0] is not None and len(r[0]['neural']) >= 2]
```

iii. Notes Step 5 decision 6: "(a) the 81 lick-sensor-error trials (>30% of frames with cumulative lick count > 2), because lick is a required output and the reference treats these lick values as invalid (NaN); (b) the first lap of each session (152 trials), because previous-trial outcome is a required input and is undefined for it (the preceding warm-up laps are not in the imaging record). Remaining trials: ~11,980 of 12,216 (98.1%)." The trajectory shows the first-lap decision was weighed explicitly against setting the previous outcome to 0: "Dropping only 152 of 12216 trials (1.2%) ... avoids inventing data". The threshold choice was also validated against the paper: 81 trials at >0.30 (paper's number), 69 at >0.35 and 43 at >0.5 (the thresholds hard-coded in the reference code), so the paper's 0.30 was used.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces `processing/ophys/Fluorescence/plane<i>/data` (F) and `processing/ophys/Neuropil/plane<i>/data` (Fneu). The stored `processing/ophys/Deconvolved` series is explicitly **not** used.

ii.
```python
Fl.append(f['processing/ophys/Fluorescence/plane%d/data' % p][:, keep].T)
Nl.append(f['processing/ophys/Neuropil/plane%d/data' % p][:, keep].T)
d['F'] = np.concatenate(Fl, axis=0).astype(np.float64)
d['Fneu'] = np.concatenate(Nl, axis=0).astype(np.float64)
```

iii. Notes Step 4: "`glmUtils.get_timeseries_data` uses `sess.timeseries['events']`, produced by `preprocessing.dff(..., deconvolve=True)` from raw F/Fneu ... NWB stores a `Deconvolved` series whose values are large (0-13,770) and have no NaNs, i.e. suite2p deconvolution of **raw F**, not of the paper's dF/F. Resolution: recompute dF/F + OASIS events from NWB F and Fneu exactly as `preprocessing.dff` does. The stored `Deconvolved` series is not used."

## 2-b. How is the `neural` data processed?

i. A direct port of the reference `reward_relative.preprocessing.dff` with the reference settings (`neuropil_method='subtract'`, `baseline_method='maximin'`, `subtract_baseline=True`, `neu_coef=0.7`, `tau=0.7`, `deconvolve=True`, `keep_teleports=False`): samples outside the lap windows are set to NaN; `0.7 * Fneu` is subtracted globally; within each lap the per-lap neuropil mean is added back; the maximin baseline is a Gaussian smooth (sigma = 15 frames) followed by a 300-frame (~20 s) minimum filter and a 300-frame maximum filter; dF/F = (F - baseline)/|baseline|; dF/F is smoothed with a 2-frame Gaussian; then OASIS deconvolution (`suite2p.extraction.dcnv.oasis`, batch 2000, tau 0.7, fs = `imaging_rate / n_planes`) gives the "events" that are stored as `neural`. Cells from both planes are pooled for the two-plane mice (m17, m18). Unlike the reference notebook, the per-animal/per-day `keep_teleports` table is not used — the baseline window is always restricted to the lap.

ii.
```python
NEU_COEF = 0.7; TAU = 0.7; BASELINE_SMOOTH_SIGMA = 15
MAXIMIN_WINDOW = 300; DFF_SMOOTH_SIGMA = 2; OASIS_BATCH = 2000

def compute_dff_and_events(F, Fneu, starts, stops, fs):
    f_ = np.full(F.shape, np.nan); fneu_ = np.full(F.shape, np.nan)
    for start, stop in zip(starts, stops):
        f_[:, start - 1:stop - 1] = F[:, start - 1:stop - 1]
        fneu_[:, start - 1:stop - 1] = Fneu[:, start - 1:stop - 1]
    nanmask = ~np.isnan(f_[0, :])
    f_ -= NEU_COEF * fneu_
    flow = np.full(F.shape, np.nan); events = np.full(F.shape, np.nan)
    for start, stop in zip(starts, stops):
        sl = slice(start - 1, stop - 1)
        f_[:, sl] = f_[:, sl] + NEU_COEF * np.nanmean(fneu_[:, sl], axis=1, keepdims=True)
        base = gaussian_filter1d(f_[:, sl], BASELINE_SMOOTH_SIGMA, axis=-1)
        base = minimum_filter1d(base, MAXIMIN_WINDOW, axis=-1)
        base = maximum_filter1d(base, MAXIMIN_WINDOW, axis=-1)
        flow[:, sl] = base
    dff = np.full(F.shape, np.nan)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    ...
    for start, stop in zip(starts, stops):
        sl = slice(start - 1, stop - 1)
        dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH_SIGMA, axis=1)
        events[:, sl] = dcnv.oasis(
            np.ascontiguousarray(dff[:, sl], dtype=np.float32), OASIS_BATCH, TAU, fs)
    return dff, events, bad_baseline
...
fs = S['imaging_rate'] / S['n_planes']
dff, events, bad_baseline = compute_dff_and_events(S['F'], S['Fneu'], starts, stops, fs)
```

iii. Notes Step 3/Step 1: the parameters are taken from the reference (`utilities.default_dff_method`: `neu_coef = 0.7`, `baseline_method = 'maximin'`, `keep_teleports = False`; `tau = 0.7` from the suite2p ops in the repo notebook), and the pipeline matches the Methods text ("baseline fluorescence was calculated within each trial independently using a maximin procedure with a 20 s sliding window ... smoothed with a two-sample (~0.129 s) s.d. Gaussian kernel ... deconvolving dF/F with a canonical calcium kernel using the OASIS algorithm as used in Suite2p"). The agent noted that the NWB `imaging_rate` is the scanner rate (31.015625 Hz on the two-plane mice) and that the deconvolution kernel must use the per-plane rate, hence `fs = imaging_rate / n_planes` = 15.5078 Hz for every session. Correctness was checked in Step 10 by an independent re-implementation that reproduced the stored event matrices with `np.allclose` (atol 1e-5) on three sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three rules. (1) Only ROIs with suite2p manual-curation flag `iscell[:,0] == 1` are loaded at all. (2) Putative interneurons are excluded: any cell whose dF/F correlates with running speed at Pearson r > 0.5 over the within-lap samples (409 cells, 0.29%). (3) An extra rule not in the reference: cells whose minimum within-lap maximin baseline falls below 5% of their median raw fluorescence are dropped, because for those cells dF/F divides by a near-zero baseline and explodes (25 cells, 0.018%).

ii.
```python
iscell = ps['iscell'][()][:, 0].astype(bool)
...
keep = iscell[rois]
Fl.append(f['processing/ophys/Fluorescence/plane%d/data' % p][:, keep].T)
...
# ---- cell curation: putative interneurons (dF/F vs speed r > 0.5) ----
valid = ~np.isnan(dff[0, :])
sp = S['speed'][valid]; X = dff[:, valid]
Xc = X - X.mean(axis=1, keepdims=True); spc = sp - sp.mean()
denom = np.sqrt((Xc ** 2).sum(axis=1) * (spc ** 2).sum())
speed_corr = (Xc @ spc) / np.where(denom == 0, np.nan, denom)
is_int = np.nan_to_num(speed_corr, nan=0.0) > INTERNEURON_R_THRESH   # 0.5
keep_cells = (~is_int) & (~bad_baseline)
```
```python
BASELINE_MIN_FRAC = 0.05
...
with np.errstate(invalid='ignore'):
    min_base = np.nanmin(flow, axis=1)
    scale = np.nanmedian(np.where(np.isnan(flow), np.nan, F), axis=1)
bad_baseline = ~(min_base > BASELINE_MIN_FRAC * scale)
```

iii. Rules 1 and 2 are the paper's ("suite2p manual curation"; "Additional putative interneurons were detected for exclusion ... by a Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed, excluding 0.42 +/- 0.85% of cells") — the agent's measured 0.29% is in that range. Rule 3 is documented as a deliberate addition (Step 9): "in 8 sessions (again only m17/m18) dF/F reached |dF/F| ~ 3000 for a handful of cells ... the neuropil trace is much larger than the ROI trace, so after the reference neuropil subtraction the corrected trace crosses zero and the per-trial maximin baseline collapses to ~0 ... Because the decoder PCA would be dominated by such numerically degenerate cells, cells whose minimum within-trial baseline is below 5% of their median raw fluorescence are now excluded (25 cells = 0.018%)." After the fix the dF/F range fell from [-931, 3013] to [-16.0, 37.2].

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is trial start. No resampling or shifting is needed: the behavioural and neural streams are already on the same imaging-frame clock, so aligning to trial start is just slicing at the trial-start frame index. The agent uses the reference window `[trial_start-1, teleport-1)`, so the emitted trial begins one frame (-64.5 ms) *before* the trial-start flag; this is declared in the metadata as `off_start = -0.0645 s`, with `off_end = None` because laps have variable length.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]
...
tt = S['t'][sl] - S['t'][s]       # time from trial start (s)
...
temporal_alignment_event='trial start (teleport into the linear track, position 0 cm)',
off_start=-float(np.mean([s['dt'] for s in sessions])),
off_end=None,
```

iii. Notes Step 5 decision 3: "Trial window `[trial_start-1, teleport-1)` exactly as in `preprocessing.dff` and `glmUtils.get_timeseries_data`; alignment event = trial start (entry to the linear track), off_start = -0.0645 s (the window begins one frame before the trial-start flag), off_end = None (variable trial length)." Alignment was checked visually in the `--show-processing` figures (neural raster, position, speed and licks on one time axis with trial-start/teleport lines) and numerically, since inputs, outputs and neural data are all sliced with the identical frame range.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame per time bin, 64.4836 ms (15.5078 Hz). **No rebinning or resampling of any kind** is applied. `time_bin_size` in the metadata is the mean over sessions of the median inter-frame interval of the behavioural timestamps, in ms.

ii.
```python
dt = float(np.median(np.diff(S['t'])))
...
time_bin_size=float(np.mean([s['dt'] for s in sessions]) * 1000.0),
```

iii. Notes Step 5 decision 2: "**Time bin = one imaging frame (64.4836 ms)**, no re-binning: the reference pipeline processes all neural and behavioural data at the imaging frame rate (~15.5 Hz) and the NWB behavioural series are already interpolated onto those frame times, so this preserves exact temporal alignment with no resampling." This matches the Methods ("All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate") and the paper's stated 0.0645 s sample period. Two-plane sessions are still 15.5 Hz per plane because `imaging_rate` there is the 31 Hz scanner rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The behavioural frame timestamps, read from `processing/behavior/BehavioralTimeSeries/position/timestamps`. All behavioural series in the file share these timestamps (verified during exploration), so any of them would do.

ii.
```python
t=beh['position']['timestamps'][()].astype(np.float64),
...
tt = S['t'][sl] - S['t'][s]       # time from trial start (s)
x[0] = tt
```

iii. Notes Step 2: the behavioural series are "all 1-D, one sample per imaging frame, shared timestamps, dt = 0.064484 s", and Step 10's independent sanity check confirmed `time_from_trial_start == timestamps[start-1:stop-1] - timestamps[start]` for the first, middle and last trial of three sessions.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtraction of the timestamp of the trial-start flag frame from the trial's timestamp vector. Because the emitted window starts one frame earlier (`[start-1, stop-1)`), the first value of this input is -0.0645 s and it crosses zero exactly at the trial-start flag; the metadata `off_start` records this.

ii.
```python
tt = S['t'][sl] - S['t'][s]       # time from trial start (s)
x = np.empty((4, T), dtype=np.float32)
x[0] = tt
```

iii. The reference origin is the trial-start flag itself, not the first sample of the window, so that the input means literally "time since the animal entered the track". The resulting range across the full dataset is [-0.064, 216.5] s, the upper bound being the longest lap in the data.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is taken from the identical frame indices as the neural data, so alignment is automatic. In 10 sessions (m17, m18) the ophys arrays have exactly one frame more than the behavioural arrays; there the agent truncates both streams to the common length before anything else, having checked that both streams start at t = 0 so the extra imaging frame is at the end.

ii.
```python
n_beh = d['pos'].shape[0]
if d['F'].shape[1] != n_beh:
    print('  NOTE %s: ophys has %d frames, behaviour %d; truncating to %d' % ...)
    n = min(d['F'].shape[1], n_beh)
    d['F'] = d['F'][:, :n]; d['Fneu'] = d['Fneu'][:, :n]
    for k in ['pos', 'speed', 'lick', 'rzone_flag', 'env', 'trialnum',
              'autoreward', 'scanning', 't']:
        d[k] = d[k][:n]
```

iii. Notes Step 3: "VR behaviour is already interpolated onto imaging frame times (`vr_align_to_2P` in the authors' TwoPUtils repo); in the NWB files every behavioural series shares the imaging-frame timestamps", so no interpolation is needed. The frame-count mismatch was found when the first full run crashed and is documented as a resolved issue in Step 9/Step 12.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioural time series (0 = ENV1, 1 = ENV2, -1 before VR sync).

ii.
```python
env=g('environment'),
...
env_trial[i] = int(np.round(np.median(S['env'][sl])))
x[1] = env_trial[i]
```

iii. Notes Step 2/Step 5: `environment` is the NWB name for the reference `morph` variable that `behavior.get_trial_types` reads (`morph = np.unique(sess.vr_data['morph'][firstI:lastI])`, `env_morph_dict = {'Env1': 0, 'Env2': 1}`). The agent verified that no trial window contains pre-sync (-1) or mixed environment values, and that the environment flips from 0 to 1 at lap 30 only on the cross-environment switch sessions (e.g. `Env1_C_to_Env2_B`).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. One value per trial: the rounded median of the `environment` samples inside the lap, broadcast as a constant over all timepoints of the trial so that every input is a (4, T) time series.

ii.
```python
env_trial[i] = int(np.round(np.median(S['env'][sl])))
...
x[1] = env_trial[i]
```

iii. The reference takes `np.unique(morph[first:last])` per trial; the median is the same thing for a within-trial-constant variable but is robust to a stray sample. Per-trial variables are stored as constant time series (Step 5 decision 7) so every trial has a uniform layout. The verification log shows the input range is exactly [0, 1], with 9 mice starting in ENV1 and m17/m18 in ENV2, matching the paper.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The within-session lap index, i.e. the position of the lap in the `trial_start`/`teleport` pairing — not the stored `trial number` behavioural series (which is loaded but not used for this input).

ii.
```python
starts=np.where(g('trial_start') > 0)[0],
stops=np.where(g('teleport') > 0)[0],
...
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    x[2] = i                           # lap index within the session
```

iii. The agent verified during exploration that the stored `trial number` series agrees with the lap index derived from `trial_start` ("trial number matches trial index"), so either source gives the same value; the loop index is used because it is the same index that drives the reward-zone switch at lap 30 and the previous-lap lookup.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond assigning the raw lap index, held constant across all timepoints of the trial. Crucially the **original** lap index is kept even though lap 0 and the lick-error laps are dropped, so the index stays interpretable relative to the reward-zone switch at lap 30. The observed range is therefore [1, 99] rather than [0, 99].

ii.
```python
x[2] = i                           # lap index within the session
```

iii. Notes Step 5 mapping table: "preserves the original lap index even when trials are dropped, so the switch at lap 30 stays interpretable". The trajectory records the same reasoning: "Dropping trial 0 doesn't affect those (keep original trial numbers)."

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The per-lap `isreward` vector, which is computed from the sparse `Reward` event timestamps (mapped onto frames with `np.searchsorted` over the behavioural timestamps) combined with the `reward_zone` flag series, following the reference `behavior.get_trial_types`.

ii.
```python
reward_t=beh['Reward']['timestamps'][()].astype(np.float64),
rzone_flag=g('reward_zone'),
...
reward_frames = np.searchsorted(S['t'], S['reward_t'])
for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s - 1, e - 1)
    rzflag = S['rzone_flag'][sl] > 0
    got_reward = np.any((reward_frames >= s - 1) & (reward_frames < e - 1))
    # behavior.get_trial_types: reward delivered AND reward zone active
    isreward[i] = int(got_reward and np.any(rzflag))
```

iii. Notes: "`Reward` is a sparse TimeSeries: one sample per delivered reward (data = 0.004 mL) with timestamps that exactly equal frame timestamps (verified on all 152 sessions)", so `searchsorted` is an exact lookup rather than a nearest-neighbour approximation. The AND with the reward-zone flag is the reference rule; the agent additionally checked (Step 12) that "across all 12,216 laps there is no lap with a delivered reward but no reward-zone flag", so the rule is equivalent to "a reward was delivered" on this dataset.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For lap `i`, the value is `isreward[i-1]`, held constant across the trial. The first lap of each session is not emitted at all, so there is never a lap with an undefined previous outcome (no 0-padding / invented label).

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    if i == 0:
        continue                      # previous-trial outcome undefined
    ...
    x[3] = isreward[i - 1]             # previous trial outcome
```

iii. Trajectory: "for trial 0 there's no previous trial. Options: drop trial 0, or set to 0 ... that's arbitrary ... warm-up trials preceded the imaging session (mice got warm-up trials before imaging), so the trial before trial 0 existed but isn't recorded. I think dropping the first trial is cleaner and well justified." The cost (152 of 12,216 laps, 1.2%) is documented in Step 5 decision 6. Step 10's independent check confirmed the value equals the recomputed `isreward` of the preceding lap for every trial of three test sessions.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` behavioural series plus the per-trial reward-zone coordinates. The reward zone is **not** inferred from the data: it is looked up from the VR scene name, which the agent recovered from the NWB `identifier` string (e.g. `Env1_LocationB_to_A`), using a port of the reference `behavior.get_reward_zones` — constant zone for non-switch scenes, and a switch to the second zone after lap 30 for the `*_to_*` scenes. The zone dictionary is the reference one (X/Y/Z renamed A/B/C): A = [80, 130], B = [200, 250], C = [320, 370] cm. The data-derived `reward_zone` flag is used only as an independent check of this assignment.

ii.
```python
REWARD_ZONE_DICT = {'A': [80.0, 130.0], 'B': [200.0, 250.0], 'C': [320.0, 370.0],
                    'T': [275.0, 325.0]}
CHANGE_TRIAL = 30           # behavior.get_reward_zones(change_trial=30)

def scene_reward_zones(scene, n_trials):
    """Mirrors reward_relative.behavior.get_reward_zones ..."""
    s = scene
    if '_to_' not in s:
        label = s[-1]
        labels = [label] * n_trials
    else:
        pre, post = s.split('_to_')
        first = pre[-1]              # ...LocationX  or  Env1_X
        second = post[-1]            # Y  or  Env2_Y
        labels = [first] * min(CHANGE_TRIAL, n_trials)
        if n_trials > CHANGE_TRIAL:
            labels += [second] * (n_trials - CHANGE_TRIAL)
    if any(l not in REWARD_ZONE_DICT for l in labels):
        raise ValueError('unrecognised scene %s' % scene)
    coords = np.array([REWARD_ZONE_DICT[l] for l in labels], dtype=float)
    return coords, np.array(labels)
...
rz_coords, rz_labels = scene_reward_zones(S['scene'], n_trials)
...
# sanity check: scene-derived zone vs zone entry position measured in the data
obs = rz_onset_pos[~np.isnan(rz_onset_pos)]
exp = rz_coords[~np.isnan(rz_onset_pos), 0]
zone_err = float(np.nanmedian(obs - exp)) if len(obs) else np.nan
if len(obs) and not (-1.0 <= zone_err <= 15.0):
    print('  WARNING %s: reward-zone mismatch, median(onset-zone_start)=%.1f cm' % ...)
```

iii. Notes Step 2/Step 4: the `identifier` field "gives the **scene name** needed for reward-zone lookup, exactly as reference `behavior.get_reward_zones` uses `sess.scene`", and the paper states "Each switch occurred after 30 trials". The assignment was validated per session against the position at which the `reward_zone` flag first fires: median error 0.78 cm over all 152 sessions (the per-session stats file shows every session between -0.06 and +1.84 cm), and the zone-label distribution is 33.1% / 33.6% / 33.2% for A / B / C, consistent with the paper's counterbalancing. Using the scene also assigns a zone on omission laps, where the flag never fires.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance in cm from the animal's position to the **nearest point of the zone**: negative before the zone, exactly 0 anywhere inside the 50 cm zone, positive after it. Then discretised (see 7-c).

ii.
```python
def reward_zone_distance(pos, rz_start, rz_end):
    """Signed distance (cm) to the nearest point of the reward zone.
    0 anywhere inside the zone, negative before the zone, positive after it."""
    d = np.zeros_like(pos)
    before = pos < rz_start
    after = pos > rz_end
    d[before] = pos[before] - rz_start
    d[after] = pos[after] - rz_end
    return d
...
d_rz = reward_zone_distance(pos, rz_coords[i, 0], rz_coords[i, 1])
y[0] = digitize_rz_distance(d_rz)
```

iii. Notes Step 10 Check 3: "distance is to the nearest point of the 50 cm zone (task asks for distance to *any* location in the zone) rather than to the zone start" — a deliberate, documented deviation from the reference `rel_pos`, which is measured from the zone start and is circular. The task specification asks for "distance to any location in the reward zone" with a dedicated "0 cm" class, which only makes sense with the nearest-point definition. The discretisation was verified with a value-vs-bin scatter over a whole session in the `--show-processing` figures.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes via explicit boolean masks, with the default class 3 reserved for exactly 0 (inside the zone): 0 for d < -50; 1 for -50 <= d < -10; 2 for -10 <= d < 0; 3 for d == 0; 4 for 0 < d <= 10; 5 for 10 < d <= 50; 6 for d > 50.

ii.
```python
def digitize_rz_distance(d):
    """7-way discretisation of the reward-zone distance (see task description)."""
    out = np.full(d.shape, 3, dtype=np.int64)   # exactly 0 -> in the zone
    out[d < -50.0] = 0
    out[(d >= -50.0) & (d < -10.0)] = 1
    out[(d >= -10.0) & (d < 0.0)] = 2
    out[(d > 0.0) & (d <= 10.0)] = 4
    out[(d > 10.0) & (d <= 50.0)] = 5
    out[d > 50.0] = 6
    return out
```
with the class names
```python
['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in reward zone (0 cm)',
 '0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
```

iii. The edges are those given in the Decoder Task section. The "0 cm" class is kept strictly for exact zero so that it means "in the reward zone", which is why the default fill value is 3 and every other class is written over it. The resulting distribution over the full dataset is [0.255, 0.102, 0.074, 0.239, 0.021, 0.072, 0.238]; class 3 is large (24%) because it covers the whole 50 cm zone where the animal slows down and licks.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with exactly the same frame range `[start-1, stop-1)` as the neural events, so no alignment step is needed.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]
pos = S['pos'][sl]
...
d_rz = reward_zone_distance(pos, rz_coords[i, 0], rz_coords[i, 1])
y[0] = digitize_rz_distance(d_rz)
```

iii. All streams share the imaging-frame clock (verified for all 152 sessions), and the 10 sessions with an extra ophys frame are truncated at load time, so identical indices imply identical times. The `processing_*_outputs.png` figure plots the converted neural raster and the position trace of the same trial on one time axis as a visual check.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural time series (cm along the 450 cm virtual track).

ii.
```python
pos=g('position'), ...
...
pos = S['pos'][sl]
y[1] = np.digitize(pos, POSITION_EDGES)
```

iii. `position` directly records the VR corridor position; the agent noted that it is -500 before VR synchronisation and -50 during the inter-trial teleport, and verified that no converted trial window contains either of those values.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing the trial's frames and discretising; the raw cm values are used as-is.

ii.
```python
pos = S['pos'][sl]
y[1] = np.digitize(pos, POSITION_EDGES)
```

iii. The Methods define the track as 0-450 cm and the NWB position is already in those units, so no rescaling is needed.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with the interior edges [90, 180, 270, 360], giving 5 classes of 90 cm each over the 450 cm track. The first and last classes are open-ended, so the handful of samples marginally outside 0-450 cm fall into the end classes rather than creating extra classes.

ii.
```python
POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]
TRACK_LENGTH = 450.0        # cm
...
y[1] = np.digitize(pos, POSITION_EDGES)
```
with the class names
```python
['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
```

iii. Directly the "5 equal-sized bins spanning the 450 cm track" of the task specification. The resulting distribution is [0.216, 0.176, 0.232, 0.227, 0.149] — near-uniform, as expected for laps run at roughly constant speed with slowing around the reward zone. The step function was verified with a position-vs-bin scatter over a whole session in the `--show-processing` figure.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame slice as the neural data; no further alignment.

ii.
```python
sl = slice(s - 1, e - 1)
pos = S['pos'][sl]
```

iii. Same justification as 7-d: a single shared frame clock, verified across all sessions, with the ophys/behaviour frame-count mismatch handled at load time.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural time series, which the agent determined is a cumulative lick count per frame (values 0-7) rather than a binary flag.

ii.
```python
lick=g('lick'),
...
lick = S['lick'][sl]
y[3] = (lick >= 1).astype(np.int64)
```

iii. Notes Step 2: "`lick` (cumulative lick count per frame, 0-7)". The reference `glmUtils.get_timeseries_data` binarises the same variable with `licks[licks > 1] = 1`.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation: any frame with at least one lick becomes 1, otherwise 0. In addition, laps whose lick trace trips the sensor-error rule (>30% of frames with a cumulative count > 2) are dropped entirely rather than emitted with an invalid lick label.

ii.
```python
y[3] = (lick >= 1).astype(np.int64)
```
```python
LICK_ERROR_THRESH = 0.30    # Methods: >30% of frames with cumulative lick count > 2
...
lick_error[i] = (np.sum(L > 2) / len(L)) > LICK_ERROR_THRESH
...
if lick_error[i]:
    continue                      # invalid lick data (reference sets to NaN)
```

iii. The task requires a binary lick output; the reference binarises the same way. For the sensor-error laps the notes state: "Lick-sensor error trials ... have lick values set to NaN in the reference; because lick is a required decoder output here, these trials cannot provide a valid lick label, so they are dropped entirely." The final lick distribution is 77.8% / 22.2%.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame slice as the neural data; no further alignment.

ii.
```python
sl = slice(s - 1, e - 1)
lick = S['lick'][sl]
```

iii. Same shared-clock argument as 7-d/8-d; the `--show-processing` figure overlays licks and reward-delivery stars on the position/neural time axis, showing licks clustering just before and inside the shaded reward zone.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The VR scene name parsed from the NWB `identifier`, mapped through the reference reward-zone dictionary with the switch after lap 30 — the same `scene_reward_zones` call used for 7-a. The `reward_zone` behavioural flag together with `position` is used only as an independent validation of the labels.

ii.
```python
rz_coords, rz_labels = scene_reward_zones(S['scene'], n_trials)
...
RZ_LABELS = ['A', 'B', 'C']
...
y[4] = RZ_LABELS.index(rz_labels[i])
```
validation:
```python
if np.any(rzflag):
    rz_onset_pos[i] = S['pos'][sl][np.argmax(rzflag)]
...
zone_err = float(np.nanmedian(obs - exp)) if len(obs) else np.nan
```

iii. See 7-a. The scene name is the authoritative session metadata the reference code itself uses, and the labels were cross-checked against the measured zone-entry position in every session (median error 0.78 cm, worst session 1.84 cm).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The per-lap letter label is mapped to an integer A = 0, B = 1, C = 2 and broadcast as a constant across all timepoints of the trial. Laps 0-29 carry the pre-switch zone, laps 30+ the post-switch zone; for sessions shorter than 30 laps the list is built with `min(CHANGE_TRIAL, n_trials)` so no negative-length tile is produced.

ii.
```python
labels = [first] * min(CHANGE_TRIAL, n_trials)
if n_trials > CHANGE_TRIAL:
    labels += [second] * (n_trials - CHANGE_TRIAL)
...
y[4] = RZ_LABELS.index(rz_labels[i])
```
with the class names
```python
['zone A (80-130 cm)', 'zone B (200-250 cm)', 'zone C (320-370 cm)'],
```

iii. Port of `behavior.get_reward_zones(change_trial=30)`, with the `min()` guard added for the one short session (m4 ses-04, 41 laps) — the reference would produce a negative tile count there. The class distribution across the converted data is [0.331, 0.336, 0.332].

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward` event timestamps together with the `reward_zone` flag series — the same `isreward` vector used for the previous-trial-outcome input.

ii.
```python
reward_t=beh['Reward']['timestamps'][()].astype(np.float64),
...
reward_frames = np.searchsorted(S['t'], S['reward_t'])
...
got_reward = np.any((reward_frames >= s - 1) & (reward_frames < e - 1))
isreward[i] = int(got_reward and np.any(rzflag))
...
y[5] = isreward[i]
```

iii. This is the reference `behavior.get_trial_types` rule (`np.any(tmp_reward > 0) and np.any(tmp_rzone > 0)`). The agent verified that the `Reward` timestamps coincide exactly with frame timestamps in all 152 sessions, so `searchsorted` maps each reward to its own frame without error.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward events are converted to frame indices; a lap is labelled 1 if at least one reward frame falls inside its window and the reward-zone flag fired at some point in the lap, else 0. The per-lap value is broadcast as a constant time series across the trial.

ii.
```python
reward_frames = np.searchsorted(S['t'], S['reward_t'])
isreward = np.zeros(n_trials, dtype=np.int64)
for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s - 1, e - 1)
    rzflag = S['rzone_flag'][sl] > 0
    got_reward = np.any((reward_frames >= s - 1) & (reward_frames < e - 1))
    isreward[i] = int(got_reward and np.any(rzflag))
...
y[5] = isreward[i]
```
with the class names `['omitted', 'rewarded']`.

iii. The measured reward rate is 84.7% of raw laps (84.3% of converted frames), matching the paper's ~15% omission rate. The agent also checked (Step 12) that `autoreward` is identically zero in every file, that there are 52 laps with a zone flag but no reward (the reference's "lapsed" trials) and none with a reward but no zone flag, and that the label is decodable mainly *after* reward-zone entry (0.531 balanced accuracy before the zone vs 0.637 in/after it), which is the expected physiology rather than a conversion error.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Seven cases:
- **Ophys/behaviour frame-count mismatch** (10 sessions of m17/m18, ophys one frame longer): both streams truncated to the common length, with a printed NOTE. This crashed the first full run and was fixed.
- **Numerically degenerate dF/F** (25 cells whose maximin baseline collapses towards 0 because Fneu >> F): those cells are excluded, bringing the dF/F range from [-931, 3013] down to [-16.0, 37.2].
- **Non-finite deconvolved events** in a trial: the trial is dropped with a warning (never triggered).
- **Lick-sensor errors** (81 laps): laps dropped (see 1-e / 9-b).
- **Undefined previous-trial outcome** on the first lap of a session: lap dropped rather than assigned a made-up value.
- **Sessions with < 2 usable trials**: dropped, since the decoder cannot be evaluated on them (never triggered).
- **Sessions with fewer than 30 laps** on a switch scene: the zone label list is built with `min(CHANGE_TRIAL, n_trials)`, avoiding the negative tile count the reference would produce.
Additionally, per-session assertions guard the invariants (equal numbers of starts and stops, starts before stops, no `trial_start` at frame 0), and an automatic warning fires if the scene-derived reward zone disagrees with the measured zone-entry position.

ii.
```python
n_beh = d['pos'].shape[0]
if d['F'].shape[1] != n_beh:
    print('  NOTE %s: ophys has %d frames, behaviour %d; truncating to %d' % ...)
    n = min(d['F'].shape[1], n_beh)
    d['F'] = d['F'][:, :n]; d['Fneu'] = d['Fneu'][:, :n]
    for k in ['pos','speed','lick','rzone_flag','env','trialnum','autoreward','scanning','t']:
        d[k] = d[k][:n]
    d['starts'] = d['starts'][d['starts'] < n - 1]
    d['stops'] = d['stops'][:len(d['starts'])]
    d['starts'] = d['starts'][:len(d['stops'])]
```
```python
assert n_trials == len(stops) and np.all(starts < stops) and starts[0] >= 1
...
bad_baseline = ~(min_base > BASELINE_MIN_FRAC * scale)
...
if not np.all(np.isfinite(ev)):
    print('  WARNING %s trial %d: non-finite events, trial dropped' % ...)
    continue
...
sessions = [r[0] for r in results if r[0] is not None and len(r[0]['neural']) >= 2]
```
```python
def _worker(args):
    fn, show, plotdir = args
    try:
        return process_session(fn, show=show, plotdir=plotdir)
    except Exception:
        traceback.print_exc()
        return None, dict(file=os.path.basename(fn), error=True)
```

iii. All of these are documented in Step 9 ("Issue found and fixed during the first full run"), Step 10 Check 5 ("Check for edge cases") and Step 12 ("Issues Found and Resolved"). The frame-mismatch fix is argued from the data ("Both streams start at t = 0 (ophys starting_time = 0 and behaviour timestamps[0] = 0), so the extra imaging frame is at the end"); the degenerate-baseline exclusion is argued from the decoder's PCA being dominated by such cells; each exclusion is counted and reported in the conversion summary so the loss of data is auditable (25 cells = 0.018%, 81 + 152 laps = 1.9%).

## 13-a. What are the most time-consuming steps of the code?

i. The script instruments itself (`t_load`, `t_dff`, `t_int`, `t_total` per session, saved to `converted_data_session_stats.csv` and printed per session). Summed over the 152 sessions: **dF/F + OASIS deconvolution 798 s (73%)**, HDF5 reading of F/Fneu 203 s (19%), interneuron correlation 32 s (3%), everything else negligible; writing the 9.47 GB pickle takes 14.5 s. With 12 worker processes the whole run is 199 s wall clock.

ii.
```python
t0 = time.time(); S = load_session(fn); t_load = time.time() - t0
t1 = time.time(); dff, events, bad_baseline = compute_dff_and_events(...); t_dff = time.time() - t1
t2 = time.time(); ... ; t_int = time.time() - t2
...
stats = dict(..., t_load=t_load, t_dff=t_dff, t_int=t_int, t_total=time.time() - t0)
...
print('[%5.1f s] %s: %d cells (%d iscell, %d interneurons), '
      '%d/%d trials kept  (load %.1fs dff %.1fs)' % ...)
```

iii. Notes Step 7 estimated "~7-10 min wall clock" for the full dataset based on the sample run and stated the per-session cost is dominated by HDF5 reading and the dF/F computation; the instructions asked for timing information to find bottlenecks, and the measured 199 s was well under the 15-minute budget, so no further optimisation was pursued.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining Python-level loops are the three per-trial loops inside `compute_dff_and_events` (window masking, maximin baseline, smoothing + OASIS) and the two per-trial loops in `process_session` (trial-type computation and per-trial array construction). These are all over ~80 variable-length laps per session and each iteration already does vectorised work over all cells at once, so the loop overhead is negligible; genuinely vectorising them would need padding or ragged-array machinery. The one loop that mattered — the per-cell Pearson correlation of dF/F against speed in the reference `is_putative_interneuron` (a Python loop over up to 2,341 cells) — was replaced by a single matrix-vector product.

ii.
```python
# vectorised replacement for the reference's per-cell np.corrcoef loop
valid = ~np.isnan(dff[0, :])
sp = S['speed'][valid]; X = dff[:, valid]
Xc = X - X.mean(axis=1, keepdims=True); spc = sp - sp.mean()
denom = np.sqrt((Xc ** 2).sum(axis=1) * (spc ** 2).sum())
speed_corr = (Xc @ spc) / np.where(denom == 0, np.nan, denom)
```
remaining per-trial loops:
```python
for start, stop in zip(starts, stops):
    sl = slice(start - 1, stop - 1)
    f_[:, sl] = f_[:, sl] + NEU_COEF * np.nanmean(fneu_[:, sl], axis=1, keepdims=True)
    base = gaussian_filter1d(f_[:, sl], BASELINE_SMOOTH_SIGMA, axis=-1)
    ...
```

iii. Notes Step 6: "Code inefficiencies identified: naive per-cell correlation loop for interneuron detection; float64 intermediates for F/Fneu. Code speedups added: vectorised correlation (matrix-vector product), OASIS on float32 arrays, reading only iscell columns from HDF5, parallel sessions." The per-trial loops are kept because the reference `preprocessing.dff` is structured that way and the baseline/deconvolution are defined per lap.

## 13-c. What processing does the code repeat multiple times?

i. Very little. Each NWB file is opened and read exactly once, and the conversion is a single pass — there is no separate survey/statistics pass over the data. Within a session the trial loop in `compute_dff_and_events` walks the lap list three times (mask, baseline, smooth+deconvolve), mirroring the reference function's structure; `process_session` then walks the laps twice more (once for trial-level variables, once to build the arrays). The raw `F` matrix is re-read from memory a second time to compute the `bad_baseline` scale. In `--show-processing` mode a few quantities (positions, reward-zone distance) are recomputed for plotting.

ii.
```python
files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
...
with ctx.Pool(min(args.nproc, len(files)), maxtasksperchild=1) as pool:
    for res in pool.imap(_worker, [(fn, fn in show_files, '/app') for fn in files]):
```
```python
for start, stop in zip(starts, stops):   # 1: mask windows
for start, stop in zip(starts, stops):   # 2: neuropil mean + maximin baseline
for start, stop in zip(starts, stops):   # 3: smooth dF/F + OASIS
...
for i, (s, e) in enumerate(zip(starts, stops)):   # 4: trial types / lick errors
for i, (s, e) in enumerate(zip(starts, stops)):   # 5: build per-trial arrays
```

iii. The single-pass design was possible because the reward-zone assignment comes from the scene name rather than from a data-driven inference over the whole dataset, so no prior survey of all sessions is needed. The repeated lap loops inside the dF/F port are inherited from the reference function and cost nothing measurable relative to the filtering and OASIS work inside them.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A small amount:
- `autoreward`, `scanning` and `trialnum` are read from every file but never used in the conversion (they were used during exploration to validate the trial windows; `autoreward` was found to be identically zero).
- dF/F and events are computed and kept as full (n_cells x n_frames) arrays for *all* curated cells, including those later dropped as interneurons or unstable-baseline cells, and including the between-lap samples that are NaN; only the kept cells' within-lap columns end up in the output. Computing dF/F for the interneurons is unavoidable, since the interneuron test is defined on dF/F.
- Per-session diagnostic statistics (`dff_min`, `dff_max`, `ev_max`, `zone_err`, timings) are computed for every session; they go into the log and the stats CSV, not into the converted data.
- `REWARD_ZONE_DICT` contains an unused `'T': [275.0, 325.0]` entry; no scene in this dataset maps to it.
- Per-trial constants (environment, trial number, previous outcome, reward-zone location, reward outcome) are materialised as full-length constant rows, which inflates the pickle but is required by the target format.

ii.
```python
trialnum=g('trial number'), autoreward=g('autoreward'),
scanning=g('scanning'),
```
```python
REWARD_ZONE_DICT = {'A': [80.0, 130.0], 'B': [200.0, 250.0], 'C': [320.0, 370.0],
                    'T': [275.0, 325.0]}
```
```python
stats = dict(..., dff_min=float(np.nanmin(dff[keep_cells])),
             dff_max=float(np.nanmax(dff[keep_cells])),
             ev_max=float(np.nanmax(events[keep_cells])), ...)
```

iii. The agent did not flag any of this as wasted work; the diagnostic statistics are deliberate (they back the consistency tables in CONVERSION_NOTES.md Steps 9-10 and the automatic reward-zone warning), and the extra behavioural series are cheap 1-D arrays. The dominant memory/compute cost — computing dF/F for cells that are subsequently excluded — is intrinsic to the reference curation rule rather than an avoidable inefficiency.
