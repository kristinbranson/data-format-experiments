# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All 152 `.nwb` files are found by a single glob over `/app/data/sub-*/sub-*.nwb` (sorted), one file per subject-session. Files are opened directly with **h5py** rather than `pynwb`, and only the datasets that are actually needed are read: per-plane `Fluorescence`/`Neuropil` traces (restricted to `iscell==1` ROIs at read time), the `ImageSegmentation/PlaneSegmentation` `iscell`/`planeIdx` tables, the behavior time series (`position`, `speed`, `lick`, `reward_zone`, `environment`, `autoreward`, `scanning`, `trial number`, `trial_start`, `teleport`), the behavior timestamps and the sparse `Reward` event timestamps, plus `identifier` (scene), `subject_id` and `session_id`. Sessions are processed independently and in parallel (`ProcessPoolExecutor`, 8 spawned workers). The AI verified against its own raw scan of all files: 152 sessions, 11 subjects, 12,216 trial starts, 138,678 `iscell` ROIs.

ii.
```python
def session_files():
    import glob
    return sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', 'sub-*.nwb')))

def load_session(path):
    """Read everything needed from one NWB file."""
    with h5py.File(path, 'r') as f:
        ident = f['identifier'][()].decode()
        scene = ident.split('/')[-1]
        subject = f['general/subject/subject_id'][()].decode()
        day = f['general/session_id'][()].decode()
        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = seg['iscell'][:, 0] > 0
        plane_idx = seg['planeIdx'][:]
        fl = f['processing/ophys/Fluorescence']
        planes = sorted(fl.keys())
        Fl, Fnl = [], []
        for p in planes:
            sel = iscell[plane_idx == int(p.replace('plane', ''))]
            Fl.append(f[f'processing/ophys/Fluorescence/{p}/data'][:, :].T[sel])
            Fnl.append(f[f'processing/ophys/Neuropil/{p}/data'][:, :].T[sel])
        F = np.concatenate(Fl, axis=0)
        Fneu = np.concatenate(Fnl, axis=0)
        b = f['processing/behavior/BehavioralTimeSeries']
        g = lambda k: b[f'{k}/data'][:]
        beh = dict(pos=g('position'), speed=g('speed'), lick=g('lick'),
                   rzone=g('reward_zone'), env=g('environment'),
                   autoreward=g('autoreward'), scanning=g('scanning'),
                   trialnum=g('trial number'))
        time = b['position/timestamps'][:]
        starts = np.where(g('trial_start') > 0)[0]
        stops = np.where(g('teleport') > 0)[0]
        reward_t = b['Reward/timestamps'][:]
```

iii. From CONVERSION_NOTES Step 2/Step 6: the DANDI release is "one file per subject-session"; reading with h5py avoids `pynwb` overhead and lets the loader slice out only the curated ROIs ("~2x less I/O"). The AI cross-checked the file count, subject count and trial count against the paper (11 switch mice, 80.5 ± 7.4 trials/session) before converting, and states the NWB release already contains the VR-to-imaging-frame-aligned streams that the reference `sess` class produces, so nothing further needs to be loaded.

## 1-b. How are the data split into subjects?

i. The subject identity of every session is read from `general/subject/subject_id` **inside** each NWB file (not from the directory name). The unique subject IDs are sorted numerically (`m3, m4, m7, m11, …, m19`) to form `data['subjects']`, and `subject_idx` indexes that list once per session. 11 subjects result, matching the paper's 11 "switch" mice.

ii.
```python
subjects = sorted({i['subject'] for i in infos}, key=lambda x: int(x[1:]))
subject_idx = np.array([subjects.index(i['subject']) for i in infos], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 4: "DANDI contains exactly the 11 switch mice; the 3 fixed-condition mice are not in the release. Use all 11." The paper's `dayData.include_ans` keeps only switch animals, and the released dataset already contains only those, so no animal-level exclusion is needed.

## 1-c. How are the data split into sessions?

i. One NWB file = one session (152 total). No merging across days or across mice is attempted; sessions are kept in sorted-filename order and the per-session experiment day is recorded from `general/session_id` in `metadata['session_info']`.

ii.
```python
files = session_files()          # 152 sorted sub-mXX_ses-NN_behavior+ophys.nwb
...
for i, r in enumerate(ex.map(_worker, [(f, False, '/app', args.signal) for f in files])):
    results.append(r)
...
info = dict(subject=S['subject'], day=S['day'], scene=S['scene'], ...)
```

iii. CONVERSION_NOTES Step 2: files are named `sub-mXX_ses-NN_behavior+ophys.nwb`, 14 days per mouse except m11 (12, "imaging started on day 3"), which reproduces the paper's description exactly. The AI kept all 152 sessions ("all are switch mice used in the paper", Step 5 decision 7).

## 1-d. How are the data split into trials?

i. Trial boundaries come from the `trial_start` and `teleport` behavior time series. The per-trial window is `[trial_start_idx - 1, teleport_idx - 1)` — **exactly** the window used by the reference `preprocessing.dff`, which deliberately excludes the teleport sample itself (its position is interpolated between the end of the track and the ITI). The same index slice is used for the neural data and for every behavioral stream, so all streams are split identically. If the number of starts and stops differs after the length truncation (see 12), the common prefix is used.

ii.
```python
starts = np.where(g('trial_start') > 0)[0]
stops = np.where(g('teleport') > 0)[0]
...
starts = starts[starts < n]
stops = stops[stops <= n]
ntr = min(len(starts), len(stops))
starts, stops = starts[:ntr], stops[:ntr]
```
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    sl = slice(s - 1, e - 1)          # reference trial window
    ev = events[:, sl]
    ...
    pos = beh['pos'][sl]
    ...
    tt = S['time'][sl] - S['time'][s - 1]
```

iii. CONVERSION_NOTES Step 5: "Trial window = reference window used everywhere in the paper code (`preprocessing.dff`, `glmUtils.get_timeseries_data`): samples `[trial_start_idx - 1 : teleport_idx - 1)`. The teleport sample itself is excluded (its position is interpolated between end of track and the ITI)." Step 10 Check 3 records the window as identical to the reference. The AI verified 12,216 trial starts == 12,216 teleports across all files (no mismatch) and that trial lengths (mean 80.4 trials/session) match the paper's 80.5 ± 7.4.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, applied per trial:
1. **Lick-sensor-error trials are dropped** — the paper's criterion: >30% of the frames in a trial have a cumulative lick count >2. This removed exactly **81 trials (0.67%)**, matching the paper's "n = 81 out of 12,376 trials".
2. Trials with fewer than 2 time bins are dropped (0 triggered).
3. Trials containing any NaN in the neural signal are dropped (0 triggered).
4. Trials containing pre-TTL-sync samples (`position < -100`, i.e. the `pos = -500` sentinel) are dropped (0 triggered).

No minimum trial-length filter beyond 2 bins is applied; the shortest surviving trial is 96 bins. 12,135 of 12,216 trials are kept. No sessions are excluded.

ii.
```python
LICK_ERR_FRAC = 0.30         # >30% of samples with cumulative lick > 2 -> error
...
    # lick sensor error (paper: >30% of samples in the trial with cum lick > 2)
    lick_tr = beh['lick'][s - 1:e - 1]
    lick_error[i] = (np.sum(lick_tr > 2) / max(1, len(lick_tr))) > LICK_ERR_FRAC
...
drop = dict(lick_error=0, too_short=0, nan_events=0, pre_sync=0)
for i, (s, e) in enumerate(zip(starts, stops)):
    if lick_error[i]:
        drop['lick_error'] += 1       # lick is an output; cannot be NaN
        continue
    sl = slice(s - 1, e - 1)
    ev = events[:, sl]
    if ev.shape[1] < 2:
        drop['too_short'] += 1
        continue
    if np.any(np.isnan(ev)):
        drop['nan_events'] += 1
        continue
    pos = beh['pos'][sl]
    if np.any(pos < -100):            # pre-TTL-sync samples (pos = -500)
        drop['pre_sync'] += 1
        continue
```

iii. CONVERSION_NOTES Step 5 decision 3 and Step 10 Check 3: "trials whose lick trace shows sensor error (>30% of samples with cumulative lick count > 2, the paper's criterion) are **dropped**, because lick is a decoder output and NaNs are not allowed. This removes 80 trials (0.65%), matching the paper's 81/12,376." (The full run reports 81.) The AI notes the paper sets these lick values to NaN rather than removing the trial, and that dropping is forced by the decoder format. The other three filters are described in Step 10 Check 5 as defensive edge-case guards that in practice never fire.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces `processing/ophys/Fluorescence/planeK/data` (F) and `processing/ophys/Neuropil/planeK/data` (Fneu), restricted to ROIs with `iscell == 1` from `ImageSegmentation/PlaneSegmentation`. The NWB `processing/ophys/Deconvolved` series is explicitly **not** used: the AI identified it as suite2p's default `spks`, computed with a different (non-per-trial) baseline, not the paper's `events`.

ii.
```python
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = seg['iscell'][:, 0] > 0
plane_idx = seg['planeIdx'][:]
fl = f['processing/ophys/Fluorescence']
planes = sorted(fl.keys())
for p in planes:
    sel = iscell[plane_idx == int(p.replace('plane', ''))]
    Fl.append(f[f'processing/ophys/Fluorescence/{p}/data'][:, :].T[sel])
    Fnl.append(f[f'processing/ophys/Neuropil/{p}/data'][:, :].T[sel])
F = np.concatenate(Fl, axis=0)
Fneu = np.concatenate(Fnl, axis=0)
```

iii. CONVERSION_NOTES Step 4 table: "paper `events` = OASIS deconvolution of *their* maximin dF/F; NWB `Deconvolved` = suite2p default spks (different baseline, no per-trial restriction) → **Recompute** dF/F (maximin, neuropil 0.7) + OASIS from NWB F/Fneu, exactly as `preprocessing.dff`." Step 5 key decision 1 repeats this rationale.

## 2-b. How is the `neural` data processed?

i. `compute_events()` is a re-implementation of the reference `reward_relative.preprocessing.dff(..., neuropil_method='subtract', baseline_method='maximin', subtract_baseline=True, neu_coef=0.7, tau=0.7, deconvolve=True)`:
- everything outside the per-trial windows `[start-1, stop-1)` is set to NaN;
- neuropil subtraction `F - 0.7·Fneu`, with the per-trial mean of `0.7·Fneu` added back;
- maximin baseline per trial: NaN-aware Gaussian smoothing (σ = 15 frames) → 300-sample `minimum_filter1d` → 300-sample `maximum_filter1d` (the Methods' ~20 s window);
- `dF/F = (F − F0)/|F0|`, then a 2-frame NaN-aware Gaussian smooth;
- `dcnv.oasis(dff, 2000, tau=0.7, frame_rate)` per trial → "events".

Planes are pooled for the two 2-plane mice. **The delivered dataset stores the dF/F stage, not the deconvolved events** (`--signal` defaults to `dff`): the AI ran a controlled comparison (12 sessions, two runs each, plus the full dataset) and found dF/F decoded better for every output, so it shipped dF/F and left `--signal events` to regenerate the deconvolved version.

Two differences from the reference pipeline: (1) the delivered signal stage (dF/F rather than events), and (2) the reference notebook `make_multi_anim_sess.ipynb` sets `keep_teleports=True` for specific animal/day combinations listed in `teleport_metadata.py` (which widens the baseline window across the teleport); the AI's code always uses the `keep_teleports=False` branch and never discusses this.

ii.
```python
def compute_events(F, Fneu, starts, stops, frame_rate):
    f_ = np.full(F.shape, np.nan, dtype=np.float64)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float64)
    for start, stop in zip(starts, stops):
        f_[:, start - 1:stop - 1] = F[:, start - 1:stop - 1]
        fneu_[:, start - 1:stop - 1] = Fneu[:, start - 1:stop - 1]
    nanmask = ~np.isnan(f_[0, :])
    f_ -= NEU_COEF * fneu_
    flow = np.full(F.shape, np.nan)
    for start, stop in zip(starts, stops):
        sl = slice(start - 1, stop - 1)
        f_[:, sl] = f_[:, sl] + NEU_COEF * np.nanmean(fneu_[:, sl], axis=1, keepdims=True)
        flow[:, sl] = nansmooth(f_[:, sl], BASELINE_SIGMA)
        flow[:, sl] = ndimage.minimum_filter1d(flow[:, sl], BASELINE_WIN, axis=-1)
        flow[:, sl] = ndimage.maximum_filter1d(flow[:, sl], BASELINE_WIN, axis=-1)
    dff = np.full(F.shape, np.nan)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    events = np.full(F.shape, np.nan, dtype=np.float32)
    for start, stop in zip(starts, stops):
        sl = slice(start - 1, stop - 1)
        dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH, axis=1)
        events[:, sl] = dcnv.oasis(np.ascontiguousarray(dff[:, sl], dtype=np.float32),
                                   2000, TAU, frame_rate)
    return dff, events
```
```python
ap.add_argument('--signal', choices=['events', 'dff'], default='dff', ...)
...
if signal == 'dff':
    events = dff_kept.astype(np.float32)
```

iii. CONVERSION_NOTES Step 3/Step 5: the pipeline parameters are taken from the Methods ("baseline fluorescence was calculated within each trial independently using a maximin procedure with a 20 s sliding window … smoothed with a two-sample Gaussian kernel … deconvolving dF/F … using the OASIS algorithm") and from the repo (`neu_coef=0.7`, `tau=0.7`). Step 12 Check 2 justifies shipping dF/F: a repeated controlled experiment (events 0.459/0.558/0.449/0.788/0.835/0.595 vs dF/F 0.569/0.724/0.548/0.812/0.878/0.601 validation balanced accuracy) plus the paper's own statement that dF/F "is the closest to the raw data", and the instruction that deviations are allowed where the decoding task requires them.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's: (1) only `iscell == 1` ROIs (suite2p manual curation), applied at load time; (2) putative interneurons removed — cells whose dF/F has Pearson r > 0.5 with running speed over all in-trial samples (`dayData.int_thresh = 0.5`). This removes 409 of 138,678 cells (0.30%), leaving 138,269 (154–2,323 per session). No further filtering (e.g. place-cell selection) is applied.

ii.
```python
INT_R_THRESH = 0.5           # putative interneuron speed-correlation threshold
...
nanmask = ~np.isnan(dff[0, :])
sp = beh['speed'][nanmask]
D = dff[:, nanmask]
Dz = D - D.mean(axis=1, keepdims=True)
spz = sp - sp.mean()
denom = (np.sqrt((Dz ** 2).sum(axis=1)) * np.sqrt((spz ** 2).sum()))
with np.errstate(invalid='ignore', divide='ignore'):
    speed_corr = (Dz @ spz) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
keep_cells = ~is_int
events = events[keep_cells]
dff_kept = dff[keep_cells]
```

iii. CONVERSION_NOTES Step 3/Step 5 decision 2: "`iscell == 1` (suite2p manual curation, already in NWB) and exclusion of putative interneurons with dF/F-speed Pearson r > 0.5 (paper value; `dayData.int_thresh = 0.5`). Planes pooled for the two 2-plane mice, as in the paper." The AI checked the resulting fraction (0.30%) against the paper's "0.42 ± 0.85% of cells" and the per-session neuron range (154–2,323) against "155–2172 putative pyramidal neurons per session". The correlation is computed as a single matrix product instead of the reference's per-cell loop (same result, faster).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is the start of the trial, so no additional alignment is needed beyond slicing: neural and behavioral samples are already 1:1 in the NWB (the VR stream was interpolated onto imaging frame times before release). Each trial's neural matrix is `events[:, start-1:stop-1]` and the time input is zeroed at the same first sample, so t = 0 is the first neural bin of the trial. `metadata['temporal_alignment_event'] = 'start of trial (entry to the linear track at position 0 cm)'`, `off_start = 0.0`, `off_end = None` (variable trial length). No pre-trial baseline window is included.

ii.
```python
sl = slice(s - 1, e - 1)          # reference trial window
ev = events[:, sl]
...
tt = S['time'][sl] - S['time'][s - 1]
...
'temporal_alignment_event': 'start of trial (entry to the linear track at position 0 cm)',
'off_start': 0.0,
'off_end': None,
```

iii. CONVERSION_NOTES Step 5: "Alignment event: **start of trial** (`trial_start` == entry to the linear track at 0 cm) … Neural and behavioral samples are already aligned 1:1 in the NWB file (VR was interpolated onto imaging frame times by `TwoPUtils.preprocessing.vr_align_to_2P` before the NWB conversion)." Step 7 states the processing plots confirm the position sawtooth resets exactly at the green trial-start lines.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **No rebinning.** Data are kept at the native imaging frame rate. The per-session frame rate is derived from the median difference of the behavior timestamps (15.5078 Hz for every session, including the two 2-plane mice, because the stored ophys `rate` attribute is the scanner rate = 2× the per-plane rate). `metadata['time_bin_size']` is the median over sessions = **64.484 ms**.

ii.
```python
# frame rate: use the median sampling interval of the behavior timestamps,
# which equals the per-plane imaging rate (the `rate` attribute of the
# ophys series is the raw scan rate, i.e. 2x for the 2-plane mice)
frame_rate = 1.0 / np.median(np.diff(time))
...
frame_rates = np.array([i['frame_rate'] for i in infos])
bin_ms = float(1000.0 / np.median(frame_rates))
...
'time_bin_size': bin_ms,
```

iii. CONVERSION_NOTES Step 5 decision 8: "Keep native 64.48 ms frame bins (no re-binning), matching 'All behavioral and neural time series were sampled at ~15.5 Hz'." Step 2 notes that the 2-plane mice (m17, m18) have `imaging_rate` 31.0156 Hz but a per-plane/volume rate of 15.5 Hz, and that the behavior dt is 0.06448 s for both 1- and 2-plane sessions, so a single bin size is valid for the whole dataset (the same rate is also used as the OASIS frame rate).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `processing/behavior/BehavioralTimeSeries/position/timestamps` array (the common imaging-frame time base shared by all behavior series and by the neural data).

ii.
```python
time = b['position/timestamps'][:]
...
tt = S['time'][sl] - S['time'][s - 1]
```

iii. CONVERSION_NOTES Step 2: all behavior series are "n_frames long with `timestamps` (imaging frame times, dt = 0.0645 s)", and Step 5 maps `behavior/position/timestamps` → `input[0]`. Any of the behavior series' timestamps would have served; the AI's Step 10 sanity check recomputed `timestamps - timestamps[trial_start-1]` directly from the raw file and found `np.allclose` True.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Only subtraction of the timestamp of the trial's first sample, so every trial's time input starts at exactly 0 and increments by ~64.484 ms. Stored as `input[0]`, a time-varying float32 row. Range over the dataset: 0 to 216.5 s.

ii.
```python
tt = S['time'][sl] - S['time'][s - 1]
...
inp = np.stack([tt,
                np.full(T, float(morph[i])),
                np.full(T, float(i)),
                np.full(T, float(isreward[i - 1]) if i > 0 else 1.0)]).astype(np.float32)
```

iii. Step 5 mapping table: "`t - t[first sample of trial]`". Step 10 Check 2 verifies this against the raw file; Step 5 planned sanity check "time_from_trial_start starts at 0 and increases by 64.48 ms per bin".

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the identical index slice as the neural data, so alignment is automatic. Before slicing, every stream (F, Fneu, all behavior series, timestamps) is truncated to the common minimum length, which fixes the 10 two-plane sessions that have one more imaging frame than behavior samples.

ii.
```python
n = min([F.shape[1], Fneu.shape[1], len(time)] + [len(v) for v in beh.values()])
if F.shape[1] != n or len(time) != n:
    print(f'  {os.path.basename(path)}: truncating streams to {n} samples ...')
F = F[:, :n]; Fneu = Fneu[:, :n]; time = time[:n]
beh = {k: v[:n] for k, v in beh.items()}
```

iii. CONVERSION_NOTES Step 9: "10 sessions (m17 ses-04/06, m18 ses-01/05/07/10/11/12/13/14, all 2-plane) have exactly one more imaging frame than behavior samples — the known 'one frame correction' that the reference alignment code (`vr_align_to_(mock_)2P`) also handles. Fixed by truncating every stream … which keeps neural and behavior sample-aligned." Step 4 confirms "neural samples == behavior samples" otherwise.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `processing/behavior/BehavioralTimeSeries/environment` time series (the `morph` variable of the reference code): 0 = ENV1, 1 = ENV2, −1 before TTL sync.

ii.
```python
beh = dict(..., env=g('environment'), ...)
...
u = np.unique(beh['env'][sl])
morph[i] = int(u[0])
```

iii. CONVERSION_NOTES Step 4: "`morph` 0 = ENV1, 1 = ENV2; `environment` timeseries values {−1 pre-sync, 0, 1}; constant within every trial; 11 sessions (day 8) contain both" — consistent with the paper's ENV1/ENV2 and with `behavior.get_trial_types`, which takes `np.unique(vr_data['morph'][firstI:lastI])`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique value of `environment` over the trial (the reference `get_trial_types` window `[start:stop]`) is taken as a per-trial scalar and then tiled across all timepoints of the trial as `input[1]`. Values are 0/1 as required.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s, e)   # get_trial_types uses [start:stop]
    ...
    u = np.unique(beh['env'][sl])
    morph[i] = int(u[0])
...
inp = np.stack([tt, np.full(T, float(morph[i])), ...])
```

iii. Step 5 mapping: "`behavior/environment` (morph) → `input[1]`, 0 = ENV1, 1 = ENV2 (constant within trial, verified)", following `behavior.get_trial_types`. Per-trial variables are tiled over time so that all input rows have a uniform `(d_input, T)` shape.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the within-session lap index derived from the enumeration of the `trial_start`/`teleport` pairs, i.e. the raw trial counter — *not* the stored `trial number` behavior series (which is loaded but never used). Because the index is the raw one, numbers are skipped when a trial is dropped for a lick-sensor error, preserving the true lap number.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    inp = np.stack([tt,
                    np.full(T, float(morph[i])),
                    np.full(T, float(i)),
                    ...])
```

iii. Step 5 mapping table: "trial index within session → `input[2]` trial_number, 0-indexed lap number, constant within trial", referencing `glmUtils.get_timeseries_data`'s `trials` variable. Verified range 0–99, consistent with sessions of up to 100 trials.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond casting the loop index to float and tiling it across the trial's timepoints.

ii.
```python
np.full(T, float(i))
```

iii. Same as 5-a; the value is a per-trial constant broadcast over time to keep `input` 2-D.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the per-trial `isreward` vector, which follows `behavior.get_trial_types`: a trial counts as rewarded if **both** a `Reward` event falls inside the trial **and** the `reward_zone` flag is non-zero in the trial. Reward events are sparse and carry their own timestamps, so they are mapped onto the behavior time base with `np.searchsorted` (clipped to the valid index range) to build a binary `reward` series.

ii.
```python
reward_t = b['Reward/timestamps'][:]
...
reward = np.zeros_like(beh['pos'])
if len(reward_t):
    idx = np.clip(np.searchsorted(time, reward_t), 0, len(reward) - 1)
    reward[idx] = 1
beh['reward'] = reward
...
isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
```

iii. CONVERSION_NOTES Step 1 documents `get_trial_types`: "`isreward` = any(reward>0) AND any(rzone>0)". Step 5 maps "previous trial `isreward` → `input[3]`". The AND with the reward-zone flag is the reference code's definition of a rewarded trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial *i* > 0 the value is `isreward[i-1]` — the outcome of the previous **raw** trial, even if that trial was itself dropped for a lick error. For the first trial of a session the value is set to **1 (rewarded)** rather than 0. The value is tiled across the trial's timepoints.

ii.
```python
np.full(T, float(isreward[i - 1]) if i > 0 else 1.0)
```

iii. CONVERSION_NOTES Step 5 mapping and Step 10 Check 5: "first trial of a session set to 1 (the mouse had just run ~30 rewarded warm-up trials on the same zone immediately before imaging)". This is supported by the Methods: "Before the imaging session, mice were provided 30 'warm-up' trials using the task and reward zone from the previous day." The AI also verified that `input[3]` is essentially uncorrelated with the current trial's outcome (r = 0.021) so it does not leak the label.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behavior time series plus the per-trial reward-zone identity. The zone identity is obtained the way the reference code does it — parsed from the **scene name** in `/identifier` (e.g. `Env1_LocationB_to_A`, `Env2_B_to_Env1_A`) with the switch occurring at trial index 30 — and mapped to the paper's fixed coordinates A = 80–130, B = 200–250, C = 320–370 cm (`reward_zone_dict` keys X/Y/Z). The `reward_zone` behavior flag is used only to validate this assignment (and in the reward-outcome definition), not to infer the zone.

ii.
```python
REWARD_ZONES = {'A': (80., 130.), 'B': (200., 250.), 'C': (320., 370.)}
CHANGE_TRIAL = 30            # behavior.get_reward_zones(change_trial=30)

def scene_zone_labels(scene, ntrials, change_trial=CHANGE_TRIAL):
    """Per-trial reward zone label from the scene name (behavior.get_reward_zones)."""
    m = re.match(r'Env\d_Location([ABC])$', scene)
    if m:
        return np.array([m.group(1)] * ntrials)
    m = re.match(r'Env\d_Location([ABC])_to_([ABC])$', scene)
    if m is None:
        m = re.match(r'Env\d_([ABC])_to_Env\d_([ABC])$', scene)
    if m:
        z0, z1 = m.group(1), m.group(2)
        n0 = min(change_trial, ntrials)
        return np.array([z0] * n0 + [z1] * (ntrials - n0))
    raise ValueError(f'unhandled scene name: {scene}')
...
zone_labels = scene_zone_labels(S['scene'], ntrials_raw)
zone = zone_labels[i]
z0, z1 = REWARD_ZONES[zone]
dist = signed_distance_to_zone(pos, z0, z1)
```

iii. CONVERSION_NOTES Step 1/Step 5 decision 6: "Reward zone location per trial derived from the scene string with the switch at trial index 30, exactly as `behavior.get_reward_zones`; verified against the measured reward-zone entry position on every trial (median error < 2 cm)." Step 4 adds the independent empirical check of the switch trial: "Empirically the first trial with the new zone = index 30 in every switch session (apparent 31/32 only when trial 30 was an omission with no zone entry)", and the resulting A/B/C balance is 0.332/0.336/0.333.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance in cm from the animal's position to the nearest edge of that trial's reward zone: negative before the zone, exactly 0 anywhere inside it, positive after it. Then discretized (see 7-c). No smoothing or other transformation.

ii.
```python
def signed_distance_to_zone(pos, zone_start, zone_end):
    """Signed distance (cm) to the nearest point of the reward zone.
    Negative before the zone, 0 inside, positive after."""
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d
```

iii. Step 5/Step 10 Check 3: this is the paper's reward-relative coordinate, expressed as a signed linear distance in cm because the Decoder Task specifies cm bins (the paper itself uses a circular reward-relative coordinate in radians; the AI lists this as a task-required deviation).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes implemented with explicit boolean masks exactly matching the instruction's edges: 0 `< −50`, 1 `[−50, −10)`, 2 `[−10, 0)`, 3 `== 0` (inside the zone), 4 `(0, 10]`, 5 `(10, 50]`, 6 `> 50` cm. Class 3 can only occur when the animal is inside the zone.

ii.
```python
def discretize_distance(d):
    """0: <-50, 1: [-50,-10), 2: [-10,0), 3: 0, 4: (0,10], 5: (10,50], 6: >50"""
    out = np.full(d.shape, 3, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```

iii. Step 5: "binned: <−50 / [−50,−10) / [−10,0) / 0 / (0,10] / (10,50] / >50" per the Decoder Task spec. Step 10 Check 2 verified on every trial of three sessions that "bin 3 occurs exactly when `zone_start <= pos <= zone_end` for the assigned zone — 0 mismatches". Resulting distribution: .256/.102/.073/.238/.021/.072/.238.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `beh['pos'][sl]` with the same slice `sl = slice(s-1, e-1)` used for the neural matrix, so it is sample-for-sample aligned by construction.

ii.
```python
sl = slice(s - 1, e - 1)          # reference trial window
ev = events[:, sl]
...
pos = beh['pos'][sl]
dist = signed_distance_to_zone(pos, z0, z1)
```

iii. Step 5: neural and behavior are already 1:1 in the NWB; Step 10 Check 2 and the `--show-processing` plots ("the discretized outputs step exactly where the continuous variable crosses each bin edge") confirm no offsets.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `processing/behavior/BehavioralTimeSeries/position` time series (cm along the 450 cm virtual track).

ii.
```python
beh = dict(pos=g('position'), ...)
...
pos = beh['pos'][sl]
```

iii. Step 2 identifies `position` as the VR position in cm; Step 9 checks the observed maximum (451.8 cm) against the paper's 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None: the raw per-trial position slice is discretized directly. Values slightly outside [0, 450] (the sample immediately before track entry is ≈ −2 cm, and the last sample can be ≈ 451 cm) are absorbed by the open end bins via `np.clip`.

ii.
```python
def discretize_position(pos):
    """5 equal bins over the 450 cm track."""
    return np.clip((pos // 90).astype(np.int64), 0, 4)
...
out = np.stack([discretize_distance(dist),
                discretize_position(pos), ...])
```

iii. Step 5 mapping: "`behavior/position` → `output[1]`, 5 equal bins of the 450 cm track". An explicit trial-level guard drops any trial containing pre-sync sentinel positions (`pos < -100`), so no `-500` values can reach the binning.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins over the 450 cm track, implemented as integer division by 90 with clipping to [0, 4]: 0 `< 90`, 1 `[90, 180)`, 2 `[180, 270)`, 3 `[270, 360)`, 4 `>= 360`.

ii.
```python
return np.clip((pos // 90).astype(np.int64), 0, 4)
```

iii. Step 5: "5 equal bins of the 450 cm track: <90 / [90,180) / [180,270) / [270,360) / >=360", following the Decoder Task spec. Step 10 Check 2 recomputed `clip(pos//90,0,4)` from the raw NWB for 9 random trials in 3 sessions: exact match. Distribution .217/.177/.231/.226/.149.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same slice as the neural data — no additional alignment.

ii.
```python
pos = beh['pos'][sl]
...
out = np.stack([..., discretize_position(pos), ...])
```

iii. As in 7-d: neural and behavioral samples share the imaging-frame time base in the NWB, and the conversion indexes both with the same `sl`.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `processing/behavior/BehavioralTimeSeries/lick` time series, which holds the cumulative lick count within each imaging frame.

ii.
```python
beh = dict(..., lick=g('lick'), ...)
...
lick = (beh['lick'][sl] > 0).astype(np.int64)
```

iii. Step 1/Step 2: "`licks` is a cumulative lick count per imaging frame; code binarizes anything >1 → 1 when computing rates" (`glmUtils.get_timeseries_data`); the Methods likewise convert remaining lick counts "to a binary vector".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarization: any frame with a lick count > 0 becomes 1, otherwise 0. In addition, whole trials whose lick trace shows the paper's sensor-error signature (>30% of frames with count > 2) are dropped from the dataset rather than having the lick values set to NaN (see 1-e). Resulting distribution: 0.777 no-lick / 0.223 lick.

ii.
```python
lick = (beh['lick'][sl] > 0).astype(np.int64)
...
lick_error[i] = (np.sum(lick_tr > 2) / max(1, len(lick_tr))) > LICK_ERR_FRAC
...
if lick_error[i]:
    drop['lick_error'] += 1       # lick is an output; cannot be NaN
    continue
```

iii. Step 5 mapping: "cumulative lick count per frame binarized (>0 → 1), as in the paper". Step 10 Check 3 difference 2 explains the trial drop: "The paper sets these lick values to NaN; NaNs are not allowed by the decoder format, and lick is an output, so the 81 affected trials (0.65%) are dropped — the same trials the paper excludes from licking analyses."

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial slice as the neural data; no shift, no smoothing (the paper's 2-sample Gaussian smoothing of licks is used only for its GLM, not here, since the output must be binary).

ii.
```python
lick = (beh['lick'][sl] > 0).astype(np.int64)
```

iii. Step 10 Check 2 recomputed `raw lick > 0` for spot-checked trials directly from the NWB: exact match.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Not from a data array but from the session's **scene name** in `/identifier`, combined with the trial index and the switch at trial 30 — i.e. a direct reproduction of `behavior.get_reward_zones`. The `reward_zone` behavior flag and `position` were used (in Step 4/Step 5 checks) only to validate the assignment, not to produce it.

ii. See 7-a (`scene_zone_labels`); the label is mapped to an integer:
```python
ZONE_TO_IDX = {'A': 0, 'B': 1, 'C': 2}
...
zone = zone_labels[i]
...
out = np.stack([..., np.full(T, ZONE_TO_IDX[zone], dtype=np.int64), ...])
```

iii. Step 4: "Reward zone coords A=[80,130], B=[200,250], C=[320,370] (code dict keys X/Y/Z); median |rzone-entry position − nominal zone start| < 2 cm (max 8.5 cm, i.e. one imaging frame of running) → Consistent, zone label/coords can be derived from scene name"; and "Switch trial: `change_trial = 30` … empirically the first trial with the new zone = index 30 in every switch session".

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regex parsing of the scene name into one or two zone labels; for switch scenes the first `min(30, ntrials)` trials get the pre-switch label and the rest the post-switch label. The label is converted to 0/1/2 and tiled across the trial's timepoints as `output[4]`. Unrecognised scene names raise an error rather than being silently defaulted.

ii.
```python
m = re.match(r'Env\d_Location([ABC])_to_([ABC])$', scene)
if m is None:
    m = re.match(r'Env\d_([ABC])_to_Env\d_([ABC])$', scene)
if m:
    z0, z1 = m.group(1), m.group(2)
    n0 = min(change_trial, ntrials)
    return np.array([z0] * n0 + [z1] * (ntrials - n0))
raise ValueError(f'unhandled scene name: {scene}')
```

iii. Step 5 decision 6 and Step 10 Check 5: "`scene_zone_labels` clips the switch index to the number of trials" for sessions with fewer than 30 trials. The resulting class balance (A 0.332 / B 0.336 / C 0.333) matches the paper's counterbalanced design.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `processing/behavior/BehavioralTimeSeries/Reward` event series (its **timestamps**) together with the `reward_zone` flag, via the same `isreward` vector used for 6-a.

ii.
```python
reward_t = b['Reward/timestamps'][:]
...
idx = np.clip(np.searchsorted(time, reward_t), 0, len(reward) - 1)
reward[idx] = 1
...
isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
```

iii. Step 1: `behavior.get_trial_types` defines `isreward` as "any(reward>0) AND any(rzone>0)". Step 2 notes `Reward` is "a sparse event series (data = reward volume 0.004 mL, timestamps of delivery)", hence the `searchsorted` mapping onto frame times.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per trial, `isreward` is computed over the reference `get_trial_types` window `[start:stop]`, and the binary value is tiled across the trial's timepoints as `output[5]`. 15.3% of trials are omissions, matching the paper's ~15%.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s, e)   # get_trial_types uses [start:stop]
    isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
...
out = np.stack([..., np.full(T, isreward[i], dtype=np.int64)]).astype(np.int64)
```

iii. Step 9 consistency table: "Omission rate — paper ~15%; data 15.34% of trials; converted 15.8% of samples → YES", with the AI explaining that the sample-weighted fraction is slightly higher because omission trials are longer. Step 12 further shows the outcome is only decodable after the reward zone (0.79 post-zone vs 0.52 pre-zone), as expected.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Neural/behavior length mismatch** (10 two-plane sessions with one extra imaging frame): every stream is truncated to the common minimum length, with a printed warning; trial start/stop indices are re-filtered to that length.
- **Pre-TTL-sync samples** (`position = -500`, `environment = -1`): a per-trial guard drops any trial containing `pos < -100` (never triggered — the first trial start is at sample ≥ 142 in every session).
- **Trials shorter than 2 bins**: dropped (never triggered).
- **NaNs in the neural signal** (possible at trial edges): trials containing any NaN are dropped (never triggered).
- **Lick-sensor errors**: 81 trials dropped (see 1-e).
- **Reward timestamps outside the behavior time base**: `np.searchsorted` result is clipped into the valid index range.
- **Sessions with only 40–60 trials** and one very long trial (3,359 bins) were investigated and kept as genuine data.
- **Unknown scene names** raise `ValueError` rather than producing a silent default.

ii.
```python
n = min([F.shape[1], Fneu.shape[1], len(time)] + [len(v) for v in beh.values()])
if F.shape[1] != n or len(time) != n:
    print(f'  {os.path.basename(path)}: truncating streams to {n} samples '
          f'(ophys {F.shape[1]}, behavior {len(time)})', flush=True)
F = F[:, :n]; Fneu = Fneu[:, :n]; time = time[:n]
beh = {k: v[:n] for k, v in beh.items()}
starts = starts[starts < n]
stops = stops[stops <= n]
ntr = min(len(starts), len(stops))
```
```python
if ev.shape[1] < 2:
    drop['too_short'] += 1
    continue
if np.any(np.isnan(ev)):
    drop['nan_events'] += 1
    continue
pos = beh['pos'][sl]
if np.any(pos < -100):            # pre-TTL-sync samples (pos = -500)
    drop['pre_sync'] += 1
    continue
```
```python
idx = np.clip(np.searchsorted(time, reward_t), 0, len(reward) - 1)
```

iii. CONVERSION_NOTES Step 9 ("One-frame length mismatch … the known 'one frame correction' that the reference alignment code also handles") and Step 10 Check 5 tabulate each edge case and its handling, with the per-session drop counters printed in `conversion_full_out.txt` so that any future trigger is visible.

## 13-a. What are the most time-consuming steps of the code?

i. Per the AI's own timing instrumentation (`t_load`, `t_dff` printed per session, cumulative elapsed printed per file): (1) reading the NWB fluorescence/neuropil arrays — I/O bound, 0.1–4 s per session; (2) the dF/F + OASIS computation — 0.3–8 s per session, dominated by the three per-trial filter passes and the OASIS call; (3) pickling the 9.63 GB output (13.8 s). The whole 152-session run took 147.6 s wall-clock with 8 worker processes (estimated ~13 min serial).

ii.
```python
t0 = time.time()
S = load_session(path)
...
t_load = time.time() - t0
t0 = time.time()
dff, events = compute_events(S['F'], S['Fneu'], starts, stops, S['frame_rate'])
t_dff = time.time() - t0
...
print(f"... load {info['t_load']:.1f}s dff {info['t_dff']:.1f}s, elapsed {time.time()-t_start:.1f}s")
...
print(f'Wrote {args.outfile} ({os.path.getsize(args.outfile)/1e9:.2f} GB) in {time.time()-t0:.1f}s; ...')
```

iii. CONVERSION_NOTES Step 7 run-time table: load "~5 min serial", dF/F + OASIS "~8 min serial", "total ~13 min serial, ~3 min with 8 workers + pickling ~10 GB". Speed-ups listed: h5py partial reads of `iscell` ROIs only (~2× less I/O), vectorised speed correlation, 8 process workers (~6–7×).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining Python loops are all over trials:
- `compute_events` contains **three** separate `for start, stop in zip(starts, stops)` passes (NaN-masking/copy, neuropil-mean + maximin baseline, smoothing + OASIS). These mirror the reference implementation and are hard to vectorise because each trial has its own baseline window, but the first pass (masking) could be replaced by a single boolean index built from the start/stop arrays, and the three passes could at least be fused into one.
- The per-trial variable loop (`isreward`, `morph`, `lick_error`) and the trial-assembly loop are separate passes over the same trials and could be fused; `np.add.reduceat`-style segment reductions would vectorise `isreward`/`morph`/`lick_error` entirely.
- Discretization (`discretize_distance`, `discretize_position`, `discretize_speed`) is called once per trial; it could be applied once to the whole session array and then sliced.
- Already vectorised: the interneuron speed correlation is a single matrix product rather than the reference's per-cell loop.

ii.
```python
for start, stop in zip(starts, stops):
    f_[:, start - 1:stop - 1] = F[:, start - 1:stop - 1]
    fneu_[:, start - 1:stop - 1] = Fneu[:, start - 1:stop - 1]
...
for start, stop in zip(starts, stops):
    sl = slice(start - 1, stop - 1)
    f_[:, sl] = f_[:, sl] + NEU_COEF * np.nanmean(fneu_[:, sl], axis=1, keepdims=True)
    flow[:, sl] = nansmooth(f_[:, sl], BASELINE_SIGMA)
    ...
for start, stop in zip(starts, stops):
    sl = slice(start - 1, stop - 1)
    dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH, axis=1)
    events[:, sl] = dcnv.oasis(...)
```
```python
Dz = D - D.mean(axis=1, keepdims=True)
speed_corr = (Dz @ spz) / denom          # vectorised instead of a per-cell loop
```

iii. CONVERSION_NOTES Step 6 lists the efficiency measures actually taken ("Correlation with speed computed as a single matrix product instead of a per-cell loop", "h5py slicing loads only iscell ROIs", "ProcessPoolExecutor (8 workers) over sessions"). The AI did not discuss vectorising the trial loops; it parallelised across sessions instead, which was sufficient to bring the full run to ~2.5 minutes.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and read **once** (there is no separate survey pass). Within a session, however:
- the trial loop is executed five times in total (three times inside `compute_events`, once for the per-trial variables, once to assemble the trials);
- `nansmooth` is applied twice to overlapping data (σ = 15 for the baseline, σ = 2 for the dF/F);
- the full `dff` array is retained after `events` is computed, and both are indexed by `keep_cells`, so two full-session (n_cells × n_frames) float arrays coexist;
- when `--signal dff` (the delivered default) the dF/F array is produced twice in effect — once as `dff`/`dff_kept` and once re-assigned into `events`.

ii.
```python
dff, events = compute_events(S['F'], S['Fneu'], starts, stops, S['frame_rate'])
...
events = events[keep_cells]
dff_kept = dff[keep_cells]
if signal == 'dff':
    events = dff_kept.astype(np.float32)
```

iii. Not discussed in CONVERSION_NOTES. The single-pass design is implicit in Step 6 ("`load_session()` reads one NWB with h5py (only the needed datasets)") and in the timing table, which reports one load per session.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
- **OASIS deconvolution is always computed** inside `compute_events`, even though the delivered dataset (`--signal dff`, the default) throws the `events` array away and stores dF/F instead. This is the single largest piece of wasted work in the conversion.
- **Unused behavior streams are read from disk**: `autoreward`, `scanning` and `trial number` are loaded into `beh` but never used anywhere.
- `TRACK_LENGTH = 450.` is defined and never used.
- `dff`/`dff_kept` is kept for the whole session even when only the interneuron correlation needs it (in the `events` variant).
- `metadata['session_info']` stores a per-session dict (including raw/kept trial counts and drop reasons) that the decoder ignores — useful provenance, but not used downstream.
- The `--show-processing` plotting path and the `--signal events` branch are present but not exercised in the delivered run.

ii.
```python
    events = np.full(F.shape, np.nan, dtype=np.float32)
    for start, stop in zip(starts, stops):
        sl = slice(start - 1, stop - 1)
        dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH, axis=1)
        events[:, sl] = dcnv.oasis(np.ascontiguousarray(dff[:, sl], dtype=np.float32),
                                   2000, TAU, frame_rate)
    return dff, events
```
```python
beh = dict(pos=g('position'), speed=g('speed'), lick=g('lick'),
           rzone=g('reward_zone'), env=g('environment'),
           autoreward=g('autoreward'), scanning=g('scanning'),
           trialnum=g('trial number'))
```
```python
TRACK_LENGTH = 450.
```

iii. Not documented in CONVERSION_NOTES. The wasted deconvolution is a side effect of the Step 12 decision to switch the delivered signal from `events` to `dff` without restructuring `compute_events`; because the full conversion runs in ~2.5 minutes with 8 workers, the cost was never large enough to be noticed.
