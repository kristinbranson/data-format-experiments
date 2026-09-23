# Decisions

*(Documentation of the decisions made by the agentic AI in `/app/convert_data.py`,
`/app/CONVERSION_NOTES.md` and `/logs/agent/trajectory.json`.)*

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every NWB file in the dataset is discovered with a single glob over `/app/data/sub-*/*.nwb`
(152 files) and sorted; there is no hard-coded subject or session list. Each file is opened once
with `pynwb.NWBHDF5IO` (no `h5py`) inside `read_session()`, which pulls (a) the scene name from
`nwb.identifier`, (b) subject / session ids from the NWB metadata, (c) the nine behavioural
`TimeSeries` plus the sparse `Reward` series from `processing['behavior'].BehavioralTimeSeries`,
(d) `Fluorescence` / `Neuropil` `RoiResponseSeries` for every imaging plane plus the
`ImageSegmentation.PlaneSegmentation` `iscell`/`planeIdx` columns, and (e) the imaging-plane rate
and location. Sessions are converted in parallel with a 16–24 worker `multiprocessing` pool
(`spawn` context, lazy `suite2p` import), and the per-session results are concatenated in file
order.

ii.
```python
def read_session(path):
    with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
        nwb = io.read()
        scene = nwb.identifier.split('/')[-1]
        subject = nwb.subject.subject_id
        session_id = nwb.session_id

        b = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
        ts = np.asarray(b['position'].timestamps[:], dtype=np.float64)
        beh = {k: np.asarray(b[k].data[:], dtype=np.float64) for k in
               ['position', 'speed', 'lick', 'reward_zone', 'environment',
                'trial number', 'trial_start', 'teleport', 'scanning']}
        reward_times = np.asarray(b['Reward'].timestamps[:], dtype=np.float64)

        oph = nwb.processing['ophys'].data_interfaces
        seg = oph['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
        iscell = np.asarray(seg['iscell'].data[:])[:, 0].astype(bool)
        plane_idx = np.asarray(seg['planeIdx'].data[:]).astype(int)
```
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
...
ctx = mp.get_context('spawn')
with ctx.Pool(min(args.nproc, len(files))) as pool:
    for k, res in enumerate(pool.imap(_worker, tasks)):
```

iii. From CONVERSION_NOTES Step 2/9: the glob finds exactly 152 files in 11 `sub-mXX/`
directories, which the AI cross-checked against the paper ("11 switch mice", "14 days/mouse
except m11 who starts on day 3" → 10×14 + 12 = 152) and against a separate metadata scan
(`cache/scan_meta.py`). The instructions mandate `pynwb`, and the AI records that the NWB
`BehavioralTimeSeries` is already the export of the authors' `vr_align_to_2P`-aligned `sess`
object, so no re-alignment of VR to imaging frames is needed.

## 1-b. How are the data split into subjects?

i. Subject identity is read from the NWB metadata field `nwb.subject.subject_id` (e.g. `'m11'`)
rather than parsed from the directory/file name. The unique set is sorted numerically and stored
in `data['subjects']`; `data['subject_idx']` indexes it per session. Result: 11 subjects
(m3, m4, m7, m11–m15, m17–m19), matching the paper's 11 switch mice.

ii.
```python
subjects = sorted({r['info']['subject'] for r in results}, key=lambda s: int(s[1:]))
data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array([subjects.index(r['info']['subject']) for r in results],
                            dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 documents `nwb.subject.subject_id` as the authoritative subject id
and notes the mapping `m<N>` ↔ the paper's internal `GCAMP<N>` animal ids, used to cross-check the
number of mice (11) and sessions per mouse against the paper.

## 1-c. How are the data split into sessions?

i. One NWB file = one session (one mouse-day). No merging or splitting of files, and no
cross-session ROI alignment is attempted. `nwb.session_id` (e.g. `'03'`) is retained as the
experiment day and stored in `metadata['session_info']`. Sessions appear in the pickle in sorted
file order (subject-major, session-minor).

ii.
```python
        subject = nwb.subject.subject_id
        session_id = nwb.session_id
...
    'session_info': [
        {k: r['info'][k] for k in
         ['subject', 'session_id', 'scene', 'n_cells', 'n_cells_iscell',
          'n_interneuron', 'n_trials_all', 'n_trials_kept', 'n_dropped_lick',
          'n_omission', 'zone_labels', 'dt']}
        for r in results],
```

iii. CONVERSION_NOTES Step 2: "`/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb` —
one NWB file per mouse-day"; Step 9 verifies 152 sessions, 14 per mouse except m11 (12, imaging
began on day 3), and that 77 of them are the paper's seven switch days across 11 mice
("n = 77 sessions, 11 mice, seven switch days").

## 1-d. How are the data split into trials?

i. Trials are delimited by the VR flags `trial_start` and `teleport`: `si = where(trial_start==1)`,
`ti = where(teleport==1)`, and trial *i* occupies the **frame window `[si[i]-1, ti[i]-1)`**. This
one-sample-earlier window is the window used throughout the authors' code
(`preprocessing.dff`, `glmUtils.get_timeseries_data` both slice `start-1 : stop-1`) and it
excludes the teleport frame, whose interpolated `position` is a meaningless jump. The
`trial number` behavioural series is loaded but not used to define boundaries. An assertion
enforces one teleport per trial start; trials whose window would fall outside the recording are
skipped (0 occur). Result: 12,216 complete trials over 152 sessions.

ii.
```python
    si = np.where(raw['trial_start'] == 1)[0]
    ti = np.where(raw['teleport'] == 1)[0]
    assert len(si) == len(ti), f'{path}: {len(si)} starts vs {len(ti)} teleports'
    # guard against a trial starting on frame 0 (window would wrap around)
    keep = (si >= 1) & (ti <= len(ts))
    n_trials_out_of_range = int((~keep).sum())
    si, ti = si[keep], ti[keep]
    assert np.all(ti > si), f'{path}: teleport before trial start'
    ntrials_all = len(si)
...
        sl = slice(s - 1, e - 1)
        act = (events if signal == 'events' else dff)[:, sl]
```

iii. CONVERSION_NOTES Step 1/5: "Trial windows in the reference code are `[trial_start-1,
teleport-1)` (`dff`, `glmUtils.get_timeseries_data` both use `start-1:stop-1`). This excludes the
teleport sample, whose `pos` is an interpolation artefact between the end of the track and −50 cm
(documented in `dff`'s docstring). I adopt the identical window." Verified in Step 10: position at
the first window sample ranges −9.1…0.7 cm and at the last sample 436.6…449.4 cm, i.e. every
window is a complete on-track lap. Trials/session 80.37 ± 6.14 matches the paper's 80.5 ± 7.4.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, in order:
1. **Lick-sensor-error trials** — the paper's criterion, ported from
   `behavior.correct_lick_sensor_error`: a trial is dropped if >30 % of its imaging frames carry a
   cumulative lick count >2. This removes exactly **81 trials**, the number the paper reports
   ("n = 81 out of 12,376 trials removed across 11 switch mice"). The reference NaNs the licks;
   the AI drops the whole trial because `lick` is a required decoder output and NaN outputs are
   not allowed.
2. **Scanning flag** — any trial containing `scanning != 1` is dropped (0 occur).
3. **Degenerate trials** — fewer than 2 samples, or any non-finite activity (0 occur).
No speed threshold is applied (see 12/13). Final: 12,135 of 12,216 trials kept.

ii.
```python
LICK_ERROR_FRAC_THR = 0.3
LICK_ERROR_COUNT_THR = 2
...
        L = lick[s:e]
        lick_error[i] = (np.sum(L > LICK_ERROR_COUNT_THR) / len(L)) > LICK_ERROR_FRAC_THR
        scan_bad[i] = np.any(scanning[s - 1:e - 1] != 1)
...
    for i, (s, e) in enumerate(zip(si, ti)):
        if lick_error[i]:
            n_dropped_lick += 1
            continue
        if scan_bad[i]:
            n_dropped_scan += 1
            continue
        sl = slice(s - 1, e - 1)
        act = (events if signal == 'events' else dff)[:, sl]
        if act.shape[1] < 2 or not np.all(np.isfinite(act)):
            n_dropped_nan += 1
            continue
```

iii. CONVERSION_NOTES Step 4: the reference code's own threshold varies (`correct_lick_sensor_error`
default 0.5, `glmUtils` 0.35), but the Methods text says ">30 % of the 0.0645 s imaging frame
samples in the trial containing a cumulative lick count >2"; the AI scanned all 152 files at
several thresholds and found 0.30 reproduces the paper's n = 81 exactly, which it treats as an
exact sanity check that both the trial definition and the criterion are right. Dropping rather
than NaN-ing is documented as a deviation forced by the decoder output spec.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces `processing['ophys']['Fluorescence']` (F) and
`processing['ophys']['Neuropil']` (Fneu), per plane, restricted at load time to ROIs with
`PlaneSegmentation.iscell[:,0] == 1`. The NWB `Deconvolved` series is explicitly **not** used.

ii.
```python
        plane_names = sorted(oph['Fluorescence'].roi_response_series.keys())
        f_list, fneu_list, roi_ids = [], [], []
        for pn in plane_names:
            rrs = oph['Fluorescence'].roi_response_series[pn]
            rid = np.asarray(rrs.rois.data[:], dtype=int)
            sel = iscell[rid]                       # only load curated cells
            roi_ids.append(rid[sel])
            f_list.append(np.asarray(rrs.data[:, :], dtype=np.float32)[:, sel].T)
            nrs = oph['Neuropil'].roi_response_series[pn]
            fneu_list.append(np.asarray(nrs.data[:, :], dtype=np.float32)[:, sel].T)
        f = np.concatenate(f_list, axis=0)
        f_neu = np.concatenate(fneu_list, axis=0)
```

iii. CONVERSION_NOTES Step 1/4: "The NWB files store raw `Fluorescence`, `Neuropil` and suite2p's
own `Deconvolved` (spks, in raw-fluorescence units, computed by suite2p from raw F, *not* the
paper's events). So the paper's `preprocessing.dff` pipeline has to be re-run on F/Fneu." The AI
confirmed empirically that `Deconvolved` is non-negative, in raw-F units (max ≈ 11,000) and has no
NaNs outside trials, i.e. it is not the paper's signal.

## 2-b. How is the `neural` data processed?

i. `dff_and_events()` is a line-for-line port of `reward_relative.preprocessing.dff` for the
configuration `neuropil_method='subtract'`, `baseline_method='maximin'`, `subtract_baseline=True`,
`neu_coef=0.7`, `deconvolve=True`, `keep_teleports=False`. Per session: mask everything outside the
`[start-1, teleport-1)` lap windows to NaN → subtract `0.7 × Fneu` → add each trial's mean neuropil
back → per-trial maximin baseline (NaN-tolerant Gaussian σ = 15 frames along time, then a
300-frame ≈ 19.4 s `minimum_filter1d` followed by a 300-frame `maximum_filter1d`) →
`dF/F = (F − baseline)/|baseline|` → 2-sample (≈ 0.129 s) Gaussian smoothing → OASIS deconvolution
(`suite2p.extraction.dcnv.oasis`, batch 2000, `tau = 0.7`, `fs = 1/dt`). Planes are pooled into one
CA1 population.

**Two choices differ from the authors' pipeline.** (a) The array actually written to
`converted_data.pkl` is the **smoothed dF/F**, not the OASIS "events" the paper analyses; both are
computed and `--signal events` reproduces the alternative. (b) `keep_teleports` is hard-coded to
`False` for every session, whereas `utilities.multi_anim_sess` takes it per animal/day from
`teleport_metadata.teleport_sessions` (≈ 47 of the 152 sessions here were imaged through the
teleport and would get `keep_teleports=True` in the authors' pipeline). The AI read
`teleport_metadata.py` (trajectory step 33) and noticed in its diagnostic plots that m17 day 8 was
imaged through the teleport, but did not implement the per-session switch.

ii.
```python
    f_ = np.full(f.shape, np.nan, dtype=np.float32)
    f_neu_ = np.full(f_neu.shape, np.nan, dtype=np.float32)
    # keep_teleports=False -> only on-track samples, window [start-1, stop-1)
    slices = [slice(s - 1, t - 1) for s, t in zip(trial_starts, teleports)]
    for sl in slices:
        f_[:, sl] = f[:, sl]
        f_neu_[:, sl] = f_neu[:, sl]
    nanmask = ~np.isnan(f_[0, :])
    # neuropil correction
    f_ -= neu_coef * f_neu_
    for sl in slices:
        # add the per-trial neuropil mean back in so dF/F is not divided by a small number
        f_[:, sl] = f_[:, sl] + neu_coef * np.nanmean(f_neu_[:, sl], axis=1, keepdims=True)
        # maximin baseline: smooth, 20 s minimum filter, then same-window maximum filter
        base = nansmooth(f_[:, sl], [0, BASELINE_SMOOTH_SIG])
        base = minimum_filter1d(base, BASELINE_WIN, axis=-1)
        base = maximum_filter1d(base, BASELINE_WIN, axis=-1)
        flow[:, sl] = base
    dff = np.full(f_.shape, np.nan, dtype=np.float32)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    for sl in slices:
        smoothed = nansmooth(dff[:, sl], DFF_SMOOTH_SIG, axis=1)
        dff[:, sl] = smoothed
        spks[:, sl] = dcnv.oasis(np.ascontiguousarray(smoothed, dtype=np.float32),
                                 OASIS_BATCH, tau, frame_rate)
```
```python
    ap.add_argument('--signal', choices=['dff', 'events'], default='dff')
...
        act = (events if signal == 'events' else dff)[:, sl]
```

iii. CONVERSION_NOTES Step 3 quotes the Methods for every parameter, and Step 10 "Check 3" calls
the port "line-for-line". The choice of dF/F over events is justified in Step 8 by a head-to-head
`train_decoder.py` run on two sample sessions: dF/F gave higher validation balanced accuracy on 5
of 6 outputs (e.g. position 0.685 vs 0.604, distance 0.578 vs 0.475), with the argument that
"OASIS deconvolution deliberately strips the calcium indicator's temporal integration, which is
exactly the information a *per-timepoint* linear decoder relies on (the paper's own decoder
integrates over a whole trial set, not per frame)". `tau = 0.7` is justified as the reference
function's default because `sess.s2p_ops['tau']` is not in the NWB export. `keep_teleports=False`
is not separately justified beyond being `default_dff_method`'s value and matching the Methods
phrase "baseline fluorescence was calculated within each trial independently".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) suite2p manual curation: only ROIs with `iscell[:,0] == 1`
are ever loaded (138,678 of 260,091 ROIs). (2) Putative interneurons: cells whose dF/F correlates
with running speed at Pearson r > 0.5 over all in-trial samples are removed (409 cells, 0.29 %
overall; per-session 0.35 ± 0.61 %). The correlation is computed vectorised over all cells at once.
Final: 138,269 neurons, 154–2,323 per session.

ii.
```python
INTERNEURON_SPEED_CORR_THR = 0.5
...
    valid = ~np.isnan(dff[0, :])          # frames inside a trial window
    sp_v = speed[valid].astype(np.float32)
    d_v = dff[:, valid]
    sp_c = sp_v - sp_v.mean()
    d_c = d_v - d_v.mean(axis=1, keepdims=True)
    denom = np.sqrt((d_c ** 2).sum(axis=1) * (sp_c ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        r_speed = (d_c @ sp_c) / denom
    r_speed = np.nan_to_num(r_speed, nan=0.0)
    keep_cells = r_speed <= INTERNEURON_SPEED_CORR_THR
    n_interneuron = int((~keep_cells).sum())
    dff = dff[keep_cells]
    events = events[keep_cells]
```

iii. CONVERSION_NOTES Step 3: "1. suite2p manual curation → keep ROIs with `iscell[:,0] == 1`
(stored in the NWB PlaneSegmentation). 2. Exclude putative interneurons: Pearson r(dF/F, running
speed) > 0.5 (expected ≈ 0.42 ± 0.85 % of cells)." Step 9 compares the achieved 0.35 ± 0.61 % with
the paper's 0.42 ± 0.85 % and the per-session minimum of 155 `iscell` cells with the paper's
"155–2172 putative pyramidal neurons per session"; the 2,341 maximum is investigated and attributed
to the paper pooling differently / including the 3 fixed-condition mice absent from DANDI.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is the start of the trial. Behaviour in the NWB export is already
interpolated onto the imaging-frame grid, so neural and behavioural streams share one index axis
and no resampling or cross-correlation alignment is needed — the same `slice(s-1, e-1)` is applied
to every stream. Time zero is set at the `trial_start` frame `s`, so the first stored sample sits
at −1 frame (−0.0645 s); this is recorded as `metadata['off_start'] = -0.0645`.
`metadata['off_end']` is `None` because trials end at a variable-latency teleport, with the
duration distribution given in a companion `off_end_note` field.

ii.
```python
        sl = slice(s - 1, e - 1)
        act = (events if signal == 'events' else dff)[:, sl]
        ...
        p = pos[sl]
        sp_t = speed[sl]
        lk = lick[sl]
        inp[0] = ts[sl] - ts[s]                 # t = 0 at the trial_start frame
...
        'temporal_alignment_event': (
            'start of trial: the VR "trial_start" flag, i.e. entry into the linear track at '
            'position 0 cm'),
        'off_start': float(-np.median(dts)),
        'off_end': None,
```

iii. CONVERSION_NOTES Step 5 and Step 7: alignment was checked visually in panel 9 of
`processing_m11_ses-03.png` / `processing_m17_ses-08.png` — the neural raster sorted by peak time
with position overlaid shows "sequential activation follow[ing] the animal's trajectory, so there
is no temporal shift" — and numerically in `cache/sanity_checks.py`
(`input[0] == ts[window] - ts[trial_start]`, PASS).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **No rebinning.** Data are kept at the native imaging frame rate. The bin size is taken from the
behavioural timestamps as `dt = median(diff(ts))` = **64.4836 ms** (15.5078 Hz), identical in all
152 sessions, and is also the `frame_rate` handed to OASIS. The AI explicitly notes that the `rate`
attribute on the `RoiResponseSeries` reads 31.015625 Hz for the two 2-plane mice (m17, m18) but the
true per-plane/per-sample period is still 64.4836 ms, verified against the behaviour timestamps.

ii.
```python
    # The frame period is the sampling interval of the (already 2P-aligned) VR data.
    dt = float(np.median(np.diff(ts)))
    frame_rate = 1.0 / dt
...
    dts = np.array([r['info']['dt'] for r in results])
    ...
        'time_bin_size': float(np.median(dts) * 1000.0),
...
    print(f'time bin            : {np.median(dts)*1000:.4f} ms '
          f'(all sessions equal: {np.allclose(dts, dts[0])})')
```

iii. CONVERSION_NOTES Step 2/4/9: "All behavioral and neural time series were sampled at ~15.5 Hz,
the imaging frame rate" (Methods) and the reference GLM code `get_timeseries_data` also works at
native frames, so no binning is applied. The run log confirms `all sessions equal: True`.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` vector of the `position` behavioural `TimeSeries`. All behavioural series in the
file share a single timestamp vector (checked), so any of them would give the same answer.

ii.
```python
        ts = np.asarray(b['position'].timestamps[:], dtype=np.float64)
```

iii. CONVERSION_NOTES Step 2 records that the behavioural series are "all on a common timestamp
vector (one sample per imaging frame, dt = 64.4836 ms)". `cache/sanity_checks.py` re-derives the
input from the raw file independently and confirms equality.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtraction of the timestamp at the `trial_start` frame (not at the first stored sample), so the
series runs from −0.0645 s to the trial's duration. Stored as a time-varying float32 row.

ii.
```python
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = ts[sl] - ts[s]                 # t = 0 at the trial_start frame
```

iii. CONVERSION_NOTES Step 5: "`time_from_trial_start = timestamps − timestamps[si[i]]`, so t = 0
at the `trial_start` sample and the first sample of the window is at −64.5 ms. `off_start =
−0.0645 s`." Observed range over the whole dataset: [−0.0645, 216.47] s, with the 216 s maximum
traced to a single trial in which the mouse stopped running.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Trivially — the timestamps *are* the neural sample grid, and the identical `slice(s-1, e-1)` is
used for both. The only alignment work is a defensive length reconciliation: 10 of the 152 files
(all from the 2-plane mice) carry one more imaging frame than 2P-aligned VR samples, and all streams
are trimmed to the shorter length.

ii.
```python
    # A handful of the 2-plane files (10 of 152) carry one more imaging frame than the
    # 2P-aligned VR data. The VR alignment defines the common sample grid, so trim the
    # trailing frames of every stream to the shorter length.
    n = min(len(ts), f.shape[1])
    n_trimmed = max(len(ts), f.shape[1]) - n
    ts = ts[:n]
    beh = {k: v[:n] for k, v in beh.items()}
    f, f_neu = f[:, :n], f_neu[:, :n]
```

iii. CONVERSION_NOTES Step 6/10: the first full run crashed on `sub-m17_ses-04` (22,790 behaviour
samples vs 22,791 ophys frames); the AI investigated, found 10 affected files (all 2-plane), and
chose the VR grid as canonical because "the NWB files are the export of exactly that `sess`" in
which VR was interpolated onto imaging frames.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioural `TimeSeries` (the authors' "morph" variable: 0 = ENV1, 1 = ENV2).

ii.
```python
    morph = raw['environment']
...
        mvals = np.unique(morph[s:e])
        trial_morph[i] = int(mvals[0]) if len(mvals) == 1 else int(np.round(np.median(morph[s:e])))
```

iii. CONVERSION_NOTES Step 1/5 maps this to `behavior.get_trial_types`, which derives the per-trial
`morph` from the same variable. Step 2 notes the variable can be −1 "before TTL sync"; the final
conversion reports an environment range of exactly [0, 1], i.e. no unsynced samples survive inside
trial windows. Step 9 checks the ENV2 trial fraction (48.93 %) against the experiment design
(ENV2 from day 8, i.e. about half the days) and the raw scan (49.03 %).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. It is reduced to one value per trial (the unique value within the trial window; the rounded
median if the trial is not constant, which never occurs), cast to int, and broadcast across all
timepoints of the trial so that every input is a `(4, T)` time-varying array.

ii.
```python
        mvals = np.unique(morph[s:e])
        trial_morph[i] = int(mvals[0]) if len(mvals) == 1 else int(np.round(np.median(morph[s:e])))
...
        inp[1] = trial_morph[i]
```

iii. CONVERSION_NOTES Step 5 "Key Decisions" #3: "All inputs/outputs stored as time-varying
`(d, T)` arrays, per-trial quantities broadcast. Costs negligible memory next to the neural array
and guarantees dimension consistency." `cache/sanity_checks.py` asserts
`input[1] == unique(environment)` within each checked trial (PASS).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The within-session trial index produced by enumerating the `trial_start`/`teleport` pairs — i.e.
the position of the trial in the *unfiltered* list of trials, so the numbering does not shift when
lick-error trials are dropped. The NWB `trial number` series is loaded but deliberately not used.

ii.
```python
    for i, (s, e) in enumerate(zip(si, ti)):
        ...
        inp[2] = i
```

iii. CONVERSION_NOTES Step 5 maps this to the reference's `glmUtils.get_timeseries_data`
`trial_ids`, and "Key Decisions" #8 states: "Per-trial `isreward` / `morph` / reward zone are
computed over **all** trials before trial filtering, so `previous_trial_rewarded` is never
corrupted by a dropped trial" — the same enumeration supplies the trial number. Observed range
[0, 99]. `cache/sanity_checks.py` asserts `input[2] == original trial index` (PASS).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the 0-based integer index across the trial's timepoints (stored as
float32).

ii.
```python
        inp = np.empty((4, T), dtype=np.float32)
        ...
        inp[2] = i
```

iii. As above; no normalisation is applied, and the value restarts at 0 in every session, which is
consistent with the reward-zone switch being defined at trial 30 of each session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The sparse `Reward` `TimeSeries` (its `timestamps`, mapped onto the behavioural grid with
`np.searchsorted`) combined with the `reward_zone` flag — exactly the reference's
`behavior.get_trial_types` definition `isreward = any(reward delivered) AND any(reward-zone flag)`.
`previous_trial_rewarded` is then the preceding trial's `isreward`.

ii.
```python
    reward_idx = np.searchsorted(ts, raw['reward_times'])
...
    for i, (s, e) in enumerate(zip(si, ti)):
        has_reward = np.any((reward_idx >= s) & (reward_idx < e))
        isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
```

iii. CONVERSION_NOTES Step 4 documents that the AI checked whether the `rzone` conjunct matters:
"the `reward_zone` flag is 0 on exactly the trials with no reward delivery… so
`isreward = any(reward) & any(rzone)` reduces to `any(reward)`. Consistent with the reference; used
as-is." The resulting omission rate, 15.34 %, matches the paper's "~15 % of trials".

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A one-step shift of the per-trial `isreward` vector, computed over **all** trials before any
trial filtering, so dropping a lick-error trial never makes a trial inherit the wrong predecessor.
The first trial of each session gets 0 (the outcome is genuinely unknown; imaging is preceded by
un-imaged warm-up trials). The value is broadcast across the trial's timepoints.

ii.
```python
    # previous-trial outcome; index 0 is undefined -> 0 (see CONVERSION_NOTES Step 5)
    prev_reward = np.zeros(ntrials_all, dtype=np.int64)
    prev_reward[1:] = isreward[:-1]
...
        inp[3] = prev_reward[i]
```

iii. CONVERSION_NOTES Step 5 "Key Decisions" #6 and #8. The AI also quantified in Step 12 that this
input leaks essentially nothing about the current trial:
"P(rewarded | prev rewarded) = 0.850 vs P(rewarded | prev omitted) = 0.827, phi = 0.023".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` behavioural series plus the active reward zone for that trial. The zone comes
from the **scene name** in `nwb.identifier` (e.g. `Env1_LocationB_to_A`, `Env1_C_to_Env2_B`)
decoded by `zone_labels_from_scene()`, a port of `reward_relative.behavior.get_reward_zones`: fixed
scenes give one zone for all trials; switch scenes give the first zone for trials 0–29 and the
second from trial 30 on. Zone coordinates are the paper's
`reward_zone_dict` (A = 80–130, B = 200–250, C = 320–370 cm). The `reward_zone` flag is used only
as an independent cross-check, not as the source.

ii.
```python
REWARD_ZONE_DICT = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_ORDER = ['A', 'B', 'C']
CHANGE_TRIAL = 30            # "Each switch occurred after 30 trials"

def zone_labels_from_scene(scene, ntrials, change_trial=CHANGE_TRIAL):
    for z in ZONE_ORDER:
        if scene.endswith('Location' + z):
            return np.array([z] * ntrials)
    first = None
    for z in ZONE_ORDER:
        if z + '_to' in scene:
            first = z
    last = scene[-1]
    if first is None or last not in ZONE_ORDER:
        raise ValueError(f'Unrecognised scene name: {scene}')
    n0 = min(change_trial, ntrials)
    return np.array([first] * n0 + [last] * (ntrials - n0))
...
    zone_lab = zone_labels_from_scene(raw['scene'], ntrials_all)
    zone_start = np.array([REWARD_ZONE_DICT[z][0] for z in zone_lab])
    zone_end = np.array([REWARD_ZONE_DICT[z][1] for z in zone_lab])
```

iii. CONVERSION_NOTES Step 4: "Cross-checked: the measured zone-entry position agreed with the
scene-derived zone on **all 10,394** trials where the flag was set → scene-based assignment and
`change_trial=30` are correct for every session." Step 10 adds a further cross-check that on
rewarded trials the reward was delivered *inside* the assigned zone (PASS). Using the scene rather
than the flag also covers the ~15 % omission trials, on which the flag is never raised.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. `signed_distance_to_zone()` computes, per timepoint, the signed distance in cm to the nearest
edge of the trial's 50 cm zone: 0 while inside the zone, `pos − zone_start` (negative) before it,
`pos − zone_end` (positive) after it. The continuous distance is then discretised (7-c).

ii.
```python
def signed_distance_to_zone(pos, zone_start, zone_end):
    """Signed distance (cm) from `pos` to the nearest point of [zone_start, zone_end].
    0 inside the zone, negative before it, positive after it.
    """
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d
...
        out[0] = discretize_distance(signed_distance_to_zone(p, zone_start[i], zone_end[i]))
```

iii. CONVERSION_NOTES Step 5: this is the decoder task's "distance to any location in the reward
zone", so distance is zero anywhere inside the zone; the zone bounds come from the paper's
`reward_zone_dict`. Verified in `cache/sanity_checks.py` by recomputing the class from position and
the scene-derived zone (PASS) and visually in panel 7 of the processing figures.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes by explicit boolean masking rather than `np.digitize`, precisely so that the
"exactly 0 cm = in the reward zone" class 3 is separated from "just above 0 cm" class 4:
`d < −50 → 0`; `−50 ≤ d < −10 → 1`; `−10 ≤ d < 0 → 2`; `d == 0 → 3`; `0 < d ≤ 10 → 4`;
`10 < d ≤ 50 → 5`; `d > 50 → 6`.

ii.
```python
DIST_EDGES = (-50.0, -10.0, 0.0, 10.0, 50.0)

def discretize_distance(d):
    """7 classes, exactly as specified by the decoder task."""
    out = np.full(d.shape, 3, dtype=np.int64)     # d == 0 -> in the reward zone
    out[d < 0] = 2                                # -10 <= d < 0
    out[d < DIST_EDGES[1]] = 1                    # -50 <= d < -10
    out[d < DIST_EDGES[0]] = 0                    # d < -50
    out[d > 0] = 4                                # 0 < d <= 10
    out[d > DIST_EDGES[3]] = 5                    # 10 < d <= 50
    out[d > DIST_EDGES[4]] = 6                    # d > 50
    return out
```

iii. CONVERSION_NOTES Step 5 restates the decoder-task binning rules verbatim and the resulting
distribution over all 2,576,026 timepoints is reported (0.256, 0.102, 0.073, 0.238, 0.021, 0.072,
0.238) — class 3 at 0.238 is the fraction of in-trial time spent inside the 50 cm zone, which is
sensible for a 450 cm track traversed at variable speed with pausing at the reward.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No separate alignment: `position` is on the same already-2P-aligned sample grid, and the same
`slice(s-1, e-1)` used for the neural matrix is used for position.

ii.
```python
        sl = slice(s - 1, e - 1)
        act = (events if signal == 'events' else dff)[:, sl]
        ...
        p = pos[sl]
        ...
        out[0] = discretize_distance(signed_distance_to_zone(p, zone_start[i], zone_end[i]))
```

iii. See 2-d/3-c: the NWB behavioural series are the authors' `vr_align_to_2P` output, one sample
per imaging frame; alignment was verified by the sanity-check script and the neural-raster-vs-
position diagnostic panel.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural `TimeSeries` (cm along the 450 cm virtual track), used directly.

ii.
```python
        beh = {k: np.asarray(b[k].data[:], dtype=np.float64) for k in
               ['position', 'speed', 'lick', 'reward_zone', 'environment',
                'trial number', 'trial_start', 'teleport', 'scanning']}
...
    pos = raw['position']
```

iii. CONVERSION_NOTES Step 2 lists `position` (cm) as one of the behavioural series; no derived or
alternative position variable exists in the file.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing the trial window and discretising. Notably, the `[start-1, teleport-1)`
window excludes the teleport frame whose interpolated position is an artefact, so no clipping or
unwrapping is needed; positions inside trials span roughly −9.1 to 449.9 cm.

ii.
```python
        p = pos[sl]
        ...
        out[1] = np.digitize(p, POS_EDGES)
```

iii. CONVERSION_NOTES Step 10 "Edge cases": "teleport frame carries an interpolated, meaningless
`pos` (jumps to ~200 then −50) — excluded by the `[start-1, stop-1)` window", verified by the
last-window-sample position range 436.6–449.4 cm; and "`position` slightly negative at the first
window sample — kept (range −9.1 to 0.7 cm); falls in position class 0 … exactly as a position just
before 0 cm should".

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with edges (90, 180, 270, 360), i.e. five 90 cm bins spanning the 450 cm track;
`<90 → 0`, `[90,180) → 1`, `[180,270) → 2`, `[270,360) → 3`, `≥360 → 4`. The first and last bins
are open, absorbing the few samples marginally outside [0, 450].

ii.
```python
POS_EDGES = (90.0, 180.0, 270.0, 360.0)
...
        out[1] = np.digitize(p, POS_EDGES)
```

iii. CONVERSION_NOTES Step 5 restates the decoder-task rule ("5 equal-sized bins spanning the
450 cm track") and Step 9 reports the resulting distribution (0.217, 0.177, 0.231, 0.226, 0.149),
noting the last bin is smaller because mice run fastest at the end of the track. Verified by
`cache/sanity_checks.py` (`position class == digitize(pos,[90,180,270,360])`, PASS) and by the
discretisation overlay in panel 7 of the processing figures.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same as 7-d — identical index window, no resampling.

ii.
```python
        sl = slice(s - 1, e - 1)
        act = (events if signal == 'events' else dff)[:, sl]
        p = pos[sl]
```

iii. Same justification as 2-d/3-c/7-d.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural `TimeSeries` — a cumulative lick count per imaging frame (observed
maximum 8 per frame).

ii.
```python
    lick = raw['lick']
...
        lk = lick[sl]
```

iii. CONVERSION_NOTES Step 2 characterises `lick` as "cumulative count/frame" and Step 1 identifies
`behavior.lickrate` as the reference function that consumes it.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation at >0, matching the reference's `behavior.lickrate` (`licks[licks>0] = 1`). No
smoothing or rate conversion is applied. Separately, the paper's lick-sensor-error criterion is used
to drop 81 whole trials (see 1-e) rather than NaN-ing their licks, because `lick` is a required
output.

ii.
```python
        out[3] = (lk > 0).astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 maps this to `behavior.lickrate` and the decoder task's binary
specification. Resulting distribution 0.777 / 0.223. Verified in `cache/sanity_checks.py`
(`lick == (lick > 0)`, PASS).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same index window as the neural data; no shift. Note that the `Reward` series (used for
`reward_outcome`) has its own sparse timestamps and *is* explicitly mapped onto the common grid,
whereas `lick` is already on it.

ii.
```python
        lk = lick[sl]
        ...
        out[3] = (lk > 0).astype(np.int64)
```

iii. Same justification as 2-d/3-c; the reward/lick coincidence was also checked visually
(panel 6: "rewards always land where licking bursts occur").

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The scene name parsed from `nwb.identifier`, via the same `zone_labels_from_scene()` port of
`behavior.get_reward_zones` described in 7-a — not from the `reward_zone` flag.

ii.
```python
        scene = nwb.identifier.split('/')[-1]
...
    zone_lab = zone_labels_from_scene(raw['scene'], ntrials_all)
    zone_code = np.array([ZONE_ORDER.index(z) for z in zone_lab], dtype=np.int64)
```

iii. See 7-a: verified against the measured zone-entry position on all 10,394 trials that raise the
flag (0 disagreements), and independently against reward-delivery position on rewarded trials. The
resulting balance across the dataset is A = 4,186 / B = 4,010 / C = 4,020 trials, consistent with
the paper's counterbalanced design.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Label → integer code A = 0, B = 1, C = 2, broadcast across the trial's timepoints. On switch
sessions the label changes at trial index 30 (clipped to `min(30, ntrials)`); on fixed sessions it
is constant.

ii.
```python
    n0 = min(change_trial, ntrials)
    return np.array([first] * n0 + [last] * (ntrials - n0))
...
    zone_code = np.array([ZONE_ORDER.index(z) for z in zone_lab], dtype=np.int64)
...
        out[4] = zone_code[i]
```

iii. CONVERSION_NOTES Step 3 quotes the Methods ("Each switch occurred after 30 trials") and the
reference `get_reward_zones(change_trial=30)`; Step 10 notes the `min(30, ntrials)` clip exists
for short sessions although none occur (minimum 41 trials). Final class fractions
0.332 / 0.336 / 0.333 of timepoints.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward` `TimeSeries` timestamps, AND-ed with the `reward_zone` flag — the reference
`behavior.get_trial_types` definition. Reward timestamps are mapped to the behaviour/imaging grid
with `np.searchsorted`.

ii.
```python
        reward_times = np.asarray(b['Reward'].timestamps[:], dtype=np.float64)
...
    reward_idx = np.searchsorted(ts, raw['reward_times'])
...
        has_reward = np.any((reward_idx >= s) & (reward_idx < e))
        isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
```

iii. CONVERSION_NOTES Step 1/4/5, as for 6-a: this reproduces `get_trial_types`, and the AI verified
that the conjunct is redundant in this dataset and that the resulting omission rate (15.34 % raw,
15.36 % of kept trials) matches the paper's ~15 %.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. One binary value per trial (1 = rewarded, 0 = omitted), computed over all trials before
filtering, then broadcast across the trial's timepoints so the output is time-varying with a
constant value.

ii.
```python
    isreward = np.zeros(ntrials_all, dtype=np.int64)
    ...
        isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
...
        out[5] = isreward[i]
```

iii. CONVERSION_NOTES Step 5 "Key Decisions" #3 and #8. Step 12 goes further and investigates the
modest decoder accuracy for this output (0.603): an independent per-session PCA + balanced logistic
regression (`cache/diagnose_reward_outcome.py`) reproduces 0.617 and confirms the predicted split
(0.565 before the zone, 0.652 from zone entry onward), i.e. the ceiling is in the data — reward is
delivered on the first lick inside the zone, so pre-zone timepoints are at chance by construction.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Frame-count mismatch (10 of 152 files, all from 2-plane mice m17/m18):** one extra imaging
  frame relative to the 2P-aligned VR samples; all streams are trimmed to the shorter length, with
  the count reported in the summary (`frames trimmed: 10`).
- **Lick-sensor errors:** 81 trials dropped (paper's criterion) instead of NaN-ing licks.
- **Trials outside the recording:** a trial starting at frame 0 or ending past the last frame would
  make the `[s-1, e-1)` window wrap; those are filtered out and counted (0 occur).
- **Degenerate/non-finite trials:** `T < 2` or any non-finite activity → dropped and counted
  (0 occur).
- **`scanning != 1` inside a trial:** dropped and counted (0 occur).
- **Missing `tau`:** `sess.s2p_ops['tau']` is not in the NWB export, so the reference function's
  default 0.7 is used and the assumption is documented.
- **`autoreward` all zeros:** noted as unpopulated in the NWB export and not used.
- **NaN-safe smoothing:** `nansmooth` (port of the authors' utility) prevents the inter-trial NaNs
  from propagating into the baseline.
- **Unresolved-but-documented:** the DANDI export has 12,216 complete trials vs the paper's 12,376;
  the AI re-counted `trial_start`/`teleport` flags in all 152 files, found them equal in every file,
  and concluded the 1.3 % shortfall is in the published files, not the conversion — deliberately
  *not* "fixed".
- Two known deviations are documented as forced by the decoder spec: no 2 cm/s speed mask (it would
  delete speed class 0), and trials dropped rather than NaN-ed.
- **Not handled:** the per-animal/per-day `keep_teleports=True` setting from
  `teleport_metadata.py` (see 2-b).

ii.
```python
    n = min(len(ts), f.shape[1])
    n_trimmed = max(len(ts), f.shape[1]) - n
    ts = ts[:n]
    beh = {k: v[:n] for k, v in beh.items()}
    f, f_neu = f[:, :n], f_neu[:, :n]
...
    keep = (si >= 1) & (ti <= len(ts))
    n_trials_out_of_range = int((~keep).sum())
...
        if act.shape[1] < 2 or not np.all(np.isfinite(act)):
            n_dropped_nan += 1
            continue
```
```python
def nansmooth(a, sig, axis=None):
    nan_inds = np.isnan(a)
    a_nanless = np.where(nan_inds, 0.0, a)
    one = np.ones(a.shape, dtype=a.dtype)
    one[nan_inds] = 0.001
```

iii. CONVERSION_NOTES Step 6 "Edge cases handled" and Step 10 "Check 5: Edge cases" enumerate each
case with the verification that fired it. The frame-mismatch fix came from an actual crash on
`sub-m17_ses-04`, after which the AI re-ran the full conversion and every check. Every drop is
counted and printed rather than silently swallowed, and `cache/sanity_checks.py` asserts that "the
dropped trials are exactly the lick-sensor-error trials".

## 13-a. What are the most time-consuming steps of the code?

i. Measured per session (`t_read`, `t_dff`, `t_total` are instrumented and printed for all 152
sessions):
1. **dF/F + OASIS deconvolution** — the dominant cost, ~3–4 s for 371 cells and ~10–16 s for a
   typical ~900-cell session, driven by the per-trial Gaussian / minimum / maximum filters and the
   OASIS loop.
2. **Reading `Fluorescence` + `Neuropil` out of the NWB file** — ~0.2–3.5 s per session (I/O bound;
   87 GB of source data).
3. **Pickling** the 9.63 GB result — 14 s.
4. Task-variable extraction and assembly — ~0.2 s, negligible.
Total wall clock: 2.4 min with 24 workers (a single-process run was estimated at ~40 min).

ii.
```python
    t0 = time.time()
    raw = read_session(path)
    t_read = time.time() - t0
...
    t1 = time.time()
    dff, events = dff_and_events(raw['f'], raw['f_neu'], si, ti, frame_rate)
    t_dff = time.time() - t1
...
              f"(dropped lick={i['n_dropped_lick']} scan={i['n_dropped_scan']} "
              f"nan={i['n_dropped_nan']}) | read {i['t_read']:.1f}s "
              f"dff {i['t_dff']:.1f}s total {i['t_total']:.1f}s | "
```

iii. CONVERSION_NOTES Step 6/7 record the profile and the three speed-ups the AI added in response:
float32 throughout (~1.3× on the filter/deconvolution stages, verified to change dF/F by < 1e-4 vs
float64), `iscell` subsetting before dF/F (~1.9×, since only 53 % of ROIs pass), and 24-way session
parallelism (~17×). "Well under the 15 min budget, so no further optimisation was needed."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining loops, and what could be done:
- The three per-trial `for sl in slices:` loops inside `dff_and_events` (neuropil-mean add-back,
  maximin baseline, smoothing + OASIS). The baseline and smoothing loops could be replaced by a
  single call on the full NaN-masked array if the filters were made gap-aware (or by ragged
  padding); as written, each ~80-trial session performs ~240 small filter calls.
- The per-trial task-variable loop (`for i, (s, e) in enumerate(zip(si, ti))`), which computes
  `isreward`, `morph`, `lick_error` and `scan_bad` one trial at a time; these are all
  segment reductions that `np.add.reduceat` / `np.maximum.reduceat` could do in one pass.
- The per-trial assembly loop, where `digitize` / `discretize_distance` are applied to each trial
  slice; these are pointwise and could be computed once on the whole session array and then sliced.
- The AI *did* vectorise the one loop the reference code leaves scalar: the per-cell
  speed-correlation for interneuron detection, replaced by a single matrix–vector product.

ii.
```python
    # (already vectorised) interneuron detection over all cells at once
    sp_c = sp_v - sp_v.mean()
    d_c = d_v - d_v.mean(axis=1, keepdims=True)
    denom = np.sqrt((d_c ** 2).sum(axis=1) * (sp_c ** 2).sum())
    r_speed = (d_c @ sp_c) / denom
```
```python
    # (not vectorised) per-trial filter loop
    for sl in slices:
        f_[:, sl] = f_[:, sl] + neu_coef * np.nanmean(f_neu_[:, sl], axis=1, keepdims=True)
        base = nansmooth(f_[:, sl], [0, BASELINE_SMOOTH_SIG])
        base = minimum_filter1d(base, BASELINE_WIN, axis=-1)
        base = maximum_filter1d(base, BASELINE_WIN, axis=-1)
        flow[:, sl] = base
```

iii. CONVERSION_NOTES Step 6 lists "Vectorised the interneuron speed-correlation over all cells at
once" as a deliberate speed-up. The per-trial loops are inherent to the reference algorithm — the
baseline is defined *within each trial independently*, so they cannot be merged without changing the
result — and the AI chose session-level parallelism instead, which reduced total runtime to 2.4 min
and made further vectorisation unnecessary.

## 13-c. What processing does the code repeat multiple times?

i. Within `convert_data.py` the repetition is modest:
- `signed_distance_to_zone` is recomputed inside `plot_processing` for the plotted trial, duplicating
  work already done in the assembly loop.
- The full `Fluorescence`/`Neuropil` arrays are materialised and *then* subset by `iscell`
  (`np.asarray(rrs.data[:, :])[:, sel]`), so the rejected ROIs are still read from disk even though
  the comment says "only load curated cells".
- Per-trial `np.nanmean`/filters are invoked once per trial rather than once per session (13-b).
Across the *workflow*, the whole 87 GB dataset is read several times: `cache/scan_meta.py`,
`cache/scan_behavior.py`, the sample conversion, the full conversion, and
`cache/sanity_checks.py` (which deliberately re-reads and re-derives everything). Unlike the
reference solution, however, the conversion script itself needs no preliminary survey pass — the
reward zone comes from the scene name, so a single pass suffices.

ii.
```python
            f_list.append(np.asarray(rrs.data[:, :], dtype=np.float32)[:, sel].T)
```
```python
    # inside plot_processing -- recomputed, already available in output_trials[tr][0]
    d = signed_distance_to_zone(pos[sl], zone_start[orig], zone_end[orig])
```

iii. CONVERSION_NOTES / `cache/README_CACHE.md` frame the repeated whole-dataset reads as
intentional: the scan scripts "derive the expected statistics" used as sanity checks before
conversion, and `sanity_checks.py` is deliberately **independent** ("it does not import
`convert_data.py`; the dF/F is re-derived directly from the Methods text in a separately written
function"), so the duplication is the price of an independent verification.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest case, which the AI did **not** flag in its notes: `dff_and_events()` always runs the
OASIS deconvolution for every trial of every session, but the default `--signal dff` throws the
`events` array away (it survives only as two summary numbers, `events_frac_zero`, and in the
diagnostic plots). Deconvolution is a substantial share of the dominant dF/F cost, so roughly a
third to a half of the pipeline's compute is discarded on the full run. Smaller cases:
- `raw['f']`/`raw['f_neu']` load every ROI's trace before the `iscell` subset is applied.
- `plane_of_cell` is computed and returned from `convert_session` but never used (all neurons get
  `brain_region_idx = 0` / 'CA1').
- `beh['trial number']` is loaded but never used (trial boundaries come from
  `trial_start`/`teleport`).
- `info` accumulates several diagnostics (`dff_median`, `dff_p99`, `events_frac_zero`) that are
  computed for every session but not written into the pickle.
None of these affect correctness, and the whole run still finishes in 2.4 min.

ii.
```python
    # events are always computed ...
    for sl in slices:
        smoothed = nansmooth(dff[:, sl], DFF_SMOOTH_SIG, axis=1)
        dff[:, sl] = smoothed
        spks[:, sl] = dcnv.oasis(np.ascontiguousarray(smoothed, dtype=np.float32),
                                 OASIS_BATCH, tau, frame_rate)
    return dff, spks
```
```python
    # ... but discarded when signal == 'dff' (the default)
        act = (events if signal == 'events' else dff)[:, sl]
```
```python
    brain_region_idx = np.zeros(dff.shape[0], dtype=np.int64)
    return dict(neural=neural_trials, input=input_trials, output=output_trials,
                brain_region_idx=brain_region_idx, info=info,
                plane_of_cell=plane_of_cell)   # plane_of_cell never consumed
```

iii. The AI's own inefficiency list (CONVERSION_NOTES Step 6) covers the cost of dF/F, the NWB read
and the speed-ups it applied, but it does not identify the discarded OASIS output. The implicit
justification for keeping both signals is that `--signal events` must remain reproducible and that
the events array feeds the diagnostic plots and the dF/F-vs-events comparison in Step 8; the
justification for the unused `plane_of_cell` and diagnostics is that they were kept for inspection.
