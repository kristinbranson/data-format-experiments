# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI:000363) is one NWB file per session, stored under `/app/data/sub-<subject_id>/`. The AI enumerates every session with a single sorted glob over that layout and processes each file exactly once. Files are opened with **`h5py` directly** (not `pynwb`), reading the HDF5 groups `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, `units/*`, `general/subject/subject_id` and `general/extracellular_ephys/*`. Sessions are converted in parallel with a `multiprocessing.Pool` (default 16 workers) via `imap`, so session order in the output follows the sorted file order. One extra external resource is used: a cached copy of the Allen CCF structure graph (`/app/allen_structure_graph.json`, downloaded during the run) that is loaded once in `main()` and passed to each worker for brain-region labelling. All 174 files are opened; 76 are later excluded (68 behavioural, 6 video coverage, 1 corrupted video clock, 1 no good units), leaving 98 sessions.

ii.
```python
DATA_DIR = '/app/data'
ONTOLOGY_FILE = '/app/allen_structure_graph.json'  # Allen CCF structure graph (cached)
...
    name2anc = load_ontology()
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
    ...
    results = []
    with Pool(args.nproc) as pool:
        for res, info in pool.imap(process_session, [(fn, name2anc) for fn in files]):
            print(info, flush=True)
            if res is not None:
                results.append(res)
```

```python
def process_session(args):
    fname, name2anc = args
    info = {'file': os.path.basename(fname)}
    try:
        with h5py.File(fname, 'r') as f:
            out = _process_session(f, fname, name2anc, info)
    except Exception as exc:  # pragma: no cover
        info['excluded'] = 'error: %s' % exc
        return None, info
    return out, info
```

```python
def _process_session(f, fname, name2anc, info):
    tr = f['intervals/trials']
    be = f['acquisition/BehavioralEvents']
    trial_start = tr['start_time'][:]
    trial_stop = tr['stop_time'][:]
    ...
    subject = str(np.asarray(f['general/subject/subject_id'][()]).astype(str))
```

iii. From the trajectory: the AI first dumped the HDF5 tree of a session ("Data are NWB files ... Let's inspect NWB file structure with h5py (faster than pynwb read for structure)"), confirmed the layout, and verified with metadata-only scans that all 174 sessions contain tongue tracking, ~69k classifier-good units and ~95k trials. It chose `h5py` explicitly for speed over `pynwb`, and multiprocessing because the machine has 128 CPUs and ~1 TB RAM (checked in step 26: "Machine is large (1TB RAM, 128 CPU, L4 GPU). Full conversion feasible (~12GB)"). The Allen ontology was fetched because "Region mapping via Allen ontology reproduces published counts exactly for striatum (7664), thalamus (12808), midbrain (7495), medulla (2928)", which the AI used as a correctness check on its loading and labelling.

## 1-b. How are the data split into subjects?

i. Each NWB file stores its animal in `general/subject/subject_id` (a numeric string, e.g. `'440956'`). That value is read per session and carried through to assembly, where `subjects` is the sorted set of unique ids and `subject_idx` gives each session's index into that list. No re-grouping is done: the containing directory name (`sub-440956`) is derived from the same id. Because 68 sessions are dropped on behavioural criteria, only **25** of the 28 released mice survive to the output.

ii.
```python
    subject = str(np.asarray(f['general/subject/subject_id'][()]).astype(str))
    info['subject'] = subject
```

```python
    subjects = sorted({r['subject'] for r in results})
    ...
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['subject']) for r in results]),
```

iii. The AI treated `subject_id` as the canonical animal identifier; it is the only subject field in the file and the directory layout is derived from it, so no inference is required. The trajectory shows the AI tracking per-subject session counts in its scans and reporting "25 mice" in the final summary as an expected consequence of session selection, not as an error.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no splitting or grouping is needed, and all probes of a session are already merged into a single `units` table. The session label is taken from the filename timestamp (`ses-20190208T133600`). **The AI then applies session-level curation that the reference does not**, in two stages:

1. **Behavioural criteria from the data paper's STAR Methods**: overall performance > 65% and at least 50 correct lick-left and 50 correct lick-right trials, computed on *control* trials (no photostimulation, no early lick). Performance is `fraction of control trials whose outcome == 'hit'`, i.e. `ignore` (no-response) trials count in the denominator as incorrect. This removes **68 of 174** sessions.
2. **Video criteria**: sessions with no tongue-tracking series, with non-monotonic (restarting) video timestamps, with fewer than 100 visible tongue frames, or where fewer than 50% of trials have >=95% video coverage of the 4 s analysis window. This removes 7 more sessions (6 coverage, 1 clock).

Plus 1 session with no classifier-good units. Net: **98 sessions** kept, 51,219 trials, 37,990 units (vs. 173 sessions / 90,860 trials / 69,453 units for the human reference).

ii.
```python
PERF_THRESH = 0.65
MIN_CORRECT_PER_DIRECTION = 50
```

```python
    # --- session selection (Chen et al. behavioural criteria) --------------
    # "Overall performance was computed as the fraction of correct control
    # trials (i.e. no photostimulation), excluding any early lick trials."
    control = (photostim_onset == 'N/A') & (early == 'no early')
    perf = float((outcome[control] == 'hit').mean()) if control.sum() else 0.0
    n_correct_left = int((control & (outcome == 'hit') & (instruction == 'left')).sum())
    n_correct_right = int((control & (outcome == 'hit') & (instruction == 'right')).sum())
    ...
    if not (perf > PERF_THRESH and n_correct_left >= MIN_CORRECT_PER_DIRECTION
            and n_correct_right >= MIN_CORRECT_PER_DIRECTION):
        info['excluded'] = 'behavior'
        return None
```

```python
    key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
    if key not in f:
        info['excluded'] = 'no tongue tracking'
        return None
    vts = f[key + '/timestamps'][:]
    if np.any(np.diff(vts) <= 0):
        info['excluded'] = 'non-monotonic video timestamps'
        return None
    ...
    if good_video.mean() < MIN_SESSION_GOOD_VIDEO:
        info['excluded'] = 'video does not cover the analysis window'
        return None
```

iii. The AI's justification, stated in the module docstring and repeated throughout the trajectory, is that `/app/methods.txt` states verbatim: *"We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each."* The AI tested two readings of "performance" and picked the one that counts `ignore` trials as incorrect because it "yields exactly 106 sessions, matching 'n = 106 sessions' in the method paper — strong evidence this is the expert's session selection" (steps 59-61). The alternative definition (hit/(hit+miss)) would have kept 145 sessions. For the video criteria the AI argued that "the tongue position is one of the decoded outputs", so sessions whose video cannot be aligned or does not cover the window are not usable; it individually characterised the 9 anomalous video sessions before deciding (steps 57-61).

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table (`intervals/trials`), with `start_time`/`stop_time` per row. Events are mapped onto trials by time with a helper that finds the trial interval containing each event, rather than assuming one event per row. The go cue is taken from `BehavioralEvents/go_start_times`; the AI verified (scan over all 174 sessions) that every trial has exactly one go cue, while `sample_start_times` can have several per trial because an early lick replays the sample epoch.

ii.
```python
def map_events_to_trials(event_times, trial_start, trial_stop):
    """Index of the trial containing each event (-1 if outside any trial)."""
    idx = np.searchsorted(trial_start, event_times, side='right') - 1
    ok = (idx >= 0) & (event_times <= trial_stop[np.clip(idx, 0, len(trial_stop) - 1)])
    idx[~ok] = -1
    return idx
```

```python
    # --- go cue: exactly one per trial -------------------------------------
    go_all = be['go_start_times/timestamps'][:]
    gidx = map_events_to_trials(go_all, trial_start, trial_stop)
    go = np.full(ntrials_all, np.nan)
    go[gidx[gidx >= 0]] = go_all[gidx >= 0]
```

iii. The trials table defines trials explicitly, so the AI used it directly. It chose the interval-containment mapping instead of positional indexing because it had established empirically that event streams are not one-per-trial in general ("Every trial has exactly one go cue; some trials have multiple sample (tone) epochs due to early-lick replays", step 42). Trials with no go cue keep `go = NaN` and are filtered out later.

## 1-e. How are trials filtered based on quality controls?

i. Five trial-level filters, all motivated by data availability rather than behaviour:

1. **Free-water and auto-water trials excluded** — reward is delivered independently of the animal's choice there, so `choice`/`outcome` are not meaningful.
2. **Video coverage** — a trial is kept only if the number of video frames inside the `[-2.5, +1.5] s` window covers >=95% of the window.
3. **Missing go cue** — trials with no go-cue event (`~np.isfinite(go)`) are dropped.
4. **Outside the ephys observation intervals** — a trial is kept only if **every** selected unit's `units/obs_intervals` includes it. In 8 sessions the ephys covers only part of the behavioural session (e.g. 320 of 480 trials unobserved in one session).
5. **All-zero population activity** — after binning, trials in which no unit fired a single spike anywhere in the window are dropped (recording stopped mid-trial), together with trials whose tone onset is unknown.

A session is dropped if fewer than 2 trials survive. **Early-lick, no-response (`ignore`) and photostimulation trials are deliberately kept**, contrary to both papers, because they are required decoder inputs/outputs.

ii.
```python
    keep = (~auto_water) & (~free_water) & good_video & np.isfinite(go)
```

```python
    nframes_win = np.searchsorted(vts, win_hi) - np.searchsorted(vts, win_lo)
    coverage = nframes_win * np.median(np.diff(vts)) / (OFF_END - OFF_START)
    good_video = coverage >= MIN_TRIAL_VIDEO_COVERAGE
```

```python
    obs_trials = {}
    observed = np.ones(ntrials_all, dtype=bool)
    for u in unit_idx:
        iv = obs[obs_start[u]:obs_idx[u], 0]
        ti = np.searchsorted(trial_start, iv + 1e-6) - 1
        ti = ti[ti >= 0]
        obs_trials[u] = ti
        m = np.zeros(ntrials_all, dtype=bool)
        m[ti] = True
        observed &= m
    keep = keep & observed
    if keep.sum() < 2:
        info['excluded'] = 'too few trials'
        return None
```

```python
    valid = [i for i, x in enumerate(input_trials)
             if np.all(np.isfinite(x)) and neural_trials[i].sum() > 0]
    if len(valid) < 2:
        info['excluded'] = 'no valid trials'
        return None
```

iii. The docstring states: *"Photostimulation, early-lick and no-response (ignore) trials are KEPT, even though the two papers discard them, because the decoder task explicitly asks for photostimulation as an input and early lick / outcome (including 'ignore') as outputs"*, and *"Trials outside the ephys observation intervals ... are dropped: the acquisition was not running, so these are missing data rather than silence."* The AI arrived at the `obs_intervals` filter empirically: the first full run produced a decoder warning about all-zero trials, it traced "930 trials across 6 sessions have all-zero neural data" to sessions where ephys stopped before the behavioural session, verified that "obs_intervals exactly match trial intervals for observed trials", and then added the filter plus a residual zero-population check for the last, partially recorded trial (steps 77-92). Auto-/free-water exclusion is attributed to Wang, Kurgyis et al.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (ragged, indexed by `units/spike_times_index`), restricted to units with `units/classification == 'good'` and surviving the per-trial `units/is_good_trials` check. The go cue times (`BehavioralEvents/go_start_times`) supply the bin edges. Region labels come from `units/anno_name`, `units/electrodes`, `general/extracellular_ephys/electrodes/x` and the probe target in each electrode group's `location` attribute.

ii.
```python
    sp_index = f['units/spike_times_index'][:]
    sp_start = np.concatenate([[0], sp_index[:-1]])
    spike_ds = f['units/spike_times']
    ...
    for i, u in enumerate(unit_idx):
        st = spike_ds[sp_start[u]:sp_index[u]]
```

iii. Spike times are the only neural representation in the file, so firing rates are computed from them directly. The AI verified its binning "exactly against brute-force histograms" from the NWB file for random units/trials (step 68).

## 2-b. How is the `neural` data processed?

i. Spike times are converted to firing rates in Hz per 50 ms bin. For each selected unit, the trial-by-bin edge grid is flattened into one array, sorted (stable) so `np.searchsorted` gives running spike counts at every edge, the permutation is inverted, and adjacent counts are differenced to give per-bin counts, divided by the bin width. No smoothing, normalisation or baseline subtraction. Rates are stored `float32`, one `(n_neurons, 80)` array per trial.

ii.
```python
    edges = (go[trials][:, None] + OFF_START
             + BIN_SIZE * np.arange(NBINS + 1)[None, :])       # (ntrials, nbins+1)
    flat_edges = edges.ravel()
    order = np.argsort(flat_edges, kind='stable')
    sorted_edges = flat_edges[order]

    nn = len(unit_idx)
    rates = np.zeros((nn, len(trials), NBINS), dtype=np.float32)
    for i, u in enumerate(unit_idx):
        st = spike_ds[sp_start[u]:sp_index[u]]
        if len(st) == 0:
            continue
        st = np.sort(st)
        counts_sorted = np.searchsorted(st, sorted_edges)
        counts = np.empty_like(counts_sorted)
        counts[order] = counts_sorted
        counts = counts.reshape(len(trials), NBINS + 1)
        rates[i] = np.diff(counts, axis=1).astype(np.float32) / BIN_SIZE
```

iii. The reference code the AI read (`VideoAnalysisUtils/preprocessing_DJ_2022Aug.py`) computes firing rates as binned spike counts divided by bin width; the AI kept that and only changed the bin geometry to the 50 ms bins prescribed by the decoder task. It cross-validated the result against brute-force histograms before running the full conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters:

1. **`units/classification == 'good'`** — the verdict of the region-specific spike-sorting QC classifiers described in the Chen/Liu white paper and used by both papers. No thresholds on individual quality metrics.
2. **`units/is_good_trials`** — the per-trial unit-stability flag from the MAP pipeline. Its columns run over the trials a unit was *observed* in, so they are mapped back to trial indices via each unit's `obs_intervals`. A unit flagged bad on **any** kept trial is dropped entirely (affects 4 sessions, ~0.8% of units). If the flag array's layout is unexpected, no filtering is applied for that unit.

A session with no classifier-good units, or none surviving the per-trial QC, is dropped. Result: 37,990 units over 98 sessions (mean 388 per session).

ii.
```python
    classification = _str(f['units/classification'][:])
    good_unit = classification == 'good'
    if good_unit.sum() == 0:
        info['excluded'] = 'no good units'
        return None
    unit_idx = np.flatnonzero(good_unit)
```

```python
    ig_all = f['units/is_good_trials'][:]
    unit_ok = np.ones(len(unit_idx), dtype=bool)
    for i, u in enumerate(unit_idx):
        ti = obs_trials[u]
        row = ig_all[u]
        flag = np.zeros(ntrials_all, dtype=bool)
        if len(row) == len(ti):
            flag[ti] = row
        else:  # unexpected layout: do not filter on it
            flag[:] = True
        unit_ok[i] = bool(flag[keep].all())
    unit_idx = unit_idx[unit_ok]
```

iii. The AI validated `classification` as the right field by reproducing the published per-area good-unit counts exactly (striatum 7664, thalamus 12808, midbrain 7495, medulla 2928 of 69,943; step 52). It added `is_good_trials` after discovering that "4 sessions have good units with some bad trials", reasoning in the docstring that this ensures "every neuron is well isolated on every kept trial". It also debugged the column indexing: "`units/is_good_trials` has one column per *observed* trial, not per session trial" (step 88).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial is aligned to its own go-cue onset (`BehavioralEvents/go_start_times`), the event mapped to that trial. All NWB streams (spikes, behavioural events, video) share one session-absolute clock, so alignment is just adding the relative bin grid to the trial's go-cue time; no resampling, interpolation or per-stream offset correction.

ii.
```python
    go_all = be['go_start_times/timestamps'][:]
    gidx = map_events_to_trials(go_all, trial_start, trial_stop)
    go = np.full(ntrials_all, np.nan)
    go[gidx[gidx >= 0]] = go_all[gidx >= 0]
```

```python
    edges = (go[trials][:, None] + OFF_START
             + BIN_SIZE * np.arange(NBINS + 1)[None, :])       # (ntrials, nbins+1)
```

iii. The decoder task prescribes go-cue alignment. The AI checked that "video timestamps share the trial clock" and that spike times are absolute session times before relying on this. It also quantified a consequence and documented it rather than hiding it: "the NWB files store spikes only inside trial intervals. Error (miss) trials end ~0.8 s after the go cue, so the last bins of such trials contain no spikes; this is a property of the data release, not of the alignment" (steps 55, 100-101).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial spanning -2.5 s to +1.5 s relative to the go cue; `metadata['time_bin_size'] = 50.0` ms, `off_start = -2.5`, `off_end = 1.5`. Spikes are binned once at this resolution straight from spike times — there is no intermediate representation and therefore no rebinning. The tongue video (~300 Hz) is *downsampled* into the same 50 ms grid by averaging visible frames within each bin. This deliberately departs from the reference papers' 40 ms window with 3.4 ms stride.

ii.
```python
OFF_START = -2.5           # s, relative to go cue
OFF_END = 1.5              # s, relative to go cue
BIN_SIZE = 0.05            # s (50 ms bins, as requested by the decoder task)
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80
```

```python
            'time_bin_size': BIN_SIZE * 1000.0,
            'temporal_alignment_event': 'go cue onset (auditory go cue, end of delay epoch)',
            'off_start': OFF_START,
            'off_end': OFF_END,
```

iii. Explicit in the docstring: *"Firing rates in non-overlapping 50 ms bins (spikes/s), i.e. 80 bins per trial. The papers use 40 ms bins with a 3.4 ms stride; the task prescribes 50 ms bins instead, which is the only change to the neural processing."* The AI had read the reference preprocessing (step 12: "bin width 0.04s stride 0.0034s, window -3 to 3s aligned to go cue") and consciously overrode it with the decoder-task specification.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the sample-epoch / instruction-tone onsets), combined with the trial's go cue. Sample onsets are mapped to trials by interval containment; where a trial has more than one (early-lick replays the sample epoch), the **last onset preceding the go cue** is used.

ii.
```python
    sam_all = be['sample_start_times/timestamps'][:]
    sidx = map_events_to_trials(sam_all, trial_start, trial_stop)
    tone_onset = np.full(ntrials_all, np.nan)
    for t, i in zip(sam_all, sidx):
        if i >= 0 and (not np.isfinite(go[i]) or t <= go[i]):
            # early-lick trials replay the sample epoch: use the last tone onset
            # that precedes the go cue
            if not np.isfinite(tone_onset[i]) or t > tone_onset[i]:
                tone_onset[i] = t
```

iii. The AI established from its scans that "some trials have multiple sample (tone) epochs due to early-lick replays" (step 42) and from `/app/methods.txt` that a lick during the sample or delay epoch restarts that epoch. It therefore took the last replay, i.e. the tone the animal actually acted on.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the absolute time of each bin centre (`go + bin_center`) minus the tone onset, giving a continuous, monotonically increasing 80-value ramp in seconds. Stored `float32` as row 0 of the `(2, 80)` input array. Trials with no tone onset get an all-NaN row and are subsequently dropped by the `valid` filter. Observed range in the converted data: -1.52 s to +11.89 s (long values on trials with many early-lick replays).

ii.
```python
    bin_centers = OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)
```

```python
        if np.isfinite(tone_onset[t]):
            time_from_tone = (g + bin_centers) - tone_onset[t]
        else:
            time_from_tone = np.full(NBINS, np.nan)
        ...
        inp = np.stack([time_from_tone.astype(np.float32), photostim])
```

iii. The decoder task specifies this input as "continuous, time-varying", so the AI represented it as a per-bin real value rather than a binary onset marker. No further processing was judged necessary beyond locating the correct tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the grid the neural data is binned on: bin *k* of the input is the centre of the same 50 ms interval that bin *k* of the firing-rate matrix counts spikes in, expressed relative to the tone rather than the go cue. Because everything is on one session clock, `(go + bin_centers) - tone_onset` needs no further correction.

ii.
```python
    bin_centers = OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)
    bin_lo = OFF_START + BIN_SIZE * np.arange(NBINS)
    bin_hi = bin_lo + BIN_SIZE
```

```python
        time_from_tone = (g + bin_centers) - tone_onset[t]
```

iii. N/A — the AI treated the shared clock as established fact after verifying it directly on the NWB files.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, mapped to trials by interval containment. (The trials table also carries `photostim_onset`/`photostim_duration` as strings relative to trial start; the AI used the event series instead — I verified the two agree exactly: onset `trial_start + photostim_onset` equals the corresponding `photostim_start_times` entry, and every stimulated trial has exactly one event pair of duration 0.5 s.) Sessions without the series simply get all-zero photostimulation.

ii.
```python
    ps_on = np.full(ntrials_all, np.nan)
    ps_off = np.full(ntrials_all, np.nan)
    if 'photostim_start_times' in be:
        pst = be['photostim_start_times/timestamps'][:]
        psp = be['photostim_stop_times/timestamps'][:]
        pidx = map_events_to_trials(pst, trial_start, trial_stop)
        for a, b, i in zip(pst, psp, pidx):
            if i >= 0:
                ps_on[i] = a
                ps_off[i] = b
```

iii. From step 39: "Photostim: onset relative to trial start, duration 0.5 s, ends at go cue; BehavioralEvents photostim_start/stop_times give absolute times." The AI preferred the absolute event times because they need no string parsing and no re-referencing through `start_time`, and it checked that event counts match the trials-table photostim flags.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1, `float32`) time series over the 80 bins: a bin is 1 if its interval **overlaps** the `[on, off)` stimulation interval (`bin_hi > on and bin_lo < off`), 0 otherwise. Non-stimulated trials keep NaN bounds and so yield all zeros without a special case. Stored as row 1 of the input array.

ii.
```python
        photostim = np.zeros(NBINS, dtype=np.float32)
        if np.isfinite(ps_on[t]):
            a, b = ps_on[t] - g, ps_off[t] - g
            photostim = ((bin_hi > a) & (bin_lo < b)).astype(np.float32)
```

iii. The decoder task asks for "whether photostimulation is on at every time point (discrete, time-varying)", so the AI encoded it per bin rather than as a per-trial flag. Overlap (rather than bin-centre containment) is the natural "is the light on during this bin" test.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The absolute stimulation onset/offset are re-expressed relative to that trial's go cue (`ps_on[t] - g`, `ps_off[t] - g`) and compared against the same go-cue-relative bin edges used for the firing rates, so bin *k* of the photostimulation input covers exactly the interval of bin *k* of the neural matrix.

ii.
```python
            a, b = ps_on[t] - g, ps_off[t] - g
            photostim = ((bin_hi > a) & (bin_lo < b)).astype(np.float32)
```

iii. N/A — single shared clock, no offset correction needed.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column, so choice is derived from two trials-table columns: `trial_instruction` (`'left'`/`'right'`) and `outcome` (`'hit'`/`'miss'`/`'ignore'`). Hit means the animal licked the instructed side, miss means it licked the other side, ignore means no lick. The derivation was cross-checked against the recorded `left_lick_times`/`right_lick_times` event series.

ii.
```python
    outcome = _str(tr['outcome'][:])              # 'hit' / 'miss' / 'ignore'
    instruction = _str(tr['trial_instruction'][:])  # 'left' / 'right'
```

```python
        if outcome[t] == 'hit':
            ch = instruction[t]
        elif outcome[t] == 'miss':
            ch = other[instruction[t]]
        else:
            ch = 'no lick'
```

iii. The docstring records the validation: *"choice (lick direction): hit -> instructed side, miss -> opposite side, ignore -> no lick. Cross-checked against the recorded lick times (>99% agreement)."* The trajectory shows this check being run explicitly on sample sessions before the converter was written (step 44: "verify ... relationship between outcome/instruction and actual licks after the go cue (for 'choice')").

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Mapped to integer codes `left=0`, `right=1`, `no lick=2`, and repeated across all 80 bins so that all four outputs share one `(4, 80)` per-trial array. `output_values[0] = ['left', 'right', 'no lick']`. Resulting distribution: 47.5% left / 47.5% right / 5.0% no lick.

ii.
```python
    other = {'left': 'right', 'right': 'left'}
    choice_map = {'left': 0, 'right': 1, 'no lick': 2}
```

```python
        out = np.stack([
            np.full(NBINS, choice_map[ch], dtype=np.int64),
            ...
        ])
```

iii. Choice is a single value per trial, but the target format asks for time-varying outputs "if at all possible"; since all outputs must share one array, the AI broadcast the per-trial value across bins. The three-way coding follows the decoder task's "left, right, no lick".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already holds exactly the three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
    outcome = _str(tr['outcome'][:])              # 'hit' / 'miss' / 'ignore'
```

iii. No derivation is needed: the field stores precisely the categories the decoder task asks for.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to `ignore=0`, `miss=1`, `hit=2` through a fixed dictionary and repeated across the 80 bins as row 1 of the output array. `output_values[1] = ['ignore', 'miss', 'hit']`. Distribution: 5.0% / 13.9% / 81.1%.

ii.
```python
    outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
```

```python
            np.full(NBINS, outcome_map[outcome[t]], dtype=np.int64),
```

iii. Code order follows the decoder task's listing ("ignore, miss, hit"). Broadcast across bins for the same reason as choice.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The trials-table `early_lick` column, holding `'early'` / `'no early'`.

ii.
```python
    early = _str(tr['early_lick'][:])             # 'early' / 'no early'
```

iii. The flag is stored explicitly, so no derivation is required. The AI kept early-lick trials (both papers exclude them) precisely because this is a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `no=0`, `yes=1` and repeated across the 80 bins as row 2. `output_values[2] = ['no', 'yes']`. Distribution: 88.5% no / 11.5% yes.

ii.
```python
            np.full(NBINS, 1 if early[t] == 'early' else 0, dtype=np.int64),
```

iii. Binary coding follows the decoder task ("no, yes"); per-trial value broadcast across bins.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, a ~300 Hz DeepLabCut series whose `data` is `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps`. Column 1 is the y position; column 2 is the tracking likelihood used to decide visibility.

ii.
```python
    key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
    ...
    vts = f[key + '/timestamps'][:]
    ...
    vdata = f[key + '/data'][:]
    tongue_y = vdata[:, 1].astype(np.float64)
    tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. The AI confirmed the column layout from the series' own attributes rather than assuming it (step 18-19: "Tongue tracking has (x, y, likelihood) at 300 Hz. ~10% frames have high likelihood (tongue visible)") and confirmed all 174 sessions contain the series.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps:
1. **Visibility mask**: frames with DLC likelihood <= 0.9 are marked not visible (the tracker still emits a position when the tongue is retracted).
2. **Outlier cleaning**: a five-sigma threshold on frame-to-frame velocity across visible frames; flagged frames are imputed by linear interpolation from neighbouring good frames (or marked invisible if too few remain).
3. **Session percentiles**: the 40th and 60th percentiles of `tongue_y` over **all visible frames of the session**.
4. **Per-bin mean**: within each 50 ms bin, the mean y over the visible frames falling in that bin (via `bincount` sum/count).

ii.
```python
LIKELIHOOD_THRESH = 0.9    # tongue counted as visible above this DLC likelihood
VELOCITY_SIGMA = 5.0       # outlier rejection on marker velocity (Wang et al.)
```

```python
def clean_marker(y, visible):
    """Remove 5-sigma velocity outliers and impute from neighbouring frames."""
    y = y.copy(); vis = visible.copy()
    iv = np.flatnonzero(vis)
    if len(iv) > 2:
        v = np.diff(y[iv])
        sd = v.std()
        if sd > 0:
            bad = np.abs(v - v.mean()) > VELOCITY_SIGMA * sd
            bad_idx = iv[1:][bad]
            if len(bad_idx):
                good = np.setdiff1d(iv, bad_idx)
                if len(good) > 1:
                    y[bad_idx] = np.interp(bad_idx, good, y[good])
                else:
                    vis[bad_idx] = False
    return y, vis
```

```python
    tongue_y, tongue_vis = clean_marker(tongue_y, tongue_vis)
    ...
    p40, p60 = np.percentile(tongue_y[tongue_vis], [40, 60])
```

```python
                nvis = np.bincount(b_idx[vis], minlength=NBINS)
                ysum = np.bincount(b_idx[vis], weights=tongue_y[lo:hi][vis], minlength=NBINS)
                ymean = np.where(nvis > 0, ysum / np.maximum(nvis, 1), np.nan)
```

iii. The docstring attributes the cleaning step to the method paper: *"Marker traces are cleaned with the five-sigma velocity outlier criterion of Wang, Kurgyis et al. and outliers are imputed from neighbouring frames."* The AI had read `align_markers.py` and the marker-handling code in `VideoAnalysisUtils` (step 17) and checked the likelihood distribution, finding it bimodal ("~10% frames have high likelihood"), which is why the exact 0.9 threshold is not critical.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes per bin: `0` if the bin's mean y is `< p40`, `1` if `<= p60`, `2` otherwise, and `3` ("not visible") if the bin contains no visible frame (or no frames at all). The percentiles are per session and are computed **over raw visible frames**, not over the binned means that are actually being discretised. Resulting distribution over all bins: 13.8% / 7.6% / 7.7% / 70.9% — i.e. among *visible* bins, 47.5% / 26.0% / 26.5%, rather than the 40/20/40 implied by the specification.

ii.
```python
    p40, p60 = np.percentile(tongue_y[tongue_vis], [40, 60])
```

```python
        tclass = np.full(NBINS, 3, dtype=np.int64)
        ...
                seen = nvis > 0
                tclass[seen] = np.where(ymean[seen] < p40, 0,
                                        np.where(ymean[seen] <= p60, 1, 2))
```

```python
        'output_values': [..., ['<40th pct', '40-60th pct', '>60th pct', 'not visible']],
```

iii. The docstring states: *"tongue y position, discretised per session: <40th percentile, 40-60th, >60th percentile of the y positions of all frames in which the tongue is visible, and a fourth class for frames where it is not visible."* The AI read the instruction's "percentile of y-position over the session" literally as a percentile over the measured y samples, and added the fourth class because the instruction lists "3: not visible". It sanity-checked the result by confirming "tongue classes behave sensibly (visible only after the go cue)" (step 68), but did not check the resulting class balance against 40/20/40.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session clock as spikes and events. For each trial, `searchsorted` finds the frame range covering `[go - 2.5, go + 1.5]`, and each frame is assigned a bin by `floor((t - (go + OFF_START)) / BIN_SIZE)`, clipped to `[0, 79]` — the identical go-cue-relative grid used for firing rates. Trials whose video does not cover >=95% of the window are dropped rather than padded (see 1-e), and sessions with non-monotonic video clocks are dropped entirely.

ii.
```python
        lo = np.searchsorted(vts, g + OFF_START)
        hi = np.searchsorted(vts, g + OFF_END)
        tclass = np.full(NBINS, 3, dtype=np.int64)
        if hi > lo:
            b_idx = np.floor((vts[lo:hi] - (g + OFF_START)) / BIN_SIZE).astype(int)
            np.clip(b_idx, 0, NBINS - 1, out=b_idx)
            vis = tongue_vis[lo:hi]
```

iii. The AI verified the shared clock before relying on it ("video timestamps share the trial clock", step 56) and specifically investigated the sessions where this assumption fails — one with only 49 s of video, five where the video stops before the go cue, three with duplicated/restarting timestamps — before deciding to exclude them (steps 57-61).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled case by case, generally by **excluding** rather than imputing:

- **Session never quality-controlled** (`classification` all NaN): becomes the string `'nan'`, no `'good'` units, session dropped (1 session).
- **Missing / corrupted video**: no tracking series, non-monotonic timestamps, <100 visible frames, or <50% of trials with adequate coverage → session dropped (7 sessions).
- **Ephys not running** (trials outside `obs_intervals`, or trials with zero population spikes): trials dropped.
- **Missing go cue or missing tone onset**: `NaN` propagates into `go` / `time_from_tone`, and the trial is removed by the `np.isfinite` filters.
- **Free-water / auto-water trials**: excluded as behaviourally meaningless.
- **Tongue not tracked in a frame**: excluded from the bin mean; a bin with no visible frame becomes the explicit `'not visible'` category.
- **Tracking outliers**: imputed by interpolation rather than dropped.
- **Unexpected `is_good_trials` layout**: treated as "all good" so the filter never silently removes units on a layout it does not understand.
- **Any other exception in a session**: caught, recorded as `'excluded': 'error: ...'`, and the session is skipped.

ii.
```python
    except Exception as exc:  # pragma: no cover
        info['excluded'] = 'error: %s' % exc
        return None, info
```

```python
    keep = (~auto_water) & (~free_water) & good_video & np.isfinite(go)
```

```python
    valid = [i for i, x in enumerate(input_trials)
             if np.all(np.isfinite(x)) and neural_trials[i].sum() > 0]
```

```python
        if len(row) == len(ti):
            flag[ti] = row
        else:  # unexpected layout: do not filter on it
            flag[:] = True
```

iii. The consistent rationale in the docstring is that absence of acquisition must not be encoded as a measurement: *"Trials outside the ephys observation intervals (`units/obs_intervals`), or without a single recorded spike, are dropped: the acquisition was not running, so these are missing data rather than silence."* Where the measurement legitimately has no value (tongue retracted), the AI instead created an explicit category. Every exclusion is logged in `info` and stored in `metadata['session_info']`, so the curation is auditable from the output pickle.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant costs are I/O and the per-unit binning:
- reading each unit's `spike_times` slice individually from the HDF5 dataset (`spike_ds[sp_start[u]:sp_index[u]]`), i.e. one read per unit rather than one bulk read of the ragged buffer;
- reading the full `(n_frames, 3)` tongue array and the `units/is_good_trials` matrix;
- the `np.searchsorted` of 80x`n_trials` edges per unit (the single largest arithmetic cost);
- the Python per-trial loop that builds inputs/outputs and the tongue classes;
- pickling the 6.5 GB result at the end, and (for the downstream decoder) reading it back.

These are largely hidden by `multiprocessing.Pool` across sessions (16 workers by default, 32 used for the real run), so the whole 174-session conversion completed in a few minutes rather than the tens of minutes a serial pass would take.

ii.
```python
    with Pool(args.nproc) as pool:
        for res, info in pool.imap(process_session, [(fn, name2anc) for fn in files]):
```

```python
    for i, u in enumerate(unit_idx):
        st = spike_ds[sp_start[u]:sp_index[u]]
        ...
        counts_sorted = np.searchsorted(st, sorted_edges)
```

iii. The AI did not profile explicitly, but it checked machine resources up front ("1TB RAM, 128 CPU") and designed for parallelism from the start, then noted after the first run that the conversion "finished quickly" (a few minutes for 174 sessions) and that the slow part of validation was loading the 6.5 GB pickle into the decoder.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python-level loops remain:
1. `for u in unit_idx:` building the `observed` mask — could be a single `bincount`/`isin` over all obs-interval starts.
2. `for i, u in enumerate(unit_idx):` applying `is_good_trials` — same structure, could be one masked matrix reduction over `ig_all`.
3. `for i, u in enumerate(unit_idx):` for spike binning — intrinsically hard to collapse, since `spike_times` is ragged, but the *trial* dimension is already vectorised by flattening the edge grid.
4. `for k, t in enumerate(trials):` building inputs/outputs/tongue — the per-trial input and the three scalar outputs are pure broadcasts and could be built as whole `(n_trials, 80)` arrays; only the tongue binning genuinely needs per-trial frame slices, and even that could use a global bin index plus one `bincount`.
5. `for t, i in zip(sam_all, sidx)` and `for a, b, i in zip(pst, psp, pidx)` iterate over events in Python; both could be done with `np.maximum.at` / fancy indexing.

ii.
```python
    for u in unit_idx:
        iv = obs[obs_start[u]:obs_idx[u], 0]
        ti = np.searchsorted(trial_start, iv + 1e-6) - 1
        ...
        m = np.zeros(ntrials_all, dtype=bool)
        m[ti] = True
        observed &= m
```

```python
    for k, t in enumerate(trials):
        ...
        out = np.stack([
            np.full(NBINS, choice_map[ch], dtype=np.int64),
            np.full(NBINS, outcome_map[outcome[t]], dtype=np.int64),
            np.full(NBINS, 1 if early[t] == 'early' else 0, dtype=np.int64),
            tclass,
        ])
```

iii. The AI never discussed vectorisation in the trajectory; it relied on multiprocessing across sessions for throughput, which made the per-session Python loops fast enough in wall-clock terms that they were not revisited. Loops 1, 2 and 4 each allocate an `ntrials_all` array per unit / a fresh `(4, 80)` stack per trial, which is the clearest avoidable overhead.

## 10-c. What processing does the code repeat multiple times?

i. Several quantities are recomputed:
- `map_events_to_trials` is run three times (go cue, sample, photostim) — unavoidable per stream, but each re-does the same `searchsorted` against `trial_start`.
- `np.median(np.diff(vts))` is computed twice (once for coverage at line 291, once as the unused `frame_dt` at line 454).
- `np.flatnonzero(keep)` is computed twice (lines 314 and 371), and the per-unit `obs_intervals -> trial index` mapping is computed once and cached in `obs_trials`, but the derived boolean mask `flag`/`m` is rebuilt per unit in both the `observed` loop and the `is_good_trials` loop.
- `np.sort(st)` re-sorts spike times that are already sorted in the NWB file, once per unit.
- `np.argsort(flat_edges)` sorts an edge array that is already monotonically increasing (trials are in time order), then inverts the permutation — both operations are no-ops in practice.
- The Allen ontology dict is built once but is re-pickled and shipped to a worker for each of the 174 tasks, since it is passed as a per-task argument rather than via an initializer.

ii.
```python
    coverage = nframes_win * np.median(np.diff(vts)) / (OFF_END - OFF_START)
    ...
    frame_dt = np.median(np.diff(vts))
```

```python
    order = np.argsort(flat_edges, kind='stable')
    sorted_edges = flat_edges[order]
    ...
        st = np.sort(st)
```

```python
        for res, info in pool.imap(process_session, [(fn, name2anc) for fn in files]):
```

iii. Not discussed in the trajectory. The defensive re-sorting (`np.sort(st)`, `argsort(flat_edges)`) is cheap insurance against an unsorted file rather than an oversight, and the ontology re-pickling is a side effect of the simple `imap(f, [(fn, ontology) ...])` idiom.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A handful of computations are performed and then discarded:
- **`frame_dt` (line 454) is computed and never used** — dead code.
- Firing rates, inputs, outputs and tongue classes are computed for **all** trials in `trials`, and only afterwards is the `valid` mask applied; work done on the dropped trials is thrown away.
- `clean_marker` runs the five-sigma velocity cleaning over **every frame of the session**, including the ~70% of frames outside any analysed 4 s window and the ~90% of frames where the tongue is not visible at all.
- `vdata` is read in full, including the unused `tongue_x` column.
- `probe_target` is parsed from the `location` JSON attribute of every electrode group, including probes with no surviving good units.
- The full `units/is_good_trials` matrix is read even for sessions with no flagged units (the common case).
- `obs_trials` caches per-unit trial-index arrays for all units, including those subsequently dropped by the QC filter.
- Extensive per-session `info` diagnostics are computed and stored in `metadata['session_info']`; these are useful for auditing but are ignored by the decoder.

ii.
```python
    frame_dt = np.median(np.diff(vts))
    for k, t in enumerate(trials):
        ...
```

```python
    valid = [i for i, x in enumerate(input_trials)
             if np.all(np.isfinite(x)) and neural_trials[i].sum() > 0]
    ...
    neural_trials = [neural_trials[i] for i in valid]
```

```python
    tongue_y, tongue_vis = clean_marker(tongue_y, tongue_vis)
```

iii. Not discussed in the trajectory. The `valid`-after-the-fact ordering is a consequence of the zero-spike filter being added late in development (steps 92-93) as a patch on top of the existing loop rather than folded into the earlier `keep` mask; the diagnostics were deliberately retained so that every exclusion decision is recoverable from the output pickle.
