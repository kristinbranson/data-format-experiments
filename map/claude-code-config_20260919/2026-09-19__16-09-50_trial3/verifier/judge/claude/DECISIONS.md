# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats the DANDI:000363 layout (`/app/data/sub-<subject_id>/*.nwb`, one NWB/HDF5 file per session) as the complete dataset and discovers all 174 files with a single sorted `glob`. It deliberately does **not** use `pynwb`; it opens each file directly with `h5py` and reads only the datasets it needs (`intervals/trials/*`, `acquisition/BehavioralEvents/*`, `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, `units/*`, `general/extracellular_ephys/electrodes`, `general/subject/subject_id`, `identifier`).

Loading happens in **two passes**. Pass 1 opens every one of the 174 files serially and reads only the trials table to evaluate the data paper's session-inclusion criteria. Pass 2 re-opens the 150 selected files in a `multiprocessing` pool (spawn context, default 16 workers, run with `--njobs 24`) and does the full conversion. An Allen CCF structure graph JSON cached at `cache/allen_structure_graph.json` is loaded once per worker for brain-region assignment.

ii.
```python
DATA_DIR = '/app/data'
...
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
print('Found %d NWB files' % len(files), flush=True)

ontology = load_ontology()

# --- pass 1: session selection (cheap: only the trials table is read) -------------
selected = []
excluded = []
for fp in files:
    with h5py.File(fp, 'r') as f:
        sid = f['identifier'][()].decode()
        tr = read_trial_table(f)
        ok, perf, ncl, ncr, reason = session_passes(f, tr)
    if ok:
        selected.append(fp)
    else:
        excluded.append((sid, perf, ncl, ncr, reason))
```
```python
# --- pass 2: full conversion ------------------------------------------------------
if args.njobs > 1 and len(jobs) > 1:
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.njobs, initializer=_worker_init) as pool:
        for i, res in enumerate(pool.imap(_worker, jobs)):
            results.append(res)
```
```python
def process_session(filepath, show_processing=False, ontology=None):
    with h5py.File(filepath, 'r') as f:
        session_id = f['identifier'][()].decode()
        subject_id = f['general/subject/subject_id'][()].decode()
        mouse = session_id.split('_')[0]
        tr = read_trial_table(f)
        ...
```

iii. From CONVERSION_NOTES.md Step 2/Step 6: the dandiset is "one NWB (HDF5) file per recording session, 174 files, 28 subjects, 50 GB total", so the directory listing is the authoritative session list and a sorted glob is complete and deterministic. `h5py` was chosen over `pynwb` for speed (the notes report 0.2–2 s per session and a 22 s wall clock for all 150 sessions with 24 workers; the reference pipeline's `loadmat`-based loader is not usable because it consumes DataJoint `.mat` exports that are not distributed). Step 4 documents an explicit field-by-field map from every reference `.mat` field to its NWB counterpart, which is how the AI justifies that reading NWB directly reproduces the reference loading. The two-pass design is justified as "cheap: only the trials table is read" in pass 1.

## 1-b. How are the data split into subjects?

i. Each session is attributed to a mouse by parsing the **mouse name out of `nwb.identifier`** (e.g. `SC015_20190207_120657_s1` → `SC015`), not by using the numeric DANDI `subject_id`. Both are read; `subject_id` is kept only in `metadata['session_info']` and used for a 1:1 consistency assertion. `subjects` is the sorted set of unique mouse names and `subject_idx` indexes into it per session. Result: 28 subjects with 3–10 sessions each.

ii.
```python
session_id = f['identifier'][()].decode()
subject_id = f['general/subject/subject_id'][()].decode()
mouse = session_id.split('_')[0]
...
result = {
    'session_id': session_id,
    'subject': mouse,
    'subject_id': subject_id,
    ...
```
```python
subjects = sorted({r['subject'] for r in results})
sub_to_idx = {s: i for i, s in enumerate(subjects)}
# the mouse name and the DANDI subject id must be in one-to-one correspondence
pairs = {(r['subject'], r['subject_id']) for r in results}
assert len(pairs) == len(subjects), 'mouse name / subject id mapping is not 1:1'

data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array([sub_to_idx[r['subject']] for r in results], dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 2 records both identifiers (`identifier` = "mouse name + date + time + session no."; `general/subject/subject_id` = "numeric DANDI subject id, e.g. 440956"). The AI uses the mouse name because it is the identifier the papers use (e.g. "SC015"), making the output directly comparable to the reference texts, and guards the substitution with an assertion that the mouse-name ↔ `subject_id` mapping is one-to-one (Step 10, Check 2: "`main` asserts that mouse name ↔ DANDI subject id is a 1:1 mapping").

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is inferred. Sessions are identified by `nwb.identifier`. **The AI then applies the data paper's session-inclusion criteria and drops 24 of 174 sessions**: control-trial performance must be > 65 %, and there must be ≥ 50 correct lick-left and ≥ 50 correct lick-right trials. Two additional practical criteria are added: at least one QC-good unit, and at least 2 usable trials. 23 sessions fail the performance criterion, 1 fails for having zero QC-good units, none fails the ≥ 50-correct criterion → **150 sessions converted**. Sessions are sorted by `session_id`, and per-session records (plus a list of excluded sessions with reasons) are written into `metadata`.

ii.
```python
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50
MIN_TRIALS_PER_SESSION = 2

def session_performance(tr):
    """Behavioral performance as defined in the data paper. ..."""
    control = ((tr['early_lick'] == 'no early') &
               (tr['photostim_power'] == 'N/A') &
               (tr['auto_water'] == 0) &
               (tr['free_water'] == 0))
    responded = control & (tr['outcome'] != 'ignore')
    if responded.sum() == 0:
        return 0.0
    return float((tr['outcome'][control] == 'hit').sum() / responded.sum())


def session_passes(f, tr):
    perf = session_performance(tr)
    ncorrect_left = int(((tr['outcome'] == 'hit') & (tr['instruction'] == 'left')).sum())
    ncorrect_right = int(((tr['outcome'] == 'hit') & (tr['instruction'] == 'right')).sum())
    ngood = int((_dec(f['units/classification'][:]) == 'good').sum())
    nkept = int(trial_mask(f, tr).sum())

    reason = ''
    if perf <= MIN_PERFORMANCE:
        reason = 'performance <= %.2f' % MIN_PERFORMANCE
    elif min(ncorrect_left, ncorrect_right) < MIN_CORRECT_PER_DIRECTION:
        reason = 'fewer than %d correct trials in one direction' % MIN_CORRECT_PER_DIRECTION
    elif ngood == 0:
        reason = 'no units passed quality control'
    elif nkept < MIN_TRIALS_PER_SESSION:
        reason = 'fewer than %d usable trials' % MIN_TRIALS_PER_SESSION
    return reason == '', perf, ncorrect_left, ncorrect_right, reason
```
```python
results = [r for r in results
           if len(r['neural']) >= MIN_TRIALS_PER_SESSION and r['nneurons'] > 0]
results.sort(key=lambda r: r['session_id'])
```

iii. CONVERSION_NOTES.md Step 3 quotes the data paper verbatim: "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each", and the performance definition "the fraction of correct control trials (i.e. no photostimulation), excluding any early lick trials". Step 4/Step 9 present this as the resolution of a statistics discrepancy: applying exactly these criteria reproduces the paper's "476 trials per session (range 130–785)" as 478.6 (130–785) and "84 % correct (65–99 %)" as 83.7 % (65.8–98.9 %), which the AI treats as decisive evidence that the criteria are the intended curation. The extra "≥ 1 good unit" / "≥ 2 usable trials" rules are justified as, respectively, a crash fix (Step 10 iteration 3) and the target format's stated requirement.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table `intervals/trials`, read once into a dict of numpy arrays. Trial identity is tied to the go cue: the code asserts there is exactly one `go_start_times` event per trials-table row, and every downstream quantity is indexed by that row order.

ii.
```python
def read_trial_table(f):
    """Read the fields of intervals/trials that we need, plus the go cue times."""
    t = f['intervals/trials']
    tr = {
        'start_time': t['start_time'][:],
        'stop_time': t['stop_time'][:],
        'outcome': _dec(t['outcome'][:]),
        'instruction': _dec(t['trial_instruction'][:]),
        'early_lick': _dec(t['early_lick'][:]),
        'auto_water': t['auto_water'][:].astype(int),
        'free_water': t['free_water'][:].astype(int),
        'photostim_onset': _dec(t['photostim_onset'][:]),
        'photostim_duration': _dec(t['photostim_duration'][:]),
        'photostim_power': _dec(t['photostim_power'][:]),
    }
    go = f['acquisition/BehavioralEvents/go_start_times']['timestamps'][:]
    ntrials = len(tr['start_time'])
    if len(go) != ntrials:
        raise ValueError('number of go cues (%d) != number of trials (%d)' % (len(go), ntrials))
    tr['go_time'] = go
```

iii. CONVERSION_NOTES.md Step 4 records the verification that `go_start_times` has "exactly one per trial, inside `[start_time, stop_time]`, in all 174 sessions", which is why the go cue can be used as the trial key. The notes also flag that this is *not* true of `sample_start_times`/`delay_start_times`, because "Licking during sample/delay triggers a *replay* of the epoch, so a trial can have more than one `sample_start_times` / `delay_start_times` event (5.8 % of trials)" — hence the trials table, not the event streams, defines trial boundaries. The length check is listed in Step 10 as a built-in cross-check.

## 1-e. How are trials filtered based on quality controls?

i. Within the 150 retained sessions, three trial filters are applied:
1. **`auto_water == 0` and `free_water == 0`** — trials where water is delivered regardless of the animal's action.
2. **`obs_intervals` coverage** — only trials the units were actually being recorded during. `units/obs_intervals` is read for the first unit, checked against the last unit for equality, and matched against the trials table by `start_time`/`stop_time`.
3. **Silent-trial rule** — any remaining trial in which *no* neuron emits a single spike in the 4 s window is dropped after binning (1 trial dataset-wide).

Early-lick, no-response (`ignore`) and photostimulation trials are deliberately **kept**. 77,521 trials survive.

ii.
```python
def observed_trial_mask(f, tr):
    u = f['units']
    index = u['obs_intervals_index'][:].astype(np.int64)
    first = u['obs_intervals'][0:index[0]]
    if len(index) > 1:
        last = u['obs_intervals'][index[-2]:index[-1]]
        if last.shape != first.shape or not np.allclose(first, last):
            raise ValueError('obs_intervals differ between units')

    start = tr['start_time']
    pos = np.searchsorted(start, first[:, 0])
    pos = np.clip(pos, 0, len(start) - 1)
    if not (np.allclose(start[pos], first[:, 0]) and
            np.allclose(tr['stop_time'][pos], first[:, 1])):
        raise ValueError('obs_intervals do not line up with the trial table')
    mask = np.zeros(len(start), dtype=bool)
    mask[pos] = True
    return mask


def trial_mask(f, tr):
    return ((tr['auto_water'] == 0) & (tr['free_water'] == 0) &
            observed_trial_mask(f, tr))
```
```python
# A 4 s window in which not one of several hundred simultaneously recorded
# neurons fires is not biologically possible; it means the window is outside the
# ephys recording ... Drop those trials.
silent = fr.sum(axis=(1, 2)) == 0
n_silent = int(silent.sum())
if n_silent:
    kept_positions = np.flatnonzero(keep)
    keep[kept_positions[silent]] = False
    fr = fr[~silent]
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 4: auto-/free-water trials are excluded because "on those trials the animal is given water irrespective of its action, so `outcome`/`choice` do not reflect a decision — the reference excludes them in `get_regular_trial_mask`"; early-lick, no-response and photostim trials are kept "because the Decoder Task requires early lick and 'ignore'/'no lick' as output classes and photostimulation as an input; excluding them would make those classes empty". The `obs_intervals` and silent-trial rules were both discovered empirically during Step 7/Step 10 review: "321/480 trials of `SC015_…_s2` (and large blocks in 8 other sessions) had all-zero firing rates → mask trials with `units/obs_intervals`" and "one residual all-zero trial (last `obs_intervals` trial past the end of the recording) → drop trials in which no neuron fires at all". The AI notes the reference paper's "Early lick trials and no response trials were excluded for analysis" is an analysis-level mask that cannot be applied here (Step 1 note, Step 10 Check 3 row (f)).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` + `units/spike_times_index` (the ragged VectorIndex giving per-unit slices), restricted to units with `units/classification == 'good'`. Spike times are absolute session seconds. The go cue times from `acquisition/BehavioralEvents/go_start_times.timestamps` supply the alignment. `units/anno_name`, `units/electrodes` and `general/extracellular_ephys/electrodes/{x,y,z}` are read only to assign brain regions, not to build the rates.

ii.
```python
def load_good_units(f, ontology):
    u = f['units']
    classification = _dec(u['classification'][:])
    good = np.flatnonzero(classification == 'good')

    anno = _dec(u['anno_name'][:])[good]
    e = f['general/extracellular_ephys/electrodes']
    exyz = np.stack([e['x'][:], e['y'][:], e['z'][:]], axis=1)
    eidx = u['electrodes'][:][good]
    ccf_xyz = exyz[eidx]

    region_idx = assign_brain_regions(anno, ccf_xyz, ontology)

    spike_times = u['spike_times'][:]
    spike_index = u['spike_times_index'][:]
    starts = np.concatenate([[0], spike_index[:-1]]).astype(np.int64)
    ends = spike_index.astype(np.int64)
    spikes = [spike_times[starts[i]:ends[i]] for i in good]

    return spikes, region_idx, anno, ccf_xyz
```

iii. CONVERSION_NOTES.md Step 4 maps the reference's `neuron_single_units` (spike times relative to the go cue) to "`units/spike_times` (absolute) − go cue time", noting that this is the only neural representation in the NWB file. The whole `spike_times` buffer is read once and sliced rather than doing one HDF5 read per unit (Step 6 efficiency notes).

## 2-b. How is the `neural` data processed?

i. Spike times are converted to **firing rates in Hz**, with no smoothing, normalisation or baseline subtraction. For each good unit, the flattened `(ntrials × 81)` array of absolute bin edges is `np.searchsorted`-ed into the unit's sorted spike train; differencing adjacent running counts gives the per-bin spike count; the whole array is then divided by the 50 ms bin width. Output is `float32`, one `(n_neurons, 80)` C-contiguous matrix per trial.

ii.
```python
def bin_spikes(spikes, go_times):
    """Bin spikes into firing rates aligned to the go cue. ..."""
    ntrials = len(go_times)
    edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()    # (ntrials*81,)
    fr = np.empty((ntrials, len(spikes), N_BINS), dtype=np.float32)
    for i, st in enumerate(spikes):
        pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
        fr[:, i, :] = np.diff(pos, axis=1)
    fr /= BIN_SIZE
    return fr
```
```python
'neural': [np.ascontiguousarray(fr[i]) for i in range(fr.shape[0])],
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 7 and Step 13 item 2: "Firing rates in Hz (count / 0.05 s), float32 — matches `sliding_histogram(rate=True)`", i.e. the reference `sliding_histogram` returns `binSpikes / bin_width`, so the rate convention is identical even though the bin width differs. Step 12 contains an explicit, quantified decision *not* to rescale: a sweep showed spike counts or arbitrary smaller units gain 1–3 accuracy points with the provided decoder, but the AI ships Hz because "(a) that is exactly what the reference pipeline produces … (b) Hz is the standard, interpretable unit … and (c) the 1–3 point difference is an artefact of one decoder's optimiser settings, not a property of the data". Step 5 Key Decision 5 justifies slicing in continuous session time rather than clipping at NWB `start_time`/`stop_time`: "15.6 % of trials stop less than 1.5 s after the go cue, so clipping would inject artefactual zero firing rates."

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single criterion: `units/classification == 'good'`, the verdict of the region-specific QC classifier. No thresholds are applied to any individual QC metric, no hemisphere split, no firing-rate or variance cut. `units/unit_quality` and `units/is_good_trials` are examined but deliberately not used. Sessions with zero good units are dropped (1 session). 59,749 good units across the 150 retained sessions (69,453 over all 174).

ii.
```python
classification = _dec(u['classification'][:])
good = np.flatnonzero(classification == 'good')
```
```python
ngood = int((_dec(f['units/classification'][:]) == 'good').sum())
...
elif ngood == 0:
    reason = 'no units passed quality control'
```

iii. CONVERSION_NOTES.md Step 4/Step 5 Key Decision 1: the criterion is validated by reproducing the data paper's Fig. 1J per-area unit counts **exactly** for 12 of 14 coarse areas (thalamus 12,808; orbital 10,223; striatum 7,664; midbrain 7,495; olfactory 4,137; medulla 2,928; cortical subplate 1,960; hippocampus 1,944; cerebellum 1,820; pallidum 1,092; hypothalamus 815; pons 347) and ALM to 0.1 % — "a decisive confirmation that `classification == 'good'` is the paper's QC criterion". No metric thresholds are added "because the classifier already consumes all 15 metrics (white paper / `methods.txt`)". Key Decision 2 explains why `units/is_good_trials` is not applied: "only 0.14 % of (unit, trial) pairs and 565 / 69,453 good units are affected. The target format requires a rectangular (n_neurons, 80) array per trial, so per-trial unit exclusion is impossible". The residual 490-unit shortfall vs the paper's 69,943 is attributed to a different dandiset version, all of it in cortex.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go cue onset** (`acquisition/BehavioralEvents/go_start_times.timestamps`). No resampling or clock correction is required: NWB spike times, event timestamps and camera timestamps all live on one session-absolute clock, so the fixed go-cue-relative edge grid is simply added to each trial's go-cue time to produce absolute bin edges, and the spikes are binned against those. The metadata records the alignment event explicitly.

ii.
```python
go = f['acquisition/BehavioralEvents/go_start_times']['timestamps'][:]
...
go = tr['go_time'][keep]
fr = bin_spikes(spikes, go)
```
```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()    # (ntrials*81,)
```
```python
'temporal_alignment_event': 'go cue onset (auditory go cue, 6 kHz, 0.1 s), '
                            'NWB acquisition/BehavioralEvents/go_start_times',
'off_start': OFF_START,
'off_end': OFF_END,
```

iii. CONVERSION_NOTES.md Step 3/Step 4: "everything is aligned to the **go cue** (`task_cue_time` in the export; `go_start_times` in NWB)", and the reference `align_markers.py` also aligns to the go cue. Step 10 Check 3 row (c) marks alignment as "same" as the reference. Multiple sanity checks are cited as confirming the alignment across streams: the population PSTH is flat before and rises sharply after t = 0; P(tongue visible) rises from 0.074 to 0.541 exactly at t = 0 and tracks the lick rate bin for bin; and choice AUC from neural activity sits at chance (0.538) before the tone and rises through the delay epoch — "any leakage of trial identity into the pre-stimulus period (e.g. an off-by-one trial shift) would push it far above 0.5".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **50 ms non-overlapping bins**, 80 bins spanning −2.5 s to +1.5 s relative to the go cue, identical for every trial and every session. The grid is defined once at module level as 81 relative edges plus 80 centres. There is no rebinning of an intermediate representation — spikes go straight from spike times into the final 50 ms grid, and the tongue video (≈300 Hz) is averaged directly into the same 50 ms grid. This deviates from the reference pipeline's 40 ms width / 3.4 ms stride sliding histogram, which the AI flags as mandated by the Decoder Task. `metadata['time_bin_size'] = 50.0` ms, and the bin centres are also stored.

ii.
```python
OFF_START = -2.5          # s, signed time from go cue to start of the extracted trial
OFF_END = 1.5             # s, signed time from go cue to end of the extracted trial
BIN_SIZE = 0.05           # s, 50 ms bins
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))            # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)         # (81,)
BIN_CENTERS = OFF_START + BIN_SIZE * (np.arange(N_BINS) + 0.5)   # (80,)
```
```python
'time_bin_size': BIN_SIZE * 1000.0,            # ms
'bin_centers': BIN_CENTERS.copy(),             # s, relative to the go cue
'neural_units': 'firing rate (spikes/s), spike counts in 50 ms '
                'non-overlapping bins divided by the bin width',
```

iii. CONVERSION_NOTES.md Step 5: "bin *k* covers `[-2.5 + 0.05k, -2.45 + 0.05k)` relative to the go cue; bin centre = `-2.475 + 0.05k`", with the half-open convention verified against `np.histogram` (Step 10 Check 5: "half-open `[edge_k, edge_{k+1})` everywhere, identical to the reference's `(t >= lo) & (t < hi)`"). Step 10 Check 3 rows (d) and (e) record the deviations from the reference as "**differs by task mandate**" — 40 ms/3.4 ms sliding → 50 ms/50 ms, and −3…+3 s → −2.5…+1.5 s — while noting the rate convention (counts / bin width) is unchanged.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `acquisition/BehavioralEvents/sample_start_times.timestamps` (the sample-epoch / instruction-tone onsets) together with each trial's go-cue time. Because an early lick replays the sample epoch, a trial can have several sample starts; the AI takes the **last sample start at or before the go cue**. A fallback of `go − 1.85 s` (the nominal 0.65 s sample + 1.2 s delay) is applied if a trial has no preceding sample start.

ii.
```python
# Tone (sample epoch) onset: the last sample-epoch start before the go cue.  A trial
# whose sample/delay epoch was replayed after an early lick has several; the one that
# actually preceded the go cue is the last.
sample_start = np.sort(f['acquisition/BehavioralEvents/sample_start_times']['timestamps'][:])
idx = np.searchsorted(sample_start, go, side='right') - 1
tone = np.where(idx >= 0, sample_start[np.maximum(idx, 0)], np.nan)
# Fall back to the nominal 0.65 s sample + 1.2 s delay if a trial has no preceding tone.
tone = np.where(np.isnan(tone), go - 1.85, tone)
tr['tone_time'] = tone
```

iii. CONVERSION_NOTES.md Step 4 maps the reference's `task_sample_time` to `BehavioralEvents/sample_start_times` and records the validation "median onset = −1.85 s re. go cue in **all** 174 sessions (= 0.65 s sample + 1.2 s delay ✓)". Step 3 explains the replay: "Licking during sample/delay triggers a *replay* of the epoch, so a trial can have more than one `sample_start_times` event (5.8 % of trials)", which is why the last one before the go cue is the relevant tone. Step 10 Check 5 records that the fallback "never triggered: 0 of 94,990 trials".

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The value in bin *k* is the bin centre expressed relative to the tone, i.e. `bin_centre_k − (tone − go)`. It is a continuous, monotonically increasing ramp with slope 1, one value per bin, stored as `float32` in row 0 of the `(2, 80)` input array. No clipping, no truncation at 0, and no binarisation: on trials whose tone falls before the window start (5.2 %) the ramp simply begins above 0. Observed range over the full dataset: [−1.5, 11.9] s, with the zero crossing at −1.85 s re. the go cue in the typical session.

ii.
```python
def build_inputs(tr, keep):
    """Build the (ntrials, 2, N_BINS) float32 decoder input array.

    input[0] = time from tone (sample epoch) onset, in seconds, at each bin centre
    input[1] = 1 while ALM photostimulation is on, 0 otherwise
    """
    go = tr['go_time'][keep]
    tone = tr['tone_time'][keep]
    n = len(go)

    inp = np.zeros((n, 2, N_BINS), dtype=np.float32)
    # Time since the tone onset at each bin centre (both expressed relative to the go cue).
    inp[:, 0, :] = (BIN_CENTERS[None, :] - (tone - go)[:, None]).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 variable-mapping table: "last `sample_start_times` before the go cue → `input[s][t][0]` — `time_from_tone_onset` (s), `bin_centre − tone_onset`, continuous", corresponding to the reference's `task_sample_time`. The Decoder Task asks for "Time from tone onset in seconds (continuous, time-varying)", so the AI keeps it continuous rather than converting to a binary onset indicator. Step 10 Check 5 notes the consequence for replayed trials: "5.2 % of trials then have their tone onset before the window start, which simply makes `time_from_tone_onset` start above 0". Sanity-checked in Step 10 Check 2 (`allclose` against an independent re-derivation) and in the `--show-processing` figure (zero crossing at −1.85 s).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same grid used to bin the spikes: `BIN_CENTERS` is the array of centres of the same 80 go-cue-relative bins whose edges (`BIN_EDGES`) define the firing-rate bins. Both `tone` and `go` are absolute session times on the shared NWB clock, so the shift `(tone − go)` puts the tone on the go-cue-relative axis with no interpolation. The same boolean `keep` mask indexes the trials for both streams.

ii.
```python
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)         # (81,)
BIN_CENTERS = OFF_START + BIN_SIZE * (np.arange(N_BINS) + 0.5)   # (80,)
```
```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()          # neural
...
inp[:, 0, :] = (BIN_CENTERS[None, :] - (tone - go)[:, None])      # input 0
```
```python
go = tr['go_time'][keep]
fr = bin_spikes(spikes, go)
...
inp = build_inputs(tr, keep)
out, y_bin, pctls = build_outputs(f, tr, keep)
```

iii. CONVERSION_NOTES.md Step 5 defines the single grid once ("bin *k* covers `[-2.5 + 0.05k, -2.45 + 0.05k)` relative to the go cue; bin centre = `-2.475 + 0.05k`") and all streams are derived from it, so bin *k* of the input covers the same interval as bin *k* of the firing rates by construction. The `--show-processing` figure plots the ramp against the population PSTH on the same axis to make the alignment visually checkable.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the trials-table string columns `intervals/trials/photostim_onset` and `photostim_duration` (both `'N/A'` when the trial was not stimulated), plus `trials/start_time` and the go-cue time to move the onset onto the go-cue-relative axis. `photostim_power` is read as well but used only for the session-performance "control trial" definition. The raw `BehavioralEvents/photostim_{start,stop}_times` event streams are used only as an independent cross-check in the diagnostic plot, not to build the input.

ii.
```python
'photostim_onset': _dec(t['photostim_onset'][:]),
'photostim_duration': _dec(t['photostim_duration'][:]),
'photostim_power': _dec(t['photostim_power'][:]),
```
```python
onset_str = tr['photostim_onset'][keep]
dur_str = tr['photostim_duration'][keep]
start = tr['start_time'][keep]
has_stim = onset_str != 'N/A'
```
```python
# cross-check against the raw photostim event times
ev_on = f['acquisition/BehavioralEvents/photostim_start_times']['timestamps'][:]
ev_off = f['acquisition/BehavioralEvents/photostim_stop_times']['timestamps'][:]
```

iii. CONVERSION_NOTES.md Step 4 maps the reference's `task_stimulation = [power, type, on, off]` to "`intervals/trials/photostim_{power,onset,duration}` + `BehavioralEvents/photostim_{start,stop}_times`" and records the validation "per-session stim-trial counts equal event counts in all 174 sessions". Step 10 Check 5 adds that `'N/A'` strings are "treated as 'no stimulation'; the trials-table count equals the `photostim_start_times` event count in all 174 sessions". The trials table is preferred over the event stream because it gives a direct per-trial onset/duration without needing to assign events to trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A **binary time series** rather than a per-trial flag. The onset string is parsed to float, interpreted as seconds from `trials.start_time`, and re-expressed relative to the go cue (`start + onset − go`); the offset is `onset + duration`. A bin is 1 iff its centre lies in the half-open interval `[on_rel, off_rel)`, else 0, and the result is additionally masked by `has_stim` so unstimulated trials are all-zero. Stored as `float32` in row 1. 2.56 % of all bins are 1; the indicator is never 1 after t = 0.

ii.
```python
has_stim = onset_str != 'N/A'
if has_stim.any():
    onset = np.zeros(n)
    dur = np.zeros(n)
    onset[has_stim] = np.array([float(x) for x in onset_str[has_stim]])
    dur[has_stim] = np.array([float(x) for x in dur_str[has_stim]])
    # photostim_onset is relative to the trial start_time; express relative to go cue.
    on_rel = start + onset - go
    off_rel = on_rel + dur
    inside = ((BIN_CENTERS[None, :] >= on_rel[:, None]) &
              (BIN_CENTERS[None, :] < off_rel[:, None]))
    inp[:, 1, :] = (inside & has_stim[:, None]).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "1 if the bin centre lies in `[onset, onset+duration)`, else 0", corresponding to the reference's `task_stimulation[:, 2:4]`. The Decoder Task asks for "Whether photostimulation is on at every time point (discrete, time-varying)", so a time-varying binary series is required. Step 3 records the expected photoinhibition parameters ("~25 % of trials, ALM, 5 mW/hemisphere, 0.5 s incl. 100 ms ramp-down, **always ends before the go cue**") and Step 9 confirms the converted result: "1 in 2.56 % of bins, never after t = 0", with onsets clustered at −1.2 and −0.5 s. Sanity-checked against the raw event timestamps in Step 10 Check 2 and in the diagnostic figure.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The onset/offset are converted from trial-relative to **go-cue-relative** seconds and then compared directly against `BIN_CENTERS` — the centres of the very bins used for the firing rates. No separate clock or interpolation is involved.

ii.
```python
# photostim_onset is relative to the trial start_time; express relative to go cue.
on_rel = start + onset - go
off_rel = on_rel + dur
inside = ((BIN_CENTERS[None, :] >= on_rel[:, None]) &
          (BIN_CENTERS[None, :] < off_rel[:, None]))
```

iii. The AI's justification (CONVERSION_NOTES.md Step 5, Step 10 Check 2) is that the trials table stores the onset relative to `start_time`, so it must be re-expressed against the alignment event before it can be laid on the bin grid; once on that axis, the same half-open `[lo, hi)` convention as the spike bins applies. The diagnostic figure includes a dedicated panel comparing the converted bin onsets against the raw `photostim_start_times` distribution relative to the go cue, and the AI fixed a bug in that panel (events were being assigned to the previous trial) during Step 7 review — a plot-only fix, listed in the Step 12 issues table.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the NWB file, so choice is **derived** from two trials-table columns: `trial_instruction` (`'left'` / `'right'`, the tone-instructed side) and `outcome` (`'hit'` / `'miss'` / `'ignore'`). `hit` ⇒ the animal licked the instructed side; `miss` ⇒ it licked the opposite side; `ignore` ⇒ it did not lick. The raw `left_lick_times` / `right_lick_times` event streams are used only as an independent verification, not as the source.

ii.
```python
def choice_codes(tr, keep):
    """Lick direction chosen by the animal.

    hit    -> the animal licked the instructed port
    miss   -> the animal licked the other port
    ignore -> no lick
    (Equivalent to the reference's `behavior_report` x `task_trial_type` combination;
    verified against the first lick after the go cue in Step 10.)
    """
    outcome = tr['outcome'][keep]
    instr = tr['instruction'][keep]
```

iii. CONVERSION_NOTES.md Step 4 maps the reference's `behavior_report` (1 correct / 0 error / −1 no response) to `trials/outcome` and `task_trial_type` (`l`/`r`) to `trials/trial_instruction`, so the instruction × outcome product is exactly the reference's representation of choice. Step 5's planned sanity check — "Choice derived from `outcome`+`trial_instruction` agrees with the first lick after the go cue (99.3–100 % over spot-checked sessions)" — is the AI's empirical validation that the derivation is correct; the `--show-processing` confusion-matrix panel reports agreement 1.000 in both sample sessions.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick` via a nested `np.where`, written into row 0 of the `(4, 80)` `int64` output array and **repeated across all 80 bins** so that every output is time-varying in shape. `output_values[0] = ['left', 'right', 'no lick']`. Full-dataset distribution: [0.443, 0.438, 0.120].

ii.
```python
CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ...
]
```
```python
    instr_code = np.where(instr == 'left', CHOICE_LEFT, CHOICE_RIGHT)
    opposite = np.where(instr == 'left', CHOICE_RIGHT, CHOICE_LEFT)
    choice = np.where(outcome == 'hit', instr_code,
                      np.where(outcome == 'miss', opposite, CHOICE_NOLICK))
    return choice.astype(np.int64)
```
```python
out = np.empty((n, 4, N_BINS), dtype=np.int64)
out[:, 0, :] = choice_codes(tr, keep)[:, None]
```

iii. CONVERSION_NOTES.md Step 5: the mapping "`ignore`→`no lick`(2); `hit`→instruction; `miss`→opposite instruction" mirrors `behavior_report` + `task_trial_type`. The left/right ordering follows the Decoder Task's "(left, right, no lick)". Repeating per-trial values across bins is justified as "as allowed and encouraged by the target format ('If at all possible, make it time-varying')" — it keeps all four outputs in one rectangular `(n_output, n_timepoints)` array. Step 11's `predictions.png` review discusses the consequence: per-trial labels can only be read out after the go cue, so "pooling all 80 bins necessarily caps the attainable balanced accuracy".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `intervals/trials/outcome`, a string column already containing exactly the three required categories `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
'outcome': _dec(t['outcome'][:]),
```
```python
out[:, 1, :] = np.array([OUTCOME_CODE[o] for o in tr['outcome'][keep]])[:, None]
```

iii. CONVERSION_NOTES.md Step 4: "`behavior_report` (1 correct / 0 error / −1 no response) ↔ `intervals/trials/outcome` (`hit` / `miss` / `ignore`) — outcome value sets match". No derivation is needed because the stored categories are the requested ones.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped through a fixed dictionary to `0 = ignore`, `1 = miss`, `2 = hit` (the order given in the Decoder Task) and written into row 1 of the output array, repeated across all 80 bins. A `KeyError` would be raised on any unexpected string, so the mapping is total by construction. Full-dataset distribution: [0.120, 0.153, 0.728], consistent with the paper's ~84 % correct rate once `ignore` trials are included in the denominator.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
```
```python
out[:, 1, :] = np.array([OUTCOME_CODE[o] for o in tr['outcome'][keep]])[:, None]
```
```python
OUTPUT_VALUES = [
    ...
    ['ignore', 'miss', 'hit'],
    ...
]
```

iii. The code order follows the Decoder Task's "Outcome (ignore, miss, hit, per-trial)". As with choice, the per-trial value is broadcast across bins so all outputs share one array (CONVERSION_NOTES.md Step 5: "All inputs and outputs are **time-varying**, shape (2, 80) and (4, 80)"). Verified in Step 10 Check 2 against an independent re-read of the raw HDF5 column (`allclose`, 0 failures).

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from `intervals/trials/early_lick`, a string column holding `'early'` / `'no early'`.

ii.
```python
'early_lick': _dec(t['early_lick'][:]),
```

iii. CONVERSION_NOTES.md Step 4 maps the reference's `behavior_early_report` to `intervals/trials/early_lick`, so the flag is stored explicitly and no derivation from lick times is needed.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A boolean comparison `== 'early'` cast to int gives `0 = no`, `1 = yes`, written into row 2 and repeated across all 80 bins. `output_values[2] = ['no', 'yes']`. Full-dataset distribution: [0.885, 0.115].

ii.
```python
out[:, 2, :] = (tr['early_lick'][keep] == 'early').astype(np.int64)[:, None]
```
```python
OUTPUT_VALUES = [
    ...
    ['no', 'yes'],
    ...
]
```

iii. The coding follows the Decoder Task's "Early lick (no, yes, per-trial)". CONVERSION_NOTES.md Step 12 Check 1 discusses the interpretive consequence at length: an early lick occurs during the sample or delay epoch and triggers a replay of that epoch, so "the neural correlate therefore often falls **before** the −2.5 s window boundary (5.2 % of trials have their tone onset before the window starts, and those are exactly the replayed, i.e. early-lick, trials)" — which the AI offers as the structural explanation for early_lick being the lowest accuracy-to-chance ratio (1.50×) rather than a conversion defect. Note that the boolean form silently maps any unexpected string to 0, unlike the dictionary lookups used for choice and outcome.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: `data` is `(n_frames, 3)` = (tongue_x, tongue_y, DeepLabCut likelihood) at ~300 Hz, with `timestamps` on the session clock. Column 1 is the y-position; column 2 is the likelihood used to decide visibility. Only the side camera is used.

ii.
```python
tt = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
data = tt['data']
ts = tt['timestamps'][:]
y = data[:, 1]
lik = data[:, 2]
vis = lik > TONGUE_LIKELIHOOD_THRESHOLD
```

iii. CONVERSION_NOTES.md Step 2 documents the channel layout ("each `data` = (n_frames, 3) = (x, y, DLC likelihood), `timestamps` = session time, 300 Hz (dt = 0.0034 s)") and Step 4 maps the reference's `tracking.camera_0_side.tongue_x/y` to "`BehavioralTimeSeries/Camera0_side_TongueTracking.data[:, 0:2]`, likelihood in column 2". Tracking is present in all 174 sessions.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with `likelihood > 0.9` count as *visible*; all other frames are discarded entirely (not imputed). For each trial, the visible frames falling in the ±window are assigned to 50 ms bins by their offset from the go cue, and the bin's value is the **mean y over the visible frames in that bin** (computed with two `np.bincount` calls, sum / count). A bin with no visible frame is left NaN and becomes the `not visible` class.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
```
```python
    ntrials = len(go_times)
    y_bin = np.full((ntrials, N_BINS), np.nan)
    lo = np.searchsorted(ts, go_times + OFF_START)
    hi = np.searchsorted(ts, go_times + OFF_END)
    for i in range(ntrials):
        sl = slice(lo[i], hi[i])
        if sl.stop <= sl.start:
            continue
        rel = ts[sl] - go_times[i]
        v = vis[sl]
        if not v.any():
            continue
        b = np.floor((rel[v] - OFF_START) / BIN_SIZE).astype(np.int64)
        np.clip(b, 0, N_BINS - 1, out=b)
        counts = np.bincount(b, minlength=N_BINS)
        sums = np.bincount(b, weights=y[sl][v], minlength=N_BINS)
        nz = counts > 0
        y_bin[i, nz] = sums[nz] / counts[nz]
    return y_bin, ~np.isnan(y_bin)
```

iii. CONVERSION_NOTES.md Step 5: "A video frame counts as *tongue visible* if `tongue_likelihood > 0.9` (DLC default cut-off; the likelihood is strongly bimodal so this is not a sensitive choice)" — Step 4 quantifies the bimodality as "10.5 % of frames > 0.9, ~89.5 % < 1e-3" and notes "any threshold in 1e-3 … 0.99 changes the visible fraction by < 0.3 %". The decision to require only *one* visible frame per bin rather than a majority is justified as: "Tongue protrusions last ~50–100 ms, so requiring a majority of the ~15 frames per bin would discard the onset and offset of most licks." The AI explicitly departs from the reference here: the method paper "set the tongue position to its mean value" when occluded, but "the Decoder Task instead defines an explicit 4th class 'not visible', which supersedes the reference's imputation" (Step 4 discrepancy table, Step 10 Check 3 row (h)).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles are computed over **all visible binned y values of that session** (i.e. over the `(ntrials, 80)` binned array, ignoring NaN), then each bin is assigned `0` if `y < p40`, `2` if `y > p60`, `1` otherwise, and `3` if the bin has no visible frame. The percentile values are stored per session in `metadata['session_info'][s]['tongue_percentile_values']`. A session with no visible tongue at all returns all-3 rather than failing. Full-dataset distribution: [0.101, 0.051, 0.101, 0.746] — i.e. of the 25.4 % visible bins, exactly 40 / 20 / 40 %.

ii.
```python
TONGUE_PCTL_LOW = 40.0
TONGUE_PCTL_HIGH = 60.0
```
```python
def discretize_tongue(y_bin):
    """Per-session discretisation of the binned tongue y-position (Decoder Task).

    0: y < 40th percentile of the session's visible y values
    1: 40th - 60th percentile
    2: y > 60th percentile
    3: tongue not visible
    """
    visible = ~np.isnan(y_bin)
    out = np.full(y_bin.shape, 3, dtype=np.int64)
    if visible.sum() == 0:
        return out, (np.nan, np.nan)
    vals = y_bin[visible]
    p40, p60 = np.percentile(vals, [TONGUE_PCTL_LOW, TONGUE_PCTL_HIGH])
    code = np.where(y_bin < p40, 0, np.where(y_bin > p60, 2, 1))
    out[visible] = code[visible]
    return out, (float(p40), float(p60))
```

iii. CONVERSION_NOTES.md Step 5: "Per-session 40th and 60th percentiles are computed over **all visible bins of that session** (this is the 'y-position over the session')". Taking the percentiles over the *binned* values rather than raw frames means the edges are defined on exactly the quantity being discretised. Excluding non-visible frames is justified because the DLC tracker still reports a position when the tongue is retracted, so including them would corrupt the percentiles. The resulting exact 40/20/40 split of visible bins is used in Steps 7 and 9 as a sanity check that the discretisation is correct.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. This is the only genuinely time-varying output. Camera timestamps are on the same session-absolute clock as spikes and events, so each trial's frame range is located with `np.searchsorted` on `ts` at `go + OFF_START` and `go + OFF_END`, and frames are assigned to bins by `floor((t − go − OFF_START) / 0.05)` — the same go-cue-relative grid used for the firing rates. Indices are clipped to `[0, 79]`. No interpolation or offset correction.

ii.
```python
lo = np.searchsorted(ts, go_times + OFF_START)
hi = np.searchsorted(ts, go_times + OFF_END)
for i in range(ntrials):
    ...
    rel = ts[sl] - go_times[i]
    ...
    b = np.floor((rel[v] - OFF_START) / BIN_SIZE).astype(np.int64)
    np.clip(b, 0, N_BINS - 1, out=b)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "Video frames are likewise taken by absolute session time. The camera pauses for ~0.5 s between trials, so a few bins genuinely have no frames; those become class 3 ('not visible'), which is the only available representation of missing tracking." The alignment is verified behaviourally in Step 5/Step 9/Step 12: "P(tongue visible) is ≈ 0 before the go cue and jumps to ~0.4 immediately after it → video and ephys clocks agree" (full-dataset values 0.074 before, 0.541 after), and the diagnostic figure overlays P(tongue visible) on the lick rate computed from `left_lick_times`/`right_lick_times`, showing both rise bin-for-bin at t = 0. Step 10 Check 2 compares the full trials × 80-bin tongue class array against an independent re-derivation (`allclose`, 0 failures).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Seven distinct cases, each handled explicitly and documented in the Step 10 Check 5 edge-case table:
- **Session never quality-controlled** (`SC017_20190216_162508_s4`, 0 good units): the session is excluded. This was added after the first full run crashed on it.
- **Ephys covering only part of the behavioural session** (9 sessions, up to 321/480 trials): trials outside `units/obs_intervals` are dropped.
- **Last `obs_intervals` trial extending past the end of the recording**: any trial in which no neuron fires at all is dropped (1 trial dataset-wide).
- **Trials shorter than the −2.5…+1.5 s window** (3 % at the start, 16 % at the end): spikes and frames are sliced in continuous session time, so nothing is zero-padded.
- **Camera pauses / occluded tongue**: frames below the likelihood threshold and bins with no visible frame become class 3 (`not visible`).
- **Trial with no preceding `sample_start`**: falls back to `go − 1.85 s` (never triggered).
- **`'N/A'` photostim strings**: treated as no stimulation.
- **`units/is_good_trials == False`** (0.14 % of unit×trial pairs): documented but *not* acted on, because the format requires a rectangular per-trial matrix.

Structural inconsistencies are treated as hard failures rather than silently patched: mismatched go-cue/trial counts, `obs_intervals` differing between units or not lining up with the trials table, and a non-1:1 mouse↔subject_id mapping all raise.

ii.
```python
if len(go) != ntrials:
    raise ValueError('number of go cues (%d) != number of trials (%d)' % (len(go), ntrials))
```
```python
if last.shape != first.shape or not np.allclose(first, last):
    raise ValueError('obs_intervals differ between units')
...
if not (np.allclose(start[pos], first[:, 0]) and
        np.allclose(tr['stop_time'][pos], first[:, 1])):
    raise ValueError('obs_intervals do not line up with the trial table')
```
```python
# Fall back to the nominal 0.65 s sample + 1.2 s delay if a trial has no preceding tone.
tone = np.where(np.isnan(tone), go - 1.85, tone)
```
```python
silent = fr.sum(axis=(1, 2)) == 0
if n_silent:
    kept_positions = np.flatnonzero(keep)
    keep[kept_positions[silent]] = False
    fr = fr[~silent]
```
```python
if visible.sum() == 0:
    return out, (np.nan, np.nan)
```
```python
def _worker(args):
    filepath, show = args
    try:
        return process_session(filepath, show_processing=show, ontology=_WORKER_ONTOLOGY)
    except Exception as exc:                      # pragma: no cover - defensive
        import traceback
        traceback.print_exc()
        return {'session_id': os.path.basename(filepath), 'excluded': True, 'error': str(exc)}
```

iii. The governing principle in CONVERSION_NOTES.md is that data which was never recorded is excluded rather than emitted as zeros ("A 4 s window in which not one of several hundred simultaneously recorded neurons fires is not biologically possible"), while a measurement that legitimately has no value (a retracted tongue) is represented as an explicit category rather than imputed. Each of the three real defects was found by the AI's own review loop and is logged with its fix in the Step 12 "Issues found and resolved" table. The `_worker` try/except is a defensive net that converts a per-session crash into an exclusion with a traceback rather than losing the whole run.

## 10-a. What are the most time-consuming steps of the code?

i. The full run takes 43.4 s wall clock: 9.2 s for the serial session-selection pass over all 174 files, 22.2 s for the parallel conversion of 150 sessions (24 workers; 0.2–2 s per session serially, so ~2.5 min of CPU work), and 11.7 s to pickle the 10.21 GB result. Within a session the AI's own instrumentation splits the cost into four phases (`read`, `units`, `bin`, `io`); its Step 7 table attributes ~0.05 s to the trials table, 0.06–0.6 s to reading the `spike_times` buffer, 0.1–0.8 s to spike binning (the per-unit `searchsorted` loop), and 0.16–0.5 s to inputs + outputs (dominated by reading and binning the ~300 Hz video). So: HDF5 reads of the two large ragged/dense arrays, the per-unit binning loop, and the final pickle write.

ii.
```python
result = {
    ...
    'timing': {'read': t_read - t0, 'units': t_units - t_read,
               'bin': t_bin - t_units, 'io': t_io - t_bin,
               'total': t_io - t0},
}
```
```python
print('[%3d/%3d] %s  ntrials=%d/%d nneurons=%d rate=%.2fHz silent=%d '
      '(%.1fs; elapsed %.1fs, eta %.1fs)' % ...)
```
```python
print('  pass 1 took %.1f s' % (time.time() - t_start), flush=True)
...
print('wrote %s (%.2f GB) in %.1f s'
      % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t_save))
print('TOTAL time %.1f s' % (time.time() - t_start))
```

iii. CONVERSION_NOTES.md Step 7 presents the per-step timing table and the extrapolation to the full dataset, concluding "Well under the 15-minute budget, so no further optimisation was needed." The cost is dominated by I/O and by one binary search per bin edge per unit, both of which scale with the data actually needed.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python-level loops remain:
1. **Per-unit loop in `bin_spikes`** — one `np.searchsorted` per unit, but already vectorised over *all* trials and bins simultaneously by flattening the `(ntrials × 81)` edge array. This cannot be collapsed further because the spike trains are ragged.
2. **Per-trial loop in `bin_tongue`** — each trial's frames are binned separately with `np.bincount`. This one *could* be vectorised with a single global bin index (`trial_index * 80 + bin_index`) and one `bincount` over all frames of the session, since the frames are already sorted by time and the trial ranges are already known from `searchsorted`.
3. **Per-unit list comprehension in `load_good_units`** (`[spike_times[starts[i]:ends[i]] for i in good]`) — builds one view per unit; unavoidable given ragged storage, but it materialises the full `spike_times` buffer for *all* units, including the ~75 % that are discarded.

There are also small per-element Python loops for the string→code conversions (`[OUTCOME_CODE[o] for o in ...]`, `[float(x) for x in onset_str[has_stim]]`) which could be `np.vectorize`/`astype(float)`, but these are negligible.

ii.
```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()    # (ntrials*81,)
fr = np.empty((ntrials, len(spikes), N_BINS), dtype=np.float32)
for i, st in enumerate(spikes):
    pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
    fr[:, i, :] = np.diff(pos, axis=1)
```
```python
for i in range(ntrials):
    sl = slice(lo[i], hi[i])
    ...
    counts = np.bincount(b, minlength=N_BINS)
    sums = np.bincount(b, weights=y[sl][v], minlength=N_BINS)
```

iii. CONVERSION_NOTES.md Step 6 states the binning is "fully vectorised — for each unit one `np.searchsorted` of the `ntrials × 81` bin edges into its sorted spike train, then `np.diff`", and that "The tongue loop is vectorised per trial with `np.bincount`", i.e. the AI considers the remaining per-trial loop acceptable. It also documents what was removed: "a naive per-(unit, trial, bin) `np.sum(mask)` (the reference `sliding_histogram` pattern) would be ~10^4 times slower; replaced by the searchsorted/diff formulation." Since the whole conversion runs in 43 s against a 15-minute budget, the residual loops are not on the critical path.

## 10-c. What processing does the code repeat multiple times?

i. The two-pass design causes substantial duplicated work for the 150 selected sessions:
- **`read_trial_table` runs twice per selected session** — once in the pass-1 selection loop and again inside `process_session`. Each call re-reads all ten trials-table columns plus the go-cue and sample-start timestamps.
- **`session_passes` runs twice per selected session** (pass 1 and again at the top of `process_session`), and with it `session_performance` and `_dec(f['units/classification'][:])`.
- **`observed_trial_mask` runs three times per selected session** — twice via `session_passes` → `trial_mask`, and a third time via the direct `trial_mask(f, tr)` call in `process_session`. Each call reads `obs_intervals_index` and two slices of `obs_intervals`.
- **`units/classification` is decoded twice** per session: once in `session_passes` (to count good units) and once in `load_good_units`.
- **Tongue tracking data is read in two separate passes** (`data[:, 1]` then `data[:, 2]`), each a strided read over the full `(n_frames, 3)` dataset; in `--show-processing` mode `tt['data'][:, 1]` and `tt['data'][:, 2]` are read a third and fourth time.
- **`load_ontology` runs once in the parent and once per worker process** (necessary with spawn, but the parent copy is then unused for the parallel path).
- The 24 *excluded* sessions are read once and discarded, which is the intended cost of the cheap first pass.

ii.
```python
    for fp in files:
        with h5py.File(fp, 'r') as f:
            sid = f['identifier'][()].decode()
            tr = read_trial_table(f)
            ok, perf, ncl, ncr, reason = session_passes(f, tr)
```
```python
def process_session(filepath, show_processing=False, ontology=None):
    with h5py.File(filepath, 'r') as f:
        ...
        tr = read_trial_table(f)
        ok, perf, ncl, ncr, reason = session_passes(f, tr)
        if not ok:
            return {...}
        keep = trial_mask(f, tr)
```
```python
def session_passes(f, tr):
    ...
    ngood = int((_dec(f['units/classification'][:]) == 'good').sum())
    nkept = int(trial_mask(f, tr).sum())      # calls observed_trial_mask
```
```python
    y = data[:, 1]
    lik = data[:, 2]
```

iii. CONVERSION_NOTES.md Step 6 justifies the structure only in terms of the pass-1 cost ("cheap: only the trials table is read", measured at 9.2 s for all 174 files) and does not discuss the re-computation inside `process_session`; the notes' claim in Step 6 that "per-trial firing-rate matrices are produced by slicing one preallocated block rather than by concatenating per-trial temporaries" is about a different redundancy. In practice the duplication is affordable — pass 1 is 9.2 s out of 43.4 s and the repeated `obs_intervals`/`classification` reads are small relative to the `spike_times` buffer — so it was not flagged as a problem.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computed quantities never reach the output or are never consumed:
- **`bin_tongue` returns a `visible` mask (`~np.isnan(y_bin)`) that is discarded** at the call site (`y_bin, _ = bin_tongue(...)`), and `discretize_tongue` recomputes the same mask internally.
- **`load_good_units` returns `anno` and `ccf_xyz`**, which after `assign_brain_regions` are never used again in `process_session` (they are only consumed inside the plotting path indirectly, and `anno` not at all).
- **`stop_time` and `photostim_power`** are read for every trial but used only inside `observed_trial_mask`'s consistency assertion and the session-performance filter respectively — neither reaches the output.
- **`session_performance` / `ncorrect_left` / `ncorrect_right` are computed for all 174 sessions** including the 150 that pass, and their values are carried into `metadata` but are not part of the decoder's data.
- **The `'Other'` brain-region bucket** is emitted in `brain_regions` with 0 neurons in the whole dataset (the verification log confirms `Other: 0 neurons`).
- **Diagnostics `n_silent_trials`, `mean_rate_hz`, and the four-way `timing` dict** are computed per session purely for logging.
- **`ev_off` (`photostim_stop_times`) is read in `plot_processing` and never used.**
- Marginally, the AI computes the full `(ntrials, n_units, 80)` firing-rate block and then splits it into 150×~500 separate contiguous arrays via `np.ascontiguousarray`, which copies the whole 10 GB payload once.

ii.
```python
y_bin, _ = bin_tongue(f, go)          # the `visible` mask is thrown away
tongue, pctls = discretize_tongue(y_bin)
```
```python
    return y_bin, ~np.isnan(y_bin)    # second return value unused by build_outputs
```
```python
    return spikes, region_idx, anno, ccf_xyz
...
spikes, region_idx, anno, ccf_xyz = load_good_units(f, ontology)   # anno, ccf_xyz unused after this
```
```python
BRAIN_REGIONS = ['ALM'] + [name for name, _ in COARSE_GROUPS] + ['Other']
```
```python
'neural': [np.ascontiguousarray(fr[i]) for i in range(fr.shape[0])],
```

iii. Most of these are deliberate and documented. The `'Other'` bucket is defended in CONVERSION_NOTES.md Step 10 Check 1: "it is kept deliberately as a fallback bucket in case an annotation ever fails to resolve in the Allen ontology — all 293 annotations in this release resolve." The performance/`ncorrect` values and the `session_info` / `excluded_sessions` metadata blocks are kept as an audit trail for the session-curation decision. The timing and silent-trial diagnostics exist to satisfy the instruction to "print timing information to find bottlenecks". The genuinely wasted work — the discarded `visible` mask, the unused `anno`/`ccf_xyz` returns, and the unused `ev_off` read — is small and not discussed in the notes.
