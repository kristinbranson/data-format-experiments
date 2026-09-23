# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is a DANDI archive (DANDI:000363) laid out as one NWB file per session under `/app/data/sub-<subject_id>/`. The AI enumerates every session with a single sorted glob over that layout (174 files), then opens each file once with `pynwb.NWBHDF5IO` and reads everything it needs from that single handle: `nwb.subject`, `nwb.trials`, `nwb.units`, `nwb.electrodes`, `nwb.acquisition['BehavioralEvents']` and `nwb.acquisition['BehavioralTimeSeries']`. `h5py` is never used. Sessions are farmed out to a `ProcessPoolExecutor` (default 16 workers, 24 used for the full run) so each file is opened exactly once in exactly one worker, and the per-session dictionaries are stitched together in `main()`.

ii.
```python
DATA_DIR = '/app/data'

def list_sessions():
    import glob
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
def convert_session(path, show_processing=False):
    """Convert one NWB session file; returns a dict or None if the session is dropped."""
    t_start = time.time()
    timing = {}
    with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
        nwb = io.read()
        ident = nwb.identifier
        subject = str(nwb.subject.subject_id)
        subject_name = str(nwb.subject.description)
        ...
        units = nwb.units
        ...
        tr = nwb.trials
        ...
        be = nwb.acquisition['BehavioralEvents'].time_series
        ...
        bts = nwb.acquisition['BehavioralTimeSeries'].time_series
```

```python
    results = []
    if args.workers > 1 and len(files) > 1:
        with ProcessPoolExecutor(min(args.workers, len(files))) as ex:
            for i, r in enumerate(ex.map(_worker, tasks)):
                results.append(r)
```

iii. From CONVERSION_NOTES.md Step 2 and Step 4: "`/app/data/` contains one directory per mouse: `sub-<subject_id>/` (28 subjects) … Each session is one NWB file … **174 files total**. All files have identical trial and unit column sets (verified across all 174 files)." The AI explicitly recorded in its Step 4 discrepancy table that the reference code loads DataJoint `.mat` exports with `scipy` `loadmat`, but that "The NWB release contains all fields used by the reference `.mat` pipeline (trials, events, units + QC classifications, CCF annotations, DLC tracking). Use `pynwb`; map each `.mat` field to its NWB equivalent." Parallelism and the one-open-per-file design are justified in Step 6 under efficiency ("Sessions are processed in parallel (16 workers), so the wall-clock cost is dominated by pickling the ~10 GB result"), and the file list is sorted so session order is deterministic. `_worker` wraps `convert_session` in a try/except so a single bad file cannot abort the whole run.

## 1-b. How are the data split into subjects?

i. The subject of a session is read from the NWB file itself, `nwb.subject.subject_id` (a numeric string such as `'440956'`). The lab mouse name (`nwb.subject.description`, e.g. `SC015`) is also read and carried into `session_info` as `subject_name`, but the numeric `subject_id` is the identifier used for `subjects`/`subject_idx`. At assembly, `subjects` is the sorted set of unique ids over the kept sessions, and `subject_idx[s]` is the index of session `s`'s subject in that list. This yields 28 subjects with 3–10 sessions each.

ii.
```python
        subject = str(nwb.subject.subject_id)
        subject_name = str(nwb.subject.description)
```

```python
    subjects = sorted({r['subject'] for r in good})
    subject_idx = np.array([subjects.index(r['subject']) for r in good], dtype=np.int64)
```

```python
        'subjects': subjects,
        'subject_idx': subject_idx,
```

iii. CONVERSION_NOTES.md Step 2: "`nwb.subject.subject_id` (e.g. `440956`), `nwb.subject.description` = lab mouse name (e.g. `SC015`); `nwb.identifier` = e.g. `SC015_20190207_120657_s1`." Step 5 mapping table: "`nwb.subject.subject_id` → `subjects`, `subject_idx` … 28 unique mice … `nwb.subject.description` (e.g. `SC015`) kept in metadata". The subject count of 28 was checked against the papers (Step 3: "data from 28 mice, including 25 VGAT-ChR2-EYFP…" and Fig. 1J "660 penetrations, 173 behavioral sessions, and 28 mice") and confirmed in the Step 9 consistency table as an exact match. The sanity check `subjects[subject_idx[s]] == NWB subject id` was re-verified against the raw files with independent code in Step 10 Check 2.

## 1-c. How are the data split into sessions?

i. No splitting is needed: one NWB file is exactly one session. The AI verified this ("Each session is one NWB file … 174 files total"). Each session is identified by `nwb.identifier` (e.g. `SC015_20190207_120657_s1`, which encodes mouse, date, time and session number) and recorded as `identifier` in the per-session `info` record, which becomes `metadata['session_info']`. Session order in all list-of-sessions fields follows the sorted file list. 173 of the 174 sessions reach the output; one (`SC017_20190216_162508_s4`) is dropped because it has no QC-`good` units, which brings the count to exactly the 173 sessions reported in the papers.

ii.
```python
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
        ident = nwb.identifier
        ...
        if len(good) == 0:
            return {'identifier': ident, 'dropped': 'no good units'}
```

```python
    dropped = [r for r in results if r.get('dropped')]
    for r in dropped:
        print('DROPPED %s: %s' % (r['identifier'], r['dropped']))
    good = [r for r in results if 'neural' in r]
```

```python
    session_info = [r['info'] for r in good]
    ...
            'session_info': session_info,
```

iii. CONVERSION_NOTES.md Step 4: "174 NWB files, one of which (`SC017_20190216_162508_s4`) has **0** good units … Dropping the session with no good units gives exactly **173** sessions ✓", against the papers' "173 behavioral sessions". Step 5 Key Decision 3: "**Session curation**: drop sessions with 0 good units → exactly **173** sessions, matching the papers." No session is dropped on behavioural-performance grounds: Step 4 notes that "Session-selection criteria (>65 % performance) are approximately satisfied; 16/174 sessions are slightly below by my computation but are part of the published dataset, so no session is dropped for performance."

## 1-d. Are the data correctly split into trials?

i. Trials come from the NWB trials table (`nwb.trials`), one row per behavioural trial. The AI asserts that the number of go-cue events equals the number of trial rows, which makes the trial ↔ go-cue mapping unambiguous, and it explicitly noted that the other epoch event streams (`sample_*`, `delay_*`) cannot be used this way because early licking replays those epochs and produces more events than trials.

ii.
```python
        tr = nwb.trials
        trial_start = np.asarray(tr['start_time'][:], dtype=float)
        trial_stop = np.asarray(tr['stop_time'][:], dtype=float)
        outcome = np.asarray(tr['outcome'][:]).astype(str)
        instruction = np.asarray(tr['trial_instruction'][:]).astype(str)
        early = np.asarray(tr['early_lick'][:]).astype(str)
        auto_water = np.asarray(tr['auto_water'][:]).astype(int)
        free_water = np.asarray(tr['free_water'][:]).astype(int)
        stim_flag_tbl = np.asarray(tr['photostim_power'][:]).astype(str) != 'N/A'

        be = nwb.acquisition['BehavioralEvents'].time_series
        go = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
        ...
        assert len(go) == len(trial_start), 'go cue count != trial count in %s' % ident
```

iii. CONVERSION_NOTES.md Step 2: "`go_start_times` (n == n_trials in **all** 174 sessions)"; "Note `sample_*` and `delay_*` have **more** events than trials because early licking triggers a **replay** of the sample/delay epoch (confirmed: extra sample events occur only in `early` trials)." Step 2 "Timing facts verified": "Go cue (`go_start_times`) exists for exactly one per trial in all sessions." The trials table is therefore used directly as the trial definition, and the assert is left in the conversion code as a per-session guard.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both aimed at trials that are not valid behaviour/neural data, plus a session-level minimum:

- **Reward not contingent on the animal's action**: `auto_water == 1` or `free_water == 1` trials are removed, mirroring the reference code's `get_regular_trial_mask`. (3,789 trials.)
- **No ephys at all**: after binning, trials whose total spike count is zero across *every* good unit over the whole 4 s window are removed as ephys-acquisition gaps. (1,657 trials, across 100 sessions; two sessions lose 376/582 and 321/480 trials because the probes stopped long before the behaviour.)
- A session is dropped if fewer than 2 trials survive either filter (the target format's requirement); in practice no session was dropped on this ground.

Explicitly **kept** are early-lick, `ignore`, `miss` and photostimulation trials, all of which `get_regular_trial_mask` would remove, because the decoder task defines them as required outputs/inputs. Net: 94,990 → 89,544 trials (5.7 % removed).

ii.
```python
        # trial curation: drop auto-water / free-water trials (reference get_regular_trial_mask);
        # early-lick, ignore/miss and photostim trials are KEPT because they are decoder outputs
        # / inputs in this task.
        keep = (auto_water == 0) & (free_water == 0)
        n_dropped_water = int((~keep).sum())
        kidx = np.where(keep)[0]
        ntr = len(kidx)
        if ntr < 2:
            return {'identifier': ident, 'dropped': 'fewer than 2 usable trials'}
```

```python
        # --- exclude trials with no ephys at all (acquisition gaps).
        # In some sessions the probes stop before the behaviour does, or single trials are
        # missing from the ephys stream; such trials contain zero spikes across *every* good
        # unit in the whole analysis window.  They carry no neural information and would be
        # pure noise for the decoder, so they are treated as invalid data periods and removed.
        spikes_per_trial = fr.sum(axis=(0, 2))
        rec = spikes_per_trial > 0
        n_dropped_norec = int((~rec).sum())
        if n_dropped_norec:
            fr = fr[:, rec, :]
            kidx = kidx[rec]
            ntr = int(rec.sum())
        if ntr < 2:
            return {'identifier': ident, 'dropped': 'fewer than 2 trials with ephys'}
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 4: "**Trial curation**: drop `auto_water == 1` or `free_water == 1` trials (reward delivered independent of the animal's action ⇒ `outcome` is not a behavioural report), matching `get_regular_trial_mask`. **Keep** early-lick, ignore/miss and photostim trials, because the decoder task defines them as outputs/inputs; the reference excluded them only for its encoding analyses." Step 6: "Issue found and fixed during development: **trials with no ephys at all**. In 100 sessions some trials (4.35 % overall; 2 sessions where the probes stop long before the behaviour: 376/582 and 321/480 trials) contain zero spikes from *every* good unit over the whole trial. NWB does not flag them (`is_good_trials` is all-True and `obs_intervals` simply mirror the trials table). Such trials are excluded as invalid data periods." The trajectory (steps 46–48, 96–97) shows the AI first inspected `obs_intervals` and `is_good_trials`, concluded they did not flag these trials in the sessions it examined, and so derived the filter empirically from the binned spike counts. The AI also documents in Step 4 that trials shorter than the window are *not* dropped: "Dropping short trials is not an option (it would delete 95 % of `miss` trials and destroy the outcome output)"; instead the coverage fractions (96.8 % of trials cover go−2.5 s, 84.4 % cover go+1.5 s) are reported in metadata.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (sorted spike times on the session-absolute clock), restricted to units with `units/classification == 'good'`, together with `BehavioralEvents/go_start_times`, which supplies the per-trial alignment time that places the bin edges. No other neural representation is used.

ii.
```python
        units = nwb.units
        classification = np.asarray(units['classification'][:]).astype(str)
        good = np.where(classification == 'good')[0]
```

```python
        spike_lists = [np.asarray(units['spike_times'][int(i)]) for i in good]
        timing['spike_read'] = time.time() - t0
        t0 = time.time()
        fr = bin_spikes(spike_lists, go[kidx])                    # (n_neurons, ntr, NBINS)
```

```python
        go = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`units.spike_times` (session clock) for units with `classification == 'good'` → `neural[session][trial]` (n_neurons, 80)". Spike times are the only neural signal in the file (Step 2 lists the units table contents); the electrophysiology is Neuropixels spike-sorted output, so no ΔF/F or similar processing applies.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each good unit, the absolute bin edges of *all* trials are built as one flat array (`go[:, None] + BIN_EDGES_REL[None, :]`), a single `np.searchsorted` gives the running spike count at every edge, and `np.diff` along the bin axis converts those running counts into a spike count per bin. Counts are divided by the 50 ms bin width to give Hz, stored `float32`. No smoothing, baseline subtraction, normalisation, or firing-rate threshold is applied.

ii.
```python
def bin_spikes(spike_times_list, go_times):
    """Spike counts -> firing rate (Hz) in non-overlapping BIN_SIZE bins around the go cue.

    Analogue of the reference `sliding_histogram(..., rate=True)` with stride == bin width.
    """
    ntr = len(go_times)
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
    out = np.empty((len(spike_times_list), ntr, NBINS), dtype=np.float32)
    for k, st in enumerate(spike_times_list):
        idx = np.searchsorted(st, edges).reshape(ntr, NBINS + 1)
        out[k] = np.diff(idx, axis=1)
    out /= np.float32(BIN_SIZE)
    return out
```

```python
        neural_trials = [np.ascontiguousarray(fr[:, i, :]) for i in range(ntr)]
```

iii. CONVERSION_NOTES.md Step 1: the reference `sliding_histogram(spikeTimes, begin, end, bin_width, stride, rate=True)` "returns … firing **rates** (counts / bin_width)"; Step 6: "`bin_spikes()` — vectorised analogue of the reference `sliding_histogram(..., rate=True)` with stride = width = 50 ms: per neuron a single `np.searchsorted` over all trial-bin edges, then `np.diff`; rates = counts / 0.05 s." Step 5 Key Decision 2 explains that the method paper's ≥ 2 Hz firing-rate filter and "≥ 10 neurons per area per session" rule are *not* applied: "these apply to their neuron-wise *encoding* analyses; for population decoding, discarding low-rate neurons throws away usable signal and is not part of the dataset-level curation." The units of the result (Hz) are recorded in metadata (`'neural_units': 'firing rate (Hz), spike counts per 50 ms bin / 0.05 s'`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept; every other unit is discarded. No thresholds are placed on any individual quality metric (`isi_violation`, `presence_ratio`, `amplitude_cutoff`, `drift_metric`, …), and the older Kilosort-level `unit_quality` (`good`/`multi`) column is not used. A session with zero such units is dropped entirely. This retains 69,453 of 272,227 clusters (25.5 %), mean 401/session, range 90–923.

ii.
```python
        units = nwb.units
        classification = np.asarray(units['classification'][:]).astype(str)
        good = np.where(classification == 'good')[0]
        if len(good) == 0:
            return {'identifier': ident, 'dropped': 'no good units'}
        anno = np.asarray(units['anno_name'][:]).astype(str)[good]
```

```python
            'neuron_curation': (
                "units with units.classification == 'good' (output of the region-specific QC "
                'classifiers of Chen, Liu et al. 2023 white paper, the same good-unit lists used '
                'by the reference pipeline). These units also all carry a CCF annotation, which '
                "reproduces the reference's requirement of joint ephys+histology."),
```

iii. CONVERSION_NOTES.md Step 3: "five region-specific logistic-regression classifiers (cortex, striatum, thalamus, midbrain, medulla) trained on manual Phy labels; units labelled **`good`** are used for all analyses. In the NWB files this classifier output is the `units.classification` column … The whitepaper explicitly argues that thresholding individual metrics is inadequate — hence using the provided classifier label is the faithful reproduction of their QC." Step 4 discrepancy table: "`classification == 'good'` **is** the shipped output of that classifier. Using it reproduces the reference QC exactly: 69,453 good units = 25.5 % of 272,227 clusters (paper: 69,943 units, 25.9 %)." Step 5 Key Decision 1 adds that unlabelled units also lack CCF annotations, "so this simultaneously enforces the reference's 'must have histology' rule." The choice was validated by per-area counts: striatum 7,664, midbrain 7,495, thalamus 12,808 and medulla 2,928 reproduce the published numbers exactly (Step 9/Step 12 iteration 3).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB times (spike times, event timestamps, camera timestamps) are on the same session-absolute clock, so no resampling or per-stream offset correction is required. Alignment is done by adding the fixed go-cue-relative edge grid to each trial's go-cue time (`go_start_times`, one per trial) and binning the raw spikes against those absolute edges. `metadata['temporal_alignment_event']` records the choice.

ii.
```python
OFF_START = -2.5          # s relative to go cue
OFF_END = 1.5             # s relative to go cue
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80
BIN_EDGES_REL = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

```python
        go = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
```

```python
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
    out = np.empty((len(spike_times_list), ntr, NBINS), dtype=np.float32)
    for k, st in enumerate(spike_times_list):
        idx = np.searchsorted(st, edges).reshape(ntr, NBINS + 1)
        out[k] = np.diff(idx, axis=1)
```

```python
            'temporal_alignment_event': 'auditory go cue onset (BehavioralEvents go_start_times)',
            'off_start': OFF_START,
            'off_end': OFF_END,
```

iii. CONVERSION_NOTES.md Step 3: "**Alignment**: everything aligned to the **go cue** (t = 0) in both papers; reference epochs sample [−1.85, −1.2] s, delay [−1.2, 0] s, response [0, 1.5] s." Step 4: "spike/behaviour times minus `go_start_times` … ✓ same" as the reference, which exported spike times already aligned to the go cue. Verification: the `--show-processing` plots overlay a raw spike raster on the binned rate image for the same trial and overlay an independently computed raw-spike PSTH on the converted `neural` mean — Step 7 reports "the two curves are indistinguishable — no temporal shift" — and Step 10 Check 2 re-derived 6 random (trial, neuron, bin) rates per session from the raw NWB with independent code and matched them with `np.allclose`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, giving 80 non-overlapping bins spanning −2.5 s to +1.5 s relative to the go cue, identical for every trial and every session. The grid is defined once at module level (81 relative edges, 80 relative centres) and reused everywhere, so the neural, input and output streams all share the same bins. There is no rebinning of an already-binned product: spikes are binned once directly from raw spike times, and the video (native dt = 3.4 ms) is averaged once into the same 50 ms grid. The bin width is exported as `metadata['time_bin_size'] = 50.0` (ms), with the bin centres also stored.

ii.
```python
BIN_SIZE = 0.05           # s, 50 ms bins (decoder task specification)
OFF_START = -2.5          # s relative to go cue
OFF_END = 1.5             # s relative to go cue
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80
BIN_EDGES_REL = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

```python
            'time_bin_size': BIN_SIZE * 1000.0,
            'time_bin_size_units': 'ms',
            ...
            'bin_centers_re_go_cue_s': BIN_CENTERS_REL.tolist(),
```

iii. CONVERSION_NOTES.md Step 3: the method paper "binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms", but "Our task specifies **50 ms bins**, which we implement as non-overlapping 50 ms bins giving rates in Hz (80 bins over −2.5→+1.5 s)". Step 4: "Task specifies **50 ms bins** → non-overlapping 50 ms bins, rates in Hz (counts/0.05 s), 80 bins for −2.5→+1.5 s", and Step 5 Key Decision 5: "Bin *i* covers [−2.5 + 0.05 i, −2.45 + 0.05 i); the value assigned to input/output time series uses the **bin centre**." Step 10 Check 5 verifies the half-open-bin convention: "`searchsorted` with `side='left'` makes bins half-open, so no spike is counted twice — verified by summing counts over bins and comparing with a direct count in [−2.5, 1.5) (equal)." The verifier confirms `T: min 80, max 80` over the whole dataset.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (the onsets of the sample/tone epoch), `trials.start_time` (used to assign each sample event to a trial), and `go_start_times`. For each trial the tone taken is the **last** `sample_start` that belongs to that trial and occurs at or before its go cue, because an early lick replays the sample epoch and a trial can therefore carry several sample-start events.

ii.
```python
def tone_onset_times(sample_starts, trial_starts, go_times):
    """Last sample(tone)-epoch start at or before the go cue, for each trial.

    Early licking triggers a replay of the sample/delay epoch, so a trial can contain several
    `sample_start_times`; the *last* one before the go cue is the tone the animal responded to.
    """
    ntr = len(go_times)
    out = np.full(ntr, np.nan)
    if len(sample_starts):
        tidx = np.searchsorted(trial_starts, sample_starts, side='right') - 1
        ok = (tidx >= 0) & (tidx < ntr)
        tidx, ev = tidx[ok], sample_starts[ok]
        ok2 = ev <= go_times[tidx]
        tidx, ev = tidx[ok2], ev[ok2]
        order = np.argsort(ev, kind='stable')          # ascending -> last write wins
        out[tidx[order]] = ev[order]
    return out
```

```python
        sample_starts = np.asarray(be['sample_start_times'].timestamps[:], dtype=float)
        ...
        tone = tone_onset_times(sample_starts, trial_start, go)[kidx]
```

iii. CONVERSION_NOTES.md Step 2: "`sample_*` and `delay_*` have **more** events than trials because early licking triggers a **replay** of the sample/delay epoch (confirmed: extra sample events occur only in `early` trials)"; "Last `sample_start` before the go cue is ~1.85–1.95 s before the go cue (tone onset); this matches the paper's sample epoch start at −1.85 s and 1.2 s delay." Step 4: "Early-lick trials trigger epoch *replays*, so the **last** sample start before the go cue is the tone the animal actually used." Step 5 mapping: "median = 1.85 s before the go cue in every session; using the **last** sample start handles early-lick replays." The verifier reports the median tone onset at exactly −1.850 s re go cue across all 173 sessions.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone time is expressed relative to the trial's go cue (`tone_rel_go = tone − go`, typically ≈ −1.85 s), and each bin's value is its go-cue-relative centre minus that offset, i.e. seconds elapsed since tone onset at that bin centre. The result is a continuous, time-varying `float32` row of length 80, stored as row 0 of the per-trial `(2, 80)` input array. Values run from ≈ −0.65 s (bin before the tone on a normal trial) upward; the full converted range is [−1.525, 5.936] s, with the large values coming from early-lick replay trials where the final tone is close to the go cue but earlier bins reach back before an earlier replay.

ii.
```python
        tone = tone_onset_times(sample_starts, trial_start, go)[kidx]
        tone_rel_go = tone - go[kidx]                             # negative, ~ -1.85 s
        # time since tone onset at each bin centre (s)
        time_from_tone = (BIN_CENTERS_REL[None, :] - tone_rel_go[:, None]).astype(np.float32)
```

```python
        input_trials = [np.stack([time_from_tone[i], photostim[i]]).astype(np.float32)
                        for i in range(ntr)]
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "last `sample_start_times` ≤ go cue, per trial → `input[0]` = `time_from_tone_onset_s`; bin-center time − tone-onset time (seconds), continuous, time-varying (80 values)". Step 5 Key Decision 5 fixes the bin-centre convention for input/output time series. Planned sanity check: "`input[0]` (time from tone onset) ≈ −0.65 … 3.35 s with median tone onset at −1.85 s re go cue"; Step 9 reports the achieved range [−1.53, 11.9] s (the max in the pickle summary is 5.936 s; the verifier reports 11.9 s) and explains it as "long values come from early-lick replay trials". Step 10 Check 2 verified the full 80-bin time-from-tone vector against independently recomputed values for 4 random trials in each of 4 sessions.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is constructed *on* the neural bin grid: the same module-level `BIN_CENTERS_REL` array that defines the 80 spike bins is what is shifted by the tone-to-go-cue gap. Bin *k* of the input therefore refers to exactly the interval whose spikes are in bin *k* of `neural`, by construction — no interpolation or index matching is needed.

ii.
```python
BIN_EDGES_REL = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

```python
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()   # neural bins
```

```python
        time_from_tone = (BIN_CENTERS_REL[None, :] - tone_rel_go[:, None]).astype(np.float32)
```

iii. Implicit in Step 5 Key Decision 5 ("go-cue aligned, −2.5 → +1.5 s, 80 × 50 ms non-overlapping bins … the value assigned to input/output time series uses the **bin centre**"), i.e. a single shared grid for all streams. The `--show-processing` panel (1,1) plots `time_from_tone` for an example trial together with a vertical marker at the raw tone-onset time, and Step 7 confirms "the go cue sits exactly at t = 0 and the tone marker at −1.85 s", so the zero-crossing of the input lands on the raw event time.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the event streams `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times` (absolute session-clock timestamps of the laser on/off), with `trials.start_time` used to assign each laser event to a trial and `go_start_times` used to re-express it relative to the go cue. The trials-table columns (`photostim_onset`, `photostim_power`, `photostim_duration`, strings with `'N/A'` when unstimulated) are read only as a cross-check: `photostim_power != 'N/A'` is compared against the derived binary input and the mismatch count is reported per session.

ii.
```python
        stim_flag_tbl = np.asarray(tr['photostim_power'][:]).astype(str) != 'N/A'
        ...
        stim_on = np.asarray(be['photostim_start_times'].timestamps[:], dtype=float)
        stim_off = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=float)
```

```python
        ps_all = photostim_binary(stim_on, stim_off, trial_start, go)
        photostim = ps_all[kidx]
```

```python
        stim_flag_evt = photostim.max(axis=1) > 0
        stim_mismatch = int(np.sum(stim_flag_evt != stim_flag_tbl[kidx]))
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`BehavioralEvents['photostim_start_times' / 'photostim_stop_times'].timestamps` → `input[1]` = `photostim_on` … `stimulation[:,2:] -= gocue_time` (reference stores on/off times re go cue) … cross-checked against the `trials.photostim_onset/power/duration` columns." The AI verified the stimulation parameters against the papers (Step 4: "18,588/94,990 trials (19.6 %) at 5.5 mW, always 0.5 s; onset −1.2 s or −0.5 s re go cue" vs "~25 % randomly interleaved trials, 5 mW/hemisphere, last 0.5 s of delay"; "Stim always **ends before the go cue** ✓ (max stim end = go − 0.499 s)"). The 21 remaining table/event mismatches were investigated in Step 9: "in those trials the laser fired during an *aborted* (replayed) delay epoch, ending before −2.5 s relative to the final go cue, i.e. outside the exported window. The conversion is correct; no change needed."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary, time-varying trace on the same 80-bin grid. Each laser event is assigned to the trial whose `start_time` last precedes it; the on/off times are converted to go-cue-relative seconds; a bin is set to 1 if the stimulation interval **overlaps** the bin interval (`bin_start < off` and `bin_end > on`), 0 otherwise. Trials with no laser event stay all-zero. A defensive swap handles any entry where the stop time precedes the start time. Stored as `float32` row 1 of the `(2, 80)` input array.

ii.
```python
def photostim_binary(stim_on, stim_off, trial_starts, go_times):
    """(n_trials, NBINS) binary array: is photostimulation on during this bin?"""
    ntr = len(go_times)
    ps = np.zeros((ntr, NBINS), dtype=np.float32)
    if len(stim_on) == 0:
        return ps
    tidx = np.searchsorted(trial_starts, stim_on, side='right') - 1
    for i, on, off in zip(tidx, stim_on, stim_off):
        if i < 0 or i >= ntr:
            continue
        on_rel, off_rel = on - go_times[i], off - go_times[i]
        if off_rel < on_rel:                      # guard against corrupt entries
            on_rel, off_rel = off_rel, on_rel
        # a bin is 'on' if the stimulation interval overlaps the bin
        overlap = (BIN_EDGES_REL[:-1] < off_rel) & (BIN_EDGES_REL[1:] > on_rel)
        ps[i, overlap] = 1.0
    return ps
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "1 if the 50 ms bin overlaps [stim_on, stim_off] (times re go cue), else 0; binary time-varying". The instructions require "Whether **photostimulation** is on at every time point (discrete, time-varying)" and "If an input is a time such as onset of some stimulus, represent it as a binary time series", which the AI cites in Step 5 Key Decision 9. Planned sanity check: "`input[1]` on only before the go cue, total duration 0.5 s per stim trial, and 0 for all non-stim trials" — confirmed in Step 9 (20.0 % of trials stimulated; laser always ends before the go cue) and in the processing plots ("Photostim input turns on exactly over the raw `photostim_start/stop` shading and always ends before the go cue"). Step 10 Check 2 rebuilt the complete `(n_trials × 80)` photostim matrix from the raw events for 4 sessions with independent code and matched it exactly.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. By converting the laser on/off times into go-cue-relative seconds (`on − go[i]`, `off − go[i]`) and comparing them against the same `BIN_EDGES_REL` array that defines the spike bins. Bin *k* of `photostim_on` therefore covers precisely the interval of bin *k* of `neural`.

ii.
```python
        on_rel, off_rel = on - go_times[i], off - go_times[i]
        ...
        overlap = (BIN_EDGES_REL[:-1] < off_rel) & (BIN_EDGES_REL[1:] > on_rel)
```

iii. Same single-clock / single-grid reasoning as for the other streams (Step 4: "spike/behaviour times minus `go_start_times` ✓ same" as the reference). The correctness of the alignment is shown directly in the `--show-processing` panel (1,1), which shades the raw `[photostim_start, photostim_stop]` interval behind the converted binary trace; Step 7: "Photostim input turns on exactly over the raw `photostim_start/stop` shading."

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the NWB trials table, so choice is derived from two columns: `trials.trial_instruction` (`'left'`/`'right'`, the side the tone instructed) and `trials.outcome` (`'hit'`/`'miss'`/`'ignore'`). A `hit` means the animal licked the instructed side, a `miss` means it licked the other side, and an `ignore` means it did not lick within the response window. The AI additionally validated this derivation against the raw `left_lick_times`/`right_lick_times` event streams.

ii.
```python
        outcome = np.asarray(tr['outcome'][:]).astype(str)
        instruction = np.asarray(tr['trial_instruction'][:]).astype(str)
```

```python
        out_k = outcome[kidx]
        instr_k = instruction[kidx]
        # choice: hit -> instructed side, miss -> opposite side, ignore -> no lick
        choice = np.where(out_k == 'ignore', 2,
                          np.where(out_k == 'hit',
                                   np.where(instr_k == 'left', 0, 1),
                                   np.where(instr_k == 'left', 1, 0))).astype(np.int64)
```

iii. CONVERSION_NOTES.md Step 4 discrepancy table, "Choice definition": the reference code uses `trial_type` (instruction) + `correctness`; in the NWB, "`outcome` + `trial_instruction` reproduces the lick-derived choice in >99 % of trials (disagreements are `ignore` trials with a lick after the response window)"; resolution: "Choice := instructed side for `hit`, opposite side for `miss`, `no lick` for `ignore`." The check was run in `/app/cache/choice_check.py` (trajectory / Step 4). Step 5 mapping table repeats: "verified to agree with the first post-go lick in > 99 % of trials."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick` (the order given in the Decoder Task), and written as row 0 of the per-trial `(4, 80)` integer output array, with the single per-trial value broadcast across all 80 bins so that all four outputs share one time-varying array. `output_values[0] = ['left', 'right', 'no lick']` names the codes. Stored `int64`.

ii.
```python
OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ...
]
```

```python
        output_trials = [np.stack([np.full(NBINS, choice[i]),
                                   np.full(NBINS, outcome_code[i]),
                                   np.full(NBINS, early_code[i]),
                                   tongue_cls[i]]).astype(np.int64) for i in range(ntr)]
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "values 0 = left, 1 = right, 2 = no lick; constant over the trial's 80 bins", and Key Decision 9: "**Outputs are time-varying** (shape (4, 80)): per-trial variables are broadcast across bins, as recommended by the task description" (the instructions say "Can be time-varying or discrete values per trial. If at all possible, make it time-varying"). The resulting distribution (left 0.429 / right 0.422 / no lick 0.148) was checked against the planned sanity check "Choice distribution ≈ left 45 % / right 45 % / no lick 10 %" and against the raw survey in Step 9 (✓).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `trials.outcome`, which already stores exactly the three strings the decoder task asks for: `'ignore'`, `'miss'`, `'hit'`. No derivation is needed.

ii.
```python
        outcome = np.asarray(tr['outcome'][:]).astype(str)
        ...
        out_k = outcome[kidx]
```

iii. CONVERSION_NOTES.md Step 2 lists `outcome` (`hit`/`miss`/`ignore`) as a trials-table column, and Step 5 maps it straight through: "`trials.outcome` → `output[1]` = `outcome` … `correctness` (−1/0/1) — identical encoding" to the reference code's variable. Raw distribution from the Step 2 survey: hit 65,254 (68.7 %), miss 15,641 (16.5 %), ignore 14,095 (14.8 %).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to `0 = ignore`, `1 = miss`, `2 = hit` (the order given in the Decoder Task) with a nested `np.where`, and written as row 1 of the output array, repeated across all 80 bins. `output_values[1] = ['ignore', 'miss', 'hit']`.

ii.
```python
        outcome_code = np.where(out_k == 'ignore', 0, np.where(out_k == 'miss', 1, 2)).astype(np.int64)
```

```python
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ...
]
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "0 = ignore, 1 = miss, 2 = hit (order given in the task)". Per Key Decision 9 the per-trial value is broadcast across bins. Consistency was checked in Step 9: converted distribution hit 0.685 / miss 0.167 / ignore 0.148 matches the raw NWB survey (0.687 / 0.165 / 0.148) and the implied performance hit/(hit+miss) = 80.4 % is close to the paper's 84 % (the AI attributes the 3 pp gap to the paper further restricting sessions/trials).

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `trials.early_lick` column, which holds the strings `'early'` and `'no early'`.

ii.
```python
        early = np.asarray(tr['early_lick'][:]).astype(str)
```

iii. CONVERSION_NOTES.md Step 2 lists `early_lick` (`early`/`no early`) among the trials-table columns and Step 5 maps it directly ("`trials.early_lick` → `output[2]` = `early_lick`", reference counterpart `behavior_early_report`). The Step 2 survey gives 10,805 early trials (11.4 %) out of 94,990.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A boolean comparison `early == 'early'` cast to integer, giving `0 = no`, `1 = yes` (the order given in the Decoder Task), written as row 2 of the output array and repeated across all 80 bins. `output_values[2] = ['no', 'yes']`.

ii.
```python
        early_code = (early[kidx] == 'early').astype(np.int64)
```

```python
OUTPUT_VALUES = [
    ...
    ['no', 'yes'],
    ...
]
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`trials.early_lick` → `output[2]` = `early_lick`; 0 = no, 1 = yes"; broadcast across bins per Key Decision 9. The converted rate (11.6 %) matches the raw survey (11.4 %) in Step 9. The AI also recorded that early-lick trials must be *kept* despite the data paper excluding them, because early lick is a required decoder output (Steps 3–5), and noted in Step 4 that the lick that sets the flag falls in the sample/delay epoch, i.e. inside the −2.5 s window.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: a DeepLabCut side-view tracking series with `data` of shape `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` and explicit session-clock `timestamps` at dt = 3.4 ms (~294 Hz). Column 1 is the y-position used as the value; column 2 (likelihood) determines whether the tongue is visible in that frame. The go-cue times supply the per-trial window.

ii.
```python
        bts = nwb.acquisition['BehavioralTimeSeries'].time_series
        tongue = bts['Camera0_side_TongueTracking']
        tt = np.asarray(tongue.timestamps[:], dtype=float)
        tdata = np.asarray(tongue.data[:], dtype=float)
        ty_binned = bin_tongue(tt, tdata[:, 1], tdata[:, 2], go[kidx])
```

iii. CONVERSION_NOTES.md Step 2: "`nwb.acquisition['BehavioralTimeSeries']`: DeepLabCut side-view (`Camera0_side_`) tracking `JawTracking`, `NoseTracking`, `TongueTracking` (all 174 sessions) … Each has `data` of shape (n_frames, 3) = (x, y, likelihood) and explicit `timestamps` in session time with **dt = 0.0034 s** (identical in all 174 sessions)." This is the only tongue measurement in the file and is present in every session. Step 1 identifies the reference counterpart `Sherlock/align_markers.py::align_markers_between_lims`, which aligns the same DLC side-camera markers (`nose/tongue/jaw/whisker` x,y) to the go cue at dt = 0.0034 s. Step 3 confirms the cameras run at 300 Hz per the data paper.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps:
1. **Visibility mask** — frames with `likelihood > 0.5` count as visible (the tracker still emits a position when the tongue is retracted).
2. **Outlier rejection** — among the visible frames, the inter-frame velocity `dy/dt` is computed and frames following a jump larger than 5 σ are invalidated, following the method paper's 5-sigma velocity outlier rule.
3. **Binning** — for each trial, the valid frames inside `[go − 2.5, go + 1.5)` are averaged (via `np.bincount` sum/count) into the same 80 × 50 ms bins; a bin with no valid frame is left `NaN`.
4. **Discretisation** — per session, the 40th and 60th percentiles of *all finite binned values* are taken and each bin is assigned class 0/1/2; `NaN` bins become class 3, "not visible".

No mean-imputation is done for occluded frames (the method paper imputes the mean), because the task requires an explicit "not visible" category.

ii.
```python
TONGUE_LIK_THRESH = 0.5   # DeepLabCut likelihood above which the tongue is 'visible'
VELOCITY_SIGMA = 5.0      # 5-sigma velocity outlier rejection (method paper)
```

```python
def bin_tongue(ts, y, lik, go_times):
    """Mean visible tongue y-position per 50 ms bin, NaN where the tongue is not visible."""
    ntr = len(go_times)
    visible = lik > TONGUE_LIK_THRESH
    valid = visible.copy()
    if visible.sum() > 10:
        iv = np.where(visible)[0]
        dy = np.diff(y[iv]) / np.maximum(np.diff(ts[iv]), 1e-6)
        s = np.std(dy)
        if s > 0:
            bad = np.abs(dy) > VELOCITY_SIGMA * s
            # a large jump invalidates the frame after the jump
            valid[iv[1:][bad]] = False
    res = np.full((ntr, NBINS), np.nan, dtype=np.float32)
    t0 = go_times + OFF_START
    lo = np.searchsorted(ts, t0)
    hi = np.searchsorted(ts, go_times + OFF_END)
    for i in range(ntr):
        a, b = lo[i], hi[i]
        if b <= a:
            continue
        tsel = ts[a:b]
        vsel = valid[a:b]
        if not vsel.any():
            continue
        bi = np.floor((tsel[vsel] - t0[i]) / BIN_SIZE).astype(int)
        np.clip(bi, 0, NBINS - 1, out=bi)
        cnt = np.bincount(bi, minlength=NBINS)
        sm = np.bincount(bi, weights=y[a:b][vsel], minlength=NBINS)
        nz = cnt > 0
        res[i, nz] = (sm[nz] / cnt[nz]).astype(np.float32)
    return res
```

iii. CONVERSION_NOTES.md Step 3: "**Video markers**: DeepLabCut side-view tongue/jaw/nose; outliers detected by 5-sigma velocity threshold and imputed from nearby frames; when the tongue is occluded (in the mouth) the method paper sets tongue position to its mean value. Our decoder task instead requires an explicit 'not visible' class (class 3), so occlusion is represented as its own category (detected via DLC likelihood)." Step 5 Key Decision 7: "**Tongue visibility threshold**: `likelihood > 0.5`. The likelihood distribution is strongly bimodal (89 % < 0.1, 10 % > 0.9) so the exact threshold is immaterial." Step 5 Key Decision 8: "**Percentiles computed per session** over the *binned, visible* tongue-y values of the exported trials, so the resulting class frequencies are exactly 40 / 20 / 40 % of visible bins in each session." (Note: the Step 5 mapping table says "median y of frames in that bin" while the code, its docstring, Step 6 and the README all say the bin **mean**; the code computes the mean.) Step 10 Check 2 recomputed the complete tongue class matrix from raw DLC data for 4 sessions with independent code: "✓ agreement 1.0000".

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles of the finite (visible) binned y-values are computed and used as the two class edges: `y < p40 → 0`, `p40 ≤ y ≤ p60 → 1`, `y > p60 → 2`; `NaN` (no visible frame in the bin) → `3`. The percentiles and the visible fraction are recorded per session in `session_info` (`tongue_p40`, `tongue_p60`, `frac_bins_tongue_visible`). Because the percentiles are taken over exactly the quantity being discretised, the visible bins split exactly 40/20/40 within each session; over the whole dataset the classes come out 0.101 / 0.050 / 0.101 / 0.749.

ii.
```python
def discretize_tongue(y_binned):
    """Per-session discretisation of tongue y into 4 classes (task specification)."""
    vis = np.isfinite(y_binned)
    cls = np.full(y_binned.shape, 3, dtype=np.int64)     # 3 = not visible
    if vis.sum() > 0:
        p40, p60 = np.percentile(y_binned[vis], [40, 60])
        v = y_binned[vis]
        c = np.where(v < p40, 0, np.where(v <= p60, 1, 2))
        cls[vis] = c
    else:
        p40 = p60 = np.nan
    return cls, float(p40), float(p60)
```

```python
OUTPUT_VALUES = [
    ...
    ['<40th pct', '40-60th pct', '>60th pct', 'not visible'],
]
```

iii. Directly from the Decoder Task specification ("0: < 40th percentile of y-position over the session; 1: 40th to 60th percentile; 2: > 60th percentile; 3: not visible"), recorded in CONVERSION_NOTES.md Step 5 Key Decision 8 and in the exported metadata: "per-session percentiles (40th, 60th) of the binned, visible tongue y-position; bins with no frame of likelihood > 0.5 (or only 5-sigma velocity outliers) are class 3 = 'not visible'". Planned sanity check: "Tongue class distribution: within-session visible bins split 40/20/40; class 3 dominates before the go cue and drops after it" — confirmed both numerically (Step 9) and visually in the `--show-processing` panels (2,0), (2,1) and (3,1), which draw the p40/p60 lines over the raw trace and plot the class time course.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as the spikes and the go cues, so alignment is a direct index lookup: `np.searchsorted` on the camera timestamps finds the frame range `[go − 2.5, go + 1.5)` for each trial, and each frame is assigned to bin `floor((t − (go − 2.5)) / 0.05)` — the identical go-cue-relative grid used for the firing rates (indices clipped into `[0, 79]` to guard floating-point edge cases). No interpolation or resampling is applied. Video is trial-gated (each trial's segment starts at that trial's `start_time`), so bins before the trial's video start simply contain no frames and become class 3.

ii.
```python
    t0 = go_times + OFF_START
    lo = np.searchsorted(ts, t0)
    hi = np.searchsorted(ts, go_times + OFF_END)
    for i in range(ntr):
        a, b = lo[i], hi[i]
        if b <= a:
            continue
        ...
        bi = np.floor((tsel[vsel] - t0[i]) / BIN_SIZE).astype(int)
        np.clip(bi, 0, NBINS - 1, out=bi)
```

iii. CONVERSION_NOTES.md Step 2 established that the tracking series carries "explicit `timestamps` in session time … segmented per trial: each trial's video segment starts exactly at that trial's `start_time`", and Step 1 records that the reference's `align_markers_between_lims` aligns the same markers to the go cue. Because all streams share the global clock and the same grid, bin *k* of the tongue output covers the same interval as bin *k* of the firing rates. Verified in the processing plots: Step 7 notes "Raw tongue y trace (visible frames) overlays the binned values; the 40th/60th percentile lines sit where the class boundaries change; class 3 dominates before the go cue and drops immediately after it", and Step 10 Check 2 reproduced the entire class matrix from the raw data with agreement 1.0000.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases, each handled explicitly and documented:

- **Session never quality-controlled** (`classification` is NaN for all units): `.astype(str)` turns the non-string entries into `'nan'`, no unit matches `'good'`, and the session is dropped with a printed reason. This is the single dropped session, `SC017_20190216_162508_s4`, and dropping it brings the count to the published 173.
- **Trials with no ephys** (acquisition gaps / probes stopped early): detected as zero total spikes across all good units in the window and removed (see 1-e); a session is dropped if fewer than 2 trials remain.
- **Trials shorter than the analysis window**: *not* dropped. Spikes are only recorded within `[start_time, stop_time]`, so 3.2 % of trials start after go − 2.5 s and 15.6 % end before go + 1.5 s; the uncovered bins are genuinely 0 spikes and are left as 0, with the per-session coverage fractions stored in `session_info` (`frac_trials_observed_at_start/end`) and the situation described in `metadata['notes_unobserved_time']`.
- **Frames with no tracked tongue, or velocity outliers**: excluded from the bin mean; a bin left with no valid frame becomes the explicit `'not visible'` class rather than being imputed.
- **Malformed photostim entries** and **per-file failures**: a stop-before-start laser interval is swapped rather than silently producing an empty mask, and `_worker` catches any exception per file, reports it with a traceback, and lets the rest of the conversion finish.

ii.
```python
        classification = np.asarray(units['classification'][:]).astype(str)
        good = np.where(classification == 'good')[0]
        if len(good) == 0:
            return {'identifier': ident, 'dropped': 'no good units'}
```

```python
        spikes_per_trial = fr.sum(axis=(0, 2))
        rec = spikes_per_trial > 0
        ...
        if ntr < 2:
            return {'identifier': ident, 'dropped': 'fewer than 2 trials with ephys'}
```

```python
        if off_rel < on_rel:                      # guard against corrupt entries
            on_rel, off_rel = off_rel, on_rel
```

```python
    cls = np.full(y_binned.shape, 3, dtype=np.int64)     # 3 = not visible
```

```python
def _worker(args):
    path, show = args
    try:
        return convert_session(path, show_processing=show)
    except Exception as e:  # pragma: no cover - defensive
        import traceback
        return {'identifier': os.path.basename(path), 'error': traceback.format_exc()}
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "**Unobserved time**: spikes exist only inside `[start_time, stop_time]` (= `obs_intervals`). 96.9 % of trials cover go−2.5 s, 84.4 % cover go+1.5 s (miss trials end early). Bins outside the recorded trial interval necessarily contain 0 spikes; these are kept as 0 (no extra information is available) and the fraction is reported in metadata." Step 4 explains why short trials are not dropped: "Dropping short trials is not an option (it would delete 95 % of `miss` trials and destroy the outcome output)." Step 10 Check 5 enumerates the edge cases checked (window edges, short trials, early-lick replays, photostim during aborted epochs, the zero-good-unit session, units without CCF annotation — "none among `good` units (checked over all 174 files)", and NaN/Inf — "no NaN/Inf in neural, input or output arrays"). The full-dataset verifier output confirms "Data format is valid, no errors or warnings."

## 10-a. What are the most time-consuming steps of the code?

i. The script instruments every stage and prints per-stage means. Per session: reading the ragged `spike_times` (0.40 s), binning the spikes (0.34 s), unit metadata + electrode/CCF lookup (0.09 s), tongue/output construction (0.08 s), trials and inputs (≈ 0 s) — so raw NWB I/O plus spike binning dominate, as expected. At whole-run level the two costs are the 174 per-session conversions (22.4 s wall clock across 24 worker processes; ≈ 0.9 s × 174 ≈ 160 s of CPU) and serialising the 11.89 GB pickle (18.7 s). Total 42 s, well inside the instructions' 15-minute budget.

ii.
```python
        t0 = time.time()
        spike_lists = [np.asarray(units['spike_times'][int(i)]) for i in good]
        timing['spike_read'] = time.time() - t0
        t0 = time.time()
        fr = bin_spikes(spike_lists, go[kidx])                    # (n_neurons, ntr, NBINS)
        timing['binning'] = time.time() - t0
```

```python
    tt_ = [i['timing'] for i in session_info]
    for k in tt_[0]:
        print('timing %-12s mean %.2f s/session' % (k, np.mean([t[k] for t in tt_])))
    print('total elapsed %.1f s' % (time.time() - t_all))
```

iii. CONVERSION_NOTES.md Step 6 "Efficiency" and Step 7 "Run Time Estimates" record the measured per-step costs and the speed-ups: "`searchsorted` binning over all trial edges at once — 0.16 s/session (vs ~30 s for a naive Python loop)"; "`VectorIndex`-based electrode lookup — 0.09 s/session (vs ~10 s using `units['electrodes'][i]`)"; "16-way process parallelism — ~16×". The estimate made from the 2-session sample ("< 5 minutes", scaled by ~1.3 for session size) was borne out by the 42 s full run reported in Step 9.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain, in decreasing order of cost:

- **Per-unit spike read**, `[np.asarray(units['spike_times'][int(i)]) for i in good]`: one ragged `VectorIndex` access (and therefore one HDF5 read) per unit, ~400 per session. This is the single largest per-session cost (0.40 s) and is genuinely avoidable — the underlying `spike_times` target dataset and its offsets could be read once and sliced in memory, exactly as the AI already does for the `electrodes` VectorIndex a few lines earlier.
- **Per-unit region mapping**, `[map_annotation(a) for a in anno]` followed by a per-unit f-string join: `map_annotation` walks up to ~400 substring tests per unit, and the dataset has only 293 distinct annotation strings, so a `np.unique` + per-unique-value lookup would remove almost all of the work.
- **Per-unit binning loop** in `bin_spikes`: this one cannot be collapsed further, because spike trains are ragged and `searchsorted` needs one sorted array; the trial dimension is already vectorised by flattening the edge array.
- **Per-event loop** in `photostim_binary` and **per-trial loop** in `bin_tongue`: both could be replaced by a single global bin index plus one `np.bincount`/broadcast comparison, but both are negligible (≈ 0 s and 0.08 s per session respectively).
- **Per-trial list comprehensions** that build `neural_trials` / `input_trials` / `output_trials`: unavoidable given the required list-of-trials output format, though the `np.full` calls per trial could be replaced with a single broadcast.

ii.
```python
        spike_lists = [np.asarray(units['spike_times'][int(i)]) for i in good]
```

```python
        groups = np.array([map_annotation(a) for a in anno])
        region_labels = np.array(['%s %s' % (s, g) for s, g in zip(side, groups)])
```

```python
    for k, st in enumerate(spike_times_list):
        idx = np.searchsorted(st, edges).reshape(ntr, NBINS + 1)
        out[k] = np.diff(idx, axis=1)
```

```python
    for i, on, off in zip(tidx, stim_on, stim_off):
```

```python
        output_trials = [np.stack([np.full(NBINS, choice[i]),
                                   np.full(NBINS, outcome_code[i]),
                                   np.full(NBINS, early_code[i]),
                                   tongue_cls[i]]).astype(np.int64) for i in range(ntr)]
```

iii. CONVERSION_NOTES.md Step 6 documents the loops the AI *did* vectorise ("binning is vectorised with `searchsorted` over concatenated bin edges", "Unit → electrode lookup uses the `VectorIndex` arrays directly instead of per-unit `units['electrodes'][i]` DataFrame construction (~100× faster; verified to give identical electrode ids)", "Video tracking arrays are read once per session and binned with `np.bincount`") and the decision to parallelise across sessions instead of optimising further. It does not identify the remaining per-unit `spike_times` read as a vectorisable loop; the AI's justification for stopping here is the measured total runtime: Step 7 concludes "Estimated full conversion: **< 5 minutes**, well under the 15-minute budget", so no further optimisation was pursued.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened exactly once and each session is processed once — there is no second pass over the data and no recomputation of the tongue percentiles (they are per-session and are computed inside the same pass). The redundancy that does exist is small:

- `tone_onset_times(...)` and `photostim_binary(...)` are computed over **all raw trials** and only then subset by `kidx`, so the work is done for the ~4 % of trials that were already excluded.
- `bin_spikes` is run on all water-filtered trials, and the no-ephys trials are removed afterwards (`fr = fr[:, rec, :]`), so ~1.8 % of the binning work is thrown away. This ordering is deliberate — the no-ephys criterion is *derived* from the binned rates — but means the rates are computed for trials that are then dropped.
- `map_annotation` re-derives the coarse group from scratch for every unit, repeating the same substring-rule chain for the same annotation string hundreds of times per session (293 distinct strings across the whole dataset).
- `nwb.electrodes.to_dataframe()` materialises the entire electrode table once per session in order to read three coordinate columns.

None of these is repeated across sessions or across passes; all are within-session inefficiencies.

ii.
```python
        tone = tone_onset_times(sample_starts, trial_start, go)[kidx]
        ...
        ps_all = photostim_binary(stim_on, stim_off, trial_start, go)
        photostim = ps_all[kidx]
```

```python
        fr = bin_spikes(spike_lists, go[kidx])                    # (n_neurons, ntr, NBINS)
        ...
        spikes_per_trial = fr.sum(axis=(0, 2))
        rec = spikes_per_trial > 0
        ...
            fr = fr[:, rec, :]
```

```python
        groups = np.array([map_annotation(a) for a in anno])
```

```python
        edf = nwb.electrodes.to_dataframe()
        ccf_x = edf['x'].values.astype(float)[elec_ids]
        ccf_y = edf['y'].values.astype(float)[elec_ids]
        ccf_z = edf['z'].values.astype(float)[elec_ids]
```

iii. The AI does not flag any of these as repeated work; CONVERSION_NOTES.md Step 6 presents the pipeline as a single pass with I/O-bound cost and states that after the speed-ups "the wall-clock cost is dominated by pickling the ~10 GB result", which is borne out by the 22.4 s conversion vs 18.7 s pickle split in `conversion_full_out.txt`. The computing-then-discarding of firing rates is an unavoidable consequence of its chosen (and documented) no-ephys trial criterion, which is defined on the binned rates themselves.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A modest amount:

- **`ccf` and `anno` are computed, returned, and never used.** `convert_session` stacks the three CCF coordinates into an `(n_neurons, 3)` array and returns it along with the raw annotation strings, but `main()` builds `data` only from `region_labels`; `r['ccf']` and `r['anno']` are never read. Because sessions run in a `ProcessPoolExecutor`, these arrays are also pickled and shipped back across the process boundary before being dropped. `ccf_y` and `ccf_z` are read from the electrodes table purely to fill this discarded field (only `ccf_x` is used, for the hemisphere split).
- **Diagnostic-only quantities**: `trial_stop`, `stim_flag_tbl`, `obs_frac_start/end`, `stim_mismatch`, `n_trials_raw`, `subject_name` and the per-stage `timing` dict. These do survive into `metadata['session_info']`, so they are documentation rather than waste, but none is used by the decoder.
- **`--show-processing` mode** recomputes a population PSTH from raw spike times with a nested Python loop over 60 neurons × 200 trials purely for the diagnostic figure. This is expensive but is explicitly what the instructions ask for, and it is off by default.
- **Output arrays are stored as `int64`** where the values are all in `[0, 3]`; an 8× smaller dtype would carry the same information (≈ 229 MB vs ≈ 29 MB). Not "processing" as such, but wasted work in serialisation.

Everything else computed — firing rates, both inputs, all four outputs, region labels, subject indices — ends up in the exported dictionary.

ii.
```python
        ccf_x = edf['x'].values.astype(float)[elec_ids]
        ccf_y = edf['y'].values.astype(float)[elec_ids]
        ccf_z = edf['z'].values.astype(float)[elec_ids]
```

```python
        result = {
            'identifier': ident,
            'subject': subject,
            'neural': neural_trials,
            'input': input_trials,
            'output': output_trials,
            'region_labels': region_labels,
            'ccf': np.stack([ccf_x, ccf_y, ccf_z], axis=1).astype(np.float32),
            'anno': anno,
            'info': info,
        }
```

```python
    data = {
        'neural': [r['neural'] for r in good],
        'input': [r['input'] for r in good],
        'output': [r['output'] for r in good],
        ...
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
```

```python
        output_trials = [np.stack([...]).astype(np.int64) for i in range(ntr)]
```

iii. The AI does not document `ccf`/`anno` as unused; they appear to be leftovers from the region-mapping validation work (Step 4/Step 12 iteration 3, where per-area unit counts were tuned against the paper's published numbers, and `/app/cache/region_map.py` was used for that). The diagnostics are justified in Step 6 and Step 9 as the basis for the consistency checks ("`convert_session()` … builds neural/input/output arrays and a `session_info` record with timings and QC statistics"; per-session coverage "is reported in session_info"), and `metadata['notes_unobserved_time']` quotes those numbers directly.
