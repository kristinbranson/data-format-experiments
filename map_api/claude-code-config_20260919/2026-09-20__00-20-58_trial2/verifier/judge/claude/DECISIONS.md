# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is the DANDI:000363 NWB release, one `.nwb` file per session under
`/app/data/sub-<subject_id>/`. The AI enumerates every session by walking the `sub-*`
sub-directories of `/app/data` (`list_session_files()`), sorting both the subject directories
and the files inside them, and keeping any file ending in `.nwb`. This finds all 174 files.
Each file is then opened exactly once with `pynwb.NWBHDF5IO(path, 'r', load_namespaces=True)`
inside `process_session()`, and everything for that session (`nwb.subject`, `nwb.trials`,
`nwb.units`, `nwb.acquisition['BehavioralEvents']`,
`nwb.acquisition['BehavioralTimeSeries']`, `nwb.electrodes`) is read from that one handle.
Sessions are distributed over a `multiprocessing.Pool` (spawn context, 16 workers by default),
with `pool.imap` preserving the input file order so the output session order is deterministic
and identical to the serial order. `--sample` takes `files[:2]`. A per-session exception is
caught in `_worker` so one bad file cannot kill the run. Total wall time for the full
conversion was 41 s (26 s conversion + 14 s pickle write).

ii.
```python
def list_session_files(data_dir=DATA_DIR):
    """Return the sorted list of NWB session files."""
    files = []
    for sub in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.nwb'):
                files.append(os.path.join(d, f))
    return files
```

```python
from pynwb import NWBHDF5IO
...
    io = NWBHDF5IO(path, 'r', load_namespaces=True)
    nwb = io.read()

    session_id = nwb.identifier
    subject = str(nwb.subject.subject_id)
    trials = nwb.trials
    units = nwb.units
    go_times = _events(nwb, 'go_start_times')
```

```python
    if args.nproc > 1 and len(files) > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(min(args.nproc, len(files))) as pool:
            for i, res in enumerate(pool.imap(_worker, jobs)):
```

iii. From CONVERSION_NOTES.md Step 2/Step 6: NWB is the published container for this dataset
and `pynwb` is mandated by the instructions (`h5py` is forbidden). Because the dandiset stores
exactly one session per file, the directory listing *is* the complete session list, so no
manifest or index is needed; sorting makes the ordering reproducible. The AI verified the
enumeration against the dandiset (174 files, 28 subjects, 659 electrode groups) and against
the papers (173 sessions, 655 insertions after dropping the one un-QC'd session). Parallelism
was added as an explicit Step 7 speed-up ("~10× overall"), since the per-session work is
completely independent.

## 1-b. How are the data split into subjects?

i. The subject of a session is read from the NWB file itself, `nwb.subject.subject_id`
(a numeric string such as `'440956'`), not from the enclosing directory name. At assembly the
AI takes the sorted set of unique subject ids as `subjects` and builds `subject_idx` as each
session's index into that list, in the same order as `neural`/`input`/`output`. This gives 28
subjects with 3–10 sessions each, matching the dandiset. The subject id is also repeated per
session in `metadata['session_info']`.

ii.
```python
        subject = str(nwb.subject.subject_id)
```

```python
    subjects = sorted({r['subject'] for r in results})
    subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
...
        'subjects': subjects,
        'subject_idx': subject_idx,
```

iii. `subject_id` is the canonical animal identifier stored in the file; the folder name
`sub-440956` is derived from it, so reading the field is authoritative and requires no
grouping heuristic. The AI notes in Step 2 that the numeric id differs from the mouse name in
the papers (session `SC015_20190207_120657_s1` ↔ subject `440956`), and keeps both: the
numeric id in `subjects`, the mouse-coded name in `session_id`/`session_info`.

## 1-c. How are the data split into sessions?

i. One NWB file is one session — no grouping or splitting is performed. Each session is
labelled by `nwb.identifier` (e.g. `SC015_20190207_120657_s1`, which encodes mouse, date, time
and session number), and the file basename, neuron count and trial counts are recorded in
`metadata['session_info']`. Session order in the output lists follows the sorted file order.
173 of the 174 files reach the output; `SC017_20190216_162508_s4` is dropped because it has no
quality-controlled units (see 2-c), and a session is also dropped if fewer than 2 trials
survive curation.

ii.
```python
        session_id = nwb.identifier
```

```python
            'session_info': [
                {'session_id': r['session_id'], 'subject': r['subject'], 'file': r['file'],
                 'n_neurons': int(r['n_neurons']), 'n_trials': int(r['n_trials']),
                 'n_trials_raw': int(r['n_trials_raw']), 'n_trials_observed': int(r['n_trials_observed']),
                 'tongue_pct_40_60': list(r['tongue_percentiles']),
                 'tongue_visible_frame_frac': r['tongue_visible_frac']}
                for r in results],
```

```python
        if len(good) == 0:
            io.close()
            return None
...
        if n_trials < 2:
            io.close()
            return None
```

iii. CONVERSION_NOTES Step 2/Step 4: the file boundary already *is* the session boundary in
this dandiset, so nothing has to be inferred. The AI used the session count as a consistency
check against the papers: 174 files, of which 173 have at least one `good` unit, and the
corresponding insertion count 659 − 4 = 655 — both matching the data paper exactly. That exact
double match is what convinced the AI to drop the un-QC'd session rather than keep it.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`), one row per behavioural trial. The AI
checks that `BehavioralEvents['go_start_times']` has exactly one timestamp per trials-table
row and raises a `RuntimeError` otherwise (verified true for all 174 files). It explicitly
does not use `sample_start_times` / `delay_start_times` for the 1:1 trial mapping, because an
early lick replays those epochs so they can have several entries per trial. A separate
`trial_index` array (index into the raw trials table) is retained per session so every kept
trial can be traced back to its original row.

ii.
```python
        trials = nwb.trials
        n_trials_raw = len(trials)
        instruction = np.asarray(trials['trial_instruction'][:])
        outcome = np.asarray(trials['outcome'][:])
        early_lick = np.asarray(trials['early_lick'][:])
        auto_water = np.asarray(trials['auto_water'][:]).astype(bool)
        free_water = np.asarray(trials['free_water'][:]).astype(bool)
        trial_start = np.asarray(trials['start_time'][:], dtype=np.float64)
        trial_stop = np.asarray(trials['stop_time'][:], dtype=np.float64)

        go_times = _events(nwb, 'go_start_times')
        if len(go_times) != n_trials_raw:
            io.close()
            raise RuntimeError(f'{session_id}: {len(go_times)} go cues for {n_trials_raw} trials')
```

iii. Step 10 Check 5, item 12: "`go_start_times` count vs. trials table: asserted equal for all
174 files." The trials table is the file's own definition of a trial, so the AI uses it
directly rather than re-deriving trial boundaries from event streams; the go-cue count check is
what makes the trial ↔ alignment-event mapping unambiguous. Step 5 decision 8 records that the
sample epoch is replayed on early-lick trials, which is why the sample-epoch stream is used
only to look up a tone time, never to count trials.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, applied in order, plus a session-level minimum:

1. **Ephys coverage** — only trials inside `units/obs_intervals` are kept. In 9 sessions the
   ephys recording covers only a subset of the behavioural trials (e.g.
   `SC015_20190208_133600_s2`: 160 of 480). The observed trials are located in the trials
   table by matching `obs_intervals[:, 0]` against `trials.start_time` with `searchsorted`,
   and the match is asserted with `np.allclose`. The AI explicitly rejected a "first N trials"
   shortcut because `SC026_20190807_134913_s20` is not a prefix. (−1,060 trials)
2. **`auto_water` and `free_water`** — dropped, reproducing the two clauses of the reference
   code's `get_regular_trial_mask` that do not conflict with the decoder spec. (−3,764)
3. **`units/is_good_trials`** — the per-insertion drift/stability annotation shipped in the
   file; a trial is kept only if it is flagged good for *every* retained unit. (−476)
4. **Trials with no spikes at all** across all (hundreds of) good units — the recording can
   stop part-way through the last annotated trial. (−2)

A session is dropped if fewer than 2 trials survive. Early-lick, `ignore` (no-response) and
photostimulation trials are deliberately **kept**, and no behavioural-performance filter is
applied at the session level. Net: 94,370 → 89,068 trials (−5.6%) across the 173 sessions.

ii.
```python
        obs = np.asarray(units['obs_intervals'][int(good[0])], dtype=np.float64)
        obs_trial = np.searchsorted(trial_start, obs[:, 0] - 1e-6)
        if not np.allclose(trial_start[obs_trial], obs[:, 0]):
            raise RuntimeError(f'{session_id}: obs_intervals do not match the trials table')

        # (a) reference `get_regular_trial_mask`: no auto-water, no free-water trials
        keep = (~auto_water[obs_trial]) & (~free_water[obs_trial])
        # (b) recording-stability annotation: keep only trials annotated 'good' for
        #     every retained unit (i.e. for every probe insertion contributing units)
        is_good_trials = np.asarray(units['is_good_trials'][:])[good]   # (n_good, n_obs)
        keep &= is_good_trials.all(axis=0)
        keep_idx = obs_trial[keep]
        n_trials = len(keep_idx)
        n_trials_obs = len(obs_trial)
        if n_trials < 2:
            io.close()
            return None
```

```python
        nonempty = fr.sum(axis=(0, 2)) > 0
        if not nonempty.all():
            n_empty = int((~nonempty).sum())
            print(f'  {session_id}: dropping {n_empty} trial(s) with no spikes at all',
                  flush=True)
            fr = fr[:, nonempty, :]
            keep_idx = keep_idx[nonempty]
            go = go[nonempty]
            n_trials = len(keep_idx)
```

iii. Step 4 and Step 10 Check 3: the reference `get_regular_trial_mask` is
`early_lick == 0 & auto_water == 0 & free_water == 0 & correctness != -1 & stimulation == 0`.
The AI reasoned that three of those five clauses directly conflict with the decoder
specification — removing early-lick trials would leave the `early_lick` output with one class,
removing `correctness == -1` would delete the `ignore` class of `outcome` and the `no lick`
class of `choice`, and removing stimulated trials would make the `photostim` input identically
zero — so those three are dropped and the conflict documented, while `auto_water` and
`free_water` are applied exactly as in the reference. `obs_intervals` is the file's own record
of which trials were recorded, so trials outside it carry no neural data at all and would
appear as 4 s of 0 Hz. `is_good_trials` is described in Step 3 as "the residue of [the] manual
[drift] annotation" at the penetration level from the QC white paper. The all-zero-spike drop
was found because it triggered an "all neural data is zero" warning from `train_decoder.py`.
The 2-trial minimum is the target format's stated requirement. The AI explicitly declined to
re-apply the papers' 65% behavioural-performance session criterion, on the grounds that the
session and insertion counts already match the paper exactly, so the released set *is* the
analysed set.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` — the sorted, session-absolute spike times of each unit, stored as a
ragged NWB `VectorData`/`VectorIndex` pair. The AI reads the flat buffer
(`units['spike_times'].target.data`) once per session and the per-unit end offsets
(`units['spike_times'].data`), then slices per unit. Only units with
`classification == 'good'` contribute. The second input is
`BehavioralEvents['go_start_times'].timestamps`, which places the bin edges.

ii.
```python
        sv = units['spike_times']
        ends = np.asarray(sv.data[:])
        starts = np.concatenate([[0], ends[:-1]])
        flat_spikes = np.asarray(sv.target.data[:], dtype=np.float64)

        n_neurons = len(good)
        fr = np.empty((n_neurons, n_trials, NBINS), dtype=np.float32)
        for i, u in enumerate(good):
            st = flat_spikes[starts[u]:ends[u]]
            fr[i] = bin_spikes_rate(st, go)
```

```python
def _events(nwb, name):
    """Timestamps of a `BehavioralEvents` TimeSeries, as a numpy array."""
    ts = nwb.acquisition['BehavioralEvents'].time_series[name]
    return np.asarray(ts.timestamps[:], dtype=np.float64)
```

iii. Step 5 variable mapping: `units.spike_times` (absolute) + `go_start_times.timestamps` →
`neural`, with the reference analogue being `process_one_area` + `sliding_histogram(rate=True)`.
Spike times are the only neural representation in the file. Step 6 records that reading
per-unit through `units['spike_times'][i]` is slow, so the whole flattened ragged dataset is
read once (~90 MB/session, 0.13 s) and sliced — a ~5× speed-up on the neural step.

## 2-b. How is the `neural` data processed?

i. Spike times are converted into per-bin **firing rates in Hz**. For each good unit the
per-trial bin edges are formed as `go[:, None] + BIN_EDGES[None, :]`, flattened, and located in
the unit's sorted spike train with a single `np.searchsorted(..., side='left')`; differencing
adjacent indices along the bin axis gives the spike count per bin, and dividing by 0.05 s gives
Hz. Bin *k* covers the half-open interval `[go + BIN_EDGES[k], go + BIN_EDGES[k+1])`. The
result is stored as `float32`. There is no smoothing, no normalisation, no baseline
subtraction, and no overlapping/sliding window. Bins that fall outside the recorded interval
are binned anyway and therefore read 0 Hz.

ii.
```python
def bin_spikes_rate(spike_times, go_times):
    """Spike counts of one unit in the fixed window around every go cue, as a rate.
    ...
        Bin k covers [go + BIN_EDGES[k], go + BIN_EDGES[k+1]).
    """
    edges = go_times[:, None] + BIN_EDGES[None, :]           # (n_trials, NBINS+1)
    idx = np.searchsorted(spike_times, edges.ravel(), side='left')
    idx = idx.reshape(edges.shape)
    counts = np.diff(idx, axis=1)
    return (counts / BIN_SIZE).astype(np.float32)
```

iii. Step 5 decision 3 and Step 10 Check 3(d): rates in Hz (counts / 0.05 s) reproduce the
reference `sliding_histogram(..., rate=True)`, which returns `binSpikes / bin_width`. The AI
documents that the counting rule, the rate normalisation and the treatment of the trial edges
(bin the whole fixed window regardless of when the trial ended, as `process_one_area` does) are
identical to the reference; only the bin width and stride differ, because the task
specification prescribes 50 ms non-overlapping bins instead of the reference's 40 ms/3.4 ms
sliding window. Sanity check 2 re-counted the rates bin-by-bin from the raw NWB with a
brute-force `np.sum((st >= lo) & (st < hi)) / 0.05` and passed with `np.allclose` on 5 sessions
× 20 trial×neuron probes; check 3 confirmed the per-neuron mean rates correlate at r = 0.988–
0.998 with the independent `units.avg_firing_rate` column.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept. No threshold is applied to any
individual quality metric, and in particular the method paper's "mean firing rate ≥ 2 Hz"
criterion is deliberately **not** applied. A session with zero good units is dropped entirely.
This retains 69,453 of the ~272 k units (mean 401.5 per session, range 90–923) across 173
sessions and 655 insertions.

ii.
```python
        units = nwb.units
        classification = np.asarray(units['classification'][:])
        good = np.flatnonzero(classification == 'good')
        if len(good) == 0:
            io.close()
            return None
```

```python
            'neuron_curation': (
                "units.classification == 'good' (region-specific logistic-regression QC "
                'classifiers of Chen, Liu et al. 2023 white paper); no firing-rate threshold'),
```

iii. Step 3/Step 5 decision 4: `classification` is the output of the region-specific
logistic-regression QC classifiers described in `ChenLiuEtAl2023_SpikeSortingQC.pdf`, and is
the same criterion as the reference code's `qc_mode='classifier'` unit index. The AI rejected
the method paper's ≥ 2 Hz cut because "it was introduced to stabilise per-neuron R² estimates
in the video→firing-rate regression, not for decoding, and discarding low-rate neurons would
throw away decodable information" — and the method paper itself states the result is
insensitive to that threshold. Consistency evidence in Step 4/Step 9: dropping the one session
with zero good units (`SC017_20190216_162508_s4`, whose `classification` column is NaN
throughout) yields exactly the paper's 173 sessions **and** exactly 655 insertions; the unit
total of 69,453 is 99.3% of the published 69,943, attributed to the DANDI release version
post-dating the paper snapshot.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go-cue onset**, `BehavioralEvents['go_start_times'].timestamps`, one per
trial. Spike times and event timestamps are already on the same session-absolute clock, so no
resampling, interpolation or per-stream offset correction is needed: the fixed grid of bin
edges relative to the go cue is simply added to each retained trial's go-cue time, and the
spikes are binned against those absolute edges. The same go-cue-relative grid is reused for the
two inputs and for the tongue output, so bin *k* covers the same interval in every stream.

ii.
```python
        go_times = _events(nwb, 'go_start_times')
        ...
        go = go_times[keep_idx]
```

```python
    edges = go_times[:, None] + BIN_EDGES[None, :]           # (n_trials, NBINS+1)
    idx = np.searchsorted(spike_times, edges.ravel(), side='left')
```

```python
            'temporal_alignment_event': 'onset of the auditory go cue (go_start_times)',
            'off_start': OFF_START,
            'off_end': OFF_END,
```

iii. Step 5 decision 1: "Alignment = go cue … Matches both papers and all reference code."
Step 10 Check 3(c) records that the reference `.mat` export already stores spike times relative
to `task_cue_time[0]` (the go cue) and aligns markers by subtracting the same quantity
(`align_markers.py`), so subtracting `go_start_times` from the NWB absolute times is the same
operation. The `--show-processing` plot panel 1 was used to verify alignment visually: trial
start sits at a median −3.2 s and the tone onset at exactly −1.85 s (= 0.65 s sample + 1.2 s
delay) on nearly every trial, which is the expected task geometry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, i.e. 80 non-overlapping bins spanning −2.5 s to +1.5 s relative to the go cue. The
grid is defined once at module level as 81 edges (and 80 centres) relative to the go cue and
reused for every trial, every session and every data stream, so every trial has exactly 80
timepoints. No rebinning/resampling of an intermediate representation is performed — spikes go
straight from raw times into the 50 ms bins, and the tongue video (~300 Hz) is averaged into the
same 50 ms bins in one step. `metadata['time_bin_size']` is 50.0 ms and the bin centres are
stored in `metadata['bin_centers_s']`.

ii.
```python
OFF_START = -2.5          # s, signed time from the go cue to the start of the trial window
OFF_END = 1.5             # s, signed time from the go cue to the end of the trial window
BIN_SIZE = 0.05           # s
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))          # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)       # (81,)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2                   # (80,)
```

```python
            'time_bin_size': BIN_SIZE * 1000.0,       # ms
            'n_time_bins': NBINS,
            'bin_centers_s': BIN_CENTERS.tolist(),
```

iii. Step 5 decision 2: the window and bin width are taken verbatim from the Decoder Task
section of the instructions. Step 9/Step 10 Check 3(d) records this as a *deliberate* departure
from the reference's 40 ms width / 3.4 ms stride sliding window, justified because the task
specifies the binning; a single fixed grid is also what makes the "same number of timepoints
for all trials and sessions" format requirement hold. Step 10 Check 5, item 6 pins down the
half-open bin convention and notes that the independent sanity check uses the same convention.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents['sample_start_times'].timestamps` (the sample-epoch / instruction-tone
onsets of the session) together with each trial's go-cue time. The tone assigned to a trial is
the **last** sample-epoch start at or before that trial's go cue, found with `searchsorted`.
A trial with no preceding sample epoch raises a `RuntimeError` (never triggered).

ii.
```python
        # tone onset := last sample-epoch start before the go cue
        sample_start = _events(nwb, 'sample_start_times')
        tone_pos = np.searchsorted(sample_start, go, side='right') - 1
        if np.any(tone_pos < 0):
            raise RuntimeError(f'{session_id}: trial without a preceding sample epoch')
        tone_time = sample_start[tone_pos]
```

iii. Step 5 decision 8 and Step 10 Check 5, item 7: an early lick *replays* the sample epoch, so
a trial can carry more than one sample-start event; the last replay is the instruction the
animal actually acted on, and the one whose delay epoch leads to the go cue. The AI validated
this with the task geometry: the median tone→go interval is 1.85 s = 0.65 s sample + 1.2 s
delay, exactly the published structure, and it notes that on replay trials the residual delay
can be as short as 0.1 s, which is why the input is not a constant offset. Sanity check 5
re-verified the 1.85 s median independently from the raw NWB on 5 sessions.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Because the bins are laid out around the go cue, the value of bin *k* on a trial is simply
the bin centre (relative to the go cue) plus the tone→go gap: `BIN_CENTERS[k] + (go − tone)`.
It is emitted as a **continuous** ramp of seconds (one value per bin), `float32`, not a binary
onset indicator. Observed range over the full dataset: [−1.5, 11.9] s.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2                   # (80,)
```
```python
        # time from tone onset at each bin centre (seconds, continuous)
        time_from_tone = BIN_CENTERS[None, :] + (go - tone_time)[:, None]   # (n_trials, NBINS)
```
```python
    inputs = [np.stack([time_from_tone[j], photostim_on[j]]).astype(np.float32)
              for j in range(n_trials)]
```

iii. Step 5 decision 9: the task specification explicitly labels this input "continuous,
time-varying", which the AI read as overriding the general format note that "if an input is a
time such as onset of some stimulus, represent it as a binary time series". Beyond locating the
tone, no further processing is applied. Sanity check 4 verified `input[0]` equals
`bin_centre + (go − last sample_start before go)` recomputed from the raw file
(`np.allclose`, 5 sessions); the `--show-processing` panel 4 shows a linear ramp crossing zero
at the tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is constructed *on* the neural bin grid: `BIN_CENTERS` are the centres of exactly the same
80 bins whose edges (`BIN_EDGES`) are used to count spikes, laid on the same per-trial go-cue
times (`go`, after trial curation). So alignment is structural — there is no separate
alignment step, and bin *k* of the input necessarily covers the same interval as bin *k* of the
firing rates.

ii.
```python
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)       # (81,)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2                   # (80,)
```
```python
    edges = go_times[:, None] + BIN_EDGES[None, :]            # neural
```
```python
        time_from_tone = BIN_CENTERS[None, :] + (go - tone_time)[:, None]   # input
```

iii. Step 10 Check 3(c): both streams are re-referenced to `go_start_times` on the single
session-absolute NWB clock, as in the reference `align_markers.py`. The AI also guarded the
order of operations — `go` is recomputed after every trial-curation step (including the
zero-spike drop) *before* the inputs are built, so the input rows always correspond to the same
trials as the neural rows.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `BehavioralEvents['photostim_start_times'].timestamps` and
`BehavioralEvents['photostim_stop_times'].timestamps` — the measured laser on/off times for the
whole session — together with each trial's go-cue time. The AI did *not* use the trials-table
`photostim_onset` / `photostim_duration` string columns for the conversion, but did use them as
an independent cross-check. Missing series are caught (`KeyError` → empty arrays) and a
start/stop count mismatch is warned about and leaves the input at 0.

ii.
```python
        photostim_on = np.zeros((n_trials, NBINS), dtype=np.float32)
        try:
            ps_start = _events(nwb, 'photostim_start_times')
            ps_stop = _events(nwb, 'photostim_stop_times')
        except KeyError:
            ps_start = ps_stop = np.zeros(0)
        if len(ps_start) and len(ps_start) == len(ps_stop):
            for j, g in enumerate(go):
                m = (ps_stop > g + OFF_START) & (ps_start < g + OFF_END)
                if m.any():
                    photostim_on[j] = interval_overlap_bins(ps_start[m], ps_stop[m], g)
        elif len(ps_start) != len(ps_stop):
            print(f'  WARNING {session_id}: photostim start/stop counts differ '
                  f'({len(ps_start)}/{len(ps_stop)}); photostim input left at 0', flush=True)
```

iii. Step 5 mapping: the reference analogue is `task_stimulation` columns 2–3, i.e. the laser
on/off *times*, which is exactly what the `BehavioralEvents` photostim series contain — so the
AI chose the event stream over the derived trials-table strings. Sanity checks 6 and 7 verify
that the set of photostim trials derived from the event stream equals the set derived from the
independent `trials.photostim_onset != 'N/A'` column (exact match on 5 sessions), so the two
sources agree. Check 8 verifies photostimulation always ends before the go cue and check 9 that
onset lies inside the delay epoch on ≥ 98% of stim trials. Overall 20.0% of trials are
stimulated, against ~19.6% in the raw data and "~25% on photostim sessions" in the papers.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) `float32` time series, one value per bin: bin *k* is 1 if a photostim
interval overlaps it by **more than 1 ms** (`OVERLAP_TOL`), else 0. Overlap is computed
vectorised over bins as `min(stop, edge_hi) − max(start, edge_lo)`; intervals entirely outside
the window are skipped. Trials with no stimulation keep an all-zero row.

ii.
```python
OVERLAP_TOL = 1e-3               # s, minimum overlap for a bin to count as photostim-on
```
```python
def interval_overlap_bins(starts, stops, go_time):
    """Binary (NBINS,) indicator: 1 where a [start, stop] interval overlaps the bin."""
    on = np.zeros(NBINS, dtype=np.float32)
    if len(starts) == 0:
        return on
    for a, b in zip(starts - go_time, stops - go_time):
        if b <= BIN_EDGES[0] or a >= BIN_EDGES[-1]:
            continue
        # Overlap duration between [a, b] and every bin.  A plain "does it touch the bin"
        # test would mark a bin that the interval overlaps by a fraction of a millisecond:
        # photoinhibition ends at the go cue to within +-0.5 ms, which would otherwise
        # switch the first post-go bin on for about half of the stimulated trials.
        ov = np.minimum(b, BIN_EDGES[1:]) - np.maximum(a, BIN_EDGES[:-1])
        on[ov > OVERLAP_TOL] = 1.0
    return on
```

iii. Step 10 Check 2 records this as a bug the sanity checks caught and fixed: photoinhibition
ends at the go cue to within ±0.5 ms and starts within ±0.5 ms of a bin edge, so the original
"does the interval touch this bin" test switched the first *post*-go bin on for about half the
stimulated trials and turned on one bin too early at the onset. Requiring > 1 ms of overlap
removes the artefact and makes the "photostim always ends before the go cue" check pass
exactly. The AI also verified, and documented as not-a-bug, one trial whose laser onset is
2.25 s pre-go because an early-lick replay pushed the go cue later while the laser fired at its
usual time. The `--show-processing` panel 5 shows the stimulation confined to the delay epoch
(−1.2 to −0.7 s) on 77/354 trials in the sample session.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The laser start/stop times are converted to go-cue-relative seconds (`starts - go_time`,
`stops - go_time`) and compared against the same `BIN_EDGES` grid used to bin the spikes, for
the same curated `go` array. Only intervals overlapping the trial's own [−2.5, +1.5] s window
are considered. So alignment is again structural, on one shared grid.

ii.
```python
            for j, g in enumerate(go):
                m = (ps_stop > g + OFF_START) & (ps_start < g + OFF_END)
                if m.any():
                    photostim_on[j] = interval_overlap_bins(ps_start[m], ps_stop[m], g)
```
```python
    for a, b in zip(starts - go_time, stops - go_time):
        ...
        ov = np.minimum(b, BIN_EDGES[1:]) - np.maximum(a, BIN_EDGES[:-1])
```

iii. Step 10 Check 3(e): the reference also re-references `task_stimulation` to the go cue; the
only difference is that the reference keeps it as per-trial scalars while this conversion
rasterises it onto the bin grid, "because the decoder wants time-varying inputs". Because the
comparison is against `BIN_EDGES` rather than `BIN_CENTERS`, the sub-millisecond tolerance
(4-b) was needed to keep the rasterisation from leaking one bin past the go cue.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the trials table, so choice is derived from two
columns: `trials.trial_instruction` (`'left'`/`'right'`, the side the tone instructed) and
`trials.outcome` (`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed
side, a miss means it licked the other side, and an `ignore` means it did not lick.

ii.
```python
        instruction = np.asarray(trials['trial_instruction'][:])
        outcome = np.asarray(trials['outcome'][:])
...
        instr_k = instruction[keep_idx]
        out_k = outcome[keep_idx]
```

iii. Step 4 (discrepancy table, "`outcome` semantics") documents that the AI validated this
derivation against the raw lick-time streams before adopting it: on `hit` trials the animal
licks the instructed port (100%), on `miss` trials it *first* licks the opposite port (100%),
and on `ignore` trials there are no licks in [go, go+1.5] (100%). Sanity check 13 re-ran the
first-post-go-lick comparison on 5 spot-check sessions and got 100% agreement (195/195,
353/353, 578/578, 573/573, …), and check 14 confirmed zero licks on "no lick" trials. So the
derived variable is equivalent to the directly measured lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded 0 = left, 1 = right, 2 = no lick, with the default being `no lick` and hit/miss trials
overwritten by the four instruction × outcome combinations. The per-trial value is then
broadcast across all 80 bins as row 0 of the per-trial output array (`int64`), and
`output_values[0] = ['left', 'right', 'no lick']` names the codes. Resulting distribution:
left 0.429, right 0.422, no lick 0.149.

ii.
```python
CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ...
]
```
```python
        # choice: hit -> instructed side, miss -> opposite side, ignore -> no lick
        choice = np.full(n_trials, CHOICE_NOLICK, dtype=np.int64)
        hit = out_k == 'hit'
        miss = out_k == 'miss'
        choice[hit & (instr_k == 'left')] = CHOICE_LEFT
        choice[hit & (instr_k == 'right')] = CHOICE_RIGHT
        choice[miss & (instr_k == 'left')] = CHOICE_RIGHT
        choice[miss & (instr_k == 'right')] = CHOICE_LEFT
```
```python
    outputs = [np.stack([np.full(NBINS, choice[j]),
                         np.full(NBINS, outcome_code[j]),
                         np.full(NBINS, early_code[j]),
                         tongue_class[j]]).astype(np.int64)
               for j in range(n_trials)]
```

iii. Step 5 decision 12: per-trial outputs are broadcast across time because the output array
must have a single `(n_output, n_timepoints)` shape and the tongue output is genuinely
time-varying. The left/right/no-lick coding follows the instructions' ordering. The AI notes in
Step 9 that the near-equal left/right split (0.429/0.422) is the expected signature of randomly
interleaved trial types, which it used as a consistency check.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `trials.outcome`, which already stores exactly the three strings the
instructions ask for: `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
        outcome = np.asarray(trials['outcome'][:])
...
        out_k = outcome[keep_idx]
```

iii. Step 5 mapping table: `trials.outcome` → `output[1]`, reference analogue `correctness`
(−1/0/1). No derivation is needed because the trials table stores the categories explicitly;
Step 4 confirms the semantics against lick times, and Step 9 checks the distribution against
the raw data (hit 0.687 / miss 0.165 / ignore 0.148 raw vs 0.685 / 0.166 / 0.149 converted).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped through a fixed dictionary to 0 = ignore, 1 = miss, 2 = hit and
broadcast across all 80 bins as row 1 of the output array. `output_values[1] =
['ignore', 'miss', 'hit']`.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
OUTPUT_VALUES = [
    ...
    ['ignore', 'miss', 'hit'],
    ...
]
```
```python
        outcome_code = np.array([OUTCOME_CODE[o] for o in out_k], dtype=np.int64)
```

iii. The code assignment follows the ordering given in the instructions ("ignore, miss, hit").
Broadcasting across bins is the same per-trial-output convention as 5-b. Sanity check 11
verified `output[1]` equals the raw `outcome` column exactly on 5 sessions. Step 12 records a
caveat the AI raised itself: because the per-trial recorded interval ends at the error lick on
`miss` trials, the zero-padding pattern partly leaks outcome information; the AI kept the
zero-padding anyway because dropping incomplete trials would remove ~92% of `miss` trials.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `trials.early_lick` column, which holds the strings `'no early'` and
`'early'`.

ii.
```python
        early_lick = np.asarray(trials['early_lick'][:])
...
        early_k = early_lick[keep_idx]
```

iii. Step 5 mapping table: `trials.early_lick` → `output[2]`, reference analogue
`behavior_early_report` / `early_lick_trials`. The flag is stored explicitly so no derivation is
needed. Step 12 notes that the lick that sets the flag occurs during the sample or delay epoch,
so the causal event falls inside the −2.5 s window even though the flag is per-trial — but also
that the task *replays* the aborted epoch, so the sample+delay immediately preceding the go cue
is lick-free by construction, which intrinsically limits how decodable this output can be.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary encoding `(early_k == 'early')` → 0 = no, 1 = yes, broadcast across all 80 bins as
row 2 of the output array. `output_values[2] = ['no', 'yes']`. Resulting distribution: no
0.884, yes 0.116 (vs 0.114 in the raw data).

ii.
```python
        early_code = (early_k == 'early').astype(np.int64)
```
```python
OUTPUT_VALUES = [
    ...
    ['no', 'yes'],
    ...
]
```

iii. The 0/1 coding follows the instructions' ordering ("no, yes"). Broadcasting follows the
same convention as the other per-trial outputs. Sanity check 12 verified `output[2]` matches
the raw `early_lick` column exactly on 5 sessions, and Step 12 cross-validated the behavioural
reality of the flag: in early-lick trials the tongue is visible in 29.2% of pre-go bins versus
6.5% in other trials, and 60.7% of early-lick trials have a non-standard tone→go interval
versus 6.7% of the rest.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']`, whose
`data` is `(n_frames, 3)` = (`tongue_x`, `tongue_y`, `tongue_likelihood`) with matching
`timestamps` (~300 Hz). Column 1 (`tongue_y`) provides the value and column 2 (the DeepLabCut
likelihood) decides visibility. `trials.start_time` is also used, to bucket each frame into the
trial it was acquired in. If the series is absent the output falls back to all "not visible"
(it is in fact present in all 174 sessions).

ii.
```python
    bts = nwb.acquisition.get('BehavioralTimeSeries', None)
    if bts is None or 'Camera0_side_TongueTracking' not in bts.time_series:
        return out, 0.0

    ts_obj = bts.time_series['Camera0_side_TongueTracking']
    frame_t = np.asarray(ts_obj.timestamps[:], dtype=np.float64)
    data = np.asarray(ts_obj.data[:, 1:3], dtype=np.float64)     # y, likelihood
    y = data[:, 0]
    visible = data[:, 1] > TONGUE_LIKELIHOOD_THRESH
```

iii. Step 5 mapping: reference analogue `tracking.camera_0_side.tongue_y` (the marker stream
used by `align_markers.py`). This is the only tongue measurement in the file. Step 4 records
that the AI checked availability across all 174 sessions and explained why the method paper
used only 105–106 sessions (it additionally required usable *raw* video and all 8 markers
including whisker), concluding that no session needs to be dropped for this output.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps, all in `bin_tongue()`:
1. Frames with DeepLabCut likelihood ≤ 0.9 are discarded — when the tongue is retracted the
   tracker still reports a position, so those frames are not real measurements.
2. Every frame is bucketed into the trial it was acquired in
   (`searchsorted(trial_start, frame_t, 'right') - 1`), and frames belonging to trials that
   were dropped by curation are discarded. A frame can therefore only contribute to *its own*
   trial's window, never to a neighbour's.
3. Surviving visible frames are mapped to `(trial, bin)` cells and the mean `y` per cell is
   computed with two `np.bincount` calls (sums / counts).
4. A cell with no visible frame stays NaN, which becomes the "not visible" class in
   `discretise_tongue`.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9   # DeepLabCut likelihood above which the tongue is "visible"
```
```python
    # frame -> trial (video only runs within a trial; frames start at trial_start)
    frame_trial = np.searchsorted(trial_start, frame_t, side='right') - 1

    # only visible frames matter for the value; keep them and their trial/bin index
    vis = np.flatnonzero(visible & (frame_trial >= 0))
    ...
    pos_of_trial = np.full(n_raw, -1, dtype=np.int64)
    pos_of_trial[keep_idx] = np.arange(n_trials)
    pos = pos_of_trial[ft]
    sel = pos >= 0
    vis, pos = vis[sel], pos[sel]
    ...
    bin_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
    np.clip(bin_idx, 0, NBINS - 1, out=bin_idx)
    flat = pos * NBINS + bin_idx
    n_cells = n_trials * NBINS
    sums = np.bincount(flat, weights=y[vis], minlength=n_cells)
    cnts = np.bincount(flat, minlength=n_cells)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    out = mean.reshape(n_trials, NBINS)
```

iii. Step 5 decision 10: "The likelihood distribution is strongly bimodal (≈ 3–6 × 10⁻⁵ vs
≈ 1.0); any threshold in [0.1, 0.99] gives the same answer to within 0.1% of frames", so the
exact cut is immaterial — ~6–11% of frames have the tongue visible. Step 10 Check 5, item 8
gives the rationale for the trial bucketing: "frames are first bucketed by trial … so a window
that extends past its own trial boundary cannot pick up a neighbouring trial's frames." Step 3
records the reference's treatment ("when the tongue was occluded while it was in the mouth …
we set the tongue position to its mean value") and the AI's reading that occlusion is thereby
an explicit distinct state, matching the required "not visible" class; the AI argues in Step 10
Check 3(f) that an explicit class "is the more informative encoding of the same fact" than the
reference's mean imputation. `np.bincount` aggregation is also a Step 6 speed-up
(~50× vs a per-bin loop).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles are taken over *all visible binned values* of that
session's retained trials (i.e. the non-NaN entries of the `(n_trials, 80)` bin-mean array —
the same quantity that is being discretised). Each bin is then labelled 0 if its value is
strictly below the 40th percentile, 2 if strictly above the 60th, 1 otherwise, and 3 if it has
no visible frame. The two percentile values are stored per session in
`metadata['session_info'][i]['tongue_pct_40_60']`. Resulting distribution across the dataset:
0.098 / 0.049 / 0.098 / 0.755, i.e. exactly 40 : 20 : 40 of the visible bins.

ii.
```python
TONGUE_LOW_PCT = 40              # percentile boundaries for the tongue-y discretisation
TONGUE_HIGH_PCT = 60
```
```python
def discretise_tongue(tongue_y):
    """Discretise tongue y into 0/1/2 by session percentiles, 3 where not visible."""
    cls = np.full(tongue_y.shape, 3, dtype=np.int64)
    vis = ~np.isnan(tongue_y)
    if vis.sum() == 0:
        return cls, (np.nan, np.nan)
    vals = tongue_y[vis]
    p_lo = np.percentile(vals, TONGUE_LOW_PCT)
    p_hi = np.percentile(vals, TONGUE_HIGH_PCT)
    v = tongue_y[vis]
    c = np.ones(v.shape, dtype=np.int64)
    c[v < p_lo] = 0
    c[v > p_hi] = 2
    cls[vis] = c
    return cls, (float(p_lo), float(p_hi))
```
```python
OUTPUT_VALUES = [
    ...
    ['low (<40th pct)', 'mid (40-60th pct)', 'high (>60th pct)', 'not visible'],
]
```

iii. Step 5 decision 11: "Tongue percentiles are computed per session over the *binned, visible*
values of all retained trials of that session (the quantity actually being discretised)" — the
per-session scope and the 40/60 split come straight from the instructions, and taking the
percentiles of the binned rather than the raw-frame values keeps the edges on the same
quantity that is classified. Frames where the tongue is retracted are excluded first, because
including them would shift the percentiles and mix noise with real protrusions. Sanity check 16
verifies the 40 : 20 : 40 split of visible bins (`np.allclose`, atol 0.02, 5 sessions), check 15
re-derives `output[3]` from the raw DLC timestamps/likelihoods exactly (32 bin probes per
session), and check 17 verifies visibility is concentrated after the go cue. The
`--show-processing` panels 6–7 plot the y histogram with the two percentile lines and the class
raster.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as the spikes and the go cues, so
each visible frame's offset from its trial's go cue is computed directly
(`rel = frame_t - go[pos]`), frames outside `[-2.5, +1.5)` are dropped, and the remainder are
assigned to bins by `floor((rel - OFF_START) / BIN_SIZE)` — the same go-cue-relative 50 ms grid
used for the firing rates. The bin index is clipped to `[0, 79]` as a guard. No interpolation
or offset correction is applied. Bins with no frames (including the leading bins of the ~3% of
trials whose go cue falls less than 2.5 s after trial start, where the trial-gated video has not
started) end up in the "not visible" class.

ii.
```python
    rel = frame_t[vis] - go[pos]
    inwin = (rel >= OFF_START) & (rel < OFF_END)
    vis, pos, rel = vis[inwin], pos[inwin], rel[inwin]
    ...
    bin_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
    np.clip(bin_idx, 0, NBINS - 1, out=bin_idx)
    flat = pos * NBINS + bin_idx
```

iii. Step 10 Check 3(c): "subtract `BehavioralEvents['go_start_times'].timestamps` from both
spike times and video timestamps" — the same operation the reference `align_markers.py`
performs. Because both streams use one shared grid, bin *k* of the tongue output covers the
same interval as bin *k* of the firing rates by construction. Sanity check 15 recomputes the
class of specific `(trial, bin)` cells directly from the raw timestamps and matches exactly,
and check 17 confirms the expected physiology (tongue visibility concentrated after the go cue,
ratio > 2), which is a direct test that the alignment has no temporal offset.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Nine distinct cases, each handled explicitly:
- **Session never quality-controlled** (`classification`/`anno_name` all NaN): no unit matches
  `'good'`, so the session returns `None` and is dropped (1 session).
- **Trials outside the ephys recording** (`obs_intervals` shorter than the trials table,
  9 sessions): located in the trials table by `start_time` and excluded; the AI refused the
  "first N trials" shortcut because one session is not a prefix, and asserts the match.
- **Trials with no spikes at all** (recording stopped mid-trial): detected post-binning and
  dropped (2 trials), rather than emitted as 4 s of 0 Hz.
- **Bins with no visible tongue frame**: represented as an explicit fourth class (3, "not
  visible") rather than imputed.
- **Missing tongue tracking series**: falls back to an all-NaN array → all "not visible".
- **Units with a missing CCF x coordinate**: hemisphere falls back to the probe insertion's
  targeted hemisphere.
- **Unmapped CCF annotation strings**: warn and fall back to `OtherCortex` (all 293 strings
  present were verified to map, so this never fires).
- **Photostim start/stop count mismatch**: warn and leave the photostim input at 0.
- **Go-cue count ≠ trials-table length**: hard `RuntimeError` (never triggered).
Additionally, `_worker` catches any per-session exception and prints a traceback rather than
aborting the whole run, and sessions with fewer than 2 surviving trials are dropped.

ii.
```python
        classification = np.asarray(units['classification'][:])
        good = np.flatnonzero(classification == 'good')
        if len(good) == 0:
            io.close()
            return None
```
```python
        obs_trial = np.searchsorted(trial_start, obs[:, 0] - 1e-6)
        if not np.allclose(trial_start[obs_trial], obs[:, 0]):
            raise RuntimeError(f'{session_id}: obs_intervals do not match the trials table')
```
```python
        nonempty = fr.sum(axis=(0, 2)) > 0
        if not nonempty.all():
            ...
            fr = fr[:, nonempty, :]
```
```python
        missing = np.isnan(elec_x)
        if missing.any():
            hemisphere[missing] = np.array([t.split(' ')[0] for t in probe_target])[missing]

        region_name = np.array([annotation_to_region(a, t) for a, t in zip(anno, probe_target)])
        unknown = region_name == None  # noqa: E711
        if unknown.any():
            print(f'  WARNING {session_id}: {unknown.sum()} units with unmapped annotation ...')
            region_name[unknown] = 'OtherCortex'
```
```python
def _worker(args):
    path, debug = args
    try:
        return process_session(path, collect_debug=debug)
    except Exception as exc:          # keep one bad file from killing the whole run
        import traceback
        traceback.print_exc()
        print(f'  ERROR processing {path}: {exc}', flush=True)
        return None
```

iii. Step 10 Check 5 enumerates all twelve edge cases and their handling. The governing
principle the AI states is a split by meaning: where *nothing was recorded* (session, trial) the
data is excluded, because emitting it would fabricate 0 Hz activity; where the measurement
legitimately has no value (retracted tongue) it is given an explicit category; where an
auxiliary annotation is missing (CCF x, region string) a documented fallback is used rather than
dropping the unit. Two of these were discovered by failures rather than anticipated: the
`obs_intervals` mismatch surfaced as an `is_good_trials` broadcasting `ValueError`, and the
all-zero trials surfaced as a `train_decoder.py` warning — both were fixed rather than
suppressed.

## 10-a. What are the most time-consuming steps of the code?

i. The script instruments itself (`timing` dict per session: `setup`, `neural`, `input`,
`output`) and prints the breakdown per session. Measured on the full run: setup (opening the
file, reading the trials table, unit metadata, annotations, electrodes) ≈ 0.2–0.4 s/session;
neural binning (reading the ~90 MB flat spike buffer plus the per-unit `searchsorted` loop)
≈ 0.2–1.7 s/session and scaling with unit count; inputs ≈ 0.0 s; outputs (the tongue video
array, ~680 k × 3) ≈ 0.0–0.1 s. So I/O plus spike binning dominate. At the whole-script level:
26.4 s for all 173 sessions with 16 workers, plus 14.5 s to pickle the 11.8 GB result — i.e. the
single largest wall-clock item in the full run is the pickle write, and total wall time is 41 s.

ii.
```python
        timing['setup'] = time.time() - t0
        ...
        timing['neural'] = time.time() - t1
        ...
        timing['input'] = time.time() - t1
        ...
        timing['output'] = time.time() - t1
```
```python
                    print('[%d/%d] %s: %d neurons, %d/%d trials (%.1fs: %s)'
                          % (i + 1, len(files), res['session_id'], res['n_neurons'],
                             res['n_trials'], res['n_trials_raw'], res['total_time'],
                             ', '.join(f'{k}={v:.1f}' for k, v in res['timing'].items())),
                          flush=True)
```

iii. Step 7 "Run Time Estimates" builds the per-step table from the sample run, extrapolates to
173 sessions (2–6 min serial, ≈ 1–2 min with 16 workers, plus 1–3 min for the pickle), and
concludes "Estimated full conversion ≪ 15 min, so no further optimisation is needed." Step 9
confirms the actual 40 s against that estimate. The AI explicitly accounted for the sample
sessions being unrepresentative in trial count (257 vs a dataset mean of ~500) when
extrapolating.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The conversion vectorises the hot paths and leaves three cheap Python loops:
- **Vectorised**: spike binning across trials — the per-trial edges are built as
  `go[:, None] + BIN_EDGES[None, :]` and located with a single `np.searchsorted`, so there is no
  per-trial loop (~20× vs a per-trial loop); tongue aggregation via two `np.bincount` calls
  instead of a per-bin loop (~50×).
- **Remaining**: (1) the per-unit loop in the neural step (`for i, u in enumerate(good)`), which
  cannot be collapsed because each unit has a different number of spikes so there is no single
  sorted array to search — this is inherent to the ragged storage; (2) the per-trial photostim
  loop `for j, g in enumerate(go)` with its inner `for a, b in zip(...)` over intervals, which is
  the one genuinely vectorisable remaining loop but is not flagged in the notes (it measures
  0.0 s/session because only ~20% of trials are stimulated and each has one interval); (3) the
  list comprehensions that slice `fr`/`inp`/`out` into per-trial arrays, which are required by
  the target format.
Session-level parallelism (`multiprocessing.Pool`, 16 workers) substitutes for further
loop-level vectorisation and gives ~10× overall.

ii.
```python
    edges = go_times[:, None] + BIN_EDGES[None, :]           # (n_trials, NBINS+1)
    idx = np.searchsorted(spike_times, edges.ravel(), side='left')
    idx = idx.reshape(edges.shape)
    counts = np.diff(idx, axis=1)
```
```python
        for i, u in enumerate(good):
            st = flat_spikes[starts[u]:ends[u]]
            fr[i] = bin_spikes_rate(st, go)
```
```python
            for j, g in enumerate(go):
                m = (ps_stop > g + OFF_START) & (ps_start < g + OFF_END)
                if m.any():
                    photostim_on[j] = interval_overlap_bins(ps_start[m], ps_stop[m], g)
```
```python
    sums = np.bincount(flat, weights=y[vis], minlength=n_cells)
    cnts = np.bincount(flat, minlength=n_cells)
```

iii. Step 6 "Code inefficiencies identified / speedups added" lists exactly these: bulk read of
the ragged spike dataset instead of per-unit reads (~5×), vectorised `searchsorted` binning
(~20×), `np.bincount` tongue aggregation (~50×), and 16-way multiprocessing (~10×). The
docstring of `bin_spikes_rate` states the intent explicitly: "`np.searchsorted` is used on the
(sorted) spike train, so the cost is O(n_trials · NBINS · log n_spikes) and no per-trial Python
loop is needed." The AI's stated stopping criterion was the instructions' 15-minute budget,
which the 41 s run clears by a wide margin.

## 10-c. What processing does the code repeat multiple times?

i. Essentially nothing is recomputed. Each NWB file is opened exactly once per run and every
quantity derived from it is computed once inside that single `process_session` call; the bin
grid (`BIN_EDGES`, `BIN_CENTERS`) and the region tables are built once at module level and
reused for every trial and session; the tongue percentiles are per-session and are computed
inside the same pass, so no second pass over the data is needed. Two very small items are
computed and then partly thrown away: firing rates are binned for all curated trials and then
the (2 total) all-zero trials are sliced out, and `--show-processing` re-derives nothing but
retains a `debug` dict for the first 2 sessions. The `multiprocessing` spawn context re-imports
the module in each worker, which is a fixed one-off cost, not repeated processing.

ii.
```python
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)       # (81,)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2                   # (80,)
```
```python
        sv = units['spike_times']
        ends = np.asarray(sv.data[:])
        starts = np.concatenate([[0], ends[:-1]])
        flat_spikes = np.asarray(sv.target.data[:], dtype=np.float64)
```
```python
        nonempty = fr.sum(axis=(0, 2)) > 0
        if not nonempty.all():
            ...
            fr = fr[:, nonempty, :]
```

iii. Step 6 records the single-read design as a deliberate speed-up: "Reading spike times per
unit through `units['spike_times'][i]` is slow; the whole flattened ragged dataset is read once
(`sv.target.data[:]`, ~90 MB per session, 0.13 s) and sliced." The conversion is a single pass
over the files, and because the tongue discretisation edges are per-session rather than global
they can be established within that pass.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Two categories.
- **Read but partly unused**: the flat `spike_times` buffer is read in full, including the ~75%
  of units that fail QC and are never binned — accepted because one contiguous read is far
  cheaper than per-unit reads.
- **Computed and kept, but not consumed by the decoder**: the per-unit hemisphere assignment
  (which requires the unit→electrode→CCF-x lookup, the `json.loads` of each electrode group's
  location, and a midline rule), `trial_stop`, `tongue_visible_frac`, `n_trials_raw`,
  `n_trials_observed`, `trial_index`, `tongue_percentiles`, `bin_centers_s`, and the per-session
  timing dict. None of these are discarded in the sense of being thrown away — all are written
  into `metadata` / `session_info` as provenance — but `train_decoder.py` reads none of them.
  The `debug` dict and `--show-processing` figures are also pure verification artefacts.
Nothing is computed and then silently dropped; the wasted work is bounded and small (the
`setup` step, which contains the hemisphere/region work, is ~0.3 s of a ~1 s session).

ii.
```python
        egroups = np.asarray(units['electrode_group'].data[:])[good]
        probe_target = np.array([json.loads(g.location)['brain_regions'] for g in egroups])

        er = units['electrodes']
        cum = np.asarray(er.data[:])
        elec_idx = np.asarray(er.target.data[:])[cum - 1][good]
        elec_x = np.asarray(nwb.electrodes['x'][:], dtype=np.float64)[elec_idx]
        hemisphere = np.where(elec_x >= ML_MIDLINE_UM, 'left', 'right')
```
```python
            'neuron_hemisphere': [[int(v) for v in r['hemisphere']] for r in results],
            'hemisphere_values': ['left', 'right'],
```
```python
    if collect_debug:
        result['debug'] = dict(
            go=go, tone_time=tone_time, trial_start=trial_start[keep_idx],
            trial_stop=trial_stop[keep_idx], time_from_tone=time_from_tone, ...)
```

iii. The AI's justification for the extra metadata is preservation rather than efficiency: the
instructions' success criteria require "preserve all relevant information from the source" and
"include complete and accurate metadata", and Step 5 decision 13 explains that hemisphere is
stored separately in metadata "rather than doubling the region list" — i.e. it is a deliberate
representational choice reproducing the reference's `ccf_x >= 5700` rule
(`helper_get_neuron_id_area`), validated by sanity checks 18–19. The debug/plot path is gated
behind `--show-processing`, which the instructions require. The AI did not flag any of this as
waste in CONVERSION_NOTES.md, because it concluded in Step 7 that runtime was already an order
of magnitude inside budget.
