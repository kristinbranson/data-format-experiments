# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI:000363) is one NWB file per session, stored under `/app/data/sub-<id>/`. The AI finds every session with a single sorted glob and opens each file **directly with `h5py`** rather than `pynwb`, reading the HDF5 groups by path (`intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, `units`, `general/subject`, `general/extracellular_ephys/electrodes`). All 174 files are found.

Loading is done in **two passes over the files**, both parallelised with a `multiprocessing.Pool` (default 16 workers, 24 used for the production run):

1. `scan_session` — a cheap metadata pass over all 174 files that reads the trials table and the unit `classification` column to decide which sessions pass the session-selection criteria (2.2 s total).
2. `convert_session` — a full conversion pass over only the 143 selected files, with a per-session pickle cache in `/tmp/map_converted_sessions` so a rerun can resume.

One external, non-NWB resource is also loaded: the Allen CCFv3 structure graph (`/app/allen_ccf_structures.csv`), downloaded once from the Allen Brain Map API and used to map `units/anno_name` strings onto coarse brain regions.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, '*', '*.nwb')))
print('found %d nwb files' % len(files), flush=True)

region_map = build_region_map()

# ---- session selection ---------------------------------------------- #
with Pool(args.nproc) as pool:
    scans = pool.map(scan_session, files)
...
with Pool(args.nproc) as pool:
    for i, result in enumerate(pool.imap(convert_session, jobs)):
        sessions.append(result)
```

```python
def convert_session(args):
    """Convert one NWB file into per-trial neural / input / output arrays."""
    path, region_map, cache_file = args
    if cache_file is not None and os.path.exists(cache_file):
        with open(cache_file, 'rb') as fh:
            return pickle.load(fh)

    with h5py.File(path, 'r') as f:
        tt = _trial_table(f)
        classification = f['units/classification'][:].astype(str)
        ...
```

```python
def _trial_table(f):
    """Read the trial table plus the go-cue and tone-onset times of a session."""
    tr = f['intervals/trials']
    out = {
        'start_time': tr['start_time'][:],
        'stop_time': tr['stop_time'][:],
        'outcome': tr['outcome'][:].astype(str),
        'early_lick': tr['early_lick'][:].astype(str),
        'instruction': tr['trial_instruction'][:].astype(str),
        'auto_water': tr['auto_water'][:].astype(int),
        'free_water': tr['free_water'][:].astype(int),
        'photostim_onset': tr['photostim_onset'][:].astype(str),
        'photostim_duration': tr['photostim_duration'][:].astype(str),
        'photostim_power': tr['photostim_power'][:].astype(str),
    }
```

iii. The AI first dumped the full HDF5 tree of a sample file and enumerated which `BehavioralEvents` / `BehavioralTimeSeries` series exist in all 174 files (`Counter({(...14 event series...): 174})`) before committing to any path. It chose `h5py` over `pynwb` because it only needs a handful of datasets per file and wants to read them inside worker processes; it validated the choice by spot-checking converted firing rates against an independent histogram of the raw `spike_times` for five random (trial, unit) pairs (`match True` in all cases). The two-pass design is justified by the session filter: the scan pass costs ~2 s and avoids fully converting the 31 sessions that are rejected. The Allen ontology CSV is needed because the NWB files give free-text CCF annotations only, with no parent-structure information.

## 1-b. How are the data split into subjects?

i. The subject of each session is read from the NWB metadata field `general/subject/subject_id` (a numeric string such as `'440956'`), which is also what the containing `sub-*` directory name is derived from. At assembly time `subjects` is the sorted set of unique ids and `subject_idx` is each session's index into that list. This gives 28 subjects, matching the dandiset.

ii.
```python
subject = str(f['general/subject/subject_id'][()].decode())
```

```python
subjects = sorted({s['subject'] for s in sessions})
subject_idx = np.array([subjects.index(s['subject']) for s in sessions],
                       dtype=np.int64)
```

```python
'subjects': subjects,
'subject_idx': subject_idx,
```

iii. `subject_id` is the canonical animal identifier inside the file, so no grouping or filename parsing is needed. The AI read it independently in both `scan_session` and `convert_session`, and the resulting 28 subjects with 2–10 sessions each were checked against the per-subject session counts printed by the verifier. No subject-level exclusion is applied; subjects only disappear implicitly if all of their sessions fail session selection (none do — all 28 survive).

## 1-c. How are the data split into sessions?

i. One NWB file is one session, so the file list *is* the session list and no splitting is needed. Session order follows the sorted file paths (which embed the acquisition timestamp, so sessions are chronological within a subject), and each session is identified in metadata by its filename plus `session_start_time`.

The AI then applies an explicit **session-selection filter taken from the data paper's STAR Methods**, which the human reference does not apply: a session is kept only if overall behavioural performance is > 65 % *and* it has at least 50 correct lick-left and 50 correct lick-right trials, and it has at least one QC-good unit. This rejects **31 of 174 sessions**, leaving 143. A session is additionally dropped after conversion if it ends up with 0 neurons or fewer than 2 trials.

Performance is deliberately defined as hits / (hits + misses) over control (non-photostim), non-early-lick, non-auto/free-water trials — i.e. no-response ("ignore") trials are excluded from the denominator.

ii.
```python
# session selection (data paper, STAR methods):
#   "We selected experimental sessions for analysis based on following criteria:
#    overall behavioral performance (> 65%), and at least 50 correct lick left
#    and lick right trials each."
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50
```

```python
def session_performance(tt):
    """Behavioural performance and correct-trial counts of a session.

    Performance is the fraction of correct responses on control (no
    photostimulation) trials, excluding early-lick trials as well as
    auto-water / free-water trials.  Trials on which the animal did not respond
    ('ignore') are not counted -- with them included the mean performance over
    the dandiset would be 68 %, whereas hits / (hits + misses) gives 81 %, which
    matches the 84 % (range 65-99 %) quoted by the data paper.
    """
    base = ((tt['photostim_power'] == 'N/A') & (tt['early_lick'] == 'no early')
            & trial_mask(tt))
    hit = tt['outcome'] == 'hit'
    miss = tt['outcome'] == 'miss'
    responded = (hit | miss) & base
    performance = hit[base].sum() / max(1, responded.sum())
    n_left = int((hit & base & (tt['instruction'] == 'left')).sum())
    n_right = int((hit & base & (tt['instruction'] == 'right')).sum())
    return float(performance), n_left, n_right
```

```python
for s in scans:
    reasons = []
    if s['performance'] <= MIN_PERFORMANCE:
        reasons.append('performance %.3f' % s['performance'])
    if min(s['n_correct_left'], s['n_correct_right']) < MIN_CORRECT_PER_DIRECTION:
        reasons.append('correct trials %d/%d'
                       % (s['n_correct_left'], s['n_correct_right']))
    if s['n_good_units'] == 0:
        reasons.append('no good units')
    if reasons:
        s['rejected_because'] = '; '.join(reasons)
        rejected.append(s)
    else:
        selected.append(s)
```

```python
# drop sessions that ended up without neurons or with fewer than 2 trials
sessions = [s for s in sessions
            if s['info']['n_neurons'] > 0 and s['info']['n_trials'] >= 2]
```

iii. The AI extracted the criterion verbatim from `/app/methods.txt` ("We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each") and applied it because the instructions require curation to match the reference papers. Every rejected session and its reason is recorded in `metadata['rejected_sessions']`, so the exclusion is auditable.

For the performance metric itself the AI justified its deviation from the literal wording empirically: counting `ignore` trials in the denominator gives a dandiset-wide mean of 68 %, while hits/(hits+misses) gives 81 %, which is consistent with the paper's stated "84% correct rate (range, 65–99%)". The `n_good_units == 0` rule catches the one session (`sub-440958_ses-20190216T162508`) whose units were never quality-controlled — the AI confirmed by enumeration that this is the only file with zero good units.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table `intervals/trials`, used directly. The go cue of each trial is taken positionally from `acquisition/BehavioralEvents/go_start_times/timestamps`, i.e. the code assumes exactly one go-cue event per trial row and indexes the two arrays with the same integer trial indices. No trial boundaries are re-derived from the event streams.

ii.
```python
events = f['acquisition/BehavioralEvents']
out['go_time'] = events['go_start_times']['timestamps'][:]
```

```python
trials = np.where(trial_mask(tt))[0]
go_time = tt['go_time'][trials]
```

iii. The AI verified the 1:1 correspondence when it dumped the HDF5 tree (e.g. `go_start_times/timestamps shape=(368,)` against 368 trial rows) and noted in the same exploration that the *other* epoch events are **not** one-per-trial — `sample_start_times` and `delay_start_times` have more entries than trials (395 vs 368 in the sample file) because an early lick replays the sample/delay epoch. That is why only `go_start_times` is used positionally and the tone onset is looked up by time instead (see 3-a). The AI also cross-checked total trials across all files (94,990), matching the dandiset.

## 1-e. How are trials filtered based on quality controls?

i. Four masks are ANDed together in `trial_mask`:

- `auto_water == 0` and `free_water == 0` — following `get_regular_trial_mask` in the reference repository, because on those trials reward is delivered independently of the animal's choice.
- `observed` — the trial must lie inside `units/obs_intervals` for every QC-good unit. In 9 sessions the ephys covers only a contiguous block of the behavioural file; the remaining trials have no spikes at all.
- `tone_valid` — a sample-epoch (tone) onset must exist before the go cue and inside the trial.

Photostimulation, early-lick and no-response (`ignore`) trials are **deliberately kept**, even though the reference repository's `get_regular_trial_mask` excludes early-lick and no-response trials and the data paper says "Early lick trials and no response trials were excluded for analysis". Sessions left with fewer than 2 trials are dropped. 74,609 trials survive across the 143 selected sessions.

Trials whose analysis window extends past the end of the trial interval (error trials end at the incorrect lick) are **not** dropped; their trailing bins are left at 0 Hz and the limitation is documented in metadata.

ii.
```python
def trial_mask(tt):
    """Trials that enter the converted dataset.

    Auto-water and free-water trials are dropped, following
    ``get_regular_trial_mask`` of the reference code: on those trials reward is
    delivered independently of the animal's choice, so the outcome label does
    not describe the animal's decision.  Photostimulation, early-lick and
    no-response trials are kept even though the reference excludes them,
    because the decoding task asks for photostimulation as a decoder input and
    for early lick and 'ignore' as decoder outputs.  Trials outside the
    electrophysiological observation interval, and the (rare) trials whose tone
    onset cannot be located, are dropped as well.
    """
    return ((tt['auto_water'] == 0) & (tt['free_water'] == 0)
            & tt['observed'] & tt['tone_valid'])
```

```python
def observed_trials(f, trial_start):
    """Trials during which all quality-controlled units were being recorded."""
    units = f['units']
    classification = units['classification'][:].astype(str)
    good_idx = np.where(classification == 'good')[0]
    if len(good_idx) == 0:
        return np.zeros(len(trial_start), dtype=bool)

    obs = units['obs_intervals'][:]
    stop = units['obs_intervals_index'][:]
    start = np.concatenate([[0], stop[:-1]])

    observed = np.ones(len(trial_start), dtype=bool)
    previous = None
    for u in good_idx:
        interval_starts = obs[start[u]:stop[u], 0]
        if previous is not None and len(previous) == len(interval_starts) \
                and np.array_equal(previous, interval_starts):
            continue                      # same coverage as the previous unit
        previous = interval_starts
        idx = np.clip(np.searchsorted(trial_start, interval_starts + 1e-6) - 1,
                      0, len(trial_start) - 1)
        matched = np.abs(trial_start[idx] - interval_starts) < 1e-3
        unit_observed = np.zeros(len(trial_start), dtype=bool)
        unit_observed[idx[matched]] = True
        observed &= unit_observed
    return observed
```

```python
idx = np.searchsorted(sample_starts, out['go_time']) - 1
out['tone_time'] = sample_starts[np.clip(idx, 0, len(sample_starts) - 1)]
out['tone_valid'] = (idx >= 0) & (out['tone_time'] >= out['start_time'])
```

iii. The AI arrived at the `obs_intervals` filter empirically: its first trial version of the converted data triggered 696 "all neural data is zero" warnings from the verifier, and it traced them to sessions where `obs_intervals` covers only the first N trials (e.g. 160 of 480 in `sub-440956_ses-20190208T133600`). It then enumerated all 174 files, confirming that the number of observation intervals is identical for every unit within a session (`sessions with >1 unique interval count: 0`) and that 9 sessions have `maxint < ntrials`. Rather than emitting those trials as 4 s of silence across every unit, it drops them; the final dataset produced **no** format warnings.

Auto-water/free-water exclusion is justified by direct reference to `get_regular_trial_mask` in `/app/code/VideoAnalysisUtils/functions.py` ("No early lick, no auto water, no free water, no no response trials"), with the explanation that reward on those trials is not contingent on the animal's choice. The deviation on early-lick / ignore / photostim trials is justified by the Decoder Task specification, which names photostimulation as a required decoder **input** and early lick and `ignore` as required decoder **output** values — excluding them would make two of the four outputs degenerate.

Keeping the trials with zero-filled trailing bins is justified explicitly: "Dropping those trials would remove essentially every 'miss' trial and collapse the outcome output to two classes."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `units/spike_times` together with its ragged index `units/spike_times_index`, restricted to units that pass quality control (see 2-c). The other input is `acquisition/BehavioralEvents/go_start_times/timestamps`, which places the bin edges. Unit metadata (`units/classification`, `units/anno_name`, `units/electrodes`, and the electrode table `general/extracellular_ephys/electrodes/x` and `location`) is used only to select units and label their brain region, not to alter the rates.

ii.
```python
units = f['units']
spike_times = units['spike_times'][:]
stop = units['spike_times_index'][:]
start = np.concatenate([[0], stop[:-1]])
unit_ids = np.where(good)[0]
```

```python
go_time = tt['go_time'][trials]
rates = bin_spikes(f, good, go_time)
```

iii. `spike_times` is the only neural representation in the NWB file. The AI reads the flat spike buffer once per session and slices it per unit with the `spike_times_index` offsets, rather than doing one HDF5 read per unit. It validated the result against the raw file: for five random (trial, unit) pairs it recomputed the bin counts independently from `spike_times` and confirmed `match True`.

## 2-b. How is the `neural` data processed?

i. Spike times are converted into per-bin **firing rates in Hz**. For each good unit, all 81 bin edges for all trials are built as one flat absolute-time array, `np.searchsorted` gives the running spike count at every edge, adjacent differences give the per-bin spike count, and the counts are divided by the 50 ms bin width. Results are stored as `float32` and transposed per trial to `(n_neurons, n_timepoints)`. No smoothing, normalisation, baseline subtraction or z-scoring is applied.

ii.
```python
n_trials = len(go_time)
edges = go_time[:, None] + BIN_EDGES[None, :]          # (ntrials, nbins+1)
flat_edges = edges.ravel()

rates = np.zeros((n_trials, N_BINS, len(unit_ids)), dtype=np.float32)
for i, u in enumerate(unit_ids):
    st = spike_times[start[u]:stop[u]]
    counts = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
    rates[:, :, i] = np.diff(counts, axis=1) / BIN_WIDTH
return rates
```

```python
neural = [np.ascontiguousarray(rates[i].T) for i in range(n_trials)]
```

iii. The AI noted from the method paper that the reference pipeline stores firing rates ("we binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms", and `sliding_histogram(..., rate=True)` in the repository), so it produces rates rather than raw counts, but uses the 50 ms non-overlapping bins the Decoder Task mandates instead of the paper's 40 ms/3.4 ms sliding window. It deliberately applies no smoothing or normalisation, leaving that to the decoder. The sanity check on a converted session gave a mean rate of 13.6 Hz and a PSTH that rises from ~10.5 Hz during the delay to ~24 Hz just after the go cue, which is the expected response profile.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A unit is kept only if `units/classification == 'good'` — the verdict of the region-specific quality-control classifier described in the spike-sorting white paper — **and** its `anno_name` maps onto one of the 14 coarse CCF divisions. No thresholds are applied to any individual quality metric (SNR, ISI violations, presence ratio, etc.), and the older `unit_quality` ('good'/'multi') column is not used. Sessions with zero good units are rejected outright. 57,560 units survive in the 143 retained sessions (a median of ~400 per session, min 90, max 923).

ii.
```python
classification = f['units/classification'][:].astype(str)
good = classification == 'good'
region_names, has_region = unit_regions(f, good, region_map)
good_idx = np.where(good)[0][has_region]
good = np.zeros(len(classification), dtype=bool)
good[good_idx] = True
region_names = region_names[has_region]
```

```python
regions = np.array([region_map.get(a.strip(), None) for a in anno],
                   dtype=object)
keep = np.array([r is not None for r in regions])
```

```python
if s['n_good_units'] == 0:
    reasons.append('no good units')
```

```python
'unit_selection':
    "units labelled 'good' by the region-specific quality-control "
    'classifiers of the spike-sorting white paper '
    '(doi:10.25378/janelia.24066108.v1) and carrying a CCF '
    'annotation',
```

iii. The AI identified `classification` as the white-paper QC classifier's output and used it as the single unit-quality criterion, counting 69,453 good units across all 174 files — the number the white paper reports. The additional CCF-annotation requirement is justified as matching the reference preprocessing, which keeps only units with both ephys and histology; in practice every QC-good unit in this dataset carries an annotation that the Allen ontology walk resolves, so this clause removes nothing. The AI cross-checked the resulting per-region unit totals against the data paper (thalamus 12,808; striatum 7,664; midbrain 7,495; medulla 2,928, before session selection).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams share one session-absolute clock, so no resampling or offset correction is needed. Alignment is done by adding the fixed set of go-cue-relative bin edges to each trial's go-cue timestamp, producing absolute edge times against which the spikes are binned. `metadata['temporal_alignment_event']` records "onset of the auditory go cue that ends the delay epoch".

ii.
```python
T_START = -2.5          # s relative to go cue
T_STOP = 1.5            # s relative to go cue
BIN_WIDTH = 0.05        # s
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))       # 80
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
```

```python
edges = go_time[:, None] + BIN_EDGES[None, :]          # (ntrials, nbins+1)
flat_edges = edges.ravel()
```

iii. The AI explicitly reasoned that the go cue is the alignment event used throughout both papers ("spike times in the reference preprocessing are relative to the go cue") and that the Decoder Task mandates it. It confirmed that `go_start_times/timestamps` is on the same clock as `units/spike_times`, `intervals/trials/start_time` and the camera timestamps by comparing their value ranges, so aligning reduces to a single additive shift per trial. The same `BIN_EDGES` / `BIN_CENTERS` grid is reused for the inputs and the tongue output, which guarantees bin *k* covers the same interval in every stream.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 of them per trial, spanning −2.5 s to +1.5 s relative to the go cue. Every trial in every session has exactly 80 timepoints. No rebinning or resampling of an intermediate representation happens: the spikes are binned once, directly onto this grid. `metadata['time_bin_size'] = 50.0` (ms), `off_start = -2.5`, `off_end = 1.5`, and the 80 bin centres are also stored in metadata.

ii.
```python
T_START = -2.5          # s relative to go cue
T_STOP = 1.5            # s relative to go cue
BIN_WIDTH = 0.05        # s
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))       # 80
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
```

```python
'time_bin_size': BIN_WIDTH * 1000.0,
...
'off_start': T_START,
'off_end': T_STOP,
'time_bin_centers': BIN_CENTERS.tolist(),
```

iii. The window and bin width come straight from the Decoder Task specification. The AI notes in the module docstring that this differs from the reference papers' own binning (40 ms width, 3.4 ms stride sliding histogram) and treats that as a permitted deviation because the instructions override it. Defining the grid once as module-level constants means all trials, sessions and data streams share it, which is what the target format requires ("Time bins should be the same size for all trials and sessions").

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `acquisition/BehavioralEvents/sample_start_times/timestamps` (the onsets of the sample epoch, during which the instruction tone is played) and the per-trial go-cue time. The tone used for a trial is the **last sample onset preceding that trial's go cue**. `intervals/trials/start_time` is used as a validity check.

ii.
```python
sample_starts = events['sample_start_times']['timestamps'][:]
# The tone is played during the sample epoch.  On early-lick trials the
# sample/delay epoch is replayed, so the tone the animal finally responded
# to is the last sample onset preceding the go cue.
idx = np.searchsorted(sample_starts, out['go_time']) - 1
out['tone_time'] = sample_starts[np.clip(idx, 0, len(sample_starts) - 1)]
out['tone_valid'] = (idx >= 0) & (out['tone_time'] >= out['start_time'])
```

iii. The AI established from the methods text and from the data that `sample_start_times` has *more* entries than trials, because licking during the sample or delay epoch triggers a replay of the epoch, so a trial can contain several sample onsets. It chose the last one before the go cue as the tone the animal actually acted on. It then measured the resulting tone-to-go interval across sampled sessions: the median is exactly −1.85 s (= 0.65 s sample + 1.2 s delay, as described in the methods), with a long negative tail on replay trials, and the tone fell inside the trial on 100 % of sampled trials.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value: for each bin, the bin centre (in go-cue-relative seconds) minus the tone's go-cue-relative time, i.e. seconds elapsed since tone onset. It is negative in bins before the tone. No clipping, rescaling or binarisation is applied. The value is broadcast to all 80 bins of the trial and stored as `float32` in row 0 of the input array. Observed range over the dataset: −1.525 to 11.894 s.

ii.
```python
tone_rel = tt['tone_time'][trials] - go_time                    # negative
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]
```

```python
inputs = [np.stack([time_from_tone[i], photostim[i]]).astype(np.float32)
          for i in range(n_trials)]
```

```python
'input_descriptions': [
    'time from the onset of the instruction tone (sample epoch) in '
    'seconds; negative before tone onset. On early-lick trials the '
    'delay epoch is replayed, so the last sample onset before the '
    'go cue is used.',
    ...
```

iii. Once the tone time is located, no further processing is required — the quantity is just a per-trial additive shift of the shared bin-centre grid. The AI kept it continuous (rather than binarising the onset) on the grounds that the Decoder Task lists it as "continuous, time-varying". It sanity-checked the first trial of a converted session, getting `[-0.62, -0.22, 0.18, 0.58, ...]` at every 8th bin, i.e. a monotone ramp in 0.4 s steps crossing zero shortly after the window start, as expected.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is defined *on* the neural binning grid: `BIN_CENTERS` is the same array of go-cue-relative bin centres whose edges (`BIN_EDGES`) were used to bin the spikes, so input bin *k* is by construction the same 50 ms interval as neural bin *k*. No separate alignment step or interpolation exists.

ii.
```python
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
```

```python
edges = go_time[:, None] + BIN_EDGES[None, :]          # neural
...
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]   # input
```

iii. Sharing one module-level grid between the two streams removes any possibility of drift between them. Both the tone times and the go-cue times come from the same `BehavioralEvents` group on the session-absolute clock, so the difference `tone - go` needs no correction.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the trials table columns `photostim_onset` and `photostim_duration` (stored as strings, with `'N/A'` on unstimulated trials), plus `start_time` (the onsets are measured from trial start) and the trial's go-cue time to re-express them on the go-cue-relative axis. `photostim_power` is read too, but only to identify control trials for the session-performance calculation.

ii.
```python
'photostim_onset': tr['photostim_onset'][:].astype(str),
'photostim_duration': tr['photostim_duration'][:].astype(str),
'photostim_power': tr['photostim_power'][:].astype(str),
```

```python
onset = tt['photostim_onset'][trials]
duration = tt['photostim_duration'][trials]
start_time = tt['start_time'][trials]
```

iii. The AI inspected the trials table and found these three columns stored as strings with `'N/A'` sentinels for unstimulated trials, requiring explicit conversion. It used the trials-table columns rather than the `photostim_start_times` / `photostim_stop_times` event series because the former are already per-trial and carry the `'N/A'` marker, avoiding a matching step (the event series has only 78 entries in the sample file, one per stimulated trial).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time series over the 80 bins, stored as `float32` in row 1 of the input array. For a stimulated trial the onset and offset are converted to go-cue-relative seconds and every bin that **overlaps** the `[on, off)` interval is set to 1 (bin start < off and bin end > on). Unstimulated trials (`'N/A'`) are skipped, leaving their row all zeros.

ii.
```python
photostim = np.zeros((n_trials, N_BINS), dtype=np.float32)
onset = tt['photostim_onset'][trials]
duration = tt['photostim_duration'][trials]
start_time = tt['start_time'][trials]
for i in range(n_trials):
    if onset[i] == 'N/A':
        continue
    # photostim_onset is given relative to the start of the trial
    on = start_time[i] + float(onset[i]) - go_time[i]
    off = on + float(duration[i])
    photostim[i] = ((BIN_EDGES[:-1] < off) & (BIN_EDGES[1:] > on))
```

iii. The Decoder Task asks for "whether photostimulation is on at every time point (discrete, time-varying)", so the AI represented it as a per-bin binary series rather than a per-trial flag. It used an overlap test rather than a bin-centre test, which marks a bin as stimulated if the light was on during any part of it. The AI verified the result by decoding the stimulated intervals back out of the converted data: across four sessions the stimulation windows cluster tightly at (−1.25 s, −0.70 s) and (−1.20 s, −0.65 s) relative to the go cue, i.e. the 0.5 s delay-epoch photoinhibition described in the data paper, with 63–132 stimulated trials per session (~25 %, as the paper states).

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stored onset is relative to `trials/start_time`; the code converts it to absolute time and then to go-cue-relative time by subtracting the trial's go-cue timestamp. The resulting interval is compared against the same `BIN_EDGES` used to bin the spikes, so the photostim series lives on exactly the neural bin grid.

ii.
```python
on = start_time[i] + float(onset[i]) - go_time[i]
off = on + float(duration[i])
photostim[i] = ((BIN_EDGES[:-1] < off) & (BIN_EDGES[1:] > on))
```

iii. The conversion chain `trial-relative → absolute → go-cue-relative` is required because the onsets are not stored on the alignment event's axis. Because all three quantities (`start_time`, `photostim_onset`, `go_time`) come from the same file on the same clock, no further correction is needed. The recovered stimulation windows falling exactly in the delay epoch confirms the arithmetic.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the NWB files, so choice is derived from two trials-table columns: `outcome` (`'hit'` / `'miss'` / `'ignore'`) and `trial_instruction` (`'left'` / `'right'`). A hit means the animal licked the instructed side, a miss means it licked the opposite side, and an ignore means it did not lick at all. `left_lick_times` / `right_lick_times` were used only to validate this derivation, not to produce it.

ii.
```python
outcome_str = tt['outcome'][trials]
instruction = tt['instruction'][trials]
```

```python
# 'hit' means the animal licked the instructed direction, 'miss' that it
# licked the other one, 'ignore' that it did not lick at all.  Checked
# against the first lick after the go cue: they agree on > 99.8 % of trials.
```

iii. The AI searched the file for a direct choice field, found none, and derived it from the instruction × outcome contingency. It then validated the derivation against the raw lick event streams: it computed the first lick after the go cue from `left_lick_times` / `right_lick_times` and compared its side to the derived choice, obtaining **100 % agreement** on the checked session (the code comment conservatively says "> 99.8 %").

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0 = left, 1 = right, 2 = no lick, with `output_values[0] = ['left', 'right', 'no lick']`. The default is 2 (no lick), which is then overwritten for the hit/miss trials. The value is one per trial, so it is repeated across all 80 bins in row 0 of the `(4, 80)` output array, stored as `int64`.

ii.
```python
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ...
]
```

```python
choice = np.full(n_trials, 2, dtype=np.int64)                   # no lick
licked_left = (((outcome_str == 'hit') & (instruction == 'left'))
               | ((outcome_str == 'miss') & (instruction == 'right')))
licked_right = (((outcome_str == 'hit') & (instruction == 'right'))
                | ((outcome_str == 'miss') & (instruction == 'left')))
choice[licked_left] = 0
choice[licked_right] = 1
```

```python
outputs = [np.stack([np.full(N_BINS, choice[i]),
                     np.full(N_BINS, outcome[i]),
                     np.full(N_BINS, early_lick[i]),
                     tongue[i]]).astype(np.int64)
           for i in range(n_trials)]
```

iii. Left = 0 / right = 1 follows the ordering given in the Decoder Task, and a third class is added for the `ignore` trials where choice is genuinely undefined. Starting from an all-"no lick" array and overwriting means any trial with an unexpected `outcome` string falls into the no-lick class instead of producing an invalid label. Per-trial values are broadcast across the 80 bins so that all four outputs can live in one `(n_output, n_timepoints)` array, which the target format requires and which the instructions encourage ("If at all possible, make it time-varying").

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of `intervals/trials`, which already stores exactly the three strings `'ignore'`, `'miss'` and `'hit'`.

ii.
```python
'outcome': tr['outcome'][:].astype(str),
```
```python
outcome_str = tt['outcome'][trials]
```

iii. No derivation is needed: the trials table records the outcome explicitly with exactly the three categories the Decoder Task asks for. The AI confirmed the value set by enumerating the column.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to 0 = ignore, 1 = miss, 2 = hit and written into row 1 of the output array, repeated across all 80 bins. The array is initialised to −1 and each of the three labels is assigned by boolean mask.

ii.
```python
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ...
]
```

```python
outcome = np.full(n_trials, -1, dtype=np.int64)
outcome[outcome_str == 'ignore'] = 0
outcome[outcome_str == 'miss'] = 1
outcome[outcome_str == 'hit'] = 2
```

iii. The code ordering follows the Decoder Task's "(ignore, miss, hit)" listing. Outcome is a per-trial property, so it is broadcast across bins like the other per-trial outputs. The AI checked the resulting distribution per session (e.g. `outcome counts [13 35 469]`) and the dataset-wide fractions (ignore 0.109, miss 0.153, hit 0.739), which are consistent with a well-trained animal.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of `intervals/trials`, which holds the strings `'no early'` and `'early'`.

ii.
```python
'early_lick': tr['early_lick'][:].astype(str),
```
```python
early = tt['early_lick'][trials]
```

iii. The trials table flags early licking explicitly, so no derivation from the lick event streams is needed. The AI noted from the methods that an early lick occurs during the sample or delay epoch, i.e. before the go cue, so the event that sets the flag falls inside the −2.5 s analysis window even though the flag is stored per trial — which is what makes it decodable at all.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 = no, 1 = yes by a direct equality test against `'early'`, then repeated across all 80 bins in row 2 of the output array.

ii.
```python
OUTPUT_VALUES = [
    ...
    ['no', 'yes'],
    ...
]
```
```python
early_lick = (early == 'early').astype(np.int64)
```
```python
outputs = [np.stack([..., np.full(N_BINS, early_lick[i]), ...]) ...]
```

iii. The 0 = no / 1 = yes coding follows the Decoder Task. Using `== 'early'` rather than a dictionary lookup means any unexpected string is treated as "no early" instead of raising; the AI verified the column only ever contains the two expected values. Dataset-wide the class balance is 88.5 % / 11.5 %, matching the reference's 88.5 % / 11.5 %.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is an `(n_frames, 3)` array of `(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps` at ~300 Hz. Column 1 is the y-position; column 2 is the DeepLabCut likelihood, which decides visibility.

ii.
```python
tracking = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
ts = tracking['timestamps'][:]
data = tracking['data'][:]
y = data[:, 1]
visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESHOLD
```

iii. The AI enumerated the `BehavioralTimeSeries` contents of all 174 files and found `Camera0_side_TongueTracking` present in every one (three sessions additionally carry a `Camera3_side_` set, which is ignored — the method paper states "only the side-view frames were used"). The `(x, y, likelihood)` column layout was read off the series description rather than assumed.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps:

1. **Visibility mask.** A frame counts only if its DeepLabCut likelihood exceeds 0.5; the tracker still emits a coordinate when the tongue is retracted.
2. **Binning.** For each trial and each of the 80 bins, the mean y over the *visible* frames in that bin is computed, via cumulative sums of `y·visible` and of `visible` differenced at the bin edges. A bin with no visible frame (or no frame at all) becomes NaN.
3. **Per-session discretisation.** The 40th and 60th percentiles are taken over all finite bin means of that session's retained trials, and each bin is assigned class 0/1/2; NaN bins get class 3.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.5
```

```python
# cumulative sums let us take the per-bin mean over visible frames with two
# lookups per bin edge instead of a python loop over frames
csum_y = np.concatenate([[0.0], np.cumsum(np.where(visible, y, 0.0))])
csum_n = np.concatenate([[0], np.cumsum(visible.astype(np.int64))])

edges = go_time[:, None] + BIN_EDGES[None, :]
idx = np.searchsorted(ts, edges)                        # (ntrials, nbins+1)
n_visible = np.diff(csum_n[idx], axis=1)
sum_y = np.diff(csum_y[idx], axis=1)
with np.errstate(invalid='ignore', divide='ignore'):
    mean_y = np.where(n_visible > 0, sum_y / np.maximum(n_visible, 1), np.nan)
return mean_y
```

iii. The AI measured the likelihood distribution across sampled sessions and found it sharply bimodal (77–92 % of frames below 0.01, 6–21 % above 0.99), concluding in the code comment that "the exact value of the threshold is inconsequential". Averaging only over visible frames is justified because the tracker reports a spurious position for the retracted tongue, which would otherwise contaminate the mean and skew the percentiles. The AI explicitly did **not** follow the method paper's own treatment (imputing the mean value while the tongue is occluded), because the Decoder Task defines a separate "not visible" category. It checked the visible-tongue fraction per bin in a converted session: ~0 before the go cue, rising to 0.86 immediately after it — the expected licking profile.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes: 0 = below the 40th percentile, 1 = between the 40th and 60th, 2 = above the 60th, 3 = not visible. The percentiles are computed **per session** over the finite (visible) bin means of that session's retained trials. The array is initialised to 3, and the three visible classes are assigned in the order "≤ p60 → 1", then "< p40 → 0", then "> p60 → 2". If a session has fewer than 10 finite bin means, every bin is left as class 3.

ii.
```python
OUTPUT_VALUES = [
    ...
    ['<40th pct', '40-60th pct', '>60th pct', 'not visible'],
]
```

```python
# tongue y-position, discretised with per-session percentiles
finite = tongue_y[np.isfinite(tongue_y)]
tongue = np.full(tongue_y.shape, 3, dtype=np.int64)             # not visible
if finite.size >= 10:
    p40, p60 = np.percentile(finite, [40, 60])
    visible = np.isfinite(tongue_y)
    tongue[visible & (tongue_y <= p60)] = 1
    tongue[visible & (tongue_y < p40)] = 0
    tongue[visible & (tongue_y > p60)] = 2
```

iii. The 40th/60th split and the per-session scope come directly from the Decoder Task. The AI computed the percentiles over the *binned* values (the same quantity that is discretised), and over the analysis windows that actually enter the dataset, so the three visible classes come out at exactly 40 % / 20 % / 40 % of the visible bins (observed fractions 0.1058 / 0.0529 / 0.1058 of all bins, against 0.2644 visible). The `finite.size >= 10` guard prevents `np.percentile` from producing meaningless edges on a session with almost no tracking. The AI also audited the per-session visible fraction (median 0.240, 1st–99th percentile 0.098–0.499) and flagged the two outlier sessions (0.5 % and 99.2 % visible) in `metadata['tongue_tracking_note']` rather than silently dropping them.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as the spikes and the go cues, so each trial's 81 absolute bin edges (the same `go_time + BIN_EDGES` array used for the spikes) are looked up in the camera timestamp array with `np.searchsorted`, and the frames between consecutive edges form bin *k*. There is no interpolation or offset correction. Bins with no camera frame — the video is trial-gated, so the leading bins of a short trial fall in the inter-trial interval — collapse to zero visible frames and become class 3.

ii.
```python
edges = go_time[:, None] + BIN_EDGES[None, :]
idx = np.searchsorted(ts, edges)                        # (ntrials, nbins+1)
n_visible = np.diff(csum_n[idx], axis=1)
sum_y = np.diff(csum_y[idx], axis=1)
```

iii. Using literally the same edge array for the video as for the spikes guarantees bin *k* of the tongue output covers the same 50 ms interval as bin *k* of the firing rates. The AI quantified the trial-gating effect before writing the code: across sampled sessions the fraction of bins containing no video frame at all is 0.00–0.25 %, which it judged small enough to fold into the "not visible" class rather than handle separately (the code comment records "< 0.3 % of all bins").

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six distinct cases, handled by exclusion where the datum was never recorded and by an explicit category or fallback where the measurement legitimately has no value:

- **Session never quality-controlled** (`classification` all `'unlabelled'`): zero good units, so the session is rejected by the session filter.
- **Trials with no ephys coverage** (9 sessions where the recording starts after the behaviour): detected via `units/obs_intervals` and dropped, rather than emitted as 4 s of silence.
- **Free-water / auto-water trials**: dropped as non-choice trials.
- **Trials with no locatable tone onset**: `tone_valid` guard, plus `np.clip` so the index lookup cannot go out of bounds.
- **Electrodes with no registered CCF coordinate** (~1.5 % have `x = NaN`): the hemisphere falls back to the side the probe was targeted at, parsed from the electrode `location` JSON.
- **Frames/bins with no visible tongue**: excluded from the bin mean; a bin left with no visible frame becomes the explicit `'not visible'` class, and a session with fewer than 10 usable bins is labelled entirely "not visible".

One case is deliberately **not** treated as an error: on error trials the trial interval ends at the incorrect lick, so the tail of the 4 s window contains no spikes. These trials are kept with zero-filled trailing bins and the limitation is recorded in metadata.

ii.
```python
if len(good_idx) == 0:
    return np.zeros(len(trial_start), dtype=bool)
```

```python
side = np.where(ml_unit >= ML_MIDLINE, 'left', 'right')
unknown = np.isnan(ml_unit)
side[unknown] = target_side[electrode_index][unknown]
```

```python
with np.errstate(invalid='ignore', divide='ignore'):
    mean_y = np.where(n_visible > 0, sum_y / np.maximum(n_visible, 1), np.nan)
```

```python
'known_limitation':
    'spike times in the NWB files are only stored inside the trial '
    'intervals, which end with the incorrect lick on error trials, '
    'so bins of the analysis window that fall outside the trial '
    'interval contain no spikes (~15 % of trials are affected at '
    'the end of the window, ~3 % at the beginning)',
```

iii. The AI's stated principle is that data which was never recorded should be excluded rather than fabricated, while a measurement that legitimately has no value should be represented as its own category. It justified keeping the zero-tailed error trials explicitly: "Dropping those trials would remove essentially every 'miss' trial and collapse the outcome output to two classes." It also chose to *document rather than drop* the two sessions with unreliable tongue tracking, on the grounds that their neural data and their other three outputs are unaffected. The final converted dataset passes `verify_data_format` with no errors and no warnings — in particular, no remaining all-zero trials.

## 10-a. What are the most time-consuming steps of the code?

i. Ranked by cost:

1. **Reading the HDF5 arrays** — `units/spike_times` (up to ~11.5 M doubles per session) and the tongue tracking array (~680 k × 3), both read whole into memory per session. This dominates per-session time (0.7 s for a 205-unit session, ~4 s for the largest).
2. **The per-unit `searchsorted` loop** in `bin_spikes`: one binary search per unit over `n_trials × 81` edges.
3. **Pickling the 9.9 GB result** — 15.4 s.
4. The metadata **scan pass** over all 174 files — 2.2 s in total.
5. Writing and reading back the ~10 GB of per-session cache pickles in `/tmp`.

The AI mitigated (1) and (2) by running sessions in parallel across a `multiprocessing.Pool` (24 workers for the production run), which brought the full 143-session conversion to well under two minutes of wall clock.

ii.
```python
with Pool(args.nproc) as pool:
    for i, result in enumerate(pool.imap(convert_session, jobs)):
        sessions.append(result)
        print('[%3d/%3d] %s: %d trials, %d neurons (%.1f s)'
              % (i + 1, len(jobs), result['info']['session_name'],
                 result['info']['n_trials'], result['info']['n_neurons'],
                 time.time() - t0), flush=True)
```

```python
if cache_file is not None:
    tmp = cache_file + '.tmp%d' % os.getpid()
    with open(tmp, 'wb') as fh:
        pickle.dump(result, fh, protocol=4)
    os.replace(tmp, cache_file)
```

iii. The work is I/O- and binary-search-bound, and both scale with the data actually needed, so the AI's optimisation was concurrency rather than algorithmic change. The per-session cache was added so that a crash or a tweak to the assembly stage does not force a re-read of every NWB file; the AI used it exactly that way (it re-ran the assembly step from cache after adding metadata fields, which took 18.6 s instead of a full re-conversion). It instrumented every stage with elapsed-time prints.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops remain, of which one is genuinely avoidable:

- **`for i in range(n_trials)` in the photostim block** — the only clearly vectorisable loop. It parses two strings and evaluates an 80-element comparison per trial; it could be replaced with a single masked array conversion and one broadcast comparison (as the reference does).
- **`for i, u in enumerate(unit_ids)` in `bin_spikes`** — not collapsible: each unit has a different number of spikes, so there is no single sorted array to search. The trial dimension is already vectorised by flattening all 81 × n_trials edges into one `searchsorted` call.
- **`for u in good_idx` in `observed_trials`** — already short-circuited, since every unit in a session shares the same observation intervals, so the body executes once.
- The per-electrode `json.loads` comprehension in `unit_regions` and the per-unit `region_map.get` comprehension — small, one pass over the electrode/unit tables.

The tongue binning, by contrast, is fully vectorised across trials and frames with cumulative sums, avoiding a per-trial loop.

ii.
```python
for i in range(n_trials):
    if onset[i] == 'N/A':
        continue
    # photostim_onset is given relative to the start of the trial
    on = start_time[i] + float(onset[i]) - go_time[i]
    off = on + float(duration[i])
    photostim[i] = ((BIN_EDGES[:-1] < off) & (BIN_EDGES[1:] > on))
```

```python
for i, u in enumerate(unit_ids):
    st = spike_times[start[u]:stop[u]]
    counts = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
    rates[:, :, i] = np.diff(counts, axis=1) / BIN_WIDTH
```

```python
target_side = np.array([json.loads(s)['brain_regions'].split(' ')[0]
                        for s in location])
```

iii. The AI's implicit position is that none of the remaining loops is on the hot path — the runtime is dominated by HDF5 reads, and per-session times (0.7–4 s) are consistent with I/O rather than Python overhead. The one avoidable loop (photostim) runs ~500 times per session with trivial bodies. The loops that would matter, over units and over frames, are respectively unavoidable (ragged spike storage) and already vectorised.

## 10-c. What processing does the code repeat multiple times?

i. The two-pass design means several quantities are computed **twice per session**:

- `_trial_table(f)` — the whole trials table plus go/tone times, once in `scan_session` and again in `convert_session`. That includes `observed_trials(f, ...)`, which reads the full `obs_intervals` array both times.
- `session_performance(tt)` — computed in `scan_session` to decide selection, and again in `convert_session` purely so the value can be stored in `metadata['session_info']`.
- `units/classification` — read in `observed_trials`, again directly in `scan_session`, and a third time in `convert_session`.
- `trial_mask(tt)` — evaluated inside `session_performance` and again in `convert_session`.

`build_region_map()` is built once in the parent, but the resulting dict is pickled into every job tuple and so is re-serialised once per session.

Per-session results are also serialised twice on a cold run: once into the `/tmp` cache, once into the final pickle.

ii.
```python
def scan_session(path):
    """Metadata needed to decide whether a session is included."""
    with h5py.File(path, 'r') as f:
        tt = _trial_table(f)
        performance, n_left, n_right = session_performance(tt)
        classification = f['units/classification'][:].astype(str)
```

```python
def convert_session(args):
    ...
    with h5py.File(path, 'r') as f:
        tt = _trial_table(f)
        classification = f['units/classification'][:].astype(str)
        ...
        performance, n_left, n_right = session_performance(tt)
```

```python
jobs.append((s['path'], region_map, cache))
```

iii. The duplication is a deliberate trade: the scan pass reads only the small trials table and the classification column (2.2 s for all 174 files) and lets the expensive conversion skip the 31 sessions that would be rejected, so it is a net saving. The recomputation of `session_performance` inside `convert_session` is not justified by anything except convenience — the value was already available in the scan result that produced the job.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, none of which affect correctness:

- **Per-trial outputs stored as `int64`.** All four outputs are small non-negative integers (max value 3); `int8` would be 8× smaller. This is the single largest avoidable cost, inflating the 9.9 GB pickle.
- **`stop_time` is read from the trials table and never used.**
- **`session_performance` is recomputed in `convert_session`** only to populate metadata.
- **The `/tmp/map_converted_sessions` cache** writes and re-reads ~10 GB that a single clean run never needs.
- **`build_region_map()` walks the entire ~1,300-entry Allen ontology**, while only 293 distinct annotation strings occur in the dataset; and the CCF-annotation requirement it feeds turns out to exclude zero units.
- **Diagnostic metadata** the decoder ignores: `rejected_sessions`, `n_correct_left` / `n_correct_right`, `performance`, `n_photostim_trials`, `fraction_tongue_visible`, `tongue_tracking_note`, `time_bin_centers`, `n_trials_total`.
- **`photostim_power`** is read for every trial but used only in the session-performance calculation.

ii.
```python
outputs = [np.stack([np.full(N_BINS, choice[i]),
                     np.full(N_BINS, outcome[i]),
                     np.full(N_BINS, early_lick[i]),
                     tongue[i]]).astype(np.int64)
           for i in range(n_trials)]
```

```python
out = {
    'start_time': tr['start_time'][:],
    'stop_time': tr['stop_time'][:],        # never used
    ...
```

```python
'rejected_sessions': [
    {'session_name': os.path.basename(s['path']),
     'subject': s['subject'],
     'reason': s['rejected_because']} for s in rejected],
```

iii. The AI's rationale for the diagnostic metadata is provenance: because it applies a session filter that the source data does not encode, it records every rejected session and its reason so the exclusion can be audited, and it flags the sessions whose tongue tracking is unreliable rather than silently dropping them. That is a defensible use of a few kilobytes. The `int64` output dtype and the unused `stop_time` are not justified anywhere; they appear to be defaults that were never revisited.
