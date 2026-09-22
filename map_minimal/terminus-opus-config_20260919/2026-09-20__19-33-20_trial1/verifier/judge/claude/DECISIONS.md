# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI 000363) is one NWB file per session, laid out as `/app/data/sub-<subject_id>/<file>.nwb`. The AI finds every session with a single sorted glob over that layout (174 files) and converts each one independently. Unlike the reference solution it does **not** use `pynwb`; it opens the raw HDF5 with `h5py` and reads the NWB groups by path (`intervals/trials`, `units`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, `general/subject`, `general/extracellular_ephys/electrodes`). Each file is actually opened **twice**: once in `session_info()` to read the small tables needed for the session-inclusion decision, and once in `process_session()` to read the heavy arrays. Sessions are converted in parallel with a `multiprocessing.Pool` (default 16 workers, run with 24); each worker pickles its session result to `/tmp/converted_sessions/` and returns only the path, and `main()` reads those back to assemble the final dictionary.

ii.
```python
DATA_DIR = '/app/data'
...
paths = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
if args.limit:
    paths = paths[:args.limit]
print('found %d nwb sessions' % len(paths))

with Pool(args.nproc) as pool:
    files = pool.map(_worker, paths)
```

```python
def session_info(path):
    """Read the small tables needed to decide whether a session is used."""
    with h5py.File(path, 'r') as f:
        t = f['intervals/trials']
        info = dict(
            path=path,
            subject=f['general/subject/subject_id'][()].decode(),
            session_id=f['identifier'][()].decode(),
            start=np.asarray(t['start_time'][:]),
            ...
            ngood=int((f['units']['classification'][:] == b'good').sum()),
        )
    return info
```

```python
def process_session(path):
    info = session_info(path)
    used, perf, ncl, ncr = session_is_used(info)
    if not used:
        return None

    start, stop = info['start'], info['stop']
    with h5py.File(path, 'r') as f:
        ...
```

```python
def _worker(path):
    try:
        out = process_session(path)
    except Exception as exc:  # pragma: no cover - defensive
        print('FAILED %s: %r' % (path, exc), flush=True)
        return None
    ...
    fn = os.path.join(TMP_DIR, os.path.basename(path) + '.pkl')
    with open(fn, 'wb') as fh:
        pickle.dump(out, fh, protocol=4)
```

iii. From the trajectory (steps 10–14, 24): the AI first read the reference repo's preprocessing (`VideoAnalysisUtils/preprocessing_DJ_2022Aug.py`) and found it operates on DataJoint `.mat` exports that are not the distributed format, so it had to re-implement the same logic against NWB. It checked that `pynwb` was installed but chose `h5py` after dumping the raw HDF5 structure, because it only needs a handful of datasets per file and wanted cheap metadata-only scans over all 174 files ("Scan all NWB files quickly (metadata only)"). The two-pass open is deliberate: the cheap pass decides session inclusion so that rejected sessions never pay for the heavy spike/video reads. Multiprocessing plus temp pickles was chosen because the full output is ~7 GB and returning those arrays through the `Pool` pipe would be expensive.

## 1-b. How are the data split into subjects?

i. The animal identity is taken from the NWB subject field `general/subject/subject_id` (a numeric string such as `'440956'`) and carried on each session result. At assembly, `subjects` is built as the list of unique ids in first-appearance order (not sorted), and `subject_idx` is the index of each session's subject into that list. No grouping by folder name or by the mouse name embedded in `identifier` (e.g. `SC015_...`) is performed. Because of the session-level behavioural filter (1-c), only **25** of the dataset's 28 subjects appear in the output.

ii.
```python
subject=f['general/subject/subject_id'][()].decode(),
```

```python
if s['subject'] not in subjects:
    subjects.append(s['subject'])
subject_idx.append(subjects.index(s['subject']))
...
data['subjects'] = subjects
data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
```

iii. The trajectory (steps 14, 24, 48) records that the AI scanned all 174 files for `subject`, `n_trials`, `n_good_units` and found 28 mice, and it reports 25 mice in the final dataset after session selection. The justification is simply that `subject_id` is the canonical per-file animal identifier in NWB, so no inference is needed; the subject directory name is derived from the same id.

## 1-c. How are the data split into sessions?

i. One NWB file is one session, so no splitting is required; `identifier` (e.g. `SC015_20190207_120657_s1`) is stored as the session id and session order follows the sorted file list. The substantive decision here is **session selection**: the AI applies the data paper's stated session-inclusion criteria — overall behavioural performance > 65 % computed on control (non-photostimulation) trials excluding early-lick trials, **and** at least 50 correct lick-left and 50 correct lick-right control trials — plus a requirement of at least one QC-good unit. This reduces 174 sessions to **106** on behaviour alone and **105** after the good-unit requirement. (The reference solution applies no behavioural session filter and keeps 173 of 174.)

ii.
```python
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50
```

```python
def session_is_used(info):
    """Data-paper session selection criteria (Chen et al. STAR Methods)."""
    ctrl = (info['photostim_onset'] == 'N/A') & (info['early'] == 'no early')
    if ctrl.sum() == 0:
        return False, 0.0, 0, 0
    perf = float(np.mean(info['outcome'][ctrl] == 'hit'))
    ncl = int(np.sum(ctrl & (info['outcome'] == 'hit') & (info['instruction'] == 'left')))
    ncr = int(np.sum(ctrl & (info['outcome'] == 'hit') & (info['instruction'] == 'right')))
    used = (perf > MIN_PERFORMANCE and ncl >= MIN_CORRECT_PER_DIRECTION
            and ncr >= MIN_CORRECT_PER_DIRECTION and info['ngood'] > 0)
    return used, perf, ncl, ncr
```

```python
info = session_info(path)
used, perf, ncl, ncr = session_is_used(info)
if not used:
    return None
```

iii. The AI justified this at length (steps 26, 31, 32, 37, 44, 63, 81). `methods.txt` states verbatim: "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each", with performance computed "as the fraction of correct control trials (i.e. no photostimulation), excluding any early lick trials". The AI implemented exactly that, then cross-checked the result against the method paper's own text extracted from the PDF and found the criteria reproduce **exactly** the n = 106 sessions the method paper analyses ("106 sessions matches the paper (n=106 sessions) and matches my filter"). It explicitly considered computing the criteria on observed trials only (step 63) and kept the behavioural-trial version to preserve the n = 106 match. The one further drop is a session with no unit passing spike-sorting QC.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table (`intervals/trials`). Each trial's go cue is obtained by assigning every `go_start_times` event to the trial whose `[start_time, ...)` interval contains it (`searchsorted` on `start_time`), keeping the first go cue per trial; a trial with no go cue gets `NaN` and is dropped. The go cue then defines the trial's analysis window.

ii.
```python
def go_cue_times(f, start, stop):
    """Go cue time for every trial (NaN if the trial has no go cue)."""
    go = _event_times(f, 'go_start_times')
    idx = np.searchsorted(start, go, 'right') - 1
    out = np.full(len(start), np.nan)
    for i, g in zip(idx, go):
        if 0 <= i < len(out) and np.isnan(out[i]):
            out[i] = g
    return out
```

```python
g = go[trials]
edges = g[:, None] + OFF_START + np.arange(N_BINS + 1)[None, :] * BIN_SIZE
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
```

iii. From the trajectory (steps 13, 33, 43): the AI verified that all `BehavioralEvents` timestamps are session-absolute, and that trial-phase events must be mapped back to trials through the trials-table intervals. It explicitly noted that sample/delay events can occur multiple times per trial (epoch replay after an early lick) and therefore wrote a mapping that tolerates multiple events per trial rather than assuming a one-to-one correspondence; the go cue is taken as the first event within the trial and a missing go cue makes the trial unusable.

## 1-e. How are trials filtered based on quality controls?

i. Within a selected session, a trial is kept only if **all** of the following hold:
1. it has a go cue (`~isnan(go)`) and an instruction-tone onset (`~isnan(tone)`);
2. it is not a `free_water` trial and not an `auto_water` trial;
3. it lies inside the units' spike observation intervals (`units/obs_intervals`), i.e. ephys was actually running;
4. after binning, it contains at least one spike across all units and bins.

A session is dropped if fewer than 2 trials survive. Early-lick, no-response (`ignore`), error (`miss`) and photostimulation trials are deliberately **kept**, because they are required decoder inputs/outputs.

ii.
```python
keep = (~np.isnan(go)) & (~np.isnan(tone)) & (~info['free_water']) & (~info['auto_water'])
keep &= observed_trials(f, start)
trials = np.where(keep)[0]
if len(trials) < 2:
    return None
```

```python
def observed_trials(f, start):
    """Boolean mask of the trials for which spikes were actually recorded."""
    u = f['units']
    oix = np.asarray(u['obs_intervals_index'][:])
    if len(oix) == 0:
        return np.zeros(len(start), dtype=bool)
    counts = np.diff(np.concatenate([[0], oix]))
    # all units of a session share the same observation intervals in this dataset
    assert len(np.unique(counts)) == 1, 'units differ in their observation intervals'
    oi = np.asarray(u['obs_intervals'][0:oix[0]])
    mask = np.zeros(len(start), dtype=bool)
    idx = np.searchsorted(start, oi[:, 0] + 1e-6, 'right') - 1
    idx = idx[(idx >= 0) & (idx < len(start))]
    mask[idx] = True
    return mask
```

```python
# a handful of trials at the very end of a recording contain no spikes
# at all (the recording stopped); they carry no neural information
nonempty = rates.sum(axis=(0, 2)) > 0
if not np.all(nonempty):
    trials = trials[nonempty]
    rates = rates[:, nonempty, :]
    edges = edges[nonempty]
    centers = centers[nonempty]
    g = g[nonempty]
    if len(trials) < 2:
        return None
```

iii. The free-water/auto-water exclusion is taken directly from the reference repository's own trial mask, which the AI read (`population_decoding_utils.get_regular_trial_mask`: "No early lick, no auto water, no free water, no no response trials, no stimulation"); the code comment says "as in the reference code's regular-trial mask". The AI explicitly departed from the rest of that mask because "Photostimulation, early-lick, error and no-response trials are kept because they are decoder inputs/outputs in this task" (step 32: "exclusion of photostim/free-water/early-lick/ignore trials for their analyses (but our decoder task requires those trials as outputs/inputs)").

The `obs_intervals` filter was discovered empirically after the format verifier warned about all-zero trials (steps 57–62): "in some NWB files the trials table spans the whole behavioural session while the ephys units are only observed during a subset of trials (obs_intervals cover only the first ~206 of 582 trials)... My converter currently emits all-zero neural data for unobserved trials." It then scanned sessions and established that all units in a session share identical observation intervals and that 9 sessions cover only a prefix of trials. The residual all-zero-trial drop was added after a single such trial survived (step 69): "The last observed trial of that session genuinely has no spikes (recording ended)."

The AI also explicitly considered and rejected dropping trials whose ±window is not fully covered by the observation interval (step 61): because miss trials end ~0.7 s after the go cue, that would "drop nearly all miss trials", destroying the outcome output; it kept them and documented the truncation as a `known_caveat`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (with `units/spike_times_index` for the ragged offsets), restricted to units whose `units/classification == b'good'`. The go cue times from `acquisition/BehavioralEvents/go_start_times` provide the bin edges. `units/anno_name` and the electrode `location` JSON are read as well, but only to label brain regions, not to build the rates.

ii.
```python
u = f['units']
good = np.where(u['classification'][:] == b'good')[0]
sti = np.asarray(u['spike_times_index'][:])
anno = _dec(u['anno_name'][:])
# probe target region (used only to identify ALM recordings)
loc = f['general/extracellular_ephys/electrodes']['location'][:]
uel = np.asarray(u['electrodes'][:])
targets = np.array([json.loads(loc[i].decode()).get('brain_regions', '') for i in uel])
```

iii. The trajectory (steps 12, 24, 26) shows the AI inspected the `units` table fields (`classification`, `unit_quality`, `anno_name`) and concluded that `classification` is the output of the white-paper QC classifier, and that `spike_times` is the only neural representation present in the files.

## 2-b. How is the `neural` data processed?

i. Per-bin firing rate in spikes/s. For each good unit the ragged spike-time slice is read, checked for sortedness (and sorted if necessary), and `np.searchsorted` against the flattened array of all trials' bin edges gives running spike counts; differencing adjacent edges gives the per-bin count, which is divided by the 50 ms bin width. No smoothing, no baseline subtraction, no normalisation, no firing-rate threshold on units. Rates are stored as `float32`, one `(n_neurons, 80)` array per trial.

ii.
```python
rates = np.zeros((len(good), len(trials), N_BINS), dtype=np.float32)
flat_edges = edges.ravel()
for k, iu in enumerate(good):
    a = 0 if iu == 0 else sti[iu - 1]
    spikes = np.asarray(u['spike_times'][a:sti[iu]])
    if spikes.size and np.any(np.diff(spikes) < 0):
        spikes = np.sort(spikes)
    pos = np.searchsorted(spikes, flat_edges).reshape(len(trials), N_BINS + 1)
    rates[k] = np.diff(pos, axis=1).astype(np.float32) / BIN_SIZE
```

```python
neural=[np.ascontiguousarray(rates[:, i, :]) for i in range(len(trials))],
```
with metadata `'neural_units': 'spikes/s (spike count per 50 ms bin divided by the bin width)'`.

iii. The AI read the reference `sliding_histogram(..., rate=True)` in `preprocessing_DJ_2022Aug.py`, which returns `binSpikes/bin_width`, and matched that definition, replacing the repo's 40 ms window / 3.4 ms stride with the task-specified 50 ms non-overlapping bins (step 81: "the reference pipeline's 40 ms/3.4 ms sliding window is replaced by the task-specified 50 ms bins"). It also checked `population_decoding_utils.py` and concluded that the repo's ">2 Hz" neuron filter "is for r2 analyses (comparison of methods), not necessarily the basic dataset" and so did not apply it (step 42). The sortedness guard was added defensively before the full run (step 64).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept if and only if `units/classification == b'good'`, i.e. the verdict of the region-specific spike-sorting QC classifiers described in the accompanying white paper. No thresholds on individual quality metrics, no firing-rate threshold, and the older `unit_quality` column is not used. A session with zero good units is excluded (via `info['ngood'] > 0` in `session_is_used`). This yields 41,197 units across the 105 retained sessions (69,453 across all 174 files).

ii.
```python
ngood=int((f['units']['classification'][:] == b'good').sum()),
```

```python
used = (perf > MIN_PERFORMANCE and ncl >= MIN_CORRECT_PER_DIRECTION
        and ncr >= MIN_CORRECT_PER_DIRECTION and info['ngood'] > 0)
```

```python
good = np.where(u['classification'][:] == b'good')[0]
```
with metadata `'unit_selection': 'units labelled "good" by the region-specific quality-control classifiers of the accompanying spike-sorting white paper (units/classification in the NWB files), as used in both reference papers'`.

iii. Step 26: "total good units 69453 matches paper's 69,943 reported good units (classification column = QC classifier output)". The AI validated the column choice by reproducing the number quoted in `methods.txt` ("the dataset consisted of 69,943 good units recorded across 173 behavioral sessions") to within 0.7 %, confirming that `classification` — rather than `unit_quality` — is the classifier output used by both papers.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go-cue onset**. All NWB timestamps (spikes, behavioural events, video) are on one session-absolute clock, so no resampling or offset correction is needed: each trial's 81 bin edges are formed by adding the fixed relative grid to that trial's go-cue time, and spikes are binned against those absolute edges directly.

ii.
```python
OFF_START = -2.5           # s relative to go cue
OFF_END = 1.5              # s relative to go cue
...
g = go[trials]
# bin edges / centres, aligned to the go cue
edges = g[:, None] + OFF_START + np.arange(N_BINS + 1)[None, :] * BIN_SIZE
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
```
metadata: `'temporal_alignment_event': 'onset of the auditory go cue (end of the delay epoch)'`, `'off_start': -2.5`, `'off_end': 1.5`.

iii. Step 13/14: the AI explicitly checked "timestamps semantics (absolute vs trial-relative)" and concluded "Timestamps are session-absolute." It also sanity-checked the alignment visually with `--plot-samples` (step 79): "Alignment checks out: photostim appears in the delay epoch (session-specific onset -1.2 s, 0.5 s duration), tongue becomes visible right after the go cue, and population rate rises after the go cue."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial spanning −2.5 s to +1.5 s around the go cue. The same grid is used for every trial and every session, so all trials have exactly 80 timepoints. There is no rebinning of an intermediate representation: spike times are binned once, directly at 50 ms. `metadata['time_bin_size']` is recorded as 50.0 ms.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05            # s (50 ms bins, as required by the task)
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80
```
```python
data['metadata'] = {
    ...
    'time_bin_size': BIN_SIZE * 1000.0,
    'n_timepoints': N_BINS,
```

iii. The window and bin width are dictated by the Decoder Task section of the instructions. The AI noted the deviation from the reference pipeline's sliding 40 ms / 3.4 ms bins and justified it as required by the task specification (step 81).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times` (sample-epoch/instruction-tone onsets), mapped to trials through the trials-table `start_time` boundaries, together with each trial's go cue. When a trial contains several sample onsets (epoch replay after an early lick), the **last** onset before the go cue is used.

ii.
```python
def tone_onset_times(f, start, go):
    """Onset of the instruction tone (sample epoch) for every trial.

    Licking during the sample/delay epoch triggers a replay of the epoch, so a
    trial can contain several sample-epoch onsets; we use the last one before
    the go cue, i.e. the tone the animal actually had to remember.
    """
    samp = _event_times(f, 'sample_start_times')
    idx = np.searchsorted(start, samp, 'right') - 1
    out = np.full(len(start), np.nan)
    for i, s in zip(idx, samp):
        if i < 0 or i >= len(out) or np.isnan(go[i]) or s > go[i]:
            continue
        if np.isnan(out[i]) or s > out[i]:
            out[i] = s
    return out
```

iii. From `methods.txt`, which the AI read in full: "Licking early during the sample/delay epoch triggered a replay of the epoch." The AI checked sample-onset coverage per trial across sessions (step 43) and settled on the last pre-go onset as "the tone the animal actually had to remember". It rewrote an earlier buggy version of this function specifically to get this rule right (step 51).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying input: for each trial, the absolute time of each of the 80 bin centres minus that trial's tone onset. Values are negative before the tone and positive after; stored as `float32` in row 0 of the `(2, 80)` input array. No clipping or normalisation is applied.

ii.
```python
# (1) time (s) since the instruction-tone onset, (2) photostimulation on
dt_tone = (centers - tone[trials][:, None]).astype(np.float32)
...
inputs = np.stack([dt_tone, stim_on], axis=1)  # (ntrials, 2, nbins)
```

iii. Step 78: the AI noticed that on trials with many sample-epoch replays the value can reach ~12 s and considered clipping, then checked the distribution and rejected the idea: "It's the specified input, and clipping would be a deviation; but maybe check distribution: 99th percentile is 4.05 s, so it's fine." Metadata documents the definition, including the replay rule.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same bin grid as the firing rates — the `centers` array is derived from the same `edges` used for spike binning — so bin *k* of the input covers the same interval as bin *k* of the neural data by construction. No interpolation is involved.

ii.
```python
edges = g[:, None] + OFF_START + np.arange(N_BINS + 1)[None, :] * BIN_SIZE
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
...
dt_tone = (centers - tone[trials][:, None]).astype(np.float32)
```

iii. N/A — a direct consequence of sharing one go-cue-anchored bin grid across all streams, confirmed by the sample-trial plots in step 79.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The `acquisition/BehavioralEvents/photostim_start_times` and `photostim_stop_times` event series, i.e. absolute on/off times of the laser. (The reference solution instead uses the trials-table `photostim_onset` / `photostim_duration` columns; I verified on a session that the two sources agree to ~5e-13 s and give identical durations of 0.5 s, so they are interchangeable.) The trials-table `photostim_onset` column is used separately, only to identify control trials for the session-selection criterion.

ii.
```python
def _event_times(f, name):
    grp = f['acquisition/BehavioralEvents']
    if name not in grp:
        return np.array([])
    return np.asarray(grp[name]['timestamps'][:])
```
```python
ps = _event_times(f, 'photostim_start_times')
pe = _event_times(f, 'photostim_stop_times')
```

iii. Step 13/36: the AI inspected the behavioural-event groups and the photostim trials-table fields, and confirmed from the data that "Photostim occurs in the last 0.5 s of delay (go-0.5 to go)" — consistent with `methods.txt` ("We silenced ALM activity during the late delay epoch (last 0.5 s)... photoinhibition always ended before the 'Go' cue"). Using the event series gives the on/off times directly on the session clock without having to parse the string-valued trials columns.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time series per trial, not a per-trial flag. For every photostim event in the session, all bins **overlapping** the `[start, stop)` interval are set to 1; bins on non-stimulated trials stay 0. Stored as `float32` in row 1 of the input array.

ii.
```python
stim_on = np.zeros((len(trials), N_BINS), dtype=np.float32)
ps = _event_times(f, 'photostim_start_times')
pe = _event_times(f, 'photostim_stop_times')
for s_, e_ in zip(ps, pe):
    ov = (edges[:, 1:] > s_) & (edges[:, :-1] < e_)
    stim_on[ov] = 1.0
```
metadata: `'binary: 1 while ALM photostimulation was on (last 0.5 s of the delay epoch on photostimulation trials), 0 otherwise'`.

iii. The instructions ask for "Whether photostimulation is on at every time point (discrete, time-varying)", so the AI represented it as a per-bin binary trace rather than a trial label. Because the loop runs over absolute event times against the full `(n_trials, n_bins+1)` edge array, a stimulation event is marked in *any* trial's window that it overlaps, not just the trial it belongs to. The AI verified the result visually (step 79): "photostim appears in the delay epoch (session-specific onset -1.2 s, 0.5 s duration)".

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim on/off times are session-absolute and are compared directly against the same absolute `edges` array used to bin spikes, so alignment to the go cue is automatic and exact.

ii.
```python
ov = (edges[:, 1:] > s_) & (edges[:, :-1] < e_)
stim_on[ov] = 1.0
```

iii. N/A — same shared bin grid as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no choice column in the NWB files. Choice is derived from two trials-table columns: `trial_instruction` (`'left'`/`'right'`, the cued side) and `outcome` (`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed port, a miss the opposite port, and an ignore means it did not lick.

ii.
```python
outcome = info['outcome'][trials]
instruction = info['instruction'][trials]
...
# choice: the lick port the animal chose in the response epoch.  A 'hit'
# means it licked the instructed port, a 'miss' the opposite one, and an
# 'ignore' trial means the animal did not lick.
choice = np.full(len(trials), 2, dtype=np.int64)          # 2 = no lick
hit = outcome == 'hit'
miss = outcome == 'miss'
choice[hit & (instruction == 'left')] = 0
choice[hit & (instruction == 'right')] = 1
choice[miss & (instruction == 'left')] = 1
choice[miss & (instruction == 'right')] = 0
```

iii. Step 33: "Choice can be derived from outcome+instruction." The AI inspected the trials-table columns and their descriptions and found no direct record of the lick port, but established that instruction × outcome determines it uniquely for hit and miss trials, with `ignore` mapping to a distinct "no lick" class required by the instructions' three-way choice output (left, right, no lick).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as `0 = left`, `1 = right`, `2 = no lick`, and — being a single value per trial — repeated across all 80 bins so that all four outputs share one `(4, 80)` integer array per trial. `output_values[0] = ['left', 'right', 'no lick']`.

ii.
```python
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ...
]
```
```python
outputs = np.stack([np.repeat(choice[:, None], N_BINS, axis=1),
                    np.repeat(outcome_code[:, None], N_BINS, axis=1),
                    np.repeat(early_code[:, None], N_BINS, axis=1),
                    tongue_code], axis=1)                 # (ntrials, 4, nbins)
```

iii. The instructions specify the three categories and ask that outputs be time-varying "if at all possible"; since choice is a single per-trial decision, the AI broadcast it across bins to keep a uniform `(n_output, n_timepoints)` layout, which the format verifier accepts.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome=_dec(t['outcome'][:]),
...
outcome = info['outcome'][trials]
```

iii. Step 13/33: the AI enumerated the unique values of the `outcome` column and found they map one-to-one onto the three categories the instructions request, so no derivation is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to `0 = ignore`, `1 = miss`, `2 = hit` (the order given in the instructions) and repeated across all 80 bins, in row 1 of the output array.

ii.
```python
outcome_code = np.zeros(len(trials), dtype=np.int64)      # 0 = ignore
outcome_code[miss] = 1
outcome_code[hit] = 2
```
```python
OUTPUT_VALUES = [..., ['ignore', 'miss', 'hit'], ...]
```

iii. The code order follows the instructions' listed order ("ignore, miss, hit"). Per-trial value broadcast across bins, as for the other categorical outputs.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The trials-table `early_lick` column, whose values are the strings `'no early'` and `'early'`.

ii.
```python
early=_dec(t['early_lick'][:]),
...
early = info['early'][trials]
```

iii. Step 43: the AI explicitly checked "unique early_lick values" across sessions before coding the mapping. The flag is stored per trial by the acquisition system, so no derivation from lick times is needed; the licks that set it occur during the sample/delay epoch, i.e. inside the −2.5 s part of the window.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `0 = no`, `1 = yes` by string comparison and repeated across all 80 bins, in row 2 of the output array.

ii.
```python
early_code = (early == 'early').astype(np.int64)
```
```python
OUTPUT_VALUES = [..., ['no', 'yes'], ...]
```

iii. Follows the instructions' ordering ("no, yes"). Note that the comparison is against the positive label `'early'`, so any unexpected string would fall into class 0.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = (tongue_x, tongue_y, DeepLabCut likelihood) with matching `timestamps`. Column 1 (y) is the value; column 2 (likelihood) determines visibility. Only the **side** camera is used. If the series is absent, the whole session's tongue output is set to class 3.

ii.
```python
grp = f['acquisition/BehavioralTimeSeries']
key = 'Camera0_side_TongueTracking'
if key not in grp:
    return np.full((ntrials, nbins), 3, dtype=np.int64)
data = np.asarray(grp[key]['data'][:])
ts = np.asarray(grp[key]['timestamps'][:])
y = data[:, 1]
vis = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
vis &= np.isfinite(y)
```

iii. Steps 14, 20, 24, 33, 47: the AI inspected the tracking series (x, y, likelihood at ~294–300 Hz), read the repo's `align_markers.py` to see how markers are handled, and scanned all 174 sessions confirming "All 174 sessions have tongue tracking". The docstring justifies the side view: "The side-view DeepLabCut tongue marker is used (the paper only analyses the side view)", consistent with `methods.txt` describing DeepLabCut tracking of tongue/jaw/nose from side and bottom views.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps: (1) a frame counts as "tongue visible" only if its DeepLabCut likelihood exceeds 0.9 and y is finite; (2) the value of a time bin is the **mean y over the visible frames falling in that bin**, computed efficiently with cumulative sums over the session and `searchsorted` on the bin edges; (3) bins containing no visible frame are labelled class 3. The class boundaries are the 40th and 60th percentiles of the y values of **all individual visible frames of the session** (not of the bin means, which is what the reference solution uses).

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
TONGUE_LOW_PCTL = 40.0
TONGUE_HIGH_PCTL = 60.0
```
```python
lo, hi = np.percentile(y[vis], [TONGUE_LOW_PCTL, TONGUE_HIGH_PCTL]) if vis.sum() else (0., 0.)

yv = np.where(vis, y, 0.0)
cum_y = np.concatenate([[0.0], np.cumsum(yv)])
cum_n = np.concatenate([[0], np.cumsum(vis.astype(np.int64))])

pos = np.searchsorted(ts, edges.ravel()).reshape(ntrials, nedges)
n = cum_n[pos[:, 1:]] - cum_n[pos[:, :-1]]
s = cum_y[pos[:, 1:]] - cum_y[pos[:, :-1]]
with np.errstate(invalid='ignore', divide='ignore'):
    mean_y = np.where(n > 0, s / np.maximum(n, 1), np.nan)
```

iii. The AI examined the likelihood histogram (steps 33, 47) and found it "strongly bimodal (~10.5 % visible)", concluding in the code comment that "the exact value in between has almost no influence" — it picked 0.9 rather than the reference's 0.5 on that basis. Percentiles are taken over the session's visible frames because the instructions say "percentile of y-position over the session"; the AI read this literally as the distribution of measured y-positions. The trajectory contains no discussion of the alternative (percentiles of bin means).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, per session: `0` if the bin's mean y is below the 40th percentile, `1` if between the 40th and 60th percentiles inclusive, `2` if above the 60th percentile, and `3` if no visible tongue frame fell in the bin. Percentile boundaries are computed per session over that session's visible frames.

ii.
```python
code = np.full((ntrials, nbins), 3, dtype=np.int64)
seen = n > 0
code[seen & (mean_y < lo)] = 0
code[seen & (mean_y >= lo) & (mean_y <= hi)] = 1
code[seen & (mean_y > hi)] = 2
return code
```
```python
OUTPUT_VALUES = [..., ['<40th pctl', '40-60th pctl', '>60th pctl', 'not visible']]
```

iii. The class definitions and the per-session scope follow the instructions verbatim. The fourth class exists because the tongue is retracted in ~90 % of frames, so most bins have no measurement; the AI documented this in `metadata['output_descriptions']`. Empirically the resulting distribution in the saved file is 76 % class 3 and, among visible bins, roughly 48 % / 30 % / 22 % for classes 0/1/2 — skewed relative to the nominal 40/20/40, a consequence of taking percentiles over frames while classifying bin means.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as the spikes, so alignment is by `searchsorted` of the trial's go-cue-anchored `edges` into the camera timestamp array. The same `edges` array used for spike binning is passed into `tongue_y_classes`, guaranteeing bin-for-bin correspondence. Bins falling where no frames exist (e.g. outside the video's coverage) get n = 0 and therefore class 3.

ii.
```python
tongue_code = tongue_y_classes(f, edges)
```
```python
pos = np.searchsorted(ts, edges.ravel()).reshape(ntrials, nedges)
n = cum_n[pos[:, 1:]] - cum_n[pos[:, :-1]]
```

iii. The AI verified the video timestamps are session-absolute and continuous at ~300 Hz (steps 14, 47) and confirmed the alignment visually in step 79: "tongue becomes visible right after the go cue".

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several distinct cases, each handled explicitly:
- **Missing go cue or missing instruction tone for a trial** → `NaN` sentinel, trial dropped.
- **Trials outside the ephys observation intervals** (9 sessions where the recording covers only a prefix of the behavioural session) → dropped rather than emitted as all-zero rates.
- **Trials with zero spikes across all units** (recording ended mid-session) → dropped.
- **Sessions with no QC-good unit** → excluded by `ngood > 0`.
- **Missing behavioural-event series** → `_event_times` returns an empty array instead of raising.
- **Missing tongue-tracking series** → the whole session's tongue output becomes class 3.
- **Non-finite / low-likelihood tongue frames** → excluded from the bin mean; a bin with no valid frame becomes the explicit "not visible" class rather than being imputed.
- **Unsorted spike times** → sorted before `searchsorted`.
- **Any other exception in a worker** → caught, printed as `FAILED`, and the session silently skipped.
- **Truncated observation windows** (spikes are stored only between trial start and stop, so the last ~0.8 s of miss trials contains no spikes) → *not* filtered; kept as zero rates and documented in `metadata['known_caveat']`, with a per-session `frac_window_observed` statistic recorded so the extent is auditable.

ii.
```python
def _event_times(f, name):
    grp = f['acquisition/BehavioralEvents']
    if name not in grp:
        return np.array([])
```
```python
if key not in grp:
    return np.full((ntrials, nbins), 3, dtype=np.int64)
```
```python
if spikes.size and np.any(np.diff(spikes) < 0):
    spikes = np.sort(spikes)
```
```python
try:
    out = process_session(path)
except Exception as exc:  # pragma: no cover - defensive
    print('FAILED %s: %r' % (path, exc), flush=True)
    return None
```
```python
'known_caveat':
    "In this NWB release spike times are only stored inside each trial's "
    'observation interval (trial start to trial stop).  Parts of the -2.5 to 1.5 s '
    'window that fall outside that interval therefore contain no spikes and are '
    'binned as zero rate; this mainly affects the last ~0.8 s of error (miss) '
    'trials, which end when the animal licks the wrong port.  The '
    'frac_window_observed field of session_info quantifies this per session.',
```

iii. Most of this was driven by the format verifier's warnings and by targeted investigation (steps 57–69): the AI traced the "all-zero neural data" warnings to the observation-interval issue, fixed it, re-ran, found a single residual all-zero trial, investigated it ("The last observed trial of that session genuinely has no spikes (recording ended)"), and added the zero-spike drop. Its stated principle is that data that were never recorded should be excluded rather than fabricated as zeros, while a legitimately absent measurement (retracted tongue) becomes an explicit category. The one case it chose to keep and document rather than fix is the within-trial truncation, because filtering it would eliminate nearly all miss trials and destroy the outcome output (step 61).

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is I/O: reading each NWB file (50 GB total), specifically the per-unit ragged `spike_times` reads (one h5py slice per good unit, ~390 per session) and the full `(n_frames, 3)` tongue-tracking array (~680 k × 3). Secondary costs are the `searchsorted` binning (one binary search per bin edge per unit), the per-unit `np.any(np.diff(spikes) < 0)` sortedness check, the `json.loads` call per unit for the electrode location, and at the end writing the 7.2 GB pickle. Each file is opened twice (`session_info` then `process_session`), and each session's result is pickled to `/tmp` and read back. The AI mitigated all of this by running sessions in parallel: the full 105-session conversion completed in well under 30 s wall-clock with 24 workers (against ~247 s serial for the reference implementation), albeit with the NWB files largely in page cache.

ii.
```python
with Pool(args.nproc) as pool:
    files = pool.map(_worker, paths)
```
```python
for k, iu in enumerate(good):
    a = 0 if iu == 0 else sti[iu - 1]
    spikes = np.asarray(u['spike_times'][a:sti[iu]])
```

iii. The AI sized the problem up front (steps 14, 15, 40, 45, 47: "174 sessions, ~500 trials, ~300 good units each — the full dataset would be ~8GB"), checked available cores and RAM, and chose multiprocessing from the start. It noticed the surprising speed and reasoned about it explicitly (step 65): "105 sessions in <30s — suspicious. Probably data was cached from previous scans... 50GB of NWB reading in 30s across 24 procs is ~1.7GB/s from page cache; plausible given 872GB buff/cache."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four remaining Python loops:
- The **per-unit spike-binning loop** — the trial dimension is already vectorised by flattening the edge array, but the per-unit dimension cannot be collapsed because each unit has a different number of spikes (ragged storage). This is inherent.
- The **photostim event loop**, which performs a full `(n_trials, n_bins)` boolean comparison per event (~100+ events per session); this could be done in one vectorised pass with `searchsorted` over sorted event times.
- The **event-to-trial mapping loops** in `go_cue_times` and `tone_onset_times`, which iterate over every event in the session in Python; both could be replaced with `np.maximum.at` / `np.minimum.at` style scatter-reductions.
- The **region-labelling comprehension** (`coarse_region` per unit, with nested substring scans over ~14 × 30 keywords), plus `regions.index(r)` in `main()`, which is a linear scan per unit.

The tongue binning, by contrast, is fully vectorised via cumulative sums (an improvement over a per-trial loop).

ii.
```python
for s_, e_ in zip(ps, pe):
    ov = (edges[:, 1:] > s_) & (edges[:, :-1] < e_)
    stim_on[ov] = 1.0
```
```python
for i, s in zip(idx, samp):
    if i < 0 or i >= len(out) or np.isnan(go[i]) or s > go[i]:
        continue
    if np.isnan(out[i]) or s > out[i]:
        out[i] = s
```
```python
cum_y = np.concatenate([[0.0], np.cumsum(yv)])
cum_n = np.concatenate([[0], np.cumsum(vis.astype(np.int64))])
pos = np.searchsorted(ts, edges.ravel()).reshape(ntrials, nedges)
```

iii. The AI did not discuss vectorisation of these loops in the trajectory; the design appears driven by clarity for the small loops and by parallelism for the overall runtime, which was already far below any budget concern.

## 10-c. What processing does the code repeat multiple times?

i. Three repeats, all deliberate or cheap:
- **Each NWB file is opened twice**: `session_info()` opens it to read the trials table and count good units for the inclusion test, then `process_session()` opens it again and reads `units/classification` a second time. The trials columns read in the first pass are reused, so only the file open and the `classification` read are duplicated.
- **Each session's result is serialised twice**: pickled to `/tmp/converted_sessions/*.pkl` by the worker and unpickled again in `main()` before being written into the final 7.2 GB pickle.
- The bin-edge grid is rebuilt per session (trivial), and `regions.index(r)` re-scans the growing region list once per unit.

Nothing is recomputed across sessions; the conversion is otherwise a single pass.

ii.
```python
info = session_info(path)        # opens the file
used, perf, ncl, ncr = session_is_used(info)
if not used:
    return None
with h5py.File(path, 'r') as f:  # opens it again
```
```python
fn = os.path.join(TMP_DIR, os.path.basename(path) + '.pkl')
with open(fn, 'wb') as fh:
    pickle.dump(out, fh, protocol=4)
...
for fn in files:
    with open(fn, 'rb') as fh:
        s = pickle.load(fh)
```

iii. Both repeats are justified by their purpose: the cheap metadata pass means the ~69 rejected sessions never pay for the heavy spike/video reads, and the temp-pickle round trip keeps multi-gigabyte arrays out of the `multiprocessing` result pipe. Neither was explicitly discussed in the trajectory.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A modest amount:
- `targets` is built by `json.loads` on the electrode `location` string for **every** unit in the session, including the ~75 % that fail QC, and is used only to decide whether a motor-cortex unit should be relabelled ALM. Likewise `anno` is decoded for all units.
- `frac_observed`, `performance`, `ncorrect_left`, `ncorrect_right`, `ntrials_total` are computed and stored in `metadata['session_info']`; they are documentation only and are not consumed by the decoder.
- The per-unit sortedness check `np.any(np.diff(spikes) < 0)` runs on every unit even though spike times are sorted in this release.
- `stop_time` is read for every session but used only for `frac_observed`.
- The temporary per-session pickles in `/tmp/converted_sessions/` are written, read once, and never cleaned up (~7 GB of duplicate data left on disk).

ii.
```python
loc = f['general/extracellular_ephys/electrodes']['location'][:]
uel = np.asarray(u['electrodes'][:])
targets = np.array([json.loads(loc[i].decode()).get('brain_regions', '') for i in uel])
```
```python
frac_observed=float(np.mean(np.clip(
    (np.minimum(stop[trials], g + OFF_END) - np.maximum(start[trials], g + OFF_START))
    / (OFF_END - OFF_START), 0, 1))),
```
```python
if spikes.size and np.any(np.diff(spikes) < 0):
    spikes = np.sort(spikes)
```

iii. The diagnostic fields were added intentionally as an audit trail for the known truncation caveat, and the sortedness guard was added defensively right before the full run (step 64: "I must confirm spike_times per unit are sorted (my binning uses searchsorted)... Add a sortedness guard"). The per-unit JSON parse over all units is an oversight rather than a stated decision — restricting it to `good` units would be a straightforward saving.
