# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session laid out as `/app/data/sub-<subject_id>/<file>.nwb`. The AI finds every session with a single sorted glob over that layout and processes each file independently in a 16-process `multiprocessing.Pool`. Files are opened with raw **`h5py`** rather than `pynwb`, reading the HDF5 groups directly (`intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, `units`, `general/extracellular_ephys/electrodes`). Each worker writes its finished session to a per-session pickle in `/app/tmp_sessions`, and the parent then re-reads those temp files in sorted-filename order and concatenates them into the final dictionary. A separate resource, the Allen CCF structure graph JSON, is downloaded once (cached at `/app/allen_structure_graph.json`) and used to label units by brain region.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
if args.limit:
    files = files[:args.limit]
print('processing %d sessions' % len(files), flush=True)

with Pool(args.nproc) as pool:
    infos = []
    for info in pool.imap_unordered(process_session, files):
        infos.append(info)
```

```python
def process_session(path):
    subject, ses = session_name(path)
    info = {'file': os.path.basename(path), 'subject': subject, 'session': ses}
    with h5py.File(path, 'r') as f:
        trials = f['intervals/trials']
        start = trials['start_time'][:]
        ...
        events = f['acquisition/BehavioralEvents']
        go = events['go_start_times/timestamps'][:]
```

```python
kept = sorted([i for i in infos if 'out_path' in i], key=lambda i: i['file'])
for info in kept:
    with open(info['out_path'], 'rb') as fh:
        sess = pickle.load(fh)
    data['neural'].append(sess['neural'])
```

iii. From the trajectory (steps 10-13, 20): the agent first confirmed `pynwb` and `h5py` were both available, then dumped the raw HDF5 tree and worked from it directly — it chose `h5py` because all its exploratory scans were already written against the raw tree and because raw reads are faster for the ragged `spike_times`/`obs_intervals` buffers. Step 16 established the machine has 128 cores / 1 TB RAM / an L4 GPU, which motivated the 16-24 way `Pool`. Step 71: "Run the full conversion in the background with 24 processes." The temp-file staging was chosen so that the per-session results (≈9 GB total) never have to be held in worker memory or shipped back through the `Pool` IPC pipe.

## 1-b. How are the data split into subjects (mice)?

i. The subject id is parsed from the **filename** (`sub-440956_ses-...nwb` → `'440956'`) rather than read from `nwb.subject.subject_id`. During assembly, `subjects` is built in order of first appearance across the sorted kept-session list, and `subject_idx` is that list index for each session. 28 subjects result.

ii.
```python
def session_name(path):
    base = os.path.basename(path)
    sub = base.split('_')[0].replace('sub-', '')
    ses = base.split('_')[1].replace('ses-', '')
    return sub, ses
```

```python
s = sess['subject']
if s not in subjects:
    subjects.append(s)
subject_idx.append(subjects.index(s))
...
data['subjects'] = subjects
data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
```

iii. The agent noted early (steps 3-5) that the DANDI layout is `sub-<id>/<session>.nwb`, so the directory/filename prefix *is* the canonical subject id (it is derived from `nwb.subject.subject_id`). Parsing it from the path avoids opening the file's `general/subject` group in the scan phase. Because `files` is sorted, first-appearance order is the same as sorted-id order.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or inference is required. Session identity is carried as `(subject, ses)` from the filename and stored per session in `metadata['session_info']`. Session order follows the sorted filename list.

Crucially, the AI **also curates sessions**, applying the data paper's published session-selection criteria (quoted in `/app/methods.txt`) on top of the file list:
- overall behavioural performance **> 65 %**, computed as hit / (hit + miss) over *control* trials (no photostim, no early lick, no auto/free water) that were responded to;
- at least **50 correct lick-left and 50 correct lick-right** control trials;
- CCF annotation / QC classification present (one session has all-NaN `classification`/`anno_name`);
- at least **50 usable trials** remaining after trial curation;
- at least 2 trials with ephys coverage.

This keeps **142 of 174** sessions (73,565 trials, 57,023 neurons, 28 mice).

ii.
```python
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50
MIN_TRIALS_PER_SESSION = 50   # sessions with fewer usable trials are dropped
```

```python
# ---- session selection (data paper criteria, on recorded trials) ----
control = recorded & (~photostim_trial) & (early == 'no early') & (~auto_water) & (~free_water)
responded = control & (outcome != 'ignore')
performance = float(np.mean(outcome[responded] == 'hit')) if responded.sum() else 0.0
n_correct_left = int(np.sum(control & (outcome == 'hit') & (instruction == 'left')))
n_correct_right = int(np.sum(control & (outcome == 'hit') & (instruction == 'right')))
if performance <= MIN_PERFORMANCE:
    info['skip'] = 'performance <= 65%%: %.3f' % performance
    return info
if min(n_correct_left, n_correct_right) < MIN_CORRECT_PER_DIRECTION:
    info['skip'] = 'fewer than 50 correct trials in one direction'
    return info
```

```python
trial_idx = np.where(keep)[0]
if len(trial_idx) < MIN_TRIALS_PER_SESSION:
    info['skip'] = 'only %d usable trials (mostly missing video)' % len(trial_idx)
    return info
```

iii. The agent grepped `/app/methods.txt` and the data paper (steps 32-37, 46) and found the explicit sentence: *"We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each."* Step 37: "145 sessions pass the paper's selection criteria." It treated the instructions' requirement that curation "match the reference papers ... filtering of low-quality neurons, trials, sessions, and mice" as mandating this filter. The `>= 50 usable trials` rule was added later (steps 83-84) after it discovered one session (`sub-484676_ses-20210413T112028`) whose video covers only 1.3 % of trials, leaving 7 usable trials; it justified 50 as "consistent with the data paper's session-level trial-count requirement". Step 39: the single session with all-NaN `classification`/`anno_name` "lacks CCF annotations/classification -> must be excluded".

## 1-d. How are the data split into trials?

i. Trials are taken directly from the NWB trials table (`intervals/trials`), one row per behavioural trial. The AI sanity-checks that the number of `go_start_times` events equals the number of trial rows and skips the session otherwise. The go cue event of trial *i* is `go[i]`, i.e. a positional 1:1 mapping between the event stream and the table rows.

ii.
```python
trials = f['intervals/trials']
start = trials['start_time'][:]
stop = trials['stop_time'][:]
outcome = to_str(trials['outcome'][:])
early = to_str(trials['early_lick'][:])
instruction = to_str(trials['trial_instruction'][:])
auto_water = trials['auto_water'][:].astype(bool)
free_water = trials['free_water'][:].astype(bool)
photostim_trial = to_str(trials['photostim_onset'][:]) != 'N/A'
ntrials_all = len(start)
...
go = events['go_start_times/timestamps'][:]
if len(go) != ntrials_all:
    info['skip'] = 'go cue count does not match trial count'
    return info
```

iii. Step 54/55: the agent ran a diagnostic scan over all 174 sessions confirming "All sessions have consistent event structure" and that exactly one go cue exists per trial row, whereas `sample_start_times` can have several entries per trial because an early lick replays the sample epoch. Hence go cues can be indexed positionally against the trials table, while tone onsets must be looked up by time (see 3-a).

## 1-e. How are trials filtered based on quality controls?

i. Four trial-level filters:

1. **Ephys coverage** (`units/obs_intervals`). In 8 sessions the recording starts after the behaviour, so leading trials contain no spikes at all. For every good unit the observed intervals are mapped back to trial indices and the coverage masks are **intersected** across units. A session with fewer than 2 covered trials is dropped.
2. **auto_water and free_water trials are dropped** — water is not contingent on the animal's choice.
3. **Video coverage**: trials whose `[-2.5, +1.5]` s window contains fewer than 50 % of the expected 300 Hz video frames (< 600 of 1200) are dropped, because the tongue output would be undefined. This removes ≈1,079 trials dataset-wide (≈1.1 %).
4. **Session minimum** of 50 usable trials (see 1-c).

Photostimulation, early-lick, error (`miss`) and no-response (`ignore`) trials are **kept**, contrary to the reference analyses, because the decoder task requires them as inputs/outputs.

ii.
```python
oii = units0['obs_intervals_index'][:]
obs = units0['obs_intervals'][:]
recorded = np.ones(ntrials_all, dtype=bool)
for uid in good0:
    o0 = 0 if uid == 0 else oii[uid - 1]
    ints = obs[o0:oii[uid]]
    idx = np.searchsorted(start, ints[:, 0] + 1e-6) - 1
    idx = idx[(idx >= 0) & (idx < ntrials_all)]
    m = np.zeros(ntrials_all, dtype=bool)
    m[idx] = True
    recorded &= m
if recorded.sum() < 2:
    info['skip'] = 'fewer than 2 trials with ephys coverage'
    return info
```

```python
# ---- trial curation ----
keep = recorded & (~auto_water) & (~free_water)
...
win_lo = go + OFF_START
win_hi = go + OFF_END
nframes = np.searchsorted(vts, win_hi) - np.searchsorted(vts, win_lo)
expected_frames = (OFF_END - OFF_START) * 300.0
keep &= nframes >= VIDEO_COVERAGE_MIN * expected_frames
```

iii. The `obs_intervals` filter was added mid-run: step 65 "There are warnings about trials with all-zero neural data"; step 66-67 "in some NWB files the units' obs_intervals cover only a subset of trials (e.g. 160 of 480) ... I must use obs_intervals to determine which trials each unit was recorded during"; step 69 confirmed "all good units within a session share identical coverage." Step 70 verified "no zero trials remain".

The auto/free-water exclusion mirrors the reference repository, which gates every analysis on `(early_lick_trials == 0) * (auto_water_trials == 0) * (free_water_trials == 0)` (`code/VideoAnalysisUtils/functions_for_r2.py:301`); the agent kept the water terms and dropped the early-lick term because early lick is a required decoder output (step 44: "our decoder task explicitly requires photostim as an input and early lick/outcome (including ignore & miss) as outputs — so we must keep these trials").

The video filter came from step 59's scan ("low video trials 1079, no video 795, sessions with any novid 3") and was justified on the grounds that the tongue output is undefined without frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (a ragged buffer indexed by `units/spike_times_index`), restricted to units passing QC (see 2-c). The go-cue timestamps from `acquisition/BehavioralEvents/go_start_times/timestamps` supply the per-trial window origin. Unit→brain-region labels come from `units/anno_name` plus the electrode table's CCF `x` coordinate and probe target string.

ii.
```python
spike_times = units['spike_times']
sidx = units['spike_times_index'][:]
edges_lo = win_lo[trial_idx]
for n, uid in enumerate(unit_ids):
    s0 = 0 if uid == 0 else sidx[uid - 1]
    sp = spike_times[s0:sidx[uid]]
```

iii. Step 26/47: the agent verified that `spike_times` are in session-absolute seconds on the same clock as the behavioural events, so no further neural representation exists or is needed.

## 2-b. How is the `neural` data processed?

i. Per unit and per trial, spikes falling in `[go-2.5, go+1.5]` are located with `searchsorted`, converted to a 0-79 bin index by integer division of the offset from the window start, histogrammed with `np.bincount`, and finally divided by the 50 ms bin width to give **firing rate in Hz** (`float32`). No smoothing, normalisation, baseline subtraction or firing-rate threshold is applied.

ii.
```python
neural = np.zeros((ntrials, nneurons, NBINS), dtype=np.float32)
spike_times = units['spike_times']
sidx = units['spike_times_index'][:]
edges_lo = win_lo[trial_idx]
for n, uid in enumerate(unit_ids):
    s0 = 0 if uid == 0 else sidx[uid - 1]
    sp = spike_times[s0:sidx[uid]]
    lo = np.searchsorted(sp, edges_lo)
    hi = np.searchsorted(sp, win_hi[trial_idx])
    for t in range(ntrials):
        if hi[t] <= lo[t]:
            continue
        b = ((sp[lo[t]:hi[t]] - edges_lo[t]) / BIN_SIZE).astype(np.int64)
        np.clip(b, 0, NBINS - 1, out=b)
        neural[t, n] = np.bincount(b, minlength=NBINS)[:NBINS]
neural /= BIN_SIZE  # spikes/s
```

iii. Step 7: the agent read the reference preprocessing and noted it uses `sliding_histogram(..., rate=True)` with a 40 ms bandwidth and 3.4 ms stride, i.e. counts divided by bin width. Step 42: "neural bin width 40 ms, stride 3.4 ms in the reference; our task requires 50 ms bins", so it kept the rate convention but adopted the instruction's non-overlapping 50 ms bins. Step 44 noted the method paper's "<2 Hz firing rate" neuron exclusion and the "<10 neurons per session per area" area exclusion but deliberately did not apply them (step 55 scanned the effect); those are analysis-stage filters in the method paper, not data-curation filters.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A unit is kept if **all** of:
- `units/classification == 'good'` — the verdict of the region-specific spike-sorting QC classifier described in the Chen/Liu white paper;
- `units/anno_name` is a non-empty string, i.e. the unit has a CCF annotation;
- its electrode's CCF `x` coordinate is finite and its annotation maps into the Allen ontology (needed to assign hemisphere × coarse region).

`units/unit_quality` is not used. No per-metric thresholds (ISI violation, presence ratio, amplitude, …) and no firing-rate threshold are applied. Result: 57,023 units over the 142 kept sessions (≈402/session, the same per-session density as the reference's 69,453/173).

ii.
```python
units = f['units']
classification = to_str(units['classification'][:])
anno = to_str(units['anno_name'][:])
good = (classification == 'good') & (anno != '')
if good.sum() == 0:
    info['skip'] = 'no QC-passing units with CCF annotation'
    return info
```

```python
for i in unit_ids:
    reg = coarse_region(anno[i], utarget[i])
    x = ux[i]
    if reg is None or not np.isfinite(x):
        keep_unit.append(False)
        regions.append(None)
        continue
    side = 'left' if x >= ML_MIDLINE else 'right'
    keep_unit.append(True)
    regions.append('%s %s' % (side, reg))
```

iii. Step 23: "All good units have annotations; total 69,453 good units matches paper's ~69,943 for 173 sessions." Step 53: "Ontology mapping reproduces the paper's per-region unit counts exactly (thalamus 12808, striatum 7664, midbrain 7495, medulla 2928), confirming 'classification==good' + anno_name is the right unit selection." This exact-count reproduction is the agent's strongest evidence for the unit-curation choice. Step 13/14 established that `classification` is the QC-classifier label (values `'good'`/`'unlabelled'`) as opposed to the older `unit_quality`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams share one session-absolute clock, so no resampling or offset correction is needed. For each kept trial the window is `win_lo = go + (-2.5)` to `win_hi = go + 1.5`; spikes are binned by their offset from `win_lo`. The go cue is taken positionally from `go_start_times` (one event per trial row).

ii.
```python
OFF_START = -2.5          # s, relative to go cue
OFF_END = 1.5             # s, relative to go cue
...
go = events['go_start_times/timestamps'][:]
...
win_lo = go + OFF_START
win_hi = go + OFF_END
...
lo = np.searchsorted(sp, edges_lo)
hi = np.searchsorted(sp, win_hi[trial_idx])
b = ((sp[lo[t]:hi[t]] - edges_lo[t]) / BIN_SIZE).astype(np.int64)
```

iii. Step 26/47: the agent explicitly verified the timing conventions — spike times, behavioural events and video timestamps are all in session time, while `trials['photostim_onset']` is relative to trial start (handled separately). An important caveat it discovered and documented (steps 49-50, 54): spikes are only *stored* inside each trial's `obs_interval` `[start_time, stop_time]`, and error (`miss`) trials typically end ~0.7-0.8 s after the go cue, so the last bins of those trials are genuinely zero. It chose to keep these trials and record the caveat in `metadata['notes']`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **50 ms**, giving exactly **80 non-overlapping bins** spanning -2.5 s to +1.5 s around the go cue, identical for every trial and session. Spikes are binned once at that resolution directly from spike times; there is **no** rebinning of an intermediate representation and no overlapping/sliding window. `metadata['time_bin_size'] = 50.0` (ms), `off_start = -2.5`, `off_end = 1.5`. The behavioural video (300 Hz) and the photostim events are binned onto the same grid.

ii.
```python
OFF_START = -2.5          # s, relative to go cue
OFF_END = 1.5             # s, relative to go cue
BIN_SIZE = 0.05           # s
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

```python
'time_bin_size': BIN_SIZE * 1000.0,
'temporal_alignment_event': 'auditory go cue onset (end of the delay epoch)',
'off_start': OFF_START,
'off_end': OFF_END,
```

iii. Step 42: "neural bin width 40 ms, stride 3.4 ms in the reference; our task requires 50 ms bins" — the agent recognised the instruction overrides the reference's sliding-window convention, and applied the 50 ms non-overlapping grid everywhere so all streams share bin boundaries.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (sample-epoch/tone onsets, session clock) and the trial's go-cue time. Because an early lick replays the sample epoch, a trial can have several sample onsets; the AI takes the **last sample onset strictly before the go cue**.

ii.
```python
sample_on = events['sample_start_times/timestamps'][:]
...
# tone (sample epoch) onset: last sample-epoch start before the go cue; with
# early licks the sample epoch is replayed, so the last one is the relevant one
tone_idx = np.searchsorted(sample_on, go) - 1
tone_rel_go = sample_on[np.maximum(tone_idx, 0)] - go   # negative (before go cue)
```

iii. Step 46/47: the agent explicitly inspected "tone (sample) onset times per trial including replays." Step 55: "tone onset (last sample start before go) available for every trial (median 1.85 s before go)" — i.e. it verified across all 174 sessions that the lookup never falls off the front of the array, and that the resulting tone-to-go interval is physiologically sensible (0.65 s sample + 1.2 s delay ≈ 1.85 s).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. It is a continuous, time-varying scalar: for each bin, (bin centre relative to go cue) − (tone time relative to go cue) = seconds elapsed since tone onset. Values are negative before the tone and positive after; typical bin-0 value is −1.525 s (i.e. the window starts ~1 s before the tone), with a long tail up to ~8 s on trials with many sample replays. Stored as `float32` in row 0 of the `(2, 80)` input array. No clipping, normalisation or binarisation is applied.

ii.
```python
bin_centers = OFF_START + (np.arange(NBINS) + 0.5) * BIN_SIZE   # rel. to go cue
tone_idx = np.searchsorted(sample_on, go) - 1
tone_rel_go = sample_on[np.maximum(tone_idx, 0)] - go   # negative (before go cue)

inputs = np.zeros((ntrials, 2, NBINS), dtype=np.float32)
for k, t in enumerate(trial_idx):
    inputs[k, 0] = bin_centers - tone_rel_go[t]
```

iii. Step 75: the agent checked the verifier summary and noted "input time from tone onset ranges to 11.9 s (long sample epochs in some trials — fine)", i.e. it inspected the distribution and accepted the long tail as a genuine consequence of sample-epoch replay after early licks rather than a bug. The instructions ask for a continuous, time-varying input, so no binarisation was applied (the "represent it as a binary time series" clause was read as applying to event-onset inputs, not to this explicitly-continuous one).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the **centres of the same 80 bins** used for the firing rates, on the same go-cue-relative grid, so bin *k* of the input covers exactly the same interval as bin *k* of the neural matrix. No interpolation or resampling is involved — the quantity is analytic given the tone time.

ii.
```python
bin_centers = OFF_START + (np.arange(NBINS) + 0.5) * BIN_SIZE   # rel. to go cue
...
inputs[k, 0] = bin_centers - tone_rel_go[t]
```
compared with the neural grid:
```python
win_lo = go + OFF_START
b = ((sp[lo[t]:hi[t]] - edges_lo[t]) / BIN_SIZE).astype(np.int64)
```

iii. Implicit in the design: every stream is placed on the single go-cue-anchored grid defined by `OFF_START`/`BIN_SIZE`/`NBINS`. The agent verified per-trial alignment by spot-checking a photostim trial's input timing against the methods description (step 95-96: photostim "correctly occupies -1.25 → -0.7 s (the last 0.5 s of the delay epoch, as described in the methods)"), which also confirms the bin-centre convention is consistent.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The **event streams** `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `photostim_stop_times/timestamps` (session clock). The trials-table column `photostim_onset` is read separately, but only as an `'N/A'`-vs-not flag used to define *control* trials for the session-performance criterion — it is not used to build the input.

ii.
```python
stim_on = events['photostim_start_times/timestamps'][:]
stim_off = events['photostim_stop_times/timestamps'][:]
...
photostim_trial = to_str(trials['photostim_onset'][:]) != 'N/A'   # control-trial flag only
```

```python
stim_trial = np.searchsorted(start, stim_on) - 1
```

iii. Step 26/28: the agent inspected "photostim onset/duration/power values" in the trials table and found the onsets are strings relative to trial start. Step 54-55: it then ran a cross-check over all sessions of "consistency between photostim events and the trials table" and concluded "photostim events match trials table exactly". Having verified equivalence, it used the event timestamps because they are already on the session clock and require no trial-start arithmetic or string parsing.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A **binary time series**, not a per-trial flag. Each stim event is attributed to a trial by `searchsorted` on trial start times; if that trial survived curation, the bins spanned by `[on, off]` are set to 1.0. The bin range uses `floor` on the onset and `ceil` on the offset, so any bin **overlapping** the stim interval is marked (slightly more inclusive than a bin-centre test). Everything else is 0. Stored as `float32` in row 1. Result: 2.79 % of all bins are on, consistent with ~25 % of trials × 0.5 s of a 4 s window.

ii.
```python
# photostimulation: 1 in bins overlapping a photostim interval
stim_trial = np.searchsorted(start, stim_on) - 1
for on, off, tr in zip(stim_on, stim_off, stim_trial):
    if tr < 0 or tr >= ntrials_all or not keep[tr]:
        continue
    k = int(np.searchsorted(trial_idx, tr))
    b0 = int(np.floor((on - win_lo[tr]) / BIN_SIZE))
    b1 = int(np.ceil((off - win_lo[tr]) / BIN_SIZE))
    b0 = max(b0, 0)
    b1 = min(b1, NBINS)
    if b1 > b0:
        inputs[k, 1, b0:b1] = 1.0
```

iii. The instructions require "whether photostimulation is on at every time point (discrete, time-varying)", so a per-bin binary series is mandatory. Step 95-96: the agent spot-checked a photostim trial and confirmed the marked bins fall at −1.25 → −0.7 s, i.e. the last 0.5 s of the 1.2 s delay epoch, which is exactly what the data paper describes for bilateral/unilateral ALM photoinhibition. Iterating over events rather than trials also handles trials with multiple stim epochs without special-casing.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Bin indices are computed as offsets from `win_lo[tr] = go[tr] - 2.5`, the identical window origin used for the spike binning, then clipped to `[0, 80)`. Stim epochs falling entirely outside the window produce no marked bins.

ii.
```python
b0 = int(np.floor((on - win_lo[tr]) / BIN_SIZE))
b1 = int(np.ceil((off - win_lo[tr]) / BIN_SIZE))
b0 = max(b0, 0)
b1 = min(b1, NBINS)
```

iii. Same rationale as 3-c: one shared go-cue-anchored grid for all streams. Using the session-clock event timestamps (4-a) rather than the trial-relative strings means no trial-start offset has to be added, removing a source of alignment error.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not stored in the file. It is derived from two trials-table columns, `outcome` (`'hit'`/`'miss'`/`'ignore'`) and `trial_instruction` (`'left'`/`'right'`): a hit means the animal licked the instructed side, a `miss` (error) means it licked the opposite side, and an `ignore` means it did not lick.

ii.
```python
outcome = to_str(trials['outcome'][:])
instruction = to_str(trials['trial_instruction'][:])
```

```python
# choice: hit -> instructed side, miss (error) -> opposite side, ignore -> no lick
choice_map = {'left': 0, 'right': 1}
for k, t in enumerate(trial_idx):
    if outcome[t] == 'hit':
        ch = choice_map[instruction[t]]
    elif outcome[t] == 'miss':
        ch = choice_map['right' if instruction[t] == 'left' else 'left']
    else:
        ch = 2
```

iii. The agent did not simply assume this mapping — at step 59 it ran a dataset-wide validation against the raw lick event streams (`left_lick_times` / `right_lick_times`), deriving choice from the first post-go lick and comparing it with the instruction × outcome derivation. Result: median agreement **99.8 %** (5th percentile 98.7 %, worst session 85.1 %). Having confirmed the derivation, it used the categorical columns rather than the lick times because they are the curated, canonical labels.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick`, written into row 0 of the `(4, 80)` per-trial output array and **repeated across all 80 bins** (choice is a per-trial quantity, but the format prefers time-varying outputs and all four outputs must share one array). `output_values[0] = ['left', 'right', 'no lick']`. Resulting distribution: 44.8 % left / 44.2 % right / 11.0 % no lick.

ii.
```python
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ...
]
```

```python
outputs = np.zeros((ntrials, 4, NBINS), dtype=np.int64)
...
    outputs[k, 0, :] = ch
```

iii. The instructions specify "Lick direction choice (left, right, no lick, per-trial)", so the three-way coding and the left→0 / right→1 order follow directly. Step 62: "Outputs: choice, outcome, early lick, tongue-y class." Broadcasting per-trial scalars across bins keeps a uniform `(n_output, n_timepoints)` shape, which the verifier requires.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which already holds exactly the three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome = to_str(trials['outcome'][:])
```

iii. Step 54: the agent ran a scan for "unique early_lick/outcome values" across all sessions and confirmed the vocabulary is closed and matches the three categories the instructions ask for, so no derivation is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped through a fixed dictionary to `0 = ignore`, `1 = miss`, `2 = hit` and written into row 1, repeated across all 80 bins. `output_values[1] = ['ignore', 'miss', 'hit']`. Resulting distribution: 11.0 % ignore / 15.1 % miss / 73.9 % hit.

ii.
```python
outputs[k, 1, :] = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome[t]]
```

iii. The code order follows the instructions' "(ignore, miss, hit)" listing. Note that the reference papers exclude `miss` and `ignore` trials from their analyses; step 44 records the agent's decision to keep them: "our decoder task explicitly requires ... early lick/outcome (including ignore & miss) as outputs — so we must keep these trials." Using a strict dict means an unexpected label would raise rather than be silently miscoded.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The `early_lick` column of the trials table, whose values are `'early'` and `'no early'`.

ii.
```python
early = to_str(trials['early_lick'][:])
```

iii. Step 52/54: the agent explicitly scanned for the "early_lick value set" across all sessions before relying on the string comparison. The flag is set by a lick during the sample or delay epoch, i.e. before the go cue, so the causing event lies inside the −2.5 s part of the analysis window even though the label is stored per trial.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` → 1, anything else → 0, written into row 2 and repeated across all 80 bins. `output_values[2] = ['no', 'yes']`. Resulting distribution: 88.5 % no / 11.5 % yes. Note these trials are excluded by every reference analysis but retained here.

ii.
```python
outputs[k, 2, :] = 1 if early[t] == 'early' else 0
```

iii. Instructions specify "Early lick (no, yes, per-trial)", giving the 0/1 order. Step 44/62 record the decision to retain early-lick trials despite the reference code's `early_lick_trials == 0` gate, because early lick is a required decoder output — excluding them would leave the class empty.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, a DeepLabCut side-camera series sampled at ~300 Hz whose `data` is `(n_frames, 3)` = `tongue_x`, `tongue_y`, `tongue_likelihood`, with matching session-clock `timestamps`. Column 1 is the y-position; column 2 (likelihood) gates visibility.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
vts = tongue['timestamps'][:]
vdata = tongue['data'][:]
tongue_y = vdata[:, 1]
visible = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. Steps 13/21/23: the agent inspected the tracking series' description and confirmed the `(x, y, likelihood)` channel layout and the 300 Hz rate, and verified via a scan that **all 174 sessions** have side-camera tongue tracking. Step 25: it checked the timestamp structure and found they are "session-time, continuous at 300 Hz within trials with gaps between trials" — i.e. video is trial-gated, which later motivated the coverage filter.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps:
1. A frame counts as "tongue visible" if `likelihood > 0.9`; the tracker emits a position even when the tongue is retracted, so low-confidence frames must be excluded.
2. Per-session class edges: the **40th and 60th percentiles of `tongue_y` over all visible frames in the session** (computed across the whole session including inter-trial intervals, not just the analysis windows).
3. Per trial and per 50 ms bin, the **median** y over the visible frames in that bin is compared against the two edges.

ii.
```python
LIKELIHOOD_THRESH = 0.9   # DeepLabCut confidence for 'tongue visible'
...
tongue_y = vdata[:, 1]
visible = vdata[:, 2] > LIKELIHOOD_THRESH
# per-session discretization thresholds, over frames where the tongue is visible
if visible.sum() > 0:
    p40, p60 = np.percentile(tongue_y[visible], [40, 60])
else:
    p40 = p60 = np.nan
```

```python
for bi in range(NBINS):
    m = (b == bi) & vv
    if not np.any(m):
        continue
    y = float(np.median(yy[m]))
    cls[bi] = 0 if y < p40 else (1 if y <= p60 else 2)
```

iii. Steps 25/26: the agent measured the likelihood distribution and found it "strongly bimodal", with the tongue visible in only ~10.5 % of frames; it therefore picked a high threshold (0.9) knowing the exact value is immaterial given the bimodality. Step 42: it read the method paper's treatment of tongue occlusion and concluded a separate "not visible" category is required. Step 60 re-checked the likelihood histogram shape (fractions below 0.01, 0.01-0.5, 0.5-0.9, ≥0.9) before fixing the threshold. The median (rather than mean) within a bin was chosen for robustness to tracking outliers.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes per bin, exactly as the instructions specify plus an occlusion class:
- `0` = y < 40th percentile,
- `1` = 40th ≤ y ≤ 60th percentile,
- `2` = y > 60th percentile,
- `3` = "not visible" (no frame in the bin with likelihood > 0.9, or no video frames at all in the bin).

Percentiles are per session. Realised distribution over all bins: 12.4 % / 5.4 % / 8.3 % / 73.9 % — i.e. among *visible* bins the split is ≈47 / 21 / 32 %, skewed relative to the nominal 40 / 20 / 40 because the percentiles are taken over raw frames while the discretised quantity is the per-bin median.

ii.
```python
OUTPUT_VALUES = [
    ...
    ['low (<40th pctile)', 'middle (40-60th pctile)', 'high (>60th pctile)', 'not visible'],
]
```

```python
cls = np.full(NBINS, 3, dtype=np.int64)
for bi in range(NBINS):
    m = (b == bi) & vv
    if not np.any(m):
        continue
    y = float(np.median(yy[m]))
    cls[bi] = 0 if y < p40 else (1 if y <= p60 else 2)
outputs[k, 3, :] = cls
```

```python
lo = np.searchsorted(vts, win_lo[t])
hi = np.searchsorted(vts, win_hi[t])
if hi <= lo:
    outputs[k, 3, :] = 3
    continue
```

iii. The 40th/60th split and the per-session scope are dictated verbatim by the instructions ("per-session discretization … 3: not visible"). The agent's reading of "percentile of y-position over the session" is the literal one — percentiles of the y-position samples. Step 42: "Tongue occlusion handled specially (we must set 'not visible' category = 3)".

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and events. For each trial the frame range is found with `searchsorted(vts, go-2.5)` / `searchsorted(vts, go+1.5)`, and each frame's bin index is `floor((t - win_lo)/0.05)` clipped to `[0, 80)` — the identical grid used for the firing rates, so bin *k* of the tongue output covers exactly the same 50 ms as bin *k* of the neural matrix. Trials whose window contains no frames at all get all-`3`; trials whose window contains fewer than half the expected frames were already dropped (1-e).

ii.
```python
for k, t in enumerate(trial_idx):
    lo = np.searchsorted(vts, win_lo[t])
    hi = np.searchsorted(vts, win_hi[t])
    if hi <= lo:
        outputs[k, 3, :] = 3
        continue
    tt = vts[lo:hi]
    yy = tongue_y[lo:hi]
    vv = visible[lo:hi]
    b = ((tt - win_lo[t]) / BIN_SIZE).astype(np.int64)
    np.clip(b, 0, NBINS - 1, out=b)
```

iii. Step 25: the agent verified the tracking timestamps are session time (not per-trial resets) — "Tracking timestamps are session-time, continuous at 300 Hz within trials with gaps between trials" — which is what makes the direct `searchsorted` alignment valid. Step 95: it spot-checked "tongue output timing" on a real trial as a final sanity check. Because the video is trial-gated, it also quantified per-trial coverage (step 59) and added the ≥50 % coverage filter rather than silently emitting long runs of class 3.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six distinct cases, each handled explicitly:

1. **Non-string / NaN text columns** (one session has all-NaN `classification` and `anno_name`): `to_str` converts any float to `''`; the session then has zero QC-passing units and is skipped with a recorded reason.
2. **Event/table count mismatch**: if `len(go) != n_trials` the session is skipped.
3. **Trials with no ephys coverage**: dropped via `obs_intervals`; sessions with <2 such trials skipped.
4. **Units with missing CCF coordinates or unmappable annotations**: dropped (`reg is None or not np.isfinite(x)`).
5. **Missing video**: trials with <50 % frame coverage dropped; trials with zero frames get all-`3`; individual bins with no confident detection get class `3`; a session with no visible tongue frames at all is skipped.
6. **Trial with no preceding tone onset**: `np.maximum(tone_idx, 0)` prevents a negative-index wrap (verified never to trigger, step 55).

Every skip records a human-readable reason in `info['skip']`, which is printed and (for kept sessions) stored in `metadata['session_info']`.

ii.
```python
def to_str(arr):
    """Decode an hdf5 string column; NaN (missing) becomes ''."""
    return np.array([x.decode() if isinstance(x, bytes) else ('' if isinstance(x, float) else str(x))
                     for x in arr])
```

```python
if len(go) != ntrials_all:
    info['skip'] = 'go cue count does not match trial count'
    return info
...
if len(good0) == 0:
    info['skip'] = 'no QC-passing units with CCF annotation'
    return info
...
if not np.isfinite(p40):
    info['skip'] = 'no visible tongue frames in session'
    return info
```

```python
tone_idx = np.searchsorted(sample_on, go) - 1
tone_rel_go = sample_on[np.maximum(tone_idx, 0)] - go
```

iii. Step 30: "Some sessions have non-string classification (likely NaN floats). Need robust decoding. Plan: Rewrite with a robust to-str helper." Step 39: "One session (sub-440958_ses-20190216T162508) lacks CCF annotations/classification (all NaN) -> must be excluded." The general principle the agent applied: where a measurement was never recorded, exclude the unit/trial/session rather than emit fabricated zeros (steps 65-69 on the all-zero-trial problem); where the measurement legitimately has no value (retracted tongue), represent it as an explicit category rather than impute.

## 10-a. What are the most time-consuming steps of the code?

i. In rough order:
1. **Reading the NWB/HDF5 arrays** — the ragged `spike_times` buffer (up to ~11.5 M doubles per session) and the full `(n_frames, 3)` tongue array (~680 k × 3). This dominates per-session wall clock.
2. **The nested neural-binning loop** — `n_units × n_trials` ≈ 400 × 520 ≈ 208,000 Python iterations per session, each doing a slice, a divide, a clip and a `bincount`.
3. **The nested tongue loop** — `n_trials × 80` ≈ 41,000 Python iterations per session, each building a boolean mask over that trial's ~1,200 frames and taking a median; this is an O(bins × frames) scan where O(frames) would do.
4. **The temp-file round trip** — every session is pickled to `/app/tmp_sessions` and immediately re-read by the parent, so the ~9.2 GB payload is serialised and deserialised one extra time.
5. **Final pickling** of the 9.2 GB dictionary.

Wall clock is largely rescued by the 16-24 way `multiprocessing.Pool`: the agent measured 1.9 s for 4 sessions in the smoke test (step 64) and the full 174-session run completed in minutes.

ii.
```python
with Pool(args.nproc) as pool:
    infos = []
    for info in pool.imap_unordered(process_session, files):
```
```python
    for t in range(ntrials):
        if hi[t] <= lo[t]:
            continue
        b = ((sp[lo[t]:hi[t]] - edges_lo[t]) / BIN_SIZE).astype(np.int64)
        np.clip(b, 0, NBINS - 1, out=b)
        neural[t, n] = np.bincount(b, minlength=NBINS)[:NBINS]
```
```python
out_path = os.path.join(TMP_DIR, '%s_%s.pkl' % (subject, ses))
with open(out_path, 'wb') as fh:
    pickle.dump(result, fh, protocol=4)
```

iii. Step 16: the agent profiled the machine ("128 cores, 1TB RAM, L4 GPU 23GB") before designing the pipeline, and chose process-level parallelism over micro-optimising the inner loops. Step 27 estimated the output at ~12 GB and step 71 launched the full run with 24 processes. Step 93 records cleaning up the 9 GB temp directory afterwards. The design trades per-session efficiency for simplicity, relying on the core count to absorb it.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Five:
1. **Per-trial spike binning** (`for t in range(ntrials)`) — avoidable entirely by building one flat array of `n_trials × 81` absolute bin edges, calling `searchsorted` once per unit and differencing, which is what the reference does. This is the largest single win.
2. **Per-bin tongue loop** (`for bi in range(NBINS)`) — each iteration rescans the whole trial's frame array with `(b == bi) & vv`, giving O(80 × n_frames); a single `np.bincount`/`np.add.at` pass (or `np.lexsort` for a true per-bin median) is O(n_frames).
3. **The `obs_intervals` loop over every good unit** — ~400 iterations each allocating a length-`n_trials` boolean array, when the agent had already verified all good units in a session share identical coverage, so one unit suffices.
4. **The three `for k, t in enumerate(trial_idx)` loops** for inputs and per-trial outputs — these assign broadcastable scalars and could be single vectorised assignments (`outputs[:, 1, :] = codes[trial_idx][:, None]`).
5. **The per-unit `regions` loop** with a dict lookup and a `'%s %s'` format per unit, plus `brain_regions.index(r)` in the assembly stage, which is a linear scan of the region list per unit (O(n_units × n_regions)) instead of a dict lookup.

ii.
```python
for uid in good0:
    o0 = 0 if uid == 0 else oii[uid - 1]
    ints = obs[o0:oii[uid]]
    idx = np.searchsorted(start, ints[:, 0] + 1e-6) - 1
    idx = idx[(idx >= 0) & (idx < ntrials_all)]
    m = np.zeros(ntrials_all, dtype=bool)
    m[idx] = True
    recorded &= m
```
```python
    for bi in range(NBINS):
        m = (b == bi) & vv
        if not np.any(m):
            continue
        y = float(np.median(yy[m]))
        cls[bi] = 0 if y < p40 else (1 if y <= p60 else 2)
```
```python
data['brain_region_idx'].append(
    np.array([brain_regions.index(r) for r in sess['regions']], dtype=np.int64))
```

iii. The agent never discusses vectorising these; its stated performance strategy is process-level parallelism (step 16, step 71: "24 processes"), and after the smoke test measured 1.9 s for 4 sessions (step 64) it had no reason to optimise further. The loops are correct, just not tight — an acceptable trade given a 128-core box, but they are the obvious targets if the code had to run single-threaded.

## 10-c. What processing does the code repeat multiple times?

i. Three genuine repetitions:
1. **Unit QC is computed twice.** `units0['classification']` and `units0['anno_name']` are read, decoded through `to_str` and thresholded into `good0` for the `obs_intervals` step; then a few lines later `units = f['units']` is re-fetched and the *same* two columns are re-read, re-decoded and re-thresholded into `good`. `good0` and `good` are identical by construction.
2. **The video `searchsorted` is computed twice per trial.** `nframes = np.searchsorted(vts, win_hi) - np.searchsorted(vts, win_lo)` is computed vectorially for the coverage filter, and then `lo`/`hi` are recomputed with the same calls inside the per-trial tongue loop.
3. **The result payload is serialised twice.** Each session is pickled to `/app/tmp_sessions` and unpickled by the parent, then pickled again into the final `converted_data.pkl`.

Minor: `load_ontology()` runs at module import in each process (on Linux `fork` this is inherited from the parent, so it is effectively once), and the conversion was run end-to-end **twice** (step 71 and again at step 84 after adding `MIN_TRIALS_PER_SESSION`).

ii.
```python
        units0 = f['units']
        cls0 = to_str(units0['classification'][:])
        anno0 = to_str(units0['anno_name'][:])
        good0 = np.where((cls0 == 'good') & (anno0 != ''))[0]
```
```python
        # ---- unit curation ----
        units = f['units']
        classification = to_str(units['classification'][:])
        anno = to_str(units['anno_name'][:])
        good = (classification == 'good') & (anno != '')
```
```python
        nframes = np.searchsorted(vts, win_hi) - np.searchsorted(vts, win_lo)
...
            lo = np.searchsorted(vts, win_lo[t])
            hi = np.searchsorted(vts, win_hi[t])
```

iii. The duplication is an artefact of how the script grew: the `obs_intervals` / ephys-coverage block was inserted *before* the pre-existing unit-curation block during the step-69 patch ("Update convert_data.py: compute ephys coverage from obs_intervals ... apply it as a trial filter before session-selection statistics"), and the agent added the columns it needed at the new location rather than hoisting the existing computation. `to_str` is a Python-level list comprehension over every unit, so this is not free, but at ~2,000 units per session it is small next to the I/O. The temp-file round trip is a deliberate memory-management choice (10-a), not an oversight.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, none of which affect correctness:
1. **`stop = trials['stop_time'][:]` is read and never used** — dead code left over from the exploratory scans, which used `stop_time` to bound lick searches.
2. **~9.2 GB of temp pickles** are written to `/app/tmp_sessions` and read back, then deleted by hand (step 93). Pure I/O overhead relative to returning results directly or streaming into the final file.
3. **Session-selection statistics are computed for sessions that are then skipped for other reasons**, and `n_photostim_trials` / `n_early_trials` are computed purely for the log line.
4. **`coarse_region` and the ontology path lookup run for every QC-passing unit**, including units subsequently dropped for non-finite CCF coordinates.
5. **The full electrode `location` JSON is parsed for every electrode** (`json.loads` per row, hundreds per probe) when only the per-unit first-electrode entries are used.
6. **Outputs are stored as `int64`** — an 8× memory/serialisation cost for values in `{0,1,2,3}`; `int8` (as the reference uses) would cut the output payload by 87 %.
7. **`brain_regions` is built in first-appearance order and then re-sorted with a remap pass**, a second sweep over every unit index that a sort-at-the-end-only design would avoid.

Also, the train/verify cycle was run twice on two different full conversions, doubling the total compute, though that was a deliberate response to a QC finding rather than wasted work in the script itself.

ii.
```python
        start = trials['start_time'][:]
        stop = trials['stop_time'][:]     # never used again
```
```python
        target = np.array([json.loads(loc)['brain_regions'] for loc in to_str(electrodes['location'][:])])
```
```python
        outputs = np.zeros((ntrials, 4, NBINS), dtype=np.int64)
```
```python
    brain_regions_sorted = sorted(brain_regions)
    remap = np.array([brain_regions_sorted.index(r) for r in brain_regions], dtype=np.int64)
    data['brain_region_idx'] = [remap[idx] for idx in data['brain_region_idx']]
```

iii. Nothing in the trajectory addresses these; they are residue from the exploratory phase (the agent wrote ~15 throwaway scan scripts, steps 20-60, and the converter reuses their idioms). The `int64` outputs are the most consequential: they inflate the pickle and the decoder's host-memory footprint without adding information. The agent's attention at the end (steps 93-94) went to documentation, temp-file cleanup and making the ontology download self-healing rather than to trimming these.
