# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats the dataset as one NWB file per session under `/app/data/sub-*/`. It finds all session files with a glob, then reads each file twice with `h5py`: once in `session_info()` to collect lightweight trial/session metadata for inclusion filtering, and again in `process_session()` to load events, units, spike times, and video.

ii. 
```python
paths = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
def session_info(path):
    with h5py.File(path, 'r') as f:
        t = f['intervals/trials']
        info = dict(
            path=path,
            subject=f['general/subject/subject_id'][()].decode(),
            session_id=f['identifier'][()].decode(),
            ...
        )
```

```python
def process_session(path):
    info = session_info(path)
    ...
    with h5py.File(path, 'r') as f:
        go = go_cue_times(f, start, stop)
        tone = tone_onset_times(f, start, go)
```

iii. In the trajectory, the AI explicitly identified the data as "NWB files per session" and decided to "write and run a scan script over all 174 NWB files" before implementing the converter. It justified the lightweight first pass as a way to reproduce the paper's session-selection criteria before loading full session contents.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the NWB subject field `general/subject/subject_id`. Each processed session stores that subject id, and the final `subjects` list and `subject_idx` array are built from the per-session values in first-seen order.

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

iii. In the trajectory, the AI repeatedly summarized the dataset as "174 sessions, 28 mice" and treated the NWB `subject_id` field as the canonical subject identifier. There was no attempt to infer subject identity from filenames or paper mouse names.

## 1-c. How are the data split into sessions?

i. The AI uses one NWB file as one session. Session identity is taken from the NWB `identifier` field, and session order follows the sorted file-path list. Only sessions passing later inclusion criteria are emitted.

ii.
```python
paths = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
session_id=f['identifier'][()].decode(),
```

```python
sessions_info.append(dict(session_id=s['session_id'], subject=s['subject'], ...))
```

iii. In the trajectory, the AI stated early that the "data are NWB files per session" and then used that assumption consistently in its dataset-wide scans and final converter.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. Go-cue events are then mapped onto those trials by assigning each `go_start_times` timestamp to the most recent trial start, and tone onsets are mapped similarly. Trials without a mapped go cue or tone onset are later dropped.

ii.
```python
start=np.asarray(t['start_time'][:]),
stop=np.asarray(t['stop_time'][:]),
```

```python
def go_cue_times(f, start, stop):
    go = _event_times(f, 'go_start_times')
    idx = np.searchsorted(start, go, 'right') - 1
    out = np.full(len(start), np.nan)
    for i, g in zip(idx, go):
        if 0 <= i < len(out) and np.isnan(out[i]):
            out[i] = g
    return out
```

iii. In the trajectory, the AI said it had verified that the behavioral events are session-absolute timestamps and that it needed to map them back onto trial rows. It justified this because it was working directly with HDF5 groups rather than `pynwb` dataframes.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials that have a mapped go cue, a mapped tone onset, are not `free_water`, are not `auto_water`, and fall inside the units' shared `obs_intervals`. After spike binning, it also drops trials whose neural array is entirely zero. Separately, it only processes sessions that pass the data paper's behavioral inclusion criteria.

ii.
```python
keep = (~np.isnan(go)) & (~np.isnan(tone)) & (~info['free_water']) & (~info['auto_water'])
keep &= observed_trials(f, start)
trials = np.where(keep)[0]
if len(trials) < 2:
    return None
```

```python
nonempty = rates.sum(axis=(0, 2)) > 0
if not np.all(nonempty):
    trials = trials[nonempty]
    rates = rates[:, nonempty, :]
    ...
```

```python
used = (perf > MIN_PERFORMANCE and ncl >= MIN_CORRECT_PER_DIRECTION
        and ncr >= MIN_CORRECT_PER_DIRECTION and info['ngood'] > 0)
```

iii. In the trajectory, the AI said it wanted to match the data paper's 106-session analysis set, keep all decoder-relevant behavior types, exclude "free-water / auto-water trials," and remove trials outside `obs_intervals` because those otherwise became all-zero neural trials. It later added the all-zero-trial drop after the verifier warned about one such trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units/spike_times`, using `units/spike_times_index` to slice the ragged spike buffer. Trial go cues provide the alignment times, and `units/classification` determines which units are included.

ii.
```python
good = np.where(u['classification'][:] == b'good')[0]
sti = np.asarray(u['spike_times_index'][:])
...
spikes = np.asarray(u['spike_times'][a:sti[iu]])
```

```python
g = go[trials]
edges = g[:, None] + OFF_START + np.arange(N_BINS + 1)[None, :] * BIN_SIZE
```

iii. In the trajectory, the AI explicitly concluded that "units have a QC-classifier label ('good')" and that "spike times are session-absolute," so it chose to derive firing rates directly from the stored spike times around each trial's go cue.

## 2-b. How is the `neural` data processed?

i. For each kept unit and trial, spike times are binned into non-overlapping 50 ms bins over `[-2.5, 1.5]` s relative to the go cue. Counts are computed by `searchsorted` against flattened bin edges and converted to firing rates by dividing by the bin width. No smoothing or normalization is applied.

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

iii. In the trajectory, the AI justified this as the direct way to satisfy the task's "50 ms bins" requirement while staying close to the reference preprocessing. It also added the sort guard because its binning assumes monotonic spike times.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == b'good'` are retained. Sessions with zero such units are rejected at the session-selection stage.

ii.
```python
ngood=int((f['units']['classification'][:] == b'good').sum()),
```

```python
good = np.where(u['classification'][:] == b'good')[0]
```

```python
used = (perf > MIN_PERFORMANCE and ncl >= MIN_CORRECT_PER_DIRECTION
        and ncr >= MIN_CORRECT_PER_DIRECTION and info['ngood'] > 0)
```

iii. In the trajectory, the AI said this `classification` field was the white-paper QC classifier output and that using its `'good'` label reproduced the paper's reported good-unit counts. That was its stated reason for preferring it over other possible labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go-cue onset. For each kept trial, the AI constructs absolute bin edges by adding the fixed relative window to that trial's go-cue time.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
```

```python
g = go[trials]
edges = g[:, None] + OFF_START + np.arange(N_BINS + 1)[None, :] * BIN_SIZE
```

iii. In the trajectory, the AI explicitly described the target as "go-cue aligned, -2.5 to +1.5 s, 50 ms non-overlapping bins" and said all event streams shared the same session-absolute clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins, producing 80 bins per trial over the 4 s window. No additional temporal rebinning or smoothing is applied.

ii.
```python
BIN_SIZE = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

```python
rates[k] = np.diff(pos, axis=1).astype(np.float32) / BIN_SIZE
```

iii. In the trajectory, the AI tied this directly to the decoder instructions and contrasted it with the 40 ms / 3.4 ms stride settings it found in the method paper, explicitly choosing the task-required 50 ms bins instead.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` in `BehavioralEvents`, together with the per-trial go cue. The AI chooses the last sample/tone onset before the trial's go cue.

ii.
```python
def tone_onset_times(f, start, go):
    samp = _event_times(f, 'sample_start_times')
    ...
    for i, s in zip(idx, samp):
        if i < 0 or i >= len(out) or np.isnan(go[i]) or s > go[i]:
            continue
        if np.isnan(out[i]) or s > out[i]:
            out[i] = s
```

iii. In the trajectory, the AI said early licks can replay the sample epoch, so it wanted "the last tone onset before the go cue," which it described as "the tone the animal actually had to remember."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After building the go-cue-centered bin centers in absolute time, the AI subtracts the selected tone onset time for that trial. The result is a continuous, time-varying signal in seconds since tone onset.

ii.
```python
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
...
dt_tone = (centers - tone[trials][:, None]).astype(np.float32)
```

iii. In the trajectory, the AI justified this as the simplest way to express "time from tone onset" on the same bins as the neural data.

## 3-c. How is `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same bin centers used for the neural firing-rate bins, so each time point in the input array corresponds to the same 50 ms interval as the neural data.

ii.
```python
edges = g[:, None] + OFF_START + np.arange(N_BINS + 1)[None, :] * BIN_SIZE
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
...
dt_tone = (centers - tone[trials][:, None]).astype(np.float32)
```

iii. In the trajectory, the AI repeatedly described the input streams as sharing the go-cue-aligned grid, with absolute event times converted onto that common axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, rather than from the trial-table `photostim_onset` and `photostim_duration` columns.

ii.
```python
ps = _event_times(f, 'photostim_start_times')
pe = _event_times(f, 'photostim_stop_times')
```

iii. In the trajectory, the AI said the behavioral-event timestamps were already in the same session-absolute clock as go cues and spikes, so it preferred the event stream as the direct source for time-varying stimulation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI creates a binary `(n_trials, n_bins)` array initialized to 0. For each photostimulation interval, it marks a bin as 1 if that bin overlaps the stim interval at all.

ii.
```python
stim_on = np.zeros((len(trials), N_BINS), dtype=np.float32)
for s_, e_ in zip(ps, pe):
    ov = (edges[:, 1:] > s_) & (edges[:, :-1] < e_)
    stim_on[ov] = 1.0
inputs = np.stack([dt_tone, stim_on], axis=1)
```

iii. In the trajectory, the AI summarized this as "photostimulation on" and said photostim occurs in the last 0.5 s of the delay. Its justification was to represent the light as a time-varying binary signal on the same grid as the decoder inputs.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Alignment is done in absolute time using the same trial-specific bin edges used for spike binning. Photostim intervals are compared directly against those edges.

ii.
```python
edges = g[:, None] + OFF_START + np.arange(N_BINS + 1)[None, :] * BIN_SIZE
...
ov = (edges[:, 1:] > s_) & (edges[:, :-1] < e_)
```

iii. In the trajectory, the AI justified this by stating that go cue, photostim, and spikes all share one clock, so no extra interpolation or offset correction was needed.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not stored directly. The AI derives it from the trial-table `outcome` and `trial_instruction` fields: hit means the instructed side, miss means the opposite side, and ignore means no lick.

ii.
```python
outcome = info['outcome'][trials]
instruction = info['instruction'][trials]
```

```python
choice = np.full(len(trials), 2, dtype=np.int64)
hit = outcome == 'hit'
miss = outcome == 'miss'
choice[hit & (instruction == 'left')] = 0
choice[hit & (instruction == 'right')] = 1
choice[miss & (instruction == 'left')] = 1
choice[miss & (instruction == 'right')] = 0
```

iii. In the trajectory, the AI said "choice can be derived from outcome+instruction" and treated `ignore` as the no-lick case.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived categorical choice is encoded as `0 = left`, `1 = right`, `2 = no lick`, then repeated across all 80 bins so it can live in the same time axis as the other outputs.

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
                    tongue_code], axis=1)
```

iii. In the trajectory, the AI justified repeating per-trial outputs across time as the way to satisfy the decoder format, which expects `(n_output, n_timepoints)` arrays.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table `outcome` column.

ii.
```python
outcome = info['outcome'][trials]
```

iii. In the trajectory, the AI described the trial table as already containing the needed outcome labels (`hit`, `miss`, `ignore`), so it did not derive them from other streams.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats the per-trial category across all bins.

ii.
```python
outcome_code = np.zeros(len(trials), dtype=np.int64)
outcome_code[miss] = 1
outcome_code[hit] = 2
```

```python
outputs = np.stack([np.repeat(choice[:, None], N_BINS, axis=1),
                    np.repeat(outcome_code[:, None], N_BINS, axis=1),
                    ...
```

iii. In the trajectory, the AI treated this as a direct categorical remapping required by the output schema.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table `early_lick` column.

ii.
```python
early = info['early'][trials]
```

iii. In the trajectory, the AI treated early licking as explicitly stored trial metadata and kept those trials because early lick is itself a decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI encodes `early == 'early'` as 1 and everything else as 0, then repeats the per-trial value across bins.

ii.
```python
early_code = (early == 'early').astype(np.int64)
```

```python
outputs = np.stack([np.repeat(choice[:, None], N_BINS, axis=1),
                    np.repeat(outcome_code[:, None], N_BINS, axis=1),
                    np.repeat(early_code[:, None], N_BINS, axis=1),
                    tongue_code], axis=1)
```

iii. In the trajectory, the AI justified keeping these trials specifically because the decoder is supposed to predict early licking.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. The AI uses column 1 as `y`, column 2 as DeepLabCut likelihood, and the matching timestamp array for alignment.

ii.
```python
grp = f['acquisition/BehavioralTimeSeries']
key = 'Camera0_side_TongueTracking'
...
data = np.asarray(grp[key]['data'][:])
ts = np.asarray(grp[key]['timestamps'][:])
y = data[:, 1]
vis = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
```

iii. In the trajectory, the AI said all sessions had side-camera tongue tracking and summarized the series as "tongue x/y/likelihood at 300 Hz," which is why it used this channel.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI marks a frame visible when likelihood exceeds `0.9`, averages visible `y` values within each trial/bin using cumulative sums over timestamps, and assigns bins with no visible frames to class 3. The 40th and 60th percentile cutoffs are computed over all visible frames in the session, not over session-wide 50 ms bin means.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
TONGUE_LOW_PCTL = 40.0
TONGUE_HIGH_PCTL = 60.0
```

```python
vis = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
vis &= np.isfinite(y)
lo, hi = np.percentile(y[vis], [TONGUE_LOW_PCTL, TONGUE_HIGH_PCTL]) if vis.sum() else (0., 0.)
...
pos = np.searchsorted(ts, edges.ravel()).reshape(ntrials, nedges)
n = cum_n[pos[:, 1:]] - cum_n[pos[:, :-1]]
s = cum_y[pos[:, 1:]] - cum_y[pos[:, :-1]]
...
mean_y = np.where(n > 0, s / np.maximum(n, 1), np.nan)
```

iii. In the trajectory, the AI justified the high likelihood cutoff by saying the DLC likelihood was "strongly bimodal" so the threshold was not important. It also explicitly said it was choosing per-session 40th/60th percentile boundaries on visible tongue data.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses per-session percentiles of visible frame-level `y` values: class 0 for values below the 40th percentile, class 1 for values between the 40th and 60th percentiles inclusive, class 2 above the 60th percentile, and class 3 when no visible frame falls in the bin.

ii.
```python
lo, hi = np.percentile(y[vis], [TONGUE_LOW_PCTL, TONGUE_HIGH_PCTL]) if vis.sum() else (0., 0.)
...
code = np.full((ntrials, nbins), 3, dtype=np.int64)
seen = n > 0
code[seen & (mean_y < lo)] = 0
code[seen & (mean_y >= lo) & (mean_y <= hi)] = 1
code[seen & (mean_y > hi)] = 2
```

iii. In the trajectory, the AI described this as "discretised per session" and emphasized that most bins have no visible tongue, motivating the explicit class-3 label.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue data with the neural data by using the same absolute trial edges used for spike binning. Camera timestamps are `searchsorted` against those edges, then visible tongue frames between successive edges are averaged into the corresponding bin.

ii.
```python
pos = np.searchsorted(ts, edges.ravel()).reshape(ntrials, nedges)
n = cum_n[pos[:, 1:]] - cum_n[pos[:, :-1]]
s = cum_y[pos[:, 1:]] - cum_y[pos[:, :-1]]
```

iii. In the trajectory, the AI said the video timestamps were on the same global clock as the spikes and go cues, so using the same bin edges was sufficient for alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several cases by exclusion or explicit coding. Sessions with no `good` units are dropped. Trials without go cues or tone onsets, outside `obs_intervals`, marked `free_water` or `auto_water`, or with all-zero neural activity are dropped. Invisible tongue frames are excluded from the bin mean, and bins with no visible frames become class 3. Spike times are defensively sorted if a unit's spike array is not monotonic.

ii.
```python
used = (perf > MIN_PERFORMANCE and ncl >= MIN_CORRECT_PER_DIRECTION
        and ncr >= MIN_CORRECT_PER_DIRECTION and info['ngood'] > 0)
```

```python
keep = (~np.isnan(go)) & (~np.isnan(tone)) & (~info['free_water']) & (~info['auto_water'])
keep &= observed_trials(f, start)
```

```python
if spikes.size and np.any(np.diff(spikes) < 0):
    spikes = np.sort(spikes)
```

```python
code = np.full((ntrials, nbins), 3, dtype=np.int64)
```

iii. In the trajectory, the AI framed these choices as avoiding fabricated neural zeros, matching the paper's selected-session set, and representing missing tongue observations as an explicit "not visible" state rather than imputing values.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive steps are session I/O, per-unit spike binning, video loading/processing for tongue labels, and multiprocessing-related serialization. The code also rereads every NWB file in `session_info()` and `process_session()`, then writes and rereads temporary per-session pickle files during assembly.

ii. 
```python
def session_info(path):
    with h5py.File(path, 'r') as f:
        ...
```

```python
def process_session(path):
    info = session_info(path)
    ...
    with h5py.File(path, 'r') as f:
        ...
```

```python
for k, iu in enumerate(good):
    ...
    pos = np.searchsorted(spikes, flat_edges).reshape(len(trials), N_BINS + 1)
```

```python
with Pool(args.nproc) as pool:
    files = pool.map(_worker, paths)
```

iii. In the trajectory, the AI explicitly called out "per-unit searchsorted," full-session video arrays, and large spike buffers as the main costs, and it chose multiprocessing plus temporary session pickles to keep the implementation simple.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining loops are the per-unit neural-binning loop, the per-event photostim loop, and several assembly-time Python list membership/index loops for regions and subjects. These were left as explicit Python loops.

ii.
```python
for k, iu in enumerate(good):
    ...
```

```python
for s_, e_ in zip(ps, pe):
    ov = (edges[:, 1:] > s_) & (edges[:, :-1] < e_)
    stim_on[ov] = 1.0
```

```python
for r in s['regions']:
    if r not in regions:
        regions.append(r)
...
if s['subject'] not in subjects:
    subjects.append(s['subject'])
subject_idx.append(subjects.index(s['subject']))
```

iii. In the trajectory, the AI mainly discussed the per-unit loop as the unavoidable ragged-data cost. It did not claim to have optimized the smaller Python loops further.

## 10-c. What processing does the code repeat multiple times?

i. It repeats file access and serialization work. Every session is opened once in `session_info()` and again in `process_session()`. After processing, each session is pickled to a temporary file and later unpickled during final assembly.

ii.
```python
def process_session(path):
    info = session_info(path)
    ...
    with h5py.File(path, 'r') as f:
        ...
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

iii. In the trajectory, the AI justified the first pass as needed for session selection and the temp files as a convenient multiprocessing boundary, but the net effect is repeated processing compared with a single-pass in-memory assembly.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main discarded work is temporary per-session pickling: each worker writes an intermediate pickle file that is immediately reread during assembly and never used downstream. The code also computes some metadata-only statistics such as session performance and `frac_observed`, which are kept in metadata but not used by the decoder.

ii.
```python
fn = os.path.join(TMP_DIR, os.path.basename(path) + '.pkl')
with open(fn, 'wb') as fh:
    pickle.dump(out, fh, protocol=4)
...
for fn in files:
    with open(fn, 'rb') as fh:
        s = pickle.load(fh)
```

```python
performance=perf,
ncorrect_left=ncl, ncorrect_right=ncr,
...
frac_observed=float(np.mean(np.clip(...)))
```

iii. In the trajectory, the AI described the temp pickles as an implementation convenience for parallel conversion, not as part of the scientific output. It kept the extra metadata for documentation and debugging rather than because the decoder consumed it.
