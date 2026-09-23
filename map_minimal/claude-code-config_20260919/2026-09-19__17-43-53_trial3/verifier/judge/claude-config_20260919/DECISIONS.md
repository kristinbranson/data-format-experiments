# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session laid out as `/app/data/sub-<id>/sub-<id>_ses-<datetime>_behavior+ecephys(+ogen).nwb`. The AI finds every session with a single sorted glob over that layout (174 files, 28 mice) and converts each file independently. Files are opened with **`h5py` directly** rather than `pynwb`, reading the HDF5 paths `intervals/trials`, `units/*`, `acquisition/BehavioralEvents/*` and `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. Sessions are processed in a 16-process `multiprocessing.Pool`, and each session's result is memoised to `/app/cache/<session>.pkl` so re-runs skip completed sessions. A separate resource, the Allen CCF structure graph, is downloaded once (`/app/allen_structure_graph.json`) and used to map each unit's `anno_name` to a coarse brain region.

ii.
```python
DATA_DIR = '/app/data'
...
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
print(f"{len(files)} NWB sessions found", flush=True)

with Pool(n_workers) as pool:
    caches = pool.map(_worker, [(p, region_map) for p in files], chunksize=1)
```
```python
def _worker(args):
    path = args[0]
    cache = os.path.join(CACHE_DIR, os.path.basename(path).replace('.nwb', '.pkl'))
    if os.path.exists(cache):
        return cache
    ...
    res = convert_session(args)
```
```python
with h5py.File(path, 'r') as f:
    tbl = session_trial_table(f)
    ...
    units = f['units']
```
```python
def session_trial_table(f):
    trials = f['intervals/trials']
    tbl = {
        'start_time': trials['start_time'][:],
        'stop_time': trials['stop_time'][:],
        ...
    }
    tbl['go_time'] = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
    assert len(tbl['go_time']) == len(tbl['start_time'])
    return tbl
```

iii. From the agent's summary: *"Loading: NWB `units/spike_times` (session clock), trials table, `BehavioralEvents` (go/sample/photostim/lick times), `BehavioralTimeSeries/Camera0_side_TongueTracking`."* The AI first ran a full survey pass (`scan.py`, `scan2.py`) over all 174 files to enumerate the trial-table columns, event streams, tracking series, unit classifications and CCF annotations before committing to a schema, and verified that the field layout is identical across all files. `h5py` was chosen over `pynwb` because only a handful of fixed HDF5 paths are needed, which avoids the cost of building the full NWB object model; the parallel pool plus per-session cache was added so the 174-session conversion fits comfortably in the time budget and is resumable.

## 1-b. How are the data split into subjects?

i. The subject id is parsed from the file/directory name (`sub-440956` → `'440956'`). `subjects` is built as an `OrderedDict` in the order sessions are first encountered; because the file list is sorted, this is the same as sorted order. `subject_idx` is the index of each kept session's subject into that list. All 28 mice survive the session filtering.

ii.
```python
subject = os.path.basename(path).split('_')[0].replace('sub-', '')
out = {'file': path, 'session_name': name, 'subject': subject}
```
```python
if res['subject'] not in subjects:
    subjects[res['subject']] = len(subjects)
...
data['subject_idx'].append(subjects[res['subject']])
...
data['subjects'] = list(subjects.keys())
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The DANDI layout puts one directory per animal, and the directory name is derived from the NWB `subject_id`, so the path is a sufficient and unambiguous animal identifier — no grouping step is needed. The AI's survey pass confirmed 28 unique subjects with 3–10 sessions each, consistent with `data/dandiset.yaml`. The numeric DANDI ids (not the `SC015`-style mouse names in the papers) are used as the subject names.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no splitting or grouping is required. The session name is the file stem. The AI then applies **session-level quality control**, rejecting 29 of 174 sessions and keeping 145:

- **Behavioral QC (20 sessions)** — the data paper's stated session selection criteria: control-trial performance > 65% and ≥ 50 correct trials in each lick direction. "Control" is defined as no photostimulation, no early lick, no free water, and a response given.
- **No quality-controlled units (1 session)** — `sub-440958_ses-20190216T162508`, whose `classification`/`anno_name` are NaN for all units.
- **Video coverage (7 sessions)** — the side camera covers < 95 % of the analysis window (in 6 of these the camera stops at the go cue; in one it ran 47 s total).
- **Tongue tracking failure (1 session)** — the tracker's "visible" frames do not agree with the lick-port sensors (precision 0.12 vs 0.29 for the next-worst session).

ii.
```python
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50
```
```python
def session_performance(tbl):
    no_early = tbl['early_lick'] == 'no early'
    control = no_early & (tbl['photostim_onset'] == 'N/A') & (tbl['free_water'] == 0)
    responded = control & (tbl['outcome'] != 'ignore')
    n_resp = int(responded.sum())
    perf = float((tbl['outcome'][responded] == 'hit').sum()) / n_resp if n_resp else 0.0
    hit = tbl['outcome'] == 'hit'
    n_left = int((hit & (tbl['instruction'] == 'left') & no_early).sum())
    n_right = int((hit & (tbl['instruction'] == 'right') & no_early).sum())
    return perf, n_left, n_right
```
```python
if perf <= MIN_PERFORMANCE or min(n_left, n_right) < MIN_CORRECT_PER_DIRECTION:
    out['rejected'] = 'behavior'
    return out
```
```python
coverage = float(has_video[observed].mean())
out['video_coverage'] = coverage
if coverage < MIN_VIDEO_COVERAGE:
    out['rejected'] = 'video does not cover the analysis window'
    return out

if precision < MIN_TONGUE_LICK_PRECISION or recall < MIN_TONGUE_LICK_RECALL:
    out['rejected'] = 'tongue tracking failed'
    return out
```

iii. From the agent's summary: *"Sessions — data paper criteria: control-trial performance > 65% and ≥ 50 correct trials per direction. This gives 154/174 sessions with mean performance 83.4% (range 65.6–98.9%), matching the paper's '84% (65–99%)'."* The AI treated the reproduction of the paper's reported performance statistics as evidence that its reconstruction of the criterion is the one the authors used. The extra video QC is justified on the grounds that the tongue output simply cannot be built for those sessions: *"7 sessions whose camera stops at the go cue (or ran 47 s total) covered only 1–64% of the window while every other session covers ≥ 99.97%; 1 session where the tracker reports the tongue visible in 96% of frames and disagrees with the lick sensors."* The AI checked that both thresholds sit in a wide empirical gap, so the cuts are not threshold-sensitive.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table (`intervals/trials`), one row per behavioural trial. The AI asserts that `go_start_times` has exactly one event per row, which makes the trial ↔ go-cue mapping one-to-one; all per-trial quantities are indexed by the trials-table row. `stop_time`/`start_time` of the row also define what part of the analysis window was actually recorded (see 2-e).

ii.
```python
tbl['go_time'] = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(tbl['go_time']) == len(tbl['start_time'])
```
```python
'trial_id': trials['trial'][:].astype(np.int64),
```

iii. The trials table is the file's own definition of a trial, so it is used directly rather than re-derived from the event streams. The AI explicitly verified the one-go-cue-per-trial property with an assertion that runs on every session (it never fired), and its exploration confirmed the contrast with `sample_start_times`/`delay_start_times`, which can have several entries per trial because an early lick replays the epoch.

## 1-e. How are trials filtered based on quality controls?

i. Three per-trial filters, all about data availability rather than behaviour:

1. **`free_water == 0`** — free-water trials excluded, per the method paper (2,247 trials).
2. **At least `MIN_BINS_PER_TRIAL = 20` observed bins** in the analysis window (in practice this never fires: 0 trials excluded).
3. **At least one spike anywhere in the trial's window** — this removes 1,061 trials in 8 sessions where the ephys recording covers only a contiguous block of the behavioural session.

A session with fewer than 2 surviving trials is rejected. Early-lick, ignore and photostim trials are deliberately **kept**, against the data paper's analysis convention, because the decoder spec makes them inputs/outputs. Net: 75,670 trials in 145 sessions.

ii.
```python
# DEVIATION: the same sentence excludes photoinhibition, early-lick and
# ignore trials.  They are kept here because the decoding task asks for
# photostimulation as an input and for early lick / ignore as outputs;
# dropping them would leave those variables constant.
bins = [mask_bins(tbl, t) for t in range(len(tbl['start_time']))]
short = np.array([len(b) < MIN_BINS_PER_TRIAL for b in bins])
keep_trial = (tbl['free_water'] == 0) & ~short
trials = np.flatnonzero(keep_trial)
```
```python
# In eight sessions the ephys recording covers only a contiguous block
# of the behavioral session ... A trial
# in which not one of several hundred units fires for four seconds is a
# recording gap rather than a physiological observation, so those trials
# are dropped.  The transition is clean: the trials bordering such a
# block fire at 0.77-1.2x the session's median population rate.
has_spikes = rates.sum(axis=(0, 2)) > 0
out['n_trials_no_ephys'] = int((~has_spikes).sum())
trials = trials[has_spikes]
rates = rates[:, has_spikes]
if len(trials) < MIN_TRIALS_PER_SESSION:
    out['rejected'] = 'no ephys'
    return out
```

iii. From the agent's summary: *"Trials — free-water trials excluded (per the method paper); photostim, early-lick and ignore trials kept — the spec makes them an input/outputs. Also dropped 1,061 trials in 8 sessions where the file contains no spikes at all (probes inserted late / recording stopped early)."* Rather than reading `units/obs_intervals`, the AI detected the recording gaps empirically and then validated the detection: it confirmed the all-zero trials form a single contiguous block per session and that the trials bordering each block fire at 0.77–1.2× the session's median population rate, i.e. the boundary is a hard recording edge rather than a gradual drop-out. The deviation from the papers on early-lick/ignore/photostim trials is explicitly labelled `DEVIATION` in the source and tied to the decoder spec.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` with `units/spike_times_index` (the ragged offsets), restricted to units selected by `units/classification` and `units/anno_name`; and the go-cue timestamps (`acquisition/BehavioralEvents/go_start_times/timestamps`), which place the bin edges. `units/anno_name` additionally supplies each neuron's brain region.

ii.
```python
units = f['units']
classification = _str_col(units['classification'])
anno = _str_col(units['anno_name'])
good = np.flatnonzero((classification == 'good') & (anno != ''))
```
```python
spike_index = units['spike_times_index'][:]
edges = (go[trials][:, None] + BIN_EDGES[None, :]).ravel()
rates = np.zeros((len(good), len(trials), NBINS), dtype=np.float32)
for i, u in enumerate(good):
    lo = 0 if u == 0 else spike_index[u - 1]
    st = units['spike_times'][lo:spike_index[u]]
```

iii. Spike times are the only neural representation in the files, so firing rates are computed from them directly. The AI checked `units/is_good_trials` and `units/unit_quality` during exploration and chose `classification` instead (see 2-c).

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin, converted to firing rate in Hz by dividing by the bin width. For each selected unit the whole session's bin edges are laid out as one flat array of absolute times, `np.searchsorted` gives the running spike count at every edge, and `np.diff` turns that into per-bin counts. No smoothing, no baseline subtraction, no normalisation, no firing-rate threshold. Stored as `float32`.

ii.
```python
edges = (go[trials][:, None] + BIN_EDGES[None, :]).ravel()
rates = np.zeros((len(good), len(trials), NBINS), dtype=np.float32)
for i, u in enumerate(good):
    lo = 0 if u == 0 else spike_index[u - 1]
    st = units['spike_times'][lo:spike_index[u]]
    counts = np.diff(
        np.searchsorted(st, edges).reshape(len(trials), NBINS + 1), axis=1)
    rates[i] = counts / BIN_WIDTH
```

iii. Binned spike count / bin width is what the method-paper code does (`sliding_histogram(..., rate=True)`), and the 50 ms width comes from the decoder spec. The AI explicitly declined the method paper's 2 Hz firing-rate cut: *"I did not apply the method paper's 2 Hz firing-rate cut, which was specific to their R² analysis, not a quality filter."*

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three conditions, applied per unit:
- `units/classification == 'good'` — the verdict of the data paper's per-brain-area logistic-regression QC classifier over 15 cluster quality metrics (`ChenLiuEtAl2023_SpikeSortingQC.pdf`). No individual metric thresholds are applied.
- `anno_name != ''` — the unit has a CCF annotation (needed for `brain_region_idx`; the AI verified the good units are exactly the annotated ones).
- at least one spike somewhere in the analysis windows (`rates.any(...)`) — in practice this never fires (0 units dropped).

A session with no surviving units is rejected. 57,774 neurons are kept across the 145 retained sessions.

ii.
```python
# The data paper's quality control trains a per-brain-area logistic
# regression classifier on 15 cluster quality metrics and keeps the
# units it labels 'good' (units/classification); 69,453 of the 272,227
# clusters in these files pass (25.5%), matching the 69,943 "good units"
# and "25.9% of clusters reported by Kilosort2" of the paper.  Those
# units are exactly the ones carrying a CCF annotation.
units = f['units']
classification = _str_col(units['classification'])
anno = _str_col(units['anno_name'])
good = np.flatnonzero((classification == 'good') & (anno != ''))
if len(good) < MIN_NEURONS_PER_SESSION:
    out['rejected'] = 'no good units'
    return out
```
```python
# Drop units without a single spike anywhere in the analysis windows:
# they carry no information and give the per-session PCA a null
# direction.
alive = rates.any(axis=(1, 2))
...
good = good[alive]
rates = rates[alive]
```

iii. From the agent's summary: *"Neurons — kept units with `classification == 'good'`, the data paper's QC-classifier label. This reproduces the paper's counts almost exactly (69,453/272,227 = 25.5% vs '25.9% of clusters'; striatum 7,664, thalamus 12,808, midbrain 7,495, medulla 2,928 — all exact)."* The AI compared `unit_quality` against `classification` during exploration and used `classification` because it is the published QC verdict. The silent-unit filter is justified on downstream grounds: an all-zero neuron gives the decoder's per-session PCA a null direction.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams share one session-absolute clock, so no resampling or offset correction is needed. Alignment is done by adding the go-cue-relative bin edges to each trial's go-cue timestamp and binning the spikes against those absolute edges.

ii.
```python
ALIGN_EVENT = 'go cue onset'
...
tbl['go_time'] = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```
```python
go = tbl['go_time']
edges = (go[trials][:, None] + BIN_EDGES[None, :]).ravel()
```
```python
'temporal_alignment_event':
    'Onset of the auditory go cue that ends the delay epoch',
```

iii. `go_start_times` is the go cue and has exactly one event per trial, so the alignment is unambiguous. Because spikes, behavioural events and camera timestamps are all on the same clock, the same go-cue-relative grid can be applied to every stream, which is what guarantees bin *k* of the neural data covers the same interval as bin *k* of the inputs and tongue output.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins on the grid −2.5 s to +1.5 s relative to the go cue, i.e. 80 bins. `time_bin_size` is recorded as 50.0 ms, `off_start = -2.5`, `off_end = 1.5`. Spikes are binned once at this resolution; there is no rebinning of an intermediate representation.

**The one substantive deviation:** the AI does **not** emit all 80 bins for every trial. It found that the NWB files contain spikes only inside each trial's `[start_time, stop_time]` interval, and that an incorrect lick ends the trial almost immediately — so on ~17 % of trials (nearly all "miss") the tail of the window is outside the recorded interval. Those bins are **dropped rather than zero-filled**, so `T` varies per trial (82.6 % of trials keep all 80 bins; mean 77.7, min 51). The bin grid and bin width are identical across trials; only the extent varies, and `trial_metadata` records each trial's first-bin index and bin count.

ii.
```python
T_START = -2.5           # s relative to the go cue
T_END = 1.5              # s relative to the go cue
BIN_WIDTH = 0.05         # s.  Non-overlapping bins (stride == width).
BIN_EDGES = np.round(T_START + BIN_WIDTH * np.arange(
    int(round((T_END - T_START) / BIN_WIDTH)) + 1), 10)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
NBINS = len(BIN_CENTERS)
```
```python
def mask_bins(tbl, trial):
    """Indices of the bins of the analysis window that hold observed spikes.

    The NWB files only contain spikes inside each trial's interval
    (units/obs_intervals is exactly [start_time, stop_time] for every unit, and
    100% of spikes fall inside it) ...
    Bins outside the observed interval are therefore dropped rather than filled
    with zeros.  Filling them would invent "all neurons silent" samples whose
    presence is perfectly correlated with the miss outcome, which the decoder
    would read as a free answer ...
    """
    lo = tbl['start_time'][trial] - tbl['go_time'][trial]
    hi = tbl['stop_time'][trial] - tbl['go_time'][trial]
    keep = np.flatnonzero((BIN_EDGES[:-1] >= lo) & (BIN_EDGES[1:] <= hi))
    return keep
```
```python
for k, t in enumerate(trials):
    b = bins[t]
    neural.append(np.ascontiguousarray(rates[:, k, b]))
```

iii. From the agent's summary: *"The one substantive judgment call: the NWB files hold spikes only inside each trial interval, and an incorrect lick ends the trial immediately — so on ~17% of trials (nearly all 'miss') the [-2.5, 1.5] s window runs past the recorded data. I kept the specified grid and bin width but dropped the unobserved bins rather than zero-filling them, so T varies per trial. Zero-filling would have manufactured 'all neurons silent' samples perfectly correlated with the miss outcome — a free answer for the decoder — and dropping the trials instead would have deleted the miss class. This is why outcome accuracy is 0.65 rather than near-ceiling."* The AI verified that `obs_intervals` equals `[start_time, stop_time]` for every unit and that 100 % of spikes fall inside it, so the dropped bins genuinely contain no observations.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the instruction-tone / sample-epoch onsets), together with the trials table's `start_time` (to assign each onset to a trial) and `go_time` (to express the onset relative to the alignment event). The tone used for a trial is the **last** sample onset inside that trial that precedes the go cue.

ii.
```python
def tone_onset_times(f, tbl):
    """Time of the instruction tone (sample epoch onset) for every trial.

    Licking during the sample/delay epoch triggers a replay of the epoch, so a
    trial can contain several sample-epoch onsets.  The onset that determines
    the trial structure leading up to the go cue is the last one before the go
    cue, which is what is used here. ...
    """
    sample = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
    go = tbl['go_time']
    start = tbl['start_time']
    tone = np.full(len(go), np.nan)
    # Assign each sample-epoch onset to the trial containing it, keep the last.
    trial_of = np.searchsorted(start, sample, side='right') - 1
    valid = (trial_of >= 0) & (trial_of < len(go))
    for t, s in zip(trial_of[valid], sample[valid]):
        if s <= go[t]:
            tone[t] = s
    rel = tone - go
    if np.any(np.isnan(rel)):
        rel[np.isnan(rel)] = np.nanmedian(rel)
    return rel
```

iii. From the agent's summary: *"signed seconds from tone onset (last sample-epoch onset before the go cue, accounting for early-lick replays)."* The AI explicitly measured the number of sample onsets per trial and the distribution of `tone − go` for both the first and last onset across several sessions before choosing the last one; `methods.txt` states that "licking early during the sample/delay epoch triggered a replay of the epoch", so a trial can legitimately contain several onsets and only the last one sets up the delay that the go cue terminates.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. `tone_rel = tone − go` is a per-trial scalar (negative: the tone precedes the go cue, by 1.85 s in most sessions and 0.95 s in the rest). The per-bin input value is the bin centre (which is go-cue-relative) minus that scalar, i.e. seconds elapsed since the tone at the centre of each bin; it is negative for bins preceding the tone. Stored as `float32`, time-varying, over exactly the bins kept for that trial.

ii.
```python
inp = np.empty((2, len(b)), dtype=np.float32)
inp[0] = BIN_CENTERS[b] - tone_rel[t]
```
```python
'time_from_tone_onset':
    'Seconds from the onset of the instruction tone (sample epoch) '
    'to the center of the time bin; negative before the tone. The '
    'tone precedes the go cue by 1.85 s (0.65 s sample + 1.2 s '
    'delay) in most sessions and by 0.95 s in the rest, and by '
    'more on trials whose epoch was replayed after an early lick.',
```

iii. No processing beyond locating the tone is needed: the quantity is a deterministic shift of the bin-centre axis. The decoder spec asks for it as a continuous, time-varying input, so it is emitted as a real-valued row rather than a binary onset indicator.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed *on* the neural bin grid: `BIN_CENTERS` are the centres of exactly the same bins used to count spikes, and the same per-trial bin mask `b` is applied, so row 0 of `input` has the same length as and is element-wise aligned with the trial's `neural` matrix.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
```
```python
for k, t in enumerate(trials):
    b = bins[t]
    neural.append(np.ascontiguousarray(rates[:, k, b]))
    inp = np.empty((2, len(b)), dtype=np.float32)
    inp[0] = BIN_CENTERS[b] - tone_rel[t]
```

iii. Everything is expressed relative to the go cue on a single shared clock, so alignment is structural rather than a separate step — no interpolation and no per-stream offset correction.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The trials table columns `photostim_onset` and `photostim_duration` (both stored as strings, with `'N/A'` on unstimulated trials), plus `start_time` (the onsets are given relative to trial start) and `go_time` (to re-express them relative to the alignment event). `photostim_power` is read but unused.

ii.
```python
def photostim_intervals(tbl):
    """(onset, offset) of photostimulation relative to the go cue, per trial.

    `photostim_onset` in the trials table is given relative to the trial start
    time; this was verified against acquisition/BehavioralEvents/
    photostim_start_times (exact match for every stimulated trial).  Trials
    without photostimulation get (nan, nan).
    """
    n = len(tbl['start_time'])
    on = np.full(n, np.nan)
    off = np.full(n, np.nan)
    stim = tbl['photostim_onset'] != 'N/A'
    idx = np.flatnonzero(stim)
    for i in idx:
        onset = float(tbl['photostim_onset'][i])
        dur = float(tbl['photostim_duration'][i])
        # relative to trial start -> relative to go cue
        on[i] = tbl['start_time'][i] + onset - tbl['go_time'][i]
        off[i] = on[i] + dur
    return on, off
```

iii. From the agent's summary: *"a binary photostim indicator from the trials table (verified against the photostim event timestamps)."* The AI cross-checked the trials-table onsets against the independent `BehavioralEvents/photostim_start_times` stream and found an exact match on every stimulated trial, which is why the trials table (which also carries the duration) is used as the source.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: a bin is 1 when its centre falls in `[onset, offset)` and 0 otherwise. Unstimulated trials are branched on explicitly (`np.isfinite(stim_on[t])`) and filled with 0. 15,299 of the 75,670 kept trials (20.2 %) carry photostimulation; the stimulus is a 0.5 s ALM photoinhibition in the late delay, always ending before the go cue.

ii.
```python
if np.isfinite(stim_on[t]):
    inp[1] = ((BIN_CENTERS[b] >= stim_on[t]) &
              (BIN_CENTERS[b] < stim_off[t])).astype(np.float32)
else:
    inp[1] = 0.0
```
```python
'photostim_on':
    '1 in bins whose center falls inside the ALM photoinhibition '
    'interval of the trial (0.5 s including a 100 ms ramp down, '
    'always ending before the go cue), 0 otherwise.',
```

iii. The decoder spec asks for "whether photostimulation is on at every time point (discrete, time-varying)", so a per-bin binary indicator is used rather than a per-trial flag. Half-open interval membership on the bin centre is the natural discretisation of a continuous-time interval onto a bin grid.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The onset/offset are converted from trial-start-relative to go-cue-relative seconds (`start_time + onset − go_time`), which is the same axis the bin centres live on, and are then compared against `BIN_CENTERS[b]` — the same masked bin set as the neural matrix.

ii.
```python
on[i] = tbl['start_time'][i] + onset - tbl['go_time'][i]
off[i] = on[i] + dur
```
```python
inp[1] = ((BIN_CENTERS[b] >= stim_on[t]) &
          (BIN_CENTERS[b] < stim_off[t])).astype(np.float32)
```

iii. The only alignment work is the change of reference from trial start to go cue; after that the comparison is directly against the neural bin centres, so the two streams are element-wise aligned by construction.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The lick-port event streams `acquisition/BehavioralEvents/left_lick_times/timestamps` and `right_lick_times/timestamps`, restricted to the 1.5 s answer period after the go cue (`go_time`). This is a **different source** from the trials table: the AI deliberately did *not* derive choice from `trial_instruction` × `outcome`.

ii.
```python
def lick_times(f):
    """Sorted lick times at the left and right ports."""
    be = f['acquisition/BehavioralEvents']
    return (np.sort(be['left_lick_times/timestamps'][:]),
            np.sort(be['right_lick_times/timestamps'][:]))
```
```python
def lick_choice(left, right, tbl):
    """Lick direction chosen by the mouse: 0 left, 1 right, 2 no lick.

    Taken from the first lick recorded at either port during the 1.5 s answer
    period that follows the go cue (the window this conversion extracts).  This
    agrees with the label implied by outcome x instruction (hit -> instructed
    port, miss -> opposite port, ignore -> no lick) on >99.3% of trials; the
    direct read-out from the lick ports is used because "lick direction choice"
    is a behavioral measurement rather than a re-encoding of `outcome`.
    """
```

iii. The AI's stated reason is that choice is a behavioural measurement in its own right, and re-deriving it from `outcome` would make the `choice` and `outcome` outputs algebraically redundant (choice would be a deterministic function of outcome and the instructed side). It quantified the risk first: the lick-port read-out agrees with the `outcome × instruction` label on **more than 99.3 % of trials**, so the two definitions are nearly interchangeable and the direct measurement was preferred.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, find the first left-port lick and the first right-port lick at or after the go cue. If neither occurs before `go + 1.5 s`, the trial is class 2 ("no lick"); otherwise the earlier of the two determines the class (0 left, 1 right). The per-trial scalar is then broadcast across all kept bins of the trial and stored as `int8` in row 0 of `output`.

ii.
```python
go = tbl['go_time']
choice = np.full(len(go), 2, dtype=np.int8)
for i, g in enumerate(go):
    li = np.searchsorted(left, g)
    ri = np.searchsorted(right, g)
    tl = left[li] if li < len(left) else np.inf
    tr = right[ri] if ri < len(right) else np.inf
    end = g + T_END
    if tl >= end and tr >= end:
        continue
    choice[i] = 0 if tl <= tr else 1
```
```python
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ...
]
```
```python
outp = np.empty((4, len(b)), dtype=np.int8)
outp[0] = choice[t]
```

iii. The 1.5 s answer period is the window `methods.txt` specifies for the response, and it coincides with the window this conversion extracts, so "first lick in the response window" is both the behaviourally correct definition and fully contained in the emitted data. Choice is a per-trial property, so it is repeated across bins — the spec asks for time-varying outputs where possible, and repeating keeps all four outputs in one `(n_output, n_timepoints)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials table column `outcome`, which already holds exactly the three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
'outcome': _str_col(trials['outcome']),
```
```python
'outcome': 'Trial outcome from the NWB trials table: ignore (no '
           'response), miss (incorrect port), hit (correct port).',
```

iii. The trials table stores the outcome explicitly with precisely the three categories the decoder spec asks for, so no derivation is needed. The AI's survey pass confirmed that these are the only three values present in any session.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps the strings to `0` ignore, `1` miss, `2` hit — the order given in the decoder spec — and the per-trial code is broadcast across the trial's kept bins into row 1 of `output` as `int8`.

ii.
```python
outcome_code = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.array([outcome_code[o] for o in tbl['outcome']], dtype=np.int8)
```
```python
outp[1] = outcome[t]
```
```python
OUTPUT_VALUES = [
    ...
    ['ignore', 'miss', 'hit'],
```

iii. The code assignment follows the ordering in the instructions, and `output_values[1]` records the names so the mapping is self-describing. As with the other per-trial labels it is repeated across bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The trials table column `early_lick`, whose only values across the dataset are `'no early'` and `'early'`.

ii.
```python
'early_lick': _str_col(trials['early_lick']),
```
```python
'early_lick': 'Whether the mouse licked during the sample or delay '
              'epoch, from the NWB trials table.',
```

iii. The flag is stored explicitly, so no derivation is needed. The event that sets it — a lick during the sample or delay epoch — falls inside the −2.5 s side of the extracted window, so the label is in principle decodable from the neural data in the window even though it is stored per trial.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A boolean comparison `early_lick == 'early'` cast to `int8` (0 no, 1 yes), broadcast across the trial's kept bins into row 2 of `output`.

ii.
```python
early = (tbl['early_lick'] == 'early').astype(np.int8)
```
```python
outp[2] = early[t]
```
```python
OUTPUT_VALUES = [
    ...
    ['no', 'yes'],
```

iii. The coding follows the instructions (`no` = 0, `yes` = 1). Per-trial value repeated across bins, like the other trial-level outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` array is `(n_frames, 3)` = (`tongue_x`, `tongue_y`, `tongue_likelihood`) at ~300 Hz with matching `timestamps`. Column 1 is the value; column 2 (the DeepLabCut likelihood) decides visibility. The left/right lick-port event streams are also read, but only as an independent check on the tracking (session QC, see 1-c).

ii.
```python
key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
if key not in f:
    return None, None, None
ts = f[key + '/timestamps'][:]
data = f[key + '/data'][:]
y = data[:, 1].astype(np.float64)
visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
```

iii. This is the only tongue measurement in the files. The AI verified the channel layout, the frame rate, and the shape of the likelihood distribution before using it, and checked that the side-view tongue series is present in all 174 sessions.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps:
1. **Visibility mask** — frames with DeepLabCut likelihood ≤ 0.5 are treated as "tongue not visible"; the tracker still emits a position when the tongue is retracted, so those positions are meaningless.
2. **Five-sigma velocity outlier imputation** — following the method paper, frame-to-frame velocity is computed among visible frames that are *adjacent in the video* (Δt < 10 ms, since the recording is trial-gated and the tongue is visible only in bouts); frames whose position jumps by more than five sigma both on the way in and on the way out are replaced by linear interpolation from neighbouring visible frames.
3. **Binning** — per trial, per 50 ms bin, the mean of `tongue_y` over the *visible* frames falling in that bin; a bin with no visible frame is left `NaN`.
4. **Discretisation** into 4 classes (see 8-c).

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.5
MARKER_OUTLIER_SIGMA = 5.0
```
```python
vis_idx = np.flatnonzero(visible)
if len(vis_idx) > 3:
    yv = y[vis_idx]; tv = ts[vis_idx]
    dt = np.diff(tv)
    # Velocity only between frames that are adjacent in the video; the
    # recording is split into per-trial segments and the tongue is only
    # visible in bouts, so most consecutive visible frames are not adjacent.
    adjacent = (dt > 0) & (dt < 0.01)
    vel = np.full(len(dt), np.nan)
    vel[adjacent] = np.diff(yv)[adjacent] / dt[adjacent]
    sigma = np.nanstd(vel) if adjacent.any() else np.nan
    if np.isfinite(sigma) and sigma > 0:
        jump = np.abs(vel) > MARKER_OUTLIER_SIGMA * sigma
        # A frame is an outlier when the position jumps on the way in and
        # again on the way out.
        outlier = np.zeros(len(yv), dtype=bool)
        outlier[1:-1] = jump[:-1] & jump[1:]
        if outlier.any() and (~outlier).sum() > 1:
            keep = np.flatnonzero(~outlier)
            yv = np.interp(np.arange(len(yv)), keep, yv[keep])
            y = y.copy(); y[vis_idx] = yv
```
```python
bin_y = np.full((len(trials), NBINS), np.nan)
has_video = np.zeros((len(trials), NBINS), dtype=bool)
for k, t in enumerate(trials):
    lo = np.searchsorted(ts, go[t] + BIN_EDGES[:-1])
    hi = np.searchsorted(ts, go[t] + BIN_EDGES[1:])
    has_video[k] = hi > lo
    for b in range(NBINS):
        if hi[b] <= lo[b]:
            continue
        sl = slice(lo[b], hi[b])
        vis = tongue_vis[sl]
        if vis.any():
            bin_y[k, b] = tongue_y[sl][vis].mean()
```

iii. From the agent's summary: *"tongue y averaged over visible frames (DLC likelihood > 0.5, bimodal so the threshold is immaterial) per 50 ms bin, with the paper's five-sigma velocity outlier imputation."* The AI measured the likelihood distribution first — *"median 6e-5 when the tongue is in the mouth vs > 0.999 when it is out; only 0.07 % of frames fall between 0.5 and 0.9"* — to show that the 0.5 cut is not sensitive. The outlier step is taken verbatim from the method paper (*"We identified outliers by a five-sigma threshold on velocity across frames and imputed outliers from nearby frames"*), with the adjacency restriction added because the video is trial-gated and naive `diff` across segment boundaries would be meaningless.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles are taken over the **per-bin visible means** (`bin_y` over observed, finite bins) — i.e. over the same quantity that is being discretised, not over raw frames. Then: class 0 if `y < p40`, class 1 if `p40 ≤ y ≤ p60`, class 2 if `y > p60`, class 3 if the bin has no visible tongue frame. If fewer than 10 visible bins exist, the cut points are set to `inf` (everything visible becomes class 0). The resulting global distribution is 0.104 / 0.052 / 0.104 / 0.740, i.e. a 40/20/40 split of the ~26 % of bins in which the tongue is visible.

ii.
```python
TONGUE_PCTILES = (40.0, 60.0)
```
```python
# Percentiles of the tongue y position over the session, computed on the
# same quantity that is discretized (per-bin position of visible bins).
observed = np.zeros((len(trials), NBINS), dtype=bool)
for k, t in enumerate(trials):
    observed[k, bins[t]] = True
...
vis_vals = bin_y[observed & np.isfinite(bin_y)]
if len(vis_vals) >= 10:
    p40, p60 = np.percentile(vis_vals, TONGUE_PCTILES)
else:
    p40 = p60 = np.inf
tongue_class = np.full((len(trials), NBINS), 3, dtype=np.int8)
vis_bin = np.isfinite(bin_y)
tongue_class[vis_bin & (bin_y < p40)] = 0
tongue_class[vis_bin & (bin_y >= p40) & (bin_y <= p60)] = 1
tongue_class[vis_bin & (bin_y > p60)] = 2
```
```python
OUTPUT_VALUES = [
    ...
    ['<40th percentile', '40th-60th percentile', '>60th percentile',
     'not visible'],
]
```

iii. The 40th/60th cut points and the per-session scope come straight from the decoder spec. The AI's comment states the reason for taking percentiles over bin means rather than raw frames: it is the quantity actually being discretised, so the intended 40/20/40 balance is achieved exactly. The fourth class exists because "not visible" is a genuine state of the measurement (the tongue is retracted ~74 % of bins) rather than missing data, and imputing a position there would be fabrication.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as spikes and events, so each bin's frame range is found by `searchsorted` on the camera timestamps at `go + BIN_EDGES[:-1]` and `go + BIN_EDGES[1:]` — literally the same go-cue-relative edges used for the spike binning. The resulting `(n_trials, 80)` class array is then indexed by the same per-trial bin mask `b`, so row 3 of `output` is element-wise aligned with `neural`. Bins with no video at all are class 3; the session-level coverage QC (≥ 95 %, actual minimum 99.97 %) ensures this is rare.

ii.
```python
lo = np.searchsorted(ts, go[t] + BIN_EDGES[:-1])
hi = np.searchsorted(ts, go[t] + BIN_EDGES[1:])
has_video[k] = hi > lo
```
```python
outp[3] = tongue_class[k, b]
```
```python
# The video is recorded per trial as well, starting at the trial start; in
# the sessions kept here it outruns the spike interval (it ends >= 1.8 s after
# the go cue), so masking to the spike interval also removes essentially every
# bin without tracking (99.99% of the kept bins have video).
```

iii. One shared clock means alignment needs no interpolation or offset correction; using the identical edge array for both streams guarantees that bin *k* of the tongue output covers the same interval as bin *k* of the firing rates. The AI additionally checked that the video interval is a superset of the spike interval in the kept sessions, so the spike-based bin mask also removes almost all video-less bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Each kind of missingness is handled according to what it means:
- **Session never quality-controlled** (`classification`/`anno_name` are NaN): `_str_col` stringifies them to `'nan'`, no unit matches `'good'`, and the session is rejected (`'no good units'`).
- **Session with no tongue series**: `tongue_trace` returns `(None, None, None)` and the session is rejected (`'no tongue tracking'`).
- **Session with a broken/short video or failed tracking**: rejected by the coverage and lick-agreement QC.
- **Trials with no spikes** (ephys started late / stopped early): dropped.
- **Bins outside the recorded trial interval**: dropped from the trial rather than zero-filled.
- **Frames with the tongue retracted or a tracking outlier**: excluded from the bin mean (outliers imputed by interpolation); a bin with no visible frame becomes the explicit `'not visible'` class rather than being imputed.
- **Trials with no sample-epoch onset**: fall back to the session median tone-to-go offset (the AI noted none were found in this dataset).
- **Trials with no photostimulation** (`'N/A'`): the interval is `(nan, nan)` and the input row is explicitly set to 0.

ii.
```python
def _str_col(dataset):
    """Read an HDF5 column of (possibly byte) strings as a numpy array of str."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x)
                     for x in dataset[:]])
```
```python
if key not in f:
    return None, None, None
...
ts, tongue_y, tongue_vis = tongue_trace(f)
if ts is None:
    out['rejected'] = 'no tongue tracking'
    return out
```
```python
rel = tone - go
if np.any(np.isnan(rel)):
    rel[np.isnan(rel)] = np.nanmedian(rel)
```
```python
if np.isfinite(stim_on[t]):
    inp[1] = ((BIN_CENTERS[b] >= stim_on[t]) &
              (BIN_CENTERS[b] < stim_off[t])).astype(np.float32)
else:
    inp[1] = 0.0
```
```python
region_idx = np.array([region_map.get(a, len(REGION_NAMES) - 1)
                       for a in anno[good]], dtype=np.int64)
```

iii. The consistent rule is: where *nothing was recorded*, exclude the session/trial/bin rather than emit a fabricated value (a zero-filled trial would look like "all neurons silent", which the decoder would exploit); where the measurement *legitimately has no value* — a retracted tongue, an unstimulated trial — encode that state explicitly (class 3, or 0) rather than impute. Unmapped CCF annotations fall back to an `'Other'` region via `dict.get` instead of raising. The final run reports every exclusion count per session in `metadata['session_info']` and every rejected session with its reason in `metadata['rejected_sessions']`, so the curation is auditable.

## 10-a. What are the most time-consuming steps of the code?

i. The conversion is I/O-bound. Per session the dominant costs are (1) reading `units/spike_times` one ragged slice at a time (up to ~11 M doubles per session) and the ~700 k × 3 tongue tracking array from HDF5; (2) the per-unit `np.searchsorted` over the flattened edge array (one binary search per bin edge per unit); (3) the nested Python loop in the tongue binning, which does two `searchsorted` calls plus up to 80 inner iterations *per trial* — by far the most interpreted work in the file; and (4) pickling the 9.54 GB result at the end. The AI mitigated all of this with a 16-process pool and a per-session on-disk cache, giving ≤ 3 s per session wall-clock and a resumable run.

ii.
```python
with Pool(n_workers) as pool:
    caches = pool.map(_worker, [(p, region_map) for p in files], chunksize=1)
```
```python
def _worker(args):
    path = args[0]
    cache = os.path.join(CACHE_DIR, os.path.basename(path).replace('.nwb', '.pkl'))
    if os.path.exists(cache):
        return cache
```
```python
for i, u in enumerate(good):
    lo = 0 if u == 0 else spike_index[u - 1]
    st = units['spike_times'][lo:spike_index[u]]
    counts = np.diff(
        np.searchsorted(st, edges).reshape(len(trials), NBINS + 1), axis=1)
```

iii. The AI did not write an explicit rationale for its performance choices, but the design is self-evident from the code: sessions are fully independent, so they parallelise trivially, and the cache makes the expensive pass re-runnable (it was in fact reused when the video QC was added late in the session, step 155 → 182). The spike binning is already vectorised across trials by flattening the edge array, so the only remaining loop is over units, which is forced by the ragged storage.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four remain, in decreasing order of cost:
- **The tongue binning double loop** (`for k, t in enumerate(trials): for b in range(NBINS)`) — the clearest miss. Up to 80 interpreted iterations per trial, each slicing and masking a tiny array. This could be a single `np.bincount` over a global bin index (weights = y, masked to visible frames) plus a count, exactly as the human reference does with its `_bin_mean` helper.
- **`lick_choice`** — a Python loop over every trial doing two `searchsorted` calls; both could be done as vectorised `np.searchsorted(left, go)` / `np.searchsorted(right, go)` over the whole go-cue array at once.
- **`tone_onset_times`** — `for t, s in zip(trial_of[valid], sample[valid])` could be replaced by a `np.maximum.at`-style reduction or by the reference's single `searchsorted(sample, go) - 1`.
- **`photostim_intervals`** and the `bins = [mask_bins(tbl, t) for t in ...]` comprehension — per-trial loops that could be array expressions (the reference does the photostim conversion in three vectorised lines).
- The **per-unit spike loop** cannot be collapsed further: each unit has a different number of spikes, so there is no single sorted array to search. That one is inherent to the ragged storage.

ii.
```python
for k, t in enumerate(trials):
    lo = np.searchsorted(ts, go[t] + BIN_EDGES[:-1])
    hi = np.searchsorted(ts, go[t] + BIN_EDGES[1:])
    has_video[k] = hi > lo
    for b in range(NBINS):
        if hi[b] <= lo[b]:
            continue
        sl = slice(lo[b], hi[b])
        vis = tongue_vis[sl]
        if vis.any():
            bin_y[k, b] = tongue_y[sl][vis].mean()
```
```python
for i, g in enumerate(go):
    li = np.searchsorted(left, g)
    ri = np.searchsorted(right, g)
    ...
```
```python
for i in idx:
    onset = float(tbl['photostim_onset'][i])
    dur = float(tbl['photostim_duration'][i])
    on[i] = tbl['start_time'][i] + onset - tbl['go_time'][i]
    off[i] = on[i] + dur
```

iii. The AI gave no explicit justification for leaving these loops in. In practice the cost is hidden by the 16-way process parallelism (≤ 3 s per session end to end), and the loops are written in a straightforwardly readable per-trial form, so the trade-off is legibility over speed rather than an oversight with a runtime consequence.

## 10-c. What processing does the code repeat multiple times?

i. Little is recomputed, but there is some redundancy:
- **Firing rates are computed for all 80 bins and then sub-indexed** by the per-trial mask, so counts are produced for bins that are immediately discarded (~3 % of bins).
- **The per-trial bin mask is materialised twice**: once as the `bins` list of index arrays, and again as the boolean `observed` matrix, which is rebuilt by looping over trials a second time.
- **Per-trial behavioural quantities are computed for every trial in the session** (`lick_choice`, `outcome`, `early`, `tone_onset_times`, `photostim_intervals` all run over the full trials table) and only then indexed by the kept trials.
- **The Allen ontology `region_map`** is built once in the parent but is pickled and shipped to a worker with *every* task (174 times) because it is passed inside the `pool.map` argument tuple rather than inherited at fork time.
- Each NWB file is otherwise opened exactly once, and the bin grid is a module-level constant reused everywhere.

ii.
```python
rates = np.zeros((len(good), len(trials), NBINS), dtype=np.float32)
...
neural.append(np.ascontiguousarray(rates[:, k, b]))
```
```python
bins = [mask_bins(tbl, t) for t in range(len(tbl['start_time']))]
...
observed = np.zeros((len(trials), NBINS), dtype=bool)
for k, t in enumerate(trials):
    observed[k, bins[t]] = True
```
```python
caches = pool.map(_worker, [(p, region_map) for p in files], chunksize=1)
```

iii. No justification is given, and none of these is expensive relative to the HDF5 reads. Computing rates on the full grid first is arguably the right call anyway: `has_spikes` is derived from the full-window sum, so the un-masked array is genuinely needed before the mask is applied.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
- **Two filters that never fire.** `MIN_BINS_PER_TRIAL = 20` excluded 0 trials and the silent-unit filter dropped 0 units across the whole dataset, yet both are computed for every trial/unit of every session.
- **Session QC computed after the expensive work.** `tongue_lick_agreement` (precision/recall against the lick ports) and the video coverage check run *after* the spike binning and tongue binning for that session, so the 8 sessions rejected on those grounds had their full neural conversion computed and thrown away.
- **The five-sigma outlier imputation** has essentially no effect on the output (it only moves individual visible frames, which are then averaged into 50 ms bins and discretised into three coarse percentile classes), but costs a pass over every session's video.
- **Unused columns and metadata.** `photostim_power` and `auto_water` are read and, in the latter case, stored; `trial_metadata` (per-trial `trial_id`, `go_time`, `first_bin`, `n_bins`, `tone_onset_rel_go`, `photostim_on/off_rel_go`, `instruction`, `auto_water`) is written into the pickle for all 75,670 trials and is never touched by `train_decoder.py`. Likewise `bin_edges_relative_to_go_cue`, `rejected_sessions` and the 145-entry `session_info` block.
- **The Allen ontology download** builds a name→region map over the full structure graph (~1,300 structures) when only the annotations actually present in the data are needed.

ii.
```python
MIN_BINS_PER_TRIAL = 20
...
short = np.array([len(b) < MIN_BINS_PER_TRIAL for b in bins])
```
```python
alive = rates.any(axis=(1, 2))
out['n_units_silent'] = int((~alive).sum())
```
```python
precision, recall = tongue_lick_agreement(
    ts, tongue_vis, np.sort(np.concatenate([left_licks, right_licks])))
```
```python
trial_info={
    'trial_id': tbl['trial_id'][trials],
    'go_time': go[trials],
    'first_bin': np.array([bins[t][0] for t in trials], dtype=np.int16),
    ...
    'auto_water': tbl['auto_water'][trials].astype(np.int8),
},
```

iii. Most of this is deliberate provenance rather than waste: the AI notes that *"`trial_metadata` in the pickle records each trial's first bin index, tone onset and photostim window"*, which is what makes the variable-`T` decision (2-e) auditable and reversible, and the per-session QC numbers are what justify each rejection. The dead filters and the late placement of the session QC are genuine inefficiencies, but they cost a fraction of the I/O time and none of them change the converted values.
