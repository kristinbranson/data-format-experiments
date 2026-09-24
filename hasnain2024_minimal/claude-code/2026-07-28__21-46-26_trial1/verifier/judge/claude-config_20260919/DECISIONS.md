# Decisions

> **Note on the instruction version.** The prompt recorded in `/logs/agent/trajectory.json`
> (step 1) that the AI actually received specifies **two-class** outputs for lick direction
> (left/right), outcome (incorrect/correct) and the three kinematic variables
> (`0: < 50th pct`, `1: >= 50th pct`). The reference solution in `/tests/` was written
> against a **three-class** version (adding `none` / `ignore` / `not visible`). Where the
> AI's choice follows only from that difference it is noted explicitly below.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Sessions are **discovered by globbing** `data_structure_*.mat` in the two ephys folders
(`/app/data/Ephys_Behavior`, `/app/data/RandomizedDelay_Ephys_Behavior`); the two
behaviour-only inhibition folders are not searched. For each hit, a sibling
`motionEnergy_<anm>_<date>.mat` is paired with it if present. 47 data-structure files are
found; 2 (`JEB24_2023-10-03`, `JEB24_2023-10-04`) throw because they have no `clu` field
and are caught by a blanket `try/except` in the session loop, leaving **45 sessions**.
The files come in two MATLAB formats, so there are two full loaders: `_load_session_data_h5`
(h5py, v7.3) and `load_session_data_v5` (`scipy.io.loadmat`, v5/v7). `load_session_data`
opens the file with h5py first purely as a format probe, then re-opens it in the real
reader or falls through to scipy. Each loader pulls a fixed set of fields into a flat
python dict: `Ntrials`, `R`, `L`, `hit`, `miss`, `early`, `autowater`, `stim.enable`, `no`,
`ev.goCue`, `ev.sample`, meta `anm`/`day`, per-cluster `quality`/`trialtm`/`trial`,
`meta.probe.loc`, the two cameras' `featNames`/`ts`/`frameTimes`/`NdroppedFrames`, and the
bitcode fields used for the video offset. The v5 loader transposes `ts` from
`(n_frames, 3, n_features)` to the HDF5 layout `(n_features, 3, n_frames)` so downstream
code sees one shape.

ii.
```python
def find_session_files(data_dirs):
    """Find all data_structure and motionEnergy file pairs."""
    sessions = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        data_files = sorted(glob.glob(os.path.join(data_dir, 'data_structure_*.mat')))
        for df in data_files:
            basename = os.path.basename(df)
            parts = basename.replace('data_structure_', '').replace('.mat', '')
            me_file = os.path.join(data_dir, f'motionEnergy_{parts}.mat')
            if not os.path.exists(me_file):
                me_file = None
            sessions.append({'data_file': df, 'me_file': me_file, 'animal_date': parts})
    return sessions
```

```python
def load_session_data(filepath):
    """Tries HDF5 (v7.3) first, falls back to scipy.io (v5/v7)."""
    try:
        with h5py.File(filepath, 'r') as f:
            if 'obj' not in f:
                raise ValueError("No obj field")
            obj = f['obj']
            if 'clu' not in obj:
                raise ValueError("No clu field - behavior-only session")
        return _load_session_data_h5(filepath)
    except (OSError, ValueError) as e:
        pass
    return load_session_data_v5(filepath)
```

```python
for idx, sf in enumerate(session_files):
    try:
        result = process_session(sf['data_file'], sf['me_file'], idx)
    except Exception as e:
        print(f"\n  ERROR processing session {idx} ({sf['animal_date']}): {e}")
        ...
        continue
```

iii. The AI's stated reasoning (step 21, step 49, step 77): the decoder needs both DR and
WC contexts, "I'll include all Ephys_Behavior sessions"; later it added the RandomizedDelay
folder too. It discovered empirically that "several files are MATLAB v5/v7 format (not
HDF5)" and that "JEB24_2023-10-03 and 04 don't have 'clu' (behavior-only sessions, no
electrophysiology)", so it wrote the second reader and relied on the `try/except` for the
behaviour-only files. It never opened the authors' `load<ANM>_ALMVideo.m` session lists.

## 1-b. How are the data split into subjects?

i. The animal id is parsed out of the **filename**, not read from inside the file: everything
before the first underscore after stripping `data_structure_`. `subjects` is then the sorted
unique set and `subject_idx` each session's index into it. This yields 14 subjects over
45 sessions (EKH1, EKH3, JEB6, JEB7, JEB11, JEB12, JEB13, JEB14, JEB15, JEB19, JEB23, JEB24,
JGR2, JGR3) — the same 14 animals as the reference.

ii.
```python
basename = os.path.basename(session_file)
parts = basename.replace('data_structure_', '').replace('.mat', '').split('_')
animal = parts[0]
date = '_'.join(parts[1:])
```

```python
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals], dtype=np.int64)
```

iii. Not explicitly justified in the trajectory. The loaders do also read `meta.anm` /
`ex.anm` into `session['animal']`, but that value is discarded and the filename-derived one
is used — which is the robust choice, since `meta.anm` is absent from several sessions
(the loaders fall back to `'unknown'` there).

## 1-c. How are the data split into sessions?

i. One session = one `data_structure_*.mat` file = one element of `neural`/`input`/`output`.
Both task folders are pooled into a single flat session list (Ephys_Behavior first, then
RandomizedDelay), so fixed-delay and randomized-delay sessions are treated uniformly.
No session-level inclusion list is applied: every file that loads and yields >= 2 valid
trials and >= 1 unit is kept. Result: **45 sessions** (25 fixed-delay + 20 randomized-delay),
one more than the reference's 44 — the extra one is `JEB23_2023-10-20`, which the authors'
loading scripts exclude.

ii.
```python
DATA_DIRS = [
    '/app/data/Ephys_Behavior',
    '/app/data/RandomizedDelay_Ephys_Behavior',
]
...
    if n_valid < 2:
        print(f"  SKIPPING: fewer than 2 valid trials")
        return None
```

iii. The AI ran a sanity check against the paper (step 105) and saw the mismatch:
"**RandomizedDelay**: 20 sessions (paper says 19) ... likely one session didn't meet the
inclusion criteria, though all my loaded sessions have at least 17 units ... The minor
discrepancies could stem from different quality filtering or other methodological choices,
so I'll move forward with what I have." It also recorded the mismatch in
`CONVERSION_NOTES.md` ("RandomizedDelay: 1 extra session compared to paper; may be
borderline session") without resolving it.

## 1-d. How are the data split into trials?

i. Trials are the rows of the Bpod table: every per-trial field is read as a length-`Ntrials`
vector and the trial index is the position in that vector. Spikes carry their own 1-based
`trial` label, so a trial's spikes are selected by `spike_trials == trial_idx + 1`. Camera
trials and motion-energy trials are indexed by the same integer (`cam['trials'][trial_idx]`,
`me_data[trial_idx]`), with bounds checks that silently skip a trial if the video/ME arrays
are shorter than the Bpod table.

ii.
```python
Ntrials = int(h5_read_scalar(bp['Ntrials']))
session['Ntrials'] = Ntrials
session['R'] = bp['R'][()].flatten().astype(float)  # (Ntrials,)
...
for ti, trial_idx in enumerate(valid_trials):
    trial_num = trial_idx + 1  # MATLAB 1-indexed
    go_time = go_cue_times[trial_idx]
    ...
    spike_mask = spike_trials == trial_num
```

```python
    for ti, trial_idx in enumerate(valid_trials):
        if trial_idx >= len(cam['trials']):
            continue
```

iii. Not discussed explicitly; the trial structure is taken directly from the data, which is
unambiguous. Note the per-trial vectors are **not** truncated to `Ntrials` (the reference
does `[:n_trials]` because a few fields are stored longer); the AI indexes them by trial
number only, so extra tail entries are simply never reached.

## 1-e. How are trials filtered based on quality controls?

i. A single boolean mask combining three conditions: no photostimulation
(`stim.enable == 0`), no early lick (`early == 0`), and the animal responded
(`hit == 1 | miss == 1`). The third condition **drops ignore/no-response trials entirely**
rather than giving them their own class. Sessions with fewer than 2 surviving trials are
dropped. There is **no filter for trials that run past the end of the recording**: the AI's
own verification found 30 trials across `JEB24_2023-10-23` and `JEB24_2023-10-31` whose
neural data is identically zero, and they were left in the dataset. Total: **12,293 trials**
(reference: 13,762).

ii.
```python
    # Include: non-stim, non-early, responding (hit or miss) trials
    # Matching: ~stim.enable & ~early & (hit | miss)
    trial_mask = (
        (session['stim_enable'] == 0) &
        (session['early'] == 0) &
        ((session['hit'] == 1) | (session['miss'] == 1))
    )
    valid_trials = np.where(trial_mask)[0]
    n_valid = len(valid_trials)
    if n_valid < 2:
        print(f"  SKIPPING: fewer than 2 valid trials")
        return None
```

iii. From `CONVERSION_NOTES.md`: "Matching reference code conditions: Excluded trials with
optogenetic stimulation (`~stim.enable`); Excluded early lick trials (`~early`); Included
only responding trials (`hit | miss`), excluding ignore/no-response trials. This includes
both correct (hit) and incorrect (miss) trials for outcome decoding." In step 21 the AI
noted that the authors' `getDefaultParams.m` conditions select hit trials only and that it
would widen this to hits and misses so that outcome is decodable. The all-zero-neural trials
were noticed at step 93 ("this looks like recording quality degradation") and written up as
Known Issue 4, but no filter was added.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}`, specifically each cluster's `quality` (curation label), `trialtm`
(spike time relative to that trial's start, on the behaviour clock) and `trial` (1-based
trial of each spike). `obj.bp.ev.goCue` supplies the alignment time and
`obj.meta.probe.loc` (or `obj.ex.probe.loc` in v5) supplies the anatomical label used both
to select probes and to fill `brain_region_idx`.

ii.
```python
                    q_ref = probe_group['quality'][unit_idx, 0]
                    unit['quality'] = h5_read_string(f, f[q_ref])
                    # Spike times (within trial, relative to trial start)
                    tm_ref = probe_group['trialtm'][unit_idx, 0]
                    unit['trialtm'] = f[tm_ref][()].flatten()
                    # Trial assignment for each spike
                    trial_ref = probe_group['trial'][unit_idx, 0]
                    unit['trial'] = f[trial_ref][()].flatten().astype(int)
```

```python
        session['probe_locations'] = []
        if 'meta' in obj and isinstance(obj['meta'], h5py.Group) and 'probe' in obj['meta']:
            probe_meta = obj['meta']['probe']
            if 'loc' in probe_meta:
                loc_data = probe_meta['loc'][()].flatten()
```

iii. Taken from the authors' `loadSessionData.m` / `getSeq.m` pipeline, which the AI had a
sub-agent read in full at step 14. No alternative source of spikes exists in the files.

## 2-b. How is the `neural` data processed?

i. Per unit and per trial: spike times are shifted by that trial's go cue, histogrammed into
the 1000 fixed 5 ms bins, divided by the bin width to give Hz, and then smoothed along time
with a **causal** Gaussian kernel of 15 samples (75 ms) reproduced from the authors'
`mySmooth.m` — a `gausswin(15)` whose first `floor(N/2)` taps are zeroed and which is then
renormalised, convolved with `mode='same'`. No baseline subtraction, normalisation or
z-scoring. Units from all ALM probes of a session are concatenated along the unit axis.
Values are stored as `float32` and transposed to (n_neurons, n_timepoints) per trial.
The AI's kernel uses `sigma = N/6 = 2.5` samples where MATLAB's `gausswin(N)` with the
default `alpha = 2.5` gives `sigma = (N-1)/(2*alpha) = 2.8` samples — a small mismatch.

ii.
```python
def causal_gaussian_smooth(x, kernel_width_samples):
    """Causal Gaussian smoothing matching mySmooth.m."""
    N = kernel_width_samples
    n = np.arange(N)
    center = (N - 1) / 2
    sigma = N / 6  # MATLAB gausswin default: alpha=2.5, sigma = (N-1)/(2*alpha)
    kern = np.exp(-0.5 * ((n - center) / sigma) ** 2)
    kern[:N // 2] = 0          # Make causal: zero out left half
    kern = kern / kern.sum()
    ...
```

```python
            aligned_times = spike_times[spike_mask] - go_time
            counts, _ = np.histogram(aligned_times, bins=edges)
            counts = counts[:n_timebins].astype(float)
            fr = counts / dt
            fr_smooth = causal_gaussian_smooth(fr, smooth_samples)
            trialdat[:, ui, ti] = fr_smooth
```

```python
    neural_data = np.concatenate(all_clusters, axis=1)   # probes concatenated
...
        neural_trial = neural_data[:, :, ti].T.astype(np.float32)
```

iii. `CONVERSION_NOTES.md`: "**Smoothing**: Causal Gaussian kernel with width N=15 samples
(matching `mySmooth.m`) — Half of kernel zeroed for causality — Gaussian sigma = N/6 (MATLAB
gausswin default with alpha=2.5)". The AI read `mySmooth.m` verbatim at step 25 and
reproduced its causal-kernel construction rather than using a symmetric filter.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. **(1) Probe location**: a probe is used only if its `meta.probe.loc`
string contains `ALM`, so e.g. the `R_brainstem` probe of `EKH1_2021-08-07` is dropped.
This replaces the authors' per-session hand-picked probe list — every ALM probe present is
used. **(2) Cluster quality**: the label is stripped and lower-cased and excluded if it is
one of `garbage`, `gabrga`, `noisy`, `real?`; unlabelled clusters are kept
(`'poor'` is **not** excluded, unlike the reference). **(3) Firing rate**: a unit is kept if
its mean smoothed rate over all bins and all kept trials exceeds **0.5 Hz**
(`removeLowFRClusters.m`'s default) rather than the 1 Hz quoted in the paper. Result:
**2,579 units** over 45 sessions, 17–136 per session (reference: 1,954 units, 15–110).

ii.
```python
def filter_clusters(clusters, quality_filter='all'):
    """Filter clusters by quality, matching findClusters.m."""
    excluded = {'garbage', 'gabrga', 'noisy', 'real?'}
    indices = []
    for i, unit in enumerate(clusters):
        q = unit.get('quality', '').strip().lower()
        if q not in excluded and q != '':
            indices.append(i)
        elif q == '':
            # Include unlabeled units (matching MATLAB 'all' behavior ...)
            indices.append(i)
    return indices
```

```python
LOW_FR = 0.5  # minimum firing rate threshold (Hz)
...
        loc_upper = loc.strip().upper()
        if 'ALM' not in loc_upper:
            print(f"  Probe {probe_idx} ({loc}): skipping non-ALM probe")
            continue
        valid_units = filter_clusters(clusters)
...
        fr_mask = mean_frs > LOW_FR
        kept_units = [valid_units[i] for i in range(len(valid_units)) if fr_mask[i]]
        trialdat_probe = trialdat_probe[:, fr_mask, :]
```

iii. `CONVERSION_NOTES.md`: "Quality filter: excluded 'garbage', 'noisy' units (matching
`findClusters.m` with quality='all'); Firing rate filter: removed units with mean FR < 0.5 Hz
(matching `removeLowFRClusters.m`, default `lowFR=0.5`); Probe filtering: Only ALM probes
included (paper: 'We recorded activity extracellularly in the ALM')". At step 33 the AI
noticed the brainstem probe and added the location filter. At step 105 it compared unit
counts with the paper (1,586 vs 1,651 for the fixed-delay set; 993 vs 845 for randomized
delay) and attributed the difference to "different firing rate thresholds — the paper may
use 1 Hz while I'm using 0.5 Hz — and possibly different quality filtering criteria",
choosing to keep the code's default.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction per trial. `clu.trialtm` is already relative to the start of its own trial
on the behaviour clock and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]`
gives seconds from the go cue with no interpolation or clock correction. Spikes outside
[−2.5, 2.5] fall outside the histogram edges and are discarded. Trials whose go cue is NaN
or exactly 0 are skipped, leaving that trial's row at zero.

ii.
```python
            go_time = go_cue_times[trial_idx]
            if np.isnan(go_time) or go_time == 0:
                continue
            spike_mask = spike_trials == trial_num
            if not np.any(spike_mask):
                continue
            # Align to go cue
            aligned_times = spike_times[spike_mask] - go_time
            counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. `ALIGN_EVENT = 'goCue'` is set as a module constant "matching getDefaultParams.m"; the
authors' `alignSpikes.m` performs the identical subtraction. The instructions also require
go-cue alignment, so there is no tension here.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **5 ms** bins (`DT = 1/200`), 1000 non-overlapping bins spanning −2.5 to +2.5 s from the
go cue. The grid is built once at module scope and is identical for every trial, session and
data stream — spikes are histogrammed into it and the camera streams are interpolated onto
its bin centres. No rebinning, downsampling or resampling to a coarser resolution is applied
anywhere, and `metadata['time_bin_size']` is reported as 5.0 ms.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 200  # 5 ms bins
# Time axis (center of bins)
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_AXIS)
```

```python
            'time_bin_size': DT * 1000,  # in ms
            'off_start': TMIN,  # -2.5 s before go cue
            'off_end': TMAX,  # +2.5 s after go cue
```

iii. "Parameters matching getDefaultParams.m" — `params.dt = 1/200` and
`params.tmin/tmax = ∓2.5` in the authors' code. Confirmed by the verification output:
T = 1000 for every trial of every session.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable. The input is the analysis time axis itself — the centres of the 1000
bins of the go-cue-aligned window defined in 2-e. It is identical for every trial and every
session, so it encodes "how far are we from the go cue" and nothing about the particular
trial. `input_names` is `['time_from_go_cue']`, giving `n_input = 1`.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_AXIS)
```

```python
        # Input: time from go cue (1, n_timebins)
        input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
        input_trials.append(input_trial)
```

iii. `CONVERSION_NOTES.md`: "Time from go cue onset (seconds): continuous time axis
[-2.4975, ..., 2.4975] in 5ms steps". The instructions name this input directly and specify
it as continuous and time-varying, so no derivation from the data is required.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. `TIME_AXIS` is computed once at import and the same `(1, 1000)` `float32` array
(a fresh reshape of the same buffer) is appended for every trial. Since the go cue is by
construction at t = 0, the value at bin *k* is `−2.5 + 0.005*k + 0.0025`. It is left
continuous rather than converted to a binary event indicator.

ii.
```python
        input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
        input_trials.append(input_trial)
```

iii. No separate justification given; the instructions specify the input as continuous.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. By construction — it *is* the neural binning grid. `EDGES` are the histogram edges used
for the spikes and `TIME_AXIS` is the centres of those same edges, so bin *k* of the input
covers exactly the same interval as bin *k* of the neural matrix, and both are the
interpolation target for the camera streams. Verified in the output: input range is exactly
[−2.5, 2.5] for all 45 sessions.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
...
            counts, _ = np.histogram(aligned_times, bins=edges)     # neural uses EDGES
...
            velocities[:, ti] = interp_fn(TIME_AXIS)                # camera uses centres
```

iii. N/A — a single shared grid was chosen so that no cross-stream alignment step is needed.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. A single field, `obj.bp.R`, taken as-is: `R == 1 → 1 (right)`, `R == 0 → 0 (left)`.
`obj.bp.L` is loaded but never used, and `hit`/`miss` are **not** consulted. `bp.R` is the
*instructed* trial type (which port was rewarded), not the port the animal actually licked,
so on the miss trials that the pipeline deliberately retains (13.7% of all kept trials) the
label is the opposite of the direction the animal licked. Because ignore trials were dropped
in 1-e, there is no third "no lick" class.

ii.
```python
    lick_direction = session['R'][valid_trials].astype(np.int32)  # R=1, L=0
```

```python
    print(f"  Lick direction: R={np.sum(lick_direction==1)}, L={np.sum(lick_direction==0)}")
```

iii. `CONVERSION_NOTES.md`: "**Lick direction**: R=1 (right), L=0 (left) from `obj.bp.R`
field". No reasoning in the trajectory addresses whether `bp.R` encodes the instructed side
or the executed lick; at step 21 the AI wrote only "I also need to figure out how lick
direction is encoded in the data" and then used `R` directly without revisiting it.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A cast, a per-trial subset, and a tile across time. `session['R']` is subset to the kept
trials and cast to `int32`; the per-trial scalar is then broadcast across all 1000 bins of
output channel 0 so that all six outputs share one `(6, 1000)` array. No combination with
outcome, no third class, no use of the actual lick-port event times available in `bp`.
The resulting distribution is 49.9% left / 50.1% right.

ii.
```python
    lick_direction = session['R'][valid_trials].astype(np.int32)  # R=1, L=0
...
        output_trial = np.zeros((6, N_TIMEBINS), dtype=np.int32)
        output_trial[0, :] = lick_direction[ti]  # per-trial
```

```python
        'output_values': [
            ['left', 'right'],
            ...
```

iii. The AI's prompt defined lick direction as a two-valued per-trial variable
(`left = 0, right = 1`), so it treated it as a direct relabelling of the trial-type flag.
No justification is offered for equating trial type with lick direction on error trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial field, `obj.bp.autowater`. A trial with `autowater == 1` is a water-cued
(WC) trial — water is delivered from a random port with no sample tone, delay or go cue —
and everything else is a delayed-response (DR) trial.

ii.
```python
        session['autowater'] = bp['autowater'][()].flatten().astype(float)
```

iii. From step 16: "the `autowater` field means WC context (autowater=1 means WC)". The AI
confirmed this against the paper's description of the two-context task before using it.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A one-line inversion of the flag — `1 - autowater` — subset to the kept trials, cast to
`int32`, and tiled across the 1000 bins, giving WC = 0, DR = 1 as the instructions specify.
`output_values[1]` is `['WC', 'DR']`. The dataset comes out 8.2% WC / 91.8% DR overall, and
16 of the 45 sessions are pure DR.

ii.
```python
    context = (1 - session['autowater'][valid_trials]).astype(np.int32)  # DR=1, WC=0
...
        output_trial[1, :] = context[ti]  # per-trial
```

iii. `CONVERSION_NOTES.md`: "**Behavioral context**: DR=1, WC=0 from `obj.bp.autowater`
(autowater=1 means WC)". The coding direction is dictated by the instructions. At step 21
the AI explicitly decided to keep DR-only sessions rather than restrict to the 12
two-context sessions: "even though DR-only sessions will always have context=1, they still
provide useful neural data".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. One flag, `obj.bp.hit`. `obj.bp.miss` is used only in the trial mask and `obj.bp.no`
(the ignore flag) is loaded but never used, because ignore trials were already removed in
1-e — within the retained set `hit == 0` is equivalent to `miss == 1`.

ii.
```python
        session['hit'] = bp['hit'][()].flatten().astype(float)
        session['miss'] = bp['miss'][()].flatten().astype(float)
        session['no'] = bp['no'][()].flatten().astype(float)   # loaded, never used
```

iii. Follows directly from the trial-filtering decision in 1-e: because only responding
trials survive, the single `hit` flag fully determines outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A cast and a tile: `hit` is subset to the kept trials, cast to `int32` (correct = 1,
incorrect = 0) and broadcast across the 1000 bins of channel 2. `output_values[2]` is
`['incorrect', 'correct']`. There is **no** third `ignore` class — those trials were dropped
rather than labelled. The dataset is 86.3% correct / 13.7% incorrect.

ii.
```python
    outcome = session['hit'][valid_trials].astype(np.int32)  # correct=1, incorrect=0
...
        output_trial[2, :] = outcome[ti]  # per-trial
```

iii. `CONVERSION_NOTES.md`: "**Outcome**: correct=1, incorrect=0 from `obj.bp.hit`" and
"Included only responding trials (`hit | miss`), excluding ignore/no-response trials. This
includes both correct (hit) and incorrect (miss) trials for outcome decoding." The AI's
prompt listed outcome as a two-valued variable, so it widened the authors' hit-only trial
conditions to hits plus misses and stopped there.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, **side camera only** (`cam_idx = 0`), feature
named exactly `tongue` (matched case-insensitively against `featNames`). From that entry the
x and y channels of `ts` are used; the third channel (likelihood) is read into the array but
never thresholded. `frameTimes` of the same camera, the session video offset from
`obj.sglx.bitcode.bitstart` / `obj.sglx.fs` / `obj.bp.ev.bitStart`, and `bp.ev.goCue` supply
the time base. The bottom camera's `top_tongue` is not used.

ii.
```python
    # Tongue velocity from side camera (cam 0), feature 'tongue'
    tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)
```

```python
    cam = session['traj'][cam_idx]
    feat_names = cam['feat_names']
    feat_idx = None
    for i, fn in enumerate(feat_names):
        if fn.lower() == feat_name.lower():
            feat_idx = i
            break
```

```python
        x_pos = ts[feat_idx, 0, :].copy()
        y_pos = ts[feat_idx, 1, :].copy()
```

iii. `CONVERSION_NOTES.md`: "**Tongue velocity**: Extracted from side camera (cam 0),
'tongue' feature". At step 37 the AI inspected the feature lists of both cameras and noted
"For tongue feature (index 0), x and y are NaN most of the time (tongue only visible during
licks)", but chose the side view alone and never combined the two views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps. **(1)** x and y are taken raw and **not smoothed** (the authors'
`findPosition.m` also skips smoothing for the tongue). **(2)** Speed is
`sqrt(dx² + dy²)` from `np.gradient` using a **hard-coded** frame interval of 1/400 s
rather than the real `frameTimes` spacing. **(3)** Every frame where the tongue is invisible
(x or y NaN) has its velocity **set to 0**, and any remaining NaN velocity is also set to 0.
**(4)** The frame-resolution speed is linearly interpolated (`interp1d`, `fill_value=np.nan`)
onto the 1000 bin centres, then any bin still NaN is filled by linear interpolation from the
neighbouring valid bins, or set to 0 if the whole trial is NaN. No per-view normalisation
is needed since only one camera is used.

ii.
```python
        dt_vid = 1 / 400
        dx = np.gradient(x_pos, dt_vid)
        dy = np.gradient(y_pos, dt_vid)
        vel = np.sqrt(dx**2 + dy**2)

        # For tongue, set velocity to 0 where tongue not visible
        if is_tongue:
            nan_mask = np.isnan(x_pos) | np.isnan(y_pos)
            vel[nan_mask] = 0.0
            vel[np.isnan(vel)] = 0.0
```

```python
        interp_fn = interp1d(aligned_frame_times, vel,
                            kind='linear', bounds_error=False, fill_value=np.nan)
        velocities[:, ti] = interp_fn(TIME_AXIS)
...
    for ti in range(n_trials):
        col = velocities[:, ti]
        mask = np.isnan(col)
        if mask.any() and not mask.all():
            valid_idx = np.where(~mask)[0]
            col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
            velocities[:, ti] = col
        elif mask.all():
            velocities[:, ti] = 0.0
```

iii. `CONVERSION_NOTES.md`: "Velocity = sqrt(dx^2 + dy^2) computed via np.gradient at 400 Hz;
Set to 0 when tongue not visible (NaN positions), matching paper: 'Missing values were filled
in with the nearest available value for all features, except for the tongue'". (The quoted
sentence actually says the tongue is the feature that is *not* filled; the AI used it to
justify substituting zeros.)

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two classes, split at the **50th percentile of all finite values pooled over the whole
session** (all trials × all bins): `>= threshold → 1`, `< threshold → 0`. There is no third
"not visible" class. Because step 7-b replaced every invisible-tongue frame with a velocity
of exactly 0 and the tongue is out of view in roughly 80–90% of bins, the session median
**is 0** in every session, and the test `values >= 0` is true everywhere. The result is that
`tongue_velocity` is the constant 1 in **all 45 sessions** — the verification output reports
`tongue_velocity: {high (1.000)}` and a range of [1.0, 1.0] for every session. The channel
carries no information, and the 0.995 decoder "balanced accuracy" it produces is an artefact
of a single-class label.

ii.
```python
def discretize_per_session(values, percentile=50):
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.zeros_like(values, dtype=np.int32)
    threshold = np.percentile(valid, percentile)
    discretized = (values >= threshold).astype(np.int32)
    return discretized
```

```python
    if tongue_vel is not None:
        tongue_vel_disc = discretize_per_session(tongue_vel)
    else:
        print(f"  Warning: No tongue velocity data, using zeros")
        tongue_vel_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)
```

iii. The AI diagnosed this exactly (step 52): "if most time points have zero velocity, the
50th percentile will be zero, making the discretization useless"; it considered computing
the percentile over non-zero values only, or flipping to `> 0`, then concluded (step 53–55):
"The discretization is correct per spec. The tongue velocity being mostly 'high' is expected
— when the tongue is not visible, velocity = 0, and median = 0, so everything >= 0 gets class
1. This is a property of the sparse tongue data." It shipped the degenerate channel and
recorded it as Known Issue 1: "This is correct per the discretization specification but
results in a degenerate output variable."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are put on the go-cue clock by `frameTimes − vidshift − goCue[trial]`, where
`vidshift` is one constant per session computed from the bitcode that both the video and
behaviour clocks record. The AI computes it as
`nanmedian(sglx.bitcode.bitstart)/sglx.fs − nanmedian(bp.ev.bitStart)`, i.e. with
**`nanmedian` where the authors' `findVideoOffset.m` (and the AI's own notes) use `mode`**.
If the whole computation throws, `vidshift` silently defaults to 0.0. The corrected frame
times are then the x-coordinate of a linear `interp1d` evaluated at the shared 1000 bin
centres, which is the same grid the spikes were histogrammed into. If a trial's `frameTimes`
are all NaN, a synthetic axis `arange(1, n+1)/400` is substituted and still shifted by
`vidshift` and the go cue.

ii.
```python
            bitStart_bp = np.nanmedian(ev['bitStart'][()].flatten())
            sglx = obj['sglx']
            fs = h5_read_scalar(sglx['fs'])
            bitstart_sglx = np.nanmedian(sglx['bitcode']['bitstart'][()].flatten())
            session['vidshift'] = bitstart_sglx / fs - bitStart_bp
        except:
            session['vidshift'] = 0.0
```

```python
        if frame_times is None or np.all(np.isnan(frame_times)):
            frame_times = np.arange(1, n_frames + 1) / 400.0
        # Align to go cue and interpolate to neural time axis
        aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
        interp_fn = interp1d(aligned_frame_times, vel, kind='linear',
                             bounds_error=False, fill_value=np.nan)
        velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. `CONVERSION_NOTES.md`: "Video offset computed as `mode(sglx.bitcode.bitstart)/fs -
mode(bp.ev.bitStart)` (matching `findVideoOffset.m`)". The AI read `findVideoOffset.m`
verbatim at step 26 and `findPosition.m` at step 27 — the latter's
`interp1(traj.frameTimes - vidshift - obj.bp.ev.(alignEv)(trix), ts, taxis)` is exactly the
alignment the AI reproduced, except that it interpolates the *velocity* rather than the
position and uses the median rather than the mode of the bitcode.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` DeepLabCut tracking, **bottom camera only** (`cam_idx = 1`), and
**both** paw features: `top_paw` and `bottom_paw`. Their x/y channels, the bottom camera's
`frameTimes`, the session `vidshift` and `bp.ev.goCue` are the inputs. Likelihood is again
ignored.

ii.
```python
    # Paw velocity from bottom camera (cam 1)
    # Average top_paw and bottom_paw velocities when both available
    paw_vel = None
    if len(session['traj']) > 1:
        paw_top = extract_kinematic_feature(session, 1, 'top_paw', valid_trials, go_cue_times)
        paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', valid_trials, go_cue_times)

        if paw_top is not None and paw_bot is not None:
            paw_vel = (paw_top + paw_bot) / 2.0
        elif paw_top is not None:
            paw_vel = paw_top
        elif paw_bot is not None:
            paw_vel = paw_bot
```

iii. `CONVERSION_NOTES.md`: "**Paw velocity**: Extracted from bottom camera (cam 1),
averaged 'top_paw' and 'bottom_paw'". The two features are two forepaws in a single camera
view, so they are on a common pixel scale and the AI averaged them without normalisation.
It did not check whether either feature is reliably tracked through the delay epoch.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same `extract_kinematic_feature` path as the tongue, with one branch different:
because `is_tongue` is False, **missing x/y are filled in first** by linear interpolation
over frame index (the AI's stand-in for MATLAB's `fillmissing(...,'nearest')`), then
`np.gradient` at a hard-coded 1/400 s gives the speed, which is interpolated onto the 1000
bin centres and NaN-filled again at the bin level. The two paws' binned velocities are then
averaged elementwise. No smoothing of position is applied, although `findPosition.m` calls
`mySmooth(ts, 1, 'reflect')` for non-tongue features (which with N = 1 is a no-op, so this
matches). No cross-view normalisation.
**The extraction silently fails for the MATLAB v5/v7 sessions**: in 11 of the 45 sessions
(`JEB23_2023-10-18`, `JEB23_2023-10-20`, `JEB23_2023-10-21` and all 8 `JEB24` sessions) the
returned array is entirely zeros, which after thresholding becomes a constant class.

ii.
```python
        # Fill missing values for non-tongue features (matching MATLAB fillmissing nearest)
        if not is_tongue:
            for pos in [x_pos, y_pos]:
                mask = np.isnan(pos)
                if mask.any() and not mask.all():
                    valid_idx = np.where(~mask)[0]
                    pos[mask] = np.interp(np.where(mask)[0], valid_idx, pos[valid_idx])
```

```python
        if paw_top is not None and paw_bot is not None:
            paw_vel = (paw_top + paw_bot) / 2.0
```

iii. `CONVERSION_NOTES.md`: "Missing values filled with nearest valid value (non-tongue
features); Interpolated to neural time axis", mirroring `findPosition.m`. The failure is
acknowledged only as Known Issue 2: "**Paw velocity for some v5/v7 sessions**: Some
RandomizedDelay sessions loaded via scipy.io have trajectory data in different format,
causing all-1 paw velocity in some sessions." It was not fixed.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: one 50th-percentile threshold per session over all finite
values pooled across trials and bins, `>= → 1`, `< → 0`, no "not visible" class. Where the
signal is real this gives the intended balanced split (exactly 0.500/0.500 in 34 sessions).
In the 11 broken sessions of 8-b the input is all zeros, so the threshold is 0, every bin
satisfies `>= 0`, and the channel is the constant 1. Over the whole dataset the split is
35.7% low / 64.3% high rather than the ~50/50 the specification implies.

ii.
```python
    if paw_vel is not None:
        paw_vel_disc = discretize_per_session(paw_vel)
    else:
        print(f"  Warning: No paw velocity data, using zeros")
        paw_vel_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)
```

```python
    threshold = np.percentile(valid, percentile)
    discretized = (values >= threshold).astype(np.int32)
```

iii. "Tongue/paw velocity and motion energy: Discretized into 2 bins using per-session 50th
percentile threshold" — a direct reading of the instruction. The AI saw the degenerate
sessions in its own verification output (step 93: "Some sessions (33+) have all one class.
These appear to be the RandomizedDelay sessions where the paw data from the v5/v7 files
might not be loading correctly") and left them in.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as for the tongue (7-d), but using the **bottom** camera's own `frameTimes`
(`session['traj'][1]`), since that is the camera the feature is read from. The same session
`vidshift` and the trial's `goCue` are subtracted and the result is the interpolation
x-axis for `interp1d` onto the shared 1000 bin centres. Trials whose `NdroppedFrames` is NaN
are skipped, reproducing `findPosition.m`'s guard.

ii.
```python
        ndf = trial_data.get('NdroppedFrames', 0)
        if isinstance(ndf, float) and np.isnan(ndf):
            continue
...
        aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
        interp_fn = interp1d(aligned_frame_times, vel, kind='linear',
                             bounds_error=False, fill_value=np.nan)
        velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. Same as 7-d: this is the alignment of `findPosition.m`, which the AI read at step 27,
applied per-camera so the paw uses the view that actually tracks it.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside each data structure, read as
`me.data` (one vector per trial, one value per side-camera frame) with `me.moveThresh` also
read but never used. The side camera's `frameTimes` from `obj.traj[0]`, the session
`vidshift` and `bp.ev.goCue` supply the time base. The copy that some sessions carry in
`obj.me` is not used. The loader handles only the `{data, moveThresh}` layout (via scipy)
and an HDF5 layout; the other layouts present in this dataset raise, and **7 sessions
(`JEB15_2022-07-26`, `JEB15_2022-07-28`, `JEB23_2023-10-10/11/12/13` and
`JEB24_2023-10-31`) end up with no motion energy at all** (the AI's notes say 8).

ii.
```python
def load_motion_energy(filepath):
    try:
        mat = scipy.io.loadmat(filepath, squeeze_me=False)
        me = mat['me']
        data_field = me['data'][0, 0]
        thresh_field = me['moveThresh'][0, 0]
        ...
        return {'data': me_data, 'moveThresh': threshold}
    except NotImplementedError:
        pass
    except Exception as e:
        pass
    try:
        with h5py.File(filepath, 'r') as f:
            ...
    except Exception as e:
        print(f"  Warning: Could not load motion energy from {filepath}: {e}")
        return None
```

iii. `CONVERSION_NOTES.md`: "**Motion energy**: From `motionEnergy_*.mat` files". The
failures are recorded as Known Issue 3: "8 sessions lack loadable motion energy files
(corrupted or incompatible format). These sessions have motion energy set to all zeros,
discretized as all-low." The files are not corrupt — they use a third wrapper layout that
the two-branch loader does not cover.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling: the value is already one scalar per frame (the paper computes it
per pixel from the median of the surrounding frames and reduces each frame to its 99th
percentile across pixels), so the AI only linearly interpolates the per-frame trace onto the
1000 bin centres and NaN-fills the result, exactly as for the velocities. If the side
camera's frame count and the motion-energy length disagree, the frame-time vector is
truncated or **extended with synthetic times at 1/400 s spacing** to force them to match.
For the 7 sessions with no ME file the array is replaced wholesale with zeros.

ii.
```python
            if len(frame_times) > n_frames:
                frame_times = frame_times[:n_frames]
            elif len(frame_times) < n_frames:
                dt_vid = 1/400
                extra = np.arange(len(frame_times), n_frames) * dt_vid + frame_times[-1] + dt_vid
                frame_times = np.concatenate([frame_times, extra])
            aligned_frame_times = frame_times - vidshift - go_time
        interp_fn = interp1d(aligned_frame_times, me_trial, kind='linear',
                             bounds_error=False, fill_value=np.nan)
        me_interp[:, ti] = interp_fn(TIME_AXIS)
```

```python
    if me_interp is not None:
        me_disc = discretize_per_session(me_interp)
    else:
        print(f"  Warning: No motion energy data, using zeros")
        me_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)
```

iii. "Interpolated to neural time axis using video frame times and video offset; Missing
values filled with nearest" — the AI treated motion energy as another camera-rate trace and
reused the same machinery it had already written for the kinematics.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize_per_session` at the 50th percentile of the pooled session values,
`>= → 1`, `< → 0`, no "no video" class. Where an ME file loaded, the split is the intended
0.500/0.500 (0.499/0.501 in a few sessions with ties). For the sessions with no ME file the
discretisation is bypassed entirely and the channel is set to the constant **0**, so 7
sessions in the output are all-low. Overall the split is 58.3% low / 41.7% high.

ii.
```python
        me_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)
...
    threshold = np.percentile(valid, percentile)
    discretized = (values >= threshold).astype(np.int32)
```

iii. Same as 7-c/8-c: a literal reading of "discretized into two bins with per-session
threshold ... 0: < 50th percentile, 1: >= 50th percentile". The zero-substitution is
documented in Known Issue 3 as a deliberate stand-in for the sessions that failed to load.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same session `vidshift` and same per-trial go-cue subtraction, using the **side camera's**
`frameTimes` (`session['traj'][0]`), which is the camera motion energy is computed from,
then linear interpolation onto the shared 1000 bin centres. If the side camera's
`frameTimes` are missing or all NaN for a trial, the code falls back to a synthetic
`arange(1, n+1)/400` axis and a **hard-coded 0.5 s offset** in place of `vidshift`.

ii.
```python
        frame_times = None
        if len(session['traj']) > 0 and trial_idx < len(session['traj'][0]['trials']):
            ft = session['traj'][0]['trials'][trial_idx].get('frameTimes')
            if ft is not None and not np.all(np.isnan(ft)):
                frame_times = ft

        if frame_times is None:
            frame_times = np.arange(1, n_frames + 1) / 400.0
            aligned_frame_times = frame_times - 0.5 - go_time  # fallback offset
        else:
            ...
            aligned_frame_times = frame_times - vidshift - go_time
```

iii. Not separately justified beyond "Interpolated to neural time axis using video frame
times and video offset (matching `findVideoOffset.m`)". Using camera 0's frame times for
motion energy matches the authors' `loadMotionEnergy.m`; the `- 0.5` fallback has no basis
in the reference code and is an invented constant.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Almost every gap is **filled or zeroed rather than marked**, and the substituted value
then becomes a confident class label:
- **Invisible tongue** (≈80–90% of bins): velocity set to 0, which collapses the session
  median to 0 and makes the channel constant (7-c).
- **Untracked non-tongue frames**: x and y linearly interpolated across the gap before
  differentiation, so a fabricated velocity is produced for frames the camera never tracked.
- **Bins with no camera coverage**: filled by linear interpolation between valid bins, or set
  to 0 if the whole trial is empty.
- **Unloadable motion-energy files** (7 sessions): whole channel set to 0.
- **Failed paw extraction** (11 sessions): whole channel set to 0.
- **Missing `frameTimes`**: replaced by a synthetic 400 Hz axis; for motion energy with a
  `- 0.5` s magic offset instead of the real `vidshift`.
- **Failed `vidshift` computation**: silently defaults to 0.0.
- **Sessions with no `clu` field**, and any other exception in a session: the whole session
  is dropped by a blanket `except Exception` in the session loop.
- **Trials past the end of the recording**: not detected; 30 trials with identically zero
  neural data remain in the dataset.
Broad bare `except:` clauses are used throughout the loaders, so a field that fails to parse
becomes `'unknown'`, `0`, or `None` without any warning.

ii.
```python
        elif mask.all():
            velocities[:, ti] = 0.0
```

```python
        except:
            session['vidshift'] = 0.0
```

```python
    except Exception as e:
        print(f"\n  ERROR processing session {idx} ({sf['animal_date']}): {e}")
        import traceback
        traceback.print_exc()
        continue
```

iii. The fills are justified in `CONVERSION_NOTES.md` as following `findPosition.m`
("Missing values were filled in with the nearest available value for all features, except
for the tongue"), though the code applies a fill to the tongue as well (zeros, then
interpolation at the bin level). The degenerate consequences are all written up under
"Known Issues" in the notes; none were repaired before the final run.

## 11-a. What are the most time-consuming steps of the code?

i. No timing instrumentation exists in the script and the AI never profiled it. From the
structure, the dominant cost is `align_and_bin_spikes`, a double Python loop over
(units × kept trials) — roughly 2,600 units × ~270 trials ≈ 700,000 iterations, each of
which recomputes a full boolean mask over that unit's entire spike vector
(`spike_trials == trial_num`), calls `np.histogram`, and calls `np.convolve` on a
1000-sample vector. Second is file I/O: every v7.3 session is opened twice (once as a format
probe, once for real) and the per-cluster/per-trial HDF5 reference dereferencing is itself a
Python loop. Third is the kinematics: three `extract_kinematic_feature` passes per session,
each looping over trials with a separate `interp1d` construction, plus per-trial NaN-fill
loops. Pickling the 3.2 GB result is also non-trivial.

ii.
```python
    for ui, ci in enumerate(cluster_indices):
        unit = clusters[ci]
        spike_times = unit['trialtm']
        spike_trials = unit['trial']

        for ti, trial_idx in enumerate(valid_trials):
            ...
            spike_mask = spike_trials == trial_num
            ...
            counts, _ = np.histogram(aligned_times, bins=edges)
            fr = counts / dt
            fr_smooth = causal_gaussian_smooth(fr, smooth_samples)
```

iii. Not discussed in the trajectory; the AI allotted long timeouts (up to 600 s) for the
full conversion and did not otherwise consider runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Four places. **(1)** The `(units × trials)` loop in `align_and_bin_spikes` is the big one:
a single `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, EDGES])` per unit
(as the reference does) removes the inner loop and the repeated boolean masking entirely.
**(2)** `causal_gaussian_smooth` loops over columns with `np.convolve`; with the trial×bin
matrix built first, one `scipy.ndimage.convolve1d(..., axis=...)` would smooth every trial at
once. **(3)** The two NaN-fill loops over trials in `extract_kinematic_feature` and
`interpolate_motion_energy` operate column by column on a rectangular array and could be done
with a single vectorised forward/backward fill. **(4)** The per-trial `interp1d` construction
is unavoidable in principle (frame counts differ per trial) but the object creation could be
replaced by a plain `np.interp` call.

ii.
```python
        out = np.zeros_like(x)
        for j in range(x.shape[1]):
            out[:, j] = np.convolve(x[:, j], kern, mode='same')
```

```python
    for ti in range(n_trials):
        col = velocities[:, ti]
        mask = np.isnan(col)
        if mask.any() and not mask.all():
            valid_idx = np.where(~mask)[0]
            col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
```

iii. Not discussed. No vectorisation was attempted anywhere in the script.

## 11-c. What processing does the code repeat multiple times?

i. Several repeats. **(1)** `spike_trials == trial_num` rebuilds an O(n_spikes) mask once per
trial per unit, so each unit's spike vector is scanned ~270 times instead of once.
**(2)** Every v7.3 file is opened by h5py twice — once in `load_session_data` purely to test
for `obj`/`clu`, then again in `_load_session_data_h5`. **(3)** `extract_kinematic_feature`
is called three times per session (tongue, `top_paw`, `bottom_paw`) and each call re-scans
`feat_names`, re-reads `vidshift` and re-walks the per-trial trajectory dicts.
**(4)** `frame_times` for a trial is fetched independently by the kinematics and by
`interpolate_motion_energy`. **(5)** Smoothing is applied to *all* quality-passing units and
only afterwards are the low-rate units discarded, so the convolution work for the dropped
units is thrown away. On the other hand `vidshift` is correctly computed once per session at
load time and `EDGES`/`TIME_AXIS` once at module scope.

ii.
```python
        for ti, trial_idx in enumerate(valid_trials):
            ...
            spike_mask = spike_trials == trial_num
```

```python
            fr_smooth = causal_gaussian_smooth(fr, smooth_samples)
            trialdat[:, ui, ti] = fr_smooth
    mean_frs = np.mean(np.mean(trialdat, axis=2), axis=0)
    ...
        fr_mask = mean_frs > LOW_FR          # smoothing already paid for dropped units
```

iii. Not discussed in the trajectory.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. **Dead code and unused imports**: `compute_velocity` is defined and never called;
`gaussian_filter1d` and `sys` are imported and never used; `h5_deref` is never called;
the `elif q == ''` branch in `filter_clusters` is unreachable. **Loaded but unused fields**:
`bp.L`, `bp.no`, `bp.ev.sample`, `bp.ev.delay`, `meta.anm` / `meta.day` (overridden by the
filename), `me.moveThresh`, and `NdroppedFrames` (read for every trial of every camera,
consulted only in one guard). **Computed then discarded**: the full x/y/likelihood array of
all 7 side-camera and all bottom-camera features is materialised per trial although only
three features are used; smoothed firing rates are computed for units that the 0.5 Hz filter
then removes; clusters on non-ALM probes are fully parsed (quality, all spike times, all
trial labels) before the probe is skipped; the entire tongue-velocity pipeline —
per-frame gradients, interpolation onto 1000 bins, NaN filling — is executed at full cost for
all 12,293 trials only to collapse to a constant 1 (7-c). The output is also stored as
`int32` where `int8` would do, inflating the 3.2 GB pickle.

ii.
```python
def compute_velocity(positions, dt_video=1/400):
    """Compute velocity magnitude from x,y positions."""   # never called
```

```python
from scipy.ndimage import gaussian_filter1d      # never used
```

```python
        thresh_field = me['moveThresh'][0, 0]
        ...
        return {'data': me_data, 'moveThresh': threshold}   # threshold never read
```

```python
        output_trial = np.zeros((6, N_TIMEBINS), dtype=np.int32)
```

iii. Not discussed. The dead helpers are leftovers from the several rewrites of the
kinematics code between steps 30 and 85.
