# Decisions

> **Note on the instruction version the AI actually received.** Step 1 of
> `/logs/agent/trajectory.json` shows the AI was given a *two-class* version of the
> Decoder Output spec:
> `Lick direction (left = 0, right = 1)`, `Outcome (incorrect = 0, correct = 1)`, and
> tongue/paw/motion energy "discretized into **two** bins" with a 50th-percentile
> threshold. The evaluation reference (`/tests/instruction_reference.md`) instead asks for
> three classes each (`none` for lick direction, `ignore` for outcome, `not visible` /
> `no video` for the three kinematic streams). Wherever the AI's output differs from the
> human reference solely by the absence of that third class, it is noted below; that
> divergence is not evidence of careless work by the AI.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** glob the data folders. It hard-codes a 44-entry table
`EPHYS_SESSIONS` of `(directory, animal, date, probe_numbers)` transcribed from the
authors' `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` scripts — 25
sessions from `data/Ephys_Behavior` and 19 from `data/RandomizedDelay_Ephys_Behavior`.
Three files present on disk but commented out of the loader scripts (`JEB23_2023-10-20`,
`JEB24_2023-10-03`, `JEB24_2023-10-04`) are omitted, and the two behaviour-only folders
(`DelayInhibition_BilatMC_Behavior`, `GoCueInhibition_BilatMC_Behavior`) are excluded
because they carry no ephys. Each session's `data_structure_<anm>_<date>.mat` is opened
once by `load_mat_file`, which tries `h5py` (MATLAB v7.3) and falls back to `scipy.io`
(v5); motion energy comes from the sibling `motionEnergy_<anm>_<date>.mat`. Field access
is then routed through format-aware helpers (`get_bp_field`, `get_clusters`,
`get_traj_data`). The session/probe table is byte-identical to the human reference's
`SESSIONS` dict (verified: 44/44 names match, 0 probe differences).

ii.
```python
EPHYS_SESSIONS = [
    # Ephys_Behavior
    ('data/Ephys_Behavior', 'JEB6', '2021-04-18', [2]),
    ...
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-11-03', [1]),
]

def load_mat_file(filepath):
    """Load a .mat file, handling both v5 and v7.3 (HDF5) formats."""
    try:
        f = h5py.File(filepath, 'r')
        return f, 'h5'
    except Exception:
        mat = scipy.io.loadmat(filepath, squeeze_me=False)
        return mat, 'v5'
```

```python
for sess_idx, (dirpath, animal, date, probes) in enumerate(sessions):
    result = process_session(dirpath, animal, date, probes, time_axis, edges, ...)
```

iii. CONVERSION_NOTES Step 4: "Sessions (randomized delay) | 19 sessions in loader scripts
(excluding JEB23_10-20, JEB24_10-03/04) | 22 data files, 19 in loaders | 19 | Consistent -
3 files not in loader scripts", and Step 5 decision 14: "Probe selection: Use probe
number(s) from session loader scripts. For multi-probe sessions (JEB15), concatenate."
Step 2 notes "Data files are a mix of HDF5 (v7.3) and MATLAB v5 format", motivating the
dual reader.

## 1-b. How are the data split into subjects?

i. The animal identifier is a literal column of the `EPHYS_SESSIONS` table (the same
string that prefixes the filename), so `JEB19_2023-04-19` belongs to subject `JEB19`. It is
carried through `process_session` as `result['animal']`, and at assembly `subjects` is the
sorted set of unique animals with `subject_idx` giving each session's index into it. The
result is 14 subjects over 44 sessions, matching the human reference exactly (including the
per-subject session counts 1/1/2/2/5/4/4/4/7/8/1/2/2/1).

ii.
```python
all_animals.append(animal)
...
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals])
...
'subjects': unique_subjects,
'subject_idx': subject_idx,
```

iii. CONVERSION_NOTES Step 2 lists the 14 animals and Step 4 discusses at length why the
paper's "nine mice" for the fixed-delay task does not contradict 14 unique animals across
both ephys datasets, concluding "We'll include all sessions from the loader scripts."
The animal is read off the loader-script filenames rather than `obj.meta.anm` (which is
absent in several sessions).

## 1-c. How are the data split into sessions?

i. One session = one row of `EPHYS_SESSIONS` = one `data_structure_*.mat` file, and the
directory is stored explicitly in the row rather than searched for, so the fixed-delay and
randomized-delay tasks are pooled into one uniform list of sessions. Each becomes one
element of `neural`, `input`, `output` and `brain_region_idx`. `process_session` returns
`None` (dropping the session) if the file is missing, if fewer than 2 valid trials remain,
or if fewer than `MIN_UNITS = 10` units survive; in practice no session was dropped, so 44
sessions are emitted.

ii.
```python
def process_session(dirpath, animal, date, probe_nums, time_axis, edges, ...):
    session_id = f"{animal}_{date}"
    filepath = os.path.join(dirpath, f"data_structure_{session_id}.mat")
    if not os.path.exists(filepath):
        print(f"  WARNING: File not found: {filepath}")
        return None
```
```python
    if len(valid_trial_indices) < 2:
        print(f"    SKIP: Too few valid trials"); ...; return None
    if len(all_clusters) < MIN_UNITS:
        print(f"    SKIP: Too few neurons ({len(all_clusters)} < {MIN_UNITS})"); ...; return None
```

iii. CONVERSION_NOTES Step 3 records the paper's session-inclusion rule — "Min unit filter
| >= 10 units per session | 'Recording sessions were included for analysis only if they had
at least 10 units'" — which is where `MIN_UNITS` comes from. Step 5 decision 16: "Both
ephys datasets: Include both Ephys_Behavior and RandomizedDelay_Ephys_Behavior".

## 1-d. How are the data split into trials?

i. Trial count comes from `obj.bp.Ntrials`; every per-trial behavioural field of `obj.bp`
is flattened to that length and indexed 0-based, the go cue is `obj.bp.ev.goCue` (one per
trial), spike times carry a 1-based `clu.trial` index, and video/motion-energy are stored
as one cell per trial. So trials are read directly from the Bpod table and no trial
boundaries are reconstructed. Unlike the human reference, the AI does **not** explicitly
truncate over-long `bp` fields to `Ntrials` (`trial_column`'s `[:n_trials]` guard); it
relies on the fields already being the right length, which held for all 44 sessions.

ii.
```python
def get_ntrials(data, fmt):
    if fmt == 'h5':
        return int(data['obj']['bp']['Ntrials'][0, 0])
    else:
        return int(data['obj']['bp'][0, 0]['Ntrials'][0, 0])
```
```python
    for j in range(ntrials):
        trial_num = j + 1  # 1-indexed
        spike_mask = trial == trial_num
```

iii. CONVERSION_NOTES Step 2: "obj.bp fields: hit, miss, R, L, autowater, early, no,
stim.enable, ev (events)"; "obj.clu: ... clusters with fields: tm, trialtm, trial,
quality"; "obj.traj: cell array [sidecam; bottomcam], each has per-trial: featNames,
frameTimes, ts". The trial table therefore defines the trials directly.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, all applied before anything is computed: (1) photostimulation trials
(`bp.stim.enable`) are dropped, (2) early-lick trials (`bp.early`) are dropped, (3) trials
whose go-cue time is NaN or non-positive are dropped, and (4) trials beyond the end of the
spike recording are dropped, where the cutoff is the largest trial number appearing in any
surviving cluster's `clu.trial`. Filter (4) was added during Step 10 after the AI found
that `JEB24_2023-10-23` (28 trials) and `JEB24_2023-11-03` (33 trials) had all-zero neural
data at the end of the session. No filter on hit/miss/ignore is applied. The result is
**13,762 trials**, which is exactly the human reference's count.

ii.
```python
# Trial filter: exclude stim and early lick trials
valid_trials = ~stim_enable & ~early
# Also exclude trials where align time is 0 or NaN (no go cue)
valid_trials &= ~np.isnan(align_times) & (align_times > 0)
valid_trial_indices = np.where(valid_trials)[0]
```
```python
# Find max trial with spike data (recording may end before session ends)
if all_clusters:
    max_spike_trial = max(
        int(np.max(c['trial'])) for c in all_clusters if len(c['trial']) > 0
    )
    beyond = np.sum(valid_trial_indices >= max_spike_trial)
    if beyond > 0:
        valid_trial_indices = valid_trial_indices[valid_trial_indices < max_spike_trial]
```

iii. CONVERSION_NOTES Step 5 decision 6: "Trial filter: Exclude stim.enable=1 and early=1
trials. Keep hit, miss, and no (ignore) as they represent different outcomes", traced in
Step 1 to `getDefaultParams.m`'s conditions (`'...&~stim.enable&...&~early'`). Step 10
issue log: "Zero neural data for late trials (sessions 36, 43): Spike recording ended
before behavioral session. Fixed by detecting max spike trial and excluding behavioral
trials beyond recording range. 28+33=61 trials excluded."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}`, the spike-sorted clusters, restricted to the probe number(s) given by
the session table. Each cluster contributes `trialtm` (spike time relative to its trial's
start, on the behaviour clock), `trial` (1-based trial index of each spike) and `quality`
(manual curation label). `obj.bp.ev.goCue` is the second input, since it defines the
alignment. Units from both probes of a two-probe session (the three JEB15 sessions) are
concatenated into one population.

ii.
```python
def get_clusters(data, fmt, probe_idx):
    """Get cluster data for a given probe.
    Returns list of dicts with keys: trialtm, trial, quality"""
    ...
        trialtm_ref = probe_data['trialtm'][i, 0]
        trialtm = f[trialtm_ref][:].flatten()
        trial_ref = probe_data['trial'][i, 0]
        trial = f[trial_ref][:].flatten().astype(int)
        clusters.append({'trialtm': trialtm, 'trial': trial, 'quality': quality})
```
```python
all_clusters = []
for p in probe_nums:
    clusters = get_clusters(data, fmt, p - 1)  # 0-indexed
    all_clusters.extend(clusters)
...
align_times = get_event_times(data, fmt, ALIGN_EVENT)   # ALIGN_EVENT = 'goCue'
```

iii. CONVERSION_NOTES Step 2: "Spike data: obj.clu{probe}(cluster).tm (session time),
.trialtm (trial time), .trial (trial number), .quality (string)". Step 1 maps this to the
reference pipeline `findTrials -> findClusters -> alignSpikes -> getSeq -> removeLowFR`.

## 2-b. How is the `neural` data processed?

i. Three steps, mirroring `getSeq.m`. Spikes of each cluster are histogrammed per trial
into the bin edges `-2.5 : 0.01 : 2.5`; counts are divided by the bin width to give
spikes/s; and each trial's rate trace is smoothed along time with a **causal** Gaussian
built by `causal_gaussian_smooth(rate, N=15, bc='reflect')` — a length-15 Gaussian
(`sigma = N/6 = 2.5` bins = 25 ms) whose first half is zeroed so only past samples
contribute, with reflect padding at the edges. No normalisation, baseline subtraction or
z-scoring is applied, so stored values are firing rates in Hz (`float32`). This differs
from the human reference, which uses a *symmetric* 14 ms-sigma Gaussian at 5 ms bins; the
AI's choice follows the reference MATLAB more literally (`getDefaultParams.m`: "% smooth
with causal gaussian kernel / params.smooth = 15", implemented by `mySmooth.m`, which
"Makes filter causal by zeroing first half of kernel").

ii.
```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    """Causal Gaussian smoothing, matching mySmooth.m from reference code."""
    t = np.arange(N); mu = (N - 1) / 2; sigma = N / 6
    kernel = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    kernel[:int(np.ceil(N / 2))] = 0        # make causal
    kernel = kernel / kernel.sum()
    if bctype == 'reflect':
        padded = np.concatenate([x[N-1:0:-1], x, x[-2:-N-1:-1]])
        smoothed = np.convolve(padded, kernel, mode='same')
        return smoothed[N-1:N-1+len(x)]
```
```python
            counts = np.histogram(aligned_times, bins=edges)[0]
            rate = counts.astype(np.float64) / DT
            smoothed = causal_gaussian_smooth(rate, SMOOTH_N, SMOOTH_BC)
            trialdat[:, i, j] = smoothed.astype(np.float32)
```

iii. CONVERSION_NOTES Step 1: "Binning: edges = tmin:dt:tmax, time = edges + dt/2 (center
of bins), drop last edge. Spike counts per bin, then divide by dt for rate, then smooth";
"Smoothing: Causal Gaussian kernel with N=15, boundary='reflect'". Step 5 decision 3:
"Smoothing: Causal Gaussian N=15, reflect boundary, as in code".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters plus one session-level filter. (1) The free-text manual curation
label `clu.quality` is stripped and lower-cased and the cluster is dropped if it is one of
`garbage`, `gabrga`, `noisy`, `real?` — exactly the drop list of the reference's
`findClusters.m`; multi-units and unlabeled clusters are kept. (2) After binning, any unit
whose mean rate (averaged over all time bins and *all* trials of the session) is not
greater than 1 Hz is dropped, following `removeLowFRClusters.m`. (3) A session with fewer
than 10 surviving units would be dropped entirely (none was). This leaves **2,457 units**,
15–141 per session, mean 55.8. Note the paper reports 1,651 + 845 = 2,496 units, so the
AI's count is within 1.6% of the paper; the human reference additionally drops the label
`poor` and ends at 1,954 units. One small deviation: the low-FR mean is taken over the full
`trialdat` (all `Ntrials`), i.e. including the photostim/early/beyond-recording trials that
are later discarded, whereas `removeLowFRClusters.m` averages the condition PSTHs (filtered
trials only).

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR_THRESHOLD = 1.0  # Hz, from paper
MIN_UNITS = 10  # minimum units per session (from paper)
...
        quality = h5_deref_string(f, q_ref).strip().lower()
        if quality in EXCLUDE_QUALITIES:
            continue
```
```python
def remove_low_fr_neurons(trialdat, clusters):
    """Matches removeLowFRClusters.m: mean of mean PSTH across conditions."""
    mean_frs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
    keep = mean_frs > LOW_FR_THRESHOLD
    return trialdat[:, keep, :], [c for c, k in zip(clusters, keep) if k], keep
```

iii. CONVERSION_NOTES Step 3 quotes the paper's 1 Hz rule ("Low FR threshold | 1 Hz |
'firing rates exceeding 1 Hz'") and the ≥10-unit session rule. Step 4 resolves a conflict:
"Low FR threshold | WorkingWithDataObjs: 1 Hz; getDefaultParams: 0.5 Hz | ... | 1 Hz | Use
1 Hz as stated in paper". Step 5 decision 5: "Quality filter: Exclude 'garbage', 'noisy',
'real?', 'gabrga' (case-insensitive, strip whitespace)". Step 9 compares the resulting
2,457 to the paper's 2,496 and calls it "Close (98.4% of paper total)".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By a single subtraction, matching `alignSpikes.m`. `clu.trialtm` is already on the
behaviour clock and already relative to its own trial's start, and `bp.ev.goCue` is on the
same clock, so `trialtm − goCue[trial]` gives seconds from the go cue with no interpolation
or offset. Those aligned times are then histogrammed into the fixed `-2.5 … 2.5` edges, so
spikes outside the window simply fall off the ends. `ALIGN_EVENT` is a module constant set
to `'goCue'`.

ii.
```python
ALIGN_EVENT = 'goCue'
...
    for i, clu in enumerate(clusters):
        trialtm = clu['trialtm']; trial = clu['trial']
        for j in range(ntrials):
            spike_mask = trial == j + 1
            if not np.any(spike_mask): continue
            aligned_times = trialtm[spike_mask] - align_times[j]
            counts = np.histogram(aligned_times, bins=edges)[0]
```

iii. CONVERSION_NOTES Step 1: "alignSpikes | ... | Align spike times to event (goCue):
trialtm_aligned = trialtm - event_time"; Step 3 "Align to goCue event; Time window: -2.5 to
2.5 s from goCue".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms** (`DT = 1/100`), giving 500 non-overlapping bins spanning −2.5 to +2.5 s from
the go cue, with bin centres at −2.495 … 2.495 s. The grid is built once by
`compute_time_axis()` (edges + dt/2, drop last — exactly `getSeq.m`) and reused for every
trial, session and data stream, so neural, input and all three camera outputs share one
time axis. No rebinning or resampling of already-binned data occurs: spikes are
histogrammed straight onto this grid, and the camera streams are linearly interpolated onto
it. The AI found two conflicting values in the repository and chose the tutorial's:
`WorkingWithDataObjs.m` uses `dt = 1/100`, `getDefaultParams.m` uses `dt = 1/200`. The human
reference chose 5 ms (1,000 bins).

ii.
```python
TMIN = -2.5  # seconds
TMAX = 2.5   # seconds
DT = 1.0 / 100  # 10 ms time bins

def compute_time_axis():
    """Compute time axis matching getSeq.m: edges + dt/2, drop last."""
    edges = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = edges[:-1] + DT / 2
    return time_axis, edges
```
```python
'time_bin_size': DT * 1000,  # in ms
```

iii. CONVERSION_NOTES Step 4: "dt (time bin) | WorkingWithDataObjs: 1/100=10ms;
getDefaultParams: 1/200=5ms | N/A | Not explicitly stated | Use 10ms (1/100) as in
WorkingWithDataObjs.m tutorial which matches the main analysis script". Step 5 decision 1
repeats this.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from the raw data at all — it is defined by the conversion as the centre
of each of the 500 bins of the analysis window, which itself is anchored on
`obj.bp.ev.goCue` (the alignment event, mapped to t = 0). The same 500-element vector is
used for every trial of every session.

ii.
```python
def compute_time_axis():
    edges = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = edges[:-1] + DT / 2
    return time_axis, edges
```
```python
input_data = time_axis.astype(np.float32).reshape(1, -1)
input_trials.append(input_data)
...
'input_names': ['time_from_go_cue'],
```

iii. CONVERSION_NOTES Step 5 variable mapping: "time from goCue | input[0] | Continuous
time axis | N/A | Same time axis for all trials: tmin:dt:tmax centered". The window
`[-2.5, 2.5]` is taken from `params.tmin`/`params.tmax` in the reference code (Step 3
"Time window: -2.5 to 2.5 s from goCue").

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the grid: bin edges `np.arange(-2.5, 2.505, 0.01)` are shifted
by `dt/2` and the last edge is dropped, reproducing `getSeq.m`'s
`obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1)`. The vector is cast to
`float32` and reshaped to `(1, 500)` per trial. It is stored as a continuous ramp, not a
binary event indicator, matching the Decoder-Input spec ("continuous, time-varying").

ii.
```python
    time_axis = edges[:-1] + DT / 2
...
        input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. No separate justification beyond the mapping table; the AI notes in Step 10 check 4
that "All sessions have identical correct time axis [-2.495, 2.495] s, 500 bins."

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `compute_time_axis()` returns both `time_axis` and
`edges`; `edges` is passed to `bin_and_smooth_spikes` and used as the histogram edges for
the go-cue-aligned spike times, while `time_axis` (the centres of those same edges) becomes
the input. Bin *k* of `input` is therefore the same interval as bin *k* of `neural` by
construction, and the same `time_axis` is used as the interpolation target for the three
camera streams, so all five streams share one axis.

ii.
```python
time_axis, edges = compute_time_axis()
...
trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
...
        counts = np.histogram(aligned_times, bins=edges)[0]
```
```python
        input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. CONVERSION_NOTES Step 5 planned sanity check: "Check that time axis is correct (bins
centered on goCue=0)"; Step 10 check 4 reports it verified.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. **Only `obj.bp.R`.** `obj.bp.L` is loaded but never used, and `hit`/`miss` are not
consulted when building this variable. `bp.R`/`bp.L` encode the *instructed / rewarded*
port for the trial (in WC blocks, the port where water was delivered) — not which way the
animal actually licked. The human reference derives lick direction from the combination of
`R` with `hit` and `miss` (a hit licks the instructed port, a miss licks the other one),
and gives non-responding trials their own `no lick` class.

ii.
```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
miss = get_bp_field(data, fmt, 'miss').astype(bool)
R = get_bp_field(data, fmt, 'R').astype(bool)
L = get_bp_field(data, fmt, 'L').astype(bool)
...
# Lick direction: R=1, L=0 (trial instruction/stimulus side)
lick_direction = R_valid.astype(np.float32)
```

iii. The AI identified the correct derivation and then explicitly abandoned it.
CONVERSION_NOTES "Revised Decision on Lick Direction": "For the decoder output 'lick
direction', we want the ACTUAL lick direction: Hit + R = licked right -> 1; Hit + L =
licked left -> 0; Miss + R = instructed right but licked left -> 0; Miss + L = instructed
left but licked right -> 1; No (ignore) = no lick -> exclude these trials.  Actually, we
should keep it simple: R=1, L=0 as the instruction/stimulus direction. The outcome variable
captures whether they got it right. ... Let me just use R=1, L=0 as the trial type
variable."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct cast of the boolean `R` flag to `{0, 1}` on the surviving trials, labelled
`['left', 'right']`, then broadcast across all 500 time bins so the output array is
rectangular. There is no `none`/`no lick` class, and no correction for error trials.
Consequence: on **miss** trials (~12% of trials) the stored label is the *opposite* of the
direction the animal licked, and on **ignore** trials (~13%) a lick direction is asserted
where no lick occurred — roughly a quarter of trials carry a label that does not describe
the animal's lick. The resulting distribution is 49.5% / 50.5%, versus the human
reference's 42.3% left / 44.6% right / 13.1% no lick.

ii.
```python
    lick_direction = R_valid.astype(np.float32)
...
        out = np.zeros((6, len(time_axis)), dtype=np.int64)
        out[0, :] = int(lick_direction[t_idx])
...
'output_values': [
    ['left', 'right'],           # lick_direction: 0=left, 1=right
```

iii. As quoted in 4-a: "we should keep it simple: R=1, L=0 as the instruction/stimulus
direction. The outcome variable captures whether they got it right." Also Step 5 mapping
row: "obj.bp.R/L | output[0]: lick_direction | L=0, R=1 | findTrials | Per-trial scalar."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. A single per-trial flag, `obj.bp.autowater`, which marks trials where water was delivered
at a random port with no sample tone, delay or go cue — the water-cued (WC) block.
Everything else is the delayed-response (DR) context. This is the same single source the
human reference uses.

ii.
```python
autowater = get_bp_field(data, fmt, 'autowater').astype(bool)
...
autowater_valid = autowater[valid_trial_indices]
```

iii. CONVERSION_NOTES Step 1: "Trial conditions include: ~stim.enable (no optogenetic
stim), ~early (no early licks), **autowater for WC vs DR context**", read off
`getDefaultParams.m`'s condition strings. Step 5 mapping: "obj.bp.autowater |
output[1]: behavioral_context".

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling of the flag with the polarity the task prescribes: autowater → WC
→ 0, otherwise DR → 1, implemented as `(~autowater).astype(float)`. It is a per-trial
scalar repeated across all 500 bins. The delivered distribution (WC 9.7% / DR 90.3%) is
identical to the human reference's (9.66% / 90.34%) to three decimals, and the 13 sessions
that are all-DR are flagged and explained in the notes rather than treated as an error.

ii.
```python
    # Behavioral context: WC=0, DR=1
    behavioral_context = (~autowater_valid).astype(np.float32)
...
        out[1, :] = int(behavioral_context[t_idx])
...
    ['WC', 'DR'],                # behavioral_context: 0=WC, 1=DR
```

iii. CONVERSION_NOTES Step 5 decision 8: "Behavioral context: WC=0 (autowater=1), DR=1
(autowater=0)". Step 9 warnings: "13 sessions have behavioral_context range [1.0, 1.0] (all
DR, no WC) - correct for sessions without autowater blocks".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `obj.bp.hit` alone. `obj.bp.miss` is loaded but not used for this variable, and
`obj.bp.no` (the ignore flag) is never read. Because the three flags are mutually exclusive,
`hit` is sufficient to define a two-way correct/not-correct split, but it cannot
distinguish an error lick from a non-response. The human reference reads both `hit` and
`miss` in order to separate `incorrect`, `correct` and `ignore`.

ii.
```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
miss = get_bp_field(data, fmt, 'miss').astype(bool)
...
hit_valid = hit[valid_trial_indices]
miss_valid = miss[valid_trial_indices]
```

iii. CONVERSION_NOTES Step 5 mapping: "obj.bp.hit | output[2]: outcome |
incorrect(miss/no)=0, correct(hit)=1 | findTrials | Per-trial scalar", i.e. the merge of
`miss` and `no` into a single "incorrect" class was deliberate and documented.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A cast of the `hit` flag: correct = 1, everything else (miss **and** ignore) = 0,
labelled `['incorrect', 'correct']`, repeated across all 500 bins. The AI's received
instruction specified exactly two outcome classes, so no `ignore` class was created; the
evaluation reference asks for three. The correct-trial fraction the AI obtains (0.749) is
identical to the human reference's (0.7490); the difference is entirely that the reference's
remaining 25.1% splits into 12.0% `incorrect` and 13.1% `ignore` whereas the AI's is one
pooled class.

ii.
```python
    # Outcome: correct=1, incorrect=0
    outcome = hit_valid.astype(np.float32)
...
        out[2, :] = int(outcome[t_idx])
...
    ['incorrect', 'correct'],    # outcome: 0=incorrect, 1=correct
```

iii. CONVERSION_NOTES Step 5 decision 9: "Outcome: correct (hit=1) -> 1, incorrect (miss=1
or no=1) -> 0", combined with decision 6, "Keep hit, miss, and no (ignore) as they represent
different outcomes" (i.e. ignore trials are retained in the dataset, just not given their
own outcome label).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, **side camera only** (`view = 0`), feature name
`'tongue'`; `obj.traj{view}` supplies `featNames`, `frameTimes` and `ts` (x, y, likelihood
per feature per frame). `obj.bp.ev.goCue` plus `obj.sglx.bitcode.bitstart` / `obj.sglx.fs` /
`obj.bp.ev.bitStart` are also needed, to put frames on the go-cue clock. The bottom
camera's `top_tongue` is *not* used; the human reference averages both views, noting the
bottom camera detects the tongue on roughly twice as many frames as the side view.

ii.
```python
    tongue_speed = extract_velocity_from_traj(
        data, fmt, view=0, feat_name='tongue',
        ntrials=ntrials, align_times=align_times,
        time_axis=time_axis, vidshift=vidshift
    )
```
```python
        ts, frame_times, feat_names = get_traj_data(data, fmt, view, trial_idx)
        feat_idx = feat_names.index(feat_name)
        ...
            xpos = ts[feat_idx, 0, :]
            ypos = ts[feat_idx, 1, :]
```

iii. CONVERSION_NOTES Step 2 lists the per-camera feature names ("Side cam features:
tongue, left_tongue, right_tongue, jaw, trident, nose, lickport"). Step 5 decision 10
records the AI first considering the jaw as a substitute and rejecting it: "Compute from
jaw feature on side cam (y-velocity), or from tongue if available... Since tongue is often
not visible, better to use jaw. But the task says 'tongue velocity' - use tongue velocity
from side cam."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Following `findPosition.m`/`findVelocity.m` in the reference MATLAB: (1) x and y for the
tongue are taken **unsmoothed** (the reference smooths only non-tongue features); (2)
`np.gradient` is applied to x and y with respect to *frame index*, so units are pixels per
frame rather than pixels per second; (3) any NaN velocity — which is what DeepLabCut writes
wherever the tongue is not visible, i.e. ~85–93% of frames — is **set to 0**, so "not
visible" becomes indistinguishable from "stationary"; (4) speed is `sqrt(xvel² + yvel²)`;
(5) remaining NaNs are filled with the nearest valid sample; (6) the frame-resolution speed
is linearly interpolated onto the 500-bin neural axis (not bin-averaged); (7) trials whose
speed is entirely NaN are set to 0. No cross-camera normalisation is needed because only one
view is used. The AI applies no explicit likelihood threshold, relying on x/y already being
NaN below the authors' cut.

ii.
```python
        is_tongue = 'tongue' in feat_name.lower()
        if not is_tongue:
            xpos_smooth = causal_gaussian_smooth(xpos, 21, 'reflect')
            ypos_smooth = causal_gaussian_smooth(ypos, 21, 'reflect')
        else:
            xpos_smooth = xpos.copy(); ypos_smooth = ypos.copy()

        xvel = np.gradient(xpos_smooth); yvel = np.gradient(ypos_smooth)
        ...
        # For tongue: set NaN velocities to 0
        if is_tongue:
            xvel[np.isnan(xvel)] = 0
            yvel[np.isnan(yvel)] = 0
        spd = np.sqrt(xvel**2 + yvel**2)
        ...
            interpolated = np.interp(time_axis, aligned_ft, spd)
            speed[:, trial_idx] = interpolated.astype(np.float32)
```

iii. CONVERSION_NOTES Step 1: "findVelocity | ... | Compute velocity from position using
gradient()" and the sub-agent report it is based on: "For non-tongue features: subtracts
baseline velocity (median velocity). For tongue features: sets NaN velocities to 0
(indicates not visible). Fills missing values with nearest neighbor interpolation."
Step 5 decision 10: "When tongue not visible, velocity = 0 (as per findVelocity code)."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. **The stated rule and the implemented rule differ.** The documented decision (Step 5
decision 13) is a per-session 50th-percentile split over all bins of all kept trials. The
code computes that percentile — but `discretize_time_series` then silently raises any
threshold below `1e-10` to `1e-10`, and because ~90% of tongue speeds are exactly 0 (see
7-b step 3) the session median *is* 0 for every session. The effective tongue rule is
therefore "0 if speed is exactly zero, 1 if speed is non-zero", i.e. a
tongue-invisible/still vs tongue-moving split, not a median split. The AI discovered the
degeneracy during Step 7 (tongue came out 100% class 1) and added the epsilon as the fix.
The result is 90.8% / 9.2%. Only two classes exist; there is no `not visible` class. The
human reference's three-class split is 6.2% below / 6.2% above / 87.5% not visible — so the
AI's class 0 corresponds roughly to the reference's class 2, with the reference's classes 0
and 1 collapsed into the AI's class 1.

ii.
```python
def discretize_time_series(values, threshold):
    """Discretize: 0 if < threshold, 1 if >= threshold.
    If threshold is 0 or very small (degenerate case where median is 0),
    use a small positive threshold so that exact zeros map to class 0.
    """
    if threshold < 1e-10:
        threshold = 1e-10  # Separate zero from positive values
    return (values >= threshold).astype(np.float32)
```
```python
    tongue_thresh = np.nanpercentile(tongue_speed_valid, 50)
    tongue_disc = discretize_time_series(tongue_speed_valid, tongue_thresh)
```

iii. Trajectory step 112: "Tongue velocity is all 1. The problem is that tongue speed is
mostly 0 (tongue not visible for most of the trial), so the 50th percentile is 0, and all
values >= 0 are classified as 1." Step 115: "When the percentile is 0 (or very small), I'll
use a small positive threshold so that 'tongue not visible' (speed=0) maps to class 0 and
'tongue moving' (speed>0) maps to class 1." CONVERSION_NOTES Step 7 records the consequence
("Tongue velocity highly imbalanced (~87% low) because tongue only visible during
licking") but Step 5 decision 13 was never updated to mention the epsilon override.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the behaviour clock, so a session-constant `vidshift` is computed
once from the bitcode pulse recorded by both streams — `sglx.bitcode.bitstart / sglx.fs`
minus `bp.ev.bitStart` — and each trial's frame times become
`frameTimes − vidshift − goCue[trial]`. The resulting frame-relative speed trace is then
linearly interpolated onto the shared 500-bin `time_axis`, so the stream lands on exactly
the same bins as the spikes. Two deviations from the reference `findVideoOffset.m`: the AI
uses `np.nanmedian` of the two bitcode series where the MATLAB uses `mode`, and it falls
back to a hard-coded `0.5 s` offset if the `sglx.bitcode` fields are missing or any
exception is raised. For trials whose `frameTimes` are entirely NaN it synthesises frame
times at 400 Hz with the same 0.5 s offset (this synthetic-frame-time fallback is what
`findPosition.m` does).

ii.
```python
def find_video_offset(data, fmt):
    """Find video offset (seconds) matching findVideoOffset.m.
    vidshift = mode(sglx.bitcode.bitstart) / sglx.fs - mode(bp.ev.bitStart)"""
            sglx_bitstart = bitcode['bitstart'][:].flatten()
            fs = float(sglx['fs'][0, 0])
            vid_offset = np.nanmedian(sglx_bitstart) / fs
            bp_offset = np.nanmedian(bp_bitstart[bp_bitstart > 0])
            return vid_offset - bp_offset
    return 0.5  # default offset
```
```python
        if len(frame_times) == 0 or np.all(np.isnan(frame_times)):
            frame_times = np.arange(n_frames) / 400.0
            ft_offset = 0.5
        else:
            ft_offset = vidshift
        # Align frame times to goCue
        aligned_ft = frame_times - ft_offset - align_times[trial_idx]
        ...
            interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. CONVERSION_NOTES Step 1: "findVideoOffset | funcs/findVideoOffset.m | PROCESSING |
Compute temporal offset between neural recording and video"; "Video offset: 0.5 sec
subtracted from frame times for synchronization (or computed via bitcode)". Step 3:
"DLC kinematics: interpolated from video time to neural time axis, video offset subtracted
(0.5s or computed via bitcode)".

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` DeepLabCut tracking, **bottom camera** (`view = 1`), feature
`'top_paw'` — the same single feature the human reference selects. The bottom camera's
other paw, `bottom_paw`, is not used. Alignment again needs `bp.ev.goCue` and the bitcode
fields.

ii.
```python
PAW = bottom cam, 'top_paw':
    paw_speed = extract_velocity_from_traj(
        data, fmt, view=1, feat_name='top_paw',
        ntrials=ntrials, align_times=align_times,
        time_axis=time_axis, vidshift=vidshift
    )
```

iii. CONVERSION_NOTES Step 2 lists "Bottom cam features: top_tongue, topleft_tongue,
bottom_tongue, bottomleft_tongue, top_paw, bottom_paw, lickport, jaw, top_nostril,
bottom_nostril". Step 5 decision 11: "Paw velocity: Use top_paw from bottom cam. Compute
speed = sqrt(xvel^2+yvel^2)."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The non-tongue branch of `extract_velocity_from_traj`, again following the reference
MATLAB: (1) x and y are smoothed with a **21-point causal Gaussian** (`mySmooth(...,21)`,
as in the reference's jaw/paw code); (2) `np.gradient` gives per-frame x and y velocity;
(3) the **median velocity is subtracted** from each as a baseline correction — this is the
reference's non-tongue branch of `findVelocity.m`; (4) speed = `sqrt(xvel² + yvel²)`;
(5) NaNs are filled with the nearest valid sample, and all-NaN trials are zero-filled;
(6) the trace is linearly interpolated onto the 500-bin axis. Because NaNs are propagated
by the convolution in step 1 and then nearest-filled in step 5, bins where DeepLabCut did
not track the paw receive a *fabricated* speed rather than being marked untracked (the human
reference leaves them NaN → `not visible`, 18.6% of bins). No cross-camera normalisation,
since only one view is involved; units remain pixels per frame.

ii.
```python
        if not is_tongue:
            xpos_smooth = causal_gaussian_smooth(xpos, 21, 'reflect')
            ypos_smooth = causal_gaussian_smooth(ypos, 21, 'reflect')
        xvel = np.gradient(xpos_smooth); yvel = np.gradient(ypos_smooth)
        # For non-tongue: subtract baseline velocity
        if not is_tongue:
            xvel = xvel - np.nanmedian(xvel)
            yvel = yvel - np.nanmedian(yvel)
        spd = np.sqrt(xvel**2 + yvel**2)
        # Fill missing with nearest
        nan_mask = np.isnan(spd)
        if np.any(nan_mask) and not np.all(nan_mask):
            valid = np.where(~nan_mask)[0]
            for idx in np.where(nan_mask)[0]:
                nearest = valid[np.argmin(np.abs(valid - idx))]
                spd[idx] = spd[nearest]
```

iii. Same reference-code basis as 7-b: "Applies causal smoothing via mySmooth() for
non-tongue features (tongue left unsmoothed)"; "For non-tongue features: subtracts baseline
velocity (median velocity)"; "Fills missing values with nearest neighbor interpolation"
(sub-agent report on `findPosition.m`/`findVelocity.m`, trajectory step 46).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A strict per-session median split: `np.nanpercentile(paw_speed_valid, 50)` over all
timepoints of all kept trials in the session, `0` below, `1` at or above. The `1e-10` floor
in `discretize_time_series` never engages here because paw speed is continuous and its
median is well above zero. There are only two classes; because untracked bins were
nearest-filled rather than left NaN (8-b), there is no `not visible` class, and every
session comes out at exactly 50.0% / 50.0%. The human reference obtains 40.7% / 40.7% /
18.6% not visible.

ii.
```python
    paw_thresh = np.nanpercentile(paw_speed_valid, 50)
    paw_disc = discretize_time_series(paw_speed_valid, paw_thresh)
...
    ['low', 'high'],             # paw_velocity: 0=<50th, 1=>=50th
```

iii. CONVERSION_NOTES Step 5 decision 13: "Discretization: Per-session 50th percentile
threshold. Values across ALL included timepoints of ALL included trials in the session.
Below median -> 0, >= median -> 1." Step 7: "Paw and motion energy well-balanced at 50/50
as expected from median split."

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue and via the same code path: the session-constant `vidshift`
from the bitcode is subtracted from `frameTimes`, then the trial's `goCue` time, and the
result is linearly interpolated onto the shared 500-bin axis. One difference from the human
reference: the AI passes `view = 1` explicitly and therefore uses the **bottom** camera's
own `frameTimes` for the paw (the reference locates the camera by feature name for the same
effect), so the one trial where the two cameras recorded different frame counts is handled
correctly.

ii.
```python
        ts, frame_times, feat_names = get_traj_data(data, fmt, view, trial_idx)
        ...
        aligned_ft = frame_times - ft_offset - align_times[trial_idx]
        ...
            interpolated = np.interp(time_axis, aligned_ft, spd)
            speed[:, trial_idx] = interpolated.astype(np.float32)
```

iii. Same as 7-d — a single `extract_velocity_from_traj` is used for both features, so the
alignment logic is shared by construction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file sitting beside each data structure,
read with `scipy.io.loadmat`; `obj.me` (present in only some sessions) is not used. The
loader handles the three layouts that occur across the 44 files: a bare cell array, a
`{data, moveThresh}` struct, and a `{data: {data, moveThresh}}` doubly-wrapped struct. It
yields one 400 Hz trace per trial. The side camera's `frameTimes` (`obj.traj{0}`) are the
second input, used for alignment.

ii.
```python
def load_motion_energy(dirpath, animal, date):
    me_file = os.path.join(dirpath, f'motionEnergy_{animal}_{date}.mat')
    if not os.path.exists(me_file):
        return None, None
    me_raw = me_mat['me']
    if hasattr(me_raw, 'dtype') and me_raw.dtype.names and 'data' in me_raw.dtype.names:
        me_data = me_raw['data'][0, 0]
        ...
        # Format 3: nested struct
        if hasattr(me_data, 'dtype') and me_data.dtype.names and 'data' in me_data.dtype.names:
            me_data = me_data['data'][0, 0]
    elif me_raw.dtype == np.dtype('O'):
        me_data = me_raw          # Format 2: direct cell array
```

iii. CONVERSION_NOTES Step 1: "loadMotionEnergy | DataLoadingScripts/loadMotionEnergy.m |
LOADING | Load motion energy, interpolate to neural time axis, align to event"; "Motion
energy: Loaded from separate files, interpolated to neural time axis at 400Hz original
rate". The three-layout handling mirrors the MATLAB's own guard
`if isstruct(me.data), me.data = me.data.data; end`; the format branches were added in
Steps 137/150 of the trajectory after real load failures.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment and resampling — the value is already one scalar per camera frame
(the paper reduces each frame's per-pixel motion to its 99th percentile upstream), so there
is nothing to smooth, differentiate or combine. The trace is truncated to
`min(len(me_trial), len(aligned_ft))` frames, linearly interpolated onto the 500-bin axis,
and any residual NaN column is nearest-filled or zero-filled. Note the AI **interpolates**
rather than bin-averaging, so with 4 frames per 10 ms bin it samples the trace at the bin
centre instead of averaging within the bin; this follows `loadMotionEnergy.m`, which also
uses `interp1`.

ii.
```python
            n_frames = min(len(me_trial), len(aligned_ft))
            me_interp[:, trial_idx] = np.interp(
                time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
            ).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 decision 12: "Motion energy: Load from motionEnergy files,
interpolate to neural time axis, align to goCue", matching the MATLAB's "trim trial length
(me.data contains motion energy for each time point in trial at 400 Hz). Want to align to
params.alignEvent and want to put it in same dt as neural data."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A strict per-session median split, exactly as for the paw: the 50th percentile of all
timepoints of all kept trials, `0` below, `1` at or above, labelled `['low', 'high']`. Every
session lands at 50.0% / 50.0%. There is no `no video` class; it would only be needed for a
session with no motion-energy file, and all 44 have one (had one been missing, the AI's
fallback fills the session with zeros, which would collapse it into class 0 rather than
flagging it). The human reference obtains 47.9% / 48.3% / 3.8% no-video, so the missing
third class affects a small fraction of bins here.

ii.
```python
    me_thresh_50 = np.nanpercentile(me_valid, 50)
    me_disc = discretize_time_series(me_valid, me_thresh_50)
...
    ['low', 'high'],             # motion_energy: 0=<50th, 1=>=50th
```
```python
    else:
        me_valid = np.zeros((len(time_axis), len(valid_trial_indices)), dtype=np.float32)
```

iii. CONVERSION_NOTES Step 5 decision 13 (the common discretisation rule) and Step 9's
distribution table, "motion_energy | low: 50.0% | high: 50.0%". The AI notes the
`moveThresh` stored in the motion-energy files was read but deliberately not used, since the
task prescribes a 50th-percentile threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy has one value per **side-camera** frame, so `interpolate_motion_energy`
fetches `get_traj_data(..., view=0, trial_idx)` purely to obtain that camera's
`frameTimes`, applies the same session `vidshift` and the trial's `goCue`, and interpolates
onto the shared 500-bin axis. Same fallbacks as the kinematics: synthetic 400 Hz frame times
with a 0.5 s offset when `frameTimes` are missing or the lookup raises. This is the same
camera and same offset the human reference uses.

ii.
```python
        ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)  # side cam
        if len(frame_times) == 0 or np.all(np.isnan(frame_times)):
            frame_times = np.arange(len(me_trial)) / 400.0
            ft_offset = 0.5
        else:
            ft_offset = vidshift
        # Align to goCue
        aligned_ft = frame_times - ft_offset - align_times[trial_idx]
```

iii. Same basis as 7-d/8-d; CONVERSION_NOTES Step 3: "Motion energy: loaded from separate
files, interpolated to neural time axis, aligned to goCue."

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI's uniform strategy is **fill and continue** rather than mark-and-keep:
- *Missing frame times* (`frameTimes` empty or all-NaN): synthetic times at 400 Hz with a
  0.5 s offset are fabricated, following `findPosition.m`.
- *Untracked DeepLabCut frames* (x/y NaN): for the tongue, velocity is set to 0; for the
  paw, NaNs are filled from the nearest valid frame. Either way the bin is given a numeric
  speed rather than a `not visible` label.
- *Entirely untracked trials*: the whole column is set to 0.
- *Missing motion-energy file*: the whole session's ME is set to 0.
- *Missing `stim.enable`*: assumed all-false.
- *Missing bitcode*: `vidshift` defaults to 0.5 s.
- *Trials past the end of the ephys recording*: detected from the largest `clu.trial` and
  dropped (the one case handled by exclusion rather than filling).
- *Three motion-energy file layouts* and *two `.mat` formats* are branch-handled; the v5
  `ts` orientation is guessed from `ts.shape[2] > ts.shape[0]` rather than by comparing
  against `len(featNames)`.
Several of these sit behind bare `except: continue` / `except: pass` handlers, so a genuine
read failure for one trial silently leaves that trial at its default value. The human
reference instead preserves NaN throughout and lets the `not visible` class carry the
missingness.

ii.
```python
        if len(frame_times) == 0 or np.all(np.isnan(frame_times)):
            frame_times = np.arange(n_frames) / 400.0
            ft_offset = 0.5
```
```python
    # Fill remaining NaN columns with zeros
    nan_cols = np.all(np.isnan(speed), axis=0)
    speed[:, nan_cols] = 0.0
    # Fill any remaining NaNs with nearest
    for t in range(ntrials):
        col = speed[:, t]
        nan_mask = np.isnan(col)
        if np.any(nan_mask) and not np.all(nan_mask):
            valid = np.where(~nan_mask)[0]
            for idx in np.where(nan_mask)[0]:
                nearest = valid[np.argmin(np.abs(valid - idx))]
                col[idx] = col[nearest]
```
```python
    try:
        stim_enable = get_bp_field(data, fmt, 'stim.enable').astype(bool)
    except Exception:
        stim_enable = np.zeros(ntrials, dtype=bool)
```

iii. The filling behaviour is justified as fidelity to the reference MATLAB — Step 1:
"findPosition ... Handles missing frame times by creating synthetic frame times (400 Hz
sampling). Fills remaining NaNs with nearest neighbor interpolation"; Step 5 decision 10:
"When tongue not visible, velocity = 0 (as per findVelocity code)." The
beyond-recording exclusion is justified empirically in Step 10: "Sessions 36
(JEB24_2023-10-23) and 43 (JEB24_2023-11-03) had trials beyond spike recording. Fixed by
detecting max trial in spike data and excluding trials beyond."

## 11-a. What are the most time-consuming steps of the code?

i. The script instruments and prints a per-step timing breakdown per session. Summing the
full run (`conversion_full_out.txt`, 266.7 s total for 44 sessions): **paw velocity
extraction 83.0 s**, **spike binning + smoothing 59.3 s**, **tongue velocity 45.0 s**,
**motion energy 42.4 s**, with the remaining ~37 s in file loading, the low-FR filter and
output assembly. So the dominant cost is the per-trial camera processing (170 s combined),
not file I/O — the opposite of the human reference, whose runtime (135 s) is dominated by
reading the `.mat` files. Within the camera path the expensive pieces are the O(n²)
nearest-neighbour NaN fill loops and the fact that `get_traj_data` re-reads and
re-dereferences `featNames` and the full `ts` array from HDF5 once per trial per feature.
The AI's own notes do not identify paw extraction as the bottleneck.

ii.
```python
    t1 = time.time()
    trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
    print(f"    Spike binning: {time.time()-t1:.1f}s")
    ...
    print(f"    Tongue velocity: {time.time()-t2:.1f}s")
    ...
    print(f"    Paw velocity: {time.time()-t3:.1f}s")
    ...
    print(f"    Motion energy: {time.time()-t4:.1f}s")
```

iii. CONVERSION_NOTES Step 7: "| None needed - 7s/session | N/A |" and "| Total | ~7-9s |
~5-7 min for 44 sessions |" — i.e. the AI judged the runtime comfortably inside the
instructions' 15-minute budget and therefore did not pursue optimisation. Step 9 confirms
"Total: 269 seconds (~6.1s/session average)".

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Four candidates, none of which were vectorised:
- `bin_and_smooth_spikes` runs a nested Python loop over **clusters × all `Ntrials`**,
  issuing one `np.histogram` and one `np.convolve` per cell-trial (~20,000 calls per
  session). This is the clearest case: the human reference replaces the entire trial loop
  with a single `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, BIN_EDGES])`
  per cluster, and smooths all trials at once with `gaussian_filter1d(..., axis=1)`.
- The nearest-neighbour NaN fill in `extract_velocity_from_traj` and
  `interpolate_motion_energy` is a Python loop over every NaN index performing an
  `argmin` over all valid indices — quadratic where `np.interp` or a
  forward/backward-fill would be linear.
- `causal_gaussian_smooth` rebuilds its kernel from scratch on every call.
- The per-trial loops in `extract_velocity_from_traj` / `interpolate_motion_energy`
  genuinely cannot be fully vectorised, since each trial has a different number of camera
  frames (the human reference keeps these as loops for the same reason).

ii.
```python
    for i, clu in enumerate(clusters):
        trialtm = clu['trialtm']; trial = clu['trial']
        for j in range(ntrials):
            spike_mask = trial == trial_num
            ...
            counts = np.histogram(aligned_times, bins=edges)[0]
            smoothed = causal_gaussian_smooth(rate, SMOOTH_N, SMOOTH_BC)
```
```python
            for idx in np.where(nan_mask)[0]:
                nearest = valid[np.argmin(np.abs(valid - idx))]
                spd[idx] = spd[nearest]
```

iii. CONVERSION_NOTES Step 6 asserts the opposite of what the code does: "Code
inefficiencies identified: Spike binning loops over neurons and trials (**vectorized with
np.histogram**); DLC extraction loops over trials (necessary due to variable frame times)"
and "Code speedups added: np.histogram for spike binning (vectorized); Float32 for arrays to
reduce memory". Using `np.histogram` inside the loop is not vectorisation of the loop. The
`float32` claim is accurate.

## 11-c. What processing does the code repeat multiple times?

i. Several things are recomputed:
- **`get_traj_data` is called once per trial per stream.** Each call re-opens the camera
  group, re-dereferences and re-decodes the full `featNames` list character by character
  (`h5_deref_string`), and reads the entire `(features, 3, frames)` `ts` array — then uses
  one feature. The side camera is walked twice per session (once for the tongue, once for
  the motion-energy frame times) and the bottom camera once, so the same HDF5 reads and
  the same `featNames` parse happen ~3 × `Ntrials` times per session.
- **`causal_gaussian_smooth` regenerates its Gaussian kernel on every invocation**
  (~20,000 times per session for the spikes alone).
- **Spikes are binned and smoothed for all `Ntrials`**, including the photostim, early-lick
  and beyond-recording trials that are discarded immediately afterwards (~10% of trials).
- Conversely, `vidshift` *is* correctly computed once per session and reused, and each
  `.mat` file is opened only once.
The human reference recomputes nothing: each file is read once, the offset once per
session, the per-feature velocity once and reused by both the binning and the percentile.

ii.
```python
def get_traj_data(data, fmt, view, trial_idx):
        # Get feature names (from first trial)
        fn_ref = cam['featNames'][0, 0]
        fn_data = f[fn_ref]
        feat_names = []
        for i in range(fn_data.shape[1]):
            name = h5_deref_string(f, fn_data[0, i])
            feat_names.append(name)
        ts_ref = cam['ts'][trial_idx, 0]
        ts = f[ts_ref][:]
```
```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    t = np.arange(N); mu = (N - 1) / 2; sigma = N / 6
    kernel = np.exp(-0.5 * ((t - mu) / sigma) ** 2)   # rebuilt on every call
```
```python
    trialdat = bin_and_smooth_spikes(all_clusters, ntrials, ...)   # all Ntrials
    ...
    trialdat_valid = trialdat[:, :, valid_trial_indices]           # then subset
```

iii. The AI documented only the second-order point ("DLC extraction loops over trials
(necessary due to variable frame times)") and did not identify the repeated `featNames`
decoding or the repeated `ts` reads. Its justification for not pursuing this is the runtime
budget (Step 7: "None needed - 7s/session").

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest amounts, in four places:
- **Neural data for discarded trials**: spikes are binned and smoothed for every trial of
  the session, then ~10% are thrown away by the trial filter. The low-FR unit filter is
  also computed on that full array, so it is influenced by trials the dataset never
  contains.
- **Whole-array reads for one feature**: `get_traj_data` materialises all ~7–10 tracked
  features' x/y/likelihood per trial when one feature is needed, and the likelihood channel
  is read but never used (the AI relies on the authors' pre-applied NaN instead).
- **Unused values**: `obj.bp.L` and `obj.bp.miss` are loaded and subset per trial but never
  enter any output; `me_thresh` (the authors' `moveThresh`) is parsed and returned but
  discarded; `ts` is fetched inside `interpolate_motion_energy` purely for its side effect
  of also returning `frame_times`.
- **Continuous streams discretised away**: the full-resolution tongue/paw/ME traces are
  computed and interpolated, then collapsed to a single bit per bin. This is unavoidable —
  the percentile threshold needs the continuous values — and the human reference does the
  same.
The only genuinely wasteful item beyond what the reference also does is the first: binning
spikes for trials that are then dropped.

ii.
```python
L = get_bp_field(data, fmt, 'L').astype(bool)      # never used
...
L_valid = L[valid_trial_indices]                   # never used
```
```python
    me_data, me_thresh = load_motion_energy(dirpath, animal, date)   # me_thresh unused
```
```python
        ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)  # ts unused here
```

iii. Not discussed in CONVERSION_NOTES; the AI's efficiency discussion stops at Step 6/7
with the conclusion that no optimisation was needed given the ~6 s/session runtime.
