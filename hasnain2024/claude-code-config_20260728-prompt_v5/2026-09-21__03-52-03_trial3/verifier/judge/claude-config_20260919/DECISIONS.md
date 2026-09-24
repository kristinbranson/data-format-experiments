# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of 44 sessions (`SESSION_META`), each entry `(animal, date, probes_matlab_indexed, data_dir_key)`, transcribed from the authors' per-animal loading scripts in `code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`. The two ephys folders (`Ephys_Behavior`, `RandomizedDelay_Ephys_Behavior`) are mapped by the `data_dir_key`; the two behavior-only folders are not used. Sessions commented out in the loading scripts (e.g. JEB23 2023-10-20) and animals with loading scripts but no data (JEB4, JEB5) are omitted. Each session is one `data_structure_<anm>_<date>.mat`; motion energy comes from the sibling `motionEnergy_<anm>_<date>.mat`. Both MATLAB formats are supported: the file header is sniffed for `7.3` and dispatched to an `h5py` reader (`load_session_h5`) or a `scipy.io.loadmat` reader (`load_session_v5`). The HDF5 reader is partially lazy — spike and behavior fields are read eagerly, but trajectory data is left as open HDF5 references and read per trial on demand; the file handle is closed by `close_session` at the end of the session.

ii.
```python
SESSION_META = [
    # Ephys_Behavior sessions
    ('EKH1', '2021-08-07', [2], 'ephys'),
    ...
    # JEB23 2023-10-20 is commented out in loading script
    ('JEB23', '2023-10-21', [1], 'randdelay'),
    ...
    ('JEB24', '2023-11-03', [1], 'randdelay'),
]

DATA_DIRS = {
    'ephys': '/app/data/Ephys_Behavior/',
    'randdelay': '/app/data/RandomizedDelay_Ephys_Behavior/',
}
```

```python
def load_session(fpath):
    """Load session, auto-detecting format."""
    with open(fpath, 'rb') as ff:
        header = ff.read(15)
    if b'7.3' in header:
        return load_session_h5(fpath)
    else:
        return load_session_v5(fpath)
```

```python
for idx, (anm, date, probes, ddir) in enumerate(sessions_to_process):
    print(f"[{idx+1}/{len(sessions_to_process)}] ", end='')
    result = process_session(anm, date, probes, ddir, ...)
    if result is not None:
        all_results.append(result)
```

iii. From CONVERSION_NOTES.md Step 1/4: "Note: JEB4 and JEB5 have loading scripts but NO data files in the data directory. Note: Some sessions in data dir are commented out in loading scripts (excluded)"; "RandDelay sessions | Loading scripts: 19 active | 20 readable sessions | 19 sessions | JEB23_10-20 commented out in script. Exclude it." Key Decision 1: "Include all sessions from both Ephys and RandomizedDelay: Both contain neural+behavior data. Use loading scripts to determine which sessions/probes. Total: ~44 sessions." Step 2 notes both file formats exist ("MATLAB v7.3 (HDF5) and v5.0 (scipy-readable)") and that `JEB24_2023-10-03/10-04` have no `clu` field and are behavior-only.

## 1-b. How are the data split into subjects?

i. The animal ID is the first element of each `SESSION_META` tuple (it is also the prefix of the filename). It is carried on each session result as `r['anm']`. At assembly, `subjects` is the sorted set of unique animal IDs over the sessions that survived filtering, and `subject_idx` is each session's index into that list. This yields 14 subjects over the 42 retained sessions.

ii.
```python
subjects = sorted(set(r['anm'] for r in all_results))
subject_idx = np.array([subjects.index(r['anm']) for r in all_results])
```

iii. The AI never states an explicit rationale beyond the session table; it takes the animal identity from the authors' loading scripts / filenames rather than from any field inside the `.mat` file. CONVERSION_NOTES Step 1 records "loadXXX_ALMVideo | DataLoadingScripts/Recording and video/ | LOADING | Per-animal session metadata (animal, date, probe number)", i.e. the loading scripts are treated as the authoritative record of animal identity.

## 1-c. How are the data split into sessions?

i. One session = one entry of `SESSION_META` = one `data_structure_*.mat` file, processed by one call to `process_session`, which returns one element each of `neural`, `input`, and `output`. Which folder to look in is given explicitly by the `data_dir_key`, so fixed-delay and randomized-delay sessions are pooled into one list rather than treated as two datasets. Two sessions are then **dropped** at the session level (see 1-e), leaving 42 of 44: 23 fixed-delay + 19 randomized-delay.

ii.
```python
data_dir = DATA_DIRS[data_dir_key]
fpath = os.path.join(data_dir, f'data_structure_{anm}_{date}.mat')
if not os.path.exists(fpath):
    print(f"  File not found: {fpath}")
    return None
```

iii. Same as 1-a: the session set follows the authors' loading scripts. Both task variants are included because, per Step 5 Key Decision 1, "Both contain neural+behavior data."

## 1-d. How are the data split into trials?

i. A trial is one index into the per-trial Bpod arrays; `ntrials = obj.bp.Ntrials` and every behavioural field (`hit`, `miss`, `no`, `R`, `L`, `autowater`, `early`, `stim.enable`, `ev.goCue`) is read as a length-`ntrials` vector. Spikes carry a 1-based `clu.trial` label, so per-trial spike sets are obtained by `trial_ids == t + 1`; trajectory data is indexed per trial by the same 0-based index (`traj_grp['ts'][trial_idx, 0]`). No trial boundaries are reconstructed. The arrays are used at their native length rather than being truncated to `Ntrials` (I verified across all 44 files that every one of these fields, plus `ev.goCue`, `ev.bitStart` and the per-camera trajectory arrays, already has exactly `Ntrials` entries, so no truncation is needed).

ii.
```python
session['ntrials'] = int(bp['Ntrials'][0, 0])
session['hit'] = bp['hit'][0, :].astype(bool)
...
session['goCue'] = ev['goCue'][0, :]
```

```python
for t in range(ntrials):
    trial_num = t + 1  # MATLAB 1-indexed
    spike_mask = trial_ids == trial_num
    if not np.any(spike_mask):
        continue
    aligned_times = trialtm[spike_mask] - go_cue[t]
```

iii. CONVERSION_NOTES Step 1 documents the structure as "`obj.bp` - Bpod/behavior data: Ntrials, hit, miss, no (ignore), R (right), L (left), autowater, early, stim.enable" and "`obj.clu{probe}(unit)` - Spike data: .tm, .trialtm, .trial, .quality", i.e. the trial index is already in the data and needs no inference.

## 1-e. How are trials filtered based on quality controls?

i. Only **one** trial-level filter is applied: photostimulation trials are removed (`~stim.enable`). Early-lick trials, autowater trials, miss trials and ignore trials are all **kept**, by explicit decision, to maximise decoder training data. There are two *session*-level filters: (a) after neuron curation a session must have ≥ 10 units (`MIN_UNITS`), and (b) the authors' inclusion criterion — ≥ 40 right-hit **and** ≥ 40 left-hit DR trials (computed on `~stim & ~autowater & ~early` trials) — must be met. The latter dropped JEB19 2023-04-19 and 2023-04-20; the former dropped nothing. A session also needs ≥ 2 surviving trials. No filter removes trials that fall past the end of the ephys recording: the verifier reported 64 all-zero-neural trials at the end of sessions 25, 34 and 41, which the AI inspected and knowingly retained.

ii.
```python
# --- 4. Check trial inclusion criteria ---
valid_mask = ~session['stim_enable'] & ~session['autowater'] & ~session['early']
r_hit = session['R'] & session['hit'] & valid_mask
l_hit = session['L'] & session['hit'] & valid_mask
if n_r_hit < 40 or n_l_hit < 40:
    print(f"insufficient DR trials (R-hit={n_r_hit}, L-hit={n_l_hit}), skipping")
    close_session(session)
    return None

# --- 5. Exclude stim-enabled trials ---
trial_mask = ~session['stim_enable']
trial_indices = np.where(trial_mask)[0]
if len(trial_indices) < 2:
    print(f"too few trials after stim exclusion, skipping")
    ...
```
```python
if n_neurons < MIN_UNITS:
    print(f"{n_neurons} units after filtering (< {MIN_UNITS}), skipping")
```

iii. CONVERSION_NOTES Step 5 Key Decisions 6–8: "**Trial inclusion for decoder**: Include ALL trials (hit, miss, no/ignore, early, autowater) to maximize training data. The decoder outputs classify these properties."; "**Exclude stim-enabled trials**: These have photoinactivation and are experimental manipulations."; "**Session inclusion**: After filtering neurons (quality + FR), require >= 10 neurons. Also apply >= 40 R-hit + >= 40 L-hit DR trial filter per paper." The session criteria are traced to `UseInclusionCritera.m`/`RemoveUnwantedSessions.m` and to the paper's "at least 10 units". On the empty trials, Step 10 says: "Zero-neural-data trials: 64 trials across 3 sessions at session ends (<0.5%) — acceptable, recording ended early."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` spike clusters — specifically `clu.trialtm` (spike time relative to trial start, on the behaviour clock), `clu.trial` (1-based trial of each spike) and `clu.quality` (manual curation label) — together with `obj.bp.ev.goCue`, which defines the alignment. Only the probes named in `SESSION_META` are read; units from both probes of a two-probe session are concatenated. `obj.ex.probe.loc` is read separately to label each unit's brain region.

ii.
```python
n_units = probe_grp['quality'].shape[0]
units = []
for i in range(n_units):
    q_str = read_h5_str(f, probe_grp['quality'][i, 0]).strip()
    trialtm = np.array(f[probe_grp['trialtm'][i, 0]]).flatten()
    trial = np.array(f[probe_grp['trial'][i, 0]]).flatten().astype(int)
    units.append({'quality': q_str, 'trialtm': trialtm, 'trial': trial})
session['clu'].append(units)
```
```python
for pi in probe_indices:
    probe_units = session['clu'][pi]
    good_idx = filter_clusters(probe_units)
    for gi in good_idx:
        all_units.append(probe_units[gi])
        all_unit_probe.append(pi)
```

iii. CONVERSION_NOTES Step 5 variable mapping: "obj.clu spike times | neural | Bin at 10ms, smooth, align to goCue | getSeq, alignSpikes, mySmooth". Step 2: "Loading scripts specify which probe(s) correspond to ALM recordings ... Must use obj.ex.probe.loc to determine brain region for each probe."

## 2-b. How is the `neural` data processed?

i. Per unit and per trial: spike times are aligned to the go cue, counted with `np.histogram` into the 10 ms edge grid spanning −2.5 to 2.5 s, divided by `DT` to give Hz, and then smoothed along time with a **causal** Gaussian kernel intended to reproduce `utils/mySmooth.m` — `gausswin(15)` (alpha = 2.5) with the first half zeroed and renormalised, applied with a `reflect` boundary condition. No normalisation, baseline subtraction or z-scoring is applied; stored values are firing rates in Hz (`float32`). Output per trial is transposed to `(n_neurons, n_timepoints)`.

**The smoothing implementation does not do what it intends.** `smooth_signal` prepends a 15-sample reflected pad, takes `np.convolve(..., mode='full')` and then slices `result[n : n+len(x)]` with `n = 15`. MATLAB's `conv(x, kern, 'same')` followed by trimming the pad corresponds to `full[22 : 22+T]`, so the AI's slice is 7 samples early, which **delays the output by 7 bins = 70 ms**. I verified this empirically: a unit impulse at bin 30 produces a response occupying bins 37–44 (peak at 37), whereas the MATLAB-equivalent slice produces bins 30–37 (peak at 30). So the stored rate at time *t* is built only from spikes in [*t*−140 ms, *t*−70 ms] and never includes the current bin; the neural stream sits ~70 ms later than the inputs and outputs it is meant to be aligned with.

ii.
```python
def causal_gaussian_kernel(n):
    alpha = 2.5  # MATLAB default
    half = (n - 1) / 2
    t = np.arange(n) - half
    kern = np.exp(-0.5 * (alpha * t / half) ** 2)
    kern[:n // 2] = 0          # Make causal: zero out first half
    kern /= kern.sum()
    return kern

def smooth_signal(x, kernel=SMOOTH_KERNEL, bctype=BC_TYPE):
    n = len(kernel)
    if bctype == 'reflect':
        pad = x[:n][::-1]
        x_padded = np.concatenate([pad, x])
        result = np.convolve(x_padded, kernel, mode='full')
        result = result[n:n + len(x)]     # <-- should be result[n + (n-1)//2 : ...]
```
```python
counts, _ = np.histogram(aligned_times, bins=EDGES)
fr = counts.astype(np.float64) / DT
trialdat[:, ui, t] = smooth_signal(fr)
```

iii. CONVERSION_NOTES Step 3/6: "Bin spikes in 10ms bins, divide by dt to get firing rate (Hz); Smooth with causal Gaussian kernel (15-point window, reflect boundary condition)"; "Causal Gaussian smoothing matching mySmooth.m (15-point window, reflect BC)". This follows the reference `getSeq.m` line `obj.trialdat{prbnum}(:,i,j) = mySmooth(N./params.dt, params.smooth, params.bctype)` and the AI's own summary of `mySmooth.m`: "Creates causal Gaussian kernel: `kern = gausswin(N)` then zeros out first half (causal) ... Convolves with 'same' option". The 70 ms slicing offset is never mentioned or checked.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, matching the reference `params.quality = {'all'}` path. (1) Clusters whose (whitespace-stripped, lower-cased) `quality` label is one of `garbage`, `gabrga`, `noisy`, `real?` are dropped; everything else, including multi-units and unlabelled clusters, is kept. (2) After binning and smoothing, units whose mean rate over the whole −2.5 to 2.5 s window and over **all** trials (including the photostim trials that are later removed) is ≤ 1 Hz are dropped. Note the ordering: binning and smoothing are run on every quality-passing unit and only then is the rate threshold applied. Result: 2,354 units over 42 sessions, 17–141 per session.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0

def filter_clusters(units, excluded_qualities=EXCLUDED_QUALITIES):
    good_idx = []
    for i, u in enumerate(units):
        q = u['quality'].lower()
        if q not in excluded_qualities:
            good_idx.append(i)
    return good_idx
```
```python
# --- 3. Remove low FR clusters ---
mean_frs = trialdat.mean(axis=(0, 2))
keep_mask = mean_frs > LOW_FR
trialdat = trialdat[:, keep_mask, :]
```

iii. CONVERSION_NOTES Step 3 "Neuron curation rules": "1. Exclude clusters with quality = 'garbage', 'gabrga', 'noisy', 'real?' (quality='all' mode). 2. Remove units with mean FR <= 1 Hz (across all trials and conditions)." Step 4 resolves the threshold discrepancy: "lowFR threshold | getDefaultParams: 0.5 | ... | 'exceeding 1 Hz' | All analysis scripts override to 1 Hz. Use 1 Hz." The drop list is copied verbatim from `findClusters.m` (`~ismember(qualityList,'garbage') & ~ismember(...,'gabrga') & ~ismember(...,'noisy') & ~ismember(...,'real?')`), and the FR rule from `removeLowFRClusters.m`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By a single subtraction, exactly as in `alignSpikes.m`: each spike's `trialtm` (already relative to its own trial's start, on the behaviour clock) minus `goCue` of that trial. The aligned times are then histogrammed into `EDGES`, a fixed grid from −2.5 to +2.5 s; spikes outside the window fall outside the outermost edges and are dropped. Every trial therefore has the same 500-bin axis centred on the go cue. (Caveat: the smoothing bug documented in 2-b shifts the *stored* rates 70 ms later than this alignment implies.)

ii.
```python
aligned_times = trialtm[spike_mask] - go_cue[t]
counts, _ = np.histogram(aligned_times, bins=EDGES)
```
```python
ALIGN_EVENT = 'goCue'
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. CONVERSION_NOTES Step 1: "alignSpikes | ... | PROCESSING | Aligns spike times to event (goCue, moveOnset, firstLick, etc.)" and "Key Parameters: `params.alignEvent = 'goCue'`". Step 3 Processing Details: "Align to go cue (params.alignEvent = 'goCue')". Step 10 Check 2: "Time axis: -2.495 to 2.495 in 10ms steps (500 bins), centered on go cue — PASS".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms (`DT = 1/100`), 500 bins spanning −2.5 to +2.5 s; `metadata['time_bin_size'] = 10.0` ms. Spikes are binned directly at 10 ms from raw spike times (no rebinning of an intermediate finer grid). The bin *centres* (`EDGES[:-1] + DT/2`, i.e. −2.495 … 2.495) form the shared time axis. All other streams are resampled onto this same axis: the 400 Hz video/DLC and motion-energy traces are linearly interpolated (`np.interp`) onto the bin centres — a downsample by point-sampling rather than by averaging within bins.

ii.
```python
TMIN = -2.5  # seconds
TMAX = 2.5   # seconds
DT = 1.0 / 100  # 10 ms bins
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_AXIS)
```
```python
interp_speed = np.interp(TIME_AXIS, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "dt (bin size) | getDefaultParams: 1/200, most scripts: 1/100 | ... | Use 1/100 (10ms) as most analysis scripts do", and Step 5 Key Decision 3: "**dt = 10ms (1/100)**: Most analysis scripts use this; consistent with paper." I confirmed from the trajectory's grep of the reference repo that `params.dt = 1/100` does appear in the large majority of analysis scripts (Figure 3a–i, Figure 8a–c, EDFigure 2/3, ParallelAnalysis, Behavior, `WorkingWithDataObjs.m`) while `1/200` appears in `getDefaultParams.m` and a handful of scripts. The time-axis construction copies `getSeq.m` (`edges = tmin:dt:tmax; obj.time = edges + dt/2; obj.time = obj.time(1:end-1)`), and the interpolation of video onto that axis copies `findPosition.m`/`loadMotionEnergy.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. None — it is not derived from a raw variable. It is the module-level bin-centre vector `TIME_AXIS`, defined by the chosen window (−2.5 to 2.5 s) and bin size (10 ms) around the go cue, and is byte-identical for every trial of every session.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```
```python
input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. CONVERSION_NOTES Step 5 mapping: "Time axis | input[0]: time_from_go_cue | Continuous, seconds | getSeq edges | Ranges -2.5 to 2.5". The window is the reference's `params.tmin`/`params.tmax`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the axis and casting to `float32`. The same `(1, 500)` array is appended once per trial (each trial holds a separate reference to the same reshaped array object). The AI did not use the alternative representation suggested by the instructions for time-like inputs (a binary event time series); it kept the continuous ramp, which is the literal reading of "Time from go cue onset in seconds (continuous, time-varying)".

ii.
```python
for i in range(n_trials_out):
    input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. Not separately justified in CONVERSION_NOTES beyond the mapping table; the decoder-input spec names the variable as continuous, and the verifier confirms `Input range: time_from_go_cue: [-2.5, 2.5]`.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid: `TIME_AXIS` holds the centres of the very `EDGES` used to histogram the go-cue-aligned spike times, so input bin *k* and neural bin *k* denote the same 10 ms interval. The only caveat is the 70 ms lag that the smoothing bug (2-b) introduces into the neural stream, which the input axis does not share.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
...
counts, _ = np.histogram(aligned_times, bins=EDGES)   # neural uses EDGES
...
input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))  # input uses centres of EDGES
```

iii. Implicit — the shared grid is copied from `getSeq.m`, which defines `obj.time` from the same `edges` used to bin spikes. Step 10 Check 2 verifies the axis is "centered on go cue — PASS".

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `obj.bp.R`, `obj.bp.L`, `obj.bp.hit` and `obj.bp.miss`. `obj.bp.no` is loaded but not used for this output — the "none" class is the fall-through case.

ii.
```python
session['hit'] = bp['hit'][0, :].astype(bool)
session['miss'] = bp['miss'][0, :].astype(bool)
session['R'] = bp['R'][0, :].astype(bool)
session['L'] = bp['L'][0, :].astype(bool)
```

iii. CONVERSION_NOTES Step 5 mapping: "obj.bp.R, obj.bp.L, obj.bp.no | output[0]: lick_direction | Categorical: left/right/none | findTrials | Per-trial".

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A per-trial relabelling, held constant across all 500 bins. If the trial was a hit **or** a miss (i.e. the animal responded) the class is the **instructed** side: `R` → right (1), `L` → left (0). Otherwise the class is none (2). Codes: left 0, right 1, none 2; `output_values[0] = ['left', 'right', 'none']`.

This encodes the *instructed* direction, not the direction actually licked. On a **miss** trial the animal licked the port opposite the instructed side, so those trials carry the wrong label. Miss trials are 12.5% of the converted dataset, so roughly one in eight trials has an inverted lick-direction label. The AI's own README states this openly: "lick_direction | left, right, none | Direction of **instructed** lick (none = ignore trial)".

ii.
```python
lick_dir = np.zeros(len(trial_indices), dtype=np.int8)
for i, ti in enumerate(trial_indices):
    if session['R'][ti] and (session['hit'][ti] or session['miss'][ti]):
        lick_dir[i] = 1  # right
    elif session['L'][ti] and (session['hit'][ti] or session['miss'][ti]):
        lick_dir[i] = 0  # left
    else:
        lick_dir[i] = 2  # none (no response / ignore)
```
```python
out[0, :] = lick_dir[i]
```

iii. No rationale is given for equating instructed side with licked side; the AI only documents the mapping. Step 10 Check 4 verifies the "none" class: "Ignore/no-lick consistency: 100% match between ignore trials and no-lick trials — PASS." Step 7/9 record the resulting distribution (left 0.444, right 0.436, none 0.120).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The single per-trial flag `obj.bp.autowater`.

ii.
```python
session['autowater'] = bp['autowater'][0, :].astype(bool)
```

iii. CONVERSION_NOTES Step 5 mapping: "obj.bp.autowater | output[1]: behavioral_context | Categorical: WC/DR | findTrials | Per-trial, autowater=1 → WC". Step 1 records the reference's condition strings, which use `autowater` to separate the two contexts.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct per-trial relabel, broadcast across all 500 bins: `autowater` → WC (0), otherwise DR (1). `output_values[1] = ['WC', 'DR']`, matching the code order requested in the decoder spec.

ii.
```python
context = np.zeros(len(trial_indices), dtype=np.int8)
for i, ti in enumerate(trial_indices):
    if session['autowater'][ti]:
        context[i] = 0  # WC
    else:
        context[i] = 1  # DR
...
out[1, :] = context[i]
```

iii. Step 2/3 of CONVERSION_NOTES describe the two-context design ("Two-context task (delayed-response + water-cued) ... alternated block-wise within sessions"); the `autowater` flag is the field the reference code uses to separate them. Resulting distribution WC 0.091 / DR 0.909 is recorded in Step 9.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `obj.bp.hit` and `obj.bp.miss`. `obj.bp.no` is loaded but not consulted — a trial that is neither hit nor miss falls through to "ignore".

ii.
```python
session['hit'] = bp['hit'][0, :].astype(bool)
session['miss'] = bp['miss'][0, :].astype(bool)
session['no'] = bp['no'][0, :].astype(bool)   # loaded, unused
```

iii. CONVERSION_NOTES Step 5 mapping: "obj.bp.hit, miss, no | output[2]: outcome | Categorical: correct/incorrect/ignore | getOutcome | Per-trial".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way per-trial relabel, broadcast across all 500 bins: hit → correct (1), miss → incorrect (0), everything else → ignore (2). `output_values[2] = ['incorrect', 'correct', 'ignore']`, matching the code order in the decoder spec. Ignore trials are retained as a class rather than dropped (the paper excludes them from its analyses).

ii.
```python
outcome = np.zeros(len(trial_indices), dtype=np.int8)
for i, ti in enumerate(trial_indices):
    if session['hit'][ti]:
        outcome[i] = 1  # correct
    elif session['miss'][ti]:
        outcome[i] = 0  # incorrect
    else:
        outcome[i] = 2  # ignore
...
out[2, :] = outcome[i]
```

iii. Step 5 Key Decision 6 (keep all trials "to maximize training data. The decoder outputs classify these properties") covers retaining ignore trials. Step 10 Check 4 cross-validates the ignore class against the no-lick class (100% agreement). Distribution: incorrect 0.125 / correct 0.755 / ignore 0.120.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side** camera only — using the feature literally named `tongue`, plus `obj.traj{1}(trial).frameTimes`, the session video offset from `obj.sglx.bitcode.bitstart`/`obj.sglx.fs` and `obj.bp.ev.bitStart`, and `obj.bp.ev.goCue`. The bottom camera's `top_tongue`/`bottom_tongue` views are not used. `ts` gives x, y and DLC likelihood per frame; the array orientation differs between the two MATLAB formats and is handled per branch.

ii.
```python
view_data = session['traj'][0]  # side cam
feat_names = view_data.get('featNames', [])
tongue_idx = None
for fi, fn in enumerate(feat_names):
    if fn.lower() == 'tongue':
        tongue_idx = fi
        break
```
```python
if is_h5:
    x = ts[tongue_idx, 0, :]  # x position
    y = ts[tongue_idx, 1, :]  # y position
    conf = ts[tongue_idx, 2, :] if ts.shape[1] >= 3 else np.ones_like(x)
else:
    x = ts[:, 0, tongue_idx]; y = ts[:, 1, tongue_idx]; conf = ts[:, 2, tongue_idx]
```

iii. CONVERSION_NOTES Step 5 Key Decision 10: "Tongue/paw velocity: Compute total speed = sqrt(xvel^2 + yvel^2). **Tongue from side cam**, paw from bottom cam." The reference's `params.traj_features` lists `tongue` as a cam0 (side) feature, and the reference's `alignSpikes.m`/`getKinematics` helpers likewise index `view = 1` (side cam) for tongue-ish features. No justification is given for not also using the bottom-camera tongue markers.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. At frame resolution (400 Hz): `xvel = np.gradient(x)`, `yvel = np.gradient(y)`, `speed = hypot(xvel, yvel)` — a per-frame displacement magnitude, **not** divided by the frame interval, so the units are pixels/frame. Frames with `conf < 0.1` are set to NaN. In practice the visibility rule is inherited from the data rather than from that threshold: the authors have already written NaN into x and y wherever the DLC likelihood was low (I confirmed ~95% of tongue frames are NaN in the raw `ts`), and `np.gradient` propagates each NaN to its neighbours, so untracked frames and the first/last frame of each visible run come out NaN. No smoothing is applied to the tongue and no gap filling is done — NaNs are deliberately preserved. The per-frame speed is then linearly interpolated onto the 10 ms bin centres with `left=np.nan, right=np.nan`. No cross-camera normalisation is needed since only one view is used.

ii.
```python
# Low confidence = tongue not visible (DLC convention: conf < 0.5 or similar)
# For tongue, keep NaNs as-is (no filling, per reference code)
xvel = np.gradient(x)
yvel = np.gradient(y)
speed = np.sqrt(xvel**2 + yvel**2)
speed[conf < 0.1] = np.nan
...
interp_speed = np.interp(taxis, aligned_ft, speed, left=np.nan, right=np.nan)
tongue_speed[:, t] = interp_speed
```

iii. This mirrors the reference `findPosition.m`/`findVelocity.m`, which the AI summarised in the trajectory as: "**No smoothing for tongue:** Preserves raw tongue motion (NaNs kept as NaN)" and "xvel(:,i) = gradient(xpos(:,i)) ... Fills missing values: Non-tongue: fillmissing(..., 'nearest'); Tongue: Sets NaN to 0". CONVERSION_NOTES Step 5 Key Decision 10: "Compute total speed = sqrt(xvel^2 + yvel^2) ... NaN (not visible) → category 2." Step 7: "Tongue ~90% not visible is expected (only visible during licking)."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, over the pooled visible (non-NaN) values across all retained trials and all 500 bins: the 50th percentile is the split. Values below it → 0, at or above → 1, NaN → 2. This is applied after the stim-trial subset is taken, so the threshold is computed on exactly the trials that are exported.

ii.
```python
def discretize_velocity(speed_matrix):
    result = np.full_like(speed_matrix, 2, dtype=np.int8)  # default: not visible
    valid = speed_matrix[~np.isnan(speed_matrix)]
    if len(valid) == 0:
        return result
    threshold = np.percentile(valid, 50)
    visible = ~np.isnan(speed_matrix)
    result[visible & (speed_matrix < threshold)] = 0
    result[visible & (speed_matrix >= threshold)] = 1
    return result
```
```python
tongue_speed = tongue_speed_all[:, trial_indices]
tongue_disc = discretize_velocity(tongue_speed)
```

iii. Directly from the Decoder Task spec ("discretized with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile, 2: not visible"), restated as Key Decision 10: "Discretize using per-session 50th percentile of visible timepoints. NaN (not visible) → category 2."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The side camera's `frameTimes` are corrected by a session-constant video offset and then by the trial's go cue: `frameTimes − vidshift − goCue[trial]`. `vidshift` is `median(sglx.bitcode.bitstart)/sglx.fs − median(bp.ev.bitStart)` — the reference's `findVideoOffset.m` with `median` in place of `mode` (I checked all v7.3 sessions: the two agree to within 5 ms everywhere, so this substitution is harmless). The per-frame speed is then linearly interpolated onto `TIME_AXIS`, the same grid as the neural bins, with NaN outside the frame range. If the bitcode fields are unreadable, `vidshift` silently falls back to 0.5 s; if `frameTimes` is absent, frame times fall back to `arange(n_frames)/400`.

ii.
```python
bit_start_bpod = float(np.nanmedian(ev['bitStart'][0, :]))
sglx = obj_grp['sglx']
bit_start_sglx = float(np.nanmedian(sglx['bitcode']['bitstart'][0, :])) / float(sglx['fs'][0, 0])
session['vidshift'] = bit_start_sglx - bit_start_bpod
```
```python
aligned_ft = frame_times - vidshift - go_cue_times[t]
taxis = TIME_AXIS
interp_speed = np.interp(taxis, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. CONVERSION_NOTES Step 1: "findVideoOffset | funcs/findVideoOffset.m | PROCESSING | Computes time offset between video and neural recording"; Step 3: "Video data interpolated to neural timebase after video-neural offset correction". The trajectory records the reference formula: "`bitStart = mode(obj.bp.ev.bitStart)`; `vidFileOffset = mode(obj.sglx.bitcode.bitstart) / obj.sglx.fs`; `vidshift = vidFileOffset - bitStart`". The `--show-processing` plots were used as the alignment check.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The **bottom** camera tracking, `obj.traj{2}`, taking the first feature whose name contains `paw` — which is `top_paw` (the bottom camera's feature order is `top_tongue, topleft_tongue, bottom_tongue, bottomleft_tongue, top_paw, bottom_paw, lickport, jaw, top_nostril, bottom_nostril`, so `top_paw` wins the search). Plus the same `frameTimes`, `vidshift` and `goCue`. `bottom_paw` is not used.

ii.
```python
view_data = session['traj'][1]  # bottom cam
feat_names = view_data.get('featNames', [])
paw_idx = None
for fi, fn in enumerate(feat_names):
    if 'paw' in fn.lower():
        paw_idx = fi
        break
```

iii. CONVERSION_NOTES Step 5 Key Decision 10: "Tongue from side cam, **paw from bottom cam**." The code comment says "Find paw feature index (try top_paw or bottom_paw)" — the AI does not state that it prefers `top_paw`, it just takes whichever appears first. (In the raw data `top_paw` is tracked on essentially every frame while `bottom_paw` is NaN on ~81% of frames, so the search happens to pick the reliable one.)

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical frame-resolution speed computation to the tongue — `hypot(gradient(x), gradient(y))` in pixels/frame, NaN where `conf < 0.1` (and inherited from the NaNs the authors already wrote into x/y) — **plus a gap-filling step that the tongue does not get**: missing frames are filled by linear interpolation over the frame index from the surrounding valid frames (flat extrapolation at the edges), following the reference's `fillmissing(..., 'nearest')` for non-tongue features. Only if every frame of a trial is invalid does the trial stay NaN. The filled speed is then interpolated onto the 10 ms grid. No baseline (median frame-to-frame drift) subtraction is applied, although the reference's `findVelocity.m` does subtract one for non-tongue features. Consequence: the "not visible" class covers only 5.2% of paw bins.

ii.
```python
xvel = np.gradient(x)
yvel = np.gradient(y)
speed = np.sqrt(xvel**2 + yvel**2)
speed[conf < 0.1] = np.nan

# For non-tongue features, fill missing with nearest (per reference code)
mask = np.isnan(speed)
if mask.any() and not mask.all():
    idx = np.where(~mask)[0]
    if len(idx) > 0:
        speed = np.interp(np.arange(len(speed)), idx, speed[idx])

aligned_ft = frame_times - vidshift - go_cue_times[t]
interp_speed = np.interp(TIME_AXIS, aligned_ft, speed, left=np.nan, right=np.nan)
```

iii. The inline comment cites the reference directly ("per reference code"), matching the AI's summary of `findPosition.m`/`findVelocity.m`: "**Fills missing values (if not tongue):** Uses nearest-neighbor interpolation for NaN". Step 7: "Paw velocity well distributed."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same `discretize_velocity` as the tongue: per-session 50th percentile over all pooled visible values of the exported trials; below → 0, ≥ → 1, NaN → 2. Because of the gap filling in 8-b, class 2 appears only where a whole trial (or the part of it inside the window) had no tracking at all.

ii.
```python
paw_speed = paw_speed_all[:, trial_indices]
paw_disc = discretize_velocity(paw_speed)
```

iii. Same as 7-c — Key Decision 10 and the Decoder Task spec. Resulting distribution: 0.474 / 0.474 / 0.052.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, except the frame times come from the **bottom** camera (`session['traj'][1]`), which is the camera the paw is tracked by: `frameTimes − vidshift − goCue[trial]`, then `np.interp` onto `TIME_AXIS`. The same session-constant `vidshift` is reused, and the same fallbacks apply.

ii.
```python
if is_h5:
    ts, frame_times = get_traj_trial_h5(view_data, t)   # view_data = session['traj'][1]
else:
    ts, frame_times = get_traj_trial_v5(view_data, t)
...
aligned_ft = frame_times - vidshift - go_cue_times[t]
interp_speed = np.interp(TIME_AXIS, aligned_ft, speed, left=np.nan, right=np.nan)
paw_speed[:, t] = interp_speed
```

iii. Same as 7-d: Step 3 "Video data interpolated to neural timebase after video-neural offset correction", following `findVideoOffset.m` + `findPosition.m`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` beside the session file, i.e. `me.data{trial}` — one value per camera frame per trial. `obj.me` (present in only some sessions) is not used. Frame times for alignment come from the **side** camera (`obj.traj{1}(trial).frameTimes`). Three on-disk layouts occur and are handled: a v7.3 HDF5 file with object references, a v5 file with `me.data` as a `(nTrials,1)` cell array, and a v5 file with a doubly nested `me.data.data`, unwrapped recursively.

ii.
```python
me_pattern = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
...
def _unwrap_me_data(arr):
    if hasattr(arr, 'dtype') and arr.dtype.names and 'data' in arr.dtype.names:
        inner = arr['data']
        if inner.shape == (1, 1):
            return _unwrap_me_data(inner[0, 0])
        return inner
    return arr
me_top = me_arr[0, 0] if me_arr.shape == (1, 1) else me_arr
raw = _unwrap_me_data(me_top)
```

iii. CONVERSION_NOTES Step 9: "**Bug Fix: Motion Energy Loading** — JEB23 sessions (10-10 through 10-13): ME files stored as plain `(nTrials,1)` object array instead of struct with `data` field. Fixed by detecting format and handling both. JEB15 and JEB24 sessions: ME files had doubly nested structs (`me.data.data`). Fixed with recursive unwrapping." This matches the reference `loadMotionEnergy.m` guard `if isstruct(me.data), me.data = me.data.data; end`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling and gap filling: the trace is already one scalar per frame. If the frame-time and motion-energy vectors have different lengths they are both truncated to the shorter one. The trace is then linearly interpolated onto the 10 ms bin centres, and finally any remaining NaN bins are filled by linear interpolation over the bin index from the surrounding valid bins, per session and per trial — reproducing the reference's `fillmissing(me.data,'nearest')`. Consequence: the "no video" class never occurs anywhere in the dataset (`output_range[motion_energy] = [0, 1]`).

ii.
```python
if len(frame_times) != len(trial_me):
    min_len = min(len(frame_times), len(trial_me))
    frame_times = frame_times[:min_len]
    trial_me = trial_me[:min_len]
aligned_ft = frame_times - vidshift - go_cue_times[t]
me_interp = np.interp(TIME_AXIS, aligned_ft, trial_me)
me_data[:, t] = me_interp
...
# Fill NaNs with nearest (per reference code)
for t in range(ntrials):
    col = me_data[:, t]
    mask = np.isnan(col)
    if mask.any() and not mask.all():
        idx = np.where(~mask)[0]
        me_data[:, t] = np.interp(np.arange(len(col)), idx, col[idx])
```

iii. CONVERSION_NOTES Step 3: "Motion energy loaded from separate files, interpolated to neural timebase, fillmissing with nearest", from `loadMotionEnergy.m` lines `me.newdata(:,trix) = interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)` and `me.data = fillmissing(me.data,'nearest')`. Step 7: "Motion energy has no 'no_video' category (all trials have ME data)."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per session, 50th percentile of the pooled non-NaN values of the exported trials; below → 0, ≥ → 1, NaN → 2. `discretize_motion_energy` is byte-for-byte the same logic as `discretize_velocity`, duplicated with different docstrings. Because of the fill in 9-b, class 2 is never emitted and the split is a clean 0.500/0.500.

ii.
```python
def discretize_motion_energy(me_matrix):
    result = np.full_like(me_matrix, 2, dtype=np.int8)  # default: no video
    valid = me_matrix[~np.isnan(me_matrix)]
    if len(valid) == 0:
        return result
    threshold = np.percentile(valid, 50)
    visible = ~np.isnan(me_matrix)
    result[visible & (me_matrix < threshold)] = 0
    result[visible & (me_matrix >= threshold)] = 1
    return result
```

iii. Key Decision 11: "Motion energy: Load from separate .mat files. Discretize per-session 50th percentile. Missing → category 2." — the Decoder Task spec. The AI notes in `output_values` that only two classes are populated (`['below_50pct', 'above_50pct']`, without a third entry).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the tracking, using the side camera's frame times because motion energy has one value per side-camera frame: `frameTimes − vidshift − goCue[trial]`, then `np.interp` onto `TIME_AXIS`. If the side-camera frame times cannot be read, the fallback is `arange(len(trial_me))/400` (400 Hz), which does **not** include the `−0.5` correction the reference's own fallback applies.

ii.
```python
if len(session['traj']) > 0 and '_traj_grp' in session['traj'][0]:
    view_data = session['traj'][0]
    try:
        ft_ref = view_data['_traj_grp']['frameTimes'][t, 0]
        frame_times = np.array(view_data['_f'][ft_ref]).flatten()
    except:
        frame_times = np.arange(len(trial_me)) / 400.0
...
aligned_ft = frame_times - vidshift - go_cue_times[t]
me_interp = np.interp(TIME_AXIS, aligned_ft, trial_me)
```

iii. Copied from `loadMotionEnergy.m`, which uses `obj.traj{1}(trix).frameTimes` and the same `vidshift`/`alignTimes` subtraction. CONVERSION_NOTES Step 3: "Motion energy loaded from separate files, interpolated to neural timebase".

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms, mostly "keep the trial, mark or fill the gap":
- **Untracked video frames**: x/y are already NaN in the source; the tongue keeps the NaN (class 2), the paw and motion energy are gap-filled by linear interpolation from surrounding valid samples (following the reference's `fillmissing 'nearest'`).
- **Missing/NaN frame times**: fall back to `arange(n_frames)/400`; if `np.interp` still fails, the trial's column stays NaN → class 2.
- **Mismatched frame-count vs motion-energy length**: both truncated to the shorter.
- **Missing bitcode / sglx fields**: `vidshift` defaults to 0.5 s.
- **Missing motion-energy file, missing `featNames`, missing camera view**: return an all-NaN matrix, so the whole session's stream becomes class 2.
- **Missing probe location**: brain region defaults to `'ALM'`.
- **Sessions with too few units or too few R/L hits**: skipped entirely.
- **Trials past the end of the ephys recording**: **not** handled — 64 such trials in 3 sessions were exported with all-zero neural data and flagged as warnings by the verifier, then accepted.
- Robustness is achieved largely with bare `except:` blocks and a module-level `warnings.filterwarnings('ignore')`, so most of these failures are silent.

ii.
```python
warnings.filterwarnings('ignore')
```
```python
try:
    bit_start_bpod = float(np.nanmedian(ev['bitStart'][0, :]))
    ...
except:
    session['vidshift'] = 0.5  # default: 0.5s as in code comments
```
```python
except:
    n_frames = ts.shape[-1] if ts.ndim == 3 else ts.shape[0]
    frame_times = np.arange(n_frames) / 400.0
```
```python
loc = session['probe_locs'].get(pi, '')
if 'ALM' in loc.upper(): ...
else:
    brain_regions_for_units.append('ALM')  # default
```

iii. CONVERSION_NOTES Step 9 documents the three motion-energy layouts and their fixes. Step 10: "Zero-neural-data trials: 64 trials across 3 sessions at session ends (<0.5%) — acceptable, recording ended early", and Step 12: "Recording ended before last trials' time window. Affects <1% of data, acceptable." The fill-vs-keep-NaN split is justified as following the reference code ("For non-tongue features, fill missing with nearest (per reference code)"; "For tongue, keep NaNs as-is (no filling, per reference code)").

## 11-a. What are the most time-consuming steps of the code?

i. The whole conversion takes 116.5 s for 44 sessions (1.5–5.0 s each), well inside the 15-minute budget, and the script only times whole sessions, so no per-step attribution is recorded. Profiling the AI's own functions on EKH1_2021-08-07 (2.3 s total) gives: paw velocity 0.78 s, tongue velocity 0.59 s, spike align+bin+smooth 0.46 s, file load 0.37 s, motion energy 0.03 s. So the dominant cost is the **per-trial video work** — ~2 × `ntrials` HDF5 reads of the full `(n_features, 3, n_frames)` trajectory array, of which one feature is used — followed by the **nested per-unit × per-trial spike loop** (`np.histogram` + `np.convolve` per unit per trial; this scales with unit count and becomes the leader on the 141-unit two-probe sessions), then file loading.

ii.
```python
for ui, u in enumerate(all_units):
    for t in range(ntrials):
        ...
        counts, _ = np.histogram(aligned_times, bins=EDGES)
        trialdat[:, ui, t] = smooth_signal(fr)
```
```python
for t in range(ntrials):
    if is_h5:
        ts, frame_times = get_traj_trial_h5(view_data, t)   # reads all features, uses one
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Per-trial spike binning loop (could be vectorized with sparse matrices but complexity not worth it for ~44 sessions)"; Step 7: "2 sessions took 7.9s → ~4s/session; 44 sessions estimated: ~3 minutes (well under 15 min limit)". The AI judged further optimisation unnecessary given the budget.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several:
- The **spike binning double loop** over units × trials: a single `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, EDGES])` per unit (or per session) would replace `n_units × n_trials` `np.histogram` calls, and `gaussian_filter1d`/a strided convolution could smooth the whole `(time, units, trials)` array in one call along axis 0 instead of one `np.convolve` per unit per trial.
- `smooth_matrix` loops over columns calling `smooth_signal` — vectorizable (it is in fact dead code, never called).
- The three per-trial **output label loops** (`lick_dir`, `context`, `outcome`) are pure boolean algebra on length-`ntrials` arrays and could each be two `np.where` calls.
- The **packaging loop** `for i in range(n_trials_out)` builds the output array row by row; the three per-trial labels could be broadcast with `out[:, 0, :] = lick_dir[:, None]`.
- The **motion-energy NaN-fill loop** over all trials.
- The per-trial video loops genuinely resist vectorization (frame counts differ per trial), but the redundant full-`ts` reads inside them could be narrowed to the single feature index.

ii.
```python
lick_dir = np.zeros(len(trial_indices), dtype=np.int8)
for i, ti in enumerate(trial_indices):
    if session['R'][ti] and (session['hit'][ti] or session['miss'][ti]):
        lick_dir[i] = 1
    ...
```
```python
def smooth_matrix(X, kernel=SMOOTH_KERNEL, bctype=BC_TYPE):
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = smooth_signal(X[:, j], kernel, bctype)
    return out
```

iii. The AI acknowledged only the spike loop, and explicitly declined to fix it: "could be vectorized with sparse matrices but complexity not worth it for ~44 sessions" (Step 6). Its claimed speed-ups — "Vectorized histogram binning with numpy; Minimal data loading (only needed fields)" — are modest given the remaining loops.

## 11-c. What processing does the code repeat multiple times?

i.
- **Kinematics for discarded trials**: `compute_tongue_velocity`, `compute_paw_velocity` and `load_motion_energy` all loop over `range(ntrials)`, i.e. over every trial including photostim trials, and the result is subsetted to `trial_indices` afterwards.
- **Spike binning and smoothing for discarded units**: every quality-passing unit is binned and smoothed before the ≤ 1 Hz filter throws away roughly 30% of them (3,363 quality-passing clusters → 2,354 kept).
- **Full processing of sessions that are then skipped**: the ≥ 40 R/L-hit check happens *after* the whole spike pipeline has run, so JEB19 2023-04-19 and 2023-04-20 were fully binned and smoothed and then discarded.
- **Redundant HDF5 reads**: the side camera's `frameTimes` are read once inside `compute_tongue_velocity` and then again, trial by trial, inside `load_motion_energy`; the full `ts` array is read per trial per view even though only one feature index is consumed.
- **Duplicated code**: `discretize_motion_energy` is an exact copy of `discretize_velocity`; `compute_paw_velocity` is a near-copy of `compute_tongue_velocity`.

ii.
```python
# --- 2. Align spikes and compute firing rates ---   (runs for all quality-passing units)
...
# --- 3. Remove low FR clusters ---                  (throws ~30% of them away)
mean_frs = trialdat.mean(axis=(0, 2))
keep_mask = mean_frs > LOW_FR
...
# --- 4. Check trial inclusion criteria ---          (may skip the whole session here)
if n_r_hit < 40 or n_l_hit < 40:
    return None
```
```python
tongue_speed_all = compute_tongue_velocity(session, go_cue, vidshift)  # all ntrials
tongue_speed = tongue_speed_all[:, trial_indices]                      # then subset
```

iii. Not discussed in CONVERSION_NOTES; the AI's only efficiency note is the one quoted in 11-b. The ordering is inherited from the reference pipeline (`findTrials → findClusters → alignSpikes → getSeq → removeLowFRClusters`), which also filters rate after binning.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Dead functions**: `align_and_bin_spikes` and `remove_low_fr_clusters` are defined but never called (`process_session` inlines both); `smooth_matrix` is never called; `from scipy.signal import convolve` is never used; the `'zeropad'`/default branches of `smooth_signal` are unreachable given `BC_TYPE = 'reflect'`.
- **Loaded-but-unused variables**: `bp.no` (outcome/lick direction fall through instead), `bp.ev.sample`, `bp.ev.delay`, and `session['L']` for the outcome path; `go_cue_filtered` is computed but used only by the plotting routine.
- **Discarded computation**: everything listed in 11-c — smoothed rates for sub-1 Hz units, all three video streams for photostim trials, and the two fully processed sessions that the inclusion criterion then rejected.
- **Over-reading from disk**: each per-trial `ts` read pulls all 7–10 tracked features × 3 channels when only one feature's x, y (and nominally likelihood) are used, i.e. ~85–90% of the bytes read from the trajectory arrays are thrown away. The `conf < 0.1` mask is itself near-inert, since the authors have already NaN-ed the low-likelihood coordinates.
- **Also stored but unused by the target format**: the intermediate `(N_TIMEBINS, n_units, ntrials)` `trialdat` is materialised for the full session before being sliced into per-trial `(n_neurons, 500)` transposed copies.

ii.
```python
def align_and_bin_spikes(units, cluster_indices, go_cue_times, ntrials):   # never called
def remove_low_fr_clusters(trialdat, cluster_indices, low_fr=LOW_FR):      # never called
def smooth_matrix(X, kernel=SMOOTH_KERNEL, bctype=BC_TYPE):                # never called
from scipy.signal import convolve                                          # never used
```
```python
session['sample'] = ev['sample'][0, :]     # never used
session['delay'] = ev['delay'][0, :]       # never used
session['no'] = bp['no'][0, :].astype(bool)  # never used
```

iii. Not addressed in CONVERSION_NOTES. Step 13 claims cleanup was done ("CONVERSION_NOTES.md finalized; All output files generated") but the `/app/cache/` folder called for by the instructions was not created and the dead code was not removed.
