# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Sessions are **discovered by globbing**, not from a curated list. `discover_sessions()` walks the two ephys folders (`/app/data/Ephys_Behavior`, `/app/data/RandomizedDelay_Ephys_Behavior`), takes every file starting with `data_structure_`, parses `<animal>_<date>` out of the filename, and pairs it with a `motionEnergy_<animal>_<date>.mat` beside it if one exists. The two behavior-only folders (`DelayInhibition_BilatMC_Behavior`, `GoCueInhibition_BilatMC_Behavior`) are never listed in `DATA_DIRS`, so they are excluded wholesale. One file, `data_structure_JEB23_2023-10-20.mat`, is hard-coded into `SKIP_FILES` after the agent found it to be a byte-equivalent duplicate of `JEB23_2023-10-19`.

Each file is opened once by `load_session()`, which tries the MATLAB v7.3/HDF5 reader (`load_session_h5`, h5py) and falls back to the v5 reader (`load_session_scipy`, `scipy.io.loadmat`) on any exception. Sessions that yield zero usable clusters, fewer than 2 valid trials, or fewer than `MIN_UNITS = 10` surviving units are dropped. 46 files were discovered, 2 were dropped for having no neural data (`JEB24_2023-10-03`, `JEB24_2023-10-04`), and 44 sessions entered the dataset — the same 44 sessions, the same 14 subjects, and the same sessions-per-subject counts as the reference solution.

ii.
```python
DATA_DIRS = {
    'Ephys': '/app/data/Ephys_Behavior',
    'RandDelay': '/app/data/RandomizedDelay_Ephys_Behavior',
}
SKIP_FILES = {
    'data_structure_JEB23_2023-10-20.mat',  # duplicate of JEB23_2023-10-19
}

def discover_sessions(sample=False):
    sessions = []
    for dir_key, data_dir in DATA_DIRS.items():
        if not os.path.exists(data_dir):
            continue
        for fn in sorted(os.listdir(data_dir)):
            if not fn.startswith('data_structure_'):
                continue
            if fn in SKIP_FILES:
                continue
            parts = fn.replace('.mat', '').split('_')
            animal = parts[2]
            date = parts[3]
            fpath = os.path.join(data_dir, fn)
            me_path = find_me_file(data_dir, animal, date)
            sessions.append({'fpath': fpath, 'me_path': me_path, 'animal': animal,
                             'date': date, 'dir': dir_key,
                             'session_id': f'{animal}_{date}'})
    ...
```

```python
def load_session(fpath):
    """Load session from either h5py or scipy format."""
    try:
        session = load_session_h5(fpath)
        return session
    except:
        pass
    try:
        session = load_session_scipy(fpath)
        return session
    except Exception as e:
        print(f"  ERROR loading {fpath}: {e}")
        return None
```

iii. From CONVERSION_NOTES.md Step 4: "**Include both Ephys_Behavior and RandomizedDelay sessions with neural data** — both have the variables needed for the decoder" and "**Exclude behavior-only sessions** (DelayInhibition, GoCueInhibition, JEB24_10-03, JEB24_10-04) — no neural data". The dual reader was needed because, per Step 2, `RandomizedDelay_Ephys_Behavior/` is a "Mix of HDF5 and older MATLAB format (scipy.io.loadmat)". The duplicate exclusion is justified in Step 10: "JEB23_2023-10-20 is duplicate of JEB23_2023-10-19 (different file format, identical data). Excluded. Session count now matches paper (44 = 25 + 19)." The `MIN_UNITS = 10` cut is attributed in Step 3/4 to a paper inclusion criterion ("Sessions need >= 10 units for inclusion").

---

## 1-b. How are the data split into subjects?

i. The subject is the `<animal>` token of the filename, parsed in `discover_sessions()` as `fn.replace('.mat','').split('_')[2]`. `subjects` is the sorted set of unique animals over **all discovered** sessions, and `subject_idx` is built per successfully-processed session by indexing into that list. This yields 14 subjects (EKH1, EKH3, JEB6, JEB7, JEB11, JEB12, JEB13, JEB14, JEB15, JEB19, JEB23, JEB24, JGR2, JGR3), identical to the reference.

ii.
```python
parts = fn.replace('.mat', '').split('_')
animal = parts[2]
...
subjects_set = sorted(set(s['animal'] for s in sessions_info))
subject_to_idx = {s: i for i, s in enumerate(subjects_set)}
...
subject_idx_list.append(subject_to_idx[sess_info['animal']])
...
'subjects': subjects_set,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. No explicit rationale is given beyond the data-organisation notes in Step 2, which list the animals per folder ("10 animals: EKH1, EKH3, JEB6, ..."). The filename is the only place the animal id is reliably recorded.

---

## 1-c. How are the data split into sessions?

i. One `data_structure_*.mat` file = one session = one element of `neural`, `input`, `output`, `subject_idx` and `brain_region_idx`. Fixed-delay and randomized-delay sessions are pooled into a single flat list of sessions (25 + 19), tagged only internally by `dir_key`. The dedup of `JEB23_2023-10-20` is the only manual intervention in the session list.

ii.
```python
for si, sess_info in enumerate(sessions_info):
    result = process_session(sess_info['fpath'], me_path=sess_info['me_path'], ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
    subject_idx_list.append(subject_to_idx[sess_info['animal']])
    brain_region_idx_list.append(np.zeros(result['n_neurons'], dtype=np.int64))
```

iii. Step 9/10 of CONVERSION_NOTES.md: after the dedup, "Session count now matches paper (44 = 25 + 19)" — the agent used the paper's reported 25 fixed-delay and 19 randomized-delay sessions as the acceptance criterion for its globbing-based discovery.

---

## 1-d. How are the data split into trials?

i. A trial is one index into `obj.bp`'s per-trial fields, of which there are `obj.bp.Ntrials`. All per-trial behaviour flags (`hit`, `miss`, `no`, `early`, `autowater`, `L`, `R`, `stim.enable`) and event times (`ev.goCue`, `ev.sample`, `ev.delay`) are read as flat arrays and indexed by that trial number. Spikes carry their own 1-based `trial` label (`neuron['trial']`), and the DLC tracking (`obj.traj{cam}.ts`, `.frameTimes`) and motion energy are stored as one entry per trial, so no trial boundaries are ever reconstructed. Trial indices are kept in the original numbering (`trial_indices = np.where(valid_trials)[0]`) and used to reach back into the raw per-trial arrays for video and behaviour.

ii.
```python
session['Ntrials'] = int(np.array(bp['Ntrials']).flatten()[0])
session['hit'] = np.array(bp['hit']).flatten().astype(bool)
...
n_trials = session['Ntrials']
...
for t in range(n_trials):
    trial_num = t + 1  # MATLAB 1-indexed
    ...
    spike_mask = trial == trial_num
```

```python
trial_indices = np.where(valid_trials)[0]
n_valid = len(trial_indices)
```

iii. Not separately justified; the Bpod trial table defines trials directly. Note the code takes the bp fields at their full stored length rather than truncating to `Ntrials` (the reference does `[:n_trials]` because a few fields are stored longer), so a length mismatch would raise rather than be absorbed — it did not occur in any of the 44 sessions.

---

## 1-e. How are trials filtered based on quality controls?

i. Three per-trial filters, all applied before any processing: early-lick trials (`bp.early`), photostimulation trials (`bp.stim.enable`), and trials with a NaN go cue. Sessions with fewer than 2 surviving trials are dropped entirely. 13,823 of ~15,200 trials survive.

There is **no** filter for trials that run past the end of the ephys recording. The agent's own verification run flagged 61 such trials (28 in session 36, 33 in session 43, both JEB24) in which **every** neuron has zero spikes for all 500 bins, and chose to keep them.

ii.
```python
valid_trials = np.ones(n_trials, dtype=bool)
valid_trials[session['early']] = False
valid_trials[session['stim_enable']] = False
# Also exclude trials with NaN goCue
valid_trials[np.isnan(goCue)] = False

trial_indices = np.where(valid_trials)[0]
n_valid = len(trial_indices)

if n_valid < 2:
    print(f"  {session_id}: Too few valid trials ({n_valid}), skipping")
    return None
```

iii. CONVERSION_NOTES.md Step 4: "**Exclude stim trials** (stim.enable == 1) and early lick trials", with Step 3 citing "Early lick trials excluded from analyses" and "Stim/opto trials excluded (stim.enable == 0)" from the paper. For the all-zero trials, Step 10 Check 1 states: "Sessions 36, 43 (both JEB24) have all-zero late trials where recording ended. **Data artifact, not a conversion bug.**" — i.e. the agent diagnosed the cause correctly and deliberately declined to remove them.

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` — the spike-sorted clusters — plus `obj.bp.ev.goCue` for the alignment. For each cluster the code reads `quality` (the manual curation label, decoded from its uint16 HDF5 reference), `trialtm` (spike time relative to that trial's start, on the behaviour clock), and `trial` (the 1-based trial each spike falls in). `tm` (whole-session spike times) is dereferenced and read but never used.

**All probes** present in `obj.clu` are read and concatenated into one population; the code does not consult the authors' per-session probe choice.

ii.
```python
clu_ds = obj['clu']
neurons = []
for pi in range(clu_ds.shape[0]):
    for pj in range(clu_ds.shape[1]):
        ref = clu_ds[pi, pj]
        dereffed = f[ref]
        if not isinstance(dereffed, h5py.Group) or 'quality' not in dereffed:
            continue
        quality_ds = dereffed['quality']
        n_neurons = quality_ds.shape[0]
        for ni in range(n_neurons):
            q_ref = quality_ds[ni, 0]
            quality = read_h5_string(f, q_ref).strip().lower()
            if quality in EXCLUDE_QUALITY:
                continue
            tm_ref = dereffed['tm'][ni, 0]
            trialtm_ref = dereffed['trialtm'][ni, 0]
            trial_ref = dereffed['trial'][ni, 0]
            tm = np.array(f[tm_ref]).flatten()
            trialtm = np.array(f[trialtm_ref]).flatten()
            trial = np.array(f[trial_ref]).flatten().astype(int)
            neurons.append({'quality': quality, 'trialtm': trialtm, 'trial': trial})
```

iii. Step 1 of CONVERSION_NOTES.md: "**clu structure per probe**: quality, tm (spike times), trial, trialtm, site, spkWavs" and "`getSeq` output: obj.trialdat{probe} = (nTimeBins x nClusters x nTrials), firing rates in Hz". Step 1 also notes the reference pipeline concatenates dual-probe sessions ("Dual probe: Concatenate data from both probes" in `loadSessionData.m`), which is the stated basis for reading every probe.

---

## 2-b. How is the `neural` data processed?

i. For each cluster and each trial: spikes are aligned to that trial's go cue, counted with `np.histogram` into the fixed 10 ms bin edges, divided by the bin width to give spikes/s, and smoothed along time with a **causal** Gaussian. The kernel is a hand-rolled port of MATLAB's `gausswin(15)` (alpha = 2.5) with the first `floor(N/2) = 7` taps zeroed and the rest renormalised to sum to 1 — i.e. only past bins contribute. Boundaries are handled by prepending the first 15 samples, convolving `'same'`, and trimming. Nothing else is done: no baseline subtraction, no z-scoring, no normalisation. Stored values are firing rates in Hz (`float32`). At 10 ms bins the kernel has a standard deviation of ~28 ms.

ii.
```python
def causal_gaussian_kernel(N):
    alpha = 2.5
    n = np.arange(N)
    w = np.exp(-0.5 * ((n - (N - 1) / 2) / (alpha * (N - 1) / 2)) ** 2)
    w[:N // 2] = 0            # Zero out first half (causal)
    w = w / w.sum()
    return w

KERNEL = causal_gaussian_kernel(SMOOTH_WIN)   # SMOOTH_WIN = 15
```

```python
            aligned = trialtm[spike_mask] - gc
            counts, _ = np.histogram(aligned, bins=edges)
            fr = counts.astype(np.float32) / dt
            fr = smooth_causal(fr, KERNEL, 'reflect')
            trialdat[ni, :, t] = fr
```

iii. CONVERSION_NOTES.md Step 1: "**Smoothing**: Causal Gaussian kernel, 15 samples = 150ms (code default)", sourced to `utils/mySmooth.m`, which the agent's exploration reported as `kern = gausswin(N); kern(1:floor(numel(kern)/2)) = 0; kern = kern./sum(kern)` with `'reflect' | 'zeropad' | 'none'` boundary handling. Step 4 records the discrepancy with the paper ("half-width 35ms") and resolves it as "Use code's smoothing: 15-sample causal Gaussian".

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters plus one session-level filter.

1. **Quality label.** The free-text `clu.quality` string is lower-cased and stripped, and the cluster is dropped if it lands in `EXCLUDE_QUALITY = {'garbage', 'noisy', 'gabrga', 'real?', ''}`. Everything else is kept, including `poor` and `multi`.
2. **Mean firing rate.** After binning, smoothing and restriction to valid trials, a unit is dropped unless its mean rate over the whole (trials × bins) window is `>= 1.0` Hz.
3. **Session.** A session is dropped if fewer than `MIN_UNITS = 10` units survive (no session actually hit this).

Because all probes are read, sessions where the authors used only one probe contribute both. The most extreme case is `JEB15_2022-07-29`, where probe 1 holds 197 clusters whose `quality` field is a null-padded empty string (`'\x00\x00'`) — i.e. an entirely uncurated probe. `'\x00\x00'.strip().lower()` is not `''`, so the intended empty-label exclusion does not fire, and all 197 enter the pipeline (229 raw clusters → 54 units for that session). The dataset ends with **3,066 units** vs the reference's 1,954 and the paper's 1,651 + 845 = 2,496.

ii.
```python
EXCLUDE_QUALITY = {'garbage', 'noisy', 'gabrga', 'real?', ''}
LOW_FR_THRESH = 1.0   # Hz, minimum mean firing rate
MIN_UNITS = 10        # minimum units per session
...
quality = read_h5_string(f, q_ref).strip().lower()
if quality in EXCLUDE_QUALITY:
    continue
```

```python
    trialdat = trialdat[:, :, trial_indices]  # (n_neurons, n_timebins, n_valid)

    # Remove low FR neurons
    mean_fr = np.mean(np.mean(trialdat, axis=2), axis=1)
    keep_neurons = mean_fr >= LOW_FR_THRESH
    trialdat = trialdat[keep_neurons, :, :]

    n_units = trialdat.shape[0]
    if n_units < MIN_UNITS:
        print(f"  {session_id}: Too few units after filtering ({n_units}), skipping")
        return None
```

iii. Step 1/4/5: "**Quality filter**: 'all' excludes garbage, noisy, gabrga, real?" (from `findClusters.m`); "**FR > 1 Hz filter**: Matches removeLowFRClusters"; "**Sessions with < 10 units after filtering excluded**: Paper criterion". The excess unit count was noticed and dismissed in Step 9: "Ephys neurons 1,651 → 2,141 Higher*; RandDelay 845 → 925 Higher*; *Paper neuron counts are for specific analyses with stricter filtering. Our 'all' quality filter includes more unit types." No investigation of the per-session probe selection was performed.

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A single subtraction. `clu.trialtm` is already on the behaviour clock and already relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm[spikes in trial t] - goCue[t]` puts every spike in seconds from the go cue with no interpolation and no offset. Trials whose `goCue` is NaN are skipped outright (and were already removed from `trial_indices`). Note the same `bp.ev.goCue` field carries the water-drop time on autowater/WC trials, so WC trials are aligned to water delivery — which the metadata states explicitly.

ii.
```python
    for ni, neuron in enumerate(neurons):
        trialtm = neuron['trialtm']
        trial = neuron['trial']
        for t in range(n_trials):
            trial_num = t + 1  # MATLAB 1-indexed
            gc = goCue[t]
            if np.isnan(gc):
                continue
            spike_mask = trial == trial_num
            if not np.any(spike_mask):
                continue
            # Align to go cue
            aligned = trialtm[spike_mask] - gc
```

```python
'temporal_alignment_event': 'Go cue onset (DR) / water drop (WC)',
```

iii. Step 1: "`alignSpikes.m` ... For each cluster, subtracts event time from spike times: `obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event`"; "**Alignment event**: goCue (standard)". Step 3 adds "DR trials aligned to go cue onset; WC trials aligned to water drop", which is what `bp.ev.goCue` encodes.

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins**, 500 non-overlapping bins spanning −2.5 s to +2.5 s from the go cue. The edge grid is built once at module level and is identical for every trial, session and data stream — neural, input and all three camera outputs land on the same 500-bin axis. No rebinning or resampling is applied afterwards: spikes are histogrammed directly into the final grid, and the camera streams are interpolated directly onto its bin centres. `metadata['time_bin_size'] = 10.0` ms.

ii.
```python
DT = 0.01            # 10 ms time bins (params.dt = 1/100)
TMIN = -2.5           # seconds before go cue
TMAX = 2.5            # seconds after go cue
...
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_CENTERS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_CENTERS)
```

iii. Step 5 Key Decision 1: "**Time bin = 10ms**: Matches reference code default (params.dt = 1/100)", and Decision 2: "**Trial window = [-2.5s, 2.5s] around go cue**: Matches code defaults (params.tmin, params.tmax)". The paper's 5 ms was seen and explicitly overruled in Step 4: "Time bin size | dt=1/100=10ms (code) | 5ms (paper) | **Use 10ms (matches code default; paper's 5ms may be for specific analyses)**". The agent's code exploration reported `getSeq.m` building `edges = params.tmin:params.dt:params.tmax` with `params.dt` defaulting to 1/100 in `WorkingWithDataObjs.m`.

---

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing in the raw data beyond the go cue that defines the alignment. The input is the analysis grid itself: the centre of each of the 500 bins of the −2.5 → +2.5 s window, i.e. −2.495, −2.485, …, +2.495 s. The identical vector is used for every trial of every session.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_CENTERS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_CENTERS)
...
time_input = TIME_CENTERS.astype(np.float32)  # (n_timebins,)
```

iii. Step 5 variable mapping: "Time from go cue (s) | input[0] | Continuous time vector | N/A | -2.5 to 2.5s". The decoder task specification calls for "Time from go cue onset in seconds (continuous, time-varying)", so a ramp rather than a binary impulse.

---

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The vector is constructed analytically from `TMIN`, `TMAX` and `DT`, cast to `float32`, and given a leading singleton axis to make shape `(1, 500)`. The *same array object* is appended once per trial rather than copied, so the pickle stores it once.

ii.
```python
time_input = TIME_CENTERS.astype(np.float32)  # (n_timebins,)

for i in range(n_valid):
    ...
    # Input: time from go cue (1, n_timebins)
    result['input'].append(time_input[np.newaxis, :])
```

iii. N/A — no rationale needed or given; the axis is defined by the analysis window.

---

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. Spikes are histogrammed into `EDGES`, and the input is `EDGES[:-1] + DT/2` — the centres of exactly those bins. Bin *k* therefore denotes the same interval in the neural array and the input array, by construction, with no possibility of drift. The same grid is also the interpolation target for tongue velocity, paw velocity and motion energy.

ii.
```python
counts, _ = np.histogram(aligned, bins=edges)     # edges == EDGES
...
TIME_CENTERS = EDGES[:-1] + DT / 2
...
taxis = TIME_CENTERS
v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
```

iii. N/A.

---

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields of `obj.bp`: the instructed side `L` and `R`, and the outcome flags `hit` and `miss`. The actual licked port is not recorded, so it is inferred from the instructed side combined with whether the animal was rewarded. `bp.ev.lickL` / `bp.ev.lickR` are loaded but are **not** used for this output.

ii.
```python
session['hit'] = np.array(bp['hit']).flatten().astype(bool)
session['miss'] = np.array(bp['miss']).flatten().astype(bool)
session['no'] = np.array(bp['no']).flatten().astype(bool)
session['L'] = np.array(bp['L']).flatten().astype(bool)
session['R'] = np.array(bp['R']).flatten().astype(bool)
```

iii. From the in-code comment and the trajectory: "L/R fields indicate STIMULUS direction, not lick direction / hit: animal licked correct side (L->left, R->right) / miss: animal licked wrong side (L->right, R->left) / no: no lick". The agent's own session summary records this as a bug it found and fixed: "**Lick direction coding bug**: L/R fields indicate STIMULUS direction, not actual lick direction... Fixed the lick_dir assignment logic."

---

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A three-way relabelling, per trial, broadcast constant across all 500 bins. Default is 2 (`none`). On a hit the animal licked the instructed port, so `L → 0 (left)`, `R → 1 (right)`. On a miss it licked the other one, so `L → 1 (right)`, `R → 0 (left)`. Everything else (ignore trials) stays 2. Resulting distribution: left 0.422, right 0.445, none 0.133 — essentially identical to the reference (0.4227 / 0.4460 / 0.1313).

ii.
```python
    lick_dir = np.full(n_valid, 2, dtype=np.int32)  # default: none
    for out_idx, t in enumerate(trial_indices):
        if session['hit'][t]:
            # Correct: lick matches stimulus
            if session['L'][t]:
                lick_dir[out_idx] = 0  # left
            elif session['R'][t]:
                lick_dir[out_idx] = 1  # right
        elif session['miss'][t]:
            # Incorrect: lick is opposite of stimulus
            if session['L'][t]:
                lick_dir[out_idx] = 1  # licked right (wrong)
            elif session['R'][t]:
                lick_dir[out_idx] = 0  # licked left (wrong)
        # 'no' trials: lick_dir stays 2 (none)
...
        out[0, :] = lick_dir[i]          # lick direction (per-trial, broadcast)
```

iii. As in 4-a. Step 10 Check 6 validates the result: "**Lick/outcome consistency**: Perfect — correct↔has lick, ignore↔no lick, incorrect↔has lick." Codes `0=left, 1=right, 2=none` follow the prompt's ordering.

---

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial field, `obj.bp.autowater`, read as an integer array. It marks trials on which water was delivered from a random port with no sensory cue — the water-cued (WC) context. Everything else is the delayed-response (DR) context.

ii.
```python
session['autowater'] = np.array(bp['autowater']).flatten().astype(int)
```

iii. Step 1: "**bp structure**: hit, miss, no, early, **autowater**, L, R, stim.enable, ...". Step 5 mapping: "obj.bp.autowater | output[1]: behavioral_context | WC=1, DR=0 per trial | N/A | 0=WC, 1=DR".

---

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabel: `autowater == 1 → 0 (WC)`, otherwise `1 (DR)`. Per-trial, broadcast across all 500 bins. Resulting distribution WC 0.097 / DR 0.903, matching the reference's 0.0966 / 0.9034.

ii.
```python
    # Behavioral context: 0=WC, 1=DR
    context = np.full(n_valid, 1, dtype=np.int32)  # default: DR
    for out_idx, t in enumerate(trial_indices):
        if session['autowater'][t] == 1:
            context[out_idx] = 0
...
        out[1, :] = context[i]           # behavioral context (per-trial, broadcast)
```

iii. Codes follow the prompt's "Behavioral context (WC, DR)" ordering. `output_values[1] = ['WC', 'DR']`.

---

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The per-trial flags `obj.bp.hit` and `obj.bp.miss`. `obj.bp.no` is loaded into the session dict but is not consulted — a trial that is neither a hit nor a miss is treated as an ignore by construction.

ii.
```python
session['hit'] = np.array(bp['hit']).flatten().astype(bool)
session['miss'] = np.array(bp['miss']).flatten().astype(bool)
session['no'] = np.array(bp['no']).flatten().astype(bool)
```

iii. Step 5 mapping: "obj.bp.hit/miss/no | output[2]: outcome | correct/incorrect/ignore per trial | N/A | 0=incorrect, 1=correct, 2=ignore". The same two flags already drive lick direction.

---

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Three classes, per trial, broadcast across all 500 bins: default 2 (`ignore`), `hit → 1 (correct)`, `miss → 0 (incorrect)`. Ignore trials are kept in the dataset as their own class rather than dropped. Distribution: incorrect 0.120 / correct 0.747 / ignore 0.133, matching the reference's 0.1197 / 0.7490 / 0.1313.

ii.
```python
    # Outcome: 0=incorrect, 1=correct, 2=ignore
    outcome = np.full(n_valid, 2, dtype=np.int32)  # default: ignore
    for out_idx, t in enumerate(trial_indices):
        if session['hit'][t]:
            outcome[out_idx] = 1
        elif session['miss'][t]:
            outcome[out_idx] = 0
...
        out[2, :] = outcome[i]           # outcome (per-trial, broadcast)
```

iii. Codes follow the prompt's "Outcome (incorrect, correct, ignore)". Step 10 Check 6 verified internal consistency against lick direction.

---

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, using **only camera index 0** and the feature named `tongue`. Camera 0 is the *side* camera (its `featNames` are `tongue, left_tongue, right_tongue, jaw, trident, nose, lickport`); the bottom camera's `top_tongue` is not used, so no cross-camera combination or rescaling is performed. Per trial the code reads `ts[feat_idx, 0, :]` (x), `ts[feat_idx, 1, :]` (y) and `ts[feat_idx, 2, :]` (DLC likelihood), plus `frameTimes`. `bp.ev.goCue` and the `obj.sglx` bitcode fields supply the alignment (see 7-d).

Note: CONVERSION_NOTES.md and the in-code comment both label camera 0 the "bottom camera" — the label is wrong, the index is the side camera.

ii.
```python
SIDE_TONGUE / BOTTOM naming is absent; the camera is selected by index:

    # Tongue velocity (bottom camera, 'tongue' feature)
    vidshift = session.get('vidshift', 0.5)
    tongue_vel, tongue_vis = compute_velocity_timeseries(
        session, cam_idx=0, feat_name='tongue',
        trial_indices=trial_indices, vidshift=vidshift)
```

```python
    cam = session['traj'][cam_idx]
    feat_idx = find_feature_index(cam['featNames'], feat_name)
    if feat_idx is None:
        return velocity, visible
    ...
        x = ts[feat_idx, 0, :]  # x positions
        y = ts[feat_idx, 1, :]  # y positions
        conf = ts[feat_idx, 2, :]  # confidence
```

iii. Step 1: "**traj structure**: 2 cameras (bottom: tongue/jaw/paw, side: tongue/paw/jaw/nose), each with ts (features x [x,y,conf] x frames), frameTimes, featNames". Step 5 Key Decision 7: "**Tongue velocity from bottom camera tongue feature**: Euclidean velocity of (x,y) position". No justification is offered for using one camera rather than both.

---

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, per trial:
1. **Frame-to-frame speed.** `v[n] = sqrt(dx² + dy²) × 400`, with a leading 0 pad, where 400 is the hard-coded `VIDEO_FPS`. The recorded `frameTimes` are *not* used for the derivative — a constant frame rate is assumed. No smoothing is applied to x or y before differencing.
2. **Visibility masking.** `v` is set to NaN wherever DLC likelihood `< 0.5`. (In practice the authors already set x and y to NaN for likelihood ≤ 0.9, so the effective cut is 0.9 and frames in 0.5–0.9 come out NaN anyway.)
3. **Resampling onto the neural grid.** `np.interp` maps the valid samples of `v` onto the 500 bin centres, with NaN outside the frame range. A separate `np.interp` of the raw likelihood over *all* frames gives the per-bin visibility flag (`conf_interp >= 0.5`). Note the velocity interpolation bridges gaps between valid frames, but the independent visibility mask marks those bins `not visible`, so bridged values do not reach the output.
4. **No normalisation** — values stay in pixels/s (up to the 400 Hz scale factor).

The resulting `not visible` fraction is 0.891, close to the reference's 0.875.

ii.
```python
        # Compute velocity at video frame rate
        dx = np.diff(x)
        dy = np.diff(y)
        v = np.sqrt(dx**2 + dy**2) * VIDEO_FPS  # pixels/second
        v = np.concatenate([[0], v])  # pad to same length

        # Set velocity to NaN where confidence is low
        low_conf = conf < DLC_CONF_THRESH
        v[low_conf] = np.nan

        # Aligned video times
        aligned_ft = ft - vidshift - gc

        # Interpolate velocity to neural time bins
        valid = ~np.isnan(v) & ~np.isnan(aligned_ft)
        if np.sum(valid) < 2:
            continue

        v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
        conf_interp = np.interp(taxis, aligned_ft, conf, left=0, right=0)

        velocity[:, out_idx] = v_interp
        visible[:, out_idx] = conf_interp >= DLC_CONF_THRESH
```

iii. Step 5 Key Decision 7: "Euclidean velocity of (x,y) position". `VIDEO_FPS = 400` is sourced from Step 3: "Video frame rate | 400 Hz | Paper". `DLC_CONF_THRESH = 0.5` is stated in the constants block as the "DLC confidence threshold for visibility"; no source is cited for the 0.5 value.

---

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, pooling all trials and all bins. The 50th percentile is taken over the velocities of bins flagged visible (NaN-safe). Bins flagged visible and below it get 0, visible and at-or-above it get 1, everything else — including bins flagged visible whose velocity is NaN — gets 2 (`not_visible`). The array is initialised to 2, so class 2 is the fall-through. Classes 0 and 1 come out at 0.055 each, class 2 at 0.891.

ii.
```python
def discretize_velocity(velocity, visible):
    """Discretize velocity per session.
    0: < 50th percentile
    1: >= 50th percentile
    2: not visible
    """
    n_time, n_trials = velocity.shape
    result = np.full((n_time, n_trials), 2, dtype=np.int32)  # default: not visible

    vis_vals = velocity[visible]
    if len(vis_vals) == 0:
        return result

    thresh = np.nanpercentile(vis_vals, 50)

    result[visible & (velocity < thresh)] = 0
    result[visible & (velocity >= thresh)] = 1

    return result
```

iii. Directly from the Decoder Task specification ("discretized with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile, 2: not visible"), restated in Step 5's mapping table: "Discretize per-session: <50th=0, >=50th=1, not_visible=2".

---

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the behaviour clock, so a per-session offset is computed once from the bitcode pulse recorded on both streams: `vidshift = mode(sglx.bitcode.bitstart) / sglx.fs − mode(bp.ev.bitStart)`. Frame times in go-cue-relative seconds are then `frameTimes − vidshift − goCue[trial]`, and those times are the x-coordinates for the `np.interp` onto the shared 500-bin centres. If anything in the offset computation raises, a bare `except` substitutes a default of 0.5 s ("padSec").

ii.
```python
        try:
            sglx = obj['sglx']
            bp_bitstart = np.array(ev['bitStart']).flatten()
            sglx_bitstart = np.array(sglx['bitcode']['bitstart']).flatten()
            sglx_fs = np.array(sglx['fs']).flatten()[0]

            mode_bp = scipy_stats.mode(bp_bitstart[~np.isnan(bp_bitstart)], keepdims=False).mode
            mode_sglx = scipy_stats.mode(sglx_bitstart[~np.isnan(sglx_bitstart)], keepdims=False).mode
            session['vidshift'] = mode_sglx / sglx_fs - mode_bp
        except:
            session['vidshift'] = 0.5  # default padSec
```

```python
        aligned_ft = ft - vidshift - gc
        ...
        v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
```

iii. Step 6: "Video offset computation matching `findVideoOffset.m`". Step 1 notes `loadMotionEnergy.m` "Interpolates motion energy to neural data timebase using frameTimes ... Handles video offset (findVideoOffset function)", which is the model for interpolating rather than bin-averaging the camera streams.

---

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` DLC tracking, using **camera index 1** and the feature `top_paw`. Camera 1 is the bottom camera. `bottom_paw` is not used. The same `x, y, likelihood, frameTimes` fields are read, and the same session `vidshift` and `goCue` are applied. (As with the tongue, the code comment and Step 5 mislabel camera 1 as the "side camera".)

ii.
```python
    # Paw velocity (side camera, 'top_paw' feature)
    paw_vel, paw_vis = compute_velocity_timeseries(
        session, cam_idx=1, feat_name='top_paw',
        trial_indices=trial_indices, vidshift=vidshift)
    paw_disc = discretize_velocity(paw_vel, paw_vis)
```

iii. Step 5 Key Decision 8: "**Paw velocity from side camera paw feature**: Euclidean velocity of (x,y) position". No rationale is given for preferring `top_paw` over `bottom_paw`.

---

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue — it is the same function, `compute_velocity_timeseries`, with a different camera index and feature name. Frame-to-frame Euclidean displacement × 400 Hz, no position smoothing, NaN where likelihood < 0.5, `np.interp` onto the 500 bin centres, separate interpolated likelihood giving the visibility mask, no normalisation (only one camera is involved, so there is nothing to reconcile). Resulting `not visible` fraction 0.167, vs the reference's 0.186.

ii.
```python
def compute_velocity_timeseries(session, cam_idx, feat_name, trial_indices, vidshift):
    """Compute velocity for a DLC feature, aligned to neural time bins."""
    ...
        dx = np.diff(x)
        dy = np.diff(y)
        v = np.sqrt(dx**2 + dy**2) * VIDEO_FPS  # pixels/second
        v = np.concatenate([[0], v])  # pad to same length
        low_conf = conf < DLC_CONF_THRESH
        v[low_conf] = np.nan
        ...
        v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
```

iii. Same as 7-b — the shared function is the stated design ("Euclidean velocity of (x,y) position" for both features).

---

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same `discretize_velocity` call, so: per-session 50th percentile over all visible bins across all trials; 0 below, 1 at-or-above, 2 for not-visible (and for visible-but-NaN). Classes come out at 0.417 / 0.417 / 0.167.

ii.
```python
    paw_disc = discretize_velocity(paw_vel, paw_vis)
```
(function body as quoted in 7-c)

iii. Same as 7-c: the Decoder Task specification, restated in Step 5's mapping table.

---

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue: the session-constant `vidshift` from the bitcode pulse is subtracted from `frameTimes`, then the trial's `goCue`, then the samples are interpolated onto the shared 500-bin grid. The frame times used are those of camera 1 (the paw's own camera), read from that camera's per-trial `frameTimes`, so a frame-count difference between the two cameras cannot misalign it.

ii.
```python
    cam = session['traj'][cam_idx]     # cam_idx = 1 for the paw
    ...
        ft = cam['frameTimes'][t]
        ...
        aligned_ft = ft - vidshift - gc
        v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
```

iii. Same as 7-d — one shared offset and one shared grid for every stream.

---

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Preferentially the standalone `motionEnergy_<animal>_<date>.mat` file sitting beside the data structure; if no such file is found, the copy embedded in `obj.me` is used instead. `load_motion_energy_file` handles two of the three layouts that occur across the 44 files: a struct `{data, moveThresh}`, and a bare object array. `moveThresh` is read but never used.

It does **not** handle the third layout, `{data: {data, moveThresh}}`, which occurs in 3 files. For those the function falls through both `isinstance` branches, returns an empty list, and — because a motion-energy *file* was found, the `obj.me` fallback is never reached — every bin of every trial in those sessions is emitted as class 2. Verified: `JEB15_2022-07-26`, `JEB15_2022-07-28` and `JEB24_2023-10-31` all load as `me['data']` being a `dict`, and all three show `motion_energy: 0.000 / 0.000 / 1.000` in the verification output. This is why the overall `no_video` fraction is 0.108 against the reference's 0.038.

ii.
```python
def load_motion_energy_file(me_path):
    """Load motion energy from separate .mat file."""
    d = scipy.io.loadmat(me_path, simplify_cells=True)
    me = d['me']

    # Format 1: struct with 'data' and 'moveThresh' fields
    if isinstance(me, dict) and 'data' in me:
        data = me['data']
        thresh = me.get('moveThresh', 10)
    elif isinstance(me, np.ndarray) and me.dtype == object:
        # Format 2: just an array of per-trial arrays (no struct wrapper)
        data = me
        thresh = 10  # default
    else:
        return [], 10.0

    me_trials = []
    if isinstance(data, list):
        ...
    elif isinstance(data, np.ndarray):
        ...
    return me_trials, float(thresh)
```

```python
    me_trials = session.get('me_embedded', None)
    has_me_file = me_path is not None and os.path.exists(me_path)

    if has_me_file:
        me_trials_loaded, me_thresh = load_motion_energy_file(me_path)
        me_aligned = compute_motion_energy_timeseries(
            session, trial_indices, me_trials_loaded, vidshift)
    elif me_trials is not None:
        ...
```

iii. Step 1: "`loadMotionEnergy.m` ... Finds motionEnergy_*.mat file using pattern matching; Extracts me.data (video motion energy at 400 Hz)". Step 2: "Motion energy integrated in data_structure (obj.me) for some, separate files for others" — hence the two-source design. The agent believed it had covered the layouts ("Fixed `load_motion_energy_file` to handle both formats") and attributed the three all-class-2 sessions to the data: Step 10 Check 7 states "41/44 sessions have motion energy data (3 have no video)."

---

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling. The value is already one scalar per camera frame (the paper computes it per pixel as a median difference over ±5 frames and reduces each frame to the 99th percentile across pixels), so nothing is smoothed, differenced or combined. The per-trial trace is truncated to the shorter of itself and the frame-time vector, NaN samples are dropped, and `np.interp` places it on the 500 bin centres with NaN outside the frame range.

ii.
```python
        me_trial = me_trials[t]
        ft = cam['frameTimes'][t]
        ...
        aligned_ft = ft - vidshift - gc

        # Trim to match me_trial length
        min_len = min(len(aligned_ft), len(me_trial))
        aligned_ft = aligned_ft[:min_len]
        me_data = me_trial[:min_len]

        valid = ~np.isnan(aligned_ft) & ~np.isnan(me_data)
        if np.sum(valid) < 2:
            continue

        me_interp = np.interp(taxis, aligned_ft[valid], me_data[valid], left=np.nan, right=np.nan)
        me_aligned[:, out_idx] = me_interp
```

iii. Step 1 describes `loadMotionEnergy.m` as interpolating motion energy onto the neural timebase using `frameTimes`, which is exactly what is reproduced here. The reference code's `me.move = me.data > me.moveThresh` binarisation is deliberately not used, since the Decoder Task specifies a 50th-percentile split instead.

---

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per session, over all trials and bins: the 50th percentile of the non-NaN interpolated values. Below → 0, at-or-above → 1, NaN → 2 (`no_video`), with the array initialised to 2 as the fall-through. Distribution 0.446 / 0.446 / 0.108 — the 0.108 inflated by the 3 sessions whose motion energy failed to load.

ii.
```python
def discretize_motion_energy(me_aligned):
    """Discretize motion energy per session.
    0: < 50th percentile
    1: >= 50th percentile
    2: no video (NaN)
    """
    n_time, n_trials = me_aligned.shape
    result = np.full((n_time, n_trials), 2, dtype=np.int32)  # default: no video

    valid = ~np.isnan(me_aligned)
    valid_vals = me_aligned[valid]
    if len(valid_vals) == 0:
        return result

    thresh = np.nanpercentile(valid_vals, 50)

    result[valid & (me_aligned < thresh)] = 0
    result[valid & (me_aligned >= thresh)] = 1

    return result
```

iii. Straight from the Decoder Task specification ("0: < 50th percentile, 1: >= 50th percentile, 2: no video"), restated in the Step 5 mapping table.

---

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset, same grid as the tracking. Motion energy has one value per frame of camera 0 (the side camera), so camera 0's `frameTimes` are used, corrected by the session `vidshift` and the trial's `goCue`, then interpolated onto the 500 bin centres. If camera 0's frame times for a trial are missing or all-NaN, the code **synthesises** them as `np.arange(n_frames) / 400 + 0.5`.

ii.
```python
    cam = session['traj'][0]  # Use camera 0 (bottom) frame times

    for out_idx, t in enumerate(trial_indices):
        ...
        ft = cam['frameTimes'][t]
        if ft is None or len(ft) == 0 or np.all(np.isnan(ft)):
            # Fallback: create frame times assuming 400 Hz
            n_frames = len(me_trial)
            ft = np.arange(n_frames) / VIDEO_FPS + 0.5  # 0.5s pad offset

        aligned_ft = ft - vidshift - gc
```

iii. Step 1's description of `loadMotionEnergy.m` ("Interpolates motion energy to neural data timebase using frameTimes ... Handles video offset"). The 400 Hz synthetic fallback follows the paper's stated camera frame rate and the 0.5 s `padSec` used as the `vidshift` default.

---

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms, mostly "skip and fall through to a default":

- **Two MATLAB formats**: `load_session` tries h5py and falls back to `scipy.io.loadmat` inside a bare `except: pass`.
- **NaN go cue**: those trials are removed from `trial_indices` and additionally skipped inside the spike loop.
- **Missing / all-NaN `frameTimes`, missing `ts`, fewer than 2 valid samples**: the trial is skipped in `compute_velocity_timeseries`, leaving all 500 bins NaN → class 2 (`not visible`). Nothing is interpolated or filled.
- **Untracked frames**: likelihood below threshold → velocity NaN → class 2.
- **Missing motion energy frame times**: uniquely, these are *fabricated* at 400 Hz with a 0.5 s pad rather than skipped.
- **Mismatched lengths** between motion energy and frame times: truncated to the shorter.
- **Missing bitcode / `sglx`**: `vidshift` silently defaults to 0.5 s.
- **Missing lick cell arrays**: per-trial `try/except` appends an empty array.
- **Missing feature in a camera's `featNames`**: `find_feature_index` returns `None` and the whole stream comes back all-NaN → all class 2, with no warning.
- **Sessions with no clusters, <2 valid trials, or <10 units**: dropped with a printed message.
- **Duplicate session file**: hard-coded into `SKIP_FILES`.
- **Doubly-nested motion-energy struct**: *not* handled — returns an empty list silently (see 9-a).
- **Trials after the ephys recording ends**: *not* handled — 61 trials across 2 sessions are kept with all-zero firing rates for every neuron.

The pervasive bare `except:` clauses mean several of these failures produce plausible-looking all-class-2 output rather than an error.

ii.
```python
        if ts is None or ft is None or np.isnan(gc):
            continue
        if ft is None or len(ft) == 0:
            continue
        if np.all(np.isnan(ft)):
            continue
```
```python
        except:
            session['vidshift'] = 0.5  # default padSec
```
```python
    feat_idx = find_feature_index(cam['featNames'], feat_name)
    if feat_idx is None:
        return velocity, visible
```
```python
    if n_valid < 2:
        print(f"  {session_id}: Too few valid trials ({n_valid}), skipping")
        return None
    ...
    if n_units < MIN_UNITS:
        print(f"  {session_id}: Too few units after filtering ({n_units}), skipping")
        return None
```

iii. Step 6: "Full trial curation (exclude early lick, stim trials, NaN goCue)". Step 10: "**No NaN**: Clean data throughout" and "**All-zero neural data**: Sessions 36, 43 (both JEB24) have all-zero late trials where recording ended. Data artifact, not a conversion bug." and "**Duplicate detection**: JEB23_2023-10-20 is duplicate of JEB23_2023-10-19 ... Excluded." The three motion-energy sessions were characterised (incorrectly) as "3 have no video". The rationale for leaving NaN-derived bins as class 2 rather than interpolating is implicit in the Decoder Task's provision of a dedicated `not visible` / `no video` class.

---

## 11-a. What are the most time-consuming steps of the code?

i. File I/O dominates, with spike binning a clear second. The script instruments both per session and prints them. Summing the full run: **loading 84.8 s** and **spike align/bin/smooth 41.8 s** out of **133.3 s** total for 44 sessions — so ~64% loading, ~31% binning, ~5% everything else (video velocity, motion energy, discretisation, pickling). Loading is heavy partly because the HDF5 reader dereferences per-trial object references one at a time for `lickL`, `lickR`, `traj.ts` and `traj.frameTimes`, and reads each cluster's full `tm` vector.

The agent did **not** record this analysis anywhere: CONVERSION_NOTES.md Step 6 ("Code inefficiencies identified", "Code speedups added") and Step 7 ("Run Time Estimates" table) were left out of the final notes entirely.

ii.
```python
def process_session(fpath, me_path=None, show_processing=False, session_id=''):
    t0 = time.time()
    session = load_session(fpath)
    ...
    t_load = time.time() - t0
    ...
    t1 = time.time()
    trialdat = align_and_bin_spikes(neurons, goCue, n_trials, EDGES, DT)
    t_bin = time.time() - t1
    ...
    print(f"  {session_id}: {len(neurons)} raw -> {n_units} units, {n_valid} trials "
          f"(load={t_load:.1f}s, bin={t_bin:.1f}s)")
```

iii. No documented justification. The 133 s total is well inside the instructions' 15-minute budget, so no optimisation pass was triggered.

---

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three groups:

1. **`align_and_bin_spikes` — the big one.** A Python double loop over (cluster × trial), and inside it a full-array boolean comparison `trial == trial_num` over that cluster's entire spike vector, plus a separate `np.histogram` call and a separate `smooth_causal` call (which itself loops over columns and calls `np.convolve`). Cost is O(n_clusters × n_trials × n_spikes). This is exactly the loop the reference eliminates: one `np.histogram2d` over the (trial, time) grid per cluster, and one `gaussian_filter1d(..., axis=1)` over the whole (trials × bins) matrix. This accounts for the 42 s of binning time.
2. **The three per-trial output loops** building `lick_dir`, `context` and `outcome` one trial at a time from Python `if`/`elif`. These are pure boolean indexing and vectorise trivially (`outcome[hit] = 1`, `np.where(right[hit], ...)`, etc., as the reference does).
3. **The per-trial loops in `compute_velocity_timeseries` / `compute_motion_energy_timeseries`.** These are genuinely hard to vectorise — each trial has a different number of camera frames, so there is no rectangular array — and the reference keeps the same loops.

ii. The un-vectorised inner loop:
```python
    for ni, neuron in enumerate(neurons):
        trialtm = neuron['trialtm']
        trial = neuron['trial']
        for t in range(n_trials):
            trial_num = t + 1  # MATLAB 1-indexed
            gc = goCue[t]
            if np.isnan(gc):
                continue
            spike_mask = trial == trial_num
            if not np.any(spike_mask):
                continue
            aligned = trialtm[spike_mask] - gc
            counts, _ = np.histogram(aligned, bins=edges)
            fr = counts.astype(np.float32) / dt
            fr = smooth_causal(fr, KERNEL, 'reflect')
            trialdat[ni, :, t] = fr
```

A vectorisable per-trial output loop:
```python
    outcome = np.full(n_valid, 2, dtype=np.int32)  # default: ignore
    for out_idx, t in enumerate(trial_indices):
        if session['hit'][t]:
            outcome[out_idx] = 1
        elif session['miss'][t]:
            outcome[out_idx] = 0
```

iii. No documented justification — the notes never identify any loop as a bottleneck or discuss vectorisation. The implicit justification is that total runtime (133 s) was acceptable.

---

## 11-c. What processing does the code repeat multiple times?

i. Little is recomputed, with two exceptions:

1. **Discarded trials are fully processed.** `align_and_bin_spikes` bins *and smooths* every cluster on **all** `Ntrials` trials, and only afterwards is `trialdat[:, :, trial_indices]` taken. The ~9% of trials removed as early-lick / photostim / NaN-go-cue are histogrammed and convolved for nothing.
2. **`load_session` may read a file twice.** On a v5 file, `load_session_h5` is attempted first and fails; the cost is small because `h5py.File` rejects a non-HDF5 file on the magic bytes, but the pattern would be expensive if the h5 path failed partway through.

Things that are correctly computed once: `vidshift` per session (not per trial), the bin grid `EDGES` / `TIME_CENTERS` at module level, the smoothing kernel `KERNEL` at module level, and the per-session discretisation thresholds. The per-trial `input` arrays all reference one shared `TIME_CENTERS` object rather than 13,823 copies.

ii.
```python
    trialdat = align_and_bin_spikes(neurons, goCue, n_trials, EDGES, DT)
    t_bin = time.time() - t1

    # Filter to valid trials
    trialdat = trialdat[:, :, trial_indices]  # (n_neurons, n_timebins, n_valid)
```

```python
KERNEL = causal_gaussian_kernel(SMOOTH_WIN)
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_CENTERS = EDGES[:-1] + DT / 2
```

iii. Not documented. The spike-binning over all trials is a natural consequence of separating `align_and_bin_spikes` (which is written against the raw trial numbering) from the trial filter applied in `process_session`.

---

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly in loading, plus the discarded-trial binning from 11-c:

- **`lickL` / `lickR`** are dereferenced and read per trial — 2 × `Ntrials` individual HDF5 object-reference resolutions per session, ~600 per session — and are never used by any output. (Lick direction is derived from `hit`/`miss`/`L`/`R` instead.)
- **`clu.tm`**, each cluster's full whole-session spike-time vector, is read into memory for every surviving cluster and never used; only `trialtm` and `trial` are needed.
- **`bp.no`, `bp.ev.sample`, `bp.ev.delay`** are loaded and never read.
- **`me.moveThresh`** is loaded and returned by `load_motion_energy_file` and never used.
- **`obj.me`** is read into `session['me_embedded']` even when a standalone `motionEnergy_*.mat` file exists (which is the case for all 44 sessions), so the embedded copy is always discarded.
- **Binning and smoothing of clusters later dropped by the 1 Hz filter** — unavoidable, since the rate must be computed to apply the filter, and the reference does the same.
- **Binning and smoothing of trials later dropped** by the trial filter — avoidable (see 11-c).

Everything computed after loading (velocities, motion energy, discretisation) does reach the output.

ii.
```python
        for t in range(n_trials):
            try:
                ref = lickL_ds[0, t] if lickL_ds.ndim > 1 else lickL_ds[t]
                licks = np.array(f[ref]).flatten()
                session['lickL'].append(licks)
            except:
                session['lickL'].append(np.array([]))
```
```python
                    tm_ref = dereffed['tm'][ni, 0]
                    ...
                    tm = np.array(f[tm_ref]).flatten()
                    ...
                    neurons.append({'quality': quality, 'trialtm': trialtm, 'trial': trial})
```
```python
        if 'me' in obj:
            me_ds = obj['me']
            me_data = []
            ...
            session['me_embedded'] = me_data
```

iii. Not documented. The lick times and `no` flag appear to be leftovers from the Step 5 plan, whose mapping table listed "obj.bp.L/R + lickL/lickR" as the source for lick direction before the agent settled on deriving it from `hit`/`miss`. `obj.me` is read unconditionally as the fallback source for motion energy.
