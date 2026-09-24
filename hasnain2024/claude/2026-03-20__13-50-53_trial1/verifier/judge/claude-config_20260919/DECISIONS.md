# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a session registry rather than globbing the data folders. Two module-level lists, `EPHYS_SESSIONS` (25 fixed-delay sessions) and `RANDOMIZED_DELAY_SESSIONS` (19 randomized-delay sessions), each entry a tuple `(animal, date, probes_for_ALM, data_subdir)`, are concatenated into `ALL_SESSIONS` (44 sessions). The lists were transcribed from the authors' `code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` scripts, with explicit in-code comments recording the exclusions (`JEB23 2023-10-20` commented out in the loading script; `JEB24 2023-10-03/04` absent from the loading scripts; `JEB4`/`JEB5` have no data files; the two behavior-only folders `DelayInhibition_BilatMC_Behavior` and `GoCueInhibition_BilatMC_Behavior` are ignored because they contain no neural data).

`main()` loops over `ALL_SESSIONS` and calls `process_session`, which calls `load_session_data`. That function reads the session `data_structure_<anm>_<date>.mat` with `mat73.loadmat` (MATLAB v7.3 / HDF5) and falls back to a hand-written `_load_v5_session` (built on `scipy.io.loadmat`) when `mat73` raises `TypeError` on the older v5 files. The companion `motionEnergy_<anm>_<date>.mat` is always read with `scipy.io.loadmat` and unwrapped through three different on-disk layouts.

ii.
```python
EPHYS_SESSIONS = [
    # (animal, date, probes_for_ALM, data_dir)
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
]
RANDOMIZED_DELAY_SESSIONS = [
    ('JEB11', '2022-05-10', [1], 'RandomizedDelay_Ephys_Behavior'),
    # JEB23 2023-10-20 commented out in loading script
    # JEB24 2023-10-03 and 2023-10-04 not in loading scripts
    ...
]
ALL_SESSIONS = EPHYS_SESSIONS + RANDOMIZED_DELAY_SESSIONS
```

```python
def load_session_data(anm, date, data_dir):
    import mat73, scipy.io
    data_path = os.path.join(DATA_ROOT, data_dir, f'data_structure_{anm}_{date}.mat')
    try:
        obj = mat73.loadmat(data_path)['obj']
    except TypeError:
        # MATLAB v5 format - use scipy and convert to similar structure
        obj = _load_v5_session(data_path)
    ...
    me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
    ...
```

```python
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    result = process_session(anm, date, probes, data_dir, ...)
```

iii. From CONVERSION_NOTES.md Step 2/Step 4: the loading scripts are treated as the definitive record of which sessions and probes entered the paper's analysis, so sessions present on disk but absent from (or commented out in) those scripts are excluded — "RandomizedDelay sessions: 19 in scripts / 22 in data dir → Use 19 from scripts". The dual reader was added during the full run (trajectory steps 130–137) after `mat73` failed on several files: "Some files are MATLAB v5 format, not v7.3. I need to handle both formats." The motion-energy unwrapping was added iteratively for the same reason (steps 120, 128), citing the reference: "The reference code handles this: `if isstruct(me.data), me.data = me.data.data; end`".

## 1-b. How are the data split into subjects (mice)?

i. The animal id is the first element of each session tuple, so it is taken from the file naming convention rather than from inside the file (`obj.ex.anm` is never read). `process_session` returns the animal string, `main()` accumulates it in `all_animals` and in a `subjects_set`, and at assembly `subjects` is the sorted unique set with `subject_idx` an int array giving each session's index into it. The result is 14 subjects across 44 sessions, matching the reference exactly (EKH1, EKH3, JEB11, JEB12, JEB13, JEB14, JEB15, JEB19, JEB23, JEB24, JEB6, JEB7, JGR2, JGR3).

ii.
```python
all_animals.append(anm)
subjects_set.add(anm)
...
subjects = sorted(subjects_set)
subject_idx = np.array([subjects.index(anm) for anm in all_animals], dtype=np.int64)
...
'subjects': subjects,
'subject_idx': subject_idx,
```

iii. Not explicitly argued in CONVERSION_NOTES.md; the notes simply enumerate the mice per folder ("Mice: EKH1, EKH3, JEB6, JEB7, JEB13, JEB14, JEB15, JEB19, JGR2, JGR3" and "JEB11, JEB12, JEB23, JEB24") and record "Unique mice | 14". The animal identity comes from the same source used to select the sessions (the authors' loading scripts / filenames), so no extra justification was needed.

## 1-c. How are the data split into sessions?

i. One session = one entry of `ALL_SESSIONS` = one `data_structure_*.mat` file. The folder the file lives in is carried in the tuple, so fixed-delay and randomized-delay sessions are loaded uniformly and concatenated into a single 44-element list of sessions; each becomes one element of `neural`, `input`, `output`, and `brain_region_idx`. Sessions with two ALM probes (`JEB15` ×3) have their units from both probes concatenated into one population for that session. A session would be dropped if it ended up with `< 2` valid trials or `< 10` neurons; in the full run no session was dropped (44/44 kept).

ii.
```python
sessions = ALL_SESSIONS
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    result = process_session(anm, date, probes, data_dir, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
    ...
    session_info.append({'animal': anm, 'date': date, 'data_dir': data_dir,
                         'probes': probes, 'n_neurons': ..., 'n_trials': ...})
```
```python
if n_valid < 2:
    print(f"  WARNING: Only {n_valid} valid trials, skipping session"); return None
...
if n_neurons_final < 10:
    print(f"  WARNING: Only {n_neurons_final} neurons after filtering, skipping session"); return None
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 1: "Include all 44 ephys sessions: Both Ephys_Behavior (25) and RandomizedDelay (19). Randomized delay sessions have context=DR for all trials." The ≥10-unit session criterion is taken from the paper (Step 3: "Min session quality | >= 10 units | Paper: 'at least 10 units'"). Key Decision 8: "Brain region: All neurons from ALM (some sessions have 2 probes both in ALM region)", and Step 4 resolves the JEB15 case as "Use both probes per loading script (concatenate)".

## 1-d. How are the data split into trials?

i. Trials are taken directly from the Bpod table: `ntrials_total = int(bp['Ntrials'])`, and every per-trial field (`R`, `L`, `hit`, `miss`, `no`, `autowater`, `early`, `stim.enable`, `ev.goCue`) is flattened to a vector of that length. Spikes carry their trial number in `clu.trial` (1-based), so the per-trial split of the neural data is by `trial_arr == j + 1`. Camera data is already stored one cell per trial (`traj[view]['ts'][trix]`, `traj[view]['frameTimes'][trix]`), and motion energy likewise (`me_raw[trix, 0]`). Nothing has to be reconstructed. Unlike the reference, the AI does **not** truncate the `bp` columns to `Ntrials`.

ii.
```python
bp = obj['bp']
ntrials_total = int(bp['Ntrials'])
ev = bp['ev']
goCue = np.array(ev['goCue']).flatten()
R = np.array(bp['R']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
...
for j in range(ntrials_total):
    trial_num = j + 1  # MATLAB 1-indexed
    spk_mask = trial_arr == trial_num
```

iii. Not separately argued. CONVERSION_NOTES.md Step 2 lists `obj.bp` as holding "Ntrials, R, L, hit, miss, no, autowater, early, stim, ev" and `obj.clu` as holding "trial, trialtm" per cluster, i.e. the trial structure is read straight out of the file. Step 10 reports an independent verification on `JEB14_2022-08-22`: "Raw: 517 total, 43 early, 0 stim, 3 no, 1 overlap → 472 valid (matches)".

## 1-e. How are trials filtered based on quality controls?

i. A single boolean mask combining **four** conditions: drop early-lick trials (`bp.early`), drop photostimulation trials (`bp.stim.enable`), drop no-response/ignore trials (`bp.no`), and additionally require the trial to be a hit or a miss. The last two are redundant with each other. Sessions whose `bp` has no usable `stim` struct get an all-false stim mask. There is **no** filter on trials that fall after the end of the ephys recording. Across the dataset 11,985 of ~15,155 trials survive (the reference keeps 13,762). The verifier flagged 30 retained trials in sessions 36 and 43 whose neural data is entirely zero; these were knowingly kept.

ii.
```python
stim_enable = np.zeros(ntrials_total, dtype=bool)
stim = bp.get('stim')
if stim is not None and isinstance(stim, dict):
    se = stim.get('enable')
    if se is not None:
        stim_enable = np.array(se).flatten().astype(bool)

# Trial selection: exclude early, stim, and no-response (ignore) trials
valid_mask = ~early & ~stim_enable & ~no
# Also need a lick response (hit or miss)
valid_mask = valid_mask & (hit | miss)
valid_trials = np.where(valid_mask)[0]  # 0-indexed
```

iii. CONVERSION_NOTES.md Step 3 ("Trial curation: Exclude early lick trials (`early=1`), stim trials (`stim.enable=1`), ignore/no-response trials (`no=1`)") and Step 5 Key Decision 4 ("Trial exclusion: Exclude early, stim.enable, and no-response trials. Keep hit and miss"). The early/stim exclusions are attributed to the reference `findTrials.m` condition strings such as `'R&hit&~stim.enable&~autowater&~early'`. No justification is given anywhere for also dropping ignore trials, and the Decoder Task specification requiring a "none"/"ignore" class is never discussed. For the all-zero-neural trials, Step 9 says: "These are likely recording artifacts; trials retained for completeness."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu[probe-1]`, the spike-sorted clusters of the ALM probe(s) named in the session registry. Three of its fields are used: `quality` (manual curation label, for filtering), `trial` (1-based trial index of each spike) and `trialtm` (spike time relative to that trial's start). The alignment times come from `obj.bp.ev.goCue`. Units from both probes of a two-probe session are concatenated along the neuron axis.

ii.
```python
for probe_num in probes:
    probe_idx = probe_num - 1
    clu_probe = clu_data[probe_idx]
    if isinstance(clu_probe, dict):
        valid_clu = get_valid_cluster_indices(clu_probe)
        all_cluster_indices.append((probe_idx, valid_clu))
...
for i, clu_idx in enumerate(valid_clu):
    trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
    trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()
```

iii. CONVERSION_NOTES.md Step 1 documents the reference chain `findClusters → alignSpikes → getSeq → removeLowFRClusters` and Step 2 records the `obj.clu` fields ("quality, site, tm, trial, trialtm, spkWavs"). Step 5's mapping table lists "obj.clu (spike times) → neural | Align to goCue, bin at 10ms, smooth, convert to firing rate | alignSpikes, getSeq, removeLowFRClusters".

## 2-b. How is the `neural` data processed?

i. For every kept cluster and every trial, the go-cue-aligned spike times are histogrammed into the fixed bin edges, divided by `DT` to give spikes/s, and smoothed along time with a **causal** Gaussian kernel of `N = 15` bins (`scipy.signal.windows.gaussian(15, std=15/6)` with the first `N//2` taps zeroed and the kernel renormalised), using a `'reflect'` boundary. Nothing else is applied: no baseline subtraction, no normalisation, no z-scoring; stored values are firing rates in Hz as `float32`, shaped `(n_neurons, 500)` per trial. Note the boundary handling prepends `x[:N]` *unreversed* rather than a true mirror, so it is a repeat-pad rather than a reflect-pad.

ii.
```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    kern = np.array(scipy_windows.gaussian(N, std=N/6.0))  # gausswin equivalent
    kern[:N//2] = 0  # causal: zero out first half
    kern = kern / kern.sum()
    if bctype == 'reflect':
        x_filt = np.concatenate([x[:N], x], axis=0); trim = N
    ...
    out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
    out = out[trim:]
```
```python
aligned = trialtm_arr[spk_mask] - align_times_all[j]
counts, _ = np.histogram(aligned, bins=edges)
fr = counts.astype(np.float32) / DT
trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```
```python
SMOOTH_WINDOW = 15  # bins, causal Gaussian
BC_TYPE = 'reflect'
```

iii. CONVERSION_NOTES.md Step 1 identifies `mySmooth` (utils) as a "Causal Gaussian kernel smoothing" routine and records the reference parameters `smooth = 15` and `bctype = 'reflect'` from `WorkingWithDataObjs.m`; Step 4 resolves the boundary ambiguity as "bctype: 'reflect' or 'none' … Use 'reflect' as in WorkingWithDataObjs.m". Step 1 also transcribes the reference pipeline step "Smooth: `mySmooth(N/dt, smooth, bctype)` for single trial (spks/sec)", which is what the code reproduces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, plus one session-level filter. (1) The manual curation label `clu.quality` is stripped and lower-cased and compared against `{garbage, gabrga, noisy, real?}`; anything else is kept, including multi-units and clusters with no label. (2) After binning and smoothing, any unit whose mean rate — averaged over all time bins and **all** trials of the session, including trials that will later be discarded — is not greater than 1 Hz is dropped. (3) A session with fewer than 10 surviving neurons would be skipped. This leaves 2,457 units over 44 sessions (17–141 per session, mean 55.8); the reference keeps 1,954, the difference coming mostly from the reference also dropping the label `poor`.

ii.
```python
LOW_FR = 1.0  # Hz, minimum firing rate
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}

def get_valid_cluster_indices(clu_probe, excluded_qualities=EXCLUDED_QUALITIES):
    """Find valid cluster indices matching findClusters.m with quality='all'."""
    for i, q in enumerate(qualities):
        q_stripped = q.strip() if isinstance(q, str) else ''
        if q_stripped.lower() not in {e.lower() for e in excluded_qualities}:
            valid.append(i)
```
```python
def remove_low_fr_neurons(trialdat, low_fr):
    mean_fr = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
    keep = mean_fr > low_fr
    return trialdat[:, keep, :], keep
```

iii. CONVERSION_NOTES.md Step 1: "findClusters … Filter clusters by quality (exclude 'garbage', 'gabrga', 'noisy', 'real?')" and "quality = {'all'}". Step 4 resolves the firing-rate ambiguity explicitly: "lowFR | 0.5 (some scripts) or 1 | 'exceeding 1 Hz' | Use 1 Hz to match paper", quoting the paper's "all units with firing rates exceeding 1 Hz were included". The `removeLowFRClusters.m` docstring in the code says the implementation matches "mean of mean across conditions".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By a single subtraction, per spike: `trialtm` (already on the behaviour clock, relative to the trial's own start) minus `goCue` of that same trial, giving seconds from go cue onset. There is no interpolation, resampling or additional offset for the neural stream (the clock correction is only needed for the video streams). `ALIGN_EVENT = 'goCue'` and `metadata['temporal_alignment_event'] = 'Go cue onset'`.

ii.
```python
ALIGN_EVENT = 'goCue'
goCue = np.array(ev['goCue']).flatten()
align_times_all = goCue  # for all trials
...
# Align: trialtm_aligned = trialtm - alignTime
aligned = trialtm_arr[spk_mask] - align_times_all[j]
```

iii. CONVERSION_NOTES.md Step 1 records the reference function directly: "alignSpikes … Align spike times to event (goCue): `trialtm_aligned = trialtm - event_time`", with `alignEvent = 'goCue'` from the reference parameters. Step 10 reports the independent verification: "Independently recomputed spike alignment, binning, and smoothing from raw .mat — All 214 trials x 48 neurons: exact match (max absolute diff = 0.0)".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms** bins (`DT = 1/100`) over a fixed window of −2.5 s to +2.5 s from the go cue, i.e. 500 bins per trial for every trial of every session. The grid is `edges = np.arange(TMIN, TMAX + DT, DT)` and the reported time axis is the bin centres `edges[:-1] + DT/2`. Spikes are binned directly onto this grid, so there is no rebinning of the neural data; the three camera streams, which are sampled at 400 Hz, are *resampled* onto the same grid by linear interpolation (`scipy.interpolate.interp1d`) rather than by averaging frames within a bin. `metadata['time_bin_size'] = 10.0` ms, `off_start = -2.5`, `off_end = 2.5`. (The human reference uses 5 ms / 1000 bins.)

ii.
```python
TMIN = -2.5  # seconds
TMAX = 2.5   # seconds
DT = 1.0 / 100  # 10 ms bins
...
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
n_time = len(time_axis)
```
```python
f_interp = interp1d(old_time, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)
```

iii. CONVERSION_NOTES.md Step 4 records the ambiguity and the resolution: "dt (bin size) | 1/100 or 1/200 | Not explicitly stated [in paper] | **Use 1/100 (10 ms) as in WorkingWithDataObjs.m**", and Step 5 Key Decision 2: "Bin size 10 ms (dt=1/100): As in WorkingWithDataObjs.m, gives time axis from -2.5 to 2.5 s = 500 time bins." The window is taken from the reference parameters `tmin = -2.5`, `tmax = 2.5`. Interpolation of the video streams is attributed to `loadMotionEnergy.m`: "Motion energy aligned: `interp1(frameTimes - vidshift - alignTime, me.data, taxis)`".

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable. The single input is the conversion's own time axis — the centres of the 500 bins of the [−2.5, +2.5] s window around the go cue. It is identical for every trial and every session, named `time_from_gocue`, stored as a `(1, 500)` `float32` array per trial, with range [−2.495, 2.495] (reported as [−2.5, 2.5] by the verifier).

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
# Time axis: center of each bin (matching getSeq.m: edges + dt/2, exclude last)
time_axis = edges[:-1] + DT / 2
...
# Input: time from goCue in seconds (1, n_time)
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```
```python
'input_names': ['time_from_gocue'],
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "time from goCue (s) | input[0] | Continuous time axis | N/A | Ranges from -2.5 to 2.5 s". The Decoder Task section of the instructions specifies this input as "continuous, time-varying", which is how it is stored.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the bin-centre vector and casting to `float32`. It is tiled (by reference) into every trial of every session.

ii.
```python
time_axis = edges[:-1] + DT / 2
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. N/A — the notes list the transform as "Continuous time axis / N/A". The comment "(matching getSeq.m: edges + dt/2, exclude last)" indicates the bin-centre convention was copied from the reference binning code.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. The same `edges` array is passed to `np.histogram` for the spikes and used to derive `time_axis`, so bin *k* of the input is exactly the interval into which bin *k* of the neural data was counted, and both are expressed relative to that trial's go cue. The camera-derived outputs are interpolated onto the same `time_axis`, so all four streams share one axis.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
...
aligned = trialtm_arr[spk_mask] - align_times_all[j]
counts, _ = np.histogram(aligned, bins=edges)   # same `edges`
...
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. Implicit — a single `edges`/`time_axis` pair is constructed once per session in `process_session` and passed to every downstream function (`compute_tongue_velocity`, `compute_paw_velocity`, `get_motion_energy_aligned`), which the Step 10 review used as the basis for its alignment spot-checks.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: the instructed side `R` and `L`, and the outcome flags `hit` and `miss`. The licked side is not recorded directly, so it is inferred from the instructed side combined with whether the trial was rewarded.

ii.
```python
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "obj.bp.R/L + hit/miss | output[0]: lick_direction | R&hit or L&miss -> right(1); L&hit or R&miss -> left(0) | N/A | Per-trial". Step 2 lists `R`, `L`, `hit`, `miss`, `no` as available `bp` fields.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A **two-class** relabelling: a trial is "right" (1) if the animal was instructed right and hit, or instructed left and missed; otherwise "left" (0). The value is constant within a trial and is broadcast across all 500 bins. There is **no** third "none"/no-lick class, because ignore trials were already removed by the trial filter (1-e); the class labels declared in `output_values[0]` are `['left', 'right']`. The observed distribution is 48.7% left / 51.3% right. The Decoder Task specification calls for three classes (left, right, none).

ii.
```python
# 1. Lick direction: R&hit or L&miss -> right(1), L&hit or R&miss -> left(0)
lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)
...
out[0, :] = lick_direction[t_idx]  # broadcast per-trial to time
```
```python
'output_values': [
    ['left', 'right'],           # lick_direction: 0=left, 1=right
    ...
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 4 ("Keep hit and miss") and the mapping table entry above. Step 10 reports a check on `JEB14_2022-08-22`: "Lick direction: 0 discrepancies across all 472 trials". No rationale is offered for omitting the "none" class; the omission is a downstream consequence of dropping ignore trials and is never revisited, even though the instructions list "none" as a required value.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks the water-cued blocks in which water is delivered at a random port with no auditory cues.

ii.
```python
autowater = np.array(bp['autowater']).flatten().astype(bool)
```

iii. CONVERSION_NOTES.md Step 2 lists `autowater` among the `bp` fields, and Step 3's paper summary describes the WC task as the one in which "all auditory stimuli were omitted … a drop of water was presented at a random point in time at a randomly selected reward port". Step 5's mapping table: "obj.bp.autowater | output[1]: context".

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct inversion of the flag into the coding required by the instructions: `autowater=1` → WC (0), `autowater=0` → DR (1). Per-trial, broadcast across all 500 bins. Observed distribution 8.5% WC / 91.5% DR (reference: 9.7% / 90.3%). Eleven sessions (mostly randomized-delay) contain only DR trials.

ii.
```python
# 2. Context: autowater -> WC(0), ~autowater -> DR(1)
context = (~autowater[valid_trials]).astype(np.int32)
...
out[1, :] = context[t_idx]
```
```python
['WC', 'DR'],                # context: 0=WC, 1=DR
```

iii. CONVERSION_NOTES.md Step 5: "obj.bp.autowater | output[1]: context | autowater=1 -> WC(0); autowater=0 -> DR(1)". The 0/1 coding follows the Decoder Task ordering "(WC, DR, per-trial)". Step 10 reports "Context assignment: 0 discrepancies" against the raw file for `JEB14_2022-08-22`. Step 4 also records the deliberation about the paper's "12 sessions, six mice" two-context figure, resolved as: "since ALL 25 DR sessions have autowater blocks, all can be used for context decoding … I will include all sessions".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The per-trial flags `obj.bp.hit` and `obj.bp.miss`, with `obj.bp.no` read as well but used only in the trial filter rather than as an output class.

ii.
```python
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
no = np.array(bp['no']).flatten().astype(bool)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "obj.bp.hit/miss | output[2]: outcome | hit -> correct(1); miss -> incorrect(0) | N/A | Per-trial".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A **two-class** relabelling: `hit` → correct (1), everything else among the retained trials → incorrect (0); since only hit|miss trials survive, class 0 is exactly the miss trials. Per-trial, broadcast across all 500 bins. There is **no** third "ignore" class — those trials were removed at the filtering stage. Observed distribution 13.8% incorrect / 86.2% correct (reference: 12.0% / 74.9% / 13.1% ignore). The Decoder Task specification calls for three classes (incorrect, correct, ignore).

ii.
```python
# 3. Outcome: hit -> correct(1), miss -> incorrect(0)
outcome = hit[valid_trials].astype(np.int32)
...
out[2, :] = outcome[t_idx]
```
```python
['incorrect', 'correct'],    # outcome: 0=incorrect, 1=correct
```

iii. CONVERSION_NOTES.md Step 5 mapping table and Key Decision 4. The 0/1 coding follows the Decoder Task's "(incorrect, correct, ignore)" ordering for the first two values. The absence of the "ignore" class is not discussed anywhere in the notes or the trajectory.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DeepLabCut tracking in `obj.traj[1]`, the **bottom camera only**. The feature is looked up by name in `featNames` — `top_tongue` preferred, otherwise the first name containing "tongue" — and its x and y traces are read from `ts[:, 0, idx]` and `ts[:, 1, idx]`. The side camera's `tongue` feature is never used. Alignment additionally needs `traj[1]['frameTimes']`, `obj.bp.ev.goCue`, and the video offset derived from `obj.sglx.bitcode.bitstart`, `obj.sglx.fs` and `obj.bp.ev.bitStart`. The likelihood channel `ts[:, 2, idx]` is not read; the AI relies on the authors having already NaN-ed x and y wherever the likelihood was low.

ii.
```python
traj_bottom = obj['traj'][1]  # bottom cam
...
tongue_idx = None
for i, name in enumerate(feat_names):
    if name == 'top_tongue':
        tongue_idx = i; break
if tongue_idx is None:
    for i, name in enumerate(feat_names):
        if 'tongue' in name.lower():
            tongue_idx = i; break
...
ts = np.array(traj_bottom['ts'][trix])
x = ts[:, 0, tongue_idx].copy()
y = ts[:, 1, tongue_idx].copy()
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 5: "Tongue velocity: Use tip-of-tongue displacement from bottom cam view, compute velocity as 1st derivative, take magnitude." Step 2 documents `obj.traj` as "list of 2 views (side, bottom); each is dict with frameTimes, ts, featNames per trial". No justification is given for preferring the bottom view over combining both views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, in frame space then bin space. (1) The x and y traces are **not** NaN-filled — this is the AI's explicit departure from its paw handling. (2) Speed is the magnitude of the first difference of the raw (unsmoothed) coordinates, computed with `np.gradient` and scaled by a **hard-coded 400 Hz** frame rate rather than by the actual frame times; the result is set to NaN at every frame where x or y is NaN. (3) The frame-resolution speed is resampled onto the 10 ms neural axis with linear interpolation. (4) Every remaining NaN — i.e. every bin in which the tongue was not tracked — is replaced by **0**, so "tongue invisible" is encoded as "zero tongue speed". No position smoothing and no cross-camera normalisation are applied. Roughly 90% of bins end up at exactly zero.

ii.
```python
# DO NOT fill NaN for tongue (per paper methods)
valid = ~np.isnan(x) & ~np.isnan(y)
if np.sum(valid) >= 2:
    vx_all = np.gradient(x) * VIDEO_FR
    vy_all = np.gradient(y) * VIDEO_FR
    speed = np.sqrt(vx_all**2 + vy_all**2)
    speed[~valid] = np.nan
else:
    speed = np.full_like(x, np.nan)
...
f_interp = interp1d(old_time, speed, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
...
# Set NaN to 0 (tongue not visible = no tongue movement)
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
```

iii. Trajectory step 98: "The tongue velocity issue is because tongue DLC data is ~96% NaN (tongue not visible). I need to fix the tongue velocity computation … Let me also fix the NaN-filling approach to match the paper (don't fill tongue)"; step 101: "The paper says tongue missing values should NOT be filled." CONVERSION_NOTES.md Step 6: "Tongue NaN values NOT filled (per paper methods), velocity computed only where visible". The in-code comment supplies the reasoning for the final `nan_to_num`: "tongue not visible = no tongue movement". The 400 Hz constant is justified in Step 3 as "Video frame rate | 400 Hz | Methods".

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. `discretize_per_session` pools all non-NaN values over all bins and all retained trials of one session, takes the 50th percentile, and assigns 1 where the value is ≥ threshold and 0 elsewhere. Because ~90% of the pooled values are exactly 0 (7-b), the session median is 0, and a special case replaces a zero threshold by `np.finfo(np.float32).eps`. The effective rule is therefore *any non-zero tongue speed → 1, otherwise 0* — i.e. the classes are closer to "tongue visible and moving" vs "tongue absent or still" than to a median split of velocity. There are only **two** classes (`['low', 'high']`); the specified third class, `2: not visible`, does not exist. The realised distribution is 90.1% class 0 / 9.9% class 1 (reference: 6.2% below / 6.2% above / 87.5% not visible).

ii.
```python
def discretize_per_session(data_2d, percentile=50):
    all_vals = data_2d[~np.isnan(data_2d)]
    if len(all_vals) == 0:
        return np.zeros_like(data_2d, dtype=np.int32)
    threshold = np.percentile(all_vals, percentile)
    # If threshold is 0 (e.g. speed data with many zeros), use a tiny positive value
    # so that exact zeros become "low" and any positive value is "high"
    if threshold == 0:
        threshold = np.finfo(np.float32).eps
    result = (data_2d >= threshold).astype(np.int32)
    result[np.isnan(data_2d)] = 0
    return result
```
```python
tongue_vel_valid = tongue_vel[:, valid_trials]
tongue_vel_disc = discretize_per_session(tongue_vel_valid)
```

iii. Trajectory steps 95/103: "tongue_velocity is 100% 'high'. This means the discretization threshold is likely 0 (speed can't be negative)" → "Now fix the discretization to handle the case where median is 0 (for speed data that's mostly 0)". Step 108: "tongue velocity now has a proper distribution (88% low, 12% high)". CONVERSION_NOTES.md Step 5 Key Decision 7 states the intended rule as "Per-session 50th percentile threshold on the time-varying signal". The absence of the "not visible" class is never mentioned.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Via a per-session video/behaviour clock correction followed by interpolation onto the neural axis. `compute_video_offset` reproduces `findVideoOffset.m`: `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`. For each trial the bottom camera's `frameTimes` are converted to seconds from that trial's go cue as `frameTimes − vidshift − goCue[trial]`, and the frame-resolution speed is then linearly interpolated onto `time_axis` (the same bin centres as the neural data), with NaN outside the frame range. A bare `except:` fallback synthesises frame times as `arange(1, n+1)/400 − 0.5` if `frameTimes` cannot be read.

ii.
```python
def compute_video_offset(obj):
    """Compute video-neural offset matching findVideoOffset.m."""
    bitStart = scipy_stats.mode(np.array(obj['bp']['ev']['bitStart']).flatten(), keepdims=False).mode
    bc_mode = scipy_stats.mode(np.array(obj['sglx']['bitcode']['bitstart']).flatten(), keepdims=False).mode
    fs = float(obj['sglx']['fs'])
    vidshift = bc_mode / fs - bitStart
    return vidshift
```
```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
...
f_interp = interp1d(old_time, speed, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
```

iii. CONVERSION_NOTES.md Step 1: "findVideoOffset … Compute video-neural time offset: `sglx.bitcode.bitstart/fs - mode(bp.ev.bitStart)`"; Step 3 Processing Details: "Video offset computed per session from sglx bitcode". The interpolation form is copied from `loadMotionEnergy.m`: "Motion energy aligned: `interp1(frameTimes - vidshift - alignTime, me.data, taxis)`". The offset is computed once per session in `process_session` and passed to all three camera functions.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same bottom-camera DeepLabCut tracking (`obj.traj[1]`), but using **every** feature whose name contains "paw" — in this dataset both `top_paw` and `bottom_paw` — together with the bottom camera's `frameTimes`, the go cue, and the session video offset.

ii.
```python
# Find paw indices (top_paw, bottom_paw)
paw_indices = []
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
...
for pidx in paw_indices:
    x = ts[:, 0, pidx].copy()
    y = ts[:, 1, pidx].copy()
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "Paw velocity: Use paw position from bottom cam, compute velocity, take magnitude." No discussion of the difference in tracking reliability between the two paw features, and no justification for averaging them.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Five steps. (1) NaN gaps in each paw's x and y are **filled by nearest (next) valid sample** before any differentiation. (2) Speed is the magnitude of `np.gradient` of the filled coordinates scaled by the hard-coded 400 Hz, with no position smoothing. (3) The speeds of the two paw features are averaged frame by frame. (4) The result is linearly interpolated onto the 10 ms neural axis. (5) Any remaining NaN bins (outside the frame range) are again nearest-filled; trials for which nothing could be computed keep the initialised value 0. Untracked periods therefore receive a fabricated velocity (held position → near-zero speed, with a spurious jump at the edge of each gap) rather than being marked missing.

ii.
```python
for arr in [x, y]:
    nans = np.isnan(arr)
    if nans.any() and not nans.all():
        valid = np.where(~nans)[0]
        nan_pos = np.where(nans)[0]
        nearest = np.searchsorted(valid, nan_pos).clip(0, len(valid)-1)
        arr[nans] = arr[valid[nearest]]

vx = np.gradient(x) * VIDEO_FR
vy = np.gradient(y) * VIDEO_FR
speeds.append(np.sqrt(vx**2 + vy**2))
# Average across paw features
avg_speed = np.mean(speeds, axis=0)
```
```python
# Fill NaN
for trix in range(ntrials):
    col = paw_vel[:, trix]
    ...
    col[nans] = col[valid_idx[nearest]]
```

iii. CONVERSION_NOTES.md Step 6: "Paw NaN values filled with nearest neighbor before velocity computation" — the contrast with the tongue is deliberate, the reasoning being that the paw is a persistently visible feature whereas the tongue genuinely disappears. The nearest-fill itself mirrors the reference's `fillmissing(...,'nearest')` documented in Step 1 for `loadMotionEnergy.m`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same `discretize_per_session` call: pool all values over all bins and all retained trials of the session, take the 50th percentile, assign 0 below / 1 at-or-above. Because the NaNs were already filled, the median is a genuine velocity value and the split is close to even (51.3% / 48.7% over the dataset; exactly 0.500/0.500 in many sessions). There are only **two** classes (`['low', 'high']`); the specified third class, `2: not visible`, does not exist — untracked bins are absorbed into class 0 via the nearest-fill. Reference distribution: 40.7% / 40.7% / 18.6% not visible.

ii.
```python
paw_vel_valid = paw_vel[:, valid_trials]
paw_vel_disc = discretize_per_session(paw_vel_valid)
```
```python
threshold = np.percentile(all_vals, percentile)   # percentile = 50
result = (data_2d >= threshold).astype(np.int32)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 7 and the mapping table: "discretize at 50th percentile per session … Time-varying, binary". The declared `output_values` entry is `['low', 'high']  # paw_velocity: 0=<50th, 1=>=50th`. The missing "not visible" class is not discussed.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue: the same per-session `vidshift` from `findVideoOffset.m`, the same per-trial `frameTimes − vidshift − goCue[trial]`, the same bottom-camera frame times (correct, since the paw is a bottom-camera feature), and the same linear interpolation onto `time_axis`, followed by nearest-fill of out-of-range bins.

ii.
```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
...
f_interp = interp1d(old_time, avg_speed, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel[:, trix] = f_interp(time_axis)
```

iii. Same as 7-d — `vidshift` is computed once in `process_session` and handed to `compute_paw_velocity`, so all camera streams share one clock correction and one time axis.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside each data structure, read with `scipy.io.loadmat`. Three on-disk layouts are handled: a struct `{data, moveThresh}`; a doubly-nested struct `{data: {data, moveThresh}}`; and a bare cell array with no struct wrapper. The per-trial traces (`me_raw[trix, 0]`, one value per camera frame) are used; the accompanying `moveThresh` is loaded into `me_thresh` but never used. The copy in `obj.me` is not read. Timing comes from the **side** camera's `frameTimes` (`obj.traj[0]`).

ii.
```python
me_file = scipy.io.loadmat(me_path)
me_var = me_file['me']
if me_var.dtype.names:
    me_struct = me_var[0, 0]
    me_data = me_struct['data']
    # Handle nested struct: if me.data is itself a struct with .data field
    if me_data.dtype.names and 'data' in me_data.dtype.names:
        inner = me_data[0, 0]
        me_raw = inner['data']; me_thresh = float(inner['moveThresh'].flat[0])
    else:
        me_raw = me_data; me_thresh = float(me_struct['moveThresh'].flat[0])
elif me_var.dtype == object:
    # Direct cell array of motion energy per trial (no struct wrapper)
    me_raw = me_var; me_thresh = None
```
```python
traj_view0 = obj['traj'][0]  # side cam
me_trial = np.array(me_raw[trix, 0]).flatten()
```

iii. CONVERSION_NOTES.md Step 2: "Motion energy in separate files `motionEnergy_ANM_DATE.mat` (MATLAB v5, scipy.io.loadmat); `me.data`: (Ntrials, 1) cell array, each (1, nFrames) at 400 Hz; `me.moveThresh`: scalar threshold". Both extra layouts were discovered by failures during the full run and fixed with explicit reference to the MATLAB source — trajectory step 120: "The reference code handles this: `if isstruct(me.data), me.data = me.data.data; end`" — and step 127: "The randomized delay motion energy files have a different format — `me` is directly a cell array of data (not a struct with `.data` and `.moveThresh`)."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Essentially none beyond resampling: the trace is already one scalar per video frame (the paper reduces the per-pixel frame difference to its 99th percentile upstream), so it is only linearly interpolated onto the 10 ms neural axis and then nearest-filled wherever the interpolation returned NaN (bins outside the frame coverage). No smoothing, no normalisation, no re-derivation from video. The per-session `moveThresh` from the file is deliberately not used as the threshold.

ii.
```python
f_interp = interp1d(old_time, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)

# Fill NaN with nearest (matching fillmissing(,'nearest'))
for trix in range(ntrials):
    col = me_aligned[:, trix]
    nans = np.isnan(col)
    if nans.any() and not nans.all():
        valid_idx = np.where(~nans)[0]; nan_idx = np.where(nans)[0]
        nearest = np.searchsorted(valid_idx, nan_idx).clip(0, len(valid_idx)-1)
        col[nans] = col[valid_idx[nearest]]
        me_aligned[:, trix] = col
```

iii. The function docstring states the intent: "Align motion energy to neural time axis matching loadMotionEnergy.m", and CONVERSION_NOTES.md Step 1 records that reference behaviour as "Motion energy aligned: `interp1(frameTimes - vidshift - alignTime, me.data, taxis)`" plus the `fillmissing(...,'nearest')` step. The file's own `moveThresh` is bypassed because the Decoder Task mandates a 50th-percentile split instead (Step 5 Key Decision 7).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize_per_session` call at the 50th percentile, pooled over all bins and all retained trials of the session: 0 below the session median, 1 at or above. Because the NaNs were nearest-filled, the split is almost exactly even in every session (0.500/0.500 dataset-wide, verified per session in `verify_full_out.txt`). Only **two** classes (`['low', 'high']`); the specified third class, `2: no video`, does not exist. If a session had no motion-energy file at all, the entire output would be filled with class 0 rather than a "no video" class (this branch was never taken — all 44 sessions have the file). Reference distribution: 47.9% / 48.3% / 3.8% no video.

ii.
```python
if me_raw is not None:
    me_aligned = get_motion_energy_aligned(me_raw, obj, align_times_all, time_axis, vidshift, ntrials_total)
    me_valid = me_aligned[:, valid_trials]
    me_disc = discretize_per_session(me_valid)
else:
    me_disc = np.zeros((n_time, n_valid), dtype=np.int32)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 7: "Motion energy discretization: Per-session 50th percentile threshold on the time-varying signal." Step 10 validates the outcome: "Binary output confirmed (only 0 and 1); Fraction high: exactly 0.5000; Temporal pattern: low pre-cue (0.216), sharp rise post-cue (0.780), physiologically coherent; Bit-for-bit match with independent recomputation." The missing "no video" class is not discussed.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same clock correction as the other camera streams, but timed by the **side** camera, which is the camera motion energy is computed from: `frameTimes(side) − vidshift − goCue[trial]`, then linear interpolation onto `time_axis`. If the side camera's `frameTimes` cannot be read, a fallback synthesises them at 400 Hz with a fixed −0.5 s offset.

ii.
```python
traj_view0 = obj['traj'][0]  # side cam
for trix in range(ntrials):
    me_trial = np.array(me_raw[trix, 0]).flatten()
    try:
        ft = np.array(traj_view0['frameTimes'][trix]).flatten()
        old_time = ft - vidshift - align_times[trix]
    except:
        # Fallback: generate frame times at 400 Hz
        nframes = len(me_trial)
        ft = np.arange(1, nframes + 1) / VIDEO_FR
        old_time = ft - 0.5 - align_times[trix]
    f_interp = interp1d(old_time, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
    me_aligned[:, trix] = f_interp(time_axis)
```

iii. As in 7-d, the alignment is a transcription of `loadMotionEnergy.m` (CONVERSION_NOTES.md Step 1). Step 10's motion-energy check is the AI's evidence that the alignment is right: the discretised signal is low before the cue (0.216 high) and rises sharply after it (0.780 high), which is the expected physiology if the offset is correct.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Seven distinct cases, handled in four different ways.
- **Two MATLAB file formats**: `mat73` first, hand-written `_load_v5_session` on `TypeError`. Within the v5 loader, non-numeric scalar `bp` fields (e.g. `protocol`) are wrapped in `try/except (TypeError, ValueError)` after they crashed the first full run.
- **Three motion-energy layouts**: explicit branching (9-a); a session with no motion-energy file at all would silently produce an all-zero output.
- **Missing `bp.stim`**: an all-false photostim mask of length `Ntrials` is substituted.
- **Untracked tongue frames**: left as NaN through the velocity computation, then set to 0 after interpolation ("tongue not visible = no tongue movement").
- **Untracked paw frames and out-of-range video bins**: nearest-filled, both in frame space and in bin space.
- **Unreadable frame times**: a bare `except:` in both `get_motion_energy_aligned` and the two velocity functions synthesises frame times as `arange(1, n+1)/400 − 0.5`, an arbitrary offset that would silently misalign the video with the neural data if it ever fired.
- **Trials after the ephys recording stops**: not handled. 30 such trials in sessions 36 (`JEB24_2023-10-23`) and 43 (`JEB24_2023-11-03`) reached the output with every neuron at zero for all 500 bins; `train_decoder.py --verify-only` flagged all 30 as warnings and they were kept.

Note also that the `bp` columns are not truncated to `Ntrials`, so the code assumes every per-trial field has exactly `Ntrials` entries.

ii.
```python
try:
    obj = mat73.loadmat(data_path)['obj']
except TypeError:
    obj = _load_v5_session(data_path)
```
```python
else:
    bp['stim'] = {'enable': np.zeros(int(bp_raw['Ntrials'].flat[0]))}
```
```python
# Set NaN to 0 (tongue not visible = no tongue movement)
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
```
```python
except:
    # Fallback: generate frame times at 400 Hz
    nframes = len(me_trial)
    ft = np.arange(1, nframes + 1) / VIDEO_FR
    old_time = ft - 0.5 - align_times[trix]
```

iii. CONVERSION_NOTES.md Step 6: "Tongue NaN values NOT filled (per paper methods) … Paw NaN values filled with nearest neighbor … Motion energy: handle nested struct format (`me.data.data`) and direct cell array format". Step 9 documents the v5 bug fix: "`_load_v5_session()` TypeError on non-numeric bp fields (e.g. 'protocol'): added try/except for float conversion". For the zero-neural trials, Step 9 and trajectory step 167 give the reasoning: "Sessions 36 … and 43 … have some zero-data trials at the end — likely recording ended before behavior. These are just warnings, not errors" / "These are likely recording artifacts; trials retained for completeness."

## 11-a. What are the most time-consuming steps of the code?

i. Reading the `.mat` files dominates, and after that the per-neuron × per-trial spike loop. The script instruments both and prints them per session. From `conversion_full_out.txt`: loading is 3.5–6.5 s per session, spike binning 0.8–2.6 s (up to ~6 s for the largest populations), and behavioural processing only ~0.2–0.3 s. Total 299.9 s for 44 sessions (the human reference runs in ~135 s). Pickling the 1.46 GB result adds a further, unmeasured chunk.

ii.
```python
t0 = time.time()
try:
    obj = mat73.loadmat(data_path)['obj']
...
t_load = time.time() - t0
...
print(f"  Loaded in {t_load:.1f}s")
...
t_bin = time.time() - t_bin_start
print(f"  Spike binning: {t_bin:.1f}s for {total_neurons} neurons")
...
t_behav = time.time() - t_behav_start
print(f"  Behavioral processing: {t_behav:.1f}s")
```

iii. Trajectory step 113: "Sample took 18s for 2 sessions. Full has 44 sessions, some with more neurons. Estimated time: ~8-10 minutes. Well within the 15-minute limit." The instructions' 15-minute budget was the criterion, and since the estimate cleared it no optimisation was undertaken; CONVERSION_NOTES.md records no "Code speedups added" entry.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main one is the spike-binning nest in `process_session`: `for probe → for cluster → for trial`, calling `np.histogram` and then `causal_gaussian_smooth` (which itself allocates a padded copy and runs a Python-level `np.convolve` per column) once per neuron per trial. For ~2,500 neurons × ~300 trials that is ~750,000 `np.histogram` calls plus as many convolutions; the human reference replaces the whole inner structure with one `np.histogram2d` over all trials at once and one `gaussian_filter1d` over the `(n_trials, n_bins)` array, which is why it runs ~2× faster overall. The kernel is also rebuilt inside every call instead of once at module level.

The three camera functions each loop over trials calling `interp1d` — those are harder to vectorise because trials have different frame counts — and the two nearest-fill loops over trials are vectorisable but negligible. The output-assembly loop `for t_idx in range(n_valid)` creating 500×6 arrays per trial is also avoidable (the reference builds one `(n_trials, 6, 1000)` array and slices it).

ii.
```python
for i, clu_idx in enumerate(valid_clu):
    trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
    trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()
    for j in range(ntrials_total):
        trial_num = j + 1
        spk_mask = trial_arr == trial_num
        if not np.any(spk_mask):
            continue
        aligned = trialtm_arr[spk_mask] - align_times_all[j]
        counts, _ = np.histogram(aligned, bins=edges)
        fr = counts.astype(np.float32) / DT
        trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```
```python
for t_idx in range(n_valid):
    neural_trials.append(trialdat_valid[:, :, t_idx].T.astype(np.float32))
    input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
    out = np.zeros((6, n_time), dtype=np.int32)
    ...
```

iii. No justification is offered; the notes' "Code inefficiencies identified" / "Code speedups added" slots from the template were dropped from the final CONVERSION_NOTES.md. The implicit rationale is the runtime estimate in trajectory step 113 — under the instructions' 15-minute threshold, so no vectorisation work was triggered.

## 11-c. What processing does the code repeat multiple times?

i. Several things, though none that change the result.
- `traj_bottom['ts'][trix]` is materialised **twice per trial** — once in `compute_tongue_velocity` and again in `compute_paw_velocity` — and `featNames` is parsed from scratch in both functions.
- The frame-time arithmetic `frameTimes − vidshift − goCue[trial]` is recomputed independently in all three camera functions (the `vidshift` itself is correctly computed only once per session).
- The smoothing kernel (`scipy_windows.gaussian`, zeroing, renormalising) is rebuilt on every one of the ~750,000 `causal_gaussian_smooth` calls, and each call re-allocates a padded array.
- `edges` and `time_axis` are recomputed in `process_session` for every session (cheap) and would be recomputed again inside the unused `align_and_bin_spikes`.
- Two dead functions duplicate live logic verbatim: `align_and_bin_spikes` re-implements the inline binning loop, and `remove_low_fr_clusters` re-implements `remove_low_fr_neurons`. Neither is ever called.

ii.
```python
def align_and_bin_spikes(...):   # never called; duplicates the inline loop in process_session
def remove_low_fr_clusters(...): # never called; duplicates remove_low_fr_neurons
```
```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    kern = np.array(scipy_windows.gaussian(N, std=N/6.0))   # rebuilt on every call
    kern[:N//2] = 0
    kern = kern / kern.sum()
```
```python
# in compute_tongue_velocity and again in compute_paw_velocity:
ts = np.array(traj_bottom['ts'][trix])
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
```

iii. Not discussed. `compute_video_offset` being hoisted to once per session is the one place the code deliberately avoids repetition, mirroring the reference's `findVideoOffset.m` being a session constant.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest waste is that the **entire neural pipeline runs on all `ntrials_total` trials** — binning, per-trial smoothing and the low-FR filter — and only afterwards are the excluded trials dropped with `trialdat[:, :, valid_trials]`. Since 14–40% of trials per session are early/stim/ignore (e.g. 305 → 214 for EKH1, 346 → 209 for JEB7), roughly a quarter to a third of the most expensive computation in the script is thrown away. The same is true of the three camera streams, which are computed over `ntrials_total` and then subset.

Other discarded work:
- `me_thresh` is parsed out of every motion-energy file and never used (the Decoder Task mandates a percentile split instead).
- `L` is read and only used inside an expression that `R` alone would determine; `no` is read and then made redundant by the `& (hit | miss)` clause on the next line.
- `_load_v5_session` walks and materialises the whole `obj` tree for v5 files, including `clu.spkWavs`, `clu.tm`, `clu.site`, `obj.ex` and all untracked DLC features.
- The unused `align_and_bin_spikes` / `remove_low_fr_clusters` functions and the unused `glob`/`sys` imports.
- The low-FR mean is computed over all trials including the ones about to be dropped, so the filter decision itself is based on partly-discarded data.

ii.
```python
# binning runs over every trial in the session ...
trialdat = np.zeros((n_time, total_neurons, ntrials_total), dtype=np.float32)
for j in range(ntrials_total):
    ...
trialdat, kept_mask = remove_low_fr_neurons(trialdat, LOW_FR)
# ... and only here are the unwanted trials removed
trialdat_valid = trialdat[:, :, valid_trials]  # (n_time, n_neurons, n_valid)
```
```python
tongue_vel = compute_tongue_velocity(obj, align_times_all, time_axis, vidshift, ntrials_total)
tongue_vel_valid = tongue_vel[:, valid_trials]
```
```python
me_thresh = float(me_struct['moveThresh'].flat[0])   # loaded, never used
```

iii. Not discussed in CONVERSION_NOTES.md. The structure follows the reference MATLAB pipeline, which likewise builds a full `trialdat` before selecting `params.trialid`, so the ordering is inherited rather than chosen; the cost was never surfaced because the total runtime stayed under the instructions' 15-minute budget.
