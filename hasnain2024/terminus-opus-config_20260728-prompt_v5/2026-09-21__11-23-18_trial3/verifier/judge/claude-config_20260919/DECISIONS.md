# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the session list rather than globbing the data folders. Two module-level lists, `EPHYS_SESSIONS` (25 entries) and `RD_SESSIONS` (19 entries), give `(animal, date, [probe numbers])` for every session, transcribed from the authors' `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` files; the two task folders are separate constants. `main()` concatenates the two lists (44 sessions) and calls `process_session(anm, date, probes, ddir)` once per session. Each session is one file `data_structure_<anm>_<date>.mat`, opened by `load_mat_file`, which tries `h5py` (MATLAB v7.3) and falls back to `scipy.io.loadmat` (v5/v7). 34 sessions load through h5py and 10 through scipy. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat` in the same folder. Fields are read lazily/individually (`get_trial_info_*`, `get_clusters_*`, `get_traj_trial_*`) rather than by materialising the whole `obj` tree, and the HDF5 handle is closed at the end of the session.

ii.
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2]),
    ...
    ('JGR3', '2021-11-18', [1]),
]
RD_SESSIONS = [
    ('JEB11', '2022-05-10', [1]),
    ...
    ('JEB24', '2023-11-03', [1]),
]
EPHYS_DATA_DIR = '/app/data/Ephys_Behavior'
RD_DATA_DIR = '/app/data/RandomizedDelay_Ephys_Behavior'
```

```python
def load_mat_file(fpath):
    try:
        f = h5py.File(fpath, 'r')
        return f, 'h5py'
    except:
        data = sio.loadmat(fpath, squeeze_me=False)
        return data, 'scipy'
```

```python
all_sessions = [(a, d, p, EPHYS_DATA_DIR) for a, d, p in EPHYS_SESSIONS] + \
               [(a, d, p, RD_DATA_DIR) for a, d, p in RD_SESSIONS]
for si, (anm, date, probes, ddir) in enumerate(all_sessions):
    result = process_session(anm, date, probes, ddir, args.show_processing)
```

iii. From CONVERSION_NOTES.md Steps 1-2 and Step 9: the loading scripts in `/app/code/DataLoadingScripts/Recording and video/` are treated as the authoritative record of which sessions and which probe entered the paper's analysis. The notes explicitly list the files on disk that were *not* included — `JEB23_2023-10-20` ("commented out in loading script") and `JEB24_2023-10-03`, `JEB24_2023-10-04` ("not in loading scripts, behavior-only") — giving 25 + 19 = 44 sessions, which the notes cross-check against the paper's "25 sessions" (DR) and "19 sessions" (randomized delay). Both MATLAB container formats occur in the shared data, hence the two readers.

## 1-b. How are the data split into subjects?

i. The subject is the animal string in the session tuple (equivalently, the part of the filename before the underscore). Subjects are accumulated in first-appearance order into `subjects`, and `subject_idx` holds the index of each session's animal. The result is 14 subjects across 44 sessions (10 in the fixed-delay folder, 4 in the randomized-delay folder), with session counts per subject ranging from 1 (EKH1, EKH3, JEB6, JGR3) to 8 (JEB24).

ii.
```python
if anm not in subjects:
    subjects.append(anm)
subject_idx.append(subjects.index(anm))
...
subject_idx = np.array(subject_idx)
...
'subjects': subjects, 'subject_idx': subject_idx,
```

iii. The notes record that the animal id is taken from the file/session name. CONVERSION_NOTES.md Step 4 flags the one inconsistency the AI found: the paper reports 9 mice for the DR task and 4 for the randomized-delay task (13 total), while the sessions actually present give 10 + 4 = 14. The AI documented it ("Paper counts 6 two-context + 3 DR-only = 9 ... Data has 10 unique animals") and chose to keep all 14 rather than drop an animal to force the paper's count.

## 1-c. How are the data split into sessions?

i. One `(animal, date)` entry is one session and one `data_structure_*.mat` file, and becomes one element of `neural`, `input`, `output`, `subject_idx` and `brain_region_idx`. The folder is fixed per session by which list the entry lives in, so the fixed-delay and randomized-delay experiments are pooled into a single 44-session dataset rather than treated separately. Sessions with two probes (`JEB15_2022-07-26/27/28`) have both probes' units concatenated into one population. A session is dropped if it has no surviving clusters or fewer than 10 units after the firing-rate filter; neither condition fired, so all 44 sessions are in the output.

ii.
```python
def process_session(anm, date, probe_nums, data_dir, show_processing=False):
    fpath = os.path.join(data_dir, f'data_structure_{anm}_{date}.mat')
```
```python
for probe_num in probe_nums:
    probe_idx = probe_num - 1
    ...
    clusters.append({'quality': q_str, 'trialtm': trialtm, 'trial': trial, 'probe': probe_num})
```
```python
if n_kept < 10:
    print(f"    WARNING: Too few neurons ({n_kept}), skipping")
    return None
```

iii. CONVERSION_NOTES.md Step 5, decision 3: "Both datasets combined: Ephys_Behavior + RandomizedDelay sessions combined into one dataset." The `< 10` units rule is justified in Step 3 from the paper's criterion "at least 10 units", and the Step 9 table reports "Min neurons/session: 17 (≥10 ✓)".

## 1-d. How are the data split into trials?

i. A trial is one entry of the per-trial fields of `obj.bp`. `n_trials = bp.Ntrials` sets the count, and `hit`, `miss`, `R`, `L`, `autowater`, `early`, `stim.enable`, `ev.goCue`, `ev.sample`, `ev.delay` are each flattened to one value per trial. Spikes carry their own 1-based `trial` index, and DLC tracking and motion energy are stored as one cell per trial, so no trial boundaries need to be reconstructed. Every one of the `Ntrials` trials of every session is emitted (no filtering, see 1-e), giving 14,972 trials.

ii.
```python
info = {
    'n_trials': int(bp['Ntrials'][()].flatten()[0]),
    'hit': bp['hit'][()].flatten().astype(bool),
    'miss': bp['miss'][()].flatten().astype(bool),
    'R': bp['R'][()].flatten().astype(bool),
    'L': bp['L'][()].flatten().astype(bool),
    'autowater': bp['autowater'][()].flatten().astype(bool),
    'early': bp['early'][()].flatten().astype(bool),
    'goCue': ev['goCue'][()].flatten(),
    ...
}
```
```python
for ci, clu in enumerate(clusters):
    for trial_num in range(1, n_trials + 1):
        spk_mask = clu['trial'] == trial_num
```

iii. The notes (Step 2) document `obj.bp` as the trial table and `Ntrials` as the trial count. Step 10 "Check 5: Edge cases" records the handling needed for the per-trial containers in the two MATLAB formats (transposed `traj` dimensions under scipy, nested motion-energy structs).

## 1-e. How are trials filtered based on quality controls?

i. **No trial-level filtering is applied at all.** Every trial of every session enters the dataset:
- Early-lick trials (`bp.early`, 976 trials) are kept; instead of being dropped they are *relabelled* — `lick_direction` is forced to `none` and `outcome` to `ignore`, overriding their recorded hit/miss and R/L values. 900 of these 976 trials are flagged hit or miss in the raw data, i.e. the animal did lick, and 472 of them are hits.
- Photostimulation trials (`bp.stim.enable`, 187 trials in 5 sessions) are kept. `stim_enable` is read into `trial_info` but the variable is never used anywhere downstream.
- Trials that run past the end of the ephys recording are kept. 64 such trials (1 in JEB12_2022-05-12, 29 in JEB24_2023-10-23, 34 in JEB24_2023-11-02) come out as 500 bins of exactly zero firing for every neuron, and `train_decoder.py --verify-only` reports each of them as an "all neural data is zero" warning. The warnings were noted in CONVERSION_NOTES.md Step 9/10 but not acted on.

ii. The only filtering-adjacent code is the relabelling:
```python
# Early lick trials: override to 'none' regardless of hit/miss
lick_direction[early] = 2
...
# Early lick trials: override to 'ignore' regardless of hit/miss
outcome[early] = 2
```
`stim_enable` is computed and then unused:
```python
if 'stim' in bp:
    stim = bp['stim']
    info['stim_enable'] = stim['enable'][()].flatten().astype(bool) if 'enable' in stim else np.zeros(...)
```

iii. CONVERSION_NOTES.md Step 3 ("Trial curation: No trials excluded (all trials included, conditions determined by hit/miss/R/L/autowater/early)") and Step 5 decision 2 ("All trials included: No trial filtering by condition - all trials used, with conditions as output labels"). The trajectory (steps 89-90, 104) shows the reasoning: the AI found that `early` overlaps `hit`/`miss`, read the paper's "excluding early lick and ignore trials, which were omitted from all analyses", and concluded that early-lick trials "should be treated as 'ignore' regardless of hit/miss status" rather than removed. For photostim it reasoned "The paper says these should be excluded from analyses. However, for the decoder task, we're including all trials", with no further justification. The zero-neural trials were described in Step 9 as "expected (recording gaps)".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the probe(s) listed for that session — specifically each cluster's `quality` (curation label), `trial` (1-based trial index of each spike) and `trialtm` (spike time relative to the start of its trial). `obj.bp.ev.goCue` supplies the alignment time. `obj.bp.Ntrials` sets the trial axis. Spike waveforms, `clu.tm` (session-clock spike times) and all other `clu` fields are not read.

ii.
```python
clu_ref = clu_dataset[probe_idx, 0]
clu = f[clu_ref]
quality_ds = clu['quality']
for i in range(n_clu):
    q_str = read_h5_string(f, quality_ds[i, 0]).lower()
    ...
    trialtm = f[clu['trialtm'][i, 0]][()].flatten()
    trial   = f[clu['trial'][i, 0]][()].flatten().astype(int)
    clusters.append({'quality': q_str, 'trialtm': trialtm, 'trial': trial, 'probe': probe_num})
```
```python
goCue = trial_info['goCue']
aligned_times = clu['trialtm'][spk_mask] - goCue[trial_num - 1]
```

iii. CONVERSION_NOTES.md Step 1 identifies `alignSpikes.m` ("trialtm_aligned = trialtm - goCue") and `getSeq.m` (binning of `trialtm_aligned`) as the reference functions, and Step 2 documents `obj.clu` as holding "quality, tm, trial, trialtm" per probe.

## 2-b. How is the `neural` data processed?

i. Per cluster and per trial: spike times relative to the go cue are histogrammed into the 500 fixed 10 ms bins, divided by the bin width to give spikes/s, then smoothed along time with a re-implementation of the authors' `mySmooth.m` — a `gausswin(15)` (alpha = 2.5) whose first `floor(15/2) = 7` taps are zeroed to make it **causal**, renormalised to sum to one, applied with `'reflect'` boundary handling (the first 15 samples are prepended before convolution and trimmed afterwards). Nothing else is done: no baseline subtraction, no z-scoring, no normalisation. Stored values are firing rates in Hz as `float32`, transposed to `(n_neurons, n_timepoints)` per trial. Units from both probes of a two-probe session are concatenated.

ii.
```python
N, _ = np.histogram(aligned_times, bins=edges)
fr = N.astype(np.float64) / DT
trialdat[:, ci, trial_num - 1] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE).astype(np.float32)
```
```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    n = np.arange(N); alpha = 2.5; center = (N - 1) / 2.0
    kern = np.exp(-0.5 * (alpha * (n - center) / center) ** 2)
    kern[:int(N // 2)] = 0
    kern = kern / kern.sum()
    if bctype == 'reflect':
        x_padded = np.concatenate([x[:N], x], axis=0); trim = N
    ...
    result[:, j] = np.convolve(x_padded[:, j], kern, mode='same')
    result = result[trim:]
```
```python
SMOOTH_WINDOW = 15
BC_TYPE = 'reflect'
neural_trials.append(trialdat[:, :, t].T.astype(np.float32))
```

iii. CONVERSION_NOTES.md Step 1 lists `getSeq.m` ("Bin spikes (histc), smooth (causal gaussian), create single-trial data") and `mySmooth.m` ("Causal gaussian kernel, reflect BC") with parameters `smooth = 15`, `bctype = 'reflect'` taken from `getDefaultParams.m`. Step 10 Check 3 states "Binning: matches getSeq.m (histc → smooth → FR)" and "Smoothing: matches mySmooth.m (causal gaussian, reflect BC)".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters plus one session-level rule.
1. **Curation label**: the `clu.quality` string is lower-cased and the cluster is dropped if it is in `{garbage, noisy, gabrga, real?}` — exactly the drop list of the authors' `findClusters.m` `'all'` branch. Clusters whose label is empty or contains a NUL byte are also dropped (the reference MATLAB keeps unlabelled clusters in the `'all'` branch). Multi-units, `poor`, `fair`, `good`, etc. are all kept.
2. **Firing rate**: after binning and smoothing, any unit whose mean rate over the whole window and all trials is ≤ 1 Hz is dropped.
3. **Session**: a session with < 10 surviving units would be skipped (never triggered).
This leaves 2,452 units of the 2,506 that pass the quality filter (the 1 Hz cut removes only 54), across 44 sessions, 17-141 per session.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'noisy', 'gabrga', 'real?'}
LOW_FR_THRESHOLD = 1.0  # Hz
...
q_str = read_h5_string(f, quality_ds[i, 0]).lower()
if q_str in EXCLUDED_QUALITIES or q_str == '' or '\x00' in q_str:
    continue
```
```python
mean_fr = trialdat.mean(axis=(0, 2))
keep_mask = mean_fr > LOW_FR_THRESHOLD
trialdat = trialdat[:, keep_mask, :]
```

iii. CONVERSION_NOTES.md Step 1/Step 10 Check 3: "Quality filtering: matches findClusters.m (excludes garbage, noisy, gabrga, real?)" and "FR filtering: matches removeLowFRClusters.m (mean FR > 1 Hz)"; the 1 Hz value is also cross-referenced in Step 3 to the paper's "firing rates exceeding 1 Hz". Step 9 compares the resulting 2,452 units with the paper's 1,651 (DR) + 845 (RD) = 2,496 and calls the ~2% shortfall consistent ("Difference due to FR filtering").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By a single subtraction, done per spike. `clu.trialtm` is already on the behaviour clock and relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]` gives seconds from go-cue onset with no interpolation or further offset. The resulting times are histogrammed into the global `edges` grid, which is identical for every trial, session and data stream, so spikes outside ±2.5 s simply fall outside the bins. The camera streams get an extra clock correction (see 7-d); the neural stream does not.

ii.
```python
aligned_times = clu['trialtm'][spk_mask] - goCue[trial_num - 1]
N, _ = np.histogram(aligned_times, bins=edges)
```
```python
ALIGN_EVENT = 'goCue'
'temporal_alignment_event': 'Go cue onset',
```

iii. CONVERSION_NOTES.md Step 1/Step 10 Check 3: "Spike alignment: matches alignSpikes.m (trialtm - goCue)", with `params.alignEvent = 'goCue'` read from `getDefaultParams.m`. This is also what the Decoder Task section of the instructions requires.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins**, 500 of them, spanning −2.5 s to +2.5 s around the go cue — the same grid for every trial and session, and shared by the neural, input and all three camera output streams. Spikes are counted directly into these bins from raw spike times, so there is no rebinning of an intermediate representation. The camera streams (≈400 Hz) are brought onto this grid by linear interpolation at the bin centres rather than by averaging (see 7-b/9-b), which is a resampling, not a rebinning. `metadata['time_bin_size']` is recorded as 10.0 ms and `off_start`/`off_end` as −2.5/+2.5.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100  # 10ms bins
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
n_time = len(time_axis)
```
```python
'time_bin_size': DT * 1000, 'temporal_alignment_event': 'Go cue onset',
'off_start': TMIN, 'off_end': TMAX,
```

iii. CONVERSION_NOTES.md Step 4 records the discrepancy the AI found in the reference code — "Bin size: 5ms (getDefaultParams) vs 10ms (WorkingWithDataObjs)" — and Step 5 decision 1 resolves it: "Used 10ms (from WorkingWithDataObjs.m tutorial) rather than 5ms (from getDefaultParams.m). Both are used in the codebase; 10ms is more standard for decoder applications." The window −2.5 to 2.5 s is taken from `params.tmin`/`params.tmax`, on which both reference scripts agree.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw data variable; it is the conversion's own time axis. It is defined by the alignment event (`bp.ev.goCue`, which sets the zero of the axis) and the analysis window (`TMIN`, `TMAX`, `DT`): the input is the vector of the 500 bin centres, identical for every trial and every session.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
input_trials.append(time_input.copy())
...
'input_names': ['time_from_go_cue'],
```

iii. CONVERSION_NOTES.md Step 5 maps "time_axis → input[0]: time from go cue, Continuous, [-2.5, 2.5]s". The window matches `params.tmin`/`params.tmax` of the reference code and the Decoder Task instruction to align on the go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the grid: `time_axis` is `edges[:-1] + DT/2`, cast to `float32` and reshaped to `(1, 500)`. A copy is stored for each trial. The values run from −2.495 to +2.495 s, which `train_decoder.py` reports as an input range of [−2.5, 2.5] in every session.

ii.
```python
time_input = time_axis.astype(np.float32).reshape(1, -1)
for t in range(n_trials):
    input_trials.append(time_input.copy())
```

iii. No justification is needed or given beyond the format requirement that inputs be `(d_input, n_timepoints)`; the notes simply record it as a continuous, time-varying input.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `edges` is used both as the histogram edges for spike times (after subtracting the trial's go cue) and as the source of `time_axis`, so bin *k* of the input is the same 10 ms interval as bin *k* of the neural array, for every trial and session. The camera outputs are interpolated onto the same `time_axis`, so all four streams share one axis.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
...
N, _ = np.histogram(aligned_times, bins=edges)        # neural uses edges
...
tv_interp = np.interp(time_axis, ft_side_aligned, tspeed)   # camera uses centres
```

iii. Implicit: a single module-level grid is defined once and reused, so no per-stream alignment step is needed. The notes describe it as "Continuous, [-2.5, 2.5]s" aligned to the go cue.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: `hit`, `miss`, `R` (right-instructed) and `L` (left-instructed), plus `early` which is used as an override. The actual lick side is not stored in the data, so it is inferred from the instructed side combined with whether the animal was correct.

ii.
```python
hit = trial_info['hit']
miss = trial_info['miss']
R = trial_info['R']
L = trial_info['L']
early = trial_info['early']
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "bp.hit, bp.miss, bp.R, bp.L → output[0]: lick_direction, Categorical, 0=left, 1=right, 2=none".

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Start from `none` (2) for all trials; a hit on a right-instructed trial is a right lick (1) and on a left-instructed trial a left lick (0); a miss is the opposite side. Then **every early-lick trial is forced back to `none`**, regardless of its hit/miss/R/L values. The per-trial code is broadcast across all 500 bins. Resulting distribution over the full dataset: left 39.5%, right 41.4%, none 19.2% — the `none` class is inflated from 14.8% to 19.2% by the early-lick override (the AI's own before/after comparison), i.e. ~900 trials in which the animal did lick in a known direction are labelled "no lick".

ii.
```python
lick_direction = np.full(n_trials, 2, dtype=int)  # 2=none
lick_direction[hit & R] = 1  # right
lick_direction[hit & L] = 0  # left
lick_direction[miss & R] = 0  # licked left (wrong on right trial)
lick_direction[miss & L] = 1  # licked right (wrong on left trial)
# Early lick trials: override to 'none' regardless of hit/miss
lick_direction[early] = 2
...
output[0, :] = lick_direction[t]
'output_values': [['left', 'right', 'none'], ...]
```

iii. Trajectory step 89: "the 'early' field overlaps with other fields ... Looking at the paper: 'excluding early lick and ignore trials, which were omitted from all analyses'. So early lick trials should be treated as 'ignore' regardless of hit/miss status." CONVERSION_NOTES.md Step 10 records this as the "Early Lick Fix (Critical Review Finding)", affecting "~611 trials (472 early-hit + 139 early-miss) across Ephys sessions", and states the change "is the correct labeling".

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. A single per-trial flag, `obj.bp.autowater`, which marks trials in the water-cued (WC) block where water is delivered at a random port with no cues.

ii.
```python
'autowater': bp['autowater'][()].flatten().astype(bool),
...
autowater = trial_info['autowater']
```

iii. CONVERSION_NOTES.md Step 5: "bp.autowater → output[1]: context, Binary, 0=DR, 1=WC". Step 2 lists `autowater` among the `obj.bp` per-trial fields.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct cast of the boolean: `autowater → 1 (WC)`, everything else `0 (DR)`, with `output_values[1] = ['DR', 'WC']` recording the mapping. No early-lick or other override is applied here. The code is broadcast across all 500 bins. Over the full dataset DR is 89.7% and WC 10.3%; 12 of the 44 sessions (mostly randomized-delay) have no WC trials at all.

ii.
```python
context = autowater.astype(int)  # 0=DR, 1=WC
...
output[1, :] = context[t]
'output_values': [..., ['DR', 'WC'], ...]
```

iii. No separate justification beyond the Step 5 mapping table; the notes treat `autowater` as the direct marker of the water-cued context. Note that the numeric coding is the reverse of the order the instructions list ("WC, DR"), but `output_values` is written to match, so the stored labels are self-consistent.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial flags of `obj.bp`, `hit` and `miss`, plus `early` as an override. `bp.no` (no-response) is not read; the AI verified in the trajectory that a trial that is neither hit nor miss is a no-response trial by construction.

ii.
```python
hit = trial_info['hit']
miss = trial_info['miss']
early = trial_info['early']
```

iii. CONVERSION_NOTES.md Step 5: "bp.hit, bp.miss, bp.early → output[2]: outcome, Categorical, 0=incorrect, 1=correct, 2=ignore". Trajectory step 104: "The `no` field (no-response trials) ... should be outcome='ignore' ... My code defaults to 2 for both, which is correct since `no` trials are neither hit nor miss."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Default `ignore` (2); `hit → correct` (1); `miss → incorrect` (0); then **every early-lick trial is forced to `ignore`**, overriding its hit/miss flag. Broadcast across all 500 bins. Resulting distribution: incorrect 11.3%, correct 69.6%, ignore 19.2%. The override moved ~900 trials out of correct/incorrect into ignore (the AI's own comparison: correct 0.772 → 0.696, incorrect 0.081 → 0.113, ignore 0.148 → 0.192).

ii.
```python
outcome = np.full(n_trials, 2, dtype=int)  # 2=ignore
outcome[hit] = 1  # correct
outcome[miss] = 0  # incorrect
# Early lick trials: override to 'ignore' regardless of hit/miss
outcome[early] = 2
...
output[2, :] = outcome[t]
'output_values': [..., ['incorrect', 'correct', 'ignore'], ...]
```

iii. Same as 4-b: the AI read the paper's statement that early-lick and ignore trials "were omitted from all analyses" and chose to fold early-lick trials into the `ignore` class rather than remove them, on the grounds that "for decoder training, including these trials with proper labels is correct since we want the decoder to predict the 'ignore' outcome" (trajectory step 104).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, **side camera only** (`traj` view index 0), feature `'tongue'` — its x, y and likelihood traces (`ts`), together with that view's `frameTimes`. The bottom camera's tongue features (`top_tongue`, `bottom_tongue`, …) are not used. `obj.bp.ev.goCue`, `obj.sglx.fs`, `obj.sglx.bitcode.bitstart` and `obj.bp.ev.bitStart` are also needed, to put the camera frames on the go-cue clock.

ii.
```python
ft_side, ts_side, feat_side = get_traj_trial_h5(file_data, obj, t, 0)
...
if 'tongue' in feat_side:
    ti = feat_side.index('tongue')
    tx, ty, tc = ts_side[ti, 0, :], ts_side[ti, 1, :], ts_side[ti, 2, :]
```
```python
def get_traj_trial_h5(f, obj, trial_idx, view_idx):
    """Get DLC data for one trial. Returns frameTimes, ts (n_feat, 3, n_frames), feat_names."""
    view_ref = obj['traj'][view_idx, 0]
    ...
```

iii. CONVERSION_NOTES.md Step 5 decision 4: "Tongue velocity from side cam: Using 'tongue' feature, confidence threshold 0.5 for visibility." Step 2 documents `obj.traj` as "DLC tracking {side_cam, bottom_cam} per trial (ts, frameTimes, featNames)"; `'tongue'` is the side camera's name for the feature in the reference code's `params.traj_features`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, per trial. (1) Speed at frame resolution is the frame-to-frame Euclidean displacement divided by the actual inter-frame interval, with the first frame's value duplicated backwards so the length is preserved; the raw DLC positions are used with **no smoothing** before differentiation. (2) Frames whose likelihood is below 0.5 are set to NaN. Because the authors already store x and y as NaN wherever the likelihood is ≤ 0.9, the effective visibility cut is 0.9, not 0.5, and speeds bracketing an untracked frame become NaN too. (3) The frame-resolution speed is **linearly interpolated** (not averaged) onto the 500 bin centres, and a separately interpolated visibility trace (threshold 0.5) forces the remaining bins to NaN. Since camera frames arrive at ~400 Hz and bins are 10 ms, this samples roughly one frame in four. (4) There is no cross-camera normalisation (only one view is used) and no per-trial normalisation; values stay in pixels/s. 91.7% of all bins end up `not_visible`.

ii.
```python
def compute_velocity(x, y, ft):
    dt = np.diff(ft); dt[dt == 0] = 1e-6
    dx = np.diff(x); dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2) / dt
    return np.concatenate([[speed[0]], speed])
```
```python
tongue_visible = tc >= DLC_CONFIDENCE_THRESHOLD
tspeed = compute_velocity(tx, ty, ft_side)
tspeed[~tongue_visible] = np.nan
tv_interp = np.interp(time_axis, ft_side_aligned, tspeed)
tv_vis = np.interp(time_axis, ft_side_aligned, tongue_visible.astype(float)) >= 0.5
tv_interp[~tv_vis] = np.nan
tongue_vel_all[t] = tv_interp
```

iii. CONVERSION_NOTES.md Step 5 decisions 4 and 6 record the feature choice, the 0.5 confidence threshold for visibility, and the per-session 50th-percentile discretisation. The interpolation-onto-the-neural-axis idiom follows the reference `loadMotionEnergy.m`, which uses `interp1(frameTimes - vidshift - alignTimes, data, taxis)`. No explicit justification is given for differentiating unsmoothed positions or for using a single camera view.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session: the 50th percentile of the tongue speed pooled over all visible bins of all trials in that session. Bins below it are 0 (`low`), bins at or above it are 1 (`high`), and bins with no valid speed are 2 (`not_visible`). Because the split is taken only over the visible bins, low and high come out exactly balanced within the visible ~8% of bins (4.1% / 4.1% of all bins dataset-wide, `not_visible` 91.7%).

ii.
```python
tongue_valid = ~np.isnan(tongue_vel_all)
tongue_thresh = np.nanpercentile(tongue_vel_all[tongue_valid], 50) if tongue_valid.any() else 0
tongue_disc = np.full(tongue_vel_all.shape, 2, dtype=int)
tongue_disc[tongue_valid & (tongue_vel_all < tongue_thresh)] = 0
tongue_disc[tongue_valid & (tongue_vel_all >= tongue_thresh)] = 1
```

iii. Directly from the Decoder Task specification ("discretized with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile, 2: not visible"), recorded as Step 5 decision 6: "Discretization: Per-session 50th percentile threshold for velocity/ME."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock runs ahead of the behaviour clock, so a per-session offset is computed once from the bitcode pulse recorded by both systems — `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` — exactly the authors' `findVideoOffset.m`. Each trial's frame times then become `frameTimes − vidshift − goCue[trial]`, i.e. seconds from the go cue, and the speed trace is interpolated onto the shared `time_axis`, so tongue bin *k* is the same interval as neural bin *k*. Note that `np.interp` clamps rather than extrapolates, so if a trial's frames do not span the full ±2.5 s the edge value is repeated out to the window boundary.

ii.
```python
info['vidshift'] = sp_stats.mode(bitstart_sglx, keepdims=False).mode / fs \
                   - sp_stats.mode(bitStart_bp, keepdims=False).mode
...
ft_side_aligned = ft_side - vidshift - goCue[t]
tv_interp = np.interp(time_axis, ft_side_aligned, tspeed)
```

iii. CONVERSION_NOTES.md Step 1 lists `findVideoOffset` as "vidshift = mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)", and Step 3 notes "Video offset: ~0.5s between neural and video timestamps". The alignment formula mirrors `loadMotionEnergy.m`'s `frameTimes - vidshift - alignTimes(trix)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, **bottom camera** (view index 1), feature `'top_paw'` — its x, y and likelihood, with that view's own `frameTimes`. `bottom_paw` is not used. The same go-cue and bitcode variables are used for alignment.

ii.
```python
ft_bot, ts_bot, feat_bot = get_traj_trial_h5(file_data, obj, t, 1)
...
if 'top_paw' in feat_bot:
    pi = feat_bot.index('top_paw')
    px, py, pc = ts_bot[pi, 0, :], ts_bot[pi, 1, :], ts_bot[pi, 2, :]
```

iii. CONVERSION_NOTES.md Step 5 decision 5: "Paw velocity from bottom cam: Using 'top_paw' feature." `top_paw` is one of the bottom-camera features in the reference code's `params.traj_features`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue: raw (unsmoothed) frame-to-frame displacement divided by the inter-frame interval, NaN wherever the likelihood is below 0.5 (effectively ≤ 0.9, since the stored coordinates are already NaN there), then linear interpolation of both the speed and the visibility trace onto the 500 bin centres. No normalisation is applied, so values are in pixels/s. The paw is tracked in nearly every frame, so only 16.4% of bins are `not_visible` dataset-wide, though this varies a lot by session (0.6% to 87.4%).

ii.
```python
paw_visible = pc >= DLC_CONFIDENCE_THRESHOLD
pspeed = compute_velocity(px, py, ft_bot)
pspeed[~paw_visible] = np.nan
pv_interp = np.interp(time_axis, ft_bot_aligned, pspeed)
pv_vis = np.interp(time_axis, ft_bot_aligned, paw_visible.astype(float)) >= 0.5
pv_interp[~pv_vis] = np.nan
paw_vel_all[t] = pv_interp
```

iii. Same as 7-b — the notes treat tongue and paw as one pipeline with a different feature and camera; no separate justification is recorded for the paw.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session 50th percentile of the paw speed pooled over all visible bins of all trials; below → 0 (`low`), at/above → 1 (`high`), invalid → 2 (`not_visible`). Dataset-wide this gives 41.8% / 41.8% / 16.4%.

ii.
```python
paw_valid = ~np.isnan(paw_vel_all)
paw_thresh = np.nanpercentile(paw_vel_all[paw_valid], 50) if paw_valid.any() else 0
paw_disc = np.full(paw_vel_all.shape, 2, dtype=int)
paw_disc[paw_valid & (paw_vel_all < paw_thresh)] = 0
paw_disc[paw_valid & (paw_vel_all >= paw_thresh)] = 1
```

iii. From the Decoder Task specification, recorded as Step 5 decision 6.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The same session-constant video offset from `findVideoOffset.m` is subtracted from the frame times, then the trial's go cue, and the speed is interpolated onto the shared 10 ms bin centres. The paw uses the **bottom** camera's own `frameTimes` rather than the side camera's, so a trial in which the two views recorded different numbers of frames is still timed correctly.

ii.
```python
ft_bot_aligned = ft_bot - vidshift - goCue[t]
pv_interp = np.interp(time_axis, ft_bot_aligned, pspeed)
```

iii. Same offset and grid as every other stream; no separate justification given.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. A separate file per session, `motionEnergy_<anm>_<date>.mat`, sitting beside the data structure, holding one motion-energy trace per trial at camera frame rate. `obj.me` is not used. The loader handles the three layouts present in the 44 files — a bare cell array, a `{data, moveThresh}` struct, and a struct whose `data` is itself a `{data, moveThresh}` struct — and copes with row- vs column-oriented cell arrays. `moveThresh` is read but never used. Alignment uses the side camera's `frameTimes`.

ii.
```python
def load_motion_energy(anm, date, data_dir):
    me_path = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
    if not os.path.exists(me_path):
        return None, None
    me_raw = sio.loadmat(me_path, squeeze_me=False)['me']
    if me_raw.dtype.names and 'data' in me_raw.dtype.names:
        d = me_s['data']
        if d.shape == (1, 1) and d.dtype.names and 'data' in d.dtype.names:
            d = d[0, 0]['data']
        ...
    elif me_raw.dtype == object:     # cell array directly, no struct wrapper
        ...
```

iii. CONVERSION_NOTES.md Step 1 lists `loadMotionEnergy.m` ("Load ME, align to neural time, interp1"); Step 6 notes "Motion energy loading handles struct, nested struct, and cell array formats"; the trajectory (step 60) documents discovering the three layouts session by session (e.g. "JEB23_2023-10-10 through 10-13: ME files are cell arrays (no struct wrapper, no moveThresh)"). The reference `loadMotionEnergy.m` has the same unwrapping guard (`if isstruct(me.data), me.data = me.data.data; end`).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling: the stored trace is already one number per camera frame, so it is simply linearly interpolated onto the 500 bin centres (with the side camera's aligned frame times as the source axis). If the trace length does not match the side camera's frame count, a fallback builds evenly spaced frame times spanning the side camera's first and last frame and interpolates against that. Trials with no motion-energy entry stay NaN and become the `no_video` class; dataset-wide that is ~0.03%, so the class is effectively empty (`np.interp` clamps at the edges rather than producing NaN).

ii.
```python
if len(me_trial) == len(ft_side):
    me_interp = np.interp(time_axis, ft_side_aligned, me_trial)
elif len(me_trial) > 0:
    me_ft = np.linspace(ft_side[0], ft_side[-1], len(me_trial))
    me_ft_aligned = me_ft - vidshift - goCue[t]
    me_interp = np.interp(time_axis, me_ft_aligned, me_trial)
else:
    me_interp = np.full(n_time, np.nan)
me_all[t] = me_interp
```

iii. This mirrors the reference `loadMotionEnergy.m`, which does `interp1(obj.traj{1}(trix).frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)` with a `catch` fallback that synthesises frame times at 400 Hz, and then `fillmissing(me.data,'nearest')` (the effect of `np.interp`'s edge clamping). CONVERSION_NOTES.md Step 1 records this function as the model.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile of the interpolated motion energy pooled over all trials and bins of that session; below → 0 (`low`), at/above → 1 (`high`), NaN → 2 (`no_video`). Because motion energy is defined in essentially every bin, the result is an almost exactly 50/50 split in every session (`{low 0.500, high 0.500, no_video 0.000}` dataset-wide). The `moveThresh` stored in the motion-energy files — the authors' own movement threshold — is deliberately not used.

ii.
```python
me_valid = ~np.isnan(me_all)
me_thresh_disc = np.nanpercentile(me_all[me_valid], 50) if me_valid.any() else 0
me_disc = np.full(me_all.shape, 2, dtype=int)
me_disc[me_valid & (me_all < me_thresh_disc)] = 0
me_disc[me_valid & (me_all >= me_thresh_disc)] = 1
...
'output_values': [..., ['low', 'high', 'no_video']],
```

iii. The Decoder Task specification mandates the per-session 50th-percentile split and the trailing `no_video` class, which overrides the reference code's `moveThresh`-based binarisation. Recorded as Step 5 decision 6.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy has one value per side-camera frame, so it is timed by the side camera: `frameTimes − vidshift − goCue[trial]`, with the same session-constant `vidshift` from `findVideoOffset.m` used for the tongue and paw, and then interpolated onto the shared bin centres. This is the reference `loadMotionEnergy.m` alignment verbatim.

ii.
```python
ft_side_aligned = ft_side - vidshift - goCue[t]
...
me_interp = np.interp(time_axis, ft_side_aligned, me_trial)
```

iii. CONVERSION_NOTES.md Step 1: "loadMotionEnergy | LOADING | Load ME, align to neural time, interp1" and "findVideoOffset | PROCESSING | Computes video-neural timing offset".

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data is handled by leaving the affected stream as NaN and letting it fall into the trailing `not_visible` / `no_video` class, never by interpolating a value across a gap or dropping the trial:
- **Untracked camera frames** (likelihood below threshold; coordinates already NaN in the file) → NaN speed → `not_visible`.
- **Any per-trial camera failure** — missing `frameTimes`, a feature absent from `featNames`, a shape mismatch, a raised exception of any kind — is swallowed by a bare `except: pass` around the whole per-trial block, leaving that trial's tongue, paw and motion energy as all-NaN.
- **Missing or unparseable motion-energy file** → `load_motion_energy` returns `(None, None)` after printing a warning, and the session's motion energy is all `no_video`.
- **Format variability** is handled explicitly: v7.3 vs v5 MATLAB containers, transposed `ts` axes under scipy, row- vs column-oriented `featNames` and cell arrays, three motion-energy layouts, missing `bp.stim` field.
- **Division by zero** in the velocity is guarded (`dt[dt == 0] = 1e-6`).
What is *not* handled: trials recorded after the probe stopped. 64 such trials are kept with 500 bins of exactly zero firing across every neuron, and `train_decoder.py` reports each one as a warning. `warnings.filterwarnings('ignore')` is set globally at import, so NaN-slice and similar numpy warnings are suppressed.

ii.
```python
warnings.filterwarnings('ignore')
```
```python
for t in range(n_trials):
    try:
        ...  # all DLC and motion-energy processing for this trial
    except Exception as e:
        pass  # Leave as NaN
```
```python
    except Exception as e:
        print(f"    WARNING: Failed to load ME for {anm}_{date}: {e}")
        return None, None
```
```python
dt = np.diff(ft)
dt[dt == 0] = 1e-6  # avoid division by zero
```

iii. CONVERSION_NOTES.md Step 10 Check 5 ("Handled both h5py and scipy formats; Handled nested ME struct formats; Handled transposed traj dimensions in scipy") and Step 9, which lists the zero-neural trials under "Warnings" and Step 10 Check 1, which declares them "expected (recording gaps)" rather than fixing them. The trajectory shows the format edge cases were found empirically after sessions silently produced 0% tongue/paw/ME coverage (steps 55, 60).

## 11-a. What are the most time-consuming steps of the code?

i. The script instruments itself at session granularity — `t0 = time.time()` per session plus a total — and prints "Processed in X.Xs" for each. The full conversion took 147.6 s for 44 sessions (2-6 s each), against a Step 7 estimate of ~148 s extrapolated from a 7.2 s two-session sample. No finer profiling was done, so the notes do not attribute time to particular steps. From the code, the dominant costs are (a) the nested cluster × trial loop that runs one `np.histogram` and one full kernel construction + convolution per cluster per trial — of order 800,000 iterations across the dataset — and (b) the per-trial HDF5/scipy reads of both camera views inside the trial loop. Pickling the 2.08 GB output is the other large fixed cost.

ii.
```python
def process_session(anm, date, probe_nums, data_dir, show_processing=False):
    t0 = time.time()
    ...
    elapsed = time.time() - t0
    print(f"    Processed in {elapsed:.1f}s")
```
```python
    print(f"Total time: {time.time()-t_start:.1f}s")
```

iii. CONVERSION_NOTES.md Step 7: "Run Time: 7.2s for 2 sessions → ~148s estimated for 44 sessions (actual: 148s)". Since the estimate was well under the instructions' 15-minute threshold, no optimisation pass was undertaken and no bottleneck analysis was recorded.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops are left unvectorised.
1. The spike-binning double loop over clusters × trials. This is the big one: it can be replaced by a single `np.histogram2d` over (trial, aligned time) per cluster, or a `bincount` over the whole session, and the smoothing can be applied to the whole (time × trials) matrix in one call instead of column by column.
2. `causal_gaussian_smooth` convolves column by column in a Python loop (`for j in range(x_padded.shape[1])`) and rebuilds the Gaussian kernel on every call — it is called once per (cluster, trial) pair with a single column.
3. The per-trial camera loop. This one is intrinsically hard to vectorise because each trial has a different number of frames, so it is a reasonable loop to keep — though the two `get_traj_trial_*` calls inside it re-resolve the view reference and re-read `featNames` on every iteration.
The trial-level discretisation, in contrast, *is* vectorised over the whole `(n_trials, n_time)` array.

ii. The loop that most needed vectorising:
```python
for ci, clu in enumerate(clusters):
    for trial_num in range(1, n_trials + 1):
        spk_mask = clu['trial'] == trial_num
        if not np.any(spk_mask):
            continue
        aligned_times = clu['trialtm'][spk_mask] - goCue[trial_num - 1]
        N, _ = np.histogram(aligned_times, bins=edges)
        fr = N.astype(np.float64) / DT
        trialdat[:, ci, trial_num - 1] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE).astype(np.float32)
```
Already vectorised:
```python
paw_disc = np.full(paw_vel_all.shape, 2, dtype=int)
paw_disc[paw_valid & (paw_vel_all < paw_thresh)] = 0
paw_disc[paw_valid & (paw_vel_all >= paw_thresh)] = 1
```

iii. The notes give no analysis of which loops could be vectorised; the efficiency section of Step 7 stops at the runtime estimate, and since 148 s was comfortably inside budget no vectorisation work was done. The instructions did ask for vectorised loops independently of the time budget.

## 11-c. What processing does the code repeat multiple times?

i. Several cheap things are recomputed rather than hoisted:
- The Gaussian kernel is rebuilt inside `causal_gaussian_smooth` on every call — once per (cluster, trial), i.e. hundreds of thousands of times per run, for a value that depends only on the constant `SMOOTH_WINDOW`.
- `get_traj_trial_h5`/`_scipy` re-resolve `obj['traj'][view_idx]` and re-read and re-decode the whole `featNames` list for **both** camera views on **every trial**, although the feature names are fixed for the session.
- `feat_side.index('tongue')` / `feat_bot.index('top_paw')` are re-computed per trial for the same reason.
- `clu['trial'] == trial_num` rescans the cluster's whole spike vector once per trial, so each cluster's spike train is scanned `n_trials` times.
- Smoothing is run on trials that are later known to be entirely empty (the post-recording trials), and on the 54 clusters that the firing-rate filter discards immediately afterwards.
What is correctly computed once: the video offset (`vidshift`, once per session in `get_trial_info_*`), the bin grid (module level), and each session's percentile thresholds.

ii.
```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    n = np.arange(N); alpha = 2.5; center = (N - 1) / 2.0
    kern = np.exp(-0.5 * (alpha * (n - center) / center) ** 2)   # rebuilt every call
```
```python
def get_traj_trial_h5(f, obj, trial_idx, view_idx):
    view_ref = obj['traj'][view_idx, 0]
    view_data = f[view_ref]
    ...
    feat_names = [read_h5_string(f, feat_data[0, i]) for i in range(feat_data.shape[1])]
```
Computed once per session (good):
```python
info['vidshift'] = sp_stats.mode(bitstart_sglx, keepdims=False).mode / fs \
                   - sp_stats.mode(bitStart_bp, keepdims=False).mode
```

iii. Not discussed in CONVERSION_NOTES.md. The repetitions are consequences of the structure chosen (helpers that take a trial index and re-open the container) rather than deliberate decisions, and none of them changes the result.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A modest amount, all cheap:
- **Variables read and never used**: `bp.ev.sample`, `bp.ev.delay`, `bp.L` (redundant with `R` — the data has no trial where both or neither is set), `bp.stim.enable` (read, documented as photostim, then ignored), the motion-energy `moveThresh` (returned by `load_motion_energy` and dropped), and each cluster's `quality` and `probe`, kept in the cluster dict after filtering.
- **Work later thrown away**: binning and smoothing are done for all clusters *before* the firing-rate filter, so the 54 of 2,506 quality-passing clusters that fall below 1 Hz are fully processed and then discarded; the same applies to the 64 post-recording trials, which are smoothed and stored as 500 bins of zeros.
- **Discretisation of both cameras' outputs for trials whose camera data failed**, which produces a full 500-bin array of class 2.
- The `--show-processing` flag is parsed and passed into `process_session` as `show_processing`, but the parameter is never referenced — no `processing_<session_id>.png` plots are ever produced, despite Steps 6 and 7 of the instructions requiring them. This is unused plumbing rather than wasted computation, but it is a required output that the script does not generate.

ii.
```python
'sample': ev['sample'][()].flatten(),
'delay': ev['delay'][()].flatten(),
```
```python
info['stim_enable'] = stim['enable'][()].flatten().astype(bool) if 'enable' in stim else ...
```
```python
me_trials, me_thresh = load_motion_energy(anm, date, data_dir)   # me_thresh never used
```
```python
def process_session(anm, date, probe_nums, data_dir, show_processing=False):
    ...   # show_processing is never referenced again
```

iii. Not discussed in CONVERSION_NOTES.md. The extra `bp` fields were read during exploration (Step 2 lists them among the available variables) and left in the loader; the notes never revisit them. The absence of the `--show-processing` plots is not acknowledged anywhere in the notes, which mark Steps 6 and 7 as COMPLETE.
