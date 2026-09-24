# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI restricts itself to a single data folder, `/app/data/Ephys_Behavior` (the fixed-delay task), and globs `data_structure_*.mat` in it. The sibling folder `/app/data/RandomizedDelay_Ephys_Behavior` (22 `data_structure` files from JEB11, JEB12, JEB23, JEB24) is never opened, nor are the two behavior-only inhibition folders. A hard-coded `PROBE_MAP`, transcribed from the authors' `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` scripts, maps each `(animal, date)` to the 0-indexed ALM probe(s); files absent from the map are skipped (in practice all 25 files are in the map, so nothing is skipped). Each session file is opened with `h5py` only (all Ephys_Behavior files are MATLAB v7.3); there is no `scipy.io` fallback for v5 data structures. The matching `motionEnergy_<anm>_<date>.mat` beside it is loaded separately with `scipy.io.loadmat`. The result is 25 sessions, 10 subjects, 7,426 trials, 1,532 units.

ii.
```python
DATA_DIR = Path('/app/data/Ephys_Behavior')
...
PROBE_MAP = {
    ('EKH1', '2021-08-07'): [1],
    ...
    ('JGR3', '2021-11-18'): [0],
}
```
```python
data_files = sorted(DATA_DIR.glob('data_structure_*.mat'))
for data_file in data_files:
    basename = data_file.stem.replace('data_structure_', '')
    parts = basename.split('_', 1)
    anm, date = parts[0], parts[1]
    key = (anm, date)
    if key not in PROBE_MAP:
        print(f"  Skipping {basename}: no probe mapping")
        continue
    me_file = data_file.parent / f'motionEnergy_{anm}_{date}.mat'
    me_path = str(me_file) if me_file.exists() else None
    result = load_session(str(data_file), PROBE_MAP[key])
```
```python
with h5py.File(data_path, 'r') as f:
    obj = f['obj']
```

iii. From the trajectory (step 25/26): *"The paper describes 25 sessions from 9 mice for the two-context plus DR-only paradigm, and 19 sessions from 4 mice for randomized delay, matching the 25 .mat files in Ephys_Behavior. … I'll stick with only the Ephys_Behavior sessions since the paper treats randomized delay as a distinct experiment, and I'll restrict probes to ALM as the loading scripts and paper's focus dictate."* The AI was explicitly aware that JEB11/JEB12/JEB23/JEB24 exist in the RandomizedDelay folder and have their own `load<ANM>_ALMVideo.m` scripts, and chose to exclude them. The probe restriction is justified as *"the loading scripts define which probes are ALM"*, and the AI noted `obj.meta.probe.loc` confirms e.g. `['R_brainstem', 'L ALM']` for EKH1.

## 1-b. How are the data split into subjects?

i. One subject per animal id. The id is read from inside the file, `obj.meta.anm`, decoded from its uint16 character array; if that read raises, the animal is taken from the filename prefix instead. Each session contributes one entry to `session_animals`; at assembly `subjects` is the sorted unique set and `subject_idx` indexes into it. This gives 10 subjects (EKH1, EKH3, JEB6, JEB7, JEB13, JEB14, JEB15, JEB19, JGR2, JGR3) over 25 sessions.

ii.
```python
with h5py.File(data_path, 'r') as f:
    try:
        anm = h5_read_string(f, f['obj']['meta']['anm'])
    except:
        basename = os.path.basename(data_path)
        parts = basename.replace('data_structure_', '').replace('.mat', '').split('_')
        anm = parts[0]
```
```python
unique_subjects = sorted(set(session_animals))
subject_idx = np.array([unique_subjects.index(a) for a in session_animals])
```

iii. The AI's data-structure exploration (step 19) recorded that `obj/meta` contains *"`anm`: `'EKH1'` (animal name)"* stored as uint16 ASCII codes, so it read that field directly and kept the filename as a fallback. There is no explicit further discussion of subject splitting; the animal id is treated as self-evidently the subject.

## 1-c. How are the data split into sessions?

i. One session is one `data_structure_<anm>_<date>.mat` file, i.e. one animal-day. The session boundary is therefore the file boundary, and each file becomes exactly one element of `neural`, `input`, `output`, `subject_idx`, and `brain_region_idx`. Sessions are ordered by the sorted filename. Where a session used two probes (the three JEB15 sessions of 2022-07-26/27/28), both probes' clusters are concatenated into one population rather than split into two sessions. Sessions would be dropped if they yielded fewer than 2 valid trials or fewer than 10 units, but neither guard ever fires on these 25 files.

ii.
```python
for prb_idx in alm_probes:
    prb_ref = clu_group[prb_idx, 0]
    prb_data = f[prb_ref]
    n_clusters = prb_data['quality'].shape[0]
    for clu_i in range(n_clusters):
        ...
        all_spike_times_aligned.append(aligned_times)
```
```python
if len(valid_trials) < 2:
    print(f"    Skipping: only {len(valid_trials)} valid trials")
    return None
...
if n_units < 10:
    print(f"    Skipping: only {n_units} units (need >= 10)")
    return None
```

iii. Session identity follows the authors' own convention of one data object per animal-day, which the AI read out of `WorkingWithDataObjs.m` and the `load<ANM>_ALMVideo.m` meta scripts. The two-probe concatenation follows the same scripts, which list `[1 2]` for those JEB15 dates. The minimum-unit and minimum-trial guards are not justified anywhere in the trajectory; they appear to be defensive coding for the decoder's "at least two trials per session" requirement.

## 1-d. How are the data split into trials?

i. A trial is one index into the per-trial vectors of `obj.bp`. `Ntrials` gives the count; `hit`, `miss`, `no`, `R`, `L`, `autowater`, `early`, `stim.enable` and `ev.goCue` are all read as `[0, :]` rows of shape `(1, Ntrials)`. Spikes carry their own trial number in `clu.trial` (1-based), so trials are selected by mask rather than by reconstructing boundaries; video trajectories and motion energy are stored per trial as cell arrays indexed by the same trial number. Note that, unlike the reference, the AI does **not** truncate the `bp` fields to `Ntrials` — it assumes they are exactly `Ntrials` long. I verified that this holds for all 25 Ephys_Behavior files, so it has no effect here.

ii.
```python
bp = obj['bp']
ntrials = int(bp['Ntrials'][0, 0])
hit = bp['hit'][0, :].astype(bool)
miss = bp['miss'][0, :].astype(bool)
no = bp['no'][0, :].astype(bool)
R = bp['R'][0, :].astype(bool)
L = bp['L'][0, :].astype(bool)
autowater = bp['autowater'][0, :].astype(bool)
early = bp['early'][0, :].astype(bool)
stim_enable = bp['stim']['enable'][0, :].astype(bool)
goCue = ev['goCue'][0, :]
```
```python
for tr in range(ntrials):
    tr_mask = strials == (tr + 1)  # 1-indexed trials
```

iii. The AI's exploration of the data object (step 19) established that `obj.bp` holds *"Per-trial behavioral parameters (305 trials typical)"* and that `obj.trials` is simply `[1, 2, ..., Ntrials]`, so the trial structure is read straight out of the Bpod table. The 1-based to 0-based conversion for `clu.trial` is noted explicitly in the code comments.

## 1-e. How are trials filtered based on quality controls?

i. Three conditions, combined into one mask before anything else is computed: the trial must be one of `hit`, `miss`, or `no` (i.e. a completed trial type); it must not be a photostimulation trial (`~stim.enable`); and it must not be an early-lick trial (`~early`). No other trial-level curation is applied — in particular there is no check that a trial falls inside the span of the ephys recording, and trials with missing or failed video are kept (their kinematic outputs simply become the "not visible" class). This keeps 7,426 of 8,260 trials (89.9%). I verified that this is numerically identical to the reference's `~early & ~stim` filter on these 25 sessions (`hit|miss|no` is true for every trial in all 25 files), and that no trial in the AI's output has an all-zero neural matrix, so the reference's extra recording-length cut would have removed nothing here.

ii.
```python
# ── Trial filtering ──
# Exclude stim trials and early-lick trials
# Include hit, miss, no (ignore) trials
valid_mask = (hit | miss | no) & ~stim_enable & ~early
valid_trials = np.where(valid_mask)[0]  # 0-indexed trial indices
```

iii. The AI derived this from the authors' condition strings in `WorkingWithDataObjs.m`, which all carry `&~stim.enable&~early`, and from the tutorial's condition 1, `'(hit|miss|no)'`, which the tutorial labels "all trials". From the trajectory: *"For trial filtering, I'm excluding stim and early trials to match the reference conditions"* and *"working through trial filtering logic, deciding which trials the decoder should use — excluding stim and early-lick trials while considering whether to include hit, miss, or ignore trials"*. Ignore trials are retained because the decoder task specifies an `ignore` outcome class and a `none` lick-direction class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the ALM probe(s) named in `PROBE_MAP`. Per cluster the AI reads three fields: `quality` (the manual curation string, used for filtering), `trialtm` (spike time relative to that trial's start, on the behaviour clock) and `trial` (1-based trial number of each spike). `obj.bp.ev.goCue` is the fourth input, since it defines the alignment. Spike waveforms (`spkWavs`) and the session-clock spike times (`tm`) are not read.

ii.
```python
prb_ref = clu_group[prb_idx, 0]
prb_data = f[prb_ref]
n_clusters = prb_data['quality'].shape[0]
for clu_i in range(n_clusters):
    quality_ref = prb_data['quality'][clu_i, 0]
    ...
    trialtm_ref = prb_data['trialtm'][clu_i, 0]
    trial_ref = prb_data['trial'][clu_i, 0]
    trialtm = f[trialtm_ref][()].flatten()
    trial_nums = f[trial_ref][()].flatten().astype(int)
```

iii. The AI's summary of `alignSpikes.m` records that the authors do `obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event` with `params.alignEvent = 'goCue'`, so `trialtm` + `trial` + `goCue` is exactly the trio the reference pipeline uses. The `quality` field was identified from `findClusters.m`.

## 2-b. How is the `neural` data processed?

i. Per unit and per trial: go-cue-relative spike times are histogrammed into 500 non-overlapping 10 ms bins spanning −2.5 to +2.5 s, divided by the bin width to give spikes/s, then smoothed with a **causal** Gaussian kernel — a 15-sample `gausswin` (alpha 2.5) with its first half zeroed and renormalised to sum to 1 — under a 'reflect' boundary condition that prepends the first 15 samples before convolving and trims them afterwards. No normalisation, baseline subtraction or z-scoring. Values are stored as `float32` firing rates in Hz. Two-probe sessions have both probes' units concatenated into a single population.

ii.
```python
def causal_gaussian_kernel(N):
    alpha = 2.5
    n = np.arange(N)
    center = (N - 1) / 2.0
    sigma = center / alpha
    kern = np.exp(-0.5 * ((n - center) / sigma) ** 2)
    kern[:int(np.floor(len(kern) / 2))] = 0   # causal
    kern = kern / kern.sum()
    return kern

def smooth_data(x, N, bctype='reflect'):
    kern = causal_gaussian_kernel(N)
    if bctype == 'reflect':
        x_filt = np.concatenate([x[:N], x])
        trim = N
    ...
    out = np.convolve(x_filt, kern, mode='same')
    return out[trim:]

def bin_and_smooth_spikes(spike_times, edges, dt, smooth_win, bctype):
    counts, _ = np.histogram(spike_times, bins=edges)
    fr = counts.astype(float) / dt
    return smooth_data(fr, smooth_win, bctype)
```
```python
trialdat = np.zeros((n_units, ntrials, n_timebins))
for u_idx in range(n_units):
    for tr in range(ntrials):
        ...
        trialdat[u_idx, tr, :] = bin_and_smooth_spikes(tr_spikes, edges, DT, SMOOTH_WIN, SMOOTH_BC)
```

iii. The AI read `mySmooth.m` in full and reimplemented it line for line; its reasoning (step 19) says the kernel *"builds a Gaussian window, zeroes out the first half so it only uses past/present samples, normalizes it to sum to 1"*. The parameters come from the authors' own tutorial `WorkingWithDataObjs.m`: `params.smooth = 15`, `params.bctype = 'reflect'` (the AI noted that `getDefaultParams.m` instead has `'none'`, and chose the tutorial's value). `getSeq.m` divides counts by `params.dt` to get spikes/s, which the AI mirrors.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First the manual curation label `clu.quality`, stripped and lower-cased, is dropped if it is in `{garbage, gabrga, noisy, real?}`; clusters with an **empty** quality string are also dropped. Everything else is kept, including `multi`, `fair` and `poor`. Second, after binning and smoothing, any unit whose mean firing rate — averaged over all `Ntrials` (not just the kept trials) and all 500 bins — is at or below 1.0 Hz is removed. This yields 1,532 units across 25 sessions (27 to 141 per session). I checked the label distribution across 8 sessions: `garbage` 5250, `multi` 380, `fair` 192, `poor` 168, `good` 80, `great` 58, `excellent` 18, empty 3, `ood` 1 — so the extra exclusion of unlabelled clusters costs a handful of units, while retaining `poor` (which the human reference drops) adds a few percent.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
quality = quality.strip()
if quality.lower() in {q.lower() for q in EXCLUDE_QUALITIES}:
    continue
if quality == '':
    continue
```
```python
LOW_FR = 1.0
...
mean_fr = np.mean(np.mean(trialdat, axis=2), axis=1)
fr_mask = mean_fr > LOW_FR
trialdat = trialdat[fr_mask]
```

iii. From the AI's final summary: *"Cluster quality: Exclude garbage/gabrga/noisy/real? (matching `findClusters.m` with `quality='all'`)"* and *"FR threshold: >1 Hz mean firing rate (matching paper and `WorkingWithDataObjs.m`)"*. The AI noted the conflict between `getDefaultParams.m` (`params.lowFR = 0.5`) and the tutorial/paper (1 Hz) and resolved it in favour of 1 Hz: *"the paper specifies 1 Hz as the threshold for including units, so I'll use that despite the different default in getDefaultParams.m."* The tutorial's comment "remove clusters with firing rates across all trials less than this val" is why the mean is taken over all trials rather than the kept subset. The exclusion of empty-label clusters is not justified anywhere.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to go cue onset is a single subtraction per spike: `trialtm[i] - goCue[trial[i] - 1]`. `trialtm` is already relative to its own trial's start and on the same behaviour clock as `bp.ev.goCue`, so no clock correction or interpolation is needed. Spikes whose trial number falls outside `[1, Ntrials]` are set to NaN and dropped before histogramming. The subtraction is done in a pure-Python loop over individual spikes.

ii.
```python
aligned_times = np.empty_like(trialtm)
for t_idx in range(len(trialtm)):
    tr = trial_nums[t_idx] - 1  # 0-indexed
    if 0 <= tr < ntrials:
        aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
    else:
        aligned_times[t_idx] = np.nan
```
```python
tr_spikes = stimes[tr_mask]
tr_spikes = tr_spikes[~np.isnan(tr_spikes)]
```

iii. The AI's summary of `alignSpikes.m` is explicit: *"`obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event`; Subtracts alignment event time from spike times"*, with `goCue` the default alignment event in both `getDefaultParams.m` and the tutorial. The decoder task also mandates go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms** bins (`DT = 1/100`), 500 bins spanning −2.5 to +2.5 s from the go cue, identical for every trial and session. Bin edges are built once per session with `np.arange(TMIN, TMAX + DT, DT)` and the reported time axis is the bin centres. Spikes are histogrammed directly at this resolution — there is no finer initial binning and no subsequent rebinning. The same 500-bin grid is the target for the video interpolation, so all four streams (neural, time input, kinematics, motion energy) share one axis. `metadata['time_bin_size']` is set to 10.0 ms.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 100         # 10 ms bins
...
edges = np.arange(TMIN, TMAX + DT, DT)
n_timebins = len(edges) - 1
time_vec = edges[:-1] + DT / 2
```

iii. The AI found and flagged a direct conflict between the two authoritative sources: *"there's a discrepancy between the two reference files on time bin size (10ms vs 5ms) — I'll go with the 10ms binning used in the tutorial code."* `getDefaultParams.m` sets `params.dt = 1/200` (5 ms), while `WorkingWithDataObjs.m` sets `params.dt = 1/100` (10 ms) under a comment that confusingly reads "use a 5 ms bin width". The AI chose the tutorial's executable value over its comment and over the defaults file. The window −2.5 to 2.5 s is uncontested between the two.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing from the raw data as such — it is the conversion's own time axis. It is the vector of centres of the 500 bins of the go-cue-aligned window, so it is implicitly derived from `bp.ev.goCue` (which sets the zero) plus the chosen window and bin size. It is identical for every trial and every session, stored as a `(1, 500)` `float32` array running from −2.495 to +2.495 s.

ii.
```python
time_vec = edges[:-1] + DT / 2
...
input_trial = neural_time.reshape(1, -1).astype(np.float32)
session_input.append(input_trial)
```
```python
'input_names': ['time_from_go_cue'],
```

iii. The decoder task specifies exactly one input, *"Time from go cue onset in seconds (continuous, time-varying)"*, and the alignment event is the go cue, so the bin-centre vector is that input by construction. The AI records the alignment in metadata as `'temporal_alignment_event': 'Go cue onset'` with `off_start = -2.5`, `off_end = 2.5`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The vector is computed once per session from `TMIN`, `TMAX` and `DT` and tiled unchanged across trials. It is not smoothed, warped, normalised, or made trial-specific (e.g. no correction for differing delay durations).

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_vec = edges[:-1] + DT / 2
...
for tr in valid_trials:
    input_trial = neural_time.reshape(1, -1).astype(np.float32)
    session_input.append(input_trial)
```

iii. Not discussed in the trajectory beyond the choice of window and bin size — the axis is definitional, so there is nothing to process.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `time_vec` is derived from the same `edges` array that `np.histogram` uses to count spikes, so bin *k* of the input is exactly bin *k* of the neural matrix, by construction rather than by any matching step. The same `neural_time` variable is then also used as the interpolation target for all three video streams, so every stream in the output shares one axis.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_vec = edges[:-1] + DT / 2
...
counts, _ = np.histogram(spike_times, bins=edges)
...
neural_time = time_vec  # (n_timebins,)
speed_interp = np.interp(neural_time, ft, speed_for_interp, left=np.nan, right=np.nan)
```

iii. Not separately justified; sharing one `edges`/`time_vec` pair is the mechanism that guarantees the alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: the instructed side `R` and `L`, and the outcome flags `hit` and `miss`, plus `no` for the no-lick class. The animal's actual lick side is not recorded as such, so it is inferred from instructed side × outcome. The lick-time cell arrays `ev.lickL` / `ev.lickR` are dereferenced into `lickL_refs` / `lickR_refs` but never used.

ii.
```python
hit = bp['hit'][0, :].astype(bool)
miss = bp['miss'][0, :].astype(bool)
no = bp['no'][0, :].astype(bool)
R = bp['R'][0, :].astype(bool)
L = bp['L'][0, :].astype(bool)
```

iii. From the trajectory (step 35): *"I'm reasoning through how R/L fields relate to actual lick direction versus stimulus side. Since R and L likely indicate which side was correct rather than which side the animal actually licked, I need to combine that with hit/miss labels to infer the true lick direction on each trial."* The authors' condition strings (`'R&hit&...'`, `'L&miss&...'`) confirm that `R`/`L` are the instructed side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A three-way relabelling, broadcast constant across all 500 bins of the trial. A right lick is `(R & hit) | (L & miss)`; a left lick is `(L & hit) | (R & miss)`; a `no` trial is `none`. Codes are left 0, right 1, none 2, matching `output_values = ['left', 'right', 'none']`. The array is initialised to −1, but since only `hit|miss|no` trials survive the trial filter, no −1 reaches the output. Resulting distribution over the dataset: left 40.2%, right 42.7%, none 17.2%.

ii.
```python
lick_direction = np.full(ntrials, -1, dtype=int)
lick_direction[(R & hit) | (L & miss)] = 1   # right
lick_direction[(L & hit) | (R & miss)] = 0   # left
lick_direction[no] = 2                       # none
...
output_trial[0, :] = lick_direction[tr]
```
```python
'output_values': [
    ['left', 'right', 'none'],
    ...
```

iii. As in 4-a: a hit means the animal licked the instructed port and a miss means it licked the other one, so the pair determines the direction. The third class follows the decoder task's explicit `(left, right, none, per-trial)` specification. The AI broadcasts the per-trial value across time because the format instructions say *"If at all possible, make it time-varying"* and the value is constant within a trial.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks trials where water was delivered from a random port without any cue.

ii.
```python
autowater = bp['autowater'][0, :].astype(bool)
```

iii. The AI's exploration identified `autowater` as the WC marker, and the authors' condition strings distinguish `~autowater` (DR) from `autowater` (WC) conditions. From the trajectory: *"Mapping context to DR versus WC"* from this flag.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct binary relabelling, broadcast constant across all 500 bins: `autowater` → WC (0), otherwise DR (1). Resulting distribution: WC 16.7%, DR 83.3%. Three of the 25 sessions turn out to be entirely DR (WC fraction 0), which the AI noticed and cross-checked.

ii.
```python
context = np.where(autowater, 0, 1).astype(int)  # 0=WC, 1=DR
...
output_trial[1, :] = context[tr]
```
```python
['WC', 'DR'],
```

iii. Code order follows the decoder task's listing *"Behavioral context (WC, DR, per-trial)"*. The AI verified the mapping against session composition (step 70): *"sessions 5-7 … all show behavioral_context [1.0, 1.0], meaning they're entirely DR trials … the paper notes six mice did two-context sessions while three mice only did DR — so I'm cross-checking the WC fractions and autowater field."* It found JEB14 has 0.000 WC on one day and 0.192 on another, consistent with per-session task composition rather than a coding error.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial flags of `obj.bp`: `hit`, `miss`, and `no`. Unlike the reference, which derives "ignore" by exclusion, the AI reads the `no` flag explicitly.

ii.
```python
hit = bp['hit'][0, :].astype(bool)
miss = bp['miss'][0, :].astype(bool)
no = bp['no'][0, :].astype(bool)
```

iii. The AI's data exploration lists these as *"`hit`, `miss`, `no`: trial outcome flags"*, and the tutorial's condition 1 is `'(hit|miss|no)'` = "all trials", establishing that the three are exhaustive.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling, broadcast constant across all 500 bins: `miss` → incorrect (0), `hit` → correct (1), `no` → ignore (2). The array is initialised to −1 but the trial filter guarantees every kept trial is one of the three. Resulting distribution: incorrect 13.4%, correct 69.4%, ignore 17.2% (ignore is identical to the `none` lick class, as it must be).

ii.
```python
outcome = np.full(ntrials, -1, dtype=int)
outcome[miss] = 0   # incorrect
outcome[hit] = 1    # correct
outcome[no] = 2     # ignore
...
output_trial[2, :] = outcome[tr]
```
```python
['incorrect', 'correct', 'ignore'],
```

iii. Codes and class names follow the decoder task's *"Outcome (incorrect, correct, ignore, per-trial)"* ordering directly. Ignore trials are kept rather than dropped (the paper excludes them from most analyses) because the task requires the third class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side camera only**. Within it: `featNames` (to locate the index of the feature named `tongue`), `ts` of shape `(n_features, 3, n_frames)` giving x, y and likelihood, and `frameTimes`. `obj.bp.ev.goCue` and the constant `PAD_SEC = 0.5` (from `obj.sglx.padSec`) are used for the alignment. The bottom camera's `top_tongue` — which the reference also uses — is never read for the tongue.

ii.
```python
SIDE_TONGUE via view 0:
side_data = f[traj_group[0, 0]]
side_feats = read_feat_names(f, side_data)
tongue_idx = side_feats.index('tongue') if 'tongue' in side_feats else None
...
ts = f[side_data['ts'][tr, 0]][()]  # (n_feats, 3, n_frames)
tongue_x = ts[tongue_idx, 0, :]
tongue_y = ts[tongue_idx, 1, :]
```
```python
def read_feat_names(f, view_data):
    """Read feature names from traj view (double-deref HDF5 cell array)."""
    feat_cell_ref = view_data['featNames'][0, 0]  # first trial
    feat_cell = f[feat_cell_ref][()]  # (1, n_feats) refs
    names = []
    for i in range(feat_cell.shape[1]):
        ref = feat_cell[0, i]
        chars = f[ref][()].flatten()
        names.append(''.join(chr(int(c)) for c in chars if c > 0))
    return names
```

iii. The AI's plan (step 16) was to *"extract tongue position from DLC side and bottom camera features"*, but the implementation uses only the side view; the trajectory records the `traj_features` lists from `getDefaultParams.m` (side: `tongue, left_tongue, right_tongue, jaw, trident, nose`; bottom: `top_tongue, …, top_paw, bottom_paw, …`) and then assigns one camera per output: *"Side cam (view 0) for tongue, Bottom cam (view 1) for paw"*. No justification is given for dropping the bottom tongue view. The AI did debug this section heavily after an initial run produced 100% "not visible" (step 43–52), tracing it to the nested cell-array structure of `featNames`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Five steps. **(1)** Visibility is defined as x and y both non-NaN; the likelihood channel is never thresholded. (I verified on EKH1 that `isnan(x)` and `likelihood <= 0.9` agree on 100.000% of frames, so this is exactly the reference's rule expressed differently.) **(2)** x and y are nearest-neighbour filled across gaps (forward fill then backward fill, in a Python loop). **(3)** Speed is `hypot(d/dt x, d/dt y)` via `np.gradient` with a **fixed** `dt_video = 1/400 s` (the true median frame interval is 0.00252 s, a 0.8% scale error that is irrelevant after a percentile split). **(4)** Speed is masked back to visible frames only, NaNs replaced by 0, and then **point-sampled** onto the 500 bin centres with `np.interp` (linear interpolation between neighbouring frames, not an average over the frames in the bin — at 10 ms bins and ~400 Hz video this uses roughly one of every four frames); visibility is interpolated the same way and thresholded at 0.5. **(5)** Bins whose interpolated visibility is ≤0.5, or whose interpolated speed is not finite, become class 2. There is no Gaussian smoothing of position (the reference applies a 5 ms kernel) and no cross-camera normalisation (not needed with one view).

ii.
```python
def nearest_fill(arr):
    """Fill NaN values with nearest non-NaN value (forward then backward)."""
    out = arr.copy()
    ...
def compute_speed(x, y, dt_video):
    vx = np.gradient(x, dt_video)
    vy = np.gradient(y, dt_video)
    return np.sqrt(vx**2 + vy**2)
```
```python
tongue_nan = np.isnan(tongue_x) | np.isnan(tongue_y)
if tongue_nan.all():
    tongue_speed_trials[tr] = None
else:
    visible = ~tongue_nan
    speed = np.full_like(tongue_x, np.nan)
    if visible.sum() > 1:
        tx_filled = nearest_fill(tongue_x)
        ty_filled = nearest_fill(tongue_y)
        speed_raw = compute_speed(tx_filled, ty_filled, dt_video)
        speed[visible] = speed_raw[visible]
    tongue_speed_trials[tr] = (aligned_ft, speed, visible)
```
```python
speed_for_interp = np.where(np.isfinite(speed), speed, 0.0)
speed_interp = np.interp(neural_time, ft, speed_for_interp, left=np.nan, right=np.nan)
visible_interp = np.interp(neural_time, ft, visible.astype(float), left=0, right=0) > 0.5
speed_interp[~visible_interp] = np.nan
```

iii. The AI's summary of `getKinematicsFromVideo.m` / `findVelocity.m` says the authors *"compute velocity via `gradient()` (numerical differentiation)"* and *"interpolate video frame times to neural data time axis"*, which is what the AI does. Its own final summary states *"Tongue: NaN positions kept (not interpolated, per paper methods)"* and *"Paw/other features: NaN filled with nearest value (per paper methods)"* — a distinction the authors make in `findVelocity.m` ("for tongue: sets NaN velocity to 0") but which the implementation does **not** honour: `nearest_fill` is applied to the tongue's x and y exactly as it is to the paw's, before differentiation.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A per-session 50th percentile. All binned, visible tongue speeds from all valid trials of the session are pooled into one list and `np.median` of that list is the threshold. Then, per bin: visible and `< threshold` → 0, visible and `>= threshold` → 1, not visible → 2. Trials with no tongue data at all get all 500 bins set to 2. Across the dataset this yields 4.35% / 4.35% / 91.29% — the two visible classes are exactly balanced, as a median split requires.

ii.
```python
valid_speeds = speed_interp[visible_interp & np.isfinite(speed_interp)]
all_tongue_speeds.extend(valid_speeds.tolist())
...
tongue_thresh = np.median(all_tongue_speeds) if len(all_tongue_speeds) > 0 else 0
```
```python
tv = np.full(n_timebins, 2, dtype=np.int64)  # default: not visible
vis = visible_interp & np.isfinite(speed_interp)
tv[vis & (speed_interp < tongue_thresh)] = 0
tv[vis & (speed_interp >= tongue_thresh)] = 1
output_trial[3, :] = tv
```
```python
['below_50th', 'above_50th', 'not_visible'],
```

iii. Directly from the decoder task: *"Tongue velocity: discretized with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile, 2: not visible"*. The AI's summary lists *"Discretization: Per-session 50th percentile thresholds for velocity/ME"*. The threshold is computed in a first pass over all valid trials before any output is built, which is what makes it per-session rather than per-trial.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. `frameTimes − 0.5 − goCue[trial]`, then `np.interp` onto the shared 500-bin centres, with `left=np.nan, right=np.nan` so bins outside the recorded frame span become "not visible". The 0.5 s is a **hard-coded constant** taken from `obj.sglx.padSec`, applied identically to every session rather than computed per session.

This is wrong in two ways. The true offset computed the reference's way (`mode(sglx.bitcode.bitstart / sglx.fs) − mode(bp.ev.bitStart)`) is **0.49 s** for 21 of the 25 sessions — a 10 ms error, exactly one bin. But for the four **JEB19** sessions (2023-04-18/19/20/21), `sglx.padSec` is **1.0**, not 0.5, and the true offset is **0.99 s**: the AI's fixed 0.5 s leaves the video streams shifted 490 ms late relative to the neural data in those sessions. I confirmed this directly — on JEB19_2023-04-18 trial 10 the AI's aligned frames span [−1.985, 4.008] s where the correct span is [−2.475, 3.518] s — and the signature is visible in the AI's own output: sessions 15–18 (the four JEB19 sessions) are the only ones with ~10.4% of motion-energy bins marked "no video", exactly the ~0.5 s of the 5 s window that now has no frames.

ii.
```python
PAD_SEC = 0.5         # video-neural offset (sglx padding)
...
ft = f[side_data['frameTimes'][tr, 0]][()].flatten()
aligned_ft = ft - PAD_SEC - goCue[tr]
side_frame_times[tr] = aligned_ft
```
```python
speed_interp = np.interp(neural_time, ft, speed_for_interp, left=np.nan, right=np.nan)
```

iii. The AI found `findVideoOffset.m` and correctly summarised it (*"Formula: `vidFileOffset - bitStart` (both in seconds)"*), then explicitly raised and dismissed the question: *"I need to verify whether this padSec offset is fixed across sessions or needs to be computed per session. … Looking at findVideoOffset.m, it seems to calculate a per-trial offset between file start and frame times, then adjusts by bitstart — but the tutorial suggests a simpler fixed 0.5s padSec approach works. … The tutorial confirms `frameTimes - 0.5` converts to behavioral/neural time, matching event times like goCue, so aligning to the go cue is just `frameTimes - 0.5 - goCue`. I'll go with this simpler approach and move on."* The check it said it needed — whether `padSec` is constant across sessions — was never performed, and `padSec` is in fact read out per session by the very exploration script the AI ran.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. `obj.traj{2}`, the bottom camera, feature `top_paw`: its x and y from `ts`, plus that camera's own `frameTimes`. The bottom camera's other paw, `bottom_paw`, is not used. Same `goCue` and `PAD_SEC` for alignment.

ii.
```python
bot_data = f[traj_group[1, 0]]
bot_feats = read_feat_names(f, bot_data)
paw_idx = bot_feats.index('top_paw') if 'top_paw' in bot_feats else None
...
ts = f[bot_data['ts'][tr, 0]][()]
paw_x = ts[paw_idx, 0, :]
paw_y = ts[paw_idx, 1, :]
```

iii. The AI's read of `getDefaultParams.m` established that `top_paw` and `bottom_paw` are bottom-camera features; it chose `top_paw` (*"Bottom cam (view 1) for paw"*) without stating a reason for preferring it over `bottom_paw` or over combining the two. This happens to be the same single feature the human reference selected.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same pipeline as the tongue, with one difference: the speed is kept for **all** frames (computed from the nearest-filled positions) rather than being masked to visible frames before interpolation, and the visibility mask is applied only after interpolation. So: NaN-based visibility, forward/backward nearest fill of x and y, `np.gradient` with fixed `dt_video = 1/400`, `hypot`, `np.interp` onto the 500 bin centres, then mask by interpolated visibility. No position smoothing, no normalisation (only one camera, so values stay in pixels/s).

ii.
```python
paw_nan = np.isnan(paw_x) | np.isnan(paw_y)
if paw_nan.all():
    paw_speed_trials[tr] = None
else:
    px = nearest_fill(paw_x)
    py = nearest_fill(paw_y)
    speed_paw = compute_speed(px, py, dt_video)
    paw_visible = ~paw_nan
    paw_speed_trials[tr] = (aligned_ft, speed_paw, paw_visible)
```
```python
speed_interp = np.interp(neural_time, ft, speed, left=np.nan, right=np.nan)
visible_interp = np.interp(neural_time, ft, visible.astype(float), left=0, right=0) > 0.5
speed_interp[~visible_interp] = np.nan
```

iii. From the AI's summary: *"Paw/other features: NaN filled with nearest value (per paper methods)"*, which follows `loadMotionEnergy.m`/`findPosition.m`'s "fills NaNs with nearest value" step and `findVelocity.m`'s use of `gradient()`. No normalisation is applied because there is only one view to reconcile.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: pooled across all valid trials of the session, `np.median` of the binned visible speeds is the per-session threshold; `< threshold` → 0, `>= threshold` → 1, untracked → 2; trials with no paw data at all get 500 bins of class 2. Dataset-wide this gives 40.25% / 40.25% / 19.50%. Per-session "not visible" varies widely (0.8% to 85.7%), reflecting genuine tracking dropout.

ii.
```python
paw_thresh = np.median(all_paw_speeds) if len(all_paw_speeds) > 0 else 0
...
pv = np.full(n_timebins, 2, dtype=np.int64)  # default: not visible
vis = visible_interp & np.isfinite(speed_interp)
pv[vis & (speed_interp < paw_thresh)] = 0
pv[vis & (speed_interp >= paw_thresh)] = 1
```

iii. Directly from the decoder task's *"Paw velocity: 0: < 50th percentile, 1: >= 50th percentile, 2: not visible"* specification, applied per session.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The same fixed correction as the tongue, but read from the bottom camera's own `frameTimes`: `frameTimes − 0.5 − goCue[trial]`, then `np.interp` onto the shared bin centres with NaN outside the frame span. Using each camera's own frame times is correct (the two views can record different frame counts). The same 0.5 s constant carries the same defect: correct to within one bin for 21 sessions, off by 490 ms for the four JEB19 sessions.

ii.
```python
ft = f[bot_data['frameTimes'][tr, 0]][()].flatten()
aligned_ft = ft - PAD_SEC - goCue[tr]
```

iii. Same as 7-d: the AI adopted the tutorial's fixed `padSec` shift in preference to `findVideoOffset.m`, and did not verify that `padSec` is constant across sessions.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside each data structure, loaded with `scipy.io.loadmat`. The AI reads `me.data` as a `(ntrials, 1)` cell array of per-frame traces and `me.moveThresh` as a scalar (loaded but never used — the decoder task specifies a percentile split instead). `obj.me`, present in some sessions, is not used. The side camera's aligned `frameTimes`, cached during tongue processing, supply the time axis.

Only **one** of the three on-disk layouts is handled. Two of the 25 sessions — JEB15_2022-07-26 and JEB15_2022-07-28 — store the payload doubly wrapped as `me.data.data`, so `me_struct['data'][0, 0]` returns a struct rather than a cell array, the subsequent indexing raises, and the bare `except` sets every trial to `None`. Both sessions come out with **100% of motion-energy bins in the "no video" class** (I confirmed this in the saved pickle: sessions 11 and 13 have a class-2 fraction of exactly 1.000, against 0.004–0.010 for the other sessions).

ii.
```python
try:
    me_data = scipy.io.loadmat(me_path)
    me_struct = me_data['me']
    me_trial_data = me_struct['data'][0, 0]  # (ntrials, 1) cell array
    me_thresh = float(me_struct['moveThresh'][0, 0].flatten()[0])
    has_me = True
except Exception as e:
    print(f"    Warning: could not load motion energy: {e}")
    has_me = False
```
```python
try:
    me_trial = me_trial_data[tr, 0].flatten()
    ...
except:
    me_all[tr] = None
```

iii. The AI's summary of `loadMotionEnergy.m` records the fields `me.data`, `me.moveThresh` and `me.move`, and the step-54 check confirmed *"ME and video frames match"* on the session it tested — a session with the ordinary layout. The nested-struct variant that `loadMotionEnergy.m` itself guards against was not encountered in that spot check and is not handled.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling. The trace is already one scalar per camera frame (the paper computes it per pixel as a difference of medians over the next and previous five frames, then reduces each frame to its 99th percentile across pixels), so the AI only checks that its length matches the side camera's frame count and then `np.interp`s it onto the 500 bin centres, with NaN outside the frame span. No smoothing, differentiation, or baseline correction. Trials whose motion-energy length disagrees with the frame-time length are set to `None` and become 500 bins of class 2.

ii.
```python
me_trial = me_trial_data[tr, 0].flatten()
if len(me_trial) > 1 and tr in side_frame_times:
    ft = side_frame_times[tr]
    if len(me_trial) == len(ft):
        me_interp = np.interp(neural_time, ft, me_trial, left=np.nan, right=np.nan)
        me_all[tr] = me_interp
        all_me_values.extend(me_interp[np.isfinite(me_interp)].tolist())
    else:
        me_all[tr] = None
```

iii. `loadMotionEnergy.m` does exactly this — *"Interpolates motion energy to neural data time axis using `interp1()`"* — and the spatial reduction is already done upstream, so there is nothing left to compute. The AI verified the frame-count correspondence explicitly (step 54).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A per-session median, computed the same way as for the two velocities: every finite interpolated motion-energy value from every valid trial of the session is pooled and `np.median` is taken; `< threshold` → 0, `>= threshold` → 1, missing → 2 (labelled `no_video` rather than `not_visible`). The authors' own `me.moveThresh` is deliberately not used. Dataset-wide: 45.19% / 45.34% / 9.47%, where the class-2 mass is dominated by the two sessions that failed to load (100% each) and the four JEB19 sessions (~10.4% each, from the alignment shift described in 7-d).

ii.
```python
me_thresh_50 = np.median(all_me_values) if len(all_me_values) > 0 else 0
...
me_disc = np.full(n_timebins, 2, dtype=np.int64)  # default: no video
vis = np.isfinite(me_interp)
me_disc[vis & (me_interp < me_thresh_50)] = 0
me_disc[vis & (me_interp >= me_thresh_50)] = 1
```
```python
['below_50th', 'above_50th', 'no_video'],
```

iii. The decoder task specifies *"Motion energy: 0: < 50th percentile, 1: >= 50th percentile, 2: no video"*, which overrides the authors' `moveThresh`-based binarisation; the AI's variable name `me_thresh_50` and its separate, unused `me_thresh` show it made that substitution knowingly.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It reuses the side camera's aligned frame times, cached per trial in `side_frame_times` during tongue processing: `frameTimes − 0.5 − goCue[trial]`, then `np.interp` onto the bin centres. Choosing the side camera is correct — motion energy has exactly one value per side-camera frame, and the AI verified the lengths match and re-checks the match per trial. The dependency on the tongue block is structural: if `tongue_idx` is `None` or the side-camera read raises, `side_frame_times[tr]` is never populated and the trial's motion energy is silently dropped. The same fixed 0.5 s offset applies, so the four JEB19 sessions carry the 490 ms shift here too.

ii.
```python
ft = f[side_data['frameTimes'][tr, 0]][()].flatten()
aligned_ft = ft - PAD_SEC - goCue[tr]
side_frame_times[tr] = aligned_ft
```
```python
if len(me_trial) > 1 and tr in side_frame_times:
    ft = side_frame_times[tr]
    if len(me_trial) == len(ft):
        me_interp = np.interp(neural_time, ft, me_trial, left=np.nan, right=np.nan)
```

iii. The AI refactored this deliberately after noticing the coupling (step 58): *"both ME and video data use the same frame times (since ME is computed from the video). So I need to get the frame times for ME from the video data. … I should refactor to always store frame times per trial regardless of tongue visibility."* It stored the frame times outside the `tongue_speed_trials is None` branch, but left them inside the `if tongue_idx is not None` branch, so the coupling is reduced but not eliminated.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Uniformly by falling back to the trailing class and continuing, never by dropping the trial and never by imputing a value into the output. Concretely: (a) a trial whose tongue or paw is NaN on every frame, or whose per-trial video read raises, becomes 500 bins of class 2; (b) individual untracked frames are excluded from the speed and their bins become class 2; (c) a motion-energy trace whose length disagrees with the side camera's frame count is discarded for that trial; (d) a session whose motion-energy file is missing or unreadable gets `has_me = False` and all-class-2 motion energy; (e) a missing `obj.meta.anm` falls back to the filename; (f) sessions with <2 valid trials or <10 units would be dropped entirely (never triggered).

The mechanism throughout is a broad `try/except` with no message, which is the weak point. It converts structural failures into plausible-looking "no data" rather than surfacing them — demonstrably so in two places: the nested motion-energy layout in the two JEB15 sessions (9-a), which produces 1,000 trials' worth of silently fabricated "no video", and, earlier in the session, a bare `except` that made *every* trial's video read as "not visible" until the AI happened to notice the flat output distribution while verifying.

ii.
```python
if tongue_nan.all():
    tongue_speed_trials[tr] = None
...
except Exception:
    tongue_speed_trials[tr] = None
```
```python
else:
    output_trial[3, :] = 2  # not visible
...
else:
    output_trial[5, :] = 2  # no video
```
```python
except Exception as e:
    print(f"    Warning: could not load motion energy: {e}")
    has_me = False
```

iii. From the trajectory (step 48), after the first debugging round: *"removing the silent exception handling that was falsely marking everything as not visible."* The AI recognised the hazard of silent catches in the one place it found it, but left the pattern in place elsewhere. The general policy — keep the trial, mark the gap — is stated implicitly by the `# default: not visible` comments and is the right one; the class exists precisely so that missing frames can be represented without NaN, which the output format forbids.

## 11-a. What are the most time-consuming steps of the code?

i. Three things, in rough order. **(1)** Reading the HDF5 files: each session's `clu`, `traj` and `bp` trees are read cluster-by-cluster and trial-by-trial through individual `h5py` dereferences, and for each trial the full `(n_features, 3, n_frames)` `ts` array is materialised for both cameras even though one feature is wanted from each. **(2)** The per-spike alignment loop, a pure-Python `for` over every spike of every kept cluster. **(3)** The nested unit × trial binning-and-smoothing loop, which performs `n_units × Ntrials` separate `np.histogram` + `np.convolve` calls — on the order of 18,000 per session. The full conversion took roughly 200 s for 25 sessions (08:58:58 → ~09:02 in the trajectory), against the human reference's ~135 s for 44 sessions, i.e. about 2.6× the per-session cost.

ii.
```python
for t_idx in range(len(trialtm)):
    tr = trial_nums[t_idx] - 1  # 0-indexed
    if 0 <= tr < ntrials:
        aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
```
```python
for u_idx in range(n_units):
    for tr in range(ntrials):
        tr_mask = strials == (tr + 1)
        ...
        trialdat[u_idx, tr, :] = bin_and_smooth_spikes(tr_spikes, edges, DT, SMOOTH_WIN, SMOOTH_BC)
```

iii. Not discussed in the trajectory. The AI never profiled or commented on runtime; the ~3 minute total was evidently acceptable to it and it moved straight from a successful run to decoder validation.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Four, all straightforwardly vectorizable. **(1)** The per-spike alignment loop is one fancy-index expression: `trialtm - goCue[trial_nums - 1]`, with an `np.clip`/mask for out-of-range trials. **(2)** The `n_units × Ntrials` histogram loop collapses to one `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, BIN_EDGES])` per cluster — exactly what the reference does — removing the inner trial loop and the repeated `strials == (tr + 1)` boolean scans, which are themselves O(n_spikes) each and so make the inner loop quadratic in trial count. **(3)** `smooth_data` could be applied to the whole `(n_trials, n_bins)` matrix along `axis=1` in one call instead of once per row. **(4)** `nearest_fill`'s two explicit element-by-element Python loops are the standard `np.maximum.accumulate` index trick.

The remaining loops — over sessions, over clusters, and over trials for the camera streams — are genuinely awkward to vectorize, since each trial has a different number of frames, so those are the same ones the reference also leaves as loops.

ii.
```python
for i in range(1, len(out)):
    if np.isnan(out[i]) and not np.isnan(out[i-1]):
        out[i] = out[i-1]
for i in range(len(out)-2, -1, -1):
    if np.isnan(out[i]) and not np.isnan(out[i+1]):
        out[i] = out[i+1]
```
```python
for tr in range(ntrials):
    tr_mask = strials == (tr + 1)  # 1-indexed trials
```

iii. Not discussed. No vectorization decisions are recorded anywhere in the trajectory.

## 11-c. What processing does the code repeat multiple times?

i. Four repetitions. **(1)** `causal_gaussian_kernel(15)` is rebuilt inside every `smooth_data` call, i.e. once per unit per trial — tens of thousands of times per session — for a kernel that is a module-level constant. **(2)** The HDF5 file is closed and then **reopened** at the end of `load_session` purely to read `obj.meta.anm`, a string already available from the filename that the function also has. **(3)** Firing rates are computed for all `Ntrials`, including the ~10% of trials the filter already rejected, and only then subset by `valid_trials`. **(4)** `{q.lower() for q in EXCLUDE_QUALITIES}` is rebuilt for every cluster inside the cluster loop.

By contrast the things that *should* be computed once are: the bin edges (computed once per session — fine), the feature-name lists (once per session — fine), and the per-session percentile thresholds (once, in a genuine first pass — fine).

ii.
```python
def smooth_data(x, N, bctype='reflect'):
    if N <= 1:
        return x.copy()
    kern = causal_gaussian_kernel(N)   # rebuilt on every call
```
```python
# Get animal name
with h5py.File(data_path, 'r') as f:      # file reopened just for this
    try:
        anm = h5_read_string(f, f['obj']['meta']['anm'])
```
```python
if quality.lower() in {q.lower() for q in EXCLUDE_QUALITIES}:
```

iii. Not discussed. The reopen in particular is an artefact of the refactor that moved motion-energy loading outside the `with h5py.File(...)` block (steps 58–60); the animal-name read was left stranded after it and a second context manager was added rather than hoisting the read.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things, none of which corrupt the output. **(1)** The largest: `trialdat` is allocated and filled as `(n_units, Ntrials, 500)` float64 and smoothed for every trial, then ~10% of those trials (stim and early-lick) are thrown away — about 830 trials × ~60 units × 500 bins of wasted histogram-plus-convolution across the dataset. **(2)** Variables read from the file and never used: `sample_times`, `delay_times`, `lickL_refs`, `lickR_refs`, `all_qualities` (accumulated per cluster then dropped), and `me_thresh` (the authors' `moveThresh`, correctly superseded by the percentile split but still parsed). **(3)** For each trial the full `ts` array is read for both cameras — 7 features × 3 channels on the side, 9 × 3 on the bottom — when one feature is needed from each. **(4)** Output arrays are built as `int64` although every value is in {0, 1, 2}; `int8` would be 8× smaller. The neural data is sensibly cast to `float32`, but the pickle is still 1.14 GB for 25 sessions, i.e. a larger per-session footprint than the reference's 2.7 GB for 44 sessions at twice the temporal resolution.

ii.
```python
trialdat = np.zeros((n_units, ntrials, n_timebins))   # all trials, incl. rejected
...
trialdat = trialdat[fr_mask]
...
neural_trial = trialdat[:, tr, :]                     # only valid_trials read out
```
```python
sample_times = ev['sample'][0, :]
delay_times = ev['delay'][0, :]
lickL_refs = ev['lickL'][0, :]
lickR_refs = ev['lickR'][0, :]
...
all_qualities.append(quality)
...
me_thresh = float(me_struct['moveThresh'][0, 0].flatten()[0])
```
```python
output_trial = np.zeros((6, n_timebins), dtype=np.int64)
```

iii. Not discussed. The unused event times and lick-time references look like leftovers from the exploration phase, when the AI was still deciding whether lick direction could be read from `ev.lickL`/`ev.lickR` directly before concluding it had to be inferred from `R`/`L` × `hit`/`miss` (step 35).
