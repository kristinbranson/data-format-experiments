# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of 44 `(directory, animal, date, probes)` records in `ALL_SESSIONS`, transcribed from the authors' `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` files rather than globbing the data folders. Each record points at one `data_structure_<anm>_<date>.mat` in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`, and a sibling `motionEnergy_<anm>_<date>.mat`. Because the shared `.mat` files come in two MATLAB formats, the loader sniffs the file (`_is_hdf5`) and dispatches to one of two hand-written readers: `_load_session_h5` (h5py, MATLAB v7.3) or `_load_session_v5` (`scipy.io.loadmat`, MATLAB v5). Both readers return the same flat dict of the handful of fields the conversion actually needs (`hit`, `miss`, `early`, `no`, `L`, `R`, `autowater`, `stim.enable`, `ev.goCue`, `ev.bitStart`, `sglx.bitcode`, per-cluster `trial`/`trialtm`/`quality`, bottom-camera `ts`/`frameTimes`/`featNames`, side-camera `frameTimes`). Motion energy is loaded separately with `scipy.io.loadmat` inside `process_session`. Sessions that raise (missing file, too few trials, too few units) are skipped with a printed message; in practice all 44 were converted.

ii.
```python
EB = 'Ephys_Behavior'
RD = 'RandomizedDelay_Ephys_Behavior'

ALL_SESSIONS = [
    {'dir': EB, 'anm': 'EKH1', 'date': '2021-08-07', 'probes': [2]},
    ...
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-11-03', 'probes': [1]},
]
```

```python
def _is_hdf5(path):
    """Check if a .mat file is HDF5 (MATLAB v7.3) format."""
    try:
        with h5py.File(path, 'r') as f:
            return True
    except Exception:
        return False
...
    if _is_hdf5(data_path):
        sd = _load_session_h5(data_path, sess)
    else:
        sd = _load_session_v5(data_path, sess)
```

```python
for sess in sessions:
    result, err = process_session(sess, data_dir)
    if result is None:
        print(f"  SKIPPED: {err}")
        continue
```

iii. From the trajectory (step 44) and `CONVERSION_NOTES.md`: "Sessions were identified from the `DataLoadingScripts/Recording and video/` MATLAB loading scripts. Each `loadANM_ALMVideo.m` file specifies which sessions and probes to use for each animal." The AI dispatched an Explore subagent (step 17) to read all 16 `load<ANM>_ALMVideo.m` files and used the result verbatim. It explicitly documented three exclusions and their reasons: JEB4/JEB5 are referenced in loading scripts but their data files are not provided; JEB23 2023-10-20 is commented out in its loading script; JEB24 2023-10-03/10-04 exist on disk but are not referenced in any loading script and have no motion-energy files. For JEB15 it reasoned (step 47) that although the script carries a comment "excluding first three sessions, they look like they were in a more sensory area", the sessions are *not* commented out, so "for consistency with the reference code, I should include all sessions listed in the loading scripts." The two-reader design was justified by observation: "Some JEB23 and all JEB24 sessions" are v5 and the rest v7.3.

## 1-b. How are the data split into subjects?

i. The subject is the `anm` field of each session record (e.g. `JEB19`), which is also the animal prefix of the filename. `convert_all` builds `subjects` incrementally in first-encountered order and records `subject_idx` per session. 14 subjects result, in the order the sessions are listed (`['EKH1','EKH3','JEB6','JEB7','JGR2','JGR3','JEB13','JEB14','JEB15','JEB19','JEB11','JEB12','JEB23','JEB24']`) rather than sorted.

ii.
```python
anm = result['anm']
if anm not in subjects:
    subjects[anm] = len(subjects)
all_subject_idx.append(subjects[anm])
...
subj_list = [''] * len(subjects)
for name, idx in subjects.items():
    subj_list[idx] = name
```

iii. Not discussed explicitly in the trajectory beyond the session-list construction: the animal id is part of the per-animal loading script the sessions were transcribed from, so it is carried in the session record itself and never read out of the `.mat` file. The AI's notes group the session list by animal ("EKH1: 1 session (probe 2) … JEB24: 8 sessions (probe 1)"), confirming the animal is treated as the grouping key.

## 1-c. How are the data split into sessions?

i. One session = one entry in `ALL_SESSIONS` = one `data_structure_*.mat` file, processed independently by `process_session` and appended as one element of `neural`, `input`, `output`, `brain_region_idx` and `subject_idx`. The directory (`Ephys_Behavior` vs `RandomizedDelay_Ephys_Behavior`) is stored per session but does not otherwise change processing — fixed-delay and randomized-delay sessions are pooled into one dataset. 25 + 19 = 44 sessions. The probe(s) named in the loading script are selected per session, and two-probe sessions (JEB15 2022-07-26/27/28) have their clusters concatenated into one population.

ii.
```python
data_path = os.path.join(data_dir, sess['dir'], f"data_structure_{sname}.mat")
me_path   = os.path.join(data_dir, sess['dir'], f"motionEnergy_{sname}.mat")
...
for prb in sess['probes']:
    pidx = prb - 1
    ...
    for ci in valid_clu:
        spike_data.append({...})
```

iii. The AI reasoned (step 20, step 44) about whether to use only the two-context sessions (the paper's 12 sessions/6 mice) or all of them, and concluded that since the decoder task asks for a WC/DR context output and for maximum data, both directories and all loading-script sessions should be included: "Both Ephys_Behavior and RandomizedDelay sessions, following the loading scripts." It noted the consequence that DR-only sessions carry `context == 1` for every trial.

## 1-d. How are the data split into trials?

i. Trials come straight from the Bpod table: `bp.Ntrials` sets the count, and every per-trial field is flattened and truncated to `[:n_trials]` (several fields are stored longer than the trial count). A trial is one index into those vectors and one entry of `bp.ev.goCue`. Spikes carry a 1-based `trial` label and a within-trial time `trialtm`, and camera data is stored as one struct/cell entry per trial, so no trial boundaries need to be reconstructed.

ii.
```python
n_trials = int(bp['Ntrials'][0, 0])
hit   = bp['hit'][0, :n_trials].astype(int)
miss  = bp['miss'][0, :n_trials].astype(int)
...
goCue = bp['ev']['goCue'][0, :n_trials].copy()
```

```python
for ti in range(n_trials):
    mask = spk['trial'] == (ti + 1)
```

iii. Implicit in the reference data layout the AI explored in steps 20–59; the AI verified the field shapes interactively before writing the loader and noted the `+1` offset for the 1-based MATLAB trial numbering.

## 1-e. How are trials filtered based on quality controls?

i. A single boolean mask: keep trials that are `(hit | miss) & ~stim.enable & ~early`. That drops early-lick trials, photostimulation trials, **and all no-response ("ignore") trials**, since an ignore trial is neither a hit nor a miss. A session is skipped entirely if fewer than 2 trials survive (`MIN_TRIALS`); none were. No filter is applied for trials that run past the end of the ephys recording. 11,985 of 15,155 trials survive.

ii.
```python
MIN_TRIALS = 2       # min trials per session (for decoder)
...
# trial filter: (hit|miss) & ~stim & ~early
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]

if len(trial_idx) < MIN_TRIALS:
    return None, f"Only {len(trial_idx)} usable trials"
```

iii. `CONVERSION_NOTES.md`: "Matches `findTrials.m` condition `'(hit|miss)&~stim.enable&~early'` … No-response trials are also excluded (they are neither hit nor miss)." The AI lifted the condition string from `params.condition` in `WorkingWithDataObjs.m`, and the paper's methods state that early-lick and ignore trials "were omitted from all analyses". The AI did notice and report the recording-truncation problem afterwards but chose not to filter it: "Late trials in JEB24_2023-10-23 (session 36) and JEB24_2023-11-03 (session 43) have all-zero neural activity. This likely indicates the recording ended before the behavioral session. These trials pass through the pipeline but contribute no information."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters of the probe(s) named for that session: `obj.clu{probe}(i).trial` (1-based trial index of each spike), `.trialtm` (spike time relative to that trial's start) and `.quality` (manual curation label). The go cue times `obj.bp.ev.goCue` provide the alignment.

ii.
```python
for ci in valid_clu:
    spike_data.append({
        'trial':   f[prb_g['trial'][ci, 0]][()].flatten(),
        'trialtm': f[prb_g['trialtm'][ci, 0]][()].flatten(),
    })
```

iii. From `CONVERSION_NOTES.md`: "Align spike times: `trialtm - goCue` (matching `alignSpikes.m`)". The AI read `alignSpikes.m`, `getSeq.m` and `findClusters.m` (steps 25, 29) and mirrored the fields those functions touch.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, histogrammed into 500 non-overlapping 10 ms bins spanning −2.5 to +2.5 s, divided by the bin width to give spikes/s, and smoothed along time with a **causal** Gaussian kernel that reimplements MATLAB's `mySmooth.m` exactly: `gausswin(15, alpha=2.5)`, first `floor(15/2)=7` taps zeroed, normalised to sum 1, applied to a signal prepended with its own first 15 samples ('reflect' boundary condition as MATLAB implements it) and then trimmed. No normalisation, baseline subtraction or z-scoring; units are Hz. Clusters from both probes of a two-probe session are concatenated.

ii.
```python
def gausswin(N, alpha=2.5):
    n = np.arange(N)
    n_norm = 2 * n / (N - 1) - 1
    return np.exp(-0.5 * (alpha * n_norm) ** 2)

def causal_smooth(x, N=SMOOTH_N, bctype=BC_TYPE):
    """Causal Gaussian smoothing matching mySmooth.m."""
    kern = gausswin(N)
    kern[:N // 2] = 0   # causal: zero out first half
    kern /= kern.sum()
    if bctype == 'reflect':
        x_padded = np.concatenate([x[:N], x])
        trim = N
    ...
    out = np.convolve(x_padded, kern, mode='same')
    return out[trim:]
```

```python
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
        if not np.any(mask):
            continue
        aligned = spk['trialtm'][mask] - goCue[ti]
        counts, _ = np.histogram(aligned, bins=EDGES)
        fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. The AI read `/app/code/utils/mySmooth.m` at step 38 and reimplemented it line for line, including the causal tap-zeroing, the `cat(1,x(1:N,:),x)` form of 'reflect' padding, and the `out(N+1:end)` trim. From its reasoning (step 59): "For the causal Gaussian kernel, I'm implementing the same approach as the MATLAB code: creating a 15-sample Gaussian window, zeroing out the first half to make it causal, then normalizing it so it only looks back about 75 ms with a 10 ms sampling rate. For boundary handling with the 'reflect' condition, I'm prepending N elements from the start of the data before convolution, then trimming them out afterward." The rate conversion mirrors `getSeq.m`'s `mySmooth(N./params.dt, params.smooth, params.bctype)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. (1) Cluster quality: the free-text label is stripped and lower-cased and dropped if it is one of `garbage`, `gabrga`, `noisy`, `real?`, **or is empty**. (2) Mean firing rate: after binning and smoothing, any unit whose mean over all bins *and all trials of the session* (including the trials the trial filter discarded) is ≤ 1 Hz is dropped. (3) A session-level filter drops any session left with fewer than 10 units, before or after the rate filter; no session actually hit it. The result is 2,457 units across 44 sessions (17–141 per session).

ii.
```python
LOW_FR = 1.0         # min firing rate Hz (params.lowFR = 1)
MIN_UNITS = 10       # min units per session (per paper methods)
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
valid_clu = [i for i, q in enumerate(qualities)
             if q.lower() not in EXCLUDED_QUALITIES and q != '']
...
n_raw = len(spike_data)
if n_raw < MIN_UNITS:
    return None, f"Only {n_raw} neurons"
...
mean_fr = np.mean(fr, axis=(1, 2))
keep = mean_fr > LOW_FR
fr = fr[keep]
n_neurons = int(keep.sum())
if n_neurons < MIN_UNITS:
    return None, f"Only {n_neurons} neurons after FR filter"
```

iii. `CONVERSION_NOTES.md`: "Matches `findClusters.m` with `'all'` quality parameter: Excludes: `garbage`, `gabrga`, `noisy`, `real?`; Includes all other quality labels (excellent, great, good, fair, poor, multi). Some sessions have null-byte quality labels (`'\x00\x00'`) which are treated as valid." The AI read `findClusters.m` at step 29 and copied its exclusion list verbatim; it matches case-insensitively because the labels are inconsistently cased. The 1 Hz cut is `params.lowFR = 1` / `removeLowFRClusters.m`, and the paper's "all units with firing rates exceeding 1 Hz were included". `MIN_UNITS = 10` is annotated "per paper methods" but is not traced to a specific reference-code check in the trajectory.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction per spike. `trialtm` is already relative to its own trial's start on the behaviour clock, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]` gives time from go cue with no interpolation or offset. Spikes outside ±2.5 s fall outside the histogram edges and are discarded.

ii.
```python
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. Direct transcription of `alignSpikes.m` (`trialtm_aligned = trialtm - goCue(trial)`), which the AI cites by name in its notes and session summary (step 94): "`/app/code/DataLoadingScripts/alignSpikes.m` - `trialtm_aligned = trialtm - goCue(trial)`". `params.alignEvent = 'goCue'` in the reference and the task instructions both specify the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (`DT = 0.01`), 500 bins spanning −2.5 to +2.5 s, bin centres from −2.495 to +2.495 s. The grid is built once at module level and every stream — neural, input, and all three camera outputs — lands on it. There is no second rebinning of the neural data: spikes are histogrammed straight onto the final grid. The camera streams are *resampled* onto it by linear interpolation rather than binned (see 7-d/8-d/9-d). `metadata['time_bin_size']` is reported as 10.0 ms.

ii.
```python
DT = 0.01           # 10ms bins (params.dt = 1/100)
TMIN = -2.5          # seconds before go cue
TMAX = 2.5           # seconds after go cue
...
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
if len(EDGES) > 501:
    EDGES = EDGES[:501]
TIME_AXIS = EDGES[:-1] + DT / 2  # bin centers
T_BINS = len(TIME_AXIS)
```

iii. Taken from `WorkingWithDataObjs.m`, which the AI read at step 15 and which sets `params.tmin = -2.5; params.tmax = 2.5; params.dt = 1/100;`. The edge construction copies `getSeq.m` (`edges = params.tmin:params.dt:params.tmax; obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1)`). The AI reasoned (step 59): "With dt=0.01, the edges go from -2.5 to 2.5 inclusive, giving 501 values, and the time axis itself has 500 points after centering each bin… I'll use `np.arange(-2.5, 2.501, 0.01)` to ensure I get exactly 501 edge values." Note that the very lines above `params.dt = 1/100` in that file read "% use a 5 ms bin width and bin spike data from -2.5 to 2.5 sec"; the AI did not flag the contradiction and did not open `getDefaultParams.m`, which is the other place the bin width is set.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing in the raw data — it is the analysis grid itself, the centres of the 500 bins the spikes are counted into, defined relative to `bp.ev.goCue`. The same vector is used for every trial and every session.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2  # bin centers
```

iii. The instructions name "Time from go cue onset in seconds (continuous, time-varying)" as the only decoder input and "Go cue" as the alignment event; the AI's reasoning at step 59 treats it as a definitional quantity: "the input time series (1, 500)".

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The single input channel is `TIME_AXIS` broadcast to shape `(1, 500)`, cast to `float32` and copied once per trial. It is a continuous ramp from −2.495 to +2.495, not a binary event marker.

ii.
```python
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())  # (1, T)
```

iii. No separate justification in the trajectory beyond the format spec.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. By construction: the input *is* the vector of centres of the bins the spikes were histogrammed into, so input sample *k* and neural bin *k* are the same 10 ms interval. Both are indexed off the same module-level `EDGES`/`TIME_AXIS`.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
...
counts, _ = np.histogram(aligned, bins=EDGES)   # neural
input_list.append(TIME_AXIS[np.newaxis, :] ...) # input
```

iii. N/A — a consequence of using one shared grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `obj.bp.R` only (its complement `obj.bp.L` is loaded but never used). `bp.R` is the **instructed** side of the trial, i.e. which port the trial's stimulus/water designated, not the port the animal actually licked. `bp.hit`/`bp.miss` are read for the outcome output but are not combined with `R` to recover the licked side.

ii.
```python
L = bp['L'][0, :n_trials].astype(int)
R = bp['R'][0, :n_trials].astype(int)
...
out[0, :] = int(R[ti])                  # lick direction: L=0, R=1
```

iii. From the AI's planning reasoning (step 44): "The decoder outputs will map lick direction from the R and L fields". For the water-cued context it reasoned explicitly that the instructed side equals the licked side — "For water-contingent trials, the animal licks wherever water is presented, so right=1 means water was on the right port and the animal should lick there to get a hit" — but never revisited the delayed-response error trials, where the instructed side and the licked side are opposite. The "no lick" case was made moot by the trial filter (1-e), which removes ignore trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct cast of the `R` flag to an integer, broadcast across all 500 time bins of the trial: 0 = left, 1 = right. Two classes; `output_values[0] = ['left', 'right']`. Across the dataset the split is 49.8% / 50.2%.

ii.
```python
out = np.zeros((6, T_BINS), dtype=np.int64)
out[0, :] = int(R[ti])                  # lick direction: L=0, R=1
```

```python
'output_values': [
    ['left', 'right'],
    ...
]
```

iii. Same as 4-a. The task instruction "Lick direction (left, right, none, per-trial)" is reduced to two classes because trials with no lick were removed by the trial filter.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks the trials on which water was delivered at a random port with no cue — the water-cued block.

ii.
```python
autowater = bp['autowater'][0, :n_trials].astype(int)
```

iii. From step 44: "context from the autowater field (with autowater=0 indicating delay-response and autowater=1 indicating water-contingent)". The AI checked (step 20) that in the randomized-delay sessions autowater is essentially always 0, so those sessions are pure DR, and accepted that consequence.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A relabelling: `1 - autowater`, so WC = 0 and DR = 1, broadcast across all time bins. Dataset-wide split 8.5% WC / 91.5% DR; 18 of the 44 sessions are pure DR (context constant at 1).

ii.
```python
out[1, :] = 1 - int(autowater[ti])       # context: WC=0, DR=1
```

iii. The coding follows the instructions' "Behavioral context (WC, DR, per-trial)" ordering.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `obj.bp.hit`. `obj.bp.miss` and `obj.bp.no` are loaded and printed in the per-session summary, and `miss` participates in the trial filter, but only `hit` enters the output value.

ii.
```python
hit   = bp['hit'][0, :n_trials].astype(int)
miss  = bp['miss'][0, :n_trials].astype(int)
no_resp = bp['no'][0, :n_trials].astype(int)
```

iii. Step 44: "outcome from hit/miss labels". Since the trial filter already guarantees every surviving trial is a hit or a miss, `hit` alone is sufficient to separate the two retained classes.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A cast of the `hit` flag broadcast across time: 0 = incorrect (a miss), 1 = correct. Two classes; the "ignore" class specified in the instructions is absent because ignore trials were dropped at 1-e. Dataset-wide 13.8% incorrect / 86.2% correct.

ii.
```python
out[2, :] = int(hit[ti])                 # outcome: incorrect=0, correct=1
```

```python
'output_values': [..., ['incorrect', 'correct'], ...]
```

iii. Follows the instructions' ordering "Outcome (incorrect, correct, ignore)" for the first two classes; the third was made unreachable by the trial filter, justified in `CONVERSION_NOTES.md` as matching `findTrials.m`'s `(hit|miss)&~stim.enable&~early` condition and the paper's statement that ignore trials "were omitted from all analyses".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{2}` — the **bottom** camera only — specifically the `top_tongue` feature's x, y and likelihood channels of `ts`, plus that camera's `frameTimes`. The side camera's `tongue` feature is not used for velocity (the side camera is read only for its `frameTimes`, which serve the motion energy). `bp.ev.goCue`, `bp.ev.bitStart` and `obj.sglx.bitcode` are also needed, for the clock correction.

ii.
```python
feat_names = bot_cam['feat_names']
tongue_fi = feat_names.index('top_tongue') if 'top_tongue' in feat_names else 0
...
# h5 format - ts shape: (nfeat, 3, nframes)
ts, ft = bot_cam['trials_data'][ti]
t_spd = compute_speed(
    ts[tongue_fi, 0], ts[tongue_fi, 1], ft,
    ts[tongue_fi, 2], DLC_CONF, fill_missing=True)
```

iii. From step 59: "Looking back at the methods, the paper actually tracks tongue using both cameras, but for tongue velocity specifically they derive it from tongue angle and length measurements relative to the jaw position, so I should use the bottom camera features which include the different tongue tracking points." The AI had enumerated both cameras' features (side: `tongue, left_tongue, right_tongue, jaw, trident, nose, lickport`; bottom: `top_tongue, topleft_tongue, bottom_tongue, bottomleft_tongue, top_paw, bottom_paw, lickport, jaw, top_nostril, bottom_nostril`) and chose the bottom view because the paper's tongue angle/length are computed there.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps. **(1)** Frames are marked invalid where x or y is NaN or the DLC likelihood is < 0.9. **(2)** Those invalid frames are **filled with the nearest valid position** (`fill_positions_nearest`) — the same treatment given to the paw. **(3)** Speed is the first difference of the filled position divided by the real inter-frame interval, `sqrt((dx/dt)^2 + (dy/dt)^2)`, with a leading 0 prepended so the length matches; non-positive `dt` is replaced by 1/400 s. If *no* frame in the trial is valid the speed is all zeros. **(4)** The frame-resolution speed is linearly interpolated onto the 500-bin time axis and any remaining NaN (outside the frame range) is nearest-filled. Because the fill in step 2 holds the position constant through the ~96% of frames where the tongue is retracted, the resulting speed is exactly 0 over those stretches.

ii.
```python
def compute_speed(x, y, ft, confidence=None, conf_thresh=DLC_CONF, fill_missing=True):
    valid = ~np.isnan(x) & ~np.isnan(y)
    if confidence is not None:
        valid = valid & (confidence >= conf_thresh)

    if np.any(valid) and not np.all(valid):
        x, y = fill_positions_nearest(x, y, valid)
    elif not np.any(valid):
        return np.zeros(len(x))

    x = np.nan_to_num(x, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    dt_vid = np.diff(ft)
    dt_vid[dt_vid <= 0] = 1.0 / 400.0

    vx = np.diff(x) / dt_vid
    vy = np.diff(y) / dt_vid
    speed = np.sqrt(vx**2 + vy**2)
    speed = np.concatenate([[0.0], speed])
    return speed
```

```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The AI cycled through several options across steps 67–82. It first zeroed the speed at low confidence, then discovered the positions are already NaN ~96% of the time ("the tongue positions are NaN when confidence is low (which is ~96% of the time)"). It considered leaving them NaN, citing the paper — "the paper distinguishes between tongue and non-tongue features: tongue values are left as missing when confidence is low (indicating retraction), while jaw, paw, and other features get filled with the nearest available value" — and then went the other way anyway, documenting in `CONVERSION_NOTES.md`: "Tongue positions are NaN ~96% of frames (tongue not visible); filled with nearest valid before velocity computation", and in the docstring "filling gives near-zero velocity during non-licking periods and meaningful velocity during licking." The paper's own sentence is the opposite instruction: "Missing values were filled in with the nearest available value for all features, **except for the tongue**."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two classes only. The threshold is the 50th percentile of the **non-zero** binned tongue speeds pooled over the session's kept trials; every bin ≥ threshold is 1, everything else 0. If no non-zero value exists the threshold is set to 1.0 so everything is 0. There is **no** "not visible" class — the `not visible` bins, which by construction have speed exactly 0, silently join class 0. Resulting dataset fractions: 93.8% low / 6.2% high.

ii.
```python
# For tongue: use 50th percentile of non-zero values since tongue is
# only visible during licking (~4% of frames). Zero values (tongue not
# visible) are automatically "low".
tongue_usable = tongue_vel[:, trial_idx].flatten()
tongue_nonzero = tongue_usable[tongue_usable > 0]
if len(tongue_nonzero) > 0:
    tongue_thresh = np.percentile(tongue_nonzero, 50)
else:
    tongue_thresh = 1.0  # no tongue data: all low
...
out[3, :] = (tongue_vel[:, ti] >= tongue_thresh).astype(np.int64)
```

iii. Extensively reasoned (steps 67–82). The AI observed that a literal 50th percentile over all bins is exactly 0, which with a `>=` test puts 100% of bins in class 1 and makes the output uninformative: "The 50th percentile is 0, so everything >= 0 is 'high'… the decoder can't learn anything from it." It chose "Option 2": "compute the percentile only on non-zero velocity points so that zero velocities are automatically classified as low and the licking frames get split evenly", justifying it as "interpreting the 50th percentile as the median of meaningful (non-zero) values, since zero velocity represents 'no tongue visible' rather than actual slow movement." The trajectory contains no consideration of the third class the instructions specify ("2: not visible"), which would have dissolved the problem.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A per-session video clock offset is computed once (`vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`, ≈0.49 s) and subtracted from the bottom camera's `frameTimes`, then the trial's go cue is subtracted: `aln = frameTimes − vidshift − goCue[trial]`. The frame-resolution speed is then linearly interpolated onto `TIME_AXIS`, and bins outside the frame coverage are nearest-filled. If the offset cannot be computed the code falls back to a hard-coded 0.5 s.

ii.
```python
bitStart_mode = compute_mode(bp['ev']['bitStart'][0, :n_trials])
bc_bs = obj['sglx']['bitcode']['bitstart'][0, :]
bc_bs_mode = compute_mode(bc_bs)
fs = float(obj['sglx']['fs'][0, 0])
vidshift = bc_bs_mode / fs - bitStart_mode
...
except Exception as e:
    vidshift = 0.5
```

```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. `CONVERSION_NOTES.md`: "Matches `findVideoOffset.m`: `vidshift = mode(sglx.bitcode.bitstart) / sglx.fs - mode(bp.ev.bitStart)`". The AI read that file (step 39) and cross-checked the value: "the video offset is about 0.49 s, and the code comment says 'subtract 0.5 seconds from frameTimes to sync'. This is consistent" (step 59) — which is also where the 0.5 s fallback comes from. Interpolation onto the neural axis mirrors `loadMotionEnergy.m`'s `interp1(frameTimes-vidshift-alignTimes(trix), ..., taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom camera's `top_paw` feature in `obj.traj{2}` — x, y and likelihood from `ts`, with that camera's `frameTimes`. `bottom_paw` is a coded fallback but is never reached, since `top_paw` is present in every session.

ii.
```python
paw_fi = -1
for pname in ['top_paw', 'bottom_paw']:
    if pname in feat_names:
        paw_fi = feat_names.index(pname)
        break
```

iii. From step 59: "Paw velocity comes from the bottom camera's `top_paw` feature using the same velocity calculation", supported by the methods text the AI read: "the paws were tracked using only the bottom view." The choice of `top_paw` over `bottom_paw` is a first-match preference rather than an argued one.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue: likelihood < 0.9 or NaN frames are filled with the nearest valid position, speed is the first difference of the filled position over the real frame interval, then linear interpolation onto the 500-bin axis with nearest-fill of the ends. No normalisation; values stay in pixels/s. The same `compute_speed` call is reused with the paw feature index.

ii.
```python
p_spd = compute_speed(
    ts_p[paw_fi, 0], ts_p[paw_fi, 1], ft_p,
    ts_p[paw_fi, 2], DLC_CONF, fill_missing=True)
interp_fn = interp1d(aln, p_spd, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. `CONVERSION_NOTES.md`: "Fill invalid positions (NaN or low confidence < 0.9) with nearest valid position; Compute instantaneous speed: `sqrt(dx/dt^2 + dy/dt^2)`". For the paw this matches the paper's stated method exactly — "Missing values were filled in with the nearest available value for all features, except for the tongue. The velocity of each feature was then calculated as the first-order derivative of the position vector."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Two classes, split at the plain 50th percentile of all binned paw values over the session's kept trials — no non-zero restriction here. Bins ≥ threshold are 1. Again no "not visible" class. This yields a clean 50/50 split in 42 of 44 sessions; in JEB13 2022-09-13 and JEB13 2022-09-21 the paw tracking fails throughout, every value is 0, the median is 0, and the `>=` test makes 100% of bins class 1.

ii.
```python
paw_thresh = np.percentile(paw_vel[:, trial_idx], 50)
...
out[4, :] = (paw_vel[:, ti] >= paw_thresh).astype(np.int64)
```

iii. `CONVERSION_NOTES.md`: "Paw Velocity and Motion Energy Discretization: Standard 50th percentile threshold computed on all values per session." The AI recorded the two degenerate sessions as a known defect: "JEB13 2022-09-13 and JEB13 2022-09-21 have paw velocity threshold of 0 (all paw velocity values are >= 0, so all are 'high'). This occurs when the DLC paw tracking has very low confidence throughout" — and left them in.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as the tongue, sharing the same `aln` vector computed once per trial from the bottom camera's frame times: `frameTimes − vidshift − goCue[trial]`, then linear interpolation onto `TIME_AXIS` with nearest-fill of the ends.

ii.
```python
aln = ft - vidshift - goCue[ti]      # computed once for the tongue, reused for the paw
...
interp_fn = interp1d(aln, p_spd, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. Same justification as 7-d — one session-wide clock correction from `findVideoOffset.m` and one shared time axis. Both features come from the same camera, so one `aln` serves both.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside the data structure, read as `me['data']`, a per-trial cell of one value per camera frame. `obj.me` is not consulted. The side camera's `frameTimes` from `obj.traj{1}` supply the time base.

ii.
```python
me_mat = scipy.io.loadmat(me_path, squeeze_me=False)
me_struct = me_mat['me']
me_trials = me_struct['data'][0, 0]
...
for ti in range(min(n_trials, me_trials.shape[0])):
    trial_me = me_trials[ti, 0].flatten().astype(np.float64)
```

iii. `CONVERSION_NOTES.md`: "Loaded from separate `motionEnergy_*.mat` files (always MATLAB v5 format); Aligned using side camera frameTimes and video offset." This follows `loadMotionEnergy.m`, which the AI read at step 30 and which locates a `motionEnergy*<date>` file and interpolates `me.data{trix}` against `obj.traj{1}(trix).frameTimes`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling: the value is already one number per frame (the paper's 99th percentile across pixels of the frame-difference image), so it is only linearly interpolated onto the 500-bin axis and nearest-filled at the ends. The three struct layouts the files come in are *not* all handled — only `me['data']` one level deep — so for 7 of the 44 sessions loading raises and the array is left as its initialised all-zeros.

ii.
```python
me_data = np.zeros((T_BINS, n_trials), dtype=np.float32)

if os.path.exists(me_path):
    try:
        ...
        interp_fn = interp1d(aln, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
        me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
        print(f"  Loaded motion energy")
    except Exception as e:
        print(f"  WARNING: motion energy: {e}")
```

iii. `CONVERSION_NOTES.md` documents the failure rather than fixing it: "Some sessions had motion energy loading issues (different struct format); these sessions have all-zero motion energy data… **JEB15 2022-07-26 and 2022-07-28**: Motion energy struct has unexpected format (`dtype([('data', 'O'), ('moveThresh', 'O')])`); **JEB23 2023-10-10 through 2023-10-13**: Array indexing issue; **JEB24 2023-10-31**: Same struct format issue." `loadMotionEnergy.m`, which the AI had read, contains exactly the guard that would have fixed this: `if isstruct(me.data), me.data = me.data.data; end`. The `fillmissing`/`interp1` steps do faithfully mirror that file.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Two classes, split at the plain 50th percentile of all binned motion-energy values over the session's kept trials, `>=` giving class 1. No "no video" class. In the 37 sessions that loaded, this gives a 50/50 split; in the 7 sessions that failed to load, every value is 0, the median is 0, and all bins become class 1. Dataset-wide 41.5% low / 58.5% high.

ii.
```python
me_thresh = np.percentile(me_data[:, trial_idx], 50)
...
out[5, :] = (me_data[:, ti] >= me_thresh).astype(np.int64)
```

iii. `CONVERSION_NOTES.md`: "Standard 50th percentile threshold computed on all values per session", and for the failures: "These sessions have all-zero motion energy, so the motion_energy output is all 'high' (0 >= threshold of 0). This is flagged in the verification output where motion_energy fraction is 0.000/1.000 for those sessions." The AI saw the consequence in its own verification output and shipped it. Note the paper's own thresholding is a manually chosen per-session bimodality split (`me.moveThresh`, which is present in the files); the instructions override this with the 50th percentile, which the AI followed.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Using the **side** camera's frame times, corrected by the same session `vidshift` and the trial's go cue, then linearly interpolated onto `TIME_AXIS` and nearest-filled. If the side camera frame times are missing for that trial, or if their count does not match the motion-energy trace length, the code falls back to synthesising frame times at 400 Hz with a fixed 0.5 s offset.

ii.
```python
if ti in side_ft:
    aln = side_ft[ti] - vidshift - goCue[ti]
    if len(trial_me) != len(aln):
        aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
else:
    aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]

interp_fn = interp1d(aln, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. This is a direct port of `loadMotionEnergy.m`, including its `try`/`catch` fallback: `frameTimes = (1:size(obj.traj{1}(trix).ts,1)) ./ 400; me.newdata(:,trix) = interp1(frameTimes-0.5-alignTimes(trix), me.data{trix}, taxis);`. The AI read that function at step 30 and reproduced both branches and the subsequent `fillmissing(...,'nearest')`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Everything is filled or zeroed rather than marked; nothing is ever dropped for being missing, and no output value ever carries a "missing" code. Concretely: **untracked camera frames** (NaN or likelihood < 0.9) are replaced with the nearest valid position, so their velocity becomes 0; **a trial with no valid frame at all** returns an all-zero speed; **bins outside the camera's coverage** after interpolation are nearest-filled (`fill_nan_nearest`), and if a whole trace is NaN it is set to 0; **a missing or unreadably-structured motion-energy file** leaves that session's motion energy at all zeros; **a failed video-offset computation** falls back to a hard-coded 0.5 s; **a mismatch between the side camera's frame count and the motion-energy length** triggers a synthetic 400 Hz time base. All of these are wrapped in broad `try/except` blocks, several of which (`except Exception: pass` inside the per-trial DLC loop) swallow the error silently. Separately, the ~49 trials that occur after the ephys recording stopped are kept with all-zero firing rates.

ii.
```python
def fill_nan_nearest(arr):
    """Fill NaN values with nearest non-NaN, matching MATLAB fillmissing('nearest')."""
    nans = np.isnan(arr)
    if not np.any(nans):
        return arr
    if np.all(nans):
        arr[:] = 0
        return arr
    ...
```

```python
    elif not np.any(valid):
        return np.zeros(len(x))
```

```python
            except Exception:
                pass
    except Exception as e:
        print(f"  WARNING: DLC processing: {e}")
```

iii. The fill-with-nearest policy is justified from the paper and from `loadMotionEnergy.m`'s `fillmissing(me.data,'nearest')` — `CONVERSION_NOTES.md` describes `fill_nan_nearest` as "matching MATLAB `fillmissing('nearest')`". The zero-filling of unrecoverable cases has no stated justification; the AI documented the resulting artefacts (all-zero neural trials, all-"high" motion energy in 7 sessions, all-"high" paw in 2 sessions) in its notes but chose to accept them: "These trials pass through the pipeline but contribute no information."

## 11-a. What are the most time-consuming steps of the code?

i. The per-neuron, per-trial spike binning and smoothing loop dominates. For each session it runs `n_clusters × n_trials` iterations (up to ~140 × ~470 ≈ 66,000), each doing a boolean mask over the cluster's full spike vector, a `np.histogram`, and a 500-point `np.convolve`. File reading is second: the HDF5 reader dereferences one object reference per cluster per field and per trial per camera field, and `_is_hdf5` opens each file an extra time before the real load.

ii.
```python
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
        if not np.any(mask):
            continue
        aligned = spk['trialtm'][mask] - goCue[ti]
        counts, _ = np.histogram(aligned, bins=EDGES)
        fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. Not discussed in the trajectory; the AI never profiled or optimised the conversion, and ran the full job as a background task.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. (1) The neuron × trial spike loop above: the inner trial loop is fully avoidable with a single `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, EDGES])` per cluster, which also removes the repeated `spk['trial'] == ti+1` scan (currently O(n_trials × n_spikes) per cluster). (2) The smoothing, which is applied one 500-sample trace at a time although `np.apply_along_axis`/FFT convolution over the whole `(n_trials, 500)` block would do. (3) The quality-label loops in both readers, which build a Python list one `h5str`/`.item()` call at a time. The per-trial camera loops genuinely cannot be vectorised, since each trial has a different frame count.

ii.
```python
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)     # rescans the whole spike vector per trial
```

```python
            for i in range(n_clu):
                try:
                    q = h5str(f, quality_ds[i, 0]).strip()
```

iii. No justification offered — the AI did not raise efficiency as a consideration anywhere in the trajectory.

## 11-c. What processing does the code repeat multiple times?

i. Three repeats. (1) `_is_hdf5` opens every `.mat` file with h5py purely to test the format, and the real reader then opens it again. (2) In the v5 branch, `bot_cam['get_trial'](ti)` re-reads the trial's `ts` and `frameTimes` out of the struct array for the tongue; the paw path then correctly reuses them (`ts_p, ft_p = ts, ft`), so the duplication is avoided there but the accessor pattern still hides a re-read per trial. (3) The alignment vector `aln` and the two `interp1d` objects are rebuilt per trial per feature, which is inherent, but the session-wide `vidshift` is correctly computed once. On the other side, side-camera `frameTimes` are read for all trials of every session even when the motion-energy file is missing or fails, and the whole `fr` array is computed for *all* trials including those the trial filter will discard.

ii.
```python
def _is_hdf5(path):
    try:
        with h5py.File(path, 'r') as f:
            return True
    ...
if _is_hdf5(data_path):
    sd = _load_session_h5(data_path, sess)
```

```python
vidshift = bc_bs_mode / fs - bitStart_mode   # once per session — correctly not repeated
```

iii. Not discussed.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four things. (1) Firing rates are computed and smoothed for **every** trial of the session, then ~20% are thrown away by `trial_idx` — about 3,200 trials' worth of binning and convolution wasted. (2) `L` and `no_resp` are loaded and, apart from a printed summary line, never used (`no_resp` only appears in the log; `L` nowhere). (3) Tongue and paw velocities are computed for all `n_trials`, again with the excluded trials discarded at assembly. (4) `side_ft` is populated for every trial of every session even where the motion energy never loads. Also, the low-firing-rate criterion is evaluated on the mean over *all* trials rather than the kept ones, so the discarded trials influence which units survive — unnecessary work that additionally changes the result.

ii.
```python
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)   # all trials
...
mean_fr = np.mean(fr, axis=(1, 2))                           # over all trials, incl. discarded
keep = mean_fr > LOW_FR
...
for ti in trial_idx:                                          # only now are trials subset
    neural_list.append(fr[:, :, ti].copy())
```

```python
L = bp['L'][0, :n_trials].astype(int)        # never used
no_resp = bp['no'][0, :n_trials].astype(int) # printed only
```

iii. Not discussed. The structure follows the MATLAB pipeline, which likewise computes `obj.trialdat` for all `Ntrials` before conditions are applied, so the extra work is inherited from the reference rather than chosen.
