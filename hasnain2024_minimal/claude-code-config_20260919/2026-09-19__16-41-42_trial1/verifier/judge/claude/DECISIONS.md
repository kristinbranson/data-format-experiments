# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. One session = one MATLAB file `data_structure_<anm>_<date>.mat`, read directly with `h5py` (all files the AI touches are MATLAB v7.3/HDF5; no `scipy.io` fallback is implemented for the data structures). The session list is **hard-coded**, transcribed from the authors' `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`, together with the ALM probe number(s) each entry specifies. `DATA_DIR` is fixed to `/app/data/Ephys_Behavior`, so only the **25 fixed-delay** sessions are loaded; the 19 randomized-delay sessions in `/app/data/RandomizedDelay_Ephys_Behavior` (`loadJEB11/12/23/24_ALMVideo.m`) and the two `*_BilatMC_Behavior` opto folders are not loaded. Motion energy is read per session from `motionEnergy_<anm>_<date>.mat` with `scipy.io.loadmat`. Within a session, `obj.bp` (behaviour), `obj.clu` (spike clusters), `obj.traj` (DeepLabCut), and `obj.sglx` (clock/bitcode) are read; every per-trial field is indexed by trial number.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'
...
SESSIONS = [
    ('EKH1',  '2021-08-07', [2]),
    ('EKH3',  '2021-08-11', [2]),
    ...
    ('JEB19', '2023-04-21', [1]),
]
```
```python
def process_session(anm, date, probes):
    path = os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat')
    with h5py.File(path, 'r') as f:
        obj = f['obj']
        beh = load_behavior(obj)
        ...
        rates, qualities = load_spikes(f, obj, probes, align, trial_mask)
        ...
        vidshift = video_shift(f, obj)
        pos, has_video, frame_times = load_video_traces(f, obj, align, n, vidshift)
    me = load_motion_energy(anm, date, align, frame_times, has_video[0], n)
```
```python
def main():
    ...
    for anm, date, probes in SESSIONS:
        neural, inp, out, info = process_session(anm, date, probes)
```

iii. From the final report: the 25 sessions are "transcribed from the authors' own `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`, including their per-session ALM probe designation (commented-out sessions stay excluded; `JEB15` uses both probes as the authors do). This is the paper's main dataset (25 sessions / 1,651 units), of which 12 are the two-context WC/DR sessions." The randomized-delay folder was excluded because "it is a separate task variant (delay 0.3–3.6 s) analysed separately in the paper, so the sample/delay epochs would land at inconsistent times inside a go-cue-aligned window, and it contributes almost no WC trials." The `*_Behavior` opto folders were excluded because "they have no neural data" — the AI verified this by listing `obj` keys (`bp, ex, me, pth, sglx, traj, trials`, no `clu`).

## 1-b. How are the data split into subjects?

i. The animal is the first element of each `SESSIONS` tuple (equivalently the prefix of the file name). `subjects` is built in order of first appearance while iterating the session list, and `subject_idx` is that animal's index for each session. The 25 sessions come from 10 animals (EKH1, EKH3, JEB6, JEB7, JGR2, JGR3, JEB13, JEB14, JEB15, JEB19).

ii.
```python
        if anm not in subjects:
            subjects.append(anm)
        ...
        data['subject_idx'].append(subjects.index(anm))
...
    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. No explicit statement in the trajectory beyond the session table itself: the animal id is part of the hard-coded session key (and of the file name), which is also how the authors' `load<ANM>_ALMVideo.m` scripts organise sessions. The AI did inspect `obj.meta` in the files (steps 27–31) and fell back on the file-name/animal grouping.

## 1-c. How are the data split into sessions?

i. One entry of `SESSIONS` = one file = one element of `neural` / `input` / `output` / `brain_region_idx` / `subject_idx`. Two-probe sessions (`JEB15` 2022-07-26/27/28) are **not** split: the units of both probes are concatenated into one population for that session. After processing, a session is dropped if it has fewer than 10 surviving units (`MIN_UNITS`, the paper's inclusion criterion) or fewer than 2 trials; in practice neither filter removed anything, so 25 sessions are written.

ii.
```python
MIN_UNITS = 10       # "sessions were included only if they had at least 10 units"
...
        if info['nunits'] < MIN_UNITS:
            print(f"  skipping {anm} {date}: only {info['nunits']} units")
            continue
        if len(neural) < 2:
            print(f"  skipping {anm} {date}: only {len(neural)} trials")
            continue
```
```python
    for prb in probes:
        cl = f[clu[prb - 1, 0]]
        ...
            rates.append(rate)
```

iii. "Sessions need ≥10 units (none were dropped)" — from the paper's methods, which the AI read: "Recording sessions were included for analysis only if they had at least 10 units." Session identity and probe assignment come from the authors' load scripts (see 1-a).

## 1-d. How are the data split into trials?

i. Trials are the rows of the Bpod table `obj.bp`: `Ntrials` gives the count and every behavioural field (`R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`, `ev.goCue`) has one entry per trial. Spikes carry their trial number (`clu.trial`, 1-based) and each camera trial has its own `frameTimes`/`ts` cell entry, so no trial boundaries have to be reconstructed. The AI explicitly verified across all 25 sessions that every one of these vectors has exactly `Ntrials` entries, that `goCue` has no NaNs, that `R+L == 1` and `hit+miss+no == 1` on every trial (step 77).

ii.
```python
def load_behavior(obj):
    bp = obj['bp']
    n = int(_vec(bp, 'Ntrials')[0])
    beh = {
        'ntrials': n,
        'R': _vec(bp, 'R') > 0.5,
        ...
        'align': _vec(bp['ev'], ALIGN_EVENT),
    }
    return beh
```
```python
            trials = np.array(f[trial_refs[iclu]]).flatten().astype(int) - 1
            times = np.array(f[tm_refs[iclu]]).flatten()
            times = times - align_times[trials]
```

iii. Not discussed in prose, but the verification in step 77 (`lens {N}`, `nanGC 0`, `R+L!=1: 0`, `hit+miss+no!=1: 0` for every session) is the evidence the AI collected before writing the script.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, applied before anything is computed: early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable != 0`) are dropped, exactly the `~stim.enable & ~early` restriction used by every `params.condition` in the paper's figure scripts. Ignore ("no response") trials are deliberately **kept**, because "none"/"ignore" are required output classes for this decoding task. 661 early-lick and 173 photostim trials are removed, leaving 7,426 of 8,260 trials. No trial is removed for missing video, and there is no filter for trials that fall after the end of the ephys recording (the AI checked spike coverage on five sessions in step 64; I verified independently that on all 25 fixed-delay sessions the largest spiking trial number equals `Ntrials`, so such a filter would be a no-op here).

ii.
```python
        # every condition in the paper's scripts is restricted to
        # `~stim.enable & ~early`: photoinactivation trials perturb ALM
        # activity, and "early lick" trials are omitted from all analyses.
        # Ignore ("no") trials are *kept*, because "ignore" / "no lick" are
        # required output categories for this decoding task.
        trial_mask = (~beh['early']) & (~beh['stim'])
        keep = np.flatnonzero(trial_mask)
```

iii. "Trials: drop photostim (`stim.enable`) and early-lick trials, as every `params.condition` in the paper does; **keep** ignore trials, since 'none'/'ignore' are required output categories."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` — the spike-sorted clusters of the probe(s) the authors' load script designates as ALM. Three fields per cluster are used: `quality` (manual curation label), `trial` (1-based trial of each spike) and `trialtm` (spike time relative to that trial's start). `obj.bp.ev.goCue` supplies the alignment time. Clusters from both probes of a two-probe session are concatenated.

ii.
```python
    clu = np.array(obj['clu'])
    ...
    for prb in probes:
        cl = f[clu[prb - 1, 0]]
        quality_refs = np.array(cl['quality']).flatten()
        trial_refs = np.array(cl['trial']).flatten()
        tm_refs = np.array(cl['trialtm']).flatten()
```

iii. The module docstring cites `alignSpikes.m` + `getSeq.m`; the AI read those files (step 12) and the data layout (steps 25–29, 43–45) before choosing these fields.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 10 ms bins over [−2.5, 2.5] s from the go cue, divided by the bin width to give spikes/s, and smoothed along time with the paper's **causal** Gaussian kernel: `gausswin(15)` with the first `floor(15/2)=7` taps zeroed and renormalised, convolved with `'reflect'` boundary handling (MATLAB's `mySmooth` prepends the first 15 samples and trims them afterwards — the AI ports this literally, including the fact that MATLAB's "reflect" is actually a copy, not a mirror). No normalisation, baseline subtraction or z-scoring. Values are stored as `float32` Hz.

ii.
```python
def gausswin(n, alpha=2.5):
    """MATLAB's gausswin(n)."""
    k = np.arange(n)
    return np.exp(-0.5 * (alpha * (k - (n - 1) / 2) / ((n - 1) / 2)) ** 2)


def my_smooth(x, n=SMOOTH_N):
    """Port of `utils/mySmooth.m` with bctype='reflect' (operates on axis 0)."""
    if n <= 1:
        return x
    kern = gausswin(n)
    kern[:n // 2] = 0.0          # causal
    kern = kern / kern.sum()
    x = np.atleast_2d(x.T).T
    padded = np.concatenate([x[:n], x], axis=0)
    out = np.empty_like(padded)
    for j in range(padded.shape[1]):
        out[:, j] = np.convolve(padded[:, j], kern, mode='same')
    return out[n:]
```
```python
            counts = np.zeros((keep_trials.size, NT))
            pos = trial_pos[trials]
            inwin = (pos >= 0) & (times >= TMIN) & (times < TMAX)
            if inwin.any():
                bin_idx = np.floor((times[inwin] - TMIN) / DT).astype(int)
                np.add.at(counts, (pos[inwin], bin_idx), 1.0)
            rate = my_smooth((counts / DT).T).T     # smooth over time
```

iii. "Alignment/binning (`alignSpikes.m`, `getSeq.m`): `obj.bp.ev.goCue`, −2.5 to 2.5 s, 10 ms bins, counts→spikes/s, smoothed with the paper's causal Gaussian kernel (`mySmooth`, N=15, `reflect`)." The AI read `mySmooth.m` and grepped every figure script for `params.smooth` (step 61) before porting it.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit filters plus one session filter. (1) The manual curation label `clu.quality`, stripped and lower-cased, is compared against `{garbage, gabrga, noisy, real?}` — exactly the drop list in `findClusters.m` for `params.quality = {'all'}`. Everything else is kept, including `Multi` and `Poor` units. Non-string labels become `''` and are kept. (2) Units whose mean rate over the whole window and all kept trials is ≤ 1 Hz are dropped (`params.lowFR`, and the paper's "all units with firing rates exceeding 1 Hz were included in all other analyses"). (3) Sessions with < 10 surviving units would be dropped. This leaves 1,531 units (27–141 per session).

ii.
```python
# `findClusters.m`: quality label 'all' keeps everything except these
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
...
            qual = _char(f[quality_refs[iclu]]).strip()
            if qual.lower() in BAD_QUALITY:
                continue
```
```python
        # `removeLowFRClusters.m`: drop units below params.lowFR (1 Hz)
        mean_fr = rates.mean(axis=(1, 2)) if rates.size else np.zeros(0)
        use = mean_fr > LOW_FR
        rates = rates[use]
```

iii. "Units: `findClusters` quality rule (drop garbage/noisy/real?) plus mean FR > 1 Hz (`params.lowFR`); sessions need ≥10 units." The AI read `findClusters.m` and `removeLowFRClusters.m` (steps 12, 14) and scanned the actual label vocabulary in the files (`Poor`, `Multi`, `Fair`, `Good`, `Excellent`, `Great`, …), which is why it matches case-insensitively.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A single subtraction. `clu.trialtm` is already on the behaviour clock and relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]` gives seconds from go-cue onset. Spikes outside [−2.5, 2.5) are discarded by the `inwin` mask. `ALIGN_EVENT = 'goCue'` for every session, DR and WC alike (in WC trials there is no audible cue; `goCue` marks the equivalent response-epoch onset). The AI checked the alignment empirically: tongue visibility, motion energy and mean firing rate all jump immediately after t = 0 (step 109).

ii.
```python
ALIGN_EVENT = 'goCue'
...
            # `alignSpikes.m`: trialtm_aligned = trialtm - event(trial)
            times = times - align_times[trials]
            ...
            inwin = (pos >= 0) & (times >= TMIN) & (times < TMAX)
```

iii. "Verified alignment empirically: tongue visibility, motion energy and mean firing rate all jump immediately after t=0." The rule itself is `alignSpikes.m` with `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (`DT = 0.01`, i.e. `params.dt = 1/100`), 500 bins spanning −2.5 to +2.5 s, identical for every trial and session. `metadata['time_bin_size'] = 10.0` ms. The grid is built once at module level; the bin centres are `obj.time` as constructed in `getSeq.m`. There is no rebinning: spikes are histogrammed straight onto this grid, and the camera streams are interpolated onto the same bin centres (they are never first binned at frame resolution and then rebinned).

ii.
```python
TMIN = -2.5          # s, relative to the alignment event
TMAX = 2.5           # s
DT = 0.01            # s (10 ms bins, params.dt = 1/100)
...
# time axis, identical to `obj.time` built in `getSeq.m`
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]
NT = len(TAXIS)
```

iii. The AI grepped every analysis script for `params.dt` (step 61) and found both `1/200` (`getDefaultParams.m`, `Figure1e.m`) and `1/100` (`WorkingWithDataObjs.m`, `EDFigure2a`, `EDFigure3`, `dimensionality.m`, all behaviour scripts) and cites `WorkingWithDataObjs.m` / `EDFigure2a_Left.m` in the header comment; `params.tmin/tmax = -2.5/2.5` is common to all of them.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing from the raw data as such — it is the analysis time axis defined by the alignment window itself (`obj.bp.ev.goCue` defines t = 0, the window is −2.5 to +2.5 s). The input is the vector of the 500 bin centres, identical for every trial and session.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]
...
    'input_names': ['time_from_go_cue'],
```

iii. Implied by the decoder spec ("Time from go cue onset in seconds (continuous, time-varying)") and by the go-cue alignment; not separately discussed.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The bin-centre vector is cast to `float32`, reshaped to `(1, 500)` and copied into every trial of every session; its range is [−2.495, +2.495].

ii.
```python
    time_input = TAXIS.astype(np.float32).reshape(1, NT)
    for pos_i, trial in enumerate(keep):
        ...
        input_trials.append(time_input.copy())
```

iii. N/A.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural grid. Spike times are expressed relative to the go cue and floored into `(t − TMIN)/DT`, so bin *k* covers `[TMIN + kΔ, TMIN + (k+1)Δ)`, and `TAXIS[k]` is the centre of that same interval. The three camera streams are interpolated at exactly these `TAXIS` points, so all four streams share one axis by construction.

ii.
```python
                bin_idx = np.floor((times[inwin] - TMIN) / DT).astype(int)
```
```python
TAXIS = (EDGES + DT / 2)[:-1]
...
                pos[(view, feat)][0][trial] = interp_nan(TAXIS, t_rel, ts[fi, 0])
```

iii. N/A (this mirrors `getSeq.m`, where `obj.time` is the common axis that `getKinematicsFromVideo.m` and `loadMotionEnergy.m` also interpolate onto).

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: the instructed side `R` and `L`, and the outcome flags `hit` and `miss`. The licked side is not recorded directly, so it is inferred from the instructed side combined with whether the animal was rewarded.

ii.
```python
        'R': _vec(bp, 'R') > 0.5,
        'L': _vec(bp, 'L') > 0.5,
        'hit': _vec(bp, 'hit') > 0.5,
        'miss': _vec(bp, 'miss') > 0.5,
```

iii. "Lick direction from `(R&hit)|(L&miss)` per `getPrevChoice.m`" — the AI read `funcs/getPrevChoice.m` (step 51), which derives the animal's choice from exactly this combination.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A right lick is `(R & hit) | (L & miss)`, a left lick is `(L & hit) | (R & miss)`, and everything else (i.e. the `no`/ignore trials) is "none". Codes: left 0, right 1, none 2. The per-trial value is broadcast across all 500 bins so the output array is time-varying in shape.

ii.
```python
    # lick direction: `funcs/getPrevChoice.m` -- the animal licked right on
    # (right trial & hit) or (left trial & miss); "no" trials have no lick.
    right = (beh['R'] & beh['hit']) | (beh['L'] & beh['miss'])
    left = (beh['L'] & beh['hit']) | (beh['R'] & beh['miss'])
    lick_dir = np.full(n, 2, dtype=np.int8)      # 2 = none
    lick_dir[left] = 0
    lick_dir[right] = 1
```
```python
        out[0] = lick_dir[trial]
```
with `'output_values'[0] = ['left', 'right', 'none']`.

iii. Same as 4-a; the third class exists because ignore trials are kept and the decoder spec asks for `left, right, none`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks the water-cued (WC) block trials where water is delivered from a random port with no auditory cue.

ii.
```python
        'autowater': _vec(bp, 'autowater') > 0.5,
```

iii. "Context from `obj.bp.autowater`." The AI also checked that autowater trials come in contiguous blocks (95–98 % of neighbouring autowater trials are consecutive, step 79), confirming the flag marks WC blocks.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: autowater → WC (0), everything else → DR (1), broadcast across the 500 bins.

ii.
```python
    # context: `obj.bp.autowater` marks water-cued (WC) trials
    context = np.where(beh['autowater'], 0, 1).astype(np.int8)
```
with `'output_values'[1] = ['WC', 'DR']`.

iii. The 0 = WC / 1 = DR ordering follows the decoder spec's "(WC, DR)".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The per-trial flags `obj.bp.hit` and `obj.bp.miss`. `bp.no` is also read into the behaviour dict but is not used: a trial that is neither hit nor miss is an ignore by construction, and the AI verified `hit + miss + no == 1` on every trial of every session (step 77).

ii.
```python
        'hit': _vec(bp, 'hit') > 0.5,
        'miss': _vec(bp, 'miss') > 0.5,
        'no': _vec(bp, 'no') > 0.5,
```

iii. "Outcome from hit/miss/no" (`funcs/getOutcome.m`, read in step 51).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Three classes, broadcast across bins: incorrect (0) on miss trials, correct (1) on hits, ignore (2) otherwise.

ii.
```python
    # outcome: hit / miss / ignore (`funcs/getOutcome.m`)
    outcome = np.full(n, 2, dtype=np.int8)       # 2 = ignore
    outcome[beh['miss']] = 0
    outcome[beh['hit']] = 1
```
with `'output_values'[2] = ['incorrect', 'correct', 'ignore']`.

iii. Ordering follows the decoder spec ("incorrect, correct, ignore"); ignore trials are retained rather than dropped so that the third class exists.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, **side camera only** (view 0), feature `tongue`: its `ts` array (x, y, likelihood per frame — the authors already store x/y as NaN where the likelihood is low) and its `frameTimes`. `obj.sglx.fs`, `obj.sglx.bitcode.bitstart` and `obj.bp.ev.bitStart` are needed for the video-clock offset, and `bp.ev.goCue` for the alignment. The bottom-camera tongue features (`top_tongue`, `bottom_tongue`, …) are not used.

ii.
```python
# DLC features used for the two kinematic outputs.  The tongue is tracked from
# the side camera (view 1) and the two paws from the bottom camera (view 2),
# following `params.traj_features` in the paper's scripts.
TONGUE_FEATURES = [(0, 'tongue')]
```
```python
    for view in views:
        tv = f[traj[view, 0]]
        names = _cellstr(f, f[np.array(tv['featNames']).flatten()[0]])
        wanted = [(feat, names.index(feat))
                  for v, feat in TONGUE_FEATURES + PAW_FEATURES
                  if v == view and feat in names]
        view_data[view] = (wanted,
                           np.array(tv['frameTimes']).flatten(),
                           np.array(tv['ts']).flatten())
```

iii. "Kinematics follow `findPosition.m`/`findVelocity.m` — side-camera `tongue` (never gap-filled, so DLC non-detection *is* 'not visible')." The AI measured the NaN fraction and mean likelihood of every tracked feature in both views (step 66: side `tongue` NaN 0.907 / conf 0.095, bottom `top_tongue` NaN 0.912) but did not state a reason for preferring the side view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Following `findPosition.m` / `findVelocity.m`: (1) frame times are put on the go-cue clock (7-d) and x, y are linearly interpolated onto the 500-bin `TAXIS` with `np.interp`, which returns NaN outside the frame support and propagates the DLC NaNs to neighbouring query points; (2) **no smoothing** is applied (the MATLAB explicitly skips smoothing for tongue features); (3) within each contiguous run of bins where the tongue is visible, `np.gradient` is taken on x and y separately (a run of length 1 gets velocity 0), and the speed is `sqrt(vx² + vy²)`; velocity is in pixels per 10 ms bin, and no baseline-drift subtraction is applied (the MATLAB applies it only to non-tongue features). Bins where the tongue was not detected stay NaN — unlike the MATLAB, which sets tongue velocity to 0 there — so they can become the "not visible" class. There is no cross-camera normalisation because only one view is used.

ii.
```python
    else:
        # tongue: never filled, so the gradient is taken within each
        # contiguous stretch of frames in which the tongue was visible
        vx = np.full(NT, np.nan)
        vy = np.full(NT, np.nan)
        idx = np.flatnonzero(visible)
        if idx.size:
            splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for seg in splits:
                if seg.size == 1:
                    vx[seg] = 0.0
                    vy[seg] = 0.0
                else:
                    vx[seg] = np.gradient(x[seg])
                    vy[seg] = np.gradient(y[seg])
    return np.sqrt(vx ** 2 + vy ** 2), visible
```
```python
    for trial in range(n):
        x, y = pos[TONGUE_FEATURES[0]][0][trial], pos[TONGUE_FEATURES[0]][1][trial]
        s, v = speed_from_position(x, y, fill=False)
        tongue_speed[trial] = s
        tongue_vis[trial] = v & has_video[tongue_view][trial]
```

iii. "Kinematics follow `findPosition.m`/`findVelocity.m` — side-camera `tongue` (never gap-filled, so DLC non-detection *is* 'not visible')." The AI read both MATLAB functions in full (step 49) before porting them.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session: the threshold is the median (50th percentile) of the tongue speed over all bins of all **kept** trials where the tongue is visible and the speed is finite. Bins at or above it get 1, below it 0, and every bin where the tongue was not detected (or the trial has no usable video) gets 2. If a session had no visible bins at all the threshold is `+inf`. Result over the dataset: 3.9 % / 3.9 % / 92.1 %.

ii.
```python
    def median_of(values, visible):
        vals = values[keep][visible[keep]]
        vals = vals[~np.isnan(vals)]
        return np.median(vals) if vals.size else np.inf

    thr_tongue = median_of(tongue_speed, tongue_vis)
    ...
    tongue_cls = discretize(tongue_speed, tongue_vis, thr_tongue)
```
```python
def discretize(values, visible, threshold):
    """0 = below the session median, 1 = at/above it, 2 = not visible."""
    out = np.full(values.shape, 2, dtype=np.int8)
    ok = visible & ~np.isnan(values)
    out[ok] = (values[ok] >= threshold).astype(np.int8)
    return out
```

iii. "Each is split at its own session's 50th percentile over visible timepoints" — directly the decoder spec's per-session 50th-percentile rule, with class 2 reserved for "not visible".

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the behaviour clock, so a per-session offset is computed once as `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` (a literal port of `findVideoOffset.m`). Per trial, frame times become `frameTimes − vidshift − goCue[trial]`; if the resulting span does not overlap [−2.5, 2.5] the trial is marked as having no video for that view. Positions are then interpolated at the same `TAXIS` bin centres used for the spikes, so the two streams share one axis. The AI's own check (step 64) showed video starting at −2.47 s from the go cue after correction, i.e. just inside the window.

ii.
```python
def video_shift(f, obj):
    """Port of `funcs/findVideoOffset.m`."""
    bit_start = _mode(_vec(obj['bp']['ev'], 'bitStart'))
    fs = _vec(obj['sglx'], 'fs')[0]
    vid_file_offset = _mode(_vec(obj['sglx']['bitcode'], 'bitstart')) / fs
    return vid_file_offset - bit_start
```
```python
            t_rel = ft - vidshift - align_times[trial]
            if t_rel[-1] < TMIN or t_rel[0] > TMAX:
                continue          # video does not overlap the analysis window
            has_video[view][trial] = True
            ...
                pos[(view, feat)][0][trial] = interp_nan(TAXIS, t_rel, ts[fi, 0])
                pos[(view, feat)][1][trial] = interp_nan(TAXIS, t_rel, ts[fi, 1])
```

iii. "Verified alignment empirically: tongue visibility, motion energy and mean firing rate all jump immediately after t=0." The offset formula is `findVideoOffset.m`, which the AI read in step 47.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` tracking, bottom camera (view 1), **both** paw features: `top_paw` and `bottom_paw` (x, y from `ts`, plus that view's `frameTimes`).

ii.
```python
PAW_FEATURES = [(1, 'top_paw'), (1, 'bottom_paw')]
...
    paw_speed_all = np.full((len(PAW_FEATURES), n, NT), np.nan)
    paw_vis_all = np.zeros((len(PAW_FEATURES), n, NT), dtype=bool)
    for i, key in enumerate(PAW_FEATURES):
        for trial in range(n):
            s, v = speed_from_position(pos[key][0][trial], pos[key][1][trial], fill=True)
```

iii. "bottom-camera `top_paw`+`bottom_paw` averaged over whichever paw is detected (class 2 only when neither is)." Both are in the authors' `params.traj_features` for view 2, and the AI measured per-session NaN fractions for both (step 68), which show that which paw is reliable varies by session (e.g. `JEB13_2022-09-21`: `top_paw` 83 % NaN, `bottom_paw` 5 % NaN; `JEB15_2022-07-26`: `top_paw` 0.8 %, `bottom_paw` 93 %), while both are simultaneously missing on only 0–18 % of frames.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Per paw, following `findPosition.m` / `findVelocity.m` for non-tongue features: x and y are interpolated onto `TAXIS`, gaps are filled with the nearest valid sample (`fillmissing(...,'nearest')`), `np.gradient` gives vx and vy, and the per-trial median frame-to-frame drift is subtracted from each before the speed `sqrt(vx² + vy²)` is taken. Visibility is recorded **before** filling, so filled bins are still marked not-visible. The two paws are then averaged bin-by-bin over whichever paw was actually detected (`np.nanmean` of the visibility-masked speeds), and a bin counts as visible if either paw was detected. Units are pixels per 10 ms bin; no normalisation.

ii.
```python
def speed_from_position(x, y, fill):
    visible = ~np.isnan(x)
    if fill:
        # paws: `findPosition.m` fills missing samples with the nearest value
        xf, yf = fill_nearest(x), fill_nearest(y)
        if np.all(np.isnan(xf)):
            return np.full(NT, np.nan), visible
        vx, vy = np.gradient(xf), np.gradient(yf)
        vx = vx - np.nanmedian(np.diff(xf))
        vy = vy - np.nanmedian(np.diff(yf))
```
```python
    paw_vis = paw_vis_all.any(axis=0)
    masked = np.where(paw_vis_all, paw_speed_all, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        paw_speed = np.nanmean(masked, axis=0)
```

iii. "Kinematics follow `findPosition.m`/`findVelocity.m` … `top_paw`+`bottom_paw` averaged over whichever paw is detected (class 2 only when neither is)." The final report also notes that paw velocity "is a median split of an unsmoothed 100 Hz DLC derivative, which the paper also leaves unsmoothed."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: the per-session median of the averaged paw speed over visible, finite bins of kept trials; 0 below, 1 at/above, 2 where neither paw was detected or the trial has no video. Result over the dataset: 47.8 % / 47.8 % / 4.4 %.

ii.
```python
    thr_paw = median_of(paw_speed, paw_vis)
    ...
    paw_cls = discretize(paw_speed, paw_vis, thr_paw)
```

iii. Same as 7-c — the decoder spec's per-session 50th-percentile rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as the tongue, but with the bottom camera's own `frameTimes`: `frameTimes − vidshift − goCue[trial]`, overlap check against the window, then interpolation onto the shared `TAXIS`. Each view's frame times are read separately, so the two cameras do not have to agree on frame count.

ii.
```python
    for trial in range(ntrials):
        for view, (wanted, ft_refs, ts_refs) in view_data.items():
            ft = np.array(f[ft_refs[trial]]).flatten()
            if ft.size == 0 or np.all(np.isnan(ft)):
                continue
            t_rel = ft - vidshift - align_times[trial]
```

iii. Same session-wide offset as every other camera stream (`findVideoOffset.m`); no separate discussion.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside the data structure, read with `scipy.io.loadmat`: `me.data` is a cell array with one trace per trial, one value per **side-camera** frame. If `me.data` is itself a struct it is unwrapped one more level. `obj.me` (present in only some sessions) is not used. The side camera's `frameTimes` (stored during the video pass) supply the time base.

ii.
```python
def load_motion_energy(anm, date, align_times, frame_times, has_video, ntrials):
    """Port of `DataLoadingScripts/loadMotionEnergy.m`."""
    md = sio.loadmat(os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat'))
    data = md['me'][0, 0]['data']
    if data.dtype != object:                      # me.data is itself a struct
        data = data[0, 0]['data']
    data = data.flatten()
```

iii. Cited in the header as a port of `loadMotionEnergy.m`, whose `if isstruct(me.data), me.data = me.data.data; end` guard is the source of the unwrapping step. The AI probed the three on-disk layouts first (steps 70–75).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling — the value is already one number per frame. The trace is linearly interpolated from the aligned frame times onto `TAXIS`, then remaining NaNs are filled with the nearest value, exactly as `loadMotionEnergy.m` does ("fill nans with nearest value (there are some nans at the start of each trial)"). Trials with no usable video, or whose motion-energy length does not match the side camera's frame count, stay all-NaN and become class 2. Because of the nearest-fill, essentially every bin of every video-covered trial ends up with a value, so the "no video" class is nearly empty (0.013 % of bins).

ii.
```python
    me = np.full((ntrials, NT), np.nan)
    for trial in range(ntrials):
        if not has_video[trial]:
            continue
        y = np.asarray(data[trial]).flatten().astype(float)
        t_rel = frame_times[trial]
        if y.size != t_rel.size:
            continue
        me[trial] = fill_nearest(interp_nan(TAXIS, t_rel, y))
    return me
```
```python
    me_vis = has_video[0][:, None] & ~np.isnan(me)
```

iii. "Motion energy via `loadMotionEnergy.m`" — that function does `interp1(frameTimes - vidshift - alignTimes, me.data{trix}, taxis)` followed by `fillmissing(me.data,'nearest')`, which is what the port reproduces.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same rule again: the per-session median over visible, finite bins of kept trials; 0 below, 1 at/above, 2 (labelled `no_video`) where the trial had no usable video. Result: 50.0 % / 50.0 % / 0.013 %. The `me.moveThresh` field that the authors ship with the file is deliberately not used, since the spec asks for a 50th-percentile split.

ii.
```python
    thr_me = median_of(me, me_vis)
    me_cls = discretize(me, me_vis, thr_me)
```
with `'output_values'[5] = ['below_median', 'above_median', 'no_video']`.

iii. Same as 7-c; the decoder spec's per-session 50th-percentile rule overrides the paper's own movement threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It reuses the side camera's already-corrected frame times (`frameTimes − vidshift − goCue[trial]`, cached during `load_video_traces`) and interpolates onto the same `TAXIS`. Motion energy is only accepted when its length equals the number of side-camera frames of that trial.

ii.
```python
            if view == 0:
                frame_times[trial] = t_rel
```
```python
        t_rel = frame_times[trial]
        if y.size != t_rel.size:
            continue
        me[trial] = fill_nearest(interp_nan(TAXIS, t_rel, y))
```

iii. `loadMotionEnergy.m` uses `obj.traj{1}(trix).frameTimes` — the side camera — for exactly this purpose.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Every case is handled by keeping the trial and marking the gap, never by dropping a trial. (1) **No/NaN frame times**: the trial is skipped for that view, `has_video` stays False, and the tongue/paw come out as 500 "not visible" bins (this affected 1 trial in the whole 25-session dataset). (2) **Video not overlapping the window**: same treatment. (3) **Untracked frames**: DLC leaves x/y NaN below its likelihood threshold; `np.interp` propagates those NaNs, the visibility mask is taken from them, and the corresponding bins become class 2 — for the paws, nearest-fill is applied to the positions (as the MATLAB does) but visibility is recorded before filling, so filled bins are still class 2. (4) **Motion-energy length mismatch** with the side camera's frames: the trial's motion energy is left NaN → class 2. (5) **Missing/non-string cluster quality labels** become `''` and the unit is kept. (6) **Empty spike lists** and spikes outside the window fall out of the `inwin` mask. (7) Divide/invalid warnings from all-NaN slices are suppressed explicitly rather than allowed to abort. Nothing is interpolated across a missing camera, and no trial is discarded for missing video, so the neural and behavioural data of those trials are preserved.

ii.
```python
            ft = np.array(f[ft_refs[trial]]).flatten()
            if ft.size == 0 or np.all(np.isnan(ft)):
                continue
```
```python
def fill_nearest(x):
    """MATLAB's fillmissing(x,'nearest') for a 1-D array."""
    x = np.asarray(x, dtype=float)
    good = ~np.isnan(x)
    if not good.any():
        return x
```
```python
def _char(dataset):
    """MATLAB char array -> python str."""
    arr = np.array(dataset).flatten()
    if arr.dtype in (np.uint16, np.uint8):
        return ''.join(chr(c) for c in arr)
    return ''
```
```python
        if np.all(np.isnan(xf)):
            return np.full(NT, np.nan), visible
```

iii. The "not visible" third class is what makes this possible: the AI notes that with the tongue "never gap-filled, … DLC non-detection *is* 'not visible'". The `n_trials_without_video` counter in `session_info` (total 1) is the AI's own audit of case (1). Nearest-filling of paw positions and motion energy is taken from `findPosition.m` and `loadMotionEnergy.m` rather than invented.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the HDF5 files. A profile of one two-probe session (2.5 s total) attributes ~1.33 s to `h5py` `read_direct` calls: `load_video_traces` 1.31 s cumulative (it dereferences `frameTimes` and `ts` once per trial per view — 321 trials × 2 views) and `load_spikes` 0.96 s cumulative (one `trial` and one `trialtm` array per cluster, ~1,000 clusters before curation). The only substantial compute is `my_smooth`, 0.54 s, of which 0.39 s is the per-trial `np.convolve`. The whole 25-session conversion runs in 53 s and writes a 0.99 GB pickle.

ii.
```python
            trials = np.array(f[trial_refs[iclu]]).flatten().astype(int) - 1
            times = np.array(f[tm_refs[iclu]]).flatten()
```
```python
            ft = np.array(f[ft_refs[trial]]).flatten()
            ...
            ts = np.array(f[ts_refs[trial]])              # (nfeat, 3, nframes)
```

iii. Not discussed in the trajectory; the AI measured only wall-clock time per session (2.2 s in step 94) and for the full run (52.7 s in step 96), and judged that fast enough not to optimise further.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Four. (1) The per-trial `np.convolve` loop inside `my_smooth` — smoothing a `(500, n_trials)` block one column at a time; `scipy.ndimage.convolve1d` or an FFT convolution along the axis would do it in one call (0.39 s of 2.5 s per session). (2) The per-cluster spike-binning loop, which allocates a `(n_trials, 500)` array and calls `np.add.at` for each of ~1,000 clusters; all spikes of a probe could be binned in a single 3-D `histogramdd`/`bincount` over (cluster, trial, bin). (3) The `for trial in range(n)` loops in `load_video_traces`, `speed_from_position` and `load_motion_energy` — these are harder to vectorise because each trial has a different number of frames, but the three separate passes could at least be fused into one. (4) `_mode` is recomputed from scratch per call. Note that `np.add.at` is itself the slow unbuffered path; `np.bincount` on a flat index would be markedly faster.

ii.
```python
    out = np.empty_like(padded)
    for j in range(padded.shape[1]):
        out[:, j] = np.convolve(padded[:, j], kern, mode='same')
```
```python
        for iclu in range(quality_refs.size):
            ...
            counts = np.zeros((keep_trials.size, NT))
            pos = trial_pos[trials]
            inwin = (pos >= 0) & (times >= TMIN) & (times < TMAX)
            if inwin.any():
                bin_idx = np.floor((times[inwin] - TMIN) / DT).astype(int)
                np.add.at(counts, (pos[inwin], bin_idx), 1.0)
```

iii. Not discussed. The AI did partially vectorise: spikes of one cluster are binned for all trials at once rather than trial by trial, and `fill_nearest` was rewritten (step 86) from an O(n²) nearest-index search to an O(n) forward/backward accumulate specifically for speed.

## 11-c. What processing does the code repeat multiple times?

i. Little, but not nothing. The video offset is computed once per session and the `featNames`/`frameTimes`/`ts` reference lists once per view, both correctly hoisted out of the trial loop. What is repeated: the `gausswin(15)` kernel is rebuilt on each of the ~1,000 `my_smooth` calls per session; the per-trial loop over the video is walked three times (positions, then tongue speed, then each paw's speed, then motion energy) instead of once; `np.diff(xf)`/`np.nanmedian` are recomputed inside `speed_from_position` for every trial and feature; and both `_mode` calls in `video_shift` re-sort their inputs. None of these is on the critical path.

ii.
```python
def my_smooth(x, n=SMOOTH_N):
    ...
    kern = gausswin(n)
    kern[:n // 2] = 0.0          # causal
    kern = kern / kern.sum()
```
```python
    view_data = {}
    for view in views:
        tv = f[traj[view, 0]]
        names = _cellstr(f, f[np.array(tv['featNames']).flatten()[0]])
        ...   # hoisted out of the per-trial loop
```

iii. Not discussed.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three things. (1) **Kinematics are computed for every trial, then thrown away for the excluded ones**: `load_video_traces`, `speed_from_position` and `load_motion_energy` all loop over `range(n)` — all `Ntrials` — while only `keep` (7,426 of 8,260 trials, i.e. ~10 % wasted) is written out. By contrast the neural path bins only the kept trials. (2) **Intermediate quantities that never reach the output**: interpolated x/y positions for three features are materialised as full `(n_trials, 500)` arrays and discarded once the speed is taken; firing rates are computed and smoothed for every quality-passing cluster and only afterwards filtered by the 1 Hz criterion, so the smoothing work done on the sub-1 Hz majority of clusters is thrown away; `beh['no']` and `beh['L']`-as-a-separate-array are loaded though `no` is never used. (3) **Metadata-only computation**: `qualities`, `nsingle_units` and `quality_counts` are tracked purely for `session_info` and play no role in the converted data.

ii.
```python
    for i, key in enumerate(PAW_FEATURES):
        for trial in range(n):                       # all trials, not just `keep`
            s, v = speed_from_position(pos[key][0][trial], pos[key][1][trial], fill=True)
```
```python
            rate = my_smooth((counts / DT).T).T     # smoothed before the 1 Hz filter
            rates.append(rate)
            qualities.append(qual)
...
        mean_fr = rates.mean(axis=(1, 2)) if rates.size else np.zeros(0)
        use = mean_fr > LOW_FR
        rates = rates[use]
```
```python
        'nsingle_units': int(sum(q.lower() in ('excellent', 'great', 'good')
                                 for q in qualities)),
        ...
        'quality_counts': dict(Counter(q for q in qualities)),
```

iii. Not discussed; the AI did remove one instance of it when tidying up (step 115 deleted the unused `sample` and `delay` event fields from `load_behavior`), which suggests the leftover unused work was an oversight rather than a decision.
