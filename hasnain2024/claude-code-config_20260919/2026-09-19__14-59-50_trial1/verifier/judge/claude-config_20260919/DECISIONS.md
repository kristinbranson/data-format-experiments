# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** load all of the ephys data. It hard-codes a list of **12 sessions** — the
"two-context" (DR + WC) sessions that live in `/app/data/Ephys_Behavior` — together with the ALM
probe number for each, transcribed from the authors' `DataLoadingScripts/Recording and
video/load<ANM>_ALMVideo.m` (and matching the session lists of `Scripts/Figure 8` and
`Scripts/EDFigure 2a-left`). The other 13 fixed-delay sessions and all 19/22 randomized-delay
sessions in `/app/data/RandomizedDelay_Ephys_Behavior` are deliberately not loaded. Each session is
one `data_structure_<ANM>_<DATE>.mat` opened with `h5py` (all 12 are MATLAB v7.3), plus a companion
`motionEnergy_<ANM>_<DATE>.mat` opened with `scipy.io.loadmat`. Per session the AI reads `obj.bp`
(per-trial task table, truncated to `Ntrials`), `obj.clu{probe}` (spike clusters), `obj.traj{1..2}`
(DeepLabCut side/bottom camera) and `obj.sglx` (bitcode/sample rate for the video clock).

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'

# The 12 two-context (DR + WC) sessions, with the ALM probe number taken from
# /app/code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m
SESSIONS = [
    ('JEB6',  '2021-04-18', 2),
    ('JEB7',  '2021-04-29', 1),
    ...
    ('JEB19', '2023-04-21', 1),
]

def session_paths(anm, date):
    return (os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat'),
            os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat'))
```
```python
f = h5py.File(data_path, 'r')
o = f['obj']
bp = load_bpod(f, o)
```
```python
def load_bpod(f, o):
    bp = o['bp']
    n = int(np.array(bp['Ntrials'])[0, 0])
    def g(name):
        return np.array(bp[name]).flatten()[:n]
    out = dict(Ntrials=n, hit=g('hit').astype(bool), miss=g('miss').astype(bool),
               no=g('no').astype(bool), R=g('R').astype(bool), L=g('L').astype(bool),
               early=g('early').astype(bool), autowater=g('autowater').astype(bool))
    try:
        out['stim'] = np.array(bp['stim']['enable']).flatten()[:n].astype(bool)
    except (KeyError, TypeError):
        out['stim'] = np.zeros(n, bool)
```

iii. From CONVERSION_NOTES Step 5, Key Decision 1: *"Use only the 12 two-context sessions …
'behavioural context (WC, DR)' is a required decoder output, and these are the only ephys sessions
containing both contexts. They are exactly the sessions loaded by `Scripts/Figure 8` and
`Scripts/EDFigure 2a-left`, and reproduce the paper's 522-unit / 214-single-unit figure. The 13
DR-only fixed-delay sessions and the 19 randomized-delay sessions have no WC trials, so adding them
would add a constant, undecodable context label."* The AI verified this by scanning every ephys
file in `/app/data` and tabulating the `autowater` fraction per session (trajectory steps 40, 42).
It also confirmed its session/probe list against the paper's reported statistics for the
two-context paradigm (12 sessions, 522 units, 214 single units) — it obtains 12 sessions,
521 units and exactly 214 single units.

## 1-b. How are the data split into subjects (mice)?

i. The subject is the animal id that is the first element of each `SESSIONS` tuple (equivalently the
`<ANM>` part of the filename). `subjects` is built in first-appearance order as sessions are
processed, and `subject_idx` is that session's index into the list. The result is 7 subjects
(JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19) over 12 sessions, i.e. 1–4 sessions per mouse.

ii.
```python
for k, (anm, date, probe) in enumerate(sessions):
    neural, inp, out, info, extras = convert_session(anm, date, probe)
    ...
    if anm not in data['subjects']:
        data['subjects'].append(anm)
    data['subject_idx'].append(data['subjects'].index(anm))
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The animal is taken from the filename / loader-script list rather than from inside the file
because `obj.meta`/`obj.ex` is missing from several sessions (CONVERSION_NOTES Step 10 Check 5:
*"Sessions without `obj.ex` (EKH3, JEB7, JGR2, JGR3): the probe number comes from the reference
loader scripts rather than the data file."*). The AI also documented the discrepancy that the paper
says "six mice" while the authors' own loader list for these 12 sessions contains 7 animals, and
noted the same off-by-one for the 25 fixed-delay sessions (paper: nine mice; loaders: ten). It
chose to follow the code/data and report 7.

## 1-c. How are the data split into sessions?

i. One entry in `SESSIONS` = one `data_structure_*.mat` file = one element of `neural`, `input`,
`output`, `brain_region_idx` and `metadata['session_info']`. Sessions are processed independently
inside `convert_session`, and all session-level quantities (video offset, unit set, the three
movement thresholds) are computed per session. Two session-level exclusion rules are applied after
conversion: a session is skipped if it has fewer than 2 usable trials, or fewer than 10 units (the
paper's inclusion criterion). Neither rule fires: all 12 sessions are exported.

ii.
```python
for k, (anm, date, probe) in enumerate(sessions):
    neural, inp, out, info, extras = convert_session(anm, date, probe)
    if len(neural) < 2:
        print(f'  SKIPPING {anm} {date}: fewer than 2 usable trials')
        continue
    if info['n_units'] < 10:
        print(f'  SKIPPING {anm} {date}: fewer than 10 units')
        continue
    data['neural'].append(neural)
    data['input'].append(inp)
    data['output'].append(out)
```

iii. *"Session curation rules: only the 12 two-context sessions, because 'behavioural context (WC,
DR)' is a required decoder output and only these sessions contain both contexts. All 12 have ≥10
units."* (Step 3 / Step 5). The ≥10-unit rule is quoted from the Methods (*"Recording sessions were
included for analysis only if they had at least 10 units"*). The AI also checked (Step 10, Check H)
the Methods' behavioural inclusion criterion (≥40 correct DR trials/direction, ≥20 correct WC
trials/direction) and found that 5 of the 12 sessions fall slightly short, but chose not to apply it
because that criterion is stated for behavioural analyses, is implemented in
`utils/UseInclusionCritera.m` which *"is not called by any figure script"*, and dropping those
sessions would break the match with the paper's 522-unit / 214-single-unit counts.

## 1-d. How are the data split into trials?

i. A trial is one row of the `obj.bp` per-trial table. Every `bp` field is flattened and truncated
to `bp.Ntrials` (some fields are stored longer). The go cue of trial *i* is `bp.ev.goCue[i]`. Spike
times carry their own 1-based trial index (`clu.trial`), and the DeepLabCut trajectories and the
motion-energy file are already stored as one cell per trial, so no trial boundaries have to be
reconstructed. Each kept trial becomes one `(n_units, 500)` neural array, one `(1, 500)` input array
and one `(6, 500)` output array.

ii.
```python
n = int(np.array(bp['Ntrials'])[0, 0])
def g(name):
    return np.array(bp[name]).flatten()[:n]
```
```python
ok = (tr >= 1) & (tr <= ntrials) & ~np.isnan(tm)     # clu.trial is 1-based
tm, tr = tm[ok], tr[ok]
aligned = tm - gocue[tr - 1]
```
```python
for j, tr in enumerate(keep_idx):
    neural_trials.append(np.ascontiguousarray(neural[:, :, tr].T))   # (nunits, T)
    input_trials.append(input_trial.copy())
    out = np.empty((6, t), dtype=np.int64)
    ...
```

iii. The Bpod table defines trials directly. The AI documented the 1-based indexing edge case
(Step 10, Check 5: *"Trial indices in `obj.clu.trial` are 1-based; spikes with out-of-range or NaN
trial indices are discarded before binning"*) and stores `trial_ids = keep_idx + 1` in
`metadata['session_info']` so every exported trial can be traced back to the raw file — which is
what makes its independent sanity checks (Step 10, Check 2) possible.

## 1-e. How are trials filtered based on quality controls?

i. Four masks, ANDed: the trial must be scored (`hit | miss | no`), must not be an early-lick trial
(`bp.early`), must not be a photoinactivation trial (`bp.stim.enable`), and must have a non-NaN go
cue. After the kinematics are extracted a fifth mask is applied: trials whose video is missing
(`frameTimes` empty, or containing any NaN) are dropped. Ignore trials (`bp.no`) are **kept**, a
documented deviation from the paper. Over the 12 sessions this keeps 3,115 of 3,626 trials
(389 early, 135 stim, 14 counted in both, 1 no-video).

ii.
```python
# every params.condition in the reference excludes early-lick and photostim trials
keep = (bp['hit'] | bp['miss'] | bp['no']) & ~bp['early'] & ~bp['stim'] & ~np.isnan(bp['goCue'])
```
```python
# Trials with no usable video are dropped: three of the six decoder outputs
# (tongue velocity, paw velocity, motion energy) are undefined for them, and the
# reference funcs/kinematics/findPosition.m likewise skips trials whose video is
# missing (NdroppedFrames = NaN).
n_dropped_no_video = int((keep & ~kin['has_video']).sum())
keep = keep & kin['has_video']
keep_idx = np.nonzero(keep)[0]
```
```python
def trial_frame_times(f, traj_view, trial, vidshift, gocue_t):
    ft = np.array(f[traj_view['frameTimes'][trial, 0]]).flatten()
    if ft.size == 0 or np.all(np.isnan(ft)):
        return None
    if np.isnan(ft).any():
        return None
    return ft - vidshift - gocue_t
```

iii. *"Trial curation: drop early-lick (`bp.early`) and photoinactivation (`bp.stim.enable`) trials,
which every `params.condition` in the reference excludes. **Keep ignore (`bp.no`) trials** even
though the paper excludes them — the decoder task requires 'ignore' as an outcome class and 'none'
as a lick-direction class. Also drop the single trial whose video is missing (JEB19 2023-04-19),
because three of the six outputs are undefined for it and `findPosition.m` skips such trials too."*
(Step 5, Decision 7). The AI verified in Step 10 (Check D) that no early or stim trial appears in the
output and that the kept-trial count equals the expected count for all 12 sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}`, the spike-sorted clusters of the single ALM probe named for that session in
`load<ANM>_ALMVideo.m`. Per cluster the AI reads `trialtm` (spike time within its trial, on the Bpod
clock), `trial` (1-based trial index) and `quality` (the manual curation label). `obj.bp.ev.goCue`
is the second input, since it defines the alignment.

ii.
```python
clu = f[o['clu'][probe - 1, 0]]
quality = [matstr(f, r).strip() for r in np.array(clu['quality']).flatten()]
qual_keep = [i for i, q in enumerate(quality) if q not in BAD_QUALITY]
rate = bin_spikes(f, clu, qual_keep, bp['goCue'], n, edges)
```
```python
for k, ci in enumerate(keep_idx):
    tm = np.array(f[clu['trialtm'][ci, 0]]).flatten()
    tr = np.array(f[clu['trial'][ci, 0]]).flatten().astype(int)
```

iii. Mapping table, Step 5: *"`obj.clu{probe}(c).trialtm`, `.trial`, `obj.bp.ev.goCue` →
`neural[s][t]`"*, following `alignSpikes.m` + `getSeq.m`. Only one probe is used because *"All 12
two-context sessions are single-ALM-probe sessions. Verified against `obj.ex.probe.loc` where
present (`R ALM` for JEB6 probe 2, `L ALM` for JEB19 probe 1)"* (Step 5, Decision 2).

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 10 ms bins over [−2.5, 2.5] s from the go cue (one
`np.histogram2d` per cluster over aligned-time × trial), divided by `dt` to give spikes/s, and then
smoothed along time with a **causal** Gaussian: a direct port of `utils/mySmooth.m` — `gausswin(15)`
with MATLAB's default α = 2.5, the first `floor(15/2) = 7` taps zeroed, renormalised to sum to 1,
convolved `'same'`, with the `'reflect'` boundary condition (prepend the first 15 samples, then
trim). Nothing else is done: no z-scoring, no baseline subtraction. The stored values are
single-trial firing rates in spikes/s (`float32`), the reference pipeline's `obj.trialdat`.

ii.
```python
def gausswin(n, alpha=2.5):
    k = np.arange(n) - (n - 1) / 2.0
    return np.exp(-0.5 * (alpha * k / ((n - 1) / 2.0)) ** 2)

def make_causal_kernel(n=SMOOTH):
    """Reference kernel from utils/mySmooth.m: gausswin(n) with the first floor(n/2) taps
    zeroed (making it causal) and renormalised to sum to 1."""
    k = gausswin(n)
    k[:n // 2] = 0.0
    return k / k.sum()
```
```python
def my_smooth(x, kernel=_KERNEL, bctype=BCTYPE):
    n = len(kernel)
    ...
    if bctype == 'reflect':
        xf = np.concatenate([x[:n], x], axis=0)
        trim = n
    ...
    for j in range(xf.shape[1]):
        out[:, j] = np.convolve(xf[:, j], kernel, mode='same')
    return out[trim:]
```
```python
h, _, _ = np.histogram2d(aligned[inwin], tr[inwin], bins=[edges, trial_bins])
rate[:, k, :] = h / DT
...
rate_s = my_smooth(rate.reshape(t, -1)).reshape(rate.shape).astype(np.float32)
```

iii. *"Neural data = smoothed single-trial firing rate (`obj.trialdat`), not raw spike counts: this
is the quantity every reference analysis uses, and it is what the reference decoding scripts feed to
their classifiers."* (Step 5, Decision 5) and *"The reference never z-scores or normalises
`trialdat` at load time … the decoder here does its own PCA/standardisation"* (Step 1). The AI
verified the port numerically: Step 10 Check C re-derives the exported rate for 9 random
(session, trial, unit) triples straight from the raw `.mat` with an independent implementation
(`np.allclose`, max abs diff < 4e-6) and checks that `sum(rate)*dt` recovers the raw spike count in
the window to within 2%.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, exactly as in the reference MATLAB. (1) The manual curation label `clu.quality` is
`strtrim`-ed and clusters labelled `garbage`, `gabrga`, `noisy` or `real?` are dropped
(case-sensitive `ismember`, matching `findClusters.m` with `params.quality = {'all'}`); everything
else — including `multi`, `poor`, `Noisy` with a capital N, and unlabelled clusters — is kept.
(2) A port of `removeLowFRClusters.m`: the condition-averaged PSTH is formed by averaging the
already-smoothed single-trial rates over each of the 7 `params.condition` sets of
`Scripts/Figure 8/Figure8a_thru_c.m` (`omitnan`), then averaged over conditions and over time, and
units with mean FR ≤ 1 Hz are dropped. Result: 2,287 raw clusters → 529 after quality → **521
units** (214 of which are single units), 27–67 per session.

ii.
```python
BAD_QUALITY = ('garbage', 'gabrga', 'noisy', 'real?')   # findClusters.m, params.quality={'all'}
LOW_FR = 1.0                                            # Hz, params.lowFR
...
quality = [matstr(f, r).strip() for r in np.array(clu['quality']).flatten()]
qual_keep = [i for i, q in enumerate(quality) if q not in BAD_QUALITY]
```
```python
def low_fr_mask(rate_smooth, conditions):
    """Port of removeLowFRClusters.m: mean over conditions (omitnan) then over time."""
    psth = np.full((rate_smooth.shape[0], rate_smooth.shape[1], len(conditions)), np.nan)
    for j, c in enumerate(conditions):
        if c.sum() == 0:
            continue
        psth[:, :, j] = rate_smooth[:, :, c].mean(axis=2)
    mean_fr = np.nanmean(np.nanmean(psth, axis=2), axis=0)
    return mean_fr > LOW_FR, mean_fr

fr_mask, mean_fr = low_fr_mask(rate_s, fig8_conditions(bp))
unit_idx = np.nonzero(fr_mask)[0]
neural = rate_s[:, unit_idx, :]
```

iii. *"Neuron curation: quality filter + >1 Hz condition-averaged mean rate, exactly as in
`findClusters.m` / `removeLowFRClusters.m`, using the condition list of `Figure8a_thru_c.m` (the
script that analyses this very data set)"* (Step 5, Decision 6). The paper's rule is quoted in
Step 3: *"All units with firing rates exceeding 1 Hz were included in all other analyses."* The AI
used the paper's unit counts as its main sanity check: 521 units vs the reported 522 (it showed the
one-unit difference is a unit sitting on the 1 Hz boundary — with Figure 8's `tmin = −3` its code
gives exactly 522), and **214 vs 214** single units, which it used to pin down the
quality→single-unit mapping (`excellent`/`great`/`good`/`fair`). Step 10, Check 5 documents the
mixed-case/`char([0 0])` quality strings and why case-sensitive matching after stripping reproduces
the reference counts.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction. `clu.trialtm` is already on the Bpod clock and relative to its own trial's
start, and `bp.ev.goCue` is on the same clock, so the aligned spike time is
`trialtm − goCue[trial]`. Spikes are then histogrammed into half-open bins spanning [−2.5, 2.5) s;
spikes outside the window are masked out explicitly so that the last bin edge behaves like MATLAB's
`histc(...)(1:end-1)`.

ii.
```python
ok = (tr >= 1) & (tr <= ntrials) & ~np.isnan(tm)
tm, tr = tm[ok], tr[ok]
aligned = tm - gocue[tr - 1]
# half-open bins, exactly like MATLAB histc(...)(1:end-1)
inwin = (aligned >= edges[0]) & (aligned < edges[-1])
h, _, _ = np.histogram2d(aligned[inwin], tr[inwin], bins=[edges, trial_bins])
```

iii. `alignSpikes.m`: *"`trialtm_aligned = trialtm - ev.(alignEvent)(trial)`; for us
`alignEvent = 'goCue'`"* (Step 1 table), which is also what every reference figure script uses
(`params.alignEvent = 'goCue'`). The AI notes that on WC trials this Bpod event marks the water
presentation, the behavioural analogue of the go cue (Step 5, Decision 3). It verified alignment by
plotting the go-cue-aligned spike raster and the lick raster on the same axis (panels 1 and 5 of
`processing_<session>.png`): lickport contacts begin exactly at t = 0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins**, 500 bins spanning [−2.5, +2.5] s around the go cue; the time axis is the bin
centres, −2.495 … 2.495 s, computed exactly as `getSeq.m` does (`obj.time = edges + dt/2`, drop the
last). Spikes are binned once directly at this resolution — there is no rebinning of an
intermediate resolution, and the camera streams (400 Hz) are resampled by `interp1` onto this same
axis, so all four streams share one grid.

ii.
```python
TMIN = -2.5          # s relative to the go cue (params.tmin)
TMAX = 2.5           # s relative to the go cue (params.tmax)
DT = 0.01            # s, 10 ms bins (params.dt = 1/100)
...
edges = np.arange(TMIN, TMAX + DT / 2, DT)
taxis = (edges[:-1] + edges[1:]) / 2.0           # == edges + dt/2 (drop last), getSeq.m
t = len(taxis)
```

iii. Step 4 discrepancy table: *"`params.dt` — `1/200` in `getDefaultParams.m`, `1/100` in all
figure scripts (and the tutorial comment wrongly says '5 ms'). Use **10 ms** (`1/100`), the value
actually used for all published analyses."* On the window: *"`tmin = -2.5` is the value in the
tutorial, `getDefaultParams.m` and EDFigure 2; Figures 1/3/8 use `-3`. I chose `-2.5` because the
high-speed video starts at trial onset (~−2.48 s relative to the go cue), so this window is fully
covered by video on essentially every trial, whereas `-3` would require extrapolating ~0.5 s of the
behavioural outputs. Cost: 521 instead of 522 units pass the >1 Hz filter."* (Step 5, Decision 4).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw variable — it is the binning grid itself, defined by the choice of
alignment event (`bp.ev.goCue`), window ([−2.5, 2.5] s) and bin size (10 ms). The input is the
vector of 500 bin centres, identical for every trial and every session, stored as `float32` with
shape `(1, 500)`.

ii.
```python
edges = np.arange(TMIN, TMAX + DT / 2, DT)
taxis = (edges[:-1] + edges[1:]) / 2.0           # == edges + dt/2 (drop last), getSeq.m
...
input_trial = taxis.astype(np.float32)[None, :]      # (1, T), identical for every trial
```

iii. Step 5 mapping table: *"time axis → `input[s][t]` (1, 500), bin centres −2.495 … 2.495 s,
reference `getSeq.m` (`obj.time`), the single decoder input required by the task"*, and Step 5,
Decision 11: *"Single input (`time_from_go_cue`, continuous), as specified by the task."*

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the grid: `np.arange` over the window and taking bin centres. The AI
kept it continuous (in seconds) rather than converting it to a binary event time series, because the
Decoder Task lists it as *"continuous, time-varying"*.

ii.
```python
input_trial = taxis.astype(np.float32)[None, :]
...
input_trials.append(input_trial.copy())
```

iii. N/A — no raw data is involved. Step 10, Check B verified that the exported `input` equals the
10 ms bin-centre grid −2.495 … 2.495 for every sampled trial of every session.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. By construction: `taxis` is the vector of centres of the very bins the spikes are counted into
(`edges`), so input bin *k* and neural bin *k* are the same 10 ms interval relative to the same go
cue. The same `taxis` is also the resampling grid for the three camera-derived outputs.

ii.
```python
edges = np.arange(TMIN, TMAX + DT / 2, DT)
taxis = (edges[:-1] + edges[1:]) / 2.0
...
h, _, _ = np.histogram2d(aligned[inwin], tr[inwin], bins=[edges, trial_bins])   # neural
...
return np.interp(taxis, times[:k], y[:k], left=np.nan, right=np.nan)            # kinematics
```

iii. N/A. Documented in `metadata` as `temporal_alignment_event = 'go cue onset (obj.bp.ev.goCue)…'`
with `off_start = -2.5`, `off_end = 2.5`, `time_bin_size = 10.0` ms.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: the instructed/rewarded side `L` and `R`, and the outcome flags
`hit`, `miss`, `no`. The lick direction itself is not recorded, so it is inferred from the
combination.

ii.
```python
out = dict(..., hit=g('hit').astype(bool), miss=g('miss').astype(bool), no=g('no').astype(bool),
           R=g('R').astype(bool), L=g('L').astype(bool), ...)
```

iii. Step 1 table: *"`getPrevChoice` — **Lick direction**: right ⇔ `(R&hit) | (L&miss)`, left ⇔
`(L&hit) | (R&miss)`, `NaN` for ignore (`no`)."* The AI took the definition straight from the
authors' `funcs/getPrevChoice.m`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A relabelling into three classes: 0 = left (`(L&hit) | (R&miss)`), 1 = right
(`(R&hit) | (L&miss)`), 2 = none (`no`). The array is initialised to −1 and an assertion checks that
no kept trial is left undefined. The per-trial value is then broadcast across all 500 time bins so
that all six outputs live in one `(6, 500)` array.

ii.
```python
lick_dir = np.full(n, -1, dtype=np.int64)
lick_dir[(bp['L'] & bp['hit']) | (bp['R'] & bp['miss'])] = 0      # licked left
lick_dir[(bp['R'] & bp['hit']) | (bp['L'] & bp['miss'])] = 1      # licked right
lick_dir[bp['no']] = 2                                            # no response
...
assert (lick_dir[keep_idx] >= 0).all(), f'{anm} {date}: undefined lick direction'
...
out[0, :] = lick_dir[tr]
```

iii. Follows `getPrevChoice.m`; the `none` class replaces the reference's `NaN` because the Decoder
Task specifies left/right/none. Step 5, Decision 8: *"Per-trial outputs are broadcast across time
rather than stored as 1-D arrays, so that all six outputs share one `(6, T)` array (required because
three of them are time-varying)."* The AI validated this against behaviour (Step 10, Check E): the
side of the **first lickport contact after the go cue** matches the exported `lick_direction` on
644/649 responded trials (99.2%), and all 151 trials labelled `none` really contain no post-go-cue
contact. The 5 disagreements are trials where the animal's first contact was on the opposite port
from the one Bpod scored; the AI kept the reference (`getPrevChoice`) definition.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater` — the flag that marks water-cued trials, where water is
delivered at a random port with no auditory cue.

ii.
```python
autowater=g('autowater').astype(bool)
```

iii. Step 1 table: *"`getBlockNum_AltContextTask` — Context blocks are defined purely by transitions
of `bp.autowater` ⇒ `autowater==1` ⇔ **WC**, `autowater==0` ⇔ **DR**."* The AI cross-checked the
block structure against the Methods (Step 9 consistency table): every session starts with a DR block
and the first WC trial appears at index 91–140 (*"approximately 100 DR trials"* first), with 8–16
blocks per session.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `autowater → 0 (WC)`, otherwise `1 (DR)`, broadcast across time.

ii.
```python
context = np.where(bp['autowater'], 0, 1).astype(np.int64)        # 0 = WC, 1 = DR
...
out[1, :] = context[tr]
```

iii. Codes follow the Decoder Task's ordering (WC, DR). Verified in Step 10, Check D by recomputing
from `obj.bp` for all 12 sessions (`np.array_equal`), and visually in panel 6 of
`processing_<session>.png`, where the exported context label is the exact complement of the raw
`bp.autowater` trace. Converted distribution: WC 0.315 / DR 0.685.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial flags of `obj.bp`: `hit`, `miss` and `no`.

ii.
```python
hit=g('hit').astype(bool), miss=g('miss').astype(bool), no=g('no').astype(bool)
```

iii. Step 1 table: *"`getOutcome` — `hit` = correct, `miss` = incorrect, `no` = ignore (set to NaN
there)."*

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A relabelling into the three classes required by the task: 0 = incorrect (`miss`), 1 = correct
(`hit`), 2 = ignore (`no`), initialised to −1 with an assertion that no kept trial is undefined,
then broadcast across time.

ii.
```python
outcome = np.full(n, -1, dtype=np.int64)
outcome[bp['miss']] = 0
outcome[bp['hit']] = 1
outcome[bp['no']] = 2
assert (outcome[keep_idx] >= 0).all(), f'{anm} {date}: undefined outcome'
...
out[2, :] = outcome[tr]
```

iii. Follows `getOutcome.m`, with `ignore` kept as a real class rather than NaN. This is one of the
AI's explicitly documented deviations from the paper: *"Ignore trials … **Retained**, because the
decoder task defines `ignore` as an outcome class and `none` as a lick-direction class"* (Step 4
table and Step 10, Check 3, deliberate difference 1). Converted distribution: incorrect 0.106 /
correct 0.670 / ignore 0.225, i.e. 86% correct among responded trials, consistent with the paper's
statement that DR performance exceeds 70%.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking of the **side camera only**, `obj.traj{1}`: the `ts` array
(frames × [x, y, confidence] × feature) for the feature named `tongue`, together with that view's
`frameTimes`. `obj.sglx.fs`, `obj.sglx.bitcode.bitstart` and `obj.bp.ev.bitStart` are also needed,
to put the frames on the Bpod clock, and `bp.ev.goCue` to align them. The bottom camera's tongue
markers (`top_tongue` etc.) are not used.

ii.
```python
v1 = f[o['traj'][0, 0]]      # side camera
v2 = f[o['traj'][1, 0]]      # bottom camera
n1 = feature_names(f, v1)
i_tongue = n1.index('tongue')
...
ts1 = np.array(f[v1['ts'][tr, 0]])   # (nfeat, [x,y,conf], nframes)
x = resample(ts1[i_tongue, 0])
y = resample(ts1[i_tongue, 1])
```

iii. Step 5 mapping table: *"`obj.traj{1}(t).ts[:, :, tongue]` → `output[3]` tongue_velocity …
reference `findPosition.m`, `findVelocity.m`"*. The AI's Step 2 notes record that the side view
tracks `tongue` and the bottom view tracks `top_tongue`/`bottom_tongue` etc., and the reference
kinematics functions operate on one `view` at a time; the AI chose the side view's `tongue` feature
and did not document a comparison of the two views' detection rates.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, ported from `findPosition.m` + `findVelocity.m`. (1) The raw x and y traces (which
DeepLabCut already sets to NaN where it does not detect the tongue) are linearly interpolated onto
the 10 ms grid with `np.interp`, NaN outside the frame range; the number of frames is clipped to
`min(len(frameTimes), ts.shape[2])`. (2) Visibility is the non-NaN mask of the interpolated x.
(3) Speed is `hypot` of `np.gradient` of the interpolated x and y — the tongue is *not* smoothed and
*no* baseline-drift term is subtracted, both following the reference's "not for tongue" branches.
(4) The speed array is then discretised (7-c). Note that the gradient is taken on a nearest-filled
copy of the position so that the derivative is defined at the edges of each visible run; the filled
samples themselves are masked out by the visibility mask and never produce a class 0/1.

ii.
```python
def resample(y, times=None):
    times = tt if times is None else times
    k = min(len(times), len(y))
    return np.interp(taxis, times[:k], y[:k], left=np.nan, right=np.nan)

# ---- tongue (side camera). NaN = tongue not detected by DeepLabCut ----
x = resample(ts1[i_tongue, 0])
y = resample(ts1[i_tongue, 1])
vis = ~np.isnan(x)
if vis.any():
    vx = np.gradient(nearest_fill(x))
    vy = np.gradient(nearest_fill(y))
    tongue_speed[:, tr] = np.hypot(vx, vy)
tongue_vis[:, tr] = vis
```

iii. Step 1 table on `findPosition.m`: *"**Tongue is not smoothed and NaNs are not filled**; all
other features `fillmissing('nearest')`"*, and on `findVelocity.m`: *"`gradient()` of the
*interpolated* position; for non-tongue features subtracts the baseline drift `median(diff(pos))`;
tongue NaN velocity → 0"*. This matches the Methods (*"Missing values were filled in with the
nearest available value for all features, except for the tongue"*). Step 7 also records a data
quirk the AI chose not to work around: *"in JEB19 2023-04-20 the last three (WC ignore) trials have
the tongue marker stuck at a fixed position with high DeepLabCut confidence, which pulls that
session's tongue-speed median down to 0.49 (other sessions: 2.5–10). This is genuine DeepLabCut
output and the reference applies no extra confidence filtering, so it was left as is."*

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session: the 50th percentile of the tongue speed is taken over **all visible time points of
all kept trials** of that session; bins below it get class 0, bins at or above it class 1, and every
bin where DeepLabCut did not detect the tongue gets class 2 (`not_visible`). Because the percentile
is computed over the visible points only, classes 0 and 1 are exactly balanced within each session
(each ~4.1% of all bins; 91.9% of bins are `not_visible`).

ii.
```python
def discretize(values, visible, keep_trials):
    """0 = below the session's 50th percentile, 1 = at/above it, 2 = not visible / no video.

    The percentile is computed over the *visible* time points of the kept trials only,
    because 'not visible' is its own category.
    """
    vals = values[:, keep_trials]
    vis = visible[:, keep_trials] & ~np.isnan(vals)
    if vis.sum() == 0:
        return np.full(vals.shape, 2, dtype=np.int64), np.nan
    thresh = np.percentile(vals[vis], 50)
    out = np.full(vals.shape, 2, dtype=np.int64)
    out[vis & (vals < thresh)] = 0
    out[vis & (vals >= thresh)] = 1
    return out, float(thresh)

tongue_cat, tongue_thresh = discretize(kin['tongue_speed'], kin['tongue_vis'], keep_idx)
```

iii. Step 5, Decision 9: *"the 50th percentile is computed **per session over the time points at
which the feature is available**, because 'not visible'/'no video' is a separate class. This makes
classes 0 and 1 exactly balanced within every session."* Verified in Step 10, Check G: *"In every
session and for each of the three movement variables, classes 0 and 1 contain the same number of
samples to within 2% — i.e. the split really is at the session median."*

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera runs on its own clock. The session's offset is computed once from the bitcode pulse
recorded on both clocks — `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` — a direct
port of `funcs/findVideoOffset.m`. Frame times are then `frameTimes − vidshift − goCue[trial]`, and
the x/y traces are `interp1`-ed onto the same 500-bin `taxis` used for the spikes, so bin *k* is the
same interval in both streams. Each camera view uses **its own** `frameTimes`.

ii.
```python
def video_offset(f, o, bp):
    """funcs/findVideoOffset.m: offset between the video clock and the Bpod clock (s)."""
    fs = float(np.array(o['sglx']['fs'])[0, 0])
    bitstart = np.array(o['sglx']['bitcode']['bitstart']).flatten()
    return mode_value(bitstart) / fs - mode_value(bp['bitStart'])

def trial_frame_times(f, traj_view, trial, vidshift, gocue_t):
    ft = np.array(f[traj_view['frameTimes'][trial, 0]]).flatten()
    ...
    return ft - vidshift - gocue_t
...
tt = trial_frame_times(f, v1, tr, vidshift, bp['goCue'][tr])
tt2 = trial_frame_times(f, v2, tr, vidshift, bp['goCue'][tr])
```

iii. Step 4: *"Video offset — `findVideoOffset` (data-driven) vs. 'subtract 0.5 s' in the tutorial;
offset is 0.5 s for 2021 sessions, 0.99 s for JEB19 → use `findVideoOffset` (the tutorial's 0.5 s is
a special case)."* The AI measured 0.490 s for the eight 2021 sessions and 0.990 s for the four
JEB19 sessions, and reproduced `vidshift` exactly in 3 sessions in an independent check (Step 10,
Check F). Panel 7 of `processing_<session>.png` overlays the raw 400 Hz DeepLabCut trace with the
resampled 10 ms trace and shows the tongue becoming visible only after the go cue. A separate edge
case is documented (Step 10, Check 5): *"Cameras have independent frame counts and frame-time
vectors; the first version of the script used the side-camera frame times for the bottom-camera
features and crashed on JEB6. Each view now uses its own `frameTimes`."*

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking of the **bottom camera**, `obj.traj{2}`: both paw markers, `top_paw`
**and** `bottom_paw`, plus that view's own `frameTimes`, the session `vidshift` and
`bp.ev.goCue`.

ii.
```python
n2 = feature_names(f, v2)
i_paws = [n2.index(p) for p in ('top_paw', 'bottom_paw') if p in n2]
...
for i in (i_paws if ts2 is not None else []):
    px = resample(ts2[i, 0], tt2)
    py = resample(ts2[i, 1], tt2)
```

iii. Step 5 mapping table: *"`obj.traj{2}(t).ts[:, :, top_paw/bottom_paw]` → `output[4]`
paw_velocity … paws are tracked only by the bottom camera (Methods)"* — matching the Methods
statement *"the paws were tracked using only the bottom view"*. Both markers are used and averaged
(see 8-b); the AI did not document a comparison of their tracking reliability.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as the tongue, plus the two "non-tongue" branches of the reference code.
(1) `interp1` of x and y onto the 10 ms grid, NaN outside the frame range; visibility is the
pre-fill non-NaN mask. (2) `fillmissing('nearest')` on the position (`nearest_fill`), as
`findPosition.m` does for every non-tongue feature. (3) `np.gradient` of the filled position, minus
the baseline drift `median(diff(pos))` — with the AI subtracting each axis's own median derivative
rather than the x-axis one for both axes as the MATLAB does. (4) Speed = `hypot(vx, vy)`, set to NaN
where the marker was not visible. (5) The two paws are averaged bin-by-bin over whichever markers
are available (`nansum / count`), and a bin is "visible" if **either** paw was tracked. Only 1.5% of
bins end up `not_visible` (0.6%–7.6% per session).

ii.
```python
speeds, viss = [], []
for i in (i_paws if ts2 is not None else []):
    px = resample(ts2[i, 0], tt2)
    py = resample(ts2[i, 1], tt2)
    vi = ~np.isnan(px)
    if not vi.any():
        continue
    pxf, pyf = nearest_fill(px), nearest_fill(py)
    vx = np.gradient(pxf) - np.median(np.diff(pxf))
    vy = np.gradient(pyf) - np.median(np.diff(pyf))
    s = np.hypot(vx, vy)
    s[~vi] = np.nan
    speeds.append(s)
    viss.append(vi)
if speeds:
    sm = np.vstack(speeds)
    cnt = np.sum(~np.isnan(sm), axis=0)
    paw_speed[:, tr] = np.where(cnt > 0, np.nansum(sm, axis=0) / np.maximum(cnt, 1), np.nan)
    paw_vis[:, tr] = np.any(np.vstack(viss), axis=0)
```

iii. Two deviations are documented in Step 10, Check 3: *"**Paw 'not visible' is kept as a class.**
`findPosition.m` nearest-fills every non-tongue feature, so the reference has no notion of an
invisible paw. The task explicitly asks for `2: not visible`, so the pre-fill DeepLabCut NaN mask is
used to define it. Velocity itself is still computed on the nearest-filled position exactly as the
reference does."* and *"**`basederiv` applied per axis.** `findVelocity.m` subtracts `basederiv(1)`
(the x-axis median derivative) from *both* the x and y velocity, which looks like an indexing typo.
I subtract the median derivative of each axis from that axis. The effect is negligible (the median
derivative of a tracked feature is ~0) and it cannot change a speed percentile appreciably."*
Step 12 adds evidence that the resulting signal is real: *"lag-1 autocorrelation of the paw speed is
0.62–0.72 (not tracking noise), and mean paw speed roughly doubles after the go cue"*.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: the same `discretize` helper, with the 50th percentile computed per
session over the visible time points of the kept trials; 0 below, 1 at/above, 2 where neither paw
was tracked. Resulting fractions: 0.492 / 0.492 / 0.015.

ii.
```python
paw_cat, paw_thresh = discretize(kin['paw_speed'], kin['paw_vis'], keep_idx)
```
(same `discretize` function quoted in 7-c)

iii. Same rationale as 7-c (Step 5, Decision 9) — the per-session 50th percentile is the Decoder
Task's specification, and it is computed over available points because `not visible` is its own
class. Verified by Step 10, Check G (classes 0 and 1 equal to within 2% in every session).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as the tongue, but using the **bottom** camera's own `frameTimes` (`tt2`), corrected by
the same session `vidshift` and the trial's `goCue`, then `interp1`-ed onto the shared 500-bin
`taxis`. If the bottom camera has no usable frame times for a trial, no paw speed is produced for
that trial and every bin becomes class 2.

ii.
```python
tt  = trial_frame_times(f, v1, tr, vidshift, bp['goCue'][tr])   # side
tt2 = trial_frame_times(f, v2, tr, vidshift, bp['goCue'][tr])   # bottom
...
ts2 = np.array(f[v2['ts'][tr, 0]]) if tt2 is not None else None
...
px = resample(ts2[i, 0], tt2)
```

iii. Same offset and same grid as every other stream (`findVideoOffset.m` + `taxis`). The
per-view frame-time handling is the documented fix for the JEB6 crash (Step 10, Check 5, quoted in
7-d). Step 10, Check F verifies that for 9 random trials the exported paw category time series is
reproduced bin-for-bin (0 mismatching bins out of 500) by an independent re-implementation.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<ANM>_<DATE>.mat` file next to each data structure: `me.data`, a
cell array with one 400 Hz trace per trial (same length as the side camera's `frameTimes`).
`me.moveThresh`, the authors' manual per-session movement threshold, is read but used only for
reporting/comparison. The side camera's `frameTimes` provide the time base. A copy inside `obj.me`
exists for some sessions and is not used.

ii.
```python
def load_motion_energy(me_path):
    """Load motionEnergy_<ANM>_<DATE>.mat -> (list of per-trial 400 Hz traces, moveThresh)."""
    if not os.path.exists(me_path):
        return None, np.nan
    m = sio.loadmat(me_path, struct_as_record=False, squeeze_me=False)
    me = m['me'][0, 0]
    data = me.data
    if not isinstance(data, np.ndarray) or data.dtype != object:
        # some objs store me.data as a struct with a .data field
        data = data[0, 0].data
    cells = [np.asarray(data[i, 0]).flatten() for i in range(data.shape[0])]
    thresh = float(np.asarray(me.moveThresh).flatten()[0]) if hasattr(me, 'moveThresh') else np.nan
    return cells, thresh
```

iii. Step 4 discrepancy table: *"`me` inside `obj` vs separate file — the active code path loads the
separate `motionEnergy_*.mat`; both exist for JEB19 → use the separate file, as the reference does;
verified they agree."* The nested-struct unwrap mirrors `loadMotionEnergy.m`'s
`if isstruct(me.data), me.data = me.data.data; end`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling — the value is already one scalar per frame (the paper's 99th percentile
across pixels of the frame-difference image). The trace is `interp1`-ed onto the 10 ms `taxis` and
then `fillmissing('nearest')` is applied within the trial, exactly as `loadMotionEnergy.m` does, so
the handful of samples at the trial edges that the video does not cover are filled rather than
labelled missing. The AI's own motion-energy median is used as the threshold rather than
`me.moveThresh`, but the two are compared in the notes.

ii.
```python
# ---- motion energy ----
if me_cells is not None and tr < len(me_cells):
    mv = np.asarray(me_cells[tr], float).flatten()
    k = min(len(mv), nf)
    if k > 1:
        # exactly as loadMotionEnergy.m: interp1 onto the aligned axis, then
        # fillmissing(...,'nearest') for the few samples at the trial edges that
        # the video does not cover. me.data itself contains no NaNs, so after
        # this a trial either has motion energy everywhere or nowhere ("no video").
        me[:, tr] = nearest_fill(
            np.interp(taxis, tt[:k], mv[:k], left=np.nan, right=np.nan))
```

iii. Step 5, Decision 10: *"Motion-energy class 2 = the trial has no video at all (a trial-level
property); within a trial the reference's `fillmissing('nearest')` is applied … exactly as
`loadMotionEnergy.m` does."* This was the subject of the AI's one substantive bug fix (Step 10,
issue 2): its first version labelled the ~0.6% of uncovered edge samples as class 2, which the
decoder's balanced loss over-weighted and dropped motion-energy validation accuracy to 0.585;
restoring the reference behaviour raised it to 0.775. The AI also cross-checked its per-session
median against the authors' hand-set `me.moveThresh` and found close agreement (e.g. 8.5 vs 8.0,
9.5 vs 8.0, 15.5 vs 10.0).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize` helper: per-session 50th percentile over the available time points of the
kept trials, 0 below / 1 at-or-above, and 2 (`no_video`) only for trials with no video at all.
Because those trials are dropped from the dataset (1-e), class 2 has **zero** instances; the
converted distribution is 0.499 / 0.501 / 0.000.

ii.
```python
me_vis = ~np.isnan(kin['me']) & kin['has_video'][None, :]
me_cat, me_thresh = discretize(kin['me'], me_vis, keep_idx)
```

iii. Step 10, Check 1: *"`motion_energy` class 2 (`no_video`) has zero instances, so the summary
prints only two of the three declared `output_values` and `train_decoder.py` reports chance as 1/3
rather than the effective 1/2. This is intentional — after the single video-less trial is dropped
every analysed trial has video. The third value is kept in `output_values` because the decoder task
defines the encoding as `0/1/2`, and it documents what a `2` would mean."*

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy has exactly one value per frame of the **side** camera, so the side camera's frame
times, corrected by the session `vidshift` and the trial's `goCue`, are used as its time base
(`tt`), and the trace is interpolated onto the shared 500-bin `taxis`. The number of samples used is
clipped to `min(len(me_trace), len(frameTimes), ts.shape[2])`.

ii.
```python
tt = trial_frame_times(f, v1, tr, vidshift, bp['goCue'][tr])   # side camera
nf = min(len(tt), ts1.shape[2])
tt = tt[:nf]
...
k = min(len(mv), nf)
me[:, tr] = nearest_fill(np.interp(taxis, tt[:k], mv[:k], left=np.nan, right=np.nan))
```

iii. Follows `loadMotionEnergy.m`, which interpolates with
`obj.traj{1}(trix).frameTimes - vidshift - alignTimes(trix)` — i.e. the side camera (`traj{1}`).
Sanity-checked in Step 10, Check F (motion-energy category time series reproduced bin-for-bin for 9
random trials by an independent re-implementation).

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Eight cases, enumerated in Step 10, Check 5. **Missing/NaN frame times** → the trial is marked as
having no video and is dropped (1 trial, JEB19 2023-04-19). **Mismatched frame counts between the
two cameras and between `frameTimes` and `ts`** → each view uses its own `frameTimes` and the count
is clipped to `min(len(frameTimes), ts.shape[2])`. **Untracked DeepLabCut frames** (x/y NaN) → the
position is nearest-filled for the paws (as the reference does) but the pre-fill mask defines the
`not visible` class; the tongue is never smoothed. **Samples at the trial edge not covered by video
for motion energy** → nearest-filled within the trial, per `loadMotionEnergy.m`. **Missing
`bp.stim` field** (JEB6) → `try/except` falls back to an all-false stim mask. **Missing `obj.ex`**
(EKH3, JEB7, JGR2, JGR3) → the probe comes from the reference loader scripts. **Unlabelled /
mixed-case / NUL cluster quality strings** → stripped and matched case-sensitively, like MATLAB
`ismember` after `strtrim`. **NaN `goCue`, out-of-range or NaN spike trial indices** → masked out.
Trials where DeepLabCut never sees the paw/tongue become entirely class 2 rather than producing a
`nanmean`-of-empty warning.

ii.
```python
    try:
        out['stim'] = np.array(bp['stim']['enable']).flatten()[:n].astype(bool)
    except (KeyError, TypeError):
        out['stim'] = np.zeros(n, bool)
```
```python
def nearest_fill(x):
    """MATLAB fillmissing(x, 'nearest') for a 1-D array."""
    x = np.asarray(x, float)
    good = ~np.isnan(x)
    if good.all() or not good.any():
        return x.copy()
    ...
```
```python
if ft.size == 0 or np.all(np.isnan(ft)):
    return None
if np.isnan(ft).any():
    return None
```
```python
nf = min(len(tt), ts1.shape[2])
tt = tt[:nf]
...
if vis.sum() == 0:
    return np.full(vals.shape, 2, dtype=np.int64), np.nan
```

iii. The guiding principle stated in the notes is to follow the reference's own missing-data
handling where it exists (`fillmissing('nearest')` for non-tongue features and motion energy, skip
trials with `NdroppedFrames = NaN`), and to use the `not visible` class only where the task
explicitly asks for it. The one trial with no video is dropped rather than labelled, because *"three
of the six decoder outputs are undefined for it and `findPosition.m` skips such trials too"*
(Step 5, Decision 7). The verification log confirms the result contains no NaN or Inf anywhere in
`neural`, `input` or `output`.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the DeepLabCut trajectories out of the HDF5 files. The AI instrumented the three stages
and reports per session: spike binning 0.11–0.35 s, smoothing 0.07–0.32 s, kinematics + motion
energy 1.3–2.2 s, total 1.6–3.1 s per session and **25 s for all 12 sessions** — i.e. the
kinematics stage is ~80% of runtime and *"1.5–2 s of that is HDF5 reads of the DeepLabCut
trajectories"*. Timings are printed per session and stored in `metadata['session_info']['timing']`.

ii.
```python
timing = {}
t0 = time.time()
...
timing['bin_spikes'] = time.time() - t0
t0 = time.time()
rate_s = my_smooth(rate.reshape(t, -1)).reshape(rate.shape).astype(np.float32)
timing['smooth'] = time.time() - t0
...
t0 = time.time()
vidshift = video_offset(f, o, bp)
me_cells, move_thresh = load_motion_energy(me_path)
kin = extract_kinematics(f, o, bp, vidshift, taxis, me_cells)
timing['kinematics'] = time.time() - t0
```

iii. Step 6: *"Run time: 25 s for all 12 sessions (~2 s/session; 1.5–2 s of that is HDF5 reads of
the DeepLabCut trajectories). No parallelism needed."* Since the data must be read once regardless,
the AI judged this irreducible and did not add multiprocessing. (Note that the restricted
12-session scope is itself the largest single contributor to the short runtime.)

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI removed the two loops that dominated the reference's approach: the per-cluster ×
per-trial `histc` loop of `getSeq.m` became one `np.histogram2d` per cluster over
(aligned spike time × trial), and per-trial smoothing became a single `my_smooth` call on the
reshaped `(T, n_units·n_trials)` matrix. `nearest_fill` was written as a vectorised `searchsorted`
rather than an O(n²) nearest-index search. Three loops remain: (1) the per-trial loop in
`extract_kinematics`, which cannot be rectangular because each trial has a different number of
camera frames; (2) the per-cluster loop in `bin_spikes`, which could in principle be a single
`histogram2d` over (spike time × trial × cluster) by concatenating all clusters' spikes with a
cluster id; and (3) — not identified by the AI — the per-column `np.convolve` loop inside
`my_smooth`, which runs over `n_units × n_trials` columns (tens of thousands per session) and could
be replaced by `scipy.ndimage.convolve1d` or an FFT convolution along axis 0.

ii.
```python
for k, ci in enumerate(keep_idx):
    ...
    h, _, _ = np.histogram2d(aligned[inwin], tr[inwin], bins=[edges, trial_bins])
    rate[:, k, :] = h / DT
```
```python
out = np.empty_like(xf, dtype=float)
for j in range(xf.shape[1]):
    out[:, j] = np.convolve(xf[:, j], kernel, mode='same')
```
```python
for tr in range(n):
    tt = trial_frame_times(f, v1, tr, vidshift, bp['goCue'][tr])
    ...
```

iii. Step 6: *"per-cluster per-trial `histc` loops (as in `getSeq.m`) replaced by one
`np.histogram2d` per cluster … smoothing applied once to the reshaped `(T, nunits*ntrials)` matrix
instead of per trial … `nearest_fill` is a vectorised `searchsorted` instead of an O(n²)
nearest-index search."* Since the binning and smoothing stages together take under 0.7 s per
session, further vectorisation would not change the wall clock, which is dominated by file I/O.

## 11-c. What processing does the code repeat multiple times?

i. Very little. The video offset is computed once per session, not per trial; each trial's
DeepLabCut `ts` array is read once and all features are resampled from that single read; the motion
energy file is read once per session; and — the AI's main saving — the condition PSTHs needed for
the >1 Hz filter are formed by averaging the already-smoothed single-trial rates instead of binning
the spikes a second time (valid because `mySmooth` is linear). What *is* repeated: `np.histogram2d`
is run per cluster over the full spike train; `nearest_fill` is called separately for x and y of
each feature; `trial_frame_times` is called once per view per trial; and `convert_session` recomputes
the bin edges and `taxis` for every session.

ii.
```python
# PSTHs for the low-FR filter are obtained by averaging the *already smoothed* single-trial data
def low_fr_mask(rate_smooth, conditions):
    psth = np.full((rate_smooth.shape[0], rate_smooth.shape[1], len(conditions)), np.nan)
    for j, c in enumerate(conditions):
        psth[:, :, j] = rate_smooth[:, :, c].mean(axis=2)
```
```python
vidshift = video_offset(f, o, bp)          # one constant for the whole session
me_cells, move_thresh = load_motion_energy(me_path)
kin = extract_kinematics(f, o, bp, vidshift, taxis, me_cells)
```
```python
ts1 = np.array(f[v1['ts'][tr, 0]])   # read once, all features resampled from it
```

iii. Step 6: *"PSTHs for the low-FR filter are obtained by averaging the *already smoothed*
single-trial data (valid because `mySmooth` is a linear operator), so spikes are binned only once"*
and *"DeepLabCut trajectories are read once per trial and all features resampled from that read."*

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four things. (1) **Binning and smoothing are done for every trial in the session**, including the
~14% of trials that the early-lick/photostim/no-video masks then discard; `keep_idx` is applied only
when the per-trial arrays are assembled, so roughly one trial in seven is binned, smoothed and then
thrown away. (2) **`bp.ev.lickL` / `bp.ev.lickR` are dereferenced for every trial of every session**
(two HDF5 reads per trial) although they are used only by the optional `--show-processing` plot.
(3) **Reporting-only quantities**: the single-unit count `n_single`, the `me.moveThresh` value, and
the full `mean_fr` vector are computed and carried; the `extras` dict retains the whole `(T, units,
trials)` rate array, the unsmoothed `rate`, and all kinematics for every session (freed by
`del extras` only after the optional plot). (4) **Both x and y of `bottom_paw`** are resampled,
filled and differentiated even though the two paws are then averaged into one number. Everything
else computed after loading ends up in the output. Note also that clusters that fail the >1 Hz
filter must be binned first, so that work is not avoidable.

ii.
```python
rate = bin_spikes(f, clu, qual_keep, bp['goCue'], n, edges)      # all n trials
rate_s = my_smooth(rate.reshape(t, -1)).reshape(rate.shape)      # all n trials
...
for j, tr in enumerate(keep_idx):                                # only kept trials exported
    neural_trials.append(np.ascontiguousarray(neural[:, :, tr].T))
```
```python
out['lickL'] = [np.array(f[r]).flatten() for r in np.array(ev['lickL']).flatten()[:n]]
out['lickR'] = [np.array(f[r]).flatten() for r in np.array(ev['lickR']).flatten()[:n]]
```
```python
extras = dict(bp=bp, kin=kin, taxis=taxis, keep_idx=keep_idx, mean_fr=mean_fr,
              neural=neural, tongue_cat=tongue_cat, paw_cat=paw_cat, me_cat=me_cat,
              ...)
```

iii. The notes do not flag these as waste. The low-FR filter genuinely needs the smoothed rates of
all condition trials, which is why binning precedes trial selection, and the AI's measured runtime
(0.1–0.35 s binning, 0.07–0.32 s smoothing per session) makes the overhead immaterial. The
lick-time reads and the `extras` payload exist to support the `--show-processing` diagnostics and
the provenance fields (`trial_ids`, `cluster_ids`, `unit_qualities`) that the AI added so its
independent sanity checks could trace every exported trial and unit back to the raw file (Step 10,
issue 3).
