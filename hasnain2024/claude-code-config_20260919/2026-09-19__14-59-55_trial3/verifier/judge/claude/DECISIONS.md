# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a `SESSIONS` list of 44 `(animal, date, probes, directory, task)` tuples
transcribed from the authors' own loading scripts
(`/app/code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`), rather than globbing
the data folders. Sessions that the authors commented out, and sessions absent from those
scripts, are excluded (25 fixed-delay + 19 randomized-delay). Each session is one
`data_structure_<anm>_<date>.mat` file plus a sibling `motionEnergy_<anm>_<date>.mat`. Because
the released `.mat` files come in two MATLAB formats, two reader classes with a common
interface are provided: `_H5Session` (v7.3/HDF5, via `h5py`) and `_V7Session` (MAT5, via
`scipy.io.loadmat`), dispatched by `open_session` which sniffs the file header. Only the
arrays actually needed are pulled out of the file (for HDF5 the DLC tensor is sliced per
feature), so a session is never fully materialised. Motion energy is read separately by
`load_motion_energy`, which unwraps the three container layouts present in the release.
The two behaviour-only directories (`DelayInhibition_BilatMC_Behavior`,
`GoCueInhibition_BilatMC_Behavior`) are excluded because they contain no neural data.

ii.
```python
SESSIONS = [
    # (animal, date, probes (1-based), data directory, task)
    ('EKH1',  '2021-08-07', [2],    DATA_FIXED, 'DR/WC fixed delay'),
    ...
    ('JEB24', '2023-11-03', [1],    DATA_RAND,  'DR randomized delay'),
]

def _is_hdf5(path):
    with open(path, 'rb') as fh:
        return b'MATLAB 7.3' in fh.read(128)

def open_session(path):
    return _H5Session(path) if _is_hdf5(path) else _V7Session(path)
```
```python
def load_motion_energy(path):
    d = sio.loadmat(path, struct_as_record=False, squeeze_me=False)
    me = d['me']
    if isinstance(me, np.ndarray) and me.dtype == object and not hasattr(me[0, 0], '_fieldnames'):
        cells = me            # me is itself the cell array of trials
    else:
        me = me[0, 0]
        cells = me.data
        if hasattr(cells[0, 0], '_fieldnames'):   # me.data is a struct -> me.data.data
            cells = cells[0, 0].data
    return [np.asarray(x, float).ravel() for x in cells.ravel()]
```
```python
for k, (anm, date, probes, datadir, task) in enumerate(sessions):
    res = process_session(anm, date, probes, datadir, task, ...)
```

iii. From CONVERSION_NOTES.md Step 1/Step 4: the `load<ANM>_ALMVideo.m` files are "the
**definitive session + probe list** … Commented-out entries = sessions the authors excluded."
The AI verified the three files that exist on disk but are not in the lists and found
independent reasons to drop each: `JEB24_2023-10-03` has no `clu` field at all,
`JEB23_2023-10-20` contains trial data identical to `2023-10-19` (a duplicate/mislabelled
session), and `JEB24_2023-10-04` is simply not listed. The 19/22 randomized-delay count also
agrees with the paper's "19 sessions". The dual readers were added after discovering that 11
data objects are MAT5, not v7.3; the three motion-energy layouts mirror the guard
`if isstruct(me.data), me.data = me.data.data; end` in `loadMotionEnergy.m`.

## 1-b. How are the data split into subjects (mice)?

i. The subject is the animal identifier, taken from the first field of each `SESSIONS` tuple
(equivalently the filename prefix). `subjects` is built in order of first appearance while
looping over sessions, and `subject_idx[s]` is the index of that session's animal into
`subjects`. This yields 14 subjects (10 in the fixed-delay set, 4 in the randomized-delay
set) for the 44 sessions, with 1–8 sessions per animal.

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

iii. The AI used the animal id from the load scripts / filename rather than an in-file field;
CONVERSION_NOTES.md Step 4 records that the paper says "nine mice" for the fixed-delay dataset
while both the authors' load scripts and the released files contain ten distinct animals for
exactly the 25 sessions the paper reports, and the AI chose to follow code+data over the paper
and to document the discrepancy rather than force the count.

## 1-c. How are the data split into sessions?

i. One session = one entry of `SESSIONS` = one `data_structure` file, processed by one call to
`process_session`, which returns one element of `neural`, `input`, `output`, and
`brain_region_idx`. The directory (`Ephys_Behavior` vs `RandomizedDelay_Ephys_Behavior`) is
carried in the tuple, so fixed- and randomized-delay sessions are handled uniformly and
appended to the same session list. The probe(s) to use are also per-session
(`[1]`, `[2]`, or `[1, 2]`); for the four dual-probe JEB15 sessions the units of both probes
are concatenated into one population, as `loadSessionData.m` does. Result: 44 sessions,
193–474 trials each, 17–141 units each.

ii.
```python
def process_session(anm, date, probes, datadir, task, show=False, outdir='/app'):
    fn = os.path.join(datadir, f'data_structure_{anm}_{date}.mat')
    obj = open_session(fn)
    ...
    data['neural'].append(res['neural'])
    data['input'].append(res['input'])
    data['output'].append(res['output'])
```
```python
    ('JEB15', '2022-07-26', [1, 2], DATA_FIXED, 'DR/WC fixed delay'),
```

iii. Step 5 Key Decision 1–2: "all 44 ephys sessions listed in the reference
`load*_ALMVideo.m` files"; randomized-delay sessions are kept because "they are valid ALM
recordings with the same task events, video and outcome variables; aligning on the go cue
makes the variable delay irrelevant to the alignment". Probes are "exactly those in the
reference load scripts (`meta.probe`), including the dual-probe JEB15 sessions where both
probes are concatenated, as `loadSessionData` does." The authors' lists are also taken as
already encoding the paper's "≥ 10 units" session criterion (the AI checked that all 44
sessions pass it, min 17 units).

## 1-d. How are the data split into trials?

i. A trial is one row of the Bpod table: `bp.Ntrials` gives the count, and every per-trial
field (`hit`, `miss`, `no`, `R`, `L`, `autowater`, `early`, `stim.enable`, `ev.goCue`,
`ev.sample`, `ev.delay`) is read as a length-`Ntrials` vector by the `bp()` accessor. Spikes
carry their own trial index (`clu.trial`, 1-based) and the DLC/motion-energy streams are
already stored as one cell per trial (`traj{view}(trial)`, `me.data{trial}`), so no trial
boundaries have to be reconstructed. The AI verified the bookkeeping: `hit + miss + no ==
Ntrials` and `R + L == Ntrials` in every session, spike `trial` indices always lie in
`1..Ntrials`, `numel(me.data) == Ntrials`, and `goCue` is finite on every trial.

ii.
```python
    ntrials = obj.Ntrials
    gocue = obj.bp('ev.goCue')
    hit, miss, no = obj.bp('hit'), obj.bp('miss'), obj.bp('no')
    R, L = obj.bp('R'), obj.bp('L')
    autowater = obj.bp('autowater')
    early = obj.bp('early')
    stim = obj.bp('stim.enable') if obj.has_bp('stim.enable') else np.zeros(ntrials)
```
```python
            ok = (tr >= 1) & (tr <= ntrials)
            tt, tr = tt[ok], tr[ok]
            al = tt - gocue[tr - 1]                      # trialtm_aligned
```

iii. Step 4 of CONVERSION_NOTES.md lists the trial-bookkeeping checks above as passing in all
44 sessions ("no unlabelled trials"; "spike `trial` indices always lie in `1..Ntrials`";
"`numel(me.data) == Ntrials` … for all 44 sessions"), so the Bpod trial table is taken as the
definition of a trial with no inference needed. `stim.enable` is guarded with `has_bp` because
it is absent in some sessions.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, combined into a single boolean `keep` mask over the `Ntrials` trials:
(1) early-lick trials (`bp.early`) are dropped; (2) photoinactivation trials
(`bp.stim.enable`) are dropped; (3) trials whose go cue is not finite are dropped (a guard —
no session actually has one); and additionally (4) trials with zero spikes across **all** units
in the whole window are dropped, which removes the trailing trials of two sessions
(JEB24 2023-10-23 and 2023-11-03) where the ephys recording stopped before the behaviour did.
Ignore (`bp.no`) trials are deliberately **kept**, unlike in the reference analyses, because
the decoder task requires a `none` lick-direction class and an `ignore` outcome class.
Across the dataset 13,762 of 14,972 trials survive (976 early-lick, 187 photostim, 64
no-ephys, 17 trials meeting more than one criterion).

ii.
```python
    # early-lick and photoinactivation trials are excluded from all analyses in
    # the reference (params.condition: '~stim.enable & ~early'; ...)
    keep = (early < 0.5) & (stim < 0.5) & np.isfinite(gocue)
    # trials with no spikes at all on any unit = no ephys coverage (the
    # recording ended before the behavioural session did)
    spk_per_trial = trialdat.sum(axis=(0, 1))
    no_ephys = spk_per_trial <= 0
    keep &= ~no_ephys
    ntrials_keep = int(keep.sum())
...
    kt = np.flatnonzero(keep)
    for j, t in enumerate(kt):
        neural.append(np.ascontiguousarray(trialdat[:, :, t].T))       # (nunits, NT)
```

iii. Step 5 Key Decisions 6–7 and Step 10 Check 3: the early/stim removal reproduces the
condition strings used in every reference ephys analysis (`'...&~stim.enable&~early'` in
`findTrials.m`) and the Methods statement that early-lick trials "were omitted from analyses".
The no-ephys filter is flagged as an addition not in the reference, justified because those
trials "would be all-zero inputs and trigger the validator's 'all neural data is zero'
warning". Keeping ignore trials is flagged as a deliberate, task-required departure from the
reference: "Every reference ephys condition string selects `hit`/`miss` only, but the decoder
task requires `none` (lick direction) and `ignore` (outcome) classes".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}(i).trialtm` (spike time relative to the start of its trial, on the
behaviour clock), `obj.clu{probe}(i).trial` (1-based trial index of each spike), and
`obj.clu{probe}(i).quality` (manual curation label). `obj.bp.ev.goCue` supplies the alignment
time, and `obj.ex.probe(p).loc` supplies each unit's brain-region label. Only the probe(s)
named in `SESSIONS` for that session are read; both probes are concatenated where the
reference specifies `[1 2]`.

ii.
```python
    def spikes(self, prb, i):
        g = self._clu(prb)
        return (np.array(self.f[g['trialtm'][i, 0]]).ravel().astype(np.float64),
                np.array(self.f[g['trial'][i, 0]]).ravel().astype(np.int64))
```
```python
    for prb in probes:
        loc = obj.probe_loc(prb)
        region = 'tjM1' if 'M1TJ' in loc.upper() else 'ALM'
```

iii. Step 1: the data are spike times, so "no dF/F is needed; the reference representation
used for every single-trial analysis is `obj.trialdat` = binned, causally smoothed firing
rate", produced from exactly these fields by `alignSpikes.m` + `getSeq.m`. Regions are read
from `ex.probe.loc` rather than assumed to be ALM because "`ex.probe.loc` says probe 2 of
JEB15 is `L M1TJ`"; sessions with no `loc` string are labelled ALM as asserted by the
`*_ALMVideo` load scripts.

## 2-b. How is the `neural` data processed?

i. Spikes of each retained unit are aligned to that trial's go cue, histogrammed into 10 ms
bins spanning [−2.5, 2.5] s, divided by the bin width to give spikes/s, and then smoothed
along time with a **causal** Gaussian kernel that is a direct port of the authors'
`mySmooth.m`: `gausswin(15)` with the first `floor(15/2) = 7` taps zeroed and the remainder
normalised, with `'reflect'` boundary handling (prepend the first 15 samples, filter, trim).
Because the zeroed taps make MATLAB's `conv(...,'same')` equivalent to a causal FIR filter,
the AI implements it with a single `scipy.signal.lfilter` over a `(500, nunits·ntrials)`
matrix. No normalisation, baseline subtraction, or z-scoring is applied — the stored values
are firing rates in Hz (float32). Units from both probes of a dual-probe session are
concatenated into one population.

ii.
```python
def _causal_kernel(N=SMOOTH_N):
    """mySmooth.m: gaussian window, first floor(N/2) taps zeroed, normalised."""
    k = gausswin(N)
    k[:N // 2] = 0.0
    k = k / k.sum()
    return k[N // 2:]

def my_smooth(x, N=SMOOTH_N, bctype=SMOOTH_BC):
    if bctype == 'reflect':
        xp = np.concatenate([x[:N], x], axis=0)
        trim = N
    ...
    y = lfilter(_FIR, [1.0], xp, axis=0)
    return y[trim:]
```
```python
            b = np.floor((al - TMIN) / DT).astype(np.int64)
            m = (b >= 0) & (b < NT)
            if m.any():
                idx = (tr[m] - 1) * NT + b[m]
                cnt = np.bincount(idx, minlength=ntrials * NT)
                out[:, col, :] = cnt.reshape(ntrials, NT).T.astype(np.float32)
    out /= DT                                            # spikes / s
    sm = my_smooth(out.reshape(NT, -1)).astype(np.float32)
```

iii. Step 5 Key Decision 4: "causally smoothed firing rate (spikes/s), exactly the
`obj.trialdat` that all reference single-trial analyses (including their choice/context
decoders) use. Smoothing is causal, so no information leaks backwards in time." Step 12 records
that the AI tested alternatives (per-neuron z-scoring, per-session min-max as the reference
decoders use, √-transform) on an 11-session subset and found raw smoothed rates as good or
better, so the dataset ships the un-normalised reference representation. Step 10 Check 2
verified 12 random (session, trial, unit) firing-rate traces against an independent
re-implementation (`np.convolve` instead of `lfilter`) with `np.allclose(atol=1e-4)`,
max difference ≈ 3e-6.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two stages, in the reference's order. (1) The manual curation label `clu.quality` is
stripped, lower-cased and dropped if it is one of `garbage`, `gabrga` (the authors' typo),
`noisy`, or `real?` — exactly the four labels excluded by `findClusters.m` under
`params.quality = {'all'}`; everything else is kept, including multi-units, `fair`, `poor`,
and unlabelled clusters. (2) Units whose mean firing rate over the [−2.5, 2.5] s window,
averaged across **all** trials, is ≤ 1 Hz are dropped (`removeLowFRClusters.m`,
`params.lowFR = 1`). The firing-rate criterion is computed before trial curation so the unit
set does not depend on the trial subset. Across the dataset this leaves 2,457 units of
10,330 clusters (1,532 fixed-delay + 925 randomized-delay), 17–141 per session. No
session-level unit-count filter is applied separately, since all 44 sessions pass the paper's
≥ 10-unit criterion.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
...
    for prb in probes:
        q = obj.qualities(prb)
        keep_clu[prb] = [i for i, qq in enumerate(q) if qq.lower() not in BAD_QUALITY]
    ...
    fr = mean_firing_rates(obj, probes, keep_clu, gocue, ntrials)
    use = fr > LOW_FR
```
```python
def mean_firing_rates(obj, probes, keep_clu, gocue, ntrials):
    """removeLowFRClusters.m: 'all those firing less than lowFR spikes per second
    on average across all trials'."""
            al = tt[ok] - gocue[tr[ok] - 1]
            n = np.count_nonzero((al >= TMIN) & (al < TMAX))
            frs.append(n / ((TMAX - TMIN) * ntrials))
```

iii. Step 1/Step 3: the two-stage rule is taken verbatim from `findClusters.m` and
`removeLowFRClusters.m`, and the 1 Hz threshold is corroborated by the paper ("All units with
firing rates exceeding 1 Hz were included in all other analyses"). The one documented
deviation is case-insensitivity: "`findClusters` uses case-sensitive `ismember` against
lower-case labels" while the H2 sessions store capitalised labels, so the AI matches
lower-cased and thereby excludes "the single `Noisy` unit in JEB6 (the reference would have
kept it by accident)" — 1 unit out of 2,513. Step 10 Check 4 documents at length that the
resulting totals (1,532 / 925) differ from the paper's reported 1,651 / 845 by −7 % / +9 %,
that no quality-label combination reproduces the paper's single-unit counts, and that the AI
chose to follow the reproducible reference *code* and document the mismatch.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is a single subtraction, exactly as in `alignSpikes.m`: each spike's
`trialtm` (already relative to its own trial's start, on the behaviour clock) minus
`bp.ev.goCue` of that spike's trial. No interpolation or extra offset is applied to the neural
stream; spikes falling outside [−2.5, 2.5] s are discarded by the bin mask. In WC (autowater)
trials the same `goCue` field holds the water-drop time, which the metadata records.

ii.
```python
            tt, tr = obj.spikes(prb, i)
            ok = (tr >= 1) & (tr <= ntrials)
            tt, tr = tt[ok], tr[ok]
            al = tt - gocue[tr - 1]                      # trialtm_aligned
            b = np.floor((al - TMIN) / DT).astype(np.int64)
            m = (b >= 0) & (b < NT)
```

iii. Step 1/Step 10 Check 3(d): "`alignSpikes.m`: `trialtm_aligned = trialtm - ev.goCue(trial)`;
`params.alignEvent='goCue'`" — the AI's formula is identical. Step 3 notes the paper's axes
read "Time from go cue/water drop", justifying using the same field in both contexts. The
alignment was independently checked in the processing plots: the population firing rate rises
sharply exactly at t = 0, lick-port contacts occur only after t = 0, and the reproduced
choice-decoding time course steps up exactly at −2.2 s (= −1.3 s sample − 0.9 s delay).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (`DT = 0.01`, i.e. `params.dt = 1/100`) over [−2.5, 2.5] s, giving exactly 500
timepoints for every trial of every session. The time axis is built once at module level as
`edges = tmin:dt:tmax` with the stored axis being the bin centres `edges + dt/2` with the last
dropped, reproducing `getSeq.m`'s `obj.time`. Spikes are binned directly at this resolution —
there is no rebinning from a finer grid — and every other stream (the time input and the three
video-derived outputs) is placed on the same 500-bin axis, so all streams share one time
index. No further temporal rebinning is applied.

ii.
```python
TMIN, TMAX = -2.5, 2.5     # params.tmin, params.tmax  (s, relative to go cue)
DT = 0.01                  # params.dt = 1/100 s  -> 10 ms bins

def time_axis():
    """getSeq.m: edges = tmin:dt:tmax, obj.time = edges + dt/2 (last dropped)"""
    edges = np.arange(TMIN, TMAX + DT / 2, DT)
    edges = edges[edges <= TMAX + 1e-9]
    t = edges + DT / 2
    return edges, t[:-1]

EDGES, TAXIS = time_axis()
NT = TAXIS.size
```

iii. Step 3/Step 5 Key Decision 3: `dt = 1/100` is the value set in `WorkingWithDataObjs.m`
(the authors' own tutorial) and in most of the paper's figure scripts, and `tmin/tmax` are
−2.5/2.5. Step 12 reports an explicit test of the choice: accuracy increases monotonically
with coarser bins (lick_direction 0.593 at 10 ms → 0.661 at 200 ms) purely because averaging
denoises, so the AI kept the reference's 10 ms rather than chasing accuracy, arguing that
10 ms "preserves the ~40 ms structure of the tongue-visibility bouts that two of the outputs
encode" and that a user wanting the reference's *decoding* representation can average 7–8 bins
as `DLC_ChoiceDecoder.m` does. Step 10 Check 5 verified the bin-edge convention matches
MATLAB `histc` + `N(1:end-1)`: a spike exactly at −2.5 s falls in the first bin, one exactly at
+2.5 s is dropped.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw variable: it is the analysis time axis itself, defined by the
alignment event (`bp.ev.goCue`) and the window/bin parameters (−2.5 to 2.5 s in 10 ms steps).
The stored input for every trial is the vector of the 500 bin centres, −2.495 … 2.495 s,
identical across trials and sessions, named `time_from_go_cue_s`.

ii.
```python
INPUT_NAMES = ['time_from_go_cue_s']
...
EDGES, TAXIS = time_axis()
...
    inp = TAXIS.astype(np.float32)[None, :]
    for j, t in enumerate(kt):
        inputs.append(inp.copy())
```

iii. Step 5 variable-mapping table: "bin-centre times → `input[session][trial]` (1 × 500);
`edges + dt/2`, i.e. −2.495 … 2.495 s; reference `getSeq` (`obj.time`); the only decoder input
requested". The decoder-task specification asks for "Time from go cue onset in seconds
(continuous, time-varying)", so a continuous-valued row is used rather than a binary indicator.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the axis: `np.arange(TMIN, TMAX + DT/2, DT)`, add `DT/2`, drop the
last element, cast to float32, and tile the same row into every trial of every session. The
axis is built once at import time and reused, so it is bit-identical everywhere.

ii.
```python
def time_axis():
    edges = np.arange(TMIN, TMAX + DT / 2, DT)
    edges = edges[edges <= TMAX + 1e-9]
    t = edges + DT / 2
    return edges, t[:-1]
```

iii. The axis is defined by the task specification and the reference `params`; Step 10 Check 2
item 3 confirms the stored input "equals the bin centres −2.495 … 2.495 for every checked trial
of every session".

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. Spike times are expressed relative to the go cue and
floored into the same `EDGES`, and the input is the centre of those same bins, so column *k* of
the input and column *k* of the neural matrix denote the same 10 ms interval. The same axis is
also the interpolation target for the tongue, paw, and motion-energy streams, so all six
outputs, the input, and the neural data are on one time index.

ii.
```python
            b = np.floor((al - TMIN) / DT).astype(np.int64)     # neural: bin index into EDGES
...
    inp = TAXIS.astype(np.float32)[None, :]                     # input: centres of those bins
...
    taxis = TAXIS + ADVANCE_MOVEMENT                            # video: interpolated onto them
```

iii. By construction — the AI notes in Step 5 that `obj.time = edges + dt/2` is the reference's
own time axis for `obj.trialdat`, and `findPosition.m` / `loadMotionEnergy.m` interpolate the
video onto that same `taxis`, so using it for the input keeps every stream aligned with no
extra work.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Five per-trial Bpod flags: `bp.R` and `bp.L` (the instructed/rewarded side) together with
`bp.hit`, `bp.miss`, and `bp.no`. The licked side is not recorded directly, so it is inferred
from instructed side × outcome; `bp.no` (no response) marks the trials that get the `none`
class.

ii.
```python
    hit, miss, no = obj.bp('hit'), obj.bp('miss'), obj.bp('no')
    R, L = obj.bp('R'), obj.bp('L')
```

iii. Step 5 mapping table cites `getPrevChoice.m`, which defines choice as `(R&hit) | (L&miss)`
= right and NaN on ignore trials. Step 4 records an independent validation: lick direction
derived this way "agrees with the side of the **first lickport contact after the go cue** on
100 % of trials in the sessions tested, and ignore (`bp.no`) trials have **no** post-go-cue
licks", with 99.91 % agreement across all 14,972 trials of all 44 sessions.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A three-way relabelling, constant within a trial and broadcast across all 500 bins:
`left = 0` on `(L & hit) | (R & miss)`, `right = 1` on `(R & hit) | (L & miss)`, and
`none = 2` on `bp.no` trials (also the initial fill value). Codes are ordered to match
`output_values = ['left', 'right', 'none']`.

ii.
```python
    # lick direction (getPrevChoice.m: choice = (R&hit) | (L&miss))
    lickdir = np.full(ntrials, 2, dtype=np.int8)                 # none
    lickdir[((R > 0.5) & (hit > 0.5)) | ((L > 0.5) & (miss > 0.5))] = 1   # right
    lickdir[((L > 0.5) & (hit > 0.5)) | ((R > 0.5) & (miss > 0.5))] = 0   # left
    lickdir[no > 0.5] = 2
...
        out[0] = lickdir[t]
```

iii. The hit/miss inversion is the reference's `getPrevChoice.m` rule; the third class is
required by the decoder task ("Lick direction (left, right, none, per-trial)") and is why
ignore trials are kept (Step 5 Key Decision 7). Step 5 Key Decision 11 explains the
broadcasting: "Per-trial outputs are emitted as time-varying rows (constant within a trial) so
that all six outputs share one `(6, 500)` array". Step 10 Check 2 item 4 re-derived
`lick_direction` from the raw `bp` fields for all trials of 5 sessions with exact agreement.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `bp.autowater`, which is 1 when water was delivered regardless of the
animal's choice — the water-cued (WC) context — and 0 otherwise (delayed-response, DR). The AI
checked the alternative field `autowaterBlock` and found it all-zero where present, so it was
not used.

ii.
```python
    autowater = obj.bp('autowater')
```

iii. Step 4: "`obj.bp.autowater=1 when water was delivered regardless of animal choice … can be
used as a proxy for obtaining water-cued blocks`" (quoted from `WorkingWithDataObjs.m`), and
`autowater` "is used as the DR/WC proxy in every reference script
(`getBlockNum_AltContextTask.m` derives blocks from it)". The AI further validated the field
behaviourally: WC block lengths come out at a median of 11 trials (paper: 10–25) and the first
WC trial occurs at trial index 76–125 in 16/20 two-context sessions (paper: after ~100 DR
trials).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct binary relabelling, per trial, broadcast across the 500 bins: `WC = 0` where
`autowater > 0.5`, `DR = 1` otherwise. All 44 sessions are kept, including the 12 that contain
no WC trials at all; in those, `context` is constant, which the AI treats as a property of the
experiment rather than a defect since the decoder is trained across sessions. Overall the
dataset is 9.7 % WC / 90.3 % DR.

ii.
```python
    context = np.where(autowater > 0.5, 0, 1).astype(np.int8)    # 0=WC, 1=DR
...
        out[1] = context[t]
```

iii. Codes follow the decoder task's "Behavioral context (WC, DR, per-trial)" ordering. Step 4
explains why no session-level two-context subset is imposed: "the paper's 12-session subset is a
behavioural-criterion subset (≥20 correct WC trials per direction). I keep all sessions and let
the per-trial `autowater` flag define the context label; sessions without WC trials simply
contribute only DR labels." Step 10 Check 5 flags the resulting constant-context sessions
explicitly as expected, not a bug.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The three mutually exclusive per-trial Bpod flags `bp.hit`, `bp.miss`, and `bp.no`, all read
directly. Unlike the reference (which derives outcome from `bp.hit` alone and sets NaN on
ignore trials), the AI reads `bp.no` explicitly so that ignore can be its own class.

ii.
```python
    hit, miss, no = obj.bp('hit'), obj.bp('miss'), obj.bp('no')
```

iii. Step 5 mapping cites `getOutcome.m` ("outcome = `bp.hit`, NaN on ignore (`bp.no`)
trials"). Step 4 verified `hit + miss + no == Ntrials` in every session, i.e. the three flags
partition the trials, so the three-class encoding is exhaustive and unambiguous.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling, per trial, broadcast across the 500 bins: `incorrect = 0` on miss,
`correct = 1` on hit, `ignore = 2` on `no`. The array is initialised to −1 and a runtime check
raises if any retained trial still carries −1, i.e. if a trial were not covered by the three
flags. Distribution over the converted dataset: 0.120 / 0.749 / 0.131.

ii.
```python
    outcome = np.full(ntrials, -1, dtype=np.int8)
    outcome[miss > 0.5] = 0                                      # incorrect
    outcome[hit > 0.5] = 1                                       # correct
    outcome[no > 0.5] = 2                                        # ignore
...
    if outcome[keep].min() < 0:
        raise RuntimeError(f'{anm} {date}: trial with no hit/miss/no outcome')
```

iii. The class order is the decoder task's ("incorrect, correct, ignore"). The paper's
analyses exclude ignore trials, but the AI keeps them as a third class because the task demands
it (Step 5 Key Decision 7). The AI also cross-checked the outcome semantics behaviourally: DR
performance comes out at 84.8 % mean (≥ 70 % in 93 % of sessions), consistent with the paper's
training criterion, and median response latency is 186 ms against the paper's "typically within
300 ms".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side camera** — feature `tongue`:
`ts[:, 0:2, featix]` (x and y per frame; the third column is the likelihood, which the authors
have already used to NaN out x and y at likelihood ≤ 0.9), together with `traj{1}(t).frameTimes`,
`traj{1}(t).NdroppedFrames`, `obj.sglx.fs`, `obj.sglx.bitcode.bitstart` and `bp.ev.bitStart`
(for the video clock offset), and `bp.ev.goCue`. Only the one side-camera tongue feature is
used; the bottom camera's `top_tongue` and the other tongue markers are not.

ii.
```python
TONGUE_VIEW, TONGUE_FEAT = 0, 'tongue'
...
    tongue_speed, tongue_vis = kinematic_speed(obj, TONGUE_VIEW, TONGUE_FEAT,
                                               gocue, ntrials, vidshift)
...
    feats = obj.featnames(view)
    featix = feats.index(featname)
    ...
        xy, ft = obj.traj_xy(view, t, featix)
```

iii. Step 2/Step 5 Key Decision 9: "tongue = side-camera `tongue`". `tongue` is the first
feature of the reference's view-1 feature list
(`params.traj_features = {{'tongue','left_tongue',...}, {...}}`) in every figure script, so it
is the canonical tongue marker. The AI notes that "Untracked timepoints are already NaN in
`ts`" (verified here: x/y are NaN on exactly the frames with likelihood ≤ 0.9), so no separate
likelihood threshold is needed.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Per trial: (1) skip the trial entirely if `NdroppedFrames` is NaN or the DLC tensor is
missing (the reference's own video-quality guard); (2) convert frame times to seconds from the
go cue (see 7-d); (3) linearly interpolate x and y from the ~400 Hz frame clock onto the 500
neural bin centres with MATLAB `interp1` semantics — NaN outside the frame range and NaN
propagated from untracked frames; (4) differentiate x and y with a NaN-aware
central-difference `gradient` (one-sided at the edges of each visibility bout, 0 for isolated
samples), giving px per 10 ms bin exactly as the reference's unit-spacing `gradient()` does;
(5) speed is `sqrt(vx² + vy²)`; (6) a bin is "visible" iff both interpolated coordinates and
the resulting speed are finite. No smoothing is applied to the positions, matching
`findPosition.m` which explicitly skips smoothing for tongue features (and calls
`mySmooth(ts, 1, …)`, a no-op, for the others). No nearest-fill is applied.

ii.
```python
    for t in range(ntrials):
        xy, ft = obj.traj_xy(view, t, featix)
        if xy is None or not np.isfinite(obj.ndropped(view, t)):
            continue                                   # no usable video
        if ft is None or not np.all(np.isfinite(ft)) or ft.size != xy.shape[1]:
            ft = np.arange(1, xy.shape[1] + 1) / 400.0 - 0.5 + vidshift
        tv = ft - vidshift - gocue[t]
        x = interp_matlab(taxis, tv, xy[0])
        y = interp_matlab(taxis, tv, xy[1])
        vis = np.isfinite(x) & np.isfinite(y)
        if not vis.any():
            continue
        vx = nan_gradient(x)
        vy = nan_gradient(y)
        speed[t] = np.sqrt(vx ** 2 + vy ** 2)
        visible[t] = vis & np.isfinite(speed[t])
```
```python
def nan_gradient(x):
    """np.gradient-like first derivative that tolerates NaN gaps. ...
    For gap-free signals this is exactly MATLAB's gradient() (unit spacing),
    which is what findVelocity.m uses."""
```

iii. Step 10 Check 3 lists the two deliberate departures from `findVelocity.m` and why:
(a) the reference sets tongue velocity to 0 wherever position is NaN, but "here 'not visible'
is its own output class, so a NaN-aware derivative (one-sided at bout edges) is used instead.
Using the reference's rule would have assigned an artificial speed of 0 — hence class 'below
median' — to ~30 % of the timepoints at which the tongue *is* visible"; (b) the single speed
`sqrt(xvel² + yvel²)` replaces the reference's separate `xvel`/`yvel` features "because the task
asks for a single 'velocity' variable to discretise", and the reference's constant
`basederiv` subtraction (applied only to non-tongue features) is omitted as "< 0.01 px/bin".
Step 10 Check 2 item 6 re-derived the discretised tongue classes from the raw DLC traces with
an independent interpolation (`scipy.interpolate.interp1d`) and gradient, matching exactly on
16 trials.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session: the 50th percentile of the tongue speed pooled over all *valid*
(visible) bins of all *retained* trials of that session. Bins at or above it get class 1, bins
below get class 0, and every bin where the tongue was not tracked (or the trial had no usable
video) gets class 2, `not_visible`. Per-session thresholds are used because pixel scales differ
between rigs. Over the converted dataset the classes come out at 0.046 / 0.046 / 0.909, and by
construction the class-0 and class-1 counts differ by < 2 % within each session.

ii.
```python
def discretize(values, valid, thresh=None):
    """0 = < median, 1 = >= median, 2 = invalid (not visible / no video)."""
    v = np.full(values.shape, 2, dtype=np.int8)
    if thresh is None:
        thresh = np.nanpercentile(values[valid], 50) if valid.any() else np.nan
    if valid.any():
        v[valid] = (values[valid] >= thresh).astype(np.int8)
    return v, thresh
...
    # per-session median thresholds, computed on the retained trials only
    tng_d, tng_thr = discretize(tongue_speed[keep], tongue_vis[keep])
```

iii. Step 5 Key Decision 8: "per-session 50th percentile of all *valid* samples of the retained
trials, exactly as instructed; class 2 = tongue/paw not tracked by DLC… Per-session thresholds
are necessary because camera/scale differences make the units incomparable across sessions
(e.g. median tongue speed 3.8 px/bin in JEB6 vs 14 px/bin in JEB23)." Step 10 Check 3 notes that
the authors' own manual `me.moveThresh` was *not* used because the task specification mandates
the 50th percentile (for reference, `moveThresh` corresponds to the 43rd–88th percentile,
mean ≈ 65th).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video clock is converted to the behaviour clock with a session-constant offset computed
once per session by a port of `findVideoOffset.m`:
`mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`. Each trial's frame times then
become `frameTimes − vidshift − goCue[trial]`, and x/y are interpolated onto the identical 500
bin centres used for the spikes, so bin *k* means the same interval in both streams. If
`frameTimes` is missing, all-NaN, or of the wrong length, the reference's fallback nominal
400 Hz clock is used (`(1:n)/400 − 0.5 − goCue`, expressed in the code as
`−0.5 + vidshift` before the shared `− vidshift` subtraction).

ii.
```python
def video_offset(obj):
    """findVideoOffset.m"""
    bitstart = obj.sglx_bitstart()
    fs = obj.sglx_fs()
    bp_bitstart = obj.bp('ev.bitStart')
    bp_bitstart = bp_bitstart[np.isfinite(bp_bitstart)]
    return (stats.mode(bitstart, keepdims=False).mode / fs
            - stats.mode(bp_bitstart, keepdims=False).mode)
...
        tv = ft - vidshift - gocue[t]
        x = interp_matlab(taxis, tv, xy[0])
```

iii. Step 1/Step 3: the formula is `findVideoOffset.m` verbatim, and the computed offsets
(0.49 s for 13 of 14 animals, 0.99 s for JEB19) bracket the tutorial's "subtract 0.5 second
from frametimes". The offset is computed once per session rather than per trial. Step 7
records the alignment sanity check: "the tongue is tracked (not 'not visible') essentially only
**after** the go cue, i.e. during licking, which independently confirms the video↔ephys
alignment."

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{2}` — the **bottom camera** — feature `top_paw`
(x, y per frame), plus the same `frameTimes`, `NdroppedFrames`, clock-offset fields and
`bp.ev.goCue`. The bottom camera's other paw marker, `bottom_paw`, is not used.

ii.
```python
PAW_VIEW, PAW_FEAT = 1, 'top_paw'   # paws are only tracked from the bottom cam;
                                    # 'top_paw_*_view2' is the paw feature used
                                    # in the reference figure code (Figure1e.m)
...
    paw_speed, paw_vis = kinematic_speed(obj, PAW_VIEW, PAW_FEAT,
                                         gocue, ntrials, vidshift)
```

iii. Step 5 Key Decision 9: "paw = bottom-camera `top_paw` (paws are only tracked from the
bottom view, and `top_paw_yvel_view2` is the paw feature the authors use for 'paw speed' in
Fig. 1e)". `Figure1e.m` and `RGBOverlayPlots_Kinematics.m` both use
`feat2use = {'jaw_yvel_view1','nose_yvel_view1','top_paw_yvel_view2'}`, so this is the
reference's own choice of paw marker.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue and implemented by the same function: video-quality guard on
`NdroppedFrames`, conversion of frame times to seconds from the go cue, MATLAB-semantics linear
interpolation of x and y onto the 500 neural bins, NaN-aware `gradient` per axis,
`speed = sqrt(vx² + vy²)` in px per 10 ms bin, and a per-bin visibility mask. The reference's
extra treatments for non-tongue features are **not** applied: no `fillmissing(…,'nearest')`
of positions or velocities, and no subtraction of the median frame-to-frame baseline
`basederiv`. No position smoothing is applied (the reference's `mySmooth(ts, 1, 'reflect')` for
non-tongue features is a no-op at N = 1).

ii.
```python
def kinematic_speed(obj, view, featname, gocue, ntrials, vidshift):
    """Interpolated DLC speed for one feature, aligned to the go cue. ...
    Mirrors findPosition.m / findVelocity.m ..."""
        vx = nan_gradient(x)
        vy = nan_gradient(y)
        speed[t] = np.sqrt(vx ** 2 + vy ** 2)
        visible[t] = vis & np.isfinite(speed[t])
```

iii. Step 10 Check 3: nearest-filling is dropped because "not visible" is a required output
class here — filling would manufacture a velocity for bins the camera never saw and hide them
in classes 0/1. The `basederiv` subtraction is documented as omitted because "it shifts speeds
by < 0.01 px/bin and cannot change a median split materially", and the magnitude
`sqrt(xvel² + yvel²)` is used because the task asks for one velocity variable. 19.4 % of paw
bins end up `not_visible`, which the AI attributes to genuine DeepLabCut drop-out plus window
edges (Step 12 Check 1).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same rule as the tongue, with its own per-session threshold: the 50th percentile of paw
speed over all visible bins of the retained trials of that session; ≥ threshold → 1,
< threshold → 0, untracked → 2 (`not_visible`). No cross-session or cross-feature normalisation
is applied, so the values stay in px per bin. Dataset distribution: 0.403 / 0.403 / 0.194.

ii.
```python
    paw_d, paw_thr = discretize(paw_speed[keep], paw_vis[keep])
```

iii. Same justification as 7-c (Step 5 Key Decision 8): the 50th-percentile split is mandated by
the decoder task, and the threshold is per session because pixel scales and camera geometry are
not comparable across rigs. The threshold actually used is recorded per session in
`metadata['session_info'][i]['paw_speed_threshold']`, and Step 10 Check 2 item 7 verifies that
the class-0 and class-1 counts differ by < 2 % in every session, as a median split requires.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, through the same `kinematic_speed` call: the session's
`findVideoOffset` shift and the trial's `goCue` are subtracted from that camera's own
`frameTimes` — the bottom camera's, since the paw is a view-2 feature — and the trace is
interpolated onto the same 500 neural bin centres. Frame times are read per view, so the two
cameras' differing frame counts cause no cross-talk.

ii.
```python
    def traj_xy(self, view, trix, featix):
        """(x, y) DLC traces for one feature, plus frame times (video clock)."""
        ...
        ft = None
        if 'frameTimes' in g:
            d = np.array(self.f[g['frameTimes'][trix, 0]])
            ...
        return xy, ft
...
        tv = ft - vidshift - gocue[t]
        x = interp_matlab(taxis, tv, xy[0])
        y = interp_matlab(taxis, tv, xy[1])
```

iii. Same as 7-d: one offset formula (`findVideoOffset.m`) and one time grid for every stream,
so no separate treatment is needed for the paw. The AI's Step 7 plot review confirms the
resulting traces show no temporal anomalies and that the "no video" bins at the start of
short-delay randomized sessions fall exactly where the video recording has not yet begun.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside each data structure — `me.data`,
one 400 Hz trace per trial with exactly one value per video frame — together with the side
camera's `traj{1}(t).frameTimes` for timing and the same clock offset and go cue. `obj.me` is
not used (it exists only for behaviour-only sessions), and the authors' per-session
`me.moveThresh` is deliberately not used.

ii.
```python
    mefn = os.path.join(datadir, f'motionEnergy_{anm}_{date}.mat')
    me_cells = load_motion_energy(mefn)
    me = motion_energy_trace(obj, me_cells, gocue, ntrials, vidshift)
    me_vis = np.isfinite(me)
```

iii. Step 2 records the verification that "`me.data{trial}` = 400 Hz motion-energy trace (one
value per video frame, verified to equal `numel(frameTimes)` for **every trial of every
session**)", which is what licenses timing it with the side camera's frame times. The three
container layouts are handled "mirroring `loadMotionEnergy.m`" (Step 6).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment and resampling. The value is already one number per frame (the paper
computes it per pixel as a difference of medians over neighbouring frames, then reduces each
frame to the 99th percentile across pixels), so the AI only interpolates the trace onto the 500
neural bin centres with the same MATLAB `interp1` semantics used for the kinematics, and then
discretises. The reference's `fillmissing(…,'nearest')` on the interpolated motion energy is
**not** applied, so bins outside the video's coverage stay NaN and become the `no_video` class.

ii.
```python
def motion_energy_trace(obj, me_cells, gocue, ntrials, vidshift):
    """loadMotionEnergy.m: interpolate 400 Hz motion energy onto the neural axis."""
    out = np.full((ntrials, NT), np.nan)
    taxis = TAXIS + ADVANCE_MOVEMENT
    for t in range(min(ntrials, len(me_cells))):
        m = me_cells[t]
        if m.size == 0:
            continue
        ft = obj.frame_times(0, t)
        if ft is None or ft.size != m.size or not np.all(np.isfinite(ft)):
            ft = (np.arange(1, m.size + 1) / 400.0) - 0.5 + vidshift  # fallback
        out[t] = interp_matlab(taxis, ft - vidshift - gocue[t], m)
    return out
```

iii. The spatial/temporal reduction has already been done by the authors upstream, so the
conversion only re-times it. Dropping the nearest-fill is the same reasoning as for the paw:
the decoder task defines a distinct `no video` class, so bins with no covering frame must be
representable instead of being silently filled (Step 5 Key Decision 8; Step 10 Check 3).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same `discretize` call with its own per-session threshold: the 50th percentile of the
interpolated motion energy over all finite bins of the retained trials; ≥ threshold → 1,
< threshold → 0, no covering video frame → 2 (`no_video`). The authors' manual `me.moveThresh`
is explicitly not used. Dataset distribution: 0.480 / 0.481 / 0.039.

ii.
```python
    me_d, me_thr = discretize(me[keep], me_vis[keep])
...
OUTPUT_VALUES = [
    ...
    ['below_median', 'above_median', 'no_video'],
]
```

iii. Step 10 Check 3 item 5: "**Median (50th-percentile) thresholds** for the three continuous
variables instead of the authors' manual `me.moveThresh`: required explicitly by the
decoder-task specification. (For reference, `me.moveThresh` corresponds to the 43rd–88th
percentile depending on the session, mean ≈ 65th.)" The per-session threshold is recorded in
`session_info` for every session.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. With the side camera's frame times (view 0), corrected by the same session
`findVideoOffset` shift and the trial's go cue, then interpolated onto the same 500 neural
bins — the same three-step alignment as the kinematics and identical to `loadMotionEnergy.m`'s
`interp1(obj.traj{1}(trix).frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)`.
The reference's `catch` fallback (`frameTimes = (1:n)/400`, then `− 0.5 − alignTime`) is
reproduced for trials whose frame times are missing, all-NaN, or of the wrong length.

ii.
```python
        ft = obj.frame_times(0, t)
        if ft is None or ft.size != m.size or not np.all(np.isfinite(ft)):
            ft = (np.arange(1, m.size + 1) / 400.0) - 0.5 + vidshift  # fallback
        out[t] = interp_matlab(taxis, ft - vidshift - gocue[t], m)
```

iii. Step 10 Check 3(d): "video: `frameTimes - findVideoOffset(obj) - ev.goCue(trial)`; motion
energy identical" — the AI's formula matches. Using view 0's frame times is justified by the
Step 2 verification that the motion-energy trace has exactly one sample per side-camera frame
in every trial of every session. Step 7 records "motion energy rises at the go cue" as the
alignment sanity check.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Each known defect is handled explicitly, and in every case the trial is kept and the gap is
marked rather than filled:
- **Two MATLAB file formats** (11 of the objects are MAT5): two readers behind one interface.
- **Three `motionEnergy` container layouts**: unwrapped by `load_motion_energy`.
- **Sessions missing from the reference lists** (`JEB24 2023-10-03` has no `clu` field at all;
  `JEB23 2023-10-20` duplicates `2023-10-19`): excluded, matching the reference.
- **Missing `stim.enable` field**: `has_bp` guard, defaults to no photostim trials.
- **Bad video for a trial** (`NdroppedFrames` NaN, missing DLC tensor, or non-3-D `ts`): the
  trial is skipped in `kinematic_speed`, exactly the reference's guard, and comes out as 500
  `not_visible` bins.
- **Missing / all-NaN / wrong-length `frameTimes`**: the reference's nominal 400 Hz fallback
  clock is used.
- **Untracked DLC frames** (likelihood ≤ 0.9, already NaN in `ts`): NaN propagates through the
  interpolation and the NaN-aware gradient, so those bins become `not_visible`; nothing is
  interpolated or nearest-filled across them.
- **Bins outside the video's coverage** (e.g. the first ~0.6 s of short-delay randomized-delay
  trials): `no_video` / `not_visible`.
- **Missing `ex.probe.loc`**: the unit is labelled ALM, as the `*_ALMVideo` load scripts assert.
- **Trials past the end of the ephys recording** (64 trials in 2 sessions): dropped.
- **Unlabelled cluster quality** (non-string `quality`): coerced to `''`, which is kept, as
  `findClusters.m` does.
- **A trial with no hit/miss/no flag**: would raise a `RuntimeError` rather than being written
  silently (never triggered).

ii.
```python
        if xy is None or not np.isfinite(obj.ndropped(view, t)):
            continue                                   # no usable video
        # findPosition.m: if frameTimes is missing/all NaN, fall back to a
        # nominal 400 Hz frame clock
        if ft is None or not np.all(np.isfinite(ft)) or ft.size != xy.shape[1]:
            ft = np.arange(1, xy.shape[1] + 1) / 400.0 - 0.5 + vidshift
```
```python
    stim = obj.bp('stim.enable') if obj.has_bp('stim.enable') else np.zeros(ntrials)
...
        out.append(_chars(self.f[q[i, 0]]).replace('\x00', '').strip())
      except Exception:
        out.append('')
```

iii. Step 10 "Issues found and resolved" tabulates each case and its fix. The governing
principle stated in Step 5 Key Decision 8 and Step 10 Check 3 is that missing video is
genuinely missing information, so it gets its own output class instead of being filled — which
is also why the reference's `fillmissing(…,'nearest')` and its "set tongue velocity to 0 if not
visible" rule are dropped: the latter "would have assigned an artificial speed of 0 — hence
class 'below median' — to ~30 % of the timepoints at which the tongue *is* visible". Trials with
missing video are retained because their neural and behavioural data are unaffected.

## 11-a. What are the most time-consuming steps of the code?

i. File I/O and video processing dominate; nothing else is close. The script prints a per-step
timing breakdown for every session. For the 33 v7.3/HDF5 sessions the video step (reading the
DLC hyperslabs and motion energy, then interpolating three streams trial by trial) is
1.6–2.9 s of a 1.9–3.6 s session; for the 11 MAT5 sessions `scipy.io.loadmat` reads the entire
object eagerly and shows up as ~1.0–1.3 s inside the "curation" timer, while their video step
drops to ~0.05 s because the arrays are already in memory. Spike binning + smoothing is
0.04–0.46 s per session. The whole 44-session conversion takes 106 s, so no parallelism was
added.

ii.
```python
    timing['curation'] = time.time() - t0
    ...
    timing['neural'] = time.time() - t1
    ...
    timing['video'] = time.time() - t1
    ...
    timing['total'] = time.time() - t0
    info['timing'] = {k: round(v, 2) for k, v in timing.items()}
```
Measured, from `conversion_full_out.txt`:
```
[2/44] EKH3 2021-08-11 (v7.3) ... {'curation': 0.26, 'neural': 0.41, 'video': 2.61, 'total': 3.3}
[43/44] JEB24 2023-11-02 (v7) ... {'curation': 1.02, 'neural': 0.08, 'video': 0.05, 'total': 1.16}
```

iii. Step 6/Step 7: the AI estimated the full run at ~2 min from the sample sessions and
measured 106 s, "well under the 15-minute budget, no parallelism needed". The video step was
already reduced from ~8 s to ~2.3 s per session by reading only the two needed DLC rows as an
HDF5 hyperslab instead of the whole `(nfeat, 3, nframes)` tensor.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain. (1) The **per-unit loop** in `bin_spikes` and `mean_firing_rates`: each
cluster's spikes are histogrammed with its own `np.bincount`. This could be vectorised into a
single `bincount` over a `(unit, trial, bin)` flat index by concatenating all clusters' spike
arrays, since the per-unit work is already vectorised over spikes. (2) The **per-trial loop** in
`kinematic_speed` and `motion_energy_trace`, which is hard to remove because each trial has a
different number of camera frames, so there is no rectangular array to interpolate at once;
the natural improvement would be to concatenate trials with an offset time axis and do one
`np.interp` per feature. The loops that were vectorised are the per-spike binning
(`np.bincount` over a flattened `(trial, bin)` index instead of a MATLAB-style double loop with
`histc`) and the smoothing (one `lfilter` over a `(500, nunits·ntrials)` matrix instead of
per-unit convolution).

ii.
```python
    for prb in probes:
        for i in keep_clu[prb]:
            ...
                idx = (tr[m] - 1) * NT + b[m]
                cnt = np.bincount(idx, minlength=ntrials * NT)
```
```python
    sm = my_smooth(out.reshape(NT, -1)).astype(np.float32)
```
```python
    for t in range(ntrials):
        xy, ft = obj.traj_xy(view, t, featix)
```

iii. Step 6 "Code inefficiencies identified and removed" lists exactly the two vectorisations
that were made ("per-spike Python loops → `np.bincount`…"; "per-unit/per-trial smoothing → one
`lfilter` call"). The remaining loops were left because the run finishes in 106 s and I/O, not
arithmetic, is the bottleneck.

## 11-c. What processing does the code repeat multiple times?

i. Little, but not nothing. The genuinely repeated work is: (a) **spike times are read and
aligned twice per unit** — once in `mean_firing_rates` to apply the 1 Hz criterion and again in
`bin_spikes` to build the rates — so every quality-passing cluster's `trialtm`/`trial` arrays are
pulled from the file twice and the `tt - gocue[tr-1]` subtraction is done twice; (b)
`obj.ndropped(view, t)` and the per-trial `frameTimes` are re-read once per feature per trial,
i.e. the side camera's frame times are read for the tongue and again for motion energy; (c)
neural binning, smoothing, and all three video streams are computed for **all** `Ntrials`
trials and only afterwards restricted to the ~92 % that survive curation. What is *not*
repeated: the video offset is computed once per session rather than per trial, the smoothing
kernel and the bin grid are built once at module level, each file is opened once, and the
per-trial speeds are computed once and reused by both the percentile and the discretisation.

ii.
```python
    fr = mean_firing_rates(obj, probes, keep_clu, gocue, ntrials)   # reads every cluster
    ...
    trialdat = bin_spikes(obj, probes, keep_clu, gocue, ntrials)    # reads them again
```
```python
    vidshift = video_offset(obj)      # once per session
_FIR = _causal_kernel()               # once per process
EDGES, TAXIS = time_axis()            # once per process
```

iii. Step 6 documents the deduplication that was done (single `lfilter`, single `bincount`, HDF5
hyperslabs) but does not flag the double spike read or the compute-then-discard ordering; both
are harmless at this scale given the 106 s total runtime, and the trial-curation ordering is in
fact partly forced, since the "no ephys coverage" filter is defined from `trialdat` and
therefore cannot precede the binning.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three kinds. (1) **Curated-away trials are processed first**: firing rates, tongue and paw
speed, and motion energy are computed for all `Ntrials` trials and then ~8 % of them
(early-lick, photostim, no-ephys) are dropped, so that fraction of the binning, smoothing, and
interpolation work is thrown away. (2) **Fields read only for metadata or plots**:
`bp.ev.sample` and `bp.ev.delay` are read on every session but only feed the reported median
delay/sample durations and the `--show-processing` figure; `obj.licks()` is read only inside
the plotting function; the quality strings, probe locations, and the per-session diagnostic
statistics (`frac_*`, thresholds, timings) are descriptive metadata rather than decoder inputs.
(3) **Loading overhead in the MAT5 path**: `scipy.io.loadmat` materialises the entire `obj`
tree, including spike waveforms (`clu.spkWavs`), absolute spike times (`clu.tm`), `sglx` index
arrays, and the tracked features other than the tongue and paw, none of which are used — the
HDF5 path avoids this by reading only the needed hyperslabs. Everything the conversion computes
after loading does end up in the output file.

ii.
```python
    sample = obj.bp('ev.sample')
    delay = obj.bp('ev.delay')          # only used for metadata / plots
...
        median_delay_s=float(np.nanmedian(gocue - delay)),
        median_sample_s=float(np.nanmedian(delay - sample)),
```
```python
    d = sio.loadmat(path, struct_as_record=False, squeeze_me=False)   # reads everything
    self.o = d['obj'][0, 0]
```
```python
    trialdat = bin_spikes(obj, probes, keep_clu, gocue, ntrials)  # all trials ...
    ...
    keep = (early < 0.5) & (stim < 0.5) & np.isfinite(gocue)      # ... then ~8% dropped
```

iii. Step 6 notes the one place this was optimised ("reading the whole `(nfeat, 3, nframes)`
DLC tensor per trial → HDF5 hyperslab of the two needed rows") and Step 7 quantifies the gain
(video step ~8 s → ~2.3 s per session). The remaining extra work is retained deliberately: the
metadata fields support the consistency checks the instructions require (delay/sample durations
against the paper's 1.3 s and 0.9 s, per-session class fractions, thresholds, timings), and the
compute-before-curate ordering is partly required by the no-ephys filter, which is defined from
the binned spikes themselves.
