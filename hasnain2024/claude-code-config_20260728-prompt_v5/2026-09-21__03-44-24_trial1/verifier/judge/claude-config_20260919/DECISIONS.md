# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 44-entry dictionary `PROBE_MAP`, transcribed from the authors'
`DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` scripts, mapping
`<anm>_<date>` to the probe number(s) that targeted ALM. It then globs
`data_structure_*.mat` in the two ephys folders (`Ephys_Behavior`,
`RandomizedDelay_Ephys_Behavior`) and **keeps only files whose key appears in
`PROBE_MAP`**, so the glob is only a file-finder and the hard-coded list is the actual
inclusion criterion. The two behavior-only folders (`DelayInhibition_BilatMC_Behavior`,
`GoCueInhibition_BilatMC_Behavior`) are never touched because they contain no
`data_structure_*.mat` ephys objects. The `.mat` files come in two MATLAB formats, so
the AI wrote two wrapper classes with the same interface: `H5Session` (h5py, v7.3) and
`V5Session` (`scipy.io.loadmat`, v5), dispatched by sniffing the first 20 bytes of the
file header for the string `7.3`. Motion energy is loaded from a sibling
`motionEnergy_<anm>_<date>.mat`. Unlike the reference, the AI does not slurp the whole
`obj` tree into memory; the HDF5 file handle stays open and fields are read lazily on
demand.

ii.
```python
PROBE_MAP = {
    # Ephys_Behavior
    'EKH1_2021-08-07': [2], 'EKH3_2021-08-11': [2],
    ...
    'JEB24_2023-11-02': [1], 'JEB24_2023-11-03': [1],
}

def get_session_list():
    sessions = []
    data_dirs = ['/app/data/Ephys_Behavior/', '/app/data/RandomizedDelay_Ephys_Behavior/']
    for dpath in data_dirs:
        files = sorted(glob.glob(os.path.join(dpath, 'data_structure_*.mat')))
        for fp in files:
            key = os.path.basename(fp).replace('data_structure_', '').replace('.mat', '')
            if key not in PROBE_MAP:
                continue
            me_fp = os.path.join(dpath, f'motionEnergy_{anm}_{date}.mat')
            ...
            sessions.append({'key': key, 'data_path': fp, 'me_path': me_fp,
                             'probes': PROBE_MAP[key]})
    return sessions
```

```python
def is_h5_format(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(20).decode('ascii', errors='ignore')
    return '7.3' in header

def open_session(filepath):
    if is_h5_format(filepath):
        return H5Session(filepath)
    else:
        return V5Session(filepath)
```

iii. From CONVERSION_NOTES Step 4 and the trajectory: the AI checked the data folders
against the authors' loading scripts and found extra files that the authors never load
(`JEB24_2023-10-03`, `JEB24_2023-10-04`) and one that is commented out
(`JEB23_2023-10-20`), and excluded them — "These sessions are NOT in the loading
scripts, so exclude them", "Excluded per loading script". It also noted that
`loadJEB4_ALMVideo.m` and `loadJEB5_ALMVideo.m` exist but their data files are absent.
Two readers were written because "v5 format uses scipy.io.loadmat, v7.3 uses h5py —
need to handle both formats uniformly".

## 1-b. How are the data split into subjects?

i. The subject is the part of the filename key before the first underscore
(`JEB19_2023-04-19` → `JEB19`). `subjects` is built in first-encounter order (not
sorted) as sessions are processed, and `subject_idx` is each session's index into that
list. Result: 14 subjects over 44 sessions, matching the reference exactly.

ii.
```python
anm = session_key.split('_')[0]
date = session_key.split('_')[1]
...
anm = result['animal']
if anm not in all_subjects:
    all_subjects.append(anm)
subject_idx.append(all_subjects.index(anm))
```

iii. Not explicitly argued in CONVERSION_NOTES beyond the data-structure description;
the filename is the identifier used by the authors' own per-animal loading scripts
(`load<ANM>_ALMVideo.m`), which the AI transcribed. The AI did investigate the subject
count against the paper (Step 4): the paper says nine mice for the DR task but the
loading scripts cover ten (`JGR3` has a single session), and it decided to "Include all
10 mice" rather than guess at the paper's exclusion.

## 1-c. How are the data split into sessions?

i. One session = one `data_structure_<anm>_<date>.mat` file = one key in `PROBE_MAP` =
one element of `neural`/`input`/`output`. Fixed-delay and randomized-delay sessions are
pooled into one list (25 + 19 = 44) rather than kept as separate datasets. Session
order is: folder order (`Ephys_Behavior` then `RandomizedDelay_Ephys_Behavior`), then
alphabetical filename order within each folder. A session is dropped entirely if it has
fewer than `MIN_UNITS = 10` units before or after the firing-rate filter, or fewer than
2 valid trials; in practice no session was dropped.

ii.
```python
MIN_UNITS = 10  # minimum units per session
...
if len(valid_trials) < 2:
    print(f"  SKIP: Only {len(valid_trials)} valid trials"); sess.close(); return None
...
if len(all_units) < MIN_UNITS:
    print(f"  SKIP: Only {len(all_units)} units (min={MIN_UNITS})"); sess.close(); return None
...
if n_kept < MIN_UNITS:
    print(f"  SKIP: Only {n_kept} units after FR filter (min={MIN_UNITS})"); sess.close(); return None
```

iii. CONVERSION_NOTES Step 5, Key Decision 1: "Include both Ephys and RandomizedDelay
sessions: Both have ephys data from ALM. The paper analyzes them separately but they
share the same recording setup." The ≥10-unit gate is taken from the paper's methods
("Session inclusion | >= 10 units | Methods", Step 3 table).

## 1-d. How are the data split into trials?

i. Trials are the rows of the Bpod table: `bp.Ntrials` gives the count and every
per-trial field (`hit`, `miss`, `no`, `early`, `L`, `R`, `autowater`, `stim.enable`,
`ev.goCue`, …) is read as a flat vector of that length. Spikes carry a 1-based `trial`
index and a within-trial time `trialtm`; camera frames are already stored per trial as
struct-array entries in `obj.traj{view}(trial)`; motion energy is a cell array with one
entry per trial. So no trial boundaries have to be reconstructed — the AI indexes trial
`j` (0-based) by `trial == j + 1` for spikes and `traj[view][j]` for video.

ii.
```python
n_trials = sess.get_ntrials()          # int(obj.bp.Ntrials)
hit  = sess.get_bp_field('hit')
miss = sess.get_bp_field('miss')
no   = sess.get_bp_field('no')
early = sess.get_early()
autowater = sess.get_bp_field('autowater')
R = sess.get_bp_field('R'); L = sess.get_bp_field('L')
stim_enable = sess.get_stim_enable()
goCue = sess.get_ev_field('goCue')
...
for j in range(n_trials):
    trial_num = j + 1                  # 1-indexed trial number
    spk_mask = trial == trial_num
```

iii. Not separately argued; it follows from the data-structure survey in
CONVERSION_NOTES Step 2 ("obj.bp — Behavioral/trial data … obj.traj — `(nTrials,1)`
struct array"). The AI does guard against the trajectory arrays being shorter than
`Ntrials` (`if ti < n_traj_trials_side`) and against the ME cell array being shorter
(`if ti < len(me_data)`).

## 1-e. How are trials filtered based on quality controls?

i. Two behavioural filters plus one data-availability filter.
(1) Photostimulation trials (`bp.stim.enable != 0`) and early-lick trials
(`bp.early != 0`) are dropped up front — this is the reference code's condition string
`'~stim.enable&~early'`. If a session has no `stim` field, `stim_enable` is filled with
zeros. (2) After binning, any surviving trial whose entire neural matrix is zero (i.e.
no unit fired anywhere in the 5 s window) is dropped, on the grounds that the probe
stopped recording before Bpod did. This removed 28 trials from `JEB24_2023-10-23` and
33 from `JEB24_2023-11-03`. Ignore/no-response trials, incorrect trials, and autowater
trials are all **kept**, because they are decoder targets. The final count is 13,762
trials — identical to the reference solution's 13,762.

ii.
```python
# Exclude stim trials (photoinactivation disrupts neural activity)
# Exclude early lick trials per paper/code conventions
# The condition strings in reference code: '~stim.enable&~early'
valid_mask = (stim_enable == 0) & (early == 0)
valid_trials = np.where(valid_mask)[0]  # 0-indexed
```
```python
# ---- Exclude trials with no neural data (recording ended early) ----
# Some sessions have behavioral trials beyond the end of ephys recording
trial_has_spikes = np.any(trialdat > 0, axis=(0, 1))
no_spike_trials = np.where(~trial_has_spikes)[0]
if len(no_spike_trials) > 0:
    valid_trials = np.array([t for t in valid_trials if trial_has_spikes[t]])
```

iii. CONVERSION_NOTES Step 3: "Paper: 'Trials in which the animal contacted the lickport
before the reward (early lick) were omitted from analyses'"; Step 5 Key Decision 2:
"Include all trials (not just correct non-stim non-early). The decoder predicts
outcome/context, so we need all trial types. However, exclude stim-enabled trials since
photoinactivation disrupts neural activity." For the zero-trial cut, the AI first saw
the decoder's "all neural data is zero" warnings, then verified against the raw spike
arrays (trajectory step 170): "For JEB24_2023-10-23, spikes only go up to trial 314 out
of 343 total trials, meaning the recording stopped before the behavioral session ended
… The correct approach is to exclude these trials."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}`, the spike-sorted clusters of the probe(s) listed in `PROBE_MAP`
for that session. Per cluster the AI reads exactly three fields: `quality` (curation
label, used for filtering), `trial` (1-based trial index of each spike) and `trialtm`
(spike time relative to that trial's start). `obj.bp.ev.goCue` supplies the alignment
time. `spkWavs`, `tm`, channel/site and everything else in `clu` are never read. Units
from both probes of a two-probe session are concatenated into one population.

ii.
```python
def get_probe_units(self, probe_idx):
    """Get unit data for a probe (0-indexed)"""
    clu_ref = self.obj['clu'][probe_idx, 0]
    clu = self.f[clu_ref]
    n_units = clu['quality'].shape[0]
    units = []
    for i in range(n_units):
        quality = self._get_string(clu['quality'][i, 0]).lower()
        if quality in EXCLUDE_QUALITIES:
            continue
        trialtm = np.array(self.f[clu['trialtm'][i, 0]]).flatten()
        trial = np.array(self.f[clu['trial'][i, 0]]).flatten().astype(int)
        units.append({'quality': quality, 'trialtm': trialtm, 'trial': trial})
    return units
```
```python
all_units = []
for p in probes:
    units = sess.get_probe_units(p - 1)  # 0-indexed
    all_units.extend(units)
```

iii. CONVERSION_NOTES Step 1: "Probe selection: Each loading script specifies which
probe(s) contain ALM recordings. Some sessions (JEB15) use both probes." Step 10c maps
the alignment onto `alignSpikes.m`: "`trialtm - goCue(trial)`" ↔ "`trialtm[spk_mask] -
goCue[j]`".

## 2-b. How is the `neural` data processed?

i. Three steps, reproducing `getSeq.m` + `mySmooth.m`. (1) Spikes of each unit on each
trial are histogrammed into the fixed 10 ms edge grid spanning −2.5 to +2.5 s from the
go cue. (2) Counts are divided by `dt` to give spikes/s. (3) Each trial's rate trace is
smoothed with the authors' **causal** Gaussian kernel: `gausswin(15)` (σ = (N−1)/(2·2.5)
= 2.8 bins), first half of the taps zeroed, renormalised, convolved with `'same'`, after
prepending the first 15 samples as a reflect boundary. No normalisation, z-scoring or
baseline subtraction; values are firing rates in Hz stored as `float32`.

ii.
```python
def make_causal_gaussian_kernel(N):
    """Create causal Gaussian kernel matching mySmooth.m"""
    kern = gaussian(N, std=(N-1)/(2*2.5))  # MATLAB gausswin(N) default alpha=2.5
    kern[:N//2] = 0  # causal: zero out first half
    kern = kern / kern.sum()
    return kern

def smooth_signal(x, kernel=SMOOTH_KERNEL, bc_type=BC_TYPE):
    ...
    if bc_type == 'reflect':
        x_padded = np.concatenate([x[:N], x], axis=0); trim = N
    ...
    out[:, j] = np.convolve(x_padded[:, j], kernel, mode='same')
    out = out[trim:]
```
```python
counts, _ = np.histogram(spk_times, bins=EDGES)
rate = counts.astype(np.float32) / DT
trialdat[:, i, j] = smooth_signal(rate)
```

iii. CONVERSION_NOTES Step 10c gives a line-by-line comparison table against the
reference MATLAB: firing rate `N./params.dt` ↔ `counts / DT`; kernel `gausswin(15)` →
std=(N−1)/5=2.8 ↔ `gaussian(15, std=2.8)`; `kern(1:floor(N/2)) = 0; kern = kern/sum(kern)`
↔ `kern[:N//2] = 0; kern /= kern.sum()`; `cat(1, x(1:N,:), x)` ↔
`np.concatenate([x[:N], x])`; `conv(x, kern, 'same')` ↔ `np.convolve(..., 'same')`. The
σ was a bug the AI found and fixed during Step 10 ("Gausswin std mismatch: Changed from
`N/6` to `(N-1)/(2*2.5)` to match MATLAB `gausswin`").

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit filters, plus a session gate. (1) The manual curation label `clu.quality` is
stripped, lower-cased and dropped if it is one of `garbage`, `gabrga`, `noisy`, `real?`
— exactly the drop list in `findClusters.m` when `params.quality = {'all'}`. Everything
else is kept, including `Poor`, `Multi`, `ood` and blank labels. (2) After binning and
smoothing, each unit's mean rate over all time bins and all valid trials must exceed
1 Hz. (3) A session needs ≥10 surviving units. Result: 2,456 units over 44 sessions
(17–141 per session).

ii.
```python
LOW_FR = 1.0    # Hz, minimum mean firing rate
MIN_UNITS = 10
# Quality labels to EXCLUDE (from findClusters.m)
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
quality = self._get_string(clu['quality'][i, 0]).lower()
if quality in EXCLUDE_QUALITIES:
    continue
```
```python
# ---- Remove low FR units (matching removeLowFRClusters.m) ----
# In reference code: meanFRs = mean(mean(obj.psth{prbnum},3,'omitnan'),'omitnan')
mean_fr = np.mean(np.mean(trialdat[:, :, valid_trials], axis=2), axis=0)
keep_units = mean_fr > LOW_FR
trialdat = trialdat[:, keep_units, :]
```

iii. The AI enumerated the actual labels in the data (trajectory step 66: "Excellent,
Great, Good, Fair, Poor, Multi, garbage, '\x00\x00' (empty/null), and 'ood'") and
concluded "findClusters with quality='all' keeps everything except 'garbage', 'gabrga'
(typo), 'noisy', and 'real?'". It knowingly deviated from MATLAB in one respect and
documented it (Step 10c note 1): "MATLAB `ismember` is case-sensitive; our code
lowercases first … we may exclude slightly more units if any have mixed-case quality
labels". It also documented that its FR filter averages over trials rather than over
condition-averaged PSTHs (note 2): "Impact: minor difference when conditions are
imbalanced, affecting a few units at the threshold boundary." It investigated the unit
count against the paper (1,531 vs 1,651 for the fixed-delay set; 925 vs 845 for the
randomized-delay set) and concluded the paper must count differently, leaving the
filter as the code specifies.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction. `clu.trialtm` is already on the behaviour clock and relative to its
own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]`
puts every spike in seconds from go-cue onset. No interpolation or extra offset. Spikes
falling outside [−2.5, 2.5] simply fall outside the histogram edges. (Only the video
streams need the separate bitcode clock correction.)

ii.
```python
# Matching alignSpikes.m: trialtm_aligned = trialtm - goCue(trial)
spk_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(spk_times, bins=EDGES)
```

iii. CONVERSION_NOTES Step 3: "Alignment: Go cue onset (`params.alignEvent = 'goCue'`)",
and Step 10c lists `alignSpikes.m: trialtm - goCue(trial)` against
`trialtm[spk_mask] - goCue[j]` with verdict "Yes".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins, 500 of them, spanning −2.5 to +2.5 s from the go cue; bin centres run
from −2.495 to +2.495 s. The grid is a module-level constant, so every trial, session
and data stream (neural, input, tongue, paw, motion energy) shares one time axis. No
rebinning is applied afterwards: spikes are counted directly at 10 ms, and the 400 Hz
video and motion-energy streams are resampled onto this grid by linear interpolation
rather than being averaged from a finer grid. `metadata['time_bin_size'] = 10.0` ms.

ii.
```python
TMIN = -2.5   # seconds before alignment event
TMAX = 2.5    # seconds after alignment event
DT = 1.0/100  # 10 ms bins
...
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME)
```

iii. CONVERSION_NOTES Step 3: "Neural time bin | 10 ms (dt=1/100) | Code:
WorkingWithDataObjs.m" and "Time window | [-2.5, 2.5] s from goCue | Code:
params.tmin/tmax". The AI read `WorkingWithDataObjs.m`, which literally sets
`params.tmin = -2.5; params.tmax = 2.5; params.dt = 1/100;`, and copied those values.
It did not comment on the fact that the comment two lines above that assignment in the
same file reads "use a 5 ms bin width", nor did it open
`DataLoadingScripts/getDefaultParams.m` (it appears in the file listing the AI printed
but was never read).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. None — the input is defined by the conversion, not read from the data. It is the
vector of bin centres of the alignment grid, i.e. the same `TIME` array described in
2-e, which is anchored to `bp.ev.goCue` only through the alignment of the other
streams. The same array is emitted for every trial of every session.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME = EDGES[:-1] + DT / 2
...
# Input: time from go cue, shape (1, n_timepoints)
input_trials.append(TIME.reshape(1, -1).astype(np.float32))
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: "Time axis | input[0]
'time_from_go_cue' | obj.time = edges + dt/2 (continuous) | getSeq.m | Time-varying,
shape (1, n_timepoints)". The construction is copied from `getSeq.m`'s
`edges = params.tmin:params.dt:params.tmax; obj.time = edges + params.dt/2;`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond `EDGES[:-1] + DT/2` and a cast to `float32`. The values are the true
signed seconds from the go cue (a linear ramp from −2.495 to 2.495), left continuous
rather than converted to a binary event marker.

ii.
```python
input_trials.append(TIME.reshape(1, -1).astype(np.float32))
```

iii. The decoder spec labels this input "continuous, time-varying", which the AI
recorded verbatim in its Step 5 mapping table.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `EDGES` defines the spike histogram bins, and the
input is the centre of those same bins, so column *k* of `input` and column *k* of
`neural` are by construction the same 10 ms interval. The same `TIME` array (`taxis`) is
also the interpolation target for the tongue, paw and motion-energy streams, so all
five streams share one axis.

ii.
```python
counts, _ = np.histogram(spk_times, bins=EDGES)   # neural on EDGES
...
TIME = EDGES[:-1] + DT / 2                        # input = centres of EDGES
...
taxis = TIME.copy()                               # same axis for video streams
me_interp = np.interp(taxis, aligned_times[:n_me], me_trial[:n_me])
```

iii. Implicit in the design; CONVERSION_NOTES Step 3 records "Video alignment:
`frameTimes - vidshift - alignTime(trial)`, then interp1 to neural time axis", i.e. the
neural axis is treated as the master clock.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields of `obj.bp`: `R` and `L` (which port was instructed / which
side the trial was), and `hit`, `miss`, `no` (the outcome flags). Lick direction is not
recorded directly and is inferred from the instructed side crossed with the outcome.

ii.
```python
hit  = sess.get_bp_field('hit')
miss = sess.get_bp_field('miss')
no   = sess.get_bp_field('no')
R = sess.get_bp_field('R')
L = sess.get_bp_field('L')
```

iii. From the in-code comment block the AI wrote while debugging its own first attempt:
"R/L indicate trial type (stimulus side), not lick direction. hit = correct lick (same
as stimulus), miss = wrong lick (opposite side)". CONVERSION_NOTES Step 5 Key Decision
3: "Lick direction: Based on R/L fields. R=1 -> right, L=1 -> left, neither (no
lick/ignore) -> none."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A five-branch relabelling, per trial, evaluated in order: `no` → class 2 (none);
right trial + hit → 1 (right); right trial + miss → 0 (left); left trial + hit → 0;
left trial + miss → 1. Default is 2. The value is then broadcast across all 500 time
bins so the output array is rectangular. Codes: left 0, right 1, none 2. Resulting
distribution 0.423 / 0.446 / 0.131, essentially identical to the reference's
0.4227 / 0.4460 / 0.1313.

Note that an earlier, incorrect version of this loop is still present in the file
immediately above the correct one; it is fully overwritten by the second loop and has
no effect on the output.

ii.
```python
# R=1 means right trial (correct answer is right) ...
# - R trial + hit -> licked right
# - R trial + miss -> licked left (wrong side)
# - L trial + hit -> licked left
# - L trial + miss -> licked right (wrong side)
# - no -> no lick
lick_dir = np.full(len(valid_trials), 2, dtype=int)  # default: none (no lick)
for vi, ti in enumerate(valid_trials):
    if no[ti] == 1:
        lick_dir[vi] = 2  # no response
    elif R[ti] == 1 and hit[ti] == 1:
        lick_dir[vi] = 1  # right lick
    elif R[ti] == 1 and miss[ti] == 1:
        lick_dir[vi] = 0  # left lick (error on right trial)
    elif L[ti] == 1 and hit[ti] == 1:
        lick_dir[vi] = 0  # left lick
    elif L[ti] == 1 and miss[ti] == 1:
        lick_dir[vi] = 1  # right lick (error on left trial)
...
out[0, :] = lick_dir[vi]  # per-trial, broadcast
```

iii. The reasoning is in the comments above (the AI explicitly caught and corrected its
own conflation of instructed side with licked side, trajectory/comment: "Re-do lick
direction properly"). The 'none' class is required by the decoder spec ("left, right,
none, per-trial").

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks trials on which water was
delivered at a random port without any instruction cue.

ii.
```python
autowater = sess.get_bp_field('autowater')
```

iii. CONVERSION_NOTES Step 5 mapping table: "obj.bp.autowater | output[1]
'behavioral_context' | 0=WC, 1=DR | WorkingWithDataObjs.m". The AI verified the
interpretation against the data, noting that randomized-delay sessions have essentially
no autowater trials ("JEB24_2023-10-23 … has almost no WC trials (only 2). The
randomized delay sessions are DR-only, which matches the paper").

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabel: `autowater == 0` → DR (1), otherwise WC (0), broadcast across all
500 bins. Distribution 0.097 WC / 0.903 DR, matching the reference exactly
(0.0966 / 0.9034).

ii.
```python
# Behavioral context: 0=WC, 1=DR
context = np.array([1 if autowater[ti] == 0 else 0 for ti in valid_trials], dtype=int)
...
out[1, :] = context[vi]   # per-trial, broadcast
```

iii. Codes follow the decoder spec's ordering ("Behavioral context (WC, DR,
per-trial)"); Key Decision 4: "autowater=0 -> DR, autowater=1 -> WC. RandomizedDelay
sessions are all DR."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial flags of `obj.bp`: `hit`, `miss` and `no`. Unlike the reference,
which derives 'ignore' as the complement of hit and miss, the AI reads `no` explicitly.

ii.
```python
hit  = sess.get_bp_field('hit')
miss = sess.get_bp_field('miss')
no   = sess.get_bp_field('no')
```

iii. CONVERSION_NOTES Step 5 mapping table: "obj.bp.hit, miss, no | output[2] 'outcome'
| 0=incorrect, 1=correct, 2=ignore | findTrials.m". The same three flags define the
condition strings in the authors' `findTrials.m`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A relabel into three classes with default 2: `hit` → 1 (correct), `miss` → 0
(incorrect), `no` → 2 (ignore), broadcast across all 500 bins. Ignore trials are kept
rather than dropped. Distribution 0.120 / 0.749 / 0.131, matching the reference
(0.1197 / 0.7490 / 0.1313).

ii.
```python
# Outcome: 0=incorrect, 1=correct, 2=ignore
outcome = np.full(len(valid_trials), 2, dtype=int)
for vi, ti in enumerate(valid_trials):
    if hit[ti] == 1:
        outcome[vi] = 1  # correct
    elif miss[ti] == 1:
        outcome[vi] = 0  # incorrect
    elif no[ti] == 1:
        outcome[vi] = 2  # ignore
...
out[2, :] = outcome[vi]   # per-trial, broadcast
```

iii. Codes follow the decoder spec ("incorrect, correct, ignore"). Keeping ignore trials
follows from Key Decision 2 ("Include all trials … The decoder predicts
outcome/context, so we need all trial types").

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, **side camera only** (`view = 0`), feature
named exactly `tongue`. From it the AI reads `featNames` (to locate the feature index),
`ts` (x, y, likelihood per frame — only x and y are used; visibility is taken from the
NaNs the authors already wrote where likelihood was low), `frameTimes`, and
`NdroppedFrames` (a NaN there marks the trial's video as unusable). The bottom camera's
`top_tongue` / `topleft_tongue` / `bottom_tongue` features are not used. `obj.sglx.fs`,
`obj.sglx.bitcode.bitstart` and `bp.ev.bitStart` supply the video clock offset, and
`bp.ev.goCue` the alignment.

ii.
```python
side_feats, _, _, _ = sess.get_traj_data(0, 0)
tongue_idx_side = None
for fi, fn in enumerate(side_feats):
    if fn == 'tongue':
        tongue_idx_side = fi
        break
...
_, ts, frame_times, is_valid = sess.get_traj_data(0, ti)
if is_valid and len(frame_times) > 1 and not np.all(np.isnan(frame_times)):
    tx = ts[:, 0, tongue_idx_side].astype(float)
    ty = ts[:, 1, tongue_idx_side].astype(float)
    vis_raw = ~(np.isnan(tx) | np.isnan(ty))
```

iii. CONVERSION_NOTES Step 2 lists the features of both cameras ("Side cam features:
tongue, left_tongue, right_tongue, jaw, trident, nose, lickport; Bottom cam features:
top_tongue, … top_paw, bottom_paw …"). Key Decision 5: "Tongue velocity: Compute from
DLC tongue position. Use Euclidean velocity = sqrt(xvel^2 + yvel^2). Tongue not visible
-> category 2." No explicit reason is given for preferring the side view; the AI
verified against the data that the side-view tongue is visible ~10% of the time and
only after the go cue (trajectory step 109: "Tongue is only visible during the response
period (~0.1 to ~2.0s from go cue) … about 10% of the trial duration").

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Five steps, following `findPosition.m` / `findVelocity.m` /
`setTongueBaselinePosition`. (1) Frames where x or y is NaN are marked not-visible
(the authors already NaN-out low-likelihood frames, so the AI applies no likelihood
threshold of its own). (2) The NaN positions are filled with a *per-session-per-trial*
baseline — the mean of all visible x and of all visible y on that trial — reproducing
the reference's "set nans in tongue x/y pos to a baseline position". (3) The filled
x and y traces are linearly interpolated from video time onto the 10 ms neural axis.
(4) Velocity is `np.gradient` of each interpolated coordinate (units of pixels per
bin, no division by dt) and speed is `sqrt(vx² + vy²)`. (5) Speed is forced to 0 wherever
the interpolated visibility mask is False, matching `findVelocity.m`'s "set tongue
velocity to 0 if not visible"; those bins are then relabelled class 2 anyway. The
visibility mask itself is carried to the neural axis by interpolating the binary
per-frame mask and thresholding at 0.5.

ii.
```python
vis_raw = ~(np.isnan(tx) | np.isnan(ty))
aligned_times = frame_times - vidshift - goCue[ti]

# Interpolate visibility mask to neural time
vis_interp = np.interp(taxis, aligned_times, vis_raw.astype(float)) > 0.5
tongue_visible[:, vi] = vis_interp

# (matching setTongueBaselinePosition in reference code)
baseline_x = np.nanmean(tx[vis_raw]); baseline_y = np.nanmean(ty[vis_raw])
tx_filled = tx.copy(); ty_filled = ty.copy()
tx_filled[~vis_raw] = baseline_x; ty_filled[~vis_raw] = baseline_y

tx_i = np.interp(taxis, aligned_times, tx_filled)
ty_i = np.interp(taxis, aligned_times, ty_filled)

# Compute velocity (gradient, matching findVelocity.m)
vx = np.gradient(tx_i); vy = np.gradient(ty_i)
speed = np.sqrt(vx**2 + vy**2)
speed[~vis_interp] = 0
tongue_vel[:, vi] = speed
```

iii. CONVERSION_NOTES Step 1: "Tongue handling: Tongue NaNs (not visible) are NOT filled
with nearest (unlike other features); velocity is set to 0 when not visible." Step 10c
compares the baseline choice and flags the one deviation: "Reference computes mean of
initial tongue positions at start of each visible bout. Our code uses mean of all
visible positions. Impact: negligible since baseline is just for filling NaN
positions." The AI arrived at this after a debugging round in which naive `np.interp`
over NaN-containing traces produced 99.8% not-visible (trajectory steps 103–109).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, the 50th percentile is taken over the pooled speeds of every
*visible* bin of every trial in that session (not-visible bins are excluded from the
percentile pool). Visible bins at or above it get class 1, visible bins below get 0,
and every bin whose visibility mask is False — or whose trial had no usable video — gets
class 2. Session thresholds ranged ~8.5–10.8 px/bin; the overall split is
0.051 / 0.051 / 0.898.

ii.
```python
# Tongue velocity: per-session 50th percentile of VISIBLE values
tongue_vel_disc = np.full((N_TIMEBINS, len(valid_trials)), 2, dtype=int)  # default: not visible
visible_tongue_vals = tongue_vel[tongue_visible & ~np.isnan(tongue_vel)]
if len(visible_tongue_vals) > 0:
    thresh_tongue = np.percentile(visible_tongue_vals, 50)
    for vi in range(len(valid_trials)):
        vis = tongue_visible[:, vi]
        valid = vis & ~np.isnan(tongue_vel[:, vi])
        tongue_vel_disc[valid, vi] = (tongue_vel[valid, vi] >= thresh_tongue).astype(int)
        tongue_vel_disc[~vis, vi] = 2  # not visible
```

iii. Directly from the decoder spec ("discretized with per-session threshold: 0: < 50th
percentile, 1: >= 50th percentile, 2: not visible"), recorded in the Step 5 mapping
table. Computing the percentile over visible bins only is the only sensible reading,
since including the zeros written into non-visible bins would push the median to 0.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Two corrections then a resample. The video clock leads the behaviour clock by a
session constant recovered from the bitcode pulse — `mode(sglx.bitcode.bitstart)/sglx.fs
− mode(bp.ev.bitStart)`, exactly `findVideoOffset.m` — computed once per session
(~0.49 s in the sessions logged). Frame times become
`frameTimes − vidshift − goCue[trial]`, and the position/visibility traces are then
linearly interpolated onto the 10 ms neural grid, so tongue bin *k* and neural bin *k*
are the same interval. `np.interp` clamps to the first/last frame outside the video's
coverage rather than returning NaN.

ii.
```python
def get_video_offset(self):
    """Compute video offset matching findVideoOffset.m: mode, not median"""
    fs = self.get_sglx_fs()
    bitstart_sglx = self.get_bitcode_bitstart()
    bitstart_ev = self.get_ev_field('bitStart')
    vidshift = scipy_mode(bitstart_sglx, keepdims=False).mode / fs \
             - scipy_mode(bitstart_ev, keepdims=False).mode
    return vidshift
```
```python
aligned_times = frame_times - vidshift - goCue[ti]
tx_i = np.interp(taxis, aligned_times, tx_filled)
```

iii. CONVERSION_NOTES Step 1: "Video offset: `vidshift = mode(sglx.bitcode.bitstart) /
sglx.fs - mode(bp.ev.bitStart)` ~0.5s"; Step 13 bug list: "Video offset: Changed from
`np.median` to `scipy_mode` to match MATLAB `mode`." Step 5 planned sanity check "Video
features have correct alignment (tongue movement after go cue)" was confirmed in
trajectory step 109 (tongue visible only from ~0.1 s after the go cue).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom camera (`view = 1`) of `obj.traj`, using **every** feature whose name
contains `paw` — i.e. both `top_paw` and `bottom_paw`. x and y from `ts`, plus
`frameTimes`, `NdroppedFrames`, and the same session video offset and `goCue`.

ii.
```python
bottom_feats, _, _, _ = sess.get_traj_data(1, 0)
paw_indices_bottom = [fi for fi, fn in enumerate(bottom_feats) if 'paw' in fn]
...
for pidx in paw_indices_bottom:
    px = ts[:, 0, pidx].astype(float)
    py = ts[:, 1, pidx].astype(float)
```

iii. CONVERSION_NOTES Key Decision 6: "Paw velocity: From bottom cam paw features. Same
discretization. Not visible -> category 2." Step 2 records that the bottom camera is the
one carrying `top_paw` and `bottom_paw`. No per-feature tracking-quality comparison was
made between the two paws.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Per paw: (1) frames with NaN x are marked not-visible; (2) the NaN positions are
filled by nearest/linear fill over the valid frame indices, following
`findPosition.m`'s `fillmissing(...,'nearest')` for non-tongue features; (3) the filled
traces are interpolated onto the 10 ms neural axis; (4) velocity is `np.gradient` of
each coordinate minus a per-trial baseline drift `median(diff(·))`, following
`findVelocity.m`; (5) speed is `sqrt(vx² + vy²)` with residual NaNs set to 0. The two
paws' speeds are then **averaged**, and a bin is called visible if *either* paw was
visible there. No cross-feature normalisation.

ii.
```python
vis_raw = ~np.isnan(px)
# Fill NaN with nearest (matching findPosition.m for non-tongue)
valid_idx = np.where(vis_raw)[0]
px_filled = np.interp(np.arange(len(px)), valid_idx, px[valid_idx])
py_filled = np.interp(np.arange(len(py)), valid_idx, py[valid_idx])

px_i = np.interp(taxis, aligned_times, px_filled)
py_i = np.interp(taxis, aligned_times, py_filled)

# Velocity with baseline subtract (matching findVelocity.m)
vx = np.gradient(px_i); vy = np.gradient(py_i)
base_vx = np.median(np.diff(px_i)); base_vy = np.median(np.diff(py_i))
vx = vx - base_vx; vy = vy - base_vy
speed = np.sqrt(vx**2 + vy**2)
speed[np.isnan(speed)] = 0
paw_speeds.append(speed)
vis_i = np.interp(taxis, aligned_times, vis_raw.astype(float)) > 0.5
paw_vis_list.append(vis_i)
...
if paw_speeds:
    paw_vel[:, vi] = np.mean(paw_speeds, axis=0)
    paw_visible[:, vi] = np.any(paw_vis_list, axis=0)
```

iii. CONVERSION_NOTES Step 10c: "Paw position | `fillmissing('nearest')` | `np.interp`
nearest fill | Yes" and "Paw velocity | `gradient() - nanmedian(gradient())` baseline |
`gradient() - median(diff())` baseline | Close", with note 4: "Both estimate baseline
drift. Impact: negligible." (The reference MATLAB in fact subtracts the x baseline from
both x and y; the AI subtracts each axis's own baseline.) No justification is given for
averaging the two paws rather than choosing one, or for calling a bin visible when only
one of the two paws is tracked.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: one 50th percentile per session over the pooled speeds of
all bins flagged visible, class 1 at or above it, class 0 below, class 2 where neither
paw was visible. Because nearest-filling plus the "either paw visible" rule leaves
~98% of bins visible, the resulting split is 0.490 / 0.490 / 0.019.

ii.
```python
paw_vel_disc = np.full((N_TIMEBINS, len(valid_trials)), 2, dtype=int)  # default: not visible
visible_paw_vals = paw_vel[paw_visible & ~np.isnan(paw_vel)]
if len(visible_paw_vals) > 0:
    thresh_paw = np.percentile(visible_paw_vals, 50)
    for vi in range(len(valid_trials)):
        vis = paw_visible[:, vi]
        valid = vis & ~np.isnan(paw_vel[:, vi])
        paw_vel_disc[valid, vi] = (paw_vel[valid, vi] >= thresh_paw).astype(int)
        paw_vel_disc[~vis, vi] = 2  # not visible
```

iii. Same as 7-c: straight from the decoder spec, recorded in the Step 5 mapping table.
The AI listed per-session visible fractions as a sanity check and accepted them
(Step 10b: "Paw visible ~94-100% of the time (matches expected tracking quality)").

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as the tongue, using the bottom camera's own `frameTimes`: subtract the
session video offset, subtract that trial's `goCue`, then linearly interpolate onto the
shared 10 ms grid. Using the bottom camera's own frame times (rather than the side
camera's) protects against the two views having different frame counts.

ii.
```python
_, ts, frame_times, is_valid = sess.get_traj_data(1, ti)
if is_valid and len(frame_times) > 1 and not np.all(np.isnan(frame_times)):
    aligned_times = frame_times - vidshift - goCue[ti]
    ...
    px_i = np.interp(taxis, aligned_times, px_filled)
```

iii. Same source as 7-d (`findVideoOffset.m`, `findPosition.m`'s
`interp1(traj.frameTimes - vidshift - alignEv, ts, taxis)`), documented in
CONVERSION_NOTES Step 3 "Video alignment" and the Step 10c comparison table.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` sitting beside the data structure,
which holds one already-reduced motion-energy trace per trial at the camera frame rate.
The copy that some sessions carry in `obj.me` is not used. The side camera's
`frameTimes` (from `obj.traj{0}`) supply the timebase. `me.moveThresh` is read where
present but never used.

ii.
```python
def load_motion_energy(me_filepath):
    """Load motion energy file. Handles multiple formats:
    - HDF5 v7.3: me.data is cell array of refs
    - v5 struct: me.data.data is (nTrials,1) object array (nested struct)
    - v5 plain: me is (nTrials,1) object array (no struct, no moveThresh)
    """
    if is_h5_format(me_filepath):
        ...
    else:
        me_raw = sio.loadmat(me_filepath, squeeze_me=False)['me']
        if me_raw.dtype.names and 'data' in me_raw.dtype.names:
            inner = me_raw[0, 0]['data']
            if hasattr(inner, 'dtype') and inner.dtype.names and 'data' in inner.dtype.names:
                data_arr = inner[0, 0]['data']       # nested me.data.data
                ...
        elif me_raw.dtype == object:
            me_data = [np.array(me_raw[i, 0]).flatten() for i in range(me_raw.shape[0])]
            thresh = None
```

iii. CONVERSION_NOTES Step 2: "Each ephys session also has a motionEnergy_ANM_DATE.mat
file with `me.data` (cell array of per-trial ME vectors at 400 Hz) and `me.moveThresh`."
The three-format handling was added after the first full run silently failed on seven
sessions (Step 10e: "JEB15_07-26, JEB15_07-28, JEB24_10-31: nested struct format
`me.data.data`; JEB23_10-10 through 10-13: plain cell array format"); the nested case
mirrors `loadMotionEnergy.m`'s `if isstruct(me.data), me.data = me.data.data; end`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling. The trace is already one scalar per frame, so the AI
linearly interpolates it onto the 10 ms neural axis and then nearest-fills any residual
NaN, exactly the two lines of `loadMotionEnergy.m` (`interp1(...)` then
`fillmissing(me.data,'nearest')`). If the trace and the frame-time vector disagree in
length, both are truncated to the shorter. No smoothing, baseline subtraction or
re-derivation from pixels.

ii.
```python
me_trial = np.array(me_data[ti]).flatten().astype(float)
_, _, frame_times, is_valid = sess.get_traj_data(0, ti)
aligned_times = frame_times - vidshift - goCue[ti]
n_me = min(len(me_trial), len(aligned_times))

# Interpolate ME to neural time (matching loadMotionEnergy.m)
me_interp = np.interp(taxis, aligned_times[:n_me], me_trial[:n_me])

# Fill NaN with nearest (matching loadMotionEnergy.m)
nan_mask = np.isnan(me_interp)
if np.any(~nan_mask) and np.any(nan_mask):
    valid_idx = np.where(~nan_mask)[0]
    me_interp = np.interp(np.arange(len(me_interp)), valid_idx, me_interp[valid_idx])
me_data_aligned[:, vi] = me_interp
```

iii. CONVERSION_NOTES Step 10c: "ME alignment | `interp1(frameTimes-vidshift-alignTimes,
me, taxis)` | `np.interp(taxis, aligned_times, me)` | Yes". The paper's own definition
of motion energy is a per-frame reduction already applied upstream, so nothing is
recomputed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per session, the 50th percentile over all non-NaN aligned ME values of that session;
at or above → 1, below → 0, NaN (or a session with no loadable ME file) → 2 ("no
video"). After the format fixes every session has ME for every trial and `np.interp`'s
edge-clamping leaves no NaN, so class 2 never occurs in the final dataset: the split is
0.500 / 0.500 / 0.000.

ii.
```python
# Motion energy: per-session 50th percentile
me_disc = np.full((N_TIMEBINS, len(valid_trials)), 2, dtype=int)  # default: no video
valid_me = ~np.isnan(me_data_aligned)
if np.any(valid_me) and me_available:
    thresh_me = np.percentile(me_data_aligned[valid_me], 50)
    for vi in range(len(valid_trials)):
        valid_t = ~np.isnan(me_data_aligned[:, vi])
        me_disc[valid_t, vi] = (me_data_aligned[valid_t, vi] >= thresh_me).astype(int)
        me_disc[~valid_t, vi] = 2
```

iii. The decoder spec ("0: < 50th percentile, 1: >= 50th percentile, 2: no video").
The AI used the 50/50 split as its own sanity check on the discretisation (Step 10b:
"Motion energy: perfect 50/50 split (all sessions now have ME data)") and discussed
leaving the unused third label in `output_values` (trajectory step 225): "leaving three
seems safer … the decoder still handles unobserved classes correctly", while noting it
makes the reported uniform chance 1/3 rather than 1/2.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset, same grid. Motion energy is computed from the side-camera video, so its
frame times are `obj.traj{0}(trial).frameTimes`, corrected by the session's bitcode
`vidshift` and the trial's `goCue`, then interpolated onto the shared 10 ms axis. If
`NdroppedFrames` is NaN or fewer than two frame times exist, the trial's ME is skipped
and left NaN.

ii.
```python
_, _, frame_times, is_valid = sess.get_traj_data(0, ti)
if not is_valid or len(frame_times) < 2:
    continue
aligned_times = frame_times - vidshift - goCue[ti]
me_interp = np.interp(taxis, aligned_times[:n_me], me_trial[:n_me])
```

iii. Copied from `loadMotionEnergy.m`, which uses `obj.traj{1}(trix).frameTimes`
(MATLAB 1-based = the side view) for exactly this purpose; recorded in
CONVERSION_NOTES Step 3 "Motion energy alignment: Same as video - interp1 from video
time to neural time".

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, all handled by keeping the trial and degrading gracefully rather than
dropping data.
- **Recording shorter than the behavioural session**: trials whose neural matrix is
  entirely zero are dropped (61 trials across two JEB24 sessions).
- **Missing `stim` field**: `get_stim_enable` returns zeros so the session is treated as
  having no photostim trials.
- **Bad video for a trial**: `NdroppedFrames` NaN, all-NaN `frameTimes`, or fewer than
  two frames ⇒ the trial's tongue/paw/ME stay NaN and become class 2.
- **Untracked frames**: NaN x/y from DeepLabCut mark not-visible; tongue NaNs are filled
  with a baseline position, paw NaNs nearest-filled, both following the reference code.
- **Trajectory/ME arrays shorter than `Ntrials`**: guarded by `ti < n_traj_trials_side`,
  `ti < len(me_data)`; ME/frame-time length mismatch truncated to the shorter.
- **Three motion-energy file layouts** and **two `.mat` versions** are each detected and
  handled.
- **Failure to compute the video offset** falls back to a hard-coded 0.5 s.
- Everything in the video and ME sections is additionally wrapped in bare
  `except Exception: pass` / `except Exception: return 0.5` blocks, so any unanticipated
  error silently leaves that trial's kinematics as NaN → class 2.

ii.
```python
def get_stim_enable(self):
    if self.has_stim():
        return np.array(self.obj['bp']['stim']['enable']).flatten()
    return np.zeros(self.get_ntrials())
```
```python
# Check NdroppedFrames
ndf = np.array(self.f[ndf_ref]).flatten()
is_valid = not np.isnan(ndf[0]) if len(ndf) > 0 else True
```
```python
if is_valid and len(frame_times) > 1 and not np.all(np.isnan(frame_times)):
    ...
except Exception:
    pass
```
```python
def get_video_offset(self):
    try:
        ...
    except Exception:
        return 0.5  # default
```

iii. CONVERSION_NOTES Step 10e enumerates the edge cases found and fixed; Step 13 lists
"All-zero trials: Added exclusion of trials beyond ephys recording range" and "ME
loading: Added support for 3 .mat file formats" as bugs fixed during review. The
`NdroppedFrames` test is copied from `findPosition.m` ("check if video data for trial is
good, skip if not"). No rationale is given for the blanket `except Exception: pass`
handlers, and the notes do not mention that the first full run's seven silently-failed
motion-energy sessions were a direct consequence of one of them.

## 11-a. What are the most time-consuming steps of the code?

i. The AI does not report a per-step timing breakdown anywhere; `convert_data.py` times
only whole sessions (`elapsed = time.time() - t0` per session, plus a grand total), and
the CONVERSION_NOTES "Run Time Estimates" table in Step 7 was left unfilled. From the
logged numbers the full conversion takes 188.7 s for 44 sessions (1.6–7.9 s per
session), well inside the 15-minute budget, and cost scales with `n_units × n_trials`
(v7.3 sessions 4–8 s, the smaller v5 sessions ~2 s). By inspection, the dominant costs
are the triple loop that bins and smooths every (unit, trial) pair — ~700k `np.histogram`
plus `np.convolve` calls — and the per-trial HDF5 trajectory reads, which re-dereference
`featNames`, `ts`, `frameTimes` and `NdroppedFrames` on each of three calls per trial.

ii. What is actually measured:
```python
t0 = time.time()
...
elapsed = time.time() - t0
print(f"  Done: {n_neurons} neurons, {n_valid} trials, {elapsed:.1f}s")
```
The hot loop:
```python
for i, unit in enumerate(all_units):
    for j in range(n_trials):
        spk_mask = trial == trial_num
        ...
        counts, _ = np.histogram(spk_times, bins=EDGES)
        trialdat[:, i, j] = smooth_signal(rate)
```

iii. No justification is offered; the instructions asked for timing information to find
bottlenecks and for an estimate of full-run time, and the AI recorded only the totals.
Its Step 7 entry is three bullet points with no timing discussion.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several, none of which the AI vectorised or flagged.
- The `unit × trial` binning loop: the whole session could be one `np.histogram2d` over
  (trial, aligned spike time), as the reference does; the current code also recomputes
  the boolean mask `trial == trial_num` for every (unit, trial) pair, which is O(n_units
  × n_trials × n_spikes).
- `smooth_signal` is called once per (unit, trial) on a 500-sample vector; it already
  supports 2-D input and loops over columns internally, so a whole unit's
  (time × trials) matrix could be smoothed in one call.
- The per-trial Python loops building `lick_dir`, `context` and `outcome` are pure
  `np.where` / boolean-index operations on length-`n_trials` arrays.
- The three discretisation loops (`for vi in range(len(valid_trials))`) operate
  column-by-column on arrays that are already rectangular and could be a single
  `np.where` over the whole (time × trials) matrix.
- The two-paw inner loop and the per-trial video loop are harder to vectorise because
  frame counts differ per trial, which is the same constraint the reference solution
  cites.

ii. The loop that is most clearly reducible:
```python
for i, unit in enumerate(all_units):
    trialtm = unit['trialtm']; trial = unit['trial']
    for j in range(n_trials):
        trial_num = j + 1
        spk_mask = trial == trial_num
        if not np.any(spk_mask):
            continue
        spk_times = trialtm[spk_mask] - goCue[j]
        counts, _ = np.histogram(spk_times, bins=EDGES)
        rate = counts.astype(np.float32) / DT
        trialdat[:, i, j] = smooth_signal(rate)
```
and the vectorisable label loops:
```python
context = np.array([1 if autowater[ti] == 0 else 0 for ti in valid_trials], dtype=int)
...
for vi in range(len(valid_trials)):
    tongue_vel_disc[valid, vi] = (tongue_vel[valid, vi] >= thresh_tongue).astype(int)
```

iii. No justification: CONVERSION_NOTES Step 6 ("Code inefficiencies identified" /
"Code speedups added" in the template) was replaced with a feature list and contains no
inefficiency analysis. Implicitly the runtime was acceptable (3.1 min), so no
optimisation was attempted.

## 11-c. What processing does the code repeat multiple times?

i. Four repetitions, none documented.
- **Trajectory re-reads**: `get_traj_data(view, trial)` rebuilds the feature-name list
  from scratch on every call (dereferencing 7–10 HDF5 string references) and loads and
  transposes the full `ts` array. It is called three times per trial — side cam for the
  tongue, bottom cam for the paw, and side cam *again* for motion energy, where only
  `frameTimes` is needed — so each session's side-camera `ts` is read twice in full.
- **Feature-name lookup**: `side_feats`/`bottom_feats` are also fetched once up front
  for the index lookup, then re-derived inside every subsequent call and discarded.
- **Lick direction** is computed twice: a first, acknowledged-incorrect loop runs to
  completion and is then overwritten by the corrected loop.
- **File header sniffing**: `is_h5_format` opens the file, then `open_session` opens it
  again; the same for each motion-energy file.
What is *not* repeated: the video offset is computed once per session, the smoothing
kernel and bin grid are module-level constants, and each `.mat` file is opened once per
run.

ii.
```python
def get_traj_data(self, view, trial_idx):
    traj = self.f[self.obj['traj'][view, 0]]
    feat_ref = traj['featNames'][0, 0]           # rebuilt on every call
    feat_ds = self.f[feat_ref]
    feat_names = []
    for j in range(feat_ds.shape[1]):
        feat_names.append(self._get_string(feat_ds[0, j]).lower())
    ts = np.array(self.f[traj['ts'][trial_idx, 0]]).transpose(2, 1, 0)   # loaded even for ME
    ...
```
```python
# for motion energy, only frame_times is used:
_, _, frame_times, is_valid = sess.get_traj_data(0, ti)
```
```python
lick_dir = np.full(len(valid_trials), 2, dtype=int)
for vi, ti in enumerate(valid_trials):
    if L[ti] == 1: ...
        pass  # Will redo below
# Re-do lick direction properly:
lick_dir = np.full(len(valid_trials), 2, dtype=int)
```

iii. Not discussed in CONVERSION_NOTES. The dead first lick-direction loop was left in
place with its own comment ("Actually: L=1 means it's a LEFT trial … Will redo below")
rather than deleted, so the repetition is visible but unacknowledged in the notes.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Binning and smoothing of excluded trials**: `trialdat` is allocated and filled for
  all `n_trials` including early-lick and photostim trials, and only afterwards sliced
  to `valid_trials` — about 9% of the dataset (1,393 of 15,155 trials) is binned,
  smoothed and then thrown away, along with the extra memory for the full array.
- **The dead lick-direction loop** (see 11-c) computes a full per-trial vector that is
  immediately overwritten.
- **Motion-energy trajectory reads**: the side camera's whole `ts` array is loaded and
  transposed on each ME trial when only `frameTimes` is required.
- **`me.moveThresh`** is parsed from every motion-energy file (and special-cased for the
  layouts that lack it) but never used, since the discretisation uses the 50th
  percentile instead.
- **`L` is loaded and used only as the complement of `R`**, and both `no` and the
  hit/miss complement encode the same information.
- **`float32` neural data but `int64` outputs**: `out = np.zeros((6, N_TIMEBINS),
  dtype=int)` stores six small categorical codes per bin in 8 bytes each, contributing
  roughly 330 MB of the 1,838 MB pickle where `int8` would have used ~41 MB.
- The `--show-processing` plotting path recomputes summary statistics from the finished
  arrays, but it is off by default.

ii.
```python
trialdat = np.zeros((N_TIMEBINS, n_units, n_trials), dtype=np.float32)   # all trials
...
trialdat_valid = trialdat[:, :, valid_trials]                            # then subset
```
```python
thresh = float(np.array(me['moveThresh']).flatten()[0])   # never used downstream
```
```python
out = np.zeros((6, N_TIMEBINS), dtype=int)   # int64 for values in {0,1,2}
```

iii. Not discussed. The AI's own Step 6 notes claim the script is efficient without
identifying anything discarded; the conversion completed in 3.1 minutes, so the waste
never became a practical constraint.
