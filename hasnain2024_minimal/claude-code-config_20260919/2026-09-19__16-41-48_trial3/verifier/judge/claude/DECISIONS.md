# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file `data_structure_<anm>_<date>.mat`, living in one of two task folders (`/app/data/Ephys_Behavior` for the fixed-delay task and `/app/data/RandomizedDelay_Ephys_Behavior` for the randomized-delay task), with its motion energy in a sibling `motionEnergy_<anm>_<date>.mat`. Sessions are **not** discovered by globbing: two hard-coded lists (`FIXED_DELAY_SESSIONS`, 25 entries; `RANDOM_DELAY_SESSIONS`, 19 entries) of `(animal, date, [probe numbers])` transcribed from the authors' own `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` loaders. Sessions the authors commented out (JEB13 2022-09-15, JEB23 2023-10-20) and the two JEB24 sessions present on disk but in no loader (2023-10-03/04) are excluded. Reading is delegated to a helper module the agent wrote, `/app/matio.py`, which dispatches on file format: `h5py` for the v7.3/HDF5 files, `scipy.io.loadmat` for the v5 files, normalising both to the same nested `dict` / `list` / `str` / `ndarray` structures. It also skips a list of large unused fields (`spkWavs`, `sglxfns`, …) while reading.

ii.
```python
FIXED_DELAY_SESSIONS = [
    ('JEB6',  '2021-04-18', [2]),
    ...
    ('JEB19', '2023-04-21', [1]),
]

RANDOM_DELAY_SESSIONS = [
    ('JEB11', '2022-05-10', [1]),
    ...
    ('JEB24', '2023-11-03', [1]),
]
```
```python
def process_session(anm, date, probes, folder, task):
    path = os.path.join(folder, 'data_structure_%s_%s.mat' % (anm, date))
    obj = load_obj(path)
```
```python
    sessions = ([(a, d, p, FIXED_DELAY_DIR, 'fixed delay (DR/WC two-context)')
                 for a, d, p in FIXED_DELAY_SESSIONS] +
                [(a, d, p, RANDOM_DELAY_DIR, 'randomized delay (DR only)')
                 for a, d, p in RANDOM_DELAY_SESSIONS])
    for anm, date, probes, folder, task in sessions:
        rez = process_session(anm, date, probes, folder, task)
```
From `/app/matio.py` (the dual-format reader):
```python
SKIP_FIELDS = ('spkWavs', 'sglxfns', 'fnEpochs', 'svpth', 'sglxpth')
```

iii. From the module docstring and the final summary: "sessions / probes: exactly the ones listed in `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` (commented-out entries are excluded, as in the paper)". The agent explicitly enumerated the loaders (`grep -nE "meta\(end\)\.(anm|date|probe)|^%"` over all `load*_ALMVideo.m`) to separate live from commented entries, and separately probed every `.mat` with `h5py` to discover that 11 data-structure files and most motion-energy files are not HDF5, which is why both readers are needed. It kept the randomized-delay sessions because "they are ALM recordings analyzed with the same go-cue-aligned pipeline in the paper (Fig. 3) and all six decoded variables are defined for them".

## 1-b. How are the data split into subjects?

i. The animal identifier is the first element of each hard-coded session tuple (`'JEB19'`, `'EKH1'`, …), so it comes from the session name rather than from inside the file. `subjects` is built in order of first appearance while looping over sessions, and `subject_idx` is that list's index for each session. Result: 14 subjects over 44 sessions.

ii.
```python
        if anm not in subjects:
            subjects.append(anm)
        ...
        subject_idx.append(subjects.index(anm))
    ...
    data['subjects'] = subjects
    data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
```

iii. Not discussed explicitly in the trajectory. The animal name is part of the loader entries the agent transcribed (`meta(end).anm = 'JEB13'`), and the agent found while surveying that `obj.ex` (which carries `anm`) is missing entirely from several older sessions — so the filename/loader name is the only identifier available for every session.

## 1-c. How are the data split into sessions?

i. One session = one `(anm, date, probes)` entry = one `data_structure_*.mat` file = one element of `neural` / `input` / `output` / `brain_region_idx` / `subject_idx`. The folder is carried alongside the entry, so the fixed-delay and randomized-delay sessions are processed by the same code path and concatenated into one list of 44 sessions (25 fixed-delay first, then 19 randomized-delay). The task type is recorded per session in `metadata['session_info']`. Sessions returning `None` (fewer than 10 units or fewer than 2 trials) would be skipped; in practice none were.

ii.
```python
        rez = process_session(anm, date, probes, folder, task)
        if rez is None:
            print('  skipped (too few units or trials)')
            continue
        ...
        data['neural'].append(rez['neural'])
        data['input'].append(rez['input'])
        data['output'].append(rez['output'])
```
```python
    info = {
        'animal': anm, 'date': date, 'task': task, 'probes': list(probes),
        'probe_locations': [locs[p - 1] for p in probes if p - 1 < len(locs)],
        ...
```

iii. Same justification as 1-a: the session list is the authors' loader list. The agent's summary states: "25 fixed-delay (the paper's 25-session DR set, 12 of which are two-context) + 19 randomized-delay sessions".

## 1-d. How are the data split into trials?

i. A trial is one row of the Bpod table `obj.bp`. `Ntrials` gives the count, and every per-trial flag (`hit`, `miss`, `no`, `R`, `L`, `early`, `stim.enable`, `autowater`, `ev.goCue`) is read and coerced to exactly `Ntrials` entries by the helper `as_bool` (which truncates or pads via `np.resize` if a field's stored length differs). Spikes carry their own trial index (`clu.trial`, 1-based) and the video is stored as one entry per trial in `obj.traj{view}`, so no trial boundary has to be reconstructed. Trials are referenced throughout by their original index `trix` into the raw arrays, with `k` the index into the kept subset.

ii.
```python
    ntrials_all = int(np.asarray(bp['Ntrials']).reshape(-1)[0])

    hit = as_bool(bp['hit'], ntrials_all)
    miss = as_bool(bp['miss'], ntrials_all)
    no = as_bool(bp['no'], ntrials_all)
    R = as_bool(bp['R'], ntrials_all)
    L = as_bool(bp['L'], ntrials_all)
    early = as_bool(bp['early'], ntrials_all)
    stim = as_bool(bp['stim']['enable'], ntrials_all) if 'stim' in bp else np.zeros(ntrials_all, bool)
```
```python
def as_bool(x, n):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != n:
        x = np.resize(x, n)
    return np.nan_to_num(x, nan=0.0) > 0.5
```

iii. Not argued explicitly; the agent inspected the file layout first (step showing `bp: hit (1, 416) float64` … `Ntrials (1,1)`, `traj` with 416 per-trial entries, `clu.trial` per spike) and then treated `Ntrials` as authoritative.

## 1-e. How are trials filtered based on quality controls?

i. Four filters. (1) Early-lick trials (`bp.early`) are dropped. (2) Photostimulation trials (`bp.stim.enable`) are dropped. (3) Trials with a non-finite `bp.ev.goCue`, or that are none of hit/miss/ignore, are dropped (both are no-ops on this dataset — every trial has a finite go cue and exactly one of the three flags). (4) After the neural data are built, trials in which **no** surviving unit fired a single spike anywhere in the window are dropped; this catches the two JEB24 sessions where the sorted spike data stop before the behaviour does (28 and 33 trials). Hit, miss and ignore trials are all kept, because outcome is one of the decoded variables. Totals: 14,972 trials on file → 962 early, 187 photostim, 61 zero-spike removed → **13,762 trials kept**.

ii.
```python
    # ---- trial selection: no early licks, no photostimulation, valid go cue
    keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
    trials = np.flatnonzero(keep)
    if trials.size < MIN_TRIALS:
        return None
```
```python
    # In two randomized-delay sessions the sorted spike data stop part way through
    # the behavioral session, leaving late trials without a single spike on any
    # unit. Those trials carry no neural data at all, so they are dropped.
    has_spikes = rates.sum(axis=(0, 1)) > 0
    n_no_spikes = int((~has_spikes).sum())
    if n_no_spikes:
        rates = rates[:, :, has_spikes]
        trials = trials[has_spikes]
```

iii. Docstring: "trials: early-lick trials and photostimulation trials are excluded, as in every `params.condition` of the paper ('~early', '~stim.enable'). Hit, miss and ignore ('no') trials are all kept, because correct / incorrect / ignore is one of the variables to be decoded." The zero-spike filter was added reactively: the first validation run emitted "Session 36 / 43, trial N: all neural data is zero" warnings; the agent then confirmed directly in the raw files that `JEB24_2023-10-23` has spikes only up to trial 314 of 343 and `JEB24_2023-11-03` only to 312 of 346, checked `obj.trials.bp.haveEphys` (all ones, so it does not flag them), and added the filter. The re-run reported "Data format is valid, no errors or warnings."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` — the spike-sorted clusters of the probe(s) the authors' loader selects for that session. Per cluster the code uses `trial` (1-based trial index of each spike), `trialtm` (spike time relative to that trial's start) and `quality` (the manual curation label). `obj.bp.ev.goCue` supplies the alignment time and `obj.ex.probe.loc` the anatomical label. Units from two-probe sessions (the four JEB15 sessions) are concatenated into one population.

ii.
```python
    clu = obj['clu']
    ...
    for prb in probes:
        ...
        region = region_name(locs[prb - 1])
        units = clu[prb - 1]
        ...
        for unit in units:
            quality = str(unit.get('quality', '')).replace('\x00', '').strip()
            if quality.lower() in BAD_QUALITY:
                continue
            utrial = np.asarray(unit['trial'], dtype=float).reshape(-1)
            utm = np.asarray(unit['trialtm'], dtype=float).reshape(-1)
```

iii. Not argued separately; the agent dumped the cluster structure early (`['channel', 'quality', 'spkWavs', 'tm', 'trial', 'trialtm']`, with `trialtm` alongside the absolute `tm`) and read `alignSpikes.m`, which uses exactly `trialtm` and `bp.ev.(alignEvent)`.

## 2-b. How is the `neural` data processed?

i. Spikes of each kept cluster are histogrammed, per trial, into the 500 bin edges; the counts are divided by the bin width to give spikes/s; then the (time × trial) matrix is smoothed along time with a **port of the authors' `mySmooth.m`** — a `gausswin(15, 2.5)` whose first `floor(15/2)` taps are zeroed to make it causal, renormalised to sum 1, convolved `'same'`, with the authors' `'reflect'` boundary handling (prepend the first N samples, convolve, trim them off). No normalisation, z-scoring or baseline subtraction; stored values are firing rates in Hz as `float32`.

ii.
```python
def gausswin(N, alpha=2.5):
    """MATLAB's gausswin(N, alpha)."""
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2.0)) ** 2)


def my_smooth(x, N=SMOOTH_N):
    """Port of utils/mySmooth.m with bctype = 'reflect' (causal Gaussian kernel)."""
    if N <= 1:
        return x
    kern = gausswin(N)
    kern[:N // 2] = 0.0          # causal half
    kern = kern / kern.sum()
    xp = np.concatenate([x[:N], x], axis=0)
    out = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, xp)
    return out[N:]
```
```python
                for k in range(trials.size):
                    if stops[k] > starts[k]:
                        aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
                        dat[:, k] = bin_spikes(aligned, edges)
            dat = my_smooth(dat / DT)
            rates.append(dat.astype(np.float32))
```

iii. Docstring: "spike counts converted to spikes/s and smoothed with the authors' causal Gaussian kernel (`mySmooth`, N = 15 bins, 'reflect' boundary) — i.e. `params.smooth = 15`, `params.bctype = 'reflect'`". The agent read `utils/mySmooth.m` (which contains `kern(1:floor(numel(kern)/2)) = 0; %causal`) and `getSeq.m` (`mySmooth(N./params.dt, params.smooth, params.bctype)`) before writing the port. It sanity-checked the result by printing the population PSTH of one session (4.3 Hz at −2.5 s rising to 11.6 Hz at +0.2 s and decaying back to 4.3 Hz by +2 s).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. (1) Manual curation label: the cluster's `quality` string is stripped (including NUL padding), lower-cased, and dropped if it is one of `garbage`, `gabrga`, `noisy`, `real?` — exactly the set `findClusters(..., {'all'})` rejects. Unlabelled clusters are kept, as that function keeps them; `multi`, `poor`, `fair`, `good`, `great`, `excellent` are all kept. (2) Mean firing rate over the whole window and all kept trials must exceed 1 Hz (`params.lowFR = 1`). (3) Sessions with fewer than 10 surviving units are dropped entirely. Result: **2,456 units** (2,311 labelled ALM, 145 tjM1), 17–141 per session; no session fell below 10 units.

ii.
```python
LOW_FR = 1.0         # Hz (params.lowFR)
MIN_UNITS = 10       # Methods: sessions need at least 10 units

# Cluster qualities rejected by findClusters(..., {'all'}); unlabelled clusters are
# kept, as they are by that function, and are still subject to the firing-rate cut.
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
```
```python
            quality = str(unit.get('quality', '')).replace('\x00', '').strip()
            if quality.lower() in BAD_QUALITY:
                continue
```
```python
    # remove low firing rate units (removeLowFRClusters.m, params.lowFR = 1 Hz)
    mean_fr = rates.mean(axis=(0, 2))
    use = mean_fr > LOW_FR
    if use.sum() < MIN_UNITS:
        return None
    rates = rates[:, use, :]
```

iii. Docstring: "units: all sorted clusters except the qualities rejected by `findClusters` with `params.quality = {'all'}` (garbage / noisy / 'real?'), then clusters with a mean firing rate <= 1 Hz are dropped (`params.lowFR = 1`, `removeLowFRClusters`); sessions with fewer than 10 remaining units are dropped ('Recording sessions were included for analysis only if they had at least 10 units', Methods)". The agent read `findClusters.m` verbatim, then tallied every quality string actually present across the selected probes (`7814 'garbage'`, `724 'multi'`, `332 'Poor'`, `212 'Fair'`, `1 'Noisy'`, `1 'gabrga'`, `1 'real?'`, `7 ''`, …). That tally drove two corrections: it stopped rejecting empty labels (its first version had `''` in `BAD_QUALITY`, which `findClusters` does not reject) and it started stripping NULs before comparing, because the labels are mixed-case and NUL-padded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction. `clu.trialtm` is already on the Bpod clock and already relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]` gives seconds from go cue with no interpolation or offset. The spikes of each cluster are sorted by trial once and sliced per trial with `searchsorted`, so each trial's spikes are shifted by that trial's own go cue. The video streams need an extra clock correction (see 7-d); the neural data do not.

ii.
```python
                order = np.argsort(utrial, kind='stable')
                utrial, utm = utrial[order], utm[order]
                starts = np.searchsorted(utrial, trials + 1, side='left')
                stops = np.searchsorted(utrial, trials + 1, side='right')
                for k in range(trials.size):
                    if stops[k] > starts[k]:
                        aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
                        dat[:, k] = bin_spikes(aligned, edges)
```

iii. Docstring: "alignment: spikes and video are aligned to the go cue (`obj.bp.ev.goCue`), which in water-cued (WC) trials is the time of the water drop". This is `alignSpikes.m`'s `obj.clu{prb}(clu).trialtm_aligned = obj.clu{prb}(clu).trialtm - event` with `params.alignEvent = 'goCue'`, which the agent read. The metadata field records the WC caveat explicitly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms** bins (`DT = 0.01`, `params.dt = 1/100`), 500 non-overlapping bins spanning −2.5 s to +2.5 s from the go cue, identical for every trial, session and stream. There is no rebinning: spikes are histogrammed directly into those edges and the video streams are interpolated directly onto those bin centres, so the neural data, the input and the three camera outputs share one time axis from the start. The grid is built from `np.arange` edges with the bin centre taken as `edges[:-1] + DT/2`, mirroring `getSeq.m`'s `obj.time = edges + params.dt/2; obj.time(1:end-1)`.

ii.
```python
TMIN = -2.5          # s, relative to the go cue
TMAX = 2.5           # s
DT = 0.01            # s (params.dt = 1/100)
```
```python
    edges = np.round(np.arange(TMIN, TMAX + DT / 2, DT), 10)
    time = edges[:-1] + DT / 2
    T = time.size
```
```python
        'time_bin_size': DT * 1000.0,
```

iii. Docstring: "binning: 10 ms bins from -2.5 s to +2.5 s around the go cue … i.e. `params.dt = 1/100`, `params.tmin/tmax = -/+2.5`", attributed to `WorkingWithDataObjs.m`, `Scripts/EDFigure 2/EDFigure2a_Left.m` and `Scripts/Figure 3/Figure3h.m` — the scripts the agent read. (`params.dt = 1/100` is indeed what most of the authors' analysis scripts set; `getDefaultParams.m` alone defaults to `1/200`.)

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable: it is the analysis grid itself, defined by the task (−2.5 s to +2.5 s about the go cue, 10 ms bins) and by `bp.ev.goCue` through the alignment. The input for every trial is the vector of the 500 bin centres, −2.495 s … +2.495 s.

ii.
```python
    edges = np.round(np.arange(TMIN, TMAX + DT / 2, DT), 10)
    time = edges[:-1] + DT / 2
```
```python
INPUT_NAMES = ['time_from_go_cue']
```

iii. N/A — the window comes from the authors' `params.tmin`/`params.tmax`, as recorded in the docstring; the decoder-input specification in the instructions asks for time from go cue onset as a continuous, time-varying input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The same `(1, 500)` `float32` row of bin centres is copied for every trial of every session; `input` is stored as `(d_input=1, n_timepoints=500)` per trial.

ii.
```python
    time_row = time.astype(np.float32)[None, :]
    for k, trix in enumerate(trials):
        ...
        inputs.append(time_row.copy())
```

iii. N/A.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural grid. Spike times are expressed relative to each trial's own go cue and histogrammed into `edges`; the input is `edges[:-1] + DT/2`, the centres of those same bins. Bin *k* therefore denotes the same interval in `neural`, `input` and all six `output` channels. The reported input range in the validation run is `[-2.495, 2.495]`.

ii.
```python
    edges = np.round(np.arange(TMIN, TMAX + DT / 2, DT), 10)
    time = edges[:-1] + DT / 2
    ...
                        aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
                        dat[:, k] = bin_spikes(aligned, edges)
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: the instructed side `R` and `L`, and the outcome flags `hit` and `miss` (with `no` used to force the ignore class). The lick direction itself is not recorded, so it is inferred from instructed side × outcome.

ii.
```python
    hit = as_bool(bp['hit'], ntrials_all)
    miss = as_bool(bp['miss'], ntrials_all)
    no = as_bool(bp['no'], ntrials_all)
    R = as_bool(bp['R'], ntrials_all)
    L = as_bool(bp['L'], ntrials_all)
```

iii. From the code comment and the final summary: "lick direction uses the paper's choice definition `(R&hit)|(L&miss)` (`getPrevChoice.m`), none = ignore". The agent read `funcs/getPrevChoice.m`, which contains exactly `choice = double((obj.bp.R & obj.bp.hit) | (obj.bp.L & obj.bp.miss)); choice(ignore) = nan;`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed port and a miss means it licked the other one, so: right (code 1) if `(R & hit) | (L & miss)`, left (code 0) if `(L & hit) | (R & miss)`, and `none` (code 2) otherwise, with ignore trials explicitly forced to 2. The per-trial value is broadcast across all 500 bins so that all six outputs share one `(6, 500)` `int8` array. `output_values[0] = ['left', 'right', 'none']`.

ii.
```python
    # lick direction follows the paper's choice definition (funcs/getPrevChoice.m):
    # a right lick is (R & hit) | (L & miss); ignore trials have no lick.
    right = (R & hit) | (L & miss)
    left = (L & hit) | (R & miss)
    lick_dir = np.full(ntrials_all, 2, dtype=np.int8)   # 'none'
    lick_dir[left] = 0
    lick_dir[right] = 1
    lick_dir[no] = 2
```
```python
        out = np.empty((6, T), dtype=np.int8)
        out[0, :] = lick_dir[trix]
```

iii. As 4-a: the definition is taken verbatim from `getPrevChoice.m`; the third class is required by the instructions ("left, right, none"). Resulting distribution over the whole dataset: left 0.423, right 0.446, none 0.131.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial field, `obj.bp.autowater`, which marks the water-cued trials (water delivered at a random port with no cues).

ii.
```python
    aw = np.asarray(bp['autowater'], dtype=float).reshape(-1)
    if aw.size != ntrials_all:
        aw = np.resize(aw, ntrials_all)
    # older sessions code autowater as 1 = off / 2 = on, newer ones as 0/1
    autowater = (aw == 2) if np.nanmax(aw) > 1 else (np.nan_to_num(aw) > 0.5)
```

iii. Final summary: "context = `obj.bp.autowater` (the authors' WC/DR proxy)". The agent confirmed the field's block structure by printing the run lengths of contiguous `autowater` values per session (`awruns=[14, 21, 19, 33]`, `[13, 12, 14, 13, 14, 12, 17]`, …), matching the paper's description of WC blocks of 10–25 trials, and cross-read `getBlockNum_AltContextTask.m`, which defines blocks by transitions in `obj.bp.autowater`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: autowater → WC (0), otherwise DR (1), broadcast across all 500 bins. `output_values[1] = ['WC', 'DR']`. The `aw == 2` branch above is a guard for a hypothesised alternative coding; on this dataset `autowater` is 0/1 in all 44 sessions, so the branch never fires.

ii.
```python
    context = np.where(autowater, 0, 1).astype(np.int8)  # 0 = WC, 1 = DR
    ...
        out[1, :] = context[trix]
```

iii. Codes follow the instructions' "(WC, DR)" ordering. The randomized-delay sessions are almost all DR, which the agent noted: "their trials are DR, so context is simply not varied there"; the overall distribution is WC 0.097 / DR 0.903.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial flags of `obj.bp`: `hit`, `miss` and `no` (the ignore flag is read explicitly rather than inferred as "neither").

ii.
```python
    hit = as_bool(bp['hit'], ntrials_all)
    miss = as_bool(bp['miss'], ntrials_all)
    no = as_bool(bp['no'], ntrials_all)
```

iii. Not argued separately beyond "outcome = miss/hit/no" in the final summary. The agent had read `funcs/getOutcome.m` (`outcome = obj.bp.hit; outcome(logical(obj.bp.no)) = nan`) and verified in its per-session survey that hit/miss/no partition every trial (`hmn_sum == Ntrials` in every session).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A relabelling into three classes: incorrect (0) on miss, correct (1) on hit, ignore (2) on `no` — the array is initialised to 2, so any trial that is neither also lands there. Broadcast across all 500 bins. `output_values[2] = ['incorrect', 'correct', 'ignore']`.

ii.
```python
    outcome = np.full(ntrials_all, 2, dtype=np.int8)     # 'ignore'
    outcome[miss] = 0
    outcome[hit] = 1
    outcome[no] = 2
    ...
        out[2, :] = outcome[trix]
```

iii. Codes follow the instructions' "(incorrect, correct, ignore)". Docstring: "Hit, miss and ignore ('no') trials are all kept, because correct / incorrect / ignore is one of the variables to be decoded" — i.e. the agent deliberately departed from the paper, which omits ignore trials from its analyses, in order to keep the third class. Distribution: incorrect 0.120, correct 0.749, ignore 0.131.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, **side camera only** (`traj{1}`, index 0), feature `'tongue'` — the first entry of that camera's `featNames`. From each per-trial entry it uses `ts` (frames × [x, y, likelihood] × features), `frameTimes`, `featNames` and `NdroppedFrames`. `obj.sglx.bitcode.bitstart`, `obj.sglx.fs` and `obj.bp.ev.bitStart` supply the video-clock offset, and `bp.ev.goCue` the alignment.

ii.
```python
# Video features used for the kinematic outputs.  The tongue is tracked from the
# side camera (view 1) and the two paws only from the bottom camera (view 2), as
# described in the Methods ("the paws were tracked using only the bottom view").
TONGUE_FEATURE = ('tongue', 0)          # (feature name, view index)
```
```python
            if tongue_ix is not None and trix < len(views[TONGUE_FEATURE[1]]):
                tt, ts = trial_video_times(views[TONGUE_FEATURE[1]][trix], vidshift,
                                           gocue[trix])
                if tt is not None:
                    xy = interp_trace(tt, ts[:, 0:2, tongue_ix], time)
                    tongue_speed[:, k] = speed(xy[:, 0], xy[:, 1])
```

iii. The comment cites the Methods statement that the tongue is tracked from both views while the paws use only the bottom view; the agent chose the side view's `'tongue'` (the authors' own `params.traj_features` lists `'tongue'` first for camera 0, and `findPosition.m`/`getKinematics.m` operate one view at a time, producing per-view features). It had confirmed the feature names of both cameras (`view1: ['tongue', 'left_tongue', 'right_tongue', 'jaw', …]`, `view2: ['top_tongue', …, 'top_paw', 'bottom_paw', …]`) and that the side-view tongue is NaN in ~92% of frames with mean likelihood 0.11, i.e. only visible during licks.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Three steps, and deliberately **no** smoothing and **no** gap filling. (1) The x and y traces are linearly interpolated from the corrected frame times onto the 500 bin centres (`interp_trace`, NaN outside the frames' coverage) — because DeepLabCut's x/y are already NaN wherever the tongue was not detected, those NaNs propagate through the interpolation and mark the invisible bins. (2) Speed is the magnitude of the two first-order derivatives, computed with central differences where both neighbours exist and one-sided differences at the edges of each visible run (`speed` / `deriv`, NaN-aware), in pixels per bin. (3) The session's finite values are split at their 50th percentile. No per-camera normalisation is needed because only one camera is used.

ii.
```python
def speed(x, y):
    """Speed from a 2-D position trace, NaN-aware (central / one-sided differences)."""
    def deriv(v):
        d = np.full(v.shape, np.nan)
        fwd = np.full(v.shape, np.nan); bwd = np.full(v.shape, np.nan)
        fwd[:-1] = v[1:] - v[:-1]
        bwd[1:] = v[1:] - v[:-1]
        both = np.isfinite(fwd) & np.isfinite(bwd)
        d[both] = (fwd[both] + bwd[both]) / 2.0
        only_f = np.isfinite(fwd) & ~np.isfinite(bwd); d[only_f] = fwd[only_f]
        only_b = np.isfinite(bwd) & ~np.isfinite(fwd); d[only_b] = bwd[only_b]
        return d
    return np.sqrt(deriv(x) ** 2 + deriv(y) ** 2)
```
```python
                    xy = interp_trace(tt, ts[:, 0:2, tongue_ix], time)
                    tongue_speed[:, k] = speed(xy[:, 0], xy[:, 1])
```
Note the contrast with the paw branch, which *does* fill interior NaNs:
```python
    """Nearest-neighbour fill of NaNs that lie between valid samples.

    The authors fill every missing sample with `fillmissing(...,'nearest')`; here
    the fill is restricted to gaps inside the covered range ...
    """
```

iii. This follows `findPosition.m` (`ts = traj(trix).ts(:,1:2,featix); if ~contains(feat,'tongue'), ts = mySmooth(ts, 1, 'reflect'); end; interp1(traj.frameTimes - vidshift - alignEv, ...)`) and the Methods sentence "Missing values were filled in with the nearest available value for all features, **except for the tongue**" — hence no fill on the tongue branch, so that untracked bins can carry the `not_visible` class. The paper defines velocity as "the first-order derivative of the position vector", which is what `speed` computes.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single per-session threshold: the 50th percentile of all finite tongue-speed values in that session (all trials × all bins pooled). Bins at or above it get 1, below get 0, and bins where the speed is NaN — the tongue was not visible, or the frame times do not cover that bin — get 2. `output_values[3] = ['below_median', 'above_median', 'not_visible']`. The threshold is recorded per session in `metadata['session_info']` (e.g. 12.42 for JEB13 2022-09-24). Dataset-wide the classes come out 0.040 / 0.040 / 0.920.

ii.
```python
    def discretize(x):
        """0 below the session median, 1 at/above it, 2 where the feature is absent."""
        out = np.full(x.shape, 2, dtype=np.int8)
        ok = np.isfinite(x)
        if ok.any():
            thresh = np.percentile(x[ok], 50)
            out[ok] = (x[ok] >= thresh).astype(np.int8)
        else:
            thresh = np.nan
        return out, thresh

    tongue_cls, tongue_thresh = discretize(tongue_speed)
```

iii. Directly from the instructions ("discretized with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile, 2: not visible"). The agent verified the class was behaving sensibly by printing visibility against time for one session: tongue visible in 0% of trials before the go cue, rising to 21% at +0.1 s and 41% at +0.5 s.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Two corrections, then the same grid. (1) The camera clock leads the Bpod clock; the per-session offset is the mode of `sglx.bitcode.bitstart / sglx.fs` minus the mode of `bp.ev.bitStart`, i.e. a direct port of `findVideoOffset.m`, computed once per session. (2) Per trial, `frameTimes − vidshift − goCue[trial]` puts each frame in seconds from that trial's go cue. The x/y traces are then interpolated onto exactly the bin centres used for the spikes, so tongue bin *k* and neural bin *k* are the same interval.

ii.
```python
def video_shift(obj):
    """funcs/findVideoOffset.m: offset between the ephys and video clocks (s)."""
    def mode_(v):
        v = np.asarray(v, dtype=float).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0.0
        vals, counts = np.unique(v, return_counts=True)
        return float(vals[np.argmax(counts)])
    try:
        return mode_(obj['sglx']['bitcode']['bitstart']) / float(obj['sglx']['fs']) \
            - mode_(obj['bp']['ev']['bitStart'])
    except (KeyError, TypeError):
        return 0.5
```
```python
def trial_video_times(trial, vidshift, align_time):
    """Frame times of one trial on the bpod clock, relative to the align event."""
    ...
    return ft - vidshift - align_time, ts
```
```python
    vidshift = video_shift(obj)
```

iii. Final summary: the kinematic streams "are interpolated onto the bin centers with the authors' video clock offset (`findVideoOffset`)". The agent read `findVideoOffset.m` and `findPosition.m` and reproduced both the mode-based offset and the `interp1` onto the analysis time axis; it checked the offset numerically on one session against a median-based variant before settling on the mode.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` tracking, **bottom camera only** (`traj{2}`, index 1), features `'top_paw'` and `'bottom_paw'` — both forepaws. Same `ts` / `frameTimes` fields, same session video offset and go cue.

ii.
```python
PAW_FEATURES = [('top_paw', 1), ('bottom_paw', 1)]
```
```python
            # paws (bottom view): mean speed of the two tracked paws
            if trix < len(views[1]):
                tt, ts = trial_video_times(views[1][trix], vidshift, gocue[trix])
```

iii. The code comment quotes the Methods directly: "the paws were tracked using only the bottom view". The agent had confirmed `top_paw` and `bottom_paw` are both present in camera 2's `featNames` and that both have a NaN fraction of ~0.00 (i.e. both are tracked essentially every frame), which is why it used both rather than picking one.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Four steps. (1) Each paw's x and y are interpolated from corrected frame times onto the 500 bin centres. (2) Interior NaNs — gaps between two valid samples — are filled with the nearest valid value (`fill_interior_nans`), following the Methods' `fillmissing(..., 'nearest')`; the fill is deliberately restricted to the covered range so that bins the camera never reached stay NaN and can be labelled `not_visible`. (3) Speed is computed exactly as for the tongue (NaN-aware central/one-sided differences, magnitude of the two derivatives), in pixels per bin, with no normalisation. (4) The two paws' speeds are averaged with `np.nanmean`, so a bin with one paw tracked still gets a value.

ii.
```python
def fill_interior_nans(y):
    """Nearest-neighbour fill of NaNs that lie between valid samples. ..."""
    y = y.copy()
    good = np.isfinite(y)
    if not good.any():
        return y
    idx = np.flatnonzero(good)
    lo, hi = idx[0], idx[-1]
    seg = y[lo:hi + 1]
    bad = ~np.isfinite(seg)
    if bad.any():
        pos = np.flatnonzero(~bad)
        nearest = pos[np.argmin(np.abs(np.flatnonzero(bad)[:, None] - pos[None, :]), axis=1)]
        seg[bad] = seg[nearest]
        y[lo:hi + 1] = seg
    return y
```
```python
                    sp = []
                    for ix, _ in paw_ix:
                        if ix is None:
                            continue
                        xy = interp_trace(tt, ts[:, 0:2, ix], time)
                        xy = np.stack([fill_interior_nans(xy[:, 0]),
                                       fill_interior_nans(xy[:, 1])], axis=1)
                        sp.append(speed(xy[:, 0], xy[:, 1]))
                    if sp:
                        paw_speed[:, k] = np.nanmean(np.stack(sp, axis=1), axis=1)
```

iii. The `fill_interior_nans` docstring is the justification: the authors fill missing samples with the nearest available value for every feature except the tongue, but filling *outside* the camera's coverage would fabricate a value where the instructions ask for a `not_visible` class. Averaging the two paws is the natural reading of the Methods, which treat "the paws" as one tracked group on the bottom view; the agent had checked that both are reliably tracked.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: the same `discretize` helper, per session, splitting all finite values at their 50th percentile, with NaN bins → 2. `output_values[4] = ['below_median', 'above_median', 'not_visible']`. Thresholds are recorded per session (0.20–0.64 px/bin). Dataset-wide: 0.477 / 0.477 / 0.045.

ii.
```python
    paw_cls, paw_thresh = discretize(paw_speed)
    ...
        'paw_velocity_threshold': float(paw_thresh),
```

iii. Directly from the instructions' 50th-percentile per-session rule. The residual 4.5% `not_visible` comes from bins outside the camera's coverage rather than from tracking failures, since both paws are tracked in essentially every frame.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue — same per-session `video_shift`, same `frameTimes − vidshift − goCue[trial]`, same interpolation onto the shared bin centres — except that the frame times are taken from the **bottom** camera's own per-trial entry rather than the side camera's, because the two views can record different numbers of frames in a trial.

ii.
```python
            if trix < len(views[1]):
                tt, ts = trial_video_times(views[1][trix], vidshift, gocue[trix])
                if tt is not None:
                    ...
                        xy = interp_trace(tt, ts[:, 0:2, ix], time)
```
```python
    ft = np.asarray(ft, dtype=float).reshape(-1)
    if ft.size != ts.shape[0]:
        n = min(ft.size, ts.shape[0])
        ft, ts = ft[:n], ts[:n]
```

iii. Same offset and grid as every other stream, so the paw needs no separate treatment; using each camera's own `frameTimes` (and truncating to the shorter of `frameTimes` / `ts`) is defensive handling of the per-view frame-count differences the agent found while surveying the `traj` structure.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside the data structure, variable `me`, which holds one trace per trial with one value per camera frame. The wrapper is unwrapped up to two levels (`me` → `me['data']` → `me['data']['data']`), because the files come in three layouts. Frame times come from the **side** camera (`traj{1}`), whose frames the motion energy is computed on.

ii.
```python
    me_path = os.path.join(folder, 'motionEnergy_%s_%s.mat' % (anm, date))
    me_data = None
    if os.path.exists(me_path):
        me = loadmat_var(me_path, 'me')
        me_data = me.get('data') if isinstance(me, dict) else me
        if isinstance(me_data, dict):
            me_data = me_data.get('data')
```
```python
            tt, _ = trial_video_times(traj[0][trix], vidshift, gocue[trix])
```

iii. Not argued at length; the agent inspected one file (`me` → `{'data', 'moveThresh'}`, `data` a 416-element cell of 4677-sample traces, exactly one value per frame of that trial's side-camera video) and noted the standalone file exists for all 44 sessions. `loadMotionEnergy.m` contains the same double-wrap guard.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Almost none: the value is already one scalar per frame (the paper computes it per pixel as the difference of medians over the next and previous five frames, then reduces each frame to its 99th percentile across pixels), so there is nothing to differentiate or combine. The trace is truncated to the shorter of `frameTimes`/values, linearly interpolated onto the 500 bin centres, interior NaNs are nearest-filled, and the result is split at the session median.

ii.
```python
            y = np.asarray(me_data[trix], dtype=float).reshape(-1)
            if y.size == 0:
                continue
            tt, _ = trial_video_times(traj[0][trix], vidshift, gocue[trix])
            if tt is None:
                continue
            n = min(tt.size, y.size)
            me_trace[:, k] = fill_interior_nans(interp_trace(tt[:n], y[:n], time))
```

iii. Implicit: the spatial reduction has already been done upstream by the authors, so the trace is used as delivered. The agent noted the file also carries the authors' manual `moveThresh` but did not use it, since the instructions prescribe a 50th-percentile split instead.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same `discretize` helper, per session: 50th percentile of all finite values, `>=` → 1, `<` → 0, NaN → 2. `output_values[5] = ['below_median', 'above_median', 'no_video']` — note the trailing label differs from the two kinematic outputs, matching the instructions' "2: no video". Thresholds range 7.5–50.1 across sessions. Dataset-wide: 0.480 / 0.481 / 0.039.

ii.
```python
    me_cls, me_thresh = discretize(me_trace)
```
```python
OUTPUT_VALUES = [
    ...
    ['below_median', 'above_median', 'not_visible'],
    ['below_median', 'above_median', 'not_visible'],
    ['below_median', 'above_median', 'no_video'],
]
```

iii. Directly from the instructions. The agent also checked the class behaves sensibly in time (`me_high` 0.34–0.59 during the delay, 0.97–0.98 just after the go cue).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same session `video_shift` and same grid as the tracking, using the **side** camera's per-trial `frameTimes` minus the offset minus that trial's go cue, then interpolation onto the bin centres. `metadata['session_info']` records `frac_trials_with_video = 1.0` for every session, i.e. every kept trial has at least some motion-energy coverage in the window.

ii.
```python
            tt, _ = trial_video_times(traj[0][trix], vidshift, gocue[trix])
            if tt is None:
                continue
            n = min(tt.size, y.size)
            me_trace[:, k] = fill_interior_nans(interp_trace(tt[:n], y[:n], time))
```

iii. Motion energy has exactly one value per side-camera frame, so it inherits that camera's timing; the agent verified the per-trial lengths agree (4677 motion-energy samples vs 4677 `frameTimes` on the example trial).

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Eight distinct cases, all handled by keeping the trial/session and marking or substituting rather than dropping:
- **Untracked DeepLabCut frames** — x/y are already NaN; NaN propagates through interpolation and the bin becomes `not_visible` (class 2). For the paws only, interior gaps are nearest-filled first.
- **Bins outside the camera's coverage** — `interp_trace` returns NaN outside `[t_src[0], t_src[-1]]`, so those bins also become class 2 rather than being extrapolated.
- **NaN `NdroppedFrames`** — the trial's video is skipped entirely (its kinematic bins become class 2), matching `findPosition.m`.
- **Missing / all-NaN `frameTimes`** — a nominal 400 Hz frame clock is synthesised and `vidshift` is *hard-coded to 0.5 s*, following `findPosition.m`'s fallback.
- **Mismatched lengths** — `frameTimes` vs `ts`, and motion-energy trace vs frame times, are both truncated to the shorter.
- **Missing `obj.ex`** (EKH1, EKH3, JEB7, JGR2, JGR3) — the probe location falls back to `ALM`; an unrecognised location string also maps to `ALM`.
- **Missing `bp.stim`** — treated as no photostimulation.
- **Per-trial field length ≠ `Ntrials`** — `np.resize` to `Ntrials`; NaNs in flags become `False`.

ii.
```python
    if ft is None or np.size(ft) == 0 or not np.any(np.isfinite(np.asarray(ft, float))):
        # findPosition.m falls back to a nominal 400 Hz frame clock
        ft = (np.arange(ts.shape[0]) + 1) / 400.0
        vidshift = 0.5
    ...
    nd = trial.get('NdroppedFrames')
    if nd is not None and np.size(nd) == 1 and not np.isfinite(float(np.asarray(nd, float))):
        return None, None      # findPosition.m skips trials with NaN NdroppedFrames
```
```python
    """... A few of the older sessions (EKH1, EKH3, JEB7, JGR2, JGR3) have no `ex` field
    at all; they are ALM recordings ... so ALM is the fallback label.
    """
    locs = ['ALM'] * nprobes
```
```python
    stim = as_bool(bp['stim']['enable'], ntrials_all) if 'stim' in bp else np.zeros(ntrials_all, bool)
```

iii. The `not_visible` / `no_video` classes exist precisely so that missing camera data can be represented honestly, and the `fill_interior_nans` docstring explains why filling stops at the edges of the covered range: "the fill is restricted to gaps inside the covered range so that timepoints the camera never saw stay NaN and can be labelled 'not visible' / 'no video'". The `ALM` fallback is justified from the paper ("we recorded activity extracellularly in the ALM") plus the fact that those sessions are loaded by the authors' ALM loaders. The 400 Hz / 0.5 s fallback cites `findPosition.m`, though the 0.5 s constant itself is not derived from anything in the data.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the `.mat` files dominates. `matio.load_obj` walks the whole `obj` tree and materialises it (the files are 100–300 MB each, 44 of them); a single session took ~6 s of which most was loading, and the full conversion ran ~8–9 minutes. Second is the neural inner loop: for every kept cluster (~2,500 across the dataset) and every kept trial (~300 per session) it calls `np.histogram` once — roughly 800,000 individual histogram calls. Third, `my_smooth` uses `np.apply_along_axis` with a Python-level `np.convolve` per trial column.

ii.
```python
                for k in range(trials.size):
                    if stops[k] > starts[k]:
                        aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
                        dat[:, k] = bin_spikes(aligned, edges)
            dat = my_smooth(dat / DT)
```
```python
    out = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, xp)
```

iii. Not analysed in the trajectory; the agent ran the conversion in the background and polled it, and reduced load cost pre-emptively by giving `matio` a `SKIP_FIELDS` list ("Fields that are large and never used by the conversion. Skipping them keeps a 300 MB session object from being materialised in full"), which excludes `spkWavs`.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. (1) The per-trial `np.histogram` loop inside the per-cluster loop could be a single `np.histogram2d` over `(spike_trial, spike_time)` for the whole cluster at once, eliminating the inner Python loop entirely. (2) `my_smooth`'s `np.apply_along_axis` could be one `scipy.ndimage.convolve1d` / `scipy.signal.fftconvolve` call on the whole `(T, ntrials)` array. (3) The per-trial video loop calls `trial_video_times`, `interp_trace`, `fill_interior_nans` and `speed` once per trial per feature; the interpolation and differentiation of all trials of one feature could be batched, although the per-trial frame counts differ so some ragged handling would remain. The `feature_index` helper also scans trials linearly to find a feature name that is identical in every trial.

ii.
```python
            dat = np.zeros((T, trials.size))
            if utrial.size:
                order = np.argsort(utrial, kind='stable')
                ...
                for k in range(trials.size):
                    if stops[k] > starts[k]:
                        ...
                        dat[:, k] = bin_spikes(aligned, edges)
```
```python
        for k, trix in enumerate(trials):
            # tongue (side view) ...
            # paws (bottom view) ...
```

iii. Not discussed; the agent did partially optimise this loop by sorting spikes once per cluster and using `searchsorted` to get each trial's slice, rather than a boolean mask per trial.

## 11-c. What processing does the code repeat multiple times?

i. Several small repeats. (1) `trial_video_times` is called up to three times per trial — once for the side camera (tongue), once for the bottom camera (paws), once again for the side camera (motion energy) — so the side camera's per-trial entry is parsed and its frame times recomputed twice. (2) Every non-rejected cluster is binned *and* smoothed before the 1 Hz cut is applied, so the smoothing of the ~57 units that fail the cut is wasted. (3) `np.flatnonzero(use)` is computed twice in adjacent lines. (4) `fill_interior_nans` is applied to the motion-energy trace, which in practice contains no interior NaNs. Things that are *not* repeated: each file is read once, `video_shift` is computed once per session (not per trial), and the bin edges/centres are built once per session and reused for every trial and stream.

ii.
```python
                tt, ts = trial_video_times(views[TONGUE_FEATURE[1]][trix], vidshift, gocue[trix])
                ...
                tt, ts = trial_video_times(views[1][trix], vidshift, gocue[trix])
                ...
            tt, _ = trial_video_times(traj[0][trix], vidshift, gocue[trix])
```
```python
            dat = my_smooth(dat / DT)
            rates.append(dat.astype(np.float32))
    ...
    mean_fr = rates.mean(axis=(0, 2))
    use = mean_fr > LOW_FR
    ...
    regions = [regions[i] for i in np.flatnonzero(use)]
    qualities = [qualities[i] for i in np.flatnonzero(use)]
```
```python
    vidshift = video_shift(obj)   # once per session
```

iii. Not discussed in the trajectory.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest. (1) The `quality` strings of all surviving clusters are accumulated in `qualities` solely to compute `n_single_units`, a metadata statistic the decoder never reads. (2) The smoothing of units that are then dropped by the 1 Hz cut (11-c). (3) The `aw == 2` branch in the autowater decoding is dead code — `autowater` is 0/1 in all 44 sessions. (4) `L` is read from `bp` although `~R` would do, and `no` although "neither hit nor miss" would do. (5) `fill_interior_nans` on motion energy. (6) `probe_locations` parses `obj.ex.probe.loc` for every probe in the file including ones not selected. (7) The `matio` reader still materialises whole sub-trees the conversion never touches (`sglx` index arrays, `clu.tm`, `obj.trials`, `obj.me`, and the tracked features other than `tongue`, `top_paw`, `bottom_paw`), although `SKIP_FIELDS` already removes the largest of them (`spkWavs`). Everything computed after loading — rates, the six outputs, the input — ends up in the pickle.

ii.
```python
            rates.append(dat.astype(np.float32))
            regions.append(region)
            qualities.append(quality)
    ...
        # single units as counted in Scripts/EDFigure 2/EDFigure2a_Left.m
        'n_single_units': int(sum(q.lower() in ('fair', 'good', 'great', 'excellent')
                                  for q in qualities)),
```
```python
    # older sessions code autowater as 1 = off / 2 = on, newer ones as 0/1
    autowater = (aw == 2) if np.nanmax(aw) > 1 else (np.nan_to_num(aw) > 0.5)
```
```python
SKIP_FIELDS = ('spkWavs', 'sglxfns', 'fnEpochs', 'svpth', 'sglxpth')
```

iii. The single-unit count was added deliberately as a cross-check against the paper's Extended Data Fig. 2a per-session statistics, after the agent read `EDFigure2a_Left.m` and corrected its definition of "single unit" to the authors' `contains('air') || contains('ood') || contains('xcellent') || contains('reat')`. The `SKIP_FIELDS` list is explained in `matio.py` as a memory optimisation.
