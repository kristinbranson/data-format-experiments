# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data directories. It hard-codes a 44-entry table `SESSION_META`, each entry being `(animal, date, probes, folder)`, transcribed from the authors' `load<ANM>_ALMVideo.m` loading scripts, and iterates over it. For each entry it builds two paths, `data_structure_<anm>_<date>.mat` and `motionEnergy_<anm>_<date>.mat`, inside the folder named in the table (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`). The `.mat` files come in two MATLAB formats, so the AI wrote a `SessionData` class that sniffs the file header and then exposes one uniform accessor API over either an `h5py.File` (v7.3) or a `scipy.io.loadmat` result (v5). The v7.3 path is *lazy*: only the fields actually requested (`bp`, `clu[probe]`, `traj[cam][trial]`, `sglx`) are read out of the HDF5 file; the v5 path reads the whole file at once. Motion energy is loaded by a separate `load_motion_energy` that unwraps three different on-disk layouts. Sessions whose data file is missing are skipped with a warning.

ii.
```python
SESSION_META = [
    # Ephys_Behavior (two-context paradigm, 25 sessions)
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]
```

```python
def detect_mat_version(path):
    with open(path, 'rb') as f:
        header = f.read(16).decode('ascii', errors='replace')
    return '7.3' if '7.3' in header else '5.0'


class SessionData:
    """Unified interface for accessing session data from either MAT format."""

    def __init__(self, data_path):
        self.version = detect_mat_version(data_path)
        self._f = None

        if self.version == '7.3':
            self._f = h5py.File(data_path, 'r')
            self._obj = self._f['obj']
        else:
            d = sio.loadmat(data_path, squeeze_me=False)
            self._obj_v5 = d['obj'][0, 0]  # unwrap (1,1) wrapper
```

```python
for animal, date, probes, data_dir in SESSION_META:
    data_path = os.path.join(DATA_ROOT, data_dir, f'data_structure_{animal}_{date}.mat')
    me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{animal}_{date}.mat')
    if not os.path.exists(data_path):
        print(f"  WARNING: Data file not found, skipping")
        continue
    sess = SessionData(data_path)
```

```python
def load_motion_energy(me_path):
    """Load motion energy from .mat file. Returns list of per-trial arrays."""
    d = sio.loadmat(me_path, squeeze_me=False)
    me_raw = d['me']
    # Format 1: me is (n_trials, 1) cell array directly (no struct wrapper)
    ...
    # Format 2: me is struct with .data and .moveThresh
    ...
    # Handle nested struct: me.data might itself be a struct with .data
```

iii. From the trajectory, the AI read the authors' `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` files and used them as the definitive session list, explicitly noting that sessions commented out there (`JEB23_2023-10-20`) and sessions present on disk but absent from the scripts (`JEB24_2023-10-03`, `JEB24_2023-10-04`, which also lack motion-energy files) must be excluded: *"the loading script only includes sessions from 2023-10-23 through 2023-11-03 even though earlier files exist … so I'll restrict my session list to the 8 sessions actually referenced in the loading script."* It discovered the two MATLAB formats only after its first version crashed (*"Some mat files might be v5 format (not HDF5/v7.3)"*) and then refactored into the unified reader: *"I need to detect file format and branch my reader accordingly - h5py for v7.3, scipy.io.loadmat for v5 - behind a unified interface."* The three motion-energy layouts were likewise found empirically (*"the JEB23 ME file has a different format … JEB15 has yet another variant with a nested struct"*).

## 1-b. How are the data split into subjects?

i. The subject is the animal string that is already the first element of each `SESSION_META` tuple (equivalently the prefix of the filename). `subjects` is accumulated in order of first appearance rather than sorted, and `subject_idx` is that animal's position in the accumulated list. This yields 14 subjects over 44 sessions, with the same sessions-per-subject counts as the reference (JEB24: 8, JEB23: 7, JEB13: 5, JEB14/JEB15/JEB19: 4 each, etc.).

ii.
```python
if animal not in subjects_set:
    subjects_set.append(animal)
all_subject_idx.append(subjects_set.index(animal))
```

```python
'subjects': subjects_set,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. Not explicitly argued in the trajectory. The animal identity is carried in the hard-coded session table, which the AI took from the authors' per-animal loading scripts, so the animal never has to be recovered from inside the file (`obj.meta.anm` is missing in several sessions).

## 1-c. How are the data split into sessions?

i. One entry of `SESSION_META` = one file on disk = one session = one element of `neural` / `input` / `output` / `brain_region_idx` / `subject_idx`. The folder (fixed-delay vs randomized-delay task) is stored in the table, so the two paradigms are treated uniformly and pooled into one 44-session dataset (25 fixed-delay + 19 randomized-delay). Sessions are additionally dropped if fewer than `MIN_UNITS = 10` units survive quality filtering, if fewer than 10 survive the firing-rate filter, or if fewer than 2 trials survive; in practice none of these triggered (minimum 17 units, minimum 193 trials).

ii.
```python
MIN_UNITS = 10        # paper: "at least 10 units"
...
if len(all_units) < MIN_UNITS:
    print(f"  WARNING: Too few units after quality filter ({len(all_units)}), skipping")
    sess.close()
    continue
...
if np.sum(fr_mask) < MIN_UNITS:
    print(f"  WARNING: Too few units after FR filter ({np.sum(fr_mask)}), skipping")
    sess.close()
    continue
```

iii. The AI reasoned that the randomized-delay sessions should be pooled with the two-context sessions even though they contain no WC trials: *"Since randomized delay sessions are DR-only, their trials would all just carry the DR context label if included, which seems workable."* The 10-unit session criterion was taken from the paper: *"Sessions needed at least 10 units to be included."*

## 1-d. How are the data split into trials?

i. Trials are the rows of the Bpod table `obj.bp`. `n_trials_total = obj.bp.Ntrials` is read once, and every per-trial field (`hit`, `miss`, `no`, `R`, `L`, `autowater`, `early`, `stim.enable`, `ev.goCue`) is flattened and sliced to `[:n_trials_total]`, because several fields are stored longer than the trial count. Spikes carry their own 1-based trial index (`clu.trial`) and a within-trial time (`clu.trialtm`), and camera trajectories are stored one entry per trial, so no trial boundaries have to be reconstructed. There is exactly one `ev.goCue` per trial, which is the alignment anchor.

ii.
```python
n_trials_total = sess.get_n_trials()

hit = sess.get_bp_field('hit')[:n_trials_total]
miss = sess.get_bp_field('miss')[:n_trials_total]
no = sess.get_bp_field('no')[:n_trials_total]
R = sess.get_bp_field('R')[:n_trials_total]
L = sess.get_bp_field('L')[:n_trials_total]
autowater = sess.get_bp_field('autowater')[:n_trials_total]
early = sess.get_bp_field('early')[:n_trials_total]
stim_enable = sess.get_stim_enable()[:n_trials_total]
go_cue = sess.get_event_field('goCue')[:n_trials_total]
```

```python
def get_n_trials(self):
    if self.version == '7.3':
        return int(np.array(self._obj['bp']['Ntrials']).flatten()[0])
    bp = self._v5_bp()
    val = self._v5_scalar(bp['Ntrials'])
    return int(val)
```

iii. Not separately argued; the Bpod table defines trials directly. The AI's reasoning notes the per-trial structure of `bp` (`hit/miss/no`, `R/L`, `autowater`) and of `traj` (one struct per trial) when planning the extraction.

## 1-e. How are trials filtered based on quality controls?

i. Three filters.
  1. **Photostimulation**: `bp.stim.enable == 0`.
  2. **Early lick**: `bp.early == 0`.
  3. **Outcome flag present**: `hit | miss | no`. This third term is a tautology in this dataset (the three flags are mutually exclusive and sum to 1 on every trial), so it removes nothing.
  These are combined into `valid_mask` before anything is computed.
  4. **Recording ended**: after spike binning, any surviving trial whose entire (n_units x n_bins) matrix is exactly zero is dropped. This was added after the format checker reported 61 all-zero trials at the ends of `JEB24_2023-10-23` and `JEB24_2023-11-03`, where the behaviour outlasted the probe.

  The result is 13,762 trials over 44 sessions, and the per-session trial counts match the human reference session-for-session.

ii.
```python
# ---- Filter trials ----
valid_mask = (stim_enable == 0) & (early == 0) & ((hit == 1) | (miss == 1) | (no == 1))
valid_idx = np.where(valid_mask)[0]

if len(valid_idx) < 2:
    print(f"  WARNING: Too few valid trials ({len(valid_idx)}), skipping")
    sess.close()
    continue
```

```python
# Filter out valid trials with all-zero neural data (recording ended)
has_spikes = np.array([np.any(trialdat[:, :, ti] != 0) for ti in valid_idx])
valid_idx = valid_idx[has_spikes]
if len(valid_idx) < 2:
    print(f"  WARNING: Too few trials with neural data, skipping")
    sess.close()
    continue
```

iii. Early-lick and photostim exclusion follow the paper and the authors' code: *"I'll also exclude stim-enabled and early trials."* The all-zero filter was added reactively after inspection: *"for trials occurring late in a session, the recording may have already ended by that point, leaving an empty window with no spikes. Since all-zero trials aren't useful for the decoder, I should add a filter to exclude trials where every neuron is silent across the whole window."* The AI considered using the `trials.sglx` field instead but chose the simpler empirical test: *"the simpler approach: just exclude trials with all-zero neural data. This is justified because these trials have no neural recordings."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the probe(s) listed for that session in `SESSION_META`. For each cluster the AI reads `trial` (1-based trial index of each spike), `trialtm` (spike time relative to that trial's start), `quality` (free-text curation label) and `tm` (session-clock spike time). `tm` is read but never used. The other input is `obj.bp.ev.goCue`, which sets the alignment. For two-probe sessions the clusters of both probes are concatenated into one population.

ii.
```python
for probe_num in probes:
    probe_idx = probe_num - 1
    units = sess.get_units_for_probe(probe_idx)
    # Quality filter
    units = [u for u in units
             if u['quality'].lower() not in {q.lower() for q in EXCLUDED_QUALITIES}
             and u['quality'] != '']
    all_units.extend(units)
```

```python
tm = np.array(self._f[probe_group['tm'][i, 0]]).flatten()
trial = np.array(self._f[probe_group['trial'][i, 0]]).flatten().astype(int)
trialtm = np.array(self._f[probe_group['trialtm'][i, 0]]).flatten()

units.append({
    'tm': tm, 'trial': trial, 'trialtm': trialtm, 'quality': quality
})
```

iii. The AI read the authors' `findClusters.m` / `getSeq.m` / `alignSpikes.m` and reproduced their inputs. The probe choice per session comes from the same `load<ANM>_ALMVideo.m` scripts as the session list; the AI verified empirically that some sessions carry two probes (`clu shape: (2, 1)`).

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, counted into 1000 x 5 ms bins spanning [-2.5, 2.5] s, divided by the bin width to give spikes/s, and then smoothed along time with a **causal** Gaussian. The smoother, `causal_gaussian_smooth`, is a direct Python port of the authors' `mySmooth.m`: it reflects the first `N = 15` samples across the start of the trace, builds a MATLAB `gausswin(15)` kernel (alpha = 2.5), zeroes the first `floor(N/2) = 7` taps to make it causal, normalises it, convolves with `mode='same'`, and trims the reflected prefix. No normalisation, baseline subtraction, or z-scoring is applied; the stored values are firing rates in Hz (`float64`). Units from both probes are concatenated into one population.

ii.
```python
def causal_gaussian_smooth(x, window, bctype='reflect'):
    """Causal Gaussian smoothing matching mySmooth.m."""
    ...
    if bctype == 'reflect':
        x_filt = np.concatenate([x[:window], x], axis=0)
        trim = window
    ...
    # Causal Gaussian kernel (matching MATLAB gausswin with alpha=2.5)
    n = np.arange(window)
    alpha = 2.5
    center = (window - 1) / 2
    kern = np.exp(-0.5 * ((n - center) / (center / alpha)) ** 2)
    kern[:int(window // 2)] = 0  # causal
    kern = kern / kern.sum()

    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
    out = out[trim:]
```

```python
for i, unit in enumerate(units):
    aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]
    for j in range(n_trials):
        mask = unit['trial'] == (j + 1)
        if not np.any(mask):
            continue
        counts = np.histogram(aligned[mask], bins=edges)[0]
        fr = counts.astype(float) / DT
        trialdat[:, i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, 'reflect')
```

iii. The AI read `/app/code/utils/mySmooth.m` in full (trajectory step 50) and ported it line for line, including the `kern(1:floor(numel(kern)/2)) = 0; %causal` step and the `trim = N + 1` (1-based) reflect trim. Its summary of parameters was: *"dt = 1/200 = 0.005 s (5ms bins), tmin = -2.5, tmax = 2.5, smooth window = 15 samples, boundary condition = 'reflect', alignEvent = 'goCue'"*, all taken from the authors' `getDefaultParams.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters plus a session-level one.
  1. **Curation label**: the free-text `clu.quality` is lower-cased and compared against `EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}` — exactly the list in the authors' `findClusters.m`. Units whose label is the empty string are *also* dropped (an extra condition not present in the authors' code; in the sessions checked no cluster carries an empty label, so it is inert here). Everything else is kept, including `Poor`, `Fair` and `Multi`.
  2. **Firing rate**: after binning and smoothing, a unit is kept if its mean rate over the whole (bins x trials) array exceeds `LOW_FR = 1.0` Hz. Note that this mean is taken over *all* trials in the file, including the photostim, early-lick and post-recording trials that were already excluded from `valid_idx`.
  3. **Session**: at least 10 units must survive each of the two filters.

  This keeps 2,457 units across 44 sessions (17-141 per session). The human reference keeps 1,954 (15-110); the whole difference is that the reference additionally drops `poor`-labelled units.

ii.
```python
LOW_FR = 1.0          # Hz (paper methods: "firing rates exceeding 1 Hz")
MIN_UNITS = 10        # paper: "at least 10 units"
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
```

```python
units = [u for u in units
         if u['quality'].lower() not in {q.lower() for q in EXCLUDED_QUALITIES}
         and u['quality'] != '']
```

```python
# ---- Remove low FR units ----
mean_frs = np.mean(trialdat, axis=(0, 2))
fr_mask = mean_frs > LOW_FR
```

iii. The AI enumerated the quality labels present in the data before choosing (`{'Poor', 'Fair', 'Good', 'Great', 'Multi', 'Excellent'}` in one session, all-lowercase in another) and therefore matched case-insensitively; it kept the authors' exclusion list verbatim. On the rate threshold it explicitly weighed the two sources and chose the paper's: *"There's some ambiguity on the firing rate threshold — getDefaultParams uses 0.5 Hz while the methods text and tutorial reference 1 Hz — I'll go with 1 Hz since it matches the paper's stated methods."* The 10-unit session minimum comes from the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A single subtraction. `clu.trialtm` is already on the behaviour clock and already relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm - goCue[trial - 1]` (the `-1` converting MATLAB's 1-based trial numbers) gives each spike's time in seconds from the go cue. No interpolation or extra offset. Spikes falling outside [-2.5, 2.5] s fall off the ends of the histogram and are discarded. The camera streams need an additional clock correction (see 7-d); the spikes do not.

ii.
```python
aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]
for j in range(n_trials):
    mask = unit['trial'] == (j + 1)
    ...
    counts = np.histogram(aligned[mask], bins=edges)[0]
```

iii. This reproduces the authors' `alignSpikes.m` with `params.alignEvent = 'goCue'`, which the AI read. Its plan: *"neural spikes get binned at 5ms resolution around the go cue."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms (`DT = 1/200`), 1000 non-overlapping bins spanning -2.5 s to +2.5 s around the go cue. The grid is constructed once per run and reused for every trial, every session, and every data stream (neural, input, tongue, paw, motion energy), so all streams share one time axis. No rebinning or resampling of the neural data is performed — spikes are histogrammed directly onto this grid. `metadata['time_bin_size']` is reported as 5.0 ms. The camera streams, which are sampled at 400 Hz, are resampled onto this grid by linear interpolation (see 7-d).

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 200         # 5 ms
```

```python
n_bins = int(round((TMAX - TMIN) / DT))
edges = np.linspace(TMIN, TMAX, n_bins + 1)
time_axis = edges[:-1] + DT / 2
```

```python
'time_bin_size': DT * 1000,
'off_start': TMIN,
'off_end': TMAX,
```

iii. All three values are the authors' defaults, which the AI read out of `getDefaultParams.m`. It noted the ambiguity against one of the tutorials and resolved it in favour of the params file: *"getDefaultParams uses 5ms bins while the tutorial example uses 10ms — so I need to figure out which convention the paper actually follows"*, settling on *"dt = 1/200 = 0.005 s (5ms bins), tmin = -2.5, tmax = 2.5"*.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. None directly — this input is defined by the conversion itself. It is the vector of bin centres of the go-cue-aligned window, which depends on the raw data only through `bp.ev.goCue`, the event the window is centred on. The same 1000-element vector is stored for every trial of every session; its range is [-2.4975, 2.4975] s, identical to the reference.

ii.
```python
edges = np.linspace(TMIN, TMAX, n_bins + 1)
time_axis = edges[:-1] + DT / 2
```

```python
'input_names': ['time_from_go_cue'],
```

iii. The AI treated it as given by the decoder specification: *"the decoder's inputs (time from go cue) … with time from go cue as the continuous input axis."*

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The bin-centre vector is reshaped to `(1, 1000)` and copied once per trial; it is stored as `float64`.

ii.
```python
session_input.append(time_axis.reshape(1, -1).copy())
```

iii. N/A — nothing to justify beyond the choice of window and bin size (see 2-e).

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `edges` is passed straight into `np.histogram` for the spikes, and `time_axis = edges[:-1] + DT/2` is the stored input, so input element *k* is the centre of the same 5 ms interval that neural column *k* counts spikes in. The same `time_axis` is also the interpolation target for tongue velocity, paw velocity and motion energy, so all four streams share one axis by construction.

ii.
```python
n_bins = int(round((TMAX - TMIN) / DT))
edges = np.linspace(TMIN, TMAX, n_bins + 1)
time_axis = edges[:-1] + DT / 2
...
counts = np.histogram(aligned[mask], bins=edges)[0]
...
session_input.append(time_axis.reshape(1, -1).copy())
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields of `obj.bp`: the outcome flags `hit` and `miss`, and the instructed-side flags `R` and `L`. The lick direction itself is not recorded anywhere, so it has to be inferred from the pair (instructed side, outcome). Trials that are neither hit nor miss (ignores) get their own class.

ii.
```python
hit = sess.get_bp_field('hit')[:n_trials_total]
miss = sess.get_bp_field('miss')[:n_trials_total]
R = sess.get_bp_field('R')[:n_trials_total]
L = sess.get_bp_field('L')[:n_trials_total]
```

iii. The AI worked this out explicitly in its reasoning: *"I'm trying to disentangle whether R/L mark the correct side versus the response side, and how hit/miss/no map onto that for correct, error, and ignore trials respectively. So miss on a right trial means the animal licked left, and miss on a left trial means it licked right."*

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed port, a miss means it licked the other one, and everything else (ignore) means it did not lick. Codes: `0 = left`, `1 = right`, `2 = none`, with `2` as the default. The scalar is then broadcast across all 1000 bins so that this per-trial variable has the same shape as the time-varying ones. Resulting class fractions are 0.4227 / 0.4460 / 0.1313, matching the human reference to four decimal places.

ii.
```python
# Lick direction: 0=left, 1=right, 2=none
lick_dir = np.full(n_valid, 2, dtype=np.int64)
for i, ti in enumerate(valid_idx):
    if hit[ti] == 1 and R[ti] == 1:
        lick_dir[i] = 1
    elif hit[ti] == 1 and L[ti] == 1:
        lick_dir[i] = 0
    elif miss[ti] == 1 and R[ti] == 1:
        lick_dir[i] = 0  # wrong side
    elif miss[ti] == 1 and L[ti] == 1:
        lick_dir[i] = 1  # wrong side
```

```python
out = np.zeros((6, n_time), dtype=np.int64)
out[0, :] = lick_dir[i]
```

```python
'output_values': [
    ['left', 'right', 'none'],
    ...
```

iii. As in 4-a, plus the format decision: *"I'm now reconsidering how to represent a mix of time-varying and per-trial outputs together, since the spec calls for uniform shape (n_output, n_timepoints) or (n_output,) — likely by broadcasting per-trial scalars across the time dimension so everything aligns consistently."*

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial field, `obj.bp.autowater`, which flags the trials where water was delivered from a random port with no cue — the water-cued (WC) context. Every other trial is delayed-response (DR).

ii.
```python
autowater = sess.get_bp_field('autowater')[:n_trials_total]
```

iii. The AI verified empirically that the randomized-delay sessions have `autowater` identically zero (*"JEB23 … autowater values: [0.], has WC trials: False"*) and that the two-context sessions have 0-122 WC trials each, and accepted pooling them: *"behavioral context from the autowater flag."*

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `autowater == 1 -> 0 (WC)`, else `1 (DR)`, broadcast across all 1000 bins. Class fractions 0.0966 / 0.9034, matching the reference exactly.

ii.
```python
# Context: 0=WC, 1=DR
context = np.where(autowater[valid_idx] == 1, 0, 1).astype(np.int64)
...
out[1, :] = context[i]
```

```python
['WC', 'DR'],
```

iii. The 0 = WC, 1 = DR coding follows the order given in the task instructions ("**Behavioral context** (WC, DR, per-trial)").

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The per-trial flags `obj.bp.hit` and `obj.bp.miss`. `obj.bp.no` is also read but is used only inside the (tautological) `(hit | miss | no)` trial mask; the outcome value itself is derived from `hit` and `miss` alone, with ignore as the default.

ii.
```python
hit = sess.get_bp_field('hit')[:n_trials_total]
miss = sess.get_bp_field('miss')[:n_trials_total]
no = sess.get_bp_field('no')[:n_trials_total]
```

iii. *"outcome from hit/miss/no fields"* — the AI planned to use all three and in the end derived the value from two of them, since they are mutually exclusive.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A relabelling into three classes: `1 = correct` on hit trials, `0 = incorrect` on miss trials, `2 = ignore` as the default for everything else. Ignore trials are kept in the dataset rather than dropped (the paper omits them from its own analyses). Broadcast across 1000 bins. Class fractions 0.1197 / 0.7490 / 0.1313, matching the reference exactly.

ii.
```python
# Outcome: 0=incorrect, 1=correct, 2=ignore
outcome = np.full(n_valid, 2, dtype=np.int64)
for i, ti in enumerate(valid_idx):
    if hit[ti] == 1:
        outcome[i] = 1
    elif miss[ti] == 1:
        outcome[i] = 0
...
out[2, :] = outcome[i]
```

```python
['incorrect', 'correct', 'ignore'],
```

iii. The 0/1/2 coding follows the order in the task instructions ("**Outcome** (incorrect, correct, ignore, per-trial)"). Keeping ignore trials is implied by the AI's decision to give "no lick" its own lick-direction class rather than discard those trials: *"lick direction (including 'no' for ignore trials)."*

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, **bottom camera only** (`cam_idx = 1`), feature index **0**, which is `top_tongue`. The feature index is hard-coded rather than looked up in `featNames`. The side camera's `tongue` feature is not used at all. The alignment also needs `obj.traj[1].frameTimes`, `obj.bp.ev.goCue`, and the bitcode fields `obj.sglx.fs` / `obj.sglx.bitcode.bitstart` / `obj.bp.ev.bitStart` (see 7-d). The `ts` array holds `[x, y, likelihood]` per feature per frame.

ii.
```python
n_traj_trials = sess.get_n_traj_trials(1)  # bottom cam
for trix in range(min(n_trials_total, n_traj_trials)):
    ts, ft = sess.get_traj_trial(1, trix)  # bottom cam
    if ts is None or len(ft) == 0:
        continue

    aligned_ft = ft - vidshift - go_cue[trix]

    # Tongue: top_tongue = feature 0 in bottom cam
    tv, tvis = compute_velocity_from_dlc(ts, aligned_ft, 0, time_axis)
    tongue_vel[:, trix] = tv
    tongue_vis[:, trix] = tvis
```

```python
x = ts[:, 0, feat_idx]
y = ts[:, 1, feat_idx]
conf = ts[:, 2, feat_idx]
```

iii. The AI chose the bottom camera from the paper's methods: *"According to the methods, DeepLabCut tracked tongue, jaw, and nose from both cameras, paws from the bottom view only, and [tongue angle and length] specifically came from the bottom cam, so I'm leaning toward using the bottom cam's top_tongue feature for velocity."* It measured tongue visibility on the side camera first (~18 % of frames on an example trial) but still went with a single view. The `(n_frames, 3, n_feats)` axis convention was established empirically for both file formats: *"In v5, traj is an array of struct arrays, with dimensions (frames, [x,y,conf], feats) NOT transposed."*

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, in `compute_velocity_from_dlc`:
  1. **Frame-to-frame speed**: `sqrt(dx^2 + dy^2) / dt_video` with `dt_video = 1/400` s held fixed (the cameras run at 400 Hz). The raw x, y traces are *not* smoothed before differentiation, and the actual `frameTimes` spacing is not used as the denominator.
  2. **Visibility**: a velocity sample is visible only if *both* frames contributing to it have likelihood `>= CONF_THRESH = 0.6`. Because the authors already set x and y to NaN wherever the likelihood is `<= 0.9` (verified: `nanfrac_x_given_lowconf = 1.000` for every feature), the effective visibility threshold is the data's 0.9, not 0.6 — sub-0.9 frames yield NaN speeds regardless.
  3. **Resampling**: each velocity sample is timestamped at the midpoint of its two frames and the trace is linearly interpolated (`np.interp`) onto the 1000 bin centres, with NaN outside the frame range. Bins whose interpolant is NaN are forced to "not visible". This is point-sampling of an interpolant, not averaging of the frames within a bin.
  4. **Discretisation** (see 7-c). No cross-camera normalisation is needed because only one camera is used.

  The result is 4.5 % / 4.5 % / 91.0 % for below / above / not-visible, against the reference's 6.2 % / 6.2 % / 87.5 %.

ii.
```python
    dt_video = 1.0 / VIDEO_FPS
    dx = np.diff(x)
    dy = np.diff(y)
    vel = np.sqrt(dx**2 + dy**2) / dt_video

    vis = (conf[:-1] >= CONF_THRESH) & (conf[1:] >= CONF_THRESH)
    ft_vel = (frame_times_aligned[:-1] + frame_times_aligned[1:]) / 2

    if len(ft_vel) < 2:
        return np.zeros(n_time), np.zeros(n_time, dtype=bool)

    vel_interp = np.interp(time_axis, ft_vel, vel, left=np.nan, right=np.nan)
    vis_interp = np.interp(time_axis, ft_vel, vis.astype(float), left=0, right=0) >= 0.5

    vis_interp[np.isnan(vel_interp)] = False
    vel_interp = np.nan_to_num(vel_interp, nan=0.0)
```

```python
VIDEO_FPS = 400
CONF_THRESH = 0.6
```

iii. *"Velocity is the first derivative of position … computing velocity as the magnitude of its derivative, using the primary tongue feature as a single representative value. For visibility flags on tongue and paw, I'll rely on DLC confidence scores, treating low-confidence frames below a standard threshold (like 0.6) as not visible."* The AI also noted that the authors' kinematics loader fills missing values for every feature except the tongue, and deliberately preserved the tongue gaps: *"unlike the paw, tongue gaps aren't filled with nearest-neighbor values."*

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. `discretize_per_session` pools every visible bin of every kept trial in that session, takes the 50th percentile of that pool as the threshold, and assigns `1` where velocity `>= threshold`, `0` below, and `2` ("not visible") to every bin whose visibility flag is false. Bins that are not visible do not enter the percentile computation. The threshold is therefore per-session, exactly as the instructions require.

ii.
```python
def discretize_per_session(values, valid_mask):
    """Discretize values per session: 0=<50th, 1=>=50th, 2=invalid."""
    result = np.full_like(values, 2, dtype=np.int64)
    all_valid = values[valid_mask]

    if len(all_valid) > 0:
        thresh = np.percentile(all_valid, 50)
        result[valid_mask] = np.where(all_valid >= thresh, 1, 0)

    return result
```

```python
tv_valid = tongue_vel[:, valid_idx]
tvis_valid = tongue_vis[:, valid_idx]
tongue_disc = discretize_per_session(tv_valid, tvis_valid)
```

```python
['below_50th', 'above_50th', 'not_visible'],
```

iii. Taken directly from the task instructions: *"For discretizing velocity, my working plan is to bin values below the session-specific 50th percentile as one state, values at or above as another, and mark 'not visible' frames as a third separate state."*

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The cameras run on the SpikeGLX clock, which starts before the behaviour clock, so frame times must be corrected before they can be compared to the go cue. The offset is computed once per session as `mode(sglx.bitcode.bitstart) / sglx.fs - mode(bp.ev.bitStart)`, using only strictly positive entries of each, and matches the authors' `findVideoOffset.m`. Frame time from the go cue is then `frameTimes - vidshift - goCue[trial]`, and the resulting trace is interpolated onto the shared 1000-bin grid, so bin *k* of the tongue output covers the same interval as bin *k* of the neural data. Measured offsets are ~0.490 s for both a v7.3 and a v5 session. If anything raises while reading the bitcode fields, the function silently returns a hard-coded 0.5 s.

ii.
```python
def get_video_offset(self):
    try:
        ...
        valid_bc = bc_bs[bc_bs > 0]
        valid_bp = bp_bs[bp_bs > 0]
        if len(valid_bc) == 0 or len(valid_bp) == 0:
            return 0.5
        bc_mode = spstats.mode(valid_bc, keepdims=False).mode
        bp_mode = spstats.mode(valid_bp, keepdims=False).mode
        return float(bc_mode / fs - bp_mode)
    except Exception:
        return 0.5
```

```python
vidshift = sess.get_video_offset()
...
aligned_ft = ft - vidshift - go_cue[trix]
```

iii. The AI read `findVideoOffset.m` and reproduced it rather than using the approximation used in the authors' plotting tutorial: *"I compute vidshift by dividing the mode of bitcode start samples by the sampling rate, then subtracting the mode of bitStart events — this gives the true video-to-neural offset rather than the rough 0.5 second approximation used elsewhere in the tutorial."* It sanity-checked the value on a real session (*"vidshift: 0.4900 s … which explains the '-0.5' approximation in the tutorial code"*), which is also why 0.5 was chosen as the fallback.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` bottom camera, but **two** features: index 4 (`top_paw`) and index 5 (`bottom_paw`). Both indices are hard-coded. Plus the same `frameTimes` / `goCue` / bitcode fields for alignment.

ii.
```python
# Paw: top_paw=4, bottom_paw=5 in bottom cam - take max
pv_top, pvis_top = compute_velocity_from_dlc(ts, aligned_ft, 4, time_axis)
pv_bot, pvis_bot = compute_velocity_from_dlc(ts, aligned_ft, 5, time_axis)
```

iii. *"since the paper tracks both top_paw and bottom_paw from the bottom camera, I'm weighing whether to average both paws' velocities or take the max at each timepoint to represent overall paw movement … I'll settle on taking the maximum of top_paw and bottom_paw velocities."* The AI had previously checked their likelihoods on an example trial (`top_paw` constant at 1.000, `bottom_paw` ranging 0.033-1.000).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw goes through the identical `compute_velocity_from_dlc` pipeline as the tongue (frame-difference speed at a fixed 400 Hz, likelihood-based visibility, linear interpolation onto the bin centres). The two are then combined per bin: where both are visible, the **maximum** of the two speeds; where only one is visible, that one; where neither is, the bin is marked not visible. No normalisation is applied, so values stay in pixels/s. Discretisation as in 8-c. Resulting classes: 47.0 % / 47.0 % / 6.0 %, against the reference's 40.7 % / 40.7 % / 18.6 % — the AI's paw stream is visible in considerably more bins because the second paw fills gaps in the first.

ii.
```python
combined_vis = pvis_top | pvis_bot
combined_vel = np.where(
    pvis_top & pvis_bot, np.maximum(pv_top, pv_bot),
    np.where(pvis_top, pv_top, np.where(pvis_bot, pv_bot, 0.0))
)
paw_vel[:, trix] = combined_vel
paw_vis[:, trix] = combined_vis
```

iii. As in 8-a — the max is intended as an "overall paw movement" summary. The AI also noted the tension between the authors' gap-filling for paws and the required "not visible" class, and resolved it by keeping the confidence-based visibility flag rather than filling: *"For the paw, missing values do get filled, yet the task still calls for a 'not visible' category, so I need to figure out when to flag the paw as not visible from its DLC confidence even though its position is technically always interpolated."*

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: `discretize_per_session` over all visible bins of all kept trials in the session, split at that session's 50th percentile, `2` for bins where neither paw was tracked.

ii.
```python
pv_valid = paw_vel[:, valid_idx]
pvis_valid = paw_vis[:, valid_idx]
paw_disc = discretize_per_session(pv_valid, pvis_valid)
...
out[4, :] = paw_disc[:, i]
```

```python
['below_50th', 'above_50th', 'not_visible'],
```

iii. Directly from the task instructions (per-session 50th percentile, third class for not visible).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as the tongue, and in the same loop: the paw comes from the same bottom-camera trial record, so it reuses the same `aligned_ft = ft - vidshift - go_cue[trix]` and the same interpolation onto the shared 1000-bin grid. No separate treatment.

ii.
```python
ts, ft = sess.get_traj_trial(1, trix)  # bottom cam
...
aligned_ft = ft - vidshift - go_cue[trix]
...
pv_top, pvis_top = compute_velocity_from_dlc(ts, aligned_ft, 4, time_axis)
pv_bot, pvis_bot = compute_velocity_from_dlc(ts, aligned_ft, 5, time_axis)
```

iii. Same justification as 7-d — one session-level video offset, one shared time grid for every stream.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone file `motionEnergy_<anm>_<date>.mat`, which holds one trace per trial with one value per **side-camera** frame. `obj.me` (present in only some sessions) is not used. The time base comes from the side camera's `frameTimes` (`cam_idx = 0`), plus the session video offset and the trial's go cue.

ii.
```python
me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{animal}_{date}.mat')
...
me_data = load_motion_energy(me_path)

# Get side cam frame times for alignment (loadMotionEnergy.m uses side cam)
n_side_trials = sess.get_n_traj_trials(0)
for trix in range(min(n_trials_total, len(me_data), n_side_trials)):
    _, ft = sess.get_traj_trial(0, trix)
```

`load_motion_energy` handles the three on-disk layouts:
```python
    # Format 1: me is (n_trials, 1) cell array directly (no struct wrapper)
    if me_raw.dtype == object and me_raw.ndim == 2 and me_raw.shape[1] == 1:
        first = me_raw[0, 0]
        if isinstance(first, np.ndarray) and first.dtype.kind == 'f':
            return [me_raw[i, 0].flatten() for i in range(me_raw.shape[0])]

    # Format 2: me is struct with .data and .moveThresh
    ...
    # Handle nested struct: me.data might itself be a struct with .data
    while hasattr(data_field, 'dtype') and data_field.dtype.names and 'data' in data_field.dtype.names:
```

iii. The three layouts were discovered by debugging failures: *"the JEB23 ME file has a different format - me is just a cell array of per-trial arrays, without the struct wrapper and without a threshold … JEB15 has yet another variant with a nested struct where me.data itself contains .data and .moveThresh, while the standard format like JEB7 just has a struct with .data and .moveThresh directly."* The side camera was chosen because the authors' `loadMotionEnergy.m` uses it, as the inline comment records.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Two steps. First, the per-frame trace is linearly interpolated onto the 1000 bin centres using the corrected side-camera frame times, with NaN outside the frame range; if the trace and the frame-time vector have different lengths, both are truncated to the shorter. Second — and this is the consequential choice — the remaining NaNs are **filled in**, by re-interpolating the column against its own valid indices (`np.interp` over bin index), which linearly bridges interior gaps and clamps the trace's end values across any leading or trailing gap. The AI labels this as reproducing `fillmissing` in the authors' `loadMotionEnergy.m`. No smoothing, differentiation, or spatial reduction is applied — the value is already one number per frame.

  The consequence is that the `no_video` class is never assigned anywhere in the dataset: the realised class fractions are 0.4999 / 0.5001 / 0.0000, whereas the reference leaves those bins NaN and assigns class 2 to 3.8 % of bins. Motion energy therefore became a two-class problem for the decoder (its balanced-loss weight vector had only two entries), while `output_values` still declares three.

ii.
```python
me_trial = me_data[trix]
min_len = min(len(aligned_ft), len(me_trial))
if min_len < 2:
    continue
me_aligned[:, trix] = np.interp(
    time_axis, aligned_ft[:min_len], me_trial[:min_len],
    left=np.nan, right=np.nan
)
```

```python
# Fill NaN with nearest value (matching loadMotionEnergy.m fillmissing)
for trix in range(n_trials_total):
    col = me_aligned[:, trix]
    nans = np.isnan(col)
    if np.any(nans) and not np.all(nans):
        valid = ~nans
        me_aligned[:, trix] = np.interp(
            np.arange(n_time), np.where(valid)[0], col[valid]
        )
```

iii. The fill was a deliberate port of the authors' loader (*"I'm also checking … the loadMotionEnergy interpolation logic"*), and the AI noticed the resulting loss of the third class and endorsed it: *"The motion energy values are only 0 and 1, never 2 (no_video), since I'm filling NaNs with nearest values, which is good."*

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize_per_session` helper: pool every non-NaN bin of every kept trial in the session, split at that session's 50th percentile (`1` if `>=`, else `0`), and assign `2` ("no_video") wherever the value is NaN. Because of the fill in 9-b, no bin is NaN in practice, so class 2 is never used.

ii.
```python
me_valid = me_aligned[:, valid_idx]
me_disc = discretize_per_session(me_valid, ~np.isnan(me_valid))
...
out[5, :] = me_disc[:, i]
```

```python
['below_50th', 'above_50th', 'no_video'],
```

iii. Straight from the task instructions: *"motion energy follows a similar binary split with a no-video flag"*, with the same per-session median rule as the two velocities.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The same two-step correction as the camera kinematics, but using the **side** camera's frame times, since the motion-energy trace has one value per side-camera frame: `aligned_ft = frameTimes(side) - vidshift - goCue[trial]`, then linear interpolation onto the shared 1000-bin grid. Trials beyond `min(n_trials_total, len(me_data), n_side_trials)`, or whose side-camera frame times are missing or shorter than two samples, are skipped and left NaN (before the fill).

ii.
```python
n_side_trials = sess.get_n_traj_trials(0)
for trix in range(min(n_trials_total, len(me_data), n_side_trials)):
    _, ft = sess.get_traj_trial(0, trix)
    if ft is None or len(ft) < 2:
        continue
    aligned_ft = ft - vidshift - go_cue[trix]
```

iii. Same session-level offset and same grid as every other stream; the side camera is used because that is the camera the authors' `loadMotionEnergy.m` pairs the trace with, as the inline comment in the code states.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms, some explicit and some implicit via broad exception handlers:
  - **Fields stored longer than `Ntrials`**: every `bp` field is sliced to `[:n_trials_total]`.
  - **Two MATLAB file formats and three motion-energy layouts**: handled by format sniffing and by unwrapping loops, as in 1-a and 9-a.
  - **Untracked DLC frames** (likelihood `<= 0.9`, where the authors already wrote NaN into x and y): the velocity becomes NaN, the visibility flag goes false, and the bin is emitted as the trailing "not visible" class. Nothing is interpolated across the gap for tongue or paw.
  - **Trials with no trajectory record** (`trix >= n_traj_trials`) or with an unreadable one: skipped, leaving the pre-initialised zeros and `visible = False`, i.e. 1000 "not visible" bins.
  - **Motion energy / frame-time length mismatch**: both truncated to `min_len`.
  - **Motion energy gaps**: filled by interpolation/edge-clamping rather than marked (see 9-b) — the only place where values are invented rather than flagged.
  - **Trials after the probe stopped**: dropped entirely (1-e).
  - **Silent fallbacks**: `get_video_offset` returns a hard-coded 0.5 s on any exception or if the bitcode arrays are empty; `get_traj_trial` and `get_n_traj_trials` return `None` / `0` on any exception; motion-energy loading is wrapped in a `try` that prints a warning and leaves the whole session's motion energy NaN. None of these print a per-session diagnostic for the offset or trajectory cases, so a session whose video clock could not be recovered would be silently misaligned by up to ~10 ms rather than flagged.

ii.
```python
    except Exception:
        return 0.5
```

```python
def get_traj_trial(self, cam_idx, trial_idx):
    try:
        ...
    except Exception:
        return None, None
```

```python
    n_time = len(time_axis)
    if ts is None or len(frame_times_aligned) < 3:
        return np.zeros(n_time), np.zeros(n_time, dtype=bool)
```

```python
            except Exception as e:
                print(f"  WARNING: Failed to load motion energy: {e}")
```

iii. The AI's stated principle for the kinematics was to preserve genuine gaps as the "not visible" class rather than invent values (*"tongue gaps aren't filled with nearest-neighbor values"*, *"treating low-confidence frames below a standard threshold … as not visible"*), and it distinguished the two kinds of NaN explicitly: *"I need to distinguish NaN from edge extrapolation versus NaN from genuine low confidence."* For motion energy it went the other way, on the grounds of matching `loadMotionEnergy.m`. The broad `except` clauses are not discussed anywhere in the trajectory; they appear to be defensive scaffolding left over from the iterative debugging of the two file formats.

## 11-a. What are the most time-consuming steps of the code?

i. Three things dominate.
  1. **File I/O.** ~7 GB of `.mat` files must be read. The v7.3 sessions are read lazily through `h5py` (cheap per-field), but the 11 v5 sessions are fully materialised by `scipy.io.loadmat`. Measured on a warm cache: ~1.4 s for a v5 `loadmat`, ~1.0 s to walk one session's bottom-camera trajectories.
  2. **Spike binning and smoothing.** `bin_and_smooth_spikes` is a Python double loop over units x trials; each iteration builds a boolean mask over the unit's entire spike array, histograms, and calls `causal_gaussian_smooth`, which rebuilds the 15-tap kernel and runs a Python-level column loop every time. Measured ~1.4 s for 67 units x 346 trials, so ~50-60 s over the whole dataset (~810,000 inner iterations).
  3. **Writing the output.** The pickle is 7.0 GB, 2.6x the reference's 2.7 GB, because the neural arrays are `float64` and the outputs `int64`.

ii.
```python
    for i, unit in enumerate(units):
        aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]
        for j in range(n_trials):
            mask = unit['trial'] == (j + 1)
            if not np.any(mask):
                continue
            counts = np.histogram(aligned[mask], bins=edges)[0]
            fr = counts.astype(float) / DT
            trialdat[:, i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, 'reflect')
```

```python
with open(output_path, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI never profiled or discussed runtime; it only noted the scale up front — *"With 44 sessions and 200-500 trials each requiring spike binning, DLC extraction, and motion energy alignment, I want to write the conversion script efficiently"* — and the conversion completed inside its 600 s budget, so it did not revisit the question.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Four.
  1. The **units x trials double loop** in `bin_and_smooth_spikes`. The whole thing is one `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, edges])` per unit — the reference does exactly that. As written it is O(n_units x n_trials) passes over each unit's spike array just to build `mask`.
  2. The **column loop** inside `causal_gaussian_smooth` (`for j in range(x_filt.shape[1])`). It is always called with a single column here, so it is pure overhead; smoothing the whole (bins x units x trials) array at once with `scipy.ndimage.convolve1d` along axis 0 would replace ~810,000 calls with one. The kernel itself is also rebuilt on every call and could be hoisted to module scope.
  3. The **all-zero trial check** `[np.any(trialdat[:, :, ti] != 0) for ti in valid_idx]`, which is `np.any(trialdat[:, :, valid_idx] != 0, axis=(0, 1))`.
  4. The **per-trial label loops** for `lick_dir` and `outcome`, which are straightforward boolean-mask assignments (the reference writes them that way); and the final `for i, ti in enumerate(valid_idx)` assembly loop, which allocates a fresh `(6, 1000)` array per trial.

  The per-trial camera loops are not vectorisable in the same way, since each trial has a different frame count.

ii.
```python
        for j in range(n_trials):
            mask = unit['trial'] == (j + 1)
```

```python
    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```

```python
has_spikes = np.array([np.any(trialdat[:, :, ti] != 0) for ti in valid_idx])
```

```python
lick_dir = np.full(n_valid, 2, dtype=np.int64)
for i, ti in enumerate(valid_idx):
    if hit[ti] == 1 and R[ti] == 1:
        lick_dir[i] = 1
    ...
```

iii. Not discussed. The AI's stated intent was to *"write the conversion script efficiently"*, but it never returned to optimise once the script ran to completion.

## 11-c. What processing does the code repeat multiple times?

i. Five repetitions, none of which change the result:
  - The **Gaussian kernel** is recomputed from scratch on every one of the ~810,000 `causal_gaussian_smooth` calls, even though it depends only on the constant `SMOOTH_WINDOW`.
  - The **lower-cased quality set** `{q.lower() for q in EXCLUDED_QUALITIES}` is rebuilt for every unit inside the list comprehension, and `EXCLUDED_QUALITIES` is already lower-case.
  - The **spike-trial mask** `unit['trial'] == (j + 1)` re-scans the unit's whole spike array once per trial, i.e. `n_trials` full passes where one `histogram2d` would do.
  - The **side-camera trajectory** of each trial is read a second time, in the motion-energy loop, after the bottom-camera trajectory of the same trial was read in the kinematics loop; the two loops could share one pass over trials.
  - **Smoothing is applied before the firing-rate filter**, so every unit that is later dropped by `fr_mask` (a large fraction, given 10,330 clusters on file) was nonetheless binned and smoothed across every trial first. The reference computes the rate, tests the mean, and only then smooths.

  Things that are correctly computed once: the video offset (once per session, not per trial), the bin grid (once per run), and each session's percentile thresholds.

ii.
```python
    n = np.arange(window)
    alpha = 2.5
    center = (window - 1) / 2
    kern = np.exp(-0.5 * ((n - center) / (center / alpha)) ** 2)
    kern[:int(window // 2)] = 0  # causal
    kern = kern / kern.sum()
```

```python
    units = [u for u in units
             if u['quality'].lower() not in {q.lower() for q in EXCLUDED_QUALITIES}
             and u['quality'] != '']
```

```python
trialdat = bin_and_smooth_spikes(all_units, go_cue, n_trials_total, edges, time_axis)
# ---- Remove low FR units ----
mean_frs = np.mean(trialdat, axis=(0, 2))
fr_mask = mean_frs > LOW_FR
...
trialdat = trialdat[:, fr_mask, :]
```

```python
ts, ft = sess.get_traj_trial(1, trix)   # kinematics loop
...
_, ft = sess.get_traj_trial(0, trix)    # motion-energy loop, same trials again
```

iii. Not discussed in the trajectory.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Five items.
  1. **All streams are computed for every trial in the file, then subset.** `bin_and_smooth_spikes` runs over `n_trials_total`, and `tongue_vel` / `paw_vel` / `me_aligned` are built as `(n_time, n_trials_total)` arrays; only `valid_idx` columns are ever used. Roughly 10 % of trials (photostim, early-lick, post-recording) are processed and thrown away.
  2. **Smoothing of units that the firing-rate filter then removes** (see 11-c).
  3. **`clu.tm` is read for every cluster and never used.** It is the session-clock spike time; only `trial` and `trialtm` are needed. On the v7.3 path this is an extra HDF5 dataset read per cluster.
  4. **Redundant behavioural fields.** `no` is read but only feeds the tautological `(hit | miss | no)` term in `valid_mask`; `L` is read but is exactly `~R` on the trials where it is consulted.
  5. **Oversized dtypes.** Neural rates are stored `float64` and outputs `int64`, producing a 7.0 GB pickle where `float32` / `int8` (what the reference uses) would give ~2.7 GB for identical information — firing rates are integer multiples of 200 Hz and the outputs are the integers 0-2. The `input` array, an identical 1000-element `float64` vector, is also stored once per trial (13,762 copies, ~110 MB) rather than shared.

ii.
```python
trialdat = np.zeros((n_time, n_units, n_trials))   # all trials, float64
```

```python
tongue_vel = np.zeros((n_time, n_trials_total))
tongue_vis = np.zeros((n_time, n_trials_total), dtype=bool)
paw_vel = np.zeros((n_time, n_trials_total))
paw_vis = np.zeros((n_time, n_trials_total), dtype=bool)
```

```python
tm = np.array(self._f[probe_group['tm'][i, 0]]).flatten()
...
units.append({
    'tm': tm, 'trial': trial, 'trialtm': trialtm, 'quality': quality
})
```

```python
out = np.zeros((6, n_time), dtype=np.int64)
...
session_input.append(time_axis.reshape(1, -1).copy())
```

iii. Not discussed. The AI's only memory-related remark in the trajectory concerns the *decoder's* footprint during training (*"13.7 million timepoint-trials with 1000 timepoints each on CPU"*), not the size of the artefact it wrote.
