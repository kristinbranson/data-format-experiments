# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hard-codes a single data folder, `/app/data/RandomizedDelay_Ephys_Behavior`, and discovers sessions by globbing `data_structure_*.mat` inside it. The three other folders under `/app/data` (`Ephys_Behavior`, which holds the 25 fixed-delay ephys sessions, plus the two behavior-only inhibition folders) are never listed or opened, so the fixed-delay half of the dataset is silently absent. Each matched file is opened once by `load_session`, which first tries `h5py` (MATLAB v7.3) and falls back to `scipy.io.loadmat(squeeze_me=True, struct_as_record=False)` on `OSError` (MATLAB v5). Motion energy is read separately from a `motionEnergy_<session>.mat` sidecar in the same folder. The realised dataset is 20 sessions / 4 mice / 874 units / 7,064 trials.

ii.
```python
BASE = Path('/app/data/RandomizedDelay_Ephys_Behavior')
...
def load_session(path):
    try:
        with h5py.File(path, 'r') as h:
            return load_session_h5(path, h)
    except OSError:
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return load_session_mat(path, mat['obj'])
...
def convert(sample=False):
    files = sorted(BASE.glob('data_structure_*.mat'))
    if sample:
        files = files[:2]
    sessions = [load_session(f) for f in files]
```

iii. From CONVERSION_NOTES.md Step 2: *"Data are stored in `/app/data/RandomizedDelay_Ephys_Behavior` as per-session MATLAB files."* The trajectory confirms the agent went straight to that subfolder at step 11 without ever listing `/app/data`, and thereafter took the randomized-delay task to be the whole dataset — Step 3 quotes only the paper's randomized-delay sentence (*"845 units ... from 19 sessions using four mice"*) as the target statistic. The dual reader is justified in Step 2: *"Not all `data_structure_*.mat` files share the same MATLAB storage format; at least one file is not readable by `h5py` and requires a fallback loader."*

## 1-b. How are the data split into subjects?

i. The subject is parsed out of the filename stem: `data_structure_JEB11_2022-05-10` split on `_` gives `JEB11` at index 2. `subjects` is the sorted unique set over the retained sessions and `subject_idx` is each session's index into it. This yields 4 subjects (JEB11, JEB12, JEB23, JEB24).

ii.
```python
'subject': path.stem.split('_')[2],
...
subjects = sorted({s['subject'] for s in sessions})
subject_map = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subject_map[sess['subject']])
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. Not discussed explicitly in CONVERSION_NOTES.md beyond the Step 2 observation that the files are named `data_structure_<subject>_<date>.mat`, i.e. the animal id is available in the filename. (`obj.meta` was seen in Step 2 but never used for this.)

## 1-c. How are the data split into sessions?

i. One `data_structure_*.mat` file = one session = one element of `neural`/`input`/`output`. Two curation filters are then applied at the session level: sessions with no `obj.clu` group are dropped (JEB24 2023-10-03 and 2023-10-04), and sessions with fewer than 10 retained units are dropped (none actually were). Sessions are not cross-referenced against the authors' `load<ANM>_ALMVideo.m` loading scripts, so `JEB23_2023-10-20` — which the authors commented out — is included. Result: 20 sessions.

ii.
```python
sessions = [load_session(f) for f in files]
sessions = [s for s in sessions if s['has_clu']]
sessions = [s for s in sessions if len(select_units(s)) >= 10]
```

iii. CONVERSION_NOTES.md Step 4: *"22 session files total; 20 with `clu`, 2 without neural data ... Exclude the 2 sessions lacking `clu`; remaining paper mismatch likely reflects one additional session exclusion in the reference subset."* The ≥10-unit rule is taken from the methods: *"Recording sessions were included for analysis only if they had at least 10 units."* The residual 20-vs-19 discrepancy was noticed and explicitly left unresolved.

## 1-d. How are the data split into trials?

i. The trial count is `obj.bp.Ntrials`, and a trial is one index into the per-trial `bp` fields (`hit`, `miss`, `no`, `early`, `autowater`, `R`, `L`, `ev.goCue`). Neural trials are built by iterating `tr` from 1 to `Ntrials` and selecting spikes with `clu.trial == tr`; outputs are built by iterating `0 .. Ntrials-1` over the `bp` arrays. No per-trial field is truncated to `Ntrials`, and no trial boundaries are reconstructed.

ii.
```python
ntrials = int(np.array(bp['Ntrials'][()]).squeeze())
...
for tr in range(1, session['ntrials'] + 1):
    arr = np.zeros((len(units), len(centers)), dtype=np.float32)
    for i, u in enumerate(units):
        mask = (u['trial'] == tr)
```

iii. Not explicitly discussed; CONVERSION_NOTES.md Step 2 records that *"`obj.bp` contains key trial-level behavioral fields including `Ntrials`, `hit`, `miss`, `no`, `early`, ... `R`, and `L`"* and that `obj.clu` stores `trial`/`trialtm`, i.e. trial membership is explicit in the raw data.

## 1-e. How are trials filtered based on quality controls?

i. **No trial filtering of any kind is applied.** All `Ntrials` trials of every retained session enter the output. In particular: early-lick trials (`bp.early` is loaded but never used), photostimulation trials (`bp.stim.enable` is never even read), and trials that run past the end of the ephys recording are all kept. The verifier flagged 66 trials in 3 sessions as "all neural data is zero" (the post-recording trials); these were left in.

ii. `bp.early` is read and then discarded:
```python
'early': np.array(bp['early'][()]).squeeze().astype(bool) if 'early' in bp else np.zeros(ntrials, dtype=bool),
```
and the only trial loop is unconditional:
```python
for tr in range(1, session['ntrials'] + 1):
```

iii. CONVERSION_NOTES.md Step 3 records the rule (*"Early lick and ignore trials were omitted from many paper analyses; need to decide whether to keep ignore trials for decoder output while potentially excluding early trials"*) and Step 5 Key Decision 5 states the intent: *"Early trials: Investigate whether early trials should be excluded from the decoder dataset; likely exclude if they disrupt standard task timing."* Nothing in the notes records a subsequent decision to keep them, and the filter was never implemented. Ignore trials are deliberately kept, which is justified: *"Keep ignore (`no`) as a decoder class even though the paper often excluded ignore trials from analysis, because decoder outputs explicitly require it."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu` — per cluster, `trialtm` (spike time relative to trial start), `trial` (1-based trial index), `quality` (manual curation label, used for QC), and `tm` (absolute spike times, used only for the firing-rate QC). For v7.3 files only the **first** probe is read (`h[obj['clu'][0,0]]`); for v5 files `obj.clu` is a flat cluster array. `obj.bp.ev.goCue` is loaded into the session dict but is **never used** by the neural pipeline.

ii.
```python
clu = h[obj['clu'][0, 0]]
units = []
for i in range(clu['tm'].shape[0]):
    q = normq(decode_h5_char(h, clu['quality'][i, 0]))
    tm = np.array(h[clu['tm'][i, 0]][()]).squeeze().astype(float)
    trial = np.array(h[clu['trial'][i, 0]][()]).squeeze().astype(int)
    trialtm = np.array(h[clu['trialtm'][i, 0]][()]).squeeze().astype(float)
    site = int(np.array(h[clu['site'][i, 0]][()]).squeeze()) if 'site' in clu else -1
    units.append({'quality': q, 'tm': tm, 'trial': trial, 'trialtm': trialtm, 'site': site})
```

iii. CONVERSION_NOTES.md Step 2: *"`obj.clu` dereferences to a struct with fields `tm`, `site`, `quality`, `spkWavs`, `trialtm`, and `trial`, confirming that spike times and trial assignments are stored explicitly and that unit/site/quality metadata are available for curation."* Step 5 maps *"`obj.clu[*].trialtm` aligned to `obj.bp.ev.goCue`"* to the `neural` field.

## 2-b. How is the `neural` data processed?

i. Per unit and per trial: spike times are histogrammed into 0.02 s bins over a fixed edge grid, divided by `dt` to give Hz, and smoothed along time with a normalised Gaussian kernel of `sigma = 2` bins (= 40 ms), truncated at ±4σ, applied with `np.convolve(..., mode='same')`. No normalisation, baseline subtraction, or z-scoring. Stored as `float32`, one `(n_units, 220)` matrix per trial.

ii.
```python
def gaussian_smooth(x, sigma_bins):
    radius = max(1, int(math.ceil(4 * sigma_bins)))
    t = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (t / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(x, k, mode='same')

def build_neural_trials(session, units, tmin=-2.4, tmax=2.0, dt=0.02, smooth_sigma=2.0):
    edges = np.arange(tmin, tmax + dt, dt)
    centers = edges[:-1] + dt / 2
    ...
            counts, _ = np.histogram(u['trialtm'][mask], bins=edges)
            arr[i] = gaussian_smooth(counts.astype(np.float32) / dt, smooth_sigma)
```

iii. CONVERSION_NOTES.md Step 1: *"Reference code bins aligned spike times using histogram edges, divides by dt to obtain rate, and smooths with `mySmooth`."* The general recipe is taken from the reference; the specific parameters are not — the notes record *"Observed trial window parameter: `params.tmin = -2.4` seconds relative to goCue; need corresponding `tmax` and `dt` from the same script"*, and no follow-up ever supplies `dt` or the smoothing width, so `dt=0.02` and `sigma=2` bins are undocumented choices.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both in `select_units`. (1) An **allow**-list on the manual curation label, lower-cased and stripped, with a typo fix (`mutli` → `multi`): a unit is kept only if its label is one of `multi, fair, good, great, excellent`. In the randomized-delay folder this drops `garbage` (1631/8 sessions sampled), `poor`, and the handful of blank labels. (2) A mean firing rate `> 1 Hz`, computed as `n_spikes / (max(tm) - min(tm))` over the **whole recording**, not over the analysis window. 874 units survive across 20 sessions.

ii.
```python
KEEP_QUALITY = {'multi', 'fair', 'good', 'great', 'excellent'}

def normq(q):
    q = str(q).strip().lower().replace('\x00', '')
    if q == 'mutli':
        q = 'multi'
    return q

def unit_fr_gt1(unit):
    tm = unit['tm']
    if tm.size < 2:
        return False
    dur = float(tm.max() - tm.min())
    if dur <= 0:
        return False
    return (tm.size / dur) > 1.0

def select_units(session):
    keep = [u for u in session['units'] if u['quality'] in KEEP_QUALITY and unit_fr_gt1(u)]
    return keep
```

iii. CONVERSION_NOTES.md Step 4: *"a simple >1 Hz filter is insufficient to match the paper (1565 units remain vs 845 reported). JEB23 sessions contain many units labeled `garbage` ... after removing obvious garbage-quality clusters, raw units drop from 3139 to 896; adding a >1 Hz filter gives 874 units, much closer to the paper's 845."* The 1 Hz cut is taken from methods: *"All units with firing rates exceeding 1 Hz were included in all other analyses."* So the filter was chosen by reconciling against the paper's reported unit count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **The documented decision is go-cue alignment; the code does not implement it.** `build_neural_trials` histograms `u['trialtm'][mask]` directly, with no subtraction of `session['goCue'][tr-1]`. Since `clu.trialtm` is measured from *trial start* (verified on `JEB11_2022-05-10`: `goCue` ranges 1.9–7.4 s while `trialtm` spans −0.49 to 13.2 s), the resulting trials are aligned to **trial onset**, not the go cue. The consequence in the shipped pickle is that bins 0–~86 (t = −2.4 to −0.66 s) are identically zero in every trial of every session, and the window only ever covers 0–2.0 s after trial start, i.e. it usually ends *before* the go cue occurs.

ii. The alignment subtraction is absent:
```python
for tr in range(1, session['ntrials'] + 1):
    arr = np.zeros((len(units), len(centers)), dtype=np.float32)
    for i, u in enumerate(units):
        mask = (u['trial'] == tr)
        if np.any(mask):
            counts, _ = np.histogram(u['trialtm'][mask], bins=edges)   # goCue never subtracted
```
while `goCue` is loaded and left unused:
```python
'goCue': np.array(bp['ev']['goCue'][()]).squeeze().astype(float),
```
and the metadata nonetheless claims:
```python
'temporal_alignment_event': 'Go cue onset',
```

iii. CONVERSION_NOTES.md Step 1: *"`alignSpikes` computes `trialtm_aligned = trialtm - event`, where `event` is a per-trial value from `obj.bp.ev.(params.alignEvent)`"*, and Step 5 Key Decision 3: *"Align all neural and behavioral/video streams to `goCue` because both code and task require it."* The intent is correct and matches the reference; the subtraction was simply never written. The Step 10 "sanity check" that supposedly validated this (*"direct histogram+smooth from raw spike times for one retained neuron/trial matched converted neural trace exactly (`np.allclose=True`)"*) re-ran the same unaligned logic, so it could not detect the omission.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins (`dt = 0.02`) over the window `tmin = -2.4` to `tmax = 2.0` s, giving 220 bins per trial, identical for every trial and session. The input time axis is the bin centres, −2.39 to 1.99. No rebinning or resampling is applied to the neural stream — the spikes are histogrammed once directly onto this grid. (The video/motion-energy streams *are* resampled onto the same 220-bin grid; see 7-d/9-d.)

ii.
```python
def build_neural_trials(session, units, tmin=-2.4, tmax=2.0, dt=0.02, smooth_sigma=2.0):
    edges = np.arange(tmin, tmax + dt, dt)
    centers = edges[:-1] + dt / 2
...
'time_bin_size': 20.0,
'off_start': -2.4,
'off_end': 2.0,
```

iii. `tmin = -2.4` is taken from the reference: *"Observed trial window parameter: `params.tmin = -2.4` seconds relative to goCue; need corresponding `tmax` and `dt` from the same script."* That follow-up never happened, so `tmax = 2.0` and `dt = 0.02` have no documented basis in the paper or code.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable. The input is the vector of bin centres of the analysis window, defined by the conversion itself, tiled identically across every trial of every session. Shape `(1, 220)` per trial.

ii.
```python
centers = edges[:-1] + dt / 2
...
def build_inputs(centers, ntrials):
    arr = centers.astype(np.float32)[None, :]
    return [arr.copy() for _ in range(ntrials)]
...
INPUT_NAMES = ['time_from_go_cue']
```

iii. CONVERSION_NOTES.md Step 5: *"Time relative to go cue → input[0]: Continuous time-varying signal repeated for every trial/time bin ... Decoder input is only time from go cue onset."*

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond building the bin-centre vector and casting it to `float32`. A fresh copy is made per trial (`arr.copy()`), which is redundant memory but harmless.

ii.
```python
arr = centers.astype(np.float32)[None, :]
return [arr.copy() for _ in range(ntrials)]
```

iii. N/A — no processing was deemed necessary; the quantity is defined by the window.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. By construction it shares the neural bin grid: `build_neural_trials` returns `centers` and that exact array is passed to `build_inputs`, so input bin *k* is neural bin *k*. However, because the neural bins are aligned to trial onset rather than the go cue (2-d), the values the input carries are not in fact time from the go cue — they are time from trial start minus 2.4 s. The input is therefore mislabelled relative to what the neural data actually contains.

ii.
```python
neural_trials, centers = build_neural_trials(sess, units)
inputs = build_inputs(centers, sess['ntrials'])
```

iii. Not separately justified; the agent's Step 10 check (*"Input sanity check: converted `time_from_go_cue` matched expected bin centers exactly (`np.allclose=True`)"*) confirms only that the values equal the bin centres, not that those centres are referenced to the go cue.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `obj.bp.L` and `obj.bp.R` only — the **instructed** lick side of each trial. `hit`/`miss`/`no` are loaded but are not consulted when building lick direction, so the actual direction the animal licked is never recovered.

ii.
```python
'R': np.array(bp['R'][()]).squeeze().astype(bool),
'L': np.array(bp['L'][()]).squeeze().astype(bool),
...
lick = np.full(ntr, 2, dtype=np.int64)
lick[session['L']] = 0
lick[session['R']] = 1
```

iii. CONVERSION_NOTES.md Step 5 maps *"`obj.bp.R`, `obj.bp.L`, and/or lick event side → output: lick direction. Map to categorical per-trial labels: left/right/none ... `none` for trials with no response / ignore."* Step 2 also notes that *"`obj.bp.ev.lickL` and `lickR` are per-trial cell arrays, likely storing lick times within each trial"*, but these were never used.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The array is initialised to 2 (`none`), then `L` trials are set to 0 and `R` trials to 1. Because `L` and `R` are complementary and cover every trial, the `none` class is never emitted — the verifier reports the distribution as `{left 0.502, right 0.498}` with zero `none`. The per-trial code is then broadcast across all 220 bins. On hit trials the label equals the true lick direction; on miss trials the animal licked the *opposite* port, so the label is inverted; on ignore trials (8.8% of the data) the animal licked nowhere but is still labelled left or right. Effectively the variable encodes the *instructed side / stimulus identity*, not the lick direction.

ii.
```python
OUTPUT_VALUES = [
    ['left', 'right', 'none'],
    ...
]
...
lick = np.full(ntr, 2, dtype=np.int64)
lick[session['L']] = 0
lick[session['R']] = 1
...
out = np.vstack([
    np.full(ntime, lick[tr], dtype=np.int64),
    ...
```

iii. No justification is given for dropping the hit/miss correction. CONVERSION_NOTES.md Step 7 reports the sample distribution as `[0.546, 0.454, 0.000]` — the all-zero `none` fraction is printed but never flagged or investigated.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`.

ii.
```python
'autowater': np.array(bp['autowater'][()]).squeeze().astype(bool),
```

iii. CONVERSION_NOTES.md Step 4: *"Context encoding — Code condition strings use `autowater` and comments label WC vs DR ... Map `autowater` to context after confirming polarity from code/comments"*, and Step 5: *"`obj.bp.autowater` → output: behavioral context. Map to categorical per-trial labels WC vs DR using code comments (`autowater`=WC, `~autowater`=DR)."*

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `autowater` → 0 (`WC`), otherwise 1 (`DR`), broadcast across all 220 bins of the trial. Nothing else.

ii.
```python
context = np.where(session['autowater'], 0, 1).astype(np.int64)  # WC, DR
...
OUTPUT_VALUES[1] = ['WC', 'DR']
```

iii. Polarity taken from the reference figure scripts' condition strings, as above. The resulting dataset is heavily skewed (WC 1.4%, DR 98.6%), which the notes acknowledge: *"behavioral_context is highly imbalanced in many sessions (some sessions all DR), which may reflect block structure and selected sessions."*

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three mutually exclusive per-trial flags: `obj.bp.hit`, `obj.bp.miss`, `obj.bp.no`.

ii.
```python
'hit': np.array(bp['hit'][()]).squeeze().astype(bool),
'miss': np.array(bp['miss'][()]).squeeze().astype(bool),
'no': np.array(bp['no'][()]).squeeze().astype(bool),
```

iii. CONVERSION_NOTES.md Step 4: *"Code uses `hit`, `miss`, `no` conditions ... Map hit/miss/no to correct/incorrect/ignore"*, matching the `(hit|miss|no)` condition strings seen in the reference Figure 3 scripts.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Initialise to 2 (`ignore`), set `miss` → 0 (`incorrect`), `hit` → 1 (`correct`), and `no` → 2 (redundant with the initialisation). Broadcast across all 220 bins. Realised distribution: incorrect 10.4%, correct 80.8%, ignore 8.8%.

ii.
```python
outcome = np.full(ntr, 2, dtype=np.int64)
outcome[session['miss']] = 0
outcome[session['hit']] = 1
outcome[session['no']] = 2
...
OUTPUT_VALUES[2] = ['incorrect', 'correct', 'ignore']
```

iii. Class codes follow the decoder spec (`incorrect, correct, ignore`). Ignore trials are kept as their own class rather than dropped: *"Keep ignore (`no`) as a decoder class even though the paper often excluded ignore trials from analysis, because decoder outputs explicitly require it."*

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. `obj.traj[0]` — the **side** camera only — specifically `featNames`, `ts` (shape `(n_features, 3, n_frames)`; channels 0/1 = x/y, channel 2 = DeepLabCut likelihood) for the first of `tongue`, `left_tongue`, `right_tongue` that is present. `frameTimes` is read but discarded. The bottom camera's `top_tongue` is never used. Critically, the whole video branch runs only when `session['format'] == 'h5py'`: the 11 MATLAB-v5 sessions are skipped entirely and receive `not_visible` for all 220 bins of all trials.

ii.
```python
def extract_h5_traj_view(h, traj_group, trial_idx):
    feat_cell = h[traj_group['featNames'][trial_idx, 0]]
    feat_names = [decode_h5_char(h, r) for r in feat_cell[()].ravel()]
    ts = np.array(h[traj_group['ts'][trial_idx, 0]][()])
    frame_times = np.array(h[traj_group['frameTimes'][trial_idx, 0]][()]).squeeze().astype(float)
    return feat_names, ts, frame_times
...
        feat0, ts0, ft0 = extract_h5_traj_view(h, view0, tr)   # ft0 never used
        spd0, valid0 = speed_from_ts(ts0, feat0, ['tongue','left_tongue','right_tongue'])
...
    if session.get('format') == 'h5py':
        try:
            t_out, p_out = build_video_outputs_h5(session, ntime)
```

iii. CONVERSION_NOTES.md Step 12: *"For the tongue feature, channels 0 and 1 are NaN at early timepoints and channel 2 has small values between 0 and 1, strongly suggesting channels 0/1 are x/y coordinates and channel 2 is visibility/likelihood."* The session-coverage gap was noticed but left unfixed: Step 9, *"tongue and paw velocity are only populated in a subset of sessions (many later sessions remain entirely class 2), likely due to trajectory-format differences between HDF5 and loadmat sessions."*

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps. (1) Valid frames = finite x, finite likelihood, and `likelihood >= 0.5`; if fewer than 2 valid frames, the next candidate feature name is tried. (2) Speed = per-frame Euclidean displacement `sqrt(dx^2 + dy^2)` from `np.diff(..., prepend=nan)` — **not** divided by the frame interval, and with **no** smoothing of x/y, so it is pixels-per-frame and gaps between valid runs are differenced across. (3) Invalid frames are then *filled with the session-trial median speed* by `nan_to_num` before resampling. (4) The full-trial trace is linearly resampled onto the 220 bins (see 7-d), and the boolean validity mask is resampled the same way and re-thresholded at 0.5.

ii.
```python
def speed_from_ts(ts, feat_names, feature_candidates, like_thresh=0.5):
    for feat in feature_candidates:
        if feat in feat_names:
            i = feat_names.index(feat)
            xy = np.asarray(ts[i, 0:2, :], dtype=float)
            like = np.asarray(ts[i, 2, :], dtype=float)
            valid = np.isfinite(xy).all(axis=0) & np.isfinite(like) & (like >= like_thresh)
            if valid.sum() < 2:
                continue
            dx = np.diff(xy[0], prepend=np.nan)
            dy = np.diff(xy[1], prepend=np.nan)
            spd = np.sqrt(dx**2 + dy**2)
            spd[~valid] = np.nan
            return spd, valid
...
    rr = resample_trace_to_bins(np.nan_to_num(spd0, nan=np.nanmedian(spd0[np.isfinite(spd0)]) if np.isfinite(spd0).any() else 0.0), ntime)
    vv = resample_trace_to_bins(valid0.astype(float), ntime)
    tongue_res[tr] = (rr, vv >= 0.5)
```

iii. CONVERSION_NOTES.md Step 12: *"compute frame-to-frame speed from x/y when likelihood exceeds a threshold, then resample to neural bins and discretize by session median."* No justification is offered for the 0.5 likelihood cut (the paper's tracking is already NaN-masked at 0.9), for omitting position smoothing, or for median-filling invalid frames.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session: the median of the pooled resampled speed values over all bins of all trials that are marked valid. Bins at or above it get 1 (`ge50`), below get 0 (`lt50`), and bins whose resampled validity is `< 0.5` get 2 (`not_visible`). This matches the spec's per-session 50th-percentile rule, and the verification output confirms an exact 50/50 split of the visible bins in every session that has data.

ii.
```python
tongue_out = np.full((ntr, ntime), 2, dtype=np.int64)
...
if len(tongue_vals) > 0:
    med = np.nanmedian(np.asarray(tongue_vals, dtype=float))
    for tr, item in enumerate(tongue_res):
        if item is not None:
            rr, valid = item
            tongue_out[tr, valid] = (rr[valid] >= med).astype(np.int64)
...
OUTPUT_VALUES[3] = ['lt50', 'ge50', 'not_visible']
```

iii. Directly from the decoder spec: *"discretized with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile, 2: not visible."*

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. It is not aligned in time at all. `resample_trace_to_bins` maps the frame index axis onto `[0, 1]` with `np.linspace` and interpolates onto 220 equally spaced points — i.e. the first camera frame of the trial is forced into bin 0 (nominally t = −2.4 s) and the last frame into bin 219 (nominally t = +1.99 s), regardless of when those frames actually occurred. `frameTimes` is extracted and thrown away, `goCue` is not subtracted, and the side/behaviour clock offset recorded in `obj.sglx.bitcode` (the reference's `findVideoOffset.m`) is never computed. Because trial durations vary (frame counts of 2998, 2920, 2718, … were observed), the time-warping factor also differs from trial to trial.

ii.
```python
def resample_trace_to_bins(trace, ntime):
    trace = np.asarray(trace, dtype=float).squeeze()
    if trace.ndim != 1 or trace.size == 0:
        return None
    x_old = np.linspace(0.0, 1.0, trace.size)
    x_new = np.linspace(0.0, 1.0, ntime)
    return np.interp(x_new, x_old, trace).astype(np.float32)
```

iii. No justification is offered. Step 69 of the trajectory frames the goal correctly — *"align/resample each trial's motion-energy trace to the go-cue-centered neural bins"* — but the implementation resamples without aligning, and `obj.sglx`/`bitcode` and the reference's video-offset logic are never mentioned anywhere in the notes.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. `obj.traj[1]` — the **bottom** camera — specifically the `top_paw` feature's x/y/likelihood channels, falling back to `bottom_paw` if `top_paw` is absent or has fewer than 2 valid frames. Same v7.3-only restriction as the tongue, so the same 11 v5 sessions are all `not_visible`.

ii.
```python
feat1, ts1, ft1 = extract_h5_traj_view(h, view1, tr)
spd1, valid1 = speed_from_ts(ts1, feat1, ['top_paw','bottom_paw'])
```

iii. CONVERSION_NOTES.md Step 12: *"In view 1, there are explicit paw features (`top_paw`, `bottom_paw`) with ts shape (10, 3, 2998). Therefore paw velocity is in fact available."* No reason is given for preferring `top_paw`, nor is the tracking reliability of `bottom_paw` examined.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue: `speed_from_ts` with a 0.5 likelihood cut, per-frame Euclidean displacement without dividing by the frame interval and without smoothing, NaN-filling of invalid frames with the trial median, then linear resampling of the trace and of the validity mask onto the 220 bins. No normalisation (only one camera is involved, so none is needed).

ii.
```python
spd1, valid1 = speed_from_ts(ts1, feat1, ['top_paw','bottom_paw'])
if spd1 is not None:
    rr = resample_trace_to_bins(np.nan_to_num(spd1, nan=np.nanmedian(spd1[np.isfinite(spd1)]) if np.isfinite(spd1).any() else 0.0), ntime)
    vv = resample_trace_to_bins(valid1.astype(float), ntime)
    paw_res[tr] = (rr, vv >= 0.5)
    paw_vals.extend(rr[(vv >= 0.5)].tolist())
```

iii. Same as 7-b — the tongue and paw share one code path; no paw-specific justification is documented.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session median of the pooled resampled speeds over valid bins; `>= median` → 1, `< median` → 0, invalid → 2. Exactly as for the tongue, and matching the spec.

ii.
```python
if len(paw_vals) > 0:
    med = np.nanmedian(np.asarray(paw_vals, dtype=float))
    for tr, item in enumerate(paw_res):
        if item is not None:
            rr, valid = item
            paw_out[tr, valid] = (rr[valid] >= med).astype(np.int64)
```

iii. Directly from the decoder spec's per-session 50th-percentile rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, and identically broken: `resample_trace_to_bins` stretches the trial's frame axis onto the 220-bin grid, `frameTimes` (`ft1`) is discarded, no go-cue subtraction and no video-clock offset are applied.

ii.
```python
feat1, ts1, ft1 = extract_h5_traj_view(h, view1, tr)   # ft1 discarded
...
rr = resample_trace_to_bins(..., ntime)
```

iii. No justification documented; see 7-d.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<session>.mat` sidecar in the same folder, field `me.data`, which is an object array with one motion-energy trace per trial (one value per camera frame). A fallback path reads `obj.me` from the session file, but only for v5 sessions and only if it happens to be a numeric vector of length `Ntrials`. `me.moveThresh` is read and never used. Six of the 20 sessions end up with `no_video` everywhere because their sidecar layout is not handled (see 10).

ii.
```python
def load_motion_energy_sidecar(session_name):
    f = BASE / f'motionEnergy_{session_name}.mat'
    if not f.exists():
        return None, None
    try:
        me = sio.loadmat(str(f), squeeze_me=True, struct_as_record=False)['me']
        if hasattr(me, 'data') and hasattr(me, 'moveThresh'):
            return np.array(me.data, dtype=object), float(me.moveThresh)
        if isinstance(me, np.ndarray) and me.dtype.names is not None:
            ...
    except Exception:
        pass
    return None, None
```

iii. CONVERSION_NOTES.md Step 12: *"`motionEnergy_*.mat` stores per-trial time series, not scalar per-trial values. For JEB11_2022-05-10, `me.data` is a length-365 object array where each element is a float vector of varying length (e.g. 2998, 2920, 2718 samples)."* Step 5: *"`motionEnergy_*.mat` `me.data` or session `obj.me` → output: motion energy ... class 2 for no video."*

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling. Each trial's trace is passed straight to `resample_trace_to_bins` and thresholded — no smoothing, no differentiation, no spatial reduction (that was already done upstream by the authors). A guard requires `len(trial_traces) == Ntrials` before anything is used.

ii.
```python
trial_traces = np.asarray(me_data, dtype=object).ravel()
if trial_traces.shape[0] == ntr:
    session_vals = []
    resampled = []
    for trc in trial_traces:
        rr = resample_trace_to_bins(trc, ntime)
        if rr is None:
            resampled.append(None)
        else:
            resampled.append(rr)
            session_vals.extend(rr.tolist())
```

iii. Implicit in Step 12's framing that motion energy is a ready-made per-frame trace that only needs putting on the neural grid.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session median over all resampled values pooled across trials; `>= median` → 1 (`ge50`), `<` → 0 (`lt50`). Because the trace has no validity mask, class 2 (`no_video`) is used only as an all-or-nothing session-level fallback: a session either gets a real 50/50 split or is entirely `no_video`. Realised: `lt50 0.344 / ge50 0.344 / no_video 0.313`.

ii.
```python
if len(session_vals) > 0:
    med = np.nanmedian(np.asarray(session_vals, dtype=float))
    for i, rr in enumerate(resampled):
        if rr is not None:
            motion[i] = (rr >= med).astype(np.int64)
...
OUTPUT_VALUES[5] = ['lt50', 'ge50', 'no_video']
```

iii. Directly from the decoder spec's per-session 50th-percentile rule, with `no_video` reserved for sessions where the sidecar could not be read.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same `resample_trace_to_bins` linear stretch as the tongue and paw: the first frame of the trial lands in bin 0, the last in bin 219. No camera frame times, no video-clock offset, and no go-cue subtraction. The frames belong to the side camera, but since frame times are never consulted this makes no difference to the code.

ii.
```python
rr = resample_trace_to_bins(trc, ntime)
...
motion[i] = (rr >= med).astype(np.int64)
```

iii. No justification documented; see 7-d. Step 10's motion-energy sanity check (*"motion-energy resample/discretization for one raw trial matched converted output exactly (`np.allclose=True`)"*) re-applies the same resampling function, so it verifies reproducibility rather than alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Five mechanisms, most of them silent. (1) **Broad exception swallowing**: `try/except Exception: pass` wraps the per-trial trajectory extraction, the whole video branch, the motion-energy branch, and the sidecar loader, so a parse failure is indistinguishable from genuinely absent data. (2) **Format fallback**: `load_session` tries `h5py` then `scipy.io.loadmat`. (3) **Absent streams → trailing class**: tongue/paw default to 2 (`not_visible`) and motion energy to 2 (`no_video`) for any trial or session the code could not populate; the arrays are pre-filled with 2 before any attempt is made. (4) **Session drops**: sessions with no `obj.clu` (2 of 22) and with fewer than 10 retained units are removed. (5) **Quality-label typo fix**: `normq` strips NUL bytes and rewrites `mutli` → `multi`.

What is *not* handled: the 66 trials in 3 sessions that run past the end of the ephys recording (flagged by the verifier as "all neural data is zero") are kept with an all-zero neural matrix; the 11 v5-format sessions' video is never attempted at all; two of the three `motionEnergy` file layouts (bare cell array, and doubly-wrapped `{data:{data, moveThresh}}`) fall through the loader and return `None`; per-trial `bp` fields are never truncated to `Ntrials`; and invalid tracking frames are imputed with the trial's median speed before resampling rather than being left missing.

ii.
```python
tongue = np.full((ntr, ntime), 2, dtype=np.int64)
paw = np.full((ntr, ntime), 2, dtype=np.int64)
motion = np.full((ntr, ntime), 2, dtype=np.int64)

if session.get('format') == 'h5py':
    try:
        t_out, p_out = build_video_outputs_h5(session, ntime)
        ...
    except Exception:
        pass
```
```python
rr = resample_trace_to_bins(np.nan_to_num(spd0, nan=np.nanmedian(spd0[np.isfinite(spd0)]) if np.isfinite(spd0).any() else 0.0), ntime)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: *"Motion/video missingness: Encode unavailable video-derived outputs with class 2 (`not visible` / `no video`) rather than dropping trials."* Step 10 claims *"Motion-energy sidecar files had heterogeneous MATLAB structures; loader was made robust"* — the loader in fact still handles only one of the three layouts present in the data, and six sessions are all-`no_video` as a result. Step 9 acknowledges the video gap (*"Video-derived outputs remain unavailable in some sessions, leading to class-2-only tongue/paw/motion outputs there"*) without resolving it.

## 11-a. What are the most time-consuming steps of the code?

i. The notes do not answer this — the "Code inefficiencies identified" and "Speed-ups Implemented" sections of CONVERSION_NOTES.md were left as unfilled template, and the Step 7 runtime table says only *"~1-3 s/session in sample mode; TBD for full dataset"*. The script does print per-session elapsed times. From the actual run (55 s total for 20 sessions), the dominant costs are: (1) `build_neural_trials`, which is an `n_trials × n_units` double loop each iteration of which builds a boolean mask over the unit's *entire* spike-time vector and calls `np.histogram` — this accounts for the 2.4–3.7 s seen on the v7.3 sessions; and (2) file reading, including a **second** full open of the HDF5 file in `build_video_outputs_h5`. Sessions where the video branch is skipped (the v5 ones) finish in 0.2–0.8 s, confirming the video re-read is a substantial share.

ii.
```python
for tr in range(1, session['ntrials'] + 1):
    for i, u in enumerate(units):
        mask = (u['trial'] == tr)          # full-length scan, ntrials x nunits times
        if np.any(mask):
            counts, _ = np.histogram(u['trialtm'][mask], bins=edges)
```
```python
with h5py.File(BASE / f"data_structure_{session['session_name']}.mat", 'r') as h:   # re-opened
```

iii. No justification documented; the bottleneck analysis required by Step 6/7 of the instructions was not performed.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Not documented. The clear candidate is the `build_neural_trials` double loop: all trials of one unit can be binned in a single `np.histogram2d(trial_index, spike_time, bins=[trial_edges, time_edges])` call, and the Gaussian smoothing can be applied to the whole `(n_trials, n_bins)` matrix at once instead of row by row through `np.convolve`. The `np.any(mask)` test is also redundant work. Secondary candidates: the per-trial output `np.vstack` loop in `build_outputs` (could be one `(n_trials, 6, 220)` array), `build_inputs`' per-trial `arr.copy()`, and the `session_vals.extend(rr.tolist())` accumulation, which converts arrays to Python lists to compute a median.

ii.
```python
    for i, u in enumerate(units):
        mask = (u['trial'] == tr)
        if np.any(mask):
            counts, _ = np.histogram(u['trialtm'][mask], bins=edges)
            arr[i] = gaussian_smooth(counts.astype(np.float32) / dt, smooth_sigma)
```
```python
    for tr in range(ntr):
        out = np.vstack([...])
        outputs.append(out)
```

iii. No justification documented.

## 11-c. What processing does the code repeat multiple times?

i. Not documented. Three real repetitions exist. (1) `select_units` — and therefore the whole quality + firing-rate pass over every cluster — is called once inside the `>= 10` session filter and again for every session inside the main loop. (2) The v7.3 session file is opened and parsed twice: once by `load_session_h5` and again by `build_video_outputs_h5`. (3) Every session dict is fully loaded into memory up front (`sessions = [load_session(f) for f in files]`) before any filtering, so the spike data of sessions that are subsequently dropped is read for nothing. There is no caching of the trajectory arrays, so tongue (view 0) and paw (view 1) each re-dereference their own HDF5 references per trial, which is necessary but is done inside the same second file handle.

ii.
```python
sessions = [load_session(f) for f in files]
sessions = [s for s in sessions if s['has_clu']]
sessions = [s for s in sessions if len(select_units(s)) >= 10]   # pass 1
...
for si, sess in enumerate(sessions, start=1):
    units = select_units(sess)                                    # pass 2
```

iii. No justification documented.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Not documented. The script loads and then never uses: `bp.ev.sample`, `bp.ev.reward`, `bp.early`, `bp.L` (only `R`/`L` as a pair is needed and `L` is redundant), `clu.site`, and — on the v5 path — `obj.traj` and `obj.me`, which are stored in the session dict and never read. `clu.tm` (the full absolute spike-time vector of every cluster, the largest single array read) is loaded only to compute the scalar `n_spikes / duration` firing rate. `me.moveThresh` is parsed and returned but never used. In `build_video_outputs_h5`, `frameTimes` (`ft0`, `ft1`) is decoded for every trial and every view and immediately discarded. `build_inputs` allocates an independent copy of the same 220-element vector for each of the 7,064 trials. Finally, the `--show-processing` flag is parsed but has no effect — no plots are produced, which the notes concede.

ii.
```python
site = int(np.array(h[clu['site'][i, 0]][()]).squeeze()) if 'site' in clu else -1
...
'traj': getattr(obj, 'traj', None),
'me_obj': getattr(obj, 'me', None),
```
```python
feat0, ts0, ft0 = extract_h5_traj_view(h, view0, tr)   # ft0 discarded
```
```python
ap.add_argument('--show-processing', action='store_true')   # never referenced again
```

iii. CONVERSION_NOTES.md Step 6 notes one of these: *"`--show-processing` currently does not emit plots"*, repeated in Step 7 as *"a workflow shortcoming to note."* The rest is undocumented.
