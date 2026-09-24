# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing the `data/` tree (a **relative** path, so the script only works when run from `/app`). Every subdirectory is scanned for `data_structure_*.mat` and `motionEnergy_*.mat`, and the two files are paired by the filename stem `<anm>_<date>`. Only stems that have **both** a data structure and a motion-energy file are processed, which incidentally excludes the two behaviour-only folders (`DelayInhibition_BilatMC_Behavior`, `GoCueInhibition_BilatMC_Behavior`) that contain 73 `data_structure_*.mat` files but no motion-energy files.

`data_structure_*.mat` is read **only** with `h5py` (MATLAB v7.3). Rather than a struct-tree walk, specific fields are pulled out by name: `bp.{L,R,Ntrials,autowater,bitRand,early,hit,miss,no}`, `bp.ev.{bitStart,delay,goCue,lickL,lickR,reward,sample}`, `clu.{tm,site,quality,trialtm,trial}`, `traj.{featNames,ts,frameTimes,fn,NdroppedFrames}` and the char fields of `meta`. Any file that is not HDF5 raises `OSError` and the **whole session is silently skipped**; 11 of the 44 analysed sessions (JEB23 ×2, JEB24 ×8 are MATLAB v5, plus JEB6 whose `clu` lacks `trialtm`) are lost this way. `motionEnergy_*.mat` is read with `scipy.io.loadmat`.

Only the **first probe** is ever read: `obj['clu'][()].reshape(-1)[0]`. 15 of the sessions were recorded with two probes, so the second probe's units are always discarded, and in the sessions where the authors analysed probe 2 (e.g. EKH1_2021-08-07, EKH3_2021-08-11, JEB13_2022-09-13/14, JEB15_2022-07-29) the AI reads a different population than the reference.

Net result: **33 sessions, 12 subjects, 8,598 trials, 4,176 units**.

ii.
```python
def discover_sessions(data_root):
    sessions = {}
    for subdir in Path(data_root).iterdir():
        if not subdir.is_dir():
            continue
        for f in subdir.glob('data_structure_*.mat'):
            stem = f.stem.replace('data_structure_', '')
            sessions.setdefault(stem, {})['data_structure'] = f
        for f in subdir.glob('motionEnergy_*.mat'):
            stem = f.stem.replace('motionEnergy_', '')
            sessions.setdefault(stem, {})['motion_energy'] = f
    return sessions
```

```python
def load_data_structure(path):
    out = {}
    try:
        hfile = h5py.File(path, 'r')
    except OSError:
        return None                      # <- v5 .mat files are dropped here
    with hfile as h:
        obj = h['obj']
        ...
        clu_ref = obj['clu'][()].reshape(-1)[0]      # <- probe 1 only
        clu = deref(h, clu_ref)
```

```python
sessions = discover_sessions('data')
keys = sorted(k for k, v in sessions.items()
              if 'data_structure' in v and 'motion_energy' in v)
```

iii. The AI documented that "Mixed MATLAB formats are present: `data_structure_*.mat` are MATLAB v7.3/HDF5, while at least `motionEnergy_*.mat` are older MAT files readable with `scipy.io.loadmat`", i.e. it assumed *all* data-structure files were v7.3. When the full run hit non-HDF5 files its stated reasoning was purely operational: "some `data_structure_*.mat` files are not HDF5 (`file signature not found`), so `load_data_structure` needs to gracefully skip or handle non-v7.3 MAT files rather than crashing" (trajectory step 95), and later "Given that many sessions were already successfully processed, this is an acceptable curation step if documented" (step 135). CONVERSION_NOTES.md records only "Sessions with unreadable `data_structure_*.mat` files are skipped with explicit logging." No justification is given anywhere for reading only the first probe.

## 1-b. How are the data split into subjects?

i. The subject is the part of the filename stem before the first underscore. Subjects are accumulated in first-encountered (alphabetical, since `keys` is sorted) order and `subject_idx` indexes into that list. 12 subjects result (the reference gets 14; JEB24 and JEB6 are missing because their sessions were dropped at load time).

ii.
```python
def infer_subject_session(stem):
    parts = stem.split('_')
    return parts[0], '_'.join(parts[1:])
```
```python
if sess['subject'] not in subject_to_idx:
    subject_to_idx[sess['subject']] = len(subjects)
    subjects.append(sess['subject'])
...
'subjects': subjects,
'subject_idx': np.array([subject_to_idx[s['subject']] for s in out_sessions], dtype=np.int64),
```

iii. Not explicitly justified. The notes record "Preliminary subject/date parsing across filenames found 18 unique subjects: EKH1, EKH3, JEB11, ...", i.e. the filename was taken as the authoritative subject identifier.

## 1-c. How are the data split into sessions?

i. One `data_structure_<anm>_<date>.mat` file = one session = one element of `neural`/`input`/`output`. Fixed-delay and randomized-delay sessions are pooled without distinction. A session is dropped if: the file is not HDF5, `clu` lacks `trialtm`/`trial`, there is no paired motion-energy file, motion energy is empty or shorter than the largest trial index, no paw feature is found, fewer than 2 trials survive, or fewer than 10 units survive. 33 of 45 candidate stems are kept.

ii.
```python
for stem in keys:
    print(f'processing {stem}', flush=True)
    sess = process_session(stem, sessions[stem], edges, centers)
    if sess is None:
        print(f'skipped {stem}', flush=True)
        continue
```
```python
if len(neural) < 2 or len(kept_units) < 10:
    print(f'  skip_reason: insufficient_neural trials={len(neural)} units={len(kept_units)}')
    return None
```

iii. The ≥10-unit rule is justified from the paper: "Recording sessions included only if they had at least 10 units." The ≥2-trial rule comes from the target-format requirement ("There needs to be at least two trials within each session"). The requirement that a motion-energy file exist is justified as "Restrict to sessions with all required outputs available: Need neural + behavior + trajectory/motion-energy data for decoder outputs." The remaining exclusions are failure-handling, not curation decisions.

## 1-d. How are the data split into trials?

i. Trials are the rows of the `obj.bp` table. `Ntrials` gives the count, and every per-trial field (`L`, `R`, `hit`, `miss`, `no`, `early`, `autowater`, `ev.goCue`) is read as a flat vector. Spikes are assigned to trials with `clu.trial` (converted from 1-based to 0-based); camera frames and motion energy are indexed per trial by the same raw trial number. A boolean `trial_mask` over the `Ntrials` rows selects the kept trials, and `np.where(trial_mask)[0]` gives the raw trial indices used consistently by the neural, behavioural and video code paths, so the ordering is preserved across streams.

ii.
```python
def valid_trials(bp):
    n = int(np.array(bp['Ntrials']).reshape(-1)[0])
    mask = np.ones(n, dtype=bool)
    ...
    return mask
```
```python
tr = np.array(tr, dtype=float).reshape(-1).astype(int) - 1   # clu.trial is 1-based
...
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. Not explicitly justified beyond the Step 2 note that `bp` "contains `Ntrials` plus trial-wise arrays `L`, `R`, `autowater`, `bitRand`, `early`, `hit`, `miss`, and `no`, each with length 365 ... This confirms that go-cue alignment and per-trial labels for lick direction, context, and outcome can be derived from the native behavioral structure."

## 1-e. How are trials filtered based on quality controls?

i. Two flags are used: `bp.early` (early lick) and `bp.no` (**ignore / no-response**). Both are dropped. Photostimulation trials (`bp.stim.enable`) are **not** dropped — `stim` is never even loaded — so 187 photoinactivation trials across 5 sessions (JEB15_2022-07-29, JEB6, JEB7 ×2, JGR3) enter the dataset. Trials that run past the end of the ephys recording are not checked for. Each flag is applied only if its length exactly equals `Ntrials`, otherwise it is silently ignored.

ii.
```python
def valid_trials(bp):
    n = int(np.array(bp['Ntrials']).reshape(-1)[0])
    mask = np.ones(n, dtype=bool)
    for name in ['early', 'no']:
        if name in bp:
            arr = np.array(bp[name], dtype=float).reshape(-1)
            if arr.size == n:
                mask &= (arr == 0)
    return mask
```

iii. Justified from the paper: "Early lick and ignore trials are omitted from analyses" and the Step 5 decision "**Exclude early and ignore trials**: Matches methods and analysis code." The AI did not reconcile this with the Decoder Task spec, which explicitly requires an `ignore` outcome class and a `none` lick-direction class. No mention of photostim trials appears anywhere in the notes or trajectory.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `clu.trialtm` (spike times relative to trial start) and `clu.trial` (the trial each spike belongs to), for the **first probe only**. `clu.quality` and `clu.site` are loaded but never used. `bp.ev.goCue` is loaded, and used for the video streams, but **is not used for the neural data**.

ii.
```python
clu_ref = obj['clu'][()].reshape(-1)[0]
clu = deref(h, clu_ref)
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    ...
```
```python
def bin_spikes_for_session(obj, trial_mask, edges):
    clu = obj['clu']
    if 'trialtm' not in clu or 'trial' not in clu:
        return None, None
    trialtm = clu['trialtm']
    trialid = clu['trial']
```

iii. "`clu.trialtm` and `clu.trial` dereference to per-unit spike-time arrays and matching trial-index arrays, confirming neural data are stored as spike times that must be binned by trial and time." The mapping table lists "`obj.clu.trialtm` + `obj.clu.trial` → neural | Bin spike times by trial on a common go-cue-aligned time grid".

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into the 75 ms bin edges, per unit and per trial, and stored as **raw spike counts** (`float32`). No conversion to Hz (a fixed ×13.33 scale factor), no Gaussian smoothing, no normalisation, no baseline subtraction. Each trial becomes an `(n_units, 66)` matrix.

ii.
```python
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
    unit_trials.append(counts.astype(np.float32))
...
return [np.stack(x, axis=0) for x in neural_trials], np.array(kept_units, dtype=int)
```

iii. Step 5 decision 2: "**Use spike counts on a fixed time grid**: Raw neural data are stored as per-unit spike times by trial, so conversion will bin to a common grid across sessions." Smoothing is mentioned as something to look for in the reference ("neural preprocessing terms (`lowFR`, PSTH smoothing)") but no conclusion is recorded and none is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only a firing-rate filter. `clu.quality` is loaded but **never referenced**, so manually curated `garbage` / `noisy` / `poor` / `real?` clusters are all kept. The rate is computed as (total spikes of the unit over the *entire* session, including spikes outside the analysis window and in excluded trials) divided by a surrogate duration `(t_end - t_start) × n_kept_trials = 5 s × n_trials`, and units with rate ≤ 1 Hz are dropped. Because real trials last ~10 s, this surrogate denominator is roughly half the true recording time, so the effective threshold is ≈0.45 Hz of true rate, not 1 Hz. Sessions with <10 surviving units are dropped. 4,176 units survive (reference: 1,954).

ii.
```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    st = np.array(st, dtype=float).reshape(-1)
    tr = np.array(tr, dtype=float).reshape(-1).astype(int) - 1
    if st.size == 0:
        continue
    approx_duration = (edges[-1] - edges[0]) * max(trial_mask.sum(), 1)
    fr = st.size / max(approx_duration, 1e-9)
    if fr <= 1.0:
        continue
    kept_units.append(ui)
```

iii. Step 5 decision 4: "**Filter units by firing rate > 1 Hz and sessions with >=10 units**: Matches methods and `removeLowFRClusters` logic", and the Step 3 curation table: "All units with firing rates > 1 Hz were included in most analyses", "Unit isolation categories come from manual curation after JRCLUST and/or Kilosort + Phy." The isolation-category rule was noted but never implemented, and no reason is given for omitting it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **It is not.** `bin_spikes_for_session` histograms `clu.trialtm` directly into the edges `[-2.5, 2.45]` without ever subtracting `bp.ev.goCue`. `trialtm` is time from **trial start** (range ≈ −0.49 s to +10.3 s), and the go cue occurs a median of 2.50 s after trial start. The stored window therefore covers roughly −3.0 s to −0.05 s relative to the go cue, i.e. the neural data stops essentially at the moment the trial is supposed to be centred on, and the first 26 of 66 bins (−2.50 s to −0.55 s) are structurally all-zero in every trial of every session:

```
mean spikes/bin, session 0, first 50 trials:
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0.596 3.041 2.991 2.989 ... 3.234]
```

The video streams *are* go-cue aligned (7-d), so the neural and behavioural streams are offset from each other by ~2.5–3 s.

ii. The whole binning path, with no `goCue` anywhere:
```python
def bin_spikes_for_session(obj, trial_mask, edges):
    clu = obj['clu']
    trialtm = clu['trialtm']
    trialid = clu['trial']
    for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
        st = np.array(st, dtype=float).reshape(-1)
        tr = np.array(tr, dtype=float).reshape(-1).astype(int) - 1
        ...
        for raw_t in np.where(trial_mask)[0]:
            counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```
Compare the video path, which *does* align:
```python
rel_t = (ft - 0.5) - float(align_time)      # align_time = go[tr]
```

iii. The AI states the intent repeatedly — Step 4: "Use go cue as the temporal alignment event in converted data"; Step 5 decision 1: "**Use go cue as alignment event**: Reference code default is `alignEvent = 'goCue'`, and the task explicitly requires go-cue alignment"; metadata `temporal_alignment_event: 'Go cue onset'`. No justification is given for the implementation, because the AI never noticed the omission. Its Step 10 sanity check — "converted spike-count vector for one unit/trial matches raw histogrammed `clu.trialtm` counts" — reproduces the same un-aligned histogram and therefore passes while the data is wrong.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 75 ms bins over a nominal window of −2.5 s to +2.5 s. `np.arange(-2.5, 2.5 + 1e-9, 0.075)` yields 67 edges / **66 bins** ending at +2.45 s, so the window is actually truncated 50 ms early; `metadata['off_end']` is set to 2.45 accordingly while `off_start` is −2.5. Spikes are histogrammed directly into these bins (no fine binning followed by rebinning). The video traces are point-sampled onto the 66 bin centres and motion energy is resampled to 66 points, so all streams share the 66-bin axis. `metadata['time_bin_size'] = 75.0` ms.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
    return edges, centers
```
```python
'time_bin_size': 75.0,
'off_start': float(edges[0]),
'off_end': float(edges[-1]),
```

iii. Taken from the reference decoding code: "`DLC_ContextDecoding.m` explicitly sets `rez.binSize = 75` ms and computes `rez.dt = floor(rez.binSize / (params(1).dt*1000))`, suggesting native behavioral/neural streams are rebinned to 75 ms for decoding analyses" and "Prior code inspection indicates go-cue-centered timing and 75 ms decoding bins for DLC features." The ±2.5 s window is not explicitly justified in the notes.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable. It is the vector of bin centres of the analysis grid, constructed from the `(t_start, t_end, bin_size)` constants and repeated identically for every trial of every session. Range −2.4625 s to +2.4125 s.

ii.
```python
edges, centers = build_time_grid()
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```
```python
'input_names': ['time_from_go_cue'],
```

iii. Step 5 decision 6: "**Represent decoder input as continuous time from go cue**: Single input channel shared across trials/sessions", mapped from "time relative to go cue → input[0] | Continuous time vector repeated for each trial".

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Only `edges[:-1] + bin_size/2`, cast to `float32`, broadcast to shape `(1, 66)` per trial. No per-trial variation.

ii.
```python
centers = edges[:-1] + bin_size_s / 2
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. None beyond the above; the quantity is defined by construction.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input vector is the centres of exactly the same `edges` array that the spikes are histogrammed into, so bin *k* of the input and bin *k* of the neural matrix are the same bin index. However, because the spikes were binned in **trial-start time** rather than go-cue time (2-d), the input labelled "time from go cue" does not actually give the time from the go cue of the neural sample it sits next to — it is off by the go-cue latency (median 2.50 s). The video outputs *are* in go-cue time, so the input is correct for them and wrong for the neural data.

ii.
```python
edges, centers = build_time_grid()          # one grid, built once
...
counts, _ = np.histogram(st[tr == raw_t], bins=edges)     # neural uses edges
inputs = [centers[None, :].astype(np.float32) for _ in idx]   # input uses centers
```

iii. Implicit: one grid is built in `main` and passed to every stream. The AI never checked the actual alignment of the neural stream against the input.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `bp.L` and `bp.R` only. These are the **instructed / rewarded port** flags (exactly one of them is 1 on every trial), not a record of where the animal actually licked. `bp.hit` / `bp.miss`, which are what disambiguate the actual lick on error trials, are loaded but not used for this output.

ii.
```python
L = np.array(bp['L'], dtype=float).reshape(-1)
R = np.array(bp['R'], dtype=float).reshape(-1)
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
```

iii. Mapping table: "`obj.bp.L`, `obj.bp.R` → output[0] lick direction | Per-trial categorical label; left=0, right=1". The notes do not distinguish instructed direction from executed lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. `R > L` → 1 (right), else 0 (left). Two classes; `output_values[0] = ['left', 'right']`. The value is held constant across all 66 bins of the trial. Because ignore trials were removed (1-e) there is no `none` class, and because error (`miss`) trials keep the instructed side the label is the *opposite* of the animal's actual lick on those trials — 15.3 % of the kept trials per the verification output.

ii.
```python
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
...
np.full(n_bins, lick_dir[i], dtype=np.int64),
```
```python
'output_values': [
    ['left', 'right'],
    ...
```

iii. No further justification; the notes simply report the resulting distribution `[0.496, 0.504]` and treat the 0.585 validation balanced accuracy as "above chance; plausible for neural decoding" without investigating.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater`, one flag per trial.

ii.
```python
autowater = np.array(bp['autowater'], dtype=float).reshape(-1)
```

iii. "`getBlockNum_AltContextTask.m` derives block/context identity from transitions in `obj(sessix).bp.autowater` across trials, implying context labels are encoded in behavioral metadata rather than a standalone variable."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater > 0` → WC (0), otherwise DR (1), held constant across the 66 bins. This matches the reference exactly. Note that three sessions (JEB13_2022-09-24/25, JEB14_2022-08-22, JEB23 ×3) end up with only the DR class, which is correct for those sessions but makes them uninformative for this output.

ii.
```python
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
...
np.full(n_bins, context[i], dtype=np.int64),
```
```python
['WC', 'DR'],
```

iii. Mapping table: "context from `bp.autowater` / trial groups → output[1] behavioral context | Per-trial categorical label; WC=0, DR=1", with the coding taken from the Decoder Task spec order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` and `bp.miss`. (`bp.no` is used, but only as a trial-exclusion filter, not as an outcome class.)

ii.
```python
hit = np.array(bp['hit'], dtype=float).reshape(-1)
miss = np.array(bp['miss'], dtype=float).reshape(-1)
```

iii. Mapping table: "`obj.bp.hit`, `obj.bp.miss` → output[2] outcome | Per-trial categorical label; incorrect=0, correct=1 | Exclude ignore / early trials."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `hit > miss` → 1 (correct), else 0 (incorrect), constant across bins. Two classes only; the `ignore` class required by the Decoder Task spec does not exist because those trials were filtered out in `valid_trials`. Resulting distribution 0.153 incorrect / 0.847 correct.

ii.
```python
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
...
np.full(n_bins, outcome[i], dtype=np.int64),
```
```python
['incorrect', 'correct'],
```

iii. As above: "Exclude early and ignore trials: Matches methods and analysis code." The conflict with the three-class spec is never raised.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Nominally `obj.traj{1}` (side camera): `featNames`, `ts` and `frameTimes`, with the feature selected by exact name match on `'tongue'` (falling back to `left_tongue`, `right_tongue`, then substring match). Only the side camera is used; the bottom camera's `top_tongue` is not. `bp.ev.goCue` is used for alignment. The likelihood channel `ts[..., 2]` is **not** read.

**In practice the code does not read the tongue at all.** As h5py returns it, `ts` has shape `(n_features, 3, n_frames)` — e.g. `(7, 3, 1792)` — but the code indexes it as `(n_frames, 3, n_features)`. With `feat_idx = 0`, `arr[:, :2, 0]` returns the *x,y of all 7 side-camera body parts at frame 0*, a `(7, 2)` array, not the tongue's trajectory:

```
xy actually used, trial 0:        featNames order
[[  nan   nan]                    tongue
 [  nan   nan]                    left_tongue
 [  nan   nan]                    right_tongue
 [189.2 141.7]                    jaw
 [261.2 151.0]                    trident
 [ 82.2  47.0]                    nose
 [141.6 158.9]]                   lickport
```
The frame-time vector (1792 entries) then mismatches this 7-row array, so the code replaces it with 7 points linearly spaced over the trial, and the "tongue velocity" trace is a 7-knot interpolation of the spatial layout of the face.

ii.
```python
tongue_idx = pick_feature(side_names, ['tongue', 'left_tongue', 'right_tongue'])
...
tongue.append(interp_feature_velocity(ts_side, ft_side, go[tr], centers, tongue_idx, tongue=True))
```
```python
def interp_feature_velocity(ts, frame_times, align_time, centers, feat_idx, tongue=False):
    arr = np.array(ts, dtype=float)
    if arr.ndim != 3 or feat_idx >= arr.shape[2]:
        return np.zeros(len(centers), dtype=np.float32)
    xy = arr[:, :2, feat_idx]                 # <- axes transposed
    ft = np.array(frame_times, dtype=float).reshape(-1)
    if ft.size != xy.shape[0]:
        ...
        ft = np.linspace(ft.min(), ft.max(), xy.shape[0])
```

iii. The AI explicitly reasoned itself into this layout from the reference documentation: "`WorkingWithDataObjs.m` explicitly states ... `obj.traj.ts` has dimensions `(frames, [x,y,confidence], bodypart)` ... our current trajectory loader is doubly wrong: it loads only one view and assumes the ts array orientation is `(features, 3, frames)`" (trajectory step 119). That is the MATLAB-side layout; h5py returns the transpose, and the AI never re-checked the shape after the change. CONVERSION_NOTES.md records the change as a fix: "Earlier trajectory-loading bug caused degenerate tongue/paw outputs; fixed by loading both camera views and using correct `(frames, coords, bodypart)` trajectory layout."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. `x` and `y` are linearly interpolated (`np.interp`, NaN outside the data range) onto the 66 bin centres — i.e. point-sampled at 13.3 Hz from a 400 Hz camera, with no averaging and no smoothing. Velocity is `np.gradient(x)` / `np.gradient(y)` **with respect to bin index**, not time (a constant scale factor given uniform bins). For the tongue, NaN velocities are replaced with **0**, so "not visible" and "not moving" become the same value. Speed is `sqrt(xv² + yv²)`. There is no explicit likelihood threshold — the code relies on the authors having already NaN-ed low-likelihood coordinates.

ii.
```python
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
y = np.interp(centers, rel_t, xy[:, 1], left=np.nan, right=np.nan)
xv = np.gradient(x)
yv = np.gradient(y)
if tongue:
    xv[np.isnan(xv)] = 0
    yv[np.isnan(yv)] = 0
...
return np.sqrt(xv**2 + yv**2).astype(np.float32)
```

iii. Mapping table: "tongue-related `traj` features → output[3] tongue velocity | Derive per-time-bin scalar, discretize by session median". No justification is recorded for the NaN→0 choice, for point-sampling rather than averaging, or for omitting smoothing and the likelihood cut.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A strict `>` comparison against the per-session `np.nanmedian` of the whole `(n_trials, 66)` tongue matrix, giving **two** classes, `['low', 'high']`. The `2: not visible` class required by the Decoder Task spec is not produced. Because untracked bins were set to speed 0 (7-b), they are pooled into the `low` class and drag the median down. Four sessions (JEB19 ×4) come out essentially degenerate: high-class fractions 0.000, 0.002, 0.003, 0.004.

ii.
```python
tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
```
```python
['low', 'high'],
```

iii. Step 5 decision 7: "**Discretize continuous behavioral outputs per session at 50th percentile**: Required by task for tongue velocity, paw velocity, and motion energy." The third class is never mentioned; the degenerate JEB19 sessions were visible in `verification_full_out.txt` and were not investigated (Step 12 records only "All outputs are above chance on the full dataset").

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted by a **hard-coded 0.5 s** and then the trial's go cue is subtracted: `rel_t = (frameTimes - 0.5) - goCue[trial]`. The true camera-to-behaviour offset, recoverable per session from the bitcode (`sglx.bitcode.bitstart/sglx.fs` minus `bp.ev.bitStart`, as in `findVideoOffset.m`), is 0.49 s in most sessions but **0.99 s in the JEB19 sessions**, so those four sessions are misaligned by 0.5 s. Values are then sampled at the shared bin centres. Since the neural data is in trial-start time (2-d), the tongue stream and the neural stream are offset from each other by roughly the go-cue latency (~2.5 s).

ii.
```python
rel_t = (ft - 0.5) - float(align_time)
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
```

iii. Taken from the reference documentation: "`frameTimes` should be shifted by 0.5 s to sync spikeGLX and camera recordings" (`WorkingWithDataObjs.m`, trajectory step 119). The per-session bitcode offset is never computed, and `sglx` is never loaded.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Nominally `obj.traj{2}` (bottom camera), feature `top_paw` (with `bottom_paw`, then any name containing `paw`, as fallbacks). Same `ts` / `frameTimes` / `goCue` inputs as the tongue. If no paw feature is found the whole session is dropped.

**As with the tongue, the code does not actually extract the paw.** With `feat_idx = 4` and the true h5py shape `(10, 3, 1792)`, `arr[:, :2, 4]` returns the x,y of all 10 bottom-camera body parts at frame 4 — a `(10, 2)` array — which is then spread over the trial as a 10-point time series.

ii.
```python
paw_idx = pick_feature(bottom_names, ['top_paw', 'bottom_paw', 'paw'])
if paw_idx is None:
    return None, None
...
paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers, paw_idx, tongue=False))
```

iii. Mapping table: "paw-related `traj` features → output[4] paw velocity ... Need to identify exact paw feature(s) from `featNames`", and trajectory step 120: "Update `build_traj_outputs` to use view 1 for tongue and view 2 for paw". The choice of `top_paw` over `bottom_paw` is only implicit in the candidate ordering and is not justified.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same interpolate-then-`np.gradient` pipeline as the tongue, but with a different missing-data rule: NaN velocity components are replaced with the **nanmedian** of that component, and then the median is **subtracted** from both components, before taking the magnitude. So the stored quantity is the magnitude of the deviation of the (per-bin) velocity from its trial median, and untracked bins are given a fabricated value of exactly 0 deviation. If a whole component is NaN the trial is zero-filled.

ii.
```python
else:
    if np.all(np.isnan(xv)) or np.all(np.isnan(yv)):
        return np.zeros(len(centers), dtype=np.float32)
    xv = np.where(np.isnan(xv), np.nanmedian(xv), xv)
    yv = np.where(np.isnan(yv), np.nanmedian(yv), yv)
    xv = xv - np.nanmedian(xv)
    yv = yv - np.nanmedian(yv)
return np.sqrt(xv**2 + yv**2).astype(np.float32)
```

iii. No justification is recorded for the median-imputation or the median-subtraction; neither appears in the notes or in the trajectory reasoning.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Strict `>` against the per-session `np.nanmedian` of the whole paw matrix; two classes `['low', 'high']`. No `not visible` class. Resulting split 0.716 / 0.284 overall (not 50/50, because median-imputed bins sit exactly at the low end).

ii.
```python
paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
```

iii. Same as 7-c: Step 5 decision 7, "Discretize continuous behavioral outputs per session at 50th percentile."

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue: `rel_t = (bottom-camera frameTimes − 0.5) − goCue[trial]`, then sampled at the shared bin centres. Each camera's own `frameTimes` is used, which is correct in principle. The same two problems apply — the 0.5 s constant is wrong by 0.5 s for the JEB19 sessions, and the neural stream it is supposed to line up with is in trial-start time.

ii.
```python
ft_bot = bottom['frameTimes'][tr] if tr < len(bottom.get('frameTimes', [])) else np.array([])
...
paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers, paw_idx, tongue=False))
```
```python
rel_t = (ft - 0.5) - float(align_time)
```

iii. Same justification as 7-d (the 0.5 s shift documented in `WorkingWithDataObjs.m`).

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file, field `me.data`, one variable-length trace per trial. `normalize_motion_energy_data` unwraps the several layouts that occur (dict with `data`, structured array, nested dict, bare object array). `me.moveThresh` is loaded by `loadmat` but not used. `obj.me` is not consulted.

ii.
```python
def load_motion_energy(path):
    return sio.loadmat(path, simplify_cells=True)['me']

def normalize_motion_energy_data(me):
    data = me
    if isinstance(data, dict):
        data = data.get('data', data)
    elif hasattr(data, 'dtype') and getattr(data.dtype, 'names', None):
        names = list(data.dtype.names)
        if 'data' in names:
            data = data['data']
    if isinstance(data, dict):
        for key in ['data', 'trials', 'values']:
            if key in data:
                data = data[key]
                break
    ...
```

iii. "Representative `motionEnergy_*.mat` files store `me.data` as a per-trial object array (length matched to `bp.Ntrials`; 365 in the inspected session), with one variable-length continuous motion-energy trace per trial, plus a scalar `moveThresh`." The multi-layout unwrapping was added reactively: "for some sessions `me` is not a dict, causing `IndexError` on `me['data']` ... patch `normalize_motion_energy_data` to handle dicts, structured numpy voids/record arrays, and direct ndarrays" (trajectory step 75).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each trial's trace is **linearly resampled onto 66 points spanning its own full length** — `np.interp` from `linspace(0,1,len(trace))` to `linspace(0,1,66)`. No smoothing, no other transformation. Empty traces become 66 zeros.

ii.
```python
def rebin_variable_trace(values, n_bins):
    arr = np.array(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    xp = np.linspace(0, 1, arr.size)
    xnew = np.linspace(0, 1, n_bins)
    return np.interp(xnew, xp, arr).astype(np.float32)
```

iii. Mapping table: "`me.data` → output[5] motion energy | Rebin to common time grid, discretize by session median", and Step 4: "`traj.ts/frameTimes` and `me.data` are variable-length per trial ... Rebin/alignment to a common time grid around go cue is required." The implementation performs the rebin but not the alignment.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Strict `>` against the per-session `np.nanmedian` of the `(n_trials, 66)` matrix; two classes `['low', 'high']`. There is no `no video` class — sessions without a motion-energy file were dropped entirely rather than being marked. The result is an exact 0.500 / 0.500 split in every session.

ii.
```python
me_stack = np.stack(me_trials, axis=0)
me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)
```

iii. Same as 7-c/8-c: Step 5 decision 7.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. **It is not aligned to anything.** `rebin_variable_trace` ignores `frameTimes`, the 0.5 s video offset, and `goCue`; it stretches each trial's entire motion-energy trace to fill the nominal −2.5 → +2.45 s window. Bin *k* therefore corresponds to the fraction *k*/66 through whatever that trial's duration happened to be, which varies from trial to trial and, in the randomized-delay sessions, varies systematically with the delay length. Since the go cue sits at roughly 55 % of a typical fixed-delay trial, it lands near bin 36 (≈ +0.24 s) rather than bin 33 (0 s) — and nowhere near a fixed position in the randomized-delay sessions.

ii.
```python
me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
```
(no `frameTimes`, no `go[tr]`, no offset)

iii. None recorded. The AI's Step 10 sanity-check list included "Check that rebinned motion energy on selected trials matches raw `me.data` after applying the same time grid", but the check that was actually reported in the notes covers only trial counts, labels and spike counts.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Almost everything is handled by **dropping data or silently substituting a value**, rather than by marking it:

- Non-HDF5 (MATLAB v5) session files → `load_data_structure` returns `None`, whole session skipped (11 sessions).
- `clu` missing `trialtm`/`trial` → whole session skipped (JEB6).
- Missing motion-energy file, empty motion energy, or motion energy shorter than the largest trial index → whole session skipped.
- No paw feature found → whole session skipped.
- <10 units or <2 trials → whole session skipped.
- Untracked (NaN) tongue coordinates → velocity forced to 0.
- Untracked (NaN) paw coordinates → velocity imputed with the component median; if an entire component is NaN, the trial's paw trace becomes all zeros.
- `frameTimes` length ≠ number of rows in the (mis-indexed) `ts` array → frame times fabricated with `np.linspace`; if `frameTimes` is empty, fabricated as `arange(n)/400`.
- A per-trial `bp` flag whose length ≠ `Ntrials` → that filter is silently skipped for the session.
- `ts` not 3-dimensional, or the (mis-computed) feature index out of range → 66 zeros returned.

No NaNs reach the output, but the cost is that missing observations are indistinguishable from real zero-velocity observations, which is exactly what the spec's third output class was for.

ii.
```python
try:
    hfile = h5py.File(path, 'r')
except OSError:
    return None
```
```python
if ft.size != xy.shape[0]:
    if ft.size == 0:
        ft = np.arange(1, xy.shape[0] + 1, dtype=float) / 400.0
    else:
        ft = np.linspace(ft.min(), ft.max(), xy.shape[0])
```
```python
if arr.size == n:
    mask &= (arr == 0)
```

iii. Consistently framed as robustness rather than as a data decision: "the simplest robust fix is to skip sessions lacking required `clu` fields ... Given that many sessions were already successfully processed, this is an acceptable curation step if documented" (step 135); "This indicates the current converter is robust across heterogeneous session structures" (step 145). CONVERSION_NOTES.md Step 10 lists the skips under "Issues Found and Resolved".

## 11-a. What are the most time-consuming steps of the code?

i. The AI performed **no timing analysis**. The CONVERSION_NOTES.md Step 6 sections ("Code inefficiencies identified", "Code speedups added") and the Step 7 "Run Time Estimates" tables were left as empty template placeholders, and the script prints no timing information — `conversion_full_out.txt` contains only per-session "processing/kept" lines.

Objectively, the dominant cost is `bin_spikes_for_session`: for every unit it loops over every kept trial and evaluates `st[tr == raw_t]`, a full boolean scan of that unit's entire spike array, then calls `np.histogram`. That is O(n_units × n_trials × n_spikes_per_unit) — for a 300-unit, 350-trial session with ~10⁴ spikes/unit this is ~10⁹ element comparisons. HDF5 dereferencing of the per-trial `traj` cell arrays is the second cost.

ii.
```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)
        unit_trials.append(counts.astype(np.float32))
```

iii. None. Step 6 and Step 7 timing sections were never filled in despite the workflow requiring "Print timing information to find bottlenecks" and a time estimate.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Not identified by the AI. The clear candidate is the unit × trial double loop in `bin_spikes_for_session`, which a single `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, edges])` per unit — or `np.add.at` over a (unit, trial, bin) index — would replace, removing the inner loop entirely and the repeated `tr == raw_t` scans. `build_traj_outputs` also loops per trial doing two `np.interp` calls; that one is harder to vectorise because frame counts differ per trial. `me_trials` is a per-trial comprehension of `np.interp` calls, similarly bounded by variable lengths.

ii.
```python
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```
```python
for tr in trial_idx:
    ...
    tongue.append(interp_feature_velocity(...))
    paw.append(interp_feature_velocity(...))
```

iii. None recorded.

## 11-c. What processing does the code repeat multiple times?

i. Not identified by the AI. Actual repetitions:

- `np.where(trial_mask)[0]`, `trial_mask.sum()` and `approx_duration` are recomputed inside the per-unit loop, once per unit, although they are session constants.
- `st[tr == raw_t]` rescans the unit's whole spike vector once per trial.
- `pick_feature` re-scans `featNames` on every call, and `side.get('ts', [])` / `len(...)` are re-evaluated for every trial.
- The video offset is not computed at all (a hard-coded constant), so this one source of repetition is avoided by accident.
- `np.nanmedian` is computed once per stream per session, which is correct.

ii.
```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    ...
    approx_duration = (edges[-1] - edges[0]) * max(trial_mask.sum(), 1)
    ...
    for raw_t in np.where(trial_mask)[0]:
```

iii. None recorded.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Not identified by the AI. Actual waste:

- Fields loaded from HDF5 and never used: `clu.tm`, `clu.site`, `clu.quality`; `bp.bitRand`, `bp.no` (used only as a mask), `bp.ev.{bitStart, delay, lickL, lickR, reward, sample}`; `traj.fn`, `traj.NdroppedFrames`; the whole `meta` dict; `me.moveThresh`.
- Dead code: `read_dataset_maybe_refs`, `read_char_ref_array`, `read_numeric_ref_array` are defined and never called; the `--show-processing` flag is parsed but no plotting code exists anywhere in the script, so the option silently does nothing.
- 26 of the 66 neural bins (39 % of every stored matrix, ≈128 MB of the 330 MB pickle) are structurally all-zero because of the alignment bug in 2-d.
- The `bottom` camera's tongue features and the second probe's units are loaded into memory and discarded.

ii.
```python
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:   # tm/site/quality unused
```
```python
ap.add_argument('--show-processing', action='store_true',
                help='Save processing plots for up to 2 sessions')   # never read again
```
```python
def read_char_ref_array(h, ds):     # never called
def read_numeric_ref_array(h, ds):  # never called
```

iii. None recorded. CONVERSION_NOTES.md Step 7 states "No plots generated yet (`--show-processing` not used in latest successful sample run)", which understates the situation — the flag has no implementation.
