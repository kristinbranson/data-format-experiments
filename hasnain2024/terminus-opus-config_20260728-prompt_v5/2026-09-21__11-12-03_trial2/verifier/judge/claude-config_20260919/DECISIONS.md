# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The 44 sessions are **hard-coded** in a module-level list `SESSION_META`, one tuple per session giving `(animal, date, dataset_folder, ALM_probe_numbers)`. The list was transcribed from the authors' per-animal loading scripts (`/app/code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`), which is why the two `JEB24_2023-10-0[34]` files that exist on disk but carry no `clu` field are absent. Nothing is globbed. Each session is one `data_structure_<anm>_<date>.mat` plus a sibling `motionEnergy_<anm>_<date>.mat`. `load_mat_file` tries `h5py` first (MATLAB v7.3) and falls back to `scipy.io.loadmat` (v5); the returned format tag then routes the session to one of two parallel implementations, `process_session_h5` or `process_session_v5`, which duplicate the whole pipeline with format-specific indexing.

ii.
```python
SESSION_META = [
    # Ephys_Behavior sessions
    ('EKH1', '2021-08-07', 'Ephys_Behavior', [2]),
    ...
    ('JEB24', '2023-11-03', 'RandomizedDelay_Ephys_Behavior', [1]),
]

def load_mat_file(fpath):
    """Load .mat file, handling both HDF5 (v7.3) and v5 formats."""
    try:
        f = h5py.File(fpath, 'r')
        return f, 'h5'
    except:
        data = sio.loadmat(fpath, squeeze_me=False)
        return data, 'v5'
```
```python
    fdata, fmt = load_mat_file(data_fpath)
    if fmt == 'h5':
        result = process_session_h5(fdata, ...)
        fdata.close()
    else:
        result = process_session_v5(fdata, ...)
```

iii. From CONVERSION_NOTES.md Step 2: ".mat files in HDF5 v7.3 format (most) and v5 format (JEB23_2023-10-18, JEB23_2023-10-21, all JEB24 sessions)", so both readers are needed. Step 1 lists `load<ANM>_ALMVideo` as "Per-animal session metadata + ALM probe assignment", and Step 10 records "Probe indexing: Initially wrong for some sessions (JEB6 probe=2 not 1). Fixed using loading scripts" — i.e. the probe column of `SESSION_META` is taken from the authors' scripts rather than guessed.

## 1-b. How are the data split into subjects?

i. The animal is the first field of each `SESSION_META` tuple and is carried on each session's result dict as `anm`. At assembly, `subjects` is the sorted set of unique animals (14) and `subject_idx` is each session's index into that list, in the same order as `neural`/`input`/`output`.

ii.
```python
    all_animals = sorted(set(r['anm'] for r in all_results))
    subjects = all_animals
    subject_idx = np.array([subjects.index(r['anm']) for r in all_results])
```

iii. CONVERSION_NOTES.md Step 2 reports "Total animals | 14", split 10 (Ephys_Behavior) + 4 (RandomizedDelay). Step 4 flags that the paper says nine mice for the fixed-delay task and resolves it as "EKH1 may not be in paper's count but has data+loading script", i.e. the animal identity is taken from the filename/loading script rather than from any field inside the `.mat`.

## 1-c. How are the data split into sessions?

i. One entry of `SESSION_META` = one file on disk = one element of `neural`, `input`, `output`, `brain_region_idx`, and one row of `subject_idx`. The task folder is stored explicitly in the tuple (rather than searched for), so the 25 fixed-delay and 19 randomized-delay sessions are handled by the same code path and concatenated into one 44-session dataset. A session is dropped entirely if it yields no clusters, fewer than 2 usable trials, or fewer than 10 units after the firing-rate filter; in the full run none of the 44 was dropped.

ii.
```python
    data_fpath = os.path.join(DATA_DIR, dataset_dir, f'data_structure_{sess_id}.mat')
    me_fpath   = os.path.join(DATA_DIR, dataset_dir, f'motionEnergy_{sess_id}.mat')
```
```python
    if keep_mask.sum() < 10:
        print(f"  {sess_id}: Too few neurons after FR filter ({keep_mask.sum()})")
        return None
```

iii. CONVERSION_NOTES.md Step 3 records the paper's session-inclusion rule, "Session inclusion | ≥10 units | 'Recording sessions were included for analysis only if they had at least 10 units'", which is what the `keep_mask.sum() < 10` guard implements. Step 5 Key Decision 1: "**Include all sessions** from both datasets (44 total)".

## 1-d. How are the data split into trials?

i. A trial is one entry of the per-trial `obj.bp` fields and one entry of `bp.ev.goCue`. `Ntrials` is read once and used to size every per-trial array; the per-trial flags (`hit`, `miss`, `no`, `early`, `autowater`, `R`, `L`, `stim.enable`) and `goCue` are read with `.flatten()` and assumed to already be exactly `Ntrials` long (no truncation guard). Spikes carry their own 1-based `trial` index, and camera frames and motion-energy traces are stored per trial, so no trial boundaries need reconstructing.

ii.
```python
    ntrials = int(f['obj/bp/Ntrials'][0, 0])
    hit  = f['obj/bp/hit'][:].flatten().astype(bool)
    miss = f['obj/bp/miss'][:].flatten().astype(bool)
    no_resp = f['obj/bp/no'][:].flatten().astype(bool)
    early = f['obj/bp/early'][:].flatten().astype(bool)
    autowater = f['obj/bp/autowater'][:].flatten().astype(bool)
    R = f['obj/bp/R'][:].flatten().astype(bool)
    L = f['obj/bp/L'][:].flatten().astype(bool)
    stim_enable = f['obj/bp/stim/enable'][:].flatten().astype(bool)
    goCue = f['obj/bp/ev/goCue'][:].flatten()
```

iii. CONVERSION_NOTES.md Step 1 notes that the reference conditions "use bp fields: R, L, hit, miss, autowater, early, stim.enable", so the Bpod trial table is treated as the definition of a trial.

## 1-e. How are trials filtered based on quality controls?

i. **One filter only**: trials flagged `bp.early` (early lick) or `bp.stim.enable` (photostimulation) are dropped. Ignore (`no`) trials are deliberately **kept**, and become the third class of `lick_direction`/`outcome`. No other trial-level QC is applied — in particular, trials that occur after the electrophysiology recording has stopped are kept. This produces 13,823 trials. `train_decoder.py --verify-only` emitted 61 warnings, "Session 36, trial 292: all neural data is zero" … "Session 43, trial 333: all neural data is zero", for exactly those post-recording trials in `JEB24_2023-10-23` and `JEB24_2023-11-03`; they were left in the dataset.

ii.
```python
    # --- Trial selection ---
    # Exclude early lick and stim trials
    valid_trials_mask = ~early & ~stim_enable
    valid_trial_indices = np.where(valid_trials_mask)[0]  # 0-indexed

    if len(valid_trial_indices) < 2:
        print(f"  {sess_id}: Too few valid trials ({len(valid_trial_indices)})")
        return None
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 5: "**Trial filter**: Exclude early lick + stim trials. Keep ignore trials." Step 4 justifies keeping ignores against the paper: "early+ignore excluded from behavioral analysis | We exclude early+stim but keep ignore trials for decoder" — i.e. the ignore trials are needed so that the `none`/`ignore` output classes exist. For the zero-neural trials, Step 10 says only: "61 warnings about zero neural data in sessions 36 and 43 (JEB24 sessions where ephys ended before behavioral session). Expected and documented." No fix was attempted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters in `obj.clu{probe}` for the ALM probe(s) named in `SESSION_META`. Three fields per cluster are used: `quality` (manual curation label), `trialtm` (spike time relative to its trial's start, on the behaviour clock) and `trial` (1-based trial index of each spike). `bp.ev.goCue` is the second ingredient, since it defines the alignment. For two-probe sessions (the four JEB15 sessions) the clusters of both probes are concatenated into one population.

ii.
```python
    for probe_num in alm_probes:
        probe_idx = probe_num - 1  # 0-indexed for HDF5 (nProbes, 1)
        ref = clu_data[probe_idx, 0]
        probe = f[ref]
        quality_refs = probe['quality']
        for i in range(n_clusters):
            qref = quality_refs[i, 0]
            label = ''.join([chr(c) for c in f[qref][:].flatten()]).strip()
            if label in EXCLUDED_QUALITIES:
                continue
            trialtm = f[probe['trialtm'][i, 0]][:].flatten()
            trial   = f[probe['trial'][i, 0]][:].flatten().astype(int)  # 1-indexed
```

iii. CONVERSION_NOTES.md Step 1 records the reference pipeline `loadObjs → findTrials → findClusters → alignSpikes → getSeq → removeLowFRClusters`, and Step 2 that "clu is indexed as clu{probenum} in MATLAB → clu[probenum-1, 0] in HDF5". Probe numbers come from the `load<ANM>_ALMVideo.m` scripts (Step 10: "Probe indexing: Initially wrong for some sessions (JEB6 probe=2 not 1). Fixed using loading scripts").

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into the 10 ms bin grid, divided by the bin width to give spikes/s, and smoothed along time with the authors' **causal** Gaussian kernel: `gausswin(15)` (alpha = 2.5) with the first `floor(15/2) = 7` taps zeroed and the rest renormalised to sum to 1, convolved `'same'` after a reflect-padding of 15 samples at the start which is trimmed off afterwards. So each bin's rate depends only on the current and previous 7 bins (70 ms of past). No normalisation, z-scoring or baseline subtraction is applied; stored values are firing rates in Hz as `float32`, shape `(n_neurons, 500)` per trial. This was verified to reproduce exactly: recomputing EKH1_2021-08-07 from the raw `.mat` gives `np.allclose(...) == True` with max difference 0.0 against `converted_data.pkl`.

ii.
```python
def causal_gaussian_kernel(n):
    """Create causal Gaussian kernel matching MATLAB mySmooth."""
    alpha = 2.5  # MATLAB default
    half = (n - 1) / 2
    t = np.arange(n) - half
    kern = np.exp(-0.5 * (alpha * t / half) ** 2)
    kern[:n // 2] = 0          # Make causal: zero out first half
    return kern / kern.sum()

def smooth_with_bc(x, kernel, bctype='reflect'):
    n = len(kernel)
    prefix = x[1:n+1][::-1]
    x_ext = np.concatenate([prefix, x])
    smoothed = np.convolve(x_ext, kernel, mode='same')
    return smoothed[n:]  # trim prefix
```
```python
            counts, _ = np.histogram(spk_t, bins=time_edges)
            fr = counts.astype(np.float32) / DT
            trialdat[:, i, tr_idx] = smooth_with_bc(fr, kernel, BOUNDARY_CONDITION)
```

iii. CONVERSION_NOTES.md Step 1: "Smoothing: causal Gaussian kernel (gausswin, zero first half, normalize)" and Step 10 Check 5: "Smoothing: matches mySmooth.m (causal Gaussian, reflect BC)". The reference `utils/mySmooth.m` is `kern = gausswin(N); kern(1:floor(numel(kern)/2)) = 0; %causal; kern = kern./sum(kern);` with `params.smooth = 15`, `params.bctype = 'reflect'`, and `getSeq.m` does `mySmooth(N./params.dt, params.smooth, params.bctype)` — the Python is a line-for-line port.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters plus one session-level filter. (1) The manual curation label `clu.quality` is stripped and compared **case-sensitively** to `{'garbage', 'gabrga', 'noisy', 'real?'}`; everything else is kept, including multi-units and all `Poor`/`poor` units. (2) After binning and smoothing, any unit whose mean rate over all time bins and **all** trials (including the early-lick/photostim trials that were excluded above) is `<= 0.5 Hz` is dropped. (3) A session with fewer than 10 surviving units is dropped. This leaves **2,497 units**, 17–142 per session (mean 56.8).

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 0.5  # Hz, minimum mean firing rate
```
```python
            label = ''.join([chr(c) for c in qdata]).strip()
            if label in EXCLUDED_QUALITIES:
                continue
```
```python
    # --- Remove low FR clusters ---
    # Compute mean FR across all trials and time points (matching removeLowFRClusters)
    mean_fr = trialdat.mean(axis=(0, 2))  # mean over time and trials
    keep_mask = mean_fr > LOW_FR
    if keep_mask.sum() < 10:
        return None
```

iii. CONVERSION_NOTES.md Step 1: "Quality filter 'all': excludes garbage, gabrga, noisy, real? (case-sensitive in MATLAB)"; Step 5 Key Decision 3 repeats "case-sensitive matching MATLAB". Step 4 explicitly records the paper/code conflict on the rate threshold and resolves it in favour of the code: "| lowFR | 0.5 Hz | N/A | 1 Hz (specific analyses) | Use 0.5 Hz as in code |", expanded in Step 3 as "FR filter (code default) | >0.5 Hz | getDefaultParams.m: params.lowFR = 0.5" vs "FR filter (most analyses) | >1 Hz". Step 9 offers the resulting total as a consistency check: "Total neurons (post-filter) | ~2496 | 2497 | ✓".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is a single subtraction per spike: for each spike the go cue of its own trial is subtracted from `trialtm`. Both are already on the same behaviour clock and `trialtm` is already relative to trial start, so no offset or interpolation is needed. Spikes whose `trial` index falls outside `[1, Ntrials]` are set to NaN and dropped. The subtraction is done in a pure-Python loop over individual spikes.

ii.
```python
            # Align to goCue
            # For each spike, subtract the goCue time of its trial
            aligned_times = np.empty_like(trialtm)
            for t_idx in range(len(trialtm)):
                tr = trial[t_idx] - 1  # 0-indexed
                if 0 <= tr < ntrials:
                    aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
                else:
                    aligned_times[t_idx] = np.nan
```

iii. CONVERSION_NOTES.md Step 1 lists "alignSpikes | Align spike times to event (goCue)", and Step 10 Check 5: "Spike alignment: matches alignSpikes.m (subtract goCue from trialtm)". `ALIGN_EVENT = 'goCue'` is set at the top of the script, matching `params.alignEvent = 'goCue'` in `getDefaultParams.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins** (`DT = 1/100`) over a window of **−2.5 s to +2.5 s** from the go cue, giving **500 bins** per trial, identical for every trial and session. Bin centres run from −2.495 to +2.495 s. Every stream — neural, the time input, and the three camera outputs — lands on this one grid, so no rebinning or resampling between streams is needed after the fact. `metadata['time_bin_size'] = 10.0` ms, `off_start = -2.5`, `off_end = 2.5`.

ii.
```python
TMIN = -2.5  # seconds
TMAX = 2.5   # seconds
DT = 1.0 / 100  # 10ms bins
```
```python
    time_edges = np.arange(TMIN, TMAX + DT, DT)
    time_centers = time_edges[:-1] + DT / 2
```

iii. CONVERSION_NOTES.md Step 4 records that the reference code is internally inconsistent and documents the choice: "| dt | 1/200 (default) or 1/100 (tutorial) | N/A | Not specified | Use 1/100 (10ms) |", elaborated in Step 5 Key Decision 2 as "**Time bin**: 10ms matching WorkingWithDataObjs tutorial" and in Step 3 as "Time bin | 10ms (1/100) | WorkingWithDataObjs.m". The window comes from `params.tmin = -2.5` / `params.tmax = 2.5`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing in the raw data — it is the bin-centre time axis defined by the conversion itself, i.e. the centres of the same 500 bins that the spikes are counted into. Implicitly it is derived from `bp.ev.goCue`, since that is what sets the zero of the axis.

ii.
```python
    time_centers = time_edges[:-1] + DT / 2
...
    input_names = ['time_from_go_cue']
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: "| time from goCue | input[0] | Continuous time axis | Time in seconds |". The Decoder Task section of the instructions specifies exactly one input, "Time from go cue onset in seconds (continuous, time-varying)".

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The identical `(1, 500)` float32 array of bin centres is emitted for every trial of every session (it is re-created inside the per-trial loop rather than shared).

ii.
```python
    for i, tr_idx in enumerate(valid_trial_indices):
        # Input: time from go cue (continuous, time-varying)
        inp = time_centers.astype(np.float32).reshape(1, -1)
        input_trials.append(inp)
```

iii. No separate justification is given; the axis is defined by the conversion. Verified in `verification_full_out.txt`: "Input range: time_from_go_cue: [-2.5, 2.5]" for all 44 sessions.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `time_edges` is built once in `main()` and passed to `process_session`, which uses `time_edges` for `np.histogram` of the spikes and `time_centers` for the input, so bin *k* of `input` and bin *k* of `neural` are by construction the same 10 ms interval relative to that trial's go cue.

ii.
```python
    time_edges = np.arange(TMIN, TMAX + DT, DT)
    time_centers = time_edges[:-1] + DT / 2
    ...
    result = process_session(anm, date, dataset_dir, alm_probes, time_edges, kernel, ...)
```
```python
            counts, _ = np.histogram(spk_t, bins=time_edges)   # neural
...
        inp = time_centers.astype(np.float32).reshape(1, -1)   # input
```

iii. Implicit — one grid is constructed and shared. CONVERSION_NOTES.md Step 10 Check 3: "Input sanity check: Time axis spans [-2.5, 2.5]s as expected."

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: `R` and `L` (the **instructed/correct** side, not the animal's response) combined with `hit` and `miss` (whether the animal licked the instructed port or the other one). The animal's actual lick direction is not stored anywhere, so it is inferred from the pair.

ii.
```python
    hit  = f['obj/bp/hit'][:].flatten().astype(bool)
    miss = f['obj/bp/miss'][:].flatten().astype(bool)
    R = f['obj/bp/R'][:].flatten().astype(bool)
    L = f['obj/bp/L'][:].flatten().astype(bool)
```

iii. CONVERSION_NOTES.md Step 1: "R/L fields indicate CORRECT direction, not animal's actual lick direction / For hit trials: animal licked the correct direction / For miss trials: animal licked the WRONG direction / For ignore (no) trials: animal didn't lick". Step 10 records this as a bug that was found and fixed: "Lick direction encoding: Initially used R/L directly, but R/L indicate correct direction not actual lick. Fixed to encode actual lick based on hit/miss."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A three-way relabelling, initialised to `none` (2): on hit trials the class is the instructed side, on miss trials the opposite side, and everything else (the ignore trials) stays `none`. Codes are `left = 0`, `right = 1`, `none = 2`. The scalar is broadcast across all 500 bins so that all six outputs share one `(6, 500)` array per trial. Resulting distribution over the full dataset: left 0.422, right 0.445, none 0.133.

ii.
```python
    lick_dir = np.full(ntrials, 2, dtype=int)  # default: none (ignore)
    lick_dir[hit & R] = 1   # correct right → licked right
    lick_dir[hit & L] = 0   # correct left → licked left
    lick_dir[miss & R] = 0  # correct right, wrong → licked left
    lick_dir[miss & L] = 1  # correct left, wrong → licked right
```
```python
        out = np.zeros((6, n_time), dtype=np.int64)
        out[0, :] = lick_dir[tr_idx]  # broadcast
```
```python
        ['left', 'right', 'none'],       # lick_direction: 0=left, 1=right, 2=none
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "**Lick direction**: Encode ACTUAL lick direction (not correct direction)". The value ordering follows the instructions' "Lick direction (left, right, none, per-trial)".

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks the water-cued block in which water is delivered from a port without any instructing cue.

ii.
```python
    autowater = f['obj/bp/autowater'][:].flatten().astype(bool)
```

iii. CONVERSION_NOTES.md Step 1: "autowater=1 → WC (water-cued) block; autowater=0 → DR (delayed-response) block".

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct two-way relabelling of the flag — `WC = 0` where `autowater` is true, `DR = 1` elsewhere — broadcast across all 500 bins. Distribution over the full dataset: WC 0.097, DR 0.903; 12 of the 44 sessions are DR-only (their `context` range is `[1.0, 1.0]`).

ii.
```python
    context = np.zeros(ntrials, dtype=int)
    context[~autowater] = 1  # DR
    context[autowater] = 0   # WC
```
```python
        ['WC', 'DR'],                     # context: 0=WC, 1=DR
```

iii. Codes follow the instructions' "Behavioral context (WC, DR, per-trial)". CONVERSION_NOTES.md Step 9 checks the resulting marginal against the paper: "| context dist | mostly DR | 9.7%WC/90.3%DR | ✓ |", and Step 12 notes DR-only sessions are expected.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial flags, `bp.hit` and `bp.miss`. `bp.no` is read into `no_resp` but never used — a trial that is neither a hit nor a miss is treated as an ignore by construction.

ii.
```python
    hit  = f['obj/bp/hit'][:].flatten().astype(bool)
    miss = f['obj/bp/miss'][:].flatten().astype(bool)
    no_resp = f['obj/bp/no'][:].flatten().astype(bool)   # loaded but never referenced
```

iii. CONVERSION_NOTES.md Step 5 mapping: "| bp.hit/miss/no | output[2]: outcome | 0=incorrect, 1=correct, 2=ignore | Per trial |".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling initialised to `ignore` (2): hits become `correct` (1), misses become `incorrect` (0), everything else stays `ignore`. Broadcast across all 500 bins. Distribution over the full dataset: incorrect 0.120, correct 0.747, ignore 0.133 — the ignore fraction equals the `none` fraction of `lick_direction`, as it must.

ii.
```python
    outcome = np.full(ntrials, 2, dtype=int)  # default: ignore
    outcome[hit] = 1  # correct
    outcome[miss] = 0  # incorrect
```
```python
        ['incorrect', 'correct', 'ignore'],  # outcome: 0=incorrect, 1=correct, 2=ignore
```

iii. Ordering follows the instructions' "Outcome (incorrect, correct, ignore, per-trial)". CONVERSION_NOTES.md Step 9: "| outcome dist | mostly correct | 12%/74.7%/13.3% | ✓ |". Keeping ignore as a class rather than dropping the trials is the Step 4 decision quoted in 1-e.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, **side camera only** (`traj{1}`), feature named `tongue`; its `ts` array supplies x, y and likelihood per frame and `frameTimes` supplies the frame clock. The bottom camera's `top_tongue` is not used. `bp.ev.goCue`, `bp.ev.bitStart`, `sglx.bitcode.bitstart` and `sglx.fs` are also read, to put frames on the go-cue clock.

ii.
```python
        # Side cam (index 0) for tongue
        side_ref = traj[0, 0]
        side_cam = f[side_ref]
        ...
        tongue_idx = None
        for j, name in enumerate(side_feats):
            if name == 'tongue':
                tongue_idx = j
                break
        ...
                    ts = f[ts_ref][:]  # (nFeats, 3, nFrames)
                    tongue_x    = ts[tongue_idx, 0, :]
                    tongue_y    = ts[tongue_idx, 1, :]
                    tongue_conf = ts[tongue_idx, 2, :]
```

iii. CONVERSION_NOTES.md Step 2 records the camera layout, "traj{1}=side cam (7 features), traj{2}=bottom cam (10 features)", and the agent enumerated both feature lists (side: `tongue, left_tongue, right_tongue, jaw, trident, nose, lickport`; bottom: `top_tongue, …, top_paw, bottom_paw, …`). No explicit reason is given anywhere for using the side view alone; Step 5 says only "| traj tongue | output[3]: tongue_velocity | Discretized 0/1/2 | 50th percentile per session |".

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps. (1) Frame times are put on the go-cue clock (see 7-d). (2) Speed is the magnitude of the first difference of the **raw, unsmoothed** x and y traces divided by the *median* frame interval — `sqrt(dx² + dy²)/dt` — with the last sample duplicated to restore the original length; no per-run handling, and no Gaussian smoothing of the position. (3) Frames whose likelihood is `< 0.9` are set to NaN. (4) The remaining frames are resampled onto the 500 bin centres with `np.interp`, i.e. **point-sampled by linear interpolation**, not averaged within bins, with NaN only outside the first/last surviving frame. A trial with 2 or fewer surviving frames is left entirely NaN.

ii.
```python
def compute_velocity_from_xy(x, y, frame_times, dt_frames):
    """Compute velocity magnitude from x,y coordinates."""
    dx = np.diff(x) / dt_frames
    dy = np.diff(y) / dt_frames
    vel = np.sqrt(dx**2 + dy**2)
    vel = np.concatenate([vel, [vel[-1] if len(vel) > 0 else 0]])
    return vel
```
```python
                    dt_frames = np.median(np.diff(aligned_ft))
                    if dt_frames > 0:
                        vel = compute_velocity_from_xy(tongue_x, tongue_y, aligned_ft, dt_frames)
                        # Set low confidence to NaN
                        vel[tongue_conf < 0.9] = np.nan
                        # Interpolate to time_centers
                        valid = ~np.isnan(vel)
                        if valid.sum() > 2:
                            tongue_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], vel[valid],
                                                              left=np.nan, right=np.nan)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 7: "**DLC velocity**: From x,y coordinates, threshold at 50th percentile, confidence < 0.9 = not visible." The likelihood cut of 0.9 is the value at which the authors already NaN out the coordinates. No reference MATLAB exists for this quantity — `loadKinData.m` merely loads pre-computed `kin_*.mat` files that are not in `/app/data` — so the velocity definition was invented.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single per-session threshold: the **median of all non-NaN tongue-velocity values pooled over every bin of every retained trial** of that session. Bins at or above it get 1, bins below get 0, and bins that are still NaN get 2 (`not_visible`). Full-dataset distribution: 0.166 / 0.166 / 0.667.

Because step (4) of 7-b interpolated *across* the gaps where the tongue is untracked, the `not_visible` class is only assigned outside the first/last tracked frame of a trial, not wherever the tongue is actually out of view. Checking EKH1_2021-08-07 against the raw file: in its first five retained trials, 6 / 97 / 64 / 115 / 93 of the bins labelled 0 or 1 contain **no** frame with likelihood ≥ 0.9 — about half of all non-`not_visible` bins carry an interpolated, fabricated velocity.

ii.
```python
def discretize_velocity(vel_all, valid_trial_indices, not_visible_val=2):
    """0: < 50th percentile, 1: >= 50th percentile, 2: not visible / no data"""
    valid_data = vel_all[:, valid_trial_indices]
    flat_valid = valid_data[~np.isnan(valid_data)]
    if len(flat_valid) == 0:
        return [np.full(n_time, not_visible_val, dtype=np.int64) for _ in valid_trial_indices]
    threshold = np.median(flat_valid)
    for tr_idx in valid_trial_indices:
        trial_vel = vel_all[:, tr_idx]
        disc = np.full(n_time, not_visible_val, dtype=np.int64)
        valid_mask = ~np.isnan(trial_vel)
        disc[valid_mask & (trial_vel <  threshold)] = 0
        disc[valid_mask & (trial_vel >= threshold)] = 1
        result.append(disc)
```
```python
        ['below_median', 'above_median', 'not_visible'],  # tongue_velocity
```

iii. The three-way split is specified verbatim by the Decoder Task section ("0: < 50th percentile, 1: >= 50th percentile, 2: not visible"), and the docstring quotes it. CONVERSION_NOTES.md Step 7 comments that "tongue_velocity: mostly not_visible (68%) - expected since tongue is only visible during licking". The interpolation-across-gaps consequence is not discussed anywhere.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Via the authors' bitcode-based video offset, computed once per camera stream per session: `vidshift = median(sglx.bitcode.bitstart)/sglx.fs − median(bp.ev.bitStart)`. Frame times become `frameTimes − vidshift − goCue[trial]`, and are then resampled onto the same 500 bin centres as the spikes. If `sglx` is missing the code falls back to a hard-coded `VIDEO_OFFSET_DEFAULT = 0.5` s. Checked against the raw file for EKH1_2021-08-07: `vidshift = 0.49004` s and the corrected frames of trial 0 start at −2.475 s from the go cue, i.e. just inside the window.

ii.
```python
VIDEO_OFFSET_DEFAULT = 0.5  # seconds
...
        try:
            bitStart = np.nanmedian(f['obj/bp/ev/bitStart'][:].flatten())
            sglx_bitstart = np.nanmedian(f['obj/sglx/bitcode/bitstart'][:].flatten())
            sglx_fs = f['obj/sglx/fs'][0, 0]
            vidshift = sglx_bitstart / sglx_fs - bitStart
        except:
            vidshift = VIDEO_OFFSET_DEFAULT
        ...
                    aligned_ft = frame_times - vidshift - goCue[tr_idx]
```

iii. CONVERSION_NOTES.md Step 1: "Video offset: mode(sglx.bitcode.bitstart)/sglx.fs - mode(bp.ev.bitStart)" and Step 6: "Video offset computed from sglx.bitcode.bitstart/fs - bp.ev.bitStart". This is `funcs/findVideoOffset.m`, except that the reference takes the `mode` of each quantity and the script takes the `nanmedian`. The 0.5 s fallback mirrors the reference's own catch branch in `loadMotionEnergy.m` (`frameTimes - 0.5 - alignTimes`).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom camera (`traj{2}`) only, and **every** feature whose name contains "paw" — i.e. both `top_paw` and `bottom_paw`. The same `ts` (x, y, likelihood) and `frameTimes` fields are used.

ii.
```python
        # Find paw feature indices (bottom cam)
        paw_indices = []
        for j, name in enumerate(bottom_feats):
            if 'paw' in name.lower():
                paw_indices.append(j)
```

iii. No explicit justification. CONVERSION_NOTES.md Step 5 says only "| traj paw | output[4]: paw_velocity | Discretized 0/1/2 | 50th percentile per session |"; Step 2 records that the bottom camera carries `top_paw` and `bottom_paw`. The bottom camera is the only view in which the paws are tracked at all.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Exactly the tongue pipeline (`compute_velocity_from_xy`, likelihood `< 0.9 → NaN`, `np.interp` onto the 500 bin centres), applied to each paw feature separately, with the two paws then combined by `np.nanmean` **before** resampling — so a bin is missing only if *both* paws are untracked. No position smoothing and no per-camera normalisation (there is only one camera).

ii.
```python
                        paw_vels = []
                        for pidx in paw_indices:
                            px = ts[pidx, 0, :]; py = ts[pidx, 1, :]; pc = ts[pidx, 2, :]
                            vel = compute_velocity_from_xy(px, py, aligned_ft, dt_frames)
                            vel[pc < 0.9] = np.nan
                            paw_vels.append(vel)

                        avg_vel = np.nanmean(paw_vels, axis=0)
                        valid = ~np.isnan(avg_vel)
                        if valid.sum() > 2:
                            paw_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], avg_vel[valid],
                                                           left=np.nan, right=np.nan)
```

iii. Same justification as 7-b — CONVERSION_NOTES.md Step 5 Key Decision 7 covers both streams with one rule ("DLC velocity: From x,y coordinates, threshold at 50th percentile, confidence < 0.9 = not visible"). Averaging the two paws is not discussed.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same `discretize_velocity` call: per-session median of all non-NaN paw velocities pooled over retained trials, `< median → 0`, `>= median → 1`, NaN → 2 (`not_visible`). Full-dataset distribution: 0.478 / 0.478 / 0.043.

The `not_visible` fraction is very low (4.3%) for two compounding reasons: `np.nanmean` over two paws means only bins where *both* paws are untracked are NaN, and `np.interp` then bridges the remaining interior gaps, so class 2 survives only outside the first/last tracked frame of a trial.

ii.
```python
    paw_disc = discretize_velocity(paw_vel_all, valid_trial_indices, not_visible_val=2)
```
```python
        ['below_median', 'above_median', 'not_visible'],  # paw_velocity
```

iii. The split is the one specified by the Decoder Task. CONVERSION_NOTES.md Step 7 notes "paw_velocity: roughly equal below/above (47%/47%), 6% not visible" and treats that as expected.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, but using the **bottom** camera's own `frameTimes`: `frameTimes − vidshift − goCue[trial]`, then `np.interp` onto the shared 500-bin grid. `vidshift` is recomputed from the same bitcode expression inside the same function.

ii.
```python
                    ft_ref = bottom_cam['frameTimes'][tr_idx, 0]
                    frame_times = f[ft_ref][:].flatten()
                    aligned_ft = frame_times - vidshift - goCue[tr_idx]
                    dt_frames = np.median(np.diff(aligned_ft))
```

iii. Same as 7-d — one session-wide offset from `findVideoOffset.m` applied to whichever camera the feature belongs to.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` sitting beside each data structure, always read with `scipy.io.loadmat`; the `obj.me` copy inside some data structures is not used. The file's `me` struct is unwrapped one level (`me['data']`), or used directly if it is a bare object array. The side camera's `frameTimes` (plus `vidshift` and `goCue`) provide the time base, with a `arange(n)/400.0` fallback if `frameTimes` is unavailable.

ii.
```python
        me_data = sio.loadmat(me_fpath, squeeze_me=False)
        me_raw = me_data['me']

        if hasattr(me_raw, 'dtype') and me_raw.dtype.names and 'data' in me_raw.dtype.names:
            # Standard format: me is a struct with 'data' field
            me_trials = me_raw['data'][0, 0]
        elif me_raw.dtype == object:
            # Alternative format: me IS the data array directly (nTrials, 1)
            me_trials = me_raw
        else:
            print(f"  {sess_id}: Unknown ME format")
            return me_all, False
```

iii. CONVERSION_NOTES.md Step 6: "Motion energy: handles both struct format (me.data) and direct array format", and Step 10 lists "ME loading for JEB23: Different file format (me is data directly, not struct). Fixed." The reference `loadMotionEnergy.m` was read and also contains a third case, `if isstruct(me.data), me.data = me.data.data; end`, which was **not** ported: in `JEB15_2022-07-26`, `JEB15_2022-07-28` and `JEB24_2023-10-31` the payload is nested one level deeper, `me['data'][0,0]` is itself a `('data','moveThresh')` struct, the per-trial access raises, and the bare `except: continue` leaves the whole session as NaN. Those 3 of 44 sessions come out as 100% `no_video`, which is the bulk of the dataset-wide 7.2%. The agent found this at the end of the run and decided against fixing it ("this is a minor issue affecting only 3 out of 44 sessions, and the decoder already achieves good accuracy"); it is not mentioned in CONVERSION_NOTES.md.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling — the file already holds one scalar per camera frame. Each trial's trace is linearly interpolated onto the 500 bin centres with `np.interp` (no `left`/`right`, so values outside the frame range are clamped to the edge value), and any remaining NaNs in a trial are then filled by interpolating over the NaN indices, i.e. a gap-fill. This is a direct port of the reference `loadMotionEnergy.m` (`interp1(...)` followed by `fillmissing(me.data,'nearest')`).

ii.
```python
                aligned_ft = frame_times - vidshift - goCue[tr_idx]
                valid = ~np.isnan(me_trial) & ~np.isnan(aligned_ft)
                if valid.sum() > 2:
                    me_all[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], me_trial[valid])

        # Fill NaNs with nearest
        for tr_idx in range(ntrials):
            col = me_all[:, tr_idx]
            nans = np.isnan(col)
            if nans.any() and not nans.all():
                col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), col[~nans])
                me_all[:, tr_idx] = col
```

iii. CONVERSION_NOTES.md Step 1: "Motion energy aligned via interpolation to neural time axis". The reference MATLAB is:
`me.newdata(:,trix) = interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis);` … `me.data = fillmissing(me.data,'nearest');` — the Python mirrors it step for step.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize_velocity` helper: per-session median over all non-NaN motion-energy values of the retained trials, `< median → 0`, `>= median → 1`, NaN → 2 (`no_video`). Full-dataset distribution: 0.464 / 0.464 / 0.072, and the 0.072 is almost entirely the three sessions whose file never loaded (9-a) — within a working session the nearest-fill means class 2 essentially never occurs.

ii.
```python
    me_disc = discretize_velocity(me_all, valid_trial_indices, not_visible_val=2)
```
```python
        ['below_median', 'above_median', 'no_video'],     # motion_energy
```

iii. The 50/50 split is specified by the Decoder Task; the third value is named `no_video` there and here. CONVERSION_NOTES.md Step 7: "motion_energy: exactly 50/50 (expected from median split)".

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the tracking, using the **side** camera's `frameTimes` (motion energy has one value per side-camera frame): `frameTimes − vidshift − goCue[trial]`, then `np.interp` onto the shared 500 bin centres. `vidshift` is recomputed a second time inside `load_motion_energy` from the same bitcode expression.

ii.
```python
            # Get frame times from traj
            traj = f['obj/traj']
            side_ref = traj[0, 0]
            side_cam = f[side_ref]
        ...
                        ft_ref = side_cam['frameTimes'][tr_idx, 0]
                        frame_times = f[ft_ref][:].flatten()
                    except:
                        frame_times = np.arange(len(me_trial)) / 400.0
                ...
                aligned_ft = frame_times - vidshift - goCue[tr_idx]
```

iii. Matches the reference exactly, which also indexes `obj.traj{1}` (side camera) for the motion-energy time base and also falls back to a synthetic 400 Hz axis with a 0.5 s shift when `frameTimes` is unusable.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Largely by **broad, silent exception handling**. Every per-trial camera/motion-energy block, and the whole DLC and ME loaders, are wrapped in bare `try/except … continue` or `except Exception`, so any per-trial or per-session failure leaves that entry NaN and is never reported. Specific cases: (a) missing/NaN `frameTimes` or fewer than 3 tracked frames → the trial is left NaN and becomes `not_visible`; (b) untracked frames (likelihood < 0.9) → NaN, but only the *leading and trailing* runs survive as `not_visible`, the interior being interpolated over (7-c, 8-c); (c) missing `sglx` → `vidshift` falls back to 0.5 s; (d) a missing motion-energy file → whole session `no_video`, and the unhandled doubly-nested `me.data.data` layout does the same for 3 sessions without any message (9-a); (e) trials after the probe stopped recording are kept with all-zero firing rates (1-e); (f) sessions with no clusters, <2 trials or <10 units return `None` and are dropped. Nothing is imputed for the neural or behavioural streams. `warnings.filterwarnings('ignore')` is set at import, which also hides the `nanmean`-of-all-NaN warnings.

ii.
```python
import warnings
warnings.filterwarnings('ignore')
```
```python
            except Exception as e:
                continue
    except Exception as e:
        print(f"  {sess_id}: DLC loading error: {e}")
        has_video = False
```
```python
            except:
                continue
```

iii. CONVERSION_NOTES.md Step 10 explains the retained zero-neural trials as "Expected and documented", and Step 6 describes the format-variant handling. The silent motion-energy failure is acknowledged only in the agent's own reasoning ("this is a minor issue affecting only 3 out of 44 sessions… won't significantly affect results") and does not appear in CONVERSION_NOTES.md, whose Step 12 states "No additional issues found in this review."

## 11-a. What are the most time-consuming steps of the code?

i. The full conversion takes **132 s** for 44 sessions (2–5 s per session), which is well inside the 15-minute budget, so no bottleneck was ever chased. The dominant costs are, in order: (1) reading the `.mat` files — for the v5 sessions `scipy.io.loadmat` reads the entire structure into memory up front; (2) the spike pipeline, which is a triple loop over clusters × all trials performing one `np.histogram` and one 500-sample `np.convolve` per pair, over *all* clusters (before the firing-rate filter) and *all* trials (including the ones about to be discarded); (3) the pure-Python per-spike loop that subtracts the go cue, which runs once per spike of every cluster; (4) the per-trial DLC loop, which does a fresh HDF5 dereference and read of `ts` (a `(nFeats, 3, nFrames)` array) once for the tongue and again for the paw. Only per-session wall time is instrumented — there is no per-step timing.

ii.
```python
    t0 = time.time()
    ...
    t1 = time.time()
    print(f"  {sess_id}: {n_neurons} neurons, {n_trials} trials ({t1-t0:.1f}s)")
```
```python
    for i in range(n_neurons_raw):
        for tr_idx in range(ntrials):
            ...
            counts, _ = np.histogram(spk_t, bins=time_edges)
            fr = counts.astype(np.float32) / DT
            trialdat[:, i, tr_idx] = smooth_with_bc(fr, kernel, BOUNDARY_CONDITION)
```

iii. CONVERSION_NOTES.md Step 7: "Processing time: 6.4s for 2 sessions → ~3.2s/session / Estimated full time: ~140s for 44 sessions ✓ (actual: 132s)". The Step 6 placeholders "Code inefficiencies identified" and "Code speedups added" were removed rather than filled in, and the Step 7 "Run Time Estimates" tables were left out.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Four, none of which were vectorised. (1) The per-spike go-cue subtraction, `for t_idx in range(len(trialtm))`, is a one-line numpy expression (`trialtm - goCue[trial - 1]`) and is the most clearly avoidable — it runs in Python once per spike. (2) The cluster × trial spike-binning loop could be a single `np.histogram2d` over (trial, aligned time) per cluster, as the reference conversion does. (3) `smooth_with_bc` is called once per (cluster, trial) on a 500-sample vector; it could convolve the whole `(n_time, n_trials)` block at once, and its internal `for j in range(x_ext.shape[1])` column loop is also scalar. (4) The DLC per-trial loop genuinely has to stay a loop (frame counts differ per trial), but it re-reads and re-slices `ts` per feature rather than once. There is also a redundant per-trial loop building `input_trials`, which appends the same array 13,823 times.

ii.
```python
            aligned_times = np.empty_like(trialtm)
            for t_idx in range(len(trialtm)):
                tr = trial[t_idx] - 1  # 0-indexed
                if 0 <= tr < ntrials:
                    aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
                else:
                    aligned_times[t_idx] = np.nan
```
```python
            out = np.zeros_like(x_ext)
            for j in range(x_ext.shape[1]):
                out[:, j] = np.convolve(x_ext[:, j], kernel, mode='same')
```

iii. No justification is recorded — CONVERSION_NOTES.md Step 6's "Code inefficiencies identified / Code speedups added" template entries were dropped from the final file. The implicit rationale is that 132 s was already acceptable, so optimisation was never revisited.

## 11-c. What processing does the code repeat multiple times?

i. Several things. (1) The video offset is computed **twice per session** from the same four raw fields — once in `load_dlc_velocities_*` and again in `load_motion_energy` — even though it is a session constant. (2) The side camera's `frameTimes` are read once per trial for the tongue and again once per trial for the motion energy. (3) `np.median(np.diff(aligned_ft))` is recomputed per trial per camera. (4) The identical `time_centers` input array is materialised once per trial, 13,823 times. (5) Smoothed firing rates are computed for every cluster before the firing-rate filter throws roughly half of them away, and for every trial before the early-lick/photostim mask throws ~9% of them away. (6) The entire session pipeline is written twice, once for HDF5 and once for v5, so every processing rule exists in two copies that must be kept in sync — and in fact one bug (the lick-direction encoding) had to be fixed in both places separately.

ii.
```python
        try:
            bitStart = np.nanmedian(f['obj/bp/ev/bitStart'][:].flatten())
            sglx_bitstart = np.nanmedian(f['obj/sglx/bitcode/bitstart'][:].flatten())
            sglx_fs = f['obj/sglx/fs'][0, 0]
            vidshift = sglx_bitstart / sglx_fs - bitStart
        except:
            vidshift = VIDEO_OFFSET_DEFAULT
```
(the identical block appears in `load_dlc_velocities_h5`, `load_dlc_velocities_v5` and `load_motion_energy`)

iii. Not discussed. The duplication of the h5/v5 paths is implicitly justified by Step 6, "Handles both HDF5 (h5py) and v5 (scipy.io) .mat formats… DLC: handles v5 traj shape (1, nCams) vs h5 shape (nCams, 1)"; the trajectory shows two separate fix rounds (steps 89 and 109) needed because the v5 branch diverged from the h5 branch.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Firing rates are computed and smoothed for **all** clusters and **all** trials, then subset — roughly half the clusters fail the 0.5 Hz filter and ~9% of trials are early-lick/photostim, so a large fraction of the most expensive computation is thrown away. (2) The six outputs are stored as `int64` although they only ever take the values 0–2; that is 8× the necessary width and is the reason `converted_data.pkl` is 1.87 GB. (3) `no_resp` (`bp.no`), `L` in the v5 branch's `context`/`outcome` paths, and the `has_video` / `has_me` return flags are computed and never used. (4) Several helpers are dead code: `smooth_signal`, `reflect_boundary`, `safe_scalar`'s h5 path, `read_h5_string`, `get_bp_field_h5/v5`, `get_event_times_h5/v5`, and the unused `from scipy.signal import convolve`. (5) The `--show-processing` flag is parsed and threaded through `process_session` into both `process_session_*` functions but is **never read**, so no `processing_<session_id>.png` plots are produced at all. (6) `me_all` is allocated and nearest-filled for all `ntrials` even though only the retained trials are used.

ii.
```python
def process_session_h5(f, ..., me_fpath, show_processing):   # show_processing never referenced
```
```python
        out = np.zeros((6, n_time), dtype=np.int64)
```
```python
    no_resp = f['obj/bp/no'][:].flatten().astype(bool)   # never used
```

iii. No justification is offered. CONVERSION_NOTES.md Step 10 records the `int64` choice as a deliberate fix — "Output dtype: Changed from float32 to int64 for categorical outputs" — i.e. it was chosen to make the outputs integral, not for size. The unimplemented `--show-processing` is not mentioned anywhere; CONVERSION_NOTES.md Step 7 claims "Processing Plots Review" was done.
