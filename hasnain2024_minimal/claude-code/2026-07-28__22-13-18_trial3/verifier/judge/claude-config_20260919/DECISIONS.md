# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. One MATLAB file per session, `data_structure_<anm>_<date>.mat`, living in one of the two task folders (`Ephys_Behavior`, `RandomizedDelay_Ephys_Behavior`), with the per-frame motion energy in a sibling `motionEnergy_<anm>_<date>.mat`. The sessions are not globbed: a hard-coded `EPHYS_SESSIONS` list of 45 `(animal, date, probes, folder)` tuples drives the loop, and `main()` first filters that list down to entries whose file actually exists on disk. Each session file is opened once by a `SessionData` wrapper class that hides the two MATLAB container formats behind one accessor API: v7.3 files are read lazily with `h5py` (dereferencing object references on demand) and v5 files are read eagerly with `scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False)`. Format is detected by sniffing `'7.3'` in the 30-byte file header. Motion energy is always read with `scipy.io.loadmat` in a separate helper that unwraps the three layouts found in the data (bare cell array, `{data, moveThresh}`, and nested `{data: {data, moveThresh}}`). Nothing is accumulated across sessions except the finished per-trial arrays; `sd.close()` releases each HDF5 handle.

ii.
```python
class SessionData:
    """Unified interface for loading session data from .mat files."""

    def __init__(self, fpath):
        self.fpath = fpath
        self.is_h5 = self._check_h5(fpath)
        if self.is_h5:
            self.f = h5py.File(fpath, 'r')
            self.obj = self.f['obj']
        else:
            mat = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
            self.obj_v5 = mat['obj']
            self.f = None

    def _check_h5(self, fpath):
        with open(fpath, 'rb') as ff:
            header = ff.read(30).decode('ascii', errors='replace')
        return '7.3' in header
```

```python
for anm, date, probes, data_dir in EPHYS_SESSIONS:
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(DATA_ROOT, data_dir, fn)
    if os.path.exists(fpath):
        available.append((anm, date, probes, data_dir))
    else:
        print(f'  Skipping {anm}_{date}: file not found')
...
for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
```

```python
def load_motion_energy(data_dir, anm, date):
    me_mat = sio.loadmat(me_path, squeeze_me=False, struct_as_record=True)
    me_var = me_mat['me']
    if me_var.dtype.names is not None and 'data' in me_var.dtype.names:
        me_struct = me_var[0, 0] if me_var.ndim >= 2 else me_var[0]
        thresh = float(me_struct['moveThresh'].flatten()[0])
        data_field = me_struct['data']
        if data_field.dtype.names is not None and 'data' in data_field.dtype.names:
            data_field = data_field[0, 0]['data']       # nested data.data
        ...
    if me_var.dtype == object:                          # bare cell array
        ...
```

iii. From the trajectory: the AI first assumed all files were v7.3, then discovered (steps 82–92) that "some are v5.0 (loadable with scipy) and some are v7.3 (HDF5). I need to handle both formats," and rewrote the script around "a unified data access layer" after empirically establishing that `ts` is `(n_features, 3, n_frames)` when read through h5py and `(n_frames, 3, n_features)` when read through scipy. The motion-energy unwrapping was likewise driven by trial and error over several sessions (steps 66–120) and the AI explicitly cites the reference: "The reference code handles this: `if isstruct(me.data); me.data = me.data.data; end`."

## 1-b. How are the data split into subjects (mice)?

i. The animal is the first element of each `EPHYS_SESSIONS` tuple (equivalently, the `<anm>` part of the filename) and is carried on each session result as `'anm'`. `subjects` is built as the list of distinct animals in order of first appearance, and `subject_idx` is each session's position in that list. The result is 14 subjects over 45 sessions.

ii.
```python
for anm, date, probes, data_dir in available:
    result = process_session(...)
    if result is not None:
        all_sessions.append(result)
        if anm not in subjects_set:
            subjects_set.append(anm)
...
subjects = subjects_set
subject_idx = np.array([subjects.index(s['anm']) for s in all_sessions])
```

iii. No explicit justification is recorded. The AI took the animal id from the filename/loading-script naming rather than from `obj.meta`, consistent with how the authors' `load<ANM>_ALMVideo.m` scripts identify animals.

## 1-c. How are the data split into sessions?

i. One session = one `(animal, date)` entry of the hard-coded list = one `data_structure` file = one element of `neural`/`input`/`output`. The folder is stored in the tuple, so fixed-delay and randomized-delay sessions are treated uniformly. The list was seeded from the authors' `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` files (probe numbers per session come from there), but it is not identical to them: it contains 45 sessions, one more than the 44 the loading scripts specify, because `JEB23_2023-10-20` — which the AI's own sub-agent survey of those scripts reported as absent (it listed only 7 JEB23 sessions) — was included anyway with probe 1. Two behaviour-only files, `JEB24_2023-10-03/04`, were removed after they failed to load for lack of a `clu` field. Sessions are additionally dropped at runtime if the file is missing, if `clu` is missing, if fewer than 5 trials survive filtering, or if fewer than 10 units survive the firing-rate cut; none of these runtime guards actually fired.

ii.
```python
# Session definitions: (animal, date, probes, data_dir)
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
    ('JEB23', '2023-10-19', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-20', [1], 'RandomizedDelay_Ephys_Behavior'),   # not in load<ANM>_ALMVideo.m
    ('JEB23', '2023-10-21', [1], 'RandomizedDelay_Ephys_Behavior'),
    ...
]
```

```python
    if keep_units.sum() < 10:
        print(f'  WARNING: Only {keep_units.sum()} units with FR>{LOW_FR_THRESH} Hz, skipping')
        sd.close()
        return None
```

iii. The AI dispatched an Explore sub-agent (step 32) to read every `load<ANM>_ALMVideo.m` and return "the animal name, the session dates, the probe number(s) used," and the probe assignments in the code match that survey exactly. For the two dropped JEB24 sessions it reasoned: "This session has no `clu` field – it's a behavior-only session. JEB24_2023-10-03 and JEB24_2023-10-04 are not listed in the loading scripts. Let me remove them." No justification is recorded anywhere for keeping `JEB23_2023-10-20`, which is equally absent from those scripts; the directory listing is the only place it appears.

## 1-d. How are the data split into trials?

i. Trials are the rows of the Bpod table: `bp.Ntrials` gives the count and every per-trial field (`hit`, `miss`, `R`, `L`, `early`, `autowater`, `stim.enable`, `ev.goCue`) is read as a flat array indexed by trial. Nothing has to be reconstructed: spikes carry a 1-based `clu.trial` label that is matched against `j+1`, and `traj{view}` and the motion-energy cell array are both indexed directly by the 0-based trial number. Each surviving trial becomes one `(n_neurons, 500)` matrix, one `(1, 500)` input, and one `(6, 500)` output.

ii.
```python
    def get_ntrials(self):
        if self.is_h5:
            return int(np.array(self.obj['bp']['Ntrials']).flatten()[0])
        return int(self.obj_v5.bp.Ntrials)

    def get_trial_array(self, field):
        """Get a boolean/numeric trial array from bp (hit, miss, no, R, L, early, autowater)."""
        if self.is_h5:
            return np.array(self.obj['bp'][field]).flatten()
        return np.array(getattr(self.obj_v5.bp, field)).flatten()
```

```python
    for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
        for j in range(ntrials):
            trial_mask = (trial_nums == (j + 1))
```

iii. No explicit justification; the AI verified the field layout empirically (step 43: "hit: 276, miss: 37, no: 69, early: 25, R: 188, L: 194, autowater: 122, stim.enable: 58, Ntrials: 382") before writing the loop. Note that `get_trial_array` does not truncate to `Ntrials`, unlike the reference solution, which does.

## 1-e. How are trials filtered based on quality controls?

i. A single mask, applied before anything is computed: a trial is kept if it is a hit **or** a miss, and is neither photostimulated (`bp.stim.enable`) nor an early lick (`bp.early`). The hit-or-miss requirement means all *ignore* (no-response, `bp.no`) trials are discarded — the AI's notes list this explicitly as "Exclude: ignore/no-response trials (`no == 1`)". No other trial-level filter is applied; in particular there is no check that a trial falls inside the span of the ephys recording, so trials recorded after the probe stopped survive and enter the dataset as 500 bins of zero firing across every unit (the format verifier reported 30 such trials in sessions 37 and 44). 12,293 of 15,155 trials survive.

ii.
```python
    # Trial selection: hit or miss, no stim, no early
    valid_trials = (hit | miss) & ~stim_enable & ~early
    valid_idx = np.where(valid_trials)[0]

    if len(valid_idx) < 5:
        print(f'  WARNING: Too few valid trials ({len(valid_idx)}), skipping')
```

iii. The AI justified the mask as "Matching reference code conditions" — the authors' `params.condition` strings in `WorkingWithDataObjs.m`, which it read at step 16, all take the form `'R&hit&~stim.enable&~autowater&~early'`, i.e. they select hits and misses and never ignore trials. The AI did not weigh this against the decoder specification in the instructions, which asks for `none` and `ignore` output classes that only exist on the trials it discarded. The all-zero-neural trials were noticed after the fact ("Session 37 (JEB24_2023-10-23) has 19 zero-neural trials at the end... The zero neural data trials likely happen because those trials had no spikes from any cluster within the -2.5 to 2.5s window. Let me check and handle this") but no handling was added.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the probe(s) listed for that session, reduced to three fields per cluster: `quality` (the manual curation string), `trial` (the 1-based trial of each spike), and `trialtm` (each spike's time relative to its trial's start). `bp.ev.goCue` is the second input, since it defines the alignment. For two-probe sessions the clusters of both probes are concatenated into one population.

ii.
```python
    def get_clusters(self, probe_nums):
        """Get list of (quality, trial_array, trialtm_array) for each cluster."""
        clusters = []
        if self.is_h5:
            clu_ref = self.obj['clu']
            for pnum in probe_nums:
                pidx = pnum - 1
                clu_group = self.f[clu_ref[pidx, 0]]
                quality_refs = clu_group['quality']
                for i in range(quality_refs.shape[0]):
                    q = ''.join(chr(int(c)) for c in np.array(self.f[quality_refs[i, 0]]).flatten()).strip().lower()
                    trial = np.array(self.f[clu_group['trial'][i, 0]]).flatten()
                    trialtm = np.array(self.f[clu_group['trialtm'][i, 0]]).flatten()
                    clusters.append((q, trial, trialtm))
```

iii. The AI read the authors' `getSeq.m` (step 26), which bins `obj.clu{prbnum}(curClu).trialtm_aligned` per trial, and `findClusters.m` for the quality field, and reproduced that field set. The v5 branch additionally handles the case where `clu` is a flat array of cluster structs for a single probe.

## 2-b. How is the `neural` data processed?

i. Spikes of one cluster in one trial are histogrammed into the 500 fixed bin edges, divided by the bin width to give spikes/s, and smoothed along time with a **causal** Gaussian: `scipy.signal.windows.gaussian(15, std=(15-1)/(2·2.5))` — MATLAB `gausswin`'s default alpha — with the first `floor(N/2)` taps zeroed and the kernel renormalised, applied by `np.convolve(..., mode='same')` after prepending the first 15 samples of the trace as a reflect boundary and trimming them off afterwards. This is a line-by-line port of the authors' `utils/mySmooth.m`. No normalisation, z-scoring or baseline subtraction is applied; stored values are firing rates in Hz (`float64`).

ii.
```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    ...
    if bctype == 'reflect':
        x_filt = np.concatenate([x[:N, :], x], axis=0)
        trim = N
    ...
    kern = windows.gaussian(N, std=(N-1)/(2*2.5))
    kern[:N//2] = 0
    kern = kern / kern.sum()
    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
    out = out[trim:, :]
```

```python
            aligned_times = spike_times[trial_mask] - gocue[j]
            counts, _ = np.histogram(aligned_times, bins=time_edges)
            fr = counts / DT
            trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The AI read `mySmooth.m` in full (step 56) and reproduced it exactly, including the `kern(1:floor(numel(kern)/2)) = 0; %causal` line, the `reflect` padding and the trim; its header comment states "Causal Gaussian smoothing (window=15 bins, reflect boundary)" and the parameters come from `params.smooth = 15; params.bctype = 'reflect'` in `WorkingWithDataObjs.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters and one session-level filter. (1) The lower-cased, stripped `clu.quality` string is dropped if it is in `{garbage, gabrga, noisy, real?}`; everything else — including `poor`, `fair` and multi-units — is kept. (2) After binning and smoothing, any unit whose mean firing rate over the window, averaged across the kept trials, is not greater than 1 Hz is dropped. (3) A session with fewer than 10 surviving units would be dropped entirely (no session triggered this; the smallest kept session has 17). This leaves 2,504 units over 45 sessions, 17–142 per session.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR_THRESH = 1.0
...
    filtered_clusters = [(q, t, tt) for q, t, tt in clusters if q not in EXCLUDE_QUALITIES]
...
    mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
    keep_units = mean_frs > LOW_FR_THRESH
```

iii. The notes say this matches "`findClusters.m` with `quality = 'all'`... Include all clusters EXCEPT those with quality labels: `garbage`, `gabrga`, `noisy`, `real?`" and "`removeLowFRClusters.m`: Remove units with mean FR <= 1 Hz", with `params.lowFR = 1` taken from `WorkingWithDataObjs.m`. The AI also explained the resulting unit count against the paper: "We include ALL quality clusters (not just single units) matching `params.quality = {'all'}`... The paper's unit counts refer to specific quality subsets used for particular analyses."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By subtraction only. `clu.trialtm` is already on the behaviour clock and already relative to its own trial's onset, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]` gives seconds from go cue directly; those values are then histogrammed into the fixed −2.5…2.5 s edges. No interpolation, no offset, and no per-session correction is applied to the neural stream (only the video streams get the bitcode offset).

ii.
```python
    gocue = sd.get_event_times('goCue')
    ...
            aligned_times = spike_times[trial_mask] - gocue[j]
            counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. `params.alignEvent = 'goCue'` in `WorkingWithDataObjs.m` and `alignSpikes.m`/`getSeq.m`, which bin `trialtm_aligned` relative to the align event; the AI's file header records "Align to go cue" as a parameter "matching reference code", and the instructions require go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms. `DT = 1/100`, giving 500 non-overlapping bins spanning −2.5 to +2.5 s from the go cue, with the input being the bin centres (−2.495 … 2.495 s). Spikes are counted directly into this grid — there is no finer intermediate binning and therefore no rebinning step — and every other stream (input, tongue, paw, motion energy) is sampled onto exactly the same 500-bin axis, so all streams share one time base for all trials and sessions.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1/100  # 10 ms
...
    time_edges = np.arange(TMIN, TMAX + DT, DT)
    time_axis = time_edges[:-1] + DT / 2
```

iii. Taken verbatim from the authors' tutorial script, which the AI read at step 16: `params.tmin = -2.5; params.tmax = 2.5; params.dt = 1/100;`, and reproduced in `getSeq.m` as `edges = params.tmin:params.dt:params.tmax; obj.time = edges + params.dt/2`. The AI labelled these "Parameters (matching reference code: WorkingWithDataObjs.m)". It did not remark on the comment sitting two lines above that assignment in the same file — "use a 5 ms bin width and bin spike data from -2.5 to 2.5 sec" — which contradicts `params.dt = 1/100`, nor did it consult `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. None — the input is not read from the data. It is the centre of each of the 500 bins of the analysis window, i.e. the same `time_axis` that defines the spike histogram, implicitly derived from `bp.ev.goCue` through the alignment. The same vector is stored for every trial of every session as a `(1, 500)` array.

ii.
```python
    time_edges = np.arange(TMIN, TMAX + DT, DT)
    time_axis = time_edges[:-1] + DT / 2
...
        input_data = time_axis.reshape(1, -1).copy()
```

iii. The notes describe it as "**time_from_go_cue**: Time axis relative to go cue onset (continuous, -2.495 to 2.495 s)", following the instructions' requirement for a continuous, time-varying "time from go cue onset in seconds" input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the bin-centre vector once in `main()` and copying it per trial. It is stored in seconds, unnormalised, as `float64`.

ii.
```python
        input_data = time_axis.reshape(1, -1).copy()
        ...
        input_trials.append(input_data)
```

iii. N/A — no justification needed or given; the quantity is defined by the analysis window.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. Perfectly by construction: `time_axis` is `time_edges[:-1] + DT/2`, and `time_edges` is the same array passed as `bins=` to the spike histogram, so input sample *k* is the centre of the bin that holds the spikes counted into neural column *k*. The same vector is reused for every session, so alignment cannot drift across sessions.

ii.
```python
    time_edges = np.arange(TMIN, TMAX + DT, DT)
    time_axis = time_edges[:-1] + DT / 2
...
            counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. A single per-trial field, `obj.bp.R` — the flag marking a right-instructed trial. `bp.hit` and `bp.miss` are read but are used only for the trial mask and the outcome variable, not for the lick direction; `bp.L` is read and never used. Because ignore trials were already removed (1-e), only two values are ever produced.

ii.
```python
    R = sd.get_trial_array('R').astype(bool)
    L = sd.get_trial_array('L').astype(bool)
...
        lick_dir = 1 if R[trial_idx] else 0
```

iii. The notes state: "**lick_direction**: left=0, right=1 (per-trial, from bp.R)". No reasoning is recorded about whether `bp.R` describes the port the animal licked or the port it was instructed to lick; the AI had read `WorkingWithDataObjs.m`, which describes `obj.bp.hit & obj.bp.R` as "right_correct_trials_mask" and `obj.bp.miss & obj.bp.L` as "left_error_trials_mask", but did not combine the flags.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct relabelling of one boolean: right-instructed trials get 1, everything else 0. The value is constant within a trial and is broadcast across all 500 bins. `output_values[0] = ['left', 'right']` — there is no third class, since no-lick trials were excluded upstream. Across the dataset the split is 49.9% / 50.1%.

ii.
```python
        lick_dir = 1 if R[trial_idx] else 0
...
        out = np.zeros((6, n_timebins), dtype=np.int64)
        out[0, :] = t['lick_dir']
```

iii. Only the mapping is justified in the notes ("left=0, right=1"); the two-class scheme follows mechanically from the decision to keep only hit and miss trials. The instructions' third value, "none", is never mentioned.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial field, `obj.bp.autowater`, which marks trials on which water was delivered irrespective of the animal's choice, i.e. the water-cued blocks.

ii.
```python
    autowater = sd.get_trial_array('autowater')
...
        context = 0 if autowater[trial_idx] == 1 else 1
```

iii. Taken from the authors' tutorial, which the AI read at step 16: "obj.bp.autowater=1 when water was delivered regardless of animal choice (0 otherwise). this field can be used as a proxy for obtaining water-cued blocks and delayed-response blocks of trials."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `autowater == 1` → WC (0), everything else → DR (1), constant within a trial and broadcast across the 500 bins. Across the dataset 8.2% WC / 91.8% DR; 17 of the 45 sessions are DR-only and carry a constant value.

ii.
```python
        context = 0 if autowater[trial_idx] == 1 else 1
...
        out[1, :] = t['context']
```
```python
        'output_values': [..., ['WC', 'DR'], ...]
```

iii. Codes follow the instructions' ordering (WC, DR). The AI flagged the degenerate sessions in its notes: "**DR-only sessions**: Behavioral context is always DR=1 for sessions without WC blocks (JEB13 sessions 11-12, JEB14, and all RandomizedDelay sessions)", and chose to keep them.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `obj.bp.hit`, with `obj.bp.miss` entering only through the trial mask — within the kept trials, not-hit is by construction miss. `obj.bp.no` (the ignore flag) is never read, and the trials it marks are gone before this point, so no third outcome class can be formed.

ii.
```python
    hit = sd.get_trial_array('hit').astype(bool)
    miss = sd.get_trial_array('miss').astype(bool)
    valid_trials = (hit | miss) & ~stim_enable & ~early
...
        outcome = 1 if hit[trial_idx] else 0
```

iii. The notes say "**outcome**: incorrect=0 (miss), correct=1 (hit) (per-trial)". The exclusion of `no` trials is justified only as "Matching reference code conditions" (the authors' `params.condition` strings never include ignore trials).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A direct relabelling of the hit flag into two classes, correct (1) and incorrect (0), constant within a trial and broadcast across the 500 bins. `output_values[2] = ['incorrect', 'correct']`; the "ignore" class required by the instructions is absent. Across the dataset 13.7% incorrect / 86.3% correct.

ii.
```python
        outcome = 1 if hit[trial_idx] else 0
...
        out[2, :] = t['outcome']
```

iii. As above: the class codes follow the instructions' ordering for the two classes that remain, and the third is an unremarked consequence of the trial filter.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut output in `obj.traj`, bottom camera only (`view = 1`), feature `top_tongue`: channels 0 and 1 of `ts` (x and y pixel position per frame) together with that view's `frameTimes`. The likelihood channel (`ts[...,2]`) is never read. If `top_tongue` is missing the code falls back to the first feature whose name contains "tongue". `bp.ev.goCue`, `bp.ev.bitStart`, `sglx.bitcode.bitstart` and `sglx.fs` are also needed, for the clock correction. The side camera's `tongue` feature is not used.

ii.
```python
        bottom_feats = sd.get_traj_feature_names(1)
        for idx, name in enumerate(bottom_feats):
            if name == 'top_tongue' and tongue_feat_idx is None:
                tongue_feat_idx = idx
        if tongue_feat_idx is None:
            for idx, name in enumerate(bottom_feats):
                if 'tongue' in name.lower():
                    tongue_feat_idx = idx
                    break
```
```python
                ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
```
```python
            x = ts[feat_idx, 0, :]      # v7.3 layout (n_feats, 3, n_frames)
            y = ts[feat_idx, 1, :]
```

iii. The AI enumerated both cameras' features (step 39: side = `tongue, left_tongue, right_tongue, jaw, trident, nose, lickport`; bottom = `top_tongue, ..., top_paw, bottom_paw, ...`) and settled on the bottom view for both tongue and paw. The paper's methods, which the AI extracted, say "Tongue angle and length were found using the bottom camera", and "the paws were tracked using only the bottom view" — using one camera for both keeps a single frame-time source per trial. No reason is recorded for not also using the side-camera tongue.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps. (1) Speed per frame is the Euclidean displacement between consecutive frames divided by a **hard-coded** 2.5 ms frame period (`VIDEO_FPS = 400`), i.e. `sqrt(dx²+dy²)·400`, timestamped at the midpoint of the frame pair. No smoothing is applied to the positions before differentiating and no explicit likelihood cut is applied — untracked frames already carry NaN x/y in the source (verified: x is NaN exactly where likelihood ≤ 0.9), so NaNs propagate through `np.diff` and mark those frames and their immediate neighbours as missing. (2) The frame times are put on the go-cue clock (7-d). (3) The signal is **point-sampled**, not averaged, onto the 500 bin centres with `np.interp`, with `left=right=NaN` so bins outside the video's coverage stay NaN; at 400 Hz frames and 10 ms bins this keeps roughly one frame in four. (4) Per session, the 50th percentile of all non-NaN values pooled over all trials and bins is the threshold. No normalisation is applied (only one camera is used).

ii.
```python
                ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
                dt_vid = 1.0 / VIDEO_FPS
                vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
                vel_t = ft[:-1] + dt_vid / 2
                vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
                tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```
```python
    all_tongue = np.concatenate([t['tongue_vel'] for t in raw_outputs])
    tongue_thresh = np.nanpercentile(all_tongue, 50) if not np.all(np.isnan(all_tongue)) else 0
```

iii. The 400 Hz rate comes from the paper's methods, which the AI extracted: "High-speed video was captured (400-Hz frame rate) from two cameras". An earlier draft carried a comment quoting the same methods on missing values — "Missing values were filled in with the nearest available value for all features, except for the tongue... For tongue, we don't fill" — and the final code indeed fills nothing. "Velocity = sqrt(dx^2 + dy^2) / dt at 400 Hz, interpolated to 10ms bins" is how the notes describe it; it corresponds to the methods' "first-order derivative of the position vector". No justification is recorded for point-sampling rather than averaging within bins, or for skipping the position smoothing.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two classes, not three. The session's 50th percentile over non-NaN values splits tracked bins into 0 (`< 50th pctl`) and 1 (`>= 50th pctl`), and **NaN bins — those where the tongue is not visible or outside video coverage — are folded into class 0** rather than into the separate "not visible" class 2 that the instructions specify. `output_values[3] = ['< 50th pctl', '>= 50th pctl']`. Because the tongue is out of view in the large majority of bins, the realised split is 94.8% / 5.2% rather than anything near 50/50.

ii.
```python
        tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                              np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
```

iii. The AI noticed the consequence and reasoned about it twice. In its notes: "**Tongue velocity skew**: ~95% of tongue velocity bins fall below threshold because the tongue is visible only during licking (~response epoch), making most time bins have near-zero velocity", and "NaN values (outside video coverage) assigned to class 0". In its reasoning at step 126 it went further — "The NaN values are converted to 0 in the discretization step... I'm realizing the core problem: I should only compute velocity for the time range actually covered by video, leaving NaN values as NaN rather than converting them to 0" — but then concluded "the sparse velocity signal is likely just because tongue movement is localized to brief licking periods" and moved on without changing the code or adding the third class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the behaviour clock, so a per-session offset is computed once from the bitcode pulse recorded on both: `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` (≈0.49 s on the session checked). Each frame's time becomes `frameTimes − vidshift − goCue[trial]`, and the velocity samples (at frame midpoints) are then interpolated onto the shared 500-bin centre grid, so tongue bin *k* refers to the same instant as neural bin *k*.

ii.
```python
    def get_video_offset(self):
        ...
        bs_mode = float(stats.mode(bitStart, keepdims=False).mode)
        sglx_mode = float(stats.mode(sglx_bitstart, keepdims=False).mode)
        return sglx_mode / sglx_fs - bs_mode
```
```python
                vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
                tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. A direct port of the authors' `findVideoOffset.m`, which the AI read at step 52 (`bitStart = mode(obj.bp.ev.bitStart); vidFileOffset = mode(obj.sglx.bitcode.bitstart) / obj.sglx.fs; vidshift = vidFileOffset - bitStart;`) and sanity-checked numerically at step 53: "the video offset is ~0.49s, close to the 0.5s mentioned in the tutorial code". The offset is computed once per session, outside the trial loop.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` bottom-camera tracking, feature `bottom_paw` (x, y channels plus that view's `frameTimes`), with a fallback to the first feature whose name contains "paw" if `bottom_paw` is absent. The other tracked forepaw, `top_paw`, is not used.

ii.
```python
PAW candidate selection:
        for idx, name in enumerate(bottom_feats):
            if name == 'bottom_paw' and paw_feat_idx is None:
                paw_feat_idx = idx
        if paw_feat_idx is None:
            for idx, name in enumerate(bottom_feats):
                if 'paw' in name.lower():
                    paw_feat_idx = idx
                    break
```
```python
                ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
```

iii. The notes say only "**paw_velocity**: Same as tongue but using `bottom_paw` feature". The paper's statement that "the paws were tracked using only the bottom view" justifies the camera; nothing in the trajectory justifies picking `bottom_paw` over `top_paw`, and the AI did not compare their tracking reliability.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Byte-for-byte the same pipeline as the tongue: frame-to-frame Euclidean displacement times 400 Hz, no position smoothing, no explicit likelihood cut (NaNs already mark untracked frames), point-sampled onto the 500 bin centres with `np.interp`, and split at the session's 50th percentile of non-NaN values. No normalisation, since a single camera is used. The code is a literal duplicate of the tongue block rather than a shared helper.

ii.
```python
        paw_vel = np.full(n_timebins, np.nan)
        if paw_feat_idx is not None:
            try:
                ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
                dt_vid = 1.0 / VIDEO_FPS
                vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
                vel_t = ft[:-1] + dt_vid / 2
                vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
                paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
            except:
                pass
```

iii. "Same as tongue but using `bottom_paw`" is the whole recorded rationale; the underlying justification is the paper's "The velocity of each feature was then calculated as the first-order derivative of the position vector." Note the paper also says missing values are nearest-filled "for all features, except for the tongue", which would apply to the paw; the AI quoted this in an earlier draft but did not implement it.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Two classes: below / at-or-above the session's 50th percentile of non-NaN values, with NaN bins (paw untracked, or outside video coverage) folded into class 0 instead of the instructions' class 2 "not visible". `output_values[4] = ['< 50th pctl', '>= 50th pctl']`. Realised split 58.9% / 41.1% across the dataset, i.e. roughly 18% of bins are silently invisible-labelled-as-slow.

ii.
```python
    paw_thresh = np.nanpercentile(all_paw, 50) if not np.all(np.isnan(all_paw)) else 0
...
        paw_disc = np.where(np.isnan(t['paw_vel']), 0,
                           np.where(t['paw_vel'] >= paw_thresh, 1, 0)).astype(np.int64)
```

iii. Same as 7-c: "NaN values (outside video coverage) assigned to class 0". The AI noticed a per-session symptom — "**Paw velocity skew in some sessions**: JEB15 sessions show very high fraction of low paw velocity, likely due to camera positioning or animal behavior" (sessions 18–21 are 88–97% class 0) — attributed it to the camera rather than to the NaN-to-0 collapse, and left it.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue: the one session-level `vidshift` from the bitcode is subtracted from `frameTimes`, then that trial's `goCue`, and the frame-midpoint velocity samples are interpolated onto the same 500-bin centre grid the spikes were counted into. Both features come from the bottom camera, so both use that camera's frame times.

ii.
```python
    vidshift = sd.get_video_offset()
...
                vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
                paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. Same as 7-d: a port of `findVideoOffset.m` plus the tutorial's `frameTimes - vidshift - goCue` recipe, applied uniformly to every video-derived stream.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` beside each data structure, which holds one already-reduced trace per trial with one value per camera frame. `obj.me` (present only in some sessions) is not used. The loader handles all three on-disk layouts and also reads `me.moveThresh` — the authors' manual per-session movement threshold — but never uses it. The bottom camera's `frameTimes` supply the time base.

ii.
```python
def load_motion_energy(data_dir, anm, date):
    me_fn = f'motionEnergy_{anm}_{date}.mat'
    me_path = os.path.join(DATA_ROOT, data_dir, me_fn)
    if not os.path.exists(me_path):
        return None, None
    ...
        thresh = float(me_struct['moveThresh'].flatten()[0])
        data_field = me_struct['data']
        if data_field.dtype.names is not None and 'data' in data_field.dtype.names:
            data_field = data_field[0, 0]['data']
        data_arr = data_field.flatten()
        me_list = [data_arr[i].flatten().astype(float) for i in range(len(data_arr))]
        return me_list, thresh
```
```python
    me_data, me_thresh = load_motion_energy(data_dir, anm, date)
```

iii. The AI iterated on this loader for many steps after hitting each layout in turn, and cites the authors' guard: "The reference code handles this: `if isstruct(me.data); me.data = me.data.data; end`". Its notes record "**Motion energy loading**: Some ME files have different formats (struct vs cell array). Both are handled in the loader."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling. The per-frame value is already the paper's spatial reduction (99th percentile of per-pixel motion energy across the frame), so the trace is only put on the go-cue clock and point-sampled onto the 500 bin centres with `np.interp` (`left=right=NaN`), then split at the session's 50th percentile of non-NaN values. `me.moveThresh`, the authors' own per-session movement threshold, is loaded and discarded in favour of the instructions' percentile rule.

ii.
```python
        me_interp = np.full(n_timebins, np.nan)
        if me_data is not None and trial_idx < len(me_data):
            try:
                ft = sd.get_trial_frame_times(1, trial_idx)
                ft_aligned = ft - vidshift - gocue[trial_idx]
                me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
            except:
                pass
```
```python
    me_thresh_50 = np.nanpercentile(all_me, 50) if not np.all(np.isnan(all_me)) else 0
```

iii. The notes describe it as "Loaded from separate `motionEnergy_*.mat` files, Interpolated from 400 Hz video frame rate to 10ms time bins" and "Discretized at per-session 50th percentile", the latter being the instructions' requirement rather than the paper's manual threshold.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Two classes: below / at-or-above the session's 50th percentile of non-NaN values, with NaN bins folded into class 0 rather than the instructions' class 2 "no video". `output_values[5] = ['< 50th pctl', '>= 50th pctl']`. The realised split is close to even (53.2% / 46.8%) because motion energy covers nearly the whole window — except in session 35 (`JEB23_2023-10-20`), where the interpolation raised on every trial and the entire output dimension is constant 0 for all 308 trials.

ii.
```python
        me_disc = np.where(np.isnan(t['motion_energy']), 0,
                          np.where(t['motion_energy'] >= me_thresh_50, 1, 0)).astype(np.int64)
```

iii. Same rule as the other two movement variables. The degenerate session was noticed — "**Session 35 (JEB23_2023-10-20)**: Motion energy output is all class 0, indicating ME data was unavailable for this session" — but not diagnosed (the file loads fine; it holds 307 traces for a 352-trial session and its trial 0 trace is 3,508 samples against 2,188 frame times, so `np.interp` raises and the bare `except: pass` swallows it) and not fixed.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. By the same session `vidshift` and the same per-trial `goCue` subtraction, applied to the **bottom** camera's `frameTimes`, then interpolated onto the shared 500-bin grid. Motion energy has one value per frame and is paired positionally with those frame times; in the sessions inspected both cameras report identical frame counts and times, so the choice of view is immaterial there, but the code has no check that the trace length matches the frame-time length (see 9-c).

ii.
```python
                ft = sd.get_trial_frame_times(1, trial_idx)
                ft_aligned = ft - vidshift - gocue[trial_idx]
                me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. "Frame times aligned: frameTimes - vidshift - goCue[trial]", quoted in the notes as "Matching `findVideoOffset.m` and `WorkingWithDataObjs.m`". The AI verified at step 49 that side and bottom cameras returned the same frame count and the same time range for the trial it inspected, which is presumably why it reused the bottom camera's times for a side-camera-derived quantity.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms. (1) **Whole sessions**: a missing file, a missing `clu` field, fewer than 5 surviving trials, fewer than 10 surviving units, or fewer than 2 trials cause `process_session` to print a warning and return `None`, dropping the session. (2) **Missing video / tracking**: every per-trial video computation sits inside `try: ... except: pass` with the output pre-filled with NaN, so any failure — a missing feature, an unreadable trial, a length mismatch — silently yields NaN for that trial. (3) **NaN bins**: all NaN, whatever its cause (feature untracked, frame outside the window, exception), is mapped to class 0 at discretisation, so missing data is indistinguishable in the output from a genuine low-velocity measurement. (4) **Missing motion energy**: `me_data is None` or `trial_idx >= len(me_data)` leaves the trial NaN → class 0. What is *not* handled: trials that fall past the end of the ephys recording are kept, producing 30 trials of all-zero neural data across two sessions, which the format verifier flagged; and the swallowed exceptions cost one session its entire motion-energy dimension without any error being printed.

ii.
```python
        me_interp = np.full(n_timebins, np.nan)
        if me_data is not None and trial_idx < len(me_data):
            try:
                ...
            except:
                pass
```
```python
        tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                              np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
```
```python
    if not os.path.exists(fpath):
        print(f'  WARNING: File not found: {fpath}')
        return None
    ...
    except (KeyError, AttributeError) as e:
        print(f'  WARNING: No cluster data found ({e}), skipping')
```

iii. The AI's stated position is that missing video simply means no movement was measured: "NaN values (outside video coverage) assigned to class 0". It recorded both resulting anomalies in its notes (the all-zero neural trials and the all-class-0 motion energy in session 35) and shipped them. Its reasoning at step 126 shows it considered keeping NaN separate and decided against it without testing the alternative.

## 11-a. What are the most time-consuming steps of the code?

i. Two things dominate. First, reading each session file — unavoidable, and for the v5 sessions `scipy.io.loadmat` materialises the whole ~120 MB object graph eagerly. Second, and specific to this implementation, the spike-binning/smoothing double loop: for every cluster × every trial (not just the kept ones) it builds a boolean mask over that cluster's entire spike vector, calls `np.histogram` on 500 edges, and then calls `causal_gaussian_smooth`, which itself allocates and convolves a 515-sample vector. That is roughly `n_units × n_trials` ≈ 20,000–40,000 Python-level iterations per session, each doing three small array allocations, against the 44 vectorised `histogram2d` calls the reference solution uses. The video work is third: one `np.interp` per trial per stream plus repeated h5py reference dereferencing.

ii.
```python
    trialdat = np.zeros((n_units, n_timebins, ntrials))
    for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
        for j in range(ntrials):
            trial_mask = (trial_nums == (j + 1))
            if not np.any(trial_mask):
                continue
            aligned_times = spike_times[trial_mask] - gocue[j]
            counts, _ = np.histogram(aligned_times, bins=time_edges)
            fr = counts / DT
            trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The AI recorded no runtime analysis at all — there is no timing instrumentation in the script and no discussion of performance in the trajectory or the notes. The conversion completed, so the cost was evidently tolerable.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main one is the cluster × trial loop above. Both levels are avoidable: all of a cluster's spikes can be counted onto the (trial × bin) grid in one `np.histogram2d` call using the per-spike trial label, and the smoothing can be applied to the whole `(n_trials, n_bins)` matrix along the time axis in one call instead of once per trial. The inner `trial_mask = (trial_nums == j+1)` additionally rescans the full spike vector `n_trials` times per cluster, turning a linear pass into a quadratic one. The per-column `for j in range(x_filt.shape[1])` loop inside `causal_gaussian_smooth` is also redundant given it is always called with a 1-D vector. The remaining per-trial loops — the three `np.interp` calls — are genuinely ragged (each trial has a different frame count) and are reasonable to leave as loops, exactly as the reference does.

ii.
```python
        for j in range(ntrials):
            trial_mask = (trial_nums == (j + 1))
```
```python
    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```

iii. No justification is recorded; the loop structure mirrors the authors' MATLAB `getSeq.m`, which loops over clusters and trials in exactly this way, so the AI appears to have transliterated it rather than chosen it.

## 11-c. What processing does the code repeat multiple times?

i. Little genuine recomputation of results, but several repeated accesses. The video offset is computed once per session (good). The per-trial `np.interp` for each stream is done once. What repeats is I/O plumbing: `get_trial_traj` re-resolves `self.obj['traj'][view, 0]` and re-opens the `frameTimes`/`ts` datasets on every call, so the same trajectory group is dereferenced three times per trial (tongue, paw, motion-energy frame times), and `get_trial_frame_times` re-reads the frame times a trial already loaded twice. The spike mask rescan described in 11-b is the largest repeated work. The tongue and paw velocity blocks are also copy-pasted rather than factored into one function. On the output side, the identical 500-sample `time_axis` is `.copy()`-ed once per trial, producing 12,293 duplicates of the same vector.

ii.
```python
    def get_trial_traj(self, view, trial_idx, feat_idx):
        if self.is_h5:
            traj_ref = self.obj['traj'][view, 0]
            traj_g = self.f[traj_ref]
            ft_ref = traj_g['frameTimes'][trial_idx, 0]
            frame_times = np.array(self.f[ft_ref]).flatten()
```
```python
        input_data = time_axis.reshape(1, -1).copy()
```

iii. Not discussed in the trajectory. The `SessionData` accessor design deliberately trades repeated lookups for a uniform interface over the two MATLAB formats, which is a reasonable motivation even though the lookups were never cached.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four things. (1) `trialdat` is allocated and filled for **all** `Ntrials`, so spikes are binned and smoothed for the ~19% of trials (early-lick, photostim, ignore) that are then discarded by `valid_idx` — about 2,900 trials' worth of convolutions computed and thrown away. (2) The array is `float64`, and the per-trial slices are pickled as such, so the output file is 3.1 GB for 500 bins where `float32` would halve it at no cost to firing-rate precision; the outputs are likewise `int64` for values in {0, 1}. (3) Values are read and never used: `bp.L`, `me.moveThresh` (the authors' per-session movement threshold, superseded by the percentile rule), and the unused `sys` import. (4) The script writes a second 177 MB `sample_data.pkl` containing the first three sessions, which is not part of the required deliverables and duplicates data already in the full pickle. The `scipy.io.loadmat` path for v5 sessions also materialises the entire object graph, including `clu.spkWavs` and the tracked features other than the tongue and paw, but that is inherent to the reader.

ii.
```python
    trialdat = np.zeros((n_units, n_timebins, ntrials))     # all trials, float64
    ...
    mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
```
```python
    L = sd.get_trial_array('L').astype(bool)                # never used
    me_data, me_thresh = load_motion_energy(data_dir, anm, date)   # me_thresh never used
```
```python
    out = np.zeros((6, n_timebins), dtype=np.int64)
```

iii. Nothing is recorded about memory or precision. The full-trial `trialdat` is a side effect of following `getSeq.m`, which also builds `obj.trialdat` over `obj.bp.Ntrials` before any condition selection; the extra sample pickle was the AI's own device for fast iteration during development.
