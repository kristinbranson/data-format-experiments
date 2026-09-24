# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Sessions are **not** discovered by globbing the data folders. The AI hard-codes a dictionary `SESSION_META` mapping `(animal, date) -> ([probe numbers], folder)`, transcribed from the authors' `DataLoadingScripts/load<ANM>_ALMVideo.m` files, containing 45 sessions across the two ephys folders (`Ephys_Behavior`, 25 entries; `RandomizedDelay_Ephys_Behavior`, 20 entries). At start-up `main()` keeps only the entries whose `data_structure_<anm>_<date>.mat` actually exists on disk (all 45 do). Each session is then opened once by a `SessionData` class that transparently handles both MATLAB formats: `h5py.File` for v7.3 files (31 sessions) and `scipy.io.loadmat` for v5 files (14 sessions). Data are read lazily per field through accessor methods (`get_bp_field`, `get_event`, `get_spike_data`, `get_traj_trial`, ...) rather than materialising the whole `obj` tree. Motion energy is read from the separate `motionEnergy_<anm>_<date>.mat` file in the same folder.

ii.
```python
SESSION_META = {
    ('EKH1', '2021-08-07'): ([2], 'Ephys_Behavior'),
    ...
    ('JEB24', '2023-11-03'): ([1], 'RandomizedDelay_Ephys_Behavior'),
}
```
```python
class SessionData:
    def _open(self):
        try:
            self.f = h5py.File(self.data_file, 'r')
            self.is_hdf5 = True
        except:
            data = scipy.io.loadmat(self.data_file, squeeze_me=False)
            self.obj = data['obj']
            self.is_hdf5 = False
```
```python
    for (anm, date), (probes, dtype) in sorted(SESSION_META.items()):
        data_file = os.path.join(DATA_DIR, dtype, f'data_structure_{anm}_{date}.mat')
        if os.path.exists(data_file):
            sessions.append((anm, date, probes, dtype))
```

iii. From CONVERSION_NOTES Step 1/Step 4: the animal-specific `load<ANM>_ALMVideo.m` scripts are the authoritative record of which sessions and which probe entered the paper's analyses ("RandDelay sessions: 20 with loading functions / 22 data files / paper says 19 — include 20 (2 files without loading functions excluded)"). The dual-format reader is justified in Step 2/Step 10 Check 5: "HDF5 (.mat v7.3) and MATLAB v5 formats ... Handled both HDF5 and MATLAB v5 file formats".

## 1-b. How are the data split into subjects?

i. The subject is the animal ID, taken as the first element of the `SESSION_META` key (i.e. the part of the filename before the underscore); it is never read from inside the file. Subjects are accumulated in first-encounter order into `all_subjects`, and `subject_idx` records each session's index into that list. Result: 14 subjects (EKH1, EKH3, JEB6, JEB7, JEB11–JEB15, JEB19, JEB23, JEB24, JGR2, JGR3) over 43 retained sessions.

ii.
```python
        if anm not in all_subjects:
            all_subjects.append(anm)
        subject_idx_list.append(all_subjects.index(anm))
```
```python
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. Step 2 of CONVERSION_NOTES counts "Unique animals (Ephys) 10 / Unique animals (RandDelay) 4" from the filenames, and Step 4 records a discrepancy with the paper ("Mice DR: 10 unique animal IDs in data vs 9 in paper — include all 10; paper may count differently"). The animal-in-the-filename convention is taken from the authors' own loading scripts.

## 1-c. How are the data split into sessions?

i. One `(animal, date)` key = one `data_structure_*.mat` file = one session = one element of `neural`/`input`/`output`/`brain_region_idx`/`subject_idx`. The folder (`Ephys_Behavior` vs `RandomizedDelay_Ephys_Behavior`) is stored only as metadata, so fixed-delay and randomized-delay sessions are pooled into one uniform session list rather than treated as two datasets. Sessions recorded with two probes (the three JEB15 sessions) are kept as **one** session with the units from both probes concatenated along the neuron axis. 45 candidate sessions are processed, 43 survive curation (2 JEB19 sessions dropped, see 1-e).

ii.
```python
    for i, (anm, date, probes, dtype) in enumerate(sessions):
        result = process_session(anm, date, probes, dtype, args.show_processing)
        if result is None:
            continue
        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
```
```python
    for probe_num in probes:
        ...
        all_trialdat.append(trialdat)
    trialdat = np.concatenate(all_trialdat, axis=1)
```

iii. Step 5 Key Decision 1: "Include both Ephys_Behavior and RandomizedDelay sessions: Both have neural + behavioral data". Step 10 Check 5: "Handled dual-probe sessions (JEB15)". Step 9 acknowledges the resulting counts (23 DR + 20 RandDelay) differ from the paper's 25 + 19 but leaves them unresolved.

## 1-d. How are the data split into trials?

i. The trial is the atomic unit of the Bpod table: `obj.bp.Ntrials` gives the count, and every behavioural field (`L`, `R`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`, `ev.goCue`) is read as a flat 1-D array with one entry per trial. Spikes carry their own 1-based trial number in `clu.trial`, and camera tracking is already stored per trial (`traj{view}(trial).ts`, `.frameTimes`), so no trial boundaries need to be reconstructed. Trial identity is kept as an index into the original, unfiltered trial axis (`valid_trial_indices`), and every per-trial array (neural, video, behaviour) is built at full length `n_trials_total` and indexed by that array at the very end.

ii.
```python
    def get_n_trials(self):
        if self.is_hdf5:
            return int(np.array(self.f['obj/bp/Ntrials']).flatten()[0])
        else:
            return int(self.obj['bp'][0,0]['Ntrials'][0,0].flatten()[0])
```
```python
    valid_trial_indices = np.where(valid_trials)[0]
    ...
    for t_idx in valid_trial_indices:
        neural_trials.append(trialdat[:, :, t_idx].T.astype(np.float32))
```
```python
        trial_int = trial.astype(int)
        valid_mask = (trial_int >= 1) & (trial_int <= n_trials)
```

iii. Step 2 of CONVERSION_NOTES documents the `bp` fields as per-trial vectors of length `Ntrials`, and Step 1 notes `alignSpikes: trialtm_aligned = trialtm - ev.goCue(trial)` — i.e. `clu.trial` indexes the same trial table. No further justification is given; the split is taken directly from the file structure.

## 1-e. How are trials filtered based on quality controls?

i. **Trial level:** a trial is kept only if `early == 0` **and** `no == 0` **and** `stim.enable == 0` — i.e. early-lick trials, photostimulation trials, **and ignore (no-response) trials** are all removed. **Session level:** two further criteria are applied, both taken from the reference code / methods: the session must have `> 40` right-hit and `> 40` left-hit DR trials (`UseInclusionCritera.m`), and must retain `>= 10` units; sessions failing either are dropped entirely. Two JEB19 sessions are dropped by the >40 criterion; none is dropped by the unit criterion. 11,981 of ~15,500 trials survive.

No filter is applied for trials that occur after the ephys recording ends. The verification log therefore reports 30 retained trials (19 in session 29 = JEB24_2023-10-23, 11 in session 36 = JEB24_2023-11-03) in which *every* neuron has exactly zero firing for all 1000 bins.

ii.
```python
    valid_trials = (early == 0) & (no_resp == 0) & (stim_enable == 0)
    valid_trial_indices = np.where(valid_trials)[0]

    # Inclusion criteria: >40 right hit DR AND >40 left hit DR
    r_hit_dr = (R == 1) & (hit == 1) & (stim_enable == 0) & (autowater == 0) & (early == 0)
    l_hit_dr = (L == 1) & (hit == 1) & (stim_enable == 0) & (autowater == 0) & (early == 0)
    ...
    if n_r_hit_dr <= 40 or n_l_hit_dr <= 40:
        print(f'    EXCLUDED: insufficient DR hit trials (need >40 each)')
        sd.close(); return None
    if len(valid_trial_indices) < 2:
        print(f'    EXCLUDED: fewer than 2 valid trials')
        sd.close(); return None
```
```python
    if n_neurons < 10:
        print(f'    EXCLUDED: only {n_neurons} neurons (need >= 10)')
        sd.close(); return None
```

iii. Step 3: "Exclude early lick and ignore trials from behavioral analysis" (read out of methods.txt); Step 5 Key Decision 2: "Trial filtering: Exclude early lick, ignore, stimulation trials (matching paper)". Step 3/Step 1: "Session min units >= 10" and "Session min DR trials: >40 correct per direction — UseInclusionCritera.m". For the all-zero trials, Step 10 Check 1 states: "These are trials that occurred after the neural recording ended... Cannot be fixed without excluding these trials, but they are valid behavioral trials. The zero neural data is correct given the spike data."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters `obj.clu{probe}`, specifically each cluster's `trial` (1-based trial number of each spike), `trialtm` (spike time relative to that trial's start) and `quality` (manual curation label). The go cue times `obj.bp.ev.goCue` provide the alignment. The probe(s) used per session come from the hard-coded `SESSION_META`. Nothing else (no waveforms, no `clu.tm`, no `obj.psth`) is read.

ii.
```python
    def get_spike_data(self, probe_idx, cluster_idx):
        """Get trial and trialtm for a cluster. Returns (trial_1indexed, trialtm)."""
        if self.is_hdf5:
            clu = self.f[self.f['obj/clu'][probe_idx, 0]]
            trial   = np.array(self.f[clu['trial'][cluster_idx, 0]]).flatten()
            trialtm = np.array(self.f[clu['trialtm'][cluster_idx, 0]]).flatten()
            return trial, trialtm
```
```python
    go_cue = sd.get_event('goCue')
```

iii. Step 1 of CONVERSION_NOTES identifies `alignSpikes.m` ("trialtm_aligned = trialtm - ev.goCue(trial)") and `getSeq.m` ("bins at dt resolution, smooths with causal gaussian") as the reference functions, and Step 2 lists the `clu` fields ("quality, site, spkWavs, tm, trial, trialtm"). The AI reproduces exactly the two fields those functions use.

## 2-b. How is the `neural` data processed?

i. For each kept cluster and each trial independently: spikes are histogrammed into the 1000 fixed 5 ms bins spanning [−2.5, +2.5] s from the go cue, divided by the bin width to give spikes/s, and then smoothed along time with a **causal** Gaussian kernel that reproduces the reference's `mySmooth.m`: `gausswin(15)` (MATLAB alpha = 2.5, so std = (15−1)/(2·2.5) = 2.8 samples = 14 ms), with its first `floor(N/2)` taps zeroed to make it causal, normalised to sum 1, applied by `np.convolve(..., 'same')` after reflect-padding the first 15 samples and trimming them off afterwards. No normalisation, baseline subtraction or z-scoring is applied — values are firing rates in Hz, stored `float32`. Units from both probes of a two-probe session are concatenated. Per-trial arrays are transposed to `(n_neurons, 1000)` at output.

ii.
```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    if bctype == 'reflect':
        x_filt = np.concatenate([x[:N, :], x], axis=0); trim = N
    ...
    kern = sig_windows.gaussian(N, std=(N-1)/(2*2.5))
    kern[:N//2] = 0  # Make causal
    kern = kern / kern.sum()
    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
    out = out[trim:, :]
```
```python
        for t in range(n_trials):
            spk_mask = trial_int == (t + 1)
            if not np.any(spk_mask):
                continue
            spk_times = trialtm_aligned[spk_mask]
            N_counts, _ = np.histogram(spk_times, bins=EDGES)
            fr = N_counts[:n_time].astype(np.float64) / dt
            trialdat[:, neuron_idx, t] = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
```

iii. Step 1: "getSeq: bins at dt resolution, smooths with causal gaussian (15-sample window)". Step 10 Check 3: "Binning: Matches getSeq.m — histogram with edges tmin:dt:tmax, divide by dt for FR. Smoothing: Matches mySmooth.m — causal gaussian kernel with window N=15, reflect boundary." The AI read `mySmooth.m` in the trajectory and reproduced the `gausswin`/causal-zeroing/reflect-pad/trim logic line-for-line.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. **(1) Manual curation label:** `clu.quality`, lower-cased and stripped (also stripping `\x00`), is rejected if it is one of `garbage`, `gabrga`, `noisy`, `real?` — the exact drop list of the reference's `findClusters.m` under `params.quality = {'all'}`. The AI additionally rejects clusters whose quality string is **empty** (`q_clean == ''`), which `findClusters.m` does *not* do; in practice no cluster in the 45 sessions carries an empty label, so this has no effect. **(2) Firing rate:** after rates are computed, any unit whose mean rate over all bins **and all trials** (including early-lick/stim/ignore trials, which is what `removeLowFRClusters.m` does) is `<= 1.0 Hz` is dropped. Applied per probe before concatenation. Result: 2,398 units from 10,380 clusters, 17–141 per session (mean 55.8).

ii.
```python
PARAMS = {..., 'lowFR': 1.0, 'quality_exclude': ['garbage', 'gabrga', 'noisy', 'real?'], ...}

def find_clusters(qualities, exclude_list):
    exclude_lower = [e.lower() for e in exclude_list]
    indices = []
    for i, q in enumerate(qualities):
        q_clean = q.lower().strip().replace('\x00', '')
        if q_clean == '' or q_clean in exclude_lower:
            continue
        indices.append(i)
    return np.array(indices, dtype=int)
```
```python
        mean_frs = np.mean(trialdat, axis=(0, 2))
        fr_mask = mean_frs > PARAMS['lowFR']
        trialdat = trialdat[:, fr_mask, :]
```

iii. Step 3: "Cluster quality: all (excl garbage, noisy, real?) — Code: params.quality = {'all'}"; "Low FR threshold: 1 Hz — Paper: 'firing rates exceeding 1 Hz'". Step 4 records the conflict "lowFR: getDefaultParams 0.5 vs analysis scripts 1.0 vs paper 1 Hz — Resolution: use 1.0 Hz (matches paper)". Step 10 Check 3: "Neuron filtering: Matches findClusters.m ... and removeLowFRClusters.m (FR > 1 Hz)".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A single subtraction per spike. `clu.trialtm` is already on the behaviour clock and already relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm_aligned = trialtm − goCue[trial−1]` puts every spike in seconds from go cue onset. No interpolation or extra offset is applied to the neural stream (unlike the camera streams, which need the bitcode correction). Spikes whose trial number falls outside `[1, Ntrials]` are dropped first; spikes outside the [−2.5, 2.5] s window simply fall outside the histogram edges.

ii.
```python
        trial_int = trial.astype(int)
        valid_mask = (trial_int >= 1) & (trial_int <= n_trials)
        trial_int = trial_int[valid_mask]
        trialtm = trialtm[valid_mask]
        trialtm_aligned = trialtm - go_cue[trial_int - 1]
```

iii. Step 1: "alignSpikes: trialtm_aligned = trialtm - ev.goCue(trial)"; Step 10 Check 3: "Temporal alignment: Matches alignSpikes.m — subtract goCue time from trialtm". The AI also verified (trajectory step 46/47) that `goCue` is non-zero on water-cued trials too, so the same alignment applies in both contexts.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms (`dt = 1/200`), 1000 non-overlapping bins spanning −2.5 to +2.5 s around the go cue. The edge grid `EDGES` and the bin-centre axis `TIME_AXIS` are built once at module level and shared by every trial, session and data stream (neural, input, and all three video outputs), so no rebinning or resampling between streams is needed. Spikes are binned directly at 5 ms (not binned finer and rebinned); the video streams (~400 Hz) are interpolated onto the same 5 ms centres. `metadata['time_bin_size'] = 5.0` ms, `off_start = -2.5`, `off_end = 2.5`.

ii.
```python
PARAMS = {'alignEvent': 'goCue', 'tmin': -2.5, 'tmax': 2.5, 'dt': 1/200, ...}
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
N_TIMEPOINTS = len(TIME_AXIS)
```

iii. Step 3: "Neural data time bin: 5ms (1/200) — Code: params.dt = 1/200". Step 4 resolves a conflict in the reference code: "dt: getDefaultParams 1/200; WorkingWithDataObjs 1/100 — Use 1/200 (matches analysis scripts)". The window is taken from `params.tmin`/`params.tmax`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from any raw variable — it is defined by the conversion as the centre of each of the 1000 bins of the analysis window, i.e. exactly the time base on which the spikes are histogrammed. The go cue itself (`bp.ev.goCue`) enters only through the alignment of the other streams; the input vector is identical for every trial and every session, running from −2.4975 to +2.4975 s.

ii.
```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```
```python
        input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. Step 5 variable mapping: "time from goCue -> input[0] -> Continuous time axis [-2.5, 2.5]". Step 10 Check 3: "Input construction: Time from go cue as continuous variable"; Check 2 sanity check 3: "Input data: Time axis matches expected range [-2.4975, 2.4975] with 1000 timepoints at 5ms bins".

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond the construction in 3-a: build the edge grid with `np.arange`, take the midpoints, cast to `float32`, and reshape to `(1, 1000)` for each trial. The array is rebuilt (a fresh reshape/astype) for every trial rather than being shared, but the values are identical.

ii.
```python
        input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))
```

iii. No separate justification is given in CONVERSION_NOTES; it follows directly from the decoder-input specification ("Time from go cue onset in seconds (continuous, time-varying)").

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. By construction it *is* the neural binning grid: `EDGES` defines the spike histogram bins and `TIME_AXIS = EDGES[:-1] + dt/2` are the centres of those same bins, so input sample *k* and neural bin *k* denote the same 5 ms interval relative to go cue onset. The same `TIME_AXIS` is also the target grid onto which the tongue, paw and motion-energy streams are interpolated (via `taxis = TIME_AXIS + PARAMS['advance_movement']`, with `advance_movement = 0`), so all six outputs, the input, and the neural data share one time axis.

ii.
```python
            N_counts, _ = np.histogram(spk_times, bins=EDGES)
```
```python
    taxis = TIME_AXIS + PARAMS['advance_movement']
    ...
            x_interp = np.interp(taxis, aligned_ft, x)
```

iii. Implicit; Step 10 Check 3 lists binning and input construction as matching `getSeq.m` (`obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1)`), which is the same bin-centre definition the reference uses for its own `obj.time` axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields of `obj.bp`: the instructed side flags `L` and `R`, and the outcome flags `hit`, `miss` and `no`. The lick direction is not recorded directly, so it is inferred from the instructed side combined with whether the animal was correct.

ii.
```python
    L = sd.get_bp_field('L')
    R = sd.get_bp_field('R')
    hit = sd.get_bp_field('hit')
    miss = sd.get_bp_field('miss')
    no_resp = sd.get_bp_field('no')
```

iii. Trajectory step 50: "for hit trials the animal licked correctly (L trial -> left lick, R trial -> right lick), for miss trials the animal licked incorrectly (wrong direction), and for no/ignore trials there was no response". Step 5 mapping: "L/R + hit/miss -> output[0]: lick_direction -> 0=left, 1=right, 2=none — Per-trial".

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A four-way relabelling into codes 0 = left, 1 = right, 2 = none: hit + L → left; hit + R → right; miss + L → right (the animal licked the wrong, i.e. right, port); miss + R → left; `no` → none. The value is per-trial and is broadcast (tiled) across all 1000 time bins in the output array. Trials matching none of these (there are none) would retain the sentinel −1.

**Consequence of the trial filter (1-e):** because `no == 1` trials are removed before assembly, class 2 ("none") never occurs in the saved dataset — the verification log reports `lick_direction: {left (0.487), right (0.513)}` and a per-session range of `[0, 1]` for every session, although `output_values[0]` still declares three classes. The decoder therefore reports "chance = 0.333" for what is really a two-class problem (validation balanced accuracy 0.648).

ii.
```python
    lick_dir = np.full(n_trials_total, -1, dtype=np.int64)
    lick_dir[(hit == 1) & (L == 1)] = 0  # left
    lick_dir[(hit == 1) & (R == 1)] = 1  # right
    lick_dir[(miss == 1) & (L == 1)] = 1  # licked right (wrong)
    lick_dir[(miss == 1) & (R == 1)] = 0  # licked left (wrong)
    lick_dir[no_resp == 1] = 2  # none
```
```python
        output_trial[0, :] = lick_dir[t_idx]
```

iii. Step 5 mapping (see 4-a). Step 10 Check 2 sanity check 2: "Output data: Verified lick_direction, context, outcome for 5 trials in session 0 — all match expected values from raw behavioral data". No note is made anywhere that the "none" class is empty in the final dataset.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. A single per-trial field, `obj.bp.autowater`, which flags the water-cued (WC) trials in which water is delivered at a random port with no auditory cues; all other trials are delayed-response (DR).

ii.
```python
    autowater = sd.get_bp_field('autowater')
```

iii. Step 5 mapping: "autowater -> output[1]: context -> 0=DR, 1=WC — Per-trial". Trajectory step 46 confirms the field's semantics from the data ("261 trials total: 181 DR, 80 WC").

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct cast of the flag to integer: `context = autowater.astype(int)`, so 0 = DR and 1 = WC. This is the **opposite** numeric coding from the prompt's listing order ("WC, DR"), but `output_values[1] = ['DR', 'WC']` is declared consistently, so the labels are correct. The value is per-trial and tiled across the 1000 bins. Resulting distribution: 92.6 % DR / 7.4 % WC, with 16 of 43 sessions containing no WC trials at all.

ii.
```python
    context = autowater.astype(np.int64)  # 0=DR, 1=WC
```
```python
        output_trial[1, :] = context[t_idx]
```
```python
        'output_values': [ ..., ['DR', 'WC'], ... ],
```

iii. Step 5 mapping as above. Step 10 Check 4: "Context distribution: 92.6% DR, 7.4% WC (many sessions are DR-only)".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three mutually exclusive per-trial flags of `obj.bp`: `hit`, `miss` and `no`. Unlike the reference, the AI reads `no` explicitly rather than treating "neither hit nor miss" as the ignore class.

ii.
```python
    hit = sd.get_bp_field('hit')
    miss = sd.get_bp_field('miss')
    no_resp = sd.get_bp_field('no')
```

iii. Trajectory step 47: "The `no` field indicates ignore trials (no response)". Trajectory step 20 (from methods): "Ignore trials: no response within 3s of go cue". Step 5 mapping: "hit/miss/no -> output[2]: outcome -> 0=incorrect, 1=correct, 2=ignore".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling: `miss` → 0 (incorrect), `hit` → 1 (correct), `no` → 2 (ignore), matching the prompt's ordering exactly. Per-trial, tiled across all 1000 bins.

**Consequence of the trial filter (1-e):** ignore trials are removed before assembly, so class 2 never occurs. Verification reports `outcome: {incorrect (0.136), correct (0.864)}` and a range of `[0, 1]` in every session, while `output_values[2]` still declares three classes; reported chance (0.333) is again wrong for the data that is actually saved (validation balanced accuracy 0.637).

ii.
```python
    outcome = np.full(n_trials_total, -1, dtype=np.int64)
    outcome[hit == 1] = 1   # correct
    outcome[miss == 1] = 0  # incorrect
    outcome[no_resp == 1] = 2  # ignore
```
```python
        output_trial[2, :] = outcome[t_idx]
```

iii. Step 5 mapping as above; Step 9 "Correct rate ~80–90 % (paper) vs 86.4 % (converted) — consistent". No note that the ignore class is empty.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` (camera index 0, the side view — the AI's notes mislabel it as the "bottom view", but the index and the feature name it selects are the side camera's), feature `'tongue'`, channels x and y of `traj(trial).ts`. Only this one feature from one camera is used; the bottom camera's `top_tongue`/`bottom_tongue` are read only to know they exist. `traj(trial).frameTimes`, `obj.sglx.bitcode.bitstart`, `obj.sglx.fs` and `bp.ev.bitStart` are also needed, for the clock correction (7-d).

ii.
```python
    tongue_speed, tongue_visible = extract_feature_velocity(sd, 0, 'tongue', go_cue, n_trials_total, is_tongue=True)
```
```python
    feat_names = sd.get_traj_feat_names(cam_idx)
    feat_idx = None
    for fi, fn in enumerate(feat_names):
        if fn == feat_name:
            feat_idx = fi
            break
    ...
            x = ts[:, 0, feat_idx].astype(float)
            y = ts[:, 1, feat_idx].astype(float)
```

iii. Trajectory step 26 enumerates the camera feature lists. Step 5 Key Decision 4: "Tongue velocity: From camera 0 'tongue' feature. Speed = sqrt(xvel^2 + yvel^2). Tongue not visible -> category 2." No justification is offered for using one view rather than both.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The pipeline reproduces the reference's `findPosition.m` / `findVelocity.m` almost exactly, including their tongue-specific branches:
1. `frameTimes` are put on the go-cue clock (7-d) and x, y are linearly interpolated from the ~400 Hz frames onto the 1000 5 ms bin centres with `np.interp`. Because `np.interp` clamps instead of extrapolating, bins outside the trial's frame coverage are explicitly set to NaN.
2. **No positional smoothing for the tongue** — matching `findPosition.m`, which applies `mySmooth` only to non-tongue features.
3. Visibility is recorded as "x and y are both non-NaN after interpolation". DeepLabCut writes NaN wherever likelihood is low, so this is the authors' own visibility criterion propagated through the interpolation.
4. `xvel = np.gradient(x_interp)`, `yvel = np.gradient(y_interp)` — per-sample differences on the 5 ms grid, **not divided by dt**, so the units are pixels/bin rather than pixels/s. No baseline-derivative subtraction (matching `findVelocity.m`'s `~contains(feat,'tongue')` guard). Non-visible samples are set to 0 (again matching `findVelocity.m`: "set tongue velocity to 0 if not visible").
5. Speed = `sqrt(xvel**2 + yvel**2)`.

Trials with `NdroppedFrames` NaN, all-NaN `frameTimes`, fewer than 10 frames, or any exception are skipped entirely and stay NaN/not-visible for all 1000 bins.

ii.
```python
            aligned_ft = ft - vidshift - go_cue[t]
            x_interp = np.interp(taxis, aligned_ft, x)
            y_interp = np.interp(taxis, aligned_ft, y)
            outside = (taxis < aligned_ft[0]) | (taxis > aligned_ft[-1])
            x_interp[outside] = np.nan
            y_interp[outside] = np.nan

            if is_tongue:
                vis = ~np.isnan(x_interp) & ~np.isnan(y_interp)
                visible[:, t] = vis
                xvel = np.gradient(x_interp); yvel = np.gradient(y_interp)
                xvel[~vis] = 0; yvel[~vis] = 0
            ...
            spd = np.sqrt(xvel**2 + yvel**2)
            speed[:, t] = spd
```

iii. Step 1: "findPosition: Interpolate DLC tracking to neural time axis; findVelocity: Compute velocity from position (gradient)". Step 5 Key Decision 4 (above). The units issue (px/bin vs px/s) is never raised, but it is a constant scale factor and therefore harmless under a percentile threshold.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, the 50th percentile of the speed is computed **over the visible, non-NaN bins only**, pooled across all trials and all time bins of that session. Bins at or above it get class 1, bins below get class 0, and every bin where the tongue was not visible (or the trial was skipped) gets class 2 regardless of speed. Result across the dataset: 3.3 % low / 3.5 % high / 93.2 % not visible.

ii.
```python
def discretize_velocity(speed, visible, percentile_thresh=50):
    categories = np.full((n_time, n_trials), 2, dtype=np.int64)
    vis_mask = visible & ~np.isnan(speed)
    if np.sum(vis_mask) > 0:
        threshold = np.percentile(speed[vis_mask], percentile_thresh)
        categories[vis_mask & (speed < threshold)] = 0
        categories[vis_mask & (speed >= threshold)] = 1
    return categories
```
```python
    tongue_cat = discretize_velocity(tongue_speed, tongue_visible, 50)
```

iii. Step 5 mapping: "tongue DLC tracking -> output[3] -> Discretized per-session 50th percentile — Time-varying"; this is a direct implementation of the prompt's Decoder Task specification ("0: < 50th percentile, 1: >= 50th percentile, 2: not visible").

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Two corrections, then interpolation onto the shared grid. The camera clock leads the behaviour clock, and the offset is recovered once per session from the bitcode pulse recorded on both streams: `mode(obj.sglx.bitcode.bitstart) / obj.sglx.fs − mode(bp.ev.bitStart)` — the reference's `findVideoOffset.m`. Frame time in seconds from go cue is then `frameTimes − vidshift − goCue[trial]`, and x/y are interpolated onto `TIME_AXIS` (with `advance_movement = 0`), which is the identical grid the spikes are binned on. I verified the offset resolves to 0.49 s for 41 sessions and 0.99 s for the four JEB19 sessions, and never falls back to the `except: return 0.0` branch.

ii.
```python
    def get_video_offset(self):
        try:
            bitStart = self.get_event('bitStart')
            bitstart_mode = float(scipy_stats.mode(bitStart, keepdims=False).mode)
            sglx_bitstart = np.array(self.f['obj/sglx/bitcode/bitstart']).flatten()
            sglx_fs = np.array(self.f['obj/sglx/fs']).flatten()[0]
            sglx_bitstart_mode = float(scipy_stats.mode(sglx_bitstart, keepdims=False).mode)
            return sglx_bitstart_mode / sglx_fs - bitstart_mode
        except:
            return 0.0
```
```python
            aligned_ft = ft - vidshift - go_cue[t]
            x_interp = np.interp(taxis, aligned_ft, x)
```

iii. Step 1: "findVideoOffset: Align video to neural recording"; "Video offset = mode(sglx.bitcode.bitstart)/sglx.fs - mode(bp.ev.bitStart)". Step 5 Key Decision 4 and Step 1 notes ("DLC tracking interpolated to neural time axis" via `findPosition.m`'s `interp1(traj.frameTimes - vidshift - obj.bp.ev.(alignEv)(trix), ts, taxis)`).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` DeepLabCut tracking, but camera index 1 (the bottom view) and the feature **`bottom_paw`**, x and y channels. The bottom view also tracks `top_paw`, which is not used.

ii.
```python
    paw_speed, paw_visible = extract_feature_velocity(sd, 1, 'bottom_paw', go_cue, n_trials_total, is_tongue=False)
```

iii. Step 5 Key Decision 5: "Paw velocity: From camera 1 'bottom_paw' feature. Speed = sqrt(xvel^2 + yvel^2). Paw not visible -> category 2." No reason is given for `bottom_paw` over `top_paw`; the reference code's `params.traj_features` lists both, and nothing in the notes or trajectory shows the two being compared.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same interpolation-then-gradient pipeline as the tongue, but taking the reference's **non-tongue** branches of `findPosition.m`/`findVelocity.m`:
1. `frameTimes` corrected by the video offset and the trial's go cue; x, y linearly interpolated onto the 1000 bin centres; bins outside the trial's frame coverage set to NaN.
2. Visibility recorded from the post-interpolation NaN mask **before** any filling.
3. `xvel`/`yvel` = `np.gradient` of the interpolated positions (units px/bin), then the session-independent baseline drift `np.nanmedian(np.diff(x_interp))` / `np.nanmedian(np.diff(y_interp))` is subtracted from each axis — the reference's `xvel = xvel - basederiv(1)` step for non-tongue features (the AI subtracts each axis's own median rather than reusing the x median, as the reference does).
4. Remaining NaNs in the velocity are filled by linear interpolation over sample index, which clamps at the edges — the equivalent of the reference's `fillmissing(...,'nearest')`.
5. Speed = `sqrt(xvel**2 + yvel**2)`.
The positional smoothing the reference applies (`mySmooth(ts, 1, 'reflect')`) is a no-op at N = 1, so omitting it is faithful.

ii.
```python
            else:
                vis = ~np.isnan(x_interp) & ~np.isnan(y_interp)
                visible[:, t] = vis
                xvel = np.gradient(x_interp); yvel = np.gradient(y_interp)
                basederiv_x = np.nanmedian(np.diff(x_interp))
                basederiv_y = np.nanmedian(np.diff(y_interp))
                if not np.isnan(basederiv_x): xvel -= basederiv_x
                if not np.isnan(basederiv_y): yvel -= basederiv_y
                for arr in [xvel, yvel]:
                    mask_nan = np.isnan(arr)
                    if np.any(mask_nan) and not np.all(mask_nan):
                        valid = ~mask_nan
                        arr[mask_nan] = np.interp(np.where(mask_nan)[0], np.where(valid)[0], arr[valid])
```

iii. Step 1: "findVelocity: Compute velocity from position (gradient)". Step 5 Key Decision 5. The fill-then-mask ordering (so that filled bins still report "not visible") is not explicitly discussed in the notes.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: `discretize_velocity(paw_speed, paw_visible, 50)` — the session's 50th percentile over visible, non-NaN bins pooled across trials and time; below → 0, at/above → 1, not visible → 2. Dataset totals: 40.9 % low / 41.8 % high / 17.3 % not visible, but the per-session not-visible fraction ranges from 1.2 % to **94.0 %** (the four JEB15 sessions are 61 %, 94 %, 75 %, 82 %, 75 % not visible; EKH1 is 58 %).

ii.
```python
    paw_cat = discretize_velocity(paw_speed, paw_visible, 50)
```

iii. Step 5 mapping: "paw DLC tracking -> output[4] -> Discretized per-session 50th percentile". Directly implements the prompt's Decoder Task spec. The extreme per-session not-visible fractions are never examined in CONVERSION_NOTES.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as the tongue (7-d), but using the **bottom camera's own** `frameTimes`: `aligned_ft = frameTimes − vidshift − goCue[trial]`, then `np.interp` onto `TIME_AXIS`. The video offset is a session constant recomputed inside each `extract_feature_velocity` call. Because each feature is timed by the camera it comes from, a mismatch in frame counts between the two views does not cause a misalignment.

ii.
```python
    vidshift = sd.get_video_offset()
    for t in range(n_trials):
            ts, ft, ndf = sd.get_traj_trial(cam_idx, t)   # cam_idx = 1 for the paw
            ...
            aligned_ft = ft - vidshift - go_cue[t]
            x_interp = np.interp(taxis, aligned_ft, x)
            y_interp = np.interp(taxis, aligned_ft, y)
```

iii. Same as 7-d: Step 1 "findVideoOffset ... Video: 2 cameras at ~400Hz, DLC tracking interpolated to neural time axis".

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file next to the data structure, field `me.data` — one trace per trial with one value per camera frame. `me.moveThresh` is also read but never used. The in-object copy `obj.me` is not used. Frame times come from camera 0 of `obj.traj`.

Only **one** of the three on-disk layouts is handled. The loader assumes `me` is a struct with both `data` and `moveThresh`, and any failure is swallowed by a bare `except: return None, None`. I verified against the raw files that this breaks for 7 of the 43 retained sessions: `me` is a **bare cell array** (no field names) in JEB23 2023-10-10/11/12/13, and **doubly wrapped** (`me.data` is itself a struct with `data`/`moveThresh`) in JEB15 2022-07-26, JEB15 2022-07-28 and JEB24 2023-10-31. In all seven the motion-energy output is class 2 ("no video") for 100 % of bins — visible in `verification_full_out.txt` as sessions 15, 17, 21, 22, 23, 24, 34 having range `[2.0, 2.0]`.

ii.
```python
    def get_motion_energy_data(self, me_file):
        if not os.path.exists(me_file):
            return None, None
        try:
            me_mat = scipy.io.loadmat(me_file, squeeze_me=False)
            me_struct = me_mat['me']
            me_data_raw = me_struct['data'][0, 0]
            me_thresh = me_struct['moveThresh'][0, 0].flatten()[0]
            me_data = []
            for i in range(me_data_raw.shape[0]):
                d = me_data_raw[i, 0]
                me_data.append(d.flatten() if isinstance(d, np.ndarray) else np.array([]))
            return me_data, float(me_thresh)
        except:
            return None, None
```

iii. Step 1: "loadMotionEnergy: Load and align motion energy". Step 5 Key Decision 6: "Motion energy: From separate files, aligned via video frame times. No ME file -> category 2." Step 10 Check 5: "Handled sessions with missing motion energy data (set to category 2)" — the notes treat the seven sessions as having no data, when in fact the files are present and readable with a more permissive reader.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment and resampling — the value is already one scalar per frame. Per trial: the trace and the camera-0 aligned frame times are truncated to a common length, then `np.interp` maps the trace onto the 1000 bin centres. Two fallbacks then run: (a) if a trial's frame times are unusable, frame times are **fabricated** as `np.arange(len(me_trial)) / 400.0` with a hard-coded 0.5 s offset (`aligned_ft = ft - 0.5 - go_cue[t]`); (b) after all trials, any remaining NaN in a trial column is filled by linear interpolation over index (clamping at the edges). Consequence of (b): within a session that has motion energy, no bin is ever marked class 2 — bins outside the video's coverage are filled instead of marked missing, unlike the paw and tongue streams.

ii.
```python
            ts, ft, ndf = sd.get_traj_trial(0, t)  # Camera 0 for frame times
            if ft is None or len(ft) < 10:
                ft = np.arange(len(me_trial)) / 400.0
                aligned_ft = ft - 0.5 - go_cue[t]  # Fallback alignment
            else:
                aligned_ft = ft - vidshift - go_cue[t]
            if len(me_trial) > len(aligned_ft):
                me_trial = me_trial[:len(aligned_ft)]
            elif len(me_trial) < len(aligned_ft):
                aligned_ft = aligned_ft[:len(me_trial)]
            me_aligned[:, t] = np.interp(taxis, aligned_ft, me_trial)
    # Fill NaN with nearest
    for t in range(n_trials):
        col = me_aligned[:, t]; mask = np.isnan(col)
        if np.any(mask) and not np.all(mask):
            col[mask] = np.interp(np.where(mask)[0], np.where(~mask)[0], col[~mask])
            me_aligned[:, t] = col
```

iii. Step 5 Key Decision 6 (above). The "create frame times from length" fallback mirrors `findPosition.m`'s `traj(trix).frameTimes = (1:size(traj(trix).ts,1)) ./ 400` guard, though the reference does not add a 0.5 s constant. The nearest-fill mirrors the reference's `fillmissing(...,'nearest')` for non-tongue features.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per session, the 50th percentile over all non-NaN bins (pooled across trials and time), below → 0, at/above → 1. Class 2 ("no video") is assigned to every bin of a session for which the motion-energy file could not be parsed (`has_video = False`), and to any bin still NaN after the fill. Dataset totals: 38.4 % low / 44.4 % high / 17.2 % no video — but the 17.2 % is almost entirely the seven whole sessions lost in 9-a, not genuine per-bin gaps (only one session, index 27, has a partial value, 7.5 %).

ii.
```python
def discretize_motion_energy(me_data, has_video, percentile_thresh=50):
    if not has_video:
        return np.full((n_time, n_trials), 2, dtype=np.int64)
    categories = np.full((n_time, n_trials), 2, dtype=np.int64)
    valid_mask = ~np.isnan(me_data)
    if np.sum(valid_mask) > 0:
        threshold = np.percentile(me_data[valid_mask], percentile_thresh)
        categories[valid_mask & (me_data < threshold)] = 0
        categories[valid_mask & (me_data >= threshold)] = 1
    return categories
```

iii. Step 5 mapping: "motion energy -> output[5] -> Discretized per-session 50th percentile"; direct implementation of the prompt's spec ("2: no video"). The reference's own `me.moveThresh` is deliberately not used, since the prompt requires a 50th-percentile split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy has one value per frame of camera 0, so it inherits that camera's `frameTimes`, corrected by the same session-constant video offset and the trial's go cue, and is then interpolated onto the shared `TIME_AXIS`. Length mismatches between the trace and the frame-time vector are resolved by truncating the longer of the two.

ii.
```python
    vidshift = sd.get_video_offset()
    for t in range(min(n_trials, len(me_data))):
            ts, ft, ndf = sd.get_traj_trial(0, t)  # Camera 0 for frame times
            aligned_ft = ft - vidshift - go_cue[t]
            ...
            me_aligned[:, t] = np.interp(taxis, aligned_ft, me_trial)
```

iii. Step 5 Key Decision 6: "aligned via video frame times". Same `findVideoOffset.m` justification as 7-d.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Almost entirely through defensive `try/except` blocks that skip the offending unit of data and leave a "not visible / no video" marker, with no logging:
- **Two MATLAB file formats**: dispatched by `try h5py / except scipy.io` in `SessionData._open`, and every accessor has an `is_hdf5` branch.
- **Bad video trials**: `NdroppedFrames` NaN, all-NaN `frameTimes`, unexpected `ts` shape, or < 10 frames → `get_traj_trial` returns `(None, None, None)` and the trial's tongue/paw stay NaN → class 2 for all 1000 bins.
- **DLC low-confidence frames**: already NaN in the raw data; the NaN survives interpolation and drives the `visible` mask → class 2 for the tongue, and class 2 plus a nearest-filled velocity for the paw.
- **Bins outside the video's coverage**: explicitly NaN-ed (because `np.interp` clamps) → class 2 for tongue/paw, but *filled* for motion energy.
- **Missing motion energy**: `get_motion_energy_data` returns `(None, None)` → whole session class 2.
- **Missing `ts` shape conventions**: `ts` is transposed to `(frames, 3, features)` when its first axis does not match `len(frameTimes)`.
- **Spikes with out-of-range trial numbers**: masked out before alignment.
- **Sessions after the recording ends**: *not* handled — 30 trials with all-zero firing across every neuron remain in the dataset and are reported as warnings by the verifier.
- **Per-trial fields longer than `Ntrials`**: not handled — `get_bp_field` returns the full flattened array without truncating to `Ntrials` (it happens not to bite on these files).

The pervasive bare `except:` clauses are the weak point: they turn parse bugs into silently missing data, which is exactly what happened with motion energy (9-a).

ii.
```python
            try:
                ndf_ref = cam['NdroppedFrames'][trial_idx, 0]
                ndf = np.array(self.f[ndf_ref]).flatten()
                if np.any(np.isnan(ndf)):
                    return None, None, None
            except:
                pass
```
```python
        except Exception:
            continue      # in extract_feature_velocity's per-trial loop
```
```python
        except:
            return None, None     # in get_motion_energy_data
```

iii. Step 10 Check 5: "Handled both HDF5 and MATLAB v5 file formats; Handled sessions with missing motion energy data (set to category 2); Handled trials with no video data (tongue/paw set to not_visible); Handled dual-probe sessions (JEB15)". Step 10 Check 1 on the all-zero trials: "These are trials that occurred after the neural recording ended... Cannot be fixed without excluding these trials, but they are valid behavioral trials."

## 11-a. What are the most time-consuming steps of the code?

i. The script instruments itself and prints per-stage timings. Across the 45-session full run (163.8 s total, ~3.6 s/session):
- **Video feature extraction dominates**: 2.1–3.9 s per session (roughly 60–70 % of session time). This is three passes over every trial (tongue on camera 0, paw on camera 1, motion energy re-reading camera 0), each dereferencing HDF5 object references per trial.
- **Neural rate computation**: 0.5–1.9 s per session. Inside it, the dominant cost is the `(neuron × trial)` double loop, which calls `np.histogram` and a Python-level `causal_gaussian_smooth` (with its own per-column `np.convolve` loop) once per neuron per trial — roughly 55 neurons × 300 trials ≈ 16,500 tiny convolutions per session.
- **File opening** is cheap for the 31 v7.3 sessions (h5py is lazy) but eager for the 14 v5 sessions (`scipy.io.loadmat` reads the whole struct).
- **Pickling** the 3.34 GB result at the end.

ii.
```python
        t_neural = time.time()
        trialdat = compute_firing_rates(sd, probe_idx, quality_indices, go_cue, n_trials_total)
        print(f'    Neural data computed in {time.time()-t_neural:.1f}s')
    ...
    t_video = time.time()
    tongue_speed, tongue_visible = extract_feature_velocity(sd, 0, 'tongue', ...)
    paw_speed, paw_visible = extract_feature_velocity(sd, 1, 'bottom_paw', ...)
    me_aligned, has_me = extract_motion_energy(sd, me_file, go_cue, n_trials_total)
    print(f'    Video data extracted in {time.time()-t_video:.1f}s')
```

iii. Step 7 "Run Time Estimates" gives "Full conversion ~3.7 s/session, ~167 s (~2.8 min)" from the two-session sample — an estimate that proved accurate (163.8 s actual). Since the estimate was comfortably under the 15-minute threshold in the instructions, no optimisation work was done; Step 6 records no "code speedups added".

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several, the clearest being in the neural path:
- **`compute_firing_rates`'s per-trial loop.** For every neuron it re-scans the whole spike vector once per trial (`trial_int == (t + 1)` is O(n_spikes) inside an O(n_trials) loop, i.e. quadratic in session length). A single `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, EDGES])` would produce the entire `(trials × bins)` count matrix in one call — which is exactly what the reference solution does.
- **The smoothing call site.** `causal_gaussian_smooth` already accepts a 2-D array and loops over columns, yet it is called with a single 1-D trial at a time. Passing the whole `(1000, n_trials)` matrix once per neuron would remove ~300 Python-level calls per neuron; using `scipy.ndimage.convolve1d`/`gaussian_filter1d` along an axis would remove the column loop too.
- **`extract_feature_velocity` and `extract_motion_energy` per-trial loops.** These are harder to vectorise because each trial has a different number of frames, but the three calls could at least share one pass over the trials instead of three.
- **The final `for t_idx in valid_trial_indices` loop** rebuilds `TIME_AXIS.reshape(1, -1).astype(np.float32)` and a fresh `(6, 1000)` int64 array per trial; a single `np.tile`/slicing would do.

ii.
```python
        for t in range(n_trials):
            spk_mask = trial_int == (t + 1)          # O(n_spikes) per trial
            ...
            N_counts, _ = np.histogram(spk_times, bins=EDGES)
            fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
            trialdat[:, neuron_idx, t] = fr_smooth
```
```python
    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):                 # column loop, always 1 column here
        out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```

iii. The instructions asked for vectorised code and timing information; the AI provided the timing but Step 6 of CONVERSION_NOTES records no inefficiencies identified and no speed-ups implemented, on the grounds (Step 7) that the projected ~2.8 min runtime was already acceptable.

## 11-c. What processing does the code repeat multiple times?

i. Four repetitions, none of them affecting correctness:
- **The video offset** is a session constant but is recomputed three times per session — once inside each of the two `extract_feature_velocity` calls and once inside `extract_motion_energy` — each time re-reading `bp.ev.bitStart`, `sglx.bitcode.bitstart` and `sglx.fs` and running `scipy.stats.mode` twice.
- **Camera-0 trial data is read twice per trial**: once by the tongue extraction and again by `extract_motion_energy`, which calls `sd.get_traj_trial(0, t)` purely to obtain `frameTimes` (and discards the `ts` array it just dereferenced).
- **Feature names** are re-resolved per call via `get_traj_feat_names`, which itself loops over trials until one parses.
- **Firing rates are computed for every trial in the session**, including the ~24 % that the trial filter will discard (see 11-d).

ii.
```python
def extract_feature_velocity(...):
    ...
    vidshift = sd.get_video_offset()      # called once per feature -> 2x per session

def extract_motion_energy(...):
    ...
    vidshift = sd.get_video_offset()      # and a 3rd time here
    for t in ...:
            ts, ft, ndf = sd.get_traj_trial(0, t)  # camera 0 re-read; ts discarded
```

iii. Not discussed in CONVERSION_NOTES; the notes' only efficiency statement is the Step 7 runtime estimate.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Firing rates for filtered-out trials.** `compute_firing_rates` builds a `(1000, n_neurons, n_trials_total)` array over *all* trials, and the video extractors likewise fill `(1000, n_trials_total)` arrays, but only `valid_trial_indices` (11,981 of ~15,500, i.e. ~77 %) are written to the output. About a quarter of the binning, smoothing, interpolation and gradient work is thrown away. (The low-FR threshold is deliberately computed over all trials, matching `removeLowFRClusters.m`, so that part of the full-length computation is needed — but only the mean is.)
- **`me.moveThresh`** is parsed out of every motion-energy file and returned by `get_motion_energy_data`, then never used (the prompt mandates a 50th-percentile split instead).
- **`ts` in `extract_motion_energy`**: the full tracked-feature array for camera 0 is dereferenced and returned for every trial only to be discarded; only `frameTimes` is needed.
- **`NdroppedFrames`** is read, converted, and returned as `ndf`, and the returned value is never inspected (only the internal NaN check matters).
- **`L`** is read but partially redundant with `R`, and `no_resp` is used both to build the class-2 labels and to delete the very trials that would carry them — so the `lick_dir == 2` / `outcome == 2` assignments are computed and then guaranteed never to reach the output.
- **The `visible` array returned for motion energy's camera-0 call** and the `speed`/`visible` values on skipped trials are allocated but unused.

ii.
```python
    trialdat = np.zeros((n_time, n_neurons, n_trials), dtype=np.float32)   # all trials
    ...
    for t_idx in valid_trial_indices:                                       # ~77% kept
        neural_trials.append(trialdat[:, :, t_idx].T.astype(np.float32))
```
```python
            me_thresh = me_struct['moveThresh'][0, 0].flatten()[0]
            return me_data, float(me_thresh)      # me_thresh never used by the caller
```
```python
    lick_dir[no_resp == 1] = 2   # never survives `valid_trials = ... & (no_resp == 0) ...`
    outcome[no_resp == 1] = 2
```

iii. Not discussed in CONVERSION_NOTES.
