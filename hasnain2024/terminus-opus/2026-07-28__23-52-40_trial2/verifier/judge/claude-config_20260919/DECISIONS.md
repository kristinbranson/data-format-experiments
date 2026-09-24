# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of 25 sessions (`SESSION_META`, animal / date / probe list) transcribed from the authors' `load<ANM>_ALMVideo.m` scripts, and loads **only** the `data/Ephys_Behavior` folder (`DATA_DIR = 'data/Ephys_Behavior'`). The 19–22 sessions of `data/RandomizedDelay_Ephys_Behavior` (animals JEB11, JEB12, JEB23, JEB24) are deliberately not loaded, even though the AI discovered and listed them in Step 2. Each session is one `data_structure_<anm>_<date>.mat` file opened once with `h5py` (all Ephys_Behavior files are MATLAB v7.3); the matching `motionEnergy_<anm>_<date>.mat` is read separately with `scipy.io.loadmat`. A session is skipped if the file is missing, if it has fewer than 2 valid trials, or if it ends up with fewer than 10 neurons. The AI's probe table also disagrees with the human reference for four sessions (EKH1 probe 1 vs 2; EKH3 probes [1,2] vs [2]; JEB7 probe 2 vs 1; JEB15 probe 1 vs [1,2]/[2]).

ii.
```python
DATA_DIR = 'data/Ephys_Behavior'

# Session metadata: (animal, date, probe_numbers)
# From the loadXXX_ALMVideo.m scripts
SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ('EKH3', '2021-08-11', [1, 2]),  # dual probe
    ('JEB6', '2021-04-18', [2]),
    ...
    ('JGR3', '2021-11-18', [1]),
]
```
```python
    data_file = os.path.join(DATA_DIR, f'data_structure_{session_id}.mat')
    me_file = os.path.join(DATA_DIR, f'motionEnergy_{session_id}.mat')

    if not os.path.exists(data_file):
        print(f'  WARNING: Data file not found: {data_file}')
        return None
    ...
    f = h5py.File(data_file, 'r')
```
```python
    for idx, (animal, date, probes) in enumerate(sessions):
        result = process_session(animal, date, probes, ...)
```

iii. From CONVERSION_NOTES.md Step 5 ("Key Decisions 1. **Use Ephys_Behavior only**: Has DR+WC two-context task") and the trajectory: *"The task description says to use the data with DR+WC (two-context) task since we need behavioral context (WC=0, DR=1) as an output. The Ephys_Behavior dataset has both DR and WC trials."* The AI also noted that the paper reports "25 sessions using nine mice" for the DR task and treated matching that number (25 sessions, 1,653 units after quality filtering vs the paper's 1,651) as its consistency check. The probe table was read out of the `load<ANM>_ALMVideo.m` files.

## 1-b. How are the data split into subjects?

i. The animal ID is the first element of each `SESSION_META` tuple (equivalently the part of the filename before the underscore). It is carried through as `result['animal']`; at assembly `subjects` is the sorted set of unique animals and `subject_idx` is each session's index into that list. This yields 10 subjects over 25 sessions.

ii.
```python
    result = {
        ...
        'session_id': session_id,
        'animal': animal,
    }
```
```python
    subjects = sorted(list(set(r['animal'] for r in all_results)))
    ...
        subj_idx = subjects.index(result['animal'])
        subject_idx_list.append(subj_idx)
    ...
        'subjects': subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. The AI never reads an animal field from inside the `.mat` files; the identity comes from the file naming convention used by the authors' loading scripts. CONVERSION_NOTES Step 4 records the one discrepancy it found: "Subjects | 10 animals loaded | 10 animals | 9 mice | Data has 10; paper counts 9. Using all 10."

## 1-c. How are the data split into sessions?

i. One entry of `SESSION_META` = one file on disk = one element of `neural` / `input` / `output`. All 25 sessions come from the single fixed-delay folder; there is no attempt to merge the two task folders. Every session that processes successfully is appended to `all_results` in the order listed, and `subject_idx` / `brain_region_idx` follow that same order.

ii.
```python
    session_id = f'{animal}_{date}'
    data_file = os.path.join(DATA_DIR, f'data_structure_{session_id}.mat')
```
```python
        if result is not None:
            all_results.append(result)
```

iii. Same justification as 1-a: the AI restricted itself to the fixed-delay two-context dataset because behavioral context (WC/DR) is a required decoder output, and it checked the resulting session count (25) against the paper's "25 sessions".

## 1-d. How are the data split into trials?

i. Trials are the rows of the Bpod table `obj.bp`: `Ntrials` gives the count, and `L`, `R`, `hit`, `miss`, `no`, `autowater`, `early`, `stim.enable` and `ev.goCue` are read as flat per-trial vectors. Spikes carry a 1-based `trial` index, so per-trial spike sets are obtained by masking on `spike_trial == trial_num`; camera data (`traj.ts`, `traj.frameTimes`) and motion energy are stored per trial and indexed by `trix`. Neural arrays are built for all `Ntrials` and only sliced down to the kept trials at packaging time.

ii.
```python
    bp = f['obj']['bp']
    Ntrials = int(bp['Ntrials'][0, 0])

    L = bp['L'][:].flatten()  # left trials
    R = bp['R'][:].flatten()  # right trials
    hit = bp['hit'][:].flatten()  # correct trials
    ...
    goCue = bp['ev']['goCue'][:].flatten()
```
```python
            for trial_num in range(1, Ntrials + 1):
                trial_mask = spike_trial_valid == trial_num
                ...
                counts, _ = np.histogram(trial_spikes, bins=EDGES)
```

iii. Not explicitly justified beyond CONVERSION_NOTES Step 2 ("obj.bp: Behavioral data (L, R, hit, miss, no, autowater, early, stim, ev)") and the observation that `obj.clu.trial` gives a 1-indexed trial number per spike. Unlike the reference, the AI does not truncate the `bp` vectors to `Ntrials`; it relies on them already having that length.

## 1-e. How are trials filtered based on quality controls?

i. A single boolean mask keeps a trial only if it is **not** an early lick, **not** a no-response ("ignore") trial, **not** a photostimulation trial, and **is** either a hit or a miss. Ignore trials are therefore removed from the dataset entirely (≈13% of trials in the reference's accounting). 6,150 of 8,260 trials survive. A session is dropped if fewer than 2 trials survive. No check is made for trials that fall after the end of the ephys recording.

ii.
```python
    # 2. Find valid trials
    # Exclude: early lick, ignore/no-response, stimulation
    # Include: hit and miss trials
    valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
    valid_trials = np.where(valid_mask)[0]  # 0-indexed

    if len(valid_trials) < 2:
        print(f'    Skipping {session_id}: only {len(valid_trials)} valid trials')
        f.close()
        return None
```

iii. CONVERSION_NOTES Step 3 quotes the methods: "Early lick and ignore trials omitted from all analyses", and Step 5 Key Decision 5 states "**Exclude early, no, stim trials**: Per paper methods". The AI never reconciles this with the decoder-output specification, which asks for an "ignore" class for Outcome and a "none" class for Lick direction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` — specifically each cluster's `quality` (string), `trial` (1-based trial index per spike) and `trialtm` (spike time relative to trial start) — together with `obj.bp.ev.goCue` for alignment. `obj.clu{probe}.tm` (absolute spike times) is also read for every cluster but never used. Which probe entry is used comes from `SESSION_META`, with a fallback to index 0 when the file stores only one probe.

ii.
```python
        clu_shape = f['obj']['clu'].shape
        if clu_shape[0] == 1:
            probe_idx = 0
        else:
            probe_idx = probe_num - 1

        clu_ref = f['obj']['clu'][probe_idx, 0]
        clu_group = f[clu_ref]
```
```python
            tm_ref = clu_group['trialtm'][clu_idx, 0]
            trial_ref = clu_group['trial'][clu_idx, 0]

            spike_trialtm = f[tm_ref][:].flatten()  # spike times relative to trial start
            spike_trial = f[trial_ref][:].flatten().astype(int)  # 1-indexed trial numbers

            # Get absolute spike times for alignment
            tm_abs_ref = clu_group['tm'][clu_idx, 0]
            spike_tm_abs = f[tm_abs_ref][:].flatten()
```

iii. CONVERSION_NOTES Step 10 Check 3: "(c) Temporal alignment: Spike times aligned to goCue, matching alignSpikes.m", which in MATLAB is `obj.clu{prb}(clu).trialtm_aligned = obj.clu{prb}(clu).trialtm - event`. The probe-index fallback is documented as an edge case: "JEB7 probe=2 but clu shape (1,1): Fixed by checking clu shape and using index 0 when only 1 probe in data."

## 2-b. How is the `neural` data processed?

i. Per cluster and per trial: spikes aligned to the go cue are histogrammed into 10 ms bins over [−2.5, 2.5] s, divided by `dt` to give spikes/s, then smoothed with a **causal** Gaussian kernel (`gausswin(15)` with the first half zeroed and renormalised, boundary handled by prepending the first 15 samples and trimming). No normalisation, z-scoring or baseline subtraction. Clusters from both probes of a dual-probe session are concatenated along the neuron axis. Stored as `float32` in `(n_neurons, n_timebins)` per trial.

ii.
```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    ...
    kern = sig_windows.gaussian(N, std=(N-1)/(2*2.5))  # gausswin default
    kern[:N//2] = 0  # causal: zero out first half
    kern = kern / kern.sum()

    if bctype == 'reflect':
        x_padded = np.concatenate([x[:N], x], axis=0)
        trim = N
    ...
```
```python
                counts, _ = np.histogram(trial_spikes, bins=EDGES)

                # Smooth: convert to firing rate then smooth
                fr = counts / PARAMS['dt']  # spks/sec
                fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])

                trialdat[:, ci, trial_num - 1] = fr_smooth
```
```python
    trialdat = np.concatenate(all_trialdat, axis=1)  # (time, neurons, trials)
```

iii. CONVERSION_NOTES Step 1/3: "Smoothing: causal Gaussian kernel (gausswin), window=15, first half zeroed for causality, boundary=reflect", taken from `utils/mySmooth.m`, with `params.smooth = 15` from `getDefaultParams.m`; Step 10 Check 3 (e) claims a direct match with `mySmooth.m`, and (d) a match with `getSeq.m` for the binning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) Manual curation label: a cluster is dropped if its `quality` string, stripped/lower-cased/null-stripped, is one of `garbage`, `gabrga`, `noisy`, `real?`. Everything else is kept, including empty/null labels (a deliberate fix — one JEB15 probe has 197 unlabeled clusters). (2) Mean firing rate: after concatenating probes, neurons whose mean rate over all bins and **all** trials (including trials later excluded) is ≤ 1 Hz are dropped. A whole session is dropped if fewer than 10 neurons survive. Result: 1,653 clusters after quality, 1,453 after the rate filter, 27–134 per session.

ii.
```python
    'lowFR': 1.0,        # minimum firing rate threshold (Hz) - paper says 1 Hz
    'quality_exclude': ['garbage', 'gabrga', 'noisy', 'real?'],  # matching MATLAB findClusters
```
```python
def find_clusters(qualities, exclude_list):
    exclude_lower = [q.lower().strip() for q in exclude_list]
    idx = []
    for i, q in enumerate(qualities):
        q_clean = q.strip().lower().replace('\x00', '')
        if q_clean in exclude_lower:
            continue
        idx.append(i)
    return np.array(idx, dtype=int)
```
```python
    meanFRs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)  # mean over trials, then time
    fr_mask = meanFRs > PARAMS['lowFR']
    trialdat = trialdat[:, fr_mask, :]
    ...
    if n_neurons < 10:
        print(f'    Skipping {session_id}: only {n_neurons} neurons (need >= 10)')
```

iii. CONVERSION_NOTES Step 3 table: quality filter "all (excl. garbage, noisy) | findClusters.m"; "Low FR threshold | 1 Hz | 'firing rates exceeding 1 Hz'"; "Min units per session | 10 | 'at least 10 units'". Step 4 notes the code default was `lowFR = 0.5` but the paper says 1 Hz, and the paper value was used. The unlabeled-cluster inclusion is justified in Step 10 Check 5 as "matching MATLAB behavior".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is one subtraction per spike: the go cue of the spike's own trial is subtracted from `trialtm`, which is already on the behaviour clock and relative to trial start. Spikes whose trial index is outside `[1, Ntrials]` are discarded first. Binning then happens on the fixed [−2.5, 2.5] s grid, so spikes outside the window fall outside the histogram edges.

ii.
```python
            valid_spike_mask = (spike_trial >= 1) & (spike_trial <= Ntrials)
            spike_trial_valid = spike_trial[valid_spike_mask]
            spike_trialtm_valid = spike_trialtm[valid_spike_mask]

            # goCue times for each spike's trial
            spike_goCue = goCue[spike_trial_valid - 1]  # 0-index into goCue
            spike_aligned = spike_trialtm_valid - spike_goCue
```

iii. Explicitly written into the code comments as the Python translation of `alignSpikes.m`: "In MATLAB: obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event, where event = obj.bp.ev.goCue(obj.clu{prbnum}(clu).trial)". `PARAMS['alignEvent'] = 'goCue'` matches `params.alignEvent`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (`dt = 1/100`), 500 bins spanning −2.5 to +2.5 s from the go cue; the stored input is the bin centre (−2.495 … +2.495). The grid is built once at module level and used for the neural data, the input, and all three camera streams, so no rebinning or resampling of the neural data occurs after the initial histogram. The camera streams are put on this grid by interpolation rather than by binning. `metadata['time_bin_size'] = 10.0` ms.

ii.
```python
PARAMS = {
    'tmin': -2.5,        # seconds relative to alignEvent
    'tmax': 2.5,
    'dt': 1/100,         # 10ms time bins
    ...
}

EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
N_TIMEBINS = len(TIME_AXIS)
```

iii. CONVERSION_NOTES Step 4 records the discrepancy explicitly: "dt | default 1/200 | N/A | 1/100 in most scripts | Using 1/100 (10ms)". Step 3 lists "Neural data time bin | 10ms | params.dt = 1/100" and "Time window | -2.5 to 2.5s from goCue | params.tmin/tmax".

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable — it is the bin-centre axis of the alignment window that the AI defined itself (`TIME_AXIS`), i.e. it is implicitly derived from `obj.bp.ev.goCue`, which defines time zero for every trial.

ii.
```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```
```python
            # Input: time from goCue (1, n_timepoints)
            time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
            session_inputs.append(time_input)
```

iii. Follows the Decoder Task specification ("Temporally align based on Go cue onset", input = "Time from go cue onset in seconds"); the window itself comes from `params.tmin/tmax` of the reference code.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The same `(1, 500)` float32 array of bin centres is emitted for every trial of every session; `input_names = ['time_from_goCue']`. Verified range [−2.495, 2.495].

ii.
```python
            time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. CONVERSION_NOTES Step 10 Check 2 sanity check 2: "Input data: Time axis matches expected [-2.495, 2.495] in 10ms steps (OK)".

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid: spikes are histogrammed into `EDGES` after subtracting the trial's go cue, and the input is `EDGES[:-1] + dt/2`, so input index *k* is the centre of neural bin *k* by construction. The camera-derived outputs are interpolated onto the same axis, so all streams share one time base.

ii.
```python
                counts, _ = np.histogram(trial_spikes, bins=EDGES)
```
```python
        taxis = TIME_AXIS + PARAMS['advance_movement']   # advance_movement = 0.0
        ...
                    xpos[:, trix] = fx(taxis)
```

iii. `PARAMS['advance_movement'] = 0.0` with the comment "no time shift between neural and movement data"; the AI's Step 10 alignment check plots mean firing rate and tongue velocity on the same axis for trial 1.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. A single field: `obj.bp.R` (1 on right-instructed trials). `obj.bp.L` is read but unused, and `hit`/`miss` are **not** used to determine which port was actually licked. The stored variable is therefore the instructed direction, not the direction the animal licked, and it is wrong on every error (miss) trial. Ignore trials are absent from the dataset, so there is no "none" value.

ii.
```python
    L = bp['L'][:].flatten()  # left trials
    R = bp['R'][:].flatten()  # right trials
```
```python
    lick_direction = R.copy()  # 1=right, 0=left
```

iii. CONVERSION_NOTES Step 5 Variable Mapping: "R (right trial) | output[0] | left=0, right=1 | Per trial". The AI treats "choice"/"lick direction" as the trial type; nothing in the notes or trajectory discusses error trials licking the opposite port. Its sanity check was only the marginal ("Lick direction: ~50% right - consistent with balanced task").

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. `lick_direction = R`, cast to int and broadcast across all 500 bins so the per-trial value becomes a constant time series. Two classes only: `output_values[0] = ['left', 'right']`. Measured distribution 0.502 left / 0.498 right.

ii.
```python
            lick_dir = np.full(N_TIMEBINS, info['lick_direction'], dtype=np.int64)
            ...
            output_array = np.stack([lick_dir, context, outcome, ...], axis=0)
```
```python
        'output_values': [
            ['left', 'right'],           # lick_direction
            ...
```

iii. Follows directly from 4-a; the "none" class was never considered because ignore trials were filtered out in Step 2 of `process_session` (justified from the paper's "ignore trials … omitted from all analyses").

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks water-cued (WC) trials.

ii.
```python
    autowater = bp['autowater'][:].flatten()  # WC trials
```

iii. CONVERSION_NOTES Step 2 lists `autowater` among the `obj.bp` behavioural fields and Step 5 maps it directly to `output[1]`; the notes describe the task as "two-context delayed response (DR) and water-cued (WC)" with autowater marking WC.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling `context = 1 - autowater` (WC = 0, DR = 1), broadcast across the 500 bins. Two classes, `['WC', 'DR']`. Measured distribution 0.152 WC / 0.848 DR; three sessions are all-DR.

ii.
```python
    context = 1 - autowater  # DR=1, WC=0 (autowater=1 means WC)
```
```python
            context = np.full(N_TIMEBINS, info['context'], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5: "autowater | output[1] | WC=0, DR=1 | Per trial", chosen to match the coding given in the Decoder Task specification. Step 10 Check 1 notes the all-DR sessions and accepts them: "these are sessions without WC blocks, which is expected".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `obj.bp.hit` alone for the label. `obj.bp.miss` and `obj.bp.no` are read, but only as part of the trial filter; since ignore trials are already removed and only hit|miss trials are kept, `hit == 0` implies miss.

ii.
```python
    hit = bp['hit'][:].flatten()  # correct trials
    miss = bp['miss'][:].flatten()  # error trials
    no = bp['no'][:].flatten()  # ignore/no-response trials
```
```python
    outcome = hit.copy()  # 1=correct, 0=incorrect
```

iii. CONVERSION_NOTES Step 5: "hit | output[2] | incorrect=0, correct=1 | Per trial". The "ignore" class is dropped as a consequence of the trial filter justified from the paper's methods.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `outcome = hit`, cast to int and broadcast across the 500 bins; two classes, `['incorrect', 'correct']`. Measured 0.162 incorrect / 0.838 correct. The specified third class, "ignore", does not exist in the converted data.

ii.
```python
            outcome = np.full(N_TIMEBINS, info['outcome'], dtype=np.int64)
```
```python
            ['incorrect', 'correct'],    # outcome
```

iii. Same as 6-a. The AI validated only the marginal: "Outcome: 83.8% correct - consistent with well-trained mice (>70%)" (Step 10 Check 4).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking of the **side camera only**: `obj.traj{1}` (index 0), feature named `tongue`, using `featNames` to find the feature index, `ts` for x/y/likelihood and `frameTimes` for timing; plus `obj.traj{1}.NdroppedFrames` as a validity flag, `obj.bp.ev.goCue`, and `obj.sglx.bitcode.bitstart` / `obj.sglx.fs` / `obj.bp.ev.bitStart` for the video clock offset. The bottom camera's `top_tongue` is not used. The `likelihood` channel (`ts[...,2,...]`) is never read; the AI relies on x/y already being NaN where tracking failed.

ii.
```python
        traj_ref = f['obj']['traj'][0, 0]  # view 1 (side)
        traj_group = f[traj_ref]
        ...
        tongue_idx = None
        for i, name in enumerate(feat_names):
            if name == 'tongue':
                tongue_idx = i
                break
        ...
                if ts_data.shape[0] == len(feat_names):  # (features, 3, timepoints)
                    x = ts_data[tongue_idx, 0, :]
                    y = ts_data[tongue_idx, 1, :]
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: "**Tongue velocity**: Side view (view 1), tongue feature, speed = sqrt(xvel^2 + yvel^2)", presented as matching `findPosition.m` + `findVelocity.m`. The trajectory shows the AI verified the NaN structure of the side-view tongue ("nan% = 88.8%", "tongue visible (like > 0.5): 11.4%") but it never checked the bottom camera's tongue tracking.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Per trial: (1) skip the trial if `NdroppedFrames` contains NaN or if all `frameTimes` are NaN; (2) convert frame times to go-cue time; (3) linearly interpolate x and y onto the 500-bin axis with `interp1d(..., fill_value=np.nan)` — so bins outside the frame range, and bins whose bracketing frames are untracked (NaN), become NaN; (4) take `np.gradient` of x and y over bin index, **set NaN velocities to 0**, and take `speed = sqrt(xvel² + yvel²)`. No position smoothing, no explicit likelihood threshold, no per-camera normalisation. Trials that were skipped stay all-zero.

ii.
```python
                fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
                fy = interp1d(aligned_times[valid], y[valid], bounds_error=False, fill_value=np.nan)
                xpos[:, trix] = fx(taxis)
                ypos[:, trix] = fy(taxis)
        ...
        for trix in range(Ntrials):
            xv = np.gradient(xpos[:, trix])
            yv = np.gradient(ypos[:, trix])
            # For tongue: set NaN velocity to 0
            xv[np.isnan(xv)] = 0
            yv[np.isnan(yv)] = 0
            ...
        speed = np.sqrt(xvel**2 + yvel**2)
```

iii. CONVERSION_NOTES Step 10 Check 3 (h): "Velocity: gradient(position), tongue NaN->0, paw subtract baseline, matching findVelocity.m". The trajectory records the AI's discovery of the consequence: *"the tongue position data has ~87–95% NaN values … When I compute velocity, NaN positions become 0 velocity (as per the reference code). So the velocity array is mostly 0."*

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two classes with a per-session threshold: pool all non-NaN values of all trials in the session, take the 50th percentile, assign `1` where value ≥ threshold and `0` otherwise. Because most bins are exactly 0 (tongue invisible), the median is 0 and everything would be class 1; the AI added a fallback that recomputes the threshold as the 50th percentile of the **strictly positive** values. There is **no third "not visible" class**. Resulting distribution: 0.973 low / 0.027 high (reference: 0.062 / 0.062 / 0.875 not-visible).

ii.
```python
    threshold = np.percentile(all_values, threshold_percentile)

    # Handle edge case: if threshold is 0 (e.g., tongue velocity is mostly 0
    # when tongue is not visible), use median of positive values instead
    if threshold == 0:
        pos_values = all_values[all_values > 0]
        if len(pos_values) > 0:
            threshold = np.percentile(pos_values, threshold_percentile)

    result = []
    for v in values_per_trial:
        disc = (v >= threshold).astype(np.float32)
        result.append(disc)
```
```python
        tongue_disc = discretize_continuous(result['tongue_vel'])
```

iii. CONVERSION_NOTES Step 5 Key Decision 10: "**Discretization edge case**: When threshold=0, use median of positive values". The trajectory reasoning: *"This is the correct interpretation of the spec, but it results in a degenerate output. The better interpretation is that the 50th percentile should create a meaningful split."* The specified `2: not visible` category is never mentioned anywhere in the notes or trajectory.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are converted to the go-cue clock by subtracting the session's video offset and then the trial's go cue, and the positions are interpolated onto the same 500-bin axis used for the spikes (`taxis = TIME_AXIS + 0.0`). The video offset reproduces `findVideoOffset.m`: mode of `sglx.bitcode.bitstart / sglx.fs` minus mode of `bp.ev.bitStart`. It is recomputed inside each of the three camera functions.

ii.
```python
def find_video_offset(f):
    """Find offset between neural data and video file start.
    Matches findVideoOffset.m
    """
    bitStart_vals = f['obj']['bp']['ev']['bitStart'][:].flatten()
    bitStart = stats.mode(bitStart_vals, keepdims=False).mode

    sglx_bitstart = f['obj']['sglx']['bitcode']['bitstart'][:].flatten()
    sglx_fs = f['obj']['sglx']['fs'][0, 0]
    vidFileOffset = stats.mode(sglx_bitstart, keepdims=False).mode / sglx_fs

    vidshift = vidFileOffset - bitStart
    return vidshift
```
```python
                aligned_times = frameTimes - vidshift - goCue[trix]
```

iii. Documented as a direct port of `findVideoOffset.m` (CONVERSION_NOTES Step 1 function table and Step 10 Check 3(g)); `advance_movement = 0.0` encodes the decision not to shift movement relative to neural data. If the offset computation throws, the function prints a warning and returns 0.0.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The **bottom/top camera** tracking, `obj.traj{2}` (index 1): every feature whose name contains "paw" — i.e. both `top_paw` and `bottom_paw` — via `featNames`, `ts`, `frameTimes`, `NdroppedFrames`, plus the same go cue and video offset. The reference used only `top_paw`.

ii.
```python
        traj_ref = f['obj']['traj'][1, 0]  # view 2 (top)
        traj_group = f[traj_ref]
        ...
        paw_indices = []
        for i, name in enumerate(feat_names):
            if 'paw' in name.lower():
                paw_indices.append(i)
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: "**Paw velocity**: Top view (view 2), top_paw + bottom_paw averaged". No analysis of the relative tracking quality of the two paw features appears in the notes or trajectory.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature separately: interpolate x/y onto the 500-bin axis, then **fill all missing bins by linear interpolation over bin index** (`np.interp` on the non-NaN bins); take `np.gradient`; subtract the trial's median first difference (`nanmedian(np.diff(pos))`) as a baseline-drift correction; fill any remaining NaN; take `speed = hypot(xvel, yvel)`. The two paw speeds are then averaged bin-by-bin. Skipped trials (bad `NdroppedFrames`, no frame times, exceptions) remain all-zero.

ii.
```python
                    # Fill missing with nearest
                    mask = ~np.isnan(xp)
                    if np.any(mask):
                        indices = np.arange(len(xp))
                        xp = np.interp(indices, indices[mask], xp[mask])
                    ...
                    basederiv_x = np.nanmedian(np.diff(xpos[:, trix]))
                    basederiv_y = np.nanmedian(np.diff(ypos[:, trix]))
                    if not np.isnan(basederiv_x):
                        xv = xv - basederiv_x
                    ...
                speed = np.sqrt(xvel**2 + yvel**2)
            all_speeds.append(speed)

        # Average across paw features
        avg_speed = np.mean(all_speeds, axis=0)
```

iii. Presented as the non-tongue branch of `findVelocity.m`: code comment "For non-tongue: subtract baseline derivative and fill missing", and CONVERSION_NOTES Step 10 Check 3(h) "paw subtract baseline, matching findVelocity.m".

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same `discretize_continuous` as the tongue: pooled per-session 50th percentile of all non-NaN values, `1` if ≥ threshold else `0`. Two classes `['low', 'high']`, no "not visible" class — and because missing frames were filled in 8-b, untracked periods are indistinguishable from tracked ones. Distribution 0.529 low / 0.471 high (reference: 0.407 / 0.407 / 0.186 not-visible).

ii.
```python
        paw_disc = discretize_continuous(result['paw_vel'])
```
```python
        disc = (v >= threshold).astype(np.float32)
```

iii. Follows the Decoder Task's "0: < 50th percentile, 1: >= 50th percentile" literally; the "2: not visible" category is not implemented or discussed.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue: `frameTimes − vidshift − goCue[trial]`, then interpolation onto `TIME_AXIS`. The frame times come from the bottom-camera `traj` entry, i.e. the camera that tracks the feature, and the video offset is recomputed by a second call to `find_video_offset`.

ii.
```python
        taxis = TIME_AXIS + PARAMS['advance_movement']
        vidshift = find_video_offset(f)
        ...
                    aligned_times = frameTimes - vidshift - goCue[trix]
                    ...
                    fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
```

iii. Same as 7-d — one session-level bitcode offset from `findVideoOffset.m`, no extra shift (`advance_movement = 0`).

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside the data structure, read with `scipy.io.loadmat`; `me['data']` is unwrapped once more if it is itself a struct with a `data` field. Timing comes from the **side camera's** `traj{1}.frameTimes` in the data-structure file (falling back to a synthetic 400 Hz axis if those are missing or all NaN), plus `goCue` and the video offset.

ii.
```python
        me_raw = sio.loadmat(me_file)
        me_struct = me_raw['me'][0, 0]
        me_data = me_struct['data']
        # Handle nested struct: if me.data is itself a struct with 'data' field
        # (matching MATLAB: if isstruct(me.data), me.data = me.data.data)
        if me_data.dtype.names is not None and 'data' in me_data.dtype.names:
            me_data = me_data[0, 0]['data']
```
```python
                    traj_ref = f['obj']['traj'][0, 0]
                    traj_group = f[traj_ref]
                    ft_ref = traj_group['frameTimes'][trix, 0]
                    frameTimes = f[ft_ref][:].flatten()
                except:
                    frameTimes = np.arange(1, len(me_trial) + 1) / 400.0
```

iii. CONVERSION_NOTES Step 1 lists `loadMotionEnergy.m` ("Loads motion energy, interpolates to neural time axis") and Step 10 Check 5 documents the nested-struct edge case: "JEB15 sessions have me.data as a struct with .data and .moveThresh fields. Fixed by unwrapping (matching MATLAB: if isstruct(me.data), me.data = me.data.data)".

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment: the per-frame trace is truncated to the length of the frame-time vector, linearly interpolated onto the 500-bin axis, and any remaining NaN bins are filled by linear interpolation over bin index. Trials that fail (exception, fewer than 2 valid points, index beyond the number of stored traces) remain all-zero.

ii.
```python
                min_len = min(len(aligned_times), len(me_trial))
                aligned_times = aligned_times[:min_len]
                me_trial = me_trial[:min_len]
                ...
                f_interp = interp1d(aligned_times[valid], me_trial[valid],
                                   bounds_error=False, fill_value=np.nan)
                me_interp = f_interp(taxis)

                # Fill NaN with nearest
                mask = ~np.isnan(me_interp)
                if np.any(mask):
                    indices = np.arange(len(me_interp))
                    me_interp = np.interp(indices, indices[mask], me_interp[mask])

                me_aligned[:, trix] = me_interp
```

iii. Described as "matching loadMotionEnergy.m" (CONVERSION_NOTES Step 1 and Step 10 Check 3(g)); motion energy is already one scalar per frame in the source file, so only resampling is required.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same per-session 50th-percentile split into two classes `['low', 'high']`; no "no video" class. The measured split is essentially exactly 0.500/0.500 in every session, because zero-filled (missing) trials are pooled with real data before the median is taken. An earlier iteration produced two sessions whose motion energy was constant 1 (all-zero data ⇒ threshold 0 ⇒ everything ≥ 0); the final run shows no such session.

ii.
```python
        me_disc = discretize_continuous(result['motion_energy'])
```
```python
    threshold = np.percentile(all_values, threshold_percentile)
    if threshold == 0:
        pos_values = all_values[all_values > 0]
        if len(pos_values) > 0:
            threshold = np.percentile(pos_values, threshold_percentile)
```

iii. Literal reading of the Decoder Task percentile rule; CONVERSION_NOTES Step 10 Check 1 records the investigation of the degenerate sessions ("Motion energy [1.0, 1.0] for 2 sessions - investigated, these sessions have all-zero motion energy data"). The "2: no video" class is not implemented.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and grid as the DLC streams: `frameTimes(side camera) − vidshift − goCue[trial]`, then interpolation onto `TIME_AXIS`. A third call to `find_video_offset` is made here. If the side-camera frame times are unavailable or all NaN, a synthetic 400 Hz axis starting at frame 1 is substituted — i.e. that trial is aligned to the *video* start rather than to the go cue.

ii.
```python
        taxis = TIME_AXIS + PARAMS['advance_movement']
        vidshift = find_video_offset(f)
        ...
                if np.all(np.isnan(frameTimes)) or len(frameTimes) == 0:
                    frameTimes = np.arange(1, len(me_trial) + 1) / 400.0

                aligned_times = frameTimes - vidshift - goCue[trix]
```

iii. Same justification as 7-d/8-d; the 400 Hz fallback is an undocumented defensive default (the camera frame rate the AI inferred from the data).

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Defensively and silently. (1) `warnings.filterwarnings('ignore')` at import. (2) Every per-trial camera/motion-energy computation is wrapped in `try/except: continue`, and each of the three top-level camera functions is wrapped in a `try/except` that prints a warning and returns `None`. (3) When a stream returns `None`, the session's trials get all-zero arrays for that output; when a single trial fails, its column stays zero. Zeros are then discretised as "low", so missing video is indistinguishable from genuinely low movement. (4) Untracked tongue frames (NaN positions) become velocity 0; untracked paw/motion-energy bins are filled by linear interpolation. (5) `NdroppedFrames` containing NaN causes the trial's tracking to be skipped. (6) Structural variability is handled: `ts` orientation is detected from `featNames` length, dual- vs single-probe `clu` layouts, nested `me.data` structs, clusters with empty/null quality strings. (7) Sessions missing a data file, having <2 valid trials, or <10 neurons are skipped. Not handled: trials after the end of the ephys recording (these would appear as all-zero firing across all neurons), and `bp` fields longer than `Ntrials`.

ii.
```python
warnings.filterwarnings('ignore')
```
```python
            except Exception as e:
                continue
    except Exception as e:
        print(f'    Warning: Could not compute tongue velocity: {e}')
        return None
```
```python
        if tongue_vel is not None:
            tongue_vel_trials.append(tongue_vel[:, trial_idx].astype(np.float32))
        else:
            tongue_vel_trials.append(np.zeros(N_TIMEBINS, dtype=np.float32))
```
```python
                ndrop = f[ndrop_ref][:].flatten()
                if np.isnan(ndrop).any():
                    continue
```

iii. CONVERSION_NOTES Step 10 Check 5 lists the edge cases found and fixed (null quality strings, single-entry `clu` for a probe-2 session, dual probes, nested motion-energy struct, zero-threshold discretisation). The zero-filling policy itself is not justified anywhere; it is the implicit default of the `None`-returning error handling.

## 11-a. What are the most time-consuming steps of the code?

i. The AI reports only whole-session timings (`t_start`/`elapsed` per session), 4–16 s per session and 188.9 s for the 25-session run, and does not profile within a session. The actual dominant cost is the neural block: a Python loop over clusters × all `Ntrials` trials that performs one `np.histogram` **and one `scipy.signal.convolve`** per (cluster, trial) — roughly 1.6 M convolutions over the dataset — on top of one HDF5 read of `trialtm`, `trial` and (unused) `tm` per cluster. Second is the camera block: three passes over all trials with a fresh `interp1d` per trial per feature. For comparison the human reference converts 44 sessions in ~135 s; the AI takes 189 s for 25.

ii.
```python
        for ci, clu_idx in enumerate(cluid):
            ...
            for trial_num in range(1, Ntrials + 1):
                ...
                counts, _ = np.histogram(trial_spikes, bins=EDGES)
                fr = counts / PARAMS['dt']
                fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
```
```python
    elapsed = time.time() - t_start
    print(f'    Done: {n_neurons} neurons, {len(valid_trials)} trials, {elapsed:.1f}s')
```

iii. CONVERSION_NOTES Step 7 Run Time Estimates: "Full processing | ~6-7s | ~180s (25 sessions)", i.e. the AI's justification is simply that the total is far under the 15-minute budget in the instructions, so no optimisation was pursued. No speed-ups are listed in the Step 6 "Code speedups added" section.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Nothing was vectorised. The clearly vectorisable loops that remain are: (1) the `for trial_num in range(1, Ntrials+1)` spike-binning loop, which is a single `np.histogram2d(spike_trial, spike_aligned, bins=[trial_edges, EDGES])` over all trials at once (as in the reference); (2) the per-trial smoothing call, since the whole `(time, trials)` matrix could be filtered in one call along axis 0; (3) the per-column `for j in range(x_padded.shape[1])` convolution loop inside `causal_gaussian_smooth`; (4) the per-trial `np.gradient` loops in `compute_tongue_velocity` and `compute_paw_velocity`, which operate on rectangular `(N_TIMEBINS, Ntrials)` arrays and could be single `np.gradient(..., axis=0)` calls; (5) the baseline-derivative and NaN-fill loops, likewise. The per-trial interpolation loops are the only ones genuinely hard to vectorise, because each trial has a different number of camera frames.

ii.
```python
    for trix in range(Ntrials):
        xv = np.gradient(xpos[:, trix])
        yv = np.gradient(ypos[:, trix])
        xv[np.isnan(xv)] = 0
        yv[np.isnan(yv)] = 0
        xvel[:, trix] = xv
        yvel[:, trix] = yv
```
```python
    for j in range(x_padded.shape[1]):
        result[:, j] = convolve(x_padded[:, j], kern, mode='same')
```

iii. Not discussed. CONVERSION_NOTES Step 6 leaves "Code inefficiencies identified" and "Code speedups added" effectively empty, and the Step 7 estimate (~180 s) was used to conclude no optimisation was needed.

## 11-c. What processing does the code repeat multiple times?

i. Four repetitions. (1) `find_video_offset(f)` is called three times per session — once in `compute_tongue_velocity`, once in `compute_paw_velocity`, once in `load_motion_energy` — each time re-reading `bp.ev.bitStart` and `sglx.bitcode.bitstart` and computing two modes, although the offset is a session constant; it also computes `np.median(bitStart)` and immediately overwrites it. (2) `compute_paw_velocity` loops over the paw features in the outer loop and over trials in the inner loop, so the whole bottom-camera `ts` and `frameTimes` dataset is read from HDF5 twice. (3) `traj{1}.frameTimes` is read once for the tongue and again for every trial in `load_motion_energy`. (4) `discretize_continuous` is run once in `main` and again inside `plot_processing` when `--show-processing` is used.

ii.
```python
        taxis = TIME_AXIS + PARAMS['advance_movement']
        vidshift = find_video_offset(f)      # in compute_tongue_velocity
```
```python
        vidshift = find_video_offset(f)      # again in compute_paw_velocity
```
```python
        vidshift = find_video_offset(f)      # and again in load_motion_energy
```
```python
        for paw_idx in paw_indices:          # outer loop over features …
            for trix in range(Ntrials):      # … re-reads every trial's ts
                ts_ref = traj_group['ts'][trix, 0]
                ts_data = f[ts_ref][:]
```
```python
def find_video_offset(f):
        bitStart = np.median(f['obj']['bp']['ev']['bitStart'][:].flatten())
        # mode equivalent
        bitStart_vals = f['obj']['bp']['ev']['bitStart'][:].flatten()
        bitStart = stats.mode(bitStart_vals, keepdims=False).mode
```

iii. Not discussed anywhere in CONVERSION_NOTES; the repetition is a by-product of writing the three camera streams as three independent self-contained functions rather than as a session-level object (the reference computes the offset once in `Camera.__init__`).

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four items. (1) `spike_tm_abs = f[tm_abs_ref][:]` reads the absolute spike-time vector of **every** cluster out of HDF5 and never uses it — pure I/O on the largest per-cluster array. (2) All neural binning and smoothing is done for **all** `Ntrials`, including the 25.5% of trials (2,110 of 8,260) that the early/ignore/stim filter has already excluded, and only then are the valid columns selected; the same is true of the tongue, paw and motion-energy arrays. (3) `obj.bp.L` is read and never used (`R` alone determines the label), as is the `likelihood` channel of `ts`. (4) The identical `(1, 500)` time-axis array is materialised and stored separately for each of the 6,150 trials, and the three per-trial scalars are expanded to 500-long int64 vectors before being written into the output (int64 rather than a small integer type, contributing to the 932 MB file).

ii.
```python
            # Get absolute spike times for alignment
            tm_abs_ref = clu_group['tm'][clu_idx, 0]
            spike_tm_abs = f[tm_abs_ref][:].flatten()     # never used again
```
```python
        trialdat = np.zeros((N_TIMEBINS, len(cluid), Ntrials), dtype=np.float32)
        ...
            for trial_num in range(1, Ntrials + 1):       # every trial, not just valid ones
```
```python
    for trial_idx in valid_trials:
        neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))
```
```python
            time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
            session_inputs.append(time_input)
```

iii. Not discussed in CONVERSION_NOTES. The unused `tm` read is left over from the AI's initial uncertainty about how to align spikes (the code comment "Get absolute spike times for alignment" precedes the block, and the alignment was subsequently implemented from `trialtm` instead).
