# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of 44 sessions, `SESSION_DEFS`, each entry being `(animal, date, probes_for_ALM, data_dir_key)`, transcribed from the authors' `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` scripts. Sessions are not discovered by globbing; files present on disk but absent from the loading scripts (`JEB23_2023-10-20`, `JEB24_2023-10-03`, `JEB24_2023-10-04`) are excluded, and `JEB4`/`JEB5` (scripts but no data) are dropped. Each session is one `data_structure_<anm>_<date>.mat`, living in either `Ephys_Behavior` (25 sessions) or `RandomizedDelay_Ephys_Behavior` (19 sessions), with a companion `motionEnergy_<anm>_<date>.mat`. `_detect_file_format` opens the file with `h5py` to decide between two readers: `_load_raw_data_hdf5` (MATLAB v7.3) and `_load_raw_data_v5` (scipy.io). Both return a common dict of `bp`, `vidshift`, spike times/trials/qualities for the requested probe(s), and `traj_data`. Sessions are processed one at a time in a single loop in `convert_all`.

ii.
```python
SESSION_DEFS = [
    ('EKH1', '2021-08-07', [2], 'ephys'),
    ...
    ('JEB24', '2023-11-03', [1], 'random'),
]
DATA_DIRS = {
    'ephys': os.path.join(DATA_ROOT, 'Ephys_Behavior'),
    'random': os.path.join(DATA_ROOT, 'RandomizedDelay_Ephys_Behavior'),
}
```
```python
fmt = _detect_file_format(data_fn)
try:
    if fmt == 'hdf5':
        raw = _load_raw_data_hdf5(data_fn, probes)
    else:
        raw = _load_raw_data_v5(data_fn, probes)
except Exception as e:
    print(f"  WARNING: Could not load {session_id}: {e}")
    return None
```
```python
for i, (anm, date, probes, ddir) in enumerate(session_defs):
    print(f"[{i+1}/{len(session_defs)}] Loading {anm}_{date}...")
    result = load_session(anm, date, probes, ddir, PARAMS)
```

iii. From CONVERSION_NOTES.md Step 2/Step 4: "Session loading scripts in DataLoadingScripts/Recording and video/ define which probe(s) per session"; "3 data files without loading scripts (JEB23_2023-10-20, JEB24_2023-10-03/04) excluded. Matches paper"; "JEB4/JEB5 have loading scripts but no data files. Use 25 available sessions." The two readers exist because 11 of the RandomizedDelay files are MATLAB v5 rather than v7.3.

## 1-b. How are the data split into subjects?

i. The animal identifier is the first element of each `SESSION_DEFS` tuple (equivalently the prefix of the session name). It is carried on each session result as `result['animal']`; at assembly the unique animals are sorted and `subject_idx` indexes into that list, one entry per session, in session order. Result: 14 subjects over 44 sessions, matching the reference exactly (EKH1, EKH3, JEB6, JEB7, JEB11–JEB15, JEB19, JEB23, JEB24, JGR2, JGR3).

ii.
```python
all_animals = sorted(set(s['animal'] for s in all_sessions))
animal_to_idx = {a: i for i, a in enumerate(all_animals)}
...
subject_idx.append(animal_to_idx[sess['animal']])
...
'subjects': all_animals,
'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. Not explicitly justified beyond the filename/loading-script convention. CONVERSION_NOTES Step 9 flags that the paper reports 9 DR mice while 10 DR animals have data: "Paper excludes 1 animal (unknown criterion); we include all with data."

## 1-c. How are the data split into sessions?

i. One session = one entry of `SESSION_DEFS` = one `data_structure_*.mat` file. The folder is not inferred but stored per session (`'ephys'` / `'random'`), so fixed-delay and randomized-delay recordings are loaded uniformly and concatenated into one list of 44 sessions. Each becomes one element of `neural`, `input`, `output`, and `brain_region_idx`. A session is dropped if the file is missing, if loading raises, if it has no units after the quality filter, if it has fewer than 2 valid trials, or if it has fewer than `min_units = 10` units; in practice no session was dropped (minimum 17 units).

ii.
```python
data_dir = DATA_DIRS[data_dir_key]
data_fn = os.path.join(data_dir, f'data_structure_{anm}_{date}.mat')
me_fn = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
session_id = f'{anm}_{date}'
...
if n_neurons < params['min_units']:
    print(f"  WARNING: {session_id} has {n_neurons} units (< {params['min_units']}), skipping")
    return None
```

iii. CONVERSION_NOTES Step 5 decision 6: "Include ALL sessions (both Ephys_Behavior and RandomizedDelay) that have loading scripts." Step 3 records the paper's rule "sessions included for analysis only if they had at least 10 units," which motivates `min_units = 10`.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the number of trials; every per-trial `bp` field (`hit`, `miss`, `no`, `early`, `L`, `R`, `autowater`, `stim.enable`, `ev.goCue/sample/delay`) is flattened and truncated to `Ntrials`, because some fields are stored longer than the trial count. Spikes are assigned to trials by the cluster's own `trial` field (1-based) and timed by `trialtm` (relative to trial start); video is split by trial because `obj.traj{view}` stores one `ts`/`frameTimes` entry per trial, and motion energy holds one trace per trial. Trial numbering is kept 1-based for indexing into raw arrays (`valid_trials = np.where(trial_mask)[0] + 1`) and 0-based for indexing into per-trial `bp` columns.

ii.
```python
bp['Ntrials'] = int(bp_group['Ntrials'][()].flat[0])
ntrials = bp['Ntrials']
for field in ['hit', 'miss', 'no', 'early', 'L', 'R', 'autowater']:
    bp[field] = bp_group[field][()].flatten().astype(np.float64)[:ntrials]
```
```python
valid_trials = np.where(trial_mask)[0] + 1  # 1-indexed
...
for t_idx, trial_num in enumerate(valid_trials):
    spk_mask = spike_trial == trial_num
    ...
    spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
```

iii. Not separately argued; it follows the Bpod trial table used by the reference code (`findTrials.m`, `getSeq.m` loop `for j = 1:obj.bp.Ntrials`). CONVERSION_NOTES Step 2 documents the `obj.bp` / `obj.clu` / `obj.traj` per-trial layout used here.

## 1-e. How are trials filtered based on quality controls?

i. Exactly two filters, both applied before anything is computed: early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`) are dropped. Nothing else is excluded — hit, miss and no-response trials are all kept, and both DR and WC trials are kept, because outcome and context are decoder targets. 13,823 of 15,155 trials survive. The AI did **not** implement any cut on trials recorded after the probe stopped: 61 trials in two sessions (session 36 = JEB23_2023-10-21, session 43 = JEB24_2023-11-02) end up with all-zero neural data, which `train_decoder.py --verify-only` reports as 61 warnings. These were documented and knowingly retained.

ii.
```python
trial_mask = (bp['early'] == 0) & (bp['stim_enable'] == 0)
valid_trials = np.where(trial_mask)[0] + 1  # 1-indexed
if len(valid_trials) < 2:
    print(f"  WARNING: {session_id} has < 2 valid trials, skipping")
    return None
```

iii. CONVERSION_NOTES Step 5 decisions 9–10: "Include ALL trial types (hit, miss, no-response) per session. The decoder should predict outcome, so needs all types"; "Exclude early lick and stim trials: Per reference code conditions (~early, ~stim.enable)" — these are the conditions used throughout `getDefaultParams.m`/`WorkingWithDataObjs.m`. On the zero trials, Step 9 Known Issues: "These are trials where no spikes fell within the [-2.5, 2.5]s window around goCue. Affects 61/13823 trials (0.44%)," and Step 10: "Acceptable (0.44% of all trials)."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}`, restricted to the probe(s) listed for that session in `SESSION_DEFS` (probes concatenated where a session has two). For each cluster the AI reads `trialtm` (spike time relative to trial start), `trial` (1-based trial number of each spike), and `quality` (manual curation label). The other input is `obj.bp.ev.goCue`, which supplies the per-trial alignment time. `spkWavs`, `tm`, `site`/`channel` are not used.

ii.
```python
for probe_num in probes:
    probe_ref = clu_refs[probe_num - 1]
    probe_group = f[probe_ref]
    quality_refs = probe_group['quality'][()].flatten()
    trialtm_refs = probe_group['trialtm'][()].flatten()
    trial_refs   = probe_group['trial'][()].flatten()
    for i in range(len(quality_refs)):
        quality = h5_read_string(f, quality_refs[i]).strip().lower()
        trialtm = f[trialtm_refs[i]][()].flatten().astype(np.float64)
        trial   = f[trial_refs[i]][()].flatten().astype(np.float64)
```
```python
align_times = ev[params['align_event']]   # params['align_event'] = 'goCue'
```

iii. CONVERSION_NOTES Step 1: the pipeline is "loadObjs -> findTrials -> findClusters -> alignSpikes -> getSeq -> removeLowFRClusters", and Step 5 maps "obj.clu spike times -> neural". Step 5 decision 8: "Dual probe sessions: Concatenate neurons from both probes (both are ALM)."

## 2-b. How is the `neural` data processed?

i. For each kept cluster and each kept trial, spikes are histogrammed into 10 ms bins on the edges `-2.5 : 0.01 : 2.5` (500 bins), divided by `dt` to give spikes/s, and smoothed along time with `causal_gaussian_smooth`, a Python reimplementation of the authors' `mySmooth.m`: `gausswin(15)` (σ = (N−1)/(2·2.5) = 2.8 bins), first `floor(N/2)` taps zeroed to make the kernel causal, renormalised to sum 1, convolved with `'same'`, with the `'reflect'` boundary handling (prepend the first N samples, then trim them). No normalisation, z-scoring, or baseline subtraction is applied; stored values are firing rates in Hz as `float32`. Trials with no spikes for a unit are left at zero. I verified this independently: re-binning the raw `trialtm` of one unit and applying the same smoothing reproduces the converted `trialdat` exactly (`np.allclose` True).

ii.
```python
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
...
counts, _ = np.histogram(spk_times, bins=edges)
fr = causal_gaussian_smooth(counts.astype(np.float64) / params['dt'],
                            params['smooth_window'], params['smooth_bctype'])
trialdat[neuron_idx, :, t_idx] = fr.astype(np.float32)
```
```python
kern = np.array(sig_windows.gaussian(N, std=(N - 1) / (2 * 2.5)))
kern[:N // 2] = 0          # Make causal: zero out first half
kern = kern / kern.sum()
if bctype == 'reflect':
    x_filt = np.concatenate([x[:N, :], x], axis=0); trim = N
```

iii. CONVERSION_NOTES Step 5 decision 3: "Causal Gaussian smooth, window=15: Matches reference code"; Step 1 notes "Spike binning: histc with edges tmin:dt:tmax, then smooth with causal Gaussian (window=15)" and "Single trial data: obj.trialdat ... units = spks/sec (divided by dt)", i.e. `getSeq.m` + `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. (1) Manual curation label: the label is stripped and lower-cased and the unit is dropped if it is in `{garbage, noisy, gabrga, real?}` — the exact drop list of `findClusters.m` under `params.quality = {'all'}` — and additionally if the label is empty or `'nan'`. `Poor`, `Multi`, `Fair`, `Good`, `Great`, `Excellent` are all kept. (2) Mean firing rate: after binning and smoothing, units whose mean rate over the whole window and all kept trials is ≤ 1 Hz are removed (`removeLowFRClusters.m` with `params.lowFR = 1`). (3) Session level: a session with fewer than 10 surviving units would be dropped. Result: 2,456 units over 44 sessions (17–141 per session).

ii.
```python
'low_fr': 1.0,  # Hz, remove neurons below this
'min_units': 10,
'quality_exclude': {'garbage', 'noisy', 'gabrga', 'real?'},
...
keep_mask = np.array([q not in quality_exclude and q != '' and q != 'nan'
                      for q in all_qualities])
good_indices = np.where(keep_mask)[0]
...
mean_fr = np.mean(trialdat, axis=(1, 2))
fr_mask = mean_fr > params['low_fr']
trialdat = trialdat[fr_mask, :, :]
```

iii. CONVERSION_NOTES Step 5 decision 5: "Quality='all': Exclude garbage, noisy, gabrga, real? - matches reference"; decision 4: "lowFR=1 Hz: Matches paper and tutorial." Step 4 resolves a conflict explicitly: "lowFR threshold | getDefaultParams: 0.5 Hz | Paper: 1 Hz | WorkingWithDataObjs.m uses 1 Hz. Use 1 Hz as per the tutorial which matches the paper."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is a single subtraction, done before binning: for each spike of a cluster falling in trial *n*, `trialtm − bp.ev.goCue[n]`. `trialtm` is already on the behaviour clock and relative to that trial's start, and `goCue` is on the same clock, so no offset or interpolation is needed. The window is then fixed at −2.5 to +2.5 s for every trial, so every trial has 500 bins.

ii.
```python
PARAMS = {'align_event': 'goCue', 'tmin': -2.5, 'tmax': 2.5, ...}
align_times = ev[params['align_event']]
...
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
counts, _ = np.histogram(spk_times, bins=edges)
```

iii. CONVERSION_NOTES Step 5 decision 1: "Align to goCue: As specified in decoder task and paper default"; Step 1 lists `alignSpikes.m` as "Align spike times to event (goCue)" and Step 3 "Temporal alignment: Align to goCue onset; Time window: -2.5 to 2.5 s from goCue."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms (`dt = 1/100`), 500 bins spanning −2.5 to +2.5 s, identical for every trial and session; `metadata['time_bin_size'] = 10.0` ms. Spikes are histogrammed directly at this resolution, so there is no rebinning or resampling of the neural data. The same `time_axis` (bin centres, −2.495 … 2.495 s) is used for the decoder input and for interpolating all three video-derived outputs, so every stream shares one time base. The AI explicitly noticed and resolved a conflict between the two reference parameter files.

ii.
```python
PARAMS = {
    'tmin': -2.5,
    'tmax': 2.5,
    'dt': 1.0 / 100,  # 10 ms bins
    ...
}
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
n_timebins = len(time_axis)          # 500
```

iii. CONVERSION_NOTES Step 4: "dt (time bin) | getDefaultParams: 1/200 | WorkingWithDataObjs.m uses 1/100 (10ms). Use 10ms as per tutorial." Step 5 decision 2: "10ms time bins: Matches WorkingWithDataObjs.m tutorial and paper." (The tutorial's own comment above `params.dt = 1/100` reads "use a 5 ms bin width", a contradiction the AI did not flag.)

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable: the input is defined by the conversion as the centre of each neural time bin, i.e. it is the go-cue-aligned time axis itself. It is implicitly derived from `bp.ev.goCue`, which defines time 0 of that axis.

ii.
```python
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
```
```python
'input_names': ['time_from_go_cue'],
```

iii. CONVERSION_NOTES Step 5 variable mapping: "Time from goCue -> input[0] | Continuous, time-varying | Linear ramp from -2.5 to 2.5s."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond taking bin centres. The same 1×500 `float32` vector is used for every trial of every session (a fresh copy is materialised per trial). Range is −2.495 to 2.495 s; the verification log reports `[-2.5, 2.5]`.

ii.
```python
for t in range(n_trials):
    ...
    time_input = time_axis.astype(np.float32).reshape(1, -1)
    sess_input.append(time_input)
```

iii. Not argued beyond the mapping table entry ("linear ramp"); the Decoder Task specifies this input as continuous and time-varying.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `time_axis` is derived from the very `edges` array used to histogram the spikes (`edges[:-1] + dt/2`), which is also exactly what `getSeq.m` does (`obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1)`). Bin *k* of `input` and bin *k* of `neural` are therefore the same interval by construction, and the same axis is the interpolation target for tongue, paw and motion energy.

ii.
```python
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
n_timebins = len(time_axis)
...
counts, _ = np.histogram(spk_times, bins=edges)
```

iii. Implicit; the shared axis is also what makes the video interpolation (`interp1d(...)(time_axis)`) aligned with the spikes.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. A single per-trial field, `obj.bp.R` (1 = right trial, 0 = left trial), subset to the valid trials. `bp.hit`, `bp.miss` and `bp.no` are loaded but **not** used for this output, so what is stored is the *instructed* lick direction, not the direction the animal actually licked. On error (miss) trials the animal licked the opposite port from `R`, and on no-response trials it licked neither, but both still receive the instructed side.

ii.
```python
valid_trial_indices = valid_trials - 1
lick_direction = bp['R'][valid_trial_indices].copy()
```
```python
'output_values': [
    ['left', 'right'],        # lick_direction
    ...
```

iii. CONVERSION_NOTES Step 5 mapping: "obj.bp.R/L -> output[0]: lick_direction | L=0, R=1, per-trial | From trial info"; planned sanity check "Verify lick direction matches R/L in bp." No justification is given for equating instructed side with lick direction, and the required third class is never discussed anywhere in the notes or in the trajectory.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. None — the 0/1 `R` flag is used directly as the class code, cast to `int`, and broadcast across all 500 time bins of the trial. Two classes only: `left` (0) and `right` (1). The Decoder Task's third class, `none`, is not created, so the ~13% of trials on which the animal did not lick are labelled left or right, and the ~12% miss trials are labelled with the wrong side. Overall distribution 0.496 / 0.504, versus the reference's 0.423 / 0.446 / 0.131.

ii.
```python
lick_dir = np.full((1, n_timebins), int(sess['lick_direction'][t]), dtype=np.int64)
...
trial_output = np.concatenate([lick_dir, context, outc, tongue_v, paw_v, me_v],
                              axis=0).astype(np.int64)
```

iii. None offered; the notes present the mapping as self-evident.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial field, `obj.bp.autowater`, subset to valid trials. `autowater = 1` marks trials where water was delivered from a random port without any cue — the water-cued (WC) context; everything else is delayed-response (DR).

ii.
```python
for field in ['hit', 'miss', 'no', 'early', 'L', 'R', 'autowater']:
    bp[field] = bp_group[field][()].flatten().astype(np.float64)[:ntrials]
...
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
```

iii. CONVERSION_NOTES Step 5 mapping: "obj.bp.autowater -> output[1]: behavioral_context ... autowater=1 -> WC=0, autowater=0 -> DR=1"; Step 3 curation rules: "For DR: ~autowater (autowater=0); For WC: autowater=1", taken from the reference conditions.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `context = 1 − autowater`, so WC → 0 and DR → 1, matching the class order in `output_values`. The value is cast to `int` and broadcast over all 500 bins. Overall distribution 0.097 WC / 0.903 DR, essentially identical to the reference (0.0966 / 0.9034). 14 sessions contain no WC trials at all.

ii.
```python
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
...
context = np.full((1, n_timebins), int(sess['behavioral_context'][t]), dtype=np.int64)
```
```python
'output_values': [..., ['WC', 'DR'], ...]
```

iii. CONVERSION_NOTES Step 9 Known Issues 3: "Behavioral context always DR: ~14 sessions have no WC trials (autowater=0 for all trials). Expected for sessions that only ran the DR task without interleaved WC blocks." The 0/1 coding follows the ordering given in the Decoder Task.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. A single per-trial field, `obj.bp.hit`. `bp.miss` and `bp.no` are read into the `bp` dict but never used for the outcome, so "not a hit" covers both error trials and no-response trials.

ii.
```python
outcome = bp['hit'][valid_trial_indices].copy()
```

iii. CONVERSION_NOTES Step 5 mapping: "obj.bp.hit -> output[2]: outcome | incorrect=0, correct=1, per-trial | hit=1 -> correct=1, miss=1 -> incorrect=0." The mapping table asserts the miss→0 relation but the code never reads `miss`, and the `ignore` class required by the Decoder Task is not mentioned anywhere.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. None — the 0/1 `hit` flag becomes the class code directly and is broadcast across the 500 bins. Two classes: `incorrect` (0) and `correct` (1). The required third class `ignore` is absent, so the ~13% of trials where the animal did not respond are labelled `incorrect`. Distribution 0.253 / 0.747, versus the reference's 0.120 incorrect / 0.749 correct / 0.131 ignore — the `correct` fraction matches, and the entire discrepancy is the merged ignore trials.

ii.
```python
outc = np.full((1, n_timebins), int(sess['outcome'][t]), dtype=np.int64)
```
```python
'output_values': [..., ['incorrect', 'correct'], ...]
```

iii. None beyond the mapping table.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, bottom camera only (`view = 2`, i.e. `traj_data['views'][1]`), feature `top_tongue` — matched by exact name against that view's `featNames`, with a substring fallback. Per trial it reads `ts` (x, y, likelihood × frames) and `frameTimes`. The side-camera tongue (`tongue`) is not used. Supporting variables are `obj.sglx.fs`, `obj.sglx.bitcode.bitstart` and `obj.bp.ev.bitStart` (for the video offset) and `bp.ev.goCue`.

ii.
```python
tongue_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_tongue',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=True)
```
```python
if ts_fmt == 'hdf5':
    x_raw = ts[feat_idx, 0, :]
    y_raw = ts[feat_idx, 1, :]
else:
    x_raw = ts[:, 0, feat_idx]
    y_raw = ts[:, 1, feat_idx]
```

iii. CONVERSION_NOTES Step 5 decision 11: "Tongue velocity: Compute from bottom cam DLC features (top_tongue or similar). Use Euclidean velocity = sqrt(vx^2 + vy^2)." Step 6 records a bug fix: "Fixed DLC data indexing bug: HDF5 shape is (nFeats, 3, nFrames), not (nFrames, 3, nFeats)." No reason is given for preferring the bottom view over the side view or over combining both.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Following `findPosition.m` / `findVelocity.m`: (1) frame times are put on the go-cue clock (7-d); (2) frames whose x, y or time is NaN are dropped and the remaining x and y are **linearly interpolated onto the 500-bin neural time axis** with `interp1d(..., bounds_error=False, fill_value=np.nan)`; (3) `np.gradient` of the interpolated x and y gives per-bin velocity components (no division by dt, so units are px/bin); (4) because this is the tongue, no position smoothing and no nearest-fill are applied, and NaN velocities are set to 0 ("set tongue velocity to 0 if not visible", as in `findVelocity.m`); (5) speed is `sqrt(vx² + vy²)`. Note the deviation from MATLAB: `findPosition.m` passes the NaN-containing `ts` straight to `interp1`, so untracked stretches stay NaN, whereas the AI removes the NaN samples first and therefore interpolates *across* untracked gaps. The tongue is untracked in ~84% of frames, so the resulting trace is 0 outside the first/last tracked frame and a fabricated slow ramp in between. There is no explicit likelihood test (the authors already NaN out x/y where likelihood ≤ 0.9), and no cross-camera normalisation (only one camera is used).

ii.
```python
valid = ~np.isnan(ft_a) & ~np.isnan(x_r) & ~np.isnan(y_r)
if valid.sum() < 2:
    continue
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
ypos = interp1d(ft_a[valid], y_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
if not is_tongue:
    xpos = _fill_nearest(xpos); ypos = _fill_nearest(ypos)
xvel = np.gradient(xpos); yvel = np.gradient(ypos)
if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0
...
speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
```

iii. The design mirrors the reference MATLAB exactly on the points the AI documented: Step 1 notes "Kinematics: DLC position -> velocity via gradient, fill missing values" and "findVelocity | Compute velocity from position via gradient". The NaN→0 rule is copied from the comment in `findVelocity.m` that the agent read. Step 9 Known Issue 4 rationalises the resulting distribution: "Tongue velocity skewed: 84.8% 'high' overall because tongue is only active during response epoch; speed=0 when tongue retracted, and median threshold at 50th percentile captures this asymmetry."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. `_discretize_velocity` pools all non-NaN values of the session (all trials × all bins), takes the 50th percentile as the threshold, and assigns 1 where `value >= threshold`, 0 elsewhere; NaN entries are also set to 0. Only two classes are created — the Decoder Task's class 2 (`not visible`) is not implemented, and `output_values[3] = ['low', 'high']`.

This produces a degenerate label. Because the tongue is retracted for most of the window, more than half of the pooled values are exactly 0, so the 50th percentile *is* 0 and the test `value >= 0` is true for every non-NaN bin. I confirmed this directly on EKH1_2021-08-07: the threshold is exactly `0.0`, and `np.array_equal(np.isnan(tongue_vel_raw), tongue_vel_disc == 0)` is `True` — class 0 is precisely the set of trials with fewer than two tracked frames and class 1 is everything else. The stored `tongue_velocity` output therefore encodes video/tracking availability, not tongue speed. Dataset-wide the split is 0.152 / 0.848 (reference: 0.062 below / 0.062 above / 0.875 not visible).

ii.
```python
def _discretize_velocity(data, n_timebins, n_trials):
    if data is None:
        return np.zeros((n_timebins, n_trials), dtype=np.int64)
    valid_vals = data[~np.isnan(data)]
    if valid_vals.size == 0:
        return np.zeros((n_timebins, n_trials), dtype=np.int64)
    threshold = np.percentile(valid_vals, 50)
    disc = (data >= threshold).astype(np.int64)
    disc[np.isnan(data)] = 0          # Replace NaN entries with 0
    return disc
```

iii. CONVERSION_NOTES Step 5 decision 14: "Per-session discretization: 50th percentile threshold computed on all timepoints across all trials in session." Step 12 interprets the 0.816 decoding accuracy as "Consistent with tongue being active only during response epoch." The notes never test whether the threshold is non-degenerate, and the `not visible` class is never mentioned.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A per-session video offset is computed once as in `findVideoOffset.m`: `median(sglx.bitcode.bitstart)/sglx.fs − median(bp.ev.bitStart)` (the MATLAB uses `mode`; I checked all 33 v7.3 sessions and median and mode give identical values, 0.49 s or 0.99 s). Frame times are then mapped to the go-cue clock as `frameTimes − vidshift − goCue[trial]`, and positions are interpolated onto the identical `time_axis` used for the spike bins, so the two streams share one axis exactly. If the `sglx`/`bitStart` read fails the offset silently falls back to 0.5 s; if `frameTimes` is empty or all-NaN it falls back to a synthetic 400 Hz axis `(1..n)/400`, as `findPosition.m` does.

ii.
```python
bit_start = np.nanmedian(bp_group['ev']['bitStart'][()].flatten())
fs = sglx['fs'][()].flat[0]
bitcode_bitstart = np.nanmedian(sglx['bitcode']['bitstart'][()].flatten())
vid_file_offset = bitcode_bitstart / fs
vidshift = vid_file_offset - bit_start
```
```python
if ft.size == 0 or np.all(np.isnan(ft)):
    ft = np.arange(1, len(x_raw) + 1) / 400.0
ft_aligned = ft - vidshift - align_times[trix]
...
xpos = interp1d(ft_a[valid], x_r[valid], ...)(time_axis)
```

iii. CONVERSION_NOTES Step 1/Step 3: "Video offset: Need to subtract 0.5s from frameTimes to sync cameras with SpikeGLX (or use findVideoOffset)"; Step 1 lists `findVideoOffset` as "Compute temporal offset between neural and video." The 400 Hz fallback matches Step 3 "Video frame rate | 400 Hz".

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` tracking, bottom camera (`view = 2`), feature `top_paw` only. `bottom_paw` is available in that view but is not used. Same supporting variables as the tongue (`frameTimes`, video offset, `goCue`).

ii.
```python
paw_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_paw',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=False)
```

iii. CONVERSION_NOTES Step 5 decision 12: "Paw velocity: From bottom cam (top_paw, bottom_paw). Use Euclidean velocity." No reason is recorded for using `top_paw` alone; Step 2 lists both paws among the bottom-camera features.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same interpolate-then-differentiate pipeline as the tongue, but taking the non-tongue branch of `findVelocity.m`/`findPosition.m`: after interpolation onto the neural time axis, missing positions are filled with the nearest valid value (`_fill_nearest`, MATLAB `fillmissing(...,'nearest')`); the per-bin median displacement is subtracted from each velocity component as a baseline-drift correction; the velocity components are again nearest-filled; and speed is `sqrt(vx² + vy²)`. Units are px/bin. Unlike MATLAB, which subtracts `basederiv(1)` from both components (an apparent typo), the AI subtracts the matching component from each. No likelihood cut and no normalisation are applied. Since `top_paw` is tracked in essentially every frame, only ~2% of bins end up NaN.

ii.
```python
if not is_tongue:
    xpos = _fill_nearest(xpos)
    ypos = _fill_nearest(ypos)
xvel = np.gradient(xpos)
yvel = np.gradient(ypos)
if is_tongue:
    ...
else:
    base_xvel = np.nanmedian(np.diff(xpos))
    base_yvel = np.nanmedian(np.diff(ypos))
    xvel -= (base_xvel if not np.isnan(base_xvel) else 0)
    yvel -= (base_yvel if not np.isnan(base_yvel) else 0)
    xvel = _fill_nearest(xvel)
    yvel = _fill_nearest(yvel)
speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
```

iii. CONVERSION_NOTES Step 1: "findVelocity | Compute velocity from position via gradient"; Step 3 "Kinematics: DLC position -> velocity via gradient, fill missing values." The branch structure (fill + baseline subtraction for non-tongue, zero-fill for tongue) is copied verbatim in intent from `findVelocity.m`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same `_discretize_velocity`: session-wide 50th percentile of all non-NaN values, `>= threshold` → 1, else 0, and NaN → 0. Two classes only (`['low','high']`); the Decoder Task's class 2 (`not visible`) is not implemented, so the (few) untracked bins are silently labelled "below threshold". The threshold here is non-degenerate (0.376 px/bin for EKH1), and the resulting split is near-even — 0.507 / 0.493 overall, per-session almost exactly 0.500/0.500 — versus the reference's 0.407 / 0.407 / 0.186.

ii.
```python
paw_vel_disc = _discretize_velocity(paw_vel, n_timebins, n_valid_trials)
```
```python
threshold = np.percentile(valid_vals, 50)
disc = (data >= threshold).astype(np.int64)
disc[np.isnan(data)] = 0
```

iii. Step 5 decision 14 (per-session 50th percentile). Step 12: "paw_velocity (0.585): Lowest accuracy. Paw movements are subtle and the 50th percentile discretization near the median makes this inherently noisy. Still above chance." No justification is given for omitting the `not visible` class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue: the same per-session `vidshift`, the same `frameTimes − vidshift − goCue[trial]`, and interpolation onto the same 500-bin `time_axis` as the spikes. Because the paw is a bottom-camera feature, the bottom camera's own `frameTimes` are used (frame times are read from the same view entry as the feature).

ii.
```python
view_data = traj_data['views'][view - 1]
...
ft = trial_info['frameTimes']
ft_aligned = ft - vidshift - align_times[trix]
min_len = min(len(ft_aligned), len(x_raw))
ft_a, x_r, y_r = ft_aligned[:min_len], x_raw[:min_len], y_raw[:min_len]
```

iii. Same as 7-d; no separate treatment is documented for the paw.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` beside each data structure, field `me.data` — one trace per trial at the camera frame rate. `obj.me` (present in some sessions) is not used. Frame times come from the **side** camera's `frameTimes` in `obj.traj` (with a 400 Hz synthetic fallback). Only one of the three layouts present in the data is handled: `me_struct['data'].item()`, i.e. `me = {data, moveThresh}` with `data` a cell array. This silently fails for the other two layouts — 4 files where `me` is a bare cell array with no `data` field (JEB23 2023-10-10/11/12/13; the scipy read raises, the h5py fallback then reports "file signature not found") and 3 files where `me.data` is itself a struct (`me.data.data`; JEB15_2022-07-26, JEB15_2022-07-28, JEB24_2023-10-31, where per-trial indexing raises `IndexError` and every trial is skipped). I verified all seven cases directly against the raw files. In those 7 of 44 sessions `me_data` is `None` or all-NaN and the motion-energy output becomes the constant 0.

ii.
```python
me_cell = None
try:
    me_file = sio.loadmat(me_fn, squeeze_me=True)
    me_struct = me_file['me']
    me_cell = me_struct['data'].item()
except Exception:
    me_h5 = h5py.File(me_fn, 'r')
    me_grp = me_h5['me']
    me_data_refs = me_grp['data'][()].flatten()
    me_cell = [me_h5[ref][()].flatten().astype(np.float64) for ref in me_data_refs]
    me_h5.close()
...
for trix in range(ntrials):
    try:
        me_trial = me_cell[trix].flatten().astype(np.float64) if hasattr(...) else ...
    except (IndexError, TypeError):
        continue
```
```python
try:
    me_data = _load_motion_energy_generic(...)
except Exception as e:
    print(f"  WARNING: Could not load motion energy for {session_id}: {e}")
```

iii. CONVERSION_NOTES Step 5 decision 13: "Motion energy: Load from motionEnergy files, interpolate to neural time axis." Step 9 Known Issue 2: "4 JEB23 sessions ... have ME .mat files that can't be opened (file signature not found). These get all-zero ME, resulting in motion_energy always 'low'. Additional sessions (JEB15_2022-07-27/28, JEB24_2023-10-31) also have ME all one class. Total: 7/44 sessions with degenerate ME." The issue was recorded but attributed to corrupt files and never fixed, even though `loadMotionEnergy.m` — which the agent read — contains the needed guard `if isstruct(me.data), me.data = me.data.data; end`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment and resampling: the per-frame trace is linearly interpolated onto the 500-bin neural time axis, and interior NaNs of a trial that has at least one valid sample are then filled with the nearest valid value (which also extrapolates flat beyond the ends of the frame coverage). The value is used as-is, on its native scale (the per-pixel differencing and 99th-percentile spatial reduction were already done upstream by the authors). Trials with fewer than two valid samples are left as NaN.

ii.
```python
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                               kind='linear', bounds_error=False,
                               fill_value=np.nan)(time_axis).astype(np.float32)
...
for trix in range(ntrials):
    col = me_aligned[:, trix]
    if not np.isnan(col).all() and np.isnan(col).any():
        me_aligned[:, trix] = _fill_nearest(col)
```

iii. CONVERSION_NOTES Step 3: "Motion energy alignment: Interpolate to neural time axis using interp1, align to goCue", following `loadMotionEnergy.m`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same `_discretize_velocity`: session-wide 50th percentile over non-NaN values, `>= threshold` → 1 else 0, NaN → 0. Two classes (`['low','high']`); the Decoder Task's class 2 (`no video`) is not implemented. For the 37 sessions where motion energy loaded, the split is essentially exactly 0.500/0.500 per session; for the 7 broken sessions `_discretize_velocity` short-circuits on `data is None` / no valid values and returns an all-zero array, so ~16% of sessions assert "below threshold" for every bin of every trial rather than flagging the data as missing. Overall 0.584 / 0.416, versus the reference's 0.479 / 0.483 / 0.038.

ii.
```python
me_disc = _discretize_velocity(me_data, n_timebins, n_valid_trials)
```
```python
if data is None:
    return np.zeros((n_timebins, n_trials), dtype=np.int64)
...
if valid_vals.size == 0:
    return np.zeros((n_timebins, n_trials), dtype=np.int64)
```

iii. Step 5 decision 14 (per-session 50th percentile). Step 12 reads the 0.816 accuracy as "Strong despite 7 degenerate ME sessions." The missing `no video` class is not discussed.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the tracking: the side camera (`traj_data['views'][0]`) supplies `frameTimes`, which are shifted by `− vidshift − goCue[trial]` and used as the interpolation abscissa onto `time_axis`. If the side camera's frame times are missing or all-NaN, a synthetic 400 Hz axis is used. The motion-energy trace and the frame-time vector are truncated to their common length before interpolating, so a mismatch in counts cannot mis-align the trace (though it silently drops the tail).

ii.
```python
side_cam = traj_data['views'][0] if traj_data is not None else None
...
if side_cam is not None and trix < len(side_cam['trials']) and side_cam['trials'][trix] is not None:
    ft = side_cam['trials'][trix]['frameTimes']
if ft is None or ft.size == 0 or np.all(np.isnan(ft)):
    ft = np.arange(1, me_trial.size + 1) / 400.0
ft_aligned = ft - vidshift - align_times[trix]
min_len = min(len(ft_aligned), len(me_trial))
```

iii. Same rationale as 7-d/8-d (`findVideoOffset.m`, `loadMotionEnergy.m`); Step 3 "Video frame rate | 400 Hz" justifies the fallback.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms, all of which keep the trial and substitute a value rather than mark the gap:
- **Untracked video frames** (x/y NaN where DLC likelihood ≤ 0.9): for the tongue the velocity is set to 0; for the paw and motion energy the value is nearest-filled. Whatever remains NaN is coded as class 0 (`low` / below threshold) by `_discretize_velocity`, so missing video is indistinguishable from genuinely slow movement. No `not visible` / `no video` class exists.
- **Trials with no usable video** (`valid.sum() < 2`, empty or all-NaN `frameTimes`): the trial is skipped, leaving a NaN column → again all class 0. A synthetic 400 Hz frame-time axis is used where `frameTimes` is absent entirely.
- **Whole streams that fail to load**: broad `try/except` blocks around the `traj` parse, the motion-energy load and the per-trial loops swallow the error, print at most a warning, and let the stream become `None`/NaN → an all-zero output for the session. This is what silently flattened motion energy in 7 of 44 sessions.
- **Missing per-trial neural data**: trials recorded after the probe stopped are kept with all-zero firing rates (61 trials in 2 sessions), and per-trial `bp` fields longer than `Ntrials` are truncated. Two format readers, a substring fallback for DLC feature names, and `min_len` truncation between mismatched frame/value vectors handle the remaining format irregularities.

ii.
```python
except Exception:
    continue
```
```python
if ft.size == 0 or np.all(np.isnan(ft)):
    ft = np.arange(1, len(x_raw) + 1) / 400.0
```
```python
def _fill_nearest(arr):
    nans = np.isnan(arr)
    if not nans.any(): return arr
    if nans.all():     return arr
    valid_idx = np.where(~nans)[0]; nan_idx = np.where(nans)[0]
    nearest = np.searchsorted(valid_idx, nan_idx).clip(0, len(valid_idx) - 1)
    arr[nans] = arr[valid_idx[nearest]]
    return arr
```
```python
disc[np.isnan(data)] = 0          # missing -> "low"
```

iii. Nearest-filling and tongue zero-filling are copied from `findPosition.m`/`findVelocity.m`, which the notes cite as "fill missing values." The all-zero neural trials are called "Acceptable (0.44% of all trials)" (Step 10). The degenerate motion-energy sessions are listed under Step 9 Known Issues but Step 10 concludes "No new issues requiring code changes. All edge cases were previously identified and documented." No rationale is given for encoding missing video as a valid movement class.

## 11-a. What are the most time-consuming steps of the code?

i. The AI instrumented timing only at session granularity (`t0 = time.time()` in `load_session`, printed per session, plus a total in `convert_all`); it did not profile individual steps. Measured: 44 sessions in 162.8 s, 1.9–6.9 s per session, 166.2 s including the pickle write. The dominant costs are (a) reading the `.mat` files — all clusters' spike arrays and both cameras' full `ts` arrays for every trial — and (b) the nested `for neuron: for trial:` loop that runs one `np.histogram` and one `np.convolve`-based smoothing per (neuron, trial) pair, i.e. ~6.9 M histogram+convolution calls over the dataset. Estimated total in Step 7 was ~220 s; actual 166 s, comfortably inside the 15-minute budget, so no optimisation was undertaken.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    ...
    for t_idx, trial_num in enumerate(valid_trials):
        spk_mask = spike_trial == trial_num
        if not np.any(spk_mask): continue
        spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
        counts, _ = np.histogram(spk_times, bins=edges)
        fr = causal_gaussian_smooth(...)
```
```python
elapsed = time.time() - t0
print(f"  {session_id}: {n_neurons} neurons, {n_valid_trials} trials, ... ({elapsed:.1f}s)")
```

iii. CONVERSION_NOTES Step 7: "| Full loading | ~5s | ~220s (~3.7 min) |". The Step 6 template fields "Code inefficiencies identified" and "Code speedups added" were left unfilled; the only Step 6 entries are two bug fixes.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI did not identify or vectorise any loop. The clearly vectorisable ones are:
- The neuron × trial spike-binning loop: a single `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, edges])` per cluster (or per session) replaces the whole inner loop, as the human reference does.
- The smoothing: `causal_gaussian_smooth` is called once per (neuron, trial) on a 500-sample vector, and internally loops over columns with `np.convolve`; it accepts 2-D input, so one call per neuron on the (time, trials) matrix — or one `scipy.ndimage` call on the full 3-D array — would do.
- The mask `spike_trial == trial_num` is recomputed for every trial of every neuron (O(n_spikes × n_trials)); a single `np.searchsorted`/`argsort` grouping would remove it.
- The per-trial `interp1d` loops in `_compute_feature_velocity_generic` and `_load_motion_energy_generic` are harder to vectorise because trials have different frame counts, but the three calls (tongue, paw, motion energy) each re-walk all trials.
- The final assembly loop builds a fresh `time_axis` copy per trial.

ii.
```python
for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```
```python
for t in range(n_trials):
    sess_neural.append(trialdat[:, :, t].astype(np.float32))
    time_input = time_axis.astype(np.float32).reshape(1, -1)
    sess_input.append(time_input)
```

iii. None recorded — the notes contain no discussion of vectorisation; the implicit justification is that the measured runtime (166 s) was well under the 15-minute threshold at which the instructions require optimisation.

## 11-c. What processing does the code repeat multiple times?

i. Several small repetitions, none load-bearing:
- Each data file is opened twice: `_detect_file_format` opens it with `h5py` purely to test the signature, then the reader opens it again.
- Tongue velocity, paw velocity and motion energy are all computed for **every** trial of the session and only afterwards subset to the valid trials (`tongue_vel[:, valid_0idx]`), so ~9% of that work is thrown away; each of the three calls also re-reads the same per-trial `frameTimes` and recomputes the same `ft - vidshift - align_times[trix]` shift.
- Firing rates are binned and smoothed for all quality-passing units and only then filtered by mean rate, so smoothing is done for units that are discarded.
- The `(1, 500)` time-input array is re-created for each of the 13,823 trials rather than shared.
- Per-cluster spike masks are recomputed per trial (see 11-b).
- Computed once and correctly reused: the video offset (`vidshift`, once per session), `edges`/`time_axis` (once per session), and the smoothing kernel is rebuilt per call but is trivial.

ii.
```python
def _detect_file_format(data_fn):
    try:
        f = h5py.File(data_fn, 'r'); f.close(); return 'hdf5'
    except Exception:
        return 'v5'
```
```python
if tongue_vel is not None: tongue_vel = tongue_vel[:, valid_0idx]
if paw_vel    is not None: paw_vel    = paw_vel[:, valid_0idx]
if me_data    is not None: me_data    = me_data[:, valid_0idx]
```

iii. Not discussed in the notes.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- Fields loaded and never used: `bp['no']`, `bp['L']`, `ev['sample']`, `ev['delay']` (and, in an earlier revision, `ev.lickL`/`ev.lickR`).
- `traj_data` is built for **both** cameras and **all** features (7 side + 10 bottom) for every trial, including the full `ts` arrays, although only `top_tongue` and `top_paw` from the bottom camera and the frame times of both cameras are ever used.
- Rates are computed for units later removed by the 1 Hz filter, and video/motion-energy for trials later removed by the early-lick/photostim filter.
- Raw (un-discretised) `tongue_vel_raw`, `paw_vel_raw`, `me_raw` are carried in the per-session dict for plotting and then dropped; they are not written to the pickle.
- Outputs are stored as `int64` for six 0/1 variables (≈330 MB of the 1.84 GB pickle); `int8` would have sufficed, and the per-trial time input is duplicated 13,823 times.
- With `--show-processing`, two 12-panel figures are rendered per run.

ii.
```python
for field in ['hit', 'miss', 'no', 'early', 'L', 'R', 'autowater']:
    bp[field] = bp_group[field][()].flatten().astype(np.float64)[:ntrials]
for field in ['goCue', 'sample', 'delay']:
    ev[field] = ev_group[field][()].flatten().astype(np.float64)[:ntrials]
```
```python
for view_idx in range(len(traj_refs)):
    ...
    for t in range(min(ntrials, len(ts_refs))):
        ts = f[ts_refs[t]][()].astype(np.float64)
```
```python
trial_output = np.concatenate([lick_dir, context, outc, tongue_v, paw_v, me_v],
                              axis=0).astype(np.int64)
```

iii. Not discussed. Step 6 does record one dtype decision in the opposite direction: "Fixed output dtype: must be int64 not float32."
