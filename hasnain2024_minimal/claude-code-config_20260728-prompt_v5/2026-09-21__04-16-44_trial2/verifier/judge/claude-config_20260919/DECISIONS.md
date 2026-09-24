# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a table of 44 sessions, `SESSION_META`, each entry being `(animal, date, probe_list, data_dir)`, transcribed from the authors' own `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` scripts. It does not glob the data folders. Sessions live in one of two folders (`Ephys_Behavior` for the fixed-delay/two-context task, `RandomizedDelay_Ephys_Behavior` for the randomized-delay task) and the folder is stored in the table rather than searched for. Each session is one `data_structure_<anm>_<date>.mat`. Because the shared files come in two MATLAB formats, `load_session` tries the v7.3 HDF5 reader (`load_session_h5`, h5py) first and falls back to the v5 reader (`load_session_scipy`, `scipy.io.loadmat`) on any exception. Motion energy is a separate file, `motionEnergy_<anm>_<date>.mat`, loaded beside it. Only the fields actually needed are pulled out of the file (bp flags, go cue, sglx bitcode, the selected probes' clusters, the bottom-camera trajectory).

ii.
```python
EPHYS_DIR = '/app/data/Ephys_Behavior'
RANDDELAY_DIR = '/app/data/RandomizedDelay_Ephys_Behavior'

SESSION_META = [
    ('EKH1', '2021-08-07', [2], EPHYS_DIR),
    ...
    ('JEB24', '2023-11-03', [1], RANDDELAY_DIR),
]
```

```python
def load_session(anm, date, probe_list, data_dir):
    """Load one session, auto-detecting file format."""
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(data_dir, fn)

    # Try HDF5 first
    try:
        sess = load_session_h5(fpath, probe_list)
    except Exception:
        # Fall back to scipy
        sess = load_session_scipy(fpath, probe_list)
```

and the driver loop:
```python
for anm, date, probes, data_dir in SESSION_META:
    print(f"Loading {anm}_{date}...")
    try:
        sess = load_session(anm, date, probes, data_dir)
    except Exception as e:
        print(f"  ERROR loading {anm}_{date}: {e}")
        continue
```

iii. From the trajectory: the AI read every `load<ANM>_ALMVideo.m` (`for f in .../*.m; do head -30 "$f"; done`) and treated them as the authoritative record of which sessions and which probe entered the paper's analyses. It explicitly checked files present on disk but absent from those scripts — e.g. `JEB24_2023-10-03` — and confirmed they have no `clu` field ("it's a behavior-only session, no ephys data. That's why it's excluded from the loading scripts"). It also excluded the `Video only` / MAH optogenetic sessions because they carry no neural data. The dual-format reader was added after the first run crashed on the v5 files ("Some files are not HDF5 format (older MATLAB format). Let me check and handle both").

## 1-b. How are the data split into subjects?

i. The subject is the animal id, taken directly from the `anm` field of the session table (equivalently, the part of the filename before the underscore). `subjects` is built as a list of unique animals in order of first appearance, and `subject_idx` records each session's index into that list. The result is 14 subjects across the 44 sessions.

ii.
```python
if anm not in all_subjects:
    all_subjects.append(anm)
subj_idx = all_subjects.index(anm)

all_sessions.append(result)
subject_idx_list.append(subj_idx)
```
```python
'subjects': all_subjects,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. Not discussed explicitly in the trajectory. The animal id is a field of the hard-coded session table (which came from the per-animal `load<ANM>_ALMVideo.m` files), so the subject grouping is inherited from the authors' own per-animal organisation rather than read out of the file contents.

## 1-c. How are the data split into sessions?

i. One session = one row of `SESSION_META` = one `data_structure_<anm>_<date>.mat` file = one element of `neural`, `input`, `output`, `brain_region_idx` and `subject_idx`. Fixed-delay and randomized-delay sessions are pooled into one list rather than kept as two datasets. Where a session was recorded with two probes (the four JEB15 sessions with `[1, 2]`), the clusters from both probes are concatenated into one population for that session. A session is dropped if it has fewer than 2 usable trials or fewer than 10 units after the firing-rate filter; in practice neither triggered, so all 44 sessions are kept.

ii.
```python
for probe_num in probe_list:
    probe_idx = probe_num - 1
    probe_ref = clu_data[probe_idx, 0]
    ...
    all_spike_trials.append(spike_trials)
```
```python
if good_units.sum() < MIN_UNITS:
    print(f"  Skipping {sess['anm']}_{sess['date']}: only {good_units.sum()} units after FR filter")
    return None
```

iii. The AI reasoned about whether to restrict to the 12 two-context sessions and concluded all sessions should be included: "the paper's two datasets are Ephys_Behavior (25 sessions, 9 mice, DR with 12 sessions/6 mice also doing WC) and RandomizedDelay_Ephys_Behavior (19 sessions, 4 mice). Since the decoder needs behavioral context as an output, I think all sessions should be included". The `MIN_UNITS = 10` rule was taken from the methods: "Recording sessions were included for analysis only if they had at least 10 units".

## 1-d. How are the data split into trials?

i. The trial count is `obj.bp.Ntrials`, and every per-trial quantity is indexed by that trial number: the Bpod flags (`hit`, `miss`, `no`, `R`, `L`, `autowater`, `early`, `stim.enable`), the go cue `bp.ev.goCue`, the per-trial camera entries of `obj.traj{2}` (`frameTimes`, `ts`), and the per-trial motion-energy traces. Spikes carry their own trial label, `clu.trial` (1-based), so trials are selected by masking spikes rather than by reconstructing boundaries. No re-segmentation of any kind is done.

ii.
```python
ntrials = int(bp['Ntrials'][0, 0])
hit = np.array(bp['hit']).flatten().astype(bool)
...
goCue = np.array(ev['goCue']).flatten()
```
```python
for ti in range(ntrials):
    spk_mask = spike_trials_raw == (ti + 1)
    if not np.any(spk_mask):
        continue
    aligned_times = spike_trialtm_raw[spk_mask] - goCue[ti]
```
```python
for t in range(ntrials):
    ft = np.array(f[frameTimes_ds[t, 0]]).flatten()
    trial_frame_times.append(ft)
    ts = np.array(f[ts_ds[t, 0]])  # (n_bodyparts, 3, n_frames)
```

iii. The AI inspected the file structure directly and confirmed the layout ("bp fields", "Ntrials", "hit shape", `obj.traj{2}` being a `1 x nTrials` struct array, `clu.trial` giving the trial of each spike). The Bpod table defines the trials, so no inference was needed. Note that, unlike the reference, the AI does not truncate the `bp` fields to `Ntrials`; it relies on their lengths already matching.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both applied at the start of `process_session`: photostimulation trials (`bp.stim.enable`) and early-lick trials (`bp.early`) are dropped, following the paper and the authors' `params.condition` strings (`~stim.enable&...&~early`). Ignore trials are deliberately kept (they become the `none`/`ignore` classes). A session is skipped if fewer than 2 trials survive. No other trial-level curation is applied — in particular, trials that occur after the electrophysiology recording has stopped are **not** removed, so 61 trials (28 in session 36, 33 in session 43) enter the dataset with every neuron at exactly zero firing rate for all 500 bins. 13,823 trials are kept.

ii.
```python
# Trial mask: exclude stim and early lick
trial_mask = ~sess['stim_enable'] & ~sess['early']
valid_trials = np.where(trial_mask)[0]

if len(valid_trials) < 2:
    print(f"  Skipping {sess['anm']}_{sess['date']}: too few valid trials ({len(valid_trials)})")
    return None
```

iii. The AI listed "Trial filtering: exclude stim.enable and early lick trials" in its module docstring and in its reasoning ("For the actual conditions, I want stim-excluded and early-lick-excluded trials"), taken from the paper's `params.condition` strings and from the methods ("Trials in which the animal contacted the lickport before the reward ('early lick') were omitted from analyses"). For the all-zero trials, the AI saw the verifier's warnings and dismissed them without acting: "Data format is valid with some warnings about zero neural data in late trials (likely very late trials in a session where spiking stopped)."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters of the selected probe(s), `obj.clu{probe}`: `trial` (1-based trial of each spike), `trialtm` (spike time relative to that trial's start) and `quality` (the manual curation label). `obj.bp.ev.goCue` supplies the alignment time. `clu.tm`, `clu.spkWavs` and `clu.site` are not read.

ii.
```python
quality_ds = probe_group['quality']
trial_ds = probe_group['trial']
trialtm_ds = probe_group['trialtm']
n_clusters = quality_ds.shape[0]

for ci in range(n_clusters):
    q_str = read_h5_string(f, quality_ds[ci, 0]).strip()
    if q_str.lower() in {q.lower() for q in EXCLUDED_QUALITIES}:
        continue
    spike_trials = np.array(f[trial_ds[ci, 0]]).flatten()
    spike_trialtm = np.array(f[trialtm_ds[ci, 0]]).flatten()
```

iii. The AI read `WorkingWithDataObjs.m` part 1.3, which documents `obj.clu{probenum}(clunum).trialtm` as "spike time in trial" and `.trial` as the trial number, and `alignSpikes.m`, which aligns using exactly these two fields.

## 2-b. How is the `neural` data processed?

i. Per unit and per trial, spikes aligned to the go cue are histogrammed into the 10 ms bin grid, divided by the bin width to give spikes/s, and then smoothed along time with the **causal** half-Gaussian kernel of the authors' `mySmooth.m`: a 15-point `gausswin` (alpha = 2.5) with its first `floor(15/2) = 7` taps zeroed and the rest renormalised to sum to 1, applied with `'reflect'` boundary handling (prepend the first 15 samples, convolve, trim). No normalisation, baseline subtraction or z-scoring; stored values are firing rates in Hz as `float32`. Units from both probes of a dual-probe session are concatenated.

ii.
```python
def causal_gaussian_kernel(N):
    """Replicate MATLAB mySmooth: gausswin(N) with first half zeroed (causal)."""
    alpha = 2.5
    n = np.arange(N)
    w = np.exp(-0.5 * (alpha * (n - (N - 1) / 2) / ((N - 1) / 2)) ** 2)
    w[:N // 2] = 0
    w /= w.sum()
    return w
```
```python
    if bc_type == 'reflect':
        x_padded = np.concatenate([x[:N], x], axis=0)
        trim = N
    ...
    out = np.zeros_like(x_padded)
    for j in range(x_padded.shape[1]):
        out[:, j] = np.convolve(x_padded[:, j], kern, mode='same')
    out = out[trim:]
```
```python
            counts, _ = np.histogram(aligned_times, bins=edges)
            fr = counts / DT
            trialdat[:, ui, ti] = smooth_data(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The AI read `utils/mySmooth.m` in full (step 62) and reimplemented it line for line, including the causal zeroing of the first half of the kernel and the `reflect` boundary condition, citing `params.smooth = 15` and `params.bctype = 'reflect'` from `WorkingWithDataObjs.m`. Its docstring records this: "Smoothing: causal half-Gaussian kernel, window = 15 bins / Boundary condition: reflect".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. (1) Cluster quality: the manual label is stripped, lower-cased and dropped if it is one of `garbage`, `gabrga`, `noisy`, `real?` — the exact drop list of `findClusters.m` for `params.quality = {'all'}`. Everything else, including `poor`, `fair` and `multi`, is kept. (2) Firing rate: units whose mean rate over the window, averaged across the surviving trials, is not above 1 Hz are removed. (3) Session level: a session with fewer than 10 surviving units would be dropped. The result is 2,456 units across 44 sessions (17 to 141 per session); no session actually fell below 10.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR_THRESH = 1.0  # Hz
MIN_UNITS = 10  # minimum units per session
```
```python
    q_str = read_h5_string(f, quality_ds[ci, 0]).strip()
    if q_str.lower() in {q.lower() for q in EXCLUDED_QUALITIES}:
        continue
```
```python
    # Remove low FR units
    mean_frs = np.mean(trialdat[:, :, valid_trials], axis=(0, 2))
    good_units = mean_frs > LOW_FR_THRESH

    if good_units.sum() < MIN_UNITS:
        print(f"  Skipping {sess['anm']}_{sess['date']}: only {good_units.sum()} units after FR filter")
        return None
```

iii. The drop list is copied from `findClusters.m`, which the AI read (step 20): for `'all'` it returns `~ismember('garbage') & ~ismember('gabrga') & ~ismember('noisy') & ~ismember('real?')`. The 1 Hz cut comes from `params.lowFR = 1` in `WorkingWithDataObjs.m` and from the methods ("All units with firing rates exceeding 1 Hz were included in all other analyses"); `removeLowFRClusters.m` applies it as a mean over trials, which is what the AI reproduces. The ≥10 unit session rule comes from the methods ("Recording sessions were included for analysis only if they had at least 10 units").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A single subtraction. `clu.trialtm` is already on the Bpod clock and relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so spike time from the go cue is `trialtm − goCue[trial]`. No interpolation, no per-session offset, no resampling. These aligned times are then histogrammed into the fixed −2.5 to +2.5 s edge grid, so spikes outside the window simply fall off the ends.

ii.
```python
ALIGN_EVENT = 'goCue'
...
for ti in range(ntrials):
    spk_mask = spike_trials_raw == (ti + 1)
    if not np.any(spk_mask):
        continue
    aligned_times = spike_trialtm_raw[spk_mask] - goCue[ti]
    counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. This is `alignSpikes.m`, which the AI read: `obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event` with `event = obj.bp.ev.(params.alignEvent)(...)` and `params.alignEvent = 'goCue'`. The task also specifies go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins. The window is −2.5 to +2.5 s, giving 500 bins per trial; `metadata['time_bin_size'] = 10.0` ms and `off_start`/`off_end` are −2.5/2.5. The grid is built once per session and is identical for every trial, session and data stream (neural, input and all three camera outputs). Spikes are binned directly at 10 ms — there is no rebinning of an intermediate finer representation. The camera streams, which are recorded at 400 Hz (2.5 ms), are brought onto the grid by interpolation at the bin centres rather than by averaging the ~4 frames that fall inside each bin.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100  # 10 ms bins
...
edges = np.arange(TMIN, TMAX + DT, DT)
time_centers = edges[:-1] + DT / 2
n_timebins = len(time_centers)
```

iii. The AI took all three numbers from the parameter block of `WorkingWithDataObjs.m`, quoting it in its docstring: "Time window: -2.5 to 2.5 s, dt = 10 ms (params.dt = 1/100)". That file does literally set `params.dt = 1/100`, although the comment on the line immediately above it reads "use a 5 ms bin width and bin spike data from -2.5 to 2.5 sec", and `getDefaultParams.m` — which the AI also read — sets `params.dt = 1/200`. The AI did not comment on the discrepancy.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing in the raw data beyond the alignment event itself: the input is defined by the analysis window. `bp.ev.goCue` fixes time zero for each trial (via the spike/frame alignment), and the input values are the centres of the 500 bins spanning −2.5 to +2.5 s around it. It is stored as a single continuous, time-varying channel named `time_from_go_cue`.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_centers = edges[:-1] + DT / 2
...
'input_names': ['time_from_go_cue'],
```

iii. The instructions define this input ("Time from go cue onset in seconds (continuous, time-varying)"), and the window endpoints come from `params.tmin`/`params.tmax` in the reference code.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the bin-centre vector once per session and replicating it, as a `(1, 500)` `float32` array, for every kept trial. The identical vector is used for all sessions, so the input carries no session- or trial-specific information; its range is −2.495 to +2.495 s.

ii.
```python
taxis = time_centers
...
for ti in valid_trials:
    neural_trials.append(trialdat[:, :, ti].T.astype(np.float32))
    input_trials.append(taxis.reshape(1, -1).astype(np.float32))
```

iii. Not discussed; the time axis is defined by the AI rather than read from the data.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `edges` is used both for the spike histogram and to derive `time_centers`, and `time_centers` is what is stored as the input, so bin *k* of `input` is exactly the interval that bin *k* of `neural` counts spikes in. The same `taxis` is also the interpolation target for the three camera outputs, so all four streams share one time axis by construction.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_centers = edges[:-1] + DT / 2
...
counts, _ = np.histogram(aligned_times, bins=edges)     # neural uses the edges
...
input_trials.append(taxis.reshape(1, -1).astype(np.float32))   # input is the centres
```

iii. Not discussed explicitly; it follows from `obj.time` in the reference pipeline being the single time axis shared by `obj.trialdat` and the interpolated video features.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial Bpod flags: `bp.hit`, `bp.miss` and `bp.R` (the instructed/rewarded side). `bp.L` is loaded but is not used in the final formula, and `bp.no` is not used either. The animal's actual lick side is not stored in the file, so it is inferred from the instructed side combined with whether the trial was scored correct or incorrect.

ii.
```python
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
no = np.array(bp['no']).flatten().astype(bool)
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
```

iii. The AI's first attempt used `R`/`L` directly and produced no `none` class at all, which it caught from the verifier output. Its reasoning: "`R` and `L` indicate the cue direction (which side has reward), not the animal's actual lick... Since R and L together cover every trial, 'none' must actually refer to the animal not licking at all on ignore trials". It also noted the WC caveat — "for WC trials this logic may not hold cleanly, since water is delivered at a random port and 'hit' just means the animal drank from wherever it was presented" — and decided the animal's actual lick is still what should be decoded.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A three-way relabelling, constant over the whole trial: on a hit the animal licked the instructed port, so the class is the instructed side; on a miss it licked the other port, so the class is the opposite side; on any other trial (an ignore) the class is `none`. Codes are left 0, right 1, none 2, matching `output_values = ['left', 'right', 'none']`. The value is broadcast across all 500 bins.

ii.
```python
# Lick direction: 0=left, 1=right, 2=none
# R/L indicate cue direction. For hit trials, animal licked same direction.
# For miss trials, animal licked opposite direction. For ignore, no lick.
if sess['hit'][ti]:
    lick_dir = 1 if sess['R'][ti] else 0
elif sess['miss'][ti]:
    lick_dir = 0 if sess['R'][ti] else 1  # opposite of cue
else:
    lick_dir = 2  # ignore/no response
...
out[0, :] = lick_dir
```

iii. As above: "Hit: animal licked correctly → same as cue (R→right, L→left); Miss: animal licked incorrectly → opposite of cue (R→left, L→right); Ignore: animal didn't lick → none." The resulting distribution is 42.2% left, 44.5% right, 13.3% none.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial Bpod flag, `bp.autowater`, which marks trials on which water was delivered irrespective of the animal's choice.

ii.
```python
autowater = np.array(bp['autowater']).flatten().astype(bool)
```

iii. Straight from `WorkingWithDataObjs.m`, which the AI read: "obj.bp.autowater=1 when water was delivered regardless of animal choice (0 otherwise). this field can be used as a proxy for obtaining water-cued blocks and delayed-response blocks of trials".

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct binary relabelling held constant over the trial: autowater → WC (0), otherwise → DR (1), matching the instructions' ordering and `output_values = ['WC', 'DR']`. 9.7% of bins are WC, 90.3% DR, and 12 of the 44 sessions contain only DR trials (the DR-only and randomized-delay sessions), which is consistent with the paper's "12 sessions, six mice" two-context subset.

ii.
```python
# Context: 0=WC, 1=DR
context = 0 if sess['autowater'][ti] else 1
...
out[1, :] = context
```

iii. Codes follow the instructions' `(WC, DR)` ordering; the mapping from `autowater` follows the tutorial comment quoted above.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial Bpod flags, `bp.hit` and `bp.miss`. `bp.no` is loaded but not used — a trial that is neither a hit nor a miss is an ignore by construction.

ii.
```python
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
no = np.array(bp['no']).flatten().astype(bool)
```

iii. The AI's reasoning records the mapping "outcome categorized as hit, miss, or ignore for correct, incorrect, and no-response respectively", taken from the Bpod flag semantics documented in the tutorial.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling held constant over the trial: hit → correct (1), miss → incorrect (0), everything else → ignore (2), matching the instructions' ordering and `output_values = ['incorrect', 'correct', 'ignore']`. Ignore trials are retained as their own class rather than dropped, even though the paper omits them from its own analyses. Distribution: 12.0% incorrect, 74.7% correct, 13.3% ignore — the ignore fraction matches the `none` lick-direction fraction exactly, as it must.

ii.
```python
# Outcome: 0=incorrect, 1=correct, 2=ignore
outcome = 1 if sess['hit'][ti] else (0 if sess['miss'][ti] else 2)
...
out[2, :] = outcome
```

iii. Class ordering is taken from the instructions.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut output in `obj.traj{2}` — the **bottom camera only**. Per trial it reads `ts`, whose three planes are x, y and DLC confidence, for the feature named `top_tongue` (found via `featNames`), plus `frameTimes`. The side camera (`obj.traj{1}`, whose tongue feature is called `tongue`) is never loaded. Alignment additionally needs `sglx.bitcode.bitstart`, `sglx.fs` and `bp.ev.bitStart` (see 7-d).

ii.
```python
    # Load trajectory (bottom cam for tongue/paw)
    traj_data = obj['traj']
    bottom_ref = traj_data[1, 0]
    bottom = f[bottom_ref]
    ...
    tongue_idx = bottom_feat_names.index('top_tongue') if 'top_tongue' in bottom_feat_names else None
    paw_idx = bottom_feat_names.index('top_paw') if 'top_paw' in bottom_feat_names else None
    ...
        ts = np.array(f[ts_ds[t, 0]])  # (n_bodyparts, 3, n_frames)
        if tongue_idx is not None:
            trial_tongue_xy.append(ts[tongue_idx, :2, :].T)
            trial_tongue_conf.append(ts[tongue_idx, 2, :])
```

iii. The AI enumerated the features of both cameras (step 46) and then chose the bottom camera because it carries both required features, `top_tongue` and `top_paw`, in one view ("Load trajectory (bottom cam for tongue/paw)"). It did not state a reason for preferring bottom over side for the tongue, nor did it consider combining the two views. The paper does support the bottom view as the canonical tongue view ("Tongue angle and length were found using the bottom camera"), though it also says the tongue was tracked with both cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps. **(1) Visibility:** a frame counts as visible only if the DLC confidence is ≥ 0.9; the AI verified the tongue confidence is bimodal and low most of the time ("mean 0.17, 84% below 0.5"). **(2) Velocity:** the first-order difference of x and y between consecutive frames, scaled by the fixed 400 Hz frame rate, taken as `sqrt(dx² + dy²)`; the first sample is duplicated so the vector keeps its length. The position is **not** smoothed first, and missing values are **not** nearest-filled for the tongue. Velocity at non-visible frames is set to NaN. **(3) Onto the grid:** the frame-resolution velocity is linearly interpolated (`interp1d`, `fill_value=np.nan` outside) at the 500 bin centres — i.e. sampled, not averaged over the ~4 frames in each bin — and the boolean visibility is nearest-neighbour interpolated onto the same centres. **(4)** The result is split at the session median (7-c). Tongue NaNs are deliberately *not* filled.

ii.
```python
def compute_velocity(xy_coords, confidence=None, fill_missing=True):
    ...
    if confidence is not None:
        visible = confidence >= DLC_CONF_THRESH
    ...
    dx = np.diff(xy[:, 0])
    dy = np.diff(xy[:, 1])
    vel = np.sqrt(dx**2 + dy**2) * VIDEO_FPS
    vel = np.concatenate([[vel[0]], vel])
    # Set velocity to NaN where not visible
    vel[~visible] = np.nan
    return vel, visible
```
```python
        # Tongue velocity (do NOT fill missing - per methods: "except for the tongue")
        tongue_xy = sess['trial_tongue_xy'][ti]
        tongue_conf = sess['trial_tongue_conf'][ti]
        if tongue_xy is not None and len(tongue_xy) > 1:
            vel, vis = compute_velocity(tongue_xy, confidence=tongue_conf, fill_missing=False)
            if vel is not None:
                try:
                    f_interp = interp1d(ft_aligned, vel, kind='linear',
                                        bounds_error=False, fill_value=np.nan)
                    v = f_interp(taxis)
                    tongue_vel_aligned[:, ti] = v
                    vis_float = vis.astype(float)
                    f_vis = interp1d(ft_aligned, vis_float, kind='nearest',
                                     bounds_error=False, fill_value=0)
                    tongue_visible[:, ti] = f_vis(taxis) > 0.5
                except Exception:
                    pass
```
```python
    # Fill NaNs with nearest for paw and ME (not tongue - per methods)
    fill_nearest(paw_vel_aligned)
    fill_nearest(me_aligned)
```

iii. The velocity definition is the methods' ("The velocity of each feature was then calculated as the first-order derivative of the position vector") at the stated 400 Hz frame rate. The no-filling rule for the tongue is the methods' too ("Missing values were filled in with the nearest available value for all features, except for the tongue"), which the AI quotes repeatedly. The confidence gate was added after the first run produced a degenerate tongue output: "The tongue velocity is always 1 (>= 50th percentile)... What I actually need is to use the confidence score in the DLC data to determine true visibility rather than treating every interpolated point as visible." The interpolation-onto-`taxis` idiom mirrors `loadMotionEnergy.m`, which does `interp1(frameTimes-vidshift-alignTimes, data, taxis)`.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session, the 50th percentile of the binned tongue velocity pooled over all bins of all kept trials, restricted to bins that are both visible and non-NaN. Bins below the threshold get 0, bins at or above get 1, and every bin that is not visible (or whose velocity is NaN) gets 2. Labels are `['below_50pct', 'above_50pct', 'not_visible']`. If a session yields no visible bins at all the threshold falls back to 0. Resulting distribution: 4.50% class 0, 4.50% class 1, 91.0% class 2 — the ~50/50 split within visible bins that the specification asks for.

ii.
```python
    # Per-session velocity thresholds (50th percentile on valid trials)
    t_vals = tongue_vel_aligned[:, valid_trials][tongue_visible[:, valid_trials]]
    t_vals = t_vals[~np.isnan(t_vals)]
    tongue_thresh = np.percentile(t_vals, 50) if len(t_vals) > 0 else 0
```
```python
        tongue_disc = np.full(n_timebins, 2, dtype=np.int64)
        vis = tongue_visible[:, ti] & ~np.isnan(tongue_vel_aligned[:, ti])
        if vis.any():
            tongue_disc[vis & (tongue_vel_aligned[:, ti] < tongue_thresh)] = 0
            tongue_disc[vis & (tongue_vel_aligned[:, ti] >= tongue_thresh)] = 1
```

iii. The three-class scheme and the 50th-percentile, per-session threshold are taken verbatim from the instructions. The AI confirmed the output afterwards: "Now tongue velocity has all three classes: below_50pct (4.5%), above_50pct (4.5%), not_visible (91%). Tongue is mostly not visible, which is correct - the tongue is only visible when the mouse is licking."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the Bpod clock, so a per-session offset `vidshift` is computed once from the bitcode pulse that both systems record: `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`. Each trial's frame times then become `frameTimes − vidshift − goCue[trial]`, i.e. seconds from the go cue on the same clock as the spikes, and the velocity is interpolated at the shared bin centres `taxis`. Frames outside the window produce NaN via `fill_value=np.nan`, which becomes the `not_visible` class.

ii.
```python
    bitcode_bs = np.array(sglx['bitcode']['bitstart']).flatten()
    sglx_fs = np.array(sglx['fs']).flatten()[0]
    ev_bs = np.array(ev['bitStart']).flatten()
    vidshift = scipy_mode(bitcode_bs, keepdims=False).mode / sglx_fs - scipy_mode(ev_bs, keepdims=False).mode
```
```python
    for ti in range(ntrials):
        ft = sess['trial_frame_times'][ti]
        if len(ft) == 0:
            continue
        ft_aligned = ft - vidshift - goCue[ti]
```

iii. This is `findVideoOffset.m`; the AI documented it in its header ("Video offset: computed from sglx.bitcode.bitstart / sglx.fs - mode(ev.bitStart)") and sanity-checked the value against the tutorial's hard-coded constant: "the video offset is approximately 0.49s, close to the 0.5s mentioned in the tutorial" (`WorkingWithDataObjs.m` uses `frameTimes-0.5`). The offset is computed once per session rather than per trial.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj{2}` bottom-camera DLC output, feature `top_paw` — its x, y and confidence planes, plus `frameTimes`. The bottom view also carries `bottom_paw`, which is not used. The paper confirms the paws are tracked only from the bottom view.

ii.
```python
paw_idx = bottom_feat_names.index('top_paw') if 'top_paw' in bottom_feat_names else None
...
if paw_idx is not None:
    trial_paw_xy.append(ts[paw_idx, :2, :].T)
    trial_paw_conf.append(ts[paw_idx, 2, :])
```

iii. Not justified explicitly beyond the choice of bottom camera; the AI picked the first of the two paw features listed in `featNames`. The methods support the camera choice: "The tongue, jaw and nose were tracked using both cameras, whereas the paws were tracked using only the bottom view."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same velocity routine as the tongue — confidence ≥ 0.9 for visibility, first-order difference of x and y scaled by 400 Hz — but with `fill_missing=True`, so missing/NaN coordinates are first replaced by the nearest valid frame's value. The result is linearly interpolated onto the bin centres, and then any remaining NaN bins are nearest-filled **along time within each trial** by `fill_nearest`. Finally the visibility mask is widened to include every bin that now holds a number: `paw_visible = paw_visible | ~np.isnan(paw_vel_aligned)`. The net effect is that the DLC confidence gate is undone for the paw and only a trial with no valid paw frame at all remains "not visible". No normalisation; units are pixels/s.

ii.
```python
    if fill_missing:
        # Fill missing/NaN values with nearest (for non-tongue features)
        for col in range(xy.shape[1]):
            mask = np.isnan(xy[:, col])
            if mask.all():
                return None, None
            if mask.any():
                valid_idx = np.where(~mask)[0]
                for idx in np.where(mask)[0]:
                    nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
                    xy[idx, col] = xy[nearest, col]
```
```python
            vel, vis = compute_velocity(paw_xy, confidence=paw_conf, fill_missing=True)
```
```python
    fill_nearest(paw_vel_aligned)
    ...
    # Update paw visibility after filling (tongue keeps DLC-based visibility)
    paw_visible = paw_visible | ~np.isnan(paw_vel_aligned)
```

iii. The filling is the methods' rule for all non-tongue features ("Missing values were filled in with the nearest available value for all features, except for the tongue"), and matches `fillmissing(...,'nearest')` in the reference `loadMotionEnergy.m`. The AI made the tongue/paw asymmetry explicit in its comments and in its reasoning: "The reference code notes that missing values are filled for all features except the tongue, which should be treated differently since it's frequently not visible."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: one per-session threshold, the 50th percentile of binned paw velocity over the kept trials' visible, non-NaN bins; below → 0, at/above → 1, otherwise 2 (`['below_50pct', 'above_50pct', 'not_visible']`). Because of the nearest-filling in 8-b, almost every bin is treated as visible, so the distribution is 49.1% / 49.2% / 1.7% — the `not_visible` class survives only for trials whose paw was never tracked.

ii.
```python
    p_vals = paw_vel_aligned[:, valid_trials][paw_visible[:, valid_trials]]
    p_vals = p_vals[~np.isnan(p_vals)]
    paw_thresh = np.percentile(p_vals, 50) if len(p_vals) > 0 else 0
```
```python
        paw_disc = np.full(n_timebins, 2, dtype=np.int64)
        vis = paw_visible[:, ti] & ~np.isnan(paw_vel_aligned[:, ti])
        if vis.any():
            paw_disc[vis & (paw_vel_aligned[:, ti] < paw_thresh)] = 0
            paw_disc[vis & (paw_vel_aligned[:, ti] >= paw_thresh)] = 1
```

iii. The scheme is the instructions'; the near-absence of the third class is a consequence of following the paper's nearest-filling rule for non-tongue features, not a separate decision. The AI did not comment on the resulting 1.7% `not_visible` rate.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as the tongue, and in the same per-trial loop: the session-constant `vidshift` and the trial's `goCue` are subtracted from the bottom camera's `frameTimes`, and the velocity is interpolated at the shared `taxis` bin centres. Since the paw comes from the bottom camera and the frame times read are the bottom camera's, the feature and its clock come from the same view.

ii.
```python
        ft_aligned = ft - vidshift - goCue[ti]
        ...
            f_interp = interp1d(ft_aligned, vel, kind='linear',
                                bounds_error=False, fill_value=np.nan)
            v = f_interp(taxis)
            paw_vel_aligned[:, ti] = v
```

iii. Same offset and same grid as every other stream; no separate treatment was considered necessary.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. A separate file per session, `motionEnergy_<anm>_<date>.mat`, sitting beside the data structure. The AI reads `me['data']` (a cell array with one trace per trial, one value per camera frame) and `me['moveThresh']` (the authors' manual per-session movement threshold, which it stores but never uses). `obj.me`, present in some sessions, is not used. The load is wrapped in nested `try`/`except` blocks — scipy first, h5py second — and if both fail, `me_data` is left `None` and the whole session's motion energy becomes the `no_video` class.

ii.
```python
    if os.path.exists(me_fpath):
        try:
            me_mat = sio.loadmat(me_fpath, squeeze_me=True)
            me = me_mat['me']
            me_data_raw = me['data'].item()
            sess['me_thresh'] = float(me['moveThresh'].item())
            sess['me_data'] = [me_data_raw[i] for i in range(len(me_data_raw))]
        except Exception:
            try:
                mf = h5py.File(me_fpath, 'r')
                ...
            except Exception:
                pass
```

iii. The AI read `loadMotionEnergy.m` (step 25) and followed its overall shape. It did not, however, carry over that function's layout guard, `if isstruct(me.data), me.data = me.data.data; end`. The motion-energy files come in three layouts across the 44 sessions and this loader handles only one of them: it raises on the three files that wrap the payload twice (`me.data.data`: JEB15_2022-07-26, JEB15_2022-07-28, JEB24_2023-10-31) and on the four that store a bare cell array with no `moveThresh` (JEB23_2023-10-10 through 2023-10-13). The exception is swallowed, and the h5py fallback also fails because these are v5 files, so all 7 sessions silently lose motion energy entirely — which is why 16.9% of all bins carry the `no_video` class.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond putting it on the time grid. The value is already one scalar per camera frame (the paper computes it per pixel as the difference of medians over the next and previous five frames, then reduces each frame to its 99th percentile across pixels), so the AI linearly interpolates the trace at the 500 bin centres and nearest-fills the remaining NaNs along time within each trial. It is not smoothed, differentiated or rescaled, and the file's own `moveThresh` is not applied.

ii.
```python
        # Motion energy
        if me_available and ti < len(sess['me_data']):
            me_trial = sess['me_data'][ti]
            if len(me_trial) == len(ft):
                try:
                    f_interp = interp1d(ft_aligned, me_trial, kind='linear',
                                        bounds_error=False, fill_value=np.nan)
                    me_aligned[:, ti] = f_interp(taxis)
                except Exception:
                    pass
```
```python
    fill_nearest(me_aligned)
```

iii. This mirrors `loadMotionEnergy.m` closely, which does `interp1(frameTimes-vidshift-alignTimes, me.data{trix}, taxis)` followed by `fillmissing(me.data,'nearest')` with the comment "fill nans with nearest value (there are some nans at the start of each trial)". The AI additionally guards on `len(me_trial) == len(ft)` and skips the trial otherwise.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. One threshold per session, the 50th percentile of the binned motion energy over all non-NaN bins of the kept trials; below → 0, at/above → 1, and any bin with no value → 2 (`['below_50pct', 'above_50pct', 'no_video']`). The authors' own bimodal `moveThresh` is deliberately not used, since the instructions specify a median split. For the 7 sessions whose motion-energy file failed to load, `me_available` is False and every bin is class 2. Distribution: 41.5% / 41.6% / 16.9%.

ii.
```python
    me_thresh = 0
    if me_available:
        m_vals = me_aligned[:, valid_trials]
        m_vals = m_vals[~np.isnan(m_vals)]
        if len(m_vals) > 0:
            me_thresh = np.percentile(m_vals, 50)
```
```python
        me_disc = np.full(n_timebins, 2, dtype=np.int64)
        if me_available:
            valid_me = ~np.isnan(me_aligned[:, ti])
            if valid_me.any():
                me_disc[valid_me & (me_aligned[:, ti] < me_thresh)] = 0
                me_disc[valid_me & (me_aligned[:, ti] >= me_thresh)] = 1
```

iii. The per-session 50th-percentile split and the `no_video` third class are the instructions'. The AI's `output_values` name the third class `no_video`, exactly as the task specifies.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. With the same `vidshift` and the same `taxis` as the tongue and paw, inside the same per-trial loop, using the **bottom** camera's `frameTimes` — the only frame times the AI loads. A trial's motion energy is used only if its length equals that frame-time vector's length; otherwise the trial is skipped and its bins become `no_video`.

ii.
```python
        ft = sess['trial_frame_times'][ti]     # bottom camera
        if len(ft) == 0:
            continue
        ft_aligned = ft - vidshift - goCue[ti]
        ...
            if len(me_trial) == len(ft):
                f_interp = interp1d(ft_aligned, me_trial, kind='linear',
                                    bounds_error=False, fill_value=np.nan)
                me_aligned[:, ti] = f_interp(taxis)
```

iii. Not discussed. The reference `loadMotionEnergy.m` uses `obj.traj{1}` — the side camera — for this interpolation. In the sessions checked, the two cameras' `frameTimes` are byte-identical (0 length mismatches and 0 s maximum difference over 60 trials of JEB6_2021-04-18), so in practice the substitution is harmless, and the explicit length guard prevents a misaligned interpolation if they ever diverged.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, all handled defensively and all silently. **Untracked frames:** DLC confidence < 0.9 marks a frame not visible; for the tongue the bin becomes class 2, for the paw and motion energy the gap is nearest-filled. **Empty frame times:** `if len(ft) == 0: continue` leaves that trial's three camera outputs as NaN → class 2. **Fully untracked features:** `compute_velocity` returns `(None, None)` when a coordinate column is entirely NaN, leaving class 2. **Interpolation failures** (e.g. NaN frame times) are caught by bare `except Exception: pass`, again leaving class 2. **Motion-energy length mismatch:** the trial is skipped. **Unreadable motion-energy files:** swallowed, whole session becomes `no_video`. **Unreadable session files:** reported and the session is skipped. Two per-session guards drop a session with < 2 usable trials or < 10 units. Globally, `warnings.filterwarnings('ignore')` is set at import. Nothing is ever dropped at trial level for missing video; the trial is kept and the gap is encoded in the class.

ii.
```python
import warnings
warnings.filterwarnings('ignore')
```
```python
        except Exception as e:
            print(f"  ERROR loading {anm}_{date}: {e}")
            continue
```
```python
    for ti in range(ntrials):
        ft = sess['trial_frame_times'][ti]
        if len(ft) == 0:
            continue
```
```python
                except Exception:
                    pass
```

iii. Keeping the trial and marking the gap is the right call given that the neural and behavioural data are unaffected by a video dropout, and it is what the `not_visible`/`no_video` classes exist for. The AI never states this rationale, however, and the uniformly silent failure mode has a real cost: the same `except Exception: pass` that absorbs a NaN frame time also absorbs the motion-energy loader's failure on 7 of 44 sessions (9-a), and nothing in the run output flags it. The one anomaly that *was* surfaced — 61 trials of all-zero neural data past the end of the recording — was seen in the verifier output and left unaddressed.

## 11-a. What are the most time-consuming steps of the code?

i. The dominant cost is spike binning: a doubly-nested Python loop over units × trials in which, for every (unit, trial) pair, the unit's entire spike vector is scanned with a boolean mask, then histogrammed and smoothed one 500-sample trial at a time. For a session with ~100 units and ~400 trials that is 40,000 full-array scans plus 40,000 separate `np.convolve` calls. Second is reading the files themselves — HDF5 traversal of thousands of per-cluster and per-trial object references. Third is `fill_nearest`, which for every NaN entry does an `argmin` over all valid indices of that column, i.e. O(n²) per column in pure Python. The AI never timed or profiled the pipeline.

ii.
```python
    for ui in range(n_units):
        spike_trials_raw = sess['spike_trials'][ui]
        spike_trialtm_raw = sess['spike_trialtm'][ui]

        for ti in range(ntrials):
            spk_mask = spike_trials_raw == (ti + 1)
            if not np.any(spk_mask):
                continue
            aligned_times = spike_trialtm_raw[spk_mask] - goCue[ti]
            counts, _ = np.histogram(aligned_times, bins=edges)
            fr = counts / DT
            trialdat[:, ui, ti] = smooth_data(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. Not discussed in the trajectory; no efficiency rationale is given anywhere in the code or reasoning.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. **(1) The spike loop** is the big one: all spikes of a unit could be binned across every trial at once with a single 2-D histogram over (trial, time-from-go-cue), eliminating the inner loop and the repeated masking entirely, and the smoothing could be applied to the whole (time × trials) matrix in one call since `smooth_data` already handles 2-D input along axis 0. **(2) `fill_nearest`'s** per-NaN `argmin` search is a classic `np.searchsorted`/forward-fill pattern. **(3) The nearest-fill inside `compute_velocity`** repeats the same O(n²) pattern per coordinate column. The per-trial video loop genuinely cannot be vectorised, because each trial has a different number of frames. Separately, `{q.lower() for q in EXCLUDED_QUALITIES}` is rebuilt on every cluster iteration instead of once at module level.

ii.
```python
def fill_nearest(arr):
    ...
        valid = np.where(~nans)[0]
        for idx in np.where(nans)[0]:
            nearest = valid[np.argmin(np.abs(valid - idx))]
            arr[idx] = arr[nearest]
```
```python
            if q_str.lower() in {q.lower() for q in EXCLUDED_QUALITIES}:
```

iii. Not discussed. These are pure performance issues; none of them changes the numbers that come out.

## 11-c. What processing does the code repeat multiple times?

i. Four repeats. **(1)** The camera streams — tongue velocity, paw velocity and motion energy — are computed for *all* `ntrials`, then only the `valid_trials` columns are ever read; the early-lick and photostim trials' video is processed and thrown away. **(2)** Likewise, spikes are binned and smoothed for *every* cluster that passed the quality label, including the ones the 1 Hz filter then discards. **(3)** The paw's missing values are nearest-filled twice: once at frame resolution inside `compute_velocity`, where the fill is then immediately undone by `vel[~visible] = np.nan`, and again at bin resolution by `fill_nearest`. **(4)** The lower-cased quality set is rebuilt per cluster. The video offset, in contrast, is correctly computed once per session, and each file is opened once.

ii.
```python
    for ti in range(ntrials):            # all trials, not just valid_trials
        ft = sess['trial_frame_times'][ti]
```
```python
    trialdat = np.zeros((n_timebins, n_units, ntrials))
    for ui in range(n_units):            # all quality-passing units
        ...
    mean_frs = np.mean(trialdat[:, :, valid_trials], axis=(0, 2))
    good_units = mean_frs > LOW_FR_THRESH
    trialdat = trialdat[:, good_units, :]
```

iii. Not discussed. Processing all trials before subsetting keeps the indexing simple (every array is indexed by raw trial number), which is a reasonable trade, but it is redundant work.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Five things. **(1)** `bp.no` and `bp.L` are loaded in both readers and never used — outcome and lick direction are derived from `hit`, `miss` and `R` alone. **(2)** `me['moveThresh']`, the authors' manual per-session movement threshold, is loaded and stored on the session dict but never read, since the instructions mandate a median split. **(3)** `all_qualities` is accumulated per cluster and never used after filtering. **(4)** The nearest-fill of the paw's x/y inside `compute_velocity` is nullified two lines later by `vel[~visible] = np.nan`. **(5)** As in 11-c, binned and smoothed rates are computed for units that the firing-rate filter then drops, and all three camera streams are computed for trials that the trial mask then drops (roughly 9% of trials). Everything that survives into `data` is used.

ii.
```python
            sess['me_thresh'] = float(me['moveThresh'].item())
```
```python
            all_qualities.append(q_str)
```
```python
    if fill_missing:
        ... xy[idx, col] = xy[nearest, col]
    ...
    vel[~visible] = np.nan     # discards the fill that was just done
```

iii. Not discussed. None of this affects the output values; it is wasted work, except for (4), which is a small internal inconsistency between the stated intent ("fill missing with nearest for non-tongue features") and where the fill actually takes effect (later, at bin resolution, via `fill_nearest`).
