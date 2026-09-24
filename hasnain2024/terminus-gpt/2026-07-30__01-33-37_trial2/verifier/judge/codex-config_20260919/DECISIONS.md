# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every `data_structure_*.mat` and `motionEnergy_*.mat` pair under every `/app/data` subdirectory. It uses a custom `h5py` reader for HDF5 data structures and `scipy.io.loadmat` for motion energy; unreadable or incomplete sessions are skipped. This produced 33 sessions, rather than using the authors' curated session/probe list.

ii.
```python
for subdir in Path(data_root).iterdir():
    ...
    for f in subdir.glob('data_structure_*.mat'):
        sessions.setdefault(stem, {})['data_structure'] = f
    for f in subdir.glob('motionEnergy_*.mat'):
        sessions.setdefault(stem, {})['motion_energy'] = f
...
keys = sorted(k for k,v in sessions.items()
              if 'data_structure' in v and 'motion_energy' in v)
```

iii. The notes say conversion should be restricted to sessions with all required neural, behavior, trajectory, and motion-energy outputs. They also document skipping unreadable files and sessions without trial-aligned spikes. They do not justify replacing the authors' curated inclusion/probe list with glob discovery.

## 1-b. How are the data split into subjects?

i. Subject identity is parsed from the filename stem before the first underscore; unique subjects are accumulated in first-kept-session order and each session gets an index into that list.

ii.
```python
def infer_subject_session(stem):
    parts = stem.split('_')
    return parts[0], '_'.join(parts[1:])
...
if sess['subject'] not in subject_to_idx:
    subject_to_idx[sess['subject']] = len(subjects)
    subjects.append(sess['subject'])
```

iii. The notes identified subject/date patterns in filenames and used those because the filenames consistently encode the animal.

## 1-c. How are the data split into sessions?

i. Each filename stem `<subject>_<date>` with both required files is treated as one session. Sessions failing loading, modality, trial, or unit checks are discarded; 33 remain.

ii.
```python
for stem in keys:
    sess = process_session(stem, sessions[stem], edges, centers)
    if sess is None:
        continue
    out_sessions.append(sess)
```

iii. The agent reasoned that every retained session must contain all decoder outputs and at least two trials and ten units. Its README reports 33 retained sessions.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the trial count. Raw trial indices passing a Boolean mask select spikes, labels, tracking entries, and motion-energy traces; each retained raw trial becomes one matrix in each session list.

ii.
```python
n = int(np.array(bp['Ntrials']).reshape(-1)[0])
mask = np.ones(n, dtype=bool)
...
idx = np.where(trial_mask)[0]
...
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. The notes state that trial-wise behavioral arrays, cluster trial IDs, trajectory cells, and motion-energy cells all share the native Bpod trial index.

## 1-e. How are trials filtered based on quality controls?

i. Trials are removed when `bp.early != 0` or `bp.no != 0`. Photostimulation trials are not removed, and trials after ephys recording ends are not explicitly removed. Entire sessions are also rejected if fewer than two retained trials or ten retained units remain.

ii.
```python
for name in ['early', 'no']:
    if name in bp:
        ...
        mask &= (arr == 0)
...
if len(neural) < 2 or len(kept_units) < 10:
    return None
```

iii. The notes cite the paper's omission of early and ignore trials and a reference requirement of at least ten units. They overlook the decoder request for an `ignore` outcome class and the reference conversion's exclusion of photostimulation, not ignore, trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices come from each cluster's `obj.clu.trialtm` spike times and `obj.clu.trial` trial IDs. Although `quality` is loaded, it is not used. `bp.ev.goCue` is not used in neural construction.

ii.
```python
trialtm = clu['trialtm']
trialid = clu['trial']
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    st = np.array(st, dtype=float).reshape(-1)
    tr = np.array(tr, dtype=float).reshape(-1).astype(int) - 1
```

iii. The notes correctly identify trial spike times/IDs as the source and claim the output will be go-cue aligned, but the implementation never brings go-cue times into this function.

## 2-b. How is the `neural` data processed?

i. For each retained unit and trial, raw spike times are histogrammed into 75 ms bins. Values remain spike counts (`float32`); there is no conversion to Hz, smoothing, normalization, or baseline subtraction.

ii.
```python
counts, _ = np.histogram(st[tr == raw_t], bins=edges)
unit_trials.append(counts.astype(np.float32))
```

iii. The agent chose fixed-grid spike counts and tied 75 ms to `DLC_ContextDecoding.m`. It did not account for the common 5 ms stream or the reference neural Gaussian smoothing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are retained when `number of spikes / (5 seconds × number of retained trials) > 1 Hz`; sessions then need at least ten such units. Manual cluster-quality labels are ignored. The rate numerator includes all stored spikes, not just retained trials or the five-second window.

ii.
```python
approx_duration = (edges[-1] - edges[0]) * max(trial_mask.sum(), 1)
fr = st.size / max(approx_duration, 1e-9)
if fr <= 1.0:
    continue
...
if ... len(kept_units) < 10:
    return None
```

iii. The notes cite the paper's `>1 Hz` and `>=10 units` rules, but do not justify ignoring `clu.quality` or using all spikes in the numerator.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The intended decision was go-cue alignment on a −2.5 to +2.5 s grid. In code, however, `trialtm` is histogrammed directly and `bp.ev.goCue` is never subtracted, so the neural data are not go-cue aligned.

ii.
```python
edges, centers = build_time_grid()
...
counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. The notes repeatedly identify go cue as the reference event and report a raw-versus-converted histogram check, but that check reproduced the same unaligned histogram rather than validating event subtraction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. A nominal 75 ms grid from −2.5 to +2.5 s is used. `np.arange` yields 66 bins and stops at 2.45 s; metadata therefore reports `off_end=2.45`. Spikes are histogrammed, while video and motion energy are interpolated to bin centers.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
...
'time_bin_size': 75.0,
```

iii. The agent inferred 75 ms from one DLC decoder's analysis bins. The task's shared stream and reference conversion instead use the native 5 ms resolution.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a synthetic vector made from the centers of the fixed nominal go-cue window, not directly calculated from a raw field for each trial.

ii.
```python
edges, centers = build_time_grid()
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The notes say the only decoder input should be a continuous time channel shared across trials, as required.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers are calculated as each left edge plus 37.5 ms and copied identically into every retained trial.

ii.
```python
edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
centers = edges[:-1] + bin_size_s / 2
```

iii. The agent treated the aligned time axis as a constructed decoder variable; this is appropriate in principle.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It has the same number of bins as the neural matrices, but only the input is expressed as nominal time from go cue. Because neural spike times do not subtract the trial's go cue, semantic alignment is absent.

ii.
```python
counts, _ = np.histogram(st[tr == raw_t], bins=edges)
...
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The notes claim both are aligned to a common grid, but no explicit check of `trialtm - goCue` was performed.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived only from the instructed-side flags `bp.L` and `bp.R`, after ignore trials have been removed. Hit/miss is not used to recover the actual lick side.

ii.
```python
L = np.array(bp['L'], dtype=float).reshape(-1)
R = np.array(bp['R'], dtype=float).reshape(-1)
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
```

iii. The mapping plan labels `bp.L`/`bp.R` as lick direction and says invalid trials are excluded. This confuses instruction with response on incorrect trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Right-instructed trials become 1 and all other retained trials become 0, then the value is repeated across time. There is no `none` class because ignore trials were dropped, and miss trials are not flipped to their actual response side.

ii.
```python
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
...
np.full(n_bins, lick_dir[i], dtype=np.int64)
```

iii. The notes justify a left/right per-trial output, but do not address the requested `none` category or incorrect-response direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from the per-trial `bp.autowater` flag.

ii.
```python
autowater = np.array(bp['autowater'], dtype=float).reshape(-1)
```

iii. The agent found that the authors' context grouping is encoded through autowater transitions/trial groups.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater trials are coded WC=0 and other trials DR=1, with the per-trial code repeated across bins.

ii.
```python
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
```

iii. The notes explicitly planned this WC/DR mapping and checked selected converted labels against raw behavior.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp.hit` and `bp.miss`, but `bp.no` has already been used to eliminate ignore trials.

ii.
```python
hit = np.array(bp['hit'], dtype=float).reshape(-1)
miss = np.array(bp['miss'], dtype=float).reshape(-1)
```

iii. The mapping plan cites hit/miss and intentionally excludes ignore trials following paper analyses, despite the decoder specification requiring ignore as a category.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A retained trial is correct=1 if `hit > miss`, otherwise incorrect=0; the label is repeated over time. No ignore=2 output exists.

ii.
```python
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
```

iii. The agent prioritized the paper's analysis exclusion over the explicit target output. Its README consequently documents only incorrect/correct.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity uses only the side-camera `obj.traj` feature named `tongue` (or fallback name), its `ts` coordinates and `frameTimes`, plus `bp.ev.goCue`. The bottom-camera tongue is not used, nor are likelihood or bitcode clock fields.

ii.
```python
tongue_idx = pick_feature(side_names, ['tongue', 'left_tongue', 'right_tongue'])
...
tongue.append(interp_feature_velocity(
    ts_side, ft_side, go[tr], centers, tongue_idx, tongue=True))
```

iii. The notes say a two-view loading bug was fixed, but the final code uses the second view only for paw; it does not implement the reference's combination of both tongue views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Side-camera x/y positions are linearly interpolated at output centers after an ad hoc 0.5 s timestamp shift; gradients are taken per bin without dividing by elapsed time, NaN gradients are replaced by zero, and Euclidean magnitude is computed. There is no likelihood filtering, smoothing, contiguous-run handling, or view normalization/combination.

ii.
```python
rel_t = (ft - 0.5) - float(align_time)
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
y = np.interp(centers, rel_t, xy[:, 1], left=np.nan, right=np.nan)
xv = np.gradient(x); yv = np.gradient(y)
if tongue:
    xv[np.isnan(xv)] = 0; yv[np.isnan(yv)] = 0
return np.sqrt(xv**2 + yv**2).astype(np.float32)
```

iii. The notes sought time-varying velocity and acknowledged earlier degenerate outputs, but provide no support for the fixed 0.5 s offset or zero-filling invisible tongue frames.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. All tongue values pooled across retained trials and times in a session are compared with the session `nanmedian`; strictly greater values are high=1, everything else low=0. There is no not-visible=2 class.

ii.
```python
tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
```

iii. The session median follows the requested 50th percentile, but the agent omitted the explicitly required visibility class and instead converted missing samples to zero/low.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code interpolates using `(frameTimes - 0.5) - goCue` at the nominal centers. It does not estimate the video/behavior clock offset from bitcodes. Moreover, the neural stream itself is not go-cue aligned.

ii.
```python
rel_t = (ft - 0.5) - float(align_time)
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
```

iii. The agent intended go-cue alignment, but the notes do not justify the hard-coded 0.5 s correction; the reference derives a session-specific clock offset.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity uses bottom-camera `obj.traj.ts`, `frameTimes`, `bp.ev.goCue`, and preferably the `top_paw` feature. Likelihood and clock-bitcode data are ignored.

ii.
```python
paw_idx = pick_feature(bottom_names, ['top_paw', 'bottom_paw', 'paw'])
...
paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers,
                                   paw_idx, tongue=False))
```

iii. The notes report fixing camera loading by using bottom-view paw, consistent with the reliable paw view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Coordinates are interpolated after the fixed 0.5 s shift, differentiated in samples rather than seconds, NaNs are median-filled, each component is median-centered, and their magnitude is taken. There is no likelihood cut or Gaussian smoothing.

ii.
```python
xv = np.gradient(x); yv = np.gradient(y)
xv = np.where(np.isnan(xv), np.nanmedian(xv), xv)
yv = np.where(np.isnan(yv), np.nanmedian(yv), yv)
xv = xv - np.nanmedian(xv)
yv = yv - np.nanmedian(yv)
```

iii. The agent intended to derive a time-varying DLC velocity, but did not document these deviations from the reference processing.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Values are pooled within session and split by a strict greater-than session median into low=0/high=1. Missing tracking is imputed and receives no not-visible class.

ii.
```python
paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
```

iii. The 50th-percentile threshold follows the prompt, but the missing visibility category does not.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frames use `(frameTimes - 0.5) - goCue` and are interpolated to the nominal 75 ms centers. No measured clock offset is applied, and neural data are not actually go-cue aligned.

ii.
```python
rel_t = (ft - 0.5) - float(align_time)
```

iii. The stated goal was a shared go-cue grid; the hard-coded shift and omitted bitcode correction undermine it.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses the standalone session `motionEnergy_*.mat` field `me.data`, unwrapped through several possible layouts. No corresponding camera timestamps are used.

ii.
```python
me = load_motion_energy(files['motion_energy'])
me_data = normalize_motion_energy_data(me)
```

iii. The notes correctly identify motion energy as a variable-length per-trial trace and the separate file as required.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each entire variable-length trace is linearly stretched/compressed to the output bin count using normalized indices, irrespective of its actual timestamps or analysis window.

ii.
```python
xp = np.linspace(0, 1, arr.size)
xnew = np.linspace(0, 1, n_bins)
return np.interp(xnew, xp, arr).astype(np.float32)
```

iii. The agent recognized rebinning was needed, but did not use side-camera frame times as required for physically meaningful temporal binning.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. All rebinned values in a session are split at the session `nanmedian`, with strict greater-than mapped to high=1 and all others low=0. There is no no-video=2 class.

ii.
```python
me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)
```

iii. The median matches the requested per-session 50th percentile, but the required third category is omitted.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It is not event aligned. Normalized trace position is merely mapped onto 66 nominal centers, with no `frameTimes`, video-clock correction, or go-cue subtraction.

ii.
```python
me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
```

iii. The notes claim a common go-cue grid, but the implementation never associates motion-energy samples with times.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Unreadable or incomplete sessions are skipped. Empty motion traces become zeros; missing/mismatched frame times are fabricated from 400 Hz indices or linear spacing; tongue NaNs become zero and paw NaNs are median-filled; absent paw rejects the session. Missing values are therefore encoded as ordinary low movement, not the requested third classes.

ii.
```python
if arr.size == 0:
    return np.zeros(n_bins, dtype=np.float32)
...
if ft.size == 0:
    ft = np.arange(1, xy.shape[0] + 1, dtype=float) / 400.0
else:
    ft = np.linspace(ft.min(), ft.max(), xy.shape[0])
```

iii. The notes document explicit session skipping as robustness, but do not justify fabricating timestamps or conflating missing video with low velocity.

## 11-a. What are the most time-consuming steps of the code?

i. The agent did not benchmark or identify them. By inspection, HDF5 dereferencing and the nested unit-by-trial spike histograms are likely dominant, followed by per-trial interpolation.

ii.
```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. The run-time tables and “Code inefficiencies identified” sections in `CONVERSION_NOTES.md` were left blank, so there is no agent justification.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested spike loop could be replaced by a two-dimensional histogram per unit. Output assembly and some per-trial operations could also be batched, although ragged camera traces still require some looping.

ii.
```python
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
...
for i in range(len(idx)):
    outputs.append(np.vstack([...]))
```

iii. The agent did not discuss vectorization; its notes only contain empty placeholders for speedups.

## 11-c. What processing does the code repeat multiple times?

i. `np.where(trial_mask)` is recomputed for every unit, each trial's camera data are separately converted/interpolated for tongue and paw, and constant per-trial label/time arrays are repeatedly allocated. Feature lookup and session medians are sensibly computed once.

ii.
```python
for ui, ...:
    ...
    for raw_t in np.where(trial_mask)[0]:
...
for tr in trial_idx:
    ... # side and bottom interpolation
```

iii. No explicit rationale or repeated-processing analysis appears in the notes.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The loader reads several unused fields (`tm`, `site`, `quality`, L, bitRand, multiple events, trajectory filenames/drop counts, metadata), and `kept_units` stores unit IDs only for its length. Conversely, most computed conversion arrays are saved and used.

ii.
```python
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
...
for name in ['bitStart', 'delay', 'goCue', 'lickL', 'lickR', 'reward', 'sample']:
```

iii. The agent did not document unnecessary processing. The broad loader was likely written for exploration and robustness, but those extra fields do not enter the final conversion.
