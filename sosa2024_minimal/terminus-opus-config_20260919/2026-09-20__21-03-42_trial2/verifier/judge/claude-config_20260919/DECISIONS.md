# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are found with a single recursive glob over `sub-*` sub-directories of `/app/data` (152 files, 11 mice). Each file is one session and is opened directly with `h5py` (not `pynwb`), reading the HDF5 groups `processing/behavior/BehavioralTimeSeries` (position, speed, lick, environment, trial number, trial_start, teleport, reward_zone, Reward) and `processing/ophys` (Fluorescence, Neuropil, ImageSegmentation/PlaneSegmentation for `iscell`/`planeIdx`). Sessions are processed independently in a `multiprocessing` pool (8 spawn workers), each writing a per-session cache pickle, which the parent then concatenates into the final dataset. Metadata (subject id, experiment day, date, VR scene) is read from `general/subject/subject_id`, `general/session_id` and `identifier`.

ii.
```python
paths = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
...
jobs = [(p, args.signal, args.cache_dir) for p in paths]
if args.workers > 1:
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        files = pool.map(worker, jobs, chunksize=1)
```
```python
def process_session(path, signal='dff'):
    f = h5py.File(path, 'r')
    ident = f['identifier'][()].decode()
    scene = ident.split('/')[-1]
    subject = f['general/subject/subject_id'][()].decode()
    exp_day = int(f['general/session_id'][()])
    b = f['processing/behavior/BehavioralTimeSeries']
    get = lambda k: b[k + '/data'][:]
```

iii. From the trajectory (steps 11–14, 30–33): the agent first inspected the NWB tree with `pynwb`/`h5py`, then ran a cheap metadata scan over every file to inventory "152 sessions, 11 subjects, ~80 trials/session, ~150-1800 curated cells/session, ~20k frames at 15.5 Hz" before committing to a loader. It used raw `h5py` rather than `pynwb` because the raw fluorescence arrays are ~87 GB total and it wanted to read only the datasets needed, per session, in parallel workers with a per-session disk cache ("Dataset is large ... I need to understand decoder memory handling"). The verified output contains all 152 sessions and all 11 mice.

## 1-b. How are the data split into subjects?

i. The subject is read from the NWB metadata field `general/subject/subject_id` of each session file (e.g. `m11`). The parent process builds `data['subjects']` in first-encounter order over the sorted file list and stores the index of each session's subject in `subject_idx`.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
sub = res['info']['subject']
if sub not in subjects:
    subjects.append(sub)
...
data['subject_idx'].append(subjects.index(sub))
data['subjects'] = subjects
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The agent's inventory scan (step 33) confirmed exactly 11 subjects, matching the paper's 11 switch-condition mice, and it noted which mice are multi-plane (m17, m18) and which are the fixed-condition group (not present in this DANDI subset). Using the in-file subject id rather than the directory name guarantees the label matches the data it is attached to.

## 1-c. How are the data split into sessions?

i. One session = one `.nwb` file = one imaging day. Sessions are never merged or split; each is processed independently and appended as one entry of `neural`/`input`/`output`. The NWB `general/session_id` (experiment day) and the acquisition date and VR scene from `identifier` are stored in `metadata['session_info']`. Sessions yielding fewer than 2 usable trials would be dropped (this never triggers).

ii.
```python
exp_day = int(f['general/session_id'][()])
date = ident.split('/')[-2]
...
for fn in files:
    with open(fn, 'rb') as fh:
        res = pickle.load(fh)
    if len(res['neural']) < 2:
        print('skipping %s: fewer than 2 usable trials' % fn)
        continue
    data['neural'].append(res['neural'])
```

iii. The file naming (`sub-m11_ses-03_behavior+ophys.nwb`) and the per-file `identifier` (`/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`) make one file = one recording day unambiguous. No cross-day ROI alignment was attempted (the repo's `multiDayROIAlign` is only needed for the paper's across-day analyses), so neurons are session-local. The `<2 trials` guard exists because the target format requires at least two trials per session for decoder evaluation.

## 1-d. How are the data split into trials?

i. A trial is a lap: it starts at the sample where the `trial_start` channel is non-zero and ends at the first sample where `teleport` is non-zero (the teleport sample itself is excluded, i.e. the half-open interval `[start, stop)`). The number of starts and stops is asserted to be equal. This is the repo's `sess.trial_start_inds` / `sess.teleport_inds` convention.

ii.
```python
starts = np.where(get('trial_start') > 0)[0]
stops = np.where(get('teleport') > 0)[0]
...
ntrials = len(starts)
assert len(stops) == ntrials
...
for i, (s, e) in enumerate(zip(starts, stops)):
    act = activity[:, s:e]
    ...
    T = e - s
```

iii. From the trajectory (steps 45–46, 49): the agent verified trial extraction on real sessions, checking per-trial position ranges, reward-zone entries, rewards, licks and morph, and confirmed "trials = trial_start→teleport (2.62M in-trial frames at 64.5 ms), no scanning gaps". Trial durations were checked (6–217 s) and the count (12,216 recorded trials) matches the paper's reported ~80.5 trials/session. The stored `trial number` channel was used only as a label, not to define boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Three filters:
1. **Lick-sensor-error trials are dropped entirely** — a trial is flagged if >30% of its frames carry a cumulative lick count >2 (the paper's criterion). This removed exactly 81 trials, matching the paper's reported 81/12,376.
2. Trials whose neural activity or whose position/speed/lick traces contain any non-finite value are skipped (this never triggers in practice).
3. Trials whose end index falls beyond the truncated common length are dropped in the ten sessions with a neural/behavior frame-count mismatch (also never triggers, since the neural stream is the longer one).
No minimum trial-length filter is applied (the shortest trial in the whole dataset is 96 samples ≈ 6 s). Final dataset: 12,135 of 12,216 trials.

ii.
```python
lick_error[i] = (np.sum(seg > LICK_ERR_COUNT) / len(seg)) > LICK_ERR_FRAC
...
for i, (s, e) in enumerate(zip(starts, stops)):
    if lick_error[i]:
        continue
    act = activity[:, s:e]
    if not np.all(np.isfinite(act)):
        continue
    p = np.clip(pos[s:e], 0.0, None)
    spd = speed[s:e]
    lk = lick[s:e]
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(spd))
            and np.all(np.isfinite(lk))):
        continue
```

iii. The agent read `behavior.correct_lick_sensor_error` and the Methods, and ran a behavior-only scan over all 152 sessions before writing the converter: "lick-sensor-error trials = 81 (paper: 81)". Its docstring states the reasoning explicitly: "Trials with lick-sensor errors ... are dropped, as these licks are unreliable (`behavior.correct_lick_sensor_error`; the paper NaNs them out -- 81 trials here, matching the paper's 81/12,376)". Because `lick` is a required decoder output and cannot be NaN in this format, the agent converted the paper's NaN-ing into trial removal.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces `processing/ophys/Fluorescence/plane<i>/data` (F) and `processing/ophys/Neuropil/plane<i>/data` (Fneu), together with `ImageSegmentation/PlaneSegmentation/iscell` and `planeIdx`. The NWB `Deconvolved` (suite2p `spks`) array is deliberately **not** used. For the two-plane mice (m17, m18) the per-plane series are scattered back into a single ROI-by-frame matrix using each plane's `rois` index list, so planes are pooled.

ii.
```python
def load_fluorescence(f):
    ophys = f['processing/ophys']
    seg = ophys['ImageSegmentation/PlaneSegmentation']
    nroi = seg['id'].shape[0]
    nframes = ophys['Fluorescence/plane0/data'].shape[0]
    F = np.empty((nroi, nframes), dtype=np.float32)
    Fneu = np.empty((nroi, nframes), dtype=np.float32)
    for plane in sorted(ophys['Fluorescence'].keys()):
        rois = ophys['Fluorescence'][plane]['rois'][:]
        F[rois] = ophys['Fluorescence'][plane]['data'][:].T
        Fneu[rois] = ophys['Neuropil'][plane]['data'][:].T
    iscell = seg['iscell'][:, 0] > 0
    plane_idx = seg['planeIdx'][:]
    return F, Fneu, iscell, plane_idx
```

iii. Trajectory steps 29–31: the agent printed statistics of `Fluorescence`, `Neuropil` and `Deconvolved` and concluded "NWB has raw F, Fneu, suite2p spks", then followed the repo pipeline that starts from F and Fneu (`preprocessing.dff`), because the paper computes its own dF/F and its own OASIS deconvolution rather than using suite2p's stored `spks`. Step 52 confirmed the multi-plane layout ("ROIs are indexed globally with planeIdx"), matching the Methods statement that planes are pooled for all analyses.

## 2-b. How is the `neural` data processed?

i. A re-implementation of the repo's `preprocessing.dff(..., neuropil_method='subtract', baseline_method='maximin', neu_coef=0.7, tau=0.7, deconvolve=True)`: samples outside trials are masked to NaN; `F - 0.7*Fneu`; per trial a maximin baseline (Gaussian smoothing with sigma = 15 samples, then a running minimum followed by a running maximum) with the trial's mean neuropil added back to both signal and baseline; `dF/F = (F - baseline)/|baseline|`; per-trial Gaussian smoothing with sigma = 2 samples; then OASIS deconvolution (`suite2p.extraction.dcnv.oasis`, tau = 0.7) to produce "events".

**The signal actually saved is dF/F, not the deconvolved events** (`--signal dff` is the default; `--signal events` is available). The deconvolved events are computed on every session and then discarded under the default.

Two implementation divergences from the repo: (a) the maximin window is `min(300, trial_length)` rather than a fixed 300 samples (~20 s), so for the majority of trials (median length 190 samples) the window is the trial length — on a test session this changes dF/F by a median per-cell r of 0.984 relative to the fixed 300-sample window; (b) the repo's per-mouse/per-day `keep_teleports` metadata (`teleport_metadata.py`, days on which the laser was not blanked and the baseline window may span the teleport) is read during exploration but never applied — teleport samples are always excluded.

ii.
```python
def compute_activity(F, Fneu, starts, stops, frame_rate):
    """dF/F and deconvolved events, computed within each trial (paper pipeline)."""
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]
    valid = ~np.isnan(f_[0])
    f_ -= NEU_COEF * fneu_
    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        seg = nansmooth(f_[:, s:e], [0, BASELINE_SMOOTH])
        win = min(BASELINE_WIN, max(1, e - s))
        seg = minimum_filter1d(seg, win, axis=-1)
        seg = maximum_filter1d(seg, win, axis=-1)
        offset = NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        f_[:, s:e] = f_[:, s:e] + offset
        flow[:, s:e] = seg + offset
    dff = np.full(F.shape, np.nan, dtype=np.float32)
    dff[:, valid] = (f_[:, valid] - flow[:, valid]) / np.abs(flow[:, valid])
    import suite2p.extraction.dcnv as dcnv
    events = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        dff[:, s:e] = nansmooth(dff[:, s:e], [0, DFF_SMOOTH])
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, frame_rate)
    return dff, events, valid
```
```python
ap.add_argument('--signal', default='dff', choices=['events', 'dff'])
...
activity = events if signal == 'events' else dff
```

iii. The docstring gives the justification: "dF/F is used here (`--signal dff`, the default) because it is the signal the paper treats as closest to the raw data (e.g. spatial peak firing, sequence analyses) and it retains the graded amplitude information that a linear decoder can exploit; deconvolution in the paper served analyses that needed the asymmetric calcium kinetics removed (spatial information, RR-position decoding). Deconvolved events can be produced instead with `--signal events`; they decode the same variables but less accurately (e.g. position 0.51 vs 0.72 balanced accuracy on a 4-session test)." The trajectory shows the agent originally implemented and defaulted to events (step 55), ran both on a 4-session subset (steps 61–62), and switched the default to dF/F on the basis of decoding accuracy, then patched the docstring so that code and stated decision agree (step 76).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both from the paper: (1) only ROIs with suite2p manual curation flag `iscell == 1` are kept; (2) putative interneurons — cells whose dF/F correlates with running speed at Pearson r > 0.5 over all in-trial samples — are excluded. 138,377 of 138,678 curated ROIs survive (301 putative interneurons, ~0.2%). No further per-cell filtering (no SNR, no activity-rate threshold) is applied.

ii.
```python
iscell = seg['iscell'][:, 0] > 0
...
F = F[iscell]
Fneu = Fneu[iscell]
plane_idx = plane_idx[iscell]
```
```python
# exclude putative interneurons: dF/F correlated with running speed (r > 0.5)
spd_valid = speed[valid]
dff_valid = dff[:, valid]
dv = dff_valid - dff_valid.mean(axis=1, keepdims=True)
sv = spd_valid - spd_valid.mean()
denom = np.sqrt((dv ** 2).sum(axis=1) * (sv ** 2).sum())
with np.errstate(invalid='ignore', divide='ignore'):
    speed_corr = (dv @ sv) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
keep_cells = ~is_int
```

iii. The agent read `spatial.is_putative_interneuron` (step 43) and noted that the repo's default threshold is 0.3 but the Methods specify 0.5; it used the paper's 0.5 (`INT_R_THRESH = 0.5`) and correlates over the same in-trial (`nanmask`/`valid`) samples as the repo function. It sanity-checked the outcome against the paper: "interneuron fraction 0.6% matches the paper's 0.42±0.85%" (step 50), and per-session cell counts (155–2,327) match the paper's stated 155–2,172 range closely.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The requested alignment event is the start of the trial, which is exactly the left edge of each trial slice, so no extra alignment, padding or cropping is done. Every trial matrix begins at the `trial_start` frame and runs to the frame before `teleport`; trials therefore have variable length. `metadata['temporal_alignment_event']` = "start of trial: teleport into the start of the virtual linear track (trial_start)", with `off_start = 0.0` and `off_end = None`.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    act = activity[:, s:e]
    ...
    neural_trials.append(np.ascontiguousarray(act, dtype=np.float32))
```
```python
'temporal_alignment_event': ('start of trial: teleport into the start of the '
                             'virtual linear track (trial_start)'),
'off_start': 0.0,
'off_end': None,
```

iii. Docstring: all VR streams in the NWB files are already synchronized to the imaging frames, so slicing neural and behavior with the same frame indices aligns them by construction; the instruction to "temporally align based on start of the trial" is satisfied by taking the trial slice itself. No pre-trial (baseline) window is included, consistent with the trial definition used throughout the repo.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling of any kind. Data are kept at the native two-photon volume rate: the frame interval is computed per session as the median difference of the behavior timestamps, which is 64.48362720402656 ms (15.51 Hz) in every one of the 152 sessions, including the two-plane mice (where the scanner runs at 31 Hz but the per-volume rate is 15.5 Hz). `metadata['time_bin_size']` is the mean of the per-session bin sizes in ms (64.4836). The same per-session `frame_rate` is passed to OASIS as the deconvolution sampling rate.

ii.
```python
frame_rate = 1.0 / np.median(np.diff(tstamps))
...
'frame_rate_hz': float(frame_rate),
...
bin_ms = float(np.mean([1000.0 / s['frame_rate_hz'] for s in session_info]))
data['metadata'] = {..., 'time_bin_size': bin_ms, ...,
    'sampling': ('native two-photon frame rate (~15.5 Hz); all VR behavior streams '
                 'in the NWB files are already synchronized to imaging frames'), ...}
```

iii. Trajectory steps 53–54: the agent confirmed the two-plane mice have "two planes with identical frame counts aligned to behavior" and checked m17's timestamp spacing (0.06448 s), i.e. the effective per-volume interval is the same for single- and multi-plane recordings, so no resampling is needed to keep the bin size constant across sessions. It explicitly sized the native-resolution dataset ("2.62M in-trial frames, 2.42G cell-frames → ~9.7 GB float32") and, after checking the machine (1 TB RAM, 23 GB GPU) and the decoder's per-session batching, decided native resolution was affordable and preferable to temporal binning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `position` time series' `timestamps` array (seconds from session start), which is shared by all behavior series and by the imaging frames.

ii.
```python
b = f['processing/behavior/BehavioralTimeSeries']
tstamps = b['position/timestamps'][:]
...
t_rel = tstamps[s:e] - tstamps[s]
inp[0] = t_rel
```

iii. The agent verified that all VR streams are sampled on the imaging frame clock ("All VR streams in the NWB files are already synchronized to the ~15.5 Hz imaging frames"), so any behavior series' timestamps give the same vector; `position` was used as the canonical clock. Median spacing is 64.48 ms in every session.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Only subtraction of the trial's first timestamp, giving a continuous, time-varying ramp starting at exactly 0 s in every trial. No smoothing, resampling or normalization. Observed range across the dataset: 0 to 216.5 s.

ii.
```python
t_rel = tstamps[s:e] - tstamps[s]
inp = np.empty((4, T), dtype=np.float32)
inp[0] = t_rel
```

iii. Direct reading of the instruction "Time from start of trial in seconds (continuous, time-varying)"; using the real timestamps rather than `arange(T)*bin` preserves any jitter in the frame clock.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is sliced with the same `[s:e]` frame indices as the neural matrix, so alignment is exact by construction. Before slicing, if the ophys and behavior streams differ in length (ten two-plane sessions have one extra imaging frame), all streams — F, Fneu, position, speed, lick, environment, trial number, reward_zone, timestamps — are truncated to the common length and any trial extending past it is dropped.

ii.
```python
nsamp = min(F.shape[1], len(pos))
if F.shape[1] != len(pos):
    F = F[:, :nsamp]
    Fneu = Fneu[:, :nsamp]
    pos = pos[:nsamp]; speed = speed[:nsamp]; lick = lick[:nsamp]
    morph = morph[:nsamp]; trialnum = trialnum[:nsamp]
    rzone_entry = rzone_entry[:nsamp]; tstamps = tstamps[:nsamp]
    keep = stops <= nsamp
    starts = starts[keep]; stops = stops[keep]
    ntrials = len(starts)
```

iii. Step 65–66: the conversion initially crashed on a 1-frame mismatch; the agent scanned all sessions, found "10 sessions from the two-plane mice (m17, m18) have one more imaging frame than behavior samples", and fixed it by truncating every stream to the common length, documented in the code comment as "the last frame of the interleaved scan".

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavior time series (the repo's `morph`: 0 = ENV1, 1 = ENV2; values of -1 occur outside trials).

ii.
```python
morph = get('environment')
...
env = np.array([1.0 if np.nanmax(morph[s:e]) > 0.5 else 0.0
                for s, e in zip(starts, stops)])
```

iii. The agent inspected the descriptions of every behavior series (step 19) and confirmed the morph coding against the repo's `env_morph_dict = {'Env1': 0, 'Env2': 1, 'Env3': 0.5}` and the scene names in the NWB `identifier` (e.g. `Env1_A_to_Env2_B`), which include cross-environment switch days.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, the maximum of the morph channel within the trial is thresholded at 0.5 to give a binary 0/1 label, which is then broadcast as a constant across all timepoints of the trial. Computing it per trial (rather than per session) correctly handles the cross-environment switch days, where the environment changes mid-session. The session-level majority environment is additionally recorded as `'ENV1'`/`'ENV2'` in `session_info`.

ii.
```python
env = np.array([1.0 if np.nanmax(morph[s:e]) > 0.5 else 0.0
                for s, e in zip(starts, stops)])
...
inp[1] = env[i]
...
'environment': 'ENV2' if env.mean() > 0.5 else 'ENV1',
```

iii. The threshold at 0.5 guards against the -1 sentinel values that appear outside trials and against the repo's intermediate morph value 0.5 (Env3, not present in this dataset); using the within-trial maximum keeps the value strictly binary as the instructions require.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The NWB `trial number` behavior time series, sampled at the trial's first frame.

ii.
```python
trialnum = get('trial number')
...
inp[2] = float(trialnum[s])
```

iii. The agent examined the behavior channel descriptions and used the file's own trial counter rather than a re-derived index. (I verified independently that `trial number` at every `trial_start` frame equals the sequential index 0…N-1 in all 152 sessions, so this is numerically identical to a loop counter; the observed range 0–99 matches the expert's.) Because lick-error trials are dropped after labelling, the stored numbers preserve the true within-session lap number rather than being renumbered.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond reading the value at the trial start frame and broadcasting it as a constant over the trial's timepoints as a float. No normalization or re-indexing.

ii.
```python
inp[2] = float(trialnum[s])
```

iii. The instruction asks for trial number as a continuous, per-trial input; the raw counter is already 0-based and sequential within a session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the same per-trial `rewarded` vector used for the reward-outcome output: the `Reward` time series (its own event timestamps, mapped to behavior frames with `searchsorted`) combined with the `reward_zone` channel — a trial counts as rewarded only if a reward event fell inside the trial *and* the animal was flagged as being in the reward zone during the trial, exactly as in the repo's `behavior.get_trial_types`.

ii.
```python
reward_frames = np.searchsorted(tstamps, b['Reward/timestamps'][:])
rzone_entry = get('reward_zone')
...
for i, (s, e) in enumerate(zip(starts, stops)):
    got_reward = np.any((reward_frames >= s) & (reward_frames < e))
    in_zone = np.any(rzone_entry[s:e] > 0)
    rewarded[i] = int(bool(got_reward) and bool(in_zone))
```

iii. Step 22: "Found reward zone dict (A:80-130, B:200-250, C:320-370) and trial types logic (isreward = reward delivered AND in reward zone)"; the docstring records "Reward outcome: reward delivered while in the reward zone, as in `behavior.get_trial_types`." The agent verified the resulting reward rate (~85%) against the paper's ~15% random omission rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The `rewarded` vector is shifted by one trial and broadcast as a constant over each trial's timepoints. **For the first trial of every session the value is set to 1 (rewarded)** rather than 0, on the grounds that the preceding (un-imaged) warm-up trials were run in the same condition and were rewarded ~85% of the time. Note that the shift is over *recorded* trials, computed before lick-error trials are dropped, so the "previous trial" is always the true preceding lap.

ii.
```python
# previous trial outcome; for the first imaged trial of a session the previous
# trial is one of the ~30 un-imaged warm-up trials run in the same condition
# immediately before imaging, which were rewarded on ~85% of trials, so it is
# coded as rewarded.
prev_rewarded = np.concatenate([[1], rewarded[:-1]]).astype(np.float64)
...
inp[3] = prev_rewarded[i]
```

iii. Step 77 shows the agent explicitly revisiting "the 'previous trial outcome' for the first trial" as a design choice to double-check. The justification comes from the Methods: "Before the imaging session, mice were provided 30 'warm-up' trials using the task and reward zone from the previous day", so the first imaged trial does have a real predecessor, which was most likely rewarded.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` behavior time series plus the per-trial reward-zone label. The zone label is **not** inferred from the data: it is parsed from the VR scene name in the NWB `identifier` (e.g. `Env1_LocationB_to_A`, `Env1_A_to_Env2_C`) and, on switch days, changes after trial 30 — a direct port of `behavior.get_reward_zones` with `reward_zone_dict` (X/A = 80–130, Y/B = 200–250, Z/C = 320–370 cm).

ii.
```python
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
SWITCH_TRIAL = 30

def scene_reward_zones(scene, ntrials, change_trial=SWITCH_TRIAL):
    m = re.match(r'^Env\d_Location([ABC])$', scene)
    if m:
        return [m.group(1)] * ntrials
    m = re.match(r'^Env\d_Location([ABC])_to_([ABC])$', scene)
    if m is None:
        m = re.match(r'^Env\d_([ABC])_to_Env\d_([ABC])$', scene)
    if m:
        return [m.group(1)] * change_trial + [m.group(2)] * (ntrials - change_trial)
    raise ValueError('unrecognized scene name: %s' % scene)
...
zone_labels = scene_reward_zones(scene, ntrials)
...
zstart, zend = REWARD_ZONES[zone_labels[i]]
```

iii. The agent first tried to derive the zone from the `reward_zone` channel and found it unusable on its own: "rzone flag marks reward-zone entry only on rewarded trials, so reward-zone location per trial must come from the scene name (identifier) plus the switch at trial 30, exactly as in `behavior.get_reward_zones`" (step 47). It then validated the scene-derived labels against the observed zone-entry positions across all 152 sessions: "zone inference from scene name + 30-trial switch matches the rzone entry positions exactly (0 mismatches)" (steps 51–52). (I reproduced this check independently: 0 mismatches over 10,394 trials with a zone entry, and all 26 distinct scene names parse.)

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is first clipped at 0 (the few marginally negative samples at the very start of a lap). The signed distance is then 0 anywhere inside the zone, `position - zone_start` (negative) before the zone, and `position - zone_end` (positive) after it — i.e. distance to the nearest point of the reward zone. It is computed per timepoint and then discretized (7-c).

ii.
```python
p = np.clip(pos[s:e], 0.0, None)
...
zstart, zend = REWARD_ZONES[zone_labels[i]]
dist = np.where(p < zstart, p - zstart, np.where(p > zend, p - zend, 0.0))
...
out[0] = discretize_reward_distance(dist)
```

iii. Step 77 lists "the reward-zone distance definition (distance to any location in the zone → 0 inside)" as an item the agent explicitly re-checked against the instruction ("Distance to any location in the reward zone"), and it spot-checked a trial to confirm the distance is exactly 0 precisely while the animal is inside the labelled zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned by explicit boolean masks: 0 for d < -50; 1 for -50 ≤ d < -10; 2 for -10 ≤ d < 0; 3 for d == 0 (inside the zone); 4 for 0 < d ≤ 10; 5 for 10 < d ≤ 50; 6 for d > 50. `output_values` names each bin. The resulting marginal distribution is 0.251/0.102/0.073/0.239/0.021/0.072/0.243.

ii.
```python
def discretize_reward_distance(d):
    """Signed distance (cm) to the nearest point of the reward zone -> 7 bins."""
    out = np.full(d.shape, 6, dtype=np.int64)
    out[d > 50] = 6
    out[(d > 10) & (d <= 50)] = 5
    out[(d > 0) & (d <= 10)] = 4
    out[d == 0] = 3
    out[(d >= -10) & (d < 0)] = 2
    out[(d >= -50) & (d < -10)] = 1
    out[d < -50] = 0
    return out
```

iii. The bin edges are transcribed directly from the Decoder Task specification; the separate `d == 0` class implements the spec's "3: 0 cm" (in-zone) category, and the outer classes are left open-ended so no sample can fall outside the seven classes.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the same `[s:e]` frame indices as the neural matrix, so the distance time series is sample-for-sample aligned with the neural data and has identical length `T`.

ii.
```python
act = activity[:, s:e]
p = np.clip(pos[s:e], 0.0, None)
...
out = np.empty((6, T), dtype=np.int64)
out[0] = discretize_reward_distance(dist)
```

iii. All VR streams are stored on the imaging frame clock in the NWB files, so no interpolation is required; the only alignment work is the common-length truncation described in 3-c.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior time series (cm along the 450 cm virtual track).

ii.
```python
pos = get('position')
...
p = np.clip(pos[s:e], 0.0, None)
```

iii. The channel is the VR position used throughout the repo (`sess.vr_data['pos']`); the agent verified the per-trial position ranges when validating trial extraction (step 45).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped at a lower bound of 0 (raw values dip marginally below 0 at lap onset and the inter-trial teleport period reaches -500, though teleport samples are never inside a trial), then divided by 90 and floored to give a bin index, clipped into [0, 4]. No smoothing or unit conversion.

ii.
```python
p = np.clip(pos[s:e], 0.0, None)
...
out[1] = np.clip(np.floor(p / 90.0), 0, 4).astype(np.int64)
```

iii. The 450 cm track divided into five equal bins gives 90 cm per bin, exactly as the instruction specifies; clipping handles the handful of samples slightly outside [0, 450] so they join the end bins rather than forming spurious classes. The resulting marginals (0.212/0.177/0.231/0.226/0.154) are close to uniform, as expected for near-uniform track traversal.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five categories with edges at 90, 180, 270 and 360 cm, implemented as `clip(floor(p/90), 0, 4)`: 0 for <90, 1 for 90–180, 2 for 180–270, 3 for 270–360, 4 for ≥360 (including anything beyond 450).

ii.
```python
out[1] = np.clip(np.floor(p / 90.0), 0, 4).astype(np.int64)
...
'output_values': [..., ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'], ...]
```

iii. Step 77: the agent re-verified "whether output 'position' bin edges match spec (<90, 90-180, ..., >360)" against the Decoder Task description before finalizing.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[s:e]` frame slice as the neural matrix — no separate alignment step, identical length `T`.

ii.
```python
act = activity[:, s:e]
p = np.clip(pos[s:e], 0.0, None)
out = np.empty((6, T), dtype=np.int64)
out[1] = np.clip(np.floor(p / 90.0), 0, 4).astype(np.int64)
```

iii. As in 3-c/7-d: behavior and imaging share the frame clock in the NWB files, verified by the timestamp spacing matching the imaging rate in all sessions.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior time series (per-frame cumulative lick counts from the capacitive sensor).

ii.
```python
lick = get('lick')
...
lk = lick[s:e]
```

iii. The agent read the NWB channel descriptions and the paper's licking-quantification methods ("The capacitive lick sensor allowed us to detect single licks ... Remaining lick counts were converted to a binary vector"), which is precisely how the channel is treated here.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarization: any frame with a lick count > 0 becomes 1, everything else 0. No smoothing (the paper's Gaussian-smoothed lick rate is only used for its GLM/lick-rate analyses, not needed for a binary decoder output). Trials flagged as lick-sensor errors are removed altogether (see 1-e) rather than having their licks set to NaN, since the format cannot carry NaN labels. Resulting marginal: 0.777 no-lick / 0.223 lick.

ii.
```python
out[3] = (lk > 0).astype(np.int64)
...
'exclusions': ('... trials with lick sensor errors (>30% of frames with cumulative '
               'lick count > 2)'),
```

iii. The instruction specifies a binary lick output ("0 = no, 1 = yes"), matching the paper's conversion of lick counts to a binary vector; the sensor-error rule is taken verbatim from the Methods and `behavior.correct_lick_sensor_error`, and the agent confirmed it flags exactly the paper's 81 trials.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[s:e]` frame slice as the neural matrix; lick counts are recorded per imaging frame, so no resampling or event-to-frame mapping is needed.

ii.
```python
lk = lick[s:e]
out[3] = (lk > 0).astype(np.int64)
```

iii. Same reasoning as 3-c: all `BehavioralTimeSeries` share the imaging-frame timestamps.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The VR scene name in the NWB `identifier` field plus the trial index, via `scene_reward_zones` (see 7-a). The `reward_zone` behavior channel is used only for the reward-outcome logic, not for labelling the zone.

ii.
```python
ident = f['identifier'][()].decode()
scene = ident.split('/')[-1]
...
zone_labels = scene_reward_zones(scene, ntrials)
```

iii. See 7-a: the agent established that the `reward_zone` channel only flags zone entries on rewarded/licked trials, so it cannot label omission trials, whereas the scene name plus the paper's fixed 30-trial switch labels every trial; it validated the two against each other with zero mismatches across all sessions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The letter label is mapped to the index 0/1/2 via `ZONE_NAMES = ['A','B','C']` and broadcast as a constant across all timepoints of the trial (so it is stored time-varying, as the format prefers). On switch sessions, trials 0–29 carry the pre-switch zone and trials 30+ the post-switch zone. Marginals are 0.332/0.336/0.333, i.e. the three zones are balanced as expected from the counterbalanced design.

ii.
```python
ZONE_NAMES = ['A', 'B', 'C']
...
out[4] = ZONE_NAMES.index(zone_labels[i])
...
'output_values': [..., ['zone A (80-130 cm)', 'zone B (200-250 cm)', 'zone C (320-370 cm)'], ...]
```

iii. Mapping to 0/1/2 follows the instruction "Reward zone location, per-trial. 0 = A, 1 = B, 2 = C"; the 30-trial switch point is the paper's ("Each switch occurred after 30 trials") and the repo's `get_reward_zones(change_trial=30)` default.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` time series (event timestamps) and the `reward_zone` behavior channel, per 6-a. Reward event times are converted to frame indices with `searchsorted` against the behavior timestamps.

ii.
```python
reward_frames = np.searchsorted(tstamps, b['Reward/timestamps'][:])
rzone_entry = get('reward_zone')
```

iii. `Reward` is stored as discrete events with their own timestamps rather than a per-frame channel, so it must be mapped onto the frame clock; the conjunction with `reward_zone` reproduces the repo's `get_trial_types` definition of `isreward`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per trial: 1 if at least one reward event frame falls in `[s, e)` **and** the `reward_zone` channel was non-zero at some point in the trial, otherwise 0; the value is broadcast as a constant across the trial's timepoints. Resulting marginal: 0.842 rewarded / 0.158 omitted, consistent with the paper's ~15% random omission rate.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    got_reward = np.any((reward_frames >= s) & (reward_frames < e))
    in_zone = np.any(rzone_entry[s:e] > 0)
    rewarded[i] = int(bool(got_reward) and bool(in_zone))
...
out[5] = rewarded[i]
```

iii. Directly ports `behavior.get_trial_types` ("isreward = reward delivered AND in reward zone"), which the agent identified at step 22 and cited in the module docstring. The agent cross-checked the resulting ~85% reward rate against the paper's stated ~15% omission rate in its all-session behavior scan (step 52).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases:
- **Neural/behavior frame-count mismatch** (10 two-plane sessions with one extra imaging frame): every stream is truncated to the common length and any trial extending past it is dropped.
- **Non-finite values** in the neural activity or in position/speed/lick within a trial: the trial is silently skipped.
- **Unreliable lick sensor**: whole trial dropped (81 trials).
- **Sessions with fewer than 2 usable trials**: skipped at assembly time (never triggers).
- Structural assumptions are asserted rather than assumed silently (`assert len(stops) == ntrials`; `scene_reward_zones` raises on an unrecognized scene name).
- NaN-safe helpers are used throughout the dF/F pipeline (`nansmooth`, `np.nanmean`), and NaN speed correlations are mapped to 0 so a degenerate cell is kept rather than dropped.

ii.
```python
nsamp = min(F.shape[1], len(pos))
if F.shape[1] != len(pos):
    ...
    keep = stops <= nsamp
    starts = starts[keep]; stops = stops[keep]
```
```python
if not np.all(np.isfinite(act)):
    continue
...
if not (np.all(np.isfinite(p)) and np.all(np.isfinite(spd))
        and np.all(np.isfinite(lk))):
    continue
```
```python
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
...
if len(res['neural']) < 2:
    print('skipping %s: fewer than 2 usable trials' % fn)
    continue
```

iii. The frame-mismatch handling was added reactively after the first full run crashed (steps 65–66): the agent scanned all sessions to find the scope of the problem (10 sessions, always 1 extra imaging frame on the two-plane mice) before choosing truncation, and documented the cause in a code comment. The finiteness guards are defensive; the agent had previously verified no trials contain `scanning == 0` or NaN positions (steps 50, 53).

## 13-a. What are the most time-consuming steps of the code?

i. In order: (1) reading and decompressing the raw `Fluorescence` and `Neuropil` HDF5 datasets (~87 GB of source data; both full ROI-by-frame matrices are materialized per session); (2) the dF/F pipeline itself — the per-trial `nansmooth` (Gaussian filters on the full ROI × frame array) and OASIS deconvolution over ~12k trials × ~900 cells, the latter of which is thrown away under the default `--signal dff`; (3) serialization — 9.6 GB is pickled once into the per-session cache files and then again into the final `converted_data.pkl`, plus 9.6 GB read back in the parent. Wall-clock was ~6 s per session with 8 parallel workers (a few minutes total for 152 sessions), so the run is dominated by I/O and by the final single-threaded assembly and write.

ii.
```python
F[rois] = ophys['Fluorescence'][plane]['data'][:].T
Fneu[rois] = ophys['Neuropil'][plane]['data'][:].T
...
events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, frame_rate)
...
with open(outfn, 'wb') as fh:
    pickle.dump(res, fh, protocol=4)
...
with open(args.out, 'wb') as fh:
    pickle.dump(data, fh, protocol=4)
```

iii. The agent benchmarked the dF/F + OASIS pipeline on a single session before scaling ("Prototype dF/F + OASIS works and is fast (<1 s for a small session)", step 50; "benchmark the dF/F + OASIS pipeline on one large session to estimate total conversion cost", step 48), and sized the output up front ("2.62M in-trial frames, 2.42G cell-frames → ~9.7 GB float32"). It then parallelized across sessions with a spawn pool and a per-session cache so that a failure or a metadata change would not require recomputing the expensive part (used twice: after the frame-mismatch fix and after the metadata string fix, both of which reassembled from cache).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Candidates, all per-trial Python loops:
- `compute_activity` walks the trial list **three separate times** (mask copy, baseline computation, smoothing + deconvolution); these could be one pass, and the mask-copy pass could be replaced by building a boolean in-trial mask once and using `np.where`.
- The per-trial reward/lick-error loop and the `env` list comprehension each iterate over trials to compute one scalar; both could be done with `np.add.reduceat`/segment reductions over the concatenated arrays.
- The main emission loop recomputes `np.clip`, `np.digitize` and `discretize_reward_distance` per trial, although all of them are pointwise and could be applied once to the whole session array before slicing.
- `load_fluorescence` reads each plane's full array and scatters it; unavoidable given the HDF5 layout.
Genuine vectorization is limited by the variable trial lengths, and the loops are not the bottleneck (I/O is), so the practical gain is small.

ii.
```python
for s, e in zip(starts, stops):
    f_[:, s:e] = F[:, s:e]
    fneu_[:, s:e] = Fneu[:, s:e]
...
for s, e in zip(starts, stops):
    seg = nansmooth(f_[:, s:e], [0, BASELINE_SMOOTH])
    ...
for s, e in zip(starts, stops):
    dff[:, s:e] = nansmooth(dff[:, s:e], [0, DFF_SMOOTH])
    events[:, s:e] = dcnv.oasis(...)
```
```python
env = np.array([1.0 if np.nanmax(morph[s:e]) > 0.5 else 0.0
                for s, e in zip(starts, stops)])
for i, (s, e) in enumerate(zip(starts, stops)):
    got_reward = np.any((reward_frames >= s) & (reward_frames < e))
```

iii. The agent did vectorize the one loop that mattered at scale — the per-cell speed correlation is computed as a single matrix product over all cells rather than the repo's per-cell `np.corrcoef` loop — and parallelized across sessions instead of micro-optimizing within a session, which is the right trade-off given that each session takes ~6 s and is I/O bound.

## 13-c. What processing does the code repeat multiple times?

i. Real repetitions:
- The three passes over the trial list inside `compute_activity` (above).
- Each session's converted arrays are pickled to `/tmp/conv/<session>.pkl` and then read back and re-pickled into the final file, so 9.6 GB is written twice and read once; the cache is also never cleaned up.
- The per-trial `rewarded` conjunction is computed once but consumed twice (as the reward-outcome output and, shifted, as the previous-outcome input) — this one is correctly computed a single time.
Notably, the code does **not** repeat the expensive part: unlike a survey-then-convert design, each NWB file is opened and its fluorescence read exactly once, and the reward-zone labels come from metadata rather than from a second pass over the behavior.

ii.
```python
def worker(args):
    path, signal, outdir = args
    outfn = os.path.join(outdir, tag + '.pkl')
    if os.path.exists(outfn):
        return outfn
    res = process_session(path, signal=signal)
    with open(outfn, 'wb') as fh:
        pickle.dump(res, fh, protocol=4)
...
for fn in files:
    with open(fn, 'rb') as fh:
        res = pickle.load(fh)
```

iii. The cache is deliberate: it makes the pipeline restartable and let the agent rebuild the final pickle twice (after the frame-mismatch fix and after a metadata typo fix) without recomputing dF/F. It is also what makes the multiprocessing design work, since returning ~10 GB of arrays through a `Pool.map` result would be far more expensive than writing them to disk.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
- **OASIS deconvolution is run on every trial of every session and then thrown away** under the default `--signal dff`; only `dff` is kept. This is the clearest piece of wasted compute (and it forces the `suite2p` dependency).
- `plane_idx` is computed, filtered by `iscell` and by the interneuron mask, returned in each session's dict and cached to disk, but is never used in the final dataset (`brain_region_idx` is just zeros for CA1).
- `valid` is used only for the speed correlation; `TRACK_LENGTH` and the `sys` import are unused.
- `info['n_trials_dropped_lick_error']`, `'date'`, `'scene'`, etc. are cheap, and are retained in `metadata['session_info']`, so they are not waste.
- Within the dF/F computation, the full-length NaN arrays `f_`, `fneu_`, `flow`, `dff` and `events` are all allocated at ROI × frame size, including the out-of-trial samples that are then never used.

ii.
```python
import suite2p.extraction.dcnv as dcnv
events = np.full(F.shape, np.nan, dtype=np.float32)
for s, e in zip(starts, stops):
    dff[:, s:e] = nansmooth(dff[:, s:e], [0, DFF_SMOOTH])
    events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, frame_rate)
return dff, events, valid
...
activity = events if signal == 'events' else dff
...
plane_idx = plane_idx[keep_cells]
return {'neural': ..., 'plane_idx': plane_idx, 'info': info}
```

iii. The agent kept both signals reachable from one code path so that the `--signal events` variant it had benchmarked remains reproducible from the same script ("Deconvolved events can be produced instead with `--signal events`"), at the cost of always paying for the deconvolution. `plane_idx` was collected while exploring the multi-plane mice (m17/m18) and retained as provenance even though the paper pools planes for all analyses, so it has no consumer here.
