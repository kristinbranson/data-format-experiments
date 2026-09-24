# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs every NWB file matching `/app/data/sub-*/sub-*_ses-*_behavior+ophys.nwb` (152 files) and opens each one directly with `h5py` rather than `pynwb`, reading the HDF5 paths by hand. From each file it pulls the ophys group (`Fluorescence`, `Neuropil`, `ImageSegmentation/PlaneSegmentation/iscell`, `planeIdx`), the behavior group (`position`, `speed`, `lick`, `environment`, `trial_start`, `teleport`, `reward_zone`), the `Reward` event timestamps, and the metadata fields `general/subject/subject_id`, `general/session_id` and `identifier`. `--sample` restricts to 2 named sessions; `--full` (the default) processes all of them.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
print(f"Found {len(nwb_files)} NWB files")
...
with h5py.File(nwb_path, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    identifier = f['identifier'][()].decode()
    scene = parse_scene_name(identifier)
    F, Fneu, n_planes = load_neural_data(f)
    behav = f['processing/behavior/BehavioralTimeSeries']
    position = behav['position/data'][:]
    speed    = behav['speed/data'][:]
    lick     = behav['lick/data'][:]
    environment      = behav['environment/data'][:]
    trial_start_sig  = behav['trial_start/data'][:]
    teleport_sig     = behav['teleport/data'][:]
    reward_zone_raw  = behav['reward_zone/data'][:]
    frame_timestamps  = behav['position/timestamps'][:]
    reward_timestamps = behav['Reward/timestamps'][:]
```

iii. From CONVERSION_NOTES.md Step 2: "NWB files: `/app/data/sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb`". The AI confirmed the glob returns 152 files across 11 subject directories and cross-checked the resulting counts against the paper in Steps 4/9/10 (11 switch mice; 12,216 trials vs the paper's 12,376 across 14 mice, the difference attributed to the 3 "fixed" mice not released; m11 missing ses-01/ses-02). The trajectory shows it enumerated the identifier/scene/trial/cell counts for every file before writing the converter, so the file list was verified to be complete. h5py was used because it reads the arrays lazily and avoids the pynwb object-construction overhead (full conversion took 4.5 min).

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `general/subject/subject_id` field inside each NWB file (not from the directory name). After all sessions are processed the unique ids are sorted to build `subjects`, and each session's index into that list is stored in `subject_idx`.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
...
subjects = sorted(set(s['subject_id'] for s in all_sessions))
sub2idx = {s: i for i, s in enumerate(subjects)}
for s in all_sessions:
    subject_idx.append(sub2idx[s['subject_id']])
...
'subjects': subjects,
'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 2 records "Subjects | 11 (m3, m4, m7, m11-m15, m17-m19)", and Steps 9/10 check this against the paper's "n = 11 mice" for the reward-switch cohort. Reading the id out of the file rather than the folder name means the label cannot drift from the file contents.

## 1-c. How are the data split into sessions?

i. One session per NWB file. Sessions are appended in glob-sorted order (subject directory, then `ses-NN`), and nothing is merged or aligned across days/mice. A session is dropped only if it yields fewer than 2 usable trials.

ii.
```python
for i, nwb_path in enumerate(nwb_files):
    sess = process_session(nwb_path, show_processing=args.show_processing and i < 2)
    if sess: all_sessions.append(sess)
...
if n_trials < 2:
    print(f"    Skipping: {n_trials} trials"); return None
...
if len(neural_trials) < 2:
    print(f"    Skipping: {len(neural_trials)} valid trials"); return None
```

iii. CONVERSION_NOTES.md Step 2: "Sessions | 152 total", "Sessions/subject | 12-14 (m11 has 12, rest have 14)". Step 10 Check 4 compares this to the paper's 14 imaging days per mouse and explains the shortfall (m11 missing ses-01/02). The `< 2 trials` guard exists because the target format requires at least two trials per session for the decoder split; in practice it never fired (all 152 sessions were kept).

## 1-d. How are the data split into trials?

i. A trial runs from a `trial_start` pulse to the next `teleport` pulse, i.e. the on-track portion of a lap, with the inter-trial teleport period excluded. Both are found as the sample indices where the signal equals 1, then truncated to a common length and zipped positionally.

ii.
```python
trial_start_inds = np.where(trial_start_sig == 1)[0]
teleport_inds    = np.where(teleport_sig == 1)[0]
n_trials = min(len(trial_start_inds), len(teleport_inds))
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds    = teleport_inds[:n_trials]
...
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
    nf = te - ts
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "**Trial = track only**: trial_start to teleport (excludes teleport zone)", and Step 10 Check 3 lists "Trial bounds | trial_start to teleport | trial_start_inds to teleport_inds | ✓" as matching the reference. The reference repository's `behavior.py` slices the same two index arrays. The AI's Step 2 exploration confirmed 80 trial starts and 80 teleports in the sample session. (I verified across all 152 files that `trial_start == 1` is identical to `trial_start != 0`, that `teleport == 1` is identical to the rising edges of `teleport`, that the two arrays always have equal length, and that `start < end` for every pair — so this is exactly the same trial segmentation as the reference, 12,216 trials.)

## 1-e. How are trials filtered based on quality controls?

i. Essentially no quality filtering. A trial is dropped only if it has fewer than 2 frames (`nf < 2`); a session is dropped only if fewer than 2 trials survive. No speed filter, no lick-error-trial removal, no minimum trial length beyond 2 samples.

ii.
```python
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
    nf = te - ts
    if nf < 2: continue
...
if len(neural_trials) < 2:
    print(f"    Skipping: {len(neural_trials)} valid trials"); return None
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 5: "**All trials included**: No speed filtering", with the reasoning in Step 3 that the paper's "<2 cm/s" exclusion applies to place-cell spatial analyses, not to raw time series. Step 10 Check 5 reports "**Short trials**: No trials with <2 frames found ✓" and "**Long trials**: Some trials up to 3359 frames (216s) in m4 - valid, mouse stopped running ✓". The paper's 81 lick-error trials were noted in Step 3 but never excluded (no rationale given for the omission).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces `processing/ophys/Fluorescence/plane*/data` (F) and `processing/ophys/Neuropil/plane*/data` (Fneu). The NWB `Deconvolved` array is explicitly **not** used. ROIs are restricted to `iscell[:,0] == 1` and cells from multiple planes are concatenated along the neuron axis.

ii.
```python
def load_neural_data(f):
    """Load F, Fneu for cells (iscell=1) from all planes, concatenated."""
    fluor = f['processing/ophys/Fluorescence']
    neu   = f['processing/ophys/Neuropil']
    seg   = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
    iscell    = seg['iscell'][:]
    plane_idx = seg['planeIdx'][:]
    planes = sorted(fluor.keys())
    F_list, Fneu_list = [], []
    for pname in planes:
        pnum  = int(pname.replace('plane', ''))
        pmask = plane_idx == pnum
        pcell = iscell[pmask, 0] == 1
        F_list.append(fluor[pname]['data'][:, pcell])
        Fneu_list.append(neu[pname]['data'][:, pcell])
    return np.concatenate(F_list, axis=1), np.concatenate(Fneu_list, axis=1), len(planes)
```

iii. CONVERSION_NOTES.md Step 1: "NWB Deconvolved is suite2p raw, NOT the post-dF/F events from reference pipeline", and Step 4: "Deconvolved NWB | suite2p raw | sparse, non-negative | Need dF/F from F and Fneu". The trajectory shows the AI inspected the `Deconvolved` array (72.4% zeros, max 15054) and reasoned that "The preprocessing code computes dF/F first (with neuropil subtraction and baseline correction), then optionally applies OASIS deconvolution to the dF/F" — so the stored array is not the paper's signal.

## 2-b. How is the `neural` data processed?

i. dF/F only — the OASIS deconvolution step of the reference pipeline is deliberately skipped. Per session: subtract `0.7 * Fneu`; build a per-trial mask spanning `[trial_start, teleport)`; within each trial compute a maximin baseline by a running minimum followed by a running maximum with window `min(300, trial_length)`; form `(F_corr - baseline)/|baseline|` on the masked samples; NaN-aware Gaussian smooth with σ = 2 samples within each trial. Outside trials the signal stays NaN. At trial extraction, any remaining NaN/±Inf is replaced with 0 and cast to float32.

ii.
```python
NEU_COEF = 0.7
BASELINE_WINDOW = 300  # ~20s at 15.5 Hz
SMOOTH_SIGMA = 2

def compute_dff(F, Fneu, trial_start_inds, teleport_inds):
    """Compute dF/F: neuropil subtract, maximin baseline, smooth."""
    n_tp, n_neu = F.shape
    f_ = (F.T - NEU_COEF * Fneu.T).astype(np.float64)  # (neurons, timepoints)

    nanmask = np.zeros(n_tp, dtype=bool)
    for s, e in zip(trial_start_inds, teleport_inds):
        if s < e: nanmask[s:e] = True

    flow = np.full_like(f_, np.nan)
    for s, e in zip(trial_start_inds, teleport_inds):
        if s >= e: continue
        w = min(BASELINE_WINDOW, e - s)
        flow[:, s:e] = scipy.ndimage.minimum_filter1d(f_[:, s:e], w, axis=-1)
        flow[:, s:e] = scipy.ndimage.maximum_filter1d(flow[:, s:e], w, axis=-1)

    dff = np.full_like(f_, np.nan)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])

    for s, e in zip(trial_start_inds, teleport_inds):
        if s >= e: continue
        dff[:, s:e] = nansmooth(dff[:, s:e], SMOOTH_SIGMA, axis=1)
    return dff
...
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 1/Step 3 state the target recipe: "dF/F pipeline: F - 0.7*Fneu → maximin baseline (min filter 300, max filter 300) → dF/F → smooth σ=2". Step 5 Key Decision 1: "**Neural signal**: dF/F (not deconvolved) - preserves more info for decoding". Step 10 Check 3 adds: "**Difference**: We use dF/F as neural signal, reference uses deconvolved events for most analyses. **Justification**: dF/F preserves more temporal information for decoding. The deconvolved events in the NWB are from suite2p directly (not from the reference pipeline's dF/F → OASIS), so using dF/F is more faithful to the reference processing while providing a richer signal for decoding." Step 6 claims the script "Computes dF/F following reference pipeline exactly", and the Step 10 comparison table marks Baseline/dF/F/Smoothing all "✓ Same". Note the trajectory shows the AI verified `from suite2p.extraction import dcnv` was importable ("dcnv OK") before deciding not to deconvolve. Three further deviations from the reference `dff()` are undocumented: the reference adds the per-trial mean neuropil back after subtraction before taking the baseline, pre-smooths the baseline with a σ=[0,15] Gaussian before the min/max filters, and always uses a 300-sample window; it also switches the baseline window to span the teleport on the mouse/day list in `teleport_metadata.py`, which this code never consults.

## 2-c. How is the `neural` data filtered based on quality controls?

i. One filter only: suite2p's manual curation flag, `iscell[:,0] == 1`, applied per plane at load time. No exclusion of putative interneurons, no place-cell selection, no activity/SNR threshold.

ii.
```python
iscell    = seg['iscell'][:]
plane_idx = seg['planeIdx'][:]
for pname in planes:
    pnum  = int(pname.replace('plane', ''))
    pmask = plane_idx == pnum
    pcell = iscell[pmask, 0] == 1
    F_list.append(fluor[pname]['data'][:, pcell])
    Fneu_list.append(neu[pname]['data'][:, pcell])
```

iii. CONVERSION_NOTES.md Step 3: "**Neuron curation**: Use iscell flag from suite2p (manual curation). No additional filtering." Step 5 Key Decision 4: "**All iscell=1 neurons**: No place cell filtering (analysis-specific)". Step 10 Check 3 records "Cell filtering | iscell==1 | Same | ✓". The resulting count (138,678 cells, mean 912.4/session) is reported in Steps 2/9 as a consistency check. The Methods' additional step of dropping putative interneurons (dF/F–speed Pearson r > 0.5) is never mentioned anywhere in the notes or the trajectory.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, which requires nothing beyond slicing: the dF/F matrix, all behaviour arrays and the timestamps share one sample index, so the trial slice `[trial_start, teleport)` starts exactly at the alignment event. `off_start` is recorded as 0.0 (no pre-event window) and `off_end` as `None` (variable trial length).

ii.
```python
ts, te = trial_start_inds[ti], teleport_inds[ti]
nf = te - ts
trial_neural   = np.nan_to_num(dff[:, ts:te], ...).astype(np.float32)
time_from_start = (frame_timestamps[ts:te] - frame_timestamps[ts]).astype(np.float32)
trial_pos   = np.clip(position[ts:te], 0, TRACK_LENGTH)
trial_speed = speed[ts:te]
trial_lick  = lick[ts:te]
...
'temporal_alignment_event': 'trial_start (entry to linear track)',
'off_start': 0.0, 'off_end': None,
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6 ("Trial = track only: trial_start to teleport") plus Step 10 Check 2, which reports that `time_from_trial_start` was compared against the raw NWB timestamps with `np.allclose` and passed. The instructions ask for alignment to trial start, and because the ophys and VR streams are already resampled onto the same imaging frames in the NWB, one shared index is sufficient.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning, resampling or re-binning of any kind: the data are kept at the native imaging frame rate. `time_bin_size` is written as 1000/15.5078125 = 64.4836 ms, from a hard-coded constant. For the two-plane sessions (m17, m18) the stored per-plane arrays are already at the per-plane rate, so they need no downsampling. `effective_rate` is computed per session from the median timestamp difference and printed, but is not used or asserted against the constant.

ii.
```python
TARGET_RATE_HZ = 15.5078125
...
dt = np.median(np.diff(frame_timestamps))
effective_rate = 1.0 / dt
print(f"    {n_cells} cells, {n_planes} planes, effective_rate={effective_rate:.1f}Hz")
...
time_bin_ms = 1000.0 / TARGET_RATE_HZ
...
'time_bin_size': time_bin_ms,
'imaging_rate_hz': TARGET_RATE_HZ,
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 2: "**No downsampling**: Multi-plane data already at ~15.5 Hz", and Step 4: "Multi-plane rate | 31 Hz header | 15.5 Hz timestamps | ~15.5 Hz | Data at per-plane rate". Step 10 Check 5: "**Multi-plane**: m17 and m18 handled correctly (concatenate planes, no downsampling) ✓". Step 9 lists "Imaging rate | ~15.5 Hz | 15.5 Hz | 15.5 Hz | ✓". (I confirmed all 152 sessions sample at 15.5078 Hz and that the two-plane files store both planes at that same rate with equal lengths, so the hard-coded constant is correct for this dataset.)

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `position` time series' own `timestamps` array (`processing/behavior/BehavioralTimeSeries/position/timestamps`), in seconds.

ii.
```python
frame_timestamps = behav['position/timestamps'][:]
...
time_from_start = (frame_timestamps[ts:te] - frame_timestamps[ts]).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: "frame timestamps | input[0] | Time from trial start (s)". The AI's Step 2 exploration showed every behavior time series carries an identical `timestamps` vector, so any one of them is equivalent; it also noted in Step 1 that "Reward timestamps in NWB use actual wall-clock time, must use position timestamps for alignment", i.e. it chose the frame timestamps as the canonical clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Only subtraction of the first timestamp of the trial, so each trial starts at 0 s. Stored as float32, time-varying, as input row 0.

ii.
```python
time_from_start = (frame_timestamps[ts:te] - frame_timestamps[ts]).astype(np.float32)
...
inp = np.zeros((4, nf), dtype=np.float32)
inp[0] = time_from_start
```

iii. CONVERSION_NOTES.md Step 10 Check 2 ("**Input data check**: Compared time_from_trial_start with actual NWB timestamps. Result: `np.allclose` passed ✓"). The resulting range, [0, 216.5 s] across the dataset, is reported in the verification output and is consistent with the longest trial (3359 frames at 64.5 ms).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the same `ts:te` slice is used for the dF/F matrix and for `frame_timestamps`, so element *k* of the time vector corresponds to column *k* of the neural matrix. No interpolation or offset is applied and no explicit check that the neural array and the behavior arrays have equal length is performed.

ii.
```python
ts, te = trial_start_inds[ti], teleport_inds[ti]
trial_neural    = np.nan_to_num(dff[:, ts:te], ...).astype(np.float32)
time_from_start = (frame_timestamps[ts:te] - frame_timestamps[ts]).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 2 notes that the ophys and behavior arrays in the NWB have the same number of samples (the VR stream is pre-aligned to the imaging frames upstream), so sharing the index is sufficient. Step 10 Check 2 verifies the time vector against the raw file with `np.allclose`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavior time series, whose valid values are 0 (ENV1) and 1 (ENV2); −1 marks non-scanning samples.

ii.
```python
environment = behav['environment/data'][:]
...
env_vals = environment[ts:te]
valid = env_vals[env_vals >= 0]
```

iii. CONVERSION_NOTES.md Step 1: "env=0 is ENV1, env=1 is ENV2"; Step 2 exploration reported "Environment unique values: [-1. 0.]" for the sample session and the AI's cross-file survey listed the valid env values per session. Step 5 mapping: "environment | input[1] | Binary ENV1=0/ENV2=1".

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, the −1 (non-scanning) samples are dropped and the rounded median of the remaining values is taken, giving one integer per trial; that constant is then broadcast across all timepoints of the trial as input row 1. If every sample in a trial were −1 the value would default to 0.

ii.
```python
trial_env = np.zeros(n_trials, dtype=np.int64)
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
    env_vals = environment[ts:te]
    valid = env_vals[env_vals >= 0]
    if len(valid) > 0:
        trial_env[ti] = int(np.round(np.median(valid)))
...
inp[1] = trial_env[ti]
```

iii. The instructions specify environment as "binary, ENV1 vs ENV2, **per trial**", so the AI collapsed the time series to one value per trial; the median-over-valid-samples is a defensive way to do that given the −1 sentinel. CONVERSION_NOTES.md Step 10 Check 2 reports "Environment correctly switches at trial 30 for switch sessions ✓". (I checked 2,435 trials across 30 sessions: the environment value is constant within every trial and never all −1, so this is numerically identical to just taking the per-sample values.)

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Not from a raw variable — it is the 0-based index of the trial within the session, i.e. the loop counter over the `trial_start`/`teleport` pairs. The NWB `trial number` time series is not used.

ii.
```python
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
    ...
    inp[2] = ti  # trial number
```

iii. CONVERSION_NOTES.md Step 5 mapping: "trial index | input[2] | 0-indexed trial number". The AI's Step 2 exploration found the stored `trial number` series ranges from −1 to 80 (with −1 marking non-scanning samples), i.e. it is not a clean per-trial label, so the index derived from the trial-boundary signals was used instead. Step 10 Check 2 reports "Trial number matches expected value ✓".

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond assigning the loop index; the value is constant across all timepoints of the trial and stored as float32 in input row 2. The index counts all detected trials, so it is not renumbered if a trial were skipped.

ii.
```python
inp[2] = ti  # trial number
```

iii. Trivial by construction. The verification output shows the per-session range of input 2 running 0..79 for the standard 80-trial sessions and 0..99 for the 100-trial ones, matching the per-session trial counts.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the per-trial reward outcome array `isreward`, which is built from the `Reward` event timestamps (`processing/behavior/BehavioralTimeSeries/Reward/timestamps`) combined with the `reward_zone` behavior time series: a trial counts as rewarded only if a reward timestamp falls inside the trial's time window **and** `reward_zone > 0` somewhere in the trial.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][:]
reward_zone_raw   = behav['reward_zone/data'][:]
...
isreward = np.zeros(n_trials, dtype=np.int64)
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
    has_rzone = np.any(reward_zone_raw[ts:te] > 0)
    t_start = frame_timestamps[ts]
    t_end   = frame_timestamps[min(te, n_timepoints-1)]
    has_reward = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
    if has_reward and has_rzone:
        isreward[ti] = 1
```

iii. CONVERSION_NOTES.md Step 1 identifies "`get_trial_types()` | behavior.py | LOADING | Determines reward per trial (reward>0 AND rzone>0)", and Step 5 Key Decision 7: "**Reward detection**: Match reference code: reward timestamp in trial AND rzone>0". Step 4 flags that "Reward timestamps in NWB use actual wall-clock time, must use position timestamps for alignment", which is why the comparison is done in timestamp space rather than by sample index.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial *i* the value is `isreward[i-1]`; for the first trial of a session it is 0. Constant across all timepoints of the trial, stored as input row 3.

ii.
```python
prev_outcome = int(isreward[ti-1]) if ti > 0 else 0
...
inp[3] = prev_outcome
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 8: "**Previous trial outcome**: First trial defaults to 0", and Step 10 Check 2: "Previous trial outcome correctly tracks previous trial's reward ✓". This follows the instruction "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)"; 0 for the first trial is the natural convention since no previous trial exists.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` behavior time series plus the active reward-zone coordinates for that trial. The zone is **not** inferred from the `reward_zone` time series; it is parsed from the VR scene name embedded in the NWB `identifier` field (e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`), using the same scene→zone mapping as the reference repository's `behavior.get_reward_zones()`, with the switch at trial 30.

ii.
```python
REWARD_ZONE_DICT = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
SWITCH_TRIAL = 30

def parse_scene_name(identifier):
    return identifier.split('/')[-1]

def get_reward_zones_from_scene(scene, n_trials, change_trial=SWITCH_TRIAL):
    zone_to_coords = {'A': [80,130], 'B': [200,250], 'C': [320,370]}
    rz_coords = np.zeros((n_trials, 2))
    rz_labels = np.empty(n_trials, dtype='U1')
    if 'Training' in scene or 'RunningTraining' in scene:
        rz_coords[:] = [275, 325]; rz_labels[:] = 'T'
        return rz_coords, rz_labels
    if '_to_Env' in scene:
        parts = scene.split('_to_')
        pre_zone  = parts[0][-1]
        post_zone = parts[1][-1]
        ct = min(change_trial, n_trials)
        rz_coords[:ct] = zone_to_coords[pre_zone];  rz_labels[:ct] = pre_zone
        rz_coords[ct:] = zone_to_coords[post_zone]; rz_labels[ct:] = post_zone
        return rz_coords, rz_labels
    if '_to_' in scene:
        parts = scene.split('_to_')
        pre_zone  = parts[0][-1]
        post_zone = parts[1]
        ct = min(change_trial, n_trials)
        rz_coords[:ct] = zone_to_coords[pre_zone];  rz_labels[:ct] = pre_zone
        rz_coords[ct:] = zone_to_coords[post_zone]; rz_labels[ct:] = post_zone
        return rz_coords, rz_labels
    if 'Location' in scene:
        zone = scene.split('Location')[-1][0]
        rz_coords[:] = zone_to_coords[zone]; rz_labels[:] = zone
        return rz_coords, rz_labels
    raise ValueError(f"Cannot parse scene: {scene}")
...
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

iii. CONVERSION_NOTES.md Step 1: "`get_reward_zones()` | behavior.py | LOADING | Maps scene to reward zone coords, switch at trial 30"; "Reward zone dict: A=[80,130], B=[200,250], C=[320,370]"; "Switch at trial 30 (0-indexed)". Step 4 records that the AI investigated the NWB `reward_zone` variable and rejected it as a zone label: "reward_zone NWB | cumsum of rzone | values 0-6 | Cumulative count, not distance". The trajectory shows it then found the scene name in the `identifier` field ("This is exactly what I need to determine the reward zone location per trial") and cross-referenced it against `sessions_dict.py` in the reference repo. Step 10 Check 2: "Reward zone labels correctly switch at trial 30 ✓". (I independently validated this on 1,669 trials from 25 random sessions: the position at which `reward_zone` first becomes non-zero always falls inside the scene-derived zone.)

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is first clipped to [0, 450] cm. The signed distance is then the distance to the nearest edge of the active zone: `pos - zone_start` when before the zone (negative), `pos - zone_end` when past it (positive), and exactly 0 while inside the zone.

ii.
```python
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
...
rz_s, rz_e = rz_coords[ti]
dist = np.where(trial_pos < rz_s, trial_pos - rz_s,
               np.where(trial_pos > rz_e, trial_pos - rz_e, 0.0))
dist_disc = discretize_distance(dist)
```

iii. CONVERSION_NOTES.md Step 5 mapping: "position → dist to RZ | output[0] | Signed distance, 7 bins". The instruction is "Distance to any location in the reward zone", which is why anywhere inside the zone maps to 0. Step 10 Check 2: "**Output data check**: Manually computed distance_to_rz, position, speed, lick discretizations. All `np.array_equal` checks passed ✓". The clipping is not separately justified in the notes; it exists to remove the teleport-zone sentinel positions (down to −50) that can bleed into a trial's first sample.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned by explicit boolean masks: `<-50` → 0; `[-50,-10)` → 1; `[-10,0)` → 2; `==0` → 3; `(0,10]` → 4; `(10,50]` → 5; `>50` → 6.

ii.
```python
def discretize_distance(d):
    r = np.zeros_like(d, dtype=np.int64)
    r[d < -50] = 0
    r[(d >= -50) & (d < -10)] = 1
    r[(d >= -10) & (d < 0)] = 2
    r[d == 0] = 3
    r[(d > 0) & (d <= 10)] = 4
    r[(d > 10) & (d <= 50)] = 5
    r[d > 50] = 6
    return r
...
'output_values': [
    ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '>0 to +10cm', '+10 to +50cm', '> +50cm'],
    ...
]
```

iii. The bin edges are copied from the Decoder Task specification in the instructions, with a dedicated `d == 0` class for "in the zone". The resulting distribution reported in `verification_full_out.txt` — {0: 0.253, 1: 0.102, 2: 0.074, 3: 0.237, 4: 0.021, 5: 0.072, 6: 0.242} — is documented in CONVERSION_NOTES.md as sensible (the 0-class fraction ≈ zone width relative to track). The masks treat the exact boundaries ±10 and ±50 as belonging to the inner bin; these are measure-zero for a continuous position variable.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same trial slice `ts:te` as the neural matrix, so the output array has the same number of columns as the neural array and index *k* refers to the same imaging frame.

ii.
```python
ts, te = trial_start_inds[ti], teleport_inds[ti]
nf = te - ts
trial_neural = np.nan_to_num(dff[:, ts:te], ...)
trial_pos    = np.clip(position[ts:te], 0, TRACK_LENGTH)
...
out = np.zeros((6, nf), dtype=np.int64)
out[0] = dist_disc
```

iii. As for 2-d/3-c: the ophys and VR streams share one sample index in the NWB, so a common slice guarantees alignment. `--show-processing` plots position, speed, raw distance and the discretised outputs on a common time axis for two sessions to make any misalignment visible; CONVERSION_NOTES.md Step 7 reports no anomalies.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior time series, in cm along the 450 cm VR corridor.

ii.
```python
position = behav['position/data'][:]
...
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
```

iii. CONVERSION_NOTES.md Step 2 lists `position` among the behavior time series, and Step 3 records the paper's "Track length | 450 cm". Step 5 mapping: "position | output[1] | 5 bins of 90cm".

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Only clipping to [0, 450] cm and then discretisation; no smoothing, unwrapping or re-referencing.

ii.
```python
TRACK_LENGTH = 450.0
...
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
pos_disc  = discretize_position(trial_pos)
...
out[1] = pos_disc
```

iii. The AI's Step 2 exploration found position ranging from −50 to ~450 (the −50 / −500 values belong to the inter-trial teleport period and to `scanning == -1` frames). Clipping keeps such samples in the first/last bin rather than creating out-of-range classes. Step 10 Check 2 reports the discretisation was re-derived from the raw file and matched with `np.array_equal`.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm bins with open ends: `<90` → 0; `[90,180)` → 1; `[180,270)` → 2; `[270,360)` → 3; `>=360` → 4.

ii.
```python
def discretize_position(p):
    r = np.zeros_like(p, dtype=np.int64)
    r[p < 90] = 0
    r[(p >= 90) & (p < 180)] = 1
    r[(p >= 180) & (p < 270)] = 2
    r[(p >= 270) & (p < 360)] = 3
    r[p >= 360] = 4
    return r
...
['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
```

iii. Exactly the "5 equal-sized bins spanning the 450 cm track" from the Decoder Task section. The observed distribution {0.211, 0.178, 0.231, 0.227, 0.154} is reported in `verification_full_out.txt`; the bins are not exactly uniform because the animal's occupancy varies with running speed along the track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `ts:te` trial slice as the neural matrix — no additional alignment.

ii.
```python
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
out[1] = discretize_position(trial_pos)
```

iii. Same reasoning as 7-d: single shared sample index for ophys and behaviour in the NWB, checked visually in the `--show-processing` plots and by the Step 10 raw-file spot checks.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior time series, which holds a per-frame lick count (values can exceed 1).

ii.
```python
lick = behav['lick/data'][:]
...
trial_lick = lick[ts:te]
```

iii. CONVERSION_NOTES.md Step 2 lists `lick` among the available behavior series; Step 5 mapping: "lick | output[3] | Binary (>0)".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarised by thresholding at zero: any positive count becomes 1. No lick-error trial exclusion, no smoothing, no spatial binning.

ii.
```python
lick_bin = (trial_lick > 0).astype(np.int64)
...
out[3] = lick_bin
...
['no_lick', 'lick'],
```

iii. The instructions specify "Lick, time-varying. 0 = no, 1 = yes", and the raw counts are >1 on some frames, so thresholding is required. CONVERSION_NOTES.md Step 3 notes the paper's "Lick error trials | 81/12376 (~0.65%)" removal but Step 5 does not apply it; the resulting lick fraction (0.230 of all frames) is reported in Step 9/verification as plausible.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `ts:te` trial slice as the neural data.

ii.
```python
trial_lick = lick[ts:te]
out[3] = (trial_lick > 0).astype(np.int64)
```

iii. Same shared-index argument as 7-d/8-d.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The VR scene name in the NWB `identifier` field, parsed by `get_reward_zones_from_scene` — see 7-a. The `reward_zone` behavior time series is used only for the reward-outcome gate, not for the zone label.

ii. See 7-a; the label array is the second return value:
```python
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

iii. See 7-a. CONVERSION_NOTES.md Step 5 mapping: "scene → RZ label | output[4] | A=0, B=1, C=2". Step 9 reports the resulting class balance "RZ balance | A:32.9% B:33.7% C:33.5% ✓", used as the sanity check that the scene parsing is right.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The per-trial letter label is mapped to an integer (A→0, B→1, C→2) and broadcast across all timepoints of the trial as output row 4. Trials before index 30 get the pre-switch zone, trials from 30 on get the post-switch zone; non-switch scenes get one zone for the whole session. A `'T'` (training) label would fall through the `.get(..., 0)` default to class 0.

ii.
```python
rz_label_int = {'A': 0, 'B': 1, 'C': 2}.get(rz_labels[ti], 0)
...
out[4] = rz_label_int
...
['zone_A', 'zone_B', 'zone_C'],
```

iii. The instruction is "Reward zone location, per-trial. 0 = A, 1 = B, 2 = C". The switch-at-trial-30 rule comes from the reference `get_reward_zones(sess, rz_dict=None, change_trial=30)` and the paper ("Each switch occurred after 30 trials", quoted in CONVERSION_NOTES.md Step 3). No `Training`/`RunningTraining` scene exists in the released 152 sessions, so the `'T'` fallback is dead code here.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` event timestamps together with the `reward_zone` behavior time series — the same `isreward` array used for the previous-trial input (see 6-a).

ii.
```python
reward_timestamps = behav['Reward/timestamps'][:]
reward_zone_raw   = behav['reward_zone/data'][:]
...
has_rzone  = np.any(reward_zone_raw[ts:te] > 0)
has_reward = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
if has_reward and has_rzone:
    isreward[ti] = 1
```

iii. CONVERSION_NOTES.md Step 1: "`get_trial_types()` | behavior.py | LOADING | Determines reward per trial (reward>0 AND rzone>0)" — the AI deliberately reproduced the reference repo's two-condition definition rather than counting reward events alone.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. One binary value per trial, broadcast across all timepoints as output row 5. The reward timestamps are compared against the trial's start and end **timestamps** (not sample indices), inclusive at both ends, because the `Reward` series carries its own clock.

ii.
```python
t_start = frame_timestamps[ts]
t_end   = frame_timestamps[min(te, n_timepoints-1)]
has_reward = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
...
out[5] = isreward[ti]
...
['no_reward', 'reward'],
```

iii. CONVERSION_NOTES.md Step 12 "Issues Found and Resolved": "Reward timestamp alignment issue for multi-plane sessions → fixed by using actual NWB timestamps". Step 9/10 use the resulting reward rate as the main behavioural consistency check: "Reward rate | ~85% (paper) | 84.3% (converted) | ✓", against the paper's "Reward was randomly omitted on approximately 15% of trials".

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases are handled:
- **NaN/Inf in dF/F**: every trial's neural slice is passed through `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)`. This covers samples where the maximin baseline came out at or near zero (two trials in the full dataset produced Inf).
- **Unequal numbers of trial starts and teleports**: both index arrays are truncated to the shorter length before pairing.
- **Invalid environment samples**: `-1` values (non-scanning frames) are excluded before taking the per-trial median; a trial with no valid sample defaults to environment 0.
- **Degenerate trials/sessions**: trials with fewer than 2 frames and sessions with fewer than 2 usable trials are skipped.

Not handled: a length mismatch between the ophys arrays and the behavior arrays. There is no cropping or assertion; `n_timepoints` is taken from `F.shape[0]` while all trial indices come from the behaviour arrays.

ii.
```python
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
...
n_trials = min(len(trial_start_inds), len(teleport_inds))
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds    = teleport_inds[:n_trials]
...
valid = env_vals[env_vals >= 0]
if len(valid) > 0:
    trial_env[ti] = int(np.round(np.median(valid)))
...
if nf < 2: continue
...
if len(neural_trials) < 2:
    print(f"    Skipping: {len(neural_trials)} valid trials"); return None
```

iii. CONVERSION_NOTES.md Step 6: "Handles NaN/Inf from baseline division (replace with 0)"; Step 10 Check 1: "Fixed 2 trials with Inf values from dF/F baseline division (now handled in code)"; Step 10 Check 5 enumerates the edge cases checked (short trials, long trials, multi-plane, first trial, Inf values, missing m11 sessions). The AI reports `verification_full_out.txt` clean of errors and warnings. (I verified that ten sessions — m17 ses-04/06 and eight m18 sessions — have exactly one more ophys sample than behaviour samples. Because the neural array is the *longer* one and all trial indices come from the behaviour stream, the extra sample is simply never read, so the missing guard is harmless for this dataset, but the code would silently emit mismatched-length trials if the neural array were ever the shorter one.)

## 13-a. What are the most time-consuming steps of the code?

i. The script is instrumented with `time.time()` around the whole session and around the dF/F computation, and prints a running estimate of remaining time. From `conversion_full_out.txt`, per session: dF/F takes 0.1–1.4 s and total session time 0.3–2.9 s, scaling with cell count, so the two dominant costs are (a) reading F and Fneu out of HDF5 and (b) the per-trial min/max filtering and Gaussian smoothing in `compute_dff`. The whole conversion took 4.5 min; the final step — pickling 9.4 GB of float32 dF/F — is the other large cost.

ii.
```python
t0 = time.time()
...
t_dff = time.time()
dff = compute_dff(F, Fneu, trial_start_inds, teleport_inds)
print(f"    dF/F: {time.time()-t_dff:.1f}s, shape={dff.shape}")
...
print(f"    Done: {len(neural_trials)} trials, {n_cells} cells, {time.time()-t0:.1f}s")
...
dt = time.time() - t0
rem = (len(nwb_files) - i - 1) * dt
print(f"    Time: {dt:.1f}s, Est remaining: {rem/60:.1f}min")
```

iii. CONVERSION_NOTES.md Step 6: "~2s/session processing time"; Step 7: "~2s/session, full estimated: ~5 min (actual: 4.5 min)". This is comfortably under the instructions' 15-minute budget, so no further optimisation was pursued.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops remain, all over trials or planes:
- the three per-trial loops inside `compute_dff` (baseline mask, min/max filtering, smoothing) — unavoidable in spirit, since the baseline must be computed within each lap, but the mask loop could be replaced by a vectorised interval fill, and the filtering/smoothing loops could be merged into one pass;
- the reward-outcome loop over trials — `np.searchsorted` of the reward timestamps into `frame_timestamps` would give all trials at once instead of an O(n_trials × n_rewards) scan;
- the environment loop over trials — replaceable with `np.add.reduceat`/segment statistics;
- the main trial-building loop, which slices and discretises per trial. The discretisation (`discretize_distance`, `discretize_position`, `discretize_speed`, the lick threshold) could be applied once to the whole session array before splitting, since the zone coordinates are the only per-trial quantity.

The discretisation helpers themselves are already vectorised (boolean masks over whole arrays), as is the neuropil subtraction.

ii.
```python
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
    has_rzone = np.any(reward_zone_raw[ts:te] > 0)
    t_start = frame_timestamps[ts]
    t_end   = frame_timestamps[min(te, n_timepoints-1)]
    has_reward = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
...
for ti in range(n_trials):
    env_vals = environment[ts:te]
    valid = env_vals[env_vals >= 0]
...
for s, e in zip(trial_start_inds, teleport_inds):
    if s >= e: continue
    w = min(BASELINE_WINDOW, e - s)
    flow[:, s:e] = scipy.ndimage.minimum_filter1d(f_[:, s:e], w, axis=-1)
    flow[:, s:e] = scipy.ndimage.maximum_filter1d(flow[:, s:e], w, axis=-1)
```

iii. CONVERSION_NOTES.md does not analyse these loops individually (the "Code inefficiencies identified / Code speedups added" slots of the template were not filled in); the AI's stated justification is simply that the measured runtime of ~2 s/session and 4.5 min total was well inside budget, so no vectorisation was needed.

## 13-c. What processing does the code repeat multiple times?

i. Very little. Each NWB file is opened exactly once and each array read once, in a single pass — there is no separate survey/precompute pass. Within a session the only repetition is that `compute_dff` walks the trial list three times (mask, baseline, smoothing) instead of once, and `np.clip(position[ts:te], 0, TRACK_LENGTH)` is recomputed inside `_plot_processing` for the plotted trials. Across runs, the sample conversion re-does two sessions that the full conversion then processes again, but that is prescribed by the instructions' Step 7/Step 9 workflow.

ii.
```python
nanmask = np.zeros(n_tp, dtype=bool)
for s, e in zip(trial_start_inds, teleport_inds): ...      # pass 1
for s, e in zip(trial_start_inds, teleport_inds): ...      # pass 2 (baseline)
for s, e in zip(trial_start_inds, teleport_inds): ...      # pass 3 (smoothing)
...
# in _plot_processing:
tp = np.clip(position[ts:te], 0, TRACK_LENGTH)
d  = np.where(tp < rz_s, tp - rz_s, np.where(tp > rz_e, tp - rz_e, 0.0))
```

iii. CONVERSION_NOTES.md Step 6 describes the design as a single streaming pass per file ("Handles single-plane and multi-plane NWB files ... ~2s/session"), and Step 7 records that the measured time made further restructuring unnecessary.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A small amount:
- `effective_rate` is computed from the timestamps for every session and printed, but the saved `time_bin_size` comes from the hard-coded `TARGET_RATE_HZ` — the measured value is never compared to it or stored.
- `compute_dff` produces a full NaN-filled `dff` array over the entire recording, including the inter-trial teleport periods, which are then dropped when trials are sliced out.
- The `nansmooth` helper builds and smooths a weights array on every call even though, by that point, there are no NaNs inside a trial window.
- `reward_zone_raw` is read in full but used only through `np.any(... > 0)` per trial.
- Dead branches: the `'Training'` / `'RunningTraining'` scene case (no such session exists in the released data) and the `'T'` → 0 label fallback.
- Non-essential outputs: `processing_*.png` plots and `sample_data.pkl`, both required by the instructions rather than by the conversion itself.

ii.
```python
dt = np.median(np.diff(frame_timestamps))
effective_rate = 1.0 / dt          # printed, then discarded
...
time_bin_ms = 1000.0 / TARGET_RATE_HZ   # hard-coded value used instead
...
if 'Training' in scene or 'RunningTraining' in scene:
    rz_coords[:] = [275, 325]; rz_labels[:] = 'T'   # never reached for this dataset
...
rz_label_int = {'A': 0, 'B': 1, 'C': 2}.get(rz_labels[ti], 0)
```

iii. None of this is discussed in CONVERSION_NOTES.md. The `Training`/`'T'` branches are a faithful port of the reference `get_reward_zones()`, which does cover training scenes, so they are defensive rather than wrong; the rest is negligible relative to the I/O and filtering cost.
