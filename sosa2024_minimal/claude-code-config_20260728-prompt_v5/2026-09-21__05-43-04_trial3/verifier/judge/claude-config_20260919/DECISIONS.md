# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcoded a copy of the reference repository's `sessions_dict.py` (mouse -> list of `{scene, exp_day}`) into the conversion script, and uses the **keys of that dictionary** as the subject list. For each subject it looks for the directory `/app/data/sub-<id>`, lists every `*.nwb` file in it, parses the session/day number out of the filename (`sub-m11_ses-03_behavior+ophys.nwb` -> day 3), and looks up the matching `exp_day` entry in the hardcoded dict to obtain the "scene" (environment + reward-zone schedule). Files with no matching dict entry are skipped with a warning. NWB files are read **directly with `h5py`** (not `pynwb`); the groups read are `processing/ophys/{Deconvolved,Fluorescence,Neuropil}/plane*/data`, `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`, `general/optophysiology/ImagingPlane/imaging_rate`, and `processing/behavior/BehavioralTimeSeries/{position,speed,lick,trial_start,teleport,environment,Reward}`. Everything is accumulated into per-session lists and pickled to `/app/converted_data.pkl`. The run loaded all 152 NWB files (11 mice, 12216 trials, 138276 neurons).

ii.
```python
subjects = sorted(SESSIONS_DICT.keys(), key=lambda x: int(x[1:]))

for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, f'sub-{subj}')
    if not os.path.isdir(subj_dir):
        print(f"Warning: directory {subj_dir} not found, skipping")
        continue

    # Get available NWB files
    nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])

    for nwb_file in nwb_files:
        # Extract session day from filename: sub-m11_ses-03_behavior+ophys.nwb
        ses_str = nwb_file.split('_ses-')[1].split('_')[0]
        exp_day = int(ses_str)

        scene = None
        for entry in SESSIONS_DICT[subj]:
            if entry['exp_day'] == exp_day:
                scene = entry['scene']
                break
        ...
        neural, inp, out, n_neurons, info = process_session(subj, exp_day, nwb_path, scene)
```
(`/app/convert_data.py:665-696`)

```python
f = h5py.File(nwb_path, 'r')
dec_group = f['processing/ophys/Deconvolved']
fl_group  = f['processing/ophys/Fluorescence']
neu_group = f['processing/ophys/Neuropil']
planes = sorted([k for k in dec_group.keys() if k.startswith('plane')])
...
beh = f['processing/behavior/BehavioralTimeSeries']
position = beh['position/data'][:]
speed    = beh['speed/data'][:]
lick     = beh['lick/data'][:]
timestamps = beh['position/timestamps'][:]
trial_start_signal = beh['trial_start/data'][:]
teleport_signal    = beh['teleport/data'][:]
env_signal         = beh['environment/data'][:]
reward_ts = beh['Reward/timestamps'][:]
f.close()
```
(`/app/convert_data.py:452-487`)

iii. From the trajectory: the AI explored the NWB layout with `h5py`, found it contained `Fluorescence`, `Neuropil`, `Deconvolved`, `iscell` and the behavioral time series, and decided direct HDF5 access was sufficient (the reference repository "doesn't load from NWB files directly - it uses TwoPUtils and suite2p outputs", step 43). It cross-checked the subject list against the paper: "I count 11 mice (sub-m3, m4, m7, m11-m15, m17-m19), matching the paper's 11 switch-condition mice, with the fixed-condition mice (GCAMP2/6/10) apparently not included" (step 46). It adopted `sessions_dict` because it "encodes the reward zone schedule" and because NWB `ses-NN` numbers map directly onto `exp_day` (step 48).

## 1-b. How are the data split into subjects?

i. One subject per `sub-<id>` directory, but the iteration list is the hardcoded `SESSIONS_DICT` keys (`m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19`) rather than a listing of `/app/data`. `subjects` is stored in the output dict, and each session appends `subjects.index(subj)` to `subject_idx`. Subjects are sorted numerically.

ii.
```python
subjects = sorted(SESSIONS_DICT.keys(), key=lambda x: int(x[1:]))
...
all_subject_idx.append(subjects.index(subj))
...
'subjects': subjects,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```
(`/app/convert_data.py:665, 705, 720-721`)

iii. The AI reasoned that the GCAMP naming in the reference code maps one-to-one onto the NWB subject IDs (`GCAMP{N}` -> `sub-m{N}`), and that the three mice present in the reference code but missing from `/app/data` (GCAMP2/6/10) are the fixed-condition mice that are not part of this dandiset (steps 46, 48). It therefore treated the dict keys and the data directories as the same set of 11 switch-task mice.

## 1-c. How are the data split into sessions?

i. One session per NWB file. The day index is parsed from the `ses-NN` field of the filename and matched to `exp_day` in `SESSIONS_DICT` to recover the scene. m11 correctly yields 12 sessions (imaging began on day 3); all others yield 14, for 152 sessions. A session is dropped only if it has no matching scene entry or fewer than 2 usable trials (neither occurred).

ii.
```python
ses_str = nwb_file.split('_ses-')[1].split('_')[0]
exp_day = int(ses_str)
scene = None
for entry in SESSIONS_DICT[subj]:
    if entry['exp_day'] == exp_day:
        scene = entry['scene']
        break
if scene is None:
    print(f"Warning: no scene found for {subj} day {exp_day}, skipping")
    continue
...
if neural is None or len(neural) < 2:
    print(f"  Skipping {subj} day {exp_day}: insufficient data")
    continue
```
(`/app/convert_data.py:677-700`)

iii. "Sessions have exp_day 0 (RunningTraining_scan, no imaging) through 14, and the NWB files use ses-01 through ses-14 matching exp_day, so for m11 starting at exp_day 3 I'd need ses-03 through ses-14" (step 48). The script header adds that day-0 running-training sessions are excluded because "they lack the main task structure" (no such files exist in `/app/data`, so this is a no-op). No attempt is made to align cells across days; each session is an independent population.

## 1-d. How are the data split into trials?

i. Trials run from each `trial_start` pulse to the corresponding `teleport` pulse, exactly as in the reference. Trial boundaries are computed **after** cropping neural and behavioral streams to a common length: `trial_start_idx = np.where(trial_start_signal > 0)[0]`, `teleport_idx = np.where(teleport_signal > 0)[0]`, paired by position and truncated to `min(len(starts), len(ends))`. The trial slice is `[trial_start_idx[t], teleport_idx[t])`, i.e. the teleport sample itself is excluded. This produced 12216 trials with T mean 216.78 / median 197.49 / min 96 / max 3359 — identical to the expert reference.

ii.
```python
trial_start_idx = np.where(trial_start_signal > 0)[0]
teleport_idx = np.where(teleport_signal > 0)[0]

n_trials = min(len(trial_start_idx), len(teleport_idx))
trial_start_idx = trial_start_idx[:n_trials]
teleport_idx = teleport_idx[:n_trials]
...
for t in range(n_trials):
    t_start = trial_start_idx[t]
    t_end = teleport_idx[t]
    if t_end <= t_start:
        continue
    neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
```
(`/app/convert_data.py:510-563`)

iii. The AI found from exploration that "position of -500 means inter-trial interval" and "trial numbers: -1 for ITI" (step 18) and concluded that the `trial_start`/`teleport` pulse pair delimits the on-track portion of each lap: "I mainly need to pull trial boundaries from trial_start/teleport signals" (step 54). It rejected the stored `trial number` series as the segmentation source.

## 1-e. How are trials filtered based on quality controls?

i. Essentially no quality filtering. A trial is dropped only if it is degenerate: `t_end <= t_start`, or fewer than 2 timepoints. A whole session is dropped if it yields fewer than 2 trials (needed for the decoder's train/test split) or zero neurons after cell filtering. There is no minimum-duration criterion, no speed criterion, and no behavioral-performance criterion. In practice nothing was dropped: the output has the same 12216 trials and the same T statistics (min 96 samples) as the reference, which applied a 50-sample minimum.

ii.
```python
if t_end <= t_start:
    continue
...
n_tp = neural.shape[1]
if n_tp < 2:
    continue
```
(`/app/convert_data.py:558-567`)
```python
if n_neurons == 0:
    return None, None, None, 0, {}
...
if neural is None or len(neural) < 2:
    print(f"  Skipping {subj} day {exp_day}: insufficient data")
    continue
```
(`/app/convert_data.py:536-537, 698-700`)

iii. The AI decided to "keep all trials without extra filtering" (step 48) and explicitly declined the paper's 2 cm/s speed threshold: "since that would create gaps in the continuous time series the decoder needs, I'll skip that threshold and instead include speed itself as one of the decoder's output variables" (step 48). The only guards are the structural ones needed for the decoder to run.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrices are the NWB **`processing/ophys/Deconvolved`** arrays, taken as-is. `Fluorescence` and `Neuropil` are also loaded, but only to compute a dF/F trace that is used exclusively for interneuron detection and then discarded. Planes are concatenated along the ROI axis (`plane0` then `plane1`), which matches the ROI ordering in `PlaneSegmentation` (verified: `planeIdx` is sorted).

ii.
```python
dec_group = f['processing/ophys/Deconvolved']
fl_group = f['processing/ophys/Fluorescence']
neu_group = f['processing/ophys/Neuropil']
planes = sorted([k for k in dec_group.keys() if k.startswith('plane')])

dec_planes = [dec_group[p]['data'][:] for p in planes]
fl_planes = [fl_group[p]['data'][:] for p in planes]
neu_planes = [neu_group[p]['data'][:] for p in planes]

# Concatenate across planes (paper: "planes were pooled for all analyses")
deconvolved = np.concatenate(dec_planes, axis=1)   # (n_frames, n_rois)
fluorescence = np.concatenate(fl_planes, axis=1)
neuropil_data = np.concatenate(neu_planes, axis=1)
```
(`/app/convert_data.py:455-467`)
```python
deconv_filtered = deconvolved[:, final_cell_indices]
...
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
```
(`/app/convert_data.py:540, 563`)

iii. The AI assumed the stored `Deconvolved` array is the end product of the paper's pipeline: "Since the NWB files already include Fluorescence, Neuropil, and Deconvolved fields directly, I should just use the precomputed Deconvolved data rather than recomputing it" (step 18); "Given the session description mentions 'processed suite2p data,' I'll treat the Deconvolved field as the output of the full standard pipeline" (step 18); "Looking at the Deconvolved data values: Range 0 to 15054, mean ~32, 72% zeros... This looks like deconvolved calcium activity" (step 33). It never verified the assumption against the paper's own output: in fact the stored `Deconvolved` is suite2p's deconvolution of **raw fluorescence**, not of dF/F (per-cell mean event amplitude correlates with per-cell mean raw F at r = 0.93 and reaches values in the thousands), whereas the paper deconvolves the maximin dF/F.

## 2-b. How is the `neural` data processed?

i. No processing at all is applied to the neural signal: the `Deconvolved` array is sliced per trial, transposed to `(n_neurons, n_timepoints)` and cast to `float32`. No neuropil subtraction, no maximin baseline, no dF/F normalization, no Gaussian smoothing and no OASIS deconvolution are performed on the data that is saved — the AI's re-implementation of the paper's `preprocessing.dff` (neuropil subtraction with `neu_coef = 0.7`, per-trial maximin baseline with a 300-sample window after sigma = 15 Gaussian smoothing, `(F - baseline)/|baseline|`, sigma = 2 Gaussian smoothing) is run but its output feeds only the interneuron test.

ii.
```python
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
```
(`/app/convert_data.py:563`)
```python
def compute_dff_for_interneuron_detection(F, Fneu, trial_starts, trial_ends, frame_rate=15.5):
    ...
    f_ -= neu_coef * fneu_
    ...
    for start, stop in zip(trial_starts, trial_ends):
        f_[:, start:stop] += neu_coef * np.nanmean(fneu_[:, start:stop], axis=1, keepdims=True)
        bl = np.copy(f_[:, start:stop])
        for c in range(n_cells):
            bl[c] = ndi.gaussian_filter1d(bl[c], sigma=15)
        bl = ndi.minimum_filter1d(bl, size=window, axis=-1)
        bl = ndi.maximum_filter1d(bl, size=window, axis=-1)
        baseline[:, start:stop] = bl
    dff[:, nanmask] = (f_[:, nanmask] - baseline[:, nanmask]) / np.abs(baseline[:, nanmask])
```
(`/app/convert_data.py:334-410`, used only at `:522-526`)

iii. "For the actual neural activity signal, I should use the Deconvolved data, since it already reflects the full processing pipeline (neuropil subtraction, dF/F, OASIS deconvolution), while Fluorescence and Neuropil are the raw inputs I'd need to compute dF/F myself for the interneuron check" (step 27). At step 54 it acknowledged it could not check the channel configuration and chose to "trust that the Deconvolved data already reflects proper processing and use it directly", while building "a simplified dF/F pipeline... just for flagging interneurons".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both from the paper. (1) suite2p manual curation: only ROIs with `iscell[:,0] == 1` are kept. (2) Putative interneurons: for each surviving ROI, the Pearson correlation between its (recomputed) dF/F and the running speed is taken over in-trial samples, and cells with r > 0.5 are removed. No other neuron-level criterion (no SNR, no activity-rate threshold). The result was 138276 neurons over 152 sessions (mean 909.7, min 154, max 2323), against 138298 (mean 909.9, min 154, max 2320) for the expert reference.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
...
cell_indices = np.where(cell_mask)[0]
F_cells = fluorescence[:, cell_indices]
Fneu_cells = neuropil_data[:, cell_indices]

dff = compute_dff_for_interneuron_detection(F_cells, Fneu_cells, trial_start_idx, teleport_idx, frame_rate)
is_int = detect_interneurons(dff, speed, threshold=0.5)
...
final_cell_indices = cell_indices[~is_int]
```
(`/app/convert_data.py:491, 518-533`)
```python
def detect_interneurons(dff, speed, threshold=0.5):
    nanmask = ~np.isnan(dff[0, :])
    for c in range(n_cells):
        if nanmask.sum() > 10:
            r = np.corrcoef(dff[c, nanmask], speed[nanmask])[0, 1]
            if np.isnan(r):
                continue
            is_int[c] = r > threshold
```
(`/app/convert_data.py:413-439`)

iii. "The NWB's `iscell` field already reflects manual curation excluding poor ROIs... but the paper also flags additional interneurons via Pearson correlation >0.5 between dF/F and running speed. So my plan is to first filter cells using `iscell`, then separately compute dF/F to apply this speed-correlation exclusion criterion" (step 27). It noticed the repository default is `r_thresh=0.3` but chose 0.5 after checking the analysis notebooks and the Methods: "the code uses `r_thresh=0.3` as the default, but the paper says '>0.5'. Let me check which value is actually used" (steps 36, 54 — `int_thresh = 0.5`).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, which requires no extra work because each trial array simply begins at the `trial_start` frame: sample 0 of every trial *is* the alignment event. No pre-event window is included and no padding/truncation is applied; trials keep their native, variable length. Metadata records `temporal_alignment_event = 'start of trial (first frame of track entry)'`, `off_start = 0.0`, `off_end = None`.

ii.
```python
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
...
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
```
(`/app/convert_data.py:563, 623`)
```python
'temporal_alignment_event': 'start of trial (first frame of track entry)',
'off_start': 0.0,
'off_end': None,  # Variable across trials
```
(`/app/convert_data.py:759-761`)

iii. Stated in the script header: "Temporal alignment: Aligned to trial start (time=0 at first frame of each trial)". The AI noted that trial durations vary and therefore left `off_end = None` ("handle temporal alignment to trial start with variable trial durations", step 54).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: data are kept at the native imaging frame rate, one sample per stored frame (~64.5 ms). `metadata['time_bin_size']` is computed as `1000 / median(frame_rate across sessions)` where `frame_rate` is the NWB `ImagingPlane/imaging_rate`; because 124 of 152 sessions are single-plane the median is 15.5078 Hz and the stored bin size (64.4836 ms) is correct and matches the reference exactly. However, the AI did **not** divide `imaging_rate` by the number of planes: for the 28 two-plane sessions (m17, m18) `imaging_rate` is the 31.0156 Hz scanner rate while the per-plane sampling period is 64.5 ms, so the per-session `dt` used to build the time input is wrong by a factor of two there (see 3-b).

ii.
```python
frame_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
...
dt = 1.0 / frame_rate  # time per frame in seconds
```
(`/app/convert_data.py:472, 550`)
```python
frame_rates = [info['frame_rate'] for info in all_session_info]
median_frame_rate = np.median(frame_rates)
time_bin_ms = 1000.0 / median_frame_rate
...
'time_bin_size': time_bin_ms,
```
(`/app/convert_data.py:710-758`)

iii. Script header: "Time bins: Native imaging frame rate (~15.5 Hz, ~64.5 ms per bin)"; step 54: "bin time at the native ~15.5 Hz frame rate". The AI took the median across sessions to guard against session-to-session variability, but never reconciled the 31 Hz value it saw in the two-plane files with the 15.5 Hz per-plane rate that the behavior timestamps imply.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is **not** derived from the behavioral timestamps. It is synthesized from the sample index within the trial and the nominal frame period `dt = 1/imaging_rate`. The NWB `position/timestamps` array is loaded, but is used only to window reward events.

ii.
```python
frame_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
...
dt = 1.0 / frame_rate  # time per frame in seconds
...
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
```
(`/app/convert_data.py:472, 550, 623`)

iii. The AI treated the recording as evenly sampled at the imaging frame rate ("the NWB already aligns behavioral series to imaging frames", step 54) and so generated the time axis analytically rather than reading it off the timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. `np.arange(n_timepoints) * dt`, i.e. index x frame period, starting at exactly 0.0 for every trial. Values are `float32` and time-varying, broadcast into row 0 of the `(4, n_timepoints)` input array. Because `dt` uses the raw `imaging_rate`, the 28 two-plane sessions (m17, m18) get `dt = 0.0322 s` instead of the true `0.0645 s`, so their time-from-trial-start values are exactly half the real elapsed time. The remaining 124 sessions are correct (the global range, 0–216.5 s, still matches the reference because the longest trial is in a single-plane mouse).

ii.
```python
time_from_start = np.arange(n_tp, dtype=np.float32) * dt

input_data = np.stack([
    time_from_start,
    np.full(n_tp, env_binary, dtype=np.float32),
    np.full(n_tp, t, dtype=np.float32),
    np.full(n_tp, prev_rewarded, dtype=np.float32),
], axis=0)  # (4, n_timepoints)
```
(`/app/convert_data.py:623-634`)

iii. No explicit justification appears in the trajectory beyond the assumption of uniform sampling at the imaging rate; the AI did not discuss multi-plane sessions or check `imaging_rate` against `np.diff(timestamps)` (which equals 0.0645 s in every file, including the two-plane ones).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the time vector has exactly `n_tp` entries taken from the same trial slice as the neural matrix, with element 0 corresponding to the `trial_start` frame. Before any of this, the neural and behavioral streams are cropped to a common number of frames, so index i of the behavior always refers to frame i of the imaging data. No interpolation or shifting is performed.

ii.
```python
n_frames_neural = deconvolved.shape[0]
n_frames_behav = len(position)
n_frames = min(n_frames_neural, n_frames_behav)
deconvolved = deconvolved[:n_frames]
...
position = position[:n_frames]
...
timestamps = timestamps[:n_frames]
```
(`/app/convert_data.py:494-507`)
```python
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
n_tp = neural.shape[1]
...
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
```
(`/app/convert_data.py:563-623`)

iii. "For behavior, the NWB already aligns behavioral series to imaging frames" (step 54) — the AI treated the behavioral time series as frame-synchronous with the imaging data, so a shared index range is sufficient.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Despite the script header stating that environment comes from the NWB `environment` time series, the implementation derives it from the **hardcoded scene name** for that session: `parse_scene()` extracts the environment number from the `Env1`/`Env2` prefix, and on the day-8 environment-switch sessions the environment flips at trial 30. The NWB `environment` array is loaded (`env_signal`) and cropped, but never used. I verified the two sources agree on every trial of all 152 sessions, so the stored values are the same either way; environment is stored as 0 = ENV1, 1 = ENV2, constant within a trial and broadcast across timepoints.

ii.
```python
env_signal = beh['environment/data'][:]     # loaded, never used again
...
def get_env_for_trial(scene_info, trial_idx):
    env_before, _, env_after, zone_after = parse_scene(scene_info)
    if zone_after is not None and trial_idx >= SWITCH_TRIAL:
        return env_after if env_after is not None else env_before
    return env_before
...
env = get_env_for_trial(scene, t)
env_binary = 0 if env == 1 else 1  # ENV1=0, ENV2=1
```
(`/app/convert_data.py:482, 276-281, 583-584`)

iii. "Environment identity comes from the 0/1 environment time series" (step 54) — the stated plan; but the code uses the `sessions_dict` scenes, which the AI also considered "the reliable source" for the per-trial schedule (step 54). It verified the result post hoc against the data: "Environment transitions at the right sessions (day 8 for most mice, matching the paper)" (step 86).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Parse the scene string; map `Env1 -> 0`, `Env2 -> 1`; on `..._to_Env..` sessions use the post-switch environment from trial 30 onward; broadcast the scalar over the trial's timepoints as row 1 of the input array.

ii.
```python
if '_to_Env' in scene:
    parts = scene.split('_to_')
    before = parts[0]   # 'Env1_C'
    after = parts[1]    # 'Env2_B'
    env_before = int(before[3])
    ...
SWITCH_TRIAL = 30
...
env_binary = 0 if env == 1 else 1
input_data = np.stack([
    time_from_start,
    np.full(n_tp, env_binary, dtype=np.float32),
    ...
```
(`/app/convert_data.py:227, 239-248, 584, 629-634`)

iii. The AI established from the reference `sessions_dict` that "On switch days (3, 5, 7, 8, 10, 12, 14), the first 30 trials use one zone and trials 30+ use the second zone, with day 8 also flipping the environment itself" (step 48) and encoded exactly that rule.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Not from a raw variable: it is the loop counter `t`, i.e. the 0-based index of the trial within the session as defined by the `trial_start`/`teleport` pulse pairs. The NWB `trial number` series is not read.

ii.
```python
for t in range(n_trials):
    t_start = trial_start_idx[t]
    t_end = teleport_idx[t]
    ...
    np.full(n_tp, t, dtype=np.float32),
```
(`/app/convert_data.py:554-632`)

iii. The AI needed a within-session trial index that is consistent with its own trial segmentation and with the trial-30 switch rule it derived from `sessions_dict` (step 48), so it used the sequential index of each segmented trial.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the integer index across the trial's timepoints as row 2 of the input array (`float32`). Note the counter is the index over *all* detected trials, so a skipped degenerate trial would still consume an index (no such trial occurred). Values run 0–99, matching the reference.

ii.
```python
input_data = np.stack([
    time_from_start,
    np.full(n_tp, env_binary, dtype=np.float32),
    np.full(n_tp, t, dtype=np.float32),
    np.full(n_tp, prev_rewarded, dtype=np.float32),
], axis=0)
```
(`/app/convert_data.py:629-634`)

iii. Implicit: the instructions ask for a per-trial continuous "trial number", and the raw index is the natural encoding; keeping it un-normalized preserves the correspondence with the trial-30 switch point.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From `processing/behavior/BehavioralTimeSeries/Reward/timestamps`, the same source as the per-trial reward outcome (11-a). The current trial's outcome is computed by testing whether any reward timestamp falls inside the trial's time window, and that value is carried forward to become the *previous* outcome of the next trial.

ii.
```python
reward_ts = beh['Reward/timestamps'][:]
...
t_start_time = timestamps[t_start]
t_end_time = timestamps[t_end - 1]
rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
```
(`/app/convert_data.py:485, 587-589`)

iii. "Reward outcome maps from Reward timestamps to trial windows" (step 54). The AI had established that the `Reward` series carries its own event timestamps rather than a frame-aligned signal, and that `reward_zone` is a lick-count-like signal, not a reward indicator (step 25).

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A running variable `prev_rewarded` is initialized to 0 for the first trial of each session, written into row 3 of the input array (broadcast over timepoints), and then updated to the current trial's `rewarded` at the end of each loop iteration. So trial t carries the outcome of trial t-1, and the first trial carries 0. (If a trial were skipped as degenerate, the carried value would come from the last *retained* trial; no trials were skipped.)

ii.
```python
prev_rewarded = 0  # For the first trial, no previous trial → 0

for t in range(n_trials):
    ...
    rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
    ...
    input_data = np.stack([
        ...,
        np.full(n_tp, prev_rewarded, dtype=np.float32),
    ], axis=0)
    ...
    # Update previous trial outcome for next iteration
    prev_rewarded = rewarded
```
(`/app/convert_data.py:552-641`)

iii. The instructions define the input as binary (omitted = 0, rewarded = 1) per trial; the AI's header states "Reward outcome: Determined by mapping Reward event timestamps to trial time windows", and the previous-trial value is just that quantity lagged by one trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the behavioral `position` series plus the active reward zone for that trial, where the zone is taken from the hardcoded scene schedule (zone letter before/after trial 30) and converted to fixed boundaries `A: 80-130 cm`, `B: 200-250 cm`, `C: 320-370 cm`. The NWB `reward_zone` series is **not** used to label the zone (the AI determined it is a within-zone lick/interaction counter, not a zone identity). I checked the AI's scene-based labels against the positions at which `reward_zone` becomes non-zero for every trial in all 152 sessions: they agree on 100% of trials, and the observed switch trial is always 30.

ii.
```python
REWARD_ZONES = {
    'A': (80, 130),
    'B': (200, 250),
    'C': (320, 370),
}
SWITCH_TRIAL = 30
...
def get_reward_zone_for_trial(scene_info, trial_idx):
    env_before, zone_before, env_after, zone_after = parse_scene(scene_info)
    if zone_after is not None and trial_idx >= SWITCH_TRIAL:
        return zone_after
    return zone_before
...
pos_trial = position[t_start:t_end]
pos_trial = np.clip(pos_trial, 0, 450)
zone_letter = get_reward_zone_for_trial(scene, t)
zone_start, zone_end = REWARD_ZONES[zone_letter]
```
(`/app/convert_data.py:220-227, 265-273, 570-579`)

iii. The AI investigated `reward_zone` at length and concluded "rzone seems to indicate the lick count within the reward zone... non-zero values occur at positions within the reward zone" (steps 23-27). It saw that the zone could be inferred from those positions but rejected that route because "omission trials where the mouse doesn't lick in the zone at all - there rzone stays zero everywhere, so I'll need a different signal to figure out which zone is active on those trials" (step 27), and settled on "the sessions_dict scene names as the reliable source" with a switch at trial 30, after empirically confirming the trial-30 switch in m11 ses-08 (step 27).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is first clipped to `[0, 450]`. Then the signed distance to the nearest edge of the active zone is computed: `position - zone_start` before the zone (negative), exactly 0 while inside `[zone_start, zone_end]`, and `position - zone_end` past the zone (positive). The continuous distance is then discretized (7-c) and stored as row 0 of the output array, time-varying at the native bin size.

ii.
```python
def compute_distance_to_reward_zone(position, zone_start, zone_end):
    dist = np.zeros_like(position, dtype=float)
    before = position < zone_start
    inside = (position >= zone_start) & (position <= zone_end)
    after = position > zone_end
    dist[before] = position[before] - zone_start
    dist[inside] = 0.0
    dist[after] = position[after] - zone_end
    return dist
...
dist = compute_distance_to_reward_zone(pos_trial, zone_start, zone_end)
dist_disc = discretize_distance(dist)
```
(`/app/convert_data.py:284-296, 593-594`)

iii. "I'm defining the reward zone boundaries for each maze (A: 80-130 cm, B: 200-250 cm, C: 320-370 cm) so I can compute the minimum distance from the animal's position to the nearest edge of the current zone at each timepoint. This gives negative values as the animal approaches, zero while inside the zone, and positive values once past it" (step 54). The AI sanity-checked the resulting class balance afterwards: "the distance_to_reward_zone output has '0cm (in zone)' at 0.237 fraction, which seems high... This actually makes sense since animals slow down and lick in the reward zone" (step 86).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit boolean masks implementing the 7 bins from the instructions: `< -50 -> 0`, `[-50,-10) -> 1`, `[-10,0) -> 2`, `== 0 -> 3`, `(0,10] -> 4`, `(10,50] -> 5`, `> 50 -> 6`. (Boundary convention at exactly +10 and +50 differs from the reference's `np.digitize`, which places them in the next-higher bin; this affects a measure-zero set of samples — the resulting class fractions 0.253/0.102/0.074/0.237/0.021/0.072/0.242 are identical to the reference's to three decimals.)

ii.
```python
def discretize_distance(dist):
    out = np.zeros(len(dist), dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
```
(`/app/convert_data.py:299-309`)

iii. The bin edges are copied from the Decoder Task specification; the "in zone" class is defined as exactly-zero distance, which follows from setting distance to 0 everywhere inside the 50 cm zone (step 54).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same indexing as the neural data: `position[t_start:t_end]` is the same slice as `deconv_filtered[t_start:t_end]`, after both streams were cropped to a common frame count. No shift or resampling.

ii.
```python
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
...
pos_trial = position[t_start:t_end]
```
(`/app/convert_data.py:563-570`)

iii. Behavior in these NWB files is stored frame-synchronously with the imaging data (step 54), so a shared index range guarantees alignment; the AI additionally truncated both streams to `min(n_frames_neural, n_frames_behav)` to keep them index-comparable.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Directly from `processing/behavior/BehavioralTimeSeries/position/data` (cm along the 450 cm VR corridor), sliced to the trial.

ii.
```python
position = beh['position/data'][:]
...
position = position[:n_frames]
...
pos_trial = position[t_start:t_end]
```
(`/app/convert_data.py:476, 501, 570`)

iii. The AI identified `position` as the VR track position and noted that the ITI is coded as -500, which is excluded automatically because trials end at the teleport pulse (step 18).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The only processing is clipping to the physical track range, `np.clip(pos_trial, 0, 450)`, before discretization into the 5 bins. (The reference does not clip but uses open end bins, so both map the handful of slightly out-of-range samples to bins 0 and 4.) Position is stored time-varying as row 1 of the output array.

ii.
```python
pos_trial = position[t_start:t_end]
...
# Clamp position to [0, 450]
pos_trial = np.clip(pos_trial, 0, 450)
...
pos_disc = discretize_position(pos_trial)
```
(`/app/convert_data.py:570-597`)

iii. No explicit discussion in the trajectory; the clip is a defensive step consistent with the AI's statement that the track is 450 cm long (metadata `track_length_cm: 450`) and prevents out-of-range samples from distorting the distance and position bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins over the 450 cm track: `<90 -> 0`, `[90,180) -> 1`, `[180,270) -> 2`, `[270,360) -> 3`, `>=360 -> 4`. Class fractions (0.211/0.178/0.231/0.227/0.154) match the reference exactly.

ii.
```python
def discretize_position(position):
    """Discretize position into 5 equal bins spanning 450 cm track."""
    out = np.zeros(len(position), dtype=np.int64)
    out[position < 90] = 0
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position < 360)] = 3
    out[position >= 360] = 4
    return out
```
(`/app/convert_data.py:312-320`)

iii. Directly from the Decoder Task specification ("5 equal-sized bins spanning the 450 cm track"), combined with the paper's 450 cm track length recorded in the metadata.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identically to 7-d: same trial slice indices as the neural matrix, after both streams were truncated to a common length. No additional alignment.

ii.
```python
n_frames = min(n_frames_neural, n_frames_behav)
...
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
pos_trial = position[t_start:t_end]
```
(`/app/convert_data.py:496-570`)

iii. Same rationale as 7-d: the behavioral series are stored at the imaging frame times.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. From `processing/behavior/BehavioralTimeSeries/lick/data`, sliced to the trial.

ii.
```python
lick = beh['lick/data'][:]
...
lick = lick[:n_frames]
...
lick_trial = lick[t_start:t_end]
```
(`/app/convert_data.py:478, 503, 572`)

iii. The AI flagged during exploration that it "still need[ed] to verify whether the lick channel is already binary or cumulative before treating it as a simple 0/1 signal" (step 54), and then binarized it rather than assuming it was already 0/1.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarization only: any value > 0 becomes 1, everything else 0 (the raw channel is a per-frame lick count that can exceed 1). Stored time-varying as row 3 of the output array; the resulting fractions (0.770 no-lick / 0.230 lick) match the reference exactly.

ii.
```python
lick_binary = (lick_trial > 0).astype(np.int64)
```
(`/app/convert_data.py:603`)

iii. The Decoder Task specifies a binary lick output ("0 = no, 1 = yes"), so counts are thresholded at > 0 (script header: "Lick: Binarized from the NWB lick signal (any lick > 0 -> 1)").

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial slice as the neural data; no shift, no resampling, no smoothing — lick is already a per-imaging-frame signal.

ii.
```python
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
...
lick_trial = lick[t_start:t_end]
```
(`/app/convert_data.py:563-572`)

iii. As with the other behavioral channels, the AI relied on the NWB behavior series being sampled at the imaging frame times (step 54).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Not from a raw NWB variable: from the hardcoded `SESSIONS_DICT` scene name for the session (parsed to a zone letter before/after trial 30), mapped `A -> 0, B -> 1, C -> 2`. See 7-a; I verified this labelling reproduces the zone implied by the NWB `reward_zone`/`position` data on every trial of all 152 sessions, and the class fractions (0.329/0.337/0.335) match the reference.

ii.
```python
zone_letter = get_reward_zone_for_trial(scene, t)
zone_start, zone_end = REWARD_ZONES[zone_letter]
zone_idx = {'A': 0, 'B': 1, 'C': 2}[zone_letter]
...
output = np.stack([
    dist_disc, pos_disc, speed_disc, lick_binary,
    np.full(n_tp, zone_idx, dtype=np.int64),
    np.full(n_tp, rewarded, dtype=np.int64),
], axis=0)
```
(`/app/convert_data.py:578-619`)

iii. See 7-a: the AI chose the scene schedule over inference from `reward_zone` specifically because `reward_zone` is silent on omission trials where the mouse never licks in the zone (step 27), and it validated the schedule against observed lick positions for several mice/sessions before committing (steps 25-27).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Parse the scene string into (zone_before, zone_after); for trials >= 30 on switch days use zone_after, otherwise zone_before; map the letter to 0/1/2 and broadcast the constant across the trial's timepoints (so the per-trial variable is stored as a time-varying row, as the instructions prefer).

ii.
```python
def parse_scene(scene):
    if '_to_Env' in scene:            # e.g. 'Env1_C_to_Env2_B'
        ...
    if '_to_' in scene:               # e.g. 'Env1_LocationA_to_B'
        parts = scene.split('_to_')
        before = parts[0]
        zone_after = parts[1]
        env = int(before[3])
        zone_before = before.split('Location')[1]
        return env, zone_before, env, zone_after
    env = int(scene[3])
    zone = scene.split('Location')[1]
    return env, zone, None, None
...
np.full(n_tp, zone_idx, dtype=np.int64),
```
(`/app/convert_data.py:232-262, 617`)

iii. "On switch days (3, 5, 7, 8, 10, 12, 14), the first 30 trials use one zone and trials 30+ use the second zone... non-switch days keep a single zone throughout" (step 48), confirmed empirically on m11 ses-08 where "trial 0-29 around 200-210 (Zone B), trial 30+ around 320-332 (Zone C)" (step 27).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From the `Reward` behavioral time series' own event timestamps (`Reward/timestamps`), compared against the behavioral `position/timestamps` of the trial's first and last frames.

ii.
```python
reward_ts = beh['Reward/timestamps'][:]
timestamps = beh['position/timestamps'][:]
...
t_start_time = timestamps[t_start]
t_end_time = timestamps[t_end - 1]
rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
```
(`/app/convert_data.py:479-485, 587-589`)

iii. The AI established that reward delivery is an event series with its own timestamps and that `reward_zone` is not a reward indicator: "what I really need is... whether reward was delivered on each trial, which I can get from the Reward timestamps" (step 25).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is labelled 1 if at least one reward timestamp falls in the closed interval `[t_first_frame, t_last_frame]` of that trial, else 0; the scalar is broadcast across the trial's timepoints as row 5 of the output array. Unlike the reference, no assertion is made that reward times land within half a time bin of a frame; the interval test uses the raw timestamps directly. Resulting fractions (0.157 no-reward / 0.843 reward) match the reference exactly and are consistent with the paper's ~15% omission rate.

ii.
```python
rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
...
np.full(n_tp, rewarded, dtype=np.int64),
```
(`/app/convert_data.py:589, 618`)

iii. Script header: "Reward outcome: Determined by mapping Reward event timestamps to trial time windows". Windowing by timestamps avoids having to snap reward events onto imaging frames.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Four defensive behaviours, all silent or warning-level:
- **Neural/behavior length mismatch** (10 of 152 sessions differ by one frame): both streams are truncated to `min(n_frames_neural, n_frames_behav)` and trial boundaries are recomputed *after* the truncation.
- **Unequal numbers of trial_start and teleport pulses**: the two index arrays are truncated to the shorter length and paired positionally (no assertion). Degenerate pairs with `t_end <= t_start` are skipped.
- **Degenerate trials/sessions**: trials with < 2 timepoints are skipped; sessions with 0 neurons after filtering or < 2 usable trials are dropped entirely.
- **Out-of-range position**: clipped to `[0, 450]`. Negative speeds are folded to their magnitude with `np.abs` before binning; NaNs in dF/F (outside trials, or degenerate baselines) are excluded by the `nanmask` in the interneuron test, and a NaN correlation leaves the cell classified as non-interneuron.
No assertions cross-check behavioral timestamps against the imaging rate, which is why the two-plane rate discrepancy (3-b) went unnoticed.

ii.
```python
n_frames = min(n_frames_neural, n_frames_behav)
deconvolved = deconvolved[:n_frames]
...
# Re-compute trial boundaries after truncation
trial_start_idx = np.where(trial_start_signal > 0)[0]
teleport_idx = np.where(teleport_signal > 0)[0]
n_trials = min(len(trial_start_idx), len(teleport_idx))
```
(`/app/convert_data.py:494-515`)
```python
if t_end <= t_start:
    continue
...
if n_tp < 2:
    continue
...
pos_trial = np.clip(pos_trial, 0, 450)
speed_disc = discretize_speed(np.abs(speed_trial))
```
(`/app/convert_data.py:558-600`)
```python
r = np.corrcoef(dff[c, nanmask], speed[nanmask])[0, 1]
if np.isnan(r):
    continue
```
(`/app/convert_data.py:434-436`)

iii. These are defensive measures added while iterating on the script (the AI edited the file several times after test runs, steps 60-73). No explicit reasoning about them appears in the trajectory beyond ensuring "neural and behavioral data have same number of frames" (code comment).

## 13-a. What are the most time-consuming steps of the code?

i. In order:
1. **Reading the NWB arrays** — for each session the full `Deconvolved`, `Fluorescence` and `Neuropil` matrices for *all* ROIs are read into memory (tens of thousands of frames x up to ~3000 ROIs, three copies).
2. **`compute_dff_for_interneuron_detection`** — per trial (~80) and per cell (up to ~2300) it calls `ndi.gaussian_filter1d` in a Python loop for the baseline, and again for the dF/F smoothing: on the order of 10^5-10^6 Python-level filter calls per session.
3. **`detect_interneurons`** — a Python loop computing `np.corrcoef` separately per cell, each call forming a 2x2 covariance over the full session.
4. **Pickling the result** — the output file is ~9.9 GB, written in a single `pickle.dump`.
The whole conversion nevertheless completed in roughly 3-5 minutes of wall clock in the agent's run (steps 78-79).

ii.
```python
for start, stop in zip(trial_starts, trial_ends):
    ...
    bl = np.copy(f_[:, start:stop])
    for c in range(n_cells):
        bl[c] = ndi.gaussian_filter1d(bl[c], sigma=15)
...
for start, stop in zip(trial_starts, trial_ends):
    for c in range(n_cells):
        dff[c, start:stop] = ndi.gaussian_filter1d(dff[c, start:stop], sigma=2)
```
(`/app/convert_data.py:380-408`)
```python
with open(output_path, 'wb') as f:
    pickle.dump(data, f)
```
(`/app/convert_data.py:774-775`)

iii. Not discussed in the trajectory; the AI ran the conversion as a single background job and only monitored completion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
- The two `for c in range(n_cells)` loops inside `compute_dff_for_interneuron_detection` are unnecessary: `ndi.gaussian_filter1d` accepts an `axis` argument and could filter the whole `(n_cells, T)` block at once (as the code already does for `minimum_filter1d`/`maximum_filter1d`).
- The per-cell `np.corrcoef` loop in `detect_interneurons` could be a single vectorized correlation of the centred dF/F matrix against the centred speed vector.
- The per-trial loop in `process_session` recomputes `discretize_*` and `compute_distance_to_reward_zone` trial-by-trial; position/speed/lick discretization could be done once per session and then sliced (the reward-zone-relative distance would need per-block zone boundaries, which is still vectorizable as two blocks).

ii.
```python
for c in range(n_cells):
    bl[c] = ndi.gaussian_filter1d(bl[c], sigma=15)     # -> ndi.gaussian_filter1d(bl, sigma=15, axis=-1)
...
for c in range(n_cells):
    r = np.corrcoef(dff[c, nanmask], speed[nanmask])[0, 1]
```
(`/app/convert_data.py:391-392, 432-434`)

iii. Not discussed in the trajectory.

## 13-c. What processing does the code repeat multiple times?

i. Little: each NWB file is opened exactly once and all needed arrays are read in that single pass (there is no separate survey pass). What is repeated is (a) `parse_scene(scene)` is called once per trial via both `get_reward_zone_for_trial` and `get_env_for_trial`, i.e. ~2x per trial instead of once per session; (b) the per-trial `REWARD_ZONES` lookup and zone-letter mapping; (c) `f_`/`fneu_` are copied and re-scanned several times inside the dF/F routine (nan-fill, neuropil subtraction, baseline, dF/F, smoothing). All are trivial relative to I/O.

ii.
```python
zone_letter = get_reward_zone_for_trial(scene, t)   # calls parse_scene
...
env = get_env_for_trial(scene, t)                   # calls parse_scene again
```
(`/app/convert_data.py:578-583`)

iii. Not discussed in the trajectory; the single-pass structure follows from the AI's decision to take the reward zone from a hardcoded schedule rather than inferring it from the data, which removed the need for a first survey pass over all files.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
- **The entire dF/F pipeline** (`compute_dff_for_interneuron_detection`, the single most expensive computation in the script) is used only to test speed correlation; the dF/F traces themselves are never saved and never used as neural data. Under the reference's approach the same computation would have produced the saved signal.
- **`Fluorescence` and `Neuropil` are read for every ROI** (including non-cells) and then immediately subset to `iscell`; reading them lazily per column, or subsetting before the copy, would avoid two full-session copies.
- **`env_signal`** is loaded, truncated and never used (environment is taken from the scene name).
- **`timestamps`** is fully loaded but used only for the two scalars per trial that window reward events.
- The deconvolved matrix for all ROIs is loaded and then subset to the retained cells; the discarded columns are pure overhead.
- `frame_rate` is passed into `compute_dff_for_interneuron_detection` and never used inside it (the 300-sample window is hardcoded).

ii.
```python
env_signal = beh['environment/data'][:]
...
env_signal = env_signal[:n_frames]      # never used again
```
(`/app/convert_data.py:482, 507`)
```python
dff = compute_dff_for_interneuron_detection(F_cells, Fneu_cells, trial_start_idx, teleport_idx, frame_rate)
is_int = detect_interneurons(dff, speed, threshold=0.5)     # dff discarded afterwards
```
(`/app/convert_data.py:522-526`)
```python
def compute_dff_for_interneuron_detection(F, Fneu, trial_starts, trial_ends, frame_rate=15.5):
    ...
    window = 300  # ~20 seconds at 15.5 Hz     (frame_rate argument unused)
```
(`/app/convert_data.py:334-378`)

iii. Not discussed in the trajectory. The discarded dF/F is a direct consequence of the decision documented in 2-a/2-b: having recomputed the paper's dF/F, the AI still used the NWB's precomputed `Deconvolved` array as the neural signal.
