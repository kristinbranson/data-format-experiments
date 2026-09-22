# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six subject IDs, then iterates through each subject's session subdirectories in sorted order. For each session it loads calcium fluorescence from `suite2p/plane0/F.npy` and `Fneu.npy`, and behavioral motion-energy data from `move_deve/motion_energy_glob.npy` plus `tstamps.npy` and `interframe_int.npy`. Trials are not loaded directly from disk; they are created later by splitting each processed session into 60-second chunks.

ii. 
```python
DATA_DIR = '/app/data'
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_session_dirs(subject):
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir) 
                       if os.path.isdir(os.path.join(subj_dir, d))])
    return [os.path.join(subj_dir, s) for s in sessions]
...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. In `CONVERSION_NOTES.md`, the AI documented that the dataset contains exactly 6 mice and described each session as containing `suite2p/plane0` and `move_deve` subdirectories. It justified including all known mice and all their sessions based on that fixed directory structure rather than dynamically discovering `jm*` folders.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by mouse directory, but the set of mice is encoded explicitly in the constant `SUBJECTS` instead of being discovered from the filesystem. Session-to-subject assignment is preserved through `subj_i` and stored in `subject_idx`.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
for subj_i, subject in enumerate(SUBJECTS):
    sess_dirs = get_session_dirs(subject)
...
all_subject_idx.append(subj_i)
...
'subjects': SUBJECTS,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. The notes say the dataset has "6 mice (jm031-jm046)" and repeatedly refer to those six subjects as the complete set. No separate justification was given for hard-coding them beyond the AI's belief that the dataset contents were fixed and fully known.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory inside a subject folder. The AI sorts those subdirectory names and processes each one as a separate session.

ii.
```python
def get_session_dirs(subject):
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir) 
                       if os.path.isdir(os.path.join(subj_dir, d))])
    return [os.path.join(subj_dir, s) for s in sessions]
...
for sess_dir in sess_dirs:
    session_name = os.path.basename(sess_dir)
    session_label = f'{subject}/{session_name}'
```

iii. The notes describe each daily recording directory as one session and emphasize deterministic sorted ordering. The AI's justification is the observed directory layout: one recording day per subdirectory.

## 1-d. How are the data split into trials?

i. Trials are artificial, non-overlapping 60-second segments carved out of each continuous session after 10-frame binning. With 30 Hz imaging and 10-frame averaging, each trial has 180 binned time points. Only full trials are kept.

ii.
```python
BINNED_FS = FS / BIN_SIZE  # 3 Hz
TRIAL_BINS = int(TRIAL_DURATION_SEC * BINNED_FS)  # 180 bins per trial
...
n_total_bins = dff_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS
...
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    input_trials.append(time_seconds[start:end].reshape(1, -1).astype(np.float32))
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The trajectory and notes both state that the dataset has no natural trial structure, so the AI followed the task instruction to "Split sessions into 60-second trials." That is the explicit justification for defining trials this way.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level quality-control filter. The only effective filtering is that any trailing partial trial is dropped because `n_trials` is floor-divided and only full 60-second blocks are appended.

ii.
```python
n_total_bins = dff_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS
...
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    ...
```

iii. `CONVERSION_NOTES.md` says "No explicit trial curation (spontaneous behavior, no task trials)." The AI therefore treated all full-length blocks as valid and did not apply any further trial rejection.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from the Suite2p fluorescence arrays `F.npy` and `Fneu.npy` in `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu)
```

iii. The notes explicitly map `F.npy, Fneu.npy -> neural` and describe them as raw fluorescence and neuropil fluorescence traces. That is the AI's stated rationale.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil subtraction with coefficient 0.7, applies a maximin-style baseline estimate with Gaussian smoothing followed by min and max filters, subtracts that baseline, and then averages the resulting traces into 10-frame bins. It calls the result `dff`, but the actual computation is baseline subtraction (`Fc - Flow`), not division by baseline.

ii.
```python
def compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF, 
                sig_baseline=SIG_BASELINE, win_baseline=WIN_BASELINE):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)
    dff = Fc - Flow
    return dff.astype(np.float32)
...
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
```

iii. The notes justify this as matching the paper's "default Suite2p parameters" and the Track2p code path `Fc - Flow`. They also record that the AI initially divided by baseline, found that this caused extreme values, and changed the implementation to subtraction-only to match the reference behavior.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no additional neuron filtering during conversion. All rows of `F.npy` are kept.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
n_neurons, n_frames = F.shape
...
all_brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The notes explicitly state that all `iscell` values are 1 and that the data are already pre-filtered by Track2p to cells tracked across days. That is the AI's justification for not filtering further.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural data are aligned to session start rather than to a stimulus or behavioral event. Each trial is simply the next contiguous 60-second window from the beginning of the session.

ii.
```python
time_seconds = (np.arange(n_total_bins) * BIN_SIZE / FS).astype(np.float32)
...
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
...
'metadata': {
    'temporal_alignment_event': 'session_start',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_SEC),
```

iii. The notes say the dataset is spontaneous behavior with no natural task trials, so the AI chose session start as the only available alignment anchor. That matches the reasoning recorded in the trajectory.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping averages, so 30 Hz becomes 3 Hz and each time bin is about 333.33 ms. Yes, temporal rebinning is applied to both neural data and motion energy.

ii.
```python
BIN_SIZE = 10
BINNED_FS = FS / BIN_SIZE  # 3 Hz
TIME_BIN_MS = (BIN_SIZE / FS) * 1000  # ~333.33 ms
...
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
me_binned = bin_data(me_full.reshape(1, -1), BIN_SIZE, axis=1).squeeze()
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes cite the paper's statement that decoding analyses averaged "10 consecutive timestamps" and use that as the explicit justification for this rebinning step.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not taken from any stored timestamp variable. The AI derives it from the bin index, frame rate, and bin size.

ii.
```python
time_seconds = (np.arange(n_total_bins) * BIN_SIZE / FS).astype(np.float32)
...
input_trials.append(time_seconds[start:end].reshape(1, -1).astype(np.float32))
```

iii. The notes say `input[0]` is "Time in seconds from session start, at binned resolution." The implied justification is that imaging runs at a fixed 30 Hz, so elapsed time can be reconstructed directly from sample index.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one time value per binned sample as the left-edge time of that 10-frame bin, then reshapes the session-long vector into per-trial `(1, time)` arrays. No smoothing or interpolation is applied.

ii.
```python
time_seconds = (np.arange(n_total_bins) * BIN_SIZE / FS).astype(np.float32)
...
input_trials.append(time_seconds[start:end].reshape(1, -1).astype(np.float32))
```

iii. No extended justification was documented beyond the Step 5 mapping note that time should be "from session start, at binned resolution." The processing is straightforwardly implied by that choice.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it is generated at the same binned resolution as the neural data and sliced with the same `[start:end]` indices for each trial.

ii.
```python
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
...
time_seconds = (np.arange(n_total_bins) * BIN_SIZE / FS).astype(np.float32)
...
neural_trials.append(dff_binned[:, start:end].astype(np.float32))
input_trials.append(time_seconds[start:end].reshape(1, -1).astype(np.float32))
```

iii. The AI's justification is implicit in the implementation and notes: both streams are built from the same session timeline after identical 10-frame binning, so matching indices keep them aligned.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived primarily from `move_deve/motion_energy_glob.npy`. The AI also loads `interframe_int.npy` to infer missing camera frames and loads `tstamps.npy`, although `tstamps.npy` is not actually used downstream.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
...
me_full = interpolate_missing_frames(me, n_frames, tstamps, ifi)
```

iii. The notes map `motion_energy_glob.npy -> output[0]` and explain that missing camera frames should be handled by interpolation using interframe timing gaps. The trajectory also mentions `tstamps.npy` and `interframe_int.npy` as possible sources for locating missing frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI first fills missing camera frames by inferring how many neural frames each camera interval spans from `ifi / median_ifi` and then linearly interpolating a full-length motion-energy vector with `np.interp`. It then averages the signal into 10-frame bins and discretizes the binned values session-by-session into five percentile bins.

ii.
```python
def interpolate_missing_frames(me, n_neural_frames, tstamps, ifi):
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    median_ifi = np.median(ifi)
    neural_indices = np.zeros(len(me), dtype=int)
    neural_idx = 0
    neural_indices[0] = 0
    for i in range(len(ifi)):
        n_skip = max(1, round(ifi[i] / median_ifi))
        neural_idx += n_skip
        if i + 1 < len(me):
            neural_indices[i + 1] = neural_idx
    me_full = np.interp(np.arange(n_neural_frames), neural_indices, me.astype(np.float64))
    return me_full
...
me_binned = bin_data(me_full.reshape(1, -1), BIN_SIZE, axis=1).squeeze()
me_discrete, bin_edges = discretize_motion_energy(me_binned, N_BINS_OUTPUT)
```

iii. The notes justify this as: missing frames "can be interpolated," both streams should be binned by 10 for decoding, and motion energy must be discretized into 5 equal-percentile bins per session. They do not justify the specific `np.interp` implementation beyond that general interpolation requirement.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After binning, motion energy is thresholded per session using the 0th, 20th, 40th, 60th, 80th, and 100th percentiles, then digitized into integer labels 0 through 4.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_BINS_OUTPUT):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(me_binned, percentiles)
    me_discrete = np.digitize(me_binned, bin_edges[1:-1], right=False)
    me_discrete = np.clip(me_discrete, 0, n_bins - 1)
    return me_discrete.astype(np.int64), bin_edges
```

iii. The notes explicitly list "5 equal-percentile bins per session" as a key decision and tie it to the decoder-task requirement that motion energy be discretized into five equal-percentile bins selected per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data by first expanding it to the neural frame count with inferred missing-frame interpolation, then applying the same 10-frame binning as the neural traces, and finally slicing trials with the same `[start:end]` indices.

ii.
```python
n_neurons, n_frames = F.shape
...
me_full = interpolate_missing_frames(me, n_frames, tstamps, ifi)
...
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
me_binned = bin_data(me_full.reshape(1, -1), BIN_SIZE, axis=1).squeeze()
...
output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes say camera frames are triggered by the microscope and missing frames should be interpolated so the behavioral trace matches neural length before binning. That is the AI's stated justification for the alignment procedure.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing behavioral frames by interpolating the motion-energy trace up to the neural frame count. It also silently drops any trailing bins that do not form a complete 60-second trial. It does not add a post-interpolation assertion that the alignment was reconstructed correctly.

ii.
```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)
...
me_full = np.interp(
    np.arange(n_neural_frames),
    neural_indices,
    me.astype(np.float64)
)
...
n_trials = n_total_bins // TRIAL_BINS
for t in range(n_trials):
    ...
```

iii. The notes explicitly justify interpolating missing camera frames and describe incomplete trailing segments as acceptable to drop because trials are fixed-length artificial windows. No explicit justification was given for omitting a hard assertion after interpolation.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies the dF/F computation as the dominant cost, with file loading and binning much cheaper. That matches the code's timing instrumentation, which measures load time, dF/F time, binning time, and total per session.

ii.
```python
t1 = time.time()
dff = compute_dff(F, Fneu)
t_dff = time.time() - t1
...
print(f'  {session_label}: {n_neurons} neurons, {n_frames} frames, '
      f'{n_total_bins} bins, {n_trials} trials, '
      f'ME missing={n_frames - len(me)} frames, '
      f'time: load={t_load:.2f}s dff={t_dff:.2f}s bin={t_bin:.2f}s total={t_total:.2f}s')
```

iii. `CONVERSION_NOTES.md` Step 7 reports dF/F at roughly `0.25-0.95s` per session and labels it as the main per-session cost. That is the AI's explicit justification.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the Python loop over `ifi` in `interpolate_missing_frames`, which incrementally builds `neural_indices`. The per-trial append loop is another smaller example. The AI did not call these out explicitly in its notes; this follows directly from the implementation it wrote.

ii.
```python
for i in range(len(ifi)):
    n_skip = max(1, round(ifi[i] / median_ifi))
    neural_idx += n_skip
    if i + 1 < len(me):
        neural_indices[i + 1] = neural_idx
...
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
```

iii. No explicit optimization rationale was documented. The code and notes focus on correctness and validation rather than vectorizing these loops.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats small per-trial conversions and reshapes even though the session-level arrays are already binned: every trial slice is re-cast with `astype(...)` and reshaped before being appended. It also repeats the same session-processing pipeline independently for every session, which is expected but not optimized away.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    input_trials.append(time_seconds[start:end].reshape(1, -1).astype(np.float32))
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
...
for subj_i, subject in enumerate(SUBJECTS):
    ...
    for sess_dir in sess_dirs:
        ...
        neural_trials, input_trials, output_trials, n_neurons = process_session(...)
```

iii. There is no explicit justification in the notes for these repeated conversions. They appear to be incidental implementation choices rather than decisions the AI discussed.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is loading `tstamps.npy` and passing it through the pipeline even though it is never used. The code also computes timing variables such as `t_interp` and `t_disc` without reporting them, and optional plotting generates inspection figures that are not used by downstream analyses.

ii.
```python
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
...
me_full = interpolate_missing_frames(me, n_frames, tstamps, ifi)
...
t2 = time.time()
me_full = interpolate_missing_frames(me, n_frames, tstamps, ifi)
t_interp = time.time() - t2
...
t4 = time.time()
me_discrete, bin_edges = discretize_motion_energy(me_binned, N_BINS_OUTPUT)
t_disc = time.time() - t4
...
if show_processing:
    plot_processing(...)
```

iii. No explicit justification was documented for these extra steps. They appear to be leftovers from exploratory validation and diagnostics rather than parts of the final data product.
