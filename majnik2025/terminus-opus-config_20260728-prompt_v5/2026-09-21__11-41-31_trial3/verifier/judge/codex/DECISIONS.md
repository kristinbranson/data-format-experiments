# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans `/app/data` for subject directories whose names start with `jm`, scans each subject directory for session subdirectories, then iterates through every `(subject, session)` pair. For each session it loads calcium traces from `suite2p/plane0/F.npy` and `Fneu.npy`, Suite2p parameters from `ops.npy`, and behavior from `move_deve/motion_energy_glob.npy` plus `tstamps.npy`. Trials are not loaded directly from disk; they are created later by splitting each processed session into fixed 60-second chunks.

ii.
```python
def get_subjects_and_sessions(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
    all_sessions = {}
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        all_sessions[subj] = sessions
    return subjects, all_sessions

def load_session_data(data_dir, subject, session):
    sess_dir = os.path.join(data_dir, subject, session)
    s2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(sess_dir, 'move_deve')

    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))

    return F, Fneu, ops, me, tstamps
```

iii. In `CONVERSION_NOTES.md`, the agent says the dataset is already organized as subject folders containing session folders and that the relevant per-session files are the Suite2p outputs plus motion-energy files. In Step 5 of the notes it explicitly maps `F.npy`, `Fneu.npy`, frame index, and `motion_energy_glob.npy` into the target fields.

## 1-b. How are the data split into subjects?

i. Subjects are identified entirely from directory names: every top-level data directory beginning with `jm` is treated as one mouse, and the list is sorted lexicographically.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

iii. The notes say there are 6 subjects, `jm031` through `jm046`, and treat each `jm*` directory as one mouse. The trajectory also repeatedly summarizes the dataset this way.

## 1-c. How are the data split into sessions?

i. Each session is defined as one subdirectory inside a subject directory. The session names are sorted lexicographically. In full mode the script processes every session; in sample mode it reduces the session list to two sessions from different subjects.

ii.
```python
for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    all_sessions[subj] = sessions

session_list = []
for subj in subjects:
    for sess in all_sessions[subj]:
        session_list.append((subj, sess))
```

iii. `CONVERSION_NOTES.md` Step 2 lists the number of sessions per subject and treats each dated subdirectory as one daily recording session. The agent's trajectory also describes the dataset as “41 total sessions.”

## 1-d. How are the data split into trials?

i. The agent treats the recordings as continuous sessions with no native trials. After binning to 3 Hz, it defines trials as non-overlapping 60-second segments, so each trial contains `60 * 3 = 180` time bins. Any leftover tail shorter than one full trial is dropped.

ii.
```python
TRIAL_DURATION_SEC = 60
BINNED_RATE = FRAME_RATE / BIN_SIZE
TIMEPOINTS_PER_TRIAL = int(TRIAL_DURATION_SEC * BINNED_RATE)  # 180

def split_into_trials(data_2d, timepoints_per_trial):
    n_timepoints = data_2d.shape[-1]
    n_trials = n_timepoints // timepoints_per_trial

    trials = []
    for t in range(n_trials):
        start = t * timepoints_per_trial
        end = start + timepoints_per_trial
        if data_2d.ndim == 2:
            trials.append(data_2d[:, start:end].copy())
        else:
            trials.append(data_2d[start:end].copy())
    return trials
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent explicitly states that the task specification requires splitting each session into 60-second trials, and it records the expected 180 binned timepoints per trial as a sanity check.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter beyond dropping incomplete final trial fragments that do not fill a full 60-second window. The script also skips whole sessions if they end up with fewer than two trials, because the decoder requires at least two trials per session.

ii.
```python
def split_into_trials(data_2d, timepoints_per_trial):
    n_timepoints = data_2d.shape[-1]
    n_trials = n_timepoints // timepoints_per_trial
    ...
    return trials

if len(neural_trials) < 2:
    print(f"  WARNING: Session {subj}/{sess} has only {len(neural_trials)} trials, skipping")
    continue
```

iii. The notes say there is no natural trial structure in this dataset and record the decision to drop incomplete last trials. The session-level minimum-trial check appears to be motivated by the task requirement that each session contain at least two trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` arrays are derived from Suite2p fluorescence traces `F.npy` and `Fneu.npy`. The script also loads `ops.npy` to obtain Suite2p preprocessing parameters used during baseline correction.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent maps `F.npy` and `Fneu.npy` to the target `neural` field and describes the transform as `preprocess(F - 0.7*Fneu)`.

## 2-b. How is the `neural` data processed?

i. The agent subtracts neuropil (`F - 0.7 * Fneu`), applies Suite2p's `preprocess` function with parameters taken from `ops.npy` (falling back to the Suite2p defaults in the notes), bins the resulting traces by averaging non-overlapping groups of 10 frames, and finally casts each trial to `float32`.

ii.
```python
Fc = F.copy() - NEUCOEFF * Fneu

dff = preprocess(
    F=Fc.copy(),
    baseline=ops.get('baseline', 'maximin'),
    win_baseline=ops.get('win_baseline', 60.0),
    sig_baseline=ops.get('sig_baseline', 10.0),
    fs=ops.get('fs', 30.0),
    prctile_baseline=ops.get('prctile_baseline', 8.0),
    device=device
)

dff_binned = bin_data(dff, BIN_SIZE, axis=1)
...
neural_trials = [n.astype(np.float32) for n in neural_trials]
```

iii. The notes and trajectory repeatedly justify this with the paper's statement that later analyses use “baseline corrected fluorescence traces as our dF/F” and “averaging in bins of 10 consecutive timestamps.” The notes also record the Suite2p default baseline parameters from `ops.npy`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code applies no additional neuron-level filtering at conversion time. It does not read or use `iscell.npy`, `stat.npy`, or any quality threshold; every row of `F.npy` is carried into the output.

ii.
```python
def load_session_data(data_dir, subject, session):
    ...
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    ...
    return F, Fneu, ops, me, tstamps
```

iii. The justification in `CONVERSION_NOTES.md` is that the provided data already contains Track2p-tracked cells and “all cells in the data are tracked cells (iscell all = 1.0).” The trajectory similarly states that no extra filtering is needed because the files already contain tracked cells only.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to session start rather than to a stimulus or behavior event. Trials are simply consecutive 60-second windows from the beginning of each recording session.

ii.
```python
time_in_trial = trial_start_sec + np.arange(TIMEPOINTS_PER_TRIAL) / BINNED_RATE
...
'metadata': {
    'temporal_alignment_event': 'session_start',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_SEC),
    ...
}
```

iii. Step 5 of the notes says the input should be “time elapsed from beginning of session in seconds,” and the trajectory explicitly describes the trial structure as continuous segmentation of each session rather than event-triggered trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10-frame bins at a 30 Hz frame rate, so the temporal resolution is 1/3 second per bin, i.e. about 333.33 ms. Both neural and behavioral data are rebinned by averaging within each non-overlapping 10-frame chunk.

ii.
```python
BIN_SIZE = 10
FRAME_RATE = 30
BINNED_RATE = FRAME_RATE / BIN_SIZE  # 3 Hz

dff_binned = bin_data(dff, BIN_SIZE, axis=1)
me_binned = bin_data(me_aligned.reshape(1, -1), BIN_SIZE, axis=1).flatten()
...
'time_bin_size': 1000.0 / BINNED_RATE,
```

iii. The notes explicitly justify 10-frame averaging using the methods text and record the derived 3 Hz effective rate and 333.33 ms bin size.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from any raw timestamp variable. It is synthesized from the trial index, within-trial bin index, the 60-second trial duration, and the fixed binned sampling rate.

ii.
```python
for t in range(n_trials):
    trial_start_sec = t * TRIAL_DURATION_SEC
    time_in_trial = trial_start_sec + np.arange(TIMEPOINTS_PER_TRIAL) / BINNED_RATE
    input_trials.append(time_in_trial.astype(np.float32).reshape(1, -1))
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent maps “frame index” to the decoder input and describes the transform as `time_seconds = frame_idx / 3.0 + trial_start`.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial the script computes the left-edge time of every binned sample in seconds: it takes the trial start time in seconds and adds `0, 1/3, 2/3, ...` up to the end of the 60-second trial. The result is cast to `float32` and reshaped to `(1, n_timepoints)`.

ii.
```python
trial_start_sec = t * TRIAL_DURATION_SEC
time_in_trial = trial_start_sec + np.arange(TIMEPOINTS_PER_TRIAL) / BINNED_RATE
input_trials.append(time_in_trial.astype(np.float32).reshape(1, -1))
```

iii. The notes justify this as representing “time elapsed from beginning of session in seconds” after 10-frame binning, which gives one sample every 0.3333 seconds.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: the script generates exactly one time value for each binned neural timepoint in a trial, using the same 60-second trial boundaries and the same 3 Hz binned sampling rate as the neural arrays.

ii.
```python
neural_trials = split_into_trials(dff_binned, TIMEPOINTS_PER_TRIAL)
...
time_in_trial = trial_start_sec + np.arange(TIMEPOINTS_PER_TRIAL) / BINNED_RATE
input_trials.append(time_in_trial.astype(np.float32).reshape(1, -1))
```

iii. The trajectory and notes both frame time as an input matched to the binned neural stream after fixed-width segmentation of each session.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output motion-energy signal is derived from `move_deve/motion_energy_glob.npy`. When frame counts disagree with the neural recording, the script additionally uses `move_deve/tstamps.npy` to interpolate motion energy onto the neural frame grid.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
...
me_aligned = align_motion_energy(me, tstamps, n_neural_frames)
```

iii. The notes say `tstamps.npy` contains cumulative camera timestamps in kiloseconds and record the decision “interpolate motion energy to match neural frame count using tstamps.” The trajectory also says the agent resolved the timestamp units before writing the script.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent first aligns motion energy to the neural frame count, using linear interpolation over camera timestamps if camera frames are missing. It then averages motion energy in non-overlapping 10-frame bins, splits the binned trace into 60-second trials, concatenates all trial values within a session to compute five percentile boundaries, and digitizes each trial into session-specific discrete bins.

ii.
```python
me_aligned = align_motion_energy(me, tstamps, n_neural_frames)
...
me_binned = bin_data(me_aligned.reshape(1, -1), BIN_SIZE, axis=1).flatten()
...
me_trials = split_into_trials(me_binned, TIMEPOINTS_PER_TRIAL)
me_discrete_trials, bin_edges = discretize_motion_energy(me_trials, N_ME_BINS)
```

iii. The notes justify this with two claims: the methods say behavioral traces should be averaged in 10-frame bins, and the data README allows missing frames to be interpolated over. The trajectory also states that motion energy should be discretized into five equal-percentile bins per session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. For each session, the agent concatenates all binned motion-energy samples from all trials in that session, computes 0/20/40/60/80/100 percentiles, and uses `np.digitize` to assign every time bin to one of five categories labeled 0 through 4.

ii.
```python
all_me = np.concatenate(me_binned_trials)
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_me, percentiles)
...
bin_indices = np.digitize(me_trial, bin_edges[1:-1], right=False)
bin_indices = np.clip(bin_indices, 0, n_bins - 1)
discretized_trials.append(bin_indices.astype(np.int64))
```

iii. `CONVERSION_NOTES.md` Step 5 says the task specification requires “5 equal-percentile bins per session (quintiles).” The notes later check that the output distribution is approximately uniform.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The code assumes both streams should share the same frame grid. If the motion-energy array already has the same length as the neural recording, it is used directly. Otherwise, the script constructs a regular 30 Hz neural time axis, converts `tstamps.npy` from kiloseconds to seconds, and linearly interpolates motion energy onto the neural times before any binning or trial splitting.

ii.
```python
def align_motion_energy(me, tstamps, n_neural_frames):
    n_cam_frames = len(me)

    if n_cam_frames == n_neural_frames:
        return me.astype(np.float64)

    neural_times = np.arange(n_neural_frames) / FRAME_RATE
    cam_times = tstamps * 1000.0
    me_aligned = np.interp(neural_times, cam_times, me.astype(np.float64))
    return me_aligned
```

iii. The notes say `tstamps` are in kiloseconds and explicitly list “interpolate motion energy to match neural frame count using tstamps” as a key decision. The trajectory also shows the agent spent time determining timestamp units and then chose timestamp-based interpolation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main error-handling path is for missing camera frames: if the motion-energy trace is shorter than the neural trace, the code fills in the missing samples by interpolating over `tstamps.npy`. Incomplete final trial fragments are silently discarded by floor division in `split_into_trials`, and sessions with fewer than two resulting trials are skipped. There is no explicit assertion after interpolation that the aligned motion-energy array exactly matches the neural length, and there is no special handling for missing neural values.

ii.
```python
if n_cam_frames == n_neural_frames:
    return me.astype(np.float64)

neural_times = np.arange(n_neural_frames) / FRAME_RATE
cam_times = tstamps * 1000.0
me_aligned = np.interp(neural_times, cam_times, me.astype(np.float64))

def split_into_trials(data_2d, timepoints_per_trial):
    n_trials = n_timepoints // timepoints_per_trial
    ...

if len(neural_trials) < 2:
    print(f"  WARNING: Session {subj}/{sess} has only {len(neural_trials)} trials, skipping")
    continue
```

iii. The notes justify the interpolation choice by citing the README's statement that missing frame indices can be interpolated over. The trajectory also describes missing camera frames as an expected dataset quirk that needs to be corrected before alignment.

## 6-a. What are the most time-consuming steps of the code?

i. The agent treats Suite2p baseline correction in `compute_dff()` as the main expensive step and prints timings for loading, dF/F computation, and binning for each session. In the notes it says there are “none significant” beyond this and estimates roughly 0.5 to 0.7 seconds per session.

ii.
```python
t1 = time.time()
dff = compute_dff(F, Fneu, ops)
t_dff = time.time() - t1
print(f"  [{subject}/{session}] dF/F computed ({t_dff:.2f}s), range: [{dff.min():.2f}, {dff.max():.2f}]")

t2 = time.time()
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
me_binned = bin_data(me_aligned.reshape(1, -1), BIN_SIZE, axis=1).flatten()
t_bin = time.time() - t2
```

iii. In Step 6 of `CONVERSION_NOTES.md`, the agent says the script “uses GPU for Suite2p baseline computation when available” and that there are no significant inefficiencies because each session processes quickly.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent did not identify specific problematic loops in its notes and instead emphasized that the binning operation is vectorized. In the actual code, however, several Python loops remain: trial splitting loops over trials, motion-energy discretization loops over trials, time-input creation loops over trials, and the main conversion loops over sessions and subjects.

ii.
```python
for t in range(n_trials):
    start = t * timepoints_per_trial
    end = start + timepoints_per_trial
    ...

for me_trial in me_binned_trials:
    bin_indices = np.digitize(me_trial, bin_edges[1:-1], right=False)
    ...

for t in range(n_trials):
    trial_start_sec = t * TRIAL_DURATION_SEC
    time_in_trial = trial_start_sec + np.arange(TIMEPOINTS_PER_TRIAL) / BINNED_RATE
    input_trials.append(time_in_trial.astype(np.float32).reshape(1, -1))
```

iii. The only explicit justification in the notes is that there were “None significant” inefficiencies and that vectorized binning plus optional GPU use made the script fast enough in practice.

## 6-c. What processing does the code repeat multiple times?

i. The agent did not explicitly call out repeated processing in its notes. The code itself makes several separate passes over the same trialized data: it splits neural and motion-energy streams in separate loops, loops again to discretize motion-energy trials, loops again to build the time input, and loops again when formatting output and optional plots.

ii.
```python
neural_trials = split_into_trials(dff_binned, TIMEPOINTS_PER_TRIAL)
me_trials = split_into_trials(me_binned, TIMEPOINTS_PER_TRIAL)
...
me_discrete_trials, bin_edges = discretize_motion_energy(me_trials, N_ME_BINS)
...
for t in range(n_trials):
    trial_start_sec = t * TRIAL_DURATION_SEC
    ...
    input_trials.append(...)

output_trials = [me.astype(np.int64).reshape(1, -1) for me in me_discrete_trials]
neural_trials = [n.astype(np.float32) for n in neural_trials]
```

iii. The notes suggest the agent considered these extra passes acceptable because overall runtime was low, but they do not provide a separate justification for the repetition.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent did not mark any unnecessary core processing in its notes. In practice, the script loads `ops.npy` only to pull preprocessing parameters that match defaults in this dataset, prints extensive diagnostics, and has an optional `--show-processing` path that generates multi-panel plots and extra statistics that are not used by the downstream decoder. It also returns per-session motion-energy bin edges only to store them in metadata.

ii.
```python
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
...
print(f"  [{subject}/{session}] Loaded: {n_neurons} neurons, {n_neural_frames} neural frames, "
      f"{n_cam_frames} camera frames (load: {t_load:.2f}s)")
...
if show_processing:
    plot_processing(subject, session, session_idx,
                   F, Fneu, dff, me, me_aligned, me_binned,
                   dff_binned, neural_trials, me_trials, me_discrete_trials,
                   input_trials, bin_edges)
...
'session_info': session_info,
```

iii. The notes only justify these extras as sanity checks and visual verification during development. They do not claim they are needed for downstream decoding.
