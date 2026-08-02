# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI lists all subject directories under `data/` whose names start with `jm`, lists all subdirectories inside each subject as sessions, then processes each session by loading `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, `move_deve/motion_energy_glob.npy`, and `move_deve/tstamps.npy`. Trials are not loaded directly; they are created later by splitting each processed session into fixed-length blocks.

ii. 
```python
def get_subjects_and_sessions(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
    sessions = {}
    for subj in subjects:
        subj_path = os.path.join(data_dir, subj)
        sess_list = sorted([s for s in os.listdir(subj_path) 
                           if os.path.isdir(os.path.join(subj_path, s))])
        sessions[subj] = sess_list
    return subjects, sessions

F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset is organized as `data/{subject_id}/{date}_a/{suite2p,move_deve}/` and that each session contains those neural and behavioral files. The trajectory shows it decided all `jm*` folders are mice and all subdirectories are daily sessions.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directory name: every directory in `data/` beginning with `jm` becomes one subject. They are sorted alphabetically. In the final output, `used_subjects` preserves first-seen order while iterating through the sorted subject/session list.

ii. 
```python
subjects = sorted([d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

used_subjects = []
...
if subj not in used_subjects:
    used_subjects.append(subj)
subject_idx_list.append(used_subjects.index(subj))
```

iii. The notes state there are 6 mice, identified by folders `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`, and the trajectory repeatedly treats each `jm*` directory as one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are split as subject subdirectories, sorted alphabetically within each subject. In full mode the AI processes all such session directories. In sample mode it keeps only two specific sessions, `session_list[0]` and `session_list[7]`, corresponding to the first session from `jm031` and `jm032`.

ii. 
```python
sess_list = sorted([s for s in os.listdir(subj_path) 
                   if os.path.isdir(os.path.join(subj_path, s))])
sessions[subj] = sess_list

session_list = []
for subj in subjects:
    for sess in sessions[subj]:
        session_list.append((subj, sess))

if args.sample:
    session_list = [session_list[0], session_list[7]]
```

iii. The AI justifies this from the directory structure it documented in `CONVERSION_NOTES.md`, where each `{date}_a` folder is one daily recording session.

## 1-d. How are the data split into trials?

i. The AI does not use natural trials from the raw data. It bins continuous sessions by 10 frames first, then splits the binned session into non-overlapping 2-minute blocks. At 30 Hz and 10-frame binning, each trial has 360 binned timepoints. Any leftover frames at the end of the session are dropped implicitly by floor division.

ii. 
```python
BIN_SIZE = 10
TRIAL_DURATION_SEC = 120
TRIAL_FRAMES_RAW = int(TRIAL_DURATION_SEC * FS)
TRIAL_FRAMES_BINNED = TRIAL_FRAMES_RAW // BIN_SIZE

def split_into_trials(data, trial_length):
    n_trials = n_timepoints // trial_length
    trials = []
    for i in range(n_trials):
        start = i * trial_length
        end = start + trial_length
        trials.append(data[:, start:end].astype(np.float32))
    return trials

neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)
```

iii. The notes explicitly justify 2-minute blocks by citing the paper’s decoder cross-validation splits: “Split each session into 2-minute blocks (as used for CV in paper).” The trajectory shows the AI deliberately chose this block length from the methods text rather than from raw trial structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies no explicit trial quality-control filtering. The only effective filtering is that incomplete final trial fragments are discarded when the total number of timepoints is not divisible by the fixed trial length.

ii. 
```python
n_trials = n_timepoints // trial_length
for i in range(n_trials):
    start = i * trial_length
    end = start + trial_length
    trials.append(data[:, start:end].astype(np.float32))
```

iii. `CONVERSION_NOTES.md` says “No explicit trial curation mentioned” and treats missing camera frames as a signal-alignment issue rather than a reason to drop trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data is derived from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy` for each session.

ii. 
```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The notes and trajectory both identify these files as the relevant raw fluorescence and neuropil traces and reject using `spks.npy` for the converted neural signal.

## 2-b. How is the `neural` data processed?

i. The AI computes neuropil-corrected fluorescence as `F - 0.7 * Fneu`, then runs Suite2p’s `preprocess(..., 'maximin', ...)` baseline-subtraction routine with `win_baseline=60`, `sig_baseline=10`, and `fs=30`. After that, it averages every 10 consecutive frames before trializing.

ii. 
```python
Fc = F - neucoeff * Fneu
device = torch.device('cpu')
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)

dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The notes say the AI examined Suite2p and concluded the relevant operation is baseline subtraction, not division, and that the paper used “baseline corrected fluorescence traces as our dF/F.” The notes then add 10-frame averaging because the paper’s decoder averaged by 10 frames.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied in the script. The AI uses every row in `F.npy` and `Fneu.npy` and does not read or apply `iscell.npy`.

ii. 
```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
n_neurons, n_neural_frames = F.shape
```

iii. The notes justify this by saying all `iscell` values are already 1 and that Track2p output already contains only tracked cells across days.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to session start rather than to any behavioral event. After optional binning, each trial is just a consecutive block from the continuous session, and the metadata records `temporal_alignment_event` as `session_start`.

ii. 
```python
'metadata': {
    'temporal_alignment_event': 'session_start',
    'off_start': 0.0,
    'off_end': None,
    ...
}
```

iii. The trajectory says the recording is continuous and the decoder input should be “time elapsed from session start,” so the AI treated session start as the only alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins the session by averaging every 10 imaging frames, so the final temporal resolution is `10 / 30 = 0.333...` seconds per time bin, or 333.33 ms.

ii. 
```python
BIN_SIZE = 10
TIME_BIN_MS = (BIN_SIZE / FS) * 1000

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. The notes justify this directly from the paper’s decoder description: “Binning for decoding: 10 frames” and “Average 10 consecutive frames (both neural and behavioral).”

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from a saved raw time variable. The AI synthesizes it from the imaging frame index and the assumed constant frame rate `FS=30`, after 10-frame binning.

ii. 
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. The notes state the decoder input is “Time in seconds from session start, binned by 10,” and the trajectory says there are no needed timestamps beyond the fixed frame rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI creates one scalar timestamp per binned timepoint, using the center of each 10-frame bin, measured in seconds from session start. It does not use raw camera or microscope timestamps.

ii. 
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
...
trial_time = time_input[start:end].astype(np.float32)
input_trials_list.append(trial_time.reshape(1, -1))
```

iii. The code comments explicitly say the AI chose absolute time from session start because the task asked for “time elapsed from beginning of experiment.”

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The same binned session timeline is used for both neural and input data. After creating `time_input` over all binned neural frames, the AI slices the identical start/end ranges for each trial, so each input timepoint aligns one-to-one with one binned neural sample.

ii. 
```python
n_binned = dff_binned.shape[1]
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
```

iii. The trajectory and inline comments both state that time is meant to be “absolute time from session start,” matched to the same binned samples used for neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The script derives motion-energy output from `move_deve/motion_energy_glob.npy` and uses `move_deve/tstamps.npy` to infer dropped camera frames before alignment. The notes also mention `interframe_int.npy`, but the script itself does not use it.

ii. 
```python
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
```

iii. The notes document both `motion_energy_glob.npy` and timing files under `move_deve/`. The trajectory shows the AI concluded `tstamps.npy` is in kiloseconds and can reveal where camera frames are missing.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI first aligns the raw motion-energy trace to neural frame count by detecting gaps in `tstamps`, placing existing camera samples into a full-length neural-frame array, and linearly interpolating missing values. It then averages 10 consecutive frames and later discretizes the binned series. The notes say motion energy should be normalized per session before discretization, but the script does not implement an explicit normalization step.

ii. 
```python
ifi = np.diff(tstamps)
median_ifi = np.median(ifi)
...
for i in range(1, len(me)):
    n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
    neural_idx[i] = neural_idx[i-1] + 1 + n_dropped
...
me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
me_full[neural_idx] = me.astype(np.float64)
...
me_full = np.interp(indices, indices[valid], me_full[valid])

me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. The notes justify interpolation because some sessions have missing camera frames, and they justify 10-frame averaging from the paper’s decoder preprocessing. The notes also claim “normalize per session ... then discretize,” which appears to be an intended decision that was not carried through in the final code.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI discretizes motion energy into 5 equal-percentile bins separately for each session, using percentile cutoffs at 20, 40, 60, and 80 percent on the binned motion-energy trace and `np.digitize` to assign labels 0-4.

ii. 
```python
def discretize_motion_energy(me_binned, n_bins=N_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    bin_edges = np.percentile(me_binned, percentiles)
    binned = np.digitize(me_binned, bin_edges)
    return binned.astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says the bins are computed per session “to account for different motion levels across days/mice,” and the trajectory repeats that the goal is balanced 5-way categorization.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is expanded to neural-frame length before any trial splitting by mapping observed camera frames onto inferred neural frame indices, interpolating missing indices, then averaging both neural and motion-energy streams with the same 10-frame binning. Trials are then cut with matching start/end indices.

ii. 
```python
me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
...
trial_me = me_discrete[start:end].astype(np.int64)
output_trials_list.append(trial_me.reshape(1, -1))
```

iii. The trajectory shows the AI focused heavily on handling sessions where camera frames are missing so that motion energy and neural data can be indexed together sample-for-sample.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI only handles one class of minor data issue explicitly: missing camera frames in the motion-energy stream. It reconstructs a full-length motion-energy vector by inferring dropped-frame counts from timestamp gaps and interpolating across missing samples. Partial trailing trial fragments are dropped implicitly during floor-division trial splitting. No additional repair logic is applied to neural data or metadata.

ii. 
```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)

ifi = np.diff(tstamps)
median_ifi = np.median(ifi)
...
me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
me_full[neural_idx] = me.astype(np.float64)
...
me_full = np.interp(indices, indices[valid], me_full[valid])
```

iii. The notes identify missing camera frames as the main data-quality issue in this dataset and plan to “interpolate ME to neural frame count before binning.”

## 6-a. What are the most time-consuming steps of the code?

i. The AI considered Suite2p preprocessing of fluorescence the dominant cost. It times `compute_dff` separately inside each session and reports overall runtime around 45 seconds for the full dataset. Binning and trial splitting are treated as cheap.

ii. 
```python
t1 = time.time()
dff = compute_dff(F, Fneu)
print(f"    dF/F computation: {time.time()-t1:.2f}s")

t2 = time.time()
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
print(f"    Binning: {n_neural_frames} -> {n_binned} timepoints ({time.time()-t2:.2f}s)")
```

iii. In the trajectory, the AI repeatedly highlights dF/F computation as the expensive part and summarizes the full conversion as roughly 0.5-1.0 seconds per session and about 45 seconds total.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI did not explicitly discuss vectorization in its notes, but its code leaves several loops unvectorized: the dropped-frame index construction in `align_motion_energy`, the per-trial slicing loop in `split_into_trials`, and separate per-trial loops for building `input` and `output`.

ii. 
```python
for i in range(1, len(me)):
    n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
    neural_idx[i] = neural_idx[i-1] + 1 + n_dropped

for i in range(n_trials):
    start = i * trial_length
    end = start + trial_length
    trials.append(data[:, start:end].astype(np.float32))

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    ...
```

iii. There is no explicit justification beyond straightforward implementation. The trajectory suggests the AI prioritized getting a valid conversion and decoder run rather than optimizing these loops.

## 6-c. What processing does the code repeat multiple times?

i. Trial-boundary computation is repeated. The script first derives `neural_trials` with `split_into_trials`, then recomputes the same `start` and `end` values in separate loops to slice `time_input` and `me_discrete`. The session traversal also repeats the same processing pattern for every session.

ii. 
```python
neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_me = me_discrete[start:end].astype(np.int64)
```

iii. The AI gives no explicit justification for this repetition. It appears to be a simple implementation choice rather than a deliberate design decision.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main unnecessary work is optional diagnostic plotting via `plot_processing`, which generates figures for the first two sessions but is not used by the saved converted dataset or downstream decoder training. Aside from that, the script does not contain major obviously discarded processing.

ii. 
```python
if show_processing and sess_idx < 2:
    plot_processing(subj, sess_name, dff, me_aligned, dff_binned, me_binned, 
                   me_discrete, time_input, neural_trials, output_trials_list,
                   sess_idx)
```

iii. The notes justify these plots as a sanity check during sample validation, not as part of the final dataset representation.
