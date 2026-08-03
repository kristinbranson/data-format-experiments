# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script enumerates all subject folders under `data/` whose names start with `jm`, enumerates each subject's session subdirectories, and processes each session in a loop. For each session it loads neural fluorescence (`F.npy`, `Fneu.npy`) and behavioral motion-energy/timestamp files (`motion_energy_glob.npy`, `tstamps.npy`). Trials are not stored in the raw data; they are created later by splitting each processed session into fixed 2-minute blocks.

ii. ```python
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

...

F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
...
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. In `CONVERSION_NOTES.md` Step 2 the agent documented the folder layout `data/{subject_id}/{date}_a/{suite2p,move_deve}` and listed the specific files present in each session. In trajectory step 4 it summarized the same structure and treated the Track2p/Suite2p session folders as the canonical loading path.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level folder name. Any directory in `data/` starting with `jm` is treated as one mouse. Session-level `subject_idx` values are then assigned from the order in which subjects first appear while iterating the sorted folder list.

ii. ```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

...

used_subjects = []
...
if subj not in used_subjects:
    used_subjects.append(subj)
subject_idx_list.append(used_subjects.index(subj))
```

iii. The data README excerpt captured in trajectory step 4 explicitly said each subject has its own folder and maps `jm031` to mouse A, `jm032` to mouse B, etc. The agent therefore used subject folder boundaries directly.

## 1-c. How are the data split into sessions?

i. Sessions are split by the immediate subdirectories under each subject folder. Each session directory name (for example `2023-10-18_a`) becomes one output session. Session order is lexical order within each mouse, and the final dataset order is all sessions of subject 1, then all sessions of subject 2, and so on.

ii. ```python
for subj in subjects:
    subj_path = os.path.join(data_dir, subj)
    sess_list = sorted([s for s in os.listdir(subj_path)
                       if os.path.isdir(os.path.join(subj_path, s))])
    sessions[subj] = sess_list

...

session_list = []
for subj in subjects:
    for sess in sessions[subj]:
        session_list.append((subj, sess))
```

iii. In trajectory steps 4 and 6, the agent quoted the dataset README language that each session subfolder corresponds to one recording day and that the suffix `_a` can be ignored. It used those directories as the session split.

## 1-d. How are the data split into trials?

i. The raw recordings are continuous sessions with no explicit trial table. The script creates synthetic trials by splitting each session into consecutive 2-minute chunks. At 30 Hz this is 3600 raw frames; after 10-frame binning each trial has 360 time bins. The same chunk boundaries are applied to neural, input, and output.

ii. ```python
TRIAL_DURATION_SEC = 120
TRIAL_FRAMES_RAW = int(TRIAL_DURATION_SEC * FS)
TRIAL_FRAMES_BINNED = TRIAL_FRAMES_RAW // BIN_SIZE

...

def split_into_trials(data, trial_length):
    if data.ndim == 2:
        n_neurons, n_timepoints = data.shape
        n_trials = n_timepoints // trial_length
        trials = []
        for i in range(n_trials):
            start = i * trial_length
            end = start + trial_length
            trials.append(data[:, start:end].astype(np.float32))
        return trials

...

neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)
```

iii. In trajectory steps 23 and 24, the agent noted that the data are continuous and not trialized, then decided to reuse the paper's decoder cross-validation unit: “splits were done on consecutive 2 minute blocks.” `CONVERSION_NOTES.md` Step 5 records the same decision.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filtering. Every full 2-minute block produced by floor division is kept. Missing camera frames are repaired before trial splitting, but trials are not removed for motion artifacts, low behavior variance, or other QC criteria.

ii. ```python
neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)

input_trials_list = []
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    input_trials_list.append(trial_time.reshape(1, -1))

output_trials_list = []
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_me = me_discrete[start:end].astype(np.int64)
    output_trials_list.append(trial_me.reshape(1, -1))
```

iii. `CONVERSION_NOTES.md` Step 3 says “No explicit trial curation mentioned,” and the agent carried that into the final code. The only data issue it chose to repair was missing camera frames before building trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw Suite2p fluorescence arrays `F.npy` and `Fneu.npy`. No use is made of `spks.npy`, `iscell.npy`, or `stat.npy` when constructing the neural output.

ii. ```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))

...

Fc = F - neucoeff * Fneu
```

iii. In trajectory steps 6, 22, and 23, the agent concluded from the paper/methods that downstream analyses used Suite2p “baseline corrected fluorescence traces” rather than `spks.npy`, so it based `neural` on `F` and `Fneu`.

## 2-b. How is the `neural` data processed?

i. The script performs neuropil correction `Fc = F - 0.7 * Fneu`, then calls Suite2p's `preprocess(..., 'maximin', ...)` to do maximin baseline subtraction using `win_baseline=60 s`, `sig_baseline=10`, and `fs=30`. It then averages every 10 frames and splits the binned traces into 2-minute blocks.

ii. ```python
def compute_dff(F, Fneu, neucoeff=NEUCOEFF, win_baseline=WIN_BASELINE,
                sig_baseline=SIG_BASELINE, fs=FS):
    Fc = F - neucoeff * Fneu
    device = torch.device('cpu')
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)
    return dff

...

dff = compute_dff(F, Fneu)
dff_binned = bin_data(dff, BIN_SIZE)
neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)
```

iii. In trajectory step 22, the agent explicitly corrected an earlier misconception and noted that Suite2p `preprocess` returns baseline-subtracted fluorescence, not `(F-Flow)/Flow`. `CONVERSION_NOTES.md` Steps 1 and 3 record the same Suite2p-default processing choice.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code applies no explicit neural filtering. It assumes the supplied Track2p/Suite2p outputs are already curated: the tracked-cell export already contains only cells present across all days, and the agent observed that `iscell` was all ones. Thus all rows of `F.npy` are kept.

ii. ```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
n_neurons, n_neural_frames = F.shape
...
brain_region_idx_list.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. `CONVERSION_NOTES.md` says “All iscell values are 1 (all cells already filtered)” and that Track2p output “only includes traces for the cells present across all days.” Trajectory steps 7 and 9 also state that all `iscell` entries were 1 and no further row filtering seemed necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script does not align neural activity to a behavioral or sensory event. Instead it treats the continuous recording start as the alignment origin, bins the whole session from frame 0 onward, and then chops the session into 2-minute blocks. Metadata declares the alignment event as `session_start`.

ii. ```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
...
neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)

...

'metadata': {
    'temporal_alignment_event': 'session_start',
    'off_start': 0.0,
    'off_end': None,
    ...
}
```

iii. In trajectory steps 23 and 24, the agent reasoned that because the requested decoder input was “time elapsed from beginning of experiment,” the natural reference event was session start. It then used 2-minute blocks only as an output-format convenience.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 333.33 ms bins. This comes from averaging 10 consecutive frames at a 30 Hz imaging rate. The same 10-frame rebinning is applied to neural activity and motion energy before the trial split.

ii. ```python
FS = 30.0
BIN_SIZE = 10
TIME_BIN_MS = (BIN_SIZE / FS) * 1000

...

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. `CONVERSION_NOTES.md` Step 3 quotes the methods statement that decoding analyses “slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps.” The agent used that as the decisive temporal-processing rule.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time series is not read from a timestamp file. It is derived from the number of binned neural timepoints plus the constants `FS=30` and `BIN_SIZE=10`, effectively using neural frame index as elapsed experiment time.

ii. ```python
n_binned = dff_binned.shape[1]
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. In trajectory steps 23 and 24 the agent treated “time elapsed from beginning” as an instruction-defined variable rather than a raw recorded channel, and computed it from frame count after confirming the imaging rate was 30 Hz.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. After neural/session-level binning, the script constructs one scalar per bin equal to the center time of that 10-frame bin, measured in seconds from session start. It then slices this session-long vector into the same 2-minute trial blocks as the neural data and reshapes each trial to `(1, n_timepoints)`.

ii. ```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS

input_trials_list = []
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    input_trials_list.append(trial_time.reshape(1, -1))
```

iii. The inline code comment records the main choice: the author first considered trial-relative time, then reversed course because “the task says time elapsed from beginning of experiment.” That same interpretation is repeated in the README and trajectory.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. It is aligned by construction. The time vector has exactly one value per binned neural sample, is generated after neural binning, and is sliced using the same trial boundaries as `neural`.

ii. ```python
dff_binned = bin_data(dff, BIN_SIZE)
n_binned = dff_binned.shape[1]
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
```

iii. In trajectory step 24 the agent listed the temporal recipe as: bin both traces by 10, define input as elapsed time, then split everything into shared 2-minute blocks. There was no separate alignment procedure beyond shared indexing.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived primarily from `move_deve/motion_energy_glob.npy`. The script also uses `move_deve/tstamps.npy` to place motion-energy samples onto neural-frame indices when camera frames are missing.

ii. ```python
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
```

iii. The data README excerpt captured in trajectory step 4 said `motion_energy_glob.npy` contains processed behavioral motion energy and that `tstamps.npy`/`interframe_int.npy` can be used to locate missing camera frames. The agent followed that recommendation.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The script aligns motion-energy samples to neural frames, fills missing camera frames by linear interpolation, averages every 10 frames, and then immediately discretizes the binned values. It does not perform an explicit normalization step in code, even though the notes and planning text discuss normalization.

ii. ```python
me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
me_binned = bin_data(me_aligned, BIN_SIZE)
me_discrete = discretize_motion_energy(me_binned, N_BINS)
```

iii. The agent's notes show some uncertainty here. `CONVERSION_NOTES.md` Step 5 says “normalize per session ... then discretize,” while trajectory steps 24, 25, and 45 continue discussing whether normalization should be added. The final code settles on direct percentile discretization without separate normalization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After session-level alignment and binning, the script computes the 20th, 40th, 60th, and 80th percentiles of the session's motion-energy values and uses `np.digitize` to convert each binned sample into one of five integer classes `0` through `4`.

ii. ```python
def discretize_motion_energy(me_binned, n_bins=N_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    bin_edges = np.percentile(me_binned, percentiles)
    binned = np.digitize(me_binned, bin_edges)
    return binned.astype(np.int64)
```

iii. The task instructions explicitly requested “five equal-percentile bins,” and the agent repeated that requirement in `CONVERSION_NOTES.md` and the README. It chose per-session percentiles to equalize bin counts within each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned before binning. If the camera and neural frame counts already match, the raw motion-energy vector is used directly. Otherwise the script infers where camera frames were dropped from timestamp gaps, places observed motion-energy samples onto the corresponding neural-frame indices, linearly interpolates missing neural-frame positions, then bins and trializes using the same boundaries as the neural data.

ii. ```python
def align_motion_energy(me, n_neural_frames, tstamps):
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    ifi = np.diff(tstamps)
    median_ifi = np.median(ifi)
    neural_idx = np.zeros(len(me), dtype=int)
    neural_idx[0] = 0
    for i in range(1, len(me)):
        n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
        neural_idx[i] = neural_idx[i-1] + 1 + n_dropped
    neural_idx = np.clip(neural_idx, 0, n_neural_frames - 1)
    me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
    me_full[neural_idx] = me.astype(np.float64)
    valid = ~np.isnan(me_full)
    if not valid.all():
        indices = np.arange(n_neural_frames)
        me_full = np.interp(indices, indices[valid], me_full[valid])
    return me_full
```

iii. The data README excerpt explicitly allowed missing motion-energy frames to be “treated as missing values ... or interpolated over.” In trajectory step 47, the agent rejected a faulty timestamp-to-time mapping and adopted the final IFI-gap reconstruction method now present in the script.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The only explicit missing-data repair is for dropped camera frames in motion energy. Those missing positions are inferred from timestamp gaps and filled by linear interpolation. Otherwise the script is not defensive: it assumes required files exist, does not validate `iscell`, and silently drops any trailing frames that would not complete a full bin or trial.

ii. ```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)
...
me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
me_full[neural_idx] = me.astype(np.float64)
valid = ~np.isnan(me_full)
if not valid.all():
    indices = np.arange(n_neural_frames)
    me_full = np.interp(indices, indices[valid], me_full[valid])

...

n_bins = n_frames // bin_size
data_trimmed = data[:, :n_bins * bin_size]
```

iii. The justification comes directly from the README excerpt quoted in trajectory step 4, which said missing camera frames could be identified from timestamps and interpolated. The agent focused on that specific known data issue and did not add broader error handling.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is per-session neural preprocessing: loading large fluorescence matrices and running Suite2p `preprocess` over all neurons and frames. Optional plotting is also expensive because it keeps and renders large unbinned arrays. In contrast, percentile binning and session bookkeeping are minor.

ii. ```python
t1 = time.time()
dff = compute_dff(F, Fneu)
print(f"    dF/F computation: {time.time()-t1:.2f}s")

...

if show_processing and sess_idx < 2:
    plot_processing(subj, sess_name, dff, me_aligned, dff_binned, me_binned,
                   me_discrete, time_input, neural_trials, output_trials_list,
                   sess_idx)
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly describes the script as doing per-session dF/F computation and gives rough runtime numbers. The code itself times dF/F and binning but not smaller bookkeeping steps, which supports the conclusion that preprocessing is the main cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops could be vectorized: the dropped-frame reconstruction loop in `align_motion_energy`, the per-trial slicing loops in `split_into_trials`, and the nearly identical loops that build `input_trials_list` and `output_trials_list`. The script already vectorizes frame averaging with reshape/mean, but not trial packing.

ii. ```python
for i in range(1, len(me)):
    n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
    neural_idx[i] = neural_idx[i-1] + 1 + n_dropped

...

for i in range(n_trials):
    start = i * trial_length
    end = start + trial_length
    trials.append(data[:, start:end].astype(np.float32))

...

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    ...
```

iii. The trajectory does not contain a detailed optimization discussion, but the structure of `convert_data.py` makes these loops the obvious vectorization candidates because they do repeated index arithmetic over contiguous blocks.

## 6-c. What processing does the code repeat multiple times?

i. The script repeats the same processing pipeline for every session: load arrays, compute neural preprocessing, align motion energy, bin both streams, and slice into trials. Within each session it also repeats the same block-boundary calculation twice, once for time input and once for motion-energy output.

ii. ```python
for sess_i, (subj, sess_name) in enumerate(session_list):
    neural_trials, input_trials, output_trials, n_neurons = process_session(
        subj, sess_name, DATA_DIR,
        show_processing=args.show_processing,
        sess_idx=sess_i
    )

...

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    ...

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_me = me_discrete[start:end].astype(np.int64)
    ...
```

iii. This repetition follows directly from the agent's session-centric design in Step 6 of `CONVERSION_NOTES.md`. It intentionally runs the same reference-inspired pipeline independently for each recording day.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and optionally plots several intermediates that are not preserved in `converted_data.pkl`: full-resolution `dff`, full-resolution aligned motion energy, the session-long `time_input`, `me_binned`, and `bin_counts`. In `--show-processing` mode it also generates diagnostic figures that are not used by the decoder itself.

ii. ```python
print(f"    ME bin distribution: {bin_counts} (total: {bin_counts.sum()})")

if show_processing and sess_idx < 2:
    plot_processing(subj, sess_name, dff, me_aligned, dff_binned, me_binned,
                   me_discrete, time_input, neural_trials, output_trials_list,
                   sess_idx)
```

iii. The agent deliberately added these steps as sanity checks. The task instructions told it to invent validation checks and a `--show-processing` mode, so it kept extra diagnostics even though downstream decoding only consumes the final trialized tensors.
