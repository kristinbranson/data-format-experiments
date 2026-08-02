# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the six subject IDs, iterates over each subject directory, finds all non-hidden session subdirectories in sorted order, and for each session loads neural fluorescence from `suite2p/plane0/F.npy` and `Fneu.npy` plus behavior from `move_deve/motion_energy_glob.npy` and `tstamps.npy`. Trials are not loaded directly; the whole session is loaded first and later split into 2-minute blocks.

ii. ```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(subject_dir):
    sessions = []
    for d in sorted(os.listdir(subject_dir)):
        full = os.path.join(subject_dir, d)
        if os.path.isdir(full) and not d.startswith('.'):
            sessions.append(full)
    return sessions

F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

iii. In `CONVERSION_NOTES.md`, the agent says the dataset consists of six mouse folders, each containing daily session folders, and that `load_data.ipynb` demonstrates loading `F.npy` directly. The trajectory shows it inspected `/app/data/README.md`, `/app/data/load_data.ipynb`, and then wrote code that mirrors that directory layout.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded `SUBJECTS` list. The outer loop enumerates this list, and `subject_idx` is built from that enumeration so each session inherits the index of its mouse.

ii. ```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
    ...
    subject_idx_list.append(subj_idx)

'subjects': SUBJECTS if not sample else SUBJECTS[:1],
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. The notes say the subject folders correspond to the six mice and match the paper's alphabetical mouse ordering. The trajectory shows the agent explicitly counted sessions per `jm031`...`jm046` and adopted that same order.

## 1-c. How are the data split into sessions?

i. Sessions are defined as the sorted child directories inside each mouse directory. Each session stays separate throughout conversion and becomes one entry in `neural`, `input`, `output`, `brain_region_idx`, and `subject_idx`.

ii. ```python
def get_sessions(subject_dir):
    sessions = []
    for d in sorted(os.listdir(subject_dir)):
        full = os.path.join(subject_dir, d)
        if os.path.isdir(full) and not d.startswith('.'):
            sessions.append(full)
    return sessions

sessions = get_sessions(subject_dir)
...
for sess_dir in sessions:
    ...
    session_data.append((subj_idx, sess_dir, dff_binned, me_binned, n_neurons))
```

iii. The notes describe each `YYYY-MM-DD_a` folder as one recording day. The trajectory shows the agent listed subject/session directories and treated each day folder as a separate session.

## 1-d. How are the data split into trials?

i. The agent decides there are no native trial markers, so it creates pseudo-trials by chopping each session into consecutive 2-minute blocks after 10-frame temporal binning. Only complete 2-minute blocks are kept.

ii. ```python
TRIAL_DURATION_S = 120.0
BIN_SIZE = 10
FS = 30.0

def split_into_trials(dff_binned, me_binned):
    bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360
    n_bins = dff_binned.shape[1]
    n_trials = n_bins // bins_per_trial

    neural_trials = []
    me_trials = []
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])
```

iii. `CONVERSION_NOTES.md` says this choice was based on the paper's decoder analysis using "consecutive 2 minute blocks" for cross-validation and on the target format requiring multiple trials per session.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering. Every complete 2-minute block is kept. Incomplete trailing data shorter than one full block are implicitly dropped by integer division.

ii. ```python
n_trials = n_bins // bins_per_trial
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. The notes say "No explicit trial curation mentioned - continuous recording." The agent therefore treated all full blocks as valid trials and did not add extra rejection criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw fluorescence `F.npy` and neuropil fluorescence `Fneu.npy` in each session's Suite2p output. The agent does not use `spks.npy`.

ii. ```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))

def compute_dff(F, Fneu, device=None):
    F_corr = F - NEUCOEFF * Fneu
    ...
```

iii. The notes state that `load_data.ipynb` loads `F.npy` and that the paper says to use baseline-corrected fluorescence traces as dF/F with Suite2p defaults, though `spks.npy` was considered as an alternative.

## 2-b. How is the `neural` data processed?

i. The agent computes dF/F by subtracting `0.7 * Fneu` from `F`, running Suite2p's `preprocess` maximin baseline correction, dividing the baseline-subtracted trace by the recovered baseline, clipping the baseline away from zero, then averaging in non-overlapping 10-frame bins. After that, the binned data are split into 2-minute blocks.

ii. ```python
F_corr = F - NEUCOEFF * Fneu
F_corr_copy = F_corr.copy()
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)
baseline = F_corr - F_subtracted
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe

dff_binned = bin_traces(dff, BIN_SIZE)
```

iii. The notes and trajectory show the agent inspected Suite2p parameters in `ops.npy` and Suite2p extraction code, then decided to reproduce the paper's "baseline corrected fluorescence traces" with default Suite2p settings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent applies no additional neuron-level QC during conversion. It assumes the provided Track2p Suite2p matrices already contain only successfully tracked neurons across all days, and it ignores `iscell.npy` during conversion.

ii. ```python
# Load neural data
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))

# No iscell filtering here
dff = compute_dff(F, Fneu, device=device)
```

iii. The notes explicitly say "All neurons are already filtered to cells tracked across ALL days" and "iscell.npy has all values = 1.0". The trajectory shows the agent inspected `iscell.npy` and then chose not to filter further.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. In practice, each neural trial is just a consecutive 2-minute slice from the binned continuous session. The metadata claims alignment to the "Start of recording session", but the per-trial organization does not preserve a single shared zero point across trials within a session.

ii. ```python
neural_trials.append(dff_binned[:, start:end].astype(np.float32))

'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The notes say the camera and microscope are synchronized and that time zero should correspond to the start of recording. But the implementation inherited the 2-minute block convention from the paper's CV split, so the stored trials are effectively block-relative chunks rather than all being represented on one session-start time axis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at 30 Hz, so each bin is 10/30 s = 333.33 ms. The agent explicitly rebins both neural and motion-energy traces by averaging over non-overlapping 10-frame windows.

ii. ```python
FS = 30.0
BIN_SIZE = 10

def bin_traces(data, bin_size):
    ...
    return trimmed.reshape(n_features, n_bins, bin_size).mean(axis=2)

time_bin_ms = BIN_SIZE / FS * 1000
```

iii. The notes repeatedly justify this choice from the paper phrase "averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not read from any raw file. It is synthesized from the binned trial length using only the constants `BIN_SIZE` and `FS`, effectively using the within-trial bin index rather than a raw timestamp variable.

ii. ```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The notes say the decoder input should be "Time in seconds from start of trial" and list the source as "Time index". That came from the agent's interpretation that a simple regularly spaced time axis was sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the agent creates a 1 x T vector `0, 0.333..., 0.666..., ...` up to 119.666... seconds. No raw timestamps, no session offset, and no missing-frame correction are used.

ii. ```python
time_bin_s = BIN_SIZE / FS  # seconds per bin
return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The justification in the notes is that the decoder input is "time elapsed" and that all data were rebinned uniformly to 10-frame bins, so a regular grid was used.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input has the same number of bins as each neural trial and is generated after neural trial slicing, so it is aligned bin-for-bin with each stored neural block. However, it resets to zero at the start of every 2-minute block instead of continuing from the beginning of the session.

ii. ```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
```

iii. The notes describe the intended alignment as "start of recording session", but the actual implementation follows the pseudo-trial structure and therefore makes the input block-relative.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output comes from `move_deve/motion_energy_glob.npy`, with `tstamps.npy` loaded to help handle frame-count mismatches.

ii. ```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

iii. The data README says `move_deve` contains processed behavioral motion energy and that `tstamps.npy` or `interframe_int.npy` can be used when camera frames are missing. The agent cites that README directly in code comments and notes.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent does not recompute motion energy from video. It loads the already processed motion-energy trace, linearly interpolates it if its length differs from the neural frame count, averages it into 10-frame bins, and later discretizes it. It does not perform a separate amplitude normalization step before discretization.

ii. ```python
def interpolate_motion_energy(me, n_neural_frames, tstamps):
    if n_me == n_neural_frames:
        return me.astype(np.float64)
    if n_me < n_neural_frames:
        neural_indices = np.arange(n_neural_frames)
        me_indices = np.linspace(0, n_neural_frames - 1, n_me)
        me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
        return me_interp
    else:
        return me[:n_neural_frames].astype(np.float64)

me_binned = bin_traces(me_interp, BIN_SIZE)
```

iii. The justification in the notes is that the dataset already provides processed motion energy, the paper bins behavior by 10 frames, and the data README explicitly allows interpolation over missing camera frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent pools all binned motion-energy values across the entire converted dataset, computes global percentile edges at 0, 20, 40, 60, 80, and 100 percent, and assigns each binned value to one of five integer classes.

ii. ```python
def compute_me_percentile_bins(all_me_values, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_me_values, percentiles)
    return bin_edges

labels = np.digitize(me_values, bin_edges[1:-1])
labels = np.clip(labels, 0, n_bins - 1)
```

iii. The notes explicitly call this a "global percentile" decision and justify it with the task requirement that motion energy be "normalized and discretized into five equal-percentile bins."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first resampled to the neural frame count, then binned with the same 10-frame windows as neural data, then cut into the same 2-minute pseudo-trials. The intended alignment is one behavior value per neural bin, but the interpolation uses evenly spaced synthetic indices rather than the loaded `tstamps`.

ii. ```python
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
me_binned = bin_traces(me_interp, BIN_SIZE)
...
neural_trials, me_trials = split_into_trials(dff_binned, me_binned)
...
me_disc = discretize_me(me_t, bin_edges)
output_trials.append(me_disc.reshape(1, -1))
```

iii. The notes say the camera was triggered by the microscope and therefore should be 1:1 with imaging frames except for occasional missing frames; the trajectory shows the agent chose interpolation to repair those mismatches before matching behavior bins to neural bins.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main error-handling choices are: interpolate motion-energy traces when camera frames are missing, truncate motion energy if it is longer than neural data, clip the dF/F baseline to `1e-6` to avoid division by zero, skip missing subject directories with a warning, and silently drop incomplete trailing partial trials.

ii. ```python
baseline_safe = np.clip(baseline, 1e-6, None)

if n_me < n_neural_frames:
    ...
    me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
elif n_me > n_neural_frames:
    return me[:n_neural_frames].astype(np.float64)

if not os.path.isdir(subject_dir):
    print(f"Warning: Subject directory not found: {subject_dir}")
    continue
```

iii. The README note about missing camera frames is the explicit justification for interpolation. The notes also mention avoiding divide-by-zero in dF/F and using all available data unless a file is missing.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is per-session dF/F computation, especially the Suite2p `preprocess` baseline correction over large `(neurons x frames)` matrices. Full-session file loading is the other main cost. The remaining steps are comparatively light.

ii. ```python
dff = compute_dff(F, Fneu, device=device)
...
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)
```

iii. The notes say the script uses GPU acceleration for Suite2p preprocessing and report roughly 1 second per session, which implies this stage was the bottleneck the agent was optimizing for.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop that slices each session into trials, and the inner loop that recreates `time_input` and discretizes output per trial, could be vectorized by reshaping whole-session arrays into `(n_trials, ...)` blocks. The subject/session directory traversal cannot be meaningfully vectorized in the same way.

ii. ```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])

for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
    me_disc = discretize_me(me_t, bin_edges)
```

iii. The notes emphasize vectorized binning via `reshape + mean`, but the later trial-building logic remains Python-loop based because the target format is a nested list of per-trial arrays.

## 6-c. What processing does the code repeat multiple times?

i. The code makes two passes over the session data: one to process sessions and collect all motion-energy values for global percentile edges, and another to split sessions into trials and discretize outputs. If plotting is enabled, it also re-reads raw files already loaded earlier.

ii. ```python
# First pass
for subj_idx, subject in enumerate(subjects_to_process):
    ...
    dff_binned, me_binned, n_neurons, n_frames_raw = process_session(...)
    session_data.append((subj_idx, sess_dir, dff_binned, me_binned, n_neurons))
    all_me_binned_values.append(me_binned)

# Second pass
for subj_idx, sess_dir, dff_binned, me_binned, n_neurons in session_data:
    neural_trials, me_trials = split_into_trials(dff_binned, me_binned)
    ...

F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
```

iii. The notes justify the two-pass structure as necessary for global percentile binning. The plotting reloads appear to have been added only for visual verification.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `tstamps.npy` but does not actually use it in interpolation, maintains an unused `session_count`, passes an unused `n_frames_raw` argument into `plot_processing`, and optionally generates processing plots that are not part of the saved dataset. It also computes string labels for output bins purely for metadata.

ii. ```python
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
...
me_interp = interpolate_motion_energy(me, n_frames, tstamps)

session_count = 0
...
session_count += 1

def plot_processing(session_dir, dff_binned, me_binned, me_disc_trials,
                    neural_trials, n_frames_raw, save_path):
    ...
```

iii. The trajectory shows plotting and metadata were added as sanity-check and usability features. The notes do not claim these steps affect the decoder itself.
