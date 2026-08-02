# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `/app/data` for subject directories beginning with `jm`, scans each subject directory for session subdirectories, and then loads each session's fluorescence and motion-energy arrays with `np.load`. Trials are not loaded from disk; they are created later by slicing the continuous session arrays.

ii. ```python
def get_subjects_and_sessions():
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
    all_sessions = {}
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        all_sessions[subj] = sessions
    return subjects, all_sessions

F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. In `CONVERSION_NOTES.md`, the agent says the provided data are already Track2p-matched Suite2p outputs, so loading `F.npy`, `Fneu.npy`, and `motion_energy_glob.npy` is sufficient for conversion.

## 1-b. How are the data split into subjects?

i. Subjects are the top-level `jm*` folders under `data/`, sorted lexicographically. The script also remaps them to paper labels `Mouse_A` through `Mouse_F` using a hard-coded dictionary.

ii. ```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])

SUBJECT_MAP = {
    'jm031': 'Mouse_A', 'jm032': 'Mouse_B', 'jm038': 'Mouse_C',
    'jm039': 'Mouse_D', 'jm040': 'Mouse_E', 'jm046': 'Mouse_F'
}
```

iii. The notes identify six subject folders in the dataset and explicitly map them to mice A-F from the paper.

## 1-c. How are the data split into sessions?

i. Each date-labeled subdirectory inside a subject folder is treated as one session. Sessions are sorted by directory name before processing.

ii. ```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    all_sessions[subj] = sessions
```

iii. `CONVERSION_NOTES.md` describes each session directory as one recording day and lists the per-subject session counts from the folder structure.

## 1-d. How are the data split into trials?

i. The experiment is treated as continuous rather than natively trial-based. After binning by 10 frames, each session is split into consecutive non-overlapping 2-minute blocks of `360` bins, and any remainder shorter than a full block is dropped.

ii. ```python
TRIAL_DURATION_SEC = 120
TRIAL_BINS = int(TRIAL_DURATION_SEC * FRAME_RATE / BIN_SIZE)

n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
```

iii. The notes state there are no discrete trials in the original dataset and justify 2-minute blocks by citing the paper's decoder cross-validation on consecutive 2-minute blocks.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering. Trials are kept if they survive interpolation and fit into a full 2-minute block after binning.

ii. ```python
me = interpolate_motion_energy(me_raw, n_frames)

n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. In the notes, the agent says there is "No explicit trial curation" because the recordings are continuous spontaneous activity rather than trial-based behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from Suite2p fluorescence arrays `F.npy` and `Fneu.npy`. The script does not use `spks.npy`.

ii. ```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))

Fc = F - NEUROPIL_COEFF * Fneu
```

iii. The notes say the paper used baseline-corrected fluorescence traces as dF/F and that Track2p itself does not compute dF/F, so the agent chose fluorescence rather than deconvolved spikes.

## 2-b. How is the `neural` data processed?

i. The script neuropil-corrects fluorescence as `F - 0.7 * Fneu`, applies Suite2p's `preprocess` maximin baseline correction, casts to `float32`, and then temporally bins by averaging every 10 frames.

ii. ```python
Fc = F - NEUROPIL_COEFF * Fneu
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)

dfof_binned = bin_timeseries(dfof, BIN_SIZE)
```

iii. `CONVERSION_NOTES.md` cites Suite2p default parameters from `ops.npy` and says this matches the paper's use of baseline-corrected fluorescence. The function docstring also argues that `preprocess` output should be used directly as the paper's dF/F.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied during conversion. The script assumes the released matrices already contain only matched, accepted cells.

ii. ```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
n_neurons, n_frames = F.shape
```

iii. The notes say all observed `iscell[:, 0]` values are `1` and conclude that the provided data are already filtered through Track2p matching and Suite2p cell selection.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session time, not to a behavioral event. The metadata declares the alignment event as the start of the continuous recording session, and each trial is just a contiguous slice from that timeline.

ii. ```python
'metadata': {
    'time_bin_size': TIME_BIN_MS,
    'temporal_alignment_event': 'start of continuous recording session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The notes justify this by saying the dataset contains continuous spontaneous recordings without a native trial onset event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at 30 Hz, so each time bin is about `333.33 ms`. Yes, both neural and behavioral traces are rebinned by averaging every 10 consecutive frames.

ii. ```python
BIN_SIZE = 10
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000

def bin_timeseries(data, bin_size):
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    return truncated.reshape(new_shape).mean(axis=-1)
```

iii. The notes cite the paper phrase "averaging in bins of 10 consecutive timestamps" and treat that as the key temporal preprocessing step.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not read from a raw timestamp file. It is synthesized from bin indices using the assumed frame rate and bin size.

ii. ```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. In the notes, the agent maps a "Time index" to the decoder input and does not mention using `tstamps.npy` or other hardware timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the script constructs a monotonic vector of elapsed seconds from session start by multiplying the binned time index by `10 / 30`.

ii. ```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The notes justify this as the requested "time from start" variable for a fixed-rate continuous recording and list monotonicity as a sanity check.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. It uses the same trial boundaries and the same 10-frame bins as the neural data, so each input sample corresponds one-to-one with a neural time bin.

ii. ```python
neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The notes report expected time ranges and say the time input is aligned by using the same indexing as the neural slices.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `output` is derived from `move_deve/motion_energy_glob.npy`, the precomputed session-level motion-energy trace.

ii. ```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The notes describe `motion_energy_glob.npy` as the motion-energy signal derived from videography.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code interpolates motion energy to the neural frame count if lengths differ, bins it by averaging every 10 frames, and then discretizes it. Although the notes and module docstring say motion energy is normalized per session, there is no explicit normalization step in the implementation.

ii. ```python
me = interpolate_motion_energy(me_raw, n_frames)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
me_discrete = discretize_motion_energy(me_binned, N_OUTPUT_BINS)
```

iii. `CONVERSION_NOTES.md` says the intended pipeline was "normalize per session, discretize to 5 quintile bins," but the final code only performs interpolation, averaging, and percentile-based discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. It is split per session into 5 equal-percentile bins using the 20th, 40th, 60th, and 80th percentiles of that session's binned motion-energy values. `np.digitize` assigns labels `0` through `4`.

ii. ```python
percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
thresholds = np.percentile(me_binned, percentiles)
labels = np.digitize(me_binned, thresholds).astype(np.int64)
```

iii. The notes say the agent chose quintile discretization to satisfy the decoder requirement for categorical outputs and to handle session-specific scale differences.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first resampled to match the neural frame count, then rebinned with the same 10-frame windows, then split into the same 2-minute trial slices as the neural data.

ii. ```python
me = interpolate_motion_energy(me_raw, n_frames)

dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()

output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes justify this from the methods description that imaging and camera acquisition are synchronized frame-by-frame, with interpolation used only to repair missing camera frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The only explicit missing-data handling is for motion-energy length mismatches. If motion energy has fewer samples than the neural recording, the code rescales the whole trace to the neural length with linear interpolation. It also drops incomplete tail bins and incomplete final trials by integer division.

ii. ```python
def interpolate_motion_energy(me, n_frames):
    if len(me) == n_frames:
        return me.astype(np.float64)
    x_orig = np.linspace(0, 1, len(me))
    x_new = np.linspace(0, 1, n_frames)
    me_interp = np.interp(x_new, x_orig, me.astype(np.float64))
    return me_interp

n_bins = n_frames // bin_size
truncated = data[..., :n_bins * bin_size]
```

iii. The notes cite the data README statement that missing frames may be interpolated over and list the sessions with motion-energy/neural length mismatches.

## 6-a. What are the most time-consuming steps of the code?

i. The main bottleneck is the per-session Suite2p baseline correction in `compute_dfof`, with large-array loading and optional plotting as secondary costs.

ii. ```python
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)
```

iii. The notes estimate full conversion at about 25 seconds for 41 sessions and tie runtime to session size, which is most consistent with the preprocessing step dominating.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunity is the per-trial slicing loop, which could be replaced with reshaping full-session arrays into trial blocks. Optional plotting also repeats per session.

ii. ```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes never filled Step 6, but the code structure makes the trial-building loop the obvious Python-level inefficiency.

## 6-c. What processing does the code repeat multiple times?

i. The script repeats the same session processing pipeline for every session: loading arrays, repairing motion-energy lengths, computing baseline-corrected fluorescence, binning, discretizing, and building trial lists. It also recomputes the Torch device inside `compute_dfof` every session.

ii. ```python
for subj in subjects_to_process:
    for i, session in enumerate(sessions):
        neural_trials, input_trials, output_trials, n_neurons = process_session(
            subj, session, show_processing=(show_processing and session_count < 2),
            ax_list=fig_ax
        )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

iii. The notes describe the intended workflow as a single repeated per-session pipeline and provide per-session runtime estimates.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Optional plotting work is not used by the saved dataset. The code also truncates tail frames that do not fit exact bin or trial boundaries, so some processed samples can be discarded before saving.

ii. ```python
if show_processing and ax_list is not None:
    plot_processing(ax_list, subj, session, F, Fneu, dfof, me, me_binned,
                   me_discrete, dfof_binned, n_frames)

n_bins = n_frames // bin_size
truncated = data[..., :n_bins * bin_size]

n_trials = n_total_bins // TRIAL_BINS
```

iii. The notes emphasize plotting as a validation aid rather than part of the output dataset and acknowledge trialization based on exact block lengths.
