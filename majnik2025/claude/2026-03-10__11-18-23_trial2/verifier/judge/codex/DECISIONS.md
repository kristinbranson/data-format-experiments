# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six subject IDs in `SUBJECTS`, iterates those subject folders under `data/`, discovers each subject's session subdirectories with `get_sessions()`, and for each session loads fluorescence (`F.npy`, `Fneu.npy`) plus motion-energy data (`motion_energy_glob.npy`, `tstamps.npy`). It processes sessions in a first pass, stores the processed session outputs in memory, and only later splits them into trials.

ii.
```python
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
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

iii. In `CONVERSION_NOTES.md`, the AI says the dataset contains six mice (`jm031`-`jm046`) and that each daily session directory contains Suite2p outputs plus motion-energy files. It justifies using all available subject/session folders and handling motion-energy mismatches during processing.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are not inferred from the directory tree at runtime. They are defined by a hard-coded ordered list of mouse IDs, and each session inherits the `subject_idx` corresponding to the loop index over that list.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

subjects_to_process = SUBJECTS
for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
    ...
    subject_idx_list.append(subj_idx)
```

iii. The notes explicitly identify the six mice and describe them as "Mouse A-F" in alphabetical order, which is the rationale for fixing the subject order up front rather than scanning for `jm*` directories.

## 1-c. How are the data split into sessions?

i. Each session is a non-hidden subdirectory inside a subject directory, sorted lexicographically. In sample mode, only the first two sessions of the first subject are kept.

ii.
```python
def get_sessions(subject_dir):
    sessions = []
    for d in sorted(os.listdir(subject_dir)):
        full = os.path.join(subject_dir, d)
        if os.path.isdir(full) and not d.startswith('.'):
            sessions.append(full)
    return sessions

sessions = get_sessions(subject_dir)
if sample:
    sessions = sessions[:2]
```

iii. `CONVERSION_NOTES.md` says each session is a daily recording directory named like `YYYY-MM-DD_a`, so sorting the subdirectories gives a deterministic session order.

## 1-d. How are the data split into trials?

i. The AI treats each session as a continuous recording, first averages into 10-frame bins, then divides the binned session into consecutive non-overlapping 2-minute blocks. Each trial therefore has 360 binned timepoints (`120 s * 30 Hz / 10`), and any leftover tail shorter than a full block is discarded by integer division.

ii.
```python
BIN_SIZE = 10
TRIAL_DURATION_S = 120.0

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

iii. The AI repeatedly justifies this with the paper's decoding description: the notes say the paper used "consecutive 2 minute blocks" for cross-validation, so it reuses that block structure as the trial definition.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering. The only practical filtering is that incomplete trailing bins or trailing partial 2-minute blocks are dropped implicitly when binning and trial-splitting use floor division.

ii.
```python
if data.ndim == 1:
    n_bins = n_frames // bin_size
    trimmed = data[:n_bins * bin_size]
    return trimmed.reshape(n_bins, bin_size).mean(axis=1)

...

n_trials = n_bins // bins_per_trial
for t in range(n_trials):
    ...
```

iii. The notes say "No explicit trial curation mentioned - continuous recording" and later list "Partial bins at end of sessions: discarded" as an edge-case handling rule rather than a trial-quality screen.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from Suite2p fluorescence arrays `F.npy` and `Fneu.npy` in `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The notes describe the dataset as already being in Suite2p format and map `F.npy` and `Fneu.npy` directly to the converted neural output.

## 2-b. How is the `neural` data processed?

i. The AI computes a dF/F-like signal rather than keeping Suite2p's baseline-corrected fluorescence directly. It subtracts neuropil (`F - 0.7 * Fneu`), calls Suite2p `preprocess()` with `maximin` baseline settings, reconstructs the baseline by subtraction, divides by the baseline to form dF/F, casts to `float32`, and then averages the result in non-overlapping 10-frame bins.

ii.
```python
F_corr = F - NEUCOEFF * Fneu
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)
baseline = F_corr - F_subtracted
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe

dff_binned = bin_traces(dff, BIN_SIZE)
```

iii. The notes say the paper used "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and therefore justify neuropil subtraction, `maximin` baseline correction, and 10-frame averaging as matching the decoding description in the paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no additional neuron-level filtering during conversion. It assumes the provided arrays already contain only tracked, valid cells and does not load or threshold `iscell.npy`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu, device=device)
```

iii. The notes explicitly state that the Track2p outputs are already restricted to neurons tracked across all days and that `iscell.npy` contains only ones, so the AI decided no extra neural QC was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of the recording session, not to any within-session behavioral event. Trials are consecutive blocks cut out of the session timeline after temporal binning, and the metadata declares the alignment event as the recording start.

ii.
```python
neural_trials.append(dff_binned[:, start:end].astype(np.float32))

'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': None,
    ...
}
```

iii. The notes say the recording is continuous, there is no explicit stimulus/event structure, and the behavioral video is synchronized to the microscope, so session start was used as the shared alignment point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at 30 Hz, so each timepoint is `10 / 30 = 0.333...` seconds (`333.33 ms`). Yes, the AI rebins the neural data before trial splitting.

ii.
```python
BIN_SIZE = 10
FS = 30.0

dff_binned = bin_traces(dff, BIN_SIZE)

time_bin_ms = BIN_SIZE / FS * 1000  # ms
```

iii. The notes justify this by quoting the paper's decoder description: "averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time is not derived from a raw timestamp file. It is synthesized from the number of binned timepoints plus the constants `BIN_SIZE` and `FS`.

ii.
```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS  # seconds per bin
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The notes' mapping table says `Time index -> input[0]`, with seconds computed as `(bin_index * 10 / 30)`.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the AI creates a regularly spaced time vector with one value per binned sample, using `np.arange(n_timebins) * (10 / 30)` seconds. This resets to zero at the start of every trial and reflects time from trial start, not absolute time from session start.

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
```

iii. Both the docstring in `make_time_input()` and the notes describe this as "Time in seconds from start of trial."

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned one-to-one with the already binned neural data. For each neural trial, the code uses that trial's number of binned columns to generate an equally long time vector, so the input and neural arrays are synchronized after binning and 2-minute splitting.

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
```

iii. The notes record an "Input check" verifying `arange * 10/30`, which is the AI's justification that the generated time series matches the binned neural timeline.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output motion-energy signal is derived from `motion_energy_glob.npy`. The AI also loads `tstamps.npy` and passes it into the interpolation helper when motion-energy and neural frame counts differ.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
...
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
```

iii. The notes describe `motion_energy_glob.npy` as the behavioral signal and treat timestamps/interframe information as the way to repair missing camera frames so motion energy can be aligned with neural data.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. In code, the AI handles frame-count mismatches by linearly interpolating the motion-energy vector onto an evenly spaced index grid of neural-frame length, truncates if motion energy is longer than neural data, and then averages the repaired signal in 10-frame bins. It does not divide by the session standard deviation before binning or discretization, even though the notes claim such normalization was intended.

ii.
```python
def interpolate_motion_energy(me, n_neural_frames, tstamps):
    n_me = len(me)
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

iii. The notes justify this with the paper's camera-synchronization story and say missing frames should be interpolated, binned in 10-frame windows, and normalized before discretization. The implemented code only carries out the interpolation and binning parts.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI concatenates motion-energy values from all processed sessions, computes global percentile edges for 5 equal-frequency bins, and discretizes each timepoint with `np.digitize`. It also creates human-readable labels from the resulting edges.

ii.
```python
def compute_me_percentile_bins(all_me_values, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_me_values, percentiles)
    return bin_edges

def discretize_me(me_values, bin_edges):
    labels = np.digitize(me_values, bin_edges[1:-1])
    labels = np.clip(labels, 0, n_bins - 1)
    return labels.astype(np.int64)
```

iii. The notes repeatedly describe the target as "5 equal-percentile bins" and report an approximately 20%/20%/20%/20%/20% global output distribution as the sanity check for this choice.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI assumes the behavioral video is synchronized to imaging, repairs any frame-count mismatch by interpolating motion energy to the neural frame count, then bins motion energy with the same 10-frame windows and cuts trials with the same 2-minute boundaries used for neural data. Although `tstamps` is passed into the helper, the actual interpolation uses only array lengths, not the timestamp values themselves.

ii.
```python
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
me_binned = bin_traces(me_interp, BIN_SIZE)

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. The notes say the camera was triggered by the microscope, so motion energy should be 1:1 with neural frames apart from occasional missing video frames that can be interpolated away.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles short motion-energy recordings by interpolating them up to the neural-frame count and would handle long motion-energy recordings by truncating them. It silently drops incomplete trailing temporal bins and incomplete trailing 2-minute trials via floor division. It also warns and skips a subject if the expected subject directory is absent.

ii.
```python
if n_me < n_neural_frames:
    me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
    return me_interp
else:
    return me[:n_neural_frames].astype(np.float64)

trimmed = data[:n_bins * bin_size]
...
n_trials = n_bins // bins_per_trial

if not os.path.isdir(subject_dir):
    print(f"Warning: Subject directory not found: {subject_dir}")
    continue
```

iii. The notes describe missing motion-energy frames as an expected edge case and treat discarded partial bins/blocks as acceptable cleanup. They do not describe additional error checks beyond that.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is the per-session neural preprocessing in `compute_dff()`, especially Suite2p baseline estimation over all neurons and frames. If enabled, the plotting path adds extra I/O and figure-generation overhead.

ii.
```python
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)

if show_processing and plot_count < 2:
    plot_processing(...)
```

iii. The notes say the script uses Suite2p preprocessing with GPU acceleration and estimate roughly one second per session, which implies the baseline-correction stage is the main computational cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the main temporal binning with reshape-and-mean and uses `np.interp` for motion-energy interpolation. The remaining obvious Python loops are the per-trial slicing loop in `split_into_trials()` and the per-trial loop that recreates time inputs and discretized outputs in the second pass.

ii.
```python
return trimmed.reshape(n_features, n_bins, bin_size).mean(axis=2)

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])

for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    me_disc = discretize_me(me_t, bin_edges)
```

iii. The notes explicitly claim "Vectorized binning (reshape + mean)" as an implementation improvement. They do not discuss the remaining trial-level loops, so that part is inferred directly from the code.

## 6-c. What processing does the code repeat multiple times?

i. The conversion is structured as two passes over session outputs: first it preprocesses every session and collects all motion-energy values, then it loops over all saved session outputs again to split trials and discretize motion energy. If plotting is enabled, `plot_processing()` also reloads raw `F`, `Fneu`, and motion-energy arrays that were already loaded once in `process_session()`.

ii.
```python
session_data = []
all_me_binned_values = []
...
for sess_dir in sessions:
    dff_binned, me_binned, n_neurons, n_frames_raw = process_session(...)
    session_data.append((subj_idx, sess_dir, dff_binned, me_binned, n_neurons))
    all_me_binned_values.append(me_binned)

for subj_idx, sess_dir, dff_binned, me_binned, n_neurons in session_data:
    neural_trials, me_trials = split_into_trials(dff_binned, me_binned)

F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
```

iii. The notes justify the two-pass structure indirectly by saying the percentile bin edges are computed globally across all sessions. The extra reloads in `plot_processing()` are not explicitly justified there; they are simply part of the visualization path.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The optional visualization path does the clearest unnecessary work: it reloads raw arrays, generates multi-panel plots, and computes plot-only diagnostics that are not used by the saved decoder dataset. The function also returns and prints some reporting-only information such as `n_frames_raw`, file-size summaries, and hand-built output-value labels that matter for documentation but not for downstream decoding.

ii.
```python
def plot_processing(...):
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    ...

return dff_binned, me_binned, n_neurons, n_frames

print(f"File size: {file_size:.1f} MB")
print(f"Total time: {total_time:.1f}s")
```

iii. The notes explicitly mention saved processing plots as a validation aid. That makes the extra work understandable, but it is not needed for the converted pickle consumed by the decoder.
