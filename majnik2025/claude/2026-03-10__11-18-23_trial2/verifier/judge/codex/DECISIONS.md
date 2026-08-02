# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the subject list as `['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`, iterates over each subject directory, discovers sessions by listing non-hidden subdirectories, and for each session loads `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `tstamps.npy`. Trials are not loaded directly; each session is processed first and then split into trial blocks later.

ii.
```python
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

iii. In `CONVERSION_NOTES.md`, the AI says the dataset has 6 mouse subjects and describes the per-session `suite2p/plane0` and `move_deve` files. It justified the subject/session structure from dataset exploration, but chose a fixed subject list instead of dynamically discovering `jm*` directories.

## 1-b. How are the data split into subjects?

i. Subjects are split using the hard-coded `SUBJECTS` list, in alphabetical order. The code does not scan the data directory for matching subject folders.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

subjects_to_process = SUBJECTS
if sample:
    subjects_to_process = SUBJECTS[:1]

for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
```

iii. The notes say there are exactly 6 mice and list them explicitly. The AI appears to have inferred that this fixed list was safe because the dataset exploration showed those exact folders.

## 1-c. How are the data split into sessions?

i. Each session is a non-hidden subdirectory inside a subject folder, sorted lexicographically. One session corresponds to one daily recording directory such as `2023-10-18_a`.

ii.
```python
def get_sessions(subject_dir):
    sessions = []
    for d in sorted(os.listdir(subject_dir)):
        full = os.path.join(subject_dir, d)
        if os.path.isdir(full) and not d.startswith('.'):
            sessions.append(full)
    return sessions
```

iii. `CONVERSION_NOTES.md` states that each subject contains daily session folders. The AI justified this from directory inspection and from the paper’s description of daily longitudinal recordings.

## 1-d. How are the data split into trials?

i. The AI treats each session as continuous data, bins it first into 10-frame averages, and then splits the binned session into consecutive 2-minute blocks. With 30 Hz acquisition and 10-frame bins, each trial is `120 * 30 / 10 = 360` time bins.

ii.
```python
BIN_SIZE = 10
TRIAL_DURATION_S = 120.0

dff_binned = bin_traces(dff, BIN_SIZE)
me_binned = bin_traces(me_interp, BIN_SIZE)

def split_into_trials(dff_binned, me_binned):
    bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360
    n_bins = dff_binned.shape[1]
    n_trials = n_bins // bins_per_trial

    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])
```

iii. The AI explicitly justifies this in `CONVERSION_NOTES.md` as matching the paper’s decoder cross-validation setup: “2-minute blocks” and “bins of 10 consecutive timestamps.” It treated the decoder-analysis settings from the paper as the conversion trial definition.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial quality-control filter. Trials are included if they exist after integer division into 2-minute blocks. Any leftover timepoints that do not fill a complete 10-frame bin or a complete 2-minute block are silently discarded by integer division and array trimming.

ii.
```python
def bin_traces(data, bin_size):
    n_bins = n_frames // bin_size
    trimmed = data[:, :n_bins * bin_size]
    return trimmed.reshape(n_features, n_bins, bin_size).mean(axis=2)

def split_into_trials(dff_binned, me_binned):
    n_trials = n_bins // bins_per_trial
    for t in range(n_trials):
        ...
```

iii. The notes say there was “No explicit trial curation mentioned” because the recordings are continuous. The AI therefore did not introduce any behavioral or quality-based trial rejection rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the Suite2p fluorescence arrays `F.npy` and `Fneu.npy` from `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The notes identify `F.npy` and `Fneu.npy` as the source variables and cite Suite2p defaults from `ops.npy` as justification for how to process them.

## 2-b. How is the `neural` data processed?

i. The AI computes a dF/F-like signal. It subtracts neuropil (`F - 0.7 * Fneu`), calls Suite2p `preprocess` to estimate and subtract baseline, reconstructs the baseline, divides by baseline to form dF/F, casts to `float32`, and then averages into non-overlapping 10-frame bins.

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

iii. The AI justified this with the paper text “baseline corrected fluorescence traces as our dF/F” and with Suite2p default parameters recovered from `ops.npy`. In the notes it repeatedly states that dF/F plus 10-frame averaging matches the paper’s decoder analysis.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering is applied during conversion. All rows in `F.npy` are carried forward.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
n_neurons, n_frames = F.shape
...
brain_region_idx_list.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The AI’s notes say the provided data already contains only tracked neurons and that `iscell.npy` is effectively pre-filtered (`iscell[:,0] == 1`). It therefore decided not to apply any extra `iscell` or quality filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI says the alignment event is the “Start of recording session,” but in practice it slices later 2-minute trial blocks from the session and does not realign each trial to a distinct event. So the code treats the recording as continuous and merely partitions it into consecutive windows.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))

'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The notes justify this by saying motion energy is already synchronized to neural recording and that there is no event structure beyond continuous recording. The metadata wording suggests session-start alignment, but the per-trial representation is just windowing of continuous time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10-frame bins at 30 Hz, giving a time bin size of `10 / 30 = 0.333...` s, or `333.33` ms. Yes, explicit temporal rebinning is applied to both neural and motion-energy traces.

ii.
```python
BIN_SIZE = 10
FS = 30.0

dff_binned = bin_traces(dff, BIN_SIZE)
me_binned = bin_traces(me_interp, BIN_SIZE)

time_bin_ms = BIN_SIZE / FS * 1000
```

iii. The AI justifies this directly from the paper’s analysis text about averaging in bins of 10 consecutive timestamps and records the 333.33 ms value in `CONVERSION_NOTES.md`.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from a raw timestamp array. It is synthesized from the number of time bins and the assumed 30 Hz frame rate after 10-frame binning.

ii.
```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The notes map “Time index” to the decoder input and describe it as “Time in seconds from start of trial,” computed from bin index and frame rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI creates a simple ramp: `0, 1, 2, ...` time bins multiplied by `10/30` seconds per bin, then reshapes it to `(1, n_timebins)`. No raw timestamp correction is applied, and the value resets at every trial.

ii.
```python
time_bin_s = BIN_SIZE / FS  # seconds per bin
return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The notes justify this as the decoder input requested by the instructions and explicitly describe the value as within-trial time.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. For each trial, the synthesized time vector has the same number of bins as the neural matrix after binning and trial splitting. Alignment is therefore by shared trial index and time-bin index, but the elapsed-time signal resets to zero at the start of every trial rather than preserving session-wide elapsed time.

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
```

iii. The AI’s notes say the input is “Time in seconds from start of trial.” That is the rationale it used for alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived primarily from `move_deve/motion_energy_glob.npy`. The code also loads `tstamps.npy` during processing, but does not actually use the timestamp values inside the interpolation routine.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
...
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
```

iii. The notes say missing-frame interpolation should use timestamps, based on the data README statement that missing frames can be identified from `tstamps.npy` or `interframe_int.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly resamples motion energy to the neural frame count if lengths differ, averages it into 10-frame bins, computes global percentile edges across all binned sessions, and discretizes each binned trace into 5 categories. Despite claiming normalization in the notes, the code does not standardize by per-session standard deviation.

ii.
```python
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
me_binned = bin_traces(me_interp, BIN_SIZE)

all_me_concat = np.concatenate(all_me_binned_values)
bin_edges = compute_me_percentile_bins(all_me_concat, N_OUTPUT_BINS)

me_disc = discretize_me(me_t, bin_edges)
output_trials.append(me_disc.reshape(1, -1))
```

iii. `CONVERSION_NOTES.md` says motion energy should be “Interpolate missing frames, bin by 10, normalize, discretize into 5 equal-percentile bins.” The trajectory and notes tie that choice to the paper’s decoder-analysis description and the README note about missing camera frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes 5 equal-percentile bins globally across all binned motion-energy values from all sessions, then applies `np.digitize` to assign category labels `0` through `4`.

ii.
```python
def compute_me_percentile_bins(all_me_values, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_me_values, percentiles)
    return bin_edges

labels = np.digitize(me_values, bin_edges[1:-1])
labels = np.clip(labels, 0, n_bins - 1)
```

iii. The AI cites the task instruction “normalized and discretized into five equal-percentile bins” and records in the notes that global percentile bins across all data are intended to give a balanced distribution.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy by forcing it to the same frame count as the neural recording through interpolation/truncation, then binning both streams identically and slicing matching trial windows. However, the interpolation ignores actual dropped-frame positions and does not use `tstamps` despite loading it.

ii.
```python
def interpolate_motion_energy(me, n_neural_frames, tstamps):
    if n_me < n_neural_frames:
        neural_indices = np.arange(n_neural_frames)
        me_indices = np.linspace(0, n_neural_frames - 1, n_me)
        me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
        return me_interp
    else:
        return me[:n_neural_frames].astype(np.float64)

me_interp = interpolate_motion_energy(me, n_frames, tstamps)
me_binned = bin_traces(me_interp, BIN_SIZE)
```

iii. The notes justify this by saying the camera was triggered by the microscope and that some sessions have missing video frames that should be interpolated to recover alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or mismatched motion-energy frames by interpolating or truncating to match the neural frame count. It also drops incomplete trailing data through integer division during binning and trial splitting. It does not keep explicit missing-value markers or assert that mismatches were repaired in a specific way.

ii.
```python
if n_me == n_neural_frames:
    return me.astype(np.float64)
...
if n_me < n_neural_frames:
    me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
    return me_interp
else:
    return me[:n_neural_frames].astype(np.float64)

n_bins = n_frames // bin_size
trimmed = data[:, :n_bins * bin_size]

n_trials = n_bins // bins_per_trial
```

iii. The notes repeatedly mention missing camera frames and say interpolation is the planned remedy. They also note that partial bins and partial trials are discarded through integer division.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive steps are session-wise neural preprocessing and the two full passes over all sessions. In particular, `compute_dff` calls Suite2p `preprocess` for every session, which is the heaviest numerical step; file loading and optional plotting add additional overhead.

ii.
```python
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)

for subj_idx, subject in enumerate(subjects_to_process):
    ...
    dff_binned, me_binned, n_neurons, n_frames_raw = process_session(
        sess_dir, device=device
    )
```

iii. `CONVERSION_NOTES.md` says the code uses Suite2p preprocessing with GPU acceleration and reports roughly 1 second per session. That implies the Suite2p baseline step is the dominant cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial append loops in `split_into_trials` and in the second pass over `session_data` could be reduced by reshaping or batching when all trial lengths are equal. The subject/session loops are structurally necessary, but trial construction is still list-based and not fully vectorized.

ii.
```python
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
    output_trials.append(me_disc.reshape(1, -1))
```

iii. The notes emphasize that the AI already added vectorized binning (`reshape + mean`) and used interpolation instead of repeated insertion. It does not explicitly call out the remaining per-trial loops as an inefficiency, but those are the main loops still amenable to vectorization.

## 6-c. What processing does the code repeat multiple times?

i. The code makes two passes over the processed sessions: one pass to compute and store binned session data and collect all motion-energy values, and a second pass to split sessions into trials and discretize outputs. If plotting is enabled, it also reloads raw `F`, `Fneu`, and motion-energy files for visualization.

ii.
```python
session_data = []  # (subject_idx, session_dir, dff_binned, me_binned, n_neurons)
all_me_binned_values = []
...
session_data.append((subj_idx, sess_dir, dff_binned, me_binned, n_neurons))
all_me_binned_values.append(me_binned)
...
for subj_idx, sess_dir, dff_binned, me_binned, n_neurons in session_data:
    neural_trials, me_trials = split_into_trials(dff_binned, me_binned)
```

iii. The AI justifies the two-pass structure because global percentile edges for motion energy must be known before trial outputs can be discretized. The plotting rereads are only for optional visual checks.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `tstamps.npy` but does not use it, computes `session_count` and `n_frames_raw` without using them downstream, imports plotting libraries and contains a large optional plotting path unrelated to the saved dataset, and generates verbose output labels that are not needed for decoder training. The optional plots also reread raw files purely for inspection.

ii.
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
...
session_count = 0
...
def plot_processing(...):
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
```

iii. The notes say plotting and extra sanity checks were added for validation, not because the final decoder format requires them. The unused timestamp load appears to come from the AI’s intended but not fully implemented timestamp-based interpolation idea.
