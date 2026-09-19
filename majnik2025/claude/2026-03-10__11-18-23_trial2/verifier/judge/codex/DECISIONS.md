# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of six subjects, iterates over each subject's session directories, and for each session loads Suite2p fluorescence files plus motion-energy files before later splitting the continuous recordings into trials. It does not discover subjects dynamically from the filesystem.

ii. <Code snippets>

```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
    ...
    sessions = get_sessions(subject_dir)
    ...
    for sess_dir in sessions:
        dff_binned, me_binned, n_neurons, n_frames_raw = process_session(
            sess_dir, device=device
        )
```

```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 5, the AI says the dataset has exactly six mouse folders (`jm031`-`jm046`) and plans to use all available data from those mice and their sessions.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the fixed `SUBJECTS` list in alphabetical order rather than by scanning directories matching a pattern.

ii. <Code snippets>

```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
'subjects': SUBJECTS if not sample else SUBJECTS[:1],
```

iii. The justification in `CONVERSION_NOTES.md` Step 2 is that the dataset contains six known mice with those IDs, so the AI treated that list as the subject set.

## 1-c. How are the data split into sessions?

i. Each subject is split into sessions by listing all non-hidden subdirectories under that subject folder and sorting them alphabetically.

ii. <Code snippets>

```python
def get_sessions(subject_dir):
    sessions = []
    for d in sorted(os.listdir(subject_dir)):
        full = os.path.join(subject_dir, d)
        if os.path.isdir(full) and not d.startswith('.'):
            sessions.append(full)
    return sessions
```

iii. The AI's notes describe each `YYYY-MM-DD_a` directory as one recording session and report the per-subject session counts found in the filesystem.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous and splits each session into consecutive non-overlapping 2-minute trials after temporal binning. This yields 360 binned timepoints per trial.

ii. <Code snippets>

```python
TRIAL_DURATION_S = 120.0  # 2 minutes per trial
...
bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360
...
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly justifies this by citing the paper's decoder cross-validation setup: "consecutive 2 minute blocks of the recording."

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial-level quality-control filtering is applied. The code keeps every full-length trial produced by the session split.

ii. <Code snippets>

```python
n_trials = n_bins // bins_per_trial

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. In `CONVERSION_NOTES.md` Step 3, the AI notes "No explicit trial curation mentioned - continuous recording," and carries that assumption into the conversion script.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` output is derived from Suite2p fluorescence traces `F.npy` and neuropil traces `Fneu.npy`.

ii. <Code snippets>

```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. In `CONVERSION_NOTES.md` Step 5, the variable map states that `F.npy` and `Fneu.npy` are the source for the neural data.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil subtraction, estimates a baseline with Suite2p's `preprocess`, reconstructs the baseline, and then computes explicit dF/F as `(F_corr - baseline) / baseline`. It then bins the result by averaging groups of 10 frames.

ii. <Code snippets>

```python
F_corr = F - NEUCOEFF * Fneu
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)
baseline = F_corr - F_subtracted
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe
```

```python
dff_binned = bin_traces(dff, BIN_SIZE)
```

iii. The AI justifies this repeatedly in `CONVERSION_NOTES.md` Step 3, Step 5, and Step 10, where it says the paper used "baseline corrected fluorescence traces as our dF/F" and interprets that as an instruction to compute explicit dF/F from the inferred baseline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural filtering is done in the conversion script. All rows in `F.npy` are retained, based on the AI's view that the provided Track2p outputs were already pre-filtered to tracked cells.

ii. <Code snippets>

```python
n_neurons, n_frames = F.shape
...
dff = compute_dff(F, Fneu, device=device)
...
brain_region_idx_list.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The justification in `CONVERSION_NOTES.md` Step 1 and Step 4 is that `iscell.npy` already marks all provided ROIs as cells and the dataset already contains tracked neurons only, so no extra curation is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns the neural data to the start of the recording session rather than to a behavioral event. Trials are consecutive chunks cut from a continuous recording.

ii. <Code snippets>

```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': None,
    ...
}
```

```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
```

iii. The AI's notes say the behavior is spontaneous, the recording is continuous, and motion energy is already synchronized to the microscope, so no event-based alignment is required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 10 imaging frames at 30 Hz, i.e. 333.33 ms per bin. The code averages non-overlapping groups of 10 frames for both neural and motion-energy traces.

ii. <Code snippets>

```python
BIN_SIZE = 10
FS = 30.0
...
return trimmed.reshape(n_features, n_bins, bin_size).mean(axis=2)
```

```python
time_bin_ms = BIN_SIZE / FS * 1000  # ms
...
'time_bin_size': time_bin_ms,
```

iii. The AI cites the methods text in `CONVERSION_NOTES.md` Step 3: decoding analyses "averag[ed] in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time signal is not loaded from a raw timestamp variable. It is synthesized from the within-trial bin index and the known frame/bin durations.

ii. <Code snippets>

```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS  # seconds per bin
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly maps `Time index` to the decoder input and describes it as "Time in seconds from start of trial."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code computes time as a simple arithmetic ramp, `0, 1, 2, ...` multiplied by `10 / 30` seconds, and recreates that ramp independently for every trial.

ii. <Code snippets>

```python
time_bin_s = BIN_SIZE / FS  # seconds per bin
return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The AI's notes justify this as a derived contextual variable rather than a measured signal, and the sample-validation section reports an input range of `[0.0, 119.7] seconds`, confirming the trial-local reset.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. For each trial, the code creates a 1-by-T time array with exactly the same number of bins as the trial's neural matrix. The alignment is therefore by matching trial length, but the clock resets to zero at each trial boundary.

ii. <Code snippets>

```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
```

iii. The AI's own checks in `CONVERSION_NOTES.md` Step 10 validate `arange * 10/30`, again showing that the alignment is trial-local rather than continuous across a session.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion-energy outputs from `motion_energy_glob.npy` and also loads `tstamps.npy` as auxiliary information for handling frame mismatches.

ii. <Code snippets>

```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

iii. In `CONVERSION_NOTES.md` Step 5 and the trajectory, the AI says missing camera frames should be handled by interpolation using timestamps because the video is synchronized to the microscope.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code linearly interpolates motion energy to the neural frame count when lengths differ, using uniformly spaced source indices, then bins the trace into non-overlapping 10-frame averages.

ii. <Code snippets>

```python
if n_me < n_neural_frames:
    neural_indices = np.arange(n_neural_frames)
    me_indices = np.linspace(0, n_neural_frames - 1, n_me)
    me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
    return me_interp
```

```python
me_binned = bin_traces(me_interp, BIN_SIZE)
```

iii. The AI justifies this in `CONVERSION_NOTES.md` Step 5 and Step 10 as interpolation over missing camera frames, based on the dataset README note that missing video frames can be interpolated.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI concatenates binned motion-energy values across all processed sessions, computes one global set of five equal-percentile bin edges, and discretizes every trial using those global edges.

ii. <Code snippets>

```python
all_me_concat = np.concatenate(all_me_binned_values)
bin_edges = compute_me_percentile_bins(all_me_concat, N_OUTPUT_BINS)
```

```python
def discretize_me(me_values, bin_edges):
    labels = np.digitize(me_values, bin_edges[1:-1])
    labels = np.clip(labels, 0, n_bins - 1)
    return labels.astype(np.int64)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly lists "Motion energy normalization: Compute percentile bins globally across all sessions" and later treats the resulting global 20/20/20/20/20 distribution as a sanity check.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI assumes the behavioral video is synchronized to calcium imaging, rescales/interpolates motion energy to the neural frame count if needed, bins both streams with the same 10-frame window, and then slices them into the same trial boundaries.

ii. <Code snippets>

```python
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
...
dff_binned = bin_traces(dff, BIN_SIZE)
me_binned = bin_traces(me_interp, BIN_SIZE)
```

```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. The AI's notes repeatedly justify this with "camera triggered by microscope" and "ME is already synced to neural," with interpolation added only to repair missing frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles shorter motion-energy arrays by interpolating them to the neural frame count, truncates motion-energy arrays that are longer than neural recordings, discards partial trailing bins/trials created by integer division, and clips inferred baselines to `1e-6` before dividing to form dF/F.

ii. <Code snippets>

```python
if n_me < n_neural_frames:
    ...
    return me_interp
else:
    return me[:n_neural_frames].astype(np.float64)
```

```python
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe
```

```python
n_bins = n_frames // bin_size
trimmed = data[:, :n_bins * bin_size]
...
n_trials = n_bins // bins_per_trial
```

iii. The AI's notes and trajectory justify the motion-energy interpolation from the dataset README and treat partial-bin dropping as acceptable because trial counts are based on full fixed-length blocks.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive step is neural preprocessing, especially the Suite2p baseline/dF/F computation performed for every session. The script also performs a full first pass over all sessions before the second pass that splits trials and discretizes outputs.

ii. <Code snippets>

```python
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)
```

```python
for subj_idx, subject in enumerate(subjects_to_process):
    ...
    for sess_dir in sessions:
        dff_binned, me_binned, n_neurons, n_frames_raw = process_session(
            sess_dir, device=device
        )
```

iii. In `CONVERSION_NOTES.md` Step 6 and the trajectory, the AI describes Suite2p preprocessing as the central transformation and reports roughly one second per session.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious Python loops are the per-trial split loop and the per-trial loop that separately creates time inputs and discretizes motion energy for each trial. Binning itself is already vectorized with reshape-and-mean.

ii. <Code snippets>

```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
    me_disc = discretize_me(me_t, bin_edges)
    output_trials.append(me_disc.reshape(1, -1))
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI specifically calls out "Vectorized binning (reshape + mean)" as an optimization, implying the remaining trial loops are the main non-vectorized pieces.

## 6-c. What processing does the code repeat multiple times?

i. The code uses a two-pass design over the session data: first it preprocesses every session and caches binned neural/motion-energy traces so it can compute global motion-energy bin edges, then it loops over all cached sessions again to split trials and build the final output structures. Optional plotting also reloads raw files that were already read once.

ii. <Code snippets>

```python
session_data = []  # (subject_idx, session_dir, dff_binned, me_binned, n_neurons)
all_me_binned_values = []
...
session_data.append((subj_idx, sess_dir, dff_binned, me_binned, n_neurons))
all_me_binned_values.append(me_binned)
```

```python
all_me_concat = np.concatenate(all_me_binned_values)
bin_edges = compute_me_percentile_bins(all_me_concat, N_OUTPUT_BINS)
...
for subj_idx, sess_dir, dff_binned, me_binned, n_neurons in session_data:
    neural_trials, me_trials = split_into_trials(dff_binned, me_binned)
```

```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
```

iii. The AI's justification for the two-pass structure appears in `CONVERSION_NOTES.md` Step 5: it wanted global percentile edges for motion energy, which forced it to collect all binned motion-energy values before final discretization.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `tstamps.npy` but never actually uses the timestamp values, computes and stores human-readable output-bin labels that are not needed for decoder training, and optionally generates diagnostic plots by reloading raw files even though those plots are not part of the converted dataset.

ii. <Code snippets>

```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
...
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
```

```python
output_value_labels = []
for i in range(N_OUTPUT_BINS):
    ...
    output_value_labels.append(...)
```

```python
if show_processing and plot_count < 2:
    plot_processing(sess_dir, dff_binned, me_binned, me_disc_trials,
                  neural_trials, dff_binned.shape[1] * BIN_SIZE, plot_path)
```

iii. The AI's notes justify the plotting as a sanity check and the extra metadata as documentation, but they do not affect the actual decoder input/output arrays used downstream.
