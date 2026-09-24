# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six subject IDs, finds every non-hidden directory under each subject as a session, and loads each session's `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `tstamps.npy`. Full mode processes all discovered sessions; sample mode restricts processing to the first two sessions of the first subject.

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

iii. The notes say the explored dataset contains exactly these six mice and 41 sessions, with Suite2p arrays and motion-energy files in a consistent directory layout. They report using all available data, including sessions longer than the paper's nominal 20 minutes.

## 1-b. How are the data split into subjects?

i. Subject membership is determined by the hard-coded `SUBJECTS` list. Each subject directory is processed in list order, and that index is attached to every session from the directory.

ii.
```python
for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
    ...
    session_data.append((subj_idx, sess_dir, dff_binned, me_binned, n_neurons))
...
subject_idx_list.append(subj_idx)
```

iii. The AI identified each `jm*` folder as a distinct mouse and described the alphabetical order as mice A–F. It validated six subjects and the expected neuron counts.

## 1-c. How are the data split into sessions?

i. Every sorted, non-hidden subdirectory of a subject directory is one session. Each resulting session becomes one outer-list element in `neural`, `input`, and `output`.

ii.
```python
sessions = get_sessions(subject_dir)
for sess_dir in sessions:
    dff_binned, me_binned, n_neurons, n_frames_raw = process_session(
        sess_dir, device=device
    )
...
all_neural.append(neural_trials)
all_input.append(input_trials)
all_output.append(output_trials)
```

iii. The notes state that each dated directory contains one daily recording, yielding 7, 7, 7, 7, 6, and 7 sessions for the six mice (41 total).

## 1-d. How are the data split into trials?

i. The AI splits each continuous session into non-overlapping 120-second blocks (360 post-binning samples), dropping any incomplete final block. This follows the paper's decoder cross-validation block length, not the task's explicit instruction to split sessions into 60-second trials.

ii.
```python
TRIAL_DURATION_S = 120.0
bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360
n_trials = n_bins // bins_per_trial
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. The AI justified two-minute trials by the paper statement that cross-validation splits used consecutive two-minute blocks. It overlooked that the conversion task explicitly overrides this by requiring 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filter is applied. All complete 120-second blocks are retained, while only the incomplete tail is discarded.

ii.
```python
n_trials = n_bins // bins_per_trial
for t in range(n_trials):
    ...
```

iii. The notes say the continuous recording has no natural trials and that the paper mentions no explicit trial curation. Therefore, the AI retained all complete blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `F.npy` (ROI fluorescence) and `Fneu.npy` (neuropil fluorescence) in each session's `suite2p/plane0` directory.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
dff = compute_dff(F, Fneu, device=device)
```

iii. The AI cites the paper's use of baseline-corrected fluorescence and the Suite2p settings stored in `ops.npy`.

## 2-b. How is the `neural` data processed?

i. The AI subtracts `0.7 * Fneu` from `F`, applies Suite2p maximin preprocessing, reconstructs the baseline as corrected fluorescence minus the preprocessor output, clips that baseline positive, and divides the baseline-subtracted trace by it. It then averages every 10 frames and casts trial arrays to float32.

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
...
dff_binned = bin_traces(dff, BIN_SIZE)
```

iii. The AI interpreted “baseline corrected fluorescence traces as our dF/F” as requiring `(F_corr - F0) / F0`. Its notes claim this matches the paper. The human reference instead uses Suite2p's baseline-subtracted preprocessing output directly, without this additional division.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is performed; every row in `F.npy` is retained.

ii.
```python
n_neurons, n_frames = F.shape
...
brain_region_idx_list.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The AI inspected `iscell.npy`, found all entries marked as cells, and concluded that the supplied Track2p data had already been filtered to neurons tracked across all days.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are consecutive slices beginning at session sample zero; there is no biological-event alignment. Metadata calls the alignment event “Start of recording session,” although later trials begin 120-second increments after that event.

ii.
```python
start = t * bins_per_trial
end = start + bins_per_trial
neural_trials.append(dff_binned[:, start:end].astype(np.float32))
...
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
```

iii. The notes explain that video and imaging are synchronous continuous recordings and that there is no stimulus event, so fixed blocks are taken from session start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI averages non-overlapping groups of 10 samples at 30 Hz for both neural and motion-energy streams, producing 333.33 ms samples. A tail shorter than 10 raw frames is dropped.

ii.
```python
BIN_SIZE = 10
trimmed = data[:, :n_bins * bin_size]
return trimmed.reshape(n_features, n_bins, bin_size).mean(axis=2)
...
time_bin_ms = BIN_SIZE / FS * 1000
```

iii. This is justified by the paper's statement that dF/F and behavioral traces were denoised by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is generated from an integer index and the known 10-frame/30-Hz bin duration, rather than read from a raw timestamp variable.

ii.
```python
time_bin_s = BIN_SIZE / FS
return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The AI says the frame rate is fixed and labels this as elapsed time. Its planning notes explicitly describe it as time from the start of each trial.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For every trial, the AI creates `0, 1/3, 2/3, ...` seconds. The sequence resets to zero at every 120-second trial and therefore is trial-relative, not elapsed time from session/experiment start.

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
```

iii. The docstring and notes interpret the requested input as “Time in seconds from start of trial.” This conflicts with the task's “beginning of the session/experiment” requirement and the human reference's continuous session time.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Each generated time vector has exactly the same number of bins as its neural trial, and corresponding columns are paired. However, because the time vector resets, its absolute temporal alignment is wrong after the first trial.

ii.
```python
n_timebins = neural_t.shape[1]
time_input = make_time_input(n_timebins)
input_trials.append(time_input)
```

iii. The AI validated the within-trial vector against `arange * 10/30`, but did not test the required session-level offset for later trials.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It is derived from each session's precomputed `move_deve/motion_energy_glob.npy`. The AI also loads `tstamps.npy` for handling length mismatches, although its interpolation does not actually use the timestamp values.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
```

iii. The notes describe motion energy as the squared, summed pixel-wise difference between consecutive video frames and cite microscope-triggered 30-Hz video acquisition.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI resamples any short motion-energy vector to the neural length using uniformly spaced synthetic indices, truncates it if too long, averages non-overlapping 10-frame bins, and later discretizes those averages.

ii.
```python
if n_me < n_neural_frames:
    neural_indices = np.arange(n_neural_frames)
    me_indices = np.linspace(0, n_neural_frames - 1, n_me)
    me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
    return me_interp
else:
    return me[:n_neural_frames].astype(np.float64)
...
me_binned = bin_traces(me_interp, BIN_SIZE)
```

iii. The AI intended to interpolate occasional dropped camera frames so the streams match. It recognized that timestamps identify missing frames, but chose uniform stretching rather than inserting values at the actual gaps.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI concatenates binned motion energy from every subject and session, computes global 0/20/40/60/80/100 percentile edges, and applies the same interior edges to every session to produce integer labels 0–4.

ii.
```python
all_me_concat = np.concatenate(all_me_binned_values)
bin_edges = compute_me_percentile_bins(all_me_concat, N_OUTPUT_BINS)
...
labels = np.digitize(me_values, bin_edges[1:-1])
labels = np.clip(labels, 0, n_bins - 1)
```

iii. The AI's notes claim that “global” equal-percentile bins satisfy the task and highlight the globally balanced 20% distribution. The task explicitly requires bins selected per session, as implemented by the human reference.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Before binning, the motion-energy vector is forced to the neural frame count; both streams are then independently averaged in groups of 10 and split with identical trial boundaries. Exact dropped-frame locations are not preserved because short vectors are uniformly stretched.

ii.
```python
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
dff_binned = bin_traces(dff, BIN_SIZE)
me_binned = bin_traces(me_interp, BIN_SIZE)
...
neural_trials, me_trials = split_into_trials(dff_binned, me_binned)
```

iii. The AI correctly notes the intended 1:1 synchronization from microscope triggering and the need to repair missing video frames. Its chosen resampling, however, shifts samples throughout affected sessions rather than only at detected gaps.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion-energy frames cause global linear resampling to the neural length. Excess motion-energy samples are silently truncated. Incomplete final bin groups and incomplete final 120-second trials are dropped. Missing subject directories generate warnings and are skipped.

ii.
```python
if not os.path.isdir(subject_dir):
    print(f"Warning: Subject directory not found: {subject_dir}")
    continue
...
me_indices = np.linspace(0, n_neural_frames - 1, n_me)
...
return me[:n_neural_frames].astype(np.float64)
```

iii. The notes say seven sessions have motion-energy length mismatches and justify interpolation using the dataset README. They report handling partial bins by integer-division truncation.

## 6-a. What are the most time-consuming steps of the code?

i. The full-session Suite2p maximin baseline filtering is the main computation; loading and retaining large arrays and serializing the roughly 395 MB pickle are also substantial. Optional plots reload raw arrays and render large figures.

ii.
```python
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)
...
with open(output_path, 'wb') as f:
    pickle.dump(data, f)
```

iii. The notes identify Suite2p preprocessing with optional GPU acceleration and report about one second per session and about 50 seconds estimated for the full conversion.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial construction and per-trial discretization use Python loops, although session arrays could be trimmed and reshaped into trial dimensions in bulk. Directory/session iteration necessarily remains at a coarse level. The binning operations themselves are already vectorized.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
...
for neural_t, me_t in zip(neural_trials, me_trials):
    me_disc = discretize_me(me_t, bin_edges)
```

iii. The AI did not explicitly discuss these loop opportunities. Its notes emphasize that reshape-plus-mean binning is vectorized and that observed run time was acceptable.

## 6-c. What processing does the code repeat multiple times?

i. `make_time_input` recreates the identical trial-relative time vector for every trial. Motion-energy digitization is repeated trial-by-trial despite shared global edges. When plotting is enabled, `F.npy`, `Fneu.npy`, and raw motion energy are loaded again after already being loaded by `process_session`.

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    time_input = make_time_input(n_timebins)
    me_disc = discretize_me(me_t, bin_edges)
...
def plot_processing(...):
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
```

iii. The AI does not document these repetitions; the plotting reloads are an implementation convenience for optional visual checks.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reconstructs a baseline array and performs a division not used by the reference pipeline. It also builds formatted threshold labels, detailed summaries, and optional diagnostic plots that do not affect decoder arrays. `tstamps.npy` is loaded and passed to interpolation but its values are unused.

ii.
```python
baseline = F_corr - F_subtracted
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe
...
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
...
me_indices = np.linspace(0, n_neural_frames - 1, n_me)
```

iii. The AI believed the baseline division was necessary for dF/F and intended timestamps to support gap repair. Plots and labels were included for validation and user-facing interpretability, not downstream numerical analysis.
