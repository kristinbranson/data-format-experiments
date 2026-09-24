# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers every `jm*` directory under `/app/data`, sorts those subject IDs, discovers and sorts every subdirectory as a session, and processes every discovered session in full mode. For each session it loads `F.npy`, `Fneu.npy`, and `motion_energy_glob.npy`. Sample mode instead keeps two sessions from the first mouse.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
...
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
...
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The agent documented that the data contain six `jm*` mouse folders and 41 date-labelled session folders, and that the same directory/file convention holds throughout. It used actual file lengths rather than forcing the paper's nominal session duration.

## 1-b. How are the data split into subjects?

i. Each sorted `jm*` directory is a subject. The saved public subject labels are mapped from raw IDs to `Mouse_A` through `Mouse_F`; each session receives the corresponding index into that mapped list.

ii.
```python
SUBJECT_MAP = {
    'jm031': 'Mouse_A', 'jm032': 'Mouse_B', 'jm038': 'Mouse_C',
    'jm039': 'Mouse_D', 'jm040': 'Mouse_E', 'jm046': 'Mouse_F'
}
...
subject_names = list(SUBJECT_MAP.values())
subj_idx = subject_names.index(SUBJECT_MAP[subj])
subject_idx_list.append(subj_idx)
```

iii. The notes identify each `jm*` folder as one of the six mice and report consistent neuron counts across days within each mouse, supporting this split. The renaming was a presentation choice.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory of a subject directory is treated as one recording session. Each session becomes one outer-list element in `neural`, `input`, and `output`.

ii.
```python
for subj in subjects_to_process:
    sessions = sessions_to_process[subj]
    for i, session in enumerate(sessions):
        neural_trials, input_trials, output_trials, n_neurons = process_session(...)
        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
```

iii. The agent found 41 daily recordings (7, 7, 7, 7, 6, and 7 per mouse) and treated the date-labelled directories as sessions, matching the dataset organization.

## 1-d. How are the data split into trials?

i. The continuous recording is split after 10-frame temporal averaging into consecutive, non-overlapping 120-second blocks (360 bins). Only complete blocks are emitted, so any tail shorter than 120 seconds is silently dropped.

ii.
```python
TRIAL_DURATION_SEC = 120
TRIAL_BINS = int(TRIAL_DURATION_SEC * FRAME_RATE / BIN_SIZE)
...
n_trials = n_total_bins // TRIAL_BINS
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
```

iii. The agent reasoned that the experiment has no natural trials and chose the paper's “consecutive 2 minute blocks” used for cross-validation. This overlooked the conversion instruction to split sessions into 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filtering. All complete 120-second blocks are retained; an incomplete terminal block is discarded solely because it cannot form a full trial.

ii.
```python
n_trials = n_total_bins // TRIAL_BINS
for t in range(n_trials):
    ...
```

iii. The notes state that the source is continuous spontaneous recording and that the paper specifies no explicit trial curation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `F.npy` and `Fneu.npy` in each session's `suite2p/plane0` directory.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The agent identified these as raw ROI and neuropil fluorescence and noted that the supplied arrays already contain Track2p-matched cells.

## 2-b. How is the `neural` data processed?

i. The code subtracts 0.7 times neuropil fluorescence, applies Suite2p `preprocess` with the maximin baseline settings, casts the result to float32, then averages every 10 consecutive frames.

ii.
```python
Fc = F - NEUROPIL_COEFF * Fneu
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)
...
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
```

iii. The agent justified this from the paper's baseline-corrected fluorescence and 10-timestamp averaging, plus Suite2p defaults (`neucoeff=0.7`, maximin, 60-second window, sigma 10, percentile 8). It explicitly corrected an earlier attempted division by a near-zero baseline and retained Suite2p's baseline-subtracted output.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No filtering is performed by `convert_data.py`; every row of `F`/`Fneu` is retained.

ii.
```python
n_neurons, n_frames = F.shape
...
neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
```

iii. The notes report that all supplied `iscell` flags equal one and that the files are already filtered to cells tracked across all days, so repeating the `iscell > 0.5` and Track2p selection would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Neural trials are contiguous slices anchored to the beginning of the continuous session, and metadata names session start as the alignment event.

ii.
```python
start = t * TRIAL_BINS
end = (t + 1) * TRIAL_BINS
neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
...
'temporal_alignment_event': 'start of continuous recording session',
'off_start': 0.0,
```

iii. The agent explained that these are spontaneous continuous recordings without discrete events. Session start is therefore the natural temporal origin.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. At 30 Hz, the code averages non-overlapping groups of 10 frames, yielding one bin every 1/3 second (333.33 ms, effectively 3 Hz). Partial final 10-frame groups are truncated.

ii.
```python
BIN_SIZE = 10
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000
...
truncated = data[..., :n_bins * bin_size]
return truncated.reshape(new_shape).mean(axis=-1)
```

iii. The agent cited the paper's decoding preprocessing: neural and behavioral traces were denoised by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the absolute binned sample index, the 10-frame bin width, and the assumed 30 Hz frame rate; it is not loaded from a timestamp file.

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The agent treated the fixed imaging rate as sufficient to reconstruct elapsed time and deliberately kept time continuous across trial boundaries.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Integer indices of the already-binned samples are multiplied by `10 / 30` seconds, reshaped to `(1, time)`, and cast to float32. No normalization is applied.

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The notes say this produces elapsed seconds from session start, with a 0.333-second step, as required by the decoder input.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The same absolute `start:end` bin indices used to slice neural data generate the time values, giving one time value per neural column.

ii.
```python
neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The agent validated monotonic elapsed time and reported that consecutive trials differ by exactly one 333-ms bin at their boundary.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is derived only from each session's precomputed `move_deve/motion_energy_glob.npy`. Although the notes mention timestamp and interframe-interval files, the final script does not load either.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The agent described this file as the precomputed pixel-wise video motion-energy trace, synchronized nominally frame-for-frame with imaging.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If lengths differ, motion energy is globally linearly resampled over a normalized 0-to-1 session axis to the neural frame count. It is then averaged in non-overlapping 10-frame bins and discretized using per-session percentiles. No explicit normalization is present despite wording in the docstring/notes.

ii.
```python
x_orig = np.linspace(0, 1, len(me))
x_new = np.linspace(0, 1, n_frames)
me_interp = np.interp(x_new, x_orig, me.astype(np.float64))
...
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
me_discrete = discretize_motion_energy(me_binned, N_OUTPUT_BINS)
```

iii. The agent intended interpolation to handle dropped camera frames and 10-frame averaging to match the paper. Its notes say missing-frame locations can be found from timestamps or interframe intervals, but the implemented global resampling does not use those locations.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four thresholds are the 20th, 40th, 60th, and 80th percentiles of the entire binned motion-energy trace separately for each session. `np.digitize` assigns integer labels 0–4.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
thresholds = np.percentile(me_binned, percentiles)
labels = np.digitize(me_binned, thresholds).astype(np.int64)
```

iii. The task requests five equal-percentile bins selected per session; the agent chose session-specific quintiles to accommodate session-to-session scale differences and verified approximately equal category counts.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is forced to the neural frame count by global linear resampling, both streams are independently averaged with the same 10-frame bin size, and identical `start:end` binned indices are used for trials.

ii.
```python
me = interpolate_motion_energy(me_raw, n_frames)
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
...
output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The agent relied on synchronous 30-Hz acquisition and sought to repair occasional dropped camera frames. However, it stretched/compressed the entire behavioral trace rather than inserting estimates specifically at detected dropped-frame locations.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. A motion-energy length mismatch is handled by global linear interpolation to exactly the neural frame count. Exact-length traces are simply cast to float64. Short tails that cannot fill a 10-frame bin or a complete 120-second trial are truncated. There is no explicit NaN handling or post-interpolation assertion.

ii.
```python
if len(me) == n_frames:
    return me.astype(np.float64)
x_orig = np.linspace(0, 1, len(me))
x_new = np.linspace(0, 1, n_frames)
return np.interp(x_new, x_orig, me.astype(np.float64))
...
n_trials = n_total_bins // TRIAL_BINS
```

iii. The notes correctly identified dropped video frames as the known data defect and selected interpolation, which the data README permits. The implementation is broader than the documented plan because it ignores the supplied missing-frame locations.

## 6-a. What are the most time-consuming steps of the code?

i. Suite2p baseline preprocessing over all neurons and frames is the main computation; loading and serializing large arrays are the principal I/O costs. Optional diagnostic figure generation can also be expensive. The code times each session and the full conversion.

ii.
```python
t_sess = time.time()
...
dfof = s2p_preprocess(..., device=device)
...
with open(output_file, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The agent estimated roughly 0.5–0.7 seconds per session for preprocessing and about 25 seconds for all 41 sessions, and used CUDA when available.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Subject/session traversal must load separate files and is not naturally vectorizable. The per-trial loop could be replaced with reshape/axis operations for neural, time, and output arrays. Plotting axes also loops, but only in optional diagnostics. Core 10-frame binning and percentile labeling are already vectorized.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    ...
```

iii. The agent did not document this optimization opportunity. Its measured runtime was already small, so it prioritized clarity over eliminating the trial loop.

## 6-c. What processing does the code repeat multiple times?

i. For each session it repeats device selection, file loading, neuropil/baseline processing, temporal averaging, percentile computation, type conversion, and construction of time vectors. Diagnostic mode also repeats figure setup for the first two sessions.

ii.
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
...
for i, session in enumerate(sessions):
    neural_trials, input_trials, output_trials, n_neurons = process_session(...)
```

iii. Most repetition is necessary because baselines and category thresholds are session-specific. Device selection could be hoisted out of `compute_dfof`, and a single full-session time vector could be made once before trial slicing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal full conversion, little computed data is discarded beyond incomplete 10-frame and 120-second tails. In `--show-processing` mode, it additionally prepares large raw/intermediate traces and figures for only the first two sessions; these plots are diagnostic and not used by the decoder. It also computes elapsed timings and verbose summaries solely for reporting.

ii.
```python
if show_processing and session_count < 2:
    fig, axes = plt.subplots(6, 1, figsize=(16, 20))
...
fig.savefig(f'processing_{subj}_{session}.png', dpi=100)
...
truncated = data[..., :n_bins * bin_size]
```

iii. The agent intentionally made plotting optional for visual verification. The discarded tails arise from enforcing fixed-size bins/trials, although the incorrect 120-second choice can discard more than the required 60-second segmentation would.
