# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six mouse IDs, iterates over each mouse directory, finds all session subdirectories whose names start with `2`, and for every session loads neural arrays from `suite2p/plane0` and behavior arrays from `move_deve`. Trials are not loaded directly from disk; they are created later by splitting each session into fixed-length blocks after preprocessing and binning.

ii. 
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for mouse_i, mouse in enumerate(MICE):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = get_sessions(mouse_dir)
    ...
    for sess_name in sessions:
        sess_dir = os.path.join(mouse_dir, sess_name)
        suite2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
        move_dir = os.path.join(sess_dir, 'move_deve')

        F = np.load(os.path.join(suite2p_dir, 'F.npy'))
        Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
        ...
        me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
        ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. In `CONVERSION_NOTES.md`, the AI says the source data are Suite2p neural outputs (`F.npy`, `Fneu.npy`) and processed behavioral data (`motion_energy_glob.npy`). In trajectory step 21 it explicitly summarizes the plan as loading neural data from Suite2p and motion energy from `move_deve/motion_energy_glob.npy`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by top-level mouse folder. The script uses the hard-coded `MICE` list as the canonical subject list and appends one `subject_idx` entry per session using the mouse's position in that list.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
subjects = MICE[:]
subject_idx = []
...
for mouse_i, mouse in enumerate(MICE):
    ...
    for sess_name in sessions:
        ...
        subject_idx.append(mouse_i)
```

iii. The justification in `CONVERSION_NOTES.md` is that the dataset contains six mice `jm031-jm046`. The data README also describes one folder per subject and says these names correspond to mice A-F in the paper.

## 1-c. How are the data split into sessions?

i. Sessions are split by subdirectories inside each mouse directory. Any subdirectory whose name starts with `2` is treated as a session; these names are assumed to be date-stamped recording days and are sorted lexicographically.

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions
```

iii. The AI relied on the folder organization as authoritative. `CONVERSION_NOTES.md` describes 6-7 daily sessions per mouse, and the dataset README says each subject folder contains one folder per recording day named `YYYY-MM-DD_a`.

## 1-d. How are the data split into trials?

i. The AI does not use experimental trial markers from raw data. Instead, after preprocessing and 10-frame temporal binning, each session is partitioned into consecutive, non-overlapping 2-minute blocks. With 30 Hz imaging and binning by 10 frames, each block becomes 360 time bins.

ii.
```python
TRIAL_DURATION_SEC = 120
TRIAL_BINS = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 360 bins per trial
...
dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
...
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
me_trials = split_into_trials(me_binned, TRIAL_BINS)
```

iii. In `CONVERSION_NOTES.md`, the AI cites the paper phrase that “splits were done on consecutive 2 minute blocks of the recording.” Trajectory steps 21 and 28 repeat the same interpretation and compute 10 trials for 20-minute sessions and 15 trials for 30-minute sessions.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies essentially no explicit trial-level quality control. Every full 2-minute block produced by integer division is kept. Motion-energy gaps are repaired by interpolation before trial splitting, so trials with missing video samples are retained. The only implicit filtering is that incomplete remainder bins or remainder timepoints would be dropped by truncation in `bin_array` and `split_into_trials`.

ii.
```python
def bin_array(data, bin_size):
    if data.ndim == 1:
        n_bins = len(data) // bin_size
        return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    else:
        n_bins = data.shape[1] // bin_size
        return data[:, :n_bins * bin_size].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)

def split_into_trials(data, trial_length):
    if data.ndim == 1:
        n_trials = len(data) // trial_length
        return [data[i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
    else:
        n_trials = data.shape[1] // trial_length
        return [data[:, i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
```

iii. `CONVERSION_NOTES.md` says “No additional neuron filtering was applied” and describes interpolation for missing video frames, but it does not document any trial rejection rule. The trajectory likewise focuses on alignment/interpolation rather than dropping problematic trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from Suite2p fluorescence traces `F.npy` and neuropil traces `Fneu.npy`. The AI ignores `spks.npy` even though the notebook mentions it as an alternative.

ii.
```python
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu)
```

iii. `CONVERSION_NOTES.md` explicitly identifies `F.npy` and `Fneu.npy` as the neural sources. Trajectory step 21 states that Suite2p stores raw fluorescence and neuropil separately, so the AI planned to compute dF/F from those files.

## 2-b. How is the `neural` data processed?

i. The AI computes neuropil-corrected fluorescence `Fc = F - 0.7 * Fneu`, estimates a baseline using a Gaussian smoothing plus minimum and maximum filters, divides by that baseline to form `(Fc - F0) / F0`, and then averages over 10-frame bins. This is the AI's claimed implementation of “Suite2p default” dF/F, but it is not identical to the installed Suite2p preprocessing code.

ii.
```python
def compute_dff(F, Fneu):
    Fc = F - NEUCOEFF * Fneu
    win = int(WIN_BASELINE_SEC * FS)
    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    F0 = maximum_filter1d(Flow, size=win, axis=1)
    F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
    dff = (Fc - F0) / F0_safe
    return dff

dff_binned = bin_array(dff, BIN_SIZE)
```

iii. The justification in `CONVERSION_NOTES.md` is that the paper used “baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters).” Trajectory steps 47-48 show that the AI originally tried a slower percentile approach, then switched to what it believed was Suite2p's faster default “maximin” baseline. Its rationale was fidelity to “default Suite2p parameters” plus runtime.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply any additional neuron-level filtering. It assumes the Track2p-exported Suite2p matrices already contain only tracked, verified cells present across all days, and it keeps every row in `F.npy`/`Fneu.npy`. There is also no trial/session exclusion based on neural quality, aside from later sanity checks for NaN/Inf.

ii.
```python
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
n_neurons, n_frames = F.shape
...
nan_count = sum(np.any(np.isnan(t)) for s in neural_all for t in s)
inf_count = sum(np.any(np.isinf(t)) for s in neural_all for t in s)
```

iii. `CONVERSION_NOTES.md` states “Track2p already identified cells present across all days for each mouse” and “No additional neuron filtering was applied.” The dataset README supports this by saying the matrices already include only cells present across all days and that row identities are matched across sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats the start of the recording session as the temporal anchor. Neural traces are not shifted to a within-session event; instead, the full session is chopped into consecutive 2-minute blocks, and the corresponding decoder input gives elapsed time from session start for each neural bin.

ii.
```python
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
...
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': None,
```

iii. `CONVERSION_NOTES.md` describes the decoder input as “Time elapsed from start of session,” and the trajectory repeatedly frames the task as using elapsed time from the beginning of the experiment/session rather than aligning to a discrete behavioral event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame temporal bins at a nominal 30 Hz frame rate, yielding 3 Hz sampling and a bin size of about 333.33 ms. Both neural and motion-energy streams are averaged within each bin; there is no additional temporal rebinning afterward.

ii.
```python
BIN_SIZE = 10
FS = 30
BIN_DURATION_MS = (BIN_SIZE / FS) * 1000  # ~333.33 ms
...
dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
```

iii. `CONVERSION_NOTES.md` says the processing follows the paper statement that dF/F and behavior were “averaging in bins of 10 consecutive timestamps.” Trajectory step 21 also computes the effective temporal resolution as roughly 333 ms.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from a dedicated raw timestamp array. It is synthesized from the hard-coded imaging rate `FS = 30`, the hard-coded bin size `BIN_SIZE = 10`, the hard-coded trial duration `TRIAL_DURATION_SEC = 120`, and the trial index within each session. The session duration itself is indirectly inherited from the neural frame count because `n_trials` comes from the binned neural length.

ii.
```python
FS = 30
BIN_SIZE = 10
TRIAL_DURATION_SEC = 120
BIN_DURATION_MS = (BIN_SIZE / FS) * 1000
...
n_bins = dff_binned.shape[1]
n_trials = n_bins // TRIAL_BINS
...
start_sec = t_i * TRIAL_DURATION_SEC
time_bins = np.linspace(
    start_sec + BIN_DURATION_MS / 2000,
    start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,
    TRIAL_BINS
)
```

iii. The AI's justification, from trajectory steps 24 and 28, is that `tstamps.npy` looked unusable for wall-clock timing, so it decided to trust the nominal 30 Hz acquisition rate and reconstruct time from frame/bin counts instead.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each 2-minute block, the AI creates a length-360 vector of bin-center times in seconds. Trial 0 spans approximately `0.1667` to `119.8333` s, trial 1 spans `120.1667` to `239.8333` s, and so on. The vector is reshaped to `(1, n_timepoints)` and stored as `float32`.

ii.
```python
input_trials = []
for t_i in range(n_trials):
    start_sec = t_i * TRIAL_DURATION_SEC
    time_bins = np.linspace(
        start_sec + BIN_DURATION_MS / 2000,
        start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,
        TRIAL_BINS
    )
    input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
```

iii. `CONVERSION_NOTES.md` explicitly says the input is “Time elapsed from start of session (in seconds)” and that each trial uses the centers of the bins. No separate justification beyond this design choice appears in the notes.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned one-to-one with the binned neural data. The AI first bins the neural traces, then creates exactly `TRIAL_BINS` timepoints for each trial, so each time value corresponds to one neural column in the session/trial matrices.

ii.
```python
dff_binned = bin_array(dff, BIN_SIZE)
...
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
...
input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
...
neural_all.append(neural_trials)
input_all.append(input_trials)
```

iii. The AI's justification is implicit: the decoder input should be time-varying and share the same trial/time indexing as the neural data. `CONVERSION_NOTES.md` describes the input in exactly those terms.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from the precomputed behavioral file `move_deve/motion_energy_glob.npy`. The AI does not compute motion energy from raw video frames because those are not included.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. `CONVERSION_NOTES.md` explicitly names `move_deve/motion_energy_glob.npy` as the behavioral source. The dataset README describes the same file as processed spontaneous-behavior motion energy extracted from videography.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI treats `motion_energy_glob.npy` as the continuous motion-energy signal, aligns it to neural frame count by inserting missing-frame gaps inferred from `interframe_int.npy`, linearly interpolates over the gaps, averages in 10-frame bins, splits into 2-minute trials, and only afterward discretizes it. It does not apply any explicit continuous normalization step before discretization.

ii.
```python
def align_motion_energy(me, interframe_int, n_neural_frames):
    ...
    aligned = np.full(n_neural_frames, np.nan)
    ...
    if np.any(nans) and not np.all(nans):
        x = np.arange(n_neural_frames)
        aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])
    return aligned

me_aligned = align_motion_energy(me_raw, ifi, n_frames)
me_binned = bin_array(me_aligned, BIN_SIZE)
me_trials = split_into_trials(me_binned, TRIAL_BINS)
```

iii. The justification comes from two places. The dataset README says missing motion-energy samples can be treated as missing values or interpolated over. Trajectory steps 24-26 show the AI deciding to use `interframe_int.npy` to detect dropped frames, insert missing values, and interpolate so the behavior stream matches neural frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After collecting all binned motion-energy values from all sessions and mice, the AI computes five equal-percentile bins globally over the full dataset. It then uses `np.digitize` on the interior percentile edges to assign category labels `0` through `4`, stored as a `(1, n_timepoints)` integer array per trial.

ii.
```python
all_me_concat = np.concatenate(all_me_values)
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
bin_edges = np.percentile(all_me_concat, percentiles)
...
for session_me_trials in output_all_raw:
    session_output = []
    for me_trial in session_me_trials:
        binned = np.digitize(me_trial, bin_edges[1:-1])
        session_output.append(binned.reshape(1, -1).astype(np.int64))
    output_all.append(session_output)
```

iii. `CONVERSION_NOTES.md` says the task required “normalized and discretized into five equal-percentile bins” and explains that the AI chose global quintiles to keep classes balanced across the full dataset. The trajectory later notes that this makes sample subsets look imbalanced even though the full dataset is exactly balanced.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data in two stages. First, the continuous motion-energy vector is expanded to neural frame count by inserting dropped frames and interpolating. Second, the now frame-aligned motion-energy vector is binned with the same 10-frame averaging as neural data and split into the same 2-minute trial boundaries.

ii.
```python
me_aligned = align_motion_energy(me_raw, ifi, n_frames)
...
dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
...
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
me_trials = split_into_trials(me_binned, TRIAL_BINS)
```

iii. The dataset README explicitly says that in sessions with missing camera frames, the missing indices can be located with `tstamps.npy` or `interframe_int.npy` and treated as missing or interpolated over. The trajectory shows the AI adopting the interpolation option and using it specifically to align motion energy to neural frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing video frames by identifying gaps in `interframe_int.npy`, inserting NaNs at those positions, and linearly interpolating over them. It also protects against division by zero or tiny baselines in neural preprocessing by replacing very small `F0` values with `1e-6`. It does not implement broader error handling for missing files, all-NaN aligned behavior arrays, or low-quality sessions.

ii.
```python
n_dropped = int(np.round(interframe_int[i] / median_ifi - 1))
...
aligned = np.full(n_neural_frames, np.nan)
...
aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])
...
F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
```

iii. `CONVERSION_NOTES.md` justifies interpolation by saying missing camera frames affect 9/41 sessions and that linear interpolation is simple and appropriate for mostly small gaps. Trajectory steps 24-26 document the same reasoning after the AI found that `tstamps.npy` was hard to interpret.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the neural preprocessing, especially `compute_dff` over full-session fluorescence matrices for every mouse and day. The trajectory shows that the original version was even slower when it used a percentile filter; the final version still spends most of its work in Gaussian/min/max filtering over large `(neurons x frames)` arrays. Full-dataset loading and repeated concatenations for motion-energy binning are secondary costs.

ii.
```python
def compute_dff(F, Fneu):
    Fc = F - NEUCOEFF * Fneu
    win = int(WIN_BASELINE_SEC * FS)
    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    F0 = maximum_filter1d(Flow, size=win, axis=1)
    ...
```

iii. Trajectory steps 38, 40, 44, and 47 explicitly say the conversion was dominated by baseline filtering and that the original percentile-filter implementation was too slow on the large matrices.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops could have been reduced or vectorized: the per-frame loop in `align_motion_energy`, the per-trial loop that constructs `input_trials`, the nested loops used to collect `all_me_values`, and the nested loops used to discretize motion energy trial by trial. The flattening used to compute `all_binned` is also a Python-heavy nested list construction.

ii.
```python
for i in range(len(me)):
    ...

for t_i in range(n_trials):
    ...
    input_trials.append(time_bins.reshape(1, -1).astype(np.float32))

for mt in me_trials:
    all_me_values.append(mt)

for session_me_trials in output_all_raw:
    session_output = []
    for me_trial in session_me_trials:
        binned = np.digitize(me_trial, bin_edges[1:-1])
        session_output.append(binned.reshape(1, -1).astype(np.int64))
```

iii. The AI did not explicitly discuss these vectorization opportunities in its notes, but they follow directly from the implementation. The only performance justification it recorded was around replacing a slow percentile filter with a faster approximation of Suite2p's maximin baseline.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats session discovery via `get_sessions`, once during the main processing pass and again while building metadata. It also processes the entire dataset even when `sample_only=True`, because the script still needs global motion-energy percentiles and never short-circuits after two sessions. There are also repeated full-dataset concatenations for statistics and class-balance checks.

ii.
```python
for mouse_i, mouse in enumerate(MICE):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = get_sessions(mouse_dir)
    ...

'session_info': {
    mouse: {
        'sessions': get_sessions(os.path.join(data_dir, mouse)),
        ...
    }
    for mouse in MICE
}

if not sample_only:
    ...

sample_data = {
    'neural': data['neural'][:n_sample],
    ...
}
```

iii. Trajectory step 77 explicitly notices that `--sample-only` still processes all sessions because the script computes global percentile edges before saving anything. That is the clearest self-identified repeated work in the record.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores continuous motion-energy trials in `output_all_raw` and `all_me_values` only to discard them after percentile discretization. It computes `all_binned` solely for logging class frequencies. It recalculates `get_sessions(...)` for metadata instead of reusing the previously discovered session lists. In addition, `PRCTILE_BASELINE` is recorded in metadata but not actually used by the final neural preprocessing path.

ii.
```python
output_all_raw = []  # store raw ME values before discretization
all_me_values = []   # collect all binned ME for global percentile computation
...
output_all_raw.append(me_trials)
...
all_me_concat = np.concatenate(all_me_values)
...
all_binned = np.concatenate([np.concatenate([t.flatten() for t in s]) for s in output_all])
...
'baseline_percentile': PRCTILE_BASELINE,
```

iii. The AI did not explicitly call all of these unnecessary pieces out, but trajectory step 77 does acknowledge one major example: `sample_only` still incurs full-dataset work even though the downstream sample file uses only two sessions.
