# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-coded the six mouse IDs in `MICE`, then iterated over those mice and over session subdirectories whose names start with `2`. For each session it loaded neural fluorescence from `suite2p/plane0/F.npy` and `Fneu.npy`, plus behavioral data from `move_deve/motion_energy_glob.npy` and `interframe_int.npy`. Trials were not loaded directly; the code first loaded full-session arrays, then split them later.

ii. ```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions

for mouse_i, mouse in enumerate(MICE):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = get_sessions(mouse_dir)
    ...
    F = np.load(os.path.join(suite2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
    ...
    me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
    ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. In the trajectory, the agent said the dataset had exactly six mice and that each session contained `suite2p/plane0` and `move_deve` folders. In `CONVERSION_NOTES.md`, it justified this as using the Track2p matched-cell pipeline outputs and videography-derived motion energy.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the six hard-coded entries in `MICE`, in that fixed order. The code does not discover subjects dynamically from the filesystem.

ii. ```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
subjects = MICE[:]
...
for mouse_i, mouse in enumerate(MICE):
    ...
    subject_idx.append(mouse_i)
```

iii. The justification in the notes is that the dataset consists of six mice `jm031-jm046`. The trajectory shows the agent inspected `/app/data` and then chose to encode that list directly.

## 1-c. How are the data split into sessions?

i. For each mouse, sessions are the sorted subdirectories whose names begin with `2`, i.e. date-like folder names such as `2023-10-18_a`.

ii. ```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions
```

iii. The trajectory shows the agent listing session directories and observing that they were date-stamped. The code and notes use that convention as the reason to filter on names beginning with `2`.

## 1-d. How are the data split into trials?

i. The AI defined trials as consecutive non-overlapping 2-minute blocks after 10-frame temporal averaging. At 30 Hz and bin size 10, each trial contains `TRIAL_BINS = 360` time bins.

ii. ```python
BIN_SIZE = 10
TRIAL_DURATION_SEC = 120
TRIAL_BINS = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 360 bins per trial
...
dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
...
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
me_trials = split_into_trials(me_binned, TRIAL_BINS)
```

iii. The trajectory explicitly says the paper “splits data into consecutive 2 minute blocks” and that 10-frame averaging should happen first. `CONVERSION_NOTES.md` repeats that this follows the paper’s decoding setup.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filter. The only effective filtering is structural: `bin_array` truncates leftover frames that do not fill a complete 10-frame bin, and `split_into_trials` truncates leftover bins that do not fill a complete 2-minute trial.

ii. ```python
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

iii. No explicit justification for trial exclusion is documented. The agent’s notes frame the segmentation as fixed 2-minute blocks and do not describe any additional QC step.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from `F.npy` and `Fneu.npy` in each session’s `suite2p/plane0` directory.

ii. ```python
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
```

iii. `CONVERSION_NOTES.md` identifies these as raw fluorescence and neuropil fluorescence from the Suite2p outputs and says they are the intended source for neural activity.

## 2-b. How is the `neural` data processed?

i. The code computes a manual dF/F-like signal: neuropil subtraction (`F - 0.7 * Fneu`), a Gaussian/minimum/maximum “maximin” baseline estimate over a 60 s window, division by baseline to get `(Fc - F0) / F0`, and then 10-frame temporal averaging.

ii. ```python
def compute_dff(F, Fneu):
    Fc = F - NEUCOEFF * Fneu
    win = int(WIN_BASELINE_SEC * FS)
    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    F0 = maximum_filter1d(Flow, size=win, axis=1)
    F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
    dff = (Fc - F0) / F0_safe
    return dff

dff = compute_dff(F, Fneu)
dff_binned = bin_array(dff, BIN_SIZE)
```

iii. The trajectory shows the agent initially trying a percentile baseline, then switching because it was too slow. It justified the final implementation as an approximation to Suite2p’s default `maximin` baseline and as consistent with the paper’s statement about default Suite2p parameters plus 10-frame averaging.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied in code. The script does not use `iscell.npy`, does not threshold cells, and keeps every row in `F.npy` / `Fneu.npy`.

ii. ```python
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
n_neurons, n_frames = F.shape
...
brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. `CONVERSION_NOTES.md` says no additional neuron filtering was applied because Track2p already produced a matched, curated cell set and “all neurons in `iscell.npy` are marked as cells.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code treats the start of the recording session as the alignment event. Trials are successive 2-minute chunks from session start, and metadata labels the alignment event as “Start of recording session.”

ii. ```python
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
...
'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': None,
    ...
}
```

iii. The notes say the decoder input is elapsed time from the start of session. The trajectory similarly frames the continuous recordings as being chopped into consecutive blocks rather than aligned to a stimulus or behavior event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame averaging at 30 Hz, so the final time bin is about 333.33 ms and the data are effectively at 3 Hz.

ii. ```python
BIN_SIZE = 10
FS = 30
BIN_DURATION_MS = (BIN_SIZE / FS) * 1000  # ~333.33 ms
...
dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
...
'time_bin_size': BIN_DURATION_MS,
```

iii. The trajectory repeatedly cites the paper’s phrase about “averaging in bins of 10 consecutive timestamps,” and `CONVERSION_NOTES.md` uses that as the reason for rebinning both neural and behavioral traces.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not loaded from any raw file. The input time is synthesized from the session/trial index plus the assumed frame rate and bin size.

ii. ```python
for t_i in range(n_trials):
    start_sec = t_i * TRIAL_DURATION_SEC
    time_bins = np.linspace(
        start_sec + BIN_DURATION_MS / 2000,
        start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,
        TRIAL_BINS
    )
```

iii. The trajectory states that the decoder input should be “elapsed time from the start of the experiment,” and the agent chose to compute it directly from the known 30 Hz rate rather than from any timestamp file.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the code computes the centers of the 333 ms bins using `np.linspace`, starting at the midpoint of the first bin and ending at the midpoint of the last bin in a 2-minute block, then reshapes the result to `(1, n_timepoints)`.

ii. ```python
time_bins = np.linspace(
    start_sec + BIN_DURATION_MS / 2000,  # center of first bin
    start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,  # center of last bin
    TRIAL_BINS
)
input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
```

iii. The notes justify this as “Time elapsed from start of session” and give an example like `[0.17, 0.50, ..., 119.83] seconds`, which matches the bin-center construction in the code.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned by construction: each trial gets exactly `TRIAL_BINS` timepoints, the same number as the binned neural trial, and both come from the same consecutive 2-minute session blocks.

ii. ```python
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
...
for t_i in range(n_trials):
    ...
    input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
...
neural_all.append(neural_trials)
input_all.append(input_trials)
```

iii. The trajectory says the input should be elapsed time from session start and that neural and behavioral data should both be binned at 10 frames. No further alignment step is documented beyond using the same bin/trial structure.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output motion-energy signal is derived from `move_deve/motion_energy_glob.npy`. The code also uses `move_deve/interframe_int.npy` to infer missing video frames during alignment.

ii. ```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The notes describe `motion_energy_glob.npy` as global motion energy from videography and `interframe_int.npy` as the source for detecting dropped frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code aligns the raw motion-energy vector to neural length by detecting dropped frames from unusually large interframe intervals, interpolates missing samples linearly, then averages the aligned trace in 10-frame bins. Despite the file header and notes claiming normalization, the code never actually normalizes motion energy before discretization.

ii. ```python
def align_motion_energy(me, interframe_int, n_neural_frames):
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    median_ifi = np.median(interframe_int)
    aligned = np.full(n_neural_frames, np.nan)
    neural_idx = 0
    for i in range(len(me)):
        if neural_idx < n_neural_frames:
            aligned[neural_idx] = me[i]
        neural_idx += 1
        if i < len(interframe_int):
            n_dropped = int(np.round(interframe_int[i] / median_ifi - 1))
            neural_idx += n_dropped
    nans = np.isnan(aligned)
    if np.any(nans) and not np.all(nans):
        x = np.arange(n_neural_frames)
        aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])
    return aligned

me_aligned = align_motion_energy(me_raw, ifi, n_frames)
me_binned = bin_array(me_aligned, BIN_SIZE)
```

iii. The trajectory explicitly justifies the dropped-frame handling as using gaps `> 1.5x` the median interval, inserting missing positions, and interpolating. `CONVERSION_NOTES.md` also justifies linear interpolation and claims normalization and 10-frame denoising, but the normalization claim is not reflected in the code.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI pools all binned motion-energy values from all sessions, computes global quintile cut points with `np.percentile`, and then uses `np.digitize` to assign each time bin to one of five categories `0-4`.

ii. ```python
all_me_concat = np.concatenate(all_me_values)
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
bin_edges = np.percentile(all_me_concat, percentiles)
...
binned = np.digitize(me_trial, bin_edges[1:-1])  # 0 to N_OUTPUT_BINS-1
session_output.append(binned.reshape(1, -1).astype(np.int64))
```

iii. The notes justify this as “global quintiles” so that each bin contains about 20% of the total data and class balance is improved for decoding.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The code first stretches motion energy to neural-frame length using `interframe_int.npy`, then bins motion energy and neural activity with the same `BIN_SIZE`, and finally splits both into the same 2-minute trial structure.

ii. ```python
me_aligned = align_motion_energy(me_raw, ifi, n_frames)
...
dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
...
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
me_trials = split_into_trials(me_binned, TRIAL_BINS)
```

iii. The trajectory says the key alignment idea was to detect dropped video frames from large interframe gaps, interpolate them, and thereby create a framewise motion-energy trace matching the neural recordings before binning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missing video frames by interpolation, prevents division by near-zero baseline by clipping `F0` to `1e-6`, and silently discards partial bins and partial trials at the ends of recordings. It also checks for NaN/Inf in the final neural trials during sanity reporting.

ii. ```python
F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
...
aligned = np.full(n_neural_frames, np.nan)
...
aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])
...
return data[:, :n_bins * bin_size].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)
...
return [data[:, i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
...
nan_count = sum(np.any(np.isnan(t)) for s in neural_all for t in s)
inf_count = sum(np.any(np.isinf(t)) for s in neural_all for t in s)
```

iii. The notes explicitly justify linear interpolation of dropped frames, including for some longer gaps. The trajectory also shows the agent choosing interpolation because it preserved framewise alignment without needing to decode the timestamp units.

## 6-a. What are the most time-consuming steps of the code?

i. The main expensive step is the full-session neural preprocessing in `compute_dff`, especially the large 1D Gaussian/minimum/maximum filters over all neurons and frames. A second major cost is that the script always processes all sessions before percentile binning, even when `sample_only=True`.

ii. ```python
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
F0 = maximum_filter1d(Flow, size=win, axis=1)
...
for mouse_i, mouse in enumerate(MICE):
    ...
    dff = compute_dff(F, Fneu)
...
if not sample_only:
    ...
# sample_data is still created after full processing
```

iii. The trajectory explicitly says the original percentile-filter version was “very slow,” then says the dF/F step remained the bottleneck even after switching to the faster maximin-style implementation. It also notes that `sample_only` still processes the whole dataset because global percentile edges are computed from all sessions.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidate is the per-frame loop in `align_motion_energy`, which advances through the motion-energy vector one element at a time while inserting dropped-frame offsets. The per-trial loop that builds `input_trials` with `np.linspace` is also unnecessary because the time grid could be built for the whole session and then sliced.

ii. ```python
for i in range(len(me)):
    if neural_idx < n_neural_frames:
        aligned[neural_idx] = me[i]
    neural_idx += 1
    if i < len(interframe_int):
        n_dropped = int(np.round(interframe_int[i] / median_ifi - 1))
        neural_idx += n_dropped

for t_i in range(n_trials):
    start_sec = t_i * TRIAL_DURATION_SEC
    time_bins = np.linspace(...)
    input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
```

iii. No explicit vectorization discussion appears in the notes, but the trajectory shows the agent repeatedly revising the preprocessing because runtime mattered, so these loops are natural follow-on optimization targets.

## 6-c. What processing does the code repeat multiple times?

i. The code makes multiple passes over motion energy: once to align/bin and collect values, again to discretize each trial, and again to verify class fractions. It also calls `get_sessions` once during conversion and again while building `metadata['session_info']`. In addition, `sample_only=True` still runs the same full preprocessing pass as the full conversion.

ii. ```python
all_me_values = []
...
for mt in me_trials:
    all_me_values.append(mt)
...
for session_me_trials in output_all_raw:
    for me_trial in session_me_trials:
        binned = np.digitize(me_trial, bin_edges[1:-1])
...
all_binned = np.concatenate([np.concatenate([t.flatten() for t in s]) for s in output_all])
...
'session_info': {
    mouse: {
        'sessions': get_sessions(os.path.join(data_dir, mouse)),
        ...
    }
    for mouse in MICE
}
```

iii. The trajectory explicitly calls out the `sample_only` inefficiency and explains that global percentile binning forced it to preprocess everything anyway. No separate justification is given for the repeated `get_sessions` call or post hoc bin-distribution pass.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script spends work on extensive logging/sanity summaries, building `sample_data.pkl`, computing bin-fraction reports, and assembling metadata such as `session_info` and `motion_energy_bin_edges` that are not used by the downstream decoder. When run with `sample_only=True`, it still does the full conversion work before discarding the full-save branch.

ii. ```python
print("SANITY CHECKS")
...
for b in range(N_OUTPUT_BINS):
    frac = np.mean(all_binned == b)
    print(f"  Bin {b}: {frac:.3f} ({frac*100:.1f}%)")
...
'motion_energy_bin_edges': bin_edges.tolist(),
'session_info': {
    mouse: {
        'sessions': get_sessions(os.path.join(data_dir, mouse)),
        ...
    }
    for mouse in MICE
}
...
sample_data = {
    'neural': data['neural'][:n_sample],
    ...
}
with open(sample_output_file, 'wb') as f:
    pickle.dump(sample_data, f)
```

iii. The notes justify these additions as sanity checks and user-facing documentation rather than as decoder requirements. The trajectory also explicitly remarks that the `sample_only` mode still does full preprocessing even though much of that work is not needed for the sample artifact.
