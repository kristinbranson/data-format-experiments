# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for subject directories whose names start with `jm`, then scans every subdirectory inside each subject as a session. For each session it loads neural fluorescence from `suite2p/plane0/F.npy` and `Fneu.npy`, and motion energy from `move_deve/motion_energy_glob.npy`. Trials are not loaded directly from disk; they are created later from the session-long arrays.

ii. ```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])

for si, subject in enumerate(subjects):
    subject_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, d))])

    for sess in sessions:
        sess_dir = os.path.join(subject_dir, sess)
        s2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
        move_dir = os.path.join(sess_dir, 'move_deve')

        F = np.load(os.path.join(s2p_dir, 'F.npy'))
        Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
        me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. In the trajectory, the AI said it had identified "6 mice (jm031-jm046), each with 6-7 sessions" and described each session as containing `F.npy`, `Fneu.npy`, and `motion_energy_glob.npy` (step 19). It treated the directory naming convention as the canonical way to enumerate the dataset.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the top-level directories under `/app/data` whose names start with `jm`, sorted lexicographically.

ii. ```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
```

iii. The trajectory shows the AI explicitly recognizing the six `jm*` directories as the six mice in the dataset (step 19). There was no more elaborate justification beyond following the dataset layout.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory inside a subject directory, again sorted lexicographically. The AI treats every such subdirectory as one recording session.

ii. ```python
subject_dir = os.path.join(DATA_DIR, subject)
sessions = sorted([d for d in os.listdir(subject_dir)
                  if os.path.isdir(os.path.join(subject_dir, d))])

for sess in sessions:
    sess_dir = os.path.join(subject_dir, sess)
```

iii. The AI’s trajectory describes "each session" as a folder containing one set of Suite2p outputs and one motion-energy folder (step 19). It relied on that directory structure directly.

## 1-d. How are the data split into trials?

i. The AI creates artificial trials by binning the continuous recording first, then cutting each session into 60-second non-overlapping segments. With `FS = 30` and `BIN_SIZE = 10`, each trial has 180 binned time points. Any leftover bins at the end of a session are ignored because `n_trials` is computed with floor division.

ii. ```python
bins_per_trial = int(TRIAL_DUR * FS / BIN_SIZE)  # 180
...
n_bins_total = dff_binned.shape[1]
n_trials = n_bins_total // bins_per_trial
...
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial

    session_neural.append(dff_binned[:, start:end])
    session_input.append(time_vec[start:end].reshape(1, -1))
    session_output.append(me_discrete[start:end].reshape(1, -1))
```

iii. In steps 22 and 27, the AI said it would "split each session into 60-second trials," which at 3 Hz gives "180 bins per trial." It justified this directly from the task instructions, since the source dataset is continuous rather than trial-based.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any trial-level quality filter. The only guard is that it skips an entire session if fewer than two 60-second trials can be formed after binning.

ii. ```python
n_trials = n_bins_total // bins_per_trial

if n_trials < 2:
    print(f"Skipping {subject}/{sess}: only {n_trials} trials")
    continue
```

iii. The trajectory does not show a separate scientific justification for filtering trials. This guard appears to come from the task requirement that each session should have at least two trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the Suite2p fluorescence arrays `F.npy` and `Fneu.npy` in `suite2p/plane0`.

ii. ```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu)
```

iii. In steps 19, 22, and 27, the AI repeatedly described the neural source data as raw fluorescence plus neuropil fluorescence from Suite2p output, and planned to apply Suite2p-style neuropil correction to those arrays.

## 2-b. How is the `neural` data processed?

i. The AI first subtracts neuropil as `Fc = F - 0.7 * Fneu`. It then manually estimates a "maximin" baseline using a Gaussian filter followed by minimum and maximum filters over a 60-second window, and outputs baseline-subtracted fluorescence `Fc - Flow`. Earlier in the trajectory it considered dividing by the baseline to form a conventional dF/F, but after decoder performance checks it changed course and kept only subtraction.

ii. ```python
def compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF,
                sig_baseline=SIG_BASELINE, win_baseline=WIN_BASELINE):
    Fc = F - neucoeff * Fneu

    win = int(win_baseline * fs)
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)

    dff = Fc - Flow
    return dff
```

iii. The trajectory shows the AI first planning "neuropil-corrected fluorescence ... then subtracting a rolling-window baseline and dividing by that baseline" (step 19, and again in step 22), but later explicitly revising that decision: "both just do baseline subtraction (`Fc - Flow`) without division" (step 43). Its final summary says the neural signal is "baseline-subtracted fluorescence (Fc - Flow)" using Suite2p-style defaults (step 52).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply additional neuron-level filtering. It includes all neurons present in `F.npy` for each session.

ii. ```python
n_neurons, n_frames = F.shape
...
all_brain_region_idx.append(np.zeros(n_neurons, dtype=int))
```

iii. The script header and final summary both state that there is "No additional neuron filtering" because the dataset is already Track2p output and `iscell` is all ones. The trajectory says, "Since all cells are already tracked (iscell = 1), no filtering is needed" (step 19; repeated in step 52).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural trials to the start of the imaging session. It treats each session as a continuous recording, bins it, and then slices contiguous 60-second windows without any event-specific realignment.

ii. ```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    session_neural.append(dff_binned[:, start:end])
...
'metadata': {
    'temporal_alignment_event': 'Start of imaging session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DUR),
}
```

iii. The AI’s trajectory says the decoder input should be "time elapsed from session start in seconds" and that the continuous recording should simply be split into 60-second trials (steps 22 and 27). That implies session-start alignment rather than a stimulus or behavioral event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and motion-energy streams into non-overlapping averages of 10 original frames. At 30 Hz this yields 3 Hz data, i.e. `10 / 30 = 0.333...` seconds per bin or about 333.3 ms.

ii. ```python
FS = 30.0
BIN_SIZE = 10
...
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
...
'time_bin_size': BIN_SIZE / FS * 1000,
```

iii. The trajectory cites the paper’s decoding methods as "average in bins of 10 consecutive timestamps" and says this gives an "effective 3 Hz rate" (steps 19, 27, 52).

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not read a raw timestamp variable for this input. It synthesizes time from the index of each binned sample, using the known frame rate and bin size.

ii. ```python
time_bin_dur = BIN_SIZE / FS  # seconds per bin
time_vec = np.arange(n_bins_total) * time_bin_dur
```

iii. In steps 27 and 52, the AI says the decoder input is "Time elapsed from session start in seconds." The trajectory does not cite a separate raw time array; the AI treated the constant sampling rate as sufficient to reconstruct time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. After binning the session to 3 Hz, the AI computes one full-session time vector in seconds with a constant step of `BIN_SIZE / FS`, then slices that vector into per-trial segments.

ii. ```python
time_bin_dur = BIN_SIZE / FS  # seconds per bin
time_vec = np.arange(n_bins_total) * time_bin_dur
...
session_input.append(time_vec[start:end].reshape(1, -1))
```

iii. The trajectory justification is minimal. The AI mainly justified this as the natural meaning of "time elapsed from beginning of session" and later checked that trial 0 spans `0-60s`, trial 1 spans `60-120s`, etc. (step 36).

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned by construction: the AI computes a full-session time vector with one entry per binned neural sample, and then slices time and neural data with the same `start:end` indices for every trial.

ii. ```python
session_neural.append(dff_binned[:, start:end])
session_input.append(time_vec[start:end].reshape(1, -1))
```

iii. In the trajectory, the AI explicitly checked that it was using session-level time so later trials continue at `60-120s`, `120-180s`, etc., and concluded that this matched the instruction (step 36).

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. In the final code, motion energy is derived directly from `move_deve/motion_energy_glob.npy`. The AI inspected `tstamps.npy` and `interframe_int.npy` during development, but it did not use those arrays in the final implementation.

ii. ```python
move_dir = os.path.join(sess_dir, 'move_deve')
...
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
me = interpolate_me(me, n_frames)
```

iii. The trajectory shows the AI noticing that some sessions had fewer motion-energy samples than neural frames, reading the README note that missing frames could be "interpolated over," and concluding that it would "simply interpolate the motion energy array to match the neural frame count" (step 22). Even though it examined `tstamps.npy` and `interframe_int.npy`, it ultimately did not use them.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates the entire motion-energy trace to the neural frame count whenever the lengths differ, using normalized positions from 0 to 1. It then averages the resampled trace into non-overlapping 10-frame bins. Discretization happens afterward.

ii. ```python
def interpolate_me(me, target_len):
    if len(me) == target_len:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_orig, me)
...
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
me = interpolate_me(me, n_frames)
me_binned = bin_data(me.reshape(1, -1), BIN_SIZE, axis=1).squeeze()
```

iii. The trajectory justification was that dropped camera frames are rare and the README says they can be interpolated. The AI therefore decided to interpolate the motion-energy array up to the neural frame count instead of trying to insert values at specific missing-frame indices (step 22; summarized again in steps 27 and 52).

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes session-specific percentile edges at 0, 20, 40, 60, 80, and 100 percent, then uses `np.digitize` to convert the binned motion-energy trace into five ordinal categories `0..4`.

ii. ```python
percentiles = np.linspace(0, 100, N_BINS_OUTPUT + 1)
bin_edges = np.percentile(me_binned, percentiles)
me_discrete = np.digitize(me_binned, bin_edges[1:-1])
```

iii. The trajectory consistently states that the decoder output should be "5 equal-percentile bins per session," taking this directly from the task specification (steps 22, 27, 52).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by first resampling the motion-energy trace to the same number of frame-level samples as the neural trace, then binning both streams with the same 10-frame windows, and finally slicing them into trials with the same `start:end` indices.

ii. ```python
n_neurons, n_frames = F.shape
...
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
me = interpolate_me(me, n_frames)
...
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
me_binned = bin_data(me.reshape(1, -1), BIN_SIZE, axis=1).squeeze()
...
session_neural.append(dff_binned[:, start:end])
session_output.append(me_discrete[start:end].reshape(1, -1))
```

iii. The AI justified this by noting that the neural and behavioral recordings are intended to be synchronous, but that some camera frames are missing. Because the README says missing values can be interpolated over, it chose linear interpolation to restore one motion-energy sample per neural frame before shared binning and trial slicing (step 22; summarized in step 52).

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing motion-energy samples by linearly interpolating the trace to the neural frame count. It also handles incomplete final trials by truncating them through floor division, and it would skip sessions with fewer than two full trials.

ii. ```python
def interpolate_me(me, target_len):
    if len(me) == target_len:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_orig, me)
...
n_trials = n_bins_total // bins_per_trial

if n_trials < 2:
    print(f"Skipping {subject}/{sess}: only {n_trials} trials")
    continue
```

iii. The AI’s reasoning was that missing camera frames are rare and the README explicitly allows interpolation over them, so a simple interpolation step was acceptable (step 22). The session-skipping guard appears to come from the decoder-format requirement rather than from the paper.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive part of the AI code is the neural preprocessing in `compute_dff`, especially the Gaussian, minimum, and maximum filters over every neuron’s full-session trace. Loading large `.npy` arrays and then binning full-session arrays are also likely substantial costs.

ii. ```python
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
```

iii. The trajectory does not contain a dedicated performance analysis, but it does show the AI spending most of its reasoning effort on the neural baseline-correction step and later revisiting it after decoder performance checks (steps 19, 22, 43). That is consistent with this being the dominant computational step.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target in the AI code is the per-trial Python loop that slices and appends one trial at a time for neural, input, and output data. The subject/session loops are structural rather than easy vectorization targets. Motion-energy interpolation itself is already vectorized via `np.interp`.

ii. ```python
session_neural = []
session_input = []
session_output = []

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial

    session_neural.append(dff_binned[:, start:end])
    session_input.append(time_vec[start:end].reshape(1, -1))
    session_output.append(me_discrete[start:end].reshape(1, -1))
```

iii. The trajectory does not discuss performance-oriented vectorization explicitly. This conclusion follows from the code structure itself.

## 6-c. What processing does the code repeat multiple times?

i. The AI repeats the same load → preprocess neural data → load/interpolate motion energy → bin → discretize → split into trials pipeline independently for every session. Within each session it also recomputes a full-session time vector and session-specific percentile edges.

ii. ```python
for si, subject in enumerate(subjects):
    ...
    for sess in sessions:
        ...
        dff = compute_dff(F, Fneu)
        ...
        me = interpolate_me(me, n_frames)
        ...
        bin_edges = np.percentile(me_binned, percentiles)
        ...
        time_vec = np.arange(n_bins_total) * time_bin_dur
```

iii. The trajectory treats this repeated work as the normal per-session conversion pipeline rather than as an optimization problem. There is no explicit separate justification.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes a full-session time vector and full-session binned arrays even though only the first `n_trials * bins_per_trial` samples are ultimately used; any tail that does not fill a complete trial is silently dropped. The summary printing and extra metadata are also not used by the downstream decoder.

ii. ```python
n_bins_total = dff_binned.shape[1]
n_trials = n_bins_total // bins_per_trial
...
time_vec = np.arange(n_bins_total) * time_bin_dur
...
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
```

iii. The trajectory does not call this out explicitly. It is an observation from the final code structure.
