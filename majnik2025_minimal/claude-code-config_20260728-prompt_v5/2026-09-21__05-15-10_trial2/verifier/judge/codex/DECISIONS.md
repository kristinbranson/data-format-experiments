# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six mouse IDs in `MICE`, loops over each mouse directory under `/app/data`, lists all subdirectories as sessions, and for each session loads calcium traces from `suite2p/plane0/F.npy` and `Fneu.npy` plus motion data from `move_deve/motion_energy_glob.npy` and `tstamps.npy`. Trials are not loaded from disk directly; they are created later by splitting each processed session into fixed 60-second segments.

ii. ```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for mouse_i, mouse in enumerate(MICE):
    mouse_dir = os.path.join(DATA_DIR, mouse)
    sessions = sorted([
        d for d in os.listdir(mouse_dir)
        if os.path.isdir(os.path.join(mouse_dir, d))
    ])

    for sess in sessions:
        session_dir = os.path.join(mouse_dir, sess)
        dff_binned, me_binned = process_session(session_dir)
```

```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. In trajectory step 23 the agent says it found "6 mice, each with 6-7 sessions" and planned "for each mouse and session, load F.npy and Fneu.npy... interpolate motion energy... split into 60-second trials". Step 33 restates the same plan as the final loading strategy.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by a fixed list of six mouse IDs rather than being discovered dynamically from the filesystem.

ii. ```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
for mouse_i, mouse in enumerate(MICE):
```

iii. The trajectory shows the agent inspected `/app/data`, saw exactly six mouse folders, and then treated those six names as the subject set. Step 23 summarizes this as "6 mice, each with 6-7 sessions."

## 1-c. How are the data split into sessions?

i. Within each mouse, every subdirectory under that mouse's folder is treated as one session, and session names are sorted alphabetically.

ii. ```python
mouse_dir = os.path.join(DATA_DIR, mouse)
sessions = sorted([
    d for d in os.listdir(mouse_dir)
    if os.path.isdir(os.path.join(mouse_dir, d))
])

for sess in sessions:
    session_dir = os.path.join(mouse_dir, sess)
```

iii. The agent explored the directory tree early in the trajectory and observed daily recording folders under each mouse. In step 23 it describes the dataset as mice with 6-7 sessions each, and step 33 keeps the same session interpretation.

## 1-d. How are the data split into trials?

i. After neural and motion traces are binned, each session is split into non-overlapping fixed-length 60-second trials. At 30 Hz with 10-frame bins, this yields 180 bins per trial.

ii. ```python
BIN_SIZE = 10
BIN_DUR = BIN_SIZE / FS
TRIAL_DUR = 60
TRIAL_BINS = int(TRIAL_DUR / BIN_DUR)  # 180 bins per 60s trial
```

```python
def split_trials(neural, me_binned, trial_bins=TRIAL_BINS):
    n_bins = neural.shape[1]
    n_trials = n_bins // trial_bins
    neural_trials = []
    me_trials = []

    for t in range(n_trials):
        start = t * trial_bins
        end = start + trial_bins
        neural_trials.append(neural[:, start:end])
        me_trials.append(me_binned[start:end])
```

iii. In steps 20, 23, 29, and 33 the agent repeatedly states that trials should be "60-second" non-overlapping segments and explicitly works out the arithmetic to 180 binned time points per trial.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering. The only effective filtering is that only full 60-second trials are kept because `n_trials` uses floor division, so any tail shorter than one full trial is dropped implicitly.

ii. ```python
def split_trials(neural, me_binned, trial_bins=TRIAL_BINS):
    n_bins = neural.shape[1]
    n_trials = n_bins // trial_bins
    ...
    for t in range(n_trials):
        start = t * trial_bins
        end = start + trial_bins
```

iii. The trajectory does not mention any trial-level QC beyond the fixed 60-second trial definition. The agent's reasoning focuses on segmentation length rather than rejecting trials for quality.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from Suite2p fluorescence arrays `F.npy` and `Fneu.npy`.

ii. ```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu)
```

iii. In steps 23, 31, and 33 the agent explicitly states that it will compute dF/F from `F.npy` and `Fneu.npy` using Suite2p-style preprocessing.

## 2-b. How is the `neural` data processed?

i. The AI computes a dF/F-like trace by subtracting neuropil (`F - 0.7 * Fneu`), then applying a maximin-style baseline estimation consisting of Gaussian smoothing followed by minimum and maximum filters, and finally subtracting that baseline. Afterward the result is temporally averaged into 10-frame bins.

ii. ```python
NEUCOEFF = 0.7
SIG_BASELINE = 10.0
WIN_BASELINE = 60.0

def compute_dff(F, Fneu, fs=FS):
    Fc = F - NEUCOEFF * Fneu
    win = int(WIN_BASELINE * fs)
    Flow = gaussian_filter(Fc.astype(np.float64), [0., SIG_BASELINE])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    dff = Fc - Flow
    return dff.astype(np.float32)
```

```python
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. Steps 23, 26, 29, 31, and 33 show the agent reasoning through the paper's statement about "default Suite2p parameters" and deciding on `neucoeff=0.7`, maximin baseline subtraction, and then 10-frame averaging because the paper says decoding used bins of 10 consecutive timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script applies no explicit neuron filtering. It does not load or apply `iscell.npy`; all rows of `F.npy` are kept, and all neurons are assigned to the single brain region.

ii. ```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
n_neurons = dff_binned.shape[0]
...
brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The agent initially considered `iscell` filtering, but in step 23 it notes "All tracked neurons are classified as cells (iscell all 1s)" and later says it is "including all tracked cells since iscell is uniformly 1." That is the justification for not adding an explicit filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to the start of the recording session rather than to a stimulus or behavioral event. Each trial is just a contiguous 60-second chunk of the continuous session.

ii. ```python
neural_trials.append(neural[:, start:end])
```

```python
'metadata': {
    'temporal_alignment_event': 'start of recording session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. In step 33 the agent defines the input as "Time elapsed from session start in seconds," and in step 42 it summarizes the trials as 60-second non-overlapping segments. That indicates session-start alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10-frame non-overlapping averages at 30 Hz, giving a bin size of 10/30 s = 333.33 ms. This rebinning is applied to both neural and motion-energy traces.

ii. ```python
BIN_SIZE = 10
BIN_DUR = BIN_SIZE / FS  # duration of each bin in seconds
```

```python
def bin_data(data, bin_size=BIN_SIZE):
    if data.ndim == 1:
        n = len(data) // bin_size * bin_size
        return data[:n].reshape(-1, bin_size).mean(axis=1)
    else:
        n = data.shape[1] // bin_size * bin_size
        return data[:, :n].reshape(data.shape[0], -1, bin_size).mean(axis=2)
```

iii. The agent repeatedly cites the paper's phrase about "averaging in bins of 10 consecutive timestamps" in steps 20, 23, 29, 33, and 42, and uses that as the justification for 333.33 ms bins.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from a raw data file. It is synthesized from the trial index and binned frame index using the known frame rate and bin size.

ii. ```python
start_bin = t * TRIAL_BINS
time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
input_trials.append(time_points.reshape(1, -1).astype(np.float32))
```

iii. In step 33 the agent states that the decoder input will be "Time elapsed from session start in seconds." The trajectory does not cite any raw timestamp variable for this input, implying it is computed from indices and sampling rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the AI computes one time value per binned sample. It uses the global bin index within the session, adds `0.5`, and multiplies by the bin duration, so the time represents the center of each 10-frame bin rather than the left edge.

ii. ```python
for t in range(len(neural_trials)):
    start_bin = t * TRIAL_BINS
    time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
    input_trials.append(time_points.reshape(1, -1).astype(np.float32))
```

iii. The trajectory only explicitly justifies using elapsed session time as input (steps 29, 33, 42). The bin-center choice comes from the code comment "Time of each bin center relative to session start" rather than a longer written rationale.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned one-to-one with the binned neural samples: each trial gets a `1 x 180` time vector, with one time value for each neural bin in that same trial.

ii. ```python
neural_trials.append(neural[:, start:end])
...
time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
input_trials.append(time_points.reshape(1, -1).astype(np.float32))
```

iii. The trajectory's justification is again the choice of elapsed session time as the decoder input. The code implements that by using the same trial partitioning and the same number of bins as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy`. The script also loads `move_deve/tstamps.npy` as auxiliary timing information for alignment.

ii. ```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. In steps 23, 24, 28, and 29 the agent investigates `tstamps.npy` in detail and decides to use motion energy plus timestamps to deal with dropped camera frames before generating the output variable.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI aligns motion energy to the neural frame count by interpolation when lengths differ, bins the aligned trace into 10-frame averages, splits it into 60-second trials, and later discretizes the binned values into five session-specific percentile bins.

ii. ```python
def align_motion_energy(me, n_neural_frames, tstamps):
    if len(me) == n_neural_frames:
        return me
    camera_pos = np.linspace(0, 1, len(me))
    neural_pos = np.linspace(0, 1, n_neural_frames)
    me_aligned = np.interp(neural_pos, camera_pos, me)
    return me_aligned
```

```python
me_aligned = align_motion_energy(me, n_neural, tstamps)
me_binned = bin_data(me_aligned, BIN_SIZE)
neural_trials, me_trials = split_trials(dff_binned, me_binned)
me_discrete = discretize_me(me_trials)
```

iii. The agent's written rationale is in steps 29, 33, and 42: missing camera frames should be handled by interpolation to the neural frame count, then both streams should be averaged in 10-frame bins before discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI pools all motion-energy values from the trials of a session, computes four inner percentile boundaries so that five equal-percentile bins are formed, and uses `np.digitize` to assign category labels `0` through `4`.

ii. ```python
def discretize_me(me_trials, n_bins=N_BINS_OUTPUT):
    all_me = np.concatenate(me_trials)
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    boundaries = np.percentile(all_me, percentiles)

    discretized = []
    for me in me_trials:
        binned = np.digitize(me, boundaries)
        discretized.append(binned.astype(np.int64))
```

iii. In steps 20, 29, 33, and 42 the agent says the output should be "motion energy discretized into 5 equal-percentile bins per session," which is exactly what this code implements.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy by making its frame count match the neural frame count before any binning or trial split. If lengths already match, it leaves the trace unchanged; otherwise it interpolates the motion-energy series across normalized positions from start to end of the session.

ii. ```python
def align_motion_energy(me, n_neural_frames, tstamps):
    if len(me) == n_neural_frames:
        return me

    camera_pos = np.linspace(0, 1, len(me))
    neural_pos = np.linspace(0, 1, n_neural_frames)
    me_aligned = np.interp(neural_pos, camera_pos, me)
    return me_aligned
```

```python
n_neural = F.shape[1]
me_aligned = align_motion_energy(me, n_neural, tstamps)
```

iii. The agent's justification appears mainly in step 29: it argues that some sessions have dropped camera frames and concludes that "the cleanest approach is interpolating ME values onto the neural frame times" and that `np.interp` over proportional positions should suffice because the mismatch is small.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are handled by interpolating the motion-energy trace up to the neural frame count whenever the lengths differ. Partial trial tails are dropped implicitly because only complete 60-second trials are emitted. There is no explicit assertion after alignment and no handling for missing neural data.

ii. ```python
def align_motion_energy(me, n_neural_frames, tstamps):
    if len(me) == n_neural_frames:
        return me
    camera_pos = np.linspace(0, 1, len(me))
    neural_pos = np.linspace(0, 1, n_neural_frames)
    me_aligned = np.interp(neural_pos, camera_pos, me)
    return me_aligned
```

```python
n_trials = n_bins // trial_bins
for t in range(n_trials):
    start = t * trial_bins
    end = start + trial_bins
```

iii. In steps 23, 29, 33, and 42 the agent consistently frames the main data problem as dropped camera frames causing motion-energy length mismatches, and it justifies interpolation as the fix because the mismatch fraction is small.

## 6-a. What are the most time-consuming steps of the code?

i. The heaviest work is the per-session dF/F preprocessing in `compute_dff`, especially the Gaussian smoothing and the min/max baseline filters over full-session fluorescence matrices. The session-wise interpolation and binning are much lighter, and the trial/list assembly is comparatively cheap Python overhead.

ii. ```python
def compute_dff(F, Fneu, fs=FS):
    Fc = F - NEUCOEFF * Fneu
    win = int(WIN_BASELINE * fs)
    Flow = gaussian_filter(Fc.astype(np.float64), [0., SIG_BASELINE])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    dff = Fc - Flow
```

iii. The trajectory's only explicit performance-related reasoning is around the neural preprocessing choice: the agent spends most of its effort deciding how to reproduce Suite2p-style baseline correction, indicating that this is the central computational step.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-building loop in `split_trials`, the per-trial loop in `discretize_me`, and the per-trial loop that constructs `input_trials` could all be reduced with reshape/slicing or by operating on a whole session tensor at once. These are not huge bottlenecks, but they are the clearest vectorization opportunities in the AI's code.

ii. ```python
for t in range(n_trials):
    start = t * trial_bins
    end = start + trial_bins
    neural_trials.append(neural[:, start:end])
    me_trials.append(me_binned[start:end])
```

```python
for me in me_trials:
    binned = np.digitize(me, boundaries)
    discretized.append(binned.astype(np.int64))
```

```python
for t in range(len(neural_trials)):
    start_bin = t * TRIAL_BINS
    time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
    input_trials.append(time_points.reshape(1, -1).astype(np.float32))
```

iii. The trajectory does not contain an explicit vectorization discussion for this script. This answer is based on the structure of the implemented code.

## 6-c. What processing does the code repeat multiple times?

i. There is no major repeated heavy processing beyond the intended per-session loop. Minor repeated work includes recomputing per-trial time vectors for every session and wrapping already split motion-energy arrays back into lists trial by trial.

ii. ```python
for t in range(len(neural_trials)):
    start_bin = t * TRIAL_BINS
    time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
    input_trials.append(time_points.reshape(1, -1).astype(np.float32))
```

```python
all_output.append([me.reshape(1, -1) for me in me_discrete])
```

iii. The trajectory does not mention repeated-processing concerns. The code itself shows only minor repetition, not a repeated heavyweight preprocessing pass.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is loading `tstamps.npy` and threading `tstamps` into `align_motion_energy`, even though the function ignores that argument and aligns by normalized array position instead. The timestamp file is therefore read from disk but not used downstream.

ii. ```python
def align_motion_energy(me, n_neural_frames, tstamps):
    if len(me) == n_neural_frames:
        return me

    camera_pos = np.linspace(0, 1, len(me))
    neural_pos = np.linspace(0, 1, n_neural_frames)
    me_aligned = np.interp(neural_pos, camera_pos, me)
    return me_aligned
```

```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
me_aligned = align_motion_energy(me, n_neural, tstamps)
```

iii. The trajectory shows the agent spent substantial time reasoning about `tstamps` (steps 23, 24, 28, 29), but the final implementation does not use the timestamp values. That mismatch is the main unnecessary processing in the delivered code.
