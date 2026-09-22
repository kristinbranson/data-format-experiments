# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks every top-level directory under `/app/data`, treats each as a subject, then walks every subdirectory under each subject as a session. Within each session it loads calcium traces from `suite2p/plane0/F.npy` and `Fneu.npy`, and motion energy from `move_deve/motion_energy_glob.npy`. Trials are not loaded directly from disk; they are created later by splitting each processed session into fixed 60-second chunks.

ii. 
```python
def get_subjects_and_sessions():
    """Get all subjects and their session directories."""
    subjects = sorted([d for d in os.listdir(DATA_ROOT)
                      if os.path.isdir(os.path.join(DATA_ROOT, d))])
    result = []
    for subj in subjects:
        subj_dir = os.path.join(DATA_ROOT, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        for sess in sessions:
            result.append((subj, sess))
    return result
...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. In `CONVERSION_NOTES.md`, the AI documented the directory layout as `data/{subject}/{session}/suite2p/plane0` plus `move_deve`, and in Step 5 it explicitly decided that each session equals one recording day for one mouse.

## 1-b. How are the data split into subjects?

i. The AI treats every directory directly inside `/app/data` as a subject and sorts those directory names alphabetically. It then builds a `subjects_list` from the unique subject names present in the `(subject, session)` pairs.

ii. 
```python
subjects = sorted([d for d in os.listdir(DATA_ROOT)
                  if os.path.isdir(os.path.join(DATA_ROOT, d))])
...
subjects_list = sorted(set(s[0] for s in all_sessions))
subjects_map = {s: i for i, s in enumerate(subjects_list)}
```

iii. The notes say the dataset contains 6 mice `jm031` through `jm046`, and the AI treated those top-level folders as the natural subject split. I did not find a more specific written justification than the observed folder structure.

## 1-c. How are the data split into sessions?

i. Each subdirectory inside a subject directory is treated as one session, sorted alphabetically. The AI also states in its notes that one session corresponds to one recording day.

ii. 
```python
for subj in subjects:
    subj_dir = os.path.join(DATA_ROOT, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        result.append((subj, sess))
```

iii. `CONVERSION_NOTES.md` Step 5 says `Session = recording day: Each session is one recording day for one mouse`, which is the explicit justification.

## 1-d. How are the data split into trials?

i. The AI does not use any natural trial markers. After binning to 3 Hz, it cuts each continuous session into non-overlapping 60-second windows, giving `180` time bins per trial, and drops any partial final trial.

ii. 
```python
binned_rate = FRAME_RATE / BIN_SIZE  # 3 Hz
timepoints_per_trial = int(TRIAL_DURATION * binned_rate)  # 180
n_binned_frames = dff_binned.shape[1]
n_trials = n_binned_frames // timepoints_per_trial
...
for t in range(n_trials):
    start = t * timepoints_per_trial
    end = start + timepoints_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. In the notes, the AI says the recordings are continuous spontaneous behavior sessions with no trial structure, so it split them into 60-second trials because that was required by the decoder task.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply an explicit trial-quality filter. The only effective filtering is structural: it keeps only complete 60-second trials because `n_trials` is computed with floor division, so any incomplete tail is discarded.

ii. 
```python
n_trials = n_binned_frames // timepoints_per_trial
...
for t in range(n_trials):
    start = t * timepoints_per_trial
    end = start + timepoints_per_trial
```

iii. The notes state that no trial curation was described in the paper and that sessions are continuous recordings, so the AI decided not to add any extra trial exclusion rule beyond fixed-length segmentation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `F.npy` and `Fneu.npy` in `suite2p/plane0`.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI explicitly maps `F.npy, Fneu.npy -> neural` and says it chose these because the paper described decoding from dF/F rather than from `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil subtraction with coefficient `0.7`, estimates a maximin baseline with a 60-second window and Gaussian smoothing, then computes a normalized dF/F ratio as `(Fc - F0) / F0`. This is its own manual reimplementation rather than a direct call to Suite2p's `dcnv.preprocess`.

ii. 
```python
def compute_baseline_maximin(Fc, win_baseline=WIN_BASELINE, sig_baseline=SIG_BASELINE, fs=FRAME_RATE):
    win = int(win_baseline * fs)
    if win % 2 == 0:
        win += 1
    ...
    for i in range(n_neurons):
        trace = Fc[i].astype(np.float32)
        smoothed = gaussian_filter1d(trace, sig_baseline)
        smoothed = minimum_filter1d(smoothed, win)
        smoothed = maximum_filter1d(smoothed, win)
        Flow[i] = smoothed
    return Flow

def compute_dff(F, Fneu):
    Fc = F.astype(np.float32) - NEUCOEFF * Fneu.astype(np.float32)
    F0 = compute_baseline_maximin(Fc)
    F0_safe = np.maximum(F0, 1e-6)
    dff = (Fc - F0) / F0_safe
    return dff
```

iii. The notes say the paper used dF/F and that the notebook suggested computing dF/F as described in the paper. In the trajectory, the AI explicitly debated whether Suite2p's preprocessing should be interpreted as baseline subtraction or ratio dF/F and decided to keep the normalized `(F - F0) / F0` form because it viewed that as the standard neuroscience meaning of dF/F.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no additional neuron-level filtering in code. It includes every row of `F.npy` and `Fneu.npy` and does not read `iscell.npy` during conversion.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
n_neurons, n_frames = F.shape
...
brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The notes justify this by saying the distributed dataset already contains only Track2p-tracked cells and that `iscell` is all ones, so no additional quality filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats the recording as continuous and effectively aligns trials to the session timeline rather than to a stimulus event. Each trial is a contiguous slice of the binned session, and the metadata says the alignment event is session start.

ii. 
```python
neural_trials.append(dff_binned[:, start:end].astype(np.float32))
...
'metadata': {
    'temporal_alignment_event': 'Session start (beginning of recording)',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION),
    ...
}
```

iii. The notes repeatedly say there is no native trial structure and that trials are artificial 60-second windows over spontaneous behavior, so no stimulus-locked realignment was used.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and motion-energy data by averaging every 10 original frames at 30 Hz, producing 3 Hz data with a bin size of about `333.33 ms`.

ii. 
```python
BIN_SIZE = 10    # frames per bin
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
...
bin_size_ms = (BIN_SIZE / FRAME_RATE) * 1000  # 333.33 ms
```

iii. In Step 5 of the notes, the AI says this exactly matches the paper's statement that decoding used averages over 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not use a raw timestamp variable for decoder input time. It derives time entirely from the binned sample index, the fixed frame rate, and the fixed bin size.

ii. 
```python
binned_rate = FRAME_RATE / BIN_SIZE  # 3 Hz
...
time_seconds = (np.arange(start, end) / binned_rate).astype(np.float32)
input_trials.append(time_seconds.reshape(1, -1))
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI explicitly maps `Time index -> input[0]` and describes it as elapsed seconds from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes time by taking each binned frame index within a session and dividing by the binned sample rate of 3 Hz. The result is a continuous time-within-session vector, expressed in seconds and stored as `float32`.

ii. 
```python
binned_rate = FRAME_RATE / BIN_SIZE  # 3 Hz
...
time_seconds = (np.arange(start, end) / binned_rate).astype(np.float32)
input_trials.append(time_seconds.reshape(1, -1))
```

iii. The AI's justification is implicit: since the imaging frame rate is fixed and the data are uniformly rebinned, a time axis can be reconstructed from index arithmetic without consulting `tstamps.npy`.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns time and neural data by deriving them from the same binned trial boundaries. For trial `t`, it uses the same `start:end` indices for the neural matrix and the time vector.

ii. 
```python
for t in range(n_trials):
    start = t * timepoints_per_trial
    end = start + timepoints_per_trial

    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    ...
    time_seconds = (np.arange(start, end) / binned_rate).astype(np.float32)
    input_trials.append(time_seconds.reshape(1, -1))
```

iii. The notes say time should be elapsed seconds from session start for each timepoint, and the trajectory notes that the "time input is linear as expected," which is how the AI justified that alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. In code, the AI derives motion energy only from `move_deve/motion_energy_glob.npy`. It inspects `tstamps.npy` and `interframe_int.npy` during exploration, but those arrays are not used by the final conversion script.

ii. 
```python
move_dir = os.path.join(sess_dir, 'move_deve')
...
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The notes say motion energy should be interpolated when camera frames are missing, but in the trajectory the AI says it was puzzled by the timestamp units and ultimately fell back to matching the motion-energy trace to the neural frame count without using the auxiliary timing arrays.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates the entire motion-energy trace to the neural frame count whenever the lengths differ, then averages into 10-frame bins, splits into trials, and finally discretizes the binned signal into five session-specific percentile bins.

ii. 
```python
def interpolate_motion_energy(me, n_target_frames):
    if len(me) == n_target_frames:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target_frames)
    me_interp = np.interp(x_target, x_orig, me)
    return me_interp
...
me = interpolate_motion_energy(me, n_frames)
...
me_binned = bin_data(me, BIN_SIZE)
...
me_discretized, bin_edges = discretize_motion_energy(me_trials, N_BINS)
```

iii. The notes justify interpolation by saying the README indicates missing camera frames. The trajectory adds that because the auxiliary timestamp units were confusing, the AI chose a simpler length-matching interpolation approach.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI concatenates all binned motion-energy values from one session, computes the 0th, 20th, 40th, 60th, 80th, and 100th percentiles for that session, widens the first and last edges to `-inf` and `inf`, and digitizes each trial into category labels `0` through `4`.

ii. 
```python
all_me = np.concatenate(me_binned_trials)
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_me, percentiles)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
...
for me_trial in me_binned_trials:
    binned = np.digitize(me_trial, bin_edges[1:-1])  # 0 to n_bins-1
    discretized.append(binned)
```

iii. In the notes, the AI calls this per-session quintile discretization and says it chose five equal-percentile bins because that was specified by the decoder task.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by forcing the motion-energy trace to the same frame count as the neural trace before binning, then slicing both with the same trial boundaries after binning.

ii. 
```python
me = interpolate_motion_energy(me, n_frames)
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
...
for t in range(n_trials):
    start = t * timepoints_per_trial
    end = start + timepoints_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. The AI justified this in its notes by saying missing camera frames must be repaired so that neural and motion-energy streams have equal length before shared binning and trialization.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mainly handles missing behavioral frames by linear interpolation of `motion_energy_glob.npy` to the neural frame count. It also silently drops leftover bins or leftover session tails that do not fill a complete 10-frame bin or a complete 60-second trial.

ii. 
```python
def interpolate_motion_energy(me, n_target_frames):
    if len(me) == n_target_frames:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target_frames)
    me_interp = np.interp(x_target, x_orig, me)
    return me_interp
...
trimmed = data[..., :n_bins * bin_size]
...
n_trials = n_binned_frames // timepoints_per_trial
```

iii. `CONVERSION_NOTES.md` Step 4 says the AI found sessions where motion energy was shorter than fluorescence and resolved that by interpolating motion energy before processing. In the trajectory it explicitly notes confusion about the timestamp units and chooses this simpler repair strategy.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant expensive step in the AI code is neural preprocessing, especially the per-neuron baseline estimation inside `compute_baseline_maximin`. Optional plotting in `--show-processing` also adds overhead, but only in debugging mode.

ii. 
```python
for i in range(n_neurons):
    trace = Fc[i].astype(np.float32)
    smoothed = gaussian_filter1d(trace, sig_baseline)
    smoothed = minimum_filter1d(smoothed, win)
    smoothed = maximum_filter1d(smoothed, win)
    Flow[i] = smoothed
...
t1 = time.time()
dff = compute_dff(F, Fneu)
t_dff = time.time() - t1
```

iii. The notes report full-pipeline timing and the runtime log printed by `process_session` breaks out `dF/F` versus binning time, which shows the AI was treating dF/F computation as the main cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest non-vectorized loop is the loop over neurons in `compute_baseline_maximin`. The per-trial motion-energy digitization loop could also be reduced, though it is less significant.

ii. 
```python
for i in range(n_neurons):
    trace = Fc[i].astype(np.float32)
    smoothed = gaussian_filter1d(trace, sig_baseline)
    smoothed = minimum_filter1d(smoothed, win)
    smoothed = maximum_filter1d(smoothed, win)
    Flow[i] = smoothed
...
for me_trial in me_binned_trials:
    binned = np.digitize(me_trial, bin_edges[1:-1])  # 0 to n_bins-1
    discretized.append(binned)
```

iii. I did not find an explicit written justification for keeping these loops. This answer is inferred directly from the implementation.

## 6-c. What processing does the code repeat multiple times?

i. The AI does not intentionally rerun whole-session preprocessing, but it does make repeated passes over the motion-energy trials: first to concatenate them for percentile estimation, then again to digitize each trial. In `--show-processing` mode it also carries additional processed arrays into plotting for visual checks.

ii. 
```python
all_me = np.concatenate(me_binned_trials)
...
for me_trial in me_binned_trials:
    binned = np.digitize(me_trial, bin_edges[1:-1])  # 0 to n_bins-1
    discretized.append(binned)
...
if show_processing:
    plot_processing(subj, sess, F, Fneu, dff, me, dff_binned, me_binned,
                   neural_trials, me_trials, me_discretized, bin_edges,
                   input_trials, session_idx)
```

iii. There is no explicit justification in the notes beyond the fact that plotting and sanity checks were intentionally added for verification.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI added an optional visualization path that produces processing plots but is not used by the downstream decoder. It also computes `bin_edges` for motion-energy discretization, uses them locally for digitization and plotting, and then discards them instead of saving them in the dataset.

ii. 
```python
me_discretized, bin_edges = discretize_motion_energy(me_trials, N_BINS)
...
if show_processing:
    plot_processing(subj, sess, F, Fneu, dff, me, dff_binned, me_binned,
                   neural_trials, me_trials, me_discretized, bin_edges,
                   input_trials, session_idx)
...
return neural_trials, input_trials, output_trials, n_neurons
```

iii. The notes justify this as visual verification and sanity checking rather than as part of the final dataset format.
