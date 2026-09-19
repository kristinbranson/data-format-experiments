# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every subject directory in `/app/data` whose name starts with `jm`, then enumerates every subdirectory inside each subject as a session. For each session it loads neural fluorescence from `suite2p/plane0/F.npy` and `Fneu.npy`, and behavioral data from `move_deve/motion_energy_glob.npy` plus `move_deve/tstamps.npy`. Trials are not loaded directly from disk; the session is loaded first and trials are produced later by splitting the processed session arrays.

ii. 
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
...
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. In the trajectory, the AI first explored `/app/data`, listed all `jm*` directories and date-stamped session folders, then inspected one session to confirm the available files and shapes. It justified this loader structure by treating the directory layout as the canonical dataset organization.

## 1-b. How are the data split into subjects?

i. Subjects are defined as top-level directories in `/app/data` whose names start with `jm`, sorted lexicographically.

ii. 
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

iii. In the trajectory, the AI listed the dataset root, saw six `jm*` folders, and treated each as one mouse. That directly drove the subject split.

## 1-c. How are the data split into sessions?

i. Sessions are defined as all subdirectories within each subject directory, sorted lexicographically. Each dated folder is treated as one recording session.

ii. 
```python
subj_dir = os.path.join(data_dir, subj)
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
...
for sess_name in sessions:
    sess_dir = os.path.join(subj_dir, sess_name)
```

iii. The AI inspected several subject folders, saw one subdirectory per day, and concluded that each dated folder is one daily recording. Sorting was used to keep the session order deterministic.

## 1-d. How are the data split into trials?

i. The AI does not use 60-second trials. It bins each continuous session first and then splits the binned session into consecutive non-overlapping 2-minute trials. Trial tails shorter than 2 minutes are discarded implicitly because `n_trials` is computed with floor division.

ii. 
```python
TRIAL_DURATION_S = 120  # 2 minutes per trial
...
def split_into_trials(dff_binned, me_binned, time_bins, trial_duration_s=TRIAL_DURATION_S):
    bins_per_trial = int(trial_duration_s * FS / BIN_SIZE)  # 120 * 30 / 10 = 360
    n_bins = dff_binned.shape[1]
    n_trials = n_bins // bins_per_trial
    ...
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end])
        me_trials.append(me_binned[start:end])
        time_trials.append(time_bins[start:end])
```

iii. The trajectory explicitly says the AI chose 2-minute blocks because it believed this matched the paper’s cross-validation scheme, quoting “consecutive 2 minute blocks” in its notes and docstring.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not implement any trial-level quality-control filter. Trials are only omitted when they do not fit evenly into a 2-minute block at the end of a session.

ii. 
```python
n_trials = n_bins // bins_per_trial
...
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
```

iii. There is no explicit trial-QC discussion in the trajectory. The AI appears to have assumed that once sessions were loaded and motion-energy length mismatches were corrected, all resulting trials should be kept.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from suite2p fluorescence arrays `F.npy` and `Fneu.npy` from `suite2p/plane0`.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu)
```

iii. The AI inspected one session’s suite2p outputs and saw `F.npy`, `Fneu.npy`, `spks.npy`, and `iscell.npy`. It chose `F` and `Fneu` because it wanted to reproduce Suite2p-style fluorescence preprocessing rather than use `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The AI computes a manual dF/F-like signal. It subtracts neuropil (`F - 0.7 * Fneu`), estimates a baseline with a Gaussian smoothing step followed by running minimum and running maximum filters, clamps the baseline away from zero, divides by the baseline to produce `(Fc - F0) / F0`, and finally bins the result in 10-frame averages.

ii. 
```python
def compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF,
                win_baseline=WIN_BASELINE, sig_baseline=SIG_BASELINE):
    Fc = F - neucoeff * Fneu

    win = int(win_baseline * fs)
    sig = int(sig_baseline * fs)

    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)

    Flow = np.maximum(Flow, 1e-6)
    dff = (Fc - Flow) / Flow
    return dff.astype(np.float32)
...
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. In the trajectory, the AI checked `ops.npy`, extracted Suite2p defaults (`neucoeff=0.7`, `baseline='maximin'`, `win_baseline=60`, `sig_baseline=10`), and justified this as reproducing the paper’s “Suite2p default parameters.” It then implemented those operations directly with SciPy filters instead of calling Suite2p’s preprocessing function.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality control is applied. All rows in `F.npy` are kept.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu)
```

iii. The AI inspected `iscell.npy` for one session and saw only `1`s in the first column, which likely reinforced its choice not to add ROI filtering. No later code uses `iscell.npy`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural trials are aligned to recording onset or session start. After session-level binning, the AI slices contiguous chunks in time; there is no separate event-triggered alignment.

ii. 
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
...
neural_trials.append(dff_binned[:, start:end])
...
'metadata': {
    'temporal_alignment_event': 'recording_onset',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_S),
}
```

iii. The trajectory and conversion notes say the data are continuous spontaneous-behavior recordings, so the AI treated recording onset as the only natural alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use non-overlapping 10-frame bins at 30 Hz, so the final time step is `10 / 30 = 0.333...` seconds, or 333.33 ms. Both neural and motion-energy data are rebinned this way.

ii. 
```python
BIN_SIZE = 10
FS = 30
...
def bin_data(data, bin_size):
    n = data.shape[-1]
    n_bins = n // bin_size
    n_use = n_bins * bin_size
    ...
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
...
'time_bin_size': BIN_SIZE / FS * 1000,
```

iii. The AI repeatedly justified this from the paper text it quoted in its notes: decoding analyses used averages over 10 consecutive timestamps. That drove the 333.33 ms bin size.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from a raw time variable. It is synthesized from the bin index, `BIN_SIZE`, and imaging frame rate `FS`.

ii. 
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
...
for t_trial in sess_data['time_trials']:
    sess_input.append(t_trial.reshape(1, -1))
```

iii. In its notes, the AI justified this by saying the decoder input should be “time elapsed from recording onset,” so with a fixed 30 Hz sampling rate it computed time analytically rather than reading timestamps from file.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one scalar per binned frame using the center of each 10-frame bin: `(bin_index * 10 + 5) / 30`. Those per-session time values are then sliced into trials and reshaped to `(1, n_timepoints)` per trial.

ii. 
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
...
time_trials.append(time_bins[start:end])
...
sess_input.append(t_trial.reshape(1, -1))
```

iii. The trajectory explicitly says the AI chose “time in seconds from recording onset” and, in its notes, described these values as bin-center times.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Time is created at the same binned-session resolution as neural activity, then split into trials with the same `start:end` indices, so every neural bin has one time value.

ii. 
```python
dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
neural_trials, me_trials, time_trials = split_into_trials(
    dff_binned, me_binned, time_bins
)
...
neural_trials.append(dff_binned[:, start:end])
time_trials.append(time_bins[start:end])
```

iii. The AI justified alignment by constructing time from the same binned indexing scheme it used for neural data, avoiding a separate resynchronization step.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion-energy output is derived primarily from `move_deve/motion_energy_glob.npy`, with `move_deve/tstamps.npy` used to infer dropped camera frames before alignment to imaging frames.

ii. 
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
...
me = interpolate_motion_energy(me, tstamps, n_frames)
```

iii. In the trajectory, the AI inspected the `move_deve` directory, saw both files, and decided `motion_energy_glob.npy` was the precomputed signal while `tstamps.npy` could be used to repair length mismatches caused by dropped video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI optionally interpolates motion energy onto the imaging-frame grid using timestamp gaps, bins the repaired trace in non-overlapping 10-frame means, pools all trial traces from all sessions to compute global percentile thresholds, and later discretizes each trial using those same global thresholds.

ii. 
```python
def interpolate_motion_energy(me, tstamps, n_frames):
    if len(me) == n_frames:
        return me
    ifi = np.diff(tstamps)
    median_ifi = np.median(ifi)
    frame_indices = np.zeros(len(me), dtype=int)
    ...
    me_interp = np.interp(all_indices, frame_indices, me)
    return me_interp
...
me_binned = bin_data(me, BIN_SIZE)
...
for me_t in me_trials:
    all_me_flat.append(me_t)
...
all_me_concat = np.concatenate(all_me_flat)
thresholds = discretize_motion_energy(all_me_concat)
...
me_disc = apply_discretization(me_trial, thresholds)
```

iii. The trajectory shows two explicit justifications: the AI investigated frame-count mismatches and concluded they were single dropped frames spread through the session, so interpolation was acceptable; later it said global percentile binning was acceptable even though it skewed some sessions because it still yielded useful decoder signal.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI thresholds motion energy into five global percentile bins. It concatenates all motion-energy trial values across all sessions, computes quintile thresholds once, and applies those thresholds to every session and trial.

ii. 
```python
def discretize_motion_energy(all_me_values, n_bins=N_BINS_OUTPUT):
    percentiles = np.linspace(0, 100, n_bins + 1)
    thresholds = np.percentile(all_me_values, percentiles)
    return thresholds
...
all_me_concat = np.concatenate(all_me_flat)
thresholds = discretize_motion_energy(all_me_concat)
...
me_disc = apply_discretization(me_trial, thresholds)
```

iii. In the trajectory, the AI explicitly noticed that some sessions became skewed under global thresholds, but kept the design because it believed “global percentile binning” gave consistent bins across the dataset and was “fine for the decoder.”

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by first interpolating the motion trace to the neural frame count when needed, then applying the same 10-frame binning, then slicing trials with the same `start:end` indices used for neural activity.

ii. 
```python
n_neurons, n_frames = F.shape
...
me = interpolate_motion_energy(me, tstamps, n_frames)
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
...
neural_trials.append(dff_binned[:, start:end])
me_trials.append(me_binned[start:end])
```

iii. The AI justified this alignment in the trajectory by checking sessions with shorter motion-energy arrays, inferring they reflected dropped camera frames, and then interpolating so the behavioral and neural streams could be indexed together.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main error-handling logic is for missing motion-energy samples. If the motion-energy array is shorter than the neural recording, the AI infers dropped frames from timestamp gaps and fills the missing positions by interpolation. It also drops partial leftovers created by binning or by trial splitting because both use floor division.

ii. 
```python
if len(me) == n_frames:
    return me
...
ifi = np.diff(tstamps)
median_ifi = np.median(ifi)
...
me_interp = np.interp(all_indices, frame_indices, me)
...
n_bins = n // bin_size
n_use = n_bins * bin_size
...
n_trials = n_bins // bins_per_trial
```

iii. The trajectory shows the AI inspecting several mismatch cases, concluding that the missing values were isolated dropped frames, and choosing interpolation as the repair strategy. It did not add explicit assertions or fallback handling beyond that.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive computation is the neural preprocessing in `compute_dff`, especially the Gaussian filter and the large-window minimum/maximum filters applied across every neuron and frame. Secondary costs come from session loading and motion-energy interpolation.

ii. 
```python
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

iii. The trajectory does not state this explicitly, but the AI’s implementation makes the computational hotspot clear: these full-session filtering passes dominate the per-session work.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the loop that reconstructs `frame_indices` from timestamp gaps in `interpolate_motion_energy`. Additional Python loops iterate over trials when splitting sessions and when applying discretization to each trial.

ii. 
```python
for i in range(len(ifi)):
    n_skipped = int(np.round(ifi[i] / median_ifi))
    cum_idx += n_skipped
    frame_indices[i + 1] = cum_idx
...
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])
    me_trials.append(me_binned[start:end])
    time_trials.append(time_bins[start:end])
...
for me_trial in sess_data['me_trials']:
    me_disc = apply_discretization(me_trial, thresholds)
```

iii. The trajectory does not call this out directly, but these are the obvious Python loops in the implementation, and the timestamp-gap loop is especially unnecessary because the same result could be built with vectorized cumulative operations.

## 6-c. What processing does the code repeat multiple times?

i. The AI uses multiple passes over motion-energy data. It first splits each session into trial traces and stores every trial in `all_me_flat`, then concatenates all trials to compute one set of global thresholds, then iterates over all sessions and trials again to discretize them. It also repeatedly calls `subjects.index(subj)` inside the session loop.

ii. 
```python
for me_t in me_trials:
    all_me_flat.append(me_t)
...
all_me_concat = np.concatenate(all_me_flat)
thresholds = discretize_motion_energy(all_me_concat)
...
for sess_data in all_sessions_data:
    subj = sess_data['subject']
    subj_i = subjects.index(subj)
    ...
    for me_trial in sess_data['me_trials']:
        me_disc = apply_discretization(me_trial, thresholds)
```

iii. There is no explicit trajectory justification for this repetition. It follows from the AI’s decision to use one global threshold set computed after all sessions had already been split into trials.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion script does extra work that is not needed for the full converted dataset used by downstream decoding: it runs `print_sanity_checks`, creates a separate `sample_data.pkl`, and computes a sample subset in `create_sample`. Those steps do not change `converted_data.pkl`.

ii. 
```python
def print_sanity_checks(data):
    ...

def create_sample(data, max_sessions=6):
    ...

data = convert_data()
print_sanity_checks(data)
...
sample = create_sample(data)
sample_path = os.path.join('/app', 'sample_data.pkl')
with open(sample_path, 'wb') as f:
    pickle.dump(sample, f)
```

iii. The trajectory shows the AI deliberately adding these steps for validation and quick testing while waiting for decoder runs. They were convenience outputs rather than part of the core required conversion.
