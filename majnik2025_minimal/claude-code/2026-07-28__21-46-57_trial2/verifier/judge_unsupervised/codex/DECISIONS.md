# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads data by scanning `/app/data` for subject directories whose names start with `jm`, then scanning every subdirectory of each subject as a session. For each session it loads neural traces from `suite2p/plane0/F.npy` and `Fneu.npy`, and behavioral traces from `move_deve/motion_energy_glob.npy` plus `tstamps.npy`. It processes each session immediately and stores the resulting per-session trialized data in `all_sessions_data`.

ii. 
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])

    for sess_name in sessions:
        sess_dir = os.path.join(subj_dir, sess_name)
        dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
```

```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. `CONVERSION_NOTES.md` says there are 6 subjects and 41 sessions. The trajectory shows the agent first inspected `/app/data`, the session directory layout, and `load_data.ipynb`, and concluded that each subject folder contains daily session folders with `suite2p` neural outputs and `move_deve` behavioral outputs.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined purely by the top-level directory names under `/app/data` that start with `jm`. They are sorted lexicographically and written directly to `data['subjects']`. Each session gets a `subject_idx` equal to the position of its subject name in that sorted list.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

```python
for sess_data in all_sessions_data:
    subj = sess_data['subject']
    subj_i = subjects.index(subj)
    subject_idx.append(subj_i)

data = {
    'subjects': subjects,
    'subject_idx': np.array(subject_idx, dtype=int),
}
```

iii. The data README says the dataset is organized into 6 subject folders and that `jm031` through `jm046` correspond to mice A-F. `CONVERSION_NOTES.md` repeats the same six subject IDs.

## 1-c. How are the data split into sessions?

i. Sessions are defined as all subdirectories inside each subject directory. The script sorts those subdirectories and treats each one as a separate session. Each session becomes one entry in the outer list of `neural`, `input`, `output`, and `brain_region_idx`.

ii.
```python
for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])

    for sess_name in sessions:
        sess_dir = os.path.join(subj_dir, sess_name)
        ...
        all_sessions_data.append({
            'subject': subj,
            'session': sess_name,
            'neural_trials': neural_trials,
            'me_trials': me_trials,
            'time_trials': time_trials,
            'n_neurons': n_neurons,
        })
```

iii. The data README says each subject folder contains session subfolders corresponding to recording days, and that the `_a` suffix can be ignored. The trajectory shows the agent listing the per-mouse directories and treating each recording-day folder as one session.

## 1-d. How are the data split into trials?

i. The dataset is continuous, so the script invents trials by splitting each session into consecutive 2-minute blocks after temporal binning. With 30 Hz imaging and 10-frame bins, each trial contains 360 time bins. Twenty-minute sessions become 10 trials, and 30-minute sessions become 15 trials.

ii.
```python
TRIAL_DURATION_S = 120  # 2 minutes per trial
BIN_SIZE = 10
FS = 30
```

```python
def split_into_trials(dff_binned, me_binned, time_bins, trial_duration_s=TRIAL_DURATION_S):
    bins_per_trial = int(trial_duration_s * FS / BIN_SIZE)  # 120 * 30 / 10 = 360
    n_bins = dff_binned.shape[1]
    n_trials = n_bins // bins_per_trial

    neural_trials = []
    me_trials = []
    time_trials = []

    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end])
        me_trials.append(me_binned[start:end])
        time_trials.append(time_bins[start:end])
```

iii. In both `CONVERSION_NOTES.md` and the trajectory, the agent justified this by citing the paper’s decoder cross-validation rule: splits were done on consecutive 2-minute blocks. It reused that cross-validation chunk size as the conversion-time trial definition.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level quality control. Every full 2-minute block is kept. The only implicit filtering is that partial trailing bins or partial trailing trials would be dropped by integer floor division, and missing behavioral frames are interpolated before trialization rather than causing trial exclusion.

ii.
```python
if len(me) == n_frames:
    return me
...
me = interpolate_motion_energy(me, tstamps, n_frames)
```

```python
n_bins = dff_binned.shape[1]
n_trials = n_bins // bins_per_trial
...
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
```

iii. `CONVERSION_NOTES.md` discusses interpolation for dropped camera frames, but it does not mention excluding any trials. The trajectory shows the agent checking missing-frame patterns and deciding to repair them by interpolation instead of dropping affected data blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw fluorescence traces `F.npy` and neuropil traces `Fneu.npy`. The script does not use `spks.npy`; it computes its own dF/F from `F` and `Fneu`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu)
```

```python
def compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF,
                win_baseline=WIN_BASELINE, sig_baseline=SIG_BASELINE):
    Fc = F - neucoeff * Fneu
    ...
    dff = (Fc - Flow) / Flow
```

iii. The methods excerpt captured in the trajectory says, “We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses.” `CONVERSION_NOTES.md` repeats that justification and lists the Suite2p defaults it extracted from `ops.npy`.

## 2-b. How is the `neural` data processed?

i. Neural traces are neuropil-corrected, converted to dF/F with Suite2p’s maximin baseline procedure, cast to `float32`, and then averaged in 10-frame bins. The resulting binned dF/F arrays are what get stored in `data['neural']`.

ii.
```python
Fc = F - neucoeff * Fneu

win = int(win_baseline * fs)
sig = int(sig_baseline * fs)

Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)

Flow = np.maximum(Flow, 1e-6)
dff = (Fc - Flow) / Flow

return dff.astype(np.float32)
```

```python
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The trajectory contains the methods text about using Suite2p default parameters for dF/F and averaging dF/F in bins of 10 consecutive timestamps. `CONVERSION_NOTES.md` explicitly cites the same two decisions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script performs no additional neural filtering. It does not read or threshold `iscell.npy`, and it does not remove neurons, sessions, or mice. It assumes the provided Track2p/Suite2p outputs are already curated and matched across days.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))

n_neurons, n_frames = F.shape
dff = compute_dff(F, Fneu)
```

iii. The data README says these files already contain traces only for neurons present across all days, with rows matched across sessions. The trajectory also shows the agent checking `iscell.npy` and finding only `1.0` values, then relying on the prefiltered tracked-cell output.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to recording onset, not to a behavioral event within each block. The script keeps the continuous session time base, bins the session uniformly, and then slices neural/activity data into matching 2-minute windows. The metadata names the alignment event `recording_onset`.

ii.
```python
dff_binned = bin_data(dff, BIN_SIZE)
...
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])
```

```python
'metadata': {
    'temporal_alignment_event': 'recording_onset',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_S),
}
```

iii. `CONVERSION_NOTES.md` says the decoder input is “Time in seconds from recording onset,” and the trajectory shows the agent reasoning that these are continuous spontaneous recordings, so it treated recording start as the only available alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data have 10-frame time bins. At 30 Hz this is 333.33 ms per bin. Yes, temporal rebinning is applied by averaging each consecutive 10-frame block.

ii.
```python
BIN_SIZE = 10
FS = 30
```

```python
def bin_data(data, bin_size):
    n = data.shape[-1]
    n_bins = n // bin_size
    n_use = n_bins * bin_size
    ...
    return data[:, :n_use].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)
```

```python
'time_bin_size': BIN_SIZE / FS * 1000,
```

iii. The methods excerpt captured in the trajectory says the paper “slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps.” `CONVERSION_NOTES.md` quotes the same sentence.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time variable is not loaded from a raw timestamp file. It is derived from the implied imaging frame index and the constant imaging rate `FS = 30`, after converting those frames into 10-frame bins.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

```python
for t_trial in sess_data['time_trials']:
    sess_input.append(t_trial.reshape(1, -1))
```

iii. `CONVERSION_NOTES.md` explicitly says the input is “Computed as bin center times: `(bin_index * 10 + 5) / 30` seconds.” The trajectory repeats that the decoder input should be time elapsed from recording onset.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The script creates one time value per 10-frame bin, using the center of each bin in seconds. It then splits that session-long time vector into the same 2-minute blocks as neural and behavioral data and reshapes each block to `(1, n_timepoints)`.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    time_trials.append(time_bins[start:end])
```

```python
for t_trial in sess_data['time_trials']:
    sess_input.append(t_trial.reshape(1, -1))
```

iii. The notes say the decoder input is time from recording onset, and the trajectory shows the agent choosing bin-center times because both neural and behavior traces had already been binned in 10-frame windows.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Input time is aligned by construction: the script computes the time vector from the same binned neural frame grid, then slices time and neural arrays with identical trial boundaries. The time values remain absolute within the session; they do not reset to zero at each 2-minute pseudo-trial.

ii.
```python
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])
    me_trials.append(me_binned[start:end])
    time_trials.append(time_bins[start:end])
```

iii. The agent’s notes describe the input as bin-center times from recording onset, and the trajectory shows it reusing the same 2-minute block boundaries for all three streams.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. In the script, output motion energy is loaded directly from `move_deve/motion_energy_glob.npy`, with `tstamps.npy` used to repair length mismatches. The agent’s written justification says this file is already the paper’s motion-energy quantity, derived upstream from consecutive videography frames by differencing, squaring, and summing pixels.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
me = interpolate_motion_energy(me, tstamps, n_frames)
```

iii. The methods excerpt in the trajectory says motion energy was computed from pixel-wise differences of consecutive video frames, squared and summed across pixels. The data README says `motion_energy_glob.npy` contains that processed behavioral output.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The script optionally interpolates motion energy when video frames are missing, then averages it in the same 10-frame bins as neural data. It does not perform any explicit normalization before discretization.

ii.
```python
def interpolate_motion_energy(me, tstamps, n_frames):
    if len(me) == n_frames:
        return me
    ...
    me_interp = np.interp(all_indices, frame_indices, me)
    return me_interp
```

```python
me = interpolate_motion_energy(me, tstamps, n_frames)
me_binned = bin_data(me, BIN_SIZE)
```

iii. `CONVERSION_NOTES.md` says dropped frames are repaired by detecting gaps in `tstamps.npy` and linearly interpolating. It also cites the paper’s 10-timestamp averaging step. The notes say the task required normalized and discretized motion energy, but the code only implements interpolation, binning, and later discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The script concatenates all binned motion-energy values from every session and trial, computes quintile thresholds on that global pool, and digitizes each value into one of five categories labeled `bin_0` through `bin_4`.

ii.
```python
def discretize_motion_energy(all_me_values, n_bins=N_BINS_OUTPUT):
    percentiles = np.linspace(0, 100, n_bins + 1)
    thresholds = np.percentile(all_me_values, percentiles)
    return thresholds
```

```python
all_me_concat = np.concatenate(all_me_flat)
thresholds = discretize_motion_energy(all_me_concat)
```

```python
def apply_discretization(me_values, thresholds):
    n_bins = len(thresholds) - 1
    binned = np.digitize(me_values, thresholds[1:-1])
    binned = np.clip(binned, 0, n_bins - 1)
    return binned
```

iii. `CONVERSION_NOTES.md` explicitly says the agent chose global quintile thresholds so that bin definitions stay consistent across sessions and classes are exactly 20% each globally. The trajectory later comments on the resulting skew in some individual sessions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data by first interpolating it to the neural frame count when necessary, then binning both streams with the same 10-frame windows, then slicing both with the same 2-minute trial boundaries. Each trial therefore has matching neural and output time axes.

ii.
```python
me = interpolate_motion_energy(me, tstamps, n_frames)

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])
    me_trials.append(me_binned[start:end])
```

```python
for me_trial in sess_data['me_trials']:
    me_disc = apply_discretization(me_trial, thresholds)
    sess_output.append(me_disc.reshape(1, -1))
```

iii. The trajectory shows the agent investigating sessions with dropped camera frames and deciding to interpolate motion energy onto the 2-photon frame grid before any binning or trial splitting.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missing behavioral frames by linear interpolation using timestamp gaps. It guards against zero or tiny dF/F baselines by clamping the baseline to `1e-6`. It would silently discard any trailing partial bins or partial trials via integer floor division. It does not perform any other explicit repair of missing or malformed data.

ii.
```python
Flow = np.maximum(Flow, 1e-6)
dff = (Fc - Flow) / Flow
```

```python
if len(me) == n_frames:
    return me
...
me_interp = np.interp(all_indices, frame_indices, me)
```

```python
n_bins = n // bin_size
n_use = n_bins * bin_size
...
n_trials = n_bins // bins_per_trial
```

iii. `CONVERSION_NOTES.md` says some recordings have dropped camera frames and describes interpolation as the chosen fix. The trajectory shows the agent checking a session with 116 missing frames and explicitly deciding to interpolate rather than exclude data.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading the full session arrays for all 41 sessions, computing Suite2p-style dF/F over every neuron and frame using Gaussian/minimum/maximum filters, and then building the full-session/trial data structures before the global motion-energy discretization pass.

ii.
```python
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

```python
for subj in subjects:
    ...
    for sess_name in sessions:
        dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
        neural_trials, me_trials, time_trials = split_into_trials(
            dff_binned, me_binned, time_bins
        )
```

iii. This follows directly from the code structure: those operations touch the full neuron-by-frame arrays for every session. The trajectory also shows the agent inspecting large full-session arrays and treating the full conversion/training runs as long jobs.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped-frame reconstruction loop in `interpolate_motion_energy`, the trial-splitting loop in `split_into_trials`, the loop that appends each motion-energy trial to `all_me_flat`, the per-trial input/output construction loops, and the repeated `subjects.index(subj)` lookup could all be vectorized or precomputed.

ii.
```python
for i in range(len(ifi)):
    n_skipped = int(np.round(ifi[i] / median_ifi))
    cum_idx += n_skipped
    frame_indices[i + 1] = cum_idx
```

```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])
    me_trials.append(me_binned[start:end])
    time_trials.append(time_bins[start:end])
```

```python
for me_t in me_trials:
    all_me_flat.append(me_t)
...
for t_trial in sess_data['time_trials']:
    sess_input.append(t_trial.reshape(1, -1))
for me_trial in sess_data['me_trials']:
    me_disc = apply_discretization(me_trial, thresholds)
```

iii. These loops are mostly repetitive indexing and reshaping over large arrays, so they are natural candidates for vectorized array operations or precomputed index maps.

## 6-c. What processing does the code repeat multiple times?

i. The script does a two-pass motion-energy workflow: it first processes all sessions and collects every motion-energy trial into `all_me_flat` so it can compute global thresholds, then it loops over all sessions and all motion-energy trials again to apply discretization and build the final output structure. It also stores session data in `all_sessions_data` and revisits it instead of streaming directly into the final dictionary.

ii.
```python
all_sessions_data = []
all_me_flat = []
...
all_sessions_data.append({
    'subject': subj,
    'session': sess_name,
    'neural_trials': neural_trials,
    'me_trials': me_trials,
    'time_trials': time_trials,
    'n_neurons': n_neurons,
})

for me_t in me_trials:
    all_me_flat.append(me_t)
```

```python
all_me_concat = np.concatenate(all_me_flat)
thresholds = discretize_motion_energy(all_me_concat)
...
for sess_data in all_sessions_data:
    ...
    for me_trial in sess_data['me_trials']:
        me_disc = apply_discretization(me_trial, thresholds)
```

iii. `CONVERSION_NOTES.md` justifies the second pass by saying global discretization should be consistent across sessions. That choice forces the script to revisit the processed trials after collecting all motion-energy values.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script keeps `session` names inside `all_sessions_data` but never saves them into the output pickle. `print_sanity_checks` computes statistics only for stdout. `create_sample` and its save path are auxiliary to the main converted dataset. There are also unused pieces of code context such as `sample_mode`, `n_2p_frames`, and the `sys` import.

ii.
```python
def load_and_process_session(session_dir, n_2p_frames=None):
```

```python
all_sessions_data.append({
    'subject': subj,
    'session': sess_name,
    'neural_trials': neural_trials,
    'me_trials': me_trials,
    'time_trials': time_trials,
    'n_neurons': n_neurons,
})
```

```python
def print_sanity_checks(data):
    ...

def create_sample(data, max_sessions=6):
    ...
```

iii. These steps do not change what `train_decoder.py` consumes from `converted_data.pkl`. They are useful for logging or auxiliary outputs, but they are discarded for the downstream decoder analysis itself.
