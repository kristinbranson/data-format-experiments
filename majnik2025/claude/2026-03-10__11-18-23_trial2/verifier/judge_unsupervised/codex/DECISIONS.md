# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the six subject IDs, iterates over them in order, lists each subject's session directories with `get_sessions()`, and processes each session with `process_session()`. `process_session()` loads `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `tstamps.npy`, computes neural and motion-energy time series for the entire session, and stores each processed session for a second pass that later splits it into trials.

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

def process_session(session_dir, device=None):
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    me_dir = os.path.join(session_dir, 'move_deve')
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
    tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

```python
subjects_to_process = SUBJECTS
for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
    sessions = get_sessions(subject_dir)
    for sess_dir in sessions:
        dff_binned, me_binned, n_neurons, n_frames_raw = process_session(
            sess_dir, device=device
        )
        session_data.append((subj_idx, sess_dir, dff_binned, me_binned, n_neurons))
```

iii. In `CONVERSION_NOTES.md` Step 2, the agent documented six subject folders and 41 total sessions, then in Step 5 said it would "use all available data." The trajectory summary also states the final dataset contains 6 subjects and 41 sessions.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level mouse directory, using the hard-coded `SUBJECTS` list. The session order in the final output follows that list, and `subject_idx` stores the index of each session's subject in that list.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
```

```python
all_neural.append(neural_trials)
all_input.append(input_trials)
all_output.append(output_trials)
subject_idx_list.append(subj_idx)

data = {
    ...
    'subjects': SUBJECTS if not sample else SUBJECTS[:1],
    'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. The notes say the subject folders correspond to mice A-F and that the subject IDs are in alphabetical order. The agent reused that ordering directly instead of discovering subjects dynamically.

## 1-c. How are the data split into sessions?

i. Within each subject directory, sessions are defined as the sorted child directories. Each session directory is processed independently and becomes one entry in `neural`, `input`, `output`, and `brain_region_idx`.

ii.
```python
def get_sessions(subject_dir):
    sessions = []
    for d in sorted(os.listdir(subject_dir)):
        full = os.path.join(subject_dir, d)
        if os.path.isdir(full) and not d.startswith('.'):
            sessions.append(full)
    return sessions
```

```python
sessions = get_sessions(subject_dir)
for sess_dir in sessions:
    dff_binned, me_binned, n_neurons, n_frames_raw = process_session(
        sess_dir, device=device
    )
```

iii. `CONVERSION_NOTES.md` Step 2 describes each session folder as one recording day under each subject, and the notebook in `data/load_data.ipynb` also sorts session directories chronologically.

## 1-d. How are the data split into trials?

i. The agent first bins each full-session neural and motion-energy trace into 10-frame averages. It then defines trials as consecutive 2-minute blocks of those binned traces. With `FS = 30` and `BIN_SIZE = 10`, each trial is 360 time bins long.

ii.
```python
BIN_SIZE = 10
TRIAL_DURATION_S = 120.0
```

```python
def split_into_trials(dff_binned, me_binned):
    bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360
    n_bins = dff_binned.shape[1]
    n_trials = n_bins // bins_per_trial

    neural_trials = []
    me_trials = []
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])
```

iii. The notes explicitly justify this by citing the paper's decoding description: "splits were done on consecutive 2 minute blocks of the recording." In Step 5 the agent again wrote "Each session split into 2-minute blocks (matching paper's CV structure)."

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. The only implicit filtering is that incomplete trailing data are dropped because the number of trials is computed with floor division.

ii.
```python
bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360
n_bins = dff_binned.shape[1]
n_trials = n_bins // bins_per_trial
```

iii. In Step 3 of the notes, the agent wrote "No explicit trial curation mentioned - continuous recording." It therefore preserved all full blocks and did not reject trials by behavioral or signal-quality criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives `neural` from Suite2p fluorescence traces `F.npy` and `Fneu.npy` in each session's `suite2p/plane0` directory.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu, device=device)
```

iii. The notes map `F.npy, Fneu.npy -> neural` and cite the data notebook plus Suite2p defaults as the basis for that choice.

## 2-b. How is the `neural` data processed?

i. The agent computes an explicit dF/F. It subtracts neuropil (`F - 0.7 * Fneu`), calls Suite2p `preprocess()` to estimate a maximin baseline, reconstructs that baseline from the returned baseline-subtracted trace, divides by the baseline to obtain dF/F, clips the baseline to avoid division by zero, casts to `float32`, and then averages in 10-frame bins.

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
return dff.astype(np.float32)
```

```python
dff = compute_dff(F, Fneu, device=device)
dff_binned = bin_traces(dff, BIN_SIZE)
```

iii. The code docstring says the paper used "baseline corrected fluorescence traces as our dF/F," and Step 5 of the notes says the agent decided to "Use Suite2p's preprocess (baseline_maximin) with default params from ops.npy" to implement that.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent applies no extra neuron filtering in `convert_data.py`. It assumes the provided Track2p/Suite2p outputs already contain only the tracked cells to keep.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
n_neurons, n_frames = F.shape
```

iii. In Step 1 and Step 3 of the notes, the agent states that the data "already contains only neurons tracked across all days" and that `iscell.npy` is already all ones, so "no additional filtering needed."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns neural activity to the start of each artificial trial block cut from the continuous recording. The metadata names the alignment event as "Start of recording session," but in practice each returned trial is just a consecutive 2-minute slice, not an event-centered peri-event window.

ii.
```python
def split_into_trials(dff_binned, me_binned):
    ...
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end].astype(np.float32))
```

```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': None,
```

iii. The notes justify this by saying the recordings are continuous and that the paper used consecutive blocks for decoding. The agent did not identify any more specific event in the raw data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 333.33 ms bins. Yes: both neural and motion-energy traces are rebinned by averaging every 10 frames at 30 Hz.

ii.
```python
FS = 30.0
BIN_SIZE = 10
```

```python
def bin_traces(data, bin_size):
    ...
    return trimmed.reshape(...).mean(axis=...)
```

```python
time_bin_ms = BIN_SIZE / FS * 1000  # ms
...
'time_bin_size': time_bin_ms,
```

iii. Step 3 and Step 5 of the notes both cite the paper line about "averaging in bins of 10 consecutive timestamps" and conclude the time bin should be 10 frames at 30 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not read from a raw file. It is synthesized from the trial length and sampling constants, using `np.arange(n_timebins)` with the frame/bin duration implied by `BIN_SIZE` and `FS`.

ii.
```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS  # seconds per bin
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The notes map the input to "Time in seconds from start of trial" and treat it as a constructed time axis rather than a direct raw-data variable.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the agent creates a one-row array from `0` to `(n_timebins - 1) * BIN_SIZE / FS` seconds. The time axis resets to zero for every trial and inherits the 10-frame temporal binning.

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
```

```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The `make_time_input()` docstring says "Time in seconds from start of trial," and Step 5 of the notes repeats that design choice explicitly.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is generated after trial splitting, using exactly the same `n_timebins` as each neural trial. That means the input is aligned one-for-one with the binned neural samples inside each 2-minute block.

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
```

iii. The notes describe this as "Time in seconds from start of trial" and verify in Step 10 that the values match `arange * 10/30`.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy`. The agent also loads `tstamps.npy` and treats it as auxiliary timing information for handling frame mismatches.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
...
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
```

iii. Step 5 of the notes maps `motion_energy_glob.npy -> output[0]` and Step 3 cites the data README note that missing camera frames can be identified from `tstamps.npy` or `interframe_int.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent interpolates motion energy to the neural frame count when needed, then averages it in 10-frame bins, splits it into 2-minute trials, and finally discretizes each trial using global percentile edges computed from all binned motion-energy values. The interpolation itself ignores the actual timestamps and instead spreads the shorter series evenly across the neural frame range.

ii.
```python
def interpolate_motion_energy(me, n_neural_frames, tstamps):
    ...
    if n_me < n_neural_frames:
        neural_indices = np.arange(n_neural_frames)
        me_indices = np.linspace(0, n_neural_frames - 1, n_me)
        me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
        return me_interp
```

```python
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
me_binned = bin_traces(me_interp, BIN_SIZE)
...
bin_edges = compute_me_percentile_bins(all_me_concat, N_OUTPUT_BINS)
...
me_disc = discretize_me(me_t, bin_edges)
```

iii. The notes justify interpolation by quoting the dataset README about missing camera frames, and justify 10-frame averaging plus global percentile discretization in Step 5 as matching the paper and decoder task.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent concatenates all binned motion-energy values across sessions, computes the 0th, 20th, 40th, 60th, 80th, and 100th percentiles, and then assigns each time bin to one of five classes using `np.digitize()`. The output labels are human-readable strings derived from those numeric thresholds.

ii.
```python
def compute_me_percentile_bins(all_me_values, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_me_values, percentiles)
    return bin_edges
```

```python
def discretize_me(me_values, bin_edges):
    n_bins = len(bin_edges) - 1
    labels = np.digitize(me_values, bin_edges[1:-1])  # 0 to n_bins-1
    labels = np.clip(labels, 0, n_bins - 1)
    return labels.astype(np.int64)
```

iii. The notes explicitly say "Compute percentile bins globally across all sessions" because the task asked for "normalized and discretized into five equal-percentile bins."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by first forcing it to the same frame count as the neural recording, then applying the same 10-frame binning, then slicing the same consecutive trial windows as the neural data. Each trial's output therefore has the same number of time bins as the corresponding neural trial.

ii.
```python
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
dff_binned = bin_traces(dff, BIN_SIZE)
me_binned = bin_traces(me_interp, BIN_SIZE)
```

```python
neural_trials, me_trials = split_into_trials(dff_binned, me_binned)
for neural_t, me_t in zip(neural_trials, me_trials):
    me_disc = discretize_me(me_t, bin_edges)
    output_trials.append(me_disc.reshape(1, -1))
```

iii. Step 5 of the notes says "Temporal alignment: ME is already synced to neural (camera triggered by microscope)" and that missing frames should be interpolated to match the neural frame count.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent mainly addresses camera-frame mismatches. If motion energy is shorter than the neural recording, it linearly interpolates it up to the neural frame count; if it is longer, it truncates it. For neural preprocessing, it clips the reconstructed baseline to `1e-6` to avoid divide-by-zero in dF/F. For trialization, it silently drops any incomplete final block via floor division.

ii.
```python
if n_me < n_neural_frames:
    neural_indices = np.arange(n_neural_frames)
    me_indices = np.linspace(0, n_neural_frames - 1, n_me)
    me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
    return me_interp
else:
    return me[:n_neural_frames].astype(np.float64)
```

```python
baseline_safe = np.clip(baseline, 1e-6, None)
...
n_trials = n_bins // bins_per_trial
```

iii. The dataset README note about missing camera frames is the stated justification for interpolation, and the dF/F docstring/comments justify baseline clipping as protection against division by zero. The notes list "Missing ME frames: Interpolate to match neural frame count using timestamps" as an explicit design decision.

## 6-a. What are the most time-consuming steps of the code?

i. The slowest step is the per-session neural preprocessing inside `compute_dff()`, especially the Suite2p baseline computation. Large full-session file loads and the full first-pass scan over every session are also significant. The logs show session times scaling with neuron count and frame count, consistent with that.

ii.
```python
dff = compute_dff(F, Fneu, device=device)
```

```python
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)
```

```python
for sess_dir in sessions:
    t_sess = time.time()
    dff_binned, me_binned, n_neurons, n_frames_raw = process_session(
        sess_dir, device=device
    )
    ...
    print(f"  {sess_name}: {n_neurons} neurons, {n_frames_raw} frames -> "
          f"{dff_binned.shape[1]} bins ({elapsed:.1f}s)")
```

iii. Step 6 of the notes says the script uses Suite2p preprocessing and reports about 1 second per session. The conversion log confirms bigger sessions take longer.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that separately creates `time_input`, discretizes motion energy, and appends arrays could have been partly vectorized or batch-reshaped after trial splitting. The `split_into_trials()` loop itself could also be replaced with a reshape/split operation because the trial boundaries are uniform.

ii.
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

iii. The notes say the agent tried to write "efficient code" and used vectorized binning, but these later list-building loops were left in straightforward Python form.

## 6-c. What processing does the code repeat multiple times?

i. The code makes two passes over session data: one pass to preprocess sessions and collect motion-energy values for global percentile estimation, then a second pass to split into trials and build outputs. In `--show-processing` mode, it also reloads raw `F.npy`, `Fneu.npy`, and motion-energy files again inside `plot_processing()`.

ii.
```python
session_data = []
all_me_binned_values = []
...
for subj_idx, subject in enumerate(subjects_to_process):
    ...
    for sess_dir in sessions:
        dff_binned, me_binned, n_neurons, n_frames_raw = process_session(
            sess_dir, device=device
        )
        session_data.append((subj_idx, sess_dir, dff_binned, me_binned, n_neurons))
        all_me_binned_values.append(me_binned)
```

```python
for subj_idx, sess_dir, dff_binned, me_binned, n_neurons in session_data:
    neural_trials, me_trials = split_into_trials(dff_binned, me_binned)
    ...
```

```python
def plot_processing(...):
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
```

iii. Step 5 says the agent wanted global percentile bins, which forced a first pass over all sessions before trial output construction. The plotting helper is also designed only for diagnostics, so it reloads data instead of reusing cached raw arrays.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `tstamps.npy` but does not actually use its values in interpolation; the interpolation is based on `np.linspace()` instead. In `--show-processing` mode it also recomputes and reloads visualization-only data that are not part of the saved dataset. `n_frames_raw` and several metadata/logging quantities are computed only for reporting.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
...
me_indices = np.linspace(0, n_neural_frames - 1, n_me)
```

```python
def plot_processing(...):
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
    ...
```

```python
return dff_binned, me_binned, n_neurons, n_frames
...
print(f"  {sess_name}: {n_neurons} neurons, {n_frames_raw} frames -> "
      f"{dff_binned.shape[1]} bins ({elapsed:.1f}s)")
```

iii. The notes emphasize visual verification and timing information, so the agent added plotting and logging machinery beyond what the final decoder dataset needs. There is no justification in the notes for loading `tstamps.npy` specifically and then not using it.
