# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers subjects by scanning `/app/data` for top-level directories whose names start with `jm`. For each subject it scans all session subdirectories, then for each session it loads `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, and `move_deve/motion_energy_glob.npy` with `np.load`. Trials are not loaded from disk directly; they are created later by splitting each processed continuous session into fixed 2-minute blocks.

ii.
```python
def get_subjects_and_sessions():
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
    all_sessions = {}
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        all_sessions[subj] = sessions
    return subjects, all_sessions

F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. `CONVERSION_NOTES.md` says the dataset is organized as subject folders containing session folders, each with `suite2p` neural outputs and `move_deve` motion-energy outputs. The notebook in `/app/data/load_data.ipynb` also demonstrates directory-based loading from session folders.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the six top-level `jm*` directories under `data/`, sorted alphabetically. The script maps those IDs to paper-style names (`Mouse_A` to `Mouse_F`) with a hard-coded dictionary, and stores one `subject_idx` entry per session.

ii.
```python
SUBJECT_MAP = {
    'jm031': 'Mouse_A', 'jm032': 'Mouse_B', 'jm038': 'Mouse_C',
    'jm039': 'Mouse_D', 'jm040': 'Mouse_E', 'jm046': 'Mouse_F'
}

subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])

subject_names = list(SUBJECT_MAP.values()) if not sample_mode else [SUBJECT_MAP[subjects_to_process[0]]]
...
subj_idx = subject_names.index(SUBJECT_MAP[subj])
...
subject_idx_list.append(subj_idx)
```

iii. The data README explicitly says each subject has its own folder and that `jm031` corresponds to mouse A, `jm032` to mouse B, and so on. The notes repeat that mapping and treat the six `jm*` directories as the full subject list.

## 1-c. How are the data split into sessions?

i. Sessions are defined as all immediate subdirectories within each subject directory, sorted lexicographically, which also preserves chronological order because the folders are named `YYYY-MM-DD_a`. Each such directory becomes one session in the output lists.

ii.
```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    all_sessions[subj] = sessions

for subj in subjects_to_process:
    sessions = sessions_to_process[subj]
    for i, session in enumerate(sessions):
        neural_trials, input_trials, output_trials, n_neurons = process_session(...)
        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
```

iii. The data README says each subject folder contains one folder per recording day, named by date, and the notebook sorts those session directories to keep them chronological. The notes use the same interpretation.

## 1-d. How are the data split into trials?

i. The raw data do not contain trial boundaries. The script treats each session as one continuous recording, bins it into 10-frame windows, then splits the binned session into non-overlapping 2-minute blocks of 360 bins each. Those blocks become the per-session trial lists.

ii.
```python
TRIAL_DURATION_SEC = 120
TRIAL_BINS = int(TRIAL_DURATION_SEC * FRAME_RATE / BIN_SIZE)  # 360 bins per trial

dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()

n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. In `CONVERSION_NOTES.md`, the agent states that the experiment is continuous spontaneous recording with no discrete trials, and justifies 2-minute blocks by citing the paper’s cross-validation structure of “consecutive 2 minute blocks of the recording.”

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level quality-control filter. The script keeps every complete 2-minute block in a session. The only practical exclusion is implicit: `n_trials = n_total_bins // TRIAL_BINS` would drop a trailing partial block, and missing motion-energy frames are repaired by interpolation rather than causing trial rejection.

ii.
```python
n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes say “No explicit trial curation (continuous spontaneous recording)” and “Missing camera frames should be interpolated,” so the agent’s justification was that the available references did not define trial rejection rules.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the Suite2p fluorescence matrices `F.npy` and `Fneu.npy`. The script ignores `spks.npy` and does not load raw movies or any other neural source.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
...
dfof = compute_dfof(F, Fneu)
```

iii. The notes and README both describe a pipeline based on neuropil-corrected fluorescence, and the notebook comments say that `F.npy` contains raw fluorescence traces while proper analysis should compute dF/F as described in the paper.

## 2-b. How is the `neural` data processed?

i. The script subtracts neuropil with `F - 0.7 * Fneu`, applies Suite2p’s `preprocess` function with the default `maximin` baseline parameters, casts the result to `float32`, and then temporally bins the result by averaging each consecutive 10-frame window.

ii.
```python
Fc = F - NEUROPIL_COEFF * Fneu

dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)

return dfof.astype(np.float32)

...
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
```

iii. The notes cite Suite2p defaults from `ops.npy` and say the paper used “baseline corrected fluorescence traces as our dF/F.” The trajectory shows the agent explicitly changed an earlier implementation so it would use Suite2p’s baseline-subtracted output directly rather than dividing by a baseline estimate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script performs no explicit neuron-level filtering in code. It assumes the provided `F.npy` rows already correspond to successfully tracked cells that passed Suite2p’s `iscell` threshold and Track2p matching across all days, so every row in `F`/`Fneu` is kept.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
n_neurons, n_frames = F.shape
...
brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The notes say “All iscell values are 1 (pre-filtered through Track2p matching)” and that the provided data already contain only cells present across all days. That is the agent’s reason for not applying additional filters in `convert_data.py`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the beginning of each continuous recording session, not to any behavioral event. After session-wide processing and 10-frame binning, the script slices the binned trace into sequential 2-minute blocks, and those blocks inherit the same session-start alignment.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'start of continuous recording session',
    'off_start': 0.0,
    'off_end': None,
    ...
}

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
```

iii. The notes describe the source recording as continuous spontaneous activity with no discrete trial event, so the agent chose session start as the alignment anchor and used the paper’s 2-minute recording blocks as trial boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10 original frames per time bin. At 30 Hz this is `10 / 30 = 0.333...` s per bin, or about 333.33 ms. The script rebins by averaging each consecutive group of 10 frames for both neural data and motion energy.

ii.
```python
FRAME_RATE = 30.0
BIN_SIZE = 10
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000

def bin_timeseries(data, bin_size):
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    return truncated.reshape(new_shape).mean(axis=-1)
```

iii. The notes and README justify this directly from the cited paper text: “averaging in bins of 10 consecutive timestamps.”

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time signal is not read from a raw timestamp array. It is synthesized from the binned frame index using the constants `BIN_SIZE` and `FRAME_RATE`. In practice it is derived from the implicit imaging frame count, not from `tstamps.npy`.

ii.
```python
FRAME_RATE = 30.0
BIN_SIZE = 10

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The notes say the decoder input should be “time elapsed” and describe it as “time index” mapped to `bin_index * (10/30) seconds`. The agent did not cite any raw timing file as the source for this variable.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial-sized block, the script creates an integer range of bin indices `[start, end)`, multiplies by seconds per bin (`10/30`), reshapes to `(1, T)`, and stores the result as `float32`. There is no use of recorded timestamps, no interpolation, and no additional normalization.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS

    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The notes justify this as “Time index” conversion, and the README describes the input as “time elapsed in seconds.” The trajectory does not show any attempt to use a measured timestamp stream.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it is generated from the same binned indices and same `start:end` trial slices used for the neural arrays. Each input bin therefore corresponds one-to-one with a neural time bin inside the same 2-minute block.

ii.
```python
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
...
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The notes describe the input as “time elapsed from start” and the trial segmentation as consecutive 2-minute blocks, so the agent’s alignment rationale was to use the same post-binning indices for both neural and input streams.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from the behavioral file `move_deve/motion_energy_glob.npy`. The script treats that array as already-computed motion energy and does not reconstruct motion energy from raw videos.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
me = interpolate_motion_energy(me_raw, n_frames)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
```

iii. The data README says `move_deve` contains processed behavioral data and specifically names `motion_energy_glob.npy` as the extracted motion-energy signal. The notes repeat that this is the source variable.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI documented an intended pipeline of per-session normalization followed by discretization into quintiles, but the actual code only interpolates missing frames, bins by averaging 10-frame windows, and then discretizes. There is no explicit normalization operation in `convert_data.py`.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
me = interpolate_motion_energy(me_raw, n_frames)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()

# Discretize motion energy into 5 equal-percentile bins
me_discrete = discretize_motion_energy(me_binned, N_OUTPUT_BINS)
```

iii. The docstring, notes, and README all say “motion energy normalized per session and discretized into 5 equal-percentile bins,” and the step-planning table in `CONVERSION_NOTES.md` explicitly lists “normalize per session” before discretization. The final code does not include that normalization step.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The script computes the 20th, 40th, 60th, and 80th percentiles of the binned session-level motion-energy trace, then uses `np.digitize` to assign each time bin to one of five labels `0` through `4`.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, thresholds).astype(np.int64)
    return labels
```

iii. The notes justify this with the decoder specification to produce “5 equal-percentile bins,” and the validation notes say the agent manually checked that the resulting distribution was 20% per bin.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The script first forces `motion_energy_glob.npy` to the same frame count as the neural recording by linearly interpolating over a normalized 0-to-1 axis whenever lengths differ. It then bins motion energy with the same 10-frame windows used for neural data and slices the same `start:end` trial ranges. The code does not use `tstamps.npy` or `interframe_int.npy` to place missing camera frames at their actual indices.

ii.
```python
def interpolate_motion_energy(me, n_frames):
    if len(me) == n_frames:
        return me.astype(np.float64)

    x_orig = np.linspace(0, 1, len(me))
    x_new = np.linspace(0, 1, n_frames)
    me_interp = np.interp(x_new, x_orig, me.astype(np.float64))
    return me_interp

...
me = interpolate_motion_energy(me_raw, n_frames)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes cite the README statement that missing frames can be “treated as missing values or interpolated over,” which is the stated justification for interpolation. However, the README also says the missing-frame indices can be recovered from `tstamps.npy` or `interframe_int.npy`, and the code does not use them.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The only explicit error-handling path for imperfect data is motion-energy length mismatch. If `motion_energy_glob.npy` has fewer entries than the neural recording, the script interpolates it to `n_frames`; otherwise it does nothing special. It does not check for NaNs, corrupt sessions, neuron-level anomalies, or timestamp irregularities beyond total-length mismatch.

ii.
```python
def interpolate_motion_energy(me, n_frames):
    if len(me) == n_frames:
        return me.astype(np.float64)

    x_orig = np.linspace(0, 1, len(me))
    x_new = np.linspace(0, 1, n_frames)
    me_interp = np.interp(x_new, x_orig, me.astype(np.float64))
    return me_interp
```

iii. The notes cite the data README’s suggestion that missing motion-energy frames “can be interpolated over,” and that appears to be the entire justification for the handling of minor data issues.

## 6-a. What are the most time-consuming steps of the code?

i. The main costs are per-session array loading, Suite2p baseline preprocessing on the full `F - 0.7*Fneu` matrix, and repeated binned slicing over large session arrays. The full-conversion log shows that the larger 54,000-frame sessions take substantially longer than the 36,000-frame sessions, which is consistent with those operations dominating runtime.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
...
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)
...
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
```

iii. The timing notes estimate about 0.5 s for 36k-frame sessions and 0.7 s for 54k-frame sessions, and the full conversion log shows the same pattern. That supports the agent’s runtime characterization.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop inside `process_session` could have been replaced with a reshape-based split for `dfof_binned` and `me_discrete`, because trials are fixed-length contiguous blocks. The time-input generation also recomputes `np.arange(start, end)` separately for every trial even though a single trial template could be broadcast or reused.

ii.
```python
neural_trials = []
input_trials = []
output_trials = []

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. This is implied by the code structure itself: trials are uniform contiguous blocks, so the loop exists for convenience rather than necessity.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly computes the torch device selection inside every `compute_dfof` call, recreates time vectors for every trial, repeats the same subject-name lookup for every session, and reimports plotting modules in optional plotting paths. It also repeats identical session-level preprocessing separately for all 41 sessions.

ii.
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
...
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
...
subj_idx = subject_names.index(SUBJECT_MAP[subj])
...
if show_processing:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
```

iii. These repetitions follow directly from the function layout. The notes do not frame them as a problem, but they are evident from the implementation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is not much truly discarded processing in the default path. The clearest unnecessary work is optional plotting support and its repeated imports, because those figures are not part of the saved dataset. Outside of plotting, the code’s intermediates (`Fc`, `dfof`, `me_binned`, `me_discrete`) are transient but are needed to produce the final outputs.

ii.
```python
if show_processing:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

...
if show_processing and fig_ax is not None:
    fig, axes = fig_ax
    fig.suptitle(f'Processing: {subj}/{session}', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'processing_{subj}_{session}.png', dpi=100)
    plt.close(fig)
```

iii. The notes describe plotting as a sanity-check aid rather than part of the converted dataset, so this is the part of the script most clearly outside downstream decoder use.
