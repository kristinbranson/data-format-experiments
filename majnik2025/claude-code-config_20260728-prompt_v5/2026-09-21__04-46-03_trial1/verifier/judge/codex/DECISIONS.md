# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for subject directories whose names start with `jm`, scans each subject directory for dated session folders whose names start with `2`, then loads per-session neural data from `suite2p/plane0/{F.npy,Fneu.npy,ops.npy}` and behavior from `move_deve/{motion_energy_glob.npy,tstamps.npy}`. Trials are not loaded directly from disk; they are created later by splitting each continuous session into fixed 60 s chunks.

ii. 
```python
SUBJECTS = sorted([d for d in os.listdir(DATA_DIR)
                    if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])

def get_sessions(subject):
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d)) and d[0] == '2'])
    return sessions

F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'ops.npy'),
              allow_pickle=True).item()
me = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset contains 6 `jm*` subject folders, each with 6-7 dated session folders containing suite2p outputs and motion-energy files. The trajectory shows it chose this layout because it matched the on-disk organization and because the session folders are all date-stamped.

## 1-b. How are the data split into subjects?

i. Subjects are the top-level directories in `/app/data` whose names begin with `jm`, sorted alphabetically. The AI treats each such directory as one mouse.

ii. 
```python
SUBJECTS = sorted([d for d in os.listdir(DATA_DIR)
                    if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
```

iii. In `CONVERSION_NOTES.md`, the AI records 6 subjects: `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`. The trajectory states that each `jm*` directory is one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are the dated subdirectories inside each subject folder, sorted lexicographically. The code only keeps subdirectories whose first character is `2`, i.e. the `2023-...` or `2024-...` recording folders.

ii. 
```python
def get_sessions(subject):
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d)) and d[0] == '2'])
    return sessions
```

iii. In the notes, the AI describes each dated folder as one daily recording session. The trajectory shows it intentionally used the date-named directories to define sessions.

## 1-d. How are the data split into trials?

i. The AI treats each session as a continuous recording, bins it from frames to 10-frame time bins, then splits the binned arrays into non-overlapping 60 s trials. At 30 Hz and 10-frame bins, each trial is 180 bins long. Any trailing partial trial is dropped.

ii. 
```python
BIN_SIZE = 10
TRIAL_DURATION_S = 60.0
BINS_PER_TRIAL = int(TRIAL_DURATION_S / BIN_DURATION_S)  # 180 bins

def split_into_trials(data, bins_per_trial):
    if data.ndim == 2:
        n_neurons, n_bins = data.shape
        n_trials = n_bins // bins_per_trial
        trials = []
        for t in range(n_trials):
            start = t * bins_per_trial
            end = start + bins_per_trial
            trials.append(data[:, start:end])
        return trials
    else:
        n_bins = len(data)
        n_trials = n_bins // bins_per_trial
        trials = []
        for t in range(n_trials):
            start = t * bins_per_trial
            end = start + bins_per_trial
            trials.append(data[start:end])
        return trials
```

iii. In `CONVERSION_NOTES.md`, the AI says there is no natural task-trial structure and that sessions should be split into 60 s chunks per the decoder specification. The trajectory explicitly computes 20 trials for 20 min sessions and 30 trials for 30 min sessions.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a trial-quality filter. The only trial-level exclusion is implicit: incomplete trailing bins are dropped because `split_into_trials` keeps only full 60 s chunks.

ii. 
```python
def split_into_trials(data, bins_per_trial):
    ...
    n_trials = n_bins // bins_per_trial
    ...
```

iii. In the notes, the AI says there is no specific trial curation because the dataset is spontaneous behavior without natural trials. It only follows the decoder requirement to create fixed-length trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from raw suite2p fluorescence `F.npy` and neuropil fluorescence `Fneu.npy`. The code also loads `ops.npy` to read Suite2p preprocessing parameters such as frame rate and baseline settings.

ii. 
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'ops.npy'),
              allow_pickle=True).item()
```

iii. In `CONVERSION_NOTES.md`, the AI says the paper uses baseline-corrected fluorescence for decoding, so it should start from `F` and `Fneu` rather than `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The AI computes a dF/F-like signal manually. It first applies neuropil subtraction (`F - 0.7 * Fneu`), then computes a maximin-style baseline using Gaussian smoothing followed by minimum and maximum filters, then divides `(Fc - baseline)` by a floored baseline (`max(baseline, 10)`). After that, it averages the trace in non-overlapping 10-frame bins.

ii. 
```python
def compute_dff(F, Fneu, ops):
    Fc = F - NEUCOEFF * Fneu
    sig_baseline = ops.get('sig_baseline', 10.0)
    win_baseline = ops.get('win_baseline', 60.0)
    fs = ops.get('fs', FRAME_RATE)

    win = int(win_baseline * fs)
    smoothed = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    flow_min = minimum_filter1d(smoothed, size=win, axis=1)
    baseline = maximum_filter1d(flow_min, size=win, axis=1)

    baseline_safe = np.maximum(baseline, 10.0)
    dff = (Fc - baseline) / baseline_safe
    return dff

dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The justification in `CONVERSION_NOTES.md` and the trajectory is that the paper says it used “baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)” and that decoding analyses binned both dF/F and behavior in 10-frame windows. The trajectory also records that the AI initially had the baseline wrong, then changed it to “Gaussian -> minimum -> maximum filter” to match its reading of Suite2p.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied in the conversion script. The AI uses every row in `F.npy`/`Fneu.npy`, assuming the data are already restricted to tracked cells and that all ROIs pass the `iscell` threshold.

ii. 
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
...
n_neurons, n_frames = F.shape
...
all_brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. In `CONVERSION_NOTES.md`, the AI states that all `iscell` values are 1 and that the Track2p outputs already contain only tracked neurons, so it decided no extra neuron-quality filter was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats the beginning of the session as the alignment event, does not perform any event-locked re-centering, and then segments the continuous session into consecutive 60 s windows. In metadata it labels the alignment event as “Session start (beginning of recording)” and sets `off_start=0.0`, `off_end=60.0`.

ii. 
```python
time_input = np.arange(n_bins) * BIN_DURATION_S
neural_trials = split_into_trials(dff_binned, BINS_PER_TRIAL)

'metadata': {
    'temporal_alignment_event': 'Session start (beginning of recording)',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_S,
    ...
}
```

iii. The notes say there is no natural trial event in the spontaneous-behavior dataset, so the AI used session start as the practical reference point. The trajectory repeatedly refers to the input as “time elapsed from session start.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping bins. At 30 Hz, this is `10 / 30 = 0.333...` s per bin, i.e. about 333.3 ms. The same rebinning is applied to neural activity and motion energy.

ii. 
```python
FRAME_RATE = 30.0
BIN_SIZE = 10
BIN_DURATION_S = BIN_SIZE / FRAME_RATE  # ~0.333 seconds

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. In `CONVERSION_NOTES.md`, the AI cites the paper’s statement that decoding analyses averaged both dF/F and behavior “in bins of 10 consecutive timestamps.”

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not read from a raw file. The AI derives it from the binned sample index after preprocessing, using a constant frame rate and bin width.

ii. 
```python
n_bins = dff_binned.shape[1]
time_input = np.arange(n_bins) * BIN_DURATION_S
```

iii. In the trajectory, the AI says there are no explicit time variables needed because time can be reconstructed from the known 30 Hz sampling rate and the 10-frame bin size.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code builds a monotonic time vector in seconds from session start, with one value per binned time step, then splits that vector into 60 s trial chunks and reshapes each chunk to `(1, n_timepoints)`.

ii. 
```python
time_input = np.arange(n_bins) * BIN_DURATION_S
input_trials = split_into_trials(time_input, BINS_PER_TRIAL)
input_trials_formatted = [t_in.reshape(1, -1).astype(np.float32) for t_in in input_trials]
```

iii. In `CONVERSION_NOTES.md`, the AI describes the input as “Time elapsed from session start (seconds).” The trajectory notes that this was chosen to satisfy the decoder-task requirement for a time-varying input.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time vector is created at the same binned resolution as the neural data, before trial splitting, and then split into trials with the same boundaries. This makes each input timepoint correspond to the same bin index as the neural data.

ii. 
```python
dff_binned = bin_data(dff, BIN_SIZE)
n_bins = dff_binned.shape[1]
time_input = np.arange(n_bins) * BIN_DURATION_S

neural_trials = split_into_trials(dff_binned, BINS_PER_TRIAL)
input_trials = split_into_trials(time_input, BINS_PER_TRIAL)
```

iii. The AI’s notes and trajectory both emphasize that the time input is defined in the same 10-frame bins as the neural data so they remain synchronized.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy` plus `move_deve/tstamps.npy`. The timestamps are used only when the motion-energy array is shorter than the neural recording and interpolation is needed.

ii. 
```python
me = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
...
me_aligned = align_motion_energy(me, ts, n_frames)
```

iii. In the notes, the AI says the motion-energy trace is the precomputed behavioral signal and that `tstamps.npy` is in kiloseconds. The trajectory says it chose timestamps because the README says missing frames can be identified from `tstamps.npy` or `interframe_int.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI optionally realigns motion energy to the neural frame grid by interpolating from camera timestamps to nominal 30 Hz neural times, bins the aligned trace in 10-frame windows, and discretizes the binned signal into five percentile bins separately within each session.

ii. 
```python
def align_motion_energy(me, me_timestamps, n_neural_frames):
    if len(me) == n_neural_frames:
        return me.copy()
    cam_times_s = me_timestamps * 1000.0
    neural_times_s = np.arange(n_neural_frames) / FRAME_RATE
    me_aligned = np.interp(neural_times_s, cam_times_s, me)
    return me_aligned

me_aligned = align_motion_energy(me, ts, n_frames)
me_binned = bin_data(me_aligned, BIN_SIZE)
me_discrete, bin_edges = discretize_motion_energy(me_binned)
```

iii. The notes justify this as matching the paper’s 10-frame denoising and the data README’s instruction that missing camera frames can be interpolated. The trajectory says the AI preferred timestamp-based interpolation because the camera is triggered by the microscope and missing frames should be filled to the neural frame grid.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. It is discretized per session into 5 equal-percentile bins using `np.percentile` and `np.digitize`, with labels clipped to the valid range `0..4`.

ii. 
```python
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, bin_edges[1:-1], right=False)
    labels = np.clip(labels, 0, n_bins - 1)
    return labels, bin_edges
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly lists “5 equal-percentile bins per session” as a key decision because the decoder task required categorical motion-energy output.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. If the motion-energy trace already has one value per neural frame, it is left unchanged. Otherwise the AI interpolates the behavioral trace from camera timestamps onto nominal neural frame times (`np.arange(n_neural_frames) / 30`) and then bins both streams with the same 10-frame boundaries.

ii. 
```python
def align_motion_energy(me, me_timestamps, n_neural_frames):
    if len(me) == n_neural_frames:
        return me.copy()
    cam_times_s = me_timestamps * 1000.0
    neural_times_s = np.arange(n_neural_frames) / FRAME_RATE
    me_aligned = np.interp(neural_times_s, cam_times_s, me)
    return me_aligned

me_aligned = align_motion_energy(me, ts, n_frames)
me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. The AI’s notes say “Camera triggered by microscope at 30 Hz” and “Missing camera frames: interpolate over them.” The trajectory also states that `tstamps.npy` or `interframe_int.npy` could be used, and it chose the timestamp route.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing camera frames by interpolation and handles incomplete trailing bins/trials by truncation. It does not add other explicit missing-data handling for neural traces.

ii. 
```python
if len(me) == n_neural_frames:
    return me.copy()
...
me_aligned = np.interp(neural_times_s, cam_times_s, me)

trimmed = data[:, :n_bins * bin_size]
...
n_trials = n_bins // bins_per_trial
```

iii. In the notes, the AI identifies several sessions where motion energy has fewer frames than the neural data and cites the data README’s suggestion to interpolate missing frames. The notes also treat leftover bins at the end of a session as acceptable data loss when enforcing fixed 60 s trials.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive part is per-session preprocessing of the full fluorescence matrix: neuropil subtraction, Gaussian/min/max baseline filtering, and binning large arrays. Optional plotting also adds overhead when `--show-processing` is enabled.

ii. 
```python
dff = compute_dff(F, Fneu, ops)
dff_binned = bin_data(dff, BIN_SIZE)
...
if show_processing:
    _plot_processing(...)
```

iii. In `CONVERSION_NOTES.md`, the AI estimates per-session processing time and specifically discusses the baseline computation as the main heavy step. Earlier in the trajectory it also flagged maximin baseline estimation as the expensive part.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still uses Python loops to split both 2-D neural arrays and 1-D time/output vectors into trials, and it loops over subjects and sessions at the Python level. Those loops could be reduced by reshaping session arrays directly into trial blocks once the valid prefix length is known.

ii. 
```python
for subj_i, subject in enumerate(subjects_list):
    sessions = get_sessions(subject)
    for sess_j, session in enumerate(sessions):
        ...

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    trials.append(data[:, start:end])
```

iii. The notes mention efficiency concerns explicitly and earlier versions in the trajectory even had a slower per-neuron baseline loop before the AI replaced it with `scipy.ndimage` filters. The remaining trial-splitting loops are straightforward but still vectorizable.

## 6-c. What processing does the code repeat multiple times?

i. The same split-into-trials operation is repeated three times per session: once for neural data, once for discretized motion energy, and once for the time input. The code also performs separate formatting passes to cast/reshape those trial lists after splitting.

ii. 
```python
neural_trials = split_into_trials(dff_binned, BINS_PER_TRIAL)
output_trials = split_into_trials(me_discrete, BINS_PER_TRIAL)
input_trials = split_into_trials(time_input, BINS_PER_TRIAL)

output_trials_formatted = [o.reshape(1, -1).astype(np.int64) for o in output_trials]
input_trials_formatted = [t_in.reshape(1, -1).astype(np.float32) for t_in in input_trials]
neural_trials_formatted = [n.astype(np.float32) for n in neural_trials]
```

iii. The AI did not frame this as a major problem in its notes, but the final code plainly repeats the same segmentation/formatting pattern across modalities.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script includes optional plotting and reporting work that is not used by downstream decoder training: it builds diagnostic figures, computes and prints output-frequency summaries, and carries a `SUBJECT_MAP` only for nicer log messages. None of this affects the saved dataset.

ii. 
```python
SUBJECT_MAP = {
    'jm031': 'Mouse A', 'jm032': 'Mouse B', ...
}

all_outputs = np.concatenate(output_trials)
unique, counts = np.unique(all_outputs, return_counts=True)
fracs = counts / counts.sum()
print(f"    ME bin distribution: {dict(zip(unique.astype(int), np.round(fracs, 3)))}")

if show_processing:
    _plot_processing(...)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly says it added a “Processing visualization option” and ran plotting-based sanity checks. Those diagnostics are useful for verification but are discarded by downstream analyses.
