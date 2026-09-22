# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every top-level directory under `/app/data` as a subject, then every subdirectory under each subject as a session. For each session it loads calcium traces from `suite2p/plane0/F.npy` and `Fneu.npy`, and motion data from `move_deve/motion_energy_glob.npy` plus `tstamps.npy`. Trials are not loaded from disk directly; they are created later by splitting each processed session into fixed 60-second chunks.

ii.
```python
def get_subjects_and_sessions(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])
    subject_sessions = {}
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        subject_sessions[subj] = sessions
    return subjects, subject_sessions
```

```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. In `CONVERSION_NOTES.md`, Step 5 says the mapping is `F.npy, Fneu.npy -> neural` and `motion_energy_glob.npy -> output`, and Step 2 describes the directory pattern as `data/{subject}/{session}/...`. The trajectory summary in step 115 gives the same pipeline.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined as all directories found directly under `/app/data`, sorted alphabetically.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))])
```

iii. `CONVERSION_NOTES.md` Step 2 records six subject folders (`jm031` to `jm046`) and treats each as one mouse. The trajectory repeatedly describes the dataset as six mice organized by subject directory.

## 1-c. How are the data split into sessions?

i. Sessions are defined as all subdirectories within a subject directory, sorted alphabetically. Each session becomes one output session in the converted dataset.

ii.
```python
subj_dir = os.path.join(data_dir, subj)
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
subject_sessions[subj] = sessions
```

```python
for sess_name in sessions:
    result = process_session(subj, sess_name, data_dir,
                            show_processing=show_processing,
                            session_idx=session_count)
    all_neural.append(result['neural'])
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says “Each session is a separate entry,” justified by the fact that neurons are tracked across days but daily recordings remain separate sessions.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous sessions and creates artificial trials by binning to 10-frame time bins first, then splitting each session into non-overlapping 60-second trials of 180 bins each. Any leftover tail shorter than one full trial is dropped implicitly.

ii.
```python
TRIAL_DURATION_S = 60
TRIAL_BINS = int(TRIAL_DURATION_S / (BIN_SIZE / FRAME_RATE))  # 180 bins per trial
```

```python
def split_into_trials(data, trial_length, axis=-1):
    n = data.shape[axis]
    n_trials = n // trial_length
    trials = []
    for i in range(n_trials):
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(i * trial_length, (i + 1) * trial_length)
        trials.append(data[tuple(slices)])
    return trials
```

iii. `CONVERSION_NOTES.md` Step 5 says there is no native task-trial structure, so sessions are split into 60-second trials as required by the decoder task. The notes compute 20-minute sessions as 20 trials and 30-minute sessions as 30 trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a separate trial-quality filter. The only effective exclusion is that incomplete trailing data shorter than a full 60-second trial is not returned because `n_trials` uses floor division.

ii.
```python
def split_into_trials(data, trial_length, axis=-1):
    n = data.shape[axis]
    n_trials = n // trial_length
    trials = []
    for i in range(n_trials):
        ...
    return trials
```

iii. `CONVERSION_NOTES.md` Step 3 says there is “No explicit trial curation mentioned” because this is spontaneous behavior rather than a task with failed trials. The justification is therefore that no additional trial QC is needed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` signal is derived from Suite2p fluorescence arrays `F.npy` and `Fneu.npy` from `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. `CONVERSION_NOTES.md` Step 1 identifies `F.npy` as raw fluorescence and `Fneu.npy` as neuropil, and Step 5 maps those files directly to the target `neural` field.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (`F - 0.7 * Fneu`) and then a custom PyTorch implementation of a Suite2p-style maximin baseline procedure. The returned neural signal is baseline-subtracted fluorescence that the AI describes as dF/F in its notes and metadata. After that, it averages the neural trace into non-overlapping 10-frame bins.

ii.
```python
Fc = F - NEUCOEFF * Fneu
Fc = Fc.astype(np.float32)
...
data = pad(data, (gwid, gwid), 'replicate')
data = conv1d(data.unsqueeze(1), gaussian.unsqueeze(0).unsqueeze(0), padding=0)
...
data = -max_pool1d(-data, kernel_size=win, stride=1, padding=0)
...
data = max_pool1d(data, kernel_size=win, stride=1, padding=0)
...
Flow[nstart:nend] = data.squeeze(1).cpu().numpy()
...
dff = Fc - Flow
```

```python
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
```

iii. `CONVERSION_NOTES.md` Step 5 says the neural transform should use “Suite2p default parameters (neucoeff=0.7, maximin baseline, win_baseline=60s).” The trajectory around steps 42 and 54 shows the AI explicitly reasoning that the paper used Suite2p default dF/F-style preprocessing and deciding to reimplement that logic.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no additional neuron filtering in the conversion script. It assumes the provided data already contains tracked, accepted cells and keeps every row of `F.npy`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
n_neurons, n_frames = F.shape
...
all_brain_region_idx.append(np.zeros(result['n_neurons'], dtype=np.int64))
```

iii. `CONVERSION_NOTES.md` Step 1 states that the provided data already contains only tracked neurons and that all cells in the checked subject pass the `iscell > 0.5` threshold. Step 4 then concludes that no additional filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to session start. Because the source recording is continuous and the trials are synthetic 60-second blocks, each trial is just a contiguous window on the session timeline.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'Session start (spontaneous activity, no task events)',
    'off_start': None,
    'off_end': None,
}
```

```python
neural_trials = split_into_trials(dff_binned, TRIAL_BINS, axis=1)
```

iii. `CONVERSION_NOTES.md` Step 5 says the data are spontaneous activity without task events, so session start is the natural alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins neural and motion-energy traces by averaging every 10 frames at 30 Hz, yielding 3 Hz sampling or 333.33 ms bins.

ii.
```python
BIN_SIZE = 10
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000
```

```python
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
me_binned = bin_data(me_interp.reshape(1, -1), BIN_SIZE, axis=1).flatten()
```

iii. `CONVERSION_NOTES.md` Steps 1, 3, and 5 all cite the paper’s decoding analysis phrase “averaging in bins of 10 consecutive timestamps,” and use that as the reason for 10-frame averaging.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from a raw time variable. It is synthesized from the binned sample index together with the known frame rate and bin size.

ii.
```python
time_s = np.arange(n_timebins) * (BIN_SIZE / FRAME_RATE)  # seconds
```

iii. `CONVERSION_NOTES.md` Step 5 says the decoder input should be “time elapsed = bin_index * (10/30) seconds from session start.” The trajectory also notes that the imaging rate is constant at 30 Hz, so index-based timing is sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a regularly spaced time vector after rebinning, one value per 10-frame bin, then slices it into 60-second trials and casts each trial to `float32`.

ii.
```python
time_s = np.arange(n_timebins) * (BIN_SIZE / FRAME_RATE)
time_trials = split_into_trials(time_s.reshape(1, -1), TRIAL_BINS, axis=1)
...
'input': [t.astype(np.float32) for t in time_trials],
```

iii. The justification in `CONVERSION_NOTES.md` Step 5 is that decoder input should be time from session start and should vary continuously with the same binning as the neural data.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it has one value per binned neural sample and is split into trials with the same boundaries as the neural and output arrays.

ii.
```python
neural_trials = split_into_trials(dff_binned, TRIAL_BINS, axis=1)
me_trials = split_into_trials(me_discrete.reshape(1, -1), TRIAL_BINS, axis=1)
time_trials = split_into_trials(time_s.reshape(1, -1), TRIAL_BINS, axis=1)
```

iii. `CONVERSION_NOTES.md` Step 10 states that the first trial starts at `0.0s`, the first trial ends at `59.67s`, and the second trial starts at `60.0s`, which the AI treats as evidence of correct alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The converted output is derived from `move_deve/motion_energy_glob.npy`. The AI also loads `tstamps.npy` when preparing to handle length mismatches, although the interpolation function does not actually use timestamp values.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
...
me_interp = interpolate_motion_energy(me, tstamps, n_frames)
```

iii. `CONVERSION_NOTES.md` Step 2 lists `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy` as available behavior files. Step 5 justifies using motion energy as the decoder output and says missing camera frames should be interpolated before binning.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates the motion-energy trace to the neural frame count whenever the video trace is shorter, bins the interpolated signal into 10-frame averages, and discretizes the binned trace into five within-session percentile bins.

ii.
```python
def interpolate_motion_energy(me, tstamps, n_neural_frames):
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    me_float = me.astype(np.float64)
    neural_indices = np.arange(n_neural_frames)
    camera_indices = np.linspace(0, n_neural_frames - 1, len(me))
    me_interp = np.interp(neural_indices, camera_indices, me_float)
    return me_interp
```

```python
me_interp = interpolate_motion_energy(me, tstamps, n_frames)
me_binned = bin_data(me_interp.reshape(1, -1), BIN_SIZE, axis=1).flatten()
me_discrete, me_edges = discretize_motion_energy(me_binned)
```

iii. `CONVERSION_NOTES.md` Step 5 says “Motion energy interpolation: For sessions with missing camera frames, interpolate ME to match neural frame count before binning.” Trajectory step 32 says the timestamps appeared to use a strange scale, so the AI opted for a simpler interpolation-based alignment strategy.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes session-specific percentile edges at 0, 20, 40, 60, 80, and 100 percent, then uses `np.digitize` to assign each time bin to one of five discrete classes.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(me_binned, percentiles)
...
binned = np.digitize(me_binned, edges[1:-1], right=False)
binned = np.clip(binned, 0, n_bins - 1)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says “5 equal-percentile bins per session (quintiles), as specified in decoder task,” and Step 7 reports that each bin occupies about 20% of the data.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI assumes the camera and neural streams are approximately 1:1 in frame index and aligns them by resampling the motion-energy trace onto a uniformly spaced neural-frame index grid. It does not insert missing values at specific dropped-frame positions.

ii.
```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)
...
neural_indices = np.arange(n_neural_frames)
camera_indices = np.linspace(0, n_neural_frames - 1, len(me))
me_interp = np.interp(neural_indices, camera_indices, me_float)
```

iii. In trajectory step 32, the AI says the timestamps appear to be in an unusual unit and that the camera is triggered by the microscope, so a simple linear interpolation should be adequate when a few frames are missing. `CONVERSION_NOTES.md` Step 4 also records the mismatch issue and resolves it by interpolation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing video frames by length-matching motion energy to neural frames with linear interpolation. It does not implement any special repair for neural data. Partial trailing data shorter than a full 60-second trial is silently omitted because trial counts are computed with floor division.

ii.
```python
me_interp = interpolate_motion_energy(me, tstamps, n_frames)
```

```python
n_trials = n_timebins // TRIAL_BINS
neural_trials = split_into_trials(dff_binned, TRIAL_BINS, axis=1)
```

iii. `CONVERSION_NOTES.md` Step 4 identifies motion-energy length mismatches as the main data issue. Step 5 says the resolution is to interpolate motion energy before binning, and Step 10 cites a session with missing frames as a successful edge-case check.

## 6-a. What are the most time-consuming steps of the code?

i. The AI treats calcium preprocessing as the main expensive step, especially the custom PyTorch baseline computation performed in batches for every neuron. Session loading and binning are secondary costs.

ii.
```python
batch_size = 100
Flow = np.zeros_like(Fc)
n_batches = int(np.ceil(ncells / batch_size))

for n in range(n_batches):
    ...
    data = conv1d(...)
    ...
    data = max_pool1d(...)
```

```python
t_dfof = time.time() - t0
...
print(f"  {subj}/{sess_name}: ... load={t_load:.1f}s dfof={t_dfof:.1f}s bin={t_bin:.2f}s")
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly says the dF/F computation is the main inefficiency and notes that it uses CPU PyTorch for reliability.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves at least two clear Python loops that could be vectorized further: the batch loop in `compute_dfof` and the loop that slices one trial at a time in `split_into_trials`. It does not highlight a different loop in its notes because its motion-energy handling uses `np.interp` instead of repeated `np.insert`.

ii.
```python
for n in range(n_batches):
    nstart = n * batch_size
    nend = min((n + 1) * batch_size, ncells)
    ...
```

```python
for i in range(n_trials):
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(i * trial_length, (i + 1) * trial_length)
    trials.append(data[tuple(slices)])
```

iii. `CONVERSION_NOTES.md` Step 6 only mentions CPU dF/F as an inefficiency; it does not give a fuller vectorization analysis. The code itself is the main evidence here.

## 6-c. What processing does the code repeat multiple times?

i. There is no major repeated downstream-discarded processing in the conversion path. The main repeated work is simply the intended per-session pass through loading, preprocessing, binning, discretization, and trial splitting. Optional plotting repeats some summary computations for display, but only when requested.

ii.
```python
for subj in subjects_to_process:
    ...
    for sess_name in sessions:
        result = process_session(subj, sess_name, data_dir, ...)
```

```python
if show_processing and session_idx < 2:
    plot_processing(...)
```

iii. The AI does not call out any important redundant processing in `CONVERSION_NOTES.md`; the notes frame the pipeline as one pass per session plus optional diagnostics.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI performs some extra diagnostic work that is not needed for the saved dataset: it loads `tstamps.npy` even though the timestamp values are not used in interpolation, records timing values like `t_interp`, computes `me_edges` for plotting/logging, and can generate processing plots for the first two sessions. These do not affect the final saved arrays.

ii.
```python
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
...
t_interp = time.time() - t0
...
me_discrete, me_edges = discretize_motion_energy(me_binned)
...
if show_processing and session_idx < 2:
    plot_processing(...)
```

iii. `CONVERSION_NOTES.md` Steps 6, 7, and 10 emphasize validation and visualization, so these extra computations appear to have been included as sanity-check machinery rather than as necessary downstream processing.
