# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every directory directly under `/app/data` as a subject and every directory beneath each subject as a session. For each session it loads Suite2p fluorescence, metadata, and cell labels plus global motion energy and camera timestamps. Trials are created later from the continuous session.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))
sessions = sorted(d for d in os.listdir(subject_dir)
                  if os.path.isdir(os.path.join(subject_dir, d)))
F = np.load(os.path.join(plane, 'F.npy')).astype(np.float64)
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
```

iii. The trajectory shows that the agent first inspected the directory tree, README, notebook, and every session's shapes. Its final explanation says the result contains all 41 sessions from six mice. It did not explicitly justify accepting every directory instead of filtering names.

## 1-b. How are the data split into subjects?

i. Each sorted top-level directory is treated as one mouse. Its enumeration index is stored once per session in `subject_idx`.

ii.
```python
for si, subject in enumerate(subjects):
    ...
    subject_idx.append(si)
```

iii. The agent inspected the directory structure and reported the six `jm*` mice. The implicit justification is that the data directory is organized one folder per animal.

## 1-c. How are the data split into sessions?

i. Each sorted subdirectory of a subject is one session/daily recording and becomes one outer list entry in `neural`, `input`, and `output`.

ii.
```python
for session in sessions:
    session_dir = os.path.join(subject_dir, session)
    ...
    neural.append(sess_neural)
    inputs.append(sess_input)
    outputs.append(sess_output)
```

iii. The agent inspected all session folders and their per-session files. Its final report characterizes the dataset as daily recordings and preserves each as a separate decoder session.

## 1-d. How are the data split into trials?

i. Continuous sessions are tiled from their beginning into consecutive, non-overlapping 60-second trials. With ten-frame averaging and approximately 30 Hz acquisition, each trial has 180 bins. Any incomplete tail is excluded by integer division.

ii.
```python
bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))
ntrials = nbins // bins_per_trial
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
```

iii. The final trajectory states that recordings are continuous and have no task structure, so fixed 60-second blocks implement the instruction. It also reports that the actual session durations divide evenly, so no trial remainder was lost.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filter. Every complete 60-second block is retained; only a hypothetical incomplete terminal block is omitted.

ii.
```python
ntrials = nbins // bins_per_trial
for tr in range(ntrials):
    ...
```

iii. The trajectory gives no separate trial-quality criterion. The data are spontaneous continuous recordings, and validation found valid complete trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `suite2p/plane0/F.npy` and `Fneu.npy`; `ops.npy` supplies sampling metadata and `iscell.npy` is checked for ROI curation.

ii.
```python
F = np.load(os.path.join(plane, 'F.npy')).astype(np.float64)
Fneu = np.load(os.path.join(plane, 'Fneu.npy')).astype(np.float64)
ops = np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item()
iscell = np.load(os.path.join(plane, 'iscell.npy'))
```

iii. The agent inspected the supplied Track2p `F_processing` implementation and Suite2p metadata to identify the source traces and parameters.

## 2-b. How is the `neural` data processed?

i. It uses Track2p-style baseline-corrected fluorescence: with `neucoeff=0`, `Fc` is raw `F`; a Gaussian-smoothed trace is passed through minimum then maximum filters using a 60-second window, and the resulting baseline is subtracted. It is not divided by the baseline despite being called dF/F. The traces are then averaged over ten frames and cast to contiguous float32 trial arrays.

ii.
```python
Fc = F - neucoeff * Fneu
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
return Fc - Flow
...
dff_b = bin_average(dff, BIN_FRAMES)
sess_neural.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
```

iii. The agent explicitly chose the implementation in `track2p/gui/data_management.py::F_processing`, including its zero neuropil coefficient, as the closest interpretation of “same processing.” It tested and rejected z-scoring because it departed from the paper pipeline for only a small decoder gain.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed during conversion. Instead, the code asserts that every provided ROI has `iscell[:,0] == 1`, relying on the distributed data already being Suite2p-classified and Track2p-matched across days.

ii.
```python
assert np.all(iscell[:, 0] == 1), f'unexpected non-cell ROI in {session_dir}'
```

iii. The final rationale says the released Track2p output is already curated at the Suite2p 0.5 threshold and matched across every day, so further filtering would be redundant; the assertion makes that assumption fail loudly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event. Neural bins begin at imaging onset and are sliced into blocks; each trial is aligned to the start of its own 60-second block. Metadata describes that event and assigns offsets 0 to 60 seconds.

ii.
```python
sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
sess_neural.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
...
'off_start': 0.0,
'off_end': TRIAL_SEC,
```

iii. The agent explained that the recordings are continuous and lack task events, so consecutive blocks from frame zero are the only instructed alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive approximately 30 Hz frames are averaged, producing 3 Hz data and a declared 333.33 ms bin size. Incomplete ten-frame tails are dropped.

ii.
```python
BIN_FRAMES = 10
return x.reshape(*x.shape[:-1], nbins, bin_frames).mean(axis=-1)
...
'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,
```

iii. The agent cites the paper's decoding analysis, which averages both dF/F and behavior in ten-timestamp bins for denoising.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is generated from the rebinned sample index, `BIN_FRAMES`, and the per-session sampling rate in `ops['fs']`; it is not loaded as a raw timestamp stream.

ii.
```python
fs = float(ops['fs'])
t = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
```

iii. The agent treated imaging onset and the stable acquisition rate as the session clock because the requested input is elapsed session time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. It computes the center time of each ten-frame bin in seconds. The sequence remains continuous across trial boundaries and trial slices are stored as float32 arrays.

ii.
```python
t = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
sess_input.append(np.ascontiguousarray(t[None, sl], dtype=np.float32))
```

iii. The final explanation emphasizes continuous time since imaging onset, rather than resetting at each artificial trial, because the instruction asks for time from session start. Center timestamps represent the averaged samples.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One center timestamp is generated per rebinned neural column, and exactly the same trial slice is applied to both.

ii.
```python
nbins = dff_b.shape[1]
t = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
sess_neural.append(...dff_b[:, sl]...)
sess_input.append(...t[None, sl]...)
```

iii. The common bin count and slice guarantee column-wise correspondence; this is implicit in the implementation rather than separately discussed in the trajectory.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses precomputed global motion energy from `move_deve/motion_energy_glob.npy` and camera timestamps from `move_deve/tstamps.npy` for dropped-frame correction.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The agent inspected the README, notebook, timestamp gaps, interframe intervals, and all session lengths before selecting the timestamp-based source and correction.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Timestamp gaps estimate missing camera frames. The agent reconstructs the full frame grid, linearly interpolates internal missing values, trims excess terminal samples or pads missing terminal samples with the last value, averages ten frames per bin, and then discretizes the session.

ii.
```python
n_missing = np.round(np.diff(tstamps) / dt).astype(int) - 1
idx = np.concatenate([[0], np.cumsum(1 + n_missing)])
full = np.full(idx[-1] + 1, np.nan)
full[idx] = me
full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), full[~bad])
me_b = bin_average(me, BIN_FRAMES)
```

iii. The README recommended interpolation, and the agent found 276 dropped camera frames in its dataset-wide inspection. It used timestamp gap multiples to handle more than one missing frame correctly.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After temporal averaging, each complete session is independently thresholded at its 20th, 40th, 60th, and 80th percentiles. `np.digitize` produces integer categories 0–4.

ii.
```python
edges = np.percentile(x, np.linspace(0, 100, nbins + 1)[1:-1])
return np.digitize(x, edges).astype(np.int64)
```

iii. The decoder task explicitly requires five equal-percentile bins selected per session. The agent verified that the resulting classes are 20% each.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera timestamps reconstruct a one-to-one imaging-frame grid because the behavior camera is triggered by imaging. After correction both streams are ten-frame averaged, then the identical trial slice is used for neural and output arrays. No lag correction is applied.

ii.
```python
me = load_motion_energy(session_dir, nframes)
dff_b = bin_average(dff, BIN_FRAMES)
me_b = bin_average(me, BIN_FRAMES)
...
sess_output.append(np.ascontiguousarray(me_cat[None, sl]))
```

iii. The final rationale says acquisition is hardware-synchronized. The agent also checked cross-correlation and interpreted the observed 0.3–0.7 s neural lag as calcium kinetics, so it intentionally applied no temporal shift.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Internal dropped camera frames are linearly interpolated. A reconstructed behavior trace longer than imaging is trimmed; a shorter trace is padded with its last value. Neural frame count is asserted against `ops['nframes']`, all ROIs are asserted to be cells, incomplete bin/trial tails are dropped, and the output directory assumptions are fixed paths.

ii.
```python
if len(full) > nframes:
    full = full[:nframes]
elif len(full) < nframes:
    full = np.concatenate([full, np.full(nframes - len(full), full[-1])])
assert nframes == ops['nframes']
```

iii. The agent followed the dataset README for camera drops and used assertions for violated dataset assumptions. Its inspection showed terminal length discrepancies are only a handful of frames.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant computation is full-session neural baseline correction, especially the Gaussian and 60-second sliding minimum/maximum filters for every neuron. Loading/casting large arrays and serializing the full dataset are secondary costs. Decoder training was time-consuming in the trajectory but is validation, not conversion.

ii.
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. The agent did not explicitly profile conversion. This assessment follows the array sizes it inspected and the global filtering operations in its code.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Session/subject loops are required for heterogeneous files and neuron counts. The trial loop could be replaced by reshaping each session into `(trials, neurons, time)` and list conversion, although its cost is small because it mostly creates 60-second slices. Core binning and missing-frame interpolation are already vectorized.

ii.
```python
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    sess_neural.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
```

iii. The trajectory gives no explicit vectorization discussion. The agent did choose a vectorized preallocated grid plus `np.interp`, avoiding the reference's repeated `np.insert` loop.

## 6-c. What processing does the code repeat multiple times?

i. Motion-energy percentiles are calculated once to categorize data and again to populate `session_info`. Directory loading, baseline correction, binning, time generation, and array conversion repeat once per session by necessity.

ii.
```python
me_cat = discretize_percentile(me_b, N_OUTPUT_BINS)
...
'motion_energy_bin_edges': np.percentile(
    me_b, np.linspace(0, 100, N_OUTPUT_BINS + 1)[1:-1]).tolist(),
```

iii. No justification is recorded for recomputing the edges; it appears to be a small convenience cost for metadata.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `Fneu` is loaded and cast to float64 but contributes nothing because `neucoeff=0`. `iscell` is used only for an assertion; `ops` fields and session statistics beyond those needed for conversion become descriptive metadata. Temporary float64 full-session arrays are later cast to float32. The extra percentile calculation is metadata-only.

ii.
```python
Fneu = np.load(os.path.join(plane, 'Fneu.npy')).astype(np.float64)
dff = f_processing(F, Fneu, fs=ops['fs'])  # default neucoeff=0.0
...
sess_neural.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
```

iii. The docstring says `Fneu` is loaded for completeness, and the cell file is used to make the assumed prior curation explicit. These checks aid provenance even though their values are not decoder features.
