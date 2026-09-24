# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers every `/app/data/jm*` subject directory and every subdirectory beneath each subject as a session. For each session it loads `suite2p/plane0/F.npy`, `Fneu.npy`, and `move_deve/motion_energy_glob.npy`; it processes sessions sequentially and saves all qualifying sessions in one pickle.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
...
sessions = sorted([d for d in os.listdir(subject_dir)
                  if os.path.isdir(os.path.join(subject_dir, d))])
...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The trajectory says the AI surveyed all mouse/session shapes and inferred the standard directory convention: six `jm*` mice, daily session folders, Suite2p fluorescence, and per-session motion energy. It reported 41 sessions and 1090 trials after conversion.

## 1-b. How are the data split into subjects?

i. A subject is each sorted directory whose name starts with `jm`. The enumeration index is stored once for every retained session in `subject_idx`.

ii.
```python
for si, subject in enumerate(subjects):
    ...
    all_subject_idx.append(si)
```

iii. The AI concluded from the directory survey and naming convention that each `jm*` folder is one mouse.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory of a subject is treated as one session. One nested list is appended to each of `neural`, `input`, and `output` per retained daily recording.

ii.
```python
for sess in sessions:
    sess_dir = os.path.join(subject_dir, sess)
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. The AI observed that the subject folders contain daily recording folders with matching Suite2p and behavioral files, and used sorted order for determinism.

## 1-d. How are the data split into trials?

i. Each continuous session is divided into non-overlapping 60-second trials after 10-frame binning. At 3 Hz this is 180 bins per trial. An incomplete tail is implicitly discarded.

ii.
```python
bins_per_trial = int(TRIAL_DUR * FS / BIN_SIZE)  # 180
n_trials = n_bins_total // bins_per_trial
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    session_neural.append(dff_binned[:, start:end])
```

iii. The trajectory explicitly identifies fixed, non-overlapping 60-second segments as required by the task and calculates 180 binned samples per trial.

## 1-e. How are trials filtered based on quality controls?

i. Individual trials are not quality-filtered. Sessions producing fewer than two complete trials are skipped, and incomplete final fragments are discarded through floor division.

ii.
```python
n_trials = n_bins_total // bins_per_trial
if n_trials < 2:
    print(f"Skipping {subject}/{sess}: only {n_trials} trials")
    continue
```

iii. The two-trial check follows the target format's explicit minimum for decoder evaluation. The trajectory otherwise states that no extra filtering is needed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from raw ROI fluorescence `F.npy` and neuropil fluorescence `Fneu.npy` in Suite2p `plane0`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
dff = compute_dff(F, Fneu)
```

iii. The AI chose these because the paper uses baseline-corrected fluorescence and the files/parameters are Suite2p outputs.

## 2-b. How is the `neural` data processed?

i. It subtracts 0.7 times neuropil, estimates a maximin baseline by Gaussian smoothing (sigma 10 frames), a 60-second minimum filter, and a 60-second maximum filter, then subtracts that baseline. Finally it averages non-overlapping groups of 10 frames.

ii.
```python
Fc = F - neucoeff * Fneu
win = int(win_baseline * fs)
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
dff = Fc - Flow
...
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
```

iii. The AI initially divided by the baseline, discovered extreme values from near-zero/negative baselines, then checked Suite2p/Track2p behavior and changed to baseline subtraction. It justified the parameters from `ops.npy` and the paper's “baseline corrected fluorescence” wording.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed; all rows of `F.npy` are retained. `iscell.npy` is not loaded or applied.

ii.
```python
n_neurons, n_frames = F.shape
...
all_brain_region_idx.append(np.zeros(n_neurons, dtype=int))
```

iii. The AI inspected `iscell` and reasoned that the supplied Track2p data already contains tracked neurons and all `iscell` flags are one, so additional filtering would have no effect.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the imaging-session start. Consecutive 60-second slices are taken from bin zero onward; there is no biological event realignment. Metadata calls the event `Start of imaging session` and records trial offsets 0–60 seconds.

ii.
```python
session_neural.append(dff_binned[:, start:end])
...
'temporal_alignment_event': 'Start of imaging session',
'off_start': 0.0,
'off_end': float(TRIAL_DUR),
```

iii. The AI recognized that these recordings are continuous and that the task defines artificial 60-second trials, so session start is the only meaningful anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The original 30 Hz streams are averaged in non-overlapping groups of 10 frames, yielding 3 Hz data and 333.33 ms bins. Short tails below 10 frames are trimmed.

ii.
```python
BIN_SIZE = 10
return data[:, :trimmed].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)
...
'time_bin_size': BIN_SIZE / FS * 1000,
```

iii. The AI cites the paper's decoding method: both dF/F and behavior traces were denoised by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is synthetic, derived from the post-binning sample index, `BIN_SIZE`, and the assumed 30 Hz sampling rate—not from a stored timestamp array.

ii.
```python
time_bin_dur = BIN_SIZE / FS
time_vec = np.arange(n_bins_total) * time_bin_dur
```

iii. The trajectory confirms the 30 Hz rate from `ops.npy` and describes this as time elapsed from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Each binned index is multiplied by 10/30 seconds. The resulting continuous session-level vector is sliced into trials and reshaped to `(1, 180)`.

ii.
```python
time_vec = np.arange(n_bins_total) * time_bin_dur
session_input.append(time_vec[start:end].reshape(1, -1))
```

iii. The AI wanted time to continue across trials because the requested variable is elapsed time from the beginning of the session, not time within a trial.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time and neural arrays share the same binned indices and identical `start:end` trial slices.

ii.
```python
session_neural.append(dff_binned[:, start:end])
session_input.append(time_vec[start:end].reshape(1, -1))
```

iii. The AI reasoned that constructing time directly from the neural bin index guarantees sample-wise alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It is derived from each session's precomputed `move_deve/motion_energy_glob.npy`. The code does not load `interframe_int.npy` or `tstamps.npy` during conversion.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
me = interpolate_me(me, n_frames)
```

iii. The AI identified this as the supplied global behavioral-video motion-energy trace. It inspected timestamp/interval files while investigating missing frames but ultimately did not use them in the converter.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If its length differs from neural data, the entire motion-energy trace is linearly resampled on normalized coordinates to the neural frame count. It is then averaged over 10-frame bins and discretized using per-session percentiles.

ii.
```python
x_orig = np.linspace(0, 1, len(me))
x_target = np.linspace(0, 1, target_len)
return np.interp(x_target, x_orig, me)
...
me_binned = bin_data(me.reshape(1, -1), BIN_SIZE, axis=1).squeeze()
```

iii. The trajectory notes that dropped frames were rare and cites the data README's permission to interpolate. Because it did not resolve the timestamp units, it chose simple whole-trace interpolation.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Five percentile bins are computed separately for every session from the full binned motion-energy trace. `np.digitize` against the 20th, 40th, 60th, and 80th percentile boundaries returns integer classes 0–4.

ii.
```python
percentiles = np.linspace(0, 100, N_BINS_OUTPUT + 1)
bin_edges = np.percentile(me_binned, percentiles)
me_discrete = np.digitize(me_binned, bin_edges[1:-1])
```

iii. The AI followed the task's requirement for five equal-percentile bins selected per session and bins before trial splitting so all trials in a session share thresholds.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is forced to the raw neural frame count by global linear resampling, both streams are separately averaged in corresponding 10-frame bins, and identical trial indices slice them.

ii.
```python
me = interpolate_me(me, n_frames)
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
me_binned = bin_data(me.reshape(1, -1), BIN_SIZE, axis=1).squeeze()
...
session_output.append(me_discrete[start:end].reshape(1, -1))
```

iii. The AI assumed synchronous 30 Hz acquisition and used interpolation to reconcile missing camera frames. Its rationale was that matching lengths before common binning and slicing would align streams.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Any motion-energy length mismatch is repaired by global linear interpolation to the neural length. Incomplete bin and trial tails are dropped. Sessions with fewer than two complete trials are skipped; there is no explicit NaN handling or post-repair assertion.

ii.
```python
if len(me) == target_len:
    return me
...
return np.interp(x_target, x_orig, me)
...
if n_trials < 2:
    continue
```

iii. The AI followed the README's suggestion to interpolate missing video data and the decoder's minimum-two-trials requirement. It selected normalized whole-trace interpolation after finding the auxiliary timing units unclear.

## 6-a. What are the most time-consuming steps of the code?

i. The full-session Gaussian/minimum/maximum baseline filters over every neuron are the principal compute cost. Loading large arrays and repeatedly resampling/binning every session are secondary costs.

ii.
```python
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

iii. The trajectory does not explicitly profile runtime. It does, however, focus on reproducing Suite2p preprocessing and reruns the complete conversion after correcting baseline normalization; this assessment follows directly from the full-array filtering operations.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-construction loop could be replaced by reshape/split operations for neural, input, and output arrays. Subject/session traversal is naturally file-oriented and not meaningfully vectorizable.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    session_neural.append(dff_binned[:, start:end])
    session_input.append(time_vec[start:end].reshape(1, -1))
    session_output.append(me_discrete[start:end].reshape(1, -1))
```

iii. The AI gave no explicit efficiency justification. The loop is simple and only creates views/small wrappers, so vectorization would mainly reduce Python overhead.

## 6-c. What processing does the code repeat multiple times?

i. Directory setup, file loading, baseline filtering, motion-energy interpolation, 10-frame binning, percentile computation, time-vector construction, and trial slicing are repeated independently for every session.

ii.
```python
for si, subject in enumerate(subjects):
    ...
    for sess in sessions:
        ...
        dff = compute_dff(F, Fneu)
        me = interpolate_me(me, n_frames)
        dff_binned = bin_data(dff, BIN_SIZE, axis=1)
```

iii. The trajectory does not call this out as waste; per-session repetition is required because sessions have different neuron counts, lengths, missing frames, and percentile thresholds.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Baseline correction and binning are required, but data in incomplete final 10-frame bins and incomplete final 60-second trials is computed/loaded and then discarded. Motion-energy categories for the incomplete final trial are also calculated but never saved. The full trace is globally resampled even when only a few localized frames are missing.

ii.
```python
trimmed = n_bins * bin_size
return data[:, :trimmed].reshape(...).mean(axis=2)
...
n_trials = n_bins_total // bins_per_trial
```

iii. The AI did not explicitly discuss discarded work. Its stated policy was to discard incomplete final trials to maintain uniform shape; the extra global resampling follows from its simplified missing-frame strategy.
