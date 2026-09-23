# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script scans every top-level directory in `/app/data` as a subject, then every subdirectory inside each subject as a session. For each session it loads calcium files from `suite2p/plane0` and behavioral motion-energy files from `move_deve`. Trials are not loaded directly; the full session is loaded first and later split into 60 s blocks.

ii.
```python
def main():
    subjects = sorted(d for d in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, d)))
    ...
    for si, subject in enumerate(subjects):
        subject_dir = os.path.join(DATA_DIR, subject)
        sessions = sorted(d for d in os.listdir(subject_dir)
                          if os.path.isdir(os.path.join(subject_dir, d)))
```

```python
def load_neural(session_dir):
    plane = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(plane, 'F.npy')).astype(np.float64)
    Fneu = np.load(os.path.join(plane, 'Fneu.npy')).astype(np.float64)
    ops = np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item()
    iscell = np.load(os.path.join(plane, 'iscell.npy'))
```

```python
def load_motion_energy(session_dir, nframes):
    move_dir = os.path.join(session_dir, 'move_deve')
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
    tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. In the trajectory, the agent said it was following the Track2p repository and the dataset layout. Its final summary says it used the “provided ROIs” and camera timestamps, and earlier inspection steps show it explicitly checked the subject/session tree and the contents of `suite2p` and `move_deve`.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted top-level directories under `/app/data`. The code does not filter for a `jm*` prefix; it assumes every directory there is a mouse.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))
```

iii. In the trajectory, the agent summarized the result as “6 mice (jm031, jm032, jm038, jm039, jm040, jm046)”, so its working assumption was that every top-level directory in the dataset corresponded to one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are the sorted subdirectories inside each subject directory. Each session directory is processed independently and becomes one element of the top-level `neural`, `input`, and `output` lists.

ii.
```python
for si, subject in enumerate(subjects):
    subject_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted(d for d in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, d)))

    for session in sessions:
        session_dir = os.path.join(subject_dir, session)
```

iii. The trajectory shows the agent enumerated daily recordings inside each mouse folder and reported the final dataset as 41 sessions, indicating that one session equals one daily recording directory.

## 1-d. How are the data split into trials?

i. The AI treats each continuous session as a sequence of non-overlapping 60 s blocks after temporal binning. With 30 Hz sampling and 10-frame bins, that gives 180 bins per trial. It keeps only the full 60 s blocks via integer division.

ii.
```python
BIN_FRAMES = 10
TRIAL_SEC = 60.0
...
bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))
ntrials = nbins // bins_per_trial

sess_neural, sess_input, sess_output = [], [], []
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    sess_neural.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
    sess_input.append(np.ascontiguousarray(t[None, sl], dtype=np.float32))
    sess_output.append(np.ascontiguousarray(me_cat[None, sl]))
```

iii. In the trajectory, the agent explicitly justified this by saying the recordings are continuous with “no task structure,” so sessions should be “tiled into consecutive non-overlapping 60 s blocks.”

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality-control filter is applied. All full 60 s blocks are kept.

ii.
```python
ntrials = nbins // bins_per_trial
...
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    ...
```

iii. The trajectory contains no separate trial-rejection rule. The agent’s final summary only discusses dropped camera frames and motion-energy interpolation, not trial exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from `F.npy` and `Fneu.npy`, and also reads `ops.npy` for the sampling rate and `iscell.npy` for a consistency check. The final signal is called `dff` in the code.

ii.
```python
F = np.load(os.path.join(plane, 'F.npy')).astype(np.float64)
Fneu = np.load(os.path.join(plane, 'Fneu.npy')).astype(np.float64)
ops = np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item()
iscell = np.load(os.path.join(plane, 'iscell.npy'))
...
dff = f_processing(F, Fneu, fs=ops['fs'])
```

iii. In the trajectory, the agent said its neural signal was “dF/F computed exactly as the track2p repo does it” and that all provided ROIs were already curated Track2p-tracked cells.

## 2-b. How is the `neural` data processed?

i. The AI applies a custom Track2p-style baseline-correction function rather than the reference converter’s Suite2p preprocessing call. It sets `neucoeff=0.0`, so there is no neuropil subtraction, smooths with a Gaussian filter, then applies a min filter and max filter over a 60 s window, and finally subtracts that baseline from `F`.

ii.
```python
def f_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin',
                 sig_baseline=10.0, win_baseline=60.0, prctile_baseline=8.0):
    Fc = F - neucoeff * Fneu

    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    ...
    return Fc - Flow
```

```python
dff = f_processing(F, Fneu, fs=ops['fs'])
```

iii. The trajectory justification is explicit. The agent wrote that it was using the Track2p repository’s `F_processing` implementation, with “neuropil coefficient 0” and maximin baseline correction, because it believed this matched the paper’s “baseline corrected fluorescence traces as our dF/F.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not drop any neurons in code. Instead, it asserts that every ROI already has `iscell[:, 0] == 1`, then keeps all of them.

ii.
```python
iscell = np.load(os.path.join(plane, 'iscell.npy'))
...
assert np.all(iscell[:, 0] == 1), f'unexpected non-cell ROI in {session_dir}'
...
dff = f_processing(F, Fneu, fs=ops['fs'])
```

iii. The agent’s final summary says the distributed Track2p output already contains only cells passing the Suite2p classifier and matched across days, so “no further neuron curation is needed.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI effectively aligns each neural trial to the start of each artificial 60 s block, not to a stimulus or behavioral event. In metadata it describes the alignment event as the start of each block, tiling the continuous recording from imaging onset.

ii.
```python
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    sess_neural.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
```

```python
'metadata': {
    'temporal_alignment_event': (
        'start of each 60 s block of the continuous recording; blocks tile the '
        'session from imaging onset (frame 0 of the 2-photon acquisition, which '
        'also triggers the behaviour camera)'),
    'off_start': 0.0,
    'off_end': TRIAL_SEC,
```

iii. The trajectory says there is “no task structure,” so the agent chose fixed-length continuous blocks and described them as aligned to block starts.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural data and motion energy by averaging 10 consecutive frames, converting 30 Hz to 3 Hz. The resulting time bin size is 333.33 ms.

ii.
```python
BIN_FRAMES = 10
...
dff_b = bin_average(dff, BIN_FRAMES)
me_b = bin_average(me, BIN_FRAMES)
```

```python
'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,  # 333.33 ms
```

iii. The agent’s final summary states that this follows the paper’s decoding analysis exactly: both dF/F and behavior are “averaged over 10 consecutive frames.”

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from a dedicated raw timestamp variable. It is derived from the session sampling rate in `ops['fs']` and the bin index after 10-frame averaging.

ii.
```python
ops = np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item()
...
fs = float(ops['fs'])
...
t = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
```

iii. The trajectory justification says the input should be “time in seconds since imaging onset,” continuous across trials, because the task requested time from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one time value per binned sample, using the center of each 10-frame bin rather than the left edge. The resulting time series is continuous across the full session and is later sliced into trials without resetting to zero inside each trial.

ii.
```python
t = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
...
sess_input.append(np.ascontiguousarray(t[None, sl], dtype=np.float32))
```

iii. In the trajectory, the agent explicitly said it chose “bin-centre time in seconds since imaging onset” and kept it “continuous across trials rather than reset per trial.”

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed on the same temporally binned grid as the neural data and then sliced with the exact same trial slices, so each time point corresponds one-to-one with a neural bin.

ii.
```python
dff_b = bin_average(dff, BIN_FRAMES)
...
t = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
...
sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
sess_neural.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
sess_input.append(np.ascontiguousarray(t[None, sl], dtype=np.float32))
```

iii. The trajectory justification is implicit in the agent’s final summary: it described the input as the continuous time axis of the same binned session used for decoding.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `motion_energy_glob.npy` and camera timestamps from `tstamps.npy`. The script does not use `interframe_int.npy`.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The trajectory shows the agent inspected both `tstamps.npy` and `interframe_int.npy`, then chose timestamps. In its final summary it justified this by saying dropped frames should be located from `tstamps.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI reconstructs a full camera-frame grid from timestamp gaps, inserts `NaN` at missing positions, linearly interpolates across dropped frames, trims or pads to the neural frame count, averages into 10-frame bins, and finally discretizes the binned trace into 5 percentile bins within each session.

ii.
```python
dt = np.median(np.diff(tstamps))
n_missing = np.round(np.diff(tstamps) / dt).astype(int) - 1
idx = np.concatenate([[0], np.cumsum(1 + n_missing)])
full = np.full(idx[-1] + 1, np.nan)
full[idx] = me

bad = np.isnan(full)
if bad.any():
    full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), full[~bad])

if len(full) > nframes:
    full = full[:nframes]
elif len(full) < nframes:
    full = np.concatenate([full, np.full(nframes - len(full), full[-1])])
```

```python
me_b = bin_average(me, BIN_FRAMES)
...
me_cat = discretize_percentile(me_b, N_OUTPUT_BINS)
```

iii. The trajectory justification is explicit: the agent said videography was hardware-triggered, dropped frames were inferred from timestamp gaps, then linearly interpolated, and only after that was motion energy binned and discretized.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After temporal binning, motion energy is discretized into 5 equal-percentile bins computed separately for each session. `np.digitize` assigns integer category labels 0 through 4.

ii.
```python
def discretize_percentile(x, nbins):
    edges = np.percentile(x, np.linspace(0, 100, nbins + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int64)
```

```python
me_cat = discretize_percentile(me_b, N_OUTPUT_BINS)
...
'output_values': [[f'quintile_{i + 1}' for i in range(N_OUTPUT_BINS)]],
```

iii. In the trajectory, the agent justified this as “5 equal-percentile bins” computed per session, and noted that this gives exactly balanced quintiles inside each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by reconstructing the camera trace onto the imaging-frame grid, forcing the result to have exactly `nframes`, then binning both streams identically and slicing them with the same trial boundaries. It does not apply any lag correction after checking cross-correlations.

ii.
```python
me = load_motion_energy(session_dir, nframes)
...
dff_b = bin_average(dff, BIN_FRAMES)
me_b = bin_average(me, BIN_FRAMES)
...
sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
sess_neural.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
sess_output.append(np.ascontiguousarray(me_cat[None, sl]))
```

iii. The trajectory justification says the camera was hardware-triggered from the 2-photon acquisition, so the correct alignment is frame-for-frame after dropped-frame interpolation. The agent also reports checking cross-correlation and deciding not to shift the traces.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main error-handling logic is for behavioral video. Missing camera frames are inferred from timestamp gaps and linearly interpolated. If the reconstructed camera trace is still longer than the neural trace it is truncated, and if shorter it is padded with its last value. Leftover frames that do not fill a whole 10-frame bin are dropped by the binning function. The code does not raise an error if a mismatch remains possible before trimming/padding.

ii.
```python
bad = np.isnan(full)
if bad.any():
    full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), full[~bad])

if len(full) > nframes:
    full = full[:nframes]
elif len(full) < nframes:
    full = np.concatenate([full, np.full(nframes - len(full), full[-1])])
```

```python
def bin_average(x, bin_frames):
    nbins = x.shape[-1] // bin_frames
    x = x[..., :nbins * bin_frames]
    return x.reshape(*x.shape[:-1], nbins, bin_frames).mean(axis=-1)
```

iii. The trajectory says the agent deliberately interpolated dropped frames from camera timestamps and trimmed/padded to `nframes`. It justified not applying any further correction by saying the residual lag visible in cross-correlation matched calcium kinetics rather than misalignment.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant costs are session-by-session calcium preprocessing over all neurons and frames, loading large `.npy` arrays from disk, and serializing the final large pickle. The custom `f_processing` function applies multiple passes over the full fluorescence matrix.

ii.
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
...
F = np.load(os.path.join(plane, 'F.npy')).astype(np.float64)
Fneu = np.load(os.path.join(plane, 'Fneu.npy')).astype(np.float64)
...
with open(OUT_FILE, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory does not contain an explicit performance analysis. The closest evidence is that the agent checked the final pickle size and ran full decoder training, but it did not separately justify runtime hotspots.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loop is trial assembly: for each session, the code iterates over trials and appends slices one by one. This could be partially vectorized by reshaping full-session arrays into `(n_trials, ...)` blocks before converting them to the required list-of-trials format. Subject/session iteration is inherently structural rather than a vectorization target.

ii.
```python
sess_neural, sess_input, sess_output = [], [], []
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    sess_neural.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
    sess_input.append(np.ascontiguousarray(t[None, sl], dtype=np.float32))
    sess_output.append(np.ascontiguousarray(me_cat[None, sl]))
```

iii. The trajectory gives no explicit efficiency justification here. Notably, unlike the reference solution, the AI already avoided a repeated `np.insert` loop for dropped-frame interpolation by using a vectorized timestamp-grid reconstruction.

## 6-c. What processing does the code repeat multiple times?

i. The code recomputes motion-energy percentile edges twice per session: once inside `discretize_percentile` to create categories, and again when storing `motion_energy_bin_edges` in `session_info`. It also repeatedly converts per-trial slices to contiguous arrays inside the trial loop.

ii.
```python
def discretize_percentile(x, nbins):
    edges = np.percentile(x, np.linspace(0, 100, nbins + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int64)
```

```python
'motion_energy_bin_edges': np.percentile(
    me_b, np.linspace(0, 100, N_OUTPUT_BINS + 1)[1:-1]).tolist(),
```

iii. The trajectory does not mention this repetition explicitly. It appears to be incidental rather than a stated choice.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is loading and checking `iscell.npy` even though no neurons are filtered; loading `Fneu.npy` even though `neucoeff=0.0` makes it irrelevant to the final signal; and computing/storing extensive metadata (`session_info`, percentile edges, descriptive strings) that the downstream decoder does not use. The script also computes motion-energy bin edges a second time just for metadata.

ii.
```python
Fneu = np.load(os.path.join(plane, 'Fneu.npy')).astype(np.float64)
ops = np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item()
iscell = np.load(os.path.join(plane, 'iscell.npy'))
...
assert np.all(iscell[:, 0] == 1), f'unexpected non-cell ROI in {session_dir}'
...
dff = f_processing(F, Fneu, fs=ops['fs'])
```

```python
session_info.append({
    ...
    'motion_energy_bin_edges': np.percentile(
        me_b, np.linspace(0, 100, N_OUTPUT_BINS + 1)[1:-1]).tolist(),
})
```

iii. The trajectory explicitly says `Fneu` was loaded “only for completeness” under the Track2p interpretation and that all ROIs were already curated, which explains why these steps were kept despite not affecting decoder inputs/outputs.
