# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every directory under `/app/data` as a subject and every directory beneath each subject as a session, both sorted. For every session it loads `F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`, `motion_energy_glob.npy`, and `tstamps.npy`. It processes every discovered session and writes one output session per recording day.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))
...
sessions = sorted(d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d)))
...
F = np.load(os.path.join(s2p, 'F.npy'))
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
me_cam = np.load(os.path.join(mov, 'motion_energy_glob.npy')).astype(np.float64)
ts_cam = np.load(os.path.join(mov, 'tstamps.npy')).astype(np.float64) * 1000.0
```

iii. The trajectory says the AI inspected the full directory tree, README, notebook, methods, repository code, and every session's array dimensions. It concluded that the shipped Suite2p folders are already the curated Track2p outputs and therefore retained all 6 mice and all 41 daily sessions.

## 1-b. How are the data split into subjects?

i. Each immediate directory under `/app/data` is treated as one mouse; alphabetical sorting defines subject order, and `subject_idx` records the corresponding integer for each session.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))
for si, subj in enumerate(subjects):
    ...
    subject_idx.append(si)
```

iii. The AI verified that the six directories are the six mouse IDs (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`).

## 1-c. How are the data split into sessions?

i. Each sorted subdirectory of a mouse directory is one session/daily recording. Each is processed independently and appended as a separate output session.

ii.
```python
sessions = sorted(d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d)))
for di, sess in enumerate(sessions):
    dff, t, me, dt = load_session(os.path.join(subj_dir, sess))
...
neural.append(ntr)
inputs.append(ninp)
outputs.append(nout)
```

iii. The trajectory records inspection of all session names and identifies them as 6–7 consecutive daily recordings per mouse; sorting preserves chronological ISO-date order.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided into consecutive, non-overlapping blocks of 180 binned samples, corresponding nominally to 60 seconds (1800 raw frames at 30 Hz). Any incomplete tail is omitted. At least two trials are required.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * NOMINAL_FS / FRAMES_PER_BIN))  # 180
n_trials = n_bins // BINS_PER_TRIAL
assert n_trials >= 2
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    ntr.append(np.ascontiguousarray(dff[:, sl]))
```

iii. The user explicitly requested 60-second trials, and the data have no natural stimulus-defined trials. The AI therefore chose fixed contiguous blocks beginning at imaging onset.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filter is applied. The code only requires at least two complete trials per session and drops the final partial block.

ii.
```python
n_trials = n_bins // BINS_PER_TRIAL
assert n_trials >= 2
```

iii. The AI found no trial QC described for these continuous recordings and stated that the released data are already curated; only completeness and the decoder's minimum-trial requirement are enforced.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from Suite2p `F.npy` and `Fneu.npy`, with the per-session sampling rate read from `ops.npy`; `iscell.npy` is loaded for validation.

ii.
```python
F = np.load(os.path.join(s2p, 'F.npy'))
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
fs = float(ops['fs'])
dff = f_processing(F.astype(np.float64), Fneu.astype(np.float64), fs)
```

iii. The AI identified these as the repository's Track2p/Suite2p fluorescence inputs and used `ops['fs']` rather than assuming that every session is exactly 30 Hz.

## 2-b. How is the `neural` data processed?

i. The AI applies the Track2p repository's `F_processing`: `Fc = F - 0 * Fneu` (thus no effective neuropil subtraction), Gaussian smoothing with sigma 10 frames, a 60-second minimum-then-maximum baseline filter, and subtraction of that baseline. It then averages non-overlapping groups of 10 frames and casts to `float32`. Despite the label “dF/F,” there is no division by baseline.

ii.
```python
def f_processing(F, Fneu, fs, neucoeff=0.0, sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc, [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow
...
dff_b = bin_mean(dff, FRAMES_PER_BIN)
return dff_b.astype(np.float32), ...
```

iii. The trajectory shows that the AI located and inspected `track2p/gui/data_management.py::F_processing`, chose to reproduce its `neucoeff=0` behavior, and declined to z-score after experimentally testing it because z-scoring would depart from the paper/repository processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed in conversion. Instead, the code asserts that all released ROIs already have `iscell[:, 0] == 1`; all row-matched Track2p neurons are retained.

ii.
```python
assert np.all(iscell[:, 0] == 1), 'unexpected non-cell ROI in Track2p output'
```

iii. The AI inspected `iscell.npy` across sessions and concluded that the released Track2p files already contain only neurons tracked on every day of a mouse and already passing Suite2p's cell criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Trials are contiguous blocks starting at imaging onset, and neural samples are sliced on those shared block boundaries. Metadata describes alignment to the start of each 60-second trial block, with offsets from 0 to approximately 60.46 seconds based on the measured mean frame interval.

ii.
```python
sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
ntr.append(np.ascontiguousarray(dff[:, sl]))
...
'temporal_alignment_event':
    'start of the 60 s trial block; each session is cut into '
    'consecutive non-overlapping 1800-frame (60 s) blocks starting '
    'at imaging onset',
'off_start': 0.0,
'off_end': float(np.mean(bin_durations) * BINS_PER_TRIAL),
```

iii. The AI reasoned that the continuous, spontaneous-behavior experiment has no stimulus event, so block onset is the only meaningful trial alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural activity, motion energy, and timestamps are averaged in non-overlapping bins of 10 frames. The nominal resolution is about 333 ms, while metadata uses the measured average frame interval and reports about 335.88 ms.

ii.
```python
FRAMES_PER_BIN = 10
dff_b = bin_mean(dff, FRAMES_PER_BIN)
me_b = bin_mean(me, FRAMES_PER_BIN)
t_b = bin_mean(tstamps, FRAMES_PER_BIN)
...
time_bin_size_ms = float(np.mean(bin_durations) * 1000.0)
```

iii. The AI cites the paper's decoding method: both dF/F and behavior were denoised by averaging 10 consecutive timestamps. It measured the real acquisition rate (about 29.78 Hz) for accurate metadata.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is derived from each session's behavioral-camera `tstamps.npy`, converted to seconds. It is not synthesized from a fixed frame index.

ii.
```python
ts_cam = np.load(os.path.join(mov, 'tstamps.npy')).astype(np.float64) * 1000.0
...
t_b = bin_mean(tstamps, FRAMES_PER_BIN)
ninp.append(t[sl][None, :].astype(np.float32))
```

iii. By inspecting timestamp magnitudes and differences, the AI inferred that stored values must be multiplied by 1000 to produce a plausible ~33.59 ms interval, and used these measured times for elapsed session time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Camera timestamps are scaled by 1000, mapped back to the complete imaging-frame grid using inferred dropped-frame counts, linearly interpolated at missing frames, and averaged within each 10-frame bin. The resulting bin-center times are cast to `float32` when placed in trials.

ii.
```python
idx = camera_frame_index(ts_cam, n_frames)
tstamps = np.full(n_frames, np.nan)
tstamps[idx] = ts_cam
tstamps = interp_nan(tstamps)
t_b = bin_mean(tstamps, FRAMES_PER_BIN)
```

iii. The AI chose timestamp-derived bin centers to preserve the actual acquisition timing and to keep time valid across dropped camera frames.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Timestamps are first placed on the neural/imaging frame grid, interpolated, binned over the same groups of 10 frames, and sliced with exactly the same trial slice as neural activity.

ii.
```python
tstamps = np.full(n_frames, np.nan)
tstamps[idx] = ts_cam
tstamps = interp_nan(tstamps)
t_b = bin_mean(tstamps, FRAMES_PER_BIN)
...
ntr.append(np.ascontiguousarray(dff[:, sl]))
ninp.append(t[sl][None, :].astype(np.float32))
```

iii. The trajectory says camera and microscope acquisition were hardware-triggered together; reconstruction of the imaging-frame indices was checked to end exactly at `nframes - 1` in every affected session.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from `move_deve/motion_energy_glob.npy`; `move_deve/tstamps.npy` and the neural frame count are used to reconstruct its position on the imaging grid.

ii.
```python
me_cam = np.load(os.path.join(mov, 'motion_energy_glob.npy')).astype(np.float64)
ts_cam = np.load(os.path.join(mov, 'tstamps.npy')).astype(np.float64) * 1000.0
idx = camera_frame_index(ts_cam, n_frames)
```

iii. The AI identified the stored signal as global pixel-difference motion energy and used timestamps because camera frames can be dropped even though acquisition is hardware synchronized.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is expanded onto the full imaging grid, missing values (including frame zero) are linearly interpolated, groups of 10 frames are averaged, and the binned continuous values are discretized into session-specific quintiles.

ii.
```python
me = np.full(n_frames, np.nan)
me[idx] = me_cam
me[0] = np.nan
me = interp_nan(me)
me_b = bin_mean(me, FRAMES_PER_BIN)
...
cls = discretize_quintiles(me)
```

iii. The AI regarded frame-zero motion energy as undefined because it has no preceding video frame. Interpolation restores framewise alignment, and 10-frame averaging follows the paper before categorical discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four thresholds at the 20th, 40th, 60th, and 80th percentiles are computed separately from each session's binned motion-energy trace. `np.digitize` produces integer classes 0–4.

ii.
```python
def discretize_quintiles(x, nbins=N_OUTPUT_BINS):
    edges = np.percentile(x, np.linspace(0, 100, nbins + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int64)
```

iii. This directly implements the requested five equal-percentile bins selected per session. The trajectory confirms approximately equal class counts.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Timestamp gaps determine each camera sample's imaging-frame index. The signal is placed on a length-`n_frames` grid and missing frames are interpolated. Motion and neural arrays then undergo identical 10-frame binning and identical trial slicing.

ii.
```python
n_missing = np.round(ifi / dt).astype(int) - 1
idx = np.concatenate([[0], np.cumsum(1 + n_missing)])
...
me = np.full(n_frames, np.nan)
me[idx] = me_cam
me = interp_nan(me)
...
ntr.append(np.ascontiguousarray(dff[:, sl]))
nout.append(cls[sl][None, :])
```

iii. The AI verified that cumulative timestamp gaps reconstructed the exact final imaging index in every session and described linear interpolation as the way to retain synchronized indexing after dropped camera frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code detects dropped camera frames from timestamp gaps, inserts NaNs on the imaging grid, and linearly interpolates them (nearest valid value at an edge). It treats the first motion-energy value as missing, asserts timestamp/motion lengths and reconstructed counts, asserts all ROIs are cells, rejects an all-missing vector, discards incomplete bin/trial tails, and requires at least two trials.

ii.
```python
assert len(me_cam) == len(ts_cam)
assert idx[-1] == n_frames - 1
assert n_missing.sum() == n_frames - ncam
...
if bad.all():
    raise ValueError('all values missing')
x[bad] = np.interp(t[bad], t[~bad], x[~bad])
...
n = (x.shape[-1] // k) * k
x = x[..., :n]
```

iii. The trajectory reports 9 sessions with 1–148 dropped frames and documents explicit checks that timestamp-derived missing counts and final indices match the neural frame grid.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant conversion work is loading full-session arrays and applying Gaussian plus long-window minimum/maximum filters to every neuron. The trajectory's decoder-training runs were much longer than conversion but are validation, not part of `convert_data.py`.

ii.
```python
F = np.load(os.path.join(s2p, 'F.npy'))
...
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. The AI did not explicitly profile conversion. Its implementation and investigation indicate that full-array I/O and per-neuron baseline filtering are the computationally substantial conversion operations.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Subject/session iteration is inherently file-oriented. The per-trial loop could be replaced by reshaping full complete blocks, although separate trial arrays/lists would still need construction. Metadata percentile calculation is redundantly loop-like across sessions. Dropped-frame interpolation itself is already vectorized with `np.interp`, improving on repeated `np.insert`.

ii.
```python
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    ntr.append(np.ascontiguousarray(dff[:, sl]))
    ninp.append(t[sl][None, :].astype(np.float32))
    nout.append(cls[sl][None, :])
```

iii. The trajectory does not discuss loop vectorization directly. The chosen vectorized timestamp-gap calculation and interpolation show that the AI deliberately avoided inserting missing frames one at a time.

## 6-c. What processing does the code repeat multiple times?

i. Motion-energy percentile thresholds are calculated once to generate classes and then calculated again for `session_info`. Per-session baseline correction, frame reconstruction, interpolation, and binning are necessarily repeated for each recording.

ii.
```python
cls = discretize_quintiles(me)
...
'motion_energy_quintile_edges': [
    float(v) for v in np.percentile(
        me, np.linspace(0, 100, N_OUTPUT_BINS + 1)[1:-1])],
```

iii. No explicit justification was given for recomputing the percentile edges; it appears to be a small convenience cost for recording metadata.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `Fneu` is loaded and converted to `float64` even though `neucoeff=0`, so it cannot affect the neural output. `iscell` is used only for an assertion. Session percentile edges and extensive descriptive metadata are not required by decoder training. The final incomplete bins and trial tails are processed during session-level preprocessing/quantiling but later omitted from trial arrays; because quintile edges are computed before trial truncation, those omitted tail bins can still slightly affect retained labels.

ii.
```python
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
dff = f_processing(F.astype(np.float64), Fneu.astype(np.float64), fs)
# f_processing default: neucoeff=0.0
...
cls = discretize_quintiles(me)
n_trials = n_bins // BINS_PER_TRIAL
```

iii. The AI loaded `Fneu` to mirror the repository function signature and loaded `iscell` to verify its curation assumption. It did not discuss the small wasted tail processing or the fact that discarded tail bins participate in percentile estimation.
