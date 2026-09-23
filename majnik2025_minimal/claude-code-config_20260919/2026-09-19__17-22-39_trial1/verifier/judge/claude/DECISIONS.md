# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as sorted directories in the data directory. Sessions are sorted subdirectories within each subject folder. For each session, calcium data is loaded from `suite2p/plane0/F.npy` and `Fneu.npy`, plus `ops.npy` and `iscell.npy`. Motion energy is loaded from `move_deve/motion_energy_glob.npy`, and camera timestamps from `move_deve/tstamps.npy`.

ii.
```python
def load_neural(session_dir):
    plane = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(plane, 'F.npy')).astype(np.float64)
    Fneu = np.load(os.path.join(plane, 'Fneu.npy')).astype(np.float64)
    ops = np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item()
    iscell = np.load(os.path.join(plane, 'iscell.npy'))
    ...

def load_motion_energy(session_dir, nframes):
    move_dir = os.path.join(session_dir, 'move_deve')
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
    tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The agent explored the data directory structure, found 6 mice with 6-7 sessions each, and identified the suite2p output files and motion energy files. It also read the data README and the load_data notebook to confirm the loading approach.

## 1-b. How are the data split into subjects?

i. Subjects correspond to all sorted directories in the data directory. Unlike the reference, no `jm` prefix filter is applied, but all directories happen to start with `jm`, so the result is equivalent.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))
```

iii. The agent enumerated the data directory and found all subject folders (jm031 through jm046). Since all directories in the data folder are subject folders, no explicit prefix filter was needed.

## 1-c. How are the data split into sessions?

i. Sessions are sorted subdirectories within each subject's folder. Each subdirectory contains one daily recording.

ii.
```python
sessions = sorted(d for d in os.listdir(subject_dir)
                  if os.path.isdir(os.path.join(subject_dir, d)))
```

iii. The agent observed that each subject directory contains date-named subdirectories, each representing a daily recording session. Sorting ensures deterministic order.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. Trials are defined as consecutive non-overlapping 60-second blocks. After binning (10 frames), each trial is 180 bins. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))
ntrials = nbins // bins_per_trial

for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    sess_neural.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
```

iii. The task instructions specify "split sessions into 60-second trials." Since there is no stimulus-driven trial structure (spontaneous behavior), the agent tiles the session into consecutive 60s blocks. Sessions of 20 min yield 20 trials and 30 min yield 30 trials, both dividing evenly.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 60-second trials are kept.

ii. N/A

iii. There are no quality-control criteria for excluding trials in this spontaneous-behavior dataset. All trials that fill a complete 60-second block are included.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. The `ops.npy` file is also loaded to get the sampling rate (`fs`).

ii.
```python
F = np.load(os.path.join(plane, 'F.npy')).astype(np.float64)
Fneu = np.load(os.path.join(plane, 'Fneu.npy')).astype(np.float64)
ops = np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item()
```

iii. These are the standard suite2p output files. The agent also read the load_data notebook which showed loading these files.

## 2-b. How is the `neural` data processed?

i. The agent applies baseline correction with `neucoeff=0.0` (no neuropil subtraction), followed by maximin baseline correction using scipy operations: Gaussian smoothing (sigma=10), minimum filter (window=60s*30Hz=1800), maximum filter (same window), then subtraction. This follows the track2p `F_processing` function rather than calling suite2p's `dcnv.preprocess` directly.

ii.
```python
def f_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin',
                 sig_baseline=10.0, win_baseline=60.0, prctile_baseline=8.0):
    Fc = F - neucoeff * Fneu  # neucoeff=0.0, so Fc = F
    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    return Fc - Flow
```

iii. The agent found the track2p source code (`track2p/gui/data_management.py::F_processing`) which uses `neucoeff=0.0` by default, and decided to follow that implementation rather than suite2p's ops.npy value of 0.7. The agent reasoned that the paper says to use dF/F "as computed by track2p" and the track2p code explicitly sets neucoeff=0.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. An assertion verifies all neurons have `iscell=1`, confirming the Track2p output only contains cells that passed Suite2p's classifier and were matched across days.

ii.
```python
assert np.all(iscell[:, 0] == 1), f'unexpected non-cell ROI in {session_dir}'
```

iii. The data README states the distributed data only contains Track2p-tracked neurons that passed the Suite2p classifier (threshold 0.5) on every day. The assertion makes this assumption explicit.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed. The metadata describes alignment as "start of each 60 s block."

ii.
```python
'temporal_alignment_event': (
    'start of each 60 s block of the continuous recording; blocks tile the '
    'session from imaging onset (frame 0 of the 2-photon acquisition, which '
    'also triggers the behaviour camera)'),
'off_start': 0.0,
'off_end': TRIAL_SEC,
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, converting from 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before motion energy discretization.

ii.
```python
BIN_FRAMES = 10
dff_b = bin_average(dff, BIN_FRAMES)
me_b = bin_average(me, BIN_FRAMES)

'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,  # 333.33 ms
```

iii. The paper's methods state "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The agent followed this exactly.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index, the number of frames per bin, and the sampling rate.

ii.
```python
t = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
```

iii. The frame rate is constant at 30 Hz and there are no stored timestamps for neural data, so computing time from bin indices is equivalent to using actual timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the center of each time bin in seconds from session start. For bin `i`, time = `(i * 10 + 4.5) / 30.0`. Time is continuous across trials (not reset per trial), so it ranges from ~0.15s to ~1799.85s for a 30-minute session.

ii.
```python
t = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
# ...
sess_input.append(np.ascontiguousarray(t[None, sl], dtype=np.float32))
```

iii. The agent chose bin centers rather than bin left edges, and kept time continuous across the entire session (not resetting per trial) since the spec says "time elapsed from the beginning of the session."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same bin indices used for the neural data, so it is inherently aligned. Each time value corresponds to the center of the same bin as the corresponding neural data column.

ii. N/A (alignment is implicit from using the same indexing)

iii. Since both time and neural data are indexed by the same bin index, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Camera timestamps from `tstamps.npy` are used to detect and interpolate dropped frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal. The timestamps file is needed to identify dropped camera frames for interpolation.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) Dropped camera frames are detected from gaps in timestamps (where inter-frame interval is approximately k times the median interval, indicating k-1 dropped frames), mapped onto a full-length grid, and linearly interpolated. The result is trimmed or padded to match the neural frame count. (2) Binned by averaging 10 consecutive frames. (3) Discretized into 5 equal-percentile bins per session using `np.digitize`.

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
# ...
me_b = bin_average(me, BIN_FRAMES)
me_cat = discretize_percentile(me_b, N_OUTPUT_BINS)
```

iii. The data README suggests using timestamps to identify dropped frames. The agent used `np.interp` for linear interpolation, which is more vectorized than inserting values one at a time.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins per session using percentile edges at 20%, 40%, 60%, 80%. Values are assigned to categories 0-4 via `np.digitize`.

ii.
```python
def discretize_percentile(x, nbins):
    edges = np.percentile(x, np.linspace(0, 100, nbins + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int64)
```

iii. The task instructions specify "five equal-percentile bins, selected per session." The agent verified that each quintile contains exactly 20% of the data.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously (camera triggered by 2-photon acquisition), so they are aligned frame-for-frame. Dropped frames are interpolated to restore the one-to-one mapping. After interpolation, the motion energy is trimmed or padded to match the neural frame count exactly.

ii.
```python
me = load_motion_energy(session_dir, nframes)
# In load_motion_energy:
if len(full) > nframes:
    full = full[:nframes]
elif len(full) < nframes:
    full = np.concatenate([full, np.full(nframes - len(full), full[-1])])
```

iii. The agent verified via cross-correlation that the neural-motion alignment is correct (peak at +1-2 bins lag, consistent with calcium indicator kinetics). The microscope triggers the camera, so they are inherently synchronized.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are detected from timestamp gaps and linearly interpolated. If the interpolated motion energy is longer than neural data, it is trimmed; if shorter, it is padded with the last value. Remainder bins that don't fill a complete 60s trial are discarded.

ii.
```python
if len(full) > nframes:
    full = full[:nframes]
elif len(full) < nframes:
    full = np.concatenate([full, np.full(nframes - len(full), full[-1])])
```

iii. The agent counted 276 total dropped frames across all sessions. The trim/pad logic handles edge cases where the interpolated length doesn't exactly match the neural frame count.

## 6-a. What are the most time-consuming steps of the code?

i. The maximin baseline correction is the most computationally expensive step, involving Gaussian smoothing, minimum filtering, and maximum filtering over the full session length for every neuron. The scipy-based implementation runs on CPU.

ii.
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. These are sliding-window operations applied to every neuron's full trace. Unlike the reference solution which uses suite2p's `dcnv.preprocess` (which can leverage GPU via PyTorch), the AI's scipy implementation is CPU-only.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials to slice arrays, but this is already efficient since it's just creating views/copies. The dropped-frame interpolation uses `np.interp` which is already vectorized.

ii. N/A

iii. The AI's code is generally well-vectorized. The dropped-frame handling uses `np.interp` rather than an iterative insertion loop, which is more efficient than the reference approach.

## 6-c. What processing does the code repeat multiple times?

i. The percentile edges for motion energy discretization are computed twice: once in `discretize_percentile` for the actual discretization, and once in the session_info metadata. However, these serve different purposes (one for the data, one for metadata logging).

ii.
```python
me_cat = discretize_percentile(me_b, N_OUTPUT_BINS)
# ...
'motion_energy_bin_edges': np.percentile(
    me_b, np.linspace(0, 100, N_OUTPUT_BINS + 1)[1:-1]).tolist(),
```

iii. This duplication is minor and serves documentation purposes.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `ops.npy` and `iscell.npy` for each session. While `ops` is used to get `fs` and `nframes`, `iscell` is only used for an assertion check and not for actual filtering. Additionally, detailed `session_info` metadata is computed and stored but not used by the decoder.

ii.
```python
iscell = np.load(os.path.join(plane, 'iscell.npy'))
assert np.all(iscell[:, 0] == 1), ...
```

iii. Loading iscell for verification is good practice but technically unnecessary since all values are 1. The session_info metadata is useful for documentation but not consumed by the decoder.
