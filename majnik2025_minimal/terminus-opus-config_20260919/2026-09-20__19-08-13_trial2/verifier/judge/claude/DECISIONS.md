# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as directories matching `jm*` in the data directory. Sessions are subdirectories within each subject folder. For each session, calcium data is loaded from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`, and session metadata from `ops.npy`. Motion energy is loaded from `move_deve/motion_energy_glob.npy` and interframe intervals from `move_deve/interframe_int.npy`.

ii.
```python
subjects = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, 'jm*')))
# ...
for si, sub in enumerate(subjects):
    sub_dir = os.path.join(DATA_DIR, sub)
    sess_dirs = sorted([f.path for f in os.scandir(sub_dir) if f.is_dir()])
    # ...
    for sd in sess_dirs:
        s2p = os.path.join(sd, 'suite2p', 'plane0')
        F = np.load(os.path.join(s2p, 'F.npy'))[keep]
        Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))[keep]
        ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
        # ...
        me = np.load(os.path.join(mv, 'motion_energy_glob.npy'))
        ifi = np.load(os.path.join(mv, 'interframe_int.npy'))
```

iii. The AI explored the data directory structure, read the data README and load_data.ipynb notebook, then inspected array shapes for all sessions to understand the data layout before writing the conversion code.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories matching the `jm*` glob pattern in the data directory, sorted alphabetically. This yields 6 subjects: jm031, jm032, jm038, jm039, jm040, jm046.

ii.
```python
subjects = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, 'jm*')))
```

iii. The AI identified the naming convention from the data README, which states subject folders are named with subject IDs (e.g., 'jm031/').

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject's folder, sorted alphabetically (which gives chronological order since folders are named with dates in YYYY-MM-DD format).

ii.
```python
sess_dirs = sorted([f.path for f in os.scandir(sub_dir) if f.is_dir()])
```

iii. The AI read the data README which explains each session folder corresponds to one recording day.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. Trials are defined as 60-second non-overlapping segments of the continuous recording. After binning (10 frames at 30 Hz = 333.33 ms bins), each trial is 180 bins. 20-minute sessions yield 20 trials, 30-minute sessions yield 30 trials. Any remainder bins that don't fill a complete trial are discarded.

ii.
```python
bins_per_trial = int(round(TRIAL_SEC / bin_sec))  # 180
ntrials = nbins // bins_per_trial
# ...
for t in range(ntrials):
    sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
    neural_sess.append(dff_b[:, sl].astype(np.float32))
```

iii. The instructions specify "Split sessions into 60-second trials." The AI followed this directly.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. All complete 60-second segments are retained.

ii. N/A

iii. The AI did not mention trial filtering in its trajectory. There is no instruction or paper guidance to filter trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. The frame rate `fs` and `nframes` are read from `ops.npy`.

ii.
```python
F = np.load(os.path.join(s2p, 'F.npy'))[keep]
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))[keep]
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
fs = float(ops['fs'])
nframes = int(ops['nframes'])
```

iii. The AI identified these as the standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI implements the track2p GUI's `F_processing` function with `neucoeff=0.0` (no neuropil subtraction), `maximin` baseline correction with `sig_baseline=10.0` and `win_baseline=60.0`. This computes `Fc = F - 0.0*Fneu`, then subtracts a maximin baseline (Gaussian smooth -> min filter -> max filter). After baseline correction, data is averaged in bins of 10 frames, then z-scored per neuron within each session.

ii.
```python
def f_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin',
                 sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    else:
        Flow = 0.
    return Fc - Flow
# ...
dff = f_processing(F.astype(np.float64), Fneu.astype(np.float64), fs)
dff_b = bin_time(dff, BIN_FRAMES)
dff_b = (dff_b - dff_b.mean(axis=1, keepdims=True)) / dff_b.std(axis=1, keepdims=True)
```

iii. The AI found the `F_processing` function in `track2p/gui/data_management.py` which uses `neucoeff=0.0` by default. The AI explicitly copied this implementation. The paper states "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." The AI also z-scored based on the track2p raster preprocessing code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI performs neuron curation: ROIs whose trace is identically zero or has zero standard deviation on any day of a mouse are dropped from ALL sessions of that mouse. This mirrors the track2p GUI's "remove zero rows" behavior.

ii.
```python
bad = None
for sd in sess_dirs:
    F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'))
    z = np.all(F == 0, axis=1) | (F.std(axis=1) == 0)
    bad = z if bad is None else (bad | z)
keep = ~bad
```

iii. The AI investigated zero-variance neurons across all sessions and found a small number of all-zero ROIs. It modeled the curation after track2p's GUI preprocessing which removes such zero rows across days.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed. The metadata specifies `temporal_alignment_event` as "start of each 60 s block" with `off_start=0.0` and `off_end=60.0`.

ii.
```python
'temporal_alignment_event': (
    'start of each 60 s block of the continuous recording (blocks are cut '
    'consecutively from the session onset; there is no trial structure in this '
    'spontaneous-activity experiment)'),
'off_start': 0.0,
'off_end': 60.0,
```

iii. There is no stimulus event to align to. The recording is continuous spontaneous activity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing 30 Hz to 3 Hz (333.33 ms bins). This binning is applied before the motion energy is discretized.

ii.
```python
BIN_FRAMES = 10
# ...
dff_b = bin_time(dff, BIN_FRAMES)
me_b = bin_time(me, BIN_FRAMES)
# ...
'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,  # 333.33 ms
```

iii. The AI followed the paper's Methods: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index, bin size, and frame rate. The time vector uses bin centres: `(bin_index + 0.5) * bin_duration_sec`.

ii.
```python
bin_sec = BIN_FRAMES / fs
tvec = (np.arange(nbins) + 0.5) * bin_sec
# ...
input_sess.append(tvec[sl][None, :].astype(np.float32))
```

iii. Since the frame rate is constant at 30 Hz and there are no external timestamps for neural data, computing time from bin indices is straightforward.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `(bin_index + 0.5) * (BIN_FRAMES / fs)` giving the centre of each time bin in seconds from session start. This is a simple arithmetic operation with no filtering or transformation.

ii.
```python
tvec = (np.arange(nbins) + 0.5) * bin_sec
```

iii. The +0.5 offset places the time at the centre of each bin rather than the left edge.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector is constructed from the same bin indices as the neural data, so alignment is inherent. Both use the same number of bins per trial (`bins_per_trial = 180`) and are sliced with the same slice indices.

ii.
```python
for t in range(ntrials):
    sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
    neural_sess.append(dff_b[:, sl].astype(np.float32))
    input_sess.append(tvec[sl][None, :].astype(np.float32))
```

iii. N/A - alignment is trivial since time is derived from the bin indices.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session. Interframe intervals from `interframe_int.npy` are used to detect and interpolate dropped camera frames.

ii.
```python
me = np.load(os.path.join(mv, 'motion_energy_glob.npy'))
ifi = np.load(os.path.join(mv, 'interframe_int.npy'))
me = align_motion_energy(me, ifi, nframes)
```

iii. The data README describes motion_energy_glob.npy as processed behavioral data and suggests using interframe_int.npy to handle missing frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) dropped camera frames are detected via interframe intervals (multiples of median interval) and interpolated by linear interpolation of neighboring values, (2) the trace is averaged into 10-frame bins along with the neural data, (3) the binned signal is discretized into 5 equal-percentile bins computed per session using `np.digitize`.

ii.
```python
# Step 1: align/interpolate dropped frames
me = align_motion_energy(me, ifi, nframes)
# Step 2: bin
me_b = bin_time(me, BIN_FRAMES)
# Step 3: discretize
edges = np.percentile(me_b, np.linspace(0, 100, N_QUANTILES + 1)[1:-1])
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. The AI followed the data README suggestion for handling missing frames and the paper's description of binning. The instructions specify "discretized into five equal-percentile bins, selected per session."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins (quintiles) per session. The bin edges are computed using `np.percentile` at [20, 40, 60, 80] percentiles, then `np.digitize` assigns each value to a bin (0 through 4).

ii.
```python
edges = np.percentile(me_b, np.linspace(0, 100, N_QUANTILES + 1)[1:-1])
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. The instructions specify "five equal-percentile bins, selected per session." This produces exactly 20% of data in each bin per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The camera was triggered by the microscope, so camera frame i corresponds to imaging frame i. When camera frames are dropped (motion energy array shorter than neural frames), the AI detects drops via interframe intervals that are multiples of the median, inserts NaN values at those positions (limited to exactly the number of missing frames), and fills NaNs by linear interpolation. After alignment, both are binned together, ensuring matching time indices.

ii.
```python
def align_motion_energy(me, ifi, nframes):
    me = me.astype(np.float64)
    gap = int(nframes - len(me))
    if gap <= 0:
        return me[:nframes]
    med = np.median(ifi)
    nmiss = np.maximum(np.round(ifi / med).astype(int) - 1, 0)
    if nmiss.sum() > gap:
        order = np.argsort(-(ifi / med))
        keepmask = np.zeros(len(ifi), dtype=int)
        remaining = gap
        for i in order:
            if remaining <= 0: break
            take = min(nmiss[i], remaining)
            keepmask[i] = take
            remaining -= take
        nmiss = keepmask
    out = [me[0]]
    for i in range(len(ifi)):
        for _ in range(int(nmiss[i])):
            out.append(np.nan)
        out.append(me[i + 1])
    arr = np.array(out, dtype=np.float64)
    nanmask = np.isnan(arr)
    if nanmask.any():
        idx = np.arange(len(arr))
        arr[nanmask] = np.interp(idx[nanmask], idx[~nanmask], arr[~nanmask])
    # ...
```

iii. The AI investigated dropped frames across all sessions, found that the median-ratio method correctly estimates the number of missing frames matching the length gap, and refined the function to insert exactly the right number of frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. (1) Dropped camera frames: detected and filled by linear interpolation (see 4-d). If the motion energy is longer than nframes, it is truncated; if shorter after interpolation, the last value is repeated. (2) All-zero ROIs: neurons with zero traces on any day are dropped from all sessions of that mouse. (3) Remainder frames at the end of a session that don't fill a complete 60s trial are discarded.

ii.
```python
# truncate or pad
if len(arr) > nframes:
    arr = arr[:nframes]
elif len(arr) < nframes:
    arr = np.concatenate([arr, np.full(nframes - len(arr), arr[-1])])
# ...
# discard remainder
ntrials = nbins // bins_per_trial
```

iii. The AI tested missing frame counts across all sessions and verified consistency between interframe interval estimates and actual length mismatches.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the `f_processing` baseline correction, which applies Gaussian filtering, minimum filtering, and maximum filtering over the full session length for every neuron. Loading the `.npy` files is also I/O bound but relatively fast.

ii. N/A

iii. The baseline correction involves sliding window operations (Gaussian filter + min/max filters with window = 1800 frames) over potentially 54000 frames for each of hundreds of neurons.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame interpolation loop in `align_motion_energy` inserts frames one at a time by appending to a Python list, which could be pre-allocated. However, the number of dropped frames is typically very small (0-148), so the performance impact is negligible.

ii. N/A

iii. N/A

## 6-c. What processing does the code repeat multiple times?

i. The F.npy file for each session is loaded twice: once in the first pass to identify all-zero ROIs across days, and again in the second pass for actual processing. This is by design (two-pass approach) but could be cached.

ii.
```python
# First pass
for sd in sess_dirs:
    F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'))
    z = np.all(F == 0, axis=1) | (F.std(axis=1) == 0)
# ...
# Second pass
for sd in sess_dirs:
    F = np.load(os.path.join(s2p, 'F.npy'))[keep]
```

iii. The two-pass design is intentional to identify bad neurons before processing any session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The z-scoring normalization applied to neural data is not explicitly required by the instructions or paper's decoding methods. It may or may not help the decoder. Additionally, computing bin centres (+0.5 offset) rather than left edges is a minor unnecessary precision that doesn't affect downstream analysis.

ii. N/A

iii. N/A
