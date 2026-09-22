# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script enumerates every directory directly under `/app/data` as a subject and every directory under each subject as a session, both sorted. Per session it loads `ops.npy`, `F.npy`, `Fneu.npy`, `iscell.npy`, `motion_energy_glob.npy`, and `interframe_int.npy`. It then constructs trials after preprocessing.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
for si, subject in enumerate(subjects):
    for session_dir in list_sessions(os.path.join(DATA_DIR, subject)):
        ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
        F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float64)
        Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float64)
        iscell = np.load(os.path.join(s2p, 'iscell.npy'))
```

iii. The trajectory reports a survey of all 41 sessions across six mice and identifies the subject/session directory hierarchy and these Suite2p and behavior files. Sorting makes the result deterministic, while the directory check excludes `/app/data/ground_truth.csv`.

## 1-b. How are the data split into subjects?

i. Each top-level directory in `/app/data` is treated as one mouse. Subjects are alphabetically sorted, and the loop index is stored once per session in `subject_idx`.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
for si, subject in enumerate(subjects):
    ...
    subject_idx.append(si)
```

iii. The agent inspected the dataset and concluded that it contains six mouse directories. It used the filesystem hierarchy rather than parsing session names.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory of a subject directory is a session/daily recording and becomes one entry in each session-level output list.

ii.
```python
def list_sessions(subject_dir):
    return sorted([f.path for f in os.scandir(subject_dir) if f.is_dir()])

for session_dir in list_sessions(os.path.join(DATA_DIR, subject)):
    ...
    neural_all.append(neural_s)
```

iii. The trajectory identifies the session folders as separate recordings and reports 41 sessions in total. Sorting preserves a reproducible order.

## 1-d. How are the data split into trials?

i. Each continuous session is divided into consecutive, non-overlapping 60-second blocks after 10-frame temporal averaging. At the nominal 30 Hz rate this gives 180 bins per trial. Any final incomplete block is omitted by integer division.

ii.
```python
bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))
ntrials = nbins // bins_per_trial
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural_s.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
```

iii. The requested decoder task explicitly requires 60-second trials, while the experiment is continuous and has no natural trials. The agent therefore selected contiguous blocks and confirmed 20 or 30 trials per session.

## 1-e. How are trials filtered based on quality controls?

i. No complete 60-second trials are quality-filtered. Only the trailing bins that cannot form a full trial are discarded.

ii.
```python
ntrials = nbins // bins_per_trial
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
```

iii. The trajectory found no trial-level quality criterion in the paper or repository. Fixed-length complete blocks satisfy the decoder requirement, so no further trial rejection was introduced.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values are derived from Suite2p `F.npy` and `Fneu.npy`; `ops.npy` supplies processing parameters and `iscell.npy` supplies the ROI inclusion mask.

ii.
```python
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float64)
Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float64)
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
```

iii. The agent traced the repository's `F_processing` implementation and inspected the actual Suite2p arrays and `ops` parameters before choosing these sources.

## 2-b. How is the `neural` data processed?

i. The code applies neuropil subtraction using `ops['neucoeff']` (default 0.7), then Suite2p-style maximin baseline subtraction: Gaussian smoothing, a temporal minimum filter, and a temporal maximum filter. The resulting baseline-corrected fluorescence is averaged over non-overlapping groups of 10 frames and cast to contiguous `float32` trial arrays.

ii.
```python
Fc = F - neucoeff * Fneu
win = int(win_baseline * fs)
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
return Fc - Flow
...
dff_b = bin_average(dff, BIN_FRAMES)
```

iii. The trajectory says this reproduces Track2p's `F_processing`/Suite2p maximin baseline correction using parameters stored in `ops`, followed by the paper's decoding-analysis denoising of 10-frame averages.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are retained only when the first column of `iscell.npy` is positive. The inspected data had all provided tracked cells passing this mask, so it does not change this dataset.

ii.
```python
keep = iscell[:, 0] > 0
F, Fneu = F[keep], Fneu[keep]
```

iii. The agent determined that the provided files contain Track2p-tracked neurons and that all had `iscell == 1`, reflecting prior classifier curation. It nevertheless encoded the explicit mask.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Trial zero starts at the first imaging frame, and subsequent neural trials are consecutive 60-second slices. Metadata defines the alignment event as the start of each 60-second block, with offsets 0 to 60 seconds.

ii.
```python
sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
neural_s.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
...
'off_start': 0.0,
'off_end': TRIAL_SEC,
```

iii. The trajectory recognizes the recordings as continuous spontaneous activity, so the only meaningful trial alignment is the artificial block boundary required by the task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30 Hz frames are averaged into each bin, yielding 3 Hz data and nominal 333.33 ms bins. Incomplete 10-frame tails are dropped.

ii.
```python
BIN_FRAMES = 10
dff_b = bin_average(dff, BIN_FRAMES)
...
'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,
```

iii. The agent cites the paper's decoding analysis, which denoised both fluorescence and behavior by averaging 10 consecutive timestamps, and applied identical binning to maintain alignment.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is generated from the binned frame index, `BIN_FRAMES`, and the session's `ops['fs']`; it is not loaded from a timestamp file.

ii.
```python
fs = float(ops['fs'])
t_b = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
```

iii. The trajectory established that imaging is sampled at a stable 30 Hz and chose the imaging frame grid as the common clock.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each averaged bin, the code computes the time in seconds at the mean sample location: the first frame index plus `(10-1)/2 = 4.5` frames, divided by the session sampling rate. Time remains continuous across the session and is cast to `float32` after slicing.

ii.
```python
t_b = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
input_s.append(t_b[sl][None, :].astype(np.float32))
```

iii. The code comment explicitly identifies these as bin-center times from session start. This is consistent with representing each averaged value at the average time of its ten samples.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector and binned neural matrix share the same `nbins` grid and are sliced with the identical trial slice, producing shape `(1, time)` inputs aligned one-for-one to neural columns.

ii.
```python
nbins = dff_b.shape[1]
t_b = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
...
neural_s.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
input_s.append(t_b[sl][None, :].astype(np.float32))
```

iii. The trajectory planned a single binned imaging time base for all streams and validated the resulting dimensions with the decoder format checker.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses precomputed global motion energy from `move_deve/motion_energy_glob.npy`; `move_deve/interframe_int.npy` provides camera interval information for reconstructing dropped-frame positions.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(session_dir, 'move_deve', 'interframe_int.npy')).astype(np.float64)
```

iii. The agent found these signals in the dataset and identified motion energy as squared pixel-wise frame differences and the interval data as the reliable way to locate missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The artificial first zero is replaced with the second sample. If camera frames are missing, interval ratios reconstruct acquired samples on the full imaging grid and `np.interp` linearly fills gaps. Motion energy is then averaged over 10-frame bins and categorized with per-session percentile thresholds.

ii.
```python
me[0] = me[1]
med = np.median(ifi)
steps = np.round(ifi / med).astype(int)
idx = np.concatenate([[0], np.cumsum(steps)])
return np.interp(np.arange(nframes), idx, me)
...
me_b = bin_average(me, BIN_FRAMES)
edges = np.percentile(me_b, np.linspace(0, 100, N_ME_BINS + 1)[1:-1])
```

iii. The trajectory's data survey showed that interval gaps are integer multiples of the nominal period and exactly account for missing samples. Ten-frame averaging follows the paper, and categorization is required by the decoder task.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four thresholds—the 20th, 40th, 60th, and 80th percentiles of the complete binned trace—are computed separately for each session. `np.digitize` maps samples to integer classes 0 through 4.

ii.
```python
edges = np.percentile(me_b, np.linspace(0, 100, N_ME_BINS + 1)[1:-1])
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. This directly implements the instruction for five equal-percentile bins selected per session. The agent's validation found the expected balanced quintiles.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera sample locations are reconstructed on the neural frame grid from interframe intervals and linearly interpolated to exactly `nframes`. Neural and motion energy are then independently averaged using the same 10-frame boundaries and sliced with the same trial slice.

ii.
```python
assert idx[-1] == nframes - 1, (session_dir, idx[-1], nframes)
return np.interp(np.arange(nframes), idx, me)
...
dff_b = bin_average(dff, BIN_FRAMES)
me_b = bin_average(me, BIN_FRAMES)
...
output_s.append(me_cat[sl][None, :].astype(np.int64))
```

iii. The agent reasoned that the camera was microscope-triggered and therefore corresponds one-to-one with imaging frames except for drops. It also performed a final correlation sanity check between population activity and motion energy.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera samples are reconstructed from interval multiples and linearly interpolated. Assertions check the recovered grid length/end point. The invalid initial motion-energy zero is replaced, and incomplete temporal bins and incomplete 60-second trailing trials are dropped.

ii.
```python
me[0] = me[1]
steps = np.round(ifi / med).astype(int)
idx = np.concatenate([[0], np.cumsum(steps)])
assert len(idx) == len(me)
assert idx[-1] == nframes - 1, (session_dir, idx[-1], nframes)
return np.interp(np.arange(nframes), idx, me)
```

iii. The trajectory explicitly investigated sessions with missing camera frames and found that interval multiples recover their locations exactly. Assertions prevent silent misalignment; dropping partial tails ensures uniform bins and trials.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant computation is full-session baseline correction over every neuron (Gaussian and long-window minimum/maximum filters). Loading all large arrays and interpolation/binning add lesser I/O and array-processing costs.

ii.
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. The agent did not benchmark individual steps, but its investigation and implementation identify full-trace Suite2p baseline processing as the heaviest operation. The reference decision likewise identifies baseline correction as dominant.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The subject/session loops are inherently file-oriented. The trial loop could be replaced by reshaping each complete session into a trial axis, though it currently also creates separate typed contiguous arrays. Heavy filtering, interpolation, bin averaging, and discretization are already vectorized.

ii.
```python
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural_s.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
    input_s.append(t_b[sl][None, :].astype(np.float32))
    output_s.append(me_cat[sl][None, :].astype(np.int64))
```

iii. The trajectory did not discuss loop optimization. The implementation already improves on repeated insertion by using one vectorized `np.interp`; only the small assembly loop remains an obvious vectorization candidate.

## 6-c. What processing does the code repeat multiple times?

i. Loading, ROI masking, baseline correction, missing-frame handling, binning, percentile computation, and trial assembly repeat once for every session. Within a session, the same `bin_average` operation is applied separately to neural and motion-energy arrays.

ii.
```python
for si, subject in enumerate(subjects):
    for session_dir in list_sessions(os.path.join(DATA_DIR, subject)):
        ...
        dff_b = bin_average(dff, BIN_FRAMES)
        me_b = bin_average(me, BIN_FRAMES)
```

iii. This repetition is intentional because parameters, neuron counts, missing frames, percentile edges, and outputs are session-specific. The trajectory explicitly chose per-session quintiles and processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `bin_size_s` is computed but never used. The code also loads and applies `iscell` even though the inspected provided ROIs all pass, and builds extensive `session_info` metadata that is not needed for decoder training. It reads `interframe_int.npy` even for sessions whose motion-energy length already matches, although this is useful for a uniform loader.

ii.
```python
bin_size_s = BIN_FRAMES / fs
...
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
keep = iscell[:, 0] > 0
...
session_info.append({...})
```

iii. The trajectory did not identify unnecessary work. These items are inferred from data-flow inspection: `bin_size_s` has no consumer, while the mask and metadata are defensible validation/provenance work but do not affect downstream decoding on this dataset.
