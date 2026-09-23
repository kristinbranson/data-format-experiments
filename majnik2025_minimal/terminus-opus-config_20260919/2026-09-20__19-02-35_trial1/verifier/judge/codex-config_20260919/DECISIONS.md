# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every directory under `/app/data` as a subject, then every directory beneath each subject as a session. For each session it loads Suite2p `ops.npy`, `F.npy`, `Fneu.npy`, and `iscell.npy`, plus `motion_energy_glob.npy` and `tstamps.npy`. It processes each session and appends it to the output; the resulting file contains 41 sessions from 6 mice.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
...
sessions = sorted([d for d in os.listdir(subj_dir)
                   if os.path.isdir(os.path.join(subj_dir, d))])
...
ops = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'ops.npy'),
              allow_pickle=True).item()
F = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'Fneu.npy'))
iscell = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'iscell.npy'))
```

iii. The trajectory says the agent surveyed the directory layout and found 6 mice and 41 sessions. It chose these files after inspecting the paper methods, dataset README, loader notebook, Track2p code, and actual array shapes.

## 1-b. How are the data split into subjects?

i. Each top-level directory under the data directory is treated as one mouse; directory names are sorted and stored in `subjects`. Each session receives the corresponding integer `subject_idx`.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
...
data['subject_idx'].append(si)
```

iii. The agent observed that the data directories were the mice `jm031` through `jm046` and explicitly skipped non-directory entries such as `ground_truth.csv`.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory of a mouse directory is one session/daily recording. One nested list is appended to `neural`, `input`, and `output` for each such directory.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                   if os.path.isdir(os.path.join(subj_dir, d))])
for sess in sessions:
    sdir = os.path.join(subj_dir, sess)
...
data['neural'].append(neural_trials)
data['input'].append(input_trials)
data['output'].append(output_trials)
```

iii. The agent determined from the layout that each mouse has 6–7 daily session directories and retained each as a distinct decoder session.

## 1-d. How are the data split into trials?

i. Each continuous session is divided into consecutive, non-overlapping 60-second trials. With 10-frame bins at 30 Hz, each trial has 180 bins. A trailing partial trial is omitted by integer division.

ii.
```python
bin_size = BIN_FRAMES / fs
bins_per_trial = int(round(TRIAL_SEC / bin_size))
ntrials = T // bins_per_trial
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
```

iii. The trajectory states this implements the explicit instruction to split sessions into 60-second trials; it verified that sessions yielded 20 or 30 complete trials.

## 1-e. How are trials filtered based on quality controls?

i. No complete 60-second trial is quality-filtered. Only an incomplete tail at the end of a session is discarded. Sessions are not excluded.

ii.
```python
ntrials = T // bins_per_trial
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
```

iii. The trajectory reports no trial-level QC criterion in the paper or data. The fixed-length decoder requirement motivates retaining all complete segments.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from Suite2p `F.npy`, with `Fneu.npy` loaded but multiplied by a chosen neuropil coefficient of zero. `ops.npy` supplies sampling and baseline parameters, and `iscell.npy` supplies the ROI inclusion mask.

ii.
```python
F = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'Fneu.npy'))
iscell = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'iscell.npy'))
keep = iscell[:, 0] > 0.5
F, Fneu = F[keep], Fneu[keep]
```

iii. The agent inspected Track2p's `F_processing` and concluded it was called with its default `neucoeff=0`. It also surveyed `iscell` and found all provided ROIs marked as cells.

## 2-b. How is the `neural` data processed?

i. After applying a zero neuropil coefficient, the agent reproduces a maximin baseline: Gaussian smoothing, a 60-second minimum filter, then a maximum filter. It returns `F - Flow` (described by the agent as baseline-corrected fluorescence/dF/F), then averages non-overlapping groups of 10 frames and casts to `float32`. There is no division by the baseline.

ii.
```python
neucoeff = 0.0
Fc = F - neucoeff * Fneu
win = int(win_baseline * fs)
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
return Fc - Flow
...
neural = bin_average(dff, BIN_FRAMES).astype(np.float32)
```

iii. The agent justified coefficient zero from Track2p's own call path, and tested a 0.7-neuropil-subtracted variant (validation balanced accuracy 0.299 versus 0.306). It retained zero as more faithful to the repository, while using the paper's 10-timestamp denoising.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are retained when the first `iscell.npy` column is greater than 0.5. No later neuron-level filtering is performed.

ii.
```python
iscell = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'iscell.npy'))
keep = iscell[:, 0] > 0.5
F, Fneu = F[keep], Fneu[keep]
```

iii. The agent described this as the criterion used in the paper, while noting that its survey found `iscell == 1` for every provided Track2p-curated ROI, so the mask does not change this dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Neural bins are indexed from the first imaging frame and trials are consecutive slices relative to session start. Metadata names the alignment event as session start and describes each trial as spanning 0–60 seconds relative to its boundary.

ii.
```python
sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
...
'temporal_alignment_event': 'start of the imaging session (first 2-photon frame)',
'off_start': 0.0,
'off_end': 60.0,
```

iii. The agent recognized that these are artificial segments of spontaneous continuous recordings, so session/trial boundaries—not stimulus events—provide the only alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30 Hz samples are averaged in non-overlapping bins, producing 3 Hz data and a nominal bin size of 333.33 ms. The operation drops a final group shorter than ten frames.

ii.
```python
BIN_FRAMES = 10
...
n = (x.shape[-1] // binsize) * binsize
return x[..., :n].reshape(newshape).mean(axis=-1)
...
'time_bin_size': 1000.0 * BIN_FRAMES / 30.0
```

iii. The trajectory quotes the methods' instruction that all decoding analyses denoised dF/F and behavior by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the binned sample index and `ops['fs']`; it is not read from camera timestamps. Values denote bin centers and continue across trial boundaries from session start.

ii.
```python
fs = float(ops['fs'])
bin_size = BIN_FRAMES / fs
tvec = (np.arange(T) + 0.5) * bin_size
```

iii. The agent established that imaging runs at 30 Hz and treated the regularly sampled imaging clock as the appropriate clock for neural-aligned elapsed time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code converts bin numbers to seconds by adding half a bin and multiplying by `10/fs`; trial slices are reshaped to `(1, time)` and cast to `float32`.

ii.
```python
tvec = (np.arange(T) + 0.5) * bin_size
input_trials.append(tvec[sl][None, :].astype(np.float32))
```

iii. The half-bin offset represents the time at the center of each averaged 10-frame bin. The trajectory does not give a separate explicit justification for center rather than left-edge timestamps.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. A single `tvec` entry is created for every binned neural column, and identical trial slices are applied to both arrays.

ii.
```python
T = min(neural.shape[1], me_b.size)
neural, me_b = neural[:, :T], me_b[:T]
tvec = (np.arange(T) + 0.5) * bin_size
...
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
input_trials.append(tvec[sl][None, :].astype(np.float32))
```

iii. The shared binned index and shared slice were chosen to guarantee one time value per neural sample.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It is derived from the precomputed global motion-energy trace `motion_energy_glob.npy`. Camera timestamps in `tstamps.npy` are used to locate dropped camera frames.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve',
                          'motion_energy_glob.npy')).astype(np.float64)
ts = np.load(os.path.join(session_dir, 'move_deve',
                          'tstamps.npy')).astype(np.float64)
```

iii. The agent identified motion energy as the paper's videography-based movement proxy and used timestamps after the README and data survey showed occasional missing frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code reconstructs a neural-frame-length trace, inserts missing locations as NaNs based on rounded timestamp gaps, marks the intrinsically invalid first motion-energy sample as NaN, linearly interpolates missing values, averages 10-frame bins, trims it and neural data to a shared length, then categorizes it using session-local quintiles.

ii.
```python
d = np.diff(ts)
med = np.median(d)
steps = np.round(d / med).astype(int)
idx = np.concatenate([[0], np.cumsum(steps)])
full[idx[keep]] = me[keep]
full[0] = np.nan
return interp_nans(full)
...
me_b = bin_average(me, BIN_FRAMES)
edges = np.quantile(me_b, np.arange(1, N_OUT_BINS) / N_OUT_BINS)
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. The trajectory says timestamp spans confirmed this reconstruction for all sessions. Binning matches the paper, and discretization follows the requested five equal-percentile bins per session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four 20%, 40%, 60%, and 80% quantiles are calculated separately for each session after temporal averaging. `np.digitize` maps values to integer classes 0–4.

ii.
```python
edges = np.quantile(me_b, np.arange(1, N_OUT_BINS) / N_OUT_BINS)
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. This directly implements “five equal-percentile bins, selected per session.” The agent verified approximately/exactly balanced 20% class frequencies.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera timestamp gaps map observed motion-energy samples onto imaging-frame indices. Missing positions are interpolated, both streams are averaged over the same 10-frame groups, cropped to a common `T`, and finally sliced with the same trial slice.

ii.
```python
full = np.full(nframes, np.nan)
...
full[idx[keep]] = me[keep]
...
T = min(neural.shape[1], me_b.size)
neural, me_b = neural[:, :T], me_b[:T]
...
output_trials.append(me_cat[sl][None, :])
```

iii. The camera was triggered by the microscope, implying nominal 1:1 frames. The agent used timestamps to restore that correspondence when frames were dropped and sanity-checked timestamp duration across sessions.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames and the invalid first motion-energy sample are represented as NaNs and linearly interpolated; edge NaNs receive the nearest valid value. An all-missing trace raises an error. Any residual neural/behavior length mismatch is cropped to the shorter length, and incomplete final trials are discarded.

ii.
```python
if nans.all():
    raise ValueError('all values missing')
if nans.any():
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
...
T = min(neural.shape[1], me_b.size)
...
ntrials = T // bins_per_trial
```

iii. The agent followed the dataset README's interpolation guidance and used explicit checks/surveys to establish the gap pattern. Cropping and floor division ensure equal stream lengths and complete trials.

## 6-a. What are the most time-consuming steps of the code?

i. The full-session maximin baseline filters dominate computation: Gaussian, long-window minimum, and long-window maximum filters are applied to every neuron. Reading large fluorescence arrays and processing 41 sessions also costs I/O/time.

ii.
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. The trajectory spent most conversion work on full-session neural preprocessing and also tested an alternative preprocessing variant. It did not provide benchmark timings, so this identification is inferred from the operations and agrees with its focus on baseline correction.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Missing-value interpolation and 10-frame averaging are already vectorized. The trial loop mostly packages views/copies into the required nested-list format and cannot be eliminated without later conversion back to lists. Directory/session loops are necessary for separate files. Metadata assembly could be compacted, but it is not a meaningful bottleneck.

ii.
```python
x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
...
return x.reshape(newshape).mean(axis=-1)
...
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
```

iii. The trajectory does not explicitly discuss vectorization. The implementation itself avoids the reference solution's repeated `np.insert` loop by constructing a full indexed array and interpolating all gaps together.

## 6-c. What processing does the code repeat multiple times?

i. Loading, ROI masking, maximin filtering, binning, timestamp reconstruction, quantile calculation, and trial packaging repeat once per session. This repetition is required because sessions have separate files, neurons, timestamps, and per-session percentile thresholds.

ii.
```python
for si, subject in enumerate(subjects):
    ...
    for sess in sessions:
        ...
        dff = maximin_dff(...)
        neural = bin_average(dff, BIN_FRAMES)
        me = load_motion_energy(sdir, nframes)
        me_b = bin_average(me, BIN_FRAMES)
```

iii. No explicit justification is recorded. The per-session output-bin requirement and independently stored recordings make the repeated session pipeline necessary rather than redundant recomputation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `Fneu.npy` is loaded, filtered, converted to `float64`, and passed into `maximin_dff`, but `neucoeff = 0.0` means it has no effect. The full final partial trial is processed through baseline correction, binning, and discretization before being discarded. Extensive metadata and contiguous trial copies are useful for validation/portability but not necessarily decoder calculations.

ii.
```python
Fneu = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'Fneu.npy'))
...
neucoeff = 0.0
Fc = F - neucoeff * Fneu
...
ntrials = T // bins_per_trial
```

iii. The trajectory does not identify these as waste. `Fneu` was retained to mirror the Track2p function and facilitate the tested 0.7 variant; tail processing follows naturally from processing continuous sessions before trial segmentation.
