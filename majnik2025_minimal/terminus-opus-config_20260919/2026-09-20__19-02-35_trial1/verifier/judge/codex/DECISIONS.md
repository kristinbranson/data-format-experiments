# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code scans `/app/data` for all top-level directories, treats each directory as a subject, scans each subject directory for subdirectories as sessions, and then loads each session's Suite2p files plus behavioral files on demand inside the session loop. Neural files come from `suite2p/plane0`, while motion energy and timestamps come from `move_deve`.

ii. 
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])

for si, subject in enumerate(subjects):
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        sdir = os.path.join(subj_dir, sess)
        ops = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'ops.npy'),
                      allow_pickle=True).item()
        F = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'F.npy'))
        Fneu = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'Fneu.npy'))
        iscell = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'iscell.npy'))
        ...
        me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
        ts = np.load(os.path.join(session_dir, 'move_deve', 'tstamps.npy')).astype(np.float64)
```

iii. In the trajectory, the agent first inspected `/app/data`, then read the dataset README and concluded the data layout was "mice ... each with session dates, each containing `move_deve` and `suite2p/plane0` outputs." It also ran a survey across all sessions and reported "41 sessions across 6 mice," which it then used as the basis for iterating over every directory and session.

## 1-b. How are the data split into subjects?

i. Subjects are defined as every top-level directory under `/app/data`, sorted lexicographically.

ii. 
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
```

iii. The agent relied on the README statement that unzipping the dataset yields six subject folders and reported in the trajectory that it had found "6 mice." It did not add an explicit `jm*` prefix filter; it assumed the subject directories were exactly the relevant top-level directories.

## 1-c. How are the data split into sessions?

i. Sessions are defined as every subdirectory inside each subject directory, again sorted lexicographically.

ii. 
```python
subj_dir = os.path.join(DATA_DIR, subject)
sessions = sorted([d for d in os.listdir(subj_dir)
                   if os.path.isdir(os.path.join(subj_dir, d))])
for sess in sessions:
    sdir = os.path.join(subj_dir, sess)
```

iii. In the trajectory, the agent described the layout as subject folders containing session-date folders, and it used that structure directly. The sorting appears to be for deterministic ordering rather than based on an additional scientific justification.

## 1-d. How are the data split into trials?

i. The code creates artificial trials by cutting each binned continuous session into consecutive non-overlapping 60-second segments. After 10-frame binning at 30 Hz, each trial contains `180` bins. Any incomplete remainder at the end of a session is implicitly discarded because only complete slices are emitted.

ii. 
```python
bin_size = BIN_FRAMES / fs
tvec = (np.arange(T) + 0.5) * bin_size

bins_per_trial = int(round(TRIAL_SEC / bin_size))
ntrials = T // bins_per_trial

neural_trials, input_trials, output_trials = [], [], []
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(tvec[sl][None, :].astype(np.float32))
    output_trials.append(me_cat[sl][None, :])
```

iii. The trajectory states that the agent would "split into 60 s trials (180 bins)" because the decoder instructions explicitly required 60-second trials and the recording is continuous rather than naturally trialized.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality-control filter. The only effective filtering is that incomplete trailing data that cannot fill a full 60-second segment is not included in the trial lists.

ii. 
```python
bins_per_trial = int(round(TRIAL_SEC / bin_size))
ntrials = T // bins_per_trial

for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    ...
```

iii. The trajectory does not show a separate justification for excluding or retaining specific trials. The agent focused on satisfying the fixed 60-second trial requirement and on keeping the data format valid.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices are derived primarily from `F.npy` and `Fneu.npy`, with `ops.npy` used for frame rate and baseline parameters, and `iscell.npy` used to select ROIs before preprocessing.

ii. 
```python
ops = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'ops.npy'),
              allow_pickle=True).item()
fs = float(ops['fs'])
F = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'Fneu.npy'))
iscell = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'iscell.npy'))
keep = iscell[:, 0] > 0.5
F, Fneu = F[keep], Fneu[keep]
```

iii. In the trajectory, the agent inspected `F`, `Fneu`, `ops`, `spks`, and `iscell`, then concluded it should use the Track2p/Suite2p fluorescence traces with Track2p-style baseline processing. It also noted that all provided ROIs had `iscell == 1`, but still chose to apply the `iscell > 0.5` filter.

## 2-b. How is the `neural` data processed?

i. The code performs a custom maximin-style baseline subtraction modeled on `track2p/gui/data_management.py::F_processing`. It explicitly sets `neucoeff = 0.0`, so no neuropil subtraction is applied. After baseline subtraction, it averages neural activity in non-overlapping 10-frame bins and stores the result as `float32`.

ii. 
```python
def maximin_dff(F, Fneu, ops):
    fs = float(ops['fs'])
    neucoeff = 0.0
    sig_baseline = float(ops.get('sig_baseline', 10.0))
    win_baseline = float(ops.get('win_baseline', 60.0))
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc, [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow

dff = maximin_dff(F.astype(np.float64), Fneu.astype(np.float64), ops)
neural = bin_average(dff, BIN_FRAMES).astype(np.float32)
```

iii. The trajectory shows the agent grepping Track2p GUI code, finding `F_processing(... neucoeff=0.0, baseline='maximin' ...)`, and then deciding to match that implementation. It later tested a neuropil-subtracted variant with `neucoeff=0.7`, observed similar decoder accuracy, and explicitly said it would keep the "track2p-repo-consistent processing (neucoeff=0)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code filters ROIs using `iscell[:, 0] > 0.5` before neural preprocessing. In the provided dataset this has no practical effect because the agent found all ROIs already marked as cells.

ii. 
```python
iscell = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'iscell.npy'))
keep = iscell[:, 0] > 0.5
F, Fneu = F[keep], Fneu[keep]
```

iii. The agent inspected `iscell.npy` during exploration and noted in the trajectory that the provided Track2p-tracked cells had "all iscell==1." Despite that, it justified keeping the filter because it believed this matched the standard paper/Suite2p cell-selection criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start of the imaging session, specifically the first 2-photon frame. Trials are contiguous session-start-relative slices rather than stimulus-locked epochs.

ii. 
```python
tvec = (np.arange(T) + 0.5) * bin_size
...
sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
...
'temporal_alignment_event': 'start of the imaging session (first 2-photon frame)',
'off_start': 0.0,
'off_end': 60.0,
```

iii. The trajectory says the agent would use "time from session start as input" and later describes the dataset as continuous spontaneous activity with no natural event other than session onset, so it treated session start as the alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping bins at 30 Hz, giving a temporal resolution of `10 / 30 = 0.333...` seconds, or `333.33 ms`. This rebinning is applied to both neural and behavioral traces.

ii. 
```python
BIN_FRAMES = 10
...
neural = bin_average(dff, BIN_FRAMES).astype(np.float32)
me_b = bin_average(me, BIN_FRAMES)
...
'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,
```

iii. The agent repeatedly cited the methods text statement that decoding analyses "denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps" and therefore rebinned both streams to 3 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded directly from a raw timestamp array. It is derived from the frame rate in `ops.npy`, the fixed bin size, and the binned sample index within a session.

ii. 
```python
fs = float(ops['fs'])
bin_size = BIN_FRAMES / fs
tvec = (np.arange(T) + 0.5) * bin_size
```

iii. The trajectory states that the agent would use "time from session start as input." It did not identify a separate raw experiment-time variable and instead derived time analytically from the constant sampling rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code computes one scalar time per binned sample, using the center of each 10-frame bin rather than the left edge. It then slices that continuous session-level vector into the same 60-second trial windows used for neural data and casts each trial to `float32`.

ii. 
```python
bin_size = BIN_FRAMES / fs
tvec = (np.arange(T) + 0.5) * bin_size
...
input_trials.append(tvec[sl][None, :].astype(np.float32))
```

iii. The trajectory does not explicitly justify choosing bin centers over bin starts. The only explicit rationale is that the input should be "time from session start" and share the same binned temporal grid as the neural data.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned by construction: the code uses the same post-binning length `T`, the same `bins_per_trial`, and the same trial slices `sl` for both `neural` and `input`.

ii. 
```python
T = min(neural.shape[1], me_b.size)
...
sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
input_trials.append(tvec[sl][None, :].astype(np.float32))
```

iii. The agent's trajectory repeatedly frames the solution around a shared 10-frame temporal grid for neural activity, motion energy, and time. No separate alignment step for input time is used beyond sharing indices with the neural matrix.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`, with `move_deve/tstamps.npy` used to reconstruct dropped-frame positions. The code does not use `interframe_int.npy`.

ii. 
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
ts = np.load(os.path.join(session_dir, 'move_deve', 'tstamps.npy')).astype(np.float64)
```

iii. In the trajectory, the agent read the README note that missing frames could be found from "`tstamps.npy` or `interframe_int.npy`" and chose timestamp-based reconstruction. It repeatedly described dropped camera frames as recoverable from `tstamps/interframe_int`, but the final code uses only `tstamps.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code loads the precomputed motion-energy trace, reconstructs a full-length vector when camera frames are missing by mapping observed samples onto imaging-frame indices inferred from timestamp gaps, marks missing positions as `NaN`, linearly interpolates them, forces the first sample to `NaN` because motion energy is defined between consecutive video frames, bins the result in 10-frame averages, and later discretizes it.

ii. 
```python
full = np.full(nframes, np.nan)
if me.size == nframes:
    full[:] = me
else:
    d = np.diff(ts)
    med = np.median(d)
    steps = np.round(d / med).astype(int)
    steps[steps < 1] = 1
    idx = np.concatenate([[0], np.cumsum(steps)])
    keep = idx < nframes
    full[idx[keep]] = me[keep]
full[0] = np.nan
return interp_nans(full)

me = load_motion_energy(sdir, nframes)
me_b = bin_average(me, BIN_FRAMES)
```

iii. The trajectory justification was that camera and imaging frames should be in 1:1 correspondence, that dropped camera frames should be recovered from timestamps, and that interpolation was appropriate because the README explicitly suggested treating those positions as missing values or interpolating over them.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized separately within each session into five equal-frequency bins using the 20th, 40th, 60th, and 80th percentiles of the binned motion-energy trace, then `np.digitize` maps each time point to an integer class `0` through `4`.

ii. 
```python
edges = np.quantile(me_b, np.arange(1, N_OUT_BINS) / N_OUT_BINS)
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. The trajectory states that the agent would "discretize motion energy into 5 per-session quintile bins" because the decoder instructions required five equal-percentile bins selected per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The code attempts to align motion energy to imaging frames before binning by reconstructing a length-`nframes` motion-energy vector indexed on imaging frames. After both streams are binned, it sets `T = min(neural.shape[1], me_b.size)` and truncates both streams to the shared minimum length before trialization.

ii. 
```python
me = load_motion_energy(sdir, nframes)
me_b = bin_average(me, BIN_FRAMES)

T = min(neural.shape[1], me_b.size)
neural, me_b = neural[:, :T], me_b[:T]
...
output_trials.append(me_cat[sl][None, :])
```

iii. The agent justified alignment by the paper/README claim that the camera was triggered by the microscope, implying one behavioral sample per imaging frame except for dropped-camera-frame cases. In the trajectory it additionally ran timestamp sanity checks and concluded the missing frames were recoverable and alignment was correct.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion-energy samples are handled by building a full-length vector with `NaN` gaps and linearly interpolating across them. The first motion-energy sample is always treated as invalid and interpolated. If the binned neural and motion-energy lengths still differ, the code silently truncates both to the shared minimum length. Any trailing partial trial is dropped because only full trials are emitted.

ii. 
```python
full = np.full(nframes, np.nan)
...
full[0] = np.nan
return interp_nans(full)

T = min(neural.shape[1], me_b.size)
neural, me_b = neural[:, :T], me_b[:T]

ntrials = T // bins_per_trial
for tr in range(ntrials):
    ...
```

iii. The trajectory cites the dataset README as justification for interpolating dropped camera frames, and the agent ran several exploratory checks on frame counts and timestamp gaps before settling on that approach. The trajectory does not provide an explicit justification for silently truncating to `min(...)`; that behavior is only evident in the final code.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive parts of the conversion code are the per-session full-trace neural preprocessing steps, especially the Gaussian smoothing plus min/max filtering over every neuron's entire recording, followed by the repeated loading of large `.npy` arrays for every session.

ii. 
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
...
for si, subject in enumerate(subjects):
    ...
    for sess in sessions:
        F = np.load(...)
        Fneu = np.load(...)
        ...
        dff = maximin_dff(...)
```

iii. The trajectory repeatedly focused on the baseline-correction step as the substantive neural preprocessing operation and even benchmarked alternative preprocessing variants, implying that this step dominated the scientific part of the computation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit per-trial Python loop used to append slices could be replaced by a reshape/split-based approach after trimming to a multiple of `bins_per_trial`. Most other expensive array operations are already vectorized.

ii. 
```python
neural_trials, input_trials, output_trials = [], [], []
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(tvec[sl][None, :].astype(np.float32))
    output_trials.append(me_cat[sl][None, :])
```

iii. The trajectory does not discuss efficiency at this level. This assessment is inferred from the final code structure rather than from an explicit optimization rationale given by the agent.

## 6-c. What processing does the code repeat multiple times?

i. There is no major redundant scientific processing pass in the final script. The only small repeated work is routine per-session recalculation of the same derived quantities such as `bin_size`, `bins_per_trial`, and trial slices, plus repeated dtype conversions before and after preprocessing.

ii. 
```python
dff = maximin_dff(F.astype(np.float64), Fneu.astype(np.float64), ops)
neural = bin_average(dff, BIN_FRAMES).astype(np.float32)
...
bin_size = BIN_FRAMES / fs
tvec = (np.arange(T) + 0.5) * bin_size
bins_per_trial = int(round(TRIAL_SEC / bin_size))
```

iii. The trajectory does not identify a repeated-processing concern. The script is mostly a single pass over sessions followed by trial slicing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is loading `Fneu.npy` even though the chosen preprocessing sets `neucoeff = 0.0`, so the neuropil trace has no effect on the final neural output. The script also loads `iscell.npy` even though the agent had already found all entries were `1`, and it stores detailed `session_info` metadata that the downstream decoder does not use.

ii. 
```python
Fneu = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'Fneu.npy'))
...
neucoeff = 0.0
...
Fc = F - neucoeff * Fneu

iscell = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'iscell.npy'))
keep = iscell[:, 0] > 0.5
...
session_info.append({'subject': subject, 'session': sess,
                     'n_neurons': int(neural.shape[0]),
                     'n_trials': int(ntrials),
                     'fs_imaging_hz': fs})
```

iii. The trajectory explicitly says the agent kept `neucoeff = 0` for Track2p consistency and kept the `iscell` filter because it thought that matched the paper. It does not frame these as efficiency costs, but they are unnecessary for the final downstream dataset actually produced.
