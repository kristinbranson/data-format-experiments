# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for subject directories whose names start with `jm`, then scans each subject directory for all session subdirectories. For each session it loads neural fluorescence from `suite2p/plane0/F.npy` and `Fneu.npy`, Suite2p parameters from `ops.npy`, and behavioral motion data from `move_deve/motion_energy_glob.npy` plus `move_deve/tstamps.npy` for alignment.

ii. 
```python
ROOT = Path('/app/data')
...
subjects = sorted(p.name for p in ROOT.glob('jm*') if p.is_dir())
...
for si, subject in enumerate(subjects):
    sessions = sorted(p for p in (ROOT / subject).iterdir() if p.is_dir())
    for session in sessions:
        plane = session / 'suite2p' / 'plane0'
        F = np.load(plane / 'F.npy', mmap_mode='r')
        Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
        ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
        ...
        motion = aligned_motion(session, F.shape[1])
```

iii. In trajectory step 4, the agent said the README "confirms six mice, daily sessions". In step 12, it summarized the final loading decision as "all six subjects and all daily sessions" and "the supplied Track2p-curated neurons."

## 1-b. How are the data split into subjects?

i. Subjects are identified purely by directory name: every directory under `/app/data` matching `jm*` is treated as one mouse, and the subject list is sorted lexicographically.

ii. 
```python
subjects = sorted(p.name for p in ROOT.glob('jm*') if p.is_dir())
```

iii. In trajectory step 12, the agent explicitly said it would include "all six subjects."

## 1-c. How are the data split into sessions?

i. Each session is any subdirectory inside a subject directory. Sessions are sorted and processed one daily recording at a time.

ii. 
```python
for si, subject in enumerate(subjects):
    sessions = sorted(p for p in (ROOT / subject).iterdir() if p.is_dir())
    for session in sessions:
        ...
```

iii. In trajectory step 4, the agent described the README as confirming "daily sessions," and in step 12 it summarized the decision as using "all daily sessions."

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous sessions with no natural trial structure. It bins the continuous data first, then splits each session into consecutive non-overlapping 60 s trials. Only complete trials are kept, because `ntrials` is the floor of total bins divided by bins per trial.

ii. 
```python
AVG_FRAMES = 10
TRIAL_SECONDS = 60
...
out_fs = fs / AVG_FRAMES
trial_bins = int(round(TRIAL_SECONDS * out_fs))
ntrials = dff.shape[1] // trial_bins
...
for tr in range(ntrials):
    a, b = tr * trial_bins, (tr + 1) * trial_bins
    ns.append(np.ascontiguousarray(dff[:, a:b], dtype=np.float32))
    ins.append(elapsed[None, a:b].copy())
    outs.append(labels[None, a:b].copy())
```

iii. In trajectory step 7, the agent reasoned that the requested "60-sec trials" should be created from the continuous recording. In step 12, it summarized the final decision as "non-overlapping complete 60-second trials."

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply an explicit trial-quality filter. The only effective exclusion is structural: incomplete trailing data that does not make a full 60 s trial is omitted because only `ntrials = floor(total_bins / trial_bins)` trials are emitted.

ii. 
```python
trial_bins = int(round(TRIAL_SECONDS * out_fs))
ntrials = dff.shape[1] // trial_bins
...
for tr in range(ntrials):
    a, b = tr * trial_bins, (tr + 1) * trial_bins
```

iii. There is no separate trial-QC justification in the trajectory. The closest statement is step 12, where the agent described the output as "complete 60-second trials."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from raw fluorescence `F.npy`, neuropil fluorescence `Fneu.npy`, and Suite2p parameter metadata in `ops.npy`.

ii. 
```python
plane = session / 'suite2p' / 'plane0'
F = np.load(plane / 'F.npy', mmap_mode='r')
Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
```

iii. In trajectory step 8, the agent said "the source `F.npy` is raw fluorescence" and concluded it should reproduce dF/F generation from `F.npy` and `Fneu.npy` rather than use spikes.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil subtraction using the session's `neucoeff`, then computes a manual Suite2p-like maximin baseline using Gaussian smoothing followed by min and max filters. It converts the trace to dF/F by dividing `(x - base)` by `base`, and later averages the resulting neural trace over 10-frame bins.

ii. 
```python
def maximin_dff(F, Fneu, ops):
    x = np.asarray(F, dtype=np.float32) - np.float32(ops.get('neucoeff', .7)) * np.asarray(Fneu, dtype=np.float32)
    smooth = gaussian_filter1d(x, float(ops.get('sig_baseline', 10.0)), axis=1, mode='reflect')
    win = int(round(float(ops.get('win_baseline', 60.0)) * float(ops['fs'])))
    base = minimum_filter1d(smooth, size=win, axis=1, mode='reflect')
    base = maximum_filter1d(base, size=win, axis=1, mode='reflect')
    eps = np.finfo(np.float32).eps
    denom = np.where(np.abs(base) > eps, base, np.where(base < 0, -eps, eps))
    x = (x - base) / denom
    return x.astype(np.float32, copy=False)
...
dff = maximin_dff(F, Fneu, ops)
...
dff = dff[:, :usable].reshape(F.shape[0], -1, AVG_FRAMES).mean(2, dtype=np.float32)
```

iii. In trajectory step 7, the agent said the paper specified "baseline-corrected fluorescence (dF/F), not Suite2p deconvolved spikes." In step 12 it summarized the chosen preprocessing as "neuropil-corrected dF/F using Suite2p's stored default maximin baseline parameters."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not filter neurons inside the conversion script. Instead, it assumes the provided `suite2p` matrices already contain the Track2p-curated neuron population: cells passing Suite2p cell classification and tracked across all days.

ii. 
```python
"""Convert the Majnik et al. Track2p developmental barrel-cortex dataset.

Processing choices follow the supplied paper/methods and data README:
* use every supplied mouse/session and the supplied Track2p-curated population
  (these are cells detected at Suite2p probability > .5 and tracked on all days);
...
"""
...
F = np.load(plane / 'F.npy', mmap_mode='r')
Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
```

iii. In trajectory step 7, the agent reasoned that it should "retain ROIs classified as cells above probability 0.5" and that "the distributed tracked matrices already contain only these." Step 12 repeats this as using "the supplied Track2p-curated neurons."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats each consecutive 60 s block as a trial-alignment event. Neural trials are aligned to the start of each recording block, not to a stimulus or to absolute session start in metadata, even though the actual trial slicing comes from the continuous session trace.

ii. 
```python
for tr in range(ntrials):
    a, b = tr * trial_bins, (tr + 1) * trial_bins
    ns.append(np.ascontiguousarray(dff[:, a:b], dtype=np.float32))
...
'metadata': {
    ...
    'temporal_alignment_event': 'start of each consecutive 60-second recording block',
    'off_start': 0.0, 'off_end': 60.0,
    ...
}
```

iii. In trajectory step 15, the agent explicitly noted that "trials align to each 60-second block start while the decoder input remains absolute session elapsed time."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and motion data by averaging non-overlapping groups of 10 imaging frames. With 30 Hz source data, that produces 3 Hz output and a `333.33 ms` time bin.

ii. 
```python
AVG_FRAMES = 10
...
usable = (F.shape[1] // AVG_FRAMES) * AVG_FRAMES
dff = dff[:, :usable].reshape(F.shape[0], -1, AVG_FRAMES).mean(2, dtype=np.float32)
motion = motion[:usable].reshape(-1, AVG_FRAMES).mean(1)
...
'time_bin_size': 1000.0 * AVG_FRAMES / 30.0,
```

iii. In trajectory steps 7 and 12, the agent said the paper's decoding "slightly denoised the dF/F as well as the behaviour traces by averaging each 10 consecutive timestamps" and summarized the decision as "10-frame averaging of both neural and motion streams."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not read a separate time variable from disk. It derives elapsed time from the binned frame index and the post-binning sample rate `out_fs = fs / AVG_FRAMES`, producing seconds from session start.

ii. 
```python
out_fs = fs / AVG_FRAMES
...
elapsed = np.arange(dff.shape[1], dtype=np.float32) / np.float32(out_fs)
...
ins.append(elapsed[None, a:b].copy())
```

iii. In trajectory step 15, the agent described this variable as "absolute session elapsed time."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a session-long regularly spaced elapsed-time vector after temporal binning, using the binned sample rate. It then slices this vector into trial segments matching the neural trial boundaries.

ii. 
```python
out_fs = fs / AVG_FRAMES
elapsed = np.arange(dff.shape[1], dtype=np.float32) / np.float32(out_fs)
for tr in range(ntrials):
    a, b = tr * trial_bins, (tr + 1) * trial_bins
    ins.append(elapsed[None, a:b].copy())
```

iii. There is no separate detailed justification for this computation in the trajectory beyond step 15's statement that the decoder input remains "absolute session elapsed time."

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The elapsed-time input is aligned by construction: it is defined on the same post-binning time base as the neural matrix, and each trial uses the same `[a:b]` bin slice for neural data and time input.

ii. 
```python
dff = dff[:, :usable].reshape(F.shape[0], -1, AVG_FRAMES).mean(2, dtype=np.float32)
...
elapsed = np.arange(dff.shape[1], dtype=np.float32) / np.float32(out_fs)
for tr in range(ntrials):
    a, b = tr * trial_bins, (tr + 1) * trial_bins
    ns.append(np.ascontiguousarray(dff[:, a:b], dtype=np.float32))
    ins.append(elapsed[None, a:b].copy())
```

iii. In trajectory step 15, the agent explicitly contrasted "trials align to each 60-second block start" with time input that stays as "absolute session elapsed time," implying alignment by shared trial slicing rather than re-zeroing.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy from `move_deve/motion_energy_glob.npy` and uses `move_deve/tstamps.npy` to infer dropped camera-frame positions before alignment to imaging frames.

ii. 
```python
def aligned_motion(session, nframes):
    md = session / 'move_deve'
    motion = np.load(md / 'motion_energy_glob.npy').astype(np.float64)
    stamps = np.load(md / 'tstamps.npy').astype(np.float64)
```

iii. In trajectory step 11, the agent said "motion timestamps reveal dropped frames exactly," and in step 12 it summarized the decision as "interpolation at dropped camera-frame indices identified from timestamps."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI first reconstructs a complete frame-aligned motion trace by mapping timestamped camera samples onto the full imaging-frame index and linearly interpolating across missing slots. It then averages motion over 10-frame bins and discretizes the binned values into session-specific quintiles.

ii. 
```python
dt = np.median(np.diff(stamps))
slots = np.rint((stamps - stamps[0]) / dt).astype(np.int64)
...
unique, idx = np.unique(slots, return_index=True)
return np.interp(np.arange(nframes), unique, motion[idx])
...
motion = motion[:usable].reshape(-1, AVG_FRAMES).mean(1)
labels = quintile_labels(motion)
```

iii. In trajectory step 12, the agent summarized the final motion-processing choice as "aligns/interpolates missing motion frames, applies the paper's 10-frame temporal averaging, [and] creates session-specific motion-energy quintiles."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded separately within each session into five equal-percentile bins. The AI computes the 20th, 40th, 60th, and 80th percentiles and assigns integer labels `0..4`.

ii. 
```python
def quintile_labels(x):
    """Five session-wise equal-percentile bins, labels 0..4."""
    edges = np.quantile(x, [.2, .4, .6, .8])
    return np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. In trajectory steps 12 and 16, the agent described the output as "session-specific motion-energy quintiles."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses camera timestamps to reconstruct motion energy on the full imaging-frame grid, then bins motion and neural traces with the same 10-frame averaging and slices them into trials using the same boundaries.

ii. 
```python
motion = aligned_motion(session, F.shape[1])
...
dff = dff[:, :usable].reshape(F.shape[0], -1, AVG_FRAMES).mean(2, dtype=np.float32)
motion = motion[:usable].reshape(-1, AVG_FRAMES).mean(1)
...
for tr in range(ntrials):
    a, b = tr * trial_bins, (tr + 1) * trial_bins
    ns.append(np.ascontiguousarray(dff[:, a:b], dtype=np.float32))
    outs.append(labels[None, a:b].copy())
```

iii. In trajectory step 12, the agent described this as interpolating "onto the complete imaging-frame index." Step 11 framed timestamps as the reliable source for locating dropped frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing camera frames by reconstructing missing frame slots from timestamps and linearly interpolating motion energy over them. It also raises errors for unexpected motion/timestamp mismatches, unexpected imaging frame rates, or `F`/`Fneu` shape mismatches. Incomplete trailing frames and incomplete final trials are dropped implicitly by truncation to `usable` and `ntrials`.

ii. 
```python
if len(motion) != len(stamps):
    raise ValueError(f'motion/timestamp mismatch in {session}')
...
if fs != 30 or F.shape != Fneu.shape:
    raise ValueError(f'unexpected imaging data in {session}: {F.shape}, fs={fs}')
...
usable = (F.shape[1] // AVG_FRAMES) * AVG_FRAMES
...
ntrials = dff.shape[1] // trial_bins
```

iii. In trajectory step 4, the agent noted that "missing camera frames can be identified from timestamps/inter-frame intervals and interpolated." Step 12 carried that into the implementation summary.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive work in the AI's code is likely the neural preprocessing in `maximin_dff`, which performs several full-trace filtering passes over every neuron for every session, plus the associated large-array I/O. Pickle serialization of the final dataset is also nontrivial because the output file is large.

ii. 
```python
smooth = gaussian_filter1d(x, float(ops.get('sig_baseline', 10.0)), axis=1, mode='reflect')
win = int(round(float(ops.get('win_baseline', 60.0)) * float(ops['fs'])))
base = minimum_filter1d(smooth, size=win, axis=1, mode='reflect')
base = maximum_filter1d(base, size=win, axis=1, mode='reflect')
...
F = np.load(plane / 'F.npy', mmap_mode='r')
Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory does not contain an explicit performance analysis for this question. The closest evidence is step 13, where the agent noted the converter was still processing all sessions, and step 14, where it reported a `414 MB` pickle output.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already avoids the reference solution's repeated `np.insert` loop for dropped frames. The main remaining loop that could be vectorized is the per-trial append loop, where each trial slice is copied into Python lists one trial at a time.

ii. 
```python
ns, ins, outs = [], [], []
elapsed = np.arange(dff.shape[1], dtype=np.float32) / np.float32(out_fs)
for tr in range(ntrials):
    a, b = tr * trial_bins, (tr + 1) * trial_bins
    ns.append(np.ascontiguousarray(dff[:, a:b], dtype=np.float32))
    ins.append(elapsed[None, a:b].copy())
    outs.append(labels[None, a:b].copy())
```

iii. The trajectory does not include an explicit justification about vectorization choices.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats loading `motion_energy_glob.npy`: once inside `aligned_motion` for actual preprocessing, and again when building `session_info` just to record the raw motion-frame count. It also repeats trial slicing logic separately for neural, input, and output list construction inside the trial loop.

ii. 
```python
motion = np.load(md / 'motion_energy_glob.npy').astype(np.float64)
...
session_info.append({
    'subject': subject, 'session': session.name,
    'source_frames': int(F.shape[1]), 'motion_frames': int(np.load(session/'move_deve'/'motion_energy_glob.npy', mmap_mode='r').shape[0]),
    'n_neurons': int(F.shape[0]), 'n_trials': ntrials,
    'suite2p_fs_hz': fs
})
...
for tr in range(ntrials):
    a, b = tr * trial_bins, (tr + 1) * trial_bins
    ns.append(...)
    ins.append(...)
    outs.append(...)
```

iii. The trajectory does not explicitly justify this repeated processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code collects and stores extra `session_info` metadata that is not needed by the downstream decoder task, and it performs an extra raw motion-file load solely for that metadata. It also keeps verbose descriptive metadata fields that are not required for decoding.

ii. 
```python
session_info.append({
    'subject': subject, 'session': session.name,
    'source_frames': int(F.shape[1]), 'motion_frames': int(np.load(session/'move_deve'/'motion_energy_glob.npy', mmap_mode='r').shape[0]),
    'n_neurons': int(F.shape[0]), 'n_trials': ntrials,
    'suite2p_fs_hz': fs
})
...
'metadata': {
    'task_description': 'Decode session-wise motion-energy quintile from barrel-cortex calcium activity.',
    ...
    'neural_measure': 'neuropil-corrected, Suite2p maximin-baseline dF/F averaged over 10 frames',
    'motion_processing': 'global squared frame-difference energy; missing camera frames linearly interpolated; 10-frame means; session quintiles',
    'trial_definition': 'consecutive non-overlapping 60-second blocks',
    'session_info': session_info,
    'source': 'Majnik et al. (2025), Track2p developmental barrel-cortex dataset'
}
```

iii. The trajectory does not explicitly justify this extra metadata or the repeated file read.
