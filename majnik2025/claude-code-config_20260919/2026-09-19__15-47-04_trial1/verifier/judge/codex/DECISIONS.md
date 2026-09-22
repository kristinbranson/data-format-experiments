# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the six released mouse IDs, scans every subdirectory under each mouse in sorted order, and processes every resulting session (41 in full mode). For each session it loads `F.npy`, `iscell.npy`, and `ops.npy`; `Fneu.npy` is loaded only for a nonzero neuropil coefficient. It also loads `motion_energy_glob.npy` and `interframe_int.npy`. The processed session is immediately appended to the target dictionary.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def find_sessions(data_root=DATA_ROOT):
    sessions = []
    for subject in SUBJECTS:
        subject_dir = os.path.join(data_root, subject)
        names = sorted(f.name for f in os.scandir(subject_dir) if f.is_dir())
        for name in names:
            sessions.append((subject, name, os.path.join(subject_dir, name)))
    return sessions

for isess, (subject, name, session_dir) in enumerate(sessions):
    res = process_session(subject, name, session_dir, neucoeff=args.neucoeff,
                          dff_mode=args.dff_mode)
```

iii. The agent states that the fixed subject order comes from the dataset README and that sorted session folder names are chronological. Its notes report that this finds all 6 mice and 41 sessions. Processing one session at a time was chosen to constrain peak memory.

## 1-b. How are the data split into subjects?

i. The top-level `jm*` identities are represented by a fixed six-element `SUBJECTS` list. Each session receives the integer position of its mouse in that list. Sessions from a mouse remain grouped because `find_sessions()` iterates subjects outermost.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
data['subject_idx'].append(SUBJECTS.index(subject))
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The notes say these IDs and their A–F ordering are specified by the dataset README; this produced 7, 7, 7, 7, 6, and 7 sessions per mouse.

## 1-c. How are the data split into sessions?

i. Every directory immediately beneath a subject is one session. Names are sorted within subject, and each directory becomes one element of `neural`, `input`, and `output`.

ii.
```python
names = sorted(f.name for f in os.scandir(subject_dir) if f.is_dir())
for name in names:
    sessions.append((subject, name, os.path.join(subject_dir, name)))

data['neural'].append(res['neural'])
data['input'].append(res['input'])
data['output'].append(res['output'])
```

iii. The agent interprets each dated subdirectory as a daily recording and uses sorting for deterministic chronological order.

## 1-d. How are the data split into trials?

i. The continuous binned session is cut from session start into consecutive, non-overlapping 60-second trials. At 30 Hz with 10-frame bins this is 180 bins (1,800 original frames) per trial. An incomplete tail is dropped, and at least two full trials are required.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * FS / BIN_FRAMES))
n_trials = n_bins // BINS_PER_TRIAL
assert n_trials >= 2
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(dff_binned[:, sl]))
    input_trials.append(bin_centre_time[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :].astype(np.int8))
```

iii. The experiment has no native trials, so the agent follows the explicit instruction to form 60-second blocks. It reports that released frame counts yield exactly 20 or 30 complete trials and no trial tail in the full data.

## 1-e. How are trials filtered based on quality controls?

i. There is no content-based trial rejection. Every complete 60-second block is retained; only an incomplete terminal block would be discarded, and sessions with fewer than two complete blocks cause an assertion failure.

ii.
```python
n_trials = n_bins // BINS_PER_TRIAL
assert n_trials >= 2, f'{session_dir}: only {n_trials} complete 60 s trials'
if n_trials * BINS_PER_TRIAL != n_bins and verbose:
    print(f'    note: dropping {n_bins - n_trials * BINS_PER_TRIAL} trailing bins '
          f'(incomplete 60 s trial)')
```

iii. The notes justify retaining all complete blocks because no experimental trial-quality labels exist and every session greatly exceeds the two-trial minimum.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. In the default conversion, neural data is derived from tracked-cell `F.npy` only. The code also loads `iscell.npy` and `ops.npy` for validation and parameters. Although the processing function accepts `Fneu`, the default `neucoeff=0.0` means `Fneu.npy` is not loaded or used.

ii.
```python
F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float32)
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
if neucoeff != 0.0:
    Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float32)
else:
    Fneu = np.float32(0.0)
```

iii. The agent found that the paper repository calls its `F_processing()` with the default `neucoeff=0.0`, and therefore prioritized that observed call over `ops['neucoeff']=0.7`. It also notes the released rows were already Track2p-curated cells.

## 2-b. How is the `neural` data processed?

i. The default path computes `Fc = F` (because `neucoeff=0`), estimates a maximin baseline by Gaussian filtering (sigma 10 frames), a 60-second minimum filter, and a maximum filter, then subtracts that baseline. The result is cast to float32, averaged in non-overlapping 10-frame bins, and sliced into trials. It does not divide by the baseline despite calling the result dF/F.

ii.
```python
Fc = F - neucoeff * Fneu
win = int(win_baseline * fs)
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
F = Fc - Flow

dff_binned = bin_time(dff).astype(np.float32)
```

iii. The agent copied the authors’ `F_processing()` mathematics and argues that subtraction, not division, is what the repository implements for its displayed “dF/F0” trace. It tested alternative neuropil and normalization choices but retained the repository-default path.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It asserts every exported ROI is marked a cell with probability above 0.5 and that frame count/rate match `ops`. It additionally flags any all-zero `F` row as failed extraction, unions failure masks across days of each mouse, and removes that tracked neuron from every session of that mouse. Eight unique failed ROIs caused 56 neuron-session removals.

ii.
```python
assert np.all(iscell[:, 0] == 1)
assert np.all(iscell[:, 1] > 0.5)
failed = np.all(F == 0, axis=1)

bad_per_subject[subject] = mask if prev is None else (prev | mask)
keep = ~bad
data['neural'][isess] = [t[keep] for t in data['neural'][isess]]
```

iii. The agent says Track2p and the export already applied the intended cell and all-day tracking curation. It treats all-zero fluorescence with normal neuropil as missing extraction rather than biological silence, and removes the corresponding tracked cell on all days to preserve row identity.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event. Trials begin at session start and then every 60 seconds; neural bins are direct contiguous slices. Metadata describes the alignment event as the first imaging frame/session start and uses offsets 0 to 60 seconds.

ii.
```python
sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
neural_trials.append(np.ascontiguousarray(dff_binned[:, sl]))

'temporal_alignment_event': ('start of the imaging session (first 2-photon frame); ...'),
'off_start': 0.0,
'off_end': TRIAL_SECONDS,
```

iii. The agent justifies session-start alignment because recordings contain spontaneous activity with no task, stimulus, or native trial event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural and behavior streams are averaged in identical non-overlapping groups of 10 frames. With nominal 30 Hz sampling, the converted bin is 333.333 ms (3 Hz). Any shorter final group is dropped.

ii.
```python
BIN_FRAMES = 10
BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FS
n = (x.shape[-1] // bin_frames) * bin_frames
return x[..., :n].reshape(new_shape).mean(axis=-1)
```

iii. The paper says decoding analyses denoised both dF/F and behavior by averaging 10 consecutive timestamps. Identical binning also maintains stream alignment.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the global index of each 10-frame bin and the nominal 30 Hz frame rate; it is not read from a raw timestamp array.

ii.
```python
bin_centre_time = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
```

iii. The notes say `ops['fs']` is 30 for every session and choose the nominal imaging grid used by the reference processing. The measured camera interval is documented separately.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code assigns each averaged bin its center time: for bin `b`, `(10*b + 4.5)/30` seconds. Values are cast to float32 when trials are created and continue monotonically across trial boundaries.

ii.
```python
bin_centre_time = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
input_trials.append(bin_centre_time[sl][None, :].astype(np.float32))
```

iii. The agent explicitly chose bin centers so the time coordinate represents the ten samples averaged into each neural/behavior bin; reported ranges begin at 0.15 s.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One time-center value is produced for every neural bin, and the exact same trial slice is applied to both arrays. The time does not reset to zero for later trials.

ii.
```python
sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
neural_trials.append(np.ascontiguousarray(dff_binned[:, sl]))
input_trials.append(bin_centre_time[sl][None, :].astype(np.float32))
```

iii. The agent’s diagnostic concatenates trials and verifies that neural and time arrays reconstruct the session-level arrays without a shift.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output begins with the already computed global motion-energy vector `move_deve/motion_energy_glob.npy`. `interframe_int.npy` supplies camera interval information used to map those samples to imaging-frame indices.

ii.
```python
raw = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(md, 'interframe_int.npy')).astype(np.float64)
```

iii. The agent notes that motion energy itself is the supplied summed squared frame difference; conversion need not recompute it from video. Interval data are necessary because the microscope triggers the camera but the camera sometimes misses triggers.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Camera samples are assigned imaging-frame indices by rounding each interval relative to the session’s median interval and cumulatively summing. Missing positions and frame zero are represented as NaN and linearly interpolated; camera samples after imaging ends are ignored. The reconstructed trace is averaged over the same 10-frame bins as neural data, then discretized.

ii.
```python
med = np.median(ifi)
n_intervals = np.round(ifi / med).astype(np.int64)
frame_index = np.concatenate([[0], np.cumsum(n_intervals)])
motion = np.full(n_frames, np.nan)
inside = frame_index < n_frames
motion[frame_index[inside]] = raw[inside]
motion[0] = np.nan
good = ~np.isnan(motion)
motion = np.interp(idx, idx[good], motion[good])
motion_binned = bin_time(motion)
```

iii. The agent follows the dataset README’s recommendation to infer missed camera frames from timestamps and interpolate them. It treats the always-zero first motion-energy sample as a frame-difference boundary artifact and avoids shifting the trace when camera recording extends beyond imaging.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four thresholds are the 20th, 40th, 60th, and 80th percentiles of each session’s full binned motion-energy trace. `np.digitize` maps values into integer labels 0–4 before trial slicing.

ii.
```python
edges = np.percentile(x, np.linspace(0, 100, n_bins + 1)[1:-1])
labels = np.digitize(x, edges, right=False).astype(np.int8)
```

iii. This directly implements five equal-percentile bins selected per session and handles large between-session scale differences; validation found approximately 20% of samples in each class.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Missing-trigger reconstruction places motion on an array of exactly the neural frame count. Neural and motion are then binned with the same frame boundaries, checked for equal bin counts, and sliced with the same trial slices.

ii.
```python
motion, n_missing, motion_raw, frame_index = load_session_motion(session_dir, n_frames)
dff_binned = bin_time(dff).astype(np.float32)
motion_binned = bin_time(motion)
assert motion_binned.shape[0] == dff_binned.shape[1]
output_trials.append(labels[sl][None, :].astype(np.int8))
```

iii. Hardware triggering implies one camera opportunity per imaging frame. Reconstructing missed trigger positions, rather than merely padding at the end, preserves framewise synchrony; plotting and trial-concatenation checks were used to inspect alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera triggers and the invalid initial motion sample are linearly interpolated on the neural frame grid; surplus terminal camera samples are discarded. All-zero neural extraction failures are removed across every day of the affected mouse. Assertions catch malformed `iscell`, frame rate/count, camera interval length, bin-length, and minimum-trial conditions. Incomplete final temporal bins or trials are dropped.

ii.
```python
motion = np.full(n_frames, np.nan)
motion[frame_index[inside]] = raw[inside]
motion[0] = np.nan
motion = np.interp(idx, idx[good], motion[good])

failed = np.all(F == 0, axis=1)
bad_per_subject[subject] |= mask
```

iii. The agent distinguishes missing extraction from silence and preserves longitudinal identity when dropping failed cells. Camera interpolation is explicitly allowed by the data README. It reports no NaN/Inf or zero-variance neurons after these corrections.

## 6-a. What are the most time-consuming steps of the code?

i. Maximin neural baseline computation—Gaussian, long minimum, and maximum filters over every neuron and frame—is the dominant computation. Reading large fluorescence arrays is the main I/O cost; pickle writing and accumulated output storage are secondary.

ii.
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. The notes identify baseline correction as the only heavy processing and report roughly 0.3–0.9 seconds per session for load plus dF/F, versus under 0.05 seconds for motion/binning/splitting.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Core numerical operations are already vectorized: temporal binning uses reshape/mean, missing frames use `np.interp`, and filters operate on full matrices. The remaining trial, session, plotting, and cross-day bookkeeping loops build ragged nested output or perform I/O and are not obvious high-value vectorization targets.

ii.
```python
return x.reshape(new_shape).mean(axis=-1)
motion = np.interp(idx, idx[good], motion[good])
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
```

iii. The agent explicitly replaced prospective per-bin and per-gap loops with vectorized operations. It says this avoids several seconds per session of Python loops, while the small structural loops remain appropriate for variable neuron/session shapes.

## 6-c. What processing does the code repeat multiple times?

i. The same load, baseline, reconstruction, binning, discretization, and trial-splitting pipeline necessarily repeats once per session. `drop_failed_neurons()` also calls `find_sessions()` for each represented subject and, in sample mode, rereads unprocessed sessions’ `F.npy` files to enforce cross-day failure removal. Summary construction later concatenates output/input data again.

ii.
```python
for subject in bad_per_subject:
    for subj, name, sdir in find_sessions():
        ...
        F = np.load(..., mmap_mode='r')

all_out = np.concatenate([o[0] for sess in data['output'] for o in sess])
all_in = np.concatenate([i[0] for sess in data['input'] for i in sess])
```

iii. The agent documents session-by-session repetition as a memory optimization. The extra scans in sample mode are intentional because its curation rule is “failed on any day,” even for days omitted from the sample conversion; summary concatenations are validation diagnostics.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `process_session()` retains large raw/intermediate arrays (`F`, baseline, unbinned dF/F, raw/reconstructed motion, labels, and timing) inside `_raw`, although they are used only for optional plots/sanity checks and discarded after each session. It also loads `iscell` and computes/returns `Flow` and several metadata diagnostics that the decoder does not consume. At program end it concatenates all input/output and samples neural values solely to print statistics. Conversely, it deliberately avoids reading `Fneu.npy` in the default path.

ii.
```python
'_raw': {'F': F, 'Flow': Flow, 'dff': dff, 'motion': motion, 'motion_raw': motion_raw,
         'frame_index': frame_index, 'dff_binned': dff_binned,
         'motion_binned': motion_binned, 'labels': labels, 'edges': edges,
         'bin_centre_time': bin_centre_time},
...
del res
```

iii. These intermediates support `--show-processing` and extensive validation, then are explicitly released and never saved in `converted_data.pkl`. The notes justify skipping default `Fneu` loading as saving about 150 MB and substantial I/O per session.
