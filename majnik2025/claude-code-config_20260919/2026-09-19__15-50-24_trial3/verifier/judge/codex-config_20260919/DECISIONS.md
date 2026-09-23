# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter enumerates every sorted session directory under six hard-coded subject IDs. For each session it loads `F.npy`, `ops.npy`, `motion_energy_glob.npy`, and, when camera frames are missing, `tstamps.npy`. Full mode processes all 41 sessions; sample mode selects two named sessions.

ii.
```python
SUBJECT_INFO = {
    'jm031': ('A', 7), 'jm032': ('B', 7), 'jm038': ('C', 8),
    'jm039': ('D', 8), 'jm040': ('E', 9), 'jm046': ('F', 8),
}

def list_sessions(data_root=DATA_ROOT):
    sessions = []
    for subject in sorted(SUBJECT_INFO.keys()):
        subj_dir = os.path.join(data_root, subject)
        for sess_dir in sorted(f.path for f in os.scandir(subj_dir) if f.is_dir()):
            sessions.append((subject, os.path.basename(sess_dir), sess_dir))
    return sessions

def load_traces(session_dir):
    return np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
```

iii. The agent states that the release contains six mice and 41 daily sessions, with one imaging plane per session. It chose the explicit subject table to retain paper mouse letters and ages, and reports that full conversion found all 41 sessions.

## 1-b. How are the data split into subjects?

i. A subject is a named mouse directory in `SUBJECT_INFO`; sessions receive the index of that subject in the sorted six-subject list.

ii.
```python
subject_list = sorted(SUBJECT_INFO.keys())
data['subject_idx'].append(subject_list.index(subject))
data['subjects'] = subject_list
```

iii. The notes map `jm031` through `jm046` to paper mice A–F and validate the mapping using the ground-truth files and Figure 5B.

## 1-c. How are the data split into sessions?

i. Every immediate subdirectory of a subject directory is one session, sorted by its path/name. One target-format session is appended per daily recording.

ii.
```python
for sess_dir in sorted(f.path for f in os.scandir(subj_dir) if f.is_dir()):
    sessions.append((subject, os.path.basename(sess_dir), sess_dir))

for (subject, sess_name, sess_dir) in sessions:
    res = convert_session(sess_dir, keep_neurons=keep_masks[subject])
    data['neural'].append(res['neural'])
```

iii. The agent describes the directories as consecutive daily recordings and reports 7, 7, 7, 7, 6, and 7 sessions for the six mice.

## 1-d. How are the data split into trials?

i. Each continuous session is cut from its first frame into consecutive, non-overlapping 60-second trials. At 3 Hz after binning, each trial contains 180 bins; an incomplete tail would be dropped.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_LEN_S / BIN_SIZE_S))
n_trials = n_bins // BINS_PER_TRIAL
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    neural.append(np.ascontiguousarray(dff_binned[:, sl]))
    inputs.append(bin_centre_s[sl][None, :].astype(np.float32))
    outputs.append(me_labels[sl][None, :].astype(np.int64))
```

iii. The decoder task explicitly requires 60-second trials. The agent notes that all source lengths are exact multiples of 60 seconds, so the full dataset loses no tail data.

## 1-e. How are trials filtered based on quality controls?

i. No sessions or complete trials are filtered. Only a hypothetical incomplete final trial is omitted by integer division; none occur in this dataset.

ii.
```python
n_trials = n_bins // BINS_PER_TRIAL
n_dropped_bins = n_bins - n_trials * BINS_PER_TRIAL
```

iii. The notes say every session has at least 20 trials, neural data are finite, and behavioral coverage is at least 99.6%, so no trial exclusion was warranted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived only from Suite2p `plane0/F.npy`. Although `f_processing` accepts `Fneu`, the converter never loads or passes `Fneu.npy` and sets the neuropil coefficient to zero.

ii.
```python
NEUCOEFF = 0.0

def load_traces(session_dir):
    return np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))

dff, baseline = f_processing(F, fs=fs, return_baseline=True)
```

iii. The agent argues that the Track2p GUI calls the reference `F_processing` without `neucoeff`, whose default is zero, and therefore no neuropil subtraction best follows that code.

## 2-b. How is the `neural` data processed?

i. `F` is cast to float32; a maximin baseline is computed by temporal Gaussian smoothing (sigma 10 frames), then 60-second minimum and maximum filters. The baseline is subtracted, and the result is averaged in non-overlapping 10-frame bins.

ii.
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
dff = Fc - Flow

dff_binned = bin_mean(dff).astype(np.float32)
```

iii. The agent calls this a verbatim port of `DataManagement.F_processing` and interprets the paper's “baseline corrected fluorescence” as subtraction rather than division. Ten-frame averaging comes directly from the paper's decoding methods.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent assumes the released `F.npy` already contains ROIs with Suite2p cell probability above 0.5 that Track2p tracked across all days. It additionally removes, from every day of a mouse, any ROI with a constant/all-zero trace on at least one day (8 of 2998 ROIs).

ii.
```python
def find_bad_neurons(subject_sessions):
    bad = None
    for _, _, sess_dir in subject_sessions:
        F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
        b = np.asarray(F).std(axis=1) == 0
        bad = b if bad is None else (bad | b)
    return bad

F = F[keep_neurons]
```

iii. The notes say these ROIs fell outside the field of view and carry no signal. Removing each affected ROI from all sessions preserves the matched cross-day population.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Trials begin at consecutive 60-second boundaries from session start, and metadata defines the alignment event as the start of each artificial trial, with offsets 0 to 60 seconds.

ii.
```python
'temporal_alignment_event': (
    'Start of the 60 s trial. Recordings are continuous spontaneous activity '
    'with no task events...'),
'off_start': 0.0,
'off_end': TRIAL_LEN_S,
```

iii. The agent explains that spontaneous continuous recordings have no stimulus or task event, leaving trial/session start as the only meaningful alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30 Hz samples are averaged without overlap, producing 3 Hz data and 333.333 ms bins for both neural activity and motion energy.

ii.
```python
FS = 30.0
BIN_FRAMES = 10
BIN_SIZE_MS = BIN_FRAMES / FS * 1000.0

v = x[..., :n * bin_frames].reshape(*x.shape[:-1], n, bin_frames)
return v.mean(axis=-1)
```

iii. The choice directly follows the paper's statement that decoding analyses average neural and behavior traces in bins of 10 timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is calculated from the global imaging-bin index and the sampling rate from `ops.npy`; it is not read from a timestamp stream. Values denote bin centers.

ii.
```python
fs = load_fs(session_dir)
bin_centre_s = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
```

iii. Since all sessions have a verified constant 30 Hz rate, the agent considers index-derived time equivalent and chooses bin centers as the representative times of averages.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For bin `j`, the converter computes `(10*j + 4.5)/30` seconds, casts to float32, and retains the global session time rather than restarting at each trial.

ii.
```python
bin_centre_s = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
inputs.append(bin_centre_s[sl][None, :].astype(np.float32))
```

iii. The notes justify 4.5 frames as the center of the inclusive 10-frame block and explicitly verify continuity across trial boundaries.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time and binned neural activity share the same global bin indices and are sliced with exactly the same trial slice.

ii.
```python
sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
neural.append(np.ascontiguousarray(dff_binned[:, sl]))
inputs.append(bin_centre_s[sl][None, :].astype(np.float32))
```

iii. The agent's checks concatenate trial inputs, confirm uniform one-third-second increments, and confirm continuity at boundaries.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output derives from `move_deve/motion_energy_glob.npy`. When its length differs from imaging data, `move_deve/tstamps.npy` determines which imaging-frame slots are missing.

ii.
```python
me_raw = np.load(os.path.join(session_dir, 'move_deve',
                              'motion_energy_glob.npy')).astype(np.float64)
ts = np.load(os.path.join(session_dir, 'move_deve', 'tstamps.npy'))
```

iii. The agent cites hardware triggering and the dataset README: camera frame indices normally match imaging indices, while timestamp gaps reveal dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Raw precomputed motion energy is mapped onto the imaging grid. Missing camera frames and the artificial first value are marked NaN. Non-overlapping 10-frame `nanmean` produces the decoder trace; a wholly missing bin would be linearly interpolated. The trace is then quantized per session.

ii.
```python
me = np.full(n_frames, np.nan, dtype=np.float64)
me[frame_idx] = me_raw
me[0] = np.nan
me_binned = bin_mean(me_frames, nan_aware=True)
```

iii. The agent reasons that dropped values should be excluded rather than invented, that `motion_energy_glob[0]` is necessarily an artifact, and that no 10-frame bin in the actual data is entirely missing.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The 20th, 40th, 60th, and 80th percentiles of each session's binned motion energy define five labels 0–4. Values equal to an edge enter the higher category.

ii.
```python
edges = np.percentile(x, np.arange(1, n_quantiles) * (100.0 / n_quantiles))
labels = np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. This follows the required “five equal-percentile bins, selected per session” and normalizes session-specific motion-energy scale.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera timestamps reconstruct each observed camera frame's imaging-frame index. Motion energy is placed on that grid, missing slots become NaN, and neural and behavior streams are binned from the same frame boundaries and sliced identically into trials.

ii.
```python
n_steps = np.round(np.diff(ts) / np.median(np.diff(ts))).astype(int)
frame_idx = np.concatenate([[0], np.cumsum(n_steps)])
me[frame_idx] = me_raw

assert me_binned.shape[0] == n_bins
outputs.append(me_labels[sl][None, :].astype(np.int64))
```

iii. The notes report verifying that reconstructed camera indices span exactly the imaging length and that neural–motion cross-correlation peaks only 0.3–1.3 seconds apart, consistent with calcium kinetics.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Extra camera samples would be truncated; dropped camera frames are reconstructed as missing positions using timestamps and excluded from bin means; fully missing bins would be interpolated. The first undefined motion-energy sample is discarded. Unexpected reconstructed length raises an error. Constant ROIs are removed across all days. Partial final bins/trials would be dropped.

ii.
```python
if frame_idx[-1] + 1 != n_frames:
    raise ValueError(...)
me = np.full(n_frames, np.nan, dtype=np.float64)
me[frame_idx] = me_raw
me[0] = np.nan

if np.any(np.isnan(me_binned)):
    me_binned[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), me_binned[good])
```

iii. The agent emphasizes preserving temporal alignment without fabricating isolated frame values, failing loudly on irreconcilable lengths, and maintaining matched neuron identities across days.

## 6-a. What are the most time-consuming steps of the code?

i. The maximin baseline filters dominate computation; loading the large fluorescence arrays is the main I/O cost. The full conversion measured about 38 seconds.

ii.
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. The notes identify these full-session, all-neuron filters as the heavy operations, while behavior processing takes less than 0.02 seconds per session.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Binning and filtering are already vectorized. The per-trial assembly loop could be replaced by reshape/transpose operations, although it mainly creates the required list structure. Plotting loops over example neurons and trial boundaries, and session/subject enumeration remains in Python.

ii.
```python
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    neural.append(np.ascontiguousarray(dff_binned[:, sl]))
    inputs.append(bin_centre_s[sl][None, :].astype(np.float32))
    outputs.append(me_labels[sl][None, :].astype(np.int64))
```

iii. The agent says no material inefficiency remains: binning is one reshape plus mean and trials are cheap slices rather than recomputation.

## 6-c. What processing does the code repeat multiple times?

i. Each subject's `F.npy` files are first scanned by `find_bad_neurons` and then loaded again during conversion. Lists of a subject's sessions are also repeatedly reconstructed to find masks and day indices. Sanity checks later traverse every trial again.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
# later, in convert_session:
F = load_traces(session_dir)

subj_sessions = [s[1] for s in all_sessions if s[0] == subject]
```

iii. The agent acknowledges the preliminary scan but justifies memory mapping it; its notes otherwise characterize the conversion as I/O/filter bound and report no material repeated processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Every session computes and stores the full baseline and a `_raw` bundle containing raw fluorescence, baseline, full-resolution dF/F, binned arrays, motion traces, labels, and time. This bundle is used only for optional plots and is popped before serialization. Numerous diagnostics and metadata are also computed but are not decoder features.

ii.
```python
dff, baseline = f_processing(F, fs=fs, return_baseline=True)
out['_raw'] = {'F': F, 'baseline': baseline, 'dff': dff,
               'dff_binned': dff_binned, 'me_frames': me_frames,
               'me_binned': me_binned, 'me_labels': me_labels,
               'bin_centre_s': bin_centre_s}
...
res.pop('_raw')
```

iii. The agent says `_raw` is retained only for `--show-processing`, but it is built even when plotting is disabled. The notes do not identify this as an inefficiency; they emphasize that processing plots and diagnostics support validation.
