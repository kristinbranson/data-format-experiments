# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six mouse IDs, enumerates every subdirectory of each mouse as a session, and loads each session's `F.npy`, `Fneu.npy`, `ops.npy`, and `motion_energy_glob.npy`. Full mode processes all 41 discovered sessions; sample mode selects two fixed sessions.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for subject in SUBJECTS:
    subject_dir = os.path.join(DATA_DIR, subject)
    sessions = get_sessions(subject_dir)
    for s in sessions:
        all_sessions.append((subject, s))

F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The notes report six subjects and 41 sessions and explain that the subject/session directory layout contains synchronized Suite2p neural files and precomputed behavior. The AI chose all available data, including sessions longer than the paper's stated 20 minutes.

## 1-b. How are the data split into subjects?

i. Subjects are the six hard-coded mouse directory names. Sessions are assigned a subject index according to first appearance in the subject-major session traversal.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
if subject not in subjects_seen:
    subjects_seen.append(subject)
subj_idx = subjects_seen.index(subject)
subject_idx_list.append(subj_idx)
```

iii. The notes identify each `jm*` folder as a mouse and verify six mice with 6–7 sessions each.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory within a subject directory is treated as one daily recording session. Each becomes one top-level element of `neural`, `input`, and `output`.

ii.
```python
def get_sessions(subject_dir):
    sessions = sorted([d for d in os.listdir(subject_dir)
                       if os.path.isdir(os.path.join(subject_dir, d))])
    return sessions

neural_all.append(neural_trials)
input_all.append(input_trials)
output_all.append(output_trials)
```

iii. The AI states that the layout is one dated session directory per daily recording and confirms 41 sessions total.

## 1-d. How are the data split into trials?

i. Each continuous session is split into non-overlapping 60-second trials after 10-frame temporal averaging. At 30 Hz this is 180 binned samples per trial; an incomplete tail is implicitly discarded.

ii.
```python
TIME_PER_BIN = BIN_SIZE / FS
BINS_PER_TRIAL = int(TRIAL_DURATION / TIME_PER_BIN)

n_trials = n_total_bins // bins_per_trial
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    trials.append(data[..., start:end])
```

iii. The notes say there is no natural trial structure and use the decoder task's requested 60-second segmentation, yielding 20 or 30 trials per session.

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial-quality filtering is performed. Only complete 180-bin trials are retained, so any incomplete session tail is omitted.

ii.
```python
n_trials = n_total_bins // bins_per_trial
for t in range(n_trials):
    trials.append(data[..., start:end])
```

iii. The notes found no trial-curation rule in the spontaneous-activity experiment and therefore applied none beyond requiring a complete 60-second segment.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `F.npy` and `Fneu.npy`; `ops.npy` supplies preprocessing parameters.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. The AI identifies `F` as raw ROI fluorescence, `Fneu` as neuropil fluorescence, and the Suite2p options as the source of the paper's default correction settings.

## 2-b. How is the `neural` data processed?

i. It subtracts neuropil using the session `neucoeff` (fallback 0.7), casts to float32, applies Suite2p maximin baseline preprocessing using options from `ops`, then averages non-overlapping groups of 10 frames.

ii.
```python
Fc = F - neucoeff * Fneu
Fc = Fc.astype(np.float32)
dFF = s2p_preprocess(Fc, baseline, win_baseline, sig_baseline, fs, device=device)
dFF_binned = bin_data(dFF, BIN_SIZE)
```

iii. The notes interpret the paper's “dF/F” as Suite2p baseline-subtracted fluorescence, not division by baseline, and cite the paper's 10-timestamp averaging for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The conversion applies no additional neuron filter and includes every row in `F.npy`.

ii.
```python
n_neurons, n_frames = F.shape
# ... all rows of dFF are binned and returned
dFF_binned = bin_data(dFF, BIN_SIZE)
```

iii. The AI inspected `iscell.npy`, found all entries already equal to one, and concluded that the supplied Track2p data was already cell-filtered and matched across days.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Neural data is anchored to session start and cut into consecutive 60-second windows. Metadata labels the alignment event `session_start`.

ii.
```python
start = t * bins_per_trial
end = start + bins_per_trial
trials.append(data[..., start:end])

'temporal_alignment_event': 'session_start',
'off_start': 0.0,
'off_end': None,
```

iii. The notes explain that these are artificial trials in continuous spontaneous activity, so session start is the relevant origin.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI averages 10 consecutive 30-Hz frames without overlap, producing 333.33-ms bins (3 Hz) for both neural and motion-energy streams. A final partial bin is discarded.

ii.
```python
BIN_SIZE = 10
FS = 30.0
TIME_PER_BIN = BIN_SIZE / FS
truncated = data[..., :n_bins * bin_size]
binned = truncated.reshape(new_shape).mean(axis=-1)
```

iii. This follows the paper's decoding method of averaging neural and behavioral traces in bins of 10 timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is generated from binned sample indices and the assumed constant 30-Hz frame rate, rather than read from `tstamps.npy`.

ii.
```python
n_bins = dFF_binned.shape[1]
time_axis = (np.arange(n_bins) + 0.5) * TIME_PER_BIN
```

iii. The notes found timestamps to be in kiloseconds but chose frame index divided by frame rate as an equivalent, simpler session-time coordinate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Each bin is represented by its center time: `(bin_index + 0.5) × 10/30` seconds. The continuous session time axis is then sliced into trials without resetting at trial boundaries.

ii.
```python
time_axis = (np.arange(n_bins) + 0.5) * TIME_PER_BIN
trial_time = time_axis[start_bin:end_bin]
input_trials.append(trial_time.reshape(1, -1))
```

iii. The AI explicitly chose bin centers because they represent the midpoint of each averaged temporal bin.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time axis has one value for each binned neural column and is sliced with the same trial start/end indices, so each input value corresponds to the same averaged neural bin.

ii.
```python
start_bin = t * BINS_PER_TRIAL
end_bin = start_bin + BINS_PER_TRIAL
trial_time = time_axis[start_bin:end_bin]
```

iii. The notes validate a converted input value against the expected center time and report matching shapes and ranges.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is derived from each session's precomputed `move_deve/motion_energy_glob.npy`. Unlike the reference, the AI does not load `interframe_int.npy` for locating dropped frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The notes describe this file as global motion energy, already computed as summed squared pixel-wise differences between successive video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The motion-energy vector is first padded with its final value or truncated to the neural frame count, then averaged over the same 10-frame bins, then discretized per session using percentile edges.

ii.
```python
me_aligned = align_me_to_neural(me, n_frames)
me_binned = bin_data(me_aligned.reshape(1, -1), BIN_SIZE).flatten()
me_discrete, bin_edges = discretize_me(me_binned, N_ME_BINS)
```

iii. The AI says synchronized acquisition makes the streams nominally one-to-one and chose tail padding to handle small camera-length mismatches.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. For each session independently, percentiles at 0, 20, 40, 60, 80, and 100 are computed on binned motion energy. `np.digitize` assigns labels 0–4, followed by clipping to that range.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(me_binned, percentiles)
me_discrete = np.digitize(me_binned, bin_edges[1:-1], right=False)
me_discrete = np.clip(me_discrete, 0, n_bins - 1)
```

iii. This implements the task's five equal-percentile categories “selected per session”; the AI verified approximately 20% occupancy per class.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI assumes frame-for-frame synchronization. If motion energy is short, it appends copies of the final value; if long, it truncates the tail. Both streams are subsequently binned identically and sliced with the same trial indices.

ii.
```python
if n_me < n_neural_frames:
    padded = np.zeros(n_neural_frames, dtype=np.float64)
    padded[:n_me] = me.astype(np.float64)
    padded[n_me:] = me[-1]
    return padded
else:
    return me[:n_neural_frames].astype(np.float64)
```

iii. The notes justify this using microscope-triggered video acquisition and describe the mismatches as minor camera-trigger drops. They do not justify why all missing samples should be placed at the tail rather than at detected drop locations.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Motion-energy length mismatches are silently repaired by repeating the final value when short or truncating when long. Incomplete temporal bins and incomplete 60-second trials are discarded by integer division. There is no explicit NaN handling or assertion after alignment.

ii.
```python
padded[n_me:] = me[-1]
return padded
# ...
n_bins = n_frames // bin_size
truncated = data[..., :n_bins * bin_size]
n_trials = n_total_bins // bins_per_trial
```

iii. The AI chose padding because it viewed the discrepancies as small dropped-camera-frame errors and reported mismatches of up to 116 frames. It preferred retaining the full neural session.

## 6-a. What are the most time-consuming steps of the code?

i. The code times loading, Suite2p baseline preprocessing, binning, and total session processing. Its documentation identifies Suite2p dF/F preprocessing as the main computational cost; the full conversion took 31.7 seconds.

ii.
```python
t1 = time.time()
dFF = compute_dff(F, Fneu, ops)
t_dff = time.time() - t1
print(f"... load={t_load:.1f}s dff={t_dff:.1f}s bin={t_bin:.2f}s total={t_total:.1f}s")
```

iii. The notes attribute the cost to Suite2p preprocessing over all neurons and frames and observe that it runs on CPU in this implementation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Frame binning is already vectorized. Trial splitting and per-trial input/output assembly use Python loops but could be reshaped into trial blocks in one operation. Session traversal necessarily handles files separately. The agent did not identify these trial loops as an issue.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    trials.append(data[..., start:end])

for t in range(n_trials):
    trial_time = time_axis[start_bin:end_bin]
    input_trials.append(trial_time.reshape(1, -1))
```

iii. The notes emphasize vectorized `reshape(...).mean(...)` binning and claim no important unnecessary I/O; they do not discuss vectorizing the small trial loops.

## 6-c. What processing does the code repeat multiple times?

i. It traverses the trial boundaries once in `split_into_trials` for neural data and again in `process_session` to slice time and motion energy. It also performs the same full preprocessing independently for every session, as required.

ii.
```python
neural_trials = split_into_trials(dFF_binned, BINS_PER_TRIAL)
for t in range(n_trials):
    start_bin = t * BINS_PER_TRIAL
    end_bin = start_bin + BINS_PER_TRIAL
    trial_time = time_axis[start_bin:end_bin]
    trial_me = me_discrete[start_bin:end_bin]
```

iii. The AI does not explicitly document repeated processing. The repeated trial-index calculation is visible in the implementation but has negligible cost.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal mode, `process_session` returns `me_binned` and `bin_edges`, and `main` receives them but never uses them. It also accepts an unused `session_idx_global`. In `--show-processing` mode only, it additionally z-scores neural data and creates large diagnostic plots, which are intentionally diagnostic rather than decoder inputs.

ii.
```python
return neural_trials, input_trials, output_trials, n_neurons, me_binned, bin_edges

neural_trials, input_trials, output_trials, n_neurons, me_binned, bin_edges = \
    process_session(...)
```

iii. The notes claim no unnecessary I/O and present plots as sanity checks. They do not mention that the returned binned motion energy and percentile edges are discarded by `main`.
