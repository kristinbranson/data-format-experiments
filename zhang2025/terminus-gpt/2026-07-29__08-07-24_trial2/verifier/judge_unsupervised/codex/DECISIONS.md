# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers sessions by recursively finding every `_ibl_trials.table.pqt` under `data/one_cache`, walking up to the enclosing `alf` directory, and treating its parent as a session. Within each kept session it directly loads ALF/ONE-cache files from disk instead of using the reference loaders.

ii. ```python
def find_sessions(base=Path('data/one_cache')):
    trial_tables = sorted(base.rglob('_ibl_trials.table.pqt'))
    sessions = []
    for p in trial_tables:
        sess = p.parent
        while sess.name != 'alf' and sess != sess.parent:
            sess = sess.parent
        if sess.name == 'alf':
            sessions.append(sess.parent)
    out = sorted(set(sessions))
    return out

def load_trials(session_path: Path):
    trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
    if not trial_files:
        return None
    trial_file = trial_files[-1]
    return pd.read_parquet(trial_file)
```

iii. In `CONVERSION_NOTES.md`, the agent justified this by stating the data are organized as an IBL ONE cache and that session-level ALF arrays should be read directly from that cache.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are inferred from the filesystem layout. For each processed session, the subject ID is taken from `session_path.parts[-3]`, then unique subjects are sorted into `subjects` and each session is mapped into `subject_idx`.

ii. ```python
subject = session_path.parts[-3]

subjects = sorted({p['subject'] for p in processed})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_to_idx[p['subject']] for p in processed], dtype=np.int64),
```

iii. The justification in the notes was that the IBL cache path already encodes `lab/Subjects/<subject>/<date>/<session>`, so subject membership can be recovered from path structure.

## 1-c. How are the data split into sessions?

i. Each parent directory of an `alf` folder that contains a trial table is treated as one session. The final dataset keeps one top-level session entry per such directory.

ii. ```python
while sess.name != 'alf' and sess != sess.parent:
    sess = sess.parent
if sess.name == 'alf':
    sessions.append(sess.parent)

'neural': [p['neural'] for p in processed],
'input': [p['input'] for p in processed],
'output': [p['output'] for p in processed],
```

iii. The notes explicitly say "use session as top-level unit in converted dataset" because both the cache structure and the reference workflow are session based.

## 1-d. How are the data split into trials?

i. Trials come from rows of the per-session trial table after filtering. Each remaining row becomes one trial, and the session’s neural/input/output lists are built in that filtered row order.

ii. ```python
trials = load_trials(session_path)
...
valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
trials = trials.loc[valid].reset_index(drop=True)
...
neural, edges = bin_spikes_for_trials(spike_times, spike_clusters, trials['stimOn_times'].to_numpy(), n_neurons)
...
for i in range(len(trials)):
    ...
    inputs.append(inp)
    outputs.append(out)
```

iii. The agent’s planning notes say the raw trial table naturally defines trials and should be the source for `choice`, `probabilityLeft`, and `stimOn_times`.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only trials with non-null `stimOn_times`, `probabilityLeft`, and binary `choice` values `{-1, 1}`. After spike binning it also drops any trial whose full neural matrix is all zeros, and drops sessions with fewer than two remaining trials.

ii. ```python
required = ['stimOn_times', 'choice', 'probabilityLeft']
...
valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
trials = trials.loc[valid].reset_index(drop=True)
...
keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
trials = trials.loc[keep_trial].reset_index(drop=True)
neural = [m for m, k in zip(neural, keep_trial) if k]
if len(neural) < 2:
    return None
```

iii. The stated justification was pragmatic rather than reference-based: the notes say zero-neural trials were removed to eliminate decoder verification warnings, and `choice==0` trials were removed to keep the output binary.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from per-probe spike times and cluster identities, with cluster metrics and channel-to-brain-location metadata used for curation and region indexing.

ii. ```python
metrics = pd.read_parquet(cand[-1])
chan_file = sorted(probe_dir.rglob('clusters.channels.npy'))[-1]
clu_channels = np.load(chan_file)
reg_id_files = sorted(probe_dir.rglob('channels.brainLocationIds_ccf_2017.npy'))
reg_ids = np.load(reg_id_files[-1]) if reg_id_files else None
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
```

iii. The notes say this is electrophysiology data, so the agent chose binned spikes rather than any imaging-style transform.

## 2-b. How is the `neural` data processed?

i. The agent merges probes within a session, remaps kept clusters to a contiguous neuron index, and bins spike counts into fixed 20 ms stimulus-aligned bins over `[-0.2, 1.0)` seconds. The stored neural values are raw spike counts as `float32`.

ii. ```python
remap = {old: i + offset for i, old in enumerate(kept_ids)}
mask = np.isin(spike_clusters, kept_ids)
sc = spike_clusters[mask]
st = spike_times[mask]
sc = np.array([remap[c] for c in sc], dtype=np.int64)
...
def bin_spikes_for_trials(spike_times, spike_clusters, stim_on, n_neurons, t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
    ...
    np.add.at(mat, (clu[good], tb[good]), 1)
```

iii. The notes justify this as matching the reference use of binned spike counts for electrophysiology while adapting temporal alignment to stimulus onset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent applies only a provisional cluster filter: `noise_cutoff < 20` and `label >= 1` when those columns exist. It does not implement the paper’s full well-isolated-neuron criteria, grey-matter filtering, or region/session minimums.

ii. ```python
# provisional curation using available fields; amplitude criterion may require conversion/field interpretation refinement later
keep = np.ones(len(metrics), dtype=bool)
if 'noise_cutoff' in metrics.columns:
    keep &= metrics['noise_cutoff'].to_numpy() < 20
if 'label' in metrics.columns:
    keep &= metrics['label'].to_numpy() >= 1
kept_ids = np.where(keep)[0]
```

iii. The agent’s own notes admit this is incomplete: it says paper-compatible filtering should use amplitude, noise cutoff, and refractory-period criteria, but the script only implemented a subset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial’s neural activity is aligned to `stimOn_times`, i.e. stimulus onset. Spike times are converted to time relative to each trial’s stimulus onset before binning.

ii. ```python
neural, edges = bin_spikes_for_trials(
    spike_times, spike_clusters, trials['stimOn_times'].to_numpy(), n_neurons
)

for s in stim_on:
    rel = spike_times - s
    mask = (rel >= t0) & (rel < t1)
```

iii. The notes explicitly justify this as an intentional task-specific override: align all modalities to stimulus onset because the instructions demanded it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted dataset uses 20 ms bins and stores 60 time points per trial across a `[-0.2, 1.0)` window. There is no additional temporal rebinning after this fixed binning step.

ii. ```python
def bin_spikes_for_trials(..., t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
...
'metadata': {
    'time_bin_size': bin_size * 1000.0,
    'off_start': -0.2,
    'off_end': 1.0,
}
```

iii. The notes tie the 20 ms choice to the wheel-decoding resolution discussed in the reference materials, while using one common binning scheme for the requested output format.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial-level `stimOn_times` event together with the fixed relative bin edges used for neural binning.

ii. ```python
required = ['stimOn_times', 'choice', 'probabilityLeft']
...
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

iii. The notes say the trial table directly exposes `stimOn_times`, so that event should anchor the decoder’s time-since-stimulus input.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent does not create a binary onset series. It uses the bin centers themselves, a continuous vector from approximately `-0.19` to `0.99` seconds, and repeats that same vector for every trial.

ii. ```python
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
...
inp = np.vstack([
    centers,
    np.full_like(centers, block_trial[i], dtype=np.float32),
]).astype(np.float32)
```

iii. The notes justify this by saying the decoder task asked for "Time since stimulus onset, continuous, time-varying."

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is perfectly co-registered with the neural data because it uses the exact same bin edges and bin centers as the neural matrices.

ii. ```python
neural, edges = bin_spikes_for_trials(...)
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
inp = np.vstack([centers, ...]).astype(np.float32)
```

iii. The notes describe this as using a common trial time grid for all aligned modalities.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived entirely from the trial table’s `probabilityLeft` column.

ii. ```python
block_trial = compute_trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. The notes explicitly planned to compute block-trial index from consecutive runs of constant `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent counts consecutive trials with the same `probabilityLeft`, resets the counter to `1` when `probabilityLeft` changes, and then repeats that scalar across all time bins within the trial.

ii. ```python
def compute_trial_number_in_block(prob_left):
    ...
    for i, v in enumerate(prob_left):
        if i == 0 or v != prev:
            c = 1
            prev = v
        else:
            c += 1
        out[i] = c
...
np.full_like(centers, block_trial[i], dtype=np.float32)
```

iii. The agent’s justification in the notes is that block identity is naturally visible as runs of constant `probabilityLeft`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the trial table’s `choice` column.

ii. ```python
choice = map_choice(trials['choice'].to_numpy())
```

iii. The notes identify `choice` in the trial table as the direct source variable for this decoder output.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent assumes IBL coding `choice==1` means left and `choice==-1` means right, maps them to `0` and `1` respectively, filters out `choice==0`, and repeats the resulting category across all time bins in the trial.

ii. ```python
def map_choice(choice_vals):
    arr = np.asarray(choice_vals)
    out = np.full(arr.shape, -1, dtype=np.int64)
    out[arr == 1] = 0
    out[arr == -1] = 1
    return out

valid = ... & trials['choice'].isin([-1, 1]) ...
...
np.full_like(centers, choice[i], dtype=np.int64)
```

iii. The code comment says this was based on IBL convention and should be "confirm[ed] from data later"; the notes later say invalid `choice==0` trials were removed to keep the output binary.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial table’s `probabilityLeft` column.

ii. ```python
prior = map_prior(trials['probabilityLeft'].to_numpy())
```

iii. The notes describe `probabilityLeft` as the natural raw source for the requested prior-probability output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent maps the three block priors `0.2`, `0.5`, and `0.8` to categorical labels `0`, `1`, and `2`, then repeats the category across the full trial.

ii. ```python
def map_prior(prob_left):
    m = {0.2: 0, 0.5: 1, 0.8: 2}
    return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
...
np.full_like(centers, prior[i], dtype=np.int64)
```

iii. This directly follows the decoder-task specification and is stated that way in the notes and README.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. ```python
def load_wheel(session_path: Path):
    alf = session_path / 'alf'
    posf = alf / '_ibl_wheel.position.npy'
    tsf = alf / '_ibl_wheel.timestamps.npy'
    ...
    return np.load(posf), np.load(tsf)
```

iii. The notes explicitly say wheel output should be derived from the raw wheel trace and then discretized because the task requires a categorical decoder target.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent sorts wheel timestamps, removes duplicate timestamps, computes finite-difference absolute speed `abs(dp/dt)` at midpoint times, and linearly interpolates that speed onto the common trial bins.

ii. ```python
order = np.argsort(wheel_ts)
wheel_ts = np.asarray(wheel_ts)[order]
wheel_pos = np.asarray(wheel_pos)[order]
uniq_mask = np.concatenate([[True], np.diff(wheel_ts) > 0])
wheel_ts = wheel_ts[uniq_mask]
wheel_pos = wheel_pos[uniq_mask]
dt = np.diff(wheel_ts)
dp = np.diff(wheel_pos)
speed_mid = np.abs(dp / dt).astype(np.float32)
ts_mid = ((wheel_ts[:-1] + wheel_ts[1:]) / 2).astype(np.float64)
wheel_trials = interp_to_trial_bins(speed_mid, ts_mid, trials['stimOn_times'].to_numpy(), edges)
```

iii. The notes justify the timestamp sorting/deduplication as a fix for runtime warnings, and justify wheel derivation from raw position/timestamps as the closest match to the reference wheel signal.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The agent computes session-level tertile thresholds across all aligned wheel-speed samples from that session, then assigns `0/1/2` for low/medium/high.

ii. ```python
wheel_q1, wheel_q2 = tertile_thresholds(wheel_trials)
...
def discretize_with_thresholds(x, q1, q2):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=np.int64)
    y[x > q1] = 1
    y[x > q2] = 2
    return y
...
discretize_with_thresholds(wheel_trials[i], wheel_q1, wheel_q2)
```

iii. The justification was only that the task demanded 3 categorical bins; the notes do not cite a reference source for the tertile rule.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated onto the same stimulus-aligned bin centers used for neural activity, so it is aligned bin-by-bin with the neural matrices.

ii. ```python
def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
    centers = (edges[:-1] + edges[1:]) / 2
    ...
    for s in stim_on:
        t = s + centers
        y = np.interp(t, timestamps, values)
...
wheel_trials = interp_to_trial_bins(speed_mid, ts_mid, trials['stimOn_times'].to_numpy(), edges)
```

iii. The notes say all modalities should share one common stimulus-aligned grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` and/or `rightCamera.ROIMotionEnergy.npy`, with the matching camera time arrays `_ibl_leftCamera.times.npy` and `_ibl_rightCamera.times.npy`.

ii. ```python
left_me = sorted(alf.rglob('leftCamera.ROIMotionEnergy.npy'))
right_me = sorted(alf.rglob('rightCamera.ROIMotionEnergy.npy'))
left_t = sorted(alf.rglob('_ibl_leftCamera.times.npy'))
right_t = sorted(alf.rglob('_ibl_rightCamera.times.npy'))
```

iii. The notes say whisker motion energy should come from the video-derived ROI motion-energy traces available in the ALF cache.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent loads the latest available left/right motion-energy files, interpolates each stream to the common trial bins, uses the single stream if only one exists, and otherwise averages the left and right streams.

ii. ```python
for vals, ts, side in me_streams:
    aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
if len(aligned) == 1:
    me_trials = aligned[0]
else:
    me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) for i in range(len(trials))]
```

iii. The notes only say to use ROI motion energy and "choose a consistent rule" if both sides exist; averaging both sides was the implemented rule.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, it is discretized by session-level tertiles computed over all aligned whisker-motion-energy samples in that session.

ii. ```python
me_q1, me_q2 = tertile_thresholds(me_trials)
...
discretize_with_thresholds(me_trials[i], me_q1, me_q2)
```

iii. The justification was again task-driven discretization rather than a cited reference procedure.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is linearly interpolated onto the same stimulus-aligned bin centers used for neural and wheel data.

ii. ```python
aligned.append(
    interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges)
)
```

iii. The notes repeatedly state that all modalities should share the same stimulus-aligned grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles missing or malformed data by skipping sessions missing trial tables, required columns, or spikes; dropping trials with missing `stimOn_times`, `probabilityLeft`, or invalid `choice`; sorting and deduplicating wheel timestamps; selecting the latest versioned ALF file when multiple exist; and substituting all-zero wheel or whisker traces when those streams are absent or unusable.

ii. ```python
if not trial_files:
    return None
trial_file = trial_files[-1]
...
valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
...
if wheel_pos is not None:
    ...
    uniq_mask = np.concatenate([[True], np.diff(wheel_ts) > 0])
    ...
else:
    wheel_trials = [np.zeros_like(centers) for _ in range(len(trials))]
...
else:
    me_trials = [np.zeros_like(centers) for _ in range(len(trials))]
```

iii. The notes justify the wheel timestamp cleanup as a bug fix, but the missing-stream behavior is mostly an uncited pragmatic fallback.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is repeatedly scanning the filesystem with `rglob`, loading large spike arrays and wheel/video streams from every session, and then looping over every trial to re-bin spikes and interpolate behavioral signals. The full-conversion log confirms per-session runtimes often in the tens to hundreds of seconds.

ii. ```python
trial_tables = sorted(base.rglob('_ibl_trials.table.pqt'))
...
for probe_dir in session_probe_dirs(session_path):
    ...
    spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
    spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
...
for s in stim_on:
    ...
for s in stim_on:
    t = s + centers
    y = np.interp(t, timestamps, values)
```

iii. The notes leave Step 6’s optimization section blank, but `conversion_full_out.txt` shows these session passes dominate runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the trial loop in `bin_spikes_for_trials`, the trial loop in `interp_to_trial_bins`, the scalar loop in `compute_trial_number_in_block`, the Python list-comprehension remap of spike clusters, and the final loop that constructs one `input`/`output` pair per trial.

ii. ```python
for i, v in enumerate(prob_left):
    ...

for s in stim_on:
    ...
    np.add.at(mat, (clu[good], tb[good]), 1)

for s in stim_on:
    t = s + centers
    y = np.interp(t, timestamps, values)

sc = np.array([remap[c] for c in sc], dtype=np.int64)

for i in range(len(trials)):
    ...
```

iii. This follows directly from the implementation; the agent did not document vectorization work in the notes.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly performs expensive recursive file discovery (`rglob`) inside each loader, repeatedly computes interpolation separately for every trial and modality, and repeatedly duplicates per-trial scalar outputs across every time bin.

ii. ```python
trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
...
left_me = sorted(alf.rglob('leftCamera.ROIMotionEnergy.npy'))
right_me = sorted(alf.rglob('rightCamera.ROIMotionEnergy.npy'))
...
chan_file = sorted(probe_dir.rglob('clusters.channels.npy'))[-1]
...
np.full_like(centers, block_trial[i], dtype=np.float32)
np.full_like(centers, choice[i], dtype=np.int64)
np.full_like(centers, prior[i], dtype=np.int64)
```

iii. The agent did not offer a formal justification here; this is simply how the script is written.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script constructs `session_id` in `process_session` but drops it in `build_dataset`, computes per-session `brain_regions` lists only to remap them globally later, allocates optional processing plots that are not part of the converted dataset, and keeps unused variables such as `pks` and the unused `reducer` argument.

ii. ```python
pks = sorted(probe_dir.rglob('pykilosort'))
if not pks:
    pks = [probe_dir]
...
def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
    ...
return {
    'session_id': '/'.join(session_path.parts[-4:]),
    'subject': subject,
    ...
}
...
if show_processing and len(neural) > 0:
    save_processing_plot(...)
```

iii. These behaviors are not justified in the notes; they are artifacts of the implementation rather than required downstream processing.
