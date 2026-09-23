# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API. It reads directly from the mounted filesystem cache at `/mnt/dataset/one_cache`, uses `/app/code/code_zhang2025/data/bwm_release.csv` to choose candidate sessions, reconstructs each session directory from lab/subject/date/session number metadata, then loads ALF files by path. Trials come from `_ibl_trials.table.pqt`; wheel and camera data come from session-level `.npy` files; spike and cluster data come from each probe directory. It also chooses the newest revisioned file on disk with `choose_file()`.

ii.
```python
SOURCE = Path('/mnt/dataset/one_cache')
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')

def session_dir(row: pd.Series) -> Path:
    return (SOURCE / str(row.lab) / 'Subjects' / str(row.subject) /
            str(row.date) / f'{int(row.session_number):03d}')

def load_trials(sdir: Path):
    p = choose_file(sdir / 'alf', '_ibl_trials.table.pqt')
    return pd.read_parquet(p), p
```

```python
wt = np.load(choose_file(alf, '_ibl_wheel.timestamps.npy'), mmap_mode='r')
wp = np.load(choose_file(alf, '_ibl_wheel.position.npy'), mmap_mode='r')
...
ct = np.load(choose_file(alf, f'_ibl_{side}Camera.times.npy'), mmap_mode='r')
cv = np.load(choose_file(alf, f'{side}Camera.ROIMotionEnergy.npy'), mmap_mode='r')
```

```python
metrics = pd.read_parquet(metrics_p)
ch = np.load(parent / 'clusters.channels.npy', mmap_mode='r')
st = np.load(parent / 'spikes.times.npy', mmap_mode='r')
sc = np.load(parent / 'spikes.clusters.npy', mmap_mode='r')
```

iii. In `CONVERSION_NOTES.md`, the AI says it intentionally “reads the staged read-only ONE cache directly from `/mnt/dataset/one_cache` without network access” and treats this as an “offline implementation” of the reference logic. The notes justify this as necessary because the staged environment already contains the cache and network access was avoided.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. The AI first gets the unique subjects, permutes them with `RandomState(42)`, then chooses one first-listed session per subject. In the final dataset, `subjects` is the sorted set of retained session subjects and `subject_idx` maps each retained session to that list.

ii.
```python
subjects = np.unique(bwm.subject)
rng = np.random.RandomState(SEED)
selected = rng.choice(subjects, len(subjects), replace=False)
by_subject = bwm.groupby('subject', sort=True)
for subject in selected:
    first_idx = by_subject.groups[subject][0]
    row = bwm.loc[first_idx]
    out.append({'eid': str(row.eid), 'subject': str(subject), 'row': row})
```

```python
subjects = sorted(set(s['info']['subject'] for s in sessions))
subject_map = {v:i for i,v in enumerate(subjects)}
'subject_idx':np.array([subject_map[s['info']['subject']] for s in sessions],dtype=np.int32),
```

iii. The AI’s notes say it corrected its scope after inspecting the reference call site and decided to match `0_data_caching.py`: “one first-listed EID per subject; subjects permuted with NumPy RandomState seed 42.”

## 1-c. How are the data split into sessions?

i. The AI does not keep all sessions. It defines a session as the single first-listed `eid` for each subject in `bwm_release.csv`, after seeded subject permutation. Full mode iterates over all such candidate subject-level sessions; sample mode takes the first two successfully converted sessions from the same ordering.

ii.
```python
def select_reference_sessions(sample: bool) -> list[dict]:
    bwm = pd.read_csv(BWM_CSV, index_col=0)
    subjects = np.unique(bwm.subject)
    rng = np.random.RandomState(SEED)
    selected = rng.choice(subjects, len(subjects), replace=False)
    by_subject = bwm.groupby('subject', sort=True)
    out = []
    for subject in selected:
        first_idx = by_subject.groups[subject][0]
        row = bwm.loc[first_idx]
        out.append({'eid': str(row.eid), 'subject': str(subject), 'row': row})
    return out
```

```python
for i, item in enumerate(candidates):
    if sample and len(sessions) >= 2:
        break
    ...
    z = process_session(item, br)
    sessions.append(z)
```

iii. The notes explicitly justify this as matching the reference caching script rather than the full release: “one selected session per subject” and “sample mode takes the first two eligible sessions under the same deterministic full-session ordering.”

## 1-d. How are the data split into trials?

i. Trials come from one row per row of the session trial parquet. After filtering, the AI uses the retained row indices `trial_idx` and iterates over the filtered `selected = trials.iloc[trial_idx]` table to create one neural matrix, one input array, and one output array per retained trial.

ii.
```python
trials, trial_path = load_trials(sdir)
...
mask, reasons = trial_mask(trials, coverage)
trial_idx = np.flatnonzero(mask)
...
selected = trials.iloc[trial_idx]
neural = bin_spikes(probes, selected.stimOn_times.to_numpy(float))
```

```python
for raw_i, (_, tr) in zip(trial_idx, selected.iterrows()):
    inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)]).astype(np.float32)
    ...
    inputs.append(inp)
    scalar.append((choice, prior))
```

iii. The AI does not give a separate justification here beyond treating the parquet as the native trial table. Its notes describe trial rows as the source of per-trial variables and say the script applies trial QC to that table before conversion.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if several conditions hold at once: required columns exist and are finite, `firstMovement_times - stimOn_times` is between 0.08 and 2.0 s, `feedback_times >= stimOn_times`, trial duration from `intervals_0`/`intervals_1` is positive and at most 10 s, `choice` is either `-1` or `1`, `probabilityLeft` is one of `0.2/0.5/0.8`, and the full wheel/whisker aligned window is covered. Sessions with fewer than two valid trials are skipped.

ii.
```python
required = ['stimOn_times', 'firstMovement_times', 'feedback_times', 'choice',
            'probabilityLeft', 'intervals_0', 'intervals_1']
...
valid_prior = np.isclose(prior[:, None], [0.2, 0.5, 0.8], atol=1e-6).any(1)
mask = (finite & behavior_coverage & (rt >= 0.08) & (rt <= 2.0) &
        (feedback >= stim) & (duration > 0) & (duration <= 10.0) &
        np.isin(choice, [-1.0, 1.0]) & valid_prior)
```

```python
coverage = ((targets[:, 0] >= time_i[0]) & (targets[:, -1] <= time_i[-1]) &
            (targets[:, 0] >= ct[0]) & (targets[:, -1] <= ct[-1]))
...
if len(trial_idx) < 2:
    raise ValueError(f'only {len(trial_idx)} valid trials')
```

iii. The notes justify this as combining paper and code curation: the “reference trial QC” plus a max-duration/order mask from `load_trials_and_mask`, the reaction-time bound from the paper, and mandatory full wheel/whisker coverage because both are required decoder outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from per-probe spike times and spike cluster assignments, with cluster metrics and anatomy used to decide which clusters are kept. Concretely, the AI loads `spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
metrics = pd.read_parquet(metrics_p)
ch = np.load(parent / 'clusters.channels.npy', mmap_mode='r')
st = np.load(parent / 'spikes.times.npy', mmap_mode='r')
sc = np.load(parent / 'spikes.clusters.npy', mmap_mode='r')
...
channel_ids = np.load(parent / 'channels.brainLocationIds_ccf_2017.npy', mmap_mode='r')
```

iii. The notes say the converter is for electrophysiology, not imaging, so the relevant raw neural sources are spike times, cluster IDs, and cluster/anatomy QC metadata.

## 2-b. How is the `neural` data processed?

i. For each retained trial, the AI bins spikes from all retained probes into 100 bins of width 20 ms over `[-0.5, 1.5)` seconds relative to stimulus onset. Probes are merged by giving each probe a neuron offset on the session-wide neuron axis. The resulting per-trial matrices are float32 spike counts; the AI does not divide by `BIN`, so it keeps counts rather than Hz firing rates.

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
N_BINS = 100
EDGES_REL = np.linspace(OFF_START, OFF_END, N_BINS + 1)
```

```python
for p, off in zip(probes, offsets):
    ...
    tb = np.floor((t - abs_edges[0]) / BIN + 1e-10).astype(np.int32)
    ok = (local >= 0) & (tb >= 0) & (tb < N_BINS)
    flat = (local[ok].astype(np.int64) * N_BINS + tb[ok])
    counts = np.bincount(flat, minlength=p['n_good'] * N_BINS)
    mat[off:off+p['n_good']] += counts.reshape(p['n_good'], N_BINS).astype(np.float32)
```

iii. The AI’s notes justify this as preserving the “event-aligned spike-count binning” seen in the reference utilities and explicitly describe the final matrices as “stimulus-aligned spike counts.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI retains only clusters with `label >= 1.0`, valid channel anatomy, and Beryl region names not in `root`, `void`, `nan`, `none`, or empty string. Only spikes from those clusters survive through the lookup table.

ii.
```python
labels = metrics.label.to_numpy(float)
channels = np.asarray(ch, int)
anatomical_ok = (channels >= 0) & (channels < len(channel_ids))
...
regions = np.asarray(br.id2acronym(atlas_ids, mapping='Beryl')).astype(str)
bad_names = np.isin(np.char.lower(regions), ['root', 'void', 'nan', 'none', ''])
good = (labels >= 1.0) & anatomical_ok & (~bad_names)
```

```python
lookup = np.full(..., -1, dtype=np.int32)
lookup[good_cids] = np.arange(len(good_cids), dtype=np.int32)
...
local[in_lookup] = p['lookup'][c[in_lookup]]
ok = (local >= 0) & (tb >= 0) & (tb < N_BINS)
```

iii. The notes justify this as keeping “good grey-matter units only.” They say `label >= 1` is the operational encoding of the paper’s unit QC and that invalid/non-grey labels should be excluded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `stimOn_times`. For each trial, the AI forms absolute bin edges `stim + EDGES_REL`, slices spikes that fall into that absolute time window, and bins them relative to the trial’s stimulus onset.

ii.
```python
abs_edges = stim + EDGES_REL
lo = int(np.searchsorted(ts, abs_edges[0], side='left'))
hi = int(np.searchsorted(ts, abs_edges[-1], side='left'))
...
tb = np.floor((t - abs_edges[0]) / BIN + 1e-10).astype(np.int32)
```

```python
neural = bin_spikes(probes, selected.stimOn_times.to_numpy(float))
```

iii. The notes repeatedly describe the dataset as “stimulus-aligned” and say the exact cached reference parameters are `align_time=stimOn_times` with a `(-0.5, 1.5)` window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 20 ms bins and exactly 100 bins per trial over a 2 s window. There is no temporal rebinning after this; neural data is binned once into that fixed grid.

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
N_BINS = 100
EDGES_REL = np.linspace(OFF_START, OFF_END, N_BINS + 1)
```

```python
'metadata':{
    ...
    'time_bin_size':20.0,
    ...
}
```

iii. The AI’s notes say it matched the reference caching parameters exactly: “100 bins over `[-0.5, +1.5)` s relative to stimulus onset.”

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI’s time input is not read from a dedicated raw column. It is a fixed relative time grid, `TIME_REL`, defined from the trial-alignment window. `stimOn_times` is used to align other modalities to the same grid, but the input row itself is just the common relative-time vector.

ii.
```python
EDGES_REL = np.linspace(OFF_START, OFF_END, N_BINS + 1)
# Reference behavior interpolation predicts the value at each spike bin's right edge.
TIME_REL = EDGES_REL[1:].astype(np.float32)
```

```python
inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)]).astype(np.float32)
```

iii. In the notes, the AI explicitly chose “behavior sampling times/right bin edges” for this input and treated it as a task-defined decoder input rather than something loaded directly from the trial table.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI uses the right edges of the 20 ms bins, not the bin centers. It defines `TIME_REL` as `EDGES_REL[1:]`, yielding values from `-0.48` to `1.50`, and repeats that same vector for every trial.

ii.
```python
EDGES_REL = np.linspace(OFF_START, OFF_END, N_BINS + 1)
TIME_REL = EDGES_REL[1:].astype(np.float32)
```

```python
for raw_i, (_, tr) in zip(trial_idx, selected.iterrows()):
    inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)]).astype(np.float32)
```

iii. The justification is explicit in both code comments and notes: “Reference behavior interpolation predicts the value at each spike bin’s right edge,” and the sample/full notes report the “exact right edges `−0.48` to `+1.50` s.”

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The time input is aligned to the same stimulus-aligned 100-bin grid as the neural data, but it encodes the right edge of each neural bin rather than the bin center.

ii.
```python
TIME_REL = EDGES_REL[1:].astype(np.float32)
...
abs_edges = stim + EDGES_REL
...
tb = np.floor((t - abs_edges[0]) / BIN + 1e-10).astype(np.int32)
```

iii. The AI justified this by trying to make the time input match where it sampled the continuous behavior streams, which it also evaluated at `stim + TIME_REL`.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. The AI does not derive this input from block structure or `probabilityLeft`. Instead, it uses the original raw trial index within the session, taken from the retained row index `raw_i`.

ii.
```python
trial_idx = np.flatnonzero(mask)
...
for raw_i, (_, tr) in zip(trial_idx, selected.iterrows()):
    inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)]).astype(np.float32)
```

```python
'input_names':['time_since_stimulus_onset_s','trial_number_in_session'],
```

iii. The notes explicitly justify this as preserving “original zero-based raw trial index” so that QC gaps are retained, even though the requested variable was trial number in block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI repeats the retained trial’s original session-wide row index across all 100 time bins. It does not reconstruct blocks from `probabilityLeft`, and it does not count position within block.

ii.
```python
for raw_i, (_, tr) in zip(trial_idx, selected.iterrows()):
    inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)]).astype(np.float32)
```

iii. The notes justify this as “original rather than compressed retained-trial order preserves trial position and block structure,” but there is no code that computes true within-block counts.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the session trial table.

ii.
```python
choice = 0 if float(tr.choice) == -1 else 1
...
scalar.append((choice, prior))
```

iii. The notes identify `trials.choice` as the source and discuss mapping it to the decoder’s left/right categories.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After filtering out non-response trials in `trial_mask()`, the AI maps `choice == -1` to decoder class `0` and everything else that survives (effectively `+1`) to class `1`. It later repeats that scalar across all 100 time bins in the output array.

ii.
```python
mask = ... np.isin(choice, [-1.0, 1.0]) & valid_prior
```

```python
choice = 0 if float(tr.choice) == -1 else 1
...
out[0]=choice
```

iii. The notes explicitly defend this sign convention, stating that native `choice=-1` is the left choice and `+1` is right, and that the repeated-over-time representation is required by the mixed temporal/scalar output format.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column of the session trial table.

ii.
```python
prior = int(np.argmin(np.abs(prior_levels - float(tr.probabilityLeft))))
...
scalar.append((choice, prior))
```

iii. The notes consistently identify `trials.probabilityLeft` as the source of block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI converts each trial’s `probabilityLeft` to the nearest entry in `[0.2, 0.5, 0.8]`, giving categories `0, 1, 2`, and later repeats that scalar across all 100 time bins in the output tensor.

ii.
```python
prior_levels = np.array([0.2, 0.5, 0.8])
...
prior = int(np.argmin(np.abs(prior_levels - float(tr.probabilityLeft))))
```

```python
out[1]=prior
```

iii. The notes justify this as the task-required categorical mapping `0.2→0`, `0.5→1`, `0.8→2`.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
wt = np.load(choose_file(alf, '_ibl_wheel.timestamps.npy'), mmap_mode='r')
wp = np.load(choose_file(alf, '_ibl_wheel.position.npy'), mmap_mode='r')
```

iii. The notes say the converter follows the IBL/reference wheel pipeline rather than deriving wheel speed from any already-processed field.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI filters nonfinite wheel samples, removes duplicate timestamps, linearly interpolates position onto a 1 kHz grid with `wheel_utils.interpolate_position`, computes filtered velocity with `wheel_utils.velocity_filtered`, takes absolute value to get speed, and then linearly interpolates each retained trial’s trace onto the aligned decoder grid.

ii.
```python
wf = np.isfinite(wt) & np.isfinite(wp)
wt0, wp0 = np.asarray(wt[wf]), np.asarray(wp[wf])
_, unique_idx = np.unique(wt0, return_index=True)
unique_idx.sort(); wt0, wp0 = wt0[unique_idx], wp0[unique_idx]
...
pos_i, time_i = wheel_utils.interpolate_position(wt0, wp0, freq=1000, kind='linear')
vel_i, _ = wheel_utils.velocity_filtered(pos_i, fs=1000)
speed_i = np.abs(vel_i)
```

```python
targets = stim[:, None] + TIME_REL[None, :]
...
wheel_vals[i] = np.interp(targets[i], time_i, speed_i).astype(np.float32)
```

iii. The notes justify this as using the “exact IBL reference primitives” for wheel interpolation and velocity, with a robustness fix for rare duplicate timestamps.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes tertile thresholds globally over all retained aligned wheel samples from all converted sessions, then bins each per-trial wheel trace with `np.searchsorted(..., side='right')` into `0/1/2`.

ii.
```python
wheel_all = np.concatenate([s['wheel'].ravel() for s in sessions])
wthr = np.quantile(wheel_all, [1/3, 2/3]).astype(float)
```

```python
out[2]=np.searchsorted(wthr,w,side='right').astype(np.int8)
```

iii. The notes explicitly justify this as “pooled empirical tertiles computed from all valid finite samples after alignment,” chosen to give deterministic, approximately balanced classes.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is sampled at `stimOn_times + TIME_REL` for each trial, using the same 100-bin stimulus-aligned grid as the neural data. As above, `TIME_REL` is the right edge of each 20 ms neural bin.

ii.
```python
targets = stim[:, None] + TIME_REL[None, :]
...
wheel_vals[i] = np.interp(targets[i], time_i, speed_i).astype(np.float32)
```

```python
abs_edges = stim + EDGES_REL
...
tb = np.floor((t - abs_edges[0]) / BIN + 1e-10).astype(np.int32)
```

iii. The AI’s justification is that all modalities should share one common stimulus-aligned grid, and its notes specifically say “behavior sampled at right bin edge.”

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from camera frame times and motion-energy values, preferring the left camera and falling back to the right camera when needed.

ii.
```python
for side in ('left', 'right'):
    try:
        ct = np.load(choose_file(alf, f'_ibl_{side}Camera.times.npy'), mmap_mode='r')
        cv = np.load(choose_file(alf, f'{side}Camera.ROIMotionEnergy.npy'), mmap_mode='r')
        ...
        camera = side
        break
```

iii. The notes justify this as preserving the reference “left-first/right-fallback” whisker camera rule.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI filters nonfinite values, removes duplicate timestamps, keeps the raw motion-energy values without further normalization or filtering, and linearly interpolates each retained trial onto the common aligned time grid.

ii.
```python
cf = np.isfinite(ct) & np.isfinite(cv)
ct0, cv0 = np.asarray(ct[cf]), np.asarray(cv[cf])
_, ci = np.unique(ct0, return_index=True)
ci.sort(); ct0, cv0 = ct0[ci], cv0[ci]
...
whisk_vals[i] = np.interp(targets[i], ct, cv).astype(np.float32)
```

iii. The notes justify this as using the released whisker motion-energy trace “as is,” with only finite/deduplication robustness and interpolation to the shared trial grid.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is discretized with global tertiles computed across all retained aligned samples from all sessions, then categorized via `np.searchsorted(..., side='right')`.

ii.
```python
whisk_all = np.concatenate([s['whisk'].ravel() for s in sessions])
qthr = np.quantile(whisk_all, [1/3, 2/3]).astype(float)
```

```python
out[3]=np.searchsorted(qthr,q,side='right').astype(np.int8)
```

iii. The notes justify this with the same “pooled empirical tertiles” argument used for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is evaluated at `stimOn_times + TIME_REL`, so it shares the same stimulus-aligned 100-bin grid as the neural data, with right-edge sampling.

ii.
```python
targets = stim[:, None] + TIME_REL[None, :]
...
whisk_vals[i] = np.interp(targets[i], ct, cv).astype(np.float32)
```

```python
abs_edges = stim + EDGES_REL
```

iii. The AI’s notes justify this as part of the single shared alignment grid for all modalities.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or inconsistent data by dropping or skipping it rather than imputing it, with one explicit repair for duplicate timestamps. It chooses the newest revisioned file on disk, filters nonfinite wheel/camera samples, deduplicates repeated timestamps, skips sessions with no complete whisker stream, rejects sessions with no good neurons or fewer than two valid trials, ignores missing probe directories, and raises on missing required trial columns or inconsistent array lengths.

ii.
```python
def choose_file(root: Path, basename: str, parent_contains: str | None = None) -> Path:
    ...
    return sorted(candidates, key=key)[-1]
```

```python
wf = np.isfinite(wt) & np.isfinite(wp)
...
_, unique_idx = np.unique(wt0, return_index=True)
...
if camera is None:
    raise ValueError('no complete left or right whisker motion-energy stream')
```

```python
if len(trial_idx) < 2:
    raise ValueError(f'only {len(trial_idx)} valid trials')
...
if not probes:
    raise ValueError('no good grey-matter neurons')
```

iii. The notes explicitly discuss one duplicate wheel timestamp as an edge case that was “fixed by retaining the first finite sample at each duplicate timestamp,” and they say the general policy is to “never extrapolate behavior” and to drop unusable trials or sessions rather than synthesize data.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified large spike-array I/O and per-trial spike binning as the main time cost. The code memory-maps spike arrays and repeatedly slices them per trial and per probe; behavior interpolation is much lighter by comparison.

ii.
```python
st = np.load(parent / 'spikes.times.npy', mmap_mode='r')
sc = np.load(parent / 'spikes.clusters.npy', mmap_mode='r')
```

```python
for k, stim in enumerate(stim_times):
    ...
    for p, off in zip(probes, offsets):
        ...
        counts = np.bincount(flat, minlength=p['n_good'] * N_BINS)
```

iii. In the notes, the AI says “Spike arrays can exceed 50 million rows/probe” and calls this the dominant cost, motivating memory maps and `searchsorted` windowing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main unvectorized loops are the per-trial interpolation loop in `load_behavior()`, the per-trial and per-probe loops in `bin_spikes()`, and the final per-trial output construction loop. The AI partially vectorized within those loops using lookup tables and `np.bincount`, but kept the outer loops explicit.

ii.
```python
for i in np.flatnonzero(coverage & np.isfinite(stim)):
    wheel_vals[i] = np.interp(targets[i], time_i, speed_i).astype(np.float32)
    whisk_vals[i] = np.interp(targets[i], ct, cv).astype(np.float32)
```

```python
for k, stim in enumerate(stim_times):
    ...
    for p, off in zip(probes, offsets):
        ...
```

```python
for (choice,prior), w, q in zip(s['scalar'],s['wheel'],s['whisk']):
    out=np.empty((4,N_BINS),dtype=np.int8)
    ...
```

iii. The notes frame this as a performance tradeoff: full-session vectorization would be harder on memory, while the implemented `searchsorted` + `bincount` approach is already fast enough for the target cohort.

## 10-c. What processing does the code repeat multiple times?

i. The code rereads `bwm_release.csv` inside every `process_session()` call even though it was already read in `select_reference_sessions()`. It also repeats per-trial interpolation for wheel and whisker traces and re-runs recursive `rglob()` file selection for each needed file.

ii.
```python
def select_reference_sessions(sample: bool) -> list[dict]:
    bwm = pd.read_csv(BWM_CSV, index_col=0)
    ...
```

```python
def process_session(item, br):
    ...
    bwm = pd.read_csv(BWM_CSV, index_col=0)
    probe_names = bwm.loc[bwm.eid.astype(str) == eid, 'probe_name'].astype(str).tolist()
```

```python
candidates = [p for p in root.rglob(basename)
              if parent_contains is None or parent_contains in p.as_posix()]
```

iii. The AI does not explicitly call this out as a problem in its notes, but the code and notes show a bias toward simplicity and per-session isolation over fully eliminating repeated work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps extra provenance and diagnostic information that the decoder does not use, such as neuron UUIDs, probe source paths, exclusion counts, and per-session elapsed time. It also carries continuous wheel and whisker traces through session processing only to convert them into tertile categories before writing the final dataset. Optional plotting is also diagnostic only. There are also unused constants like `SESSIONS_PQT`, `DATASETS_PQT`, and `target_n`.

ii.
```python
SESSIONS_PQT = SOURCE / '2025_Q3_IBL_et_al_BWM' / 'sessions.pqt'
DATASETS_PQT = SOURCE / 'Brainwidemap' / 'datasets.pqt'
```

```python
info = {
    'eid': eid, 'subject': item['subject'], 'session_path': str(sdir),
    'trial_source': str(trial_path), 'camera_side': camera,
    ...
    'probe_sources': [p['source'] for p in probes],
    'neuron_uuids': uuids.tolist(), 'elapsed_s': time.time()-t0,
}
```

```python
if args.show_processing:
    for s in sessions[:2]:
        plot_processing(s,(wthr,qthr),f"processing_{s['info']['eid']}.png")
```

iii. The notes justify most of this as support for “sanity checks,” provenance, and critical review, not as something needed by the downstream decoder itself.
