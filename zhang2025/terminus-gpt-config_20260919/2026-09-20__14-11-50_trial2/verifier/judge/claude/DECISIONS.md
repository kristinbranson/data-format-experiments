# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads data directly from a staged read-only ONE cache on disk (`/mnt/dataset/one_cache`) without using the ONE API or network access. It uses the frozen `bwm_release.csv` file from the reference code repository to identify sessions, then navigates the ALF directory structure directly using `Path` operations and a `choose_file()` helper to find the newest revisioned files. Each session's directory is constructed from the sessions parquet table (`lab/Subjects/subject/date/number`). Trials, wheel, camera, and spike data are loaded by reading `.npy` and `.pqt` files directly.

ii.
```python
SOURCE = Path('/mnt/dataset/one_cache')
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')

def session_dir(row: pd.Series) -> Path:
    return (SOURCE / str(row.lab) / 'Subjects' / str(row.subject) /
            str(row.date) / f'{int(row.session_number):03d}')

def choose_file(root: Path, basename: str, parent_contains: str | None = None) -> Path:
    candidates = [p for p in root.rglob(basename)
                  if parent_contains is None or parent_contains in p.as_posix()]
    ...
    return sorted(candidates, key=key)[-1]
```

iii. The AI chose to read the cache directly rather than using the ONE API because "Network unavailable" and to "avoid downloading" data. The staged cache already contained all needed files.

## 1-b. How are the data split into subjects?

i. The AI follows the reference code `0_data_caching.py` to select **one session per subject** using a seeded random permutation (`np.random.RandomState(seed=42)`) of all unique subjects in `bwm_release.csv`. Only the first CSV-listed EID for each subject is used, resulting in ~139 candidate sessions (one per subject) rather than all ~459 available sessions.

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

iii. CONVERSION_NOTES Step 4: "Matching reference `0_data_caching.py`, conversion selects one first-listed EID per subject; subjects are ordered by `np.random.seed(42)` random selection."

## 1-c. How are the data split into sessions?

i. Each entry from the seeded one-per-subject selection is treated as one session. Sessions are processed sequentially. Each session is identified by its EID and resolved to a directory path.

ii.
```python
candidates = select_reference_sessions(sample)
for i, item in enumerate(candidates):
    ...
    z = process_session(item, br)
    sessions.append(z)
```

iii. The AI follows the reference code's session-selection logic to determine the session list.

## 1-d. How are the data split into trials?

i. Each session's trials are loaded from the `_ibl_trials.table.pqt` parquet file. Each row is one trial. Trials are then filtered and the surviving trial indices are used.

ii.
```python
def load_trials(sdir: Path):
    p = choose_file(sdir / 'alf', '_ibl_trials.table.pqt')
    return pd.read_parquet(p), p
```

iii. The trial table already provides one row per trial; no further splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a conjunction of several filters: (1) all required columns (`stimOn_times`, `firstMovement_times`, `feedback_times`, `choice`, `probabilityLeft`, `intervals_0`, `intervals_1`) must be finite; (2) reaction time (first movement - stimulus onset) must be between 0.08 and 2.0 s; (3) feedback must occur after stimulus; (4) trial duration must be >0 and <=10 s; (5) choice must be -1 or +1; (6) prior must be one of 0.2, 0.5, 0.8; (7) behavior window (wheel and camera) must cover the trial window.

ii.
```python
def trial_mask(trials: pd.DataFrame, behavior_coverage: np.ndarray):
    ...
    mask = (finite & behavior_coverage & (rt >= 0.08) & (rt <= 2.0) &
            (feedback >= stim) & (duration > 0) & (duration <= 10.0) &
            np.isin(choice, [-1.0, 1.0]) & valid_prior)
    ...
    return mask, reasons
```

iii. CONVERSION_NOTES Step 5: "Require finite `stimOn_times`, `firstMovement_times`, `feedback_times`, `choice`, and `probabilityLeft`; valid event order; trial duration <=10 s; first movement latency 0.08-2.00 s; choice +/-1; prior in {0.2,0.5,0.8}; and complete neural/wheel/whisker temporal coverage."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` for each probe, plus `clusters.metrics.pqt` for QC labels, `clusters.channels.npy` for channel assignments, and `channels.brainLocationIds_ccf_2017.npy` for anatomy.

ii.
```python
def load_probe(probe_root: Path, br: BrainRegions):
    metrics_p = choose_file(probe_root, 'clusters.metrics.pqt', 'pykilosort')
    parent = metrics_p.parent
    metrics = pd.read_parquet(metrics_p)
    ch = np.load(parent / 'clusters.channels.npy', mmap_mode='r')
    st = np.load(parent / 'spikes.times.npy', mmap_mode='r')
    sc = np.load(parent / 'spikes.clusters.npy', mmap_mode='r')
```

iii. The AI loads spike times and cluster assignments directly from the pykilosort revision directories.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20 ms bins over the trial window [-0.5, +1.5) s relative to stimulus onset, giving 100 bins of raw spike counts per neuron per trial. The counts are stored as `float32` but are **not** divided by bin width (i.e., they remain as counts, not firing rates in Hz). Multiple probes within a session are merged with a continuous neuron index.

ii.
```python
def bin_spikes(probes, stim_times):
    n_neurons = sum(p['n_good'] for p in probes)
    out = []
    for k, stim in enumerate(stim_times):
        mat = np.zeros((n_neurons, N_BINS), dtype=np.float32)
        abs_edges = stim + EDGES_REL
        for p, off in zip(probes, offsets):
            ...
            tb = np.floor((t - abs_edges[0]) / BIN + 1e-10).astype(np.int32)
            ok = (local >= 0) & (tb >= 0) & (tb < N_BINS)
            flat = (local[ok].astype(np.int64) * N_BINS + tb[ok])
            counts = np.bincount(flat, minlength=p['n_good'] * N_BINS)
            mat[off:off+p['n_good']] += counts.reshape(p['n_good'], N_BINS).astype(np.float32)
        out.append(mat)
    return out
```

iii. CONVERSION_NOTES: "Counts, not smoothed rates; fixed stable UUID/order per session."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1.0` (passing all RIGOR QC criteria) are kept. Additionally, clusters are filtered by anatomy: channels must have valid indices, and the Beryl-mapped region must not be `root`, `void`, `nan`, `none`, or empty string.

ii.
```python
labels = metrics.label.to_numpy(float)
regions = np.asarray(br.id2acronym(atlas_ids, mapping='Beryl')).astype(str)
bad_names = np.isin(np.char.lower(regions), ['root', 'void', 'nan', 'none', ''])
good = (labels >= 1.0) & anatomical_ok & (~bad_names)
```

iii. CONVERSION_NOTES Step 5: "Explicitly keep merged cluster QC `label>=1`, Beryl grey-matter labels only."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are binned relative to that trial's `stimOn_times`. The bin edges are computed as `stim + EDGES_REL` where `EDGES_REL = np.linspace(-0.5, 1.5, 101)`. Spikes within this absolute window are assigned to bins relative to the stimulus onset.

ii.
```python
abs_edges = stim + EDGES_REL
...
t = np.asarray(ts[lo:hi])
tb = np.floor((t - abs_edges[0]) / BIN + 1e-10).astype(np.int32)
```

iii. CONVERSION_NOTES: "Exact same" as reference `align_time=stimOn_times`, `time_window=(-.5,1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, with 100 bins spanning [-0.5, +1.5) s. No rebinning or interpolation is applied to the neural data.

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
N_BINS = 100
EDGES_REL = np.linspace(OFF_START, OFF_END, N_BINS + 1)
```

iii. CONVERSION_NOTES Step 5: "Exactly use reference caching parameters: `interval_len=2`, `binsize=0.02`, `align_time=stimOn_times`, `time_window=(-0.5, 1.5)`."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial's `stimOn_times` and the fixed bin grid. The AI uses the **right edges** of the bins as the time values, not the bin centers.

ii.
```python
TIME_REL = EDGES_REL[1:].astype(np.float32)
```
This produces values: -0.48, -0.46, ..., 1.48, 1.50 s.

iii. CONVERSION_NOTES: "Bin centers relative to stimulus onset" (though the code actually uses right edges, not centers). The AI's metadata states: `'bin_semantics':'neural counts in [edge_i,edge_i+1); time/behavior sampled at right bin edge'`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are the right edges of each 20 ms bin, computed once as a constant array and repeated for every trial.

ii.
```python
EDGES_REL = np.linspace(OFF_START, OFF_END, N_BINS + 1)
TIME_REL = EDGES_REL[1:].astype(np.float32)
```

iii. No trial-specific processing; the time grid is a constant.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time values correspond to the right edges of the same bin grid used for spike counting. Neural spikes are counted in bins `[edge_i, edge_i+1)` and the time input for that bin is `edge_i+1` (the right edge).

ii.
```python
abs_edges = stim + EDGES_REL  # spike binning edges
# TIME_REL = EDGES_REL[1:]    # right edges used as time input
```

iii. The AI's metadata explicitly states this: `'bin_semantics':'neural counts in [edge_i,edge_i+1); time/behavior sampled at right bin edge'`.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. The AI does **not** compute trial number in block. Instead, it uses the **raw trial index within the session** (the original zero-based row index from the trial table). The input is named `'trial_number_in_session'`.

ii.
```python
for raw_i, (_, tr) in zip(trial_idx, selected.iterrows()):
    inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)]).astype(np.float32)
```
Where `trial_idx = np.flatnonzero(mask)` are the indices of valid trials in the original trial table.

iii. CONVERSION_NOTES Step 5: "Native trial index within session ... Preserves gaps caused by trial QC and thus true trial position/block progression; continuous `float32`." The AI named this `'trial_number_in_session'` rather than `'trial_number_in_block'`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. As noted above, the AI computes the **original trial index** (position in the raw trial table), not the trial number within a block. This index is broadcast as a constant across all 100 time bins for each trial.

ii.
```python
inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)]).astype(np.float32)
```

iii. The AI's justification was that the original trial index "preserves gaps caused by trial QC and thus true trial position/block progression."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column from the trials table, which takes values +1, -1, or 0 in the IBL convention.

ii.
```python
choice = 0 if float(tr.choice) == -1 else 1
```

iii. CONVERSION_NOTES Step 5: "Per-trial scalar: native -1 (left wheel/left choice) -> 0; native +1 (right) -> 1; reject 0/no-go."

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps IBL `choice = -1` to output 0 and `choice = +1` to output 1. The AI believed that IBL `choice = -1` means "left" and `choice = +1` means "right". However, the standard IBL convention (and the reference solution) is that `choice = +1` means LEFT and `choice = -1` means RIGHT. This means the AI's mapping is **inverted** relative to the instructions ("left = 0, right = 1").

ii.
```python
choice = 0 if float(tr.choice) == -1 else 1
```
Metadata: `'choice_mapping':{'-1':0,'1':1}`

iii. CONVERSION_NOTES: "IBL native `choice=-1` denotes a left wheel turn/left choice and maps to required left=0; `choice=+1` maps to right=1." This understanding of the IBL convention is incorrect.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column from the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
prior_levels = np.array([0.2, 0.5, 0.8])
prior = int(np.argmin(np.abs(prior_levels - float(tr.probabilityLeft))))
```

iii. The AI maps using closest-match to the three expected levels.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI uses `np.argmin(np.abs(prior_levels - value))` to find the closest of [0.2, 0.5, 0.8] and assigns the index (0, 1, or 2). This produces the mapping 0.2->0, 0.5->1, 0.8->2, matching the instructions.

ii.
```python
prior_levels = np.array([0.2, 0.5, 0.8])
prior = int(np.argmin(np.abs(prior_levels - float(tr.probabilityLeft))))
```

iii. No additional processing beyond the mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, loaded directly from the ALF directory.

ii.
```python
wt = np.load(choose_file(alf, '_ibl_wheel.timestamps.npy'), mmap_mode='r')
wp = np.load(choose_file(alf, '_ibl_wheel.position.npy'), mmap_mode='r')
```

iii. Same raw source as the reference solution.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI uses the IBL `brainbox.behavior.wheel` utilities to interpolate wheel position to 1 kHz, compute filtered velocity via a Butterworth low-pass filter, then takes the absolute value for speed. The speed is then linearly interpolated (`np.interp`) to the bin right-edge time points for each trial.

ii.
```python
pos_i, time_i = wheel_utils.interpolate_position(wt0, wp0, freq=1000, kind='linear')
vel_i, _ = wheel_utils.velocity_filtered(pos_i, fs=1000)
speed_i = np.abs(vel_i)
...
wheel_vals[i] = np.interp(targets[i], time_i, speed_i).astype(np.float32)
```

iii. CONVERSION_NOTES: "Derive velocity with IBL `SessionLoader`/reference wheel interpolation, take absolute value for speed."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI uses **pooled global tertiles** across all retained sessions and trials. The 1/3 and 2/3 quantiles of all aligned wheel speed values (pooled across all sessions) are computed, and `np.searchsorted(side='right')` assigns each value to one of three bins (0=low, 1=medium, 2=high).

ii.
```python
wheel_all = np.concatenate([s['wheel'].ravel() for s in sessions])
wthr = np.quantile(wheel_all, [1/3, 2/3]).astype(float)
...
out[2] = np.searchsorted(wthr, w, side='right').astype(np.int8)
```

iii. CONVERSION_NOTES Step 5: "Use pooled empirical tertiles computed from all valid finite samples after alignment."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to the **right edges** of the bin grid (`TIME_REL = EDGES_REL[1:]`), which differs from the reference's use of bin centers. The interpolation uses `np.interp` with absolute time points (`stim + TIME_REL`).

ii.
```python
targets = stim[:, None] + TIME_REL[None, :]
wheel_vals[i] = np.interp(targets[i], time_i, speed_i).astype(np.float32)
```

iii. The AI documents this as sampling "at right bin edge."

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `{left,right}Camera.ROIMotionEnergy.npy` and `_ibl_{left,right}Camera.times.npy`, with left camera preferred and right as fallback.

ii.
```python
for side in ('left', 'right'):
    try:
        ct = np.load(choose_file(alf, f'_ibl_{side}Camera.times.npy'), mmap_mode='r')
        cv = np.load(choose_file(alf, f'{side}Camera.ROIMotionEnergy.npy'), mmap_mode='r')
        ...
        camera = side
        break
    except FileNotFoundError:
        continue
```

iii. CONVERSION_NOTES: "Prefer left camera to match reference code, use right only when left is unavailable/unusable."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no filtering or normalization). It is linearly interpolated to the bin right-edge time points for each trial using `np.interp`. The AI also handles non-finite values and duplicate timestamps by filtering them before interpolation.

ii.
```python
cf = np.isfinite(ct) & np.isfinite(cv)
ct0, cv0 = np.asarray(ct[cf]), np.asarray(cv[cf])
_, ci = np.unique(ct0, return_index=True)
ci.sort(); ct0, cv0 = ct0[ci], cv0[ci]
...
whisk_vals[i] = np.interp(targets[i], ct, cv).astype(np.float32)
```

iii. No additional processing beyond interpolation and duplicate/NaN handling.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: **pooled global tertiles** across all sessions. The 1/3 and 2/3 quantiles of all aligned whisker motion energy values are computed, and `np.searchsorted(side='right')` assigns each value to bins 0, 1, or 2.

ii.
```python
whisk_all = np.concatenate([s['whisk'].ravel() for s in sessions])
qthr = np.quantile(whisk_all, [1/3, 2/3]).astype(float)
...
out[3] = np.searchsorted(qthr, q, side='right').astype(np.int8)
```

iii. CONVERSION_NOTES: "Quantile bins create approximately balanced classes and are deterministic."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel: interpolated to right bin edges (`TIME_REL`) using `np.interp`, with coverage checked to prevent extrapolation.

ii.
```python
targets = stim[:, None] + TIME_REL[None, :]
whisk_vals[i] = np.interp(targets[i], ct, cv).astype(np.float32)
```

iii. Same alignment approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Duplicate wheel timestamps are resolved by keeping the first finite sample at each time. (2) Non-finite values in wheel and camera timestamps/values are filtered out. (3) Sessions without a complete whisker camera stream are skipped. (4) Sessions with fewer than 2 valid trials are skipped. (5) Sessions with no good neurons are skipped. (6) Trials without full behavior window coverage are excluded.

ii.
```python
# Wheel duplicate handling
wf = np.isfinite(wt) & np.isfinite(wp)
wt0, wp0 = np.asarray(wt[wf]), np.asarray(wp[wf])
_, unique_idx = np.unique(wt0, return_index=True)
unique_idx.sort(); wt0, wp0 = wt0[unique_idx], wp0[unique_idx]

# Session skip
if len(trial_idx) < 2:
    raise ValueError(f'only {len(trial_idx)} valid trials')
if not probes:
    raise ValueError('no good grey-matter neurons')
```

iii. CONVERSION_NOTES Step 10: "Duplicate wheel timestamp: one exact duplicate initially caused a sample-session skip. Fixed by deterministic finite filtering/deduplication before interpolation."

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike arrays from disk (memory-mapped reads of potentially hundreds of millions of spike times/clusters per probe) and the per-trial spike binning loop dominate processing time.

ii.
```python
st = np.load(parent / 'spikes.times.npy', mmap_mode='r')
sc = np.load(parent / 'spikes.clusters.npy', mmap_mode='r')
```

And the spike binning loop in `bin_spikes()` which iterates over each trial.

iii. The AI used memory mapping and `searchsorted` to minimize actual I/O. Sample session processing took ~0.4-0.6 s each.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_spikes()` function loops over trials (`for k, stim in enumerate(stim_times)`), and within each trial loops over probes. The trial loop could potentially be vectorized by offsetting spike indices. The behavior interpolation loop (`for i in np.flatnonzero(coverage ...)`) also iterates per-trial.

ii.
```python
for k, stim in enumerate(stim_times):
    mat = np.zeros((n_neurons, N_BINS), dtype=np.float32)
    ...
    for p, off in zip(probes, offsets):
        ...
```

iii. No explicit justification given for not vectorizing.

## 10-c. What processing does the code repeat multiple times?

i. The `bwm_release.csv` is read twice: once in `select_reference_sessions()` at startup and again inside `process_session()` to find probe names for each session.

ii.
```python
# In select_reference_sessions():
bwm = pd.read_csv(BWM_CSV, index_col=0)

# In process_session():
bwm = pd.read_csv(BWM_CSV, index_col=0)
probe_names = bwm.loc[bwm.eid.astype(str) == eid, 'probe_name'].astype(str).tolist()
```

iii. No justification given; this appears to be an oversight.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores detailed per-session metadata including `neuron_uuids`, `trial_indices`, `probe_sources`, `trial_exclusion_counts`, and `skipped_sessions` in the metadata dictionary. While useful for debugging, these are not used by the decoder. The duplicate timestamp handling code also does extra work (sorting unique indices) even when there are no duplicates.

ii.
```python
info = {
    'eid': eid, 'subject': item['subject'], 'session_path': str(sdir),
    'trial_source': str(trial_path), 'camera_side': camera,
    'n_trials_raw': len(trials), 'trial_indices': trial_idx.astype(np.int32),
    ...
    'neuron_uuids': uuids.tolist(), ...
}
```

iii. No explicit justification; the extra metadata is for documentation/reproducibility purposes.
