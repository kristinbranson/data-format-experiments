# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the ONE API. It reads ALF files directly off the read-only staged cache at `/mnt/dataset/one_cache`, resolving paths itself from `lab/Subjects/<subject>/<date>/<number>/alf/...` and picking the newest `#YYYY-MM-DD#` revision directory with a helper (`choose_file`). Spike arrays are opened with `mmap_mode='r'` and only the trial windows are paged in.

The set of data loaded is *not* the whole release. Following `code_zhang2025/src/0_data_caching.py`, the AI seeds `np.random.RandomState(42)`, permutes the unique subjects of the frozen `bwm_release.csv` (459 eids / 139 subjects), and takes the **first-listed eid for each subject** — i.e. exactly one session per mouse. Full mode therefore starts from 139 candidate sessions, not 459; 134 are retained (5 dropped for having no usable whisker camera stream), yielding 55,475 trials and 21,396 units. Probe insertions are taken from the `probe_name` column of `bwm_release.csv` for that eid and merged within a session.

ii.
```python
SOURCE = Path('/mnt/dataset/one_cache')
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')

def select_reference_sessions(sample: bool) -> list[dict]:
    """Match 0_data_caching.py: seeded subject permutation, first CSV EID/subject."""
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

def session_dir(row: pd.Series) -> Path:
    return (SOURCE / str(row.lab) / 'Subjects' / str(row.subject) /
            str(row.date) / f'{int(row.session_number):03d}')

def choose_file(root: Path, basename: str, parent_contains: str | None = None) -> Path:
    candidates = [p for p in root.rglob(basename)
                  if parent_contains is None or parent_contains in p.as_posix()]
    ...
    return sorted(candidates, key=key)[-1]
```
```python
    bwm = pd.read_csv(BWM_CSV, index_col=0)
    probe_names = bwm.loc[bwm.eid.astype(str) == eid, 'probe_name'].astype(str).tolist()
    for name in probe_names:
        root = sdir / 'alf' / name
        if root.exists():
            p = load_probe(root, br)
```

iii. From CONVERSION_NOTES.md Step 5 Key Decision 1: *"Use frozen `bwm_release.csv`. Exactly match reference `0_data_caching.py`: `np.random.seed(42)`, unique subjects, random subject order, then the first CSV-listed EID for each selected subject. Full mode selects all 139 subjects (one session/subject) … rather than treating repeated sessions from the same animal as independent."* Step 4 records that the AI initially planned to use all 459 frozen sessions and then reversed: *"Subsequent inspection of the actual reference caching call site showed that BWM processing samples one EID per subject (`SEED=42`) … Steps 4–5 were corrected before implementation."* Network access was unavailable, so direct staged-cache reads replaced ONE downloads (Step 10, "Network unavailable").

## 1-b. How are the data split into subjects (mice)?

i. The subject label comes straight from the `subject` column of `bwm_release.csv` (carried through `item['subject']`); no path parsing or inference. At assembly the subject list is the sorted unique set over retained sessions and `subject_idx` is the index of each session's subject into that list. Because exactly one session per subject is kept, the mapping is one-to-one: 134 sessions / 134 subjects, 1 session each.

ii.
```python
        out.append({'eid': str(row.eid), 'subject': str(subject), 'row': row})
```
```python
    subjects = sorted(set(s['info']['subject'] for s in sessions))
    subject_map = {v:i for i,v in enumerate(subjects)}
    ...
      'subject_idx':np.array([subject_map[s['info']['subject']] for s in sessions],dtype=np.int32),
```

iii. The release freeze table already carries a unique animal id per session, so nothing has to be derived. Step 9's consistency table records "Subjects: 139 candidates → 134 retained; five subjects excluded solely for unavailable mandatory whisker output."

## 1-c. How are the data split into sessions?

i. A session is one eid = one `lab/Subjects/<subject>/<date>/<number>` directory, which is already the unit the release is organised by, so there is nothing to split. Each retained eid becomes one element of `neural`/`input`/`output`, with its own `brain_region_idx` entry and `session_info` metadata record. All probes belonging to the eid are merged into one population (see 2-b) rather than being treated as separate sessions. The only non-trivial part is that only one session per subject enters the loop at all (see 1-a).

ii.
```python
    for i, item in enumerate(candidates):
        ...
            z = process_session(item, br)
            sessions.append(z)
```
```python
    info = {
        'eid': eid, 'subject': item['subject'], 'session_path': str(sdir),
        'trial_source': str(trial_path), 'camera_side': camera, ...}
```

iii. CONVERSION_NOTES.md Step 4: *"Session is the target-format session unit. All probes in an EID are merged because they share trials/behavior and are not independent."*

## 1-d. How are the data split into trials?

i. The trials table `_ibl_trials.table.pqt` has one row per trial, so the split is given by the data. The AI keeps the *raw* row index of every retained trial (`trial_idx`) and stores it in metadata, so the original position of each trial in the session is recoverable.

ii.
```python
def load_trials(sdir: Path):
    p = choose_file(sdir / 'alf', '_ibl_trials.table.pqt')
    return pd.read_parquet(p), p
```
```python
    mask, reasons = trial_mask(trials, coverage)
    trial_idx = np.flatnonzero(mask)
    ...
    selected = trials.iloc[trial_idx]
```

iii. No decision to make — the trials table is already one row per trial. The AI notes it retains `trial_indices` so QC gaps remain visible ("Original rather than compressed retained-trial order preserves trial position and block structure", Step 5 Decision 9).

## 1-e. How are trials filtered based on quality controls?

i. A single conjunctive mask, combining the reference code's `load_trials_and_mask` rules, the data paper's reaction-time rule, and a stream-coverage requirement:
- all of `stimOn_times, firstMovement_times, feedback_times, choice, probabilityLeft, intervals_0, intervals_1` finite;
- reaction time `firstMovement_times - stimOn_times` in [0.08, 2.0] s;
- `feedback_times >= stimOn_times` and `0 < intervals_1 - intervals_0 <= 10 s`;
- `choice ∈ {-1, +1}` (no-go dropped) and `probabilityLeft ∈ {0.2, 0.5, 0.8}` (atol 1e-6);
- the full [-0.5, +1.5] s window covered by both the wheel and the camera stream.
Per-reason exclusion counts are recorded in metadata. Sessions left with fewer than 2 valid trials are dropped. Zero-contrast trials are deliberately kept. Net effect: 55,475 of 85,688 raw trials retained (mean 414/session).

ii.
```python
def trial_mask(trials: pd.DataFrame, behavior_coverage: np.ndarray):
    required = ['stimOn_times', 'firstMovement_times', 'feedback_times', 'choice',
                'probabilityLeft', 'intervals_0', 'intervals_1']
    ...
    rt = move - stim
    duration = trials.intervals_1.to_numpy(float) - trials.intervals_0.to_numpy(float)
    valid_prior = np.isclose(prior[:, None], [0.2, 0.5, 0.8], atol=1e-6).any(1)
    mask = (finite & behavior_coverage & (rt >= 0.08) & (rt <= 2.0) &
            (feedback >= stim) & (duration > 0) & (duration <= 10.0) &
            np.isin(choice, [-1.0, 1.0]) & valid_prior)
    reasons = {'nonfinite_required': ..., 'reaction_time_outside_0.08_2.0': ...,
               'invalid_duration_or_order': ..., 'invalid_choice': ...,
               'invalid_prior': ..., 'behavior_window_uncovered': ...}
    return mask, reasons
```
```python
    trial_idx = np.flatnonzero(mask)
    if len(trial_idx) < 2:
        raise ValueError(f'only {len(trial_idx)} valid trials')
```

iii. Step 4 discrepancy table: *"`load_trials_and_mask` checks finite/order constraints and max length 10 s … Paper also requires stimulus-to-first-movement latency 0.08–2.00 s. Resolution: Use the conjunction of code mask and paper latency criterion; preserve zero-contrast trials because stimulus side is not decoded."* Step 3 quotes the data paper: trials excluded if choice/probabilityLeft/feedbackType/feedback times/stimOn times/firstMovement times are undetectable, or if stim-onset→first-movement is outside 0.08–2.00 s. Coverage is required because *"both time-varying outputs are mandatory; do not impute missing streams."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Per probe: `spikes.times.npy` and `spikes.clusters.npy` for the activity itself; `clusters.metrics.pqt` (`cluster_id`, `label`) for quality; `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy` for anatomy (mapped to Beryl); `clusters.uuids.csv` only for a duplicate-identity check. All are taken from the same coherent `pykilosort` revision directory.

ii.
```python
def load_probe(probe_root: Path, br: BrainRegions):
    metrics_p = choose_file(probe_root, 'clusters.metrics.pqt', 'pykilosort')
    parent = metrics_p.parent
    metrics = pd.read_parquet(metrics_p)
    ch = np.load(parent / 'clusters.channels.npy', mmap_mode='r')
    st = np.load(parent / 'spikes.times.npy', mmap_mode='r')
    sc = np.load(parent / 'spikes.clusters.npy', mmap_mode='r')
    channel_ids = np.load(parent / 'channels.brainLocationIds_ccf_2017.npy', mmap_mode='r')
    cids = metrics.cluster_id.to_numpy(int) if 'cluster_id' in metrics else np.arange(len(metrics))
    labels = metrics.label.to_numpy(float)
```

iii. Step 1 identifies `load_spiking_data`/`merge_probes`/`bin_spiking_data` as the reference functions; Step 10's reference-comparison table states the converter uses "Direct staged ALF reads using the same frozen CSV and listed probes … newest coherent revisions selected", i.e. an offline re-implementation of the same loader inputs.

## 2-b. How is the `neural` data processed?

i. Spikes of the surviving clusters are histogrammed into 100 non-overlapping 20 ms bins spanning `[stimOn-0.5, stimOn+1.5)`, giving **raw spike counts** (float32), not firing rates and without smoothing. Only the spikes inside each trial window are touched (`searchsorted` on the memory-mapped time array). Clusters are renumbered densely through a `lookup` table; when a session has two probes their good units are concatenated along the neuron axis with a running offset, so one session yields one pooled population with a stable neuron ordering.

ii.
```python
def bin_spikes(probes, stim_times):
    n_neurons = sum(p['n_good'] for p in probes)
    offsets = np.cumsum([0] + [p['n_good'] for p in probes[:-1]])
    for k, stim in enumerate(stim_times):
        mat = np.zeros((n_neurons, N_BINS), dtype=np.float32)
        abs_edges = stim + EDGES_REL
        for p, off in zip(probes, offsets):
            ts, cs = p['times'], p['clusters']
            lo = int(np.searchsorted(ts, abs_edges[0], side='left'))
            hi = int(np.searchsorted(ts, abs_edges[-1], side='left'))
            ...
            local[in_lookup] = p['lookup'][c[in_lookup]]
            tb = np.floor((t - abs_edges[0]) / BIN + 1e-10).astype(np.int32)
            ok = (local >= 0) & (tb >= 0) & (tb < N_BINS)
            flat = (local[ok].astype(np.int64) * N_BINS + tb[ok])
            counts = np.bincount(flat, minlength=p['n_good'] * N_BINS)
            mat[off:off+p['n_good']] += counts.reshape(p['n_good'], N_BINS).astype(np.float32)
```

iii. Step 5 mapping table: *"Merge probes, keep `label>=1` good grey-matter units, histogram counts into 20 ms bins from −0.5 to +1.5 s relative to `stimOn_times` … Counts, not smoothed rates; fixed stable UUID/order per session."* Merging probes follows the data paper — Step 4: *"All probes in an EID are merged because they share trials/behavior and are not independent."*

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cluster is kept only if (a) its merged IBL `label >= 1.0` — the encoding of "passes all three RIGOR metrics" (amplitude > 50 µV, noise cutoff < 20 µV, refractory-period violations) — and (b) its peak channel index is valid, and (c) its Beryl acronym is not `root`, `void`, `nan`, `none` or empty. So, unlike the reference, `root` units are also dropped. 21,396 of 199,417 raw clusters in the selected sessions survive (10.7%, vs the paper's 12.2% release-wide). A session with no surviving unit is skipped.

ii.
```python
    regions = np.asarray(br.id2acronym(atlas_ids, mapping='Beryl')).astype(str)
    bad_names = np.isin(np.char.lower(regions), ['root', 'void', 'nan', 'none', ''])
    good = (labels >= 1.0) & anatomical_ok & (~bad_names)
    good_cids = cids[good]
    good_regions = regions[good]
```
```python
    if not probes:
        raise ValueError('no good grey-matter neurons')
```

iii. Step 4: *"Filter to merged IBL `label >= 1`, the reference pipeline's operational encoding of passing unit QC; verify metric fields/spot checks and compare total to 75,708. Do not merely store the indicator while retaining noise units."* and *"Use Beryl/Allen mapping as in IBL utilities, exclude void/root/non-grey labels, retain good neurons."* Step 10 additionally verified that the odd-looking acronym `x` is the genuine Allen "Nucleus x" (id 765) and correctly retained.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every stream in an IBL session is already on one synchronised clock, so alignment is a subtraction: the bin edges for trial *k* are `stimOn_times[k] + EDGES_REL`, with `EDGES_REL` spanning −0.5 to +1.5 s. Bins are left-closed/right-open; the `+1e-10` nudge inside `np.floor` guards against a spike landing exactly on an edge being pushed into the wrong bin by floating-point error.

ii.
```python
OFF_START, OFF_END = -0.5, 1.5
EDGES_REL = np.linspace(OFF_START, OFF_END, N_BINS + 1)
```
```python
        abs_edges = stim + EDGES_REL
        ...
            tb = np.floor((t - abs_edges[0]) / BIN + 1e-10).astype(np.int32)
```
```python
    neural = bin_spikes(probes, selected.stimOn_times.to_numpy(float))
```

iii. Step 5 Decision 2: *"Exactly use reference caching parameters: `interval_len=2`, `binsize=0.02`, `align_time=stimOn_times`, `time_window=(-0.5, 1.5)`."* Step 10 Check 10: *"Spike bins are left-closed/right-open over `[stim−0.5, stim+1.5)`."* An independent raw-data `np.allclose` histogram check (`/app/cache/raw_sanity_checks.py`, session 0 / trial 5 / neuron 3) reproduced the converted 100-bin vector exactly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 of them, covering a 2 s window; `time_bin_size` is recorded as 20.0 ms in metadata. Spikes are histogrammed once directly onto that grid — there is no intermediate binning and no resampling/rebinning of the neural data. Every trial in every session has exactly T = 100 (asserted before pickling and confirmed by the validator).

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
N_BINS = 100
EDGES_REL = np.linspace(OFF_START, OFF_END, N_BINS + 1)
```
```python
        for n,x,y in zip(data['neural'][si],data['input'][si],data['output'][si]):
            assert n.shape[1]==x.shape[1]==y.shape[1]==N_BINS
```

iii. Step 5 Decision 2 copies the reference caching parameters verbatim (`binsize=0.02`, `time_window=(-0.5,1.5)`); Step 3 corroborates from the data paper: *"We averaged wheel values in nonoverlapping 20-ms bins… Spike counts were similarly binned"*, and notes 12.5 ms is used only for the continuous Granger analysis.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not a raw variable but the binning grid itself, defined relative to `stimOn_times`. The AI uses the **right edge** of each bin, i.e. −0.48, −0.46, …, +1.50 s, and uses the same vector for every trial and session.

ii.
```python
EDGES_REL = np.linspace(OFF_START, OFF_END, N_BINS + 1)
# Reference behavior interpolation predicts the value at each spike bin's right edge.
TIME_REL = EDGES_REL[1:].astype(np.float32)
```
```python
        inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)]).astype(np.float32)
```

iii. The right-edge convention is copied from the reference code, whose `get_behavior_per_interval` interpolates onto `np.linspace(interval_beg + binsize, interval_end, n_bins)`. Metadata records `'bin_semantics': 'neural counts in [edge_i,edge_i+1); time/behavior sampled at right bin edge'`. (Note the Step 5 mapping table still describes this row as "bin centers … −0.49, −0.47, …, 1.49 s", a stale line that the implementation and all later documentation contradict.)

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None — the vector is constructed analytically from the window and bin size and broadcast to every trial as row 0 of the `(2, 100)` float32 input array. It is a continuous ramp, not a binary event indicator, because the task specifies "Time since stimulus onset, continuous, time-varying".

ii.
```python
TIME_REL = EDGES_REL[1:].astype(np.float32)
...
        inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)]).astype(np.float32)
        inputs.append(inp)
```

iii. Step 5 mapping table: *"A continuous time input is explicitly requested, so it is not a binary event indicator."*

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Bin for bin, by construction: `TIME_REL[i]` is the right edge of the same interval `[EDGES_REL[i], EDGES_REL[i+1])` in which the spikes of neural column *i* were counted, both measured from the same `stimOn_times`. The behavioural outputs are sampled at those same right edges, so all four streams share one axis. A structural assertion enforces equal T across neural/input/output for every trial.

ii.
```python
        abs_edges = stim + EDGES_REL          # neural bins
        ...
    targets = stim[:, None] + TIME_REL[None, :]   # behavior sample times
```
```python
            assert n.shape[1]==x.shape[1]==y.shape[1]==N_BINS
```

iii. Metadata `bin_semantics` (above) plus Step 10 Check 10: *"Spike bins are left-closed/right-open over `[stim−0.5, stim+1.5)`; behavior is evaluated at exact right edges as in reference interpolation."*

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. **It is not computed.** The AI never derives blocks from `probabilityLeft`. Row 1 of the input is the raw row index of the trial in the session's trials table (`raw_i`, i.e. trial number *in the session*), and the input is named `trial_number_in_session`. Its values run up to 1437 for long sessions, as shown in the validator's per-session input ranges.

ii.
```python
    for raw_i, (_, tr) in zip(trial_idx, selected.iterrows()):
        inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)]).astype(np.float32)
```
```python
      'input_names':['time_since_stimulus_onset_s','trial_number_in_session'],
```

iii. Step 5 mapping table: *"Native trial index within session → `input[1]`; Zero-based original trial number … Preserves gaps caused by trial QC and thus true trial position/block progression"*, and Decision 9: *"repeat original trial number over bins. Original rather than compressed retained-trial order preserves trial position and block structure."* The trajectory shows no point at which the agent revisited the "Trial number in block" wording of the specification.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Only a cast and a broadcast: the retained trial's raw table index is repeated across all 100 bins as a float32 row. There is no block detection (no `probabilityLeft` change-point/cumsum), no reset of the counter at block boundaries, and no normalisation of the magnitude.

ii.
```python
    trial_idx = np.flatnonzero(mask)
    ...
    for raw_i, (_, tr) in zip(trial_idx, selected.iterrows()):
        inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)]).astype(np.float32)
```

iii. As above — the AI's stated rationale is that keeping the un-renumbered index preserves the true position of the trial within the session (and, it claims, "block progression") despite QC gaps.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 / −1 / 0. Trials with `choice == 0` (no response) are already removed by the trial mask. The surviving values are recoded as **−1 → 0 ("left") and +1 → 1 ("right")**, with `output_values[0] = ['left','right']`.

ii.
```python
    mask = (... np.isin(choice, [-1.0, 1.0]) ...)
```
```python
        choice = 0 if float(tr.choice) == -1 else 1
```
```python
      'output_values':[['left','right'], ...],
      'choice_mapping':{'-1':0,'1':1},
```

iii. Step 5 Decision 5: *"IBL native `choice=-1` denotes a left wheel turn/left choice and maps to required left=0; `choice=+1` maps to right=1. This will be checked against raw trials on named examples."* Step 12 reports the check as passed: *"Direct raw checks on three named converted trials confirmed native −1→left/0 and +1→right/1 exactly."* The check in `/app/cache/raw_sanity_checks.py` re-applies the same rule (`choice=0 if tr.choice==-1 else 1`) and compares it to the pickle, so it verifies the recoding but not the semantics.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Nothing beyond the ±1 → {0,1} recoding described above plus a broadcast: because a single NumPy array cannot mix per-trial and time-varying rows, the scalar choice is repeated across all 100 bins as row 0 of the `(4, 100)` int8 output array.

ii.
```python
            out=np.empty((4,N_BINS),dtype=np.int8)
            out[0]=choice; out[1]=prior
```

iii. Step 5 Decision 8: *"Use shape `(4, 100)` integer output arrays for every trial. Per-trial choice/prior are repeated across time, while wheel/whisker vary. This is required because a single NumPy array cannot mix scalar and temporal rows."*

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table. The trial mask already restricts it to {0.2, 0.5, 0.8} (within 1e-6), and the value is recoded to the nearest of those three levels by index: 0.2→0, 0.5→1, 0.8→2.

ii.
```python
    valid_prior = np.isclose(prior[:, None], [0.2, 0.5, 0.8], atol=1e-6).any(1)
```
```python
    prior_levels = np.array([0.2, 0.5, 0.8])
    ...
        prior = int(np.argmin(np.abs(prior_levels - float(tr.probabilityLeft))))
```
```python
      'output_values':[..., ['0.2','0.5','0.8'], ...],
      'prior_mapping':{'0.2':0,'0.5':1,'0.8':2},
```

iii. Step 5 mapping table: *"Exact tolerance mapping 0.2→0, 0.5→1, 0.8→2; repeat over time … Unexpected levels excluded and reported."* The mapping is given verbatim by the Decoder Task spec; Step 3 confirms the native levels from the data paper's block design (20:80 / 80:20 after the first 90 unbiased trials).

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Only the three-level recoding plus a broadcast over the 100 bins (row 1 of the int8 output array). The unbiased 0.5 block at the start of each session is kept, not excluded, and shows up as the minority class (7,847 trials vs 22,613 / 25,015).

ii.
```python
        prior = int(np.argmin(np.abs(prior_levels - float(tr.probabilityLeft))))
        scalar.append((choice, prior))
...
            out[1]=prior
```

iii. Step 4: *"preserve zero-contrast trials because stimulus side is not decoded"* and `exclude_unbiased` is left off, matching the reference `load_trials_and_mask` default. Step 9 records the resulting distribution as expected: *"Correct values; unbiased 0.5 blocks less frequent as expected."*

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. Speed is `|velocity|`, where velocity is produced by the IBL primitives `brainbox.behavior.wheel.interpolate_position` followed by `velocity_filtered` — i.e. the same computation `SessionLoader.load_wheel` performs internally.

ii.
```python
from brainbox.behavior import wheel as wheel_utils
...
    wt = np.load(choose_file(alf, '_ibl_wheel.timestamps.npy'), mmap_mode='r')
    wp = np.load(choose_file(alf, '_ibl_wheel.position.npy'), mmap_mode='r')
    ...
    # Exact IBL reference primitives: 1 kHz linear position then filtered derivative.
    pos_i, time_i = wheel_utils.interpolate_position(wt0, wp0, freq=1000, kind='linear')
    vel_i, _ = wheel_utils.velocity_filtered(pos_i, fs=1000)
    speed_i = np.abs(vel_i)
```

iii. Step 5 mapping table: *"Derive velocity with IBL `SessionLoader`/reference wheel interpolation, take absolute value for speed"*, matching the reference utility's `load_target_behavior('wheel-speed')`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps. (1) Non-finite samples are removed and duplicate timestamps collapsed to their first occurrence (a real edge case: one frozen session has an exact duplicate at index 80); the stream must then be strictly increasing and have ≥ 20 samples or the session is dropped. (2) Position is linearly interpolated onto a uniform 1 kHz grid. (3) A filtered derivative (`velocity_filtered`, default 20 Hz Butterworth low-pass) gives velocity in rad/s, and the absolute value gives speed. (4) The speed trace is linearly interpolated onto the 100 per-trial right bin edges. Discretisation happens later, globally (7-c).

ii.
```python
    wf = np.isfinite(wt) & np.isfinite(wp)
    wt0, wp0 = np.asarray(wt[wf]), np.asarray(wp[wf])
    _, unique_idx = np.unique(wt0, return_index=True)
    unique_idx.sort(); wt0, wp0 = wt0[unique_idx], wp0[unique_idx]
    if len(wt0) < 20 or np.any(np.diff(wt0) <= 0):
        raise ValueError('wheel timestamps cannot be made strictly increasing')
    pos_i, time_i = wheel_utils.interpolate_position(wt0, wp0, freq=1000, kind='linear')
    vel_i, _ = wheel_utils.velocity_filtered(pos_i, fs=1000)
    speed_i = np.abs(vel_i)
```
```python
    targets = stim[:, None] + TIME_REL[None, :]
    for i in np.flatnonzero(coverage & np.isfinite(stim)):
        wheel_vals[i] = np.interp(targets[i], time_i, speed_i).astype(np.float32)
```

iii. Step 7 "Edge Case Found and Fixed": *"The first seeded session initially failed because its wheel timestamps contain one exact duplicate at index 80. Whole-session exclusion was unjustified. Fixed by retaining the first finite sample at each duplicate timestamp before reference interpolation."* The 1 kHz + filtered-derivative recipe is described as "Exact IBL reference primitives" in the code comment.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three classes at **pooled** tertiles: after every session has been converted, all retained aligned wheel samples from all sessions are concatenated, `np.quantile(..., [1/3, 2/3])` gives two global thresholds (0.0139 and 0.3776 rad/s), and each sample is assigned with `np.searchsorted(..., side='right')`. The thresholds are stored in metadata. Globally the classes are exact thirds (1,849,167 / 1,849,166 / 1,849,167); per session the largest class fraction ranges up to 0.58, so no session is degenerate.

ii.
```python
    wheel_all = np.concatenate([s['wheel'].ravel() for s in sessions])
    wthr = np.quantile(wheel_all, [1/3, 2/3]).astype(float)
...
            out[2]=np.searchsorted(wthr,w,side='right').astype(np.int8)
```
```python
        'discretization':'pooled empirical tertiles over retained aligned bins; np.searchsorted(side=right)',
```

iii. Step 5 Decision 6: *"Use pooled empirical tertiles computed from all valid finite samples after alignment … Thresholds are fixed once computed and stored in metadata. Quantile bins create approximately balanced classes and are deterministic … To avoid domination by high-frame-rate sessions, each aligned time bin contributes once; because every retained trial has 100 bins, sessions contribute according to valid trial count as in decoder training."*

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The 1 kHz speed trace is evaluated at `stimOn_times[k] + TIME_REL`, the right edges of the very bins the spikes were counted in, so column *i* of the wheel row and column *i* of the neural matrix describe the same interval. Only trials whose full target vector lies inside the stream's time range are used; the rest are marked uncovered and dropped by the trial mask, so nothing is extrapolated.

ii.
```python
    targets = stim[:, None] + TIME_REL[None, :]
    coverage = ((targets[:, 0] >= time_i[0]) & (targets[:, -1] <= time_i[-1]) &
                (targets[:, 0] >= ct[0]) & (targets[:, -1] <= ct[-1]))
    ...
        wheel_vals[i] = np.interp(targets[i], time_i, speed_i).astype(np.float32)
    coverage &= np.isfinite(wheel_vals).all(1) & np.isfinite(whisk_vals).all(1)
```

iii. Wheel timestamps are on the same session clock as spikes, so evaluating at the bin grid is all the alignment needed; the coverage test implements the reference's "no extrapolation" rule (Step 5 Decision 11: *"Never extrapolate behavior. Drop individual trials without full window coverage"*). `--show-processing` plots put wheel, whisker and population spike count on one stimulus-relative axis to make a misalignment visible.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`; if the left pair is missing, unequal in length, too short or non-monotonic, the right camera is used instead. If neither is usable the whole session is dropped (5 of 139 candidates). The camera actually used is recorded per session (131 left, 3 right).

ii.
```python
    camera = None
    for side in ('left', 'right'):
        try:
            ct = np.load(choose_file(alf, f'_ibl_{side}Camera.times.npy'), mmap_mode='r')
            cv = np.load(choose_file(alf, f'{side}Camera.ROIMotionEnergy.npy'), mmap_mode='r')
            if len(ct) == len(cv) and len(ct) > 20:
                ...
                    camera = side
                    break
        except FileNotFoundError:
            continue
    if camera is None:
        raise ValueError('no complete left or right whisker motion-energy stream')
```

iii. Step 4: *"Retain left-first/right-fallback behavior and exclude the 14 sessions lacking both; record camera used per session"*, following the reference utility (Step 1: *"The generic code prefers left whisker motion energy and falls back to right if left is unavailable"*). Step 3 records the definition from the paper: mean absolute adjacent-frame difference inside the whisker-pad ROI, 60 Hz left / 150 Hz right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, no normalisation, no cross-camera rescaling. Only non-finite samples are dropped and duplicate frame times collapsed (the same robustness applied to the wheel), then the trace is linearly interpolated onto the 100 per-trial right bin edges. Discretisation is applied afterwards, globally.

ii.
```python
                cf = np.isfinite(ct) & np.isfinite(cv)
                ct0, cv0 = np.asarray(ct[cf]), np.asarray(cv[cf])
                _, ci = np.unique(ct0, return_index=True)
                ci.sort(); ct0, cv0 = ct0[ci], cv0[ci]
```
```python
        whisk_vals[i] = np.interp(targets[i], ct, cv).astype(np.float32)
```

iii. Step 5 mapping table: *"Left-first/right-fallback; interpolate/bin-average to grid; discretize globally into low/medium/high … no cross-camera normalization within a session beyond quantile transformation."*

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: **pooled** tertiles over every retained aligned sample of every session (thresholds 3.071 and 8.169 in the cameras' arbitrary motion-energy units), assigned by `np.searchsorted(side='right')`, stored in metadata. Globally the three classes are exact thirds. Per session they are not: from the validator's per-session fractions, 20 of 134 sessions contain essentially none of at least one class, 10 sessions have a single class over 80% of bins, and one session is 100% class "low" (its output row 3 has range [0, 0]).

ii.
```python
    whisk_all = np.concatenate([s['whisk'].ravel() for s in sessions])
    qthr = np.quantile(whisk_all, [1/3, 2/3]).astype(float)
    print('wheel tertiles',wthr,'whisker tertiles',qthr,flush=True)
...
            out[3]=np.searchsorted(qthr,q,side='right').astype(np.int8)
```

iii. Same rationale as 7-c (Step 5 Decision 6: pooled, deterministic, "independent of output labels"). Step 9's consistency table reports only the global counts — *"1,849,167 / 1,849,166 / 1,849,167 timepoints — Exact pooled tertiles"* — and the per-session degeneracy is not examined anywhere in the notes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The camera trace is evaluated at the same `stimOn_times[k] + TIME_REL` grid as the wheel and the spike bins, on the shared session clock. The same endpoint-coverage test gates the trial, so no camera value is extrapolated beyond the recorded stream.

ii.
```python
    coverage = ((targets[:, 0] >= time_i[0]) & (targets[:, -1] <= time_i[-1]) &
                (targets[:, 0] >= ct[0]) & (targets[:, -1] <= ct[-1]))
    ...
        whisk_vals[i] = np.interp(targets[i], ct, cv).astype(np.float32)
```

iii. Step 10 Check 10 (behaviour evaluated at exact right edges as in reference interpolation) and Step 5 Decision 11 (no extrapolation). Step 10's independent raw-output check re-derived the camera interpolation for three trials outside the conversion code and matched the stored categorical series exactly.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Layered, and always by dropping rather than imputing:
- **Trials**: any non-finite required event, bad event order, out-of-range reaction time or trial duration, invalid choice/prior, or a window not covered by wheel and camera → trial dropped, with per-reason counts stored.
- **Streams**: non-finite samples removed and duplicate timestamps collapsed to the first occurrence (wheel and camera); if the wheel still cannot be made strictly increasing, or is shorter than 20 samples, or wheel/position lengths disagree, the session is dropped.
- **Sessions**: missing session directory, no usable whisker camera, no good grey-matter unit, or fewer than 2 valid trials → session skipped, recorded in `metadata['skipped_sessions']` with the exception text (5 skips, all missing-whisker).
- **Clusters**: an out-of-range peak-channel index makes a unit "anatomically not OK" and it is dropped; spike cluster ids outside the lookup table are ignored.
- **Probes**: a listed probe directory that does not exist, or that yields zero good units, is silently skipped; duplicate neuron UUIDs after a probe merge raise.
Nothing is ever extrapolated or filled.

ii.
```python
    if len(wt) != len(wp) or len(wt) < 20:
        raise ValueError('invalid wheel timestamps/position lengths')
    ...
    if len(wt0) < 20 or np.any(np.diff(wt0) <= 0):
        raise ValueError('wheel timestamps cannot be made strictly increasing')
```
```python
    anatomical_ok = (channels >= 0) & (channels < len(channel_ids))
    ...
    in_lookup = c < len(p['lookup'])
```
```python
        except Exception as e:
            skipped.append({'eid': item['eid'], 'subject': item['subject'],
                            'reason': f'{type(e).__name__}: {e}'})
            print('  SKIP', skipped[-1]['reason'], flush=True)
```

iii. Step 5 Decision 11: *"Never extrapolate behavior. Drop individual trials without full window coverage; drop sessions with fewer than two valid trials or no good neurons. Do not synthesize values."* Step 7/10 document the duplicate-wheel-timestamp fix in detail and explicitly argue that whole-session exclusion for one duplicated sample "was unjustified".

## 10-a. What are the most time-consuming steps of the code?

i. In practice the per-session work is dominated by file I/O: paging the memory-mapped `spikes.times`/`spikes.clusters` arrays (tens of millions of rows per probe) and the 1 kHz `interpolate_position` + `velocity_filtered` pass over the whole session's wheel. Measured per-session times in `conversion_full_out.txt` are 0.2–1.4 s, so the single largest wall-clock item in full mode is actually writing the 3.3 GB pickle at the end. Repeated `Path.rglob` scans of each session's `alf` tree (once per file looked up) and the per-session re-read of `bwm_release.csv` add a small constant.

ii.
```python
    st = np.load(parent / 'spikes.times.npy', mmap_mode='r')
    sc = np.load(parent / 'spikes.clusters.npy', mmap_mode='r')
```
```python
    pos_i, time_i = wheel_utils.interpolate_position(wt0, wp0, freq=1000, kind='linear')
    vel_i, _ = wheel_utils.velocity_filtered(pos_i, fs=1000)
```
```python
    with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
```

iii. Step 6: *"Spike arrays can exceed 50 million rows/probe; loading full arrays or looping over every spike/trial would be prohibitive"*; mitigations were *"Memory-map spike arrays and use `searchsorted` to read only each trial window"* and *"Process/release one session at a time"*. Step 7's timing table concludes *"Session processing is sub-second for sample sessions despite tens of millions of source spikes … pickle I/O may dominate."*

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four per-trial Python loops remain:
- `bin_spikes` loops over trials (and over probes inside), allocating a fresh `(n_neurons, 100)` matrix and calling `searchsorted`/`bincount` per trial. All trials of a probe could be binned with one `bincount` by folding the trial index into the flat index.
- `load_behavior` loops over covered trials calling `np.interp` twice per trial; a single `np.interp` over the flattened `targets` array would do the whole session at once.
- the input-construction loop uses `selected.iterrows()` (slow pandas row objects) and rebuilds the same `TIME_REL` row 55,475 times.
- the output-assembly loop in `main` builds one `(4,100)` int8 array per trial instead of thresholding each session's `wheel`/`whisk` matrices in one `searchsorted` call.
None of these were vectorised; at 0.2–1.4 s per session there was little to gain, and the AI's own inefficiency list does not mention them.

ii.
```python
    for k, stim in enumerate(stim_times):
        mat = np.zeros((n_neurons, N_BINS), dtype=np.float32)
        for p, off in zip(probes, offsets):
```
```python
    for i in np.flatnonzero(coverage & np.isfinite(stim)):
        wheel_vals[i] = np.interp(targets[i], time_i, speed_i).astype(np.float32)
        whisk_vals[i] = np.interp(targets[i], ct, cv).astype(np.float32)
```
```python
    for raw_i, (_, tr) in zip(trial_idx, selected.iterrows()):
```
```python
        for (choice,prior), w, q in zip(s['scalar'],s['wheel'],s['whisk']):
            out=np.empty((4,N_BINS),dtype=np.int8)
```

iii. Step 6 claims the hot inner work *was* vectorised — *"Vectorize cluster lookup, time-bin assignment, and count accumulation with `np.bincount`"* — which is accurate at the within-trial level; the remaining trial-level loops are not identified as opportunities anywhere in the notes.

## 10-c. What processing does the code repeat multiple times?

i. Three things. `bwm_release.csv` is parsed once in `select_reference_sessions` and then **again inside `process_session` for every session** (134 extra full CSV reads) just to look up that session's probe names. `choose_file` runs a recursive `rglob` over the session's `alf` tree separately for each of the seven files it fetches, re-walking the same directories. And `load_behavior` interpolates wheel and whisker for every trial whose window is covered, including the many trials that the trial mask will discard immediately afterwards. All three are cheap relative to spike I/O, and none is documented.

ii.
```python
def process_session(item, br):
    ...
    # Use exactly the probe insertions listed in the frozen release for this EID.
    bwm = pd.read_csv(BWM_CSV, index_col=0)
    probe_names = bwm.loc[bwm.eid.astype(str) == eid, 'probe_name'].astype(str).tolist()
```
```python
    candidates = [p for p in root.rglob(basename) ...]
```
```python
    for i in np.flatnonzero(coverage & np.isfinite(stim)):   # coverage, not the final trial mask
        wheel_vals[i] = np.interp(targets[i], time_i, speed_i).astype(np.float32)
```

iii. Not discussed in CONVERSION_NOTES.md; the efficiency discussion (Step 6) covers only memory-mapping, per-trial windowing and one-session-at-a-time memory use.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several bookkeeping products that the decoder never reads: `clusters.uuids.csv` is loaded and a full per-neuron UUID list (~21,400 strings) is built and stored in `metadata['session_info']`, solely for a duplicate-identity assertion; the whole `trial_indices` array, `trial_exclusion_counts`, `probe_sources`, `trial_source` and `elapsed_s` are stored per session; `time_bin_right_edges_s` duplicates information already implied by `time_bin_size`/`off_start`/`off_end`; and the behaviour interpolation is computed for trials later dropped (see 10-c). Together these inflate the metadata but are harmless. `--show-processing` plotting is also pure diagnostics, though it is explicitly required by the instructions.

ii.
```python
    uuids_p = parent / 'clusters.uuids.csv'
    if uuids_p.exists():
        u = pd.read_csv(uuids_p, header=None).iloc[:, -1].astype(str).to_numpy()
        uuids = u[good] if len(u) == len(good) else np.array([f'{parent}:{x}' for x in good_cids])
```
```python
        'neuron_uuids': uuids.tolist(), 'elapsed_s': time.time()-t0,
```
```python
        'time_bin_right_edges_s':TIME_REL.tolist(),
```

iii. The AI justifies the UUIDs as an identity check after probe merging (Step 5 planned sanity check: *"Check no duplicated neuron UUIDs after probe merging"*, verified in Step 10 Check 6) and the rest as provenance (Step 5 Decision 12: *"Record release, EIDs/subjects, camera side, original/retained counts, unit counts, exclusion reasons, bin edges/centers, discretization thresholds, choice/prior mappings, QC rules, and source/reference versions."*). It does not flag any of it as unnecessary.
