# Decisions

> Scope note: the AI's pipeline never completed. Its final `--sample` run skipped all
> 459 sessions with `trials object missing columns [...]`, raised
> `RuntimeError: no sessions converted`, and produced no `sample_data.pkl` or
> `converted_data.pkl`. Steps 7–13 of the workflow were never started, and
> `CONVERSION_NOTES.md` Steps 7–13 are still empty templates. Every decision below is
> therefore a *documented intent plus unexecuted code*, never validated against data.
> The root cause was a mis-set ONE client (`mode='local'`), not missing data: the
> required datasets are on disk under revision folders (e.g.
> `alf/#2025-03-03#/_ibl_trials.table.pqt`), which a local-only client does not resolve.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The session/probe inventory is taken by reading the reference repo's release manifest
`/app/code/code_zhang2025/data/bwm_release.csv` directly with pandas (459 eids, 699
probe rows), optionally intersected with `/app/data/DATALIMIT_SUBSET.csv` if present.
`one.search` is not used. All neuroscience arrays are then loaded through ONE/brainbox:
a single `ONE` client is built with `cache_dir=/app/data/one_cache`,
`tables_dir=.../Brainwidemap`, `mode='local'`; per session a `SessionLoader` reads
trials, wheel and camera motion energy, and per probe row a `SpikeSortingLoader` reads
spikes/clusters/channels. Sessions are iterated serially in a single process inside one
`try/except`; any exception skips the whole session and is recorded in
`metadata['failed_sessions']`.

In practice this loads nothing. A `mode='local'` client resolves datasets against the
pre-revision paths listed in the release parquet tables, while the files on disk live in
dated revision folders, so `SessionLoader.load_trials()` returned only the one
non-revisioned legacy array (`goCueTrigger_times`) for every session. The AI recorded
this as a "blocking source-cache defect" rather than a client-configuration problem.

ii.
```python
CACHE = Path('/app/data/one_cache')
RELEASE = Path('/app/code/code_zhang2025/data/bwm_release.csv')
SUBSET = Path('/app/data/DATALIMIT_SUBSET.csv')

def get_one():
    """Construct a local ONE client over the aggregate release tables."""
    return ONE(cache_dir=CACHE, tables_dir=CACHE / 'Brainwidemap', mode='local')

def release_rows():
    rows = pd.read_csv(RELEASE).drop(columns=['Unnamed: 0'], errors='ignore')
    if SUBSET.exists():
        subset = pd.read_csv(SUBSET)
        key = next((x for x in ('eid', 'session', 'session_id') if x in subset), None)
        if key:
            rows = rows[rows.eid.astype(str).isin(subset[key].astype(str))]
    return rows
```
```python
    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
```
```python
        ssl = SpikeSortingLoader(pid=str(row.pid), eid=str(row.eid), pname=row.probe_name, one=one)
        spikes, clusters, channels = ssl.load_spike_sorting()
        clusters = ssl.merge_clusters(spikes, clusters, channels, compute_metrics=False)
```
```python
    rows = release_rows(); one = get_one()
    grouped = list(rows.groupby('eid', sort=False))
    for k, (eid, erows) in enumerate(grouped, 1):
        try:
            vals = process_session(one, str(eid), erows, ...)
        except Exception as exc:
            failures.append((str(eid), f'{type(exc).__name__}: {exc}'))
```

iii. From CONVERSION_NOTES.md Step 4: "`bwm_release.csv`: 699 probes, 459 EIDs, 139
subjects … Use the exact CSV EIDs and probe membership; ONE/brainbox perform all actual
loading." The AI wanted the session set to be exactly the paper's release rather than
whatever the cache tables happen to list. On the failure it wrote: "ONE's cache table
nevertheless reports that table with `exists=True` … required source datasets are absent
from the staged filesystem despite metadata claiming presence… An authenticated remote
ONE attempt was also unavailable because outbound access to OpenAlyx was refused." The
trajectory shows it tried a fresh remote `ONE(base_url=..., password=...)` with a *new*
temp cache dir (step 50), which had no cached REST responses and so needed the network;
it never retried `mode='remote'` against the existing cache's `.rest` directory.

## 1-b. How are the data split into subjects?

i. Not derived from the data at all: the subject name is read from the `subject` column
of `bwm_release.csv` for the session's first probe row. At assembly the unique subjects
are sorted and `subject_idx` indexes each session into that list.

ii.
```python
    info = dict(eid=str(eid), subject=str(rows.subject.iloc[0]), n_trials=len(idx), ...)
```
```python
    subjects = sorted({x['subject'] for x in infos}); subject_lookup = {s:i for i,s in enumerate(subjects)}
    ...
    'subject_idx': np.asarray([subject_lookup[x['subject']] for x in infos], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 4 treats the release CSV as authoritative for release
membership (459 sessions / 139 subjects), so the subject label comes from the same table
rather than being parsed from paths.

## 1-c. How are the data split into sessions?

i. A session is one `eid`; the release CSV is grouped by `eid` (`groupby('eid',
sort=False)`), which also gathers that session's probe rows so both insertions of a
two-probe session are merged into one population. Session order in the output follows the
CSV row order. A session is dropped if any step raises, or if fewer than 2 trials survive.

ii.
```python
    grouped = list(rows.groupby('eid', sort=False))
    target = 2 if args.sample else None
    for k, (eid, erows) in enumerate(grouped, 1):
        vals = process_session(one, str(eid), erows, ...)
```
```python
    if idx.size < 2:
        raise RuntimeError(f'only {idx.size} jointly valid trials')
```

iii. Step 5: "all release EIDs, with all simultaneously recorded probes merged per EID";
Step 4 notes the reference decodes 433 of the 459 sessions, so the AI expected ~433 to
survive. The ≥2-trial rule comes from the target-format requirement that each session have
at least two trials.

## 1-d. How are the data split into trials?

i. The split is taken as given: one row of the `SessionLoader` trials table is one trial,
and each trial becomes the window `stimOn_times + [-0.5, +1.5) s`. No re-derivation of
trial boundaries from `intervals`.

ii.
```python
    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
    tr = sl.trials.copy()
```
```python
    onsets = tr.stimOn_times.to_numpy(dtype=float)
```

iii. Implicit in the Step 5 mapping table, which sources every per-trial variable from
`trials.*` columns; the reference code does the same via `load_trials_and_mask`.

## 1-e. How are trials filtered based on quality controls?

i. Five conditions, ANDed:
(1) finiteness of `stimOn_times`, `choice`, `probabilityLeft`, `firstMovement_times`,
`feedback_times`, `feedbackType`;
(2) reaction time `firstMovement_times - stimOn_times` in [0.08, 2.0] s;
(3) trial duration `feedback_times - goCue_times` ≤ 10 s;
(4) `choice != 0` (no-response trials dropped);
(5) full wheel and whisker coverage of the trial window, tested only against the *global*
first/last sample of each stream plus a finiteness check on the interpolated values.
The unbiased first 90 trials (prior 0.5) are retained. A session with <2 surviving trials
is dropped.

ii.
```python
    finite = np.ones(len(tr), dtype=bool)
    for col in ['stimOn_times', 'choice', 'probabilityLeft', 'firstMovement_times',
                'feedback_times', 'feedbackType']:
        finite &= np.isfinite(tr[col].to_numpy(dtype=float))
    rt = tr.firstMovement_times.to_numpy() - tr.stimOn_times.to_numpy()
    duration = tr.feedback_times.to_numpy() - tr.goCue_times.to_numpy()
    mask = finite & (rt >= .08) & (rt <= 2.) & (duration <= 10.) & (tr.choice.to_numpy() != 0)
```
```python
def interpolate_trials(times, values, onsets):
    query = onsets[:, None] + TIME[None, :]
    good = ((query[:, 0] >= times[0]) & (query[:, -1] <= times[-1]))
    out = np.full(query.shape, np.nan, dtype=np.float32)
    if np.any(good):
        out[good] = np.interp(query[good].ravel(), times, values).reshape((-1, N_TIME))
    return out, good & np.isfinite(out).all(axis=1)
```
```python
    valid = mask & wheel_good & whisk_good
    idx = np.flatnonzero(valid)
    if idx.size < 2:
        raise RuntimeError(f'only {idx.size} jointly valid trials')
```

iii. Step 1 of the notes identifies `load_trials_and_mask` as the reference curation
function and lists its rules; Step 3 quotes the data paper: "trials were excluded if one
of the following trial events could not be detected: choice, probabilityLeft,
feebackType, feeback times, stimON times and firstMovement times… outside the range of
0.08–2.00 s." Step 5 Key Decision 2: "Apply reference event/RT/no-choice/max-duration QC
and additionally require finite coverage of both requested dynamic outputs across the
full window." The 10 s cap is `prepare_data(..., max_trial_len=10.0)` in the reference.
The AI also noted a bug in the reference (`target_mask = target_mask and beh_mask`) and
decided to "preserve intended semantics … not reproduce an implementation defect."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from every probe of the session, via
`SpikeSortingLoader.load_spike_sorting`. `clusters['acronym']` (after
`merge_clusters`) supplies the anatomy for `brain_region_idx`; `clusters['channels']`
supplies the per-probe cluster count used to offset cluster ids when merging probes.
Cluster quality metrics are loaded but not used.

ii.
```python
def load_spikes(one, probe_rows):
    all_times, all_clusters, all_regions = [], [], []
    offset = 0
    for row in probe_rows.itertuples():
        ssl = SpikeSortingLoader(pid=str(row.pid), eid=str(row.eid), pname=row.probe_name, one=one)
        spikes, clusters, channels = ssl.load_spike_sorting()
        clusters = ssl.merge_clusters(spikes, clusters, channels, compute_metrics=False)
        n = len(clusters['channels'])
        all_times.append(np.asarray(spikes['times'], dtype=float))
        all_clusters.append(np.asarray(spikes['clusters'], dtype=np.int64) + offset)
        all_regions.extend(np.asarray(clusters['acronym']).astype(str).tolist())
        offset += n
```

iii. Step 1 of the notes: `prepare_data` "resolves all probes with `one.eid2pid`, loads
each using `SpikeSortingLoader`, merges probes"; Step 5 maps "`spikes.times`,
`spikes.clusters` from every release probe → `neural`".

## 2-b. How is the `neural` data processed?

i. Probes are pooled into one population by offsetting each probe's cluster ids by the
running cluster count, then all spikes are sorted by time (stable sort) so that trial
windows can be sliced with `searchsorted`. For each retained trial the spikes in
`[stimOn-0.5, stimOn+1.5)` are histogrammed into 100 half-open 20 ms bins with a single
flattened `np.bincount`, giving a `(n_neurons, 100)` matrix of **raw spike counts**
(stored float32). No conversion to Hz, no smoothing, no z-scoring, no baseline
subtraction, no neuron-level subselection.

ii.
```python
    times = np.concatenate(all_times); clu = np.concatenate(all_clusters)
    order = np.argsort(times, kind='stable')
    return times[order], clu[order], np.asarray(all_regions, dtype=str)
```
```python
def bin_spikes(times, clusters, onsets, n_neurons):
    result = []
    for onset in onsets:
        beg, end = onset + OFF_START, onset + OFF_END
        lo, hi = np.searchsorted(times, [beg, end])
        relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
        ok = (relbin >= 0) & (relbin < N_TIME)
        flat = clusters[lo:hi][ok] * N_TIME + relbin[ok]
        counts = np.bincount(flat, minlength=n_neurons * N_TIME).reshape(n_neurons, N_TIME)
        result.append(counts.astype(np.float32))
    return result
```
```python
    'neural_measure': 'spike counts per 20-ms bin',
```

iii. Step 1: "`bin_spiking_data` / `get_spike_data_per_interval` … Builds
stimulus-aligned windows and bins spike counts with half-open time intervals and
`bincount2D`"; Step 5: "Merge probes, count spikes in 100 half-open 20-ms bins spanning
stimOn−0.5 to stimOn+1.5; transpose to neuron×time; float32". Merging probes is justified
from the data paper: probes in one session "are not independent".

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not. Every spike-sorted cluster is kept — no `label >= 1` cut, no amplitude /
refractory / noise-cutoff criteria, no removal of `void` (outside-brain) or `root`
units, no minimum firing rate, no minimum spike count. Region labels are mapped to Beryl
purely for naming; units whose Beryl acronym is `void` or `root` stay in the population.
Expected consequence: ~1,354 clusters/session (621,733 total) instead of the ~165/session
(75,708) that the data paper's stringent QC yields.

ii.
```python
        spikes, clusters, channels = ssl.load_spike_sorting()
        clusters = ssl.merge_clusters(spikes, clusters, channels, compute_metrics=False)
        n = len(clusters['channels'])
        all_clusters.append(np.asarray(spikes['clusters'], dtype=np.int64) + offset)
        all_regions.extend(np.asarray(clusters['acronym']).astype(str).tolist())
```
```python
    mapped = [br.acronym2acronym(x, mapping='Beryl') for x in region_names_all]
    vocabulary = sorted(set(np.concatenate(mapped).astype(str)))
```
```python
    'cluster_filter': 'all spike-sorted clusters, matching Zhang et al. caching code',
```

iii. Step 1: "Although quality labels are saved as metadata, the published caching call
does **not** filter to good clusters (`qc=None`). Therefore all loaded sorted clusters
are the reference choice." Step 4 resolves the conflict explicitly: "Method paper says
all neurons; data paper primary analyses emphasize 75,708 good neurons → Follow decoder
caching pipeline: all clusters. This explains expected 621,733-scale rather than
75,708-scale count." methods.txt supports this literally: "we bin spike counts using all
neurons, sorted by Kilosort 2.5, from each session."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams share one session clock, so alignment is a subtraction: for each
retained trial the window is `[stimOn_times - 0.5, stimOn_times + 1.5)` and bin index is
computed relative to the window start. `metadata['temporal_alignment_event'] = 'visual
stimulus onset (stimOn_times)'`, `off_start=-0.5`, `off_end=1.5`. No resampling or clock
conversion is applied.

ii.
```python
OFF_START, OFF_END = -0.5, 1.5
```
```python
        beg, end = onset + OFF_START, onset + OFF_END
        lo, hi = np.searchsorted(times, [beg, end])
        relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
```
```python
            'off_start': OFF_START, 'off_end': OFF_END, 'bin_coordinate': 'right edge',
            'temporal_alignment_event': 'visual stimulus onset (stimOn_times)',
```

iii. Step 4: "Downstream mandates stimulus alignment/common bins; use cache-script common
window." Step 3 notes the method paper uses different windows per target variable
(−0.6..−0.1 s for prior, first-movement-aligned for the dynamic behaviours) but
concludes: "The downstream task explicitly requires stimulus-onset alignment and common
timepoints, so the caching implementation's common stimulus window is the applicable
reference."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per 2 s trial, identical for every trial and session;
`metadata['time_bin_size'] = 20.0` (ms). Spikes are binned once directly at 20 ms — there
is no finer intermediate representation and therefore no rebinning. The behavioural
streams (wheel at 1000 Hz after `SessionLoader` interpolation, camera at 60/150 Hz) are
*resampled by linear interpolation* onto the same 100-point grid rather than averaged
within bins.

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
N_TIME = 100
TIME = np.arange(1, N_TIME + 1, dtype=np.float32) * BIN + OFF_START
```
```python
            'time_bin_size': 20.0, ...
```

iii. Step 3: "Neural data time bin | 20 ms for supplied decoder cache | Method paper: 2-s
trials divided into 20-ms bins (100 steps)." Step 5 Key Decision 1: "Use exactly −0.5 to
+1.5 s and 20 ms (100 bins), matching supplied cache code and decoder-task alignment."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From no raw variable directly — it is the analysis grid defined by the alignment on
`trials.stimOn_times` and the fixed window/bin size. One fixed 100-element vector, reused
identically for every trial and session, holding the **right edge** of each bin:
−0.48, −0.46, …, +1.50 s. Named `time_since_stimulus_onset`, stored float32.

ii.
```python
TIME = np.arange(1, N_TIME + 1, dtype=np.float32) * BIN + OFF_START
```
```python
        inp = np.vstack([TIME, np.full(N_TIME, block_no[raw_i], dtype=np.float32)])
```
```python
        'input_names': ['time_since_stimulus_onset', 'trial_number_in_block'],
```

iii. Step 5 mapping: "Bin right-edge offsets → `input[0]` time since stimulus onset
`[-0.48, ..., 1.50]` s repeated identically per trial … using reference interpolation
coordinates ensures exact stream alignment", citing `get_behavior_per_interval`, whose
grid is `np.linspace(interval_beg + binsize, interval_end, n_bins)`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid once at module import. It is kept as a continuous
signed time in seconds (not a binary onset indicator), which is what the decoder-task
spec asks for ("Time since stimulus onset, continuous, time-varying").

ii.
```python
TIME = np.arange(1, N_TIME + 1, dtype=np.float32) * BIN + OFF_START
# -> array([-0.48, -0.46, ..., 1.48, 1.50], dtype=float32)
```

iii. Step 5 Key Decision 5: "Time coordinate: Use bin right edges as the reference
behavior code does; neural counts cover corresponding preceding half-open bins."

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. By construction: bin *k* of the neural matrix covers
`[stimOn - 0.5 + 0.02k, stimOn - 0.5 + 0.02(k+1))` and `TIME[k]` is the closing edge of
that same interval, so index *k* of the input and index *k* of the neural matrix refer to
the same 20 ms of the same trial. The label is offset by +10 ms from the bin's centre
(the human reference uses centres), and the behavioural outputs are sampled at exactly
these same right edges, so the input/output/neural indices remain mutually consistent.

ii.
```python
        relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)   # neural bin index
```
```python
    query = onsets[:, None] + TIME[None, :]                              # behaviour sampled at TIME
```
```python
            'bin_coordinate': 'right edge',
```

iii. Step 5 Key Decision 5 (above) and the Step 5 mapping note "using reference
interpolation coordinates ensures exact stream alignment".

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft` alone. The trials table carries no block id, so a block
boundary is inferred wherever `probabilityLeft` changes from the previous trial.

ii.
```python
    block_number = trial_number_in_block(tr.probabilityLeft.to_numpy())
```
```python
def trial_number_in_block(prob):
    out = np.zeros(len(prob), dtype=np.float32)
    for i in range(1, len(prob)):
        out[i] = out[i - 1] + 1 if prob[i] == prob[i - 1] else 0
    return out
```

iii. Step 5 mapping: "`trials.probabilityLeft` run position → `input[1]` trial number in
block | Zero-based count within each consecutive prior block | planned run-length
transform".

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based running counter that resets whenever `probabilityLeft` changes, computed
with a scalar Python loop over **all** trials of the session *before* the QC mask is
applied, so a trial that is later dropped still advances the counter and the surviving
trials keep their true position in the block. The per-trial scalar is then broadcast
across all 100 time bins as a float32 constant row of `input`.

ii.
```python
    block_number = trial_number_in_block(tr.probabilityLeft.to_numpy())   # before masking
    ...
    return sl, tr, mask, block_number
```
```python
    for j, raw_i in enumerate(idx):                 # raw_i indexes the unfiltered table
        inp = np.vstack([TIME, np.full(N_TIME, block_no[raw_i], dtype=np.float32)])
```

iii. Step 5 mapping note: "Computed before trial filtering so excluded trials do not
collapse experimental block position." Broadcasting over time follows Key Decision 4:
"Repeat choice and prior across 100 bins because the target container must have a uniform
`n_output × n_timepoints` shape."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table (+1 / −1 / 0). Trials with `choice == 0`
(no response) are removed by the QC mask, so only ±1 reaches the encoder.

ii.
```python
    mask = finite & (rt >= .08) & (rt <= 2.) & (duration <= 10.) & (tr.choice.to_numpy() != 0)
```
```python
        choice = 0 if tr.choice.iloc[raw_i] == -1 else 1
```

iii. Step 5 mapping: "`trials.choice` → `output[0]` choice … No-choice 0 excluded."

## 5-b. What processing is involved in computing `output` *Choice*?

i. A binary recode broadcast over the 100 bins, stored int64, with
`output_values[0] = ['left', 'right']`. The mapping used is **−1 → 0 ("left") and
+1 → 1 ("right")**. This is inverted with respect to the IBL convention: in ibllib,
`choice == -1` is a CCW wheel turn that reports the stimulus on the **right**
(`brainbox/behavior/training.py`: `rightward = trials.choice == -1`, and the comment
"choice == -1 means contrast on right hand side"), while `+1` reports **left**. The
human reference uses `{1.0: 0 (left), -1.0: 1 (right)}`. The AI's labels are therefore
swapped relative to the instruction's "left = 0, right = 1". Because the code has no
`else`-guard, a `choice` value that is neither −1 nor 0 falls through to 1.

ii.
```python
        choice = 0 if tr.choice.iloc[raw_i] == -1 else 1
        prior = prior_map[round(float(tr.probabilityLeft.iloc[raw_i]), 1)]
        out = np.vstack([np.full(N_TIME, choice), np.full(N_TIME, prior),
                         wheel_cat[j], whisk_cat[j]]).astype(np.int64)
```
```python
        'output_values': [['left', 'right'], ['0.2', '0.5', '0.8'], ['low', 'medium', 'high'], ['low', 'medium', 'high']],
```

iii. Step 5 mapping states the belief explicitly: "`trials.choice` → `output[0]` choice |
IBL −1 (left choice)→0, +1 (right choice)→1, repeated over time". The trajectory shows no
check of the sign convention against the papers, the ibllib source, or the data; the
reading appears to come from the extractor docstring "−1 is a CCW turn (towards the
left)", which describes wheel direction, not the reported stimulus side.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, used unmodified (not a
model-inferred subjective prior). Trials with non-finite `probabilityLeft` are dropped by
the QC mask; the unbiased 0.5 block is retained as its own class.

ii.
```python
        finite &= np.isfinite(tr[col].to_numpy(dtype=float))   # incl. probabilityLeft
```
```python
        prior = prior_map[round(float(tr.probabilityLeft.iloc[raw_i]), 1)]
```

iii. Step 5 mapping: "`trials.probabilityLeft` → `output[1]` prior probability of left |
0.2→0, 0.5→1, 0.8→2 | Exact categorical mapping." Step 3/Step 1 note the unbiased first
90 trials are retained ("It retains the initial 0.5-prior block").

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Round to one decimal, look up in `{0.2: 0, 0.5: 1, 0.8: 2}`, broadcast the scalar over
100 bins as int64. No filtering on allowed prior values: an unexpected `probabilityLeft`
raises `KeyError`, which the outer handler converts into skipping the **entire session**.

ii.
```python
    prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
    for j, raw_i in enumerate(idx):
        ...
        prior = prior_map[round(float(tr.probabilityLeft.iloc[raw_i]), 1)]
```

iii. The rounding is defensive against float representation of 0.2/0.8. Step 5 records
the mapping as the instruction-mandated encoding ("Exact categorical mapping"); no
justification is given for the absence of a value whitelist.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()` — i.e. `_ibl_wheel.position` / `_ibl_wheel.timestamps`
turned by brainbox into an evenly sampled position and a filtered velocity. Wheel speed
is the absolute value of `sl.wheel.velocity`, on `sl.wheel.times`.

ii.
```python
def load_behavior(sl):
    sl.load_wheel()
    wt = sl.wheel.times.to_numpy(dtype=float)
    ws = np.abs(sl.wheel.velocity.to_numpy(dtype=float))
```

iii. Step 1: "`load_target_behavior` … wheel speed is absolute interpolated wheel
velocity"; Step 5 maps "absolute `SessionLoader.wheel.velocity` → `output[2]` wheel
speed". This matches the reference's `'wheel-speed'` target.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) Inside `SessionLoader.load_wheel`: interpolation of the
event-driven wheel position onto a uniform 1000 Hz grid and differentiation to velocity
with a 20 Hz Butterworth low-pass (brainbox defaults, untouched). (2) Absolute value, then
linear resampling (`np.interp`) onto `onset + TIME`, i.e. the 100 bin right edges of every
trial — computed for *all* trials of the session, including ones the QC mask will drop.
(3) Discretisation into 3 classes (see 7-c). No smoothing, normalisation or unit
conversion is applied.

ii.
```python
def interpolate_trials(times, values, onsets):
    query = onsets[:, None] + TIME[None, :]
    good = ((query[:, 0] >= times[0]) & (query[:, -1] <= times[-1]))
    out = np.full(query.shape, np.nan, dtype=np.float32)
    if np.any(good):
        out[good] = np.interp(query[good].ravel(), times, values).reshape((-1, N_TIME))
    return out, good & np.isfinite(out).all(axis=1)
```
```python
    wheel, wheel_good = interpolate_trials(wt, ws, onsets)   # onsets = ALL trials
    ...
    wheel_cat, wheel_edges = tertiles(wheel[idx])            # then restricted to kept trials
```

iii. Step 3: "Reference behavioral streams are linearly interpolated to bin right edges."
Step 1 identifies `get_behavior_per_interval` as the reference routine; the AI replaced
its per-interval `interp1d` + multiprocessing pool with one vectorised `np.interp` call,
listed under Step 6 "Code speedups added: … vectorized interpolation for every trial".

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 ordinal classes (`low`/`medium`/`high`) at the 1/3 and 2/3 **quantiles of that
session's own** pooled trial×time samples, restricted to the retained trials. Thresholds
are per session, not global and not per trial, so classes are ~equally sized within each
session. If the two quantiles coincide (degenerate/constant trace) the code falls back to
a deterministic stable-rank split into three equal-count groups. The realised edges are
recorded per session in `metadata['session_info'][i]['wheel_edges']`.

ii.
```python
def tertiles(x):
    edges = np.quantile(x, [1 / 3, 2 / 3])
    if edges[0] < edges[1]:
        return np.digitize(x, edges, right=False).astype(np.int64), edges
    # Deterministic rank fallback for degenerate signals.
    order = np.argsort(x.ravel(), kind='stable'); labels = np.empty(order.size, dtype=np.int64)
    labels[order] = np.minimum(2, np.arange(order.size) * 3 // order.size)
    return labels.reshape(x.shape), edges
```
```python
    wheel_cat, wheel_edges = tertiles(wheel[idx])
```
```python
    'dynamic_discretization': 'within-session tertiles over all retained trial-time samples',
```

iii. Step 5 Key Decision 3: "Tertiles are computed within session after joint-valid
masking, but before categorical conversion. This is robust to between-camera/rate
calibration differences and ensures all three requested categories are represented;
repeated quantile edges are handled deterministically with rank-based fallback." The
decoder task only says "discretized into 3 bins", so the rule is the AI's own choice.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is evaluated at `stimOn + TIME`, the same 100 right-edge offsets used
to index the neural bins, so output column *k* and neural column *k* are the same 20 ms
slot of the same trial. Both derive from the same `stimOn_times` on the shared session
clock, so no clock conversion is needed. The sample is an instantaneous value at the
bin's closing edge rather than a bin average/centre.

ii.
```python
    query = onsets[:, None] + TIME[None, :]
```
```python
        relbin = np.floor((times[lo:hi] - beg) / BIN).astype(np.int64)
```

iii. Step 5 Key Decision 5: "Use bin right edges as the reference behavior code does;
neural counts cover corresponding preceding half-open bins." The `--show-processing` plot
was intended to demonstrate this ("no temporal misalignments"), but it never ran.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `SessionLoader.load_motion_energy(views=[view])` → the `whiskerMotionEnergy` column and
its `times` for a side camera; the **left** camera is tried first and the **right** camera
is used only if the left load raises. The released ROI motion energy is used as-is (IBL
computes it as the mean absolute inter-frame difference in a whisker-pad box). The chosen
view is recorded per session as `metadata['session_info'][i]['whisker_view']`.

ii.
```python
    last_error = None
    for view in ('left', 'right'):
        try:
            sl.load_motion_energy(views=[view])
            df = sl.motion_energy[f'{view}Camera']
            return wt, ws, df.times.to_numpy(dtype=float), df.whiskerMotionEnergy.to_numpy(dtype=float), view
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'no whisker motion-energy view: {last_error}')
```

iii. Step 1: "whisker motion energy prefers left camera, falling back to right only on
load failure" — a direct transcription of the reference `bin_behaviors` logic
(`if 'skip' in target_dict.keys(): … 'right-whisker-motion-energy'`). Step 4: "Match
reference fallback behavior; never average views."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Identical to the wheel path minus the velocity step: the released trace is taken
unfiltered and un-normalised, linearly resampled with `np.interp` onto `onset + TIME`
for every trial, then discretised (see 8-c). No per-camera rescaling despite the 60 Hz
left / 150 Hz right sampling difference — the per-session tertile rule absorbs the
amplitude difference instead.

ii.
```python
    whisk, whisk_good = interpolate_trials(mt, me, onsets)
    ...
    whisk_cat, whisk_edges = tertiles(whisk[idx])
```

iii. Step 5 mapping: "left (fallback right) `SessionLoader.motion_energy.*.whiskerMotionEnergy`
→ `output[3]` whisker motion energy | Same interpolation and per-session tertile
discretization | Do not average camera views." Step 3 notes the raw rates (60 Hz left /
150 Hz right) are "interpolated to neural bins".

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Exactly the same `tertiles()` rule as the wheel: 1/3 and 2/3 quantiles of the pooled
retained trial×time samples of that session, `np.digitize` into 0/1/2 labelled
low/medium/high, with the stable-rank fallback for degenerate traces; edges stored as
`whisker_edges` per session.

ii.
```python
    whisk_cat, whisk_edges = tertiles(whisk[idx])
```
```python
def tertiles(x):
    edges = np.quantile(x, [1 / 3, 2 / 3])
    if edges[0] < edges[1]:
        return np.digitize(x, edges, right=False).astype(np.int64), edges
```

iii. Step 5 Key Decision 3 (quoted under 7-c) covers both dynamic outputs jointly, and
explicitly motivates per-session thresholds by "between-camera/rate calibration
differences".

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: the camera trace is sampled at `stimOn + TIME`, the
neural bins' right edges, on the shared session clock, so column *k* of the output
matches column *k* of the neural matrix. Camera frame times come from
`_ibl_<side>Camera.times`, already synchronised to the ephys clock by IBL, so no
resynchronisation is done.

ii.
```python
    whisk, whisk_good = interpolate_trials(mt, me, onsets)
    # inside: query = onsets[:, None] + TIME[None, :]
```

iii. Same justification as 7-d (Step 5 Key Decision 5). Axis 2 of the intended
`--show-processing` figure was meant to show the camera trace and its tertiles against
the shared time axis.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several layers, applied bluntly:
- **Missing trial fields** → per-trial `np.isfinite` mask over six required columns
  (implicitly also drops NaN reaction times, since NaN comparisons are False).
- **Missing trials columns entirely** → `RuntimeError` listing the missing names, which
  skips the whole session. This is the branch that fired for all 459 sessions.
- **Missing/short behavioural coverage** → trials whose window falls outside the global
  `[times[0], times[-1]]` span of the wheel or camera stream, or whose interpolated values
  contain a NaN, are dropped. Gaps *inside* a stream are not detected: `np.interp` silently
  bridges them, unlike the reference's per-interval edge test
  (`abs(interval_beg - target_time[0]) > binsize`).
- **Missing left camera** → falls back to the right camera; if neither loads, the session
  is skipped.
- **No probes / too few trials** → `RuntimeError('no probes')`, `RuntimeError('only N
  jointly valid trials')` → session skipped.
- **Degenerate behaviour traces** → rank-based tertile fallback rather than a crash.
- Every skip is caught per session, printed, and recorded in
  `metadata['failed_sessions']`; sessions never partially fail.
The coarse granularity is the weak point: any single anomalous trial value (e.g. a
`probabilityLeft` outside {0.2,0.5,0.8}, which raises `KeyError`) discards the entire
session rather than that one trial. And at the top level, the response to the
revision-resolution problem was to declare the dataset defective and stop, rather than to
reconfigure the ONE client.

ii.
```python
    missing = NEEDED - set(tr.columns)
    if missing:
        raise RuntimeError(f'trials object missing columns {sorted(missing)}')
```
```python
    return out, good & np.isfinite(out).all(axis=1)
```
```python
    if not all_times:
        raise RuntimeError('no probes')
```
```python
        except Exception as exc:
            failures.append((str(eid), f'{type(exc).__name__}: {exc}'))
            print(f'[{k}/{len(grouped)}] SKIP {eid}: {failures[-1][1]}', flush=True)
    if not infos:
        raise RuntimeError(f'no sessions converted; first failures: {failures[:5]}')
```
```python
            'session_info': infos, 'failed_sessions': failures,
```

iii. Step 5 Key Decision 2 ("Sessions need >=2 valid trials") and Key Decision 3
(degenerate-signal fallback). For the global failure, Step 6 of the notes: "this is not a
converter exception or a QC choice: required source datasets are absent from the staged
filesystem despite metadata claiming presence… Per the required ordered workflow, Step 6
cannot be marked complete … and Steps 7–13 have not been started."

## 10-a. What are the most time-consuming steps of the code?

i. Never measured — the script instruments itself (`time.perf_counter` per session,
printed as `info['seconds']`) but no successful timing exists and CONVERSION_NOTES.md's
"Run Time Estimates" table in Step 7 is an empty template. By inspection the dominant
cost is `SpikeSortingLoader.load_spike_sorting()`, made worse than necessary in two ways:
(a) `brainbox.io.one.SPIKES_ATTRIBUTES` is left at its default `['clusters', 'times',
'amps', 'depths']`, so two extra multi-hundred-MB arrays are read per probe and thrown
away; (b) `check_hash` is left at its default `True`, so every dataset is re-read to
compute an md5 (the "local md5 mismatch on dataset:" lines in the log are this running).
Secondary costs: `np.argsort` over all merged spike times, and the per-trial
`bin_spikes` loop. Structurally the largest cost is that all ~459 sessions run
**serially in one process** — the reference uses a 10-worker `ProcessPoolExecutor`.

ii.
```python
        spikes, clusters, channels = ssl.load_spike_sorting()   # no check_hash=False, all attributes
```
```python
    order = np.argsort(times, kind='stable')
```
```python
    for k, (eid, erows) in enumerate(grouped, 1):     # serial; no pool
        vals = process_session(one, str(eid), erows, ...)
```
```python
    elapsed = time.perf_counter() - t0
    info = dict(..., seconds=elapsed)
```

iii. Step 6 notes only: "Code inefficiencies identified: Reference code creates a
multiprocessing pool per stream/session and performs redundant behavior loads" and "Code
speedups added: Single SessionLoader per session; vectorized interpolation for every
trial; sorted-spike `searchsorted` windows; flattened `np.bincount`; no redundant behavior
loading." Workflow Step 7.2 (estimate runtime, optimise if >15 min) was never reached.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python-level loops plus the missing process pool:
- `trial_number_in_block` iterates trial-by-trial in pure Python; this is one line with
  pandas (`(p != p.shift()).cumsum()` then `groupby(...).cumcount()`) as in the reference.
- `bin_spikes` loops over trials; it could be a single `bincount` with a
  trial-offset flat index.
- The `for j, raw_i in enumerate(idx)` assembly loop rebuilds `np.full(...)` rows per
  trial; the whole input/output tensor could be built with broadcasting once.
- `load_spikes` loops over probes (unavoidable, one file read each).
- The session loop in `main()` is the important one: it is serial, so the whole
  conversion is the sum of ~459 session times rather than their maximum over N workers.
The one loop the AI *did* vectorise is the behavioural interpolation, which it flattens
into a single `np.interp` call over all trials.

ii.
```python
def trial_number_in_block(prob):
    out = np.zeros(len(prob), dtype=np.float32)
    for i in range(1, len(prob)):
        out[i] = out[i - 1] + 1 if prob[i] == prob[i - 1] else 0
    return out
```
```python
    for onset in onsets:
        beg, end = onset + OFF_START, onset + OFF_END
        ...
        result.append(counts.astype(np.float32))
```
```python
    for j, raw_i in enumerate(idx):
        inp = np.vstack([TIME, np.full(N_TIME, block_no[raw_i], dtype=np.float32)])
```
```python
    for k, (eid, erows) in enumerate(grouped, 1):
```

iii. Step 6 "Code speedups added: … vectorized interpolation for every trial; sorted-spike
`searchsorted` windows; flattened `np.bincount`". No justification is offered for keeping
the remaining loops or for not using multiprocessing; the notes instead criticise the
reference for using pools ("Reference code creates a multiprocessing pool per
stream/session").

## 10-c. What processing does the code repeat multiple times?

i. - **Behavioural interpolation over rejected trials**: `interpolate_trials` is called
  with *all* onsets and only afterwards subset with `wheel[idx]` / `whisk[idx]`, so work is
  done for every trial the QC mask discards (typically a substantial minority).
- **md5 hashing of every dataset on every load** (`check_hash=True` default), which
  re-reads each spike/cluster file a second time.
- **Camera load attempted twice** whenever the left view fails: the `try/except` re-enters
  `sl.load_motion_energy` for the right view after paying for the failed left attempt.
- **Beryl mapping** is applied per session at assembly (`br.acronym2acronym` once per
  session) rather than once over the pooled vocabulary — minor.
- `np.full(N_TIME, ...)` is re-allocated for four constant rows on every trial.
- Across the whole run, the same `ONE` client re-resolves dataset paths per session,
  which is inherent to the API.

ii.
```python
    wheel, wheel_good = interpolate_trials(wt, ws, onsets)     # all trials
    whisk, whisk_good = interpolate_trials(mt, me, onsets)
    valid = mask & wheel_good & whisk_good
    idx = np.flatnonzero(valid)
    ...
    wheel_cat, wheel_edges = tertiles(wheel[idx])              # only kept trials used
```
```python
    for view in ('left', 'right'):
        try:
            sl.load_motion_energy(views=[view])
```
```python
    mapped = [br.acronym2acronym(x, mapping='Beryl') for x in region_names_all]
```

iii. Not addressed in CONVERSION_NOTES.md. The notes claim the opposite of the first item
("no redundant behavior loading"), which is true of the *loading* but not of the
interpolation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. - `spikes['amps']` and `spikes['depths']` are read from disk by the default
  `SPIKES_ATTRIBUTES` and never used; the reference explicitly patches
  `bio.SPIKES_ATTRIBUTES = ['clusters', 'times']` to avoid exactly this.
- `merge_clusters(..., compute_metrics=False)` still pulls `CLUSTERS_ATTRIBUTES`
  (`channels`, `depths`, `metrics`, `uuids`) and the channels/histology tables, of which
  only `acronym` and the length of `channels` are used. Cluster `label` is loaded and
  then not used at all, since no QC filter is applied.
- `goCue_times`, `feedback_times` and `feedbackType` are loaded and required, but only to
  build the QC mask — legitimate, though `goCue_times` is required in `NEEDED` while not
  appearing in the finiteness loop.
- Behavioural traces are interpolated for trials that the mask then removes (see 10-c).
- `wheel_edges` / `whisker_edges` / `raw_trials` / `source_release_probes` are stored in
  metadata and unused by the decoder (cheap, and useful for provenance).
- `--show-processing` renders and saves a 4-panel figure per session; it is diagnostic
  only, and correctly gated behind the flag and capped at 2 sessions.
- Most significantly, keeping all ~1,354 clusters/session means ~8× more neural data is
  binned, stored and pickled than the QC-filtered population the reference papers analyse
  — a projected pickle on the order of 100 GB rather than ~15 GB.

ii.
```python
        spikes, clusters, channels = ssl.load_spike_sorting()
        clusters = ssl.merge_clusters(spikes, clusters, channels, compute_metrics=False)
        n = len(clusters['channels'])
```
```python
NEEDED = {'stimOn_times', 'choice', 'probabilityLeft', 'firstMovement_times',
          'feedback_times', 'feedbackType', 'goCue_times'}
```
```python
                wheel_edges=wheel_edges.tolist(), whisker_edges=whisk_edges.tolist(), seconds=elapsed)
```
```python
    if show:
        fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        ...
        fig.tight_layout(); fig.savefig(f'/app/processing_{eid}.png', dpi=140); plt.close(fig)
```

iii. Not addressed in CONVERSION_NOTES.md. Step 6 lists only the speedups quoted above;
the memory plan (Step 5 Key Decision 6: "Store neural counts as float32 and categorical
outputs as int64. Build one session at a time and serialize once") does not account for
the size consequence of retaining every cluster, and int64 outputs are 8× larger than the
int8 the values need.
