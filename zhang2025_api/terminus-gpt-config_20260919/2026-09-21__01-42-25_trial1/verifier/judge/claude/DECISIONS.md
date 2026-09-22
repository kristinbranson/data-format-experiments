# Decisions

> Scope note: the AI's run ended part-way through Step 9 (it was still benchmarking
> parallel workers). `/app/converted_data.pkl`, `/app/conversion_full_out.txt`,
> `/app/verification_full_out.txt`, `/app/train_decoder_full_out.txt` and `/app/README.md`
> do **not** exist; only the 2-session sample (`sample_data.pkl`) and two worker benchmarks
> (`benchmark4.pkl`, `benchmark8.pkl`) were produced. All decisions below are therefore read
> from `/app/convert_data.py` + `/app/CONVERSION_NOTES.md` (Steps 0–8 complete, 9–13 not started)
> and from the trajectory.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything scientific is loaded through the ONE API and the brainbox loaders. `init_one()`
opens `ONE(silent=True)` and loads the `Brainwidemap` release index with
`one.load_cache(tag='Brainwidemap', clobber=True)`, leaving ONE in its default (cache-backed
`remote`) query mode. The AI found that in this staged cache every canonical
`_ibl_trials.table.pqt` row has `default_revision=False`, so ONE's revision resolver refuses
to load it and `SessionLoader.load_trials()` returns only the supplemental trial columns; it
repairs that one boolean flag in the in-memory dataset table before any load. The **cohort** is
not obtained from `one.search`: it is read from the methods-paper freeze file
`/app/code/code_zhang2025/data/bwm_release.csv` (459 eids, 699 pids, 139 subjects — exactly the
data paper's released cohort), which also supplies the per-session probe list (`pid`,
`probe_name`) and provenance (subject/lab/date fallback). Per session, `SessionLoader` loads
trials, wheel and camera motion energy; per probe, `SpikeSortingLoader(pid=...)` loads and
merges spike sorting. Sessions are processed one per worker (`ProcessPoolExecutor`, 4 workers by
default, each worker calling `init_one()` once); `--sample` takes the first 2 eids of the freeze
order.

ii.
```python
def init_one() -> ONE:
    """Open the staged release and repair its canonical trials-table default metadata."""
    one = ONE(silent=True)
    one.load_cache(tag='Brainwidemap', clobber=True)
    # The frozen cache marks the canonical table non-default, causing load_object to return
    # only supplemental columns.  Repair metadata in memory; no source file is touched.
    ds = one._cache['datasets']
    mask = ds.rel_path.str.endswith('_ibl_trials.table.pqt')
    ds.loc[mask, 'default_revision'] = True
    return one


def cohort(sample: bool) -> tuple[pd.DataFrame, list[str]]:
    f = pd.read_csv(FREEZE).drop(columns=['Unnamed: 0'], errors='ignore')
    # Preserve freeze order (paper/reference order), grouping all probes from each EID.
    eids = list(dict.fromkeys(f.eid.astype(str)))
    if sample:
        eids = eids[:2]
    return f[f.eid.astype(str).isin(eids)].copy(), eids
```
```python
sl = SessionLoader(one=one, eid=eid); sl.load_trials()
sl.load_wheel(); sl.load_motion_energy(views=[side])
loader = SpikeSortingLoader(pid=str(row.pid), one=one)
spikes, clusters, channels = loader.load_spike_sorting()
clusters = loader.merge_clusters(spikes, clusters, channels)
```

iii. Notes Step 5 decision 10/11: "Use cache-backed remote query mode and repair only the
release table's canonical trial-table default flag in memory … never open scientific files
directly." The freeze CSV is justified as identifiers only — "The methods-paper freeze CSV
supplies only the curated EID/PID cohort and probe names… all scientific arrays for those EIDs
still load through ONE/brainbox" — and as the correct cohort definition because it "exactly
matches the data-paper cohort: 699 probe insertions, 459 sessions, and 139 subjects" (trajectory
step 50). The trials-table revision repair is documented as a metadata bug in the staged cache
("Local query mode alone returns only supplemental fields"), diagnosed over ~8 trajectory steps.

## 1-b. How are the data split into subjects?

i. Subject identity comes from ONE's session table, looked up by UUID, with a fallback to the
freeze CSV's `subject` column if the lookup fails. At assembly the subject list is the sorted
unique set over converted sessions and `subject_idx` is each session's index into it (int32).

ii.
```python
    try:
        details = one._cache['sessions'].loc[uuid.UUID(str(eid))]
        subject, lab, date = str(details.subject), str(details.lab), str(details.date)
    except (KeyError, ValueError):
        # Freeze metadata are identifiers/provenance only; scientific arrays remain ONE-loaded.
        fr = freeze.iloc[0]
        subject, lab, date = str(fr.subject), str(fr.lab), str(fr.date)
```
```python
subjects = sorted({x['subject'] for x in infos})
subject_idx = np.array([subjects.index(x['subject']) for x in infos], dtype=np.int32)
```

iii. Notes Step 5: "Stable sorted unique subject names and integer lookup" from the "ONE session
release table". The UUID indexing was added after the first sample run crashed with a `KeyError`
("the ONE session table is UUID-indexed but EIDs were strings", Step 7 iteration log); the freeze
fallback is labelled "provenance only".

## 1-c. How are the data split into sessions?

i. No splitting is performed: one eid = one session. The eid list is the de-duplicated `eid`
column of the freeze CSV in file order, and each eid becomes one element of `neural` / `input` /
`output` / `subject_idx` / `brain_region_idx`. Sessions that raise are dropped and recorded in
`metadata['excluded_sessions']`.

ii.
```python
eids = list(dict.fromkeys(f.eid.astype(str)))
...
payloads = [(eid, freeze[freeze.eid.astype(str) == eid].to_dict('records')) for eid in eids]
with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
    for i, result in enumerate(pool.map(_worker_run, payloads, chunksize=1), 1):
        accept(result, i)
```

iii. Notes Step 5: "Session order is stable by EID unless a subset manifest specifies order";
Step 2 documents 480 indexed release sessions, 459 with task+ephys, 445 also with camera motion
energy, so ~445 of the 459 freeze sessions are expected to survive.

## 1-d. How are the data split into trials?

i. The trials table returned by `SessionLoader.load_trials()` has one row per trial, so the split
is taken from the data. The code checks that all required columns exist, records the native trial
count, and keeps a `valid` boolean mask of the same length; retained trial indices (`raw_trial_indices`)
are stored in metadata so every converted trial can be traced back to its raw row.

ii.
```python
    required = ['choice','probabilityLeft','feedbackType','feedback_times','stimOn_times',
                'firstMovement_times','intervals_0','intervals_1']
    missing = [x for x in required if x not in tr]
    if missing:
        raise KeyError(f'missing trial columns {missing}; got {list(tr.columns)}')
    n_native = len(tr)
    stim = tr.stimOn_times.to_numpy(float)
```

iii. Notes Step 2: "Trials are primarily `_ibl_trials.table.pqt` … Complete trial objects contain
`stimOn_times`, `choice`, `probabilityLeft`, contrasts, feedback, movement and interval fields."
No decision was needed; the AI's emphasis was on making the canonical table load at all.

## 1-e. How are trials filtered based on quality controls?

i. One combined boolean mask, in this order: (a) all of
`choice, probabilityLeft, feedbackType, feedback_times, stimOn_times, firstMovement_times,
intervals_0, intervals_1` finite; (b) `choice ∈ {-1,+1}` (no-go dropped); (c)
`probabilityLeft ∈ {0.2,0.5,0.8}`; (d) trial duration `intervals_1 - intervals_0 ≤ 10 s`;
(e) first-movement latency `0.08 s ≤ firstMovement_times - stimOn_times ≤ 2.0 s`; (f) the whole
[-0.5, +1.5] s window must be spanned by both the wheel and the camera stream, enforced by
requiring all 100 interpolated samples of wheel speed and whisker motion energy to be finite
(the interpolator returns NaN outside stream support). A session with <2 surviving trials raises
and is excluded. On the sample this retained 402/565 and 241/425 trials (≈65%).

ii.
```python
    vals = tr[required].to_numpy(float)
    valid = np.all(np.isfinite(vals), axis=1)
    valid &= np.isin(choice, [-1, 1]) & np.isin(prior, [0.2, 0.5, 0.8])
    valid &= (tr.intervals_1.to_numpy() - tr.intervals_0.to_numpy() <= 10.0)
    latency = tr.firstMovement_times.to_numpy() - stim
    valid &= (latency >= 0.08) & (latency <= 2.0)

    speed, whisk, camera, behavior_errors = load_behavior(one, eid, stim)
    valid &= np.all(np.isfinite(speed), axis=1) & np.all(np.isfinite(whisk), axis=1)
    if valid.sum() < 2:
        raise RuntimeError(f'only {valid.sum()} valid trials')
```

iii. Notes Step 3/Step 4: the NaN list and the 0.08–2.00 s latency window are quoted from the data
paper ("trials were excluded if one of the following trial events could not be detected: choice,
probabilityLeft, feedbackType, feedback times, stimON times and firstMovement times"), and the
10 s cap from the reference code, which calls
`load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)`. Step 4 resolution: "Combine both
sources: finite required fields, interval duration <=10 s, movement latency in [0.08,2.00],
binary choice only, valid prior, and complete behavior windows." Requiring complete behavior
windows is justified by the outputs: "Reject trials whose complete window is outside stream
support or contains non-finite values."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from `SpikeSortingLoader.load_spike_sorting()`, per
probe. The merged cluster table supplies only the selection/annotation fields — `cluster_id`,
`label` (quality) and `acronym` (Allen region) — read through a defensive helper that checks the
three arrays are the same length.

ii.
```python
def cluster_fields(clusters):
    """Return cluster IDs, labels and acronyms from merged SpikeSortingLoader output."""
    n = len(clusters.get('channels', clusters.get('depths', [])))
    ids = np.asarray(clusters.get('cluster_id', np.arange(n)), dtype=np.int64)
    labels = np.asarray(clusters.get('label', np.zeros(n)), dtype=float)
    acr = np.asarray(clusters.get('acronym', np.repeat('', n))).astype(str)
```
```python
            pm = bin_probe(spikes['times'], spikes['clusters'], good_ids, stim)
```

iii. Notes Step 1/Step 5: the reference `load_spiking_data` uses `SpikeSortingLoader` and
`merge_clusters`, "Reference metadata defines good clusters as `clusters['label'] >= 1`", and
"Cluster acronyms provide regions". Step 1 also records that there is no imaging data, so no
dF/F is needed.

## 2-b. How is the `neural` data processed?

i. Spikes of the retained units are counted into 100 non-overlapping 20 ms bins covering
[-0.5, +1.5] s around each trial's `stimOn_times`. Counting is done per probe with
`searchsorted` to slice the trial's spikes, a dense cluster→row mapper, and a single
`np.bincount` over a flattened (unit, bin) index; the per-probe matrices are then concatenated
along the unit axis so all probes of a session form one population, with `regions` extended in
the same order. **No rate conversion, smoothing, z-scoring or PCA** is applied: the stored array
is raw integer spike counts, kept as `uint16` (with an explicit overflow guard), shaped
`(n_neurons, 100)` per trial.

ii.
```python
    out = np.zeros((len(stim), len(good_ids), 100), dtype=np.uint16)
    mapper = np.full(max(int(spike_clusters.max(initial=0)), int(good_ids.max(initial=0))) + 1, -1, dtype=np.int32)
    mapper[good_ids] = np.arange(len(good_ids), dtype=np.int32)
    for ti, st in enumerate(stim):
        lo, hi = np.searchsorted(spike_times, (st + OFF_START, st + OFF_END))
        ...
        bi = np.floor((ts[keep] - (st + OFF_START)) / BIN).astype(np.int32)
        valid = (bi >= 0) & (bi < 100)
        flat = ui[keep][valid].astype(np.int64) * 100 + bi[valid]
        cnt = np.bincount(flat, minlength=len(good_ids) * 100).reshape(len(good_ids), 100)
        if cnt.max(initial=0) > np.iinfo(np.uint16).max:
            raise OverflowError('spike count exceeds uint16')
        out[ti] = cnt.astype(np.uint16)
```
```python
    return np.concatenate(mats, axis=1), regions, probe_info, failures
...
    assert all(a.shape == (len(regions), 100) for a in neural), 'neural orientation/region mismatch'
```

iii. Notes Step 5 decision 2: "Store raw spike counts … The reference cache bins counts; no
smoothing, z-scoring, or firing-rate conversion is applied before saving", with Step 1 recording
the reference cache config `interval_len=2, binsize=0.02, align_time='stimOn_times',
time_window=(-0.5,1.5)`. Probe merging follows the paper: "Multiple probes in one session are not
treated independently: neurons in the same session/region are combined across probes." `uint16`
is justified under "Compact dtypes are used" to hold the full pickle size down. (Caveats: Step 5
says "float32 counts" while the code/metadata say `uint16`, and the `uint16` choice makes the
validator emit one warning per trial — "neural dtype is uint16, expected float32" — which the
notes wrongly report as "No errors or warnings remain".)

## 2-c. How is the `neural` data filtered based on quality controls?

i. A unit is kept only if its merged-cluster `label >= 1` **and** its Allen acronym is not in
`{'', 'void', 'root', 'nan', 'none'}` (case-insensitive, stripped). Acronyms are used as
returned — no Beryl (or any other) atlas remapping — so `brain_regions` holds fine-grained Allen
labels (e.g. `LGd-co`, `SSp-bfd5`, `VISa2/3`, and white-matter tracts `alv`, `or`, `int`, `ccs`).
A probe that raises on load is skipped with a printed WARNING and recorded in
`probe_failures`; a session with no qualified unit on any probe raises and is excluded. No
minimum-neurons-per-region or region-recorded-in-≥2-sessions rule is applied.

ii.
```python
INVALID_REGIONS = {'', 'void', 'root', 'nan', 'none'}
...
            ids, labels, acr = cluster_fields(clusters)
            region_ok = np.array([x.strip().lower() not in INVALID_REGIONS for x in acr])
            keep = (labels >= 1) & region_ok
            good_ids, good_acr = ids[keep], acr[keep]
            if not len(good_ids):
                probe_info.append({'pid': str(row.pid), 'probe': row.probe_name, 'units': 0})
                continue
    ...
    if not mats:
        raise RuntimeError('no probe yielded qualified units')
```

iii. Notes Step 3/4: "Well-isolated units satisfy all three stated RIGOR metrics: amplitude
>50 µV, noise cutoff <20 µV, and refractory-period-violation criterion. The reference code
operationalizes this as merged cluster `label >= 1`" and "Apply `label >= 1`, matching code's
operational version of RIGOR." Region exclusion: "Exclude empty/`void`/`root`/NaN acronyms;
retain fine Allen acronyms rather than collapsing regions." Skipping the paper's region-level
rules is argued explicitly: "Do not impose analysis-specific >=5/region or >=2-session threshold
because target format is session-level and should preserve neurons; document this justified
difference."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams share one session clock, so alignment is a subtraction: for each trial the
spikes in `[stimOn - 0.5, stimOn + 1.5)` are sliced by `searchsorted` and their bin index is
`floor((t - (stimOn - 0.5)) / 0.02)`, i.e. bin 0 starts exactly 0.5 s before stimulus onset and
bin 25 starts at onset. The same `stim` vector (`trials.stimOn_times` of the retained trials) is
the anchor for the inputs and for both continuous outputs, and `metadata['temporal_alignment_event']`
records `'visual stimulus onset (trials.stimOn_times)'` with `off_start=-0.5`, `off_end=1.5`.

ii.
```python
OFF_START, OFF_END = -0.5, 1.5
...
        lo, hi = np.searchsorted(spike_times, (st + OFF_START, st + OFF_END))
        ts = spike_times[lo:hi]
        ...
        bi = np.floor((ts[keep] - (st + OFF_START)) / BIN).astype(np.int32)
```
```python
    idx = np.flatnonzero(valid)
    neural_all, regions, probes, probe_failures = load_neural(one, freeze, stim[idx])
```

iii. Notes Step 4: "Trial and continuous timestamps share session clock … Follow decoder-task
alignment and methods-code cache: stimulus onset, [-0.5,+1.5] s, 20 ms, exactly 100 bins. This is
a justified task-required difference from movement-aligned paper analysis." `--show-processing`
plots draw a red line at t=0 over population counts, wheel trace and both class rasters as the
visual alignment check.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 per trial, from a fixed edge grid `np.arange(-0.5, 1.5+0.01, 0.02)` asserted
to have 101 edges / 100 centres; identical for every trial and session, and reported as
`time_bin_size = 20.0` ms. Spikes are binned once at this resolution (no finer binning and no
re-binning), and the continuous behavioural streams are not averaged within bins but linearly
interpolated onto the 100 bin centres — i.e. resampled from 1 kHz (wheel) and 60/150 Hz (camera)
to 50 Hz.

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + BIN / 2, BIN, dtype=np.float64)
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
assert EDGES_REL.size == 101 and CENTERS_REL.size == 100
```
```python
'time_bin_size': 20.0, 'off_start': OFF_START, 'off_end': OFF_END, 'n_time_bins': 100,
```

iii. Notes Step 3: "We averaged wheel values in nonoverlapping 20-ms bins … Spike counts were
similarly binned" (paper) and Step 1: the reference cache uses `binsize=0.02` with
`interval_len=2`. Step 5 edge case: "use common edges `stimOn + np.arange(-0.5,1.5+0.02,0.02)`;
verify exactly 101 edges and 100 bins."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Not from a raw variable: it is the fixed vector of bin centres of the alignment window
(-0.49, -0.47, …, 1.49 s), which is defined relative to `trials.stimOn_times` — the only raw
variable involved. The same row is written for every trial of every session.

ii.
```python
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
...
        inp = np.vstack((CENTERS_REL, np.full(100, block_num[raw_i], np.float32)))
```

iii. Notes Step 5: "fixed bin centers → `input[...,0,:]`: `-0.49, -0.47, ..., 1.49` seconds
relative to stimulus onset … Continuous time-varying decoder input, identical for all trials",
from the "caching config" window. The decoder task specifies this input as continuous and
time-varying, so it is kept as a ramp rather than a binary onset marker.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid once at import (edges → midpoints → float32) and asserting
its length, then broadcasting it to every trial as row 0 of the `(2, 100)` input matrix.

ii.
```python
EDGES_REL = np.arange(OFF_START, OFF_END + BIN / 2, BIN, dtype=np.float64)
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
assert EDGES_REL.size == 101 and CENTERS_REL.size == 100
```

iii. Defined by the conversion, not the data; the notes' only stated requirement is that it be
the centre of the same bins the spikes are counted in and that the verification report shows
range `[-0.5, 1.5]` (Step 7: "time since stimulus range | [-0.49, 1.49] s at bin centers").

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. By construction it *is* the neural bin grid: `EDGES_REL` defines both the spike bin edges
(via `floor((t - (stimOn + OFF_START))/BIN)`) and the centres used as the input, so input column
k and neural column k describe the same 20 ms interval of the same trial. It is also the query
grid for the two continuous outputs, so all four streams share one time axis.

ii.
```python
        bi = np.floor((ts[keep] - (st + OFF_START)) / BIN).astype(np.int32)
```
```python
    q = np.asarray(stim)[:, None] + centers_rel[None, :]
```
```python
    assert all(a.shape == (2, 100) for a in inputs), 'input shape mismatch'
```

iii. Notes Step 5 decision 1 and the Step 5 sanity-check list: "Input time: require every trial
row 0 to equal bin centers with `np.allclose`"; "Shape invariant: neural time dimension, input
time dimension and output time dimension are all 100 for every trial". (The `np.allclose`
sanity checks were planned for Step 10, which was never reached; the shape assertions in the
code did run.)

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft` alone, on the **unfiltered** trial sequence: since the prior is
constant within a block, a change of value (or a non-finite value) marks a new block.

ii.
```python
    prior = tr.probabilityLeft.to_numpy(float)
    choice = tr.choice.to_numpy(float)
    block_num = trial_number_in_block(prior)
```

iii. Notes Step 5: "`trials.probabilityLeft` contiguous runs → `input[...,1,:]`… Counter resets
whenever probabilityLeft differs from preceding native trial." The trials table has no block
identifier, so the block must be recovered from the prior.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based running counter over native trials: 0 on the first trial of a block, +1 while
the prior repeats, reset to 0 whenever the prior changes or is not finite. It is computed
**before** any trial filtering, so dropped trials still advance the counter and the value remains
the animal's true position in the block; it is then repeated across all 100 bins as row 1 of the
input. On the sample its range is [0, 89], consistent with the 90-trial unbiased block that
opens every session.

ii.
```python
def trial_number_in_block(prior: np.ndarray) -> np.ndarray:
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = out[i - 1] + 1 if np.isfinite(prior[i]) and prior[i] == prior[i - 1] else 0
    return out
```
```python
        inp = np.vstack((CENTERS_REL, np.full(100, block_num[raw_i], np.float32)))
```

iii. Notes Step 5 decision 5: "Count native trials since the current contiguous probabilityLeft
run began, zero-based. Compute on the unfiltered trial sequence because it is an experimental
variable, not an index into the converted subset." Decision 6 explains the broadcast over time:
"Per-trial choice/prior are repeated across time, allowing all outputs to share the required
temporal dimension and avoiding object/ragged output arrays."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The single column `trials.choice` (values -1, 0, +1). Trials with `choice == 0` (no response)
or NaN are dropped by the trial mask, and the surviving ±1 values are recoded to a binary label
that is repeated over the 100 bins as output row 0, with `output_values[0] = ['left','right']`.
The AI's mapping is **-1 → 0 ("left") and +1 → 1 ("right")**.

ii.
```python
    valid &= np.isin(choice, [-1, 1]) & np.isin(prior, [0.2, 0.5, 0.8])
...
        out = np.empty((4, 100), dtype=np.uint8)
        out[0] = 0 if choice[raw_i] == -1 else 1
```
```python
'output_names':['choice','prior_probability_left','wheel_speed_bin','whisker_motion_energy_bin'],
'output_values':[['left','right'], ...]
```

iii. Notes Step 5 mapping table: "`trials.choice` → `output[...,0,:]`: IBL `-1` (left) -> 0;
`+1` (right) -> 1; repeat over 100 bins; exclude choice 0/NaN — Required binary convention."
The stated basis is the `ibllib` extractor docstring ("-1 is a CCW turn (towards the left)"); no
independent check of the sign against stimulus side/feedback is recorded anywhere in the notes or
trajectory.

## 5-b. What processing is involved in computing `output` *Choice*?

i. None beyond the recode above and the broadcast to 100 bins in a `uint8` `(4, 100)` array; the
per-trial class counts are logged to `metadata['session_info'][i]['choice_counts']`.

ii.
```python
        out[0] = 0 if choice[raw_i] == -1 else 1
        ...
        'choice_counts': dict(Counter(int(outputs[k][0,0]) for k in range(len(outputs)))),
```

iii. Notes Step 5 decision 6 (repeat static variables across time so every output shares the
time axis). Sample fractions were [0.428/0.572] and [0.577/0.423], i.e. roughly balanced as
expected for a well-trained mouse.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The single column `trials.probabilityLeft`, restricted to exactly {0.2, 0.5, 0.8} by the trial
mask and recoded 0.2→0, 0.5→1, 0.8→2 as required by the decoder task; written as output row 1
and labelled `['0.2','0.5','0.8']`.

ii.
```python
    valid &= np.isin(choice, [-1, 1]) & np.isin(prior, [0.2, 0.5, 0.8])
...
    prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
        out[1] = prior_map[float(prior[raw_i])]
```

iii. Notes Step 5 mapping: "`0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`; repeat over 100 bins — Reject
unexpected/NaN prior values", with Step 3 confirming "Prior values | 0.2, 0.5, 0.8 |
Biased-choice task and source trials table".

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the three-way recode (an exact-float dictionary lookup, guarded by the
`np.isin` membership test in the mask) and the broadcast to 100 bins; per-session class counts
are logged as `prior_counts`. Note the prior is treated as the *block* probability actually used
by the task, not as a fitted trial-by-trial Bayesian estimate of the animal's belief.

ii.
```python
        out[1] = prior_map[float(prior[raw_i])]
        ...
        'prior_counts': dict(Counter(int(outputs[k][1,0]) for k in range(len(outputs)))),
```

iii. The decoder task specifies the mapping; the notes add no further processing. Sample
fractions [0.445, 0.142, 0.413] and [0.539, 0.183, 0.278] reproduce the expected ~2:1:2 block
structure (the 0.5 class exists only in the opening 90 unbiased trials).

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()`, i.e. the raw `_ibl_wheel.position` / `_ibl_wheel.timestamps`
turned by brainbox into an evenly-sampled, low-pass-filtered `velocity`; speed is
`np.abs(velocity)` (rad/s). No DLC/paw signal is used.

ii.
```python
def load_behavior(one: ONE, eid: str, stim: np.ndarray):
    sl = SessionLoader(one=one, eid=eid)
    sl.load_wheel()
    w = sl.wheel
    speed = interpolate_trials(w['times'].to_numpy(), np.abs(w['velocity'].to_numpy()), stim)
```

iii. Notes Step 4: "`load_target_behavior` derives velocity from wheel position/timestamps and
takes absolute value for speed … Use brainbox/reference velocity derivation, absolute speed";
trajectory step 49: "`SessionLoader.load_wheel()` provides uniformly sampled … wheel
position/velocity/acceleration, and speed is `abs(velocity)`. This should be reused directly
rather than reimplementing wheel derivatives."

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps. (1) brainbox's own pipeline inside `load_wheel`: interpolate position onto a
uniform 1 kHz grid and differentiate with a low-pass filter → `velocity`; take `abs`.
(2) Resample onto the trial grid with a single vectorized `np.interp` over all
trial × bin query times (`stim[:,None] + CENTERS_REL`), after dropping non-finite samples,
sorting by time and removing duplicate timestamps, and returning NaN outside the stream's
support (no extrapolation). (3) Discretize into 3 classes (below). No averaging within bins and
no additional smoothing.

ii.
```python
def interpolate_trials(times, values, stim, centers_rel=CENTERS_REL):
    """Linear interpolation onto common trial bin centers; NaN outside stream support."""
    good = np.isfinite(times) & np.isfinite(values)
    times, values = times[good], values[good]
    if times.size < 2:
        return np.full((len(stim), len(centers_rel)), np.nan, dtype=np.float32)
    order = np.argsort(times)
    times, values = times[order], values[order]
    # Remove duplicate timestamps, retaining first; np.interp requires increasing x.
    keep = np.r_[True, np.diff(times) > 0]
    times, values = times[keep], values[keep]
    q = np.asarray(stim)[:, None] + centers_rel[None, :]
    flat = np.interp(q.ravel(), times, values, left=np.nan, right=np.nan)
    return flat.reshape(q.shape).astype(np.float32)
```

iii. Notes Step 5 decision 7: "Use interpolation at reference bin centers/ends following
`bin_behaviors`; wheel speed derives from native wheel position and timestamps. No extrapolation
outside recorded support." Step 4: wheel values are to be summarized "on the same 20 ms grid" as
the spikes.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per session, into three equal-frequency classes. The 1/3 and 2/3 quantiles are computed on
the finite speed samples of the **retained** trials only (all trials × all 100 bins pooled), and
classes are assigned with `np.digitize(..., right=False)` → 0/1/2 = low/medium/high. Degenerate
thresholds (q1 == q0, e.g. a session whose wheel is mostly still) are nudged apart with
`np.nextafter`; an empty finite set raises. The two thresholds are stored per session
(`wheel_tertiles`) together with the realized class counts. Sample class fractions came out
[0.333, 0.333, 0.333].

ii.
```python
def quantile_classes(x: np.ndarray, valid_trials: np.ndarray):
    vals = x[valid_trials]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError('no finite values for discretization')
    q = np.quantile(vals, [1/3, 2/3]).astype(float)
    if q[1] <= q[0]:
        q[1] = np.nextafter(q[0], np.inf)
    cls = np.digitize(x, q, right=False).astype(np.uint8)
    return cls, q.tolist()
...
    speed_cls, speed_q = quantile_classes(speed, valid)
```

iii. Notes Step 5 decision 8: "The task gives class count but no physical thresholds.
Session-level tertiles provide balanced classes despite rig/camera scale differences and are fit
without using neural activity or validation labels. Save thresholds per session in metadata for
reproducibility." Step 5 also flags the tie case: "Repeated quantile thresholds: create monotonic
thresholds with `np.nextafter`; record class counts and warn if any class remains empty."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is evaluated at exactly the neural bin centres, `stim[:, None] + CENTERS_REL`, from the
same `stimOn_times` anchor, so wheel column k and neural column k cover the same 20 ms interval.
Any trial whose window is not fully covered by the wheel stream is dropped (NaN → invalid)
rather than padded, and the `(4, 100)` output shape is asserted.

ii.
```python
    q = np.asarray(stim)[:, None] + centers_rel[None, :]
    flat = np.interp(q.ravel(), times, values, left=np.nan, right=np.nan)
```
```python
    valid &= np.all(np.isfinite(speed), axis=1) & np.all(np.isfinite(whisk), axis=1)
    assert all(a.shape == (4, 100) for a in outputs), 'output shape mismatch'
```

iii. Notes Step 4: "Trial and continuous timestamps share session clock", so sampling at the bin
centres is the whole alignment. The `--show-processing` panels plot the continuous wheel trace and
its class raster on the same -0.5…1.5 s axis as the population counts to make any misalignment
visible.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with `_ibl_<side>Camera.times`, loaded through
`SessionLoader.load_motion_energy(views=[side])` and read from the `whiskerMotionEnergy` column.
The left camera is tried first and the right is used as a fallback if the left load raises or
yields no finite value in any trial window; the chosen side is recorded per session
(`camera_side`) and a session with neither usable camera raises with the accumulated errors.

ii.
```python
    for side in ('left', 'right'):
        try:
            sl.load_motion_energy(views=[side])
            key = f'{side}Camera'
            me = sl.motion_energy[key]
            vals = me['whiskerMotionEnergy'].to_numpy()
            whisk = interpolate_trials(me['times'].to_numpy(), vals, stim)
            if np.isfinite(whisk).any():
                return speed, whisk, side, errors
        except Exception as ex:
            errors.append(f'{side}:{type(ex).__name__}:{ex}')
    raise RuntimeError('no usable whisker motion energy; ' + '; '.join(errors))
```

iii. Notes Step 4/Step 5 decision 9: "Prefer left camera when valid, fall back to right exactly as
reference helper … Do not average cameras because rates, ROI geometry, and scale differ. Record
camera side per session." Step 3 records the provenance of the signal: "Whisker motion energy is
the mean absolute adjacent-frame difference in a DLC-anchored whisker-pad ROI", i.e. it is used
as released.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, normalisation or unit conversion — and is
passed through the same `interpolate_trials` resampler onto the 100 bin centres (NaN outside
support), then discretized into three per-session classes.

ii.
```python
            whisk = interpolate_trials(me['times'].to_numpy(), vals, stim)
...
    whisk_cls, whisk_q = quantile_classes(whisk, valid)
```

iii. Notes Step 4: "Whisker-pad motion energy is frame-difference mean at camera temporal
resolution … resample to common 20 ms bins before three-class discretization." Step 3 notes the
native rates (60 Hz left, 150 Hz right), which is why one side per session is kept rather than
mixing them.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to wheel speed: `np.quantile([1/3, 2/3])` of the finite samples of the retained
trials of that session, `np.digitize(..., right=False)` → 0/1/2 (low/medium/high), degenerate
thresholds nudged with `np.nextafter`, thresholds stored as `whisker_tertiles` and counts as
`whisker_class_counts`. Sample fractions [0.333, 0.333, 0.333].

ii.
```python
    whisk_cls, whisk_q = quantile_classes(whisk, valid)
        out[3] = whisk_cls[raw_i]
```
```python
'behavior_discretization':'per-session tertiles fitted on retained finite samples',
```

iii. Same justification as 7-c (Step 5 decision 8): the task fixes the number of classes but not
the thresholds, and per-session tertiles keep the classes balanced despite per-camera/per-rig
differences in absolute motion-energy scale.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: the camera trace is interpolated at `stim[:, None] + CENTERS_REL`
on the shared session clock, so it is bin-for-bin aligned with the spikes; trials not fully
covered by the camera stream are dropped.

ii.
```python
    q = np.asarray(stim)[:, None] + centers_rel[None, :]
    flat = np.interp(q.ravel(), times, values, left=np.nan, right=np.nan)
```
```python
    valid &= np.all(np.isfinite(speed), axis=1) & np.all(np.isfinite(whisk), axis=1)
```

iii. As 7-d: camera frame times are on the same clock as the spikes (Step 4: "Trial and
continuous timestamps share session clock"), and the class raster is plotted on the common
stimulus-aligned axis in `--show-processing`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or broken data is dropped at the smallest possible granularity and always recorded,
never imputed: (a) the malformed `default_revision` flag on the canonical trials table is
repaired in memory, otherwise no trials would load at all; (b) a missing required trial column
raises a `KeyError` naming the columns; (c) trials with non-finite events or incomplete
wheel/camera windows are dropped, and the retained raw indices are saved; (d) a probe that fails
to load is caught, logged as a WARNING and recorded in `probe_failures`, and the session is kept
if any other probe yields units; (e) a missing/unusable left camera falls back to the right, with
the per-side error strings kept in `behavior_load_notes`; (f) sessions with <2 valid trials, no
qualified unit, or no camera raise and are collected in `metadata['excluded_sessions']` with the
error text; (g) non-monotonic/duplicate/NaN timestamps in a continuous stream are sorted and
de-duplicated before interpolation; (h) degenerate tertiles are nudged; (i) a spike count above
the `uint16` range raises rather than wrapping; (j) a failed UUID lookup of session metadata falls
back to the freeze CSV's subject/lab/date.

ii.
```python
        except Exception as ex:
            failures.append({'pid': str(row.pid), 'probe': row.probe_name,
                             'error': f'{type(ex).__name__}: {ex}'})
            print(f'  WARNING probe {row.probe_name}/{row.pid} failed: {type(ex).__name__}: {ex}', flush=True)
```
```python
def _worker_run(payload):
    eid, records = payload
    try:
        fr = pd.DataFrame.from_records(records)
        return ('ok', eid, process_session(_WORKER_ONE, fr, eid))
    except Exception as ex:
        return ('error', eid, f'{type(ex).__name__}: {ex}', traceback.format_exc(limit=3))
```
```python
        if status == 'error':
            err, tb = rest
            excluded.append({'eid':eid,'error':err})
            print(f'  EXCLUDED {err}\n{tb}', flush=True)
            return
```

iii. Notes Step 5 "Edge Cases" enumerates exactly these cases ("Failed probe: log EID/probe/error;
retain the session only if another valid probe provides at least one good unit", "Missing left
whisker stream: use complete right stream; exclude session if neither side works", "Repeated
quantile thresholds: create monotonic thresholds with `np.nextafter`", "Empty/short sessions after
QC: exclude, with reason; decoder requires >=2 trials"). Step 2 motivates the probe-level
robustness: "Thirty-seven of 676 probe metric loads raised `OSError` during the aggregate scan.
These probes/sessions require explicit robust handling and logging during conversion; no silent
substitution is acceptable."

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting through `SpikeSortingLoader.load_spike_sorting()` /
`merge_clusters()` — file I/O over the tens of millions of spikes per probe — dominates;
everything else (trials, wheel, camera, binning, interpolation, discretization) is a fraction of
a second. Measured per-session wall time was 4.5 s (1 probe) and 7.4 s (2 probes), logged per
session via `elapsed_s`, giving a ~48 min serial estimate for 459 sessions; the AI therefore
added a 4-worker process pool (benchmark: 4 sessions in 10.1 s ⇒ ~19 min) and was benchmarking 8
workers when the run ended. Loading is deliberately deferred until after trial QC so spikes are
binned only for retained trials.

ii.
```python
            loader = SpikeSortingLoader(pid=str(row.pid), one=one)
            spikes, clusters, channels = loader.load_spike_sorting()
            clusters = loader.merge_clusters(spikes, clusters, channels)
```
```python
    # Bin neural only after behavioral/trial QC to avoid unnecessary work.
    idx = np.flatnonzero(valid)
    neural_all, regions, probes, probe_failures = load_neural(one, freeze, stim[idx])
    ...
    print(f"  retained {len(idx)}/{n_native} trials, {len(regions)} units, camera={camera}, {info['elapsed_s']:.1f}s", flush=True)
```

iii. Notes Step 6: "Full raw release has tens of millions of spikes per probe … Loading neural
data before behavioral QC wastes work on trials later excluded", and Step 7: "The serial estimate
exceeds 15 minutes. Before Step 9, session-level parallel processing or another safe optimization
is required and will be benchmarked."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain. (1) `bin_probe`'s per-trial loop (`for ti, st in enumerate(stim)`) — could be a
single global `bincount` by offsetting each spike's flat index by its trial; it is the largest
one but each iteration is already a vectorized slice+bincount. (2) `trial_number_in_block`'s
Python loop over native trials — replaceable by a `np.cumsum`/`groupby.cumcount` over the
run-change indicator (a few hundred iterations, negligible). (3) The per-trial assembly loop that
builds the `(2,100)` input and `(4,100)` output arrays (and `neural_all[j].copy()`), which could
be one strided/stacked operation. A fourth, `region_ok`, is a Python list comprehension over
cluster acronyms where `np.isin` would do. The behaviour resampling, by contrast, *is* fully
vectorized across all trials and bins in one `np.interp`.

ii.
```python
    for ti, st in enumerate(stim):
        lo, hi = np.searchsorted(spike_times, (st + OFF_START, st + OFF_END))
```
```python
    for i in range(1, len(prior)):
        out[i] = out[i - 1] + 1 if np.isfinite(prior[i]) and prior[i] == prior[i - 1] else 0
```
```python
    for j, raw_i in enumerate(idx):
        inp = np.vstack((CENTERS_REL, np.full(100, block_num[raw_i], np.float32)))
        ...
        inputs.append(inp); outputs.append(out); neural.append(neural_all[j].copy())
```

iii. Notes Step 6 identifies the generic alternative it avoided ("Generic brainbox
`get_spike_counts_in_bins` loops over every interval and assumes cluster-ID layout") and lists
the speed-ups it did implement: "Spike searches use sorted-time `searchsorted`, dense
selected-cluster lookup, and `np.bincount` per retained trial. Trial behavior interpolation is
vectorized across all trial/bin query points." The remaining loops are not discussed; since I/O
dominates at seconds per session, vectorizing them would change little.

## 10-c. What processing does the code repeat multiple times?

i. Minor repetitions only. (a) Two `SessionLoader` objects are constructed per session — one in
`process_session` for trials, a second inside `load_behavior` for wheel/motion energy — so the
session's dataset metadata is resolved twice. (b) Wheel speed and whisker energy are interpolated
for **all** native trials and only afterwards subsetted by `valid`, so ~35% of the interpolation
work is repeated on trials that are already known to be invalid; `quantile_classes` likewise
digitizes the whole array. (c) `init_one()` runs `one.load_cache(tag=..., clobber=True)` in every
worker process (4–8×), re-reading the release index each time. (d) The finite-value check reads
`stimOn_times`/`firstMovement_times` that are then re-read for the latency test. (e) If the left
camera loads but is unusable, its interpolation is redone for the right camera. None of these is
a scientific repetition — no stream is loaded from disk twice.

ii.
```python
    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
...
def load_behavior(one: ONE, eid: str, stim: np.ndarray):
    sl = SessionLoader(one=one, eid=eid)
    sl.load_wheel()
```
```python
    speed, whisk, camera, behavior_errors = load_behavior(one, eid, stim)   # stim = ALL native trials
    valid &= np.all(np.isfinite(speed), axis=1) & np.all(np.isfinite(whisk), axis=1)
```
```python
def _worker_init():
    global _WORKER_ONE
    _WORKER_ONE = init_one()
```

iii. Not discussed in the notes; the notes' efficiency claims are about what was avoided
("Probe arrays are concatenated once per session; no repeated pickle writes occur", "Behavioral
and trial validity is computed before neural binning"). The behaviour-before-mask ordering is
in fact necessary as written, because the behaviour coverage test is itself one of the QC
criteria.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three small items. (1) `process_session` returns the full continuous `speed[idx]` and
`whisk[idx]` float32 arrays (n_trials × 100 each) for **every** session, but they are used only
by `processing_plot` for at most the first two sessions — in full parallel mode they are pickled
back through the worker IPC and then thrown away. (2) The continuous traces are computed for
trials that the trial mask has already rejected (see 10-c b). (3) Rich per-session bookkeeping is
always computed and stored — `raw_trial_indices` for every retained trial, per-session choice /
prior / wheel-class / whisker-class counts, tertile edges, probe tables — which the decoder never
reads (it is useful provenance for a human, but it is carried in the pickle for all 441+
sessions). Also loaded-but-only-for-QC are `feedbackType`, `feedback_times` and `intervals_*`,
and three module imports (`re`, `sys`, `defaultdict`) are unused. Nothing large is computed and
discarded: no neurons are binned before filtering and no smoothing/PCA is precomputed.

ii.
```python
    return neural, inputs, outputs, regions, info, speed[idx], whisk[idx]
...
        n,x,y,r,info,speed,whisk = rest[0]
        ...
        if args.show_processing and len(infos)<=2: processing_plot(eid,n,y,speed,whisk)
```
```python
    info = {'eid': eid, ..., 'raw_trial_indices': idx.tolist(), ...,
            'wheel_tertiles': speed_q, 'whisker_tertiles': whisk_q, 'probes': probes,
            'probe_failures': probe_failures, 'behavior_load_notes': behavior_errors,
            'choice_counts': ..., 'prior_counts': ...,
            'wheel_class_counts': ..., 'whisker_class_counts': ...}
```

iii. Not identified in the notes as waste; the metadata is presented as a deliberate feature
("stores detailed provenance/QC metadata", "Save thresholds per session in metadata for
reproducibility"). The AI's stated efficiency priority was the opposite direction — avoiding
work on excluded trials and keeping the pickle small with compact dtypes.
