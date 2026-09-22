# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses a hybrid of the ONE API and direct file reads against the local cache in `/app/data/one_cache`, never over the network (`mode='local'`). The *list of what to process* does not come from a ONE search: it comes from the Brain-Wide Map release table shipped inside the reference repository, `/app/code/code_zhang2025/data/bwm_release.csv` (699 probe-insertion rows, 459 unique `eid`s, 139 subjects), which is grouped by `eid` so that all insertions of a session are processed together. Each stream is then loaded differently:

- session folder: `one.eid2path(eid)`;
- trials: read straight from parquet with a hand-written revision resolver, `newest()`, which `rglob`s the session's `alf/` tree for `*trials.table.pqt` and takes the lexically largest `#revision#` folder;
- spikes/clusters/channels: `SpikeSortingLoader(eid=..., pname=..., one=one).load_spike_sorting()` + `merge_clusters()`, once per probe row;
- wheel: `one.load_object(eid, 'wheel', collection='alf')`;
- whisker motion energy and its frame times: direct `np.load(..., mmap_mode='r')` on the newest-revision `leftCamera.ROIMotionEnergy.npy` and `_ibl_leftCamera.times.npy`.

The whole release is processed in a single sequential process (~1 h wall clock, ~14 GB RSS), with every session wrapped in `try/except` so a failure only skips that session and is recorded in `metadata['skipped_sessions']`.

ii.
```python
one = ONE(mode='local', cache_dir=CACHE)
one.load_cache(CACHE / 'Brainwidemap')
release = pd.read_csv(RELEASE).drop(columns=['Unnamed: 0'], errors='ignore')
...
for si, (eid, probes) in enumerate(release.groupby('eid', sort=False), 1):
    spath = one.eid2path(eid)
    tr = load_trials(spath)
```
```python
def newest(path, pattern):
    """Return newest revision of a file (revision dates sort lexically)."""
    fs = list(path.rglob(pattern))
    if not fs:
        raise FileNotFoundError(f'{pattern} below {path}')
    def key(p):
        rev = next((x[1:-1] for x in p.parts if x.startswith('#') and x.endswith('#')), '')
        return (rev, str(p))
    return max(fs, key=key)

def load_trials(session_path):
    return pd.read_parquet(newest(session_path / 'alf', '*trials.table.pqt'))
```
```python
ssl = SpikeSortingLoader(eid=eid, pname=r.probe_name, one=one)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)
...
wheel = one.load_object(eid, 'wheel', collection='alf')
mef = newest(alf, 'leftCamera.ROIMotionEnergy.npy')
tf  = newest(alf, '_ibl_leftCamera.times.npy')
me, mt = np.load(mef, mmap_mode='r'), np.load(tf, mmap_mode='r')
```

iii. From the trajectory: the AI first confirmed the cache was a real ONE cache with the `Brainwidemap` release tables plus lab symlinks into `/mnt/dataset`, and that no `DATALIMIT_SUBSET.csv` existed, concluding "the intended conversion processes the full available dataset" (step 14). It found that an online ONE tried to reach a blocked Alyx, so it moved to `mode='local'` after explicitly loading the release table (step 15). It chose `bwm_release.csv` because "The release contains 699 probe insertions across 459 sessions and 139 subjects" and because "the current release CSV likely has one row per probe insertion, while sessions must combine all probes for each eid as in the reference `prepare_data` function" (steps 17–18). It moved to direct parquet/npy reads after ONE object loading failed on revision collections: "Trials and camera objects fail through the chosen ONE collection syntax despite files existing, so direct revision-aware file loading is more robust" (step 19).

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of the release table, one value per insertion row, so the subject of a session is `probes.subject.iloc[0]` of its group. The unique subject list is built in first-appearance order with `dict.fromkeys`, and `subject_idx` indexes into it for each *retained* session. The result is 133 subjects over 433 sessions. No path or filename parsing is used.

ii.
```python
sessions.append(dict(eid=str(eid), subject=str(probes.subject.iloc[0]), ...))
...
subjects = list(dict.fromkeys(s['subject'] for s in sessions))
subj_map = {x: i for i, x in enumerate(subjects)}
...
subject_idx=np.array([subj_map[s['subject']] for s in kept], np.int32)
```

iii. No explicit justification is given beyond the observation at step 18 that the release table already carries `subject` and `lab` per insertion ("The release contains 699 probe insertions across 459 sessions and 139 subjects"), so the subject identity is read off the release rather than derived.

## 1-c. How are the data split into sessions?

i. A session is one `eid`. The release table is grouped by `eid` (`release.groupby('eid', sort=False)`), which both defines the session split and gathers that session's probe insertions so their units can be pooled into one population. Session order in the output follows first appearance in `bwm_release.csv`.

ii.
```python
for si, (eid, probes) in enumerate(release.groupby('eid', sort=False), 1):
    ...
    probe_data = []
    for r in probes.itertuples(index=False):
        ssl = SpikeSortingLoader(eid=eid, pname=r.probe_name, one=one)
```

iii. Step 17: "The current release CSV likely has one row per probe insertion, while sessions must combine all probes for each eid as in the reference `prepare_data` function." The data paper also states that probes in one session are not independent and should be combined, which the grouping implements.

## 1-d. How are the data split into trials?

i. The trials table has one row per trial, so the split is given by the data. The AI takes the row indices of the trials that pass QC (`ti = np.flatnonzero(valid)`) and uses `stimOn_times[ti]` as the per-trial alignment onsets; every stream is then cut into one `[-0.5, 1.5]` s window per onset.

ii.
```python
ti = np.flatnonzero(valid)
if len(ti) < 2:
    raise ValueError('fewer than two valid trials')
onsets = tr.stimOn_times.to_numpy(float)[ti]
```

iii. No separate justification — the AI treated the trials table as authoritative (step 20: "Direct loading confirms the full trial table and aligned camera arrays are available. The trial table has all required per-trial variables.").

## 1-e. How are trials filtered based on quality controls?

i. Five criteria, applied in two stages.

Stage 1 (on the trials table):
1. all six trial events must be finite: `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, `firstMovement_times`;
2. reaction time `firstMovement_times - stimOn_times` must lie in [0.08, 2.00] s;
3. `choice` must be ±1 (no-response trials dropped);
4. `probabilityLeft` (rounded to 1 dp) must be one of 0.2/0.5/0.8;
5. a session with fewer than 2 surviving trials is dropped entirely.

Stage 2 (after behaviour binning): a trial is kept only if all 100 wheel bins and all 100 whisker bins are finite after within-trial gap filling; a session with fewer than 2 such trials is dropped. Sessions with no left-camera motion energy (22 sessions) or no region with ≥5 good units (4 sessions) are also dropped, giving 433 of 459 sessions and 185,891 trials.

ii.
```python
needed = ['choice','probabilityLeft','feedbackType','feedback_times',
          'stimOn_times','firstMovement_times']
if any(c not in tr for c in needed):
    raise ValueError('missing required trial fields')
valid = np.ones(len(tr), bool)
for c in needed:
    valid &= np.isfinite(tr[c].to_numpy(float))
rt = tr.firstMovement_times.to_numpy(float) - tr.stimOn_times.to_numpy(float)
valid &= (rt >= .08) & (rt <= 2.00)
valid &= np.isin(tr.choice.to_numpy(float), [-1, 1])
valid &= np.isin(np.round(tr.probabilityLeft.to_numpy(float), 1), [.2, .5, .8])
```
```python
good_trials = np.isfinite(wheel_b).all(1) & np.isfinite(whisk_b).all(1)
if good_trials.sum() < 2:
    raise ValueError('fewer than two trials with complete behavior')
ti, onsets = ti[good_trials], onsets[good_trials]
```

iii. Step 22: "The methods now provide exact curation. ... Trials must have choice, probabilityLeft, feedbackType, feedback_times, stimOn_times, and firstMovement_times, with first movement latency 0.08–2.00 s." This is a literal transcription of `/app/methods.txt` line 51. The ±1 choice and prior-value tests are needed to make the categorical outputs well defined; the behaviour-completeness test is needed because the decoder outputs must exist in every bin. Metadata records the rule: `trial_filter='required events finite; first movement latency 0.08-2.00 s; valid binary choice/prior; complete wheel and whisker bins'`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from each probe's spike sorting. The merged cluster table supplies only the selection and labelling information: `clusters['label']` (RIGOR QC score), `clusters['cluster_id']` (row used to index `spikes.clusters`), and `clusters['atlas_id']` (Allen id, remapped to Beryl for `brain_regions`).

ii.
```python
ssl = SpikeSortingLoader(eid=eid, pname=r.probe_name, one=one)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)
labels = np.asarray(clusters['label'])
ids = np.asarray(clusters['cluster_id'], int)
atlas_ids = np.asarray(clusters['atlas_id'])
...
probe_data.append((np.asarray(spikes.times), np.asarray(spikes.clusters), gids, acr))
```

iii. Step 19: "Spike sorting loads correctly offline when eid and probe name are supplied. Cluster metrics include the IBL QC `label` (good units are label == 1), anatomical acronym, and atlas ID." Note the AI did not restrict `bio.SPIKES_ATTRIBUTES`, so `amps` and `depths` are also read from disk and never used.

## 2-b. How is the `neural` data processed?

i. Spikes of the surviving units are histogrammed into 100 non-overlapping 20 ms bins spanning [-0.5, 1.5) s around each trial's `stimOn_times`. Units from all probes of a session are pooled into one population, with the second probe's rows continuing after the first probe's (`col` offset). Binning is done per trial by `searchsorted`-slicing the spike train to the trial window, mapping cluster ids to output rows through a lookup array, computing the bin index by `floor`, and accumulating with `np.add.at`. The result is stored as **raw spike counts in `uint16`** — unlike the reference, it is *not* divided by the bin width to give Hz. No smoothing, normalisation or z-scoring is applied.

ii.
```python
neural = np.zeros((len(ti), n_neurons, len(CENTERS)), dtype=np.uint16)
...
max_id = int(max(np.max(sc), np.max(gids)))
cmap = np.full(max_id + 1, -1, np.int32)
cmap[gids] = np.arange(col, col + len(gids), dtype=np.int32)
order = np.argsort(st) if np.any(np.diff(st) < 0) else None
if order is not None:
    st, sc = st[order], sc[order]
for j, onset in enumerate(onsets):
    lo, hi = np.searchsorted(st, [onset + OFF0, onset + OFF1])
    cc = cmap[sc[lo:hi]]
    ok = cc >= 0
    if np.any(ok):
        bb = np.floor((st[lo:hi][ok] - onset - OFF0) / DT).astype(np.int32)
        inside = (bb >= 0) & (bb < len(CENTERS))
        np.add.at(neural[j], (cc[ok][inside], bb[inside]), 1)
region_names.extend(acr.tolist()); col += len(gids)
```

iii. Step 22: bin "good-unit spikes in 100 bins over [-0.5,1.5) relative to stimulus"; metadata says `neural_measure='spike counts per 20-ms bin'`, following the methods text "we bin spike counts using all neurons". `uint16` was a deliberate memory choice — step 22: "To keep the full dataset tractable, neural counts will be stored as uint16". The AI later considered converting to float32 (step 98) because the validator warns once per trial, but decided the warning is benign since "the validator explicitly supports [it] by converting during training" (step 100). Step 29 explains the binning algorithm: the first version scanned the whole spike vector once per neuron and was replaced by "vectorized trial-wise binning: sort spikes once, map cluster IDs to output rows, slice each 2-second trial window with searchsorted, and accumulate counts using `np.add.at`".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Four cuts, in order:
1. **Unit QC**: `clusters['label'] == 1`, i.e. the RIGOR "well-isolated neuron" criterion, plus a requirement that `atlas_id` be finite and > 0.
2. **Grey matter**: the Allen `atlas_id` is remapped to Beryl and units in `root`, `void` and `fiber tracts` are dropped.
3. **Within-session region size**: a Beryl region must contribute ≥ 5 surviving units *in that session*, otherwise all of its units are dropped; a session with no such region is dropped (4 sessions).
4. **Across-session region prevalence**: after all sessions are processed, a region must appear in ≥ 2 sessions, otherwise its units are removed from every session.

The net effect is 59,790 units over 209 regions (mean 138 units/session), against the expert's 72,417 units over 264 regions (mean 164) — the difference is essentially `root` (10,028 units in the expert output), `fiber tracts`, and the small-region cuts.

ii.
```python
good = (labels == 1) & np.isfinite(atlas_ids) & (atlas_ids > 0)
mapped = atlas.remap(atlas_ids[good].astype(int), source_map='Allen', target_map='Beryl')
acr = np.asarray(atlas.get(mapped)['acronym']).astype(str)
keep = ~np.isin(acr, ['root','void','fiber tracts'])
gids, acr = ids[good][keep], acr[keep]
```
```python
# Region criterion: >=5 good neurons in this session.
all_regions = np.concatenate([x[3] for x in probe_data])
vals, cnt = np.unique(all_regions, return_counts=True)
allowed = set(vals[cnt >= 5])
```
```python
# Across-session region criterion from the data paper.
prevalence = {}
for s in sessions:
    for r in np.unique(s['regions']): prevalence[r] = prevalence.get(r, 0) + 1
allowed_global = {r for r, n in prevalence.items() if n >= 2}
```

iii. Step 22: "Neurons are well-isolated units (`label == 1`), and final regions must be grey matter, have at least five good neurons in a session, and occur in at least two sessions. I will implement these rules." This is taken verbatim from `/app/methods.txt` line 54 ("Final analyses were additionally restricted to regions that were designated grey matter ..., contained at least five well-isolated neurons per session and were recorded from in at least two such sessions"). Step 83 confirms the AI treated a resulting session drop as expected curation, not an error: "Session 383 was correctly skipped because no Beryl region contained at least five well-isolated neurons, matching the paper's region inclusion rule."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to visual stimulus onset, `stimOn_times`. All IBL streams are already on one synchronised session clock, so alignment is a subtraction: for each trial the spike train is sliced to `[onset - 0.5, onset + 1.5)` with `searchsorted`, the onset is subtracted from each spike time, and the bin index is `floor((t - onset + 0.5)/0.02)`, clipped to [0, 99] by an explicit in-range mask. `metadata['temporal_alignment_event'] = 'visual stimulus onset (stimOn_times)'`, `off_start=-0.5`, `off_end=1.5`.

ii.
```python
OFF0, OFF1 = -0.5, 1.5
onsets = tr.stimOn_times.to_numpy(float)[ti]
...
lo, hi = np.searchsorted(st, [onset + OFF0, onset + OFF1])
bb = np.floor((st[lo:hi][ok] - onset - OFF0) / DT).astype(np.int32)
inside = (bb >= 0) & (bb < len(CENTERS))
```

iii. Step 8: "The exact temporal parameters are now established: 2 s intervals, 20 ms bins, alignment to stimOn_times, window (-0.5, 1.5)", read out of the Zhang caching script. Step 20 notes the tension with the methods paper, which aligns wheel/whisker decoding to first movement, and resolves it in favour of the task instructions: "the paper text ... indicates its original continuous wheel decoder used a different first-movement window, but this task explicitly overrides alignment to stimulus onset, so the common (-0.5, 1.5) stimulus window from the provided repository should be used for all streams."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms (`DT = 0.020`, `metadata['time_bin_size'] = 20.0` ms), 100 bins per trial over a 2 s window, identical for every trial and session. The spikes are histogrammed directly onto this grid, so there is no rebinning or resampling of neural data; behaviour is averaged onto the same grid (see 7-b/8-b). Edges are `np.arange(-0.5, 1.5 + DT/2, DT)` and the reported times are the bin centres, -0.49 … 1.49 s.

ii.
```python
DT = 0.020
OFF0, OFF1 = -0.5, 1.5
EDGES = np.arange(OFF0, OFF1 + DT/2, DT)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
...
time_bin_size=20.0, temporal_alignment_event='visual stimulus onset (stimOn_times)',
off_start=-0.5, off_end=1.5,
```

iii. Step 4: "The methods establish the key reference setup: 2-second stimulus-aligned trials, 20 ms bins (100 time points), spike counts". Step 8 confirms it from the reference repository's own parameters. The methods paper describes exactly this configuration for the choice/stimulus-aligned dataset.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from the raw data at all beyond `stimOn_times` defining bin zero: the input is the fixed vector of the 100 bin centres of the analysis window, identical for every trial and session (-0.49 to 1.49 s in 20 ms steps).

ii.
```python
EDGES = np.arange(OFF0, OFF1 + DT/2, DT)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
...
input_out.append([np.vstack((CENTERS.astype(np.float32), ...))
                  for j in range(len(s['trials']))])
```

iii. Implicit: the decoder task asks for "Time since stimulus onset, continuous, time-varying", and since every trial is cut on the same grid around `stimOn_times`, the time axis is the grid itself. `input_names[0] = 'time since stimulus onset'`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None. The centres are computed once at module level and broadcast into every trial's input array, cast to float32.

ii.
```python
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
...
np.vstack((CENTERS.astype(np.float32),
           np.full(len(CENTERS), s['block_trial'][j], np.float32)))
```

iii. N/A — the variable is defined by the chosen window, not derived from data.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the neural binning grid. Spikes are assigned to bin `floor((t - onset + 0.5)/0.02)`, so bin *k* covers `[onset - 0.5 + 0.02k, onset - 0.5 + 0.02(k+1))`, and the input value for bin *k* is that bin's centre. The two are aligned bin for bin by construction.

ii.
```python
bb = np.floor((st[lo:hi][ok] - onset - OFF0) / DT).astype(np.int32)   # neural
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2                                # input
```

iii. N/A.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. The trials table has no block identifier, so blocks are recovered from the fact that `probabilityLeft` is constant within a block: a change of its (1-dp rounded) value starts a new block.

ii.
```python
probs = np.round(tr.probabilityLeft.to_numpy(float), 1)
for j in range(len(tr)):
    if j == 0 or probs[j] != probs[j-1] or not np.isfinite(probs[j-1]): k = 0
    else: k += 1
    block_trial[j] = k
```

iii. No explicit trajectory statement; the code comment reads "Trial number within probability block, zero based." The approach follows from the task description in the methods, which says the prior is constant over a block and block changes are uncached/uncued.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based running counter over the **unfiltered** trials table (so trials later discarded by QC still advance the count, and the value reflects the animal's true position in the block), which is then indexed by the surviving trial indices `ti` and broadcast as a constant across the 100 bins of that trial, as float32. The observed range is 0–98, matching the expert's.

ii.
```python
block_trial = np.zeros(len(tr), np.float32); k = 0
probs = np.round(tr.probabilityLeft.to_numpy(float), 1)
for j in range(len(tr)):
    if j == 0 or probs[j] != probs[j-1] or not np.isfinite(probs[j-1]): k = 0
    else: k += 1
    block_trial[j] = k
sessions.append(dict(..., block_trial=block_trial[ti], ...))
...
np.full(len(CENTERS), s['block_trial'][j], np.float32)
```

iii. Not discussed explicitly in the trajectory; the code places the computation before trial selection and indexes afterwards, which preserves the true within-block position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (leftward wheel turn), -1 (rightward) or 0 (no response). Trials with 0 have already been removed by the trial filter.

ii.
```python
valid &= np.isin(tr.choice.to_numpy(float), [-1, 1])
...
choice = (tr.choice.to_numpy(float)[ti] == -1).astype(np.int8)  # IBL: +1 left, -1 right
```

iii. Step 99: "in IBL, `choice == +1` is leftward and `choice == -1` is rightward, while the task requires left=0 and right=1."

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding +1 → 0 (left) and -1 → 1 (right), as int8, then broadcast as a constant across the 100 bins of the trial. `output_values[0] = ['left','right']`. The resulting class balance is 0.508/0.492, essentially identical to the expert's 0.507/0.493.

ii.
```python
choice = (tr.choice.to_numpy(float)[ti] == -1).astype(np.int8)
...
np.full(len(CENTERS), s['choice'][j], np.int8)
```

iii. The first version of the script wrote `(choice == 1)`, i.e. left = 1, which the AI caught during validation (step 99): "The current converter encoded `choice == +1` as 1, reversing the requested labels." It fixed `convert_data.py` and, rather than re-running the ~1 h conversion, patched the existing pickle in place with `y[0] = 1 - y[0]` for every trial, adding `metadata['choice_encoding']`. The patch is logically equivalent to re-running with the fixed code.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, rounded to one decimal, which takes the values 0.2, 0.5 and 0.8.

ii.
```python
valid &= np.isin(np.round(tr.probabilityLeft.to_numpy(float), 1), [.2, .5, .8])
...
pleft = np.round(tr.probabilityLeft.to_numpy(float)[ti], 1)
```

iii. Not separately justified; the task instructions give the three values and their codes, and the methods describe the 20:80 / 50:50 / 80:20 block structure.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A dictionary recode 0.2 → 0, 0.5 → 1, 0.8 → 2 as int8, broadcast as a constant across the 100 bins. `output_values[1] = ['0.2','0.5','0.8']`. Class fractions 0.418 / 0.140 / 0.442 match the expert's 0.418 / 0.141 / 0.442.

ii.
```python
prior = np.array([{.2:0,.5:1,.8:2}[float(x)] for x in pleft], np.int8)
...
np.full(len(CENTERS), s['prior'][j], np.int8)
```

iii. Directly from the instructions ("Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2"). The explicit `np.round(..., 1)` guards against float representation of the stored probabilities.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position` and `_ibl_wheel.timestamps`, loaded as the ONE `wheel` object. Speed is the absolute value of the filtered wheel velocity derived from them.

ii.
```python
wheel = one.load_object(eid, 'wheel', collection='alf')
wpos, wt = interpolate_position(np.asarray(wheel.timestamps), np.asarray(wheel.position), freq=1000)
ws, _ = velocity_filtered(wpos, 1000)
wheel_b = fill_short_gaps(binned_mean(wt, np.abs(ws), onsets))
```

iii. Step 26: "The correct IBL function is `velocity_filtered`, which requires uniformly sampled position. The standard pipeline first uses `interpolate_position`, then computes filtered velocity." The code comment states: "Standard IBL wheel processing: interpolate to 1 kHz then 20-Hz low-pass filtered velocity (brainbox.behavior.wheel)."

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps:
1. the sparse, movement-triggered wheel trace is interpolated onto a uniform 1000 Hz grid (`interpolate_position`);
2. differentiated into velocity with the default 20 Hz Butterworth low-pass (`velocity_filtered`), and the absolute value is taken → speed in rad/s;
3. **averaged within each 20 ms trial bin** — `binned_mean` sorts the trace once, builds a cumulative sum, and for each trial takes `searchsorted` at the bin edges, so each bin gets the mean of the ~20 samples inside it; bins with no sample become NaN and are then filled by linear interpolation across bins within the trial (`fill_short_gaps`, which needs ≥ 2 finite bins and clamps at the trial edges);
4. discretised into tertiles over the whole session (see 7-c).

ii.
```python
def binned_mean(times, values, onsets):
    """Mean samples in each trial-relative bin; NaN where no samples exist."""
    times = np.asarray(times, float); values = np.asarray(values, float).squeeze()
    good = np.isfinite(times) & np.isfinite(values)
    times, values = times[good], values[good]
    order = np.argsort(times); times, values = times[order], values[order]
    cs = np.r_[0., np.cumsum(values)]
    out = np.full((len(onsets), len(CENTERS)), np.nan, np.float32)
    for i, onset in enumerate(onsets):
        ix = np.searchsorted(times, onset + EDGES)
        n = np.diff(ix)
        sm = cs[ix[1:]] - cs[ix[:-1]]
        np.divide(sm, n, out=out[i], where=n > 0)
    return out

def fill_short_gaps(x):
    """Interpolate occasional empty temporal bins within each trial."""
    x = np.asarray(x, np.float32)
    q = np.arange(x.shape[1])
    for row in x:
        ok = np.isfinite(row)
        if ok.sum() >= 2:
            row[~ok] = np.interp(q[~ok], q[ok], row[ok])
    return x
```

iii. Step 22: "average wheel speed and left-camera whisker motion energy in the same bins". Bin-averaging is what the data paper's own wheel decoding does ("We averaged wheel values in nonoverlapping 20-ms bins", methods line 72). Metadata records it: `behavior_binning='Mean absolute wheel angular velocity and left-camera ROI motion energy in each neural time bin; session-wise tertiles.'`

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Equal-frequency tertiles computed **per session** over all (trial × bin) values of that session: the 1/3 and 2/3 quantiles of the finite values become the two cut points, and `np.digitize` assigns 0/1/2 (`['low','medium','high']`). A degenerate case where the two quantiles coincide is handled with a deterministic tie-breaking jitter. The thresholds are stored per session in `metadata['session_info'][i]['wheel_tertiles']`. Resulting class fractions are 0.3333 / 0.3333 / 0.3333.

ii.
```python
def tertiles(x):
    """Session-wise equal-frequency discretization, preserving NaNs as invalid."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise ValueError('behavior contains no finite samples')
    q = np.quantile(finite, [1/3, 2/3])
    if q[0] == q[1]:
        # deterministic fallback for unusually constant traces
        q = np.quantile(finite + np.linspace(0, 1e-7, finite.size), [1/3, 2/3])
    return np.digitize(x, q, right=False).astype(np.int8), q.tolist()
...
wheel_c, wheel_q = tertiles(wheel_b)
```

iii. Step 21: "The remaining key ambiguity is discretization: this task requires 3 categorical bins for wheel and whisker, unlike the paper's continuous regression. I need choose and document a statistically defensible scheme, most likely session-wise tertiles over valid time bins to avoid scale differences across sessions/cameras." The per-session scope is chosen so that absolute wheel/camera scale differences between sessions do not turn the class label into a session label, and equal frequency keeps the decoder's three classes balanced.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is averaged on exactly the same bin edges, measured from the same `stimOn_times`, as the spikes: `binned_mean` evaluates `np.searchsorted(times, onset + EDGES)`, where `EDGES` is the same array used to define the neural bins. So bin *k* of the wheel output covers the same interval as bin *k* of the neural matrix. Only trials whose 100 wheel bins are all finite are retained.

ii.
```python
ix = np.searchsorted(times, onset + EDGES)      # same EDGES as the neural grid
...
good_trials = np.isfinite(wheel_b).all(1) & np.isfinite(whisk_b).all(1)
```

iii. The wheel timestamps are on the same synchronised session clock as the spike times, so binning both against `onset + EDGES` is all the alignment required.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` (one value per video frame, IBL's mean absolute frame difference over the whisker-pad bounding box) with frame times `_ibl_leftCamera.times.npy`, both read directly from the newest revision folder with `np.load(..., mmap_mode='r')`. **Only the left camera is used** — there is no fall-back to the right camera, so the 22 sessions lacking left-camera motion energy raise `FileNotFoundError` and are dropped.

ii.
```python
alf = spath / 'alf'
# Left camera is the high-frame-rate side view used for whisker-pad motion.
mef = newest(alf, 'leftCamera.ROIMotionEnergy.npy')
tf = newest(alf, '_ibl_leftCamera.times.npy')
me, mt = np.load(mef, mmap_mode='r'), np.load(tf, mmap_mode='r')
n = min(len(me), len(mt))
```

iii. The only justification is the in-code comment, "Left camera is the high-frame-rate side view used for whisker-pad motion" (which is factually inverted — the data paper states the left camera is full-resolution at 60 Hz and the *right* camera is the 150 Hz one). At step 96 the AI noticed and accepted the resulting loss: "The 26 excluded sessions consist of 22 without left-camera whisker motion energy and four without a region containing at least five good neurons, yielding the expected 433 sessions" — i.e. it took the drop as confirmation because 433 is the session count quoted in the methods paper.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is (no filtering, no normalisation). The frame array and time array are truncated to their common length, non-finite samples are dropped, and the trace is averaged into the same 20 ms bins as the neural data with `binned_mean`. Because the left camera runs at 60 Hz, only ~1.2 frames fall in each 20 ms bin and a substantial fraction of bins contain none; those NaN bins are filled by linear interpolation across bins within the trial (`fill_short_gaps`). Trials that still contain a NaN after filling are dropped.

ii.
```python
n = min(len(me), len(mt))
whisk_b = fill_short_gaps(binned_mean(mt[:n], me[:n], onsets))
good_trials = np.isfinite(wheel_b).all(1) & np.isfinite(whisk_b).all(1)
```

iii. Step 22 ("average ... left-camera whisker motion energy in the same bins"); `fill_short_gaps` is documented in code as "Interpolate occasional empty temporal bins within each trial", which is required because the camera is slower than the bin grid.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to wheel speed: per-session equal-frequency tertiles of all finite (trial × bin) values, `np.digitize` into 0/1/2 with labels `['low','medium','high']`, thresholds saved in `metadata['session_info'][i]['whisker_tertiles']`. Class fractions 0.334 / 0.333 / 0.333.

ii.
```python
whisk_c, whisk_q = tertiles(whisk_b)
...
output_values=[['left','right'],['0.2','0.5','0.8'],['low','medium','high'],['low','medium','high']],
```

iii. Step 21: session-wise tertiles were chosen specifically "to avoid scale differences across sessions/cameras" — motion energy is in arbitrary pixel units that depend on camera, illumination and ROI, so a global threshold would be meaningless.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The same `binned_mean(..., onsets)` call with the same `EDGES` referenced to each trial's `stimOn_times`, so camera bin *k* and neural bin *k* cover the same interval. The camera frame times are on the same synchronised session clock as the spikes, so no further alignment is applied.

ii.
```python
whisk_b = fill_short_gaps(binned_mean(mt[:n], me[:n], onsets))
# inside binned_mean:
ix = np.searchsorted(times, onset + EDGES)
```

iii. As for the wheel: the streams share the session clock, so binning on the same onset-referenced edges is sufficient.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms:
- **Per-session isolation**: the entire per-session body is wrapped in `try/except`, and any failure is appended to `failures` and written to `metadata['skipped_sessions']` with the exception type and message, so one bad session cannot abort the release-wide run.
- **Missing columns / events**: a missing required trials column raises and skips the session; non-finite trial events drop the trial.
- **Camera array length mismatch** (a known IBL quirk): `n = min(len(me), len(mt))` truncates both arrays to the common length.
- **Non-finite behaviour samples** are removed inside `binned_mean` before binning.
- **Empty behaviour bins** (unavoidable at 60 Hz) are filled by within-trial linear interpolation; trials that still have any NaN are dropped, as are sessions left with < 2 such trials.
- **Degenerate tertile thresholds** get a deterministic jitter fallback; a behaviour trace with no finite sample raises and skips the session.
- **Empty neuron sets**: probes with no good units are skipped, and sessions with no qualifying region are dropped.
- Trials with zero spikes in all retained neurons are *kept* deliberately (16 such trials), because trial inclusion is event/behaviour based.

ii.
```python
except Exception as exc:
    failures.append((str(eid), f'{type(exc).__name__}: {exc}'))
    print(f'[{si}] SKIP {eid}: {failures[-1][1]}', flush=True)
...
skipped_sessions=failures
```
```python
n = min(len(me), len(mt))
good = np.isfinite(times) & np.isfinite(values)
...
if ok.sum() >= 2:
    row[~ok] = np.interp(q[~ok], q[ok], row[ok])
...
if q[0] == q[1]:
    q = np.quantile(finite + np.linspace(0, 1e-7, finite.size), [1/3, 2/3])
```

iii. Step 100: "16 of 185,891 individual trials with no spikes in their retained neurons. Such occasional silent trials are valid and should not be removed because trial inclusion is behavioral/event based." The remaining handling is implicit in the code (defensive `try/except`, length truncation, gap filling) and recorded in metadata so that skipped sessions are auditable.

## 10-a. What are the most time-consuming steps of the code?

i. Measured from the run itself (~1 h for 459 sessions, ~5–9 sessions/min, ~14 GB RSS):
1. **Reading the spike sorting from disk** — the dominant cost, and it is inflated here because `bio.SPIKES_ATTRIBUTES` is left at its default `['clusters','times','amps','depths']`, so roughly twice as much spike data is read as the conversion needs (`amps` and `depths` are never touched).
2. **The per-trial spike binning loop** using `np.add.at`, which is an unbuffered scatter-add and is several times slower than a `bincount` on a flat index; it runs once per trial per probe (~186k × ~1.5 iterations).
3. **Instantiating `AllenAtlas()`** once at start-up purely to reach `.regions` — this loads the full Allen volume/label arrays when `iblatlas.regions.BrainRegions()` alone would have sufficed.
4. **Single-process, fully sequential execution**, holding every processed session in memory until the end (the expert ran 10 worker processes).

ii.
```python
ssl = SpikeSortingLoader(eid=eid, pname=r.probe_name, one=one)
spikes, clusters, channels = ssl.load_spike_sorting()   # loads amps/depths too
```
```python
atlas = AllenAtlas().regions
```
```python
for j, onset in enumerate(onsets):
    lo, hi = np.searchsorted(st, [onset + OFF0, onset + OFF1])
    ...
    np.add.at(neural[j], (cc[ok][inside], bb[inside]), 1)
```

iii. The AI was aware of the I/O and loop cost and optimised the worst case at step 29: "the current neural binning repeatedly scans all spikes once per neuron and would be too slow over 459 sessions. Plan: Replace per-neuron spike scans with vectorized trial-wise binning". It monitored the remaining runtime as acceptable (step 47: "it reached session 103/459 in 13:43 ... At this rate the full pass will take about an hour") and chose not to parallelise.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four remain:
1. the per-trial spike binning loop — could be a single `np.bincount` over `unit * N_BINS + bin` for all trials at once, as the expert does; at minimum `np.add.at` could be replaced by `np.bincount`, which is much faster for the same result;
2. the per-trial loop inside `binned_mean` — one `searchsorted` over a concatenated `onsets[:,None] + EDGES` query would do all trials in one call;
3. the pure-Python `for j in range(len(tr))` loop that computes `block_trial`, which is exactly `trials.groupby((p != p.shift()).cumsum()).cumcount()` in pandas;
4. the per-row Python loop in `fill_short_gaps`, and the final per-trial list comprehensions that `np.vstack` each trial's input/output arrays (186k × 2 small allocations).

ii.
```python
for j, onset in enumerate(onsets):            # (1)
    np.add.at(neural[j], (cc[ok][inside], bb[inside]), 1)
```
```python
for i, onset in enumerate(onsets):            # (2)
    ix = np.searchsorted(times, onset + EDGES)
```
```python
for j in range(len(tr)):                      # (3)
    if j == 0 or probs[j] != probs[j-1] or not np.isfinite(probs[j-1]): k = 0
    else: k += 1
```
```python
for row in x:                                 # (4)
    ok = np.isfinite(row)
```

iii. The AI addressed only the biggest one (step 29) and did not revisit the others; loops (2)–(4) are cheap per session, and the AI's monitoring showed session throughput was dominated by spike I/O, so it left them alone.

## 10-c. What processing does the code repeat multiple times?

i.
- `newest()` is called three times per session (trials, motion energy, camera times) and each call does a full `Path.rglob` over the session's `alf/` tree, re-walking the same directory.
- `np.isin(acr, list(allowed))` rebuilds a Python list from the `allowed` set on every probe, and `np.isin(s['regions'], list(allowed_global))` does the same per session at assembly.
- The good-unit selection and Beryl remap are computed in the first probe pass, then the region membership test is redone in the second pass over `probe_data`.
- `binned_mean` re-sorts and re-cumsums the full stream once per behaviour variable (unavoidable) but then repeats a `searchsorted` per trial over the whole timestamp array rather than over a per-trial slice.
- `np.round(tr.probabilityLeft, 1)` is computed three times (trial filter, prior, block detection).

ii.
```python
tr = load_trials(spath)                       # newest() #1
mef = newest(alf, 'leftCamera.ROIMotionEnergy.npy')   # newest() #2
tf = newest(alf, '_ibl_leftCamera.times.npy')         # newest() #3
```
```python
use = np.isin(acr, list(allowed)); gids, acr = gids[use], acr[use]
...
nk = np.isin(s['regions'], list(allowed_global))
```

iii. Not discussed in the trajectory; none of these is a correctness problem and each is small relative to spike I/O.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Spike attributes never used**: `amps` and `depths` are read from disk for every insertion because `SPIKES_ATTRIBUTES` is not narrowed (the expert sets it to `['clusters','times']`).
- **The full Allen atlas volume** is loaded via `AllenAtlas()` only to obtain `.regions`; `BrainRegions()` would give the same acronym/remap tables without the volume.
- **Neural binning done for trials that are then dropped**: the `(n_valid_trials, n_neurons, 100)` array is filled *before* the wheel/whisker completeness mask `good_trials` is applied, so spikes are binned for trials that are immediately discarded.
- **Neural binning done for neurons that are then dropped**: units are binned if their region has ≥ 5 units in that session, but the across-session prevalence filter (`≥ 2 sessions`) later removes some of those rows at assembly, so their bins were computed and stored for nothing.
- **Fields loaded only to be thrown away**: `feedbackType` and `feedback_times` are read and finiteness-tested but never exported; `traceback` is imported and never used; `channels` is only a pass-through to `merge_clusters`.
- **Metadata bookkeeping** that the decoder never reads: `source_trial_indices` per session, `wheel_tertiles` / `whisker_tertiles`, `skipped_sessions`. These are cheap and useful for auditing, but they are not consumed downstream.
- **Storage**: `uint16` counts keep the file at 5.5 GB, but the validator/decoder converts every trial to float32 at load time anyway, so the compactness buys nothing at training time and costs 185,891 validator warnings.

ii.
```python
import pickle, warnings, traceback          # traceback unused
from iblatlas.atlas import AllenAtlas       # full volume loaded for .regions only
...
spikes, clusters, channels = ssl.load_spike_sorting()   # amps/depths unused
```
```python
neural = np.zeros((len(ti), n_neurons, len(CENTERS)), dtype=np.uint16)
... # filled here, for all `ti` and all `allowed` regions
good_trials = np.isfinite(wheel_b).all(1) & np.isfinite(whisk_b).all(1)
neural, wheel_b, whisk_b = neural[good_trials], wheel_b[good_trials], whisk_b[good_trials]
...
nk = np.isin(s['regions'], list(allowed_global))     # more rows discarded at assembly
neural_out.append([s['neural'][j, nk] for j in range(len(s['trials']))])
```

iii. The AI justified the `uint16` choice as memory management ("To keep the full dataset tractable, neural counts will be stored as uint16", step 22) and then explicitly decided not to convert to float32 after seeing the validator warnings, reasoning that the validator "explicitly supports [it] by converting during training" (steps 98, 100). The other items are not mentioned in the trajectory.
