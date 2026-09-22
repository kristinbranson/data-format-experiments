# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read through the ONE API pointed at the local IBL cache (`/app/data/one_cache`); no file is opened directly. The list of sessions is *not* obtained from a ONE search but from the brain-wide-map public release freeze CSV that ships with the reference code, `/app/code/code_zhang2025/data/bwm_release.csv` (700 probe-insertion rows → 459 unique `eid`s, 139 mice), exactly the file `0_data_caching.py` uses. The unique `eid`s in order of first appearance become the session order. If `/app/data/DATALIMIT_SUBSET.csv` exists the list is intersected with it (it does not exist in this run, so all 459 sessions were attempted; 443 survived). Everything else is resolved from the `eid`: `one.eid2pid(eid)` gives the probe insertions, `SpikeSortingLoader` reads the spikes/clusters/channels of each probe, the vendored reference helper `U.load_trials_and_mask(one=one, eid=eid, ...)` reads the trials table through `SessionLoader`, and `U.bin_behaviors` → `U.load_target_behavior` reads the wheel and the camera motion energy, again through `SessionLoader`. The reference repository is put on `sys.path` and its functions are reused unchanged, with one compatibility shim: ibllib 4.0.1 made `SessionLoader` a keyword-only dataclass while the reference code calls `SessionLoader(one, eid)` positionally, so a subclass accepting positional arguments is monkey-patched into both `brainbox.io.one` and the reference module.

ii.
```python
REPO = '/app/code/code_zhang2025/src'
sys.path.insert(0, REPO)

class SessionLoaderCompat(_SessionLoader):
    def __init__(self, one=None, eid='', **kwargs):
        super().__init__(one=one, eid=eid, **kwargs)

import utils.ibl_data_utils as U  # noqa: E402
U.SessionLoader = SessionLoaderCompat
_bbone.SessionLoader = SessionLoaderCompat
```

```python
one = ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True,
          cache_dir='/app/data/one_cache')

bwm_df = pd.read_csv(BWM_RELEASE, index_col=0)
# session order = order of first appearance in the public release freeze
eids = list(dict.fromkeys(bwm_df.eid.tolist()))
if os.path.exists(DATALIMIT):
    subset = pd.read_csv(DATALIMIT)
    col = 'eid' if 'eid' in subset.columns else subset.columns[0]
    allowed = set(subset[col].astype(str))
    eids = [e for e in eids if e in allowed]
```

```python
pids, probe_names = one.eid2pid(eid)
...
    sp, cl = load_clusters_and_spikes(one, str(pid), eid, probe_name)
...
trials_df, trials_mask = U.load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)
...
binned_beh, _ = U.bin_behaviors(one, eid, BEH_NAMES, trials_df=trials_df,
                                allow_nans=True, n_workers=n_workers, **PARAMS)
```

iii. From the trajectory (steps 31–45): the agent read `0_data_caching.py` and `ibl_data_utils.py`, established that the reference pipeline is `prepare_data` → `list_brain_regions` → `select_brain_regions` → `bin_spiking_data` → `bin_behaviors` → `align_spike_behavior` driven by the `bwm_release.csv` freeze, and decided to reuse those functions verbatim so that "all loading / curation / alignment / binning decisions follow the reference implementation". It only deviated where the vendored code cannot run on the installed ibllib (the `SessionLoader` signature) or reads data that is unavailable/unused (see 9).

## 1-b. How are the data split into subjects?

i. No splitting is done by the conversion: the release freeze CSV carries a `subject` column, so the mapping `eid → subject` is read straight from it (`eid2subject = dict(zip(bwm_df.eid, bwm_df.subject))`). `data['subjects']` is built in order of first appearance (not sorted) and `subject_idx` records each session's index into that list. Result: 136 subjects over 443 sessions (1–13 sessions per animal).

ii.
```python
eid2subject = dict(zip(bwm_df.eid, bwm_df.subject))
eid2lab = dict(zip(bwm_df.eid, bwm_df.lab))
...
    sub = eid2subject[eid]
    if sub not in subjects:
        subjects.append(sub)
    ...
    data['subject_idx'].append(subjects.index(sub))
...
data['subjects'] = subjects
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. Implicit in the decision to drive the conversion from the reference freeze file: the freeze already names the subject of every session, so nothing has to be derived or parsed. The subject is also carried into `metadata['session_info']` per session.

## 1-c. How are the data split into sessions?

i. A session is the unit of the release, identified by its `eid`. The freeze CSV has one row per probe insertion, so unique-ifying the `eid` column (preserving order) gives the sessions; each `eid` is then processed independently by `load_session`, and one element of `data['neural'] / ['input'] / ['output']` is appended per successfully converted session.

ii.
```python
eids = list(dict.fromkeys(bwm_df.eid.tolist()))
...
for k, eid in enumerate(eids):
    try:
        sess = load_session(one, eid, args.n_workers)
    except Exception as exc:
        print(f'  SKIPPED {eid}: {exc!r}', flush=True)
        skipped.append({'eid': eid, 'reason': repr(exc)})
        continue
```

iii. No decision to make — the release is organised by session, and the multiple rows per `eid` are the two probes of a session, which are merged rather than treated as separate recordings (see 2-b).

## 1-d. How are the data split into trials?

i. The trials table returned by `U.load_trials_and_mask` has one row per trial, so the split is given by the data. Trial boundaries for the conversion are not the IBL `intervals` but the fixed decoding window: the reference `bin_spiking_data` / `get_behavior_per_interval` build `intervals = [stimOn_times - 0.5, stimOn_times + 1.5]` per trial, and both the spikes and the behavioural traces are cut with those.

ii.
```python
trials_df, trials_mask = U.load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)
trials_mask = np.asarray(trials_mask).astype(bool)
...
binned_spikes, clusters_used = U.bin_spiking_data(
    reg_clu_ids, neural_dict, trials_df=trials_df, n_workers=n_workers, **PARAMS)
```
with (reference `bin_spiking_data`, used unchanged):
```python
intervals = np.vstack([
    trials_df[kwargs['align_time']] + kwargs['time_window'][0],
    trials_df[kwargs['align_time']] + kwargs['time_window'][1]]).T
```

iii. Not discussed as a decision; the trials table is already one row per trial and the reference code defines the per-trial window, which the agent adopted wholesale ("identical params (`align_time='stimOn_times'`, `time_window=(-0.5,1.5)`, `binsize=0.02` → T=100)").

## 1-e. How are trials filtered based on quality controls?

i. Two stages, ANDed together.
(1) The reference trial mask, `U.load_trials_and_mask(one, eid, max_trial_len=10.0)` with all other defaults, exactly as reference `prepare_data`. That drops: reaction time (`firstMovement_times - stimOn_times`) < 80 ms or > 2 s; trial length (`feedback_times - goCue_times`) > 10 s; any NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; and no-response trials (`choice == 0`). The unbiased first block (`probabilityLeft == 0.5`) is *kept* (`exclude_unbiased=False`).
(2) A behavioural-coverage/finiteness check: a trial is dropped unless both the wheel-speed and the whisker-motion-energy trace exist for the full window (the reference `get_behavior_per_interval` returns `None` when the stream starts more than one bin late, ends more than one bin early, or is absent) and contain no NaN. The reference runs with `allow_nans=True`; the agent keeps that setting but then rejects NaN trials itself, because a NaN cannot be assigned to one of the three output categories and the decoder rejects non-finite values.
A session is dropped if fewer than 2 trials survive. Net effect: 188,554 trials over 443 sessions. The agent's own diagnostic showed stage (2) removes very little on top of stage (1) (e.g. one session: 148 trials pass the reference mask, 147 after the behaviour check).

ii.
```python
trials_df, trials_mask = U.load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)
trials_mask = np.asarray(trials_mask).astype(bool)
...
good = trials_mask.copy()
for beh in BEH_NAMES:
    traces = binned_beh[beh]
    ok = np.array([
        (tr is not None) and (np.asarray(tr).size == NBINS)
        and bool(np.all(np.isfinite(np.asarray(tr, dtype=float))))
        for tr in traces])
    good &= ok
keep = np.flatnonzero(good)
if len(keep) < MIN_TRIALS:
    raise RuntimeError(f'only {len(keep)} trials survive curation')
```

iii. Docstring: "Trial curation: `load_trials_and_mask(max_trial_len=10.0)`, exactly as reference `prepare_data`"; and "Behaviour curation: as in `align_spike_behavior`, trials whose behavioural trace could not be interpolated across the whole window are dropped. Trials whose wheel or whisker trace still contains a NaN are also dropped: the reference runs with `allow_nans=True`, but a NaN cannot be assigned a category, and the decoder rejects non-finite values." Trajectory step 53: the agent verified with a per-filter breakdown that the loss is "driven almost entirely by the reference quality mask".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` of every probe insertion of the session (from `SpikeSortingLoader.load_spike_sorting()`), plus the merged cluster table's `acronym` column, which is used only for the region labels (`brain_region_idx`), not for the counts. These are packed into the reference's `neural_dict`.

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
...
clusters_labeled = SpikeSortingLoader.merge_clusters(
    spikes, clusters, channels, compute_metrics=False).to_df()
...
spikes, clusters = U.merge_probes(spikes_list, clusters_list)
neural_dict = {
    'spike_times': spikes['times'],
    'spike_clusters': spikes['clusters'],
    'cluster_regions': clusters['acronym'].to_numpy(),
}
```

iii. This is literally the `neural_dict` the reference `prepare_data` builds; the agent reproduced it so that `list_brain_regions`, `select_brain_regions` and `bin_spiking_data` could be called unchanged.

## 2-b. How is the `neural` data processed?

i. All probes of a session are merged with the reference's `merge_probes` (cluster indices offset, spikes concatenated and re-sorted by time) "as if recorded by one probe", because probes of one session are not statistically independent. Spikes are then counted into non-overlapping 20 ms bins over the (-0.5, +1.5) s window with the reference `bin_spiking_data`, giving `(ntrials, 100, nclusters)`; the array is transposed per trial to `(n_neurons, 100)` and cast to float32. **The values are raw spike counts per 20 ms bin, not firing rates** (`metadata['neural_units'] = 'spike counts per 20 ms bin'`); no smoothing, no normalisation, no rebinning. Clusters that emit no spike anywhere in the session are implicitly dropped, since `bin_spiking_data` returns only `np.unique(spike_clusters)`.

ii.
```python
spikes, clusters = U.merge_probes(spikes_list, clusters_list)
...
binned_spikes, clusters_used = U.bin_spiking_data(
    reg_clu_ids, neural_dict, trials_df=trials_df, n_workers=n_workers, **PARAMS)
cluster_regions = np.asarray(beryl_reg)[clusters_used]
...
spk = binned_spikes[keep]                                 # (ntrials, T, nneurons)
...
    neural.append(np.ascontiguousarray(spk[i].T, dtype=np.float32))
```

iii. Docstring: "All probes of a session are merged as if recorded by one probe (`merge_probes`), as both papers do, because probes in one session are not statistically independent." Spike counts (rather than rates) follow the methods paper directly: "For each trial, we constructed X ∈ R^{N×T} by aggregating spike counts"; the agent also checked (trajectory step 36–42) that the decoder accepts float arrays and verified count magnitudes before choosing the dtype.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No unit-level quality control at all.** `load_clusters_and_spikes` is the reference `load_spiking_data` with the default `qc=None`, i.e. every sorted cluster is kept, including multi-unit clusters. No region-based exclusion either: `list_brain_regions` returns the full set of Beryl acronyms and `select_brain_regions(..., regions[0], ...)` selects all of them, so clusters whose Beryl acronym is `root` (85,960 units) or `void` (12,175 units, i.e. sites the histology placed outside the brain) are retained. The only unit that disappears is one with zero spikes in the session. The result is 599,213 "neurons" over 443 sessions (mean 1,353 per session) and a 106 GB pickle — versus the 75,708 well-isolated neurons the data paper reports after its stringent QC.

ii.
```python
def load_clusters_and_spikes(one, pid, eid, pname):
    """Spike sorting of one probe insertion, all clusters (no quality threshold).
    Same as utils.ibl_data_utils.load_spiking_data with qc=None, ..."""
    loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = loader.load_spike_sorting()
    ...
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=False).to_df()
    return spikes, clusters_labeled
```

```python
regions, beryl_reg = U.list_brain_regions(neural_dict, **PARAMS)
reg_clu_ids = U.select_brain_regions(neural_dict, beryl_reg, regions[0], **PARAMS)
```
and in the metadata:
```python
'spike_sorting': 'pykilosort, all clusters (no unit-quality threshold, qc=None)',
```

iii. Docstring: "Neural: pykilosort spike-sorted spikes of ALL clusters (reference `prepare_data` calls `load_spiking_data` with `qc=None`, i.e. no unit-quality threshold; methods paper: 'we bin spike counts using all neurons ... from each session')." Trajectory step 118: "all clusters (qc=None), Beryl region mapping" listed as part of "the reference pipeline". The agent never weighed the data paper's competing curation (75,708 well-isolated neurons after stringent QC) against the methods-paper/reference-code choice, nor considered dropping `void` units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. On `trials.stimOn_times`, with the window (-0.5, +1.5) s. All IBL streams are already on one synchronised session clock, so alignment is simply cutting each trial's spikes at `stimOn_times + time_window` and binning from that origin — done inside the reference `bin_spiking_data`, which builds `intervals = [stimOn - 0.5, stimOn + 1.5]` and calls `bincount2D` with `xlim=[t_beg, t_end]` per trial. `metadata['temporal_alignment_event'] = 'visual stimulus onset (trials.stimOn_times)'`, `off_start = -0.5`, `off_end = 1.5`.

ii.
```python
PARAMS = {
    'interval_len': 2,
    'binsize': 0.02,
    'single_region': False,
    'align_time': 'stimOn_times',
    'time_window': (-0.5, 1.5),
}
...
binned_spikes, clusters_used = U.bin_spiking_data(
    reg_clu_ids, neural_dict, trials_df=trials_df, n_workers=n_workers, **PARAMS)
```
```python
'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
'off_start': float(PARAMS['time_window'][0]),
'off_end': float(PARAMS['time_window'][1]),
```

iii. Docstring: "Alignment: `stimOn_times`, window (-0.5, +1.5) s, non-overlapping 20 ms bins → T = 100. These are exactly the params of `0_data_caching.py` and match the methods paper (2-s trials in 20-ms bins, T = 100; for choice 'from 0.5 s before to 1.5 s post-onset')." The instructions also require stimulus-onset alignment, so reference and task agree.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial, identical for every trial and session; `metadata['time_bin_size'] = 20.0` (ms). No rebinning, resampling or smoothing of the neural data — the spikes are binned once at 20 ms. The agent explicitly rejected the methods paper's 50 ms bins (used there only for the static choice/prior targets) because this task also requires time-varying wheel-speed and whisker outputs, for which both paper and code use 20 ms.

ii.
```python
NBINS = int(round((PARAMS['time_window'][1] - PARAMS['time_window'][0])
                  / PARAMS['binsize']))          # 100
...
'time_bin_size': PARAMS['binsize'] * 1000.0,
'n_timepoints': NBINS,
```

iii. Docstring: "The 20 ms bin size (rather than the 50 ms the paper uses for the purely static choice/prior targets) is required here because this task also asks for time-varying wheel-speed and whisker-motion-energy outputs, for which paper and code use 20 ms." This is also the `binsize` of `0_data_caching.py`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Not from any raw variable per se: it is the fixed 100-point time grid of the decoding window around `trials.stimOn_times`, defined by `time_window = (-0.5, 1.5)` and `binsize = 0.02`. The same vector is reused for every trial of every session.

ii.
```python
# time grid = end of each 20 ms bin relative to stimulus onset (-0.48 ... 1.50 s),
# the grid the reference code interpolates the behavioural traces onto.
tgrid = (np.arange(1, NBINS + 1) * PARAMS['binsize']
         + PARAMS['time_window'][0]).astype(np.float32)
```

iii. Docstring: "0 `time_from_stim_onset` — signed time (s) of the end of each 20 ms bin relative to stimulus onset: -0.48 ... +1.50. This is the same time grid the reference code interpolates the behaviour onto."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid. The convention chosen is the **right edge** of each 20 ms bin (-0.48, -0.46, …, +1.50 s), which is the grid the reference `get_behavior_per_interval` uses for behaviour (`x_interp = np.linspace(beg + binsize, end, n_bins)`), rather than bin centres. The vector is broadcast unchanged into every trial's `(2, 100)` input array as row 0.

ii.
```python
tgrid = (np.arange(1, NBINS + 1) * PARAMS['binsize']
         + PARAMS['time_window'][0]).astype(np.float32)
...
    inputs.append(np.stack([tgrid, np.full(NBINS, tinb[i], dtype=np.float32)]))
```

iii. Consistency with the reference behavioural interpolation grid, as stated in the code comment and docstring; the instructions ask for time since stimulus onset as a continuous time-varying input, and the grid is defined by the alignment window.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the same grid the neural bins are built on, index for index: neural bin *i* spans `[stimOn - 0.5 + 0.02·i, stimOn - 0.5 + 0.02·(i+1))` and `tgrid[i]` is the right edge of that bin. So the alignment is exact up to the (deliberate) choice of labelling a bin by its end rather than its centre — a fixed 10 ms offset relative to a bin-centre convention, identical for every trial, session and stream.

ii.
```python
tgrid = (np.arange(1, NBINS + 1) * PARAMS['binsize'] + PARAMS['time_window'][0])
```
against the reference binning grid used for the spikes:
```python
intervals = np.vstack([trials_df['stimOn_times'] - 0.5,
                       trials_df['stimOn_times'] + 1.5]).T   # inside bin_spiking_data
```

iii. Docstring: it is "the same time grid the reference code interpolates the behaviour onto", so inputs, outputs and neural bins all share one axis.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. `trials.probabilityLeft`, which is constant within a block, so a change of value marks a new block. No block identifier exists in the trials table.

ii.
```python
tinb = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())[keep]
```

iii. Not explicitly argued in the code, but the trajectory (step 33) lists "how to define block trial number" as one of three open questions the agent resolved by inspecting the block structure of an example session (step 38–39).

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based counter that increments while `probabilityLeft` is unchanged from the previous trial and resets to 0 when it changes. Crucially it is computed on the **full, unfiltered** trials table and only then indexed with the surviving-trial mask, so a dropped trial still advances the count and the value is the animal's true position in the block. The scalar is broadcast over all 100 bins as row 1 of the input array (float32, un-normalised; the agent checked that the decoder does no input normalisation, trajectory step 40–41).

ii.
```python
def trial_number_in_block(prob_left):
    """0-based index of each trial within its block of constant probabilityLeft."""
    p = np.asarray(prob_left, dtype=float)
    idx = np.zeros(len(p), dtype=np.float32)
    counter = 0
    for i in range(len(p)):
        if i > 0 and np.isclose(p[i], p[i - 1]):
            counter += 1
        else:
            counter = 0
        idx[i] = counter
    return idx
```
```python
tinb = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())[keep]
...
    inputs.append(np.stack([tgrid, np.full(NBINS, tinb[i], dtype=np.float32)]))
```

iii. Docstring: "1 `trial_number_in_block` — 0-based index of the trial within its block of constant `probabilityLeft`; constant within a trial, broadcast in time." Metadata: "trials elapsed since the start of the current `probabilityLeft` block (0-based)".

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table (+1 / -1 / 0).

ii.
```python
choice = trials_df['choice'].to_numpy()[keep]
```

iii. Trajectory steps 33–34 and 44–45: the agent grepped ibllib/the reference code for the sign convention and confirmed "choice == -1 is rightward, choice == +1 leftward" before writing the mapping.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Recoding only: `+1 (leftward) → 0`, `-1 (rightward) → 1`, matching the instructions (left = 0, right = 1). No-response trials (`choice == 0`) never reach this line because `load_trials_and_mask(exclude_nochoice=True)` has already removed them. The per-trial value is broadcast over the 100 bins as output row 0, with `output_values[0] = ['left', 'right']`.

ii.
```python
# IBL convention: choice == +1 -> leftward turn, choice == -1 -> rightward turn.
choice_out = np.where(choice == -1, 1, 0).astype(np.int64)
...
    outputs.append(np.stack([
        np.full(NBINS, choice_out[i], dtype=np.int64),
        ...
```

iii. Docstring: "0 `choice` — 0 = left, 1 = right. IBL convention: `choice == +1` is a leftward wheel turn, `choice == -1` a rightward one." Broadcasting per-trial variables in time is justified as "so one array holds both static and time-varying targets".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes the values 0.2, 0.5 and 0.8.

ii.
```python
pleft = trials_df['probabilityLeft'].to_numpy()[keep]
```

iii. Given by the instructions; the agent inspected the block structure of an example session (trajectory step 38–39) to confirm the three values.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A lookup 0.2→0, 0.5→1, 0.8→2 (the instruction's mapping), applied after rounding the float to one decimal to make the dictionary lookup safe. The unbiased 0.5 block is kept as its own class. The per-trial value is broadcast over the 100 bins as output row 1, with `output_values[1] = ['0.2', '0.5', '0.8']`. Note the lookup is strict: a `probabilityLeft` outside {0.2, 0.5, 0.8} would raise a `KeyError` and cause the whole session to be skipped (this did not happen on this release).

ii.
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}
...
prior_out = np.array([PRIOR_MAP[round(float(p), 1)] for p in pleft], dtype=np.int64)
```

iii. Docstring: "1 `prior_prob_left` — `probabilityLeft`, 0.2 → 0, 0.5 → 1, 0.8 → 2", i.e. the mapping the instructions prescribe.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position` / `_ibl_wheel.timestamps`, read by `SessionLoader.load_wheel()` inside the reference `load_target_behavior(one, eid, 'wheel-speed')`, which returns `|velocity|` on the interpolated wheel clock.

ii. In `convert_data.py` the stream is requested by name:
```python
BEH_NAMES = ['wheel-speed', 'whisker-motion-energy']
...
binned_beh, _ = U.bin_behaviors(
    one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
    n_workers=n_workers, **PARAMS)
```
and the reference helper does:
```python
elif target == 'wheel-speed':
    sess_loader.load_wheel()
    beh_dict = {'times': sess_loader.wheel['times'].to_numpy(),
                'values': np.abs(sess_loader.wheel['velocity'].to_numpy())}
```

iii. `'wheel-speed'` is the reference code's own target name and is one of the behaviours `0_data_caching.py` caches; the agent reused it unchanged.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages, all inherited from IBL/the reference code. (1) `SessionLoader.load_wheel()` interpolates the (event-driven) wheel position onto a uniform 1 kHz grid and differentiates it into a velocity with a Butterworth low-pass; the speed is `|velocity|` in rad/s. (2) The reference `get_behavior_per_interval` slices the trace to each trial window and linearly interpolates (`interp1d`, `fill_value='extrapolate'`) onto the 100-point grid `linspace(stimOn - 0.5 + 0.02, stimOn + 1.5, 100)`, i.e. the right edges of the 20 ms bins. (3) The resulting `(ntrials, 100)` float array of the retained trials is discretised into three classes (7-c). No filtering, normalisation or unit conversion is added.

ii.
```python
wheel = np.stack([np.asarray(binned_beh['wheel-speed'][i], dtype=float).ravel()
                  for i in keep])
...
wheel_out = discretize_tertiles(wheel)
```

iii. Docstring and metadata: the trace is the reference's, and "wheel speed and whisker motion energy are binned into 3 classes at the within-session 33.3rd and 66.7th percentiles of all retained samples". The interpolation/filtering is not an agent choice — it is what `SessionLoader` and `get_behavior_per_interval` do.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three classes at the **within-session** tertiles (33.3 / 66.7 percentiles) of all retained samples of that session (pooled over trials and time bins), giving roughly equinumerous low/medium/high classes per session. `np.searchsorted(edges, values, side='right')` is used rather than `np.digitize` so the mapping stays monotone and in range when the two edges coincide (e.g. a session where more than a third of wheel samples are exactly zero).

ii.
```python
def discretize_tertiles(values):
    """Map a (ntrials, T) float array to {0,1,2} using this session's pooled tertiles."""
    edges = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
    # searchsorted keeps the mapping monotone even when the two edges coincide (e.g. a
    # session in which more than a third of the wheel samples are exactly zero).
    return np.searchsorted(edges, values, side='right').astype(np.int64)
```

iii. Docstring: "Both signals are heavy-tailed and in session-specific units (wheel speed in rad/s; motion energy in arbitrary camera units that depend on the camera, illumination and ROI of that session), so fixed global thresholds would collapse whole sessions into one class. Session tertiles give three roughly equinumerous, interpretable low/medium/high classes in every session." The agent inspected the distributions of both signals on an example session before deciding (trajectory steps 38–39).

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is interpolated onto exactly the 100-point grid derived from the same `stimOn_times` and the same (-0.5, 1.5) window used to bin the spikes, sample *i* falling on the right edge of neural bin *i* — the same convention as the `time_from_stim_onset` input. Any trial whose wheel trace does not span the window (start later or end earlier than one bin from the edges) is dropped rather than extrapolated (see 1-e).

ii.
```python
binned_beh, _ = U.bin_behaviors(
    one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
    n_workers=n_workers, **PARAMS)     # PARAMS carries align_time, time_window, binsize
```
with, in the reference helper:
```python
x_interp = np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)
y_interp = interp1d(target_time, target_vals, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. All streams share the IBL session clock, so passing the same `PARAMS` (align event, window, bin size) to `bin_spiking_data` and `bin_behaviors` is the whole alignment — which is what the reference `align_spike_behavior` assumes.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with `_ibl_<side>Camera.times`, read by `SessionLoader.load_motion_energy(views=[side])`; the left camera is used if available, else the right, following the reference `bin_behaviors`, which tries `'left-whisker-motion-energy'` and falls back to `'right-whisker-motion-energy'` when the left load reports `skip`. The released whisker-pad motion-energy trace is used as-is.

ii. Requested by name (`'whisker-motion-energy'`), with the left/right fallback in the reference helper:
```python
if beh == 'whisker-motion-energy':
    target_dict = load_target_behavior(one, eid, 'left-whisker-motion-energy')
    if 'skip' in target_dict.keys():
        target_dict = load_target_behavior(one, eid, 'right-whisker-motion-energy')
```
and the availability pre-check in `convert_data.py`:
```python
for beh, targets in (('wheel-speed', ['wheel-speed']),
                     ('whisker-motion-energy', ['left-whisker-motion-energy',
                                                'right-whisker-motion-energy'])):
    if all('skip' in U.load_target_behavior(one, eid, t) for t in targets):
        raise RuntimeError(f'{beh} is not available for this session')
```

iii. Docstring: the reference `bin_behaviors` path is used, and "a handful of BWM sessions have no whisker motion energy at all (neither the left nor the right camera `ROIMotionEnergy` exists) ... the reference `bin_behaviors` then crashes inside `np.searchsorted`; `0_data_caching.py` catches that and skips the session. Whisker motion energy is a required decoder output here, so such a session cannot be used either way — checked up front so the skip is reported for what it is." 14 sessions were skipped for this reason.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. None on the trace itself: the released per-frame motion energy is taken as it is, sliced to the trial window and linearly interpolated onto the same 100-point grid as the wheel and the spikes, then discretised into three classes with the same session-tertile rule. No smoothing, normalisation, or z-scoring; no correction for the differing frame rates of the left (60 Hz) and right (150 Hz) cameras beyond the interpolation.

ii.
```python
whisk = np.stack([np.asarray(binned_beh['whisker-motion-energy'][i],
                             dtype=float).ravel() for i in keep])
...
whisk_out = discretize_tertiles(whisk)
```

iii. Docstring: "3 `whisker_motion_energy` — whisker-pad motion energy discretised into 3 bins", using the reference loader; the units are "arbitrary camera units that depend on the camera, illumination and ROI of that session", which is exactly why the discretisation is done per session.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: within-session tertiles (33.3 / 66.7 percentiles) of all retained samples of that session, pooled over trials and time bins, via the same `discretize_tertiles` helper; `output_values[3] = ['low', 'medium', 'high']`.

ii.
```python
whisk_out = discretize_tertiles(whisk)
```
(same function as in 7-c)

iii. Same justification as 7-c — heavy-tailed, session-specific arbitrary units, so a fixed global threshold would collapse whole sessions into a single class.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: the camera frame times are on the shared session clock, and `bin_behaviors` is called with the same `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`, `binsize=0.02`, so the trace is interpolated onto the right edges of the same 100 neural bins. Trials whose camera trace does not cover the window are dropped.

ii.
```python
binned_beh, _ = U.bin_behaviors(
    one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
    n_workers=n_workers, **PARAMS)
```

iii. Same as 7-d: one clock, one set of alignment parameters shared by all three streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several layers, mostly "drop and record":
- **Missing trial fields**: handled by the reference mask's NaN exclusion list (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`).
- **Missing / short / NaN behavioural traces**: the trial is dropped (1-e). `allow_nans=True` is kept so the reference's own interpolation path is unchanged, and the NaN rejection is done afterwards by the converter.
- **Session with no whisker motion energy at all** (neither camera): detected up front and the session is skipped with an explicit message (14 sessions).
- **Probe with no spike sorting**: `load_spike_sorting` returns `clusters is None`; an explicit `RuntimeError` is raised so the reason is reported — which skips the *whole session*, not just that probe (1 session).
- **Unused-but-crashing reads are removed**: the reference `load_spiking_data` calls `SpikeSortingLoader.raw_electrophysiology(band='ap')` purely to report the AP sampling frequency; it raises `ALFObjectNotFound` when the raw-ephys metadata is not cached, so the agent dropped the call "so a session is only ever skipped for a reason that actually affects the data". Likewise `prepare_data`'s `load_anytime_behaviors` (pupil etc.) is not called, since it errors on the read-only cache and its output is unused.
- **Session with < 2 surviving trials or 0 neurons**: skipped (1 session, plus an explicit `no neurons` guard).
- Every skip is caught per session, printed, and recorded in `metadata['skipped_sessions']` with its reason; 16 of 459 sessions were skipped.
Not handled: an unexpected `probabilityLeft` value would raise a `KeyError` and discard the entire session rather than the offending trial.

ii.
```python
    if clusters is None:
        raise RuntimeError(f'no spike sorting for probe insertion {pid} ({pname})')
```
```python
    if all('skip' in U.load_target_behavior(one, eid, t) for t in targets):
        raise RuntimeError(f'{beh} is not available for this session')
```
```python
    if binned_spikes.shape[2] == 0:
        raise RuntimeError('no neurons')
...
    if len(keep) < MIN_TRIALS:
        raise RuntimeError(f'only {len(keep)} trials survive curation')
```
```python
        try:
            sess = load_session(one, eid, args.n_workers)
        except Exception as exc:
            print(f'  SKIPPED {eid}: {exc!r}', flush=True)
            skipped.append({'eid': eid, 'reason': repr(exc)})
            continue
```

iii. Docstring: "Skipped sessions: a session is dropped when it has no wheel or no whisker-motion-energy trace at all (both required decoder outputs; the reference `0_data_caching.py` skips these sessions too, via its per-session try/except) or when fewer than 2 trials survive curation, or when one of its probe insertions has no spike sorting at all." Trajectory steps 117–119: the agent audited all 16 skipped sessions individually, confirmed the one spike-sorting failure had no usable sorting at all ("the reference `0_data_caching.py` would also skip this session"), and turned the resulting obscure `AttributeError` into an explicit guard.

## 10-a. What are the most time-consuming steps of the code?

i. Measured by the agent at ~9 s per session and ~5 sessions/minute for the full run (≈90 min for 459 sessions), the dominant costs are:
1. **Reading the spike sorting off disk** — `SpikeSortingLoader.load_spike_sorting()` per probe; the agent did not restrict `SPIKES_ATTRIBUTES`, so `amps` and `depths` are read as well as `times`/`clusters`, roughly doubling the bytes read for arrays that are never used.
2. **`bin_spiking_data`** — one multiprocessing task per trial over ~1,400 clusters, mitigated by `n_workers=16` but still the main CPU cost, and made ~8× larger than necessary by the absence of any unit QC.
3. **Loading the behavioural streams**, which is done twice (see 10-c) because of the availability pre-check.
4. **Serialising the result**: 106 GB of float32 spike counts written as one pickle at the end (and held in RAM until then) — by far the largest single I/O cost, and a direct consequence of the no-QC decision.

ii.
```python
    spikes, clusters, channels = loader.load_spike_sorting()
...
    binned_spikes, clusters_used = U.bin_spiking_data(
        reg_clu_ids, neural_dict, trials_df=trials_df, n_workers=n_workers, **PARAMS)
...
    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Not discussed in the code, but the trajectory shows the agent profiled a single session before launching ("One session takes ~9s and yields (407 trials, 100 bins, 898 neurons)", step 36) and projected the dataset size from a 3-session run ("3 sessions → 389 MB, so the full 459-session dataset will be ~60 GB (fits in 1 TB RAM / 3.4 TB disk)", step 55) — the actual result was 106 GB. Sessions are processed strictly serially; parallelism is only *within* a session (`n_workers=16` inside `bin_spiking_data` / `bin_behaviors`).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three.
1. `trial_number_in_block` is a plain Python `for` loop over all trials; it is exactly `trials.groupby(block).cumcount()` (or `np.arange - np.maximum.accumulate(...)`) and could be vectorised. It is cheap (a few hundred iterations) but it is the one loop the agent wrote itself.
2. The per-trial trial-quality list comprehension building `ok` (constructing an `np.asarray` per trial) could be a single array operation.
3. Inside the reused reference code: the per-trial spike-binning loop (`get_spike_data_per_interval`) and the per-trial interpolation loop (`get_behavior_per_interval`). Both could be done in one pass — the binning by offsetting each spike's bin index by its trial and using a single `bincount`, the interpolation by building one query vector for `np.interp`. The agent left them as they are and threw processes at them instead (`n_workers=16`), which pays the pickling cost of the trial data per task.
4. The final per-trial assembly loop (`for i in range(len(keep))`) does `np.stack`/`np.full` per trial; unavoidable given the required list-of-arrays output format.

ii.
```python
    counter = 0
    for i in range(len(p)):
        if i > 0 and np.isclose(p[i], p[i - 1]):
            counter += 1
        else:
            counter = 0
        idx[i] = counter
```
```python
        ok = np.array([
            (tr is not None) and (np.asarray(tr).size == NBINS)
            and bool(np.all(np.isfinite(np.asarray(tr, dtype=float))))
            for tr in traces])
```
```python
    for i in range(len(keep)):
        neural.append(np.ascontiguousarray(spk[i].T, dtype=np.float32))
        inputs.append(np.stack([tgrid, np.full(NBINS, tinb[i], dtype=np.float32)]))
```

iii. Never discussed; the agent's efficiency effort went into multiprocessing (`--n_workers 16`) and into a one-session timing check, not into vectorising loops. Keeping the reference loops untouched is consistent with its stated aim of reusing the reference implementation verbatim.

## 10-c. What processing does the code repeat multiple times?

i. **Every behavioural stream is loaded twice.** The availability pre-check calls `U.load_target_behavior` for `'wheel-speed'` and for `'left-whisker-motion-energy'` (and, only if the left camera fails, `'right-whisker-motion-energy'` as well, since `all(...)` short-circuits as soon as one load succeeds); each call builds a fresh `SessionLoader` and fully loads and processes the wheel (1 kHz interpolation + Butterworth differentiation) or the camera motion energy, and then throws the result away. `U.bin_behaviors` immediately loads all of them again.
Secondary repetitions: `SessionLoader` is instantiated separately by `load_trials_and_mask` and by each `load_target_behavior` call, so the trials table/session metadata is resolved several times; `merge_clusters` is run per probe and the Beryl mapping is applied over the full cluster table in `list_brain_regions` and then indexed again for `cluster_regions`.

ii.
```python
    for beh, targets in (('wheel-speed', ['wheel-speed']),
                         ('whisker-motion-energy', ['left-whisker-motion-energy',
                                                    'right-whisker-motion-energy'])):
        if all('skip' in U.load_target_behavior(one, eid, t) for t in targets):
            raise RuntimeError(f'{beh} is not available for this session')

    # --- bin the behavioural traces onto the same time grid --------------------------
    binned_beh, _ = U.bin_behaviors(
        one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
        n_workers=n_workers, **PARAMS)
```

iii. The duplication is a deliberate trade: the docstring explains the pre-check exists because the reference `bin_behaviors` "crashes inside `np.searchsorted`" on a session with no motion energy, and the agent wanted "the skip ... reported for what it is" rather than as an opaque exception. It did not consider reusing the already-loaded traces.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, the first being by far the most consequential:
1. **~525,000 clusters that no sensible downstream decoding would use.** With `qc=None`, every sorted cluster is kept, including multi-unit clusters and the 12,175 units whose Beryl acronym is `void` (histologically outside the brain) and 85,960 labelled `root`. This is 8× the 75,708 well-isolated neurons of the data paper, and it is the reason the output is a 106 GB pickle (mean 1,353 "neurons" per session) that must be loaded whole by the decoder.
2. **Unused spike attributes**: `load_spike_sorting()` reads `amps` and `depths` in addition to `times` and `clusters`; only the latter two are used.
3. **Full cluster metrics table**: `merge_clusters(...).to_df()` builds the whole labelled cluster dataframe per probe, of which only the `acronym` column is used (the `label` column, which the QC would need, is loaded and then ignored).
4. **The duplicated behavioural loads of 10-c.**
5. **`list_brain_regions` / `select_brain_regions`**: with `single_region=False` the "selection" is the identity (`reg_clu_ids` is every cluster), so the `np.isin` over all spikes in `bin_spiking_data` is a no-op filter over hundreds of millions of spikes.
6. Bins for trials that are later discarded: `bin_spiking_data` and `bin_behaviors` run over *all* trials of the session and the mask is applied afterwards, so on a session where the mask keeps 147 of 557 trials, ~74 % of the binning work is thrown away.

ii.
```python
regions, beryl_reg = U.list_brain_regions(neural_dict, **PARAMS)
reg_clu_ids = U.select_brain_regions(neural_dict, beryl_reg, regions[0], **PARAMS)

binned_spikes, clusters_used = U.bin_spiking_data(
    reg_clu_ids, neural_dict, trials_df=trials_df, n_workers=n_workers, **PARAMS)
...
spk = binned_spikes[keep]                                 # (ntrials, T, nneurons)
```
```python
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=False).to_df()
```

iii. Not identified as waste by the agent: items 1, 3 and 5 are direct consequences of its decision to "follow the reference implementation" (`qc=None`, `single_region=False`, `prepare_data`'s cluster table), and item 6 is the reference's own ordering (`bin_*` first, mask afterwards, as in `align_spike_behavior`). The agent did remove two genuinely unused reference calls (`raw_electrophysiology`, `load_anytime_behaviors`) because they crash on this cache; it noticed the size growth only as a resource question ("~60 GB ... fits in 1 TB RAM / 3.4 TB disk") rather than as unnecessary processing.
