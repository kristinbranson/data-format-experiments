# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read through the ONE API in local (offline) mode against the staged cache at `/app/data/one_cache`, using the same loader classes as the reference pipeline: `SessionLoader` for trials / wheel / camera motion energy and `SpikeSortingLoader` for each probe insertion. The set of sessions is not obtained from `one.search`; it is the 459 `eid`s of the brain-wide-map freeze shipped with the reference repo (`/app/code/code_zhang2025/data/bwm_release.csv`), which also supplies the `pid` and `probe_name` of every insertion, so `one.eid2pid` is never needed. That list is intersected with the session UUIDs actually present in the staged cache.

One significant departure: the AI found that the ONE release tables shipped in the container (`2022_Q4_IBL_et_al_BWM`, `Brainwidemap`, `2025_Q3_IBL_et_al_BWM`) list *unrevisioned* dataset paths while the files staged on disk sit under newer revision folders (`alf/#2025-03-03#/_ibl_trials.table.pqt`, `#2025-05-xx#` motion energy, `#2024-05-06#` spike sorting), and with no network ONE silently resolved only a one-column trials object and no motion energy. It therefore rebuilds `datasets.pqt` / `sessions.pqt` at the cache root by walking the staged `alf` trees, synthesising dataset UUIDs with `uuid5`, and flagging the lexicographically newest revision of each dataset as `default_revision`. After that rebuild all loading goes through the normal ONE API.

ii.
```python
CACHE_DIR = '/app/data/one_cache'
FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'

def get_one():
    """ONE client in local (offline) mode against the staged cache."""
    return ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True,
               cache_dir=CACHE_DIR, mode='local')
```

```python
    for f in (d / 'alf').rglob('*'):
        ...
        rev = rel_path_parts(rel, assert_valid=True)[1]
        base = rel.replace(f'#{rev}#/', '') if rev else rel
        rows.append((eid, rel, base, rev or '', f.stat().st_size))

ds = pd.DataFrame(rows, columns=['eid', 'rel_path', 'base', 'revision', 'file_size'])
newest = ds.groupby(['eid', 'base'])['revision'].transform('max')
ds['default_revision'] = ds['revision'].values == newest.values
```

```python
bwm_df = pd.read_csv(FREEZE_FILE, index_col=0)
available = set(one._cache['sessions'].index.astype(str))
eids = [e for e in bwm_df.eid.unique() if e in available]
```

```python
    ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
    sp, cl, ch = ssl.load_spike_sorting()
    ...
sess_loader = SessionLoader(one=one, eid=eid)
sess_loader.load_trials()
```

iii. From the trajectory (steps 24–59): the AI first tried `one.load_cache(tag='Brainwidemap')` and found trials returned a single column; it traced this to the revision mismatch, confirmed there was no network, and concluded "the cleanest fix is to rebuild the ONE cache tables directly from the files present on disk, keyed to the real eids from the sessions tables." Its docstring stresses "Nothing else about the loading is changed: the files read are the ones the reference pipeline reads." The session list is taken from `bwm_release.csv` because that is exactly the freeze the reference `0_data_caching.py` uses, and it already contains `pid`/`probe_name`, avoiding an Alyx round trip.

## 1-b. How are the data split into subjects?

i. No splitting is performed: `bwm_release.csv` carries a `subject` column, so the subject of a session is read straight off the freeze table and carried through in the per-session metadata. At assembly, `subjects` is the list of unique subject names in order of first appearance and `subject_idx` is each session's index into that list. Result: 136 subjects over 444 sessions.

ii.
```python
    meta = dict(eid=eid, subject=sub_df.subject.iloc[0], lab=sub_df.lab.iloc[0], ...)
```

```python
        sub = str(meta['subject'])
        if sub not in subjects:
            subjects.append(sub)
        subject_idx.append(subjects.index(sub))
```

iii. Not discussed explicitly in the trajectory; the freeze table already provides a unique subject identifier per session, so nothing has to be derived or parsed.

## 1-c. How are the data split into sessions?

i. A session is the natural unit of the release. The AI groups `bwm_release.csv` by `eid` (one row per probe insertion, so one group per session), keeps only eids staged locally, sorts them, and submits one job per session to a multiprocessing pool. Probes belonging to the same session are merged rather than treated as separate sessions.

ii.
```python
eids = [e for e in bwm_df.eid.unique() if e in available]
eids.sort()
...
jobs = [(e, bwm_df[bwm_df.eid == e]) for e in eids]
```

```python
    spikes, clusters = merge_probes(spikes_list, clusters_list)
```

iii. Module docstring: "Probes: all insertions of a session are merged (`merge_probes`), because the data paper states probes from one session 'are not independent' and must be combined." This mirrors `prepare_data` in `ibl_data_utils.py`, which also merges all probes of an eid.

## 1-d. How are the data split into trials?

i. No splitting is needed: the IBL trials table has one row per trial. It is loaded through `SessionLoader.load_trials()` and then re-used by the reference function `load_trials_and_mask`, which returns the same table plus a boolean mask. Each trial becomes one `(n_neurons, 100)` neural matrix, one `(2, 100)` input matrix and one `(4, 100)` output matrix.

ii.
```python
sess_loader = SessionLoader(one=one, eid=eid)
sess_loader.load_trials()
trials_df, trials_mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
```

iii. No justification is offered because no decision is required — the trials table is already one row per trial, and the AI deliberately reuses the reference repo's own loader (`from utils.ibl_data_utils import load_trials_and_mask, merge_probes`) rather than reimplementing it.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, ANDed together.

1. `load_trials_and_mask(one, eid, max_trial_len=10.0)` with all other arguments left at the reference defaults: reaction time (`firstMovement_times - stimOn_times`) in [0.08 s, 2 s]; no NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times` or `feedbackType`; `feedback_times - goCue_times < 10 s`; and `exclude_nochoice=True`, which drops trials where the mouse did not respond. `exclude_unbiased` is left `False`, so the 0.5 block is kept (it is needed for the three-way prior output).
2. A behavioural coverage test, reimplemented from `get_behavior_per_interval`: the trial is dropped unless the wheel trace *and* the whisker trace each have a sample within one bin (20 ms) of both edges of the [-0.5, 1.5] s window, and contain no NaN inside the window.

A session is dropped entirely if fewer than 2 trials survive. Result: 188,925 trials over 444 sessions.

ii.
```python
MAX_TRIAL_LEN = 10.0
...
trials_df, trials_mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
trials_mask = trials_mask.to_numpy()
```

```python
        if len(vv) == 0 or np.any(np.isnan(vv)):
            continue
        if np.abs(interval_begs[k] - tt[0]) > BINSIZE:      # starts too late
            continue
        if np.abs(interval_ends[k] - tt[-1]) > BINSIZE:     # ends too early
            continue
```

```python
    keep = trials_mask & wheel_ok & me_ok & ~np.isnan(align)
    if keep.sum() < 2:
        raise RuntimeError(f'only {keep.sum()} usable trials')
```

iii. Module docstring: "Trials: `load_trials_and_mask(..., max_trial_len=10.0)`, i.e. the data paper's trial exclusions … Trials whose behavioural traces do not cover the decoding window are dropped as well, which is what `get_behavior_per_interval` + `align_spike_behavior` do." The AI deliberately calls the reference repo's function verbatim with the same `max_trial_len=10.0` that `prepare_data` uses, rather than reimplementing the criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe insertion, `spikes['times']` and `spikes['clusters']`, returned by `SpikeSortingLoader.load_spike_sorting()`. The merged cluster table (from `SpikeSortingLoader.merge_clusters`) is used only for the `acronym` column, which supplies each unit's brain region; it is not used to filter units. `spikes['amps']` and `spikes['depths']` are also loaded (they are in the brainbox default `SPIKES_ATTRIBUTES`) but never used.

ii.
```python
    ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
    sp, cl, ch = ssl.load_spike_sorting()
    if len(sp) == 0:
        continue
    cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
    spikes_list.append(sp)
    clusters_list.append(cld)
```

```python
    st = spikes['times']
    sc = spikes['clusters']
```

iii. This is exactly what `load_spiking_data` in `ibl_data_utils.py` returns and what `prepare_data` packs into `neural_dict` as `spike_times` / `spike_clusters` / `cluster_regions`. The AI bypasses `load_spiking_data` itself only to avoid its `raw_electrophysiology(band="ap", stream=True)` call, which needs the network to report the sampling frequency.

## 2-b. How is the `neural` data processed?

i. All probes of a session are merged with the reference's `merge_probes`, which offsets the second probe's cluster ids and re-sorts the pooled spikes by time. Non-finite spike times/clusters are removed and the spikes are (re-)sorted. Spikes are then counted into 100 non-overlapping 20 ms bins spanning [stimOn - 0.5 s, stimOn + 1.5 s] for each kept trial, with a half-open [beg, end) window and the bin index clipped into range. The values stored are **raw spike counts per 20 ms bin** stored as `float32` — they are *not* divided by the bin width, and `metadata['neural_units']` records this as `'spike counts per 20 ms bin'`. No smoothing and no normalisation are applied.

ii.
```python
    spikes, clusters = merge_probes(spikes_list, clusters_list)
```

```python
    finite = np.isfinite(st) & np.isfinite(sc)
    st, sc = st[finite], sc[finite].astype(np.int64)
    order = np.argsort(st, kind='stable')
    st, sc = st[order], sc[order]
    ...
    binned = np.zeros((len(kidx), nneurons, NBINS), dtype=np.float32)
    for j, k in enumerate(kidx):
        i0 = np.searchsorted(st, beg[k], side='left')
        i1 = np.searchsorted(st, end[k], side='left')
        if i1 <= i0:
            continue
        b = ((st[i0:i1] - beg[k]) / BINSIZE).astype(np.int64)
        np.clip(b, 0, NBINS - 1, out=b)
        np.add.at(binned[j], (remap[sc[i0:i1]], b), 1.0)
```

iii. The docstring cites the methods paper: "Within each trial, we segment neural activity into … non-overlapping time bins. For each time bin, we bin spike counts using all neurons". The reference `get_spike_data_per_interval` likewise selects `(times >= t_beg) & (times < t_end)` and calls `bincount2D` to produce counts, not rates, so leaving the values as counts matches the reference. Probe merging is justified in the docstring by the data paper's statement that probes in one session are not independent.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No quality-control filter is applied.** Every Kilosort cluster of every insertion is kept; the only exclusion is clusters that emit no spike at all in the session (they never appear in `np.unique(spike_clusters)`), which is implicit in the reference's `get_spike_data_per_interval`. In particular the IBL `label` metric is not used, and units whose Beryl acronym is `void` (outside the brain) or `root` are kept. The delivered dataset therefore contains 599,865 units — 1,351 per session on average, up to 3,140 — including 12,175 `void` units and 86,127 `root` units, and the pickle is 106 GB.

ii.
```python
    # clusters that fire at least once, as in get_spike_data_per_interval
    used = np.unique(sc)
    remap = np.full(len(clusters), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nneurons = len(used)
```

```python
    acronyms = clusters['acronym'].to_numpy()[used]
    beryl = np.asarray(brainreg.acronym2acronym(acronyms, mapping='Beryl'))
```
(There is no `label >= 1` test and no `!= 'void'` test anywhere in the file.)

iii. Module docstring: "Neurons: every Kilosort cluster is kept. `prepare_data` calls `load_spiking_data` with the default `qc=None`, and the methods paper says 'we bin spike counts using all neurons, sorted by Kilosort 2.5, from each session'. Clusters that emit no spike at all in the session are dropped, which is what `get_spike_data_per_interval` does implicitly." The trajectory shows the AI was aware of the alternative: at step 62 it estimated "~100 GB … very heavy for the decoder. I need to check what the data paper says about unit quality control before deciding", read the passage about the 75,708 well-isolated neurons, and at step 118 wrote "If one epoch takes ~45+ min, 200 epochs is infeasible and I must reduce the dataset size (e.g. restrict to good-quality units as in the data paper's neuron inclusion criteria, which is also well justified)." Training eventually finished, so it kept all units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `stimOn_times`. All IBL streams (spike times, trial event times, wheel timestamps, camera frame times) are already expressed in seconds on one synchronised session clock, so alignment is simply a subtraction: per trial the window `[stimOn - 0.5, stimOn + 1.5]` is located in the sorted spike-time array with `searchsorted`, and the bin index of each spike is `floor((t - beg) / 0.02)`. Trials whose `stimOn_times` is NaN are dropped.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)          # seconds relative to stimulus onset
...
    align = trials_df[ALIGN_TIME].to_numpy(dtype=float)
    beg = align + TIME_WINDOW[0]
    end = align + TIME_WINDOW[1]
```

```python
        i0 = np.searchsorted(st, beg[k], side='left')
        i1 = np.searchsorted(st, end[k], side='left')
        b = ((st[i0:i1] - beg[k]) / BINSIZE).astype(np.int64)
```

```python
    keep = trials_mask & wheel_ok & me_ok & ~np.isnan(align)
```

iii. Docstring: "Alignment / binning: stimulus onset, window (-0.5, 1.5) s, 20 ms non-overlapping bins -> T = 100, exactly the `params` dict of `0_data_caching.py`." The instructions also mandate stimulus-onset alignment, so the reference code and the task specification agree. The `metadata` records `temporal_alignment_event: 'stimulus onset (stimOn_times)'`, `off_start: -0.5`, `off_end: 1.5`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 per trial over a 2 s window, identical for every trial and every session. `metadata['time_bin_size']` is 20.0 (ms). No rebinning, resampling or smoothing of the neural data is applied — the spikes are binned once, directly at 20 ms, straight from spike times.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)          # seconds relative to stimulus onset
BINSIZE = 0.02                     # seconds
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

```python
            'time_bin_size': BINSIZE * 1000.0,
```

iii. These are exactly the reference `params = {'interval_len': 2, 'binsize': 0.02, 'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}` from `0_data_caching.py`. Note that the methods paper uses 50 ms bins for choice/prior but 20 ms for the dynamic behaviours; the AI adopted the single 20 ms setting of the caching script, which is also what the decoder format requires (one bin size for all outputs).

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from raw data at all — it is the fixed bin grid defined by `TIME_WINDOW` and `BINSIZE` relative to `stimOn_times`. The value stored is the **right edge** of each 20 ms bin, i.e. `linspace(-0.48, 1.5, 100)`, identical for every trial in every session.

ii.
```python
# bin right edges relative to the alignment event; this is the grid that the
# reference code interpolates the behavioural traces onto
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)
```

```python
    inputs = np.empty((len(kidx), 2, NBINS), dtype=np.float32)
    inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. `metadata['input_descriptions']`: "seconds from stimulus onset to the right edge of each 20 ms bin, from -0.48 to 1.5 s". The right-edge convention was chosen to match the reference `get_behavior_per_interval`, which interpolates behaviour onto `np.linspace(interval_beg + binsize, interval_end, n_bins)`; using the same grid for the time input keeps one common time axis for inputs, outputs and neural bins.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None. A single 100-element vector is computed once at module import and broadcast to every trial of every session as row 0 of the input array, cast to `float32`. It is kept as a continuous ramp rather than a binary onset indicator, since the instructions list it as "continuous, time-varying".

ii.
```python
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)
...
    inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. N/A — the variable is defined by the alignment and binning choices, not measured. The Decoder Task specifies it as a continuous time-varying input, which is what a ramp provides.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. By construction they are on the same grid. Neural bin `j` of a trial covers `[stimOn - 0.5 + 0.02j, stimOn - 0.5 + 0.02(j+1))`, and `BIN_TIMES[j] = -0.5 + 0.02(j+1)` is the right edge of that same bin. The behavioural outputs are interpolated at `beg + BIN_TIMES - TIME_WINDOW[0]`, i.e. at those same instants, so all four streams share one axis bin for bin.

ii.
```python
        b = ((st[i0:i1] - beg[k]) / BINSIZE).astype(np.int64)     # neural bin index
```

```python
        x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]         # same instants
        values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. N/A — alignment is automatic because a single `BIN_TIMES` constant defines the grid for every stream.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. The trials table has no block identifier, so blocks are recovered from the fact that `probabilityLeft` is constant within a block: a change in its value starts a new block.

ii.
```python
def trial_in_block(pleft):
    """Index of each trial within its block of constant probabilityLeft."""
```

```python
    tib = trial_in_block(trials_df['probabilityLeft'].to_numpy())[kidx]
```

iii. Not commented beyond the docstring. Implicit justification: `probabilityLeft` is the only per-trial block variable the released trials table carries.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based running counter that resets whenever `probabilityLeft` changes from the previous trial. Crucially it is computed on the **full, unfiltered** trials table and only then indexed with `kidx`, so a trial that is later dropped still advances the counter and the stored number is the animal's true position within the block. The value is constant across the 100 bins of a trial (broadcast as a time-varying row) and stored as `float32`.

ii.
```python
def trial_in_block(pleft):
    out = np.zeros(len(pleft), dtype=np.int64)
    count = 0
    for i in range(len(pleft)):
        if i > 0 and pleft[i] != pleft[i - 1]:
            count = 0
        out[i] = count
        count += 1
    return out
```

```python
    tib = trial_in_block(trials_df['probabilityLeft'].to_numpy())[kidx]
    inputs[:, 1, :] = tib[:, None]
```

iii. `metadata['input_descriptions']`: "index of the trial within its block of constant probabilityLeft (0-based), constant within a trial". Broadcasting it across time is consistent with the target format's preference for time-varying arrays of uniform shape.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (leftward turn), -1 (rightward turn) or 0 (no response). No-response trials never reach this point because `load_trials_and_mask` is called with the default `exclude_nochoice=True`.

ii.
```python
    choice = trials_df['choice'].to_numpy()[kidx]
```

iii. Docstring comment in the code: "IBL convention: choice == +1 -> stimulus/response on the LEFT, choice == -1 -> RIGHT (verified against contrastLeft on correct trials)." Trajectory step 59/60: "Choice convention confirmed: choice==+1 corresponds to stimulus on the left (left choice), choice==-1 to right" — the AI checked this empirically against `contrastLeft` on correct trials rather than assuming it.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only a recode: `choice < 0` gives 1 for right and 0 for left, matching the instruction "left = 0, right = 1". The scalar is broadcast across all 100 bins and stored as `int64`.

ii.
```python
    choice_out = (choice < 0).astype(np.int64)             # left 0, right 1
    ...
    outputs[:, 0, :] = choice_out[:, None]
```

with `output_values[0] = ['left', 'right']`.

iii. As above: the mapping follows the IBL sign convention, verified empirically, and the direction (left = 0) is dictated by the Decoder Task specification.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes the three values 0.2, 0.5 and 0.8.

ii.
```python
    pleft = trials_df['probabilityLeft'].to_numpy()[kidx]
```

iii. `metadata['output_descriptions']`: "block probabilityLeft, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2". This is the block prior the task holds constant within a block; the mapping is given by the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A three-way recode with `np.select` and `np.isclose` (float-safe), giving 0/1/2; any other value raises, aborting the session rather than silently emitting a bad label. The value is broadcast across the 100 bins as `int64`. The unbiased 0.5 block is deliberately retained (`exclude_unbiased` left at `False`), which is why it must be a three-class output.

ii.
```python
    prior_out = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                           np.isclose(pleft, 0.8)], [0, 1, 2], default=-1)
    if np.any(prior_out < 0):
        raise RuntimeError('unexpected probabilityLeft value')
    ...
    outputs[:, 1, :] = prior_out[:, None]
```

with `output_values[1] = ['0.2', '0.5', '0.8']`. The realised distribution is 41.7 % / 14.1 % / 44.2 %.

iii. The mapping is prescribed by the Decoder Task; `np.isclose` and the hard failure on an unexpected value are defensive choices the AI added, not taken from the reference.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `sess_loader.wheel['times']` and `sess_loader.wheel['velocity']`, i.e. the `_ibl_wheel.position` / `_ibl_wheel.timestamps` datasets after `SessionLoader.load_wheel()` has interpolated the position onto an even 1000 Hz grid and differentiated it with a 20 Hz Butterworth low pass. Wheel speed is the absolute value of that velocity.

ii.
```python
    sess_loader.load_wheel()
    wheel_speed, wheel_ok = interp_behavior(
        sess_loader.wheel['times'].to_numpy(),
        np.abs(sess_loader.wheel['velocity'].to_numpy()), beg, end)
```

iii. Docstring: "wheel speed is |velocity| of the SessionLoader wheel trace … as in `load_target_behavior` / `get_behavior_per_interval`." This is byte-for-byte what the reference `load_target_behavior(one, eid, 'wheel-speed')` does.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps. (1) `SessionLoader.load_wheel()` produces the filtered 1000 Hz velocity; the absolute value is taken. (2) For each trial, the samples strictly inside `[beg, end]` are selected with `searchsorted` and linearly interpolated (`interp1d`, `fill_value='extrapolate'`) onto the 100 bin right-edges. A trial is rejected if there are no samples, if any sample is NaN, or if the first/last sample is more than one bin (20 ms) from the window edge. (3) The resulting `(n_trials, 100)` matrix is discretised into tertiles (see 7-c). No additional smoothing or normalisation.

ii.
```python
    idx_beg = np.searchsorted(target_times, interval_begs, side='right')
    idx_end = np.searchsorted(target_times, interval_ends, side='left')

    for k in range(ntrials):
        ...
        tt = target_times[idx_beg[k]:idx_end[k]]
        vv = target_vals[idx_beg[k]:idx_end[k]]
        if len(vv) == 0 or np.any(np.isnan(vv)):
            continue
        ...
        x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]
        values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
        good[k] = True
```

iii. `interp_behavior`'s docstring: "Mirrors `utils.ibl_data_utils.get_behavior_per_interval`: samples strictly inside the interval are used, a trial is rejected if the trace does not cover the interval to within one bin, and the values are linearly interpolated onto `linspace(beg + binsize, end, nbins)`." The one deliberate tightening is the NaN rejection: `0_data_caching.py` calls `bin_behaviors(..., allow_nans=True)`, whereas the AI always rejects NaN-containing windows, because a NaN in an output would break the tertile split.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three equal-occupancy classes (tertiles) computed **within each session**, over all bins of all kept trials of that session, using `np.nanquantile` at 1/3 and 2/3 followed by `np.digitize`. Duplicate edges are collapsed with `np.unique`, so a degenerate session yields fewer than three classes rather than an empty one. The realised marginal is 33.33 % / 33.33 % / 33.33 %.

ii.
```python
NBEHBINS = 3                       # tertiles for the continuous behaviours

def discretize(values, nbins=NBEHBINS):
    edges = np.nanquantile(values, np.arange(1, nbins) / nbins)
    edges = np.unique(edges)
    return np.digitize(values, edges).astype(np.int64)
```

```python
    wheel_bin = discretize(wheel_speed[kidx])
    ...
    outputs[:, 2, :] = wheel_bin
```

iii. `discretize` docstring: "Quantiles are computed within a session. Wheel speed and, above all, whisker motion energy are in units that are not comparable across sessions (motion energy depends on camera, illumination and ROI size), so a single global threshold would put whole sessions into one class. Equal-occupancy (tertile) edges keep the three classes usable in every session." `metadata['behaviour_discretisation']` records "equal-occupancy tertiles computed per session".

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is evaluated at exactly the 100 bin right-edges that define the neural bins of the same trial, measured from the same `stimOn_times`, on the same session clock. No lag or shift is introduced.

ii.
```python
        x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]
        values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```
where `interval_begs = align + TIME_WINDOW[0]` is the same `beg` used to index the spikes.

iii. Not explicitly argued; the wheel timestamps are already on the session clock (IBL synchronises the streams upstream), so sampling at the bin grid is all the alignment required. The coverage test in 7-b guarantees no bin is extrapolated by more than one bin width.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with its frame times `_ibl_<side>Camera.times`, loaded through `SessionLoader.load_motion_energy(views=[view])` and read from the `whiskerMotionEnergy` column. The left camera is preferred; the right camera is used if the left is missing, raises, or yields zero usable trials. 14 sessions were dropped outright for having no whisker motion energy at all.

ii.
```python
    me_vals, me_ok, me_view = None, None, None
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            df = sess_loader.motion_energy[cam]
            v, ok = interp_behavior(df['times'].to_numpy(),
                                    df['whiskerMotionEnergy'].to_numpy(), beg, end)
        except Exception:
            continue
        if ok.sum() > 0:
            me_vals, me_ok, me_view = v, ok, view
            break
    if me_vals is None:
        raise RuntimeError('no whisker motion energy')
```
The chosen view is recorded per session as `metadata['session_info'][i]['motion_energy_view']`.

iii. Docstring: "whisker motion energy is the left camera trace (right camera as fallback)". This reproduces `bin_behaviors`, which calls `load_target_behavior(..., 'left-whisker-motion-energy')` and falls back to the right camera when the loader reports `skip`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, no normalisation. It goes through the identical `interp_behavior` path as the wheel: samples strictly inside the window, rejection if empty / NaN / not covering both edges to within one bin, linear interpolation onto the 100 bin right-edges, then a per-session tertile split.

ii.
```python
            v, ok = interp_behavior(df['times'].to_numpy(),
                                    df['whiskerMotionEnergy'].to_numpy(), beg, end)
```

```python
    me_bin = discretize(me_vals[kidx])
```

iii. `metadata['output_descriptions']`: "whisker-pad motion energy of the left camera (right camera if left is unavailable), interpolated to each bin then split at the within-session tertiles." IBL computes the motion energy over a square ROI on the whisker pad upstream, so no further processing is warranted.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: three equal-occupancy classes at the 1/3 and 2/3 quantiles of that session's own interpolated trace, over all bins of all kept trials, via the same `discretize` helper. Realised marginal 33.26 % / 33.48 % / 33.26 %.

ii.
```python
    me_bin = discretize(me_vals[kidx])
    ...
    outputs[:, 3, :] = me_bin
```
with `output_values[3] = ['low', 'medium', 'high']`.

iii. Same `discretize` docstring as 7-c, which singles out motion energy as the variable that most needs a per-session threshold: "motion energy depends on camera, illumination and ROI size, so a single global threshold would put whole sessions into one class."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The camera trace is sampled at the same 100 bin right-edges as the neural data, from the same `stimOn_times`, on the same session clock — bin for bin identical to the neural and wheel axes. Camera frame rate (60 Hz left / 150 Hz right) is coarser than the 20 ms grid for the left camera, so the values are linear interpolations between frames; the coverage test guarantees frames exist within one bin of both window edges.

ii.
```python
        x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]
        values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. Not separately argued; the camera times are already synchronised to the session clock by the IBL pipeline, and the shared `BIN_TIMES` grid does the rest.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Failures are handled by dropping the affected unit of data and recording why, never by imputing.

- Probe with no spike sorting: skipped (`if len(sp) == 0: continue`); a session with no probe at all raises and is skipped.
- Non-finite spike times or cluster ids: masked out before binning.
- Clusters with zero spikes in the session: excluded from the neuron axis.
- Trial-level NaNs (`stimOn_times`, `choice`, `probabilityLeft`, `firstMovement_times`, `feedback_times`, `feedbackType`): dropped by `load_trials_and_mask`; `~np.isnan(align)` is re-applied defensively.
- Behavioural trace missing, NaN, or not spanning the window: that trial is dropped (`wheel_ok`, `me_ok`).
- No whisker motion energy for the whole session, or fewer than 2 usable trials: the session raises and is skipped. 15 of 459 sessions were dropped this way (14 with no whisker motion energy, 1 with 0 usable trials).
- An unexpected `probabilityLeft` value raises rather than being silently coded.
- Any other exception in a worker is caught, logged with the eid, and the session is skipped.
- The missing-revision problem in the ONE cache is repaired up front by `rebuild_cache_tables()` rather than being worked around per session.

Residual imperfection: three trials in one session ended up with an all-zero neural matrix (flagged as warnings by the verifier) and were kept.

ii.
```python
        if len(sp) == 0:
            continue
    if len(spikes_list) == 0:
        raise RuntimeError('no spike sorting')
```

```python
    finite = np.isfinite(st) & np.isfinite(sc)
    st, sc = st[finite], sc[finite].astype(np.int64)
```

```python
    keep = trials_mask & wheel_ok & me_ok & ~np.isnan(align)
    if keep.sum() < 2:
        raise RuntimeError(f'only {keep.sum()} usable trials')
```

```python
def _worker(args):
    eid, sub_df = args
    try:
        binned, inputs, outputs, beryl, meta = process_session(eid, sub_df)
    except Exception as e:                                  # noqa: BLE001
        return eid, None, f'{type(e).__name__}: {e}'
```

iii. `metadata['trial_inclusion']` documents the full drop rule. The trajectory (steps 48–55) shows the cache rebuild was diagnosed and fixed rather than patched around, and the run log prints an explicit reason for each of the 15 skipped sessions.

## 10-a. What are the most time-consuming steps of the code?

i. Three dominate, and the second and third are consequences of the decision to keep all ~600 k clusters (2-c).

1. **Reading the spike sorting from disk.** `ssl.load_spike_sorting()` is called once per insertion (699 in total) and is left at its default `check_hash=True`, so every spike file is re-read to compute its md5 — the smoke-test log is full of `local md5 mismatch on dataset` lines. It also loads the brainbox default `SPIKES_ATTRIBUTES = ['clusters', 'times', 'amps', 'depths']`, roughly doubling the bytes read versus the two arrays actually used.
2. **Serialising and staging the output.** Each session is written to an intermediate `.npz` (≈104 GB in total across `/app/work/sessions`), then read back and pickled into a single 106 GB file. From the run log, session processing across 32 workers took ~4 minutes while assembly + pickling took ~7 minutes.
3. **The per-trial binning loop**, specifically `np.add.at`, which is an unbuffered scatter-add and is roughly an order of magnitude slower than the `np.bincount` formulation.

Downstream the size dominates everything: `train_decoder.py` needed ~25 minutes just to load the pickle and run SVD initialisation before the first epoch.

ii.
```python
        sp, cl, ch = ssl.load_spike_sorting()          # check_hash left on, 4 attributes loaded
```

```python
        np.add.at(binned[j], (remap[sc[i0:i1]], b), 1.0)
```

```python
    f = TMP_DIR / f'{eid}.npz'
    np.savez(f, neural=binned, input=inputs, output=outputs, regions=beryl)
    ...
        z = np.load(TMP_DIR / f'{eid}.npz', allow_pickle=True)
    ...
    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
```

iii. Not analysed in the code or the trajectory. The AI did address throughput by parallelising over sessions with a 32-worker pool and `maxtasksperchild=4`, and by staging to `.npz` so the assembly step does not need every session in memory at once, but it never profiled the individual steps.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python-level loops, in decreasing order of cost.

1. The per-trial spike-binning loop. The `np.add.at` scatter-add could be replaced by a single `np.bincount` over a flat `unit * NBINS + bin` index (which is what the human reference does), or even one `bincount` for the whole session by folding the trial index into the flat index. This is the largest avoidable cost.
2. The per-trial loop in `interp_behavior`, which constructs a fresh `scipy.interpolate.interp1d` object per trial. A single `np.interp` call per trial, or one concatenated query vector for the session, would be considerably cheaper.
3. `trial_in_block`, a pure-Python loop over every trial of every session. It is a two-line vectorised operation: `block = (pleft != shift(pleft)).cumsum(); groupby(block).cumcount()`.

None of these changes the output; they are pure efficiency.

ii.
```python
    for j, k in enumerate(kidx):
        ...
        np.add.at(binned[j], (remap[sc[i0:i1]], b), 1.0)
```

```python
    for k in range(ntrials):
        ...
        values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

```python
    for i in range(len(pleft)):
        if i > 0 and pleft[i] != pleft[i - 1]:
            count = 0
        out[i] = count
        count += 1
```

iii. Not discussed. The AI's stated efficiency strategy was process-level parallelism (`mp.Pool(args.n_workers)`), and it noted at step 61 that "one session processes in ~4 s", which was fast enough that it did not pursue intra-session vectorisation.

## 10-c. What processing does the code repeat multiple times?

i. Several things are redone that need not be.

1. **`BrainRegions()` and `get_one()` are constructed inside `process_session`**, i.e. once per session (459 times) rather than once per worker process. `BrainRegions()` parses the Allen atlas tables and `ONE(..., mode='local')` re-reads the cache parquet tables each time. The human reference instead builds `BrainRegions` at module level and connects ONE once per worker through a pool `initializer`.
2. **The merged spikes are sorted twice.** `merge_probes` already ends with `sort_idx = np.argsort(merged_spikes['times'], kind='stable')`; `process_session` then runs `np.argsort(st, kind='stable')` again on the already-sorted array.
3. **Behavioural traces are interpolated for every trial**, including the ~20–30 % that `trials_mask` will discard; only `[kidx]` is used afterwards.
4. **The neural/input/output arrays are written to `.npz` and immediately read back** during assembly — a full 104 GB round trip through disk.
5. `sess_loader.load_trials()` is called and then `load_trials_and_mask` is passed the same `sess_loader`; this one is *not* duplicated work, because `load_trials_and_mask` re-loads only `if sess_loader.trials.empty`.

ii.
```python
def process_session(eid, sub_df):
    one = get_one()
    brainreg = BrainRegions()
```

```python
    order = np.argsort(st, kind='stable')        # merge_probes already sorted by time
    st, sc = st[order], sc[order]
```

```python
    wheel_speed, wheel_ok = interp_behavior(..., beg, end)   # all trials, then [kidx]
    ...
    wheel_bin = discretize(wheel_speed[kidx])
```

```python
    np.savez(f, neural=binned, input=inputs, output=outputs, regions=beryl)
    ...
        z = np.load(TMP_DIR / f'{eid}.npz', allow_pickle=True)
```

iii. Not discussed. The `.npz` round trip is a deliberate design choice — it lets `--skip_processing` re-run only the assembly stage, and it keeps peak memory bounded since the 106 GB dictionary is built incrementally rather than held by a pool of workers. The per-session `get_one()`/`BrainRegions()` construction appears to be an artefact of making `process_session` self-contained for `multiprocessing.Pool` (which cannot share a database connection across a fork) rather than a considered trade-off.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Five things, one of them very large.

1. **~524,000 units that the decoder cannot use.** Keeping every Kilosort cluster yields 599,865 units (1,351 per session, max 3,140) against the 75,708 well-isolated neurons the data paper curates. The decoder then random-projects sessions down to 2,000 neurons for SVD initialisation anyway (`Using random projection to 2000 of 3140 neurons`), so most of the extra data is discarded at training time — after costing 106 GB of storage and ~25 minutes of load/init. Among those retained are 12,175 units whose Beryl acronym is `void`, i.e. channels the histology placed outside the brain, and 86,127 `root` units.
2. **`spikes['amps']` and `spikes['depths']`** are loaded for every one of the 699 insertions and never referenced.
3. **md5 verification of every spike file** (`check_hash` left at its default `True`), which re-reads the data purely to compute a hash that is then ignored.
4. **Behavioural interpolation for trials that the trial mask discards** (see 10-c.3).
5. **`SpikeSortingLoader.merge_clusters(...).to_df()`** builds a full cluster DataFrame with metrics and histology for every cluster, of which only the `acronym` column is used.

ii.
```python
    used = np.unique(sc)                    # no label/QC filter, no 'void' filter
    nneurons = len(used)
```

```python
        sp, cl, ch = ssl.load_spike_sorting()          # loads clusters, times, amps, depths; check_hash=True
        cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
```

```python
    acronyms = clusters['acronym'].to_numpy()[used]    # only column used from `cld`
```

iii. For item 1 the AI's stated justification is fidelity to the reference code (`qc=None`) and the methods paper's "all neurons" wording — see 2-c. It was explicitly aware of the cost: step 62, "Full dataset with all clusters would be ~100 GB, which is very heavy for the decoder", and step 118, "I must reduce the dataset size (e.g. restrict to good-quality units as in the data paper's neuron inclusion criteria, which is also well justified)". It did not reduce it, because training eventually completed. Items 2–5 are not discussed anywhere.
