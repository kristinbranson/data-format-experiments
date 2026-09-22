# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read through the ONE API and the `brainbox` loaders; no file in `/app/data` is opened directly. A single `ONE` client is built per worker process exactly as in the reference caching script (`src/0_data_caching.py`): `ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True, cache_dir='/app/data/one_cache')`. The agent determined that this "remote"-mode client works fully offline because the staged cache ships 6,483 Alyx REST responses that expire in 2076, and that a `mode='local'` client would *not* work because the staged ALF files are later revisions than those listed in the local parquet tables.

The list of sessions is taken from the reference repo's release freeze file `/app/code/code_zhang2025/data/bwm_release.csv` (459 eids, 699 pids, 139 subjects, 12 labs) — the same file the reference caching script uses — deduplicated on `eid` and sorted. Everything else follows from the `eid`:
- trials: reference `load_trials_and_mask(one, eid, max_trial_len=10.0, sess_loader=SessionLoader(one=one, eid=eid))`, imported unchanged from `utils/ibl_data_utils.py`;
- wheel: `SessionLoader.load_wheel()`;
- whisker motion energy: `SessionLoader.load_motion_energy(views=[...])`;
- spikes: `one.eid2pid(eid)` then `SpikeSortingLoader(pid, one, eid, pname).load_spike_sorting()` once per probe, merged with the reference `merge_probes`.

Sessions are processed in parallel with a 24-worker `multiprocessing.Pool`, one session per task; the whole release converted in 200 s.

ii.
```python
CACHE_DIR = '/app/data/one_cache'
BASE_URL = 'https://openalyx.internationalbrainlab.org'
BWM_RELEASE = '/app/code/code_zhang2025/data/bwm_release.csv'

def get_one():
    global _ONE
    if _ONE is None:
        _ONE = ONE(base_url=BASE_URL, silent=True, cache_dir=CACHE_DIR)
    return _ONE
```
```python
from utils.ibl_data_utils import merge_probes, load_trials_and_mask
...
bwm = pd.read_csv(BWM_RELEASE, index_col=0)
sess_df = bwm.drop_duplicates('eid')[['eid', 'subject', 'lab']].sort_values('eid')
...
jobs = [(r.eid, r.subject, r.lab, args.show_processing) for r in sess_df.itertuples()]
with Pool(args.n_workers) as pool:
    for i, r in enumerate(pool.imap_unordered(process_session, jobs)):
```
```python
sess_loader = SessionLoader(one=one, eid=eid)
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
...
sess_loader.load_wheel()
sess_loader.load_motion_energy(views=[view])
...
pids, pnames = one.eid2pid(eid)
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
```

iii. From CONVERSION_NOTES Steps 2 and 5: the ONE call is "exactly as `src/0_data_caching.py`", chosen because it "works offline from the staged REST cache and resolves the staged dataset revisions (a `mode='local'` ONE cannot see them)". `bwm_release.csv` is used as the session list because it *is* the release freeze the reference code reads (`freeze_file = 'data/bwm_release.csv'`) and it agrees with the data paper's "459 sessions, 699 insertions, 139 mice". The decision to "convert **all** sessions that have the required data streams" (rather than the reference script's random `--n_sessions` subset) is recorded in the Step 4 discrepancy table.

## 1-b. How are the data split into subjects?

i. The subject name comes straight from the `subject` column of `bwm_release.csv`, carried alongside each `eid` through the worker job tuple and returned in the per-session result. At assembly, `subjects` is the sorted set of unique subject names of the sessions that survived, and `subject_idx` is each session's index into that list, in the same order as `neural`/`input`/`output`. 136 of the 139 released subjects survive; the 3 missing ones had only skipped sessions.

ii.
```python
sess_df = bwm.drop_duplicates('eid')[['eid', 'subject', 'lab']].sort_values('eid')
jobs = [(r.eid, r.subject, r.lab, args.show_processing) for r in sess_df.itertuples()]
```
```python
good.sort(key=lambda r: r['eid'])
...
subjects = sorted({r['subject'] for r in good})
sub_idx = np.array([subjects.index(r['subject']) for r in good], dtype=np.int64)
```

iii. Not explicitly argued — the release table already carries a unique subject id per session, so nothing has to be derived or parsed. The lab is also carried and stored in `metadata.session_info`, but is not used to split the data.

## 1-c. How are the data split into sessions?

i. No splitting is done: a session is the unit the release is organised by. `bwm_release.csv` has one row per probe insertion, so it is deduplicated on `eid` to give 459 sessions, sorted by `eid`, and one worker task is submitted per session. The two probes of a two-probe session are *merged into one session* rather than treated as two recordings.

ii.
```python
sess_df = bwm.drop_duplicates('eid')[['eid', 'subject', 'lab']].sort_values('eid')
```
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
```

iii. Step 5, key decision 3: "**Probes merged** per session (`merge_probes`), as both papers do ('neurons in the same session and region were combined across probes')" — i.e. two probes in one session are not statistically independent because they share the same behaviour.

## 1-d. How are the data split into trials?

i. No splitting is needed: the trials table returned by `SessionLoader.load_trials()` (via the reference `load_trials_and_mask`) has one row per trial. Each trial becomes the 2 s window `[stimOn_times - 0.5, stimOn_times + 1.5)`, identical in length for every trial and every session, so a trial is defined by the single number `stimOn_times`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
...
align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
finite_align = np.isfinite(align)
t0s_all = align + TIME_WINDOW[0]
```

iii. Step 5: "Trial geometry (identical for every trial and session) `t0 = stimOn_times - 0.5`, `t1 = stimOn_times + 1.5`, `binsize = 0.02 s`, `T = 100` bins" — declared "identical to the reference caching script `src/0_data_caching.py`" whose params are `align_time: 'stimOn_times'`, `time_window: (-.5, 1.5)`, `binsize: 0.02`.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, combined with a logical AND into a single `keep` mask:

1. The reference trial mask, from `load_trials_and_mask(max_trial_len=10.0)` **imported unchanged** from the reference module: first movement 0.08–2.0 s after stimulus onset, no NaN in `stimOn_times / choice / feedback_times / probabilityLeft / firstMovement_times / feedbackType`, `choice != 0` (no-go dropped), and `feedback_times - goCue_times <= 10 s`.
2. An explicit `np.isfinite(stimOn_times)` test (redundant with the mask, but guards the placeholder interval used to keep array shapes).
3. Behavioural coverage of the 2 s window for **both** the wheel and the selected camera, re-implementing the four criteria of the reference `get_behavior_per_interval`: data present in the interval, no NaN inside the interval, starts no more than one bin late, ends no more than one bin early.
4. Trials whose spike-count matrix is entirely zero are dropped (16 trials in 3 sessions), traced to gaps in the ephys recording or to trials occurring after the ephys recording ended.

The agent also noted that the reference `align_spike_behavior` contains a Python bug (`list and list` returns the second list) so that only the trials mask effectively survives, and deliberately implements the intended conjunction. Result: 187,936 trials over 441 sessions, mean 426.2/session (the reference mask alone keeps 426.5/session in the agent's independent survey).

ii.
```python
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
mask = mask.to_numpy()
...
keep = mask & finite_align & ws_valid & me_valid
n_keep = int(keep.sum())
if n_keep < MIN_TRIALS:
    return {'eid': eid, 'skip': f'only {n_keep} usable trials'}
```
```python
    for k in range(ntrials):
        a, b = i_beg[k], i_end[k]
        if b <= a:                                   # 'target data not present'
            continue
        if not np.all(finite[a:b]):                  # NaNs inside the interval
            continue
        if abs(t0s[k] - times[a]) > binsize:         # 'target data starts too late'
            continue
        if abs(t1s[k] - times[b - 1]) > binsize:     # 'target data ends too early'
            continue
        valid[k] = True
```
```python
nonempty = counts.sum(axis=(1, 2)) > 0
n_empty = int((~nonempty).sum())
if n_empty:
    kept_idx = np.flatnonzero(keep)
    keep[kept_idx[~nonempty]] = False
```

iii. Steps 3–5: the trial curation rules are "identical in paper and reference code `load_trials_and_mask`", so the reference function is used unchanged rather than re-implemented. The coverage criteria are "the reference `get_behavior_per_interval` criteria". The conjunction is applied because "`align_spike_behavior` contains a Python quirk … I implement the intended behaviour". The zero-spike drop is justified in Step 9 issue 4: "They are dropped: they carry no neural information and were the only remaining verifier warnings."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from `SpikeSortingLoader.load_spike_sorting()`, for every probe of the session. The cluster table produced by `SpikeSortingLoader.merge_clusters(...).to_df()` supplies two further columns used only for curation and labelling: `label` (the QC score) and `acronym` (the Allen acronym, mapped to Beryl for `brain_regions` / `brain_region_idx`).

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
if not clusters or not spikes:
    return None, None
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
iok = clusters_labeled['label'] >= qc
selected_clusters = clusters_labeled[iok]
spike_idx, ib = ismember(spikes['clusters'], selected_clusters.index)
selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
selected_spikes['clusters'] = selected_clusters.index[ib].astype(np.int32)
```
```python
beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
```

iii. Step 1 / Step 5 variable-mapping table: "`spikes.times`, `spikes.clusters` (all probes merged) → `neural[session][trial]`", via `prepare_data`, `merge_probes`, `load_spiking_data(qc=1)`, `bin_spiking_data`. The agent's `load_spiking_data` is described as "a copy of the reference function with `qc=1` … and without the network-only `raw_electrophysiology(...).fs` call", which was omitted because "raw `.cbin` files are not staged; this needs the network … Not used in any processing".

## 2-b. How is the `neural` data processed?

i. Spikes of all probes of a session are merged into one population with the reference `merge_probes` (cluster ids offset, spikes re-sorted by time), QC- and region-filtered (see 2-c), cluster ids remapped to `0..n_keep-1` preserving order, and spikes re-sorted by time. Then spikes are **counted** into 100 non-overlapping 20 ms bins per trial: bin *k* covers `[t0 + 0.02k, t0 + 0.02(k+1))`. The counts are stored as `float32` **spike counts, not converted to a rate** (no division by the bin width) and are **not smoothed and not z-scored** — the agent noted that the reference decoder z-scores internally (`standardize_spike_data`), so raw counts are cached.

The binning is a vectorised rewrite of the reference per-trial `bincount2D` loop: `np.searchsorted` locates each trial's spike slice, a flat `unit * nbins + bin` index is built, and one `np.bincount` fills the whole (unit × bin) grid. The agent verified it is **bit-identical** to the reference implementation over all 565 trials of a test session.

ii.
```python
def bin_spikes(spike_times, spike_clusters, t0s, n_neurons, nbins=NBINS, binsize=BINSIZE):
    ntrials = len(t0s)
    out = np.zeros((ntrials, n_neurons, nbins), dtype=np.float32)
    t1s = t0s + nbins * binsize
    i0 = np.searchsorted(spike_times, t0s, side='left')
    i1 = np.searchsorted(spike_times, t1s, side='left')
    for k in range(ntrials):
        a, b = i0[k], i1[k]
        if b <= a:
            continue
        bins = ((spike_times[a:b] - t0s[k]) / binsize).astype(np.int64)
        np.clip(bins, 0, nbins - 1, out=bins)
        idx = spike_clusters[a:b] * nbins + bins
        counts = np.bincount(idx, minlength=n_neurons * nbins)
        out[k] = counts.reshape(n_neurons, nbins).astype(np.float32)
    return out
```
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
...
remap = np.full(len(clusters), -1, dtype=np.int64)
remap[keep_ids] = np.arange(keep_ids.size)
sc = remap[sc]
order = np.argsort(st, kind='stable')
```

iii. Step 1 note: "The cached dataset stores *spike counts* per (trial, 20 ms bin, cluster); z-scoring happens in the decoder, so I will save raw spike counts in `neural`." Step 5 key decision 6: "20 ms counts, vectorised with `np.searchsorted` + `np.bincount` (mathematically identical to the reference `bincount2D` per-trial loop, verified by a direct comparison in Step 10)". Step 10 Check 3 records "spike counts identical to the reference implementation: True, max abs diff 0.0".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters and one session-level filter:

1. **`clusters.label >= 1`** — the IBL label is 0, 1/3, 2/3 or 1, one third per RIGOR single-unit metric (amplitude > 50 µV, noise cut-off < 20 µV, refractory-period violation), so `>= 1` selects the data paper's "well-isolated neurons". Measured: 890.1 clusters/probe before QC (paper: 889) and 108.5 units/probe after (paper: 108).
2. **Beryl acronym not in `{'root', 'void'}`** — the grey-matter restriction. This takes 73,010 → 62,757 neurons (142.3/session).
3. **Sessions with fewer than 5 surviving units are dropped** (`MIN_NEURONS = 5`); 3 sessions dropped (with 1, 2 and 3 units).

Probes whose spike sorting was never released return `clusters is None` and are skipped without losing the rest of the session.

ii.
```python
QC_LABEL = 1.0          # 'well-isolated' units: all three RIGOR single-unit metrics pass
NON_GREY = ('root', 'void')
MIN_NEURONS = 5         # data paper: >= 5 well-isolated neurons per session
```
```python
iok = clusters_labeled['label'] >= qc
selected_clusters = clusters_labeled[iok]
```
```python
beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
keep = ~np.isin(beryl, NON_GREY)
keep_ids = np.flatnonzero(keep)
if keep_ids.size == 0:
    return None
smask = np.isin(spikes['clusters'], keep_ids)
```
```python
if n_units < MIN_NEURONS:
    return {'eid': eid,
            'skip': f'only {n_units} well-isolated grey-matter units (< {MIN_NEURONS})'}
```

iii. Step 4 discrepancy table: the reference `prepare_data` calls `load_spiking_data` with `qc=None` (keeps every cluster) but records `good_clusters = label >= 1`, and `qc=1` is the reference's own provided "good units" path; the data paper's analyses use only the 75,708 well-isolated neurons. Resolution: "Use **label >= 1** … This matches the data paper's definition of 'neurons', is the reference code's own `qc=1` option, and keeps the converted dataset a manageable size (~13 GB vs ~90 GB)." For regions: "`MultiRegionDataModule.list_regions` drops `root` and `void` … Data paper restricts analyses to grey-matter Allen CCF regions → Map to Beryl and **drop units in `root`/`void`**". For `MIN_NEURONS`: Step 9 issue 3, "Sessions with < 5 well-isolated units (1, 2 and 3 units) produced many all-zero trials. The data paper requires at least five well-isolated neurons per session, so these sessions are now dropped."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams (spike times, trial event times, wheel timestamps, camera frame times) are already expressed in seconds on one synchronised session clock, so alignment is a subtraction. Each trial's window is `[stimOn_times - 0.5, stimOn_times + 1.5)`; the spike slice of that window is found with `np.searchsorted`, and the bin index of each spike is computed from `spike_time - t0` where `t0 = stimOn_times - 0.5`. Bin 25 is therefore the first bin at or after stimulus onset, and the time input row (see 3-c) runs from -0.49 to +1.49 s. Spikes exactly at the window end are excluded (`side='left'`), matching `bincount2D`.

ii.
```python
align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)   # ALIGN_TIME = 'stimOn_times'
t0s_all = align + TIME_WINDOW[0]                        # TIME_WINDOW = (-0.5, 1.5)
...
t0s = t0s_all[keep]
counts = bin_spikes(spike_times, spike_clusters, t0s, n_units)
```
```python
i0 = np.searchsorted(spike_times, t0s, side='left')
i1 = np.searchsorted(spike_times, t1s, side='left')
...
bins = ((spike_times[a:b] - t0s[k]) / binsize).astype(np.int64)
```
```python
'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
'off_start': TIME_WINDOW[0],
'off_end': TIME_WINDOW[1],
```

iii. Step 4: "the Decoder Task here mandates **stimulus-onset alignment**, which is exactly the reference caching configuration, so stimOn +/- (-0.5, 1.5) is used for all variables." Step 10 Check 5: "Bin edges: bin k is `[t0+0.02k, t0+0.02(k+1))`, the last bin ends exactly at stimOn+1.5 s; spikes exactly at the interval end are excluded (`side='left'`), matching `bincount2D`." Alignment was checked visually (wheel speed rises just after the plotted first-movement marker, which lies 0.08–2 s after the stimulus-onset line) and numerically (the behaviour grid agrees with the reference to 5.7e-14 s).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 per trial, for every trial and session — `metadata['time_bin_size'] = 20.0` ms. No rebinning, resampling or smoothing of the neural data: the spikes are counted directly into the final 20 ms grid in a single pass, so there is no intermediate resolution. (The continuous behavioural streams *are* resampled onto this grid; see 7-b/8-b.)

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```
```python
'time_bin_size': BINSIZE * 1000.0,
'n_time_bins': NBINS,
'neural_units': 'spike counts per 20 ms bin',
```

iii. Step 3: "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps" (methods paper) and `binsize: 0.02` in the reference caching script. Step 4 resolves the paper's mention of 50 ms bins for choice/prior: "Use **20 ms / T = 100**, i.e. the cached-dataset configuration. Required anyway because wheel speed and whisker ME must be decoded as time-varying outputs."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable — it is defined by the trial geometry, i.e. by `trials.stimOn_times` together with the chosen window and bin size. The value stored is the **centre of each 20 ms bin**, `-0.49, -0.47, …, 1.49` s, identical for every trial and session; the only per-trial content is the implicit fact that the grid is measured from that trial's `stimOn_times`.

ii.
```python
bin_centers = (TIME_WINDOW[0] + (np.arange(NBINS) + 0.5) * BINSIZE).astype(np.float32)
```
```python
INPUT_NAMES = ['time_from_stim_onset', 'trial_number_in_block']
```

iii. Step 5 variable-mapping table: "time relative to stimOn → `input[·][·][0, :]`; bin-centre time in seconds, -0.49 … 1.49 (same for every trial); (new; required by Decoder Task); 'Time since stimulus onset', continuous, time-varying; negative before onset". The window and bin size are the reference caching script's.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the bin-centre grid once and broadcasting it to every trial: `TIME_WINDOW[0] + (arange(100) + 0.5) * 0.02`, cast to `float32`. It is stored as a continuous signed time (not a binary onset indicator), so the sign tells the decoder whether a bin is before or after the stimulus.

ii.
```python
bin_centers = (TIME_WINDOW[0] + (np.arange(NBINS) + 0.5) * BINSIZE).astype(np.float32)
...
for i in range(n_keep):
    inputs.append(np.stack([bin_centers,
                            np.full(NBINS, tib[i], dtype=np.float32)]))
```

iii. Purely a definition; Step 10 Check 2 verifies it with an independent sanity check ("`time_from_stim_onset` row equals the bin centres `-0.49 ... 1.49` — PASS"), and Step 9's consistency table records the realised range `[-0.49, 1.49]`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural binning grid. Spikes are counted into bins `[t0 + 0.02k, t0 + 0.02(k+1))` with `t0 = stimOn_times - 0.5`, and `input[0, k]` is the centre of that same bin `k`, so column *k* of `neural` and column *k* of `input` describe the same 20 ms interval for every trial. Because the grid is defined relative to each trial's own `stimOn_times`, the alignment holds trial by trial.

ii.
```python
# neural
bins = ((spike_times[a:b] - t0s[k]) / binsize).astype(np.int64)
# input
bin_centers = (TIME_WINDOW[0] + (np.arange(NBINS) + 0.5) * BINSIZE).astype(np.float32)
```
```python
'alignment_note': ('bin k spans [stimOn - 0.5 + 0.02k, stimOn - 0.5 + 0.02(k+1)); '
                   'the time input is the bin centre; continuous behaviours are '
                   'interpolated at the bin right edge, as in the reference code'),
```

iii. The `alignment_note` metadata field states the convention explicitly. Step 5 "Trial geometry": "Bin *k* covers `[t0 + k*0.02, t0 + (k+1)*0.02)`; its centre is `t0 + (k+0.5)*0.02`."

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft` alone. The trials table carries no block identifier, so blocks are recovered as maximal runs of constant `probabilityLeft`; a change of value (including NaN, whose comparison is False) starts a new block.

ii.
```python
def trial_number_in_block(prob_left):
    """0-based index of each trial within its block of constant ``probabilityLeft``."""
    pl = np.asarray(prob_left, dtype=np.float64)
    changed = np.ones(len(pl), dtype=bool)
    changed[1:] = ~(pl[1:] == pl[:-1])       # NaN comparisons are False -> new block
    block_id = np.cumsum(changed) - 1
```
```python
tib_all, _ = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. Step 5 variable-mapping table: "`trials.probabilityLeft` → `input[·][·][1, :]`; trial index within the current block (0-based, counted over *all* trials of the session), broadcast along time; (new)". Step 2 verified the block structure in the data: "`trials.probabilityLeft` takes exactly the three values {0.2, 0.5, 0.8}; the first 90 trials of a session are the 0.5 unbiased block; biased blocks then alternate."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The 0-based position of a trial within its block. Crucially it is computed on the **full, unfiltered** trials table and only then subset by `keep`, so a trial that is later dropped by the QC mask still advances the counter and the stored number is the animal's real position in the block. The scalar is cast to `float32` and broadcast along all 100 time bins. Realised range `[0, 98]`, mean 31.2 — consistent with a 90-trial unbiased first block and biased blocks of 20–100 trials.

ii.
```python
    idx = np.zeros(len(pl), dtype=np.int64)
    for b in np.unique(block_id):
        m = block_id == b
        idx[m] = np.arange(m.sum())
    return idx, block_id
```
```python
tib_all, _ = trial_number_in_block(trials['probabilityLeft'].to_numpy())
tib = tib_all[keep].astype(np.float32)
...
inputs.append(np.stack([bin_centers, np.full(NBINS, tib[i], dtype=np.float32)]))
```

iii. Step 10 Check 5: "`trial_number_in_block` restarts at 0 at each change of `probabilityLeft`, including the transition out of the initial 90-trial unbiased block, and is computed over *all* trials (not only curated ones) so that the index reflects the animal's actual experience." Step 5 key decision 9 justifies the broadcast: "the format spec asks for time-varying representations 'if at all possible', and the decoder requires a consistent `dinput`/`doutput`." Verified independently in Step 10 Check 2 (PASS).

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The single column `trials.choice`, which is +1 (leftward choice), -1 (rightward choice) or 0 (no response). The agent independently verified the sign convention in the data: "all correct trials with `contrastLeft>0` have choice=+1". The 0 entries are removed with the trial by the reference mask (`exclude_nochoice=True`), so only ±1 reaches the mapping.

ii.
```python
choice = trials['choice'].to_numpy()[keep]
# trials.choice: +1 = left, -1 = right  ->  left = 0, right = 1
choice_cls = (choice < 0).astype(np.int32)
```
```python
OUTPUT_NAMES = ['choice', 'prior_prob_left', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [['left', 'right'], ...]
```

iii. Step 2: "`trials.choice`: +1 = mouse turned wheel so that a **left** stimulus was correct (verified: all correct trials with `contrastLeft>0` have choice=+1), -1 = **right** choice, 0 = no-go." The Decoder Task specifies left = 0, right = 1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding `+1 → 0`, `-1 → 1`, implemented as `(choice < 0)`, then broadcast along the 100 time bins and stored as `int32`. Realised distribution `[left 0.508, right 0.492]`, matching the agent's independent survey of the raw data (0.494 right).

ii.
```python
choice_cls = (choice < 0).astype(np.int32)
...
outputs.append(np.stack([
    np.full(NBINS, choice_cls[i], dtype=np.int32),
    ...]))
```

iii. Step 5 key decision 9: per-trial variables are broadcast along time so every trial has the same `(d, T)` shape. Step 10 Check 2 spot-checks: "choice class equals `int(trials.choice < 0)` of the matched raw trial — PASS".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The single column `trials.probabilityLeft`, verified to take exactly the three values {0.2, 0.5, 0.8}. It is matched with `np.isclose` (float tolerance) and mapped to {0, 1, 2}; any other value aborts the session (a guard that never triggered).

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()[keep]
prior_cls = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                       np.isclose(pleft, 0.8)], [0, 1, 2], default=-1).astype(np.int32)
if np.any(prior_cls < 0):
    return {'eid': eid, 'skip': 'unexpected probabilityLeft values'}
```
```python
OUTPUT_VALUES = [..., ['p(left)=0.2', 'p(left)=0.5', 'p(left)=0.8'], ...]
```

iii. Step 3: "Prior values: p(left) in {0.2, 0.5, 0.8} — data paper / verified in data". The 0.2→0, 0.5→1, 0.8→2 mapping is prescribed by the Decoder Task. Step 10 Check 5 lists the guard as an edge case: "`probabilityLeft` values other than {0.2, 0.5, 0.8} would abort the session (never triggered)."

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Only the recoding to {0, 1, 2}, broadcast along the 100 bins as `int32`. Realised distribution `[0.417, 0.140, 0.442]`, consistent with the agent's raw-data survey ([0.43, 0.16, 0.41] before behavioural filtering) and with 90 unbiased trials followed by alternating 0.2/0.8 blocks of mean length ~51.

ii.
```python
outputs.append(np.stack([
    np.full(NBINS, choice_cls[i], dtype=np.int32),
    np.full(NBINS, prior_cls[i], dtype=np.int32),
    ...]))
```

iii. Same as 5-b: broadcast so that the output is time-varying in shape, per Step 5 key decision 9. Verified independently in Step 10 Check 2 ("prior class equals the {0.2,0.5,0.8} → {0,1,2} map of the matched raw trial — PASS").

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position` and `_ibl_wheel.timestamps`, loaded through `SessionLoader.load_wheel()`, which returns a dataframe with `times`, `position`, `velocity` and `acceleration`. Wheel speed is `np.abs(velocity)` in rad/s — exactly the reference `load_target_behavior(one, eid, 'wheel-speed')`.

ii.
```python
sess_loader.load_wheel()
wheel_times = sess_loader.wheel['times'].to_numpy()
wheel_speed = np.abs(sess_loader.wheel['velocity'].to_numpy())
```

iii. Step 5 variable-mapping table: "`wheel.velocity` → `output[·][·][2, :]`; `abs(velocity)` … Reference code: `load_target_behavior('wheel-speed')`, `get_behavior_per_interval`". Step 3: "**Wheel speed** = `abs(velocity)` of the 1 kHz-resampled, Gaussian-smoothed wheel (`SessionLoader.load_wheel`)."

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages.

1. Inside `SessionLoader.load_wheel()` (not re-implemented): the raw, movement-triggered wheel position is interpolated onto a uniform 1000 Hz grid and differentiated into a velocity with smoothing. The agent verified this in Step 2 ("wheel from `SessionLoader.load_wheel()` is resampled to 1000 Hz with Gaussian-smoothed velocity").
2. `|velocity|` is linearly interpolated onto the 100 per-trial sample points `t0 + (1…100)·0.02` — the **bin right edges**, which is the grid the reference `get_behavior_per_interval` uses (`np.linspace(t_beg + binsize, t_end, n_bins)`). A single `np.interp` call over the flattened (trial × bin) grid does all trials at once. NaN samples are excluded from the interpolation source but are used for the per-trial validity test.
3. Discretisation into 3 classes (see 7-c).

One deliberate deviation from the reference: the reference restricts the interpolation source to samples *strictly inside* the interval and therefore **extrapolates** the last grid point; the agent interpolates from the full stream and uses the true neighbouring sample. The agent measured the effect (validity masks agree on 100% of trials; values agree to float32 precision in 99 of 100 bins; only the last bin differs, by 1.3e-2 rad/s on average) and kept the more accurate value.

ii.
```python
def bin_behavior(times, values, t0s, nbins=NBINS, binsize=BINSIZE):
    ntrials = len(t0s)
    t1s = t0s + nbins * binsize
    grid = t0s[:, None] + (np.arange(1, nbins + 1)[None, :]) * binsize
    ...
    finite = np.isfinite(times) & np.isfinite(values)
    i_beg = np.searchsorted(times, t0s, side='right')
    i_end = np.searchsorted(times, t1s, side='left')
    interp = np.interp(grid.ravel(), times[finite], values[finite]).reshape(grid.shape)
```
```python
ws, ws_valid = bin_behavior(wheel_times, wheel_speed, safe_t0)
...
ws_k, me_k = ws[keep], me[keep]
ws_cls, ws_edges = discretize_tertiles(ws_k)
```

iii. Step 5 key decision 7: "Behaviour binning: linear interpolation onto the bin right-edges `linspace(t0+0.02, t1, 100)`, identical to `get_behavior_per_interval`." Step 10 Check 3 documents and justifies the extrapolation difference: "Keeping the more accurate value is a deliberate improvement and affects 1% of bins."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Three equal-sized classes per session: the 33.3rd and 66.7th percentiles are computed over **all retained trials × all 100 bins of that session** (one pair of edges per session, computed once, not per trial), and `np.digitize` assigns 0/1/2. The edges are stored in `metadata.session_info[...]['wheel_speed_tertile_edges']`. Realised fractions `[0.333, 0.333, 0.333]`.

ii.
```python
NCLASSES = 3            # wheel speed / whisker motion energy discretisation

def discretize_tertiles(values):
    """Discretise a (ntrials, nbins) continuous signal into 3 per-session tertile bins."""
    edges = np.percentile(values, [100.0 / 3.0, 200.0 / 3.0])
    classes = np.digitize(values, edges, right=False).astype(np.int32)
    return classes, edges
```

iii. Step 5 key decision 8: "Per-session tertiles … Per-session rather than global because whisker motion energy is in arbitrary camera-dependent units (its session medians vary by an order of magnitude), so a global threshold would collapse whole sessions into one class; per-session tertiles also give the balanced 1/3 classes that balanced accuracy is defined against. Applied to wheel speed as well for consistency."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel is on the same session clock as the spikes, so alignment is the same subtraction: the resampling grid is `stimOn_times - 0.5 + (1…100)·0.02`, i.e. one sample per neural bin, measured from the same stimulus onset. Column *k* of the wheel-speed output therefore corresponds to neural bin *k*. The sample is taken at the **right edge** of bin *k* (following the reference code) rather than at the bin centre used for the `time_from_stim_onset` input — a 10 ms, i.e. half-bin, offset within the bin, explicitly recorded in `metadata['alignment_note']`.

ii.
```python
grid = t0s[:, None] + (np.arange(1, nbins + 1)[None, :]) * binsize
interp = np.interp(grid.ravel(), times[finite], values[finite]).reshape(grid.shape)
```
```python
'alignment_note': ('bin k spans [stimOn - 0.5 + 0.02k, stimOn - 0.5 + 0.02(k+1)); '
                   'the time input is the bin centre; continuous behaviours are '
                   'interpolated at the bin right edge, as in the reference code'),
```

iii. Step 10 Check 3: "(c) temporal alignment — intervals `stimOn + (-0.5, 1.5)`; behaviour grid `linspace(t_beg+binsize, t_end, 100)` → identical; verified numerically (grid max difference 5.7e-14 s)". Step 7: "binned wheel speed and whisker ME track the raw traces with no time offset; wheel speed rises just after the plotted first-movement marker, which itself falls 0.08–2 s after the stimulus-onset line at t = 0".

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with its frame times `_ibl_<side>Camera.times`, loaded via `SessionLoader.load_motion_energy(views=[view])`, which exposes a dataframe with `times` and `whiskerMotionEnergy`. The released trace is used as-is.

Camera choice differs from a plain left-preference: **both** cameras are loaded when available, each is binned, and the one that yields **more valid trials** is used (left preferred on a tie). Over the release this gives 424 sessions on the left camera and 17 on the right. Sessions with neither camera (14) are skipped.

ii.
```python
cameras = {}
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        df = sess_loader.motion_energy[cam]
        cameras[cam] = (df['times'].to_numpy(), df['whiskerMotionEnergy'].to_numpy())
    except Exception:
        continue
...
if not cameras:
    return {'eid': eid, 'skip': 'no whisker motion energy'}
```
```python
best = None
for cam in ('leftCamera', 'rightCamera'):
    if cam not in cameras:
        continue
    me_c, valid_c = bin_behavior(cameras[cam][0], cameras[cam][1], safe_t0)
    if best is None or valid_c.sum() > best[2].sum():
        best = (cam, me_c, valid_c)
camera_used, me, me_valid = best
```

iii. Step 9 issue 2: "Camera choice: taking the left camera whenever it exists (the literal reference fallback) gave 0 usable trials for `f8041c1e`, whose right-camera motion energy covers only part of the session. Fixed: when both cameras are available the one covering more trials is used (left preferred on a tie)." The base behaviour (left, falling back to right) is the reference `bin_behaviors` logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. None on the signal itself — no filtering, no normalisation. The released trace (left camera 60 Hz, right camera 150 Hz, as the agent verified) is linearly interpolated onto the same 100 bin-right-edge grid by the same `bin_behavior` function used for the wheel, with the same four validity criteria, and then discretised into 3 per-session tertile classes. The same last-bin extrapolation difference from the reference applies (mean abs diff 2.2 in whisker-ME units, 1 bin in 100).

ii.
```python
me_c, valid_c = bin_behavior(cameras[cam][0], cameras[cam][1], safe_t0)
...
me_k = me[keep]
me_cls, me_edges = discretize_tertiles(me_k)
```

iii. Step 3: "**Whisker motion energy** = `leftCamera.whiskerMotionEnergy` (`SessionLoader.load_motion_energy(views=['left'])`), with fallback to the right camera, as in `bin_behaviors`." Step 5 variable-mapping table: "interpolated to the 100 bin right-edges, then discretised into 3 per-session tertile bins".

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to wheel speed: the same `discretize_tertiles` function, 33.3rd/66.7th percentiles over all retained trials × bins of that session, one pair of edges per session, stored in `metadata.session_info[...]['whisker_me_tertile_edges']`. Realised fractions `[0.333, 0.333, 0.335]`.

ii.
```python
me_cls, me_edges = discretize_tertiles(me_k)
```
```python
'discretization': ('wheel speed and whisker motion energy: 3 classes at the '
                   'per-session 33.3/66.7 percentiles over all kept trials x bins'),
```

iii. Step 5 key decision 8 — the per-session choice is argued *primarily* for whisker ME: "whisker motion energy is in arbitrary camera-dependent units (its session medians vary by an order of magnitude), so a global threshold would collapse whole sessions into one class".

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The camera frame times are on the same session clock as the spikes, so the trace is simply evaluated at the same 100 per-trial grid points `stimOn_times - 0.5 + (1…100)·0.02`, bin right edges, one per neural bin. Column *k* of the whisker output therefore corresponds to neural bin *k*, with the same half-bin (10 ms) right-edge convention as the wheel.

ii.
```python
grid = t0s[:, None] + (np.arange(1, nbins + 1)[None, :]) * binsize
interp = np.interp(grid.ravel(), times[finite], values[finite]).reshape(grid.shape)
```
```python
if abs(t0s[k] - times[a]) > binsize:         # 'target data starts too late'
    continue
if abs(t1s[k] - times[b - 1]) > binsize:     # 'target data ends too early'
    continue
```

iii. As for the wheel: Step 10 Check 3 confirms the grid matches the reference to 5.7e-14 s, and the `--show-processing` plots were inspected to confirm the binned trace tracks the raw trace with no offset.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or defective data is dropped at the smallest granularity that is still valid, and every case is recorded in the metadata:

- **Probe with no released spike sorting** (`clusters is None`): the probe is skipped and the session is kept with its remaining probes; the probe name is recorded in `metadata.session_info[...]['probes_without_spike_sorting']`. (This case previously raised and lost a whole session.)
- **Session with no whisker motion energy from either camera**: skipped (14 sessions).
- **Camera that covers only part of the session**: the better-covering camera is used; if neither covers any curated trial, the session is skipped (1 session).
- **NaN `stimOn_times`**: excluded by the reference mask and again by an explicit `np.isfinite` test; a placeholder interval (the median `t0`) is substituted only to keep array shapes and is never kept.
- **NaN inside a behavioural interval / trace that starts late or ends early**: that trial is dropped (reference criteria).
- **Trials in ephys recording gaps** (all-zero spike counts, 16 trials in 3 sessions): dropped.
- **Sessions with < 5 well-isolated grey-matter units** (3) or **< 2 usable trials** (`MIN_TRIALS = 2`, needed for a train/validation split): skipped.
- **Unexpected `probabilityLeft` values**: abort the session (guard, never triggered).
- Any uncaught exception in a session is caught, converted to a skip record with a truncated traceback, and does not abort the run.

All 18 skipped sessions and their reasons are listed in `metadata['skipped_sessions']` and in `conversion_full_out.txt`.

ii.
```python
if not clusters or not spikes:
    # No spike sorting available for this insertion (e.g. it was not resolved);
    # the probe is skipped and the remaining probes of the session are used.
    return None, None
```
```python
safe_t0 = np.where(finite_align, t0s_all, np.nanmedian(t0s_all[finite_align]))
...
keep = mask & finite_align & ws_valid & me_valid
```
```python
except Exception as e:  # noqa: BLE001
    import traceback
    return {'eid': eid, 'skip': f'error: {e}', 'traceback': traceback.format_exc()[-800:]}
```
```python
'skipped_sessions': [{'eid': r['eid'], 'reason': r['skip']} for r in skipped],
```

iii. Step 9 "Issues found and fixed" and Step 10 Check 5 "Check for edge cases" enumerate each case with the investigation behind it (e.g. "a 5.1 s gap in `b182b754`", "`8c2f7f4d`, 3 trials" after the ephys recording ended). The result is that `verification_full_out.txt` reports "Data format is valid, no errors or warnings."

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting off disk, by a wide margin. Per-session timings collected by the script itself: spike-sorting load 3.5–8.3 s (scaling with the number of probes), trials + behaviour load ~0.8 s, all binning + assembly ~0.05 s. Whole-session wall times printed during the full run range from ~15 s to 33 s under 24-way parallelism. The full conversion took 200 s wall-clock for 459 candidate sessions; writing the 11.43 GB pickle is the other notable cost.

ii.
```python
t = time.time()
neural = load_session_neural(one, eid)
timings['spikes'] = time.time() - t
```
```python
out = {..., 'timings': timings, 'total_time': time.time() - t_start}
```
```python
print(f"  {r['eid'][:8]}: " + (... f"{r['total_time']:.1f}s, timings={ {k: round(v,2) for k,v in r['timings'].items()} }"))
```

iii. Step 7 "Run Time Estimates": "spike sorting load (dominant, scales with n probes) 3.5–8.3 s → 40–60 min serial", against "binning + assembly 0.05 s → < 1 min". Step 6: "Result: 4–9 s per session (spike-sorting load dominates: 3.5–8.3 s), 0.01–0.03 s for all binning."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops remain, all cheap relative to I/O:

1. `bin_spikes`: a `for k in range(ntrials)` loop over trials. Could be a single `np.bincount` over all trials at once by folding the trial index into the flat index (`(trial * n_neurons + unit) * nbins + bin`), as the spike slices are contiguous.
2. `bin_behavior`: a `for k in range(ntrials)` loop that only computes the *validity* flags; the four criteria are all elementwise and could be evaluated as array expressions over `i_beg`/`i_end`. (The interpolation itself is already fully vectorised in one `np.interp`.)
3. `trial_number_in_block`: `for b in np.unique(block_id)` with a boolean mask per block — O(n_blocks × n_trials). The standard vectorised form is `arange(n) - repeat(cumsum(counts) - counts, counts)`, or `pandas.groupby(block).cumcount()` (which is what the reference-style one-liner would be).
4. The per-trial assembly loop `for i in range(n_keep)` that `np.stack`s the input/output rows — unavoidable given that the target format is a *list* of per-trial arrays, but the broadcast `np.full(NBINS, x)` calls could be replaced by slicing one pre-built array.

ii.
```python
    for k in range(ntrials):
        a, b = i0[k], i1[k]
        ...
        counts = np.bincount(idx, minlength=n_neurons * nbins)
        out[k] = counts.reshape(n_neurons, nbins).astype(np.float32)
```
```python
    for b in np.unique(block_id):
        m = block_id == b
        idx[m] = np.arange(m.sum())
```
```python
        for i in range(n_keep):
            inputs.append(np.stack([bin_centers,
                                    np.full(NBINS, tib[i], dtype=np.float32)]))
```

iii. The agent's position (Step 6) is that these are not worth vectorising further: it already replaced the reference's per-trial multiprocessing tasks with per-session vectorisation, bringing all binning down to 0.01–0.03 s/session, so the remaining loops are negligible against the 3.5–8.3 s spike load. "Code inefficiencies identified: the reference code bins each trial in a separate multiprocessing task … which costs more in task overhead than the binning itself."

## 10-c. What processing does the code repeat multiple times?

i. Three genuine repetitions, all modest:

1. **`BrainRegions()` is constructed inside `load_session_neural`, i.e. once per session** (441 times), instead of once at module import. Each construction reads and builds the Allen/Beryl region tables.
2. **Both cameras are loaded and binned for every session that has both.** `SessionLoader.load_motion_energy` is called for `left` and `right`, and `bin_behavior` is run on both, so that the better-covering one can be chosen; the losing camera's work is thrown away. This is ~2× the camera I/O and binning for ~430 sessions.
3. **Behaviour is interpolated for every trial of the session, including trials that the trials mask has already rejected** — `bin_behavior` is called on `safe_t0` (all trials), and only afterwards subset by `keep`. Roughly 1.5× more interpolation than needed (296,090 raw vs 187,936 kept trials).

Conversely, several potential repetitions are explicitly avoided: the `ONE` client is a per-process singleton; `SessionLoader` is shared between `load_trials_and_mask` and the wheel/camera loads so trials are read once; the spike sorting is read once per probe; and the tertile edges are computed once per session rather than per trial.

ii.
```python
beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
```
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
```
```python
safe_t0 = np.where(finite_align, t0s_all, np.nanmedian(t0s_all[finite_align]))
ws, ws_valid = bin_behavior(wheel_times, wheel_speed, safe_t0)
```

iii. Not discussed as repetition in CONVERSION_NOTES. The camera duplication is a deliberate cost accepted in Step 9 issue 2 to recover sessions whose left-camera motion energy has poor coverage. Binning all trials before masking is a consequence of combining the behavioural-validity mask with the trials mask (`keep = mask & finite_align & ws_valid & me_valid`), which requires validity to be known for every trial.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.

1. **Unused spike attributes are read from disk.** `SpikeSortingLoader.load_spike_sorting()` is called without restricting `brainbox.io.one.SPIKES_ATTRIBUTES`, so `spikes['depths']` and `spikes['amps']` are loaded (and carried through the `{k: v[spike_idx] for k, v in spikes.items()}` subsetting and through `merge_probes`) even though only `times` and `clusters` are ever used. Since spike-sorting I/O is the dominant cost (10-a), this is the largest piece of wasted work.
2. **The losing camera's motion energy** is loaded and binned and then discarded (see 10-c).
3. **Behaviour interpolated for masked-out trials** (see 10-c).
4. `bin_behavior` writes `vals[~valid] = 0.0` for invalid trials; those trials are then dropped by `keep`, so the zero-fill is never read.
5. `trial_number_in_block` returns `block_id`, which the caller discards (`tib_all, _ = ...`).
6. Diagnostic quantities computed for every session but only used for reporting: `mean_rate`, `n_clusters_all`, `n_good_units`, `n_probes`, `n_trials_raw`, `n_trials_mask`.
7. The 100-element `bin_centers` row is materialised and stored for **every one of the 187,936 trials**, although it is identical everywhere — a pure storage cost in the 11.43 GB pickle rather than a compute cost.

ii.
```python
selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}   # keeps depths/amps too
```
```python
vals = interp.astype(np.float32)
vals[~valid] = 0.0
return vals, valid
```
```python
tib_all, _ = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```
```python
'mean_rate': float(counts.mean() / BINSIZE),
```

iii. Not discussed in CONVERSION_NOTES; the agent's efficiency narrative (Step 6, Step 7) is about binning and parallelism rather than about trimming loaded fields. Items 5–6 are deliberate bookkeeping used for the consistency tables in Steps 9–10 (e.g. `n_clusters_all` / `n_good_units` produce the 890.1 and 108.5 per-probe figures compared against the data paper's 889 and 108). Item 7 follows from Step 5 key decision 9, which broadcasts everything to `(d, T)` so that the decoder sees a consistent shape.
