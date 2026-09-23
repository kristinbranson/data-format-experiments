# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All data are read through the ONE API / `brainbox` loaders, never by opening files in `/app/data`. The list of sessions and probe insertions is not obtained from `one.search`, but from the reference repository's own release freeze file `/app/code/code_zhang2025/data/bwm_release.csv` (the same file `src/0_data_caching.py` reads), grouped by `eid`; the AI chose this because `one.eid2pid` requires a live Alyx connection, which is unavailable offline. This yields 459 sessions / 699 pids / 139 subjects. A prerequisite fix was needed first: the staged ONE cache tables did not index the dataset revisions actually present on disk (e.g. `alf/#2025-03-03#/_ibl_trials.table.pqt`), so ONE silently returned a one-column trials table; `build_one_cache.py` rebuilds the parquet tables from disk with `one.alf.cache.make_parquet_db` while preserving the true eids, and `convert_data.py` invokes it automatically if the tables are missing. Per session, `SessionLoader` supplies trials, wheel and camera motion energy, and `SpikeSortingLoader` is called once per probe insertion. Sessions are processed in a 32-process `multiprocessing.Pool`, one ONE client and one `BrainRegions` per worker.

ii.
```python
def build_jobs(sample):
    bwm = pd.read_csv(BWM_FREEZE, index_col=0)
    jobs = []
    for eid, g in bwm.groupby('eid', sort=True):
        probes = list(zip(g['pid'], g['probe_name']))
        jobs.append((eid, g['subject'].iloc[0], probes))
```

```python
def get_one():
    """One ONE instance and one BrainRegions per process (both are expensive)."""
    global _ONE, _BR
    if _ONE is None:
        _ONE = ONE(base_url='https://openalyx.internationalbrainlab.org',
                   silent=True, mode='local')
        _BR = BrainRegions()
    return _ONE, _BR
```

```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
...
for pid, pname in probes:
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
```

```python
def ensure_cache_tables():
    tables = Path('/app/data/one_cache/sessions.pqt')
    if not tables.exists():
        print('ONE cache tables missing; rebuilding from the staged cache ...')
        import build_one_cache
        build_one_cache.main()
```

iii. From CONVERSION_NOTES.md Step 10 Check 3: "Same loaders and the same `merge_probes`; pids come from `bwm_release.csv` (the same freeze file the reference reads) because `one.eid2pid` needs a live Alyx connection, unavailable offline". Key Decision 1: "Rebuild the ONE cache tables from disk. Without it the trials table and motion energy cannot be loaded at all. The genuine eids are preserved so that `bwm_release.csv`'s pid↔eid mapping — which the reference code relies on — still works."

## 1-b. How are the data split into subjects?

i. The subject name is taken from the `subject` column of `bwm_release.csv`, carried along with each session job, and never parsed out of a path. At assembly the subject list is the sorted unique set of names of the successfully converted sessions, and `subject_idx` is each session's index into that list. 136 of the 139 released subjects survive (3 are lost together with the 15 skipped sessions).

ii.
```python
jobs.append((eid, g['subject'].iloc[0], probes))
```

```python
subjects = sorted({r['subject'] for r in ok})
sub2idx = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([sub2idx[r['subject']] for r in ok], dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 5 variable-mapping table: "`bwm_release.csv` `subject` → `subjects`, `subject_idx`, unique sorted subject names, `0_data_caching.py` reads the same freeze file". An edge-case check verifies the subject_idx ↔ subject correspondence and that `subjects` is sorted and unique.

## 1-c. How are the data split into sessions?

i. A session is the natural unit of the release; the freeze file is grouped by `eid`, one job per `eid`, and the jobs (and hence the output session ordering) are sorted by eid. Each session's result becomes one element of `neural`/`input`/`output`. Sessions that cannot supply usable whisker video or ≥2 usable trials are dropped, leaving 444 of 459.

ii.
```python
for eid, g in bwm.groupby('eid', sort=True):
    probes = list(zip(g['pid'], g['probe_name']))
    jobs.append((eid, g['subject'].iloc[0], probes))
jobs.sort(key=lambda j: j[0])
```

```python
results.sort(key=lambda r: r['eid'])
ok = [r for r in results if 'skip' not in r]
```

iii. No justification was needed beyond the data organisation — Step 4 notes "699 pids / 459 eids in `bwm_release.csv`; 459 sessions on disk", and multiple probes of one session are explicitly pooled into a single session population (`merge_probes`) rather than kept separate.

## 1-d. How are the data split into trials?

i. The trials table returned by `SessionLoader.load_trials()` has one row per trial, so the split is given by the data. Each retained row becomes one trial entry, and the row index into the *full* trials table is preserved in metadata (`session_info[i]['trial_idx']`) so any converted trial can be traced back to the source.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials, mask = load_trials_and_mask(
    one=one, eid=eid, min_rt=MIN_RT, max_rt=MAX_RT, nan_exclude='default',
    max_trial_len=MAX_TRIAL_LEN, exclude_unbiased=False, exclude_nochoice=True,
    sess_loader=sl)
...
keep = np.flatnonzero(mask)
```

```python
'trial_idx': [int(x) for x in r['trial_idx']],
```

iii. Step 10 Check 5 records the corresponding edge-case checks: "`trial_idx` strictly increasing and inside the trials table; `n_trials == len(trial_idx) == len(neural[session])`".

## 1-e. How are trials filtered based on quality controls?

i. Three layers. (1) The reference implementation's own function `load_trials_and_mask` is imported and called with the reference's parameters (`max_trial_len=10.0`, defaults otherwise): no NaN in {`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`}; reaction time `firstMovement_times − stimOn_times` in [0.08, 2.0] s; `feedback_times − goCue_times ≤ 10 s`; `choice != 0` (no-go dropped); the unbiased block is *kept* (`exclude_unbiased=False`). (2) Behavioural coverage, replicating `get_behavior_per_interval`'s rejection rules: the wheel and the whisker trace must each have samples inside the 2 s window, start no more than one bin late, end no more than one bin early, and contain no NaN. (3) An extra check that the window lies inside the recorded spike train. Sessions left with <2 trials are dropped. Accounting over the 444 kept sessions: 286,532 trials in the tables → 189,011 pass the mask → 2 lost to wheel coverage, 84 to whisker coverage, 3 to the spike-recording check → 188,922 kept (425.5/session).

ii.
```python
MAX_TRIAL_LEN = 10.0            # reference: prepare_data(..., max_trial_len=10.0)
MIN_RT, MAX_RT = 0.08, 2.0      # data paper trial exclusions
...
trials, mask = load_trials_and_mask(
    one=one, eid=eid, min_rt=MIN_RT, max_rt=MAX_RT, nan_exclude='default',
    max_trial_len=MAX_TRIAL_LEN, exclude_unbiased=False, exclude_nochoice=True,
    sess_loader=sl)
```

```python
good = (nonempty
        & np.isfinite(t_begs) & np.isfinite(t_ends)
        & (np.abs(t_begs - first) <= BINSIZE)
        & (np.abs(t_ends - last) <= BINSIZE))
```

```python
if len(spike_times_k):
    in_rec = ((t_begs >= spike_times_k[0] - BINSIZE)
              & (t_begs + N_BINS * BINSIZE <= spike_times_k[-1] + BINSIZE))
valid = beh_good & in_rec
if valid.sum() < MIN_TRIALS_PER_SESSION:
    res['skip'] = f'only {int(valid.sum())} trials survive behaviour/spike checks'
```

iii. Key Decision 4: "Trial curation = `load_trials_and_mask` with the reference's `max_trial_len=10.0` … The unbiased block is **kept** (`exclude_unbiased=False`, the reference default), because `probabilityLeft == 0.5` is one of the three required prior classes." Key Decision 5: "Trials with unusable behaviour are dropped, using the reference's own criteria (trace missing, starts more than one bin late, ends more than one bin early, NaNs) — this is what `align_spike_behavior` does." Step 3 quotes the data paper's matching text ("trials were excluded if one of the following trial events could not be detected … outside the range of 0.08–2.00 s"). The extra spike-recording check is justified as "A trial whose window falls outside the recorded spike train would silently become an all-zero matrix; drop it instead."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` of every probe insertion of the session — and only those two arrays; `spikes.depths` and `spikes.amps` are deliberately not read. The cluster table (merged with channels by `SpikeSortingLoader.merge_clusters`) supplies `label` (QC) and `acronym` (anatomy), which are used only for filtering and for `brain_region_idx`, not for the activity itself.

ii.
```python
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
ssl.download_spike_sorting_object('spikes')
wanted = [f for f in ssl.files['spikes']
          if f.name.split('.')[1] in ('times', 'clusters')]
spikes = dict(ssl._load_object(wanted))
clusters = ssl.load_spike_sorting_object('clusters')
channels = ssl.load_channels()
clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
spikes, clusters = merge_probes(spikes_list, clusters_list)
beryl = br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
return spikes['times'], spikes['clusters'], clusters, np.asarray(beryl)
```

iii. Step 6 lists this as an efficiency decision: "`SpikeSortingLoader.load_spike_sorting()` reads `spikes.depths` and `spikes.amps` (hundreds of MB per probe) that the conversion never uses" → "Only `spikes.times` and `spikes.clusters` are read" (~4× less spike I/O).

## 2-b. How is the `neural` data processed?

i. Probes of one session are pooled into a single population with the reference's own `merge_probes` (which re-indexes cluster ids), the surviving clusters are renumbered 0…n−1, the merged spike train is re-sorted by time, and spikes are counted into 100 non-overlapping 20 ms bins per trial. The stored values are **raw spike counts per 20 ms bin** (float32), not converted to Hz and not z-scored or smoothed; the metadata records `'neural_units': 'spike counts per 20 ms bin'`. The binning is a fully vectorised single `np.bincount` over all trials at once, verified bit-identical against the reference's `bincount2D`.

ii.
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
```

```python
kept_ids = np.flatnonzero(good_unit)
remap = np.full(len(clusters), -1, dtype=np.int64)
remap[kept_ids] = np.arange(len(kept_ids))
sel = remap[spike_clusters] >= 0
spike_times_k = spike_times[sel]
spike_clusters_k = remap[spike_clusters[sel]]
order = np.argsort(spike_times_k, kind='stable')
```

```python
rel = spike_times[idx] - t_begs[trial_id]
b = np.floor(rel / BINSIZE).astype(np.int64)
ok = (b >= 0) & (b < N_BINS)          # the reference truncates to n_bins columns
...
flat = (trial_id * n_clusters + spike_clusters[idx]) * N_BINS + b
binned.reshape(-1)[:] = np.bincount(
    flat, minlength=n_trials * n_clusters * N_BINS).astype(np.float32)
```

iii. Docstring of `bin_spikes`: "Equivalent to the reference's `bin_spiking_data`, which calls `bincount2D(times, clusters, xbin=binsize, xlim=[t_beg, t_end])` per trial and then truncates to `n_bins` columns … Implemented here with a single `np.bincount` over all trials instead of a per-trial multiprocessing pool, which is ~100x faster and produces bit-identical counts (verified in Step 10)." Key Decision 9: "Neural data are stored as raw spike counts (float32), not z-scored: the reference z-scores inside its data loader, and `train_decoder` does its own SVD-based projection." Probe pooling is justified by the data paper ("neurons in the same session and region were combined across probes") and by the methods paper's "bin spike counts using all neurons … from each session".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, applied together per session after the probes are merged: `clusters.label >= 1` (the IBL RIGOR single-unit criteria — amplitude > 50 µV, noise cut-off < 20 µV, refractory-period violation — all passed, i.e. the data paper's "well-isolated" neurons), **and** the Beryl-mapped acronym not in {`root`, `void`} (the data paper's grey-matter restriction). A session with no surviving unit is skipped. Over the kept sessions: 599,865 clusters → 73,044 well-isolated → 62,763 well-isolated in grey matter (141.4/session). The AI separately verified through ONE that `label >= 1` over the whole release reproduces the paper's 621,733 clusters / 75,708 good units / 889.5 and 108.3 per probe exactly.

ii.
```python
GOOD_UNIT_LABEL = 1.0           # RIGOR: all three single-unit metrics passed
NON_GREY = ('root', 'void')     # excluded from "grey matter" analyses
...
label = clusters['label'].to_numpy()
good_unit = (label >= GOOD_UNIT_LABEL) & ~np.isin(beryl, NON_GREY)
res['n_units_good'] = int((label >= GOOD_UNIT_LABEL).sum())
res['n_units_kept'] = int(good_unit.sum())
if res['n_units_kept'] == 0:
    res['skip'] = 'no well-isolated grey-matter units'
    return res
```

iii. Key Decision 2 / Step 10 Check 3 flag this as the one deliberate departure from the reference caching script (which passes `qc=None` and keeps all clusters): "(i) the data paper that produced this dataset states that its analyses use only the 75,708 well-isolated neurons and are 'restricted to regions that were designated grey matter'; (ii) the reference code itself records `good_clusters = label >= 1` in its metadata and its own multi-region loader excludes `root`/`void` [`data_loader_utils.py:252`]; (iii) keeping all clusters would make the converted tensor ≈8× larger (≈95 GB) by adding multi-unit clusters, which carry no additional single-neuron signal for a PCA + linear decoder." The paper's further "≥5 neurons per session and ≥2 sessions per region" criterion is explicitly *not* applied, with the reasoning that it is a region-wise decoding criterion while here all neurons of a session are pooled.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams are already on one synchronised session clock, so alignment is a subtraction: each trial's window begins at `stimOn_times + (−0.5)` and runs 100 × 20 ms, and every spike's bin index is `floor((t − t_beg)/binsize)`. `metadata['temporal_alignment_event'] = 'visual stimulus onset (trials.stimOn_times)'`, `off_start = −0.5`, `off_end = 1.5`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
t_begs = trials[ALIGN_TIME].to_numpy()[keep] + TIME_WINDOW[0]
```

```python
rel = spike_times[idx] - t_begs[trial_id]
b = np.floor(rel / BINSIZE).astype(np.int64)
```

iii. Step 4: "Alignment event — `stimOn_times` for the cached dataset; papers say stimulus onset for choice/prior, first movement for wheel/whisker; **Task specifies stimulus onset → `stimOn_times` for everything**." Verified empirically in Step 7/12: wheel speed is identically zero throughout the pre-stimulus window (the enforced quiescence period) and departs from zero within ~0.1 s of t = 0 on every trial, and per-bin choice decodability is at chance before t = 0 and peaks at +0.22 s — both only possible if t = 0 really is stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, `N_BINS = ceil(2.0/0.02) = 100`, identical for every trial and session; `metadata['time_bin_size'] = 20.0` ms. Spikes go straight from spike times into the final bins — there is no intermediate binning and hence no rebinning or resampling of the neural data. The AI explicitly considered and rejected the 50 ms figure mentioned in the methods paper text.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

```python
'time_bin_size': BINSIZE * 1000.0,          # ms
'n_time_bins': N_BINS,
```

iii. Step 3: "20 ms non-overlapping bins, T = 100. (The methods-paper text mentions 50 ms bins for the choice/prior analyses, but the released caching code — `params['binsize'] = 0.02` — and the model description ('2-s trials … 20-ms bins … T = 100') both use 20 ms. 20 ms is used here; it is also the only choice that resolves the 60 Hz whisker signal.)"

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from the data at all: it is the analytic bin-centre grid of the window around `stimOn_times`, `−0.5 + (i + 0.5)·0.02` for i = 0…99, in seconds, identical for every trial and session, range [−0.49, 1.49].

ii.
```python
bin_centers = (TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE).astype(np.float32)
inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
inputs[:, 0, :] = bin_centers[None, :]
```

iii. Step 5 mapping table: "bin centre time → `input[s][k][0, :]`, `−0.5 + (i + 0.5)·0.02` s — 'time since stimulus onset', time-varying, in seconds". A sanity check verifies "Input 0 equals the analytically computed bin-centre times" and an edge-case check that it "spans exactly [−0.49, 1.49]".

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None — it is constructed, not processed. The only choice is representing the bin by its centre rather than its edge, and broadcasting the same 100-value vector to every trial.

ii.
```python
bin_centers = (TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE).astype(np.float32)
inputs[:, 0, :] = bin_centers[None, :]
```

iii. Documented in metadata as "time of the bin centre relative to stimulus onset, in seconds (-0.49 ... 1.49)". No further justification given (none needed).

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It labels the same bins the spikes are counted into: the neural bin *i* covers `[t_beg + i·0.02, t_beg + (i+1)·0.02)` with `t_beg = stimOn − 0.5`, and input 0 at index *i* is the centre of that interval. So the two share a time axis bin for bin by construction. (Note that the two continuous *outputs* are sampled at the bin's right edge rather than its centre — see 7-d/8-d — so they sit 10 ms later than this label.)

ii.
```python
t_begs = trials[ALIGN_TIME].to_numpy()[keep] + TIME_WINDOW[0]
...
b = np.floor(rel / BINSIZE).astype(np.int64)          # neural bin index
...
bin_centers = (TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE)
```

iii. Implicit; the `--show-processing` figures plot the binned spike counts on the `[-0.5, 1.5]` extent with a red line at t = 0 to make the correspondence visible.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft` alone, which is constant within a block, so a block boundary is exactly a change of its value.

ii.
```python
pl = np.asarray(probability_left, dtype=float)
change = np.ones(len(pl), dtype=bool)
change[1:] = pl[1:] != pl[:-1]
block_id = np.cumsum(change) - 1
starts = np.flatnonzero(change)
```

iii. Docstring: "The session starts with a 90-trial unbiased block (p = 0.5); afterwards the block prior alternates between 0.2 and 0.8 with lengths drawn from a truncated geometric distribution. Block boundaries are exactly the points where `probabilityLeft` changes." The AI verified against the data that the first block is exactly 90 trials in all 459 sessions and that the mean biased-block length is 48.99 against the paper's 51 (its own count truncates each session's final block).

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The 0-based position of the trial within its block, `arange(n) − start_of_its_block`, computed vectorised on the **full, unfiltered** trials table so that excluded trials still advance the counter; then indexed by the kept-trial indices and broadcast over the 100 time bins as a constant. Observed range [0, 98].

ii.
```python
def trial_number_in_block(probability_left):
    ...
    return (np.arange(len(pl)) - starts[block_id]).astype(np.float32)
```

```python
# trial-in-block is computed on the full table so exclusions do not renumber it
tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
...
inputs[:, 1, :] = tib_all[keep][:, None]
```

iii. Docstring: "Computed on the *full* trials table before any trial exclusion, so that excluded trials still advance the counter." Key Decision 7 explains the broadcasting: "Per-trial variables are broadcast across time rather than stored as 1-D arrays, so that `input` and `output` have a single consistent shape and the brief's 'if at all possible, make it time-varying' is satisfied." Checks: "Input 1 equals trial-in-block recomputed from `probabilityLeft` on the raw trials table"; "trial-in-block never exceeds 89 inside the 90-trial unbiased block (off-by-one check)".

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (left), −1 (right) or 0 (no response). It is recoded +1 → 0 (left), −1 → 1 (right); 0-choice trials never reach this point because `exclude_nochoice=True` removed them, and an assertion fails loudly if any value is unmapped.

ii.
```python
def map_choice(choice):
    """IBL `choice` (+1 = left, -1 = right) -> task coding (left = 0, right = 1)."""
    out = np.full(len(choice), -1, dtype=np.int64)
    out[np.asarray(choice) == 1] = 0
    out[np.asarray(choice) == -1] = 1
    return out
```

```python
choice = map_choice(trials['choice'].to_numpy()[keep])
assert choice.min() >= 0, 'unmapped choice value'
```

iii. Step 4 discrepancy table: "`choice` sign — Verified: on correct trials with a left stimulus `choice == +1`; with a right stimulus `choice == −1` → `+1 → 0 (left)`, `−1 → 1 (right)`." The sanity check re-derives it independently: "every correct trial with a left stimulus is coded 0 and every correct trial with a right stimulus is coded 1". Observed distribution [0.508, 0.492].

## 5-b. What processing is involved in computing `output` *Choice*?

i. None beyond the recoding; the per-trial value is broadcast across all 100 bins so that the output array is time-varying with a single consistent shape.

ii.
```python
outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int64)
outputs[:, 0, :] = choice[:, None]
```

iii. Key Decision 7 (see 4-b). Step 12 notes the consequence: choice is constant within a trial and is only decodable after movement onset, which the AI verified with a per-bin decoding time course (chance before t = 0, peak 0.748 at +0.22 s).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes exactly the three values 0.2, 0.5 and 0.8, recoded 0.2 → 0, 0.5 → 1, 0.8 → 2 (the mapping the decoder task prescribes), with `np.isclose` for float safety and an assertion against unmapped values.

ii.
```python
def map_prior(probability_left):
    """`probabilityLeft` -> 0.2 -> 0, 0.5 -> 1, 0.8 -> 2 (per the task spec)."""
    pl = np.asarray(probability_left, dtype=float)
    out = np.full(len(pl), -1, dtype=np.int64)
    out[np.isclose(pl, 0.2)] = 0
    out[np.isclose(pl, 0.5)] = 1
    out[np.isclose(pl, 0.8)] = 2
    return out
```

```python
prior = map_prior(trials['probabilityLeft'].to_numpy()[keep])
assert prior.min() >= 0, 'unmapped probabilityLeft value'
```

iii. Step 4: "`probabilityLeft` — values ∈ {0.2, 0.5, 0.8}; papers 0.2/0.5/0.8 → `0.2 → 0`, `0.5 → 1`, `0.8 → 2` (task)." The unbiased (0.5) block is deliberately retained because it is one of the three required classes (Key Decision 4). Resulting distribution [0.417, 0.141, 0.442], consistent with a 90-trial unbiased block per session.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the recoding; broadcast across the 100 bins like choice.

ii.
```python
outputs[:, 1, :] = prior[:, None]
```

iii. As 5-b / Key Decision 7. The sanity check "Output 1 equals `probabilityLeft` remapped (0.2/0.5/0.8 → 0/1/2)" passed on three whole sessions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()`, i.e. `_ibl_wheel.position` / `_ibl_wheel.timestamps` interpolated by the loader to a 1000 Hz grid and differentiated into a low-pass-filtered velocity; wheel speed is `np.abs(velocity)`, in rad/s. This is the same source as the reference's `load_target_behavior(one, eid, 'wheel-speed')`.

ii.
```python
def load_wheel_speed(one, eid):
    """Wheel speed = |velocity| of the 1000 Hz interpolated wheel trace.

    Same source as the reference's `load_target_behavior(one, eid, 'wheel-speed')`.
    """
    sl = SessionLoader(one=one, eid=eid)
    sl.load_wheel()
    return (sl.wheel['times'].to_numpy(),
            np.abs(sl.wheel['velocity'].to_numpy()))
```

iii. Step 3: "**Wheel speed** = |velocity| of the 1000 Hz interpolated wheel trace." Step 5 maps `wheel.velocity → output[s][k][2, :]` via `load_target_behavior('wheel-speed')`, `get_behavior_per_interval`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps. (1) `SessionLoader` produces the filtered velocity (interpolation to 1000 Hz + Butterworth low pass, all inside the loader), absolute value taken. (2) The trace is linearly interpolated onto the per-trial 100-point grid `t_beg + binsize … t_end` with a single vectorised `np.interp` over all trials at once; trials the trace does not cover are marked bad and dropped. (3) The (n_trials × 100) matrix of the session is cut into three classes at its 33.3rd and 66.7th percentiles. No smoothing or normalisation is added.

ii.
```python
grid = np.linspace(BINSIZE, N_BINS * BINSIZE, N_BINS)       # t_beg + binsize ... t_end
query = t_begs[good][:, None] + grid[None, :]
interp = np.interp(query.ravel(), target_times, target_vals).reshape(-1, N_BINS)
values[good] = interp
```

```python
wheel_cls, wheel_edges = discretize_tertiles(wheel_vals)
```

iii. `bin_behavior` docstring: "Reproduces the reference's `get_behavior_per_interval`: the trace is evaluated at `np.linspace(t_beg + binsize, t_end, n_bins)`, i.e. at the *right edge* of each spike-count bin, and a trial is rejected when the trace does not cover the window." Step 6 records the speed-up: "one `np.interp` over the whole (n_trials × 100) query grid" instead of the reference's per-trial `interp1d` in a pool (~50×). A sanity check re-interpolated and re-discretised the raw wheel trace independently: "0 of 43,400 / 47,200 / 37,100 bins differ".

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 classes by **per-session tertiles**: the 33.3rd and 66.7th percentiles of all binned wheel values of that session's retained trials, applied with `np.digitize(..., right=False)`. Degenerate cases (more than a third of the signal on one value, e.g. a wheel that barely moves) fall back to percentiles of the *distinct* values, then to the two/one distinct values, and finally to `np.nextafter` so the edges are never equal. Resulting global distribution [0.3333, 0.3333, 0.3333].

ii.
```python
def discretize_tertiles(values):
    flat = values.ravel()
    edges = np.percentile(flat, [100.0 / 3.0, 200.0 / 3.0])
    if edges[0] == edges[1]:
        uniq = np.unique(flat)
        if len(uniq) >= 3:
            edges = np.percentile(uniq, [100.0 / 3.0, 200.0 / 3.0])
        elif len(uniq) == 2:
            edges = np.array([uniq[0], uniq[1]])
        else:
            edges = np.array([uniq[0], uniq[0] + 1.0])
        if edges[0] == edges[1]:
            edges[1] = np.nextafter(edges[1], np.inf)
    return np.digitize(values, edges, right=False).astype(np.int64), edges
```

iii. Key Decision 6: "Per-session because whisker motion energy is in arbitrary camera/ROI-dependent units that are not comparable between sessions, and wheel-speed scale varies with the mouse; tertiles because they give balanced classes, so the decoder's balanced accuracy has a clean 1/3 chance level. Degenerate edges (ties) are handled explicitly." The actual edges used are stored per session in `metadata['session_info'][i]['wheel_tertile_edges']`, and `--show-processing` plots a value→class scatter with the edges overlaid.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is on the same session clock as the spikes, so the alignment is just evaluating it on the same per-trial grid anchored at `stimOn − 0.5`. The AI samples it at the **right edge** of each 20 ms neural bin (`t_beg + 0.02 … t_beg + 2.0`), copying the reference's `get_behavior_per_interval`, so output bin *i* is the wheel speed at the end of neural bin *i* — 10 ms after the bin-centre time reported by input 0.

ii.
```python
t_begs = trials[ALIGN_TIME].to_numpy()[keep] + TIME_WINDOW[0]
...
grid = np.linspace(BINSIZE, N_BINS * BINSIZE, N_BINS)       # t_beg + binsize ... t_end
query = t_begs[good][:, None] + grid[None, :]
```

iii. Step 3 Processing Details: "**Behaviour binning**: linear interpolation of the continuous trace onto the bin grid (`np.linspace(t_beg + binsize, t_end, 100)`), i.e. the value at the *right edge* of each spike-count bin." Step 10 Check 3 lists binning as "✅ identical semantics" to the reference. The alignment was checked visually (`processing_<eid>_alignment.png`): wheel speed is exactly zero across the whole −0.5→0 s quiescence window and rises within ~0.1 s of t = 0 on every trial.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<view>Camera.ROIMotionEnergy` with frame times `_ibl_<view>Camera.times`, loaded through `SessionLoader.load_motion_energy(views=[view])` and read from the `whiskerMotionEnergy` column, left camera preferred and right camera as fallback — the same preference order as the reference's `bin_behaviors`. 437 sessions used the left camera, 7 the right; 14 sessions have neither and are skipped.

ii.
```python
def load_whisker_me(one, eid):
    for view in ('left', 'right'):
        try:
            sl = SessionLoader(one=one, eid=eid)
            sl.load_motion_energy(views=[view])
            me = sl.motion_energy[f'{view}Camera']
            return (me['times'].to_numpy(),
                    me['whiskerMotionEnergy'].to_numpy(), view)
        except Exception:
            continue
    return None, None, None
```

iii. Step 3: "**Whisker motion energy** = mean absolute frame-to-frame difference in a bounding box anchored between nose tip and eye; left camera preferred, right camera as fallback." The camera actually used is recorded per session in `metadata['session_info'][i]['whisker_camera']`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as it is (no filtering, no normalisation), passed through exactly the same `bin_behavior` path as the wheel: coverage test, one vectorised `np.interp` onto the per-trial right-edge grid, rejection of trials with NaNs in the window, then per-session tertiles.

ii.
```python
mt, mv, view = load_whisker_me(one, eid)
if mt is None:
    res['skip'] = 'no whisker motion energy (neither camera)'
    return res
res['whisker_camera'] = view
me_vals, me_good = bin_behavior(mt, mv, t_begs)
...
me_cls, me_edges = discretize_tertiles(me_vals)
outputs[:, 3, :] = me_cls
```

iii. Same as 7-b; metadata: "whisker-pad motion energy (left camera, right camera as fallback) interpolated to the bin grid then discretised into per-session tertiles". Independently re-derived in the Step 10 sanity checks with zero mismatching bins. The 60 Hz camera rate is also the AI's stated reason for keeping 20 ms bins rather than 50 ms.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to wheel speed: 3 classes at the 33.3rd/66.7th percentiles of all binned whisker values of that session, `np.digitize`, same degenerate-edge fallback, edges stored in `metadata['session_info'][i]['whisker_tertile_edges']`. Resulting distribution [0.3345, 0.3327, 0.3328].

ii.
```python
me_cls, me_edges = discretize_tertiles(me_vals)
```

iii. Key Decision 6 — the per-session choice is justified most strongly for this variable: "whisker motion energy is in arbitrary camera/ROI-dependent units that are not comparable between sessions" (and the left/right camera fallback makes cross-session pooling of raw values even less meaningful).

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as the wheel: camera frame times are on the shared session clock, and the trace is evaluated on the per-trial right-edge grid anchored at `stimOn − 0.5`, so output bin *i* is the motion energy at the end of neural bin *i*. Trials whose camera trace starts more than one bin late or ends more than one bin early are dropped (84 trials).

ii.
```python
me_vals, me_good = bin_behavior(mt, mv, t_begs)
...
beh_good = wheel_good & me_good
```

iii. As 7-d. Step 7: "Whisker motion energy likewise steps up just after t = 0"; the per-trial × time image in `processing_<eid>_alignment.png` is the visual check.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable data are dropped, never imputed, and every drop is counted and reported. Specifically: every session is processed inside a try/except so a single failure cannot abort the run, and the reason (plus traceback) is printed and stored in `metadata['skipped_sessions']`; sessions with no whisker video on either camera are skipped (14); a session whose camera trace does not overlap its trials ends with 0 usable trials and is skipped (1); trials whose wheel or camera trace does not span the window, or contains NaNs, are dropped (2 and 84); trials whose window falls outside the recorded spike train are dropped (3); sessions with no well-isolated grey-matter unit, or fewer than 2 usable trials, are skipped; degenerate tertile edges are handled explicitly; the stale/incorrect ONE cache tables were rebuilt rather than worked around. The 35 remaining all-zero-neural trials flagged by the verifier were investigated one by one and deliberately **kept**, on the grounds that removing them would be a behaviour-correlated exclusion.

ii.
```python
except Exception as e:          # keep the run going, report at the end
    import traceback
    res['skip'] = f'{type(e).__name__}: {e}'
    res['traceback'] = traceback.format_exc()
```

```python
# An interpolated NaN means the trace itself has a gap of NaNs across the window.
nan_rows = np.isnan(values[good]).any(axis=1)
if nan_rows.any():
    gi = np.flatnonzero(good)
    good[gi[nan_rows]] = False
```

```python
if len(keep) < MIN_TRIALS_PER_SESSION:
    res['skip'] = f'only {len(keep)} trials pass the trial mask'
...
'skipped_sessions': [{'eid': r['eid'], 'reason': r['skip']} for r in skipped],
```

iii. Step 10 Check 1: the zero-neural warnings "affect 35 of 188,922 trials (0.019 %). Removing zero-spike trials would be a *behaviour-correlated* exclusion (trials are removed precisely when the neurons were silent), which would bias the neural distribution conditioned on the decoded variables; and dropping low-yield sessions would discard real data that the per-session projection handles perfectly well." Each was traced to a cause: a 1-neuron session, a 10-low-rate-neuron session, and a genuine 2.07 s recording dropout. Step 10 Check 5 lists the defect classes exercised by the real data and confirms none crashes.

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting off disk dominates: 1.8 s of the ~3.5 s per session in a worker, because a probe's `spikes.times`/`spikes.clusters` run to hundreds of MB. Behaviour loading is 0.6 s, spike binning 0.4 s, trials+mask 0.2 s. At the whole-run level the second cost is writing the 11.73 GB pickle (~16 s). Total wall time for 459 sessions on 32 workers: 70 s. The script instruments each stage (`timing` dict) and prints per-session timing and an ETA.

ii.
```python
t0 = time.time()
spike_times, spike_clusters, clusters, beryl = load_session_spikes(
    one, br, eid, probes)
...
timing['spikes_load'] = time.time() - t0
```

```python
timing['trials'] = ...; timing['behavior'] = ...; timing['bin_spikes'] = ...
```

iii. Step 7 run-time table gives the per-stage breakdown and the prediction (459 × 3.5 / 32 ≈ 50 s + 17 s write ≈ 70 s), which matched the measured 70 s. The AI notes "Well under the 15-minute budget, so no further optimisation was needed", and cut spike I/O ~4× by reading only `times` and `clusters`.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI had already vectorised the two loops the reference code runs per trial — `bincount2D` per trial in a pool became a single gather + one `np.bincount` over all trials (~100×), and `interp1d` per trial became one `np.interp` over the whole (n_trials × 100) query grid (~50×) — and moved parallelism up to the session level (32 workers). What remains unvectorised is cheap: the per-probe loop in `load_session_spikes` (1–2 iterations, I/O bound), and the list comprehensions in `main` that split each session's `(n_trials, …)` arrays into per-trial lists, which are views rather than copies. Nothing per-trial remains in the hot path.

ii.
```python
# vectorised binning over all trials at once
offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
trial_id = np.repeat(np.arange(n_trials), counts)
idx = np.arange(total) - np.repeat(offsets, counts) + np.repeat(i0, counts)
```

```python
# remaining loops, both negligible
for pid, pname in probes:
    ...
'neural': [[r['neural'][k] for k in range(r['neural'].shape[0])] for r in ok],
```

iii. Step 6: "The reference bins spikes with one `bincount2D` call **per trial** inside a `multiprocessing.Pool`, and interpolates behaviour with one `interp1d` call per trial in another pool. For 459 sessions that is ~200,000 pool tasks" → replaced by the two vectorised implementations, "verified bit-identical to `bincount2D`, Step 10 Check 2", and "Parallelism moved up one level: 32 worker processes, one session each, instead of per-trial pools."

## 10-c. What processing does the code repeat multiple times?

i. Little, and nothing expensive, but three repetitions exist. (1) Three separate `SessionLoader` objects are constructed per session — one for trials, one inside `load_wheel_speed`, one (per attempted view) inside `load_whisker_me` — instead of reusing the first; each re-resolves the session through ONE. (2) When the left camera is absent, `load_whisker_me` builds and fails a whole loader before retrying the right camera. (3) In `--show-processing` mode, `plot_processing` reloads the wheel and motion-energy traces that the conversion already loaded. Elsewhere repetition is explicitly avoided: ONE and `BrainRegions` are created once per worker process and cached in module globals, and the tertile edges are computed once per session rather than per trial.

ii.
```python
sl = SessionLoader(one=one, eid=eid)      # trials
...
def load_wheel_speed(one, eid):
    sl = SessionLoader(one=one, eid=eid)  # again
    sl.load_wheel()

def load_whisker_me(one, eid):
    for view in ('left', 'right'):
        try:
            sl = SessionLoader(one=one, eid=eid)   # and again, per view
```

```python
def get_one():
    """One ONE instance and one BrainRegions per process (both are expensive)."""
```

iii. The AI documents the caching it did do ("`get_one()` — one `ONE` (mode `local`) and one `BrainRegions` per worker process"), and the repeated `SessionLoader` construction is not called out anywhere in CONVERSION_NOTES.md — it is implicitly accepted as negligible next to the 1.8 s spike read.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A few small things. Every worker returns `wheel_raw`, `me_raw` and `t_begs` (the full un-discretised behaviour matrices and window starts) for **every** session even when `--show-processing` is not given; they are pickled across the process boundary and then discarded — only `wheel_edges`/`me_edges`/`trial_idx` reach the metadata. `load_session_spikes` attaches a `clusters['pid']` column that is never read. `sl.load_trials()` is called explicitly before `load_trials_and_mask`, which would have loaded the trials itself. The `output` arrays are stored as `int64` although they only hold values 0–2 (≈0.5 GB of the 11.73 GB file could be saved with `int8`). Conversely, the AI actively removed the reference pipeline's genuinely wasted work: it does not read `spikes.depths`/`spikes.amps`, and it loads only the two behaviours needed rather than the six `load_anytime_behaviors` returns.

ii.
```python
res.update({
    ...
    'wheel_raw': wheel_vals.astype(np.float32),   # only used by plot_processing
    'me_raw': me_vals.astype(np.float32),
    't_begs': t_begs,
```

```python
clusters['pid'] = pid      # never read again
```

```python
outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int64)   # values are 0..2
```

iii. Step 6 documents only the waste the AI removed: "`SpikeSortingLoader.load_spike_sorting()` reads `spikes.depths` and `spikes.amps` (hundreds of MB per probe) that the conversion never uses"; "`prepare_data` calls `load_anytime_behaviors`, which loads six behavioural traces (including pupil diameter and both cameras) when only two are needed." The retained raw behaviour arrays are justified implicitly by the `--show-processing` requirement, and the total run time (70 s) left no pressure to trim them.
