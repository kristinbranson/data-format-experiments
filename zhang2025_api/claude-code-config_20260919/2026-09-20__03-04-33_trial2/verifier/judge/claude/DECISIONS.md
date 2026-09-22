# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All access goes through the ONE API and the `brainbox` loaders; no file under `/app/data` is opened directly. The master list of sessions is **not** obtained from `one.search`, but from the reference code's own release freeze file `/app/code/code_zhang2025/data/bwm_release.csv` (459 eids / 699 pids / 139 subjects), exactly as `0_data_caching.py` does. From each `eid` everything else is resolved by the API: `one.eid2pid(eid)` gives the probe insertions, `SpikeSortingLoader(pid=...).load_spike_sorting()` + `merge_clusters(...).to_df()` gives spikes and clusters per probe, the reference `merge_probes()` pools the probes of a session, `SessionLoader` + the reference `load_trials_and_mask()` gives the trials table and its exclusion mask, and `SessionLoader.load_wheel()` / `load_motion_energy()` give the wheel and camera streams. The ONE client is built in the default (remote) mode against the staged cache — the agent found that `mode='local'` plus `load_cache(tag='Brainwidemap')` silently returns an empty trials table because the shipped release parquet predates the dataset revisions that are actually on disk. Sessions are fanned out over a `ProcessPoolExecutor` (32 workers), one ONE client per process.

ii.
```python
_ONE = None
def get_one():
    global _ONE
    if _ONE is None:
        _ONE = ONE(base_url=ONE_BASE_URL, silent=True, cache_dir=ONE_CACHE_DIR)
    return _ONE
```
```python
bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eids = list(pd.unique(bwm_df.eid))
```
```python
pids, pnames = one.eid2pid(eid)
...
for pid, pname in zip(pids, pnames):
    ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
    sp, cl, ch = ssl.load_spike_sorting()
    ...
    cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
return merge_probes(spikes_list, clusters_list)
```
```python
sess_loader = SessionLoader(one=one, eid=eid)
trials, mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
```
```python
with ProcessPoolExecutor(max_workers=n_workers) as pool:
    futures = {pool.submit(_worker, eid): eid for eid in remaining}
```

iii. CONVERSION_NOTES Step 4: "**Use `bwm_release.csv`**, exactly as the reference does" — the freeze file is the reference code's own session list and matches the data paper's 459 sessions / 699 insertions / 139 mice. On the client mode: "`mode='local'` + `load_cache(tag='Brainwidemap')` cannot see the staged dataset revisions → Default (remote) mode against the staged `.rest` response cache and auth token; verified to return all 20 trial columns." The reference `load_trials_and_mask` and `merge_probes` are *imported* rather than re-implemented "so the logic cannot drift".

## 1-b. How are the data split into subjects?

i. The subject of a session is read from the `subject` column of `bwm_release.csv` (one row per pid, de-duplicated on eid). After conversion the unique subject names are sorted and `subject_idx` records each session's index into that list. 136 of the 139 released subjects survive (3 lose their only session).

ii.
```python
subject_of_eid = bwm_df.drop_duplicates('eid').set_index('eid')['subject'].to_dict()
```
```python
subjects = sorted({subject_of_eid[res['eid']] for res in results})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_index[subject_of_eid[r['eid']]] for r in results],
                        dtype=np.int64),
```

iii. The release table already carries a unique subject id per session, so nothing has to be derived or parsed from paths. CONVERSION_NOTES Step 5 maps "`bwm_release.csv` `subject` → `subjects`, `subject_idx`, unique sorted subject names", citing `0_data_caching.py` as the reference. Sanity check in Step 9: "Subjects | 139 | … | 136 | ✓ (3 lost with their only session)".

## 1-c. How are the data split into sessions?

i. No splitting is needed — a session *is* an `eid`, and the release file lists one row per (eid, pid). The conversion iterates `pd.unique(bwm_df.eid)` and processes one eid per worker task. Session order in the output follows the release-file order, which the agent enforces explicitly so the parallel run is deterministic.

ii.
```python
eids = list(pd.unique(bwm_df.eid))
...
# deterministic session order (release-file order)
order = {e: i for i, e in enumerate(eids)}
results.sort(key=lambda r: order[r['eid']])
```

iii. CONVERSION_NOTES Step 5, Key Decision 10: "**Session order** follows `bwm_release.csv`, so the conversion is deterministic." The unit of the release is the session, so there is no decision to make about how to split.

## 1-d. How are the data split into trials?

i. The trials table returned by `SessionLoader.load_trials()` (via the reference `load_trials_and_mask`) has one row per trial, so the split is given by the data. Each trial becomes a 2 s window `stimOn_times + (-0.5, +1.5)` s.

ii.
```python
trials, mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
mask = np.asarray(mask, dtype=bool)
n_trials_all = len(trials)
align_times = trials[ALIGN_TIME].to_numpy()
```
```python
beg = align_times + TIME_WINDOW[0]
end = align_times + TIME_WINDOW[1]
```

iii. The trials table is already one row per trial; the agent's only decision is the window, taken verbatim from the reference `params` (`interval_len=2`, `time_window=(-.5, 1.5)`, `align_time='stimOn_times'`), quoted in the script docstring and in CONVERSION_NOTES Step 3/4.

## 1-e. How are trials filtered based on quality controls?

i. Four filters are ANDed together.
1. **Reference trial mask** — the reference function `load_trials_and_mask(max_trial_len=10.0)` is imported and used verbatim: drop trials whose reaction time `firstMovement_times − stimOn_times` is outside [0.08, 2.0] s, trials with NaN in any of `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType`, trials with `choice == 0` (no response), and trials with `feedback_times − goCue_times > 10 s`.
2. **Wheel coverage** — the reference `get_behavior_per_interval` rejection rules: no wheel samples in the window, wheel trace starting more than one bin late or ending more than one bin early, or NaN in the slice.
3. **Whisker-ME coverage** — the same three rules applied to the camera trace.
4. **Ephys coverage (the agent's own addition)** — a trial is dropped if its 2 s window falls before the first / after the last spike of the session, or intersects an inter-spike gap longer than 0.5 s in the *pooled, unfiltered* spike train.

Session-level QC: ≥ 5 well-isolated grey-matter neurons, a usable whisker trace, ≥ 2 usable trials; sessions that fail raise and are reported. Result: 441/459 sessions, 187,901 trials (66.0 % of raw trials pass the reference mask; 86 further trials lost to behaviour coverage, 33 to ephys coverage).

ii.
```python
MAX_TRIAL_LEN = 10.0               # `load_trials_and_mask(max_trial_len=10.0)` in the ref
MAX_SPIKE_GAP = 0.5                # s; a longer gap in the pooled spike train = no data
MIN_TRIALS = 2
MIN_NEURONS = 5
```
```python
has_neural = neural_data_available(all_spike_times, align_times)
keep_trials = mask & wheel_ok & me_ok & has_neural
n_keep = int(keep_trials.sum())
if n_keep < MIN_TRIALS:
    raise RuntimeError(f'only {n_keep} usable trials')
```
```python
dt = np.diff(all_spike_times)
idx = np.nonzero(dt > MAX_SPIKE_GAP)[0]
gap_start = np.concatenate([[-np.inf], all_spike_times[idx], [all_spike_times[-1]]])
gap_end = np.concatenate([[all_spike_times[0]], all_spike_times[idx + 1], [np.inf]])
overlaps = (safe_beg[:, None] < gap_end[None, :]) & (safe_end[:, None] > gap_start[None, :])
return ok & ~overlaps.any(axis=1)
```
```python
if abs(beg[k] - tt[0]) > BINSIZE:      # target data starts too late
    continue
if abs(end[k] - tt[-1]) > BINSIZE:     # target data ends too early
    continue
```

iii. CONVERSION_NOTES Step 4: "**Use `load_trials_and_mask` verbatim** (imported from the reference module), including `max_trial_len=10.0`, which is the reference code's only addition"; the data paper "lists exactly the 6 NaN events and the 0.08–2.00 s reaction-time window". Step 5, Key Decision 6: the ephys-coverage test is "Necessary because some sessions keep running the behavioural rig after the ephys stops; those trials would otherwise be all-zero" — it was introduced in Step 10 after 38 `all neural data is zero` warnings were root-caused to a 5 s spike-train dropout (`b182b754…`) and an ephys stream ending 12 s before the behaviour (`8c2f7f4d…`); adding it plus the ≥ 5-neuron rule cut the warnings from 38 to 12, and the remaining 12 were verified to be genuine sparse firing. On NaNs: "Use the reference's `allow_nans=False` branch (reject the trial). The converted outputs are class labels, so there is no sensible 'NaN class', and `verify_data_format` errors on NaNs."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` of every probe of the session, pooled by the reference `merge_probes`. The cluster table (`SpikeSortingLoader.merge_clusters(...).to_df()`) supplies only the `label` (quality) and `acronym` (anatomy) columns used for selection and for `brain_region_idx`; the count matrix itself is built from the two spike arrays alone. The unfiltered pooled `spikes['times']` is additionally reused as the ephys-coverage signal.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
...
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
...
return merge_probes(spikes_list, clusters_list)
```
```python
all_spike_times = spikes['times']       # pooled, unfiltered: used for the data-gap test
remap = np.full(n_clusters_all, -1, dtype=np.int64)
remap[keep_idx] = np.arange(n_neurons)
sc = remap[spikes['clusters']]
sel = sc >= 0
spike_times = np.ascontiguousarray(spikes['times'][sel])
spike_clusters = np.ascontiguousarray(sc[sel])
```

iii. This is exactly the reference `prepare_data`'s `neural_dict = {'spike_times': spikes['times'], 'spike_clusters': spikes['clusters'], 'cluster_regions': clusters['acronym']}` (CONVERSION_NOTES Step 10, Check 3: "(a) spike loading … identical", "(a) probe merge … the same function, imported").

## 2-b. How is the `neural` data processed?

i. Probes of a session are pooled into one population by the reference `merge_probes` (which re-indexes `spikes['clusters']` and re-sorts the merged spike train by time), quality/anatomy filtering is applied, and the surviving clusters are renumbered contiguously. Spikes are then counted into 100 non-overlapping 20 ms bins per trial with a single `np.bincount` over a flattened (cluster × bin) index. **Raw spike counts are stored, not firing rates** — carried as `uint8` from the workers (clipped at 255) and expanded to `float32` in the parent. No smoothing, no z-scoring, no normalisation.

ii.
```python
def bin_spiking_data_fast(spike_times, spike_clusters, n_clusters, align_times):
    beg = align_times + TIME_WINDOW[0]
    end = align_times + TIME_WINDOW[1]
    out = np.zeros((len(align_times), n_clusters, NBINS), dtype=np.uint8)
    i0 = np.searchsorted(spike_times, np.where(finite, beg, np.inf), side='left')
    i1 = np.searchsorted(spike_times, np.where(finite, end, np.inf), side='left')
    for k in range(len(align_times)):
        t = spike_times[i0[k]:i1[k]]
        c = spike_clusters[i0[k]:i1[k]]
        b = np.floor((t - beg[k]) / BINSIZE).astype(np.int64)
        ok = (b >= 0) & (b < NBINS)
        counts = np.bincount(c[ok] * NBINS + b[ok], minlength=n_clusters * NBINS)
        np.clip(counts, 0, 255, out=counts)
        out[k] = counts.reshape(n_clusters, NBINS).astype(np.uint8)
    return out
```
```python
data['neural'].append([np.ascontiguousarray(neural[k], dtype=np.float32)
                       for k in range(neural.shape[0])])
```

iii. CONVERSION_NOTES Step 5, Key Decision 2: "**Spike counts, unnormalised.** The reference caches raw counts (`np.ubyte`) and z-scores only inside the model (`standardize_spike_data`). `train_decoder.py` does its own SVD-based projection, so raw counts are the right thing to store." Probe merging is justified by the reference docstring's reasoning that probes in one session "are not statistically independent as they have the same underlying behaviour". `bin_spiking_data_fast` is documented as a vectorised, bin-for-bin-identical replacement for the reference's `bincount2D`-per-trial-in-a-pool implementation, including the reference's `[:, :n_bins]` truncation convention (a spike exactly on `t_end` is discarded); Step 10 Check 2 rebuilds entire count matrices from raw spike times for 5 trials in each of 4 sessions and gets exact agreement.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two cluster-level filters, applied together before any binning:
* `clusters['label'] >= 1` — the IBL "well-isolated neuron" flag (the three RIGOR single-unit metrics: amplitude > 50 µV, noise cut-off < 20 µV, refractory-period violation);
* Beryl acronym not in `{'root', 'void'}` — the data paper's grey-matter restriction.

Surviving clusters are renumbered 0…n−1 and spikes of rejected clusters are discarded. Sessions left with fewer than 5 such neurons are dropped (3 sessions). Result: 890.1 clusters/probe → 108.5 well-isolated/probe → 62,757 grey-matter neurons over 441 sessions.

ii.
```python
QC_LABEL = 1.0                     # cluster['label'] >= 1  <=> IBL "well-isolated neuron"
NON_GREY_MATTER = ('root', 'void')  # Beryl acronyms that are not grey matter
MIN_NEURONS = 5                    # data paper: ">= 5 well-isolated neurons per session"
```
```python
def select_neurons(clusters, brain_regions):
    beryl = np.asarray(brain_regions.acronym2acronym(
        clusters['acronym'].to_numpy(), mapping='Beryl'))
    good = clusters['label'].to_numpy() >= QC_LABEL
    grey = ~np.isin(beryl, NON_GREY_MATTER)
    keep = np.nonzero(good & grey)[0]
    return keep, beryl[keep]
```
```python
if n_neurons < MIN_NEURONS:
    raise RuntimeError(f'only {n_neurons} well-isolated grey-matter neurons '
                       f'(< {MIN_NEURONS})')
```

iii. This is an explicit, documented **departure from the method-paper code**, which calls `load_spiking_data` with the default `qc=None` and therefore bins every Kilosort cluster. CONVERSION_NOTES Step 4 gives three reasons: "(a) this is the inclusion criterion the *data* paper applies to every analysis of this dataset and the standard `brainwidemap.load_good_units` behaviour; (b) the method paper's phrase 'all neurons' contrasts with its *region-restricted* decoders, not with QC; (c) including all 621k multi-unit clusters would make the converted dataset ~10x larger (≈120 GB) for units that the data paper explicitly says are not separable neurons." Dropping `root`/`void` is justified from the data paper ("restricted to regions that were designated grey matter in the adult mouse Allen CCF") and from the reference code's own `data_loader_utils.py:252`, which excludes exactly those two acronyms. Sanity checks: 108.5 vs the paper's 108 well-isolated units/probe, and 62,757 vs the paper's canonical 62,857 grey-matter neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams are already on one session clock, so alignment is a subtraction: the window of trial *k* is `[stimOn_times[k] − 0.5, stimOn_times[k] + 1.5)`, spikes in that range are located with `searchsorted` on the time-sorted merged spike train and their bin index is `floor((t − beg_k) / 0.02)`, i.e. time measured from the alignment event. NaN `stimOn_times` are guarded (replaced by `inf` in the `searchsorted` and skipped), and such trials are excluded by the trials mask anyway.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)          # seconds relative to the alignment event
```
```python
align_times = trials[ALIGN_TIME].to_numpy()
...
binned = bin_spiking_data_fast(spike_times, spike_clusters, n_neurons,
                               align_times[keep_trials])
```
```python
beg = align_times + TIME_WINDOW[0]
b = np.floor((t - beg[k]) / BINSIZE).astype(np.int64)
```

iii. `align_time='stimOn_times'`, `time_window=(-.5, 1.5)` are the reference `params` verbatim, and the Decoder Task here also mandates "Temporally align based on stimulus onset" (CONVERSION_NOTES Step 4 resolves this explicitly against the method paper's alternative first-movement alignment for the dynamic behaviours, choosing stimulus onset "as the task specification here mandates … and as the reference caching code does for *all* behaviours simultaneously"). Verified visually: the population PSTH is flat before 0 and rises sharply exactly at t = 0 (`processing_<eid>.png`), and an independent per-timepoint logistic-regression benchmark shows choice decodability jumping exactly at t = 0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins over a 2 s window → T = 100 bins per trial, identical for every trial and session; `metadata['time_bin_size'] = 20.0` (ms), `off_start = -0.5`, `off_end = 1.5`. Spikes are binned once, directly at 20 ms — there is no rebinning or resampling of the neural data. (The behavioural traces *are* resampled onto the same 100-bin grid; see 7-b/8-b.)

ii.
```python
TIME_WINDOW = (-0.5, 1.5)          # seconds relative to the alignment event
BINSIZE = 0.02                     # seconds
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # -> 100
BIN_CENTRES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 1)
```
```python
'time_bin_size': BINSIZE * 1000.0,
'off_start': TIME_WINDOW[0],
'off_end': TIME_WINDOW[1],
```

iii. The reference `params` set `'binsize': 0.02`, and the method paper states "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps". CONVERSION_NOTES Step 4 records and resolves the one conflicting source: the method paper's STAR-Methods prose mentions 50 ms bins for the choice/prior decoders — "The released caching code, the model description and the abstract-level description all use 20 ms/T = 100; the 50 ms sentence describes a separate per-trial decoding experiment. 20 ms is also required here because wheel speed and whisker motion energy must be decoded *per timepoint*."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Not from raw data at all — it is the fixed bin grid defined by `stimOn_times` alignment plus `TIME_WINDOW` and `BINSIZE`. The stored value is the bin **centre** in seconds relative to onset, so the same 100-value vector `[-0.49, -0.47, …, 1.49]` is written for every trial of every session.

ii.
```python
BIN_CENTRES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```
```python
inp = np.empty((n_keep, 2, NBINS), dtype=np.float32)
inp[:, 0, :] = BIN_CENTRES.astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 variable map: "bin index → `input[…][0]`, signed bin-centre time, −0.49 … +1.49 s — (task spec: 'time since stimulus onset, continuous, time-varying'), constant grid, identical for every trial". The window and bin size come from the reference `params`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid. The task specification asks for a continuous time-varying input, so the signed bin-centre time is used directly (rather than a binary onset indicator); it is stored as `float32` and broadcast identically to all trials.

ii.
```python
BIN_CENTRES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
inp[:, 0, :] = BIN_CENTRES.astype(np.float32)
```

iii. Purely definitional. Verified in Step 10 Check 2: "`input[0]` equals the bin-centre grid −0.49 … 1.49", and range-checked in Step 9 ("`time_from_stimulus_onset` range | −0.5…1.5 s | … | [−0.49, 1.49] (bin centres) | ✓").

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural binning grid: bin *k* of `neural` counts spikes in `[stimOn + (-0.5 + 0.02k), stimOn + (-0.5 + 0.02(k+1)))`, and `input[0][k]` is the centre of that same interval. So the two share a time axis bin for bin, with no offset. (Note the two behavioural *outputs* are instead sampled at the bin **right** edge, following the reference `get_behavior_per_interval`, so those lag the quoted input time by half a bin, 10 ms.)

ii.
```python
b = np.floor((t - beg[k]) / BINSIZE).astype(np.int64)   # neural bin index
```
```python
BIN_CENTRES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
inp[:, 0, :] = BIN_CENTRES.astype(np.float32)
```

iii. No alignment step is needed because the input is derived from the same grid. The `--show-processing` figure plots the time input against the PSTH and the raster on the same axis to show the ramp crosses zero exactly at stimulus onset (CONVERSION_NOTES Step 7, panel 6).

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. `trials['probabilityLeft']`, which is constant within a block, so a change of value marks a new block. No block-id column exists in the trials table.

ii.
```python
in_block, _ = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```
```python
p = np.asarray(probability_left, dtype=float)
new_block = np.ones(len(p), dtype=bool)
if len(p) > 1:
    new_block[1:] = ~(p[1:] == p[:-1])
block_id = np.cumsum(new_block) - 1
```

iii. CONVERSION_NOTES Step 5 variable map: "`trials.probabilityLeft` → `input[…][1]`". The `--show-processing` figure includes a panel overlaying `probabilityLeft` and the counter to show the counter "resets exactly where `probabilityLeft` changes, the first block is 90 trials long at pLeft = 0.5" — the IBL task design.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The 0-based position of the trial within its run of constant `probabilityLeft`, computed on the **complete, unfiltered** trials table and only then subset to the kept trials, so a trial that is later dropped still advances the counter. The scalar is broadcast over all 100 bins and stored as `float32`. Observed range over the full dataset: [0, 98]. NaN `probabilityLeft` always starts a new block (`NaN != NaN`), but such trials are removed by the trials mask.

ii.
```python
starts = np.nonzero(new_block)[0]
within = np.arange(len(p)) - starts[block_id]
return within.astype(np.float32), block_id
```
```python
inp[:, 1, :] = in_block[keep_trials][:, None]
```

iii. Docstring: "Computed on the *complete* trials table before any trial is excluded, so that the numbering reflects the block structure the mouse actually experienced (the IBL task has an initial 90-trial unbiased block at pLeft = 0.5, then blocks of 20-100 trials alternating between pLeft = 0.8 and 0.2)." CONVERSION_NOTES Step 5: "block structure must come from all trials, not the filtered subset." Verified in Step 10 Check 2 by independent recomputation from `trials.probabilityLeft`, and in Step 9 (`trial_number_in_block` range [0, 98], unbiased block = 90 trials).

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 / −1 / 0. `+1 → 0` (left), `−1 → 1` (right); `choice == 0` (no response) trials never reach this point because `load_trials_and_mask(exclude_nochoice=True)` removes them.

ii.
```python
choice = tr['choice'].to_numpy()
choice_out = (choice < 0).astype(np.int64)          # left -> 0, right -> 1
```
with the comment
```python
# choice: IBL codes +1 / -1. Verified empirically (see CONVERSION_NOTES Step 4):
# on correct trials with the stimulus on the left, choice == +1; on correct trials with
# the stimulus on the right, choice == -1. So +1 is a leftward choice.
```

iii. The sign convention was not assumed but **verified against the data**: CONVERSION_NOTES Step 10 Check 2 reports that in each of four independently re-loaded sessions, `choice == +1` occurs only on correct left-stimulus trials and `choice == −1` only on correct right-stimulus trials, "confirming the left = 0 / right = 1 mapping" required by the Decoder Task spec.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding above, plus broadcasting the per-trial scalar across all 100 time bins (the format instructions prefer time-varying outputs). Stored as `int64` in `output[0]`, with `output_values[0] = ['left', 'right']`. Full-dataset distribution: left 0.508 / right 0.492.

ii.
```python
out = np.empty((n_keep, 4, NBINS), dtype=np.int64)
out[:, 0, :] = choice_out[:, None]
```
```python
OUTPUT_VALUES = [
    ['left', 'right'],
    ...
]
```

iii. CONVERSION_NOTES Step 5, Key Decision 9: "**Both inputs and all four outputs are stored time-varying (d, 100)**, per-trial variables being broadcast along time, as the format instructions prefer." The near-50/50 split is checked in Step 12 as evidence that no output is dominated by one class.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. `trials['probabilityLeft']`, matched against the three task values with `np.isclose` and recoded `0.2 → 0`, `0.5 → 1`, `0.8 → 2`. Any unexpected value raises, so a silent mis-mapping cannot occur.

ii.
```python
pleft = tr['probabilityLeft'].to_numpy()
prior_out = np.full(n_keep, -1, dtype=np.int64)
for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_out[np.isclose(pleft, value)] = code
if np.any(prior_out < 0):
    bad = np.unique(pleft[prior_out < 0])
    raise RuntimeError(f'unexpected probabilityLeft values {bad}')
```

iii. Exactly the mapping the Decoder Task specifies ("Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2"); `probabilityLeft` is the reference code's `block` behaviour variable in `bin_behaviors`. Sanity-checked in Step 5/9: "`probabilityLeft` only ever 0.2 / 0.5 / 0.8, first block = 90 trials at 0.5".

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the recoding, then broadcasting the per-trial scalar over the 100 bins. `output_values[1] = ['p(left)=0.2', 'p(left)=0.5', 'p(left)=0.8']`. Full-dataset distribution 0.417 / 0.140 / 0.442, consistent with a 90-trial unbiased block in sessions averaging ~645 trials.

ii.
```python
out[:, 1, :] = prior_out[:, None]
```

iii. Step 9 consistency table: "prior distribution | 90 unbiased of ~645 ⇒ ~14% pLeft = 0.5, rest split | … | 0.417 / 0.140 / 0.442 | ✓". Step 10 Check 2 re-derives `output[1]` from `trials.probabilityLeft` independently of the conversion code.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()`, i.e. `_ibl_wheel.position` / `_ibl_wheel.timestamps` turned by the loader into a uniformly-resampled (~1 kHz) position and a low-pass-filtered velocity; wheel speed is `|velocity|` in rad/s. This is the reference's `load_target_behavior(one, eid, 'wheel-speed')` definition.

ii.
```python
sess_loader.load_wheel()
out['wheel-speed'] = (sess_loader.wheel['times'].to_numpy(),
                      np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. Docstring of `load_dynamic_behaviour`: "Reproduces `ibl_data_utils.load_target_behavior`: wheel speed = |velocity| of the uniformly resampled (~1 kHz) wheel trace". CONVERSION_NOTES Step 10 Check 3 marks the behaviour loading "identical" to the reference.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) `SessionLoader` internally interpolates wheel position onto an even ~1 kHz grid and differentiates it with a Butterworth low-pass to get velocity; the absolute value is taken. (2) For each trial the trace is sliced to the open interval `(beg, end)`, rejected if empty / NaN-containing / starting more than one bin late / ending more than one bin early, and otherwise linearly interpolated (with linear extrapolation at the edges) onto `np.linspace(beg + 0.02, end, 100)` — the **right edge** of each 20 ms bin, which is the reference convention. (3) The resulting (trials × 100) matrix of the session is discretized into tertiles (see 7-c).

ii.
```python
x = np.linspace(beg[k] + BINSIZE, end[k], NBINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
good[k] = True
```
```python
wheel_vals, wheel_ok = bin_behaviour_per_trial(*beh_traces['wheel-speed'], align_times)
...
wheel_kept = wheel_vals[keep_trials]
wheel_lab, wheel_thr = discretize_tertiles(wheel_kept)
```

iii. `bin_behaviour_per_trial` is documented as a "Faithful re-implementation of `ibl_data_utils.get_behavior_per_interval`", with the single documented deviation that NaNs always reject a trial: "The reference caching script passes `allow_nans=True` and lets the model impute NaNs with the trial average at fit time; here the converted outputs must be valid class labels at every timepoint, and `verify_data_format` rejects NaNs outright, so NaN-containing trials are dropped instead." Step 10 Check 2 reproduces `output[2]` bin-for-bin by re-slicing the raw wheel trace and re-running the same interpolation.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session equal-occupancy tertiles: the 1/3 and 2/3 quantiles are computed over **all retained (trial, time-bin) samples of that session**, and `np.searchsorted(thresholds, v, side='right')` assigns each sample to class 0 / 1 / 2 ('low' / 'medium' / 'high'). Thresholds are recorded per session in `metadata['session_info'][…]['wheel_speed_tertiles']`. Resulting distribution over the whole dataset: 0.333 / 0.333 / 0.333.

ii.
```python
def discretize_tertiles(values):
    finite = values[np.isfinite(values)]
    thresholds = np.quantile(finite, [1. / 3., 2. / 3.])
    labels = np.searchsorted(thresholds, values, side='right').astype(np.int64)
    return labels, thresholds
```

iii. Docstring and CONVERSION_NOTES Step 5, Key Decision 8: "Per-session thresholds are used because both signals are in session-specific units: … wheel-speed statistics depend on how vigorously a given mouse turns the wheel. A single global threshold would map whole sessions into one class; per-session tertiles give ~1/3 of samples per class in every session, so chance is a well-defined 1/3 and the shared decoder sees a comparable target in every session." Degenerate (coincident) thresholds are noted as still producing valid labels (Step 10 Check 5).

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel is on the same session clock as the spikes, so the only alignment is evaluating the trace at the same 100 per-trial sample times derived from the same `stimOn_times`. The sample times used are the bin **right edges** `stimOn + linspace(-0.48, 1.5, 100)`, matching the reference `get_behavior_per_interval`, so wheel bin *k* corresponds to the end of neural bin *k* (a 10 ms lead relative to the bin-centre time stored in `input[0]`).

ii.
```python
beg = align_times + TIME_WINDOW[0]
end = align_times + TIME_WINDOW[1]
...
x = np.linspace(beg[k] + BINSIZE, end[k], NBINS)
vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```
```python
# Behaviour is sampled at the *right edge* of each bin, exactly as the reference
# `get_behavior_per_interval` does (`np.linspace(beg + binsize, end, n_bins)`).
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 1)
```

iii. Explicitly chosen to match the reference convention (Step 10 Check 3: "(d) behaviour binning … identical, with `allow_nans=False`"). Alignment was checked visually: the `--show-processing` figure plots the resampled points on top of the raw ~1 kHz trace, and a wheel-speed heat map with trials sorted by reaction time in which the speed onset tracks the plotted `firstMovement_times` curve — "i.e. neural and behavioural streams are aligned to the same clock" (Step 7).

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `SessionLoader.load_motion_energy(views=['left'])` → `motion_energy['leftCamera']['whiskerMotionEnergy']` with its frame times (i.e. `leftCamera.ROIMotionEnergy.npy` + `_ibl_leftCamera.times.npy`), used as released with no further processing. If the left camera is missing, empty or all-NaN, the right camera is used instead. Over the full run: left 434 sessions, right 7, 14 sessions have neither and are dropped.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        me = sess_loader.motion_energy[key]
        times = me['times'].to_numpy()
        vals = me['whiskerMotionEnergy'].to_numpy()
        if len(times) == 0 or np.all(np.isnan(vals)):
            continue
        out['whisker-motion-energy'] = (times, vals)
        camera_used = view
        break
    except Exception:
        continue
if camera_used is None:
    raise RuntimeError('no whisker motion energy available')
```

iii. Docstring: "whisker motion energy = `whiskerMotionEnergy` of the **left** camera (60 Hz), falling back to the right camera (150 Hz) when the left one is unavailable — exactly the fallback in `ibl_data_utils.bin_behaviors`" (the reference tries `'left-whisker-motion-energy'` and falls back to `'right-whisker-motion-energy'` on `skip`).

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is (no filtering, no normalisation). It goes through the same `bin_behaviour_per_trial` as the wheel: slice to `(beg, end)`, reject on empty / NaN / late start / early end, then linear interpolation with edge extrapolation onto the 100 bin right edges; then per-session tertiles.

ii.
```python
me_vals, me_ok = bin_behaviour_per_trial(*beh_traces['whisker-motion-energy'], align_times)
...
me_kept = me_vals[keep_trials]
me_lab, me_thr = discretize_tertiles(me_kept)
...
out[:, 3, :] = me_lab
```

iii. Same justification as the wheel — the reference `get_behavior_per_interval` reproduced faithfully with `allow_nans=False`. Step 10 Check 2 reproduces `output[3]` bin-for-bin from the raw camera trace with the same `interp1d(..., fill_value='extrapolate')` onto `linspace(beg+0.02, end, 100)`.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: per-session 1/3 and 2/3 quantiles over all retained (trial, bin) samples of that session, `np.searchsorted(..., side='right')` → 0/1/2 ('low'/'medium'/'high'); thresholds recorded per session as `whisker_me_tertiles`. Full-dataset distribution 0.333 / 0.333 / 0.335.

ii.
```python
me_lab, me_thr = discretize_tertiles(me_kept)
```
```python
thresholds = np.quantile(finite, [1. / 3., 2. / 3.])
labels = np.searchsorted(thresholds, values, side='right').astype(np.int64)
```

iii. Docstring: "whisker motion energy is an uncalibrated pixel-intensity difference whose scale depends on the camera (left 1280x1024 @60 Hz vs right 640x512 @150 Hz), on illumination and on ROI placement … A single global threshold would map whole sessions into one class; per-session tertiles give ~1/3 of samples per class in every session, so chance is a well-defined 1/3." The `--show-processing` figure includes a class-balance bar chart verifying all six classes sit at 1/3.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The camera frame times are on the same session clock as the spikes, so the trace is simply evaluated at the same 100 per-trial sample times (bin right edges) computed from the same `stimOn_times` window as the neural bins. Trials where the camera does not span the window (start > 1 bin late or end > 1 bin early) are dropped rather than extrapolated across a gap.

ii.
```python
idx_beg = np.searchsorted(target_times, np.where(finite, beg, np.inf), side='right')
idx_end = np.searchsorted(target_times, np.where(finite, end, np.inf), side='left')
...
if abs(beg[k] - tt[0]) > BINSIZE:      # target data starts too late
    continue
if abs(end[k] - tt[-1]) > BINSIZE:     # target data ends too early
    continue
x = np.linspace(beg[k] + BINSIZE, end[k], NBINS)
```

iii. Same as 7-d: no clock correction is needed, and the sampling grid is the reference's. The `--show-processing` figure overlays the resampled points on the raw 60 Hz camera trace for a single trial.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Everything questionable is dropped rather than imputed, and every drop is counted and reported.
* **Probe with no released spike sorting** — skipped (`sp is None` / empty times / no cluster channels); if no probe of a session loads, the session fails.
* **NaN trial events** — removed by the reference trials mask; NaN `stimOn_times` additionally guarded in all three per-trial routines.
* **NaN in a behavioural slice** — the trial is dropped (the reference's `allow_nans=False` branch) instead of the reference's impute-at-fit-time behaviour, because outputs must be valid class labels.
* **Behavioural trace not spanning the window** — trial dropped.
* **Missing / interrupted electrophysiology** — trials whose window precedes the first spike, follows the last spike, or intersects a > 0.5 s gap in the pooled unfiltered spike train are dropped (33 trials dataset-wide).
* **Session with no whisker ME at all** — raises, session dropped (14 sessions).
* **Sessions with < 5 neurons or < 2 usable trials** — dropped (3 + 1 sessions).
* **Any other per-session exception** — caught in `_worker`, printed with a traceback, and recorded in `metadata['failed_sessions']`, so one bad session cannot abort the 441-session run.
* **Counting overflow** — spike counts clipped to 255 before the `uint8` cast.

ii.
```python
def _worker(eid):
    try:
        return convert_session(eid, verbose=True)
    except Exception as exc:       # noqa: BLE001 -- one bad session must not kill the run
        return {'eid': str(eid), 'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc()}
```
```python
if sp is None or 'times' not in sp or len(sp['times']) == 0:
    continue
if cl is None or len(cl.get('channels', [])) == 0:
    continue
```
```python
if np.isnan(tv).any():
    continue
```
```python
'failed_sessions': [{'eid': e, 'error': m} for e, m in failures],
```

iii. CONVERSION_NOTES Step 10, Check 1 and the iteration log: the 38 original "all neural data is zero" warnings were root-caused to (i) a 5 s spike-train dropout, (ii) ephys stopping 12 s before the behaviour, (iii) a one-neuron session; the fixes were the ephys-coverage test and the ≥ 5-neuron rule, after which 12 warnings remained and were each verified by counting spikes in the pooled unfiltered train (1,501–2,742 spikes present) — "the ten well-isolated units simply happen to be silent for those 2 s … Dropping them would throw away genuine, correctly-converted data, so they are kept." Step 10 Check 5 additionally enumerates edge cases (first/last trial, NaN `stimOn_times`, bin-edge spikes, one-probe vs two-probe sessions, missing left camera, degenerate tertiles, worker failures).

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting off disk dominates, by an order of magnitude: 3–30 s per session for 1–2 probes (200–500 MB of `spikes.*.npy` per probe), versus ~0.02 s for `eid2pid`, ~0.3 s for trials, ~0.5 s for behaviour and ~0.3 s for all binning. Serial total ≈ 2.2 h; with 32 session-level workers the measured wall clock was 203 s including an 18.5 s pickle write of the 11.7 GB output. The script instruments this itself with a per-step `timing` dict printed per session.

ii.
```python
t = time.time()
spikes, clusters = load_session_spikes(one, eid, pids, pnames)
timing['load_spikes'] = time.time() - t
```
```python
print(f'  [{eid}] probes={len(pids)} clusters={n_clusters_all} '
      ...
      f'({sum(timing.values()):.1f}s)', flush=True)
```

iii. CONVERSION_NOTES Step 7 run-time table attributes the cost to "spike sorting load (dominant; 1–2 probes) | 3–30 s | ~2.0 h serial", and Step 6 notes that parallelism was "moved up to the session level (`ProcessPoolExecutor`, 32 workers), which is where the real cost (loading 200–500 MB `spikes.*.npy` per probe) lies." Two I/O costs remain that the script does not remove: `SPIKES_ATTRIBUTES` is left at its default, so `spikes.amps` and `spikes.depths` are read although never used, and `load_spike_sorting()` is called without `check_hash=False`, so files are re-read for md5 verification.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain, all cheap relative to I/O. (1) The per-trial loop in `bin_spiking_data_fast` — already reduced to one `np.bincount` per trial via `searchsorted`, but could be a single global `bincount` over a `trial × unit × bin` flat index. (2) The per-trial loop in `bin_behaviour_per_trial` — `interp1d` is rebuilt per trial; a single `np.interp` over one concatenated query vector would do. (3) The per-trial list comprehensions in `main` that split the stacked `(n_trials, …, 100)` arrays into the per-trial lists the target format requires (unavoidable given the format). Total binning cost is ~0.3 s/session, so vectorising further would save ~2 min of a 203 s run. The important loops the *reference* runs — a multiprocessing pool calling `bincount2D` once per trial, plus a pool per behaviour per session — were already replaced.

ii.
```python
for k in range(len(align_times)):
    ...
    counts = np.bincount(c[ok] * NBINS + b[ok], minlength=n_clusters * NBINS)
```
```python
for k in range(n):
    ...
    vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```
```python
data['neural'].append([np.ascontiguousarray(neural[k], dtype=np.float32)
                       for k in range(neural.shape[0])])
```

iii. CONVERSION_NOTES Step 6 lists the reference inefficiencies ("`get_spike_data_per_interval` spawns a multiprocessing pool and calls `bincount2D` once per trial (≈600 pool tasks per session), each of which re-derives a cluster index with `np.intersect1d`"; "`get_behavior_per_interval` likewise uses a pool per behaviour per session") and the speedups adopted ("Spike binning vectorised with `searchsorted` + a single `np.bincount` per trial: ~0.02 s per session instead of seconds of pool overhead"). The agent does not flag its own residual loops, presumably because at ~0.3 s/session they are negligible next to the spike-file I/O.

## 10-c. What processing does the code repeat multiple times?

i. Little of consequence, and no data is loaded twice (one `SessionLoader` instance is shared between `load_trials_and_mask` and the behaviour loading; one ONE client and one `BrainRegions` per worker process via module-level singletons). The genuine repeats are small: the behavioural traces are interpolated for **all** trials, including the ~34 % later removed by the trials mask (the spikes, by contrast, are binned only for the kept trials); `clusters['label'] >= QC_LABEL` is evaluated twice (once in `select_neurons`, once for the `n_good_all` statistic); `BrainRegions()` is instantiated once per worker process rather than shared; and in `--show-processing` mode the first two sessions are converted serially with debug payloads and the rest in the pool.

ii.
```python
wheel_vals, wheel_ok = bin_behaviour_per_trial(*beh_traces['wheel-speed'], align_times)
me_vals, me_ok = bin_behaviour_per_trial(*beh_traces['whisker-motion-energy'], align_times)
...
binned = bin_spiking_data_fast(spike_times, spike_clusters, n_neurons,
                               align_times[keep_trials])   # kept trials only
```
```python
good = clusters['label'].to_numpy() >= QC_LABEL          # in select_neurons
...
n_good_all=int((clusters['label'].to_numpy() >= QC_LABEL).sum()),   # again in convert_session
```

iii. Not discussed as a problem in CONVERSION_NOTES; the agent's efficiency notes (Step 6) instead emphasise what it avoided repeating — "Spikes are binned **only for the trials that survive curation**" and per-process singletons for the ONE client and the atlas. Binning behaviour before the mask is in fact necessary as written, since `wheel_ok`/`me_ok` are themselves inputs to the trial mask.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent removed most of the reference's wasted work, and a little remains.

Removed: the reference's `raw_electrophysiology(band='ap', stream=True).fs` call (streams raw ephys over the network purely to fill a metadata field); the reference's `load_anytime_behaviors`, which loads six behavioural streams of which four (wheel velocity, right whisker ME, both pupil diameters) are never used — only the two required streams are loaded here.

Remaining: `brainbox.io.one.SPIKES_ATTRIBUTES` is left at its default `['clusters', 'times', 'amps', 'depths']`, so two large spike arrays are read per probe and never used (this roughly doubles the dominant I/O cost); `load_spike_sorting()` is called without `check_hash=False`, so files are re-read for md5 verification; `merge_clusters(...).to_df()` builds the full cluster-metrics table when only `label` and `acronym` are needed; the `wheel`/`me` traces are interpolated for trials later discarded; and several diagnostic statistics (`frac_correct`, `n_clusters_all`, `n_good_all`, per-session tertile thresholds, `failed_sessions`) are computed and stored in `metadata` but are not consumed by `train_decoder.py`. The last group is cheap and serves the required validation/consistency reporting.

ii.
```python
    The reference additionally queries `raw_electrophysiology(band='ap', stream=True).fs`
    to record the AP sampling rate. That streams raw ephys over the network and is not
    used by any downstream computation, so it is skipped.
```
```python
sp, cl, ch = ssl.load_spike_sorting()          # SPIKES_ATTRIBUTES default -> also amps, depths
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
```
```python
frac_correct=float((trials['feedbackType'].to_numpy() == 1).mean()),
...
'wheel_speed_tertiles': [float(x) for x in r['wheel_thresholds']],
'whisker_me_tertiles': [float(x) for x in r['me_thresholds']],
```

iii. CONVERSION_NOTES Step 6 lists the removals explicitly under "Code inefficiencies identified in the reference" and "Code speedups added" ("Only the two behaviours actually needed are loaded; no raw-ephys streaming"). The retained metadata is justified by the task's own requirement to document statistics and consistency checks. The unused `amps`/`depths` load and the hash check are not mentioned anywhere in the notes.
