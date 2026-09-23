# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read through the ONE API against the local cache at `/app/data/one_cache`; no ALF file is opened directly. The list of sessions is **not** obtained by searching the release index but taken from the reference repository's own freeze file, `/app/code/code_zhang2025/data/bwm_release.csv` (699 insertions / 459 eids / 139 subjects / 12 labs), which is the same file `0_data_caching.py` uses. Rows are grouped by `eid`, so one job carries an eid, all of its probe `pid`s / probe names, and the subject. If `/app/data/DATALIMIT_SUBSET.csv` exists the freeze is restricted to the eids it names. Per session, `SessionLoader` supplies the trials table (`load_trials`), the wheel (`load_wheel`) and the camera motion energy (`load_motion_energy`), and `SpikeSortingLoader` is instantiated once per probe insertion for `spikes.times` / `spikes.clusters` / clusters. Each worker process builds its own offline ONE client (no password, so the staged auth token is used and no network call is made).

ii.
```python
CACHE_DIR = '/app/data/one_cache'
FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'

def _get_one():
    global _ONE, _BR
    if _ONE is None:
        from one.api import ONE
        from iblatlas.regions import BrainRegions
        # no password: the staged auth token is used and ONE stays offline
        _ONE = ONE(base_url='https://openalyx.internationalbrainlab.org',
                   silent=True, cache_dir=CACHE_DIR)
        _BR = BrainRegions()
    return _ONE, _BR
```

```python
bwm = pd.read_csv(FREEZE_FILE, index_col=0)
...
limit_csv = '/app/data/DATALIMIT_SUBSET.csv'
if os.path.exists(limit_csv):
    sub = pd.read_csv(limit_csv)
    col = 'eid' if 'eid' in sub.columns else sub.columns[0]
    keep = set(sub[col].astype(str))
    bwm = bwm[bwm.eid.astype(str).isin(keep)]

jobs = []
for eid, g in bwm.groupby('eid', sort=False):
    jobs.append((eid, list(g.pid), list(g.probe_name), g.subject.iloc[0]))
```

```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
...
ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. From CONVERSION_NOTES Step 1/2: "Entry point: `0_data_caching.py` … `freeze_file = 'data/bwm_release.csv'`", and the agent iterates over exactly that set ("461 session directories are staged; the BWM freeze file … lists 699 probe insertions / 459 eids / 139 subjects / 12 labs, which is the set we iterate over"). It also recorded that ONE only works offline when constructed *without* a password ("passing password='international' forces a network re-auth which fails"). Loading is deliberately identical to the reference: Step 10 Check 3 row (a) states "one.eid2pid; SpikeSortingLoader.load_spike_sorting + merge_clusters; merge_probes; SessionLoader.load_trials/load_wheel/load_motion_energy → identical calls → yes".

## 1-b. How are the data split into subjects?

i. The subject is the `subject` column of the freeze file, carried along with each eid into the job tuple and stored on the per-session result. At assembly, `subjects` is the sorted set of subject names over the surviving sessions and `subject_idx` is each session's index into that list. 136 of the 139 freeze subjects survive; the 3 missing ones contributed only sessions that were skipped for missing whisker video.

ii.
```python
jobs.append((eid, list(g.pid), list(g.probe_name), g.subject.iloc[0]))
```

```python
subjects = sorted(set(s['subject'] for s in sessions))
subj_lut = dict((s, i) for i, s in enumerate(subjects))
...
'subject_idx': np.array([subj_lut[s['subject']] for s in sessions], dtype=np.int64),
```

iii. No derivation is needed — the release freeze already names the mouse for every insertion, so subject identity is read rather than parsed. The agent used the 139-subject figure from the data paper as a sanity target and explained the 136 vs 139 gap explicitly in Step 10 Check 4.

## 1-c. How are the data split into sessions?

i. The session is the natural unit: the freeze file is grouped by `eid` (`bwm.groupby('eid')`), so one job = one session = one entry of `neural`/`input`/`output`. Sessions with two insertions contribute both probes to the *same* session entry (probes merged, see 2-b). Jobs are sorted by eid for determinism, processed independently in a 24–32-worker `multiprocessing.Pool`, and the surviving sessions are re-sorted by eid before assembly so the ordering of `neural`, `subject_idx` and `brain_region_idx` agree. 459 sessions were attempted, 19 skipped, 440 kept.

ii.
```python
for eid, g in bwm.groupby('eid', sort=False):
    jobs.append((eid, list(g.pid), list(g.probe_name), g.subject.iloc[0]))
jobs.sort(key=lambda j: j[0])
```

```python
with ctx.Pool(processes=args.n_workers) as pool:
    for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
...
# order sessions deterministically
sessions.sort(key=lambda s: s['eid'])
```

iii. Nothing to decide — the release is organised by session and the reference code likewise loops over eids. Merging the probes of a session into one entry is the reference `merge_probes` behaviour, justified in Step 5 decision 7 ("Probes merged within a session (data paper: we did not perform decoding on these probes separately because they are not independent)").

## 1-d. How are the data split into trials?

i. The trials table has one row per trial, so the split is given. Each trial becomes the interval `[stimOn_times - 0.5, stimOn_times + 1.5)`, exactly the reference `bin_spiking_data` interval construction; those interval edges drive the spike binning, the behaviour interpolation and the coverage checks.

ii.
```python
tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
stim_on = trials[PARAMS['align_time']].to_numpy(dtype=float)
begs_all = stim_on + WIN[0]
ends_all = stim_on + WIN[1]
```

iii. CONVERSION_NOTES Step 1: "bin_spiking_data … Intervals = stimOn_times + (-0.5, 1.5)". The agent copied the reference `params` dict verbatim into the script (`PARAMS = {'interval_len': 2.0, 'binsize': 0.02, 'align_time': 'stimOn_times', 'time_window': (-0.5, 1.5)}`).

## 1-e. How are trials filtered based on quality controls?

i. Five filters are applied, all combined into one boolean `valid`:
1. **The reference trial mask, reproduced verbatim** — `load_trials_and_mask(..., min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True)`: reaction time (`firstMovement_times - stimOn_times`) must be in [0.08, 2.0] s, trial duration (`feedback_times - goCue_times`) ≤ 10 s, no NaN in `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType`, and `choice != 0` (no-response trials dropped).
2. **Behaviour coverage** — the reference `get_behavior_per_interval` criteria: the wheel and the whisker traces must have samples inside the window, must not start more than one bin (20 ms) late, and must not end more than one bin early; the interpolated values must all be finite.
3. **Prior sanity** — `probabilityLeft` must be one of 0.2 / 0.5 / 0.8.
4. **Neural coverage (added by the agent)** — the trial window must lie inside `[spike_times[0], spike_times[-1]]` and the binned matrix must contain at least one spike.
5. Sessions with fewer than 2 surviving trials are dropped entirely.
Unbiased (0.5) trials are deliberately **kept**, since the decoder output requires the 0.5 class. Result: 187,651 trials over 440 sessions (mean 426.5/session).

ii.
```python
def load_trials_and_mask(sess_loader):
    """Reference load_trials_and_mask with max_trial_len=10.0."""
    if sess_loader.trials.empty:
        sess_loader.load_trials()
    query = '(firstMovement_times - stimOn_times < %s)' % MIN_RT
    query += ' | (firstMovement_times - stimOn_times > %s)' % MAX_RT
    query += ' | (feedback_times - goCue_times > %s)' % MAX_TRIAL_LEN
    for event in NAN_EXCLUDE:
        query += ' | %s.isnull()' % event
    query += ' | (choice == 0)'
    mask = ~sess_loader.trials.eval(query)
    return sess_loader.trials, mask.to_numpy()
```

```python
valid = mask & wheel_ok & me_ok
...
pl = trials['probabilityLeft'].to_numpy()
valid &= (np.isclose(pl, 0.2) | np.isclose(pl, 0.5) | np.isclose(pl, 0.8))
...
in_span = (begs_all >= st[0]) & (ends_all <= st[-1])
valid &= in_span
idx = np.where(valid)[0]
...
has_spikes = binned.sum(axis=(1, 2)) > 0
if not has_spikes.all():
    idx = idx[has_spikes]
    binned = binned[has_spikes]
```

iii. Step 5 decision 4: "reference `load_trials_and_mask` logic with `max_trial_len=10.0` … plus the behaviour coverage mask from `get_behavior_per_interval`. Unbiased (pLeft = 0.5) trials are KEPT, because the Decoder Task requires the 0.5 prior class." The neural-coverage criterion was added in Step 10 Check 1 after 16 `all neural data is zero` warnings were traced to an ephys recording that stops before the behaviour (eid 8c2f7f4d, 3 trials start after the last spike) and a 2.07 s recording dropout (eid b182b754); the agent calls it "the exact analogue of the reference `get_behavior_per_interval` coverage check but for the spike train — an all-zero matrix is not a measurement, it is a hole in the recording". It noted that in a third session (eid 195443eb, 6 units at ~3 Hz) the zeros were genuine Poisson silence, and removed those trials as well (16 trials total, 0.009%).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` of every probe insertion of the session, returned by `SpikeSortingLoader.load_spike_sorting()`. The merged cluster table (`SpikeSortingLoader.merge_clusters(...).to_df()`) supplies two further columns used only for curation and labelling: `label` (the RIGOR single-unit QC score) and `acronym` (the Allen acronym, mapped to Beryl for `brain_regions`).

ii.
```python
ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
if spikes is None or len(spikes) == 0:
    continue
cl = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
st = np.asarray(spikes['times'])
sc = np.asarray(spikes['clusters'])
```

iii. Step 1 table: "`load_spiking_data` … `SpikeSortingLoader(pid).load_spike_sorting()` + `merge_clusters` → spikes dict, clusters DataFrame (has `label`, `acronym`)". The agent uses exactly these calls so the loading path is identical to the reference.

## 2-b. How is the `neural` data processed?

i. All probes of a session are merged into a single population: each probe's surviving clusters are renumbered with a lookup table that continues the numbering of the previous probe (`offset`), the spike arrays are concatenated and re-sorted by time. Spikes are then counted into 100 non-overlapping 20 ms bins covering `[stimOn - 0.5, stimOn + 1.5)`, using `floor((t - t_beg)/0.02)` with a clip at the last bin, one `np.bincount` per trial over a flattened (unit × bin) index. **Raw spike counts per 20 ms bin are stored (float32), not firing rates in Hz**; no smoothing and no normalisation (the reference decoder z-scores per time bin at training time).

ii.
```python
cluster_ids = cl.index.to_numpy()
kept_ids = cluster_ids[keep]
lut = np.full(int(cluster_ids.max()) + 2, -1, dtype=np.int64)
lut[kept_ids] = np.arange(len(kept_ids)) + offset
...
offset += len(kept_ids)
st = np.concatenate(times_l); sc = np.concatenate(clu_l)
order = np.argsort(st, kind='stable')
return st[order], sc[order], reg, n_all, n_good
```

```python
def bin_spikes(spike_times, spike_clusters, n_units, begs, ends):
    """Identical to the reference bincount2D(xbin=binsize, xlim=[t_beg, t_end])
    followed by [:, :n_bins]: bin i covers [t_beg + i*bs, t_beg + (i+1)*bs)."""
    out = np.zeros((n_trials, n_units, NBINS), dtype=np.float32)
    i0 = np.searchsorted(spike_times, begs, side='left')
    i1 = np.searchsorted(spike_times, ends, side='left')
    for k in range(n_trials):
        t = spike_times[a:b] - begs[k]
        bi = (t / BINSIZE).astype(np.int64)
        np.clip(bi, 0, NBINS - 1, out=bi)
        flat = spike_clusters[a:b] * NBINS + bi
        counts = np.bincount(flat, minlength=n_units * NBINS)
        out[k] = counts.reshape(n_units, NBINS)
```

iii. Step 3: "Neural: raw spike counts per bin (standardisation is done by the model, not the cache)"; metadata records `'neural_units': 'spike counts per 20 ms bin'`. Probe merging follows `merge_probes` and the data paper's argument that two probes in a session are not independent. The binning arithmetic was checked against the reference binner: "0 mismatches against `bincount2D` over 2000 random trials deliberately seeded with spikes sitting exactly on bin edges" — and an apparent mismatch found with `np.histogram2d` was traced to the floating-point edge convention (0.54/0.02 = 26.999999999999996), with the agent's `floor` arithmetic being the one that matches the reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Four cuts, of which only the first is a per-neuron quality criterion:
1. `clusters['label'] >= 1` — the "well-isolated" units of the data paper (all three RIGOR metrics passed). 599,865 Kilosort clusters → 73,044 units (108.2/probe).
2. Beryl acronym not in `('root', 'void')` — grey matter only.
3. A unit's Beryl region must have ≥ 5 surviving units **in that session**.
4. A region must appear in ≥ 2 sessions (applied globally after all sessions are processed; relaxed to ≥1 in `--sample` mode).
Cuts 2–4 are applied to neurons, not just to region-level analyses, so the final dataset holds 60,483 neurons of the 73,044 that pass the QC label (≈17% of QC-passing units discarded) over 209 of the 243 Beryl regions present. Sessions left with no units are dropped (4 sessions).

ii.
```python
good = cl['label'].to_numpy(dtype=float) >= 1.0
beryl = np.asarray(br.acronym2acronym(cl['acronym'].to_numpy(), mapping='Beryl'))
keep = good & ~np.isin(beryl, EXCLUDE_REGIONS)          # EXCLUDE_REGIONS = ('root', 'void')
```

```python
# >= MIN_UNITS_PER_REGION well-isolated units per region in this session
counts = Counter(reg.tolist())
keep_unit = np.array([counts[r] >= MIN_UNITS_PER_REGION for r in reg])
```

```python
region_session_count = Counter()
for s in sessions:
    for r in set(s['regions'].tolist()):
        region_session_count[r] += 1
keep_regions = set(r for r, c in region_session_count.items() if c >= min_sess)
...
km = np.isin(s['regions'], list(keep_regions))
if not km.all():
    s['neural'] = [np.ascontiguousarray(n[km]) for n in s['neural']]
    s['regions'] = s['regions'][km]
```

iii. The agent recorded the conflict explicitly (Step 4 table): the reference code calls `load_spiking_data` with `qc=None` (all clusters) but stores `good_clusters = label >= 1`, while the data paper restricts *every* analysis to the 75,708 well-isolated units. It chose `label >= 1` because "(a) it is the criterion the data paper applies to every analysis and gives a checkable statistic (75,708 total, 108 per probe); (b) the reference cache explicitly records the flag; (c) keeping all 621k units would make the pickle ~8x larger with mostly noise/MUA clusters". Cuts 2–4 are justified by one sentence of the data paper quoted in Step 3: "restricted to regions designated grey matter, containing at least five well-isolated neurons per session, and recorded from in at least two such sessions". The resulting 108.2 units/probe vs the paper's 108, and 888.7 vs 889 Kilosort units/probe, are presented as "the single strongest check that loading, probe merging and the `label >= 1` neuron curation reproduce the data paper exactly".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams share one session clock, so alignment is a subtraction: the trial interval is `stimOn_times + (-0.5, 1.5)`, spikes are sliced with `searchsorted` on the interval edges and their times are expressed relative to `begs = stimOn - 0.5` before binning. The behaviour is evaluated on `stimOn + BIN_TIMES`, i.e. the right edges of the very same bins, so neural bin *i* and behaviour sample *i* end at the same instant.

ii.
```python
stim_on = trials[PARAMS['align_time']].to_numpy(dtype=float)   # 'stimOn_times'
begs_all = stim_on + WIN[0]
ends_all = stim_on + WIN[1]
...
t = spike_times[a:b] - begs[k]
bi = (t / BINSIZE).astype(np.int64)
```
```python
grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]   # = stim_on + BIN_TIMES
vals = f(grid)
```

iii. Step 6 "Alignment proof": "`BIN_TIMES = linspace(-0.5 + 0.02, 1.5, 100)`. The behaviour grid is `stim_on + BIN_TIMES`, which is exactly the reference `np.linspace(t_beg + binsize, t_end, n_bins)` … Spike bin `i` covers `[stim_on - 0.5 + i*0.02, …)`, whose right edge is `BIN_TIMES[i]`. So behaviour sample `i` and spike bin `i` end at the same instant — no temporal offset." Checked empirically by a population PSTH that steps up at t = 0 and by a time-course analysis showing choice information appearing only after t = 0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial spanning (-0.5, +1.5) s — `binsize` and `time_window` copied from the reference `params`. `metadata['time_bin_size'] = 20.0` (ms), `off_start = -0.5`, `off_end = 1.5`. No rebinning, resampling or smoothing of the neural data: spikes are counted once, directly into the final grid. The behaviour traces are resampled (linear interpolation) onto that same grid, which is what the reference does.

ii.
```python
PARAMS = {'interval_len': 2.0, 'binsize': 0.02,
          'align_time': 'stimOn_times', 'time_window': (-0.5, 1.5)}
BINSIZE = PARAMS['binsize']
WIN = PARAMS['time_window']
NBINS = int(np.ceil((WIN[1] - WIN[0]) / BINSIZE))   # 100
# right edge of each bin relative to the alignment event; matches the reference
# get_behavior_per_interval grid np.linspace(t_beg + binsize, t_end, n_bins)
BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)
```

iii. Step 4 resolution: the methods paper text mentions 50 ms bins for choice/prior and a movement-aligned window for wheel/whisker, but "the reference CODE unifies all four variables on the stimulus-onset / 20 ms / (-0.5,1.5) grid, which is also exactly what the Decoder Task here specifies", and the methods paper overview ("2-s trials, each divided into 20-ms bins, producing T = 100 time steps") agrees.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable; it is the fixed bin-time grid relative to `stimOn_times`, i.e. the right edge of each of the 100 bins: `-0.48, -0.46, …, 1.50` s. The same 100-vector is used for every trial of every session, and is also stored in `metadata['bin_times_s']` with an explicit note that it is the right edge convention.

ii.
```python
BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)
...
time_row = BIN_TIMES.astype(np.float32)
inputs.append(np.stack([time_row, np.full(NBINS, tib[k], dtype=np.float32)]))
```

iii. Step 5 mapping table: "bin right-edge time relative to stimOn → input[s][k][0], -0.48 … 1.50 s, identical for every trial; new; required by Decoder Task". The right-edge convention was chosen because it is the reference `get_behavior_per_interval` grid, so the time input, the behaviour outputs and the neural bins are on one clock.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid — one `np.linspace`, broadcast to every trial as row 0 of the `(2, 100)` input array, cast to float32.

ii.
```python
BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)
time_row = BIN_TIMES.astype(np.float32)
```

iii. N/A — defined by the window and bin size, which come from the reference `params`. Verified in Step 10 Check 2 with `np.allclose(input[0], linspace(-0.48, 1.50, 100))`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural grid: `BIN_TIMES[i]` is the right edge of neural bin *i* (`[stimOn - 0.5 + i·0.02, stimOn - 0.5 + (i+1)·0.02)`), so element *i* of the time input and column *i* of the neural matrix describe the same 20 ms interval. It is also the grid on which the two behavioural outputs are interpolated.

ii.
```python
grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]
```
```python
'bin_time_convention': ('input[0] and bin_times_s give the RIGHT edge of '
                        'each 20 ms bin relative to stimulus onset'),
```

iii. See the "Alignment proof" quoted in 2-d. Note that `metadata['off_start'] = -0.5` is the window start while `input[0][0] = -0.48`; the agent documents this with the `bin_time_convention` field rather than shifting either one.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. `trials['probabilityLeft']`, which is constant within a block, so any change of value starts a new block. No block identifier is present in the trials table.

ii.
```python
tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. Step 5 mapping: "trial index within probabilityLeft block → input[s][k][1] … 0-based count since the last change of probabilityLeft". The agent verified the block structure against the data paper (90 unbiased trials, then blocks of 20–100 trials) and used max index ≤ 100 as a sanity check (observed max 98).

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A vectorised 0-based counter: mark the trials where `probabilityLeft` changes, carry the index of the last such change forward with `np.maximum.accumulate`, and subtract. It is computed on the **full** trials table *before* any masking, so a trial that is later dropped still advances the count and the value is the animal's true position in the block. The scalar is broadcast across all 100 time bins as row 1 of the input array (float32). Observed range over the full dataset: [0, 98].

ii.
```python
def trial_number_in_block(prob_left):
    """0-based index of each trial within its block of constant probabilityLeft.

    Computed on the full trials table so the value reflects the true position in
    the block even when intervening trials are excluded by the trial mask.
    """
    p = np.asarray(prob_left, dtype=float)
    new_block = np.ones(len(p), dtype=bool)
    new_block[1:] = p[1:] != p[:-1]
    starts = np.maximum.accumulate(np.where(new_block, np.arange(len(p)), 0))
    return np.arange(len(p)) - starts
```
```python
tib = tib_all[idx].astype(np.float32)
inputs.append(np.stack([time_row, np.full(NBINS, tib[k], dtype=np.float32)]))
```

iii. Step 10 Check 5: "`trial_number_in_block` is 0-based and computed on the FULL trials table, so it is the true position in the block even when neighbouring trials are excluded. Maximum observed 98, consistent with a block cap of 100." Re-derived independently from `probabilityLeft` in the Step 10 sanity checks (pass). The agent also checked whether the decoder normalises its inputs before deciding to leave the count unscaled (trajectory step 53).

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `trials['choice']`, which takes values +1 / −1 / 0. The convention was verified empirically rather than assumed: on correct, non-zero-contrast trials of eid 6713a4a7, `choice == +1` for every left-stimulus trial (236/236) and `choice == −1` for every right-stimulus trial (203/203), so +1 = reported LEFT and −1 = reported RIGHT. `choice == 0` (no response) trials are already removed by the trial mask.

ii.
```python
choice_raw = trials['choice'].to_numpy()[idx]
# IBL convention (verified empirically): +1 = reported LEFT, -1 = RIGHT
choice = (choice_raw < 0).astype(np.int64)
```

iii. Step 2/Step 4: "choice in {-1, 0, +1}. Empirically verified on eid 6713a4a7-…: on correct trials with a left-side stimulus choice == +1 (236/236) and with a right-side stimulus choice == -1 (203/203)." Re-checked in Step 10 Check 2 as a *semantic* test on 4 further sessions ("on correct non-zero-contrast trials the decoded choice must equal the stimulus side (n = 285–493 trials/session) → pass, confirms 0 = left / 1 = right").

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding +1 → 0 (left), −1 → 1 (right), broadcast across the 100 time bins so the output is time-varying in shape (constant within a trial), stored as int64. No-response trials never reach this point.

ii.
```python
outputs.append(np.stack([
    np.full(NBINS, choice[k], dtype=np.int64),
    np.full(NBINS, prior[k], dtype=np.int64),
    wheel_d[k], me_d[k]]))
```
```python
OUTPUT_VALUES = [['left', 'right'], ...]
```

iii. Step 5 decision 6: "per-trial variables (choice, prior, trial-in-block) are broadcast across the 100 time bins so that input/output are (d, T) arrays, as the format spec prefers." Observed distribution [0.508, 0.492], consistent with the near-balanced choice expected from the task.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. `trials['probabilityLeft']`, the block prior, which takes the three values 0.2 / 0.5 / 0.8. Trials whose value is not one of those three are dropped before the mapping.

ii.
```python
pl = trials['probabilityLeft'].to_numpy()
valid &= (np.isclose(pl, 0.2) | np.isclose(pl, 0.5) | np.isclose(pl, 0.8))
```

iii. Step 2: "`probabilityLeft` in {0.2, 0.5, 0.8}. First block is 90 trials of 0.5, then alternating 0.2/0.8." The instructions prescribe the 0.2→0 / 0.5→1 / 0.8→2 mapping; the agent kept the unbiased 0.5 trials precisely because the task requires that class (Step 5 decision 4).

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A three-way `np.select` on `np.isclose` comparisons (float-safe) giving 0 / 1 / 2, broadcast across the 100 bins as row 1 of the output array. Observed distribution over the full dataset [0.417, 0.140, 0.442], i.e. ~14% unbiased trials, which the agent checked against "90 unbiased trials of ~600 collected, with more exclusions early in the session".

ii.
```python
prior = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                   np.isclose(pleft, 0.8)], [0, 1, 2]).astype(np.int64)
```
```python
OUTPUT_VALUES = [..., ['0.2', '0.5', '0.8'], ...]
```

iii. Step 5 mapping table: "trials['probabilityLeft'] → output[s][k][1], 0.2 → 0, 0.5 → 1, 0.8 → 2 … per-trial, broadcast over T", the mapping given in the Decoder Task. Re-derived from `probabilityLeft` in the Step 10 sanity checks (pass).

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.wheel`, i.e. `_ibl_wheel.position` / `_ibl_wheel.timestamps` processed by `load_wheel` into a 1 kHz uniformly-sampled position, velocity and acceleration; the speed is `abs(velocity)`. This is literally the reference `load_target_behavior(one, eid, 'wheel-speed')` definition.

ii.
```python
sl.load_wheel()
wheel_t = sl.wheel['times'].to_numpy()
wheel_v = np.abs(sl.wheel['velocity'].to_numpy())
```

iii. Step 1 table: "`load_target_behavior` … wheel-speed = abs(SessionLoader.wheel['velocity'])". Step 10 Check 3 row (f) confirms the same source is used as the reference.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) `SessionLoader.load_wheel` does the IBL-standard processing — interpolation of the irregular wheel encoder onto a uniform 1 kHz grid and differentiation with a Butterworth low pass — which the agent uses with its defaults, so the trace is identical to the reference's. (2) The absolute value is taken and the trace is linearly interpolated (`scipy.interpolate.interp1d`, `kind='linear'`, `fill_value='extrapolate'`) onto `stimOn + BIN_TIMES` for every trial, with the reference coverage checks marking a trial bad if the trace is absent, starts more than one bin late or ends more than one bin early. One `interp1d` is built per session and evaluated on the whole `(n_trials, 100)` grid at once, rather than the reference's one-per-trial construction. (3) The resulting values are discretised (7-c).

ii.
```python
def interpolate_behavior(target_times, target_vals, begs, ends):
    """Mirrors get_behavior_per_interval: a trial is rejected if the trace has no
    samples inside the interval, starts more than one bin late, or ends more than
    one bin early."""
    ...
    ib = np.searchsorted(tt, np.nan_to_num(begs), side='right')
    ie = np.searchsorted(tt, np.nan_to_num(ends), side='left')
    good &= ie > ib
    ...
    first_t = tt[np.clip(ib[idx], 0, len(tt) - 1)]
    last_t = tt[np.clip(ie[idx] - 1, 0, len(tt) - 1)]
    ok = ((np.abs(begs[idx] - first_t) <= BINSIZE) &
          (np.abs(ends[idx] - last_t) <= BINSIZE))
    good[idx[~ok]] = False

    f = interp1d(tt, tv, kind='linear', fill_value='extrapolate',
                 bounds_error=False, assume_sorted=True)
    grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]
    vals = f(grid)
    good &= np.all(np.isfinite(vals), axis=1)
```

iii. Step 1: "Behaviour is interpolated, not averaged, onto the bin right-edges"; Step 10 Check 3 row (d): "behaviour resampling — reference `interp1d(..., kind='linear', fill_value='extrapolate')` on `linspace(t_beg+bs, t_end, 100)`; mine: same function, same grid, evaluated for all trials at once → yes". The vectorisation is documented as a pure speed change ("single per-session `interp1d` evaluated on the whole grid, ~50x on the behaviour step"). Non-finite samples are stripped from the trace before the interpolator is built.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 equally populated classes using the 33.3rd and 66.7th percentiles **of that session's own kept (trial × timepoint) samples**, via `np.nanquantile` + `np.digitize`. If the trace is degenerate (constant, or non-finite quantiles) the code falls back to a rank-based split so the output is still 3-valued and never NaN. The thresholds are recorded per session in the run log and drawn on the `--show-processing` figures. Resulting class fractions: exactly [0.333, 0.333, 0.333].

ii.
```python
def discretize_terciles(values):
    """Split a (n_trials, NBINS) array into 3 equally-populated per-session bins."""
    q1, q2 = np.nanquantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 == q2:
        flat = values.ravel()
        ranks = np.argsort(np.argsort(flat))
        d = (ranks * 3 // max(len(flat), 1)).reshape(values.shape)
        return np.clip(d, 0, 2).astype(np.int64), (float('nan'), float('nan'))
    return np.digitize(values, [q1, q2]).astype(np.int64), (float(q1), float(q2))
```
```python
wheel_k = wheel_vals[idx]
wheel_d, wheel_thr = discretize_terciles(wheel_k)
```

iii. Step 5 decision 5: "wheel speed and whisker motion energy are discretised into 3 bins at the per-session 33.3rd/66.7th percentiles of all (trial × timepoint) samples in that session. Per-session rather than global, because whisker motion energy is in uncalibrated camera-dependent units (left camera 60 Hz vs right camera 150 Hz, different resolution and illumination) so a global threshold would largely encode session identity; terciles also give exactly balanced classes." The agent also argued in Step 12 Check 3 that per-session thresholds are not leakage, being "a fixed monotone recoding of the behaviour [that] does not transfer label information between trials".

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The trace is evaluated at `stimOn + BIN_TIMES`, the right edges of the same 100 neural bins, so sample *i* of the wheel output is the instantaneous speed at the end of neural bin *i*. Trials whose wheel trace does not cover the window are dropped from *both* streams, since the coverage flag feeds the common `valid` mask.

ii.
```python
grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]
vals = f(grid)
```
```python
wheel_vals, wheel_ok = interpolate_behavior(wheel_t, wheel_v, begs_all, ends_all)
me_vals, me_ok = interpolate_behavior(me_t, me_v, begs_all, ends_all)
valid = mask & wheel_ok & me_ok
```

iii. Step 6 "Alignment proof" (quoted in 2-d) plus the `--show-processing` panel that overlays the raw `abs(velocity)` trace with the interpolated samples for one trial; Step 10 Check 2 re-interpolates and re-tercilises the wheel from the raw files for 4 sessions and compares with `np.allclose` (pass).

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `SessionLoader.motion_energy['leftCamera']['whiskerMotionEnergy']` with its `times` (i.e. `leftCamera.ROIMotionEnergy.npy` + `_ibl_leftCamera.times.npy`), falling back to the right camera if the left is unavailable — the reference `bin_behaviors` preference order. The camera actually used is recorded per session (`info['motion_energy_view']`). 14 sessions had neither and were skipped.

ii.
```python
me_t = me_v = me_view = None
for view in ('left', 'right'):
    try:
        sl.load_motion_energy(views=[view])
        key = view + 'Camera'
        if key in sl.motion_energy and \
                'whiskerMotionEnergy' in sl.motion_energy[key]:
            me_t = sl.motion_energy[key]['times'].to_numpy()
            me_v = sl.motion_energy[key]['whiskerMotionEnergy'].to_numpy()
            me_view = view
            break
    except Exception:
        continue
if me_t is None:
    return None, dict(info, skip='no whisker motion energy')
```

iii. Step 4 table: "Whisker camera — code says left, fall back to right … Left camera preferred, right as fallback (exactly the reference `bin_behaviors` logic)". Step 2 notes the sampling rates (left ~60 Hz, right 150 Hz), matching the data paper's "sampled at 60 Hz".

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released ROI motion-energy trace is used as-is — no filtering, no normalisation, no baseline subtraction. It goes through exactly the same `interpolate_behavior` path as the wheel: non-finite samples stripped, coverage checks at one-bin tolerance, one linear `interp1d` per session evaluated on `stimOn + BIN_TIMES` for all trials, then tercile discretisation.

ii.
```python
me_vals, me_ok = interpolate_behavior(me_t, me_v, begs_all, ends_all)
...
me_k = me_vals[idx]
me_d, me_thr = discretize_terciles(me_k)
```

iii. Step 5 mapping: "motion_energy leftCamera whiskerMotionEnergy (fallback right) → output[s][k][3]: interp to bin right edges, then per-session terciles → {0,1,2}", citing `load_target_behavior('left-whisker-motion-energy')` and `get_behavior_per_interval` as the reference counterparts. The raw-vs-interpolated overlay panel of the processing figure is the visual check.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: per-session 33.3 / 66.7 percentiles over all kept (trial × timepoint) samples of that session, `np.digitize`, with the rank-split fallback for degenerate traces. Class fractions exactly [0.333, 0.333, 0.333]; the two thresholds are logged per session.

ii.
```python
me_d, me_thr = discretize_terciles(me_k)
info['whisker_thresholds'] = me_thr
```
```python
'discretization': ('wheel speed and whisker motion energy: per-session '
                   '33.3/66.7 percentiles over all (trial x timepoint) samples'),
```

iii. Same justification as 7-c, with the additional camera-specific argument: motion energy is in uncalibrated, camera-dependent units (left 60 Hz vs right 150 Hz, different resolution and illumination), so a dataset-wide threshold "would largely encode session identity".

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same grid as everything else: values at `stimOn + BIN_TIMES`, the right edges of the neural bins. The camera frame times are on the session clock, so no further synchronisation is needed; a trial whose camera trace starts more than one bin late or ends more than one bin early is dropped from every stream at once.

ii.
```python
valid = mask & wheel_ok & me_ok
```
```python
ok = ((np.abs(begs[idx] - first_t) <= BINSIZE) &
      (np.abs(ends[idx] - last_t) <= BINSIZE))
```

iii. Step 6 alignment proof and the Step 10 Check 2 sanity check ("output[3] whisker ME re-interpolated and re-tercilised → pass"). Note that interpolating a ~60 Hz trace onto a 50 Hz grid is upsampling of a slower stream, which the agent accepted because it is precisely what the reference `get_behavior_per_interval` does.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Everything problematic is either repaired locally or dropped, and the reason is logged:
- A probe whose spike sorting is empty is skipped; the cluster renumbering continues consistently for the remaining probes.
- Non-finite spike times and out-of-range cluster ids are filtered out before binning.
- Non-finite behaviour samples are removed before the interpolator is built; a trial whose interpolated behaviour is not all finite is dropped.
- Trials without full wheel / camera coverage are dropped; sessions with no whisker motion energy at all are skipped (14 sessions).
- `probabilityLeft` values outside {0.2, 0.5, 0.8} are dropped.
- Trial windows outside the spike-sorting span, or inside a recording dropout (all-zero matrix), are dropped.
- Degenerate tercile thresholds fall back to a rank split, so no NaN can reach the output.
- Sessions with < 2 usable trials or no surviving units are dropped with an explicit message, and any unexpected exception in a session is caught, recorded with its traceback, and that session alone is skipped.
19 of 459 sessions and 16 trials (0.009%) were removed this way; every skip reason is printed in `conversion_full_out.txt`.

ii.
```python
if spikes is None or len(spikes) == 0:
    continue
...
ok = np.isfinite(st) & (sc >= 0) & (sc < len(lut))
st, sc = st[ok], sc[ok]
```
```python
finite = np.isfinite(tt) & np.isfinite(tv)
tt, tv = tt[finite], tv[finite]
if len(tt) < 2:
    return vals, np.zeros(n_trials, dtype=bool)
```
```python
except Exception as exc:
    info['skip'] = 'error: %s' % exc
    info['traceback'] = traceback.format_exc()
    return None, info
```

iii. Step 10 Check 5 lists each edge case and its handling; Step 9 tabulates the 19 skipped sessions by reason (14 no whisker motion energy, 4 no region with ≥ 5 units, 1 too few trials after the behaviour mask). The guiding rule the agent states for the zero-spike case is that "an all-zero matrix is not a measurement, it is a hole in the recording, and it carries no information for the decoder" — although it had itself established that in one session (195443eb) the zeros were genuine Poisson silence rather than a recording fault.

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting from disk (`load_spike_sorting` + `merge_clusters`), which dominates a session: whole-session processing measured 4.1 s and 7.3 s on a 1-probe and a 2-probe session, of which the binning and the behaviour interpolation together are a small fraction. At the whole-run level the remaining costs are the 32-way process pool (459 sessions in 2.6 min wall clock) and then serialising the 11.33 GB pickle plus the end-of-run `np.concatenate` over every session's outputs for the summary table (total run 3.0 min). Per-session timings are printed (`info['time_s']`) and a running ETA is logged every session.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
```
```python
print('[%d/%d] %s %s | elapsed %.1f min, eta %.1f min'
      % (i + 1, len(jobs), info['eid'], status, el / 60.0, eta), flush=True)
```
```python
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. Step 7 documents the measured per-session times, the extrapolation to "459 × 5.7 s = 44 min serial", and the conclusion that with 24–32 workers the run is "well under 15 min"; the actual full run took 3.0 min.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain, all cheap: (1) the per-trial loop in `bin_spikes` — it could be a single `np.bincount` over a trial-offset flat index for the whole session, but each trial is a different slice and the loop body is already pure NumPy; (2) the per-trial loop that assembles the `neural` / `input` / `output` lists — unavoidable in practice because the target format is a list of per-trial arrays; (3) the Python comprehension `[counts[r] >= 5 for r in reg]` and the `[reg_lut[r] for r in s['regions']]` region lookups, which are O(n_units) Python loops that could be `np.unique`-based. The agent had already vectorised the two loops that mattered, relative to the reference: the reference spawns a multiprocessing pool per *trial* and calls `bincount2D` per trial, and builds a separate `interp1d` per trial.

ii.
```python
for k in range(n_trials):
    a, b = i0[k], i1[k]
    ...
    counts = np.bincount(flat, minlength=n_units * NBINS)
    out[k] = counts.reshape(n_units, NBINS)
```
```python
for k in range(n_trials):
    neural.append(np.ascontiguousarray(binned[k]))
    inputs.append(np.stack([time_row, np.full(NBINS, tib[k], dtype=np.float32)]))
    outputs.append(np.stack([...]))
```
```python
counts = Counter(reg.tolist())
keep_unit = np.array([counts[r] >= MIN_UNITS_PER_REGION for r in reg])
```

iii. Step 6 "Code speedups added": "Binning is a single `np.searchsorted` + per-trial `np.bincount` on a flattened (unit, bin) index; no pools, no Python-level per-spike work. Interpolation builds ONE `interp1d` per session and evaluates it on the full (n_trials × 100) grid in one call. Parallelism is moved up to the session level." Claimed savings: ~20× on binning, ~50× on behaviour, ~20× wall clock from session-level parallelism.

## 10-c. What processing does the code repeat multiple times?

i. Little of consequence, and nothing inside the hot path:
- `sl.load_trials()` is called in `process_session` and `load_trials_and_mask` re-checks and would load again (guarded by `if sess_loader.trials.empty`, so the second call is a no-op).
- The prior is validated twice: `probabilityLeft.isnull()` is already in the reference NaN mask, and the code then re-tests membership in {0.2, 0.5, 0.8}.
- In `--show-processing` mode the wheel and the motion energy of a session are loaded a *second* time in `main` purely to draw the raw traces.
- If the left camera is missing, `load_motion_energy` is attempted twice (left then right).
- The end-of-run summary re-stacks and concatenates every session's outputs and inputs to print distributions, after they were already built.

ii.
```python
sl.load_trials()
trials, mask = load_trials_and_mask(sl)     # re-checks sl.trials.empty
```
```python
if args.show_processing:
    sl = SessionLoader(one=one, eid=job[0])
    sl.load_wheel()
    ...
    sl.load_motion_energy(views=[view])
```
```python
allout = np.concatenate([np.stack(s['output']) for s in sessions], axis=0)
```

iii. Not discussed as such in CONVERSION_NOTES; the repeats are all diagnostic or guarded no-ops, and the agent's documented efficiency effort went into removing the reference's per-trial pools and per-trial interpolators instead.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A modest amount, all of it bookkeeping or diagnostics:
- Each session result carries a `raw` dict (un-discretised wheel and whisker traces, `stim_on`) that exists only for the `--show-processing` plots and is `pop`ped before pickling — so it is built for all 440 sessions but used for at most 2.
- `n_clusters_all` / `n_clusters_good` are accumulated for every probe purely to reproduce the paper's 889 and 108 units-per-probe statistics.
- The end-of-run concatenation of all inputs and outputs (≈0.75 GB of temporaries) exists only to print the distribution table.
- Trials that are later discarded by the `has_spikes` test are binned first, and `interpolate_behavior` evaluates the interpolator on the full grid for *all* trials including ones already marked bad.
- The neural counts are stored as float32 although they are small integers, which inflates the pickle ~4× (11.33 GB) relative to an integer dtype; the outputs are int64 rather than int8, adding another ~0.5 GB.

ii.
```python
session = {..., 'raw': {'wheel_vals': wheel_k, 'me_vals': me_k,
                        'stim_on': stim_on[idx]}}
...
for s in sessions:
    s.pop('raw', None)
```
```python
out = np.zeros((n_trials, n_units, NBINS), dtype=np.float32)
```
```python
outputs.append(np.stack([
    np.full(NBINS, choice[k], dtype=np.int64),
    np.full(NBINS, prior[k], dtype=np.int64),
    wheel_d[k], me_d[k]]))
```

iii. Not flagged in CONVERSION_NOTES as waste; the agent's stated memory decision was the opposite direction — "Spike counts stored as `float32` (the format expects float and this halves the pickle)" (Step 6) — i.e. it considered float32 the economical choice relative to float64 rather than relative to an integer dtype. The diagnostic extras are deliberate, being the material for the Step 9/10 consistency tables.
