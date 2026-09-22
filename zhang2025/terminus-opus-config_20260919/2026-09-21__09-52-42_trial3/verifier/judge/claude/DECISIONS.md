# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read through the ONE API against the local cache at `/app/data/one_cache`, in
offline `mode='local'`, using the same loaders as the reference code: `SessionLoader` for the
trials table, wheel and camera motion energy, and `SpikeSortingLoader` (one per probe insertion)
for spikes and clusters. Two loading decisions differ in mechanism from a plain ONE search:

1. **The shipped ONE release tables are stale, so the datasets index is rebuilt from the
   filesystem.** The AI found that essential datasets live inside dated revision folders
   (e.g. `alf/#2025-03-03#/_ibl_trials.table.pqt`) that no shipped table indexes (459/459
   sessions have the trials table only under a revision). The symptom was silent, not an
   exception: `SessionLoader.load_trials()` returned a `(565, 1)` table containing only
   `goCueTrigger_times`. `cache/build_one_cache.py` regenerates the `datasets` index from disk
   into `/app/data/one_cache/LocalIndex`, keeping the real eids from the shipped sessions table
   and flagging the newest revision of each (session, collection, filename) as
   `default_revision=True`. It is rebuilt automatically by `get_one()` if absent.
2. **Sessions and probe insertions come from the reference repo's release freeze,
   `bwm_release.csv`**, rather than from `one.search`/`one.eid2pid` (`eid2pid` raises
   `NotImplementedError` in local mode). This is the same file the reference
   `0_data_caching.py` loads, and it maps eid -> pid -> probe_name directly (459 eids, 699 pids,
   139 subjects). `DATALIMIT_SUBSET.csv` would restrict the list if present; it is absent here,
   so all 459 released sessions are attempted.

Per-session loading then happens inside one worker process per session
(`ProcessPoolExecutor`, 24 workers): spikes for each probe, the trials table, the wheel and the
motion energy.

ii.
```python
CACHE_DIR = '/app/data/one_cache'
TABLES_DIR = '/app/data/one_cache/LocalIndex'
FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'

def get_one():
    from one.api import ONE
    if not os.path.exists(os.path.join(TABLES_DIR, 'datasets.pqt')):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache'))
        import build_one_cache
        build_one_cache.main()
    return ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local',
               cache_dir=CACHE_DIR, tables_dir=TABLES_DIR)

def session_list():
    """Sessions to convert: the BWM release freeze, as used by the reference."""
    bwm = pd.read_csv(FREEZE_FILE, index_col=0)
    if os.path.exists(DATALIMIT):
        keep = pd.read_csv(DATALIMIT)
        col = 'eid' if 'eid' in keep.columns else keep.columns[0]
        bwm = bwm[bwm.eid.isin(set(keep[col].astype(str)))]
    return bwm
```

```python
def load_session_spikes(one, eid, probes):
    from brainbox.io.one import SpikeSortingLoader
    for _, r in probes.iterrows():
        ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
        spikes, clusters, channels = ssl.load_spike_sorting()
        ...
        df = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()

def trials_and_mask(one, eid):
    from brainbox.io.one import SessionLoader
    sl = SessionLoader(one=one, eid=eid)   # kw-only in ibllib 4.0.1
    sl.load_trials()
```

iii. From CONVERSION_NOTES Step 2/4: the release tables "predate the staged files; essential
datasets are revision-only and unindexed ... Without it `load_trials()` silently returns a single
column", so rebuilding the index is "mandatory". `bwm_release.csv` is used because "`eid2pid`
raises NotImplementedError in local mode (requires remote connection). Use `bwm_release.csv`, the
same freeze file the reference loads, which already maps pid to eid and probe_name."
`raw_electrophysiology(...).fs` is deliberately skipped because it needs the network and "the
reference only stores it as metadata and never uses it in processing."

## 1-b. How are the data split into subjects?

i. The subject name is taken from the `subject` column of `bwm_release.csv` for the session's
first row, so no path parsing is involved. At assembly, `subjects` is the sorted set of unique
names and `subject_idx` is the index of each session's subject into that list. Result: 136
subjects over the 441 converted sessions (139 in the freeze; 3 subjects are lost because their
only session was dropped), mean 3.24 sessions/subject, max 13.

ii.
```python
    result = dict(
        eid=eid,
        subject=str(probes.iloc[0].subject),
        lab=str(probes.iloc[0].lab),
        ...
```
```python
    subjects = sorted({r['subject'] for r in results})
    sub_idx = {s: i for i, s in enumerate(subjects)}
    ...
        'subjects': subjects,
        'subject_idx': np.array([sub_idx[r['subject']] for r in results], dtype=int),
```

iii. The freeze file already carries a unique subject id per session, so nothing has to be
derived; the notes verify 139 subjects against the paper's "We trained 139 mice".

## 1-c. How are the data split into sessions?

i. The session is the unit of the release: the unique `eid` values of `bwm_release.csv` are the
sessions (459), and one job/one worker call is submitted per eid. The two probe insertions of a
two-probe session are merged into that single session rather than treated as separate sessions.

ii.
```python
    bwm = session_list()
    eids = list(dict.fromkeys(bwm.eid))
    if args.sample:
        eids = eids[:2]
    jobs = [(e, bwm[bwm.eid == e], args.show_processing and i < 2, outdir)
            for i, e in enumerate(eids)]
```

iii. No decision to make — the release is organised by session. Merging probes within a session
follows the data paper ("neurons in the same session and region were combined across probes") and
the reference `merge_probes`, whose rationale is that probes in one session share the same
behaviour and so are not independent.

## 1-d. How are the data split into trials?

i. The trials table has one row per trial, so the split is given by the data. Each retained trial
becomes a 2 s window, `stimOn_times + (-0.5, 1.5)` s, of 100 bins.

ii.
```python
    sl.load_trials()
    tr = sl.trials
...
    keep = np.where(mask)[0]
    align = trials[ALIGN_TIME].to_numpy()[keep]
    finite = np.isfinite(align)
    keep, align = keep[finite], align[finite]
```

iii. No decision to make — the trials table is already one row per trial; the reference builds its
intervals the same way (`trials_df[align_time] + time_window`).

## 1-e. How are trials filtered based on quality controls?

i. The reference code's inclusion mask is re-implemented exactly as the reference *calls* it,
`load_trials_and_mask(max_trial_len=10.0)`, i.e. a trial is dropped if any of:
- reaction time `firstMovement_times - stimOn_times` < 0.08 s or > 2 s;
- `feedback_times - goCue_times` > 10 s;
- `choice == 0` (no response);
- NaN in any of `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`,
  `firstMovement_times`, `feedbackType`.

Three further cuts are added: `stimOn_times` must be finite; the wheel and the whisker trace must
each span the trial window to within one bin (the reference's own `get_behavior_per_interval`
rejection rule); and a trial whose `probabilityLeft` is not one of 0.2/0.5/0.8 is dropped so the
prior output is always a valid class. A session with fewer than 2 surviving trials is dropped
entirely. Result: 188,020/285,031 = 66.0 % kept by the reference mask, 187,934 after the coverage
requirement (86 more removed).

ii.
```python
MIN_RT, MAX_RT, MAX_TRIAL_LEN = 0.08, 2.0, 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
...
    rt = tr['firstMovement_times'] - tr['stimOn_times']
    bad = (rt < MIN_RT) | (rt > MAX_RT)
    bad |= (tr['feedback_times'] - tr['goCue_times']) > MAX_TRIAL_LEN
    bad |= tr['choice'] == 0
    for col in NAN_EXCLUDE:
        bad |= tr[col].isna()
    return tr, (~bad).to_numpy(), sl
```
```python
    valid = ok_w & ok_m                      # wheel and whisker both cover the window
    if valid.sum() < 2:
        raise RuntimeError(f'only {valid.sum()} trials with complete behaviour')
    keep, align = keep[valid], align[valid]
```
```python
    if (y_prior < 0).any():                  # probabilityLeft not in {0.2, 0.5, 0.8}
        okp = y_prior >= 0
        keep, align, counts, wheel, whisk = (keep[okp], align[okp], counts[okp],
                                             wheel[okp], whisk[okp])
```

iii. "identical predicate, re-implemented because `SessionLoader` is now keyword-only" (Step 10,
Check 3). The notes cross-check the criteria against the data paper's own trial curation
("exclude trials with undetected choice ... outside 0.08-2.00 s") and measure the mask's effect on
the raw files independently (195,781/296,090 = 66.1 % over all 459 sessions) before conversion.
`exclude_unbiased` is deliberately left False because the 0.5 block is a required class of the
prior output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` of every probe insertion of the session, merged. The
merged clusters table (from `SpikeSortingLoader.merge_clusters`) supplies `label` (quality) and
`acronym` (anatomy), which select which units contribute, but the array itself is built only from
spike times and spike cluster ids.

ii.
```python
        spikes, clusters, channels = ssl.load_spike_sorting()
        df = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
        times.append(spikes['times'])
        clus.append(spikes['clusters'] + offset)
        offset += int(df.index.max()) + 1
        tables.append(df)
    return (np.concatenate(times), np.concatenate(clus),
            pd.concat(tables, ignore_index=True))
```

iii. Mirrors reference `prepare_data` + `merge_probes`: "cluster ids are offset per probe so they
stay unique, and the clusters tables are concatenated in the same order."

## 2-b. How is the `neural` data processed?

i. Spikes of the selected units are counted into 20 ms bins over each trial's 2 s window, giving a
`(n_neurons, 100)` matrix per trial. Counts are **not** divided by the bin width, so the stored
values are spike counts per 20 ms bin (metadata `neural_units` says so); the provided decoder
z-scores internally. No smoothing is applied. Probes of one session are pooled into a single
population with continuous unit numbering. Implementation is a vectorised equivalent of the
reference: spikes are restricted to the kept units, sorted once per session, each trial window is
sliced with `searchsorted`, and one `np.bincount` fills the unit-by-bin grid. Stored as float32
(after a first version used int16, which made the verifier warn).

ii.
```python
def bin_spikes(spike_times, spike_clusters, unit_idx, align_times):
    remap = np.full(int(spike_clusters.max()) + 2, -1, dtype=np.int64)
    remap[unit_idx] = np.arange(len(unit_idx))
    sel = np.isin(spike_clusters, unit_idx)
    st, sc = spike_times[sel], remap[spike_clusters[sel]]
    order = np.argsort(st, kind='stable')
    st, sc = st[order], sc[order]

    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    i0 = np.searchsorted(st, begs, side='left')
    i1 = np.searchsorted(st, ends, side='left')

    out = np.zeros((len(align_times), n_units, N_BINS), dtype=np.float32)
    for k in range(len(align_times)):
        ...
        tb = ((st[s] - begs[k]) / BINSIZE).astype(np.int64)
        np.clip(tb, 0, N_BINS - 1, out=tb)
        flat = np.bincount(sc[s] * N_BINS + tb, minlength=n_units * N_BINS)
        out[k] = flat.reshape(n_units, N_BINS)
    return out
```

iii. "vectorised `searchsorted` + `bincount`; **proved bit-identical** to `bincount2D`
(`np.array_equal` True over 40x61x100, 29,223 spikes)" — the AI ran the reference
`get_spike_data_per_interval` path against its own on the same session before trusting it.
Counts rather than rates because "binning yields counts; the provided decoder z-scores
internally"; float32 because int16 produced one verifier warning per trial (Step 7).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three cuts:
1. `clusters.label >= 1` — all three RIGOR single-unit metrics passed (amplitude > 50 µV, noise
   cut-off < 20 µV, refractory-period violation). Verified against the paper: scanning all 699
   `clusters.metrics.pqt` files gives 621,733 units total and **exactly 75,708** with
   `label >= 1`, the paper's well-isolated count (108.3/probe vs the paper's 108).
2. Beryl acronym not in `('root', 'void')` — the grey-matter restriction.
3. A session must retain at least 5 such units (`MIN_UNITS_PER_SESSION = 5`), otherwise it is
   dropped (3 sessions dropped).

Final: 62,757 neurons over 441 sessions (73,010 were `label >= 1`; ~10k lost to the root/void cut).

ii.
```python
NON_GREY = ('root', 'void')
MIN_UNITS_PER_SESSION = 5

def good_grey_units(clusters):
    from iblatlas.regions import BrainRegions
    beryl = np.asarray(BrainRegions().acronym2acronym(
        clusters['acronym'].to_numpy(), mapping='Beryl'), dtype=object)
    keep = (clusters['label'].to_numpy() >= 1) & ~np.isin(beryl, NON_GREY)
    return np.where(keep)[0], beryl
```
```python
    unit_idx, beryl = good_grey_units(clusters)
    if len(unit_idx) < MIN_UNITS_PER_SESSION:
        raise RuntimeError(f'only {len(unit_idx)} well-isolated grey-matter units '
                           f'(minimum {MIN_UNITS_PER_SESSION})')
```

iii. This is a documented, deliberate deviation from the reference *caching script*, which passes
`qc=None` and keeps all clusters: "Use `label >= 1`. It reproduces the paper number exactly, is
the documented analysis standard, and keeps the converted dataset tractable (all clusters would be
roughly 8x larger)." The grey-matter cut is justified by the data paper's "Final analyses were
additionally restricted to regions that were designated grey matter ... contained at least five
well-isolated neurons per session", and by the reference repo's own region selection
(`data_loader_utils.py:252` excludes `root` and `void`). `MIN_UNITS_PER_SESSION = 5` was added in
Step 10 after the first full run produced 38 "all neural data is zero" warnings from 1-3 neuron
sessions; it is the paper's five-neuron rule applied at session rather than region level, and it
cut the warnings to 16/187,934 trials, which the AI then showed were genuine silent stretches
(session 43 has 128 inter-spike gaps > 2 s and every zero window falls inside one).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams share one session clock, so alignment is subtraction of the trial's
`stimOn_times`. Each trial window is `[stimOn - 0.5, stimOn + 1.5)` s; spike times are sliced
with `searchsorted` on that interval and the bin index is `floor((t - (stimOn - 0.5)) / 0.02)`, so
bin *i* spans `[-0.5 + i*0.02, -0.5 + (i+1)*0.02)` and a spike exactly at onset falls in bin 25.
Trials with non-finite `stimOn_times` are dropped.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
    align = trials[ALIGN_TIME].to_numpy()[keep]
...
    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    i0 = np.searchsorted(st, begs, side='left')
    i1 = np.searchsorted(st, ends, side='left')
    ...
        tb = ((st[s] - begs[k]) / BINSIZE).astype(np.int64)
```

iii. Mandated by the Decoder Task ("Temporally align based on stimulus onset") and identical to the
reference `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`. The AI verified the bin
convention with synthetic spikes: a spike at exactly t = 0 bins to 25, one 1 ms earlier to 24; it
also checked alignment physiologically (single-bin choice decoding is at chance for t < 0 and rises
from ~+100 ms, peaking at 0.74 around +220-320 ms).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per 2 s trial, for every trial of every session; `time_bin_size` is
recorded as 20.0 ms and `n_timepoints` as 100. No rebinning or resampling of the neural data: the
spikes are binned once, directly at 20 ms. (The behavioural traces are interpolated onto that same
100-point grid; see 7-b/8-b.)

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
...
            'time_bin_size': BINSIZE * 1000.0,
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'n_timepoints': N_BINS,
```

iii. The reference code's `params = {'binsize': 0.02, ...}` and `n_bins = ceil(interval_len /
binsize)`; the method paper states "split into 2-s trials, each divided into 20-ms bins, producing
T = 100 time steps". The notes record the one nuance and how it was resolved: the method paper text
also mentions 50 ms bins and first-movement alignment for some analyses, but "Follow the code
(stimOn, (-0.5, 1.5), 20 ms, T = 100). It matches the 2-s trial / 20-ms bin / T = 100 statement and
the stimulus-onset alignment mandated by the Decoder Task."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Not from a raw variable as such: it is the trial's own time axis, defined by the alignment event
`trials.stimOn_times` and the window/bin constants (-0.5 to 1.5 s, 20 ms). One 100-value vector
serves every trial of every session. The AI emits **two** inputs for this quantity: `input[0]`
`time_from_stim_onset`, the continuous time value of each bin, and `input[1]` `stim_onset`, a
binary series with a single 1 in the bin containing t = 0. `input[2]` is `trial_in_block`.

ii.
```python
INPUT_NAMES = ['time_from_stim_onset', 'stim_onset', 'trial_in_block']
...
    tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
    onset = np.zeros(N_BINS, dtype=np.float32)
    onset_bin = int(round(-TIME_WINDOW[0] / BINSIZE))
    onset[onset_bin] = 1.0
    inputs[:, 0, :] = tgrid.astype(np.float32)
    inputs[:, 1, :] = onset
```

iii. The continuous ramp is the Decoder Task's "Time since stimulus onset, continuous,
time-varying"; the binary indicator is included because the Target Format says "If an input is a
time such as onset of some stimulus, represent it as a binary time series" and because "it makes
the alignment explicit to the decoder" (Step 5).

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid. The grid chosen is the **bin end times**,
`np.linspace(-0.5 + 0.02, 1.5, 100)` = -0.48 ... 1.50, which is exactly the grid the reference
`get_behavior_per_interval` interpolates behaviour onto (`linspace(interval_beg + binsize,
interval_end, n_bins)`). The binary onset row is placed at bin `round(0.5/0.02) = 25`, the first
bin whose spike-count interval contains t >= 0.

ii.
```python
    # bin end times, matching the behaviour grid; bin i covers (t_i - binsize, t_i]
    tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
    # bin_spikes uses tb = floor((t - beg)/binsize), so bin i spans
    # [off_start + i*binsize, off_start + (i+1)*binsize) and a spike exactly at stimulus
    # onset lands in bin round(-off_start/binsize) (= 25 here, the first bin at t >= 0).
    onset = np.zeros(N_BINS, dtype=np.float32)
    onset_bin = int(round(-TIME_WINDOW[0] / BINSIZE))
    onset[onset_bin] = 1.0
```

iii. The time grid is chosen to be the reference behaviour grid so that the time input and the
behavioural outputs are sampled at identical instants (Step 10, Check 3: "behaviour resampling ...
identical"). The onset-bin placement was an explicit bug fix in Step 7: the first version used
`argmin(|t_grid - binsize/2|)`, which resolved a tie to bin 24, the last pre-onset bin; it was
corrected to bin 25 and verified with synthetic spikes at t = 0 and t = -1 ms. Note one stale cell
in the Step 5 mapping table calls this input the "bin centre time ... -0.49 .. 1.5"; every other
statement (Step 5 curation, metadata `alignment_note`, Step 7/9 statistics, sanity checks) and the
code itself use the bin end times -0.48 ... 1.50.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the label of the neural bin grid itself: both are built from the same `align`
(`stimOn_times`), the same window and the same bin size, and `tgrid[i]` is the right edge of the
interval over which `neural[:, i]` counts spikes (bin *i* covers `[-0.5 + 0.02 i, -0.5 + 0.02(i+1))`
and is labelled -0.48 + 0.02 i). The behavioural outputs are interpolated at exactly the same
`tgrid`, so all three streams are indexed bin-for-bin. The half-bin (10 ms) difference between a
bin's label and its centre is inherited from the reference behaviour grid and is spelled out in the
metadata.

ii.
```python
            'alignment_note': (
                'bin i spans [off_start + i*20ms, off_start + (i+1)*20ms); the input '
                'time_from_stim_onset reports the bin end time t_i = linspace(-0.48, 1.5, 100), '
                'and behaviour is interpolated at those same t_i, as in the reference '
                'get_behavior_per_interval. The stim_onset indicator marks bin 25, the first '
                'bin containing t >= 0.'),
```
```python
        grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)   # same grid, per trial
        out[k] = interp1d(t[finite], v[finite], kind='linear',
                          fill_value='extrapolate')(grid)
```

iii. Same clock, same grid — no alignment step is needed beyond using the same constants. The
`--show-processing` plots overlay the raw aligned raster, the binned counts, the raw and binned
behaviour and the inputs on one time axis to make this visually checkable, and the independent
sanity check re-derives `linspace(-0.48, 1.5, 100)` and the bin-25 indicator from scratch.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft`, which is constant within a block, so any change of value starts a
new block. No block id exists in the trials table.

ii.
```python
    tib_all = trial_in_block(trials['probabilityLeft'].to_numpy())
```
```python
def trial_in_block(prob_left):
    pl = np.asarray(prob_left, dtype=float)
    new = np.ones(len(pl), dtype=bool)
    new[1:] = pl[1:] != pl[:-1]
```

iii. Step 5: the count is "within the current `probabilityLeft` block, including the initial
unbiased block. This is the natural reading of *trial number in block* and is what makes the prior
learnable (the mice need several trials after a switch to adapt, per Fig. 1g of the data paper)."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based position within the block, computed vectorised over the **full, unmasked** trial
sequence and only then indexed by the kept trials, so a trial that is later dropped still advances
the counter and the value is the animal's true position in the block. It is a per-trial scalar
broadcast across all 100 bins so that all inputs share one `(3, 100)` array. Observed range
[0, 98], consistent with blocks of 20-100 trials plus the 90-trial unbiased block.

ii.
```python
def trial_in_block(prob_left):
    """Index of each trial within its probabilityLeft block (0-based).

    Computed over the full trial sequence before masking, so that removing trials does not
    corrupt the count.
    """
    pl = np.asarray(prob_left, dtype=float)
    new = np.ones(len(pl), dtype=bool)
    new[1:] = pl[1:] != pl[:-1]
    idx = np.arange(len(pl))
    return idx - np.maximum.accumulate(np.where(new, idx, 0))
```
```python
    tib = tib_all[keep].astype(np.float32)
    inputs[:, 2, :] = tib[:, None]
```

iii. Step 10, Check 5: "the first block begins at index 0, and the counter is computed over the
*full* trial sequence before masking, so removing trials does not corrupt it. Range [0, 98] is
consistent with blocks of 20-100 trials plus the 90-trial unbiased block." It is independently
recomputed from the raw `probabilityLeft` column in `cache/sanity_checks.py` (PASS, `np.allclose`).

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `trials.choice`, which is +1, -1 or 0. The sign convention was verified empirically rather than
assumed: over 40 sessions, correct trials with a left stimulus all have `choice == +1` (10,801 vs
0) and correct trials with a right stimulus all have `choice == -1` (8,925 vs 0). So +1 is a
leftward report and -1 rightward; 0 (no response) trials are already removed by the trial mask.

ii.
```python
    choice = trials['choice'].to_numpy()[keep]
    y_choice = np.where(choice > 0, 0, 1).astype(np.int16)   # +1 left -> 0, -1 right -> 1
```

iii. Step 4 documents the empirical verification table and concludes "The task spec wants left = 0
and right = 1, giving the mapping +1 -> 0, -1 -> 1". Distribution in the converted data:
50.8 % left / 49.2 % right, matching the 50.6 % measured on the raw trials tables.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding above, then broadcasting the per-trial value across all 100 bins so that
`choice` is a time-varying row of the `(4, 100)` output array (the task prefers time-varying
outputs). No other processing.

ii.
```python
    outputs = np.empty((n_tr, 4, N_BINS), dtype=np.int16)
    outputs[:, 0, :] = y_choice[:, None]
    outputs[:, 1, :] = y_prior[:, None]
```

iii. Step 5: "`choice` and `prior_prob_left` are constant within a trial but are emitted as
time-varying rows so that every output shares one (4, 100) array, as the task prefers time-varying
outputs." The recoded values are compared trial-by-trial against the raw trials table in
`cache/sanity_checks.py` (PASS, exact).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. `trials.probabilityLeft`, which takes the values 0.2, 0.5, 0.8, recoded to 0, 1, 2 with
`np.isclose` (float-safe). Any trial whose value is none of the three is dropped rather than
assigned a class. The semantics were checked against the data: the measured fraction of
left-stimulus trials is 0.201 / 0.500 / 0.816 for `probabilityLeft` = 0.2 / 0.5 / 0.8.

ii.
```python
    pl = trials['probabilityLeft'].to_numpy()[keep]
    y_prior = np.select([np.isclose(pl, 0.2), np.isclose(pl, 0.5), np.isclose(pl, 0.8)],
                        [0, 1, 2], default=-1).astype(np.int16)
    if (y_prior < 0).any():
        okp = y_prior >= 0
        ...
```

iii. The mapping is given by the Decoder Task ("0.2 -> 0, 0.5 -> 1, 0.8 -> 2"); Step 4 confirms
`probabilityLeft` "is literally the probability that the stimulus appears on the left". Keeping the
0.5 unbiased block is deliberate: "it is a required class of the prior output, so
`exclude_unbiased` stays False (as in the reference)".

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Only the recoding above plus broadcasting across the 100 bins. Resulting distribution
41.7 / 14.1 / 44.2 %, against 41.2 / 15.5 / 43.2 % measured directly on the raw trials tables.

ii.
```python
    outputs[:, 1, :] = y_prior[:, None]
```
```python
OUTPUT_VALUES = [['left', 'right'],
                 ['p_left_0.2', 'p_left_0.5', 'p_left_0.8'],
                 ['low', 'medium', 'high'],
                 ['low', 'medium', 'high']]
```

iii. As above; verified exactly against the raw table in `cache/sanity_checks.py`.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position` / `_ibl_wheel.timestamps`, loaded through `SessionLoader.load_wheel()`,
which returns times/position/velocity/acceleration; wheel speed is `abs(velocity)`. This is exactly
the reference's `load_target_behavior(one, eid, 'wheel-speed')`. A session with no loadable wheel is
dropped.

ii.
```python
    try:
        sl.load_wheel()
        out['wheel_speed'] = (sl.wheel['times'].to_numpy(),
                              np.abs(sl.wheel['velocity'].to_numpy()))
    except Exception:
        out['wheel_speed'] = None
```

iii. "Matches `load_target_behavior`: wheel speed is |velocity|" (docstring). `SessionLoader`
internally interpolates the irregularly sampled wheel onto a uniform grid (measured at 1024 Hz) and
differentiates it with a low-pass filter, which the notes record as the IBL-recommended derivation
rather than a choice of the AI's.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) `SessionLoader` produces the uniformly sampled, low-pass-filtered velocity, of
which the absolute value is taken; (2) for each trial the trace is sliced to the window with
`searchsorted` and linearly interpolated (`scipy.interpolate.interp1d`, `fill_value='extrapolate'`)
onto the 100-point grid `linspace(beg + 0.02, end, 100)` — the reference
`get_behavior_per_interval` procedure, evaluated at the bin end times, i.e. sampled rather than
averaged within the bin; (3) the trial is rejected unless the trace covers the window to within one
bin (and has at least 2 finite samples). Then discretisation (7-c).

ii.
```python
def bin_behaviour(times, values, align_times):
    """Follows `get_behavior_per_interval`: evaluate a linear interpolant at
    `linspace(beg + binsize, end, n_bins)` (bin end times) and mark a trial invalid when
    the trace is missing or does not cover the window to within one bin."""
    i0 = np.searchsorted(times, begs, side='right')
    i1 = np.searchsorted(times, ends, side='left')
    for k in range(n):
        ...
        if i1[k] - i0[k] < 2:
            continue
        t = times[i0[k]:i1[k]]
        v = values[i0[k]:i1[k]]
        if abs(begs[k] - t[0]) > BINSIZE or abs(ends[k] - t[-1]) > BINSIZE:
            continue
        finite = np.isfinite(v)
        if finite.sum() < 2:
            continue
        grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
        out[k] = interp1d(t[finite], v[finite], kind='linear',
                          fill_value='extrapolate')(grid)
        ok[k] = True
```

iii. Step 10, Check 3 lists "behaviour resampling: `get_behavior_per_interval`: `interp1d` linear at
`linspace(beg+bin, end, n_bins)`, skip if trace starts late / ends early by > 1 bin | identical |
YES". The independent sanity check rebuilds wheel speed from the raw `_ibl_wheel.*` npy files with
`interpolate_position` + `velocity_filtered` and recovers the same session tertiles to ~0.1 %.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Three classes split at the **33.3rd and 66.7th percentiles of that session's own** retained
(trial, bin) samples, i.e. equal-sized classes per session, giving a 1/3 chance level. A degenerate
(near-constant) trace has its upper threshold nudged so `np.digitize` cannot emit a 4th class. The
thresholds of every session are stored in `metadata['session_info']`. Achieved distribution
0.333/0.333/0.333.

ii.
```python
def tertile_bins(values):
    """Discretise into 3 balanced classes using per-session tertiles.

    DEVIATION (required): the reference treats wheel speed and whisker ME as continuous
    regression targets, but the task requires categorical outputs. Tertiles are computed
    per session because these signals are in session-specific units (camera gain / ROI,
    wheel rig), so a global threshold would largely encode session identity. The reference
    likewise rescales behaviour per session before decoding.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.int16), (np.nan, np.nan)
    lo, hi = np.percentile(finite, [100 / 3, 200 / 3])
    if not (hi > lo):  # degenerate (e.g. a near-constant trace)
        hi = lo + 1e-12
    return (np.digitize(values, [lo, hi]).astype(np.int16), (float(lo), float(hi)))
```
```python
    y_wheel, thr_w = tertile_bins(wheel)
    y_whisk, thr_m = tertile_bins(whisk)
```

iii. Step 5: balanced classes "give an interpretable 1/3 chance level and avoid a degenerate
majority class"; per session because "wheel gain varies by rig" and whisker ME is "in raw camera
units whose scale depends on lighting, camera distance and ROI size, so a global threshold would
mostly encode which session a trial came from rather than behaviour. The reference likewise
z-scores behaviour per session before decoding (`SingleSessionDataset` uses a per-session
`StandardScaler`)." Flagged in the code and notes as a task-required deviation, since the reference
regresses these variables continuously.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is evaluated at the same per-trial grid as everything else — `linspace(stimOn - 0.48,
stimOn + 1.5, 100)` — i.e. at the right edge of each of the 100 spike-count bins, on the same
session clock, so the wheel row and the neural matrix are indexed bin-for-bin. No lag or shift is
applied.

ii.
```python
    wt, wv = beh['wheel_speed']
    wheel, ok_w = bin_behaviour(wt, wv, align)      # align = stimOn_times of kept trials
...
        grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
```
```python
    outputs[:, 2, :] = y_wheel
```

iii. Same clock and same grid as the neural data, so no further alignment is required; the
`--show-processing` panels plot the raw `|velocity|` trace, the interpolated 100-point trace and the
resulting class steps on one stimulus-onset-aligned axis, and the notes record that "the binned
traces sit on top of the raw traces and the discretised steps change exactly where the binned trace
crosses a threshold."

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with its frame times `_ibl_<side>Camera.times`, loaded through
`SessionLoader.load_motion_energy(views=[view])` and read from the `whiskerMotionEnergy` column.
The **left** camera is tried first and the **right** used as a fallback, and the loaded trace must
contain at least one finite value. Availability was surveyed up front: left in 437/459 sessions,
right in 420/459, neither in 14 — those 14 sessions are dropped because whisker ME is a required
output. Final usage: 434 left, 7 right.

ii.
```python
    me = None
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sl.load_motion_energy(views=[view])
            df = sl.motion_energy[cam]
            vals = df['whiskerMotionEnergy'].to_numpy()
            if np.isfinite(vals).any():
                me = (df['times'].to_numpy(), vals, view)
                break
        except Exception:
            continue
    out['whisker_motion_energy'] = me
```
```python
    if beh['whisker_motion_energy'] is None:
        raise RuntimeError('no whisker motion energy (left or right)')
```

iii. "whisker ME prefers the left camera and falls back to the right (`bin_behaviors`)" — the
reference `bin_behaviors` does exactly this for `'whisker-motion-energy'`. Step 4: "Keep the
left-then-right preference. The 14 sessions with no whisker ME must be dropped because it is a
required output." The chosen camera is recorded per session in `metadata['session_info']`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released ~60 Hz trace is used as-is, with no filtering or normalisation, and goes through the
identical `bin_behaviour` path as the wheel: window slice, linear interpolation onto
`linspace(beg + 0.02, end, 100)`, coverage rejection to within one bin, then per-session tertiles.
NaN samples are excluded from the interpolant rather than propagated (a trial with fewer than 2
finite samples is rejected).

ii.
```python
    mt, mv, me_side = beh['whisker_motion_energy']
    whisk, ok_m = bin_behaviour(mt, mv, align)
...
        finite = np.isfinite(v)
        if finite.sum() < 2:
            continue
        out[k] = interp1d(t[finite], v[finite], kind='linear',
                          fill_value='extrapolate')(grid)
```

iii. Same justification as the wheel — it is the reference `get_behavior_per_interval` procedure.
The notes describe the released quantity ("mean across pixels of the absolute difference between
adjacent frames in a nose-tip-to-eye bounding box") and the independent sanity check rebuilds it
from the raw `<side>Camera.ROIMotionEnergy.npy` + camera times, recovering the same tertiles to
~0.1 %.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: `tertile_bins` on all retained (trial, bin) samples of the session, so
three equal-sized classes per session, thresholds stored in the metadata. Achieved distribution
0.333/0.334/0.333.

ii.
```python
    y_whisk, thr_m = tertile_bins(whisk)
    outputs[:, 3, :] = y_whisk
```
```python
            'discretisation': ('wheel_speed and whisker_motion_energy split at per-session '
                               'tertiles of all retained (trial, bin) samples'),
```

iii. As 7-c: balanced 3-way classes, per session because the ME scale is camera/ROI/lighting
specific, so a global threshold would encode session identity rather than behaviour.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The camera frame times are on the same session clock as the spikes, and the trace is evaluated at
the same 100 grid points measured from the same `stimOn_times`, so it is bin-for-bin aligned with
the neural matrix. The only camera-specific consequence is the lower sampling rate (60 Hz, about one
frame per 20 ms bin), which the linear interpolation handles.

ii.
```python
    whisk, ok_m = bin_behaviour(mt, mv, align)
...
        grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
    outputs[:, 3, :] = y_whisk
```

iii. No alignment step needed beyond using the common clock and grid; verified in the
`--show-processing` plots (raw ME, interpolated ME and discretised classes on one onset-aligned
axis) and in the raw-file sanity checks.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable data is dropped at the appropriate granularity and every drop is logged and
reconciled:
- **Probe**: a probe whose spike sorting has no `times` is skipped; a session with no usable probe
  raises and is skipped.
- **Session**: no whisker ME on either camera (14 sessions), fewer than 5 well-isolated
  grey-matter units (3), no trial with complete behaviour coverage (1) -> 441/459 kept. Any
  exception in a worker is caught, printed with its reason, and the session skipped rather than
  aborting the run.
- **Trial**: NaN in any of the six reference trial events, non-finite `stimOn_times`, wheel or
  camera trace not spanning the window to within one bin, `probabilityLeft` not one of the three
  block values.
- **Samples**: NaNs inside a behaviour trace are excluded from the interpolant (rather than
  propagating NaN as the reference's `allow_nans=True` path would); a trial left with < 2 finite
  samples is dropped; a degenerate constant trace cannot produce an out-of-range class.
- **Stale index**: the rebuilt ONE index leaves `hash` null because the shipped md5s describe
  superseded file versions and would raise spurious mismatch warnings; `file_size` is still checked.
- **All-zero trials**: 16/187,934 trials have no spikes at all. These are kept, after the AI
  verified from the raw spike train that they fall inside genuine > 2 s silent stretches of sparse
  units rather than being a binning bug.

ii.
```python
        if len(spikes) == 0 or 'times' not in spikes:
            continue
    if not tables:
        raise RuntimeError('no spike data')
```
```python
    if beh['whisker_motion_energy'] is None:
        raise RuntimeError('no whisker motion energy (left or right)')
    if beh['wheel_speed'] is None:
        raise RuntimeError('no wheel data')
    ...
    if valid.sum() < 2:
        raise RuntimeError(f'only {valid.sum()} trials with complete behaviour')
```
```python
def _worker(args):
    eid, probes, show, outdir = args
    try:
        return convert_session(eid, probes, show=show, outdir=outdir)
    except Exception as exc:
        return {'eid': eid, 'error': f'{type(exc).__name__}: {exc}'}
```

iii. Step 9/10: the session and trial accounting is fully reconciled (459 - 14 - 3 - 1 = 441
sessions; 285,031 raw -> 188,020 after the reference mask -> 187,934 after coverage), and each
dropped session is itemised with its reason in `conversion_full_out.txt`. The all-zero trials
"cannot be 'fixed' because they are a true property of the recordings ... Discarding them would
remove genuine data - silence is a real neural observation".

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting off disk, by a wide margin. The script prints cumulative per-stage
timings: `load_spikes` 2330 s, `bin_spikes` 665 s, `load_behaviour` 429 s, `bin_behaviour` 37 s,
`load_trials` 17 s, summed over 441 sessions — so spike I/O is ~70 % of the CPU time and binning
~20 %. With 24 worker processes the wall clock is 2.8-3.0 min for the full dataset, well inside the
15-minute budget, and 11.35 GB is written at the end.

ii.
```python
    t = time.time()
    spike_times, spike_clusters, clusters = load_session_spikes(one, eid, probes)
    timings['load_spikes'] = time.time() - t
...
    agg = {}
    for r in results:
        for k, v in r['timings'].items():
            agg[k] = agg.get(k, 0.0) + v
    print('cumulative stage timings (s):',
          {k: round(v, 1) for k, v in sorted(agg.items(), key=lambda x: -x[1])})
```

iii. Step 7 measured the per-session profile before the full run ("load spikes 2.4 s - dominant
cost, scales with probe count") and used it to project the full run at 2-4 min wall clock; Step 9
confirmed 2.8 min. Cost is file I/O, which is why the remedy chosen was parallelism across
sessions rather than faster arithmetic.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two `for` loops remain, both over trials:
- `bin_spikes`: one `np.bincount` per trial. This could be a single `bincount` over the whole
  session by offsetting each spike's flat index by `trial * n_units * N_BINS` (spikes would have to
  be duplicated only if windows overlapped, which they do not by construction).
- `bin_behaviour`: one `interp1d` call per trial. This could be one `np.interp` over a single
  concatenated query vector, since the trace is monotonic in time.
- `plot_processing` also builds a Python dict comprehension remap and a list comprehension over
  spikes, but it only runs for 2 sessions in `--show-processing` mode.

The loops the reference *did* have were removed: the reference spawns a multiprocessing pool per
session and per behaviour and calls `bincount2D` once per interval; this script instead sorts the
spike train once, slices with `searchsorted`, and parallelises across sessions.

ii.
```python
    for k in range(len(align_times)):
        if i1[k] <= i0[k]:
            continue
        s = slice(i0[k], i1[k])
        tb = ((st[s] - begs[k]) / BINSIZE).astype(np.int64)
        np.clip(tb, 0, N_BINS - 1, out=tb)
        flat = np.bincount(sc[s] * N_BINS + tb, minlength=n_units * N_BINS)
        out[k] = flat.reshape(n_units, N_BINS)
```
```python
    for k in range(n):
        ...
        grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
        out[k] = interp1d(t[finite], v[finite], kind='linear',
                          fill_value='extrapolate')(grid)
```

iii. The AI documented the vectorisation it did do ("vectorised `searchsorted` + `np.bincount`
binning instead of a pool per interval - binning is 0.7 s per 2 sessions") and judged the result
fast enough: the full conversion runs in ~3 min against a 15-min budget, so it did not pursue the
remaining per-trial loops. Behaviour binning (37 s cumulative) is negligible; spike binning
(665 s cumulative, ~1.5 s/session) is the only loop where full vectorisation would still buy
anything, and it is not the bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. A few small repeats remain, none of them on the hot path:
- **A ONE client is constructed per session, not per worker.** `convert_session` calls `get_one()`
  on every call, which re-reads the rebuilt `datasets.pqt`/`sessions.pqt` index (18,450 rows) each
  time. A pool initializer would have built it once per worker process. The script does avoid the
  bigger version of this problem deliberately: the *index build* is forced once in the parent so
  that 24 workers do not each regenerate it.
- **`BrainRegions()` is instantiated per session** inside `good_grey_units`, reloading the Allen/
  Beryl tables for each of the 441 sessions.
- **The trial-window index search is done three times per session** (spikes, wheel, camera), once
  per stream — unavoidable, since the streams have different time bases.
- In `--show-processing` mode, `plot_processing` re-selects and re-maps the raw spikes that
  `bin_spikes` already processed.

ii.
```python
def convert_session(eid, probes, show=False, outdir='/app'):
    t0 = time.time()
    one = get_one()          # once per session, not once per worker
```
```python
def good_grey_units(clusters):
    from iblatlas.regions import BrainRegions
    beryl = np.asarray(BrainRegions().acronym2acronym(
        clusters['acronym'].to_numpy(), mapping='Beryl'), dtype=object)
```
```python
    # Build the rebuilt ONE index once in the parent; otherwise every worker process
    # discovers it missing and regenerates it redundantly.
    get_one()
```

iii. The AI documented the repeats it removed relative to the reference (pool-per-session-per-
behaviour, per-interval `bincount2D`, the `raw_electrophysiology` network call, "Behaviour traces
are loaded once per session and sliced with `searchsorted`") and the one it explicitly guarded
against (index rebuild in the parent). It did not flag the per-session ONE/`BrainRegions`
construction; at ~3 min total wall clock these costs are small next to the 2330 s of spike I/O.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three items:
- **Unused spike arrays are read from disk.** `ssl.load_spike_sorting()` is called with the default
  `brainbox.io.one.SPIKES_ATTRIBUTES = ['clusters', 'times', 'amps', 'depths']`, so `spikes.amps`
  and `spikes.depths` — two arrays as large as the two that are used — are loaded and immediately
  discarded. Since spike loading is the dominant cost (2330 s of 3477 s), restricting
  `SPIKES_ATTRIBUTES` to `['clusters', 'times']` would have roughly halved it.
- **The full merged cluster table is built for every probe** (`merge_clusters(...).to_df()`, 34
  columns of metrics plus histology) when only `label` and `acronym` are used; the Beryl mapping is
  also computed for *all* clusters, including the ~88 % that fail QC, before the `label >= 1` cut.
- **Redundant/constant rows are materialised per trial.** `input[1]` (the onset indicator) is a
  constant vector, identical for all 187,934 trials and fully determined by `input[0]`, yet is
  stored 187,934 times; the same is true of the time ramp, and `choice`/`prior` are broadcast to
  100 identical values each. Storing the neural counts as float32 rather than a small integer type
  likewise triples the file size (11.35 GB). These are consequences of the required
  `(d, n_timepoints)`-per-trial format and of a verifier warning about non-float32 neural arrays,
  so they are only partly avoidable.

ii.
```python
        ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
        spikes, clusters, channels = ssl.load_spike_sorting()      # also loads amps, depths
        ...
        df = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```
```python
    beryl = np.asarray(BrainRegions().acronym2acronym(
        clusters['acronym'].to_numpy(), mapping='Beryl'), dtype=object)   # all clusters
    keep = (clusters['label'].to_numpy() >= 1) & ~np.isin(beryl, NON_GREY)
```
```python
    onset = np.zeros(N_BINS, dtype=np.float32)
    onset[onset_bin] = 1.0
    inputs[:, 1, :] = onset          # same vector for every trial of every session
```

iii. The AI did document and remove one piece of useless work in the reference
(`raw_electrophysiology(..., stream=True).fs`, "purely to record a sampling frequency it never
uses"), and it reconsidered the neural dtype twice (int16 -> float32 after the verifier warned).
It did not identify the unused `amps`/`depths` arrays or the whole-table Beryl mapping, judging the
pipeline fast enough once it measured ~3 min wall clock against the 15-minute budget.
