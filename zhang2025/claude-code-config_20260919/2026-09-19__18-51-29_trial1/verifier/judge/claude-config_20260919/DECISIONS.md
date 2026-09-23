# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Nothing is opened by path. The AI takes its session list from the reference repository's own
release freeze, `/app/code/code_zhang2025/data/bwm_release.csv`, which has one row per probe
insertion and carries the columns `eid`, `pid`, `probe_name`, `subject` and `lab`. De-duplicating
the `eid` column in freeze order gives the 459 released sessions; grouping the rows of one `eid`
gives that session's probe insertions. Every subsequent read goes through the IBL ONE API against
the local cache: `SpikeSortingLoader(pid=..., eid=..., pname=...).load_spike_sorting()` +
`merge_clusters()` per probe, the reference's `merge_probes()` to pool them, the reference's
`load_trials_and_mask(one, eid, max_trial_len=10.0, sess_loader=...)` for the trials table, and
`SessionLoader.load_wheel()` / `load_motion_energy(views=[...])` for the behaviour. This is the same
entry point and the same loader chain as `0_data_caching.py` → `prepare_data()`. Sessions are
dispatched to a `ProcessPoolExecutor` (16–24 workers), each worker building its own ONE handle
(`get_one()`) because a connection cannot be shared across a fork.

ii.
```python
bwm = pd.read_csv(BWM_FREEZE, index_col=0)
eids = list(dict.fromkeys(bwm.eid.tolist()))       # preserve freeze order
...
for i, eid in enumerate(eids):
    sub = bwm[bwm.eid == eid]
    jobs.append((eid, sub.pid.tolist(), sub.probe_name.tolist(),
                 sub.subject.iloc[0], sub.lab.iloc[0], show, out_dir))
```

```python
def get_one():
    """Process-local ONE handle (served entirely from the local cache)."""
    global _ONE
    if _ONE is None:
        from one.api import ONE
        _ONE = ONE(base_url=ALYX_URL, silent=True)
    return _ONE
```

```python
for pid, pname in zip(pids, probe_names):
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    sp, cl, ch = ssl.load_spike_sorting()
    ...
    cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
...
spikes, clusters = merge_probes(spikes_list, clusters_list)
```

```python
sess_loader = SessionLoader(one=one, eid=eid)
trials, ref_mask = load_trials_and_mask(one=one, eid=eid,
                                        max_trial_len=MAX_TRIAL_LEN,
                                        sess_loader=sess_loader)
```

iii. From CONVERSION_NOTES Step 6: "The reference module is *imported* (`sys.path` →
`code/code_zhang2025/src`) for `merge_probes` and `load_trials_and_mask`, so trial curation is
literally the reference implementation rather than a copy." Step 10 Check 3(a) records the loading
chain as "identical" to `prepare_data`, with the single documented omission of the reference's
`raw_electrophysiology(...).fs` call, which streams raw binary and "feeds no data value".

## 1-b. How are the data split into subjects?

i. The subject name is read straight off the freeze file (`bwm_release.csv.subject`) for each `eid`,
so no path or filename parsing is involved. After conversion the results are sorted by
`(subject, eid)`, `subjects` is the sorted set of names, and `subject_idx[s]` is the index of that
session's subject. 135 of the 139 released mice survive (the other 4 contributed only sessions that
were skipped for the documented reasons).

ii.
```python
results.sort(key=lambda r: (r['subject'], r['eid']))

subjects = sorted({r['subject'] for r in results})
subj_lookup = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subj_lookup[r['subject']] for r in results], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 maps "`bwm_release.csv.subject` → `subjects`, `subject_idx`, unique
sorted list + index per session". The freeze already carries a unique subject id per session, so
nothing has to be derived. Step 9 uses "135 subjects vs 139 released" as an explicit consistency
check and attributes the shortfall to the 19 skipped sessions.

## 1-c. How are the data split into sessions?

i. A session *is* an `eid`, and the freeze lists one row per (session, probe). De-duplicating the
`eid` column yields 459 sessions; each is processed independently by one worker and emitted as one
element of `neural` / `input` / `output` / `brain_region_idx`. Both probes of a two-probe session are
merged into that one session rather than becoming two sessions.

ii.
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))       # preserve freeze order
```
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
```

iii. CONVERSION_NOTES Step 1 records `merge_probes`'s own rationale — probes from the same session
are "not statistically independent as they have the same underlying behaviour" — and Step 9 checks
"670 probes in the 440 kept sessions" against the 699 released insertions.

## 1-d. How are the data split into trials?

i. The trials table returned by the reference's `load_trials_and_mask` has one row per trial, so the
split is given by the data. Each retained row becomes one 2-s interval beginning at
`stimOn_times - 0.5`, and every stream (spikes, wheel, whisker, task variables) is cut on that same
interval.

ii.
```python
align_times = trials[PARAMS['align_time']].to_numpy(dtype=float)
mask &= np.isfinite(align_times)
interval_begs = align_times + WIN[0]
```

iii. Step 5 decision 1: "Alignment / window / bin size = `stimOn_times`, (−0.5, +1.5) s, 20 ms →
T = 100. Exactly the reference `params`." The trials table is already one row per trial, so there is
no decision beyond choosing the alignment event and window.

## 1-e. How are trials filtered based on quality controls?

i. Six filters, ANDed into one mask.
 1. The reference's own mask, obtained by *calling the reference function*:
    `load_trials_and_mask(one, eid, max_trial_len=10.0)` — reaction time
    (`firstMovement_times − stimOn_times`) in [0.08, 2.0] s, `feedback_times − goCue_times ≤ 10 s`,
    no NaN in `stimOn_times / choice / feedback_times / probabilityLeft / firstMovement_times /
    feedbackType`, and `choice != 0` (`exclude_nochoice=True`). The unbiased block is kept
    (`exclude_unbiased=False`), which is needed for the 3-class prior output.
 2. `probabilityLeft` must be one of {0.2, 0.5, 0.8}.
 3. `choice` must be ±1.
 4. `stimOn_times` must be finite.
 5. The 2-s window must lie inside the interval during which *every* probe of the session was
    producing spikes (otherwise "zero spikes" would mean "not recorded").
 6. The wheel trace and the whisker trace must each cover the window with the reference's own
    tolerance (first sample ≤ 1 bin late, last sample ≤ 1 bin early, no NaN after resampling).
 Result: 187,513 of 284,440 raw trials (65.9 %) retained; sessions left with < 2 usable trials are
 dropped.

ii.
```python
trials, ref_mask = load_trials_and_mask(one=one, eid=eid,
                                        max_trial_len=MAX_TRIAL_LEN,
                                        sess_loader=sess_loader)
mask = ref_mask.to_numpy().astype(bool)

prior_code = map_prior(trials['probabilityLeft'].to_numpy())
mask &= prior_code >= 0
choice_raw = trials['choice'].to_numpy(dtype=float)
mask &= np.isin(choice_raw, (-1.0, 1.0))

align_times = trials[PARAMS['align_time']].to_numpy(dtype=float)
mask &= np.isfinite(align_times)
interval_begs = align_times + WIN[0]

# The trial window has to be covered by the ephys recording of every probe;
# otherwise the "spike counts" would be zeros that mean "not recorded".
with np.errstate(invalid='ignore'):
    mask &= (interval_begs >= rec_span[0]) & (interval_begs + BINSIZE * NBINS <= rec_span[1])
...
mask &= wheel_ok & me_ok
n_trials_kept = int(mask.sum())
if n_trials_kept < MIN_TRIALS:
    raise RuntimeError(f'only {n_trials_kept} usable trials')
```

iii. Step 5 decision 3: "Trial QC = `load_trials_and_mask(..., max_trial_len=10.0)`, i.e. the
reference call verbatim". Step 3 quotes the data paper's matching text ("trials were excluded if one
of the following trial events could not be detected … time between stimulus onset and the first
movement … outside the range of 0.08–2.00 s"). The ephys-coverage rule was added in Step 10 Check 1
after `verify_data_format` warned about all-zero trials: session `8c2f7f4d…` "had its last three
trials at t ≈ 1789 s while the spike sorting ended at t = 1779 s: the behaviour outlasted the
recording, so 'zero spikes' meant 'not recorded'." Step 10 Check 3 labels the combined mask a
"superset" of the reference's, noting the extra rule "removes 3 trials in 187,513".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` of every probe insertion of the session (loaded by
`SpikeSortingLoader.load_spike_sorting`). The cluster table produced by `merge_clusters` supplies
two things used only for selection, not for the array itself: `clusters.label` (the RIGOR quality
score) and `clusters.acronym` (the Allen acronym, mapped to Beryl). `clusters.uuids` is kept in the
metadata so neurons can be traced back to the raw files.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
...
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
```
```python
su = remap[spikes['clusters']]
sel = su >= 0
binned = bin_spikes(np.ascontiguousarray(spikes['times'][sel]),
                    np.ascontiguousarray(su[sel]),
                    len(keep_idx),
                    interval_begs[mask])
```

iii. Step 5's mapping table: "`spikes.times`, `spikes.clusters` (all probes merged) → `neural[s][k]`
(n_neurons, 100) float32". Step 10 Check 3(a) records this as identical to `prepare_data`, which
builds `neural_dict` from exactly `spikes['times']`, `spikes['clusters']` and `clusters['acronym']`.

## 2-b. How is the `neural` data processed?

i. Probes are merged into one population with the reference's `merge_probes` (single-probe sessions
bypass it and just reset the cluster index, because `merge_probes` mutates `spikes['clusters']` in
place). Clusters are filtered (see 2-c) and renumbered through a lookup array. Spikes are then
counted into 100 non-overlapping 20 ms bins per trial, half-open `[t_beg, t_end)`, using a single
`np.bincount` over the flat index `unit * NBINS + bin`. The stored value is the **raw spike count**
per bin as `float32` — not a firing rate and not normalised or smoothed. Mean rate over the dataset
is 11.80 Hz.

ii.
```python
def bin_spikes(spike_times, spike_units, n_units, interval_begs):
    n_trials = len(interval_begs)
    out = np.zeros((n_trials, n_units, NBINS), dtype=np.float32)
    ...
    i0 = np.searchsorted(spike_times, safe, side='left')
    i1 = np.searchsorted(spike_times, safe + BINSIZE * NBINS, side='left')
    for k in range(n_trials):
        if not np.isfinite(begs[k]) or i1[k] <= i0[k]:
            continue
        tt = spike_times[i0[k]:i1[k]]
        uu = spike_units[i0[k]:i1[k]]
        b = ((tt - begs[k]) / BINSIZE).astype(np.int64)
        np.clip(b, 0, NBINS - 1, out=b)
        counts = np.bincount(uu * NBINS + b, minlength=n_units * NBINS)
        out[k] = counts.reshape(n_units, NBINS)
    return out
```

```python
remap = np.full(int(np.max(spikes['clusters'])) + 2, -1, dtype=np.int64)
remap[keep_idx] = np.arange(len(keep_idx))
```

iii. Step 5 decision 7: "Neural data are raw spike counts (float32), unnormalised, as cached by the
reference." Step 10 Check 3(d) compares the binning logic directly: the reference uses
`bincount2D(xbin=0.02, xlim=[t_beg,t_end])` truncated to `ceil(2/0.02)=100` bins on spikes with
`t_beg <= t < t_end`; the script uses `np.bincount` on `floor((t−t_beg)/0.02)` over the same
selection, "verified against the raw files in Check 2" (`max|diff| = 0.0` for 5 sessions × 3 trials,
full (n_neurons, 100) matrices re-derived from `spikes.times.npy` / `spikes.clusters.npy`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two cluster-level filters and one session-level one.
 * `clusters.label >= 1` — the IBL label is the fraction of the three RIGOR single-unit metrics
   (amplitude > 50 µV, noise cut-off < 20 µV, refractory-period violation) that a cluster passes, so
   `label >= 1` is exactly the data paper's "well-isolated neuron". This yields 108.8 good units per
   probe, against the paper's stated 108.
 * The cluster's Beryl acronym must not be `root` or `void`, i.e. grey matter only. This is the one
   place the AI departs from the human reference, which keeps `root`.
 * A session needs at least 5 such neurons (`MIN_NEURONS = 5`) or it is skipped.
 62,650 neurons across 440 sessions survive (86 % of the 72,876 label-good units in those sessions).

ii.
```python
NON_GREY = ('root', 'void')     # Beryl acronyms that are not grey matter
QC_LABEL = 1.0                  # BWM "well-isolated neuron" == clusters.label >= 1
MIN_NEURONS = 5                 # BWM: >= 5 well-isolated neurons per session
```
```python
def select_units(clusters):
    br = get_brain_regions()
    beryl = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
    label = clusters['label'].to_numpy(dtype=float)
    keep = (label >= QC_LABEL) & ~np.isin(beryl, NON_GREY)
    return np.nonzero(keep)[0], beryl
```
```python
if len(keep_idx) < MIN_NEURONS:
    raise RuntimeError(f'only {len(keep_idx)} well-isolated grey-matter units')
```

iii. This is the AI's single largest documented deviation from the reference *code*, and it is
argued at length in Step 4: `prepare_data` calls `load_spiking_data(qc=None)`, i.e. all ~889
clusters per probe, whereas the data paper analyses only the 75,708 well-isolated neurons. The AI
resolves in favour of the paper — "(a) it is the explicit inclusion criterion of the paper that
produced the data; (b) ≈108 good units/probe reproduces the paper's headline number, while 'all
clusters' would be 889/probe; (c) keeping all 621 k units would make the converted dataset ≈110 GB
and put mostly-noise clusters into the decoder." The grey-matter cut quotes the same paper: "Final
analyses were additionally restricted to regions that were designated grey matter in the … Allen
Common Coordinate framework." The ≥5-neuron rule was added in Step 10 Check 1 after three sessions
with 1–3 neurons produced many all-zero trials; the AI notes it is "the same threshold the BWM paper
applies to a region within a session … applied here to the pooled population because the decoder
uses the whole session at once." Step 5 decision 2 explicitly declines the paper's further
"region recorded in ≥2 sessions" rule as irrelevant to a whole-session decoder.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams already live on one synchronised session clock, so alignment is arithmetic: each
trial's interval begins at `stimOn_times + (−0.5)` and the bin index of a spike is
`floor((t − interval_beg) / 0.02)`, which puts stimulus onset at the boundary between bin 24 and
bin 25. The window is half-open `[stimOn−0.5, stimOn+1.5)`; the bin index is clipped to [0, 99] only
to guard against floating-point drift at the edge. Trials whose window is not fully inside the ephys
recording span of every probe are dropped (1-e).

ii.
```python
align_times = trials[PARAMS['align_time']].to_numpy(dtype=float)
interval_begs = align_times + WIN[0]
```
```python
b = ((tt - begs[k]) / BINSIZE).astype(np.int64)
np.clip(b, 0, NBINS - 1, out=b)
```
```python
'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
'off_start': float(WIN[0]),
'off_end': float(WIN[1]),
```

iii. Step 5 decision 1 ties the alignment to both the reference `params`
(`'align_time': 'stimOn_times'`, `'time_window': (-.5, 1.5)`) and the Decoder Task's "Temporally
align based on stimulus onset", and Step 3 quotes the methods paper: "For choice, we align trials to
the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset." Step 10
Check 5 spells out the half-open convention: "bin *i* covers `[−0.5+0.02i, −0.48+0.02i)` and no
spike is counted twice; a spike exactly at the window end is excluded, matching `times < t_end` in
the reference." Step 12 verifies the alignment empirically with an independent per-timepoint
logistic regression: choice decodability is flat at 0.52 for the whole pre-stimulus half-second and
"rises abruptly within one or two bins of t = 0", peaking at 0.786 at +0.20 s.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms, 100 bins spanning the 2-s window, for every trial of every session. No rebinning,
resampling, smoothing or overlap: spikes are binned once, directly at 20 ms, at their native
resolution. `metadata['time_bin_size']` is recorded as 20.0 (ms) and `n_time_bins` as 100.

ii.
```python
PARAMS = {
    'interval_len': 2,
    'binsize': 0.02,
    'single_region': False,
    'align_time': 'stimOn_times',
    'time_window': (-0.5, 1.5),
}
BINSIZE = PARAMS['binsize']
WIN = PARAMS['time_window']
NBINS = int(np.ceil((WIN[1] - WIN[0]) / BINSIZE))          # 100
BIN_LEFT_EDGES = WIN[0] + BINSIZE * np.arange(NBINS)        # -0.50 ... 1.48
BIN_RIGHT_EDGES = BIN_LEFT_EDGES + BINSIZE                  # -0.48 ... 1.50
```
```python
'time_bin_size': BINSIZE * 1000.0,           # ms
'n_time_bins': NBINS,
```

iii. Step 5 decision 1 and the Step 4 discrepancy table: the reference code sets `binsize: 0.02`
everywhere, and Step 3 quotes the methods paper — "Recordings are split into 2-s trials, each
divided into 20-ms bins, producing T = 100 time steps". The AI notes that the STAR Methods also
mention a 50 ms variant for choice and a `firstMovement_times` alignment for the dynamic behaviours,
and rejects both because "the Decoder Task fixes the alignment to stimulus onset and requires
per-trial *and* time-varying outputs in one dataset, so the unified scheme is the only consistent
choice."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Not from a raw variable at all — it is the binning grid itself, defined by the alignment event
`trials.stimOn_times` and the constants `time_window = (−0.5, 1.5)` and `binsize = 0.02`. Because
every trial uses the same grid, the same 100-vector is broadcast to every trial of every session.

ii.
```python
BIN_LEFT_EDGES = WIN[0] + BINSIZE * np.arange(NBINS)        # -0.50 ... 1.48
```
```python
inputs = np.empty((n_kept, len(INPUT_NAMES), NBINS), dtype=np.float32)
inputs[:, 0, :] = BIN_LEFT_EDGES[None, :]
```

iii. Step 5 maps "bin index → `input[s][k][0]` 'time_from_stim_on_s' → left edge of bin *i* =
−0.5 + 0.02·i, i = 0…99 … time-varying, seconds, range [−0.5, 1.48]", with reference code column
"—" because the reference caches no decoder inputs at all (Step 10 Check 3(e): "reference caches no
decoder inputs (its models take neural activity only) … required by the task").

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None. It is a deterministic linear ramp from −0.50 to +1.48 in 20 ms steps, stored as float32 and
identical for every trial. The AI uses the **left edge** of each bin (the reference's
`get_spike_data_per_interval` docstring likewise says its timepoints "refer to the start/left edge
of a bin"); the human reference uses the bin centre instead, a fixed 10 ms offset.

ii.
```python
BIN_LEFT_EDGES = WIN[0] + BINSIZE * np.arange(NBINS)        # -0.50 ... 1.48
BIN_RIGHT_EDGES = BIN_LEFT_EDGES + BINSIZE                  # -0.48 ... 1.50
...
inputs[:, 0, :] = BIN_LEFT_EDGES[None, :]
```
```python
'time_from_stim_on': 'seconds from stimulus onset to the left edge of the '
                     'time bin; -0.50 ... 1.48',
```

iii. Step 9's consistency table records "time_from_stim_on … [−0.50, 1.48] (bin left edges)" as
matching the reference window, and Step 10 sanity check 4 verifies that "`input[0]` equals −0.50,
−0.48, …, 1.48 for 40 sampled trials".

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural grid: element *i* of the input is the left edge of the same bin *i* that the
spikes were counted into, both measured from the same `stimOn_times`. So the alignment is exact by
construction, index for index.

ii.
```python
b = ((tt - begs[k]) / BINSIZE).astype(np.int64)     # neural bin index
```
```python
inputs[:, 0, :] = BIN_LEFT_EDGES[None, :]           # left edge of that same bin
```

iii. Step 5 decision 1; Step 10 Check 5 confirms "bin *i* covers `[−0.5+0.02i, −0.48+0.02i)`", so
`BIN_LEFT_EDGES[i]` is literally the opening edge of neural bin *i*. The `--show-processing` figure
puts the raw raster, the binned counts and the time axis on a common axis with `stimOn` at 0.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. `trials.probabilityLeft`, which the task holds constant within a block, so a change of value marks
a block boundary. The trials table carries no block identifier, so the blocks must be recovered this
way. The full, unfiltered trials table is used.

ii.
```python
block_idx = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. Step 5's mapping table: "`probabilityLeft` transitions → `input[s][k][1]` 'trial_num_in_block'".
Reference code column is "—": the reference stores `block = probabilityLeft` as a behaviour but never
derives a within-block counter, so this is a task-specific construction.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based counter that resets whenever `probabilityLeft` changes value (NaN-to-NaN counts as no
change), computed on the **complete** trials table *before* any exclusion, then broadcast constant
across the 100 bins of each retained trial. Because the count is taken before filtering, a trial
that is later dropped still advances the counter, so the number is the animal's real position in the
block. Observed range across the dataset is 0–98, consistent with the 90-trial unbiased block
followed by 20–100-trial blocks.

ii.
```python
def trial_number_in_block(probability_left):
    """0-based index of each trial within its constant-``probabilityLeft`` block.

    Computed on the complete trials table (before any trial exclusion) so that the
    value reflects the animal's actual position in the block.
    """
    p = np.asarray(probability_left, dtype=float)
    out = np.zeros(len(p), dtype=np.int64)
    count = 0
    for i in range(len(p)):
        if i > 0 and not (p[i] == p[i - 1] or (np.isnan(p[i]) and np.isnan(p[i - 1]))):
            count = 0
        out[i] = count
        count += 1
    return out
```
```python
inputs[:, 1, :] = block_idx[mask][:, None]
```

iii. Step 5: "0-based index of the trial within its constant-`probabilityLeft` block, computed on the
**full** trials table before exclusions, broadcast over the 100 bins". Step 10 Check 5 (edge cases):
"`trial_number_in_block` is computed on the *raw* trials table and starts at 0 on the first trial of
the session; the first block is the 90-trial unbiased block". Sanity check 5 recomputes it from
`probabilityLeft` in the raw trials table for 5 sessions, all trials — "pass; first block is exactly
90 trials in each".

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The single column `trials.choice`, which is +1, −1 or 0. Trials with `choice == 0` (no response)
are removed both by the reference mask's `exclude_nochoice=True` and by the explicit
`np.isin(choice_raw, (-1.0, 1.0))` test, so only ±1 reaches the output.

ii.
```python
choice_raw = trials['choice'].to_numpy(dtype=float)
mask &= np.isin(choice_raw, (-1.0, 1.0))
```

iii. Step 5's mapping table: "`trials.choice` → `output[s][k][0]` 'choice'", reference function
`bin_behaviors` (`choice`), which likewise takes `trials_df['choice'].to_numpy()`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The recoding `(1 − choice)/2`, i.e. `+1 → 0` (left) and `−1 → 1` (right), stored as an integer and
broadcast constant across the 100 bins. The sign convention was not assumed: the AI verified it
empirically against the stimulus side.

ii.
```python
choice_out = ((1 - choice_raw[mask]) / 2).astype(np.int64)      # +1 -> 0 (left)
...
outputs[:, 0, :] = choice_out[:, None]
```
```python
'choice': "mouse's reported stimulus side; 0 = left (trials.choice == +1), "
          '1 = right (trials.choice == -1); constant within a trial',
```

iii. Step 4's discrepancy table records the empirical check: "On correct trials with the stimulus on
the **left**, `choice == +1` in every session checked (236/236, 141/141, 231/231); with the stimulus
on the **right**, `choice == −1` (203/203, 194/194, 194/194). So `choice == +1` ⇒ reported LEFT →
`out = (1 − choice)/2`." Step 10 sanity checks 6 and 8 re-verify this from `_ibl_trials.table.pqt`
across 5 sessions and then on 2,279 correct trials in 8 further sessions. Step 5 decision 6 explains
the broadcast: "`verify_data_format` requires one common `doutput` for all trials, and the task says
to make outputs time-varying when possible." Observed balance: left 0.508 / right 0.492.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. `trials.probabilityLeft`, which takes the three values 0.2, 0.5 and 0.8. Any trial whose value is
not one of those three is dropped.

ii.
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}

def map_prior(probability_left):
    """{0.2, 0.5, 0.8} -> {0, 1, 2}; anything else -> -1 (trial dropped)."""
    p = np.asarray(probability_left, dtype=float)
    out = np.full(len(p), -1, dtype=np.int64)
    for value, code in PRIOR_MAP.items():
        out[np.isclose(p, value)] = code
    return out
```

iii. Step 5's mapping table: "`trials.probabilityLeft` → `output[s][k][1]` 'prior_prob_left'",
reference function `bin_behaviors` (`block`), which stores `trials_df['probabilityLeft']`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Only the recoding {0.2 → 0, 0.5 → 1, 0.8 → 2} the Decoder Task specifies, using `np.isclose` for
float safety, broadcast constant across the 100 bins. The unbiased 0.5 block is deliberately kept
(the reference's `exclude_unbiased=False`) because it is one of the three required classes; it is
~14 % of trials, matching the paper's 90/645 = 0.139. Class shares are 41.8 / 14.0 / 44.2 %.

ii.
```python
prior_code = map_prior(trials['probabilityLeft'].to_numpy())
mask &= prior_code >= 0
...
prior_out = prior_code[mask]
outputs[:, 1, :] = prior_out[:, None]
```
```python
OUTPUT_VALUES = [
    ['left', 'right'],
    ['p(left)=0.2', 'p(left)=0.5', 'p(left)=0.8'],
    ...
]
```

iii. Step 5 decision 3 notes the unbiased block is "kept (needed for the 3-class prior output)".
Step 9 checks the unbiased-block fraction (0.140 vs 90/645 = 0.139) and that pLeft takes only the
three released values. Sanity check 7 re-derives the mapping from the raw trials table for 5 sessions
and confirms it is constant across bins.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position` and `_ibl_wheel.timestamps`, read through `SessionLoader.load_wheel()`,
which returns a dataframe of interpolated `times`, `position`, `velocity` and `acceleration`. The
speed is `abs(velocity)`.

ii.
```python
def load_wheel_speed(sess_loader):
    """|wheel velocity| -- ``load_target_behavior(one, eid, 'wheel-speed')``."""
    if sess_loader.wheel is None or len(sess_loader.wheel) == 0:
        sess_loader.load_wheel()
    return (sess_loader.wheel['times'].to_numpy(dtype=float),
            np.abs(sess_loader.wheel['velocity'].to_numpy(dtype=float)))
```

iii. The docstring names the reference function it reproduces: `load_target_behavior(one, eid,
'wheel-speed')`, which is literally `np.abs(sess_loader.wheel['velocity'].to_numpy())`. Step 5's
mapping table lists the source as `wheel.velocity`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) `SessionLoader.load_wheel()` does the IBL-standard preprocessing internally —
the wheel is recorded only on movement, so the position is interpolated onto a uniform 1000 Hz grid
and differentiated into a velocity with a low-pass filter; the absolute value gives speed in rad/s.
(2) The session-long speed trace is resampled onto the 100 per-trial query points
`interval_beg + 0.02·(i+1)`, i.e. the **bin right edges**, which is exactly the reference's
`np.linspace(beg + binsize, end, n_bins)`. `np.interp` is called once per session over the whole
(n_trials × 100) query grid, against the full trace rather than the per-trial slice. (3) The
resampled values are discretised into tertiles (7-c). A trial is rejected if the trace does not cover
its window (first sample > 1 bin late or last sample > 1 bin early — the reference's own criteria) or
if any resampled value is non-finite.

ii.
```python
    # Query grid = bin right edges, exactly linspace(beg + binsize, end, NBINS).
    q = begs[ok][:, None] + BIN_RIGHT_EDGES[None, :] - WIN[0]
    interp = np.interp(q.ravel(), target_times, target_vals).reshape(q.shape)
    vals[ok] = interp
    good[ok] = True
```
```python
    idx_beg = np.searchsorted(target_times, safe_b, side='right')
    idx_end = np.searchsorted(target_times, safe_e, side='left')
    ...
    ok = nonempty & (np.abs(begs - first_t) <= BINSIZE) & (np.abs(ends - last_t) <= BINSIZE)
```
```python
wt, wv = load_wheel_speed(sess_loader)
wheel_vals, wheel_ok = behavior_per_interval(wt, wv, interval_begs)
```

iii. Step 5 decision 4: "Behaviour interpolation follows `get_behavior_per_interval`: the same
coverage checks (data must start ≤ 1 bin late and end ≤ 1 bin early) and the same query grid
`linspace(beg + binsize, end, 100)` (bin right edges). Trials failing the checks are dropped from
every stream, exactly as `align_spike_behavior` does." The one documented departure is interpolating
against the full session trace instead of the per-trial slice: "linear interpolation is local, so
interior values are identical, and the reference's `fill_value='extrapolate'` only ever applied
within one bin of the slice edge." Sanity check 9 re-derives the wheel speed from
`_ibl_wheel.position/timestamps.npy` with its own `interpolate_position` + `velocity_filtered` for 4
sessions: "100.000 % of bins identical".

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 within-session equal-occupancy classes. The 33.33rd and 66.67th percentiles are computed
over **all retained trials × all 100 time bins of that session** (one pair of thresholds per session,
not per trial), and values are assigned with `np.searchsorted(edges, x, side='right')` — equivalent
to `np.digitize`. The thresholds are stored in the metadata. If the two edges are not strictly
increasing (a stuck wheel or dead ROI) the whole session is rejected.

ii.
```python
def discretize_tertiles(values):
    flat = values.ravel()
    edges = np.percentile(flat, [100.0 / N_DISCRETE_BINS * i
                                 for i in range(1, N_DISCRETE_BINS)])
    binned = np.searchsorted(edges, flat, side='right').reshape(values.shape)
    return binned.astype(np.int64), edges
```
```python
wheel_bin, wheel_edges = discretize_tertiles(wheel_kept)
me_bin, me_edges = discretize_tertiles(me_kept)
for name, edges in (('wheel speed', wheel_edges), ('whisker motion energy', me_edges)):
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError(f'{name} is degenerate (tertile edges {list(edges)})')
```

iii. Step 5 decision 5: "both signals are in arbitrary, session-specific units … so a global
threshold would put whole sessions in a single class. The reference decoder makes the same choice in
spirit: `SingleSessionDataset` fits a per-session `StandardScaler` on the training trials. Tertiles
also give equal class priors, so chance = 1/3 … and the balanced accuracy reported by
`train_decoder.py` is directly interpretable." The degenerate-edge rule came from sanity check 12,
which initially failed with a worst class-fraction deviation of 0.667 on session `5b44c40f…`; after
the fix the worst deviation across all sessions is 0.0001. Step 12 Check 3 addresses the leakage risk
of fitting the thresholds on all trials rather than the training split: train/val ratio 1.01 for both
dynamic outputs, "as expected when two scalars are estimated from ~40,000 values."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Through the same `interval_begs = stimOn_times − 0.5` used for the spikes: element *i* of the
output is the wheel speed at `interval_beg + 0.02·(i+1)`, the closing edge of the same 20 ms bin *i*
that the spikes were counted into. Wheel and spike times are already on one synchronised session
clock, so no further alignment is needed. A trial that the wheel does not cover is dropped from
*every* stream, so the neural, input and output lists stay index-aligned.

ii.
```python
wheel_vals, wheel_ok = behavior_per_interval(wt, wv, interval_begs)
...
mask &= wheel_ok & me_ok
...
wheel_kept = wheel_vals[mask]
```
```python
q = begs[ok][:, None] + BIN_RIGHT_EDGES[None, :] - WIN[0]
```

iii. Step 5 decision 4 (the reference query grid is the bin right edges) and Step 10 Check 3(f). The
`--show-processing` figure overlays the raw wheel trace, the resampled trace and the discretised
output on a single time axis with `stimOn` at 0; Step 7 records that these panels were inspected.
Step 12's time-resolved analysis shows wheel-speed decodability peaking just after t = 0 rather than
before it, consistent with correct alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy.npy` with its frame times `_ibl_<side>Camera.times.npy`, read
through `SessionLoader.load_motion_energy(views=[view])` as the `whiskerMotionEnergy` column. The
left camera is preferred and the right used as a fallback — the same preference order as the
reference's `bin_behaviors`. In the converted dataset 433 sessions use the left camera and 7 the
right; 13 sessions were skipped entirely because neither camera has motion energy.

ii.
```python
def load_whisker_me(sess_loader):
    """Whisker motion energy, left camera with the right camera as fallback.

    Same preference order as ``ibl_data_utils.bin_behaviors``.
    """
    for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            df = sess_loader.motion_energy[key]
            t = df['times'].to_numpy(dtype=float)
            v = df['whiskerMotionEnergy'].to_numpy(dtype=float)
            finite = np.isfinite(t)
            if finite.sum() < 2:
                continue
            return t[finite], v[finite], view
        except Exception:
            continue
    raise RuntimeError('no whisker motion energy available')
```

iii. Step 4's discrepancy table: "Whisker camera — code: left, fall back to right; data: 436 left,
433 right, 445 either; paper: 'near the whisker pad' (side camera) → Follow the code: left first,
right as fallback." Step 3 records the definition: "mean absolute difference between adjacent video
frames in a bounding box between nose tip and eye, from the side camera."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used exactly as it is — no filtering, normalisation or rescaling. Frames
with non-finite timestamps are dropped first. The trace is then put through the identical
`behavior_per_interval` path as the wheel: the same coverage checks, the same bin-right-edge query
grid, one `np.interp` per session, rejection of any trial whose resampled trace contains a non-finite
value; then the same per-session tertile split (8-c).

ii.
```python
finite = np.isfinite(t)
if finite.sum() < 2:
    continue
return t[finite], v[finite], view
```
```python
mt, mv, me_view = load_whisker_me(sess_loader)
me_vals, me_ok = behavior_per_interval(mt, mv, interval_begs)
```
```python
    bad = ~np.all(np.isfinite(vals[ok]), axis=1)
    if np.any(bad):
        idx = np.nonzero(ok)[0][bad]
        good[idx] = False
```

iii. Step 5's mapping table lists no transform beyond "interpolated onto the 100 bin right-edges,
then discretised into 3 within-session tertiles", with reference functions
`load_target_behavior('*-whisker-motion-energy')` and `get_behavior_per_interval`. Step 10 Check 5
notes "NaN camera timestamps are dropped before interpolation". Sanity check 11 re-derives the whole
resampled trace straight from `{left,right}Camera.ROIMotionEnergy.npy` + `_ibl_*Camera.times.npy`
for 4 sessions: "100.000 % of bins identical".

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel speed: 33.33 / 66.67 percentiles of that session's own retained
trials × bins, one pair of thresholds per session, `searchsorted(..., side='right')` to assign the 3
classes, edges stored in the metadata, and the session rejected if the edges tie. Resulting class
fractions are 0.333 / 0.333 / 0.333 with a worst-case per-session deviation of 0.0001.

ii.
```python
me_bin, me_edges = discretize_tertiles(me_kept)
...
if not np.all(np.diff(edges) > 0):
    raise RuntimeError(f'{name} is degenerate (tertile edges {list(edges)})')
```
```python
'whisker_motion_energy': 'whisker-pad motion energy (left camera, right as '
                         'fallback) resampled to the 20 ms bins and '
                         'discretised into 3 equal-occupancy within-session '
                         'bins',
```

iii. Step 5 decision 5 makes the per-session case most strongly for this variable: "whisker ME
depends on camera, lighting and ROI size". The degenerate-edge rule is documented in Step 10 Check 2:
session `5b44c40f…` "has a broken whisker ROI whose motion energy is exactly 0 for 78 % of in-trial
samples, so both tertile edges came out at 0.0 and every bin was assigned to one class"; that session
is now rejected, and it is the only one affected.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: sampled at `interval_beg + 0.02·(i+1)` for bin *i*, on the same
session clock and from the same `stimOn_times`-derived interval, so bin index *i* of the whisker
output and bin index *i* of the neural matrix describe the same 20 ms of the same trial. Trials the
camera does not cover are dropped from every stream at once.

ii.
```python
me_vals, me_ok = behavior_per_interval(mt, mv, interval_begs)
mask &= wheel_ok & me_ok
...
me_kept = me_vals[mask]
outputs[:, 3, :] = me_bin
```

iii. Step 5 decision 4 and Step 10 Check 3(f); the `--show-processing` panel plots the raw camera
trace against the resampled trace and the tertile edges on the stimulus-onset axis. Step 12's
time-resolved decodability for whisker ME is flat before t = 0 and rises after, as expected.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Everything degenerate is dropped rather than patched, and every drop is recorded.
 * **Missing datasets**: a session with no whisker motion energy on either camera, or with no spike
   sorting at all, raises inside its worker and is reported in `metadata['failed_sessions']` with its
   reason (13 of the 19 skipped sessions are of this kind). A probe insertion whose spike sorting was
   never released is skipped with the rest of the session kept.
 * **Missing trial events**: NaN in any of the six key events is removed by the reference mask; NaN
   `probabilityLeft` outside {0.2, 0.5, 0.8} and `choice == 0` reject the trial; non-finite
   `stimOn_times` rejects the trial.
 * **Missing / short behaviour**: the reference coverage rules reject a trial whose wheel or camera
   trace starts more than one bin late or ends more than one bin early; NaN camera timestamps are
   dropped before interpolation; a NaN anywhere in a resampled trace rejects the trial.
 * **Missing ephys**: the window must lie inside the interval during which every probe was spiking.
 * **Degenerate behaviour**: tied tertile edges reject the whole session.
 * **Too little data**: < 5 neurons or < 2 usable trials rejects the session.
 * **Robustness**: the cluster remap array is sized `max(spikes['clusters']) + 2` so a spike
   referencing a cluster id past the end of the clusters table cannot index out of bounds; the
   single-probe path bypasses `merge_probes`, which mutates its input in place.
 13 all-zero-neural warnings survive in `verification_full_out.txt` and are argued to be real
 observations rather than conversion errors.

ii.
```python
def _worker(job):
    try:
        res = convert_session(...)
        return ('ok', eid, res)
    except Exception as exc:                       # mirrors the reference try/except
        return ('fail', eid, f'{type(exc).__name__}: {exc}')
```
```python
if len(sp) == 0 or 'times' not in sp or len(sp['times']) == 0:
    continue
```
```python
remap = np.full(int(np.max(spikes['clusters'])) + 2, -1, dtype=np.int64)
```
```python
'failed_sessions': [{'eid': e, 'reason': m} for e, m in failures],
```
```python
'kept_trial_idx': np.nonzero(mask)[0].astype(np.int32),
```

iii. Step 9: "Nothing is lost silently: every one of the 19 skipped sessions is recorded with its
reason in `data['metadata']['failed_sessions']`, and every retained trial's index in the raw trials
table is recorded in `session_info[s]['kept_trial_idx']`." Step 10 Check 1 explains why the 13
remaining zero-neural warnings are not fixed: 12 are in a 10-neuron session firing at 3.8 Hz, where
"dropping them would bias the dataset by removing exactly the low-activity trials", and the 13th is
"a momentary acquisition dropout well inside the recording span … a single trial out of 187,513".
Step 10 Check 5 lists the edge cases individually.

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting off disk, by a wide margin. The script instruments itself and prints the
breakdown: `load_spikes` 8.30 s/session (3650 s total), `load_behavior` 1.05 s, `bin_spikes` 0.71 s,
`load_trials` 0.51 s, total 10.61 s/session. Wall clock for the full 459-session run was 239 s
because sessions run 16–24-wide in a process pool; pickling the 11.7 GB output took a further 18.4 s.
So ~78 % of the CPU time is file I/O on the two large spike arrays (plus `merge_clusters`, which is
bundled into the same timer).

ii.
```python
    t = time.time()
    spikes, clusters, rec_span = load_session_spikes(eid, pids, probe_names)
    timing['load_spikes'] = time.time() - t
```
```python
    for k in ('load_spikes', 'load_trials', 'load_behavior', 'bin_spikes', 'total'):
        vals = [t_[k] for t_ in ttl if k in t_]
        if vals:
            print(f'timing {k:14s}: mean {np.mean(vals):.2f}s  total {np.sum(vals):.1f}s')
```

iii. Step 6: "Spike sorting was being loaded once per probe with its own I/O … Parallelism is moved
up one level, to a `ProcessPoolExecutor` over **sessions** (24 workers), which keeps the NFS reads
(the real bottleneck, ~8 s/session) overlapped." Step 9: "239 s wall clock with 24 workers (vs the
~4 min estimate from Step 7; no re-optimisation needed)."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorized the two loops the reference code spends its time in, and left three small ones.
 * **Vectorized**: `behavior_per_interval` builds the whole (n_trials × 100) query grid and calls
   `np.interp` once per behaviour per session, replacing the reference's one `interp1d` object per
   trial per behaviour; `bin_spikes` replaces the reference's `multiprocessing.Pool` over trials
   (with a `bincount2D` and an `intersect1d` per trial) by one `np.searchsorted` for all trial
   boundaries and a single flat `np.bincount` per trial — "~0.7 s/session vs ~10 s for the reference
   path".
 * **Still scalar**: the per-trial `for k in range(n_trials)` in `bin_spikes` (could be done in one
   `bincount` by offsetting each spike's flat index by its trial, but each trial is a different slice
   of the session); `trial_number_in_block`, which is a pure-Python loop over ~650 trials that could
   be written as `(p != shift(p)).cumsum()` + `groupby().cumcount()`; and `bwm[bwm.eid == eid]` inside
   the job-building loop in `main`, which rescans the 699-row freeze once per session. All three are
   negligible next to the 8.3 s of I/O.

ii.
```python
    for k in range(n_trials):
        ...
        counts = np.bincount(uu * NBINS + b, minlength=n_units * NBINS)
        out[k] = counts.reshape(n_units, NBINS)
```
```python
    for i in range(len(p)):
        if i > 0 and not (p[i] == p[i - 1] or (np.isnan(p[i]) and np.isnan(p[i - 1]))):
            count = 0
        out[i] = count
        count += 1
```
```python
    for i, eid in enumerate(eids):
        sub = bwm[bwm.eid == eid]
```

iii. Step 6 lists the reference inefficiencies found and the speedups added, and notes the result:
"~0.7 s/session vs ~10 s for the reference path" for binning, and one `np.interp` call per behaviour
per session. The AI does not separately flag the three remaining loops, but its own timing table
shows they are not the bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. Very little of consequence, and nothing at the scale of the I/O.
 * `clusters['label'].to_numpy(dtype=float)` is materialised twice — once inside `select_units` for
   the mask and once in `convert_session` for the `n_clusters_label_good` statistic.
 * `bwm[bwm.eid == eid]` is a full-table scan performed once per session in `main`, and twice more in
   the `--sample` branch.
 * Brain-region indices are built twice: `np.unique(regions, return_inverse=True)` per session, then
   remapped into the global `brain_regions` list during assembly.
 * `BrainRegions()` and the `ONE` handle are constructed once per worker process rather than once
   overall — unavoidable, since a connection cannot be shared across a fork; both are memoised within
   a process (`get_one`, `get_brain_regions`).
 * `SessionLoader` is created once and reused for trials, wheel and motion energy, so the trials file
   is not re-read (this is why `sess_loader=` is passed into `load_trials_and_mask`).
 Nothing re-reads a spike or camera file.

ii.
```python
_ONE = None
_BR = None

def get_one():
    global _ONE
    if _ONE is None:
        from one.api import ONE
        _ONE = ONE(base_url=ALYX_URL, silent=True)
    return _ONE
```
```python
    keep_idx, beryl = select_units(clusters)
    n_all_clusters = len(clusters)
    n_label_good = int((clusters['label'].to_numpy(dtype=float) >= QC_LABEL).sum())
```
```python
    region_names, region_idx = np.unique(regions, return_inverse=True)
```

iii. Step 4 records the reason the loader is shared: "`SessionLoader(one, eid)` positional call in the
reference module fails on the installed ibllib → pass a pre-built loader through `sess_loader=`",
which also has the effect of avoiding a second read of the trials table. Step 6 lists "Avoid
unnecessary file I/O" among the optimisations, and the process-local `get_one` / `get_brain_regions`
memoisation is explicitly a per-process cache.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Some, all cheap, and most of it deliberately retained for validation rather than for the decoder.
 * `behavior_per_interval` resamples the wheel and whisker traces for **all** trials of the session,
   including the ~34 % that the trial mask will later discard; only `vals[mask]` is used. (The spike
   binning, by contrast, is already restricted to `interval_begs[mask]`.)
 * `SpikeSortingLoader.merge_clusters` computes the full metrics and histology table for every
   cluster, of which only `label`, `acronym` and `uuids` are used, and the spike sorting is loaded in
   full before ~93 % of clusters are filtered out.
 * `rec_span` takes `np.nanmin`/`np.nanmax` over every spike time of every probe.
 * A block of per-session diagnostics is computed and pickled but never consumed by the decoder:
   `n_clusters_all`, `n_clusters_label_good`, `n_trials_ref_mask`, `n_trials_in_recording`,
   `n_trials_all_zero_neural`, `frac_correct_raw`, `mean_firing_rate_hz`, `recording_span_s`,
   `kept_trial_idx` and the full `cluster_uuids` list per session.
 * `block_idx` is computed for the whole trials table though only the masked entries are stored.
 None of this is on the critical path — the whole conversion is 239 s wall clock — and the metadata
 is what makes the independent sanity checks possible.

ii.
```python
    wheel_vals, wheel_ok = behavior_per_interval(wt, wv, interval_begs)   # all trials
    ...
    wheel_kept = wheel_vals[mask]                                          # only the kept ones
```
```python
            'n_trials_all_zero_neural': int(np.sum(binned.sum(axis=(1, 2)) == 0)),
            'recording_span_s': [float(rec_span[0]), float(rec_span[1])],
            'kept_trial_idx': np.nonzero(mask)[0].astype(np.int32),
            'cluster_uuids': list(clusters['uuids'].to_numpy()[keep_idx]),
            'frac_correct_raw': float(np.nanmean(trials['feedbackType'].to_numpy() == 1)),
            'mean_firing_rate_hz': float(binned.mean() / BINSIZE),
```

iii. The AI does not list these as waste; it justifies the metadata as provenance — the code comment
reads "index of each retained trial in the raw trials table, for provenance and for the independent
sanity checks in CONVERSION_NOTES Step 10" — and Step 10 Check 2 does use `cluster_uuids` and
`kept_trial_idx` to re-derive neural, input and output values straight from the ALF files. Step 9
uses `n_clusters_all`, `n_clusters_label_good` and `frac_correct_raw` for the paper-statistics
comparison. The behaviour-before-mask ordering is a consequence of Step 5 decision 4, where the
behaviour coverage flags are themselves part of the trial mask and so must be computed for every
trial.
