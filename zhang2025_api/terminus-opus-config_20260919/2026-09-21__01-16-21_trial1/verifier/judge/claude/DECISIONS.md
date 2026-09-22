# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read through the ONE API against the staged IBL cache; no data file is
opened directly. The session/insertion inventory is taken from the reference code's own
freeze file `/app/code/code_zhang2025/data/bwm_release.csv` (699 insertions, 459 eids,
139 subjects, 12 labs), exactly as `src/0_data_caching.py` does. From that table the AI
gets, per session, the list of `(pid, probe_name)` pairs. `ONE()` is instantiated with no
arguments so that it reads `~/.one/.caches`, which points `CACHE_DIR` at
`/app/data/one_cache` and lets dataset queries be answered offline from the cached Alyx
REST responses in `.rest/`; the AI documents that building ONE with an explicit
`tables_dir` instead resolves the *pre-revision* release tables and silently returns a
nearly empty trials table. Per session, `SessionLoader` loads trials, wheel and camera
motion energy; per probe, `SpikeSortingLoader.load_spike_sorting()` +
`SpikeSortingLoader.merge_clusters(...).to_df()` load spikes and the cluster table.
Sessions are processed in parallel with a `ProcessPoolExecutor` (24–32 workers); the full
run took 3.5 min.

ii.
```python
def get_one():
    """ONE instance that works offline against the staged cache.
    NOTE: ONE() with no arguments reads ~/.one/.caches, which points CACHE_DIR at
    /app/data/one_cache ...  Building ONE with an explicit tables_dir instead uses the
    frozen release tables, which do NOT list the revisioned datasets that are actually on
    disk (e.g. alf/#2025-03-03#/_ibl_trials.table.pqt) and silently returns a nearly empty
    trials table.
    """
    from one.api import ONE
    return ONE()
```

```python
BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
...
bwm = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
print(f'bwm_release.csv: {len(bwm)} insertions, {bwm.eid.nunique()} sessions, '
      f'{bwm.subject.nunique()} subjects, {bwm.lab.nunique()} labs')
...
by_eid = {e: list(g[['pid', 'probe_name']].itertuples(index=False, name=None))
          for e, g in bwm.groupby('eid')}
```

```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
...
for pid, pname in probes:
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. From CONVERSION_NOTES Step 4/Step 6: `bwm_release.csv` is the reference pipeline's own
session freeze (`freeze_file = 'data/bwm_release.csv'` in `0_data_caching.py`), and reading
it reproduces the data paper's headline numbers exactly ("699 pids / 459 eids / 139
subjects / 12 labs — exactly the data paper's numbers"), which the AI used as a hard
sanity check. All actual data access goes through ONE/brainbox as the instructions require;
the `get_one()` docstring records the concrete failure mode (empty trials tables) that
motivated using the default `ONE()` constructor.

## 1-b. How are the data split into subjects?

i. Not derived — the subject name is a column of `bwm_release.csv` and is attached to each
eid before processing. At assembly the `subjects` list is the sorted unique subject names
over the *kept* sessions and `subject_idx` is each session's index into that list. Result:
136 subjects over the 442 kept sessions (139 in the release; 3 lost with their sessions).

ii.
```python
subj_of = dict(zip(bwm.eid, bwm.subject))
lab_of  = dict(zip(bwm.eid, bwm.lab))
...
subjects = sorted({subj_of[e] for e in kept})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_index[subj_of[e]] for e in kept], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: "`bwm_release.csv` subject →
`subjects`, `subject_idx` … 139 mice". The release table already carries a unique subject
id per session, so nothing has to be parsed or inferred; the count is cross-checked against
the data paper's 139 mice.

## 1-c. How are the data split into sessions?

i. A session is the natural unit: one `eid` per session. The eid list is the unique,
file-ordered set of `bwm.eid`; each eid becomes one task, one element of `neural`/`input`/
`output`, and one entry of `metadata['session_info']`. 459 sessions are attempted, 442 kept.
An optional `/app/data/DATALIMIT_SUBSET.csv` restricts the eid list if present.

ii.
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))   # preserve file order, unique
if args.sample:
    eids = eids[:2]
...
tasks = [(e, by_eid[e], i < n_show) for i, e in enumerate(eids)]
...
kept = [e for e in eids if results[e].get('skip') is None]
```

iii. No decision to make — the release is organised by session and `bwm_release.csv` lists
one row per insertion with its eid, so grouping by eid recovers the sessions (and the 1–2
probes that must be merged into a single population).

## 1-d. How are the data split into trials?

i. The trials table returned by `SessionLoader.load_trials()` has one row per trial, so the
split is given by the data. A trial is materialised as the window
`[stimOn_times − 0.5, stimOn_times + 1.5)`, i.e. `t_beg = stimOn_times − 0.5` and 100 × 20 ms
bins from there.

ii.
```python
sl.load_trials()
trials = sl.trials
if len(trials) == 0:
    return {'eid': eid, 'skip': 'no trials'}
...
align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
t_beg = align + TIME_WINDOW[0]
```

iii. CONVERSION_NOTES Step 5 ("Trial geometry"): alignment event `trials.stimOn_times`,
window (−0.5, +1.5) s, 20 ms bins, T = 100 — "methods paper + reference code". The trials
table is already one row per trial, so only the window definition is a decision.

## 1-e. How are trials filtered based on quality controls?

i. Five layers, intersected:
1. A literal transcription of the reference `load_trials_and_mask(min_rt=0.08, max_rt=2.0,
   max_trial_len=10.0, nan_exclude='default', exclude_nochoice=True)` pandas query: reaction
   time (`firstMovement_times − stimOn_times`) in [0.08, 2.0] s; `feedback_times − goCue_times
   ≤ 10 s`; no NaN in `stimOn_times, choice, feedback_times, probabilityLeft,
   firstMovement_times, feedbackType`; `choice != 0`.
2. `np.isfinite(t_beg)`.
3. Behavioural coverage: the trial window must be covered by both the wheel trace and the
   whisker motion-energy trace (the reference `get_behavior_per_interval` "target data
   starts too late / ends too early" tests, with the same one-bin tolerance).
4. Spike coverage (an *addition* to the reference): the window must lie inside the spike
   sorting and contain at least one spike.
5. `probabilityLeft` must be one of {0.2, 0.5, 0.8}.
The unbiased (pLeft = 0.5) block is deliberately **kept**, because it is class 1 of the
prior output. A session with < 2 surviving trials is dropped. Result: 188,044 trials from
442 sessions (mean 425/session, median 392, min 85, max 1445).

ii.
```python
def build_trials_mask(trials):
    """Reproduce reference ibl_data_utils.load_trials_and_mask(max_trial_len=10.)."""
    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in NAN_EXCLUDE:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'
    return np.asarray(~trials.eval(query), dtype=bool).copy()
```

```python
spk_ok = np.zeros(len(t_beg), dtype=bool)
if len(spk_times_k) > 1:
    t_end_all = t_beg + NBINS * BINSIZE
    inside = np.isfinite(t_beg) & (t_beg >= spk_times_k[0] - BINSIZE) & \
             (t_end_all <= spk_times_k[-1] + BINSIZE)
    i0 = np.searchsorted(spk_times_k, np.where(inside, t_beg, spk_times_k[0]), 'left')
    i1 = np.searchsorted(spk_times_k, np.where(inside, t_end_all, spk_times_k[0]), 'left')
    spk_ok = inside & (i1 > i0)
trial_mask &= spk_ok

ws_vals, ws_ok = interp_behavior(wheel_t, wheel_v, t_beg)
wm_vals, wm_ok = interp_behavior(whisker_t, whisker_v, t_beg)
# reference align_spike_behavior: a trial survives only if it is good in the trials
# mask AND in every behaviour mask
keep_trial = trial_mask & ws_ok & wm_ok
if keep_trial.sum() < 2:
    return {'eid': eid, 'skip': f'only {int(keep_trial.sum())} usable trials'}
```

```python
prior_cls = np.full(len(p_left), -1, dtype=np.int64)
prior_cls[np.isclose(p_left, 0.2)] = 0
prior_cls[np.isclose(p_left, 0.5)] = 1
prior_cls[np.isclose(p_left, 0.8)] = 2
keep_trial = keep_trial & (prior_cls >= 0)
```

iii. CONVERSION_NOTES Step 5 Key Decision 3 and Step 10 Check 3: the trial mask is "the
same pandas query, transcribed literally" from the reference, including `max_trial_len=10.0`
which the reference code passes in `prepare_data`. The behaviour masks are "same tests, same
intersection" as `align_spike_behavior`. The spike-coverage test is an explicit addition:
Step 10 Check 1 documents that the first full run emitted 38 "all neural data is zero"
warnings, traced to session `8c2f7f4d` (three trials starting after its last spike) and
`b182b754` (one trial inside a 2.07 s drop-out); "the reference pipeline applies a coverage
test to the behavioural traces … but **not** to the spikes, so these silently became
all-zero matrices", so the same rule was applied to the neural stream and 16 trials were
removed. Keeping the pLeft = 0.5 trials is justified in Step 4: they "are class 1 of the
prior output".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from `SpikeSortingLoader.load_spike_sorting()`,
concatenated over the probes of a session. The merged cluster table
(`SpikeSortingLoader.merge_clusters(...).to_df()`) supplies only `label` (quality) and
`acronym` (anatomy), which are used for curation and for `brain_region_idx`, not for the
neural array itself.

ii.
```python
for pid, pname in probes:
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
    if spikes is None or len(spikes) == 0 or 'times' not in spikes:
        continue
    cdf = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    # merge_probes: shift this probe's cluster ids so they are unique in the session
    spk_times.append(np.asarray(spikes['times'], dtype=np.float64))
    spk_clusters.append(np.asarray(spikes['clusters'], dtype=np.int64) + offset)
    acronyms.append(cdf['acronym'].to_numpy().astype(str))
    labels.append(cdf['label'].to_numpy(dtype=np.float64))
    offset += len(cdf)
```

iii. Step 10 Check 3 records this as identical to the reference `load_spiking_data`:
"`SpikeSortingLoader(pid).load_spike_sorting()` + `merge_clusters(...).to_df()`, then
`merge_probes` | identical calls; probes merged with the same cluster-id offsetting and time
sort | **yes**".

## 2-b. How is the `neural` data processed?

i. Probes are merged into one population by offsetting the second probe's cluster ids by the
first probe's cluster count (the reference `merge_probes`), NaN spike times are dropped, and
all spikes are sorted by time. After neuron curation the surviving clusters are renumbered
contiguously. Spikes are then counted into 20 ms bins over each trial's window with a
`searchsorted` slice per trial plus a single `np.add.at` onto a flat (cluster × bin) view.
The stored value is the **raw spike count per 20 ms bin**, float32, *not* converted to Hz and
not smoothed or normalised. Only trials that survive curation are binned. Final array per
trial: `(n_neurons, 100)`.

ii.
```python
def bin_spikes_trials(spike_times, spike_clusters, n_clusters, t_beg):
    """Spike counts per (trial, cluster, bin).
    Equivalent to the reference bincount2D(times, clusters, xbin=BINSIZE,
    xlim=[t_beg, t_end]) truncated to NBINS bins, but computed for all trials at once."""
    out = np.zeros((n_trials, n_clusters, NBINS), dtype=np.float32)
    t_end = t_beg + NBINS * BINSIZE
    i0 = np.searchsorted(spike_times, t_beg, side='left')
    i1 = np.searchsorted(spike_times, t_end, side='left')
    flat = out.reshape(n_trials, n_clusters * NBINS)
    for k in range(n_trials):
        a, b = i0[k], i1[k]
        if b <= a:
            continue
        bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)
        np.clip(bi, 0, NBINS - 1, out=bi)
        ci = spike_clusters[a:b]
        np.add.at(flat[k], ci * NBINS + bi, 1.0)
    return out
```

```python
finite = np.isfinite(spk_times)
spk_times, spk_clusters = spk_times[finite], spk_clusters[finite]
order = np.argsort(spk_times, kind='stable')
spk_times, spk_clusters = spk_times[order], spk_clusters[order]
...
kt = np.flatnonzero(keep_trial)
binned = bin_spikes_trials(spk_times_k, spk_clusters_k, n_neurons, t_beg[kt])
```

iii. Step 5 Key Decision 5: "Spike counts, not rates, un-normalised. The reference caches raw
counts; the decoder does its own SVD/PCA. Counts keep the data integral-valued and sparse."
Key Decision 8: "Both probes of a session merged into one population (`merge_probes`), as the
reference does and as the data paper requires". Step 10 Check 3 labels the binning "same bin
edges, vectorised over trials | **yes** (verified `allclose` against `np.histogram`)", and
`metadata['neural_units']` is explicitly `'spike counts per 20 ms bin (not normalised)'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two cuts, applied together per session after the probes are merged:
(a) `clusters.label >= 1` — the IBL "well-isolated"/single-unit criterion (all three RIGOR
metrics passed); (b) the cluster's **Beryl** acronym must not be `root` or `void`, i.e.
grey matter with a named summary structure. The surviving clusters are renumbered and the
spikes of dropped clusters are discarded. A session with zero surviving units, or fewer than
5, is dropped entirely (3 sessions). Result: 62,773 neurons over 442 sessions (mean 142,
median 123, min 7, max 516), from 599,331 units seen and 73,026 well-isolated units in those
sessions.

ii.
```python
QC_LABEL = 1.0
NON_GREY = ('root', 'void')
...
beryl = np.asarray(br.acronym2acronym(acronyms, mapping='Beryl')).astype(str)
n_good = int((labels >= QC_LABEL).sum())
keep_unit = (labels >= QC_LABEL) & ~np.isin(beryl, NON_GREY)
if keep_unit.sum() == 0:
    return {'eid': eid, 'skip': 'no well-isolated grey-matter units', ...}
unit_idx = np.flatnonzero(keep_unit)
remap = np.full(offset, -1, dtype=np.int64)
remap[unit_idx] = np.arange(len(unit_idx))
new_clu = remap[spk_clusters]
sel = new_clu >= 0
spk_times_k, spk_clusters_k = spk_times[sel], new_clu[sel]
unit_regions = beryl[unit_idx]
```

```python
MIN_NEURONS_PER_SESSION = 5
for e in list(kept):
    if results[e]['n_neurons'] < MIN_NEURONS_PER_SESSION:
        results[e]['skip'] = (f"only {results[e]['n_neurons']} well-isolated "
                              f"grey-matter neurons (< {MIN_NEURONS_PER_SESSION})")
```

iii. Step 4 records that the two references disagree — the reference code calls
`load_spiking_data` with `qc=None` (keeps every Kilosort cluster) while the data paper
analyses only well-isolated neurons — and the AI follows the **data paper**, for five stated
reasons: it is the documented QC standard of the dataset; `brain_region_idx` must name a real
region so `root`/`void` units "have no anatomical identity"; scanning all 699 insertions
reproduced **621,733 total / 75,708 well-isolated** units, "exactly the data paper's numbers",
giving a hard check that the filter is right; all 621,733 units would be ~90 GB and would not
fit the decoder; and the excluded units are "by construction noisy/contaminated". The
grey-matter cut is attributed to the data paper ("restricted to regions that were designated
grey matter") and to the reference `3_decode_multi_region.py`. The ≥ 5-neuron session rule is
justified in Step 10 Check 1 and in the source comment: the data paper's 5-neuron threshold is
stated *per region* because its analyses are per region, whereas this decoder pools a whole
session, so applying it per region "cut the sample from 253 to 83 neurons and 16 regions to 1,
and on the full data would have discarded ~40% of well-isolated neurons"; it was therefore
applied at the session-population level, removing only 3 degenerate recordings.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. To `trials.stimOn_times`. All IBL streams are already on one session clock (synchronised
upstream), so alignment is just an offset: `t_beg = stimOn_times − 0.5`, and each spike's bin
is `floor((t − t_beg)/0.02)`. The window is [−0.5, +1.5) s around stimulus onset; bin `i`
covers `[stimOn − 0.5 + 0.02i, stimOn − 0.5 + 0.02(i+1))`, so bin 25 is the last fully
pre-stimulus bin and bin 25's right edge is exactly t = 0.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
t_beg = align + TIME_WINDOW[0]
...
bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)
```

```python
'temporal_alignment_event': 'stimulus onset (trials.stimOn_times)',
'off_start': float(TIME_WINDOW[0]),
'off_end': float(TIME_WINDOW[1]),
```

iii. Step 5 Key Decision 1: "Alignment = `stimOn_times`, window (−0.5, +1.5) s, 20 ms bins,
T = 100. Mandated by the decoder task (Temporally align based on stimulus onset) and
identical to the reference `params`" (`{'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}`
in `0_data_caching.py`). The alignment was verified two ways: the `--show-processing` figure
puts a stimulus-aligned raw raster next to the binned matrix of the same trial, and a
time-resolved logistic probe (Step 12) shows choice decodability at chance before t = 0 and
peaking at 0.796 at t = +0.24 s, which a misalignment would destroy.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, `NBINS = ceil(2.0 / 0.02) = 100` bins per trial, identical for every trial and
session. No rebinning, resampling or smoothing of the neural data — spikes are counted
directly onto the final grid. `metadata['time_bin_size'] = 20.0` (ms).

ii.
```python
TIME_WINDOW = (-0.5, 1.5)      # seconds relative to ALIGN_TIME
BINSIZE = 0.02                 # seconds
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

```python
'time_bin_size': BINSIZE * 1000.0,            # ms
'n_timepoints': NBINS,
'bin_time_convention': (
    'element i of a trial covers [stimOn - 0.5 + 0.02*i, stimOn - 0.5 + 0.02*(i+1)); '
    'behavioural traces and input[0] are evaluated at the bin right edge, matching '
    'the reference get_behavior_per_interval'),
```

iii. Step 4 resolves an apparent conflict: the methods text mentions 50 ms for choice/prior
and 20 ms for dynamic behaviours, but the reference `params` use `'binsize': 0.02` and the
model section says "2-s trials … 20-ms bins, producing T = 100". The AI chose **20 ms, T = 100**
because "One dataset must serve both per-trial and time-varying outputs", and verified T = 100
for every trial in every session.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from the data — it is the deterministic time grid implied by
`trials.stimOn_times` (the alignment event), the window (−0.5, +1.5) s and the 20 ms bin size.
The same 100-element vector is used for every trial of every session.

ii.
```python
time_axis = (TIME_WINDOW[0] + BINSIZE * np.arange(1, NBINS + 1)).astype(np.float32)
...
inp = np.empty((2, NBINS), dtype=np.float32)
inp[0] = time_axis
```

iii. Step 5 variable-mapping table: "bin right-edge time → `input[0]` = `time_from_stim_on` |
seconds, −0.48 … 1.50, time-varying | required by the decoder task (Time since stimulus onset,
continuous, time-varying)". The window and bin size come from the reference `params`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid. The value assigned to bin `i` is its **right edge**,
`−0.5 + 0.02·(i+1)`, giving −0.48 … +1.50 s. The same convention is used for the behavioural
traces, so all four time-varying streams share one representative time per bin.

ii.
```python
time_axis = (TIME_WINDOW[0] + BINSIZE * np.arange(1, NBINS + 1)).astype(np.float32)
```

iii. Step 5 "Trial geometry": "its representative time (used for behaviour interpolation and
for the time input) is its right edge, `-0.5 + 0.02*(i+1)`, i.e. −0.48 … +1.50 s. This is
exactly the reference `np.linspace(beg + binsize, end, n_bins)`" — i.e. the grid the reference
`get_behavior_per_interval` uses. The conversion log reports the realised range as
[−0.4800, 1.5000], matching the design.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural binning grid, expressed as the right edge of each neural bin: spikes for
bin `i` are counted over `[t_beg + 0.02i, t_beg + 0.02(i+1))` and `input[0][i] = −0.5 + 0.02(i+1)`.
Element `i` of the input therefore describes exactly the same interval as column `i` of the
neural matrix, for every trial and session, with no interpolation or offset.

ii.
```python
bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)      # neural bin index
...
time_axis = (TIME_WINDOW[0] + BINSIZE * np.arange(1, NBINS + 1))     # its right edge
```

iii. Step 5 "Trial geometry" and `metadata['bin_time_convention']` record the convention
explicitly. The `--show-processing` figure plots `input[0]` against time as a straight line
through the origin at t = 0 as a visual check; the sanity script re-derived `input[0]` from
scratch and matched with `np.allclose`.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft`, which is constant within a block, so a change of value marks
a new block. The trials table carries no block identifier, so blocks are recovered by
run-length decoding the prior over the **raw, unfiltered** trials table.

ii.
```python
p_left = trials['probabilityLeft'].to_numpy(dtype=np.float64)
...
tib = trial_number_in_block(p_left).astype(np.float32) / TRIAL_IN_BLOCK_SCALE
```

iii. Step 5 variable-mapping table: "`trials.probabilityLeft` run-lengths → `input[1]` =
`trial_number_in_block`". The AI verified against the data paper that `probabilityLeft` takes
only {0.2, 0.5, 0.8}, that the first block is the 90-trial unbiased block (measured median
90), and that biased blocks run 20–100 trials (measured mean 48.6 vs the paper's 51) — i.e.
that the recovered blocks are the task's real blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A vectorised 0-based run-length index (`np.maximum.accumulate` over block-start positions),
computed on the full trials table **before** any trial filtering, so a dropped trial still
advances the counter and the value is the animal's true position in the block. NaN runs are
treated as their own value rather than being split (NaN != NaN). The result is then **divided
by 100** and broadcast across the 100 time bins of the trial, giving a per-trial constant in
[0.00, 0.98].

ii.
```python
def trial_number_in_block(prob_left):
    """0-based index of each trial within its block of constant probabilityLeft."""
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    # NaN != NaN, so a NaN run would be split; treat NaNs as their own value.
    prev, cur = prob_left[:-1], prob_left[1:]
    is_new[1:] = ~((prev == cur) | (np.isnan(prev) & np.isnan(cur)))
    block_start = np.maximum.accumulate(np.where(is_new, np.arange(n), 0))
    return np.arange(n) - block_start
```

```python
TRIAL_IN_BLOCK_SCALE = 100.0   # keeps the input O(1); see CONVERSION_NOTES Step 5
...
tib = trial_number_in_block(p_left).astype(np.float32) / TRIAL_IN_BLOCK_SCALE
...
inp[1] = tib[k]
```

iii. Step 5 variable-mapping table: "0-based index of the trial within its block, divided by
100, per-trial (broadcast over T). … Divided by 100 so the value is O(1): the decoder
concatenates inputs with 100 neural PCs and feeds a linear layer with no input
standardisation." `metadata['input_scaling']` records the same. Step 10 Check 5 notes the edge
cases handled: the counter restarts at 0 on the first trial and at every change of
`probabilityLeft`, and NaN runs are not merged. The `--show-processing` figure plots the
expected sawtooth with a first tooth of 90 and later teeth of 20–100.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The single column `trials.choice`, which is +1, −1 or 0. `choice == 0` (no response) trials
are already removed by the trial mask.

ii.
```python
choice = trials['choice'].to_numpy(dtype=np.float64)
# choice == +1 is a LEFT report, choice == -1 a RIGHT report (verified against
# contrastLeft/contrastRight and feedbackType).  Required coding: left = 0, right = 1.
choice_cls = (choice < 0).astype(np.int64)
```

iii. Step 4 Cross-checks: "`choice` sign convention verified against
`contrastLeft`/`contrastRight` and `feedbackType` (choice = +1 ⟺ reported left)" — the AI did
not take the sign convention on trust but re-derived it from the stimulus side and the reward
outcome.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding: `+1 → 0` (left), `−1 → 1` (right), matching the required `left = 0,
right = 1`. The per-trial value is broadcast across all 100 time bins and stored as int64.
No-response trials are dropped rather than given a third class. Realised distribution over all
time bins: [0.5081, 0.4919].

ii.
```python
choice_cls = (choice < 0).astype(np.int64)
...
out = np.empty((4, NBINS), dtype=np.int64)
out[0] = choice_cls[k]
```

```python
OUTPUT_VALUES = [['left', 'right'], ...]
'choice_coding': ('trials.choice == +1 is a leftward report -> class 0 (left); '
                  'trials.choice == -1 -> class 1 (right)'),
```

iii. Step 5 variable-mapping table: "`choice == +1` (reported LEFT) → 0; `choice == −1`
(reported RIGHT) → 1 … matches the required left = 0, right = 1. `choice == 0` trials are
already excluded." Broadcasting per-trial variables over T is justified in Step 5: "The
decoder does this broadcast internally anyway (`SessionData.__getitem__`), and storing
everything 2-D keeps `d_input`/`d_output` unambiguous". The near-50/50 balance was listed as a
planned sanity check and confirmed.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The single column `trials.probabilityLeft`, the block prior, which takes only the values
0.2, 0.5 and 0.8.

ii.
```python
p_left = trials['probabilityLeft'].to_numpy(dtype=np.float64)
```

iii. Step 4 Cross-checks: "`probabilityLeft` takes exactly {0.2, 0.5, 0.8}, with 0.5 confined
to the first ~90 trials" — verified directly against the data before use.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Recoding 0.2 → 0, 0.5 → 1, 0.8 → 2 with `np.isclose` (float-safe). Any trial whose
`probabilityLeft` is none of the three is marked −1 and dropped from the trial mask. The
per-trial class is broadcast over the 100 bins. The 90-trial unbiased block is **kept**
because it constitutes class 1. Realised distribution: [0.4175, 0.1403, 0.4422].

ii.
```python
prior_cls = np.full(len(p_left), -1, dtype=np.int64)
prior_cls[np.isclose(p_left, 0.2)] = 0
prior_cls[np.isclose(p_left, 0.5)] = 1
prior_cls[np.isclose(p_left, 0.8)] = 2
keep_trial = keep_trial & (prior_cls >= 0)
if keep_trial.sum() < 2:
    return {'eid': eid, 'skip': 'no trials with a valid probabilityLeft'}
...
out[1] = prior_cls[k]
```

iii. Step 5 variable-mapping table: "0.2 → 0, 0.5 → 1, 0.8 → 2 | exactly as specified in the
decoder task". Step 4: the reference code sets `exclude_unbiased=False`, and the AI keeps the
pLeft = 0.5 trials because they are "required, because 'prior probability of left' is a
3-class output". The realised 14% share of class 1 was checked against the expectation of
~90 unbiased trials out of ~645 raw trials per session.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()`, i.e. `_ibl_wheel.position` / `_ibl_wheel.timestamps` turned
into a velocity by the loader; the speed is `abs(velocity)`, in rad/s, with its `times`.

ii.
```python
sl.load_wheel()
wheel_t = sl.wheel['times'].to_numpy(dtype=np.float64)
wheel_v = np.abs(sl.wheel['velocity'].to_numpy(dtype=np.float64))
```

iii. Step 10 Check 3: "(a) loading — wheel | `SessionLoader.load_wheel()`, `abs(velocity)` |
same | **yes**" — the same derivation as the reference's `'wheel-speed'` target.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) `SessionLoader.load_wheel()` does the standard IBL processing internally:
the sparsely-sampled wheel position is interpolated onto a uniform 1000 Hz grid and
differentiated with a 20 Hz Butterworth low-pass to give a velocity; the AI takes its absolute
value. (2) Non-finite samples are dropped and the trace is linearly interpolated (`np.interp`,
fully vectorised over trials) onto the 100 bin right edges of each trial. (3) The resulting
`(n_trials, 100)` matrix is discretised into 3 classes (see 7-c). No smoothing, z-scoring or
other normalisation is applied to the values themselves.

ii.
```python
gw = np.isfinite(wheel_t) & np.isfinite(wheel_v)
wheel_t, wheel_v = wheel_t[gw], wheel_v[gw]
...
def interp_behavior(times, values, t_beg):
    """Linear interpolation of a continuous behaviour onto the bin RIGHT EDGES.
    Matches reference get_behavior_per_interval, which evaluates at
    np.linspace(interval_beg + binsize, interval_end, n_bins)."""
    grid = BINSIZE * np.arange(1, NBINS + 1)                     # (NBINS,)
    xi = t_beg[:, None] + grid[None, :]                          # (n_trials, NBINS)
    ok = np.isfinite(t_beg)
    t_end = t_beg + NBINS * BINSIZE
    ok &= (t_beg >= times[0] - BINSIZE) & (t_end <= times[-1] + BINSIZE)
    safe = np.where(np.isfinite(xi), xi, times[0])
    vals = np.interp(safe, times, values).astype(np.float32)
    ok &= np.isfinite(vals).all(axis=1)
    return vals, ok
...
ws_vals, ws_ok = interp_behavior(wheel_t, wheel_v, t_beg)
```

iii. Step 6: the reference calls `get_behavior_per_interval` (a per-trial multiprocessing pool)
per behaviour; the AI "replaced [it] with a single vectorised `np.interp` over an
`(n_trials, 100)` grid" (~10× faster) while keeping the same interpolation kind (linear) and
the same evaluation grid. Step 10 Check 3 rates "(d) binning — behaviour |
`interp1d(kind='linear')` at `linspace(beg+binsize, end, 100)` | `np.interp` at the same right
edges | **yes**". The `--show-processing` figure overlays the raw ~1 kHz `|velocity|` trace with
the 100 interpolated points for four trials ("the binned curve sits exactly on the raw trace"),
and an independent re-interpolation from the raw `SessionLoader.wheel` matched with
`np.allclose` on 5 random sessions.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session tertiles. The thresholds are the 33.3rd and 66.7th percentiles of the pooled
finite values over **every time bin of every kept trial of that session**; a value is class 0
below the low threshold, 1 between, 2 above. If the two percentiles coincide (a degenerate,
near-constant trace) the code falls back to splitting on unique values. Thresholds are stored
per session in `metadata['session_info'][i]['wheel_speed_tertiles']`. Realised class fractions:
[0.3333, 0.3333, 0.3333].

ii.
```python
def tertile_bins(values, mask):
    """Discretise into 3 per-session classes at the 33.3 / 66.7 percentiles.
    Thresholds are computed over every time bin of every KEPT trial of the session, so
    the three classes are (near) equally populated within a session."""
    pool = values[mask].ravel()
    pool = pool[np.isfinite(pool)]
    lo, hi = np.percentile(pool, [100.0 / 3.0, 200.0 / 3.0])
    if not (hi > lo):
        # degenerate (e.g. a constant trace) -- fall back to unique-value splits
        uq = np.unique(pool)
        ...
    cls = (values > lo).astype(np.int64) + (values > hi).astype(np.int64)
    return cls, (float(lo), float(hi))
...
ws_cls, ws_thr = tertile_bins(ws_vals, keep_trial)
```

iii. Step 5 Key Decision 6: "Wheel speed and whisker ME are strongly right-skewed and their
absolute scale differs by an order of magnitude between sessions (different cameras, different
wheel gains, different lighting). Equal-width bins would put > 95% of samples in one class for
most sessions and would make the class meaning session-dependent. Tertiles … give three equally
populated, comparable classes: low/medium/high. This mirrors the reference pipeline, which
z-scores these behaviours per session before modelling (`SingleSessionDataset`:
`StandardScaler().fit(train_behavior)`)." The `--show-processing` figure draws the pooled
distribution with the two cuts and the resulting class fractions against a 1/3 reference line.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is evaluated at the same 100 bin right edges, measured from the same
`t_beg = stimOn_times − 0.5`, as the neural bins — so element `i` of the wheel output and
column `i` of the neural matrix describe the same 20 ms interval of the same trial. No
resampling or shifting beyond that interpolation; the wheel timestamps are already on the
session clock shared with the spikes.

ii.
```python
grid = BINSIZE * np.arange(1, NBINS + 1)
xi = t_beg[:, None] + grid[None, :]      # same t_beg used to bin the spikes
vals = np.interp(safe, times, values).astype(np.float32)
...
out[2] = ws_cls[k]
```

iii. `metadata['bin_time_convention']`: "behavioural traces and input[0] are evaluated at the
bin right edge, matching the reference `get_behavior_per_interval`". The alignment was checked
visually (raw trace vs binned points, row 3 of the processing figure) and quantitatively
(time-resolved probe, Step 12: wheel-speed decodability peaks at t = +0.18 s).

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with its frame times, loaded as
`SessionLoader.load_motion_energy(views=[view])['{view}Camera']['whiskerMotionEnergy']`. The
**left** camera is tried first and the **right** used as a fallback, matching the reference
`bin_behaviors`. A session with neither is dropped (14 sessions).

ii.
```python
whisker_t = whisker_v = None
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl.load_motion_energy(views=[view])
        me = sl.motion_energy[key]
        wt = me['times'].to_numpy(dtype=np.float64)
        wv = me['whiskerMotionEnergy'].to_numpy(dtype=np.float64)
        good = np.isfinite(wt) & np.isfinite(wv)
        if good.sum() > 1:
            whisker_t, whisker_v, whisker_view = wt[good], wv[good], view
            break
    except Exception:
        continue
if whisker_t is None:
    return {'eid': eid, 'skip': 'no whisker motion energy'}
```

iii. Step 4: "Whisker camera | code tries `left` then falls back to `right` | 437 sessions have
left, 433 have right, 445 have either | data paper: left camera 60 Hz, right 150 Hz | Same
left-then-right fallback as the reference." The chosen camera is recorded per session in
`session_info['whisker_camera']`, and the availability scan (445 of 459) exactly predicted the
14 sessions dropped for lack of a trace.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, smoothing or normalisation. Non-finite
samples are dropped, then the same vectorised `np.interp` resamples it onto the 100 bin right
edges of each trial, and the result is discretised into 3 per-session tertile classes. This is
the identical code path used for wheel speed.

ii.
```python
wm_vals, wm_ok = interp_behavior(whisker_t, whisker_v, t_beg)
...
wm_cls, wm_thr = tertile_bins(wm_vals, keep_trial)
```

iii. Step 5 variable-mapping table: "`SessionLoader.motion_energy[leftCamera].whiskerMotionEnergy`
(fallback rightCamera) → `output[3]` … interpolated to bin right edges, then per-session
tertiles → 0/1/2 | `load_target_behavior(left-whisker-motion-energy)` + fallback | time-varying".
The processing figure overlays the raw 60 Hz trace with the 100 interpolated points; an
independent re-interpolation from the raw `motion_energy` object matched with `np.allclose`.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Exactly the same rule as wheel speed: per-session 33.3 / 66.7 percentiles of the pooled
values over all time bins of all kept trials, with the unique-value fallback for degenerate
traces. Realised class fractions: [0.3345, 0.3327, 0.3328] (not exactly 1/3 because the
motion-energy trace is quantised and ties fall on one side of a threshold).

ii.
```python
wm_cls, wm_thr = tertile_bins(wm_vals, keep_trial)
...
out[3] = wm_cls[k]
```

```python
'output_discretisation': (
    'wheel_speed = |wheel velocity| and whisker_motion_energy are binned into '
    'three classes at the 33.3rd and 66.7th percentiles computed over all kept '
    'time bins of that session, so the classes are equally populated and '
    'comparable across sessions of different absolute scale.'),
```

iii. Same as 7-c: Step 5 Key Decision 6 — the absolute scale of motion energy differs between
sessions (different cameras at 60 vs 150 Hz, different lighting), so a per-session percentile
split is what makes "low/medium/high" mean the same thing across sessions, mirroring the
reference's per-session `StandardScaler`.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as the wheel: evaluated at the 100 bin right edges measured from the same
`t_beg = stimOn_times − 0.5`, so bin for bin it matches the neural matrix. Camera frame times
are on the same session clock as the spikes, so no further correction is applied.

ii.
```python
xi = t_beg[:, None] + grid[None, :]
vals = np.interp(safe, times, values).astype(np.float32)
```

iii. `metadata['bin_time_convention']` (quoted above) and Step 12's time-resolved probe, which
shows whisker-ME decodability rising after stimulus onset rather than being flat — evidence
against a temporal offset.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or defective data is detected and the affected unit of data is dropped, at the
smallest level that works, and every drop is reported:
- **Probe with no released spike sorting** → skipped, other probes of the session still used.
- **NaN spike times** → dropped before sorting and binning.
- **NaN camera/wheel samples** → dropped from the trace before interpolation.
- **NaN `stimOn_times`** → excluded by the trials NaN mask and re-checked with `np.isfinite(t_beg)`
  so no NaN ever reaches `searchsorted`; NaN query points are replaced by a safe value before
  `np.interp` and the trial is masked out anyway.
- **Trial not covered by the wheel / whisker trace** → trial dropped.
- **Trial not covered by the spike sorting** (recording stopped early, or a drop-out) → trial
  dropped; this is an addition to the reference, prompted by 38 "all neural data is zero"
  warnings.
- **Spike exactly on the window edge** → bin index clipped into range.
- **Degenerate (near-constant) behavioural trace** → tertile fallback to unique-value splits
  rather than an empty class.
- **Session with no whisker ME / < 5 well-isolated grey-matter neurons / < 2 usable trials** →
  session dropped, and the eid and reason are printed and stored in
  `metadata['skipped_sessions']`.
- **Worker exception** → caught per session, the session is skipped with the exception type and
  traceback, so a single bad session cannot abort the run.

ii.
```python
if spikes is None or len(spikes) == 0 or 'times' not in spikes:
    continue
...
finite = np.isfinite(spk_times)
spk_times, spk_clusters = spk_times[finite], spk_clusters[finite]
```

```python
good = np.isfinite(wt) & np.isfinite(wv)
...
gw = np.isfinite(wheel_t) & np.isfinite(wheel_v)
wheel_t, wheel_v = wheel_t[gw], wheel_v[gw]
...
safe = np.where(np.isfinite(xi), xi, times[0])
```

```python
def _worker(task):
    eid, probes, show = task
    try:
        res = load_session(eid, probes, show_processing=show)
    except Exception as exc:
        return {'eid': eid, 'skip': f'EXCEPTION {type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc()}
```

```python
if r.get('skip'):
    print(f"[{done}/{len(tasks)}] SKIP {r['eid']}: {r['skip']}", flush=True)
...
'skipped_sessions': skipped,
```

iii. Step 10 Check 1 and Check 5 document the two real defects found and fixed: the read-only
array returned by `trials.eval()` (which was silently skipping *every* session until an
explicit `.copy()` was added — found only because the driver prints a reason for each skip),
and the all-zero neural trials caused by trials lying past the last spike or inside a 2.07 s
drop-out. The AI's stated principle is to drop rather than impute: "17 sessions were dropped,
each for a concrete, reported reason", and after the fixes the verification log reads "Data
format is valid, no errors or warnings", so no warning had to be explained away.

## 10-a. What are the most time-consuming steps of the code?

i. Per-session timings are instrumented (`timings` dict for trials / spikes / behavior /
binning) and printed. The dominant cost is **loading the spike sorting** (`load_spike_sorting`
+ `merge_clusters`), measured at 2–6 s per session against 0.3 s for trials, 0.5–1 s for
behaviour and 0.3–1 s for binning — i.e. file I/O, since the two spike arrays of a probe run
to hundreds of megabytes. At the whole-run level the other notable cost is writing the 11.7 GB
pickle (17.5 s). Total: 3.5 min wall-clock with 32 workers, ~61 min serial.

ii.
```python
timings = {}
t0 = time.time()
... ssl.load_spike_sorting() ...
timings['spikes'] = time.time() - t0
...
print(f"[{done}/{len(tasks)}] {r['eid']} neurons={r['n_neurons']} "
      f"trials={r['n_trials_kept']}/{r['n_trials_raw']} t={r['total_time']:.1f}s "
      f"| elapsed {el/60:.1f} min, eta {el/done*(len(tasks)-done)/60:.1f} min", flush=True)
```

iii. Step 7 "Run Time Estimates" tabulates the per-step timings and the extrapolation to the
full dataset; Step 9 confirms the realised 3.5 min, "well inside the 15 min budget". The AI
addressed the I/O bottleneck not by making the load faster but by parallelising at the session
level (`ProcessPoolExecutor`, ~24× wall-clock) and by loading each probe's spike sorting
exactly once.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorised the two loops the reference ran per trial: behaviour interpolation is now
one `np.interp` over an `(n_trials, 100)` grid instead of a per-trial multiprocessing pool, and
the spike-window lookup is one `searchsorted` for all trials instead of `bincount2D` per trial.
What remains loop-based, and could still be vectorised, is the inner accumulation in
`bin_spikes_trials`: it loops over trials and calls `np.add.at`, which is a well-known slow
path; giving each spike a flat index of `(trial, cluster, bin)` and calling a single
`np.bincount` would remove the Python loop entirely and be several times faster. The final
assembly loop over kept trials (building the 2 × 100 input and 4 × 100 output arrays
trial-by-trial) is likewise per-trial rather than a single stacked allocation. Neither was
changed, and neither matters at the achieved 3.5 min runtime.

ii.
```python
    for k in range(n_trials):
        a, b = i0[k], i1[k]
        if b <= a:
            continue
        bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)
        np.clip(bi, 0, NBINS - 1, out=bi)
        ci = spike_clusters[a:b]
        np.add.at(flat[k], ci * NBINS + bi, 1.0)
```

```python
    for j, k in enumerate(kt):
        neural.append(binned[j])
        inp = np.empty((2, NBINS), dtype=np.float32)
        inp[0] = time_axis
        inp[1] = tib[k]
        inputs.append(inp)
        out = np.empty((4, NBINS), dtype=np.int64)
        ...
```

iii. Step 6 "Code inefficiencies identified and removed" claims the vectorisations that were
done: "The reference bins spikes with a `multiprocessing.Pool` **per trial** … Replaced with one
`searchsorted` for all trials plus a single `np.add.at` per trial — about 2 orders of magnitude
fewer Python-level calls"; "The reference calls `get_behavior_per_interval` (another per-trial
process pool) for each behaviour. Replaced with a single vectorised `np.interp`". The AI did
**not** flag the remaining `np.add.at` trial loop as a further vectorisation opportunity; its
stated justification for stopping was that the measured runtime was already far inside the
15-minute budget.

## 10-c. What processing does the code repeat multiple times?

i. Three things are repeated more often than necessary:
- `get_one()` (constructing a `ONE` client) and `BrainRegions()` (loading the Allen/Beryl
  region table) are called **inside `load_session`**, i.e. once per session — 459 times across
  the run — rather than once per worker process. The human reference avoids this with a
  `ProcessPoolExecutor(initializer=...)` that connects once per process.
- `sl.load_motion_energy` is attempted for the left camera and then re-attempted for the right
  whenever the left is absent (22 sessions), which is unavoidable given the fallback rule but
  does mean a failed load per such session.
- `br.acronym2acronym(..., mapping='Beryl')` is recomputed per session over that session's
  cluster list; the mapping itself is a fixed lookup.
- In `main`, the skipped-session list is rebuilt twice (once before and once after the
  neuron-count curation), and the output statistics loop re-concatenates the full output arrays
  per output dimension.

ii.
```python
def load_session(eid, probes, show_processing=False):
    ...
    one = get_one()
    br = BrainRegions()
```

```python
    kept = [e for e in kept if results[e].get('skip') is None]
    skipped = [(e, results[e]['skip']) for e in eids
               if results[e].get('skip') is not None]
```

iii. Not documented by the AI. Step 6 states the opposite emphasis — "Spike sorting is loaded
once per probe, and the trials/wheel/motion-energy objects once per session, via a single
shared `SessionLoader`" and "Parallelism is at the **session** level … which is the natural
grain and avoids the reference nested-pool overhead" — which is true of the expensive I/O, but
does not cover the per-session reconstruction of the ONE client and the region table. The cost
is small relative to the 2–6 s spike load, which is presumably why it was not noticed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of work produce values that are never consumed:
- **Dead computation**: in `main`, `allout` concatenates and reshapes *every* output value of
  the entire dataset (≈ 75 M int64, ~600 MB) and is then never used — the per-output loop
  immediately below recomputes what it needs into `vals`.
- **Behaviour computed for discarded trials**: `interp_behavior` and `tertile_bins` are run over
  **all** trials of a session, including the ~40% that the trial mask removes, and only
  `ws_cls[k]`/`wm_cls[k]` for kept trials are stored. The spike binning, by contrast, is
  deliberately restricted to kept trials ("Only the kept trials are binned"), so the behaviour
  path is inconsistent with the stated optimisation.
- **Tertile thresholds from a slightly stale mask**: the thresholds are computed with
  `keep_trial` *before* the `probabilityLeft ∈ {0.2,0.5,0.8}` filter narrows it, so a few
  to-be-dropped trials contribute to the percentiles (a harmless but avoidable inconsistency).
- **Unused cluster metadata**: `merge_clusters(...).to_df()` materialises the full cluster table
  (metrics, depths, uuids, channels, histology) for every cluster of every probe, of which only
  `label` and `acronym` are ever read.
- **Metadata never read by the decoder**: `kept_trial_idx` per session, `n_units_all`,
  `n_well_isolated`, the per-session tertile thresholds, the per-session region lists and
  `bin_time_axis_s` are all stored in the pickle but unused downstream (they were added
  deliberately as provenance for the sanity checks).

ii.
```python
        allout = np.concatenate([o.ravel() for s in data['output'] for o in s]).reshape(-1)
        for i, name in enumerate(OUTPUT_NAMES):
            vals = np.concatenate([o[i] for s in data['output'] for o in s])
```

```python
    ws_vals, ws_ok = interp_behavior(wheel_t, wheel_v, t_beg)   # all trials
    wm_vals, wm_ok = interp_behavior(whisker_t, whisker_v, t_beg)
    ...
    ws_cls, ws_thr = tertile_bins(ws_vals, keep_trial)          # classes for all trials
    ...
    kt = np.flatnonzero(keep_trial)
    binned = bin_spikes_trials(spk_times_k, spk_clusters_k, n_neurons, t_beg[kt])  # kept only
```

```python
        session_info.append({
            ...
            'kept_trial_idx': r['kept_trial_idx'],
            'wheel_speed_tertiles': r['wheel_thresholds'],
            'whisker_me_tertiles': r['whisker_thresholds'],
            'regions': sorted(set(r['regions'].tolist())),
        })
```

iii. Not documented as waste by the AI. Step 6 claims "Only the kept trials are binned, so no
work is done for trials that will be discarded", which is true of the spikes but not of the
behaviour. The extra metadata is justified in Step 10 Check 2: "To make these checks exact
rather than heuristic I added `kept_trial_idx` to each session's `session_info`, which also
serves as provenance back to the raw trials table" — so that part is deliberate and cheap. The
`allout` line appears to be a leftover from an earlier version of the statistics block.
