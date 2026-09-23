# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates the dataset from the *reference code's own freeze file*,
`/app/code/code_zhang2025/data/bwm_release.csv` (459 unique `eid`s, 699 insertions, 139
subjects), exactly as `code_zhang2025/src/0_data_caching.py` does. That CSV supplies, per row,
the `eid`, `pid`, `probe_name`, `subject` and `lab`, so no `one.search` / `one.eid2pid` call is
needed. Every actual *data* read then goes through the ONE API and the `brainbox` loaders: a
per-worker `ONE(base_url=..., tables_dir='/app/data/one_cache/Brainwidemap')` client (offline,
resolved from the staged cache + cached Alyx REST store), `SpikeSortingLoader` once per probe
insertion, and `SessionLoader` once per session for trials, wheel and camera motion energy.
Sessions are processed independently in a `multiprocessing` fork pool (default 24 workers,
28 used for the full run), each worker holding its own `ONE` client and `BrainRegions` object.
No file under `/app/data` is opened directly.

ii.
```python
BWM_FREEZE = '/app/code/code_zhang2025/data/bwm_release.csv'
ONE_TABLES_DIR = '/app/data/one_cache/Brainwidemap'

def get_one():
    """Return this process's ONE client (created once, offline-friendly)."""
    global _ONE
    if _ONE is None:
        from one.api import ONE
        _ONE = ONE(base_url=ONE_BASE_URL, silent=True, tables_dir=ONE_TABLES_DIR)
    return _ONE
```

```python
bwm = pd.read_csv(BWM_FREEZE, index_col=0)
eids = list(dict.fromkeys(bwm.eid.tolist()))  # preserve freeze order
...
for i, eid in enumerate(eids):
    rows = bwm[bwm.eid == eid]
    tasks.append((eid, rows[['pid', 'probe_name']].copy(),
                  rows.subject.iloc[0], rows.lab.iloc[0], i < n_show))
```

```python
def load_session_spikes(one, eid, probe_rows):
    from brainbox.io.one import SpikeSortingLoader
    from utils.ibl_data_utils import merge_probes
    spikes_list, clusters_list = [], []
    for _, row in probe_rows.iterrows():
        ssl = SpikeSortingLoader(pid=row.pid, one=one, eid=eid, pname=row.probe_name)
        sp, cl, ch = ssl.load_spike_sorting()
        cl_df = SpikeSortingLoader.merge_clusters(sp, cl, ch, compute_metrics=False).to_df()
        cl_df['pid'] = row.pid
        spikes_list.append(sp)
        clusters_list.append(cl_df)
    return merge_probes(spikes_list, clusters_list)
```

```python
def load_trials(one, eid):
    from brainbox.io.one import SessionLoader
    from utils.ibl_data_utils import load_trials_and_mask
    sess_loader = SessionLoader(one=one, eid=eid)
    trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                        sess_loader=sess_loader)
    return sess_loader, trials, mask.to_numpy()
```

iii. From CONVERSION_NOTES Step 1/5/6: the relevant reference pipeline is
`0_data_caching.py` → `prepare_data`, which itself starts from `bwm_release.csv`, so the AI
"reuses code from the reference code base where appropriate" (it literally imports
`merge_probes` and `load_trials_and_mask` from `utils/ibl_data_utils.py`). It notes that taking
the pids from the freeze file gives the *same mapping* as `one.eid2pid` "without a REST
round-trip", and that the one reference call it drops —
`spike_loader.raw_electrophysiology(band="ap", stream=True).fs` — only records a metadata field
and requires streaming raw binary from the remote server, which is impossible offline.
Parallelism is moved "up one level" to one worker per session so the reference's per-session
multiprocessing pools are created once for the whole run instead of ~2000 times.

## 1-b. How are the data split into subjects?

i. The subject name is read straight off the freeze table (`bwm_release.csv` column `subject`)
for each `eid` and carried through the per-session result dict. At assembly, `subjects` is the
sorted set of unique subject names over the *converted* sessions and `subject_idx` is that
session's index into the list. 136 subjects survive (3 of the 139 lose their only session).
`lab` is also carried through and stored in `metadata['session_info']`.

ii.
```python
tasks.append((eid, rows[['pid', 'probe_name']].copy(),
              rows.subject.iloc[0], rows.lab.iloc[0], i < n_show))
```

```python
subjects = sorted({r['subject'] for r in ok})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_index[r['subject']] for r in ok], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: "`bwm_release.csv` `subject` → `subjects`,
`subject_idx`, unique sorted list". The subject identifier is already unique and authoritative
in the release table, so nothing has to be derived or parsed from paths.

## 1-c. How are the data split into sessions?

i. A session is the natural unit of the release: one `eid` per session. The AI takes the unique
`eid`s from the freeze file in freeze order (`dict.fromkeys`), builds one task per `eid`, groups
that session's insertions with `bwm[bwm.eid == eid]`, and processes each `eid` independently.
Results are re-sorted back into freeze order after the unordered pool finishes, so the session
ordering is deterministic. `--sample` takes the first 2 `eid`s. Sessions that fail are dropped
and enumerated (18 of 459 → 441 kept).

ii.
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))  # preserve freeze order
if args.sample:
    eids = eids[:2]
```

```python
ok = [r for r in results if 'error' not in r]
failed = [r for r in results if 'error' in r]
order = {eid: i for i, eid in enumerate(eids)}
ok.sort(key=lambda r: order[r['eid']])
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: "sessions from the 459-eid BWM freeze; dropped if
no whisker motion energy is available, if no neuron survives curation, or if fewer than 2
trials survive." Step 9 records the expectation that this lands near the 433 sessions the method
paper reports, and the achieved value of 441 is explicitly checked against 433–459.

## 1-d. How are the data split into trials?

i. The split is given by the data: `load_trials_and_mask` (imported verbatim from the reference
code) returns the ONE trials table with one row per trial, and the AI works with row indices
into that table throughout. Trial *k*'s 2 s window is `stimOn_times[k] + (-0.5, +1.5)`. The row
indices of every retained trial are stored in `metadata['session_info'][s]['trial_idx']` so the
conversion can be audited back against the raw table.

ii.
```python
stim_on_all = trials[PARAMS['align_time']].to_numpy()
...
cand = np.nonzero(trials_mask)[0]
interval_begs = stim_on_all[cand] + PARAMS['time_window'][0]
```

```python
'trial_idx': trial_idx.astype(np.int32),
```

iii. No decision was needed — the trials table is already one row per trial. The AI's stated
motivation for keeping `trial_idx` (Step 10, Iteration 2) is auditability: "so the conversion
can be re-derived and checked against the raw ONE objects".

## 1-e. How are trials filtered based on quality controls?

i. Four layers, applied in order:
1. **The reference trial mask, called verbatim**:
   `load_trials_and_mask(one, eid, max_trial_len=10.0, sess_loader=...)` with the reference's
   defaults — reaction time (`firstMovement_times − stimOn_times`) must lie in [0.08, 2.0] s;
   no NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`,
   `firstMovement_times`, `feedbackType`; `feedback_times − goCue_times ≤ 10 s`; `choice != 0`
   (no-go trials dropped). This retains 66.0 % of trials (188,020 of 285,031).
2. **Behavioural coverage**, reproducing `get_behavior_per_interval`'s four skip rules for both
   the wheel and the camera: no samples in the window, NaN interval bounds, first sample more
   than one bin after the window start ("starts too late"), last sample more than one bin before
   the window end ("ends too early"). 86 trials dropped.
3. **NaN in the behavioural trace** — the reference passes `allow_nans=True` and imputes later;
   the AI drops the trial instead.
4. **Zero-spike trials**: trials in which the entire retained population emits no spike over the
   whole 2 s window are dropped as recording dropouts (16 trials, 0.009 %).
Sessions with < 2 surviving trials are dropped (1 session). Final: 187,918 trials, mean 426 per
session (min 125, max 1445).

ii.
```python
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
```

```python
    if len(v) == 0:
        reasons[k] = 'target data not present'; continue
    if np.isnan(interval_begs[k]) or np.isnan(interval_ends[k]):
        reasons[k] = 'bad interval data'; continue
    if np.abs(interval_begs[k] - t[0]) > binsize:
        reasons[k] = 'target data starts too late'; continue
    if np.abs(interval_ends[k] - t[-1]) > binsize:
        reasons[k] = 'target data ends too early'; continue
    if np.any(np.isnan(v)):
        # the reference keeps these (allow_nans=True) ... a categorical target cannot be
        # imputed without inventing a class, so the trial is dropped
        reasons[k] = 'nans in target data'; continue
```

```python
keep_trial = np.ones(len(cand), dtype=bool)
for name in beh_vals:
    keep_trial &= beh_good[name]
if keep_trial.sum() < 2:
    return {'eid': eid, 'error': f'only {int(keep_trial.sum())} trials with valid behaviour'}
```

```python
has_spikes = binned.any(axis=(1, 2))
n_zero_spike = int((~has_spikes).sum())
if n_zero_spike:
    binned = binned[has_spikes]
    trial_idx = trial_idx[has_spikes]
    interval_begs = interval_begs[has_spikes]
    keep_trial[np.nonzero(keep_trial)[0][~has_spikes]] = False
```

iii. Step 4/5/10: the mask is "the reference call verbatim" and matches the data paper's own
trial criteria. The NaN deviation is justified because "the target format forbids NaN and
imputing a value then discretising it would fabricate labels". The zero-spike rule was added in
Step 10 Iteration 1 in response to 38 `all neural data is zero` warnings from the decoder's
verifier: "trials in which the entire population records zero spikes over the whole 2 s window
are dropped as recording dropouts". The re-run produced zero warnings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from `SpikeSortingLoader.load_spike_sorting()`,
for every probe insertion of the session, merged with the reference's `merge_probes`. The
cluster table produced by `SpikeSortingLoader.merge_clusters(..., compute_metrics=False)`
supplies `label` (quality) and `acronym` (anatomy), which select which clusters' spikes are kept
and provide `brain_region_idx`, but the array itself is built only from spike times and spike
cluster ids.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
cl_df = SpikeSortingLoader.merge_clusters(sp, cl, ch, compute_metrics=False).to_df()
...
return merge_probes(spikes_list, clusters_list)
```

```python
spike_cl = remap[spikes['clusters']]
sel = spike_cl >= 0
spike_times = np.ascontiguousarray(spikes['times'][sel])
spike_cl = np.ascontiguousarray(spike_cl[sel])
```

iii. Step 1/5: this is `prepare_data`'s `neural_dict = {'spike_times': spikes['times'],
'spike_clusters': spikes['clusters'], 'cluster_regions': clusters['acronym']}` reproduced with
the same loader calls. Probes are merged because, per `merge_probes`' docstring (quoted in the
notes), "data from the probes recorded in the same session are not statistically independent as
they have the same underlying behaviour", and the data paper likewise combines neurons across
probes within a session.

## 2-b. How is the `neural` data processed?

i. Spikes of the retained units are counted into 100 non-overlapping 20 ms bins spanning
`stimOn − 0.5 s` to `stimOn + 1.5 s`, per trial. Bin *k* covers
`[t_beg + 0.02k, t_beg + 0.02(k+1))`; a spike exactly on the closing edge is clipped into the
last bin. Counts are stored as **raw spike counts** in float32 — no conversion to Hz, no
smoothing, no z-scoring/standardisation. Cluster ids are remapped to a contiguous
`0…n_neurons−1` after quality selection; the two probes of a session are pooled into one
population by `merge_probes`, which also re-sorts all spikes by time so `searchsorted` can slice
trial windows. The implementation replaces the reference's per-trial `bincount2D` call inside a
per-session process pool with one vectorised `searchsorted` over all trial edges plus a single
flat `np.bincount` per trial.

ii.
```python
def bin_spikes(spike_times, spike_clusters, n_neurons, interval_begs, binsize=0.02,
               n_bins=N_BINS):
    n_intervals = len(interval_begs)
    out = np.zeros((n_intervals, n_neurons, n_bins), dtype=np.float32)
    interval_ends = interval_begs + n_bins * binsize
    i0s = np.searchsorted(spike_times, interval_begs, side='left')
    i1s = np.searchsorted(spike_times, interval_ends, side='left')
    for k in range(n_intervals):
        i0, i1 = i0s[k], i1s[k]
        if i1 <= i0:
            continue
        rel = spike_times[i0:i1] - interval_begs[k]
        bin_idx = np.floor(rel / binsize).astype(np.int64)
        np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
        flat = spike_clusters[i0:i1].astype(np.int64) * n_bins + bin_idx
        counts = np.bincount(flat, minlength=n_neurons * n_bins)
        out[k] = counts.reshape(n_neurons, n_bins)
    return out
```

```python
'neural_units': 'spike counts per 20 ms bin',
```

iii. Step 5 Key Decision 4: "raw spike counts stored as float32 (matching the reference's
`ubyte` CSR cache). Standardisation is a model-side step in the reference
(`standardize_spike_data` in `data_loader_utils.py`) and the provided decoder does its own SVD
projection." Step 10 Check 3 argues the fast binning is "algebraically the same operation" as
`bincount2D(t, c, xbin=0.02, xlim=[t_beg, t_end])[:, :100]`, and Check 2 verifies it against an
independent `np.histogram` re-derivation (`allclose` on all 100 bins for 6 random
(trial, neuron) pairs in each of 4 sessions).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three criteria:
* **`clusters.label >= 1`** — the IBL `label` is the fraction of the three RIGOR single-unit
  metrics passed (amplitude > 50 µV, noise cut-off < 20 µV, refractory-period violation), so
  `>= 1` is exactly the data paper's "well-isolated neuron". This is also the reference's
  `load_spiking_data(qc=1)` branch (the reference's `prepare_data` itself calls it with
  `qc=None`, i.e. keeps everything).
* **Beryl acronym ∉ {`root`, `void`}** — grey matter only, as in the reference's
  `data_loader_utils.py` line `unique_regions = [roi for roi in ... if roi not in ['root','void']]`
  and the data paper's "restricted to regions that were designated grey matter".
* **≥ 5 retained neurons per session** — otherwise the session is dropped (3 sessions).

Result: 73,010 well-isolated units over 672 probes (108.6/probe vs the paper's 108), of which
62,757 survive the grey-matter cut (mean 142.3/session).

ii.
```python
NON_REGION_ACRONYMS = ('root', 'void')
MIN_NEURONS_PER_SESSION = 5

def select_neurons(clusters):
    br = get_brain_regions()
    beryl_all = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
    good = clusters['label'].to_numpy() >= 1
    in_brain = ~np.isin(beryl_all, NON_REGION_ACRONYMS)
    keep = good & in_brain
    keep_idx = np.nonzero(keep)[0]
    return keep_idx, beryl_all[keep_idx], int(good.sum())
```

```python
keep_idx, beryl, n_good = select_neurons(clusters)
n_neurons = len(keep_idx)
if n_neurons < MIN_NEURONS_PER_SESSION:
    return {'eid': eid, 'error': f'only {n_neurons} well-isolated grey-matter neurons '
                                 f'(< {MIN_NEURONS_PER_SESSION})'}
```

iii. Step 4 contains an explicit four-point argument for deviating from the reference code's
`qc=None`: (1) the data paper defines its analysis population as the 75,708 well-isolated
neurons and excludes the rest as multi-unit activity; (2) the reference code exposes exactly
this switch and `merge_probes` is documented against `brainwidemap.load_good_units`;
(3) feasibility — all 621,733 units would be ≈120 GB of float32, vs ≈8 GB for good units;
(4) the measured cost from the reference's own example decoder outputs is small (choice AUC
0.896 → 0.869). Dropping `root`/`void` is justified as "not brain regions" that "would pollute
`brain_regions`". The ≥5-neuron floor is the data paper's per-region criterion applied at the
session level "because here the decoded population is the whole session", added in Step 10 to
eliminate the all-zero-neural warnings. The AI explicitly declines to apply the paper's other
region criteria (≥5 neurons per region, region in ≥2 sessions) because "they exist to make
region-wise statistical maps comparable, whereas here every session is decoded from its whole
population".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By subtraction on the shared session clock. All IBL streams (spike times, trial event times,
wheel timestamps, camera frame times) are already expressed in seconds on one synchronised
timebase, so alignment is just `t_beg = stimOn_times + (−0.5)` and binning spike times relative
to `t_beg`. No resampling, warping or cross-stream synchronisation is performed. The alignment
event is `trials.stimOn_times`, recorded in `metadata['temporal_alignment_event']` with
`off_start = −0.5`, `off_end = +1.5`.

ii.
```python
PARAMS = {'interval_len': 2.0, 'binsize': 0.02, 'single_region': False,
          'align_time': 'stimOn_times', 'time_window': (-0.5, 1.5)}
...
stim_on_all = trials[PARAMS['align_time']].to_numpy()
interval_begs = stim_on_all[cand] + PARAMS['time_window'][0]
```

```python
rel = spike_times[i0:i1] - interval_begs[k]
bin_idx = np.floor(rel / binsize).astype(np.int64)
```

```python
'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
'off_start': PARAMS['time_window'][0],
'off_end': PARAMS['time_window'][1],
```

iii. Step 3/4: "Alignment: `stimOn_times`, window `(-0.5, +1.5) s` ⇒ 2 s interval" — the
reference `params` verbatim, the configuration in the method paper's Results, and the alignment
mandated by the Decoder Task. The notes record that the STAR Methods use `firstMovement_times`
for the dynamic behaviours but that the Decoder Task fixes stimulus onset. Alignment was
verified empirically: the population PSTH in `processing_<eid>.png` is flat before 0 and jumps
from 6.5 Hz to 11.7 Hz peaking at +0.25 s, and per-timepoint choice decoding peaks at +0.30 s
(the median first-movement latency).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, `N_BINS = ceil(2.0 / 0.02) = 100` per trial, uniform across every trial and
session; `metadata['time_bin_size'] = 20.0` ms. Spikes are binned once directly from spike
times into this grid — there is no rebinning of a pre-binned array and no interpolation of the
neural data. The two continuous behaviours are resampled onto the same 100-point grid (right
bin edges). The AI tested 100 ms rebinning as a diagnostic (`cache/timepoint_analysis.py`,
choice 0.617 → 0.654) but kept 20 ms in the delivered dataset.

ii.
```python
PARAMS = {'interval_len': 2.0, 'binsize': 0.02, ...}
N_BINS = int(np.ceil(PARAMS['interval_len'] / PARAMS['binsize']))  # 100
```

```python
'time_bin_size': PARAMS['binsize'] * 1000.0,  # ms
'n_timepoints': N_BINS,
```

iii. Step 3/4: the reference code sets `'binsize': 0.02` and the method paper's Results state
"Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time
steps". The notes address the STAR Methods' mention of 50 ms for choice/prior: "a single binning
has to serve all four outputs here, and only 20 ms is implemented in the released code."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. `trials.stimOn_times` — the alignment event — combined with the fixed window/bin constants.
The value is not read from any raw array per trial: it is the deterministic grid of the **right
edge** of each of the 100 spike bins relative to onset, i.e. `−0.5 + 0.02·(k+1)` for
k = 0…99, spanning [−0.48, +1.50] s, identical for every trial and session.

ii.
```python
stim_on_all = trials[PARAMS['align_time']].to_numpy()
interval_begs = stim_on_all[cand] + PARAMS['time_window'][0]
```

```python
bin_times = PARAMS['time_window'][0] + PARAMS['binsize'] * np.arange(1, N_BINS + 1)
```

iii. Step 5 mapping table: "bin right-edge time relative to `stimOn_times` → `input[s][k][0,:]`,
`−0.5 + 0.02·(k+1)` ∈ [−0.48, 1.50]", with the reference counterpart listed as
"`get_behavior_per_interval`'s `x_interp` grid" — i.e. the AI deliberately re-uses the exact
grid the reference code uses to resample behaviour, so the input, the behaviour outputs and the
spike bins all share one definition of "time".

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the constant grid. The 100-vector `bin_times` is computed once and
broadcast into every trial's input array as row 0, cast to float32. It is continuous-valued and
time-varying, as the Decoder Task specifies (it is not represented as a binary event marker,
because the specification asks for "time since stimulus onset" as a continuous time-varying
input).

ii.
```python
bin_times = PARAMS['time_window'][0] + PARAMS['binsize'] * np.arange(1, N_BINS + 1)
tib = tib_all[trial_idx].astype(np.float32)
inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
inputs[:, 0, :] = bin_times[None, :]
inputs[:, 1, :] = tib[:, None]
```

iii. Step 5 Key Decision 6 ("Everything time-varying"): both inputs are stored as `(d, 100)`
per trial, per-trial quantities broadcast across time, "as the format instructions prefer". The
grid is documented in metadata as "time (s) of the right edge of each 20 ms bin relative to
stimulus onset, −0.48 … 1.50".

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural binning grid, offset by one bin width. Spike bin *k* covers
`[t_beg + 0.02k, t_beg + 0.02(k+1))` and `input[0, k] = −0.5 + 0.02(k+1)` is the closing edge of
that same bin, both measured from the same `stimOn_times`. Column *k* of the neural array, of
the input array and of both continuous outputs therefore all refer to the same 20 ms interval —
the two behavioural traces are sampled at exactly this same right-edge grid, so the alignment
convention is uniform across all three streams.

ii.
```python
i0s = np.searchsorted(spike_times, interval_begs, side='left')
i1s = np.searchsorted(spike_times, interval_ends, side='left')
...
bin_idx = np.floor(rel / binsize).astype(np.int64)
```
```python
bin_times = PARAMS['time_window'][0] + PARAMS['binsize'] * np.arange(1, N_BINS + 1)
```
```python
x = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
vals[k] = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

iii. Step 3 Processing Details: "Spike bin `k` spans `[stimOn − 0.5 + 0.02k, stimOn − 0.5 +
0.02(k+1))` ... Continuous behaviour: linearly interpolated onto `t_beg + 0.02·(k+1)`, i.e. the
**right edge** of each spike bin" — the reference's own convention. Step 10 Check 5 asserts
that `input[0]` equals the bin-edge grid on the first and last trial of every session.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. `trials.probabilityLeft` alone. The trials table carries no block identifier, so blocks are
recovered as maximal runs of constant `probabilityLeft`; a change of value (including to or from
NaN) starts a new block.

ii.
```python
tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

```python
p = np.asarray(probability_left, dtype=float)
# a change (including to/from NaN) starts a new block
changed = np.ones(len(p), dtype=bool)
if len(p) > 1:
    same = (p[1:] == p[:-1]) | (np.isnan(p[1:]) & np.isnan(p[:-1]))
    changed[1:] = ~same
block_id = np.cumsum(changed) - 1
```

iii. Step 5 lists this input as "new; required by Decoder Task" — the reference code has no
decoder-input concept. The block structure described in the papers ("After an initial 90
unbiased trials … blocks of 20–100 trials, empirical mean 51") is the basis for the sanity
check that the first block is exactly 90 trials long.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The 0-based position of each trial within its block: `arange(n) − (index of the first trial
of that block)`. Crucially it is computed on the **unfiltered** trials table and only then
indexed by the retained rows, so a trial dropped by curation still advances the counter and the
value is the animal's true position in the block. The per-trial integer is cast to float32 and
broadcast across all 100 bins. Observed range over the full dataset: [0, 98].

ii.
```python
def trial_number_in_block(probability_left):
    """0-based index of each trial within its constant-``probabilityLeft`` block.

    Computed on the *unfiltered* trials table, because the block structure is a
    property of the experiment, not of which trials survive curation.
    """
    ...
    block_id = np.cumsum(changed) - 1
    first_of_block = np.zeros(block_id[-1] + 1, dtype=np.int64)
    starts = np.nonzero(changed)[0]
    first_of_block[:] = starts
    return np.arange(len(p)) - first_of_block[block_id]
```

```python
tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
...
tib = tib_all[trial_idx].astype(np.float32)
inputs[:, 1, :] = tib[:, None]
```

iii. The docstring states the rationale directly ("the block structure is a property of the
experiment, not of which trials survive curation"). Step 10 Check 2 re-derives the value with a
plain Python loop over the raw trials table (`allclose`), confirms the first unbiased block is
exactly 90 trials in every session checked, and Check 5 asserts the value is constant within a
trial and lies in the expected range. The `--show-processing` figure overlays
`probabilityLeft` with the trial-in-block sawtooth to show it resets exactly at block
boundaries.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `trials.choice`, which takes values +1 (left report), −1 (right report) and 0 (no response).
No-response trials never reach this point because `load_trials_and_mask(exclude_nochoice=True)`
has already removed them.

ii.
```python
choice = trials['choice'].to_numpy()[trial_idx]
# choice == +1 is a LEFT report, choice == -1 a RIGHT report
choice_lbl = ((1 - choice) / 2).astype(np.int64)
```

iii. Step 2/4: the sign convention was not assumed but verified empirically — on 100 %-contrast
trials a left stimulus gave `choice = +1` in 71/72 trials and a right stimulus `choice = −1` in
61/63, so "`choice == +1` ⇒ **left** report ⇒ label 0; `choice == −1` ⇒ **right** report ⇒
label 1", which is the mapping the Decoder Task requires (left = 0, right = 1).

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the arithmetic recoding `(1 − choice)/2`: +1 → 0 (left), −1 → 1 (right). The per-trial
label is then broadcast across all 100 time bins and stored as int64 in row 0 of the output
array. `output_values[0] = ['left', 'right']`. Observed distribution: 0.508 left / 0.492 right.

ii.
```python
outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int64)
outputs[:, 0, :] = choice_lbl[:, None]
```
```python
OUTPUT_VALUES = [['left', 'right'], ...]
'choice': 'trials.choice: +1 (left report) -> 0, -1 (right report) -> 1',
```

iii. Step 5 Key Decision 6: per-trial outputs are broadcast over time so that every output is
time-varying, "as the format instructions prefer". Step 10 Check 2 re-derives choice for every
retained trial of 4 sessions from the raw trials table with `np.where(choice == 1, 0, 1)` and
gets an exact match; Step 12 confirms both classes are present in every session (per-session
left fraction 5–95 % range 0.35–0.65).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. `trials.probabilityLeft`, the block prior held constant within a block, which takes exactly
the three values 0.2, 0.5 and 0.8.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()[trial_idx]
prior_lbl = np.full(n_trials, -1, dtype=np.int64)
for val, lbl in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_lbl[np.isclose(pleft, val)] = lbl
if np.any(prior_lbl < 0):
    bad = np.unique(pleft[prior_lbl < 0])
    return {'eid': eid, 'error': f'unexpected probabilityLeft values {bad}'}
```

iii. Step 5 maps `trials.probabilityLeft` to `output[1]` and names the reference counterpart as
`bin_behaviors`' `block` variable, which is literally `trials_df['probabilityLeft']`. The AI
uses the same field the reference uses as its "block" decoding target.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A three-way recoding 0.2 → 0, 0.5 → 1, 0.8 → 2, done with `np.isclose` to avoid float
equality problems, with a hard error if any trial's value falls outside the three (never
triggered). The label is broadcast across all 100 bins into row 1 of the output array. Note the
unbiased 0.5 block is **kept** (`exclude_unbiased=False`, the reference default), so all three
classes are present: 0.417 / 0.140 / 0.442 — the 0.140 reflecting the ~90 unbiased trials at the
start of a ~646-trial session.

ii.
```python
outputs[:, 1, :] = prior_lbl[:, None]
```
```python
['0.2 (right block)', '0.5 (unbiased)', '0.8 (left block)'],
```

iii. The mapping is dictated by the Decoder Task ("0.2 -> 0, 0.5 -> 1, 0.8 -> 2"). Step 9
cross-checks the observed class fractions against the expectation derived from the block
structure described in the papers ("90 unbiased of ~646 trials ⇒ ~0.14 at p=0.5, rest split
~evenly") and finds agreement. Step 10 Check 2 re-derives the labels independently with
`np.select` on the raw `probabilityLeft` (`allclose`).

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()`, i.e. `_ibl_wheel.position` / `_ibl_wheel.timestamps` processed
by the loader into a 1 kHz uniformly sampled position plus a smoothed `velocity`; wheel speed is
`|velocity|`. This is exactly the reference's `load_target_behavior(one, eid, 'wheel-speed')`.

ii.
```python
sess_loader.load_wheel()
beh['wheel-speed'] = {
    'times': sess_loader.wheel['times'].to_numpy(),
    'values': np.abs(sess_loader.wheel['velocity'].to_numpy()),
    'source': 'wheel.velocity (abs)',
}
```

iii. Step 1/5: the docstring of `load_continuous_behaviors` states "Same sources as
`ibl_data_utils.load_target_behavior`: wheel speed = |velocity| from
`SessionLoader.load_wheel` (1 kHz, Gaussian-smoothed)". The interpolation-to-1 kHz and the
velocity filtering are not the AI's choices — they are what `SessionLoader` does internally, so
the reference and this conversion see the same trace.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) `SessionLoader` interpolates wheel position to a uniform 1 kHz grid and
differentiates it into a smoothed velocity; the AI takes `np.abs` of that. (2) Per trial, the
samples strictly inside the 2 s window are selected with `searchsorted` (`side='right'` at the
start, `side='left'` at the end — the reference's sides) and linearly interpolated with
`scipy.interp1d(..., fill_value='extrapolate')` onto `linspace(t_beg + 0.02, t_end, 100)`, the
right edges of the spike bins. (3) The resulting `(n_trials, 100)` float array is discretised
into three classes (see 7-c). Values are in rad/s; no normalisation or smoothing beyond what
`SessionLoader` applies.

ii.
```python
interval_ends = interval_begs + n_bins * binsize
idxs_beg = np.searchsorted(times, interval_begs, side='right')
idxs_end = np.searchsorted(times, interval_ends, side='left')
...
    x = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
    vals[k] = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

iii. Step 6 table: `interpolate_behavior` is documented as having "same slicing (`searchsorted`
right/left), same target grid `linspace(t_beg+bin, t_end, 100)`, same four skip rules" as
`get_behavior_per_interval`, with the only deviation being `allow_nans=False`. Step 10 Check 2
re-derives the trace with a *different* interpolator (`np.interp`) and finds 99.99–100 % of bins
identical.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Equal-frequency tertiles computed **per session**: the 1/3 and 2/3 quantiles of *all* binned
wheel-speed values pooled over every retained trial × every bin of that session, then
`np.digitize` into {0, 1, 2} = low / medium / high. A degenerate fallback handles heavily tied
distributions (if `q1 == q2`, digitize on the distinct quantile values that exist). Thresholds
are stored per session in `metadata['session_info'][s]['wheel_speed_tertiles']`. Achieved
distribution: 0.333 / 0.333 / 0.333.

ii.
```python
TERTILES = (1.0 / 3.0, 2.0 / 3.0)

def discretize_tertiles(values):
    flat = values.reshape(-1)
    q1, q2 = np.quantile(flat, TERTILES)
    if not (q1 < q2):
        edges = np.unique([q1, q2])
        labels = np.digitize(values, edges, right=False)
        return labels.astype(np.int64), (float(q1), float(q2))
    labels = np.digitize(values, [q1, q2], right=False).astype(np.int64)
    return labels, (float(q1), float(q2))
```

```python
ws_lbl, ws_thr = discretize_tertiles(ws)
outputs[:, 2, :] = ws_lbl
```

iii. Step 5 Key Decision 5: "per-session tertiles … Per-session rather than global because
whisker motion energy is in arbitrary units that depend on camera, illumination and ROI size and
is therefore not comparable across sessions; tertiles because the Decoder Task asks for 3 bins
and balanced accuracy is the evaluation metric, so equal-frequency bins are the natural choice."
The thresholds are computed once per session over all retained trials rather than per trial, so
within-trial dynamics are preserved. The `--show-processing` figure overlays the tertile lines
on the continuous trace and the resulting class trace to show the discretisation is correct.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is evaluated at the identical 100-point grid used for the neural bins and the time input —
the right edge of each 20 ms spike bin, measured from the same `stimOn_times` — so column *k* of
the wheel-speed output and column *k* of the neural array describe the same 20 ms interval. The
wheel timestamps are already on the same session clock as the spikes, so no further
synchronisation is done. Trials whose window is not fully spanned by wheel samples (gap > one
bin at either edge) are dropped rather than extrapolated.

ii.
```python
interval_begs = stim_on_all[cand] + PARAMS['time_window'][0]
...
v, g, r = interpolate_behavior(d['times'], d['values'], interval_begs,
                               PARAMS['binsize'], N_BINS)
```
```python
x = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
```

iii. Step 3: behaviour is interpolated onto "the **right edge** of each spike bin", the
reference's convention. Verified in Step 7 ("wheel speed raw 1 kHz trace overlaid with the
interpolated 20 ms samples — they coincide; speed is ≈0 before onset and rises ~0.15 s after
it") and Step 10 Check 5 ("the all-trial wheel-speed class image is dark (class 0) before
stimulus onset and bright afterwards, with no trial-to-trial jitter in the transition").

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with its frame times `_ibl_<side>Camera.times`, loaded through
`SessionLoader.load_motion_energy(views=[view])` and read from the `whiskerMotionEnergy` column.
The **left** camera (≈60 Hz) is used when available and the **right** camera (≈150 Hz) as a
fallback; the camera actually used is recorded per session in
`metadata['session_info'][s]['whisker_source']` (7 of 441 sessions used the right camera).
Sessions with neither are skipped (14).

ii.
```python
me = None
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        df = sess_loader.motion_energy[cam]
        me = {'times': df['times'].to_numpy(),
              'values': df['whiskerMotionEnergy'].to_numpy(),
              'source': f'{cam}.ROIMotionEnergy'}
        break
    except Exception:
        continue
if me is None:
    raise RuntimeError('no whisker motion energy available (left or right camera)')
```

iii. This mirrors the reference's `bin_behaviors`, which tries
`load_target_behavior(..., 'left-whisker-motion-energy')` and falls back to the right camera if
the load sets `skip`. Step 5 records the mapping as "`leftCamera.ROIMotionEnergy` (fallback
`rightCamera`)".

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is used as-is — no filtering, normalisation or unit
conversion. It goes through exactly the same `interpolate_behavior` path as the wheel: samples
strictly inside the 2 s window, linear `interp1d` with extrapolation onto the 100 right-edge bin
times, the four reference coverage checks, plus the NaN rejection, then per-session tertile
discretisation.

ii.
```python
for name, d in beh.items():
    v, g, r = interpolate_behavior(d['times'], d['values'], interval_begs,
                                   PARAMS['binsize'], N_BINS)
    beh_vals[name], beh_good[name], beh_reasons[name] = v, g, r
```
```python
me = beh_vals['whisker-motion-energy'][keep_trial]
me_lbl, me_thr = discretize_tertiles(me)
```

iii. Step 5/6: the same reference function (`get_behavior_per_interval`) governs both
behavioural streams, so one implementation serves both. The AI notes that motion energy "is in
arbitrary units that depend on camera, illumination and ROI size", which is why the
discretisation is per session rather than global — this also makes the left/right-camera
fallback harmless.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to wheel speed: per-session 1/3 and 2/3 quantiles of all binned values pooled
over the session's retained trials and bins, `np.digitize` into {0, 1, 2} = low / medium / high,
with the same degenerate-distribution fallback. Thresholds stored as
`metadata['session_info'][s]['whisker_me_tertiles']`. Achieved distribution:
0.333 / 0.335 / 0.333.

ii.
```python
me_lbl, me_thr = discretize_tertiles(me)
outputs[:, 3, :] = me_lbl
```
```python
OUTPUT_VALUES = [..., ['low', 'medium', 'high'], ['low', 'medium', 'high']]
```

iii. Same rationale as 7-c (Step 5 Key Decision 5): equal-frequency 3-class bins because the
Decoder Task asks for 3 bins and the evaluation metric is balanced accuracy; per-session because
the units are arbitrary and camera-dependent. Step 10 Check 2 reports 99.6–99.8 % of bins
identical to an independent re-derivation, with the residual attributed to values sitting
exactly on a tertile boundary where `np.interp` clamps and `interp1d(fill_value='extrapolate')`
extrapolates.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as the wheel: evaluated at the 100 right-edge bin times measured from that trial's
`stimOn_times`, so it shares a time axis bin-for-bin with the neural array and the time input.
Camera frame times are already on the session clock, so no resynchronisation is done; trials
whose window is not spanned by camera frames (gap > one bin at either edge) or that contain NaN
are dropped.

ii.
```python
idxs_beg = np.searchsorted(times, interval_begs, side='right')
idxs_end = np.searchsorted(times, interval_ends, side='left')
...
x = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
vals[k] = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

iii. Step 7 Processing Plots Review: the raw 60 Hz camera trace and the interpolated 20 ms
samples are overlaid and "coincide". Step 10 Check 3 confirms the slicing sides, the target grid
and the skip rules are identical to `get_behavior_per_interval`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Everything is dropped-and-logged rather than imputed, at the finest granularity that makes
sense:
* **Missing trial events** (NaN in `stimOn_times`, `choice`, `feedback_times`,
  `probabilityLeft`, `firstMovement_times`, `feedbackType`) — handled by the reference mask.
* **Missing / short behavioural coverage** — the reference's four coverage rules; 86 trials.
* **NaN inside a behavioural trace** — trial dropped (the reference would keep and impute).
* **Zero-spike trials** — 16 trials dropped as recording dropouts.
* **No left camera** — fall back to the right camera; **no camera at all** — session skipped
  (14 sessions).
* **Too few neurons** (< 5) — session skipped (3 sessions).
* **No usable trial / < 2 usable trials** — session skipped (1 session).
* **Unexpected `probabilityLeft`** — hard error returned for the session (never triggered).
* **Any other exception** — caught per session, the session is skipped with its exception type,
  message and traceback, so one bad session cannot kill the run.
Every skipped session is printed with its reason and stored in
`metadata['skipped_sessions']`; per-session drop counts are stored in
`info['behavior_drop_counts']` and `info['n_zero_spike_trials_dropped']`.

ii.
```python
except Exception as e:  # noqa: BLE001 - one bad session must not kill the run
    return {'eid': eid, 'error': f'{type(e).__name__}: {e}',
            'traceback': traceback.format_exc()}
```
```python
if np.any(np.isnan(v)):
    # the reference keeps these (allow_nans=True) and imputes the trial
    # mean inside its data loader; a categorical target cannot be
    # imputed without inventing a class, so the trial is dropped
    reasons[k] = 'nans in target data'
    continue
```
```python
'skipped_sessions': [{'eid': r['eid'], 'error': r['error']} for r in failed],
'behavior_drop_counts': {name: int((~beh_good[name]).sum()) for name in beh_good},
```

iii. Step 4/10: the NaN deviation is justified because "the target format forbids NaN and a
categorical label cannot be imputed without inventing a class"; the zero-spike and ≥5-neuron
rules were added specifically to eliminate 38 `all neural data is zero` verifier warnings rather
than tolerate them. Step 9 states "No data was lost silently: every dropped session is listed
with its reason … and every dropped trial is accounted for by the reference trial mask (97,011),
missing behavioural coverage (86) or an empty population recording (16)."

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting, by a wide margin. The script prints a cumulative per-stage timing
report; over the full 441-session run: `load_spikes` 3890 s (8.82 s/session, 93 % of worker
time), `load_behavior` 456 s (1.03 s), `load_trials` 219 s (0.50 s), `bin_behavior` 28.5 s
(0.06 s), `bin_spikes` 9.1 s (0.02 s). Total wall time 211 s with 28 workers; pickling the
11.7 GB output takes a further 17.8 s. The cost is essentially file I/O on the two-to-four
hundred-MB spike arrays per probe.

ii.
```python
t0 = time.time()
spikes, clusters = load_session_spikes(one, eid, probe_rows)
timings['load_spikes'] = time.time() - t0
```
```python
agg = defaultdict(float)
for r in ok:
    for k, v in r['timings'].items():
        agg[k] += v
print('\nCumulative worker time by stage (s):')
for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
    print(f'  {k:16s} {v:9.1f}  ({v / max(1, len(ok)):.2f} / session)')
```

iii. Step 7 estimated 6.7 s/session (up to 10 s with two probes) dominated by spike loading, and
projected ≈3 min wall-clock with 24 workers; the actual run took 211 s, confirming the estimate.
The AI's stated strategy was therefore to parallelise at the session level rather than
micro-optimise the compute, since "the cost is mainly file I/O".

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial Python loops remain, both of which could in principle be collapsed:
* `bin_spikes` loops over trials, doing one `np.bincount` per trial. It could be done in a single
  `bincount` over all trials by offsetting each spike's flat index by `trial * n_neurons *
  n_bins`, at the cost of materialising per-trial spike copies. The AI already vectorised the
  window-boundary search (`searchsorted` over all trials at once) and the cost is now 0.02
  s/session, so there is nothing material left to win.
* `interpolate_behavior` loops over trials, constructing a fresh `scipy.interp1d` object per
  trial per behaviour. This is the more wasteful of the two (object construction dominates) and
  could be replaced by a single `np.interp` over one concatenated query vector, or by
  `np.searchsorted` + manual linear blend. It costs 0.06 s/session.
Together the two loops account for 0.08 s of the ~10 s per session, i.e. < 1 %.
The final list-comprehensions that split the stacked arrays into per-trial lists
(`[binned[k] for k in range(n_trials)]`) are a format requirement, not a computation.

ii.
```python
for k in range(n_intervals):
    i0, i1 = i0s[k], i1s[k]
    ...
    counts = np.bincount(flat, minlength=n_neurons * n_bins)
    out[k] = counts.reshape(n_neurons, n_bins)
```
```python
for k in range(n):
    t = times[idxs_beg[k]:idxs_end[k]]
    v = values[idxs_beg[k]:idxs_end[k]]
    ...
    vals[k] = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```
```python
'neural': [binned[k] for k in range(n_trials)],
'input': [inputs[k] for k in range(n_trials)],
'output': [outputs[k] for k in range(n_trials)],
```

iii. The AI documents the vectorisation it *did* do rather than the loops it left (Step 6):
"Spike binning replaced by a single `np.searchsorted` for all trial edges plus one `np.bincount`
on `cluster*T + bin` per trial: 0.01 s/session instead of pool startup + ~400 `bincount2D`
calls", and "Parallelism moved up one level: one worker process per session (default 24), so the
pools are created once for the whole run instead of ~2000 times". Its implicit justification for
stopping there is the timing table, which shows binning and interpolation are already
negligible next to I/O.

## 10-c. What processing does the code repeat multiple times?

i. Very little is recomputed. Deliberate de-duplication: the `ONE` client and the
`BrainRegions` atlas are per-worker singletons created once (`get_one`, `get_brain_regions`);
a single `SessionLoader` instance is created in `load_trials` and then reused for the wheel and
the camera, so the trials table is loaded once per session;
`SpikeSortingLoader.merge_clusters(..., compute_metrics=False)` avoids recomputing QC metrics
that are already stored; and spikes are binned only for the trials that survive curation.
What does repeat:
* `interpolate_behavior` is run once per behavioural stream (twice per session), each time
  re-running `searchsorted` and re-slicing — unavoidable, the two streams have different
  timestamps.
* `np.searchsorted` over the spike times is effectively done twice (once for interval starts,
  once for ends) — trivial.
* `SpikeSortingLoader` is instantiated and `load_spike_sorting` called once per probe — required.
* `acronym2acronym(..., 'Beryl')` is applied to the full cluster table including clusters that
  the `label >= 1` cut will immediately discard (≈88 % of them); it could be applied after the
  quality cut.
* The per-trial `interp1d` object construction (see 10-b) repeats setup work 400× per session.

ii.
```python
def get_one():
    global _ONE
    if _ONE is None:
        from one.api import ONE
        _ONE = ONE(base_url=ONE_BASE_URL, silent=True, tables_dir=ONE_TABLES_DIR)
    return _ONE
```
```python
sess_loader, trials, trials_mask = load_trials(one, eid)
...
beh = load_continuous_behaviors(one, eid, sess_loader)   # reuses the same loader
```
```python
beryl_all = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
good = clusters['label'].to_numpy() >= 1
```

iii. Step 6 records the avoided repetitions as explicit speed-ups ("Spikes are binned only for
trials that survive curation"; "the pools are created once for the whole run instead of ~2000
times"). The remaining repeats are not discussed in CONVERSION_NOTES — they are visible only
from the code — and are individually negligible against the 8.8 s/session I/O cost.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI's notes claim the pipeline is lean (it removed the reference's
`raw_electrophysiology(...).fs` metadata query and binned only surviving trials), but the code
still does three things whose results are thrown away:
* **`load_spike_sorting()` is called with the default `SPIKES_ATTRIBUTES = ['clusters', 'times',
  'amps', 'depths']`**, so `spikes.amps` and `spikes.depths` — ≈330 MB per probe on top of the
  ≈250 MB actually needed — are read from disk and never used. Since spike loading is 93 % of
  the runtime, this roughly doubles the dominant cost. (The reference conversion avoids it by
  setting `bio.SPIKES_ATTRIBUTES = ['clusters', 'times']` before importing the loader.)
  `merge_probes` then concatenates and re-sorts those unused arrays as well.
* **Spikes are loaded before the session is known to be usable.** `load_session_spikes` runs
  first; the camera check and the trial-mask check happen afterwards, so for the 15 sessions with
  no whisker video or no usable trial the entire (most expensive) spike load was wasted.
* **Per-session diagnostics that are computed but not used downstream**: `n_good`,
  `n_clusters_total`, `mean_firing_rate_hz`, `behavior_drop_counts`, `beh_reasons` (built for
  every trial and then discarded), `lab`, and the full cluster DataFrame kept in memory after
  only `label` and `acronym` are needed. These are cheap and serve auditability.
Beyond that, nothing substantive is computed and discarded: every array written to the pickle is
consumed by the decoder.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()          # default attributes: also reads amps + depths
```
```python
# ---- neural ---- (runs first, before the camera/trial checks that may skip the session)
spikes, clusters = load_session_spikes(one, eid, probe_rows)
...
me = None
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    ...
if me is None:
    raise RuntimeError('no whisker motion energy available (left or right camera)')
```
```python
reasons = [None] * n
...
'behavior_drop_counts': {name: int((~beh_good[name]).sum()) for name in beh_good},
'mean_firing_rate_hz': float(binned.sum() / (n_neurons * n_trials * 2.0)),
```

iii. CONVERSION_NOTES does not identify any of these; Step 6 lists only the *reference's*
inefficiencies and the AI's own speed-ups, and Step 7's timing table treats the 8.8 s/session
spike load as an irreducible I/O cost rather than something that could be halved. The retained
diagnostics are implicitly justified by the auditing requirement of Step 10 ("so the conversion
can be re-derived and checked against the raw ONE objects"). The AI did not run the conversion
again to test whether restricting the spike attributes would cut the runtime.
