# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All data are read through the ONE API against the staged IBL cache at `/app/data/one_cache`,
using the IBL loader classes (`SessionLoader`, `SpikeSortingLoader`) exactly as the reference
code does — no raw `.npy`/`.pqt` files are opened directly by the conversion. Two deviations from
a plain `one.search()`:

1. The **session/probe inventory** is taken from `bwm_release.csv`, the release manifest shipped
   inside the reference repo (`/app/code/code_zhang2025/data/bwm_release.csv`), which enumerates
   699 pids / 459 eids / 139 subjects / 12 labs. Session directory paths are resolved from the
   ONE `sessions.pqt` table (`lab/Subjects/subject/date/number`). This was done because ONE's
   `eid2pid()` requires a network connection, which is unavailable.
2. The ONE **`datasets` cache table is rebuilt from the filesystem** (`build_dataset_table`) by
   walking each session's `alf/` tree. The AI found that the shipped release tables are stale
   with respect to the staged files: `alf/_ibl_trials.table.pqt` is listed at the un-revised path
   but on disk exists only inside revision folders (454 × `#2025-03-03#`), and loading through the
   shipped table *silently* returns a 1-column trials frame instead of raising.

Per session, `SessionLoader` supplies trials, wheel and camera motion energy; `SpikeSortingLoader`
is called once per probe insertion. Sessions are processed in a `ProcessPoolExecutor` (24 workers).

ii.
```python
ROOT = '/app/data/one_cache'
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
SESSIONS_PQT = f'{ROOT}/Brainwidemap/sessions.pqt'

def get_session_table():
    """Return (bwm_df, sessions_df, {eid: session_path}) for the 699 released probes."""
    sess = pd.read_parquet(SESSIONS_PQT)
    sess.index = sess.index.astype(str)
    bwm = pd.read_csv(BWM_CSV, index_col=0)
    paths = {}
    for eid in bwm.eid.unique():
        r = sess.loc[eid]
        paths[eid] = (f"{ROOT}/{r['lab']}/Subjects/{r['subject']}/"
                      f"{str(r['date'])}/{int(r['number']):03d}")
    return bwm, sess, paths

def make_one(session_paths_map):
    from one.api import ONE
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local')
    one.load_cache(tables_dir=f'{ROOT}/Brainwidemap')
    one._cache['datasets'] = build_dataset_table(session_paths_map)
    return one
```

```python
def build_dataset_table(session_paths_map):
    rows = []
    for eid, sp in session_paths_map.items():
        eid_uuid = uuid.UUID(eid)
        for dirpath, _, filenames in os.walk(os.path.join(sp, 'alf')):
            rel = os.path.relpath(dirpath, sp)
            for f in filenames:
                rows.append((eid_uuid, uuid.uuid4(),
                             os.path.getsize(os.path.join(dirpath, f)),
                             None, True, 'NOT_SET', True, f'{rel}/{f}'))
    ...
```

Everything downstream follows from the `eid` / `pid`:
```python
sl = SessionLoader(one=one, eid=eid)
trials, mask = load_trials_and_mask(sl)
...
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
```

iii. From CONVERSION_NOTES.md Step 2: *"the shipped release tables are stale relative to the
staged files … Loading through the shipped table silently returns a 1-column trials frame
(`goCueTrigger_times` only) instead of raising. The conversion therefore rebuilds the `datasets`
cache table by walking each session's `alf` tree … ONE's offline `eid2pid` also needs the network,
so probe IDs come from `bwm_release.csv` instead."* Step 10 Check 5 verifies: *"trials frames have
all 20 columns for all 459 sessions"* and *"699/699 probes loaded"*.

---

## 1-b. How are the data split into subjects?

i. The subject name comes directly from the `subject` column of `bwm_release.csv` (one row per
probe insertion), so nothing is parsed out of paths. `subjects` is the sorted set of unique subject
names over the *retained* sessions, and `subject_idx` is each session's index into that list.
Result: 136 subjects over 442 kept sessions (139 subjects in the 459-session release; 3 subjects
are lost with the 17 dropped sessions).

ii.
```python
sub = bwm[bwm.eid == eid]
info['subject'] = sub.subject.iloc[0]
...
subjects = sorted({r['subject'] for r in kept})
subj_idx = np.array([subjects.index(r['subject']) for r in kept], dtype=np.int64)
```

iii. The release manifest already carries a unique subject id per session, so no derivation is
needed. CONVERSION_NOTES Step 9 documents the count: *"Subjects | 139 | 139 | 139 | 136 (3 lost
with the 17 dropped sessions)"*.

---

## 1-c. How are the data split into sessions?

i. No split is performed: a session (`eid`) is already the organising unit of the release.
`bwm.eid.unique()` gives 459 eids, and each is converted independently by `convert_session`.
Multiple probe insertions belonging to the same eid are *merged* into one session rather than
treated as separate sessions.

ii.
```python
bwm, sess, paths = get_session_table()
eids = list(bwm.eid.unique())
if args.sample:
    eids = eids[:2]
...
for res, info in ex.map(_worker, rest, chunksize=1):
```

iii. CONVERSION_NOTES Step 2: *"`bwm_release.csv` … enumerates the release: **699 pids, 459 eids,
139 subjects, 12 labs** — identical to the data paper."* Probes are merged because (reference
`merge_probes` docstring) *"data from the probes recorded in the same session are not statistically
independent as they have the same underlying behaviour"*.

---

## 1-d. How are the data split into trials?

i. No split is performed: the ALF trials table has one row per trial, loaded by
`SessionLoader.load_trials()`. Each row supplies `stimOn_times`, which defines that trial's
[-0.5, +1.5] s window used for all streams.

ii.
```python
if sess_loader.trials.empty:
    sess_loader.load_trials()
...
trials_sel = trials[mask]
align_times = trials_sel[ALIGN_TIME].to_numpy()   # ALIGN_TIME = 'stimOn_times'
```

iii. Nothing to decide — the trials table is already one row per trial. CONVERSION_NOTES Step 2
lists the 20 available trial columns.

---

## 1-e. How are trials filtered based on quality controls?

i. Three successive masks, ANDed together:

1. **BWM trial mask** — `load_trials_and_mask`, a verbatim transcription of the reference
   `ibl_data_utils.load_trials_and_mask` with its defaults plus `max_trial_len=10.0` (the value
   the reference caching script passes): drop trials with NaN in any of `stimOn_times`, `choice`,
   `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; drop reaction times
   (`firstMovement_times − stimOn_times`) outside [0.08, 2.0] s; drop `feedback_times − goCue_times
   > 10 s`; drop no-choice trials (`choice == 0`).
2. **Behaviour coverage mask** — the reference's four `get_behavior_per_interval` skip rules
   applied to *both* the wheel and the whisker trace with `allow_nans=False`: no samples in the
   window, NaN inside the window, trace starts more than one bin late, or ends more than one bin
   early.
3. **No-spike mask** — a trial whose whole population is silent for the full 2 s is dropped. This
   is an addition beyond the reference, added after the first full run produced 16 "all neural
   data is zero" warnings, which the AI root-caused to ephys dropouts.

Attrition over the 442 kept sessions: 285,682 raw → 188,469 after the BWM mask (66.0 %) →
188,383 after the behaviour mask (−86) → **188,367** after the no-spike mask (−16).

ii.
```python
def load_trials_and_mask(sess_loader, min_rt=0.08, max_rt=2., nan_exclude='default',
                         min_trial_len=None, max_trial_len=10.0,
                         exclude_unbiased=False, exclude_nochoice=True):
    if nan_exclude == 'default':
        nan_exclude = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
                       'firstMovement_times', 'feedbackType']
    query = f'(firstMovement_times - stimOn_times < {min_rt})' if min_rt is not None else ''
    if max_rt is not None:
        query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    if exclude_nochoice:
        query += ' | (choice == 0)'
    mask = ~sess_loader.trials.eval(query)
    return sess_loader.trials, mask
```

```python
beh_good = wheel_good & whisk_good          # element-wise AND of the two behaviour masks
...
neural_covered = binned.sum(axis=(1, 2)) > 0
keep = beh_good & neural_covered
```

iii. CONVERSION_NOTES Step 3 quotes the data paper's own criterion verbatim (*"trials were
excluded if one of the following trial events could not be detected: choice, probabilityLeft,
feedbackType, feedback times, stimOn times and firstMovement times … outside the range of
0.08–2.00 s"*) and notes the mask is identical to the reference `load_trials_and_mask` defaults.
On `max_trial_len` (Step 4): *"Kept (follows the reference decoding pipeline); its effect is
negligible (≈0.2 % of trials)."* On NaNs (Step 4): *"reference caches with `allow_nans=True`, then
mean-imputes in the model loader … Target format forbids NaN, so such trials are dropped instead
of imputed (`allow_nans=False` path of the same reference function)."* On the no-spike mask
(Step 10 Check 1): *"These are recording dropouts, not conversion errors, but a trial with an
identically-zero population carries no neural information … the neural counterpart of the
reference's own 'target data ends too early' behaviour check."* The AI also flagged a genuine bug
in the reference: *"`align_spike_behavior`: `target_mask = target_mask and beh_mask` uses Python's
`and` on lists, which returns the second operand whole — so only the last behaviour's mask …
actually survives. Fixed here by AND-ing the masks element-wise."*

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from each probe's pykilosort spike sorting
(revision `#2024-05-06#`). `clusters.metrics.label` supplies the quality cut and
`clusters.acronym` (via `channels`) supplies the anatomical label written to
`brain_region_idx`, but the neural array itself is built from the two spike arrays only.

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
iok = clusters_labeled['label'] >= qc
selected_clusters = clusters_labeled[iok]
spike_idx, ib = ismember(spikes['clusters'], selected_clusters.index)
selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
selected_spikes['clusters'] = selected_clusters.index[ib].astype(np.int32)
```

```python
binned = bin_spiking_data(st, sc, n_clusters, align_times)   # st = spike times, sc = cluster ids
```

iii. CONVERSION_NOTES Step 1 documents `load_spiking_data` as a direct transcription of the
reference `ibl_data_utils.load_spiking_data`; the only omission is *"the `raw_electrophysiology(...).fs`
call, which needs a network connection and is only used for a metadata field"*.
Step 1 also notes *"No dF/F is involved: this is electrophysiology (spike times), not imaging."*

---

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 100 non-overlapping 20 ms bins spanning
[`stimOn_times` − 0.5 s, `stimOn_times` + 1.5 s). The value stored is the **raw spike count per
bin** (float32), *not* converted to a firing rate and *not* smoothed or z-scored. When a session
has two probes (240 of 459 sessions do), their clusters and spike trains are concatenated by
`merge_probes`, cluster indices are offset so the second probe continues the numbering, and the
merged spike train is re-sorted by time. Every retained cluster gets a row, including clusters
with zero spikes in the window.

ii.
```python
def bin_spiking_data(spike_times, spike_clusters, n_clusters, align_times):
    ntrials = len(align_times)
    out = np.zeros((ntrials, n_clusters, NBINS), dtype=np.float32)
    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    i0 = np.searchsorted(spike_times, begs, side='left')
    i1 = np.searchsorted(spike_times, ends, side='left')
    for k in range(ntrials):
        a, b = i0[k], i1[k]
        if b <= a:
            continue
        t = spike_times[a:b]; c = spike_clusters[a:b]
        idx = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
        keep = (idx >= 0) & (idx < NBINS)
        flat = c[keep].astype(np.int64) * NBINS + idx[keep]
        counts = np.bincount(flat, minlength=n_clusters * NBINS)
        out[k] = counts.reshape(n_clusters, NBINS)
    return out
```

```python
def merge_probes(spikes_list, clusters_list):
    merged_spikes, merged_clusters, cluster_max = [], [], 0
    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes = dict(spikes)
        spikes['clusters'] = spikes['clusters'] + cluster_max
        cluster_max += len(clusters)
        ...
    sort_idx = np.argsort(merged_spikes['times'], kind='stable')
    merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}
```

iii. CONVERSION_NOTES Step 5 Key Decision 8: *"Neural values: raw spike counts (float32), not
rates or z-scores — the reference caches counts and z-scoring is a model-side step."* Step 1
confirms: *"`standardize_spike_data` … Per-time-bin z-scoring — applied **in the model's data
loader**, not in the cached dataset."* Step 10 Check 3 records the binning equivalence:
*"`bincount2D(t, c, xbin=0.02, xlim=[t_beg, t_end])[:, :100]` ⇒ bin `floor((t − t_beg)/0.02)`,
left-closed | `np.bincount` on `cluster*100 + floor((t − t_beg)/0.02)` — same bin definition,
vectorised."* The AI also found and corrected a bug in the reference `merge_probes`:
*"`cluster_max = clusters.index.max() + 1` replaces rather than accumulates the offset. Harmless
for this release (no session has more than 2 probes) … but corrected here."*

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters:

* **Unit level**: only clusters with `clusters['label'] >= 1.0` are kept — the "well-isolated
  neurons" of the data paper (label is the fraction of the three RIGOR metrics passed: amplitude
  > 50 µV, noise cut-off < 20 µV, refractory-period violation). Verified to reproduce the paper's
  numbers exactly: 621,733 total units → 75,708 with `label == 1`.
* **Session level**: a session with fewer than `MIN_NEURONS = 5` well-isolated neurons is dropped
  (2 sessions, with 2 and 3 neurons). A session with < 2 usable trials is also dropped.

**No anatomical filtering is applied**: units mapping to `root` or `void` in the Beryl atlas are
retained, following the reference's `single_region=False` / `region='all'` setting. Result:
73,039 neurons over 442 sessions, 265 Beryl regions, mean 165.2 neurons/session.

ii.
```python
QC_LABEL = 1.0
MIN_NEURONS = 5
...
iok = clusters_labeled['label'] >= qc
selected_clusters = clusters_labeled[iok]
```

```python
if n_clusters < MIN_NEURONS:
    info['skip_reason'] = (f'fewer than {MIN_NEURONS} well-isolated neurons '
                           f'({n_clusters})')
    return None, info
```

```python
finite = np.isfinite(spikes['times'])       # spikes with no sample-to-time solution
st = spikes['times'][finite]
```

iii. This is an explicit, documented deviation from the reference *code* toward the reference
*paper*. CONVERSION_NOTES Step 4: *"`0_data_caching.py` calls `load_spiking_data` with default
`qc=None` → all 621,733 units … **Keep `label == 1` (well-isolated).** The data paper states the
criterion explicitly and quantitatively, the number reproduces exactly (a hard sanity check),
`load_spiking_data` exposes `qc=1` for precisely this, and the reference still records
`good_clusters` in its metadata."* On region filtering (Step 4): *"No region filtering: the
decoder here is a whole-session ('region = all') decoder, exactly the reference's `region='all'`
setting. Region labels are still recorded per neuron."* On `MIN_NEURONS` (Step 10 Check 1):
*"the BWM paper's 'at least five well-isolated neurons per session'."*

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams (spike times, trial event times, wheel timestamps, camera frame times) are
already on one synchronised session clock in seconds, so alignment is just subtraction. For each
retained trial the window is [`stimOn_times` − 0.5, `stimOn_times` + 1.5) s; spikes in that range
are located with two `np.searchsorted` calls and their bin index is
`floor((t − (stimOn − 0.5)) / 0.02)`, clipped to [0, 100). Alignment was verified by the population
PSTH (flat before t = 0, sharp rise at t = 0, peak ≈ 0.22 s) and by a per-timepoint decoding
accuracy curve that is flat pre-stimulus and rises at exactly t = 0.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
align_times = trials_sel[ALIGN_TIME].to_numpy()
...
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
i0 = np.searchsorted(spike_times, begs, side='left')
i1 = np.searchsorted(spike_times, ends, side='left')
...
idx = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
keep = (idx >= 0) & (idx < NBINS)
```

iii. CONVERSION_NOTES Step 5 Key Decision 1: *"Alignment / binning: `stimOn_times`, window
(−0.5, +1.5) s, 100 × 20 ms bins — identical to the reference caching parameters and to the
decoder task's requirement to 'temporally align based on stimulus onset'."* Step 10 Check 5:
*"Bin-edge off-by-one | `floor((t − t_beg)/binsize)`, clipped to `[0, 100)`; first bin starts
exactly at `stimOn − 0.5`, last ends at `stimOn + 1.5` | independent recomputation reproduces
every trial"* (1,192 trials across 4 sessions re-derived from raw `.npy` files, exact).

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial, identical for every trial and session
(`metadata['time_bin_size'] = 20.0` ms). No resampling, interpolation or rebinning of the neural
data — spikes are counted once, directly onto the final grid. `NBINS` is computed as
`ceil(interval_len / binsize)` exactly as the reference does.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

```python
'time_bin_size': BINSIZE * 1000.,
'off_start': TIME_WINDOW[0],
'off_end': TIME_WINDOW[1],
'n_timepoints': NBINS,
```

iii. CONVERSION_NOTES Step 1: *"Binning parameters are fixed once, in `0_data_caching.py`:
`{'interval_len': 2, 'binsize': 0.02, 'single_region': False, 'align_time': 'stimOn_times',
'time_window': (-.5, 1.5)}` → T = 100 bins of 20 ms."* Step 3 quotes the methods paper:
*"Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time
steps."*

---

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `trials.stimOn_times` — indirectly. The input is not read from the data at all; it is the
bin-centre time vector of the analysis grid defined around `stimOn_times`, so it is the same 100
values, −0.49 … +1.49 s, for every trial of every session.

ii.
```python
align_times = trials_sel[ALIGN_TIME].to_numpy()       # ALIGN_TIME = 'stimOn_times'
...
# Time from stimulus onset at the centre of each 20 ms neural bin.
tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: *"bin index | `input[0]` `time_from_stim_on`
| bin centre time, −0.49 … +1.49 s | (new; the decoder task asks for it) | time-varying"*. The
window and bin size come from the reference caching parameters.

---

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the bin-centre vector. It is stored as row 0 of each trial's
(2, 100) float32 input array, identical across trials.

ii.
```python
tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)

for k in range(ntrials):
    inp = np.empty((2, NBINS), dtype=np.float32)
    inp[0] = tvec
    inp[1] = tnib_keep[k]
    input_list.append(inp)
```

iii. N/A — the variable is defined by the conversion, not derived from data. Sanity-checked
independently (Step 10 Check 2): *"`input[0]` == bin-centre time vector | `np.allclose` pass"*.
Step 9 confirms the realised range is [−0.49, 1.49].

---

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural binning grid: spikes are binned on edges
`stimOn + (−0.5 + 0.02·k)`, and the input is the centre of those same bins, so element `k` of the
input and column `k` of the neural matrix describe the same 20 ms interval.

ii.
```python
# neural: bin index of each spike on the same grid
idx = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
```
```python
# input: the centre of those bins
tvec = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```

iii. N/A — by construction. `plot_processing` overlays the raster, the binned counts and the PSTH
on this same axis (`edges = TIME_WINDOW[0] + BINSIZE * np.arange(NBINS + 1)`) to make the identity
visible.

---

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft`. The trials table has no block identifier, so blocks are recovered
as maximal runs of constant `probabilityLeft`; a change of value starts a new block.

ii.
```python
def trial_number_in_block(probability_left):
    p = np.asarray(probability_left, dtype=float)
    newblock = np.ones(len(p), dtype=bool)
    if len(p) > 1:
        # NaN-safe change detection: a NaN probabilityLeft starts a new block.
        same = (p[1:] == p[:-1])
        newblock[1:] = ~same
    block_id = np.cumsum(newblock) - 1
```

iii. CONVERSION_NOTES Step 5: *"`trials.probabilityLeft` | `input[1]` `trial_number_in_block` |
0-based position inside the run of constant `probabilityLeft`, computed over **all** trials of the
session"*. Sanity-checked (Step 10 Check 2): *"first block of every session is the 90-trial
`p = 0.5` unbiased block | pass"*, consistent with the data paper's task description.

---

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based cumulative count within each block, **computed on the full trials table before any
trial exclusion**, then subset by the trial mask. So a trial that is later dropped still advances
the counter, and the surviving trials carry the animal's true position in the block. The value is
broadcast across all 100 bins as row 1 of the input array. Realised range over the full dataset:
[0, 98].

ii.
```python
block_id = np.cumsum(newblock) - 1
out = np.zeros(len(p), dtype=np.int64)
for b in np.unique(block_id):
    m = block_id == b
    out[m] = np.arange(m.sum())
return out, block_id
```
```python
tnib_all, block_id_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
trials_sel = trials[mask]
tnib = tnib_all[mask]
...
inp[1] = tnib_keep[k]
```

iii. Docstring: *"Computed on the full trials table before any trial exclusion, so the value is
the animal's true position in the block rather than a position among surviving trials."*
CONVERSION_NOTES Step 10 Check 5: *"`trial_number_in_block` computed over the *full* trials table
before exclusion, so the first block still runs 0…89 even when its first trials are excluded |
checked against raw `probabilityLeft`."*

---

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. One column of the trials table, `trials.choice`, which takes values +1, −1 and 0. Trials with
`choice == 0` (no response) have already been removed by the BWM trial mask
(`exclude_nochoice=True`), so only ±1 reaches the encoding step.

ii.
```python
choice_raw = trials_keep['choice'].to_numpy()
choice = (choice_raw < 0).astype(np.int8)
```
```python
if exclude_nochoice:
    query += ' | (choice == 0)'
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: *"IBL `choice == +1` means the mouse reported the
**left** stimulus (verified: on correct trials with a left stimulus, `choice` is always +1). Mapped
to left = 0, right = 1 as the task specifies."* Independently sanity-checked against raw parquet
(Step 10 Check 2): *"`output[0]` == `(choice == −1)` from the raw parquet | exact"* and *"on
correct trials with a left stimulus, `output[0] == 0` | pass (150 / 177 / 105 / 107 trials)"*.

---

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding +1 → 0 (left), −1 → 1 (right), stored as `int8` and broadcast constant across
the 100 time bins of the trial (the spec asks for time-varying outputs where possible; choice is a
per-trial variable so it is held constant). Realised distribution over the full dataset:
left 0.508 / right 0.492.

ii.
```python
choice = (choice_raw < 0).astype(np.int8)
...
out = np.empty((4, NBINS), dtype=np.int8)
out[0] = choice[k]
```
```python
OUTPUT_VALUES = [
    ['left', 'right'],
    ...
]
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: *"Time-varying wherever possible: all four outputs
are emitted as (4, 100) arrays, with the two per-trial variables constant across the 100 bins."*

---

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. One column of the trials table, `trials.probabilityLeft`, which takes exactly the three values
0.2, 0.5 and 0.8 (verified on all 459 sessions).

ii.
```python
pleft = trials_keep['probabilityLeft'].to_numpy()
prior = np.full(ntrials, -1, dtype=np.int8)
for val, lab in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(pleft, val)] = lab
if np.any(prior < 0):
    info['skip_reason'] = f'unexpected probabilityLeft values {np.unique(pleft)}'
    return None, info
```

iii. CONVERSION_NOTES Step 5: *"`trials.probabilityLeft` | `output[1]` `prior_prob_left` |
0.2→0, 0.5→1, 0.8→2 | `bin_behaviors` (`block`) | per-trial"* — the same variable the reference
caches as `block`. The mapping is dictated by the task spec.

---

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Only the recoding, done with `np.isclose` (float-safe) rather than exact equality, with a
defensive session-level rejection if any value falls outside the three expected ones (never
triggered). Broadcast constant across the 100 bins as `int8`. Realised distribution:
0.2 → 0.417, 0.5 → 0.140, 0.8 → 0.442 — consistent with the 90 unbiased trials out of ≈645 per
session.

ii.
```python
for val, lab in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(pleft, val)] = lab
...
out[1] = prior[k]
```
```python
OUTPUT_VALUES = [..., ['p_left=0.2', 'p_left=0.5', 'p_left=0.8'], ...]
```

iii. CONVERSION_NOTES Step 9 checks the realised distribution against the task structure:
*"prior distribution | 90 unbiased of ~645 ⇒ ~14 % at 0.5 | 0.417 / 0.140 / 0.442 | ✓"*.
Step 10 Check 5: *"`probabilityLeft` outside {0.2, 0.5, 0.8} | session rejected with a reported
reason | never triggered."*

---

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position.npy` / `_ibl_wheel.timestamps.npy`, read through
`SessionLoader.load_wheel()`, which returns `times`, `position`, `velocity`, `acceleration`. The
speed is `np.abs(velocity)`, in rad/s — the same derivation as the reference's
`load_target_behavior('wheel-speed')`.

ii.
```python
sess_loader.load_wheel()
out['wheel-speed'] = (sess_loader.wheel['times'].to_numpy(),
                      np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. CONVERSION_NOTES Step 1: *"`load_target_behavior` … `SessionLoader.load_wheel()` →
`wheel-speed = abs(velocity)`"*; Step 10 Check 3 marks loading as identical to the reference.

---

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps:

1. `SessionLoader.load_wheel()` interpolates the (event-driven) raw wheel position onto a uniform
   1 kHz grid and differentiates it with a 20 Hz Butterworth low-pass (`interpolate_position` +
   `velocity_filtered`) — IBL's default, not a choice made here.
2. Absolute value → speed.
3. Per trial, linear interpolation (`interp1d(..., fill_value='extrapolate')`) onto
   `np.linspace(t_beg + binsize, t_end, 100)`, i.e. the **right edge** of each of the 100 neural
   bins, using only samples strictly inside the window. Trials failing any of the reference's four
   coverage/NaN rules are dropped.
4. Discretisation into 3 classes at within-session tertiles (see 7-c).

ii.
```python
def bin_behavior(target_times, target_values, align_times):
    ...
    idxs_beg = np.searchsorted(target_times, begs, side='right')
    idxs_end = np.searchsorted(target_times, ends, side='left')
    for k in range(ntrials):
        tt = target_times[idxs_beg[k]:idxs_end[k]]
        vv = target_values[idxs_beg[k]:idxs_end[k]]
        if len(vv) == 0:            continue
        if np.isnan(vv).any():      continue
        if np.isnan(begs[k]) or np.isnan(ends[k]):  continue
        if np.abs(begs[k] - tt[0]) > BINSIZE:       continue
        if np.abs(ends[k] - tt[-1]) > BINSIZE:      continue
        x = np.linspace(begs[k] + BINSIZE, ends[k], NBINS)
        values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
        good[k] = True
```

iii. CONVERSION_NOTES Step 6: *"`bin_behavior` is a transcription of `get_behavior_per_interval`
including all four skip conditions, with `allow_nans=False`."* Step 10 Check 3 confirms the
reference does *"`interp1d(linear, extrapolate)` at `linspace(t_beg + binsize, t_end, 100)` = bin
right edges, with 4 coverage/NaN skip rules"* and the conversion *"transcribed verbatim"*.
Independently sanity-checked from `_ibl_wheel.position.npy` via
`brainbox.behavior.wheel.interpolate_position` + `velocity_filtered`: class agreement 0.9994–0.9999.

---

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Equal-occupancy tertiles computed **per session** over all retained trials × 100 timepoints
pooled: the 1/3 and 2/3 quantiles of the session's own resampled speed trace are used as the two
`np.digitize` edges, giving classes 0/1/2 = low/medium/high. A degenerate fallback handles a
near-constant trace (never needed: all sessions produced strictly increasing edges). The edges
actually used are recorded per session in `metadata['session_info']['wheel_speed_tertile_edges']`.
Realised distribution: 0.333 / 0.333 / 0.333.

ii.
```python
def discretize_tertiles(values):
    flat = values.ravel()
    edges = np.quantile(flat, [1. / 3., 2. / 3.])
    if not edges[1] > edges[0]:
        uniq = np.unique(flat)
        if len(uniq) >= 3:
            edges = np.quantile(uniq, [1. / 3., 2. / 3.])
        if not edges[1] > edges[0]:
            edges = np.array([edges[0], edges[0] + np.finfo(float).eps])
    labels = np.digitize(values, edges).astype(np.int8)
    return labels, edges
```
```python
wheel_lab, wheel_edges = discretize_tertiles(wheel_vals)
info['wheel_edges'] = wheel_edges.tolist()
```

iii. Docstring and CONVERSION_NOTES Step 5 Key Decision 6: *"The spec asks for 3 bins but does not
fix the edges … Wheel speed … its distribution is strongly zero-inflated, so fixed edges would
leave near-empty classes in quiet sessions … Equal-occupancy edges give the same class semantics
(low / medium / high for this session) everywhere and make chance exactly 1/3 for the
balanced-accuracy metric that `train_decoder.py` reports."*

---

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Same trial window, same session clock, same 100-bin grid: the speed trace is evaluated at the
**right edge** of each of the 100 neural bins (`t_beg + 0.02, …, t_beg + 2.00`), where `t_beg =
stimOn − 0.5`. So sample `k` of the output corresponds to the end of neural bin `k` — a
half-bin (10 ms) offset relative to the bin centre used for the time input. This is the
reference code's own convention, transcribed verbatim.

ii.
```python
x = np.linspace(begs[k] + BINSIZE, ends[k], NBINS)
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```
```python
begs = align_times + TIME_WINDOW[0]      # identical to the neural window
ends = align_times + TIME_WINDOW[1]
```

iii. CONVERSION_NOTES Step 3: *"Behaviour resampling: linear interpolation onto the right edge of
each 20 ms bin"*; Step 10 Check 3 marks this stage *"transcribed verbatim"*. The `--show-processing`
plot overlays the raw wheel trace and the resampled points on the bin right edges
(`ax.plot(edges[1:], pd_['wheel_vals'][idx], 'o-')`) to make the alignment visible; Step 7 review
reports *"Resampled wheel speed and whisker motion energy overlay the raw traces."*

---

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` with its frame times `_ibl_leftCamera.times.npy`, loaded via
`SessionLoader.load_motion_energy(views=['left'])` and read from the `whiskerMotionEnergy` column.
If the left camera is unavailable the right camera is used instead (7 of 442 kept sessions). A
session with neither is dropped (14 sessions).

ii.
```python
whisker = None
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        df = sess_loader.motion_energy[cam]
        whisker = (df['times'].to_numpy(), df['whiskerMotionEnergy'].to_numpy())
        out['whisker-camera'] = view
        break
    except Exception:
        continue
if whisker is None:
    raise RuntimeError('no whisker motion energy available')
```

iii. CONVERSION_NOTES Step 1: *"Whisker motion energy uses the *left* camera, falling back to the
*right* camera when the left is unavailable (`bin_behaviors`)"* — i.e. the reference's own
preference order. Step 4: *"Follow the reference: left first, right as fallback."*

---

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, normalisation or unit conversion. It is
resampled per trial onto the same 100 bin right edges by the same `bin_behavior` function used for
the wheel (same four coverage/NaN rejection rules), then discretised into 3 within-session
tertiles.

ii.
```python
whisk_vals, whisk_good = bin_behavior(*traces['whisker-motion-energy'], align_times)
...
beh_good = wheel_good & whisk_good
...
whisk_lab, whisk_edges = discretize_tertiles(whisk_vals)
```

iii. CONVERSION_NOTES Step 5 mapping table: *"`leftCamera.ROIMotionEnergy` (else right) |
`output[3]` `whisker_motion_energy` | interpolate to bin right edges, digitise at within-session
tertiles | `load_target_behavior('left-whisker-motion-energy')`"*. Sanity-checked independently
from `{left,right}Camera.ROIMotionEnergy.npy` + camera times: class agreement **1.000**.

---

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: `discretize_tertiles` on the pooled (trials × 100 bins) trace of the
session, `np.digitize` at the 1/3 and 2/3 quantiles → classes 0/1/2 = low/medium/high. Edges are
stored per session in `metadata['session_info']['whisker_me_tertile_edges']`. Realised
distribution: 0.333 / 0.333 / 0.333.

ii.
```python
whisk_lab, whisk_edges = discretize_tertiles(whisk_vals)
info['whisker_edges'] = whisk_edges.tolist()
...
out[3] = whisk_lab[k]
```
```python
OUTPUT_VALUES = [..., ['low', 'medium', 'high']]
```

iii. Docstring of `discretize_tertiles`: *"Whisker motion energy is in arbitrary camera-dependent
units (the left camera is 1280x1024 @ 60 Hz, the right 640x512 @ 150 Hz), so no fixed threshold
transfers across sessions; equal-occupancy (tertile) edges computed within a session give the same
class semantics — low / medium / high movement for this animal in this session — everywhere, and
make chance performance exactly 1/3 for balanced accuracy."*

---

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: the camera frame times are on the same synchronised session clock
as the spikes, and the trace is evaluated at the right edge of each of the 100 neural bins of the
same [stimOn − 0.5, stimOn + 1.5] s window. A trial whose camera coverage starts more than one bin
late or ends more than one bin early is dropped rather than extrapolated across the gap.

ii.
```python
x = np.linspace(begs[k] + BINSIZE, ends[k], NBINS)
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```
```python
if np.abs(begs[k] - tt[0]) > BINSIZE:   continue    # 'target data starts too late'
if np.abs(ends[k] - tt[-1]) > BINSIZE:  continue    # 'target data ends too early'
```

iii. Same as 7-d. Additionally, CONVERSION_NOTES Step 11 uses the alignment as a cross-check:
*"`sample_trials.png` shows the two time-varying outputs rising together ~0.2 s after stimulus
onset in every sampled trial"*, and Step 12: *"the decoding-accuracy time course … is flat before
t = 0 and rises at t = 0 … independent confirmation that the alignment carries no temporal
offset."*

---

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Systematically, with every drop counted and reported. Trial level: NaN trial events → excluded
by the BWM mask; wheel/camera trace absent, NaN, starting late or ending early inside the window →
trial dropped (86 trials); all-zero neural population (ephys dropout or gap) → trial dropped
(16 trials); NaN spike times → those spikes dropped before binning. Session level: no whisker
motion-energy dataset (14 sessions); camera timestamps that cover none of the trials
(1 session); fewer than 5 well-isolated neurons (2 sessions); fewer than 2 usable trials;
`probabilityLeft` outside {0.2, 0.5, 0.8}. Infrastructure level: stale ONE release tables →
dataset table rebuilt from disk; missing `eid2pid` → probe ids from `bwm_release.csv`; a
degenerate (near-constant) behaviour trace → fallback tertile edges. Any unexpected exception in a
worker is caught, recorded with its traceback, and the session is dropped rather than aborting the
run. All per-session exclusion counters are written into `metadata['session_info']`.

ii.
```python
def _worker(eid):
    try:
        res, info = convert_session(...)
        ...
    except Exception as e:
        import traceback
        return None, {'eid': eid, 'skip_reason': f'{type(e).__name__}: {e}',
                      'traceback': traceback.format_exc()}
```
```python
if mask.sum() < 2:
    info['skip_reason'] = 'fewer than 2 trials pass the trial mask';  return None, info
...
if beh_good.sum() < 2:
    info['skip_reason'] = 'fewer than 2 trials with valid behaviour'; return None, info
...
if n_clusters < MIN_NEURONS:
    info['skip_reason'] = f'fewer than {MIN_NEURONS} well-isolated neurons ({n_clusters})'
    return None, info
...
if keep.sum() < 2:
    info['skip_reason'] = 'fewer than 2 trials with both behaviour and spikes'
    return None, info
```
```python
# Spikes with NaN times (no sample-to-time solution) cannot be binned.
finite = np.isfinite(spikes['times'])
st = spikes['times'][finite]; sc = spikes['clusters'][finite]
```
```python
reasons = defaultdict(int)
for d in dropped:
    reasons[d['skip_reason'].split(':')[0]] += 1
for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f'  dropped ({v}): {k}')
```

iii. CONVERSION_NOTES Step 9 tabulates all 17 dropped sessions with evidence for each, and Step 10
Check 5 tabulates 13 edge cases with how each is handled and how it was verified. On the ephys
dropouts: *"`8c2f7f4d-…`: the spike train ends at 1779.4 s but the last three trials have stimulus
onsets at 1790–1800 s … `b182b754-…`: a 4.6 s gap in the spike train at t ≈ 185 s swallows one
trial … These are recording dropouts, not conversion errors."* Result: the verification log reports
*"Data format is valid, no errors or warnings."*

---

## 10-a. What are the most time-consuming steps of the code?

i. Measured per-stage timings are recorded in `info['timings']` for every session. Serial cost per
session: spike-sorting load **1.5–3 s per probe** (file I/O, the dominant term, and inflated
because the default `SPIKES_ATTRIBUTES` also reads `spikes.amps`/`spikes.depths` and because the
md5 `check_hash` is left on), behaviour load + bin ≈ 1.5 s, trials 0.25 s, spike binning ≈ 0.2 s.
One-off costs: the filesystem scan that builds the ONE `datasets` table, 10.5 s, and pickling the
12.96 GB output, 23.4 s. With 24 worker processes the whole 459-session run took **162 s**.

ii.
```python
t0 = time.time()
spikes_list, clusters_list, n_units_total = [], [], 0
for _, r in sub.iterrows():
    sp, cl, n_tot = load_spiking_data(one, r.pid, eid, r.probe_name)
    ...
info['timings']['spikes'] = time.time() - t0
```
```python
print(f'  {done}/{len(rest)} sessions  ({el:.0f}s elapsed, '
      f'{el / done:.1f}s/session, ETA {el / done * (len(rest) - done):.0f}s)', flush=True)
```

iii. CONVERSION_NOTES Step 7 "Run Time Estimates" gives the per-step table above and the estimate
*"459 × ~6 s ≈ 46 min serial … with 24 workers ≈ 3–5 min + pickling"*; Step 9 reports the realised
*"Run time 162 s (459 sessions, 24 workers, 0.4 s/session) — well inside the 15-minute budget."*

---

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already removed the reference's dominant loop cost: `get_spike_data_per_interval` spawns
a multiprocessing pool and calls `bincount2D` once per trial with a tqdm bar; the AI replaced this
with two whole-session `np.searchsorted` calls plus one `np.bincount` per trial (~100× faster), and
moved parallelism up to the session level. Four per-trial/per-block loops remain, all cheap:

* `bin_spiking_data`: one `np.bincount` per trial — could be a single bincount over all trials by
  offsetting the flat index by `trial * n_clusters * NBINS`.
* `bin_behavior`: one `interp1d` per trial — could be one `np.interp` over a concatenated query
  vector (the guard conditions would need to be vectorised too).
* `trial_number_in_block`: `for b in np.unique(block_id)` is O(n_blocks × n_trials) — a vectorised
  cumcount (`arange(n) - maximum.accumulate(where(newblock, arange(n), 0))`) is O(n).
* The final assembly loop builds a fresh (2, 100) and (4, 100) array per trial — could be built as
  two stacked 3-D arrays and sliced.

Measured, these cost ~0.2 s (binning) and ~1.5 s (behaviour, mostly I/O) per session, so removing
them would not materially change the 162 s total, which is dominated by spike-file I/O.

ii.
```python
for k in range(ntrials):
    a, b = i0[k], i1[k]
    ...
    counts = np.bincount(flat, minlength=n_clusters * NBINS)
    out[k] = counts.reshape(n_clusters, NBINS)
```
```python
for b in np.unique(block_id):
    m = block_id == b
    out[m] = np.arange(m.sum())
```
```python
for k in range(ntrials):
    neural_list.append(binned[k])
    inp = np.empty((2, NBINS), dtype=np.float32)
    inp[0] = tvec
    inp[1] = tnib_keep[k]
```

iii. CONVERSION_NOTES Step 6: *"Code inefficiencies identified: Reference
`get_spike_data_per_interval` / `get_behavior_per_interval` create a process pool per session per
signal and a tqdm bar per trial — dominant cost for 459 sessions. Code speedups added: Vectorised
spike binning (no per-trial pool): 500-trial session bins in < 0.2 s. Parallelism moved up to the
session level, 24 workers."* The remaining loops are not discussed in the notes; their cost is
negligible against the I/O-bound total.

---

## 10-c. What processing does the code repeat multiple times?

i. Three small redundancies, none of them behaviour-changing:

* **Double sort of the spike train.** `merge_probes` already returns spikes sorted by time
  (`np.argsort(..., kind='stable')`), but `convert_session` then filters to finite times and
  argsorts again. The second sort is defensive but always operates on already-ordered data.
* **Filtering every spike attribute.** `load_spiking_data` applies the cluster mask to all keys of
  the `spikes` dict (`{k: v[spike_idx] for k, v in spikes.items()}`), i.e. to `amps` and `depths`
  as well as `times` and `clusters`, even though only the latter two are ever used.
* **Repeated mask recomputation in the plotting path.** `plot_processing` calls
  `np.nonzero(pd_['keep'])` three separate times for the same array.

Additionally `_plotdata` (raw traces, full binned array, all spike times) is attached to the result
dict in `--show-processing` mode and popped again immediately after plotting.

ii.
```python
# merge_probes already sorted:
sort_idx = np.argsort(merged_spikes['times'], kind='stable')
merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}
```
```python
# ... and convert_session sorts again:
finite = np.isfinite(spikes['times'])
st = spikes['times'][finite]
sc = spikes['clusters'][finite]
order = np.argsort(st, kind='stable')
st, sc = st[order], sc[order]
```
```python
selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
```
```python
trial_ids = np.nonzero(pd_['keep'])[0]
kk = trial_ids[min(5, len(trial_ids) - 1)]
...
idx = np.nonzero(pd_['keep'])[0].tolist().index(kk) if kk in np.nonzero(pd_['keep'])[0] else 0
```

iii. Not discussed in CONVERSION_NOTES. The re-sort is implicitly justified as defensive — Step 10
Check 5 lists *"Spike times not sorted after merge | re-sorted with a stable argsort before
`searchsorted` | binning matches raw recomputation exactly"* — since `bin_spiking_data` relies on
`np.searchsorted` and would silently produce wrong trials on unsorted input.

---

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four items, all small relative to the 162 s run:

* **`spikes.amps` and `spikes.depths` are loaded and carried through.** The AI does not override
  `brainbox.io.one.SPIKES_ATTRIBUTES` (default `['clusters', 'times', 'amps', 'depths']`), so two
  extra large arrays per probe are read from disk, boolean-indexed in `load_spiking_data`,
  concatenated and re-sorted in `merge_probes`, and then never used.
* **md5 hash verification is left on.** `loader.load_spike_sorting()` is called without
  `check_hash=False`, so every spike file is re-read to verify its checksum.
* **Unused per-cluster metadata.** `merge_clusters` builds the full 22-column metrics table;
  `clusters['pid']` is attached; `result['acronyms']` (raw Allen acronyms) is carried to assembly
  but never written into the output — only the Beryl mapping is.
* **Diagnostic counters computed for every session** (`n_rt_short`, `n_rt_long`, `n_rt_nan`,
  `n_nochoice`, `n_long_trial`, `n_correct`, `n_incorrect`, `n_units_total`) are stored in
  `metadata['session_info']` and are not used by the decoder. `SessionLoader.load_wheel()` also
  computes `acceleration`, which is discarded.

The AI *did* remove some unnecessary reference work: the `raw_electrophysiology(...).fs` metadata
call, and the model-side `standardize_spike_data` / `StandardScaler` steps.

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()      # no check_hash=False,
                                                              # no SPIKES_ATTRIBUTES override
selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}   # amps/depths too
```
```python
merged_spikes = {k: np.concatenate([s[k] for s in merged_spikes])
                 for k in merged_spikes[0].keys()}
sort_idx = np.argsort(merged_spikes['times'], kind='stable')
merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}
```
```python
'acronyms': clusters['acronym'].to_numpy(),      # never used downstream
```
```python
info['n_rt_short'] = int(np.nansum(rt < 0.08))
info['n_rt_long'] = int(np.nansum(rt > 2.0))
info['n_correct'] = int(np.nansum(fb == 1))
```

iii. Only the removals are documented. CONVERSION_NOTES Step 6: *"`load_spiking_data` /
`merge_probes` / `load_trials_and_mask` are transcriptions of the reference functions (the
`raw_electrophysiology(...).fs` call is dropped: it needs the network and only fed a metadata
field)"*; Step 10 Check 3: *"`standardize_spike_data` (per-bin z-score), `StandardScaler` on
behaviour — inside the *model's* data loader | not applied | ✓ (model-side step; target format
wants raw activity and categorical outputs)."* The retained diagnostic counters are deliberate —
Step 10 Iteration 2: *"added the per-session exclusion counters … to `metadata['session_info']`"* —
so that the trial-attrition table can be audited from the pickle. The extra spike attributes and
the hash check are not mentioned anywhere.
