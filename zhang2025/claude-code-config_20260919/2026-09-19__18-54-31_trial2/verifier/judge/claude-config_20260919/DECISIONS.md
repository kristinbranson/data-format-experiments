# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read through the ONE API against the local cache at `/app/data/one_cache`; no
file is opened by path. The AI does **not** use `one.search` to enumerate sessions. Instead it
reads the reference repository's own freeze file
`/app/code/code_zhang2025/data/bwm_release.csv` (700 rows = 699 probe insertions, 459 eids,
139 subjects) and groups it by `eid`, so each work unit is one session together with the list of
`pid`/`probe_name` of its insertions. This is exactly the table `0_data_caching.py` uses, so the
session/probe inventory is taken from the reference code rather than re-derived. A
`DATALIMIT_SUBSET.csv` restriction is supported if present (it is not present in this
environment, so all 459 sessions were attempted). Per session, one `SessionLoader` instance is
built once and shared for the trials table, the wheel and the camera motion energy, and one
`SpikeSortingLoader` is built per `pid`. The ONE client is created *without* `password=` so that
it answers from the staged REST cache instead of re-authenticating over the (absent) network.
Sessions are processed in parallel with a `ProcessPoolExecutor` (24 workers).

ii.
```python
def get_one():
    from one.api import ONE
    return ONE(base_url='https://openalyx.internationalbrainlab.org',
               silent=True, cache_dir=ONE_CACHE_DIR)


def build_session_list():
    bwm = pd.read_csv(BWM_FREEZE_FILE, index_col=0)
    if os.path.exists(DATALIMIT_FILE):
        sub = pd.read_csv(DATALIMIT_FILE)
        col = 'eid' if 'eid' in sub.columns else sub.columns[0]
        bwm = bwm[bwm.eid.isin(sub[col].astype(str))]
    sessions = []
    for eid, g in bwm.groupby('eid', sort=False):
        g = g.sort_values('probe_name')
        sessions.append({
            'eid': str(eid),
            'pids': [str(p) for p in g.pid],
            'probe_names': list(g.probe_name),
            'subject': str(g.subject.iloc[0]),
            'lab': str(g.lab.iloc[0]),
            'date': str(g.date.iloc[0]),
        })
    sessions.sort(key=lambda s: (s['lab'], s['subject'], s['date']))
    return sessions
```

Per session, one shared loader for every behavioural stream:
```python
sess_loader = SessionLoader(one=one, eid=eid)
trials_df, trials_mask = load_trials_and_mask(one, eid, sess_loader=sess_loader)
...
traces, cam_used = load_behaviour_traces(one, eid, sess_loader=sess_loader)
```

and one spike-sorting loader per insertion:
```python
for pid, pname in zip(session_info['pids'], session_info['probe_names']):
    times, clu, clusters_df, collection = load_spiking_data(one, pid, eid, pname)
```

iii. From CONVERSION_NOTES Step 5: *"Session set — the 459 eids / 699 pids of `bwm_release.csv`
(the same freeze file the reference code uses)."* The AI notes that most datasets on disk are
staged under ALF *revisions* (`alf/#2025-03-03#/…`), so paths must be resolved by ONE rather
than hard-coded, and that pure `mode='local'` fails because the release parquet tables do not
list those revisions. It also documents that `password=` must be omitted because supplying one
forces a network re-authentication that fails offline.

## 1-b. How are the data split into subjects (mice)?

i. The subject name is taken straight from the `subject` column of `bwm_release.csv`, carried on
each session record, and never parsed from a path. At assembly, `subjects` is the sorted set of
unique names over the sessions that survived conversion and `subject_idx` is each session's
index into that list. 136 of the 139 released subjects are represented (three subjects were lost
with the 18 dropped sessions).

ii.
```python
'subject': str(g.subject.iloc[0]),
...
subjects = sorted({r['subject'] for r in results})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_index[r['subject']] for r in results], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 records 139 subjects in the freeze file, matching the data paper's
"We trained 139 mice". No derivation is needed because the release table already carries a
unique subject id per session.

## 1-c. How are the data split into sessions?

i. A session is the unit `bwm_release.csv` is keyed on (one row per insertion, `eid` repeated
for two-probe sessions), so the split is a `groupby('eid')`. The two insertions of a session are
kept together in one record and later merged into a single population, i.e. a session — not a
probe — is the unit of the converted dataset. Sessions are sorted by (lab, subject, date) for a
deterministic output order.

ii.
```python
for eid, g in bwm.groupby('eid', sort=False):
    g = g.sort_values('probe_name')
    sessions.append({'eid': str(eid), 'pids': [str(p) for p in g.pid],
                     'probe_names': list(g.probe_name), ...})
sessions.sort(key=lambda s: (s['lab'], s['subject'], s['date']))
```

iii. CONVERSION_NOTES Step 2: the freeze is *"699 pids, 459 eids, 139 subjects — exactly the
public BWM release"*. Grouping probes by `eid` reproduces the reference `prepare_data`, which
resolves `one.eid2pid(eid)` and merges the probes of a session *"to account for the fact that
data from the probes recorded in the same session are not statistically independent"*.

## 1-d. How are the data split into trials?

i. The trials table returned by `SessionLoader.load_trials()` has one row per trial, so no split
has to be derived. Each trial becomes the interval
`[stimOn_times − 0.5 s, stimOn_times + 1.5 s)`, from which the spike, wheel and camera streams
are cut. `interval_begs` is built once from the whole (uncurated) table and then indexed by the
keep mask, so every stream uses the same trial definition.

ii.
```python
align_times = trials_df[ALIGN_TIME].to_numpy(dtype=float)
interval_begs = align_times + TIME_WINDOW[0]
...
interval_ends = interval_begs + n_bins * binsize
```

iii. Matches the reference `bin_spiking_data`, which builds
`intervals = np.vstack([trials_df[align_time] + time_window[0],
trials_df[align_time] + time_window[1]]).T` with
`params = {'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}`.

## 1-e. How are trials filtered based on quality controls?

i. Two masks are combined. (1) A verbatim re-implementation of the reference
`load_trials_and_mask` with the settings `prepare_data` uses: reaction time
`firstMovement_times − stimOn_times` must lie in [0.08, 2.0] s; trial length
`feedback_times − goCue_times` must be ≤ 10 s (`max_trial_len=10.0`); none of
`stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType`
may be NaN; and no-response trials (`choice == 0`) are dropped. (2) A behaviour-coverage mask
from the port of `get_behavior_per_interval`: a trial is dropped if the wheel or the whisker
trace is absent in the window, starts more than one bin (20 ms) after the window start, ends more
than one bin before the window end, or still contains a non-finite value after interpolation.
`np.isfinite(interval_begs)` is added for safety. A session with fewer than 2 surviving trials
is dropped entirely. Result: 187,934 of 285,031 raw trials kept (65.9 %).

ii.
```python
query  = f'(firstMovement_times - stimOn_times < {MIN_RT})'
query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
for event in NAN_EXCLUDE:
    query += f' | {event}.isnull()'
query += ' | (choice == 0)'
mask = ~trials.eval(query)
```

```python
keep = np.asarray(trials_mask, dtype=bool).copy()
keep &= np.isfinite(interval_begs)
for name in beh_good:
    keep &= beh_good[name]
keep_idx = np.flatnonzero(keep)
if len(keep_idx) < MIN_TRIALS_PER_SESSION:
    raise RuntimeError(f'only {len(keep_idx)} usable trials')
```

and the coverage test inside `bin_behaviour`:
```python
if len(tv) == 0:                                    continue  # 'target data not present'
if np.abs(interval_begs[k] - tt[0])  > binsize:     continue  # 'starts too late'
if np.abs(interval_ends[k] - tt[-1]) > binsize:     continue  # 'ends too early'
...
if not np.all(np.isfinite(y)):                      continue
```

iii. CONVERSION_NOTES Step 5 decision 3: *"`load_trials_and_mask` defaults + `max_trial_len=10.0`
(reference code) + trials whose wheel/whisker traces do not cover the full window (reference
`get_behavior_per_interval` / `align_spike_behavior`)."* The NaN-drop is justified as a required
deviation: *"the reference keeps them (`allow_nans=True`) and imputes the trial mean at
model-fitting time. The decoder here rejects non-finite values outright."* Step 10 Check 3
reports the port's mask is **bit-identical** to the actual reference function on a real session
(407/565 trials kept by both).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` of every insertion of a session, loaded through
`SpikeSortingLoader.load_spike_sorting()`. The merged cluster table
(`SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()`) supplies `label` (the
QC score) and `acronym` (the Allen location), which are used only for curation and for
`brain_region_idx`, not for the activity itself. The reference's `raw_electrophysiology(...).fs`
call is deliberately omitted (network-only, used just to report the sampling rate).

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
iok = clusters_labeled['label'] >= qc
sel_clusters = clusters_labeled[iok]
spike_idx, ib = ismember(spikes['clusters'], sel_clusters.index)
sel_clusters = sel_clusters.reset_index(drop=True)
sel_times = spikes['times'][spike_idx]
sel_cl = sel_clusters.index.to_numpy()[ib].astype(np.int64)
```

iii. CONVERSION_NOTES Step 1 documents `load_spiking_data` as the reference loader and Step 6
states the port *"drops the `raw_electrophysiology` call (network-only, used solely to report
`sampling_freq`)"*.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 100 non-overlapping 20 ms bins spanning the trial window, giving one
integer count per unit per bin; the values are stored as **float32 spike counts per 20 ms bin**
(not converted to Hz — `metadata['neural_units'] = 'spike counts per 20 ms bin'`). No smoothing,
no normalisation, no z-scoring (the reference does its per-time-bin standardisation model-side,
in `data_loader_utils.standardize_spike_data`, not at caching time). Each probe is binned
separately and the resulting `(n_trials, n_units, 100)` arrays are concatenated along the unit
axis, so the two probes of a session become a single pooled population in probe-name order. The
binning is a vectorised rewrite of `get_spike_data_per_interval`: one `searchsorted` per trial
edge, one flat linear index per spike, one `np.bincount` for the whole session.

ii.
```python
i0 = np.searchsorted(times, interval_begs, side='left')
i1 = np.searchsorted(times, interval_ends, side='left')
flat_idx = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1) if b > a])
trial_of_spike = np.repeat(np.arange(n_trials), counts)
t_rel = times[flat_idx] - interval_begs[trial_of_spike]
bin_of_spike = np.floor(t_rel / binsize).astype(np.int64)
np.clip(bin_of_spike, 0, n_bins - 1, out=bin_of_spike)
lin = ((trial_of_spike * n_clusters) + clusters[flat_idx]) * n_bins + bin_of_spike
out += np.bincount(lin, minlength=n_trials * n_clusters * n_bins).reshape(
    n_trials, n_clusters, n_bins).astype(np.float32)
```

```python
neural = np.concatenate(neural_trials, axis=1)          # (n_trials, n_neurons, T)
regions = np.concatenate(region_acronyms)
```

iii. CONVERSION_NOTES Step 5 decision 4: *"Binning — identical arithmetic to `bincount2D`
(`floor((t − t_beg)/binsize)`, `t_beg <= t < t_end`), vectorised over trials for speed; verified
against the reference implementation."* Step 10 Check 3 reports `np.allclose → True,
max abs diff = 0` over 60 trials × all clusters × 100 bins against the real
`get_spike_data_per_interval`, and Check 2 reports a zero-difference match against an
independent `np.histogram` reconstruction from the raw `spikes.times` files.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Four cuts, in order. (1) `clusters.label >= 1` — the "well-isolated" units of the data paper
(passing all three RIGOR metrics); this is the reference loader's documented `qc=1` option, and
the AI verified it reproduces the paper's 75,708 of 621,733 exactly. (2) Grey matter only: a
unit is dropped if its **Beryl** acronym is `root` or `void`. (3) Units that emit no spike in
the session are dropped, reproducing the reference `bin_spiking_data`'s
`clusters_used_in_bins = np.unique(regclu)`. (4) A session-level cut: a session with fewer than
5 surviving neurons is dropped (3 sessions). The result is 62,757 neurons over 441 sessions
(82.9 % of the 75,708 good units survive the grey-matter cut), mean 142.3 per session.

ii.
```python
QC_LABEL = 1.0
NON_GREY_MATTER = ('root', 'void')
MIN_NEURONS_PER_SESSION = 5
```

```python
beryl = np.asarray(br.acronym2acronym(clusters_df['acronym'].to_numpy(),
                                      mapping='Beryl'), dtype=object)
has_spikes = np.zeros(len(clusters_df), dtype=bool)
if clu.size:
    has_spikes[np.unique(clu)] = True
sel = (~np.isin(beryl, NON_GREY_MATTER)) & has_spikes
...
if neural.shape[1] < MIN_NEURONS_PER_SESSION:
    raise RuntimeError(f'only {neural.shape[1]} well-isolated grey-matter neurons '
                       f'(< {MIN_NEURONS_PER_SESSION})')
```

iii. CONVERSION_NOTES Step 5 decision 2 records this as a **deliberate deviation** from
`0_data_caching.py`, which calls `load_spiking_data` with `qc=None` and therefore bins all
621,733 units: *"(a) the data paper defines 'neurons' as exactly these units; (b) multi-unit
clusters are not neurons and are excluded from every analysis in the data paper; (c) … keeping
621,733 multi-unit clusters would make the converted dataset ~110 GB."* The grey-matter cut is
attributed to the data paper's "Neurons and brain regions" section, and the reference repo does
the same exclusion (`data_loader_utils.py:252` filters `roi not in ['root', 'void']`). The
≥ 5-neuron rule is taken from the data paper's *"at least five well-isolated neurons per
session"* (stated there per region) and was added in Step 10 Iteration 1 to remove sessions that
produced all-zero trials. Step 12 Check 2 reports a control experiment: using all units instead
of `label >= 1` raises per-bin choice accuracy by only 0.019 for an 8.4× larger dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams are already expressed in seconds on one synchronised session clock, so
alignment is a subtraction. Each trial's window begins at `stimOn_times − 0.5 s`; spikes are
selected with `t_beg <= t < t_beg + 100·0.02` and their bin index is
`floor((t − t_beg)/0.02)`, so bin 0 starts exactly 0.5 s before stimulus onset and t = 0 falls
on the boundary between bins 24 and 25. `metadata['temporal_alignment_event']` is
`'visual stimulus onset (trials.stimOn_times)'` with `off_start = −0.5`, `off_end = +1.5`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
align_times = trials_df[ALIGN_TIME].to_numpy(dtype=float)
interval_begs = align_times + TIME_WINDOW[0]
...
t_rel = times[flat_idx] - interval_begs[trial_of_spike]
bin_of_spike = np.floor(t_rel / binsize).astype(np.int64)
```

iii. CONVERSION_NOTES Step 4 resolves this against all three sources: the reference `params`
(`align_time='stimOn_times'`, `time_window=(-.5, 1.5)`), the method paper (*"For choice, we
align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s
post-onset"*), and the decoder task's explicit "Temporally align based on stimulus onset". Step
7's plot panel 6 is offered as evidence: the population PSTH is flat before t = 0 and rises
sharply at t = 0, peaking just after the median first-movement time; Step 12 adds that the
decoder's own choice-accuracy time course peaks at +0.23…+0.31 s, the IBL median reaction time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, `N_BINS = ceil(2.0/0.02) = 100` per trial, identical for every trial and session;
`metadata['time_bin_size'] = 20.0` (ms). The spikes are binned once, directly at 20 ms — there
is no rebinning, resampling or decimation of the neural data.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. CONVERSION_NOTES Step 4 flags and resolves a source conflict: `methods.txt` contains one
sentence saying *"we segment neural activity into 50-ms non-overlapping time bins"*, which
contradicts both the released code (`binsize: 0.02`) and the method paper's own *"Recordings are
split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps."* The AI
chose 20 ms, noting it follows the code and the paper body and is *"required here because two of
the four decoder outputs are time-varying behaviours that must be resolved within the trial."*

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Not derived from raw data at all — it is the conversion's own bin grid, fixed by
`ALIGN_TIME = 'stimOn_times'`, `TIME_WINDOW = (−0.5, 1.5)` and `BINSIZE = 0.02`. The same
100-element vector is used for every trial of every session (the per-trial offset is implicit,
because everything is expressed relative to that trial's `stimOn_times`).

ii.
```python
bin_centres = (TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)).astype(np.float32)
inputs[:, 0, :] = bin_centres[None, :]
```

iii. CONVERSION_NOTES Step 5: the variable is listed as *"(new; required by the task)"*; its
window and bin size come from the reference `params`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond computing bin centres: `−0.5 + 0.02·(i + 0.5)` for i = 0…99, i.e. the values run
from −0.49 s to +1.49 s. It is emitted as a time-varying float32 row of the `(2, 100)` input
array (a continuous ramp, not a binary onset indicator, since the variable is "time since
onset" rather than "onset time").

ii.
```python
inputs = np.empty((n_keep, 2, N_BINS), dtype=np.float32)
inputs[:, 0, :] = bin_centres[None, :]
```

iii. CONVERSION_NOTES Step 5: *"bin index → time … `-0.5 + 0.02·(i+0.5)` s (bin centre),
−0.49 … +1.49 … continuous, time-varying"*, and Step 10 Check 5 explicitly notes *"The input's
time value is the bin centre, so its range is [−0.49, +1.49] rather than [−0.5, +1.5]."*

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural grid: element i of the input is the centre of exactly the same 20 ms bin
that column i of the neural matrix counts spikes in, for the same trial's `stimOn_times`. Index
i therefore refers to the same instant in both arrays by construction; no interpolation or
offset is applied.

ii.
```python
bin_centres = (TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)).astype(np.float32)
# neural bin i spans [interval_begs + 0.02*i, interval_begs + 0.02*(i+1))
bin_of_spike = np.floor((times[flat_idx] - interval_begs[trial_of_spike]) / binsize)
```

iii. CONVERSION_NOTES Step 5 "Trial geometry": *"spike bin i covers
`[stimOn − 0.5 + 0.02·i, stimOn − 0.5 + 0.02·(i+1))`"* and the input is that bin's centre.
`assert neural.shape[2] == N_BINS` and `assert neural.shape[0] == n_keep == inputs.shape[0]`
enforce the shape correspondence in code.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft` alone. The trials table carries no block identifier, so blocks
are recovered as maximal runs of constant `probabilityLeft`, and the input is the position of
the trial within its run.

ii.
```python
tnb_all = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
```

iii. CONVERSION_NOTES Step 5 variable map: *"`probabilityLeft` run-lengths → `input[1]`
`trial_number_in_block`"*, listed as *"(new; required by the task)"*. Step 10 Check 5 confirms
the recovered blocks behave as the data paper describes: the counter restarts at 0 on every
`probabilityLeft` change, the first block reaches 89 (the 90 unbiased trials at session start)
and the maximum over the dataset is 98 (blocks are 20–100 trials).

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based counter that increments while `probabilityLeft` is unchanged and resets to 0 when
it changes (with NaN treated as equal to NaN). Crucially it is computed on the **full, uncurated**
trials table and only afterwards indexed by the keep mask, so a trial that is later dropped still
advances the count and the number reflects the animal's true position in the block. It is stored
as float32 and broadcast constant across all 100 time bins.

ii.
```python
def trial_number_in_block(probability_left):
    p = np.asarray(probability_left, dtype=float)
    out = np.zeros(len(p), dtype=np.float64)
    counter = 0
    for i in range(1, len(p)):
        same = (p[i] == p[i - 1]) or (np.isnan(p[i]) and np.isnan(p[i - 1]))
        counter = counter + 1 if same else 0
        out[i] = counter
    return out
```

```python
inputs[:, 1, :] = tnb_all[keep_idx].astype(np.float32)[:, None]
```

iii. CONVERSION_NOTES Step 5 decision 6: *"`trial_number_in_block` is computed before trial
curation, so it reflects the animal's true position in the block even when neighbouring trials
are excluded."* Step 12 Check 3 adds a leakage argument: *"the only per-trial input,
`trial_number_in_block`, is independent of block identity and therefore of every output."*

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The single column `trials.choice`, which takes +1, −1 or 0. Trials with `choice == 0` (no
response) have already been removed by the trial mask, so only ±1 reaches the encoder.

ii.
```python
choice = trials_df['choice'].to_numpy()[keep_idx]        # +1 = left, -1 = right
```

iii. CONVERSION_NOTES Step 4 records that the sign convention was *verified empirically from the
data*: correct trials with a left stimulus all have `choice == +1`. Step 5 decision 7 states the
resulting mapping.

## 5-b. What processing is involved in computing `output` *Choice*?

i. A recode only: `choice == +1` (leftward) → class 0, `choice == −1` (rightward) → class 1,
implemented as `(choice < 0)`. It is a per-trial label broadcast constant across all 100 bins
and stored in row 0 of the `(4, 100)` output array, with `output_values[0] = ['left', 'right']`.

ii.
```python
choice_cls = (choice < 0).astype(np.int64)               # left -> 0, right -> 1
...
outputs[:, 0, :] = choice_cls[:, None]
```

iii. The decoder task specifies "Choice, binary, per-trial, left = 0, right = 1". The
time-broadcast follows the format guidance *"If at all possible, make it time-varying"*
(CONVERSION_NOTES Step 5). Step 10 Check 2 reports an independent raw-file re-derivation of
`choice` for every trial of 3 sessions matching exactly.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The single column `trials.probabilityLeft`, the block prior the task holds constant within a
block. Trials whose value is NaN are already removed by the trial mask.

ii.
```python
pleft = trials_df['probabilityLeft'].to_numpy()[keep_idx]
```

iii. CONVERSION_NOTES Step 5 maps `trials.probabilityLeft → output[1] prior_prob_left`, noting
the reference `bin_behaviors` exposes the same column under the name `block`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A recode of the three admissible values to classes via `np.isclose`: 0.2 → 0, 0.5 → 1,
0.8 → 2. Any value that matches none of the three raises, so an unexpected prior cannot be
silently mislabelled (it never fired on the 459 sessions). The label is per-trial, broadcast
constant across the 100 bins, `output_values[1] = ['0.2', '0.5', '0.8']`.

ii.
```python
pleft_cls = np.full(len(pleft), -1, dtype=np.int64)
for val, cls in ((0.2, 0), (0.5, 1), (0.8, 2)):
    pleft_cls[np.isclose(pleft, val)] = cls
if np.any(pleft_cls < 0):
    bad = np.unique(pleft[pleft_cls < 0])
    raise RuntimeError(f'unexpected probabilityLeft values {bad}')
...
outputs[:, 1, :] = pleft_cls[:, None]
```

iii. The mapping is dictated by the decoder task. `np.isclose` rather than `==` is used because
the column is float. CONVERSION_NOTES Step 9 uses the resulting distribution as a sanity check:
0.5 accounts for 14.0 % of retained trials, matching the expectation of 90 unbiased trials in a
~645-trial session.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position` / `_ibl_wheel.timestamps`, loaded through `SessionLoader.load_wheel()`,
which returns an evenly sampled `times`/`position`/`velocity`/`acceleration` frame; the speed is
`np.abs(velocity)`. This is exactly the reference `load_target_behavior(one, eid, 'wheel-speed')`.

ii.
```python
sess_loader.load_wheel()
traces['wheel_speed'] = (sess_loader.wheel['times'].to_numpy(),
                         np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. CONVERSION_NOTES Step 1 documents `load_target_behavior`'s `'wheel-speed'` branch as
`abs(SessionLoader.wheel['velocity'])`, and Step 3 records that `SessionLoader` interpolates the
raw wheel encoder onto a uniform 1 kHz grid and computes a smoothed velocity from it.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) `SessionLoader.load_wheel()` interpolates the event-driven wheel encoder
onto a uniform grid and differentiates it into a filtered velocity; the absolute value is taken.
(2) Per trial the trace is sliced to the window and linearly interpolated (`interp1d`, with
`fill_value='extrapolate'`) onto the 100 sample points `linspace(t_beg + 0.02, t_end, 100)` —
i.e. the **right edge** of each 20 ms bin, not its centre. (3) The resulting
`(n_trials, 100)` matrix is discretised into 3 classes (see 7-c). No smoothing or normalisation
is added on top of what `SessionLoader` already does.

ii.
```python
tt = target_times[idxs_beg[k]:idxs_end[k]]
tv = target_vals[idxs_beg[k]:idxs_end[k]]
...
x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
y = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. CONVERSION_NOTES Step 6 describes `bin_behaviour` as a *"line-for-line port of
`get_behavior_per_interval`'s `interpolate_behavior`"*, and Step 10 Check 3 reports the port's
values and good-trial masks are identical to the real reference function (max diff = 0) for both
the wheel and the camera. Step 5 "Trial geometry" states the right-edge convention explicitly:
*"continuous behaviour for bin i is interpolated at the bin's right edge … exactly
`get_behavior_per_interval`'s `np.linspace(t_beg + binsize, t_end, n_bins)`."*

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session terciles. The two edges are the 1/3 and 2/3 quantiles of *that session's* pooled
binned wheel-speed values over all retained trials (n_trials × 100 samples), and the class is
`0` for `v ≤ e1`, `1` for `e1 < v ≤ e2`, `2` for `v > e2`. The `≤` on the lower edge deliberately
keeps the mass point at rest (speed ≈ 0) inside the "low" class. A degenerate case where
`e1 == e2` is repaired by splitting the remaining above-edge mass at its median. The edges are
stored per session in `metadata['session_info'][…]['wheel_speed_edges']`. Achieved marginals over
the whole dataset: 0.333 / 0.333 / 0.333.

ii.
```python
def discretize_terciles(values):
    v = np.asarray(values, dtype=np.float64).ravel()
    e1, e2 = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
    if e1 == e2:
        above = v[v > e1]
        e2 = np.quantile(above, 0.5) if above.size else e1
        if e2 == e1:
            above = v[v > e1]
            e2 = above.min() if above.size else e1
    return float(e1), float(e2)


def apply_terciles(values, e1, e2):
    return ((values > e1).astype(np.int64) + (values > e2).astype(np.int64))
```

```python
ws_edges = discretize_terciles(ws)
ws_cls = apply_terciles(ws, *ws_edges)
outputs[:, 2, :] = ws_cls
```

iii. CONVERSION_NOTES Step 5 decision 5: *"per-session terciles … (a) whisker motion energy is in
arbitrary camera-dependent units … so a global threshold would mean different things in
different sessions; (b) the reference decoding code likewise standardises each behaviour per
session (`StandardScaler` in `SingleSessionDataset`); (c) terciles give ≈ 1/3 per class so that
chance = balanced-accuracy chance = 1/3."* Step 10 Check 5 documents the one degenerate session
(`5b44c40f…`, 78 % of its samples exactly 0).

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The trace is evaluated on the same stimulus-onset-locked 100-bin grid as the spikes, from the
same `interval_begs = stimOn_times − 0.5`, so element i of the wheel row and column i of the
neural matrix describe the same 20 ms bin of the same trial. Within a bin the sample is taken at
the bin's right edge rather than its centre (the reference convention), a 10 ms offset relative
to the bin centre. The wheel and the spikes are on the same session clock, so no further
alignment is applied; trials where the wheel does not cover the window are dropped rather than
padded.

ii.
```python
interval_begs = align_times + TIME_WINDOW[0]          # shared by spikes and behaviour
...
vals, good = bin_behaviour(tt, tv, interval_begs)
...
x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
```

iii. Step 5 "Trial geometry" and Step 13's summary table: *"Behaviour sampling — linear
interpolation at each bin's right edge — exactly `get_behavior_per_interval`."* The
`--show-processing` figure (panel 1) overlays the raw |wheel velocity| trace with the
interpolated bin values and marks `stimOn` at t = 0; CONVERSION_NOTES Step 7 reports *"the red
samples lie exactly on the raw trace … no temporal misalignment."*

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy` with `_ibl_leftCamera.times`, loaded via
`SessionLoader.load_motion_energy(views=['left'])` and read from the `whiskerMotionEnergy`
column; if the left camera is unavailable the right camera is used instead. The released
motion-energy trace is used as-is. 434 of the 441 converted sessions used the left camera, 7 the
right; 14 sessions with neither were dropped.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        me = sess_loader.motion_energy[key]
        traces['whisker_motion_energy'] = (me['times'].to_numpy(),
                                           me['whiskerMotionEnergy'].to_numpy())
        cam_used = view
        break
    except Exception:
        continue
if cam_used is None:
    raise RuntimeError('no whisker motion energy available (neither camera)')
```

iii. CONVERSION_NOTES Step 1 documents the reference `bin_behaviors`: *"For
`whisker-motion-energy` it uses the left camera, falling back to the right camera when the left
is unavailable."* Step 3 records the data paper's definition (mean absolute difference between
adjacent video frames in a box anchored between nose tip and eye, left camera at 60 Hz).

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used unfiltered and unnormalised. It goes through the identical
`bin_behaviour` path as the wheel: sliced to the trial window, linearly interpolated onto the
100 bin right edges, then discretised into 3 per-session classes. Trials whose camera trace
starts >20 ms late, ends >20 ms early, is absent, or yields non-finite values are dropped.

ii.
```python
for name, (tt, tv) in traces.items():
    vals, good = bin_behaviour(tt, tv, interval_begs)
    beh_vals[name] = vals
    beh_good[name] = good
...
wme = beh_vals['whisker_motion_energy'][keep_idx]
```

iii. Same justification as the wheel (Step 6 / Step 10 Check 3: bit-identical to
`get_behavior_per_interval`). CONVERSION_NOTES Step 5 decision 5 notes the camera-dependent
units (left 60 Hz 1280×1024 vs right 150 Hz 640×512) as the reason nothing global is applied to
the raw values.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: per-session 1/3 and 2/3 quantiles of the session's pooled binned
values, classes `v ≤ e1` / `e1 < v ≤ e2` / `v > e2`, with the degenerate-edge repair. Edges are
stored per session as `whisker_me_edges`. Dataset marginals: 0.334 / 0.333 / 0.333.

ii.
```python
wme_edges = discretize_terciles(wme)
wme_cls = apply_terciles(wme, *wme_edges)
outputs[:, 3, :] = wme_cls
```

iii. CONVERSION_NOTES Step 5 decision 5 (per-session terciles, camera-dependent units); Step 10
Check 5 reports the single session whose trace is 78 % exactly zero, giving 0.784/0.108/0.108
instead of equal thirds, and calls that *"the best available 3-way split of that trace."*

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same grid as the neural data and the wheel: the camera frame times are on the same session
clock, and the trace is evaluated at the right edge of each of the 100 stimulus-onset-locked
20 ms bins. Element i of the whisker row corresponds bin-for-bin to column i of the neural
matrix. Because the left camera runs at 60 Hz (16.7 ms/frame) and the right at 150 Hz, the
interpolation is an up-sampling for the left camera and a down-sampling for the right.

ii.
```python
vals, good = bin_behaviour(tt, tv, interval_begs)      # same interval_begs as the spikes
...
x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
y = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. CONVERSION_NOTES Step 7 plot panel 2 overlays the raw 60 Hz camera trace with the
interpolated bin values for a single trial and confirms they coincide with `stimOn` at t = 0.
Step 12 adds that the whisker decoding accuracy time course peaks at +0.23 s, *"again consistent
with stimulus-locked alignment."*

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is dropped, never imputed, at whatever level it occurs, and every drop is
recorded. (a) Trials: NaNs in any of the six required trial events, and windows not fully
covered by the wheel or camera trace, remove the trial (the reference's "target data not
present / starts too late / ends too early" tests are reproduced verbatim, plus a post-
interpolation finiteness test that the reference does not have, because the decoder rejects
non-finite values). (b) Probes: an insertion whose cluster table is empty after curation, or
whose clusters are all non-grey-matter, is skipped and the session continues on its other probe.
(c) Sessions: a session raises and is recorded as a failure if it has no whisker motion energy
from either camera (14 sessions), fewer than 2 usable trials (1 session), no surviving neuron, or
fewer than 5 neurons (3 sessions). A session that raises for any other reason is caught in the
worker, logged with its traceback, and does not abort the run. (d) Unexpected values: a
`probabilityLeft` outside {0.2, 0.5, 0.8} raises rather than being silently mapped. (e) All 18
failures are listed with their reason in `metadata['failed_sessions']`, and in-line assertions
check shapes, finiteness and non-negativity before a session is returned.

ii.
```python
def _worker(args):
    session_info, show_processing = args
    try:
        return convert_session(session_info, show_processing=show_processing)
    except Exception as e:
        return {'eid': session_info['eid'], 'error': f'{type(e).__name__}: {e}',
                'traceback': traceback.format_exc()}
```

```python
if len(clusters_df) == 0:
    continue
...
if not sel.any():
    continue
...
if not neural_trials:
    raise RuntimeError('no neurons passed curation')
```

```python
assert neural.shape[0] == n_keep == inputs.shape[0] == outputs.shape[0]
assert neural.shape[2] == N_BINS
assert len(regions) == neural.shape[1]
assert np.all(np.isfinite(neural)) and np.all(np.isfinite(inputs))
assert outputs.min() >= 0
```

```python
'failed_sessions': [{'eid': f['eid'], 'error': f['error']} for f in failures],
```

iii. CONVERSION_NOTES Step 9: *"No data was lost silently: every one of the 459 sessions is
either present in `data['neural']` or listed with its failure reason in
`metadata['failed_sessions']`."* Step 10 Check 3 deviation 4 explains the NaN drop
(*"the reference keeps them (`allow_nans=True`) and imputes the trial mean at model-fitting time.
The decoder here rejects non-finite values outright"*) and deviation 5 the session drop
(*"here whisker ME is a required output for every trial"*). The 16 residual "all neural data is
zero" warnings are explicitly analysed and left in place because removing them would mean
censoring trials on the basis of their neural activity.

## 10-a. What are the most time-consuming steps of the code?

i. Measured per-session timings are printed for every session (`timings` dict). Loading and
binning the spike sorting dominates: ~3.1 s for a one-probe session and ~6.5 s for two probes,
of which nearly all is reading `spikes.times`/`spikes.clusters` (hundreds of MB per probe) off
disk plus `merge_clusters`; the vectorised binning itself is a small fraction. Next is loading
the behaviour (~0.6 s: wheel interpolation and camera motion energy), then the trials table
(~0.4–0.5 s). Behaviour binning is negligible (~0.05 s). Outside the per-session work, the final
`pickle.dump` of the 11.72 GB dictionary took 18.6 s of the 274 s total wall clock. The whole
459-session conversion ran in 274 s with 24 workers.

ii.
```python
t0 = time.time()
... loader.load_spike_sorting() / merge_clusters / bin_spikes ...
timings['spikes'] = time.time() - t0
...
print(f'[{i + 1}/{n}] {r["eid"]} {r["subject"]}: '
      f'{r["n_trials_kept"]}/{r["n_trials_raw"]} trials, {r["n_neurons"]} neurons, '
      f'{r["runtime"]:.1f}s ({tm})  [{el:.0f}s elapsed, eta ...]', flush=True)
```

iii. CONVERSION_NOTES Step 7 gives the timing table and the projection
(*"≈ 8 s × 459 / 24 ≈ 3 min + pickle write … Comfortably below the 15-minute budget, so no
further optimisation was needed"*), and attributes the spike cost to file I/O
(*"spike binning 3–7 s/probe (dominated by file I/O)"*).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The expensive loop of the reference — one `bincount2D` call per trial inside a
`multiprocessing.Pool` spawned per session per signal — was removed: `bin_spikes` computes the
whole `(n_trials, n_clusters, n_bins)` array of a probe with two `searchsorted` calls and a
single `np.bincount`. Three loops remain, all cheap and none of them documented as remaining
inefficiencies: (1) the per-trial loop in `bin_behaviour`, kept because it is a line-for-line
port of the reference (it calls `interp1d` once per trial; ~0.05 s/session, so nothing to gain);
(2) the Python loop in `trial_number_in_block`, which is a run-length index that could be done
with `np.cumsum` on a change mask in the style of the reference's
`(probabilityLeft != probabilityLeft.shift()).cumsum()` + `groupby().cumcount()` — it runs once
per session over ~645 elements, so the cost is negligible; (3) the
`np.concatenate([np.arange(a, b) for a, b in zip(i0, i1) if b > a])` list comprehension inside
the otherwise-vectorised `bin_spikes`, which materialises one small array per trial and could be
replaced by an offset-`cumsum` construction.

ii. The loop that was vectorised away (reference → AI):
```python
# reference: one pooled call per trial
binned_spikes_tmp, t_idxs, cluster_idxs = bincount2D(
    times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end])
```
```python
# AI: one bincount for all trials of a probe
lin = ((trial_of_spike * n_clusters) + clusters[flat_idx]) * n_bins + bin_of_spike
out += np.bincount(lin, minlength=n_trials * n_clusters * n_bins).reshape(...)
```

The three remaining loops:
```python
flat_idx = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1) if b > a])
```
```python
for k in range(n_trials):
    ...
    y = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x_interp)
```
```python
for i in range(1, len(p)):
    same = (p[i] == p[i - 1]) or (np.isnan(p[i]) and np.isnan(p[i - 1]))
    counter = counter + 1 if same else 0
```

iii. CONVERSION_NOTES Step 6 lists the reference inefficiencies (*"each spawn a
`multiprocessing.Pool` per session per signal and bin one trial at a time with `bincount2D`. For
459 sessions that is hours of pool start-up alone"*) and the speed-ups applied (*"spike binning
vectorised with one `np.bincount` over all trials of a probe … behaviour interpolation done
in-process (no pool) — ~0.05 s per session … parallelism moved up to the session level"*). The
remaining loops are not discussed; the implicit justification for `bin_behaviour` is fidelity —
Step 10 Check 3 requires it to be bit-identical to the reference function.

## 10-c. What processing does the code repeat multiple times?

i. Several per-session objects are rebuilt that could be built once per worker process. (1)
`get_one()` constructs a fresh ONE client at the top of every `convert_session` call, i.e. 459
times rather than 24 (once per worker) — the reference solution avoids this with a
`ProcessPoolExecutor(initializer=…)`. (2) `BrainRegions()` is instantiated inside
`convert_session`, so the Allen/Beryl region tables are re-read once per session. (3) The
`brainbox`/`iblatlas`/`scipy` imports are inside the functions, so they are re-resolved on every
call (cheap after the first, but repeated). (4) `interp1d` is constructed once per trial per
behaviour rather than once per behaviour. Within a session the code is careful *not* to repeat
work: a single `SessionLoader` is shared across the trials, wheel and motion-energy loads, so the
session lookup happens once instead of four times. None of these repeats is documented.

ii.
```python
def convert_session(session_info, show_processing=False, plot_dir='/app'):
    ...
    one = get_one()                      # new ONE client for every session
    ...
    from brainbox.io.one import SessionLoader
    from iblatlas.regions import BrainRegions
    ...
    br = BrainRegions()                  # region tables re-read for every session
```
```python
# the one place repetition IS avoided:
sess_loader = SessionLoader(one=one, eid=eid)
trials_df, trials_mask = load_trials_and_mask(one, eid, sess_loader=sess_loader)
traces, cam_used = load_behaviour_traces(one, eid, sess_loader=sess_loader)
```

iii. CONVERSION_NOTES Step 6 documents only the repetition that was removed: *"only the two
required behaviour traces are loaded, and the `SessionLoader` instance is shared between the
trials / wheel / motion-energy loads (one ONE session lookup instead of four)."* The per-session
re-creation of the ONE client and of `BrainRegions` is not mentioned; the whole conversion
finished in 274 s, so the AI had no reason to pursue it.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three items, none of them documented. (1) `bin_behaviour` is run over **all** raw trials of a
session (mean 646) even though the reference trial mask has already excluded ~34 % of them, so
roughly a third of the wheel and camera interpolation is thrown away by `keep_idx`. This does
mirror the reference (`get_behavior_per_interval` also runs over the full `trials_df`, and
`align_spike_behavior` intersects the masks afterwards), and it is cheap (~0.05 s/session). (2)
The outputs are built and pickled as `int64` although they only ever hold the values 0–2; `int8`
would cut that array 8-fold (≈ 600 MB → 75 MB of the 11.72 GB file). Similarly the neural array
holds integer spike counts in float32. (3) Diagnostic quantities are computed and stored that no
downstream step reads: `n_trials_raw`, `lab`, `date`, `spike_sorter`/`collection`, the per-step
`timings` dict, and the per-session tercile edges (the last three are at least useful
documentation). The AI did remove the reference's genuinely large waste — `load_anytime_behaviors`
loads six continuous behaviours of which two are used, and `load_spiking_data` streams raw AP
data just to read a sampling rate — neither of which the AI's code does.

ii.
```python
# binned for every raw trial, then subset
for name, (tt, tv) in traces.items():
    vals, good = bin_behaviour(tt, tv, interval_begs)      # interval_begs = ALL trials
    beh_vals[name] = vals
...
ws = beh_vals['wheel_speed'][keep_idx]                      # ~2/3 of the work is kept
```
```python
outputs = np.empty((n_keep, 4, N_BINS), dtype=np.int64)     # values are only 0..2
```
```python
'n_trials_raw': int(n_trials_raw),
'spike_sorter': sorted(sorters),
'timings': timings,
'runtime': time.time() - t_start,
```

iii. CONVERSION_NOTES Step 6 documents the reference's discarded work that the AI avoided
(*"`prepare_data` loads six continuous behaviours (`load_anytime_behaviors`) of which only two
are used"*, *"`load_spiking_data` streams raw AP data just to read the sampling rate"*). The
behaviour-coverage mask is part of the trial mask, which is the implicit reason the behaviour is
binned before the mask is finalised. The dtype and metadata items are not discussed; Step 9
simply records the resulting 11.72 GB file size.
