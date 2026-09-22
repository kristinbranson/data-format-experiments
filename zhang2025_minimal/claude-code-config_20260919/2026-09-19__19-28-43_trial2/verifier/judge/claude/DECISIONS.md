# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read through the ONE API against the staged IBL cache at `/app/data/one_cache`; no data file is opened directly. The list of what to load is not obtained from a ONE search but from the brain-wide-map release freeze that ships with the methods-paper repository, `/app/code/code_zhang2025/data/bwm_release.csv`, which is exactly the file `src/0_data_caching.py` uses. That CSV has one row per probe insertion and carries `eid`, `pid`, `probe_name`, `subject` and `lab`, so grouping it by `eid` yields 459 sessions together with their probe list and subject, and no further lookup is needed. If `/app/data/DATALIMIT_SUBSET.csv` exists the freeze is restricted to the `eid`s it names (it does not exist in this run, so all 459 sessions were attempted). Per session, a `SessionLoader` reads the trials table, the wheel, and the camera motion energy; a `SpikeSortingLoader` is instantiated once per `pid` to read spikes/clusters/channels. The ONE client is deliberately built **without** `password=`, because passing one forces a network re-authentication, while the cached token works offline. Sessions are converted in a `spawn` multiprocessing pool (16 workers in the full run), each worker building its own ONE client.

ii.
```python
ONE_CACHE_DIR = '/app/data/one_cache'
BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT_SUBSET = '/app/data/DATALIMIT_SUBSET.csv'  # present only for the reduced dataset

def get_one():
    """ONE client reading the staged cache (no network access to Alyx)."""
    from one.api import ONE
    return ONE(base_url='https://openalyx.internationalbrainlab.org',
               silent=True, cache_dir=ONE_CACHE_DIR)
```

```python
def session_list(args):
    """(eid, subject, lab, pids, probe_names) for every session to convert."""
    bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
    if os.path.exists(DATALIMIT_SUBSET):
        subset = pd.read_csv(DATALIMIT_SUBSET)
        col = 'eid' if 'eid' in subset.columns else subset.columns[0]
        bwm_df = bwm_df[bwm_df.eid.isin(subset[col].astype(str))]
    sessions = []
    for eid, rows in bwm_df.groupby('eid', sort=False):
        sessions.append((eid, rows.subject.iloc[0], rows.lab.iloc[0],
                         list(rows.pid), list(rows.probe_name)))
```

```python
sess_loader = SessionLoader(one=one, eid=eid)
...
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
```

iii. From the agent's summary: *"ONE reads the staged cache offline (no `password=` kwarg, which would force a network re-auth); probe IDs come from the BWM release freeze `bwm_release.csv`. All 459 released sessions were attempted."* The trajectory shows the agent first tried `ONE(mode='local')`, found `eid2pid` unavailable/failing offline, then tested an Alyx-backed client without a password against the staged `.rest` cache and confirmed it resolves `eid2pid`, trials, wheel and motion energy without network access (steps 44–65). Using `bwm_release.csv` is justified as being literally the session/probe list of the reference pipeline, which avoids having to re-derive the release inclusion criteria with a ONE search.

## 1-b. How are the data split into subjects (mice)?

i. The subject name is taken from the `subject` column of the release freeze, one value per session, so no path or filename parsing is involved. At assembly the unique subject names are sorted into `subjects` and `subject_idx` records, per session, the index into that list. The result is 136 subjects over 444 converted sessions.

ii.
```python
sessions.append((eid, rows.subject.iloc[0], rows.lab.iloc[0],
                 list(rows.pid), list(rows.probe_name)))
```

```python
subjects = sorted({r['subject'] for r in results})
subject_lookup = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_lookup[r['subject']] for r in results], dtype=np.int64),
```

iii. No explicit reasoning is recorded beyond the choice of `bwm_release.csv` as the session index; the subject is a column of that table, so nothing has to be derived. The agent additionally carries `lab` through into `session_info` as descriptive metadata.

## 1-c. How are the data split into sessions?

i. A session is the unit the release freeze is organised by. `bwm_df.groupby('eid')` collapses the per-insertion rows into one entry per session, carrying the list of `pid`s so that all probes of a session are processed together and merged into one population (as `merge_probes` does in the reference code). Results are sorted by `eid` before assembly so that session order is deterministic.

ii.
```python
for eid, rows in bwm_df.groupby('eid', sort=False):
    sessions.append((eid, rows.subject.iloc[0], rows.lab.iloc[0],
                     list(rows.pid), list(rows.probe_name)))
```

```python
results.sort(key=lambda r: r['eid'])
```

iii. From the agent's summary: *"probes within a session are merged as in `merge_probes`."* Nothing has to be decided about the split itself — the release index is already one row per insertion within a session, and the reference pipeline treats a session (not a probe) as the recording unit because two probes in a session share the same behaviour and are not statistically independent.

## 1-d. How are the data split into trials?

i. The trials table returned by `SessionLoader.load_trials()` has one row per trial, so the split is given by the data. Each trial becomes the interval `[stimOn_times - 0.5, stimOn_times + 1.5]` s, which is the `intervals` construction of `bin_spiking_data`/`get_behavior_per_interval` in the reference code.

ii.
```python
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
# NaN alignment times are excluded by the mask; replace them so that the
# searchsorted/interpolation below stay well defined.
align_times_filled = np.where(np.isnan(align_times), 0.0, align_times)
interval_begs = align_times_filled + TIME_WINDOW[0]
interval_ends = align_times_filled + TIME_WINDOW[1]
```

iii. No explicit reasoning recorded — the trials table is already one row per trial. The agent's summary records the window as *"Stimulus onset, window (−0.5, 1.5) s, 20 ms bins → 100 bins — exactly the `params` dict of `src/0_data_caching.py`."*

## 1-e. How are trials filtered based on quality controls?

i. Two masks are combined. The first is a re-implementation of the reference code's `load_trials_and_mask`, called with the arguments `prepare_data` uses: reaction time (`firstMovement_times - stimOn_times`) must be between 0.08 s and 2.0 s; trial length (`feedback_times - goCue_times`) must be ≤ 10 s; the six events `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType` must not be NaN; and no-response trials (`choice == 0`) are excluded. The second is behavioural coverage: a trial is dropped if the wheel trace or the whisker motion-energy trace does not span the whole 2 s window (starts more than one bin late, ends more than one bin early, or has no sample at all), or if the interpolated trace contains a non-finite value. A session with fewer than 2 surviving trials is dropped entirely. 444 of 459 sessions and 188,925 trials survived.

ii.
```python
MIN_RT, MAX_RT, MAX_TRIAL_LEN = 0.08, 2.0, 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

def load_trials_and_mask(sess_loader):
    ...
    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in NAN_EXCLUDE:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'   # exclude_nochoice
    mask = ~trials.eval(query)
    return trials, mask.to_numpy()
```

```python
keep = trials_mask & wheel_ok & whisker_ok
if keep.sum() < MIN_TRIALS_PER_SESSION:
    raise RuntimeError(f'only {keep.sum()} trials left after filtering')
```

```python
if np.abs(interval_begs[trial] - t[0]) > BINSIZE:      # data starts too late
    continue
if np.abs(interval_ends[trial] - t[-1]) > BINSIZE:     # data ends too early
    continue
...
if np.any(~np.isfinite(y)):
    continue
```

iii. From the docstring: *"Re-implementation of `load_trials_and_mask` of the reference code … with the arguments used by its `prepare_data`: min_rt=0.08, max_rt=2.0, max_trial_len=10.0, the default NaN exclusions and exclude_nochoice=True. These are also the trial exclusions described in the data paper."* For the extra behavioural mask the agent's summary states the deviation explicitly: *"the reference code runs with `allow_nans=True` and then (through a bug in `align_spike_behavior`) ignores the behaviour masks entirely; NaNs are not allowed in the target format, so trials whose wheel/whisker traces are missing, short, or NaN are dropped."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` of every probe insertion of the session, loaded through `SpikeSortingLoader.load_spike_sorting()`. The merged cluster table (`SpikeSortingLoader.merge_clusters(...).to_df()`) is used only for its `acronym` column, which supplies the anatomical label of each unit; its `label` (quality) column is deliberately *not* used for selection.

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
...
clusters_labeled = SpikeSortingLoader.merge_clusters(
    spikes, clusters, channels, compute_metrics=False).to_df()
...
spikes = {k: np.concatenate([s[k] for s in merged_spikes])
          for k in ('times', 'clusters')}
```

```python
acronyms = clusters['acronym'].to_numpy()[unit_ids]
beryl = BrainRegions().acronym2acronym(acronyms, mapping='Beryl')
```

iii. Mirrors `load_spiking_data` + `merge_probes` of `ibl_data_utils.py`, which the agent read in full (trajectory steps 12–14) before writing the script. Note that only `times` and `clusters` are retained after the merge, although the loader itself still reads `spikes.amps` and `spikes.depths` (the module default `SPIKES_ATTRIBUTES`), which are never used.

## 2-b. How is the `neural` data processed?

i. Probes are merged first: cluster ids of the second probe are offset by the number of clusters of the first, the cluster tables are concatenated with `ignore_index=True`, and the merged spike train is re-sorted by time — the exact procedure of `merge_probes`. Spikes are then counted into 100 non-overlapping 20 ms bins per trial, using `floor((t - t_beg)/binsize)` on the spikes falling in `[t_beg, t_end)`, implemented as a single `np.bincount` over a flattened `unit × bin` index. Only the units that fired at least one spike anywhere in the session are kept as rows, in increasing cluster order, which is what the reference's `clusters_used_in_bins` returns. No smoothing and no conversion to firing rate is applied: `neural` holds raw spike **counts**, stored as `float32` (binned internally as `uint16`).

ii.
```python
# merge_probes(): concatenate probes, re-indexing cluster ids, and sort by time
for clusters, spikes in zip(clusters_list, spikes_list):
    spikes = dict(spikes)
    spikes['clusters'] = spikes['clusters'] + cluster_max
    cluster_max = clusters.index.max() + 1
    ...
sort_idx = np.argsort(spikes['times'], kind='stable')
spikes = {k: v[sort_idx] for k, v in spikes.items()}
```

```python
unit_ids = np.unique(spike_clusters)
n_units = len(unit_ids)
lookup = np.zeros(int(unit_ids.max()) + 1, dtype=np.int64)
lookup[unit_ids] = np.arange(n_units)
...
for trial in range(n_trials):
    sl = slice(i_beg[trial], i_end[trial])
    bin_idx = np.floor((spike_times[sl] - interval_begs[trial]) / BINSIZE).astype(np.int64)
    keep = (bin_idx >= 0) & (bin_idx < N_BINS)   # guard against rounding at the edge
    rows = lookup[spike_clusters[sl][keep]]
    counts = np.bincount(rows * N_BINS + bin_idx[keep], minlength=n_units * N_BINS)
    binned[trial] = counts.reshape(n_units, N_BINS)
```

```python
neural.append(binned[i].astype(np.float32))
```

iii. The docstring states the binning is *"Equivalent to `bin_spiking_data` / `get_spike_data_per_interval` of the reference code (spikes in [t_beg, t_end), bin index floor((t - t_beg) / binsize)), but vectorized"*, and the agent's summary reports *"I verified my vectorized spike and behaviour binning are **bit-identical** to the reference implementations on a test session"* — this check is visible in the trajectory (step 74, comparing against `get_spike_data_per_interval` and `get_behavior_per_interval` imported from the reference repo). Spike counts rather than rates are stored because that is what the reference pipeline caches.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No single-unit quality control is applied at all.** Every spike-sorted unit of every probe is kept, subject only to having fired at least one spike in the session. Units whose Beryl acronym is `root` (86,127 units) or `void` (12,175 units, i.e. channels the histology placed outside the brain) are also kept and appear as brain regions in the output. The result is 599,865 units over 444 sessions (mean 1351, min 135, max 3140 per session) and a 107 GB pickle. Session- and insertion-level QC is inherited implicitly from the release freeze, which only contains insertions that already passed the data paper's RIGOR/histology criteria.

ii.
```python
"""Load and merge the spike sorting of every probe of a session.

Mirrors `load_spiking_data` (with qc=None, i.e. all units) followed by
`merge_probes` in the reference code.
"""
...
spikes, clusters, channels = loader.load_spike_sorting()
```

```python
unit_ids = np.unique(spike_clusters)   # only units that fired at least once
```

```python
'neurons': ('all units of the merged probes of a session that fired at '
            'least one spike'),
```

iii. The agent is explicit and flags the conflict itself: *"**Neurons: all spike-sorted units, no QC selection.** `prepare_data` → `load_spiking_data` in the methods-paper code runs with `qc=None`, and the paper states spike counts are binned 'using all neurons, sorted by Kilosort, from each session'. This is the one place the data paper differs (it restricts *its* analyses to RIGOR well-isolated units); I followed the decoding pipeline being replicated. Units with zero spikes in the session are dropped, as in `bin_spiking_data`."* Both claims check out against the sources: `load_spiking_data(one, pid, eid=..., pname=...)` in `prepare_data` leaves `qc=None`, and `methods.txt` line 13 contains the quoted sentence. The agent did not comment on `root`/`void` units, which `3_decode_multi_region.py` of the reference repo removes at the decoding stage.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `trials.stimOn_times`, as the instructions require and as `params['align_time']` in `0_data_caching.py` specifies. All IBL streams (spikes, trial events, wheel, camera) are already on one session clock, so alignment is just building the interval `[stimOn - 0.5, stimOn + 1.5]` per trial and taking the bin index relative to its start. Trials with a NaN `stimOn_times` have the alignment time replaced with 0.0 purely so that `searchsorted`/`interp1d` stay well defined; those trials are removed by the NaN mask and never reach the output.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)   # seconds relative to the alignment event
...
interval_begs = align_times_filled + TIME_WINDOW[0]
interval_ends = align_times_filled + TIME_WINDOW[1]
...
bin_idx = np.floor((spike_times[sl] - interval_begs[trial]) / BINSIZE).astype(np.int64)
```

```python
'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
'off_start': TIME_WINDOW[0],
'off_end': TIME_WINDOW[1],
```

iii. Agent summary: *"**Alignment.** Stimulus onset, window (−0.5, 1.5) s, 20 ms bins → 100 bins — exactly the `params` dict of `src/0_data_caching.py`."* It also noted the tension with the methods text: *"The methods text uses 50 ms for choice and 20 ms for the dynamic behaviours … 20 ms is what the caching code uses and is required for the time-varying outputs here"*, and that the methods paper aligns the dynamic behaviours to first-movement onset, which the instructions override in favour of stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, `ceil(2.0 / 0.02) = 100` bins per trial, identical for every trial and session (verified: `T_min = T_max = 100`). `metadata['time_bin_size']` is written in ms (20.0). Spikes are binned once directly at 20 ms; no rebinning, resampling or smoothing of the neural data takes place. The behavioural traces are resampled (not rebinned) onto the same 20 ms grid by linear interpolation.

ii.
```python
BINSIZE = 0.02              # seconds
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
...
'time_bin_size': BINSIZE * 1000.,
'n_timepoints': N_BINS,
```

iii. Taken from `params = {'interval_len': 2, 'binsize': 0.02, ..., 'time_window': (-.5, 1.5)}` in `0_data_caching.py`; the `np.ceil` matches the reference's own comment *"np.ceil because we want to make sure our bins contain all data"*. The agent justified choosing 20 ms over the 50 ms the methods text mentions for choice/prior on the grounds that 20 ms is what the caching code uses and is needed for the time-varying wheel/whisker outputs.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from the raw data at all: it is the analytic bin grid of the alignment window, defined by `trials.stimOn_times` (the alignment event) plus the constants `TIME_WINDOW` and `BINSIZE`. Because the window and bin size are identical for all trials, the same 100-element vector is used for every trial of every session.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)
```

iii. Implicit: the window and bin size are the reference `params`, and the decoder-input specification asks for "time since stimulus onset, continuous, time-varying", which given the fixed alignment is fully determined by the bin grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond forming the bin centres, −0.49 s to +1.49 s in 20 ms steps, and casting to `float32`. The observed range in the converted file is exactly [−0.49, 1.49].

ii.
```python
inp = np.empty((2, N_BINS), dtype=np.float32)
inp[0] = bin_centres
```

```python
'time_from_stimulus_onset': (
    'seconds from stimulus onset, centre of each 20 ms bin, from '
    f'{TIME_WINDOW[0] + BINSIZE / 2} to {TIME_WINDOW[1] - BINSIZE / 2} s'),
```

iii. Agent summary: *"**Inputs:** time from stimulus onset (bin centres, −0.49…1.49 s)."* No further justification recorded; bin centres are the natural representative time of a bin.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. By construction it *is* the neural time axis: column *k* of the neural matrix counts spikes in `[stimOn - 0.5 + 0.02k, stimOn - 0.5 + 0.02(k+1))`, and column *k* of the input holds that bin's centre. The two therefore describe the same instant, bin for bin, in every trial. (The behavioural outputs are sampled at the *right edge* of the same bins — a 10 ms offset from the input's centre convention — because that is what the reference `get_behavior_per_interval` does.)

ii.
```python
bin_idx = np.floor((spike_times[sl] - interval_begs[trial]) / BINSIZE).astype(np.int64)
```
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)
```

iii. No separate justification recorded — nothing has to be aligned, since both are generated from the same `TIME_WINDOW`/`BINSIZE` constants relative to the same `stimOn_times`.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. `trials.probabilityLeft`, which is held constant within a block, so every change of its value marks a block boundary. The trials table carries no explicit block index, so the blocks are recovered from the prior.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()
new_block = np.ones(len(pleft), dtype=bool)
new_block[1:] = pleft[1:] != pleft[:-1]
block_id = np.cumsum(new_block) - 1
```

iii. Implicit in the docstring (*"index of each trial within its block of constant probabilityLeft"*). The reference code exposes `block = trials_df['probabilityLeft']` as its "block" variable, so the identification of a block with a run of constant `probabilityLeft` follows the reference.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The 0-based position of the trial within its block: block starts are the indices where `probabilityLeft` changes, and the value is `trial index − index of the start of its block`. It is computed on the **full, unfiltered** trials table and only then subset by the trial mask, so a trial that is later excluded still advances the counter and the number reflects the animal's real position in the block. The value is constant across the 100 bins of a trial (broadcast to a time series, as the format requires) and stored as `float32`. Observed range 0–98.

ii.
```python
def trial_number_in_block(trials):
    """0-based index of each trial within its block of constant probabilityLeft.

    Computed on the full (unfiltered) trials table so that the count reflects the
    animal's actual position in the block, independent of the trial exclusions.
    """
    ...
    block_starts = np.flatnonzero(new_block)
    idx_in_block = np.arange(len(pleft)) - block_starts[block_id]
    return idx_in_block.astype(np.float64)
```

```python
in_block = trial_number_in_block(trials)[keep]
...
inp[1] = in_block[i]
```

iii. Agent summary: *"the 0-based trial index within the current `probabilityLeft` block, computed on the full trials table so exclusions don't shift the count."*

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is `+1` (left), `-1` (right) or `0` (no response). `0` never reaches this point because `exclude_nochoice` already removed those trials.

ii.
```python
# choice: IBL codes +1 for a left choice (counter-clockwise wheel turn that
# brings a left stimulus to the centre) and -1 for a right choice.
choice = np.where(trials['choice'].to_numpy()[keep] > 0, 0, 1).astype(np.int64)
```

iii. The inline comment gives the sign convention. The mapping left = 0, right = 1 is the one prescribed by the instructions.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding `+1 → 0` (left), `−1 → 1` (right), broadcast constant across the 100 bins of the trial and stored as `int64`; `output_values[0] = ['left', 'right']`. The resulting class balance is 50.8% / 49.2%.

ii.
```python
out = np.empty((4, N_BINS), dtype=np.int64)
out[0] = choice[i]
```
```python
'output_values': [
    ['left', 'right'],
    ...
```

iii. *"choice: side the mouse reported (IBL choice +1 -> left, -1 -> right)"* in the metadata. Made time-varying (constant over bins) because the format asks for time-varying outputs "if at all possible".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes the three values 0.2, 0.5 and 0.8.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()[keep]
prior = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                   np.isclose(pleft, 0.8)], [0, 1, 2], default=-1).astype(np.int64)
if np.any(prior < 0):
    raise RuntimeError(f'unexpected probabilityLeft values: {np.unique(pleft)}')
```

iii. *"prior_probability_left: trials.probabilityLeft of the block"*. The recoding is the one given in the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Recoding 0.2 → 0, 0.5 → 1, 0.8 → 2 with a float-tolerant comparison, plus a hard failure if any other value appears (which would otherwise silently produce a `-1` class). The value is broadcast constant over the 100 bins. Resulting distribution: 41.7% / 14.1% / 44.2%, i.e. the 0.5 unbiased block at the start of each session is retained (`exclude_unbiased=False`, as in the reference).

ii.
```python
out[1] = prior[i]
```
```python
'output_values': [..., ['0.2', '0.5', '0.8'], ...]
```

iii. No separate justification recorded beyond the instructions' mapping; the `np.select` + guard pattern is a defensive check the agent added.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position` / `_ibl_wheel.timestamps`, read through `SessionLoader.load_wheel()`, which returns an evenly sampled position together with a derived `velocity`. Wheel speed is `np.abs(velocity)`.

ii.
```python
if name == 'wheel-speed':
    if sess_loader.wheel.empty:
        sess_loader.load_wheel()
    return (sess_loader.wheel['times'].to_numpy(),
            np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. The docstring says *"Same sources as `load_target_behavior` in the reference code: wheel speed is the absolute value of the Gaussian-smoothed wheel velocity"* — which is `load_target_behavior(one, eid, 'wheel-speed')` verbatim (the "Gaussian-smoothed" wording is copied from the reference docstring; current `SessionLoader` in fact interpolates to a uniform rate and applies a low-pass Butterworth filter, but this is whatever the loader does, not a choice made here).

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) `SessionLoader.load_wheel()` interpolates the event-driven wheel position onto a uniform grid and differentiates/filters it into a velocity; the absolute value is taken. (2) For each trial the session-wide trace is cut to the samples strictly inside the window and linearly interpolated (`scipy.interpolate.interp1d`, `fill_value='extrapolate'`) onto `np.linspace(t_beg + binsize, t_end, 100)` — i.e. the right edge of each of the 100 bins. Trials whose trace is absent, starts more than one bin late, ends more than one bin early, or interpolates to a non-finite value are dropped. (3) The retained trace is discretized (see 7-c).

ii.
```python
x = np.linspace(interval_begs[trial] + BINSIZE, interval_ends[trial], N_BINS)
y = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
if np.any(~np.isfinite(y)):
    continue
binned[trial] = y
good[trial] = True
```

iii. Docstring: *"Follows `get_behavior_per_interval`: the trace is evaluated (linear interpolation) at the right edge of each of the `N_BINS` bins, a trial is discarded when the trace does not cover the trial window … Unlike the reference code, which is called with allow_nans=True, trials whose interpolated values contain NaNs are discarded as well: the decoder data format does not allow NaNs."* The agent verified this reproduces `get_behavior_per_interval` bit-identically on a test session (trajectory step 74).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 classes at the within-session tertiles: the 1/3 and 2/3 quantiles are computed over **all retained trials and all time bins of that session** pooled together, and `np.digitize` assigns classes 0/1/2 (`['low','medium','high']`). The two boundaries are passed through `np.maximum.accumulate` before use and are recorded per session in `metadata['session_info'][i]['wheel_speed_class_edges']`. The resulting global class balance is 33.33% / 33.33% / 33.33%.

ii.
```python
def discretize(values, n_classes=N_BEHAVIOR_CLASSES):
    quantiles = np.quantile(values, np.arange(1, n_classes) / n_classes)
    # ties (e.g. the many exactly-zero wheel-speed samples) all fall in the lowest
    # class, so make the boundaries strictly increasing to avoid empty classes
    edges = np.maximum.accumulate(quantiles)
    return np.digitize(values, edges, right=False).astype(np.int64), edges

wheel_class, wheel_edges = discretize(wheel[keep])
```

iii. Docstring: *"Absolute values are not comparable across sessions — whisker motion energy is in camera-specific units and depends on illumination and the position of the bounding box, and wheel-speed distributions differ between animals — so session-wise boundaries keep the class labels comparable across the sessions that the decoder is trained on jointly, and keep the classes balanced."* Echoed in the agent's summary.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is evaluated at the right edge of the same 100 bins, derived from the same `stimOn_times` and the same `TIME_WINDOW`/`BINSIZE`, so wheel bin *k* and neural bin *k* refer to the same 20 ms interval (the wheel value being the instantaneous speed at that interval's end, the neural value the count over the whole interval). The wheel is on the same session clock as the spikes, so no further alignment is needed. Trial identity is preserved because the same `keep` mask indexes the trials, spikes and both behavioural arrays.

ii.
```python
wheel, wheel_ok = bin_behavior(wheel_times, wheel_vals, interval_begs, interval_ends)
...
keep = trials_mask & wheel_ok & whisker_ok
...
binned, unit_ids = bin_spikes(spikes['times'], spikes['clusters'],
                              interval_begs[keep], interval_ends[keep])
...
wheel_class, wheel_edges = discretize(wheel[keep])
```

iii. Follows `get_behavior_per_interval`, which builds its intervals from the same `align_time` and `time_window` as `bin_spiking_data`; the agent verified numerical equivalence against the reference implementation.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with its frame times `_ibl_<side>Camera.times`, loaded through `SessionLoader.load_motion_energy(views=[view])` and read from the `whiskerMotionEnergy` column. The left camera is used when available, falling back to the right camera; if neither loads, the whole session is rejected (14 sessions were rejected this way).

ii.
```python
if name == 'whisker-motion-energy':
    for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            me = sess_loader.motion_energy[key]
            return (me['times'].to_numpy(),
                    me['whiskerMotionEnergy'].to_numpy())
        except Exception:
            continue
    raise RuntimeError('no whisker motion energy for either camera')
```

iii. This is the `bin_behaviors` fallback of the reference code (`load_target_behavior(..., 'left-whisker-motion-energy')`, then `'right-...'` if it reports `skip`), restated in the docstring as *"whisker motion energy is taken from the left camera, falling back to the right camera when it is missing."* The agent individually verified the 14 rejected sessions have no camera data at all (trajectory step 100).

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, normalisation or z-scoring. It goes through exactly the same `bin_behavior` path as the wheel: cut to the window, linearly interpolated onto the right edge of each of the 100 bins, with trials rejected for absent/short/NaN coverage, and then discretized.

ii.
```python
whisker_times, whisker_vals = load_behavior_trace(sess_loader, 'whisker-motion-energy')
whisker, whisker_ok = bin_behavior(whisker_times, whisker_vals,
                                   interval_begs, interval_ends)
```

iii. Same justification as the wheel: it reproduces `get_behavior_per_interval` of the reference pipeline, which applies no additional processing to the released motion-energy trace.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: 3 classes at the within-session tertiles of all retained trials × bins, labelled `['low','medium','high']`, boundaries stored in `metadata['session_info'][i]['whisker_motion_energy_class_edges']`. Global class balance 33.26% / 33.26% / 33.48%.

ii.
```python
whisker_class, whisker_edges = discretize(whisker[keep])
```

iii. From the `discretize` docstring: *"whisker motion energy is in camera-specific units and depends on illumination and the position of the bounding box … so session-wise boundaries keep the class labels comparable across the sessions that the decoder is trained on jointly."* This is the stronger of the two arguments for per-session tertiles, since raw motion energy is not comparable between a left and a right camera at all.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as the wheel: right edge of the same 100 bins around the same `stimOn_times`, on the same session clock (IBL synchronises camera frame times to the ephys clock upstream), and the same `keep` mask applied to trials, spikes and both behaviours. Note the camera runs at 60 Hz (left) or 150 Hz (right), i.e. below/near the 50 Hz bin rate for the left camera, so the 20 ms samples are interpolated between frames rather than being independent measurements.

ii.
```python
interval_begs = align_times_filled + TIME_WINDOW[0]
interval_ends = align_times_filled + TIME_WINDOW[1]
...
whisker, whisker_ok = bin_behavior(whisker_times, whisker_vals,
                                   interval_begs, interval_ends)
...
out[3] = whisker_class[i]
```

iii. As for the wheel — the interval construction is shared with the spike binning, and the agent verified bit-identity with the reference `get_behavior_per_interval`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several layers, all of which drop rather than impute:
- **Trials with NaN events** are removed by the `NAN_EXCLUDE` list; a NaN `stimOn_times` is temporarily replaced by 0.0 only so the vectorized `searchsorted`/`interp1d` stay well defined, and the trial is then masked out.
- **Trials with missing/partial/NaN behavioural traces** are dropped by `bin_behavior` (`good=False`).
- **Sessions with no whisker motion energy on either camera** raise and are skipped (14 sessions).
- **Sessions with fewer than 2 usable trials** raise and are skipped (1 session).
- **Any other per-session exception** is caught by `_safe_process_session` so a single bad session cannot abort the 459-session run; the eid and error are recorded in `metadata['failed_sessions']`.
- **Rounding at the window edge** in spike binning is guarded by dropping out-of-range bin indices.
- A probe with no spike sorting raises (`no spike sorting for probe`), which kills the session rather than silently converting only the other probe.
- Provenance is kept per session: `n_trials_released`, `n_trials_passing_trial_qc`, `n_trials`, `n_probes`, `n_neurons` and the discretization edges.

Not handled: three trials in one session contain all-zero neural data (a recording gap); these survive into the output and are reported by the format verifier as warnings.

ii.
```python
align_times_filled = np.where(np.isnan(align_times), 0.0, align_times)
```
```python
keep = (bin_idx >= 0) & (bin_idx < N_BINS)   # guard against rounding at the edge
```
```python
if len(spikes) == 0 or len(clusters) == 0:
    raise RuntimeError(f'no spike sorting for probe {pname}')
```
```python
def _safe_process_session(args):
    try:
        return process_session(args)
    except Exception as exc:  # noqa: BLE001 - a bad session must not stop the run
        return {'eid': args[0], 'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc()}
```
```python
'failed_sessions': failures,
```

iii. Agent summary: *"15 sessions were excluded this way — 14 have no camera data at all, 1 has video covering only one trial (each verified individually)"* and *"Three trials in one session carry all-zero neural data (a recording gap); they surface as format warnings, not errors."* The trajectory shows both diagnoses being run explicitly (steps 100 and 115) rather than assumed.

## 10-a. What are the most time-consuming steps of the code?

i. Reading and merging the spike sorting dominates: `load_spike_sorting()` per probe pulls the multi-hundred-MB `spikes.times`/`spikes.clusters` arrays (and, unnecessarily, `spikes.amps`/`spikes.depths`), verifies file hashes, and `merge_clusters` builds the full cluster table. Session wall times in `convert_log.txt` are ~8–30 s, essentially all I/O, against ~0.1–1 s for the actual binning. Second is the final serialisation: pickling a 107 GB dictionary in one `pickle.dump`, which also requires the whole dataset to be held in the parent process. The agent mitigated the first cost by running 16 sessions in parallel (`spawn` pool, `maxtasksperchild=4`), which brought the full 459-session conversion to roughly 10 minutes.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
```
```python
with ctx.Pool(processes=args.n_workers, maxtasksperchild=4) as pool:
    for i, res in enumerate(pool.imap_unordered(_safe_process_session, sessions)):
```
```python
with open(args.out, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Implicit in the design (worker pool, `maxtasksperchild=4` to bound memory growth). The agent checked machine resources first (trajectory step 32: RAM, cores, disk, GPU) and monitored RSS during the run, indicating the parallelism level was chosen against memory rather than CPU.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three per-trial loops remain, but the two expensive ones were already partly vectorized relative to the reference (which spawns a multiprocessing pool per trial):
- `bin_spikes`: the `for trial in range(n_trials)` loop. The `searchsorted` bounds are computed for all trials at once, but the `bincount` is per trial. It could be a single `bincount` over `trial * n_units * N_BINS + unit * N_BINS + bin` for all in-window spikes at once.
- `bin_behavior`: the per-trial `interp1d` loop. Since the wheel trace is uniformly sampled, all trials could be interpolated with one `np.interp` over a concatenated query vector.
- `process_session`: the final `for i in range(n_trials)` loop that materialises the per-trial `neural`/`input`/`output` arrays. This is mostly unavoidable given the target format (a list of arrays per trial), though `inputs` could be built once and shared since `bin_centres` is constant.

The gain would be small — these loops are milliseconds against tens of seconds of I/O.

ii.
```python
for trial in range(n_trials):
    sl = slice(i_beg[trial], i_end[trial])
    ...
    counts = np.bincount(rows * N_BINS + bin_idx[keep], minlength=n_units * N_BINS)
    binned[trial] = counts.reshape(n_units, N_BINS)
```
```python
for trial in range(n_trials):
    t = target_times[idxs_beg[trial]:idxs_end[trial]]
    ...
    y = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```
```python
for i in range(n_trials):
    neural.append(binned[i].astype(np.float32))
```

iii. The agent's stated goal was equivalence, not speed: *"I verified my vectorized spike and behaviour binning are bit-identical to the reference implementations on a test session."* Replacing the reference's per-trial multiprocessing pools with in-process loops was itself the main speed decision.

## 10-c. What processing does the code repeat multiple times?

i. Little at the level of the data itself, but several fixed costs are paid per session instead of once:
- `get_one()` constructs a new ONE client for every session, and `maxtasksperchild=4` tears the worker down every 4 sessions, so the client (and the ONE cache tables) are rebuilt repeatedly.
- `BrainRegions()` is instantiated inside `process_session`, i.e. once per session, rather than once per worker.
- `import` statements for `SessionLoader`, `SpikeSortingLoader` and `BrainRegions` sit inside the functions and so execute per session.
- `np.searchsorted` over the window bounds is recomputed three times per session (once in `bin_spikes`, once per behavioural trace in `bin_behavior`) on three different time bases — this one is genuinely necessary.
- `trial_number_in_block` and the behavioural interpolation are computed for **all** trials and then subset by `keep`.

ii.
```python
def get_one():
    from one.api import ONE
    return ONE(base_url='https://openalyx.internationalbrainlab.org',
               silent=True, cache_dir=ONE_CACHE_DIR)
```
```python
from iblatlas.regions import BrainRegions
acronyms = clusters['acronym'].to_numpy()[unit_ids]
beryl = BrainRegions().acronym2acronym(acronyms, mapping='Beryl')
```
```python
with ctx.Pool(processes=args.n_workers, maxtasksperchild=4) as pool:
```

iii. Not discussed by the agent. The per-process construction of the ONE client is forced by multiprocessing (a client cannot be shared across a `spawn`); `maxtasksperchild=4` was evidently chosen to bound worker memory growth, which the agent monitored with `free -g` during the run.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, one of them large:
- **`spikes.amps` and `spikes.depths` are loaded and never used.** The script does not narrow `brainbox.io.one.SPIKES_ATTRIBUTES` (default `['clusters', 'times', 'amps', 'depths']`), so roughly twice the necessary bytes are read and concatenated per probe before being thrown away by the `for k in ('times', 'clusters')` comprehension.
- **Behavioural interpolation is run for every trial in the session**, including trials already excluded by `trials_mask`; only `keep` survives. For a typical session that is ~25–50% wasted `interp1d` calls.
- **`merge_clusters` builds the full cluster table** (metrics, uuids, depths, channels) when only `acronym` is used.
- **No unit curation**, so ~600k units are binned, stored and later fed to the decoder, including 86,127 `root` and 12,175 `void` units — the latter are channels the histology placed outside the brain and are explicitly discarded by the reference repo's own multi-region decoding script. This is the main driver of the 107 GB file size.
- **Per-trial scalars are tiled across 100 bins** (`choice`, `prior`, `trial_number_in_block`), and the outputs are stored as `int64` where `int8` would do — an 8× overhead on the output arrays.
- `bin_centres` is identical for every trial of every session but is materialised into a separate `(2, 100) float32` array per trial (188,925 copies).
- Minor: `lab`, `n_trials_released`, `n_trials_passing_trial_qc` and the per-session class edges are computed and stored but unused by the decoder (they are, however, useful provenance).

ii.
```python
spikes = {k: np.concatenate([s[k] for s in merged_spikes])
          for k in ('times', 'clusters')}      # amps/depths were read, then dropped
```
```python
wheel, wheel_ok = bin_behavior(wheel_times, wheel_vals, interval_begs, interval_ends)
...
keep = trials_mask & wheel_ok & whisker_ok    # trials_mask applied only afterwards
```
```python
out = np.empty((4, N_BINS), dtype=np.int64)
out[0] = choice[i]
out[1] = prior[i]
```
```python
inp = np.empty((2, N_BINS), dtype=np.float32)
inp[0] = bin_centres
inp[1] = in_block[i]
```

iii. Not discussed by the agent; the design priority visible throughout the trajectory was fidelity to the reference implementation (verified bit-identical) and getting the full 459-session run to complete, rather than minimising redundant work. The agent did check disk headroom before launching the full conversion and confirmed the 107 GB file both wrote and trained successfully.
