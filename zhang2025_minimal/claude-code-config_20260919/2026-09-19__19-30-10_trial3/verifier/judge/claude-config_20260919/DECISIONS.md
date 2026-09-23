# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Nothing is read from disk directly. The AI takes the session/insertion list from the **freeze file that ships with the reference repository**, `/app/code/code_zhang2025/data/bwm_release.csv` (699 `pid`s, 459 `eid`s, 139 subjects, with `subject` and `lab` columns), which is exactly the file `0_data_caching.py` uses. Every actual data read then goes through the ONE API against the local cache at `/app/data/one_cache`: a single `ONE(base_url=..., silent=True, cache_dir=ONE_CACHE_DIR)` client per process (no password, because the container ships an auth token and Alyx is unreachable). Per session it uses `SessionLoader` for the trials table, the wheel and the camera motion energy, and one `SpikeSortingLoader` per released probe insertion (the `pid`/`probe_name` rows grouped out of the release CSV). Sessions are farmed out to a `ProcessPoolExecutor` (default 16 workers; the full run used 24), one session per task, and each worker builds its own ONE client and `BrainRegions()`. A `DATALIMIT_SUBSET.csv` hook is present to restrict the eid list when the reduced dataset variant is mounted; it did not exist in this run, so all 459 eids were processed.

ii.
```python
ONE_CACHE_DIR = '/app/data/one_cache'
BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT_SUBSET_CSV = '/app/DATALIMIT_SUBSET.csv'

def get_one():
    global _ONE
    if _ONE is None:
        from one.api import ONE
        _ONE = ONE(base_url=ONE_BASE_URL, silent=True, cache_dir=ONE_CACHE_DIR)
    return _ONE
```

```python
bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eids = list(dict.fromkeys(bwm_df['eid'].tolist()))
if os.path.exists(DATALIMIT_SUBSET_CSV):
    subset = pd.read_csv(DATALIMIT_SUBSET_CSV)
    col = 'eid' if 'eid' in subset.columns else subset.columns[0]
    allowed = set(subset[col].astype(str))
    eids = [e for e in eids if e in allowed]
...
by_eid = bwm_df.groupby('eid')
for eid in eids:
    rows = by_eid.get_group(eid)
    tasks.append((eid, rows[['pid', 'probe_name']].to_dict('records'),
                  str(rows['subject'].iloc[0]), str(rows['lab'].iloc[0])))
```

```python
sess_loader = SessionLoader(one=one, eid=eid); sess_loader.load_trials()
ssl = SpikeSortingLoader(pid=str(row['pid']), one=one, eid=eid, pname=row['probe_name'])
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. From the trajectory: the AI read `0_data_caching.py`, `ibl_data_utils.py` and the repo README first, then inspected `bwm_release.csv` and confirmed it holds "eids 459 pids 699 subs 139" — i.e. the exact release freeze the reference pipeline iterates over, so using it reproduces the reference's session universe without re-deriving it. It then spent several steps probing how ONE behaves offline (steps 42–53): passing `password='international'` forced a network round-trip and failed, so it dropped the password and relied on the cached token; it also checked that revision folders (`#2024-05-06#`) resolve through the API rather than by globbing paths. The docstring states the choice explicitly: "the 459 eids of the BWM public release (`data/bwm_release.csv`), i.e. the sessions that already passed the release criteria (>=250 trials, >=90% correct on 100% contrast, hardware QC, resolved histology alignment)".

## 1-b. How are the data split into subjects?

i. The subject name is read straight off the release CSV row for each `eid` (`rows['subject'].iloc[0]`) and carried through the worker into the per-session result. At assembly `subjects` is the sorted set of unique names and `subject_idx` is each session's index into it. No path or filename parsing. Result: 136 subjects over 444 sessions.

ii.
```python
tasks.append((eid, rows[['pid', 'probe_name']].to_dict('records'),
              str(rows['subject'].iloc[0]), str(rows['lab'].iloc[0])))
```

```python
subjects = sorted({r['subject'] for r in results})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_index[r['subject']] for r in results], dtype=np.int64),
```

iii. No explicit narration; implicit in the choice of `bwm_release.csv` as the driver — the release table already carries a canonical `subject` (and `lab`) per `eid`, so subject identity is looked up rather than derived. The AI also kept `lab` in `session_info` for provenance.

## 1-c. How are the data split into sessions?

i. A session is the unit of the release table, so no splitting is done: the unique `eid`s of `bwm_release.csv` are the sessions, `bwm_df.groupby('eid')` collects that session's probe insertions, and each `eid` becomes one independent worker task and one element of `neural`/`input`/`output`. Session order in the final dictionary is made deterministic by sorting the results on `eid` rather than leaving them in completion order.

ii.
```python
eids = list(dict.fromkeys(bwm_df['eid'].tolist()))
by_eid = bwm_df.groupby('eid')
...
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(_worker, task): task[0] for task in tasks}
...
# Deterministic session order, independent of completion order.
results.sort(key=lambda r: r['eid'])
```

iii. Not separately argued; it follows from the release table being indexed by `eid`. The one deliberate note in the code is the determinism comment — the AI wanted session order independent of `as_completed` ordering so runs are reproducible.

## 1-d. How are the data split into trials?

i. Not derived at all: the IBL trials table has one row per trial, so `SessionLoader.load_trials()` gives the split. A trial's window is defined as `stimOn_times + (-0.5, 1.5) s`, i.e. `interval_begs = stimOn_times - 0.5` and a fixed 100 × 20 ms grid after it, which is the reference repo's `bin_spiking_data` interval construction.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

```python
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
interval_begs_all = align_times + TIME_WINDOW[0]
```

iii. The module docstring pins this to the reference: "These are exactly the reference caching parameters (`src/0_data_caching.py`: interval_len 2, binsize 0.02, align_time 'stimOn_times', time_window (-.5, 1.5))". The AI read `0_data_caching.py` (step 10) where `params = {'interval_len': 2, 'binsize': 0.02, 'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}`, and the methods excerpt: "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps."

## 1-e. How are trials filtered based on quality controls?

i. Three stacked filters.

1. **The reference trial mask.** The AI re-implements `load_trials_and_mask(one, eid, max_trial_len=10.)` verbatim (same defaults as `prepare_data` calls it with): reaction time `firstMovement_times - stimOn_times` must be in [0.08 s, 2.0 s]; trial length `feedback_times - goCue_times` must be ≤ 10 s; no NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; and `choice != 0` (a response was made). `exclude_unbiased` is left off, so the 0.5 block is kept.
2. **Valid label values.** `stimOn`-derived window start must be finite and `probabilityLeft` must be one of 0.2/0.5/0.8 (`prior_all >= 0`).
3. **Behavioural coverage.** Each trial's 2 s window must be spanned by both the wheel trace and the whisker trace, to within one bin at either edge, and the interpolated values must be finite — the AI's port of the reference's 'target data not present' / 'starts too late' / 'ends too early' checks.

Sessions left with fewer than 2 usable trials are dropped. Net result: 188,925 trials over 444 sessions.

ii.
```python
MIN_RT, MAX_RT, MAX_TRIAL_LEN = 0.08, 2.0, 10.0

def load_trials_and_mask(one, eid):
    nan_exclude = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
                   'firstMovement_times', 'feedbackType']
    ...
    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'          # no response
    mask = ~trials.eval(query)
    return trials, mask.to_numpy().astype(bool)
```

```python
keep = trials_mask & np.isfinite(interval_begs_all) & (prior_all >= 0)
...
for name, (t, v) in traces.items():
    binned, good = bin_behaviour(t, v, safe_begs)
    beh_binned[name] = binned
    keep &= good
if keep.sum() < MIN_TRIALS_PER_SESSION:
    return None, 'too few trials with usable behaviour'
```

```python
    i0 = np.searchsorted(times, interval_begs, side='right')
    i1 = np.searchsorted(times, interval_ends, side='left')
    good = i1 > i0
    idx = np.flatnonzero(good)
    good[idx] &= np.abs(interval_begs[idx] - times[i0[idx]]) <= BINSIZE
    idx = np.flatnonzero(good)
    good[idx] &= np.abs(interval_ends[idx] - times[i1[idx] - 1]) <= BINSIZE
    good &= np.all(np.isfinite(binned), axis=1)
```

iii. The AI's stated reason for re-implementing rather than importing is compatibility, not a change of criteria: "Reimplemented here only because the repository's copy calls `SessionLoader(one, eid)` positionally, which the installed ibllib (4.0.1) no longer accepts. The query and its defaults are unchanged." It picked `max_trial_len=10.` because that is how `prepare_data` in `ibl_data_utils.py` invokes it. The coverage mask is explicitly labelled as the reference's own checks: "the reference's 'target data not present' / 'starts too late' / 'ends too early' checks". In step 76 the AI ran a dedicated "Diagnose trial attrition" probe on two sessions to confirm the attrition was coming from the intended criteria and not a bug. `MIN_TRIALS_PER_SESSION = 2` is justified in a comment: "the decoder is trained on 80% of each session's trials and validated on the rest, so a session with a single trial cannot contribute to both splits."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from `SpikeSortingLoader.load_spike_sorting()`, one call per released probe insertion of the session. The merged cluster table (`SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()`) is used only for the QC `label` column and the `acronym` (histology) column — it contributes the neuron mask and the region labels, not the activity values.

ii.
```python
ssl = SpikeSortingLoader(pid=str(row['pid']), one=one, eid=eid, pname=row['probe_name'])
spikes, clusters, channels = ssl.load_spike_sorting()
if len(spikes) == 0 or len(clusters) == 0:
    continue
clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
sc = np.asarray(spikes['clusters'], dtype=np.int64)
st = np.asarray(spikes['times'], dtype=np.float64)
```

iii. Mirrors the reference's `load_spiking_data` → `prepare_data`, where `neural_dict` is built from `spikes['times']`, `spikes['clusters']` and `clusters['acronym']`. The AI did not re-implement `load_spiking_data` because its `raw_electrophysiology(band="ap", stream=True)` call (used only to report sampling frequency) needs network access; it uses the loader directly instead.

## 2-b. How is the `neural` data processed?

i. Spikes of the surviving units are counted into 100 non-overlapping 20 ms bins per trial. **The values stay raw spike counts per 20 ms bin — no division by the bin width, no smoothing, no normalisation, no z-scoring.** Units from all probe insertions of a session are pooled into one population: each probe's kept clusters are renumbered into a contiguous 0..n-1 block continuing from the previous probe's offset via a lookup table, and the concatenated spike times are stable-sorted, which is the reference's `merge_probes`. Arrays are stored as `float32`, one `(n_units, 100)` array per trial. Mean 141.4 neurons per session (min 1, max 516); 62,763 neurons total.

ii.
```python
lut = np.full(int(clusters.shape[0]), -1, dtype=np.int64)
lut[keep_ids] = np.arange(keep_ids.size) + offset
...
mapped = lut[sc]
ok = mapped >= 0
ok &= np.isfinite(st)
times_list.append(st[ok]); units_list.append(mapped[ok]); acronym_list.append(beryl[keep_ids])
offset += keep_ids.size
...
order = np.argsort(spike_times, kind='stable')
return spike_times[order], spike_units[order], acronyms
```

```python
def bin_spikes(spike_times, spike_units, n_units, interval_begs):
    out = np.zeros((n_trials, n_units, N_BINS), dtype=np.float32)
    interval_ends = interval_begs + N_BINS * BINSIZE
    i0 = np.searchsorted(spike_times, interval_begs, side='left')
    i1 = np.searchsorted(spike_times, interval_ends, side='left')
    for k in range(n_trials):
        t = spike_times[i0[k]:i1[k]]
        if t.size == 0: continue
        u = spike_units[i0[k]:i1[k]]
        b = np.floor((t - interval_begs[k]) / BINSIZE).astype(np.int64)
        np.clip(b, 0, N_BINS - 1, out=b)
        counts = np.bincount(u * N_BINS + b, minlength=flat)
        out[k] = counts.reshape(n_units, N_BINS)
```

iii. The docstring of `bin_spikes` ties the scheme to the reference: "Bin i of a trial covers [beg + i*BINSIZE, beg + (i+1)*BINSIZE), which is the binning the reference performs with `bincount2D(..., xbin=binsize, xlim=[t_beg, t_end])` followed by keeping the first `N_BINS` columns." The AI verified this by printing the `bincount2D` source (step 63) before writing the code. Keeping raw counts follows the method paper ("we constructed X by aggregating spike counts") and is recorded in metadata as `'neural_units': 'spike count per 20 ms bin'`. Probe merging follows `merge_probes`, whose docstring reason (same underlying behaviour, so probes in a session are not independent) the AI had read.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two cuts, applied per probe before spikes are kept:
- `clusters['label'] >= 1` — the IBL "well-isolated neuron" label, i.e. all three RIGOR single-unit metrics passed (amplitude > 50 µV, noise cut-off < 20 µV, refractory-period violation).
- Beryl acronym not in `('root', 'void')` — i.e. restricted to grey matter / atlas-assigned structures.

Spikes of dropped clusters are discarded; spikes with out-of-range cluster ids or non-finite times are also dropped. A probe contributing zero surviving clusters is skipped, and a session with zero surviving units returns `None` and is dropped. This yields 62,763 neurons across 263 Beryl regions.

ii.
```python
NON_GREY_MATTER = ('root', 'void')
...
beryl = np.asarray(brain_regions.acronym2acronym(
    clusters['acronym'].to_numpy(), mapping='Beryl'))

# Data-paper inclusion criteria for neurons: well isolated (all three RIGOR
# single-unit metrics passed, which ibllib encodes as label == 1) and located
# in grey matter.
keep = (clusters['label'].to_numpy() >= 1) & ~np.isin(beryl, NON_GREY_MATTER)
keep_ids = np.flatnonzero(keep)
if keep_ids.size == 0:
    continue
```

```python
if spike_times is None:
    return None, 'no well-isolated grey-matter units'
```

iii. Justified from the data paper rather than from the reference repo (whose `prepare_data` calls `load_spiking_data` with `qc=None` and therefore keeps every sorted cluster). The AI quotes the criteria in its metadata: "Well-isolated units only (ibllib label == 1: amplitude > 50 uV, noise cut-off < 20 uV, refractory-period violation passed), located in grey matter (Beryl acronym not root/void). Units from all probe insertions of a session are merged into one population." The methods excerpt it read says exactly this — "stringent quality-control metrics … identified 75,708 well-isolated neurons" and "Final analyses were additionally restricted to regions that were designated grey matter". The `root`/`void` exclusion also matches the reference repo's own region filter in `data_loader_utils.py`: `unique_regions = [roi for roi in np.unique(dm.train.neuron_regions) if roi not in ['root', 'void']]`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams are already on one synchronised session clock, so alignment is a subtraction. Each trial's window starts at `interval_begs = stimOn_times - 0.5`; the spike times inside `[interval_begs, interval_begs + 2.0)` are located with `searchsorted` and the bin index is `floor((t - interval_begs) / 0.02)`, clipped to [0, 99]. Zero on the resulting axis is stimulus onset, sitting at the boundary between bins 24 and 25. `temporal_alignment_event` is recorded as `'stimulus onset (trials.stimOn_times)'` with `off_start = -0.5`, `off_end = 1.5`.

ii.
```python
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
interval_begs_all = align_times + TIME_WINDOW[0]
...
b = np.floor((t - interval_begs[k]) / BINSIZE).astype(np.int64)
np.clip(b, 0, N_BINS - 1, out=b)
```

```python
'temporal_alignment_event': 'stimulus onset (trials.stimOn_times)',
'off_start': TIME_WINDOW[0],
'off_end': TIME_WINDOW[1],
'alignment_note': (
    'Neural bin i spans [stimOn - 0.5 + 0.02*i, stimOn - 0.5 + 0.02*(i+1)). '
    'time_from_stimulus_onset is the centre of that bin; wheel speed and '
    'whisker motion energy are linearly interpolated onto its right edge, '
    'as in the reference pipeline.'),
```

iii. `align_time='stimOn_times'` is the reference caching parameter and the instructions' stated alignment event. The AI spent steps 81–86 chasing an all-zero neural trial and concluded "Spike gap is real data, not a bug" — i.e. it verified the alignment arithmetic against the raw spike times before committing, and recorded the bin/edge convention explicitly in `alignment_note` so downstream users know where t=0 sits.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms, 100 bins spanning the 2 s window, identical for every trial and session; `time_bin_size` is written as 20.0 (ms). Spikes are binned once, directly at 20 ms, from raw spike times — there is no resampling, no rebinning from a finer grid, and no smoothing. The behavioural traces are *interpolated* onto the same 100-bin grid (see 7-b/8-b), but the neural stream is not.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
...
'time_bin_size': BINSIZE * 1000.0,
'n_timepoints': N_BINS,
```

iii. From the docstring: "20 ms non-overlapping bins -> 100 timepoints per trial. These are exactly the reference caching parameters". Backed by the method paper text the AI read in `methods.txt`: "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps." The `np.ceil` mirrors the reference's `n_bins = int(np.ceil(interval_len / binsize))`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from raw data at all — it is the fixed bin grid defined by `TIME_WINDOW` and `BINSIZE`, anchored on `trials.stimOn_times`. The value in bin *i* is the centre of that bin, `-0.5 + (i + 0.5) * 0.02`, so the vector runs from -0.49 to 1.49 and is byte-identical for every trial in every session.

ii.
```python
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
...
inputs[:, 0, :] = bin_centres[None, :]
```

iii. The only raw quantity involved is `stimOn_times`, which defines where the grid is placed; the values themselves are definitional. The AI unit-tested the vector in step 97, printing `[-0.49 -0.47 -0.45] ... [1.45 1.47 1.49]` to confirm the endpoints.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the bin-centre vector and broadcasting it across trials into `inputs[:, 0, :]` as `float32`. The instructions asked for a continuous, time-varying input, so it is left as a real-valued ramp rather than a binary onset indicator.

ii.
```python
n_trials = int(keep.sum())
inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
inputs[:, 0, :] = bin_centres[None, :]
inputs[:, 1, :] = n_in_block[:, None]
```

iii. A comment states the choice: "Time since stimulus onset at the centre of each neural bin." The centre rather than an edge is chosen so the input labels the instant the bin represents; the AI notes the distinction from the behaviour sampling (right edge) explicitly in `alignment_note`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. By construction: bin *i* of the neural array covers `[stimOn - 0.5 + 0.02i, stimOn - 0.5 + 0.02(i+1))`, and `inputs[:, 0, i]` is the centre of that same bin. The two share the same index axis and the same origin, so column *i* of `neural` and column *i* of `input` describe the same 20 ms of the same trial. Both arrays are length 100 for all trials, which the verifier confirmed (`T: min 100, max 100`).

ii.
```python
b = np.floor((t - interval_begs[k]) / BINSIZE).astype(np.int64)   # neural grid
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE  # input = centres of that grid
```

iii. Recorded in `alignment_note`: "Neural bin i spans [stimOn - 0.5 + 0.02*i, stimOn - 0.5 + 0.02*(i+1)). time_from_stimulus_onset is the centre of that bin". No further alignment is needed because both derive from the same `interval_begs`.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials['probabilityLeft']` alone. The IBL trials table carries no block identifier, so blocks are recovered as maximal runs of constant `probabilityLeft`; a change of value starts a new block.

ii.
```python
def trial_number_in_block(trials):
    p_left = trials['probabilityLeft'].to_numpy(dtype=float)
    # NaN != NaN, so a NaN run is broken up; that is fine, those trials are excluded.
    new_block = np.ones(len(p_left), dtype=bool)
    new_block[1:] = ~(p_left[1:] == p_left[:-1])
    block_id = np.cumsum(new_block) - 1
```

iii. The AI knew from the papers (and states in `task_description`) that "After 90 unbiased trials (p(left) = 0.5) the stimulus side is drawn in blocks of 20-100 trials with p(left) = 0.2 or 0.8; block switches are uncued" — so `probabilityLeft` is piecewise constant and fully determines the block boundaries. The NaN case is handled deliberately (a NaN run splits blocks, but those trials are excluded anyway).

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based counter of the trial's position within its block, computed on the **full** trials table before any trial is excluded, then subsampled with `keep` — so a trial dropped by QC still advances the counter and the value reflects the animal's real position in the block. Stored as `float32`, constant across the 100 bins of a trial.

**Caveat found in the trajectory:** the first implementation of the offset lookup was wrong — it prepended a 0 and indexed `block_id` into the shifted array, so every block after the first was offset by the *previous* block's start. The AI unit-tested it at step 97 on `[0.5]*4 + [0.2]*3 + [0.8]*2 + [0.2]*1` and got `[0 1 2 3 4 5 6 3 4 2]` instead of `[0 1 2 3 0 1 2 0 1 0]`, said "Found a bug in the block-index helper", and fixed it at step 101. **The fix landed after the full conversion had already been written** (`converted_data.pkl` 23:51, `convert_data.py` 23:57) and the pickle was never regenerated, so the shipped data carries the buggy values. The verifier output on the shipped pickle confirms it: `trial_number_in_block: [0.0, 178.0]`, whereas the largest IBL block is ~100 trials (the expert file tops out at 98).

ii. Shipped code (fixed):
```python
    block_id = np.cumsum(new_block) - 1
    block_start = np.flatnonzero(new_block)
    idx_in_block = np.arange(len(p_left)) - block_start[block_id]
    return idx_in_block.astype(np.float32)
```

Buggy version used for `converted_data.pkl`:
```python
    block_id = np.cumsum(new_block) - 1
    idx_in_block = np.arange(len(p_left)) - np.concatenate(
        [[0], np.flatnonzero(new_block)])[block_id]
    return idx_in_block.astype(np.float32)
```

Where it is applied:
```python
n_in_block = trial_number_in_block(trials)[keep]
...
inputs[:, 1, :] = n_in_block[:, None]
```

iii. Docstring: "0-based index of each trial within its block of constant probabilityLeft. Computed on the *full* trials table, before any trial is excluded, so that the value reflects the animal's actual position in the block rather than a position in the filtered sequence." The AI caught the arithmetic error itself via a hand-built unit test but ran out of runway to re-run the 5.5-minute conversion (its attempts to free the GPU by killing the training job at steps 100/102 failed, and the session ended).

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The single column `trials['choice']`, which is IBL-coded +1 / -1 / 0.

ii.
```python
# IBL codes choice as +1 when the mouse reported the stimulus on the LEFT
# (verified: trials with contrastLeft > 0 and feedbackType == +1 all have
# choice == +1) and -1 for a right report. The task asks for left = 0, right = 1.
choice = (trials['choice'].to_numpy()[keep] == -1).astype(np.int64)
```

iii. The AI did not take the sign convention on faith. At steps 56–57 it loaded a trials table and cross-checked `choice` against `contrastLeft`/`feedbackType`, and recorded the result in the comment quoted above: trials with a left stimulus and positive feedback all have `choice == +1`, so +1 is a leftward report.

## 5-b. What processing is involved in computing `output` *Choice*?

i. A single recode: `choice == -1 → 1` (right), everything else → 0 (left). No-response trials (`choice == 0`) cannot reach this line because the trial mask already removed them, so the `else` branch only ever sees +1. The scalar is broadcast across all 100 bins so the output is time-varying in shape, and `output_values[0] = ['left', 'right']`. Observed split 0.508 / 0.492.

ii.
```python
outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int64)
outputs[:, 0, :] = choice[:, None]
```

iii. The instructions specify "Choice, binary, per-trial, left = 0, right = 1", and the format spec says "If at all possible, make it time-varying" — hence the broadcast to a constant time series rather than a bare scalar.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. `trials['probabilityLeft']`, the block prior held constant within a block.

ii.
```python
p_left_all = trials['probabilityLeft'].to_numpy(dtype=float)
prior_all = np.full(len(trials), -1, dtype=np.int64)
for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_all[np.isclose(p_left_all, value)] = code
```

iii. Directly from the instructions ("Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2"). This is the same column that defines the block structure used for input 2, so the two are consistent by construction.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Recode 0.2/0.5/0.8 → 0/1/2 using `np.isclose` (float tolerance rather than exact equality), initialising to a `-1` sentinel; any trial still at -1 is excluded by `keep &= (prior_all >= 0)`, so an unexpected prior value drops the trial rather than silently mislabelling it. The code is broadcast across the 100 bins and labelled `['p_left=0.2', 'p_left=0.5', 'p_left=0.8']`. Observed fractions 0.417 / 0.141 / 0.442 — the 0.5 minority is the unbiased opening block, which is kept.

ii.
```python
keep = trials_mask & np.isfinite(interval_begs_all) & (prior_all >= 0)
...
prior = prior_all[keep]
outputs[:, 1, :] = prior[:, None]
```

iii. Comment: "Block identity, one of the three probabilities the task defines." The `-1` sentinel plus the `keep` gate is a defensive choice so that a value outside the three the task defines can never be encoded as a valid class. The unbiased block is kept because the reference's `load_trials_and_mask` is called with `exclude_unbiased=False` (its default).

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `sess_loader.wheel['times']` and `sess_loader.wheel['velocity']` from `SessionLoader.load_wheel()`, i.e. the wheel velocity that ibllib derives from `_ibl_wheel.position` / `_ibl_wheel.timestamps`. Speed is the absolute value of that velocity.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_wheel()
traces['wheel-speed'] = (sl.wheel['times'].to_numpy(dtype=np.float64),
                         np.abs(sl.wheel['velocity'].to_numpy(dtype=np.float64)))
```

iii. Copied from the reference's `load_target_behavior(..., 'wheel-speed')`, which is `np.abs(sess_loader.wheel['velocity'].to_numpy())`. The docstring says so: "Follows the reference `load_target_behavior` / `bin_behaviors`: wheel speed is the absolute value of the Gaussian-smoothed wheel velocity". (Note the AI repeats the reference's own docstring wording "Gaussian-smoothed"; the current ibllib `load_wheel` interpolates position to 1 kHz and differentiates with a Butterworth low-pass.)

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps: (1) `SessionLoader.load_wheel()` interpolates the (movement-triggered) wheel position onto a uniform grid and computes a smoothed velocity; (2) absolute value → speed in rad/s; (3) linear interpolation of that trace onto the 100 per-trial sample times; (4) discretisation into 3 per-session levels. The interpolation is fully vectorised across all trials at once via a single `np.interp` on a flattened `(n_trials, 100)` query matrix. `np.interp` clamps rather than extrapolates outside the trace, which is harmless because the coverage mask already guarantees samples within one bin of both window edges.

ii.
```python
    offsets = (np.arange(N_BINS) + 1) * BINSIZE
    sample_times = interval_begs[:, None] + offsets[None, :]
    binned = np.interp(sample_times.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. Docstring of `bin_behaviour`: "The sample time of bin i is `beg + (i+1) * BINSIZE`, i.e. the right edge of the bin, matching the reference `get_behavior_per_interval` (`x_interp = linspace(beg + binsize, end, n_bins)`)." The AI read that function (step 13) and reproduced its grid exactly rather than choosing its own.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 levels at the 1/3 and 2/3 quantiles **of that session's own kept-trial values** (all trials × all bins pooled), via `np.digitize`. A degenerate trace (both quantiles equal) is nudged with `np.nextafter` so the bins stay ordered instead of collapsing. The two thresholds are stored per session in `metadata['session_info']`. Resulting marginals are 0.333 / 0.333 / 0.333.

ii.
```python
def discretize_tertiles(values):
    lo, hi = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if not hi > lo:  # degenerate (e.g. a mostly constant trace): keep bins ordered
        hi = np.nextafter(lo, np.inf)
    return np.digitize(values, [lo, hi]).astype(np.int64), (float(lo), float(hi))

wheel_lvl, wheel_thr = discretize_tertiles(beh_binned['wheel-speed'][keep])
```

iii. Argued at length in the docstring: "Thresholds are the 1/3 and 2/3 quantiles of that session's own values. Both behaviours are in arbitrary, session-dependent units … and wheel speed on how vigorously that mouse turns — so a global threshold would mostly encode which session a trial came from. Per-session tertiles make the three levels mean 'slow / medium / fast for this animal' in every session and keep the classes balanced, which is what the balanced-accuracy score the decoder is graded with assumes." The AI also inspected the empirical distributions of the behaviour traces (steps 59/61) before settling on tertiles.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The same `interval_begs = stimOn_times - 0.5` origin and the same 20 ms grid as the spikes, so index *i* of the output matches index *i* of the neural matrix. The only difference is the sampling point within the bin: the wheel trace is evaluated at the **right edge** `beg + (i+1)*0.02`, whereas the neural bin *i* aggregates `[beg + i*0.02, beg + (i+1)*0.02)` and the time input is the bin centre — a 10 ms offset. The wheel is already on the session clock, so no extra clock alignment is needed.

ii.
```python
    interval_ends = interval_begs + N_BINS * BINSIZE
    offsets = (np.arange(N_BINS) + 1) * BINSIZE
    sample_times = interval_begs[:, None] + offsets[None, :]
```

```python
'alignment_note': ('... wheel speed and whisker motion energy are linearly '
                   'interpolated onto its right edge, as in the reference pipeline.'),
```

iii. The offset is deliberate and documented in `alignment_note`: the AI chose fidelity to the reference's `x_interp = np.linspace(beg + binsize, end, n_bins)` over a bin-centre convention, and flagged the difference in the metadata so it is visible downstream.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `sess_loader.motion_energy['leftCamera']['times']` and `['whiskerMotionEnergy']` from `SessionLoader.load_motion_energy(views=['left'])` — the IBL-released ROI motion energy over a whisker-pad bounding box. If the left camera fails to load, the right camera is used instead; if neither is available the whole session is dropped (13 of the 15 skipped sessions).

ii.
```python
    whisker = None
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sl_me = SessionLoader(one=one, eid=eid)
            sl_me.load_motion_energy(views=[view])
            df = sl_me.motion_energy[cam]
            whisker = (df['times'].to_numpy(dtype=np.float64),
                       df['whiskerMotionEnergy'].to_numpy(dtype=np.float64))
            break
        except Exception:
            continue
    if whisker is None:
        raise RuntimeError('no whisker motion energy available')
```

iii. This is the reference's own fallback in `bin_behaviors`: "if beh == 'whisker-motion-energy': target_dict = load_target_behavior(one, eid, 'left-whisker-motion-energy'); if 'skip' in target_dict.keys(): … 'right-whisker-motion-energy'". The method-paper excerpt the AI read defines the signal: "Whisker motion energy is computed as the mean absolute difference between adjacent video frames within a bounding box anchored between nose tip and eye."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, normalisation or unit conversion. It goes through the identical `bin_behaviour` path as the wheel: linear interpolation onto the right edge of each of the 100 bins, a coverage/NaN mask, then per-session tertiles. Because left and right cameras run at different frame rates (60 vs 150 Hz), the interpolation is what puts both on a common grid.

ii.
```python
for name, (t, v) in traces.items():
    binned, good = bin_behaviour(t, v, safe_begs)
    beh_binned[name] = binned
    keep &= good
```

iii. Same justification as the wheel — the AI treats both continuous behaviours through one function so their grid, coverage test and discretisation are guaranteed identical, and it notes the camera-rate dependence as one reason the thresholds must be per-session: "whisker motion energy depends on the camera (left at 60 Hz vs right at 150 Hz), the illumination and the size of the bounding box".

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: 3 levels split at the 1/3 and 2/3 quantiles of that session's own kept values, `np.digitize`, `np.nextafter` guard for degenerate traces, thresholds recorded in `session_info`, labels `['low', 'medium', 'high']`. Marginals 0.333 / 0.334 / 0.333.

ii.
```python
whisk_lvl, whisk_thr = discretize_tertiles(beh_binned['whisker-motion-energy'][keep])
outputs[:, 3, :] = whisk_lvl
```

iii. See 7-c — the AI's argument is explicitly about *both* behaviours being in arbitrary session-dependent units, with the camera model/illumination/box size named as the specific confounds for motion energy.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as the wheel: the camera frame times are on the shared session clock, and the trace is interpolated onto `stimOn_times - 0.5 + (i+1)*0.02` for i = 0..99, giving one value per neural bin at the bin's right edge. Trials whose window is not covered by the camera to within one bin at either end are dropped rather than extrapolated.

ii.
```python
    i0 = np.searchsorted(times, interval_begs, side='right')
    i1 = np.searchsorted(times, interval_ends, side='left')
    good = i1 > i0
    good[idx] &= np.abs(interval_begs[idx] - times[i0[idx]]) <= BINSIZE
    good[idx] &= np.abs(interval_ends[idx] - times[i1[idx] - 1]) <= BINSIZE
```

iii. Same as 7-d; the coverage tolerance of one bin is the reference's ("`np.abs(interval_begs[interval_idx] - target_time[0]) > binsize` → 'target data starts too late'"), which the AI ported rather than inventing a tolerance.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data is dropped at the narrowest level that still leaves the rest usable:
- **Trial level:** NaN in any of the six required trial events; non-finite `stimOn`-derived window start; a `probabilityLeft` outside {0.2, 0.5, 0.8}; a wheel or camera trace that does not span the window or yields non-finite interpolated values.
- **Probe level:** an insertion whose spike sorting is empty, or that contributes zero surviving clusters, is skipped and the session continues on its other probe.
- **Spike level:** cluster ids out of range for the cluster table and non-finite spike times are discarded.
- **Session level:** no surviving units → dropped; fewer than 2 usable trials → dropped; no whisker motion energy on either camera → dropped. Any unhandled exception in a worker is caught, recorded with its traceback, and that session alone is skipped so the run completes.
- **Degenerate distributions:** a near-constant behavioural trace gets `hi = nextafter(lo)` so `digitize` still returns ordered classes.
- **Bookkeeping:** every dropped session and its reason is written into `metadata['skipped_sessions']` (15 entries), and per-session trial counts before/after filtering are kept as `n_trials_recorded` / `n_trials`.

ii.
```python
        ok = (sc >= 0) & (sc < lut.size)
        sc, st = sc[ok], st[ok]
        mapped = lut[sc]
        ok = mapped >= 0
        # Spike times can contain NaN for samples outside the sync range.
        ok &= np.isfinite(st)
```

```python
    except Exception as exc:  # noqa: BLE001 - a bad session must not stop the run
        return eid, None, f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'
```

```python
'skipped_sessions': [{'eid': e, 'reason': str(m).splitlines()[0]}
                     for e, m in sorted(skipped)],
```

iii. The catch-all is justified inline ("a bad session must not stop the run") and paired with a full traceback so failures are diagnosable rather than silent. The AI also investigated rather than papered over an anomaly: 38 trials verify as all-zero neural data, and at steps 81–86 it traced one of them to a genuine gap in the spike record, concluding "Spike gap is real data, not a bug", and left those trials in.

## 10-a. What are the most time-consuming steps of the code?

i. Dominated by I/O in `load_session_units`: `ssl.load_spike_sorting()` per probe, which reads the large `spikes.times`/`spikes.clusters` arrays (699 insertions over 459 sessions) and, because `check_hash` is left at its default, also re-reads each file to verify its md5. Two secondary costs: `load_spike_sorting` pulls the loader's full default `SPIKES_ATTRIBUTES` (`amps` and `depths` as well as `times` and `clusters`), roughly doubling the bytes read for arrays that are never used; and `SessionLoader` is constructed and made to load three separate times per session (trials, wheel, motion energy). Everything after loading is cheap — the per-trial spike-binning loop, and the behaviour interpolation which is a single vectorised `np.interp`. With 24 worker processes the whole 459-session run took 334 s wall-clock, and pickling the 11.73 GB result is itself a noticeable share of that.

ii.
```python
        ssl = SpikeSortingLoader(pid=str(row['pid']), one=one, eid=eid,
                                 pname=row['probe_name'])
        spikes, clusters, channels = ssl.load_spike_sorting()
```

```python
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, task): task[0] for task in tasks}
```

iii. The AI never states this in prose, but its design shows it identified I/O as the bottleneck: the whole conversion is process-parallel over sessions, and it pre-warms the Allen atlas in the parent with a comment — "Make sure the Allen atlas files are cached before the workers start, otherwise every worker downloads them at once" — i.e. it removed a known startup stampede. It measured single-session load times in steps 51–55 ("test the full pipeline on one session to understand timing") before choosing the worker count.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
1. `bin_spikes`'s `for k in range(n_trials)` loop. It could be done in one pass by offsetting each spike's flat index by its trial (`trial * n_units * N_BINS + u * N_BINS + b`) and calling `bincount` once, since `searchsorted` has already located every trial's slice. As written it also builds a fresh `(n_units * N_BINS,)` count vector per trial.
2. The per-probe loop in `load_session_units`. Genuinely sequential because each probe is a separate file read, so there is little to gain.
3. The three list comprehensions that unpack the stacked `(n_trials, ...)` arrays into per-trial lists (`[binned_spikes[k] for k in range(n_trials)]` etc.). These are required by the target format, but they leave `n_trials` non-contiguous views alive per session, which inflates the pickle.

Notably the behaviour path is *already* vectorised across trials — a single `np.interp` over a flattened query matrix, and a fully vectorised coverage mask — where the reference repo spawns a multiprocessing pool per behaviour per session.

ii.
```python
    for k in range(n_trials):
        t = spike_times[i0[k]:i1[k]]
        if t.size == 0:
            continue
        u = spike_units[i0[k]:i1[k]]
        b = np.floor((t - interval_begs[k]) / BINSIZE).astype(np.int64)
        np.clip(b, 0, N_BINS - 1, out=b)
        counts = np.bincount(u * N_BINS + b, minlength=flat)
        out[k] = counts.reshape(n_units, N_BINS)
```

```python
    binned = np.interp(sample_times.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. No explicit reasoning is given for leaving the spike loop scalar. The vectorisation of the behaviour path is implicitly justified by the AI having replaced the reference's nested-`multiprocessing` `get_behavior_per_interval` with plain NumPy — it already parallelises at the session level, so a second, inner pool would oversubscribe the machine.

## 10-c. What processing does the code repeat multiple times?

i. Four repetitions, all cheap-to-moderate:
1. **`SessionLoader` is built and loaded three (sometimes four) times per session** — once in `load_trials_and_mask`, once in `load_behaviour_traces` for the wheel, and once per camera view tried. One loader instance can serve `load_trials()`, `load_wheel()` and `load_motion_energy()`; the reference repo does exactly that, and its `load_trials_and_mask` even accepts a `sess_loader` argument for this purpose.
2. **`BrainRegions()` is instantiated once per session** inside `convert_session`, in addition to the one built in `main` to pre-warm the cache — so the atlas tables are re-parsed 459 times instead of once per worker.
3. **`acronym2acronym(..., mapping='Beryl')` is applied to the whole cluster table per probe** and then indexed down to the kept clusters; mapping only the kept rows would do the same work on ~10% of the rows (though the mask depends on the mapping, so one pass is unavoidable).
4. **Behaviour binning is computed for every trial in the session**, including trials the trials mask has already rejected (see 10-d).

ii.
```python
    sess_loader = SessionLoader(one=one, eid=eid)     # in load_trials_and_mask
    ...
    sl = SessionLoader(one=one, eid=eid); sl.load_wheel()          # in load_behaviour_traces
    ...
        sl_me = SessionLoader(one=one, eid=eid)                     # again, per camera view
        sl_me.load_motion_energy(views=[view])
```

```python
def convert_session(eid, probe_rows, subject, lab):
    from iblatlas.regions import BrainRegions
    one = get_one()
    brain_regions = BrainRegions()      # rebuilt for every session
```

iii. Not addressed in the trajectory. The repeated `SessionLoader` construction looks like a consequence of keeping `load_trials_and_mask` a faithful standalone re-implementation of the reference function (it creates its own loader, as the original does) rather than an oversight about cost; `get_one()` *is* memoised per process, so the expensive part (the ONE client and its cache tables) is shared.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Five items:
1. **Behaviour is binned for all trials, then thrown away for most of them.** `bin_behaviour` is called on `safe_begs`, which spans the *entire* trials table, so wheel and whisker traces are interpolated onto 100 bins for every recorded trial; only the `keep` subset survives (`beh_binned[...][keep]`). Typically ~30–40% of the interpolation work is discarded. It also fabricates `safe_begs = 0.0` windows for NaN-`stimOn` trials purely so the array shapes line up, then interpolates those too.
2. **Unused spike attributes are read from disk.** `load_spike_sorting()` fetches `amps` and `depths` in addition to `times` and `clusters`; only the latter two are ever touched. Setting `brainbox.io.one.SPIKES_ATTRIBUTES = ['clusters', 'times']` would halve the bytes read.
3. **The md5 hash of every spike file is recomputed** (`check_hash` defaults to True), re-reading data that has just been read.
4. **`SessionLoader.load_wheel()` computes and returns position and acceleration**, of which only `velocity` is used.
5. **The outputs are stored as `int64`** for four variables whose full range is {0, 1, 2}, and the neural counts as `float32` for small integers — 8× and 4× more bytes than needed, and the dominant reason the pickle is 11.73 GB.

Also discarded downstream but deliberately kept for provenance rather than by accident: `lab`, `n_trials_recorded`, the per-session region lists, the tertile thresholds and `skipped_sessions` in `metadata`.

ii.
```python
    safe_begs = np.where(np.isfinite(interval_begs_all), interval_begs_all, 0.0)
    for name, (t, v) in traces.items():
        binned, good = bin_behaviour(t, v, safe_begs)   # all trials, not just kept ones
        beh_binned[name] = binned
        keep &= good
    ...
    wheel_lvl, wheel_thr = discretize_tertiles(beh_binned['wheel-speed'][keep])
```

```python
    outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int64)   # values are only 0/1/2
```

iii. Item 1 is a deliberate trade, and the AI says so in a comment: "behaviour, on all trials so the masks line up with the trials table" — computing on the full table keeps every mask indexed against the same trial axis, which removes a class of off-by-one bugs at the cost of some wasted interpolation (and the interpolation is vectorised, so the cost is small). Items 2–5 are not discussed anywhere in the trajectory and appear to be unexamined defaults; the AI did check that the resulting file size and memory fit the machine (step 31 inspected `df -h`, `free -g`, `nproc`, `nvidia-smi`) and 11.73 GB was acceptable, so it had no forcing reason to tighten the dtypes.
