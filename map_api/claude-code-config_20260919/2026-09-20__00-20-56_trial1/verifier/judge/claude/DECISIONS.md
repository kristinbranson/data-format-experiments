# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI:000363) is one NWB file per session stored under `/app/data/sub-<subject_id>/`. The AI enumerates every file with a single sorted `glob` over that layout and processes each file exactly once. Each file is opened with `pynwb.NWBHDF5IO(path, 'r', load_namespaces=True)` and all streams are read from within that one handle: `nwb.subject`, `nwb.trials.to_dataframe()`, `nwb.units`, `nwb.electrodes`, `nwb.acquisition['BehavioralEvents']`, `nwb.acquisition['BehavioralTimeSeries']`. Sessions are farmed out to a `multiprocessing` `spawn` pool (default `--jobs 12`, run with 14), with per-session failures caught and reported rather than aborting the run. `--sample` selects 2 sessions (`files[1]` and `files[-3]`, deliberately one early and one late session). The full run found 174 files, converted 173, skipped 1, 0 errors, in 42 s.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
print(f'Found {len(files)} NWB files in {DATA_DIR}', flush=True)
if args.sample:
    # one early session and one late session, so the sample covers different rigs
    files = [files[1], files[-3]]
```

```python
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
    session_id = nwb.identifier
    subject = nwb.subject.description or nwb.subject.subject_id
    units = nwb.units
    ...
    trials = nwb.trials.to_dataframe()
    be = nwb.acquisition['BehavioralEvents']
```

```python
import multiprocessing as mp
ctx = mp.get_context('spawn')
with ctx.Pool(args.jobs) as pool:
    for i, res in enumerate(
            pool.imap(_worker, [(f, s, plot_dir) for f, s in zip(files, show_flags)])):
        results.append(res)
        _report(i, len(files), res, t0)
```

iii. From CONVERSION_NOTES Step 2: "`/app/data/sub-<subject_id>/sub-<subject_id>_ses-<...>.nwb` — 28 subject directories, **174 NWB files** (one per session), 50 GB. Read with `pynwb.NWBHDF5IO(..., load_namespaces=True)`." Because the release stores one session per file, the directory listing is the complete set of sessions and a glob suffices; sorting makes the order deterministic. `pynwb` is mandated by the instructions and is the standard reader for the published format. Parallelism was added as a documented Step-6/7 speed-up ("14 parallel worker processes — ~11x wall clock"), bringing the full conversion to ~42 s, far inside the 15-minute budget.

## 1-b. How are the data split into subjects?

i. One NWB file belongs to exactly one animal, so no grouping logic is needed. The AI reads the animal identity per session as `nwb.subject.description` (the lab mouse name, e.g. `SC015`), falling back to `nwb.subject.subject_id` (the numeric DANDI id, e.g. `440956`) if the description is empty. At assembly, `subjects` is the sorted set of unique names and `subject_idx` is each session's index into that list, in the same order as `neural`/`input`/`output`. Result: 28 subjects, 3–10 sessions each.

ii.
```python
subject = nwb.subject.description or nwb.subject.subject_id
...
return {
    'session_id': session_id,
    'subject': subject,
    ...
    'info': info,   # info also stores 'subject_id': nwb.subject.subject_id
}
```

```python
subjects = sorted({s['subject'] for s in sessions})
subj_idx = {s: i for i, s in enumerate(subjects)}

data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array([subj_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. Step 2 of CONVERSION_NOTES records that "`nwb.subject.subject_id` (e.g. `440956`), `nwb.subject.description` (lab ID, e.g. `SC015`)", and Step 5's variable mapping lists "`nwb.subject.description` → `subjects` / `subject_idx` — lab mouse ID, e.g. `SC015`". The lab ID is the identifier used in the papers (`nwb.identifier` is `SC015_20190208_133600_s2`), so using it makes the converted data directly comparable to the published figures; the numeric `subject_id` is retained in `metadata['session_info']` so nothing is lost. The count was checked against the paper: "Subjects | 28 | 'This study is based on data from 28 mice'", and the verification output confirms 28 subjects.

## 1-c. How are the data split into sessions?

i. The file boundary is the session boundary — one NWB file is one session — so nothing has to be inferred. Each session is identified by `nwb.identifier` (mouse, date, time, session number), and session order in the output follows the sorted file list. Per-session provenance (`session_id`, `subject`, `subject_id`, `session_start_time`, trial/neuron counts, timings, tongue percentiles, etc.) is written to `metadata['session_info']`. A session is dropped only if it has no units at all, no `classification == 'good'` units, or fewer than 2 usable trials. Exactly one file was dropped (`SC017_20190216_162508_s4`, 1,852 clusters but 0 good units), leaving 173 sessions.

ii.
```python
session_id = nwb.identifier
...
units = nwb.units
if units is None or len(units) == 0:
    return {'session_id': session_id, 'skipped': 'no units'}
classification = np.asarray(units['classification'][:])
good = np.where(classification == 'good')[0]
if len(good) == 0:
    return {'session_id': session_id, 'skipped': 'no good units'}
```

```python
'session_curation': 'All released sessions with at least one good unit (173 of 174 files).',
'session_info': [s['info'] for s in sessions],
```

iii. Step 4 records the discrepancy and its resolution: "Number of sessions | 173 (papers) | 174 NWB files | One file (`SC017_20190216_162508_s4`) has 1852 clusters but **0 good units** → 173 usable sessions. Dropped (no neural data to decode from)." Decision D2 justifies keeping every other session: "The DANDI release *is* the curated dataset of the paper (173 behavioural sessions), so the paper's session-selection criteria have already been applied upstream; re-applying my own reconstruction of them would only remove sessions the authors kept." The AI explicitly checked the paper's stated criteria (>65 % performance, ≥50 correct L and R trials) and found 22 released sessions fail them, which is why it chose not to re-apply them.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials.to_dataframe()`), one row per behavioural trial. The AI asserts that the number of `go_start_times` events equals the number of trial rows, i.e. exactly one go cue per trial, so the mapping trial-row ↔ go-cue is unambiguous. Every per-trial quantity (start/stop time, outcome, instruction, early lick, auto/free water, photostim) is taken from that same row index, and the retained trials are tracked through a single index vector `trial_idx` so all streams stay in register.

ii.
```python
trials = nwb.trials.to_dataframe()
n_trials_all = len(trials)
...
go_all = np.asarray(be['go_start_times'].timestamps[:], dtype=np.float64)
assert len(go_all) == n_trials_all, \
    f"{session_id}: {len(go_all)} go cues for {n_trials_all} trials"

start_all = trials['start_time'].values.astype(np.float64)
stop_all = trials['stop_time'].values.astype(np.float64)
outcome_all = np.asarray(trials['outcome'].values, dtype=object).astype(str)
instr_all = np.asarray(trials['trial_instruction'].values, dtype=object).astype(str)
early_all = np.asarray(trials['early_lick'].values, dtype=object).astype(str)
auto_all = np.asarray(trials['auto_water'].values).astype(int)
free_all = np.asarray(trials['free_water'].values).astype(int)
```

iii. Step 4's consistency table records the verification: "`task_cue_time[0,:]` (go cue) | `BehavioralEvents/go_start_times.timestamps` | 174/174 sessions: exactly one Go cue per trial, always inside `[start_time, stop_time]`". The AI also documented (Step 3) that licking during the sample/delay *replays* those epochs, so `sample_start_times` and `delay_start_times` can have several entries per trial — which is precisely why the go cue, not the sample onset, is used to index trials.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, applied in order, plus a session-level minimum:
1. **Ephys coverage** — a trial is kept only if one of `units.obs_intervals` rows matches its `(start_time, stop_time)`. Matching is done by time (`searchsorted` + `np.isclose` on both endpoints), *not* by assuming the ephys block is a prefix of the session, because in `SC026_20190807_134913_s20` the block runs from trial 126 to 630.
2. **Water-delivery trials** — `auto_water == 0` and `free_water == 0`.
3. **Trials with no spikes at all** — after binning, trials in which the entire simultaneously-recorded population fires zero spikes in the window are dropped (2 trials dataset-wide).
A session is skipped if fewer than 2 trials survive. Early-lick, `ignore` (no-response) and photostimulation trials are deliberately **kept**, contrary to the reference pipeline's `get_regular_trial_mask`. Net result: 89,544 of 94,990 trials retained (mean 517.6/session).

ii.
```python
# Ephys coverage.  units.obs_intervals lists, for each unit, the intervals the
# unit was observed in; every row matches exactly one trial's
# (start_time, stop_time) and all good units of a session share the same list
# (verified over all 173 sessions).  In 9 sessions it covers only part of the
# behavioural trials -- usually a prefix, but in SC026_20190807_134913_s20 a
# block starting at trial 126 -- so match by time rather than assuming a prefix.
obs = np.asarray(units['obs_intervals'][int(good[0])], dtype=np.float64)
...
ephys_covered = np.zeros(n_trials_all, dtype=bool)
j = np.clip(np.searchsorted(start_all, obs[:, 0] + 1e-6) - 1, 0, n_trials_all - 1)
matched = np.isclose(start_all[j], obs[:, 0]) & np.isclose(stop_all[j], obs[:, 1])
ephys_covered[j[matched]] = True

keep = ephys_covered.copy()                 # ephys coverage
keep &= (auto_all == 0) & (free_all == 0)   # no auto / free water (reference)
trial_idx = np.where(keep)[0]
n_trials = len(trial_idx)
if n_trials < 2:
    return {'session_id': session_id, 'skipped': f'only {n_trials} usable trials'}
```

```python
# A trial in which the entire simultaneously-recorded population fires zero
# spikes carries no neural data at all: it happens when the recording stops
# part-way through the session (typically the single last ephys trial).
has_spikes = rates.sum(axis=(0, 2)) > 0
n_empty = int(np.sum(~has_spikes))
if n_empty:
    rates = rates[:, has_spikes, :]
    trial_idx = trial_idx[has_spikes]
    go = go_all[trial_idx]
    n_trials = len(trial_idx)
```

iii. Decision D3: keep a trial iff "(a) it is within the range of trials for which the probes were recording (`is_good_trials`/`obs_intervals` length — drops 1,060 trials in 9 sessions that have *no* ephys at all), and (b) it is not an auto-water and not a free-water trial (3,766 trials, 4.0 %) — following the reference ('water administration regardless of the animals' choice (free water trials) … were excluded from all analyses'); on these trials reward is decoupled from the animal's action so the choice/outcome labels are not behaviourally meaningful." The retention of photostim/early-lick/ignore trials is called out as an explicitly allowed discrepancy: "the decoder task defines photostimulation as an input and early-lick and `ignore` as output classes." The non-prefix ephys block was found as a bug (Step 10, Issue 2) and fixed; a dedicated scan (`cache/scan_obs.py`) then confirmed every `obs_intervals` row in all 173 sessions maps to exactly one trial. The zero-spike filter was added in response to a `train_decoder --verify-only` warning (Step 10, Issue 1). Decision D5 explains why `is_good_trials` is used only for the coverage range and not to drop individual unit×trial pairs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units['spike_times']` (the ragged spike-time column, in session-absolute seconds) restricted to the rows with `units['classification'] == 'good'`, together with `BehavioralEvents/go_start_times.timestamps`, which supply the per-trial alignment time. `units['anno_name']` plus the CCF coordinates `x,y,z` from `nwb.electrodes` (via `units['electrodes']`) supply `brain_region_idx` for those same neurons.

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.where(classification == 'good')[0]
...
go_all = np.asarray(be['go_start_times'].timestamps[:], dtype=np.float64)
...
spike_lists = []
for i in good:
    st = np.asarray(units['spike_times'][int(i)], dtype=np.float64)
    if st.size > 1 and not np.all(np.diff(st) >= 0):
        st = np.sort(st)
    spike_lists.append(st)
```

```python
anno = np.asarray(units['anno_name'][:])[good]
elec_rows = np.asarray(units['electrodes'].target.data[:])[good]
etab = nwb.electrodes.to_dataframe()
xyz = etab[['x', 'y', 'z']].values[elec_rows]
region_idx = assign_brain_regions(anno, xyz)
```

iii. Step 1 notes this is "Electrophysiology, not imaging → no dF/F", and that the reference pipeline's neural representation is spike times binned into firing rates. Step 4 maps "`neuron_single_units` (spike times rel. Go) | `units.spike_times` (session clock) − Go time". `spike_times` is the only neural representation in the file.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin **firing rates in Hz**. For each good unit, the 81 bin edges of every trial are concatenated into one flat absolute-time query array, a single `np.searchsorted` gives the running spike count at each edge, `np.diff` over the bin axis gives the spike count per bin, and the whole array is divided by the 50 ms bin width. Output is `float32`, shape `(n_neurons, n_trials, 80)`, then sliced into one `(n_neurons, 80)` array per trial. No smoothing, normalisation, baseline subtraction or firing-rate threshold is applied.

ii.
```python
def bin_spike_rates(spike_times_list, go_times):
    """
    Equivalent to VideoAnalysisUtils.preprocessing_DJ_2022Aug.sliding_histogram with
    bin_width == stride == BIN_SIZE and rate=True, i.e. non-overlapping 50 ms bins
    covering [-T_PRE, +T_POST] relative to each trial's Go cue.
    """
    n_trials = len(go_times)
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()   # (n_trials*81,)
    out = np.empty((len(spike_times_list), n_trials, N_BINS), dtype=np.float32)
    for i, st in enumerate(spike_times_list):
        idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
        out[i] = np.diff(idx, axis=1)
    out /= np.float32(BIN_SIZE)
    return out
```

```python
neural_trials = [np.ascontiguousarray(rates[:, t, :]) for t in range(n_trials)]
```

iii. Decision D10: "**Firing rates (Hz), not spike counts**, matching `sliding_histogram(..., rate=True)`." Decision D4 justifies applying no rate threshold: "The method paper's 2 Hz cut is specific to its video→firing-rate ridge regressions ('the results were not sensitive to the exact value of this threshold'); it is not part of the dataset's quality control. Removing low-rate neurons would throw away information a population decoder can use." Step 10 Check 2 ran the **verbatim** reference `sliding_histogram` from `preprocessing_DJ_2022Aug.py` with `bin_width = stride = 0.05` on 5 neurons × 3 trials in 4 sessions and confirmed the stored rates are numerically identical and the bin centres match.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units['classification'] == 'good'` are retained — the verdict of the region-specific logistic-regression QC classifier. No thresholds are applied to any of the 15 individual quality metrics, and no firing-rate threshold is applied. A session with zero such units is dropped entirely. Result: 69,453 of 272,227 clusters (25.5 %), mean 401.5 and median 390 per session.

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.where(classification == 'good')[0]
if len(good) == 0:
    return {'session_id': session_id, 'skipped': 'no good units'}
```

```python
'neuron_curation': 'units.classification == "good" (region-specific logistic-regression QC '
                   'classifier of Chen, Liu et al. 2023).',
```

iii. Decision D1: "**Neuron curation: `classification == 'good'` only.** This is exactly the classifier-based QC described in `methods.txt` / the QC white paper and used by the reference pipeline (which loads only the `goodunits/*.mat` index). Reproduces 69,453 units (25.5 % of clusters) vs. the paper's 69,943 (25.9 %), and per-area counts within 2 % (Medulla exact)." Step 4 cross-checks per-area counts against the data paper: Medulla 2,928 vs 2,928 (exact), Midbrain −0.2 %, Striatum +1.0 %, Thalamus +1.6 %, ALM +1.9 %. Step 10 Check 4 investigated the −0.7 % total shortfall and concluded the published DANDI release simply contains 490 fewer classifier-'good' units than the paper quotes, confirmed by an independent scan of all 174 files.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Everything in the NWB file shares one session-absolute clock, so alignment requires no resampling or offset correction: the fixed grid of bin edges relative to the go cue is added to each trial's go-cue time to give that trial's absolute edge times, and spikes are binned directly against those edges. The alignment event is `BehavioralEvents/go_start_times.timestamps`, one per trial.

ii.
```python
T_PRE = 2.5          # s before the Go cue
T_POST = 1.5         # s after the Go cue
BIN_SIZE = 0.05      # s, non-overlapping bins
N_BINS = int(round((T_PRE + T_POST) / BIN_SIZE))          # 80
BIN_EDGES_REL = -T_PRE + np.arange(N_BINS + 1) * BIN_SIZE  # (81,) relative to Go
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2.0      # (80,)
```

```python
go = go_all[trial_idx]
...
edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()   # (n_trials*81,)
idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
```

```python
'temporal_alignment_event': 'Go cue onset (auditory, 6 kHz, 0.1 s), BehavioralEvents/go_start_times',
'off_start': -T_PRE,
'off_end': T_POST,
```

iii. Step 3: "**Temporal alignment**: everything is aligned to the **Go cue**. The reference spike data are already Go-cue-relative; markers are aligned by subtracting `task_cue_time[0]` (= `go_start_times`)." Step 10 Check 3 records alignment as "same" between reference and conversion. The half-open `[lo, hi)` edge convention was explicitly checked against the reference `sliding_histogram` (Step 10, Check 5: "half-open `[lo, hi)`, matching the reference `sliding_histogram`; verified numerically"). The AI further validated alignment behaviourally in Step 12: population choice AUC in ALM is at chance (0.528) before the instruction tone and first rises 0.18 s after tone onset, which a one-bin misalignment would destroy.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 80 non-overlapping 50 ms bins spanning −2.5 s to +1.5 s relative to the go cue, identical for every trial of every session (`T = 80` everywhere, confirmed by the verification log). The grid is defined once at module level as 81 relative edges and reused; spikes are binned straight from raw spike times onto that grid (no intermediate binning and hence no rebinning). The tongue video (~300 Hz) is *down*-binned onto the same 50 ms grid by averaging frames within each bin; the two scalar inputs are evaluated at the 50 ms bin centres. `metadata['time_bin_size'] = 50.0` ms.

ii.
```python
BIN_SIZE = 0.05      # s, non-overlapping bins
N_BINS = int(round((T_PRE + T_POST) / BIN_SIZE))          # 80
BIN_EDGES_REL = -T_PRE + np.arange(N_BINS + 1) * BIN_SIZE  # (81,) relative to Go
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2.0      # (80,)
```

```python
'time_bin_size': BIN_SIZE * 1000.0,          # ms
'n_time_bins': N_BINS,
'bin_centers_s': BIN_CENTERS_REL.tolist(),
```

iii. Step 4 lists this as a required, documented discrepancy from the reference: "Bin width | 40 ms width / 3.4 ms stride (reference) | **Required discrepancy**: the task specifies 50 ms bins. We use non-overlapping 50 ms bins (width = stride = 50 ms), i.e. the same `sliding_histogram` formula with `bw = stride = 0.05`", and likewise "Window | [−3, +3] s (reference) | **Required discrepancy**: the task specifies [−2.5, +1.5] s." A constant 80 bins per trial is also what the target format requires.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times.timestamps` (the onsets of the instruction-tone/sample epoch) together with the trial's go cue. Because an early lick replays the sample+delay epochs, a trial can contain several sample onsets; the AI takes the **last** sample onset at or before that trial's go cue.

ii.
```python
sample_t = np.asarray(be['sample_start_times'].timestamps[:], dtype=np.float64)
sample_t = np.sort(sample_t)
# last sample-epoch onset at or before the Go cue (an early lick replays the
# sample+delay epochs; the final presentation is the instructive one)
si = np.searchsorted(sample_t, go, side='right') - 1
tone_onset = np.where(si >= 0, sample_t[np.clip(si, 0, len(sample_t) - 1)], np.nan)
```

iii. Decision D7: "'Time from tone onset' = time since the *instruction* tone (sample epoch onset), not the Go cue: the trials are already aligned to the Go cue, so a Go-relative clock would carry no per-trial information, whereas the sample-onset-to-Go interval varies from 0.95 s to 10.4 s (early-lick replays) and is genuine task context. The *last* sample onset before the Go cue is used, because an early lick replays the sample+delay epochs and it is the final presentation that instructs the animal." Verified in Step 3/4: "median Go − last sample onset = 1.85 s = 0.65 + 1.2 s exactly", i.e. sample epoch plus delay epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The per-trial scalar `go − tone_onset` is added to the fixed vector of go-cue-relative bin centres, giving a (n_trials, 80) `float32` array of seconds since tone onset — a linearly increasing ramp within each trial whose offset varies per trial. A defensive fallback replaces the value with `go − median(go − tone)` if no sample onset precedes the go cue or if the found onset precedes the trial's own start time (this never triggered: "trials w/o tone onset before Go: 0"). Stored as row 0 of each trial's input array; observed range over the full dataset is [−1.5, 11.9] s.

ii.
```python
bad_tone = ~np.isfinite(tone_onset) | (tone_onset < start_all[trial_idx])
if np.any(bad_tone):
    # fall back to the median Go-to-tone interval of this session
    fallback = np.nanmedian(go[~bad_tone] - tone_onset[~bad_tone]) if np.any(~bad_tone) else 1.85
    tone_onset[bad_tone] = go[bad_tone] - fallback
go_minus_tone = go - tone_onset                                   # (n_trials,)
time_from_tone = (BIN_CENTERS_REL[None, :] + go_minus_tone[:, None]).astype(np.float32)
```

```python
input_trials = [np.stack([time_from_tone[t], photostim[t]]).astype(np.float32)
                for t in range(n_trials)]
```

iii. The instructions ask for "Time from **tone onset** in seconds (continuous, time-varying)", so a continuous per-bin value is emitted rather than a binary onset marker. Step 5's mapping table states the transform: "bin-centre time minus (Go − tone-onset), in seconds; time-varying". Step 10 Check 2 independently re-derived `bin_centre_abs − last_sample_onset_before_Go` for 24 random trials across 4 sessions and matched (24/24).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the input is evaluated at the **centres of the same bins** used for the spike histogram (`BIN_CENTERS_REL`, derived from the same `BIN_EDGES_REL` grid anchored on the same trial's go cue), so bin *k* of the input covers exactly the interval of bin *k* of the firing rates. No separate alignment or interpolation step exists.

ii.
```python
BIN_EDGES_REL = -T_PRE + np.arange(N_BINS + 1) * BIN_SIZE  # (81,) relative to Go
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2.0      # (80,)
```
```python
time_from_tone = (BIN_CENTERS_REL[None, :] + go_minus_tone[:, None]).astype(np.float32)
```
```python
edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()   # spikes use the same grid
```

iii. Step 7's plot review states the check: "the tone-onset input crosses zero exactly at the plotted tone onset" in `processing_<session>.png`, where the same panel also shows the raw event time and the go cue. The single shared grid is what makes the alignment automatic.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `BehavioralEvents/photostim_start_times.timestamps` and `BehavioralEvents/photostim_stop_times.timestamps` — the session-clock onset and offset of each laser event. (The trials table also carries `photostim_onset`/`photostim_duration`/`photostim_power`, which the AI used during exploration to verify the events, but the converted input is built from the event time series.) Sessions with no photostim events are handled by the `if len(ps_start)` guard.

ii.
```python
ps_start = np.asarray(be['photostim_start_times'].timestamps[:], dtype=np.float64)
ps_stop = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=np.float64)
order = np.argsort(ps_start)
ps_start, ps_stop = ps_start[order], ps_stop[order]
```

iii. Step 4's consistency table: "`task_stimulation` (power, type, on, off) | `trials.photostim_*` + `photostim_start/stop_times` | exactly one photostim event per photostim trial, 5.5 mW, 0.5 s, onset at −1.2 s or −0.5 s rel. Go ✓ (matches 'late delay, 0.5 s, ends before Go')". Using the event time series gives the laser on/off directly on the session clock, with no string parsing of the `'N/A'`-coded trials-table column, and matches the reference's `task_stimulation` on/off fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0.0/1.0) `float32` time series: a bin is 1 if its centre falls inside `[photostim_start, photostim_stop)` of some event, else 0. Implemented by `searchsorted` of the absolute bin centres into the sorted event-start array, taking the most recent event and testing the half-open interval. Trials with no stimulation are all-zero. Stored as row 1 of each trial's input array. 20.02 % of retained trials carry stimulation inside the window, against 20.04 % stim trials in the raw data.

ii.
```python
photostim = np.zeros((n_trials, N_BINS), dtype=np.float32)
if len(ps_start):
    abs_centers = go[:, None] + BIN_CENTERS_REL[None, :]           # (n_trials, N_BINS)
    k = np.searchsorted(ps_start, abs_centers, side='right') - 1
    valid = k >= 0
    kk = np.clip(k, 0, len(ps_start) - 1)
    on = valid & (abs_centers >= ps_start[kk]) & (abs_centers < ps_stop[kk])
    photostim[on] = 1.0
```

```python
'Binary indicator that ALM photoinhibition was on during the bin (bin centre inside '
'[photostim_start, photostim_stop)).',
```

iii. The instructions specify "Whether **photostimulation** is on at every time point (discrete, time-varying)", so a per-bin binary series is required rather than a per-trial flag. Retaining photostim trials at all is decision D3's documented departure from the reference's `get_regular_trial_mask`, justified because "the decoder task defines photostimulation as an input". Step 7's plot review confirms "the photostim step coincides with the raw event shading (-0.5 to 0 s or -1.2 to -0.7 s)", and Step 10 Check 2 re-derived the input with an independent loop over the raw events for 24 trials (24/24 OK).

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The bin centres are converted to absolute session time by adding the trial's go cue (`abs_centers = go[:, None] + BIN_CENTERS_REL[None, :]`) and compared directly against the raw event timestamps, which live on the same clock. Because those centres are the centres of the same bins used for the firing rates, the photostim series is on the neural grid by construction.

ii.
```python
abs_centers = go[:, None] + BIN_CENTERS_REL[None, :]           # (n_trials, N_BINS)
k = np.searchsorted(ps_start, abs_centers, side='right') - 1
on = valid & (abs_centers >= ps_start[kk]) & (abs_centers < ps_stop[kk])
```

iii. As with the tone input, no offset correction is needed since NWB stores everything on one global clock. The AI verified the result both statistically (stim onset always at −1.2 s or −0.5 s relative to the go cue, duration 0.5 s, always ending before the go cue) and visually (`processing_*.png` panel 4 overlays the binary input on the shaded raw laser interval).

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the file, so choice is derived from two trials-table columns: `trials['outcome']` (`hit`/`miss`/`ignore`) and `trials['trial_instruction']` (`left`/`right`). A hit means the animal licked the instructed side, a miss means it licked the opposite side, and an `ignore` means it did not lick.

ii.
```python
outcome = outcome_all[trial_idx]
instr = instr_all[trial_idx]
...
opposite = np.where(instr == 'left', 'right', 'left')
choice_str = np.where(outcome == 'ignore', 'no lick',
                      np.where(outcome == 'hit', instr, opposite))
```

iii. Step 4's consistency table records the independent validation: "`behavior_report` (1/0/−1) | `trials.outcome` (hit/miss/ignore) | choice reconstructed from `outcome`+`trial_instruction` agrees with the choice reconstructed independently from `left_lick_times`/`right_lick_times` on **99.7 %** of all 95k trials". The reference `.mat` export encodes the same information as `behavior_report` × `task_trial_type`, so the derivation reproduces the reference variable.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived string is mapped to integer codes `0 = left`, `1 = right`, `2 = no lick`, and written as row 0 of the per-trial `(4, 80)` output array, constant across all 80 bins. `output_values[0] = ['left', 'right', 'no lick']`. Dataset-wide distribution: left 0.429 / right 0.422 / no lick 0.148.

ii.
```python
choice = np.select([choice_str == 'left', choice_str == 'right'], [0, 1], default=2).astype(np.int64)
```

```python
OUTPUT_VALUES = [
    ["left", "right", "no lick"],
    ...
]
```

```python
output_trials = [
    np.stack([
        np.full(N_BINS, choice[t], dtype=np.int64),
        ...
    ])
    for t in range(n_trials)
]
```

iii. The instructions list choice as "(left, right, no lick, per-trial)", so three classes and a per-trial value. The AI repeats the value across the 80 bins so that all four outputs share one `(n_output, n_timepoints)` array, as the target format prefers time-varying outputs. Step 10 Check 2 spot-checked choice against the raw `outcome`/`trial_instruction` for 24 trials (all exact).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the NWB trials table, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'`. Step 2 verified there are no missing/`N/A` values in this column in any of the 174 files.

ii.
```python
outcome_all = np.asarray(trials['outcome'].values, dtype=object).astype(str)
...
outcome = outcome_all[trial_idx]
```

iii. Step 2: "`trial_instruction` ∈ {left, right}; `outcome` ∈ {hit, miss, ignore}; `early_lick` ∈ {early, no early} … No missing/`N/A` values in any of these." The column is already categorical with exactly the three classes the instructions ask for, so no derivation is needed; Step 4 maps it to the reference's `behavior_report`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to `0 = ignore`, `1 = miss`, `2 = hit` and written as row 1 of the output array, constant across the 80 bins. `output_values[1] = ['ignore', 'miss', 'hit']`. Dataset-wide distribution ignore 0.148 / miss 0.167 / hit 0.685, which matches the raw recomputation exactly (0.1484 / 0.1668 / 0.6849).

ii.
```python
outcome_code = np.select([outcome == 'ignore', outcome == 'miss'], [0, 1], default=2).astype(np.int64)
```

```python
np.full(N_BINS, outcome_code[t], dtype=np.int64),
```

iii. The code order follows the instructions' "Outcome (ignore, miss, hit, per-trial)". As with choice, the per-trial value is broadcast across bins to keep all outputs in one array. The AI notes in Step 12 that outcome only becomes decodable *after* the response (peak at +1.03 s), consistent with the data paper's report of outcome selectivity after licking offset — a behavioural sanity check on the labelling.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds the strings `'early'` and `'no early'`.

ii.
```python
early_all = np.asarray(trials['early_lick'].values, dtype=object).astype(str)
...
early = early_all[trial_idx]
```

iii. Step 4 maps this to the reference's `behavior_early_report` variable. The trials table flags early licking explicitly, so no derivation is needed. The AI also notes (Step 3) that the lick that sets the flag occurs during the sample or delay epoch, i.e. inside the −2.5 s pre-go window, which is why the flag is decodable from the window at all — and Step 12's time-resolved analysis confirms early_lick is most decodable during the sample/delay epochs (peak at −1.18 s).

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A boolean test `early == 'early'` cast to int gives `0 = no`, `1 = yes`, written as row 2 of the output array, constant across the 80 bins. `output_values[2] = ['no', 'yes']`. Dataset-wide distribution no 0.884 / yes 0.116 (raw 0.1161).

ii.
```python
early_code = (early == 'early').astype(np.int64)
```

```python
np.full(N_BINS, early_code[t], dtype=np.int64),
```

iii. Follows the instructions' "Early lick (no, yes, per-trial)". Step 10 Check 5 notes that 3 sessions have no early-lick trials at all and were kept, "a real property of those sessions". Step 12 Check 1 addresses the comparatively modest 1.51× chance ratio for this binary output: "the ceiling for a 2-class balanced accuracy is 1.0, i.e. only 2x chance, so 1.51x corresponds to 0.755 of a maximum of 1.0."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `nwb.acquisition['BehavioralTimeSeries']['Camera0_side_TongueTracking']` — the side-view DeepLabCut tracking, whose `data` is `(n_frames, 3)` = `(tongue_x, tongue_y, likelihood)` with matching per-frame `timestamps` at ~300 Hz. Column 1 (`y`) is the value; column 2 (`likelihood`) determines visibility. Present in all 174 sessions.

ii.
```python
tongue = nwb.acquisition['BehavioralTimeSeries']['Camera0_side_TongueTracking']
ttimes = np.asarray(tongue.timestamps[:], dtype=np.float64)
tdata = np.asarray(tongue.data[:], dtype=np.float64)
if ttimes.size > 1 and not np.all(np.diff(ttimes) >= 0):
    # two sessions store every trial's frames twice -> timestamps not monotone
    o = np.argsort(ttimes, kind='stable')
    ttimes, tdata = ttimes[o], tdata[o]
visible = tdata[:, 2] > TONGUE_LIKELIHOOD_THRESH
```

iii. Step 2: "`nwb.acquisition['BehavioralTimeSeries']` — DeepLabCut side-view markers, each `(n_frames, 3)` = `(x, y, likelihood)` with per-frame `timestamps`: `Camera0_side_TongueTracking` … (present in **all 174** sessions) … Frame period 3.4 ms (300 Hz)." Step 4 maps it to the reference's `tracking.camera_0_side.tongue_x/tongue_y`, adding that "NWB additionally provides col 2 = DLC likelihood", which the `.mat` export used by the reference does not expose.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with DLC `likelihood > 0.9` are marked visible; all others are discarded. Within each 50 ms bin of each trial, the mean `y` over the visible frames is computed (vectorised with a `searchsorted` on the visible-frame times plus a cumulative sum, so no Python loop over bins). A bin with no visible frame gets `count = 0` and mean NaN. The per-session 40th and 60th percentiles are then computed over **all visible binned means of that session** (all kept trials × 80 bins), and each visible bin is digitised against those two edges.

ii.
```python
def bin_visible_mean(times, values, visible, go_times):
    n_trials = len(go_times)
    tv = times[visible]
    yv = values[visible]
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
    idx = np.searchsorted(tv, edges).reshape(n_trials, N_BINS + 1)
    csum = np.concatenate([[0.0], np.cumsum(yv.astype(np.float64))])
    counts = np.diff(idx, axis=1)
    sums = np.diff(csum[idx], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return mean, counts
```

```python
mean_y, vis_counts = bin_visible_mean(ttimes, tdata[:, 1], visible, go)
tongue_class, pcts = discretise_tongue(mean_y, vis_counts)
```

iii. Decision D8: "**Tongue visibility threshold: DLC likelihood > 0.9.** The likelihood distribution is strongly bimodal (85.6 % < 0.01, 14.0 % > 0.99), so the result is insensitive to the threshold between 0.1 and 0.99. A bin counts as visible if *any* frame in the 50 ms bin is above threshold (tongue protrusions last only ~50–100 ms), and the reported y is the mean over those frames. The method paper's own outlier handling ('when the tongue was occluded … we set the tongue position to its mean value') is not applicable here because the task specifies an explicit 'not visible' class." Decision D9: "Percentiles for the tongue discretisation are computed per session over all *visible* binned y values of that session (all kept trials × 80 bins), matching '40th percentile of y-position over the session'." Step 10 Check 3 flags the deliberate difference from the reference marker alignment: "frames assigned to 50 ms bins, using the **mean** of the visible frames in each bin … at 50 ms a bin holds ~15 frames, so the mean is a better summary than the last sample."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, exactly as the instructions define: `0` if the binned mean y is below the session's 40th percentile, `1` if between the 40th and 60th percentiles inclusive, `2` if above the 60th percentile, and `3` ("not visible") if the bin contains no frame above the likelihood threshold. If a session has no visible bin at all, every bin is class 3 and the percentiles are NaN. The realised distribution is 0.098 / 0.049 / 0.098 / 0.755 — exactly 40 : 20 : 40 of the visible bins, as the percentile definition demands.

ii.
```python
def discretise_tongue(mean_y, counts):
    """
      0 : y  < 40th percentile of the session's visible y values
      1 : 40th <= y <= 60th percentile
      2 : y  > 60th percentile
      3 : tongue not visible in this bin
    """
    vis = counts > 0
    out = np.full(mean_y.shape, 3, dtype=np.int64)
    if not np.any(vis):
        return out, (np.nan, np.nan)
    p40, p60 = np.percentile(mean_y[vis], [TONGUE_PCT_LOW, TONGUE_PCT_HIGH])
    y = mean_y[vis]
    cls = np.where(y < p40, 0, np.where(y > p60, 2, 1))
    out[vis] = cls
    return out, (float(p40), float(p60))
```

```python
OUTPUT_VALUES = [..., ["<40th pct", "40-60th pct", ">60th pct", "not visible"]]
```

iii. This is a direct transcription of the Decoder Task specification ("0: < 40th percentile of y-position over the session; 1: 40th to 60th percentile; 2: > 60th percentile; 3: not visible"), with the percentiles taken per session as specified. The AI verified the resulting class proportions against the definition (Step 7: "The tongue distribution is exactly 40 % / 20 % / 40 % of the *visible* bins, as the percentile definition requires (0.100 : 0.050 : 0.100 = 4 : 2 : 4)") and independently recomputed the percentiles from the raw tracking with a per-bin boolean mask in 4 sessions, agreeing to 1e-12 (Step 10, Check 2).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps share the session-absolute clock with spikes and events, so the identical `BIN_EDGES_REL` grid anchored on each trial's go cue is used: the absolute bin edges are the same array used for spike binning, and frames are assigned to bins by `searchsorted` into the visible-frame timestamps. Bin *k* of the tongue output therefore covers exactly the same interval as bin *k* of the firing rates. Video timestamps are stably sorted first in the two sessions where each trial's frames are stored twice. Where the trial ends before +1.5 s (miss trials) or video is missing, the affected bins simply have no frames and become class 3.

ii.
```python
edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
idx = np.searchsorted(tv, edges).reshape(n_trials, N_BINS + 1)
```
```python
if ttimes.size > 1 and not np.all(np.diff(ttimes) >= 0):
    o = np.argsort(ttimes, kind='stable')
    ttimes, tdata = ttimes[o], tdata[o]
```

iii. Step 10 Check 3 records "temporal alignment | Go cue (`task_cue_time[0]`), also for the video markers | Go cue (`go_start_times.timestamps`), also for the tongue marker | same". Using one shared grid for both streams removes any possibility of a per-stream offset. Step 7's plot review confirms "the binned tongue means track the raw visible y samples and the class switches to 'not visible' exactly where no visible frame exists". The truncation consequence is documented in `metadata['known_limitations']` and decision D6.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Eight distinct cases, each handled explicitly and documented in Step 10 Check 5:
- **Session never quality-controlled** (`classification` NaN for all units, `SC017_20190216_162508_s4`): no unit compares equal to `'good'`, so the session is skipped and reported.
- **Session with no units at all**: skipped.
- **Trials outside the ephys recording** (9 sessions): dropped via `obs_intervals` time matching.
- **Ephys block not starting at trial 1** (`SC026_20190807_134913_s20`, trials 126–630): handled by matching on `(start_time, stop_time)` rather than assuming a prefix — this was a bug found and fixed in Step 10.
- **Trials with zero population spikes** (2 dataset-wide): dropped after binning.
- **Non-monotonic video timestamps** (2 sessions store every trial's frames twice): stably sorted before binning.
- **Missing/short video** (2 sessions) and retracted tongue: represented as the explicit `'not visible'` class rather than imputed.
- **No tone onset before the go cue** (0 occurrences) and **NaN CCF coordinates**: defensive fallbacks (session-median go-to-tone interval; NaN fails the ALM coordinate test and the unit falls back to `OtherCortex`).
Sessions left with fewer than 2 usable trials are skipped. Errors in a worker are caught, logged with a traceback, and do not abort the run.

ii.
```python
if units is None or len(units) == 0:
    return {'session_id': session_id, 'skipped': 'no units'}
classification = np.asarray(units['classification'][:])
good = np.where(classification == 'good')[0]
if len(good) == 0:
    return {'session_id': session_id, 'skipped': 'no good units'}
```

```python
if st.size > 1 and not np.all(np.diff(st) >= 0):
    st = np.sort(st)
```

```python
bad_tone = ~np.isfinite(tone_onset) | (tone_onset < start_all[trial_idx])
if np.any(bad_tone):
    fallback = np.nanmedian(go[~bad_tone] - tone_onset[~bad_tone]) if np.any(~bad_tone) else 1.85
    tone_onset[bad_tone] = go[bad_tone] - fallback
```

```python
# Coordinates are NaN for a handful of units; treat those as failing the ALM test.
with np.errstate(invalid='ignore'):
    is_alm = is_frontal & (ap_mm > ALM_MIN_AP_MM) & (ml_mm < ALM_MAX_ML_MM)
```

```python
def _worker(args):
    path, show, plot_dir = args
    try:
        return process_session(path, show_processing=show, plot_dir=plot_dir)
    except Exception as exc:       # keep the whole run alive, report at the end
        import traceback
        return {'session_id': os.path.basename(path), 'error': repr(exc),
                'traceback': traceback.format_exc()}
```

iii. The governing principle is stated across Step 5 and Step 10: where nothing was recorded, the unit/trial/session is excluded rather than emitted as fabricated zeros; where a measurement legitimately has no value (retracted tongue), it becomes an explicit category. Decision D6 explains the one case where partially missing data is *kept*: "81.5 % of trials fully cover [−2.5, +1.5] s … Excluding partially-covered trials would delete ~91 % of all *miss* trials and make the `outcome` output undecodable, so all trials are kept and the missing tail is simply empty (zero firing rate, tongue 'not visible'). This is documented as a known property of the release", and is surfaced to the user in `metadata['known_limitations']`.

## 10-a. What are the most time-consuming steps of the code?

i. The AI instrumented every stage (`timing` dict per session, printed for every file). Per session (~1.3 s serial): opening the NWB file and reading the trials table ~0.25 s; reading the good units' spike trains ~0.4 s; binning the spikes ~0.4 s; regions + inputs + outputs + assembly ~0.2 s. The whole conversion is I/O- and `searchsorted`-bound. With 14 worker processes the full 174-file run took 42 s wall clock, plus ~15 s to pickle the 11.89 GB result.

ii.
```python
t0 = time.time()
spike_lists = []
for i in good:
    st = np.asarray(units['spike_times'][int(i)], dtype=np.float64)
    ...
timing['read_spikes'] = time.time() - t0

t0 = time.time()
rates = bin_spike_rates(spike_lists, go)      # (n_neurons, n_trials, N_BINS)
timing['bin_spikes'] = time.time() - t0
```

```python
print(f'[{i + 1}/{n}] {tag}: {info["n_neurons"]} neurons, '
      f'{info["n_trials_kept"]}/{info["n_trials_all"]} trials, '
      f'{info["total_time"]}s  {info["timing"]}  '
      f'[elapsed {el:.0f}s, eta {el / (i + 1) * (n - i - 1):.0f}s]', flush=True)
```

iii. Step 7's run-time table gives the per-step breakdown and the estimate ("~220 s serial, ~25 s on 14 workers"), and Step 9 reports the realised 42 s. The instructions require the estimate to be below 15 minutes; the AI concluded "Estimated (and realised) full conversion: ~40 s, far below the 15 minute budget, so no further optimisation was needed."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python loops remain, and the AI's position (Step 6) is that the expensive ones are already vectorised:
1. **Per-unit HDF5 read** (`for i in good: units['spike_times'][int(i)]`) — one ragged-row read per good unit. This is the one genuinely vectorisable loop left: the whole `spike_times` target buffer plus the index offsets could be read once and sliced in memory (as the reference solution does), avoiding ~400 separate HDF5 fancy-index reads per session. It is also where the per-unit monotonicity check `np.all(np.diff(st) >= 0)` is paid.
2. **Per-unit `searchsorted`** in `bin_spike_rates` — inherently ragged (each unit has a different number of spikes, so there is no single sorted array to search); the trial dimension is already vectorised by flattening all 81 edges of all trials into one query.
3. **Per-trial list comprehensions** building `neural_trials` / `input_trials` / `output_trials` — required by the target format, which asks for a list of per-trial arrays.
The tongue binning, which in a naive implementation would loop over bins or trials, is fully vectorised with a cumulative sum.

ii.
```python
spike_lists = []
for i in good:
    st = np.asarray(units['spike_times'][int(i)], dtype=np.float64)
    if st.size > 1 and not np.all(np.diff(st) >= 0):
        st = np.sort(st)
    spike_lists.append(st)
```

```python
for i, st in enumerate(spike_times_list):
    idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
    out[i] = np.diff(idx, axis=1)
```

```python
csum = np.concatenate([[0.0], np.cumsum(yv.astype(np.float64))])
counts = np.diff(idx, axis=1)
sums = np.diff(csum[idx], axis=1)
```

iii. Step 6: "Binning is fully vectorised. For a session, the 81 bin edges of every trial are concatenated into one `(n_trials*81,)` query array and `np.searchsorted` is called once per neuron; the per-bin counts are `np.diff` of the resulting indices. Equivalent to a per-trial `np.histogram` but ~100x faster … The tongue y mean per bin uses the same trick plus a cumulative sum of the visible-frame y values, so no Python loop over bins is needed." Identified inefficiencies and fixes: "Code inefficiencies identified: naive per-trial `np.histogram` over ~400 neurons x ~500 trials; per-bin boolean masking of ~1e6 video frames. Code speedups added: single `searchsorted` per neuron over all trial edges; cumulative-sum binning for the video; parallelism over sessions; reading only the `good` units' spike trains." The remaining per-unit read loop is not called out in the notes; given the realised 42 s runtime it costs nothing practically, and reading only good units (≈25 % of clusters) partially offsets the extra read overhead.

## 10-c. What processing does the code repeat multiple times?

i. Very little. Each NWB file is opened exactly once and each quantity derived from it is computed once; the bin grid (`BIN_EDGES_REL`, `BIN_CENTERS_REL`) is built once at module level and reused for every trial, session and stream; CCF annotation → coarse-area lookups are memoised in `_ANNO_CACHE` so each distinct annotation string is parsed once per worker process; `nwb.electrodes.to_dataframe()` is called once per session. The only repetitions are trivial: `go = go_all[trial_idx]` is recomputed after the zero-spike filter, the firing rates of the (2 dataset-wide) zero-spike trials are computed and then discarded, and a monotonicity check is run over every unit's spike train and over the video timestamps of every session even though only a handful are affected. Because the tongue percentiles are per-session, they are computed inside the same single pass — no second pass over the data is needed.

ii.
```python
_ANNO_CACHE = {}

def _coarse_group(anno):
    """Coarse group for one CCF annotation string (before the ALM coordinate test)."""
    if anno in _ANNO_CACHE:
        return _ANNO_CACHE[anno]
    ...
    _ANNO_CACHE[anno] = out
    return out
```

```python
BIN_EDGES_REL = -T_PRE + np.arange(N_BINS + 1) * BIN_SIZE  # (81,) relative to Go
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2.0      # (80,)
```

```python
if n_empty:
    rates = rates[:, has_spikes, :]
    trial_idx = trial_idx[has_spikes]
    go = go_all[trial_idx]          # recomputed
```

iii. Not discussed explicitly as "repetition" in CONVERSION_NOTES, but the single-pass design is implicit in the Step 6/7 timing analysis and in the speed-up list ("avoid unnecessary file I/O" is satisfied by one `with NWBHDF5IO(...)` block per session). The 42 s full-run wall clock corroborates that nothing substantial is recomputed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A modest amount, all of it either mandated by the target format or kept deliberately for documentation:
- **Brain-region assignment** — the ~150-line CCF substring ruleset plus the ALM coordinate test, the electrodes dataframe read and the coordinate conversion. `brain_region_idx` is required by the target format but `train_decoder.py` never uses it (it only reads `neural`, `input`, `output`, `output_values`, `input_names`, `output_names`), so this work does not affect any decoding result.
- **Unused video channel** — the full `(n_frames, 3)` tracking array is materialised, but only columns 1 (`y`) and 2 (`likelihood`) are used; `tongue_x` is read and discarded.
- **Diagnostics** — the per-session `info` dict (timings, `n_obs_intervals_unmatched`, `n_trials_empty_dropped`, `frac_tongue_visible`, `go_minus_tone_median`, tongue percentiles, probe counts) is computed and stored in `metadata['session_info']`; it is documentation, not decoder input.
- **Discarded work** — firing rates are computed for the trials subsequently removed by the zero-spike filter, and the monotonicity checks over every spike train / timestamp array are no-ops in all but a few sessions.
- `--show-processing` plotting is opt-in and off by default.
None of this is on the critical path: the whole conversion runs in 42 s.

ii.
```python
anno = np.asarray(units['anno_name'][:])[good]
elec_rows = np.asarray(units['electrodes'].target.data[:])[good]
etab = nwb.electrodes.to_dataframe()
xyz = etab[['x', 'y', 'z']].values[elec_rows]
region_idx = assign_brain_regions(anno, xyz)
```

```python
tdata = np.asarray(tongue.data[:], dtype=np.float64)   # cols 0,1,2; col 0 unused
```

```python
info = {
    ...
    'n_obs_intervals_unmatched': int(len(obs) - matched.sum()),
    'n_trials_empty_dropped': n_empty,
    'tongue_pct40': pcts[0], 'tongue_pct60': pcts[1],
    'frac_tongue_visible': float(np.mean(vis_counts > 0)),
    'frac_photostim_trials': float(np.mean(photostim.max(axis=1) > 0)),
    'go_minus_tone_median': float(np.median(go_minus_tone)),
    'timing': {k: round(v, 3) for k, v in timing.items()},
    'total_time': round(time.time() - t_start, 2),
}
```

iii. CONVERSION_NOTES does not frame any of this as wasted work. The region mapping is justified in decision D11 as reproducing "the region list used by the reference preprocessing code" and is validated against the paper's per-area unit counts (Medulla exact, others within 2 %) — i.e. it doubles as a consistency check on the neuron curation, and the target format requires `brain_regions`/`brain_region_idx` regardless of whether the reference decoder consumes them. The per-session `info` is explicitly intended for the user, and Step 13 lists `metadata['session_info']` as part of the deliverable documentation.
