# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session under `/app/data/sub-<subject_id>/`. The AI discovers every session with a single sorted glob over that layout, then hands each file to a worker process that opens it once with `pynwb.NWBHDF5IO` inside a `with` block and reads everything it needs from that handle (`nwb.intervals['trials']`, `nwb.acquisition['BehavioralEvents']`, `nwb.acquisition['BehavioralTimeSeries']`, `nwb.units`, `nwb.electrodes`, `nwb.subject`). Sessions are converted in parallel with a `multiprocessing.Pool` (default 16 workers, 24 used for the full run). No `h5py` is used anywhere.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
if args.sample:
    files = files[:2]
```

```python
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
    session_id = nwb.identifier
    subject_id = str(nwb.subject.subject_id)
    mouse_name = str(nwb.subject.description)
    trials = nwb.intervals['trials']
    df = trials.to_dataframe()
    ...
    be = nwb.acquisition['BehavioralEvents'].time_series
    ...
    units = nwb.units
```

```python
from multiprocessing import Pool
with Pool(min(args.nproc, len(files))) as pool:
    for i, r in enumerate(pool.imap(_worker, jobs)):
```

iii. From CONVERSION_NOTES Step 2/Step 6: the NWB release already contains all probes of a session in one file, so one file = one complete session and a directory glob is the complete session list; sorting makes the order deterministic. The AI cross-checked the file count against the papers (174 files, 28 subjects, 173 sessions with good units) and against `dandiset.yaml`. Parallelism was chosen because each worker only needs its own file handle, giving ~16-24x throughput (full conversion: 23.3 s).

## 1-b. How are the data split into subjects?

i. Each NWB file names its animal in `nwb.subject.subject_id` (numeric string, e.g. `'440956'`). That id is carried per session; at assembly `subjects` is the sorted set of unique ids and `subject_idx` indexes into it per session. The AI additionally records the mouse name used in the papers (`nwb.subject.description`, e.g. `SC015`) in `metadata['session_info']` as provenance, but does not use it for grouping.

ii.
```python
subject_id = str(nwb.subject.subject_id)
mouse_name = str(nwb.subject.description)
```

```python
subjects = sorted(set(r['subject_id'] for r in sessions))
subject_index = {s: i for i, s in enumerate(subjects)}
data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array([subject_index[r['subject_id']] for r in sessions], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 4/Step 9: `subject_id` is the canonical animal identifier in the file (the folder name `sub-440956` is derived from it), so no separate grouping step is needed. The AI verified the result gives 28 subjects, matching the papers' "28 mice", with 3-10 sessions each. It kept the numeric id as the key and stored the paper-facing mouse name alongside it.

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no grouping or splitting is performed. Session identity is `nwb.identifier` (e.g. `SC015_20190207_120657_s1`, encoding mouse, date, time and session number). Session order in the output is the sorted file order. One session is dropped (no good units), giving 173 sessions.

ii.
```python
session_id = nwb.identifier
```

```python
sessions = [r for r in results if 'neural' in r]
...
'session_info': [{'session_id': r['session_id'], 'subject_id': r['subject_id'],
                  'mouse_name': r['mouse_name'], 'n_units': r['n_units_used'],
                  'n_trials': r['n_trials_used'], 'n_trials_in_file': r['n_trials_all'],
                  ... } for r in sessions],
```

iii. CONVERSION_NOTES Step 4 "Discrepancies Found": the data contain 174 files but the papers report 173 behavioral sessions. The AI resolved this by finding that `SC017_20190216_162508_s4` has 0 good units, and independently confirmed the resolution with the probe-count distribution (data `{2:2, 3:53, 4:99, 5:20}` vs paper `{2:2, 3:53, 4:98, 5:20}`; the dropped session has 4 probes, giving an exact match).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.intervals['trials'].to_dataframe()`), one row per behavioural trial. The AI asserts there is exactly one `go_start_times` event per trial row and raises if not, and additionally requires each kept trial's go cue to fall inside `[start_time, stop_time]`.

ii.
```python
trials = nwb.intervals['trials']
df = trials.to_dataframe()
n_trials_all = len(df)
start_time = df['start_time'].values.astype(float)
stop_time = df['stop_time'].values.astype(float)
...
go_times = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
if len(go_times) != n_trials_all:
    raise RuntimeError('%s: %d go cues for %d trials' % (session_id, len(go_times), n_trials_all))
```

```python
keep &= (go_times >= start_time) & (go_times <= stop_time)
```

iii. CONVERSION_NOTES Step 4: the AI verified that `go_start_times` has "exactly 1 per trial in all 174 sessions", which makes the trial-row/go-cue mapping unambiguous, while noting that `sample_start_times`/`delay_start_times` have *more* events than trials because early-lick trials replay those epochs. It therefore uses the trials table directly rather than re-deriving trial boundaries from event streams.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, all conjunctive:
1. `auto_water == 0` and `free_water == 0` (reward not contingent on choice) — taken from the reference code's `get_regular_trial_mask`.
2. Go cue inside the trial interval (sanity).
3. Electrophysiological coverage: the trial must be listed in `units/obs_intervals` for **every** retained good unit.
4. At least one spike across all retained good units within the trial.
A session is dropped if fewer than 2 trials survive. Early-lick, no-response (`ignore`) and photostim trials are deliberately **kept**. Dataset totals: 3,789 auto/free-water trials, 1,060 trials without ephys coverage and 2 zero-spike trials removed; 89,544 trials retained.

ii.
```python
keep = (auto_water == 0) & (free_water == 0)
# the go cue must lie inside the trial (sanity)
keep &= (go_times >= start_time) & (go_times <= stop_time)
```

```python
obs_trial = np.ones(n_trials_all, dtype=bool)
obs_index = units['obs_intervals']
for ui in unit_ids:
    oi = np.asarray(obs_index[int(ui)], dtype=float)
    m = np.zeros(n_trials_all, dtype=bool)
    if oi.size:
        rows = np.searchsorted(start_time, oi[:, 0] + 1e-6) - 1
        rows = rows[(rows >= 0) & (rows < n_trials_all)]
        m[rows] = True
    obs_trial &= m
keep &= obs_trial

trial_idx = np.where(keep)[0]
n_trials = len(trial_idx)
if n_trials < 2:
    return None
```

```python
has_spikes = spikes_in_trial > 0
n_trials_nospike = int(np.sum(~has_spikes))
if n_trials_nospike:
    trial_idx = trial_idx[has_spikes]
    ...
```

iii. CONVERSION_NOTES Step 5 Key Decision 3 and Step 10 Check 1: auto-water/free-water trials are excluded "because on those trials reward is delivered independently of the animal's choice, so choice/outcome labels do not reflect a decision", matching the reference `get_regular_trial_mask`. The reference's other exclusions (early lick, no-response, photostim) are documented as a deliberate deviation because "the decoder task explicitly requires early lick and outcome=ignore as outputs and photostimulation as an input". The `obs_intervals` and zero-spike filters were added in response to `train_decoder.py` warnings: "In 8/173 sessions the ephys recording stops before the behavioural session ends" and "`obs_intervals` lists one extra trailing trial after the last recorded spike (last spike 1107.44 s, trial spans 1109.39-1114.47 s)". The 2-trial minimum follows the target-format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (session-absolute seconds), restricted to units passing the quality filter (`units/classification == 'good'` with a mappable `units/anno_name`). The other inputs are `BehavioralEvents/go_start_times` (bin placement) and `trials.start_time`/`stop_time` (edge clipping).

ii.
```python
spike_index = units['spike_times']
for k, ui in enumerate(unit_ids):
    st = np.asarray(spike_index[int(ui)], dtype=float)
    pos = np.searchsorted(st, flat_edges).reshape(n_trials, NBINS + 1)
    rates[k] = np.diff(pos, axis=1).astype(np.float32)
```

```python
go = go_times[trial_idx]
edges = go[:, None] + BIN_EDGES_REL[None, :]           # (n_trials, NBINS+1)
edges_clipped = np.clip(edges, tstart[:, None], tstop[:, None])
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: `units.spike_times` is the only neural representation in the file, and the reference pipeline likewise computes rates from spike times (`sliding_histogram`, `process_one_area`). The AI notes the difference that in the reference `.mat` export spike times were already go-cue-aligned, whereas in NWB they are in session time so the go cue must be subtracted.

## 2-b. How is the `neural` data processed?

i. Per-bin spike counts converted to firing rate in Hz. For each retained unit, the flattened `(n_trials x 81)` array of absolute bin edges — clipped to the trial interval — is passed to a single `np.searchsorted` against that unit's sorted spike times; differencing adjacent positions gives the count per bin; the whole array is then divided by the 50 ms bin width. Stored as `float32`. No smoothing, normalisation or baseline subtraction.

ii.
```python
flat_edges = edges_clipped.ravel()
trial_bounds = np.stack([tstart, tstop], axis=1).ravel()
rates = np.zeros((n_units, n_trials, NBINS), dtype=np.float32)
spikes_in_trial = np.zeros(n_trials, dtype=np.int64)
spike_index = units['spike_times']
for k, ui in enumerate(unit_ids):
    st = np.asarray(spike_index[int(ui)], dtype=float)
    pos = np.searchsorted(st, flat_edges).reshape(n_trials, NBINS + 1)
    rates[k] = np.diff(pos, axis=1).astype(np.float32)
    tb = np.searchsorted(st, trial_bounds).reshape(n_trials, 2)
    spikes_in_trial += tb[:, 1] - tb[:, 0]
rates /= BIN_SIZE                                        # spikes/s, as in the reference
```

iii. CONVERSION_NOTES Step 5 and Step 10 Check 3: "Same estimator, different bin width (required by the task)" — the reference `sliding_histogram` also returns `binSpikes / bin_width` in Hz, but with a 40 ms width and 3.4 ms stride chosen to match the 300 Hz video; the decoder specification here mandates non-overlapping 50 ms bins. Edge clipping is justified in Step 10 Check 5: "bin edges are clipped to the trial interval so spikes/video from the neighbouring trial can never leak in; the unobserved bins are empty (rate 0, tongue class 3)".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept if `units/classification == 'good'` **and** their CCF annotation (`units/anno_name`) maps onto one of the 14 coarse region groups used by the reference code **and** the unit's electrode has a finite CCF coordinate. Sessions with no surviving units are dropped. No thresholds are applied to any individual QC metric, and no firing-rate threshold is used. Result: 69,453 units over 173 sessions, mean 401.5 per session (range 90-923) — identical to the count obtained by `classification == 'good'` alone, i.e. the region requirement removed nothing.

ii.
```python
classification = np.asarray([str(c) for c in units['classification'][:]])
anno = np.asarray([str(a) for a in units['anno_name'][:]])
good = classification == 'good'
if good.sum() == 0:
    return None
```

```python
region_idx = np.full(len(classification), -1, dtype=np.int64)
for i in np.where(good)[0]:
    reg = coarse_region(anno[i])
    if reg is None or not np.isfinite(ml[i]):
        continue
    side = 'left' if ml[i] >= ML_MIDLINE else 'right'   # reference helper_get_neuron_id_area
    region_idx[i] = BRAIN_REGION_INDEX['%s %s' % (side, reg)]
use_unit = good & (region_idx >= 0)
unit_ids = np.where(use_unit)[0]
n_units = len(unit_ids)
if n_units == 0:
    return None
```

iii. CONVERSION_NOTES Step 5 Key Decisions 1 and 2: `classification` "is the output of the region-specific QC classifiers described in the methods and white paper, i.e. exactly the `goodunits` lists the reference code loads", confirmed by reproducing the paper's per-area totals (ALM 7,885 vs 8,717; striatum 7,736 vs 7,664; thalamus 13,021 vs 12,808; midbrain 7,374 vs 7,495; medulla 2,856 vs 2,928). The method paper's 2 Hz firing-rate cut was deliberately not applied because "that is specific to their regression analysis (not a data-quality criterion) so ... dropping low-rate neurons would throw away decodable signal". The region/hemisphere requirement mirrors the reference `helper_get_neuron_id_area`, which intersects the good-unit list with a side mask and a non-empty CCF label.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams share one session-absolute clock, so alignment is done by adding the fixed go-cue-relative bin-edge grid to each trial's `go_start_times` value and binning spikes against those absolute edges. There is no resampling, interpolation or per-stream offset.

ii.
```python
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

```python
go = go_times[trial_idx]
edges = go[:, None] + BIN_EDGES_REL[None, :]           # (n_trials, NBINS+1)
edges_clipped = np.clip(edges, tstart[:, None], tstop[:, None])
...
pos = np.searchsorted(st, flat_edges).reshape(n_trials, NBINS + 1)
```

iii. CONVERSION_NOTES Step 3 "Processing Details": "everything is aligned to the **go cue**. In the reference .mat export spike times were *already* go-cue-aligned; in NWB they are in session time so I subtract `go_start_times[trial]`." The `--show-processing` figures overlay raw spike rasters, the go cue, the tone onset and the trial end on the same axis to demonstrate there is no misalignment.

## 2-e. What is the temporal resolution of the converted data? Is any temporal rebinning applied?

i. 80 non-overlapping 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue, identical for every trial and session. The grid is built once at module level as 81 relative edges and reused everywhere. This *is* a rebinning relative to the source: spike times are raw event times, and the reference paper's own analysis used 40 ms sliding windows with a 3.4 ms stride; the AI replaces that with the task-mandated 50 ms non-overlapping bins. The 300 Hz video is likewise re-binned onto the same grid.

ii.
```python
BIN_SIZE = 0.05            # s, decoder specification
T_START = -2.5             # s relative to go cue, decoder specification
T_STOP = 1.5               # s relative to go cue
NBINS = int(round((T_STOP - T_START) / BIN_SIZE))   # 80
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

```python
'time_bin_size': BIN_SIZE * 1000.0,
'temporal_alignment_event': 'go cue onset (auditory Go cue, BehavioralEvents.go_start_times)',
'off_start': T_START,
'off_end': T_STOP,
'bin_centers': BIN_CENTERS_REL.tolist(),
```

iii. CONVERSION_NOTES Step 3/Step 5: "The decoder task here mandates **50 ms bins**, so I use non-overlapping 50 ms bins (spike counts / 0.05 s = rate in Hz)" and "Window: reference -3 .. +3 s (ephys) and -3 .. +1.5 s (video markers). Task mandates **-2.5 .. +1.5 s**." The AI verified bin centres run -2.475 to +1.475 s and that the bin nearest the go cue is index 49 (centre -0.025 s).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the sample-epoch tone onsets), combined with each trial's go cue. The tone taken for a trial is the **last** sample onset before that trial's go cue, and it must also fall at or after the trial start.

ii.
```python
sample_times = np.asarray(be['sample_start_times'].timestamps[:], dtype=float)
```

```python
pos = np.searchsorted(sample_times, go, side='left') - 1
tone_onset = np.where(pos >= 0, sample_times[np.clip(pos, 0, len(sample_times) - 1)], np.nan)
bad_tone = ~np.isfinite(tone_onset) | (tone_onset < tstart - 1e-9)
```

iii. CONVERSION_NOTES Step 4/Step 5: `sample_start_times` has "**more events than trials** because early-lick trials replay the sample epoch", so the AI takes the last onset preceding the go cue — the tone the animal actually used. It verified that "go - last sample onset = 1.85 s exactly for non-early-lick trials", matching the methods' 0.65 s sample + 1.2 s delay structure.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the go-cue-to-tone gap is added to the fixed bin centres, giving a continuous, signed, per-bin seconds-since-tone value. If no valid tone onset is found the code falls back to `go - 1.85 s` (the canonical trial structure); dataset-wide this fallback was never triggered (0 trials). Stored as `float32`.

ii.
```python
tone_onset = np.where(bad_tone, go - 1.85, tone_onset)   # fallback: standard structure
time_from_tone = (go - tone_onset)[:, None] + BIN_CENTERS_REL[None, :]   # (n_trials, NBINS)
...
inputs = np.stack([time_from_tone.astype(np.float32), photostim], axis=1)  # (n_trials, 2, NBINS)
```

iii. CONVERSION_NOTES Step 7 sanity check: "the bin nearest the go cue is index 49 (centre -0.025 s) and `time_from_tone_onset` there equals **1.825 s = 1.85 - 0.025** for standard trials (larger for early-lick trials with replayed epochs). Correct." Step 9 reports the full-dataset range as [-1.525, 11.894] s, with the long values attributed to "early-lick trials with replayed sample/delay epochs".

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed from `BIN_CENTERS_REL`, the centres of exactly the same go-cue-relative grid used to bin the spikes, so bin *k* of the input covers the same interval as bin *k* of the firing rates by construction. No separate alignment step exists.

ii.
```python
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```
```python
time_from_tone = (go - tone_onset)[:, None] + BIN_CENTERS_REL[None, :]
```
```python
edges = go[:, None] + BIN_EDGES_REL[None, :]
```

iii. Both streams are derived from the same module-level grid anchored on `go_start_times`, so alignment is structural. The AI's `--show-processing` plots draw the tone-onset marker on the same axis as the input trace to confirm visually (CONVERSION_NOTES Step 7 "Processing Plots Review").

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `BehavioralEvents/photostim_start_times` and `photostim_stop_times` (the actual laser on/off event timestamps), with `trials.start_time`/`stop_time` used to attribute each event to a trial. The AI cross-checked these against the trials-table columns `photostim_onset`/`photostim_duration` and found them to agree exactly. Sessions lacking the photostim event series are handled with empty arrays.

ii.
```python
if 'photostim_start_times' in be:
    stim_on = np.asarray(be['photostim_start_times'].timestamps[:], dtype=float)
    stim_off = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=float)
else:
    stim_on = np.zeros(0)
    stim_off = np.zeros(0)
```

iii. CONVERSION_NOTES Step 4 field-mapping table: "`task_stimulation` [power, type, on, off] | `trials.photostim_onset/power/duration` (strings, rel. trial start) and `photostim_start/stop_times` | yes: table onset == event time - trial start (exact)". Having verified the two sources are identical, the AI used the event timestamps because they are already on the session-absolute clock and need no string parsing.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary per-bin time series. Each laser event is attributed to the kept trial whose `[start_time, stop_time]` contains its onset; a bin is set to 1 if the bin interval **overlaps** the laser interval at all (strict edge overlap, not centre containment). Non-stimulated trials stay all-zero. Stored as `float32` alongside the tone input.

ii.
```python
photostim_b = np.zeros((n_trials, NBINS), dtype=bool)
if len(stim_on):
    which = np.searchsorted(tstart, stim_on, side='right') - 1
    for s_on, s_off, w in zip(stim_on, stim_off, which):
        if w < 0 or w >= n_trials:
            continue
        if s_on < tstart[w] or s_on > tstop[w]:
            continue
        ov = (edges[w, :-1] < s_off) & (edges[w, 1:] > s_on)
        photostim_b[w] |= ov
photostim = photostim_b.astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping table: "1 if the stim interval overlaps the bin, else 0", chosen because the decoder spec asks for "whether photostimulation is on at every time point". Step 7 sanity check: "Photostim occupies 10-11 consecutive bins (0.50-0.55 s), matching the 0.5 s laser; onsets fall between -2.28 and -1.17 s relative to the go cue ... Always **before** the go cue, as the methods state." Step 6 records a bug found and fixed here: the accumulator was originally float and `|=` misbehaved, so it was switched to boolean. Full-dataset result: 20.0% of trials stimulated, 2.73% of bins.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The overlap test is performed against `edges[w]`, the same absolute bin edges (`go + BIN_EDGES_REL`) used for the spike binning, so the photostim series is on the neural grid by construction.

ii.
```python
edges = go[:, None] + BIN_EDGES_REL[None, :]           # (n_trials, NBINS+1)
...
ov = (edges[w, :-1] < s_off) & (edges[w, 1:] > s_on)
```

iii. Structural, as for the tone input. Verified in CONVERSION_NOTES Step 10 Check 2: "INPUT 1: photostim == bins overlapping a laser on/off interval, for **every** trial | PASS" and "photostim trials == `trials.photostim_onset != 'N/A'` | PASS (e.g. 136 vs 136)", recomputed from a fresh `pynwb` load with different code. The `--show-processing` figures shade the laser interval under the plotted input trace.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From the raw lick event streams `BehavioralEvents/left_lick_times` and `right_lick_times`, restricted to the answer period. Choice is *not* taken from the trials table; the AI cross-validated its lick-derived choice against `trial_instruction x outcome` and found 99.67% agreement.

ii.
```python
left_licks = np.asarray(be['left_lick_times'].timestamps[:], dtype=float)
right_licks = np.asarray(be['right_lick_times'].timestamps[:], dtype=float)
```

iii. CONVERSION_NOTES Step 5 mapping table and Step 10 Check 2: the lick streams are "the more direct measure of what the animal did and ... what the reference code uses (`behavior_lick_times` / `behavior_lick_directions`)". The residual 0.33% disagreement with `instruction x outcome` is attributed to trials where "the licks recorded in the NWB events differ from the behavioural report (e.g. a single lick that did not trigger the state machine)".

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Per trial, licks are collected in `[go, min(trial_stop, go + 1.5 s)]` — the methods' 1.5 s answer period, also the end of the decoding window. The direction of the **first** lick in that window gives the class: 0 = left, 1 = right, 2 = no lick (default). The per-trial value is broadcast across all 80 bins.

ii.
```python
answer_end = np.minimum(tstop, go + 1.5)
choice = np.full(n_trials, 2, dtype=np.int64)            # 2 = no lick
for i in range(n_trials):
    l = left_licks[(left_licks >= go[i]) & (left_licks <= answer_end[i])]
    r = right_licks[(right_licks >= go[i]) & (right_licks <= answer_end[i])]
    if len(l) == 0 and len(r) == 0:
        continue
    if len(r) == 0:
        choice[i] = 0
    elif len(l) == 0:
        choice[i] = 1
    else:
        choice[i] = 0 if l[0] < r[0] else 1
```
```python
outputs[:, 0, :] = choice[:, None]
```

iii. CONVERSION_NOTES Step 10 Check 2 documents the iteration that produced this window: an initial whole-trial lick search disagreed with the behavioural report on 0.55% of trials, caused by "4 'ignore' trials in which a lick was detected 19-41 ms after the go cue (continuation of ongoing licking) or at 1.75 s (after the answer period), which the behavioural state machine did not count as a response". Restricting to the methods-defined 1.5 s answer period reduced the disagreement to 0.33%. Full-dataset distribution: left 0.431 / right 0.421 / no lick 0.148, which the AI notes is "balanced, as expected for the randomized task".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already holds the strings `'ignore'`, `'miss'` and `'hit'`.

ii.
```python
outcome_str = np.asarray(df['outcome'].values, dtype=object)
```

iii. CONVERSION_NOTES Step 4 field mapping: `trials.outcome` is the NWB equivalent of the reference's `behavior_report` (1 correct / 0 error / -1 no response), with "Same 3 classes, same order as the -1/0/1 coding of the reference". No derivation needed since the three categories the instructions ask for are stored explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped through a fixed dictionary to 0 = ignore, 1 = miss, 2 = hit, subset to the kept trials, and broadcast across all 80 bins into row 1 of the output array.

ii.
```python
omap = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.array([omap[str(o)] for o in outcome_str[trial_idx]], dtype=np.int64)
...
outputs[:, 1, :] = outcome[:, None]
```
```python
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ...
]
```

iii. The code order follows the instructions' "(ignore, miss, hit)". Outcome is a per-trial variable, so it is repeated across bins to keep all four outputs in one `(n_output, n_timepoints)` array. Verified in CONVERSION_NOTES Step 10 Check 2 ("OUTPUT 1: outcome == `trials.outcome` | PASS", "OUTPUT 0/1/2 constant across time within a trial | PASS"); Step 9 reports hit 0.685 / miss 0.167 / ignore 0.148, matching the raw NWB distribution .687/.165/.148.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the trials-table `early_lick` column, which holds the strings `'early'` and `'no early'`.

ii.
```python
early_str = np.asarray(df['early_lick'].values, dtype=object)
```

iii. CONVERSION_NOTES Step 4 field mapping: `trials.early_lick` is the NWB equivalent of the reference's `behavior_early_report`, so no derivation is required. The AI notes the reference *excluded* early-lick trials, but they must be kept here because early lick is a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A string comparison maps `'early'` to 1 and anything else to 0, subset to the kept trials and broadcast across all 80 bins into row 2.

ii.
```python
early = np.array([1 if str(e) == 'early' else 0 for e in early_str[trial_idx]], dtype=np.int64)
...
outputs[:, 2, :] = early[:, None]
```
```python
OUTPUT_VALUES = [..., ['no', 'yes'], ...]
```

iii. The coding follows the instructions ("no, yes"). Per-trial variable, repeated across bins like choice and outcome. Verified by the independent sanity check "OUTPUT 2: early lick == `trials.early_lick` | PASS". Full-dataset rate 11.6%, against 11.4% measured in the raw NWB files (CONVERSION_NOTES Step 9).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, a 300 Hz DeepLabCut series whose `data` is `(n_frames, 3)` = tongue_x, tongue_y, likelihood, with matching `timestamps`. Column 1 gives the y position, column 2 the tracking likelihood.

ii.
```python
ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
vts = np.asarray(ts_obj.timestamps[:], dtype=float)
vdata = np.asarray(ts_obj.data[:], dtype=float)
tongue_y = vdata[:, 1]
visible = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. CONVERSION_NOTES Step 4 field mapping: this is the NWB equivalent of the reference's `tracking.camera_0_side.tongue_x/tongue_y`, confirmed at 300 Hz. It is the only tongue measurement in the file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with DeepLabCut likelihood <= 0.9 are treated as "tongue not visible" and excluded. The remaining y values are averaged within each 50 ms bin, computed for all trials and bins at once with cumulative sums of `y` and of the visibility indicator, indexed by `searchsorted` of the camera timestamps against the (trial-clipped) bin edges. A bin with no visible frame yields NaN.

ii.
```python
LIKELIHOOD_THRESH = 0.9    # DeepLabCut likelihood for "tongue visible"
```
```python
visible = vdata[:, 2] > LIKELIHOOD_THRESH
ysum = np.concatenate([[0.0], np.cumsum(np.where(visible, tongue_y, 0.0))])
vcnt = np.concatenate([[0], np.cumsum(visible.astype(np.int64))])
vidx = np.searchsorted(vts, edges_clipped)               # (n_trials, NBINS+1)
s_y = ysum[vidx[:, 1:]] - ysum[vidx[:, :-1]]
n_y = vcnt[vidx[:, 1:]] - vcnt[vidx[:, :-1]]
with np.errstate(invalid='ignore', divide='ignore'):
    ybin = np.where(n_y > 0, s_y / np.maximum(n_y, 1), np.nan)
```

iii. CONVERSION_NOTES Step 5 Key Decisions 6 and 7: the 0.9 threshold is justified because "the likelihood distribution is strongly bimodal (frac > 0.5 = 0.1059, > 0.9 = 0.1052, > 0.99 = 0.1045 in an example session), so the result is insensitive to the exact threshold". Averaging (rather than the reference's "last frame in the bin") is justified because "With 50 ms bins there are ~15 frames per bin, so I average the visible frames in the bin, which is less noisy and keeps the 'visible' information". The cumsum implementation is noted as a ~50x speed-up over per-bin masks.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes. The 40th and 60th percentiles are computed per session over the *binned* visible y values (the finite entries of `ybin`), then each bin is assigned 0 (< p40), 1 (p40 <= y <= p60), 2 (> p60), or 3 if no visible frame fell in the bin. If a session has fewer than 10 visible bins, all bins get class 3 (no such session occurs).

ii.
```python
vis_vals = ybin[np.isfinite(ybin)]
if vis_vals.size >= 10:
    p40, p60 = np.percentile(vis_vals, [40, 60])
else:
    p40 = p60 = np.nan
tongue_cls = np.full((n_trials, NBINS), 3, dtype=np.int64)    # 3 = not visible
if np.isfinite(p40):
    fin = np.isfinite(ybin)
    tongue_cls[fin & (ybin < p40)] = 0
    tongue_cls[fin & (ybin >= p40) & (ybin <= p60)] = 1
    tongue_cls[fin & (ybin > p60)] = 2
```
```python
OUTPUT_VALUES = [..., ['<40th pct', '40-60th pct', '>60th pct', 'not visible']]
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: "percentiles are computed on the same binned quantity so the class proportions are exactly 40/20/40 among visible bins", and Step 7 confirms "Tongue classes among visible bins are 0.3999 / 0.2002 / 0.3999 - exactly the specified 40/20/40 split". The fourth class is required because the tongue is only out during licking; full-dataset distribution 0.096 / 0.048 / 0.096 / 0.759, which the AI argues "matches the physiology" (Step 12).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as the spikes and go cues, so each bin's frame range is found by `searchsorted` of `vts` against `edges_clipped` — exactly the same go-cue-relative, trial-clipped edges used for the firing rates. Bins falling outside the trial interval collapse to zero frames and therefore become class 3.

ii.
```python
edges = go[:, None] + BIN_EDGES_REL[None, :]
edges_clipped = np.clip(edges, tstart[:, None], tstop[:, None])
...
vidx = np.searchsorted(vts, edges_clipped)               # (n_trials, NBINS+1)
s_y = ysum[vidx[:, 1:]] - ysum[vidx[:, :-1]]
n_y = vcnt[vidx[:, 1:]] - vcnt[vidx[:, :-1]]
```

iii. CONVERSION_NOTES Step 5 Key Decision 5 and Step 10 Check 5: the video is trial-gated, so "Trials begin ~3.2 s before the go cue (so -2.5 s is covered in 96.9% of trials) but end 1.3-1.8 s after it, so +1.5 s is covered in only 84.6% of trials"; the unobserved bins are assigned the explicit "not visible" class rather than imputed. The independent sanity check "OUTPUT 3: tongue class == recomputed (mean y over visible frames vs session percentiles) | PASS (24 random cells)" verifies the alignment, and the summary plot showing tongue visibility rising at t = 0 confirms it qualitatively.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Seven cases, each handled explicitly:
- **Session never quality-controlled** (`classification`/`anno_name` are NaN): `str()` turns them into `'nan'`, no unit matches `'good'`, and the session returns `None` and is skipped (1 session).
- **Trials with no ephys coverage** (8 sessions, 1,060 trials) and **trailing zero-spike trials** (2): dropped.
- **Bins outside the trial interval**: edges are clipped, so firing rate is 0 and tongue class is 3; this is documented in `metadata['partial_observation']` rather than silently emitted.
- **Frames with low DLC likelihood**: excluded from the bin mean; a bin with no visible frame becomes class 3.
- **Sessions with no photostim series**: empty arrays, photostim input all zero (6 sessions).
- **No sample-epoch tone before the go cue**: falls back to the canonical `go - 1.85 s` (never triggered: 0 trials).
- **Units with an unmappable annotation or non-finite CCF coordinate**: excluded (none occur).
- **Per-session and per-worker failures**: `_worker` catches exceptions, prints a traceback and returns an error record rather than aborting the run.

ii.
```python
classification = np.asarray([str(c) for c in units['classification'][:]])
good = classification == 'good'
if good.sum() == 0:
    return None
```
```python
bad_tone = ~np.isfinite(tone_onset) | (tone_onset < tstart - 1e-9)
n_bad_tone = int(bad_tone.sum())
tone_onset = np.where(bad_tone, go - 1.85, tone_onset)   # fallback: standard structure
```
```python
if 'photostim_start_times' in be:
    ...
else:
    stim_on = np.zeros(0)
    stim_off = np.zeros(0)
```
```python
def _worker(args):
    path, make_plots = args
    try:
        res = process_session(path, make_plots=make_plots)
    except Exception as exc:  # noqa
        import traceback
        traceback.print_exc()
        return {'path': path, 'error': repr(exc)}
```

iii. CONVERSION_NOTES Step 10 Check 5 enumerates each edge case and its handling. The governing principle stated there and in Step 5 Key Decision 5 is that where nothing was recorded the trial or session is excluded (so that absent data never appears as genuine 0 Hz activity), except where excluding it would destroy a required output class: "Excluding partially observed trials would delete the entire `miss` output class, so all trials are kept and **unobserved bins are zero-filled**". Where a measurement legitimately has no value (retracted tongue) it becomes an explicit category.

## 10-a. What are the most time-consuming steps of the code?

i. Per-session work is timed in four stages (`setup`, `neural`, `input`, `output`) and recorded in the returned dict. The dominant costs are NWB I/O and the per-unit loops: reading each unit's `spike_times` and `obs_intervals` ragged arrays, the `searchsorted` per unit over all `n_trials x 81` edges, and reading the ~680k x 3 tongue-tracking array. Sessions are processed in parallel, so the full conversion took 23.3 s wall (174 sessions, 24 workers; ~3 s CPU per session), plus 22.1 s to pickle the 11.89 GB result.

ii.
```python
t0 = time.time()
timing = {}
...
timing['setup'] = time.time() - t0
...
timing['neural'] = time.time() - t1
...
timing['input'] = time.time() - t1
...
timing['output'] = time.time() - t1
```
```python
print('[%3d/%3d] %-28s %.1fs elapsed (%.1fs/session, eta %.1f min)' % (...))
```

iii. CONVERSION_NOTES Step 7 "Run Time Estimates" attributes the speed-ups to "One `searchsorted` per unit for all trial x bin edges (~100x vs per-trial loops)", "Cumulative-sum binning of the 300 Hz video (~50x vs per-bin masks)" and "16-way multiprocessing over sessions (~16x)". The estimated full-conversion time (1-2 min) was well inside the instructions' 15-minute budget, and the actual 23 s beat it.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops remain in the conversion path:
1. `for i in np.where(good)[0]` — coarse-region keyword matching, run once per *unit* (~400/session) even though a session has only a few dozen distinct `anno_name` strings; it could run over `np.unique(anno)` and be mapped back.
2. `for ui in unit_ids` in the `obs_intervals` block — one ragged HDF5 read and one `searchsorted` per unit, to compute an intersection that is identical across units on the same probe.
3. `for k, ui in enumerate(unit_ids)` in the spike binning — genuinely hard to remove because spike arrays are ragged, though the per-unit *reads* could be replaced by one read of the flat `units['spike_times'].target.data` buffer.
4. `for i in range(n_trials)` in the choice computation — each iteration masks the session's entire lick arrays, giving O(n_trials x n_licks) work where two `searchsorted` calls would suffice.

The AI's own notes describe only loop 3, and state that it is the *only* Python loop over units.

ii.
```python
for i in np.where(good)[0]:
    reg = coarse_region(anno[i])
```
```python
for ui in unit_ids:
    oi = np.asarray(obs_index[int(ui)], dtype=float)
```
```python
for i in range(n_trials):
    l = left_licks[(left_licks >= go[i]) & (left_licks <= answer_end[i])]
    r = right_licks[(right_licks >= go[i]) & (right_licks <= answer_end[i])]
```

iii. CONVERSION_NOTES Step 6 "Efficiency notes": "Every operation is vectorised over trials; the only python loop over units is the spike-time lookup (unavoidable, ragged storage), which uses one `searchsorted` per unit for all trials at once." The practical justification for not pushing further is that the whole conversion finishes in 23.3 s, far below the instructions' budget — but the stated inventory of loops is incomplete.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened exactly once and each session is converted in a single pass; the bin grid is built once at module level and reused everywhere. Within a session, however, three things are repeated unnecessarily:
- `units['obs_intervals'][ui]` is read and converted for **every** retained unit (hundreds of ragged HDF5 reads per session) even though the intervals are a per-probe property; the observed-trial mask could be built from one read per probe.
- `coarse_region()` re-runs the same long chain of substring tests for every unit sharing an annotation string.
- `spikes_in_trial` adds a second `searchsorted` per unit over the trial bounds, solely to detect the 2 zero-spike trials in the entire dataset.
Across sessions there is no second pass: because the tongue percentiles are per-session, they are computed inside the same pass as everything else.

ii.
```python
obs_index = units['obs_intervals']
for ui in unit_ids:
    oi = np.asarray(obs_index[int(ui)], dtype=float)
    ...
    obs_trial &= m
```
```python
    tb = np.searchsorted(st, trial_bounds).reshape(n_trials, 2)
    spikes_in_trial += tb[:, 1] - tb[:, 0]
```

iii. The AI does not explicitly discuss repeated work. Its stated rationale for reading `obs_intervals` per unit is correctness rather than efficiency: the trial mask "requires every good unit to have an `obs_intervals` entry for the trial" (Step 6, bug fix 2), i.e. it deliberately takes the intersection over units rather than trusting a single unit. The `spikes_in_trial` accumulation is likewise justified by bug fix 3 ("`obs_intervals` may list one final trial in which the recording had already stopped").

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts, all negligible in runtime:
- The per-stage `timing` dict is measured for every session, stored in the result, and then never printed or written into the output — it is dropped at assembly.
- `frac_bins_observed`, `n_units_total`, `n_trials_autowater` and `n_trials_freewater` are computed per session and carried in the result dict but never reach `data` or the printed summary (only `frac_fully_observed` is used).
- The full `observed` boolean array `(n_trials, 80)` is built only to produce those two summary scalars.
- Clipping the spike bin edges to the trial interval (`edges_clipped`) is a no-op for the neural stream: I verified directly from the NWB files that there are no spikes outside trial intervals, so clipped and unclipped edges give identical counts. It is still needed for the tongue stream.
- `plot_session` recomputes rasters, binned means and percentiles for the plotted sessions, but only under `--show-processing`.

ii.
```python
timing['setup'] = time.time() - t0
...
'timing': timing,
'total_time': time.time() - t0,
```
```python
'frac_fully_observed': float(np.mean(observed.all(axis=1))),
'frac_bins_observed': float(np.mean(observed)),
...
'n_trials_autowater': int((auto_water > 0).sum()),
'n_trials_freewater': int((free_water > 0).sum()),
```
```python
edges_clipped = np.clip(edges, tstart[:, None], tstop[:, None])
...
flat_edges = edges_clipped.ravel()
```

iii. The AI did not flag any of these as waste. The instructions asked it to "Print timing information to find bottlenecks", which explains why the `timing` dict exists; it simply never made it into the printed summary. The diagnostic counters exist to support the consistency tables in CONVERSION_NOTES Steps 9-10, and the edge clipping is justified defensively in Step 10 Check 5 ("so spikes/video from the neighbouring trial can never leak in") even though for spikes it changes nothing.
