# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats the dataset as one NWB file per session laid out as `/app/data/sub-<id>/*.nwb`, and enumerates every session with a single sorted glob over that layout (174 files). Each file is opened **directly with `h5py`** rather than with `pynwb`, and every quantity is read from the raw HDF5 paths (`identifier`, `general/subject/subject_id`, `session_start_time`, `units/*`, `intervals/trials/*`, `acquisition/BehavioralEvents/*`, `acquisition/BehavioralTimeSeries/*`, `general/extracellular_ephys/electrodes`). The 174 files are distributed over a `multiprocessing.Pool` of 16 workers, one file per task, and `pool.map` preserves the sorted file order so session order in the output is deterministic. Sessions that return `None` (unusable) are filtered out afterwards. The ragged `units/spike_times` dataset is read once per session as a single flat buffer and sliced with `spike_times_index`, instead of one HDF5 read per unit.

ii.
```python
DATA_DIR = '/app/data'
...
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
if args.sample:
    files = files[:2]
...
if args.nproc > 1 and len(files) > 1:
    with Pool(min(args.nproc, len(files))) as pool:
        results = pool.map(_worker, list(zip(files, plot_flags)), chunksize=1)
else:
    results = [_worker(a) for a in zip(files, plot_flags)]
results = [r for r in results if r is not None]
```

```python
def process_session(path, make_plots=False, plot_dir='/app'):
    """Convert one NWB session. Returns a dict, or None if the session is unusable."""
    with h5py.File(path, 'r') as f:
        sess_id = f['identifier'][()].decode()
        subject = f['general/subject/subject_id'][()].decode()
        session_start = f['session_start_time'][()].decode()
        u = f['units']
        ...
        t = f['intervals/trials']
        ...
        go_all = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

```python
def _spike_slices(spike_times_index):
    """Start/stop indices into the ragged units/spike_times dataset."""
    stop = np.asarray(spike_times_index)
    start = np.concatenate([[0], stop[:-1]])
    return start, stop
```

iii. From CONVERSION_NOTES Step 2 the AI established that the release is "28 subject folders `sub-<id>/`, 174 NWB (HDF5, NWB 2.6) files", verified the internal layout with `h5py.visititems`, and cross-checked the resulting counts (174 files, 28 subjects, 272,227 clusters) against `dandiset.yaml` and the papers. It chose `h5py` over `pynwb` because it only needs a fixed set of datasets and wanted raw-array speed ("Reading `units/spike_times` per unit from HDF5 is slow -> the whole ragged array is read once and sliced with the index vector"). Parallelism over sessions was chosen to satisfy the instruction's 15-minute budget: "16-process pool | ~10x wall clock", giving 0.14 s/session wall clock and 24.2 s for the whole dataset.

## 1-b. How are the data split into subjects (mice)?

i. Each session's animal is read from `general/subject/subject_id` (a numeric string such as `'440956'`). At assembly time the unique ids are sorted to form `subjects`, and `subject_idx` holds each session's index into that list, in session order. This yields 28 subjects with 3-10 sessions each. The AI also noted that this numeric id is not the mouse name used in the papers (`identifier` is e.g. `SC015_20190207_120657_s1`), but used the NWB subject field as-is.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
```

```python
subjects = sorted({r['subject'] for r in results})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_index[r['subject']] for r in results], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 lists `general/subject/subject_id` as the canonical animal id, and the mapping table in Step 5 records "`general/subject/subject_id` -> `subjects`, `subject_idx` | string id per session | 28 mice". The count was used as a sanity check against the papers ("aggregated over 660 penetrations, 173 behavioral sessions, and 28 mice"); Step 9 records 28 = 28, match.

## 1-c. How are the data split into sessions?

i. One NWB file is one session, so no splitting is needed. The session identity is `f['identifier']` (e.g. `SC015_20190207_120657_s1`), and `session_start_time` is also carried through. Session order follows the sorted glob. Each session's record (`session_id`, `subject`, `session_start_time`, `n_neurons`, `n_units_in_file`, `n_trials`, `n_trials_in_file`) is written to `metadata['session_info']`. 173 of the 174 files reach the output; `SC017_20190216_162508_s4` is dropped because it has no QC-good units.

ii.
```python
sess_id = f['identifier'][()].decode()
session_start = f['session_start_time'][()].decode()
```

```python
'session_info': [{'session_id': r['session_id'], 'subject': r['subject'],
                  'session_start_time': r['session_start'],
                  'n_neurons': len(r['region_names']),
                  'n_units_in_file': r['n_units_total'],
                  'n_trials': len(r['neural']),
                  'n_trials_in_file': r['n_trials_file']} for r in results],
```

iii. Step 2 of CONVERSION_NOTES records the one-file-per-session layout. Step 4 explicitly reconciles the file count with the paper: "174 NWB files, **173** with >=1 good unit | 173 behavioral sessions | The extra file (`SC017_20190216_162508_s4`) has 0 good units -> naturally dropped. Consistent." Key decision 9: "All 173 sessions with >=1 good unit are kept... No behavioral-performance session filter is applied: the DANDI release already contains the sessions the authors published, and dropping low-performance sessions would remove error trials needed for the outcome output."

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table `intervals/trials`, one row per behavioural trial, from which `start_time`, `stop_time`, `outcome`, `early_lick`, `trial_instruction`, `auto_water`, `free_water`, `photostim_onset`, `photostim_duration` are read. The trial's temporal anchor is its go cue, taken from `acquisition/BehavioralEvents/go_start_times/timestamps`, which the AI verified has exactly one event per trial; a session whose go-cue count disagrees with its trial count is rejected outright.

ii.
```python
t = f['intervals/trials']
trial_start = t['start_time'][:]
trial_stop = t['stop_time'][:]
outcome = _decode(t['outcome'][:])
early = _decode(t['early_lick'][:])
instruction = _decode(t['trial_instruction'][:])
auto_water = t['auto_water'][:] > 0
free_water = t['free_water'][:] > 0
ntrials_file = len(trial_start)

go_all = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
if len(go_all) != ntrials_file:
    print('%s: %d go cues for %d trials, skipping' % (sess_id, len(go_all), ntrials_file),
          flush=True)
    return None
```

iii. Step 2 documents "`intervals/trials` (one row per trial)" and Step 4 verifies `task_cue_time[0]` (go cue) maps to `go_start_times/timestamps`, "yes, 1 per trial". The AI also noted the contrast with the other epoch events: "`sample_start_times` has MORE entries than trials because the sample epoch is **replayed after early licks**" — so only the go cue is safe to use as a one-per-trial index.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, all motivated by absence of usable data or by reward not being driven by the animal's choice:
1. `auto_water` or `free_water` trials are dropped (the reference's `get_regular_trial_mask` also drops them).
2. Trials with a non-finite go cue are dropped.
3. Trials outside the units' `obs_intervals` are dropped — `observed_trial_mask` keeps only trials observed by *every* selected good unit. This handles the 8 sessions where the ephys recording covers only a contiguous subset of the behavioural session.
4. After binning, trials whose whole extracted window contains zero spikes across all units are dropped (the last trial of a truncated recording).
A session with fewer than 2 surviving trials is dropped. Deliberately **kept**: early-lick trials, no-response (`ignore`) trials, and photostim trials. No behavioural-performance filter and no window-coverage filter is applied. Result: 89,544 of 94,990 trials (94.3%) across 173 sessions.

ii.
```python
# reference `get_regular_trial_mask` also removes auto/free water trials; early
# lick / no-response / photostim trials are decoder targets or inputs and kept
keep = ~(auto_water | free_water)
keep &= np.isfinite(go_all)

# In 8 of the 173 sessions the ephys recording covers only part of the
# behavioural session: the units' obs_intervals then span a contiguous subset
# of trials and the remaining trials contain no spikes at all.  Keep only
# trials that are inside the observation window of every good unit.
keep &= observed_trial_mask(u, unit_ids, trial_start, trial_stop)
trial_idx = np.where(keep)[0]
if len(trial_idx) < 2:
    print('%s: <2 usable trials, skipping' % sess_id, flush=True)
    return None
```

```python
def observed_trial_mask(units, unit_ids, trial_start, trial_stop):
    oi_index = units['obs_intervals_index'][:]
    oi = units['obs_intervals'][:]
    starts = np.concatenate([[0], oi_index[:-1]])
    counts = np.zeros(len(trial_start), dtype=np.int64)
    for k in unit_ids:
        iv = oi[starts[k]:oi_index[k]]
        if len(iv) == 0:
            continue
        idx = np.searchsorted(trial_start, iv[:, 0] + 1e-6) - 1
        idx = idx[(idx >= 0) & (idx < len(trial_start))]
        counts[np.unique(idx)] += 1
    return counts == len(unit_ids)
```

```python
# A handful of trials sit at the very end of a recording: they are listed in
# obs_intervals but the probe was already switched off, so not a single spike
# was recorded in the extracted window.  Such trials carry no neural data.
has_spikes = fr.sum(axis=(0, 2)) > 0
if not has_spikes.all():
    fr = fr[:, has_spikes, :]
    trial_idx = trial_idx[has_spikes]
    ...
if len(trial_idx) < 2:
    print('%s: <2 usable trials after spike check, skipping' % sess_id, flush=True)
    return None
```

iii. Key decision 2: "drop only `auto_water` and `free_water` trials (1,339 + 2,450; reward not driven by the animal's choice, and the reference `get_regular_trial_mask` also drops them). Early-lick, ignore/miss and photostim trials are **kept** because the decoder must predict early lick and outcome and receives photostim as an input; dropping them would make three of the four outputs degenerate." Key decision 3 rejects a coverage filter: "Error (miss) trials are terminated ~0.8 s after the go cue, so requiring the whole [-2.5, +1.5] s window to be observed would delete 95% of miss trials." Filters 3 and 4 were discovered from validator warnings ("all neural data is zero") and are documented in Step 6/Step 10 as issues found and fixed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (session-absolute seconds) with `units/spike_times_index` to slice the ragged buffer per unit, restricted to the QC-good units selected by `units/classification` (+ `units/anno_name`). The go cue times `acquisition/BehavioralEvents/go_start_times/timestamps` supply the per-trial bin edges.

ii.
```python
spikes = u['spike_times'][:]
s0, s1 = _spike_slices(u['spike_times_index'][:])
fr = bin_spikes(spikes, s0, s1, unit_ids, edges_abs)     # (n_units, n_trials, NBINS)
del spikes
```

```python
go = go_all[trial_idx]
edges_abs = go[:, None] + BIN_EDGES[None, :]
```

iii. Step 2 documents "`units` table: `spike_times` (+`spike_times_index`) with **absolute session times**"; Step 4 maps the reference variable `neuron_single_units` (per-trial, already go-cue aligned in the DataJoint export) onto `units/spike_times` + `units/obs_intervals` in NWB, resolving the difference by subtracting the go-cue time per trial. Spike times are the only neural representation in the file, and no dF/F step is needed for ephys (Step 1).

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin divided by the bin width, i.e. firing rate in Hz, stored as `float32`. Per unit, the flattened absolute bin-edge array for all trials is passed to a single `np.searchsorted` against that unit's sorted spike times; differencing adjacent running counts gives the count per bin. No smoothing, no normalisation, no baseline subtraction, no firing-rate threshold on units.

ii.
```python
def bin_spikes(spike_times, starts, stops, unit_ids, edges_abs):
    """Firing rates in Hz.

    Reference `sliding_histogram(..., rate=True)` computes
    (number of spikes in the bin) / bin_width; here the bins are the 50 ms
    non-overlapping bins required by the decoder specification.
    """
    ntrials = edges_abs.shape[0]
    flat = edges_abs.ravel()
    out = np.empty((len(unit_ids), ntrials, NBINS), dtype=np.float32)
    for i, k in enumerate(unit_ids):
        sp = spike_times[starts[k]:stops[k]]
        idx = np.searchsorted(sp, flat).reshape(ntrials, NBINS + 1)
        out[i] = np.diff(idx, axis=1) / BIN_SIZE
    return out
```

iii. Key decision 4: "**Rates in Hz** (spike count / 0.05 s), as in `sliding_histogram(rate=True)`." Step 1 identified the reference function `sliding_histogram(spikeTimes, begin, end, bin_width, stride, rate=True)` as computing "**firing rates (spikes/s)** = count / bin_width", and Step 10 Check 3 records the comparison: "(d) binning | `sliding_histogram`, count/bin width -> Hz | identical formula, 50 ms non-overlapping bins (spec) | same up to the bin size". Step 6 explains the implementation choice: "Binning with nested Python loops (as in the reference `sliding_histogram`) is O(n_bins x n_trials) per unit -> replaced by one `np.searchsorted` per unit over all bin edges (vectorised over trials)." Step 10 Check 2 verified the rates by brute force (`np.sum((sp>=e_b)&(sp<e_{b+1}))/0.05`) for 8 random (trial, neuron) pairs in each of 5 sessions, all passing `np.allclose`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept iff `units/classification == 'good'` **and** `units/anno_name` is non-empty (a CCF histology label present). No thresholds on any individual QC metric, and explicitly **no** firing-rate threshold. Sessions with zero good units are dropped (1 file). Result: 69,453 of 272,227 clusters (25.5%), mean 401.5 per session, range 90-923.

ii.
```python
u = f['units']
classification = u['classification'][:]
anno = _decode(u['anno_name'][:])
good = (classification == b'good')
# reference additionally requires a CCF annotation (histology present)
good &= np.array([a.strip() != '' for a in anno])
unit_ids = np.where(good)[0]
if len(unit_ids) == 0:
    print('%s: no good units, skipping' % sess_id, flush=True)
    return None
```

iii. Key decision 1: "**Neuron curation = `units/classification == 'good'`**: this column is exactly the output of the region-specific QC classifiers described in the white paper and used by the reference code (`qc_mode='classifier'`). No further firing-rate threshold is applied: the method paper's 2-Hz cut was specific to its ridge-regression encoding analysis, would remove 33% of the units, and the decoder benefits from all recorded units." The `anno_name` condition mirrors the reference's requirement that units have "both ephys and histology" (`helper_get_neuron_id_area` "drops units with empty CCF label"); the AI noted in Step 2 that `anno_name` is "empty string exactly for the `unlabelled` units", i.e. the two conditions coincide in NWB. Step 4 reconciles the count with the paper: "69,453 units with `classification=='good'` (25.5% of 272,227 clusters)" vs "69,943 good units, 25.9% of KS2 clusters" — "Agree within 0.7%; the small difference is the published-vs-archived QC list."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset. Spike times and event timestamps share one session-absolute clock, so no resampling or offset correction is required: a fixed grid of 81 edges relative to the go cue is added to each trial's go-cue time to give that trial's absolute bin edges, and spikes are binned against those edges directly.

ii.
```python
OFF_START = -2.5          # s, signed time from go cue to start of the extracted window
OFF_END = 1.5             # s, signed time from go cue to end of the extracted window
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))     # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)  # (81,) relative to go cue
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2.0            # (80,)
```

```python
go = go_all[trial_idx]
edges_abs = go[:, None] + BIN_EDGES[None, :]
centers_abs = go[:, None] + BIN_CENTERS[None, :]
```

```python
'temporal_alignment_event': 'Go cue onset (auditory go cue, end of the delay epoch)',
'off_start': OFF_START,
'off_end': OFF_END,
```

iii. Step 3: "the reference DataJoint export stores spike times **relative to the go cue**; the reference marker alignment script (`align_markers.py`) also aligns video to the **go cue** over [-3, 1.5] s", so go-cue alignment matches both the reference and the decoder spec. Step 4 resolves the format difference: "Spike alignment | spike times already relative to go cue | absolute session times... | Subtract the go cue time per trial; identical result." Step 12 verifies alignment empirically: the processing plots overlay the raw raster and the binned rates on the same axis with the go cue at 0, and the per-time-bin choice decoding rises from chance in the sample epoch to AUC 0.87-0.99 in the response epoch, "strong evidence that the temporal alignment (go cue at bin 50) ... [is] correct."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial spanning -2.5 s to +1.5 s around the go cue, identical for every trial and session. `metadata['time_bin_size'] = 50.0` ms. There is no rebinning of an already-binned stream: spike times are binned once, straight from the raw times, onto this grid. The video (~294 Hz) is averaged directly onto the same 50 ms grid. This deliberately departs from the reference pipeline's 40 ms window / 3.4 ms stride sliding histogram, because the decoder spec requires 50 ms bins.

ii.
```python
BIN_SIZE = 0.05           # s, bin width == stride (non overlapping), decoder spec
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))     # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)  # (81,) relative to go cue
```

```python
'time_bin_size': BIN_SIZE * 1000.0,
...
'n_timepoints': NBINS,
'bin_centers_sec': BIN_CENTERS.tolist(),
'neural_units': 'firing rate (spikes/s), spike count per 50 ms bin / 0.05 s',
```

iii. Step 1: "Our task specifies **50 ms non-overlapping bins**, i.e. bw = stride = 50 ms (documented deviation required by the decoder spec)." Step 4 records it as an accepted discrepancy: "Binning | 40 ms window, 3.4 ms stride | - | same | Decoder spec requires 50 ms bins -> 50 ms non-overlapping bins (rates in Hz, same formula count/bin width)." Step 9's consistency table lists it as "deviation required by the spec".

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the instruction-tone onsets, session-absolute), together with `intervals/trials/start_time` to attribute each tone event to a trial and the trial's go cue to pick the relevant one. The tone used for a trial is the **last** sample-epoch onset within that trial that precedes its go cue. If a trial has no such event, a nominal 1.85 s (0.65 s sample + 1.2 s delay) before the go cue is used as a fallback.

ii.
```python
DEFAULT_SAMPLE_TO_GO = 1.85         # s, 0.65 s sample + 1.2 s delay (fallback only)
```

```python
def tone_onset_times(trial_start, go, sample_start):
    """Onset of the (last) instruction tone of each trial.

    The sample epoch is replayed when the animal licks early, so there can be
    several `sample_start_times` events inside one trial.  The tone that the
    animal finally responded to is the last sample start before the go cue.
    """
    ntrials = len(go)
    tone = np.full(ntrials, np.nan)
    if len(sample_start):
        idx = np.searchsorted(trial_start, sample_start, side='right') - 1
        for i, s in zip(idx, sample_start):
            if 0 <= i < ntrials and s < go[i]:
                if np.isnan(tone[i]) or s > tone[i]:
                    tone[i] = s
    missing = np.isnan(tone)
    if missing.any():                                   # fallback, should be rare
        tone[missing] = go[missing] - DEFAULT_SAMPLE_TO_GO
    return tone
```

```python
sample_start = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:] \
    if 'sample_start_times' in f['acquisition/BehavioralEvents'] else np.array([])
tone = tone_onset_times(trial_start, go_all, sample_start)[trial_idx]
```

iii. Step 2: "`sample_start_times` has MORE entries than trials because the sample epoch is **replayed after early licks**" (405 events vs 368 trials in the first session). Step 10 Check 5: "Early-lick trials replay the sample epoch, giving several `sample_start_times` per trial -> the **last** sample start before the go cue is used as the tone onset; if none exists the nominal 1.85 s is used (this fallback was never triggered)." Step 2's timing table records "go cue - sample (tone) onset | 1.85 s = 0.65 s sample + 1.2 s delay", which is where the fallback constant comes from.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A single subtraction: the absolute time of each bin centre minus the trial's absolute tone-onset time, in seconds, cast to `float32`. It is a continuous, time-varying input, one value per bin, monotonically increasing by 0.05 s across the trial. Observed range over the full dataset: [-1.52, 11.89] s.

ii.
```python
centers_abs = go[:, None] + BIN_CENTERS[None, :]
...
time_from_tone = (centers_abs - tone[:, None]).astype(np.float32)   # (n_trials, NBINS)
```

```python
inputs = [np.stack([time_from_tone[i], photostim[i]]).astype(np.float32)
          for i in range(len(trial_idx))]
```

iii. The Step 5 mapping table specifies "`sample_start_times` (last one before the go cue) -> `input[0]` = 'time from tone onset (s)' | bin-center time minus tone onset time, seconds, float32 | new (spec) | continuous, time-varying". Step 7's plot review confirms the expected shape: "Input 0 is a straight line through the trial, crossing zero ~1.85 s before the go cue (the tone onset)." Step 10 Check 2 re-derived it independently from the NWB file ("bin centre minus the last `sample_start_time` before the go cue") and it passed `np.allclose`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the input is evaluated at the centres of exactly the same go-cue-anchored 50 ms bins that define the firing-rate grid (`centers_abs` is `edges_abs` shifted by half a bin), so bin *k* of the input refers to the same interval as bin *k* of `neural`. Nothing is resampled or interpolated.

ii.
```python
edges_abs = go[:, None] + BIN_EDGES[None, :]
centers_abs = go[:, None] + BIN_CENTERS[None, :]
...
time_from_tone = (centers_abs - tone[:, None]).astype(np.float32)
```

iii. Not separately argued in CONVERSION_NOTES beyond Step 10 Check 3 — "(c) temporal alignment | spikes and video aligned to the go cue | spike times minus `go_start_times`; video binned on the same absolute grid | same" — i.e. every stream, including the inputs, is placed on the one go-cue-relative grid. The `--show-processing` plot of input 0 against time from go cue is cited as visual confirmation of no misalignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Primary source: `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `photostim_stop_times/timestamps`, which are session-absolute laser on/off times (one pair per stimulated trial). Fallback, if those events are absent or empty: the trials table strings `photostim_onset` and `photostim_duration` (onset relative to trial start, `'N/A'` when unstimulated), reconstructed as absolute times via `start_time`.

ii.
```python
def photostim_intervals(f, trial_start, go):
    """(n_stim, 2) array of absolute [on, off] photostimulation times."""
    be = f['acquisition/BehavioralEvents']
    if 'photostim_start_times' in be and len(be['photostim_start_times/timestamps']) > 0:
        on = be['photostim_start_times/timestamps'][:]
        off = be['photostim_stop_times/timestamps'][:]
        n = min(len(on), len(off))
        return np.stack([on[:n], off[:n]], axis=1)
    # fallback: reconstruct from the trials table (onset is relative to trial start)
    t = f['intervals/trials']
    onset = _decode(t['photostim_onset'][:])
    dur = _decode(t['photostim_duration'][:])
    iv = []
    for i in range(len(trial_start)):
        if onset[i] != 'N/A' and dur[i] != 'N/A':
            a = trial_start[i] + float(onset[i])
            iv.append([a, a + float(dur[i])])
    return np.array(iv).reshape(-1, 2)
```

iii. Step 2 documents both sources: "`photostim_start_times` (data = laser power in mW, `control` = stim site code 1/2/4/6), `photostim_stop_times`" and "`photostim_onset` / `photostim_duration` / `photostim_power` (strings, `'N/A'` when no stim, onset given relative to trial start)". Step 4 maps the reference variable "`task_stimulation` [power, type, on, off]" to both. Step 10 Check 5 explains the preference: "`photostim_power`/`onset`/`duration` are strings with `'N/A'` -> the event timestamps are used, with the trial-table strings as a fallback." Step 10 Check 2 verified the resulting number of stimulated trials exactly against `trials/photostim_power != 'N/A'`.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1, stored `float32`) time-varying input: a bin is set to 1 if any stimulation interval **overlaps** the bin, i.e. `bin_end > stim_on and bin_start < stim_off`. All intervals of the session are applied against the full `(n_trials, 80)` edge array, so a stimulation is marked in whatever trial windows it falls into, not only in "its own" trial. Non-stimulated trials stay all-zero. Over the full dataset 20.0% of trials have at least one bin on.

ii.
```python
stim_iv = photostim_intervals(f, trial_start, go_all)
photostim = np.zeros((len(trial_idx), NBINS), dtype=np.float32)
if len(stim_iv):
    for a, b in stim_iv:
        # a bin is 'on' if the stimulation overlaps the bin
        photostim[(edges_abs[:, 1:] > a) & (edges_abs[:, :-1] < b)] = 1.0
```

iii. Step 5 mapping: "`photostim_start_times` / `photostim_stop_times` -> `input[1]` = 'photostimulation on' | 1 if the bin overlaps a stim interval else 0 | `task_stimulation` in reference | binary, time-varying" — matching the instruction "Whether photostimulation is on at every time point (discrete, time-varying)". Step 7's plot review sanity-checks the result against the protocol: "Input 1 is a 0.5 s block ending before the go cue, present on ~20% of trials -> matches the protocol (photoinhibition of the last 0.5 s of the delay epoch on ~25% of trials)", and Step 12 re-checks "the photostim input ends before t = 0, as the protocol requires ('photoinhibition always ended before the Go cue')". Step 9 compares 20.0% against the paper's "~25% randomly interleaved".

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulation on/off times are kept in session-absolute seconds and compared directly against `edges_abs`, the same absolute bin edges used to bin the spikes. There is therefore no separate alignment step, and no per-trial offset arithmetic that could go wrong.

ii.
```python
edges_abs = go[:, None] + BIN_EDGES[None, :]
...
photostim[(edges_abs[:, 1:] > a) & (edges_abs[:, :-1] < b)] = 1.0
```

iii. Same rationale as 3-c: all streams are placed on one absolute clock and cut with the same go-cue-anchored edges (Step 10 Check 3, row "(c) temporal alignment ... same"). The AI's chosen source (absolute event timestamps) requires no conversion at all, which it cites in Step 10 Check 5 as the reason for preferring the events over the trial-table strings.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Lick direction is not stored, so it is derived from two trials-table columns, `outcome` (`'hit'`/`'miss'`/`'ignore'`) and `trial_instruction` (`'left'`/`'right'`): a hit means the animal licked the instructed side, a miss means it licked the opposite side, and `ignore` means it did not lick. The AI cross-validated this derivation against the raw lick streams (`left_lick_times` / `right_lick_times`) and found >99.3% agreement with "first lick after the go cue".

ii.
```python
outcome = _decode(t['outcome'][:])
instruction = _decode(t['trial_instruction'][:])
...
oc = outcome[trial_idx]
instr = instruction[trial_idx]
```

iii. Key decision 5: "**Choice from the trial table** (outcome x instruction) rather than from lick times: it is the same variable the reference code uses (`behavior_report`), is defined for every trial, and agrees with the first-lick-after-go-cue definition on 99.4% of trials." Step 4 maps `behavior_report` (1 correct / 0 error / -1 no response) to `outcome`, and `task_trial_type` ('l'/'r') to `trial_instruction`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Boolean masks over (outcome x instruction) produce an integer code per trial: **0 = no lick, 1 = left, 2 = right**. The per-trial value is broadcast across all 80 bins so that all four outputs live in one `(4, 80)` array per trial (stored `int64`). `output_values[0] = ['no lick', 'left', 'right']` documents the coding. Full-dataset distribution: no lick 14.8%, left 42.9%, right 42.2%.

ii.
```python
OUTPUT_VALUES = [
    ['no lick', 'left', 'right'],
    ...
]
```

```python
# choice: 0 no lick, 1 left, 2 right.  hit -> instructed side, miss -> other side,
# ignore -> no lick (identical definition to reference `behavior_report` x trial type)
choice = np.zeros(len(trial_idx), dtype=np.int64)
licked_left = ((oc == 'hit') & (instr == 'left')) | ((oc == 'miss') & (instr == 'right'))
licked_right = ((oc == 'hit') & (instr == 'right')) | ((oc == 'miss') & (instr == 'left'))
choice[licked_left] = 1
choice[licked_right] = 2
```

```python
for i in range(len(trial_idx)):
    o = np.empty((4, NBINS), dtype=np.int64)
    o[0] = choice[i]
    ...
```

iii. Step 5 mapping: "ignore->'no lick'; hit->instruction; miss->opposite of instruction ... cross-checked against first lick after the go cue (99.4% agreement)". The value ordering is declared in `output_values` rather than following the order in which the instruction lists the classes. Step 10 Check 2 recomputed outputs 0-2 from the raw `outcome`/`trial_instruction`/`early_lick` columns and confirmed they are constant within a trial; Step 12 confirms the class balance is not degenerate ("choice 15/43/42%").

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `intervals/trials/outcome`, which already holds exactly the three strings required (`'ignore'`, `'miss'`, `'hit'`).

ii.
```python
outcome = _decode(t['outcome'][:])
...
oc = outcome[trial_idx]
```

iii. Step 4 verifies the mapping `behavior_report` (reference) -> `intervals/trials/outcome` (NWB) and Step 2 records the value set "{hit, miss, ignore}", so no derivation is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to **0 = ignore, 1 = miss, 2 = hit** with `np.select` (default 0), and the per-trial code is repeated across all 80 bins into row 1 of the output array. `output_values[1] = ['ignore', 'miss', 'hit']`. Full-dataset distribution: ignore 14.8%, miss 16.7%, hit 68.5%.

ii.
```python
outcome_code = np.select([oc == 'ignore', oc == 'miss', oc == 'hit'], [0, 1, 2],
                         default=0).astype(np.int64)
```

```python
o[1] = outcome_code[i]
```

iii. The coding follows the order given in the instructions ("Outcome (ignore, miss, hit, per-trial)"). Step 9 checks the distribution against the raw trial table and the paper: "hit 68.7 / miss 16.5 / ignore 14.8 (all trials) ... | hit 68.5 / miss 16.7 / ignore 14.8 | yes", and separately reports 81.0% correct on control non-early trials against the paper's 84%.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from `intervals/trials/early_lick`, whose values are the strings `'early'` and `'no early'`.

ii.
```python
early = _decode(t['early_lick'][:])
...
el_tr = early[trial_idx]
```

iii. Step 4 verifies the mapping "`behavior_early_report` -> `intervals/trials/early_lick` ('early'/'no early')". The flag is explicit in the file, so nothing is derived.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A boolean comparison to `'early'` cast to int — **0 = no, 1 = yes** — repeated across all 80 bins into row 2. `output_values[2] = ['no', 'yes']`. Full-dataset distribution: no 88.4%, yes 11.6%.

ii.
```python
early_code = (el_tr == 'early').astype(np.int64)
```

```python
o[2] = early_code[i]
```

iii. Coding follows the instruction order ("Early lick (no, yes, per-trial)"). Step 9 compares the converted fraction (11.6%) with the raw data (11.4%): match. The AI notes these trials are kept precisely because early lick is a required decoder target, even though the reference analyses exclude them.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: `data` is `(n_frames, 3)` = (tongue_x, tongue_y, DLC likelihood) with matching absolute `timestamps` at ~294 Hz. Column 1 is the y position; column 2 decides visibility. `Camera3_side_TongueTracking` is used as a fallback if the Camera0 series is missing; if neither exists, the whole session's tongue output becomes "not visible".

ii.
```python
bt = f['acquisition/BehavioralTimeSeries']
key = None
for k in ('Camera0_side_TongueTracking', 'Camera3_side_TongueTracking'):
    if k in bt:
        key = k
        break
ntrials = edges_abs.shape[0]
if key is None:
    return (np.full((ntrials, NBINS), np.nan), np.zeros((ntrials, NBINS), dtype=bool))

data = bt[key]['data'][:]
ts = bt[key]['timestamps'][:]
y = data[:, 1]
lik = data[:, 2]
```

iii. Step 2: "`acquisition/BehavioralTimeSeries/Camera0_side_{Jaw,Nose,Tongue}Tracking`: DeepLabCut markers, `data` = (n_frames, 3) = (x, y, likelihood), `timestamps` = absolute session time, **dt = 0.0034 s (~294 Hz)**. ... All 174 sessions have the side Jaw/Nose/Tongue markers." Step 10 Check 5: "Sessions with a second camera (`Camera3_side_*`) or extra markers -> the side camera used by the paper (`Camera0_side_...`) is preferred, with `Camera3` as a fallback." Step 1 notes the reference's own marker handling (`align_markers.py`) uses the same side-camera tongue markers aligned to the go cue.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames count as visible only if DLC likelihood > 0.9 and y is finite. For each 50 ms bin, the mean y over the visible frames inside that bin is computed; bins containing no visible frame are marked not-visible. The averaging is done without a loop, via cumulative sums of `y` (zeroed on invisible frames) and of the visibility indicator, indexed by `searchsorted` of the camera timestamps at the bin edges.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9   # DLC likelihood; distribution is strongly bimodal
```

```python
visible = (lik > TONGUE_LIKELIHOOD_THRESHOLD) & np.isfinite(y)

# cumulative sums let us average the frames inside every bin without a loop
yv = np.where(visible, y, 0.0)
cs_y = np.concatenate([[0.0], np.cumsum(yv)])
cs_n = np.concatenate([[0], np.cumsum(visible.astype(np.int64))])
idx = np.searchsorted(ts, edges_abs)                  # (ntrials, NBINS+1)
lo, hi = idx[:, :-1], idx[:, 1:]
nvis = cs_n[hi] - cs_n[lo]
sumy = cs_y[hi] - cs_y[lo]
with np.errstate(invalid='ignore', divide='ignore'):
    mean_y = np.where(nvis > 0, sumy / np.maximum(nvis, 1), np.nan)
return mean_y, nvis > 0
```

iii. Key decision 6: "**Tongue visibility threshold 0.9** on the DLC likelihood: the likelihood distribution is strongly bimodal (99.8% of frames are <0.1 or >0.9), so the exact threshold is irrelevant." The averaging choice is justified against the reference in the docstring: "As in the reference alignment script the marker stream is aligned to the go cue; here all frames falling inside a bin are averaged (the reference used the last frame of each 3.4 ms bin, which for a 3.4 ms bin is the same operation)." Step 6 lists the cumsum implementation as a deliberate speedup ("~50x vs per-bin loops"). The trajectory records that the papers specify no DLC likelihood threshold, so the AI set one from the observed bimodality.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles of the binned mean y are computed over **all visible bins of that session's extracted trial windows**, and each bin is assigned: 0 if `mean_y < p40`, 1 if `p40 <= mean_y <= p60`, 2 if `mean_y > p60`, and 3 ("not visible") if the bin has no visible frame. Resulting full-dataset distribution 0.098 / 0.049 / 0.098 / 0.755 — i.e. exactly 40/20/40 among the visible bins, with 75.5% of all bins not visible.

ii.
```python
def discretize_tongue(mean_y, visible):
    """0: < 40th pctile, 1: 40-60th pctile, 2: > 60th pctile, 3: not visible.

    Percentiles are computed per session over all bins in which the tongue is visible.
    """
    cls = np.full(mean_y.shape, 3, dtype=np.int64)
    vals = mean_y[visible]
    if vals.size == 0:
        return cls
    p40, p60 = np.percentile(vals, [40, 60])
    cls[visible & (mean_y < p40)] = 0
    cls[visible & (mean_y >= p40) & (mean_y <= p60)] = 1
    cls[visible & (mean_y > p60)] = 2
    return cls
```

```python
OUTPUT_VALUES = [
    ...
    ['low (<40th pctile)', 'middle (40-60th pctile)', 'high (>60th pctile)', 'not visible'],
]
```

iii. Key decision 7: "**Per-session tongue percentiles** computed over the visible bins of that session's extracted windows, exactly as the spec states ('per-session discretization')." The 40/60 split and the fourth "not visible" class follow the decoder spec directly. Step 10 Check 2 verified the class balance independently: "tongue class balance | 40/20/40 among visible bins | PASS ([0.40, 0.20, 0.40])", and the `--show-processing` plot shows the y histogram with the two percentile cuts drawn on it (Step 7: "the 40th/60th percentile cuts inside the bulk").

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are session-absolute, on the same clock as the spikes, so the per-trial frame ranges are found by `np.searchsorted(ts, edges_abs)` against the very same absolute bin edges used for the firing rates — bin *k* of the tongue output covers exactly the interval of bin *k* of `neural`. No interpolation or offset correction. Where the video is off (inter-trial intervals, and the leading bins of trials whose go cue is less than 2.5 s after trial start) the bins simply contain no frames and become class 3.

ii.
```python
idx = np.searchsorted(ts, edges_abs)                  # (ntrials, NBINS+1)
lo, hi = idx[:, :-1], idx[:, 1:]
nvis = cs_n[hi] - cs_n[lo]
```

```python
tongue_y, tongue_visible = tongue_y_per_bin(f, edges_abs)
tongue_class = discretize_tongue(tongue_y, tongue_visible)
```

iii. Step 10 Check 3: "(c) temporal alignment | spikes and video aligned to the go cue | spike times minus `go_start_times`; video binned on the same absolute grid | same". Step 1 notes the reference `align_markers_between_lims(marker_data, go_times, t_min=-3, t_max=1.5)` also aligns markers to the go cue, so the alignment event matches. Step 2's timing table records that video exists only within `[trial start, trial stop]` and that 3.1% of trials have less than 2.5 s between trial start and the go cue, which the AI accepts as genuinely-unobserved bins rather than filtering those trials (Key decision 3). Step 7's plot review confirms the expected structure: "the tongue class raster shows licking bouts only after the go cue (or before it on early-lick trials)."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Seven distinct cases, each handled explicitly:
- **Session never quality-controlled / no good units** -> session dropped (1 of 174 files; matches the paper's 173 sessions).
- **Go-cue count disagreeing with the trial count** -> session rejected; non-finite go cues -> those trials dropped.
- **Ephys covering only part of the behavioural session** (8 sessions) -> trials outside every good unit's `obs_intervals` dropped.
- **Trial listed in `obs_intervals` but with no spikes at all** -> dropped after binning.
- **Bins with no recorded spikes because the window extends past the trial boundary** -> kept and left at 0 Hz, deliberately not filtered.
- **No visible tongue in a bin** (75.5% of bins), or no tongue camera at all -> explicit fourth class 3, 'not visible', rather than imputation.
- **Missing tone onset** -> nominal 1.85 s before the go cue (never triggered); **missing photostim event stream** -> reconstruct from the trials-table strings; **missing/NaN CCF ML coordinate** -> hemisphere defaults to 'right'.
Individual sessions are also protected by a try/except in the worker, so one bad file cannot abort the run.

ii.
```python
good = (classification == b'good')
good &= np.array([a.strip() != '' for a in anno])
unit_ids = np.where(good)[0]
if len(unit_ids) == 0:
    print('%s: no good units, skipping' % sess_id, flush=True)
    return None
```

```python
keep &= np.isfinite(go_all)
keep &= observed_trial_mask(u, unit_ids, trial_start, trial_stop)
...
has_spikes = fr.sum(axis=(0, 2)) > 0
```

```python
if key is None:
    return (np.full((ntrials, NBINS), np.nan), np.zeros((ntrials, NBINS), dtype=bool))
...
cls = np.full(mean_y.shape, 3, dtype=np.int64)
```

```python
side = 'left' if (not np.isnan(ml[k]) and ml[k] >= ML_MIDLINE) else 'right'
```

```python
def _worker(args):
    path, make_plots = args
    try:
        return process_session(path, make_plots=make_plots)
    except Exception as exc:                                   # keep going on bad files
        ...
        return None
```

iii. Key decision 3 gives the governing principle: where nothing was recorded *for the whole trial* the trial (or session) is excluded, but where only part of the window is unobserved the data is kept, because "requiring the whole [-2.5, +1.5] s window to be observed would delete 95% of miss trials" and make the `outcome` output nearly degenerate; "Worst-case coverage is 66% of the window, mean ~97%". Where the measurement legitimately has no value (retracted tongue) an explicit category is used rather than imputation. Step 10 Check 5 enumerates the edge cases and Step 12 records that all of them were found by following up validator warnings rather than suppressing them.

## 10-a. What are the most time-consuming steps of the code?

i. The AI instrumented `process_session` with a `timing` dict per phase and reported: loading + binning neural data ~0.35 s/session, inputs + outputs (including the video stream) ~0.1 s/session, whole session 0.5 s typical and up to ~2.1 s for the largest sessions. Serially this would be ~60 s + ~20 s; with the 16-worker pool the full 174-session conversion took **24.2 s** (0.14 s/session wall clock), and writing the 11.89 GB pickle took a further **20.8 s** — so pickling is nearly half the total runtime. Within a session the dominant costs are the single bulk read of `units/spike_times` (up to ~11.5 M doubles) plus the ~400 per-unit `searchsorted` calls, and the read of the ~680k x 3 tongue-tracking array.

ii.
```python
t0 = time.time()
spikes = u['spike_times'][:]
s0, s1 = _spike_slices(u['spike_times_index'][:])
fr = bin_spikes(spikes, s0, s1, unit_ids, edges_abs)
...
timing['neural'] = time.time() - t0
...
timing['input'] = time.time() - t0
...
timing['output'] = time.time() - t0
...
'duration': time.time() - t_start,
```

```python
print('Processed %d sessions in %.1f s (%.2f s/session)'
      % (len(results), time.time() - t0, (time.time() - t0) / max(len(results), 1)), flush=True)
...
print('Wrote %s (%.2f GB) in %.1f s'
      % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t0), flush=True)
```

iii. Step 6/Step 7 document the bottleneck analysis and the four speedups with their measured gains: single bulk read of `spike_times` (~20x vs per-unit HDF5 reads), `searchsorted` binning instead of nested loops (~100x vs the reference `sliding_histogram`), cumsum video binning (~50x vs per-bin loops), and the 16-process pool (~10x wall clock). The estimate was "< 2 min with 16 workers, plus ~2-4 min to write the ~12 GB pickle", well inside the instruction's 15-minute budget, so no further optimisation was pursued.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorised the two loops that dominated runtime (spike binning via one `searchsorted` per unit over all trials at once; video bin means via cumulative sums). Several Python-level loops remain, none on the critical path but all vectorisable:
- `observed_trial_mask` loops over **every** good unit (~400/session) and repeats the same `searchsorted` mapping, even though (as the AI itself established) all good units in a session share one `obs_intervals` set — it could read one unit's intervals once, or compute a single `np.isin` on `start_time`.
- `tone_onset_times` uses an explicit `for i, s in zip(idx, sample_start)` scan with a running maximum; this is a `np.maximum.at`/`searchsorted` one-liner.
- The photostim loop rebuilds a full `(n_trials, 80)` boolean mask once per stimulation interval (O(n_stim x n_trials x n_bins)); comparing per-trial onsets against the bin grid once would be O(n_trials x n_bins).
- The per-unit region assignment calls `map_annotation`, which linearly scans ~14 keyword lists (~200 strings) per unit; a dict/`np.unique` lookup over the ~293 distinct annotations would remove the repetition.
- The final per-trial `for i in range(len(trial_idx))` loops that slice `fr` and build the `input`/`output` arrays are required by the target list-of-arrays format, so only the array construction (not the list) could be hoisted.
- `bin_spikes`'s per-unit loop cannot be collapsed further: each unit has a different number of spikes, so there is no single sorted array to search.

ii.
```python
for k in unit_ids:                      # ~400 iterations of identical work
    iv = oi[starts[k]:oi_index[k]]
    idx = np.searchsorted(trial_start, iv[:, 0] + 1e-6) - 1
    idx = idx[(idx >= 0) & (idx < ntrials)]
    counts[np.unique(idx)] += 1
```

```python
for i, s in zip(idx, sample_start):     # scalar Python loop over all tone events
    if 0 <= i < ntrials and s < go[i]:
        if np.isnan(tone[i]) or s > tone[i]:
            tone[i] = s
```

```python
for a, b in stim_iv:                    # full (n_trials, NBINS) mask per interval
    photostim[(edges_abs[:, 1:] > a) & (edges_abs[:, :-1] < b)] = 1.0
```

iii. CONVERSION_NOTES Step 6 documents the loops the AI chose to vectorise and why ("Binning with nested Python loops (as in the reference `sliding_histogram`) is O(n_bins x n_trials) per unit -> replaced by one `np.searchsorted` per unit over all bin edges (vectorised over trials)"; "Averaging the 3.4 ms video frames inside each 50 ms bin -> done with cumulative sums, no loop"). The notes do not discuss the remaining loops; given the measured 0.14 s/session wall clock, they were evidently judged not worth the effort, but the notes never state that explicitly.

## 10-c. What processing does the code repeat multiple times?

i. The main conversion is a single pass: each NWB file is opened once, the spike buffer is read once, and the bin grid is built once at module level and reused for every trial and session. Three repetitions remain:
- `observed_trial_mask` recomputes the identical interval-to-trial mapping once per good unit (~400x per session) even though all good units share the same `obs_intervals`.
- In `--show-processing` mode, `plot_processing` **re-opens the same NWB file** and re-derives the good-unit mask, the `anno_name` decode and the spike-time index that `process_session` had already computed, and recomputes the 40th/60th tongue percentiles that `discretize_tongue` already computed.
- `_decode` is applied to the full `anno_name` column for all ~1,500 units per session, and `map_annotation`'s keyword scan is then run per good unit rather than per distinct annotation string.

ii.
```python
for k in unit_ids:
    iv = oi[starts[k]:oi_index[k]]
    idx = np.searchsorted(trial_start, iv[:, 0] + 1e-6) - 1
```

```python
def plot_processing(res, path, go, tone, edges_abs, tongue_y, tongue_visible, plot_dir):
    ...
    with h5py.File(path, 'r') as f:              # second open of the same file
        u = f['units']
        s0, s1 = _spike_slices(u['spike_times_index'][:])
        classification = u['classification'][:]
        anno = _decode(u['anno_name'][:])
        good = np.where((classification == b'good') & np.array([a.strip() != '' for a in anno]))[0]
    ...
    p40, p60 = np.percentile(vals, [40, 60])     # recomputed for the plot
```

iii. CONVERSION_NOTES Step 6 records the intent to avoid repeated I/O ("single-pass HDF5 reads", "Reading `units/spike_times` per unit from HDF5 is slow -> the whole ragged array is read once and sliced with the index vector", "Avoid unnecessary file I/O"). The re-open in the plotting path is confined to `--show-processing` for at most 2 sessions, and the `observed_trial_mask` redundancy is not mentioned in the notes; the AI's justification for leaving both is implicit in the measured runtime (24 s total), not stated.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Very little, and nothing large:
- `trial_stop` is read and passed into `observed_trial_mask` but never used there — dead argument.
- Firing rates are binned for **all** trials that pass the pre-binning filters, and only then are zero-spike trials discarded, so a small amount of binning work is thrown away (affects the handful of trials at the end of truncated recordings).
- The per-session result dict carries `trial_idx`, `timing`, `n_units_total` and `n_trials_file`; `trial_idx` and `timing` never reach the output pickle (the other two go into `session_info`).
- `ml` (CCF x) is gathered for every unit in the file, but only the good units' values are used.
- `outcome`, `early_lick`, `trial_instruction`, `auto_water`, `free_water` are decoded for all trials and then subset to the kept trials.
- The outputs are stored as `int64` even though they only take the values 0-3; `int8` would use one eighth of the memory (~230 MB -> ~29 MB over the full dataset). Firing rates and inputs are correctly `float32`.
- `metadata['bin_centers_sec']` and the extra `session_info`/`neuron_curation`/`trial_curation` fields are informational and unused by the decoder, but the target format explicitly invites them.

ii.
```python
keep &= observed_trial_mask(u, unit_ids, trial_start, trial_stop)   # trial_stop unused inside
```

```python
fr = bin_spikes(spikes, s0, s1, unit_ids, edges_abs)   # all trials binned ...
...
has_spikes = fr.sum(axis=(0, 2)) > 0                   # ... then some discarded
if not has_spikes.all():
    fr = fr[:, has_spikes, :]
```

```python
o = np.empty((4, NBINS), dtype=np.int64)               # values are only 0..3
```

```python
result = {
    ...
    'trial_idx': trial_idx,      # not propagated to the output pickle
    'timing': timing,            # not propagated to the output pickle
    'duration': time.time() - t_start,
}
```

iii. CONVERSION_NOTES does not discuss discarded work explicitly; the Key Considerations it does address are memory and dtype ("Use appropriate data types (float32 vs. float64)"), which it followed for the neural payload (float32, 11.89 GB rather than 23.8 GB) and the inputs, but not for the outputs. The zero-spike filter is applied after binning because it is defined on the binned rates themselves ("Trials whose extracted window contains no spikes at all are dropped", Step 6 issue 2), which is the reason the ordering was chosen.
