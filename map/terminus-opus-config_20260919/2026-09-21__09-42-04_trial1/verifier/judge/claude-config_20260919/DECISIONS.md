# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is the DANDI:000363 (Mesoscale Activity Map) release, laid out as one NWB/HDF5 file per session
under `/app/data/sub-<subject_id>/`. The AI enumerates every session with a single sorted `glob` over that
layout (174 files) and converts each file exactly once. Rather than using `pynwb`, it opens each file
**directly with `h5py`** and reads the HDF5 groups by path (`intervals/trials`, `acquisition/BehavioralEvents`,
`acquisition/BehavioralTimeSeries/...`, `units`, `general/subject/subject_id`,
`general/extracellular_ephys/electrodes`). Ragged NWB columns (`spike_times`/`spike_times_index`,
`obs_intervals`/`obs_intervals_index`) are de-referenced by hand, and byte/str columns are decoded with a
tolerant `_decode` helper. Sessions are processed in parallel with a `multiprocessing.Pool` (12 workers by
default, 16 for the full run); `--sample` takes the first 2 files.

ii.
```python
DATA_DIR = '/app/data'
...
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
if args.sample:
    files = files[:2]
...
if args.nproc > 1 and not args.show_processing:
    with Pool(args.nproc) as pool:
        results = []
        for i, r in enumerate(pool.imap(_worker, list(zip(files, show)))):
```

```python
def convert_session(filepath, show_processing=False):
    """Convert a single NWB session. Returns a dict or None if the session is dropped."""
    with h5py.File(filepath, 'r') as f:
        subject = _decode(f['general/subject/subject_id'][()])
        identifier = _decode(f['identifier'][()])
        tr = f['intervals/trials']
        ...
        be = f['acquisition/BehavioralEvents']
        go_all = be['go_start_times']['timestamps'][:]
        sample_all = be['sample_start_times']['timestamps'][:]
        ...
        u = f['units']
        spike_times = u['spike_times'][:]
        spike_index = u['spike_times_index'][:]
```

```python
def _decode(x):
    if isinstance(x, bytes):
        return x.decode()
    if isinstance(x, str):
        return x
    return ''
```

iii. From CONVERSION_NOTES Step 2: `/app/data` is "a DANDI-style layout of DANDI:000363 … 29 `sub-*`
directories, but one is `dandiset.yaml`, so 28 subjects; 174 NWB files (~50 GB)", so the directory listing
*is* the complete set of sessions and a glob suffices. Sorting makes session order deterministic. The AI notes
`pynwb` is available but chose raw `h5py` for speed; Step 6 lists "whole-session HDF5 reads" and the
"12-process pool" among its speed-ups, and the full conversion runs in 43 s. Counts were cross-checked against
the papers (174 files → 173 sessions, 28 mice, 272,227 clusters, 94,990 trials).

## 1-b. How are the data split into subjects?

i. Each file's animal is read from `general/subject/subject_id` (a numeric string such as `'440956'`). At
assembly time `subjects` is the sorted set of unique ids across kept sessions and `subject_idx` is each
session's index into that list. This yields 28 subjects with 3–10 sessions each. No renaming to the papers'
mouse names (e.g. `SC015`) is attempted; the numeric NWB id is used directly (the paper name is still
recoverable because `identifier` is stored per session in `metadata['session_info']`).

ii.
```python
subject = _decode(f['general/subject/subject_id'][()])
...
sess = {..., 'subject': subject, 'session_id': identifier, ...}
```

```python
subjects = sorted({s['subject'] for s in sessions})
subject_index = {s: i for i, s in enumerate(subjects)}
data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array([subject_index[s['subject']] for s in sessions]),
```

iii. `subject_id` is the canonical animal identifier inside the file and the containing `sub-*` folder name is
derived from it, so no separate grouping step is needed. CONVERSION_NOTES Steps 2/9 use the resulting count as
a consistency check: 28 mice, matching "660 penetrations, 173 behavioral sessions, and 28 mice" in
`methods.txt`/the data paper.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no splitting or grouping is performed. Each session is labelled by
`nwb.identifier` (e.g. `SC015_20190207_120657_s1`), which encodes mouse, date, time and session number, and is
carried into `metadata['session_info']` together with the file name, subject, trial count, neuron count,
per-session tongue thresholds and per-unit hemisphere. Session order in the output follows the sorted file
list. Sessions with no QC-good unit or fewer than 2 usable trials are dropped (1 session), giving 173.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
identifier = _decode(f['identifier'][()])
...
'session_id': identifier,
'file': os.path.basename(filepath),
```

```python
'session_info': [{'session_id': s['session_id'], 'subject': s['subject'],
                  'file': s['file'], 'n_trials': len(s['neural']),
                  'n_neurons': int(s['neural'][0].shape[0]),
                  'n_units_total': s['n_units_total'],
                  'n_trials_total': s['n_trials_total'],
                  'tongue_pct40_pct60': s['tongue_thresholds'],
                  'hemisphere': list(s['hemisphere'])}
                 for s in sessions],
```

iii. CONVERSION_NOTES Step 2 establishes the one-file-per-session layout, so the file boundary is the session
boundary and nothing has to be inferred. Step 4 records the resolution of the 174-vs-173 discrepancy: "174 NWB
files, one of which (sub-440958_ses-20190216T162508) has 0 good units … Dropping the session with no
QC-passing units gives exactly 173 sessions", matching the papers. Step 5 Key Decision 2 also argues *against*
re-applying the paper's behavioural session criterion (>65% performance, ≥50 correct trials per side) on the
grounds that the DANDI release is already the curated set.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table `intervals/trials`, one row per behavioural trial (`start_time`,
`stop_time`, `outcome`, `early_lick`, `trial_instruction`, `auto_water`, `free_water`, `photostim_*`). Each
trial's go cue is taken from `BehavioralEvents/go_start_times`. The AI verified that there is exactly one go
event per trial in all 174 sessions, so it uses the go array directly when lengths match; it also implements a
fallback that assigns each go event to the trial whose `start_time` precedes it, leaving `NaN` (and therefore a
dropped trial) for any trial without a go cue.

ii.
```python
tr = f['intervals/trials']
start_time = tr['start_time'][:]
stop_time = tr['stop_time'][:]
outcome = np.array([_decode(x) for x in tr['outcome'][:]])
early = np.array([_decode(x) for x in tr['early_lick'][:]])
instruction = np.array([_decode(x) for x in tr['trial_instruction'][:]])
auto_water = tr['auto_water'][:].astype(int)
free_water = tr['free_water'][:].astype(int)
ntrials_all = len(start_time)
```

```python
go_all = be['go_start_times']['timestamps'][:]
if len(go_all) != ntrials_all:
    # assign each go event to its trial; trials without a go cue are dropped
    gi = np.searchsorted(start_time, go_all, side='right') - 1
    go = np.full(ntrials_all, np.nan)
    go[gi] = go_all
else:
    go = go_all
```

iii. CONVERSION_NOTES Step 2: "Exactly one `go_start_times` event per trial in all 174 sessions (verified).
`sample_start_times` can be more numerous than trials because early licks trigger a replay of the epoch." So
the trials table is used directly as the trial definition, and only the go cue (the one event stream that is
1:1 with trials) is used for alignment. Step 10 Check 5 lists "Trials where the go cue is missing – dropped
(none exist, but the code checks `np.isfinite(go)`)" as a defensive edge case.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, all motivated by absence of valid data rather than by behaviour:

1. **`auto_water == 0` and `free_water == 0`** — water delivered irrespective of the animal's choice, following
   the reference `get_regular_trial_mask`.
2. **Finite go cue** — trials with no go event.
3. **Inside `units/obs_intervals`** — spikes are only exported for trials during which the probes were
   recording. The AI intersects the observation intervals of *every* kept unit; in 8 sessions this removes a
   large leading block of behavioural trials that would otherwise be all-zero firing rates.
4. **Non-empty spike data** — after binning, trials with zero spikes summed over all (hundreds of) QC-good
   neurons are dropped (a recording stopped mid-trial).

A session is dropped if fewer than 2 trials survive either before or after step 4. The reference's further
exclusions (early lick, `ignore`, photostim) are **deliberately not applied** because those are required
decoder variables. Result: 89,544 of 94,990 trials kept (94.3%), mean 518 per session.

ii.
```python
oii = u_pre['obs_intervals_index'][:]
oi_starts = np.concatenate([[0], oii[:-1]])
observed = np.ones(ntrials_all, dtype=bool)
for ui in np.where(good_pre)[0]:
    iv = u_pre['obs_intervals'][oi_starts[ui]:oii[ui]]
    if len(iv) == ntrials_all:
        continue
    m = np.zeros(ntrials_all, dtype=bool)
    k = np.searchsorted(start_time, iv[:, 0] + 1e-6) - 1
    k = k[(k >= 0) & (k < ntrials_all)]
    m[k] = True
    observed &= m

# Reference `get_regular_trial_mask` additionally removes early-lick, ignore and
# photostim trials; those are decoder variables here so they are kept.
keep = (auto_water == 0) & (free_water == 0) & np.isfinite(go) & observed
trial_idx = np.where(keep)[0]
if len(trial_idx) < 2:
    return None
```

```python
# Drop trials in which not a single spike was recorded from any of the hundreds of
# QC-passing neurons: the amplifier was not running for that trial ...
nonempty = fr.sum(axis=(0, 2)) > 0
if not np.all(nonempty):
    fr = fr[:, nonempty, :]
    trial_idx = trial_idx[nonempty]
    go_keep = go_keep[nonempty]
if len(trial_idx) < 2:
    return None
```

iii. CONVERSION_NOTES Step 5 Key Decision 3: auto/free-water trials are "trials in which water is delivered
irrespective of the animal's choice, so the behavioural variables do not reflect a decision", and this matches
the reference `get_regular_trial_mask` (`early_lick==0 * auto_water==0 * free_water==0 * correctness!=-1 *
stimulation[:,0]==0`). Early-lick, `ignore` and photostim trials are kept "because the decoder specification
explicitly requires early lick and outcome (which contains `ignore`) as outputs and photostimulation as an
input" — recorded as a *deliberate* difference from the reference in Step 4 and Step 10 Check 3. The
`obs_intervals` and zero-spike filters were discovered from `train_decoder.py` warnings ("all neural data is
zero") and are documented in Step 6 issues 2–3 and Step 10 Check 1/Check 5. Step 9 notes that the resulting
mean trials/session (518) exceeds the paper's 476 precisely because early-lick/no-response trials are retained,
and that the range (159–796) brackets the paper's (130–785).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` together with `units/spike_times_index` (the ragged-array offsets), restricted to units
selected by QC (`units/classification`, `units/anno_name`). The only other input is
`BehavioralEvents/go_start_times`, used to place the bin edges. Region labels come from `units/anno_name` plus
the electrode CCF coordinates (`general/extracellular_ephys/electrodes/x,y,z`) via `units/electrodes`.

ii.
```python
u = f['units']
classification = np.array([_decode(x) for x in u['classification'][:]])
anno = np.array([_decode(x) for x in u['anno_name'][:]])
good = (classification == 'good') & (anno != '')
unit_idx = np.where(good)[0]
...
spike_times = u['spike_times'][:]
spike_index = u['spike_times_index'][:]
...
fr = bin_spikes(spike_times, spike_index, unit_idx, go_keep)   # (nunits, ntr, nbins)
```

iii. CONVERSION_NOTES Step 5 maps "`units/spike_times` (+`spike_times_index`), `classification=='good'`" →
`neural`. Step 2 verified that spike times are in session-clock seconds and "exist **only inside** trial
[start_time, stop_time] intervals". This is the only neural representation in the file.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz with no smoothing, normalisation or baseline
subtraction. For each good unit, the absolute bin edges of every trial are flattened into one monotonic array,
one `np.searchsorted` gives the running spike count at each edge, reshaping to `(n_trials, N_BINS+1)` and
differencing gives the count per bin, and the whole array is divided by the 0.05 s bin width. Output is
`float32`, stored per trial as a C-contiguous `(n_neurons, 80)` array.

ii.
```python
def bin_spikes(spike_times, spike_index, unit_idx, go_times):
    ntrials = len(go_times)
    # absolute bin edges for every trial: (ntrials, N_BINS+1) -> flattened, monotonic
    edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
    out = np.zeros((len(unit_idx), ntrials, N_BINS), dtype=np.float32)
    starts = np.concatenate([[0], spike_index[:-1]])
    for k, u in enumerate(unit_idx):
        st = spike_times[starts[u]:spike_index[u]]
        if st.size == 0:
            continue
        pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
        out[k] = np.diff(pos, axis=1)
    out /= BIN_SIZE       # spike count -> firing rate (Hz), as in sliding_histogram(rate=True)
    return out
```

```python
neural = [np.ascontiguousarray(fr[:, i, :]) for i in range(ntr)]
```

iii. CONVERSION_NOTES Step 1/5: the reference `sliding_histogram(..., rate=True)` returns
`binSpikes / bin_width`, so "I keep the reference convention firing rate = spike count / bin width but use
non-overlapping 50 ms bins (80 bins/trial)" as required by the task. Step 6 records the reshape-before-`diff`
bug that was found and fixed ("`np.diff` over the flattened edges array mixed adjacent trials"), and Step 10
Check 2 re-verified 24/24 random (trial, neuron, bin) firing rates against raw `units/spike_times` with
`np.allclose`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept if `units/classification == 'good'` **and** `units/anno_name` is a non-empty string (i.e. the
unit has a CCF/histology annotation). No thresholds are applied to any of the 15 individual QC metrics
(`amplitude_cutoff`, `presence_ratio`, `isi_violation`, `drift_metric`, `unit_snr`, …), and `unit_quality`
('good'/'multi') is not used. Sessions left with no such unit are dropped. Result: 69,453 of 272,227 clusters
(25.5%), median 390 and mean 401 per session.

ii.
```python
u = f['units']
classification = np.array([_decode(x) for x in u['classification'][:]])
anno = np.array([_decode(x) for x in u['anno_name'][:]])
good = (classification == 'good') & (anno != '')
unit_idx = np.where(good)[0]
if len(unit_idx) == 0:
    return None
```

```python
'neuron_curation': ("units/classification == 'good' (region-specific logistic-regression "
                    'quality-control classifiers of Chen, Liu et al. 2023) and a non-empty '
                    'CCF annotation'),
```

iii. CONVERSION_NOTES Step 1: "Neuron curation in the reference is the classifier-based QC of Chen/Liu et al.
2023. In the DANDI NWB files this is materialised as `units/classification` … So `classification == good`
reproduces the reference `goodunits` files"; and "The reference additionally requires units to have histology
(CCF annotation); in NWB this is a non-empty `anno_name`" (reference `process_one_sess` keeps only units with
both ephys and histology entries). Step 2/4 record that in practice all 69,453 `good` units already have an
annotation, so the second condition is a no-op. The count was validated three ways: 69,453 vs the paper's
69,943 (0.7% low, attributed to the DANDI release differing from the paper snapshot), 25.5% vs 25.9% of
Kilosort2 clusters, and per-region counts within 1–4% of `methods.txt` (thalamus exact at 12,808; medulla
2,925 vs 2,928; midbrain 7,505 vs 7,495; striatum 7,737 vs 7,664; ALM 8,387 vs 8,717).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams share one session-absolute clock, so no resampling or offset correction is needed. The
alignment event is the go cue (`BehavioralEvents/go_start_times/timestamps`, exactly one per trial). The
fixed grid of 81 go-cue-relative bin edges is added to each trial's go-cue time to give that trial's absolute
edge times, and spikes are binned against those edges directly. `metadata['temporal_alignment_event']` records
"go cue onset (auditory go cue ending the delay epoch)", with `off_start = -2.5`, `off_end = 1.5`.

ii.
```python
go_all = be['go_start_times']['timestamps'][:]
...
go_keep = go[trial_idx]
fr = bin_spikes(spike_times, spike_index, unit_idx, go_keep)
```

```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
out[k] = np.diff(pos, axis=1)
```

iii. CONVERSION_NOTES Step 3: "Trials are aligned to the **go cue** (reference preprocessing stores spike
times relative to go cue; marker alignment script uses `go_times` as t = 0)", consistent with the task
instruction. Step 7/Step 12 verify the alignment empirically three independent ways: the population PSTH is
"flat around 5 Hz during sample/delay and rises sharply just after the go cue (8–9 Hz at +0.3 s)"; the tongue
becomes visible only after t = 0; and a per-timepoint choice-decoding AUC jumps from ~0.6 to ~0.98 in the bin
immediately after t = 0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 per trial, spanning −2.5 s to +1.5 s relative to the go cue; every trial and
session has exactly 80 timepoints. The grid is defined once at module level as 81 relative edges plus 80 bin
centres and reused everywhere (spikes, inputs, tongue output). No rebinning or resampling is performed: spike
times are binned straight onto this grid from the raw event times, and the camera frames are assigned to the
same bins. `metadata['time_bin_size'] = 50.0` (ms).

ii.
```python
BIN_SIZE = 0.05           # s, width of the (non-overlapping) time bins
OFF_START = -2.5          # s, start of the trial window relative to the go cue
OFF_END = 1.5             # s, end of the trial window relative to the go cue
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
```

```python
'time_bin_size': BIN_SIZE * 1000.0,
'off_start': OFF_START,
'off_end': OFF_END,
```

iii. The window and bin width are set by the task instructions. CONVERSION_NOTES Step 5 Key Decision 4:
"non-overlapping 50 ms bins (task spec) over [-2.5, +1.5] s = 80 bins; firing rate in Hz = count/0.05, as in
the reference (`rate=True`)". Step 3/4 flag the deviation from the reference's 40 ms width / 3.4 ms stride
sliding window as required by the decoder specification, while keeping the reference's rate convention.
Step 5 Key Decision 5 notes that ~3% of trials start less than 2.5 s before the go cue and ~16% end before
go + 1.5 s, so some edge bins legitimately contain no spikes — the same truncation the reference pipeline has
— and those trials are kept.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times/timestamps` (the onsets of the instruction-tone/sample epoch) together
with `go_start_times`. For each trial the tone is the **last** sample-epoch onset at or before that trial's go
cue. `trials/start_time` is used only to sanity-check the result.

ii.
```python
sample_all = be['sample_start_times']['timestamps'][:]
...
# tone (sample epoch) onset = last sample-epoch start before the go cue.
# Early licks trigger a replay of the sample epoch, so the last one is the tone
# the animal responded to.
j = np.searchsorted(sample_all, go, side='right') - 1
tone = np.where(j >= 0, sample_all[np.clip(j, 0, len(sample_all) - 1)], np.nan)
# fall back to the nominal 1.85 s (0.65 s sample + 1.2 s delay) if missing
bad_tone = ~np.isfinite(tone) | (tone < start_time) | (tone > go)
tone = np.where(bad_tone, go - 1.85, tone)
```

iii. CONVERSION_NOTES Step 2/5: "`sample_start_times` can be more numerous than trials because early licks
trigger a replay of the epoch", so "the *last* onset before the go cue is the tone the animal actually
responded to". Step 10 Check 5 lists this explicitly as an edge case, along with the fallback: if the selected
onset falls outside the trial (missing sample event), the nominal 0.65 s sample + 1.2 s delay = 1.85 s is
used. Step 3 confirms the nominal epoch structure and Step 9 reports the median tone-to-go interval as 1.85 s.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Since the bins sit on a go-cue-relative grid, the value of a bin is its centre plus the tone-to-go gap:
`time_from_tone[trial, bin] = (go − tone) + BIN_CENTERS[bin]`. This is fully vectorised over trials and bins
and stored as `float32` in row 0 of the `(2, 80)` per-trial input array. It is a continuous, signed,
monotonically increasing ramp with 0.05 s steps that crosses zero at the tone onset. Observed range over the
full dataset: [−1.5, 11.9] s (long values come from trials where the sample epoch was replayed several times).

ii.
```python
# 0: signed time from tone (sample epoch) onset, in seconds, per bin
time_from_tone = (go_keep - tone[trial_idx])[:, None] + BIN_CENTERS[None, :]
...
inputs = [np.stack([time_from_tone[i].astype(np.float32), photostim[i]]) for i in range(ntr)]
```

iii. CONVERSION_NOTES Step 5 maps it as "signed seconds from the last sample(tone)-epoch onset preceding the
go cue to each bin centre". Step 7 sanity check: "`input 0` is a set of parallel ramps with 0.05 s steps,
crossing zero at the tone onset (median −1.85 s relative to the go cue)"; Step 10 Check 2 re-derived the full
80-bin ramp from raw `sample_start_times`/`go_start_times` for 12 trials and matched with `atol=1e-4`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: it is built from `BIN_CENTERS`, the centres of the exact same go-cue-relative bin grid
used to bin the spikes, offset by the per-trial tone-to-go gap. Bin *k* of input 0 therefore describes the same
50 ms interval as bin *k* of the firing rates, with no separate alignment step, interpolation or padding.

ii.
```python
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
```
```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()   # neural
...
time_from_tone = (go_keep - tone[trial_idx])[:, None] + BIN_CENTERS[None, :]   # input 0
```

iii. Both streams use `go_keep` (the post-filter go-cue times) and the single module-level grid, so alignment
is guaranteed. The `--show-processing` figure plots input 0 against the population PSTH on a common
time-from-go-cue axis to make the alignment visually checkable (CONVERSION_NOTES Step 6/7).

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The trials-table columns `photostim_onset`, `photostim_duration` and `photostim_power` (all stored as
strings, `'N/A'` when the trial was not stimulated), plus `trials/start_time` and the go cue to put them on the
bin axis. A trial counts as stimulated only if `photostim_power > 0` and both onset and duration parse to
finite floats. (`BehavioralEvents/photostim_start_times`/`photostim_stop_times` were identified in Step 5 as an
alternative source but the trials-table columns are what the code uses.)

ii.
```python
ps_onset = np.array([_tofloat(x) for x in tr['photostim_onset'][:]])
ps_dur = np.array([_tofloat(x) for x in tr['photostim_duration'][:]])
ps_power = np.array([_tofloat(x, 0.0) for x in tr['photostim_power'][:]])
```
```python
has_stim = (ps_power > 0) & np.isfinite(ps_onset) & np.isfinite(ps_dur)
```
```python
def _tofloat(x, default=np.nan):
    try:
        return float(_decode(x))
    except Exception:
        return default
```

iii. CONVERSION_NOTES Step 2 documents the string encoding and the `'N/A'` sentinel; Step 5 maps the
`intervals/trials/photostim_*` columns to `input[1]`, noting the reference pipeline's equivalent `stimulation`
columns (`laser_power`, `laser_on/off` relative to the go cue) and that the reference's regular-trial mask
gates on `stimulation[:, 0] == 0`, i.e. on power. Step 2 counts 18,588 photostim trials in 168 sessions
(19.6%), against the paper's "~25% of trials".

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1, `float32`) time series per trial rather than a per-trial flag. The onset is converted from
trial-relative to go-cue-relative time (`start_time + photostim_onset − go`), the offset is onset + duration,
and every bin that **overlaps** `[s0, s1)` at all is set to 1. Non-stimulated trials are left at 0. The loop
runs per trial. Over the full dataset 20.0% of retained trials have at least one photostim bin.

ii.
```python
photostim = np.zeros((len(trial_idx), N_BINS), dtype=np.float32)
has_stim = (ps_power > 0) & np.isfinite(ps_onset) & np.isfinite(ps_dur)
for k, i in enumerate(trial_idx):
    if not has_stim[i]:
        continue
    s0 = start_time[i] + ps_onset[i] - go[i]     # relative to the go cue
    s1 = s0 + ps_dur[i]
    overlap = (BIN_EDGES[1:] > s0) & (BIN_EDGES[:-1] < s1)
    photostim[k, overlap] = 1.0
```

iii. The instructions require "whether photostimulation is on at every time point (discrete, time-varying)" and
"If an input is a time such as onset of some stimulus, represent it as a binary time series"; CONVERSION_NOTES
Step 5 maps it to "binary per time bin, 1 if the laser was on during the bin" — hence the overlap (rather than
bin-centre) test. Step 7 sanity check: "`input 1` is on only in the 0.5 s window ending at the go cue …, never
after the go cue – matches 'photoinhibition always ended before the Go cue'". Step 10 Check 2 re-derived the
photostim bin mask from the raw trials table for 12 trials, 12/12 matching.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulation interval is re-expressed relative to the go cue (`start_time + onset − go`), the same event
the neural bins are aligned to, and then compared against the shared `BIN_EDGES` grid. So bin *k* of input 1
covers the same interval as bin *k* of the firing rates; no interpolation or offset correction is involved.

ii.
```python
s0 = start_time[i] + ps_onset[i] - go[i]     # relative to the go cue
s1 = s0 + ps_dur[i]
overlap = (BIN_EDGES[1:] > s0) & (BIN_EDGES[:-1] < s1)
```

iii. The onsets are stored relative to trial start, so they must be re-referenced to the go cue before they can
be laid on the bin grid (CONVERSION_NOTES Step 5). The `--show-processing` figure renders input 1 as a
trials × time image with the go cue marked, which makes the timing of the stimulation window directly visible
(Step 7 review).

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the file, so choice is derived from two trials-table columns:
`trial_instruction` (`'left'`/`'right'`, the side the tone instructed) and `outcome`
(`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed side, a miss means it licked the
other side, and `ignore` means it never licked. The actual lick-time streams (`left_lick_times`,
`right_lick_times`) were used only as an independent validation, not in the conversion.

ii.
```python
outcome = np.array([_decode(x) for x in tr['outcome'][:]])
instruction = np.array([_decode(x) for x in tr['trial_instruction'][:]])
...
oc = outcome[trial_idx]
ins = instruction[trial_idx]
# lick direction: hit -> instructed side, miss -> opposite side, ignore -> no lick
licked_left = np.where(oc == 'hit', ins == 'left', ins == 'right')
```

iii. CONVERSION_NOTES Step 5 maps `outcome` + `trial_instruction` → `output[0]` and records that it was
"validated against the actual first lick after the go cue: 100% / 99.8% agreement in two test sessions"; Step
10 Check 2 repeated this on 247 spot-checked trials against `left/right_lick_times`, with 247/247 (100%)
agreement. The reference pipeline's equivalent variables are `trial_type` and `behavior_report`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded 0 = left, 1 = right, 2 = no lick, then broadcast across all 80 bins so that every output shares one
`(4, 80)` `int64` array. `output_values[0] = ['left', 'right', 'no lick']`. Full-dataset distribution:
left 0.429, right 0.422, no lick 0.148.

ii.
```python
OUTPUT_NAMES = ['lick_direction', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [['left', 'right', 'no lick'], ...]
```
```python
licked_left = np.where(oc == 'hit', ins == 'left', ins == 'right')
choice = np.where(oc == 'ignore', 2, np.where(licked_left, 0, 1)).astype(np.int64)
```
```python
outputs = [np.stack([np.full(N_BINS, choice[i]), np.full(N_BINS, outcome_code[i]),
                     np.full(N_BINS, early_code[i]), tongue[i]]).astype(np.int64)
           for i in range(ntr)]
```

iii. Left = 0 / right = 1 follows the instructions' ordering, with a third class for the no-lick case required
by "Lick direction choice (left, right, no lick, per-trial)". CONVERSION_NOTES Step 5 Key Decision 6: "All
outputs are time-varying (shape (4, 80)); the three per-trial variables are held constant across bins, which
the decoder spec prefers" ("If at all possible, make it time-varying"). Step 12 checks that no output is
dominated by a single class.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already holds exactly the three strings the
instructions ask for (`'ignore'`, `'miss'`, `'hit'`). No derivation from lick streams is needed.

ii.
```python
outcome = np.array([_decode(x) for x in tr['outcome'][:]])
...
oc = outcome[trial_idx]
```

iii. CONVERSION_NOTES Step 5 maps `intervals/trials/outcome` → `output[1]` and notes it is the "same three-way
variable" as the reference's `correctness` (1 correct, 0 error, −1 no response). Step 2 counts hit 65,254
(68.7%), miss 15,641 (16.5%), ignore 14,095 (14.8%) in the raw data.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to 0 = ignore, 1 = miss, 2 = hit with nested `np.where`, then repeated across all 80 bins into row 1
of the output array. `output_values[1] = ['ignore', 'miss', 'hit']`. Full-dataset distribution:
ignore 0.148, miss 0.167, hit 0.685 — within 0.2 pp of the raw-data fractions.

ii.
```python
outcome_code = np.where(oc == 'ignore', 0, np.where(oc == 'miss', 1, 2)).astype(np.int64)
```
```python
OUTPUT_VALUES = [..., ['ignore', 'miss', 'hit'], ...]
```
```python
np.full(N_BINS, outcome_code[i])
```

iii. The 0/1/2 assignment follows the order given in the instructions ("Outcome (ignore, miss, hit,
per-trial)"). It is one value per trial, so it is repeated across bins like the other per-trial outputs
(Step 5 Key Decision 6). Step 10 Check 2 verified the codes against the raw trials-table strings; Step 9
compares the converted distribution to the raw NWB distribution (0.687/0.165/0.148 → 0.685/0.167/0.148).

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the trials-table `early_lick` column (strings `'no early'` / `'early'`).

ii.
```python
early = np.array([_decode(x) for x in tr['early_lick'][:]])
...
el_tr = early[trial_idx]
```

iii. CONVERSION_NOTES Step 5 maps `intervals/trials/early_lick` → `output[2]`, corresponding to the reference's
`behavior_early_report`. The flag is stored per trial but the triggering lick happens during the sample/delay
epoch, i.e. inside the −2.5 s window (Step 12 notes that in replayed trials the early lick can fall *before*
the window, which is why this output has the lowest accuracy-to-chance ratio).

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 for `'no early'` and 1 for anything else, then repeated across all 80 bins into row 2.
`output_values[2] = ['no', 'yes']`. Full-dataset distribution: no 0.884, yes 0.116 (raw data: 11.4%).

ii.
```python
early_code = np.array([0 if e == 'no early' else 1 for e in el_tr], dtype=np.int64)
```
```python
OUTPUT_VALUES = [..., ['no', 'yes'], ...]
```
```python
np.full(N_BINS, early_code[i])
```

iii. 0 = no / 1 = yes follows the instructions ("Early lick (no, yes, per-trial)"). CONVERSION_NOTES Step 5
records the mapping as "`no early` → 0, anything containing `early` → 1" — a deliberately permissive rule,
because Step 2 recorded the column as possibly containing compound labels beyond the two canonical strings. The
value is per trial, so it is held constant across bins (Step 5 Key Decision 6).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` =
(`tongue_x`, `tongue_y`, `tongue_likelihood`) with matching `timestamps` at ~300 Hz (dt = 0.0034 s). Column 1
(`tongue_y`) supplies the value and column 2 (the DeepLabCut likelihood) decides visibility. Present in all 174
sessions. The jaw/nose/whisker channels are not used.

ii.
```python
tt_ds = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
vdata = tt_ds['data'][:]
vts = tt_ds['timestamps'][:]
vis = vdata[:, 2] > TONGUE_LIKELIHOOD_THRESH
yv = vdata[:, 1]
```

iii. CONVERSION_NOTES Step 2 documents the channel layout `(nframes, 3) = (x, y, likelihood)` at 300 Hz with
"session-continuous timestamps with gaps between trials (video only recorded during trials)", and Step 4
confirms it against the reference code's `camera_0_side` markers and the method paper. This is the only tongue
measurement in the file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) A frame is "visible" only if its DeepLabCut likelihood exceeds 0.9 — the tracker still
reports a position when the tongue is retracted. (2) The 40th and 60th percentiles of `tongue_y` over **all
visible frames of the session** give two class edges. (3) Per trial, visible frames are assigned to the 50 ms
go-cue-relative bins and the **last visible frame in each bin** supplies the bin's value (the convention of the
reference `align_markers_between_lims`), implemented as a scatter-assignment into a NaN-filled array so that
later frames overwrite earlier ones. Bins with no visible frame keep class 3. The per-session thresholds are
stored in `metadata['session_info'][…]['tongue_pct40_pct60']`.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9   # DeepLabCut likelihood above which the tongue is visible
```
```python
vis = vdata[:, 2] > TONGUE_LIKELIHOOD_THRESH
yv = vdata[:, 1]
if np.any(vis):
    thr_lo, thr_hi = np.percentile(yv[vis], [40, 60])
else:
    thr_lo, thr_hi = np.nan, np.nan
tongue = tongue_class_per_bin(vts, yv, vis, go_keep, thr_lo, thr_hi)
```
```python
    for i in range(ntrials):
        a, b = lo[i], hi[i]
        if b <= a:
            continue
        v = vis[a:b]
        if not np.any(v):
            continue
        tt = ts[a:b][v]
        yy = y[a:b][v]
        idx = np.floor((tt - t0[i]) / BIN_SIZE).astype(int)
        keep = (idx >= 0) & (idx < N_BINS)
        idx, yy = idx[keep], yy[keep]
        ...
        ybin = np.full(N_BINS, np.nan)
        ybin[idx] = yy               # frames are ordered, so the last one in a bin wins
```

iii. CONVERSION_NOTES Step 5 maps the series to `output[3]` as "per 50 ms bin: if any frame in the bin has
DeepLabCut likelihood > 0.9 use the **last visible frame's** y, else class 3 (not visible)", citing
`align_markers_between_lims` ("Use the embedding at the last time point within the time range") as the
reference convention. The threshold is justified as non-critical because "likelihood is strongly bimodal
(<0.5% of frames between 0.05 and 0.9)"; Step 2 records that only ~10–16% of frames per session are visible.
Step 3 notes the method paper instead mean-imputes occluded tongue positions, which is incompatible with the
instruction's explicit fourth "not visible" class.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, exactly as specified: 0 if `y < thr_lo` (below the session's 40th percentile), 2 if
`y > thr_hi` (above the 60th), 1 in between, and 3 if no visible frame fell in the bin. The percentiles are
per session, computed over the visible frames of the whole session (not per trial and not pooled across
sessions). `output_values[3] = ['<40th pct', '40-60th pct', '>60th pct', 'not visible']`. Resulting
distribution over all bins: 0.144 / 0.033 / 0.068 / 0.755.

ii.
```python
    cls = np.full((ntrials, N_BINS), 3, dtype=np.int64)
    ...
        good = ~np.isnan(ybin)
        c = np.where(ybin[good] < thr_lo, 0, np.where(ybin[good] > thr_hi, 2, 1))
        cls[i, good] = c
```
```python
thr_lo, thr_hi = np.percentile(yv[vis], [40, 60])
```
```python
OUTPUT_VALUES = [..., ['<40th pct', '40-60th pct', '>60th pct', 'not visible']]
```

iii. The 40/60 split and the per-session scope follow the instructions verbatim ("< 40th percentile of
y-position over the session"), and class 3 is the instruction's "not visible". CONVERSION_NOTES Step 7
verifies the discretisation visually: "The tongue overlay figure shows the discrete class following the raw y
trace across the 40th/60th percentile lines"; Step 10 Check 2 recomputed the session thresholds from the raw
video (matching to 1e-9) and 12/12 random (trial, bin) classes. Step 7/9 note that class 3 dominates
(75.5% of bins) because "the tongue is only outside the mouth while licking", dropping from 95–99% before the
go cue to 9–27% after it.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and events, so each trial's frame range
is found with `np.searchsorted` on the timestamps at `go + OFF_START` and `go + OFF_END`, and frames are
binned by their offset from `go + OFF_START` — the same 50 ms go-cue-relative grid used for the firing rates.
Out-of-range bin indices are masked out. Because the video is trial-gated, trials whose window extends outside
the trial interval simply have no frames in the leading/trailing bins, which fall into class 3.

ii.
```python
    t0 = go_times + OFF_START
    t1 = go_times + OFF_END
    lo = np.searchsorted(ts, t0, side='left')
    hi = np.searchsorted(ts, t1, side='left')
    for i in range(ntrials):
        ...
        idx = np.floor((tt - t0[i]) / BIN_SIZE).astype(int)
        keep = (idx >= 0) & (idx < N_BINS)
```
```python
tongue = tongue_class_per_bin(vts, yv, vis, go_keep, thr_lo, thr_hi)
```

iii. Sharing the global clock means no interpolation or offset correction is needed, and using the identical
bin grid guarantees bin *k* of the tongue output covers the same interval as bin *k* of the firing rates
(CONVERSION_NOTES Step 5, Step 10 Check 3(c)). The alignment is checked by the fact that class 3 collapses
exactly at t = 0 ("covers 95–99% of bins before the go cue and drops to 9–27% after it", Step 7) and by the
per-trial tongue overlay figure that plots the raw y trace, the percentile lines and the resulting class on a
common time-from-go-cue axis. Note that `go_keep` — the filtered go-cue array — is passed, so the tongue rows
correspond to the same trials as the neural rows.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Seven cases, all documented in Step 10 Check 5:

- **`anno_name`/`classification` stored as a float (NaN) instead of a string** (one session was never
  QC-labelled): `_decode` returns `''`, the unit fails the `good` test, the session ends up with zero units and
  is dropped → exactly 173 sessions.
- **`photostim_*` strings equal to `'N/A'`**: `_tofloat` returns NaN (or 0.0 for power), so the trial is simply
  not stimulated.
- **Trials outside the ephys recording** (`units/obs_intervals`, 8 sessions): dropped.
- **Trial with no spikes at all** (recording stopped mid-trial): dropped via the `nonempty` mask.
- **Missing go cue**: `np.isfinite(go)` drops the trial (none occur).
- **Missing/implausible sample-epoch onset**: falls back to the nominal `go − 1.85 s`.
- **Bins with no visible tongue frame** (including the ~3%/~16% of trials whose window extends beyond the
  trial and therefore has no video): assigned the explicit class 3 rather than imputed.
- **Sessions with < 2 usable trials**: dropped (required by the target format).

Truncated windows for the neural data are *not* treated as missing: the affected bins legitimately contain no
spikes and the trials are kept.

ii.
```python
def _decode(x):
    if isinstance(x, bytes):
        return x.decode()
    if isinstance(x, str):
        return x
    return ''


def _tofloat(x, default=np.nan):
    try:
        return float(_decode(x))
    except Exception:
        return default
```
```python
good = (classification == 'good') & (anno != '')
unit_idx = np.where(good)[0]
if len(unit_idx) == 0:
    return None
```
```python
bad_tone = ~np.isfinite(tone) | (tone < start_time) | (tone > go)
tone = np.where(bad_tone, go - 1.85, tone)
```
```python
nonempty = fr.sum(axis=(0, 2)) > 0
...
if len(trial_idx) < 2:
    return None
```
```python
    cls = np.full((ntrials, N_BINS), 3, dtype=np.int64)   # default: not visible
```

iii. The AI's stated principle (CONVERSION_NOTES Steps 5, 6, 10) is to **exclude** data where nothing was
recorded — "they would otherwise appear as zero firing rates", which the decoder's validator flagged as "all
neural data is zero" warnings — and to **represent as an explicit category** measurements that legitimately
have no value, i.e. a retracted tongue. Truncated windows are kept because "This identical truncation exists in
the reference pipeline (per-trial spike times, window [-3, +3] s)". Step 10 Check 1 reports that after these
fixes `verification_full_out.txt` contains "no errors and no warnings".

## 10-a. What are the most time-consuming steps of the code?

i. The code instruments itself: `convert_session` records `timings` for `read_spikes`, `bin_spikes` and
`tongue`, and `main` prints per-session and cumulative elapsed time. Measured per session: reading
`units/spike_times` 0.05–0.14 s, binning 0.08–0.22 s, tongue discretisation 0.06–0.08 s, whole session
0.3–0.6 s (larger sessions more). With a 12–16-process pool the full 174-session conversion takes **43 s**, so
the dominant single cost is writing the 11.89 GB pickle at the end, followed by HDF5 I/O and the per-unit
`searchsorted` loop.

ii.
```python
        t0 = time.time()
        spike_times = u['spike_times'][:]
        spike_index = u['spike_times_index'][:]
        timings['read_spikes'] = time.time() - t0
        ...
        timings['bin_spikes'] = time.time() - t0
        ...
        timings['tongue'] = time.time() - t0
```
```python
    t0 = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print(f'Wrote {args.outfile} ({os.path.getsize(args.outfile)/1e9:.2f} GB) in {time.time()-t0:.0f}s')
```

iii. CONVERSION_NOTES Step 7 gives the timing table and the extrapolation ("~2–4 min with 12 processes
(dominated by pickling ~12 GB)"), well inside the instructions' 15-minute budget, so no further optimisation
was pursued. The remaining costs are intrinsic: one whole-session read of the ragged spike buffer (up to ~11.5 M
doubles) and of the ~680 k × 3 video array, plus one binary search per bin edge per unit.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Five loops remain, in decreasing importance:

1. **Per-unit loop in `bin_spikes`** — one `searchsorted` per unit, but already vectorised across *all* trials
   and bins by flattening the edge array. It cannot be collapsed further because `spike_times` is ragged.
2. **Per-trial loop in `tongue_class_per_bin`** — could be replaced by a single global bin index plus one
   `np.bincount`/scatter over all frames of the session.
3. **Per-trial photostim loop** — trivially vectorisable: the onset/offset arrays are already computed for all
   trials, so the mask could be built as one `(n_trials, N_BINS)` broadcast comparison (which is what the
   reference does).
4. **Per-unit `obs_intervals` loop** — reads `obs_intervals` once per good unit (up to ~900 HDF5 slice reads
   per session); the whole ragged buffer could be read once and sliced in memory, and the `continue`
   short-circuit happens *after* the read.
5. **Python-level comprehensions** — `[_decode(x) for x in ...]` over every string column, the
   `[classify_region(anno[i], ap[i]) for i in unit_idx]` region loop, `early_code`'s list comprehension, and
   the three final `[... for i in range(ntr)]` comprehensions that materialise the per-trial lists.

None is a measurable share of runtime at the achieved 43 s total, and the process pool hides them.

ii.
```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
for k, u in enumerate(unit_idx):
    st = spike_times[starts[u]:spike_index[u]]
    pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
    out[k] = np.diff(pos, axis=1)
```
```python
for k, i in enumerate(trial_idx):
    if not has_stim[i]:
        continue
    s0 = start_time[i] + ps_onset[i] - go[i]
    s1 = s0 + ps_dur[i]
    overlap = (BIN_EDGES[1:] > s0) & (BIN_EDGES[:-1] < s1)
    photostim[k, overlap] = 1.0
```
```python
for ui in np.where(good_pre)[0]:
    iv = u_pre['obs_intervals'][oi_starts[ui]:oii[ui]]
    if len(iv) == ntrials_all:
        continue
```

iii. CONVERSION_NOTES Step 6 identifies "Naive per-trial/per-unit loops over spike times (as in the reference
`sliding_histogram`, which is O(units × trials × bins))" as the inefficiency to attack, and reports the fix:
"Single `searchsorted` per unit over the flattened bin-edge array (vectorised over trials and bins)", claimed
at "~50× vs per-trial loops", plus whole-session HDF5 reads (~3×) and the 12-process pool (~10×). The notes do
**not** mention the photostim, `obs_intervals` or `_decode` loops as remaining vectorisation opportunities.

## 10-c. What processing does the code repeat multiple times?

i. Two genuine repetitions, both cheap:

1. **`units/classification` and `units/anno_name` are read and decoded twice per session** — once as
   `classification_pre`/`anno_pre` (to build `good_pre` for the `obs_intervals` filter) and again as
   `classification`/`anno` (to build `good`), with `f['units']` also re-opened as `u_pre` and then `u`. The two
   masks are identical by construction.
2. **`obs_intervals` is read per good unit** rather than once for the session (see 10-b), and the AI's own
   Step 2 finding that "obs_intervals = the trial intervals" for all units makes most of those reads redundant.

Everything else is computed once: the bin grid (`BIN_EDGES`, `BIN_CENTERS`) is built once at module level and
reused for spikes, both inputs and the tongue output; each NWB file is opened exactly once inside a single
`with` block; the per-session tongue percentiles are computed in the same pass, so no second pass over the data
is needed.

ii.
```python
        u_pre = f['units']
        classification_pre = np.array([_decode(x) for x in u_pre['classification'][:]])
        anno_pre = np.array([_decode(x) for x in u_pre['anno_name'][:]])
        good_pre = (classification_pre == 'good') & (anno_pre != '')
```
```python
        # ---------------- units ---------------------------------------------------
        u = f['units']
        classification = np.array([_decode(x) for x in u['classification'][:]])
        anno = np.array([_decode(x) for x in u['anno_name'][:]])
        good = (classification == 'good') & (anno != '')
```

iii. The duplication is a structural consequence of ordering the `obs_intervals` trial filter *before* the unit
block (so that trials are curated before the expensive binning), and CONVERSION_NOTES does not flag it. Its
cost is negligible (~1 ms per session for the two string columns, ~40 ms for the `obs_intervals` reads), which
is consistent with the notes' position that the conversion is already comfortably inside budget.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts:

- **`trials/stop_time` is read and never used.** Likewise the electrode DV coordinate `ey`/`uy` is read and
  never used (only `ux` for hemisphere and `uz` for AP), and `tongue_x` (column 0 of the video array) is loaded
  but unused.
- **The 150-line `classify_region` keyword taxonomy and the ALM AP-boundary computation** produce
  `brain_region_idx`, which the decoder only validates and prints — it is not an input to the model. The same
  is true of the per-unit **`hemisphere`** array, which is computed for every unit and stored in
  `metadata['session_info']` but is not part of the target format at all.
- **Assorted metadata**: `n_units_total`, `n_trials_total`, `trial_idx`, `tongue_thresholds`, `timings`,
  `elapsed` are carried per session; only some reach the output dictionary.
- **`--show-processing` plotting** (two figures per session) is optional and off for the full run.

Nothing that enters the neural/input/output arrays is wasted, and the per-trial lists are materialised exactly
once.

ii.
```python
        stop_time = tr['stop_time'][:]          # never used
```
```python
        ex, ey, ez = el['x'][:], el['y'][:], el['z'][:]
        ux, uy, uz = ex[eidx], ey[eidx], ez[eidx]   # uy unused
        ap = CCF_AP_BREGMA - uz
        hemi = np.where(ux >= CCF_ML_MIDLINE, 'left', 'right')
        regions = np.array([classify_region(anno[i], ap[i]) for i in unit_idx])
```
```python
BRAIN_REGIONS = ['ALM', 'Orbital', 'OtherCortex', 'Striatum', 'Pallidum', 'Thalamus',
                 'Hypothalamus', 'Midbrain', 'Pons', 'Medulla', 'Cerebellum',
                 'Hippocampus', 'Olfactory', 'CorticalSubplate', 'Unknown']
```

iii. The region taxonomy is not framed as waste in CONVERSION_NOTES — it is deliberate, so that per-region unit
counts can be compared against `methods.txt` as a consistency check on the unit selection (Step 9: thalamus
exact, medulla/midbrain/striatum within 1%, ALM 96%), and Step 10 Check 4 explicitly notes "The residual
difference only affects the `brain_region_idx` labels, not the neural data". The hemisphere split is retained
because the reference `helper_get_neuron_id_area` splits units by the CCF ML midline at 5,700 µm. `stop_time`
and `uy` are simply dead reads. Note also that `BRAIN_REGIONS` is a fixed 15-element list including
`'Unknown'`, which ends up with 0 neurons in the full dataset — a declared-but-empty category.
