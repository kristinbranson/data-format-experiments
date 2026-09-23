# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI:000363) is one NWB file per session laid out as
`/app/data/sub-<subject_id>/sub-<subject_id>_ses-<timestamp>_behavior+ecephys+ogen.nwb`. The AI
finds every session with a single sorted `glob` over that layout (174 files), and converts them
with a `ProcessPoolExecutor` (default 24 workers, one file per task). Each file is opened once
with `pynwb.NWBHDF5IO(path, 'r', load_namespaces=True)` inside a `with` block, and
`read_session(nwb)` pulls everything the conversion needs out of the open `NWBFile` in one place:
the trials table (`nwb.intervals['trials']`), the behavioural event timestamps
(`nwb.acquisition['BehavioralEvents']`), the DeepLabCut tracking
(`nwb.acquisition['BehavioralTimeSeries']`), the unit table (`nwb.units`), the electrode CCF
coordinates (`nwb.electrodes`) and the subject/identifier metadata. No `h5py` is used anywhere.
A worker-level `try/except` keeps a single bad file from killing the run. In `--sample` mode
only the first 8 files are globbed and the first 2 curated sessions are kept.

ii.
```python
DATA_DIR = '/app/data'
...
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
print(f'found {len(files)} NWB files in {DATA_DIR}')
...
if args.workers > 1 and len(jobs) > 1:
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(_worker, jobs):
            results.append(res)
            _report(res)
```

```python
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
    raw = read_session(nwb)
```

```python
def read_session(nwb):
    trials = nwb.intervals['trials']
    events = nwb.acquisition['BehavioralEvents'].time_series
    tracking = nwb.acquisition['BehavioralTimeSeries'].time_series
    out = {
        'identifier': nwb.identifier,
        'mouse': nwb.subject.description,
        'subject_id': str(nwb.subject.subject_id),
        ...
        'go': np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64),
        ...
    }
```

```python
def _worker(args):
    path, collect_debug = args
    try:
        return convert_session(path, collect_debug=collect_debug)
    except Exception as exc:  # keep one bad file from killing the whole run
        ...
```

iii. From CONVERSION_NOTES.md Step 2/Step 5 Key Decision 1: "one NWB file per recording session,
174 files, 28 subjects, 50 GB. All files read with `pynwb.NWBHDF5IO` (lazy/HDF5-backed; no `h5py`
used directly)" and "`pynwb.NWBHDF5IO` only. Trials, events, units and tracking are read through
the `pynwb`/`hdmf` API … No direct `h5py` access." Because the release stores one session per
file, the directory listing is the complete set of sessions. Step 6 documents the parallelism as
a deliberate speed-up: "sessions are converted in 24 worker processes (`ProcessPoolExecutor`)",
giving 174 files in 26 s wall-clock vs ~2.5 min serial.

## 1-b. How are the data split into subjects?

i. Each session's animal is taken from `nwb.subject.description`, which holds the mouse name used
in the papers (e.g. `SC015`); the numeric `nwb.subject.subject_id` (e.g. `440956`) is also read but
only stored in `metadata['session_info']`. At assembly, `subjects` is the sorted set of unique
mouse names and `subject_idx` is each session's index into that list. The AI verified that
`subject_id` ↔ `subject.description` is 1:1 (28 ↔ 28), so the two labellings give the same
partition. All 28 mice survive curation.

ii.
```python
'mouse': nwb.subject.description,
'subject_id': str(nwb.subject.subject_id),
```

```python
mice = sorted({r['mouse'] for r in sessions})
data = {
    ...
    'subjects': mice,
    'subject_idx': np.array([mice.index(r['mouse']) for r in sessions], dtype=np.int64),
```

iii. Step 5 variable-mapping table: "`nwb.subject.description` (e.g. `SC015`) → `subjects` /
`subject_idx` … unique mouse names … mouse name in the `.mat` filename". The mouse name is chosen
because it is the identifier the reference code and the papers use (the reference `.mat` export is
named by mouse), making the converted data directly comparable to the papers. Step 10 Check 5
records the edge case "Sessions with identical mouse but different subject_id: none —
`subject_id` ↔ `subject.description` is 1:1 (28 ↔ 28)."

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no grouping or splitting is needed. Sessions are identified by
`nwb.identifier` (e.g. `SC015_20190207_120657_s1`) and are ordered in the output by sorting the
per-session results on that identifier (i.e. by mouse, then date). Beyond the file split, the AI
applies **session-level curation**, which is the largest divergence from the reference: a session
is rejected if it has no `classification == 'good'` unit, if fewer than 2 trials survive trial
curation, if the control non-early-lick performance is ≤ 0.65, or if it has fewer than 50 correct
lick-left or fewer than 50 correct lick-right control trials. This drops **32 of the 174 sessions**
(1 for no good units, 24 for performance, 7 for the ≥50-correct rule), leaving **142 sessions**
(73,845 trials, 57,023 neurons). Full per-session provenance is written to
`metadata['session_info']`.

ii.
```python
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50
```

```python
def session_performance(raw, keep):
    control = (raw['photostim_onset_str'] == 'N/A') & (raw['early_lick'] == 'no early') \
        & keep
    hit = raw['outcome'] == 'hit'
    miss = raw['outcome'] == 'miss'
    n_hit = int(np.sum(hit & control))
    n_miss = int(np.sum(miss & control))
    performance = n_hit / (n_hit + n_miss) if (n_hit + n_miss) else 0.0
    n_left = int(np.sum(hit & control & (raw['instruction'] == 'left')))
    n_right = int(np.sum(hit & control & (raw['instruction'] == 'right')))
    return performance, n_left, n_right
```

```python
performance, n_left, n_right = session_performance(raw, keep)
reject = None
if n_good == 0:
    reject = 'no good units'
elif len(trial_idx) < 2:
    reject = f'only {len(trial_idx)} usable trials'
elif performance <= MIN_PERFORMANCE:
    reject = f'performance {performance:.3f} <= {MIN_PERFORMANCE}'
elif n_left < MIN_CORRECT_PER_DIRECTION or n_right < MIN_CORRECT_PER_DIRECTION:
    reject = f'correct trials L={n_left} R={n_right} < {MIN_CORRECT_PER_DIRECTION}'
if reject is not None:
    return {'identifier': raw['identifier'], 'path': path, 'rejected': reject, ...}
```

```python
sessions = sorted(sessions, key=lambda r: r['identifier'])
```

iii. Step 3 quotes the data paper's STAR Methods verbatim: "We selected experimental sessions for
analysis based on following criteria: overall behavioral performance (> 65%), and at least 50
correct lick left and lick right trials each", with "Overall performance … computed as the
fraction of correct control trials (i.e. no photostimulation), excluding any early lick trials."
The AI adopted these criteria and validated them: the surviving sessions have mean performance
83.8 % (range 65.8–98.9 %) against the paper's "84 % correct rate (range, 65–99 %)", which it
treats as confirmation that both the criteria and its performance definition are right. Step 4
records the conflicting evidence — "173 behavioral sessions" in `methods.txt` versus 173 sessions
with good units in the release — and resolves it as "The release contains all recorded sessions;
the paper's *unit* counts are pre-selection." Step 10 issue 2 notes the criteria were moved to be
evaluated on the *curated* trials, which changed 150 → 142 sessions and moved performance closer
to the paper's numbers.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table, one row per behavioural trial. The go cue for each trial
is `BehavioralEvents/go_start_times.timestamps`, and the code asserts there is exactly one go-cue
event per trials-table row (verified in all 174 sessions). Trial bounds `start_time` / `stop_time`
are read and used for the `obs_intervals` match, the photostim→trial assignment and diagnostics.

ii.
```python
'start_time': np.asarray(trials['start_time'][:], dtype=np.float64),
'stop_time': np.asarray(trials['stop_time'][:], dtype=np.float64),
...
'go': np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64),
```

```python
n_trials_all = len(raw['start_time'])
assert len(go) == n_trials_all, (
    f"{raw['identifier']}: {len(go)} go cues for {n_trials_all} trials")
```

iii. Step 4: "Go cue | one per trial | exactly one `go_start_times` per trial in **all 174
sessions** | Consistent; used as the alignment event." The AI explicitly contrasts this with the
sample/delay events, which have more than one entry per trial because "Licking early during the
sample/delay epoch triggered a replay of the epoch" (405 sample and 395 delay starts for 368
trials in the example session), so only the go cue gives an unambiguous 1:1 trial mapping.

## 1-e. How are trials filtered based on quality controls?

i. Four trial-level filters, all about *missing* rather than *atypical* data, plus a deliberate
decision **not** to apply the reference's behavioural filters:

1. **Outside the ephys recording** — trials whose `[start_time, stop_time]` is not one of the
   units' `obs_intervals` are dropped (736 trials in the kept sessions; in 8 sessions the ephys
   covers only a contiguous block of the behavioural trials). The first and last good unit are
   both checked and asserted to agree.
2. **`auto_water` or `free_water` trials** dropped (2,773 trials), following the reference
   `get_regular_trial_mask`.
3. **Trials with no video frame in the [-2.5, +1.5] s window** dropped (239 trials).
4. Sessions with < 2 surviving trials are rejected.

Early-lick, no-response (`ignore`) and photostimulation trials are **kept**. Net: 94,990 trials in
the release → 77,593 in the 142 curated sessions → **73,845 kept**.

ii.
```python
not_recorded = ~raw['recorded']
keep = raw['recorded'] & (raw['auto_water'] == 0) & (raw['free_water'] == 0)

# Drop trials with no video frame in the analysis window; every bin would
# otherwise be labelled "tongue not visible" when the truth is "not measured".
edges_all = window_edges(go)
n_frames_all = (np.searchsorted(raw['video_t'], edges_all[:, -1])
                - np.searchsorted(raw['video_t'], edges_all[:, 0]))
no_video = keep & (n_frames_all == 0)
keep &= n_frames_all > 0
trial_idx = np.flatnonzero(keep)
```

```python
def recorded_trials(units, good, start_time, stop_time, identifier):
    mask = np.zeros(len(start_time), dtype=bool)
    if len(good) == 0:
        return mask
    for unit in (good[0], good[-1]):
        obs = np.asarray(units['obs_intervals'][int(unit)])
        idx = np.searchsorted(start_time, obs[:, 0])
        idx = np.clip(idx, 0, len(start_time) - 1)
        ok = (np.abs(start_time[idx] - obs[:, 0]) < 1e-6) & \
             (np.abs(stop_time[idx] - obs[:, 1]) < 1e-6)
        assert ok.all(), ...
        unit_mask = np.zeros(len(start_time), dtype=bool)
        unit_mask[idx] = True
        ...
    return mask
```

iii. Step 5 Key Decision 4 and Step 10 Check 3(b2): the reference `get_regular_trial_mask`
requires "no early lick, no auto-water, no free-water, no no-response, no photostim", but
"early-lick, no-response (`ignore`) and photostimulation trials are **kept**, because the decoder
specification requires `early_lick` (no/yes) and `outcome` (ignore/miss/hit) as outputs and
photostimulation as an input — removing those trials would leave those variables constant and the
task undefined." The auto/free-water exclusion is kept because "their `outcome` is not a genuine
behavioural report — the DataJoint comment for the equivalent field reads '1: correct **or free
water**'". The `obs_intervals` filter was added in Step 7 after a processing plot revealed trials
in which every neuron was silent for the whole 4 s window; re-check confirms "0 all-silent trials
in the full dataset". The no-video filter exists "because every bin would otherwise be labelled
'tongue not visible' when the truth is 'not measured'".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `nwb.units['spike_times']` (absolute session seconds) for the units with
`units['classification'] == 'good'`, together with `BehavioralEvents/go_start_times.timestamps`,
which places the bin edges. Per-neuron metadata comes from `units['anno_name']`,
`units['electrodes']` and `nwb.electrodes[x, y, z]`.

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.flatnonzero(classification == 'good')
...
out['good_units'] = good
```

```python
units = nwb.units
spike_index = units['spike_times']
counts = np.empty((n_good, n_trials, N_BINS), dtype=np.int32)
for i, unit in enumerate(raw['good_units']):
    counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
```

iii. Step 5 mapping table: "`units['spike_times']` (absolute s) for units with
`classification == 'good'` → `neural[session][trial]`, shape `(n_neurons, 80)`, float32". Step 1
notes that in the reference the `.mat` export already stores go-cue-relative spike times, "the NWB
stores absolute session times plus a go-cue event per trial, so I do the subtraction". Spike times
are the only neural representation in the file.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin **firing rate in Hz**. For each good unit, the 81 absolute
bin edges of every kept trial are stacked into one `(n_trials, 81)` array, a single
`np.searchsorted` over the flattened edges gives the running spike count at each edge, and
differencing adjacent counts gives the spike count per bin. The `(n_good, n_trials, 80)` count
block is transposed to `(n_trials, n_good, 80)`, cast to `float32` and divided by 0.05 s. No
smoothing, normalisation or baseline subtraction.

ii.
```python
def window_edges(go_times):
    return go_times[:, None] + BIN_EDGES[None, :]


def bin_spike_times(spike_times, edges_abs):
    pos = np.searchsorted(spike_times, edges_abs.ravel()).reshape(edges_abs.shape)
    return np.diff(pos, axis=1).astype(np.int32)
```

```python
counts = np.empty((n_good, n_trials, N_BINS), dtype=np.int32)
for i, unit in enumerate(raw['good_units']):
    counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
neural = np.ascontiguousarray(
    counts.transpose(1, 0, 2).astype(np.float32) / BIN_WIDTH)
```

iii. Step 5 Key Decision 6: "**Firing rate**, not spike count (reference
`sliding_histogram(rate=True)`): count / 0.05 s." Step 1 records that the reference
`sliding_histogram` "returns **firing rate = count / bin_width** when `rate=True`". Step 10 Check 3
(d) states the rate convention is kept even though the bin width/stride deviate from the
reference's 40 ms / 3.4 ms sliding bins, because the task prescribes 50 ms bins. Step 6 documents
the `searchsorted` formulation as the deliberate speed-up over a per-(unit × trial × bin) loop:
"`np.searchsorted` requires only the *haystack* to be sorted, so the trial windows need not be
globally monotone."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units['classification'] == 'good'` are kept — the materialised output of the
region-specific logistic-regression QC classifier of Chen, Liu et al. (2023). No thresholds are
applied to any of the 15 individual Kilosort quality metrics, `unit_quality` is not used, and the
per-unit-per-trial `is_good_trials` column is not used. A session with zero good units is rejected.
This retains 69,453 / 272,227 units (25.5 %) over the release and 57,023 units in the 142 kept
sessions (mean 402/session, range 90–923).

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.flatnonzero(classification == 'good')
```

```python
if n_good == 0:
    reject = 'no good units'
```

```python
'neuron_curation': "units['classification'] == 'good' (per-region quality-control "
                   'classifier of Chen, Liu et al. 2023); all such units carry a CCF '
                   'annotation.',
```

iii. Step 1 Notes: "The reference uses `qc_mode='classifier'`, i.e. the per-region
logistic-regression classifier … In the NWB release this is materialised as the
`units['classification']` column (`good` / `unlabelled`), so `classification == 'good'` is the
exact NWB equivalent of the reference's `goodunits/*.mat` lists. (Confirmed: every `good` unit has
a non-empty CCF annotation `anno_name`…)" Step 5 Key Decision 2 adds "No further metric
thresholds: the 15 quality metrics were already consumed by that classifier, so re-thresholding
them would double-filter." Validation: 69,453 good units / 25.5 % of clusters against the paper's
69,943 / 25.9 % (attributed to the dandiset version difference), and nine of the eleven per-region
counts in the data paper's Fig. 1E reproduced exactly. Step 3 also records that `is_good_trials`
is True for 99.98 % of unit×trial pairs, "so no additional per-trial unit masking is applied —
same as the reference."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. To **go-cue onset**, using `BehavioralEvents/go_start_times.timestamps` (one per trial,
asserted). Spike times and event timestamps are already on the same session-absolute clock, so no
resampling or offset correction is needed: the fixed relative edge grid is added to each trial's
go-cue time to give absolute bin edges, and spikes are binned against those directly.

ii.
```python
OFF_START = -2.5          # s, signed offset of the window start from the go cue
OFF_END = 1.5             # s, signed offset of the window end from the go cue
BIN_EDGES = OFF_START + BIN_WIDTH * np.arange(N_BINS + 1)       # (81,)
```

```python
go = go[trial_idx]
edges = window_edges(go)          # go[:, None] + BIN_EDGES[None, :]
...
counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
```

```python
'temporal_alignment_event': 'go cue onset (auditory go cue, BehavioralEvents/'
                            'go_start_times; one per trial)',
'off_start': OFF_START,
'off_end': OFF_END,
```

iii. Step 5 Key Decision 5: "**Alignment**: go-cue onset (`go_start_times`), one per trial,
verified in all 174 sessions. Window −2.5 s → +1.5 s, 80 bins of 50 ms." Step 10 Check 3(c)
compares this with the reference, where "spike times [are] already go-cue-relative in the `.mat`
export", and concludes "same". The alignment was verified visually (panel 3 of the processing
figures shows the population PSTH jumping exactly at t = 0 and then splitting by lick direction)
and numerically (a whole trial re-binned from the raw NWB spike times with `np.histogram`).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, non-overlapping, 80 bins spanning [-2.5, +1.5) s relative to the go cue, identical for
every trial and every session. The grid is defined once at module level as 81 relative edges and
reused everywhere; bin *b* is the half-open interval `[-2.5 + 0.05b, -2.5 + 0.05(b+1))` and bin
centres `-2.475 + 0.05b` are used for time-valued inputs. This is a deviation from the reference's
40 ms-wide / 3.4 ms-stride sliding bins, made because the task prescribes 50 ms bins. There is no
second rebinning step: spikes and video frames are each binned once, directly onto this grid.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_WIDTH = 0.05          # s
N_BINS = int(round((OFF_END - OFF_START) / BIN_WIDTH))          # 80
BIN_EDGES = OFF_START + BIN_WIDTH * np.arange(N_BINS + 1)       # (81,)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])            # (80,)
```

```python
'time_bin_size': BIN_WIDTH * 1000.0,
'bin_centers_s': BIN_CENTERS.astype(np.float32),
```

iii. Step 3 Processing Details: "reference uses 40 ms width / 3.4 ms stride sliding bins → firing
**rate** (spikes / bin_width). The present task instead prescribes **50 ms bins**, so I use
non-overlapping 50 ms bins over `[-2.5, +1.5)` s (80 bins), rate = count / 0.05 s." Step 10 Check 5
records the half-open convention: "bin *b* is `[-2.5 + 0.05b, -2.5 + 0.05(b+1))`, half-open, so no
spike or frame is double counted; verified by the exact re-binning sanity check."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times.timestamps` (the sample-epoch / instruction-tone onsets of
the session) together with the trial's go cue. A trial can contain several sample onsets because an
early lick replays the sample epoch, so the tone of a trial is defined as the **last**
`sample_start_times` at or before that trial's go cue. The code asserts that such a tone exists
and that it lies inside the trial.

ii.
```python
'sample_start': np.asarray(events['sample_start_times'].timestamps[:], dtype=np.float64),
```

```python
# Tone onset = last sample-epoch onset at or before the go cue.  Early licking
# replays the sample/delay epoch, so a trial can contain several onsets; the
# most recent one is the tone the animal is acting on.
tone_idx = np.searchsorted(raw['sample_start'], go, side='right') - 1
assert np.all(tone_idx >= 0), f"{raw['identifier']}: go cue before any tone"
tone = raw['sample_start'][tone_idx]
assert np.all(tone >= raw['start_time'][trial_idx] - 1e-9), (
    f"{raw['identifier']}: tone onset outside its trial")
```

iii. Step 4: "Replays produce extra sample/delay onsets. 'Tone onset' is therefore defined as the
**last** `sample_start_times` at or before the trial's go cue (always inside the trial: 0
violations dataset-wide)." Step 5 maps it to the reference's `task_sample_time`.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Because the bins are laid out around the go cue, the value of bin *b* on trial *t* is simply the
bin centre plus the go-cue-to-tone gap: `BIN_CENTERS[b] + (go[t] - tone[t])`. The result is a
continuous, time-varying `float32` row; it is negative for bins before the tone. Observed range on
the full dataset is [-1.525, 11.894] s (the long tail comes from early-lick replays that push the
tone far back from the go cue).

ii.
```python
time_from_tone = (BIN_CENTERS[None, :] + (go - tone)[:, None]).astype(np.float32)
...
inputs = np.stack([time_from_tone, photostim], axis=1)  # (n_trials, 2, 80)
```

```python
'seconds elapsed since the onset of the instruction tone (the last sample-epoch '
'onset at or before the go cue; negative before the tone). Continuous.',
```

iii. Step 5 mapping table: "input value at bin *b* = `bin_centre_b − (tone_onset − go)` (seconds
since the tone)". Step 9 cross-checks the resulting range against the raw data: "go − tone: median
1.85 s, min 0.95, max 10.42" and notes the converted range [-1.525, 11.894] s is consistent once
bin centres (±2.475 s) are added. Step 10 Check 2 re-derives 25 random (trial, bin) values straight
from the NWB with `np.allclose(atol=1e-4)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the grid that defines the neural binning — the same 80 bin centres of
the same go-cue-anchored window — so bin *k* of the input covers the same interval as bin *k* of
the firing rates by construction. No interpolation or separate alignment step exists.

ii.
```python
BIN_EDGES = OFF_START + BIN_WIDTH * np.arange(N_BINS + 1)       # (81,)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])            # (80,)
```

```python
edges = window_edges(go)                                     # neural
time_from_tone = (BIN_CENTERS[None, :] + (go - tone)[:, None]) # input 0
```

iii. Step 5 Key Decision 5: "Bin *b* spans `[-2.5 + 0.05b, -2.5 + 0.05(b+1))`; the bin centre
`-2.475 + 0.05b` is used for time-valued inputs." The `--show-processing` figure (panel 4) was used
to confirm the alignment visually: "the ramp cross[es] zero exactly at the marked true tone onset".

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `BehavioralEvents/photostim_start_times.timestamps` and
`BehavioralEvents/photostim_stop_times.timestamps` — the absolute on/off times of each
photostimulation event. The trials-table column `photostim_onset` (a string, `'N/A'` when there was
no stimulation) is read but used only for the control-trial definition in session curation and as a
cross-check. The trials table `start_time` is used to assign each stim event to its trial.

ii.
```python
'photostim_onset_str': np.asarray(trials['photostim_onset'][:]),
'photostim_start': np.asarray(events['photostim_start_times'].timestamps[:], dtype=np.float64),
'photostim_stop': np.asarray(events['photostim_stop_times'].timestamps[:], dtype=np.float64),
```

iii. Step 1: "The reference's photostim variable is `[laser_power, stim_type, laser_on_time,
laser_off_time]`, with on/off times re-referenced to the go cue. The NWB equivalent is
`BehavioralEvents/photostim_start_times`/`photostim_stop_times` (absolute) plus the trials-table
columns `photostim_onset`/`photostim_power`/`photostim_duration`." Step 4 verifies the two are
consistent: "`len(photostim_start_times)` equals the number of trials with
`photostim_onset != 'N/A'` and they map 1:1 to those trials **in all 174 sessions**", which
justifies using the event timestamps (which need no string parsing) as the source.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1 `float32`) time series, one value per 50 ms bin: a bin is 1 if its half-open
interval **overlaps** any `[photostim_start, photostim_stop)` interval. Each stim event is first
mapped to its trial via `searchsorted` on `start_time`, then to the row of the kept-trial array;
events belonging to dropped trials are discarded. Trials and sessions with no stimulation come out
all-zero.

ii.
```python
photostim = np.zeros((n_trials, N_BINS), dtype=np.float32)
if len(raw['photostim_start']):
    # A bin [t0, t1) is "on" if it intersects any [stim_start, stim_stop).
    stim_trial = np.searchsorted(raw['start_time'], raw['photostim_start'],
                                 side='right') - 1
    position = np.searchsorted(trial_idx, stim_trial)
    in_kept = (position < n_trials) & (trial_idx[np.clip(position, 0, n_trials - 1)]
                                       == stim_trial)
    for row, on, off in zip(position[in_kept], raw['photostim_start'][in_kept],
                            raw['photostim_stop'][in_kept]):
        overlap = (edges[row, :-1] < off) & (edges[row, 1:] > on)
        photostim[row, overlap] = 1.0
```

iii. Step 5 mapping table: "1 if the bin `[t0,t1)` intersects `[stim_start, stim_stop)`, else 0",
mapped to the reference's `task_stimulation[:,2:]`. The task specification requires
"Whether **photostimulation** is on at every time point (discrete, time-varying)", so a per-trial
flag was rejected in favour of a binary time series. Step 10 Check 5 covers the boundary case:
"Photostim offset 1 ms *after* the go cue (34.6 % of stim events) — kept as-is; it affects the
single bin [0, 50 ms) and reflects the real stimulus", and Step 4 confirms the timing against the
paper's "always ended before the Go cue".

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Via the same absolute bin-edge array used for the spikes: `edges` is `go[:, None] +
BIN_EDGES[None, :]`, and the stim on/off times are absolute session times, so they are compared
against those edges directly. No re-referencing or interpolation is done beyond mapping the event
to its trial row.

ii.
```python
edges = window_edges(go)               # same array used to bin the spikes
...
overlap = (edges[row, :-1] < off) & (edges[row, 1:] > on)
photostim[row, overlap] = 1.0
```

iii. Everything in the NWB file is on one session-absolute clock (Step 4), so using the same
`edges` array guarantees bin-for-bin correspondence with the firing rates. Verified in Step 10
Check 2 ("**input 1**, 100 random (trial, bin) cells — overlap of the bin with any
`[photostim_start, photostim_stop)` — PASS") and visually in panel 1/4 of the processing figures,
where the shaded photostim window in the spike raster coincides with the square wave in the input.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The trials table has no choice column, so choice is derived from two columns:
`trials['trial_instruction']` (`'left'` / `'right'`) and `trials['outcome']`
(`'hit'` / `'miss'` / `'ignore'`).

ii.
```python
'instruction': np.asarray(trials['trial_instruction'][:]),
'outcome': np.asarray(trials['outcome'][:]),
...
instruction = raw['instruction'][trial_idx]
outcome = raw['outcome'][trial_idx]
```

iii. Step 5 mapping table: "`trials['trial_instruction']` + `trials['outcome']` →
`output[0]` = `lick_direction_choice`", mapped to the reference's `trial_type` × `correctness`
(`create_4fold_trial_type_mask`). Key Decision 9: "This is the standard reading of `outcome` in
this task and matches the reference's `correctness` semantics (1 correct / 0 error / −1 no
response)."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Start from the instructed side (`left` → 0, `right` → 1); on a `miss` (error) trial flip it,
because the mouse licked the opposite port; on an `ignore` trial set it to 2 (`no lick`). The
result is one integer per trial, written into row 0 of the `(n_trials, 4, 80)` `int8` output array
and repeated across all 80 bins so that all four outputs share one time-varying array.
`output_values[0] = ['left', 'right', 'no lick']`. Full-dataset distribution: left 0.447,
right 0.443, no lick 0.109.

ii.
```python
CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
...
choice = np.where(instruction == 'left', CHOICE_LEFT, CHOICE_RIGHT)
# An error trial means the mouse licked the port opposite the instruction.
choice = np.where(outcome == 'miss', 1 - choice, choice)
choice = np.where(outcome == 'ignore', CHOICE_NOLICK, choice)
...
outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int8)
outputs[:, 0, :] = choice[:, None]
```

iii. Step 5 Key Decision 9: "**Choice on error trials** is the *opposite* of the instruction (the
mouse licked the wrong port), and `no lick` on `ignore` trials." Step 10 Check 2 verifies the
derivation for every trial of 6 random sessions directly from the NWB, and includes the global
consistency check "choice 'no lick' ⇔ outcome 'ignore' — PASS".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `trials['outcome']`, which already holds exactly the three strings `'ignore'`,
`'miss'` and `'hit'`.

ii.
```python
'outcome': np.asarray(trials['outcome'][:]),
...
outcome = raw['outcome'][trial_idx]
```

iii. Step 2 records that `outcome ∈ {hit, miss, ignore}` in the trials table and Step 5 maps it
one-to-one onto the requested output categories; no derivation is needed. The reference analogue is
`behavior_report` / `correctness` (1 / 0 / −1).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped through a fixed dictionary to `0` ignore, `1` miss, `2` hit — the
order given in the task specification — and written into row 1 of the output array, repeated across
all 80 bins. `output_values[1] = ['ignore', 'miss', 'hit']`. Full-dataset distribution: ignore
0.109, miss 0.153, hit 0.737.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
...
outcome_code = np.array([OUTCOME_CODE[o] for o in outcome], dtype=np.int8)
outputs[:, 1, :] = outcome_code[:, None]
```

iii. Step 5 mapping table: "`trials['outcome']` → `output[1]` = `outcome`; `ignore`→0, `miss`→1,
`hit`→2", i.e. the code order of the instructions' "Outcome (ignore, miss, hit, per-trial)". It is
a per-trial label, so it is broadcast across bins like the other per-trial outputs.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from `trials['early_lick']`, which holds `'early'` / `'no early'`.

ii.
```python
'early_lick': np.asarray(trials['early_lick'][:]),
...
early = raw['early_lick'][trial_idx]
```

iii. Step 2/Step 5: the trials table flags early licking explicitly, so no derivation is needed;
the reference analogue is `behavior_early_report`.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A boolean comparison to `'early'`, cast to `int8` (0 no, 1 yes), written into row 2 of the
output array and repeated across all 80 bins. `output_values[2] = ['no', 'yes']`. Full-dataset
distribution: no 0.885, yes 0.115 — matching the 0.114 early-lick fraction measured directly on
the raw trials tables.

ii.
```python
early_code = (early == 'early').astype(np.int8)
...
outputs[:, 2, :] = early_code[:, None]
```

iii. Step 5 mapping table: "`trials['early_lick']` → `output[2]`; `no early`→0, `early`→1". The
early lick itself occurs during the sample or delay epoch, i.e. inside the -2.5 s window, but the
flag is stored per trial, so it is broadcast across bins.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']`, whose
`data` is `(n_frames, 3)` = (tongue x, tongue y, DeepLabCut likelihood) with matching per-frame
`timestamps` (dt = 0.0034 s, ≈294 Hz). Column 1 is the y value, column 2 decides visibility; column
0 is read but unused. The series is present in all 174 sessions.

ii.
```python
tongue = tracking['Camera0_side_TongueTracking']
out['video_t'] = np.asarray(tongue.timestamps[:], dtype=np.float64)
tongue_data = np.asarray(tongue.data[:], dtype=np.float64)
out['tongue_y'] = tongue_data[:, 1]
out['tongue_likelihood'] = tongue_data[:, 2]
```

iii. Step 2: "`nwb.acquisition['BehavioralTimeSeries']` — DeepLabCut tracking, `(n_frames, 3)` =
`(x, y, likelihood)` with per-frame `timestamps`: `Camera0_side_JawTracking`,
`Camera0_side_NoseTracking`, `Camera0_side_TongueTracking` (present in **all 174** sessions)". Step
5 maps it to the reference's `align_markers_between_lims` / `tongue_y` marker.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) A frame counts as **visible** if its DeepLabCut likelihood > 0.9. (2) For each
(trial, bin), the number of visible frames and the sum of their y values are obtained with prefix
sums over the whole session indexed by the bin edges, giving the per-bin mean y of the *visible*
frames only; bins with no visible frame get NaN. (3) Those per-bin means are discretised (see 8-c).
A bin is treated as visible if **any** frame in it is visible.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
...
def segment_sums(values, mask, edges_abs, sample_times):
    pos = np.searchsorted(sample_times, edges_abs.ravel()).reshape(edges_abs.shape)
    cum_n = np.concatenate([[0], np.cumsum(mask)])
    cum_v = np.concatenate([[0.0], np.cumsum(np.where(mask, values, 0.0))])
    n_frames_bin = np.diff(pos, axis=1)
    n_masked_bin = cum_n[pos[:, 1:]] - cum_n[pos[:, :-1]]
    masked_sum = cum_v[pos[:, 1:]] - cum_v[pos[:, :-1]]
    return n_frames_bin, n_masked_bin, masked_sum
```

```python
visible = raw['tongue_likelihood'] > TONGUE_LIKELIHOOD_THRESHOLD
_, n_visible_bin, y_sum_bin = segment_sums(
    raw['tongue_y'], visible, edges, raw['video_t'])
bin_visible = n_visible_bin > 0
y_mean = np.where(bin_visible, y_sum_bin / np.maximum(n_visible_bin, 1), np.nan)
```

iii. Step 5 Key Decision 7: "**Tongue visibility threshold** `likelihood > 0.9`. The DeepLabCut
likelihood is extremely bimodal (87 % of frames < 1e-4, 11 % > 0.999); the visible fraction changes
by < 0.5 % of frames between thresholds of 0.01 and 0.999, so the choice is immaterial (verified in
Step 10 — 0.1216 at 0.5 vs 0.1168 at 0.9 vs 0.1094 at 0.999). A bin counts as visible if **any**
frame inside it is visible, because a tongue protrusion lasts only ~50–100 ms and a majority rule
would drop genuine short protrusions straddling a bin edge." Step 10 Check 3 explains the
divergence from the reference's marker handling ("occluded tongue set to its mean"): "the task
defines 'not visible' as its own output class, so imputing the mean would destroy exactly the
distinction being decoded." Step 6 documents the prefix-sum reduction as an efficiency measure.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes. The 40th and 60th percentiles are computed **per session**, over the per-bin mean
y values of all *visible* bins of all *retained* trials of that session. A bin is then labelled
0 if its mean y is below the 40th percentile, 2 if above the 60th, 1 in between, and 3 if no frame
in the bin was visible. `output_values[3] = ['y < 40th pct', 'y 40th-60th pct', 'y > 60th pct',
'not visible']`. By construction the visible bins split 40/20/40; over the full dataset (where
74 % of bins have no visible tongue) the four classes are 0.104 / 0.052 / 0.104 / 0.740.

ii.
```python
TONGUE_LOW_PCT, TONGUE_HIGH_PCT = 40.0, 60.0
TONGUE_INVISIBLE = 3
...
tongue_class = np.full((n_trials, N_BINS), TONGUE_INVISIBLE, dtype=np.int8)
percentiles = (np.nan, np.nan)
if bin_visible.any():
    visible_values = y_mean[bin_visible]
    p_low, p_high = np.percentile(visible_values, [TONGUE_LOW_PCT, TONGUE_HIGH_PCT])
    percentiles = (float(p_low), float(p_high))
    cls = np.where(y_mean < p_low, 0, np.where(y_mean > p_high, 2, 1))
    tongue_class[bin_visible] = cls[bin_visible].astype(np.int8)
outputs[:, 3, :] = tongue_class
```

iii. Step 5 Key Decision 8: "**Percentiles for the tongue discretisation** are computed **per
session** over the visible per-bin y-values of all retained trials of that session — the population
that is actually being labelled — so classes 0/1/2 hold 40 %/20 %/40 % of the visible bins by
construction." The fourth class exists because the task's discretisation lists "3: not visible".
Step 10 Check 2 re-derives the classes for every trial × bin of 6 sessions with an independent
frame-by-frame Python loop and recomputed percentiles: "PASS, exact equality, percentiles identical
to 3 decimals", and the global check "tongue classes 0/1/2 split the visible bins 0.400 / 0.200 /
0.400 — PASS".

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. This is the only genuinely time-varying output. The camera `timestamps` are on the same
session-absolute clock as the spikes and go cues, so the frames are binned with `searchsorted`
against exactly the same `edges = go[:, None] + BIN_EDGES[None, :]` array used for the firing
rates — bin *k* of the tongue output therefore covers the same interval as bin *k* of the neural
data. Trials with no frame at all in the window are dropped (see 1-e); bins with no visible frame
inside a covered trial become class 3.

ii.
```python
edges = window_edges(go)                     # shared with the neural binning
...
_, n_visible_bin, y_sum_bin = segment_sums(
    raw['tongue_y'], visible, edges, raw['video_t'])
```

```python
pos = np.searchsorted(sample_times, edges_abs.ravel()).reshape(edges_abs.shape)
```

iii. Step 4 establishes that all NWB streams share one clock and that `dt = 0.0034 s` exactly,
matching the reference's marker alignment `times_for_frames = arange(n_frames)*0.0034 − go_time`;
Step 10 Check 3(c) records the comparison as "same". The alignment was checked visually in panel 5
of the processing figures ("the raw 300 Hz tongue trace, the per-bin mean of the *visible* frames …
and the resulting class trace") and panel 7 ("class 3 almost everywhere before the go cue and
licking structure after it").

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is either excluded (when nothing was recorded) or given an explicit category (when
the measurement legitimately has no value):

- **Session never quality-controlled** (`classification`/`anno_name` all NaN, 1 session): no unit
  matches `== 'good'`, so `good` is empty and the session is rejected with `'no good units'`.
- **Ephys stops before the behaviour ends** (8 sessions): trials outside the units'
  `obs_intervals` are dropped, which removed 736 trials and left "0 all-silent trials" in the
  output.
- **Video stops before the session ends** (3 sessions): trials with no frame in the window are
  dropped (239 trials); one session then falls below the behavioural criteria and is dropped.
- **Tongue not tracked in a bin**: that bin gets the explicit class 3 (`'not visible'`).
- **Trials with no genuine behavioural report** (`auto_water`, `free_water`): dropped.
- **Bins that fall outside the trial's `[start_time, stop_time]`** (2.9 % on average) necessarily
  contain no spikes and get rate 0; this is documented, not patched, in
  `metadata['known_limitation']`.
- **Session-level failure of any kind**: a worker-level `try/except` records the exception as a
  rejection reason instead of aborting the run.

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.flatnonzero(classification == 'good')
...
if n_good == 0:
    reject = 'no good units'
```

```python
keep = raw['recorded'] & (raw['auto_water'] == 0) & (raw['free_water'] == 0)
no_video = keep & (n_frames_all == 0)
keep &= n_frames_all > 0
```

```python
tongue_class = np.full((n_trials, N_BINS), TONGUE_INVISIBLE, dtype=np.int8)
...
tongue_class[bin_visible] = cls[bin_visible].astype(np.int8)
```

```python
'known_limitation': 'The NWB release stores spikes only within each trial\'s '
                    '[start_time, stop_time] interval ... so those bins contain no '
                    'spikes and have a firing rate of 0.',
```

iii. Step 10 Check 5 enumerates every edge case and its handling, and Step 10 "Issues Found and
Resolved" documents that the `obs_intervals` problem was discovered from a processing plot showing
an empty raster and fixed by adding `recorded_trials()`. The reasoning for the no-video drop is
"every bin would otherwise be labelled 'tongue not visible' when the truth is 'not measured'", and
for the tongue class 3 that "the task defines 'not visible' as its own output class, so imputing
the mean would destroy exactly the distinction being decoded." Counts of every dropped category are
preserved per session in `metadata['session_info']`.

## 10-a. What are the most time-consuming steps of the code?

i. Per session, reading the NWB file (the `spike_times` buffer and the ~680 k × 3 tongue array) and
the per-unit `searchsorted` spike binning dominate; a session takes 0.5–1.3 s. Across the run, with
24 worker processes, all 174 files convert in ~15 s and the 9.66 GB pickle write takes ~11 s, for
25 s total — far below the 15-minute budget in the instructions. The script prints a per-session
elapsed time and the total, and `--show-processing`/plotting is optional and excluded from the
normal path.

ii.
```python
t0 = time.time()
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    ...
    'elapsed': time.time() - t0,
```

```python
print(f'  + {res["identifier"]}: {res["neural"].shape[0]} trials x '
      f'{res["neural"].shape[1]} units  ({res["elapsed"]:.1f}s)')
```

```python
t_write = time.time()
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f'wrote {size / 1e9:.2f} GB in {time.time() - t_write:.1f} s')
print(f'total elapsed {time.time() - t_start:.1f} s')
```

iii. Step 7 "Run Time Estimates" tabulates measured times per step and the speed-ups that produced
them; Step 7 concludes "Well under the 15-minute budget, so no further optimisation was needed."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two of the three natural loops were removed. Spike binning keeps one Python iteration **per
unit** but handles all trials and bins in a single `searchsorted` per unit (the edges of all trials
are flattened into one call); this loop cannot be collapsed further because `spike_times` is ragged
and each unit has a different number of spikes. The video reduction has **no** Python loop at all —
prefix sums over the whole session, indexed by the edge positions, give every (trial, bin) statistic
in three vectorised operations. The loops that remain and could in principle be vectorised are the
per-photostim-event loop (one iteration per stim event, ≤ a few hundred per session), the
`OUTCOME_CODE` / `annotation_to_region` list comprehensions (one per trial / per neuron), and the
per-trial list splitting in `assemble`; none is a measurable share of runtime.

ii.
```python
# vectorised over trials and bins; one iteration per unit
for i, unit in enumerate(raw['good_units']):
    counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
```

```python
# fully vectorised video reduction
cum_n = np.concatenate([[0], np.cumsum(mask)])
cum_v = np.concatenate([[0.0], np.cumsum(np.where(mask, values, 0.0))])
n_masked_bin = cum_n[pos[:, 1:]] - cum_n[pos[:, :-1]]
masked_sum = cum_v[pos[:, 1:]] - cum_v[pos[:, :-1]]
```

```python
# remaining small loop, one iteration per photostim event
for row, on, off in zip(position[in_kept], raw['photostim_start'][in_kept],
                        raw['photostim_stop'][in_kept]):
    overlap = (edges[row, :-1] < off) & (edges[row, 1:] > on)
    photostim[row, overlap] = 1.0
```

iii. Step 6 "Efficiency notes": "**Inefficiency identified**: the obvious implementation loops over
(unit × trial × bin) in Python. **Speed-up**: all trials' bin edges are stacked into one
`(n_trials, 81)` array and a single `np.searchsorted(spike_times, edges.ravel())` per unit yields
every bin count by differencing"; and "**Inefficiency identified**: reducing 300 Hz video (≈1,180
frames per trial window) bin by bin. **Speed-up**: prefix sums of the visible-frame indicator and
of the masked y values, indexed by the same edge positions → whole session in three vectorised
operations."

## 10-c. What processing does the code repeat multiple times?

i. In the normal `--full` path, essentially nothing: each NWB file is opened once inside a single
`with` block, `read_session` pulls every needed array once, and the bin grid (`BIN_EDGES`,
`BIN_CENTERS`) is a module-level constant reused by every trial and session. The small repetitions
that do exist are: `recorded_trials` reads `obs_intervals` for **two** units (first and last good
unit) rather than one, deliberately, as a cross-check that all units share the same observation
block; `window_edges(go)` is called twice per session (once on all trials for the video-coverage
test, once on the kept trials); and the tongue percentiles are recomputed per session by design,
which is required since the discretisation is per-session. In `--sample` mode the first 8 files are
fully converted although only 2 are kept.

ii.
```python
for unit in (good[0], good[-1]):
    obs = np.asarray(units['obs_intervals'][int(unit)])
    ...
    assert np.array_equal(mask, unit_mask), (
        f'{identifier}: units disagree about which trials were recorded')
```

```python
edges_all = window_edges(go)        # all trials, for the video-coverage test
...
edges = window_edges(go)            # kept trials, used for binning
```

iii. Step 6 describes the conversion as a single pass per session with the results returned as
whole `(n_trials, ...)` blocks. The duplicated `obs_intervals` read is justified in the
`recorded_trials` docstring: "Every good unit of a session shares the same observation block
(verified across the whole release), so the first and last good unit are enough, and they are
cross-checked against each other." Because the tongue percentiles are per session, they can be
computed inside the same pass, so no second pass over the data is needed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little, and it is mostly retained as provenance rather than discarded:

- `segment_sums` computes and returns `n_frames_bin` (total frames per bin), which the caller
  discards with `_` — the only genuinely thrown-away computation in the hot path (it is one
  `np.diff`).
- Column 0 of the tongue tracking array (`tongue_x`) is read from disk and never used.
- A large amount of extra per-neuron and per-session metadata is computed — CCF coordinates,
  hemisphere from the 5700 µm midline, `anno_name` strings, `n_silent_trials`,
  `frac_bins_no_spikes_in_trial`, per-session tongue percentiles, per-trial and per-neuron index
  maps. None of this is used by the decoder, but all of it is written into `metadata` as an audit
  trail, so it is not discarded.
- In `--sample` mode all of the first 8 files are converted in parallel and only the first 2 kept,
  so up to 6 full session conversions are thrown away. In `--show-processing` mode, debug traces
  (raw spike times of 60 units, the full video trace) are collected for 8 sessions but only 2
  figures are drawn.
- Session curation is evaluated only after the tracking and trials data have been read, so the 32
  rejected sessions still pay their file-read cost; neural binning, however, is correctly skipped
  for them.

ii.
```python
_, n_visible_bin, y_sum_bin = segment_sums(
    raw['tongue_y'], visible, edges, raw['video_t'])
```

```python
tongue_data = np.asarray(tongue.data[:], dtype=np.float64)
out['tongue_y'] = tongue_data[:, 1]
out['tongue_likelihood'] = tongue_data[:, 2]      # column 0 (x) unused
```

```python
if args.sample:
    files = files[:8]
...
kept = [r for r in results if r.get('rejected') is None]
if args.sample:
    kept = kept[:2]
```

```python
'neuron_ccf_coordinates': [r['neuron_ccf'] for r in sessions],
'neuron_hemisphere': [r['neuron_hemisphere'] for r in sessions],
'neuron_annotation': [r['neuron_annotation'] for r in sessions],
```

iii. The metadata is deliberate: Step 5 Key Decision 11 states "Region names carry no hemisphere;
the hemisphere of every neuron (from CCF ML vs the reference's 5700 µm midline) is stored separately
in `metadata['neuron_hemisphere']`", and Step 10 Check 2 uses the stored CCF coordinates,
annotations and index maps as the basis for independent sanity checks against the raw NWB. The
sample-mode over-conversion is acknowledged in a code comment ("Curation is cheap compared with
conversion, but not free; in sample mode just walk the list until 2 sessions have been converted"),
and is irrelevant to the `--full` path that produces the delivered dataset.
