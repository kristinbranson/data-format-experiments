# Decisions

> **Context for all entries below.** The agent never finished the task. `CONVERSION_NOTES.md`
> shows Step 6 and Step 7 as `IN PROGRESS` and Steps 8–13 as `NOT STARTED`; `/app` contains no
> `converted_data.pkl`, no `sample_data.pkl`, no `README.md`, and no verification or decoder
> logs. The only substantive artifact is a single first-draft `/app/convert_data.py`, written in
> one shot at trajectory step 27 before the agent had confirmed any NWB field names, plus two
> small patches at steps 2474–2478. The script's own metadata string admits this:
> `'notes': 'Initial conversion implementation; field mappings may require refinement after
> sample validation.'` Roughly 2,800 of the 2,940 trajectory steps were spent stuck on an
> unresponsive terminal while a ~4-hour, 120 GB two-session sample conversion ran. The
> justifications quoted below are therefore drawn mostly from Steps 1–5 of `CONVERSION_NOTES.md`
> and from the agent's trajectory reasoning; for most decisions there is no justification at all.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers every session by recursively globbing `*.nwb` under the hard-coded
`/app/data` directory and sorting the paths. Each file is opened once with `pynwb`'s
`NWBHDF5IO(..., load_namespaces=True)` and read into an `NWBFile`. From that object the agent
pulls the trials table (`nwb.trials`, copied column-by-column into a dict), the units table
(`nwb.units`), the behavioural event streams
(`nwb.acquisition['BehavioralEvents'].time_series`) and the video tracking streams
(`nwb.acquisition['BehavioralTimeSeries'].time_series`). Sessions are processed serially in a
single pass and accumulated into one in-memory dict that is pickled at the end. `--sample`
truncates the file list to the first 2 files; `--full` is the default. Note that the file handle
is closed with an explicit `io.close()` that is *not* in a `try`/`finally` or `with` block, so an
exception mid-session leaks the handle.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
if args.sample:
    files = files[:2]
```
```python
def process_session(path, show_processing=False):
    io = NWBHDF5IO(str(path), 'r', load_namespaces=True)
    nwb = io.read()
    subj = get_subject_name(nwb, path)
    trial_starts, trial_stops, td = infer_trial_intervals(nwb)
    go_times = assign_events_to_trials(get_event_times(nwb, 'go_start_times'), trial_starts, trial_stops)
    sample_times = assign_events_to_trials(get_event_times(nwb, 'sample_start_times'), trial_starts, trial_stops)
    phot_int = get_photostim_intervals(nwb)
    unit_mask, regions, good_col, region_col = get_unit_mask_and_regions(nwb)
    kept_unit_idx, spikes = spike_times_list(nwb, unit_mask)
```
```python
def get_trial_table_dict(nwb):
    cols = list(nwb.trials.colnames)
    out = {}
    for c in cols:
        out[c] = np.asarray(nwb.trials[c][:])
    return out
```

iii. From `CONVERSION_NOTES.md` Step 2: *"Data are organized as NWB files in subject-specific
folders under `/app/data/sub-<subject>/`. Each session is one `*_behavior+ecephys+ogen.nwb` file
containing trial and unit tables plus acquisition groups `BehavioralEvents` and
`BehavioralTimeSeries`."* The agent counted 174 files / 28 subjects / 272,227 units / 94,990
trials before writing the script, so the glob was verified to reach the whole dataset.

## 1-b. How are the data split into subjects?

i. One subject per NWB file, read from `nwb.subject.subject_id` (the numeric id, e.g. `'440956'`),
with a fallback to the parent directory name (`sub-440956`) if the subject object or field is
missing. `subjects` is built lazily in first-encountered order via a `subject_to_idx` dict, and
`subject_idx` records, per retained session, the index into that list. Because the file list is
sorted, first-encountered order is the same as sorted subject order in practice.

ii.
```python
def get_subject_name(nwb, path):
    sid = getattr(getattr(nwb, 'subject', None), 'subject_id', None)
    return sid if sid is not None else path.parent.name
```
```python
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
data['subject_idx'].append(subject_to_idx[subj])
```

iii. No explicit justification is given. `CONVERSION_NOTES.md` Step 2 records the per-subject
session counts derived from the directory layout (`sub-440956: 4, sub-440957: 4, ...`, 28
subjects total), so the agent knew the folder name and the subject id agree.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session; there is no grouping or splitting step. Session
identity is recorded as `path.stem` (the filename without extension, e.g.
`sub-440956_ses-20190207T120657_behavior+ecephys+ogen`) rather than `nwb.identifier`. Session
order in the output follows the sorted file list, which puts sessions in chronological order
within each subject because the filename embeds the acquisition timestamp. A session is dropped
if it yields fewer than 2 trials or zero retained units. Per-session diagnostics are appended to
`metadata['session_info']`.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
```
```python
info = {
    'session_id': path.stem,
    'subject': subj,
    'n_trials': len(neural_trials),
    'n_units_kept': len(kept_unit_idx),
    ...
}
```
```python
if len(neural_trials) < 2 or len(reg_kept) == 0:
    print('SKIP insufficient trials or units', path)
    continue
```

iii. Step 2 of `CONVERSION_NOTES.md`: *"Each session is one `*_behavior+ecephys+ogen.nwb` file"*.
Step 4 flags the 174-files-vs-173-sessions discrepancy against the methods text and resolves it
as: *"Treat local data release as the authoritative source for conversion; investigate whether
one NWB file is extra, auxiliary, or fails paper inclusion criteria during later filtering."*
That investigation was never performed.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. `infer_trial_intervals` finds the `start_time` and
`stop_time` columns by fuzzy name matching and returns them along with the whole trials dict.
Each trial is then assigned the first `go_start_times` event falling inside
`[start_time, stop_time]`; `assign_events_to_trials` returns `NaN` for a trial with no such
event. The trial loop skips any trial whose go time is `NaN`, so the surviving trial set is
"trials-table rows that contain a go cue". No assertion is made that the number of go cues
equals the number of trials.

ii.
```python
def infer_trial_intervals(nwb):
    td = get_trial_table_dict(nwb)
    cols = list(td.keys())
    start_col = find_first(cols, ['start_time', 'start'])
    stop_col = find_first(cols, ['stop_time', 'stop'])
    if start_col is None or stop_col is None:
        raise RuntimeError('Could not find trial start/stop columns')
    return np.asarray(td[start_col], float), np.asarray(td[stop_col], float), td
```
```python
def assign_events_to_trials(event_times, trial_starts, trial_stops):
    out = np.full(len(trial_starts), np.nan, dtype=float)
    for i, (a, b) in enumerate(zip(trial_starts, trial_stops)):
        m = (event_times >= a) & (event_times <= b)
        if np.any(m):
            out[i] = event_times[m][0]
    return out
```
```python
for i, go in enumerate(go_times):
    if not np.isfinite(go):
        continue
```

iii. No explicit justification. Step 5 of the notes only records *"Use Go cue onset as alignment
event: Required by decoder task and supported by explicit `go_start_times`/`go_stop_times` event
streams in NWB."* The event-interval containment approach appears to be defensive coding against
unverified field semantics rather than a reasoned choice.

## 1-e. How are trials filtered based on quality controls?

i. Essentially no trial quality control is applied. The only exclusion is "no go-cue event inside
the trial interval" (see 1-d), which in this dataset removes nothing — every trial has exactly
one go cue. In particular the script does **not** consult `units/obs_intervals` (which in 8
sessions marks up to 376 leading behavioural trials as having no ephys coverage at all) and does
not consult the `free_water` trials column. Those trials are emitted as 4 s of exactly 0 Hz
across every neuron. Early-lick and `ignore` (no-response) trials are deliberately kept. Session
level: drop if `< 2` trials survive or if zero units are retained.

ii. The complete trial filter:
```python
for i, go in enumerate(go_times):
    if not np.isfinite(go):
        continue
    neural_trials.append(bin_spikes_for_trial(spikes, go).astype(np.float32))
    ...
```
```python
if len(neural_trials) < 2 or len(reg_kept) == 0:
    print('SKIP insufficient trials or units', path)
    continue
```

iii. `CONVERSION_NOTES.md` Step 5, Key Decision 2: *"Retain trial-level outputs even if papers
excluded some trial types: Decoder task explicitly requires predicting no-lick/early-lick/outcome
variables, so exclusion rules from paper analyses may need to be relaxed for conversion while
still documented."* This justifies keeping early-lick/ignore trials and matches the reference's
reasoning. Step 3 additionally notes the paper's session criteria (*">65% control-trial
performance plus at least 50 correct left and 50 correct right trials"*) and Step 4 states
*"Expect converted trial counts to differ before applying paper-style trial/session filters;
final decisions to be documented in mapping step"* — but no decision about trials lacking spike
data was ever made or documented, and `obs_intervals`/`free_water` are never mentioned anywhere
in the notes or the trajectory.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (session-absolute seconds, read one unit at a time), combined with the
per-trial go-cue times from `BehavioralEvents/go_start_times` to place the bin edges. Which units
contribute is decided by `get_unit_mask_and_regions`, which picks a "quality" column by fuzzy
name matching over `nwb.units.colnames` with the candidate list
`['good', 'quality', 'label', 'unit_quality']`. Brain regions are similarly searched for with
`['location', 'brain_region', 'structure', 'ccf_acronym', 'acronym']`.

ii.
```python
def get_unit_mask_and_regions(nwb):
    cols = list(nwb.units.colnames)
    good_col = find_first(cols, ['good', 'quality', 'label', 'unit_quality'])
    region_col = find_first(cols, ['location', 'brain_region', 'structure', 'ccf_acronym', 'acronym'])
    n_units = len(nwb.units.id[:])
    mask = np.ones(n_units, dtype=bool)
    if good_col is not None:
        vals = np.asarray(nwb.units[good_col][:])
        if vals.dtype.kind in 'OUS':
            ...
        else:
            mask = vals.astype(bool)
    regions = np.array(['unknown'] * n_units, dtype=object)
    if region_col is not None:
        regions = np.array([str(v) for v in nwb.units[region_col][:]], dtype=object)
    return mask, regions, good_col, region_col
```
```python
def spike_times_list(nwb, unit_mask):
    idx = np.where(unit_mask)[0]
    spikes = []
    for i in idx:
        spikes.append(np.asarray(nwb.units['spike_times'][i], dtype=float))
    return idx, spikes
```

iii. Step 5 mapping table: *"NWB `units` spike times for good units → neural … Need exact
good-unit flag from NWB unit columns; if unavailable, document fallback"*. The fuzzy-matching
helper is the agent's stand-in for that verification: from trajectory step 27, *"we should
include robust field discovery and verbose sanity checks because exact NWB column names may
vary"*. The exact names were listed in `CONVERSION_NOTES.md` Step 2 (they include
`classification`, `anno_name`, `unit_quality`, `is_good_trials`) but were never matched against
the candidate lists.

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin, converted to firing rate in Hz by dividing by the bin width. No
smoothing, normalisation or baseline subtraction. For each trial the absolute bin edges are
`go + BIN_EDGES`, and `np.histogram` is called once per selected unit against those edges,
filling an `(n_selected_units, n_bins)` float32 array. After the trial loop, every trial matrix
is transposed before being stored.

ii.
```python
def bin_spikes_for_trial(spike_times, align_time):
    arr = np.zeros((len(spike_times), len(BIN_CENTERS)), dtype=np.float32)
    rel_edges = align_time + BIN_EDGES
    for i, st in enumerate(spike_times):
        arr[i] = np.histogram(st, bins=rel_edges)[0].astype(np.float32) / BIN
    return arr
```
```python
neural_trials = [x.T.astype(np.float32, copy=False) for x in neural_trials]
```

iii. Rate-in-Hz is not justified in the notes, but Step 1 records that the reference code stores
trial-aligned firing rates: *"`count_neurons_per_file.py` counts neurons using
`ephys_data['fr'].shape[2]`, implying `fr` is organized with neurons on axis 2"*. The transpose
was added at trajectory step 2477 in response to the verification report
(`n_neurons: mean 497968.00, min 277600, max 718336`): *"the critical transpose bug is not yet
fixed … Patch convert_data.py so `neural_trials = [x.T.astype(np.float32, copy=False) …]`"*, with
the stated goal *"ensure each trial neural matrix is shape (n_neurons, n_timepoints), not
(n_timepoints, n_neurons)"*.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The intended rule, per the notes, is "keep only units the NWB file labels `good`". In
practice the fuzzy match `find_first(cols, ['good', 'quality', 'label', 'unit_quality'])`
resolves to the column `is_good_trials` — the substring `good` matches it before `unit_quality`
is ever tried. `is_good_trials` is a per-unit × per-trial boolean matrix, not a per-unit flag, so
`np.asarray(...)` returns a 2-D `(n_units, n_trials)` array, `mask.astype(bool)` keeps it 2-D,
and `np.where(mask)[0]` returns *row indices with repeats* — one entry per `True` cell. I
confirmed this directly against
`sub-440956_ses-20190207T120657`: 1,952 units × 368 trials → mask shape `(1952, 368)`, 718,336
`True` cells, `kept_unit_idx` of length 718,336. The genuine QC column
(`units/classification == 'good'`, 459 of 1,952 units in that session) is never read, and
`region_col` resolves to `None` so every neuron's brain region is the string `'unknown'`.

ii.
```python
good_col = find_first(cols, ['good', 'quality', 'label', 'unit_quality'])
...
    vals = np.asarray(nwb.units[good_col][:])
    if vals.dtype.kind in 'OUS':
        sval = np.array([str(v).lower() for v in vals])
        mask = np.array([('good' in v) or (v == '1') or (v == 'true') for v in sval], dtype=bool)
    else:
        mask = vals.astype(bool)
```
```python
def find_first(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        for c in cols:
            if c.lower() == cand.lower():
                return c
        for c in cols:
            if cand.lower() in c.lower():      # 'good' matches 'is_good_trials'
                return c
    return None
```

iii. `CONVERSION_NOTES.md` Step 3, Neuron curation rules: *"Use units labeled as `good` by the
reference quality-control pipeline/classifiers when that information is available in the
NWB/unit metadata. Reference text reports 69,943 good units across 173 behavioral sessions in the
full paper dataset."* Step 4: *"Need to inspect NWB unit metadata for quality labels and use those
rather than raw counts when building the converted dataset."* That inspection never happened; the
agent relied on `find_first` instead. It did observe the consequence
(`Brain region distribution: unknown: 995936 neurons`, `n_neurons … max 718336`) at trajectory
step 2468 and planned to fix it, but ran out of steps.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset, as required. Spike times, event timestamps and camera timestamps
are all on the same session-absolute clock, so no resampling or offset correction is applied: the
fixed relative edge grid is simply added to each trial's go-cue time to give the absolute
histogram edges for that trial. The go time itself is the first `go_start_times` event inside the
trial's `[start_time, stop_time]` interval.

ii.
```python
go_times = assign_events_to_trials(get_event_times(nwb, 'go_start_times'), trial_starts, trial_stops)
```
```python
def bin_spikes_for_trial(spike_times, align_time):
    ...
    rel_edges = align_time + BIN_EDGES
```
```python
'temporal_alignment_event': 'Go cue onset',
'off_start': T_START,
'off_end': T_END,
```

iii. `CONVERSION_NOTES.md` Step 5, Key Decision 1: *"Use Go cue onset as alignment event:
Required by decoder task and supported by explicit `go_start_times`/`go_stop_times` event streams
in NWB."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Nominally 50 ms bins over −2.5 s to +1.5 s, as the instructions require, and
`metadata['time_bin_size'] = 50.0` ms with `off_start = -2.5`, `off_end = 1.5`. There is no
rebinning: one grid is built once at module level and reused for every trial and session. However
the grid is built with `np.arange(T_START, T_END + BIN + 1e-9, BIN)`, which produces **82 edges /
81 bins** spanning −2.5 s to **+1.55 s** — one bin more than requested, and a window that does not
match the `off_end` recorded in the metadata. The agent's own verification output confirms this
(`T: mean: 81.00, median: 81.00, min: 81, max: 81`, where the reference produces 80).

ii.
```python
BIN = 0.05
T_START = -2.5
T_END = 1.5
BIN_EDGES = np.arange(T_START, T_END + BIN + 1e-9, BIN)   # 82 edges -> 81 bins, ends at 1.55
BIN_CENTERS = BIN_EDGES[:-1] + BIN / 2
```
```python
'time_bin_size': 50.0,
'off_start': T_START,
'off_end': T_END,
'n_timepoints': len(BIN_CENTERS),
```

iii. Directly from the Decoder Task section of the instructions (*"Extract 2.5 s before to 1.5 s
after the go cue … Use 50-ms-width bins"*). The off-by-one is not acknowledged anywhere; the
agent saw `T: 81` in the verification output at step 2467 and did not flag it.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (the sample-epoch tone onsets), paired with the
trial's go-cue time. The tone for a trial is the **first** `sample_start_times` event inside
`[start_time, stop_time]`. If a trial has no sample onset, the tone is faked as `go - 1.5` s.

ii.
```python
sample_times = assign_events_to_trials(get_event_times(nwb, 'sample_start_times'), trial_starts, trial_stops)
```
```python
input_trials.append(build_inputs(go, sample_times[i] if np.isfinite(sample_times[i]) else go - 1.5, phot_int).astype(np.float32))
```
```python
def build_inputs(go_time, sample_start, photostim_intervals):
    tone_time = sample_start
    time_from_tone = (go_time + BIN_CENTERS) - tone_time
```

iii. Step 5 mapping table: *"Trial-relative time from tone onset → input[0] … Tone onset occurs
before Go cue by sample + delay timing; use trial event timestamps from NWB"*. No justification
is offered for taking the first rather than the last tone, and the `go - 1.5` fallback is not
justified either (the task structure implies ~1.85 s: 0.65 s sample + 1.2 s delay).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A single vectorised shift: each bin's value is its absolute centre time minus the tone time,
i.e. `CENTERS + (go − tone)`. The result is stored as row 0 of a `(2, n_timepoints)` float32
array, so it is a genuinely time-varying continuous input, as the instructions ask.

ii.
```python
def build_inputs(go_time, sample_start, photostim_intervals):
    tone_time = sample_start
    time_from_tone = (go_time + BIN_CENTERS) - tone_time
    ...
    return np.vstack([time_from_tone.astype(np.float32), phot])
```
```python
'input_names': ['time_from_tone_onset_sec', 'photostimulation_on'],
```

iii. No explicit justification beyond the Step 5 mapping entry (*"Time-varying continuous input
aligned to Go cue; likely represent as per-bin elapsed time since sample/tone onset within trial
window"*). The agent's verification output showed a range of `[-0.6, 5.8]` s; the 5.8 s upper end
is a direct consequence of the first-tone choice on sample-replay trials, and was noted
(*"Time_from_tone input range also spans -0.6 to 5.8 s, which may be acceptable"*) but not
investigated.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same grid used to bin the spikes: `BIN_CENTERS` are the centres
of `BIN_EDGES`, and both the neural histogram edges and the input time axis are formed by adding
those offsets to the same per-trial `go` time. So bin *k* of the input covers the same interval
as bin *k* of the firing rates by construction, and both arrays have the same (81) length.

ii.
```python
BIN_EDGES = np.arange(T_START, T_END + BIN + 1e-9, BIN)
BIN_CENTERS = BIN_EDGES[:-1] + BIN / 2
```
```python
rel_edges = align_time + BIN_EDGES        # neural
```
```python
time_from_tone = (go_time + BIN_CENTERS) - tone_time   # input
```

iii. Not separately justified; it follows from the single shared grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the `BehavioralEvents` streams `photostim_start_times` and `photostim_stop_times`, read
as session-absolute timestamp arrays and zipped into `(start, stop)` intervals. The agent does
*not* use the trials-table columns `photostim_onset` / `photostim_duration`. If either stream is
absent the session gets an empty interval array (all zeros). The two arrays are truncated to
their common length and paired by index without any validation that the pairing is correct.

ii.
```python
def get_photostim_intervals(nwb):
    keys = list(nwb.acquisition['BehavioralEvents'].time_series.keys())
    if 'photostim_start_times' not in keys or 'photostim_stop_times' not in keys:
        return np.empty((0, 2), dtype=float)
    s = get_event_times(nwb, 'photostim_start_times')
    e = get_event_times(nwb, 'photostim_stop_times')
    n = min(len(s), len(e))
    return np.c_[s[:n], e[:n]] if n else np.empty((0, 2), dtype=float)
```

iii. Step 5 mapping table: *"Photostimulation on/off → input[1]: Binary time-varying indicator
from photostim start/stop event streams over aligned bins … BehavioralEvents
photostim_start/stop_times. Photostim ends before Go cue per methods."* Step 3 adds
*"Photoinhibition occurred on a subset of ~25% randomly interleaved trials and ended before the
Go cue; this supports constructing a time-varying photostim input aligned to Go cue."* I verified
the streams are populated and consistent with the trials table (78 stim events vs 78 trials with
`photostim_onset != 'N/A'` in `sub-440956_ses-20190207T120657`).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary per-bin indicator: a bin is 1 if its centre falls within any photostim interval,
0 otherwise (interval endpoints inclusive on both sides). Stored as row 1 of the float32 input
array, so it is time-varying rather than a per-trial flag, as the instructions require. The
per-trial loop over intervals is over all of the session's intervals, not just the current
trial's.

ii.
```python
    phot = np.zeros(len(BIN_CENTERS), dtype=np.float32)
    abs_centers = go_time + BIN_CENTERS
    for a, b in photostim_intervals:
        phot[(abs_centers >= a) & (abs_centers <= b)] = 1.0
```

iii. Step 5: *"Binary time-varying indicator from photostim start/stop event streams over aligned
bins"*, and the instructions' requirement *"Whether photostimulation is on at every time point
(discrete, time-varying)"*.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The bin centres are converted to absolute session time (`go + BIN_CENTERS`) and compared
directly against the absolute photostim timestamps, so the photostim row sits on exactly the same
go-cue-anchored grid as the firing rates and the time-from-tone row. Because the comparison is in
absolute time, stimulation that occurs in a neighbouring trial but falls inside this trial's
−2.5/+1.55 s window is still captured.

ii.
```python
    abs_centers = go_time + BIN_CENTERS
    for a, b in photostim_intervals:
        phot[(abs_centers >= a) & (abs_centers <= b)] = 1.0
```

iii. Not separately justified; follows from using one shared grid and the global clock.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Nothing — this is a silent failure. The agent looks for a trials column named
`choice`, `lick_direction` or `response_side`; the NWB trials table contains none of these
(columns are `start_time, stop_time, trial, photostim_onset, photostim_power,
photostim_duration, trial_uid, task, task_protocol, trial_instruction, early_lick, outcome,
auto_water, free_water`). `choice_col` is therefore `None` and the code falls back to the
constant `2` (`'no_lick'`) for **every trial in the dataset**. The information needed to derive
choice is present (`trial_instruction` × `outcome`, or the `left_lick_times` /
`right_lick_times` event streams), but neither is used.

ii.
```python
def infer_trial_labels(td):
    cols = list(td.keys())
    choice_col = find_first(cols, ['choice', 'lick_direction', 'response_side'])
    ...
```
```python
choice = map_choice(td[choice_col][i]) if choice_col is not None else 2
```
```python
def map_choice(v):
    s = str(v).lower()
    if 'left' in s:  return 0
    if 'right' in s: return 1
    if 'no' in s or 'ignore' in s or 'miss' in s or s in ('nan', ''): return 2
    ...
    return 2
```

iii. Step 5 mapping table: *"Choice → output[0]: Per-trial categorical label: left / right / no
lick … Determine exact NWB trial column encoding."* The determination was never done. The agent
saw the consequence in verification (`choice: {no_lick (1.000)}`) and diagnosed it at trajectory
step 2468 — *"Choice is constant no_lick for all data … These strongly indicate incorrect trial
label mapping"* — but never fixed it.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The per-trial code is broadcast across all 81 bins as row 0 of a `(4, n_timepoints)` uint8
array, so all four outputs live in one time-varying array. The label space and ordering are
correct: `output_values[0] = ['left', 'right', 'no_lick']` with left = 0, right = 1, no-lick = 2.
Only the value is wrong (always 2).

ii.
```python
out = np.vstack([
    np.full(len(BIN_CENTERS), choice, dtype=np.uint8),
    np.full(len(BIN_CENTERS), outcome, dtype=np.uint8),
    np.full(len(BIN_CENTERS), early, dtype=np.uint8),
    tongue_disc[j].astype(np.uint8),
])
```
```python
'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
'output_values': [
    ['left', 'right', 'no_lick'],
    ...
],
```

iii. The instructions state the output is per-trial but also *"If at all possible, make it
time-varying"*, and the target format allows `(n_output, n_timepoints)`; repeating the per-trial
value across bins is how the agent keeps all four outputs in one array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the trials-table `outcome` column, located by the fuzzy search
`['outcome', 'trial_outcome', 'result', 'correctness']`, which matches the real column `outcome`
exactly. The column holds the strings `'hit'`, `'miss'`, `'ignore'` — precisely the three
categories the instructions ask for.

ii.
```python
outcome_col = find_first(cols, ['outcome', 'trial_outcome', 'result', 'correctness'])
```
```python
outcome = map_outcome(td[outcome_col][i]) if outcome_col is not None else 0
```

iii. Step 5 mapping table: *"Outcome → output[1]: Per-trial categorical label: ignore / miss /
hit … trial metadata and/or correctness columns. Need exact mapping from NWB trial columns."* The
`correctness` candidate comes from Step 1's reading of the reference code
(*"filters to successful trials using `new_ephys_data['correctness'][trial_inds]`"*).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String → integer mapping by substring test, in the order ignore → 0, miss → 1, hit → 2,
matching the instructions' ordering and `output_values[1] = ['ignore', 'miss', 'hit']`. The value
is repeated across all 81 bins as row 1 of the output array. The observed distribution in the
agent's sample verification was `ignore 0.090 / miss 0.200 / hit 0.710`, consistent with the
paper's ~84% correct rate on control trials once photostim and ignore trials are included.

ii.
```python
def map_outcome(v):
    s = str(v).lower()
    if 'ignore' in s: return 0
    if 'miss' in s:   return 1
    if 'hit' in s or 'correct' in s or s == '1': return 2
    try:
        iv = int(v)
        return 2 if iv == 1 else 1
    except Exception:
        return 0
```

iii. The 0/1/2 assignment follows the instructions' listed order (*"Outcome (ignore, miss, hit,
per-trial)"*). No further justification in the notes.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the trials-table `early_lick` column, found by the fuzzy search `['early', 'early_lick']`
(the substring `early` matches `early_lick`). The column holds the strings `'early'` and
`'no early'`.

ii.
```python
early_col = find_first(cols, ['early', 'early_lick'])
```
```python
early = map_early(td[early_col][i]) if early_col is not None else 0
```

iii. Step 5 mapping table: *"Early lick → output[2]: Per-trial categorical label: no / yes … NWB
trial metadata. Important because papers often excluded these trials."*

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The intended mapping is `no → 0`, `yes → 1`, repeated across all 81 bins as row 2, with
`output_values[2] = ['no', 'yes']`. The implementation is broken: `map_early` lower-cases the
value and tests `'early' in s`, which is `True` for both `'early'` **and** `'no early'`, so every
trial is labelled 1 (`yes`). I confirmed this directly: `map_early('no early') == 1` and
`map_early('early') == 1`.

ii.
```python
def map_early(v):
    s = str(v).lower()
    if 'true' in s or 'yes' in s or s == '1' or 'early' in s:   # 'no early' also matches
        return 1
    try:
        return int(bool(int(v)))
    except Exception:
        return 0
```

iii. No justification for the substring approach. The agent observed the result in verification
(`early_lick: {yes (1.000)}`) and wrote at step 2468: *"early_lick is constant yes … These
strongly indicate incorrect trial label mapping"*, but never applied a fix.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: `data` is
`(n_frames, 3)` and `timestamps` is the matching camera clock. Column 1 is taken as y and column
2 as the tracking likelihood, with defensive fallbacks if the array has fewer columns. The column
layout is *assumed* rather than read from the series' `description` attribute (which does state
`('tongue_x', 'tongue_y', 'tongue_likelihood')`); the assumption happens to be correct.

ii.
```python
def tongue_series(nwb):
    ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
    data = np.asarray(ts.data[:])
    t = np.asarray(ts.timestamps[:], dtype=float)
    return t, data
```
```python
    # assume columns include x,y,likelihood or similar; use second column as y when available
    y = data[:, 1] if data.ndim > 1 and data.shape[1] > 1 else data[:, 0]
    vis = np.ones_like(y, dtype=bool)
    if data.ndim > 1 and data.shape[1] > 2:
        vis = np.asarray(data[:, 2] > 0.5)
```

iii. Step 5 mapping table: *"Tongue y-position → output[3]: Time-varying categorical output from
tongue tracking y coordinate discretized by session percentiles; 3 = not visible …
BehavioralTimeSeries `Camera0_side_TongueTracking`. Need to identify coordinate/confidence
columns from timeseries array."* The identification was replaced by the in-code assumption, as
the comment admits.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Per trial: build the sample times `go + BIN_CENTERS`, linearly interpolate the raw y trace
onto them (`np.interp`, `NaN` outside the camera's time range), separately interpolate the
boolean visibility trace and threshold at 0.5, and set y to `NaN` wherever visibility is false.
So each bin takes a single *point sample at the bin centre* rather than the mean of the ~15
camera frames (~294 Hz) that fall inside the bin. Note also that the raw y is interpolated
*before* masking, so a bin centre adjacent to a visibility transition can take a value blended
between a visible and an invisible frame.

ii.
```python
def interpolate_tongue_y(nwb, go_time):
    t, data = tongue_series(nwb)
    y = data[:, 1] if data.ndim > 1 and data.shape[1] > 1 else data[:, 0]
    vis = np.ones_like(y, dtype=bool)
    if data.ndim > 1 and data.shape[1] > 2:
        vis = np.asarray(data[:, 2] > 0.5)
    sample_t = go_time + BIN_CENTERS
    y_interp = np.interp(sample_t, t, y, left=np.nan, right=np.nan)
    vis_interp = np.interp(sample_t, t, vis.astype(float), left=0, right=0) > 0.5
    y_interp[~vis_interp] = np.nan
    return y_interp
```

iii. No justification is documented for point-sampling versus averaging, nor for the 0.5
likelihood threshold. The masking-to-`NaN` step is implicitly justified by the requirement to
have a separate `'not visible'` class.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, as the instructions specify. All finite (i.e. visible) y samples from all of the
session's retained trials are concatenated, and `np.percentile(all_y, [40, 60])` gives two
edges. Each bin is then assigned 0 (`< q40`), 1 (`q40 ≤ y ≤ q60`), 2 (`> q60`), or 3
(`not_visible`) where y is `NaN`. If a session has no visible tongue at all, every bin becomes
class 3. The percentile population is the set of *bin-centre samples inside the analysis
windows*, not all frames or all bins of the whole session. The agent's sample verification gave
`lt_40pct 0.063 / 40_to_60pct 0.031 / gt_60pct 0.063 / not_visible 0.843` — a 40/20/40 split of
the visible bins, as intended.

ii.
```python
def discretize_tongue_session(trial_y_list):
    all_y = np.concatenate([y[np.isfinite(y)] for y in trial_y_list if np.any(np.isfinite(y))]) if trial_y_list else np.array([])
    if len(all_y) == 0:
        return [np.full(len(BIN_CENTERS), 3, dtype=np.int64) for _ in trial_y_list], (np.nan, np.nan)
    q40, q60 = np.percentile(all_y, [40, 60])
    out = []
    for y in trial_y_list:
        d = np.full(len(y), 3, dtype=np.int64)
        m = np.isfinite(y)
        d[m & (y < q40)] = 0
        d[m & (y >= q40) & (y <= q60)] = 1
        d[m & (y > q60)] = 2
        out.append(d)
    return out, (float(q40), float(q60))
```
```python
['lt_40pct', '40_to_60pct', 'gt_60pct', 'not_visible'],
```

iii. Directly from the Decoder Task spec (*"per-session discretization: 0: < 40th percentile … 3:
not visible"*). The realised quantiles are recorded per session in
`metadata['session_info'][i]['tongue_quantiles']`, which is a reasonable auditing choice.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Via the same go-cue-anchored grid: the camera trace is sampled at `go + BIN_CENTERS`, the
identical bin centres used for the input rows and the centres of the spike histogram edges. The
camera timestamps are on the same session-absolute clock as the spikes, so no offset correction
is applied, and the returned vector has the same length (81) as the neural matrix's time axis.
One consequence not handled: the video is trial-gated, so `np.interp` will interpolate *across*
the inter-trial gap rather than marking it missing; in practice the visibility mask usually turns
those bins into class 3 anyway.

ii.
```python
    sample_t = go_time + BIN_CENTERS
    y_interp = np.interp(sample_t, t, y, left=np.nan, right=np.nan)
```
```python
    tongue_y_trials.append(interpolate_tongue_y(nwb, go))
```

iii. Not separately justified; follows from the shared grid. The `--show-processing` plot does
overlay the discretised tongue trace on the same bin-centre axis as the inputs, which was the
agent's intended visual check (the plot file was written, but never reviewed — the "Processing
Plots Review" section of the notes is still the empty placeholder).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script's uniform strategy is **silent fallback to a default**, with no warning and no
assertion anywhere in the file:

- Missing/unrecognised column name → `find_first` returns `None` → a constant is substituted:
  choice → `2` (`no_lick`) for every trial; outcome → `0`; early lick → `0`; brain region →
  the string `'unknown'` for every neuron; unit quality mask → keep all units.
- Trial with no go-cue event in `[start_time, stop_time]` → skipped.
- Trial with no sample-onset event → tone time faked as `go - 1.5` s.
- Missing `photostim_start_times`/`photostim_stop_times` streams → empty interval list, i.e. the
  photostim input is all zeros for that session.
- Mismatched photostim start/stop counts → silently truncated to the shorter of the two.
- Tongue frame with likelihood ≤ 0.5, or a bin centre outside the camera's coverage → `NaN` → the
  explicit `'not_visible'` class (3). A session with no visible tongue at all → all bins class 3.
- Session with < 2 trials or 0 retained units → dropped.

Only the tongue handling surfaces missingness as an explicit category; the rest convert a
detection failure into fabricated, plausible-looking constant data. Three of the four outputs and
the entire `brain_regions` field are wrong as a direct result, and nothing in the conversion log
(`PROCESS <path>` / `WROTE` / counts) reveals it.

ii.
```python
choice = map_choice(td[choice_col][i]) if choice_col is not None else 2
outcome = map_outcome(td[outcome_col][i]) if outcome_col is not None else 0
early = map_early(td[early_col][i]) if early_col is not None else 0
```
```python
regions = np.array(['unknown'] * n_units, dtype=object)
if region_col is not None:
    regions = np.array([str(v) for v in nwb.units[region_col][:]], dtype=object)
```
```python
input_trials.append(build_inputs(go, sample_times[i] if np.isfinite(sample_times[i]) else go - 1.5, phot_int).astype(np.float32))
```
```python
d = np.full(len(y), 3, dtype=np.int64)
m = np.isfinite(y)
```

iii. The design intent is stated in trajectory step 27: *"we should include robust field
discovery and verbose sanity checks because exact NWB column names may vary"* — but the "verbose
sanity checks" were never written. `CONVERSION_NOTES.md` Step 5 lists four planned sanity checks
(raw-spike histogram spot check, photostim vector vs raw times, tongue discretisation vs raw rows,
three hand-checked label trials) and all four are still unchecked boxes.

## 10-a. What are the most time-consuming steps of the code?

i. Measured, not estimated: the agent's two-session sample run took **13,990 s (3.9 hours)** and
produced a **120 GB** pickle, versus 247 s for all 174 sessions in the reference. Extrapolated,
the full run would take ~2 weeks and ~10 TB — far past the instructions' 15-minute budget. The
dominant costs, in order:

1. `bin_spikes_for_trial`, called once per trial, which calls `np.histogram` once per selected
   unit. With the `is_good_trials` mask bug the "unit" list is 718,336 entries long, so this is
   ~2.6×10⁸ `np.histogram` calls per session, each scanning that unit's full spike train.
2. `spike_times_list`, which issues one ragged HDF5 read per entry of `kept_unit_idx` — again
   718,336 reads per session, re-reading each real unit ~368 times.
3. `interpolate_tongue_y`, called once per trial, which re-reads the entire
   `(680500, 3)` camera array and its timestamps from HDF5 *every trial*.
4. Pickling and holding the 120 GB result in memory.

Only a single total-elapsed number is printed; there is no per-step or per-session timing, so the
script gives the user no way to localise the bottleneck. The instructions explicitly asked to
*"Print timing information to find bottlenecks"* and the notes' "Run Time Estimates" table is an
empty placeholder.

ii.
```python
def bin_spikes_for_trial(spike_times, align_time):
    arr = np.zeros((len(spike_times), len(BIN_CENTERS)), dtype=np.float32)
    rel_edges = align_time + BIN_EDGES
    for i, st in enumerate(spike_times):
        arr[i] = np.histogram(st, bins=rel_edges)[0].astype(np.float32) / BIN
    return arr
```
```python
print('elapsed_sec', time.time() - t0)
```

iii. No justification; the agent noted the problem at trajectory step 2466 (*"a very long runtime
(~13,991 s), which is a major efficiency concern for Step 7/9 … we need to … likely optimize
convert_data.py substantially before full conversion"*) and never got to it.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four, all of which the reference either vectorises or avoids:

1. **Trial × unit spike binning.** The per-trial call plus per-unit `np.histogram` is an
   O(n_trials × n_units) double loop. The reference collapses the trial dimension entirely by
   flattening all trials' edges into one array and doing a single `np.searchsorted` per unit
   (`pos = np.searchsorted(s, edges).reshape(n_trials, N_BINS + 1); np.diff(pos, axis=1)`), and
   also reads the ragged spike buffer once rather than per unit. The per-unit loop is irreducible
   (ragged storage), the per-trial one is not.
2. **`assign_events_to_trials`.** A Python loop over trials doing a full boolean scan of the
   event array each iteration — O(n_trials × n_events). One `np.searchsorted` replaces it.
3. **Per-trial tongue interpolation.** One `np.interp` call per trial over the full session
   trace; all trials' sample times could be concatenated into one `np.interp` call (and the trace
   read once, see 10-c).
4. **Per-trial output assembly and the region-index loop.** `np.vstack` of four `np.full` arrays
   per trial, and a Python `for i, r in enumerate(reg_kept)` dict lookup over (in the buggy state)
   718,336 entries per session, could both be vectorised.

ii.
```python
for i, (a, b) in enumerate(zip(trial_starts, trial_stops)):
    m = (event_times >= a) & (event_times <= b)
    if np.any(m):
        out[i] = event_times[m][0]
```
```python
for i, go in enumerate(go_times):
    ...
    neural_trials.append(bin_spikes_for_trial(spikes, go).astype(np.float32))
    ...
    tongue_y_trials.append(interpolate_tongue_y(nwb, go))
```
```python
for i, r in enumerate(reg_kept):
    if r not in region_to_idx:
        region_to_idx[r] = len(brain_regions)
        brain_regions.append(r)
    sess_region_idx[i] = region_to_idx[r]
```

iii. None given. `CONVERSION_NOTES.md` Step 6's "Code inefficiencies identified" and "Code
speedups added" fields are both still the literal placeholder `[Note]`.

## 10-c. What processing does the code repeat multiple times?

i. Three substantial repetitions, all inside the per-trial loop:

1. **The camera trace is fully re-read and re-processed once per trial.**
   `interpolate_tongue_y` calls `tongue_series(nwb)`, which does `ts.data[:]` and
   `ts.timestamps[:]` — a full `(680500, 3)` HDF5 read — and then recomputes the y column and the
   visibility mask, for *every trial*. With ~546 trials/session that is ~546 redundant reads of
   the same ~16 MB array per session. Hoisting it out of the loop is a one-line change.
2. **Each unit's spike train is re-scanned once per trial.** `np.histogram` scans the whole spike
   array on every call, so each unit's spikes are traversed n_trials times instead of once.
3. **Each real unit is read and binned ~n_trials times over.** Because `kept_unit_idx` contains
   repeated row indices (the `is_good_trials` bug), `spike_times_list` reads the same unit's
   spike times up to 368 times and `bin_spikes_for_trial` computes the same histogram row 368
   times, storing 368 identical copies.

Minor: `find_first` rebuilds an unused `lower` dict on every call, and the trials table is copied
column-by-column into a dict even though only four columns are used.

ii.
```python
def interpolate_tongue_y(nwb, go_time):
    t, data = tongue_series(nwb)          # full HDF5 read, once PER TRIAL
    ...
```
```python
for i, go in enumerate(go_times):
    ...
    tongue_y_trials.append(interpolate_tongue_y(nwb, go))
```
```python
def spike_times_list(nwb, unit_mask):
    idx = np.where(unit_mask)[0]          # repeated row indices
    spikes = []
    for i in idx:
        spikes.append(np.asarray(nwb.units['spike_times'][i], dtype=float))
```

iii. None given. The instructions asked to *"Avoid unnecessary file I/O"*; this was never
addressed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.

- **The 368-fold duplicated neuron axis.** ~99.7% of every neural matrix is redundant copies of
  the same unit, computed, stored, pickled and then fed to the decoder as if they were distinct
  neurons. This is the single largest piece of wasted work (and the reason the two-session sample
  was 120 GB).
- **The 81st time bin.** The window runs to +1.55 s rather than the specified +1.5 s, so one bin
  per trial per neuron is outside the requested window and inconsistent with the `off_end = 1.5`
  the script itself writes into metadata.
- **Whole-file column materialisation.** `get_trial_table_dict` reads and converts all 14 trials
  columns; only `start_time`, `stop_time`, `outcome`, `early_lick` are ever used.
- **The tongue x column and the per-trial re-derivation of the visibility mask** (see 10-c) are
  computed and thrown away.
- **Unused imports and dead code**: `from collections import Counter` is never used; the `lower`
  dict inside `find_first` is built and never read; `interpolate_tongue_y`'s
  `vis = np.ones_like(...)` default is always overwritten for this dataset.
- **`metadata['bin_centers_sec']`** duplicates information already implied by
  `off_start`/`off_end`/`time_bin_size`, though it is cheap and arguably useful.

ii.
```python
from collections import Counter          # never used
```
```python
def find_first(cols, candidates):
    lower = {c.lower(): c for c in cols}   # built, never used
```
```python
BIN_EDGES = np.arange(T_START, T_END + BIN + 1e-9, BIN)   # 81 bins, last ends at 1.55 s
```
```python
def get_trial_table_dict(nwb):
    cols = list(nwb.trials.colnames)
    out = {}
    for c in cols:                        # all 14 columns; 4 are used
        out[c] = np.asarray(nwb.trials[c][:])
    return out
```

iii. None given; Step 6 of `CONVERSION_NOTES.md` was never filled in.
