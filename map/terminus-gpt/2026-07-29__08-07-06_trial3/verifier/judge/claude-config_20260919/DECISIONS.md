# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats the dataset as one NWB file per session under `data/sub-<subject_id>/`, finds them with a single sorted glob, and processes each file exactly once. Unlike the reference, it does **not** use `pynwb`; it opens each file directly with `h5py` and reads the raw HDF5 paths (`intervals/trials`, `units/*`, `acquisition/BehavioralEvents/*`, `acquisition/BehavioralTimeSeries/*`, `general/subject/subject_id`). Ragged NWB columns (`spike_times` / `spike_times_index`, `obs_intervals` / `obs_intervals_index`) are decoded by hand with a helper. Text columns are byte-decoded with `decode_arr`. All 174 files are opened; 173 survive (see 2-c). The data directory is hard-coded as the relative path `data`, so the script only works when run from `/app`.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
if args.sample:
    files = files[:2]
sessions = []
for p in files:
    info = process_session(p, edges, bin_centers, show_processing=args.show_processing)
```

```python
with h5py.File(path, 'r') as f:
    trial = load_trial_table(f)
    go_times = load_event_times(f, 'go_start_times')
    sample_event_times = load_event_times(f, 'sample_start_times')
    left_licks = load_event_times(f, 'left_lick_times')
    right_licks = load_event_times(f, 'right_lick_times')
    tongue_ts, tongue_data = load_tongue(f)
```

```python
def get_ragged_row(values, index, i):
    start = 0 if i == 0 else int(index[i - 1])
    end = int(index[i])
    return values[start:end]
```

iii. From CONVERSION_NOTES Step 2: "Data are organized under `data/` by subject folders named `sub-<subject_id>`. Each session is a single NWB file". The AI verified the counts directly from the files (174 sessions, 28 subjects, 272,227 units, 94,990 trials) and reconciled them against the papers in Step 4. It never states why it chose `h5py` over `pynwb`; the trajectory shows it went straight to `h5py` when first probing the file structure and stayed with it.

## 1-b. How are the data split into subjects?

i. Each session's animal is read from `general/subject/subject_id` (a numeric string such as `'440956'`). At assembly, `subjects` is the sorted set of unique ids and `subject_idx` is each session's index into that list. This yields 28 subjects with 3–10 sessions each, matching the dandiset. The subject folder name is not used for grouping.

ii.
```python
subject = f['general']['subject']['subject_id'][()]
if isinstance(subject, bytes):
    subject = subject.decode()
```

```python
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. Step 5 mapping table: "subject folder / `general/subject/subject_id` → subjects / subject_idx ... Use subject IDs from NWB". Step 2 notes that "Subject metadata are under `general/subject` and include `subject_id`". The AI corrected an earlier error in its own notes (it had written 11 subjects) after recounting from the files and getting 28.

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no grouping or splitting is done. Session order follows the sorted file list (which, because the filename embeds the acquisition timestamp, is chronological within each subject). The session identifier is the file stem (`sub-440956_ses-20190207T120657_behavior+ecephys+ogen`) rather than `nwb.identifier`. 173 of 174 sessions reach the output. The per-session identifiers are kept only inside the script's intermediate dict — `metadata` records `n_sessions` but no `session_info` list, so session identity is not recoverable from the saved pickle.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
```

```python
session_id = path.stem
...
info = {'session_id': session_id, 'subject': subject, ...}
```

```python
'metadata': {
    ...
    'n_sessions': len(sessions),
    'bin_centers_s': bin_centers.astype(np.float32),
}
```

iii. Step 2: "Each session is a single NWB file named like `sub-<subject_id>_ses-<timestamp>_behavior+ecephys+ogen.nwb`", so the file boundary is the session boundary. Step 4 anticipated the 174-vs-173 discrepancy: "Treat raw NWB as source universe; expect one session to be excluded by curation." Step 6 decision: "Start from all 174 NWB sessions, then exclude only sessions that fail required data integrity checks."

## 1-d. Are the data correctly split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioural trial. The AI asserts that the number of `go_start_times` events equals the number of rows, which holds in all sessions. It does not re-derive trial boundaries from the event streams, and it correctly avoids treating `sample_start_times` / `delay_start_times` as one-per-trial when locating the tone (it searches within `[start_time, stop_time]` instead — see 3-a).

ii.
```python
tr = f['intervals/trials']
out = {}
for k in tr.keys():
    arr = tr[k][()]
    ...
```

```python
n_trials = len(trial['start_time'])
assert len(go_times) == n_trials, (len(go_times), n_trials)
```

iii. Step 5 mapping table treats the trial table as the per-trial source of `trial_instruction`, `outcome`, `early_lick`, `photostim_onset`, `photostim_duration`, `start_time`, `stop_time`. The assertion is the AI's own sanity check that the go-cue stream is one-per-trial; it is the only structural assertion in the script.

## 1-e. How are trials filtered based on quality controls?

i. Almost no trial filtering is applied. Three guards exist:
  - a trial is skipped if `go - 2.5 s` falls before absolute time 0, or if its window would run past the end of the session-global bin grid;
  - a trial is skipped if `units/is_good_trials` exists **with a column count equal to the number of behavioural trials** and no good unit is valid on that trial (this never actually fires — every trial has at least some valid good unit);
  - a session is dropped if fewer than 2 trials survive, or if no bin anywhere has a finite tongue value.

Crucially, the AI does **not** filter on `units/obs_intervals` and does **not** filter `free_water` trials. In 9 sessions the ephys recording covers only part of the behavioural session; the AI detects the resulting shape mismatch in `is_good_trials` but responds by *disabling* the check rather than by excluding the unobserved trials. The consequence is visible in its own artifact: `verification_full_out.txt` reports **3,511 trials whose neural matrix is entirely zero**, and 94,364 of the 94,370 trials in the retained sessions are kept (only 6 trials dropped in total).

ii.
```python
is_good_trials = np.asarray(f['units']['is_good_trials'][()], dtype=bool) if ('is_good_trials' in f['units'] and f['units']['is_good_trials'].shape[1] == len(trial['start_time'])) else None
```

```python
for i in range(n_trials):
    go = float(go_times[i])
    start = go + edges[0]
    stop = go + edges[-1]
    if start < 0:
        continue

    if is_good_trials is not None:
        valid_units = is_good_trials[good_idx, i]
        if not np.any(valid_units):
            continue
    else:
        valid_units = np.ones(len(good_idx), dtype=bool)

    start_idx = int(go_bin_start[i])
    end_idx = start_idx + len(bin_centers)
    if start_idx < 0 or end_idx > global_rates.shape[1]:
        continue
```

```python
if len(session_trial_neural) < 2:
    return None
```

iii. Step 5 decision 6: "Start from all 174 NWB sessions, then exclude only sessions that fail required data integrity checks (e.g. missing variables or too few valid trials)." Step 4 decided not to adopt the reference code's `get_regular_trial_mask` (which excludes early-lick, auto-water, free-water, no-response and stimulation trials) because "we must still include photostimulation as an input variable... so we may need a broader trial set" — a sound argument, since early lick, ignore and photostim trials are all required by the decoder spec.

The all-zero trials were noticed. The trajectory shows the AI investigating them (steps ~241–256), initially suspecting `is_good_trials`, finding it "does not explain the all-zero neural warnings", then hitting the shape mismatch in the second session and patching it away. It finally concluded (step ~310) that "the warnings in `verification_full_out.txt` are stale from an earlier run/file" after checking `data['neural'][0][i]` — i.e. session index 0 — whereas the verifier's "Session 1" is the *second* session, which is exactly the session whose `obs_intervals` covers only 160 of 480 trials. The conclusion is therefore wrong and the warnings were never addressed; CONVERSION_NOTES Step 10 is left as `IN PROGRESS` with empty placeholders.

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. From `units/spike_times` together with `units/spike_times_index` (the ragged offset array), restricted to units whose `units/classification` is `'good'`. `acquisition/BehavioralEvents/go_start_times/timestamps` supplies the alignment times. `units/is_good_trials` is used as a secondary per-unit-per-trial validity mask where its shape permits.

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
spike_times = f['units']['spike_times'][()]
spike_index = f['units']['spike_times_index'][()]
```

```python
go_times = load_event_times(f, 'go_start_times')
```

iii. Step 5 mapping table: "`units/spike_times` + `units/spike_times_index` for units with `classification == 'good'` → neural". Step 3 quotes the methods: "69,943 good units recorded across 173 behavioral sessions". The trajectory confirms the AI checked that the event `data` field is a constant array of 1s and that the real times live in `timestamps` ("We confirmed that the correct event times are in the `timestamps` field, not `data`").

## 2-b. How is the `neural` data processed?

i. Spike counts are converted to firing rates in Hz. The implementation is unusual: for each good unit, `np.histogram` bins **the whole session** onto a single global 50 ms grid spanning `min(go) - 2.5 s` to `max(go) + 1.5 s`, counts are divided by the bin width, and each trial's 80-bin matrix is then extracted as a slice of that global array. No smoothing, normalisation, or baseline subtraction is applied. On the 4 sessions where `is_good_trials` flags some unit×trial pairs as bad, those rows are overwritten with zeros in the affected trials rather than being dropped (0.15% of unit×trial entries).

ii.
```python
bin_size = edges[1] - edges[0]
session_t0 = float(np.min(go_times) + edges[0])
session_t1 = float(np.max(go_times) + edges[-1])
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
global_centers = (global_edges[:-1] + global_edges[1:]) / 2
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
```

```python
trial_mat = global_rates[:, start_idx:end_idx].copy()
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0
```

iii. Step 5: "Bin spike times into 50 ms bins from -2.5 s to +1.5 s relative to each trial's `go_start_times`; convert to firing rates or keep counts consistently across all sessions." Step 6 notes the inefficiency it was trying to avoid: "Current implementation loops over good units within each trial and may be slow for full conversion", and the speed-up recorded is "Used direct NumPy histogram binning and avoided repeated file opens" — i.e. the whole-session pre-binning was chosen as a performance optimisation over a per-unit-per-trial loop.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept; no thresholds are placed on any individual quality metric (`isi_violation`, `presence_ratio`, `amplitude_cutoff`, etc.), and the older `unit_quality` label is not used. A session with zero good units is dropped. This keeps 173 sessions and 69,452 good units (mean 401.5 per session, min 90, max 923) — within one unit of the reference's 69,453 and close to the white paper's 69,943. Additionally, on the 4 sessions where `is_good_trials` marks some good unit as unreliable on some trial, the AI zeroes that unit's firing rate on that trial instead of excluding the unit or the trial.

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
```

```python
def decode_arr(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode())
        else:
            out.append(str(x))
    return np.array(out, dtype=object)
```

iii. Step 4 resolution: "Use `units/classification == 'good'` to match paper curation rather than all units." Step 3 records the white paper's rule: "we used 15 cluster quality metrics to train classifiers ... Applying the trained classifiers ... provided lists of units that were labeled as 'good'", so the `classification` column is the published verdict and re-thresholding metrics would double-count the QC. Step 4 also predicted the session drop: "expect one session to be excluded by curation" — the dropped file (`sub-440958_ses-20190216T162508`) has NaN `classification` for all units, which `decode_arr` turns into the string `'nan'`, so no unit matches `'good'` and the session returns `None`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Nominally to go-cue onset (`go_start_times/timestamps`), which shares the session-absolute clock with the spikes, so no resampling or offset correction is needed. In practice the alignment is **not exact**: because the spikes were pre-binned onto one session-global grid, each trial's window start is obtained by *rounding* the go cue onto that grid. The resulting per-trial offset between the nominal go cue and the actual bin boundary averages 12.5 ms and reaches 25 ms; 59% of trials are off by more than 10 ms. The inputs and the tongue output, by contrast, are computed on the exact go-relative axis, so the neural stream is jittered relative to the other streams by a trial-dependent sub-bin amount.

ii.
```python
go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
```

```python
start_idx = int(go_bin_start[i])
end_idx = start_idx + len(bin_centers)
if start_idx < 0 or end_idx > global_rates.shape[1]:
    continue
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. Step 5 decision 2: "Align all trials to `go_start_times` and extract [-2.5, +1.5] s windows with 50 ms bins." The rounding is never mentioned in CONVERSION_NOTES or in the trajectory; it is an unremarked side effect of the whole-session pre-binning optimisation described in 2-b. Step 3 notes the relevant justification for using the go cue at all: "photoinhibition always ended before the 'Go' cue", so go-cue alignment keeps the stimulation epoch inside the window.

## 2-e. How is the `neural` data temporally binned/resampled?

i. 80 non-overlapping 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue. The edge/center grid is built once in `main()` from `pre=2.5`, `post=1.5`, `bin_size=0.05` and passed into every session, so every trial in every session has exactly 80 timepoints (confirmed in `verification_full_out.txt`: "T: mean: 80.00, median: 80.00, min: 80, max: 80"). `time_bin_size` is recorded as 50.0 ms and the bin centers are stored in metadata.

ii.
```python
def build_edges(pre, post, bin_size):
    n_bins = int(round((pre + post) / bin_size))
    edges = np.linspace(-pre, post, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers
```

```python
pre = 2.5
post = 1.5
bin_size = 0.05
edges, bin_centers = build_edges(pre, post, bin_size)
```

```python
'time_bin_size': 50.0,
'off_start': -2.5,
'off_end': 1.5,
'bin_centers_s': bin_centers.astype(np.float32),
```

iii. Directly from the Decoder Task section of the instructions ("2.5 s before to 1.5 s after the go cue", "50-ms-width bins"). Step 5 decision 2 restates this. Step 7 verified it: "All sessions have 80 time bins as expected."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `acquisition/BehavioralEvents/sample_start_times/timestamps` (the auditory sample-epoch tone onsets) together with the trial's go cue. For each trial the AI takes the **first** sample onset falling inside `[trials.start_time, trials.stop_time]`. The reference instead takes the **last** sample onset before the go cue; the two differ on 5,534 of 94,990 trials (5.8%), because a lick during the sample or delay epoch replays that epoch and emits a new tone. The two conventions produce the same overall range of go-minus-tone intervals (0.95 s to 10.42 s), so the discrepancy is confined to which replayed tone a multi-tone trial is referred to.

ii.
```python
sample_event_times = load_event_times(f, 'sample_start_times')
...
sample_times = np.full(n_trials, np.nan, dtype=float)
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
    if len(hits):
        sample_times[i] = hits[0]
```

iii. Step 5 mapping table: "Time relative to go cue and tone onset derived from event times → input[0] (`time_from_tone_onset`) ... Need exact tone event mapping from task structure; likely `sample_start_times` corresponds to auditory cue epoch in this task." Step 3 records the task structure ("presample, sample, delay, go, and trial end epochs"). The choice of the *first* tone in the trial is never explicitly justified, and the epoch-replay behaviour that makes the choice non-trivial is not discussed anywhere in the notes.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The bins sit on the go-cue axis, so each bin's value is its center minus the tone's go-relative time, i.e. `center + (go - tone)`. This is a continuous, time-varying input, identical in form to the reference. It is stored as `float32` in row 0 of the `(2, 80)` input array. Observed range over the full dataset: `[-1.5, 11.9]` s. If a trial had no sample event inside its window, `tone_rel` would be NaN and the whole row would become NaN; in practice every trial in all 174 sessions has at least one sample event, so this branch never fires and no NaNs reach the output.

ii.
```python
def build_time_from_tone(bin_centers, tone_time_rel):
    return bin_centers - tone_time_rel
```

```python
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
```

```python
inp = np.stack([inp0, inp1], axis=0)
```

iii. Step 5 mapping table: "Build time-varying continuous vector per trial: bin center time minus tone/sample onset time." The instructions require "Time from tone onset in seconds (continuous, time-varying)", which this satisfies directly. Step 7 recorded the sample-data range `[-0.6, 5.7]` as "plausible but should be checked later".

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the same `bin_centers` array that defines the go-relative grid is used to build the input, so bin *k* of the input nominally covers the same interval as bin *k* of the firing rates. No interpolation or offset correction is applied. The one caveat is that the neural bins are snapped to the session-global grid (see 2-d) while this input uses the exact go-relative centers, so the correspondence is exact only up to the 0–25 ms rounding of the neural stream.

ii.
```python
edges, bin_centers = build_edges(pre, post, bin_size)
...
info = process_session(p, edges, bin_centers, show_processing=args.show_processing)
```

```python
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
```

iii. Implicit — the notes do not discuss it beyond Step 5's statement that the reference "indexes `session_dict['bin_centers']` as the common time axis", i.e. one shared time axis for all streams. The `--show-processing` plot overlays the inputs and outputs on the same axis as the neural raster as a visual check.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the trials-table columns `photostim_onset` and `photostim_duration`, both stored as strings with `'N/A'` on unstimulated trials. The AI identified these correctly and also knew that `acquisition/BehavioralEvents/photostim_start_times` / `photostim_stop_times` exist (it listed them in Step 2 and probed them in Step 5), but did not use them. Critically, it did not use `trials.start_time`, which is the reference frame `photostim_onset` is measured in.

ii.
```python
ps_on = parse_optional_float(trial['photostim_onset'][i]) if 'photostim_onset' in trial else np.nan
ps_dur = parse_optional_float(trial['photostim_duration'][i]) if 'photostim_duration' in trial else np.nan
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
```

```python
def parse_optional_float(x):
    if isinstance(x, str):
        if x in ('N/A', 'nan', ''):
            return np.nan
        return float(x)
    return float(x)
```

iii. Step 5 mapping table: "`photostim_onset` / `photostim_duration` or `photostim_start_times` / `photostim_stop_times` → input[1] (`photostimulation_on`) ... Methods say photostim ends before Go cue; binary vector should therefore mostly occupy pre-go bins on stim trials." Step 4: "Build time-varying photostim input from raw trial/event fields aligned to go cue." Step 3 records the expected prevalence: "~25% randomly interleaved trials".

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1, stored `float32`) time series: a bin is 1 where its center lies in `[onset, onset + duration)`. Non-finite onset or duration (i.e. `'N/A'` trials) yields an all-zero vector without a special branch. **The onset is used verbatim, without converting it from the trial-start reference frame to the go-cue reference frame.** In the data, `photostim_onset` ranges roughly 1.8–2.7 s after trial start, which is about -1.2 s relative to the go cue; used as-is against go-relative bin centers it lands at +1.8 s to +3.2 s, entirely outside the [-2.5, +1.5] s window. The result is that the photostimulation input is **identically zero for every trial in every session** — `verification_full_out.txt` reports `photostimulation_on: [0.0, 0.0]`. The input carries no information at all.

ii.
```python
def build_photostim_vector(bin_centers, onset_rel, duration):
    x = np.zeros(bin_centers.shape[0], dtype=np.float32)
    if onset_rel is None or duration is None:
        return x
    if not np.isfinite(onset_rel) or not np.isfinite(duration):
        return x
    off = onset_rel + duration
    x[(bin_centers >= onset_rel) & (bin_centers < off)] = 1.0
    return x
```

iii. Step 5 decision 3: "Represent photostimulation as a binary time-varying input built from raw timing fields, not by excluding stim trials" — the binary-time-series form follows the instructions ("Whether photostimulation is on at every time point (discrete, time-varying)"). The all-zero result was noticed twice but never diagnosed. Step 7: "Sampled sessions had photostimulation input all zeros, likely because these particular sessions/trials lacked stimulation; this should be checked on broader data." The trajectory (step ~225) says the same: "the input ranges reveal a likely issue: `photostimulation_on` is all zeros in both sample sessions, probably because the first two sample sessions lacked stimulation or our construction missed it." The full-dataset verification then showed the same `[0.0, 0.0]` range and the AI never returned to it (Step 10 and Step 12 are unfilled).

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is not aligned. The vector is built against `bin_centers`, which is the go-cue-relative axis, but the onset value fed in is measured from `trials.start_time`. The reference performs the conversion `stim_on = start_time + onset - go`; the AI omits both `start_time` and `go`. Since the go cue occurs ~3.1–3.3 s after trial start, this is a systematic shift of about +3 s, pushing every stimulation epoch past the end of the trial window.

ii. The reference-frame conversion that is missing — the AI's entire treatment is:
```python
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
```
compared with what the go-relative axis requires (`start_time + onset - go`), which appears nowhere in `convert_data.py`.

iii. No justification is offered: the AI never verified what `photostim_onset` is measured relative to. Its own Step 3 note — "photoinhibition always ended before the 'Go' cue" — directly contradicts the code's implicit assumption that onsets fall at +1.8 s to +2.7 s *after* the go cue, and would have caught the error had it been checked against the computed values.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Primarily from the lick event streams `left_lick_times` and `right_lick_times`: the choice is the side of the **first** lick in the window `[go_time, trials.stop_time]`. When neither stream has a lick in that window, the AI falls back to `trials.trial_instruction` and assigns the instructed side. `trials.outcome` is not used for choice. Two differences from the reference: the response window extends to `stop_time` rather than `go + 1.5 s` (so licks after the response window can determine the label), and no third "no lick" category exists.

ii.
```python
left_licks = load_event_times(f, 'left_lick_times')
right_licks = load_event_times(f, 'right_lick_times')
```

```python
def find_choice_from_licks(left_licks, right_licks, go_time, stop_time):
    l = left_licks[(left_licks >= go_time) & (left_licks <= stop_time)]
    r = right_licks[(right_licks >= go_time) & (right_licks <= stop_time)]
    tl = l[0] if len(l) else np.inf
    tr = r[0] if len(r) else np.inf
    if tl == np.inf and tr == np.inf:
        return None
    return 0 if tl < tr else 1
```

```python
choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1
```

iii. Step 5 mapping table: "`trial_instruction` and/or lick event times → output choice ... Need to determine whether choice should reflect instructed side or actual lick direction; prefer actual choice semantics from reference code if available", cross-referenced to the reference code's `lick_directions` field. Step 4: "Use raw NWB trial annotations and event times to reconstruct decoder outputs consistent with reference semantics." Deriving choice from the actual licks rather than from `instruction × outcome` is a defensible and arguably more direct reading of "lick direction choice". The fallback for no-lick trials is not justified anywhere.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0` = left, `1` = right, written into row 0 of the `(4, 80)` `int64` output array and repeated across all 80 bins. `output_values[0] = ['left', 'right']` — **only two classes**, whereas the instructions specify three ("left, right, no lick"). The 14.9% of trials whose `outcome` is `ignore` (no lick anywhere in the response window) are therefore labelled with the side the tone instructed, i.e. a lick direction the animal never produced. The resulting distribution is `{left 0.497, right 0.503}` (from `verification_full_out.txt`), compared with the reference's three-way split that separates out the no-lick trials.

ii.
```python
out = np.zeros((4, len(bin_centers)), dtype=np.int64)
out[0, :] = choice
out[1, :] = outcome
out[2, :] = early
out[3, :] = ycat
```

```python
'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['lt_40pct', '40to60pct', 'gt_60pct'],
],
```

iii. Step 5 mapping table: "Map left=0, right=1 per trial", matching the instruction's ordering. Repetition across bins follows the target format's preference that outputs be time-varying where possible while the underlying variable is per-trial. The absence of a third class and the treatment of no-lick trials are not discussed in CONVERSION_NOTES; the AI checked only that the distribution was balanced ("choice distribution approx [0.448, 0.552]") and that accuracy exceeded the 2-class chance level of 0.5.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which already contains exactly the strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome = out_outcome_map[str(trial['outcome'][i])]
```

iii. Step 4: "Raw NWB uses ... `outcome` values `hit/ignore/miss`", verified by the AI enumerating the unique values of every categorical trial column in the trajectory (step ~135). Step 5 mapping table: "Direct categorical mapping from NWB strings."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped through a fixed dictionary to `0` = ignore, `1` = miss, `2` = hit, written into row 1 of the output array and repeated across all 80 bins. The dictionary is (wastefully) reconstructed inside the per-trial loop. Distribution over the full dataset: `{ignore 0.149, miss 0.165, hit 0.686}` — a 68.6% hit rate, consistent with a well-trained animal.

ii.
```python
out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_early_map = {'no early': 0, 'early': 1}
outcome = out_outcome_map[str(trial['outcome'][i])]
```

```python
out[1, :] = outcome
```

iii. Step 5 decision 4: "Use direct NWB categorical values `ignore`, `miss`, `hit` mapped to 0/1/2" — the code assignment follows the ordering given in the instructions ("Outcome (ignore, miss, hit, per-trial)"). Per-trial value repeated across bins for the same reason as choice.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `early_lick` column of the trials table, whose values are the strings `'no early'` and `'early'`.

ii.
```python
early = out_early_map[str(trial['early_lick'][i])]
```

iii. Step 4: "`early_lick` is `early`/`no early`", again verified by enumerating unique values. Step 5 mapping table cross-references the reference code's `early_lick_trials` field: "Direct categorical mapping from NWB strings."

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `0` = no, `1` = yes via a fixed dictionary, written into row 2 of the output array and repeated across all 80 bins. Distribution: `{no 0.886, yes 0.114}`. Notably, the AI kept early-lick trials in the dataset rather than excluding them as the data paper does, which is required here because early lick is a decoder output.

ii.
```python
out_early_map = {'no early': 0, 'early': 1}
early = out_early_map[str(trial['early_lick'][i])]
```

```python
out[2, :] = early
```

iii. Step 5 decision 5: "Use direct NWB categorical values `no early` and `early` mapped to 0/1", following the instruction's ordering ("Early lick (no, yes, per-trial)"). Step 4's reasoning for not applying `get_regular_trial_mask` covers keeping the early-lick trials: they are needed as an output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using `data[:, 1]` as the y-position and the matching `timestamps`. The column index is chosen by a helper that hard-codes `1` after only checking that the array is 2-D with at least 2 columns; the series' own `description` attribute (`"Time series for TongueTracking position: ('tongue_x', 'tongue_y', 'tongue_likelihood')"`) is never read. Column 2, the tracking likelihood, is loaded but never used.

ii.
```python
def load_tongue(f):
    grp = f['acquisition/BehavioralTimeSeries']['Camera0_side_TongueTracking']
    data = np.asarray(grp['data'][()], dtype=float)
    ts = np.asarray(grp['timestamps'][()], dtype=float)
    return ts, data


def choose_tongue_y_column(data):
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError('Unexpected tongue tracking shape')
    return 1
```

```python
ycol = choose_tongue_y_column(tongue_data)
tongue_y = tongue_data[:, ycol]
```

iii. Step 5 mapping table: "`Camera0_side_TongueTracking/data` + timestamps → output tongue y-position ... Need to identify which column is y-position; likely second column if data are [x, y, likelihood]." The trajectory shows the AI printing the first rows of the array and the NaN fraction (step ~178) but it stopped at "likely x/y/likelihood or similar" and never confirmed against the description string, nor examined the distribution of the third column.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Per trial, frames whose timestamps fall in `[go - 2.5, go + 1.5)` are assigned to the 80 go-relative bins with `np.digitize`, and the **raw mean y** of the frames in each bin is taken. No likelihood filtering is applied, so frames where the tongue is retracted — 89.4% of all frames in the session checked, for which the tracker still emits a position — are averaged in with genuine protrusions. Bins with no frames are left NaN at this stage. The class edges are then the 40th and 60th percentiles of all finite bin means pooled over the session's trial windows (as opposed to the reference's percentiles over a whole-session bin grid, which additionally covers inter-trial time). Only NaN frames — not low-confidence ones — are excluded.

ii.
```python
mask = (tongue_ts >= start) & (tongue_ts < stop)
yt = tongue_y[mask]
tt = tongue_ts[mask] - go
binned_y = np.full(len(bin_centers), np.nan, dtype=np.float32)
if len(tt):
    inds = np.digitize(tt, edges) - 1
    ok = (inds >= 0) & (inds < len(bin_centers)) & np.isfinite(yt)
    if np.any(ok):
        sums = np.zeros(len(bin_centers), dtype=np.float64)
        cnts = np.zeros(len(bin_centers), dtype=np.int64)
        np.add.at(sums, inds[ok], yt[ok])
        np.add.at(cnts, inds[ok], 1)
        nz = cnts > 0
        binned_y[nz] = (sums[nz] / cnts[nz]).astype(np.float32)
all_binned_y.append(binned_y)
```

```python
all_y = np.concatenate([x[np.isfinite(x)] for x in all_binned_y if np.any(np.isfinite(x))]) if any(np.any(np.isfinite(x)) for x in all_binned_y) else np.array([], dtype=float)
if len(all_y) == 0:
    return None
q40, q60 = np.percentile(all_y, [40, 60])
```

iii. Step 5 decision 7: "Compute session-wide percentiles on valid tongue y-values, then discretize each binned time point into 3 categories", and the mapping-table entry "Interpolate/assign tongue y to decoder bins, then discretize per session using 40th/60th percentiles into 3 categories. ... our discretization is task-specific". The per-session scope and the 40/60 split follow the instructions. The decision to treat *all* frames as valid is never stated or justified; "valid" in the notes means only "finite".

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three classes: `0` below the 40th percentile, `1` between the 40th and 60th inclusive, `2` above the 60th — thresholds computed per session. The array is initialised to `1`, so **bins that contain no camera frame at all silently become class 1 (the middle class)** rather than a distinct category. The instructions require a fourth class, `3: not visible`, and `output_values[3]` accordingly has only three entries. Because no likelihood filter is applied, "visible" effectively never occurs as a concept: the observed distribution is `{lt_40pct 0.392, 40to60pct 0.216, gt_60pct 0.392}`, i.e. almost exactly the 40/20/40 split you get by construction, with roughly 2% of bins (frameless ones) folded into the middle class. The reference, which discards the ~89% of frames where the tracker's likelihood is below 0.5, ends up with ~75% of bins in the "not visible" class — a completely different label distribution.

ii.
```python
final_outputs = []
for choice, outcome, early, binned_y in session_trial_output:
    ycat = np.full(len(bin_centers), 1, dtype=np.int64)
    finite = np.isfinite(binned_y)
    ycat[finite & (binned_y < q40)] = 0
    ycat[finite & (binned_y > q60)] = 2
    ycat[finite & (binned_y >= q40) & (binned_y <= q60)] = 1
    out = np.zeros((4, len(bin_centers)), dtype=np.int64)
    ...
    out[3, :] = ycat
```

```python
'output_values': [
    ...
    ['lt_40pct', '40to60pct', 'gt_60pct'],
],
```

iii. Step 5 decision 7 as quoted above. The AI noticed in the trajectory (step ~225) that "tongue categories are..." off in the sample run, and Step 8 records the lowest accuracy of all four outputs (`tongue_y_position` validation 0.4531 on sample, 0.5368 on the full run, against 0.333 chance for three classes), but Steps 10 and 12 — where the instructions require investigating exactly this — were never completed, so the missing fourth class and the missing visibility filter were never revisited.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps share the session-absolute clock with the spikes and events, so frames are selected by a boolean mask on `[go - 2.5, go + 1.5)` and assigned to bins by `np.digitize(t - go, edges)` — the exact go-relative grid, with no interpolation or offset correction. This is correct in itself, but it is *not* the grid the neural data actually sits on: the firing rates were snapped to the session-global grid (see 2-d), so the tongue output and the neural matrix are offset from one another by a trial-dependent 0–25 ms. The AI also does not account for the video being trial-gated (it stops during the inter-trial interval), which is why a few percent of leading bins have no frames — those bins are absorbed into the middle class rather than flagged.

ii.
```python
mask = (tongue_ts >= start) & (tongue_ts < stop)
yt = tongue_y[mask]
tt = tongue_ts[mask] - go
...
inds = np.digitize(tt, edges) - 1
ok = (inds >= 0) & (inds < len(bin_centers)) & np.isfinite(yt)
```

iii. Not explicitly justified; the implicit rationale is Step 5's shared `bin_centers` time axis. The `--show-processing` plots (`processing_<session_id>.png`) overlay the tongue category trace with the neural raster for trial 0 as the AI's visual alignment check, but they plot only one trial per session and would not reveal a 12 ms mean jitter.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases, handled with mixed success:
  - **Session never quality-controlled** (`classification` is NaN for all units): `decode_arr` stringifies NaN to `'nan'`, nothing matches `'good'`, and the session returns `None`. This correctly drops `sub-440958_ses-20190216T162508` and yields 173 sessions.
  - **`photostim_onset` / `photostim_duration` == `'N/A'`**: parsed to NaN and short-circuited to an all-zero vector. Handled correctly (though moot, given 4-b).
  - **`units/is_good_trials` with a column count that does not match the trial table** (9 sessions): the check is disabled for those sessions. This is the *symptom* of partial ephys coverage, and disabling the check discards the signal rather than acting on it.
  - **Trials with no spike data at all** (outside `units/obs_intervals`, plus `free_water` trials): not handled. 3,511 trials are emitted with an all-zero neural matrix and fully populated labels, which the verifier flags and the AI wrongly dismissed as a stale log.
  - **Bins with no visible / no recorded tongue frame**: NaN frames are excluded from the bin mean, but a bin left with no value is assigned class 1 (the middle class) rather than an explicit "missing" class; low-confidence frames are not treated as missing at all.
  - One latent hazard: if a trial had no `sample_start_times` event, `tone_rel` would be NaN and the entire `time_from_tone_onset` row would be NaN. Every trial in the dataset has one, so it never fires, but there is no guard.

ii.
```python
def decode_arr(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode())
        else:
            out.append(str(x))
    return np.array(out, dtype=object)
```

```python
if len(good_idx) == 0:
    return None
```

```python
is_good_trials = np.asarray(f['units']['is_good_trials'][()], dtype=bool) if ('is_good_trials' in f['units'] and f['units']['is_good_trials'].shape[1] == len(trial['start_time'])) else None
```

```python
def parse_optional_float(x):
    if isinstance(x, str):
        if x in ('N/A', 'nan', ''):
            return np.nan
        return float(x)
    return float(x)
```

```python
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
finite = np.isfinite(binned_y)
```

```python
region = np.array(['unknown' if (x is None or x == 'nan' or x == '') else x for x in region], dtype=object)
```

iii. Step 6: "Handle missing data appropriately (consult references, use sensible defaults, document)". Step 4 anticipated the session drop. For the all-zero trials, the trajectory records the (mistaken) resolution at step ~310: "the warnings in `verification_full_out.txt` are stale from an earlier run/file, not reflective of the current converted dataset" — reached by inspecting `data['neural'][0][...]` when the verifier's "Session 1" is the second session, whose `obs_intervals` covers only 160 of 480 trials. The AI re-ran verification three times, obtained the same warnings each time, and still concluded at step 167 that only "some documentation/cleanup incompleteness and an unresolved discrepancy between stale verification warnings and direct inspection" remained.

## 10-a. What are the most time-consuming steps of the code?

i. The AI instrumented per-session timing and printed it during conversion. The full run took 439 s of session processing (mean 2.54 s/session, max 6.25 s), plus pickling of the 12.5 GB output. The dominant cost is the per-unit `np.histogram` over the **entire session** grid in `process_session` — for each good unit this bins every spike in the recording, not just those in the trial windows — followed by reading the full `spike_times` buffer and the `(n_frames, 3)` tongue array, and then the Python per-trial loop that builds inputs, outputs and tongue bins one trial at a time. Notably the AI's own estimate was off: Step 7 predicted "~1.5 s/session ... ~4.5 min for 174 sessions" against an actual ~7.3 min.

ii.
```python
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
```

```python
t0 = time.time()
...
'elapsed_sec': time.time() - t0,
print(f'processed {session_id}: trials={info["n_trials"]} good_units={info["n_good_units"]} time={info["elapsed_sec"]:.2f}s')
```

iii. Step 6: "Code inefficiencies identified: Current implementation loops over good units within each trial and may be slow for full conversion. Code speedups added: Used direct NumPy histogram binning and avoided repeated file opens." Step 7 records the estimate and the conclusion that the run would fit inside the 15-minute budget, which it did. The bottleneck was measured but not re-attributed after the optimisation.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four loops remain, three of them avoidably so:
  - **Tone lookup** (`for i in range(n_trials)`): rebuilds a full boolean mask over every `sample_start_times` event for every trial — O(n_trials × n_events). A single `np.searchsorted` replaces it, as the reference does.
  - **Per-unit binning**: unavoidably a loop over units (spike arrays are ragged), but the AI bins the whole session per unit instead of vectorising over trial edges; the reference flattens all trial edge times into one array and does one `searchsorted` per unit, covering all trials at once.
  - **Main per-trial loop**: builds `time_from_tone`, the photostim vector, the outcome/early/choice scalars and the tongue binning one trial at a time. All four input/output constructions are pure array arithmetic over `(n_trials, 80)` and vectorise directly; the reference computes them as whole-session matrix expressions.
  - **Second per-trial loop** for tongue discretisation: also vectorisable as one `np.digitize` over the stacked `(n_trials, 80)` array.
  - Minor: `decode_arr` is a Python loop over every element of every text column, called on all trial columns and on `classification`/`anno_name` (up to 3,191 units per session).

ii.
```python
sample_times = np.full(n_trials, np.nan, dtype=float)
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
    if len(hits):
        sample_times[i] = hits[0]
```

```python
for i in range(n_trials):
    ...
    inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
    inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
    inp = np.stack([inp0, inp1], axis=0)
```

```python
for choice, outcome, early, binned_y in session_trial_output:
    ycat = np.full(len(bin_centers), 1, dtype=np.int64)
    ...
```

iii. Step 6 acknowledges the general problem ("Current implementation loops over good units within each trial and may be slow") and the fix applied ("direct NumPy histogram binning"), but the remaining per-trial loops are never identified. The implicit justification is that the measured runtime (~7.3 min) fit the instructions' 15-minute budget, so no further optimisation was pursued.

## 10-c. What processing does the code repeat multiple times?

i. Three kinds of repetition:
  - **Binning far more time than is needed**: every good unit is binned across the whole session, including all inter-trial intervals. For the two sessions measured, 41% and 65% of the resulting bins are never sliced into any trial, and the transient `global_rates` array reaches ~134,000 bins × n_units.
  - **Two passes over the trials**: the per-trial loop stores `binned_y` in `all_binned_y` and in `session_trial_output`, then a second loop re-walks every trial to apply the percentile thresholds. This second pass is a consequence of needing session-level percentiles; the reference avoids it by computing the class edges from an independent whole-session bin grid before the per-trial loop.
  - **Trivial re-creation inside the loop**: `out_outcome_map` and `out_early_map` are rebuilt once per trial, and the `finite &` mask is recomputed three times per trial in the discretisation.
  - The lick arrays are also re-scanned in full for every trial inside `find_choice_from_licks`.

ii.
```python
session_t0 = float(np.min(go_times) + edges[0])
session_t1 = float(np.max(go_times) + edges[-1])
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
```

```python
all_binned_y.append(binned_y)
...
session_trial_output.append([choice, outcome, early, binned_y])
...
for choice, outcome, early, binned_y in session_trial_output:
    ycat = np.full(len(bin_centers), 1, dtype=np.int64)
```

```python
out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_early_map = {'no early': 0, 'early': 1}
```

iii. The whole-session pre-binning was a deliberate optimisation, documented in Step 6 as "Used direct NumPy histogram binning and avoided repeated file opens" — trading redundant computation for a simpler inner loop. The two-pass structure for tongue discretisation is inherent to per-session percentiles and is not commented on. Each NWB file is opened exactly once, which the AI does note as an explicit goal.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three items:
  - **The `photostimulation_on` input**: computed for all 94,364 trials and identically zero everywhere (see 4-b), so it contributes nothing to the decoder — the full cost of parsing and constructing it is wasted, and one of the two input channels is dead.
  - **Session-wide firing rates outside the trial windows**: 41–65% of the `global_rates` matrix is binned and then never sliced (measured on two sessions), along with the accompanying `global_centers` array which is computed and never used at all.
  - **Retained but unused intermediates**: `all_binned_y` duplicates the `binned_y` arrays already held in `session_trial_output`; `info['elapsed_sec']` and `info['n_good_units']` are carried through but dropped at assembly; `metadata['bin_centers_s']` is stored but is fully determined by `off_start`, `off_end` and `time_bin_size`.
  - Conversely, some work that *should* have been done was skipped: brain region names are stored as full CCF strings (`'Agranular insular area, dorsal part, layer 5'`), giving 293 distinct "regions" rather than the reference's collapsed ~coarse labels.

ii.
```python
global_centers = (global_edges[:-1] + global_edges[1:]) / 2   # computed, never used
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
```

```python
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)   # always all zeros
inp = np.stack([inp0, inp1], axis=0)
```

```python
all_binned_y.append(binned_y)
session_trial_output.append([choice, outcome, early, binned_y])
```

iii. None of this is discussed in CONVERSION_NOTES; Steps 10, 12 and 13, where the instructions require this kind of review and cleanup, are left `IN PROGRESS` / `NOT STARTED`, and the required `/app/README.md` and `/app/cache/` were never created. The whole-session binning waste is the intentional cost of the Step 6 speed-up; the dead photostim channel is an undetected bug the AI twice flagged and twice deferred.
