# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session under `/app/data/sub-<subject_id>/`. The AI discovers every session with a single sorted glob over that layout and processes each file exactly once. Rather than using `pynwb`, it opens each file directly with `h5py` and reads only the groups it needs (`general/subject`, `units`, `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`). It switched from `pynwb` to `h5py` after measuring that `pynwb` took ~24 s/session versus ~0.2 s. 174 files are found; 173 are converted (one is dropped for having no quality-controlled units).

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
print(f"Found {len(nwb_files)} NWB files")
...
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, show_processing=show, session_idx=i)
```

```python
def load_session_h5py(nwb_path):
    """Load session data from NWB file using h5py for speed."""
    data = {}
    with h5py.File(nwb_path, 'r') as f:
        data['subject_id'] = f['general']['subject']['subject_id'][()].decode() ...
        units_grp = f['units']
        data['classification'] = np.array([x.decode() ... for x in units_grp['classification'][()]])
        data['anno_name'] = np.array([x.decode() ... for x in units_grp['anno_name'][()]])
        spike_times_data = units_grp['spike_times'][()]
        spike_times_index = units_grp['spike_times_index'][()]
        ...
        trials_grp = f['intervals']['trials']
        ...
        acq = f['acquisition']
        be = acq['BehavioralEvents']
        data['go_times'] = be['go_start_times']['timestamps'][()]
```

iii. From CONVERSION_NOTES Step 6: "Uses h5py for fast NWB loading (12x speedup vs pynwb)"; Step 2 documents the `sub-*/…nwb` layout and 174 files / 28 subjects. The trajectory (steps 47, 51) shows the AI first wrote a `pynwb` loader, measured 24 s/session (~70 min projected for the full dataset, over the 15-minute budget in the instructions), and replaced it with direct HDF5 reads. NWB files are HDF5, so reading the datasets directly returns identical arrays.

## 1-b. How are the data split into subjects (mice)?

i. Each NWB file names its animal in `general/subject/subject_id` (a numeric string such as `'440956'`). That value is read per session; a running dictionary assigns each new id the next index, so `subjects` is the list of unique ids in first-encountered (i.e. file-sorted) order and `subject_idx` holds each session's index into that list. The result is 28 subjects with 3–10 sessions each.

ii.
```python
data['subject_id'] = f['general']['subject']['subject_id'][()].decode() if isinstance(...) else str(...)
```

```python
subj_id = result['subject_id']
if subj_id not in subject_to_idx:
    subject_to_idx[subj_id] = len(subjects_list)
    subjects_list.append(subj_id)
...
all_subject_idx.append(subject_to_idx[subj_id])
...
'subjects': subjects_list,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 identifies `subject_id` as the animal identifier in the file and reports 28 subjects, which matches the 28 `sub-*` directories and the dandiset metadata. Because the directory name is derived from `subject_id`, no separate grouping step is needed.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session; no grouping or splitting is performed. Session order in the output is the sorted file order (which is chronological within subject, since filenames embed the acquisition timestamp). The AI does not carry any session identifier into the saved dictionary — `metadata` has no `session_info` field and session identity is only recoverable from the conversion log.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
for i, nwb_path in enumerate(nwb_files):
    print(f"\nProcessing session {i+1}/{len(nwb_files)}: {os.path.basename(nwb_path)}")
    result = process_session(nwb_path, show_processing=show, session_idx=i)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

```python
'metadata': {
    'task_description': 'Audio delayed-response licking task. ...',
    'time_bin_size': BIN_WIDTH * 1000,
    'temporal_alignment_event': 'Go cue onset',
    'off_start': ALIGN_START,
    'off_end': ALIGN_END,
    'dataset': 'Mesoscale Activity Map (MAP) Dataset, DANDI:000363',
    'bin_width_s': BIN_WIDTH,
    'n_timebins': N_TIMEBINS,
}
```

iii. CONVERSION_NOTES Step 2: "NWB files organized as `/app/data/sub-{id}/sub-{id}_ses-{datetime}_behavior+ecephys[+ogen].nwb`", i.e. the file boundary is the session boundary, so nothing has to be inferred. Step 4 records the reconciliation of 174 files against the paper's 173 sessions (the extra one has no good units).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioural trial. Per-trial columns (`start_time`, `stop_time`, `trial_instruction`, `outcome`, `early_lick`) are read as parallel arrays, and the go cue for trial *t* is taken as element *t* of `BehavioralEvents/go_start_times`, i.e. the two are assumed to be index-aligned. The code never asserts that `len(go_times) == len(trial_start)` (this does hold in all 174 files, so no mismatch occurs in practice).

ii.
```python
trials_grp = f['intervals']['trials']
data['trial_start'] = trials_grp['start_time'][()]
data['trial_stop'] = trials_grp['stop_time'][()]
data['trial_instruction'] = np.array([...for x in trials_grp['trial_instruction'][()]])
data['outcome'] = np.array([... for x in trials_grp['outcome'][()]])
data['early_lick'] = np.array([... for x in trials_grp['early_lick'][()]])
data['go_times'] = be['go_start_times']['timestamps'][()]
```

```python
n_trials = len(data['trial_start'])
go_times = data['go_times']
...
go_times_valid   = go_times[valid_indices]
trial_starts_valid = data['trial_start'][valid_indices]
outcome_valid    = data['outcome'][valid_indices]
```

iii. CONVERSION_NOTES Step 2 lists the trials table columns as the per-trial source. The trajectory (steps 9–12) shows the AI inspecting the trials table and BehavioralEvents and concluding that `go_start_times` gives "the go cue onset for alignment", one per trial. Total trials found: 94,990 before filtering.

## 1-e. How are trials filtered based on quality controls?

i. Exactly one trial filter is applied: a trial is kept only if its whole analysis window `[go - 2.5, go + 1.5]` lies inside the session's spike-recording period, where that period is approximated by the **global minimum and maximum spike time across all good units**, padded by one bin width. A session is dropped if fewer than 2 trials survive (never triggered) or if it has no good units (1 session). No behavioural quality filter is applied: early-lick, `ignore`, `auto_water` and `free_water` trials are all kept. This removed 1,700 of 94,990 trials (27 sessions affected, up to 321 trials in one session), leaving 93,290.

The AI explicitly examined and rejected two alternatives available in the file: per-unit `units/obs_intervals` and per-unit `units/is_good_trials`. As a consequence, **2,446 trials (2.6%) in the delivered dataset contain literally zero spikes across all neurons** — the verifier emits a warning for every one of them. The AI investigated these, concluded they were "real recording gaps … genuine data quality issues, not conversion bugs", and left them in. I checked this against the raw files: in `sub-456772_ses-20191119T115109` the 24 all-zero trials are *exactly* the 24 `free_water == 1` trials, and likewise 22/22 in the next session. The AI had looked at `free_water` (trajectory step 102) and decided "they're valid behavioral trials" and should not be excluded.

ii.
```python
def get_valid_trial_mask(spike_times_list, go_times, align_start=ALIGN_START, align_end=ALIGN_END):
    """Determine which trials have valid neural recording coverage. ...
    We use the overall max spike time across all good units as the recording end."""
    max_spike = 0
    min_spike = float('inf')
    for spikes in spike_times_list:
        if len(spikes) > 0:
            max_spike = max(max_spike, spikes[-1])
            min_spike = min(min_spike, spikes[0])
    if max_spike == 0:
        return np.zeros(len(go_times), dtype=bool)
    valid = (go_times + align_start >= min_spike - BIN_WIDTH) & (go_times + align_end <= max_spike + BIN_WIDTH)
    return valid
```

```python
valid_mask = get_valid_trial_mask(spike_times_good, go_times)
valid_indices = np.where(valid_mask)[0]
n_valid = len(valid_indices)
if n_valid < 2:
    print(f"  Skipping {basename}: only {n_valid} valid trials")
    return None
if n_valid < n_trials:
    print(f"  Filtering: {n_valid}/{n_trials} trials have neural coverage")
```

iii. CONVERSION_NOTES Step 3: "Filter trials where neural recording doesn't cover the analysis window. Each unit has is_good_trials but this varies per unit; we use recording coverage instead." The trajectory (steps 65–68) documents the discovery: in `sub-440956_ses-20190208T133600` spikes stop at 1107 s while go cues run to 3708 s, so only 159/480 trials have neural data; `is_good_trials` was rejected because its length (160) does not match the trial count (480). Step 67 states the chosen rule verbatim: "for each session, find the maximum time covered by the neural recording, and exclude trials whose go cue + 1.5s exceeds this time."

For the residual all-zero trials, CONVERSION_NOTES Step 10 says: "These occur because individual probes may end recording before the behavioral session ends. Our trial filter uses the max spike time across ALL good units, but individual probes may end earlier… This is a genuine data quality issue, not a conversion bug." Trajectory step 104 adds the "recording gaps (probe disconnection, software pauses)" theory. Early-lick and `ignore` trials are kept because they are required decoder outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (plus `units/spike_times_index` to split the ragged buffer into per-unit arrays), restricted to units with `units/classification == 'good'`. The go cue times (`BehavioralEvents/go_start_times`) supply the alignment point that positions the bin edges.

ii.
```python
spike_times_data = units_grp['spike_times'][()]
spike_times_index = units_grp['spike_times_index'][()]
n_units = len(data['classification'])
all_spike_times = []
prev_idx = 0
for i in range(n_units):
    end_idx = spike_times_index[i]
    all_spike_times.append(spike_times_data[prev_idx:end_idx])
    prev_idx = end_idx
data['all_spike_times'] = all_spike_times
```

```python
good_indices = np.where(good_mask)[0]
spike_times_good = [data['all_spike_times'][i] for i in good_indices]
...
neural_trials = compute_firing_rates_all_trials(spike_times_good, go_times_valid)
```

iii. CONVERSION_NOTES Step 5 maps `units.spike_times` → `neural` ("Bin into 50ms firing rates, align to go cue, window [-2.5, 1.5]"). Spike times are the only neural representation in the file; Step 1 notes "Spike times in NWB are in absolute session time (not relative to go cue)".

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each trial the 81 absolute bin edges are formed by adding the go-cue time to the fixed relative edge grid; for each good unit the spikes inside the window are located with two `searchsorted` calls and counted with `np.histogram` against those edges; counts are divided by the 50 ms bin width. No smoothing, normalisation, or baseline subtraction is applied (unlike the reference code's sliding Gaussian-like histogram, which the AI deliberately did not copy). A unit with no spikes in the window keeps its pre-allocated row of zeros.

ii.
```python
bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
results = []
for t in range(n_trials):
    go_time = go_times[t]
    abs_bin_edges = bin_edges_rel + go_time
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    for i, spikes in enumerate(spike_times_list):
        if len(spikes) == 0:
            continue
        if spikes[-1] < abs_bin_edges[0] or spikes[0] > abs_bin_edges[-1]:
            continue
        left = np.searchsorted(spikes, abs_bin_edges[0])
        right = np.searchsorted(spikes, abs_bin_edges[-1])
        spikes_in_window = spikes[left:right]
        if len(spikes_in_window) == 0:
            continue
        counts, _ = np.histogram(spikes_in_window, bins=abs_bin_edges)
        fr[i] = counts / bin_width
    results.append(fr)
```

iii. CONVERSION_NOTES Step 1: "The reference code computes firing rates using sliding_histogram… Our task requires 50ms bins (not 40ms from reference) and non-overlapping bins", and Step 10 Check 3: "We use 50ms non-overlapping bins; reference uses 40ms sliding window with 3.4ms stride. Different per task spec." Dividing counts by bin width reproduces the reference's `rate=True` behaviour (`binSpikes / bin_width`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are retained; no thresholds are applied to any individual quality metric, and the older `unit_quality` label and the per-unit `is_good_trials` flags are not used. A session with zero good units is dropped entirely. This keeps 69,453 of 272,227 units (25.5%), a mean of 401 per session, and drops exactly one session (`sub-440958_ses-20190216T162508`, whose classification column is unlabelled).

ii.
```python
good_mask = data['classification'] == 'good'
n_good = np.sum(good_mask)
if n_good == 0:
    print(f"  Skipping {basename}: no good units")
    return None
good_indices = np.where(good_mask)[0]
spike_times_good = [data['all_spike_times'][i] for i in good_indices]
anno_names = data['anno_name'][good_indices]
```

iii. CONVERSION_NOTES Step 3 Curation: "Use classifier-based QC: classification == 'good' in NWB files. Region-specific logistic regression classifiers trained on manual curation" — i.e. the verdict of the QC classifier described in `ChenLiuEtAl2023_SpikeSortingQC.pdf`, which is also what the reference code selects via `qc_mode='classifier'` (Step 1, Step 10 Check 3: "We use classification=='good'; reference uses qc_mode='classifier'. Same filter."). Step 4 reconciles the resulting counts against the papers: 173 sessions (exact match) and 69,453 good units vs the white paper's 69,943 (99.3%), attributed to a data-version difference; the paper's 25.9% good-unit fraction matches the observed 25.5%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times, go cues, camera timestamps and photostim events are all on the same session-absolute clock, so alignment requires no resampling or offset correction: the fixed relative edge grid is simply added to each trial's go-cue time to give that trial's absolute window, and spikes are binned against those absolute edges.

ii.
```python
data['go_times'] = be['go_start_times']['timestamps'][()]
...
go_times_valid = go_times[valid_indices]
neural_trials = compute_firing_rates_all_trials(spike_times_good, go_times_valid)
```

```python
for t in range(n_trials):
    go_time = go_times[t]
    abs_bin_edges = bin_edges_rel + go_time
```

iii. CONVERSION_NOTES Step 5 Key Decision 2 ("Time window: -2.5 to 1.5s relative to go cue (80 bins)") and Step 10 Check 3 ("Temporal alignment: We align to go cue; reference uses task_cue_time. Same alignment."). Step 1 records that spike times are in absolute session time, so only the go-cue lookup is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 80 non-overlapping 50 ms bins spanning −2.5 s to +1.5 s relative to the go cue. The relative edge grid is derived once from module-level constants and reused for every trial and every session, so every trial has exactly 80 timepoints (confirmed by the verifier: T min = max = 80 across all 173 sessions). `metadata['time_bin_size']` is recorded as 50.0 ms and `off_start`/`off_end` as −2.5/1.5.

ii.
```python
BIN_WIDTH = 0.05  # 50ms bins
ALIGN_START = -2.5  # seconds before go cue
ALIGN_END = 1.5    # seconds after go cue
N_TIMEBINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
```

```python
n_bins = int((align_end - align_start) / bin_width)
bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
```

iii. Directly from the Decoder Task section of the instructions. CONVERSION_NOTES Step 5 Key Decisions 1–2: "Bin width: 50ms as specified in task (not 40ms from reference code)"; "Time window: -2.5 to 1.5s relative to go cue (80 bins)". Step 4 lists both as deliberate, spec-driven departures from the reference code's 40 ms / ±3 s settings.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (the tone onsets) together with `trials.start_time`, `trials.stop_time` and the trial's go cue. For each trial the AI takes the **first** sample-start event that falls inside `[trial_start, trial_stop)` and stores its time relative to the go cue; if no tone is found in the trial, a default of −1.85 s is used.

The AI observed that some trials carry more than one tone ("There are more sample_starts (405) than trials (368)") but did not investigate why, and did not consider taking the last tone before the go cue. The reason is that a lick during the sample or delay epoch replays that epoch, so a trial can contain up to ~13 tones. On such trials the first tone is 2.6–2.9 s (up to 4.3 s in the sessions I checked, and up to ~10.4 s across the dataset) before the go cue, whereas the last tone is at the canonical 1.85 s. This affects ~2–3% of trials, and shows up in the verifier output as a `time_from_tone_onset` maximum of 11.9 s in one session.

ii.
```python
def find_sample_starts_for_trials(trial_starts, trial_stops, sample_starts, go_times):
    """Find tone onset time relative to go cue for each trial."""
    n_trials = len(trial_starts)
    tone_rel_go = np.full(n_trials, -1.85, dtype=np.float64)  # default
    for t in range(n_trials):
        idx_start = np.searchsorted(sample_starts, trial_starts[t])
        idx_end = np.searchsorted(sample_starts, trial_stops[t])
        if idx_start < idx_end:
            tone_rel_go[t] = sample_starts[idx_start] - go_times[t]
    return tone_rel_go
```

iii. CONVERSION_NOTES Step 5: "sample_start_times → input[0] (time_from_tone): Continuous: t - tone_onset for each timepoint". Trajectory step 45: "Tone onset relative to go cue: most common is -1.85s (sample period 0.65s + delay 1.2s = 1.85s), but varies due to variable delay periods. There are more sample_starts (405) than trials (368) - some extra sample events. For the tone onset time input: I need to compute for each trial the time of the **first** sample_start within that trial." The −1.85 default is the modal sample-to-go interval.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The value of a bin is its centre (relative to the go cue) minus the tone's time relative to the go cue — i.e. bin centre plus the tone→go interval — giving seconds elapsed since the tone at each bin centre. It is stored as a `float32` continuous time series in row 0 of the `(2, 80)` input array. No clipping, normalisation or truncation at the tone is applied, so values are negative for bins before the tone.

ii.
```python
bin_centers_rel = np.arange(n_bins) * bin_width + align_start + bin_width / 2
...
for t in range(n_trials):
    time_from_tone = (bin_centers_rel - tone_rel_go[t]).astype(np.float32)
    ...
    input_data = np.vstack([time_from_tone.reshape(1, -1),
                            photostim.reshape(1, -1)])
```

iii. CONVERSION_NOTES Step 5 Key Decision 9: "Time from tone onset: Continuous, = bin_center_time - sample_start_time (relative to go cue)." The instructions specify this input as "continuous, time-varying". The AI's processing-plot review (Step 7) confirms "Time from tone onset increases monotonically".

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on the same go-cue-referenced grid as the firing rates: `bin_centers_rel` is the set of centres of the very same 80 bins whose edges are used for spike binning, and the tone time is expressed relative to that trial's go cue. Bin *k* of the input therefore covers the same interval as bin *k* of the neural array by construction.

ii.
```python
# neural:
bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
abs_bin_edges = bin_edges_rel + go_time
# input:
bin_centers_rel = np.arange(n_bins) * bin_width + align_start + bin_width / 2
time_from_tone = (bin_centers_rel - tone_rel_go[t]).astype(np.float32)
```

```python
tone_rel_go[t] = sample_starts[idx_start] - go_times[t]
```

iii. Implicit in Step 5's mapping (everything is expressed relative to the go cue). Step 10's sanity checks include "Input data: Time from tone onset increases monotonically within each trial", and the `--show-processing` plots overlay the input on a time axis whose zero is the go cue.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `BehavioralEvents/photostim_start_times` and `photostim_stop_times` — the absolute onset and offset timestamps of each photostimulation event in the session. Sessions without those datasets (non-VGAT-ChR2-EYFP mice) get empty arrays and therefore an all-zero photostim input. The trials-table columns `photostim_onset` / `photostim_duration` were inspected but not used. I verified the two sources agree: in `sub-456772_ses-20191119T115109` the 94 events match the 94 stim rows of the trials table to ~1 ms in onset and give identical 0.5 s durations.

ii.
```python
if 'photostim_start_times' in be:
    data['photostim_starts'] = be['photostim_start_times']['timestamps'][()]
    data['photostim_stops'] = be['photostim_stop_times']['timestamps'][()]
else:
    data['photostim_starts'] = np.array([])
    data['photostim_stops'] = np.array([])
```

iii. CONVERSION_NOTES Step 5 Key Decision 8: "Photostim input: Binary time series based on BehavioralEvents photostim_start/stop_times." Trajectory step 27 records the comparison of the two sources: "`photostim_onset` in trials table is relative to trial start (onset_val + trial_start = absolute photostim time); BehavioralEvents photostim_start_times gives absolute times" — the AI chose the absolute event times, which need no string parsing or re-referencing.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1 float32) time series: a bin is 1 if its **centre**, expressed in absolute session time, falls in `[stim_start, stim_stop)` for any photostim event in the session, else 0. It occupies row 1 of the input array. Because every event in the session is tested against every trial's bins, stimulation that spills into a neighbouring trial's window would also be captured.

ii.
```python
photostim = np.zeros(n_bins, dtype=np.float32)
abs_bin_centers = bin_centers_rel + go_time
for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
    mask = (abs_bin_centers >= ps_start) & (abs_bin_centers < ps_stop)
    photostim[mask] = 1.0
```

iii. The instructions require "Whether photostimulation is on at every time point (discrete, time-varying)", and the format spec says a time input should be represented as a binary time series. CONVERSION_NOTES Step 2 and trajectory step 27 establish the expected timing: "Photostim occurs at -1.200s relative to go cue… duration is 0.5s. So photostim is ON from -1.2 to -0.7 relative to go cue". Step 7's plot review confirms: "Photostim input shows correct timing (~-1.2 to -0.7s)". Overall 19.5% of trials are stimulated, which the AI reconciles with the paper's "~25% of trials in stim sessions" (only 17 of 28 mice are VGAT-ChR2-EYFP).

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The comparison is done in absolute session time: the go-cue-referenced bin centres are converted to absolute times by adding the trial's go cue, and tested against the absolute event timestamps. Since those centres are the centres of the same 80 bins used for spike counting, the photostim row is aligned bin-for-bin with the neural array.

ii.
```python
abs_bin_centers = bin_centers_rel + go_time
for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
    mask = (abs_bin_centers >= ps_start) & (abs_bin_centers < ps_stop)
    photostim[mask] = 1.0
```

iii. All NWB streams share one clock (CONVERSION_NOTES Step 1: spike times are in absolute session time; the same holds for behavioural event timestamps), so no interpolation or offset correction is needed. Step 10 sanity check: "Photostim timing: Verified photostim occurs at -1.2s relative to go cue with 0.5s duration."

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From the measured lick events themselves: `BehavioralEvents/left_lick_times` and `right_lick_times`, together with the trial's go cue. The choice is the side of the **first lick in the 1.5 s response window after the go cue**; if neither spout is licked in that window, the trial is coded "no lick". The trials-table `trial_instruction` and `outcome` columns are *not* used to derive choice (the reference derives it as instruction × outcome instead). The two derivations agree closely in aggregate: the AI's no-lick fraction (0.151) is identical to its `ignore` fraction (0.151), and left/right split 0.429/0.419.

ii.
```python
data['left_lick_times'] = be['left_lick_times']['timestamps'][()]
data['right_lick_times'] = be['right_lick_times']['timestamps'][()]
```

```python
def get_lick_choices(go_times, left_lick_times, right_lick_times,
                     response_window=LICK_RESPONSE_WINDOW):
    """Compute lick choice for all trials. Returns array of int."""
    choices = np.full(len(go_times), 2, dtype=np.int64)  # default: no lick
    for t, go_time in enumerate(go_times):
        left_idx = np.searchsorted(left_lick_times, go_time)
        first_left = left_lick_times[left_idx] if left_idx < len(left_lick_times) and left_lick_times[left_idx] < go_time + response_window else float('inf')
        right_idx = np.searchsorted(right_lick_times, go_time)
        first_right = right_lick_times[right_idx] if right_idx < len(right_lick_times) and right_lick_times[right_idx] < go_time + response_window else float('inf')
        if first_left < first_right:
            choices[t] = 0  # left
        elif first_right < float('inf'):
            choices[t] = 1  # right
    return choices
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "Choice determination: First lick direction within 1.5s after go cue." Trajectory steps 42–43: "Lick direction can be determined from first lick after go cue — 'hit' trials match instruction, 'ignore' trials have no lick… 'hit' = correct lick (matches instruction), 'miss' = wrong direction lick (opposite to instruction), 'ignore' = no lick at all. Choice is determined by first lick direction after go cue." The 1.5 s window is the response window given in the instructions/Decoder Task. Step 10 sanity check: "Output data: Choice matches first lick direction after go cue."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as `0` left, `1` right, `2` no lick, and written into row 0 of the per-trial `(4, 80)` int64 output array, repeated identically across all 80 bins so that all four outputs share one array. `output_values[0] = ['left', 'right', 'no_lick']` names the codes.

ii.
```python
output_data = np.zeros((4, n_bins), dtype=np.int64)
output_data[0, :] = choices[t]
```

```python
'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
'output_values': [
    ['left', 'right', 'no_lick'],
    ...
],
```

iii. The left=0 / right=1 / no-lick=2 coding follows the instructions' output specification ("Lick direction choice (left, right, no lick, per-trial)"). Choice is a per-trial quantity, so it is broadcast across bins to satisfy the target format's `(n_output, n_timepoints)` layout (CONVERSION_NOTES Step 5: "Per-trial (replicated)").

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the NWB trials table, which already contains the strings `'hit'`, `'miss'` and `'ignore'` — exactly the three categories the instructions ask for. No derivation from licks or rewards is needed.

ii.
```python
data['outcome'] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in trials_grp['outcome'][()]])
...
outcome_valid = data['outcome'][valid_indices]
```

iii. CONVERSION_NOTES Step 2 lists `outcome (hit/miss/ignore)` among the trials-table columns and Step 5 maps `trials.outcome → output[1]` directly. Trajectory step 10: "outcome is 'hit'/'ignore'/'miss'".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to `0` ignore, `1` miss, `2` hit through a fixed dictionary (with an unknown string silently defaulting to `0`), then written into row 1 of the output array, repeated across all 80 bins. Resulting distribution over the full dataset: ignore 0.151, miss 0.166, hit 0.684.

ii.
```python
def get_outcome_codes(outcomes):
    """Map outcome strings to integer codes."""
    mapping = {'ignore': 0, 'miss': 1, 'hit': 2}
    return np.array([mapping.get(o, 0) for o in outcomes], dtype=np.int64)
```

```python
output_data[1, :] = outcome_codes[t]
```

iii. The code assignment follows the instructions' ordering ("Outcome (ignore, miss, hit, per-trial)"). Step 5 lists "trials.outcome → output[1] (outcome): ignore=0, miss=1, hit=2, Per-trial (replicated)". CONVERSION_NOTES Step 12 sanity-checks the resulting distribution against the paper: "Performance (hit/(hit+miss)): 80.5% (paper: 83.2%) - close match."

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the NWB trials table, which holds the strings `'no early'` and `'early'`. No derivation from lick timestamps is attempted.

ii.
```python
data['early_lick'] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in trials_grp['early_lick'][()]])
...
early_lick_valid = data['early_lick'][valid_indices]
```

iii. CONVERSION_NOTES Step 2 lists `early_lick` as a trials-table column; trajectory step 10 confirms the two values ("early_lick is 'early'/'no early'"). The flag is explicit in the file, so no derivation is needed.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `0` for the exact string `'no early'` and `1` for anything else, then written into row 2 of the output array, repeated across all 80 bins. Dataset distribution: no 0.885, yes 0.115.

ii.
```python
def get_early_lick_codes(early_licks):
    """Map early_lick strings to integer codes."""
    return np.array([0 if e == 'no early' else 1 for e in early_licks], dtype=np.int64)
```

```python
output_data[2, :] = early_lick_codes[t]
```

iii. The coding follows the instructions ("Early lick (no, yes, per-trial)"); Step 5 maps "trials.early_lick → output[2] (early_lick): no=0, yes=1, Per-trial (replicated)". The early lick itself happens during the sample or delay epoch, i.e. inside the −2.5 s window, so a per-trial flag is still decodable from the extracted window.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: its `data` array is `(n_frames, 3)` — x, y, tracking likelihood — with matching `timestamps` at a ~3.4 ms stride (~294 Hz). Column 1 is taken as the y-position and column 2 as the visibility likelihood.

ii.
```python
bts = acq['BehavioralTimeSeries']
data['tongue_data'] = bts['Camera0_side_TongueTracking']['data'][()]
data['tongue_timestamps'] = bts['Camera0_side_TongueTracking']['timestamps'][()]
```

```python
tongue_y = tongue_data[:, 1]
tongue_conf = tongue_data[:, 2]
```

iii. CONVERSION_NOTES Step 2: "BehavioralTimeSeries: Camera0_side_TongueTracking (x, y, confidence at 0.0034s stride)". Trajectory step 11: "TongueTracking has x, y, and confidence (3rd column) - low confidence means tongue not visible"; step 42 adds the empirical check: "confidence is bimodal (~0 or ~1), threshold of 0.5 works well. Only ~10% of timepoints have visible tongue. Y-position for visible tongue: mean=280.98, range 210-327." This is the only tongue measurement in the file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) A frame counts as visible if its likelihood is > 0.5. (2) The two class edges are the 40th and 60th percentiles of the **raw y values of all visible frames in the session** (not of binned means, and non-visible frames are excluded entirely); if a session had no visible frame at all, both edges fall back to 0.0. (3) Within a trial, each 50 ms bin takes the mean y over the visible frames in that bin, and that mean is compared against the two edges.

Because the percentiles are taken over frames while the digitised quantity is a bin mean, the intended 40/20/40 split is not reproduced: over the delivered dataset the three visible classes hold 0.044 / 0.053 / 0.062 of all bins, i.e. 28% / 33% / 39% of the classified bins rather than 40% / 20% / 40%.

ii.
```python
TONGUE_CONFIDENCE_THRESHOLD = 0.5  # threshold for tongue visibility
...
tongue_y = tongue_data[:, 1]
tongue_conf = tongue_data[:, 2]

# Session-wide percentiles from visible frames
visible_mask = tongue_conf > TONGUE_CONFIDENCE_THRESHOLD
y_visible = tongue_y[visible_mask]
if len(y_visible) > 0:
    p40 = np.percentile(y_visible, 40)
    p60 = np.percentile(y_visible, 60)
else:
    p40 = 0.0
    p60 = 0.0
```

```python
vis_mask = bin_conf > TONGUE_CONFIDENCE_THRESHOLD
mean_y = np.mean(bin_y[vis_mask])
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: "Tongue y discretization: Per-session percentiles of y-position when visible (confidence > 0.5)." The instructions specify the percentiles be computed "over the session", and the AI's own measurement that the likelihood is bimodal (trajectory step 42) makes the exact 0.5 threshold immaterial. Restricting to visible frames is justified because the tracker still emits a position when the tongue is retracted.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes: `0` mean y < p40, `1` p40 ≤ mean y < p60, `2` mean y ≥ p60, `3` not visible. A bin is only classified at all if **more than 50% of the camera frames in that bin are visible**; otherwise it is left at the default class 3, as are bins containing no camera frames. This majority-visible rule is an additional criterion beyond the instructions (which define class 3 simply as "not visible") and is not mentioned in CONVERSION_NOTES. It is materially stricter than requiring one visible frame: 84.0% of all bins in the delivered data are class 3, against 75% for the reference solution, i.e. roughly a third of the bins that contain some visible tongue are labelled "not visible".

ii.
```python
trial_tongue_y = np.full(n_bins, 3, dtype=np.int64)  # default: not visible

for b in range(n_bins):
    idx_start = np.searchsorted(tongue_timestamps, abs_bin_edges[b])
    idx_end = np.searchsorted(tongue_timestamps, abs_bin_edges[b + 1])
    if idx_start >= idx_end or idx_start >= len(tongue_timestamps):
        continue
    bin_conf = tongue_conf[idx_start:idx_end]
    bin_y = tongue_y[idx_start:idx_end]
    visible_frac = np.mean(bin_conf > TONGUE_CONFIDENCE_THRESHOLD)
    if visible_frac > 0.5:
        vis_mask = bin_conf > TONGUE_CONFIDENCE_THRESHOLD
        mean_y = np.mean(bin_y[vis_mask])
        if mean_y < p40:
            trial_tongue_y[b] = 0
        elif mean_y < p60:
            trial_tongue_y[b] = 1
        else:
            trial_tongue_y[b] = 2
```

```python
'output_values': [..., ['below_p40', 'p40_to_p60', 'above_p60', 'not_visible']],
```

iii. The 0/1/2/3 scheme is taken verbatim from the instructions' Decoder Task ("0: < 40th percentile … 3: not visible"). CONVERSION_NOTES Step 7 records the plot review: "Tongue y mostly 'not visible' (3) with visible periods around licking", and trajectory step 42 justifies the ~90% not-visible rate ("Only ~10% of timepoints have visible tongue"). The majority-visible rule itself is not justified anywhere in the notes or trajectory.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. On the same go-cue-referenced grid as the firing rates. For each trial the absolute bin edges are `bin_edges_rel + go_time` — identical to the neural edges — and the frames belonging to bin *b* are found by `searchsorted` on the camera timestamps at edge *b* and edge *b+1*. Because the camera timestamps are on the same session clock as the spikes, no interpolation or offset correction is applied. Bins whose window contains no camera frame (the video is trial-gated, so the leading bins of short-trial-start trials are empty) fall through to class 3.

ii.
```python
bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
...
for t in range(n_trials):
    go_time = go_times[t]
    abs_bin_edges = bin_edges_rel + go_time
    trial_tongue_y = np.full(n_bins, 3, dtype=np.int64)
    for b in range(n_bins):
        idx_start = np.searchsorted(tongue_timestamps, abs_bin_edges[b])
        idx_end = np.searchsorted(tongue_timestamps, abs_bin_edges[b + 1])
        if idx_start >= idx_end or idx_start >= len(tongue_timestamps):
            continue
```

iii. Same rationale as for the other streams: everything in the NWB file shares one clock (CONVERSION_NOTES Step 1), so applying the identical bin grid to both streams guarantees that bin *k* of the tongue output covers the same interval as bin *k* of the firing rates. The `--show-processing` plots (Step 6/7) plot the discretised tongue output on the same go-cue time axis as the neural data to make any misalignment visible.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six cases, handled by a mix of exclusion and silent defaults:

- **Session never quality-controlled**: the classification strings are not `'good'` for any unit, so `n_good == 0` and the session is skipped (1 session, `sub-440958_ses-20190216T162508`).
- **Trials outside the spike recording**: excluded by the coverage mask (see 1-e); a session with fewer than 2 survivors would be dropped.
- **Units with no spikes at all / no spikes in the window**: skipped, leaving that neuron's row as zeros.
- **No tone event inside a trial**: the tone-to-go interval silently defaults to the modal −1.85 s.
- **Session with no photostim datasets**: empty arrays, giving an all-zero photostim input.
- **Frames with likelihood ≤ 0.5, and bins that are mostly or entirely non-visible**: represented as the explicit `not_visible` class rather than imputed. A session with no visible frame anywhere would silently get `p40 = p60 = 0.0` (and hence all visible bins classed as 2).
- An unrecognised `outcome` string would silently be coded as `ignore` (`mapping.get(o, 0)`).

What is *not* handled: the 2,446 trials (2.6%) that survive the coverage filter but contain zero spikes across every neuron. These are kept with their full behavioural labels (see 1-e).

ii.
```python
if n_good == 0:
    print(f"  Skipping {basename}: no good units")
    return None
...
if n_valid < 2:
    print(f"  Skipping {basename}: only {n_valid} valid trials")
    return None
```

```python
tone_rel_go = np.full(n_trials, -1.85, dtype=np.float64)  # default
```

```python
if len(y_visible) > 0:
    p40 = np.percentile(y_visible, 40)
    p60 = np.percentile(y_visible, 60)
else:
    p40 = 0.0
    p60 = 0.0
```

```python
if 'photostim_start_times' in be:
    ...
else:
    data['photostim_starts'] = np.array([])
    data['photostim_stops'] = np.array([])
```

iii. CONVERSION_NOTES Step 10 Check 5 (Edge cases): "Trials at end of recording: Filtered using neural coverage check; Sessions with 0 good units: Skipped (1 session); Trials with no lick: Coded as choice=2 (no_lick)." For the all-zero trials, Step 10 Check 1 argues they are "a genuine data quality issue, not a conversion bug" caused by individual probes ending early, and trajectory step 104 attributes them to "real recording gaps … probe disconnection/reconnection, recording software pauses". Trajectory step 102 explicitly considered and rejected excluding `free_water` trials: "The free_water trials (24 in this session) are worth noting but shouldn't be excluded - they're valid behavioral trials."

## 10-a. What are the most time-consuming steps of the code?

i. The script instruments every phase per session and prints the breakdown. Over the full 173-session run (546.8 s total, plus pickling of the 11.8 GB output):

| phase | total | share | per session |
|---|---|---|---|
| neural binning | 422.2 s | 77% | 2.44 s (max 6.7 s) |
| output construction (tongue + choice) | 45.6 s | 8% | 0.26 s |
| file loading (h5py) | 62.3 s | 11% | 0.36 s |
| input construction | 17.4 s | 3% | 0.10 s |
| trial-validity filter | ~0 s | 0% | 0.00 s |

Loading was the original bottleneck (24 s/session with `pynwb`, projected ~70 min) and was fixed by switching to `h5py`. After that fix the dominant cost — the per-trial × per-neuron spike-binning loop — was measured but never optimised further, because the total was already inside the 15-minute budget.

ii.
```python
    print(f"    load={t_load-t0:.1f}s, filter={t_filter-t_load:.1f}s, "
          f"neural={t_neural-t_filter:.1f}s, input={t_input-t_neural:.1f}s, "
          f"output={t_output-t_input:.1f}s, total={t_output-t0:.1f}s")
```

```python
        elapsed = time.time() - total_t0
        rate = elapsed / (i + 1)
        remaining = rate * (len(nwb_files) - i - 1)
        print(f"  Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
```

iii. CONVERSION_NOTES Step 6: "Code speedups: h5py direct loading instead of pynwb (0.2s vs 24s per session); searchsorted for fast spike time lookups; Vectorized bin counting", and Step 7's estimate "Full pipeline ~3s/session → ~9 min". Trajectory step 47: "2 sessions processed in 48.1s… For 174 sessions, this would take ~70 minutes, which is too slow. The bottleneck is loading."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops remain, all vectorizable, and they account for essentially all of the runtime:

- **`compute_firing_rates_all_trials`**: a double loop over trials × neurons, doing one `np.histogram` per (trial, neuron) pair — ~93,000 × ~400 ≈ 37 M histogram calls. The trial dimension can be collapsed by flattening all trials' edges into one array and using a single `searchsorted` per neuron (as the reference does), reducing this to one call per neuron. This loop is 77% of runtime.
- **`compute_tongue_y_all_trials`**: a double loop over trials × 80 bins, with two `searchsorted` calls over the whole ~600k-sample timestamp array per bin (160 per trial where 2 would suffice). A single global bin index plus `np.bincount` would replace the whole thing.
- **`compute_inputs_all_trials`**: a loop over trials, and inside it a loop over *every* photostim event in the session — O(n_trials × n_events) mask evaluations where a per-trial lookup would be O(1).
- **`get_lick_choices`**: a per-trial loop doing two `searchsorted` calls; both are directly vectorizable over the whole `go_times` array.

Despite this, CONVERSION_NOTES claims the binning is vectorized ("Vectorized spike binning using np.histogram", "Vectorized bin counting"), which the code does not do across trials or neurons.

ii.
```python
for t in range(n_trials):
    ...
    for i, spikes in enumerate(spike_times_list):
        ...
        counts, _ = np.histogram(spikes_in_window, bins=abs_bin_edges)
        fr[i] = counts / bin_width
```

```python
for t in range(n_trials):
    ...
    for b in range(n_bins):
        idx_start = np.searchsorted(tongue_timestamps, abs_bin_edges[b])
        idx_end = np.searchsorted(tongue_timestamps, abs_bin_edges[b + 1])
```

```python
for t in range(n_trials):
    ...
    for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
        mask = (abs_bin_centers >= ps_start) & (abs_bin_centers < ps_stop)
```

iii. CONVERSION_NOTES Step 6 lists the intended speedups; trajectory step 46 states the intent explicitly — "the compute_firing_rates function is called once per trial, iterating over all neurons and all spike times. For 400+ neurons and 500+ trials, this could be slow. Let me optimize by vectorizing the spike binning. Also, the tongue_y computation has a nested loop (trials x bins) that could be slow" — but the AI then decided to "run the sample conversion first", the `h5py` change alone brought the estimate under the 15-minute budget, and the loops were never revisited.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and read exactly once, and no session-level quantity is recomputed. Within a session, however, several cheap quantities are recomputed per trial or per bin:

- `bin_edges_rel` / `bin_centers_rel` are rebuilt on every call to the three per-session helpers instead of being derived once at module level (module-level `N_TIMEBINS` exists but the helpers re-derive `n_bins` themselves).
- `abs_bin_edges = bin_edges_rel + go_time` and the per-neuron bounds checks are redone for every trial.
- The tongue code performs 160 `searchsorted` calls per trial over the full camera timestamp array, where consecutive bins share an edge and the whole trial needs only two lookups.
- Every photostim event in the session is re-tested against every trial.
- The ragged `spike_times` buffer is split into per-unit arrays for **all** ~1,500–2,000 units per session before the good-unit mask is applied.
- `min_spike`/`max_spike` are computed by a separate pass over the good-unit list, duplicating the bounds information the binning loop later recomputes per trial.

ii.
```python
def compute_firing_rates_all_trials(...):
    n_bins = int((align_end - align_start) / bin_width)
    bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
def compute_tongue_y_all_trials(...):
    n_bins = int((align_end - align_start) / bin_width)
    bin_edges_rel = np.arange(n_bins + 1) * bin_width + align_start
def compute_inputs_all_trials(...):
    n_bins = int((align_end - align_start) / bin_width)
    bin_centers_rel = np.arange(n_bins) * bin_width + align_start + bin_width / 2
```

```python
n_units = len(data['classification'])
all_spike_times = []
prev_idx = 0
for i in range(n_units):
    end_idx = spike_times_index[i]
    all_spike_times.append(spike_times_data[prev_idx:end_idx])
    prev_idx = end_idx
```

iii. Not discussed in CONVERSION_NOTES beyond the general claim in Step 6 that the conversion was optimised for speed. None of these repeats change the result; they are pure overhead, and none was flagged because the measured 9-minute runtime met the instructions' budget.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts of work whose results never reach the output file:

- **Spike arrays for non-good units**: the full ragged buffer is split for every unit (272,227 units dataset-wide) but only the 69,453 `'good'` ones are used — ~75% of that work is discarded.
- **Dead code in the tongue function**: `dt` (median inter-frame interval) and `t0_tongue` are computed with a comment about precomputing a stride, then never used.
- **`subject_desc`** is read from the file and carried in the per-session result dict but never written to the output.
- **`anno_names`** is returned alongside `coarse_regions` but only the coarse labels are used to build `brain_region_idx`.
- **`p40` / `p60`** are returned from the tongue function and used only by the optional plotting path; the thresholds themselves are not saved to `metadata`, so the discretisation cannot be reproduced from the output.
- **`trial_stop`** is loaded only to bound the tone search.
- The plotting path (`plot_processing`) recomputes distributions and is only meaningful in `--show-processing` mode.

ii.
```python
    dt = np.median(np.diff(tongue_timestamps[:100]))
    t0_tongue = tongue_timestamps[0]
```

```python
    return {
        'neural': neural_trials,
        ...
        'anno_names': anno_names,
        'subject_id': data['subject_id'],
        'subject_desc': data['subject_desc'],
        ...
    }
```

```python
n_units = len(data['classification'])
...
for i in range(n_units):
    end_idx = spike_times_index[i]
    all_spike_times.append(spike_times_data[prev_idx:end_idx])
```

iii. Not discussed in CONVERSION_NOTES. All of these are minor except the splitting of non-good units' spike buffers, which contributes to the per-session load time; the AI's optimisation effort (trajectory steps 46–51) was directed at the I/O format rather than at trimming work inside the loader.
