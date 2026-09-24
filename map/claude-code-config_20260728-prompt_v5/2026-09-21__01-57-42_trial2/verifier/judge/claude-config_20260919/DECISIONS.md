# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is the DANDI:000363 MAP dataset, distributed as one NWB file per session under `/app/data/sub-<subject_id>/`. The AI finds every session with a single hard-coded glob over that layout, sorts the result for deterministic ordering, and processes one file at a time in a `for` loop wrapped in a `try/except` so that a failure in one session does not abort the run. Rather than using `pynwb`, the AI opens each NWB file directly as an HDF5 file with `h5py` and reads the internal paths itself (`general/subject/subject_id`, `intervals/trials/*`, `units/*`, `acquisition/BehavioralEvents/*`, `acquisition/BehavioralTimeSeries/*`). All 174 files are found; 173 are written to the output (one is dropped for having no QC-'good' units). In `--sample` mode it takes the first file from each of the first 5 subjects.

ii.
```python
    # Find all NWB files
    nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
    print(f"Found {len(nwb_files)} NWB files")
```

```python
    for i, nwb_file in enumerate(nwb_files):
        print(f"[{i+1}/{len(nwb_files)}] Processing {os.path.basename(nwb_file)}...")
        try:
            result = process_session(nwb_file, show_processing=args.show_processing)
            if result is not None:
                all_sessions.append(result)
        except Exception as e:
            print(f"  ERROR processing {nwb_file}: {e}")
```

```python
    f = h5py.File(nwb_path, 'r')

    # ── Subject info ──
    subject_id = f['general']['subject']['subject_id'][()].decode()

    # ── Trial info ──
    n_trials_total = f['intervals']['trials']['id'].shape[0]
    outcome = np.array([x.decode() for x in f['intervals']['trials']['outcome'][:]])
    ...
    go_start_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
```

iii. From CONVERSION_NOTES Step 2: "NWB files organized as `/app/data/sub-{id}/sub-{id}_ses-{datetime}_behavior+ecephys[+ogen].nwb` — 28 subjects, 174 sessions". Because there is exactly one file per session, the directory listing is the complete set of sessions and a glob suffices. The AI chose raw `h5py` over `pynwb` for speed (the trajectory shows repeated profiling of load time against the 15-minute budget in the instructions; file loading ended up at ~0.2 s/session). The resulting counts (28 subjects, 174 files) match `data/dandiset.yaml` and the notes' Step 2 table.

## 1-b. How are the data split into subjects?

i. Each NWB file names its animal in `general/subject/subject_id` (a numeric string such as `'440956'`). The AI reads that byte string, decodes it, and returns it per session. At assembly, `subjects` is the sorted set of unique ids across kept sessions and `subject_idx` is each session's index into that list. This yields 28 subjects.

ii.
```python
    subject_id = f['general']['subject']['subject_id'][()].decode()
```

```python
    # Subjects
    all_subject_ids = sorted(set(s['subject_id'] for s in all_sessions))
    subject_to_idx = {sid: i for i, sid in enumerate(all_subject_ids)}
    ...
        subject_idx_list.append(subject_to_idx[sess['subject_id']])
    ...
        'subjects': all_subject_ids,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 maps `general/subject/subject_id` → `subjects`/`subject_idx` as a straight "index mapping". The subject id is the canonical animal identifier stored in the file, and the containing folder name (`sub-440956`) is derived from it, so no separate grouping is needed. The final count of 28 subjects is checked against the paper/dandiset in Steps 2, 9 and 10 ("Subjects: 28 (match)").

## 1-c. How are the data split into sessions?

i. One NWB file is one session, so no splitting or grouping is done: the sorted file list *is* the session list, and session order in every output field follows it. Each session is labelled by its file basename (`sub-440956_ses-20190207T120657_behavior+ecephys+ogen.nwb`), recorded in `metadata['session_info']` together with the subject id, unit count, trial count and a computed behavioural performance. Sessions are dropped only if they have no QC-'good' units or fewer than 2 surviving trials; 173 of 174 files reach the output.

Note an internal inconsistency in the AI's documentation: CONVERSION_NOTES Step 5 "Key Decisions" #2 states that the paper's session-selection criteria (>65% correct on control trials, ≥50 correct left and right trials) would be applied, but the final script does not implement any such filter — Step 6 instead records "Session filtering: none needed (DANDI archive already curated)". The conversion log shows a retained session with `perf=53.7%`, confirming the criteria were not applied.

ii.
```python
    nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
```

```python
    # ── Session selection ──
    # The DANDI archive already contains sessions selected by the paper authors
    # (performance > 65%, >= 50 correct L and R trials). One session has no good units.
    is_auto_or_free = (auto_water == 1) | (free_water == 1)
```

```python
    if n_good == 0:
        print(f"  SKIP {basename}: no good units")
        f.close()
        return None
```

```python
            'session_info': [
                {
                    'session_name': s['session_name'],
                    'subject_id': s['subject_id'],
                    'n_good_units': s['n_good_units'],
                    'n_trials': s['n_trials'],
                    'performance': s['performance'],
                }
                for s in all_sessions
            ],
```

iii. The AI's stated rationale (code comment and Step 6 of CONVERSION_NOTES) is that the DANDI archive already contains only the sessions the paper authors selected, so re-applying the behavioural session criteria is unnecessary. The AI treats the 174 → 173 reduction as the resolution of the 174-vs-173 discrepancy it flagged in Step 4, attributing the single drop to the session whose units were never quality-controlled.

## 1-d. Are the data correctly split into trials?

i. Trials come from the NWB trials table `intervals/trials`, one row per behavioural trial. The AI takes `n_trials_total` from the length of `intervals/trials/id`, reads the per-trial columns (`outcome`, `early_lick`, `trial_instruction`, `auto_water`, `free_water`, `photostim_power`, `photostim_onset`, `photostim_duration`, `start_time`), and asserts that the number of `go_start_times` events equals the number of trial rows, so the go cue can be matched to trials by position. A boolean `trial_mask` and the index array `trial_indices` carry the mapping from kept trials back to original trial rows, which is used when indexing per-trial photostim intervals.

ii.
```python
    n_trials_total = f['intervals']['trials']['id'].shape[0]
    outcome = np.array([x.decode() for x in f['intervals']['trials']['outcome'][:]])
    early_lick = np.array([x.decode() for x in f['intervals']['trials']['early_lick'][:]])
    trial_instruction = np.array([x.decode() for x in f['intervals']['trials']['trial_instruction'][:]])
```

```python
    go_start_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
    assert len(go_start_times) == n_trials_total, f"Go cue count mismatch: {len(go_start_times)} vs {n_trials_total}"
    go_times = go_start_times[trial_indices]
```

iii. The trials table defines trials explicitly, so the AI uses it directly rather than re-deriving trial boundaries from event streams. The assertion encodes the AI's finding (trajectory steps 168-172) that `go_start_times` has exactly one event per trial in every session, unlike `sample_start_times` and `delay_start_times`, which have *more* events than trials (e.g. 518 sample and 511 delay events for 450 trials in one session) because a lick during the sample or delay epoch replays that epoch.

## 1-e. How are trials filtered based on quality controls?

i. A single behavioural filter is applied: trials flagged `auto_water == 1` or `free_water == 1` are removed, because they are not genuine behavioural trials. Early-lick trials, `ignore` (no-response) trials and photostimulation trials are deliberately **kept**, because they are decoder outputs/inputs. A session is dropped if fewer than 2 trials survive. 4,385 of 94,990 trials are removed, leaving 90,605.

No filter is applied for *availability of spike data*. The AI did not use `units/obs_intervals`, which records which trials were actually observed by the ephys recording. As a result 1,061 trials (1.2%) enter the output with literally zero spikes across all neurons and all 80 bins; the verification script reports each of them as a "all neural data is zero" warning. I confirmed the cause directly from the raw data: e.g. `sub-440956_ses-20190208T133600` has 480 behavioural trials but `obs_intervals` covers only 160 of them, and all 480 are written to the output.

ii.
```python
    # ── Trial filtering: remove auto_water and free_water ──
    trial_mask = ~is_auto_or_free
    trial_indices = np.where(trial_mask)[0]
    n_trials = len(trial_indices)

    if n_trials < 2:
        print(f"  SKIP {basename}: only {n_trials} valid trials")
        f.close()
        return None
```

```python
    is_auto_or_free = (auto_water == 1) | (free_water == 1)
```

iii. CONVERSION_NOTES Step 3/Step 5: the reference code's `get_regular_trial_mask()` removes early-lick, auto-water, free-water, no-response and photostim trials; the AI keeps the last three categories because "these are decoder outputs/inputs" and removes only auto/free water as "not genuine behavioral trials". This is a well-reasoned adaptation of the reference curation to the decoder task. For the all-zero trials, the AI's Step 10 review notes them as "Zero neural trials: 1,061/90,605 (1.2%) - edge-of-recording trials" and the trajectory (step 126) speculates they are trials "where no spikes occurred during the time window", but no fix was attempted and the warnings were left unresolved and unexplained in CONVERSION_NOTES.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `units/spike_times` together with its ragged-array offset index `units/spike_times_index`, restricted to the units whose `units/classification` is `'good'`. The go-cue times from `acquisition/BehavioralEvents/go_start_times/timestamps` are the other input, used to place the bin edges. `units/anno_name` supplies each retained unit's brain-region label.

ii.
```python
    # ── Good units ──
    classification = f['units']['classification'][:]
    good_mask = classification == b'good'
    good_indices = np.where(good_mask)[0]
```

```python
    # ── Spike times ──
    spike_times_data = f['units']['spike_times'][:]
    spike_times_index = f['units']['spike_times_index'][:]

    # Extract spike times for all good units once
    unit_spike_times = extract_unit_spike_times(spike_times_data, spike_times_index, good_indices)
```

```python
def extract_unit_spike_times(spike_times_data, spike_times_index, unit_indices):
    """Extract spike times for multiple units at once.
    Returns list of arrays, one per unit."""
    starts = np.zeros(len(unit_indices), dtype=np.int64)
    ends = spike_times_index[unit_indices].astype(np.int64)
    for i, idx in enumerate(unit_indices):
        if idx == 0:
            starts[i] = 0
        else:
            starts[i] = int(spike_times_index[idx - 1])
    return [spike_times_data[starts[i]:ends[i]] for i in range(len(unit_indices))]
```

iii. CONVERSION_NOTES Step 2 documents `units/spike_times` + `spike_times_index` as "compressed spike time arrays" and notes spike times are absolute seconds from session start. Spike times are the only neural representation in the file, so firing rates are computed from them directly. The whole ragged buffer is read once per session rather than one HDF5 read per unit.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each good unit and each trial, the trial's absolute time window is found by `searchsorted`, and `np.histogram` with edges `BIN_EDGES + go_t` counts spikes per bin. Counts are accumulated into a `(n_units, n_trials, n_bins)` `float32` array, divided by the 0.05 s bin width to convert to Hz, then split into a list of `(n_units, 80)` arrays, one per trial. No smoothing, no sliding window, no normalisation and no baseline subtraction.

ii.
```python
def compute_firing_rates_fast(unit_spike_times_list, go_times, bin_edges):
    ...
    all_rates = np.zeros((n_units, n_trials, n_bins), dtype=np.float32)

    for ni in range(n_units):
        st = unit_spike_times_list[ni]
        if len(st) == 0:
            continue

        for ti in range(n_trials):
            go_t = go_times[ti]
            t_lo = go_t + bin_edges[0]
            t_hi = go_t + bin_edges[-1]
            i_lo = np.searchsorted(st, t_lo)
            i_hi = np.searchsorted(st, t_hi)
            if i_lo >= i_hi:
                continue
            # np.histogram is C-optimized and faster than searchsorted+add.at
            counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
            all_rates[ni, ti] = counts

    all_rates /= bin_width

    # Split into list of per-trial arrays
    return [all_rates[:, ti, :] for ti in range(n_trials)]
```

iii. CONVERSION_NOTES Step 5 Key Decision #3: "Use 50ms non-overlapping bins (not sliding window) as specified by decoder task. Different from reference code's 40ms sliding window with 3.4ms stride." The reference `sliding_histogram(..., rate=True)` likewise returns `binSpikes / bin_width`, so the Hz convention matches; only the bin width/stride differ, and that difference is mandated by the decoder-task spec. Step 10 records a spot-check of the resulting rates against raw NWB spike times ("exact match") and that there are no negative rates and a plausible 940 Hz maximum.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == b'good'` are kept; no thresholds are applied to any individual quality metric. A session with zero good units is dropped entirely. This retains 69,453 of the ~272k units (the notes report 401.5 good units/session across 173 sessions).

ii.
```python
    # ── Good units ──
    classification = f['units']['classification'][:]
    good_mask = classification == b'good'
    good_indices = np.where(good_mask)[0]
    n_good = len(good_indices)

    if n_good == 0:
        print(f"  SKIP {basename}: no good units")
        f.close()
        return None
```

iii. CONVERSION_NOTES Steps 1/3: the reference pipeline runs `qc_mode='classifier'` with five region-specific QC classifiers (`helper_get_neuron_id_area()`), and "In NWB data, QC is encoded as `units/classification` with values 'good' or 'unlabelled'", i.e. the classifier verdict of the Chen/Liu et al. 2023 spike-sorting QC white paper is already materialised in the file. The AI checks the result against the paper: 69,453 good units vs the reported 69,943 (99.3%), and 173 sessions vs the reported 173, which it treats as a match.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB times are on one session-absolute clock, so no resampling or per-stream offset correction is needed. The fixed relative bin-edge grid `BIN_EDGES` (−2.5 s … +1.5 s) is added to each trial's go-cue time to give that trial's absolute edges, and spikes are binned against those edges directly. Go cues are taken from `BehavioralEvents/go_start_times/timestamps`, subset to the kept trials.

ii.
```python
T_START = -2.5         # seconds before go cue
T_END = 1.5            # seconds after go cue
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

```python
    go_times = go_start_times[trial_indices]
    ...
    neural_data = compute_firing_rates_fast(unit_spike_times, go_times, BIN_EDGES)
```

```python
            counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
```

iii. CONVERSION_NOTES Step 3: "Spike times aligned to go cue onset (go cue = time 0)", matching the reference code, where spike times are already go-cue aligned. Step 10 records the check "Verify temporal alignment by checking go cue corresponds to t=0 → confirmed", and the `--show-processing` plots draw a red dashed line at t = 0 on the neural heatmap and population average.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, giving 80 non-overlapping bins spanning −2.5 s to +1.5 s relative to the go cue. The grid is defined once at module level as 81 edges plus 80 centres and reused for every trial, every session and every data stream (neural, inputs, tongue output), so every trial has exactly 80 timepoints. There is no rebinning of an intermediate representation — spikes go straight from raw times to the 50 ms grid — and no sliding window. `metadata['time_bin_size']` is recorded as 50.0 ms with `off_start = -2.5`, `off_end = 1.5`.

ii.
```python
BIN_WIDTH = 0.05       # 50 ms bins
T_START = -2.5         # seconds before go cue
T_END = 1.5            # seconds after go cue
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

```python
            'time_bin_size': BIN_WIDTH * 1000,  # 50 ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'bin_centers': BIN_CENTERS.tolist(),
```

iii. The window and bin width are set by the instructions' Decoder Task section. CONVERSION_NOTES Step 3: "Our decoder task: 50ms bins, -2.5s to +1.5s around go cue = 80 time bins", explicitly contrasted with the reference preprocessing parameters (bw=40 ms/stride=3.4 ms, or bw=100 ms/stride=50 ms) which are superseded here by the task spec. The verification output confirms Mean/Min/Max T = 80 for all 173 sessions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. **No raw variable.** The final script derives it entirely from a hard-coded constant, `TONE_ONSET_REL_GO = -1.85` s, taken from the nominal task protocol (0.65 s sample epoch + 1.2 s delay). An earlier version did read `acquisition/BehavioralEvents/sample_start_times/timestamps` and take the last tone onset before each go cue — the same variable the human reference uses — but that code was deleted; the final script never opens `sample_start_times`. The resulting input vector is therefore byte-identical for every trial in every session (range [−0.625, 3.325] s), as the verification output confirms (input 0 range is `[-0.6, 3.3]` for all 173 sessions).

ii.
```python
# Task protocol defines fixed timing: sample epoch (0.65s) + delay (1.2s) = 1.85s before go cue.
# Early lick replays create extra sample_start events in the NWB data that confuse event-based
# matching (~12.5% of trials affected). Since the task timing is fixed, we use the protocol-defined
# offset for all trials.
TONE_ONSET_REL_GO = -1.85  # seconds before go cue
```

```python
    # ── Inputs ──
    # Input 0: Time from tone onset (fixed task protocol timing)
    time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO  # same for all trials
```

iii. The trajectory (steps 157-190) shows the reasoning. The AI found that 12.5% of trials have a last-`sample_start`-before-go offset different from −1.85 s, and initially attributed this to early-lick epoch replays. It then investigated a specific case (trial 10 of `sub-456772_ses-20191119T115109`) and found presample at −1.70 s, sample at −0.95 s, delay at −0.30 s, with **exactly one** sample event in the trial and `early_lick == 'no early'` — i.e. a genuinely shortened trial, not a replay. It also found per-session delay offsets taking three distinct values (−1.8, −1.2, −0.3 s). Despite this evidence, it concluded (step 185) "The correct approach is to use the fixed task timing: tone onset is ALWAYS at go - 1.85s. The event-based lookup is unreliable because early lick replays create extra events", and hard-coded the constant. CONVERSION_NOTES Step 10 records this as a "Bug Found and Fixed".

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. One vectorised subtraction: each bin centre (relative to the go cue) minus the constant tone offset, i.e. `BIN_CENTERS + 1.85`. The result is computed once per session, cast to `float32`, and stacked into row 0 of every trial's `(2, 80)` input array. Because the offset is a constant, every trial receives an identical copy of the vector `[-0.625, -0.575, …, 3.325]`; the value never encodes the actual measured tone-to-go interval of the trial.

ii.
```python
    time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO  # same for all trials
```

```python
        trial_input = np.stack([time_from_tone.astype(np.float32), photostim_ts], axis=0)  # (2, n_bins)
        input_data.append(trial_input)
```

iii. Same justification as 3-a: the AI treats the protocol-defined 1.85 s as ground truth and the per-trial event timestamps as unreliable. CONVERSION_NOTES Step 10 lists "time_from_tone: consistent [-0.625, 3.35] across all sessions" as a passing sanity check — i.e. the constancy of the value is presented as evidence of correctness rather than as a loss of information.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: it is built from `BIN_CENTERS`, the same go-cue-relative grid whose edges (`BIN_EDGES`) define the spike bins, so bin *k* of the input covers exactly the same interval as bin *k* of the firing rates. No interpolation or offset correction is involved.

ii.
```python
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

```python
    time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO
```

```python
            counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
```

iii. Not separately justified in CONVERSION_NOTES; the single shared bin grid makes alignment automatic. The `--show-processing` plots overlay the input traces on the same time axis as the neural data with a go-cue marker at 0.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration`, both stored as byte strings with the sentinel `b'N/A'` on unstimulated trials, together with `intervals/trials/start_time` (the onset is measured from trial start) and the go-cue times (to re-express the interval on the go-cue axis). `photostim_power` is read as well, but only to identify control trials for the performance statistic.

ii.
```python
def get_photostim_intervals(f, n_trials, go_times, trial_start_times):
    """Get photostim intervals relative to go cue for each trial.
    Returns list of (onset_rel, offset_rel) or None for each trial."""
    photostim_onset = f['intervals']['trials']['photostim_onset'][:]
    photostim_dur = f['intervals']['trials']['photostim_duration'][:]

    intervals = []
    for i in range(n_trials):
        if photostim_onset[i] == b'N/A':
            intervals.append(None)
        else:
            onset_rel_trial = float(photostim_onset[i])
            dur = float(photostim_dur[i])
            # Convert from trial-relative to go-cue-relative
            onset_abs = onset_rel_trial + trial_start_times[i]
            onset_rel_go = onset_abs - go_times[i]
            offset_rel_go = onset_rel_go + dur
            intervals.append((onset_rel_go, offset_rel_go))
    return intervals
```

iii. CONVERSION_NOTES Step 2 documents the storage convention: "`intervals/trials/photostim_onset`: relative to trial start, or 'N/A'"; "`photostim_duration`: '0.5000' or 'N/A'". Step 3 cross-checks against the paper ("Photostim duration 0.5s ... 'last 0.5 s' of delay") and Step 4 records a discrepancy the AI resolved in favour of the data: "Data shows photostim starts at delay onset, not the last 0.5s. Trust the data."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series, not a per-trial flag: for each trial a zero vector of 80 bins is created and set to 1.0 in every bin whose **centre** falls in `[onset_rel_go, offset_rel_go)`. Trials with `b'N/A'` onset get an all-zero vector. The result is row 1 of the `(2, 80)` input array, stored as `float32`. Across the dataset ~20.7% of trials in optogenetics sessions are stimulated.

ii.
```python
        # Photostim binary time series
        photostim_ts = np.zeros(N_BINS, dtype=np.float32)
        ps_interval = photostim_intervals[trial_idx]
        if ps_interval is not None:
            onset_rel, offset_rel = ps_interval
            photostim_ts[(BIN_CENTERS >= onset_rel) & (BIN_CENTERS < offset_rel)] = 1.0
```

iii. The instructions require "Whether photostimulation is on at every time point (discrete, time-varying)", and CONVERSION_NOTES Step 5 maps it as "Binary: 1 if photostim is on at time point, 0 otherwise". The half-open `[onset, offset)` comparison on bin centres is the natural discretisation of an interval onto the bin grid. Step 10 checks the resulting rate (20.7% of ogen-session trials) against the paper's "~25% randomly interleaved trials".

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stored onset is trial-relative, so it is converted to absolute time by adding `trial_start_time`, then to go-cue-relative time by subtracting that trial's go cue — the same axis the neural bins live on. The offset is onset + duration. Both are then compared against `BIN_CENTERS` directly. Note that `get_photostim_intervals` is computed over *all* original trial rows and indexed with the original `trial_idx`, so the trial filtering cannot desynchronise the lookup.

ii.
```python
            onset_abs = onset_rel_trial + trial_start_times[i]
            onset_rel_go = onset_abs - go_times[i]
            offset_rel_go = onset_rel_go + dur
```

```python
    photostim_intervals = get_photostim_intervals(f, n_trials_total, go_start_times, trial_start_times)
    ...
    for ti, trial_idx in enumerate(trial_indices):
        ...
        ps_interval = photostim_intervals[trial_idx]
```

iii. CONVERSION_NOTES Step 6: "Photostim: converted from trial-relative to go-cue-relative timing." Step 3 records the expected result of the conversion as a check: "Data shows onset at -1.2s (delay start), duration 0.5s, ending at -0.7s relative to go cue", which the `--show-processing` photostim panel visualises against the go-cue line.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the file, so choice is derived from two trials-table columns: `intervals/trials/trial_instruction` (`'left'`/`'right'`, the side the tone instructed) and `intervals/trials/outcome` (`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed side, a miss means it licked the other side, an ignore means it never licked.

ii.
```python
    outcome = np.array([x.decode() for x in f['intervals']['trials']['outcome'][:]])
    trial_instruction = np.array([x.decode() for x in f['intervals']['trials']['trial_instruction'][:]])
    ...
    trial_outcomes = outcome[trial_indices]
    trial_instructions = trial_instruction[trial_indices]
```

iii. CONVERSION_NOTES Step 5 Key Decision #7: "Choice derivation: Determine from outcome + trial_instruction: hit=same as instruction, miss=opposite, ignore=no_lick." Step 1 notes the equivalent encoding in the reference code (trial type 1=left/0=right, correctness 1/0/−1) and maps it onto the NWB fields. Step 10 records "Output variables: all match raw NWB for first 10 trials".

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A per-trial Python branch maps (outcome, instruction) to the codes 0 = left, 1 = right, 2 = no lick, and the scalar is then broadcast across all 80 bins with `np.full` into row 0 of the `(4, 80)` per-trial output array (`int64`). `output_values[0]` names the three codes `['left', 'right', 'no_lick']`.

ii.
```python
        # Choice: 0=left, 1=right, 2=no_lick
        out = trial_outcomes[ti]
        inst = trial_instructions[ti]
        if out == 'hit':
            choice = 0 if inst == 'left' else 1
        elif out == 'miss':
            choice = 1 if inst == 'left' else 0  # opposite of instruction
        else:  # ignore
            choice = 2
```

```python
        trial_output = np.stack([
            np.full(N_BINS, choice, dtype=np.int64),
            np.full(N_BINS, outcome_val, dtype=np.int64),
            np.full(N_BINS, early_val, dtype=np.int64),
            tongue_y_disc,
        ], axis=0)  # (4, n_bins)
```

iii. `left = 0`, `right = 1` follow the instructions' ordering, with a third class added for the no-lick case since choice is undefined on `ignore` trials. The instructions ask for time-varying outputs "if at all possible", and the target format requires one `(n_output, n_timepoints)` array, so the per-trial scalar is repeated across bins to share the array with the time-varying tongue output.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `intervals/trials/outcome`, which already stores exactly the three strings `'ignore'`, `'miss'`, `'hit'` the instructions ask for. No derivation is needed.

ii.
```python
    outcome = np.array([x.decode() for x in f['intervals']['trials']['outcome'][:]])
    ...
    trial_outcomes = outcome[trial_indices]
```

iii. CONVERSION_NOTES Step 2 lists "`intervals/trials/outcome`: 'hit', 'miss', 'ignore'", and Step 1 notes the correspondence with the reference code's `correctness` variable (1 = correct, 0 = error, −1 = no response). Step 5 maps it straight to `output[1]`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps the three strings to 0 = ignore, 1 = miss, 2 = hit; the scalar is broadcast across all 80 bins into row 1 of the output array. `output_values[1] = ['ignore', 'miss', 'hit']`.

ii.
```python
        # Outcome: 0=ignore, 1=miss, 2=hit
        outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[out]
```

```python
            np.full(N_BINS, outcome_val, dtype=np.int64),
```

iii. The code assignment follows the ordering given in the instructions ("Outcome (ignore, miss, hit, per-trial)"). As with choice, the per-trial value is repeated across bins to fit the single per-trial output array. The AI sanity-checked the resulting distribution against the paper's 84% correct rate, obtaining 81.8% on control trials (CONVERSION_NOTES Step 10).

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from `intervals/trials/early_lick`, a string column holding `'early'` / `'no early'`.

ii.
```python
    early_lick = np.array([x.decode() for x in f['intervals']['trials']['early_lick'][:]])
    ...
    trial_early_lick = early_lick[trial_indices]
```

iii. CONVERSION_NOTES Step 2 lists "`intervals/trials/early_lick`: 'early', 'no early'", so the flag is explicit in the data and no derivation from lick times is needed. Step 3 notes that the reference code's `get_regular_trial_mask()` *removes* these trials, and Step 5 records the deliberate deviation: early-lick trials are kept because early lick is a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A two-way test maps `'no early'` → 0 and anything else → 1, broadcast across all 80 bins into row 2 of the output array. `output_values[2] = ['no', 'yes']`.

ii.
```python
        # Early lick: 0=no, 1=yes
        early_val = 0 if trial_early_lick[ti] == 'no early' else 1
```

```python
            np.full(N_BINS, early_val, dtype=np.int64),
```

iii. The 0 = no / 1 = yes coding follows the instructions. The value is per-trial in the source, so it is repeated across bins like the other per-trial outputs. The early lick itself occurs during the sample or delay epoch, so the event that sets the flag falls inside the −2.5 s pre-go window and is in principle decodable.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: its `data` array is `(n_frames, 3)` = `tongue_x`, `tongue_y`, `tongue_likelihood`, with a matching `timestamps` array at ~294 Hz. Column 1 (`tongue_y`) supplies the value; column 2 (the DeepLabCut likelihood) decides whether the tongue counts as visible. Column 0 is read but unused.

ii.
```python
    tongue_ts = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
    tongue_data_raw = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
```

```python
    tongue_y = tongue_data[:, 1]
    tongue_lh = tongue_data[:, 2]
```

iii. CONVERSION_NOTES Step 2/Step 3: "acquisition/BehavioralTimeSeries/: tongue/jaw/nose tracking at ~300Hz (x, y, likelihood)"; trajectory step 37: "Tongue tracking: 3 columns (tongue_x, tongue_y, tongue_likelihood) at ~294 Hz (300 Hz nominal)". This is the only tongue measurement in the file, and the frame rate is cross-checked against the paper's "acquired at 300 Hz".

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Two passes over the camera stream, both restricted to the per-trial windows:

1. **Pass 1 (percentile pool)**: for each trial, `searchsorted` finds the frame range covering `[go − 2.5 s, go + 1.5 s)`; frames with `likelihood >= 0.9` are collected. All such frames from all trials of the session are concatenated and the 40th and 60th percentiles of *raw frame* `tongue_y` are taken as the two class edges.
2. **Pass 2 (per-trial discretisation)**: each trial's frames are assigned to bins via `searchsorted` on the relative bin edges. For each bin the code computes the **mean likelihood over all frames in the bin** and requires it to be `>= 0.9` for the bin to count as visible; if so, the **mean y over all frames in the bin** (not only the visible ones) is compared to the two edges. Bins failing the gate keep the default value 3.

The frame ranges from pass 1 are cached and reused in pass 2, so the camera stream is scanned once.

ii.
```python
    # First pass: collect visible y values for percentile computation
    visible_y_all = []
    trial_tongue_ranges = []
    for ti in range(n_trials):
        go_t = go_times[ti]
        t_lo = go_t + bin_edges[0]
        t_hi = go_t + bin_edges[-1]
        i_lo = np.searchsorted(tongue_ts, t_lo)
        i_hi = np.searchsorted(tongue_ts, t_hi)
        trial_tongue_ranges.append((i_lo, i_hi))

        if i_lo < i_hi:
            vis_mask = tongue_lh[i_lo:i_hi] >= likelihood_thresh
            if vis_mask.any():
                visible_y_all.append(tongue_y[i_lo:i_hi][vis_mask])
```

```python
        # Compute mean y and mean likelihood per bin using bincount
        for b in range(n_bins):
            b_mask = bin_idx_v == b
            if not b_mask.any():
                continue
            avg_lh = trial_lh_v[b_mask].mean()
            if avg_lh >= likelihood_thresh:
                avg_y = trial_y_v[b_mask].mean()
```

iii. CONVERSION_NOTES Step 5 Key Decision #5: "Compute session-level percentiles on ALL visible tongue y values (likelihood > 0.9), then discretize per time point. Not visible = tongue_likelihood < 0.9." The rationale for excluding low-likelihood frames is that the tracker still emits a position when the tongue is retracted. No justification is given in the notes for the specific 0.9 threshold, for gating on the *mean* likelihood of a bin, or for averaging y over all frames in the bin rather than only the visible ones. Consequences visible in `verification_full_out.txt`: 92% of all bins land in the 'not visible' class (vs 75% for the human reference), and requiring a bin's *mean* likelihood ≥ 0.9 at ~15 frames/bin effectively demands that nearly every frame in the bin be tracked, so partial protrusions are discarded.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes: `0` = below the session's 40th percentile, `1` = between the 40th and 60th, `2` = above the 60th, `3` = not visible. The edges `p40`/`p60` are per session. Comparison is `avg_y < p40` → 0, `avg_y <= p60` → 1, else 2; the default for a bin is 3.

Crucially, the percentiles are taken over **individual camera frames**, while the quantity actually digitised is the **50 ms bin mean**. Combined with the strict per-bin visibility gate, the resulting class balance departs substantially from the intended 40/20/40 split: the verification output gives dataset-wide fractions of roughly 0.013 / 0.028 / 0.041 / 0.918 for classes 0/1/2/3, i.e. among classified (visible) bins only ~12% fall below the nominal 40th percentile and ~47% above the 60th.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9  # threshold for tongue visibility
```

```python
    all_visible = np.concatenate(visible_y_all)
    p40 = np.percentile(all_visible, 40)
    p60 = np.percentile(all_visible, 60)
```

```python
        trial_bins = np.full(n_bins, 3, dtype=np.int64)
        ...
            if avg_lh >= likelihood_thresh:
                avg_y = trial_y_v[b_mask].mean()
                if avg_y < p40:
                    trial_bins[b] = 0
                elif avg_y <= p60:
                    trial_bins[b] = 1
                else:
                    trial_bins[b] = 2
```

iii. The 40th/60th cut points, the per-session scope and the fourth 'not visible' class all come straight from the instructions' Decoder Output specification. CONVERSION_NOTES Step 5 states the percentiles are computed on "ALL visible tongue y values", i.e. the AI's reading of "percentile of y-position over the session" is a percentile over raw position samples. The mismatch between the distribution the edges are drawn from (frames) and the distribution they are applied to (bin means) is not discussed anywhere in the notes or the trajectory, and the resulting skew was not flagged in Step 10's statistics review.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as spikes and events, so alignment is direct: `searchsorted(tongue_ts, go_t + BIN_EDGES[0])` and `... + BIN_EDGES[-1]` give the trial's frame range, and each frame's bin index is `searchsorted(bin_edges, frame_time − go_t, 'right') − 1`, clipped to `[0, 80)`. This is the same go-cue-relative grid used for the firing rates, so bin *k* of the tongue output covers the same interval as bin *k* of the neural data. Bins with no frames at all (the video is trial-gated, so leading bins can be empty when the go cue falls less than 2.5 s after trial start) retain the default class 3.

ii.
```python
        i_lo = np.searchsorted(tongue_ts, t_lo)
        i_hi = np.searchsorted(tongue_ts, t_hi)
```

```python
        # Assign each tongue sample to a bin
        rel_ts = trial_ts - go_t
        bin_idx = np.searchsorted(bin_edges, rel_ts, side='right') - 1
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
```

iii. Not separately argued in CONVERSION_NOTES beyond Step 5's "Time-varying at 50ms bins"; the shared bin grid makes the alignment automatic, and no interpolation or offset correction is applied. The `--show-processing` plot panel "Output: Tongue y discretized" plots the discretised trace on the same axis with the go-cue line, as a visual alignment check.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases are handled:

- **Session never quality-controlled**: `classification` is not `b'good'` for any unit, so `good_indices` is empty and the session is skipped (`SKIP ... no good units`). This drops exactly 1 of 174 sessions.
- **Session with too few trials**: fewer than 2 surviving trials → session skipped, satisfying the format requirement of ≥2 trials per session.
- **Unparseable/absent brain-region annotation**: `classify_brain_region` returns `'unknown'` for an empty annotation or one matching no rule; 1,369 neurons end up in `'unknown'`.
- **Unstimulated trials**: the `b'N/A'` sentinel in `photostim_onset` is detected explicitly and yields an all-zero photostim row rather than a parse error.
- **Untracked tongue**: frames below the likelihood threshold, and bins with no frames at all, become the explicit 'not visible' class 3 rather than being imputed.
- **Any other per-session failure**: the whole `process_session` call is wrapped in `try/except`, which prints the traceback and continues with the remaining sessions.

**Not handled**: trials with no spike data at all. `units/obs_intervals` is never read, so 1,061 trials whose ephys recording had not yet started are emitted as 4 s of exactly 0 Hz across every neuron (see 1-e).

ii.
```python
    if n_good == 0:
        print(f"  SKIP {basename}: no good units")
        f.close()
        return None
```

```python
    if n_trials < 2:
        print(f"  SKIP {basename}: only {n_trials} valid trials")
        f.close()
        return None
```

```python
def classify_brain_region(anno_name):
    """Classify a detailed CCF annotation into a major brain region."""
    if not anno_name or anno_name == '':
        return 'unknown'
    ...
    return 'unknown'
```

```python
    if len(visible_y_all) == 0:
        p40, p60 = None, None
        return [np.full(n_bins, 3, dtype=np.int64) for _ in range(n_trials)], p40, p60
```

```python
        except Exception as e:
            print(f"  ERROR processing {nwb_file}: {e}")
            import traceback
            traceback.print_exc()
```

iii. CONVERSION_NOTES Step 4 attributes the 174-vs-173 session discrepancy to one session being excludable, and the code comment records that "One session has no good units". For the tongue, the missing measurement is treated as a legitimate state of the world (tongue retracted) and so gets its own category rather than being imputed — this is required by the instructions' four-class specification. For the all-zero trials the AI's Step 10 review records them as "edge-of-recording trials" and accepts them; no explanation is given in CONVERSION_NOTES for leaving the 1,061 verification warnings unresolved, even though Step 10 of the instructions requires each warning to be fixed or explained.

## 10-a. What are the most time-consuming steps of the code?

i. As shipped, the full conversion takes 498.7 s for 174 files (~2.9 s/session), plus pickling of the 12.0 GB result. Within a session the dominant costs are (a) reading the whole ragged `units/spike_times` buffer and the `(n_frames, 3)` tongue array from HDF5, and (b) the nested `n_units × n_trials` Python loop in `compute_firing_rates_fast`, which performs two `searchsorted` calls plus one `np.histogram` (and one fresh `bin_edges + go_t` allocation) for each of, e.g., 400 × 520 ≈ 208,000 unit-trial pairs. The AI profiled this explicitly during development (trajectory steps 84-94): at first the per-trial × per-bin boolean-mask loop in the tongue discretisation dominated at 54 s for 520 trials; after it was restructured with cached `searchsorted` ranges the per-session time fell from ~48 s to ~3.2 s, a ~15× speedup, bringing the projected full run to ~9 minutes and under the instructions' 15-minute budget.

ii.
```python
    for ni in range(n_units):
        st = unit_spike_times_list[ni]
        ...
        for ti in range(n_trials):
            ...
            counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
```

```python
    spike_times_data = f['units']['spike_times'][:]
    spike_times_index = f['units']['spike_times_index'][:]
```

```python
    elapsed = time.time() - t0
    print(f"  {basename}: {n_good} good units, {n_trials} trials, perf={performance:.1%}, time={elapsed:.1f}s")
```

iii. The script prints per-session timing and a running average so bottlenecks are visible in the log, as the instructions require. CONVERSION_NOTES Step 7 records "~2.4s/session" for the sample and Step 9 "~500s (2.9s/session)" for the full run; the "Run Time Estimates" tables in the notes template were left unfilled, so the profiling evidence lives only in the trajectory.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Five Python loops remain that could be collapsed:

1. **The trial loop inside `compute_firing_rates_fast`** — the largest. The per-trial dimension can be removed entirely by flattening all trials' edges into one array and calling `np.searchsorted(st, edges)` once per unit, then differencing; the human reference does exactly this. The unit loop itself cannot be vectorised because the spike arrays are ragged.
2. **The per-bin loop `for b in range(n_bins)` in `compute_tongue_y_all_trials`** — 80 iterations per trial, each building a boolean mask over the trial's frames. Despite the inline comment "Compute mean y and mean likelihood per bin using bincount", `np.bincount` is not actually used; two `bincount` calls would replace the whole loop.
3. **The per-trial loop in `compute_tongue_y_all_trials`'s first pass**, which appends to a Python list; the frame ranges could be obtained with two vectorised `searchsorted` calls on all go cues at once.
4. **`get_photostim_intervals`**, a per-trial Python loop with `float()` conversions; the whole column can be converted with a masked `astype(float)`.
5. **The per-trial input and output assembly loops**, which call `np.stack`/`np.full` once per trial instead of building `(n_trials, d, 80)` arrays and slicing.

ii.
```python
        for ti in range(n_trials):
            go_t = go_times[ti]
            t_lo = go_t + bin_edges[0]
            t_hi = go_t + bin_edges[-1]
            i_lo = np.searchsorted(st, t_lo)
            i_hi = np.searchsorted(st, t_hi)
            if i_lo >= i_hi:
                continue
            counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
            all_rates[ni, ti] = counts
```

```python
        # Compute mean y and mean likelihood per bin using bincount
        for b in range(n_bins):
            b_mask = bin_idx_v == b
            if not b_mask.any():
                continue
```

iii. The AI vectorised only as far as it needed to hit the instructions' 15-minute budget, and stopped once the projection reached ~9 minutes (trajectory step 94). The function docstring "Optimized: iterate over units (outer) and vectorize across trials" overstates what was done — the trial dimension is still a Python loop. CONVERSION_NOTES has "Code inefficiencies identified" / "Code speedups added" placeholders in the Step 6 template, but the final notes do not fill them in, so no remaining inefficiency is documented.

## 10-c. What processing does the code repeat multiple times?

i. Three genuine repetitions:

1. **Bin-edge array reconstruction**: `bin_edges + go_t` allocates a new 81-element array for *every* (unit, trial) pair — ~208,000 allocations per session of only ~520 distinct arrays.
2. **Redundant spike search**: each iteration calls `np.searchsorted` twice to trim the spike array, then `np.histogram` re-searches the same trimmed range internally.
3. **Constant input replication**: `time_from_tone` is a single vector identical for all 90,605 trials, yet `np.stack` materialises and pickles a separate copy in each trial's input array.

Against that, several things are correctly computed once: the bin grid is module-level; `trial_tongue_ranges` from the tongue percentile pass is cached and reused in the discretisation pass; `unit_spike_times` is extracted once per session; `get_photostim_intervals` is called once per session. The camera stream is necessarily traversed twice (once to establish the per-session percentiles, once to discretise), which is inherent to per-session percentiles and not wasteful.

ii.
```python
            counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
```

```python
        trial_input = np.stack([time_from_tone.astype(np.float32), photostim_ts], axis=0)  # (2, n_bins)
```

```python
        trial_tongue_ranges.append((i_lo, i_hi))
    ...
        i_lo, i_hi = trial_tongue_ranges[ti]
```

iii. None of these repetitions is discussed in CONVERSION_NOTES; the caching of `trial_tongue_ranges` came out of the step-89 profiling work, where the AI identified the tongue loop as the bottleneck and restructured it into the current two-pass form.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items:

- **Dead variable**: `trial_starts = trial_start_times[trial_indices]` is computed and never used anywhere in the function.
- **Discarded return values**: `compute_tongue_y_all_trials` returns `p40, p60`, which `process_session` unpacks and then throws away — the per-session class edges are never recorded in the output, so the discretisation cannot be audited from the pickle.
- **Photostim intervals for filtered-out trials**: `get_photostim_intervals` loops over all `n_trials_total` rows even though auto/free-water rows are never indexed.
- **Unused tracker column**: the whole `(n_frames, 3)` tongue array is read, but `tongue_x` (column 0) is never used.
- **Per-session `performance`**: computed from `photostim_power`/`early_lick`/`auto_water`/`free_water` and stored in `metadata['session_info']`, but never used to filter anything (see 1-c) and not consumed downstream.
- **Constant input channel**: because `time_from_tone` is identical for every trial (see 3-a), input row 0 carries no across-trial information for the decoder, yet is stored 90,605 times.

ii.
```python
    go_times = go_start_times[trial_indices]
    trial_starts = trial_start_times[trial_indices]   # never used again
```

```python
    tongue_discretized, p40, p60 = compute_tongue_y_all_trials(
        tongue_ts, tongue_data_raw, go_times, BIN_EDGES)
```

```python
    performance = n_correct / n_control_responded if n_control_responded > 0 else 0
```

iii. The `performance` computation is the residue of the session-selection filter the AI planned in Step 5 and then decided against in Step 6 ("Session filtering: none needed (DANDI archive already curated)") — the statistic was retained as metadata and used in Step 10 as a sanity check against the paper's 84% correct rate, so it is not purely wasted. The remaining items (`trial_starts`, discarded `p40`/`p60`, the unused `tongue_x` column) are unexamined leftovers; none is discussed in CONVERSION_NOTES.
