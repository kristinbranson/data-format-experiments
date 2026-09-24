# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI 000363) is one NWB file per session under `/app/data/sub-<id>/`. `get_all_nwb_files()` lists every `sub-*` directory (sorted), then every `*.nwb` file inside each (sorted), and returns `(subject_dir_name, path)` pairs — 174 files, 28 subjects. `main()` iterates over that list and calls `process_session()` once per file. Inside `process_session()` the file is opened with `pynwb.NWBHDF5IO(..., 'r')` and everything is read from the one handle: `nwb.units` (spike times, `classification`, `anno_name`, `electrode_group`), `nwb.trials` (per-trial columns), `nwb.acquisition['BehavioralEvents']` (`go_start_times`, `photostim_start_times`, `photostim_stop_times`), and `nwb.acquisition['BehavioralTimeSeries']` (`Camera0_side_TongueTracking`). The handle is closed at the end of the session. `--sample` restricts to the first file of the first two subjects.

ii.
```python
def get_all_nwb_files(data_dir, sample=False):
    """Get list of all NWB files."""
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for f in nwb_files:
            all_files.append((subj, os.path.join(subj_dir, f)))
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
session_id = nwb.identifier
units = nwb.units
...
trials = nwb.trials
...
be = nwb.acquisition['BehavioralEvents']
bts = nwb.acquisition['BehavioralTimeSeries']
```

iii. CONVERSION_NOTES Step 2 records the layout: "`/app/data/sub-XXXXXX/` - one directory per subject (28 subjects); Each contains 1-10 NWB files (one per session)". Because the directory listing is the complete set of sessions, a single directory walk is sufficient, and sorting makes session order deterministic. The reference code operates on DataJoint `.mat` exports, which do not exist here, so the agent used `pynwb` against the published NWB release ("Reference code works with .mat files from DataJoint; our data is in NWB format").

## 1-b. How are the data split into subjects?

i. The subject label is the containing directory name (e.g. `'sub-440956'`), passed into `process_session()` as `subject_id` and carried through to assembly. At assembly, `subject_list` is built in order of first appearance (which, because the file list is sorted by subject directory, is alphabetical), and `subject_idx` holds each session's index into it. The NWB field `nwb.subject.subject_id` (`'440956'`) is not used; only the directory-derived string. Result: 28 subjects, 3–10 sessions each.

ii.
```python
result = process_session(fpath, subj, ...)
```

```python
seen_subjects = {}
subject_list = []
for sd in session_data_list:
    subj = sd['subject_id']
    if subj not in seen_subjects:
        seen_subjects[subj] = len(seen_subjects)
        subject_list.append(subj)
...
all_subject_idx.append(seen_subjects[sd['subject_id']])
...
'subjects': subject_list,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. The DANDI layout puts one directory per animal, so the directory name is a complete and unambiguous grouping key; the agent noted "28 subjects" as a sanity check against the papers ("Subjects | 28 | datapaper Fig 1J") and the verification output confirms 28 subjects with the expected session counts.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping is needed. The session is identified by `nwb.identifier` (e.g. `SC015_20190207_120657_s1`) and stored in `metadata['session_ids']`. Session order in the output follows the sorted (subject, filename) order. A session is dropped if it has zero `classification == 'good'` units or fewer than 2 surviving trials; `process_session()` returns `None` and `main()` skips it. 173 of 174 sessions reach the output — `sub-440958_ses-20190216T162508` is dropped for having no QC-labelled units.

ii.
```python
session_id = nwb.identifier
...
if n_good == 0:
    print(f"  Skipping {session_id}: 0 good units")
    io.close()
    return None
...
if n_valid < 2:
    print(f"  Skipping {session_id}: only {n_valid} valid trials")
    io.close()
    return None
```

```python
for file_idx, (subj, fpath) in enumerate(nwb_files):
    result = process_session(fpath, subj, ...)
    if result is None:
        continue
    session_data_list.append(result)
```

iii. CONVERSION_NOTES Step 4: "Total sessions | N/A | 174 NWB files | 173 behavioral sessions | 1 session (sub-440958_ses-20190216) has 0 good units; exclude it -> 173 sessions" — i.e. the file-per-session boundary was cross-checked against the 173 sessions quoted in `methods.txt`, and the single dropped file reconciles the count. The 2-trial minimum comes from the target-format requirement ("There needs to be at least two trials within each session").

## 1-d. How are the data split into trials?

i. Trials are taken directly from the NWB trials table (`nwb.trials`), one row per behavioural trial. Per-trial columns (`trial_instruction`, `outcome`, `early_lick`, `auto_water`, `free_water`) are read as full-length arrays and indexed by row. Go-cue times come from `BehavioralEvents/go_start_times.timestamps`, which is assumed to be parallel to the trials table (one event per row) and is indexed with the same trial indices. No assertion checks that the two lengths agree.

ii.
```python
trials = nwb.trials
n_trials_total = len(trials)
trial_instructions = trials['trial_instruction'][:]
outcomes = trials['outcome'][:]
early_licks = trials['early_lick'][:]
auto_water = np.array(trials['auto_water'][:])
free_water = np.array(trials['free_water'][:])
...
go_start_times = be.time_series['go_start_times'].timestamps[:]
go_cues_valid = go_start_times[valid_trial_indices]
```

iii. Step 2 of CONVERSION_NOTES documents the trials-table columns, and the agent's exploration (trajectory step 48) worked out the epoch structure from a single trial's `presample/sample/delay/go` events: "go_start_times has fewer entries (368) than delay_start_times (395), meaning each trial has exactly one go cue but potentially multiple delay starts". Having established one go cue per trial, the trials table is used as the trial definition directly rather than re-deriving boundaries from event streams.

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept if `auto_water == 0` **and** `free_water == 0`. Nothing else is excluded: early-lick, `ignore` (no-response) and photostim trials are deliberately retained because they are decoder inputs/outputs. Sessions left with fewer than 2 trials are dropped. The `units/obs_intervals` record of which trials the ephys actually covered is **not** consulted; as a result 1,061 trials across 9 sessions whose go-cue window lies outside the recording remain in the output with all-zero firing rates and valid behavioural labels. This was found during verification, root-caused, and knowingly kept. Across the 173 retained sessions 3,765 of 94,370 trials (4.0%) are removed, leaving 90,605.

ii.
```python
# Filter: exclude auto_water and free_water
trial_mask = (auto_water == 0) & (free_water == 0)
valid_trial_indices = np.where(trial_mask)[0]
n_valid = len(valid_trial_indices)

if n_valid < 2:
    print(f"  Skipping {session_id}: only {n_valid} valid trials")
    io.close()
    return None
```

iii. CONVERSION_NOTES Step 3/5: the reference code's `get_regular_trial_mask` "removes early lick, auto water, free water, no-response, and stimulation trials", but "Since we decode early_lick, outcome (incl. ignore), and use photostim as input, we keep these. We exclude only auto_water and free_water trials." For the zero-neural trials, Step 9 states: "Neuropixels recording didn't span the entire behavioral session... Verified by checking NWB file for session 2: max spike time across all good units is 1107.4s, but go cue for trial 159 is at 1112.6s. These trials have valid behavioral labels but no neural coverage. This is acceptable - the decoder should learn that zero neural activity provides no information." Step 12 lists "Excluding zero-neural trials might improve accuracy slightly" as a potential improvement that was not implemented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `units/spike_times` — the sorted spike times of each unit in session-absolute seconds — restricted to units with `classification == 'good'`. The second ingredient is `BehavioralEvents/go_start_times.timestamps`, which positions the bin edges. `units/anno_name` and `units/electrode_group.location` are used only to label each retained neuron's brain region.

ii.
```python
good_indices = np.where(good_mask)[0]

# Get spike times for good units (sorted for searchsorted)
spike_times_list = []
for i in good_indices:
    st = np.sort(np.array(units['spike_times'][i], dtype=np.float64))
    spike_times_list.append(st)
```

```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
go_cues_valid = go_start_times[valid_trial_indices]
...
trial_neural = compute_firing_rates_fast(spike_times_list, go_cues_valid, n_good)
```

iii. CONVERSION_NOTES Step 1/4: "Spike times in the reference .mat data are already aligned to go cue (time 0)... In NWB files, spike times are in absolute session time; go cue times available in BehavioralEvents". `spike_times` is the only neural representation in the file, so firing rates are computed from it directly.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz with no smoothing, normalisation or baseline subtraction. For each trial, absolute bin edges are formed as `BIN_EDGES + go_cue`; for each neuron, `np.searchsorted` isolates the spikes in `[go-2.5, go+1.5)` and `np.histogram` counts them into the 80 bins; counts are divided by the 50 ms bin width and stored as `float32`. Output is one `(n_neurons, 80)` array per trial.

ii.
```python
def compute_firing_rates_fast(spike_times_list, go_cue_times, n_neurons):
    for t_idx in range(n_trials):
        gc = go_cue_times[t_idx]
        t_start = gc - TIME_BEFORE
        t_end = gc + TIME_AFTER
        edges = BIN_EDGES + gc  # absolute time edges

        fr_matrix = np.zeros((n_neurons, N_BINS), dtype=np.float32)

        for n_idx in range(n_neurons):
            st = spike_times_list[n_idx]
            # Binary search for spikes in window
            i_start = np.searchsorted(st, t_start, side='left')
            i_end = np.searchsorted(st, t_end, side='left')
            spikes_in_window = st[i_start:i_end]

            if len(spikes_in_window) > 0:
                counts = np.histogram(spikes_in_window, bins=edges)[0]
                fr_matrix[n_idx, :] = counts / BIN_WIDTH

        trial_data.append(fr_matrix)
```

iii. CONVERSION_NOTES Step 10, Check 3: "Firing rate: histogram counts / bin_width (not Gaussian kernel as in reference, but appropriate for 50ms bins)". The reference `sliding_histogram(..., rate=True)` likewise returns `binSpikes / bin_width`; the agent kept the rate definition and changed only the bin geometry, which the Decoder Task section mandates (50 ms non-overlapping rather than the reference's 40 ms width / 3.4 ms stride).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept; no thresholds are applied to any individual QC metric and `units/unit_quality` is not used. Sessions with zero good units are dropped entirely. This retains 69,453 units over 173 sessions (mean 401.5, min 90, max 923 per session). No further filtering by brain region, firing rate, or trial coverage is applied.

ii.
```python
units = nwb.units
classification = np.array(units['classification'][:])
good_mask = classification == 'good'
n_good = int(np.sum(good_mask))

if n_good == 0:
    print(f"  Skipping {session_id}: 0 good units")
    io.close()
    return None

good_indices = np.where(good_mask)[0]
```

iii. CONVERSION_NOTES Step 1/5: "QC filtering: The NWB `classification` column ('good'/'unlabelled') corresponds to classifier QC", matching `qc_mode = 'classifier'` in the reference `preprocess_all_ephys.py` and the QC classifier described in `ChenLiuEtAl2023_SpikeSortingQC.pdf`. The resulting count was checked against the paper: "Total neurons | 69,453 | 69,943 | CLOSE (-490, 0.7%)" and "Mean neurons/session | 401.5 | median 393".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset. Spike times and `go_start_times` are already on the same session-absolute clock, so the fixed go-cue-relative edge grid is simply added to each trial's go-cue time to give the absolute window, and spikes are binned against those absolute edges. No resampling, interpolation or per-stream offset correction is applied.

ii.
```python
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
...
gc = go_cue_times[t_idx]
t_start = gc - TIME_BEFORE
t_end = gc + TIME_AFTER
edges = BIN_EDGES + gc  # absolute time edges
```

```python
'temporal_alignment_event': 'Go cue onset',
'off_start': -TIME_BEFORE,
'off_end': TIME_AFTER,
```

iii. CONVERSION_NOTES Step 4 lists the discrepancy "Spike time reference | Aligned to go cue | Absolute session time in NWB | Aligned to go cue | Subtract go_start_time from spike times", i.e. the agent explicitly reconciled the reference pipeline (which stores go-cue-aligned spikes) with the NWB convention (absolute times). Step 10 Check 2 reports "Spike alignment: correctly uses go cue as t=0, bin edges [-2.5, 1.5]".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial spanning −2.5 s to +1.5 s relative to the go cue. The edge/centre grid is built once at module level and reused for every trial, every session and every data stream, so all trials have exactly 80 timepoints. There is no rebinning, no overlapping/sliding window, and no downsampling of a pre-binned stream — spikes are binned once, directly at the target resolution. `metadata['time_bin_size']` is recorded as 50.0 ms.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins
TIME_BEFORE = 2.5  # seconds before go cue
TIME_AFTER = 1.5   # seconds after go cue
N_BINS = int((TIME_BEFORE + TIME_AFTER) / BIN_WIDTH)  # 80
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

```python
'time_bin_size': BIN_WIDTH * 1000,  # 50 ms
```

iii. CONVERSION_NOTES Step 4/5: "Firing rate bins | bw=40ms, stride=3.4ms | ... | Task requires 50ms bins; use non-overlapping 50ms bins" and Key Decision 4: "Bin size: 50ms non-overlapping bins as specified by task (not 40ms/3.4ms as in reference)". This is one of the sanctioned departures from the reference, since the Decoder Task section fixes the window and the bin width.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. No raw data variable is used. The tone onset is assumed to sit at a **fixed** −1.85 s relative to the go cue (0.65 s sample epoch + 1.2 s delay epoch), hard-coded as `TONE_OFFSET`. The input is therefore a function of the bin centres alone and is byte-identical for every trial in every session. The `BehavioralEvents/sample_start_times` stream (the actual tone onsets, which the agent had catalogued in Step 2) is never read, nor is `delay_start_times`.

ii.
```python
TONE_OFFSET = -1.85  # tone onset relative to go cue (sample=0.65s + delay=1.2s)
```

```python
# === Compute inputs ===
# Time from tone onset (same for all trials)
time_from_tone = (BIN_CENTERS - TONE_OFFSET).astype(np.float32)
```

iii. CONVERSION_NOTES Step 3: "Sample epoch: 0.65s ... Delay epoch: 1.2s ... Tone onset = go_cue - 1.85s", and Step 5's mapping table: "time from tone onset | input[0] | Continuous ramp: bin_center + 1.85 | Tone onset = go_cue - 1.85s". The trajectory (step 48) shows the agent was aware of the complication and chose not to address it: "For tone onset timing, I realize I may need to back-calculate it from the go cue time minus the delay and sample durations, since sample epochs can replay on early-lick trials... tone onset should be about go_cue minus 1.85s, but I should verify this against actual trial data since the delay could vary slightly... So tone onset should really be computed as delay_start minus 0.65s". The verification of a single trial (`sample 1.2115–1.8615, go 3.0615` ⇒ 1.85 s) was taken as sufficient.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. One vectorised subtraction: `time_from_tone = BIN_CENTERS - TONE_OFFSET`, i.e. a linear ramp from −0.625 s to +3.325 s in 50 ms steps, cast to `float32`. That same 80-element vector is then stacked with the photostim row and stored separately for each of the 90,605 trials. No per-trial value is ever computed.

ii.
```python
time_from_tone = (BIN_CENTERS - TONE_OFFSET).astype(np.float32)

trial_inputs = []
for t_idx in range(n_valid):
    input_data = np.stack([time_from_tone, photostim_inputs[t_idx]], axis=0)  # (2, N_BINS)
    trial_inputs.append(input_data)
```

iii. The Decoder Task specifies this input as "continuous, time-varying", so it is represented as a real-valued ramp rather than a binary event marker. CONVERSION_NOTES Step 10 Check 2 reports the resulting range as a sanity check: "Input: time_from_tone [-0.625, 3.325], photostim [0, 1] - correct".

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the ramp is defined on `BIN_CENTERS`, the centres of exactly the same 80 go-cue-relative bins used to bin the spikes, so bin *k* of the input covers the same interval as bin *k* of the firing rates. No separate alignment step exists.

ii.
```python
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
...
edges = BIN_EDGES + gc          # neural
...
time_from_tone = (BIN_CENTERS - TONE_OFFSET).astype(np.float32)   # input
```

iii. Both streams share one module-level grid, so the agent treated alignment as automatic; the `--show-processing` plots draw the go cue (red) and tone onset (green, at `TONE_OFFSET`) on both the neural heatmap and the input traces to make the shared axis visible ("Plots for `--show-processing` mode should visually convince the user that ... There are no temporal misalignments").

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `BehavioralEvents/photostim_start_times.timestamps` and `photostim_stop_times.timestamps`, which are absolute session times. If the session has no `photostim_start_times` series the input is all zeros. The trials-table columns `photostim_onset` / `photostim_duration` are not used.

ii.
```python
photostim_starts = None
photostim_stops = None
if 'photostim_start_times' in be.time_series:
    photostim_starts = be.time_series['photostim_start_times'].timestamps[:]
    photostim_stops = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. CONVERSION_NOTES Step 2 lists `photostim_start_times`/`photostim_stop_times` among the BehavioralEvents, and Step 5 maps "photostim on/off | input[1] | Binary: 1 if photostim active at timepoint | From photostim_start/stop_times". Trajectory step 48: "For photostimulation, I can check whether a timepoint falls between the start and stop times of the stim" — the event stream was preferred because it gives absolute on/off times directly, with no string parsing or trial-start arithmetic.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series per trial. For each trial, every photostim event in the session is examined; events that do not overlap `[go−2.5, go+1.5]` are skipped, and for the rest the on/off times are converted to go-cue-relative seconds and every bin whose **centre** falls in `[on, off)` is set to 1.0. Non-stimulated trials stay all-zero. Stored as `float32` in row 1 of the `(2, 80)` input array.

ii.
```python
for t_idx in range(n_trials):
    gc = go_cue_times[t_idx]
    photostim_binary = np.zeros(N_BINS, dtype=np.float32)

    trial_start = gc - TIME_BEFORE
    trial_end = gc + TIME_AFTER

    for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
        if ps_stop < trial_start or ps_start > trial_end:
            continue
        ps_start_rel = ps_start - gc
        ps_stop_rel = ps_stop - gc
        active = (BIN_CENTERS >= ps_start_rel) & (BIN_CENTERS < ps_stop_rel)
        photostim_binary[active] = 1.0
```

iii. The Decoder Task asks for "Whether photostimulation is on at every time point (discrete, time-varying)", so the agent produced a per-bin 0/1 indicator rather than a per-trial flag. The bin-centre convention matches the one used for the tone ramp. Sanity checks in Step 10 Check 5: "167/173 sessions have photostim events... Photostim fraction per session: ~22% (close to expected ~25%)".

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The event times are re-expressed relative to each trial's go cue (`ps_start - gc`), then compared against `BIN_CENTERS` — the same go-cue-relative grid the spikes are binned on — so no further alignment is needed. The per-trial overlap test uses the same `[go−2.5, go+1.5]` window as the neural extraction.

ii.
```python
ps_start_rel = ps_start - gc
ps_stop_rel = ps_stop - gc
active = (BIN_CENTERS >= ps_start_rel) & (BIN_CENTERS < ps_stop_rel)
```

iii. Same rationale as 3-c: all NWB streams share one session clock, so subtracting the go-cue time places photostim on the identical axis as the neural bins. The `--show-processing` figure plots the photostim row against `BIN_CENTERS` with the go cue marked, as a visual alignment check.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no choice column in the file. Choice is derived from two trials-table columns: `trial_instruction` (`'left'`/`'right'`) and `outcome` (`'hit'`/`'miss'`/`'ignore'`).

ii.
```python
trial_instructions = trials['trial_instruction'][:]
outcomes = trials['outcome'][:]
...
instruction = trial_instructions[trial_idx]
outcome = outcomes[trial_idx]
choice_val = get_choice(instruction, outcome)
```

iii. CONVERSION_NOTES Step 5, Key Decision 8: "Choice derivation: hit + instruction -> correct side; miss + instruction -> wrong side; ignore -> no_lick". In a two-alternative forced-choice task the licked side is fully determined by the instructed side and whether the trial was scored correct, so the agent reconstructed it rather than parsing the lick-time streams.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. `get_choice()` returns 0 for a left lick, 1 for a right lick and 2 for no lick (`outcome == 'ignore'`), with 2 also as the catch-all default. The value is one integer per trial, written into row 0 of the `(4, 80)` output array and repeated across all 80 bins. `output_values[0] = ['left', 'right', 'no_lick']`.

ii.
```python
def get_choice(trial_instruction, outcome):
    """Determine lick direction choice."""
    if outcome == 'ignore':
        return 2  # no_lick
    elif outcome == 'hit':
        return 0 if trial_instruction == 'left' else 1
    elif outcome == 'miss':
        return 1 if trial_instruction == 'left' else 0
    return 2
```

```python
output_combined = np.zeros((4, N_BINS), dtype=np.int64)
output_combined[0, :] = choice_val
```

iii. `left = 0`, `right = 1` follows the ordering given in the Decoder Task, with a third class for the no-lick case. Choice is a per-trial variable, so it is broadcast across bins to keep all four outputs in one `(n_output, n_timepoints)` array, as the target format allows. Step 9 reports the resulting distribution (43.0% / 42.3% / 14.7%) as a consistency check.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which already contains exactly `'hit'`, `'miss'` and `'ignore'`.

ii.
```python
outcomes = trials['outcome'][:]
...
outcome = outcomes[trial_idx]
```

iii. Step 2 of CONVERSION_NOTES records the column and its three levels ("outcome (hit/miss/ignore)"), which map one-to-one onto the three categories the Decoder Task asks for, so no derivation is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped through `outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}` with a `.get(outcome, 0)` fallback, and written into row 1 of the output array, repeated across all 80 bins. `output_values[1] = ['ignore', 'miss', 'hit']`.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
...
outcome_val = outcome_map.get(outcome, 0)
...
output_combined[1, :] = outcome_val
```

iii. The 0/1/2 assignment follows the order listed in the Decoder Task ("Outcome (ignore, miss, hit)"). Like choice, it is a per-trial value repeated across bins. Step 9 records the distribution (ignore 14.7%, miss 16.6%, hit 68.7%) and explicitly reconciles it with the paper's 84% correct rate: "Lower (we include early lick + ignore trials)".

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, whose values are `'early'` / `'no early'`.

ii.
```python
early_licks = trials['early_lick'][:]
...
early_val = early_map.get(early_licks[trial_idx], 0)
```

iii. Step 2 documents the column ("early_lick (early/no early)"); the flag is stored explicitly by the acquisition software, so no derivation from lick-time streams is needed.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped through `early_map = {'no early': 0, 'early': 1}` with a `.get(..., 0)` fallback, written into row 2 of the output array and repeated across all 80 bins. `output_values[2] = ['no', 'yes']`.

ii.
```python
early_map = {'no early': 0, 'early': 1}
...
early_val = early_map.get(early_licks[trial_idx], 0)
...
output_combined[2, :] = early_val
```

iii. The 0 = no / 1 = yes coding follows the Decoder Task ordering. Per-trial value repeated across bins, like the other categorical outputs. Step 9 reports 88.5% / 11.5%; the agent kept early-lick trials rather than following the reference's `get_regular_trial_mask`, precisely because early lick is a required decoder output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: `data[:, 1]` is the DeepLabCut tongue *y* coordinate, `data[:, 2]` is the tracking likelihood, and `timestamps` gives the frame times on the session clock. If the series is absent the whole session's tongue output is set to the "not visible" class.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
has_tongue = 'Camera0_side_TongueTracking' in bts.time_series
...
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_timestamps_all = tongue_ts_obj.timestamps[:]
tongue_y_all = tongue_data_all[:, 1]
tongue_lik_all = tongue_data_all[:, 2]
```

iii. Step 2 of CONVERSION_NOTES records the channel layout — "Camera0_side_TongueTracking (x, y, likelihood)" — which matches the series' own `description` attribute. This is the only tongue measurement in the file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps. (1) Frames are called visible when `likelihood >= 0.9`. (2) Two per-session thresholds are taken as the 40th and 60th percentiles of the **raw visible frame** *y* values across the whole session (guarded by a ≥10-visible-frame check, else `p40 = p60 = 0`). (3) Within each trial, frames in the window are assigned to the 80 go-cue-relative bins with `np.digitize`, and each bin's value is the mean *y* over the visible frames it contains. (4) That bin mean is compared against `p40`/`p60` to give class 0/1/2; bins with no visible frame keep the default class 3.

ii.
```python
DLC_LIKELIHOOD_THRESH = 0.9  # DeepLabCut confidence threshold for tongue visibility
...
visible_mask = tongue_lik_all >= DLC_LIKELIHOOD_THRESH
if np.sum(visible_mask) >= 10:
    visible_y = tongue_y_all[visible_mask]
    p40 = float(np.percentile(visible_y, 40))
    p60 = float(np.percentile(visible_y, 60))
```

```python
bin_assignments = np.digitize(t_rel, BIN_EDGES) - 1  # 0-indexed bins

for b in range(N_BINS):
    in_bin = bin_assignments == b
    if not np.any(in_bin):
        continue
    y_bin = y_slice[in_bin]
    l_bin = l_slice[in_bin]
    visible = l_bin >= DLC_LIKELIHOOD_THRESH
    if np.any(visible):
        mean_y = np.mean(y_bin[visible])
```

iii. CONVERSION_NOTES Step 5, Key Decision 6: "Tongue visibility: Use DLC likelihood threshold of 0.9 to determine tongue visibility", and the mapping table: "tongue y-position | output[3]: tongue_y | Discretized per session | 0:<40th, 1:40-60th, 2:>60th, 3:not visible". Percentiles are taken over visible frames only because the tracker still emits a coordinate when the tongue is retracted. The per-session scope and the 40/60 split follow the Decoder Task verbatim. Step 10 Check 5 notes "tongue_y has all 4 values represented".

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per bin: `mean_y < p40` → 0 ("low"); `p40 <= mean_y <= p60` → 1 ("mid"); `mean_y > p60` → 2 ("high"); no visible frame in the bin (or no frames at all, or no tongue series in the session) → 3 ("not_visible"), which is the array's initialised default. Thresholds are per session. Resulting distribution over all bins: 11.7% / 6.3% / 6.5% / 75.5%.

ii.
```python
tongue_y_binned = np.full(N_BINS, 3, dtype=np.int64)  # default: not visible
...
if np.any(visible):
    mean_y = np.mean(y_bin[visible])
    if mean_y < p40:
        tongue_y_binned[b] = 0
    elif mean_y <= p60:
        tongue_y_binned[b] = 1
    else:
        tongue_y_binned[b] = 2
```

```python
'output_values': [
    ...
    ['low', 'mid', 'high', 'not_visible'],
],
```

iii. The three visible classes and the fourth "not visible" class are exactly the four categories listed in the Decoder Task. Making 3 the array default means any bin that is not positively shown to contain a tracked tongue falls into it, which the agent treated as the safe default rather than imputing a position.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the session-absolute clock with spikes and go cues. Per trial, `np.searchsorted` on `tongue_ts` finds the frame range spanning `[go−2.5, go+1.5]`, frame times are converted to go-cue-relative seconds, and `np.digitize` against `BIN_EDGES` assigns each frame to the same 80-bin grid used for the firing rates. If the trial contains no frames at all the whole row stays class 3.

ii.
```python
gc = go_cue_times[t_idx]
t_start = gc + BIN_EDGES[0]
t_end = gc + BIN_EDGES[-1]
idx_start = np.searchsorted(tongue_ts, t_start, side='left')
idx_end = np.searchsorted(tongue_ts, t_end, side='right')

if idx_end <= idx_start:
    results.append(tongue_y_binned.reshape(1, -1))
    continue
...
t_rel = t_slice - gc
bin_assignments = np.digitize(t_rel, BIN_EDGES) - 1  # 0-indexed bins
```

iii. Reusing `BIN_EDGES` guarantees bin *k* of the tongue output covers the same interval as bin *k* of the neural data; the agent verified this visually in `processing_0/1.png`, whose bottom row overlays the raw per-frame tongue *y* (coloured by visibility, with `p40`/`p60` lines) against the discretised output on a shared go-cue-relative axis. Step 10 Check 2: "Output: choice/outcome/early_lick constant across time bins, tongue_y time-varying - correct".

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, handled by exclusion or by an explicit category:

- **Session never quality-controlled** (`classification` is NaN for all units): the `== 'good'` comparison yields no matches, `n_good == 0`, and the session is dropped (this is the single dropped file).
- **Session with fewer than 2 usable trials**: dropped.
- **Missing tongue series**: all trials of the session get class 3; fewer than 10 visible frames in a session leaves `p40 = p60 = 0`.
- **Frames with an untracked tongue** (`likelihood < 0.9`): excluded from the bin mean; a bin with no surviving frame becomes class 3.
- **Sessions with no photostim**: the input row is all zeros.
- **Unparseable electrode-group location JSON**: caught by a bare `except`, target region set to `''`, which falls back to the annotation-only region rules; an empty annotation maps to `'Unknown'`.
- **Unexpected category strings**: `outcome_map.get(outcome, 0)` and `early_map.get(..., 0)` silently default to `ignore` / `no early`, and `get_choice` defaults to `no_lick`.
- **Trials with no neural coverage** are *not* handled — they are kept with all-zero firing rates (see 1-e).

ii.
```python
if n_good == 0:
    print(f"  Skipping {session_id}: 0 good units")
    io.close()
    return None
```

```python
else:
    trial_tongue = [np.full((1, N_BINS), 3, dtype=np.int64) for _ in range(n_valid)]
```

```python
try:
    loc = json.loads(eg.location)
    target = loc.get('brain_regions', '')
except:
    target = ''
```

```python
outcome_val = outcome_map.get(outcome, 0)
early_val = early_map.get(early_licks[trial_idx], 0)
```

iii. CONVERSION_NOTES Step 4 documents the unlabelled session ("1 session (sub-440958_ses-20190216) has 0 good units; exclude it -> 173 sessions"). For tongue frames the agent's plan (Step 5) was "a separate 'not visible' category flagged by low DeepLabCut likelihood", i.e. a genuinely absent measurement is represented as its own class rather than imputed. The zero-neural trials were investigated and explicitly retained: "This is acceptable - the decoder should learn that zero neural activity provides no information."

## 10-a. What are the most time-consuming steps of the code?

i. Per-step timers are printed for every session. Summed over the full run of 173 sessions: firing-rate binning 409.4 s (70%), unit loading — the per-unit `spike_times`/`anno_name`/`electrode_group` reads plus `np.sort` — 112.0 s (19%), tongue processing 56.7 s (10%), trials/event loading and input construction ~0 s. Processing totalled 582 s, and pickling the 11.5 GB result took a further ~21 s (603 s total, ~3.4 s/session). This is within the instructions' 15-minute budget but ~2.4× the human reference's 247 s.

ii.
```python
t_fr = time.time()
print(f"  Firing rates ({t_fr - t_load:.1f}s)")
...
t_tongue = time.time()
print(f"  Tongue ({t_tongue - t_inp:.1f}s)")
...
elapsed = time.time() - t_start
avg_per_session = elapsed / (file_idx + 1)
remaining = avg_per_session * (len(nwb_files) - file_idx - 1)
print(f"  ETA: {remaining/60:.1f} min remaining")
```

iii. CONVERSION_NOTES Step 6 claims "Processing time: ~4s/session" and Step 7 estimates "~4s/session * 174 sessions = ~12 minutes", which the actual 10 min run beat. The notes do not break the per-session time down or name the dominant stage, so no further optimisation was attempted once the estimate came in under the 15-minute threshold.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops account for essentially all of the runtime, and all three are described in the code as "vectorized" while being plain nested Python loops:

- `compute_firing_rates_fast`: a `for t_idx` × `for n_idx` double loop calling `np.searchsorted` twice and `np.histogram` once per (trial, neuron) — ~200,000 NumPy calls per session. The reference collapses the trial dimension by flattening all trials' edges into one array and issuing a single `searchsorted` per unit, which is ~5× faster.
- `compute_tongue_y_fast`: a per-trial loop containing a `for b in range(N_BINS)` loop that recomputes the boolean mask `bin_assignments == b` 80 times over the whole frame slice. A single `np.bincount` on the bin index would replace both.
- `compute_photostim_input_fast`: for every trial it re-scans *every* photostim event in the session (O(n_trials × n_events)). A `searchsorted` of the event times against the trial windows — or simply reading `photostim_onset`/`photostim_duration` from the trials table, which is already per-trial — is O(n_trials).

Also unvectorised: the per-unit loops reading `units['spike_times'][i]`, `units['anno_name'][i]` and `units['electrode_group'][i]` element by element instead of reading the ragged buffer once.

ii.
```python
for t_idx in range(n_trials):
    ...
    for n_idx in range(n_neurons):
        st = spike_times_list[n_idx]
        i_start = np.searchsorted(st, t_start, side='left')
        i_end = np.searchsorted(st, t_end, side='left')
        spikes_in_window = st[i_start:i_end]
        if len(spikes_in_window) > 0:
            counts = np.histogram(spikes_in_window, bins=edges)[0]
```

```python
for b in range(N_BINS):
    in_bin = bin_assignments == b
```

```python
for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
    if ps_stop < trial_start or ps_start > trial_end:
        continue
```

iii. CONVERSION_NOTES Step 6 lists the optimisations actually made — "Used `np.searchsorted` for efficient spike windowing; Vectorized tongue y binning with `np.digitize`" — and no further work was done because the projected runtime met the instructions' 15-minute limit. The notes' "Code inefficiencies identified" section of the template was not filled in.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened once, but several quantities are recomputed:

- The window search over each neuron's full spike train is redone for every trial (`n_trials` × `n_neurons` binary searches, plus a second internal search inside `np.histogram`), instead of one pass per unit over all trial edges.
- `np.sort` is applied to every unit's spike times although NWB stores them already sorted.
- `time_from_tone` is a single constant vector, yet it is re-`np.stack`ed and stored independently for each of the 90,605 trials.
- The full photostim event list is re-scanned once per trial.
- Inside each trial's tongue processing, `bin_assignments == b` and the `likelihood >= 0.9` test are recomputed 80 times over the same slice.
- `json.loads(eg.location)` is called once per unit even though electrode groups are shared by hundreds of units in a session.

ii.
```python
st = np.sort(np.array(units['spike_times'][i], dtype=np.float64))
```

```python
for t_idx in range(n_valid):
    input_data = np.stack([time_from_tone, photostim_inputs[t_idx]], axis=0)  # (2, N_BINS)
    trial_inputs.append(input_data)
```

```python
eg = units['electrode_group'][i]
try:
    loc = json.loads(eg.location)
```

iii. Not discussed in CONVERSION_NOTES. The redundancy does not change any output value — it is a pure cost — and the agent stopped optimising once the run fit the time budget ("Processing time: ~4s/session").

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A handful of small items, none of which changes the converted values:

- `np.sort` on spike times that are already sorted — the result is identical to the input, so the work is wholly discarded.
- Firing rates are computed for the 1,061 trials with no ephys coverage; the resulting all-zero matrices carry no information (and, being retained, add noise to the decoder).
- The constant `time_from_tone` ramp is duplicated across all 90,605 stored trials rather than being recorded once in metadata.
- Outputs are stored as `int64` when the values are 0–3; `int8` would be sufficient and 8× smaller (~230 MB → ~29 MB).
- `metadata['bin_centers']` and `metadata['tone_offset_from_go_cue']` are written but never consumed by the decoder.
- `main()` computes a full per-region neuron tally and per-output distribution for printing only.

ii.
```python
st = np.sort(np.array(units['spike_times'][i], dtype=np.float64))
```

```python
output_combined = np.zeros((4, N_BINS), dtype=np.int64)
```

```python
'bin_centers': BIN_CENTERS.tolist(),
'tone_offset_from_go_cue': TONE_OFFSET,
```

iii. Not discussed in CONVERSION_NOTES. The printed summaries are deliberate — they serve the Step 9/10 consistency checks against the paper statistics — and the extra metadata fields are documentation. The `int64` outputs and duplicated input ramp are unexamined defaults rather than considered choices; the notes record only that `float32` was used for the neural arrays.
