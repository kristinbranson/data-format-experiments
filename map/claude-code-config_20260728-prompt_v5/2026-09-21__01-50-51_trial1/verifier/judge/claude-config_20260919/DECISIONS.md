# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI 000363, MAP) is one NWB file per session under `/app/data/sub-<subject_id>/`. The AI finds every session with a single hard-coded glob over that layout, sorts the paths for deterministic order, and opens each file once with `pynwb.NWBHDF5IO`. Inside a session it reads the units table (`classification`, `anno_name`, `electrode_group`, `spike_times`, `obs_intervals`), the trials table (column by column rather than as a dataframe), `acquisition['BehavioralEvents']` (go cue, sample onset, photostim start/stop) and `acquisition['BehavioralTimeSeries']['Camera0_side_TongueTracking']`. 174 files are found; 173 are processed (one returns `None`), giving 28 subjects, 69,453 units and 89,532 trials. There is no parallelism: sessions are processed serially in a single `for` loop.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
print(f'Found {len(nwb_files)} NWB files')

if args.sample:
    nwb_files = nwb_files[:2]
...
for i, nwb_path in enumerate(nwb_files):
    print(f'[{i+1}/{len(nwb_files)}] {os.path.basename(nwb_path)}')
    result = process_session(nwb_path, bin_edges, n_bins,
                             show_processing=args.show_processing, session_idx=i)
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()

subject_id = nwb.subject.subject_id
...
classification = nwb.units['classification'][:]
...
auto_water = nwb.trials['auto_water'][:]
...
be = nwb.acquisition['BehavioralEvents']
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
...
bts = nwb.acquisition['BehavioralTimeSeries']
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
```

iii. From CONVERSION_NOTES Step 2: "NWB format (Neurodata Without Borders) — Each file: one session with behavior + electrophysiology (+ optogenetics for most)". The AI notes the original reference code works on `.mat` files exported from DataJoint while the provided data is NWB from DANDI, so the loading layer is necessarily re-implemented while the processing is matched. The file count (174) and subject count (28) were checked against the data directory and against `dandiset.yaml`/the data paper in Step 2/Step 3 tables.

## 1-b. How are the data split into subjects?

i. Each session's animal is taken from `nwb.subject.subject_id` (a numeric string such as `'440956'`, the same id that names the containing `sub-*` folder). The unique ids are sorted to form `subjects`, and `subject_idx` holds each session's index into that list, in session order. Result: 28 subjects with 3-10 sessions each.

ii.
```python
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description
...
subject_ids = sorted(set(r['subject_id'] for r in all_results))
subjects = [str(sid) for sid in subject_ids]
subject_id_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
...
subject_idx.append(subject_id_to_idx[r['subject_id']])
...
subject_idx = np.array(subject_idx, dtype=np.int64)
```

iii. The notes' mapping table (Step 5) simply lists "subject_id from NWB → subjects/subject_idx : Map unique subject IDs". The AI verified against the data paper that this yields 28 mice ("Subjects | 28 | Yes" in the Step 10 statistics table), matching "n = 28" in the data paper. No comment is made about the numeric id differing from the mouse names (SC015 etc.) used in the papers.

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no grouping or splitting is performed. Session order in all output lists is the sorted-glob order (subject, then acquisition timestamp, because the filename embeds both). Sessions that return `None` from `process_session` (no good units / fewer than 2 valid trials) are dropped from every list together, so `neural`, `input`, `output`, `subject_idx` and `brain_region_idx` stay aligned. Unlike the reference, the AI does **not** store per-session identifiers in `metadata` — only aggregate counts (`n_sessions`, `n_subjects`, `n_neurons_total`, `n_trials_total`).

ii.
```python
for i, nwb_path in enumerate(nwb_files):
    result = process_session(...)
    if result is not None:
        all_results.append(result)
    else:
        print(f'  SKIPPED')
...
for r in all_results:
    neural.append(r['neural'])
    inputs.append(r['input'])
    outputs.append(r['output'])
    subject_idx.append(subject_id_to_idx[r['subject_id']])
```

```python
'metadata': {
    ...
    'n_sessions': len(all_results),
    'n_subjects': len(subjects),
    'n_neurons_total': total_neurons,
    'n_trials_total': total_trials,
}
```

iii. Step 2 of CONVERSION_NOTES establishes one file = one session. Step 4 records the discrepancy "174 NWB files" vs "173 sessions" in the data paper, and the AI resolves it by keeping "all sessions that have >= 2 valid trials", which happens to drop exactly one session and reproduce the paper's 173. The AI's Step 9 note attributes the drop to "no valid trials within obs_intervals"; the actual cause in the code is that this session (`sub-440958_ses-20190216T162508`) has no `classification == 'good'` units at all (its QC fields are NaN), so `n_good == 0` triggers the early return.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table, and `BehavioralEvents/go_start_times` is assumed to carry exactly one go cue per trial row, indexed by the same integer trial index (`go_cue_times_all[trial_idx]`). Every per-trial quantity (instruction, outcome, early lick, auto/free water, photostim window, tone, tongue, spikes) is taken at that index. No assertion is made that `len(go_start_times) == len(trials)` (the reference asserts this); the assumption does hold in all 174 files.

ii.
```python
auto_water = nwb.trials['auto_water'][:]
free_water = nwb.trials['free_water'][:]
trial_instruction = nwb.trials['trial_instruction'][:]
early_lick_arr = nwb.trials['early_lick'][:]
outcome_arr = nwb.trials['outcome'][:]
...
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
...
valid_trial_indices = np.where(valid_trial_mask)[0]
go_cue_times = go_cue_times_all[valid_trial_indices]
```

iii. Step 2 of the notes records the trials table as the per-trial source ("Trials table: start_time, stop_time, trial_instruction (left/right), early_lick, outcome, auto_water, free_water, photostim fields") and `go_start_times` as the alignment event. The AI checked in Step 3/4 that sample-to-go-cue is a consistent 1.85 s, which is an indirect check that trial rows and go-cue events are in register.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, combined into one boolean mask over trial rows:
   1. `auto_water == 0` (automatic-reward trials removed),
   2. `free_water == 0` (free-water trials removed),
   3. the **entire** analysis window must fall inside the electrophysiology observation span: `go - 2.5 >= obs_intervals[0][0]` and `go + 1.5 <= obs_intervals[-1][1]`, where `obs_intervals` is taken from the first good unit and only its first start and last stop are used (i.e. the span, not the individual per-trial intervals).
   
   A session with fewer than 2 surviving trials is dropped. Early-lick, ignore/no-response, and photostim trials are deliberately kept. This yields 89,532 trials (vs. 90,860 for the human reference); I verified over all 174 files that the AI's kept set is a strict subset of the reference's, the difference being auto-water trials (~1,300) plus a handful of window-edge trials. I also verified that the observed trials are contiguous in every session, so the "span" shortcut never wrongly admits an unobserved trial in this dataset. Two trials with all-zero neural data survive and are reported as warnings by the verifier.

ii.
```python
valid_trial_mask = (auto_water == 0) & (free_water == 0)
...
# Filter trials to those within the recording observation interval
# Use obs_intervals of the first good unit to determine the recording window
obs_intervals = nwb.units['obs_intervals'][good_indices[0]]
obs_start = obs_intervals[0, 0]
obs_end = obs_intervals[-1, 1]

# Only include trials whose go cue + analysis window falls within obs range
begin_time = bin_edges[0]
end_time = bin_edges[-1]
within_obs = ((go_cue_times_all + begin_time) >= obs_start) & \
             ((go_cue_times_all + end_time) <= obs_end)
valid_trial_mask = valid_trial_mask & within_obs

valid_trial_indices = np.where(valid_trial_mask)[0]
n_valid_trials = len(valid_trial_indices)

if n_valid_trials < 2:
    io.close()
    return None
```

iii. Step 3/Step 5 of CONVERSION_NOTES: the reference code's `get_regular_trial_mask` excludes "early_lick != 0, auto_water != 0, free_water != 0, correctness == -1 (no response/ignore), photostim != 0", and the AI states "For our decoder: keep early_lick, ignore, photostim trials (they are decoder inputs/outputs). Only exclude auto_water and free_water" — i.e. it keeps exactly those reference exclusions that the decoder task requires as outputs/inputs, and keeps the two water exclusions. The `obs_intervals` filter was added after the sample run flagged many all-zero trials: "Each NWB file contains recordings from one 'insertion' which covers only part of the session... The obs_intervals tell us the observation window [0, 1114.5]s, and only 160 go cues fall within that range." The two surviving all-zero trials were investigated and dismissed: "Trial 159 has go cue at 1112.6s but the unit's max spike is ~1107... The 1 trial with zero data is fine - it's an edge case."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (session-absolute seconds, ragged, read one unit at a time), restricted to units with `classification == 'good'`, together with `BehavioralEvents/go_start_times` for the valid trials, which sets the window for each trial.

ii.
```python
all_spike_times_obj = nwb.units['spike_times']
fr_all = np.zeros((n_good, n_valid_trials, n_bins), dtype=np.float32)

for i, unit_idx in enumerate(good_indices):
    st = all_spike_times_obj[unit_idx]
    fr_all[i] = bin_spikes_all_trials(st, go_cue_times, bin_edges)
```

iii. Step 2: "Units table: spike_times (absolute time), classification (good/unlabelled)..."; Step 5 mapping table: "units.spike_times (good only) → neural : Bin spikes in 50ms bins, aligned to go cue, window [-2.5, 1.5]s". Spike times are the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin are converted to firing rate in Hz by dividing by the bin width. For each good unit and each trial the code masks the unit's full spike vector to the trial window, subtracts the go-cue time, and histograms the aligned times into the 80 fixed bins. No smoothing, no baseline subtraction, no normalisation, no minimum-firing-rate threshold. Output dtype is `float32`; a trial with no spikes stays all-zero.

ii.
```python
def bin_spikes_all_trials(spike_times_unit, go_cue_times, bin_edges):
    n_trials = len(go_cue_times)
    n_bins = len(bin_edges) - 1
    bin_width = bin_edges[1] - bin_edges[0]
    fr = np.zeros((n_trials, n_bins), dtype=np.float32)

    begin = bin_edges[0]
    end = bin_edges[-1]

    for t in range(n_trials):
        gc = go_cue_times[t]
        # Get spikes in window
        mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
        if np.any(mask):
            aligned = spike_times_unit[mask] - gc
            counts, _ = np.histogram(aligned, bins=bin_edges)
            fr[t] = counts / bin_width

    return fr
```

iii. Step 1 of the notes identifies `sliding_histogram` in the reference code as the binning routine (which returns `binSpikes / bin_width`, i.e. Hz), and Step 5 decision 3 states: "Firing rate binning: 50ms non-overlapping bins (as specified by decoder task), NOT the 40ms/3.4ms stride from reference code. This is required by the decoder task specification." Decision 2 explains why the method paper's 2 Hz firing-rate cut is not applied: "that was specific to video prediction analysis, not a general data quality criterion".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `units/classification` string equals `'good'` (the spike-sorting QC classifier verdict of Chen, Liu et al. 2023) are kept; all other units are dropped before any binning. No thresholds on individual quality metrics, no `unit_quality` column use, and no firing-rate threshold. A session with zero good units returns `None` and is dropped entirely (this is what removes `sub-440958_ses-20190216T162508`, whose QC fields are all NaN). The retained total is 69,453 units, mean 401 per session.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)

if n_good == 0:
    io.close()
    return None
```

iii. Step 1: "QC: `classification == 'good'` in NWB corresponds to the classifier-based QC used in the reference code" (the reference code's `helper_get_neuron_id_area` with `qc_mode='classifier'`). Step 3 curation rules: "1. Classifier-based QC: keep only `classification == 'good'` units; 2. (Method paper only) Exclude neurons with avg firing rate < 2 Hz" — with the second rule deliberately not applied. Step 4 records the residual discrepancy with the data paper: "Good units | classifier QC | 69,453 | 69,943 | ~490 unit difference; likely minor NWB conversion differences. Will use NWB classification field."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams share one session-absolute clock, so alignment is done by subtracting each trial's go-cue time from the spike times before histogramming (equivalently, the window `[go - 2.5, go + 1.5)` is cut out of the spike train). No interpolation or per-stream offset correction. The go cue therefore sits exactly at t = 0, between bins 49 and 50.

ii.
```python
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
...
go_cue_times = go_cue_times_all[valid_trial_indices]
...
for t in range(n_trials):
    gc = go_cue_times[t]
    mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
    if np.any(mask):
        aligned = spike_times_unit[mask] - gc
        counts, _ = np.histogram(aligned, bins=bin_edges)
```

iii. Step 2: "Spike times in NWB are in absolute session time (not aligned to go cue)"; Step 3: "Spike times aligned to go cue (time 0)". The Step 10 check list includes "5. Verified temporal alignment (go cue at t=0)", and the `--show-processing` plots draw a red dashed line at t = 0 on the firing-rate heatmaps to make the alignment visible.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial, spanning -2.5 s to +1.5 s about the go cue. The edge grid is built once in `main()` with `np.linspace(-2.5, 1.5, 81)` and passed into every session, so all trials and sessions have identical 80-bin axes. This is a *different* binning from the reference code (40 ms width / 3.4 ms stride sliding window, window -3.0 to +3.5 s); the AI adopts the decoder-task binning instead. All other streams (tongue video at ~294 Hz, photostim, tone) are rebinned onto this same grid. `metadata['time_bin_size']` is recorded as 50.0 ms with `off_start = -2.5`, `off_end = 1.5`.

ii.
```python
BEGIN_TIME = -2.5
END_TIME = 1.5
BIN_WIDTH = 0.05
n_bins = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))
bin_edges = np.linspace(BEGIN_TIME, END_TIME, n_bins + 1)

print(f'Parameters: [{BEGIN_TIME}, {END_TIME}]s, {n_bins} bins of {BIN_WIDTH*1000:.0f}ms')
```

```python
'time_bin_size': BIN_WIDTH * 1000,
'temporal_alignment_event': 'Go cue onset',
'off_start': BEGIN_TIME,
'off_end': END_TIME,
```

iii. Step 5 decisions 3 and 4: "50ms non-overlapping bins (as specified by decoder task), NOT the 40ms/3.4ms stride from reference code. This is required by the decoder task specification." and "Time window: [-2.5, 1.5]s relative to go cue = 80 time bins. Different from reference code's [-3.0, 3.5]s but required by decoder task." The verification log confirms `T: mean 80.00, min 80, max 80` across all 173 sessions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the sample-epoch tone onsets) and `go_start_times`. For each valid trial the tone used is the **last** sample onset at or before the go cue (with a 10 ms tolerance). If no such onset exists, a fallback of -1.85 s (the nominal sample-to-go interval) is used.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
...
tone_onsets_rel = np.zeros(n_valid_trials, dtype=np.float64)
for t, trial_idx in enumerate(valid_trial_indices):
    gc = go_cue_times_all[trial_idx]
    candidates = sample_start_times[sample_start_times <= gc + 0.01]
    if len(candidates) > 0:
        tone_onsets_rel[t] = candidates[-1] - gc
    else:
        tone_onsets_rel[t] = -1.85  # fallback
```

iii. Step 5 mapping table: "sample_start relative to go cue → input[0]: time_from_tone_onset". Step 3 records "Sample-to-go-cue | 1.85 s | Confirmed from data: consistent 1.85s", which is also the origin of the fallback value. The "last onset before the go cue" rule handles the fact that a lick during the sample/delay epoch replays the epoch, so a trial can contain several sample onsets.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A per-bin continuous ramp: for every trial, the value of bin *k* is the bin centre (relative to the go cue) minus the tone offset (also relative to the go cue), i.e. seconds elapsed since that trial's tone onset. It is stored as the first row of the `(2, 80)` float32 input array. Values are negative before the tone and grow linearly; the observed range over the dataset is [-1.5, 11.9] s (large values occur on trials whose last preceding tone is several seconds earlier).

ii.
```python
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
...
for t in range(n_valid_trials):
    ...
    time_from_tone = (bin_centers - tone_onsets_rel[t]).astype(np.float32)
    input_data = np.stack([time_from_tone, photostim_all[t]], axis=0)
    input_trials.append(input_data)
```

iii. The notes' mapping table gives the formula explicitly: "t - (sample_start - go_cue) for each time bin t — Continuous, time-varying ramp". The decoder-task specification asks for "Time from tone onset in seconds (continuous, time-varying)", which the AI reads as a per-bin continuous value rather than a binary onset marker. Step 9 records the resulting range as a sanity check: "Input ranges correct: time_from_tone_onset [-1.5, ~12]".

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is defined directly on the neural bin grid: `bin_centers` are the centres of exactly the same 80 bins used to histogram the spikes, and the tone time is expressed relative to the same go cue. No separate alignment step or resampling is needed.

ii.
```python
bin_width = bin_edges[1] - bin_edges[0]
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
```
```python
time_from_tone = (bin_centers - tone_onsets_rel[t]).astype(np.float32)
```

iii. Implicit in the design (one shared `bin_edges` array is passed into `process_session` and used for spikes, photostim, tone and tongue). The `--show-processing` plot "Input: time from tone onset" overlays the ramp on the same time axis as the firing-rate heatmaps with the go cue marked, as a visual alignment check.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `BehavioralEvents/photostim_start_times` and `photostim_stop_times` — the absolute-time onsets/offsets of every photostimulation event in the session — rather than the trials-table `photostim_onset` / `photostim_duration` columns used by the human reference. Sessions with no optogenetics (6 of 174) have no such time series, and the code substitutes empty arrays so the input is all zeros.

ii.
```python
has_photostim = 'photostim_start_times' in be.time_series
if has_photostim:
    ps_start_times = be.time_series['photostim_start_times'].timestamps[:]
    ps_stop_times = be.time_series['photostim_stop_times'].timestamps[:]
else:
    ps_start_times = np.array([])
    ps_stop_times = np.array([])
```

iii. Step 5 mapping table: "photostim_start/stop times → input[1]: photostim_on : Binary: 1 if photostim active at time t, 0 otherwise — Time-varying binary", and decision 8: "Photostim timing: Photostim times are in absolute session time. Convert to go-cue-relative for each trial." Step 2 lists both the trials-table photostim fields and the BehavioralEvents photostim series as available; the AI chose the event series because it already carries absolute start/stop times.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time-varying series on the 80-bin grid: for each trial, every photostim event in the session that overlaps the trial window is converted to go-cue-relative times and the bins whose **centre** falls in `[start, stop)` are set to 1; overlapping contributions are accumulated and then clipped to [0, 1]. Stored as row 1 of the float32 input array. I checked this against the reference's trials-table construction on a session with 78 stim trials: the two produce bit-identical binary masks (780 "on" bins, 78 stim trials), because each stim event's onset equals `trial start_time + photostim_onset` to within 5e-13 s and durations are exactly 0.5 s.

ii.
```python
photostim_all = np.zeros((n_valid_trials, n_bins), dtype=np.float32)
if len(ps_start_times) > 0:
    for t in range(n_valid_trials):
        gc = go_cue_times[t]
        for ps_s, ps_e in zip(ps_start_times, ps_stop_times):
            ps_s_rel = ps_s - gc
            ps_e_rel = ps_e - gc
            if ps_s_rel < bin_edges[-1] and ps_e_rel > bin_edges[0]:
                photostim_all[t] += ((bin_centers >= ps_s_rel) & (bin_centers < ps_e_rel)).astype(np.float32)
    photostim_all = np.clip(photostim_all, 0, 1)
```

iii. The instructions ask for "Whether photostimulation is on at every time point (discrete, time-varying)", and the notes state the binary time-varying representation explicitly. Step 3 records the expected physiology ("Photostim duration 0.5 s — last 0.5 s of delay + 100ms ramp-down", "~25% randomly interleaved") which the AI used as the expected prevalence check. Because the code scans *all* session events per trial rather than only that trial's entry, it also captures stimulation that spills over from an adjacent trial into the window.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Each event's absolute start/stop is converted to go-cue-relative time by subtracting the same `go_cue_times[t]` used for the spike alignment, and compared against the shared `bin_centers`. So bin *k* of the photostim input covers exactly the same interval as bin *k* of the firing rates.

ii.
```python
gc = go_cue_times[t]
for ps_s, ps_e in zip(ps_start_times, ps_stop_times):
    ps_s_rel = ps_s - gc
    ps_e_rel = ps_e - gc
    if ps_s_rel < bin_edges[-1] and ps_e_rel > bin_edges[0]:
        photostim_all[t] += ((bin_centers >= ps_s_rel) & (bin_centers < ps_e_rel)).astype(np.float32)
```

iii. Step 5 decision 8: "Photostim times are in absolute session time. Convert to go-cue-relative for each trial." The `--show-processing` figure plots the photostim trace for the first stimulated trial against the same time axis with the go cue marked, so the stimulation is visibly in the late delay epoch just before t = 0.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the NWB files, so choice is reconstructed from two trials-table columns: `trial_instruction` (`'left'`/`'right'`, the side the tone instructed) and `outcome` (`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed side, a miss the opposite side, and any other outcome (`ignore`) means no lick. The lick-time series (`left_lick_times`, `right_lick_times`) are not used.

ii.
```python
trial_instruction = nwb.trials['trial_instruction'][:]
outcome_arr = nwb.trials['outcome'][:]
...
for t, trial_idx in enumerate(valid_trial_indices):
    instr = trial_instruction[trial_idx]
    out = outcome_arr[trial_idx]

    if out == 'hit':
        choices[t] = 0 if instr == 'left' else 1
    elif out == 'miss':
        choices[t] = 1 if instr == 'left' else 0
    else:
        choices[t] = 2  # no lick
```

iii. Step 5 mapping table: "trial_instruction + outcome → output[0]: choice : left=0, right=1, no_lick=2"; decision 9: "Choice determination: hit → lick matches instruction; miss → lick opposite to instruction; ignore → no lick." The resulting distribution (left 42.9%, right 42.2%, no lick 14.8%) is roughly balanced between the two sides, as expected for a randomised left/right task.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived code (0 = left, 1 = right, 2 = no lick) is computed once per trial and then broadcast across all 80 bins so that all four outputs share one `(4, 80)` int64 array per trial. `output_values[0] = ['left', 'right', 'no_lick']` names the codes. The intermediate `choices` array is float32 and cast to `int` when written.

ii.
```python
choices = np.zeros(n_valid_trials, dtype=np.float32)
...
output_data = np.stack([
    np.full(n_bins, int(choices[t]), dtype=np.int64),
    np.full(n_bins, int(outcomes[t]), dtype=np.int64),
    np.full(n_bins, int(early_vals[t]), dtype=np.int64),
    tongue_disc[t].astype(np.int64)
], axis=0)
```
```python
'output_values': [
    ['left', 'right', 'no_lick'],
    ...
],
```

iii. The 0 = left / 1 = right coding and the third "no lick" class follow the Decoder Task specification. The per-trial value is repeated across bins because the target format wants `(n_output, n_timepoints)` and the decoder reference expects integer class labels; the AI changed the output dtype to int64 after the sample run failed ("The float32 issue in decoder.py. I need to make my output values integers.").

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already contains exactly the three strings `'hit'`, `'miss'` and `'ignore'`.

ii.
```python
outcome_arr = nwb.trials['outcome'][:]
```

iii. Step 2 lists `outcome (hit/miss/ignore)` as a native trials-table column, and Step 5's mapping table maps it straight through. No derivation is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped through a fixed dictionary to 0 = ignore, 1 = miss, 2 = hit (with `.get(out, 0)` defaulting to `ignore` for anything unexpected) and written as row 1 of the per-trial output array, repeated across all 80 bins. Resulting distribution: hit 68.5%, miss 16.7%, ignore 14.8%.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
...
outcomes[t] = outcome_map.get(out, 0)
...
np.full(n_bins, int(outcomes[t]), dtype=np.int64),
```

iii. The code assignment follows the Decoder Task order ("Outcome (ignore, miss, hit, per-trial)"). In Step 10 the AI compared the resulting hit rate to the paper's: "Hit rate: 61326/89532 = 68.5% (includes early_lick/photostim trials which lower the apparent rate)" versus the methods' "84% correct rate" quoted for control trials only.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the trials-table `early_lick` column, whose values are the strings `'early'` and `'no early'`.

ii.
```python
early_lick_arr = nwb.trials['early_lick'][:]
```

iii. Step 2 identifies the column; Step 5's mapping table maps it straight to `output[2]`. Step 3 notes the reference code normally *excludes* early-lick trials (`get_regular_trial_mask`), but the AI keeps them because early lick is a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A binary per-trial flag: 1 if the string equals `'early'`, else 0, written as row 2 of the output array and repeated across the 80 bins. Resulting distribution: no 88.4%, yes 11.6%.

ii.
```python
early_vals = np.zeros(n_valid_trials, dtype=np.float32)
...
early_vals[t] = 1 if early_lick_arr[trial_idx] == 'early' else 0
...
np.full(n_bins, int(early_vals[t]), dtype=np.int64),
```

iii. Coding follows the Decoder Task ("Early lick (no, yes, per-trial)"), with `output_values[2] = ['no', 'yes']`. The AI treats it as a per-trial constant rather than marking the bin in which the early lick occurred.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, a ~294 Hz side-camera DeepLabCut trace whose `data` is `(n_frames, 3)` = (`tongue_x`, `tongue_y`, `tongue_likelihood`) with matching absolute `timestamps`. Column 1 (y) is the value; column 2 (likelihood) determines visibility. Column 0 (x) is read but unused. The jaw/nose tracking series are not used.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]
tongue_timestamps = tongue_ts_obj.timestamps[:]
```

iii. Step 2: "BehavioralTimeSeries: Camera0_side_TongueTracking (x, y, likelihood at ~300Hz/3.4ms)", consistent with the methods' "Behavior video rate 300 Hz" recorded in Step 3. Step 5's mapping table: "tongue_y from TongueTracking → output[3]: tongue_y_position".

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Two separate computations:
   1. **Class edges (per session):** the 40th and 60th percentiles of `tongue_y` over all *frames* in the session whose likelihood >= 0.9 (i.e. per-frame values, not bin averages). If no frame is visible, both edges are set to 0.
   2. **Per bin:** frames in the trial window are assigned to bins; for each bin, the **mean likelihood** of its frames is computed, and if that mean is >= 0.9 the bin's value is the **mean of `tongue_y` over all frames in the bin** — including frames whose individual likelihood is low — otherwise the bin is left as "not visible".
   
   So the quantity that is discretised (a bin-mean y, possibly contaminated by non-visible frames) is not the quantity the percentiles were computed on (per-frame visible y). Measured on session 1, the AI's procedure labels 93.6% of bins "not visible" versus 79.4% for the human reference's procedure, and the visible bins split 9%/35%/56% across the three classes rather than the ~40/20/40 the percentile definition implies. Dataset-wide the AI's output is 90.8% "not visible" (reference reports ~75%).

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9

# Per-session tongue y percentiles (visible tongue only)
visible_mask_all = tongue_data[:, 2] >= TONGUE_LIKELIHOOD_THRESH
if np.sum(visible_mask_all) > 0:
    visible_y = tongue_data[visible_mask_all, 1]
    tongue_y_p40 = np.percentile(visible_y, 40)
    tongue_y_p60 = np.percentile(visible_y, 60)
else:
    tongue_y_p40 = 0.0
    tongue_y_p60 = 0.0
```
```python
        for b in range(n_bins):
            frame_mask = bin_idx == b
            if not np.any(frame_mask):
                continue
            frames = trial_data[frame_mask]
            avg_likelihood = np.mean(frames[:, 2])
            if avg_likelihood >= likelihood_thresh:
                avg_y = np.mean(frames[:, 1])
```

iii. Step 5 decision 6: "Tongue visibility: Use likelihood threshold of 0.9 (DeepLabCut convention). Tongue with likelihood < 0.9 is 'not visible' (category 3)."; decision 7: "Tongue percentiles: Compute 40th and 60th percentiles over all visible tongue y-positions in the session (across all time points and trials), then discretize." The notes give no rationale for using the bin-mean likelihood as the visibility test, nor for averaging y over non-visible frames, and the resulting class imbalance (Step 9/verification: `below_p40 0.014, p40_to_p60 0.028, above_p60 0.050, not_visible 0.908`) was reported but not questioned.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes on the bin-mean y: `0` if `avg_y < p40`, `1` if `p40 <= avg_y <= p60`, `2` if `avg_y > p60`, and `3` ("not visible") if the bin has no frames at all or if the bin's mean likelihood is below 0.9. The array is initialised to 3 so "not visible" is the default. Percentiles are per session, as required. Note the boundary convention is `<= p60` for class 1 (the reference uses `np.digitize`, i.e. `< p60`); this is immaterial.

ii.
```python
result = np.full((n_trials, n_bins), 3, dtype=np.int64)  # default: not visible
...
            avg_likelihood = np.mean(frames[:, 2])
            if avg_likelihood >= likelihood_thresh:
                avg_y = np.mean(frames[:, 1])
                if avg_y < tongue_y_p40:
                    result[t, b] = 0
                elif avg_y <= tongue_y_p60:
                    result[t, b] = 1
                else:
                    result[t, b] = 2
```
```python
'output_values': [
    ...
    ['below_p40', 'p40_to_p60', 'above_p60', 'not_visible'],
],
```

iii. The four classes and the 40th/60th per-session percentile split are taken verbatim from the Decoder Task specification. The 0.9 likelihood cut is justified in the notes as the "DeepLabCut convention"; applying it to the per-bin *average* likelihood (rather than per frame) is not discussed. The AI's Step 12 review accepted the result on the grounds that decoding was above chance ("Tongue y-position (0.66 val) ... 0.66 balanced across 4 classes (chance 0.25) is strong") without examining the class imbalance.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as spikes and events. For each trial the code `searchsorted`s the camera timestamps at `go + bin_edges[0]` and `go + bin_edges[-1]` to slice out that trial's frames, then assigns each frame to a bin with `searchsorted(abs_edges, t, 'right') - 1`, clipped to [0, 79] — the same go-cue-anchored 80-bin grid used for the firing rates. No interpolation. Bins with no frames (e.g. leading bins on trials where the video started less than 2.5 s before the go cue, since the video is trial-gated) stay in the "not visible" class.

ii.
```python
    for t in range(n_trials):
        gc = go_cue_times[t]
        abs_edges = gc + bin_edges

        # Find frame indices for this trial's window
        i_start = np.searchsorted(tongue_timestamps, abs_edges[0])
        i_end = np.searchsorted(tongue_timestamps, abs_edges[-1])

        if i_start >= i_end:
            continue

        trial_ts = tongue_timestamps[i_start:i_end]
        trial_data = tongue_data[i_start:i_end]

        # Assign each frame to a bin
        bin_idx = np.searchsorted(abs_edges, trial_ts, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
```

iii. Not discussed at length in the notes beyond the mapping table entry ("Discretized per session ... Time-varying"); the alignment is implicit in reusing `bin_edges` and `go_cue_times`. The `--show-processing` figure plots the discretised tongue trace for the first trial that has any visible bin, against the same time axis with the go cue marked, as the visual alignment check.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled, mostly implicitly:
   - **Session with no QC labels** (`classification` all NaN): `classification == 'good'` is all-False, `n_good == 0`, and the session is dropped (`return None`). This is the one dropped session; the notes mis-attribute it to the `obs_intervals` filter.
   - **Trials outside the ephys recording**: dropped by the `obs_intervals` span filter (see 1-e). Sessions left with < 2 trials are dropped.
   - **Trials that still have zero spikes** (2 of 89,532, at recording boundaries): kept, and the resulting verifier warnings are accepted as "expected edge cases" after investigation.
   - **Sessions without optogenetics** (6 files): `has_photostim` guard produces an all-zero photostim input instead of raising.
   - **No tone onset before the go cue**: falls back to the nominal -1.85 s.
   - **Unmapped CCF annotation**: `classify_brain_region` returns `'Unknown'` and the code falls back to parsing the electrode group's JSON `brain_regions` field inside a `try/except`; after the Step 10 mapping fixes no unit falls through. Note `classify_brain_region` calls `.lower()` on its argument, so a non-string `anno_name` would raise — this is only safe because such units exist solely in the dropped session.
   - **Tongue frames with low likelihood**: the bin becomes class 3 "not visible" rather than being imputed.

ii.
```python
if n_good == 0:
    io.close()
    return None
```
```python
if n_valid_trials < 2:
    io.close()
    return None
```
```python
has_photostim = 'photostim_start_times' in be.time_series
if has_photostim:
    ...
else:
    ps_start_times = np.array([])
    ps_stop_times = np.array([])
```
```python
    else:
        tone_onsets_rel[t] = -1.85  # fallback
```
```python
    if region == 'Unknown' and electrode_group_location:
        try:
            loc = json.loads(electrode_group_location)
            eg_region = loc.get('brain_regions', '').replace('left ', '').replace('right ', '')
            if eg_region:
                return eg_region
        except (json.JSONDecodeError, KeyError):
            pass
```

iii. The notes' rationale is mostly given per case in Steps 7-10: the `obs_intervals` filter was introduced after the sample run produced many all-zero trials ("Each NWB file contains recordings from one 'insertion' which covers only part of the session"); the two residual all-zero trials were traced to a recording that stops a few seconds before `obs_end` and judged acceptable; the brain-region fallback and `Unknown` class were introduced so no unit is silently lost, and the Step 10 review then eliminated all `Unknown` units by fixing the Allen-CCF mapping (Hypothalamus 200 → 815, exactly matching the data paper).

## 10-a. What are the most time-consuming steps of the code?

i. The script prints a per-stage breakdown for every session (`read`, `spike`, `tongue`, `assemble`). Spike binning dominates overwhelmingly: e.g. `472 units, 448 trials, 7.9s (read:0.4s spike:7.2s tongue:0.3s assemble:0.0s)` and `368 units, 626 trials, 10.6s (read:0.4s spike:9.7s tongue:0.5s assemble:0.0s)` — 85-92% of per-session time. File reading is only ~0.3-0.4 s per session and the tongue pass ~0.1-0.5 s. Full conversion took **1015 s (5.9 s/session)** plus pickling of the 11.9 GB output; the human reference does the same work in 247 s. The AI's Step 7 estimate (3.7 s/session, "~10 minutes for all 174 sessions") was exceeded by ~1.6x, and the instructions' 15-minute budget was exceeded, without triggering the re-optimisation the instructions call for.

ii.
```python
    t1 = time.time()
    print(f'  {os.path.basename(nwb_path)}: {n_good} units, {n_valid_trials} trials, '
          f'{t1-t0:.1f}s (read:{t_read-t0:.1f}s spike:{t_spike-t_read:.1f}s '
          f'tongue:{t_tongue-t_spike:.1f}s assemble:{t1-t_tongue:.1f}s)')
```

iii. The AI did profile and act once: the first version ran at ~96 s/session and was rewritten ("The sample works but is too slow (~96s/session → ~4.6h for 174 sessions). Let me optimize the bottlenecks." → "3.7s/session now"). After that it did not revisit timing, and CONVERSION_NOTES contains no discussion of the final 1015 s runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four nested/per-item loops remain, none of them documented as such:
   1. **`bin_spikes_all_trials`: the per-trial loop.** For each of ~500 trials it evaluates a boolean mask over the unit's *entire* spike vector. This is O(n_trials x n_spikes) per unit and is the 85-92% bottleneck. The human reference vectorises the whole trial dimension with a single `np.searchsorted` over the flattened edge array plus `np.diff`, which is why it is ~4x faster overall.
   2. **`bin_tongue_all_trials`: the inner `for b in range(n_bins)` loop**, with a fresh `bin_idx == b` mask per bin (80 masks per trial); a single `np.bincount` over the bin index would do it in one pass.
   3. **The tone-onset loop**, which recomputes `sample_start_times <= gc + 0.01` over the full event array for every trial; one `np.searchsorted(sample, go)` replaces the whole loop.
   4. **The photostim double loop** over trials x all session stim events (O(n_trials x n_events)), where only the trial's own event is relevant.
   
   Session processing itself is also serial; the instructions suggest parallel processing where beneficial, and sessions are embarrassingly parallel.

ii.
```python
    for t in range(n_trials):
        gc = go_cue_times[t]
        mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
        if np.any(mask):
            aligned = spike_times_unit[mask] - gc
            counts, _ = np.histogram(aligned, bins=bin_edges)
            fr[t] = counts / bin_width
```
```python
        for b in range(n_bins):
            frame_mask = bin_idx == b
            if not np.any(frame_mask):
                continue
```
```python
for t, trial_idx in enumerate(valid_trial_indices):
    gc = go_cue_times_all[trial_idx]
    candidates = sample_start_times[sample_start_times <= gc + 0.01]
```
```python
        for t in range(n_valid_trials):
            gc = go_cue_times[t]
            for ps_s, ps_e in zip(ps_start_times, ps_stop_times):
```

iii. The AI's only stated efficiency rationale is the one-off rewrite at Step 7 ("Let me optimize the bottlenecks", 96 s → 3.7 s per session), achieved by hoisting the per-unit `spike_times` read out of the trial loop and by slicing the tongue frames with `searchsorted`. The function is even named `bin_spikes_all_trials` / `bin_tongue_all_trials` as though the trial dimension were vectorised, but both still loop over trials. No further profiling or vectorisation was attempted.

## 10-c. What processing does the code repeat multiple times?

i. The main repetition is re-scanning whole arrays once per trial:
   - the unit's full spike vector is compared against the window bounds once per trial (≈500 full-array passes per unit, ~400 units per session);
   - `sample_start_times` is fully re-scanned once per trial to find the preceding tone;
   - all session photostim events are re-tested once per trial;
   - inside the tongue binning, the `bin_idx` array is re-scanned 80 times per trial (once per bin).
   
   Per-session, no quantity is computed twice in different places, and the bin grid is built once in `main()` and passed down. At the process level, the full conversion was run twice end-to-end (once before, once after the brain-region mapping fix), i.e. ~34 min of compute for a change that only affects `brain_region_idx`; a cheaper re-mapping of the existing output was possible.

ii.
```python
    for t in range(n_trials):
        gc = go_cue_times[t]
        mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
```
```python
    candidates = sample_start_times[sample_start_times <= gc + 0.01]
```
```python
        for b in range(n_bins):
            frame_mask = bin_idx == b
```

iii. Not discussed in CONVERSION_NOTES. The repeated full-array scans are an artefact of the "mask then histogram" idiom chosen in the Step 7 optimisation; the AI's stated justification was only that the resulting speed (3.7 s/session in the sample) was acceptable.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts, all cheap:
   - `subject_desc = nwb.subject.description` is read and carried in the per-session result dict but never written to the output.
   - `electrode_groups = nwb.units['electrode_group'][:]` is dereferenced for every good unit purely to feed the `Unknown` fallback in `get_brain_region_for_unit`; after the Step 10 CCF-mapping fixes no unit is `Unknown`, so the fallback (and the JSON parse it guards) never fires.
   - The tongue `x` column is read and carried through slicing but never used.
   - `session_idx` and `show_processing` are passed into `process_session` and ignored there.
   - `tongue_y_p40` / `tongue_y_p60` are returned per session for plotting only.
   - Per-trial scalar outputs (choice, outcome, early lick) are materialised as 80-long int64 vectors; this is required by the chosen `(n_output, n_timepoints)` representation, but it is also what makes the pickle 11.9 GB.
   - All firing rates are kept in float32 and all outputs in int64 (int8 would suffice for the outputs).
   
   Nothing computed is scientifically wasted — every field except `subject_desc` reaches the output in some form.

ii.
```python
    subject_id = nwb.subject.subject_id
    subject_desc = nwb.subject.description
...
        'subject_desc': subject_desc,
```
```python
    electrode_groups = nwb.units['electrode_group'][:]
    ...
        eg = electrode_groups[idx]
        eg_loc = eg.location if hasattr(eg, 'location') else ''
        region = get_brain_region_for_unit(an, eg_loc)
```
```python
def process_session(nwb_path, bin_edges, n_bins, show_processing=False, session_idx=0):
```

iii. Not discussed in CONVERSION_NOTES. The electrode-group fallback was introduced deliberately as a safety net for unmapped annotations ("Brain region mapping: Use electrode_group brain_regions ... combined with CCF anno_name for finer mapping"), and simply became redundant once the Step 10 fixes brought every annotation into an explicit category.
