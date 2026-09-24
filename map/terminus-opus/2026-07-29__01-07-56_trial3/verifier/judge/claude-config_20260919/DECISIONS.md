# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is read directly from the published NWB files (DANDI 000363), one file per session, using `pynwb`. All files are found with a single relative glob `data/sub-*/sub-*.nwb` (so the script must be run from `/app`), sorted for determinism, giving 174 files. In `--sample` mode the AI hard-codes two specific files (indices 1 and 4) rather than the first two, because the first file fails its session-performance filter. Each file is opened once with `pynwb.NWBHDF5IO(...).read()` and everything needed is pulled from `nwb.subject`, `nwb.trials`, `nwb.units`, and `nwb.acquisition` (`BehavioralEvents`, `BehavioralTimeSeries`). The file handle is closed explicitly at the end of `process_session` (and on each early-exit path).

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
print(f'Found {len(nwb_files)} NWB files')

if args.sample:
    nwb_files = [nwb_files[1], nwb_files[4]]
```
```python
io = pynwb.NWBHDF5IO(nwb_file, 'r')
nwb = io.read()

# Extract subject info
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description

# Get trials info
n_trials_total = len(nwb.trials)
trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
early_lick = nwb.trials['early_lick'][:]
auto_water = nwb.trials['auto_water'][:]
free_water = nwb.trials['free_water'][:]
photostim_power = nwb.trials['photostim_power'][:]
```

iii. From CONVERSION_NOTES Step 2: "NWB files: `data/sub-{id}/sub-{id}_ses-{datetime}_behavior+ecephys[+ogen].nwb`; Each NWB file contains: trials, units, BehavioralEvents, BehavioralTimeSeries". The AI verified in Step 2 that the glob yields 174 files / 28 subjects / 272,227 units / 94,990 trials, and checked these against the papers in Step 3–4. The reference code loads `.mat` files, which are not distributed here; the AI documents this in Step 10 Check 3: "(a) Data loading: We load from NWB instead of .mat, but extract same variables".

## 1-b. How are the data split into subjects?

i. The subject is read from `nwb.subject.subject_id` (the numeric DANDI id, e.g. `'440956'`) for each session. At assembly the unique ids are sorted to form `subjects`, and `subject_idx` holds each session's index into that list. 28 subjects are found, matching the dandiset and the papers.

ii.
```python
subject_id = nwb.subject.subject_id
```
```python
all_subject_ids = [s['subject_id'] for s in all_sessions]
unique_subjects = sorted(set(all_subject_ids))
subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
...
subject_idx_list = [subject_to_idx[s['subject_id']] for s in all_sessions]
...
'subjects': unique_subjects,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2/Step 9 record "Subjects | 28 | 28 | Yes" as a consistency check against the dandiset and the papers. The subject id is the canonical identifier stored in the file (and the `sub-*` directory name is derived from it), so no separate grouping step is needed.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is inferred. Session order follows the sorted file list. Sessions are then *dropped* by three criteria before being emitted: fewer than 2 classifier-'good' units, fewer than 2 trials covered by the neural recording, and behavioural performance criteria taken from the data paper (>65% correct on control/no-early-lick trials, and at least 50 correct left and 50 correct right trials). 151 of 174 sessions survive (1 for no good units, 15 for correct rate, 5 for the L/R count, 2 others). Session identity is recorded in `metadata['session_info']` by NWB file path rather than by `nwb.identifier`.

ii.
```python
if n_good < 2:
    print(f'  Skipping: only {n_good} good units')
    io.close(); return None
...
if len(valid_trial_indices) < 2:
    print(f'  Skipping: only {len(valid_trial_indices)} trials with neural coverage')
    io.close(); return None
...
if correct_rate < 0.65:
    print(f'  Skipping: correct rate {correct_rate:.3f} < 0.65')
    io.close(); return None

if correct_left < 50 or correct_right < 50:
    print(f'  Skipping: correct_left={correct_left}, correct_right={correct_right} (need >=50)')
    io.close(); return None
```
```python
'session_info': [
    {'nwb_file': s['nwb_file'], 'subject_id': s['subject_id'],
     'n_neurons': s['n_good'], 'n_trials': s['n_trials_valid'],
     'n_trials_total': s['n_trials_total'], 'correct_rate': float(s['correct_rate'])}
    for s in all_sessions
],
```

iii. CONVERSION_NOTES Step 3 quotes the data paper: "overall behavioral performance (> 65%)" and "at least 50 correct lick left and lick right trials each", and Step 6 lists "Session selection: >65% correct rate, >=50 correct L/R trials, >=2 good units". Trajectory step 106: "The paper says 173 sessions – this is just the count of sessions with good units, NOT after applying behavioral criteria… So the paper's 173 sessions is the total with good units, and my 151 is after behavioral filtering."

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`), one row per behavioural trial, and the per-trial go cue is taken from the matching index of `BehavioralEvents/go_start_times`. The code assumes positional correspondence between the trials table and the go-cue event stream (i.e. `go_times[trial_idx]`) without asserting that the lengths are equal. Only trials passing the recording-coverage mask are emitted, each as one `(n_neurons, 80)` neural array plus a `(2, 80)` input and a `(4, 80)` output array.

ii.
```python
n_trials_total = len(nwb.trials)
...
go_times = beh_events.time_series['go_start_times'].timestamps[:]

# Filter trials by recording coverage
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
valid_trial_indices = np.where(valid_trial_mask)[0]
...
for local_idx, trial_idx in enumerate(valid_trial_indices):
    go_time = go_times[trial_idx]
```

iii. CONVERSION_NOTES Step 2 lists the trials table fields; trajectory step 21 notes that `go_start_times` has exactly one event per trial ("368 events = number of trials") whereas `sample_start_times` has more ("405 events… because of early lick replays"), so the go cue is the unambiguous per-trial anchor.

## 1-e. How are trials filtered based on quality controls?

i. A single trial-level filter is applied: the whole analysis window must lie inside the span of actual spike times of the good units, i.e. `go + (-2.5) >= min(all good-unit spike times)` and `go + 1.5 <= max(all good-unit spike times)`. `obs_intervals` is *not* used despite the call-site comment saying so (the helper's docstring is explicit: "using actual spike times"). No behavioural trial filter is applied — early-lick, no-response (`ignore`), auto-water, free-water and photostim trials are all kept, because early lick / outcome / photostim are decoder outputs or inputs. Session-level behavioural criteria (1-c) remove 23 sessions. Result: 81,188 trials over 151 sessions. 2,304 of these trials (2.8%) contain all-zero neural data; the AI investigated and chose not to fix this.

ii.
```python
def get_recording_range(nwb, good_indices):
    """Get the recording time range for good units using actual spike times."""
    spike_times_all = nwb.units['spike_times'][:]
    rec_start = np.inf; rec_end = -np.inf
    for i in good_indices:
        st = np.array(spike_times_all[i])
        if len(st) > 0:
            rec_start = min(rec_start, st.min())
            rec_end = max(rec_end, st.max())
    ...
    return rec_start, rec_end
```
```python
# Get recording range from obs_intervals
rec_start, rec_end = get_recording_range(nwb, good_indices)
...
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
```

iii. CONVERSION_NOTES Step 4: "obs_intervals | Some sessions have partial recording coverage | Filter trials by actual spike time range"; Step 5 Key Decision 1: "Include all trials: Unlike reference code's `get_regular_trial_mask`, we include all trials because early_lick, outcome, and photostim are decoder outputs/inputs"; Key Decision 2: "Recording coverage: Filter trials by actual spike time range (not obs_intervals end) to avoid all-zero neural data". Trajectory step 114 on the residual all-zero trials: "this could be caused by the recording gap issue… My current approach only checks if the go_time is within the overall recording range… but doesn't check for gaps within the recording. However, 2.8% is a small fraction and shouldn't significantly affect decoder training."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (session-absolute seconds) for the units whose `units/classification == 'good'`, together with `BehavioralEvents/go_start_times`, which sets the per-trial window. `units/anno_name` supplies the brain-region label for each retained unit.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
...
anno_names = nwb.units['anno_name'][:]
brain_regions_raw = [anno_names[i] for i in good_indices]
...
spike_times_all = nwb.units['spike_times'][:]
spike_times_good = []
for i in good_indices:
    st = np.array(spike_times_all[i], dtype=np.float64)
    spike_times_good.append(np.sort(st))
```

iii. CONVERSION_NOTES Step 2: "Key fields: classification ('good'/'unlabelled'), anno_name (brain region), spike_times, obs_intervals". Trajectory step 20: "459 out of 1952 units are 'good' by classifier QC (23.5%, close to the 25.9% mentioned in methods)… Spike times are in absolute time (seconds from session start)".

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each trial, absolute bin edges `go + (-2.5) + k*0.05` (81 edges) are built; for each neuron, `searchsorted` narrows the spike array to the window and `np.histogram` counts spikes per bin; counts are divided by the 50 ms bin width. Rates are stored `float32`. No smoothing, normalisation, z-scoring or baseline subtraction. Spike arrays are re-sorted defensively before use.

ii.
```python
def compute_firing_rates_session(spike_times_list, go_times, bin_width=BIN_WIDTH,
                                  window_start=WINDOW_START, window_end=WINDOW_END):
    """Compute firing rates for all neurons across all trials."""
    ...
    for trial_idx in range(n_trials):
        go_time = go_times[trial_idx]
        bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        for i, spikes in enumerate(spike_times_list):
            if len(spikes) > 0:
                left = np.searchsorted(spikes, bin_edges[0])
                right = np.searchsorted(spikes, bin_edges[-1])
                if left < right:
                    counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
                    fr[i, :] = counts / bin_width
        fr_all.append(fr)
    return fr_all
```

iii. CONVERSION_NOTES Step 1 identifies the reference function `sliding_histogram` ("Compute firing rates from spike times") and Step 10 Check 3(d): "Binning: 50ms bins as specified (reference uses 100ms bins with 50ms stride, or 40ms bins with 3.4ms stride)" — i.e. the AI deliberately used the instruction's non-overlapping 50 ms bins instead of the reference's sliding window. Spot-checks in Step 10 Check 2 confirm converted rates equal rates recomputed from the raw NWB file ("np.allclose = True, max diff = 0.0").

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept if `classification == 'good'`, the verdict of the Chen/Liu spike-sorting QC classifier. No individual metric thresholds (ISI violation, drift, amplitude, presence ratio) are applied, and `unit_quality` (the older 'good'/'multi' label) is explicitly not used. A session with fewer than 2 good units is dropped. No zero-variance / low-rate neuron removal is done (unlike the reference's `check_fr`). This retains 60,324 units across the 151 retained sessions (69,453 across all 173 sessions before the session-level behavioural filter).

ii.
```python
# Get units info - filter by classifier QC
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
n_good = np.sum(good_mask)

if n_good < 2:
    print(f'  Skipping: only {n_good} good units')
    io.close()
    return None
```

iii. CONVERSION_NOTES Step 3 "Curation Steps — Neuron curation: classification == 'good' (classifier-based QC)"; Step 1 records "QC mode is 'classifier' (region-specific classifiers)". Trajectory step 20: "`classification`: 'good' (459) vs 'unlabelled' (1493) – this is the classifier-based QC. Only 'good' units should be kept. `unit_quality`: 'good' vs 'multi' – this is the old QC method." Step 4 checks the resulting 69,453 good units / 25.5% against the paper's 69,943 / 25.9%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset. Spike times, go cues, photostim events and camera timestamps all share the session-absolute NWB clock, so alignment is done by adding the relative bin grid to each trial's `go_start_times` value; no resampling, interpolation, or per-stream offset correction.

ii.
```python
go_times = beh_events.time_series['go_start_times'].timestamps[:]
...
valid_go_times = go_times[valid_trial_indices]
fr_all = compute_firing_rates_session(spike_times_good, valid_go_times)
```
```python
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```
```python
'temporal_alignment_event': 'Go cue onset',
'off_start': WINDOW_START,   # -2.5
'off_end': WINDOW_END,       # +1.5
```

iii. CONVERSION_NOTES Step 3 "Temporal alignment: Go cue onset (time 0)" and Step 10 Check 3(c): "Temporal alignment: Aligned to go cue onset, consistent with reference". Trajectory step 20 established that "Spike times are in absolute time (seconds from session start)", so the go-cue timestamp can be used directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins over -2.5 s to +1.5 s relative to the go cue = exactly 80 bins per trial for every trial and session. Spikes are binned once at this resolution straight from spike times; there is no intermediate binning and no rebinning/downsampling. The same 80-bin grid is reused for the two inputs and for the tongue output. `metadata['time_bin_size'] = 50.0` (ms).

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins for firing rates
WINDOW_START = -2.5  # seconds before go cue
WINDOW_END = 1.5    # seconds after go cue
N_TIMEBINS = int((WINDOW_END - WINDOW_START) / BIN_WIDTH)  # 80 bins
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
```
```python
'time_bin_size': BIN_WIDTH * 1000,
```

iii. Set by the Decoder Task section of the instructions. Trajectory step 117: "the task says to use 50ms-width bins… My BIN_WIDTH = 0.050 seconds = 50ms. This is correct. Also, let me verify the window: -2.5s to +1.5s = 4.0s total, with 50ms bins = 80 bins." The verification log confirms Mean/Min/Max T = 80 for all 151 sessions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is **not** derived from any raw per-trial variable. The AI hard-codes the tone onset at a constant −1.85 s relative to the go cue (0.65 s sample epoch + 1.2 s delay), inferred by inspecting the first five trials of one session, and never reads `sample_start_times` in the conversion script. The resulting input is therefore identical for every trial in every session.

ii.
```python
TONE_ONSET_REL_GO = -1.85  # tone onset relative to go cue (seconds)
```
```python
# Time from tone onset (same for all trials)
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. CONVERSION_NOTES Step 3: "Trial timing: sample -1.85 to -1.20, delay -1.20 to 0.00, response 0.00 to 1.50"; Step 5 mapping table: "BIN_CENTERS - TONE_ONSET | input[0] | Time from tone onset in seconds | Tone at -1.85s rel to go". Trajectory step 29: "Sample start: -1.85s (tone onset)… time_from_tone = t_rel_to_go + 1.85"; step 117: "My implementation uses a fixed time from tone onset that's the same for every trial (since tone onset is always at -1.85s relative to the go cue). This is correct because the tone onset is fixed relative to the go cue." (The AI had earlier observed in step 21 that `sample_start_times` has *more* events than trials "because of early lick replays", but did not revisit the constant.)

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. One vectorised subtraction: the 80 go-cue-relative bin centres minus the constant −1.85, cast to `float32`, computed once per session and reused for every trial. Values run from −0.625 s to +3.325 s in every trial, as confirmed in the verification log (`0: [-0.6, 3.3]` for all 151 sessions).

ii.
```python
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
...
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
input_trials.append(input_data)
```

iii. CONVERSION_NOTES Step 10 Check 2: "Input: Time from tone onset range [-0.625, 3.325] matches expected [-2.5+1.85, 1.5+1.85]". The AI treated the fixed offset as a property of the task structure, so no per-trial computation was thought necessary.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the input uses `BIN_CENTERS`, the centres of exactly the same go-cue-relative 80-bin grid used to build the neural bin edges, so bin *k* of the input covers the same interval as bin *k* of the firing rates. No separate alignment step exists.

ii.
```python
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
...
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```
```python
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```

iii. Implicit in the design; the `--show-processing` plots draw the neural raster, the tone-onset input and the go cue on the same time axis with vertical markers at 0 and at `TONE_ONSET_REL_GO` so that alignment can be inspected visually (CONVERSION_NOTES Step 7 "Processing Plots Review").

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the event streams `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times` (absolute timestamps), rather than from the trials-table `photostim_onset` / `photostim_duration` columns. The trials-table `photostim_power` column is read separately, but only to identify control trials for the session-level performance criterion. Sessions without optogenetics still have the event streams present but empty.

ii.
```python
photostim_starts_all = beh_events.time_series['photostim_start_times'].timestamps[:]
photostim_stops_all = beh_events.time_series['photostim_stop_times'].timestamps[:]
```
```python
is_control = np.array([str(p) == 'N/A' for p in photostim_power])
```

iii. CONVERSION_NOTES Step 5 mapping table: "photostim_start/stop | input[1] | Binary: 1 if photostim on, 0 otherwise | Per time bin". Trajectory step 23 explores both representations (trials-table onset relative to trial start vs. the absolute event stream) and step 27 confirms "sessions without ogen still have the photostim fields but with 0 stim trials", so one code path covers both file types.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, stim events overlapping the trial's 4 s window are selected; each surviving `[start, stop]` pair sets to 1 every bin whose centre lies within it (inclusive at both ends); all other bins stay 0. Non-stim trials get an all-zero vector without entering the loop. The result is a binary `float32` time series, not a per-trial flag.

ii.
```python
def get_photostim_timeseries(photostim_start_times, photostim_stop_times,
                              go_time, bin_centers=BIN_CENTERS):
    """Create binary photostimulation time series for a trial."""
    n_bins = len(bin_centers)
    photostim = np.zeros(n_bins, dtype=np.float32)
    abs_bin_centers = go_time + bin_centers
    for start, stop in zip(photostim_start_times, photostim_stop_times):
        mask = (abs_bin_centers >= start) & (abs_bin_centers <= stop)
        photostim[mask] = 1.0
    return photostim
```
```python
stim_mask = (photostim_stops_all > trial_window_start) & (photostim_starts_all < trial_window_end)
if np.any(stim_mask):
    ...
    photostim_ts = get_photostim_timeseries(trial_stim_starts, trial_stim_stops, go_time)
else:
    photostim_ts = np.zeros(N_TIMEBINS, dtype=np.float32)
```

iii. The instructions require "Whether photostimulation is on at every time point (discrete, time-varying)". CONVERSION_NOTES Step 10 Check 2: "Photostim: Correctly zero for non-stim trials, correctly 1 for stim trials"; the verification log shows input 1 ranging [0,1] in all but three sessions, which are [0,0] (no-ogen sessions).

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The bin centres are converted to absolute session time (`go_time + BIN_CENTERS`) and compared directly against the absolute stim start/stop timestamps, so the photostim vector sits on exactly the same go-cue-locked grid as the firing rates. Because the selection is done on absolute times with an overlap test, stimulation from a neighbouring trial that physically extends into this trial's 4 s window would also be marked.

ii.
```python
abs_bin_centers = go_time + bin_centers
for start, stop in zip(photostim_start_times, photostim_stop_times):
    mask = (abs_bin_centers >= start) & (abs_bin_centers <= stop)
    photostim[mask] = 1.0
```

iii. Same rationale as 2-d: all NWB streams share one absolute clock, so no offset correction is needed. The `--show-processing` plot panel "Input: Photostim" is drawn against the same axis as the neural raster.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From the trials-table column `trial_instruction` alone, i.e. the **tone-instructed** lick direction, mapped left→0, right→1. The animal's actual lick direction is not derived: `outcome` is not combined with the instruction, and the available `BehavioralEvents/left_lick_times` and `right_lick_times` streams (which the AI found in trajectory step 23) are not used. There is no "no lick" class, even though the instructions specify three values (left, right, no lick) and 12.5% of retained trials have `outcome == 'ignore'`.

ii.
```python
CHOICE_MAP = {'left': 0, 'right': 1}
...
trial_instruction = nwb.trials['trial_instruction'][:]
...
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
```
```python
'output_values': [
    ['left', 'right'],
    ...
```

iii. CONVERSION_NOTES Step 5 mapping table: "trial_instruction | output[0] | left=0, right=1 | Per trial, broadcast to all time bins"; trajectory step 42 lists "Choice: left=0, right=1 (from trial_instruction)". No justification is given anywhere in the notes or trajectory for equating the instructed side with the animal's choice, nor for dropping the third class; trajectory step 21 shows the AI did look for a lick-direction field ("Let me also check if there's a `lick_direction` field or if I need to infer it") but the question is never resolved.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A dictionary lookup per trial with a silent fallback to 0 (`left`) for any unrecognised string, then the scalar is broadcast across all 80 bins into row 0 of the `(4, 80)` `int64` output array. The stored `output_values[0]` is `['left', 'right']`. Resulting distribution over the full dataset: 48.9% left, 51.1% right.

ii.
```python
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
output_choice.append(choice)
...
out_arr = np.stack([
    np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64),
    ...
], axis=0)
```

iii. The instructions allow per-trial outputs to be broadcast across time; `int64` was chosen after the AI found (trajectory steps 58–63) that `decoder.py`'s `print_data_summary` indexes `output_values` with the raw unique values, which fails for `float32`. CONVERSION_NOTES Step 10 Check 2 states "Output: Trial instruction, outcome, early_lick all match NWB data exactly" — i.e. the check verified the copy of `trial_instruction`, not that it represents the animal's choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already holds exactly the three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome = nwb.trials['outcome'][:]
```

iii. CONVERSION_NOTES Step 5: "outcome | output[1] | ignore=0, miss=1, hit=2 | Per trial, broadcast". The column matches the instruction's categories one-to-one, so no derivation was needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Dictionary mapping `{'ignore': 0, 'miss': 1, 'hit': 2}` (with a silent fallback to 0) broadcast across all 80 bins into row 1. Full-dataset distribution: ignore 12.5%, miss 15.4%, hit 72.1%.

ii.
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
...
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
output_outcome.append(out)
...
np.full(N_TIMEBINS, output_outcome[t_idx], dtype=np.int64),
```

iii. The 0/1/2 coding follows the instruction's ordering ("ignore, miss, hit"). CONVERSION_NOTES Step 9 checks the resulting hit rate against the paper: "Hit rate | 84% | 72.1% (all trials) | Consistent (84% is control-only)".

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the trials-table `early_lick` column, whose values are the strings `'no early'` and `'early'`.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
```

iii. CONVERSION_NOTES Step 5: "early_lick | output[2] | no early=0, early=1 | Per trial, broadcast". Step 5 Key Decision 1 explains why early-lick trials are retained rather than excluded as in the reference `get_regular_trial_mask`: "we include all trials because early_lick, outcome, and photostim are decoder outputs/inputs".

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Dictionary mapping `{'no early': 0, 'early': 1}` (silent fallback to 0) broadcast across all 80 bins into row 2. Full-dataset distribution: 88.8% no, 11.2% yes.

ii.
```python
EARLY_LICK_MAP = {'no early': 0, 'early': 1}
...
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
output_early_lick.append(el)
...
np.full(N_TIMEBINS, output_early_lick[t_idx], dtype=np.int64),
```

iii. Follows the instruction's "Early lick (no, yes, per-trial)" ordering. The early lick itself occurs during the sample or delay epoch, i.e. inside the −2.5 s window, so a per-trial flag broadcast over the window is still decodable.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, an `(n_frames, 3)` array of `(tongue_x, tongue_y, tongue_likelihood)` at ~294 Hz with matching timestamps. Column 1 (`tongue_y`) is used. Column 2 (`tongue_likelihood`) is loaded as part of the array but **never used** — no visibility masking is applied.

ii.
```python
tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_ts_all = tongue_ts_obj.timestamps[:]
```
```python
local_y = tongue_data[idx_start:idx_end, 1]
```

iii. Trajectory step 22: "Tongue tracking: columns are (tongue_x, tongue_y, tongue_likelihood) at 300Hz (0.0034s spacing = ~294Hz)". Step 34 also records "For the -2.5 to +1.5s window, tongue likelihood > 0.5 is only ~12.4% of the time (tongue mostly not visible during delay)" and "I should probably only use high-likelihood tongue detections for the percentile computation" — but CONVERSION_NOTES Step 5 Key Decision 3 states the opposite and is what was implemented: "Tongue y discretization: Use all tongue y values regardless of likelihood for percentile computation".

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Per trial, camera frames inside the 4 s window are located with `searchsorted`, assigned to the 80 go-cue-locked bins with `np.digitize`, and the raw `tongue_y` values in each bin are averaged (no likelihood mask, no NaN handling of untracked frames). Bins with no camera frame at all remain NaN at this stage. A per-bin Python loop with a boolean mask is used for the averaging.

ii.
```python
def get_tongue_y_for_trial(tongue_ts, tongue_data, go_time,
                           bin_centers=BIN_CENTERS, bin_width=BIN_WIDTH):
    """Extract tongue y-position for a trial, averaged within each time bin."""
    tongue_y = np.full(n_bins, np.nan, dtype=np.float32)
    t_start = go_time + bin_centers[0] - bin_width / 2
    t_end = go_time + bin_centers[-1] + bin_width / 2
    idx_start = np.searchsorted(tongue_ts, t_start, side='left')
    idx_end = np.searchsorted(tongue_ts, t_end, side='right')
    if idx_start >= idx_end:
        return tongue_y
    local_ts = tongue_ts[idx_start:idx_end]
    local_y = tongue_data[idx_start:idx_end, 1]
    abs_bin_edges = go_time + bin_centers[0] - bin_width/2 + np.arange(n_bins + 1) * bin_width
    bin_indices = np.digitize(local_ts, abs_bin_edges) - 1
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            tongue_y[b] = np.mean(local_y[mask])
    return tongue_y
```

iii. CONVERSION_NOTES Step 5 Key Decision 3 ("Use all tongue y values regardless of likelihood"). The AI reports that this is the quantity the percentiles are then taken over, so the discretised value refers to the same per-bin average.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, all non-NaN 50 ms bin means from that session's trial windows are pooled, and the 40th and 60th percentiles are taken as the two class edges: `<p40` → 0, `[p40, p60]` → 1, `>p60` → 2. Only three classes exist; the instruction's fourth class ("3: not visible") is not implemented, and bins that contain no camera frame are silently given class 1 ("mid") by the array initialiser. `output_values[3] = ['low','mid','high']`. Because the percentiles are computed over *all* frames rather than tracked ones, the middle band is extremely narrow (e.g. in `sub-440957_ses-20190211`, p40 = 262.44 and p60 = 262.98 over all frames, versus 257.76 and 270.32 using only frames with likelihood ≥ 0.5), so the three classes largely partition tracker noise from untracked frames rather than genuine tongue protrusion. The verification log confirms an exact 0.40/0.20/0.40 class split in nearly every session.

ii.
```python
def discretize_tongue_y(tongue_y_session, percentile_low=40, percentile_high=60):
    """Discretize tongue y-position per session."""
    all_values = []
    for y in tongue_y_session:
        valid = y[~np.isnan(y)]
        if len(valid) > 0:
            all_values.append(valid)
    if len(all_values) == 0:
        return [np.ones(len(y), dtype=np.int64) for y in tongue_y_session]
    all_values = np.concatenate(all_values)
    p_low = np.percentile(all_values, percentile_low)
    p_high = np.percentile(all_values, percentile_high)
    tongue_y_discrete = []
    for y in tongue_y_session:
        discrete = np.ones(len(y), dtype=np.int64)  # default middle
        valid = ~np.isnan(y)
        if np.any(valid):
            discrete[valid & (y < p_low)] = 0
            discrete[valid & (y >= p_low) & (y <= p_high)] = 1
            discrete[valid & (y > p_high)] = 2
        tongue_y_discrete.append(discrete)
    return tongue_y_discrete
```

iii. CONVERSION_NOTES Step 5: "tongue_y | output[3] | Discretized per session: <40th=0, 40-60th=1, >60th=2 | Time-varying", and Key Decision 3 on ignoring likelihood. Trajectory step 67 treats the resulting split as a confirmation rather than a warning sign: "Tongue y: 40% low, 20% mid, 40% high (expected from 40th/60th percentile split)". No justification is recorded for omitting the "not visible" class or for the NaN→"mid" default.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same absolute clock as the spikes, so the trial's frame range is found by `searchsorted` at `go - 2.5` and `go + 1.5`, and frames are assigned to bins using edges built from `go + BIN_CENTERS[0] - BIN_WIDTH/2` in 50 ms steps — the identical grid used for the firing rates. Bin *k* of the tongue output therefore covers the same interval as bin *k* of the neural data.

ii.
```python
t_start = go_time + bin_centers[0] - bin_width / 2
t_end = go_time + bin_centers[-1] + bin_width / 2
idx_start = np.searchsorted(tongue_ts, t_start, side='left')
idx_end = np.searchsorted(tongue_ts, t_end, side='right')
...
abs_bin_edges = go_time + bin_centers[0] - bin_width/2 + np.arange(n_bins + 1) * bin_width
bin_indices = np.digitize(local_ts, abs_bin_edges) - 1
```

iii. Same rationale as 2-d and 4-c: a single global clock means no interpolation or offset correction. The `--show-processing` figure plots the discretised tongue output on the same time axis as the neural raster with the go cue marked, as a visual alignment check (CONVERSION_NOTES Step 7).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases, handled as follows:
- **Sessions never quality-controlled / with too few good units**: dropped (`n_good < 2`). This removes the one session whose `classification` is NaN for all units.
- **Trials outside the ephys recording**: dropped by the spike-time-range mask (1-e). Residual trials falling in *gaps* inside the recording are not detected, leaving 2,304 all-zero trials (2.8%) in the output, which the AI knowingly accepted.
- **Missing camera frames in a bin** (the video is trial-gated, so leading bins of short trials have none): silently assigned tongue class 1 ("mid") by the `np.ones` initialiser; a trial or session with no tongue data at all also becomes all-1s.
- **Unrecognised categorical strings**: `dict.get(..., 0)` silently maps anything unexpected to the first class for choice, outcome and early lick.
- **Empty photostim streams** (non-ogen sessions): handled by the `if np.any(stim_mask)` branch, giving an all-zero input.

ii.
```python
if n_good < 2:
    print(f'  Skipping: only {n_good} good units'); io.close(); return None
...
if len(valid_trial_indices) < 2:
    print(f'  Skipping: only {len(valid_trial_indices)} trials with neural coverage'); io.close(); return None
```
```python
if len(all_values) == 0:
    return [np.ones(len(y), dtype=np.int64) for y in tongue_y_session]
...
discrete = np.ones(len(y), dtype=np.int64)  # default middle
```
```python
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
```

iii. CONVERSION_NOTES Step 10 Check 5: "Recording coverage: Handled by checking actual spike time range; Sessions with 0 good units: Skipped (1 session)". On the all-zero trials, Step 10 Check 1: "Warnings: Some trials with all-zero neural data in sessions 43, 44 (brief recording gaps) — These are rare and don't significantly affect training"; trajectory step 114 gives the same reasoning for the full count of 2,304. The `get`-with-default and NaN→"mid" behaviours are not documented or justified anywhere.

## 10-a. What are the most time-consuming steps of the code?

i. The AI instrumented `process_session` with three timers and reported: data loading 1.0–1.5 s/session, firing-rate computation 1.6–7.1 s/session, input/output construction 0.2–0.5 s/session. The firing-rate double loop is the dominant cost, and it scales as n_trials × n_neurons. Full conversion took 1,143.9 s (19.1 min) for 174 files, about 4.6× the expert reference's 247 s for the same files; pickling the 10.2 GB result adds further time. The AI estimated ~17–25 min before the full run, above the instructions' 15-minute budget, and proceeded without further optimisation.

ii.
```python
t_load = time.time()
print(f'  Data loaded in {t_load - t0:.1f}s ({n_good} good units, {len(valid_trial_indices)}/{n_trials_total} valid trials, rec=[{rec_start:.0f},{rec_end:.0f}]s)')
...
t_fr = time.time()
print(f'  Firing rates computed in {t_fr - t_load:.1f}s')
...
t_io = time.time()
print(f'  Inputs/outputs built in {t_io - t_fr:.1f}s')
```

iii. CONVERSION_NOTES Step 7 records the per-step table and "Actual full conversion time: 1143.9s (19.1min)". Trajectory step 68: "Average is ~5.8s per session, so total would be ~5.8 * 174 ≈ 17 minutes. Close to the 15 min target." No speed-up measures are listed in the notes' "Code speedups added" section.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
- `compute_firing_rates_session` has a **nested Python loop over trials × neurons**, calling `np.histogram` once per (trial, neuron) — ~400 neurons × ~540 trials ≈ 216,000 histogram calls per session. It can be reduced to one `searchsorted`+`diff` per neuron over a flattened edge array (as the reference does), i.e. ~400 calls.
- `get_tongue_y_for_trial` loops over the 80 bins and builds a full boolean mask per bin (`mask = bin_indices == b`), an O(n_bins × n_frames) scan where `np.bincount` would be O(n_frames).
- The per-trial loop that builds inputs/outputs recomputes the photostim overlap test and re-enters `get_tongue_y_for_trial` per trial; the per-trial choice/outcome/early-lick lookups are Python-level `dict.get` calls that could be a single vectorised map per session.
None of these were identified or fixed; CONVERSION_NOTES describes the implementation as already optimised.

ii.
```python
for trial_idx in range(n_trials):
    ...
    for i, spikes in enumerate(spike_times_list):
        if len(spikes) > 0:
            left = np.searchsorted(spikes, bin_edges[0])
            right = np.searchsorted(spikes, bin_edges[-1])
            if left < right:
                counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
                fr[i, :] = counts / bin_width
```
```python
for b in range(n_bins):
    mask = bin_indices == b
    if np.any(mask):
        tongue_y[b] = np.mean(local_y[mask])
```

iii. CONVERSION_NOTES Step 6 claims: "`compute_firing_rates_session`: Vectorized histogram computation with searchsorted optimization" — in fact `searchsorted` is only used to narrow the spike array before a scalar `np.histogram` call inside a doubly-nested loop, so the claim overstates what the code does. Step 7's "Speed-ups Implemented" table is left empty.

## 10-c. What processing does the code repeat multiple times?

i. Three repetitions:
- **`nwb.units['spike_times'][:]` is read twice per session** — once inside `get_recording_range` and again in `process_session`. Each read materialises the ragged spike buffer for *all* units (not just good ones, e.g. ~1,950 units where only ~460 are good), so the largest I/O in the pipeline is done twice.
- **Bin edges are rebuilt on every trial** (`bin_edges = go_time + window_start + np.arange(n_bins+1)*bin_width` inside the trial loop, and again inside `get_tongue_y_for_trial`), where a constant relative grid could be added to the go time.
- **Spike arrays are re-sorted** (`np.sort(st)`) although NWB `spike_times` are already stored in ascending order.

ii.
```python
def get_recording_range(nwb, good_indices):
    spike_times_all = nwb.units['spike_times'][:]
    ...
```
```python
    # Get spike times for good units (sorted)
    spike_times_all = nwb.units['spike_times'][:]
    spike_times_good = []
    for i in good_indices:
        st = np.array(spike_times_all[i], dtype=np.float64)
        spike_times_good.append(np.sort(st))
```

iii. Not identified in CONVERSION_NOTES; the notes' "Code inefficiencies identified" section lists nothing on this point. The duplicated read is a side-effect of factoring the recording-range computation into a helper that takes `nwb` rather than the already-loaded spike arrays.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items:
- For the 23 rejected sessions, the expensive `get_recording_range` (full spike-times read and per-unit min/max scan) runs *before* the behavioural session criteria are evaluated, so that work is thrown away.
- `auto_water` and `nwb.subject.description` are read every session; `auto_water` is never used at all and `subject_desc` is carried through `process_session`'s return dict but never written into the output.
- `np.sort` on already-sorted spike times, and the redundant second spike-times read (10-c).
- Firing rates are computed for trials that are later found to be all-zero (2,304 trials) and for trials in sessions that pass filtering but whose data is unusable.
- `simplify_brain_region` runs a 90-entry substring scan per unit yet leaves 138 distinct region labels in the output (many raw CCF strings such as "Dorsal auditory area, layer 6b"), so the mapping effort is largely ineffective rather than discarded.

ii.
```python
auto_water = nwb.trials['auto_water'][:]
...
subject_desc = nwb.subject.description
```
```python
# Get recording range from obs_intervals
rec_start, rec_end = get_recording_range(nwb, good_indices)
...
if correct_rate < 0.65:
    print(f'  Skipping: correct rate {correct_rate:.3f} < 0.65')
    io.close(); return None
```
```python
for key, abbrev in region_map.items():
    if key.lower() in name.lower():
        return abbrev
return name
```

iii. Not discussed in CONVERSION_NOTES. Trajectory step 117 raises "The brain regions (138) – many are unsimplified CCF annotations" as a possible issue but the AI concludes "I think everything looks good. Let me mark as complete."
