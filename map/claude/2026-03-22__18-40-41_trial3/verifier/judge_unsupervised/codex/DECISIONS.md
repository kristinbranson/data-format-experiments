# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script walks the `data/` directory, finds `sub-*` folders, and treats every `.nwb` file inside them as one candidate session. Each session is opened with `pynwb.NWBHDF5IO`, then the trials table, `BehavioralEvents`, `BehavioralTimeSeries`, and `units` tables are read inside `process_session()`.

ii. 
```python
def list_nwb_files(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('sub-')])
    all_files = []
    for sub in subjects:
        sub_dir = os.path.join(data_dir, sub)
        nwb_files = sorted([os.path.join(sub_dir, f)
                           for f in os.listdir(sub_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append((sub, nwb_file))
    return all_files

io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
trials = nwb.trials
be = nwb.acquisition['BehavioralEvents']
bts = nwb.acquisition['BehavioralTimeSeries']
units = nwb.units
```

iii. In `CONVERSION_NOTES.md`, the agent says the NWB archive has 174 session files across 28 subjects and that the reference code used `.mat` exports, so it decided to map NWB fields to the same variables. The trajectory summary for Step 21 also says the source `.mat` pipeline had to be translated to NWB equivalents.

## 1-b. How are the data split into subjects?

i. Subjects are split first by directory name (`sub-*`), then each kept session stores `subject_id` from `nwb.subject.subject_id`. The final dataset builds a unique `subjects` list and a per-session `subject_idx`.

ii. 
```python
subjects = sorted([d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('sub-')])

subject_id = nwb.subject.subject_id if nwb.subject else os.path.basename(nwb_path).split('_')[0]

for sess in session_results:
    if sess['subject_id'] not in subjects:
        subjects.append(sess['subject_id'])
    subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The notes say the NWB files are “organized by subject directories” and list 28 subjects. The agent therefore used directory structure plus NWB subject metadata rather than inferring subject IDs from filenames alone.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one behavioral session. `process_session()` returns one session record, and only sessions that pass the behavioral and neuron filters are appended to `session_results`.

ii. 
```python
for i, (subject, nwb_path) in enumerate(all_files):
    result = process_session(nwb_path,
                            show_processing=args.show_processing,
                            session_idx=i)
    if result is None:
        n_skipped += 1
        continue
    session_results.append(result)
```

iii. `CONVERSION_NOTES.md` says “Each NWB file = one behavioral session” and notes 174 NWB files versus 173 sessions reported in the papers. The agent used the NWB file boundary as the session boundary.

## 1-d. How are the data split into trials?

i. Trials are taken from `nwb.trials`, with one row treated as one trial. Go-cue timestamps from `BehavioralEvents/go_start_times` are asserted to have the same length as the trials table, and per-trial arrays are built by iterating over `valid_indices`.

ii. 
```python
trials = nwb.trials
n_trials = len(trials)
trial_starts = trials['start_time'][:]
trial_stops = trials['stop_time'][:]

go_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_times) == n_trials

valid_indices = np.where(valid_mask)[0]
for trial_idx in valid_indices:
    go_time = go_times[trial_idx]
    ...
```

iii. The notes describe the NWB trial table and event streams, and the agent’s trajectory inspection of one NWB file confirmed that `trials` has 368 rows while `go_start_times` also has 368 timestamps. That was the basis for the one-row-per-trial split.

## 1-e. How are trials filtered based on quality controls?

i. There are two filters. For session inclusion, the script computes a “regular trial” mask excluding auto-water, free-water, photostim, early-lick, and ignore trials, then applies session criteria `correct_rate >= 0.65` and at least 50 hit-left and 50 hit-right trials. For the converted dataset, it keeps trials with no auto/free water, a non-missing tone onset, and trial end not later than a representative unit’s `obs_intervals` coverage.

ii. 
```python
behav_valid = (auto_water == 0) & (free_water == 0)
regular_mask = behav_valid.copy()
for i in range(n_trials):
    if photostim_onset[i] != 'N/A':
        regular_mask[i] = False
    if early_licks[i] == 'early':
        regular_mask[i] = False
regular_mask &= (outcomes != 'ignore')

correct_left = np.sum(regular_mask & (outcomes == 'hit') & (instructions == 'left'))
correct_right = np.sum(regular_mask & (outcomes == 'hit') & (instructions == 'right'))

valid_mask = behav_valid.copy()
valid_mask &= ~np.isnan(tone_onset_per_trial)
for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False
```

iii. In Step 5 notes, the agent explicitly justified keeping early-lick, ignore, and stim trials because they are required for decoder outputs and inputs, while still using the paper’s “regular trial” definition for session selection. Later notes say it fixed the correct-rate denominator to exclude ignore trials to better match `get_regular_trial_mask()`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB `units['spike_times']` ragged array after filtering units by `classification == 'good'` and non-empty `anno_name`.

ii. 
```python
units = nwb.units
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])

spike_times_all = units['spike_times']
spike_times_good = [spike_times_all[idx] for idx in good_indices]
```

iii. The notes say the reference `.mat` pipeline started from spike times and good-unit QC files, and the trajectory says the NWB `classification == 'good'` field was treated as the equivalent QC pass flag.

## 2-b. How is the `neural` data processed?

i. The script bins absolute spike times into a fixed `[-2.5, 1.5] s` window around each trial’s go cue using non-overlapping 50 ms bins, then divides counts by bin width to get firing rates in Hz. It does this separately for every kept trial.

ii. 
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)

bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
np.add.at(fr[i], bin_idx, 1)
fr /= bin_width

fr = compute_firing_rates_vectorized(
    spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
)
```

iii. The notes say the reference code uses `sliding_histogram()` with 40 ms bandwidth and 3.4 ms stride, but the agent chose 50 ms bins because the decoder instructions explicitly required them. Step 4 notes call this “Different by design.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `classification == 'good'` and `anno_name` is present. Sessions with zero such units are dropped.

ii. 
```python
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]

if len(good_indices_units) == 0:
    print(f'  SKIP: no good neurons')
    return None
```

iii. The notes say the white paper and reference code use classifier-based QC and that the NWB `classification` field is the equivalent curated label. The trajectory also records the agent’s decision that `classification == 'good'` is the right NWB-side proxy for the QC files.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go cue onset. For each trial, the script keeps spikes whose absolute timestamps fall between `go_time + T_START` and `go_time + T_END`, so every neural matrix is go-cue-aligned.

ii. 
```python
go_times = be.time_series['go_start_times'].timestamps[:]

mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
spk_window = spk[mask]
bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
```

iii. The instructions required alignment to “Go cue onset,” and the notes say the NWB spike times are absolute so they must be converted to go-relative values trial by trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 50 ms bins and 80 bins per trial across a 4 s window. There is no extra temporal rebinning step; spikes are directly counted into those final bins.

ii. 
```python
BIN_WIDTH = 0.05  # 50 ms bins
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
```

iii. Step 5 notes say “Use 50ms bins as specified by decoder task,” explicitly choosing that over the 40 ms reference preprocessing.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times` and `BehavioralEvents/go_start_times`, using `trial_starts` to decide which sample events belong to each trial.

ii. 
```python
sample_starts = be.time_series['sample_start_times'].timestamps[:]
go_times = be.time_series['go_start_times'].timestamps[:]
trial_starts = trials['start_time'][:]

in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
tone_onset_per_trial[i] = in_trial[-1]
```

iii. The notes map `sample_start_times` to the target input “time_from_tone,” and trajectory Step 56 says the agent reasoned that tone timing has to vary trial by trial relative to go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the script finds sample-start events between trial start and go cue, chooses the last one, converts it to go-relative time, and subtracts that from every 50 ms bin center. Negative values mean the current bin is before that tone onset.

ii. 
```python
tone_onset_per_trial = np.full(n_trials, np.nan)
for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
    if len(in_trial) > 0:
        tone_onset_per_trial[i] = in_trial[-1]

bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
tone_time = tone_onset_per_trial[trial_idx]
tone_relative = tone_time - go_time
time_from_tone = bin_centers - tone_relative
```

iii. In trajectory Step 56, the agent explicitly says it used “the last sample onset before the go cue” because some trials have replays after early licking and it wanted the event that “actually drove the trial forward.”

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled at the same `bin_centers` used for neural firing rates, so each element of `time_from_tone` corresponds to the same go-cue-centered 50 ms bin as the neural matrix.

ii. 
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
fr = compute_firing_rates_vectorized(...)
time_from_tone = bin_centers - tone_relative
input_data = np.stack([time_from_tone.astype(np.float32), photostim], axis=0)
```

iii. The notes describe the target input as a time-varying variable aligned to the go-cue-centered trial window, so the agent reused the neural bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from `trials['photostim_onset']`, `trials['photostim_duration']`, `trials['start_time']`, and go-cue times from `BehavioralEvents/go_start_times`.

ii. 
```python
photostim_onset = trials['photostim_onset'][:]
photostim_duration = trials['photostim_duration'][:]
trial_starts = trials['start_time'][:]
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The notes map “photostim_onset/duration” to the photostim input and say it should be represented as a binary time series. The trajectory shows the agent checked NWB event fields before deciding the trial-table onset was relative to trial start.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For stim trials, the script parses onset and duration strings as floats, converts onset from trial-start-relative time to absolute time, then to go-relative time, and sets bins to 1 when their centers fall inside that interval.

ii. 
```python
photostim = np.zeros(N_BINS, dtype=np.float32)
if photostim_onset[trial_idx] != 'N/A':
    ps_onset = float(photostim_onset[trial_idx])
    ps_duration = float(photostim_duration[trial_idx])
    ps_onset_abs = trial_starts[trial_idx] + ps_onset
    ps_end_abs = ps_onset_abs + ps_duration
    ps_onset_rel = ps_onset_abs - go_time
    ps_end_rel = ps_end_abs - go_time

    for b in range(N_BINS):
        bc = bin_centers[b]
        if ps_onset_rel <= bc < ps_end_rel:
            photostim[b] = 1.0
```

iii. The notes say photostimulation should be “Binary time series; 1 during photostim window, 0 otherwise.” The trajectory shows the agent explored whether the onset field was absolute or relative and settled on trial-start-relative after inspecting example values.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is expressed on the same 80-bin go-cue-centered timeline as the neural firing rates. After converting stim onset/end into go-relative coordinates, the script marks the same `bin_centers` used for neural data.

ii. 
```python
ps_onset_rel = ps_onset_abs - go_time
ps_end_rel = ps_end_abs - go_time
for b in range(N_BINS):
    bc = bin_centers[b]
    if ps_onset_rel <= bc < ps_end_rel:
        photostim[b] = 1.0
```

iii. Step 5 notes say the photostim input is aligned as a binary time series over the trial window, and the code implements exactly that on the neural bin grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The script derives `choice` from `trial_instruction` plus `outcome`; it does not use lick-event streams or any explicit per-trial lick-direction variable.

ii. 
```python
instr = instructions[trial_idx]
outcome = outcomes[trial_idx]
if outcome == 'hit':
    choice = 0 if instr == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

iii. The notes say “trial_instruction + outcome” map to choice, and Step 56 trajectory shows the agent debated how to handle ignore trials because there is no lick on those trials and the decoder expects only left/right labels.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The script encodes hit trials as the instructed side, miss trials as the opposite side, and ignore trials as the instructed side. It then repeats that categorical value across all 80 bins for the trial.

ii. 
```python
if outcome == 'hit':
    choice = 0 if instr == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1

np.full(N_BINS, choice, dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly justifies the ignore-trial rule as “Set to instruction direction ... since there’s no actual lick.” The trajectory shows the same reasoning.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is taken directly from `trials['outcome']`.

ii. 
```python
outcomes = trials['outcome'][:]
outcome = outcomes[trial_idx]
```

iii. The notes list `outcome` as a direct NWB trial-table field with values `hit`, `miss`, and `ignore`, so no indirect derivation was needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script maps strings to the decoder’s categorical encoding `ignore=0`, `miss=1`, `hit=2`, then repeats the trial-level label across all time bins.

ii. 
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. This mapping comes directly from the decoder task specification and is also recorded verbatim in Step 5 notes.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is derived directly from `trials['early_lick']`.

ii. 
```python
early_licks = trials['early_lick'][:]
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
```

iii. The notes describe `early_lick` as a direct trial-table variable and keep these trials because early lick is itself a requested decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The script binarizes the string labels: `no early -> 0`, anything else (`early`) -> 1. It then repeats that label over all 80 bins.

ii. 
```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
np.full(N_BINS, early_val, dtype=np.int64)
```

iii. The Step 5 mapping table in `CONVERSION_NOTES.md` says `early_lick` is a direct categorical output with `no=0, yes=1`.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from the side-camera tongue-tracking time series, specifically the y coordinate and confidence stored in `Camera0_side_TongueTracking`.

ii. 
```python
has_tongue = 'Camera0_side_TongueTracking' in bts.time_series
if has_tongue:
    tongue_data = bts.time_series['Camera0_side_TongueTracking'].data[:]  # x, y, likelihood
    tongue_ts = bts.time_series['Camera0_side_TongueTracking'].timestamps[:]
    tongue_y = tongue_data[:, 1]
    tongue_likelihood = tongue_data[:, 2]
```

iii. The notes identify side-camera tongue tracking as the intended source for this decoder output, and the trajectory inspection of NWB files confirmed the data shape `(n_frames, 3)`.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The script computes session-level 40th and 60th percentiles from frames with likelihood > 0.5 when enough such frames exist. For each trial bin, it finds the closest tongue-tracking frame in absolute time and discretizes that y value into low/mid/high.

ii. 
```python
tongue_visible = tongue_likelihood > 0.5
if np.sum(tongue_visible) > 100:
    visible_y = tongue_y[tongue_visible]
    p40 = np.percentile(visible_y, 40)
    p60 = np.percentile(visible_y, 60)

for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
    t_idx = min(t_idx, len(tongue_ts) - 1)
    ty = tongue_y[t_idx]
```

iii. Step 5 notes say the agent chose “the side camera tongue tracking y-coordinate” and a per-session percentile discretization. It also notes that the underlying tracking is sampled at about 294 Hz, so the code uses nearest-frame lookup to place it on the 50 ms neural bins.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. It uses per-session thresholds: category 0 below the 40th percentile, category 1 between the 40th and 60th percentiles, and category 2 above the 60th percentile.

ii. 
```python
if ty < p40:
    tongue_y_trial[b] = 0
elif ty < p60:
    tongue_y_trial[b] = 1
else:
    tongue_y_trial[b] = 2
```

iii. This rule is directly stated in the task instructions and repeated in the notes as the chosen discretization plan.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The script adds each neural bin’s go-relative center to the trial’s absolute go time, finds the nearest tongue-tracking frame at that absolute time, and writes the resulting category into the same bin index as the neural data.

ii. 
```python
bc_abs = go_time + bin_centers[b]
t_idx = np.searchsorted(tongue_ts, bc_abs)
t_idx = min(t_idx, len(tongue_ts) - 1)
ty = tongue_y[t_idx]
```

iii. The agent’s notes say tongue output should be time-varying and aligned to the same trial window as neural data. The chosen implementation is nearest-neighbor alignment from the 3.4 ms camera timeline onto the 50 ms neural timeline.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing tone onsets cause a trial to be dropped. Sessions with no good neurons or fewer than two valid trials are dropped. Trials beyond a representative good unit’s recording coverage are dropped. Unmapped `anno_name` values fall back to `OtherCortex`. If tongue tracking exists but has too little high-confidence data, percentiles are computed from all tongue frames; if tongue tracking is absent entirely, the output defaults to all ones (middle class).

ii. 
```python
tone_onset_per_trial = np.full(n_trials, np.nan)
...
valid_mask &= ~np.isnan(tone_onset_per_trial)

if len(good_indices_units) == 0:
    return None
if len(valid_indices) < 2:
    return None

if trial_end_abs > max_recording_time + 1.0:
    valid_mask[i] = False

print(f'  WARNING: Unmapped annotation: "{anno_name}"')
return 'OtherCortex'

if np.sum(tongue_visible) > 100:
    ...
else:
    p40 = np.percentile(tongue_y, 40)
    p60 = np.percentile(tongue_y, 60)
...
else:
    tongue_y_trial = np.ones(N_BINS, dtype=np.float32)
```

iii. The notes explicitly mention fallback-to-`OtherCortex`, negligible zero-neural-data edge cases, and the need to discard trials outside recording coverage. They treat these as pragmatic repairs rather than reference-derived processing steps.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive work is inside the per-session trial loop: recomputing firing rates from raw spike times for every trial and every neuron, loading full tongue-tracking arrays, and then scanning every trial/bin again to align tongue and photostim signals.

ii. 
```python
for trial_idx in valid_indices:
    fr = compute_firing_rates_vectorized(...)
    ...
    for b in range(N_BINS):
        if ps_onset_rel <= bc < ps_end_rel:
            photostim[b] = 1.0
    ...
    for b in range(N_BINS):
        t_idx = np.searchsorted(tongue_ts, bc_abs)
        ...
```

iii. The notes estimate about 5-10 seconds per session and about 30 minutes full runtime, which is consistent with repeated trial-by-trial spike binning and per-bin alignment work dominating cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are still scalar: the neuron loop in `compute_firing_rates_vectorized()`, the tone-onset search over trials, the recording-coverage loop, the per-bin photostim loop, and the per-bin tongue alignment loop.

ii. 
```python
for i, spk in enumerate(spike_times_list):
    ...

for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
    ...

for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    ...

for b in range(N_BINS):
    ...

for b in range(N_BINS):
    ...
```

iii. The agent never claims these are optimal. The notes only give runtime estimates, so the remaining Python loops reflect implementation convenience rather than a stated efficiency rationale.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly scans all good units’ spike times once per trial, repeatedly searches the tongue timestamp vector once per bin per trial, and repeatedly rebuilds per-trial constant output arrays for choice, outcome, and early lick.

ii. 
```python
for trial_idx in valid_indices:
    fr = compute_firing_rates_vectorized(spike_times_good, go_time, ...)
    ...
    t_idx = np.searchsorted(tongue_ts, bc_abs)
    ...
    np.full(N_BINS, choice, dtype=np.int64)
    np.full(N_BINS, outcome_val, dtype=np.int64)
    np.full(N_BINS, early_val, dtype=np.int64)
```

iii. The notes focus on correctness and validation, not reuse or caching, so these repeated computations are an artifact of the straightforward implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script carries some work that is not needed for the final decoder payload: it computes `trial_stops` but never uses them, creates optional processing plots, stores trial-level categorical outputs as full 80-bin arrays, and builds detailed progress/statistics purely for console output. It also computes `bin_edges_start` and `bin_edges_end` inside the firing-rate helper without using them directly.

ii. 
```python
trial_stops = trials['stop_time'][:]

bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width
bin_edges_end = bin_edges_start + bin_width

if args.show_processing and len(session_results) <= 2:
    make_processing_plots(...)

output_data = np.array([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_val, dtype=np.int64),
    tongue_y_trial.astype(np.int64),
], dtype=np.int64)
```

iii. The notes emphasize validation plots and summary statistics as sanity checks. Those help debugging, but they are not required by the final dataset consumer. The repeated time-axis expansion of trial-level outputs is likewise redundant from an information-content perspective.
