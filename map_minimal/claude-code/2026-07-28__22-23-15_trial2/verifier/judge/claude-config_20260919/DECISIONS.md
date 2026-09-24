# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is the DANDI 000363 release: one NWB file per session under `data/sub-<subject_id>/`. The AI finds every session with a single sorted glob over that layout and opens each file once with `pynwb.NWBHDF5IO`. Inside a file it reads the trials table (`nwb.trials.to_dataframe()`), the unit table (`nwb.units`, including `classification`, `anno_name`, `spike_times`, `obs_intervals`), the event streams (`nwb.acquisition['BehavioralEvents']`), the video tracking (`nwb.acquisition['BehavioralTimeSeries']`) and the subject record (`nwb.subject`). 174 files are found; each is processed independently inside a `try/except` and the per-session results are concatenated afterwards in `convert_all`.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
print(f"Found {len(nwb_files)} NWB files")
...
for i, nwb_file in enumerate(nwb_files):
    print(f"[{i+1}/{len(nwb_files)}] {os.path.basename(nwb_file)}")
    try:
        result = process_session(nwb_file, verbose=verbose)
    except Exception as e:
        print(f"  ERROR: {e}")
        ...
        continue
```
```python
io = NWBHDF5IO(nwb_file, 'r')
nwb = io.read()
...
trials = nwb.trials.to_dataframe()
be = nwb.acquisition['BehavioralEvents']
go_start_times = be.time_series['go_start_times'].timestamps[:]
units = nwb.units
```

iii. From the trajectory (step 18): *"The data is in NWB format (from DANDI), not the .mat format the reference code uses."* The AI recognised that the reference repository's `.mat`/DataJoint loading path did not exist for the distributed data, and that `pynwb` over the one-file-per-session DANDI layout is the equivalent entry point. It cross-checked the file count against the papers (step 27: *"174 sessions, 28 subjects. The paper says 173 behavioral sessions"*).

## 1-b. How are the data split into subjects?

i. Each NWB file carries its animal in two places: `nwb.subject.subject_id` (a numeric id such as `'440956'`) and `nwb.subject.description` (the mouse name used in the papers, e.g. `'SC015'`). The AI reads both but keys the subject list on `subject_desc`, accumulating names into an `OrderedDict` in first-encounter order; `subject_idx` is the position of each session's mouse name in that list. The result is 28 subjects.

ii.
```python
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description  # e.g. 'SC015'
```
```python
if result is not None:
    all_sessions.append(result)
    sub_desc = result['subject_desc']
    if sub_desc not in all_subjects:
        all_subjects[sub_desc] = result['subject_id']
```
```python
subjects = list(all_subjects.keys())
...
subject_idx.append(subjects.index(sess['subject_desc']))
```

iii. The AI did not narrate this choice explicitly, but the logging line it wrote (`sub={subject_desc}`) shows it wanted the human-readable mouse name that the data paper and method paper use, so that the converted dataset can be cross-referenced against the papers. The numeric `subject_id` is retained as the dictionary value in case the DANDI id is needed.

## 1-c. How are the data split into sessions?

i. One NWB file is one session, so no splitting is needed; session order follows the sorted file list. On top of that, the AI applies a **session-level quality control** taken from the method paper: a session is kept only if behavioural performance on control trials exceeds 65% and there are at least 50 correct lick-left *and* 50 correct lick-right control trials. "Control" is defined as no photostimulation, no `auto_water`, no `free_water`, no early lick and a response given. Crucially, these statistics are computed over **all** trials of the session, not just the electrophysiologically covered subset. Sessions with no control trials, no good units, or fewer than 2 usable trials are also dropped. 144 of 174 sessions survive (21 for performance < 65%, 6 for too few correct trials per side, the rest for no good/annotated units or too few trials).

ii.
```python
MIN_CORRECT_PER_SIDE = 50  # minimum correct trials per side for session inclusion
MIN_PERFORMANCE = 0.65  # minimum performance for session inclusion
```
```python
all_valid = (auto_water == 0) & (free_water == 0)
control_mask_all = (all_valid &
                    (early_lick == 'no early') &
                    (outcome != 'ignore') &
                    no_photostim)

n_control = control_mask_all.sum()
...
n_correct = ((outcome == 'hit') & control_mask_all).sum()
performance = n_correct / n_control

correct_left = ((outcome == 'hit') & (trial_instruction == 'left') & control_mask_all).sum()
correct_right = ((outcome == 'hit') & (trial_instruction == 'right') & control_mask_all).sum()

if performance < MIN_PERFORMANCE:
    ...
    return None

if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    ...
    return None
```

iii. The AI took the criterion verbatim from the methods text (step 97: *"the paper says 'We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each.' So these criteria ARE applied"*). It then corrected the scope of the statistic after seeing a session rejected on a recorded subset that passed on the whole session (step 76: *"The session filtering should be based on ALL trials in the session, not just the recorded trials... this refers to the overall session, not just the recorded portion"*). It noted the cost: 144 sessions / 57,935 units against the white paper's 173 sessions / 69,943 units, attributing the gap to exactly this filter.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the NWB trials table, one row per behavioural trial. The AI asserts that the number of `go_start_times` events equals the number of trial rows, so the go cue that anchors every trial is unambiguous. All per-trial variables (`trial_instruction`, `early_lick`, `outcome`, `photostim_power`, `auto_water`, `free_water`, `start_time`) are pulled as column arrays and indexed by trial number.

ii.
```python
trials = nwb.trials.to_dataframe()
n_trials_total = len(trials)

be = nwb.acquisition['BehavioralEvents']
go_start_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_start_times) == n_trials_total
```
```python
trial_instruction = trials['trial_instruction'].values
early_lick = trials['early_lick'].values
outcome = trials['outcome'].values
photostim_power = trials['photostim_power'].values
auto_water = trials['auto_water'].values
free_water = trials['free_water'].values
```

iii. The AI established early (step 30) that the other epoch streams cannot be used as trial delimiters — *"There are 405 sample_starts but 368 trials (extra are from early lick replays)"* — whereas `go_start_times` has exactly one entry per trial. The assertion documents and enforces that one-to-one mapping.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both aimed at removing trials that are not usable rather than trials the animal performed badly:
 - **Neural coverage.** Probes are not always in place for the whole behavioural session, so `units/obs_intervals` is used to find which trials have spike data. The AI takes the obs-interval list of the first good unit, matches its *first* interval start to the nearest `trials.start_time`, and then assumes the covered trials are the contiguous block of `n_obs_trials` rows starting there (`n_obs_trials` is the minimum interval count over good units).
 - **Non-behavioural water trials.** Trials with `auto_water != 0` or `free_water != 0` are dropped.
Early-lick trials and no-response (`ignore`) trials are deliberately **kept**, contrary to the data paper, because they are required decoder outputs. A session with fewer than 2 surviving trials is dropped. 74,769 trials survive across the 144 retained sessions.

ii.
```python
obs_0 = units.get_unit_obs_intervals(good_indices[0])
n_obs_trials = len(obs_0)

for ui in good_indices[1:]:
    oi = units.get_unit_obs_intervals(ui)
    if len(oi) != n_obs_trials:
        n_obs_trials = min(n_obs_trials, len(oi))

# Find which trial index the first obs_interval corresponds to
trial_starts = trials['start_time'].values
first_obs_start = obs_0[0, 0]
first_recorded_trial = np.argmin(np.abs(trial_starts - first_obs_start))

recorded_trial_indices = np.arange(first_recorded_trial, first_recorded_trial + n_obs_trials)
recorded_trial_indices = recorded_trial_indices[recorded_trial_indices < n_trials_total]

# Only use recorded trials that pass trial-level filters (no auto_water, no free_water)
valid_trial_mask = np.zeros(n_trials_total, dtype=bool)
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True

valid_indices = np.where(valid_trial_mask)[0]
n_trials = len(valid_indices)

if n_trials < 2:
    ...
    return None
```

iii. The coverage filter was discovered empirically by debugging all-zero trials (steps 53–59): *"the spike times for unit 3 only go up to 1107.09, but trial 159 has go_cue at 1112.64... the obs_intervals has shape (160, 2), meaning this unit was only observed during 160 trials"*, then *"mice perform many trials, but Neuropixels recordings only cover a portion of the session"*. The initial implementation assumed coverage began at trial 0; the AI found a counter-example and fixed it (step 111: *"the obs_intervals don't correspond to trials 0..504, they correspond to trials 125..629. My code assumed they correspond to the first N trials, which is wrong"*). `auto_water`/`free_water` were excluded as *"training/hint trials where outcomes are artificially correct"* (step 44). Keeping early-lick and ignore trials was an explicit, reasoned deviation from the paper (step 39): *"If I exclude early lick trials, that variable becomes uninformative. If I exclude no-response trials, outcome loses one of its values. The instructions allow deviations when needed for the decoder specifications."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (session-absolute seconds), restricted to units that pass the QC filter of 2-c, together with `BehavioralEvents/go_start_times`, which supplies the alignment point for each trial. Spike times for all good units are read once per session into a Python list before the per-trial loop.

ii.
```python
# --- Pre-load all spike times for good units (faster than per-unit reads) ---
all_spike_times = []
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)
```

iii. Spike times are the only neural representation in the NWB file; the AI confirmed at step 24 that *"the spike times are in absolute session time (not aligned to go cue)"* and therefore that `go_start_times` is required alongside them.

## 2-b. How is the `neural` data processed?

i. For each trial and each good unit the spike train is windowed to `[go - 2.5, go + 1.5]` with two `searchsorted` calls, re-expressed relative to the go cue, histogrammed into the 80 fixed 50 ms bins, and divided by the bin width to give a firing rate in spikes/s. The result is one `(n_neurons, 80)` `float32` array per trial. No smoothing, normalisation, or baseline subtraction is applied.

ii.
```python
neural_data = []
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    trial_fr = np.zeros((n_good, N_BINS), dtype=np.float32)
    abs_start = go_time + BEGIN_TIME
    abs_end = go_time + END_TIME

    for i, st in enumerate(all_spike_times):
        # Fast filter to window
        lo = np.searchsorted(st, abs_start)
        hi = np.searchsorted(st, abs_end)
        if hi > lo:
            rel_spikes = st[lo:hi] - go_time
            counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
            trial_fr[i, :] = counts / BIN_WIDTH

    neural_data.append(trial_fr)
```

iii. The header of the script states the intent: *"Neural data: spike counts per bin, converted to firing rates (spikes/s)"*. This mirrors the reference repository's `sliding_histogram(..., rate=True)`, which the AI had read (the context summary at step 126 records the reference parameters `bw=0.04, stride=0.0034, begin_time=-3, end_time=3`); the AI overrode the bin width and window with the values mandated by the decoder instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A unit is kept only if `units/classification == 'good'` — the verdict of the spike-sorting QC classifier — **and** its CCF annotation `anno_name` maps to one of 14 broad brain regions. Units whose annotation is empty or unmapped are dropped (in practice this removes essentially nothing once the mapping was completed). No individual QC metric thresholds are applied and `unit_quality` is not used. A session with zero surviving units is dropped. 57,935 units are kept across the 144 retained sessions (69,453 `classification == 'good'` units exist across all 174 files; the difference is entirely the session filter of 1-c).

ii.
```python
classification = units['classification'].data[:]
anno_names = units['anno_name'].data[:]

# Filter to good units with valid annotations
good_indices = []
unit_regions = []
for ui in range(n_units_total):
    if classification[ui] != 'good':
        continue
    region = map_anno_to_region(anno_names[ui])
    if region is not None:
        good_indices.append(ui)
        unit_regions.append(region)

n_good = len(good_indices)
if n_good == 0:
    ...
    return None
```
```python
def map_anno_to_region(anno_name):
    """Map a CCF annotation name to one of 14 broad brain regions."""
    if not anno_name or anno_name.strip() == '':
        return None
    al = anno_name.strip().lower()
    # ALM: Secondary motor area (= Anterior Lateral Motor cortex)
    if 'secondary motor area' in al:
        return 'ALM'
    ...
```

iii. The AI validated the filter against the white paper's headline number (step 97: *"Total good units across all sessions: 69,453 (close to paper's 69,943)"*). The additional annotation requirement exists because `brain_region_idx` must be populated for every unit; the AI iterated on the keyword mapping after the first full run flagged 835 thalamic units whose CCF names lack the word "thalamus" (step 90: *"These are all thalamic nuclei! They don't have 'thalamus' in their name"*). A subagent confirmed the reference code instead relies on pre-sorted DataJoint QC files with region keys, which are not shipped with the NWB release, so string matching was used as a substitute (step 154). Note that this substitute assigns *all* secondary-motor-area units to ALM, whereas the reference pipeline defines ALM by voxel coordinates.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go cue onset. Spike times and `go_start_times` share one session-absolute clock, so no resampling or offset correction is needed: for each trial the absolute window `[go + BEGIN_TIME, go + END_TIME]` is cut out and the spike times inside it are shifted by `-go_time` before binning on the fixed go-cue-relative edge grid.

ii.
```python
BEGIN_TIME = -2.5  # seconds relative to go cue
END_TIME = 1.5    # seconds relative to go cue
```
```python
go_time = go_start_times[trial_idx]
abs_start = go_time + BEGIN_TIME
abs_end = go_time + END_TIME
...
rel_spikes = st[lo:hi] - go_time
counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```
```python
'temporal_alignment_event': 'Go cue onset',
'off_start': BEGIN_TIME,
'off_end': END_TIME,
```

iii. The instructions specify go cue onset and the −2.5 s/+1.5 s window. The AI confirmed the clock convention before writing the code (step 39: *"since spike times in the NWB data are in absolute session time (unlike Susu's data where they're relative to go cue), I can use go_start_times to convert"*).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, non-overlapping, giving exactly 80 bins per trial spanning −2.5 s to +1.5 s. The edge grid and bin centres are computed once at module level from `BEGIN_TIME`, `END_TIME` and `BIN_WIDTH` and reused for every trial, unit and session, so all trials have identical shape. Spikes are binned once at this resolution — there is no rebinning of an intermediate representation, and the reference repository's own parameters (40 ms width, 3.4 ms stride sliding histogram) were deliberately not used.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5  # seconds relative to go cue
END_TIME = 1.5    # seconds relative to go cue

# Time bin edges and centers
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```
```python
'time_bin_size': BIN_WIDTH * 1000,
```

iii. The decoder instructions mandate 50 ms bins; the script header records *"Bin size: 50ms (non-overlapping) -> 80 time bins"*. The AI knew the reference used `bw=0.04, stride=0.0034` (step 126 summary) and treated the instruction as the allowed override, consistent with *"Discrepancies are only allowed if required by the Decoder Input and Decoder Output specifications"*.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. **None at conversion time.** The AI hard-codes a constant `TONE_ONSET_REL = -1.85` s, justified by the fixed task structure (0.65 s sample epoch + 1.2 s delay). `sample_start_times` was inspected only during exploration, to establish the constant; it is not read by `convert_data.py`. Because the offset is a constant, the input row is computed once at module level and is byte-for-byte identical for every trial in every session.

ii.
```python
TONE_ONSET_REL = -1.85  # tone onset relative to go cue (sample 0.65s + delay 1.2s)
...
# Precompute time_from_tone (same for all trials)
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```
```python
# --- Input 1: Time from tone onset (same for all trials) ---
...
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. The AI initially matched each go cue to the preceding `sample_start_times` event and found scatter (step 42: *"The sample-to-go time varies slightly per session (around 1.85-1.88), and within a session there's some variability (std up to 0.24). This means the tone onset timing is not perfectly fixed across trials!"*). It re-examined the matching and concluded the scatter was its own error (step 44: *"The sample period is 0.65s and delay period is 1.2s, so tone onset is always 1.85s before go cue. This is very consistent. The earlier variability I saw was because I was matching incorrectly."*). `CONVERSION_NOTES.md` records the conclusion as *"Tone onset is consistently 1.85s before Go cue across all sessions."*

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. One subtraction, done once for the whole dataset: each bin centre minus the constant −1.85 s, yielding the fixed vector `[-0.6, -0.55, ..., 3.3]`. The values are stored as `float32` and replicated into every trial's input array as row 0.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```
```python
'tone_onset_relative_to_go': TONE_ONSET_REL,
'sample_period': 0.65,
'delay_period': 1.2,
```

iii. Per the AI's reasoning at step 36: *"For any time bin t relative to the go cue, the time from tone onset would be t + 1.85s. Before the tone actually starts at -1.85s, this value would be negative, which makes sense — I should allow negative values to represent the period before tone onset rather than clamping to zero."* The reported range `[-0.6, 3.3]` in the verification output matches.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is defined on the same grid as the neural data: it is a function of `BIN_CENTERS`, which are the centres of the same `BIN_EDGES` used to histogram the spikes. Bin *k* of the input therefore covers exactly the interval of bin *k* of the firing rates, with no interpolation or resampling.

ii.
```python
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```
```python
counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. Not narrated separately — deriving both streams from the single module-level grid makes the alignment true by construction.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, read as absolute session timestamps, together with each trial's go cue. (`trials['photostim_power']` is also read, but only to define "control" trials for the session-level performance filter of 1-c; it does not enter the input.)

ii.
```python
# --- Get photostim times ---
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```
```python
photostim_power = trials['photostim_power'].values
no_photostim = np.array([str(p) == 'N/A' or str(p) == 'nan' or
                         (isinstance(p, (int, float)) and p == 0)
                         for p in photostim_power])
```

iii. The AI wanted an explicit on/off interval on the same absolute clock as the spikes rather than a per-trial flag (step 39: *"working through the photostimulation data by checking whether stimulation occurs during each trial window using the onset, duration, and power values from the NWB file, converting absolute times to relative times by subtracting the go cue timestamp... Since photostimulation happens during the late delay epoch ending before the go cue, I'll create a binary time series marking when it's active."*). The event streams give that directly; they are numerically identical to `trials['photostim_onset'] + trials['start_time']` with duration `photostim_duration`.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial the code scans **all** photostim events of the session, keeps those that overlap the trial's 4 s window, converts their start/stop to go-cue-relative time, and sets to 1.0 every bin whose centre falls in `[rel_start, rel_stop)`. Trials with no overlapping event keep an all-zero row. The result is a binary `float32` time series, not a per-trial scalar.

ii.
```python
photostim_binary = np.zeros(N_BINS, dtype=np.float32)
trial_start_abs = go_time + BEGIN_TIME
trial_end_abs = go_time + END_TIME

for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    if ps_stop > trial_start_abs and ps_start < trial_end_abs:
        rel_start = ps_start - go_time
        rel_stop = ps_stop - go_time
        # Vectorized bin marking
        mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
        photostim_binary[mask] = 1.0
```

iii. The instructions ask for *"Whether photostimulation is on at every time point (discrete, time-varying)"*, so the AI represented it as a per-bin binary rather than a trial-level label. Scanning all events (rather than only the current trial's) means stimulation delivered in a neighbouring trial that happens to fall inside this trial's window is also marked on, which is what "is photostimulation on at this time point" literally asks for.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Event times are converted to go-cue-relative seconds by subtracting the same `go_time` used to window the spikes, then compared against `BIN_CENTERS` — the centres of the spike-histogram edges. Same grid, same trial anchor, no resampling.

ii.
```python
go_time = go_start_times[trial_idx]
...
rel_start = ps_start - go_time
rel_stop = ps_stop - go_time
mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
```

iii. Not narrated separately; as with the tone input, using the shared `BIN_CENTERS` grid makes alignment automatic.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Two trials-table columns, `trial_instruction` (`'left'`/`'right'`) and `outcome` (`'hit'`/`'miss'`/`'ignore'`). There is no lick-direction column in the file, so choice is inferred: on a hit the animal licked the instructed side, on a miss it licked the other side. On an `ignore` trial the AI assigns the **instructed** side.

ii.
```python
instr = trial_instruction[trial_idx]
outc = outcome[trial_idx]

# Choice: for hit, choice = instruction; for miss, choice = opposite; for ignore, use instruction
if outc == 'hit':
    choice = 0 if instr == 'left' else 1
elif outc == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

iii. The AI identified the ambiguity explicitly and weighed three options (step 39): *"For ignore trials where there's no lick, I'm wrestling with how to assign a choice value — I could exclude them, use the trial instruction as a proxy, or handle them differently"*, settling on *"for ignore trials, it defaults to the instruction... using the instruction as a proxy for what the neural activity likely encoded during the delay period."* It did not consider adding a third "no lick" category, and the `left_lick_times`/`right_lick_times` event streams present in the file were not consulted to recover the actual lick side.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived choice is coded `0 = left`, `1 = right` (two categories only), broadcast across all 80 bins with `np.full`, and written into row 0 of the per-trial `(4, 80)` `int64` output array. `output_values[0]` is `['left', 'right']`. The resulting dataset-wide distribution is 49.9% left / 50.1% right, i.e. the ~10.8% of `ignore` trials are folded into whichever side was instructed.

ii.
```python
output_values = [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
]
```
```python
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_lick_val, dtype=np.int64),
    tongue_y_disc.astype(np.int64),
], axis=0)
```

iii. `0 = left`, `1 = right` follows the ordering in the instructions. Per-trial values are broadcast across bins so that all four outputs share one `(n_output, n_timepoints)` array, satisfying the format requirement that outputs be time-varying where possible. The integer dtype was forced by a decoder crash (step 69: *"the decoder is trying to use float32 values as list indices, which Python doesn't allow"*).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The `outcome` column of the trials table, which already stores exactly the three strings the instructions ask for: `'ignore'`, `'miss'`, `'hit'`. Nothing is derived.

ii.
```python
outcome = trials['outcome'].values
...
outc = outcome[trial_idx]
```

iii. No derivation is needed because the categories in the file match the requested ones one-for-one. (`CONVERSION_NOTES.md` describes outcome as *"Derived from `outcome` and `early_lick` columns"*, but the code uses `outcome` alone — the note is inaccurate.)

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps the three strings to `0 = ignore`, `1 = miss`, `2 = hit`; the scalar is broadcast across all 80 bins into row 1 of the output array. Observed distribution: 10.8% ignore, 15.3% miss, 73.9% hit.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
```
```python
output_names = ['choice', 'outcome', 'early_lick', 'tongue_y_position']
output_values = [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ...
]
```

iii. The code assignment follows the order given in the decoder instructions (`ignore, miss, hit`). The `[...]` dictionary lookup will raise a `KeyError` on any unexpected string, which acts as a guard that the three categories are exhaustive.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The `early_lick` column of the trials table, which holds `'no early'` / `'early'`.

ii.
```python
early_lick = trials['early_lick'].values
...
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The flag is stored explicitly, so no derivation is needed. The AI confirmed that it is independent of `outcome` (step 39: *"early_lick and outcome are independent fields — a trial can have early_lick='early' and still result in a hit, miss, or ignore outcome"*), which is why it is kept as its own output rather than folded into outcome.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A string comparison mapping `'early' -> 1` and everything else to `0`, broadcast across all 80 bins into row 2. Observed distribution: 88.5% no, 11.5% yes. Note this is a truthy test rather than a dictionary lookup, so an unexpected string would silently become `0`.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
...
np.full(N_BINS, early_lick_val, dtype=np.int64),
```

iii. `0 = no`, `1 = yes` follows the instructions. Keeping early-lick trials in the dataset at all was the deliberate deviation described in 1-e: *"If I exclude early lick trials, that variable becomes uninformative."* The lick that sets the flag happens during the sample or delay epoch, so the event itself falls inside the −2.5 s window even though the label is stored per trial.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Column 1 (`tongue_y`) of `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking.data`, an `(n_frames, 3)` DeepLabCut array of `(tongue_x, tongue_y, tongue_likelihood)` sampled at ~294 Hz, together with its `timestamps`. **Column 2 (`tongue_likelihood`) is loaded implicitly as part of the array but never used**: no visibility test is applied anywhere in `convert_data.py`.

ii.
```python
# --- Get tongue tracking data ---
bt = nwb.acquisition['BehavioralTimeSeries']
tongue_ts = bt.time_series['Camera0_side_TongueTracking']
tongue_y_all_data = tongue_ts.data[:, 1]  # y column only
tongue_timestamps = tongue_ts.timestamps[:]
```

iii. The AI knew the channel layout and the meaning of the third column (step 24: *"The tongue tracking has (x, y, confidence) columns"*) and explicitly planned to use it (step 36: *"The tricky part is handling the likelihood column — low confidence values indicate the tongue is retracted and not visible, so I should probably filter for high-likelihood points when computing the percentiles and discretizing"*). The plan was not carried into the code. `CONVERSION_NOTES.md` still claims the filter exists (*"Computed per-session using 33rd/67th percentile thresholds on valid (likelihood > 0.9) tongue_y values"*), which matches neither the code's percentiles (40/60) nor its lack of a likelihood filter.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps: (1) the 40th and 60th percentiles of the raw `tongue_y` trace are computed once per session over **every** frame of the session; (2) for each trial and each bin, the single camera frame whose timestamp is nearest the bin centre is selected; (3) that frame's y value is compared against the two thresholds. No averaging within a bin, no confidence filtering, no interpolation. In practice ~89% of frames in a session have `tongue_likelihood < 0.5` — the tongue is retracted and the tracker is reporting a spurious position — so the thresholds and the resulting classes are dominated by untracked frames. (For the first session, the 40/60 percentiles are 282.5/296.5 over all frames but 273.7/288.5 over visible frames only.)

ii.
```python
# Compute tongue y percentiles over entire session
tongue_y_p40 = np.percentile(tongue_y_all_data, 40)
tongue_y_p60 = np.percentile(tongue_y_all_data, 60)
```
```python
# Tongue y-position (time-varying, discretized)
# Vectorized: find closest tongue timestamp for each bin center
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
# Check if previous index is closer
prev_indices = np.clip(tongue_indices - 1, 0, len(tongue_timestamps) - 1)
dist_curr = np.abs(tongue_timestamps[tongue_indices] - abs_times)
dist_prev = np.abs(tongue_timestamps[prev_indices] - abs_times)
use_prev = dist_prev < dist_curr
tongue_indices[use_prev] = prev_indices[use_prev]

tongue_y_values = tongue_y_all_data[tongue_indices]
```

iii. The intent stated at step 36 was *"I'll extract all tongue y-positions from the session, compute the 40th and 60th percentiles, then discretize each time bin accordingly"*, with the likelihood filter noted as desirable but left out of the implementation. Nearest-neighbour sampling was chosen as the "vectorized" replacement for an earlier per-bin loop during the optimisation pass at step 50 (*"Vectorize the tongue y computation instead of looping over bins"*). The AI noticed the consequence in one session but did not trace it back (step 105: *"session 131 has tongue_y_position distribution of (0.991, 0.003, 0.006)... This might be a session where the tongue tracking is bad"*).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories, per session: `0` below the 40th percentile, `1` between the 40th and 60th, `2` above the 60th. The **fourth category required by the instructions — `3: not visible` — is not implemented**; every bin receives one of the three "visible" classes regardless of whether the tongue was actually tracked. The thresholds are recomputed per session, as specified. Dataset-wide distribution: 38.5% / 19.0% / 42.4%.

ii.
```python
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
```
```python
output_values = [
    ...
    ['low', 'mid', 'high'],
]
```

iii. The percentile split and the per-session scope follow the decoder instructions directly. The AI's plan at step 36 was to *"discretize the y-position into three bins (low, mid, high)"* — it never revisited the fourth class listed in the task specification, and its running notes record the output only as *"3 classes... 0=low, 1=mid, 2=high"*.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as the spikes, so each bin's absolute centre time `go_time + BIN_CENTERS` is looked up against the camera timestamp array with `searchsorted`, and the nearer of the bracketing frames is taken. The result is one frame per bin on the same go-cue-relative grid as the firing rates. Indices are clipped to the array bounds, so a bin with no nearby frame (e.g. before the camera started, or across one of the session's inter-recording gaps) silently takes the value of the closest frame in time, however distant. At ~294 Hz and 50 ms bins this normally selects one of ~15 frames inside the bin — spot-checking one session, every bin centre had a frame within 25 ms.

ii.
```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
prev_indices = np.clip(tongue_indices - 1, 0, len(tongue_timestamps) - 1)
dist_curr = np.abs(tongue_timestamps[tongue_indices] - abs_times)
dist_prev = np.abs(tongue_timestamps[prev_indices] - abs_times)
use_prev = dist_prev < dist_curr
tongue_indices[use_prev] = prev_indices[use_prev]
```

iii. From step 36: *"I need to pull the y-position data from the BehavioralTimeSeries... and resample it to align with the 50ms neural bins. The tongue data comes in at roughly 300Hz, so I'll need to aggregate it down to match the neural sampling rate."* The implementation sub-samples rather than aggregates, which the AI adopted in the step-50 vectorisation pass as the cheaper way to get one value per bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several distinct mechanisms, of differing thoroughness:
 - **Whole-session failure**: every session is processed inside a blanket `try/except Exception`; any error prints a traceback and the session is skipped rather than aborting the run.
 - **Unlabelled / unusable sessions**: a session with no `classification == 'good'` unit that also maps to a region is returned as `None` and dropped (this catches the one session whose `classification`/`anno_name` are NaN); so is a session with no control trials, or with fewer than 2 usable trials.
 - **Trials with no spike data**: excluded via `obs_intervals` and the `auto_water`/`free_water` filters (1-e). One residual trial in the final dataset (session 0, trial 159 — the last trial of a recording block) still has all-zero firing rates and was knowingly left in.
 - **Heterogeneous `photostim_power` encoding**: `'N/A'`, `'nan'` and numeric `0` are all treated as "no photostim".
 - **Unmapped CCF annotations**: `map_anno_to_region` returns `None`, the unit is dropped, and a warning is printed so the gap can be found and fixed.
 - **Missing tongue tracking**: *not* handled. Frames where the tongue is not visible are treated as valid measurements, and bins with no nearby frame silently inherit a distant frame's value.

ii.
```python
try:
    result = process_session(nwb_file, verbose=verbose)
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    continue
```
```python
no_photostim = np.array([str(p) == 'N/A' or str(p) == 'nan' or
                         (isinstance(p, (int, float)) and p == 0)
                         for p in photostim_power])
```
```python
if n_good == 0:
    if verbose:
        print(f"  SKIP: no good units with valid regions")
    io.close()
    return None
```
```python
    # Fallback
    print(f"  WARNING: Unmapped annotation: '{anno_name}'")
    return None
```

iii. The AI's approach was to exclude whatever cannot be measured and to log loudly so gaps surface on the next run — the unmapped-annotation warning is exactly what let it find the 835 missing thalamic units after the first full pass (step 90). Residual all-zero trials were judged acceptable (step 82: *"this is just 1 trial out of 160 - it's probably fine. The warning is not critical."*). The tongue case was never revisited after the likelihood filter was dropped.

## 10-a. What are the most time-consuming steps of the code?

i. Two things dominate: reading each NWB file (in particular pulling every good unit's ragged `spike_times` out of HDF5, and the ~0.7–1 M × 3 tongue tracking array), and the nested per-trial × per-unit binning loop, which performs two `searchsorted` calls plus one `np.histogram` for each of up to ~900 units × ~600 trials in a session. The AI measured ~3.5 s per session, giving ~10 minutes for the 174 files; pickling the 9.96 GB result adds to that. The photostim inner loop, which re-scans every photostim event of the session for each trial, is a smaller but real cost.

ii.
```python
for trial_idx in valid_indices:
    ...
    for i, st in enumerate(all_spike_times):
        lo = np.searchsorted(st, abs_start)
        hi = np.searchsorted(st, abs_end)
        if hi > lo:
            rel_spikes = st[lo:hi] - go_time
            counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
            trial_fr[i, :] = counts / BIN_WIDTH
```
```python
all_spike_times = []
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)
```

iii. The AI profiled and optimised once (step 50): *"The main bottleneck is `units.get_unit_spike_times(ui)` for each unit and then computing firing rates... Key optimizations: 1. Load all spike times at once rather than one at a time 2. Vectorize the tongue y computation instead of looping over bins 3. Vectorize the photostim computation"*. After that it judged the runtime acceptable and stopped (step 85: *"~3.5s per session means the full conversion should take ~10 minutes"*).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain:
 - The **trial × unit binning loop** is the big one. Because every trial uses the same relative edge grid, all trials for a unit can be binned in one pass by flattening `go[:, None] + BIN_EDGES[None, :]` into a single edge array and calling `searchsorted` once per unit, then differencing — turning an O(n_trials × n_units) Python loop into O(n_units).
 - The **per-trial photostim loop** re-scans the session's whole photostim event list for every trial; a single `searchsorted` of the event starts against the trial windows would do.
 - The **unit selection loop** in `process_session` and the `recorded_trial_indices` loop that builds `valid_trial_mask` are both plain Python loops over arrays that `np.isin` / boolean masking would express directly.
The tongue and photostim *bin-marking* steps were already vectorised in the step-50 optimisation pass.

ii.
```python
for trial_idx in valid_indices:
    ...
    for i, st in enumerate(all_spike_times):
```
```python
for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    if ps_stop > trial_start_abs and ps_start < trial_end_abs:
```
```python
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True
```

iii. The AI vectorised only where it believed the cost was (step 50, quoted above), and accepted the remaining loops once the projected total runtime fit comfortably in budget. It did not identify the trial dimension of the binning loop as vectorisable.

## 10-c. What processing does the code repeat multiple times?

i. Three repetitions:
 - `valid_indices` is iterated **twice** — once to build the neural arrays, then again from scratch to build inputs and outputs — so `go_time`, `trial_start_abs`/`trial_end_abs` are recomputed in both passes.
 - The full photostim event list is re-scanned for every trial (see 10-b), i.e. O(n_trials × n_photostim_events) comparisons per session.
 - `units.get_unit_obs_intervals(ui)` is called for every good unit purely to take the minimum length, and the obs-interval dataset is therefore read repeatedly.
Per-session quantities that genuinely need computing once — the tongue percentiles, the bin grid, the region mapping — are each computed once.

ii.
```python
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    trial_fr = np.zeros((n_good, N_BINS), dtype=np.float32)
    ...
# --- Compute inputs and outputs per trial ---
for trial_idx in valid_indices:
    go_time = go_start_times[trial_idx]
    ...
```
```python
for ui in good_indices[1:]:
    oi = units.get_unit_obs_intervals(ui)
    if len(oi) != n_obs_trials:
        n_obs_trials = min(n_obs_trials, len(oi))
```

iii. Not discussed in the trajectory. The two trial passes are a readability choice — neural extraction and behavioural extraction are kept as separate blocks — and the cost is negligible next to the binning loop inside the first pass.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
 - **The constant tone input is materialised per trial.** `TIME_FROM_TONE` is identical for all 74,769 trials, yet an 80-element copy is stacked into every trial's input array and pickled.
 - **Outputs are stored as `int64`.** All four outputs take values in 0–2, so 8 bytes per value is 8× more than needed in a 9.96 GB pickle; the decoder only uses them as class indices.
 - **A second 269 MB `sample_data.pkl`** containing the first 5 sessions is always written, in addition to the required `converted_data.pkl`. Nothing downstream reads it.
 - **Per-session statistics carried but unused**: `performance`, `correct_left`, `correct_right`, `sess_name`, `n_good`, `n_trials` are returned from `process_session` and used only for console logging and the metadata counters; `subject_id` is stored in the `all_subjects` dict and never read again.
 - **`tongue_x`** is loaded as part of the `(n_frames, 3)` slice and discarded.
 - The `Found N NWB files` pass, the per-region count tally and the summary printing traverse the assembled structure again after it is complete.

ii.
```python
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
input_data.append(input_trial)
```
```python
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
], axis=0)
```
```python
# Save sample data (first 5 sessions)
n_sample = min(5, len(all_sessions))
sample_data = { ... }
print(f"Saving sample data to {sample_output_file}...")
with open(sample_output_file, 'wb') as f:
    pickle.dump(sample_data, f)
```

iii. The `int64` dtype was adopted to fix a decoder crash on float indices (step 69) rather than chosen for size; the AI did not revisit it. The sample file was created deliberately as a fast-iteration artefact — the AI used it to test the decoder before the full run (step 122: *"let me train the decoder on sample data first, then on full data"*) — and then kept it as a deliverable.
