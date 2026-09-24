# Decisions

> Source files: `/app/convert_data.py` (code), `/app/CONVERSION_NOTES.md` (justifications),
> `/logs/agent/trajectory.json` (reasoning), `/app/conversion_full_out.txt`,
> `/app/verification_full_out.txt` (results).
>
> **Note on the instruction version the AI received.** The copy of the task in the agent's
> trajectory (step 1) specifies `choice` as *"left = 0, right = 1, per-trial"* (no "no lick"
> class) and `tongue y-position` as *"0/1/2"* only (no "not visible" class). The reference
> instructions in `/tests/instruction_reference.md` add a third choice class ("no lick") and a
> fourth tongue class ("not visible"). This difference is relevant to items 5-b, 8-c and 9 and
> is called out where it applies.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session under `data/sub-<subject_id>/`. The AI walks the
`data/` directory, collects every `sub-*` subdirectory (sorted) and every `*.nwb` file inside it
(sorted), producing a list of `(subject_dir, path)` pairs — 174 files, 28 subjects. Each file is
opened once with `pynwb.NWBHDF5IO` in a single sequential pass and all needed tables are read
from it: `nwb.subject`, `nwb.trials`, `nwb.units`, `nwb.acquisition['BehavioralEvents']` and
`nwb.acquisition['BehavioralTimeSeries']`. Each session is wrapped in a `try/except` so a failing
file is skipped rather than aborting the run. No parallelism is used.

ii.
```python
def list_nwb_files(data_dir):
    """List all NWB files organized by subject."""
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
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
...
trials = nwb.trials
n_trials = len(trials)
...
be = nwb.acquisition['BehavioralEvents']
go_times = be.time_series['go_start_times'].timestamps[:]
...
units = nwb.units
```

```python
for i, (subject, nwb_path) in enumerate(all_files):
    try:
        result = process_session(nwb_path, ...)
    except Exception as e:
        print(f'  ERROR: {e}')
        ...
        n_skipped += 1
        continue
```

iii. CONVERSION_NOTES Step 2: *"NWB files organized by subject directories: `data/sub-XXXXXX/`.
Each NWB file = one behavioral session with behavior + ecephys + (usually) ogen data."* The AI
verified the directory listing gives 174 files / 28 subjects, matching the DANDI archive
(`DANDI:000363`) and the paper's stated cohort of 28 mice. `pynwb` is used because the reference
code reads DataJoint `.mat` exports that are not available here, so the NWB fields had to be
mapped onto the equivalent `.mat` variables (Step 4 discrepancy table).

## 1-b. How are the data split into subjects?

i. The subject of a session is read from the NWB file itself (`nwb.subject.subject_id`, a numeric
string such as `'440956'`), falling back to the first underscore-delimited token of the filename
if `nwb.subject` is absent. At assembly time, `subjects` is built as a list of unique ids in
order of first appearance and `subject_idx` is the index of each session's subject in that list.
The final dataset has 28 subjects.

ii.
```python
subject_id = nwb.subject.subject_id if nwb.subject else os.path.basename(nwb_path).split('_')[0]
```

```python
for sess in session_results:
    if sess['subject_id'] not in subjects:
        subjects.append(sess['subject_id'])
    subject_idx.append(subjects.index(sess['subject_id']))
```

iii. CONVERSION_NOTES Step 2 records that the subject identity is carried both by the directory
name (`sub-XXXXXX`) and by the NWB `subject` object, and Step 9's consistency table checks
"Subjects: 28 (papers) / 28 (data) / 28 (converted) — YES". The AI used the file's own subject
field rather than the folder name so that the id comes from the data, not the layout.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is performed. The session label is
`nwb.identifier` (e.g. `SC015_20190208_133600_s2`, encoding mouse name, date, time and session
number). Session order in the output follows the sorted subject-directory / filename order.
However, sessions are **not** all retained: a behavioural session-selection filter is applied
(see 1-e), which reduced 174 candidate files to **144 output sessions** (30 skipped: ~22 for
correct rate < 65%, ~7 for left/right imbalance, 1 for no good neurons).

ii.
```python
session_id = nwb.identifier
```

```python
for i, (subject, nwb_path) in enumerate(all_files):
    ...
    result = process_session(nwb_path, ...)
    if result is None:
        n_skipped += 1
        continue
    session_results.append(result)
```

```python
'metadata': {
    ...
    'n_sessions': len(session_results),
    'n_subjects': len(subjects),
}
```

iii. CONVERSION_NOTES Step 9: *"Sessions: 173 (papers) | 174 NWB files (data) | 144 (30 skipped)
— ~83%"*, with the explanation *"Paper's 173 may use slightly different criteria (e.g., L/R counts
on all trials vs regular trials); DANDI archive likely includes some training sessions not in the
paper's 173."* The AI considered the residual gap acceptable because *"per-session neuron count
(402.3 vs 404.3) matches extremely well, confirming neuron extraction is correct"*.

## 1-d. How are the data split into trials?

i. Trials come straight from the NWB trials table (`nwb.trials`), one row per behavioural trial.
All per-trial columns are pulled as whole arrays (`start_time`, `stop_time`,
`trial_instruction`, `outcome`, `early_lick`, `auto_water`, `free_water`, `photostim_onset`,
`photostim_duration`). The trial↔go-cue mapping is checked with an assertion that there is
exactly one `go_start_times` event per trials-table row.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
trial_starts = trials['start_time'][:]
trial_stops = trials['stop_time'][:]
instructions = trials['trial_instruction'][:]  # 'left' or 'right'
outcomes = trials['outcome'][:]  # 'hit', 'miss', 'ignore'
early_licks = trials['early_lick'][:]  # 'early' or 'no early'
auto_water = trials['auto_water'][:]  # 0 or 1
free_water = trials['free_water'][:]  # 0 or 1
photostim_onset = trials['photostim_onset'][:]  # time or 'N/A'
photostim_duration = trials['photostim_duration'][:]  # duration or 'N/A'

be = nwb.acquisition['BehavioralEvents']
go_times = be.time_series['go_start_times'].timestamps[:]

assert len(go_times) == n_trials, f"Go times ({len(go_times)}) != trials ({n_trials})"
```

iii. CONVERSION_NOTES Step 2 lists the trials-table fields as the canonical per-trial record.
The trajectory (step 39) notes that `sample_start_times` and `delay_start_times` can contain
several events per trial when an early lick replays an epoch, which is why the go cue (exactly
one per trial) is used as the anchor and asserted against the trial count.

## 1-e. How are trials filtered based on quality controls?

i. Filtering happens at two levels.

**Session level (behavioural performance, from the data paper's session-selection criteria):**
a session is dropped unless (a) the correct rate on "regular" trials ≥ 0.65 and (b) there are at
least 50 correct lick-left *and* 50 correct lick-right regular trials. "Regular" = not auto-water,
not free-water, not photostim, not early-lick, and not `ignore` (ignore trials are excluded from
the denominator, so correct rate = hits / (hits + misses)). A session with zero good units is
also dropped. 30 of 174 sessions were removed this way.

**Trial level:** within a kept session a trial is retained only if
(a) `auto_water == 0` and `free_water == 0`;
(b) a tone (`sample_start`) exists between the trial start and the go cue (i.e. tone onset is not
NaN);
(c) the trial's window end (`go + 1.5 s`) is not more than 1 s past the end of the unit
`obs_intervals` (i.e. the ephys recording had not already stopped).
Early-lick, `ignore` and photostim trials are deliberately **kept** because they are required
decoder outputs/inputs. A session with fewer than 2 surviving trials is dropped. 74,894 of 78,626
trials in the kept sessions survive (95.3%).

ii.
```python
# ------ Session selection criteria (applied to ALL behavioral trials) ------
behav_valid = (auto_water == 0) & (free_water == 0)
regular_mask = behav_valid.copy()
for i in range(n_trials):
    if photostim_onset[i] != 'N/A':
        regular_mask[i] = False
    if early_licks[i] == 'early':
        regular_mask[i] = False
regular_mask &= (outcomes != 'ignore')  # exclude ignore from denominator

n_regular = np.sum(regular_mask)
if n_regular > 0:
    correct_regular = np.sum(regular_mask & (outcomes == 'hit'))
    correct_rate = correct_regular / n_regular
else:
    correct_rate = 0.0

correct_left = np.sum(regular_mask & (outcomes == 'hit') & (instructions == 'left'))
correct_right = np.sum(regular_mask & (outcomes == 'hit') & (instructions == 'right'))

if correct_rate < MIN_CORRECT_RATE:
    print(f'  SKIP: correct rate {correct_rate:.2f} < {MIN_CORRECT_RATE}')
    io.close()
    return None

if correct_left < MIN_CORRECT_LEFT or correct_right < MIN_CORRECT_RIGHT:
    print(f'  SKIP: correct left={correct_left}, right={correct_right} ...')
    io.close()
    return None
```

```python
# Get max recording time from obs_intervals of a representative unit
obs_intervals = units['obs_intervals'][good_indices_units[0]]
max_recording_time = obs_intervals[-1, 1] if len(obs_intervals) > 0 else 0

# ------ Trial filtering (behavioral + recording coverage) ------
valid_mask = behav_valid.copy()
valid_mask &= ~np.isnan(tone_onset_per_trial)

# Exclude trials beyond the neural recording period
for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False

valid_indices = np.where(valid_mask)[0]

if len(valid_indices) < 2:
    print(f'  SKIP: too few valid trials after recording filter ({len(valid_indices)})')
    io.close()
    return None
```

iii. CONVERSION_NOTES Step 3 records the paper's criteria (*"Overall behavioral performance >
65%; at least 50 correct lick-left and 50 correct lick-right trials"*) and Step 5 decision 5 is
*"Session selection: Apply paper's criteria"*. Step 4/5 decision 1 explains the trial rule:
*"Keep all trials EXCEPT auto_water and free_water. This is required by decoder specs (early_lick
as output, photostim as input, outcome includes ignore)"*, deviating from the reference code's
`get_regular_trial_mask()` which also removes early-lick/stim/ignore. Step 9 documents a mid-run
fix: the correct rate originally included `ignore` trials in the denominator, which cut sessions
to 105; excluding them (matching `get_regular_trial_mask`) raised it to 144. The recording-coverage
filter is justified in Step 10 as removing trials whose window falls outside the ephys recording;
the 126 remaining all-zero-neural trials (0.17%) are dismissed as *"negligible edge cases from
recording coverage boundaries"*.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `nwb.units['spike_times']` — the sorted spike times of each unit in session-absolute
seconds — restricted to units passing QC (see 2-c). The other inputs are
`BehavioralEvents/go_start_times` (to place the window) and `units['anno_name']` (for the
brain-region label of each retained unit).

ii.
```python
units = nwb.units
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]
...
spike_times_all = units['spike_times']
spike_times_good = [spike_times_all[idx] for idx in good_indices]
```

iii. CONVERSION_NOTES Step 1/2: the reference `.mat` pipeline bins `spike_times` from
`goodunits/` files; in NWB the equivalent is `units/spike_times` filtered by `classification`.
Step 3: *"In NWB: spike times in absolute time, need to subtract go_start_time per trial."*
Spike times are the only neural representation in the file.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For every trial, and for every good
unit, the spikes falling in `[go - 2.5 s, go + 1.5 s)` are selected with a boolean mask, assigned
to one of 80 bins by `floor((t - window_start) / 0.05)`, accumulated with `np.add.at`, and the
resulting counts are divided by the 50 ms bin width to give Hz. No smoothing, no normalisation,
no baseline subtraction, no sliding/overlapping windows (the reference code's 40 ms bin /3.4 ms
stride sliding histogram was deliberately replaced by the 50 ms non-overlapping bins the decoder
task specifies).

ii.
```python
def compute_firing_rates_vectorized(spike_times_list, go_time, t_start, t_end, bin_width, n_bins):
    n_neurons = len(spike_times_list)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width
    bin_edges_end = bin_edges_start + bin_width

    for i, spk in enumerate(spike_times_list):
        if len(spk) == 0:
            continue
        # Only keep spikes in our window
        mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
        spk_window = spk[mask]
        if len(spk_window) == 0:
            continue
        # Assign spikes to bins
        bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        np.add.at(fr[i], bin_idx, 1)

    fr /= bin_width  # convert to firing rate (Hz)
    return fr
```

```python
for trial_idx in valid_indices:
    go_time = go_times[trial_idx]
    fr = compute_firing_rates_vectorized(
        spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
    )  # (n_neurons, n_bins)
    neural_trials.append(fr)
```

iii. CONVERSION_NOTES Step 1 identifies `sliding_histogram()` in the reference code as the
firing-rate function (`binSpikes / bin_width`, i.e. Hz), and Step 4 records the deliberate
parameter change: *"Firing rate params — code: bw=0.04, stride=0.0034; Decoder spec: 50ms bins.
Use 50ms as specified"*, allowed because the decoder task overrides the reference where they
conflict.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A unit is kept only if `units['classification'] == 'good'` **and** its `anno_name` (CCF
annotation) is non-empty. No thresholds are applied to individual QC metrics. A session with zero
such units is dropped. This yields 57,935 units over the 144 retained sessions, a mean of 402.3
per session. Each retained unit's `anno_name` is then mapped to one of 14 broad regions
(ALM, Orbital, Striatum, Pallidum, Hippocampus, Thalamus, Hypothalamus, Midbrain, Pons, Medulla,
Cerebellum, Olfactory, CorticalSubplate, OtherCortex) via a hand-written lookup with prefix and
substring fallbacks, defaulting to `OtherCortex` for unmatched names.

ii.
```python
units = nwb.units
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]

if len(good_indices_units) == 0:
    print(f'  SKIP: no good neurons')
    io.close()
    return None
```

```python
annos_good = anno_names[good_mask_units]
neuron_regions = [map_anno_to_region(a) for a in annos_good]
```

iii. CONVERSION_NOTES Step 1: *"QC mode: 'classifier' — region-specific classifiers... In NWB,
`classification == 'good'` is equivalent to passing the QC classifier."* Step 3 curation rules:
*"QC classifier: classification == 'good' in NWB; must have both ephys and histology (CCF
coordinates); anno_name must not be empty"*, mirroring the reference
`helper_get_neuron_id_area()`, which filters on QC **and** CCF annotation. `unit_quality`
('good'/'multi') was noted in Step 2 but not used. The sanity check in Steps 9/10 is the mean
units/session (402.3 vs the paper's 404.3), which the AI treats as confirming the unit filter even
though the total (57,935 vs 69,943) is short because of the dropped sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset, taken from `BehavioralEvents/go_start_times.timestamps`, one per
trial. Spike times and event timestamps live on the same session-absolute clock, so no resampling
or offset correction is applied: for each trial the absolute window `[go - 2.5, go + 1.5)` is used
directly when masking and binning spikes, which places bin 0 at -2.5 s and the go cue at the
boundary between bins 49 and 50.

ii.
```python
be = nwb.acquisition['BehavioralEvents']
go_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_times) == n_trials, f"Go times ({len(go_times)}) != trials ({n_trials})"
```

```python
mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
spk_window = spk[mask]
bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
```

```python
'temporal_alignment_event': 'Go cue onset',
'off_start': T_START,   # -2.5
'off_end': T_END,       #  1.5
```

iii. CONVERSION_NOTES Step 3/4: *"Spike times already aligned to go cue in source data [reference
.mat]; in NWB: spike times in absolute time, need to subtract go_start_time per trial."* The
decoder task specifies go-cue alignment and the [-2.5, +1.5] s window. Trajectory step 35: *"Spike
times are in absolute time (not relative to go cue). Need to subtract go cue time."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial spanning -2.5 s to +1.5 s relative to the go cue,
identical for every trial and every session. Rebinning relative to the reference pipeline is
applied: the reference code uses a 40 ms sliding window with a 3.4 ms stride over [-3, 3] s; the
AI replaced this with the decoder task's 50 ms non-overlapping binning. Spikes are binned directly
from raw spike times (i.e. there is no two-stage rebinning of an intermediate rate). Tongue video
(~294 Hz) is subsampled to the same 80-bin grid. `metadata['time_bin_size']` is recorded as 50.0 ms.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins (decoder task spec)
T_START = -2.5    # seconds before go cue
T_END = 1.5       # seconds after go cue
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
```

```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
```

```python
'time_bin_size': BIN_WIDTH * 1000,  # in ms
```

iii. CONVERSION_NOTES Step 4: *"Bin size: Decoder task specifies 50ms bins, different from
reference code's 40ms bins. This is allowed per instructions."* Step 5 decision 6 repeats this,
and Step 9's consistency table lists "Time bins: 40 ms (ref code) vs 50 ms (ours) — Different by
design" and "Window: [-3,3] s (ref code) vs [-2.5,1.5] s (ours) — Different by design".
Verification confirms T = 80 for every trial of every session.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times.timestamps` (the sample-epoch tone onsets), combined
with the trial's `start_time` and its go-cue time. For each trial the AI takes the sample events
falling in `[trial_start, go_cue]` and uses the **last** one; if none exists the tone onset is NaN
and the trial is dropped (see 1-e).

ii.
```python
sample_starts = be.time_series['sample_start_times'].timestamps[:]

# Map sample_start to each trial: find the last sample_start before the go cue within each trial
tone_onset_per_trial = np.full(n_trials, np.nan)
for i in range(n_trials):
    # Find sample events within this trial's window
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
    if len(in_trial) > 0:
        # Take the LAST sample onset (in case of replays from early licking)
        tone_onset_per_trial[i] = in_trial[-1]
```

iii. Trajectory step 39: *"Trial 1 has multiple sample_start events within it — this might be from
early licks causing re-samples."* The AI initially considered the first sample event and then
settled on the last, as the code comment states: *"Take the LAST sample onset (in case of replays
from early licking)"*, i.e. the tone the animal actually used to make its decision.
CONVERSION_NOTES Step 5 maps `sample_start_times → input[0]`.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value per bin: the bin centre (go-cue-relative) minus the
go-cue-relative tone time, i.e. seconds elapsed since the tone at the centre of each bin. It is
stored as `float32` in row 0 of the `(2, 80)` input array. No clipping, no normalisation, no
binarisation; values run from about -1.5 s (bin centres before the tone) up to ~11.9 s on trials
with a long replayed sample epoch.

ii.
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
...
tone_time = tone_onset_per_trial[trial_idx]
tone_relative = tone_time - go_time  # tone onset relative to go cue (negative)
time_from_tone = bin_centers - tone_relative  # time since tone onset at each bin
...
input_data = np.stack([time_from_tone.astype(np.float32), photostim], axis=0)  # (2, n_bins)
```

iii. CONVERSION_NOTES Step 5 decision 7: *"Time from tone onset: Continuous variable = current_time
- sample_start_in_trial (in go-cue-relative coords)"*, following the decoder task's requirement for
a continuous, time-varying input. Step 12 sanity-checks the resulting range: *"time_from_tone_onset
range: [-1.5, 11.9]. Sessions 0-41 have min -0.6 (shorter presample), sessions 42+ have min -1.5.
Max values up to 11.9 are plausible for long-latency response trials."*

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same go-cue-relative grid as the firing rates: `bin_centers` is built once from
`T_START`, `BIN_WIDTH` and `N_BINS`, so element *k* of the input corresponds to the same 50 ms
interval as column *k* of the neural matrix. The tone time is converted into the same
go-cue-relative frame by subtracting `go_time`, so no separate alignment step is required.

ii.
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
```

```python
tone_relative = tone_time - go_time
time_from_tone = bin_centers - tone_relative
```

iii. CONVERSION_NOTES Step 5 "Trial Timeline (relative to go cue = 0): Window [-2.5, +1.5] s;
50 ms bins → 80 time bins per trial; Sample onset: typically at ~-1.85 s; Go cue: 0 s." Everything
in the NWB file is on one global clock, so subtracting the go cue puts all streams on the same
axis. The `--show-processing` plots overlay `time_from_tone` for 10 trials against the same
`bin_centers` axis used for firing rates as a visual alignment check.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the trials-table columns `photostim_onset` and `photostim_duration` (stored as strings,
with `'N/A'` on unstimulated trials), plus `trials['start_time']` and the go-cue time to put them
on the trial's time axis. `BehavioralEvents/photostim_start_times` exists but is not used.

ii.
```python
photostim_onset = trials['photostim_onset'][:]  # time or 'N/A'
photostim_duration = trials['photostim_duration'][:]  # duration or 'N/A'
```

```python
if photostim_onset[trial_idx] != 'N/A':
    ps_onset = float(photostim_onset[trial_idx])
    ps_duration = float(photostim_duration[trial_idx])
```

iii. CONVERSION_NOTES Step 2 lists `photostim_onset/power/duration` among the trials-table fields,
and Step 5's mapping table gives `photostim_onset/duration → input[1]: photostim_on`, "Binary
(0/1) per time bin, 1 during photostim window, 0 otherwise". Step 3 records the paper's
expectation of photostim on ~25% of trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0.0/1.0) time series over the 80 bins. The string onset is parsed as a float
*relative to trial start*, converted to absolute time by adding `trial_starts[i]`, then to
go-cue-relative time by subtracting `go_time`; the offset is onset + duration. A bin is set to 1
if its centre lies in `[onset_rel, end_rel)`. Trials with `'N/A'` keep an all-zero row. The loop
over the 80 bins is explicit rather than vectorised.

ii.
```python
photostim = np.zeros(N_BINS, dtype=np.float32)
if photostim_onset[trial_idx] != 'N/A':
    ps_onset = float(photostim_onset[trial_idx])
    ps_duration = float(photostim_duration[trial_idx])
    # photostim_onset is relative to trial start, need to convert to absolute then go-relative
    ...
    ps_onset_abs = trial_starts[trial_idx] + ps_onset
    ps_end_abs = ps_onset_abs + ps_duration
    ps_onset_rel = ps_onset_abs - go_time
    ps_end_rel = ps_end_abs - go_time

    for b in range(N_BINS):
        bc = bin_centers[b]
        if ps_onset_rel <= bc < ps_end_rel:
            photostim[b] = 1.0
```

iii. The in-code comment records the reasoning: *"Looking at the data: photostim_onset values are
small (1.8) suggesting relative to trial start"* — the AI inferred the reference frame from the
magnitude of the stored values rather than from documentation (several exploratory comments
remain in the shipped code). CONVERSION_NOTES Step 5 decision 8: *"Photostim input: Binary time
series; 1 when photostim is active, 0 otherwise"*, per the decoder task's requirement that
photostimulation be represented at every time point. Step 12 verifies *"photostim_on: Binary [0,1]
as expected. 3 sessions have no photostim trials (max=0), which is valid."*

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. By converting the onset/offset into go-cue-relative seconds (`- go_time`) and comparing against
the same `bin_centers` array used for the neural binning, so bin *k* of the photostim row covers
the same interval as column *k* of the firing-rate matrix.

ii.
```python
ps_onset_rel = ps_onset_abs - go_time
ps_end_rel = ps_end_abs - go_time

for b in range(N_BINS):
    bc = bin_centers[b]
    if ps_onset_rel <= bc < ps_end_rel:
        photostim[b] = 1.0
```

iii. Same rationale as 3-c: one global clock, so subtracting the go cue is sufficient. The
`--show-processing` plot "Input: photostim" draws the stimulus rows of up to 10 stimulated trials
on the `bin_centers` axis; CONVERSION_NOTES Step 3 notes the paper's photoinhibition is delivered
in the late delay (0.5 s), which the AI used as the expected location of the 1s in the plot.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no explicit choice column, so it is derived from two trials-table columns,
`trial_instruction` (`'left'`/`'right'`) and `outcome` (`'hit'`/`'miss'`/`'ignore'`): a hit means
the animal licked the instructed side, a miss means it licked the other side.

ii.
```python
instructions = trials['trial_instruction'][:]  # 'left' or 'right'
outcomes = trials['outcome'][:]  # 'hit', 'miss', 'ignore'
...
instr = instructions[trial_idx]
outcome = outcomes[trial_idx]
```

iii. CONVERSION_NOTES Step 5 mapping table: *"trial_instruction + outcome → output[0]: choice;
hit: choice=instruction; miss: choice=opposite; ignore: choice=instruction"*. The lick-direction
itself is not stored in the trials table, so it has to be reconstructed from the instruction and
the outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded `left = 0`, `right = 1` (two classes only). Hit → the instructed side; miss →
the opposite side; **`ignore` → the instructed side** as well. The scalar is repeated across all
80 bins of row 0 of the `(4, 80)` int64 output array, and `output_values[0] = ['left', 'right']`.
Because ignore trials are folded back onto the instruction, the resulting distribution is exactly
50.0% / 50.0% left/right.

ii.
```python
# --- Output 0: choice (left=0, right=1) ---
instr = instructions[trial_idx]
outcome = outcomes[trial_idx]
if outcome == 'hit':
    choice = 0 if instr == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instr == 'left' else 0  # wrong lick = opposite
else:  # ignore
    choice = 0 if instr == 'left' else 1  # assign instruction direction
```

```python
output_data = np.array([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
], dtype=np.int64)  # (4, n_bins)
```

```python
'output_values': [
    ['left', 'right'],           # choice
    ...
```

iii. CONVERSION_NOTES Step 5 decision 2: *"Choice for ignore trials: Set to instruction direction
(the 'correct' choice), since there's no actual lick."* The version of the task the agent received
defined choice as only `left = 0, right = 1`, with no "no lick" category, so the AI had to place
the ~11% of ignore trials into one of the two classes rather than giving them their own.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which already contains exactly the
three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcomes = trials['outcome'][:]  # 'hit', 'miss', 'ignore'
```

iii. CONVERSION_NOTES Step 5 mapping table: *"outcome → output[1]: outcome, ignore=0, miss=1,
hit=2 — Direct mapping from NWB"*. No derivation is needed because the categories in the file are
the categories the decoder task asks for.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The string is mapped with a fixed dictionary to `ignore = 0, miss = 1, hit = 2` and the scalar
is repeated across all 80 bins in row 1 of the output array. `output_values[1] =
['ignore', 'miss', 'hit']`. The resulting distribution is 10.8% ignore / 15.3% miss / 73.9% hit.

ii.
```python
# --- Output 1: outcome (ignore=0, miss=1, hit=2) ---
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
```

```python
output_data = np.array([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    ...
```

iii. The code assignment follows the decoder task verbatim. Repeating the per-trial value across
bins keeps all four outputs in a single `(n_output, n_timepoints)` array as the target format
requires. CONVERSION_NOTES Step 10 checks the distribution: *"outcome ~74% hit (paper: 84% on
regular trials — our lower rate is expected since we include early_lick/stim/ignore trials)"*.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'no early'` / `'early'`.

ii.
```python
early_licks = trials['early_lick'][:]  # 'early' or 'no early'
```

iii. CONVERSION_NOTES Step 5 mapping table: *"early_lick → output[2]: early_lick, no=0, yes=1 —
Direct mapping from NWB"*. The flag is explicit in the file, so nothing has to be re-derived from
the lick-time streams.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'no early'` → 0, anything else → 1 (implemented as an `if/else` rather than a dictionary
lookup, so any unexpected string would silently become 1). The scalar is repeated across all 80
bins in row 2, and `output_values[2] = ['no', 'yes']`. Distribution: 88.5% no / 11.5% yes.

ii.
```python
# --- Output 2: early lick (no=0, yes=1) ---
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
```

```python
output_data = np.array([
    ...
    np.full(N_BINS, early_val, dtype=np.int64),
    ...
```

iii. The code assignment follows the decoder task. The AI kept early-lick trials in the dataset
(CONVERSION_NOTES Step 3/4: *"NOTE: For decoder task, we KEEP early lick, ignore, and stimulation
trials (required by decoder specs)"*) even though the data paper excludes them from its analyses,
because early lick is a required output. Step 10 sanity-checks the rate (*"early_lick ~88.5% no
(reasonable)"*).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: its `data` array is
`(n_frames, 3)` with columns interpreted as x, y, likelihood, and its `timestamps` give the frame
times (~294 Hz). Column 1 is the y position and column 2 is the DeepLabCut likelihood. Presence of
the series is checked first (`has_tongue`).

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
has_tongue = 'Camera0_side_TongueTracking' in bts.time_series
if has_tongue:
    tongue_data = bts.time_series['Camera0_side_TongueTracking'].data[:]  # (n_frames, 3): x, y, likelihood
    tongue_ts = bts.time_series['Camera0_side_TongueTracking'].timestamps[:]
    tongue_y = tongue_data[:, 1]
    tongue_likelihood = tongue_data[:, 2]
```

iii. CONVERSION_NOTES Step 2: *"BehavioralTimeSeries: Camera0_side_JawTracking,
Camera0_side_NoseTracking, Camera0_side_TongueTracking (x, y, likelihood at dt=0.0034 s)"* — the
column layout was established during data exploration (trajectory steps 35/39) and this is the only
tongue measurement in the file. Step 3 notes the paper's 300 Hz video rate, matching the observed
dt = 0.0034 s.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Two stages. (1) **Per-session thresholds**: frames with `likelihood > 0.5` are taken as
"visible", and the 40th and 60th percentiles of the raw y values of those visible frames are
computed once per session (falling back to percentiles over *all* frames if fewer than 100 visible
frames exist). (2) **Per-bin value**: for each of the 80 bins, the frame index at or after the bin
centre is found with `np.searchsorted` and that single frame's y value is taken — no averaging
within the bin, and no likelihood check at this stage. The value is then thresholded (8-c). If a
session had no tongue series at all, the whole row is filled with class 1.

ii.
```python
if has_tongue:
    tongue_visible = tongue_likelihood > 0.5
    if np.sum(tongue_visible) > 100:
        visible_y = tongue_y[tongue_visible]
        p40 = np.percentile(visible_y, 40)
        p60 = np.percentile(visible_y, 60)
    else:
        # Not enough visible tongue data
        p40 = np.percentile(tongue_y, 40)
        p60 = np.percentile(tongue_y, 60)
```

```python
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    # Find closest tongue frame
    t_idx = np.searchsorted(tongue_ts, bc_abs)
    t_idx = min(t_idx, len(tongue_ts) - 1)
    ty = tongue_y[t_idx]
```

```python
else:
    tongue_y_trial = np.ones(N_BINS, dtype=np.float32)  # default to middle
```

iii. CONVERSION_NOTES Step 5 decision 3: *"Tongue y-position: Use the side camera tongue tracking
y-coordinate. Discretize per session using 40th/60th percentile thresholds over ALL valid tongue
positions in the session."* The likelihood > 0.5 criterion is how "valid" is operationalised for
the percentile computation. Step 12: *"tongue_y distribution: 64% low, 19% mid, 17% high. Skewed
toward 'low' because tongue is mostly retracted. Per-session discretization (40th/60th percentile)
is working correctly."*

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three classes via the per-session percentile edges: `y < p40 → 0`, `p40 ≤ y < p60 → 1`,
`y ≥ p60 → 2`, with `output_values[3] = ['low', 'mid', 'high']`. The thresholds are derived from
visible frames only, but they are **applied to the nearest frame's y value regardless of its
likelihood**, so bins in which the tongue is not protruding are still assigned one of the three
classes from the tracker's residual (meaningless) output. There is no "not visible" class. In the
sample session inspected here only ~14% of frames are visible, so the majority of bins are
classified from low-likelihood values, and the resulting class fractions (0.638 / 0.190 / 0.172)
are far from the 40/20/40 split the percentile definition would produce if applied to the
quantity the thresholds were derived from.

ii.
```python
ty = tongue_y[t_idx]
if ty < p40:
    tongue_y_trial[b] = 0
elif ty < p60:
    tongue_y_trial[b] = 1
else:
    tongue_y_trial[b] = 2
```

```python
'output_values': [
    ...
    ['low', 'mid', 'high'],      # tongue_y
],
```

iii. CONVERSION_NOTES Step 5 decision 3 (percentile-based per-session discretisation) and Step 12
(*"Skewed toward 'low' because tongue is mostly retracted"*) are the AI's justification: it
interpreted the skew as a real property of the behaviour rather than as an artefact of classifying
frames in which no tongue is present. The task version the agent received listed only classes
0/1/2, with no "not visible" option, so every bin had to receive one of the three.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Nearest-frame lookup on the shared session-absolute clock: for bin *b* the absolute time of the
bin centre is `go_time + bin_centers[b]`, and `np.searchsorted` on the camera timestamps returns
the first frame at or after that time (clamped to the last frame). The resulting class array is
stored in row 3 of the same `(4, 80)` output array, so bin *k* nominally corresponds to the same
interval as neural column *k*. There is no check that the retrieved frame is temporally close to
the bin centre, and the video is trial-gated (gaps of up to ~300 s between blocks were observed),
so bins that fall in periods when the camera was off are filled with a frame from an arbitrarily
distant time.

ii.
```python
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    # Find closest tongue frame
    t_idx = np.searchsorted(tongue_ts, bc_abs)
    t_idx = min(t_idx, len(tongue_ts) - 1)
    ty = tongue_y[t_idx]
```

iii. Camera timestamps share the global clock with the spikes and the go cues, so the AI treated a
direct timestamp lookup as sufficient and used exactly the same `bin_centers` grid for both
streams. The `--show-processing` plot "Output: tongue y (discretized)" overlays 5 trials on the
same time axis as the firing-rate plots as the visual alignment check
(CONVERSION_NOTES Step 7: "Processing Plots Review" reports no anomalies).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, handled with a mixture of exclusion and defaults:
- **Session that was never quality-controlled** (all `classification` NaN/non-'good'): no good
  units → session returns `None` and is skipped.
- **Any exception while processing a session**: caught, the traceback printed, the session counted
  as skipped, and the run continues.
- **Trial with no tone event** between trial start and the go cue: tone onset is NaN and the trial
  is excluded.
- **Trials past the end of the ephys recording**: excluded via the `obs_intervals` end time plus a
  1 s slack. Trials *before* the start of the recording are **not** excluded, which leaves 126
  all-zero-neural trials (0.17%, mostly in one session) in the output.
- **Session with no tongue tracking**: the whole tongue row is filled with class 1 ("mid").
- **Bins with no visible tongue**: not treated as missing; the tracker's value is used anyway.
- **CCF annotation not in the region lookup**: a warning is printed and the neuron is assigned to
  `OtherCortex`.
- **Empty `anno_name`**: the unit is dropped.

ii.
```python
except Exception as e:
    print(f'  ERROR: {e}')
    import traceback
    traceback.print_exc()
    n_skipped += 1
    continue
```

```python
valid_mask &= ~np.isnan(tone_onset_per_trial)
for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False
```

```python
else:
    tongue_y_trial = np.ones(N_BINS, dtype=np.float32)  # default to middle
```

```python
# Fallback
print(f'  WARNING: Unmapped annotation: "{anno_name}"')
return 'OtherCortex'
```

iii. CONVERSION_NOTES Step 10: *"Zero-neural-data trials: 126/74894 (0.17%) — negligible edge cases
from recording coverage boundaries"* and *"Unmapped CCF annotations: Several annotations not in
mapping (Suprageniculate nucleus, Fields of Forel, etc.) → defaulting to OtherCortex is acceptable
since these are rare regions."* Step 12 repeats that the all-zero trials are *"concentrated in
session 34. Edge cases at recording boundaries. Negligible impact."* The AI's general stance is to
drop data only where a whole session or trial is unusable and otherwise substitute a default.

## 10-a. What are the most time-consuming steps of the code?

i. By far the dominant cost is the per-trial firing-rate computation. `compute_firing_rates_vectorized`
is called once per trial and, inside, loops over every good unit and builds a boolean mask over
that unit's **entire** session-long spike train. With ~400 units and ~500 trials per session this
is ~200,000 full-array scans per session, i.e. the whole spike data of the session is re-scanned
~500 times. Secondary costs are reading the tongue tracking array (~1 M x 3 doubles) and the
`spike_times` ragged reads, and the per-bin/per-trial Python loops for photostim and tongue. The
full run took **30.2 minutes** for 144 processed sessions (~12.6 s per session, up to 51.6 s for
the largest), plus pickling a 10.0 GB output. This exceeds the 15-minute budget the instructions
set for the full conversion; the AI estimated ~30 minutes in Step 7 and ran it anyway rather than
optimising first.

ii.
```python
for trial_idx in valid_indices:
    go_time = go_times[trial_idx]
    fr = compute_firing_rates_vectorized(
        spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
    )  # (n_neurons, n_bins)
```
```python
for i, spk in enumerate(spike_times_list):
    ...
    mask = (spk >= go_time + t_start) & (spk < go_time + t_end)   # full-array scan, per trial
    spk_window = spk[mask]
```
```python
spike_times_good = [spike_times_all[idx] for idx in good_indices]   # one ragged read per unit
```

iii. CONVERSION_NOTES Step 7 records only *"~5-10 s per session, ~30 min total for full
conversion"*, with no per-step breakdown and no "Speed-ups Implemented" table (that part of the
template was left unfilled). The script prints per-session timing and a running estimate of the
remaining time, which is the AI's only profiling mechanism.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Most loops in the script are vectorisable:
1. **The trial loop x unit loop for firing rates** — the entire binning can be done with one
   `np.searchsorted` per unit against a flattened array of all trials' bin edges (as the reference
   does), removing the per-trial re-scan entirely; this alone accounts for most of the runtime.
2. **`tone_onset_per_trial` loop** — a single `np.searchsorted(sample_starts, go_times)` gives the
   last tone before each go cue.
3. **The `regular_mask` loop** over trials (`photostim_onset != 'N/A'`, `early_licks == 'early'`)
   — both are elementwise array comparisons.
4. **The recording-coverage loop** over trials — one vectorised comparison.
5. **The photostim per-bin loop** — `(bin_centers >= on) & (bin_centers < off)`.
6. **The tongue per-bin loop** — one `np.searchsorted(tongue_ts, go_time + bin_centers)` call for
   all 80 bins, and `np.digitize` for the thresholding.
7. `map_anno_to_region` does a linear scan over a ~200-entry dict for every unit, and
   `brain_regions.index(r)` in `build_dataset` is an O(n) list search per neuron.

ii.
```python
for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
    if len(in_trial) > 0:
        tone_onset_per_trial[i] = in_trial[-1]
```
```python
for b in range(N_BINS):
    bc = bin_centers[b]
    if ps_onset_rel <= bc < ps_end_rel:
        photostim[b] = 1.0
```
```python
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
    t_idx = min(t_idx, len(tongue_ts) - 1)
    ty = tongue_y[t_idx]
    if ty < p40: ...
```

iii. The instructions asked for vectorised loops and for optimisation if the estimated runtime
exceeded 15 minutes. CONVERSION_NOTES Step 6 leaves both "Code inefficiencies identified" and
"Code speedups added" blank, and Step 7's run-time table is not filled in, so no justification for
leaving these loops in place is given. The function name
`compute_firing_rates_vectorized` suggests the AI believed the inner computation was already
vectorised (the per-spike binning is; the per-trial/per-unit iteration is not).

## 10-c. What processing does the code repeat multiple times?

i.
- **The full spike-train scan is repeated once per trial per unit** (`mask = (spk >= ...) & (spk < ...)`
  over the entire session spike train), so each session's spike data is traversed ~n_trials times
  instead of once.
- **`bin_centers` is recomputed** inside `make_processing_plots` (and is a per-session recomputation
  of a constant that could live at module level).
- **`behav_valid` / trial-mask logic is computed twice** — once for the session-selection
  `regular_mask` and again for the trial `valid_mask`.
- **`photostim_onset[i] != 'N/A'` is evaluated twice** per trial (once in the session filter loop,
  once in the per-trial input construction).
- The per-trial loop re-derives `go_time`, the window bounds and the bin edges for every trial
  rather than building one grid once.

ii.
```python
bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width   # rebuilt every trial
bin_edges_end = bin_edges_start + bin_width
for i, spk in enumerate(spike_times_list):
    mask = (spk >= go_time + t_start) & (spk < go_time + t_end)       # rescans all spikes
```
```python
regular_mask = behav_valid.copy()
for i in range(n_trials):
    if photostim_onset[i] != 'N/A':
        regular_mask[i] = False
...
valid_mask = behav_valid.copy()
```

iii. Not discussed in CONVERSION_NOTES — the repeated spike scanning is a side effect of
structuring the conversion as "loop over trials, compute everything for this trial", which makes
the per-trial code readable but forces the neural data to be re-read for each trial. Each NWB file
is at least opened only once.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A handful of small items, none of them expensive relative to the firing-rate loop:
- `bin_edges_start` and `bin_edges_end` are computed in `compute_firing_rates_vectorized` on every
  call and **never used** (the binning is done by `floor()` arithmetic instead).
- `trial_stops` is read from the trials table and never used.
- `correct_left` / `correct_right` are always computed even when the correct-rate test has already
  been decided, and `n_regular` is used only transiently.
- The whole `obs_intervals` array of the first good unit is read although only `[-1, 1]` is used.
- `correct_rate` is stored in each session result but only reaches a text box in the optional
  processing plot.
- `process_session` accepts `show_processing` and `session_idx` parameters that it never uses.
- The `auto_water` / `free_water` columns are read in full for every session even when the session
  is skipped immediately afterwards.
- `map_anno_to_region` walks the full mapping dict for every unit even after an exact-match miss
  could have been resolved by a normalised key.

ii.
```python
bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width
bin_edges_end = bin_edges_start + bin_width     # never referenced again
```
```python
trial_stops = trials['stop_time'][:]            # never referenced again
```
```python
def process_session(nwb_path, show_processing=False, session_idx=0):
    ...                                         # neither argument is used in the body
```

iii. Not discussed in CONVERSION_NOTES; these are leftovers from the exploratory development of
the script (several exploratory comments, e.g. *"Actually check: is it absolute or relative?"*,
also remain in the shipped code). None of them changes the output.
