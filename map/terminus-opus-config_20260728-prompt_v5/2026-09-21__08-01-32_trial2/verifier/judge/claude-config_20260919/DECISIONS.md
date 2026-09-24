# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI 000363) is one NWB file per session, stored under `/app/data/sub-<subject_id>/`. The AI enumerates subject directories with `os.listdir` (filtering for the `sub-` prefix), then globs `*.nwb` inside each one, producing a sorted list of `(subject_dir, file_path)` pairs. This yields 174 files from 28 subjects. Each file is opened once with `pynwb.NWBHDF5IO` inside a `with` block and everything needed (subject, trials table, behavioral events, behavioral time series, units) is read from that single open handle in one pass. Results are accumulated per session into lists and assembled into the target dictionary at the end. `--sample` restricts processing to 2 sessions (deliberately skipping the first subject's first session); `--full` (the default) processes all of them.

ii.
```python
DATA_DIR = '/app/data'

def get_nwb_files(data_dir):
    """Get all NWB files organized by subject."""
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
        for f in nwb_files:
            all_files.append((subj, f))
    return all_files
```

```python
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
    subject_id = nwb.subject.subject_id
    session_id = nwb.identifier
    trials = nwb.trials
    be = nwb.acquisition['BehavioralEvents']
    units = nwb.units
    bts = nwb.acquisition['BehavioralTimeSeries']
```

```python
all_files = get_nwb_files(DATA_DIR)
print(f"Found {len(all_files)} NWB files from {len(set(s for s,_ in all_files))} subjects")
...
for i, (subj, fpath) in enumerate(all_files):
    result = process_session(fpath, show_processing=show, session_idx=i)
```

iii. From the trajectory (steps 4, 9–10): the AI first determined that the reference code in `/app/code` operates on preprocessed `.mat`/pickle exports from DataJoint, not on NWB, so the loading layer had to be written from scratch against the published NWB files; it then explored one NWB file to enumerate `units`, `trials`, `acquisition/BehavioralEvents` and `acquisition/BehavioralTimeSeries`. CONVERSION_NOTES Step 2 records "NWB files with units (spike_times, classification, anno_name, obs_intervals), trials, behavioral events/timeseries" and Step 0 records "28 subject directories with 174 NWB files". The directory listing is treated as the complete set of sessions, and sorting makes the order deterministic.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the NWB file itself, `nwb.subject.subject_id` (a numeric string such as `'440957'`), not from the enclosing folder name. Each retained session contributes its subject id to a list; at assembly the unique ids are sorted into `subjects`, and `subject_idx` is built by looking each session's id up in that sorted list. This gives 28 subjects with 3–10 sessions each.

ii.
```python
subject_id = nwb.subject.subject_id
...
return {
    'subject_id': subject_id,
    'session_id': session_id,
    ...
}
```

```python
all_subject_ids.append(result['subject_id'])
...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
...
'subjects': unique_subjects,
'subject_idx': subject_idx,
```

iii. `subject_id` is the canonical animal identifier stored in the NWB `Subject` object, and the directory name `sub-440957` is derived from it, so no separate grouping step is required. The AI's verification output confirms 28 subjects with the expected per-subject session counts, matching the dandiset. The AI did not map the numeric ids back to the paper's mouse names (e.g. `440957` → `SC016`), but it does retain `nwb.identifier` (which contains the mouse name) in `metadata['session_ids']`.

## 1-c. How are the data split into sessions?

i. No splitting is performed: one NWB file is one session. Session order in the output follows the sorted `(subject dir, filename)` enumeration. Each session is labelled with `nwb.identifier` (e.g. `SC016_20190211_143614_s1`, encoding mouse / date / time / session number) and the full ordered list is written to `metadata['session_ids']`. Sessions that yield no good units, or fewer than 2 usable trials, return `None` and are dropped; 173 of 174 files survive.

ii.
```python
session_id = nwb.identifier
```

```python
if n_good == 0:
    print(f"  Session {session_id}: No good units, skipping")
    return None
...
if n_valid_trials < 2:
    print(f"  Session {session_id}: Only {n_valid_trials} valid trials within recording, skipping")
    return None
```

```python
'metadata': {
    ...
    'session_ids': all_session_ids,
    'n_sessions': len(all_neural),
    'n_subjects': len(unique_subjects),
```

iii. CONVERSION_NOTES Step 4 states the key finding: "The paper reports 173 sessions = all sessions with good units (174 total − 1 with no good units)." Reaching that conclusion took several iterations. The trajectory (steps 36, 67, 70, 96–99) shows the AI first implemented the data paper's *analysis* session criteria (>65% overall performance, ≥50 correct trials per side), which cut the dataset to 150–155 sessions. It then tested eight variants of the criterion, found that "exactly 173 sessions have good units", and concluded the behavioural criteria "were for specific analyses, not for defining the dataset", removing them. The final rule is therefore: keep every session that has at least one quality-controlled unit and ≥2 usable trials.

## 1-d. How are the data split into trials?

i. Trials come straight from the NWB trials table (`nwb.trials`), one row per behavioural trial; `n_trials = len(trials)` and all per-trial columns (`trial_instruction`, `outcome`, `early_lick`, `auto_water`, `free_water`, `photostim_power`, `photostim_onset`) are read as whole-column arrays. The trial's temporal anchor is the corresponding entry of `BehavioralEvents/go_start_times`, indexed positionally by trial number, i.e. the code assumes exactly one go cue per trials-table row. A boolean `valid_trial_mask` over these rows selects the trials that are emitted (see 1-e), and `valid_trial_indices` is used to index both the trials table and `go_times` consistently.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
trial_instruction = trials['trial_instruction'][:]
outcome = trials['outcome'][:]
early_lick = trials['early_lick'][:]
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
photostim_power_str = trials['photostim_power'][:]

be = nwb.acquisition['BehavioralEvents']
go_times = be.time_series['go_start_times'].timestamps[:]
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
```

```python
valid_trial_indices = np.where(valid_trial_mask)[0]
n_valid_trials = len(valid_trial_indices)
...
valid_go_times = go_times[valid_trial_indices]
```

iii. From the trajectory (step 15): the AI explicitly checked event counts and found `go_start_times` has exactly 368 events for 368 trials, whereas `sample_start_times` (405) and `delay_start_times` (395) have *more* than one entry per trial "because of replayed trials due to early licks". It therefore treats `go_start_times` as the one event stream that maps 1:1 onto trials-table rows and uses positional indexing. The trials table itself is taken as the authoritative trial definition, so no boundaries are re-derived from event streams.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both aimed at excluding trials that lack valid neural data, and no behavioural quality filter:

1. **Water-delivery trials**: rows with `auto_water != 0` or `free_water != 0` are dropped.
2. **Recording coverage**: the recording span is taken from `units/obs_intervals` of the first good unit as `[obs[0,0], obs[-1,1]]`, and a trial is kept only if its *entire* analysis window `[go − 2.5 s, go + 1.5 s]` falls inside that span.

A session with fewer than 2 surviving trials is dropped entirely. Early-lick trials, `ignore` (no-response) trials, and photostimulation trials are all deliberately *kept*, because they are required decoder outputs/inputs. 89,532 trials survive across 173 sessions (mean 517.5/session, range 160–796).

ii.
```python
def get_recording_time_range(units, good_indices):
    """Get the recording time range from obs_intervals of good units."""
    obs = np.array(units['obs_intervals'][good_indices[0]])
    min_time = obs[0, 0]
    max_time = obs[-1, 1]
    return min_time, max_time
```

```python
# === Filter trials for data extraction ===
# 1. Exclude auto_water and free_water
# 2. Exclude trials outside recording range
valid_trial_mask = (auto_water == 0) & (free_water == 0)
for i in range(n_trials):
    if valid_trial_mask[i]:
        window_start = go_times[i] + ALIGN_START
        window_end = go_times[i] + ALIGN_END
        if window_start < rec_min or window_end > rec_max:
            valid_trial_mask[i] = False

valid_trial_indices = np.where(valid_trial_mask)[0]
n_valid_trials = len(valid_trial_indices)

if n_valid_trials < 2:
    print(f"  Session {session_id}: Only {n_valid_trials} valid trials within recording, skipping")
    return None
```

iii. The `auto_water`/`free_water` exclusion is taken from the reference code: CONVERSION_NOTES Step 1 records `get_regular_trial_mask` (in `functions_for_r2.py`) as the curation function, described as "Filter: no early lick, auto water, free water, no-response, stim", and Step 3 states the decoder-specific rule "exclude auto_water and free_water" — i.e. the AI deliberately kept the early-lick / no-response / stim trials that the reference code drops, because those are required decoder outputs, but retained the water-delivery exclusions.

The recording-coverage filter was added in response to a bug the AI found itself. The trajectory (steps 51–57) shows the first full run produced warnings that 321 of 480 trials in one session had all-zero neural data; investigation showed "the `obs_intervals` has only 160 intervals ... the last observation interval ends at 1114.47, but the go cue times extend to 3707.74 ... the recording only covers the first 160 trials." The AI also verified that "all good units in this session have the same obs_intervals", which is why reading the interval list from `good_indices[0]` alone is sufficient. It requires the full window (not just the go cue) inside the span so that no partially-unobserved window is emitted. The 2-trial minimum comes from the target-format requirement.

CONVERSION_NOTES Step 10 documents the one residual case that survives this filter: "Session 0, trial 159: Last trial of partially recorded session. obs_intervals indicate recording extends to this trial, but spike times end slightly before the trial window. Valid data — neurons simply had no spikes."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (session-absolute seconds, read per unit), restricted to units with `units/classification == 'good'`. The second ingredient is `BehavioralEvents/go_start_times`, which supplies the per-trial anchor for the bin edges. `units/obs_intervals` supplies the recording span used to filter trials, and `units/anno_name` supplies the per-neuron brain-region label.

ii.
```python
units = nwb.units
classification = units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)
...
anno_names = units['anno_name'][:]
good_anno_names = anno_names[good_indices]

# Read spike times for good units
good_spike_times = []
for idx in good_indices:
    st = units['spike_times'][idx]
    good_spike_times.append(np.sort(st))
```

```python
go_times = be.time_series['go_start_times'].timestamps[:]
...
valid_go_times = go_times[valid_trial_indices]
neural_trials = compute_firing_rates_vectorized(
    good_spike_times, valid_go_times, n_good)
```

iii. `spike_times` is the only neural representation in the NWB files — CONVERSION_NOTES Step 2 lists the unit fields as "spike_times, classification, anno_name, obs_intervals". The reference code's preprocessing (`sliding_histogram` in `preprocessing_DJ_2022Aug.py`, recorded in Step 1 as "Firing rates from spike times") likewise builds firing rates from spike times, so the AI used the same source variable.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin **firing rates in Hz**. For every valid trial, 81 absolute bin edges are formed as `go_time + (−2.5 + 0.05·k)`; for every good unit the spikes inside the window are located with two `searchsorted` calls, histogrammed against those edges with `np.histogram`, and the counts are divided by the 50 ms bin width. Units with no spikes in the window are left as a row of zeros. The result is one `(n_neurons, 80)` `float32` array per trial. No smoothing, no sliding/overlapping windows, no normalisation, and no baseline subtraction.

ii.
```python
def compute_firing_rates_vectorized(spike_times_list, go_cue_times, n_neurons,
                                     bin_width=BIN_WIDTH, align_start=ALIGN_START,
                                     align_end=ALIGN_END):
    """Compute firing rates for all neurons across all trials."""
    n_trials = len(go_cue_times)
    n_bins = int((align_end - align_start) / bin_width)
    bin_offsets = align_start + np.arange(n_bins + 1) * bin_width

    fr_all = []
    for t in range(n_trials):
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        bin_edges = go_cue_times[t] + bin_offsets

        for i, spikes in enumerate(spike_times_list):
            if len(spikes) == 0:
                continue
            left = np.searchsorted(spikes, bin_edges[0])
            right = np.searchsorted(spikes, bin_edges[-1])
            if left >= right:
                continue
            spikes_window = spikes[left:right]
            counts, _ = np.histogram(spikes_window, bins=bin_edges)
            fr[i] = counts / bin_width

        fr_all.append(fr)

    return fr_all
```

iii. CONVERSION_NOTES Step 5 states the mapping as "units/spike_times (good) → neural; 50 ms bin firing rates, −2.5 to +1.5 s from go cue". Step 1/Step 10 record the deliberate departure from the reference preprocessing: "Binning: 50 ms (task) vs 100 ms/50 ms stride (reference)", and the trajectory (step 28) states it explicitly: "the task specifies 50 ms bin WIDTH. The reference code uses 100 ms bin width with 50 ms stride. I need to use 50 ms bins as specified in the task." Rate (rather than count) units match the reference `sliding_histogram(..., rate=True)`. CONVERSION_NOTES Step 10 reports the sanity check: "Neural: Spot-checked firing rates against raw NWB spike times — exact match (np.allclose=True)"; I reproduced this independently on session `SC016_20190211_143614_s1`, trial 5, neuron 3 (`np.allclose` → True).

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single criterion: keep only units whose `units/classification` string equals `'good'`. No thresholds are applied to any individual quality metric (amplitude, ISI violation, presence ratio, etc.), and `units/unit_quality` (the older `'good'`/`'multi'` Phy label) is not used. A session with zero good units returns `None` and is dropped. This retains 69,453 units across 173 sessions (mean 401.5/session, range 90–923).

ii.
```python
units = nwb.units
classification = units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)

if n_good == 0:
    print(f"  Session {session_id}: No good units, skipping")
    return None
```

iii. CONVERSION_NOTES Step 3 gives the curation rule as "**Neuron**: classification == 'good'", cross-referenced to the methods statement that 25.9% of Kilosort2 clusters are labelled good, totalling 69,943 units over 173 sessions. Step 1 notes the reference code's "QC: 'classifier' mode — region-specific classifiers", and Step 10 records the comparison: "Neuron filtering: classification=='good' matches classifier QC mode". The trajectory (step 13) shows the AI distinguished the two available labels — "classification: 'good' vs 'unlabelled' — this is the QC classifier result ... unit_quality: 'good' vs 'multi' — this is the Phy manual curation result" — and chose the classifier verdict. CONVERSION_NOTES Step 9 reports the resulting count against the paper: 69,453 vs 69,943, "~99.3%", attributed to partial recordings.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go cue onset**, taken from `BehavioralEvents/go_start_times`. Spike times, event timestamps and camera timestamps all live on the same session-absolute clock, so no resampling, interpolation or per-stream offset correction is needed: the fixed relative edge grid is simply added to each trial's go-cue time to give the absolute window, and the spikes are binned against those absolute edges.

ii.
```python
go_times = be.time_series['go_start_times'].timestamps[:]
...
valid_go_times = go_times[valid_trial_indices]
neural_trials = compute_firing_rates_vectorized(
    good_spike_times, valid_go_times, n_good)
```

```python
bin_offsets = align_start + np.arange(n_bins + 1) * bin_width
for t in range(n_trials):
    bin_edges = go_cue_times[t] + bin_offsets
```

```python
'temporal_alignment_event': 'Go cue onset',
'off_start': ALIGN_START,   # -2.5
'off_end': ALIGN_END,       #  1.5
```

iii. The Decoder Task section of the instructions specifies go-cue alignment, and CONVERSION_NOTES Step 10 records that this also matches the reference: "Temporal alignment: Go cue onset (same)". The trajectory (steps 6, 16, 33) shows the AI verified this empirically: `preprocessing_utils.py` defines its epochs relative to the go cue (sample −1.9 to −1.2, delay −1.2 to 0, post_go 0 to 1.0); delay start is "exactly −1.2 s before go cue, consistent with methods: 1.2 s delay epoch"; and "Spike times are in absolute session time (same timebase as trial start/stop and go cue times)".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins. The window −2.5 s to +1.5 s around the go cue gives exactly `N_TIMEBINS = 80` bins per trial, defined once at module level as 81 edge offsets / 80 centres and reused for every trial, session, and data stream (neural, inputs, tongue output). Every trial in the converted dataset therefore has exactly 80 timepoints (verification output: "T: mean 80.00, median 80.00, min 80, max 80"). There is no rebinning of an already-binned intermediate: spikes are binned once, directly at 50 ms, from raw spike times. `metadata['time_bin_size']` is recorded as 50.0 ms and `metadata['bin_centers']` stores the grid.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins
ALIGN_START = -2.5  # seconds before go cue
ALIGN_END = 1.5    # seconds after go cue
N_TIMEBINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
BIN_CENTERS = ALIGN_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
```

```python
'time_bin_size': BIN_WIDTH * 1000,
'temporal_alignment_event': 'Go cue onset',
'off_start': ALIGN_START,
'off_end': ALIGN_END,
...
'bin_centers': BIN_CENTERS.tolist(),
```

iii. The 50 ms width and the −2.5/+1.5 s window are set by the instructions. CONVERSION_NOTES Step 5 lists "**Bin width**: 50 ms (task requirement)" as key decision 1, and Step 10 records the deliberate deviation from the reference preprocessing (100 ms width, 50 ms stride), which would have produced overlapping bins and correlated adjacent timepoints. Binning directly from spike times avoids a two-stage rebinning.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the onsets of the auditory sample/tone epoch) together with the trial's go-cue time from `go_start_times`. The tone assigned to a trial is the **last** `sample_start_times` entry at or before that trial's go cue (with a 10 ms tolerance). If no such entry exists, a fallback of `go_time − 1.85 s` (the session-mean tone-to-go interval) is used.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
...
prev_samples = sample_start_times[sample_start_times < go_time + 0.01]
if len(prev_samples) > 0:
    tone_onset = prev_samples[-1]
else:
    tone_onset = go_time - 1.85
```

iii. The trajectory (steps 15–16) shows why the *last* preceding onset is used rather than a positional index: `sample_start_times` has 405 entries for 368 trials because "an early lick replays the sample epoch", so a trial can carry more than one tone and only the final one is the tone the animal actually acted on. The AI also verified the expected geometry: "Tone (sample) onset is ~−1.85 s before go cue (mean) ... The sample epoch has 3 tones of 150 ms with 100 ms inter-tone intervals = 650 ms total. Sample onset at ~−1.85 s, delay at −1.2 s, so sample duration ~0.65 s". CONVERSION_NOTES Step 5 records the mapping "sample_start_times → input[0]: Time from tone onset (s), continuous".

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A per-trial constant offset applied to the shared bin-centre grid: for each of the 80 bins, the value is the bin's absolute centre time minus the trial's tone onset, i.e. `(go_time + BIN_CENTERS) − tone_onset`. This is a continuous, monotonically increasing, time-varying input stored as `float32` in row 0 of the `(2, 80)` per-trial input array. No clipping, no normalisation, no binarisation. Observed range over the full dataset: [−1.5, 11.9] s (values well above ~3.4 s occur on trials where an early lick replayed the sample epoch, stretching the tone-to-go interval).

ii.
```python
time_from_tone = (go_time + BIN_CENTERS) - tone_onset
...
input_data = np.stack([time_from_tone.astype(np.float32), photostim_binary], axis=0)
input_trials.append(input_data)
```

iii. CONVERSION_NOTES Step 5 lists it as a continuous time-varying input. The trajectory (step 39) shows the AI checked the resulting range against its own prediction and reconciled the discrepancy: "tone onset is ~1.85 s before go cue, so at −2.5 s from go cue, time from tone is about −0.65 ... But the max is 5.3, which suggests some trials have tone onset much earlier (due to early lick replays where the sample was replayed)." The instructions' note that "if an input is a time such as onset of some stimulus, represent it as a binary time series" was read as applying to event-marker inputs; the Decoder Task explicitly labels this one "continuous, time-varying", which is how it is encoded.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction, on the same grid. `BIN_CENTERS` are the centres of exactly the same 80 bins whose edges are used to histogram the spikes, both anchored to the same `go_time`. Bin *k* of the input therefore describes the same 50 ms interval as bin *k* of the firing-rate matrix, with no separate alignment, interpolation, or offset step.

ii.
```python
BIN_CENTERS = ALIGN_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
```

```python
# neural
bin_offsets = align_start + np.arange(n_bins + 1) * bin_width
bin_edges = go_cue_times[t] + bin_offsets

# input
time_from_tone = (go_time + BIN_CENTERS) - tone_onset
```

iii. N/A — alignment is inherited from the shared go-cue-anchored grid; the `--show-processing` plots in `process_session` overlay the input against the same `BIN_CENTERS` axis with a go-cue marker at 0, which the AI used to confirm visually that there are no temporal misalignments.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Two sources, used together. The *set* of stimulated trials comes from the trials table column `photostim_power` (the string `'N/A'` on unstimulated trials, a numeric string such as `'5.500'` otherwise). The *timing* comes from the recorded event streams `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`. The go-cue time defines the window the events are compared against. If the session has no `photostim_start_times` series, the input is all zeros.

ii.
```python
photostim_power_str = trials['photostim_power'][:]
...
has_photostim_events = 'photostim_start_times' in be.time_series
if has_photostim_events:
    photostim_event_starts = be.time_series['photostim_start_times'].timestamps[:]
    photostim_event_stops = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The trajectory (steps 11, 15, 17) shows the AI inspected both representations. It found the trials-table fields are strings relative to trial start — "photostim_onset in trials table is relative to trial_start (not go cue)" — and that the event stream carries absolute timestamps on the same clock as everything else. It verified the geometry against the methods: "Photostim starts at −1.2 s relative to go cue (= delay start) with 0.5 s duration, so photostim is from −1.2 s to −0.7 s relative to go cue (last 0.5 s of delay, consistent with methods)", and noted "Some photostim events start earlier (up to −2.25 s relative to go), possibly on early-lick replay trials". Using the absolute event timestamps avoids the string-parsing and trial-start arithmetic needed for the trials-table version. CONVERSION_NOTES Step 5 records "photostim events → input[1]: Binary on/off, time-varying".

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) `float32` time series, row 1 of the per-trial input array. It starts as 80 zeros. If the trial's `photostim_power` is not `'N/A'`, the code scans every photostim event in the session, skips those that do not overlap the trial's `[go − 2.5, go + 1.5]` window, and for each surviving event marks every bin whose **interval overlaps** the event interval (`bin_start < ps_stop and bin_end > ps_start`). Unstimulated trials stay all-zero. I verified on session `SC016_20190211_143614_s1` that the set of trials with a non-zero photostim input exactly matches the set with `photostim_power != 'N/A'` (77/520 trials, 100% agreement), and that a representative stimulation of −1.2005 s to −0.7005 s relative to the go cue marks bins 25–35.

ii.
```python
photostim_binary = np.zeros(N_TIMEBINS, dtype=np.float32)
if has_photostim_events and photostim_power_str[trial_idx] != 'N/A':
    trial_window_start = go_time + ALIGN_START
    trial_window_end = go_time + ALIGN_END
    for ps_idx in range(len(photostim_event_starts)):
        ps_start = photostim_event_starts[ps_idx]
        ps_stop = photostim_event_stops[ps_idx]
        if ps_stop < trial_window_start or ps_start > trial_window_end:
            continue
        for b in range(N_TIMEBINS):
            bin_start_abs = go_time + ALIGN_START + b * BIN_WIDTH
            bin_end_abs = bin_start_abs + BIN_WIDTH
            if bin_start_abs < ps_stop and bin_end_abs > ps_start:
                photostim_binary[b] = 1.0
```

iii. The Decoder Task requires "whether photostimulation is on at every time point (discrete, time-varying)", so it is encoded as a per-bin binary series rather than a per-trial flag. CONVERSION_NOTES Step 5 records this. Gating on `photostim_power != 'N/A'` before consulting the events means the trials-table flag is the authority on *which* trials were stimulated and the event stream only supplies *when* — a belt-and-braces combination that the AI's plots (`axes[1,0]`, "Input: Photostim (N trials with stim)") were used to inspect. Any-overlap (rather than centre-membership) is the inclusive reading of "photostimulation is on" during a bin.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Bin boundaries are reconstructed in absolute session time from the same anchor and the same grid as the neural bins (`go_time + ALIGN_START + b * BIN_WIDTH`), and the photostim event timestamps are already in that same absolute clock, so the comparison is direct. Bin *k* of the photostim input covers exactly the interval of bin *k* of the firing-rate matrix.

ii.
```python
trial_window_start = go_time + ALIGN_START
trial_window_end = go_time + ALIGN_END
...
bin_start_abs = go_time + ALIGN_START + b * BIN_WIDTH
bin_end_abs = bin_start_abs + BIN_WIDTH
```

iii. N/A — as with the tone input, alignment follows from using the same go-cue-anchored 50 ms grid; no interpolation or offset correction is applied because the NWB file timestamps every stream on one global clock.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The NWB files contain no explicit "choice" column, so it is derived from two trials-table columns: `trial_instruction` (`'left'`/`'right'`, the side the tone instructed) and `outcome` (`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed side; a miss means it licked the other side; an `ignore` means it did not lick.

ii.
```python
trial_instruction = trials['trial_instruction'][:]
outcome = trials['outcome'][:]
...
choice = determine_choice(trial_instruction[trial_idx], outcome[trial_idx])
```

```python
def determine_choice(trial_instruction, outcome):
    """Determine lick direction choice.
    Returns: 0=left, 1=right, 2=no_lick
    """
    if outcome == 'ignore':
        return 2  # no lick
    elif outcome == 'hit':
        return 0 if trial_instruction == 'left' else 1
    elif outcome == 'miss':
        return 1 if trial_instruction == 'left' else 0
    return 2
```

iii. The trajectory (step 11) records the AI enumerating the trials-table columns and their value sets — "trial_instruction: 'left' or 'right' (the correct answer); outcome: 'hit', 'miss', 'ignore'" — and step 34 lists "How to determine choice (lick direction) from the NWB data" as an open question it resolved before writing the script. The instruction × outcome cross fully determines the licked side, so no lick-time stream is needed. CONVERSION_NOTES Step 5 records "instruction + outcome → output[0]: Choice: L(0), R(1), no_lick(2)".

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded 0 = left, 1 = right, 2 = no lick, per trial, then broadcast unchanged across all 80 time bins via `np.repeat` so that all four outputs share one `(4, 80)` `int64` array per trial. `output_values[0]` names the three codes `['left', 'right', 'no_lick']`. The full-dataset distribution is left 0.429, right 0.422, no_lick 0.148.

ii.
```python
output_per_trial.append(np.array([choice, outcome_val, early_lick_val], dtype=np.int64))
...
per_trial = output_per_trial[i]
per_trial_expanded = np.repeat(per_trial[:, np.newaxis], N_TIMEBINS, axis=1)
tongue_expanded = tongue_discrete[np.newaxis, :]
output_trials.append(
    np.concatenate([per_trial_expanded, tongue_expanded], axis=0).astype(np.int64)
)
```

```python
'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
'output_values': [
    ['left', 'right', 'no_lick'],
    ...
],
```

iii. The Decoder Task specifies choice as a per-trial output with values left/right/no lick; the target format requires `(n_output, n_timepoints)` arrays and prefers time-varying encodings, so the per-trial value is tiled across bins to keep all outputs in one array of uniform shape. The AI's `--show-processing` plots include a choice-distribution bar chart (`axes[1,1]`) used to confirm the classes are populated and roughly balanced between left and right.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already stores exactly the three strings the instructions ask for: `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome = trials['outcome'][:]
```

iii. No derivation is needed — the trials table stores the outcome explicitly with matching categories. The trajectory (step 11) confirms the AI enumerated the value set before relying on it.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps the strings to `0` ignore, `1` miss, `2` hit (with a `.get(..., 0)` default), and the value is written into row 1 of the per-trial output array, repeated across all 80 bins alongside the other per-trial outputs. `output_values[1] = ['ignore', 'miss', 'hit']`. Full-dataset distribution: ignore 0.148, miss 0.167, hit 0.685 — i.e. a hit rate of 68.5% overall, or ~80% among trials where the animal responded.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome[trial_idx], 0)
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
output_per_trial.append(np.array([choice, outcome_val, early_lick_val], dtype=np.int64))
```

iii. The code assignment follows the ordering given in the Decoder Task ("ignore, miss, hit"). The trajectory (steps 37, 40) shows the AI separately computed a behavioural "correct rate" from the same column to cross-check against the paper's reported 84% mean performance, reasoning that "the correct rate should be hit / (hit + miss), excluding both early lick AND ignore/no-response trials from the denominator" — the sample session gave 81.1% against the paper's 84%. That statistic is only reported, not used for filtering in the final script.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The trials-table `early_lick` column, which holds the strings `'early'` and `'no early'`.

ii.
```python
early_lick = trials['early_lick'][:]
```

iii. The trials table flags early licking explicitly (trajectory step 11: "early_lick: 'early' or 'no early'"), so no derivation from lick-time streams is needed. The lick that sets the flag occurs during the sample or delay epoch, i.e. before the go cue and inside the −2.5 s window, so the event is within the extracted data even though the flag itself is per-trial.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `1` if the string equals `'early'` and `0` otherwise, written into row 2 of the output array and repeated across all 80 bins. `output_values[2] = ['no', 'yes']`. Full-dataset distribution: no 0.884, yes 0.116.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

```python
'output_values': [
    ...
    ['no', 'yes'],
    ...
],
```

iii. The code assignment follows the instructions ("no, yes"). One value per trial, repeated across bins like the other per-trial outputs so all four outputs share a single `(4, 80)` array. Note that early-lick trials are deliberately retained rather than filtered (contrary to the reference code's `get_regular_trial_mask` and the data paper's "early lick trials ... were excluded for analysis") precisely because early lick is a required decoder output — CONVERSION_NOTES Step 1 flags that the reference exclusion applies to "main analysis only".

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, a `(n_frames, 3)` DeepLabCut array of `(tongue_x, tongue_y, tongue_likelihood)` sampled at ~300 Hz with matching `timestamps`. Column 1 (`tongue_y`) supplies the value and column 2 (`tongue_likelihood`) decides whether the tongue is visible in a frame. Presence of the series is checked first; if absent the output is all "not visible".

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
has_tongue = 'Camera0_side_TongueTracking' in bts.time_series
if has_tongue:
    tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
    tongue_data_all = tongue_ts_obj.data[:]
    tongue_ts_all = tongue_ts_obj.timestamps[:]
```

```python
frames = data_window[mask]
visible = frames[:, 2] >= TONGUE_LIKELIHOOD_THRESHOLD
visible_binned[b] = visible.mean()
if visible.sum() > 0:
    tongue_y_binned[b] = frames[visible, 1].mean()
```

iii. The trajectory (step 14) records the AI reading the channel layout off the file rather than assuming it: "Camera0_side_TongueTracking: (680500, 3) — tongue_x, tongue_y, tongue_likelihood ... at ~300 Hz", alongside the jaw and nose series which it correctly did not use. This is the only tongue measurement in the file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps, all per session:

1. For each trial, the camera frames inside `[go − 2.5, go + 1.5]` are located with `searchsorted` and assigned to one of the 80 bins with `np.digitize`.
2. Within each bin, a frame counts as showing the tongue if `likelihood >= 0.9`. The bin records (a) `visible_binned[b]` = the *fraction* of its frames that are visible, and (b) `tongue_y_binned[b]` = the mean `tongue_y` over just those visible frames.
3. A bin is treated as a genuine tongue observation only if that fraction is **≥ 0.5**. The `tongue_y` values of all such bins, pooled across every trial of the session, form the distribution from which the 40th and 60th percentiles are computed.
4. Each bin is then digitised against those two per-session edges; bins failing the ≥ 0.5 visibility test are assigned class 3.

Across the full dataset 84.0% of bins are "not visible"; within the visible bins the split is exactly 40/20/40 (0.064 / 0.032 / 0.064 of all bins), confirming the percentile logic is self-consistent. On the session I checked, 16.2% of raw frames pass the 0.9 likelihood threshold.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9  # DLC confidence threshold
```

```python
def get_tongue_y_for_trial(tongue_ts, tongue_data, go_cue_time, ...):
    bin_edges = go_cue_time + align_start + np.arange(n_bins + 1) * bin_width
    tongue_y_binned = np.full(n_bins, np.nan, dtype=np.float32)
    visible_binned = np.zeros(n_bins, dtype=np.float32)
    left_idx = np.searchsorted(tongue_ts, bin_edges[0])
    right_idx = np.searchsorted(tongue_ts, bin_edges[-1])
    if left_idx >= right_idx:
        return tongue_y_binned, visible_binned
    ts_window = tongue_ts[left_idx:right_idx]
    data_window = tongue_data[left_idx:right_idx]
    bin_idx = np.digitize(ts_window, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        frames = data_window[mask]
        visible = frames[:, 2] >= TONGUE_LIKELIHOOD_THRESHOLD
        visible_binned[b] = visible.mean()
        if visible.sum() > 0:
            tongue_y_binned[b] = frames[visible, 1].mean()
    return tongue_y_binned, visible_binned
```

```python
visible_mask = tongue_vis >= 0.5
if visible_mask.sum() > 0:
    tongue_y_all_visible.extend(tongue_y[visible_mask].tolist())
...
if len(tongue_y_all_visible) > 0:
    tongue_y_arr = np.array(tongue_y_all_visible)
    p40 = np.percentile(tongue_y_arr, 40)
    p60 = np.percentile(tongue_y_arr, 60)
else:
    p40 = 0
    p60 = 0
```

iii. CONVERSION_NOTES Step 5 lists key decision 4, "**Tongue visibility**: 0.9 DLC likelihood threshold", and the mapping "tongue_y → output[3]: 0–3 categories, time-varying". The DLC tracker still emits a coordinate when the tongue is retracted, so low-likelihood frames must be excluded rather than averaged in; the AI chose a conservative 0.9 confidence. The trajectory (step 39) shows it inspected the resulting class balance and reasoned about it: "Tongue y position: 83.2% not visible — this seems very high. This could be because the tongue is only visible during licking, which happens mainly after the go cue." Percentiles are taken over the 50 ms bin means (the same quantity that gets discretised) rather than over raw frames, and their scope is per session, as the instructions specify.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Exactly the four-way scheme in the instructions, with per-session edges: `0` if `y < p40`, `1` if `p40 <= y <= p60`, `2` if `y > p60`, and `3` for any bin that is not a valid tongue observation (no frames at all, fewer than half its frames above the likelihood threshold, or no tongue series in the session). The array is initialised to 3 and only visible bins are overwritten. `output_values[3] = ['below_40th', '40th_to_60th', 'above_60th', 'not_visible']`.

ii.
```python
tongue_discrete = np.full(N_TIMEBINS, 3, dtype=np.int64)
visible_mask = tongue_vis >= 0.5
if visible_mask.sum() > 0 and len(tongue_y_all_visible) > 0:
    ty = tongue_y[visible_mask]
    tongue_discrete[visible_mask] = np.where(ty < p40, 0,
                                             np.where(ty <= p60, 1, 2))
```

```python
'output_values': [
    ...
    ['below_40th', '40th_to_60th', 'above_60th', 'not_visible'],
],
```

iii. The three visible classes and the 40th/60th per-session percentile edges are taken verbatim from the Decoder Task; class 3 ("not visible") is the fourth category the instructions define. The AI validated the discretisation visually via the `--show-processing` plot (`axes[3,0]`, y-ticks labelled `<40th / 40-60th / >60th / not visible`) and numerically via the verification output, where the visible classes come out in the expected 40/20/40 proportion.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. This is the only genuinely time-varying output, so it is the only one requiring real alignment. The camera timestamps share the session-absolute clock with the spikes and the go cues, so the per-trial frame range is found with `searchsorted` on `tongue_ts` at the same absolute window edges used for the spikes, and frames are assigned to bins by `np.digitize` against the *same* `go_cue_time + ALIGN_START + k·BIN_WIDTH` edge array. Bin *k* of the tongue output therefore covers the same interval as bin *k* of the firing rates. No interpolation or resampling. Frames falling exactly on the final edge are clipped into the last bin.

ii.
```python
bin_edges = go_cue_time + align_start + np.arange(n_bins + 1) * bin_width
...
left_idx = np.searchsorted(tongue_ts, bin_edges[0])
right_idx = np.searchsorted(tongue_ts, bin_edges[-1])
...
bin_idx = np.digitize(ts_window, bin_edges) - 1
bin_idx = np.clip(bin_idx, 0, n_bins - 1)
```

iii. The AI established early (trajectory step 33) that all streams share one clock — "Spike times are in absolute session time (same timebase as trial start/stop and go cue times)" — so reusing the same edge construction is sufficient and guarantees synchrony. The `--show-processing` panels plot the discretised tongue trace against the same `BIN_CENTERS` axis with the go cue marked at 0, which the AI used to confirm the tongue signal rises after the go cue as expected. The video is trial-gated (off during the inter-trial interval), so on trials where the go cue falls less than 2.5 s after trial start the leading bins simply contain no frames and fall into class 3.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six distinct cases, each handled explicitly:

- **Session never quality-controlled** (`classification` is NaN for all units): `classification == 'good'` yields no matches, `n_good == 0`, and the session is dropped with a printed message. This drops exactly one session (174 → 173).
- **Trials outside the ephys recording** (`obs_intervals` covers fewer trials than the behaviour, in some sessions only 160 of 480): excluded by the recording-coverage filter (1-e), which was added specifically to fix all-zero neural trials found in the first full run.
- **Water-delivery trials** (`auto_water`, `free_water`): excluded.
- **Session with too few surviving trials** (< 2): dropped, since the target format requires at least two trials per session to evaluate the decoder.
- **No tone onset before the go cue**: falls back to `go_time − 1.85 s`, the session-typical tone-to-go interval.
- **Missing or untracked tongue** (no `Camera0_side_TongueTracking` series; a bin with no camera frames; a bin whose frames are mostly below the likelihood threshold): represented as the explicit "not visible" category rather than imputed or dropped. If a whole session yields no visible bins, `p40 = p60 = 0` and every bin is class 3.
- **Residual all-zero neural trials** (2 of 89,532): kept, and documented as real data rather than an error.

ii.
```python
if n_good == 0:
    print(f"  Session {session_id}: No good units, skipping")
    return None
...
if n_valid_trials < 2:
    print(f"  Session {session_id}: Only {n_valid_trials} valid trials within recording, skipping")
    return None
```

```python
prev_samples = sample_start_times[sample_start_times < go_time + 0.01]
if len(prev_samples) > 0:
    tone_onset = prev_samples[-1]
else:
    tone_onset = go_time - 1.85
```

```python
if has_tongue:
    tongue_y, tongue_vis = get_tongue_y_for_trial(...)
    ...
else:
    tongue_y_per_trial.append(np.full(N_TIMEBINS, np.nan))
    tongue_visible_per_trial.append(np.zeros(N_TIMEBINS))
```

```python
if len(tongue_y_all_visible) > 0:
    ...
else:
    p40 = 0
    p60 = 0
```

iii. The governing principle is visible in the trajectory: where *nothing was recorded*, the trial or session is excluded rather than emitted as fabricated zeros (steps 51–57: "the recording only covered 160 trials, and the remaining 320 trials have no neural data ... I need to use obs_intervals to determine which trials have valid neural data"); where a measurement *legitimately has no value*, as with a retracted tongue, it is given an explicit category. For the two residual all-zero trials the AI investigated and then justified keeping them (CONVERSION_NOTES Step 10): "obs_intervals indicate recording extends to this trial, but spike times end slightly before the trial window. Valid data — neurons simply had no spikes." It explicitly weighed the alternative and rejected it as not worth the loss: "One trial with zero data out of 78,111 is negligible."

## 10-a. What are the most time-consuming steps of the code?

i. The script instruments each stage with `time.time()` and prints per-stage timings, so the profile is directly measurable from `conversion_full_out.txt` (173 sessions, 594.8 s total, ~3.4 s/session):

| Stage | Total | Share | Mean/session |
|---|---|---|---|
| `compute_firing_rates_vectorized` | 409.5 s | 69% | 2.37 s |
| I/O construction (dominated by the per-bin tongue loop) | 80.7 s | 14% | 0.47 s |
| Reading `spike_times` unit-by-unit + `np.sort` | 61.0 s | 10% | 0.35 s |
| Reading the tongue tracking array | 5.9 s | 1% | 0.03 s |

Pickling the ~11.9 GB result adds further time on top of the 594.8 s. The dominant cost is the `n_trials × n_neurons` Python loop in the firing-rate routine, which issues roughly 520 × 526 ≈ 273,000 `np.histogram` calls per session.

ii.
```python
t_fr_start = time.time()
valid_go_times = go_times[valid_trial_indices]
neural_trials = compute_firing_rates_vectorized(
    good_spike_times, valid_go_times, n_good)
t_fr_end = time.time()
print(f"    Computing firing rates: {t_fr_end - t_fr_start:.1f}s")
```

```python
print(f"    Reading units: {t_units_end - t_units_start:.1f}s")
print(f"    Reading tongue: {t_tongue_end - t_tongue_start:.1f}s")
print(f"    Processing I/O: {t_io_end - t_io_start:.1f}s")
```

iii. The instructions required printing timing information and optimising if the full run would exceed 15 minutes. The trajectory (step 36) shows the AI hit exactly this: the first implementation took "13.1 s for one session ... For 174 sessions, that would be ~2282 s (~38 min). I need to optimize the code." It then rewrote the spike reading and firing-rate computation, bringing the estimate to "~4.6 s per session ... ~800 s (~13 minutes), which is under 15 minutes" (step 38) and concluded "no optimization needed" (step 40). The final run came in at 594.8 s. CONVERSION_NOTES Step 6 records only the headline "~3.5 s per session, ~10 min total"; the "Code inefficiencies identified / Code speedups added" subsections of the template were not filled in.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Despite its name, `compute_firing_rates_vectorized` contains no vectorisation across the trial or neuron axis — it is a plain nested Python loop. Five loops could be collapsed:

1. **Firing rates (`for t in range(n_trials)` × `for i, spikes in ...`)** — the largest win by far (69% of runtime). Flattening all 81 edges × all trials into one array and issuing a single `np.searchsorted(spikes, edges)` per unit, then `np.diff`, removes the trial loop entirely and replaces ~273,000 `np.histogram` calls per session with ~400 `searchsorted` calls.
2. **Tongue per-bin loop (`for b in range(n_bins)` with a boolean `mask = bin_idx == b`)** — an O(n_bins × n_frames) scan that `np.bincount(bin_idx, weights=...)` computes in one pass.
3. **Photostim (`for ps_idx ...` × `for b in range(N_TIMEBINS)`)** — a nested scan over every session event × every bin, per trial; a single vectorised interval comparison against the 80 bin edges would replace it.
4. **Trial coverage filter (`for i in range(n_trials)` checking window bounds)** — a pure elementwise comparison, directly expressible as `(go_times + ALIGN_START >= rec_min) & (go_times + ALIGN_END <= rec_max)`.
5. **Assembly index lookups (`unique_subjects.index(s)`, `all_region_names.index(r)`)** — linear scans inside loops over 173 sessions × up to 923 neurons against a 293-element region list; a dict lookup is O(1).

Additionally, the per-unit spike read `units['spike_times'][idx]` issues one ragged-array read per unit instead of pulling the flat buffer and its offsets once.

ii.
```python
    fr_all = []
    for t in range(n_trials):
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        bin_edges = go_cue_times[t] + bin_offsets

        for i, spikes in enumerate(spike_times_list):
            ...
            counts, _ = np.histogram(spikes_window, bins=bin_edges)
            fr[i] = counts / bin_width
```

```python
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        frames = data_window[mask]
```

```python
    for ps_idx in range(len(photostim_event_starts)):
        ...
        for b in range(N_TIMEBINS):
```

```python
for i in range(n_trials):
    if valid_trial_mask[i]:
        window_start = go_times[i] + ALIGN_START
        ...
```

```python
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
...
idx = np.array([all_region_names.index(r) for r in regions])
```

iii. The AI's justification for stopping where it did is the 15-minute budget in the instructions: having measured ~4.6 s/session in the sample run it concluded the full run would fit and did not optimise further (trajectory steps 38, 40). That reasoning is sound with respect to the stated constraint, but the notes never identify these loops as remaining inefficiencies, and the function name `compute_firing_rates_vectorized` asserts a vectorisation that the body does not perform.

## 10-c. What processing does the code repeat multiple times?

i. Several quantities are recomputed that could be computed once:

- **Per-(trial, neuron) window search.** `np.searchsorted(spikes, bin_edges[0])` / `[-1]` are executed once per trial *per neuron*, and `np.histogram` then performs its own internal search over the same edges — so each unit's spike array is searched ~520 times per session instead of once.
- **`bin_edges` reconstruction.** Rebuilt from `bin_offsets` on every trial in the firing-rate routine, and again from scratch on every trial in `get_tongue_y_for_trial`; the photostim block rebuilds each bin's boundaries a third time inside its innermost loop (`bin_start_abs = go_time + ALIGN_START + b * BIN_WIDTH`).
- **`np.sort(st)` on every unit's spike times**, which NWB already stores sorted.
- **Two passes over the trial list for the tongue output**: one loop computes and stores `tongue_y_per_trial` / `tongue_visible_per_trial`, then after the percentiles are known a second loop re-walks the same trials to discretise them. (This one is genuinely necessary — the percentile edges are per session, so they cannot be known during the first pass.)
- **List `.index()` lookups** in the assembly step re-scan `unique_subjects` and `all_region_names` for every session and every neuron.

Each NWB file is opened exactly once and each raw array (`spike_times`, tongue data, trial columns) is read only once, so there is no repeated file I/O.

ii.
```python
    for t in range(n_trials):
        bin_edges = go_cue_times[t] + bin_offsets
        for i, spikes in enumerate(spike_times_list):
            left = np.searchsorted(spikes, bin_edges[0])
            right = np.searchsorted(spikes, bin_edges[-1])
            ...
            counts, _ = np.histogram(spikes_window, bins=bin_edges)
```

```python
        for idx in good_indices:
            st = units['spike_times'][idx]
            good_spike_times.append(np.sort(st))
```

```python
            for b in range(N_TIMEBINS):
                bin_start_abs = go_time + ALIGN_START + b * BIN_WIDTH
                bin_end_abs = bin_start_abs + BIN_WIDTH
```

iii. The notes do not discuss this. The implicit justification is again the runtime budget: the redundancy is the main reason the conversion takes 595 s rather than the ~250 s an edge-grid-once implementation achieves, but it still finishes comfortably inside the 15-minute limit the instructions set, and none of it changes the output values. The two-pass tongue structure is a deliberate consequence of the per-session percentile requirement rather than an oversight.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A handful of vestigial computations survive in the final script, all cheap but none reaching the output file:

- **The behavioural performance statistic.** `control_mask_all`, `ctrl_out`, `n_hit`, `n_miss` and `correct_rate` are computed for every session and only printed. This is the residue of the session-filtering criterion the AI implemented and then removed; the comment `# Compute correct rate for reporting (not filtering)` marks it as deliberate.
- **`np.sort()` on already-sorted spike times** — pure overhead on up to 11.5 M values per session.
- **Tongue means for bins that end up "not visible".** `tongue_y_binned[b]` is computed whenever *any* frame in the bin is visible, but is only used when at least *half* are; the rest are discarded at discretisation.
- **`visible_binned` as a full fraction array.** Only the boolean `>= 0.5` comparison is ever consumed; the fraction itself is never stored.
- **Unused fields returned from `process_session`**: `n_good_units`, `n_valid_trials`, `correct_rate`, `n_total_trials`, `rec_range` are returned in the per-session dict but never read by `main()` and never written to the pickle — so the converted data carries no per-session unit/trial counts, unlike the reference's `metadata['session_info']`.
- **`metadata['bin_centers']`** is stored although it is fully determined by `time_bin_size`, `off_start` and `off_end`.
- An unused `session_idx` parameter on `process_session`, and a `--full` flag that is a no-op (it is the default).

ii.
```python
# Compute correct rate for reporting (not filtering)
control_mask_all = (photostim_power_str == 'N/A') & (early_lick == 'no early')
ctrl_out = outcome[control_mask_all]
n_hit = (ctrl_out == 'hit').sum()
n_miss = (ctrl_out == 'miss').sum()
correct_rate = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0
```

```python
            st = units['spike_times'][idx]
            good_spike_times.append(np.sort(st))
```

```python
        visible_binned[b] = visible.mean()
        if visible.sum() > 0:
            tongue_y_binned[b] = frames[visible, 1].mean()
```

```python
    return {
        ...
        'n_good_units': n_good,
        'n_valid_trials': n_valid_trials,
        'correct_rate': correct_rate,
        'n_total_trials': n_trials,
        'rec_range': (rec_min, rec_max),
    }
```

iii. The `correct_rate` computation is retained intentionally as a per-session diagnostic — the trajectory (steps 36–37, 70–71, 96–99) shows the AI spent a long stretch of the run using exactly this statistic to decide whether the paper's >65%/≥50-per-side session criteria should gate the dataset, ultimately concluding they should not; keeping the printout preserves the evidence in `conversion_full_out.txt`. The remaining items (redundant sort, unused return fields, stored bin centres) are not discussed in the notes and appear to be leftovers rather than considered choices. None of them affect correctness, and their combined cost is small relative to the firing-rate loop.
