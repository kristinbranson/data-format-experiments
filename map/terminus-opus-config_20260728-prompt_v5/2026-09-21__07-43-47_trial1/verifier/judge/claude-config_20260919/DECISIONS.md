# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI 000363) is one NWB file per session, stored under `/app/data/sub-<id>/`. The AI walks that two-level directory tree with `os.listdir`, keeping only directories that start with `sub-` and files that end with `.nwb`, sorting at both levels so the session order is deterministic. Each file is opened once with `pynwb.NWBHDF5IO` inside a `with` block and everything (units, trials, behavioral events, video tracking) is read from that single handle. 174 files are found; 173 are converted (one is dropped for having no quality-controlled units). Every session is processed serially in a single pass, with per-session timing printed.

ii.
```python
def get_nwb_files(data_dir='/app/data'):
    """Get sorted list of all NWB file paths."""
    nwb_files = []
    for subdir in sorted(os.listdir(data_dir)):
        sub_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(sub_path) or not subdir.startswith('sub-'):
            continue
        for f in sorted(os.listdir(sub_path)):
            if f.endswith('.nwb'):
                nwb_files.append(os.path.join(sub_path, f))
    return nwb_files
```

```python
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
    ...
    units = nwb.units
    ...
    trials = nwb.intervals['trials']
    ...
    be = nwb.acquisition['BehavioralEvents']
    ...
    bts = nwb.acquisition['BehavioralTimeSeries']
```

```python
nwb_files = get_nwb_files()
print(f"Found {len(nwb_files)} NWB files")
...
for i, nwb_path in enumerate(nwb_files):
    print(f"[{i+1}/{len(nwb_files)}] Processing {os.path.basename(nwb_path)}")
    result = process_session(nwb_path, t_start=t_start, t_end=t_end, bin_width=bin_width, ...)
```

iii. From CONVERSION_NOTES.md Step 2: the AI identified the layout as "28 subject directories with 174 NWB files" and noted the original reference code reads `.mat` files exported from DataJoint while "our data is in NWB format", so `pynwb` is used as the standard reader for the published format. The AI cross-checked the counts it found against the papers (174 files, 173 sessions with good units, 28 subjects) in Steps 2–4.

## 1-b. How are the data split into subjects?

i. Each session's subject is taken from the *parent directory name* of the NWB file, e.g. `sub-440956`, via `nwb_path.split('/')[-2]`. At assembly time the unique subject strings are sorted to form `subjects`, and each session gets an index into that list via `subject_idx`. This yields 28 subjects with 3–10 sessions each. Note the AI does **not** read `nwb.subject.subject_id`; the folder name is derived from it, so the grouping is identical, but the stored ids carry the `sub-` prefix (`sub-440956` rather than `440956`).

ii.
```python
subject_id = nwb_path.split('/')[-2]
```

```python
all_subjects = sorted(set(s['subject_id'] for s in all_sessions_data))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx.append(subject_to_idx[sess['subject_id']])
...
'subjects': all_subjects,
'subject_idx': np.array(subject_idx),
```

iii. CONVERSION_NOTES.md Step 2 documents the data as "28 subject directories", i.e. the directory level *is* the animal level in the DANDI layout, so no additional grouping step is needed. The AI validated the result against the reference (28 subjects, 3–10 sessions per subject) in the Step 9/10 consistency tables.

## 1-c. How are the data split into sessions?

i. One NWB file is one session, so no splitting is required. The session identifier is the file basename with the `.nwb` suffix stripped (e.g. `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`). Sessions appear in the output in sorted-path order, which is chronological within each subject because the filename embeds the acquisition timestamp. 173 of the 174 files reach the output.

ii.
```python
session_id = os.path.basename(nwb_path).replace('.nwb', '')
```

```python
for f in sorted(os.listdir(sub_path)):
    if f.endswith('.nwb'):
        nwb_files.append(os.path.join(sub_path, f))
```

```python
'n_sessions': len(all_sessions_data),
```

iii. CONVERSION_NOTES.md Step 2 records the one-file-per-session structure, and Step 4 reconciles "174 NWB files" in the data against "173 behavioral sessions" in the papers, concluding that the single extra file (`sub-440958_ses-20190216T162508`) has no good units and so is not a usable session — an explicit match to the reference count.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the NWB trials table (`nwb.intervals['trials']`), one row per behavioural trial, with all per-trial columns read as parallel arrays (`start_time`, `stop_time`, `outcome`, `trial_instruction`, `early_lick`, `auto_water`, `free_water`, `photostim_onset`, `photostim_duration`). The go-cue timestamps are read separately from `BehavioralEvents/go_start_times` and are assumed to be in one-to-one, same-order correspondence with the trials-table rows: they are indexed with the same trial indices (`go_times[valid_indices]`). Unlike the reference, the AI does **not** assert `len(go_times) == len(trials)`.

ii.
```python
trials = nwb.intervals['trials']
n_trials_total = len(trials)
trial_starts = trials['start_time'][:]
trial_stops = trials['stop_time'][:]
outcomes = trials['outcome'][:]
instructions = trials['trial_instruction'][:]
early_lick = trials['early_lick'][:]
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
photostim_onset_str = trials['photostim_onset'][:]
photostim_duration_str = trials['photostim_duration'][:]
```

```python
be = nwb.acquisition['BehavioralEvents']
go_times = be.time_series['go_start_times'].timestamps[:]
...
go_times_valid = go_times[valid_indices]
trial_starts_valid = trial_starts[valid_indices]
```

iii. CONVERSION_NOTES.md Step 2 lists the trials table as the source of per-trial structure, and the trajectory (step 30) records the AI's check that `obs_intervals` "correspond to trial periods (368 intervals = 368 trials)", confirming the one-row-per-trial interpretation. The AI also noted (step 23) that `sample_start_times` can have several entries per trial, which is why the go cue — exactly one per trial — is used as the trial anchor.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both applied as a single boolean mask over the trials table:

1. **Reward-delivery trials removed**: `auto_water == 0 & free_water == 0`. These are trials where water was given automatically rather than earned by the animal's choice.
2. **Recording-coverage filter**: the recording window is defined as the *intersection* over all good units of `[obs_intervals[0,0], obs_intervals[-1,1]]`, and a trial is kept only if its *entire* analysis window `[go - 2.5, go + 1.5]` lies inside that range.

A session is dropped if fewer than 2 trials survive. No behavioural quality filter is applied — early-lick, no-response (`ignore`) and photostimulation trials are deliberately **kept**, because they are decoder outputs/inputs. Result: 89,532 of 94,990 trials retained (94.3%).

ii.
```python
def get_recording_time_range(nwb, good_indices):
    """Get the common recording time range across all good units.

    Returns (rec_start, rec_end) - the intersection of all units' recording periods.
    """
    rec_start = -np.inf
    rec_end = np.inf

    for idx in good_indices:
        obs = nwb.units['obs_intervals'][idx]
        unit_start = obs[0, 0]
        unit_end = obs[-1, 1]
        rec_start = max(rec_start, unit_start)
        rec_end = min(rec_end, unit_end)

    return rec_start, rec_end
```

```python
# === Filter trials ===
# 1. Remove auto_water and free_water trials
valid_trials = (auto_water == 0) & (free_water == 0)

# 2. Only include trials where the analysis window falls within the recording
recording_covered = (go_times + t_start >= rec_start) & (go_times + t_end <= rec_end)
valid_trials = valid_trials & recording_covered

valid_indices = np.where(valid_trials)[0]
n_valid = len(valid_indices)
...
if n_valid < 2:
    print(f"  Skipping {session_id}: only {n_valid} valid trials")
    return None
```

iii. Two distinct justifications are documented.

For the behavioural filter, CONVERSION_NOTES.md Steps 1/3/4 note the reference code's `get_regular_trial_mask` excludes `early_lick`, `auto_water`, `free_water`, `correctness == -1` (ignore) and photostim trials, and the AI explicitly departs from it: "for our decoder task, we want to KEEP all trials (including photostim, early lick, ignore) since these are decoder outputs/inputs… We should still filter auto_water and free_water trials as they are not genuine behavioral trials."

For the coverage filter, the trajectory (steps 55–58) records that the AI first ran the conversion without it, saw 1,061 trials with all-zero neural data in the verification log, investigated, and found that "NWB files contain behavioral data for full sessions but neural recordings only cover a subset of trials. In this example, 160/480 trials are covered." The fix reduced all-zero trials from 1,061 to 2. The 2-trial minimum per session is taken from the target-format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units['spike_times']`, the sorted spike times of each unit in session-absolute seconds, restricted to units with `classification == 'good'`. The go-cue timestamps (`BehavioralEvents/go_start_times`) supply the per-trial reference time that positions the bin edges. Spike times are pulled unit-by-unit into a Python list of arrays before binning.

ii.
```python
units = nwb.units
classifications = units['classification'][:]
good_mask = classifications == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)
...
# === Get spike times for good units ===
spike_times_list = []
for idx in good_indices:
    spike_times_list.append(units['spike_times'][idx])
```

```python
go_times = be.time_series['go_start_times'].timestamps[:]
...
fr_list, bin_centers = compute_firing_rates_vectorized(
    spike_times_list, go_times_valid, t_start, t_end, bin_width
)
```

iii. CONVERSION_NOTES.md Step 5 maps "units.spike_times (good only) → neural". The trajectory (step 30) records the AI's verification that "Spike times in NWB are absolute times (not relative to go cue)" and that "Spikes relative to go cue range from about -3 to +1.6s for trial 0", confirming that spike times are the only neural representation available and that they must be re-referenced per trial.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each good unit and each trial, `np.searchsorted` locates the spikes inside `[go - 2.5, go + 1.5]`, those spikes are re-expressed relative to the go cue, `np.histogram` counts them into the 80 fixed bins, and the counts are divided by the 50 ms bin width to give Hz. Results are stored in a `(n_trials, n_neurons, n_bins)` `float32` array and sliced into a list of per-trial `(n_neurons, n_bins)` matrices. No smoothing, no normalisation, no baseline subtraction, and no sliding/overlapping windows.

ii.
```python
def compute_firing_rates_vectorized(spike_times_list, go_times, t_start, t_end, bin_width):
    """Compute firing rates for all neurons across all trials."""
    n_bins = int(round((t_end - t_start) / bin_width))
    bin_edges = t_start + np.arange(n_bins + 1) * bin_width
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_neurons = len(spike_times_list)
    n_trials = len(go_times)

    fr_all = np.zeros((n_trials, n_neurons, n_bins), dtype=np.float32)

    for j, spk_times in enumerate(spike_times_list):
        if len(spk_times) == 0:
            continue
        for i in range(n_trials):
            go = go_times[i]
            lo = np.searchsorted(spk_times, go + t_start)
            hi = np.searchsorted(spk_times, go + t_end)
            if hi > lo:
                relative_spikes = spk_times[lo:hi] - go
                counts, _ = np.histogram(relative_spikes, bins=bin_edges)
                fr_all[i, j, :] = counts / bin_width

    fr_list = [fr_all[i] for i in range(n_trials)]
    return fr_list, bin_centers
```

iii. CONVERSION_NOTES.md Step 1 identifies `sliding_histogram` in the reference `preprocessing_DJ_2022Aug.py` as the function that "Convert[s] spike times to firing rates with sliding bins", and Step 4/5 records the deliberate choice to match its *rate* output (counts / bin width) while changing the bin geometry to the 50 ms non-overlapping bins the decoder task requires. Step 10 Check 2 reports a spot-check against the raw NWB file: "Computed firing rate for session 0, trial 10, neuron 5, bin 50 (go cue) = 40.0 Hz. Matches direct NWB computation. np.allclose = True."

## 2-c. How is the `neural` data filtered based on quality controls?

i. A unit is kept iff `units['classification'] == 'good'`. No thresholds are applied to any individual quality metric (amplitude, ISI violation, presence ratio, etc.), and the older `unit_quality` column is not used. A session with zero good units is dropped entirely. This retains 69,453 of 272,227 units (25.5%), mean 401.5 and median 390 per session, range 90–923.

ii.
```python
# === Filter units ===
units = nwb.units
classifications = units['classification'][:]
good_mask = classifications == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)

if n_good == 0:
    print(f"  Skipping {session_id}: no good units")
    return None
```

iii. CONVERSION_NOTES.md Step 3 states the neuron curation rule: "Use `classification == 'good'` from NWB units table (QC classifier output). This is equivalent to the region-specific logistic regression classifiers described in the paper." Step 1 records that the reference code's QC mode is "classifier" using "region-specific logistic regression classifiers", and Step 3 lists the paper's "5 QC classifiers: cortex, striatum, thalamus, midbrain, medulla" and the 25.9% good-unit fraction. Step 4 explicitly reconciles the small residual discrepancy: 69,453 obtained vs 69,943 reported (~0.7%), attributed to "minor version differences in QC classifier", and 25.5% vs 25.9% good-unit fraction.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go-cue onset**. All NWB streams share one session-absolute clock, so no resampling or offset correction is needed: for each trial the spike window `[go + t_start, go + t_end]` is located by `searchsorted` on that unit's spike times, and the spikes inside are shifted by subtracting the trial's go-cue time before histogramming against the fixed go-cue-relative bin edges. The same `bin_edges`/`bin_centers` grid is reused for the inputs and the tongue output, so bin *k* means the same interval in every stream.

ii.
```python
go_times = be.time_series['go_start_times'].timestamps[:]
...
go_times_valid = go_times[valid_indices]
```

```python
for i in range(n_trials):
    go = go_times[i]
    lo = np.searchsorted(spk_times, go + t_start)
    hi = np.searchsorted(spk_times, go + t_end)
    if hi > lo:
        relative_spikes = spk_times[lo:hi] - go
        counts, _ = np.histogram(relative_spikes, bins=bin_edges)
        fr_all[i, j, :] = counts / bin_width
```

```python
'temporal_alignment_event': 'Go cue onset',
'off_start': t_start,
'off_end': t_end,
```

iii. The decoder task specifies "Temporally align based on Go cue onset". CONVERSION_NOTES.md Step 3 confirms this also matches the reference: "Temporal alignment: All times relative to Go cue onset", and Step 10 Check 3 states "Temporal alignment: We align to go cue onset, same as reference." The trajectory (step 30) records the prerequisite check that NWB spike times are absolute, hence the explicit per-trial subtraction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial spanning −2.5 s to +1.5 s relative to the go cue. The edge grid is `t_start + np.arange(81) * 0.05` and is identical for every trial and every session, so `T = 80` everywhere. This is a deliberate *change* from the reference code's binning: the reference uses a 100 ms sliding window advanced with a 50 ms stride over a −3.0 to +3.5 s window. `metadata['time_bin_size']` is stored as 50.0 (ms).

ii.
```python
t_start = -2.5
t_end = 1.5
bin_width = 0.05
```

```python
n_bins = int(round((t_end - t_start) / bin_width))
bin_edges = t_start + np.arange(n_bins + 1) * bin_width
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
```

```python
'time_bin_size': bin_width * 1000,
'off_start': t_start,
'off_end': t_end,
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 1: "**Bin width = 50ms**: Task specifies 50ms bins. The reference code uses 100ms bin width with 50ms stride, but the task says '50-ms-width bins'. Use 50ms width with 50ms stride (non-overlapping)." Step 4's discrepancy table records the same reconciliation ("Bin width vs stride | bw=0.1, stride=0.05 | Task says 50ms bins | Task specifies 50ms bins"), and Step 10 Check 3 repeats that the difference is "justified by task specification". Verification confirms T = 80 for every trial.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (the sample-epoch tone onsets), together with the trials table's `start_time` and the trial's go-cue time. For each trial the AI selects the sample events falling in `[trial_start, go)` and takes the **last** one. If no sample event is found in that interval, it falls back to `go − 1.85 s`, the nominal tone-to-go interval.

ii.
```python
def get_tone_onset_per_trial(go_times, trial_starts, sample_start_times):
    """Find the tone (sample) onset time for each trial."""
    tone_onsets = np.full(len(go_times), np.nan)
    for i in range(len(go_times)):
        go = go_times[i]
        ts = trial_starts[i]
        samp_in_trial = sample_start_times[(sample_start_times >= ts) & (sample_start_times < go)]
        if len(samp_in_trial) > 0:
            tone_onsets[i] = samp_in_trial[-1]
        else:
            tone_onsets[i] = go - 1.85
    return tone_onsets
```

```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
...
tone_onsets = get_tone_onset_per_trial(go_times_valid, trial_starts_valid, sample_start_times)
```

iii. CONVERSION_NOTES.md Step 3 records the trial structure "presample -> sample (tone, -1.85s before go) -> delay (-1.2s before go) -> go cue (0) -> response", and the trajectory (step 23) records the key observation motivating "take the last": "Some trials have multiple sample events (likely failed trials that were restarted)". Step 42 of the trajectory re-derives this explicitly for trial 1 of session 1 ("there were TWO sample events: -2.5992 and -1.85. We take the last one"). Step 5 Key Decision 5 states the fallback: the tone is nominally at −1.85 s, so that value is used when no event is present.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone time is converted to a go-cue-relative offset (`tone − go`, typically −1.85 s) and subtracted from each bin centre, giving seconds elapsed since the tone at each of the 80 timepoints. It is stored as a continuous `float32` row 0 of the `(2, 80)` per-trial input array. No clipping, no normalisation, no binarisation. In the full dataset the value spans [−1.5, 11.9] s; the large upper values come from trials in which delay-epoch licking replayed the delay repeatedly, pushing the go cue far past the tone.

ii.
```python
input_list = []
for i in range(n_valid):
    inputs = np.zeros((2, n_bins), dtype=np.float32)

    # Time from tone onset (continuous)
    tone_onset_rel = tone_onsets[i] - go_times_valid[i]
    inputs[0, :] = bin_centers - tone_onset_rel
```

```python
'input_names': ['time_from_tone_onset', 'photostim_on'],
```

iii. The decoder task lists this input as "Time from tone onset in seconds (continuous, time-varying)", so a per-timepoint continuous value is the specification. In the trajectory (step 42) the AI explicitly sanity-checked the arithmetic: "at bin_center = -2.475 (first bin), time_from_tone = -2.475 - (-1.85) = -0.625 … at bin_center = 1.475 (last bin) … 3.325", then investigated why the observed maximum was larger and attributed it to trials with longer/replayed pre-go epochs. CONVERSION_NOTES.md Step 10 Check 2 reports the spot-check "Time from tone onset for session 0, trial 0, bin 0 = -0.625. Matches direct computation. np.allclose = True."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed directly on the neural bin grid: the same `bin_centers` array returned by `compute_firing_rates_vectorized` is the argument to the input construction, so element *k* of the input row corresponds by construction to bin *k* of the firing-rate matrix. Both are expressed relative to the same go-cue time for the same trial, so alignment is exact with no interpolation.

ii.
```python
fr_list, bin_centers = compute_firing_rates_vectorized(
    spike_times_list, go_times_valid, t_start, t_end, bin_width
)
n_bins = len(bin_centers)
...
tone_onset_rel = tone_onsets[i] - go_times_valid[i]
inputs[0, :] = bin_centers - tone_onset_rel
```

iii. No separate justification is recorded beyond the general alignment decision (CONVERSION_NOTES.md Step 10 Check 3, "Temporal alignment: We align to go cue onset, same as reference"); reusing the single `bin_centers` object is what guarantees there is no offset between the streams, and the `--show-processing` plot "Input 0: Time from tone onset" was used to visually confirm the go cue sits at 0 (Step 7, "Tone onset input correctly shows time from tone").

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the trials table columns `photostim_onset` and `photostim_duration` — both stored as *strings*, with the sentinel `'N/A'` on unstimulated trials, and with the onset measured relative to the trial's `start_time`. The trial's `start_time` and go-cue time are used to move them onto the go-cue-relative axis. The `BehavioralEvents/photostim_start_times` stream is not used.

ii.
```python
photostim_onset_str = trials['photostim_onset'][:]
photostim_duration_str = trials['photostim_duration'][:]
...
photostim_onset_valid = photostim_onset_str[valid_indices]
photostim_duration_valid = photostim_duration_str[valid_indices]
```

```python
if photostim_onset_valid[i] != 'N/A':
    stim_onset_abs = trial_starts_valid[i] + float(photostim_onset_valid[i])
    stim_onset_go_rel = stim_onset_abs - go_times_valid[i]
    stim_dur = float(photostim_duration_valid[i])
    stim_end_go_rel = stim_onset_go_rel + stim_dur
```

iii. CONVERSION_NOTES.md Step 2 documents the columns ("photostim_onset/power/duration"), and the trajectory (step 11) records the AI's explicit inspection of their encoding: "photostim_onset/power/duration: strings, 'N/A' for non-stim trials". Step 3 records the expected physiology — "Photostimulation: During last 0.5s of delay (onset at -1.2s relative to go), 100ms ramp-down" and "~25% randomly interleaved trials" in "N = 17 VGAT-ChR2-EYFP mice" — which the AI planned as a sanity check in Step 5.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) `float32` time series occupying row 1 of the per-trial input array: a bin is 1 when its centre falls in `[stim_onset, stim_onset + stim_duration)` and 0 otherwise. Unstimulated trials are left at the array's initial value of all zeros (the `if ... != 'N/A'` branch is simply not taken). No per-trial scalar flag and no laser-power scaling is stored.

ii.
```python
inputs = np.zeros((2, n_bins), dtype=np.float32)
...
# Photostim on/off
if photostim_onset_valid[i] != 'N/A':
    stim_onset_abs = trial_starts_valid[i] + float(photostim_onset_valid[i])
    stim_onset_go_rel = stim_onset_abs - go_times_valid[i]
    stim_dur = float(photostim_duration_valid[i])
    stim_end_go_rel = stim_onset_go_rel + stim_dur
    inputs[1, :] = ((bin_centers >= stim_onset_go_rel) & (bin_centers < stim_end_go_rel)).astype(np.float32)

input_list.append(inputs)
```

iii. The decoder task specifies "Whether photostimulation is on at every time point (discrete, time-varying)", and the target-format notes add "If an input is a time such as onset of some stimulus, represent it as a binary time series." CONVERSION_NOTES.md Step 5 Key Decision 6 records this: "**Photostim input**: Binary time series, 1 during photostim period (onset to onset+duration)." Verification confirms the range is exactly [0.0, 1.0], and Step 7 notes the `--show-processing` panel "Photostim input correctly marks stimulation period".

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulus bounds are converted from trial-start-relative to go-cue-relative (`trial_start + onset − go`) and then compared against the same `bin_centers` array that defines the firing-rate bins, so the on/off mask lands on exactly the neural grid. No interpolation or offset correction, since all times come from the one session clock.

ii.
```python
stim_onset_abs = trial_starts_valid[i] + float(photostim_onset_valid[i])
stim_onset_go_rel = stim_onset_abs - go_times_valid[i]
...
inputs[1, :] = ((bin_centers >= stim_onset_go_rel) & (bin_centers < stim_end_go_rel)).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 10 Check 5 lists this under edge cases handled: "Photostim timing: Correctly computed from trial_start + photostim_onset string." The go-cue-relative conversion is required because the raw field is trial-start-relative while every other stream in the converted dataset is go-cue-relative.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no choice column in the NWB file. The AI derives it from the *measured licks*: `BehavioralEvents/left_lick_times` and `BehavioralEvents/right_lick_times`, gated by the trial's go-cue time and the trials table's `stop_time`. Whichever spout is licked **first** after the go cue defines the choice; if neither is licked, the trial is "no lick". (The reference instead derives choice arithmetically from `trial_instruction` × `outcome`; the AI loads `trial_instruction` but never uses it.)

ii.
```python
def determine_lick_choice(go_time, trial_stop, left_lick_times, right_lick_times):
    """Determine lick choice based on first lick after go cue.
    Returns: 0='left', 1='right', 2='no_lick'
    """
    left_after = left_lick_times[(left_lick_times > go_time) & (left_lick_times < trial_stop)]
    right_after = right_lick_times[(right_lick_times > go_time) & (right_lick_times < trial_stop)]
    first_left = left_after[0] if len(left_after) > 0 else np.inf
    first_right = right_after[0] if len(right_after) > 0 else np.inf
    if first_left < first_right:
        return 0
    elif first_right < first_left:
        return 1
    else:
        return 2
```

```python
left_lick_times = be.time_series['left_lick_times'].timestamps[:]
right_lick_times = be.time_series['right_lick_times'].timestamps[:]
...
choice = determine_lick_choice(
    go_times_valid[i], trial_stops_valid[i],
    left_lick_times, right_lick_times
)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 3: "**Lick choice**: Determine from first lick after go cue (left_lick_times vs right_lick_times). No lick = ignore outcome." The Step 5 variable-mapping table lists the source as "First lick direction after go cue". The trajectory (step 28) records that the reference `.mat` data also carries `lick_directions`/`lick_times` per trial, i.e. the reference pipeline likewise treats the recorded licks as the ground truth for direction; the NWB equivalent is the two lick-event streams.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The result is coded `0 = left`, `1 = right`, `2 = no_lick`, matching the task's stated order, written into row 0 of a `(4, 80)` `int64` per-trial output array and repeated across all 80 bins so that all four outputs share one time-varying array. `output_values[0] = ['left', 'right', 'no_lick']` names the codes. In the full dataset the distribution is left 43.2%, right 42.2%, no_lick 14.6% — consistent with the independently-derived `ignore` rate of 14.8%.

ii.
```python
outputs = np.zeros((4, n_bins), dtype=np.int64)
outputs[0, :] = choice
outputs[1, :] = outcome
outputs[2, :] = early
outputs[3, :] = tongue_y_list[i]

output_list.append(outputs)
```

```python
'output_names': ['lick_choice', 'outcome', 'early_lick', 'tongue_y_position'],
'output_values': [
    ['left', 'right', 'no_lick'],
    ...
],
```

iii. The task specifies "Lick direction choice (left, right, no lick, per-trial)", so the three-way coding and the per-trial scope follow directly. The target format says outputs should be made time-varying "if at all possible", which is why the per-trial value is broadcast across the 80 bins rather than stored as a length-4 vector. CONVERSION_NOTES.md Step 10 Check 2 spot-checks the result against raw NWB: "Session 0, trial 0: choice=2 (no_lick), outcome=2 (ignore), early_lick=0 (no early). All match NWB data."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials table `outcome` column, which already holds exactly the three strings `'hit'`, `'miss'`, `'ignore'`.

ii.
```python
outcomes = trials['outcome'][:]
...
outcomes_valid = outcomes[valid_indices]
```

iii. CONVERSION_NOTES.md Step 2 records the inspected column values: "outcome: 'hit', 'ignore', 'miss'", and the Step 5 mapping table maps `trials.outcome` straight to `output[1]`. No derivation is needed because the stored categories are precisely the ones the task asks for.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped through a fixed dictionary to `0 = hit`, `1 = miss`, `2 = ignore`, written into row 1 and repeated across all 80 bins. Note this is the **reverse** of the order listed in the task ("ignore, miss, hit") and of the reference coding (`ignore=0, miss=1, hit=2`); the AI's `output_values[1] = ['hit', 'miss', 'ignore']` is consistent with its own codes, so the labelling is internally coherent. The dictionary lookup uses `.get(..., 2)`, so any unexpected string would be silently coded as `ignore`. Full-dataset distribution: hit 68.5%, miss 16.7%, ignore 14.8%.

ii.
```python
outcome_map = {'hit': 0, 'miss': 1, 'ignore': 2}
outcome = outcome_map.get(outcomes_valid[i], 2)
...
outputs[1, :] = outcome
```

```python
'output_values': [
    ['left', 'right', 'no_lick'],
    ['hit', 'miss', 'ignore'],
    ...
],
```

iii. The task lists Outcome as a three-class per-trial categorical; the target format requires `output_values` order to match the categorical output values, which the AI satisfies. CONVERSION_NOTES.md Step 9 addresses the resulting distribution against the paper: "Hit rate is lower than paper's 83.2% because we include photostim trials (which reduce performance) and early lick trials. The paper's 83.2% is for regular non-stim trials only", and Step 10 Check 4 reports that restricting to regular trials recovers 81.6% vs the paper's 83.2%.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the trials table `early_lick` column, which stores the strings `'early'` and `'no early'`.

ii.
```python
early_lick = trials['early_lick'][:]
...
early_lick_valid = early_lick[valid_indices]
```

iii. CONVERSION_NOTES.md Step 2 documents the inspected values ("early_lick: 'early' or 'no early'"), and Step 5 maps `trials.early_lick` to `output[2]`. The flag is explicit in the file, so no derivation from lick times is needed; the lick that sets the flag occurs in the sample or delay epoch, i.e. inside the −2.5 s pre-go window that the neural data covers.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A binary mapping `'early' → 1`, everything else `→ 0`, written into row 2 and repeated across all 80 bins; `output_values[2] = ['no_early', 'early']`. Full-dataset distribution: no_early 88.4%, early 11.6%. The `else 0` form means an unexpected string would silently become "no early".

ii.
```python
early = 1 if early_lick_valid[i] == 'early' else 0
...
outputs[2, :] = early
```

```python
'output_values': [
    ...
    ['no_early', 'early'],
    ...
],
```

iii. The task specifies "Early lick (no, yes, per-trial)", so `no = 0`, `yes = 1` follows the stated order directly. As with the other per-trial outputs, the value is broadcast across bins to satisfy the format's preference for time-varying outputs. CONVERSION_NOTES.md Step 10 Check 2 verifies trial 0 of session 0 against raw NWB.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is an `(n_frames, 3)` array of `(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps` (~294 Hz). Column 1 supplies the y-position, column 2 the DeepLabCut-style tracking likelihood that gates visibility. The presence of the series is checked before use.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
has_tongue = 'Camera0_side_TongueTracking' in bts.time_series
if has_tongue:
    tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
    tongue_data = tongue_ts_obj.data[:]
    tongue_timestamps = tongue_ts_obj.timestamps[:]
```

```python
tongue_y = tongue_data[:, 1]
tongue_lk = tongue_data[:, 2]
```

iii. CONVERSION_NOTES.md Step 2 lists "BehavioralTimeSeries: Camera0_side_TongueTracking (x,y,likelihood), JawTracking, NoseTracking", and the trajectory (step 12) records the AI's reading of the channel layout: "Camera tracking data with (x, y, likelihood) for Jaw, Nose, Tongue — Tongue tracking has likelihood values - low likelihood means tongue not visible". Step 23 adds the measured properties: "Tongue tracking at ~294 Hz, only ~10.5% of frames have tongue visible (likelihood > 0.9); Tongue y-position when visible: range 238-327, 40th percentile=274, 60th=289".

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps:
1. **Visibility gate**: a frame counts only if `likelihood >= 0.9`.
2. **Session-level percentiles**: the 40th and 60th percentiles are taken over the *raw y-values of all visible frames in the whole session* (not over binned means), giving `p40`, `p60`.
3. **Per-trial binning**: frames in `[go − 2.5, go + 1.5]` are assigned to the 80 go-cue-relative bins; within each bin the *mean of the visible frames' y* is computed.
4. **Discretisation**: that mean is compared against `p40`/`p60` to give class 0/1/2; a bin with no visible frame keeps its initial value 3.

Full-dataset distribution: below_p40 11.7%, p40_to_p60 6.3%, above_p60 6.5%, not_visible 75.5%.

ii.
```python
p40, p60 = None, None
if has_tongue:
    visible_mask = tongue_data[:, 2] >= 0.9
    if np.any(visible_mask):
        y_visible = tongue_data[visible_mask, 1]
        p40 = np.percentile(y_visible, 40)
        p60 = np.percentile(y_visible, 60)
```

```python
result = np.full((n_trials, n_bins), 3, dtype=np.int64)

for i in range(n_trials):
    go = go_times[i]
    abs_edges = go + bin_edges
    idx_start = np.searchsorted(tongue_timestamps, abs_edges[0])
    idx_end = np.searchsorted(tongue_timestamps, abs_edges[-1])
    if idx_start >= idx_end:
        continue
    trial_ts = tongue_timestamps[idx_start:idx_end]
    trial_y = tongue_y[idx_start:idx_end]
    trial_lk = tongue_lk[idx_start:idx_end]

    bin_indices = np.searchsorted(abs_edges, trial_ts, side='right') - 1
    valid = (bin_indices >= 0) & (bin_indices < n_bins)
    ...
    for b in range(n_bins):
        mask = bin_indices_v == b
        if not np.any(mask):
            continue
        visible = trial_lk_v[mask] >= likelihood_threshold
        if np.any(visible):
            mean_y = np.mean(trial_y_v[mask][visible])
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 4: "**Tongue y-position**: Use likelihood > 0.9 threshold for visibility. Discretize visible positions using session-wide 40th/60th percentiles. Bin tongue data into 50ms bins aligned to go cue." The 0.9 threshold is empirically grounded in the trajectory (step 23): only ~10.5% of frames exceed it, i.e. the likelihood is effectively bimodal. Taking the percentiles over *visible* frames only is justified by the fact that the tracker still emits a coordinate when the tongue is retracted, so including those frames would corrupt the distribution. Step 7 notes the plot review: "Tongue y-position shows expected pattern (mostly not visible, visible around response)."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-bin mean y is compared to the two session percentiles with an explicit if/elif/else: `mean_y < p40 → 0`, `p40 <= mean_y <= p60 → 1`, `mean_y > p60 → 2`; bins with no visible frame stay at the pre-filled `3`. The sessions where the tongue series is absent, or where no frame is ever visible, get class 3 for every bin of every trial. `output_values[3] = ['below_p40', 'p40_to_p60', 'above_p60', 'not_visible']`.

ii.
```python
if mean_y < p40:
    result[i, b] = 0
elif mean_y <= p60:
    result[i, b] = 1
else:
    result[i, b] = 2
```

```python
if has_tongue and p40 is not None:
    tongue_y_list = compute_tongue_y_vectorized(
        tongue_data, tongue_timestamps, go_times_valid,
        t_start, t_end, bin_width, p40, p60
    )
else:
    tongue_y_list = [np.full(n_bins, 3, dtype=np.int64) for _ in range(n_valid)]
```

```python
['below_p40', 'p40_to_p60', 'above_p60', 'not_visible'],
```

iii. This is a direct transcription of the decoder-task specification ("0: < 40th percentile of y-position over the session; 1: 40th to 60th percentile; 2: > 60th percentile; 3: not visible"), including the per-session scope of the percentiles. CONVERSION_NOTES.md Step 5 Key Decision 4 and the Step 5 mapping table both record it as "Discretized: 0=<p40, 1=p40-p60, 2=>p60, 3=not_visible | Time-varying, per-session percentiles".

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. This is the only genuinely time-varying output, so it is the only one requiring real alignment. The camera timestamps share the session-absolute clock with the spikes and events, so the AI builds the trial's absolute bin edges as `go + bin_edges` — the *same* `bin_edges` grid used for the firing rates — brackets the relevant frames with two `searchsorted` calls, and assigns each frame to a bin with `searchsorted(abs_edges, trial_ts, side='right') - 1`. No interpolation or offset correction is applied. Bins of trials whose window extends before the video (the camera is trial-gated) simply contain no frames and fall into class 3.

ii.
```python
n_bins = int(round((t_end - t_start) / bin_width))
bin_edges = t_start + np.arange(n_bins + 1) * bin_width
...
for i in range(n_trials):
    go = go_times[i]
    abs_edges = go + bin_edges

    idx_start = np.searchsorted(tongue_timestamps, abs_edges[0])
    idx_end = np.searchsorted(tongue_timestamps, abs_edges[-1])
    ...
    bin_indices = np.searchsorted(abs_edges, trial_ts, side='right') - 1
    valid = (bin_indices >= 0) & (bin_indices < n_bins)
```

iii. Reusing the identical `bin_edges` definition guarantees bin *k* of the tongue output covers the same 50 ms interval as bin *k* of the firing-rate matrix. CONVERSION_NOTES.md Step 6 records that the `searchsorted` formulation was introduced as an optimisation of a much slower frame-by-frame version ("52-117s -> 0.2-0.3s per session"), and Step 7's plot review confirms the resulting raster shows tongue visibility appearing after the go cue, as expected.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six cases are handled, all by exclusion or by an explicit fallback value:

- **Session never quality-controlled** (`classification` is NaN for all units): no unit matches `'good'`, so the session returns `None` and is dropped. One session (`sub-440958_ses-20190216T162508`).
- **Neural recording shorter than the behavioural session**: trials whose analysis window is not fully covered by the intersection of the good units' `obs_intervals` are dropped (~1,073 trials), after the AI traced 1,061 all-zero-neural trials to this cause. Two all-zero trials remain and are accepted as boundary edge cases.
- **Sessions reduced below 2 trials**: dropped.
- **Missing tone event in a trial**: fall back to the nominal `go − 1.85 s`.
- **Missing / unusable tongue tracking**: if the series is absent, or no frame ever exceeds the likelihood threshold, every bin of every trial is set to class 3; within a trial, a bin with no visible frame likewise stays at 3.
- **Unexpected categorical strings**: `outcome_map.get(x, 2)` silently defaults to `ignore`, `early_lick` defaults to `no_early`, and `region_to_idx.get(r, 0)` defaults to the first brain region.

ii.
```python
if n_good == 0:
    print(f"  Skipping {session_id}: no good units")
    return None
```

```python
recording_covered = (go_times + t_start >= rec_start) & (go_times + t_end <= rec_end)
valid_trials = valid_trials & recording_covered
...
if n_valid < 2:
    print(f"  Skipping {session_id}: only {n_valid} valid trials")
    return None
```

```python
if len(samp_in_trial) > 0:
    tone_onsets[i] = samp_in_trial[-1]
else:
    tone_onsets[i] = go - 1.85
```

```python
if has_tongue and p40 is not None:
    tongue_y_list = compute_tongue_y_vectorized(...)
else:
    tongue_y_list = [np.full(n_bins, 3, dtype=np.int64) for _ in range(n_valid)]
```

```python
outcome = outcome_map.get(outcomes_valid[i], 2)
early = 1 if early_lick_valid[i] == 'early' else 0
...
region_indices = np.array([region_to_idx.get(r, 0) for r in sess['brain_regions']])
```

iii. CONVERSION_NOTES.md Step 10 Check 5 enumerates these as the edge cases addressed: "Recording coverage: Fixed issue where trials outside recording window had zero neural data; Tone onset: Fallback to go_time - 1.85 if no sample event found; Photostim timing: Correctly computed from trial_start + photostim_onset string; Sessions with 0 good units: Correctly skipped (1 session)", and the Issues log states "Zero neural data in 1061 trials: Fixed by filtering trials outside recording window. After fix: Only 2 edge-case trials with zero data remain." The general principle applied is that where nothing was recorded the data are excluded (emitting them would fabricate 4 s of 0 Hz activity), whereas where a measurement legitimately has no value — a retracted tongue — an explicit extra category is used rather than imputation, as the task's class 3 requires.

## 10-a. What are the most time-consuming steps of the code?

i. The AI instrumented each stage with `time.time()` and printed per-session breakdowns. From `conversion_full_out.txt`, full conversion took 589.8 s for 174 files (3.4 s/session average), split roughly as:

1. **Firing-rate computation** (`compute_firing_rates_vectorized`) — 0.7–3.1 s/session, the dominant cost; it performs `n_good_units × n_trials` iterations (up to ~900 × ~700 ≈ 600k) of `searchsorted` + `np.histogram`.
2. **NWB reads** — 0.3–0.7 s/session, dominated by the per-unit ragged reads `units['spike_times'][idx]` and by `get_recording_time_range`, which reads each good unit's full `obs_intervals` array from HDF5.
3. **Tongue discretisation** — 0.1–0.3 s/session after optimisation (it was 52–117 s/session before).
4. **Pickling the 11.3 GB result**, outside the reported 589.8 s.

ii.
```python
t0 = time.time()
...
t_load_spikes = time.time()
print(f"  Loaded {n_good} good units in {t_load_spikes - t0:.1f}s (rec: {rec_start:.0f}-{rec_end:.0f}s)")
...
t_fr = time.time()
print(f"  Computed firing rates in {t_fr - t_load:.1f}s")
...
t_tongue = time.time()
print(f"  Computed tongue y in {t_tongue - t_fr:.1f}s")
...
t_end_proc = time.time()
print(f"  Total processing time: {t_end_proc - t0:.1f}s")
```

```python
total_time = time.time() - total_start
print(f"\nProcessed {len(all_sessions)} sessions in {total_time:.1f}s")
print(f"Average time per session: {total_time/max(len(all_sessions),1):.1f}s")
```

iii. CONVERSION_NOTES.md Step 6 records the profiling result and the fixes: "Initial tongue computation was O(n_trials * n_bins * n_frames) = very slow; Initial firing rate used mask operations instead of searchsorted… Used np.searchsorted for tongue frame lookup (52-117s -> 0.2-0.3s per session); Used np.searchsorted for spike time windowing. Total speedup: 182s -> 8.2s for 2 sessions (22x)". Step 7's run-time table extrapolates "Total 4.1s avg → ~12 min", within the instructions' 15-minute budget, which is the stated reason no further optimisation was pursued.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Despite the `_vectorized` suffixes, three nested Python loops remain:

1. **`compute_firing_rates_vectorized`: the inner `for i in range(n_trials)` loop.** The outer per-unit loop is unavoidable (spike arrays are ragged), but the trial dimension can be collapsed: building one flat array of all `n_trials × 81` absolute edges, calling `searchsorted` once per unit, reshaping and differencing removes ~700 Python iterations and ~700 `np.histogram` calls per unit. This is the single largest available win and would roughly halve total runtime.
2. **`compute_tongue_y_vectorized`: the inner `for b in range(n_bins)` loop** with a boolean `mask = bin_indices_v == b` recomputed for each of the 80 bins. This is O(n_bins × n_frames_in_trial) and is exactly what `np.bincount` on the bin index (sum and count, then divide) does in one pass. The outer per-trial loop could also be collapsed onto a single global bin index.
3. **The per-trial `for i in range(n_valid)` loops that build `input_list` and `output_list`**, plus `get_tone_onset_per_trial`'s per-trial loop. `time_from_tone`, the photostim mask, and the tone lookup (`searchsorted(sample_times, go) - 1`) are all expressible as single broadcast expressions over `(n_trials, n_bins)`.
4. **`determine_lick_choice` called once per trial**, each call masking the entire session-long lick arrays (see 10-c).

ii.
```python
for j, spk_times in enumerate(spike_times_list):
    if len(spk_times) == 0:
        continue
    for i in range(n_trials):                      # <- vectorizable over trials
        go = go_times[i]
        lo = np.searchsorted(spk_times, go + t_start)
        hi = np.searchsorted(spk_times, go + t_end)
        if hi > lo:
            relative_spikes = spk_times[lo:hi] - go
            counts, _ = np.histogram(relative_spikes, bins=bin_edges)
            fr_all[i, j, :] = counts / bin_width
```

```python
for b in range(n_bins):                            # <- replaceable by np.bincount
    mask = bin_indices_v == b
    if not np.any(mask):
        continue
    visible = trial_lk_v[mask] >= likelihood_threshold
    if np.any(visible):
        mean_y = np.mean(trial_y_v[mask][visible])
```

```python
for i in range(n_valid):                           # <- broadcastable
    inputs = np.zeros((2, n_bins), dtype=np.float32)
    tone_onset_rel = tone_onsets[i] - go_times_valid[i]
    inputs[0, :] = bin_centers - tone_onset_rel
```

iii. The AI's own account (CONVERSION_NOTES.md Step 6) claims these are already handled: "Vectorized firing rate computation using np.searchsorted and np.histogram; Vectorized tongue y-position computation". Its stated rationale for stopping there is the Step 7 time estimate ("~12 min"), which met the instructions' 15-minute threshold, so no further optimisation was attempted. The AI never identified the remaining loops as remaining loops — the function names assert a vectorisation the bodies do not implement.

## 10-c. What processing does the code repeat multiple times?

i. Four repetitions, none of which changes the result but all of which cost time:

1. **`determine_lick_choice` re-scans the full session lick arrays on every trial.** Each call applies two boolean masks over the complete `left_lick_times`/`right_lick_times` arrays (tens of thousands of entries), so the session cost is O(n_trials × n_licks) when a single pair of `searchsorted` calls (or one pass with pre-sorted merge) would suffice.
2. **The bin grid is rebuilt twice per session**: `bin_edges`/`bin_centers` are computed inside `compute_firing_rates_vectorized` and `bin_edges` again inside `compute_tongue_y_vectorized`, from the same three constants. It is not hoisted to module level.
3. **`outcome_map = {'hit': 0, 'miss': 1, 'ignore': 2}` is reconstructed on every trial**, inside the output loop.
4. **`obs_intervals` is read from HDF5 once per good unit** in `get_recording_time_range` (up to ~900 separate ragged reads per session), and `spike_times` likewise once per good unit, rather than reading each backing buffer once and slicing it.

ii.
```python
for i in range(n_valid):
    choice = determine_lick_choice(
        go_times_valid[i], trial_stops_valid[i],
        left_lick_times, right_lick_times)      # full arrays re-masked each trial

    outcome_map = {'hit': 0, 'miss': 1, 'ignore': 2}   # rebuilt each trial
    outcome = outcome_map.get(outcomes_valid[i], 2)
```

```python
def compute_firing_rates_vectorized(...):
    n_bins = int(round((t_end - t_start) / bin_width))
    bin_edges = t_start + np.arange(n_bins + 1) * bin_width       # (1)

def compute_tongue_y_vectorized(...):
    n_bins = int(round((t_end - t_start) / bin_width))
    bin_edges = t_start + np.arange(n_bins + 1) * bin_width       # (2) identical
```

```python
for idx in good_indices:
    obs = nwb.units['obs_intervals'][idx]     # one HDF5 read per unit
...
for idx in good_indices:
    spike_times_list.append(units['spike_times'][idx])   # one HDF5 read per unit
```

iii. The AI does not document any repeated processing; CONVERSION_NOTES.md Step 6 lists only the two inefficiencies it fixed (the naive tongue loop and mask-based spike selection). The implicit justification is again the Step 7 budget check: each session is read exactly once from disk and converted in a single pass, so the redundancies above were never large enough to push the run past the 15-minute target (actual: 9.8 min).

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Five items of genuinely wasted work:

1. **`trial_instruction` is loaded, sliced per trial, and never used.** `instructions = trials['trial_instruction'][:]` and `instructions_valid = instructions[valid_indices]` are computed for every session and passed to `plot_processing`, whose body ignores the argument entirely. (Static check of unused parameters: `plot_processing` never references `go_times`, `outcomes`, `instructions`, `early_lick`, or `tongue_y_list`.) This is the variable the reference uses to derive choice; here it is dead weight.
2. **`get_recording_time_range` reads each good unit's entire `obs_intervals` array** (one row per trial) from HDF5 but uses only `obs[0, 0]` and `obs[-1, 1]` — two scalars out of up to ~1,600 per unit, for up to ~900 units per session.
3. **`bin_centers` is stored in every session's result dict and then dropped**: `assemble_dataset` never reads `sess['bin_centers']`, so the array is carried through the whole pipeline for nothing.
4. **Outputs are stored as `int64`** although every value is in 0–3; `int8` would do. At 4 × 80 × 8 bytes per trial × 89,532 trials this is ~230 MB of pickle written and re-read for no benefit (the reference uses `int8`).
5. **Five arguments passed to `plot_processing` are unused**, and `n_trials_total`, `n_auto_free`, `n_not_covered` are computed purely for log lines.

ii.
```python
instructions = trials['trial_instruction'][:]
...
instructions_valid = instructions[valid_indices]
...
plot_processing(session_id, bin_centers, fr_list, input_list, output_list,
                go_times_valid, outcomes_valid, instructions_valid,
                early_lick_valid, tongue_y_list, session_idx)
```

```python
for idx in good_indices:
    obs = nwb.units['obs_intervals'][idx]   # full array read
    unit_start = obs[0, 0]                  # only two scalars used
    unit_end = obs[-1, 1]
```

```python
return {
    ...
    'bin_centers': bin_centers,     # never read by assemble_dataset
}
```

```python
outputs = np.zeros((4, n_bins), dtype=np.int64)   # values are only 0..3
```

iii. The AI does not acknowledge any of these; CONVERSION_NOTES.md Step 12 concludes "No issues found. All accuracies are reasonable and consistent." The implicit justification is that none of them affects correctness — the unused reads and the oversized dtype cost time and disk but not accuracy — and the AI's efficiency review stopped once the Step 7 estimate met the 15-minute budget.
