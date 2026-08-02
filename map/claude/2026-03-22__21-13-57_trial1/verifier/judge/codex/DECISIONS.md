# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent treats the NWB files in `/app/data/sub-*/sub-*_ses-*.nwb` as the full corpus. It discovers every file with `glob`, sorts them, then opens each file with `pynwb.NWBHDF5IO` and reads subject metadata, units, trials, behavioral events, and tongue tracking from the NWB object.

ii.
```python
def get_nwb_files():
    """Get list of all NWB files sorted by subject then session."""
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*_ses-*.nwb')))

def load_nwb_session(nwb_path):
    import pynwb
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    ...
    units = nwb.units
    trials = nwb.trials
    be = nwb.acquisition['BehavioralEvents']
    bts = nwb.acquisition['BehavioralTimeSeries']
```

iii. In `CONVERSION_NOTES.md`, Step 2 records “174 NWB files total” and Step 4 explicitly resolves the format discrepancy by using NWB rather than the original `.mat` export. Trajectory step 58 says the agent concluded the NWB files contain the needed units, trials, behavioral events, and video streams.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity comes from `nwb.subject.subject_id`. The agent stores unique subject IDs in insertion order, then creates `subject_idx` so each retained session points back to its mouse.

ii.
```python
subject_id = nwb.subject.subject_id
...
subjects_set = OrderedDict()
...
sid = result['subject_id']
if sid not in subjects_set:
    subjects_set[sid] = len(subjects_set)
...
subjects = list(subjects_set.keys())
subject_idx.append(subjects_set[sess['subject_id']])
subject_idx = np.array(subject_idx, dtype=np.int64)
```

iii. Step 5 of the notes maps `subject.subject_id` to `subjects` and `subject_idx`. The notes also state the dataset has 28 subject directories and that the converted output should preserve those mouse IDs.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. During conversion, the agent processes files one-by-one, keeps only sessions that pass session-level selection rules, and appends each accepted file as one entry in `neural`, `input`, `output`, and `brain_region_idx`.

ii.
```python
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
    if result is None:
        continue
    all_sessions.append(result)
...
for sess in all_sessions:
    neural.append(sess['neural'])
    inputs.append(sess['input'])
    outputs.append(sess['output'])
```

iii. In Step 4 the agent notes that the reference code grouped probe files into sessions in the `.mat` version, but in the NWB release each file is already a session. The full conversion log confirms it iterates over 174 NWB files and retains 144 sessions after filtering.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `trials` table, and the per-trial alignment anchor is `BehavioralEvents/go_start_times`. After building a boolean trial mask, the agent uses `trial_indices` to subset trial-level arrays, so each retained row becomes one trial in the output session lists.

ii.
```python
trials = nwb.trials
trials_data = {
    'start_time': trials['start_time'][:],
    'stop_time': trials['stop_time'][:],
    'trial_instruction': trials['trial_instruction'][:],
    'outcome': trials['outcome'][:],
    ...
}
go_times = be.time_series['go_start_times'].timestamps[:]
...
trial_indices = np.where(trial_mask)[0]
go_times = data['go_times'][trial_indices]
```

iii. Trajectory step 60 says the agent checked that go-start events map 1:1 onto trials and used that to define trial segmentation. Step 5 of the notes also states the conversion window is trialized around go cue onset.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps most behavioral trial types for decoder targets/inputs, but removes `auto_water` and `free_water` trials from the dataset. Separately, it applies session-level inclusion criteria based on performance on control, non-early, non-ignore trials, and it tries to drop trials whose aligned neural window lies outside the recording span.

ii.
```python
trial_mask = np.ones(data['n_trials'], dtype=bool)
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False
...
performance, correct_left, correct_right = compute_session_performance(data['trials_data'])
if performance < MIN_PERFORMANCE: return None
if correct_left < MIN_CORRECT_LEFT: return None
if correct_right < MIN_CORRECT_RIGHT: return None
...
recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                 (go_times + ALIGN_END >= min_spike_time)
trial_indices = trial_indices[valid_positions]
```

iii. Step 3 of the notes quotes the paper’s session criteria (>65% performance and at least 50 correct left/right trials). Step 5 says the decoder conversion will exclude only `auto_water` and `free_water`, while keeping early-lick, ignore, and stimulation trials because those become decoder outputs or inputs. Trajectory step 60 repeats that rationale.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from NWB unit spike times. The agent reads `units['spike_times']` for units whose `classification` equals `'good'`, then uses `anno_name` to decide which of those units belong to one of the retained major brain regions.

ii.
```python
classifications = units['classification'][:]
good_mask = classifications == 'good'
anno_names = units['anno_name'][:]
spike_times_vi = units['spike_times']
all_spike_times = np.array(spike_times_vi.target.data[:])
all_st_idx = np.array(spike_times_vi.data[:])
...
good_anno_names = anno_names[good_mask]
good_spike_times.append(all_spike_times[start_idx:end_idx])
```

iii. Step 1 and Step 4 of the notes explicitly state that the agent mapped the reference QC classifier onto NWB’s `classification == 'good'` field, and used `anno_name` as the histological region label.

## 2-b. How is the `neural` data processed?

i. The agent converts absolute session spike times into go-cue-centered firing rates. For each neuron and trial, it extracts spikes in the `[-2.5, 1.5]` s window around the go cue, histograms them into 50 ms non-overlapping bins, and divides by bin width to produce firing rates in Hz.

ii.
```python
BIN_WIDTH = 0.050
ALIGN_START = -2.5
ALIGN_END = 1.5
...
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
for n in range(n_neurons):
    for t in range(n_trials):
        go_t = go_times[t]
        idx_lo = np.searchsorted(st, go_t + align_start, side='left')
        idx_hi = np.searchsorted(st, go_t + align_end, side='left')
        if idx_hi > idx_lo:
            rel_spikes = st[idx_lo:idx_hi] - go_t
            counts, _ = np.histogram(rel_spikes, bins=bin_edges)
            all_matrices[t, n, :] = counts / bin_width
```

iii. The notes show the agent recognized the conflict between the original reference preprocessing (`40 ms` sliding windows with `3.4 ms` stride) and the decoder spec (`50 ms` bins). It chose the task-specified decoder representation; trajectory steps 25, 58, and 60 document that decision.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC consists of keeping units labeled `'good'` in the NWB units table and dropping units whose `anno_name` cannot be mapped into the agent’s 14 major regions. No additional per-unit trial-validity mask such as `is_good_trials` is applied.

ii.
```python
good_mask = classifications == 'good'
...
for i, anno in enumerate(good_anno):
    region = map_anno_to_region(anno)
    if region is not None:
        region_labels.append(region)
        neuron_mask.append(i)
...
spike_times_list = [data['good_spike_times'][i] for i in neuron_mask]
```

iii. Step 1 and Step 4 of the notes justify `classification == 'good'` as the NWB analogue of the paper’s classifier-based QC. Trajectory step 60 confirms the agent considered but did not use `is_good_trials`, and instead filtered only by `classification` plus region mapping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every neural trial is aligned to go cue onset. The agent uses `BehavioralEvents/go_start_times` as the alignment event, subtracts that trial’s go time from spike timestamps, and bins the resulting relative times in a fixed `[-2.5, 1.5]` s window.

ii.
```python
go_times = be.time_series['go_start_times'].timestamps[:]
...
rel_spikes = st[idx_lo:idx_hi] - go_t
counts, _ = np.histogram(rel_spikes, bins=bin_edges)
...
'temporal_alignment_event': 'Go cue onset',
'off_start': ALIGN_START,
'off_end': ALIGN_END,
```

iii. The notes repeatedly state that the target alignment event is the go cue. Step 4 also records that, unlike the `.mat` reference data, the NWB spike times are absolute and must be explicitly realigned to go cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins and 80 time points per trial. The agent does not do any secondary rebinning or smoothing; it bins raw spikes directly into the target decoder resolution.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins (decoder task spec)
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
...
neural_trials = bin_spikes(spike_times_list, go_times, ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS)
```

iii. Step 1 of the notes captures the original paper/code parameters (`bw = 0.04`, `stride = 0.0034`), while Step 5 records the decision that the decoder task specification overrides those values. Trajectory steps 25 and 60 say the same.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from the sample/tone onset events in `BehavioralEvents/sample_start_times`, together with `go_start_times` and the trial `start_time`/`stop_time` bounds so the agent can choose the tone onset belonging to each trial.

ii.
```python
sample_start_ts = be.time_series['sample_start_times'].timestamps[:]
go_times = be.time_series['go_start_times'].timestamps[:]
...
tone_onsets = get_tone_onset_for_trials(
    go_times, data['sample_start_ts'],
    td['start_time'][trial_indices], td['stop_time'][trial_indices]
)
```

iii. Step 2 of the notes and trajectory step 60 both highlight that sample events can replay on early-lick trials, so trial start/stop times are needed to assign the correct sample onset to each trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the agent selects the last sample-start event between trial start and the go cue, because early licks can replay the sample epoch. It then computes, for every 50 ms bin center, the elapsed time since that selected tone onset.

ii.
```python
for i in range(n_trials):
    mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
    matching = sample_start_ts[mask]
    if len(matching) > 0:
        tone_onsets[i] = matching[-1]
...
tone_rel = tone_t - go_t
tone_input = (bin_centers - tone_rel).astype(np.float32)
```

iii. Trajectory step 60 is explicit: “Sample events can be repeated due to early licks … need the LAST sample start before go cue.” Step 5 of the notes documents the same mapping.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same go-cue-centered 80-bin grid as the neural matrix. The input is evaluated at the same bin centers used for spike binning, so input time series and neural activity are time-locked sample-by-sample.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
...
tone_input = (bin_centers - tone_rel).astype(np.float32)
...
inp = np.stack([tone_onset_input[t], photostim_input[t]], axis=0).astype(np.float32)
```

iii. Step 5 of the notes states that all decoder variables share the same `-2.5 s` to `+1.5 s` go-cue-aligned window. The processing plot code also treats the tone-onset input and neural activity on the same x-axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The photostimulation input is derived from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`. The agent does not use the per-trial `photostim_onset`/`power`/`duration` fields for the time-varying input itself.

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
...
photostim_input = compute_photostim_input(
    go_times, data['photostim_start_ts'], data['photostim_stop_ts'],
    ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS
)
```

iii. Step 5 of the notes maps photostimulation to a binary time-varying decoder input sourced from photostim start/stop events. Trajectory step 60 says the same.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each retained trial, the agent subtracts the trial’s go cue time from all photostim start/stop timestamps, checks whether each event overlaps the aligned window, and sets bins to `1.0` when the bin center falls inside an active photostimulation interval.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
for t in range(n_trials):
    ps = np.zeros(n_bins, dtype=np.float32)
    for si in range(len(photostim_start_ts)):
        ps_start = photostim_start_ts[si] - go_t
        ps_stop = photostim_stop_ts[si] - go_t
        if ps_stop < align_start or ps_start > align_end:
            continue
        for b in range(n_bins):
            if bin_centers[b] >= ps_start and bin_centers[b] < ps_stop:
                ps[b] = 1.0
```

iii. The notes say the variable is intended as a binary “photostim on” signal. The agent’s reasoning in trajectory step 60 describes it that way as well.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned by subtracting go cue time and sampling onto the same 80 go-cue-centered 50 ms bins used for the neural data.

ii.
```python
ps_start = photostim_start_ts[si] - go_t
ps_stop = photostim_stop_ts[si] - go_t
...
inp = np.stack([tone_onset_input[t], photostim_input[t]], axis=0).astype(np.float32)
```

iii. Step 5 in the notes places photostimulation in the common `-2.5` to `+1.5` s go-cue-aligned frame shared by all converted trial data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The choice output is taken from the NWB trial column `trial_instruction`, not from lick-event timestamps. The agent interprets the instructed left/right trial label as the decoder’s left/right choice variable.

ii.
```python
instructions = td['trial_instruction'][trial_indices]
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
```

iii. Step 5 of the notes maps `trial_instruction` to `choice`. The trajectory summary also describes the chosen output as left/right from the trial table.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The processing is a categorical remap: left becomes `0`, right becomes `1`. The per-trial label is then broadcast across all 80 time bins so the output tensor has a uniform time dimension.

ii.
```python
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
...
np.full(N_BINS, choices[t], dtype=np.int64)  # choice (per-trial, broadcast)
```

iii. The notes explicitly list “left=0, right=1 (per-trial)” in the Step 5 variable map. The broadcast is a consequence of the target format requiring matched time dimensions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the NWB trial column `outcome`.

ii.
```python
outcomes_raw = td['outcome'][trial_indices]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
```

iii. Step 5 of the notes maps `trials.outcome` directly to the categorical decoder output with the requested ignore/miss/hit ordering.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent remaps string labels to integer categories `ignore=0`, `miss=1`, and `hit=2`, then broadcasts the per-trial value across all 80 bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
...
np.full(N_BINS, outcomes[t], dtype=np.int64)  # outcome (per-trial, broadcast)
```

iii. This exact mapping appears in Step 5 of `CONVERSION_NOTES.md`. The agent chose a time-broadcast representation because the decoder format allows categorical outputs to be per-trial or time-varying.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. The agent does not create any `distance to reward zone` variable. Instead, it aligns `outcome` with neural data by broadcasting the session’s per-trial outcome category across the same 80 go-cue-centered bins used for the neural matrix.

ii.
```python
out = np.array([
    np.full(N_BINS, choices[t], dtype=np.int64),
    np.full(N_BINS, outcomes[t], dtype=np.int64),
    np.full(N_BINS, early_licks[t], dtype=np.int64),
    tongue_y_discrete[t].astype(np.int64),
], dtype=np.int64)
```

iii. The task instructions and the agent’s notes never introduce a reward-zone-distance output, so the agent justified only an `outcome` output. Step 5 documents `outcome` as the second decoder output and says nothing about reward-zone distance.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from the NWB trial column `early_lick`.

ii.
```python
early_lick_raw = td['early_lick'][trial_indices]
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
```

iii. Step 5 of the notes maps `trials.early_lick` to the decoder output `no=0, yes=1`. Trajectory step 60 states that early-lick trials are kept because early lick itself is a decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The agent maps the string `'early'` to `1` and everything else to `0`, then broadcasts the result across all bins within the trial.

ii.
```python
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
...
np.full(N_BINS, early_licks[t], dtype=np.int64)  # early lick (per-trial, broadcast)
```

iii. The notes describe exactly this binary remap. The decision is also tied to the earlier decision to retain early-lick trials rather than exclude them.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the side-view tracking time series `BehavioralTimeSeries/Camera0_side_TongueTracking`: specifically the y coordinate column, the confidence column, and the associated timestamps.

ii.
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]      # (n_frames, 3): x, y, confidence
tongue_timestamps = tongue_ts_obj.timestamps[:]
...
tongue_y = tongue_data[:, 1].astype(np.float64)
tongue_conf = tongue_data[:, 2].astype(np.float64)
```

iii. Step 2 of the notes documents the tongue stream as `(x, y, confidence)` at about 294 Hz, and Step 5 maps the y coordinate into the decoder output.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent replaces low-confidence frames with the session mean y value, extracts data in each trial’s go-cue-aligned window, and averages the imputed y positions within each 50 ms bin to form a continuous per-trial tongue-y trace before discretization.

ii.
```python
visible_mask = tongue_conf >= confidence_threshold
session_mean_y = np.mean(tongue_y[visible_mask])
tongue_y_imputed = tongue_y.copy()
tongue_y_imputed[~visible_mask] = session_mean_y
...
mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
trial_ts = tongue_timestamps[mask] - go_t
trial_y = tongue_y_imputed[mask]
...
trial_tongue_y[b] = np.mean(in_bin)
```

iii. Step 5 says the agent followed the paper’s qualitative rule that occluded tongue positions should be set to the mean. Step 6 notes the implementation detail and the later optimization of this code path.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The agent concatenates all aligned, binned tongue-y values across trials in a session, computes the 40th and 60th percentiles on that pooled array, and maps each bin to `0`, `1`, or `2`. If the two percentiles collapse to the same value, it artificially separates them by a small epsilon so a middle class exists in principle.

ii.
```python
all_values = np.concatenate([t for t in tongue_y_trials])
p40 = np.percentile(all_values, 40)
p60 = np.percentile(all_values, 60)
if np.isclose(p40, p60):
    eps = max(1e-6, abs(p40) * 1e-4)
    p40 = p40 - eps
    p60 = p60 + eps
...
d[trial_y >= p40] = 1
d[trial_y >= p60] = 2
```

iii. Step 5 says the target discretization is session-wise 40/60 percentile thresholding. Step 6 notes the added `p40 == p60` fix, and trajectory step 122 explains the agent added that because imputation could collapse the distribution.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue trace is cut into the same go-cue-centered `[-2.5, 1.5]` s window as the neural data and then averaged into the same 50 ms bins, so each tongue sample lines up with the neural bins.

ii.
```python
window_start = go_t + align_start
window_end = go_t + align_end
mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
trial_ts = tongue_timestamps[mask] - go_t
...
tongue_y_trials, session_mean_y = compute_tongue_y_per_trial(
    data['tongue_data'], data['tongue_timestamps'], go_times,
    ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS
)
```

iii. Step 5 explicitly says the entire converted dataset is go-cue aligned with 50 ms bins, and tongue y-position is listed as a time-varying output in that common frame.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent uses ad hoc fallback rules. Missing tone-onset assignment becomes an all-zero input trace. Missing tongue frames in a window become the session mean y. Low-confidence tongue samples are imputed with the session mean. Sessions with no mapped neurons or too few valid trials are skipped. Trials outside the recording span are intended to be dropped. If tongue percentiles collapse, the thresholds are separated by epsilon.

ii.
```python
if np.isnan(tone_t):
    tone_input = np.zeros(n_bins, dtype=np.float32)
...
if len(trial_ts) == 0:
    tongue_y_trials.append(np.full(n_bins, session_mean_y, dtype=np.float32))
...
tongue_y_imputed[~visible_mask] = session_mean_y
...
if np.isclose(p40, p60):
    eps = max(1e-6, abs(p40) * 1e-4)
...
if n_neurons < 1:
    return None
if n_valid_trials < 2:
    return None
```

iii. Step 4 and Step 6 of the notes document the main discrepancy handling rules, especially the occluded-tongue imputation and percentile-collapse patch. Trajectory step 122 also explains the percentile issue the agent thought it needed to solve.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant costs are NWB loading, spike binning, and tongue processing for every retained session. The agent’s own runtime notes say spike binning is the biggest single compute block, followed by tongue processing and file I/O.

ii.
```python
t_load = time.time() - t0
...
t1 = time.time()
neural_trials = bin_spikes(...)
t_bin = time.time() - t1
...
t2 = time.time()
tongue_y_trials, session_mean_y = compute_tongue_y_per_trial(...)
t_tongue = time.time() - t2
```

iii. Step 7 of `CONVERSION_NOTES.md` includes explicit per-session runtime estimates: about `1.5 s` for NWB loading, `3.3 s` for spike binning, and `1.5 s` for tongue processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still contains several nested Python loops that could be vectorized: neuron-by-trial spike histogramming, trial-by-stimulation-bin photostim filling, and the per-bin averaging loop inside tongue processing.

ii.
```python
for n in range(n_neurons):
    for t in range(n_trials):
        ...

for t in range(n_trials):
    for si in range(len(photostim_start_ts)):
        for b in range(n_bins):
            ...

for t in range(n_trials):
    ...
    for b in range(n_bins):
        in_bin = trial_y[bin_indices == b]
```

iii. Step 6 of the notes says the agent already optimized some of these paths relative to earlier drafts, but the final code still leaves the core nested loops in place.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly scans event arrays per trial, repeatedly computes per-trial broadcast outputs, repeatedly scans every photostim event for every trial, and repeatedly remaps regions with string matching on every good neuron in every session.

ii.
```python
for i, anno in enumerate(good_anno):
    region = map_anno_to_region(anno)
...
for t in range(n_valid_trials):
    inp = np.stack([tone_onset_input[t], photostim_input[t]], axis=0)
...
for t in range(n_valid_trials):
    out = np.array([
        np.full(N_BINS, choices[t], dtype=np.int64),
        np.full(N_BINS, outcomes[t], dtype=np.int64),
        np.full(N_BINS, early_licks[t], dtype=np.int64),
        tongue_y_discrete[t].astype(np.int64),
    ], dtype=np.int64)
```

iii. Step 6 and Step 7 show the agent was actively profiling and revising runtime, which implies it recognized these repeated passes as important implementation details.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads several streams it never uses for the final converted output: left and right lick timestamps, photostim power/duration, and `subject_desc`. It also computes raw continuous tongue traces only to discard them after discretization, and it contains optional plotting logic that is not part of the saved decoder dataset.

ii.
```python
left_lick_ts = be.time_series['left_lick_times'].timestamps[:]
right_lick_ts = be.time_series['right_lick_times'].timestamps[:]
...
'photostim_power': trials['photostim_power'][:],
'photostim_duration': trials['photostim_duration'][:],
...
tongue_y_trials, session_mean_y = compute_tongue_y_per_trial(...)
tongue_y_discrete = discretize_tongue_y(tongue_y_trials, session_mean_y)
...
if show_processing:
    plot_processing(...)
```

iii. The notes focus on the saved pickle structure and decoder validation, not these extra fields, which indicates they were loaded mainly for exploration or debugging rather than for the final downstream analyses.
