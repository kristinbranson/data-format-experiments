# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `data/` for `sub-*` directories, lists `.nwb` files inside each subject directory, and opens each file directly with `h5py`. Within each file it reads HDF5 datasets from `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, `units`, and `general/extracellular_ephys/electrodes`.

ii.
```python
def get_nwb_files():
    subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({
                'subject': subj,
                'path': os.path.join(subj_dir, nwb_file),
                'filename': nwb_file
            })
```

```python
f = h5py.File(nwb_path, 'r')
trials = f['intervals']['trials']
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
spike_times_flat = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
```

iii. In `CONVERSION_NOTES.md` Step 6 the AI says it “used h5py for NWB file reading.” In trajectory Steps 10-17 it explicitly explored the NWB HDF5 layout and decided to map NWB fields directly rather than use `pynwb`.

## 1-b. How are the data split into subjects?

i. Subjects are split by folder name, not by the NWB subject field. Each file discovered under `data/sub-<id>/` is tagged with `'subject': subj`, and the final `subjects` list is assembled from the distinct folder names in first-seen order.

ii.
```python
all_files.append({
    'subject': subj,
    'path': os.path.join(subj_dir, nwb_file),
    'filename': nwb_file
})
```

```python
subj = result['subject']
if subj not in all_subjects:
    all_subjects.append(subj)
all_subject_idx.append(all_subjects.index(subj))
```

iii. The AI’s justification in `CONVERSION_NOTES.md` Step 2 is that the dataset is organized as “28 subjects (sub-440956 through sub-484677).” There is no sign it considered `nwb.subject.subject_id`; it treated the directory structure as the subject split.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Session order comes from sorted subject directories and sorted filenames within each directory. The AI does not keep a separate `session_id` in the final output.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
...
nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```

```python
for i, nwb_info in enumerate(nwb_files):
    result = process_session(
        nwb_info['path'],
        nwb_info['subject'],
        show_processing=args.show_processing,
        session_idx=session_count
    )
```

iii. `CONVERSION_NOTES.md` Step 2 states there are 174 NWB files total, and the AI consistently treats each NWB file as a session unit throughout the script and logs.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table rows, indexed in parallel with the go-cue timestamps. After building a boolean `trial_mask`, the AI uses `trial_indices = np.where(trial_mask)[0]` and processes those selected rows as the retained trials for the session.

ii.
```python
trials = f['intervals']['trials']
n_trials_total = len(trials['id'])
...
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
...
trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
trial_indices = np.where(trial_mask)[0]
```

iii. In trajectory Step 14 the AI noted that go-cue timestamps matched the session trial count and used that as the basis for trial-level alignment. It did not try to reconstruct trials from raw event streams.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials by three rules: exclude `auto_water`, exclude `free_water`, and exclude trials whose go cue is outside an inferred valid neural recording span (`neural_valid`). It also drops entire sessions failing control-trial performance thresholds: correct rate below 65%, or fewer than 50 correct left and 50 correct right control non-early trials.

ii.
```python
neural_valid = np.zeros(n_trials_total, dtype=bool)
for t_idx in range(min(n_trials_total, n_recorded_trials)):
    go = go_times[t_idx]
    if (go + WINDOW_START >= min_spike_time - 1.0 and
        go + WINDOW_END <= max_spike_time + 1.0):
        neural_valid[t_idx] = True

trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
trial_indices = np.where(trial_mask)[0]
```

```python
is_control = (photostim_onset_trial == b'N/A') & trial_mask
is_not_early = early_lick == b'no early'
control_non_early = is_control & is_not_early
...
if correct_rate < MIN_CORRECT_RATE:
    return None
if correct_left < MIN_CORRECT_TRIALS_PER_SIDE or correct_right < MIN_CORRECT_TRIALS_PER_SIDE:
    return None
```

iii. The justification is explicit in trajectory Steps 35, 50, and 57-59 and in `CONVERSION_NOTES.md` Steps 3-5, 9, and 10: the AI decided to keep early-lick, ignore, and photostim trials because they are decoder outputs/inputs, but still apply method-paper session thresholds and a heuristic neural-coverage filter using `is_good_trials.shape[1]` plus spike-time range.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units/spike_times` and `units/spike_times_index`, with `acquisition/BehavioralEvents/go_start_times/timestamps` providing the alignment event. The unit subset is restricted to `units/classification == b'good'`.

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
good_indices = np.where(good_mask)[0]
```

```python
spike_times_flat = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
```

iii. `CONVERSION_NOTES.md` Step 5 maps “units/spike_times” to `neural`, and Step 6 says firing rates are computed from spike times of “good” units.

## 2-b. How is the `neural` data processed?

i. The AI converts spike times to firing rates by making 50 ms bins from -2.5 s to +1.5 s around each trial’s go cue, histogramming spikes per unit per trial, and dividing counts by bin width to get Hz. No smoothing or baseline subtraction is applied.

ii.
```python
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
...
idx_lo = np.searchsorted(unit_spikes, abs_start)
idx_hi = np.searchsorted(unit_spikes, abs_end)
if idx_hi > idx_lo:
    aligned = unit_spikes[idx_lo:idx_hi] - go_time
    counts = np.histogram(aligned, bins=bin_edges)[0]
    fr_trial[i, :] = counts / bin_width
```

iii. `CONVERSION_NOTES.md` Step 6 states that it “computed firing rates as spike counts / bin_width (Hz),” and Step 5 says the transform is “50ms histogram bins, [-2.5, 1.5]s.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Unit-level filtering is `classification == b'good'`. If a session has no such units, it is skipped. Trial-level neural filtering is handled separately by `neural_valid`.

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
n_good = int(np.sum(good_mask))
...
if n_good == 0:
    return None
```

iii. `CONVERSION_NOTES.md` Steps 1, 3, and 5 all say the QC rule is the classifier-based “good” label from the NWB `classification` field.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns every trial to go-cue onset. It subtracts each trial’s `go_time` from spike times before histogramming, so bin zero is relative to the go cue.

ii.
```python
go_time = go_cue_times[trial_idx]
abs_start = go_time + window_start
abs_end = go_time + window_end
...
aligned = unit_spikes[idx_lo:idx_hi] - go_time
counts = np.histogram(aligned, bins=bin_edges)[0]
```

iii. `CONVERSION_NOTES.md` Steps 3 and 5 say “Temporal alignment: go cue = time 0.” Trajectory Step 46 says spike times are absolute and “need to align to go cue.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms non-overlapping bins, 80 total bins over the 4 s window from -2.5 s to +1.5 s. This is a rebinning from raw spike times to coarse rate bins.

ii.
```python
BIN_WIDTH = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_TIMEBINS = int(round((WINDOW_END - WINDOW_START) / BIN_WIDTH))
```

```python
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
counts = np.histogram(aligned, bins=bin_edges)[0]
fr_trial[i, :] = counts / bin_width
```

iii. `CONVERSION_NOTES.md` Steps 1, 3, and 5 repeatedly justify this as following the decoder task specification rather than the method paper’s 40 ms sliding window.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` and `go_start_times`. The AI chooses the last sample-start timestamp before each trial’s go cue.

ii.
```python
sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
...
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
```

iii. Trajectory Steps 14 and 17 note that `sample_start_times` has more entries than trials because early licks replay the sample epoch. The AI therefore chose the sample onset immediately preceding the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the AI computes the tone onset relative to go cue and then subtracts that relative onset from the fixed bin centers. If no prior sample start exists, it falls back to `-1.85` s.

ii.
```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
...
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
time_from_tone = bin_centers - tone_onset_rel
```

iii. `CONVERSION_NOTES.md` Step 5 maps this variable as “bin_centers - tone_onset_rel.” The fallback is not justified there; it appears to be an invented safeguard rather than something from the references.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same 80 go-cue-relative bin centers as the neural firing rates, so each timepoint in the tone-onset input is aligned one-to-one with the neural bins.

ii.
```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
time_from_tone = bin_centers - tone_onset_rel
input_trial = np.stack([time_from_tone.astype(np.float32), photostim_binary], axis=0)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says this input is time-varying over the same 80-bin trial window used for neural data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial-table fields `photostim_onset`, `photostim_duration`, and `start_time`, together with `go_times` to express the stimulation window relative to the alignment event.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][:]
photostim_duration_trial = trials['photostim_duration'][:]
trial_start_times = trials['start_time'][:]
...
ps_onset_float = float(ps_onset_val)
ps_dur_float = float(photostim_duration_trial[trial_idx])
trial_start = trial_start_times[trial_idx]
ps_rel_start = (trial_start + ps_onset_float) - go_time
ps_rel_end = ps_rel_start + ps_dur_float
```

iii. Trajectory Step 17 says the AI verified `photostim_onset` is relative to trial start, `photostim_start_times` are absolute, and the onset relative to go cue is about `-0.5 s`, matching the methods description.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI creates a binary per-bin vector. For each selected trial, bins whose centers fall between the stimulation onset and offset are set to 1; all other bins stay 0.

ii.
```python
photostim_binary = np.zeros(n_bins, dtype=np.float32)
...
photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 describes this as “Binary from photostim_onset/duration,” and the trajectory shows the AI chose to keep photostim trials because photostimulation is a decoder input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulation onset and offset are converted into go-cue-relative coordinates, and then compared against the same `bin_centers` used for neural data.

ii.
```python
ps_rel_start = (trial_start + ps_onset_float) - go_time
ps_rel_end = ps_rel_start + ps_dur_float
photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. The AI’s justification in trajectory Step 17 is that the photostim timing should be represented relative to go cue because the neural data are aligned to go cue.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives `choice` only from `trial_instruction`, not from actual outcome. Left-instruction trials become choice 0 and right-instruction trials become choice 1.

ii.
```python
trial_instruction = trials['trial_instruction'][:]
...
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly maps `trial_instruction -> output[0] choice | left=0, right=1`. There is no evidence the AI considered deriving actual lick direction from `trial_instruction` together with `outcome`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as a two-class per-trial label, repeats it across all 80 bins, and does not create a no-lick class for `ignore` trials.

ii.
```python
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
...
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[0, :] = choice
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['below_p40', 'p40_to_p60', 'above_p60'],
],
```

iii. The only justification is the Step 5 mapping in `CONVERSION_NOTES.md`; the AI treated the instructed side as the choice variable the decoder should predict.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table `outcome` column.

ii.
```python
outcome = trials['outcome'][:]
...
out = outcome[trial_idx]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `outcome` directly to `output[1]`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats the resulting code across all 80 bins of the trial.

ii.
```python
if out == b'ignore':
    outcome_val = 0
elif out == b'miss':
    outcome_val = 1
elif out == b'hit':
    outcome_val = 2
else:
    outcome_val = 0
...
output_trial[1, :] = outcome_val
```

iii. `CONVERSION_NOTES.md` Step 5 lists exactly this mapping and describes it as a per-trial output repeated across time.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table `early_lick` field.

ii.
```python
early_lick = trials['early_lick'][:]
...
early = 1 if early_lick[trial_idx] == b'early' else 0
```

iii. `CONVERSION_NOTES.md` Step 5 maps `early_lick` directly to `output[2]`.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `b'early'` to 1 and everything else to 0, then repeats that code across all 80 bins of the trial.

ii.
```python
early = 1 if early_lick[trial_idx] == b'early' else 0
...
output_trial[2, :] = early
```

iii. `CONVERSION_NOTES.md` Step 5 describes this as “no=0, yes=1” and per-trial repeated across time.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 of `data` is used as `tongue_y`, column 2 as confidence/likelihood, and `timestamps` provide temporal alignment.

ii.
```python
tongue_data = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
tongue_timestamps = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
...
y_slice = tongue_data[idx_start:idx_end, 1]
lk_slice = tongue_data[idx_start:idx_end, 2]
```

iii. Trajectory Steps 12-14 and 46 say the AI verified the tongue tracking columns are `(tongue_x, tongue_y, tongue_likelihood)` and that the frame rate is about 300 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each retained trial, the AI extracts frames around the go cue, keeps frames with `likelihood > 0.1`, averages `tongue_y` within each 50 ms bin, then pools all non-NaN trial-bin means from the selected trials to compute session-level 40th and 60th percentile thresholds.

ii.
```python
high_conf = lk_slice > 0.1
...
for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
    n_in_bin = np.sum(bin_mask)
    if n_in_bin > 0:
        tongue_y_binned[b] = np.mean(y_slice[bin_mask])
```

```python
if len(all_tongue_y) > 0:
    all_tongue_y_arr = np.array(all_tongue_y)
    p40 = np.percentile(all_tongue_y_arr, 40)
    p60 = np.percentile(all_tongue_y_arr, 60)
```

iii. `CONVERSION_NOTES.md` Step 5 says “Tongue y percentiles: Computed per-session from all valid (likelihood>0.1) tracking data.” In trajectory Step 61 the AI noticed the middle category dominated because many bins had missing tongue data, but it did not revise the method.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI thresholds binned tongue-y values into three categories using session percentiles: below `p40` -> 0, between `p40` and `p60` inclusive -> 1, above `p60` -> 2. Bins with no valid tongue sample remain at the default middle category 1.

ii.
```python
tongue_y_disc = np.ones(n_bins, dtype=np.int64)
valid_mask = ~np.isnan(tongue_y)
if np.any(valid_mask):
    tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
    tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
    tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
```

iii. `CONVERSION_NOTES.md` Step 10 explicitly states “Missing tongue tracking: Defaults to middle category (1) for NaN values.” The AI’s stated rationale was mainly that the task requested a 3-level discretization.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses the same go-cue-centered 50 ms bin grid as the neural data. For each retained trial it subtracts `go_time` from tongue timestamps and bins the resulting offsets with the same window edges.

ii.
```python
abs_start = go_time + window_start - bin_width
abs_end = go_time + window_end + bin_width
...
ts_slice = tongue_timestamps[idx_start:idx_end] - go_time
...
bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
```

iii. Trajectory Steps 13-14 and `CONVERSION_NOTES.md` Steps 3 and 5 show the AI’s alignment model was “go cue = time 0” for every stream, including tongue tracking.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases heuristically: sessions with no spikes are dropped; sessions with no good units are dropped; trials outside inferred spike coverage are excluded; sessions with no valid control trials are dropped; missing tongue data is encoded as the middle tongue class; missing photostim is encoded as all zeros; missing or unparsable electrode-location JSON becomes `'unknown'`; and a missing prior sample onset falls back to `-1.85`.

ii.
```python
if len(spike_times_flat) > 0:
    max_spike_time = spike_times_flat.max()
    min_spike_time = spike_times_flat.min()
else:
    f.close()
    return None
```

```python
def extract_brain_region(location_json_bytes):
    try:
        ...
        return loc_dict.get('brain_regions', 'unknown')
    except (json.JSONDecodeError, AttributeError):
        return 'unknown'
```

```python
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
...
tongue_y_disc = np.ones(n_bins, dtype=np.int64)
```

iii. The justifications are in `CONVERSION_NOTES.md` Steps 5, 9, and 10 and trajectory Steps 57-61: the AI wanted to prevent zero-neural trials from entering the dataset and to keep output arrays dense even when tongue tracking was absent.

## 10-a. What are the most time-consuming steps of the code?

i. The AI treated firing-rate computation as the dominant cost, with tongue processing and raw file loading as the next largest costs. Its notes estimate roughly 3-14 s per session for firing rates and 0.4-0.7 s for tongue tracking.

ii.
```python
t1 = time.time()
print(f"    Data loading: {t1-t0:.1f}s")
...
t2 = time.time()
print(f"    Firing rate computation: {t2-t1:.1f}s")
...
t4 = time.time()
print(f"    Tongue tracking: {t4-t3:.1f}s")
```

iii. `CONVERSION_NOTES.md` Steps 6 and 7 explicitly discuss timing, optimization, and the estimate that the full run would take about 16 minutes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI left several vectorizable loops in place: the outer per-trial and inner per-unit neural loops in `compute_firing_rates_fast`, the per-trial and per-bin loops in `get_tongue_y_for_trials`, the per-trial input-construction loop, and the per-session `brain_regions.index` loop when building `brain_region_idx`.

ii.
```python
for trial_idx in trial_indices:
    ...
    for i, unit_spikes in enumerate(good_spike_times):
        ...
        counts = np.histogram(aligned, bins=bin_edges)[0]
```

```python
for trial_idx in trial_indices:
    ...
    for b in range(n_bins):
        bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
```

iii. The only explicit justification is in `CONVERSION_NOTES.md` Step 6, where the AI says it used some optimizations (`searchsorted`, pre-extracted spike times) and considered the resulting speed acceptable.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats several computations: it rebuilds `bin_centers` inside every session, loops trial-by-trial to recompute tone and photostim inputs, loops trial-by-trial and bin-by-bin for tongue binning, and linearly searches `brain_regions.index(r)` for every neuron when constructing `brain_region_idx`.

ii.
```python
n_bins = N_TIMEBINS
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)

inputs_list = []
for trial_idx in trial_indices:
    ...
```

```python
brain_regions = sorted(list(all_brain_regions_set))
brain_region_idx = []
for session_regions in all_brain_region_idx_raw:
    idx = np.array([brain_regions.index(r) for r in session_regions], dtype=np.int64)
    brain_region_idx.append(idx)
```

iii. The AI did not explicitly call these repetitions out as a problem. Its notes instead emphasize that the run time was acceptable after basic optimizations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and logs several quantities that do not enter the final saved dataset: session-level `correct_rate`, `correct_left`, `correct_right`, and timing/plotting diagnostics. In `--show-processing` mode it also creates plots solely for manual inspection. The returned per-session `correct_rate` is later discarded when building the final pickle.

ii.
```python
correct_rate = hits / (hits + misses)
correct_left = np.sum((outcome == b'hit') & (trial_instruction == b'left') & control_non_early)
correct_right = np.sum((outcome == b'hit') & (trial_instruction == b'right') & control_non_early)
...
return {
    ...
    'correct_rate': correct_rate,
}
```

```python
if show_processing and session_idx < 2:
    ...
    plt.savefig(f'processing_{sess_name}.png', dpi=100)
```

iii. The justification from `CONVERSION_NOTES.md` Steps 6-10 is that these computations support filtering, debugging, and sanity checks. They are not retained as decoder inputs/outputs in the saved dataset.
