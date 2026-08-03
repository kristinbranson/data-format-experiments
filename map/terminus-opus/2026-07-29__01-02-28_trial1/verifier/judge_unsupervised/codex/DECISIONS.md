# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script recursively loads every `.nwb` file under `data/sub-*`. It treats each subject directory as one mouse and each NWB file as one session, then opens each file with `h5py` and reads trial, unit, event, and behavior tables from the NWB hierarchy.

ii. ```python
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
    return all_files

f = h5py.File(nwb_path, 'r')
trials = f['intervals']['trials']
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
spike_times_flat = f['units']['spike_times'][:]
```

iii. In `CONVERSION_NOTES.md`, the agent says the data are organized as 28 subject directories with 174 NWB files and that the reference pipeline had to be adapted from `.mat`/DataJoint exports to NWB. The trajectory repeatedly states that the agent mapped NWB fields to the reference analysis variables before writing `convert_data.py`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by top-level directory name. The subject ID is the `sub-*` directory name and is carried forward as the session’s `subject` field; later the script builds `subjects` and `subject_idx` from those strings.

ii. ```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])

all_files.append({
    'subject': subj,
    'path': os.path.join(subj_dir, nwb_file),
    'filename': nwb_file
})

if subj not in all_subjects:
    all_subjects.append(subj)
all_subject_idx.append(all_subjects.index(subj))
```

iii. The notes say the dataset contains 28 subjects and report that the output keeps 28 subjects. The trajectory shows the agent explicitly counting sessions per `sub-*` directory and using that as the mouse split.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The outer loop in `main()` iterates over the list returned by `get_nwb_files()`, and `process_session()` converts exactly one NWB file at a time.

ii. ```python
for i, nwb_info in enumerate(nwb_files):
    result = process_session(
        nwb_info['path'],
        nwb_info['subject'],
        show_processing=args.show_processing,
        session_idx=session_count
    )
```

iii. `CONVERSION_NOTES.md` states “174 NWB files total” and “Sessions (total NWB files) 174.” The trajectory also records the agent deciding that one NWB file corresponds to one session after exploring the NWB structure.

## 1-d. How are the data split into trials?

i. Trials are split from `intervals/trials`. The code indexes trial-level arrays by row, then builds `trial_indices` from a boolean mask. Each retained `trial_idx` produces one neural matrix, one input matrix, and one output matrix.

ii. ```python
trials = f['intervals']['trials']
n_trials_total = len(trials['id'])

trial_instruction = trials['trial_instruction'][:]
outcome = trials['outcome'][:]
early_lick = trials['early_lick'][:]

trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
trial_indices = np.where(trial_mask)[0]

for trial_idx in trial_indices:
    go_time = go_times[trial_idx]
    ...
```

iii. The notes describe `intervals/trials` as the source of `trial_instruction`, `outcome`, `early_lick`, and photostim metadata, and the trajectory shows the agent inspecting those fields before deciding to use trial-table rows as the trial split.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering happens in two stages. First, the script keeps only trials that are not `auto_water`, not `free_water`, and marked `neural_valid` by a recording-coverage heuristic. Second, it drops whole sessions unless control, non-early-lick trials reach at least 65% hit rate and at least 50 hit trials on both left and right.

ii. ```python
n_recorded_trials = f['units']['is_good_trials'].shape[1]
...
for t_idx in range(min(n_trials_total, n_recorded_trials)):
    go = go_times[t_idx]
    if (go + WINDOW_START >= min_spike_time - 1.0 and
        go + WINDOW_END <= max_spike_time + 1.0):
        neural_valid[t_idx] = True

trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
trial_indices = np.where(trial_mask)[0]

is_control = (photostim_onset_trial == b'N/A') & trial_mask
is_not_early = early_lick == b'no early'
control_non_early = is_control & is_not_early
correct_rate = hits / (hits + misses)
if correct_rate < MIN_CORRECT_RATE:
    return None
if correct_left < MIN_CORRECT_TRIALS_PER_SIDE or correct_right < MIN_CORRECT_TRIALS_PER_SIDE:
    return None
```

iii. In the notes, the agent says it intentionally kept early-lick, ignore, and photostim trials because those are decoder outputs/inputs, while still using session-performance thresholds “from methods.” The trajectory shows the agent explicitly reasoning through the conflict between the reference analysis filter and the decoder-task requirements, then deciding to exclude only `auto_water` and `free_water` at the per-trial level.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw spike timestamps in `units/spike_times` and `units/spike_times_index`, aligned to per-trial `go_start_times`. Unit QC comes from `units/classification`.

ii. ```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
good_indices = np.where(good_mask)[0]

spike_times_flat = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
```

iii. The notes explicitly map `units/spike_times` to `neural` and describe `classification="good"` as the QC filter. The trajectory says the agent verified that spike times are absolute timestamps in NWB and need to be aligned to go cue.

## 2-b. How is the `neural` data processed?

i. For each good unit and retained trial, the script extracts spikes in a fixed go-cue-centered window, subtracts the go-cue time to align them, bins them into 50 ms bins, and divides counts by bin width to convert to firing rate in Hz.

ii. ```python
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
...
abs_start = go_time + window_start
abs_end = go_time + window_end
...
aligned = unit_spikes[idx_lo:idx_hi] - go_time
counts = np.histogram(aligned, bins=bin_edges)[0]
fr_trial[i, :] = counts / bin_width
```

iii. `CONVERSION_NOTES.md` says the reference preprocessing computes firing rates from spike times, but the decoder task required changing the final representation to 50 ms bins over `[-2.5, 1.5]` s. The trajectory records the agent confirming spike times are absolute and then implementing alignment plus histogramming.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Unit-level filtering keeps only units with `classification == b'good'`. Trial-level neural filtering keeps only trials within the recorded portion of the session, using `is_good_trials.shape[1]` plus spike-time min/max coverage checks.

ii. ```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
good_indices = np.where(good_mask)[0]

n_recorded_trials = f['units']['is_good_trials'].shape[1]
...
for t_idx in range(min(n_trials_total, n_recorded_trials)):
    go = go_times[t_idx]
    if (go + WINDOW_START >= min_spike_time - 1.0 and
        go + WINDOW_END <= max_spike_time + 1.0):
        neural_valid[t_idx] = True
```

iii. The notes say the QC paper/reference code use classifier-based “good” labels and that `classification` is the mapped NWB equivalent. The trajectory shows the agent discovering that many NWB sessions have more behavioral than neurally recorded trials and using `is_good_trials` plus spike-time coverage to filter them.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to go cue onset. The code subtracts the per-trial `go_time` from spike timestamps and bins the aligned values over `[-2.5, 1.5]` seconds.

ii. ```python
WINDOW_START = -2.5
WINDOW_END = 1.5
...
go_time = go_cue_times[trial_idx]
abs_start = go_time + window_start
abs_end = go_time + window_end
aligned = unit_spikes[idx_lo:idx_hi] - go_time
```

iii. The task instructions explicitly require temporal alignment to go cue onset. The notes and trajectory both state that the agent adopted go cue as time zero after locating `BehavioralEvents/go_start_times`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms non-overlapping bins, producing 80 time bins per trial over the 4 s window. There is no second-stage temporal rebinning.

ii. ```python
BIN_WIDTH = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_TIMEBINS = int(round((WINDOW_END - WINDOW_START) / BIN_WIDTH))  # 80 bins
...
counts = np.histogram(aligned, bins=bin_edges)[0]
fr_trial[i, :] = counts / bin_width
```

iii. The notes acknowledge that the reference code used different smoothing/binning parameters, but the agent switched to the decoder task’s required 50 ms bins. The instructions explicitly specify 50 ms width bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times/timestamps` and per-trial `BehavioralEvents/go_start_times/timestamps`.

ii. ```python
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
...
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
tone_onset_rel = sample_starts_all[ss_idx] - go_time
```

iii. In the trajectory, the agent reasoned that “tone onset = sample_start_time” and that sample onset is typically about 1.85 s before go cue. The notes map “Time from tone onset” to sample-start timing relative to go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the script finds the last sample-start timestamp before the trial’s go cue using `np.searchsorted`, converts that timestamp to a relative offset from go cue, and then subtracts that offset from every neural bin center. If no earlier sample start exists, it falls back to `-1.85` s.

ii. ```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
...
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
time_from_tone = bin_centers - tone_onset_rel
```

iii. The trajectory shows the agent deciding to use the last `sample_start` before `go_time` because the NWB files can contain more sample starts than trials due to replayed/aborted attempts. The notes say sample onset is “consistently 1.85 s before go cue,” but later validation notes also report much larger `time_from_tone_onset` values, so the agent appears to have accepted this last-sample-before-go heuristic despite edge cases.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned on the exact same 80 go-cue-centered time bins as the neural data by evaluating the time-from-tone variable at each neural bin center.

ii. ```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
...
time_from_tone = bin_centers - tone_onset_rel
input_trial = np.stack([time_from_tone.astype(np.float32), photostim_binary], axis=0)
```

iii. The notes say the input values were checked against computed tone-onset times, and the trajectory says the agent wanted the non-neural streams expressed on the same aligned timeline as the firing-rate bins.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from per-trial `intervals/trials/photostim_onset`, `intervals/trials/photostim_duration`, `intervals/trials/start_time`, and per-trial `go_start_times`.

ii. ```python
photostim_onset_trial = trials['photostim_onset'][:]
photostim_duration_trial = trials['photostim_duration'][:]
trial_start_times = trials['start_time'][:]
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
```

iii. The notes map photostimulation to a binary time-varying input and state that NWB stores trial-level onset/duration while behavioral events also contain absolute photostim timestamps. The trajectory shows the agent choosing the trial-table onset/duration fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The script creates a length-80 binary vector for each trial. If the trial’s `photostim_onset` is not `N/A`, it converts onset and duration to floats, converts onset from trial-relative time to go-cue-relative time using `trial_start`, then marks bins whose centers fall inside the stimulation interval as 1.

ii. ```python
photostim_binary = np.zeros(n_bins, dtype=np.float32)
ps_onset_val = photostim_onset_trial[trial_idx]
if ps_onset_val != b'N/A':
    ps_onset_float = float(ps_onset_val)
    ps_dur_float = float(photostim_duration_trial[trial_idx])
    trial_start = trial_start_times[trial_idx]
    ps_rel_start = (trial_start + ps_onset_float) - go_time
    ps_rel_end = ps_rel_start + ps_dur_float
    photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. The trajectory shows the agent first checking the photostim fields and concluding that `photostim_onset` is stored relative to trial start. The notes then record the decision to include photostim trials and encode stim on/off as a decoder input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is converted to go-cue-relative coordinates and sampled on the same bin-center timeline as the neural firing rates.

ii. ```python
ps_rel_start = (trial_start + ps_onset_float) - go_time
ps_rel_end = ps_rel_start + ps_dur_float
photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. The notes and trajectory both say go cue is the common alignment event and that photostim was transformed into a binary time series on that shared aligned grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. It is derived directly from the trial-table field `intervals/trials/trial_instruction`.

ii. ```python
trial_instruction = trials['trial_instruction'][:]
...
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
```

iii. The notes map `trial_instruction` to the choice output and specify the left/right coding. The trajectory shows the agent confirming the field values are `left` and `right`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The script maps `left -> 0` and `right -> 1`, then repeats that scalar across all 80 time bins for the trial.

ii. ```python
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
...
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[0, :] = choice
```

iii. The notes list the intended coding `left = 0, right = 1` and say per-trial outputs are repeated across time so they match the decoder’s `(n_output, n_timepoints)` format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is derived directly from `intervals/trials/outcome`.

ii. ```python
outcome = trials['outcome'][:]
...
out = outcome[trial_idx]
```

iii. The notes map the trial-table `outcome` field to the decoder output and list the three possible values observed in the NWB data: `ignore`, `miss`, and `hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`; any unexpected value also falls back to 0. The chosen category is then repeated across all bins in the trial.

ii. ```python
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

iii. The notes record exactly this categorical mapping. The trajectory shows the agent deciding to keep ignore/no-response trials because outcome is a decoder target.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is derived directly from `intervals/trials/early_lick`.

ii. ```python
early_lick = trials['early_lick'][:]
...
early = 1 if early_lick[trial_idx] == b'early' else 0
```

iii. The notes identify `early_lick` as a target output and say the NWB field contains `early` and `no early`. The trajectory shows the agent inspecting those values early in the conversion.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The script maps `early -> 1` and every other value (effectively `no early`) to 0, then repeats that value across all time bins.

ii. ```python
early = 1 if early_lick[trial_idx] == b'early' else 0
...
output_trial[2, :] = early
```

iii. The notes explicitly state the coding `no = 0, yes = 1`. The trajectory shows the agent deliberately retaining early-lick trials so this variable could be predicted.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` and its corresponding `timestamps`. Within the data array, column 1 is treated as tongue y-position and column 2 as tracking likelihood.

ii. ```python
tongue_data = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
tongue_timestamps = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
...
y_slice = tongue_data[idx_start:idx_end, 1]
lk_slice = tongue_data[idx_start:idx_end, 2]
```

iii. The notes say the tongue tracking stream contains `(x, y, likelihood)` at about 300 Hz, and the trajectory records the agent explicitly verifying that y is column index 1.

## 8-b. How is `output` *Tongue y-position* processed?

i. For each retained trial, the script extracts tongue samples around the go-cue window, keeps only points with likelihood greater than 0.1, averages y within each 50 ms bin, and stores NaN where no high-confidence samples exist. It also concatenates all valid binned y-values across the session to compute session-level percentiles.

ii. ```python
high_conf = lk_slice > 0.1
...
for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
    n_in_bin = np.sum(bin_mask)
    if n_in_bin > 0:
        tongue_y_binned[b] = np.mean(y_slice[bin_mask])
...
valid_y = tongue_y_binned[~np.isnan(tongue_y_binned)]
if len(valid_y) > 0:
    all_tongue_y.extend(valid_y.tolist())
```

iii. The notes state that percentiles are computed “from all valid (likelihood>0.1) tracking data.” The trajectory shows the agent verifying the tongue-tracking columns, then later noticing that many bins are missing and that the final discretized distribution is strongly concentrated in the middle class.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After collecting all valid binned y-values in a session, the script computes the 40th and 60th percentiles and discretizes each bin as 0 below `p40`, 1 from `p40` through `p60`, and 2 above `p60`. Bins with missing y stay at the default category 1 because the category array is initialized to ones.

ii. ```python
if len(all_tongue_y) > 0:
    all_tongue_y_arr = np.array(all_tongue_y)
    p40 = np.percentile(all_tongue_y_arr, 40)
    p60 = np.percentile(all_tongue_y_arr, 60)
else:
    p40, p60 = 0, 0

tongue_y_disc = np.ones(n_bins, dtype=np.int64)
valid_mask = ~np.isnan(tongue_y)
if np.any(valid_mask):
    tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
    tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
    tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
```

iii. The notes say the target should use per-session percentiles, but they also note a heavy middle-class skew caused by missing tongue measurements defaulting to the middle category. The trajectory explicitly flags that issue after sample verification and then keeps the implementation.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is aligned to go cue and binned onto the same 50 ms `[-2.5, 1.5]` timeline as the neural data. The script subtracts `go_time` from tongue timestamps before binning.

ii. ```python
go_time = go_times[trial_idx]
abs_start = go_time + window_start - bin_width
abs_end = go_time + window_end + bin_width
...
ts_slice = tongue_timestamps[idx_start:idx_end] - go_time
...
tongue_y_binned = np.full(n_bins, np.nan)
```

iii. The instructions require all streams to be aligned to go cue. The notes say the tongue data are binned “to match neural data,” and the trajectory shows the agent locating the 300 Hz timestamps specifically for that alignment step.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses silent fallbacks rather than explicit error handling. If a session has no spikes, too few valid trials, no valid control trials, no good units, or poor behavioral criteria, it skips the whole session. If no sample start precedes a go cue, it uses `-1.85` s as a hard-coded tone-onset offset. If tongue tracking is absent, it fills every trial with NaN and later all middle-category tongue labels. If region JSON parsing fails, it uses `'unknown'`. Unexpected outcome labels are mapped to 0.

ii. ```python
else:
    f.close()
    return None

if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85

except (json.JSONDecodeError, AttributeError):
    return 'unknown'

if has_tongue:
    ...
else:
    tongue_y_trials = [np.full(n_bins, np.nan) for _ in trial_indices]
    all_tongue_y = []

else:
    outcome_val = 0
```

iii. The notes describe some of these as “minor mistakes” or data issues, especially partial neural coverage and missing tongue observations. The trajectory shows the agent repeatedly choosing pragmatic defaults instead of adding explicit provenance flags or per-trial missing-data markers.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is firing-rate computation, followed by tongue-tracking alignment/binning. The script itself logs both timings per session, and the notes report firing-rate computation as the clear bottleneck.

ii. ```python
t2 = time.time()
print(f"    Firing rate computation: {t2-t1:.1f}s")
...
t4 = time.time()
print(f"    Tongue tracking: {t4-t3:.1f}s")
```

iii. `CONVERSION_NOTES.md` estimates firing-rate computation at 3-14 s per session versus roughly 0.4-0.7 s for tongue tracking. The trajectory also says the agent optimized the firing-rate section because it dominated runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the nested trial-by-unit loop in `compute_firing_rates_fast()`, the per-trial/per-bin loop in `get_tongue_y_for_trials()`, the per-trial input-construction loop, and the per-unit brain-region extraction loop.

ii. ```python
for trial_idx in trial_indices:
    ...
    for i, unit_spikes in enumerate(good_spike_times):
        ...
        counts = np.histogram(aligned, bins=bin_edges)[0]

for trial_idx in trial_indices:
    ...
    for b in range(n_bins):
        bin_mask = ...

for i in range(n_total):
    ...
```

iii. The notes mention only partial optimization (`searchsorted`, pre-extracted spike arrays), which implies the remaining explicit Python loops were left in place. The trajectory’s runtime debugging focused on these exact sections.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly recomputes per-trial `searchsorted` operations, per-trial bin-center comparisons for photostim, per-trial replication of static outputs across all time bins, and per-session list-based region-to-index conversion.

ii. ```python
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
...
photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
...
output_trial[0, :] = choice
output_trial[1, :] = outcome_val
output_trial[2, :] = early
...
idx = np.array([brain_regions.index(r) for r in session_regions], dtype=np.int64)
```

iii. The notes emphasize optimizations only for the spike histogram path, not these other repeated transformations. The trajectory also shows the agent re-running similar computations during debugging and validation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script spends time on optional plotting, extensive logging/timing, and building intermediate session statistics that are not used by the saved decoder dataset. It also computes raw continuous tongue traces (`tongue_y_trials`) only to immediately discretize them for the final output.

ii. ```python
if show_processing and session_idx < 2:
    ...
    plt.savefig(f'processing_{sess_name}.png', dpi=100)

print(f"    Data loading: {t1-t0:.1f}s")
print(f"    Firing rate computation: {t2-t1:.1f}s")
...
return {
    'neural': firing_rates,
    'input': inputs_list,
    'output': outputs_list,
    'brain_regions': good_regions,
    'subject': subject_id,
    'n_good': n_good,
    'n_trials': len(firing_rates),
    'correct_rate': correct_rate,
}
```

iii. The notes explicitly describe plotting as a sanity-check aid and list multiple runtime diagnostics. Those outputs help development, but they are not part of the final `converted_data.pkl` structure used downstream.
