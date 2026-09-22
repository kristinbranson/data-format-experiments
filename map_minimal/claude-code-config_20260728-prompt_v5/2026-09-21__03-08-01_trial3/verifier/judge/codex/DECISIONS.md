# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script walks `/app/data` by subject directory, then by `.nwb` file inside each subject directory. Each NWB file is treated as one session and opened with `h5py.File`; the code then reads trial, unit, event, and video arrays directly from HDF5 paths inside that file.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
...
nwb_files = sorted([f for f in os.listdir(sub_dir) if f.endswith('.nwb')])
...
with h5py.File(nwb_path, 'r') as f:
    n_trials = len(f['intervals/trials/id'])
    outcome = np.array([x.decode() for x in f['intervals/trials/outcome'][:]])
```

iii. In the trajectory the agent first inspected the `/app/data/sub-*` layout and then inspected one NWB file's internal groups. It concluded that processing should happen one NWB file at a time by reading the HDF5 datasets directly rather than using `pynwb`.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the top-level folder names such as `sub-440956`. The output `subjects` list is built from subject directories that end up contributing at least one kept session, and `subject_idx` is the position of each session's directory name in that list.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
...
if not sub_has_session:
    subject_names.append(sub)
    sub_has_session = True

sub_idx = subject_names.index(sub)
all_subject_idx.append(sub_idx)
...
'subjects': subject_names,
'subject_idx': np.array(all_subject_idx),
```

iii. The trajectory shows the agent enumerating `sub-*` directories and treating them as the mouse identifiers. There is no evidence that it used `nwb.subject.subject_id`; the folder structure itself was treated as the subject split.

## 1-c. How are the data split into sessions?

i. One `.nwb` file is one session. Sessions are processed in sorted order within each subject directory, and each kept file contributes one entry to `neural`, `input`, `output`, and `subject_idx`.

ii.
```python
for sub in subjects:
    sub_dir = os.path.join(DATA_DIR, sub)
    nwb_files = sorted([f for f in os.listdir(sub_dir) if f.endswith('.nwb')])
    ...
    for nf in nwb_files:
        nwb_path = os.path.join(sub_dir, nf)
        result = process_session(nwb_path)
```

iii. The trajectory reflects the same assumption throughout: the agent listed files like `sub-440956_ses-20190207T120657_behavior+ecephys+ogen.nwb` and worked session-by-session from those file boundaries.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trial table under `intervals/trials`. The code treats each row index as a trial index, builds a boolean mask on those rows, and uses the same indices into `go_start_times` and the other per-trial arrays.

ii.
```python
n_trials = len(f['intervals/trials/id'])
...
trial_mask = (auto_water == 0) & (free_water == 0)
trial_indices = np.where(trial_mask)[0]
...
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
for trial_idx in trial_indices:
    go_t = go_times[trial_idx]
```

iii. In the trajectory the agent explicitly noted that the trial table had 368 rows and that `go_start_times` also had 368 timestamps, while `sample_start_times` had extra entries because early licks replay the sample epoch. That is the justification it used for indexing trials from the trial table and not from the repeated event streams.

## 1-e. How are trials filtered based on quality controls?

i. The script applies two layers of filtering. First, it drops whole sessions unless control-trial performance exceeds 65% and there are at least 50 correct left and 50 correct right control trials. Second, inside kept sessions it drops only `auto_water` and `free_water` trials. It explicitly keeps early-lick, ignore/no-response, and photostim trials. It does not use `obs_intervals` or any per-trial spike-observation QC.

ii.
```python
control_mask = (
    (auto_water == 0) & (free_water == 0) &
    (early_lick == 'no early') & (~has_stim) &
    (outcome != 'ignore')
)
...
if performance <= 0.65 or n_correct_left < 50 or n_correct_right < 50:
    return None
...
trial_mask = (auto_water == 0) & (free_water == 0)
trial_indices = np.where(trial_mask)[0]
```

iii. The trajectory shows the agent relying on the paper's session-selection criteria (`>65%` control performance and `>=50` correct left/right trials). It also reasoned that early-lick, ignore, and photostim trials should remain because they are needed as decoder outputs or inputs, and it explicitly decided not to use `is_good_trials` or other trial-level spike-availability filters because it considered zero-spike trials acceptable.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` and `units/spike_times_index` for units whose `units/classification` equals `'good'`. Trial go-cue times from `acquisition/BehavioralEvents/go_start_times/timestamps` provide the alignment event.

ii.
```python
clf_raw = f['units/classification'][:]
...
good_indices = np.where(good_mask)[0]
...
spike_times_flat = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]
...
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. The trajectory repeatedly states that the NWB `classification` field represents the classifier-based QC output and that spike times should be aligned to the go cue and binned.

## 2-b. How is the `neural` data processed?

i. For each kept trial and each good unit, the code subtracts that trial's go-cue time from the unit's spike times, bins the aligned spikes into non-overlapping 50 ms bins from `-2.5` to `+1.5` s, and divides counts by bin width to convert to firing rate. There is no smoothing or normalization.

ii.
```python
def bin_spike_times(spike_times, t_start, t_end, bin_width):
    n_bins = int(round((t_end - t_start) / bin_width))
    edges = np.linspace(t_start, t_end, n_bins + 1)
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts / bin_width
...
aligned = spks - go_t
fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
```

iii. The trajectory says the neural data should be "extracted spike times aligned to the go cue and binned into 50 ms windows." No additional neural preprocessing was justified.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Unit-level neural QC is `classification == 'good'`. If `classification` is missing/non-object for a session, the whole session is skipped; if no units are classified as good, the session is also skipped. The code does not use `unit_quality`, `is_good_trials`, or metric thresholds.

ii.
```python
clf_raw = f['units/classification'][:]
if clf_raw.dtype == object:
    classification = np.array([x.decode() if isinstance(x, bytes) else str(x)
                               for x in clf_raw])
    good_mask = classification == 'good'
else:
    # No classifier QC available (all NaN) – skip session
    return None
...
if len(good_indices) == 0:
    return None
```

iii. The trajectory explicitly compares `classification` with `unit_quality`, decides that `classification` is the stricter classifier-based QC, and decides to skip the one session where classifier labels are missing rather than fall back to `unit_quality`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The per-trial neural window is centered on each trial's go cue. Alignment is implemented by subtracting `go_t` from each unit's absolute spike times and binning the aligned spikes over the fixed window `[-2.5, 1.5]`.

ii.
```python
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
for trial_idx in trial_indices:
    go_t = go_times[trial_idx]
    ...
    aligned = spks - go_t
    fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
```

iii. The trajectory states that the decoder should be aligned to go-cue onset and that spike times and other signals are all already on the same session clock, so subtraction of the per-trial go time is sufficient.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural activity is represented in 50 ms bins over a 4 s window, producing 80 bins per trial. There is no secondary temporal rebinning or smoothing.

ii.
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_END = 1.5
...
n_bins = int(round((t_end - t_start) / bin_width))
```

iii. The trajectory cites the task instructions directly here: align from `-2.5 s` to `+1.5 s` around the go cue with `50 ms` bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` and `acquisition/BehavioralEvents/go_start_times/timestamps`. For each trial, the code uses the last sample-start timestamp strictly before that trial's go cue.

ii.
```python
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
sample_start_ts = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
...
before = sample_start_ts[sample_start_ts < go_times[i]]
if len(before) > 0:
    tone_onset_rel_go[i] = before[-1] - go_times[i]
```

iii. The trajectory explicitly investigated the mismatch between 368 go cues and 405 sample starts, concluded that extra sample starts come from replayed sample epochs after early licks, and therefore chose the last sample onset before each go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The code first computes `tone_onset_rel_go`, a negative offset from go cue to the chosen sample onset. It then defines the time-varying input as `bin_centers - tone_rel`, which equals seconds since tone onset at each go-cue-centered bin. If no sample onset is found before a go cue, it falls back to a hard-coded `-1.85 s` offset.

ii.
```python
tone_onset_rel_go = np.full(n_trials, np.nan)
for i in range(n_trials):
    before = sample_start_ts[sample_start_ts < go_times[i]]
    if len(before) > 0:
        tone_onset_rel_go[i] = before[-1] - go_times[i]
...
if np.isnan(tone_onset_rel_go[trial_idx]):
    tone_rel = -1.85
else:
    tone_rel = tone_onset_rel_go[trial_idx]
time_from_tone = bin_centers - tone_rel
```

iii. The trajectory shows the agent measuring that sample start is about `1.85 s` before go cue and reasoning that the decoder input should be "current_time + 1.85", but then deciding to compute it per trial from the actual sample-start timestamps.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same `bin_centers` used for neural firing rates, so each element of the time-from-tone input corresponds to the same go-cue-relative bin as the neural data.

ii.
```python
bin_centers = get_bin_centers(T_START, T_END, BIN_WIDTH)
...
time_from_tone = bin_centers - tone_rel
...
fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
```

iii. The trajectory describes both signals as go-cue-centered time series over the same `-2.5` to `+1.5 s` window.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from trial-table fields `photostim_onset`, `photostim_duration`, and `start_time`, together with trial go-cue times.

ii.
```python
photostim_onset_str = np.array(
    [x.decode() for x in f['intervals/trials/photostim_onset'][:]])
photostim_dur_str = np.array(
    [x.decode() for x in f['intervals/trials/photostim_duration'][:]])
trial_start_times = f['intervals/trials/start_time'][:]
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. The trajectory explicitly checked the trial-table stim onset/duration values against `photostim_start_times` and confirmed that they encode the per-trial photostim window.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial with stimulation, the script converts the onset and offset from trial-start-relative time into go-cue-relative time, then marks bins whose centers lie in `[on, off)` as 1 and the rest as 0. Non-stim trials stay all-zero.

ii.
```python
photostim_on_rel = np.full(n_trials, np.nan)
photostim_off_rel = np.full(n_trials, np.nan)
for i in range(n_trials):
    if photostim_onset_str[i] != 'N/A':
        onset_from_trial_start = float(photostim_onset_str[i])
        duration = float(photostim_dur_str[i])
        go_from_trial_start = go_times[i] - trial_start_times[i]
        photostim_on_rel[i] = onset_from_trial_start - go_from_trial_start
        photostim_off_rel[i] = onset_from_trial_start + duration - go_from_trial_start
...
photostim_binary = ((bin_centers >= on_t) & (bin_centers < off_t)).astype(float)
```

iii. The trajectory says the photostim feature should be a binary time series built from the actual timestamps, and it explicitly chose to trust the data values rather than paper prose about "late delay."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Alignment is achieved by expressing stim onset and offset relative to the go cue and then comparing those times against the same go-cue-relative `bin_centers` used for neural data.

ii.
```python
go_from_trial_start = go_times[i] - trial_start_times[i]
photostim_on_rel[i] = onset_from_trial_start - go_from_trial_start
photostim_off_rel[i] = onset_from_trial_start + duration - go_from_trial_start
...
photostim_binary = ((bin_centers >= on_t) & (bin_centers < off_t)).astype(float)
```

iii. The trajectory explicitly states that photostim should be "aligned with go-cue-relative time" by converting from trial-start offsets to go-relative offsets.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read directly from a lick-direction field. Instead, it is derived from `trial_instruction` and `outcome`.

ii.
```python
out = outcome[trial_idx]
instr = trial_instruction[trial_idx]
if out == 'ignore':
    choice = 2
elif out == 'hit':
    choice = 0 if instr == 'left' else 1
else:
    choice = 1 if instr == 'left' else 0
```

iii. In the trajectory the agent reasoned that hits correspond to licking the instructed side, misses to licking the opposite side, and ignores to no lick, and chose this derivation instead of using left/right lick timestamps directly.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived choice is coded as `0=left`, `1=right`, `2=no lick`, and then repeated across all 80 bins in output row 0.

ii.
```python
if out == 'ignore':
    choice = 2
elif out == 'hit':
    choice = 0 if instr == 'left' else 1
else:
    choice = 1 if instr == 'left' else 0
...
output_data = np.zeros((4, n_bins))
output_data[0, :] = choice
```

iii. The trajectory justifies the coding by tying it to outcome semantics and by noting that `ignore` should become an explicit no-lick class for decoder training.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trial-table `outcome` column.

ii.
```python
outcome = np.array([x.decode() for x in f['intervals/trials/outcome'][:]])
...
out = outcome[trial_idx]
```

iii. The trajectory notes that the NWB file already stores `'hit'`, `'miss'`, and `'ignore'`, so no further derivation was needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped to integers `ignore=0`, `miss=1`, `hit=2`, then repeated across all 80 bins in output row 1.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[out]
...
output_data[1, :] = outcome_val
```

iii. The trajectory uses the trial-table semantics directly and treats outcome as a per-trial label that is repeated across time for compatibility with the `(n_output, n_timepoints)` format.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick comes directly from the trial-table `early_lick` column.

ii.
```python
early_lick = np.array([x.decode() for x in f['intervals/trials/early_lick'][:]])
...
early_val = 0 if early_lick[trial_idx] == 'no early' else 1
```

iii. The trajectory explicitly discusses keeping early-lick trials because the decoder must predict that category, rather than excluding them as in some analyses from the paper.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The strings are mapped as `no early -> 0`, `early -> 1`, then repeated across all 80 bins in output row 2.

ii.
```python
early_val = 0 if early_lick[trial_idx] == 'no early' else 1
...
output_data[2, :] = early_val
```

iii. The trajectory provides no extra derivation beyond the need to preserve early-lick information as a decoder target.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue position is derived from the side-camera time series `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically column 1 (`tongue_y`) and column 2 (`likelihood`), together with the camera timestamps.

ii.
```python
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The trajectory explicitly inspected the side-camera tracking arrays and concluded that tongue y-position should come from the side view because only the side camera is available.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The code keeps only frames whose likelihood exceeds `0.9`, computes session-wide 40th and 60th percentiles from raw visible-frame `tongue_y` values, and then, for each trial/bin, finds the nearest video frame to the neural bin center and classifies that single frame. If the chosen frame is not visible, the bin remains class `3` (`not visible`).

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
...
visible_mask = tongue_likelihood > TONGUE_LIKELIHOOD_THRESH
if visible_mask.sum() > 0:
    y_visible = tongue_y[visible_mask]
    pct40 = np.percentile(y_visible, 40)
    pct60 = np.percentile(y_visible, 60)
...
for b, tc in enumerate(bin_centers):
    abs_t = go_t + tc
    frame_idx = np.searchsorted(tongue_ts, abs_t)
    ...
    if tongue_likelihood[frame_idx] > TONGUE_LIKELIHOOD_THRESH:
        y_val = tongue_y[frame_idx]
```

iii. The trajectory shows the agent examining the likelihood distribution and choosing a high threshold because visible frames cluster near likelihood 1 while invisible frames cluster near 0. It also noted that the per-bin tongue lookup was slow because of the nested loop over trials and bins.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Visible tongue positions are thresholded against session-wide percentiles of raw visible-frame `tongue_y`: below the 40th percentile becomes class `0`, between 40th and 60th becomes class `1`, above the 60th becomes class `2`, and bins without a visible frame become class `3`.

ii.
```python
pct40 = np.percentile(y_visible, 40)
pct60 = np.percentile(y_visible, 60)
...
if tongue_likelihood[frame_idx] > TONGUE_LIKELIHOOD_THRESH:
    y_val = tongue_y[frame_idx]
    if y_val < pct40:
        tongue_y_disc[b] = 0
    elif y_val <= pct60:
        tongue_y_disc[b] = 1
    else:
        tongue_y_disc[b] = 2
```

iii. The trajectory justifies the visibility threshold by the bimodal confidence distribution and states that the intended categories are percentile bins plus a separate "not visible" category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Alignment is approximate and bin-center-based: for each neural bin center `tc`, the code looks up the nearest camera frame to absolute time `go_t + tc` and assigns the tongue class from that one frame.

ii.
```python
for b, tc in enumerate(bin_centers):
    abs_t = go_t + tc
    frame_idx = np.searchsorted(tongue_ts, abs_t)
    frame_idx = min(frame_idx, len(tongue_ts) - 1)
    if abs(tongue_ts[frame_idx] - abs_t) > 0.01:
        if frame_idx > 0 and abs(tongue_ts[frame_idx - 1] - abs_t) < abs(tongue_ts[frame_idx] - abs_t):
            frame_idx = frame_idx - 1
```

iii. The trajectory notes that video timestamps and go cues are on the same clock, but the implemented alignment choice is to sample the nearest frame to each neural bin center rather than binning all frames inside the 50 ms interval.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three handling choices are explicit. Sessions whose `classification` field is missing/non-object are dropped entirely. If no sample onset is found before a go cue, the code imputes a default tone offset of `-1.85 s`. For tongue tracking, low-confidence or missing-visibility bins remain in class `3` (`not visible`). The code does not treat missing per-trial spike coverage as missing data; it accepts resulting zero-rate bins.

ii.
```python
if clf_raw.dtype == object:
    ...
else:
    # No classifier QC available (all NaN) – skip session
    return None
...
if np.isnan(tone_onset_rel_go[trial_idx]):
    tone_rel = -1.85
...
tongue_y_disc = np.full(n_bins, 3)
...
if tongue_likelihood[frame_idx] > TONGUE_LIKELIHOOD_THRESH:
    ...
```

iii. The trajectory explicitly debates whether to fall back to `unit_quality` when classifier QC is missing and decides not to. It also reasons from observed timing that `-1.85 s` is the typical tone-to-go interval and therefore uses that as a fallback if tone extraction fails.

## 10-a. What are the most time-consuming steps of the code?

i. The script's heaviest computations are the nested per-trial/per-unit spike histogramming and the per-trial/per-bin nearest-frame tongue lookup. The trajectory especially identifies tongue processing as slow.

ii.
```python
for trial_idx in trial_indices:
    ...
    for u, spks in enumerate(unit_spikes):
        aligned = spks - go_t
        fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
...
for b, tc in enumerate(bin_centers):
    abs_t = go_t + tc
    frame_idx = np.searchsorted(tongue_ts, abs_t)
```

iii. Near the end of the trajectory the agent explicitly says the script is "slow due to the per-bin tongue tracking lookup" and considers vectorizing that part.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the nested tongue loop over trials and bins. Other obvious candidates are the loop that searches `sample_start_ts` once per trial, the loop that converts photostim timings per trial, and the per-trial/per-unit spike histogram loop.

ii.
```python
for i in range(n_trials):
    before = sample_start_ts[sample_start_ts < go_times[i]]
...
for i in range(n_trials):
    if photostim_onset_str[i] != 'N/A':
        ...
...
for trial_idx in trial_indices:
    ...
    for u, spks in enumerate(unit_spikes):
        ...
    for b, tc in enumerate(bin_centers):
        ...
```

iii. The trajectory explicitly calls out the tongue loop as a performance problem and mentions vectorizing it; the other loops are not discussed there but are evident in the implementation.

## 10-c. What processing does the code repeat multiple times?

i. Several computations are repeated rather than cached or vectorized: `bin_centers` is recomputed every session, the code rescans `sample_start_ts` for each trial to find the last tone, photostim onset/offset conversion is looped per trial, and the nearest camera-frame search is repeated for every bin of every trial.

ii.
```python
bin_centers = get_bin_centers(T_START, T_END, BIN_WIDTH)
...
for i in range(n_trials):
    before = sample_start_ts[sample_start_ts < go_times[i]]
...
for i in range(n_trials):
    if photostim_onset_str[i] != 'N/A':
        ...
...
for b, tc in enumerate(bin_centers):
    abs_t = go_t + tc
    frame_idx = np.searchsorted(tongue_ts, abs_t)
```

iii. The trajectory does not explicitly frame these as repeated work, but it does notice the repeated tongue-bin lookups when discussing performance.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is not much large discarded computation, but there are minor unused intermediates and scaffolding: `n_units_total` is computed and never used, `all_brain_region_idx_per_session` is allocated but never populated into the output, and imports such as `json` and `sys` are unused. The more consequential issue is inefficiency rather than fully discarded analysis results.

ii.
```python
import sys
import json
...
all_brain_region_idx_per_session = []
...
n_units_total = len(classification)
```

iii. The trajectory does not justify these items; they appear to be leftovers from drafting rather than deliberate analytical steps.
