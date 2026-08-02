# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data by scanning `/app/data` for subject directories named `sub-*`, then scanning each subject directory for `.nwb` files. Each NWB file is opened with `pynwb.NWBHDF5IO`, and sessions are processed one by one in a top-level loop.

ii. <Code snippets>

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
```

```python
all_files = list_nwb_files(DATA_DIR)
for i, (subject, nwb_path) in enumerate(all_files):
    result = process_session(nwb_path, ...)
```

iii. The justification in `CONVERSION_NOTES.md` says the dataset is organized as 28 subject folders with 174 NWB session files, so the agent treated one NWB file as one session and iterated across all of them.

## 1-b. How are the data split into subjects?

i. Subjects are split by folder name and by the NWB subject metadata. The script records unique `subject_id` values per processed session, then builds `subjects` and `subject_idx` from that session-level metadata.

ii. <Code snippets>

```python
subject_id = nwb.subject.subject_id if nwb.subject else os.path.basename(nwb_path).split('_')[0]
```

```python
subjects = []
subject_idx = []
for sess in session_results:
    if sess['subject_id'] not in subjects:
        subjects.append(sess['subject_id'])
    subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The notes explicitly say the NWB files are organized by subject directories and that the target format needs subject indices per session, so the agent used the NWB subject metadata as the canonical subject split.

## 1-c. How are the data split into sessions?

i. The agent treats each NWB file as one session. Session-level outputs are returned by `process_session()` and appended as one element in the top-level `neural`, `input`, and `output` lists.

ii. <Code snippets>

```python
def process_session(nwb_path, show_processing=False, session_idx=0):
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    session_id = nwb.identifier
```

```python
for sess in session_results:
    neural.append(sess['neural_trials'])
    inputs.append(sess['input_trials'])
    outputs.append(sess['output_trials'])
```

iii. `CONVERSION_NOTES.md` says “Each NWB file = one behavioral session,” and the trajectory repeatedly refers to selecting or skipping “sessions” by selecting individual NWB files.

## 1-d. How are the data split into trials?

i. Trials are split from the NWB `trials` table. The code reads the full trial table fields, assumes one go cue per trial, and then iterates over `valid_indices` to generate one neural/input/output block per kept trial.

ii. <Code snippets>

```python
trials = nwb.trials
n_trials = len(trials)
trial_starts = trials['start_time'][:]
trial_stops = trials['stop_time'][:]
instructions = trials['trial_instruction'][:]
outcomes = trials['outcome'][:]
```

```python
go_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_times) == n_trials
...
for trial_idx in valid_indices:
    go_time = go_times[trial_idx]
```

iii. The trajectory shows the agent explored how event streams map back to the NWB trial table and settled on using the trial table plus `go_start_times` as the trial backbone.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two layers. Session selection uses “regular trials” for the paper-style performance threshold: no auto water, no free water, no photostim, no early lick, and no ignore outcomes. Trial inclusion for the converted dataset is looser: it keeps behaviorally valid trials, requires a mapped tone onset, and drops trials extending past recording coverage.

ii. <Code snippets>

```python
behav_valid = (auto_water == 0) & (free_water == 0)
regular_mask = behav_valid.copy()
for i in range(n_trials):
    if photostim_onset[i] != 'N/A':
        regular_mask[i] = False
    if early_licks[i] == 'early':
        regular_mask[i] = False
regular_mask &= (outcomes != 'ignore')
```

```python
valid_mask = behav_valid.copy()
valid_mask &= ~np.isnan(tone_onset_per_trial)
for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False
valid_indices = np.where(valid_mask)[0]
```

iii. The notes say the agent intentionally kept early-lick, ignore, and stimulation trials for decoder outputs/inputs, while still using the paper’s regular-trial definition for session-level performance filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` after unit filtering, with unit metadata from `classification`, `anno_name`, and `obs_intervals`.

ii. <Code snippets>

```python
units = nwb.units
classification = units['classification'][:]
anno_names = units['anno_name'][:]
spike_times_all = units['spike_times']
obs_intervals = units['obs_intervals'][good_indices_units[0]]
```

iii. The notes and trajectory state that the NWB data stores absolute spike times, and that the reference code’s relative-times representation must therefore be recreated by subtracting each trial’s go cue during binning.

## 2-b. How is the `neural` data processed?

i. The agent bins each good unit’s absolute spike times into non-overlapping 50 ms bins over `[-2.5, 1.5)` s relative to the go cue, then divides counts by bin width to convert to Hz. This produces one `(n_neurons, 80)` firing-rate matrix per trial.

ii. <Code snippets>

```python
BIN_WIDTH = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)
```

```python
mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
spk_window = spk[mask]
bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
np.add.at(fr[i], bin_idx, 1)
fr /= bin_width
```

iii. `CONVERSION_NOTES.md` says the reference code used a sliding histogram in the original analysis, but the agent switched to 50 ms bins because the decoder task explicitly required 50 ms bins over the narrower window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `classification == 'good'` and `anno_name` is non-empty. Sessions with no such units are skipped.

ii. <Code snippets>

```python
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]
if len(good_indices_units) == 0:
    return None
```

iii. The notes say the agent interpreted NWB `classification == 'good'` as equivalent to the paper’s classifier-based QC pass and required histology annotation availability via non-empty `anno_name`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial’s neural data is aligned to go cue onset. The code uses absolute `go_start_times` timestamps and defines the binning window relative to that time.

ii. <Code snippets>

```python
go_times = be.time_series['go_start_times'].timestamps[:]
...
for trial_idx in valid_indices:
    go_time = go_times[trial_idx]
    fr = compute_firing_rates_vectorized(
        spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
    )
```

iii. Both the task instructions and the notes emphasize go-cue alignment, and the trajectory repeatedly mentions subtracting each trial’s go cue from absolute spike times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 50 ms bins and no secondary rebinning step. The binning is done directly from spike times into 80 non-overlapping bins.

ii. <Code snippets>

```python
BIN_WIDTH = 0.05  # 50 ms bins (decoder task spec)
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
```

iii. The notes explicitly call this a deliberate departure from the original 40 ms / 3.4 ms sliding-histogram preprocessing because the decoder specification overrode that detail.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents.sample_start_times` and `BehavioralEvents.go_start_times`, plus the trial table `start_time` to decide which sample events belong to each trial.

ii. <Code snippets>

```python
sample_starts = be.time_series['sample_start_times'].timestamps[:]
go_times = be.time_series['go_start_times'].timestamps[:]
trial_starts = trials['start_time'][:]
```

```python
in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
```

iii. The trajectory shows the agent explored the mismatch between the number of `sample_start_times` and trials, then decided to map sample events back into trials using trial start and go-cue windows.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the script picks the last `sample_start` event occurring between trial start and that trial’s go cue, converts it to go-cue-relative time, and then computes time since tone onset at each bin center. The result is a continuous ramp-like signal over the 80 bins.

ii. <Code snippets>

```python
tone_onset_per_trial = np.full(n_trials, np.nan)
for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
    if len(in_trial) > 0:
        tone_onset_per_trial[i] = in_trial[-1]
```

```python
tone_time = tone_onset_per_trial[trial_idx]
tone_relative = tone_time - go_time
time_from_tone = bin_centers - tone_relative
```

iii. The code comment says it uses the last sample onset “in case of replays from early licking.” The trajectory shows earlier uncertainty about first-versus-last sample onset, so this was a conscious but somewhat unsettled choice.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled at the same 80 go-cue-centered bin centers used for the neural data, so each neural bin and each time-from-tone value share the same per-trial temporal grid.

ii. <Code snippets>

```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
fr = compute_firing_rates_vectorized(...)
time_from_tone = bin_centers - tone_relative
```

iii. The notes describe the decoder format as requiring common time bins across `neural` and `input`, and the code implements that by computing all time-varying inputs from the same `bin_centers`.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `trials['photostim_onset']`, `trials['photostim_duration']`, `trials['start_time']`, and `BehavioralEvents.go_start_times`.

ii. <Code snippets>

```python
photostim_onset = trials['photostim_onset'][:]
photostim_duration = trials['photostim_duration'][:]
trial_starts = trials['start_time'][:]
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The trajectory includes an explicit check where the agent compared trial-table photostim values to `BehavioralEvents.photostim_start_times` and concluded the onset field is relative to trial start.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The code converts per-trial photostim onset and duration into absolute time using trial start, converts that interval into go-cue-relative time, and then fills a binary vector whose entries are `1` when the bin center falls inside the stimulation interval.

ii. <Code snippets>

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

iii. The notes and trajectory both say the agent specifically verified the reference frame of `photostim_onset` before settling on this conversion.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is represented on the same 80 go-cue-centered bins as the neural activity. Alignment is done by comparing the photostimulation interval to the shared `bin_centers`.

ii. <Code snippets>

```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
if ps_onset_rel <= bc < ps_end_rel:
    photostim[b] = 1.0
```

iii. This follows the same alignment rule used for the other time-varying streams, and the notes explicitly describe photostimulation as a binary time series over the decoder bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent derives choice from the trial instruction and outcome, not from raw lick-event timestamps. Hits map to the instructed side, misses map to the opposite side, and ignores are assigned the instructed side.

ii. <Code snippets>

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

iii. `CONVERSION_NOTES.md` says the agent planned exactly this mapping, including assigning ignore trials to the instructed direction because there is no actual lick choice available in those trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The processing is a rule-based categorical mapping: `left/right instruction` plus `hit/miss/ignore outcome` becomes a binary label `0/1`, and then that label is repeated across all 80 bins for the trial.

ii. <Code snippets>

```python
output_data = np.array([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_val, dtype=np.int64),
    tongue_y_trial.astype(np.int64),
], dtype=np.int64)
```

iii. The notes frame this as a decoder-format decision rather than a direct replay of the paper code: a per-trial categorical choice label had to exist even for ignore trials, so the agent imputed it from task instruction.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from `trials['outcome']`.

ii. <Code snippets>

```python
outcomes = trials['outcome'][:]  # 'hit', 'miss', 'ignore'
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
```

iii. The notes describe this as a direct mapping from the NWB trial table to the decoder’s required `ignore/miss/hit` categorical ordering.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The string-valued NWB outcome is mapped to integer classes `ignore=0`, `miss=1`, `hit=2`, and that scalar is repeated across all 80 bins for each trial.

ii. <Code snippets>

```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. The notes say this was chosen to match the decoder specification exactly, even though the paper’s internal representations used different trial-analysis structures.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is not computed at all. This script has no distance-to-reward-zone output; instead it makes `Outcome` a trial-constant vector repeated across the same 80 bins as the neural data.

ii. <Code snippets>

```python
output_data = np.array([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_val, dtype=np.int64),
    tongue_y_trial.astype(np.int64),
], dtype=np.int64)
```

iii. Neither the notes nor the trajectory mention any reward-zone-distance computation. This appears to be a template mismatch in the evaluation questions rather than something the agent attempted.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from `trials['early_lick']`.

ii. <Code snippets>

```python
early_licks = trials['early_lick'][:]  # 'early' or 'no early'
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
```

iii. The notes say the agent deliberately retained early-lick trials in the converted dataset because early lick itself is one of the decoder outputs.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The NWB string label is converted to a binary categorical variable: `no early -> 0`, `early -> 1`. That value is then repeated across all 80 bins in the trial.

ii. <Code snippets>

```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
np.full(N_BINS, early_val, dtype=np.int64)
```

iii. The notes explicitly list early lick as a direct mapping rather than something inferred from events.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries['Camera0_side_TongueTracking']`, using column 1 as y-position and column 2 as likelihood.

ii. <Code snippets>

```python
tongue_data = bts.time_series['Camera0_side_TongueTracking'].data[:]
tongue_ts = bts.time_series['Camera0_side_TongueTracking'].timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The trajectory includes a direct exploration step where the agent inspected the tongue tracking array shape, timestamps, and likelihood column before implementing this mapping.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent computes per-session percentile thresholds from frames with likelihood above 0.5 when enough such frames exist. For each trial bin, it finds the nearest tongue-tracking frame to the go-cue-aligned bin center and uses that frame’s y-value.

ii. <Code snippets>

```python
tongue_visible = tongue_likelihood > 0.5
if np.sum(tongue_visible) > 100:
    visible_y = tongue_y[tongue_visible]
    p40 = np.percentile(visible_y, 40)
    p60 = np.percentile(visible_y, 60)
else:
    p40 = np.percentile(tongue_y, 40)
    p60 = np.percentile(tongue_y, 60)
```

```python
t_idx = np.searchsorted(tongue_ts, bc_abs)
t_idx = min(t_idx, len(tongue_ts) - 1)
ty = tongue_y[t_idx]
```

iii. The notes say the session-wide percentiles were intentional and that the tracking stream is sampled at video-frame resolution, so the agent aligned it back to the decoder grid by nearest timestamp.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. It is discretized per session into three categories using the 40th and 60th percentiles: `< p40 -> 0`, `p40 to < p60 -> 1`, and `>= p60 -> 2`.

ii. <Code snippets>

```python
if ty < p40:
    tongue_y_trial[b] = 0
elif ty < p60:
    tongue_y_trial[b] = 1
else:
    tongue_y_trial[b] = 2
```

iii. This matches the decoder task specification, and the notes explicitly mention using per-session 40th/60th percentiles.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. It is aligned to the same go-cue-centered 50 ms bins as the neural data by sampling the closest tongue frame to each bin center.

ii. <Code snippets>

```python
bc_abs = go_time + bin_centers[b]
t_idx = np.searchsorted(tongue_ts, bc_abs)
ty = tongue_y[t_idx]
```

iii. The notes tie this to the reference code’s general temporal alignment pattern for video-derived data, but the implementation here is a simpler nearest-frame alignment onto the neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent uses a mix of dropping and fallback defaults. Trials with missing tone onset or insufficient recording coverage are dropped; sessions with no good units or fewer than two valid trials are skipped; missing/unmapped brain annotations default to `OtherCortex`; poor tongue visibility falls back to all frames for percentile computation; sessions without tongue tracking get a constant middle-class tongue output.

ii. <Code snippets>

```python
valid_mask &= ~np.isnan(tone_onset_per_trial)
if len(good_indices_units) == 0:
    return None
if len(valid_indices) < 2:
    return None
```

```python
print(f'  WARNING: Unmapped annotation: "{anno_name}"')
return 'OtherCortex'
```

```python
else:
    tongue_y_trial = np.ones(N_BINS, dtype=np.float32)  # default to middle
```

iii. `CONVERSION_NOTES.md` documents these as pragmatic fixes to keep the conversion running, especially for rare unmapped annotations and missing or weak tongue tracking.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive parts are per-trial firing-rate computation across all neurons, per-trial/per-bin tongue alignment, and the overall session loop over all NWB files. The notes estimate about 5 to 10 seconds per session and about 30 minutes for the full run.

ii. <Code snippets>

```python
for trial_idx in valid_indices:
    fr = compute_firing_rates_vectorized(...)
```

```python
for i, spk in enumerate(spike_times_list):
    ...
    np.add.at(fr[i], bin_idx, 1)
```

```python
for b in range(N_BINS):
    t_idx = np.searchsorted(tongue_ts, bc_abs)
```

iii. The notes explicitly call out runtime estimates and the trajectory comments on per-session conversion speed while deciding whether a full run was feasible.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops could have been vectorized: trial-to-tone mapping, session masks over trials, photostim bin filling, tongue-frame lookup per bin, and the per-neuron spike binning loop. The “vectorized” firing-rate helper is still neuron-looped.

ii. <Code snippets>

```python
for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
```

```python
for i in range(n_trials):
    if photostim_onset[i] != 'N/A':
        regular_mask[i] = False
```

```python
for b in range(N_BINS):
    if ps_onset_rel <= bc < ps_end_rel:
        photostim[b] = 1.0
```

iii. The code structure shows the agent prioritized clarity and debugging over full vectorization, and the notes mention performance was acceptable enough for the full run.

## 10-c. What processing does the code repeat multiple times?

i. The script repeatedly scans trial-level event arrays, repeatedly bins the same session’s spike trains separately for each trial, and repeatedly searches tongue timestamps per bin per trial. It also recomputes session and output summary statistics after conversion.

ii. <Code snippets>

```python
for trial_idx in valid_indices:
    fr = compute_firing_rates_vectorized(
        spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
    )
```

```python
for b in range(N_BINS):
    t_idx = np.searchsorted(tongue_ts, bc_abs)
```

```python
for sess_outputs in data['output']:
    for trial_out in sess_outputs:
        all_choices.append(int(trial_out[0, 0]))
```

iii. The trajectory shows the agent focused more on correctness and getting the decoder to validate than on removing repeated work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes optional plotting, extensive console summaries, and some intermediate/session metadata that are not used by downstream decoder training. It also computes a few unused values, such as `bin_edges_start` and `bin_edges_end` in the firing-rate helper.

ii. <Code snippets>

```python
bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width
bin_edges_end = bin_edges_start + bin_width
```

```python
if args.show_processing and len(session_results) <= 2:
    make_processing_plots(result, ...)
```

```python
print(f'  Neurons per region:')
for region in sorted(all_region_counts.keys()):
    print(f'    {region}: {all_region_counts[region]}')
```

iii. The notes emphasize validation and sanity checking, so the agent intentionally kept extra reporting and plotting machinery even though it is not part of the final converted dataset used by the decoder.
