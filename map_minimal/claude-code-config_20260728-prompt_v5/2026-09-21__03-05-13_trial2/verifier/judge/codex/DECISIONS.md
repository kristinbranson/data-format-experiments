# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script finds all NWB files with a sorted glob under `/app/data/sub-*/sub-*.nwb`, then opens each file directly with `h5py`. Inside each file it reads the trial table, behavioral event timestamps, behavioral time series, units table, and electrode metadata from raw HDF5 paths rather than using `pynwb`.

ii. 
```python
nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))
...
with h5py.File(fpath, 'r') as f:
    trials = f['intervals/trials']
    ...
    go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
    sample_starts = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
    ...
    units = f['units']
```

iii. In the trajectory, the agent first explored the dataset layout and concluded that the dataset consists of one NWB file per session under subject folders. It explicitly chose direct HDF5 access after inspecting the NWB structure and listed the key groups it needed: trials, behavioral events, tongue tracking, units, and electrodes.

## 1-b. How are the data split into subjects (mice)?

i. The AI identifies a subject from the NWB filename prefix, e.g. `sub-440956`, not from the NWB subject metadata. It builds `subjects` incrementally in first-seen order as sessions are processed and computes `subject_idx` by looking up each session's parsed `subject_id`.

ii. 
```python
basename = os.path.basename(fpath)
subject_id = basename.split('_')[0]
...
if result['subject_id'] not in all_subjects:
    all_subjects.append(result['subject_id'])
...
'subjects': all_subjects,
'subject_idx': np.array([all_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64),
```

iii. In the trajectory, the agent noted that subject directories are named `sub-<id>` and treated the filename/folder subject token as sufficient to define mice. It did not mention using `nwb.subject.subject_id`; instead it kept using the file naming scheme it had already inspected in `/app/data`.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. No additional grouping is done. The per-session identifier stored in the returned dict is the NWB basename (`session_name`), and the output session order follows the sorted NWB file list.

ii. 
```python
nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))
...
return {
    ...
    'session_name': basename,
}
```

iii. In the trajectory, the agent repeatedly described the data as “one NWB per session” and based its whole conversion loop on iterating files. That was its rationale for using file boundaries as session boundaries.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table rows as trials. It reads the trial table from `intervals/trials`, counts rows using the `id` dataset, and uses integer trial indices into that table. Trial-level event times like `go_times` are then indexed by those trial indices.

ii. 
```python
trials = f['intervals/trials']
n_trials = len(trials['id'][:])
...
trial_indices = np.where(trial_mask)[0]
...
go_t = go_times[trial_idx]
```

iii. In the trajectory, the agent stated that the NWB files contain a trial table with outcome, early lick, instruction, and photostim information, so it treated those rows as the natural behavioral trials rather than reconstructing trials from event streams.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters that differ from the reference solution. It drops whole sessions unless control-trial performance exceeds 65% and there are at least 50 correct left and 50 correct right control trials. Within kept sessions, it removes `auto_water` and `free_water` trials, then further removes trials whose go-cue-centered neural window falls outside the global minimum/maximum spike time range across good units.

ii. 
```python
is_control = (photostim_onset_raw == 'N/A') & (auto_water == 0) & (free_water == 0)
is_not_early = (early_licks == 'no early')
control_regular = is_control & is_not_early
...
if performance < MIN_PERFORMANCE:
    return None
...
if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    return None
...
trial_mask = (auto_water == 0) & (free_water == 0)
trial_indices = np.where(trial_mask)[0]
...
valid_trial_mask = np.array([
    (go_times[ti] - TIME_BEFORE >= min_spike_time - 1.0) and
    (go_times[ti] + TIME_AFTER <= max_spike_time + 1.0)
    for ti in trial_indices
])
trial_indices = trial_indices[valid_trial_mask]
```

iii. In the trajectory, the agent justified these filters from the methods text. It reasoned that session-level performance criteria from the paper should still apply, but that early-lick and ignore trials should be retained because `early_lick` and `outcome` are decoder outputs. After seeing many all-zero neural trials, it added the global spike-range filter as a fix for trials outside the recording span.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the units spike-time arrays `units['spike_times']` and `units['spike_times_index']`, together with go-cue timestamps `go_start_times`. It first restricts units to those whose `unit_quality` equals `'good'`.

ii. 
```python
units = f['units']
unit_quality = np.array([x.decode() if isinstance(x, bytes) else str(x)
                        for x in units['unit_quality'][:]])
good_indices = np.where(unit_quality == 'good')[0]
...
all_spike_times = units['spike_times'][:]
spike_times_index = units['spike_times_index'][:]
...
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. In the trajectory, the agent described spike times as absolute session-time timestamps and the go cue as the alignment event. It also explicitly chose to filter units using the stored “good” quality label rather than reconstructing the QC classifier.

## 2-b. How is the `neural` data processed?

i. The AI converts spike times into per-trial firing rates in 50 ms bins. For each retained unit and each retained trial, it builds absolute bin edges by adding the go-cue time to the shared relative bin edges, uses `np.searchsorted` on that unit's spike times, differences adjacent edge counts, and divides by bin width.

ii. 
```python
def compute_firing_rates_all_trials(spike_times, spike_idx, unit_indices, go_times, trial_indices):
    ...
    rates_all = np.zeros((n_neurons, n_trials, N_BINS), dtype=np.float32)
    for i, st in enumerate(unit_spikes):
        ...
        for t_idx in range(n_trials):
            go_t = go_subset[t_idx]
            abs_edges = go_t + BIN_EDGES
            edge_counts = np.searchsorted(st, abs_edges)
            rates_all[i, t_idx, :] = np.diff(edge_counts) / BIN_SIZE
```

iii. In the trajectory, the agent said it wanted the same general firing-rate representation as the paper code: spike counts per 50 ms bin divided by bin width, with no smoothing. It later optimized an earlier slower histogram approach into this `searchsorted`-based implementation after noticing runtime issues.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `unit_quality == 'good'` and drops sessions with no such units. It does not use the newer classifier verdict in `classification`, nor does it apply per-metric QC thresholds.

ii. 
```python
unit_quality = np.array([x.decode() if isinstance(x, bytes) else str(x)
                        for x in units['unit_quality'][:]])
good_indices = np.where(unit_quality == 'good')[0]
if len(good_indices) == 0:
    print(f"  Skip: no good units")
    return None
```

iii. In the trajectory, the agent repeatedly stated that “good vs multi” looked like the relevant QC split and described this as matching the paper's classifier-based QC, even though it was actually using `unit_quality` rather than the classifier verdict field used by the reference solution.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset. It uses a fixed relative window from -2.5 s to +1.5 s and adds each trial's `go_time` to those relative edges to get absolute bin edges.

ii. 
```python
TIME_BEFORE = 2.5
TIME_AFTER = 1.5
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
...
go_t = go_subset[t_idx]
abs_edges = go_t + BIN_EDGES
```

iii. In the trajectory, the agent explicitly identified go cue onset as the alignment event from the instructions and described all timestamps as being in the same absolute session-time clock, so alignment only required shifting the shared bin grid by each trial's go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms bins and a 4.0 s window, giving 80 bins per trial. It does not apply any later rebinning or smoothing.

ii. 
```python
BIN_SIZE = 0.05
N_BINS = int(round((TIME_BEFORE + TIME_AFTER) / BIN_SIZE))  # 80
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. In the trajectory, the agent repeatedly checked that the requested `[-2.5, 1.5]` window with 50 ms bins produces 80 time bins and used that as a fixed design constraint.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times` and `go_start_times`. For each trial it chooses the last sample-start timestamp occurring after the previous trial's go cue and before the current trial's go cue, treating that as the tone onset that led into the final delay/go sequence.

ii. 
```python
sample_starts = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
...
tone_onsets = np.full(n_trials, np.nan)
for i in range(n_trials):
    go_t = go_times[i]
    lower = go_times[i - 1] if i > 0 else 0.0
    mask_s = (sample_starts > lower) & (sample_starts < go_t)
    candidates = sample_starts[mask_s]
    if len(candidates) > 0:
        tone_onsets[i] = candidates[-1]
```

iii. In the trajectory, the agent reasoned that early licks replay the sample epoch, so there can be multiple sample starts per behavioral trial. It explicitly decided to use the last sample start before the go cue as the behaviorally relevant tone onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI expresses the value at each neural bin as continuous elapsed time from tone onset. It computes the tone time relative to go cue, then subtracts that offset from the shared go-cue-centered bin centers. If no tone onset is found, it falls back to a hard-coded canonical 1.85 s tone-to-go interval.

ii. 
```python
if np.isnan(tone_t):
    time_from_tone = BIN_CENTERS + 1.85
else:
    tone_rel = tone_t - go_t
    time_from_tone = BIN_CENTERS - tone_rel
```

iii. In the trajectory, the agent described this feature as “for each time bin relative to the go cue, subtract the tone onset time to get a continuous signal.” It justified the last-sample choice as the successful presentation leading to the final go cue, and later sanity-checked the resulting range against the expected 1.85 s sample-plus-delay timing.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI uses the same 80 go-cue-centered bin centers as the neural data. `time_from_tone` is computed directly on that bin grid, so each value corresponds to the same timepoint as the firing-rate bin at that index.

ii. 
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
...
time_from_tone = BIN_CENTERS - tone_rel
...
inputs = np.stack([time_from_tone.astype(np.float32), photostim_vec], axis=0)
```

iii. In the trajectory, the agent explicitly framed `time_from_tone` as a transformation of the neural bin centers rather than a separately resampled signal, so alignment was built into the definition.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The actual photostimulation input stream is derived from the behavioral-event timestamp series `photostim_start_times` and `photostim_stop_times` when those series are present. Separately, the trial-table field `photostim_onset` is only used to label control trials during session filtering.

ii. 
```python
photostim_onset_raw = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                for x in trials['photostim_onset'][:]])
...
ps_start_abs = np.array([])
ps_stop_abs = np.array([])
if 'photostim_start_times' in f['acquisition/BehavioralEvents']:
    ps_ts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps']
    if ps_ts.shape[0] > 0:
        ps_start_abs = ps_ts[:]
        ps_stop_abs = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
```

iii. In the trajectory, the agent first noticed that `photostim_onset` is stored in trial-relative form, but then decided it could use the absolute start/stop event streams “for finer precision” to build the per-bin photostimulation signal.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI creates a binary 80-bin vector per trial. For each photostimulation episode that overlaps the trial window, it converts the start and stop times to go-cue-relative coordinates and sets bins to 1 when the bin center lies inside that interval.

ii. 
```python
photostim_vec = np.zeros(N_BINS, dtype=np.float32)
...
for ps_s, ps_e in zip(ps_start_abs, ps_stop_abs):
    if ps_e < win_start or ps_s > win_end:
        continue
    ps_s_rel = ps_s - go_t
    ps_e_rel = ps_e - go_t
    on_mask = (BIN_CENTERS >= ps_s_rel) & (BIN_CENTERS < ps_e_rel)
    photostim_vec[on_mask] = 1.0
```

iii. In the trajectory, the agent justified this as the decoder-required representation: a time-varying “whether photostimulation is on” signal rather than a per-trial flag. It also cross-checked the expected delay-epoch timing from the methods text.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation to the neural data by converting each stimulation interval from absolute session time into coordinates relative to the same go cue used for neural binning, then comparing those intervals against the shared bin centers.

ii. 
```python
win_start = go_t + BIN_EDGES[0]
win_end = go_t + BIN_EDGES[-1]
...
ps_s_rel = ps_s - go_t
ps_e_rel = ps_e - go_t
on_mask = (BIN_CENTERS >= ps_s_rel) & (BIN_CENTERS < ps_e_rel)
```

iii. In the trajectory, the agent described spikes, go cues, and photostim timestamps as sharing the same absolute session clock, so the only alignment step it considered necessary was subtraction of the trial's go time.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the per-trial `outcome` and `trial_instruction` fields. It does not use an explicit lick-direction variable.

ii. 
```python
outcome = outcomes[trial_idx]
instruction = instructions[trial_idx]
...
if outcome == 'ignore':
    choice = 2  # no lick
elif outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
```

iii. In the trajectory, the agent reasoned that choice is not stored directly but can be inferred: a hit means the instructed side was chosen, a miss means the opposite side, and ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0 = left`, `1 = right`, `2 = no lick`, and repeats that single per-trial value across all 80 time bins in output row 0.

ii. 
```python
outputs = np.zeros((4, N_BINS), dtype=np.int64)
outputs[0, :] = choice
...
'output_values': [
    ['left', 'right', 'no_lick'],
    ...
]
```

iii. In the trajectory, the agent described the decoder output as left/right/no-lick choice and chose to broadcast the trial label across time so all outputs share the same `(n_output, n_timepoints)` shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI takes outcome directly from the trials-table `outcome` field.

ii. 
```python
outcomes = np.array([x.decode() if isinstance(x, bytes) else str(x)
                    for x in trials['outcome'][:]])
...
outcome = outcomes[trial_idx]
```

iii. In the trajectory, the agent treated `outcome` as an explicit trial-level category already present in the dataset and used it directly as one of the requested decoder outputs.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps the strings to integer codes in the order `hit = 0`, `miss = 1`, `ignore = 2`, stores that value in output row 1, and repeats it across all 80 bins.

ii. 
```python
outcome_val = {'hit': 0, 'miss': 1, 'ignore': 2}.get(outcome, 2)
...
outputs[1, :] = outcome_val
...
'output_values': [
    ['hit', 'miss', 'ignore'],
    ...
]
```

iii. In the trajectory, the agent framed outcome as a categorical decoder output and chose a simple integer encoding consistent with its `output_values` list, without attempting to match the reference solution's code order.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The AI derives early lick directly from the trials-table `early_lick` field.

ii. 
```python
early_licks = np.array([x.decode() if isinstance(x, bytes) else str(x)
                       for x in trials['early_lick'][:]])
```

iii. In the trajectory, the agent specifically noted that early lick is already a trial-level label and that those trials must be retained because `early_lick` is one of the decoder outputs.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early` to `0` and any other value (effectively `early`) to `1`, writes that value into output row 2, and repeats it across all 80 bins.

ii. 
```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
...
outputs[2, :] = early_val
```

iii. In the trajectory, the agent justified keeping and decoding this field because the instructions explicitly require early-lick prediction, even though some analysis pipelines in the paper excluded such trials.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from `Camera0_side_TongueTracking`: timestamps plus the `data` array, using column 1 as `tongue_y` and column 2 as the confidence/visibility score.

ii. 
```python
tt = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
tongue_ts = tt['timestamps'][:]
tdata = tt['data'][:]
tongue_y = tdata[:, 1]
tongue_conf = tdata[:, 2]
```

iii. In the trajectory, the agent identified the tongue tracking series as a side-camera signal sampled near 300 Hz and explicitly described it as `x`, `y`, and confidence values.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first computes two session-wide percentile thresholds from raw visible-frame `tongue_y` values using a high confidence threshold (`> 0.9`). Then, for each trial/bin, it finds the nearest tongue frame to the bin center, requires that frame to be within 5 ms and above the same confidence threshold, and uses that frame's y value for discretization. If no acceptable frame is found, the bin remains class 3.

ii. 
```python
DLC_CONFIDENCE_THRESHOLD = 0.9
...
vis = tongue_conf > DLC_CONFIDENCE_THRESHOLD
if np.any(vis):
    tongue_y_p40 = np.percentile(tongue_y[vis], 40)
    tongue_y_p60 = np.percentile(tongue_y[vis], 60)
...
idx_all = np.searchsorted(tongue_ts, t_abs_all)
...
close_enough = best_dist < 0.005
confident = tongue_conf[best_idx] > DLC_CONFIDENCE_THRESHOLD
valid = close_enough & confident
```

iii. In the trajectory, the agent examined the confidence distribution, observed that only about 10.5% of frames exceed 0.9, and concluded that a high DeepLabCut-style confidence threshold was appropriate because the tongue is visible only during licking. It also decided against averaging frames within bins, opting instead for nearest-frame sampling.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses session-wide 40th and 60th percentiles of raw visible-frame `tongue_y` values as cut points. It assigns class 0 below the 40th percentile, class 1 between the 40th and 60th percentiles, class 2 at or above the 60th percentile, and class 3 when the tongue is not confidently visible near that bin center.

ii. 
```python
if np.any(vis):
    tongue_y_p40 = np.percentile(tongue_y[vis], 40)
    tongue_y_p60 = np.percentile(tongue_y[vis], 60)
...
tongue_y_disc[valid & (y_vals < tongue_y_p40)] = 0
tongue_y_disc[valid & (y_vals >= tongue_y_p40) & (y_vals < tongue_y_p60)] = 1
tongue_y_disc[valid & (y_vals >= tongue_y_p60)] = 2
```

iii. In the trajectory, the agent said it wanted to discretize tongue position per session as instructed, but used percentiles of visible raw frames rather than percentiles of 50 ms bin means. It justified class 3 as the explicit not-visible state.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue y-position to neural data by sampling the nearest tongue frame to each neural bin center. It converts each bin center to absolute session time (`go_t + BIN_CENTERS`), finds the nearest timestamp in the tongue-tracking series, and only accepts that frame if it is within 5 ms of the bin center.

ii. 
```python
t_abs_all = go_t + BIN_CENTERS
idx_all = np.searchsorted(tongue_ts, t_abs_all)
idx_all = np.clip(idx_all, 0, len(tongue_ts) - 1)
idx_prev = np.clip(idx_all - 1, 0, len(tongue_ts) - 1)
dist_cur = np.abs(tongue_ts[idx_all] - t_abs_all)
dist_prev = np.abs(tongue_ts[idx_prev] - t_abs_all)
best_idx = np.where(dist_prev < dist_cur, idx_prev, idx_all)
best_dist = np.minimum(dist_cur, dist_prev)
close_enough = best_dist < 0.005
```

iii. In the trajectory, the agent explicitly chose bin-center sampling instead of binning all camera frames inside each 50 ms interval. Its reasoning was that nearest-frame alignment was a simple way to match the neural timeline while enforcing a visibility/temporal proximity check.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing-data cases heuristically. Missing tone-onset assignments fall back to a canonical 1.85 s tone-to-go gap. Missing or low-confidence tongue measurements become class 3 (`not visible`). Missing detailed unit annotations fall back to electrode target regions. Missing photostim event streams leave the photostim vector all zeros. Trials outside the inferred global spike-time span are dropped.

ii. 
```python
if np.isnan(tone_t):
    time_from_tone = BIN_CENTERS + 1.85
...
if 'Camera0_side_TongueTracking' in f['acquisition/BehavioralTimeSeries']:
    ...
vis = tongue_conf > DLC_CONFIDENCE_THRESHOLD
...
tongue_y_disc = np.full(N_BINS, 3, dtype=np.int64)
...
if 'photostim_start_times' in f['acquisition/BehavioralEvents']:
    ...
region = get_unit_brain_region(anno_names[uid], target_region)
...
valid_trial_mask = np.array([
    (go_times[ti] - TIME_BEFORE >= min_spike_time - 1.0) and
    (go_times[ti] + TIME_AFTER <= max_spike_time + 1.0)
    for ti in trial_indices
])
```

iii. In the trajectory, the agent treated several anomalies as practical edge cases to patch over. It explicitly added the spike-range filter after finding sessions with long blocks of all-zero neural trials, and it chose the 1.85 s fallback because the standard sample-plus-delay timing looked consistent with many sessions.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive step in the AI code is firing-rate computation, especially the nested loop over good units and retained trials in `compute_firing_rates_all_trials`. Per-trial tongue processing and file I/O are secondary costs.

ii. 
```python
rates_all = np.zeros((n_neurons, n_trials, N_BINS), dtype=np.float32)
for i, st in enumerate(unit_spikes):
    if len(st) == 0:
        continue
    for t_idx in range(n_trials):
        go_t = go_subset[t_idx]
        abs_edges = go_t + BIN_EDGES
        edge_counts = np.searchsorted(st, abs_edges)
        rates_all[i, t_idx, :] = np.diff(edge_counts) / BIN_SIZE
```

iii. In the trajectory, the agent explicitly diagnosed the original nested per-trial/per-neuron spike binning as the runtime bottleneck and rewrote it into the current `searchsorted` approach. It still regarded this step as the dominant cost during the full conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain vectorizable: the per-trial loop nested inside the per-unit neural loop, the tone-onset search loop over trials, the per-trial scan over all photostim episodes, and the per-trial output-construction loop.

ii. 
```python
for i in range(n_trials):
    go_t = go_times[i]
    lower = go_times[i - 1] if i > 0 else 0.0
    mask_s = (sample_starts > lower) & (sample_starts < go_t)
    candidates = sample_starts[mask_s]
```

```python
for i, st in enumerate(unit_spikes):
    ...
    for t_idx in range(n_trials):
        go_t = go_subset[t_idx]
        abs_edges = go_t + BIN_EDGES
```

```python
for ps_s, ps_e in zip(ps_start_abs, ps_stop_abs):
    ...
for t_i, trial_idx in enumerate(trial_indices):
    ...
```

iii. In the trajectory, the agent recognized this directly: it killed an even slower implementation after realizing it was effectively doing too many small histogram calls, then described the remaining neural loop as the next thing that would matter if further optimization were needed.

## 10-c. What processing does the code repeat multiple times?

i. The AI code repeats several computations instead of doing them once in a more global form. It rebuilds absolute bin edges separately for every unit-trial pair, repeatedly scans all photostim intervals for every trial, repeatedly searches for per-trial tone onsets, and recomputes `all_subjects.index(...)` for every session while building `subject_idx`.

ii. 
```python
for i, st in enumerate(unit_spikes):
    ...
    for t_idx in range(n_trials):
        go_t = go_subset[t_idx]
        abs_edges = go_t + BIN_EDGES
```

```python
for i in range(n_trials):
    ...
    candidates = sample_starts[mask_s]
```

```python
for ps_s, ps_e in zip(ps_start_abs, ps_stop_abs):
    ...
'subject_idx': np.array([all_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64),
```

iii. In the trajectory, the agent's performance debugging focused on the repeated per-trial neural computations. It did not fully eliminate repeated work; instead it stopped after making the code fast enough to finish and pass decoder validation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI code does some work that is not preserved in the final dataset. It parses electrode hemisphere (`side`) and then discards it, reads trial-level photostim onset strings mostly for session filtering rather than for the final input representation, and computes/prints several summary counts used only for logging. It also builds session-local variables such as `n_good_units` and `session_name` that are not stored in the final output dictionary.

ii. 
```python
def get_electrode_target_region(location_json):
    d = json.loads(location_json)
    brain_regions = d['brain_regions']
    parts = brain_regions.split()
    side = parts[0]
    region_name = ' '.join(parts[1:])
    major = TARGET_TO_REGION.get(region_name, region_name)
    return side, major
```

```python
photostim_onset_raw = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                for x in trials['photostim_onset'][:]])
...
print(f"  -> {result['n_good_units']} units, {result['n_trials']} trials")
```

iii. In the trajectory, the agent described several of these computations as pragmatic helpers for filtering, debugging, or sanity checks rather than as intended contents of the converted dataset. It kept them because they helped it get the script to run and validate.
