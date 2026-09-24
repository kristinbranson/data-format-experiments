# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed all subject NWB files, sorted them, opened each with `pynwb.NWBHDF5IO`, processed it, and skipped files that errored or failed later filters. Thus it discovered all files but did not retain all sessions/trials.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*.nwb'))
for i, nwb_path in enumerate(nwb_files):
    try:
        result = process_session(nwb_path)
    except Exception as e:
        ...
        continue
```
```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. The trajectory says NWB files contain the behavioral events/time series, trials, and units, so the agent designed a per-file session pipeline. It reported finding 174 files and ultimately retaining 150.

## 1-b. How are the data split into subjects?

i. The subject is read from `nwb.subject.subject_id`; retained unique IDs are sorted, and every retained session receives the corresponding integer index.

ii.
```python
subject_id = str(nwb.subject.subject_id) if nwb.subject else 'unknown'
subjects = sorted(list(all_subjects))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_to_idx[s['subject_id']] for s in all_sessions], dtype=np.int64),
```

iii. The agent treated the NWB subject field as the canonical grouping identifier. The fallback `'unknown'` was defensive; the trajectory did not identify missing subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. A file either contributes one session or is discarded; output order follows the sorted path order.

ii.
```python
def process_session(nwb_path):
    """Process a single NWB session file. Returns dict or None if filtered."""
...
all_sessions.append(result)
```

iii. The agent inferred the file/session correspondence from the dataset organization. It added performance and trial-count session filters based on the method paper, yielding 150 retained sessions.

## 1-d. How are the data split into trials?

i. Trial rows and go-cue timestamps are assumed one-to-one. Trials are indexed by rows of `nwb.trials`, checked against the number of go cues, then subset by `valid_idx`. Per-trial neural/input/output arrays are built in one final loop.

ii.
```python
n_trials_total = len(trials)
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
if len(go_cue_times_all) != n_trials_total:
    io.close()
    return None
...
for t_i in range(n_trials):
    neural_trials.append(fr_all[:, t_i, :])
```

iii. The trajectory recognized `go_start_times` as the alignment event and the trials table as the source of trial labels. A mismatch causes the entire file to be dropped.

## 1-e. How are trials filtered based on quality controls?

i. A trial is retained when its go cue lies inside any `obs_intervals` interval from the first good unit. Sessions with fewer than two such trials are dropped. `free_water` trials are not explicitly removed. After this, whole sessions are filtered unless control, non-early trials have at least 65% hit rate among responded trials and at least 50 hits for each instructed side. Within accepted sessions, early, ignore, miss, hit, and photostim trials remain.

ii.
```python
for t_i in range(n_trials):
    go = go_cue_times[t_i]
    for obs_start, obs_end in obs_intervals:
        if go >= obs_start and go <= obs_end:
            valid[t_i] = True
            break
```
```python
if performance < MIN_PERFORMANCE:
    return None
if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    return None
```

iii. The agent discovered all-zero neural trials and concluded `obs_intervals` identifies trials with recordings. It retained behaviorally poor trial types because early lick and outcome are decoder targets, but applied method-paper session-selection criteria to control trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units['spike_times']` for units whose `units['classification']` is `'good'`, with `go_start_times` providing trial alignment.

ii.
```python
classifications = units['classification'][:]
good_idx = np.where(classifications == 'good')[0]
spike_times_list = [units['spike_times'][int(idx)] for idx in good_idx]
fr_all = compute_firing_rates_all(spike_times_list, go_cue_times)
```

iii. The trajectory states spike times are absolute and `classification == 'good'` is the paper/white-paper QC verdict.

## 2-b. How is the `neural` data processed?

i. Each unit's spikes are sorted. For each retained trial, spikes in the four-second window are selected, histogrammed into 80 bins, and divided by 0.05 s to produce Hz as `float32`. No smoothing or normalization is applied.

ii.
```python
st = np.sort(spike_times)
rel = st[i_lo:i_hi] - go
counts, _ = np.histogram(rel, bins=BIN_EDGES)
fr_all[u_i, t_i] = counts / BIN_WIDTH
```

iii. The agent explicitly chose non-overlapping 50-ms spike counts divided by bin width, consistent with firing-rate computation and the decoder specification.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `classification == 'good'` units are retained; a session with zero good units is dropped. No individual metric thresholds or `unit_quality` filter are used.

ii.
```python
good_idx = np.where(classifications == 'good')[0]
if len(good_idx) == 0:
    io.close()
    return None
```

iii. The agent described this as matching the QC classifier approach in the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute spike times are aligned to each absolute go-cue timestamp by extracting `[go-2.5, go+1.5)` and subtracting `go` before histogramming against relative edges.

ii.
```python
i_lo = np.searchsorted(st, go + T_START, side='left')
i_hi = np.searchsorted(st, go + T_END, side='left')
rel = st[i_lo:i_hi] - go
counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. The trajectory correctly identified the go cue as required and noted that spike/event timestamps share an absolute session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The result has 80 non-overlapping 50-ms bins from -2.5 to +1.5 s. Raw spike events are newly binned; there is no later rebinning or resampling.

ii.
```python
BIN_WIDTH = 0.05
N_BINS = int(round((T_END - T_START) / BIN_WIDTH))
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
```

iii. The agent calculated 4 s / 0.05 s = 80 bins directly from the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, `go_start_times`, and the common go-relative bin centers.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
tone_onset_rel_go[t_i] = sorted_ss[i_hi - 1] - go
time_from_tone = BIN_CENTERS - tone_onset_rel_go[t_i]
```

iii. The agent found that early licks can replay the sample and therefore selected the last sample start before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Sample starts are sorted. For each trial, the last sample start after the preceding go cue and at or before the current go cue is selected; if none is found, a default offset of -1.85 s is used. That offset is subtracted from every relative bin center.

ii.
```python
tone_onset_rel_go = np.full(n_trials, -1.85)
i_lo = np.searchsorted(sorted_ss, prev_go, side='right')
i_hi = np.searchsorted(sorted_ss, go, side='right')
if i_hi > i_lo:
    tone_onset_rel_go[t_i] = sorted_ss[i_hi - 1] - go
```

iii. The agent justified “last sample before go” by replayed epochs and used -1.85 s as the usual 0.65-s sample plus 1.2-s delay fallback.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the exact same 80 go-relative bin centers used for neural bins, shifted by that trial's tone-to-go offset.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
time_from_tone = (BIN_CENTERS - tone_onset_rel_go[t_i]).astype(np.float32)
```

iii. The trajectory says the feature should be continuous at every decoder timepoint and share the neural bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It comes from trial-table `photostim_onset`, `photostim_duration`, and `start_time`, plus each trial's go-cue timestamp.

ii.
```python
photostim_onset_all = trials['photostim_onset'][:]
photostim_duration_all = trials['photostim_duration'][:]
trial_starts_all = trials['start_time'][:]
```

iii. The agent investigated the fields and concluded onset is relative to trial start and duration defines the offset.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. `'N/A'` trials remain all zero. Otherwise onset is converted to absolute time, then to go-relative time; bin centers in the half-open on/off interval are coded 1.

ii.
```python
ps_onset_abs = trial_starts[t_i] + float(photostim_onset[t_i])
ps_onset_rel = ps_onset_abs - go_cue_times[t_i]
ps_offset_rel = ps_onset_rel + ps_dur
photostim_on_all[t_i] = ((BIN_CENTERS >= ps_onset_rel) &
                         (BIN_CENTERS < ps_offset_rel)).astype(np.float32)
```

iii. The agent chose a binary time series because the requested input asks whether light is on at every timepoint.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostim times are converted to go-relative coordinates and tested at the same bin centers as the neural data.

ii.
```python
ps_onset_rel = ps_onset_abs - go_cue_times[t_i]
photostim_on_all[t_i] = ((BIN_CENTERS >= ps_onset_rel) & ...)
```

iii. The trajectory explicitly planned to place photostimulation on the common go-cue-aligned grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the trials-table `outcome` and `trial_instruction` fields.

ii.
```python
o = outcomes[t_i]
inst = instructions[t_i]
```

iii. The agent reasoned that a hit is an instructed-side lick, a miss is the opposite-side lick, and ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Codes are 0 left, 1 right, 2 no lick. Hits map to the instructed side, misses to its opposite, ignores and unknown outcomes to no lick. The scalar class is repeated over all 80 bins.

ii.
```python
if o == 'ignore': choice = 2
elif o == 'hit': choice = 0 if inst == 'left' else 1
elif o == 'miss': choice = 1 if inst == 'left' else 0
...
np.full(N_BINS, choice, dtype=np.int64)
```

iii. This mapping was explicitly worked out in the trajectory; repetition allows all outputs to share one rectangular array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is taken directly from the trials-table `outcome` column.

ii.
```python
outcomes_all = trials['outcome'][:]
```

iii. The source already supplies the three requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to ignore=0, miss=1, hit=2; an unexpected value defaults to 0. The class is repeated over 80 bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
np.full(N_BINS, outcome_map.get(o, 0), dtype=np.int64)
```

iii. The agent followed the requested category order and later changed output dtype to integer so the decoder could index category names.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` column.

ii.
```python
early_licks_all = trials['early_lick'][:]
```

iii. The agent retained early-lick trials specifically because this flag is a requested decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` maps to 1 and every other value to 0, repeated across 80 bins.

ii.
```python
np.full(N_BINS, 1 if early_licks[t_i] == 'early' else 0, dtype=np.int64)
```

iii. The binary coding matches the requested no/yes classes; integer output was chosen for decoder compatibility.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 (y) and 2 (DLC likelihood) of `Camera0_side_TongueTracking` under `BehavioralTimeSeries`.

ii.
```python
tongue_data = tongue_ts.data[:]
tongue_timestamps = tongue_ts.timestamps[:]
tongue_y_vals = tongue_data[:, 1]
tongue_lk_vals = tongue_data[:, 2]
```

iii. The trajectory identified the tracking array as `(x, y, likelihood)` at roughly 300 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Visible frames are those with likelihood at least 0.9. Percentiles are calculated from all visible raw frames within retained trial windows. Within each 50-ms trial bin, visible y samples are averaged; bins without one remain class 3. If no visible frames exist session-wide, both thresholds become zero. If the series is missing, every bin is class 3.

ii.
```python
vis = lk >= TONGUE_LIKELIHOOD_THRESHOLD
all_visible_y.append(tongue_y_vals[i_start:i_end][vis])
p40 = np.percentile(all_vis, 40)
p60 = np.percentile(all_vis, 60)
...
mean_y = np.mean(tongue_y_vals[i_start:i_end][vis])
```

iii. The agent selected 0.9 as a common DLC convention and interpreted “over the session” as all visible frames in retained trial windows.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. A visible-bin mean below p40 is 0; p40 through p60 inclusive is 1; above p60 is 2; no visible frame is 3.

ii.
```python
if mean_y < p40:
    ty_binned[b_i] = 0
elif mean_y <= p60:
    ty_binned[b_i] = 1
else:
    ty_binned[b_i] = 2
```

iii. The cutoffs and four class meanings were taken from the task; visibility was operationalized with the 0.9 likelihood threshold.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For every go cue, the agent creates absolute camera-bin edges by adding the common relative edges, finds frame ranges with `searchsorted`, and averages frames in each interval. Thus tongue bin `b` covers the same nominal interval as neural bin `b`.

ii.
```python
abs_edges = go + BIN_EDGES
edge_indices = np.searchsorted(tongue_timestamps, abs_edges, side='left')
for b_i in range(N_BINS):
    i_start = edge_indices[b_i]
    i_end = edge_indices[b_i + 1]
```

iii. The agent used the shared absolute timestamp clock and same go-relative 50-ms edges, requiring no interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Go/trial count mismatches, no-good-unit files, too-few-valid-trial files, and failed session criteria return `None`. Any exception in a file is printed and skipped. Missing tongue tracking becomes all “not visible”; no visible tongue frames use zero thresholds. Missing tone matches use -1.85 s. Unknown subject becomes `'unknown'`, unexpected outcome becomes class 0. Files are explicitly closed on handled exits.

ii.
```python
except Exception as e:
    print(f"ERROR: {e}")
    ...
    continue
```
```python
tone_onset_rel_go = np.full(n_trials, -1.85)
...
tongue_y_per_trial = [np.full(N_BINS, 3, dtype=np.int64) for _ in range(n_trials)]
```

iii. The agent introduced fallbacks to preserve rectangular output and skip unusable sessions. Its trajectory specifically diagnosed missing recording coverage via all-zero trials and corrected that with `obs_intervals` filtering.

## 10-a. What are the most time-consuming steps of the code?

i. NWB I/O, the nested unit-by-trial spike histogram loop, and tongue processing dominate. The initial conversion was slow enough that the agent stopped it and rewrote portions; the final full conversion still performs nested loops.

ii.
```python
for u_i, spike_times in enumerate(spike_times_list):
    for t_i in range(n_trials):
        ...
        counts, _ = np.histogram(rel, bins=BIN_EDGES)
```
```python
for t_i in range(n_trials):
    ...
    for b_i in range(N_BINS):
```

iii. The trajectory explicitly called tongue per-bin work the initial bottleneck, then identified both per-bin tongue tracking and per-trial spike binning as optimization targets.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial validity has a trial-by-interval nested loop; spike rates have a unit-by-trial loop; tongue percentile gathering loops over trials and discretization loops over trials and all 80 bins; photostim and tone matching loop over trials; final assembly also loops over trials. The spike trial dimension, interval membership, photostim masks, tone lookup, and much tongue aggregation could be vectorized.

ii.
```python
for t_i in range(n_trials):
    for obs_start, obs_end in obs_intervals:
```
```python
for u_i, spike_times in enumerate(spike_times_list):
    for t_i in range(n_trials):
```

iii. The agent knew the spike and tongue loops were bottlenecks and attempted optimization, but retained these explicit nested loops for the final result.

## 10-c. What processing does the code repeat multiple times?

i. Camera windows are searched twice: once to collect percentile frames and again to bin output. Trial arrays are repeatedly traversed for validity, session QC, tone, photostim, tongue, and assembly. Spike arrays are sorted even though NWB spike times are expected to be sorted. The output is also reconstructed into new top-level lists after accumulating session dictionaries.

ii.
```python
# First pass: collect visible y values for percentiles
for t_i in range(n_trials): ...
# Second pass: discretize per bin
for t_i in range(n_trials): ...
```

iii. The two-pass tongue design follows from needing session thresholds before class assignment. The trajectory focused on runtime optimization but did not document eliminating these remaining repeated passes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It calculates/stores per-session `performance` only for filtering/logging, and basenames and summary sets only for progress/assembly. Sorting each unit's spike times may be redundant. Session dictionaries retain counts/performance temporarily, but only neural/input/output/subject/regions enter the final dataset. Exception tracebacks and extensive progress summaries do not affect conversion.

ii.
```python
return {
    ...
    'n_trials': n_trials,
    'n_neurons': len(good_idx),
    'performance': performance,
}
```
```python
st = np.sort(spike_times)
```

iii. These values support selection and diagnostics rather than downstream decoding. The trajectory used performance and counts to monitor conversion and verify the result.
