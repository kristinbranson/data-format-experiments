# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all sessions by globbing all NWB files under `/app/data/sub-*/sub-*.nwb`, sorting the file list, and opening each file with `pynwb.NWBHDF5IO`. Within each session file it reads `nwb.subject`, `nwb.trials`, `nwb.units`, and the `BehavioralEvents` acquisition group.

ii. 
```python
def process_session(nwb_path):
    """Process a single NWB session file. Returns dict or None if filtered."""
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()

    subject_id = str(nwb.subject.subject_id) if nwb.subject else 'unknown'
    trials = nwb.trials
    n_trials_total = len(trials)
    units = nwb.units

    # Get behavioral events
    be = nwb.acquisition['BehavioralEvents']
    go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
    sample_start_times = be.time_series['sample_start_times'].timestamps[:]
```

```python
def main():
    nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*.nwb'))
    print(f"Found {len(nwb_files)} NWB files")

    all_sessions = []
    ...
    for i, nwb_path in enumerate(nwb_files):
        ...
        result = process_session(nwb_path)
```

iii. In the trajectory, the agent first inspected the dataset layout and concluded that the data were one NWB file per session under subject folders. Its final summary in step 97 explicitly says it processed 174 NWB files and used NWB event, unit, and trial tables.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by `nwb.subject.subject_id`. The agent stores one subject id per kept session, then creates `subjects` as the sorted unique ids and `subject_idx` as the per-session index into that list.

ii.
```python
subject_id = str(nwb.subject.subject_id) if nwb.subject else 'unknown'
```

```python
subjects = sorted(list(all_subjects))
subject_to_idx = {s: i for i, s in enumerate(subjects)}

data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array([subject_to_idx[s['subject_id']] for s in all_sessions], dtype=np.int64),
    ...
}
```

iii. The trajectory shows the agent inspecting the subject-folder layout early on, then reporting 28 subjects in steps 90 and 97. There is no alternate grouping logic; it uses the NWB subject field directly.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. The output session order follows the sorted NWB file list, and each surviving file contributes one element to the top-level `neural`, `input`, and `output` lists.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*.nwb'))
...
for i, nwb_path in enumerate(nwb_files):
    ...
    if result is None:
        print("FILTERED")
        continue
    all_sessions.append(result)
```

```python
for sess in all_sessions:
    data['neural'].append(sess['neural'])
    data['input'].append(sess['input'])
    data['output'].append(sess['output'])
```

iii. In the trajectory, the agent repeatedly describes the dataset as “174 NWB files” and later reports “150 sessions passed filtering,” which implies that file boundaries were used as session boundaries.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table. The agent checks that the number of `go_start_times` matches the number of trial rows, then subsets all per-trial arrays with a trial-validity mask.

ii.
```python
trials = nwb.trials
n_trials_total = len(trials)
...
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
...
if len(go_cue_times_all) != n_trials_total:
    io.close()
    return None
```

```python
valid_idx = np.where(valid_trial_mask)[0]
...
go_cue_times = go_cue_times_all[valid_idx]
outcomes = outcomes_all[valid_idx]
early_licks = early_licks_all[valid_idx]
instructions = instructions_all[valid_idx]
```

iii. The trajectory shows the agent inspecting trial-table columns directly and then building the conversion around those columns rather than re-deriving trials from event boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The agent filters trials by requiring the trial’s go-cue time to fall inside `units['obs_intervals']` from the first good unit. It also requires at least 2 valid trials in a session. In addition, it applies session-level behavioral filtering: only sessions with `>65%` correct control/non-early trials and at least `50` correct left and `50` correct right control trials are kept. It does not filter `free_water` trials.

ii.
```python
def get_valid_trial_mask(obs_intervals, go_cue_times):
    """
    Determine which trials have go cues within the observation intervals.
    A trial is valid if its go cue time falls within any observation interval,
    and the full extraction window [-2.5, 1.5] around the go cue is within
    the observation range.
    """
    n_trials = len(go_cue_times)
    valid = np.zeros(n_trials, dtype=bool)

    for t_i in range(n_trials):
        go = go_cue_times[t_i]
        for obs_start, obs_end in obs_intervals:
            if go >= obs_start and go <= obs_end:
                valid[t_i] = True
                break
```

```python
obs_intervals = units['obs_intervals'][int(good_idx[0])]
valid_trial_mask = get_valid_trial_mask(obs_intervals, go_cue_times_all)
...
if len(valid_idx) < 2:
    io.close()
    return None
```

```python
is_control = np.array([po == 'N/A' for po in photostim_onset])
is_no_early = early_licks == 'no early'
is_responded = outcomes != 'ignore'
control_responded = is_control & is_no_early & is_responded
...
if performance < MIN_PERFORMANCE:
    io.close()
    return None
if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    io.close()
    return None
```

iii. The agent’s justification came from the methods text it read in step 7: it copied the paper’s behavioral session-selection criteria into its plan. In step 97 it summarized the decision as “Trial filtering: Only trials within recording `obs_intervals` are included” and “Session selection: >65% correct on control non-early-lick trials, ≥50 correct per lick direction.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units['spike_times']` for units whose `classification` is `'good'`, using `BehavioralEvents/go_start_times` to place each trial’s alignment window.

ii.
```python
classifications = units['classification'][:]
good_idx = np.where(classifications == 'good')[0]
...
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
...
spike_times_list = [units['spike_times'][int(idx)] for idx in good_idx]
fr_all = compute_firing_rates_all(spike_times_list, go_cue_times)
```

iii. In step 97 the agent explicitly described “good units only” and “Alignment: Go cue onset from `BehavioralEvents/go_start_times`.” The trajectory also shows it inspecting the `spike_times` field directly.

## 2-b. How is the `neural` data processed?

i. For each good unit and each kept trial, the agent sorts the unit’s spike times, extracts spikes in the window `[go-2.5 s, go+1.5 s)`, subtracts the go-cue time to make them relative to the alignment event, histograms them into 50 ms non-overlapping bins, and divides by bin width to convert counts to firing rates in spikes/s.

ii.
```python
def compute_firing_rates_all(spike_times_list, go_cue_times):
    ...
    for u_i, spike_times in enumerate(spike_times_list):
        if len(spike_times) == 0:
            continue
        st = np.sort(spike_times)
        for t_i in range(n_trials):
            go = go_cue_times[t_i]
            i_lo = np.searchsorted(st, go + T_START, side='left')
            i_hi = np.searchsorted(st, go + T_END, side='left')
            if i_hi > i_lo:
                rel = st[i_lo:i_hi] - go
                counts, _ = np.histogram(rel, bins=BIN_EDGES)
                fr_all[u_i, t_i] = counts / BIN_WIDTH
```

iii. The trajectory shows the agent deciding on “50ms non-overlapping bins -> firing rates (spikes/s)” in both its file header and its final summary in step 97.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered by retaining only units with `classification == 'good'`. If a session has no such units, the whole session is dropped.

ii.
```python
classifications = units['classification'][:]
good_idx = np.where(classifications == 'good')[0]

if len(good_idx) == 0:
    io.close()
    return None
```

iii. In step 38 the agent reasoned that it should “filter units to `good` classification,” and step 97 repeats that this matches the spike-sorting QC classifier approach from the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to go-cue onset. Each trial’s spike times are converted to relative time by subtracting that trial’s `go_cue_time`, and the fixed window `[-2.5, 1.5]` seconds around the go cue is binned.

ii.
```python
T_START = -2.5
T_END = 1.5
...
go = go_cue_times[t_i]
i_lo = np.searchsorted(st, go + T_START, side='left')
i_hi = np.searchsorted(st, go + T_END, side='left')
...
rel = st[i_lo:i_hi] - go
counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. The trajectory and the script header both state that temporal alignment is to go-cue onset from `BehavioralEvents/go_start_times`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data use 50 ms bins, yielding 80 bins from `-2.5 s` to `+1.5 s` relative to the go cue. No additional smoothing or temporal rebinning is applied after that histogramming step.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
```

iii. The agent explicitly computed 80 bins from the instructed window in step 38 and described the result again in step 97.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times` and `BehavioralEvents/go_start_times`. The code finds the last sample-start event before each kept trial’s go cue.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
...
sorted_ss = np.sort(sample_start_times)
tone_onset_rel_go = np.full(n_trials, -1.85)
for t_i in range(n_trials):
    go = go_cue_times[t_i]
    ...
    i_lo = np.searchsorted(sorted_ss, prev_go, side='right')
    i_hi = np.searchsorted(sorted_ss, go, side='right')
    if i_hi > i_lo:
        tone_onset_rel_go[t_i] = sorted_ss[i_hi - 1] - go
```

iii. In the trajectory the agent states that tone time should come from the last `sample_start_times` event before the go cue, while also adding a fallback to the nominal sample-plus-delay duration.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the agent finds the last sample-start timestamp between the previous go cue and the current go cue, subtracts the current go cue to get `tone_onset_rel_go`, and then computes time-from-tone for each neural bin center as `BIN_CENTERS - tone_onset_rel_go`. If it finds no candidate tone onset, it falls back to `-1.85` seconds relative to the go cue.

ii.
```python
tone_onset_rel_go = np.full(n_trials, -1.85)
for t_i in range(n_trials):
    go = go_cue_times[t_i]
    orig_idx = valid_idx[t_i]
    prev_go = go_cue_times_all[orig_idx - 1] if orig_idx > 0 else 0
    i_lo = np.searchsorted(sorted_ss, prev_go, side='right')
    i_hi = np.searchsorted(sorted_ss, go, side='right')
    if i_hi > i_lo:
        tone_onset_rel_go[t_i] = sorted_ss[i_hi - 1] - go
...
time_from_tone = (BIN_CENTERS - tone_onset_rel_go[t_i]).astype(np.float32)
```

iii. The trajectory shows the agent reasoning that the relevant tone is the last tone before the go cue, but also that when none is found it should “fallback: standard timing (sample=0.65s + delay=1.2s = 1.85s before go).”

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same 80 go-cue-centered bin centers used for the neural data. The only difference is that each bin center is shifted by the trial’s tone-to-go offset.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
...
time_from_tone = (BIN_CENTERS - tone_onset_rel_go[t_i]).astype(np.float32)
inp = np.stack([time_from_tone, photostim_on_all[t_i]], axis=0)
```

iii. The agent’s plan in the trajectory was to align everything to go cue onset and then derive time-from-tone on that shared bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from `trials['photostim_onset']`, `trials['photostim_duration']`, `trials['start_time']`, and each trial’s `go_cue_time`.

ii.
```python
photostim_onset_all = trials['photostim_onset'][:]
photostim_duration_all = trials['photostim_duration'][:]
trial_starts_all = trials['start_time'][:]
...
photostim_onset = photostim_onset_all[valid_idx]
photostim_duration = photostim_duration_all[valid_idx]
trial_starts = trial_starts_all[valid_idx]
```

```python
ps_onset_abs = trial_starts[t_i] + float(photostim_onset[t_i])
ps_dur = float(photostim_duration[t_i])
ps_onset_rel = ps_onset_abs - go_cue_times[t_i]
ps_offset_rel = ps_onset_rel + ps_dur
```

iii. In step 38 the agent explicitly notes that `photostim_onset` is relative to trial start and that photostim should be converted into a go-cue-relative binary input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, if `photostim_onset` is not `'N/A'`, the agent converts onset and duration into a go-cue-relative interval and marks each bin center as `1` if it falls inside that interval and `0` otherwise. Trials with `'N/A'` stay all zeros.

ii.
```python
photostim_on_all = np.zeros((n_trials, N_BINS), dtype=np.float32)
for t_i in range(n_trials):
    if photostim_onset[t_i] != 'N/A':
        ps_onset_abs = trial_starts[t_i] + float(photostim_onset[t_i])
        ps_dur = float(photostim_duration[t_i])
        ps_onset_rel = ps_onset_abs - go_cue_times[t_i]
        ps_offset_rel = ps_onset_rel + ps_dur
        photostim_on_all[t_i] = ((BIN_CENTERS >= ps_onset_rel) & (BIN_CENTERS < ps_offset_rel)).astype(np.float32)
```

iii. Step 97 summarizes the decision as “Photostim: Binary time series from trial-level onset/duration, converted to go-cue-relative timing.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim interval is converted to coordinates relative to the same go cue that defines the neural bins, and then evaluated on the same `BIN_CENTERS`.

ii.
```python
ps_onset_abs = trial_starts[t_i] + float(photostim_onset[t_i])
ps_onset_rel = ps_onset_abs - go_cue_times[t_i]
ps_offset_rel = ps_onset_rel + ps_dur
photostim_on_all[t_i] = ((BIN_CENTERS >= ps_onset_rel) & (BIN_CENTERS < ps_offset_rel)).astype(np.float32)
```

iii. The trajectory shows that the agent intentionally put photostim on the go-cue-relative axis so it would share the neural alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read from a direct NWB field. The agent derives it from `trials['outcome']` and `trials['trial_instruction']`.

ii.
```python
outcomes_all = trials['outcome'][:]
instructions_all = trials['trial_instruction'][:]
...
o = outcomes[t_i]
inst = instructions[t_i]
if o == 'ignore':
    choice = 2
elif o == 'hit':
    choice = 0 if inst == 'left' else 1
elif o == 'miss':
    choice = 1 if inst == 'left' else 0
else:
    choice = 2
```

iii. In step 97 the agent explicitly justifies this as “Choice derivation: From outcome + trial_instruction (hit→instructed side, miss→opposite, ignore→no_lick).”

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The agent maps choice to `0=left`, `1=right`, `2=no lick` and repeats the per-trial value across all 80 bins.

ii.
```python
if o == 'ignore':
    choice = 2
elif o == 'hit':
    choice = 0 if inst == 'left' else 1
elif o == 'miss':
    choice = 1 if inst == 'left' else 0
else:
    choice = 2

out = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
], axis=0)
```

iii. The trajectory shows that the agent wanted all outputs in a common `(n_output, n_timepoints)` array, so it broadcast per-trial categorical values across time.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from `trials['outcome']`.

ii.
```python
outcomes_all = trials['outcome'][:]
...
outcomes = outcomes_all[valid_idx]
```

iii. The trajectory inspection in step 25 shows the agent reading the `outcome` trial column directly and using its values as the basis for the outcome output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent maps outcome strings with `{'ignore': 0, 'miss': 1, 'hit': 2}` and repeats the category across all 80 bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
...
out = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_map.get(o, 0), dtype=np.int64),
    ...
], axis=0)
```

iii. The trajectory shows the agent preserving the three outcome classes requested by the decoder task and packing them into the shared output array.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from `trials['early_lick']`.

ii.
```python
early_licks_all = trials['early_lick'][:]
...
early_licks = early_licks_all[valid_idx]
```

iii. In step 25 the agent inspected the `early_lick` column and later described keeping all trial types because early lick is itself a decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The agent converts `early` to `1` and anything else to `0`, then repeats that per-trial value across all 80 bins.

ii.
```python
out = np.stack([
    ...,
    np.full(N_BINS, 1 if early_licks[t_i] == 'early' else 0, dtype=np.int64),
    ...
], axis=0)
```

iii. The trajectory summary in step 97 says “All trial types included (early lick, ignore, miss, hit) since these are decoder outputs,” which explains why the flag is retained rather than used to exclude trials.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`. The code uses `timestamps`, `data[:, 1]` as tongue y, and `data[:, 2]` as DeepLabCut likelihood.

ii.
```python
bts = nwb.acquisition.get('BehavioralTimeSeries', None)
has_tongue = (bts is not None and 'Camera0_side_TongueTracking' in bts.time_series)

if has_tongue:
    tongue_ts = bts.time_series['Camera0_side_TongueTracking']
    tongue_data = tongue_ts.data[:]
    tongue_timestamps = tongue_ts.timestamps[:]
```

```python
def discretize_tongue_y(tongue_timestamps, tongue_data, go_cue_times):
    tongue_y_vals = tongue_data[:, 1]
    tongue_lk_vals = tongue_data[:, 2]
```

iii. In step 38 the agent reasoned explicitly about the tongue-tracking likelihoods, noting that the tongue is visible only in a minority of frames and that likelihood should determine visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent uses a two-pass procedure. First, within each trial window it collects all frames whose likelihood is at least `0.9`; from the concatenated visible-frame y-values it computes session-level 40th and 60th percentiles. Second, for each trial and each 50 ms bin, it averages visible y-values in that bin and assigns class `0/1/2` relative to those percentile thresholds. If a bin has no visible frames, it remains class `3`.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
...
all_visible_y = []
for t_i in range(n_trials):
    ...
    lk = tongue_lk_vals[i_start:i_end]
    vis = lk >= TONGUE_LIKELIHOOD_THRESHOLD
    if np.any(vis):
        all_visible_y.append(tongue_y_vals[i_start:i_end][vis])

if len(all_visible_y) > 0:
    all_vis = np.concatenate(all_visible_y)
    p40 = np.percentile(all_vis, 40)
    p60 = np.percentile(all_vis, 60)
```

```python
ty_binned = np.full(N_BINS, 3, dtype=np.int64)
...
if i_end > i_start:
    lk = tongue_lk_vals[i_start:i_end]
    vis = lk >= TONGUE_LIKELIHOOD_THRESHOLD
    if np.any(vis):
        mean_y = np.mean(tongue_y_vals[i_start:i_end][vis])
        if mean_y < p40:
            ty_binned[b_i] = 0
        elif mean_y <= p60:
            ty_binned[b_i] = 1
        else:
            ty_binned[b_i] = 2
```

iii. The trajectory in step 38 shows the justification: the agent observed that tongue likelihood was high in only about 10% of frames and inferred that a threshold of `0.9` was “probably” appropriate. It also decided percentiles should be computed from visible tongue frames within trial windows.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Category `3` means not visible. Otherwise, the per-bin mean visible y-position is compared to percentiles computed from all visible raw frames across kept trial windows: `< p40 -> 0`, `p40 to p60 -> 1`, `> p60 -> 2`.

ii.
```python
if len(all_visible_y) > 0:
    all_vis = np.concatenate(all_visible_y)
    p40 = np.percentile(all_vis, 40)
    p60 = np.percentile(all_vis, 60)
...
if mean_y < p40:
    ty_binned[b_i] = 0
elif mean_y <= p60:
    ty_binned[b_i] = 1
else:
    ty_binned[b_i] = 2
```

iii. In step 38 the agent states that it is “settling on” a likelihood threshold of `0.9` and percentile cuts at the 40th and 60th percentiles over visible tongue values.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue frames are aligned to the same go-cue-relative window and 50 ms bins as the neural data. For each trial, the code finds video samples between `go + T_START` and `go + T_END`, then bins them using absolute edges `go + BIN_EDGES`.

ii.
```python
for t_i in range(n_trials):
    go = go_cue_times[t_i]
    i_start = np.searchsorted(tongue_timestamps, go + T_START, side='left')
    i_end = np.searchsorted(tongue_timestamps, go + T_END, side='left')
```

```python
abs_edges = go + BIN_EDGES
edge_indices = np.searchsorted(tongue_timestamps, abs_edges, side='left')
for b_i in range(N_BINS):
    i_start = edge_indices[b_i]
    i_end = edge_indices[b_i + 1]
```

iii. The trajectory shows the agent’s overall plan to align all streams to go-cue onset and then compute time-varying inputs and outputs on the same 80-bin grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several missing-data cases by dropping or default-filling. Sessions with mismatched trial and go-cue counts are dropped. Sessions with no `classification == 'good'` units are dropped. Trials without matching `obs_intervals` coverage are removed. If a trial has no detected tone onset in the searched range, the code substitutes a fixed `-1.85 s` tone-to-go offset. If tongue tracking is absent entirely, every bin is set to category `3`; if a bin has no visible tongue frames, it also stays `3`.

ii.
```python
if len(go_cue_times_all) != n_trials_total:
    io.close()
    return None
...
if len(good_idx) == 0:
    io.close()
    return None
...
tone_onset_rel_go = np.full(n_trials, -1.85)
...
if has_tongue:
    ...
else:
    tongue_y_per_trial = [np.full(N_BINS, 3, dtype=np.int64) for _ in range(n_trials)]
```

iii. The trajectory shows the agent treating missing tone detections with a “standard timing” fallback and treating low-likelihood or absent tongue measurements as not visible. It also preferred dropping unusable sessions rather than trying to repair them.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive parts are likely per-session NWB reads, the nested `unit x trial` spike-binning loop in `compute_firing_rates_all`, and the two-pass tongue discretization loop over trials and bins. These are the dominant bulk operations in the script.

ii.
```python
for u_i, spike_times in enumerate(spike_times_list):
    ...
    for t_i in range(n_trials):
        ...
        counts, _ = np.histogram(rel, bins=BIN_EDGES)
        fr_all[u_i, t_i] = counts / BIN_WIDTH
```

```python
all_visible_y = []
for t_i in range(n_trials):
    ...
for t_i in range(n_trials):
    ...
    for b_i in range(N_BINS):
        ...
```

iii. The trajectory does not contain a formal runtime analysis, but it shows the agent focusing on these loops when refining the implementation and then running the full dataset conversion and decoder training successfully.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The neural firing-rate calculation is the clearest target: it loops over both units and trials, instead of vectorizing all trial edges for a unit at once. The tongue code also loops over trials twice and then over bins within each trial. Region extraction over units is also loop-based, though comparatively cheap.

ii.
```python
for u_i, spike_times in enumerate(spike_times_list):
    ...
    for t_i in range(n_trials):
        ...
```

```python
for t_i in range(n_trials):
    ...
for t_i in range(n_trials):
    ...
    for b_i in range(N_BINS):
        ...
```

iii. The trajectory shows the agent trying to make some operations faster with `searchsorted`, but it still left the main spike and tongue calculations in explicit Python loops.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats trial-wise searches through the tongue timestamps in two separate passes, and it repeats per-trial spike window searches for every unit instead of sharing a vectorized edge computation across trials. It also sorts each unit’s spike times inside `compute_firing_rates_all`, even though the underlying NWB spike-time arrays are typically already sorted.

ii.
```python
all_visible_y = []
for t_i in range(n_trials):
    go = go_cue_times[t_i]
    i_start = np.searchsorted(tongue_timestamps, go + T_START, side='left')
    i_end = np.searchsorted(tongue_timestamps, go + T_END, side='left')
    ...
...
for t_i in range(n_trials):
    go = go_cue_times[t_i]
    ty_binned = np.full(N_BINS, 3, dtype=np.int64)
    abs_edges = go + BIN_EDGES
    edge_indices = np.searchsorted(tongue_timestamps, abs_edges, side='left')
```

```python
st = np.sort(spike_times)
for t_i in range(n_trials):
    go = go_cue_times[t_i]
    i_lo = np.searchsorted(st, go + T_START, side='left')
    i_hi = np.searchsorted(st, go + T_END, side='left')
```

iii. The trajectory reflects iterative refinement of these same computations but does not show the agent eliminating the repeated searches or consolidating them into one vectorized pass.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes behavioral session-filtering statistics such as `performance`, `correct_left`, and `correct_right` that are used only to exclude sessions and are not stored in the final saved dataset. It also carries `performance` in each per-session result only for logging. More broadly, the extra behavioral session-selection step itself is not needed for the target output format.

ii.
```python
n_correct_control = np.sum((outcomes == 'hit') & is_control & is_no_early)
performance = n_correct_control / np.sum(control_responded)

correct_left = np.sum((outcomes == 'hit') & (instructions == 'left') & is_control & is_no_early)
correct_right = np.sum((outcomes == 'hit') & (instructions == 'right') & is_control & is_no_early)
```

```python
return {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'subject_id': subject_id,
    'regions': regions,
    'n_trials': n_trials,
    'n_neurons': len(good_idx),
    'performance': performance,
}
```

iii. The trajectory makes clear that the agent added paper-style session selection on purpose. Its final summary in step 97 highlights these performance thresholds even though they are not part of the saved target structure.
