# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `/app/data/sub-*/`. All sessions are found with a glob pattern and each file is opened with `pynwb.NWBHDF5IO`. Subjects, trials, units, and behavioral events are read from within each NWB file.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*.nwb'))
...
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path)
```

Loading one session:
```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
subject_id = str(nwb.subject.subject_id) if nwb.subject else 'unknown'
trials = nwb.trials
units = nwb.units
be = nwb.acquisition['BehavioralEvents']
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
```

iii. The agent recognized the data was in NWB format (rather than the .mat files used in the reference code) and chose `pynwb` as the standard NWB reader. It explored a single NWB file to understand the structure before iterating over all 174 files.

## 1-b. How are the data split into subjects?

i. Each NWB file provides `nwb.subject.subject_id`, a numeric string. The unique subject IDs are collected across all sessions, sorted, and indexed.

ii.
```python
subject_id = str(nwb.subject.subject_id) if nwb.subject else 'unknown'
...
subjects = sorted(list(all_subjects))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_to_idx[s['subject_id']] for s in all_sessions], dtype=np.int64),
```

iii. The agent used the numeric `subject_id` from the NWB file, which is the canonical identifier available per file.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. No grouping or splitting is needed. Sessions are processed in sorted file order.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*.nwb'))
```

iii. The agent recognized that the dandiset stores one session per file, making the file boundary the session boundary.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`), one row per behavioral trial. The go cue count is checked against the trial count.

ii.
```python
trials = nwb.trials
n_trials_total = len(trials)
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
if len(go_cue_times_all) != n_trials_total:
    io.close()
    return None
```

iii. The agent verified that go_cue events match the trials table rows, dropping any session where they disagree (though none were found to disagree).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on observation intervals (`obs_intervals`): a trial is included if its go cue time falls within any observation interval of the first good unit. Additionally, sessions are filtered by behavioral performance criteria: >65% correct on control (non-photostim, non-early-lick) trials, and at least 50 correct lick-left and 50 correct lick-right control trials. Free water trials are NOT filtered.

ii.
```python
obs_intervals = units['obs_intervals'][int(good_idx[0])]
valid_trial_mask = get_valid_trial_mask(obs_intervals, go_cue_times_all)
...
# Session selection criteria
is_control = np.array([po == 'N/A' for po in photostim_onset])
is_no_early = early_licks == 'no early'
is_responded = outcomes != 'ignore'
control_responded = is_control & is_no_early & is_responded
n_correct_control = np.sum((outcomes == 'hit') & is_control & is_no_early)
performance = n_correct_control / np.sum(control_responded)
correct_left = np.sum((outcomes == 'hit') & (instructions == 'left') & is_control & is_no_early)
correct_right = np.sum((outcomes == 'hit') & (instructions == 'right') & is_control & is_no_early)
if performance < MIN_PERFORMANCE:
    return None
if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    return None
```

The `get_valid_trial_mask` function:
```python
def get_valid_trial_mask(obs_intervals, go_cue_times):
    n_trials = len(go_cue_times)
    valid = np.zeros(n_trials, dtype=bool)
    for t_i in range(n_trials):
        go = go_cue_times[t_i]
        for obs_start, obs_end in obs_intervals:
            if go >= obs_start and go <= obs_end:
                valid[t_i] = True
                break
    return valid
```

iii. The agent applied session-level performance criteria (>65% correct, 50+ correct per side) based on the reference code's session selection logic. For trial-level filtering, it checked whether the go cue fell within observation intervals. Free water filtering was not implemented, though the agent observed that `free_water` values were all 0 in the files it inspected.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']` for each good unit, binned relative to go cue times from `BehavioralEvents/go_start_times`.

ii.
```python
spike_times_list = [units['spike_times'][int(idx)] for idx in good_idx]
fr_all = compute_firing_rates_all(spike_times_list, go_cue_times)
```

iii. The agent identified `spike_times` as the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50ms bins spanning -2.5s to +1.5s relative to the go cue. Spike counts per bin are divided by the bin width (0.05s) to get firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
...
def compute_firing_rates_all(spike_times_list, go_cue_times):
    n_units = len(spike_times_list)
    n_trials = len(go_cue_times)
    fr_all = np.zeros((n_units, n_trials, N_BINS), dtype=np.float32)
    for u_i, spike_times in enumerate(spike_times_list):
        st = np.sort(spike_times)
        for t_i in range(n_trials):
            go = go_cue_times[t_i]
            i_lo = np.searchsorted(st, go + T_START, side='left')
            i_hi = np.searchsorted(st, go + T_END, side='left')
            if i_hi > i_lo:
                rel = st[i_lo:i_hi] - go
                counts, _ = np.histogram(rel, bins=BIN_EDGES)
                fr_all[u_i, t_i] = counts / BIN_WIDTH
    return fr_all
```

iii. The agent used `np.histogram` for spike counting within each trial window, consistent with the 50ms bin width specified in the instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. No additional quality metric thresholds are applied. Sessions with no good units are dropped.

ii.
```python
classifications = units['classification'][:]
good_idx = np.where(classifications == 'good')[0]
if len(good_idx) == 0:
    io.close()
    return None
```

iii. The agent identified `classification` as the QC classifier verdict from the spike sorting quality control paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and go cue times are on the same session-absolute clock. Each trial's bin edges are computed as `go_cue + BIN_EDGES`, and spikes are histogrammed into those bins.

ii.
```python
go = go_cue_times[t_i]
i_lo = np.searchsorted(st, go + T_START, side='left')
i_hi = np.searchsorted(st, go + T_END, side='left')
rel = st[i_lo:i_hi] - go
counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. The agent recognized that all NWB times share a global clock, so alignment to go cue only requires looking up each trial's go cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50ms. Spike times are binned into 80 non-overlapping 50ms bins spanning -2.5s to +1.5s relative to the go cue. The bin grid is defined once and reused for every trial and session.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
```

iii. The window and 50ms bin width are set by the instructions. No rebinning is applied beyond the initial binning from spike times.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onsets) in BehavioralEvents, together with the go cue of each trial. The tone taken for a trial is the last `sample_start_time` before its go cue.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
sorted_ss = np.sort(sample_start_times)
tone_onset_rel_go = np.full(n_trials, -1.85)
for t_i in range(n_trials):
    go = go_cue_times[t_i]
    orig_idx = valid_idx[t_i]
    prev_go = go_cue_times_all[orig_idx - 1] if orig_idx > 0 else 0
    i_lo = np.searchsorted(sorted_ss, prev_go, side='right')
    i_hi = np.searchsorted(sorted_ss, go, side='right')
    if i_hi > i_lo:
        tone_onset_rel_go[t_i] = sorted_ss[i_hi - 1] - go
```

iii. The agent recognized that early licks can replay the sample epoch, causing multiple tone onsets per trial, and chose the last sample start before the go cue as the relevant one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset is expressed relative to the go cue, then time from tone at each bin is computed as `BIN_CENTERS - tone_onset_rel_go`. A default of -1.85s is used if no sample start is found.

ii.
```python
tone_onset_rel_go = np.full(n_trials, -1.85)
...
tone_onset_rel_go[t_i] = sorted_ss[i_hi - 1] - go
...
time_from_tone = (BIN_CENTERS - tone_onset_rel_go[t_i]).astype(np.float32)
```

iii. The agent noted that the typical sample-to-go gap is ~1.85s (sample 0.65s + delay 1.2s) and used this as a default fallback.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers defined relative to the go cue, so alignment is automatic.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
time_from_tone = (BIN_CENTERS - tone_onset_rel_go[t_i]).astype(np.float32)
```

iii. The shared bin grid ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` and the go cue used to compute absolute onset/offset times.

ii.
```python
photostim_onset = photostim_onset_all[valid_idx]
photostim_duration = photostim_duration_all[valid_idx]
trial_starts = trial_starts_all[valid_idx]
...
ps_onset_abs = trial_starts[t_i] + float(photostim_onset[t_i])
ps_dur = float(photostim_duration[t_i])
ps_onset_rel = ps_onset_abs - go_cue_times[t_i]
ps_offset_rel = ps_onset_rel + ps_dur
```

iii. The agent recognized that `photostim_onset` is stored as a string relative to trial start, with 'N/A' for non-stimulated trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: each bin is 1 if its center falls between onset and offset, 0 otherwise. Non-stimulated trials ('N/A') remain all zeros.

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

iii. The agent correctly identified photostim onset is relative to trial start and converted it to go-cue-relative coordinates.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are expressed relative to the go cue, which is the same alignment event used for neural bins, so comparison with bin centers provides alignment.

ii.
```python
ps_onset_rel = ps_onset_abs - go_cue_times[t_i]
photostim_on_all[t_i] = ((BIN_CENTERS >= ps_onset_rel) & (BIN_CENTERS < ps_offset_rel)).astype(np.float32)
```

iii. Same go-cue-relative coordinate system as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore'). A hit means the animal licked the instructed side, a miss means the opposite side, and ignore means no lick.

ii.
```python
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

iii. The agent recognized that the animal's actual lick direction is not stored directly but can be derived from the instructed side and outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=left, 1=right, 2=no_lick, and repeated across all 80 time bins.

ii.
```python
out = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
], axis=0)
```

iii. Choice is a per-trial value, replicated across time bins for the output format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, which holds 'ignore', 'miss', and 'hit'.

ii.
```python
outcomes = outcomes_all[valid_idx]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
```

iii. The trials table stores outcome explicitly with exactly the three categories needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to 0=ignore, 1=miss, 2=hit and repeated across all 80 time bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
np.full(N_BINS, outcome_map.get(o, 0), dtype=np.int64),
```

iii. Direct mapping from string to integer code.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'no early' and 'early'.

ii.
```python
early_licks = early_licks_all[valid_idx]
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes and repeated across all 80 time bins.

ii.
```python
np.full(N_BINS, 1 if early_licks[t_i] == 'early' else 0, dtype=np.int64),
```

iii. Direct binary mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose data is (n_frames, 3) = tongue_x, tongue_y, tongue_likelihood, with timestamps.

ii.
```python
tongue_ts = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts.data[:]
tongue_timestamps = tongue_ts.timestamps[:]
tongue_y_vals = tongue_data[:, 1]
tongue_lk_vals = tongue_data[:, 2]
```

iii. The agent identified Camera0_side_TongueTracking as the tongue measurement source.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps: (1) Frames with likelihood < 0.9 are excluded. (2) Visible y-values from within trial windows are pooled to compute the 40th and 60th percentiles. (3) Per bin per trial, the mean of visible y-values is discretized: < p40 -> 0, <= p60 -> 1, > p60 -> 2, not visible -> 3.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
...
# First pass: collect visible y values for percentiles
for t_i in range(n_trials):
    go = go_cue_times[t_i]
    i_start = np.searchsorted(tongue_timestamps, go + T_START, side='left')
    i_end = np.searchsorted(tongue_timestamps, go + T_END, side='left')
    if i_end > i_start:
        lk = tongue_lk_vals[i_start:i_end]
        vis = lk >= TONGUE_LIKELIHOOD_THRESHOLD
        if np.any(vis):
            all_visible_y.append(tongue_y_vals[i_start:i_end][vis])

all_vis = np.concatenate(all_visible_y)
p40 = np.percentile(all_vis, 40)
p60 = np.percentile(all_vis, 60)

# Second pass: discretize per bin
for b_i in range(N_BINS):
    ...
    if np.any(vis):
        mean_y = np.mean(tongue_y_vals[i_start:i_end][vis])
        if mean_y < p40:
            ty_binned[b_i] = 0
        elif mean_y <= p60:
            ty_binned[b_i] = 1
        else:
            ty_binned[b_i] = 2
```

iii. The agent used a likelihood threshold of 0.9 (a common DLC convention) and computed percentiles over raw visible frames within trial windows rather than over session-wide bin means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The boundary check uses strict less-than for p40 and less-than-or-equal for p60: < p40 -> 0, <= p60 -> 1, > p60 -> 2, not visible -> 3.

ii.
```python
if mean_y < p40:
    ty_binned[b_i] = 0
elif mean_y <= p60:
    ty_binned[b_i] = 1
else:
    ty_binned[b_i] = 2
```

iii. This follows the instructions: 0 = below 40th percentile, 1 = 40th to 60th, 2 = above 60th, 3 = not visible.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps share the same session-absolute clock. For each trial, frames in the [-2.5, 1.5] window around the go cue are found via `searchsorted`, then assigned to 50ms bins using the same bin edges as the neural data.

ii.
```python
abs_edges = go + BIN_EDGES
edge_indices = np.searchsorted(tongue_timestamps, abs_edges, side='left')
for b_i in range(N_BINS):
    i_start = edge_indices[b_i]
    i_end = edge_indices[b_i + 1]
```

iii. The same go-cue-relative bin grid ensures alignment with the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with no good units are dropped. (2) Trials outside observation intervals are excluded. (3) Sessions without tongue tracking get all-not-visible (class 3) tongue data. (4) If no sample_start is found for a trial, a default tone onset of -1.85s relative to go is used.

ii.
```python
if len(good_idx) == 0:
    io.close()
    return None
...
has_tongue = (bts is not None and 'Camera0_side_TongueTracking' in bts.time_series)
if not has_tongue:
    tongue_y_per_trial = [np.full(N_BINS, 3, dtype=np.int64) for _ in range(n_trials)]
...
tone_onset_rel_go = np.full(n_trials, -1.85)
```

iii. The agent handled edge cases by dropping sessions/trials where neural data is absent, using fallback values for tongue tracking and tone onset when data is missing.

## 10-a. What are the most time-consuming steps of the code?

i. The firing rate computation (double loop over units and trials with `np.searchsorted` and `np.histogram`) and the tongue tracking discretization (triple loop over trials and bins) are the most time-consuming. The agent encountered performance issues during initial runs and had to optimize the code.

ii.
```python
for u_i, spike_times in enumerate(spike_times_list):
    st = np.sort(spike_times)
    for t_i in range(n_trials):
        ...
        counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. The agent noted that processing was slow on initial runs (stuck at session 49/174) and rewrote the code with `np.searchsorted` optimizations.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The firing rate computation uses a double loop (per unit, per trial) where the per-trial dimension could be vectorized by flattening all trial edges into one array (as the reference code does). The tongue discretization uses a triple loop (per trial, per bin) that could use vectorized bin assignment.

ii.
```python
# Could be vectorized over trials:
for u_i, spike_times in enumerate(spike_times_list):
    for t_i in range(n_trials):
        ...
        counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. The reference code vectorizes the trial dimension by computing `edges = (go[:, None] + REL_EDGES[None, :]).ravel()` and calling `np.searchsorted` once per unit over all trials.

## 10-c. What processing does the code repeat multiple times?

i. The tongue discretization performs two full passes over the trial data: once to collect visible y-values for percentile computation, and once to discretize per bin. The reference code computes percentiles over the whole session in one pass.

ii.
```python
# First pass: collect visible y values for percentiles
for t_i in range(n_trials):
    ...
# Second pass: discretize per bin
for t_i in range(n_trials):
    ...
```

iii. The two-pass approach is logically necessary given that percentiles must be known before discretization, but the first pass could pool from the whole session rather than just trial windows.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores per-session performance metrics (`performance`) that are used only for session filtering and not included in the final output dictionary (beyond being used as a filter). The `np.sort(spike_times)` call is also performed per unit even though spike times in NWB files are already sorted.

ii.
```python
result = {
    ...
    'performance': performance,  # used only for filtering, not in final output
}
...
st = np.sort(spike_times)  # likely unnecessary as NWB spike times are already sorted
```

iii. The performance metric is a minor overhead used only for session selection.
