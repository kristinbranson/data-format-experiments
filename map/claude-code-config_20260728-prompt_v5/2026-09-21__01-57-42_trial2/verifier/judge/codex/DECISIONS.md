# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script finds all NWB files with a sorted glob over `/app/data/sub-*/*.nwb`, treats each file as one session, and opens each session directly with `h5py.File`. Within each file it reads HDF5 groups for subject metadata, the trials table, behavioral event timestamps, behavioral time series, and unit data.

ii.
```python
# Find all NWB files
nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
print(f"Found {len(nwb_files)} NWB files")
```

```python
f = h5py.File(nwb_path, 'r')

subject_id = f['general']['subject']['subject_id'][()].decode()
n_trials_total = f['intervals']['trials']['id'].shape[0]
go_start_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
classification = f['units']['classification'][:]
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI justified this by noting that the dataset is organized as one NWB file per session under subject folders. The trajectory shows it deliberately used raw HDF5 access with `h5py` after exploring the NWB layout.

## 1-b. How are the data split into subjects (mice)?

i. The AI uses `general/subject/subject_id` from each NWB file as the subject identifier, collects the unique IDs across sessions, sorts them, and stores per-session indices into that subject list.

ii.
```python
subject_id = f['general']['subject']['subject_id'][()].decode()
```

```python
all_subject_ids = sorted(set(s['subject_id'] for s in all_sessions))
subject_to_idx = {sid: i for i, sid in enumerate(all_subject_ids)}

'subjects': all_subject_ids,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. `CONVERSION_NOTES.md` Step 5 maps `general/subject/subject_id` directly to `subjects` and `subject_idx`, treating that NWB field as the canonical mouse identifier.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order is the sorted file order from the glob, and the script stores the session identifier as the filename basename rather than `nwb.identifier`.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
```

```python
basename = os.path.basename(nwb_path)
...
return {
    'subject_id': subject_id,
    'session_name': basename,
    ...
}
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI documented the dataset as “28 subjects, 174 sessions” with one NWB per session, so it used the file boundary as the session boundary.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table row count (`intervals/trials/id`) as the master trial count, checks that `go_start_times` has the same length, and then applies a boolean trial mask to get the kept trial indices. All trial-level arrays are indexed with those kept trial indices.

ii.
```python
n_trials_total = f['intervals']['trials']['id'].shape[0]
...
go_start_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
assert len(go_start_times) == n_trials_total, f"Go cue count mismatch: {len(go_start_times)} vs {n_trials_total}"
```

```python
trial_mask = ~is_auto_or_free
trial_indices = np.where(trial_mask)[0]
n_trials = len(trial_indices)

go_times = go_start_times[trial_indices]
trial_outcomes = outcome[trial_indices]
trial_early_lick = early_lick[trial_indices]
trial_instructions = trial_instruction[trial_indices]
```

iii. The trajectory shows the AI concluded that `go_start_times` had one entry per trial and used that to validate the trial table, then treated the table rows as the trial definition.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out trials marked `auto_water` or `free_water`, keeps early-lick, ignore, and photostimulation trials, and drops a session if fewer than 2 trials survive. It does not use `units/obs_intervals` to remove behavioral trials without spike data.

ii.
```python
auto_water = f['intervals']['trials']['auto_water'][:]
free_water = f['intervals']['trials']['free_water'][:]
...
is_auto_or_free = (auto_water == 1) | (free_water == 1)
...
trial_mask = ~is_auto_or_free
trial_indices = np.where(trial_mask)[0]
...
if n_trials < 2:
    print(f"  SKIP {basename}: only {n_trials} valid trials")
    f.close()
    return None
```

iii. In `CONVERSION_NOTES.md` Step 5 and the README “Key Decisions”, the AI justified this by saying early lick, outcome/ignore, and photostim must be kept because they are decoder outputs or inputs, while auto/free water are not genuine behavioral trials. The trajectory also explicitly states that choice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from `units/spike_times` and `units/spike_times_index`, restricted to units whose `classification` equals `b'good'`. Go-cue timestamps from `BehavioralEvents/go_start_times` define the per-trial alignment windows.

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
good_indices = np.where(good_mask)[0]
```

```python
spike_times_data = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
...
go_start_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
go_times = go_start_times[trial_indices]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `units/spike_times` to neural activity and Step 3 says classifier-based QC is already encoded in `units/classification`.

## 2-b. How is the `neural` data processed?

i. The AI reconstructs each good unit’s spike train from the ragged spike buffer, bins spikes into 50 ms windows around each trial’s go cue with `np.histogram`, and divides by bin width to convert counts to firing rates in Hz. It returns one `(n_units, 80)` array per trial.

ii.
```python
def extract_unit_spike_times(spike_times_data, spike_times_index, unit_indices):
    ...
    return [spike_times_data[starts[i]:ends[i]] for i in range(len(unit_indices))]
```

```python
def compute_firing_rates_fast(unit_spike_times_list, go_times, bin_edges):
    ...
    all_rates = np.zeros((n_units, n_trials, n_bins), dtype=np.float32)

    for ni in range(n_units):
        st = unit_spike_times_list[ni]
        ...
        for ti in range(n_trials):
            go_t = go_times[ti]
            ...
            counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
            all_rates[ni, ti] = counts

    all_rates /= bin_width
    return [all_rates[:, ti, :] for ti in range(n_trials)]
```

iii. `CONVERSION_NOTES.md` Step 6 says the AI intentionally used 50 ms non-overlapping bins over `[-2.5, 1.5]`, matching the decoder task rather than the paper’s sliding-window preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only unit-level QC is `classification == b'good'`. If a session has zero good units, the whole session is skipped.

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)

if n_good == 0:
    print(f"  SKIP {basename}: no good units")
    f.close()
    return None
```

iii. `CONVERSION_NOTES.md` Steps 1 and 3 say the reference pipeline uses classifier-based spike QC and that NWB already stores this verdict in `units/classification`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script aligns neural data to go-cue onset. For each trial it takes absolute `go_start_times`, adds the relative bin edges `[-2.5, 1.5]`, and bins spikes in those windows.

ii.
```python
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
...
go_times = go_start_times[trial_indices]
...
counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
```

iii. `CONVERSION_NOTES.md` Step 6 lists “50ms non-overlapping bins, -2.5s to +1.5s around go cue” as a core design choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins and 80 total time bins per trial, spanning `-2.5 s` to `+1.5 s` around go cue. The raw spike times are directly histogrammed into that grid; there is no additional rebinning or smoothing.

ii.
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. The AI cites the decoder specification in `CONVERSION_NOTES.md` Steps 3 and 6 as the reason for using fixed 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. In the final code, this input is not derived from a raw event variable at all. It is derived from the fixed constant `TONE_ONSET_REL_GO = -1.85` and the shared bin centers.

ii.
```python
TONE_ONSET_REL_GO = -1.85  # seconds before go cue
```

```python
# Input 0: Time from tone onset (fixed task protocol timing)
time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO  # same for all trials
```

iii. `CONVERSION_NOTES.md` Step 6 and the trajectory state that the AI initially tried using `sample_start_times`, but replaced that with a hard-coded `-1.85 s` offset because replayed sample epochs after early licks made event-based lookup “incorrect” for about 12.5% of trials.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The script computes a single 80-element vector equal to `BIN_CENTERS - (-1.85)`, so the value at each bin is treated as seconds since a protocol-defined tone onset. The same vector is used for every kept trial.

ii.
```python
time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO  # same for all trials
...
trial_input = np.stack([time_from_tone.astype(np.float32), photostim_ts], axis=0)
```

iii. The AI’s justification in `CONVERSION_NOTES.md` and the trajectory is that the task protocol defines 0.65 s of sample plus 1.2 s of delay, so tone onset should always be 1.85 s before go cue.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is placed on the same fixed 80-bin go-cue-aligned grid as the neural data. Because the vector is constant across trials, alignment is achieved only through shared bin centers, not through trial-specific tone timestamps.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
...
time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO
...
trial_input = np.stack([time_from_tone.astype(np.float32), photostim_ts], axis=0)
```

iii. The AI argues in `CONVERSION_NOTES.md` Step 6 and the README that go-cue alignment is fixed and that a fixed tone offset avoids replay-related event mismatches.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `intervals/trials/photostim_onset`, `intervals/trials/photostim_duration`, `intervals/trials/start_time`, and the per-trial go-cue times.

ii.
```python
photostim_onset = f['intervals']['trials']['photostim_onset'][:]
photostim_dur = f['intervals']['trials']['photostim_duration'][:]
...
onset_abs = onset_rel_trial + trial_start_times[i]
onset_rel_go = onset_abs - go_times[i]
offset_rel_go = onset_rel_go + dur
```

iii. `CONVERSION_NOTES.md` Step 5 maps photostim from the trial table and notes that onsets are stored relative to trial start, so they must be converted into the go-cue frame used by the decoder.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts trial-relative photostim onset to go-cue-relative onset and offset, then marks each 50 ms bin center as 1 when it falls inside that interval and 0 otherwise.

ii.
```python
def get_photostim_intervals(f, n_trials, go_times, trial_start_times):
    ...
    if photostim_onset[i] == b'N/A':
        intervals.append(None)
    else:
        onset_rel_trial = float(photostim_onset[i])
        dur = float(photostim_dur[i])
        onset_abs = onset_rel_trial + trial_start_times[i]
        onset_rel_go = onset_abs - go_times[i]
        offset_rel_go = onset_rel_go + dur
        intervals.append((onset_rel_go, offset_rel_go))
```

```python
photostim_ts = np.zeros(N_BINS, dtype=np.float32)
ps_interval = photostim_intervals[trial_idx]
if ps_interval is not None:
    onset_rel, offset_rel = ps_interval
    photostim_ts[(BIN_CENTERS >= onset_rel) & (BIN_CENTERS < offset_rel)] = 1.0
```

iii. `CONVERSION_NOTES.md` Step 5 says the decoder should receive photostim as a binary time-varying input rather than a per-trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is expressed relative to the same go-cue-centered bin centers used for neural activity, so the binary photostim time series and the neural bins share the same time axis.

ii.
```python
onset_rel_go = onset_abs - go_times[i]
offset_rel_go = onset_rel_go + dur
```

```python
photostim_ts[(BIN_CENTERS >= onset_rel) & (BIN_CENTERS < offset_rel)] = 1.0
```

iii. The trajectory shows the AI explicitly reasoned that photostim must be converted from trial-relative timing into go-cue-relative timing before comparison to the bin centers.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `intervals/trials/outcome` and `intervals/trials/trial_instruction`.

ii.
```python
trial_outcomes = outcome[trial_indices]
trial_instructions = trial_instruction[trial_indices]
```

```python
if out == 'hit':
    choice = 0 if inst == 'left' else 1
elif out == 'miss':
    choice = 1 if inst == 'left' else 0
else:  # ignore
    choice = 2
```

iii. In `CONVERSION_NOTES.md` Step 5 and the trajectory, the AI justifies this as: hit means the mouse licked the instructed side, miss means the opposite side, and ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0=left`, `1=right`, `2=no lick`, derives it from outcome plus instruction, and broadcasts that single trial value across all 80 bins.

ii.
```python
if out == 'hit':
    choice = 0 if inst == 'left' else 1
elif out == 'miss':
    choice = 1 if inst == 'left' else 0  # opposite of instruction
else:  # ignore
    choice = 2
```

```python
trial_output = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_val, dtype=np.int64),
    tongue_y_disc,
], axis=0)
```

iii. `CONVERSION_NOTES.md` Step 5 lists this coding explicitly and says choice is a per-trial output.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from `intervals/trials/outcome`.

ii.
```python
outcome = np.array([x.decode() for x in f['intervals']['trials']['outcome'][:]])
...
trial_outcomes = outcome[trial_indices]
```

iii. `CONVERSION_NOTES.md` Step 2 identifies `outcome` as already having the needed `hit`, `miss`, and `ignore` labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, `hit -> 2` and repeats the per-trial label across all bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[out]
```

```python
trial_output = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_val, dtype=np.int64),
    tongue_y_disc,
], axis=0)
```

iii. `CONVERSION_NOTES.md` Step 5 lists this mapping exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from `intervals/trials/early_lick`.

ii.
```python
early_lick = np.array([x.decode() for x in f['intervals']['trials']['early_lick'][:]])
...
trial_early_lick = early_lick[trial_indices]
```

iii. `CONVERSION_NOTES.md` Step 2 identifies `early_lick` as an explicit trials-table field with values `early` and `no early`.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI encodes `no early` as `0`, `early` as `1`, and broadcasts the per-trial label across all 80 bins.

ii.
```python
early_val = 0 if trial_early_lick[ti] == 'no early' else 1
```

```python
trial_output = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_val, dtype=np.int64),
    tongue_y_disc,
], axis=0)
```

iii. `CONVERSION_NOTES.md` Step 5 lists `0=no, 1=yes` for the early-lick output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. The script uses the tracking timestamps plus column 1 of `data` as tongue y and column 2 as tracking likelihood.

ii.
```python
tongue_ts = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
tongue_data_raw = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
```

```python
tongue_y = tongue_data[:, 1]
tongue_lh = tongue_data[:, 2]
```

iii. `CONVERSION_NOTES.md` Step 2 describes this NWB time series as tongue/jaw/nose tracking at about 300 Hz with `(x, y, likelihood)` columns.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI does a two-pass computation. First it gathers visible tongue y samples from each trial window, where visibility means `likelihood >= 0.9`, concatenates them, and computes the session’s 40th and 60th percentiles from those raw visible y values. Second, for each trial it bins tongue frames into the shared 50 ms grid, requires each bin’s mean likelihood to be at least `0.9`, computes the bin’s mean y, and discretizes that mean using the session percentiles.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
```

```python
visible_y_all = []
trial_tongue_ranges = []
for ti in range(n_trials):
    ...
    if i_lo < i_hi:
        vis_mask = tongue_lh[i_lo:i_hi] >= likelihood_thresh
        if vis_mask.any():
            visible_y_all.append(tongue_y[i_lo:i_hi][vis_mask])
...
all_visible = np.concatenate(visible_y_all)
p40 = np.percentile(all_visible, 40)
p60 = np.percentile(all_visible, 60)
```

```python
for b in range(n_bins):
    b_mask = bin_idx_v == b
    if not b_mask.any():
        continue
    avg_lh = trial_lh_v[b_mask].mean()
    if avg_lh >= likelihood_thresh:
        avg_y = trial_y_v[b_mask].mean()
        if avg_y < p40:
            trial_bins[b] = 0
        elif avg_y <= p60:
            trial_bins[b] = 1
        else:
            trial_bins[b] = 2
```

iii. `CONVERSION_NOTES.md` Step 5 says the AI wanted session-level 40th/60th percentile discretization on visible tongue data with a separate “not visible” class; Step 6 says it used a 0.9 likelihood threshold.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The script thresholds tongue bins into four categories: `0` if the bin mean y is below session `p40`, `1` if it is between `p40` and `p60`, `2` if it is above `p60`, and `3` if the bin has no valid samples or fails the visibility criterion.

ii.
```python
trial_bins = np.full(n_bins, 3, dtype=np.int64)
...
if avg_lh >= likelihood_thresh:
    avg_y = trial_y_v[b_mask].mean()
    if avg_y < p40:
        trial_bins[b] = 0
    elif avg_y <= p60:
        trial_bins[b] = 1
    else:
        trial_bins[b] = 2
```

iii. `CONVERSION_NOTES.md` Step 5 states the intended categories as `<40th`, `40th-60th`, `>60th`, and `not visible`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue data is aligned on the same go-cue-centered `[-2.5, 1.5]` window as the neural data. For each trial the code finds the tongue frames between `go + T_START` and `go + T_END`, converts their timestamps to trial-relative time by subtracting `go_t`, and assigns them to the same 50 ms bins.

ii.
```python
for ti in range(n_trials):
    go_t = go_times[ti]
    t_lo = go_t + bin_edges[0]
    t_hi = go_t + bin_edges[-1]
    i_lo = np.searchsorted(tongue_ts, t_lo)
    i_hi = np.searchsorted(tongue_ts, t_hi)
```

```python
trial_ts = tongue_ts[i_lo:i_hi]
rel_ts = trial_ts - go_t
bin_idx = np.searchsorted(bin_edges, rel_ts, side='right') - 1
valid = (bin_idx >= 0) & (bin_idx < n_bins)
```

iii. `CONVERSION_NOTES.md` Step 6 lists tongue y as a time-varying output at the same 50 ms resolution as neural activity.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles some missing-data cases by skipping sessions with no good units, skipping sessions with fewer than 2 non-auto/free trials, and assigning tongue bins to class `3` when no visible tongue is available. It also wraps per-session processing in a broad `try/except` and skips sessions that raise exceptions. It does not explicitly remove behavior trials that lack spike observations.

ii.
```python
if n_trials < 2:
    print(f"  SKIP {basename}: only {n_trials} valid trials")
    f.close()
    return None
...
if n_good == 0:
    print(f"  SKIP {basename}: no good units")
    f.close()
    return None
```

```python
trial_bins = np.full(n_bins, 3, dtype=np.int64)
...
if len(visible_y_all) == 0:
    p40, p60 = None, None
    return [np.full(n_bins, 3, dtype=np.int64) for _ in range(n_trials)], p40, p60
```

```python
for i, nwb_file in enumerate(nwb_files):
    ...
    try:
        result = process_session(nwb_file, show_processing=args.show_processing)
        if result is not None:
            all_sessions.append(result)
    except Exception as e:
        print(f"  ERROR processing {nwb_file}: {e}")
        import traceback
        traceback.print_exc()
```

iii. `CONVERSION_NOTES.md` Step 6 says one session with no good units is skipped. The notes and trajectory also justify the tongue “not visible” category and describe the hard-coded tone timing as a fix for event irregularities.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming parts are reading the large spike and tongue arrays from each NWB file, the nested unit-by-trial spike histogram loop in `compute_firing_rates_fast`, and the two-pass tongue processing over all trials. Optional plotting in `--show-processing` mode adds extra overhead.

ii.
```python
spike_times_data = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
...
tongue_ts = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
tongue_data_raw = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
```

```python
for ni in range(n_units):
    ...
    for ti in range(n_trials):
        ...
        counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
```

```python
for ti in range(n_trials):
    ...
for ti in range(n_trials):
    ...
    for b in range(n_bins):
```

iii. `CONVERSION_NOTES.md` Step 9 reports a full conversion time of about 500 s and the function docstrings/commentary repeatedly describe these parts as the optimized heavy work.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain unvectorized: the inner per-trial loop inside spike binning, the per-bin loop inside tongue discretization, and the per-trial loops that build `input_data` and `output_data`.

ii.
```python
for ni in range(n_units):
    ...
    for ti in range(n_trials):
        ...
        counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
```

```python
for ti in range(n_trials):
    ...
    for b in range(n_bins):
        b_mask = bin_idx_v == b
```

```python
input_data = []
for ti, trial_idx in enumerate(trial_indices):
    ...
    input_data.append(trial_input)
...
output_data = []
for ti in range(n_trials):
    ...
    output_data.append(trial_output)
```

iii. The `compute_firing_rates_fast` docstring says the AI was trying to optimize the spike loop, but the final code still uses multiple nested Python loops.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations: it reuses the same `time_from_tone` vector but casts and stacks it inside every trial loop, computes photostim intervals for all original trials and then iterates again only over kept trials, and constructs per-trial output arrays one trial at a time even for variables that are constant within a trial.

ii.
```python
time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO  # same for all trials
...
for ti, trial_idx in enumerate(trial_indices):
    ...
    trial_input = np.stack([time_from_tone.astype(np.float32), photostim_ts], axis=0)
```

```python
photostim_intervals = get_photostim_intervals(f, n_trials_total, go_start_times, trial_start_times)
...
for ti, trial_idx in enumerate(trial_indices):
    ps_interval = photostim_intervals[trial_idx]
```

```python
for ti in range(n_trials):
    ...
    trial_output = np.stack([
        np.full(N_BINS, choice, dtype=np.int64),
        np.full(N_BINS, outcome_val, dtype=np.int64),
        np.full(N_BINS, early_val, dtype=np.int64),
        tongue_y_disc,
    ], axis=0)
```

iii. These are not explicitly justified in the notes; they are visible in the code structure itself. The notes instead emphasize correctness and decoder performance over minimizing repeated assembly work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some work that is not needed for downstream decoding: it computes `performance` only for metadata/logging, returns `p40` and `p60` from tongue processing but never stores them, assigns `trial_starts` without using it later, and includes substantial optional plotting that is not part of the saved decoder dataset.

ii.
```python
trial_starts = trial_start_times[trial_indices]
```

```python
n_control_responded = np.sum((outcome[is_control] == 'hit') | (outcome[is_control] == 'miss'))
n_correct = np.sum(outcome[is_control] == 'hit')
performance = n_correct / n_control_responded if n_control_responded > 0 else 0
```

```python
tongue_discretized, p40, p60 = compute_tongue_y_all_trials(
    tongue_ts, tongue_data_raw, go_times, BIN_EDGES)
```

```python
if show_processing:
    plot_processing(basename, neural_data, input_data, output_data,
                   go_times, BIN_CENTERS, good_regions, trial_outcomes, trial_instructions)
```

iii. `CONVERSION_NOTES.md` Step 6 says the plots were added for visual verification, not because they were needed by the output format. The remaining unused intermediates are simply artifacts of how the script was written.
