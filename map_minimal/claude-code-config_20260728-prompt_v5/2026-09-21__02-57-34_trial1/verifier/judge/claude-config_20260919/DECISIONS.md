# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovered the dataset as one NWB file per session laid out under `data/sub-<subject_id>/`, collected every file with a single sorted glob, and opened each one with `pynwb.NWBHDF5IO`. Within a file it reads the trials table (`nwb.trials`), the unit table (`nwb.units`), the behavioural event streams (`nwb.acquisition['BehavioralEvents']`), the side-camera tracking (`nwb.acquisition['BehavioralTimeSeries']`) and the subject record (`nwb.subject`). All 174 files are visited once, in sorted order; each file is processed inside a `try/except` so a failing file is reported and skipped rather than aborting the run. Brain regions per neuron are taken from the probe's `electrode_group.location` JSON (`brain_regions` field) rather than from the per-unit CCF annotation.

ii.
```python
def main():
    nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*.nwb'))
    print(f"Found {len(nwb_files)} NWB files")
    ...
    for i, nwb_path in enumerate(nwb_files):
        ...
        try:
            result = process_session(nwb_path)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            continue
```

```python
def process_session(nwb_path):
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()

    subject_id = str(nwb.subject.subject_id) if nwb.subject else 'unknown'
    trials = nwb.trials
    n_trials_total = len(trials)
    units = nwb.units

    be = nwb.acquisition['BehavioralEvents']
    go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
    sample_start_times = be.time_series['sample_start_times'].timestamps[:]
```

iii. From the trajectory: the AI first listed `/app/data`, found `sub-*` directories each holding a single `*.nwb` file, and confirmed "The data is in NWB format" with `find /app/data -name "*.nwb" | wc -l` returning 174. It then probed one file interactively to enumerate `nwb.trials.colnames`, `nwb.units.colnames`, `nwb.acquisition` and `nwb.subject` before writing any conversion code. The reference code in `/app/code` works on `.mat` files, so the AI explicitly noted "the reference code works with .mat files. But our data is in NWB format" and mapped the reference quantities onto the NWB fields. For brain regions it weighed the per-unit CCF annotation (`anno_name`) against the probe-level `electrode_group.location`, and chose the latter because "electrode_group.location.brain_regions already reflects the recording target consistently across all units", whereas many `anno_name` values were empty and would have needed a hand-written mapping into major areas.

## 1-b. How are the data split into subjects (mice)?

i. Each session's animal is read from `nwb.subject.subject_id` (the numeric DANDI id, e.g. `'440956'`), falling back to the string `'unknown'` if the file has no subject record. The unique ids are accumulated in a set over all retained sessions, sorted into the `subjects` list, and each session gets an index into that list in `subject_idx`. No re-grouping by folder name or by mouse name from the paper is done. The result is 28 subjects with 3-10 sessions each.

ii.
```python
subject_id = str(nwb.subject.subject_id) if nwb.subject else 'unknown'
```

```python
all_subjects.add(result['subject_id'])
...
subjects = sorted(list(all_subjects))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_to_idx[s['subject_id']] for s in all_sessions], dtype=np.int64),
```

iii. The AI treated `subject_id` as the canonical animal identifier because it is the field NWB provides and it is the same string that names the containing directory (`sub-440956`), so no separate grouping step is needed. The trajectory shows it verified the per-subject session counts in the decoder's verification output (`Subject 480927: 10 sessions`, etc.) and reported "28 subjects" as a sanity check against the dandiset.

## 1-c. How are the data split into sessions?

i. One NWB file is one session; the file boundary is the session boundary and no grouping is performed. Session order in the output follows the sorted file list (which, because the filename embeds the acquisition timestamp, is chronological within each subject). The AI explicitly considered splitting a file into several "sessions" when different probes covered different trial ranges, but after checking the data found that all good units within a file share one `obs_intervals` set, so it kept one session per file. Sessions are not given an id field in the output (no `session_info` in `metadata`; only `n_sessions` is stored). 24 of the 174 files are dropped: 1 because it has no QC-classified units, the rest by the behavioural session-selection criteria described in 1-e; 150 sessions reach the output.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*.nwb'))
```

```python
    # --- Determine valid trials based on obs_intervals ---
    # Use the obs_intervals from the first good unit (all units in a session
    # share the same obs_intervals since they come from the same probe set)
    obs_intervals = units['obs_intervals'][int(good_idx[0])]
```

```python
'metadata': {
    ...
    'n_sessions': len(all_sessions),
}
```

iii. From the trajectory, the AI initially suspected the NWB file bundled several recording blocks ("A better fix might be to treat each distinct group of units with matching trial coverage as its own separate 'session'"), then tested this by grouping good units by their `obs_intervals` across several files and found "ALL 375 good units come from a single probe ... All units in each session share the same obs_intervals (single group)". It therefore kept the one-file-one-session mapping and instead solved the coverage problem at the trial level.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table, and each trial is paired with the corresponding entry of the `go_start_times` event stream by position. The AI verified that the two have equal length and drops the whole session if they do not. Per-trial columns (`outcome`, `early_lick`, `trial_instruction`, `photostim_onset`, `photostim_duration`, `start_time`) are read as whole-column arrays and then indexed by the retained trial indices. The AI checked explicitly that `sample_start_times` and `delay_start_times` can have *more* entries than trials (because an early lick replays the epoch) while `go_start_times` has exactly one per trial, which is why the go cue is used as the per-trial anchor.

ii.
```python
be = nwb.acquisition['BehavioralEvents']
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
sample_start_times = be.time_series['sample_start_times'].timestamps[:]

if len(go_cue_times_all) != n_trials_total:
    io.close()
    return None

outcomes_all = trials['outcome'][:]
early_licks_all = trials['early_lick'][:]
instructions_all = trials['trial_instruction'][:]
photostim_onset_all = trials['photostim_onset'][:]
photostim_duration_all = trials['photostim_duration'][:]
trial_starts_all = trials['start_time'][:]
```

iii. The trajectory records the check that produced this design: "(368 trials vs 405 sample starts and 395 delay starts) suggest early licks during the sample period triggered replays, restarting the sample/delay period without changing the trial count, while go_start_times still lines up exactly with the 368 trials for spike alignment." The trials table is therefore used directly as the trial definition, and the length equality with `go_start_times` is enforced as a guard.

## 1-e. How are trials filtered based on quality controls?

i. Two levels of curation are applied.

**Trial level:** a trial is kept only if its go cue falls inside one of the unit `obs_intervals` (taken from the first good unit, since all good units in a session share them). This removes the behavioural trials for which the ephys recording was not running — in some sessions the great majority (e.g. 160 of 480 trials retained). A session with fewer than 2 surviving trials is dropped. No behavioural trial filter is applied: early-lick, `miss` and `ignore` trials are deliberately kept because they are decoder outputs. **No `free_water` filter is applied**, so the `free_water` trials — which lie inside `obs_intervals` but contain no spikes at all — remain, and the final dataset still contains ~629 trials whose neural matrix is entirely zero (reported as warnings by `train_decoder.py`, which the AI did not act on).

**Session level:** after trial filtering, the paper's behavioural session-selection criteria are re-applied to the surviving trials — overall performance > 65% on control (no-photostim, no-early-lick) trials, and at least 50 correct lick-left and 50 correct lick-right control trials. Sessions failing either criterion are dropped. Together with the one un-curated session this removes 24 of 174 files, leaving 150 sessions and 81,045 trials.

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
        # Check if go cue falls within any obs interval
        for obs_start, obs_end in obs_intervals:
            if go >= obs_start and go <= obs_end:
                valid[t_i] = True
                break

    return valid
```

```python
    valid_idx = np.where(valid_trial_mask)[0]
    if len(valid_idx) < 2:
        io.close()
        return None
```

```python
    # --- Session selection criteria (on valid trials only) ---
    is_control = np.array([po == 'N/A' for po in photostim_onset])
    is_no_early = early_licks == 'no early'
    is_responded = outcomes != 'ignore'
    control_responded = is_control & is_no_early & is_responded
    ...
    n_correct_control = np.sum((outcomes == 'hit') & is_control & is_no_early)
    performance = n_correct_control / np.sum(control_responded)

    correct_left = np.sum((outcomes == 'hit') & (instructions == 'left') & is_control & is_no_early)
    correct_right = np.sum((outcomes == 'hit') & (instructions == 'right') & is_control & is_no_early)

    if performance < MIN_PERFORMANCE:
        io.close()
        return None
    if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
        io.close()
        return None
```

iii. The trial filter was added reactively: the first conversion produced hundreds of all-zero trials, and the AI traced this to `obs_intervals` — "the first unit has 160 observation intervals for 480 trials ... This means I need to filter trials to only include those where neural data exists." It verified across several sessions that all good units share one `obs_intervals` set before using the first good unit's intervals as the session-wide mask.

The session filter comes straight from `methods.txt`: "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each." The AI restated this in its plan as ">65% correct on control non-early-lick trials, ≥50 correct per lick direction" and computed performance on control, non-early-lick, responded trials as the methods specify. It deliberately did **not** apply the paper's companion rule that "Early lick trials and no response trials were excluded for analysis", reasoning that "the decoder outputs explicitly include early lick and outcome as per-trial labels—meaning those trials should likely be kept rather than dropped." It never revisited the fact that it applies the session criteria to the ephys-covered subset rather than the full behavioural session, nor that all-zero (`free_water`) trials survive.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` array is built from `units/spike_times` (session-absolute spike times) for the units passing quality control, together with `BehavioralEvents/go_start_times` which supplies the per-trial alignment time. Spike trains are pulled one unit at a time into a Python list before binning. `units/classification` selects the units and `units/obs_intervals` selects the trials; `electrode_group.location` supplies each retained unit's brain-region label.

ii.
```python
    classifications = units['classification'][:]
    good_idx = np.where(classifications == 'good')[0]
    ...
    spike_times_list = [units['spike_times'][int(idx)] for idx in good_idx]
    fr_all = compute_firing_rates_all(spike_times_list, go_cue_times)
```

```python
    regions = []
    for idx in good_idx:
        eg = units['electrode_group'][int(idx)]
        regions.append(get_brain_region(eg))
```

iii. The AI confirmed interactively that `spike_times` is the only neural representation in the file and that the times are on the session-absolute clock ("Spike times are in absolute time - need to subtract go cue time"), so firing rates are computed directly from them.

## 2-b. How is the `neural` data processed?

i. For each good unit and each retained trial, the spikes falling in `[go - 2.5 s, go + 1.5 s)` are located with two `searchsorted` calls, expressed relative to the go cue, histogrammed into the 80 fixed 50-ms bins, and divided by the bin width to give a firing rate in spikes/s (Hz). Units with no spikes in a trial keep the pre-allocated zeros. No smoothing, normalisation, baseline subtraction or z-scoring is applied. The result is a `(n_units, n_trials, 80)` `float32` array, sliced per trial into the output list.

ii.
```python
    fr_all = np.zeros((n_units, n_trials, N_BINS), dtype=np.float32)

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

```python
    for t_i in range(n_trials):
        neural_trials.append(fr_all[:, t_i, :])
```

iii. The AI's stated decision was "Firing rates computed as spike counts divided by bin width" with "50ms non-overlapping bins", which it took from the instructions ("Use 50-ms-width bins for computing firing rates") and matched against the reference code's `sliding_histogram(..., rate=True)`. It kept the rates raw because the downstream decoder does its own normalisation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `units/classification` equals `'good'` are retained; no thresholds are placed on any individual quality metric, and the older `unit_quality` label is not used. A session with zero `'good'` units is dropped entirely (this happens for `sub-440958_ses-20190216T162508`, whose `classification` column is NaN for all units, so the string comparison yields no matches). Across the 150 retained sessions this keeps 59,949 units, a mean of 399.7 per session.

ii.
```python
    # --- Filter good units ---
    classifications = units['classification'][:]
    good_idx = np.where(classifications == 'good')[0]

    if len(good_idx) == 0:
        io.close()
        return None
```

iii. The AI's header comment states: "Unit filtering: classification == 'good' (QC classifier approach from the paper)". It arrived at this from `methods.txt`, which describes the five region-specific logistic-regression classifiers that label each cluster `'good'` or `'unlabeled'` and reports "69,943 good units recorded across 173 behavioral sessions"; the trajectory shows it checking `unique(classification)` to confirm the two labels and counting good units per session against that figure.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go cue onset, taken from `BehavioralEvents/go_start_times`. Spike times, go cues and camera timestamps all live on the same session-absolute clock, so alignment is a subtraction: for each trial the spike window is found with `searchsorted` at `go + T_START` and `go + T_END`, the spikes are re-expressed as `st - go`, and the shared go-cue-relative bin edges are applied. There is no resampling, interpolation, or per-stream offset correction.

ii.
```python
T_START = -2.5    # seconds relative to go cue
T_END = 1.5       # seconds relative to go cue
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
```

```python
            i_lo = np.searchsorted(st, go + T_START, side='left')
            i_hi = np.searchsorted(st, go + T_END, side='left')
            if i_hi > i_lo:
                rel = st[i_lo:i_hi] - go
                counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

```python
'temporal_alignment_event': 'Go cue onset',
'off_start': T_START,
'off_end': T_END,
```

iii. The instructions require alignment on go cue onset. The AI identified `go_start_times` as the go cue stream ("Go cue times are in `go_start_times` timestamps - this is what we align to"), verified there is exactly one such event per trial, and verified that spike times are absolute so that subtracting the go cue is sufficient.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 of them, spanning -2.5 s to +1.5 s relative to the go cue. The edge grid is computed once at module scope with `np.linspace` and reused for every trial in every session, so `n_timepoints` is exactly 80 everywhere (the verification output confirms `T: min 80, max 80`). Bins are non-overlapping and contiguous; no second rebinning, downsampling or sliding-window smoothing is applied to any stream — the input and output streams are built directly on the same grid. `time_bin_size` is recorded in the metadata in ms.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms
T_START = -2.5    # seconds relative to go cue
T_END = 1.5       # seconds relative to go cue
N_BINS = int(round((T_END - T_START) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
```

```python
'time_bin_size': BIN_WIDTH * 1000,  # in ms
```

iii. Bin width and window are set by the instructions ("Extract 2.5 s before to 1.5 s after the go cue", "Use 50-ms-width bins"). The AI computed the count explicitly in its planning — "the [-2.5, 1.5] window with 50ms bins yields 80 total time bins ... using non-overlapping bins for spike counting" — and defined the grid once so that every trial has identical length, which the target format requires.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (the sample-epoch/tone onsets) together with the trial's go cue and the *previous* trial's go cue. For each trial the AI takes the last sample-start that falls in the interval (previous go cue, this go cue]; if no sample-start falls in that interval, it falls back to the nominal offset of -1.85 s (0.65 s sample + 1.2 s delay).

ii.
```python
    # --- Tone onset per trial ---
    sorted_ss = np.sort(sample_start_times)
    tone_onset_rel_go = np.full(n_trials, -1.85)
    for t_i in range(n_trials):
        go = go_cue_times[t_i]
        # Find the original trial index to get the previous go cue correctly
        orig_idx = valid_idx[t_i]
        prev_go = go_cue_times_all[orig_idx - 1] if orig_idx > 0 else 0
        i_lo = np.searchsorted(sorted_ss, prev_go, side='right')
        i_hi = np.searchsorted(sorted_ss, go, side='right')
        if i_hi > i_lo:
            tone_onset_rel_go[t_i] = sorted_ss[i_hi - 1] - go
```

iii. The AI established empirically that `sample_start_times` has more entries than trials and that the gap from sample start to go cue "is usually 1.85s (sample=0.65s + delay=1.2s), but can vary because of early lick replays. The last sample_start before the go cue is the one that matters - that's when the tone actually started for that trial." It bounded the search below by the previous trial's go cue so that a tone from an earlier trial can never be picked up, and used the nominal 1.85 s as a fallback for the degenerate case where no candidate exists in that interval.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The per-trial tone offset (`tone - go`, a negative number) is subtracted from each go-cue-relative bin centre, giving seconds elapsed since tone onset at each of the 80 bins. It is stored as a continuous `float32` time series in row 0 of the `(2, 80)` input array. Values are not clipped, so bins before the tone are negative (typically down to about -0.6 s) and trials with replayed sample epochs can reach ~12 s.

ii.
```python
        time_from_tone = (BIN_CENTERS - tone_onset_rel_go[t_i]).astype(np.float32)
        inp = np.stack([time_from_tone, photostim_on_all[t_i]], axis=0)
        input_trials.append(inp)
```

iii. "time from tone onset tracked continuously as sample start minus go cue" — the AI treated this as a plain affine shift of the bin grid, noting in its planning that "since sample period timing shifts per trial due to early lick replays, I need to find the actual sample_start_time per trial, compute its offset from go cue, and then for each time bin subtract that offset from the bin's relative time to get time-from-tone at each point." The instructions ask for a continuous, time-varying input, so no discretisation is applied.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is defined on exactly the grid used for the firing rates: the same `BIN_CENTERS` array derived from the same `BIN_EDGES`, placed on the same per-trial go cue. Bin *k* of the input therefore covers the same interval as bin *k* of the neural matrix by construction, and no separate alignment step exists.

ii.
```python
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
```

```python
        time_from_tone = (BIN_CENTERS - tone_onset_rel_go[t_i]).astype(np.float32)
```

iii. Both streams are expressed relative to the same go cue on the same absolute clock, so the AI simply reused the module-level grid rather than re-deriving or interpolating anything.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the trials-table columns `photostim_onset` and `photostim_duration` (both stored as strings, with `'N/A'` on unstimulated trials), plus `start_time` for the trial and the trial's go cue, which are needed to move the onset onto the go-cue-relative axis. The `photostim_power` column is not used, and no distinction is made between left, right and bilateral ALM stimulation — the input is a single binary channel.

ii.
```python
    photostim_onset_all = trials['photostim_onset'][:]
    photostim_duration_all = trials['photostim_duration'][:]
    trial_starts_all = trials['start_time'][:]
```

```python
        if photostim_onset[t_i] != 'N/A':
            ps_onset_abs = trial_starts[t_i] + float(photostim_onset[t_i])
            ps_dur = float(photostim_duration[t_i])
```

iii. The AI checked a VGAT-ChR2 session and found "photostim_onset is stored as strings ('N/A' or numeric strings)". It then tested whether the number is absolute or trial-relative, reasoning "if photostim_onset is relative to trial start and go_time ≈ trial_start + presample(0.5s) + sample(0.65s) + delay(1.2s) ≈ trial_start + 2.35s, then a photostim_onset of 2.522 would fall about 0.17s before the go cue -- consistent with the late delay window. This confirms photostim_onset is measured from trial start". That is consistent with `methods.txt`: "We silenced ALM activity during the late delay epoch (last 0.5 s) ... photoinhibition always ended before the 'Go' cue."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The onset is converted to absolute time (`trial start + onset`), then to go-cue-relative time (`- go`), and the offset is onset + duration. A bin is set to 1.0 when its centre lies in `[onset, offset)` and 0.0 otherwise, producing a binary time series rather than a per-trial flag. Unstimulated trials are left as the pre-allocated all-zero row, so the `'N/A'` strings are never parsed. Stored as `float32` in row 1 of the input array.

ii.
```python
    # --- Photostim binary per trial ---
    photostim_on_all = np.zeros((n_trials, N_BINS), dtype=np.float32)
    for t_i in range(n_trials):
        if photostim_onset[t_i] != 'N/A':
            ps_onset_abs = trial_starts[t_i] + float(photostim_onset[t_i])
            ps_dur = float(photostim_duration[t_i])
            ps_onset_rel = ps_onset_abs - go_cue_times[t_i]
            ps_offset_rel = ps_onset_rel + ps_dur
            photostim_on_all[t_i] = ((BIN_CENTERS >= ps_onset_rel) & (BIN_CENTERS < ps_offset_rel)).astype(np.float32)
```

iii. The instructions ask for "Whether photostimulation is on at every time point (discrete, time-varying)", which the AI implemented literally as a per-bin binary indicator: "Photostim: Binary time series from trial-level onset/duration, converted to go-cue-relative timing." Skipping the `'N/A'` trials entirely avoids having to parse a non-numeric string.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulation onset and offset are re-expressed relative to the same go cue that anchors the neural bins, and are then compared directly against the shared `BIN_CENTERS`, so the photostim row lives on exactly the same 80-bin grid as the firing rates.

ii.
```python
            ps_onset_rel = ps_onset_abs - go_cue_times[t_i]
            ps_offset_rel = ps_onset_rel + ps_dur
            photostim_on_all[t_i] = ((BIN_CENTERS >= ps_onset_rel) & (BIN_CENTERS < ps_offset_rel)).astype(np.float32)
```

iii. Because the raw onset is stored relative to trial start, it had to be converted before it could be compared with the go-cue-relative bin grid; the AI noted this explicitly ("the reference paper's code shifted stimulation timestamps by subtracting the go-cue time"). With that conversion done, no further alignment is needed.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the file, so choice is derived from two trials-table columns: `trial_instruction` (`'left'`/`'right'`) and `outcome` (`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed port, a miss the opposite port, and an ignore means it did not lick.

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

iii. The AI reasoned it out in the trajectory: "trial_instruction gives the correct direction while outcome (hit/miss/ignore) tells us if the animal licked correctly, incorrectly, or not at all, so I can derive choice by combining instruction with outcome—hit means choice matches instruction, miss means choice is the opposite direction, and ignore means no lick." It also checked the cross-tabulation of `early_lick` against `outcome` and confirmed that early-lick trials still carry a valid hit/miss/ignore outcome after the epoch replay, so they can be assigned a choice like any other trial.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived code (0 = left, 1 = right, 2 = no lick) is one value per trial, broadcast across all 80 bins so that it can share a single `(4, 80)` `int64` output array with the other three outputs. `output_values[0] = ['left', 'right', 'no_lick']` names the codes. Any unexpected outcome string would fall into the `else` branch and be coded as no-lick.

ii.
```python
        out = np.stack([
            np.full(N_BINS, choice, dtype=np.int64),
            ...
        ], axis=0)
        output_trials.append(out)
```

```python
'output_values': [
    ['left', 'right', 'no_lick'],
    ...
],
```

iii. Left = 0 / right = 1 follows the ordering in the instructions, with a third class for the no-lick case the instructions list. The AI kept the per-trial values time-invariant but replicated them across bins after reading the decoder code, concluding that "choice, outcome, and early_lick remain fixed per-trial constants" while the format expects one `(n_output, n_timepoints)` array per trial. The dtype was switched from `float64` to `int64` after the verification script failed indexing `output_values` with a float.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which already contains exactly the three strings `'ignore'`, `'miss'` and `'hit'` that the instructions ask for.

ii.
```python
    outcomes_all = trials['outcome'][:]
    ...
    outcomes = outcomes_all[valid_idx]
```

iii. The AI enumerated `np.unique(trials['outcome'][:])` early on and found exactly `hit`/`miss`/`ignore`, so no derivation was needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The string is mapped through a fixed dictionary to 0 = ignore, 1 = miss, 2 = hit and written into row 1 of the per-trial output array, repeated across all 80 bins. `.get(o, 0)` is used, so an unrecognised string would silently become `ignore`.

ii.
```python
    outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
```

```python
            np.full(N_BINS, outcome_map.get(o, 0), dtype=np.int64),
```

iii. The code order follows the order the instructions list the categories in ("Outcome (ignore, miss, hit, per-trial)"), and, like the other per-trial outputs, it is replicated across bins so all four outputs fit one array.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, whose values are the strings `'early'` and `'no early'`.

ii.
```python
    early_licks_all = trials['early_lick'][:]
    ...
    early_licks = early_licks_all[valid_idx]
```

iii. The flag is stored explicitly per trial, so no derivation is needed. The AI checked `np.unique(trials['early_lick'][:])` and cross-tabulated it with outcome to confirm that early-lick is an independent label that can co-occur with any outcome ("early_lick trials DO have outcomes (mostly hit after replay)").

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A direct string test: `'early'` → 1, anything else → 0. The scalar is repeated across all 80 bins in row 2 of the output array, with `output_values[2] = ['no', 'yes']`.

ii.
```python
            np.full(N_BINS, 1 if early_licks[t_i] == 'early' else 0, dtype=np.int64),
```

iii. The 0 = no / 1 = yes coding follows the instructions ("Early lick (no, yes, per-trial)"). The AI kept early-lick trials in the dataset rather than following the paper's analysis-time exclusion, precisely because this flag is a required decoder output; the lick that sets it happens during the sample/delay epoch and so falls inside the -2.5 s window.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 of `data` is the tongue y-coordinate and column 2 is the DeepLabCut likelihood, with `timestamps` giving the ~300 Hz frame times on the session clock. Column 0 (x) is read but unused. If a session lacked the series entirely, the AI falls back to marking every bin `not_visible`.

ii.
```python
    tongue_y_vals = tongue_data[:, 1]
    tongue_lk_vals = tongue_data[:, 2]
```

```python
    bts = nwb.acquisition.get('BehavioralTimeSeries', None)
    has_tongue = (bts is not None and 'Camera0_side_TongueTracking' in bts.time_series)

    if has_tongue:
        tongue_ts = bts.time_series['Camera0_side_TongueTracking']
        tongue_data = tongue_ts.data[:]
        tongue_timestamps = tongue_ts.timestamps[:]
        tongue_y_per_trial = discretize_tongue_y(tongue_timestamps, tongue_data, go_cue_times)
    else:
        tongue_y_per_trial = [np.full(N_BINS, 3, dtype=np.int64) for _ in range(n_trials)]
```

iii. This is the only tongue measurement in the file; the AI inspected the series and recorded "Tongue tracking is available as `Camera0_side_TongueTracking` with (x, y, likelihood) at ~300Hz", consistent with `methods.txt` ("High-speed videos from a side view and a bottom view were acquired at 300 Hz ... We trained DeepLabCut to track the movement of tongue, jaw and nose"). It also checked that tongue tracking "appears available everywhere", and still guarded against its absence.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames are treated as showing the tongue only when the DLC likelihood is ≥ 0.9. Within each 50 ms bin of each trial, the y-values of the visible frames are averaged; that bin mean is what gets discretised. Bins with no frame, or no visible frame, are left in the pre-filled `not_visible` class. The percentile thresholds are computed once per session, in a first pass over the same trial windows, from the *raw visible frame* y-values (not from the bin means).

ii.
```python
    # First pass: collect visible y values for percentiles
    all_visible_y = []
    for t_i in range(n_trials):
        go = go_cue_times[t_i]
        i_start = np.searchsorted(tongue_timestamps, go + T_START, side='left')
        i_end = np.searchsorted(tongue_timestamps, go + T_END, side='left')
        if i_end > i_start:
            lk = tongue_lk_vals[i_start:i_end]
            vis = lk >= TONGUE_LIKELIHOOD_THRESHOLD
            if np.any(vis):
                all_visible_y.append(tongue_y_vals[i_start:i_end][vis])

    if len(all_visible_y) > 0:
        all_vis = np.concatenate(all_visible_y)
        p40 = np.percentile(all_vis, 40)
        p60 = np.percentile(all_vis, 60)
    else:
        p40 = p60 = 0.0
```

iii. The AI found that the likelihood is strongly bimodal and that the tongue is visible in only about 10% of frames: "Tongue tracking has low likelihood most of the time (tongue not visible), only ~10% of frames have high confidence." It chose the threshold accordingly — "'Not visible' is probably defined by a DLC likelihood threshold, likely 0.9 given that's a common convention and matches the ~10% visible frame rate observed" — and decided that "the percentiles for y-position are likely computed across all visible tongue frames within the session, since the tongue only shows during licking bouts", i.e. invisible frames must not contaminate the percentiles.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session: class 0 if the bin's mean visible y is below the 40th percentile, class 1 if it is between the 40th and 60th percentiles (inclusive of the upper edge), class 2 if above the 60th, and class 3 if the bin contains no visible frame. Percentiles are per session, computed from raw visible frames pooled over that session's retained trial windows. In the delivered dataset this yields class fractions 0.123 / 0.065 / 0.063 / 0.749 — i.e. of the ~25% visible bins, roughly 49% / 26% / 25% rather than the nominal 40% / 20% / 40%, because the thresholds are percentiles of individual frames but are applied to within-bin averages.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
```

```python
    # Second pass: discretize per bin
    result = []
    for t_i in range(n_trials):
        go = go_cue_times[t_i]
        ty_binned = np.full(N_BINS, 3, dtype=np.int64)
        abs_edges = go + BIN_EDGES
        edge_indices = np.searchsorted(tongue_timestamps, abs_edges, side='left')

        for b_i in range(N_BINS):
            i_start = edge_indices[b_i]
            i_end = edge_indices[b_i + 1]
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
        result.append(ty_binned)
```

```python
'output_values': [
    ...
    ['low', 'mid', 'high', 'not_visible'],
],
```

iii. The class definitions are taken verbatim from the instructions' per-session discretisation table, including the fourth "not visible" class. The AI read "percentile of y-position over the session" literally as percentiles of the y-position samples, and scoped "over the session" to the frames inside that session's extracted trial windows. It did not check the resulting class balance of the emitted bin labels.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the session-absolute clock with the spikes and the go cues, so the same go-cue-relative edge grid is converted to absolute times (`go + BIN_EDGES`) and `searchsorted` gives the frame index range of each of the 80 bins directly. Bin *k* of the tongue output therefore spans the same interval as bin *k* of the firing rates. No interpolation or offset correction is applied. Where the video does not cover a bin (the video is trial-gated, so the leading bins of a short trial contain no frames), the bin stays in the `not_visible` class.

ii.
```python
        abs_edges = go + BIN_EDGES
        edge_indices = np.searchsorted(tongue_timestamps, abs_edges, side='left')

        for b_i in range(N_BINS):
            i_start = edge_indices[b_i]
            i_end = edge_indices[b_i + 1]
```

iii. The AI treated all NWB streams as sharing one clock (it verified this for spikes and events) and therefore reused the neural bin grid for the camera, which guarantees bin-for-bin correspondence by construction. This is the only genuinely time-varying output, so it is the only one that required real alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases, handled defensively but not uniformly:

- **Session never quality-controlled** (`classification` NaN for all units): the `== 'good'` comparison matches nothing, `good_idx` is empty, and the session is dropped.
- **Missing subject record**: `subject_id` falls back to the string `'unknown'`.
- **Go-cue/trial-count mismatch**: the session is silently dropped (`return None`) rather than raising.
- **Missing tongue tracking series**: every bin of every trial is set to the `not_visible` class rather than crashing.
- **Frames with no visible tongue**: excluded from the bin mean by the likelihood threshold; a bin with no visible frame becomes the explicit `not_visible` category rather than being imputed.
- **Any unexpected exception in a session**: caught in `main`, the traceback is printed and the session is skipped.

What is *not* handled: trials that are inside `obs_intervals` but contain no spikes at all (the `free_water` trials). These are kept, so the delivered dataset contains roughly 629 trials whose entire neural matrix is zero — `train_decoder.py` reported each of them as a warning on the final run. Unknown category strings are also silently coerced rather than flagged (`outcome_map.get(o, 0)` → ignore; `early_lick != 'early'` → no; non-left instruction → right).

ii.
```python
    subject_id = str(nwb.subject.subject_id) if nwb.subject else 'unknown'
```

```python
    if len(go_cue_times_all) != n_trials_total:
        io.close()
        return None
```

```python
    if len(good_idx) == 0:
        io.close()
        return None
```

```python
    else:
        tongue_y_per_trial = [np.full(N_BINS, 3, dtype=np.int64) for _ in range(n_trials)]
```

```python
        try:
            result = process_session(nwb_path)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            continue
```

iii. The AI's general approach was to drop whatever cannot be trusted and to represent genuinely-absent measurements as an explicit category. The obs_intervals trial filter was introduced specifically in response to the all-zero-trial warnings from the first verification run ("There are many trials with all-zero neural data ... this suggests an alignment issue"), and the AI verified the fix reduced them. It did not re-read the warning block of the final verification run (it inspected only the tail of the training log), so it never noticed that a residual population of all-zero trials survived.

## 10-a. What are the most time-consuming steps of the code?

i. Three steps dominate. (1) Reading each NWB file, in particular pulling `spike_times` one unit at a time through `units['spike_times'][int(idx)]` (one HDF5 ragged read per unit, hundreds per session) and loading the full `(n_frames, 3)` tongue array. (2) The spike-binning double loop, which runs `n_units × n_trials` iterations of `searchsorted` + `np.histogram` — for a 600-unit, 600-trial session that is ~360,000 `np.histogram` calls. (3) The tongue discretisation, which makes two passes over the session's frames and executes an inner Python loop over all 80 bins for every trial. Writing the 10.7 GB pickle at the end is also significant. The first version of the script was slow enough that the AI aborted the run (~49 of 174 sessions after several minutes) and rewrote the hot paths.

ii.
```python
    spike_times_list = [units['spike_times'][int(idx)] for idx in good_idx]
```

```python
    for u_i, spike_times in enumerate(spike_times_list):
        ...
        for t_i in range(n_trials):
            ...
                counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

```python
        for b_i in range(N_BINS):
            i_start = edge_indices[b_i]
            i_end = edge_indices[b_i + 1]
```

iii. The AI diagnosed this at runtime: "It's still running but slow - the tongue tracking per-bin computation is the bottleneck. Let me kill this and optimize", then "Let me optimize the tongue tracking and spike binning using vectorized operations." The rewrite narrowed the spike search with `searchsorted` before histogramming and precomputed the per-bin edge indices for the tongue, which was enough to finish the full 174 files inside the run budget; it did not go further.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Five, none of which is inherently sequential:

- `get_valid_trial_mask`: a pure-Python `n_trials × n_obs_intervals` double loop; this is a single `np.isin`/`searchsorted` over the interval starts.
- `compute_firing_rates_all`: the inner per-trial loop could be removed by flattening all trials' bin edges into one array and calling `searchsorted` once per unit, then differencing (the outer per-unit loop is inherent, since the spike trains are ragged).
- `discretize_tongue_y`: both the first (percentile-collection) pass and the inner 80-iteration per-bin loop could be replaced by a single global bin index plus `np.bincount` for sums and counts.
- the tone-onset loop, which is one vectorised `searchsorted` of `sample_start_times` against the go cues.
- the final per-trial assembly loop, which builds `np.full`/`np.stack` arrays one trial at a time instead of broadcasting into one preallocated `(n_trials, 4, 80)` array.

ii.
```python
    for t_i in range(n_trials):
        go = go_cue_times[t_i]
        # Check if go cue falls within any obs interval
        for obs_start, obs_end in obs_intervals:
            if go >= obs_start and go <= obs_end:
```

```python
    for t_i in range(n_trials):
        go = go_cue_times[t_i]
        orig_idx = valid_idx[t_i]
        prev_go = go_cue_times_all[orig_idx - 1] if orig_idx > 0 else 0
        i_lo = np.searchsorted(sorted_ss, prev_go, side='right')
        i_hi = np.searchsorted(sorted_ss, go, side='right')
```

```python
    for t_i in range(n_trials):
        neural_trials.append(fr_all[:, t_i, :])
        time_from_tone = (BIN_CENTERS - tone_onset_rel_go[t_i]).astype(np.float32)
        inp = np.stack([time_from_tone, photostim_on_all[t_i]], axis=0)
```

iii. The AI's stated goal in the rewrite was to "optimize the tongue tracking and spike binning using vectorized operations", and it did vectorise the innermost work (bin-edge lookup, boolean masks). It stopped once the script completed in acceptable time and did not revisit the remaining per-trial loops, which it never identified as costly.

## 10-c. What processing does the code repeat multiple times?

i. Several quantities are recomputed:

- `np.sort(spike_times)` is applied to every unit's spike train even though `units/spike_times` is already stored sorted; likewise `np.sort(sample_start_times)`, which is already monotonic.
- `discretize_tongue_y` walks each trial's frame range twice — once to collect visible y-values for the percentiles, once to bin them — duplicating the `searchsorted` lookups and the likelihood masking.
- `json.loads(electrode_group.location)` is called once per unit, re-parsing the same handful of probe JSON blobs hundreds of times per session.
- `units['spike_times'][int(idx)]` and `units['electrode_group'][int(idx)]` are per-unit HDF5 accesses rather than one read of the underlying buffer.
- The whole-column trial arrays are read and then re-sliced by `valid_idx` into seven parallel copies.

ii.
```python
        st = np.sort(spike_times)
```

```python
    sorted_ss = np.sort(sample_start_times)
```

```python
def get_brain_region(electrode_group):
    """Extract simplified brain region name from electrode group location."""
    loc = json.loads(electrode_group.location)
```

```python
    regions = []
    for idx in good_idx:
        eg = units['electrode_group'][int(idx)]
        regions.append(get_brain_region(eg))
```

iii. The repeats are defensive or incidental rather than deliberate — the AI never discussed them. The two-pass tongue structure is a genuine design choice: the percentile thresholds have to exist before any bin can be labelled, so a first pass over the session is needed; the AI simply did not cache the per-trial frame ranges between the two passes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little is outright discarded, but some work is wasted:

- The tongue x-coordinate (`tongue_data[:, 0]`) is loaded with the rest of the `(n_frames, 3)` array and never used.
- `st = np.sort(...)` and `sorted_ss = np.sort(...)` are no-ops on already-sorted data.
- The four outputs are stored as `int64` even though every value is in 0-3; `int8` would be 8× smaller, and the four per-trial scalars are each replicated 80× to fill the time axis (the replication itself is required by the chosen single-array output format).
- `performance` is computed for every session and carried in the result dict, but beyond the session filter and a progress printout it is not saved into the output (nor is any session identifier, so the provenance of each session is lost from the pickle).
- For the 24 rejected sessions all the file-opening and trials-table reading is thrown away (the session filter is at least placed before the expensive spike binning, so that work is not wasted).

ii.
```python
    tongue_y_vals = tongue_data[:, 1]
    tongue_lk_vals = tongue_data[:, 2]
```

```python
        out = np.stack([
            np.full(N_BINS, choice, dtype=np.int64),
            np.full(N_BINS, outcome_map.get(o, 0), dtype=np.int64),
            np.full(N_BINS, 1 if early_licks[t_i] == 'early' else 0, dtype=np.int64),
            tongue_y_per_trial[t_i].astype(np.int64)
        ], axis=0)
```

```python
    return {
        ...
        'performance': performance,
    }
```

iii. The `int64` dtype was a deliberate late fix, not an oversight about size: the first version used `float64` and `train_decoder.py` crashed indexing `output_values` with a `numpy.float64`, so the AI switched to `int64` ("I should switch the output dtype to int64 to fix this indexing issue") without considering a narrower integer type. The session filter is placed before the spike binning, so the most expensive step is not run for rejected sessions. The remaining items are incidental and were never discussed.
