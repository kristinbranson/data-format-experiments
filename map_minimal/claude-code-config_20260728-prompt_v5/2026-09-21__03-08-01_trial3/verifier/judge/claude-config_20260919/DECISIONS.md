# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session laid out as `/app/data/sub-<id>/*.nwb`. The AI walks that layout explicitly with `os.listdir`: it lists every `sub-*` directory, then every `*.nwb` file inside it, and calls `process_session` once per file (174 files). Each file is opened **not** with `pynwb` but directly as raw HDF5 with `h5py`, and the needed arrays are pulled by hard-coded HDF5 paths (`intervals/trials/*`, `units/*`, `acquisition/BehavioralEvents/*`, `acquisition/BehavioralTimeSeries/*`). Subjects, sessions and trials are therefore all enumerated in one nested loop; nothing is loaded lazily or a second time.

ii.
```python
def main():
    subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
    ...
    for sub in subjects:
        sub_dir = os.path.join(DATA_DIR, sub)
        nwb_files = sorted([f for f in os.listdir(sub_dir) if f.endswith('.nwb')])
        ...
        for nf in nwb_files:
            nwb_path = os.path.join(sub_dir, nf)
            print(f'Processing {nf}...', end=' ', flush=True)
            result = process_session(nwb_path)
```
```python
def process_session(nwb_path):
    with h5py.File(nwb_path, 'r') as f:
        n_trials = len(f['intervals/trials/id'])
        outcome = np.array([x.decode() for x in f['intervals/trials/outcome'][:]])
        ...
        go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
        spike_times_flat = f['units/spike_times'][:]
        spike_times_index = f['units/spike_times_index'][:]
```

iii. From the trajectory, the AI first explored the NWB hierarchy with `h5py` (steps 7-26: printing `intervals/trials` columns, `units` columns, `acquisition` groups, tracking array shapes) and then simply kept using `h5py` for the conversion because it already knew the exact dataset paths and h5py avoids the `pynwb` object-construction overhead. It never stated an explicit justification for choosing h5py over pynwb; the directory walk is justified implicitly by the one-file-per-session layout it had confirmed (174 files, 28 subject folders).

## 1-b. How are the data split into subjects?

i. Subjects are the `sub-*` directory names. The AI treats the folder name (e.g. `'sub-440956'`, with the prefix retained) as the subject id, rather than reading `general/subject/subject_id` from inside the file. The `subjects` list is built lazily: a subject name is appended the first time one of its sessions actually survives the selection criteria, and `subject_idx` for each session is `subject_names.index(sub)`. The list is therefore in order of first surviving session rather than sorted, and subjects with no surviving session would simply never be added (empirically none: all 28 subjects retain ≥1 session).

ii.
```python
    for sub in subjects:            # sub == 'sub-440956', ...
        ...
        sub_has_session = False
        for nf in nwb_files:
            result = process_session(nwb_path)
            if result is None:
                ...
                continue
            if not sub_has_session:
                subject_names.append(sub)
                sub_has_session = True
            sub_idx = subject_names.index(sub)
            all_subject_idx.append(sub_idx)
```
```python
        'subjects': subject_names,
        'subject_idx': np.array(all_subject_idx),
```

iii. The AI's reasoning (step 40 onward) took the `sub-*` folder layout as the definition of the animal grouping; it counted "sessions and subjects" from the directory structure before writing the script. It did not comment on the difference between the folder name and the in-file `subject_id`, nor on the mismatch between the numeric DANDI id and the mouse names (`SC015`, ...) used in the papers.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no further splitting. The AI adds a **session-level selection filter** on top: a session is kept only if, over "control" trials (no auto/free water, no early lick, no photostim, response present), behavioural performance is `> 0.65` **and** there are `>= 50` correct-left **and** `>= 50` correct-right control trials. Sessions whose `units/classification` column is all-NaN (never quality-controlled) are also dropped. Empirically this keeps **144 of the 174 sessions** (29 dropped on behaviour, 1 on missing QC) across all 28 subjects; the reference keeps 173. No session identifier (`nwb.identifier`) or per-session metadata is recorded in the output.

ii.
```python
        has_stim = photostim_onset_str != 'N/A'
        control_mask = (
            (auto_water == 0) & (free_water == 0) &
            (early_lick == 'no early') & (~has_stim) &
            (outcome != 'ignore')
        )
        n_control = control_mask.sum()
        if n_control == 0:
            return None

        correct_control = (outcome == 'hit') & control_mask
        performance = correct_control.sum() / n_control
        n_correct_left = ((trial_instruction == 'left') & correct_control).sum()
        n_correct_right = ((trial_instruction == 'right') & correct_control).sum()

        if performance <= 0.65 or n_correct_left < 50 or n_correct_right < 50:
            return None
```

iii. The criterion is quoted verbatim from `methods.txt`: *"We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each."* The AI reasoned (step 42/46) that performance should be computed on control trials only and that no-response (`ignore`) trials should be excluded from the denominator, mirroring how the reference code's `get_regular_trial_mask` treats `correctness == -1`. It decided to apply the criterion for *session selection* while still keeping early-lick/ignore/photostim trials *within* selected sessions, because those are required decoder output categories. It did not check its computed performance values against the paper's own statement that the released dataset already consists of 173 sessions with 65-99% correct rates.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table (`intervals/trials/id`), and every per-trial column (`outcome`, `early_lick`, `trial_instruction`, `auto_water`, `free_water`, `photostim_onset`, `photostim_duration`, `start_time`) is read positionally by that row index. The go cue for trial `i` is taken as `go_start_times.timestamps[i]`, i.e. a strict 1:1 positional correspondence between the trials table and the `go_start_times` event stream (true in this dataset, but never asserted in the code). A session needs `>= 2` surviving trials to be emitted.

ii.
```python
        n_trials = len(f['intervals/trials/id'])
        ...
        go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
        trial_start_times = f['intervals/trials/start_time'][:]
        ...
        for trial_idx in trial_indices:
            go_t = go_times[trial_idx]
```

iii. In step 34/38 the AI explicitly noticed that `sample_start_times` (405) and `delay_start_times` (395) have **more** entries than trials (368) because an early lick replays the sample/delay epoch, while `go_start_times` has exactly one entry per trial. It concluded that the go-cue stream is the one stream that can be indexed by trial number and used it as the trial anchor, deriving the tone onset separately (see 3-a).

## 1-e. How are trials filtered based on quality controls?

i. Only two behavioural columns are used: `auto_water` and `free_water` trials are dropped; everything else is kept, explicitly including early-lick, `ignore` (no-response) and photostim trials. A session is dropped if fewer than 2 trials survive. **No filter on spike-data availability is applied.** The AI considered `units/obs_intervals` and the `is_good_trials` matrix and decided not to use them. Empirically that leaves **1,056 trials in 7 of the 144 kept sessions that fall outside `obs_intervals`** (one session contributes 376 such trials); these trials have no recorded spikes at all and are emitted as 4 s of exactly 0 Hz across every unit. On the other side, the AI additionally removes **627 `auto_water` trials** that the expert keeps. In total the AI keeps ~75,812 trials versus the reference's 90,860.

ii.
```python
        # ── trial filtering: keep all except auto/free water ───────────
        trial_mask = (auto_water == 0) & (free_water == 0)
        trial_indices = np.where(trial_mask)[0]
        if len(trial_indices) < 2:
            return None
```

iii. The AI reasoned at length (steps 34, 42) that `methods.txt` ("Early lick trials and no response trials were excluded for analysis") and the reference code's `get_regular_trial_mask` conflict with the decoder specification, which *requires* `early lick = yes`, `no lick` and `ignore` as output classes and photostimulation as an input; it therefore kept those trials and applied the paper's exclusions only to the session-selection statistic. Auto/free-water trials were excluded as "outside the normal task structure". For spike availability it explicitly decided to "filter units to those classified as 'good' and use their spike times directly, letting `obs_intervals` and trial windows naturally handle cases with zero spikes rather than explicitly masking with `is_good_trials`" — i.e. it assumed a trial with no spikes is a legitimate zero.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (the flat ragged buffer) together with `units/spike_times_index` (the per-unit end offsets), restricted to units with `units/classification == 'good'`, and `acquisition/BehavioralEvents/go_start_times/timestamps` for the alignment. Per-unit spike arrays are sliced out once per session before the trial loop.

ii.
```python
        spike_times_flat = f['units/spike_times'][:]
        spike_times_index = f['units/spike_times_index'][:]
        ...
        unit_spikes = []
        for uid in good_indices:
            start_idx = 0 if uid == 0 else spike_times_index[uid - 1]
            end_idx = spike_times_index[uid]
            unit_spikes.append(spike_times_flat[start_idx:end_idx])
```

iii. The AI inspected the `units` group early (steps 8-20) and established that `spike_times` is the only neural representation in the file, stored as a ragged `VectorData`/`VectorIndex` pair. It never considered any pre-binned representation because none exists.

## 2-b. How is the `neural` data processed?

i. For each kept trial and each good unit, the unit's **entire** session-long spike train is shifted by the trial's go-cue time and passed to `np.histogram` with 81 edges spanning [-2.5, 1.5] s; the resulting counts are divided by the 50 ms bin width to give firing rate in Hz. No smoothing, no baseline subtraction, no normalisation. Result per trial is a `(n_good_units, 80)` **float64** array.

ii.
```python
def bin_spike_times(spike_times, t_start, t_end, bin_width):
    n_bins = int(round((t_end - t_start) / bin_width))
    edges = np.linspace(t_start, t_end, n_bins + 1)
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts / bin_width
```
```python
        for trial_idx in trial_indices:
            go_t = go_times[trial_idx]
            fr_matrix = np.zeros((n_good, n_bins))
            for u, spks in enumerate(unit_spikes):
                aligned = spks - go_t
                fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
            neural_trials.append(fr_matrix)
```

iii. The docstring states "Bin width: 50 ms non-overlapping bins for firing rates", following the decoder instructions; the AI noted in step 49 that the reference code's `sliding_histogram` returns `binSpikes / bin_width`, i.e. a rate, and matched that convention. It planned to "lean on vectorized numpy operations rather than loops" but the implementation did not do so (see 10-b).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept iff `units/classification == 'good'` — the verdict of the 15-metric logistic-regression QC classifier described in the Chen/Liu white paper. `units/unit_quality` (the Kilosort `good`/`multi` label) is deliberately **not** used, and no individual metric thresholds are applied. If the `classification` column is a float array (all NaN, i.e. the session was never run through the classifier) the whole session is dropped; that affects exactly one session, `sub-440958_ses-20190216T162508`, which the AI verified is also the only file with no CCF annotations. This is identical to the expert's unit-level curation.

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
        n_units_total = len(classification)
        good_indices = np.where(good_mask)[0]

        if len(good_indices) == 0:
            return None
```

iii. Step 34: *"the reference code uses 'classifier' QC mode, which is represented in the NWB as `classification == 'good'`; the `unit_quality` field is separate, with 'good'/'multi' labels distinct from the classifier-based `classification` field."* Step 57: after hitting the NaN session it checked `unit_quality` (1201/1852 'good') and reasoned that `unit_quality` "looks like the Kilosort output label rather than the more stringent 15-metric classifier QC. Since the reference code relies exclusively on the classifier QC, sessions missing `classification` may simply not have been part of that analysis", and that the same session also has no `anno_name`, so it dropped it (step 58 confirmed it is 1/174).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go-cue onset**, done by subtracting the trial's absolute go-cue timestamp from the unit's absolute spike times before binning in the fixed window [-2.5 s, +1.5 s]. Spike times, event timestamps and video timestamps are all on the same session-absolute clock (the AI verified this in step 45), so no offset correction, resampling or interpolation is used.

ii.
```python
T_START = -2.5         # seconds relative to go cue
T_END = 1.5
...
            go_t = go_times[trial_idx]
            for u, spks in enumerate(unit_spikes):
                aligned = spks - go_t
                fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
```
```python
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START,
            'off_end': T_END,
```

iii. Directly from the decoder instructions ("Temporally align based on Go cue onset. Extract 2.5 s before to 1.5 s after"). The AI confirmed in step 45 that `go_start_times` timestamps and the video/ephys streams share one time base starting at 0 for the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 bins per trial, fixed for every trial and session, generated by `np.linspace(-2.5, 1.5, 81)`. Bin centres are `edges[:-1] + 0.025`. No rebinning or resampling is applied to anything: spikes are binned once at 50 ms, and the two inputs and the tongue output are evaluated directly on the same 80 bin centres. `time_bin_size` is recorded in metadata as 50.0 ms.

ii.
```python
BIN_WIDTH = 0.05       # 50 ms
T_START = -2.5
T_END = 1.5
...
def get_bin_centers(t_start, t_end, bin_width):
    n_bins = int(round((t_end - t_start) / bin_width))
    edges = np.linspace(t_start, t_end, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2
```
```python
            'time_bin_size': BIN_WIDTH * 1000,  # in ms
```

iii. The 50 ms width and the window come straight from the decoder instructions; the AI noted (step 49) that the reference preprocessing also uses a 300 Hz video stride and histogram binning, and that a fixed grid is required because the target format demands equal `n_timepoints` across trials and sessions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample-epoch/instruction-tone onsets) and the trial's go-cue time. Because an early lick replays the sample epoch, there can be several sample starts per trial; the AI takes the **last sample start strictly before the trial's go cue** and stores it as a signed offset relative to the go cue.

ii.
```python
        sample_start_ts = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
        tone_onset_rel_go = np.full(n_trials, np.nan)
        for i in range(n_trials):
            before = sample_start_ts[sample_start_ts < go_times[i]]
            if len(before) > 0:
                tone_onset_rel_go[i] = before[-1] - go_times[i]
```

iii. Step 38: the AI counted 405 sample starts against 368 trials and 368 go cues and concluded "the extra 37 likely come from replayed epochs triggered by early licking", so it could not index sample starts by trial number. Step 40: it verified that the tone onset sits a consistent ~1.85 s before the go cue (0.65 s sample epoch + 1.2 s delay), which matches the task description in `methods.txt`, and chose "the last sample start before the go cue as the true tone onset (excluding replays)".

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial the input is a continuous, time-varying row: `bin_center - tone_onset_rel_go`, i.e. seconds elapsed since the (last) tone onset at each of the 80 bin centres. It is negative before the tone and increases by 0.05 per bin. If no sample start precedes the go cue (first trial edge case) a hard-coded fallback of `-1.85 s` is substituted for the tone offset. The row is stored as float64 in row 0 of the `(2, 80)` input array; the measured value range is about -0.6 to +5.7 s (long values coming from replayed sample epochs).

ii.
```python
            if np.isnan(tone_onset_rel_go[trial_idx]):
                # Fallback: use approximate value (-1.85s before go cue)
                tone_rel = -1.85
            else:
                tone_rel = tone_onset_rel_go[trial_idx]
            time_from_tone = bin_centers - tone_rel  # positive = after tone
            ...
            input_data = np.stack([time_from_tone, photostim_binary], axis=0)  # (2, n_bins)
```

iii. Step 40: *"For the decoder input 'time from tone onset': at each time bin, the input value = current_time - (-1.85) = current_time + 1.85. Actually, I should compute it per trial by finding the sample_start that corresponds to that trial."* The -1.85 s constant is used only as a fallback because the AI had measured that to be the typical sample-to-go interval. The instructions ask for a continuous time-varying input, so the quantity is emitted per bin rather than as a scalar.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same grid as the firing rates: `bin_centers` comes from `get_bin_centers(T_START, T_END, BIN_WIDTH)`, the centres of the same 80 edges used by `bin_spike_times`, and the tone offset is expressed relative to the same go cue. Bin *k* of the input therefore covers the same interval as bin *k* of the neural matrix by construction; no interpolation or resampling is involved.

ii.
```python
        bin_centers = get_bin_centers(T_START, T_END, BIN_WIDTH)
        n_bins = len(bin_centers)
        ...
            tone_rel = tone_onset_rel_go[trial_idx]   # = tone_time - go_time
            time_from_tone = bin_centers - tone_rel
```

iii. Not separately justified in the trajectory — the AI treated the go-cue-relative grid as the single common time axis for all streams once it had verified (step 45) that events, spikes and video share one session clock.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the trials-table columns `photostim_onset` and `photostim_duration` (both stored as **strings**, with `'N/A'` on unstimulated trials, and measured from trial start), plus `intervals/trials/start_time` and the trial's go-cue time, which are needed to re-express the onset relative to the go cue.

ii.
```python
        photostim_onset_str = np.array(
            [x.decode() for x in f['intervals/trials/photostim_onset'][:]])
        photostim_dur_str = np.array(
            [x.decode() for x in f['intervals/trials/photostim_duration'][:]])
        ...
        for i in range(n_trials):
            if photostim_onset_str[i] != 'N/A':
                onset_from_trial_start = float(photostim_onset_str[i])
                duration = float(photostim_dur_str[i])
```

iii. Step 40/42: the AI enumerated the photostim-related columns (`photostim_onset`, `photostim_duration`, `photostim_power`, `photo_stim_type`), found 78 stimulated trials in the example session, and checked their timing against `methods.txt` ("late delay epoch, last 0.5 s, including the 100 ms ramp-down"). It found onsets at ≈ -1.2 s re go cue with 0.5 s duration, briefly worried this conflicted with "late delay", and concluded: *"I'll stop trying to reconcile the terminology and just use the actual timestamps from the data going forward."*

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time-varying row: a bin is 1 if its centre lies in `[onset, onset + duration)` expressed relative to the go cue, else 0. Unstimulated trials have NaN on/off times and get an all-zero row through an explicit `np.isnan` branch. Stored as float64 in row 1 of the input array; measured occupancy ≈ 3% of all bins in a test session.

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
            photostim_binary = np.zeros(n_bins)
            if not np.isnan(photostim_on_rel[trial_idx]):
                on_t = photostim_on_rel[trial_idx]
                off_t = photostim_off_rel[trial_idx]
                photostim_binary = ((bin_centers >= on_t) & (bin_centers < off_t)).astype(float)
```

iii. The instructions ask for "whether photostimulation is on at every time point (discrete, time-varying)", so the AI built a per-bin binary series rather than a per-trial flag (step 42: *"For the decoder feature representing whether photostim is on, I need to construct a binary time series per trial based on that onset window"*). Keeping photostim trials in the dataset instead of excluding them (as the reference analysis code does) was justified by photostimulation being a required decoder *input*.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stored onset is relative to `trial start`, so it is converted to go-cue-relative time by subtracting `go_times[i] - trial_start_times[i]`, and then compared against the same `bin_centers` used for the neural binning. No separate time base or interpolation is used.

ii.
```python
                go_from_trial_start = go_times[i] - trial_start_times[i]
                photostim_on_rel[i] = onset_from_trial_start - go_from_trial_start
                photostim_off_rel[i] = onset_from_trial_start + duration - go_from_trial_start
```

iii. Step 42: *"To align this with go-cue-relative time, I'm computing the photostim on/off times relative to the go cue using trial start, onset, and duration offsets."* The AI sanity-checked the result against the expected late-delay window (≈ -1.2 to -0.7 s re go cue).

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no choice column in the file; the AI derives it from two trials-table columns, `outcome` (`'hit'`/`'miss'`/`'ignore'`) and `trial_instruction` (`'left'`/`'right'`). A hit means the animal licked the instructed side, a miss the opposite side, and `ignore` means no lick. It considered but did not use the left/right lick event streams.

ii.
```python
        outcome = np.array([x.decode() for x in f['intervals/trials/outcome'][:]])
        trial_instruction = np.array([x.decode() for x in f['intervals/trials/trial_instruction'][:]])
        ...
            out = outcome[trial_idx]
            instr = trial_instruction[trial_idx]
```

iii. Step 46: *"For determining choice, I can infer it from trial_instruction and outcome: hits mean the mouse chose the instructed side, misses mean it chose the opposite side, and ignore trials mean no lick occurred at all. Though the left/right lick timestamps could give the actual choice directly, this instruction-outcome mapping is a simpler way to derive it."* It also confirmed the NWB semantics: hit = correct lick, miss = incorrect lick, ignore = no response.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as `0 = left`, `1 = right`, `2 = no lick`, then broadcast unchanged across all 80 bins in row 0 of a `(4, 80)` float64 output array. `output_values[0] = ['left', 'right', 'no_lick']`.

ii.
```python
            if out == 'ignore':
                choice = 2  # no lick
            elif out == 'hit':
                choice = 0 if instr == 'left' else 1
            else:  # miss
                choice = 1 if instr == 'left' else 0
            ...
            output_data = np.zeros((4, n_bins))
            output_data[0, :] = choice
```
```python
        'output_values': [
            ['left', 'right', 'no_lick'],
            ...
```

iii. The instructions list "Lick direction choice (left, right, no lick, per-trial)" as a categorical output, so the AI used exactly those three classes and replicated the per-trial scalar across bins to satisfy the `(n_output, n_timepoints)` shape requirement (the format doc says "If at all possible, make it time-varying", and all four outputs are packed into one time-varying array).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `intervals/trials/outcome` column, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'` (verified: no other values occur anywhere in the dataset).

ii.
```python
        outcome = np.array([x.decode() for x in f['intervals/trials/outcome'][:]])
```

iii. No derivation was needed — the AI noted in step 46 that the NWB outcome categories map one-to-one onto the reference code's `correctness` encoding (1 = correct, 0 = error, -1 = no response) and onto the three classes the instructions ask for.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps `ignore → 0`, `miss → 1`, `hit → 2` (the order given in the instructions), and the value is broadcast across all 80 bins in row 1 of the output array. `output_values[1] = ['ignore', 'miss', 'hit']`.

ii.
```python
            outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[out]
            ...
            output_data[1, :] = outcome_val
```

iii. Code order follows the instruction text "Outcome (ignore, miss, hit, per-trial)"; as with choice, the per-trial scalar is repeated across bins for a uniform output array.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `intervals/trials/early_lick` column, whose only values in this dataset are `'no early'` and `'early'`.

ii.
```python
        early_lick = np.array([x.decode() for x in f['intervals/trials/early_lick'][:]])
```

iii. The column flags early licking explicitly, so no derivation from lick times was needed. The AI's main reasoning about this variable was about *keeping* early-lick trials (step 42): because `early_lick = yes` is a required decoder output class, excluding those trials as the paper does would make the output degenerate.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary encoding `no early → 0`, anything else → 1, broadcast across the 80 bins in row 2 of the output array. `output_values[2] = ['no', 'yes']`. Note the mapping is written as a catch-all `else` rather than an explicit dictionary, so any unexpected label would silently be coded as "yes" (harmless here — only two labels exist).

ii.
```python
            early_val = 0 if early_lick[trial_idx] == 'no early' else 1
            ...
            output_data[2, :] = early_val
```

iii. Follows the instruction "Early lick (no, yes, per-trial)"; the per-trial flag is repeated across bins like the other categorical per-trial outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: the `(n_frames, 3)` DeepLabCut array whose columns the AI determined to be `x`, `y`, `likelihood`, plus the matching `timestamps` (≈294-300 Hz, same session clock). Column 1 supplies the y-position and column 2 the visibility likelihood. Only the side camera exists in these files (no bottom view), which the AI verified.

ii.
```python
        tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
        tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
        tongue_y = tongue_data[:, 1]
        tongue_likelihood = tongue_data[:, 2]
```

iii. Step 23: the AI printed all three `BehavioralTimeSeries` groups (`Camera0_side_JawTracking`, `NoseTracking`, `TongueTracking`), noted `data shape=(680500, 3)` with the third column bounded in [0,1], and concluded "Only side view camera data. The tongue y-position from side camera tracking." Step 44: it checked the likelihood distribution (10.6% of frames > 0.5, 10.5% > 0.9) and the y-statistics when visible (mean 281 ± 18) vs all frames (mean 266 ± 41), confirming the column assignment.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) A frame counts as "tongue visible" iff `likelihood > 0.9`. (2) The 40th and 60th percentiles are computed **per session over the raw y-values of all visible frames** (not over binned means, and not restricted to trial windows). (3) Each 50 ms bin is represented by the **single video frame nearest to the bin centre** (see 8-d); if that frame is not visible the bin gets class 3. If a session has no visible frames at all, both percentiles fall back to 0.0 so everything becomes classes 2/3.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
...
        visible_mask = tongue_likelihood > TONGUE_LIKELIHOOD_THRESH
        if visible_mask.sum() > 0:
            y_visible = tongue_y[visible_mask]
            pct40 = np.percentile(y_visible, 40)
            pct60 = np.percentile(y_visible, 60)
        else:
            pct40 = pct60 = 0.0  # fallback, all will be "not visible"
```

iii. Step 44/45: the AI measured the likelihood distribution and found it effectively bimodal (median 1e-4, 90th percentile 1.0), so ~10.5% of frames are visible regardless of whether the cut is at 0.1, 0.5 or 0.9, and chose 0.9 as a conservative "tongue really is out" threshold. It reasoned that percentiles must be taken only over visible frames, since the tracker still reports a position when the tongue is retracted and including those would corrupt the thresholds. Per-session scope and the 40/60 split follow the instructions literally ("percentile of y-position over the session").

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes exactly as the instructions specify, evaluated per bin: `y < p40 → 0`, `p40 <= y <= p60 → 1`, `y > p60 → 2`, frame not visible → `3`. The array is initialised to 3 so any bin whose nearest frame fails the likelihood test stays "not visible". Empirically (session `sub-440957_ses-20190214T144611`) the class frequencies are 7.0% / 3.5% / 8.0% / 81.4%.

ii.
```python
            tongue_y_disc = np.full(n_bins, 3)  # default: not visible
            for b, tc in enumerate(bin_centers):
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
```python
            ['below_40pct', '40_to_60pct', 'above_60pct', 'not_visible'],
```

iii. Directly transcribed from the "Tongue y-position, per-session discretization" specification; the fourth class exists because the instructions list "3: not visible" and because the tongue is out only ~10% of the time (step 44/45 reasoning: the DLC confidence "shows values near zero when the tongue isn't actually visible in frame").

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Nearest-neighbour sampling, not averaging. For every bin the absolute time of the bin centre (`go_t + bin_center`) is computed, `np.searchsorted` finds the insertion point in the camera timestamps, and if that frame is more than 10 ms away the preceding frame is tried and the closer of the two is kept. The value of that one frame is used for the whole 50 ms bin. Since the camera runs at ~294 Hz, roughly 15 frames fall in each bin and ~14 of them are discarded. There is no rejection of frames that are far from the bin centre (the ±10 ms test only chooses between two candidates, it never falls back to "not visible"); empirically this matters little — in sampled sessions only ~0.2% of bins have no frame within 50 ms, and none of those were assigned a visible class.

ii.
```python
            for b, tc in enumerate(bin_centers):
                abs_t = go_t + tc
                # Find closest video frame
                frame_idx = np.searchsorted(tongue_ts, abs_t)
                frame_idx = min(frame_idx, len(tongue_ts) - 1)
                if abs(tongue_ts[frame_idx] - abs_t) > 0.01:
                    # Also check previous frame
                    if frame_idx > 0 and abs(tongue_ts[frame_idx - 1] - abs_t) < abs(tongue_ts[frame_idx] - abs_t):
                        frame_idx = frame_idx - 1
                if tongue_likelihood[frame_idx] > TONGUE_LIKELIHOOD_THRESH:
                    ...
```

iii. Step 42: *"Since the NWB video timestamps share the same timeframe as neural and trial event data, I can directly map video frames to absolute times. For each 50 ms bin I'll compute the bin center time relative to the go cue, find the nearest video frame, check tongue visibility against a likelihood threshold, and extract the y-position or flag it as not visible."* The AI verified the shared clock in step 45 (video 0-2493 s, go cues 3.06-2490 s, frame 901 at the first go cue). It did not discuss averaging within bins, nor that the video is gated to trials.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases are handled, one is not:
- **Session with no classifier QC** (`units/classification` all NaN, 1 of 174): detected via the column dtype and the whole session is dropped. The same session is also the only one with no `anno_name`.
- **Sessions/units with unmappable CCF annotation**: `map_anno_to_region` falls back to a keyword scan for cortical areas and then to the literal string `'Unknown'`.
- **Trial with no preceding tone onset**: `tone_onset_rel_go` stays NaN and a hard-coded `-1.85 s` is substituted.
- **Session with no visible tongue frames**: percentiles fall back to `0.0` (all bins end up class 2 or 3).
- **Not handled: trials with no spike data.** Trials outside `units/obs_intervals` are kept and emitted as all-zero firing rates (1,056 such trials in 7 kept sessions). Free-water trials, which are also spike-less, are removed — but as a behavioural exclusion rather than as a missing-data exclusion.

ii.
```python
        if clf_raw.dtype == object:
            ...
        else:
            # No classifier QC available (all NaN) – skip session
            return None
```
```python
def map_anno_to_region(anno_name):
    if not anno_name:
        return 'Unknown'
    ...
    return 'Unknown'
```
```python
            if np.isnan(tone_onset_rel_go[trial_idx]):
                # Fallback: use approximate value (-1.85s before go cue)
                tone_rel = -1.85
```
```python
        else:
            pct40 = pct60 = 0.0  # fallback, all will be "not visible"
```

iii. The QC-NaN case was found by an actual crash (`'numpy.float64' object has no attribute 'decode'`, step 53) and then investigated: the AI checked how many sessions were affected (1/174, step 58) and decided *"I'll skip it (no classifier QC and no CCF annotations)"*. The other fallbacks were written defensively without specific evidence that they trigger. For missing spike data the AI made an explicit (and mistaken) decision: *"letting `obs_intervals` and trial windows naturally handle cases with zero spikes rather than explicitly masking with `is_good_trials`"* — i.e. it treated absence of recording as a genuine zero firing rate.

## 10-a. What are the most time-consuming steps of the code?

i. By a wide margin, the neural binning: the nested `for trial → for unit` loop calls `np.histogram` on the unit's **entire session-long** spike train after allocating a full shifted copy (`spks - go_t`). Cost is O(n_trials × n_units × n_spikes_per_unit). Measured on one small session (169 units, 660 trials, median 12,680 spikes/unit): 14.3 s total, of which **13.2 s is the neural loop and 0.1 s the tongue loop**. Loading `units/spike_times` (up to ~11.5 M doubles) and the ~680k×3 tracking array is the second cost. Extrapolating from the agent's own run log, the full 174-session pass would take roughly 25-30 minutes plus the time to pickle an estimated ~18-20 GB of float64 output (the expert's script takes 247 s + ~40 s to pickle 11.9 GB).

This mattered: the script was launched in the background at step 62, had reached ~160/174 sessions after ~20 minutes of blocking waits, and the AI then **killed it** (step 69, `kill $(ps aux | grep 'convert_data.py' ...)`) intending to optimise the tongue loop. The session ended there, so **`/app/converted_data.pkl` was never produced** — the required output file does not exist.

ii.
```python
        for trial_idx in trial_indices:
            go_t = go_times[trial_idx]
            fr_matrix = np.zeros((n_good, n_bins))
            for u, spks in enumerate(unit_spikes):
                aligned = spks - go_t                     # full copy of the spike train
                fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
```
```python
def bin_spike_times(spike_times, t_start, t_end, bin_width):
    ...
    counts, _ = np.histogram(spike_times, bins=edges)     # scans all spikes in the session
    return counts / bin_width
```

iii. The AI anticipated the problem before writing the code (step 49: *"with 174 sessions each containing hundreds of trials and up to thousands of units, I'll need to bin spike times efficiently... so I should lean on vectorized numpy operations rather than loops"*) but did not act on it. When the run turned out to be slow it misattributed the cost: step 67/68, *"The tongue tracking processing is slow because it iterates over each bin for each trial... Actually the tongue y-position binning is really slow because it does a nested loop over bins and trials. Let me optimize this with vectorized operations."* Measurement shows the tongue loop is ~1% of the runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops, all avoidable:
1. **Neural binning (`for trial_idx` × `for u`)** — the dominant cost. The expert's approach removes the trial loop entirely by flattening all trials' bin edges into one array and calling `np.searchsorted(spikes, edges)` once per unit, then differencing; only the per-unit loop remains (unavoidable because `spike_times` is ragged). That is ~1-2 orders of magnitude faster.
2. **Tongue discretisation (`for b, tc in enumerate(bin_centers)` inside the trial loop)** — one `searchsorted` per bin; a single vectorised `searchsorted(tongue_ts, go_t + bin_centers)` (or a `bincount`-based bin mean over all frames at once) replaces 80 calls.
3. **Tone onset (`for i in range(n_trials)`)** — `sample_start_ts[sample_start_ts < go_times[i]]` performs a full O(n) boolean scan and allocation per trial; `np.searchsorted(sample, go) - 1` does all trials at once.
4. **Photostim timing (`for i in range(n_trials)`)** — string parsing and arithmetic per trial; maskable with `on_s != 'N/A'` plus vectorised `astype(float)`.

Also `brain_regions.index(r)` in the assembly does a linear list scan per unit (a dict lookup in the expert's code), though that is negligible in absolute terms.

ii.
```python
        for i in range(n_trials):
            before = sample_start_ts[sample_start_ts < go_times[i]]     # O(n_samples) per trial
            if len(before) > 0:
                tone_onset_rel_go[i] = before[-1] - go_times[i]
```
```python
        for trial_idx in trial_indices:
            ...
            for u, spks in enumerate(unit_spikes):
                aligned = spks - go_t
                fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
            ...
            for b, tc in enumerate(bin_centers):
                frame_idx = np.searchsorted(tongue_ts, abs_t)
```
```python
        idx = np.array([brain_regions.index(r) for r in regions_list])
```

iii. The AI planned to vectorise (step 49) but wrote scalar loops, and only at the very end recognised a loop problem — and picked the wrong loop (the tongue one). No justification is given anywhere for the per-trial/per-unit histogram structure; it appears to be a straightforward-but-naive first implementation.

## 10-c. What processing does the code repeat multiple times?

i. The major repetition is inside the neural binning: **each good unit's complete spike train is re-scanned once per trial**. For a 660-trial session with a median 12,680 spikes per unit, that is ~8.4 M element operations per unit (plus 660 full-array allocations from `spks - go_t`) where ~81 binary searches would do — roughly a 500× redundancy factor. Smaller repetitions: `get_bin_centers` (and `np.linspace` inside `bin_spike_times`) is recomputed for every call rather than once at module level; `tone_onset_rel_go` and the photostim on/off arrays are computed for **all** `n_trials`, including trials that the `trial_mask` then discards; `brain_regions.index(r)` re-scans the region list for every unit. The AI does avoid one obvious repetition — spike-time slices are pre-extracted per unit before the trial loop rather than re-sliced per trial — and each NWB file is opened exactly once.

ii.
```python
        unit_spikes = []
        for uid in good_indices:                       # good: sliced once per session
            ...
            unit_spikes.append(spike_times_flat[start_idx:end_idx])
        ...
        for trial_idx in trial_indices:
            for u, spks in enumerate(unit_spikes):
                aligned = spks - go_t                  # but re-copied and re-scanned per trial
                fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)
```
```python
def bin_spike_times(spike_times, t_start, t_end, bin_width):
    n_bins = int(round((t_end - t_start) / bin_width))
    edges = np.linspace(t_start, t_end, n_bins + 1)    # rebuilt on every call
```

iii. Not discussed in the trajectory. The AI's only stated efficiency measure is the pre-extraction of `unit_spikes` ("Pre-extract spike times per good unit" comment in the code).

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest but real:
- **Memory format.** Firing rates are `np.zeros(...)` → **float64**, and outputs (four small integer codes) are also float64. The expert stores rates as float32 and outputs as int8. At 144 sessions × ~500 trials × ~390 units × 80 bins this is roughly 18-20 GB instead of ~9-10 GB, doubling both pickle size and the memory the decoder must load, for no precision benefit (rates are integer multiples of 20 Hz).
- **Brain-region mapping.** A 130-entry keyword table maps ~293 distinct CCF annotations onto 14 broad categories. The format requires `brain_regions`/`brain_region_idx`, but the decoder training script never uses them (only `verify_data_format` touches them), so the entire ontology effort is effectively discarded downstream. Units are also mapped only for the good subset, which is correct, but `n_units_total` is computed and never used.
- **Discarded trials.** Tone onset and photostim on/off are computed for every row of the trials table, then indexed only at the kept rows.
- **Dead code.** `import sys` and `import json` are unused; `n_units_total` is unused.
- Per-trial scalars (choice/outcome/early lick) are replicated across 80 bins, but that is required by the target format, not waste.

ii.
```python
            fr_matrix = np.zeros((n_good, n_bins))      # float64
            ...
            output_data = np.zeros((4, n_bins))         # float64 for values in {0,1,2,3}
```
```python
        n_units_total = len(classification)             # never used
```
```python
        tone_onset_rel_go = np.full(n_trials, np.nan)   # computed for all trials,
        for i in range(n_trials):                       # used only for kept trials
```
```python
import sys
import json
```

iii. The region mapping is justified in the trajectory (steps 46-49) as matching "the 14 broad regions with left/right splits" used by the reference code, so the AI regarded it as required fidelity to the reference rather than waste. No justification is given for float64 storage; the AI never discussed memory footprint or output file size, and never got to the point of writing the pickle.
