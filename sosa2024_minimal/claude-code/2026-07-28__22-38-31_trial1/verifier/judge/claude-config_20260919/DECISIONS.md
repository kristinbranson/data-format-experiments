# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every subdirectory of `data/` whose name starts with `sub-` is treated as a subject, and every `*.nwb` file inside it (sorted) is treated as one session. Files are opened with `pynwb.NWBHDF5IO` and every needed array is read eagerly into memory in `load_nwb()`: all 11 behavioral time series (`position`, `speed`, `lick`, `reward_zone`, `teleport`, `trial_start`, `trial number`, `environment`, `scanning`, `autoreward`, `Reward`), the behavior `timestamps`, the ophys `Fluorescence`, `Neuropil` and `Deconvolved` ROI response series (concatenated across imaging planes), and the `ImageSegmentation/PlaneSegmentation` columns `iscell` and `planeIdx`. The result is 11 subjects / 152 sessions / 12,216 trials, identical to the reference.

ii.
```python
def load_nwb(filepath):
    from pynwb import NWBHDF5IO
    io = NWBHDF5IO(filepath, 'r')
    nwb = io.read()
    bts = nwb.processing['behavior']['BehavioralTimeSeries']
    ophys = nwb.processing['ophys']
    seg = ophys['ImageSegmentation']['PlaneSegmentation']
    plane_keys = sorted(ophys['Fluorescence'].roi_response_series.keys())
    n_planes = len(plane_keys)
    ...
    data = {
        'position': np.array(bts.time_series['position'].data[:]),
        ...
        'fluorescence': fluorescence, 'neuropil': neuropil_data, 'deconvolved': deconvolved,
        'iscell': np.array(seg['iscell'].data[:]),
        'plane_idx': np.array(seg['planeIdx'].data[:]),
        'session_id': nwb.session_id,
        'subject_id': nwb.subject.subject_id if nwb.subject else None,
        'n_planes': n_planes,
    }
```
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
for subj_dir in subjects:
    nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
    for nwb_file in nwb_files:
        result = process_session(nwb_file, compute_own_dff=True)
```

iii. From the trajectory: the agent enumerated the directory tree (`for d in /app/data/sub-*/; do ... ls $d/*.nwb ...`), found 11 subject folders with 12–14 NWB files each, and verified against the dandiset metadata and the paper that these are the 11 "switch" mice. It then inspected each NWB file with `pynwb` (`nwb.processing` → `behavior`, `ophys`) and enumerated every field before deciding what to read, so that "all relevant data from the source" is included.

## 1-b. How are the data split into subjects?

i. Subject identity comes from the `sub-<id>` directory name; the string after `sub-` is the subject name (`m3`, `m11`, ...). A session's `subject_idx` is the index of that name in the running `all_subjects` list. The subject id stored inside the NWB file (`nwb.subject.subject_id`) is read but only used for printing.

ii.
```python
subject_name = subj_dir.replace('sub-', '')
if subject_name not in all_subjects:
    all_subjects.append(subject_name)
subj_idx = all_subjects.index(subject_name)
...
subject_idx.append(subj_idx)
...
data['subjects'] = all_subjects
data['subject_idx'] = np.array(subject_idx)
```

iii. The agent's exploration established that each directory holds exactly one animal's sessions and that `nwb.subject.subject_id` (e.g. `m11`) agrees with the directory name, and that the count (11) matches the paper's 11 switch mice.

## 1-c. How are the data split into sessions?

i. One session = one NWB file. Files are processed in sorted filename order (`ses-01`, `ses-02`, ...), so session order within a subject is experiment-day order. No cross-session neuron alignment (the paper's multi-day ROI matching) is attempted; each session's cells are treated independently.

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
for nwb_file in nwb_files:
    result = process_session(nwb_file, compute_own_dff=True)
    if result is None:
        continue
    all_neural.append(result['neural'])
    ...
```

iii. The file naming `sub-m11_ses-03_behavior+ophys.nwb` plus `nwb.session_id == '03'` told the agent each file is one recording day; it cross-checked that m11 starts at `ses-03` and that session counts (12–14/mouse) match the paper.

## 1-d. How are the data split into trials?

i. Trial boundaries come from the behavioral `trial_start` and `teleport` signals: each sample where `trial_start > 0` opens a trial, and the first subsequent sample where `teleport > 0` closes it. The trial is the half-open sample window `[start, end)`. A trial is additionally required to contain at least one `scanning == 1` sample. This yields exactly 80 (or 41–100) trials per session and 12,216 trials overall — identical to the reference's `trial_start`/`teleport`-rising-edge segmentation.

ii.
```python
def get_trial_boundaries(trial_start_signal, teleport_signal, trial_numbers, scanning):
    start_indices = np.where(trial_start_signal > 0)[0]
    end_indices = np.where(teleport_signal > 0)[0]
    ...
    for s in start_indices:
        next_ends = end_indices[end_indices > s]
        if len(next_ends) > 0:
            e = next_ends[0]
            if np.any(scanning[s:e] == 1):
                trial_starts.append(s)
                trial_ends.append(e)
    return trial_starts, trial_ends
```
```python
for t in range(len(trial_starts)):
    s, e = trial_starts[t], trial_ends[t]
    n_timepoints = e - s
```

iii. The agent examined `trial number` (found it runs `-1 .. 80`, i.e. includes an inter-trial `-1` state), `trial_start` (0/1) and `teleport` (0/1), and chose the `trial_start → teleport` window so that each trial is one traversal of the track and the inter-trial teleport period is excluded. Note: `CONVERSION_NOTES.md` describes trial boundaries as "from one trial_start to the next (or end of recording)", which is **not** what the code does — the code correctly stops at the teleport.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level: a trial is dropped if it has fewer than 5 samples, or if the scanner was never on during it (`scanning == 1` never true). Session-level: a session is dropped if it has fewer than 5 curated cells, fewer than 3 detected trials, or fewer than 2 surviving trials. In practice none of these fired: all 152 sessions and all 12,216 trials were kept (shortest trial 96 samples), the same set the reference keeps (the reference's nominal 50-sample minimum is implemented as `idx.sum() < 50`, which also never fires).

ii.
```python
if np.any(scanning[s:e] == 1):   # in get_trial_boundaries
    ...
if n_total_cells < 5:
    print(f"    Skipping: only {n_total_cells} curated cells"); return None
if len(trial_starts) < 3:
    print(f"    Skipping: only {len(trial_starts)} trials"); return None
...
if n_timepoints < 5:
    continue
...
if valid_trial_count < 2:
    print(f"    Skipping: only {valid_trial_count} valid trials"); return None
```

iii. These are defensive guards rather than data-driven criteria: the agent's exploration found `scanning ∈ {-1, 1}` and used it as an "imaging was on" check, and the ≥2-trial rule comes straight from the instructions ("There needs to be at least two trials within each session"). The agent did not report finding any short/bad trials in the data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From the raw suite2p traces: `ophys/Fluorescence` (F) and `ophys/Neuropil` (Fneu), pooled across imaging planes. The agent's own dF/F + OASIS pipeline is applied to these (`compute_own_dff=True` is the default and the mode actually used). The NWB `Deconvolved` array is read and concatenated but is only used on the unused `compute_own_dff=False` branch, so it does not enter the saved data.

ii.
```python
fluorescence = np.concatenate([np.array(ophys['Fluorescence'][k].data[:]) for k in plane_keys], axis=1)
neuropil_data = np.concatenate([np.array(ophys['Neuropil'][k].data[:]) for k in plane_keys], axis=1)
deconvolved   = np.concatenate([np.array(ophys['Deconvolved'][k].data[:]) for k in plane_keys], axis=1)
...
F = nwb_data['fluorescence'].T      # (n_neurons, n_timepoints)
Fneu = nwb_data['neuropil'].T
deconv_nwb = nwb_data['deconvolved'].T  # Pre-computed deconvolved
...
if compute_own_dff:
    dff = compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, frame_rate=effective_frame_rate)
```

iii. The agent explicitly considered using the stored `Deconvolved` array (it even tested interneuron correlations on it) and concluded: "the NWB data comes from suite2p's pipeline, but the paper recomputes dF/F and deconvolution on their own ... I should probably just compute my own dF/F and events following the paper's approach." For multi-plane mice it pooled planes citing the paper: "ROIs were identified separately per plane, but planes were pooled for all analyses."

## 2-b. How is the `neural` data processed?

i. A re-implementation of the paper's `preprocessing.dff`, per session, on the pooled-plane traces:
1. mask F and Fneu to within-trial samples only (NaN elsewhere);
2. subtract `0.7 * Fneu`;
3. per trial, add back `0.7 * mean(Fneu_trial)` so the ratio is a true dF/F;
4. per trial, Gaussian-smooth along time with sigma = 15 samples, then a 300-sample running minimum followed by a 300-sample running maximum (maximin baseline, the Methods' ~20 s window);
5. `dF/F = (F - baseline)/|baseline|`;
6. per trial, smooth dF/F with a 2-sample Gaussian (NaN-aware);
7. per trial, deconvolve with suite2p's OASIS, `tau = 0.7`, `fs = 15.5078125` Hz (the per-plane rate).
Samples outside trials remain NaN and are never emitted. The per-mouse/per-day `keep_teleports` distinction from the paper's `teleport_metadata.py` (baseline window allowed to span the teleport on some days) is **not** implemented; the baseline window is always restricted to the lap, i.e. the repo default `keep_teleports = False`.

ii.
```python
def compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, frame_rate=FRAME_RATE):
    f_ = np.full((n_neurons, n_samples), np.nan); f_neu_ = np.full((n_neurons, n_samples), np.nan)
    for start, end in zip(trial_starts, trial_ends):
        f_[:, start:end] = F[:, start:end]; f_neu_[:, start:end] = Fneu[:, start:end]
    nanmask = ~np.isnan(f_[0, :])
    f_[:, nanmask] = f_[:, nanmask] - NEUROPIL_COEF * f_neu_[:, nanmask]
    window_size = 300  # ~20s at 15.5 Hz, matching reference code int(300)
    for start, end in zip(trial_starts, trial_ends):
        f_[:, start:end] = f_[:, start:end] + NEUROPIL_COEF * np.nanmean(
            f_neu_[:, start:end], axis=1, keepdims=True)
        f_smooth = nansmooth(f_[:, start:end], 15, axis=1)
        baseline = ndimage.minimum_filter1d(f_smooth, window_size, axis=-1)
        baseline = ndimage.maximum_filter1d(baseline, window_size, axis=-1)
        dff[:, start:end] = (f_[:, start:end] - baseline) / np.abs(baseline)
    for start, end in zip(trial_starts, trial_ends):
        dff[:, start:end] = nansmooth(dff[:, start:end], 2, axis=1)
    return dff
```
```python
def deconvolve_oasis(dff_trial, tau=TAU, frame_rate=FRAME_RATE):
    from suite2p.extraction.dcnv import oasis
    events = oasis(dff_trial, 2000, tau, frame_rate)
    return events
...
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    trial_dff_clean = np.nan_to_num(dff[cell_indices, start:end], nan=0.0)
    events_trial = deconvolve_oasis(trial_dff_clean, frame_rate=effective_frame_rate)
    events[:, start:end] = events_trial
```

iii. The agent read `src/reward_relative/preprocessing.py` and `TwoPUtils` and mirrored the steps, explicitly matching parameters (`neu_coef=0.7`, `maximin`, `int(300)` window, sigma-2 smoothing, `tau=0.7`, OASIS batch 2000). It iterated once after an early version gave 3.8% putative interneurons ("The issue is in my dF/F computation. Let me fix it to more closely match the reference code and use 300-sample window like the reference"), and for the two-plane mice it adjusted the deconvolution rate citing the paper's "sampling rate of ~15.5 Hz per plane". It also reasoned about `[0, 15]` vs scalar sigma and concluded a time-axis-only sigma-15 Gaussian is equivalent.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, the same two the paper uses: (1) only suite2p-curated ROIs, `iscell[:, 0] == 1`; (2) of those, putative interneurons — cells whose dF/F correlates with running speed at Pearson r > 0.5 — are dropped. The correlation is computed over all within-trial samples of the session. Over the whole dataset 402 cells were removed and 138,276 kept (reference: 138,298).

ii.
```python
iscell = nwb_data['iscell'][:, 0].astype(bool)
...
def filter_interneurons(dff, speed, iscell_mask):
    mask = np.copy(iscell_mask)
    valid = ~np.isnan(dff[0, :]) & ~np.isnan(speed)
    for idx in np.where(mask)[0]:
        corr = np.corrcoef(dff[idx, valid], speed[valid])[0, 1]
        if corr > INTERNEURON_SPEED_CORR_THR:   # 0.5
            mask[idx] = False
    return mask
...
cell_mask = filter_interneurons(dff, speed, iscell)
cell_indices = np.where(cell_mask)[0]
```

iii. The agent confirmed `iscell[:,0]` is the binary manual-curation column (values {0,1}) and used the paper's 0.5 speed-correlation criterion. When a single test session gave 3.7–3.8% interneurons versus the paper's quoted 0.42 ± 0.85%, it re-examined its dF/F, considered dropping the filter, and decided: "the instructions explicitly require matching the reference paper's curation process, including neuron filtering ... Let me keep the filtering in place." Across the full dataset the rate came out at 0.29%, in line with the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, which requires nothing beyond the trial segmentation: each trial's neural matrix is `events[:, trial_start:teleport]`, so sample 0 of every trial is the `trial_start` sample. No padding, no pre-trial window (`off_start = 0.0`, `off_end = None`).

ii.
```python
s, e = trial_starts[t], trial_ends[t]
neural_trial = events[:, s:e].copy()
neural_trial = np.nan_to_num(neural_trial, nan=0.0)
...
'temporal_alignment_event': 'start of trial (entry to linear track at 0 cm)',
'off_start': 0.0,
'off_end': None,
```

iii. The instructions say "Temporally align based on start of the trial"; since behavior and imaging share one sample grid in the NWB files, slicing from the `trial_start` sample is the alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: data are kept at the native imaging/behavior sample grid. The bin size is written as `1000/15.5078125 = 64.4836 ms`, from a hard-coded module constant `FRAME_RATE = 15.5078125` rather than from each file's stored `rate`. The stored `rate` is the scanner rate (15.5078125 Hz on the 124 single-plane sessions, 31.015625 Hz on the 28 two-plane sessions), so the per-plane rate is 15.5078125 Hz everywhere and the hard-coded constant happens to be right for all 152 sessions — the in-code comment `effective_frame_rate = FRAME_RATE  # already per-plane in NWB data` is, however, not a correct statement about the file contents. The emitted `time_bin_size` (64.48362720403023 ms) is bit-identical to the reference's.

ii.
```python
FRAME_RATE = 15.5078125  # Hz, from NWB files
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~64.5 ms
...
n_planes = nwb_data.get('n_planes', 1)
# For multi-plane: frame rate per plane is the effective rate
effective_frame_rate = FRAME_RATE  # already per-plane in NWB data
...
'time_bin_size': TIME_BIN_MS,
```

iii. The agent measured the behavior timestamp spacing directly ("Timestamp diffs: mean=0.064484, std=0.000000; Approx frame rate: 15.5078125") and saw it was constant, so it kept the native resolution and used that rate for OASIS; for the two-plane mice it reasoned from the paper's "~15.5 Hz per plane" that 15.5078125 Hz is the correct per-plane rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attached to the `position` behavioral time series (all behavioral series share the same timestamps; the reference uses `trial number`'s, which are identical).

ii.
```python
'timestamps': np.array(bts.time_series['position'].timestamps[:]),
...
timestamps = nwb_data['timestamps']
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The agent verified the timestamps are uniformly spaced at the imaging rate and shared across behavioral series, so any series' timestamps give the same answer.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the trial's first timestamp, giving seconds since trial start; stored as row 0 of the `(4, n_timepoints)` input matrix.

ii.
```python
input_trial = np.zeros((4, n_timepoints))
input_trial[0, :] = time_from_start
```

iii. Straightforward and matches the requested variable definition ("Time from start of trial in seconds"). The resulting range (0 to 216.5 s) is identical to the reference's.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Nothing extra is needed: neural and behavioral samples are the same grid, and both are sliced with the same `[s:e]` indices. Where the neural and behavioral arrays differ in length (the two-plane sessions are one frame longer), every array is truncated to the common minimum before trials are extracted.

ii.
```python
n_behav = len(speed); n_neural = F.shape[1]
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]; Fneu = Fneu[:, :min_len]; deconv_nwb = deconv_nwb[:, :min_len]
    speed = nwb_data['speed'][:min_len]
    nwb_data['position'] = nwb_data['position'][:min_len]
    ... nwb_data['timestamps'] = nwb_data['timestamps'][:min_len]
```

iii. The agent hit the mismatch on `sub-m17_ses-04` (neural 22791 frames vs behavior 22790), checked it was an off-by-one, and truncated to the shorter length — the same remedy the reference applies.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioral time series (values −1 between trials, 0 = ENV1 or 1 = ENV2 within trials).

ii.
```python
environment = nwb_data['environment']
env_type = float(environment[s])  # per trial
```

iii. The agent inspected the series and found only `{-1, 0}` or `{-1, 1}` per session, consistent with the paper's two environments plus an inter-trial value.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The value at the trial's first sample is taken as the trial's environment and broadcast across all timepoints of the trial (the signal is constant within a trial). A defensive fallback maps any negative value to 0 (ENV1). The fallback never fires in this dataset — `environment` is always 0 or 1 at every within-trial sample — and the resulting ENV1/ENV2 trial split (6226/5990) matches the reference.

ii.
```python
env_type = float(environment[s])  # per trial
if env_type < 0:
    env_type = 0.0  # default to ENV1 if unknown
...
input_trial[1, :] = env_type
```

iii. The instructions ask for a per-trial binary ENV1/ENV2 input; the agent sampled it once per trial since it is constant within a trial. The `< 0 → ENV1` fallback is an undocumented guard against the inter-trial −1 value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Not from the stored `trial number` series: it is the position of the trial in the list produced by `get_trial_boundaries`, i.e. a 0-based within-session index derived from the `trial_start`/`teleport` signals. (`trial_number` is loaded and passed into `get_trial_boundaries` but never used there.)

ii.
```python
trial_starts, trial_ends = get_trial_boundaries(
    nwb_data['trial_start'], nwb_data['teleport'],
    nwb_data['trial_number'], nwb_data['scanning'])
...
for t in range(len(trial_starts)):
    trial_num = float(t)
```

iii. The agent had observed that the stored `trial number` series runs −1…80 (an extra inter-trial state) and segmented trials from `trial_start`/`teleport` instead, so the loop index is the natural, self-consistent trial number. The emitted range (0–99) matches the reference exactly.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the integer index across the trial's timepoints as row 2 of the input matrix.

ii.
```python
input_trial[2, :] = trial_num
```

iii. The instruction asks for a continuous per-trial trial number; the sequential index within a session is that.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the per-trial reward outcome of the preceding trial, which is computed from the `Reward` time series' timestamps together with the `reward_zone` series: a trial counts as rewarded if a `Reward` timestamp falls inside the trial's time window **and** the `reward_zone` signal was active somewhere in the trial.

ii.
```python
def determine_trial_rewarded(position, rz_signal, reward_timestamps, timestamps, trial_start, trial_end):
    t_start = timestamps[trial_start]
    t_end = timestamps[min(trial_end, len(timestamps) - 1)]
    reward_in_trial = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
    entered_rz = np.any(rz_signal[trial_start:trial_end] > 0)
    return int(reward_in_trial and entered_rz)
...
is_rewarded = [determine_trial_rewarded(...) for i in range(len(trial_starts))]
```

iii. The agent found `Reward` is a sparse event series with its own timestamps (74 events, 0.004 mL each) rather than a per-frame signal, so it matched reward times to trial time windows. The extra "was in the reward zone" condition encodes its reading that a reward is earned by entering the hidden zone.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial *t* > 0, the input is `is_rewarded[t-1]`, broadcast across the trial's timepoints. For the first trial of a session, where no previous trial exists, the value is set to **1 (rewarded)**. (The reference sets the first trial to 0.) This affects 152 of 12,216 trials.

ii.
```python
if t == 0:
    prev_outcome = 1.0  # assume rewarded before session
else:
    prev_outcome = float(is_rewarded[t - 1])
input_trial[3, :] = prev_outcome
```

iii. The only justification is the inline comment "assume rewarded before session"; the trajectory contains no further discussion. Since ~84% of trials are rewarded, 1 is the majority-class filler, but it does assert an outcome that did not occur.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` time series plus a per-trial reward-zone label (A/B/C). The label is inferred from the `reward_zone` time series: for each trial the mean of the positions where `reward_zone > 0` is matched to the nearest of the three zone centres (A 80–130, B 200–250, C 320–370 cm, the `X/Y/Z` ranges from the paper's `behavior.py::reward_zone_dict` renamed A/B/C). Trials where `reward_zone` never activates (omissions / never-entered) inherit the label of the nearest *following* labelled trial, or, failing that, the nearest preceding one.

ii.
```python
REWARD_ZONES = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}

def determine_reward_zone_label(position, rz_signal, trial_start, trial_end):
    rz_positions = position[trial_start:trial_end][rz_signal[trial_start:trial_end] > 0]
    if len(rz_positions) == 0:
        return None
    mean_rz_pos = np.mean(rz_positions)
    for zone, (start, end) in REWARD_ZONES.items():
        dist = abs(mean_rz_pos - (start + end) / 2)
        if dist < best_dist:
            best_dist, best_zone = dist, zone
    return best_zone

def determine_reward_zone_for_all_trials(position, rz_signal, trial_starts, trial_ends):
    labels = [determine_reward_zone_label(position, rz_signal, trial_starts[i], trial_ends[i])
              for i in range(n_trials)]
    for i in range(n_trials):
        if labels[i] is None:
            for j in range(i + 1, n_trials):      # look forward
                if labels[j] is not None: labels[i] = labels[j]; break
            if labels[i] is None:
                for j in range(i - 1, -1, -1):    # then backward
                    if labels[j] is not None: labels[i] = labels[j]; break
    return labels
...
rz_label = rz_labels[t]
if rz_label is None:
    rz_label = 'A'  # fallback
rz_start, rz_end = REWARD_ZONES[rz_label]
```

iii. The agent probed the `reward_zone` series trial by trial ("Trial 2: rz positions = [201.4, 211.8]", "Trial 30: rz positions = [82.5, 133.2]", "Trial 1: no rz detected (omission?)"), read `behavior.py`'s `reward_zone_dict` and the `A→X, B→Y, C→Z` mapping, and concluded that positions where `reward_zone > 0` identify which of the three hidden zones is active on that trial — including mid-session switches ("trial 2 had reward zone B and trial 30+ had zone A, which makes sense since the switch happened on day 3 after 30 trials"). Neighbour-filling handles trials the animal ran through without triggering the zone.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Per timepoint, the signed distance from the animal's position to the nearest edge of that trial's reward zone: negative before the zone, exactly 0 inside it, positive past it. Computed with a Python loop over timepoints.

ii.
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    if position < rz_start:
        return position - rz_start   # negative
    elif position > rz_end:
        return position - rz_end     # positive
    else:
        return 0.0                   # inside zone
...
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end) for p in pos_trial])
```

iii. Matches the instruction's "Distance to any location in the reward zone" (hence 0 anywhere inside the 50 cm zone) and the paper's notion of position relative to reward.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Into the 7 categories given in the instructions, via an explicit if/elif chain: `< -50 → 0`, `[-50, -10) → 1`, `[-10, 0) → 2`, `== 0 → 3`, `(0, 10] → 4`, `(10, 50] → 5`, `> 50 → 6`. This is value-for-value identical to the reference's `np.digitize(..., [-inf, -50, -10, 0, 1e-6, 10, 50, inf]) - 1`; the resulting class fractions agree with the reference to ~0.001.

ii.
```python
def discretize_distance(distances):
    for i, d in enumerate(distances):
        if d < -50: result[i] = 0
        elif d < -10: result[i] = 1
        elif d < 0: result[i] = 2
        elif d == 0: result[i] = 3
        elif d <= 10: result[i] = 4
        elif d <= 50: result[i] = 5
        else: result[i] = 6
    return result
...
dist_binned = discretize_distance(dist_to_rz)
output_trial[0, :] = dist_binned
```

iii. The bin edges are taken verbatim from the Decoder Task specification, with the "0 cm" class reserved for being inside the zone.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `position[s:e]`, the same sample window used for the neural matrix, so alignment is automatic (after the common-length truncation described in 3-c).

ii.
```python
pos_trial = position[s:e]
...
output_trial = np.zeros((n_output, n_timepoints), dtype=int)
output_trial[0, :] = dist_binned
```

iii. Behaviour and imaging share one sample grid in the NWB files; the agent verified the behavioural timestamps are the imaging timestamps.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural time series (cm along the 450 cm virtual corridor).

ii.
```python
'position': np.array(bts.time_series['position'].data[:]),
...
pos_trial = position[s:e]
```

iii. The agent checked the range ("Position: min=-500.0, max=450.8"), recognising −500/−50 as the inter-trial teleport values that lie outside the extracted trials.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The trial's raw positions are clipped to `[0, 449.999]` and then floor-divided into 90 cm bins. Clipping only affects the handful of samples marginally outside the track (e.g. 451 cm, or slightly negative at trial onset), which end up in the end bins — exactly what the reference's open-ended first/last `np.digitize` edges do. The emitted class fractions are identical to the reference's to all printed digits.

ii.
```python
pos_clipped = np.clip(pos_trial, 0, TRACK_LENGTH - 0.001)
pos_binned = discretize_position(pos_clipped, n_bins=5)

def discretize_position(positions, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins  # 90 cm bins
    return np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
```

iii. The instructions specify 5 equal bins spanning the 450 cm track; the agent took the track length from the paper/Methods (450 cm) and clipped so out-of-range samples fall into the terminal bins rather than producing out-of-range classes.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins: `<90 → 0`, `90–180 → 1`, `180–270 → 2`, `270–360 → 3`, `≥360 → 4`, implemented as `floor(pos/90)` with clipping to [0, 4].

ii.
```python
bin_size = TRACK_LENGTH / n_bins  # 90 cm bins
result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
...
'output_values': [..., ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'], ...]
```

iii. Directly from the Decoder Task specification.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[s:e]` slice as the neural data; no further alignment.

ii.
```python
pos_trial = position[s:e]
output_trial[1, :] = pos_binned
```

iii. Shared sample grid, as in 7-d.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural time series (per-frame lick counts, integer values 0–7).

ii.
```python
'lick': np.array(bts.time_series['lick'].data[:]),
...
lick_cumul = nwb_data['lick']
lick_trial = lick_cumul[s:e].copy()
```

iii. The agent inspected the series ("Lick: min=0.0, max=6.0, unique count=7") and treated any non-zero value as a lick occurring in that frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Simple binarisation of the raw per-frame lick counts: `lick > 0 → 1`, else 0. This is exactly the reference's rule and gives an identical lick fraction (0.2304). Note the internal inconsistency in the AI's own documentation: `CONVERSION_NOTES.md` states "The NWB lick timeseries stores cumulative lick counts. Converted to binary per-frame by computing the diff and marking frames with positive diff as lick=1", and the variable is even named `lick_cumul`, but no diff is computed anywhere in the code. The signal is in fact not cumulative (it is non-monotonic per-frame counts), so the implemented thresholding is the correct treatment and the note is simply wrong.

ii.
```python
lick_trial = lick_cumul[s:e].copy()
lick_binary = (lick_trial > 0).astype(float)
...
output_trial[3, :] = lick_binary.astype(int)
```

iii. The instructions require a binary lick output; the agent thresholded the count series at > 0.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[s:e]` slice as the neural data; no further alignment.

ii.
```python
lick_trial = lick_cumul[s:e].copy()
output_trial[3, :] = lick_binary.astype(int)
```

iii. Shared sample grid, as in 7-d.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The same per-trial A/B/C label described in 7-a, i.e. from the `reward_zone` and `position` time series.

ii.
```python
rz_labels = determine_reward_zone_for_all_trials(position, rz_signal, trial_starts, trial_ends)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
output_trial[4, :] = rz_loc
```

iii. See 7-a. The near-perfect thirds in the resulting distribution (0.328 / 0.337 / 0.335, versus the reference's 0.329 / 0.337 / 0.334) support the labelling.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The label is mapped A→0, B→1, C→2 and broadcast across all timepoints of the trial (a per-trial constant emitted as a time series). Omission trials take the neighbour-filled label; a final `None` would fall back to 'A' (never reached in this dataset).

ii.
```python
rz_label = rz_labels[t]
if rz_label is None:
    rz_label = 'A'  # fallback
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
output_trial[4, :] = rz_loc
```

iii. The instruction specifies a per-trial categorical 0 = A, 1 = B, 2 = C; the format guidance prefers time-varying outputs, so the constant is broadcast over time.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` time series' timestamps (sparse delivery events with their own clock), in conjunction with the `reward_zone` series, as in 6-a.

ii.
```python
reward_timestamps = nwb_data['reward_timestamps']   # bts['Reward'].timestamps
reward_in_trial = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
entered_rz = np.any(rz_signal[trial_start:trial_end] > 0)
return int(reward_in_trial and entered_rz)
```

iii. The agent found `Reward` has 74 entries with separate timestamps and constant amounts (0.004 mL), i.e. an event list, so it tested event times against trial time windows.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per trial: 1 if at least one reward event falls in `[timestamps[start], timestamps[end]]` and the reward zone was entered, else 0; broadcast across the trial's timepoints. The extra reward-zone conjunction is not in the reference (which uses reward events alone), but makes no difference here — the emitted reward-outcome fractions (0.1572 / 0.8428) are identical to the reference's.

ii.
```python
is_rewarded.append(determine_trial_rewarded(position, rz_signal, reward_timestamps, timestamps,
                                            trial_starts[i], trial_ends[i]))
...
rew_outcome = is_rewarded[t]
output_trial[5, :] = rew_outcome
```

iii. The instruction asks for a per-trial binary reward outcome; the agent's reading is that a reward is "earned" only in the hidden zone, so it required both conditions. No tolerance check on the reward-timestamp-to-frame alignment is performed (the reference asserts the error is within half a time bin).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards, all silent-ish (printed warnings at most):
- **Neural/behaviour length mismatch** (the two-plane sessions are one imaging frame longer): every array is truncated to the common minimum. Note this happens *after* `get_trial_boundaries` has run on the untruncated behaviour arrays, which is safe only because the surplus samples lie past the last trial.
- **Missing reward-zone activation** on a trial: label inherited from the next (else previous) labelled trial; a final fallback of 'A'.
- **`environment < 0`** (the inter-trial value): coerced to ENV1.
- **NaNs in the neural data** (samples outside trials, or any failed cell): replaced with 0 before deconvolution and again before saving.
- **OASIS failure**: caught, and the rectified dF/F (`max(dff, 0)`) is used for that trial instead.
- **Degenerate sessions/trials**: skipped (see 1-e).
No assertions are used; nothing is checked against the raw file after truncation, and no count of how often each fallback fires is reported.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]; ... nwb_data['timestamps'] = nwb_data['timestamps'][:min_len]
...
if env_type < 0:
    env_type = 0.0  # default to ENV1 if unknown
...
neural_trial = np.nan_to_num(events[:, s:e].copy(), nan=0.0)
...
try:
    events_trial = deconvolve_oasis(trial_dff_clean, frame_rate=effective_frame_rate)
    events[:, start:end] = events_trial
except Exception as e:
    print(f"    Warning: deconvolution failed for trial: {e}")
    events[:, start:end] = np.maximum(trial_dff_clean, 0)
```

iii. The truncation was written in response to an actual failure on `sub-m17_ses-04` ("Off by one between neural and behavioral data. I need to handle this by truncating to the shorter length"). The other guards are precautionary; the agent verified afterwards with `train_decoder.py --verify-only` that the full dataset passes with "no errors or warnings".

## 13-a. What are the most time-consuming steps of the code?

i. In rough order: (1) reading each NWB file's ophys arrays — `Fluorescence`, `Neuropil` *and* `Deconvolved` are all pulled fully into memory (three (T × N) float arrays per plane, tens of thousands of frames × up to ~2,300 ROIs); (2) the dF/F computation, which runs the sigma-15 Gaussian and the two 300-sample rank filters over **all** ROIs, curated or not (only ~40% are cells); (3) the per-trial OASIS deconvolution (~80 calls per session × 152 sessions); (4) pickling the 19.5 GB output file, plus a second 961 MB sample pickle. There is no profiling in the code or the trajectory; the trajectory shows the full conversion running long enough to be backgrounded and polled.

ii. N/A (no timing instrumentation in the code).

iii. Not discussed explicitly by the agent; it simply backgrounded the conversion and polled it to completion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several, more than in the reference:
- `discretize_distance` and `discretize_speed` loop over individual timepoints in Python; both are one-line `np.digitize` calls (which is what the reference uses).
- `dist_to_rz = np.array([distance_to_reward_zone(p, ...) for p in pos_trial])` is a per-sample list comprehension; the reference does the same computation with two boolean masks.
- `filter_interneurons` calls `np.corrcoef` once per ROI; the whole speed-correlation vector can be computed with one centred matrix–vector product.
- `get_trial_boundaries` recomputes `end_indices[end_indices > s]` inside the loop over starts (O(n²) in trial count); `np.searchsorted` does it in one call.
- `compute_dff_per_trial` walks the trial list three separate times (mask, baseline, smooth) and `process_session` walks it a fourth time to deconvolve; these could be one pass.
- The `determine_*_for_all_trials` helpers loop per trial, and the neighbour-filling loop is itself O(n²) in the worst case.

ii. N/A (these are the loops as written; see snippets in 7-b, 7-c and 2-c).

iii. Not discussed by the agent. None of these dominate runtime relative to file I/O and deconvolution, which is presumably why they were left alone.

## 13-c. What processing does the code repeat multiple times?

i. The code makes a single pass over each NWB file (unlike the reference, which surveys every file and then re-reads it), but it repeats work within a session: the trial list is traversed four times in the neural pipeline (see 13-b); the full dF/F pipeline is run on all ROIs and then two-thirds of the result is thrown away when `cell_indices` is applied; the reward-zone label loop and the reward-outcome loop each re-slice the same per-trial windows that the main trial loop slices again; and every behavioural array is sliced twice (once in the label/outcome helpers, once in the main loop).

ii.
```python
dff = compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, ...)   # all ROIs
cell_mask = filter_interneurons(dff, speed, iscell)                   # curated only
cell_indices = np.where(cell_mask)[0]
...
rz_labels = determine_reward_zone_for_all_trials(position, rz_signal, trial_starts, trial_ends)
is_rewarded = [determine_trial_rewarded(...) for i in range(len(trial_starts))]
for t in range(len(trial_starts)):     # third traversal of the same windows
```

iii. Not discussed by the agent. The single-pass file reading is a genuine advantage over the reference's survey-then-convert structure; the per-session repetition is the cost of keeping the helpers independent.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Real waste exists here:
- `ophys['Deconvolved']` is read in full and concatenated across planes for **every** session, but is only consumed on the `compute_own_dff=False` branch, which is never taken — a whole (T × N) array read and copied per session for nothing.
- dF/F, baseline and smoothing are computed for non-curated ROIs (~60% of ROIs in a typical session) and then discarded; the reference explicitly subsets to `iscell` first "on a third of the memory".
- `autoreward`, `reward_data` and `trial_number` are loaded but never used (`trial_number` is even passed into `get_trial_boundaries` and ignored there); `plane_idx` is only used to fill a `plane_idx` key of the per-session dict that the final data structure never reads; `SPEED_THR` and the `interpolate` import are dead.
- A second 961 MB `sample_data.pkl` is always written alongside the 19.5 GB output, even in `--sample-only` mode (which, as the agent itself noticed, "doesn't actually limit processing").
- The `position` argument to `determine_trial_rewarded` is unused.

ii.
```python
deconv_planes = [np.array(ophys['Deconvolved'][k].data[:]) for k in plane_keys]
deconvolved = np.concatenate(deconv_planes, axis=1)      # never used in the default path
...
'autoreward': np.array(bts.time_series['autoreward'].data[:]),
'reward_data': np.array(bts.time_series['Reward'].data[:]),
...
dff = compute_dff_per_trial(F, Fneu, ...)   # F contains all ROIs, not just iscell
```

iii. The agent kept the `compute_own_dff=False` branch as an alternative after testing the stored `Deconvolved` signal for interneuron filtering, and never removed the now-unused load. The sample pickle was created to iterate quickly on the decoder before running on the full dataset.
