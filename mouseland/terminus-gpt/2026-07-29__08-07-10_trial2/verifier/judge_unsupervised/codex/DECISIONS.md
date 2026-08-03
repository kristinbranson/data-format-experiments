# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads every behavior file matching `data/beh/Beh_*.npy`, merges all contained session dicts into one in-memory `beh_by_session` map, and later loads neural data session-by-session from `data/spk/<base_session>_neural_data.npy`. It only keeps sessions whose base name exists in `data/spk`, and it drops any session whose `TrialStim` contains the placeholder string `stimulus_of_trial`.

ii.
```python
def load_all_behavior():
    beh_by_session = {}
    for bf in sorted(BEH_DIR.glob('Beh_*.npy')):
        obj = np.load(bf, allow_pickle=True).item()
        for k, v in obj.items():
            beh_by_session[k] = v
    return beh_by_session, beh_source

...

for sess in sorted(beh_by_session.keys()):
    base = base_session_name(sess)
    if base not in spk_sessions:
        continue
    ts = np.asarray(beh_by_session[sess].get('TrialStim', [])).astype(str)
    if ts.size and 'stimulus_of_trial' in set(ts.tolist()):
        continue
    selected.append(sess)
```

iii. `CONVERSION_NOTES.md` Step 5 says the agent decided to "use only sessions with both behavior and neural data" and to handle `_swap1/_swap2` behavior sessions explicitly. The trajectory does not give a separate justification for dropping sessions whose `TrialStim` contains `stimulus_of_trial`; that filter appears only in the final script.

## 1-b. How are the data split into subjects?

i. Subjects are inferred by splitting each selected session name at `'_'` and taking the prefix, for example `DR10` from `DR10_2022_07_12_1`. The script then builds `subjects` and `subject_idx` in the order subjects are first encountered while iterating selected sessions.

ii.
```python
subjects = []
subj_to_idx = {}
subject_idx = []

for i, sess in enumerate(selected):
    subj = sess.split('_')[0]
    if subj not in subj_to_idx:
        subj_to_idx[subj] = len(subjects)
        subjects.append(subj)
    ...
    subject_idx.append(subj_to_idx[subj])
```

iii. Step 5 notes say the subject mapping is based on the "subject prefix of session id" and that 19 subjects were observed in the data. The script follows that plan directly, but only for whatever sessions survived filtering.

## 1-c. How are the data split into sessions?

i. Each retained behavior-session key becomes one output session. `_swap1` and `_swap2` behavior sessions are preserved as separate output sessions, but both are mapped back to the same base neural recording by stripping the suffix before spike loading.

ii.
```python
def base_session_name(session):
    if session.endswith('_swap1') or session.endswith('_swap2'):
        return session.rsplit('_', 1)[0]
    return session

...

for sess in sorted(beh_by_session.keys()):
    base = base_session_name(sess)
    if base not in spk_sessions:
        continue
    ...
    selected.append(sess)
```

iii. Step 4 and Step 5 notes say the agent saw a naming mismatch between behavior sessions and neural files and decided to "handle `_swap1/_swap2` behavior sessions explicitly." The notes treat those swap sessions as behavior sessions that may reuse the same neural base session.

## 1-d. How are the data split into trials?

i. Trials are defined from the frame-wise `ft_trInd` array. The script trims all frame-wise streams and the spike matrix to a common frame count, keeps frames with finite `ft_trInd`, enumerates unique integer trial ids from those valid frames, and then builds one boolean frame mask per trial.

ii.
```python
frame_lengths.append(spk.shape[1])
nfr = min(frame_lengths)

spk = spk[:, :nfr]
ft_trInd = np.asarray(beh['ft_trInd'], dtype=float)[:nfr]

valid = np.isfinite(ft_trInd)
tr_idx = ft_trInd[valid].astype(int)
uniq_trials = np.unique(tr_idx)

for tr in uniq_trials:
    mask = valid.copy()
    mask[valid] = tr_idx == tr
```

iii. The trajectory shows the agent inspected `ft_trInd`, `StartFr`, and `EndFr`, noticed pre-trial `NaN` values in `ft_trInd`, and concluded that trial membership should be taken from `ft_trInd` after trimming behavior and neural arrays to a common length.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. A trial is dropped if its integer id is negative, out of bounds relative to `ntrials`, or if its frame mask has fewer than 2 frames. A whole session is dropped only if fewer than 2 valid trials remain.

ii.
```python
if tr < 0 or tr >= ntrials_declared:
    continue
...
if mask.sum() < 2:
    continue

...

if len(nt) < 2:
    print(f'skipping {sess}: fewer than 2 valid trials')
    continue
```

iii. `CONVERSION_NOTES.md` Step 3 says exact trial curation rules still needed confirmation from code and consistency checks. The final script never adds any richer trial-quality filtering beyond the basic frame-count and bounds checks above.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `spks` entry inside each `<session>_neural_data.npy` file. The script concatenates all arrays inside `obj['spks']` along the neuron axis.

ii.
```python
def load_spk_session(session_base):
    fn = SPK_DIR / f'{session_base}_neural_data.npy'
    obj = np.load(fn, allow_pickle=True).item()
    spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
    return spk
```

iii. Step 4 notes say the sample spike file had a `spks` list with three float32 arrays and that the reference `load_spk` function concatenated them. The notes explicitly interpret these as deconvolved calcium-event or fluorescence traces, not extracellular spikes.

## 2-b. How is the `neural` data processed?

i. The script trims the concatenated spike matrix to the common frame count shared with behavior arrays, splits it into per-trial frame blocks using `ft_trInd`, and then forces every trial to exactly 60 bins by index-based resampling over time. The saved neural arrays are cast to `float16`.

ii.
```python
nfr = min(frame_lengths)
spk = spk[:, :nfr]

...

nmat_raw = spk[:, mask].astype(np.float32)
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. Step 5 notes say the agent initially wanted to "respect reference position-based interpolation as an intermediate representation," but Step 78 of the trajectory shows the final implementation choice changed to frame-wise trial segmentation plus a compact 60-bin per-trial representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is effectively no neuron-level quality filtering in the conversion script. Neural preprocessing is assumed to have happened upstream; the script only trims neural frames to the common frame count and later pads or truncates brain-region labels if retinotopy and neural neuron counts differ.

ii.
```python
spk = spk[:, :nfr]

...

if len(idx) != n_neurons:
    m = min(len(idx), n_neurons)
    idx = idx[:m]
    if m < n_neurons:
        pad = np.full(n_neurons - m, lab_to_idx.get('unknown', 0), dtype=np.int64)
        idx = np.concatenate([idx, pad])
```

iii. Step 3 notes say Suite2p cell classification was part of preprocessing but that exact additional neuron inclusion or exclusion criteria still needed confirmation. The final script does not implement any explicit neuron-quality filtering beyond trusting the provided `spks`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script aligns neural data to each trial's first retained `ft_trInd` frame, not to an explicit `Trial_start_time` or `StartFr` event. Within each trial, it uses the frame mask for that trial and then rescales the trial to 60 bins.

ii.
```python
for tr in uniq_trials:
    mask = valid.copy()
    mask[valid] = tr_idx == tr
    ...
    nmat_raw = spk[:, mask].astype(np.float32)
    tvec = ft[mask]
    ...
    nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. The trajectory shows the agent reasoned that `ft_trInd` defines trial membership and that pre-trial `NaN` frames should be excluded. Although the notes acknowledge the task alignment event is "trial start / corridor entry," the code never uses `Trial_start_time` or `StartFr` to enforce that alignment explicitly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Every trial is rebinned to exactly 60 samples regardless of its native duration, using nearest-index selection in `resample_matrix_time`. The script does not preserve or report a real millisecond bin width and stores `metadata['time_bin_size'] = None`.

ii.
```python
def resample_matrix_time(mat, n_bins=60):
    ...
    idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
    return mat[:, idx].astype(np.float32, copy=False)

...

'metadata': {
    'time_bin_size': None,
    'temporal_alignment_event': 'trial start / corridor entry',
}
```

iii. The trajectory shows the agent moved from raw frame-wise trials to a "compact 60-bin representation" to reduce file size and runtime. The notes describe 60 bins as a convenient common representation, but they do not justify a physical time-bin size because the script ultimately normalizes each trial to 60 samples.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from trial-wise `SoundTime` and frame-wise `ft`, after converting both from MATLAB day units to seconds.

ii.
```python
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
...
soundtime = matlab_days_to_seconds(np.asarray(beh['SoundTime'], dtype=float))
...
cue_rel = soundtime[tr] - tvec
```

iii. Step 5 notes map "`SoundTime` or `SoundFr` relative to trial" to the decoder input `time_to_sound_cue`. The final code chooses `SoundTime` plus `ft` rather than using `SoundFr`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the script subtracts each retained frame time from that trial's `SoundTime`, producing a continuous countdown or signed time-to-cue trace. It then rebins that trace to 60 samples and stores it as the first input channel.

ii.
```python
cue_rel = soundtime[tr] - tvec
...
inp_raw = np.vstack([
    cue_rel.astype(np.float32),
    day_arr,
    time_since_start.astype(np.float32),
    reward_available,
])
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. Step 5 notes explicitly planned "`cue time minus current trial time`" as the transform for this decoder input. The final code implements that plan literally.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the same per-trial frame mask as neural data and then resampling to the same 60-bin trial representation.

ii.
```python
nmat_raw = spk[:, mask].astype(np.float32)
tvec = ft[mask]
...
cue_rel = soundtime[tr] - tvec
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. The notes say inputs and outputs "must align with neural activity" on a common per-trial axis. In the final code that common axis is the 60-sample trial representation obtained from the same frame mask as the neural trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is not derived from a field in the raw behavior data. Instead, it is derived from the session name string by parsing the year, month, and day components.

ii.
```python
def session_day_value(session_name):
    parts = session_name.split('_')
    y, m, d = map(int, parts[1:4])
    return y * 10000 + m * 100 + d
```

iii. Step 5 notes originally said this input should come from "session date / experiment ordering." Later, Step 90 of the trajectory says the agent realized this probably should be a within-subject ordinal day, but the final script still uses the calendar date encoded in the session name.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The parsed `YYYYMMDD` integer is converted to float, repeated across all frames in the trial, and then carried through the same 60-bin resampling as the other input channels.

ii.
```python
day_val = float(session_day_value(base))
...
day_arr = np.full(mask.sum(), float(day_val), dtype=np.float32)
...
inp_raw = np.vstack([
    cue_rel.astype(np.float32),
    day_arr,
    time_since_start.astype(np.float32),
    reward_available,
])
```

iii. The notes never justify using the raw calendar date rather than a training-day index. The only explicit later justification in the trajectory is that the agent recognized this was a likely mistake but did not change the final full-data script.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. In the final code it is derived from `ft` alone, not from `Trial_start_time` or `StartFr`, even though `Trial_start_time` is loaded.

ii.
```python
soundtime = matlab_days_to_seconds(np.asarray(beh['SoundTime'], dtype=float))
trialstart = matlab_days_to_seconds(np.asarray(beh['Trial_start_time'], dtype=float))
...
tvec = ft[mask]
time_since_start = tvec - tvec[0]
```

iii. Step 5 notes planned "trial-relative time" from trial timing variables. The trajectory shows the agent inspected `Trial_start_time`, `StartFr`, and `ft_trInd`, but the final code simplifies this to time since the first kept frame rather than using the stored trial-start variable.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The script subtracts the first retained frame time in that trial from every retained frame time in the same trial, producing a nonnegative trace that starts at 0 at the first kept frame. That trace is then rebinned to 60 samples.

ii.
```python
tvec = ft[mask]
...
time_since_start = tvec - tvec[0]
...
inp_raw = np.vstack([
    cue_rel.astype(np.float32),
    day_arr,
    time_since_start.astype(np.float32),
    reward_available,
])
```

iii. The script's own implementation is the only clear justification here. The notes and trajectory suggest the agent wanted trial-relative time, but they do not justify replacing true trial start with the first retained `ft_trInd` frame.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same per-trial frame mask as the neural data and the same 60-bin resampling, so it is aligned to the saved neural trial representation, though that representation itself is anchored to the first kept trial frame.

ii.
```python
nmat_raw = spk[:, mask].astype(np.float32)
tvec = ft[mask]
...
time_since_start = tvec - tvec[0]
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. Step 5 notes say all decoder variables should share a common trial axis. The final code achieves that mechanically by reusing the same frame mask and resampling function for neural and input tensors.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean array `isRew`.

ii.
```python
isrew = np.asarray(beh['isRew']).astype(bool)
...
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
```

iii. Step 5 notes explicitly say reward availability should come from rewarded corridor identity and should use "`isRew` / category logic" rather than actual reward delivery timing.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The script makes reward availability a constant per-trial binary value, repeating `1.0` across all time bins for rewarded trials and `0.0` otherwise.

ii.
```python
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
...
inp_raw = np.vstack([
    cue_rel.astype(np.float32),
    day_arr,
    time_since_start.astype(np.float32),
    reward_available,
])
```

iii. Step 5 notes justify this as "1 if in rewarded corridor, 0 if not" and say the mapping should be from corridor identity, not reward delivery, because unsupervised sessions can still contain the cue without reward.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial string array `TrialStim`. The script does not derive the output from `WallName`, `UniqWalls`, or `stim_id` in the final implementation.

ii.
```python
def build_global_mappings(beh_by_session, selected_sessions):
    ...
    if 'TrialStim' in beh:
        vals = [str(x) for x in np.unique(beh['TrialStim']) if str(x) != 'stimulus_of_trial']
        stim_names.update(vals)

...

trialstim = np.asarray(beh['TrialStim']).astype(str)
stim_idx = stim_to_idx[str(trialstim[tr])]
```

iii. Step 5 notes mention `TrialStim` and "canonical stimulus identity from `stim_id`" as candidate raw variables. The final script picks `TrialStim`, and the trajectory shows that choice also drove the later exclusion of sessions containing the placeholder token `stimulus_of_trial`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script builds one global sorted list of all unique `TrialStim` strings across selected sessions, maps each string to a global integer id, and then fills every time bin within a trial with that trial's stimulus id.

ii.
```python
stim_names = sorted(stim_names)
stim_to_idx = {s: i for i, s in enumerate(stim_names)}

...

stim_idx = stim_to_idx[str(trialstim[tr])]
stim_arr = np.full(mask.sum(), stim_idx, dtype=np.uint8)
out_raw = np.vstack([stim_arr, lick, pos_bin.astype(np.uint8), speed_bin.astype(np.uint8)])
```

iii. The notes say visual stimulus should remain a categorical per-trial output and list names such as `circle1`, `leaf2`, and swap variants. The final script implements exactly that kind of global categorical mapping, though it never falls back to `WallName` or `stim_id` when `TrialStim` is imperfect.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickTime` and `LickTrind`, together with the per-trial frame times `ft` used as the target alignment grid.

ii.
```python
lick = np.zeros(mask.sum(), dtype=np.uint8)
if len(beh['LickTrind']) > 0:
    lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
    if len(lick_times) > 0:
        inds = np.searchsorted(tvec, lick_times, side='left')
        ...
        lick[inds] = 1
```

iii. Step 5 notes map `LickTime` / `LickFr` / `LickTrind` to the decoder output `licking` and say it should be a binary time-varying series. The final code chooses the time-based path rather than the frame-based `LickFr` path.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the script starts from an all-zero binary vector, selects the lick times belonging to that trial, maps those lick times to the nearest retained trial frame with `np.searchsorted`, sets those frame positions to 1, and then rebins the binary vector to 60 samples.

ii.
```python
lick = np.zeros(mask.sum(), dtype=np.uint8)
if len(beh['LickTrind']) > 0:
    lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
    if len(lick_times) > 0:
        inds = np.searchsorted(tvec, lick_times, side='left')
        inds = inds[(inds >= 0) & (inds < len(tvec))]
        lick[inds] = 1
...
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. Step 5 notes only say licking should remain binary and time-varying. The trajectory later shows the agent debugging degeneracies in sample outputs and specifically suspected the lick-time mapping and time-base handling.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned by using the same per-trial frame mask as the neural data and by resampling the lick vector to the same 60-bin trial representation.

ii.
```python
nmat_raw = spk[:, mask].astype(np.float32)
tvec = ft[mask]
...
inds = np.searchsorted(tvec, lick_times, side='left')
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. The notes say outputs must align with neural activity on the common per-trial axis. The final code enforces that alignment mechanically, although the trajectory shows the agent later worried that the lick-time and frame-time units might not match.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-wise position array `ft_Pos` and the session-level scalar `Corridor_Length`.

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
...
corridor_length = float(beh.get('Corridor_Length', 40.0))
...
pos = ft_pos[mask]
```

iii. Step 5 notes map `ft_Pos` or interpolated position to the position output and say the output should be discretized into four equal 1 m bins as required by the decoder task.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The script takes the retained frame-wise position samples inside a trial and discretizes each sample into one of four equal-width bins spanning the corridor length.

ii.
```python
def discretize_position(pos, corridor_length):
    pos = np.asarray(pos, dtype=float)
    bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)).astype(int)
    return np.clip(bins, 0, 3)

...

pos_bin = discretize_position(pos, corridor_length)
```

iii. The notes justify this as the decoder-task-required discretization of corridor position into four equal bins. The final script implements that requirement directly.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholding rule is `floor(pos / (corridor_length / 4))`, after clipping positions to the interval `[0, corridor_length)`, followed by a final clip into integer categories `0..3`.

ii.
```python
bins = np.floor(
    np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)
).astype(int)
return np.clip(bins, 0, 3)
```

iii. There is no richer justification in the notes beyond the decoder requirement of four equal-length bins. The code implements the obvious thresholding rule corresponding to that requirement.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is aligned frame-by-frame within each trial using the same trial mask as neural data, then resampled to the same 60-bin representation as the neural trial.

ii.
```python
pos = ft_pos[mask]
...
pos_bin = discretize_position(pos, corridor_length)
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. Step 5 notes say outputs should share the neural trial axis. Because position is already frame-wise, the final script simply uses the same frame mask and then resamples to 60 bins.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the frame-wise running-speed array `ft_RunSpeed`.

ii.
```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
...
speed = ft_speed[mask]
```

iii. Step 5 notes map `ft_RunSpeed` directly to the running-speed output, with the extra decoder-task requirement that it be discretized into four bins containing 25% of the data each.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script first computes three global quartile edges from all finite `ft_RunSpeed` values across the selected sessions. It then assigns every retained frame in a trial to one of four bins using those global edges and resamples the labels to 60 bins.

ii.
```python
def compute_speed_edges(beh_by_session, selected_sessions):
    ...
    allv = np.concatenate(vals) if vals else np.array([0.0, 1.0])
    edges = np.quantile(allv, [0.25, 0.5, 0.75])
    return edges.astype(float)

...

speed_bin = discretize_speed(speed, speed_edges)
```

iii. Step 5 notes say running speed should be binned into quartiles over the dataset. The final code does exactly that by computing one set of global edges over all selected sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The categories are defined by three global thresholds: values at or below the 25th percentile are bin 0, above the 25th percentile are bin 1, above the median are bin 2, and above the 75th percentile are bin 3.

ii.
```python
def discretize_speed(speed, edges):
    speed = np.asarray(speed, dtype=float)
    out = np.zeros(speed.shape, dtype=int)
    out[speed > edges[0]] = 1
    out[speed > edges[1]] = 2
    out[speed > edges[2]] = 3
    return out
```

iii. The notes explicitly planned quartile-based running-speed bins. The code implements that plan with strict `>` comparisons at the three quantile edges.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. It is aligned on the same retained trial frames as the neural data and then resampled to the same 60-bin trial representation.

ii.
```python
speed = ft_speed[mask]
speed_bin = discretize_speed(speed, speed_edges)
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. Step 5 notes say outputs must align with neural activity. The final code achieves that by using the same per-trial frame mask and the same resampling scheme as the neural trial.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles minor inconsistencies with simple fallbacks: it trims neural and behavior arrays to the shortest common frame length, ignores frames where `ft_trInd` is `NaN`, skips malformed or too-short trials, uses a default corridor length of `40.0` if missing, and uses an `'unknown'` brain region or pads/truncates retinotopy labels if retinotopy metadata are missing or length-mismatched.

ii.
```python
nfr = min(frame_lengths)
spk = spk[:, :nfr]
...
valid = np.isfinite(ft_trInd)
...
if mask.sum() < 2:
    continue
...
corridor_length = float(beh.get('Corridor_Length', 40.0))
...
if ret is None or 'iarea' not in ret:
    return np.zeros(n_neurons, dtype=np.int64), ['unknown']
...
if len(idx) != n_neurons:
    ...
```

iii. The trajectory shows the agent noticed small frame-count mismatches between neural and behavior arrays and chose trimming to the common minimum length as the practical fix. The notes do not describe any more principled missing-data handling beyond these ad hoc safeguards.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive work is session-wise loading and concatenation of very large `spks` arrays, trial-by-trial extraction of neural frame blocks, and repeated 60-bin resampling of those per-trial neural matrices. The conversion log confirms these per-session extraction steps dominate runtime.

ii.
```python
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)

for tr in uniq_trials:
    ...
    nmat_raw = spk[:, mask].astype(np.float32)
    ...
    nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. The trajectory repeatedly discusses runtime and memory pressure and eventually introduces the compact 60-bin representation specifically to keep conversion feasible. `conversion_full_out.txt` also shows long per-session elapsed times during trial extraction.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest vectorization opportunities are the Python loop over trials in `extract_session`, repeated construction of boolean masks per trial, and the loop over unique region labels in `build_brain_region_idx`. The per-session selection loop also reloads shared neural recordings for separate swap sessions instead of caching them.

ii.
```python
for tr in uniq_trials:
    mask = valid.copy()
    mask[valid] = tr_idx == tr
    ...
    neural_trials.append(nmat)

...

for b in br_names:
    if b not in all_brain_regions:
        all_brain_regions.append(b)
    local_to_global[br_names.index(b)] = all_brain_regions.index(b)
```

iii. There is no explicit justification in the notes for these Python loops. The trajectory instead shows the agent making pragmatic changes for runtime, such as switching from interpolation loops to index-based resampling, while leaving the broader trial loop structure intact.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly converts arrays with `np.asarray`, repeatedly rescales trials to 60 bins for neural, input, and output tensors separately, and reloads identical base spike files for multiple `_swap1` / `_swap2` sessions. It also re-derives per-trial masks from `ft_trInd` inside every session rather than reusing grouped trial indices.

ii.
```python
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
ft_trInd = np.asarray(beh['ft_trInd'], dtype=float)[:nfr]
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]

...

nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. The trajectory shows the agent was aware of repetition around trial extraction and resampling because these paths kept surfacing in runtime debugging. The final script still repeats them because simplicity was favored over a more consolidated implementation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes or stores several things that are not used in the final saved tensors: `trialstart` is loaded but unused, `kept_trial_ids` is accumulated but never returned, `beh_source` is built but not used downstream, and `metadata['time_bin_size']` is left as `None` even though all trials were normalized to 60 bins. The script also repeats per-frame constants like day and reward across all 60 bins even though they are trial-level variables.

ii.
```python
beh_by_session, beh_source = load_all_behavior()

...

trialstart = matlab_days_to_seconds(np.asarray(beh['Trial_start_time'], dtype=float))
...
kept_trial_ids = []
...
kept_trial_ids.append(int(tr))

...

'metadata': {
    'time_bin_size': None,
}
```

iii. The trajectory shows the agent focused mainly on getting a valid decoder input format and reducing runtime. These unused or redundant computations were never cleaned up after the conversion logic stabilized.
