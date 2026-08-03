# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use `beh/Imaging_Exp_info.npy` as the master index. Instead it loads every `Beh_*.npy` file into one `beh_by_session` dictionary, then keeps only behavior sessions whose base name has a matching spike file. Retinotopy is loaded per selected session later. This means the conversion is driven by behavior-file keys plus spike-file existence, not by the paper's master recording index.

ii.
```python
def load_all_behavior():
    beh_by_session = {}
    for bf in sorted(BEH_DIR.glob('Beh_*.npy')):
        obj = np.load(bf, allow_pickle=True).item()
        for k, v in obj.items():
            beh_by_session[k] = v
    return beh_by_session, beh_source

spk_sessions = set(p.name.replace('_neural_data.npy', '') for p in SPK_DIR.glob('*_neural_data.npy'))

selected = []
for sess in sorted(beh_by_session.keys()):
    base = base_session_name(sess)
    if base not in spk_sessions:
        continue
    ...
    selected.append(sess)
```

iii. The justification in `CONVERSION_NOTES.md` Step 5 is that the decoder needs sessions with paired behavior and neural data, and that `_swap1/_swap2` behavior sessions may need to map to an unsuffixed neural session. The trajectory and notes also show a later pressure to reduce dataset size, which reinforced this session-selection approach rather than using the full reference index.

## 1-b. How are the data split into subjects?

i. Subjects are inferred by taking the prefix before the first underscore in each selected session name. Only subjects represented in the post-filtered `selected` list are kept, which in the agent's full run produced 14 subjects rather than the 19 subjects noted in its dataset-exploration notes.

ii.
```python
subj = sess.split('_')[0]
if subj not in subj_to_idx:
    subj_to_idx[subj] = len(subjects)
    subjects.append(subj)
...
subject_idx.append(subj_to_idx[subj])
```

iii. The justification in Step 5 of `CONVERSION_NOTES.md` is the session naming convention: "subject prefix of session id". No separate subject index source is used.

## 1-c. How are the data split into sessions?

i. Sessions are taken from behavior dictionary keys, sorted lexicographically. `_swap1` and `_swap2` keys are treated as distinct behavior sessions, but `base_session_name()` maps them back to the same unsuffixed spike file for neural loading. The AI does not deduplicate sessions using the master imaging index.

ii.
```python
def base_session_name(session):
    if session.endswith('_swap1') or session.endswith('_swap2'):
        return session.rsplit('_', 1)[0]
    return session

for sess in sorted(beh_by_session.keys()):
    base = base_session_name(sess)
    if base not in spk_sessions:
        continue
    selected.append(sess)
```

iii. The explicit justification in Step 4 and Step 5 of `CONVERSION_NOTES.md` is that many behavior-only swap sessions appear to reuse an unsuffixed neural recording, so they should be "handled explicitly" by mapping them to the corresponding base session.

## 1-d. How are the data split into trials?

i. Trials are recovered from framewise trial labels in `ft_trInd`. For each session, the AI finds the unique finite values of `ft_trInd`, treats each value as a trial id, and uses all frames carrying that id. It does not restrict the trial to corridor-texture frames via `ft_CorrSpc`.

ii.
```python
ft_trInd = np.asarray(beh['ft_trInd'], dtype=float)[:nfr]
valid = np.isfinite(ft_trInd)
tr_idx = ft_trInd[valid].astype(int)
uniq_trials = np.unique(tr_idx)

for tr in uniq_trials:
    if tr < 0 or tr >= ntrials_declared:
        continue
    mask = valid.copy()
    mask[valid] = tr_idx == tr
```

iii. In Step 5, the AI justified trial segmentation as using "trial start / corridor entry as alignment event" with "frame-wise trial segments or a derived uniform trial binning based on behavior/neural frame indices." The code operationalizes that idea through `ft_trInd`, not through the reference `ft_CorrSpc` window.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if the derived trial mask contains at least 2 frames. Sessions are then dropped if they end up with fewer than 2 valid trials. Upstream session filtering also excludes sessions without a matching spike file or with placeholder `TrialStim` values.

ii.
```python
for tr in uniq_trials:
    ...
    if mask.sum() < 2:
        continue
    ...

if len(nt) < 2:
    print(f'skipping {sess}: fewer than 2 valid trials')
    continue
```

iii. The only explicit justification is the decoder requirement that each session contain at least two trials. The notes do not document a reference-matched trial quality-control rule beyond that.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `spks` entry of each session's `*_neural_data.npy` file, concatenated across imaging planes. Retinotopy `iarea` is loaded separately only to label brain regions, not to construct the neural signal itself.

ii.
```python
obj = np.load(fn, allow_pickle=True).item()
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 4 and Step 5 explicitly state that the neural signal should be treated as deconvolved calcium-event activity from `spks`, concatenated across planes.

## 2-b. How is the `neural` data processed?

i. The AI concatenates plane-wise `spks`, trims the session to a common frame count, slices frames belonging to each trial, then resamples every trial to 60 bins by nearest-index selection and casts the result to `float16`.

ii.
```python
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
...
nmat_raw = spk[:, mask].astype(np.float32)
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. The main documented justification is in Step 10 and Step 12 of `CONVERSION_NOTES.md`: the first full dataset became 334 GB, so the script was revised to a fixed 60-bin representation with lower-precision dtypes to make the artifact smaller while supposedly matching the reference's 60-bin processing "more closely."

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is effectively no neural-unit filtering. All concatenated neurons are kept. Retinotopy is used only to assign labels such as `V1`, `medial`, `anterior`, `lateral`, `aHV`, or `unknown`; neurons outside the target visual areas are not removed.

ii.
```python
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
...
labels = ['V1' if x == 0 else 'medial' if x == 1 else 'anterior' if x == 2
          else 'lateral' if x == 3 else 'aHV' if x == 4 else 'unknown'
          for x in iarea]
...
return idx, uniq
```

iii. The notes justify preserving region labels from retinotopy, but they do not document a decision to filter neurons to the four coarse visual areas used by the reference. The trajectory later reports large `unknown` counts, showing that keeping all neurons was accepted.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned by taking all frames with a given `ft_trInd` value, treating the first retained frame as the start of the trial segment, and then resampling that segment to 60 bins. The code does not explicitly anchor the neural window to corridor-entry frames defined by `ft_CorrSpc` or `StartFr`.

ii.
```python
mask = valid.copy()
mask[valid] = tr_idx == tr
...
nmat_raw = spk[:, mask].astype(np.float32)
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. The notes say the decoder should be aligned to trial start / corridor entry, but the implementation choice was shaped by the AI's broader plan to use a fixed 60-bin trial representation from frame-wise segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted trials have 60 bins regardless of their original duration. The binning is index-based resampling, not frame-preserving or time-preserving resampling, and the metadata does not report a time bin size.

ii.
```python
def resample_matrix_time(mat, n_bins=60):
    ...
    idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
    return mat[:, idx].astype(np.float32, copy=False)

'time_bin_size': None,
```

iii. The stated justification in the notes is efficiency and file-size control: the AI switched to a "fixed 60-bin trial representation" after the initial artifact became too large.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives this input from frame times `ft` and per-trial absolute cue times `SoundTime`. It does not use `SoundFr`.

ii.
```python
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
soundtime = matlab_days_to_seconds(np.asarray(beh['SoundTime'], dtype=float))
...
cue_rel = soundtime[tr] - tvec
```

iii. In Step 5, the AI documented that `SoundTime` or `SoundFr` could be used and described the desired variable as "cue time minus current trial time." No more specific justification was recorded.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the AI subtracts each frame time in that trial from the trial's absolute `SoundTime`, producing a continuous time-to-cue trace, and then resamples that trace to 60 bins along with the other trial variables.

ii.
```python
tvec = ft[mask]
cue_rel = soundtime[tr] - tvec
...
inp_raw = np.vstack([
    cue_rel.astype(np.float32),
    ...
])
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. The Step 5 mapping table says the input should be "SoundTime or SoundFr relative to trial" and should be a continuous time-varying cue-minus-current-time variable.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same per-trial `tvec = ft[mask]` used to define the neural trial segment, then resampled to the same 60 bins as the neural data.

ii.
```python
nmat_raw = spk[:, mask].astype(np.float32)
tvec = ft[mask]
cue_rel = soundtime[tr] - tvec
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. The justification is implicit in the Step 5 plan to place decoder variables on a common per-trial time axis aligned with neural activity.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date embedded in the session name, not from the relative ordering of all sessions for a mouse. The code parses year, month, and day directly from the session id string.

ii.
```python
def session_day_value(session_name):
    parts = session_name.split('_')
    y, m, d = map(int, parts[1:4])
    return y * 10000 + m * 100 + d
```

iii. The notes justify using session ids and experiment ordering for this field, but the implementation simplifies that to a date code extracted from the session name.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI converts each session's date to a scalar `YYYYMMDD` value and repeats that same value across all 60 bins of every trial in the session.

ii.
```python
day_val = float(session_day_value(base))
...
day_arr = np.full(mask.sum(), float(day_val), dtype=np.float32)
...
inp_raw = np.vstack([
    cue_rel.astype(np.float32),
    day_arr,
    ...
])
```

iii. The explicit documented intent was only to create a continuous per-trial scalar repeated across time. The code does not implement the training-order count discussed in the notes.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. In the implemented code it is effectively derived from `ft` alone: the AI subtracts the first retained frame time of the trial from each frame time in that trial. `Trial_start_time` is loaded but never used.

ii.
```python
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
trialstart = matlab_days_to_seconds(np.asarray(beh['Trial_start_time'], dtype=float))
...
tvec = ft[mask]
time_since_start = tvec - tvec[0]
```

iii. Step 5 says this variable should be "trial-relative time ... from 0 to trial duration" on the same time base as neural data. The AI appears to have implemented that idea using the first retained frame rather than the recorded trial-start variable.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The trial time vector `tvec` is formed from frame times belonging to the trial, the first element of that vector is treated as time zero, and the result is then resampled to 60 bins.

ii.
```python
tvec = ft[mask]
time_since_start = tvec - tvec[0]
...
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. No explicit justification beyond "trial-relative time" was documented. The implementation reflects a pragmatic frame-local definition rather than use of `Trial_start_time`.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from the same `tvec` and trial mask used to slice the neural signal, and then it is resampled to the same 60 bins as the neural trial matrix.

ii.
```python
nmat_raw = spk[:, mask].astype(np.float32)
tvec = ft[mask]
time_since_start = tvec - tvec[0]
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. The justification is the same common-axis plan documented in Step 5.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from `isRew`, the per-trial rewarded-corridor indicator.

ii.
```python
isrew = np.asarray(beh['isRew']).astype(bool)
...
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
```

iii. Step 5 explicitly says reward availability should come from corridor identity (`isRew`) rather than actual reward delivery.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. For each trial, the AI fills a constant vector of 0s or 1s using the trial's `isRew` value, then resamples it with the other inputs to 60 bins.

ii.
```python
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
...
inp_raw = np.vstack([
    ...,
    reward_available,
])
```

iii. The Step 5 notes justify this as a binary per-trial indicator because unsupervised sessions can still contain cues while no water is available.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `TrialStim`, not `WallName`. The AI first builds a global vocabulary from unique `TrialStim` strings in selected sessions and then indexes each trial's `TrialStim` value.

ii.
```python
if 'TrialStim' in beh:
    vals = [str(x) for x in np.unique(beh['TrialStim']) if str(x) != 'stimulus_of_trial']
    stim_names.update(vals)
...
trialstim = np.asarray(beh['TrialStim']).astype(str)
...
stim_idx = stim_to_idx[str(trialstim[tr])]
```

iii. In Step 5, the AI explicitly considered "`TrialStim` and/or canonical stimulus identity from `stim_id`" and said categories could include swap variants.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI collects unique `TrialStim` strings from all selected sessions, sorts them to build `stim_names`, assigns each string an integer id, broadcasts that id across the trial, and resamples/broadcasts it to 60 bins. It does not collapse stimuli to four base textures.

ii.
```python
stim_names = sorted(stim_names)
stim_to_idx = {s: i for i, s in enumerate(stim_names)}
...
stim_idx = stim_to_idx[str(trialstim[tr])]
stim_arr = np.full(mask.sum(), stim_idx, dtype=np.uint8)
out_raw = np.vstack([stim_arr, lick, pos_bin.astype(np.uint8), speed_bin.astype(np.uint8)])
```

iii. The notes justify this by saying visual stimulus should remain categorical and that categories may include swap variants. The trajectory later shows the resulting `stim_names` list contained seven categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. The implemented code derives licking from `LickTime` and `LickTrind`, not from `LickFr`.

ii.
```python
lick = np.zeros(mask.sum(), dtype=np.uint8)
if len(beh['LickTrind']) > 0:
    lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
```

iii. Step 5 lists `LickTime / LickFr / LickTrind` as candidate sources for a binary time-varying lick output. No stronger justification is documented.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI looks up lick times whose `LickTrind` equals the trial id, uses `np.searchsorted(..., side='left')` to map those lick times onto the trial's frame-time vector `tvec`, and sets the corresponding bins to 1. After that it resamples the binary vector to 60 bins. Because `tvec` is converted to seconds but `LickTime` is not, this implementation compares mismatched units.

ii.
```python
tvec = ft[mask]
...
lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
if len(lick_times) > 0:
    inds = np.searchsorted(tvec, lick_times, side='left')
    inds = inds[(inds >= 0) & (inds < len(tvec))]
    lick[inds] = 1
...
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. The only documented justification is that licking should be a binary time-varying output. The unit-handling detail is not discussed in the notes.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is intended to be aligned by constructing lick indicators on the same per-trial frame-time vector `tvec` used for the neural slice and then resampling both to 60 bins. In practice the unit mismatch between `LickTime` and `tvec` makes this alignment faulty.

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

iii. The alignment rationale is implicit in the Step 5 common-axis plan, but the implementation details are not justified in the notes.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the framewise position variable `ft_Pos`.

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
...
pos = ft_pos[mask]
pos_bin = discretize_position(pos, corridor_length)
```

iii. Step 5 directly maps `ft_Pos` or interpolated position to the position-bin output.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI clips positions to the corridor length, divides the corridor into four equal sections using `corridor_length / 4`, floors the result to integers 0-3, and then resamples the per-frame labels to 60 bins.

ii.
```python
def discretize_position(pos, corridor_length):
    pos = np.asarray(pos, dtype=float)
    bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)).astype(int)
    return np.clip(bins, 0, 3)
```

iii. The notes justify this as meeting the decoder requirement for four equal 1 m bins and describe `ft_Pos` as the source.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholding is equal-width binning: `[0, corridor_length/4)`, `[corridor_length/4, corridor_length/2)`, `[corridor_length/2, 3*corridor_length/4)`, and `[3*corridor_length/4, corridor_length]`, clipped into category ids 0-3.

ii.
```python
bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)).astype(int)
return np.clip(bins, 0, 3)
```

iii. This follows the decoder-task requirement noted in Step 5 for four equal 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled from the same per-trial frame mask as the neural data and then resampled to the same 60 bins as the neural trial.

ii.
```python
nmat_raw = spk[:, mask].astype(np.float32)
pos = ft_pos[mask]
pos_bin = discretize_position(pos, corridor_length)
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. The justification is the general common per-trial axis described in Step 5.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the framewise running-speed variable `ft_RunSpeed`.

ii.
```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
...
speed = ft_speed[mask]
speed_bin = discretize_speed(speed, speed_edges)
```

iii. Step 5 directly maps `ft_RunSpeed` to the running-speed-bin output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first computes three global speed quantiles over all finite `ft_RunSpeed` values from all selected sessions, then thresholds each trial's framewise speed values against those global edges and resamples the labels to 60 bins.

ii.
```python
def compute_speed_edges(beh_by_session, selected_sessions):
    vals = []
    for sess in selected_sessions:
        x = np.asarray(beh_by_session[sess]['ft_RunSpeed'], dtype=float)
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)
    allv = np.concatenate(vals) if vals else np.array([0.0, 1.0])
    edges = np.quantile(allv, [0.25, 0.5, 0.75])
    return edges.astype(float)
```

iii. Step 5 explicitly says running speed should be discretized into 4 quartile bins "over dataset", and the code follows that plan.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Category 0 is assigned by default; categories 1, 2, and 3 are assigned when speed exceeds the first, second, and third global quantile thresholds, respectively.

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

iii. The justification is again the Step 5 plan to discretize speed into four quartile bins, though the notes do not discuss ties or per-session vs global thresholds.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is taken from the same per-trial frame mask as the neural data and then resampled to the same 60 bins as the neural trial matrix.

ii.
```python
nmat_raw = spk[:, mask].astype(np.float32)
speed = ft_speed[mask]
speed_bin = discretize_speed(speed, speed_edges)
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. The notes justify a shared per-trial axis for all neural and behavioral streams.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI trims each session to the minimum available frame count across several framewise behavior arrays and the spike matrix, ignores non-finite `ft_trInd` entries, skips trials with fewer than 2 retained frames, skips sessions with fewer than 2 valid trials, allows missing retinotopy by assigning all neurons to `unknown`, and truncates or pads retinotopy label arrays to match the neural population size.

ii.
```python
frame_lengths = [len(np.asarray(beh[k])) for k in frame_keys if k in beh]
frame_lengths.append(spk.shape[1])
nfr = min(frame_lengths)
...
valid = np.isfinite(ft_trInd)
...
if mask.sum() < 2:
    continue
...
if ret is None or 'iarea' not in ret:
    return np.zeros(n_neurons, dtype=np.int64), ['unknown']
...
if len(idx) != n_neurons:
    m = min(len(idx), n_neurons)
    idx = idx[:m]
    if m < n_neurons:
        pad = np.full(n_neurons - m, lab_to_idx.get('unknown', 0), dtype=np.int64)
        idx = np.concatenate([idx, pad])
```

iii. The notes discuss the need to handle missing data sensibly and document the session-count mismatch between behavior and spike files, but they do not justify these fallback rules in detail. The trajectory also shows the AI accepted large `unknown` brain-region counts.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading and concatenating the very large per-session spike files, then constructing/resampling every trial to 60 bins and finally serializing the resulting nested object. The notes especially emphasize that full-dataset size and verification time drove the redesign.

ii.
```python
obj = np.load(fn, allow_pickle=True).item()
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
...
for tr in uniq_trials:
    ...
    nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. Step 10 and Step 12 of `CONVERSION_NOTES.md` explicitly say the first full artifact was 334 GB and that the code was revised around a compact 60-bin representation to reduce runtime and file size.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-trial reconstruction of boolean masks from `ft_trInd`, the repeated per-output call to `resample_labels_1d`, the loop that computes global speed edges by concatenating per-session arrays, and the region-remapping loops in `build_brain_region_idx`.

ii.
```python
for tr in uniq_trials:
    mask = valid.copy()
    mask[valid] = tr_idx == tr
    ...

out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)

for sess in selected_sessions:
    x = np.asarray(beh_by_session[sess]['ft_RunSpeed'], dtype=float)
```

iii. The AI documented a general goal of writing efficient, vectorized code, but it did not separately analyze these loops in the notes. These opportunities are inferred from the final implementation.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly rebuilds per-trial masks from `ft_trInd`, repeatedly resamples neural, input, and each output stream to 60 bins trial by trial, and repeatedly remaps local brain-region labels into the growing global region list.

ii.
```python
for tr in uniq_trials:
    mask = valid.copy()
    mask[valid] = tr_idx == tr
    ...
    nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
    inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
    out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)

for b in br_names:
    if b not in all_brain_regions:
        all_brain_regions.append(b)
```

iii. The trajectory and notes show that the AI's main response to scalability problems was to introduce the repeated 60-bin resampling step everywhere, rather than to preserve the frame grid used in the reference solution.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several values are computed or stored without downstream use in the final dataset: `beh_source`, `trialstart`, `kept_trial_ids`, and the helper functions `neu_area_ID()` and `get_cat_id()` are unused; the per-session `base_session` and `elapsed_sec` fields are only reporting metadata; and optional plotting code performs additional work solely for inspection. More importantly, the 60-bin resampling is an extra processing layer that is not part of the reference solution.

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
def neu_area_ID(iarea):
    ...
def get_cat_id(WallName, isRew):
    ...
```

iii. The notes justify some of this extra work as debugging or compaction support, especially the 60-bin representation and optional processing plots, but they do not argue that these steps are required by the reference processing.
