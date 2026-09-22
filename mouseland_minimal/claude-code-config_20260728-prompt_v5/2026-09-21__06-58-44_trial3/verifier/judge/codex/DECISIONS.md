# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the master session index from `beh/Imaging_Exp_info.npy`, deduplicates recordings by `(mname, datexp, blk)`, then reads behavior from `Beh_<exp_type>.npy`, spikes from `spk/<mname>_<datexp>_<blk>_neural_data.npy`, and retinotopy from `retinotopy/<mname>_<datexp>_trans.npz`.

ii. 
```python
exp_info = np.load(
    os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True
).item()
```
```python
beh_cache[exp_type] = np.load(
    os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True
).item()
```
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```
```python
dtrans = np.load(os.path.join(root, fn), allow_pickle=True)
return dtrans['iarea']
```

iii. The trajectory docstring says the goal is to include all sessions from all experiment types, deduplicated by `(mname, datexp, blk)`, and the code comments say a duplicate recording can appear under multiple experiment types so the first behavior source is reused.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name `mname`. The output `subjects` list is built in first-seen order, and `subject_idx` is the index of each session's mouse in that list.

ii. 
```python
if mname not in subjects_set:
    subjects_set.append(mname)
```
```python
subject_idx_all.append(subjects_set.index(mname))
```

iii. The trajectory contains no deeper justification beyond using `mname` as the subject identifier from the experiment index.

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` triple. The agent keeps only the first occurrence of each triple when it appears under multiple experiment types.

ii. 
```python
key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if key not in unique_sessions:
    ...
    unique_sessions[key] = {
        'mname': s['mname'],
        'datexp': s['datexp'],
        'blk': s['blk'],
        'exp_type': exp_type,
        'beh_key': beh_key,
    }
```

iii. The trajectory docstring explicitly says to deduplicate by `(mname, datexp, blk)` and to pick the first encountered experiment type/behavior key for that physical recording.

## 1-d. How are the data split into trials?

i. Trials are split using framewise trial labels `ft_trInd`, keeping only frames inside the textured corridor where `ft_CorrSpc` is true. Each trial is the set of those corridor frames for one trial index.

ii. 
```python
mask = (ft_trInd == t) & ft_CorrSpc
frames = np.where(mask)[0]
```

iii. The trajectory docstring states that trial start is corridor entry and that all textured-corridor frames should be extracted for each trial. It also says this is done regardless of running state.

## 1-e. How are trials filtered based on quality controls?

i. The agent drops any trial with fewer than 3 corridor frames and drops sessions with fewer than 2 surviving trials. It does not implement the reference solution's long-trial outlier filter.

ii. 
```python
MIN_FRAMES_PER_TRIAL = 3
...
if len(frames) < MIN_FRAMES_PER_TRIAL:
    continue
```
```python
if len(neural_trials) < 2:
    print(f"  Only {len(neural_trials)} valid trials, skipping session")
    skipped_sessions += 1
    continue
```

iii. The only explicit justification in the trajectory is the top-level note that trials need enough frames to be useful and sessions need at least two valid trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the `spks` arrays in the spike file, concatenated across planes. Brain-region assignment comes from retinotopy `iarea`.

ii. 
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```
```python
return dtrans['iarea']
```

iii. The trajectory docstring says neural data are deconvolved calcium traces from Suite2p output, concatenated across planes.

## 2-b. How is the `neural` data processed?

i. The agent concatenates planes, filters neurons to four visual areas, then slices trial-specific corridor frames. It does not further transform the traces or cast them to `float16`.

ii. 
```python
spk_filtered = spk[neuron_indices, :]
...
neural_trial = spk_filtered[:, frames]
```

iii. The trajectory justifies this by calling the traces "deconvolved calcium traces" and by stating that the per-trial extraction should use corridor frames aligned to trial start.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons in V1, mHV, lHV, or aHV are kept. All others are dropped; sessions with zero surviving neurons are skipped.

ii. 
```python
idx['V1'] = iarea == 8
idx['mHV'] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
idx['lHV'] = (iarea == 5) | (iarea == 6)
idx['aHV'] = (iarea == 3) | (iarea == 4)
```
```python
valid_neurons = area_idx['V1'] | area_idx['mHV'] | area_idx['lHV'] | area_idx['aHV']
neuron_indices = np.where(valid_neurons)[0]
```

iii. The trajectory docstring says this matches the reference area's mapping and explicitly notes excluding neurons outside those visual cortex areas.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to trial start defined as corridor entry. The kept bins are the corridor frames for that trial, so each trial begins at its own first corridor frame and remains variable length.

ii. 
```python
mask = (ft_trInd == t) & ft_CorrSpc
frames = np.where(mask)[0]
...
neural_trial = spk_filtered[:, frames]
```

iii. The trajectory docstring explicitly says "Temporal alignment: trial start = corridor entry."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is inferred from the median difference of `ft` in the first session, converted to milliseconds. No temporal rebinning is applied; one neural frame is one time bin.

ii. 
```python
dt_days = np.median(np.diff(ft))
time_bin_ms = dt_days * 24 * 3600 * 1000
time_bin_s = time_bin_ms / 1000
```

iii. The trajectory comments say the decoder needs uniform time bins and the data are kept on the native frame grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and the integer frame indices in the trial window, scaled by a single global `time_bin_s`. The code does not use per-session `ft` timestamps for this variable.

ii. 
```python
sound_fr = SoundFr[t]
time_to_cue = (sound_fr - frames) * time_bin_s
```

iii. The trajectory comment says this should be "positive before cue, negative after." No separate justification for using frame indices instead of timestamps is present.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the agent subtracts each kept frame index from `SoundFr[t]` and multiplies by `time_bin_s`, yielding a continuous per-bin time-to-cue trace.

ii. 
```python
time_to_cue = (sound_fr - frames) * time_bin_s
```

iii. The trajectory only justifies the sign convention in the inline comment.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the same `frames` array used to slice neural activity for that trial.

ii. 
```python
neural_trial = spk_filtered[:, frames]
...
time_to_cue = (sound_fr - frames) * time_bin_s
```

iii. The trajectory's general alignment rationale is that all trial variables are built on the same corridor-frame window.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session dates `datexp` grouped by mouse name `mname`.

ii. 
```python
def get_date(datexp):
    return datetime.strptime(datexp, '%Y_%m_%d')
```
```python
mname = info['mname']
d = get_date(info['datexp'])
```

iii. The trajectory docstring says day of training is computed as calendar days since the mouse's first recording session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the first session date is found, and each session gets `(session_date - first_date).days`. That scalar is then broadcast across all bins in each trial.

ii. 
```python
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
...
days[key] = (d - first_d).days
```
```python
day_arr = np.full(n_frames, day_of_training, dtype=np.float32)
```

iii. The trajectory explicitly states the choice to use calendar days since first recording.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the kept corridor-frame indices themselves, specifically `frames` and `frames[0]`, plus the global bin duration `time_bin_s`. The code does not use `StartFr` or per-frame timestamps.

ii. 
```python
time_since_start = (frames - frames[0]) * time_bin_s
```

iii. The trajectory comment only says this is "Time since trial start (seconds)."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The first retained frame in the trial is treated as time zero, and later bins are offsets in multiples of `time_bin_s`.

ii. 
```python
time_since_start = (frames - frames[0]) * time_bin_s
```

iii. No additional justification appears in the trajectory beyond the intent to align to trial start.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed directly from the same `frames` array used for neural slicing, so it has exactly the same length and binning as the neural trial.

ii. 
```python
neural_trial = spk_filtered[:, frames]
time_since_start = (frames - frames[0]) * time_bin_s
```

iii. The trajectory's alignment logic is the shared frame window per trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial `isRew` field.

ii. 
```python
isRew = beh['isRew']
...
reward_avail = np.full(n_frames, float(isRew[t]), dtype=np.float32)
```

iii. No separate justification appears; the code directly uses the reward-trial flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The scalar `isRew[t]` is converted to float and repeated across all bins of the trial.

ii. 
```python
reward_avail = np.full(n_frames, float(isRew[t]), dtype=np.float32)
```

iii. The trajectory treats reward availability as a per-trial decoder input, so it is broadcast in time.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName` for each trial, with the possible category names first collected from session-level `UniqWalls`.

ii. 
```python
for name in beh['UniqWalls']:
    all_stim_names.add(name)
```
```python
stim_cat = stim_to_idx[WallName[t]]
```

iii. The trajectory comments say sessions may expose different wall names, so it builds a dataset-wide category list from those names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent sorts all unique wall names seen across sessions, maps each exact wall name to an integer category, and broadcasts that category across all time bins of the trial. It does not collapse `circle1/circle2/...` into a four-class base texture label.

ii. 
```python
stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(stim_names)}
```
```python
stim_cat = stim_to_idx[WallName[t]]
stim_arr = np.full(n_frames, stim_cat, dtype=int)
```

iii. The trajectory provides no stronger justification than preserving the observed stimulus names from the behavior files.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`.

ii. 
```python
LickFr = beh['LickFr']
LickTrind = beh['LickTrind']
```

iii. The trajectory docstring says licking is a binary per-frame variable derived from `LickFr`; the implementation also relies on `LickTrind` to select licks belonging to each trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the agent selects licks whose `LickTrind` matches that trial, rounds their `LickFr` values to integers, maps those global frame numbers into local trial-frame indices, and marks those bins as 1.

ii. 
```python
trial_lick_mask = (LickTrind.astype(int) == t)
trial_lick_frames = np.round(LickFr[trial_lick_mask]).astype(int)
frame_to_local = {f: i for i, f in enumerate(frames)}
for lf in trial_lick_frames:
    if lf in frame_to_local:
        lick_arr[frame_to_local[lf]] = 1
```

iii. The trajectory docstring explicitly says the choice was to derive licking from `LickFr` rounded to the nearest integer frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned by converting lick frame numbers into indices within the exact same `frames` window used for the neural trial.

ii. 
```python
neural_trial = spk_filtered[:, frames]
...
frame_to_local = {f: i for i, f in enumerate(frames)}
```

iii. The trajectory's general justification is shared framewise alignment to the trial window.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from framewise `ft_Pos`.

ii. 
```python
ft_Pos = beh['ft_Pos'][:nfr]
...
pos_bins = discretize_position(ft_Pos[frames])
```

iii. The trajectory docstring says position comes from `ft_Pos` in decimeters.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position values are integer-divided by 10 decimeters and clipped to `[0, 3]`, producing four 1 m bins.

ii. 
```python
bins = np.clip(positions // 10, 0, 3).astype(int)
```

iii. The trajectory docstring justifies this as the required 4 equal-length 1 m bins over the 4 m corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40+]` decimeters after clipping.

ii. 
```python
bins = np.clip(positions // 10, 0, 3).astype(int)
```

iii. The thresholds are spelled out in the `discretize_position` docstring.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled from `ft_Pos` using the same `frames` array used to extract neural data, so it is already on the same frame grid.

ii. 
```python
neural_trial = spk_filtered[:, frames]
pos_bins = discretize_position(ft_Pos[frames])
```

iii. The trajectory's trial-window logic is the alignment justification.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise `ft_RunSpeed`.

ii. 
```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
...
speed_bins = discretize_speed(ft_RunSpeed[frames], speed_quartiles)
```

iii. The trajectory docstring says running speed comes from `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first collects all corridor-frame speeds across all sessions, computes three global percentile thresholds at 25, 50, and 75%, then digitizes each trial's speeds with those thresholds.

ii. 
```python
corridor_speeds = ft_RunSpeed[ft_CorrSpc]
all_speeds.append(corridor_speeds)
...
quartiles = np.percentile(all_speeds, [25, 50, 75])
```
```python
bins = np.digitize(speeds, quartiles)
```

iii. The trajectory docstring explicitly justifies this as "4 quartile bins computed across all corridor frames from all sessions."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The category boundaries are the three global percentile thresholds computed from pooled corridor-frame speeds, and `np.digitize` maps them into categories 0 to 3.

ii. 
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
```
```python
bins = np.digitize(speeds, quartiles)
```

iii. The trajectory gives the percentile-bin rationale directly in the module docstring.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same kept frame indices `frames` used for the neural trial.

ii. 
```python
neural_trial = spk_filtered[:, frames]
speed_bins = discretize_speed(ft_RunSpeed[frames], speed_quartiles)
```

iii. The trajectory aligns all time-varying variables through the same corridor-frame window.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code truncates framewise behavior arrays to the number of neural frames, skips sessions if spike or retinotopy loading fails, skips sessions with zero retained neurons, and skips trials shorter than 3 frames. Licks that do not land on kept frames are ignored.

ii. 
```python
ft_trInd = beh['ft_trInd'][:nfr]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
ft_Pos = beh['ft_Pos'][:nfr]
ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
```
```python
except Exception as e:
    print(f"  Error loading neural data: {e}, skipping")
```
```python
if len(neuron_indices) == 0:
    print(f"  No valid neurons, skipping")
```
```python
if lf in frame_to_local:
    lick_arr[frame_to_local[lf]] = 1
```

iii. The trajectory offers no separate data-cleaning theory beyond practical skipping/truncation and the requirement to keep decodable sessions.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are repeated spike-file loads and concatenations. In particular, `compute_speed_quartiles` loads every spike file once just to get `nfr`, and the main session-processing loop loads them again for conversion.

ii. 
```python
spk_data = np.load(spk_path, allow_pickle=True).item()['spks']
nfr = spk_data[0].shape[1]
```
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```

iii. The trajectory contains no explicit efficiency analysis, but the code structure shows a full-dataset pre-pass over spike files plus the main pass.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial scan over all frames, the per-trial per-lick loop that builds `lick_arr`, and the per-region loop used to fill `region_idx` could all have been vectorized further.

ii. 
```python
for t in range(ntrials):
    mask = (ft_trInd == t) & ft_CorrSpc
    frames = np.where(mask)[0]
```
```python
for lf in trial_lick_frames:
    if lf in frame_to_local:
        lick_arr[frame_to_local[lf]] = 1
```
```python
for r_idx, region in enumerate(brain_regions):
    mask = area_idx[region][neuron_indices]
    region_idx[mask] = r_idx
```

iii. The trajectory gives no efficiency justification for these loops.

## 12-c. What processing does the code repeat multiple times?

i. It repeats spike-file loading: once in `compute_speed_quartiles` and again during session conversion. It also repeatedly casts `LickTrind` inside the trial loop.

ii. 
```python
spk_data = np.load(spk_path, allow_pickle=True).item()['spks']
...
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
```
```python
trial_lick_mask = (LickTrind.astype(int) == t)
```

iii. No explicit trajectory justification is given for this repeated work.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code creates `frame_set` but never uses it. It also computes a dataset-wide stimulus-name vocabulary that preserves fine-grained wall identities even though the human reference collapses them to four base texture classes for decoding.

ii. 
```python
frame_set = set(frames.tolist())
frame_to_local = {f: i for i, f in enumerate(frames)}
```
```python
all_stim_names = set()
...
stim_names = sorted(all_stim_names)
```

iii. The trajectory gives no justification for `frame_set`; the fine-grained stimulus vocabulary is only indirectly justified by preserving observed `WallName` labels.
