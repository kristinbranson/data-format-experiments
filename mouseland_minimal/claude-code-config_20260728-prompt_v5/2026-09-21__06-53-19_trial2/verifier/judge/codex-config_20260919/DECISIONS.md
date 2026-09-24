# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `Imaging_Exp_info.npy`, then each `Beh_<exp_type>.npy`, retaining the first occurrence of every unique mouse/date/block session. Neural planes and retinotopy are loaded per session.

ii.
```python
exp_info = np.load(os.path.join(BEH_ROOT, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
for exp_type in exp_info:
    Beh = np.load(os.path.join(BEH_ROOT, f'Beh_{exp_type}.npy'), allow_pickle=True).item()
    session_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
```

iii. The trajectory says all 89 unique sessions were included to maximize data and allow reward/day inputs to represent condition differences.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname`; indices are assigned in first-encounter order while sorted sessions are assembled.

ii.
```python
mname = info['ndb']['mname']
if mname not in subject_to_idx:
    subject_to_idx[mname] = len(subjects_list)
    subjects_list.append(mname)
session_subject_idx.append(subject_to_idx[mname])
```

iii. The agent observed that the experiment index explicitly supplies mouse identity.

## 1-c. How are the data split into sessions?

i. A session is the unique tuple `(mname, datexp, blk)`. Duplicate listings across experiment types are skipped.

ii.
```python
session_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if session_key in sessions:
    continue
```

iii. The trajectory reports finding 89 unique recordings and intentionally deduplicating experiment-type listings.

## 1-d. How are the data split into trials?

i. For each declared trial index, the agent selects frames labeled with that trial and marked as corridor space. Trials remain variable length.

ii.
```python
for trial_idx in range(ntrials):
    mask = (ft_trInd == trial_idx) & ft_CorrSpc
    frame_indices = np.where(mask)[0]
```

iii. It chose texture-corridor traversal because this matches corridor-focused analyses, four 1 m bins, and contiguous decoder sequences.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than two selected frames are dropped; sessions with fewer than two remaining trials are also dropped. There is no long-trial/outlier filter.

ii.
```python
if len(frame_indices) < 2:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. The trajectory justifies retaining both running and stopped frames for continuity, but gives no specific justification for the two-frame cutoff or failure to remove pathological long stops.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from `spks` arrays in each session neural file, concatenated over imaging planes. `iarea` from retinotopy supplies region labels.

ii.
```python
spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
ret = np.load(os.path.join(RET_ROOT, fn))
```

iii. The agent identified `spks` as the paper's deconvolved fluorescence and retinotopy as the neuron-area mapping.

## 2-b. How is the `neural` data processed?

i. Planes are concatenated, trial-frame columns are selected, and values are cast to float32. There is no normalization, padding, rebinning, or further signal processing.

ii.
```python
spk = np.concatenate(spk_data['spks'], axis=0)
neural_trial = spk[:, frame_indices].astype(np.float32)
```

iii. The trajectory cites the methods statement that analyses used deconvolved traces, so it did not redo fluorescence processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It does not filter neurons. V1/mHV/lHV/aHV are labeled, but neurons outside those areas and unknown codes are retained as `unassigned`.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
return np.array([BRAIN_REGION_MAP.get(int(ia), 4) for ia in iarea], dtype=np.int64)
```

iii. The agent explicitly reasoned that all neurons should be included because a decoder can learn which are useful, despite recognizing the visual-area mapping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Selected corridor frames are placed in their native order; the first selected corridor frame is treated as trial start. No padding or fixed window is used.

ii.
```python
frame_indices = np.where((ft_trInd == trial_idx) & ft_CorrSpc)[0]
neural_trial = spk[:, frame_indices].astype(np.float32)
```

iii. The agent chose corridor entry as the requested alignment and variable-duration trials to preserve contiguous native data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame is one bin; no rebinning is applied. A single bin duration is estimated from the median `ft` difference in the first sorted session (about 315 ms) and used globally.

ii.
```python
dt_seconds = float(np.median(np.diff(ft)) * 86400)
'time_bin_size': dt_ms
```

iii. The trajectory says native ~3.17 Hz frames preserve the finest available common grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr`, selected imaging-frame indices, and the globally estimated frame duration.

ii.
```python
time_to_sound = ((SoundFr[trial_idx] - frame_indices) * dt_seconds).astype(np.float32)
```

iii. The agent identified `SoundFr` as the sound event's frame coordinate.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every selected frame, its index is subtracted from the possibly fractional sound-frame coordinate, then multiplied by constant seconds per frame. Values are positive before and negative after the cue.

ii.
```python
(SoundFr[trial_idx] - frame_indices) * dt_seconds
```

iii. The trajectory describes this as a continuous time-to-event feature; it did not justify replacing actual timestamps/interpolation with constant frame spacing.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed at exactly the same `frame_indices` used to slice neural columns.

ii.
```python
neural_trial = spk[:, frame_indices]
time_to_sound = (SoundFr[trial_idx] - frame_indices) * dt_seconds
```

iii. The agent relied on behavior and neural streams sharing frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It uses `mname` and the calendar date in `datexp`, relative to that mouse's earliest included date.

ii.
```python
dt = parse_date(info['ndb']['datexp'])
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
```

iii. The module docstring explicitly describes the decision as calendar days from each mouse's first recording date.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The calendar-day difference is calculated and broadcast across every frame of every trial in the session.

ii.
```python
(parse_date(info['ndb']['datexp']) - mouse_first_date[info['ndb']['mname']]).days
day_train = np.full(T, day_of_training, dtype=np.float32)
```

iii. No further justification for calendar elapsed days rather than ordinal recorded training day appears in the trajectory.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the selected corridor frame indices and global frame duration; `StartFr` and `ft` are not used.

ii.
```python
time_since_start = ((frame_indices - frame_indices[0]) * dt_seconds).astype(np.float32)
```

iii. The agent treated the first corridor frame as corridor entry/trial start.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The first retained index is subtracted from each retained index and the result is scaled by constant seconds per frame, forcing the first value to zero.

ii.
```python
(frame_indices - frame_indices[0]) * dt_seconds
```

iii. This follows its stated corridor-entry alignment, but the trajectory does not discuss fractional `StartFr` or timestamp jitter.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The same frame-index vector defines both neural columns and time values.

ii.
```python
neural_trial = spk[:, frame_indices]
time_since_start = (frame_indices - frame_indices[0]) * dt_seconds
```

iii. The agent relied on common per-frame indexing.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is read from the per-trial `isRew` flag.

ii.
```python
isRew = beh['isRew']
reward_avail = np.full(T, float(isRew[trial_idx]), dtype=np.float32)
```

iii. The trajectory notes this varies for supervised sessions and remains zero for unrewarded cohorts.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trial flag is cast to float and broadcast to all retained frames.

ii.
```python
np.full(T, float(isRew[trial_idx]), dtype=np.float32)
```

iii. It is naturally per-trial, so the agent applied no transformation beyond broadcasting.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from per-trial `WallName`.

ii.
```python
WallName = beh['WallName']
stim_idx = STIM_TO_IDX[str(WallName[trial_idx])]
```

iii. The agent inspected wall names and chose them as the unmasked stimulus source.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each of 15 exact wall names gets its own sorted category index, which is broadcast across the trial. Swap variants and numbered exemplars are not collapsed into circle/leaf/rock/wood.

ii.
```python
ALL_STIMULI = sorted(['circle1', ..., 'wood5'])
stim_out = np.full(T, STIM_TO_IDX[str(WallName[trial_idx])], dtype=np.int64)
```

iii. The trajectory reports treating distinct textures as 15 classes and later interprets decoding chance as 1/15; it offers no justification for departing from the broader example categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses `LickFr` events selected by matching `LickTrind` to the current trial.

ii.
```python
lick_mask = beh['LickTrind'] == trial_idx
lick_frames_set = set(np.round(beh['LickFr'][lick_mask]).astype(int))
```

iii. The agent identified both fields as the lick event and its trial association.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick-frame coordinates are rounded to nearest integer and converted to a binary flag for each retained frame; multiple licks in one frame remain 1.

ii.
```python
return np.array([1.0 if f in lick_frames_set else 0.0 for f in frame_indices], dtype=np.float32)
```

iii. The stated decision was to derive a per-frame binary output from lick events. No rationale for rounding rather than truncation is recorded.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Binary flags are generated in the order of the same `frame_indices` used for neural data, with an additional trial-index filter.

ii.
```python
lick_out = compute_lick_per_frame(beh, trial_idx, frame_indices)
neural_trial = spk[:, frame_indices]
```

iii. The agent relied on `LickFr` being expressed on the imaging-frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from per-frame `ft_Pos` at the retained corridor frames.

ii.
```python
ft_Pos = beh['ft_Pos'][:nfr]
pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. The agent recognized `ft_Pos` as decimeter-scale corridor position.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Positions are floor-divided by 10 and clipped to codes 0–3.

ii.
```python
np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. This implements the requested four 1 m bins over the 4 m texture corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The implicit thresholds are 10, 20, and 30 decimeters, yielding 0–1, 1–2, 2–3, and 3–4 m categories; clipping handles out-of-range values.

ii.
```python
['0-1m', '1-2m', '2-3m', '3-4m']
np.clip(ft_Pos[frame_indices] // 10, 0, 3)
```

iii. The task explicitly asks for four equal 1 m spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed by the same selected imaging frames as neural data.

ii.
```python
neural_trial = spk[:, frame_indices]
pos_out = ... ft_Pos[frame_indices] ...
```

iii. The agent relied on the already frame-aligned behavior stream.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `ft_RunSpeed` at corridor frames.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
all_speeds.append(ft_RunSpeed[corridor_mask])
```

iii. The agent identified this as the direct per-frame speed measurement.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global 25th, 50th, and 75th value percentiles are computed over corridor frames from all sessions, then reused for every session.

ii.
```python
q25, q50, q75 = np.percentile(all_speeds, [25, 50, 75])
speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles).astype(np.int64)
```

iii. The agent's stated decision was global quartiles across all corridor frames so the output has four dataset-wide speed categories.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` assigns codes 0–3 using the three global percentile values. Ties, especially at zero, are not rank-split, so bins need not contain equal counts.

ii.
```python
np.digitize(ft_RunSpeed[frame_indices], speed_quartiles)
```

iii. The trajectory describes quartile edges but does not address tied values.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The speed stream is sliced at exactly the same imaging frames as neural data.

ii.
```python
neural_trial = spk[:, frame_indices]
speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles)
```

iii. The agent relied on native frame alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior streams are truncated to neural frame count; sessions missing from a behavior dictionary are not collected; trials under two frames and sessions under two valid trials are skipped. Unknown area codes become `unassigned`. There is no explicit NaN handling beyond excluding NaN trial indices in the speed-threshold pass.

ii.
```python
if beh_key in Beh: sessions[session_key] = ...
ft_trInd = beh['ft_trInd'][:nfr]
if len(frame_indices) < 2: continue
BRAIN_REGION_MAP.get(int(ia), 4)
```

iii. The agent focused on robust matching and format validity; no broader missing-data policy was documented.

## 12-a. What are the most time-consuming steps of the code?

i. Loading each very large neural file, concatenating planes, copying each trial slice to float32, and serializing the roughly 276 GB result dominate. Neural files are also reopened once during the global speed pass merely to obtain frame counts.

ii.
```python
spk_data = np.load(...).item()
spk = np.concatenate(spk_data['spks'], axis=0)
neural_trial = spk[:, frame_indices].astype(np.float32)
```

iii. The trajectory observed that the full product was enormous; most recorded runtime discussion concerned downstream decoder projection/PCA, while conversion I/O and copies are the evident conversion bottlenecks.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Region mapping uses a Python comprehension, licking tests membership once per frame, trials rescan the full session mask, and session speed collection loops serially. Region mapping and licking in particular could use lookup arrays/`np.isin`; trial frames could be grouped once.

ii.
```python
[BRAIN_REGION_MAP.get(int(ia), 4) for ia in iarea]
[1.0 if f in lick_frames_set else 0.0 for f in frame_indices]
for trial_idx in range(ntrials):
    mask = (ft_trInd == trial_idx) & ft_CorrSpc
```

iii. The trajectory provides no explicit vectorization analysis.

## 12-c. What processing does the code repeat multiple times?

i. Every spike file is loaded in `get_nfr` for speed thresholds and again in `load_spk` for conversion. Corridor masks are recomputed per trial, and trial slicing repeatedly copies the same session-scale neural array into separate arrays.

ii.
```python
nfr = get_nfr(ndb)
...
spk = load_spk(ndb)
```

iii. The agent created `get_nfr` to avoid retaining full data, but did not discuss the repeated disk reads it causes.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `n_neurons` is assigned but unused; `ft_trInd` is loaded in the speed pass but used only for a NaN check; full neural files are deserialized just to inspect one plane's frame count; garbage collection is invoked repeatedly. More materially, retaining unassigned neurons and float32 precision greatly enlarges data and downstream projection work compared with the reference's filtered float16 representation.

ii.
```python
n_neurons, nfr = spk.shape
ft_trInd = beh['ft_trInd'][:nfr]
gc.collect()
```

iii. The trajectory later found downstream training consumed hundreds of GB and spent a long time projecting tens of thousands of neurons, but it did not revisit these conversion choices.
