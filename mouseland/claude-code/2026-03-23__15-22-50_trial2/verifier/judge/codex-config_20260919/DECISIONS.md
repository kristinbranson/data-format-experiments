# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `Imaging_Exp_info.npy` as the master index, builds one entry per unique spike-file key, then loads retinotopy, plane-wise `spks`, and the associated `Beh_<experiment>.npy` dictionary for every session. The full run found all 89 sessions and 19 mice.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f"Beh_{session_info['exp_type']}.npy"),
                  allow_pickle=True).item()
```

iii. The notes say the index contains 142 references but only 89 physical recordings, matching the 89 neural files and the paper's 89 recordings. The agent therefore used each physical recording once.

## 1-b. How are the data split into subjects?

i. The `mname` field defines a mouse. Subject indices are assigned in first-session order while iterating sorted session keys; `subjects` preserves that same order.

ii.
```python
mname = info['db']['mname']
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
subject_idx_list.append(subjects_seen[mname])
subjects = sorted(subjects_seen.keys(), key=lambda x: subjects_seen[x])
```

iii. The notes report 19 subjects and state that the mouse name is the intended subject identifier.

## 1-c. How are the data split into sessions?

i. A session is the unique `(mname, datexp, blk)` key. Duplicate index entries are collapsed, preferring an entry without `stimtype` when available.

ii.
```python
spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if spk_key not in session_map or 'stimtype' not in ndb:
    session_map[spk_key] = {'exp_type': exp_type, 'beh_key': beh_key, 'db': dict(ndb)}
```

iii. The agent verified that duplicate experiment-type entries refer to the same neural recording and described using one entry per neural file.

## 1-d. How are the data split into trials?

i. Trial `i` is sliced from integer `StartFr[i]` up to `StartFr[i+1]`, or to the usable end of the session for the final trial. Thus it includes the 4 m texture, gray space, and any remaining frames before the next start.

ii.
```python
start = StartFr[i]
end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
start = max(0, start)
end = min(nfr_use, end)
trial_spk = spk[:, start:end].copy()
```

iii. The notes explicitly choose start-to-next-start because the agent wanted to capture “corridor + gray space.”

## 1-e. How are trials filtered based on quality controls?

i. Only trials with fewer than two available frames are removed. A session is discarded if fewer than two trials survive. There is no stalled/outlier-trial filter.

ii.
```python
n_frames = end - start
if n_frames < 2:
    continue
if len(neural_trials) < 2:
    return None
```

iii. The notes state “use all trials (no filtering)” because the basic reference loading had no explicit trial filter. The full output consequently contains trials as long as 5,621 frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the `spks` list in each `<session>_neural_data.npy`; neuron area assignments come from `iarea` in the session's retinotopy file.

ii.
```python
planes = spk_data['spks']
iarea = load_retino(mname, datexp)
```

iii. The agent identified these as Suite2p deconvolved calcium traces and noted that retinotopy provides visual-area labels.

## 2-b. How is the `neural` data processed?

i. Valid neurons are selected within each plane, cast to `float16`, concatenated across planes, clipped to the common neural/behavior length, and sliced into trial matrices. No dF/F, deconvolution, interpolation, padding, or temporal rebinning is performed.

ii.
```python
filtered.append(plane[plane_mask].astype(np.float16))
return np.concatenate(filtered, 0)
spk = spk[:, :nfr_use]
trial_spk = spk[:, start:end].copy()
```

iii. The notes say the files already contain the deconvolved traces on which the paper's analyses were based. `float16` and pre-concatenation filtering were chosen to reduce the very large memory footprint.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons whose `iarea` is `-1` or `7` are excluded; retained codes are mapped to V1, mHV, lHV, or aHV. No activity/selectivity filter is applied.

ii.
```python
valid_mask = np.array([int(ia) not in EXCLUDED_AREAS for ia in iarea])
region_idx = np.array([BRAIN_REGIONS.index(AREA_MAP.get(int(ia), 'V1'))
                       for ia in iarea[valid_mask]], dtype=np.int64)
```

iii. The notes cite the reference area mapping and interpret `-1` and `7` as outside visual cortex. They report the lower neuron counts as the expected consequence.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The first column is integer `StartFr`, described as corridor entry/trial start. Each trial then continues until the next trial's start, with variable length and no padding.

ii.
```python
start = StartFr[i]
end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
trial_spk = spk[:, start:end].copy()
```

iii. The notes identify `StartFr` as the required alignment event and justify variable-length trials, but also intentionally include the gray-space portion.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging frames are retained at a nominal 3.17 Hz, recorded as approximately 315.5 ms per bin. There is no temporal rebinning.

ii.
```python
FRAME_RATE = 3.17
TIME_BIN_MS = 1000.0 / FRAME_RATE
'time_bin_size': TIME_BIN_MS,
```

iii. The notes say behavior and neural activity already share the imaging-frame grid, so resampling is unnecessary.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr`, the current absolute frame index, and a session sampling rate estimated from `ft` timestamps.

ii.
```python
SoundFr = beh['SoundFr']
dt = np.median(np.diff(ft[:min(1000, len(ft))])) * 86400
frame_idx = np.arange(start, end)
```

iii. The mapping plan describes the variable as frame distance from `SoundFr`, converted to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The agent computes `(SoundFr - frame_idx) / fs`, positive before and negative after the cue. If `SoundFr` is NaN, the whole trial is set to zero.

ii.
```python
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
else:
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. The intended justification is conversion of frame difference to seconds. The mapping table accidentally describes the sign backwards, while the code and variable name implement time remaining until the cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same absolute `[start:end)` frame indices and has exactly the same number of columns as the trial's neural matrix.

ii.
```python
frame_idx = np.arange(start, end)
trial_spk = spk[:, start:end].copy()
```

iii. The notes state that behavioral streams are frame-level and already share the neural frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is taken directly from the session metadata field `days`, falling back to `sess#`, and then to zero.

ii.
```python
for key in ['days', 'sess#']:
    if key in db:
        return int(db[key])
return 0
```

iii. The notes planned to use `sess#`/`days` when available and chronological order only as a fallback, although that chronological fallback was not implemented.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The chosen integer is cast to `float32` and broadcast over every frame of every trial in the session; it is not recomputed as ordinal recording day per mouse.

ii.
```python
day = np.float32(get_session_day(db))
np.full(n_frames, day, dtype=np.float32)
```

iii. The agent treated the metadata as a direct measure of training day. Full-run values range from 0 to 15.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the number of frames since the selected `StartFr` boundary and the estimated session sampling rate.

ii.
```python
start = StartFr[i]
fs = 1.0 / dt if dt > 0 else FRAME_RATE
```

iii. The mapping plan identifies `StartFr` as trial/corridor entry and calls for elapsed time from that event.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. A zero-based sequence of frame offsets is divided by `fs`, producing seconds starting at zero.

ii.
```python
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. The notes describe this as `(current_frame - StartFr) / fs`.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Its zero is the first selected neural column and it contains one value per neural column.

ii.
```python
trial_spk = spk[:, start:end].copy()
inp = np.stack([... (np.arange(n_frames) / fs).astype(np.float32), ...], axis=0)
```

iii. The agent relied on the common frame grid and identical slice length.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the per-trial `isRew` flag.

ii.
```python
isRew = beh['isRew']
```

iii. The notes identify `isRew` as the binary rewarded-corridor indicator.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trial flag is converted to a float and broadcast over all frames in that trial.

ii.
```python
np.full(n_frames, float(isRew[i]), dtype=np.float32)
```

iii. The agent considered it a per-trial scalar requiring no transformation beyond broadcasting.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It comes from per-trial `WallName`.

ii.
```python
WallName = beh['WallName']
stim = standardize_stim_name(str(WallName[i]))
```

iii. The notes preferred `WallName` as the direct per-trial visual identity.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. A hard-coded map folds rock/wood/brick names into crop-level circle/leaf labels but preserves crop and swap suffixes. All observed standardized names are sorted globally, integer encoded, and broadcast across the trial. The actual full output has nine labels, despite notes referring to eight.

ii.
```python
return STIM_CATEGORY_MAP.get(name, name)
all_stim_sorted = sorted(all_stim_names)
stim_to_idx = {s: i for i, s in enumerate(all_stim_sorted)}
output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. The agent reasoned that rock/wood/brick were equivalent to circle/leaf, but chose to retain crop and swap variants instead of collapsing to four broad texture categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, the recorded frame number for each lick.

ii.
```python
lick_fr = beh['LickFr'].astype(int)
```

iii. The notes identify `LickFr`/`LickTrind` as the lick source; the implementation uses `LickFr`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frames are truncated to integers; in-range frames are marked 1 in an otherwise-zero frame vector. Multiple licks in a frame remain one binary event.

ii.
```python
valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
lick_binary[lick_fr[valid_lick]] = 1
```

iii. The agent intended a binary “any lick in this imaging frame” output.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary vector is sliced with the identical `[start:end)` interval as neural activity.

ii.
```python
trial_spk = spk[:, start:end].copy()
lick_binary[start:end]
```

iii. The justification is that lick frame numbers and neural columns use the same imaging-frame coordinate system.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-level `ft_Pos` and `ft_CorrSpc`.

ii.
```python
ft_Pos = beh['ft_Pos'][:nfr_use]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr_use].astype(bool)
```

iii. The notes state that position is recorded in decimeters and distinguish the 4 m texture from the 2 m gray region.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code initializes every frame as category 4 (“gray”), then assigns categories 0–3 to texture frames in successive 10-decimeter ranges. Texture-edge positions at or above 40 are forced into category 3.

ii.
```python
pos_bins = np.full(nfr_use, 4, dtype=np.int64)
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3
```

iii. The agent recognized that the task asks for four 1 m bins but deliberately added a gray category because its trial windows include gray-space frames.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40)` decimeters for codes 0–3, plus code 4 for non-texture/gray frames.

ii.
```python
['0-1m', '1-2m', '2-3m', '3-4m', 'gray']
```

iii. The notes explicitly record the unresolved contradiction—“task says 4 bins”—then choose a fifth gray bin anyway.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The frame-level category array is sliced with the same session-frame interval as the neural matrix.

ii.
```python
pos_bins[start:end]
trial_spk = spk[:, start:end].copy()
```

iii. The agent relies on position and neural activity sharing the imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level `ft_RunSpeed` across all selected sessions.

ii.
```python
speeds = beh['ft_RunSpeed']
all_speeds.append(speeds)
```

iii. The notes call for discretizing raw running speed into four categories.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. All behavior speeds are concatenated globally. Zero/nonpositive values are excluded when computing the 25th, 50th, and 75th percentile thresholds; those three global thresholds are then applied to every frame, including stopped frames.

ii.
```python
all_speeds = np.concatenate(all_speeds)
valid = all_speeds > 0
return np.percentile(all_speeds[valid], [25, 50, 75])
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
```

iii. The agent described these as quartiles across all running frames. Its critical review accepted the resulting 47.1%/17.6%/17.6%/17.6% class split because stopped frames were excluded from threshold estimation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Codes 0, 1, 2, and 3 correspond to values below 6.91, at least 6.91, at least 22.18, and at least 39.23 respectively in the full run.

ii.
```python
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3
```

iii. The agent wanted common thresholds across the full dataset, although this does not make each final bin contain 25% of converted observations.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The per-frame speed category vector is sliced by the same `[start:end)` bounds as neural data.

ii.
```python
speed_bins[start:end]
trial_spk = spk[:, start:end].copy()
```

iii. The notes state that speed is already sampled per imaging frame.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavioral streams are clipped to their common usable length; out-of-range lick frames are ignored; NaN sound cues become all-zero cue-time vectors; trial bounds are clipped; trials under two frames and sessions under two trials are dropped. Any unhandled session exception is not caught inside conversion and would terminate the run.

ii.
```python
nfr_use = min(nfr, len(beh['ft']))
valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
start = max(0, start); end = min(nfr_use, end)
```

iii. The agent emphasized format validation and clipping inconsistent stream lengths. It did not document evidence that zero is a semantically safe replacement for a missing sound cue.

## 12-a. What are the most time-consuming steps of the code?

i. Loading, filtering, converting, copying, and serializing the enormous neural arrays dominate. The full conversion took about 31 minutes and wrote 177.26 GB.

ii.
```python
spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
filtered.append(plane[plane_mask].astype(np.float16))
trial_spk = spk[:, start:end].copy()
pickle.dump(data, f, protocol=4)
```

iii. The notes estimate roughly 15–25 seconds of neural loading/filtering plus processing per session and identify memory pressure as the principal constraint.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python comprehensions that build neuron masks/region indices, the four-position-bin loop, the per-trial construction loop, and the later nested stimulus-relabeling loop could be vectorized or replaced with array indexing. Trial creation still needs per-trial outputs because lengths vary, but much frame-level work could be precomputed.

ii.
```python
valid_mask = np.array([int(ia) not in EXCLUDED_AREAS for ia in iarea])
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
for i in range(ntrials):
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
```

iii. The agent did not discuss these vectorization opportunities; its optimization discussion focused instead on reducing peak memory by filtering each plane before concatenation.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are loaded once during global speed-threshold collection and again for session conversion; because `load_beh` loads a whole experiment dictionary on each call, the same large behavior file may also be reloaded for several sessions. Stimulus outputs are first stored as placeholders and then traversed again for encoding.

ii.
```python
speed_quartiles = collect_speed_quartiles(session_map, keys)
beh = load_beh(info)
# later, inside process_session:
beh = load_beh(session_info)
output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. The notes call behavior-only speed collection “fast” and prioritize avoiding neural reloads; they do not acknowledge the repeated behavior deserialization or second stimulus pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Each trial slice is explicitly copied even though serialization will materialize it; stimulus rows are initialized with zeros only to be overwritten later. If `--show-processing` is used, plotting concatenates and summarizes data solely for diagnostic figures. The stored `stim_names` intermediates are discarded after global encoding.

ii.
```python
trial_spk = spk[:, start:end].copy()
np.full(n_frames, 0, dtype=np.int64)  # placeholder for stim idx
all_pos = np.concatenate([outputs[ti][2] for ti in range(len(trials))])
```

iii. The trial copies were justified as avoiding references to the full session array, and plots as verification. The agent did not identify these as downstream-discarded work.
