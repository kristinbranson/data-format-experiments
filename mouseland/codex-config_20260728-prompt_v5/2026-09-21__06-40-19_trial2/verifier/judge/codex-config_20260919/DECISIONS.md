# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `beh/Imaging_Exp_info.npy` as the master index, canonicalizes entries to 89 unique `<mouse>_<date>_<block>` recordings, selects one behavior-file/key alias per recording, and loads that behavior dictionary, the session `spks`, and the date-level retinotopy file. Behavior files are cached two at a time.

ii.
```python
def load_experiment_index():
    return np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()

spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
retino = np.load(session.retino_path, allow_pickle=True)
```

iii. The notes say 142 index entries and 99 behavior keys reduce to 89 actual neural recordings; duplicate aliases were checked on core arrays, so collapsing them matches the paper's 89 recordings and avoids duplicate conversion.

## 1-b. How are the data split into subjects?

i. Subjects are taken from index field `mname`; the sorted unique names form `subjects`, and each retained session receives the corresponding integer `subject_idx`.

ii.
```python
subject=rec["subject"]
subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx.append(subject_to_idx[session.subject])
```

iii. The agent states that `mname` directly identifies the 19 imaging mice and that the converted count matches the paper.

## 1-c. How are the data split into sessions?

i. A session is the unique raw ID made from mouse, date, and block. Multiple experiment-type/behavior aliases for that ID are collapsed, with the first valid behavior source retained.

ii.
```python
def canonical_raw_id(entry):
    return f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"

rec = per_raw.setdefault(raw_id, {...})
rec["aliases"].add(beh_key)
```

iii. The agent verified reused aliases are duplicates and chose canonicalization to reproduce the paper's 89 recordings rather than count analysis labels as new sessions.

## 1-d. How are the data split into trials?

i. Frames are grouped by integer `ft_trInd`, but only frames simultaneously marked as corridor (`ft_CorrSpc`) and moving (`ft_move > 0`) are retained. Thus a trial is a possibly non-contiguous sequence of running frames, not every corridor frame.

ii.
```python
keep = finite & ft_corr & ft_move
frame_idx = np.flatnonzero(keep)
trial_ids = tr_int[keep]
...
groups[int(tids[0])] = frames.astype(np.int32, copy=False)
```

iii. The agent cites the paper's statement that analyses considered only running timepoints and the reference analyses' `ft_CorrSpc & (ft_move > 0)` masks.

## 1-e. How are trials filtered based on quality controls?

i. Trials with no retained corridor-running frames are dropped. No duration/outlier filter is applied; sessions are dropped only if fewer than two trials remain.

ii.
```python
for trial, frames in enumerate(groups):
    if frames.size == 0:
        continue
...
if len(neural_trials) < 2:
    continue
```

iii. The notes say this excludes unusable trials, satisfies decoder requirements, and preserves long trials because inspection attributed their long wall-clock spans to genuine sparse movement bouts rather than indexing errors.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from every plane array in `spk/<session>_neural_data.npy['spks']`; `retinotopy['iarea']` supplies one region annotation per neural row.

ii.
```python
spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
iarea = np.asarray(retino["iarea"], dtype=np.float32)
```

iii. The paper and source code analyze the released deconvolved `spks`; the notes therefore reject recomputing fluorescence or dF/F.

## 2-b. How is the `neural` data processed?

i. For each trial, the three plane arrays are sliced at its retained frames, concatenated by neuron, and cast to float32. There is no normalization, interpolation, padding, or temporal rebinning.

ii.
```python
neural_trial = np.concatenate(
    [part[:, frames] for part in spk_parts], axis=0
).astype(np.float32, copy=False)
```

iii. The agent says native deconvolved traces are already the analysis signal and trial-wise slicing avoids the memory cost of concatenating an entire session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed. `iarea` values are mapped to V1, mHV, lHV, aHV, `unassigned_7`, or `outside_visual`, retaining all 4,691,034 rows.

ii.
```python
out[iarea == 7] = 4
out[iarea == -1] = 5
...
brain_region_idx = map_iarea_to_region_idx(iarea)
```

iii. The agent argues that Suite2p already curated cells, the source has no universal additional cell-QC function, and retaining all rows preserves paper-level neuron counts and information.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial starts with its first retained corridor-running frame and is described as aligned to corridor entry/trial start. Values are selected at native timestamps; stationary frames within a trial are omitted, so elapsed-time gaps can remain between adjacent columns.

ii.
```python
groups = trial_frame_groups(beh)
...
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0)
```

iii. The agent chose corridor entry because the decoder task requires it, while retaining only running frames to follow the paper's analysis rule. It deliberately keeps absolute timestamps rather than compressing running time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The declared bin size is one native imaging frame, `1000/3.17 = 315.46 ms`; no temporal rebinning is applied, although removing stationary frames makes adjacent retained samples potentially farther apart in real time.

ii.
```python
FRAME_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
"time_bin_size": float(TIME_BIN_MS)
```

iii. The notes say native imaging frames are the source signal's natural resolution and behavior variables already map to that frame grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from trial-level `SoundTime` and frame-level `ft` timestamps.

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
frame_times = ft[frames]
```

iii. The agent selected the native timestamps used by cue-related analyses rather than reconstructing cue time from position.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the MATLAB-datenum difference `SoundTime[trial] - ft[frame]` is multiplied by 86,400 to obtain seconds; it is positive before and negative after the cue.

ii.
```python
time_to_cue = ((sound_time[trial] - frame_times) * 86400.0).astype(np.float32)
```

iii. The notes call this a direct, source-faithful time difference and validated it against raw values.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses `ft[frames]` for exactly the same frame indices used to slice neural activity.

ii.
```python
frame_times = ft[frames]
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0)
```

iii. The agent reports direct spot checks showing converted neural and cue-time arrays match raw slicing/calculation on the same retained frames.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `datexp`, parsed as a calendar date, and the mouse ID.

ii.
```python
return datetime.strptime(self.dateexp, "%Y_%m_%d").date()
subject_first_date[session.subject] = min(...)
```

iii. The agent says `sess#` and `days` metadata are inconsistent across branches, whereas recording dates exist for every canonical session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The value is the number of calendar days since that subject's first included recording, then repeated across every retained frame in the trial.

ii.
```python
session.raw_id: float((session.date_obj - subject_first_date[session.subject]).days)
...
day_of_training = np.full(frames.size, day_value, dtype=np.float32)
```

iii. The agent chose a continuous, universally defined measure and documented a full-data range of 0–92 days.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from trial-level `Trial_start_time` and frame-level `ft`.

ii.
```python
trial_start_time = np.asarray(beh["Trial_start_time"], dtype=np.float64)
frame_times = ft[frames]
```

iii. The agent identifies trial start with corridor entry, as required by the decoder task.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. It subtracts the trial start timestamp from each retained frame timestamp and converts days to seconds.

ii.
```python
time_since_start = ((frame_times - trial_start_time[trial]) * 86400.0).astype(np.float32)
```

iii. The agent retains wall-clock elapsed time, including pauses omitted by the running mask, rather than compressing time.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The calculation uses the same `frames` indices as the neural slice.

ii.
```python
frame_times = ft[frames]
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0)
```

iii. Raw-to-converted spot checks were reported as exact/all-close matches.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from per-trial boolean `isRew`.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
```

iii. The notes describe this as the native rewarded-corridor identity, including in conditions where actual water delivery may be absent.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is converted to float 0/1 and repeated over all retained frames of the trial.

ii.
```python
reward_availability = np.full(frames.size, float(is_rew[trial]), dtype=np.float32)
```

iii. The agent considers no additional transformation necessary because the requested variable is trial-constant reward availability.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It comes from per-trial `WallName` values collected across all canonical sessions.

ii.
```python
vocab.update(map(str, np.unique(np.asarray(beh["WallName"]))))
wall_name = np.asarray(beh["WallName"])
```

iii. The agent chose raw labels to avoid ambiguity between paper prose and released names and to preserve crops/swap variants.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Fifteen sorted raw wall names are assigned global integer IDs; the trial's ID is repeated over time. Variants are not collapsed into the four base categories.

ii.
```python
stim_vocab = gather_stimulus_vocabulary(sessions)
stim_to_idx = {stim: idx for idx, stim in enumerate(stim_vocab)}
stimulus_out = np.full(frames.size, stim_to_idx[str(wall_name[trial])], dtype=np.int16)
```

iii. The agent says preserving actual raw names is more faithful and enables time-varying output formatting through broadcasting.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from session-level lick frame indices `LickFr`.

ii.
```python
lick_frames = np.asarray(beh["LickFr"])
```

iii. The agent notes that `LickFr` uses the imaging-frame convention employed by the reference lick-aligned analyses.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Finite lick frame numbers are truncated to integers, restricted to `[0, len(ft))`, uniqued, and marked 1 in an otherwise-zero binary frame vector.

ii.
```python
lick_idx = lick_frames[finite].astype(np.int64, copy=False)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nframes)]
lick_binary[np.unique(lick_idx)] = 1
```

iii. The agent says this follows the reference's integer frame assignment and safely handles empty, non-finite, duplicate, or out-of-range lick records.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary frame vector is indexed by the same retained `frames` as the neural arrays.

ii.
```python
lick_out = lick_binary[frames].astype(np.int16, copy=False)
```

iii. The agent validated selected trials directly against raw `LickFr` and the neural-frame slice.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from frame-level `ft_Pos` at retained frames.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=np.float32)
pos_out = position_to_bin(ft_pos[frames])
```

iii. The agent reconciled raw units with the paper: ten native units equal one metre, and the textured corridor spans 0–40 units.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Values are clipped to the 0–40 corridor range, divided by ten, floored, and cast to integer categories.

ii.
```python
bins = np.floor(np.clip(pos, 0.0, 39.999999) / 10.0).astype(np.int16)
return np.clip(bins, 0, 3)
```

iii. This directly implements the requested four equal 1 m spatial bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Categories are `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40]` native units, labelled 0–1 m through 3–4 m.

ii.
```python
["0-1m", "1-2m", "2-3m", "3-4m"]
```

iii. The notes use the paper's 4 m texture length and 0.1 m native units to justify these boundaries.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is sampled at exactly the same retained frame indices as neural data.

ii.
```python
pos_out = position_to_bin(ft_pos[frames])
```

iii. The agent reports direct raw-value checks and plausible monotonic position progression in plots.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from frame-level `ft_RunSpeed` on frames selected by the corridor-running trial mask.

ii.
```python
ft_run_speed = np.asarray(beh["ft_RunSpeed"], dtype=np.float32)
speed_out = speed_to_bin(ft_run_speed[frames], speed_edges)
```

iii. The paper says speed is interpolated to imaging frames; this released frame-level field is therefore used directly.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A behavior-only pass gathers all finite retained speeds across the selected dataset, computes global 25th/50th/75th percentile edges, stabilizes tied edges with `nextafter`, and digitizes each value.

ii.
```python
edges = np.quantile(values, [0.25, 0.50, 0.75]).astype(np.float32)
...
return np.digitize(speed, speed_edges, right=False).astype(np.int16)
```

iii. The agent wanted consistent global thresholds and approximately equal-frequency output classes, and computed them without loading neural files.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It uses three global quantile value edges; equal values remain together, so exact quarter counts are not guaranteed despite globally near-equal reported totals.

ii.
```python
speed_edges = gather_speed_edges(sessions)
np.digitize(speed, speed_edges, right=False)
```

iii. The agent calls these global quartiles and documents their edges in metadata for reproducibility.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Both speed and neural activity use the same retained `frames` indices.

ii.
```python
speed_out = speed_to_bin(ft_run_speed[frames], speed_edges)
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0)
```

iii. Direct checks on six trials reportedly confirmed the converted category equals digitization of raw frame speed.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent validates neural-row/retinotopy length equality and unexpected area codes, ignores non-finite trial indices and licks, bounds lick indices, tolerates empty lick arrays, drops empty trials and sessions with fewer than two trials, and raises on absent speed samples. It does not explicitly truncate behavior arrays to the neural frame count and does not catch per-session conversion exceptions.

ii.
```python
if total_rows != brain_region_idx.shape[0]:
    raise ValueError(...)
finite = np.isfinite(ft_tr)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nframes)]
if frames.size == 0:
    continue
```

iii. The notes emphasize direct validation and a clean dataset. Defensive finite/range checks were added, while long time outliers were investigated and intentionally retained as genuine source behavior.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and trial-wise slicing/concatenating the very large `spks` arrays, followed by writing the 161.6 GiB pickle, dominate. The agent estimated roughly 12.65 seconds per large sample session and 17–20 minutes for full conversion core work.

ii.
```python
spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0)
pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes identify neural I/O/copying as the hot path and benchmarked trial-wise slicing as faster and less memory-intensive than whole-session concatenation.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial neural slicing/concatenation and per-trial speed-collection loops could be batched or grouped, though variable-length trial outputs still require per-trial objects. Session/index loops are small relative to neural I/O.

ii.
```python
for frames in groups:
    vals = ft_run_speed[frames]
    speeds.append(vals)
...
for trial, frames in enumerate(groups):
    neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0)
```

iii. The agent explicitly optimized trial grouping with split operations and avoided whole-session concatenation; it considered remaining neural loading/slicing the unavoidable dominant cost.

## 12-c. What processing does the code repeat multiple times?

i. Behavior is traversed in separate passes to build canonical sessions, collect stimulus vocabulary, collect speed values, choose samples, convert sessions, and optionally plot. `trial_frame_groups` is recomputed during the speed scan and conversion (and again for plots). Neural plane slices are concatenated anew for every trial.

ii.
```python
stim_vocab = gather_stimulus_vocabulary(sessions)
speed_edges = gather_speed_edges(sessions)
...
groups = trial_frame_groups(beh)
```

iii. The agent acknowledges behavior rereads as overhead and adds a two-file LRU cache; it accepts a behavior-only first pass because it avoids neural loading and provides global thresholds.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It scans all behavior twice for vocabulary/speed edges; stores extensive alias/session metadata; keeps and serializes neurons outside the four target visual regions; and, when requested, computes plots not used by the decoder. Per-session metadata includes fields such as aliases and conversion timing that training ignores.

ii.
```python
"aliases": list(session.aliases),
"exp_types": list(session.exp_types),
"conversion_seconds": time.time() - t0,
...
if plots_remaining > 0:
    plot_processing_summary(...)
```

iii. The agent views metadata and optional plots as validation/provenance aids and keeps all neural rows to preserve source information, despite the resulting much larger downstream dataset.
