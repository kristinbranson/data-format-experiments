# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads behavior by scanning every `Beh_*.npy` file under `data/beh`, collapses behavior aliases to a physical session base, separately loads `Imaging_Exp_info.npy` to recover registry metadata, and then iterates over the spike-file bases in `data/spk`. For each session it loads the matching retinotopy file and spike file on demand inside `convert_session`.

ii. 
```python
for path in sorted((DATA_ROOT / "beh").glob("Beh_*.npy")):
    loaded = np.load(path, allow_pickle=True).item()
```
```python
registry = np.load(DATA_ROOT / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
```
```python
spk_bases = {
    p.name.removesuffix("_neural_data.npy")
    for p in (DATA_ROOT / "spk").glob("*_neural_data.npy")
}
```

iii. In Step 2 and Step 4 of `CONVERSION_NOTES.md`, the agent says the registry has overlapping aliases but there are 89 physical recordings, so it collapses everything to unique `<mouse>_<date>_<block>` sessions and includes each one once.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the mouse prefix before the first underscore in the physical session base. `subjects` is the sorted unique set of those prefixes, and `subject_idx` maps each session to that subject list.

ii.
```python
subjects = sorted({base.split("_")[0] for base in bases})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
subject_idx = np.asarray([subject_lookup[base.split("_")[0]] for base in bases], dtype=np.int16)
```

iii. `CONVERSION_NOTES.md` Step 2 states that the dataset contains 19 mice and that collapsing to physical session bases preserves that mouse/session structure.

## 1-c. How are the data split into sessions?

i. A session is one physical recording identified by `<mouse>_<YYYY>_<MM>_<DD>_<block>`. Behavior aliases such as swap analyses are reduced to that base, and the session inventory is driven by the unique spike-file bases after cross-checking against behavior and registry catalogs.

ii.
```python
def physical_base(session_key: str) -> str:
    return re.sub(r"_swap[12]$", "", session_key)
```
```python
base = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
aliases[base].append(plain)
```
```python
if spk_bases != set(behaviors) or spk_bases != set(aliases):
    raise ValueError(...)
```

iii. In Step 4 and Step 5, the notes say the registry contains repeated labels for the same recording, so the converter must keep each physical recording exactly once.

## 1-d. How are the data split into trials?

i. Trials are defined from framewise trial labels, but only after frame curation. For each trial index from `0` to `ntrials - 1`, the agent takes the common-length frames where `ft_trInd == trial`, `ft_CorrSpc` is true, `ft_move > 0`, and the trial stamp is finite. A trial with no such curated frames is dropped.

ii.
```python
trial_stamp = np.asarray(beh["ft_trInd"][:common_nframes])
corridor = np.asarray(beh["ft_CorrSpc"][:common_nframes], dtype=bool)
movement = np.asarray(beh["ft_move"][:common_nframes]) > 0
valid = np.isfinite(trial_stamp) & corridor & movement
```
```python
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
    if frames.size == 0:
        excluded_trials.append(trial)
        continue
```

iii. Step 3 and Step 5 of the notes explicitly justify restricting decoder timepoints to the textured corridor and to the running-frame mask `ft_CorrSpc & (ft_move > 0)` to match the paper’s selectivity analyses and avoid stationary reward-consumption periods.

## 1-e. How are trials filtered based on quality controls?

i. The script does not apply the reference solution’s long-trial percentile filter. Instead it excludes only trials with zero surviving curated frames and fails the session if fewer than two trials remain.

ii.
```python
if frames.size == 0:
    excluded_trials.append(trial)
    continue
```
```python
if len(neural_trials) < 2:
    raise ValueError(f"{base} has fewer than two usable trials after curation")
```

iii. In Step 3 and Step 5, the notes say the running-frame mask should remove stopping periods, so the converter omits only trials with no remaining curated frame.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the plane-wise `spks` arrays in `spk/<base>_neural_data.npy`, and neuron region labels come from `iarea` in the matching retinotopy file.

ii.
```python
retino = np.load(retinotopy_path(base), allow_pickle=True)
iarea = np.asarray(retino["iarea"])
```
```python
neural_dict = np.load(path, allow_pickle=True).item()
planes = neural_dict["spks"]
```

iii. Step 1, Step 4, and Step 5 in the notes say the provided `spks` traces are already the Suite2p deconvolved signal used by the paper and that `iarea` should be used to group neurons into visual regions.

## 2-b. How is the `neural` data processed?

i. The agent does not compute dF/F or other signal transforms. It preallocates a matrix of only mapped neurons, copies the selected neurons plane by plane into float32, then extracts per-trial columns at the retained frame indices. The final trial matrices stay variable-length.

ii.
```python
selected = np.empty((int(mapped.sum()), nframes), dtype=np.float32)
...
selected[dst0 : dst0 + count] = plane[local_mask]
```
```python
neural = spk[:, frames].copy()
...
neural_trials.append(neural)
```

iii. The notes say the paper’s analyses use the supplied deconvolved `spks` directly, and Step 5 explicitly chooses float32 and native variable-length trials while applying running-frame curation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered only by retinotopy area membership. The script keeps `iarea` values in the grouped V1/mHV/lHV/aHV sets and excludes unmapped or outside labels.

ii.
```python
mapped = np.isin(iarea, [0, 1, 2, 3, 4, 5, 6, 8, 9])
codes = np.full(iarea.shape, -1, dtype=np.int8)
codes[iarea == 8] = 0
codes[np.isin(iarea, [0, 1, 2, 9])] = 1
codes[np.isin(iarea, [5, 6])] = 2
codes[np.isin(iarea, [3, 4])] = 3
return mapped, codes[mapped]
```

iii. Step 4 and Step 5 state that the paper’s relevant visual-region analyses use these grouped area IDs and that unknown/outside labels such as `-1` and `7` should be excluded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script treats entry into the 0–4 m textured corridor as the alignment event in metadata, but the actual neural frames retained for a trial are the curated moving textured frames from that trial. Time covariates are referenced to trial start, but neural columns begin at the first retained moving frame rather than necessarily the first corridor-entry frame.

ii.
```python
"temporal_alignment_event": "entry into the 0-4 m textured visual corridor",
```
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
neural = spk[:, frames].copy()
```

iii. Step 3 through Step 5 in the notes argue that trial timepoints should be limited to the textured corridor while running, but that temporal inputs should preserve elapsed time from true trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The nominal bin size is the native imaging frame interval, `1000 / 3.17` ms, and no explicit temporal rebinning or interpolation is applied to the neural signal. The script does, however, omit stationary frames via curation.

ii.
```python
FS_HZ = 3.17
MS_PER_FRAME = 1000.0 / FS_HZ
```
```python
"time_bin_size": MS_PER_FRAME,
"native_frame_rate_hz": FS_HZ,
```

iii. The notes say the paper used a 3.17 Hz imaging rate, so the converter keeps native frame bins and only changes which native frames are retained.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and the frame timestamps `ft` on the curated frames.

ii.
```python
ft = np.asarray(beh["ft"][:common_nframes], dtype=np.float64)
...
time_to_sound = (float(beh["SoundTime"][trial]) - ft[frames]) * DAY_TO_SECONDS
```

iii. In Step 5, the notes say the converter should use timestamp-derived temporal inputs so that elapsed-time gaps remain explicit even when stationary frames are removed.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame in a trial, the script subtracts the frame timestamp from the trial’s cue timestamp and converts days to seconds. The value is positive before the cue and negative after it.

ii.
```python
time_to_sound = (float(beh["SoundTime"][trial]) - ft[frames]) * DAY_TO_SECONDS
```
```python
"time_to_sound_sign": "positive before cue, zero at cue, negative after cue",
```

iii. Step 5 says the temporal inputs should be derived from original timestamps rather than from curated frame spacing, specifically so pauses removed by curation still appear as elapsed-time gaps.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The cue-time row is computed on exactly the same retained frame indices used to slice the neural trial matrix.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
neural = spk[:, frames].copy()
time_to_sound = (float(beh["SoundTime"][trial]) - ft[frames]) * DAY_TO_SECONDS
```

iii. The notes repeatedly describe all per-frame decoder variables as being derived from the same curated frame set so that each input timepoint matches a neural column.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session-level metadata in `Imaging_Exp_info.npy`: first from any explicit `days` field, otherwise from the minimum alias-specific `sess#` value attached to that physical session.

ii.
```python
explicit_days = [e["days"] for e in entries if "days" in e]
session_indices = [e["sess#"] for e in entries if "sess#" in e]
if explicit_days:
    training_day[base] = float(min(explicit_days))
elif session_indices:
    training_day[base] = float(min(session_indices))
```

iii. Step 4 and Step 5 justify this as a way to avoid alias-specific conflicts and to use the most direct stage/day information the registry exposes.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The converter resolves one scalar day value per physical session using the rule above, then broadcasts that constant across every retained frame of each trial in the session.

ii.
```python
decoder_input = np.vstack(
    [
        time_to_sound,
        np.full(frames.size, day, dtype=np.float64),
        time_since_start,
        np.full(frames.size, float(bool(beh["isRew"][trial])), dtype=np.float64),
    ]
).astype(np.float32)
```

iii. The Step 5 notes describe the training-day rule explicitly and say the result should be a continuous session-level covariate repeated over trial timepoints.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from per-trial `Trial_start_time` and the retained-frame timestamps `ft`.

ii.
```python
time_since_start = (ft[frames] - float(beh["Trial_start_time"][trial])) * DAY_TO_SECONDS
```

iii. Step 5 says elapsed trial time should come from original timestamps so removed stationary frames do not falsely compress time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame in a trial, the script subtracts the trial’s start timestamp from that frame’s timestamp and converts the result from days to seconds.

ii.
```python
time_since_start = (ft[frames] - float(beh["Trial_start_time"][trial])) * DAY_TO_SECONDS
```

iii. The notes explicitly say temporal inputs should preserve “real gaps” in elapsed time despite running-frame curation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same retained frame indices as the neural data for that trial, so the time series length matches the neural columns exactly.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
neural = spk[:, frames].copy()
time_since_start = (ft[frames] - float(beh["Trial_start_time"][trial])) * DAY_TO_SECONDS
```

iii. Step 5 says the decoder input arrays are `(4, T)` arrays defined on the same curated native frames as the neural trials.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the per-trial boolean `isRew`.

ii.
```python
np.full(frames.size, float(bool(beh["isRew"][trial])), dtype=np.float64)
```

iii. Step 4 and Step 5 say reward availability should be based on `isRew` rather than on session-mode strings because unsupervised files can still carry active-mode labels.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is converted to a numeric `0.0` or `1.0` and broadcast across every retained frame of the trial.

ii.
```python
np.full(frames.size, float(bool(beh["isRew"][trial])), dtype=np.float64)
```

iii. The Step 5 notes explicitly describe reward availability as a per-trial covariate repeated across all timepoints.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, one stimulus label per trial.

ii.
```python
category = stimulus_category(str(beh["WallName"][trial]))
```

iii. Step 4 and Step 5 say the native `WallName` is the cleanest source because alias views of `stim_id` can be ambiguous in swap sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script extracts the alphabetic prefix of the `WallName` string, lowercases it, maps it to one of `circle`, `leaf`, `rock`, or `wood`, and then broadcasts that category code across the retained frames of the trial.

ii.
```python
match = re.match(r"([A-Za-z]+)", str(name))
return CATEGORY_TO_CODE[match.group(1).lower()]
```
```python
decoder_output[0] = category
```

iii. Step 4 and Step 5 justify pooling exemplar numbers and swap suffixes into the four physical texture categories described in the paper.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, the frame indices of lick events.

ii.
```python
lick_float = np.asarray(beh["LickFr"], dtype=np.float64)
lick_frames = lick_float[np.isfinite(lick_float)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < common_nframes)]
lick_global[lick_frames] = 1
```

iii. The notes say licking is already supplied as event times on the imaging-frame grid, so it can be represented as a binary per-frame series without interpolation.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The converter filters lick indices to finite in-range frames, truncates them to integers, and marks a binary session-level lick vector with `1` wherever at least one lick falls on that frame. Trial outputs then read from that vector.

ii.
```python
lick_global = np.zeros(common_nframes, dtype=np.int8)
...
lick_global[lick_frames] = 1
```
```python
decoder_output[1] = lick_global[frames]
```

iii. Step 5 says the desired output is a categorical binary lick flag at each neural frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The per-trial lick output is `lick_global[frames]`, using the same retained frame indices as the neural matrix.

ii.
```python
neural = spk[:, frames].copy()
decoder_output[1] = lick_global[frames]
```

iii. The notes describe `LickFr` as already being frame-aligned and say every decoder output row should use the same curated frame set as the neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the framewise corridor position `ft_Pos`.

ii.
```python
position = np.asarray(beh["ft_Pos"][:common_nframes])[frames]
```

iii. Step 3 and Step 5 say the decoder trial should cover the 0–4 m textured corridor, for which `ft_Pos` already gives framewise position.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The retained-frame positions are interpreted in decimeters, divided by 10, floored, clipped into `0..3`, and stored as the four 1 m position classes.

ii.
```python
position_class = np.clip(np.floor(position / 10.0), 0, 3).astype(np.int8)
decoder_output[2] = position_class
```

iii. Step 5 says the task requires four equal 1 m spatial bins over the 0–4 m textured corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Category `0` is `0–1 m`, `1` is `1–2 m`, `2` is `2–3 m`, and `3` is `3–4 m`, implemented by `floor(position / 10)` on decimeter units and clipping to `0..3`.

ii.
```python
position_class = np.clip(np.floor(position / 10.0), 0, 3).astype(np.int8)
```
```python
"output_values": [
    ...,
    ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
    ...,
],
```

iii. Step 5 explicitly records the source units as decimeters and the target categories as four 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is read from `ft_Pos` only at the same retained frame indices used for the neural matrix of that trial.

ii.
```python
neural = spk[:, frames].copy()
position = np.asarray(beh["ft_Pos"][:common_nframes])[frames]
decoder_output[2] = position_class
```

iii. The notes say position is a time-varying output on the same curated native-frame grid as neural activity.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise `ft_RunSpeed`.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"][:common_nframes], dtype=np.float32)[frames]
speed_trials.append(speed)
```

iii. Step 4 and Step 5 say running speed should be discretized from `ft_RunSpeed` on the curated frame set.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script first stores the continuous speed vector for every retained trial, then after all sessions are converted it computes one global set of quartile thresholds across all retained speed samples and fills each trial’s fourth output row from those thresholds.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"][:common_nframes], dtype=np.float32)[frames]
speed_trials.append(speed)
```
```python
all_speed = np.concatenate([trial for session in speeds for trial in session]).astype(np.float64)
thresholds = np.quantile(all_speed, [0.25, 0.50, 0.75])
```

iii. Step 5 and the later runtime notes justify a global quartile definition so the class meanings are comparable across sessions while still avoiding a second neural-data pass.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global thresholds are computed at the 25th, 50th, and 75th percentiles of all retained speed samples. Each retained frame is labeled with `np.searchsorted(..., side="right")`, yielding classes `0..3`.

ii.
```python
thresholds = np.quantile(all_speed, [0.25, 0.50, 0.75])
labels = np.searchsorted(thresholds, speed_trial, side="right").astype(np.int8)
out_trial[3] = labels
```

iii. The notes say a single physical set of thresholds should define quartiles dataset-wide.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The continuous speed vectors are first sliced on the same retained frames as the neural data, and the final quartile labels are written back into the corresponding trial output arrays with the same length.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"][:common_nframes], dtype=np.float32)[frames]
...
for out_trial, speed_trial in zip(out_session, speed_session, strict=True):
    labels = np.searchsorted(thresholds, speed_trial, side="right").astype(np.int8)
    out_trial[3] = labels
```

iii. Step 5 says all outputs should be time-varying on the same curated frame grid as the neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script truncates all frame-aligned streams to a common neural/behavior length, drops non-finite trial labels and out-of-range lick frames, and raises errors on structural inconsistencies such as alias conflicts, retinotopy/neural mismatches, non-finite neural or input values, or sessions with fewer than two usable trials.

ii.
```python
common_nframes = min([neural_nframes] + [len(beh[k]) for k in frame_fields])
valid = np.isfinite(trial_stamp) & corridor & movement
```
```python
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < common_nframes)]
```
```python
if not np.all(np.isfinite(neural)):
    raise ValueError(...)
...
if len(neural_trials) < 2:
    raise ValueError(...)
```

iii. Step 4 through Step 6 say the converter should “fail loudly” on unexpected corruption but handle expected edge cases such as frame-count mismatches by truncation to common length.

## 12-a. What are the most time-consuming steps of the code?

i. The main expensive step is reading the large spike matrices. The agent also treats writing the large pickle as a meaningful cost and explicitly optimized the conversion to avoid rereading neural files.

ii.
```python
neural_dict = np.load(path, allow_pickle=True).item()
planes = neural_dict["spks"]
```
```python
with args.outpicklefile.open("wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 6 and Step 7 describe the full spike corpus as roughly 404 GiB and say the single-pass design was chosen to avoid a second neural-data read; the runtime table separately calls out pickle writing time.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious non-vectorized loop is the per-trial search over all frames using `np.flatnonzero(valid & (trial_stamp == trial))`. The later nested loop that fills speed classes trial by trial is another candidate for batching.

ii.
```python
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
```
```python
for out_session, speed_session in zip(outputs, speeds, strict=True):
    for out_trial, speed_trial in zip(out_session, speed_session, strict=True):
        labels = np.searchsorted(thresholds, speed_trial, side="right").astype(np.int8)
```

iii. The notes focus more on I/O avoidance than on vectorizing these loops, implying the agent regarded neural-file I/O as the dominant bottleneck.

## 12-c. What processing does the code repeat multiple times?

i. Inside the per-trial loop, the code repeatedly scans `valid & (trial_stamp == trial)` to find trial frames and repeatedly rebuilds sliced arrays from `ft_Pos` and `ft_RunSpeed`. It also performs a second pass over every trial later to convert stored continuous speeds into quartile labels.

ii.
```python
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
    ...
    position = np.asarray(beh["ft_Pos"][:common_nframes])[frames]
    speed = np.asarray(beh["ft_RunSpeed"][:common_nframes], dtype=np.float32)[frames]
```
```python
thresholds, speed_counts = fill_speed_classes(output_all, speed_all)
```

iii. Step 6 justifies the second pass as cheaper than rereading neural files, but the code still repeats some behavior slicing and trial scans.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter stores `speed_trials` as full continuous speed vectors only to derive quartile labels later; those continuous vectors are not saved in the final dataset. Optional `plot_context` data and the `source_ranges` return value from `load_selected_spikes` are also temporary and discarded after diagnostics/conversion.

ii.
```python
speed_trials: list[np.ndarray] = []
...
speed_trials.append(speed)
```
```python
return selected, nframes, source_ranges
```
```python
plot_context = {
    "raw_neural": spk[: min(spk.shape[0], 24), :raw_n].copy(),
    ...
}
```

iii. The notes acknowledge these temporaries as part of the single-pass strategy and optional processing diagnostics rather than part of the final saved decoder dataset.
