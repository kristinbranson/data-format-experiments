# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `Imaging_Exp_info.npy` as the master index, maps each unique mouse/date/block recording to the first behavior copy, loads each required `Beh_<experiment_type>.npy` once, and loads the spike and retinotopy files once per session. It checks that indexed sessions and spike files match.

ii.
```python
exp_info = np.load(data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
sources = make_source_map(exp_info, data_root / "spk")
behaviors = load_behaviors(data_root, sources)
raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.load(data_root / "retinotopy" / retino_name)["iarea"]
```

iii. The trajectory says behavior and retinotopy are separate from session-wide neural arrays, and that behavior files contain several sessions. It chose one canonical copy because repeated behavior copies differ only in analysis labels, and verified that all 89 neural recordings were represented.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `mname` values in the experiment table; each session receives the corresponding integer `subject_idx`.

ii.
```python
subjects = sorted({sources[sid]["record"]["mname"] for sid in session_ids})
subject_to_id = {name: idx for idx, name in enumerate(subjects)}
subject_idx.append(subject_to_id[str(record["mname"])])
```

iii. The agent treated `mname` as the explicit mouse identifier and validation reported 19 mice.

## 1-c. How are the data split into sessions?

i. A session ID is `mname_datexp_blk`. Duplicate appearances across experiment types are collapsed by keeping the first occurrence, and the 89 unique IDs are sorted for conversion.

ii.
```python
def session_id(record: dict) -> str:
    return f"{record['mname']}_{record['datexp']}_{record['blk']}"

sources.setdefault(sid, {"experiment_type": experiment_type,
                         "behavior_key": behavior_key(record), "record": record})
session_ids = sorted(sources)
```

iii. The trajectory established that the table contains repeated analysis-specific entries for the same physical recording, so it selected the first repository-table occurrence deterministically.

## 1-d. How are the data split into trials?

i. Frames are first restricted to valid trial indices, the textured corridor, and periods with `ft_move > 0`. The retained frame-level `ft_trInd` values define trials; each unique retained ID becomes one variable-length trial. Thus stationary frames in the middle of a trial are removed.

ii.
```python
return (valid_trial & beh["ft_CorrSpc"][:nframes].astype(bool)
        & (beh["ft_move"][:nframes] > 0))
trial_ids = beh["ft_trInd"][: len(frame_mask)][frame_mask].astype(np.int64)
for trial in np.unique(trial_ids):
    columns = np.flatnonzero(trial_ids == trial)
```

iii. The agent stated that this reproduces `code/utils.py::Get_dprime_selective_neuron` and chose original imaging frames rather than the paper's spatial interpolation because the requested decoder is time-aligned. It explicitly accepted timestamp gaps caused by removed pauses.

## 1-e. How are trials filtered based on quality controls?

i. A trial is retained if it has at least one frame satisfying valid-index, corridor, and movement filters. No whole-trial duration/outlier filter is applied; long stopped intervals disappear because their stationary frames are removed. The code aborts a session if fewer than two usable trials remain.

ii.
```python
frame_indices = np.flatnonzero(frame_mask)
kept_trials = np.unique(trial_ids)
if len(session_neural) < 2:
    raise RuntimeError(f"Fewer than two usable trials in {sid}")
```

iii. The trajectory justified the frame filter as the paper code's critical running-within-corridor mask. It found all 38,110 trials retained and did not identify or remove the pathological long traversals as whole trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the per-plane `spks` arrays in each `<session>_neural_data.npy`; neuron-region filtering and labels come from retinotopy `iarea`.

ii.
```python
raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.load(data_root / "retinotopy" / retino_name)["iarea"]
selected_neural = np.concatenate(selected_planes, axis=0)
```

iii. The agent inspected the files and methods and concluded that `spks` already contains Suite2p non-negative deconvolved fluorescence, so it used it directly.

## 2-b. How is the `neural` data processed?

i. The agent applies no smoothing, normalization, deconvolution, temporal rebinning, or spatial interpolation. It selects retained visual-area rows and valid running-corridor columns plane by plane, concatenates planes, and makes a contiguous float32 array for every trial.

ii.
```python
selected_planes.append(plane[np.ix_(local_keep, frame_indices)])
selected_neural = np.concatenate(selected_planes, axis=0)
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
```

iii. It reasoned that the source is already deconvolved and that spatial interpolation belonged to position-aligned paper analyses, not this temporally aligned decoder. It kept the source float32 dtype.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are retained only for `iarea` codes assigned to V1, medial, anterior, or lateral visual-area groups; codes -1 and 7 are excluded. Temporally, only valid moving frames inside the textured corridor are retained. Shape and dtype mismatches cause errors.

ii.
```python
AREA_CODES = {"V1": (8,), "medial": (0, 1, 2, 9),
              "anterior": (3, 4), "lateral": (5, 6)}
keep_neuron, regions = area_labels(iarea)
frame_mask = valid_frame_mask(beh, nframes)
```

iii. The trajectory traced the region groups and running mask to the paper repository. It assumed Suite2p had already curated cells and added consistency checks rather than further cell-quality filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are described as aligned to entry into the 4 m textured corridor. They are variable length with no padding or fixed end, but only moving corridor frames are retained, so pauses create temporal gaps even though the arrays concatenate the remaining samples.

ii.
```python
frames = frame_indices[columns]
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
"temporal_alignment_event": "entry into the 4 m textured corridor",
"off_start": 0.0,
"off_end": None,
```

iii. The agent said temporal alignment should preserve original imaging samples rather than spatially interpolate, while the decoder can accept variable-length trials. It considered the running-frame mask mandated by the paper code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Each retained imaging frame is one time bin at nominally 3.17 Hz, recorded as about 315.46 ms. No temporal rebinning or resampling is applied, although stationary frames are removed.

ii.
```python
FRAME_RATE_HZ = 3.17
"time_bin_size": 1000.0 / FRAME_RATE_HZ,
```

iii. The agent reasoned that imaging frames are the native temporal resolution and behavioral streams are already on that frame grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and the timestamps `ft` at retained frames.

ii.
```python
times = beh["ft"][frames]
time_to_sound = ((beh["SoundTime"][trial] - times) * SECONDS_PER_DAY)
```

iii. The agent inspected both frame and timestamp fields and chose the directly recorded event time, with MATLAB day units converted to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every retained frame, its timestamp is subtracted from the trial's sound timestamp and multiplied by 86,400; values are positive before and negative after the cue, then cast to float32.

ii.
```python
time_to_sound = (
    (beh["SoundTime"][trial] - times) * SECONDS_PER_DAY
).astype(np.float32)
```

iii. The script and metadata explicitly justify the sign as “time to” the cue and the factor as conversion from days to seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses `times = ft[frames]` for exactly the same retained `frames` as the neural columns, producing one value per neural sample.

ii.
```python
frames = frame_indices[columns]
times = beh["ft"][frames]
trial_input[0] = time_to_sound
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
```

iii. The agent relied on the behavior streams and neural data sharing imaging-frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It uses `mname` and the calendar date `datexp` from the experiment index, not the sparsely present `days` field.

ii.
```python
date = datetime.strptime(record["datexp"], "%Y_%m_%d").date()
subject = str(record["mname"])
```

iii. The agent found that `days` is only present for selected recordings and considered calendar date the only uniform, non-imputed measure across all groups.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest imaging date is day 0; each session gets the elapsed number of calendar days from that date, broadcast across all retained bins of every trial.

ii.
```python
first_date[subject] = min(date, first_date.get(subject, date))
day_by_session[sid] = float((date - first_date[subject]).days)
trial_input[1] = day_by_session[sid]
```

iii. The trajectory notes that recording/training days are incomplete and nonconsecutive; the agent deliberately chose elapsed calendar days as a uniform measure.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from per-trial `Trial_start_time` and retained-frame timestamps `ft`.

ii.
```python
times = beh["ft"][frames]
time_from_start = ((times - beh["Trial_start_time"][trial]) * SECONDS_PER_DAY)
```

iii. The agent used the directly stored trial-start time rather than converting `StartFr` to a timestamp.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The trial start timestamp is subtracted from every retained frame timestamp, converted from days to seconds, and cast to float32.

ii.
```python
time_from_start = (
    (times - beh["Trial_start_time"][trial]) * SECONDS_PER_DAY
).astype(np.float32)
```

iii. The agent treated this as the direct elapsed-time definition and verified timestamp/frame-boundary agreement during exploration.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is evaluated at the same retained frame indices as neural activity, including the same removal of stationary frames.

ii.
```python
frames = frame_indices[columns]
times = beh["ft"][frames]
trial_input[2] = time_from_start
```

iii. The agent relied on common imaging-frame indices for all streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from the trial-level `isRew` value.

ii.
```python
trial_input[3] = float(beh["isRew"][trial])
```

iii. The field directly represents whether the corridor/trial is rewarded, so no inferred reward rule was needed.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The scalar is converted to float and broadcast across all retained time bins in the preallocated trial input row.

ii.
```python
trial_input = np.empty((4, ntime), dtype=np.float32)
trial_input[3] = float(beh["isRew"][trial])
```

iii. The agent treated it as an already categorical per-trial variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from each trial's `WallName` across the loaded behavior sessions.

ii.
```python
stimulus_values = sorted({str(wall) for beh in behaviors.values()
                          for wall in beh["WallName"]})
trial_output[0] = stimulus_to_id[str(beh["WallName"][trial])]
```

iii. The agent used actual `WallName` rather than analysis labels such as masked or swapped stimulus IDs.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Every distinct wall-name string is sorted into a global vocabulary and assigned an integer. This preserves 15 wall variants rather than collapsing them into the four broad categories circle, leaf, rock, and wood. The trial scalar is broadcast over time.

ii.
```python
stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
trial_output[0] = stimulus_to_id[str(beh["WallName"][trial])]
```

iii. The code provides no explicit discussion of broad-category collapsing; it chose a data-derived vocabulary to cover all recorded wall names.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It comes from session-level lick frame numbers in `LickFr`.

ii.
```python
lick_frame = beh["LickFr"]
lick_frame = lick_frame[np.isfinite(lick_frame)].astype(np.int64)
```

iii. The agent identified `LickFr` as already expressed on the neural/imaging frame index.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Nonfinite lick entries are removed and remaining values are truncated to integers. Each retained trial frame is labeled 1 if its frame number occurs in that session lick list, otherwise 0.

ii.
```python
lick_frame = lick_frame[np.isfinite(lick_frame)].astype(np.int64)
trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. The agent treated licking as a binary per-frame event and used membership to handle one or more licks in a frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Membership is tested for the exact `frames` corresponding to the neural columns, so the lick vector has the same length and the same running-frame omissions.

ii.
```python
frames = frame_indices[columns]
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. Alignment is by shared imaging-frame number.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position comes from frame-level `ft_Pos`; the denominator is derived from the session's `Texture_Length` (stored as decimeters).

ii.
```python
position_bin = np.floor(
    beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0)
).astype(np.int64)
```

iii. The agent inspected the 40-decimeter texture length and used the recorded setting to keep bins at one quarter of the corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by one quarter of the texture length, floored to an integer, and clipped to the range 0–3.

ii.
```python
position_bin = np.floor(beh["ft_Pos"][frames] /
                        (beh["texture_length_dm"] / 4.0)).astype(np.int64)
position_bin = np.clip(position_bin, 0, 3)
```

iii. This implements the requested four equal-length spatial bins and remains valid if a source session's configured corridor length differs.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. With the observed 40 dm corridor, thresholds are 10, 20, and 30 dm, yielding labels `0-1 m`, `1-2 m`, `2-3 m`, and `3-4 m`; endpoints outside the nominal range are clipped.

ii.
```python
"output_values": [..., ["0-1 m", "1-2 m", "2-3 m", "3-4 m"], ...]
position_bin = np.clip(position_bin, 0, 3)
```

iii. The agent's comment says `Texture_Length` is 40 dm, so quartering it makes exactly the requested 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed by the same retained `frames` used for each neural trial.

ii.
```python
frames = frame_indices[columns]
position_bin = np.floor(beh["ft_Pos"][frames] / ...).astype(np.int64)
trial_output[2] = position_bin
```

iii. The agent used the common frame grid and applied its movement filter consistently to both streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed` at all globally retained valid moving-corridor frames.

ii.
```python
mask = valid_frame_mask(beh, len(beh["ft"]))
speeds.append(beh["ft_RunSpeed"][: len(mask)][mask].astype(np.float32))
```

iii. The agent considered `ft_RunSpeed` the direct behavioral stream and matched its sample selection to the converted data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speeds from all sessions are concatenated, global 25th/50th/75th percentile value thresholds are computed, and every trial speed is assigned with `np.digitize`.

ii.
```python
all_speeds = np.concatenate(speeds)
edges = np.quantile(all_speeds, (0.25, 0.50, 0.75))
speed_bin = np.digitize(beh["ft_RunSpeed"][frames], speed_edges,
                        right=False).astype(np.int64)
```

iii. The agent interpreted “four bins, each corresponding to 25% of the data” as dataset-wide quartile thresholds and checks that the three edges are distinct.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Values below the global 25th percentile are category 0; values from the 25th to below the median are 1; median to below the 75th percentile are 2; and values at or above the 75th percentile are 3 (`right=False`). Ties can make category counts unequal.

ii.
```python
speed_bin = np.digitize(beh["ft_RunSpeed"][frames], speed_edges,
                        right=False).astype(np.int64)
```

iii. The output labels call these lowest 25%, 25–50%, 50–75%, and highest 25%; no tie-ranking step is used.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The speed values indexed by the same retained `frames` as neural activity are digitized, yielding one speed label per neural column.

ii.
```python
frames = frame_indices[columns]
speed_bin = np.digitize(beh["ft_RunSpeed"][frames], speed_edges, right=False)
trial_output[3] = speed_bin
```

iii. The agent used the common imaging-frame index for alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent truncates usable frames to the minimum neural-plane length and relevant behavior-stream lengths, rejects invalid/nonfinite trial IDs, removes nonfinite lick entries, and tolerates lick indices outside imaging by membership testing. It performs strict session/file, dtype, neuron-count, chronology, nonempty-frame, quartile-edge, and minimum-trial checks; violations generally raise errors.

ii.
```python
nframes = min(plane.shape[1] for plane in raw_planes)
nframes = min(nframes, len(beh["ft"]), len(beh["ft_trInd"]),
              len(beh["ft_CorrSpc"]), len(beh["ft_move"]))
valid_trial = np.isfinite(trial) & (trial >= 0) & (trial < beh["ntrials"])
lick_frame = lick_frame[np.isfinite(lick_frame)].astype(np.int64)
```

iii. The trajectory explored dimensions and mismatches before conversion. The agent favored explicit validation so silent inconsistencies would not corrupt the 142 GB output.

## 12-a. What are the most time-consuming steps of the code?

i. Reading hundreds of gigabytes of per-session spike files, copying selected neural data plane by plane, constructing per-trial contiguous arrays, writing the 142 GB pickle, and the subsequent 200-epoch decoder run dominate. The script itself does not time individual phases.

ii.
```python
raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
selected_planes.append(plane[np.ix_(local_keep, frame_indices)])
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory measured about 405 GB of source spikes, designed plane-wise selection for memory, and spent most execution time converting/serializing and then training. The final float32 dataset was 142 GB.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial `np.flatnonzero(trial_ids == trial)` rescans all retained session frames for every trial and could be replaced by one grouping/split pass. Per-trial lick membership and array construction could also be based on a precomputed session-level lick flag. Small loops over region groups and sessions are not meaningful bottlenecks.

ii.
```python
for trial in kept_trials:
    columns = np.flatnonzero(trial_ids == trial)
    ...
    trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. The trajectory did not explicitly discuss these vectorization opportunities; its optimization effort focused on avoiding concatenation of unfiltered multi-gigabyte recordings.

## 12-c. What processing does the code repeat multiple times?

i. `valid_frame_mask` is computed once globally for speed edges and again per session during conversion. Every trial rescans `trial_ids`, and `np.isin(frames, lick_frame)` repeatedly scans the session lick list. Data are also copied into a session selection and then copied again into contiguous per-trial arrays.

ii.
```python
for beh in behaviors.values():
    mask = valid_frame_mask(beh, len(beh["ft"]))
...
frame_mask = valid_frame_mask(beh, nframes)
columns = np.flatnonzero(trial_ids == trial)
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
```

iii. The trajectory does not justify the repeated masks/scans directly. The second valid mask is necessary to respect actual neural length; the extra trial copies are justified in code as preventing views from retaining a whole-session backing allocation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and copies all required behavior sessions into a compact in-memory dictionary, globally scans them for vocabulary/speed edges, and records extensive metadata used for validation rather than decoder features. More importantly, float32 neural values are preserved even though the reference safely stores float16, greatly increasing copying, serialization, storage, and downstream I/O. The strict validation scans are not decoder inputs but are defensible safeguards.

ii.
```python
behaviors[sid] = compact_behavior(behavior_file[key])
stimulus_values = visual_vocabulary(behaviors)
speed_edges = global_speed_edges(behaviors)
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
"session_info": session_info,
```

iii. The agent deliberately retained float32 because it verified that source planes were float32, but gave no precision-based need for doing so. It justified compact behavior copies and contiguous trial copies as memory/lifetime controls, and validation metadata as reproducibility information.
