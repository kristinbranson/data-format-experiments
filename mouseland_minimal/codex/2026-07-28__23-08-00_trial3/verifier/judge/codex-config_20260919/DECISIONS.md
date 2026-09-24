# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `data/beh/Imaging_Exp_info.npy` as the master recording list, scans and loads every `Beh_*.npy` file into a canonical behavior lookup, and loads each session's neural and retinotopy files during conversion. It converts 89 unique recordings.

ii.
```python
exp_info = load_exp_info(args.root)
sessions = collect_unique_sessions(exp_info)
canonical_lookup, provenance, duplicate_mismatches = build_behavior_lookup(args.root)
spk_obj = np.load(spk_path, allow_pickle=True).item()
ret = np.load(ret_path, allow_pickle=True)
```

iii. The trajectory says the behavior files are figure-specific and duplicate sessions, so the agent built a canonical lookup and deduplicated them to the paper's 89 recordings and 19 mice.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname`. A first-seen ordered subject list is made, and each session receives the corresponding integer `subject_idx`.

ii.
```python
if session["mname"] not in subject_to_idx:
    subject_to_idx[session["mname"]] = len(subjects)
    subjects.append(session["mname"])
subject_idx = np.array([subject_to_idx[s["mname"]] for s in sessions], dtype=np.int64)
```

iii. The agent relied on the explicit mouse identifier in the experiment index and checked that this yields the reported 19 mice.

## 1-c. How are the data split into sessions?

i. A session is the unique tuple `(mname, datexp, blk)`. Repeated appearances across experiment categories are discarded after the first.

ii.
```python
key = (item["mname"], item["datexp"], item["blk"])
if key in seen:
    continue
seen.add(key)
```

iii. The agent found that the same recordings occur in multiple behavior files/categories and deduplicated them to match the paper's 89 recordings.

## 1-d. How are the data split into trials?

i. For each integer trial from `0` to `ntrials-1`, frames are selected where `ft_trInd == tr`, `ft_CorrSpc` is true, and `ft_move > 0`. Thus a trial contains only moving corridor frames, not every contiguous corridor frame.

ii.
```python
for tr in range(ntrials):
    mask = (ft_trind == tr) & is_corr & is_move
    out.append(np.flatnonzero(mask).astype(np.int64))
```

iii. The agent stated that it was following the paper/code frame-level fields and chose `ft_CorrSpc && ft_move > 0` to retain running periods for decoder-ready data.

## 1-e. How are trials filtered based on quality controls?

i. Trials with no selected moving corridor frames are skipped. There is no long-trial/outlier filter; stationary frames are removed within every trial instead. All 38,110 trials happened to remain in the produced dataset.

ii.
```python
chunks = chunk_indices(frame_idx, frames_per_bin)
if not chunks:
    continue
```

iii. The trajectory emphasizes retaining running-only corridor samples. It does not document investigation or rejection of abnormally long stopped trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the per-plane `spks` arrays in each session's neural-data file. `iarea` from the retinotopy file supplies region membership used for filtering and metadata.

ii.
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
iarea = np.asarray(ret["iarea"], dtype=np.int64)
```

iii. The agent treated `spks` as the paper's already deconvolved neural signal and used retinotopy to restrict it to named visual regions.

## 2-b. How is the `neural` data processed?

i. Visual-cortex neurons are ranked by variance on up to 2,048 selected frames and at most 128 are retained in proportions reflecting the four regions. Neural activity is then averaged over successive chunks of three retained frames and stored as float32.

ii.
```python
selected_neurons = select_neurons(spk, region_idx_full, selected_frames, neurons_per_session)
spk_sel = spk[selected_neurons].astype(np.float32, copy=False)
neural_bin = spk_sel[:, chunk].mean(axis=1)
```

iii. The agent described this as a memory/compute-conscious decoder representation: 128 visual-cortex neurons and three-frame bins. It also sought region representation and high-variance units.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons outside mapped V1/mHV/lHV/aHV are excluded. Among mapped neurons, only up to 128 are kept, allocated proportionally by region and chosen primarily by highest temporal variance.

ii.
```python
region_counts = [(region_idx_full == ridx).sum() for ridx in range(len(REGION_NAMES))]
targets = proportional_region_targets(region_counts, neurons_per_session)
order = np.argsort(var[candidates])[::-1]
picked = candidates[order[:target]]
```

iii. The trajectory reports this deliberate neuron selection as part of reducing the full conversion to a tractable decoder dataset; no paper-based quality defect motivated the 128-neuron cap.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bins are ordered from selected frames within each trial, beginning at the first moving frame in the corridor. Trials remain variable length and are not padded. Because nonmoving frames are removed, the first bin need not be the actual corridor-entry frame and elapsed time between adjacent bins can be discontinuous.

ii.
```python
mask = (ft_trind == tr) & is_corr & is_move
chunks = chunk_indices(frame_idx, frames_per_bin)
neural_trials.append(np.stack(trial_neural, axis=1).astype(np.float32))
```

iii. The agent labeled the alignment event “corridor entry / trial start” and viewed grouping ordered in-trial corridor-running frames as sufficient alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The median imaging-frame interval is computed across sessions and multiplied by the default three frames per bin, yielding roughly 946 ms. Each bin averages three retained frames; the final partial chunk is also retained even if it contains fewer than three frames.

ii.
```python
frame_dt_ms = compute_frame_dt_ms(canonical_lookup, sessions)
"time_bin_size": float(frame_dt_ms * args.frames_per_bin)
neural_bin = spk_sel[:, chunk].mean(axis=1)
```

iii. The agent explicitly chose three-frame averaging to reduce dataset size and decoder cost.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived directly from per-trial `SoundTime` and per-frame timestamp `ft`.

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
```

iii. The agent considered these timestamp variables already on the same MATLAB-day time base, avoiding reconstruction from `SoundFr`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each chunk, it subtracts each frame timestamp from the trial's sound timestamp, converts days to seconds, and averages the resulting signed times. Values are positive before and negative after the cue.

ii.
```python
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The agent aimed to preserve a continuous time-to-event covariate while matching the three-frame decoder bins.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from exactly the same frame chunk whose neural columns are averaged, producing one value for every neural bin.

ii.
```python
chunk_ft = ft[chunk]
neural_bin = spk_sel[:, chunk].mean(axis=1)
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The agent consistently co-binned every time-varying stream with the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session's `datexp`, grouped by `mname`.

ii.
```python
by_subject[session["mname"]].append((idx, parse_date(session["datexp"])))
```

iii. The agent interpreted “day of training” as elapsed calendar days for each mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Dates are parsed, the earliest imaging date for a mouse is assigned day 0, and later sessions use integer calendar-day differences. The value is repeated across all bins of every trial in that session.

ii.
```python
first_date = min(date for _, date in entries)
offsets[idx] = float((date - first_date).days)
day_value = np.float32(training_days[session_idx])
```

iii. The agent chose a simple mouse-relative calendar-day measure and documented it in metadata.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from frame timestamps `ft` and the per-trial raw timestamp `Trial_start_time`.

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
```

iii. The agent used the direct timestamp fields rather than converting `StartFr` to time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. It subtracts the trial-start timestamp from each retained frame timestamp, converts days to seconds, and averages within the three-frame chunk.

ii.
```python
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The agent intended a continuous elapsed-time feature on the same binned grid as neural activity.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the exact chunk of frame indices averaged for the corresponding neural bin. Since stopped frames are omitted, values can jump between consecutive stored bins.

ii.
```python
chunk_ft = ft[chunk]
neural_bin = spk_sel[:, chunk].mean(axis=1)
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The agent considered common chunk membership to provide alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is taken from the per-trial boolean `isRew`.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
```

iii. The field directly identifies rewarded-corridor trials.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is cast to float and repeated in every time bin of the trial.

ii.
```python
float(is_rew[tr])
```

iii. The agent treated reward availability as a binary per-trial contextual variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial `WallName` values across all sessions.

ii.
```python
wall_name = np.asarray(beh["WallName"])
names.update(map(str, np.unique(beh["WallName"])))
```

iii. The agent regarded `WallName` as the direct visual-category label.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique wall-name strings are sorted globally, mapped to integer class indices, and the trial's class is repeated for every bin.

ii.
```python
stim_names = output_stimulus_names(canonical_lookup, sessions)
stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. The agent used a deterministic vocabulary so decoder outputs are categorical and consistent across sessions.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from the event-frame list `LickFr`.

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=np.float64)
```

iii. The agent interpreted each valid listed frame as a lick event.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Finite, in-range lick indices set a framewise boolean mask. A rebinned output is 1 if any retained frame in that chunk contains a lick, otherwise 0.

ii.
```python
lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
lick_mask[lick_idx] = True
int(lick_mask[chunk].any())
```

iii. The agent used “any” rather than averaging so a sparse event remains a binary positive after temporal rebinning.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick mask is indexed by exactly the retained frames used for each neural average. Licks on excluded stationary/non-corridor frames are absent.

ii.
```python
neural_bin = spk_sel[:, chunk].mean(axis=1)
int(lick_mask[chunk].any())
```

iii. The agent aligned behavior and neural activity through shared frame chunks.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from framewise `ft_Pos`.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
```

iii. The agent identified `ft_Pos` as the corridor-position stream.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is averaged over each retained three-frame chunk before categorization.

ii.
```python
mean_pos = float(ft_pos[chunk].mean())
position_to_bin(mean_pos)
```

iii. The agent coarsened position on the same temporal grid as the averaged neural signal.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The mean position, expressed in decimeters, is divided by 10, floored, and clipped to class 0–3, corresponding to four 1 m intervals.

ii.
```python
def position_to_bin(value):
    return int(np.clip(math.floor(value / 10.0), 0, 3))
```

iii. This directly implements the requested four equal-length 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is averaged over exactly the same retained frame chunk as neural activity, then categorized.

ii.
```python
neural_bin = spk_sel[:, chunk].mean(axis=1)
mean_pos = float(ft_pos[chunk].mean())
```

iii. The agent used shared frame chunks for all time-varying variables.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise `ft_RunSpeed`.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
```

iii. The agent treated the supplied run-speed stream as the direct source.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is averaged within each retained three-frame chunk. The same chunk means are also collected globally to calculate category thresholds.

ii.
```python
speed_values.append(float(ft_speed[chunk].mean()))
mean_speed = float(ft_speed[chunk].mean())
```

iii. The agent made the speed statistic correspond to the temporal scale of the neural bins.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The 25th, 50th, and 75th percentiles are computed globally over means of all retained chunks; `searchsorted(..., side="right")` maps each mean to classes 0–3.

ii.
```python
q = np.quantile(speed_values, [0.25, 0.5, 0.75])
return int(np.searchsorted(thresholds, value, side="right"))
```

iii. The agent followed the instruction that each speed class correspond to 25% of the data, using the actual decoder-bin population.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is averaged over exactly the same retained frame chunk as neural activity and then discretized.

ii.
```python
neural_bin = spk_sel[:, chunk].mean(axis=1)
mean_speed = float(ft_speed[chunk].mean())
```

iii. Shared chunk indices provide direct binwise alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavior streams are truncated to their common length; invalid lick indices are discarded; duplicate behavior copies are compared and the first canonical copy is used; missing behavior sessions and sessions with no usable frames or neurons cause errors. Broad exceptions while loading behavior files are silently skipped. Final partial three-frame chunks are accepted.

ii.
```python
nfr = min(spk.shape[1], len(beh["ft"]))
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
if missing:
    raise RuntimeError(f"Missing behavior entries for sessions: {missing[:5]}")
except Exception:
    continue
```

iii. The agent explicitly audited duplicate behavior entries and structural consistency, favoring robust bounds checks and hard failures for missing session-level essentials.

## 12-a. What are the most time-consuming steps of the code?

i. Loading/concatenating large spike arrays, calculating variance for neuron selection, computing global speed thresholds by nested session/trial/chunk loops, converting every chunk, serializing the full pickle, and downstream decoder training are the dominant costs.

ii.
```python
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
var = variance_over_columns(spk, selected_frames)
for tr, frame_idx in enumerate(frames_by_trial):
    for chunk in chunks:
```

iii. The trajectory repeatedly monitored long conversion and training jobs and introduced neuron selection and temporal binning to control resource use.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial masks, global speed-bin means, per-session chunk means, construction of per-bin inputs/outputs, and parts of proportional neuron allocation could be vectorized or grouped. Variable trial/chunk lengths make some outer loops reasonable, but the inner chunk loop is a clear candidate.

ii.
```python
for tr in range(ntrials):
    mask = (ft_trind == tr) & is_corr & is_move
for tr, frame_idx in enumerate(frames_by_trial):
    for chunk in chunks:
```

iii. The trajectory gives no explicit vectorization rationale; implementation clarity and variable-length trials appear to have been prioritized.

## 12-c. What processing does the code repeat multiple times?

i. Trial-frame/chunk construction is performed once for global speed thresholds and again per session during conversion. Behavior files are loaded once to build the canonical lookup, while spike data are loaded once per conversion. Sample selection later rescans converted trials.

ii.
```python
for indices in trial_frame_indices(beh, nfr):
    for chunk in chunk_indices(indices, frames_per_bin):
frames_by_trial = trial_frame_indices(beh, nfr)
chunks = chunk_indices(frame_idx, frames_per_bin)
```

iii. The repeated preliminary pass was intentional so global quartiles could be known before categorical outputs were emitted.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and retains provenance, duplicate comparisons, detailed session metadata, `speed_values`, `kept_trial_indices`, the unused random seed state, and a separate sample-dataset selection path. `kept_trial_indices` is built but never returned. Variance ranking and discarded neurons are also extra work caused by the agent's non-reference subsampling choice.

ii.
```python
speed_values, speed_thresholds = compute_speed_thresholds(...)
kept_trial_indices = []
kept_trial_indices.append(tr)
np.random.seed(args.seed)
"behavior_entry_provenance": {base: list(entries) for base, entries in provenance.items()},
```

iii. The trajectory shows that provenance checks, a representative sample dataset, and extensive metadata were added for auditing and validation, even though the requested final downstream dataset does not require most of them.
