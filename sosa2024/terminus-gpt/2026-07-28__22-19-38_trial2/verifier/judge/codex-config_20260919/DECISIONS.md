# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds every `.nwb` file under `data`, sorts the paths, and opens each twice with `h5py`: once for a global reward-zone survey and once for conversion. `--sample` restricts this to the first two files; otherwise `--full` and the default both process all files.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
if args.sample:
    files = files[:2]
...
with h5py.File(path, 'r') as h:
```

iii. The notes report 152 sessions, 11 subjects, and 12,217 trials and state that all NWB sessions were loaded. The agent chose direct HDF5 access for speed and relied on observed NWB paths rather than `pynwb`.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each file's NWB subject field. A first-seen list supplies `subjects`, and each retained session receives the corresponding list index.

ii.
```python
subject = dec(h['general/subject/subject_id'][()])
...
if info['subject'] not in subject_names:
    subject_names.append(info['subject'])
subject_idx.append(subject_names.index(info['subject']))
```

iii. The agent documented that the files are grouped by subject directories and confirmed 11 subjects. Reading the embedded subject ID was treated as the authoritative split.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; its converted trials become one element of the session-level lists. Sessions with fewer than two retained trials are skipped.

ii.
```python
for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(...)
    if len(neural_trials) < 2:
        continue
    sessions_neural.append(neural_trials)
```

iii. The notes describe NWB files as session files and report that all 152 supplied sessions survived conversion.

## 1-d. How are the data split into trials?

i. Trials are the nonnegative unique values of the raw `trial number` stream. Within a trial, only samples having that value and `scanning > 0` are retained. The loaded `trial_start` stream is not used.

ii.
```python
trial_ids = np.unique(trial_num)
trial_ids = trial_ids[trial_ids >= 0]
for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
```

iii. The planning notes say trial-number segmentation was selected and scanning periods were to be treated as valid imaging samples. They record a planned comparison with `trial_start`, but no such reconciliation appears in the final code.

## 1-e. How are trials filtered based on quality controls?

i. A candidate trial is dropped if it has fewer than two scanning samples or contains no non-NaN reward-zone values. Entire sessions are dropped if fewer than two trials remain. There is no 50-sample minimum.

ii.
```python
if len(idx) < 2:
    continue
...
if len(rz_vals) == 0:
    continue
...
if len(neural_trials) < 2:
    continue
```

iii. The notes emphasize the decoder's two-trial requirement but do not justify the two-sample threshold or reward-zone-presence filter. They report that all raw trial counts were retained in practice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from every plane in NWB `processing/ophys/Deconvolved`; raw `Fluorescence`, `Neuropil`, and ROI `iscell` metadata are not used.

ii.
```python
plane_names = sorted(h['processing/ophys/Deconvolved'].keys())
arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
```

iii. The agent reasoned that the Methods mention deconvolved activity and concluded that the stored NWB `Deconvolved` signal was the analysis-ready neural source.

## 2-b. How is the `neural` data processed?

i. Plane matrices are concatenated along neurons, trial samples are selected, the result is transposed to neuron-by-time, and values are cast to `float16`. No neuropil subtraction, dF/F baseline, smoothing, or OASIS deconvolution is performed by the converter.

ii.
```python
neural_full = np.concatenate(deconv_planes, axis=1)
...
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. The notes say concatenating planes once avoids per-neuron loops, and float16 was introduced to reduce the full pickle from roughly 29 GB to 15 GB. The stored signal was assumed already processed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not cell-filtered: all columns in each Deconvolved plane are retained. The code does not apply `iscell` or remove speed-correlated putative interneurons.

ii.
```python
for plane in plane_names:
    arr = np.array(...)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
```

iii. The notes considered `iscell` filtering, but the final stated decision was to use all ROIs represented in the deconvolved matrices. They reported 312,110 neurons as matching the raw Deconvolved arrays.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is implicit: the first sample with the trial's stored ID and `scanning > 0` is treated as time zero, and those same indices slice neural and behavioral arrays.

ii.
```python
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = qts - trial_t0
neural_trial = neural_full[q_idx, :].T
```

iii. The agent described this as trial-start alignment and assumed the streams were already sample-aligned. It did not use the raw `trial_start` event to establish the boundary.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native samples are retained without rebinning or resampling. Per-session `dt` is the median position-timestamp difference; metadata stores the median `dt` over retained sessions in milliseconds.

ii.
```python
dt = float(np.median(np.diff(pos_t)))
...
'time_bin_size': float(np.median(dts) * 1000.0)
```

iii. The notes explicitly say the implementation uses native samples rather than explicit rebinning. This was chosen because behavior and stored neural matrices appeared to share sample indices.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `position/timestamps` at the retained trial indices.

ii.
```python
pos_t = np.array(beh['position/timestamps'], dtype=np.float64)
qts = pos_t[q_idx]
```

iii. The agent treated position timestamps as the common behavior clock and observed a stable native sampling interval.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first retained timestamp is subtracted from every timestamp in that trial, then the vector is cast to float32.

ii.
```python
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. This implements the required zero-at-trial-start representation, under the agent's trial-boundary definition.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both are indexed by the identical `q_idx` array, so each time value corresponds positionally to a neural column. There is no timestamp interpolation or explicit length assertion.

ii.
```python
qts = pos_t[q_idx]
neural_trial = neural_full[q_idx, :].T
```

iii. The notes say neural and behavior were assumed already aligned in the stored data; verification of array shapes passed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The saved input is derived from the NWB session `identifier` (`Env2` maps to 1; everything else to 0), not from the loaded environment stream. An environment-stream mapping is computed but unused.

ii.
```python
def parse_env_from_identifier(identifier):
    if 'Env2' in identifier:
        return 1
    return 0
...
env_from_identifier = parse_env_from_identifier(identifier)
```

iii. The notes say both the identifier and environment stream encode ENV1/ENV2, and that environment is constant within a trial. The final implementation favored the identifier.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The identifier is searched for `Env2`; the resulting 0/1 session value is broadcast across each trial.

ii.
```python
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32)
```

iii. The agent used this after sanity-checking that reward/environment labels varied as expected across sample sessions.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the raw `trial number/data` value used to define each trial.

ii.
```python
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
for tr in trial_ids:
```

iii. The notes call this the trial index within session and selected the raw stream directly.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Each raw trial ID is cast to float and broadcast across all retained timepoints; it is not reindexed after filtering.

ii.
```python
np.full(len(q_idx), float(tr), dtype=np.float32)
```

iii. The agent considered the stored trial number already continuous and suitable without transformation.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the preceding processed trial's reward outcome, itself computed from `Reward/data` events and `Reward/timestamps`.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
prev_outcome = 0
```

iii. The agent documented reward delivery as the source for rewarded versus omitted trials and set the first trial's previous outcome to zero.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `prev_outcome` is broadcast over the current trial, then updated to the current outcome only after that trial is appended. Thus it follows the preceding retained trial, not necessarily the preceding raw trial when a trial is skipped.

ii.
```python
np.full(len(q_idx), float(prev_outcome), dtype=np.float32)
...
prev_outcome = outcome
```

iii. The notes describe a one-trial lag with first trial zero; no special handling of skipped trials is justified.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses `position/data`, `reward_zone/data`, and indirectly the session identifier. For each raw reward-zone value, a session-wide median position is computed; a trial takes the mode reward-zone value and uses its median position as a single zone center.

ii.
```python
reward_zone_centers[zval] = float(np.nanmedian(pos[mask & (speed > -np.inf)]))
...
rz_mode = float(Counter(rz_vals.tolist()).most_common(1)[0][0])
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
```

iii. The agent initially interpreted the reward-zone stream as the location label, then changed categorical A/B/C to the identifier. It retained the stream-derived center for distance, intending to form a reward-relative position.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The converter subtracts the inferred scalar center from position. Consequently values inside the physical reward zone are generally nonzero; it does not calculate distance to the nearest zone edge.

ii.
```python
dist = pos_trial - rz_center
dist_bin = discretize_distance(dist)
```

iii. The notes describe this as continuous position relative to reward zone. They do not discuss the instruction's “distance to any location in the reward zone” or fixed A/B/C ranges.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Boolean masks create seven classes. Notably, `-10` is assigned to class 1 and class 2 begins strictly above `-10`; zero alone is class 3.

ii.
```python
out[(x >= -50) & (x <= -10)] = 1
out[(x > -10) & (x < 0)] = 2
out[x == 0] = 3
```

iii. The agent intended these masks to directly implement the requested labels. It did not note the boundary discrepancy with the reference's `np.digitize` implementation.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and reward-zone samples are selected with the same `q_idx` used for neural activity, yielding one distance category per neural timepoint.

ii.
```python
neural_trial = neural_full[q_idx, :].T
pos_trial = pos[q_idx]
```

iii. The agent assumed samplewise alignment of all continuous NWB streams and validated compatible shapes.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `position/data`.

ii.
```python
pos = np.array(beh['position/data'], dtype=np.float32)
pos_trial = pos[q_idx]
```

iii. The notes identify the NWB position stream as corridor position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The minimum and maximum over the entire session position stream define six evenly spaced edges; trial positions are digitized and clipped to classes 0–4.

ii.
```python
abs_pos_global_min = float(np.nanmin(pos))
abs_pos_global_max = float(np.nanmax(pos))
edges = np.linspace(pmin, pmax, 6)
```

iii. The agent intended five equal-width bins, but inferred the span from each session. Notes acknowledge that sample sessions occupied only bins 2–4 and say this should be monitored.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Internal session-derived edges are passed to `np.digitize`; out-of-range values are clipped.

ii.
```python
bins = np.digitize(pos, edges[1:-1], right=False)
bins = np.clip(bins, 0, 4)
```

iii. The agent interpreted “five equal-sized bins” as equal subdivisions of the observed session range rather than the specified fixed 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same sample indices and has one categorical value per neural timepoint.

ii.
```python
neural_trial = neural_full[q_idx, :].T
pos_trial = pos[q_idx]
```

iii. Direct index alignment was assumed and passed format validation.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `lick/data`.

ii.
```python
lick = np.array(beh['lick/data'], dtype=np.float32)
```

iii. The notes map the raw lick stream directly to the decoder's lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Values greater than zero are mapped to 1 and all others to 0.

ii.
```python
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. This was chosen to meet the requested binary no/yes representation.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural data are sliced with the same trial indices.

ii.
```python
neural_trial = neural_full[q_idx, :].T
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. The agent relied on common sample indexing rather than timestamp resampling.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The saved category is derived solely from the NWB identifier strings `LocationA`, `LocationB`, or `LocationC`; missing/unrecognized identifiers default to A. The reward-zone stream is not used for this output.

ii.
```python
def parse_location_from_identifier(identifier):
    if 'LocationA' in identifier: return 0
    if 'LocationB' in identifier: return 1
    if 'LocationC' in identifier: return 2
    return 0
```

iii. The agent says an initial interpretation of `reward_zone` was corrected after sample review: identifier-derived labels varied across sessions as expected and decoded strongly.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The parsed session category is broadcast unchanged over every timepoint of every trial in that session.

ii.
```python
rz_loc = session_reward_location
...
np.full(len(q_idx), rz_loc, dtype=np.int64)
```

iii. The agent treated reward location as session-level/per-trial context and broadcast it to satisfy the time-series decoder interface.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/data` and its independent `Reward/timestamps`, compared with the first and last retained position timestamps of the trial.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
```

iii. The notes identify actual reward delivery, rather than autoreward or a trial label, as the rewarded/omitted source.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward events whose timestamps lie inclusively within the retained trial interval are selected; outcome is 1 if any selected event value is positive and 0 otherwise, then broadcast across the trial.

ii.
```python
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
```

iii. The agent compared converted labels with raw event-derived outcomes in several sessions and found exact agreement; it attributed near-chance balanced decoding to the roughly 85/15 class imbalance.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code ignores NaNs when enumerating environment/reward-zone values and computing medians, drops trials with no reward-zone value, drops trials shorter than two samples, clips discretized position classes, defaults unknown environment/reward identifiers to ENV1/A, and drops sessions with fewer than two valid trials. It does not crop mismatched neural/behavior lengths, verify stream timestamps, or explicitly handle missing neural values.

ii.
```python
vals = np.unique(env_stream[~np.isnan(env_stream)])
...
if len(rz_vals) == 0: continue
...
return 0  # identifier fallbacks
```

iii. The agent describes these as pragmatic defensive choices and relied heavily on successful format verification. No documented investigation supports the silent identifier defaults or lack of cross-stream assertions.

## 13-a. What are the most time-consuming steps of the code?

i. Loading full neural arrays from every NWB, holding/concatenating them per session, serializing the very large pickle, and downstream decoder training are the dominant costs. The preliminary all-file reward-zone survey adds another I/O pass.

ii.
```python
arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
...
pickle.dump(data, f)
```

iii. The notes report about 0.6 seconds per sample session for conversion and emphasize the 29 GB float32 versus 15 GB float16 serialization cost. They identify full-array loading as the main conversion inefficiency.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop repeatedly forms masks, extracts arrays, finds a Python `Counter` mode, and broadcasts labels; much of the session-wide binning and trial outcome assignment could be vectorized before splitting. The plane loop is small and necessary for separate datasets. `infer_env_binary` unnecessarily uses `np.vectorize` rather than array comparisons.

ii.
```python
for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
...
return np.vectorize(lambda z: mapping.get(z, 0))(env_stream)
```

iii. The notes say performance was improved by concatenating planes once and avoiding per-neuron loops, while accepting the natural per-trial loop because trials have variable lengths.

## 13-c. What processing does the code repeat multiple times?

i. Every file's full `reward_zone` array is read in a preliminary survey, then read again in `load_session`. Within trials, constant label arrays are repeatedly allocated. The input matrix is redundantly cast to float32 twice.

ii.
```python
for f in files:
    all_rz.append(np.array(...'reward_zone/data'...))
...
rz = np.array(beh['reward_zone/data'], dtype=np.float32)
...
inp = np.vstack(...).astype(np.float32)
inp = inp.astype(np.float32)
```

iii. The first pass was intended to infer a global reward-zone map, although that map is never used. The agent otherwise highlights single concatenation as a speed optimization but does not document these repetitions.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `trial_start`, `env_trial`, `env_binary`, and the argument/global result `reward_zone_value_map` are computed or loaded but never affect saved values. `speed > -np.inf` is a tautological mask except for NaNs. The second `inp.astype(np.float32)` is redundant. Optional plots are diagnostic and not consumed downstream.

ii.
```python
trial_start = np.array(...)
env_binary = infer_env_binary(identifier, env)
env_trial = env_binary[q_idx]
reward_zone_value_map = infer_reward_zone_map(...)
inp = inp.astype(np.float32)
```

iii. The agent did not explicitly identify these discarded computations. It only documented that plotting was optional and that concatenating once avoided other unnecessary work.
