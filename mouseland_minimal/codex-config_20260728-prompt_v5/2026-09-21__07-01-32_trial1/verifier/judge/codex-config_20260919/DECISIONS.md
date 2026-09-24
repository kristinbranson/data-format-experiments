# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans every `*_neural_data.npy` spike file to define the 89 sessions, scans all `Beh_*.npy` files to build a behavior-candidate index, uses `Imaging_Exp_info.npy` only to attach experiment types, and loads the matching retinotopy file per session. If a recording occurs in several behavior files, it selects one candidate by a heuristic score (more unique walls and finite stimulus IDs, fewer placeholders and swap suffixes). Neural data are written to per-session memmap sidecars; the pickle stores mapped trial views.

ii.
```python
spk_files = sorted(SPK_DIR.glob("*_neural_data.npy"))
for spk_file in spk_files:
    raw_key = spk_file.name.replace("_neural_data.npy", "")
    beh_fname, beh_key = select_behavior_candidate(raw_key, behavior_index[raw_key])
    beh = np.load(BEH_DIR / beh_fname, allow_pickle=True).item()[beh_key]
    retino = np.load(RETINO_DIR / f"{raw_key.rsplit('_', 1)[0]}_trans.npz", allow_pickle=True)
```

iii. The trajectory says the index contains 142 analysis entries but only 89 unique recordings reused across figures, so the agent chose to deduplicate by raw recording. It introduced candidate scoring to avoid masked or split behavior representations, and sidecar memmaps because directly pickling all neural trial matrices was estimated to require tens of gigabytes of RAM.

## 1-b. How are the data split into subjects?

i. Subject identity is parsed as the first component of the spike-file session key. Unique subject names are sorted, and every session receives the corresponding integer `subject_idx`.

ii.
```python
subject, _, block = parse_session_key(raw_key)
subjects = sorted({plan.subject for plan in session_plans})
subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent treated the mouse prefix in the recording key as the explicit subject identifier; no inferred clustering was needed.

## 1-c. How are the data split into sessions?

i. Each unique spike filename defines one session keyed by mouse, date, and block (`<mouse>_<YYYY>_<MM>_<DD>_<block>`). Sorting the spike paths fixes session order and inherently deduplicates repeated experiment-index entries.

ii.
```python
spk_files = sorted(SPK_DIR.glob("*_neural_data.npy"))
raw_key = spk_file.name.replace("_neural_data.npy", "")
subject, _, block = parse_session_key(raw_key)
```

iii. The trajectory explicitly notes that recordings are reused across paper figures and that swap sessions must deduplicate using the full key including block.

## 1-d. How are the data split into trials?

i. It loops over `range(ntrials)` and assigns to trial `tr` all frames whose `ft_trInd == tr` and whose `ft_CorrSpc` flag is true. Thus trials are variable-length corridor-only frame sequences.

ii.
```python
for tr in range(ntrials):
    idx = np.flatnonzero((ft_tr == tr) & corr)
```

iii. The agent concluded that raw imaging frames restricted to corridor time provide direct corridor-entry alignment without resampling, and that the decoder supports variable-length trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are discarded if they contain fewer than 5 corridor frames, if `Gray_space_time - Trial_start_time` exceeds 40 seconds, or if the maximum timestamp gap between retained frames exceeds 1 second. A session with fewer than two remaining trials causes an error.

ii.
```python
if idx.size < MIN_FRAMES_PER_TRIAL:
    continue
duration_s = float((gray_time[tr] - trial_start[tr]) * 24.0 * 3600.0)
max_gap_s = float(np.max(np.diff(ft[idx]) * 24.0 * 3600.0))
if duration_s > TRIAL_DURATION_MAX_S or max_gap_s > MAX_FRAME_GAP_S:
    continue
```

iii. The trajectory reports many pathological, pause-heavy corridor durations and characterizes them as isolated bad trials. The agent chose explicit absolute filters to remove them without discarding otherwise usable sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the per-plane arrays in `spks` within each `<session>_neural_data.npy`; planes are laid into adjacent neuron columns. Retinotopy `iarea` supplies region labels but not neural values.

ii.
```python
spk_obj = np.load(SPK_DIR / f"{plan.raw_key}_neural_data.npy", allow_pickle=True).item()
spk_chunks = spk_obj["spks"]
for chunk in spk_chunks:
    block = chunk[:, keep_idx[c0:c1]].T
```

iii. The agent found that the authors' loader concatenates `spks` chunks and that these files already contain the deconvolved activity used by the paper.

## 2-b. How is the `neural` data processed?

i. No signal transformation, normalization, interpolation, or padding is applied. Selected frame columns from every plane are transposed into time-by-neuron float16 memmaps; each exposed trial view transposes its row slice back to neuron-by-time.

ii.
```python
base = np.memmap(out_path, dtype=np.float16, mode="w+", shape=base_shape)
base[c0:c1, col_start:col_start + n_chunk] = block
arr = base[row_start:row_stop, :].T.view(MappedTrialArray)
```

iii. The trajectory justifies raw frame resolution as the native data grid and float16/memmaps as necessary to make the full dataset loadable without an 80+ GB RAM spike.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It does not filter neurons. Every spike row is retained, including retinotopy codes not mapped to the four paper regions; these are assigned an additional `unknown` region.

ii.
```python
out = np.full(iarea.shape, REGION_TO_INDEX["unknown"], dtype=np.int16)
total_neurons = int(sum(chunk.shape[0] for chunk in spk_chunks))
base_shape = (int(keep_idx.size), total_neurons)
```

iii. The agent chose coarse paper-style region grouping but added `unknown`; it reasoned that the spike files had already undergone cell curation and did not apply another quality metric.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial begins with its first frame jointly labeled as that trial and inside the corridor (`ft_CorrSpc`), interpreted as corridor entry/trial start, and ends at the last such frame. Trials remain variable length, with no padding.

ii.
```python
idx = np.flatnonzero((ft_tr == tr) & corr)
MappedTrialArray(memmap_path, base_shape, np.dtype(np.float16).str,
                 tp.row_start, tp.row_stop)
```

iii. The agent explicitly selected corridor entry and corridor-only frames because the requested alignment is trial start and the downstream decoder accepts variable lengths.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One native imaging frame is one bin (about 315 ms); there is no rebinning. Metadata uses the median of session median `ft` intervals rather than the reference's fixed 3.17 Hz constant.

ii.
```python
dt = np.diff(ft) * 24.0 * 3600.0
median_dt_s = float(np.median(dt))
median_time_bin_ms = 1000.0 * float(np.median(np.asarray(time_bin_values)))
```

iii. The agent confirmed an approximately 3.18 Hz imaging time base and chose the native resolution to avoid invented resampling.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and per-frame timestamps `ft`.

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
t_frame = ft[idx]
```

iii. The agent checked that cue timing was available across sessions and selected the direct timestamp fields.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. It subtracts each selected frame timestamp from the trial's sound timestamp and converts MATLAB serial-day units to seconds; values are positive before and negative after the cue.

ii.
```python
((sound_time[tr] - t_frame) * 24.0 * 3600.0).astype(np.float32)
```

iii. The trajectory's concern was avoiding NaNs in sessions lacking cues; after inspection it used the supplied direct times without imputation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly `tp.frame_idx`, the same frame indices used to extract the neural trial, yielding one value per neural column.

ii.
```python
idx = tp.frame_idx
t_frame = ft[idx]
```

iii. The agent relied on the shared imaging-frame time base for all streams.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date embedded in each session key and the earliest recorded date for that mouse.

ii.
```python
date = session_date(raw_key)
first_date = {subject: min(plan.date for plan in session_plans if plan.subject == subject)
              for subject in subjects}
```

iii. The agent interpreted “day” literally as elapsed calendar days from the first recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The integer calendar-day difference from that mouse's first session is converted to float and broadcast across every frame of every trial in the session.

ii.
```python
plan.day_of_training = float((plan.date - first_date[plan.subject]).days)
np.full((T,), plan.day_of_training, dtype=np.float32)
```

iii. No explicit trajectory justification beyond using session dates was recorded.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses the per-trial `Trial_start_time` timestamp and per-frame `ft` timestamps.

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
t_frame = ft[idx]
```

iii. The agent viewed these direct timestamps as the cleanest way to express corridor-entry-relative time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. It subtracts trial start from every selected frame time and converts days to seconds, producing float32 values.

ii.
```python
((t_frame - trial_start[tr]) * 24.0 * 3600.0).astype(np.float32)
```

iii. The operation was chosen to retain the native irregular timestamp information without resampling.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same trial frame-index array as neural extraction, so every neural column has the corresponding elapsed-time value.

ii.
```python
idx = tp.frame_idx
t_frame = ft[idx]
```

iii. The shared frame grid provides alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from the per-trial boolean `isRew`.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
```

iii. The raw field directly expresses whether reward is available.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is cast to float (`0.0` or `1.0`) and broadcast over every selected frame of the trial.

ii.
```python
np.full((T,), float(is_rew[tr]), dtype=np.float32)
```

iii. No further processing was considered necessary for this per-trial flag.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the trial-level `WallName`, not `TrialStim` or `stim_id`.

ii.
```python
stim_idx = canonical_stimulus_index(str(beh["WallName"][tr]))
```

iii. The agent observed that `TrialStim` can contain placeholders/masking in swap sessions and therefore preferred `WallName`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Fifteen raw wall names are mapped to eight labels. Rock textures are collapsed into matching numbered circles, wood textures into matching numbered leaves, while `circle1/2/3`, `leaf1/2/3`, and two leaf swap labels remain distinct. The resulting index is broadcast over time.

ii.
```python
WALL_TO_CANONICAL = {"rock1": "circle1", "wood1": "leaf1",
                     "wood1_swap1": "leaf1_swap1", ...}
return np.full((T,), stim_idx, dtype=np.int16)
```

iii. The trajectory states that raw rock/wood names were mapped into canonical circle/leaf labels. It did not justify retaining number and swap variants as eight separate categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses both lick frame positions `LickFr` and lick trial indices `LickTrind`.

ii.
```python
lick_frames = np.asarray(np.rint(beh["LickFr"]).astype(np.int64))
lick_trials = np.asarray(np.rint(beh["LickTrind"]).astype(np.int64))
```

iii. The agent checked fractional lick indices and chose a per-frame binary raster while using trial labels to prevent cross-trial assignment.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frames are rounded to nearest integers, clipped to the session range, mapped into relative positions of the retained trial frames, deduplicated, and marked 1; all other frames are 0.

ii.
```python
lick_frames = np.clip(lick_frames[sel], 0, nfr - 1)
frame_to_rel[frame_idx] = np.arange(frame_idx.size, dtype=np.int32)
rel = frame_to_rel[lick_frames]
lick[np.unique(rel[rel >= 0])] = 1
```

iii. The trajectory says it specifically tested whether nearest-frame assignment was suitable for fractional lick indices.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are mapped onto the exact retained frame indices, so the returned vector has one element for every neural time bin.

ii.
```python
frame_to_rel[frame_idx] = np.arange(frame_idx.size, dtype=np.int32)
```

iii. Alignment is by raw imaging-frame number.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from framewise `ft_Pos` at the trial's retained indices.

ii.
```python
pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
pos_bins = position_series_for_trial(pos[idx])
```

iii. The raw stream already shares the neural imaging grid.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped into `[0, 39.999]`, divided by 10 (the raw unit is decimeters), floored, and cast to integer.

ii.
```python
pos_clipped = np.clip(pos_trial, 0.0, 39.999)
return np.floor(pos_clipped / 10.0).astype(np.int16)
```

iii. This directly implements four equal 1 m bins over the 4 m corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 10, 20, and 30 decimeters, producing codes 0–3 for 0–1 m, 1–2 m, 2–3 m, and 3–4 m; clipping assigns out-of-range values to an endpoint bin.

ii.
```python
np.floor(np.clip(pos_trial, 0.0, 39.999) / 10.0).astype(np.int16)
```

iii. The categories follow the decoder instruction's equal-length spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed by the same `idx` array used for the neural trial.

ii.
```python
pos_bins = position_series_for_trial(pos[idx])
```

iii. Both streams are framewise and require no interpolation.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise `ft_RunSpeed`; negative values are clipped to zero.

ii.
```python
speed = np.clip(np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32), 0.0, None)
```

iii. The agent used the raw speed stream requested by the decoder specification.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. During planning, all retained-trial speeds from all sessions are concatenated and global 25th/50th/75th value quantiles are computed. Each selected trial value is then digitized against those thresholds.

ii.
```python
all_speed = np.concatenate(speed_values).astype(np.float32)
speed_quantiles = np.quantile(all_speed, [0.25, 0.50, 0.75]).astype(np.float32)
return np.digitize(speed_trial, speed_quantiles, right=True).astype(np.int16)
```

iii. The trajectory says speed bins looked close to uniform “by construction”; the choice of global value thresholds rather than per-session ranks was not explicitly defended.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The global retained-frame 25th, 50th, and 75th percentiles are three thresholds. `np.digitize(..., right=True)` yields categories 0–3, although ties (especially at zero) prevent guaranteed equal counts.

ii.
```python
np.digitize(speed_trial, speed_quantiles, right=True).astype(np.int16)
```

iii. The agent interpreted “each corresponding to 25% of the data” as global quartile thresholds.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The same `idx` used for neural extraction selects speed samples before categorization.

ii.
```python
speed_bins = speed_series_for_trial(speed[idx], speed_quantiles)
```

iii. Both streams use the raw frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles behavior/spike frame counts differing by up to two by recomputing trial plans using spike length; larger differences raise an error. Out-of-range lick frames are clipped, negative speed and out-of-range position are clipped, behavior duplicates are resolved heuristically, short/gappy/long trials are dropped, and sessions with fewer than two trials fail. Unknown brain areas are retained. There is no general NaN imputation.

ii.
```python
if spk_nfr != plan.nframes_behavior:
    if abs(spk_nfr - plan.nframes_behavior) > 2:
        raise RuntimeError(...)
    trial_plans, median_dt_s, _ = compute_trial_plans_from_behavior(beh, spk_nfr)
lick_frames = np.clip(lick_frames[sel], 0, nfr - 1)
```

iii. The trajectory identifies the one/two-frame mismatch as a real edge case and deliberately re-trims to the actual spike frame count. Other choices aim to keep verification free of malformed shapes and NaNs.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant conversion cost is reading every large `spks` chunk and copying retained frames, in 4096-frame blocks, into roughly 108 GB of float16 memmap sidecars. The planning pass also repeatedly loads behavior files. Downstream exhaustive validation is especially slow because it scans all neural values.

ii.
```python
for chunk in spk_chunks:
    for c0 in range(0, keep_idx.size, time_block):
        block = chunk[:, keep_idx[c0:c1]].T
        base[c0:c1, col_start:col_start + n_chunk] = block
```

iii. The trajectory records 89-session conversion progress and says the verifier took over ten minutes because it scanned all 108 GB for NaNs and consistency.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial planning loops over every trial to compute masks, durations, gaps, and speed lists; output assembly loops over every session and trial; lick mapping allocates an `nfr` lookup per trial. Some trial summary work and output construction could be vectorized session-wise, and one frame-to-relative map could be reused per session, though variable trial lengths limit complete vectorization. The neural chunk/block loops are intentional bounded-memory I/O.

ii.
```python
for tr in range(ntrials):
    idx = np.flatnonzero((ft_tr == tr) & corr)
...
for tp in plan.trial_plans:
    lick = lick_series_for_trial(beh, tr, idx, nfr)
```

iii. The agent did not discuss vectorization directly; it prioritized bounded memory and progress through very large recordings.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are loaded while building the candidate index, again once per session during planning, and again during conversion. Trial plans may be recomputed after spike-length trimming. Trial masks are scanned separately for each trial, and lick frame lookup arrays are rebuilt per trial. Full verification was also repeated when the training entry point was started.

ii.
```python
beh = np.load(BEH_DIR / beh_fname, allow_pickle=True).item()[beh_key]
...
def load_behavior(plan):
    beh = np.load(BEH_DIR / plan.behavior_file, allow_pickle=True).item()
```

iii. The trajectory explicitly calls the training script's second exhaustive scan redundant and stops it; it also notes an early memory trap from retaining behavior dictionaries, explaining the reload-for-low-memory tradeoff.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Candidate scoring computes `stim_id` finiteness and placeholder counts used only to choose a duplicate behavior record. Planning gathers experiment types and several diagnostic fields only for metadata. It computes duration and gap values for retained trials, but downstream decoding uses only their filtering effect. Neural data are transposed/copied into sidecars and then exposed through transposed views, which is storage plumbing rather than analysis. It also retains unknown-area neurons, which the reference would discard.

ii.
```python
"n_finite_stim_id": int(np.isfinite(np.asarray(dat["stim_id"], dtype=float)).sum()),
"placeholder_count": int(np.sum(np.asarray(dat["TrialStim"]) == "stimulus_of_trial")),
...
experiment_types=exp_index.get(raw_key, [])
```

iii. The trajectory frames the candidate diagnostics as safeguards for duplicated/swap behavior files and memmap work as required for tractable storage, rather than scientific preprocessing.
