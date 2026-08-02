# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment metadata from `Imaging_Exp_info.npy`, which contains 23 experiment types with 142 total entries. It deduplicates these to 89 unique sessions by grouping on the base session ID `<mouse>_<date>_<blk>` and choosing one canonical entry per group. For each session, it loads behavior from `Beh_<exp_type>.npy`, neural data via `utils.load_spk()`, and retinotopy from `<mouse>_<date>_trans.npz`.

ii.
```python
def build_session_catalog() -> list[SessionCandidate]:
    exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
    grouped: dict[str, list[SessionCandidate]] = defaultdict(list)
    for exp_type, db_list in exp_info.items():
        for db in db_list:
            session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            beh_key = session_id
            if "stimtype" in db:
                beh_key = f"{beh_key}_{db['stimtype']}"
            grouped[session_id].append(...)
    return [choose_canonical(grouped[sid]) for sid in sorted(grouped, key=session_sort_key)]

def load_behavior(exp_type: str, beh_key: str) -> dict:
    beh = np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
    return beh[beh_key]
```

iii. The AI documented that `Imaging_Exp_info.npy` contains 142 entries across 23 experiment types, mapping to 89 unique neural recordings matching the paper's "89 recordings in 19 mice." It reuses `utils.load_spk()` from the reference code for neural data loading.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field in the session metadata. A sorted list of unique mouse names forms the `subjects` list, and each session is assigned a `subject_idx` mapping to its mouse.

ii.
```python
subjects_all = sorted({cand.db["mname"] for cand in catalog})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects_all)}
# ...
"subject_idx": np.asarray([subject_to_idx[s.subject] for s in processed_sessions], dtype=np.int64),
```

iii. The AI noted that `mname` is the mouse identifier used throughout the reference code and that there are 19 unique subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Sessions are identified by the base key `<mname>_<datexp>_<blk>`. Multiple experiment-type entries sharing the same base key are deduplicated to one canonical session, preferring entries without a `stimtype` field. Sessions are sorted by subject, date, then block number.

ii.
```python
def choose_canonical(candidates: list[SessionCandidate]) -> SessionCandidate:
    def key_fn(c: SessionCandidate) -> tuple[int, str, str]:
        has_stimtype = 1 if "stimtype" in c.db else 0
        return has_stimtype, c.exp_type, c.beh_key
    return sorted(candidates, key=key_fn)[0]
```

iii. CONVERSION_NOTES.md documents that the same recording appears under multiple analysis labels in `Imaging_Exp_info.npy`, requiring deduplication from 142 entries to 89 unique sessions.

## 1-d. How are the data split into trials?

i. Within each session, trials are iterated from index 0 to `beh['ntrials']-1`. Each trial spans from `StartFr[trial_idx]` to `GrayFr[trial_idx]` (corridor entry to gray-space entry), covering only the texture area (0–4 m). Only running frames (`ft_move > 0`) within this range are retained.

ii.
```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    frames = frames[ft_move[frames] > 0]
    if frames.size == 0:
        continue
```

iii. The AI documented that using `StartFr:GrayFr` keeps exactly the 0–4 m texture interval, and filtering to running frames matches the paper's rule "We only considered timepoints during running for analysis."

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded only if they have zero running frames within the texture segment (frames from `StartFr` to `GrayFr` where `ft_move > 0`). No additional trial-level quality filtering (e.g., minimum trial length, lick requirements) is applied.

ii.
```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
if frames.size == 0:
    continue
```

iii. CONVERSION_NOTES.md states that all 38,110 trials across 89 sessions are retained after filtering, matching the raw data count, implying no trials had zero running frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spk/*_neural_data.npy` files, loaded via `utils.load_spk()` which concatenates the `['spks']` list into a neuron-by-frame matrix. These are deconvolved fluorescence traces from calcium imaging.

ii.
```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
# ...
neural = spk_sel[:, frames].astype(np.float32, copy=False)
```

iii. The AI noted that `load_spk()` loads preprocessed deconvolved traces and that "the saved neural signal is already the processed neural activity used downstream" — no additional dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. The raw deconvolved traces are not further processed (no normalization, z-scoring, spatial interpolation, or temporal rebinning). They are simply sliced to running frames within the texture segment and subset to selected neurons.

ii.
```python
spk_sel = spk[selected_mask]
# ...
neural = spk_sel[:, frames].astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES.md documents that the reference code does not compute delta-F-over-F inside the repository and that `load_spk()` directly provides the processed signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are curated using a two-part selection: (1) stimulus-selective neurons in visual cortex regions with `|d'| >= 0.3` between primary rewarded and non-rewarded stimuli, computed on running corridor frames; (2) reward-prediction neurons in aHV requiring cue-frame stimulus d' > 0.3 and late-vs-early cue d' above the 95th percentile within aHV. A fallback selects top-|d'| neurons if fewer than 64 pass.

ii.
```python
stim1_fr = (ft_wall == rew_primary) & corr & running
stim2_fr = (ft_wall == nonrew_primary) & corr & running
dp = safe_dprime(spk[:, stim1_fr], spk[:, stim2_fr])
stim_selective = visual_mask & np.isfinite(dp) & (np.abs(dp) >= DP_THRESHOLD)
# ...
reward_pred = (ahv_mask & np.isfinite(reward_dp) & np.isfinite(dp_sound)
               & (dp_sound > DP_THRESHOLD) & (reward_dp >= reward_dp_thr))
selected = stim_selective | reward_pred
```

iii. CONVERSION_NOTES.md documents that this follows the two main neuron-selection motifs in the paper. Step 10 describes an iteration where the initial curation was tightened to better match the reference reward-prediction code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to corridor entry (trial start). Each trial's neural data begins at `StartFr` (the frame when the animal enters the corridor) and ends at `GrayFr` (entry into gray space), with only running frames retained. The first frame of each trial corresponds to the trial start event.

ii.
```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
neural = spk_sel[:, frames].astype(np.float32, copy=False)
```

iii. The metadata records `"temporal_alignment_event": "corridor entry (trial start / StartFr)"` and `"off_start": 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native calcium imaging frame rate (~315 ms per frame, ~3.17 Hz). No temporal rebinning is applied. The time bin size is computed from the median inter-frame interval across sessions.

ii.
```python
def compute_time_bin_ms(catalog: list[SessionCandidate]) -> float:
    medians = []
    for cand in catalog:
        ft = np.asarray(load_behavior(cand.exp_type, cand.beh_key)["ft"], dtype=float)
        d = np.diff(ft)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            medians.append(float(np.median(d) * SECONDS_PER_DAY * 1000.0))
    return float(np.median(medians))
```

iii. CONVERSION_NOTES.md references the notebook statement that the calcium imaging frame rate is 3.17 Hz, and notes that the time bin is derived from the frame timestamps rather than assumed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from: (1) the retained frame indices within the trial, (2) `beh['SoundFr']` (the neural frame when the sound cue was delivered), and (3) the median inter-frame interval computed from `beh['ft']` timestamps.

ii.
```python
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. CONVERSION_NOTES.md maps this input to `beh['SoundTime']`, `beh['ft']`, and `StartFr:GrayFr` running frames.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound cue frame index is located within the retained (running-only) frame array using `np.searchsorted`. The time to cue at each retained frame is `(cue_frame_position - current_frame_position) * frame_dt`, where `frame_dt` is the median inter-frame interval. Positive values indicate frames before the cue; negative values indicate frames after the cue.

ii.
```python
dft = np.diff(ft)
dft = dft[np.isfinite(dft) & (dft > 0)]
frame_dt = float(np.median(dft) * SECONDS_PER_DAY)
retained_idx = np.arange(frames.size, dtype=np.float32)
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The AI justified using frame index differences multiplied by `frame_dt` rather than raw `ft` timestamp differences, noting this provides a cleaner time representation when non-running frames are dropped.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time-to-cue is computed for each retained running frame, so it is inherently aligned with the neural data — both share the same frame indices.

ii.
```python
# frames is the same array used for neural[:, frames]
retained_idx = np.arange(frames.size, dtype=np.float32)
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. By construction, each element of `t_to_cue` corresponds to the same retained frame as the corresponding column of the neural matrix.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from `db['datexp']` (the session recording date) for each session. The day offset is computed as the number of calendar days since the first recording date for each mouse.

ii.
```python
def compute_day_offsets(catalog: list[SessionCandidate]) -> dict[str, float]:
    by_subject: dict[str, list[datetime]] = defaultdict(list)
    session_dates: dict[str, datetime] = {}
    for cand in catalog:
        date = datetime.strptime(cand.db["datexp"], "%Y_%m_%d")
        by_subject[cand.db["mname"]].append(date)
        session_dates[cand.session_id] = date
    first_dates = {subject: min(dates) for subject, dates in by_subject.items()}
    return {
        cand.session_id: float((session_dates[cand.session_id] - first_dates[cand.db["mname"]]).days)
        for cand in catalog
    }
```

iii. The AI noted that a complete per-session training-day label is not available for all sessions, so calendar date relative to each mouse's first recording is used as a proxy. The range is [0, 92] days.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date is identified. Each session's day-of-training is the integer number of days between the session date and the first date. This scalar value is broadcast to all retained frames in each trial.

ii.
```python
day = np.full(frames.shape, session_day, dtype=np.float32)
```

iii. Documented in CONVERSION_NOTES.md as Key Decision 8: "Use day-since-first-recording as the training-day proxy."

## 4-a (Environment type). What variables in the raw data is `input` *Environment type* derived from?

i. The AI maps *Environment type* to `input` *Reward availability*, derived from `beh['isRew']`. The instruction's "Environment type" concept is not separately represented; instead, `isRew` indicates whether the current corridor is rewarded (1) or not (0).

ii.
```python
is_rew = np.asarray(beh["isRew"]).astype(np.float32)
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. The instructions specify "Reward availability: 1 if in rewarded corridor, 0 if not" as a decoder input. The AI treated environment type and reward availability as the same variable, since the corridor identity determines both.

## 4-b (Environment type). What processing is involved in computing `input` *Environment type*?

i. The per-trial `isRew` boolean is cast to float32 (0.0 or 1.0) and broadcast to all retained frames in the trial. No further processing is applied.

ii.
```python
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. Documented as a binary time-varying input that is constant within each trial.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the retained frame indices within each trial and the session's median inter-frame interval (`frame_dt`), which is computed from `beh['ft']` timestamps.

ii.
```python
retained_idx = np.arange(frames.size, dtype=np.float32)
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. CONVERSION_NOTES.md maps this to behavior frame timing fields `beh['ft']` and `StartFr`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The time since trial start at each retained frame is simply the zero-based index of that frame in the retained array multiplied by the median frame interval. This assumes uniform frame spacing among retained frames.

ii.
```python
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The AI chose to use frame index × frame_dt rather than actual `ft` timestamp differences for consistency.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Inherently aligned — each element of `t_since` corresponds to the same retained running frame used for the neural data.

ii.
```python
# frames is used for both neural and t_since
retained_idx = np.arange(frames.size, dtype=np.float32)
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. Same frame array is used for all variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh['isRew']`, a per-trial boolean indicating whether the trial is in a rewarded corridor.

ii.
```python
is_rew = np.asarray(beh["isRew"]).astype(np.float32)
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. CONVERSION_NOTES.md states this "Uses rewarded-corridor identity even in unsupervised sessions, matching reference use of isRew."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is cast to float (0.0 or 1.0) and broadcast to all retained frames per trial. No further processing.

ii.
```python
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. The AI treats this as a binary per-trial constant, repeated to be time-varying for format consistency.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh['WallName']`, which contains the stimulus name for each trial (e.g., 'leaf1', 'circle1', 'rock1', etc.).

ii.
```python
wall_name = np.asarray(beh["WallName"]).astype(str)
stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)
```

iii. CONVERSION_NOTES.md Key Decision 5: "Use exact WallName strings as stimulus labels" to avoid ambiguity from `stim_id` NaNs in swap sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique `WallName` values across all sessions are collected and sorted alphabetically. Each is assigned an integer index. The per-trial wall name is mapped to its integer index and broadcast to all retained frames.

ii.
```python
def collect_stimulus_values(catalog: list[SessionCandidate]) -> list[str]:
    names = set()
    for cand in catalog:
        beh = load_behavior(cand.exp_type, cand.beh_key)
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names)

stimulus_to_idx = {name: idx for idx, name in enumerate(stimulus_values)}
stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)
```

iii. 15 unique stimulus categories were identified across all sessions.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh['LickFr']` (neural frame indices of lick events) and `beh['LickTrind']` (trial index of each lick event).

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float)
lick_tr = np.asarray(beh["LickTrind"], dtype=float)
```

iii. CONVERSION_NOTES.md maps licking to `beh['LickFr']` and `beh['LickTrind']`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick frames matching the trial index are identified. The retained running frames are checked for membership in the lick frame set. The result is a binary array: 1 if the frame is a lick frame, 0 otherwise.

ii.
```python
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
lick_frames_trial = lick_frames_trial.astype(int)
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. The AI documented this as "Binary per retained frame: 1 if one or more licks occur on that imaging frame, else 0."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is computed on the same retained running frame indices used for neural data. Each lick binary value corresponds to the same imaging frame as the corresponding neural activity column.

ii.
```python
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
# frames is the same array used for neural data
```

iii. Inherently aligned by shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `beh['ft_Pos']`, the position within the corridor for each neural frame (in decimeters, range 0–60).

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
pos = ft_pos[frames]
```

iii. CONVERSION_NOTES.md maps this to `beh['ft_Pos']` on retained frames.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position values are extracted for retained running frames. Since only texture-area frames are kept (StartFr to GrayFr), positions range from 0 to ~40 dm (0–4 m).

ii.
```python
pos = ft_pos[frames]
```

iii. No additional processing beyond frame extraction.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is discretized into 4 equal-length 1-meter bins: [0,10), [10,20), [20,30), [30,40] dm. This uses `np.floor(pos / 10.0)` clipped to range [0, 3].

ii.
```python
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The AI noted this "Exactly matches the requested 4 one-meter bins" from the decoder task specification.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position values are extracted from the same retained running frame indices used for neural data. Alignment is inherent.

ii.
```python
pos = ft_pos[frames]  # frames is the same array used for neural data
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. Same frame array used throughout.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh['ft_RunSpeed']`, the running speed for each neural frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)
raw_speed = ft_speed[frames].astype(np.float32)
```

iii. CONVERSION_NOTES.md maps this to `beh['ft_RunSpeed']` on retained frames.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Raw speed values are extracted for retained running frames. They are collected across all sessions and trials for global quartile computation.

ii.
```python
raw_speed = ft_speed[frames].astype(np.float32)
all_speeds.append(raw_speed)
```

iii. No normalization or smoothing is applied to the raw speed values.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quartile edges are computed from all retained running-frame speeds across all sessions using `np.quantile` at [0, 0.25, 0.5, 0.75, 1.0]. Each frame's speed is binned using `np.searchsorted` into 4 bins. Adjacent equal edges are nudged with `np.nextafter` to ensure monotonicity.

ii.
```python
all_speed_concat = np.concatenate(all_speed_values).astype(np.float32)
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
for i in range(1, len(speed_edges)):
    if speed_edges[i] <= speed_edges[i - 1]:
        speed_edges[i] = np.nextafter(speed_edges[i - 1], np.float32(np.inf))
# ...
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. The AI documents that "Quartiles computed on the included running-only texture frames" match the instruction to discretize into "4 bins, each corresponding to 25% of the data."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed values are extracted from the same retained running frame indices used for neural data. Alignment is inherent.

ii.
```python
raw_speed = ft_speed[frames].astype(np.float32)
```

iii. Same frame array used throughout.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Trials with zero running frames are skipped. (2) Sound frame indices outside valid range are handled by `np.searchsorted` placing them at array boundaries. (3) Sessions with no finite aHV reward d' values get an infinite threshold, preventing spurious neuron selection. (4) The d' computation uses `np.nanmean`/`np.nanstd` to handle NaN values. (5) The `stim_id` NaN issue in swap sessions is avoided by using `WallName` directly. (6) The fallback neuron selection activates when fewer than 64 neurons pass the primary criteria.

ii.
```python
if frames.size == 0:
    continue
# ...
reward_dp_thr = np.inf
ahv_mask = region_idx_all == 3
if np.isfinite(reward_dp[ahv_mask]).any():
    reward_dp_thr = float(np.nanpercentile(reward_dp[ahv_mask], 95))
# ...
if selected.sum() < MIN_NEURONS_FALLBACK:
    # fallback selection...
```

iii. CONVERSION_NOTES.md documents edge-case handling in Steps 10 and 12.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the neural data files (`utils.load_spk()`), which reads files totaling ~405 GB from disk. The full conversion takes ~501 seconds for all 89 sessions. The per-session d-prime computation for neuron selection is also significant.

ii.
```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
```

iii. CONVERSION_NOTES.md estimates ~2.10 s/GB for the full conversion, with the spike file loading dominating runtime.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop in `build_trial_arrays()` iterates over all trials sequentially, building per-trial arrays one at a time. This could potentially be vectorized using the frame-level arrays and split operations. The reward d-prime computation in `compute_neuron_selection()` also loops over trials for `trial_means`.

ii.
```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    # ...

for trial_idx in np.flatnonzero(valid_reward_trials):
    frames = np.arange(start_fr[trial_idx], gray_fr_trial[trial_idx], dtype=int)
    # ...
    trial_means[:, trial_idx] = np.nanmean(spk[:, frames], axis=1)
```

iii. The AI noted these loops but did not fully vectorize them. CONVERSION_NOTES.md lists some speedups implemented but acknowledges remaining inefficiencies.

## 12-c. What processing does the code repeat multiple times?

i. (1) Behavior files are loaded multiple times: once in `collect_stimulus_values()`, once in `compute_time_bin_ms()`, and once per session in `process_session()`. (2) The session catalog is rebuilt or iterated multiple times for metadata collection. (3) For `--sample` mode, `sample_candidate_score()` loads behavior for all sessions to rank them.

ii.
```python
# In collect_stimulus_values:
beh = load_behavior(cand.exp_type, cand.beh_key)
# In compute_time_bin_ms:
ft = np.asarray(load_behavior(cand.exp_type, cand.beh_key)["ft"], dtype=float)
# In process_session:
beh = load_behavior(candidate.exp_type, candidate.beh_key)
```

iii. CONVERSION_NOTES.md acknowledges: "Full-mode metadata currently recomputes frame-interval medians by reloading behavior files" and "Stimulus vocabulary collection reloads behavior dictionaries once more after catalog creation."

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The neuron selection process computes reward-prediction d-prime values and stimulus selectivity for neuron curation, but the downstream decoder only uses the selected neurons — the d-prime values themselves are not used. (2) Cue position histograms are computed for processing plots but not used in the final data. (3) The `processing_info` dict per session stores various statistics that are not saved to the output pickle. (4) Trial length summaries are computed but not included in the final output.

ii.
```python
proc_info = {
    **selection_info,
    **trial_summary,
    "cue_positions_dm": cue_positions,
    "example_trial": example_trial,
}
```

iii. These are primarily diagnostic outputs used during development and verification.
